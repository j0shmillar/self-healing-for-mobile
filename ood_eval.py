import argparse
import numpy as np
import torch
import copy
import io
import random
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models, transforms
from torch.utils.data import DataLoader

from tqdm import tqdm

import agent
from code_guard import validate_and_build_model_patch

# 1st
# update to add ssl but only the last layer
# update to add feedback loop?
# 2nd 
# reformat for tidiness (main core, agent/etc in its own file)
# 3rd
# add heuristic, oracle, TENT, etc baselines in their own files
# 4th
# add another modality (e.g. audio)
# - add new model
# - keep core general (for any modality)

# TODO
# add a feedback loop e.g. fix applied, eval, eval results fed back to LLM, new fix suggested
# turn into a useable library
# eval on real HW with multiple downstream nodes and models 

# ---------- helpers ----------

def recompute_normalization_from_dataset(model_wrapper, dataset, max_samples=2000, device="cpu"):
    """
    Estimate per-channel mean/std from dataset (assuming images in [0,1]).
    Updates model_wrapper.preprocess = Resize(224) + Normalize(new_mean, new_std).
    """
    n = min(max_samples, len(dataset))
    if n == 0:
        print("[recompute_normalization] WARNING: empty dataset, skipping.")
        return

    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    num_pixels = 0
    sum_c = torch.zeros(3, device=device)
    sum_sq_c = torch.zeros(3, device=device)

    for i, (imgs, _) in enumerate(loader):
        if i * loader.batch_size >= n:
            break
        # imgs = imgs.to(device)  # [B,C,H,W], assumed in [0,1]
        b, c, h, w = imgs.shape
        n_pix = b * h * w
        num_pixels += n_pix
        sum_c += imgs.sum(dim=[0, 2, 3])
        sum_sq_c += (imgs ** 2).sum(dim=[0, 2, 3])

        if num_pixels >= n * h * w:
            break

    if num_pixels == 0:
        print("[recompute_normalization] WARNING: no pixels accumulated, skipping.")
        return

    mean = (sum_c / num_pixels)
    var = (sum_sq_c / num_pixels) - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=1e-6))

    mean_list = mean.tolist()
    std_list = std.tolist()

    print(f"[recompute_normalization] new mean={mean_list}, std={std_list}")

    # Rebuild preprocess: keep Resize, update Normalize
    new_transforms = []
    have_resize = False
    if isinstance(model_wrapper.preprocess, T.Compose):
        for t in model_wrapper.preprocess.transforms:
            if isinstance(t, T.Resize):
                new_transforms.append(T.Resize((224, 224)))
                have_resize = True
            elif isinstance(t, T.Normalize):
                # replace with new stats
                continue
            else:
                new_transforms.append(t)
    else:
        # best effort: wrap old preprocess
        new_transforms.append(T.Resize((224, 224)))
        new_transforms.append(model_wrapper.preprocess)

    if not have_resize:
        new_transforms.insert(0, T.Resize((224, 224)))

    new_transforms.append(T.Normalize(mean=mean_list, std=std_list))
    model_wrapper.preprocess = T.Compose(new_transforms)


def update_resize_in_preprocess(model_wrapper, size: int):
    """
    Change the Resize in preprocess to the given (size, size).
    """
    print(f"[change_input_resolution] Setting resolution to {size}x{size}")
    if isinstance(model_wrapper.preprocess, T.Compose):
        new_ts = []
        replaced = False
        for t in model_wrapper.preprocess.transforms:
            if isinstance(t, T.Resize):
                new_ts.append(T.Resize((size, size)))
                replaced = True
            else:
                new_ts.append(t)
        if not replaced:
            new_ts.insert(0, T.Resize((size, size)))
        model_wrapper.preprocess = T.Compose(new_ts)
    else:
        model_wrapper.preprocess = T.Compose([
            T.Resize((size, size)),
            model_wrapper.preprocess,
        ])

def apply_model_arch_patch(monitor, backbone_obj, device="cpu"):
    """
    Replace monitor.model.model with backbone_obj (an nn.Module from a code patch),
    and re-register the avgpool hook + softmax.
    Assumes backbone_obj already has a valid classifier head.
    """
    print("[CODE PATCH] Applying model_arch patch to DriftMonitor.model.model")
    model_wrapper = monitor.model

    # Replace model
    # backbone_obj.to(device)
    model_wrapper.model = backbone_obj

    # Re-register avgpool hook if present
    if hasattr(model_wrapper.model, "avgpool"):
        def hook_fn(m, i, o):
            model_wrapper.latest_features = o.detach().view(-1)
        model_wrapper.model.avgpool.register_forward_hook(hook_fn)
    else:
        print("[CODE PATCH] WARNING: patched model has no .avgpool; "
              "latest_features hook not set, Mahalanobis may break.")

    # Softmax over last dim
    model_wrapper.softmax = nn.Softmax(dim=1)


def apply_preprocess_patch(monitor, build_preprocess_callable):
    """
    Replace monitor.model.preprocess with the callable returned by
    build_preprocess_callable().
    """
    print("[CODE PATCH] Applying preprocess patch")
    model_wrapper = monitor.model
    try:
        new_tf = build_preprocess_callable()
    except Exception as e:
        print(f"[CODE PATCH] ERROR: build_preprocess() failed: {e}")
        return
    model_wrapper.preprocess = new_tf


def forward_logits(model_wrapper, imgs, device="cpu"):
    """
    model_wrapper: your Model (with .model, .preprocess)
    imgs: torch.Tensor [B,C,H,W] in [0,1], or list of such tensors
    """
    if isinstance(imgs, list):
        imgs = torch.stack(imgs, dim=0)
    # imgs = imgs.to(device)

    with torch.no_grad():
        x = model_wrapper.preprocess(imgs)
        logits = model_wrapper.model(x)
    return logits


# ----------------- Your Corruptions ---------------------- #
def corrupt_cutout(img, severity=3, max_frac=0.5):
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1,3) and arr.shape[0] != arr.shape[2]:
        arr = np.transpose(arr, (1,2,0))
        chw = True
    else:
        chw = False
    h, w = arr.shape[:2]
    frac = np.clip(severity / 5.0, 0.05, max_frac)  # severity 1–5 → 5%–max_frac
    size = int(min(h, w) * frac)
    size = max(1, size)
    top = np.random.randint(0, h - size + 1)
    left = np.random.randint(0, w - size + 1)
    arr[top:top+size, left:left+size, ...] = 0
    if chw:
        arr = np.transpose(arr, (2,0,1))
    return arr

def corrupt_blur(img, kernel_size=3):
    assert kernel_size % 2 == 1
    pad = kernel_size // 2
    h,w,c = img.shape
    padded = np.pad(img, ((pad,pad),(pad,pad),(0,0)), mode='reflect')
    out = np.zeros_like(img)
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            out += padded[dy:dy+h, dx:dx+w, :]
    return out / (kernel_size**2)

def corrupt_gaussian_noise(img, std_range=(0.1, 0.4)):
    img = np.asarray(img).astype(np.float32)
    std = np.random.uniform(*std_range)
    noise = np.random.normal(0.0, std, img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0)

def corrupt_jpeg(img, quality_range=(10, 50)):
    """Simulate JPEG compression artifacts."""
    from PIL import Image
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    h, w, c = img.shape
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    q = int(np.random.uniform(*quality_range))
    pil.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    out = Image.open(buf).convert("RGB").resize((w, h), Image.BILINEAR)
    arr = np.asarray(out).astype(np.float32) / 255.0
    return arr

def corrupt_pixelate(img, downsample_range=(4, 8)):
    """Downsample then upsample with nearest to create pixelation."""
    from PIL import Image
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    h, w, c = img.shape
    pil = Image.fromarray(img)
    factor = int(np.random.randint(*downsample_range))
    small = pil.resize((max(1, w // factor), max(1, h // factor)), Image.BILINEAR)
    out = small.resize((w, h), Image.NEAREST)
    arr = np.asarray(out).astype(np.float32) / 255.0
    return arr

def corrupt_motion_blur(img, kernel_size_range=(5, 15)):
    """Simple horizontal motion blur."""
    img = np.asarray(img).astype(np.float32)
    k = int(np.random.randint(*kernel_size_range))
    if k % 2 == 0:
        k += 1
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    pad = k // 2
    padded = np.pad(img, ((0, 0), (pad, pad), (0, 0)), mode="reflect")
    out = np.zeros_like(img)
    for i in range(img.shape[1]):  # width
        out[:, i, :] = np.tensordot(
            padded[:, i : i + k, :],
            kernel[k // 2, :],
            axes=([1], [0]),
        )
    return np.clip(out, 0.0, 1.0)

def corrupt_strong_color_jitter(img,
                                brightness_range=(-0.4, 0.4),
                                contrast_range=(0.5, 1.5),
                                saturation_range=(0.5, 1.5)):
    img = np.asarray(img).astype(np.float32)
    # brightness
    delta = np.random.uniform(*brightness_range)
    img = img + delta
    # contrast
    c = np.random.uniform(*contrast_range)
    img = (img - 0.5) * c + 0.5
    # saturation (convert to gray)
    s = np.random.uniform(*saturation_range)
    gray = np.mean(img, axis=2, keepdims=True)
    img = gray + (img - gray) * s
    return np.clip(img, 0.0, 1.0)

def corrupt_grayscale(img):
    img = np.asarray(img).astype(np.float32)
    gray = np.mean(img, axis=2, keepdims=True)
    return np.repeat(gray, 3, axis=2)

def corrupt_translate(img, max_shift_frac=0.25):
    """Translate image by random dx,dy and fill with zeros."""
    img = np.asarray(img).astype(np.float32)
    h, w, c = img.shape
    max_dx = int(w * max_shift_frac)
    max_dy = int(h * max_shift_frac)
    dx = np.random.randint(-max_dx, max_dx + 1)
    dy = np.random.randint(-max_dy, max_dy + 1)
    out = np.zeros_like(img)
    y1_src = max(0, -dy)
    y1_dst = max(0, dy)
    x1_src = max(0, -dx)
    x1_dst = max(0, dx)
    y2_src = min(h, h - dy)
    y2_dst = min(h, h + (h - dy))
    x2_src = min(w, w - dx)
    x2_dst = min(w, w + (w - dx))
    out[y1_dst:y1_dst+(y2_src-y1_src), x1_dst:x1_dst+(x2_src-x1_src), :] = \
        img[y1_src:y2_src, x1_src:x2_src, :]
    return out

def corrupt_big_cutout(img, max_frac=0.7):
    """Large occlusion square."""
    arr = np.asarray(img).copy()
    h, w = arr.shape[:2]
    frac = np.random.uniform(0.3, max_frac)
    size = int(min(h, w) * frac)
    size = max(1, size)
    top = np.random.randint(0, h - size + 1)
    left = np.random.randint(0, w - size + 1)
    arr[top:top+size, left:left+size, :] = 0
    return arr

def corrupt_channel_swap(img):
    img = np.asarray(img).astype(np.float32)
    perm = np.random.permutation(3)
    return img[:, :, perm]

def random_corruption():
    corruptions = [
        ("gaussian_noise", corrupt_gaussian_noise),
        ("blur", corrupt_blur),            
        ("motion_blur", corrupt_motion_blur),
        ("jpeg", corrupt_jpeg),
        ("pixelate", corrupt_pixelate),
        # ("color_jitter", corrupt_strong_color_jitter),
        ("grayscale", corrupt_grayscale),
        # ("translate", corrupt_translate),
        # ("cutout", corrupt_cutout),
        # ("big_cutout", corrupt_big_cutout),
        ("channel_swap", corrupt_channel_swap),
    ]
    return corruptions[np.random.randint(len(corruptions))]

# ==============================================================================
# Drift Monitor
# ==============================================================================

class Model:
    def __init__(self, num_classes=100, ckpt_path="miniimagenet_resnet18.pth"):
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(512, num_classes)
        self.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

        def hook_fn(m, i, o):
            self.latest_features = o.detach().view(-1)
        self.model.avgpool.register_forward_hook(hook_fn)

        self.model.eval()
        self.softmax = nn.Softmax(dim=1)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


    def infer(self, img):
        """
        img: torch.Tensor [C, H, W] in [0,1]
        """
        if not torch.is_tensor(img):
            img = torch.as_tensor(img)
        if img.ndim == 3:
            x = img.unsqueeze(0)
        elif img.ndim == 4:
            x = img
        else:
            raise ValueError(f"Unexpected img shape: {img.shape}")

        x = self.preprocess(x)

        with torch.no_grad():
            logits = self.model(x)
            probs = self.softmax(logits)

        feats = self.latest_features.numpy()  # (512,)
        return feats, probs

class DriftMonitor:
    def __init__(self, feat_dim=512, thresh_std=3.0):
        """
        Mahalanobis-based drift detector.
        - feat_dim: dimension of feature vector (512 for ResNet18 avgpool).
        - thresh_std: how many stds above reference mean distance to flag drift.
        """
        self.model = Model()
        self.feat_dim = feat_dim
        self.thresh_std = thresh_std

        self.mu = None
        self.inv_cov = None
        self.ref_mean_dist = None
        self.ref_std_dist = None

        print("Using fixed Mahalanobis drift detector")

    def ingest_reference(self, img_list):
        """
        Compute reference mean, covariance and distance stats
        from a list of clean images (torch tensors [C,H,W]).
        """
        feats = []
        for img in img_list:
            f, _ = self.model.infer(img)
            feats.append(f)
        feats = np.stack(feats)  # [N, 512]

        self.mu = feats.mean(axis=0)

        # Covariance + regularization
        cov = np.cov(feats, rowvar=False)  # [512,512]
        cov += 1e-5 * np.eye(self.feat_dim, dtype=np.float32)
        self.inv_cov = np.linalg.inv(cov)

        # Mahalanobis distances on reference set
        dists = []
        for f in feats:
            delta = f - self.mu
            d = float(np.sqrt(delta @ self.inv_cov @ delta))
            dists.append(d)
        dists = np.array(dists)
        self.ref_mean_dist = float(dists.mean())
        self.ref_std_dist = float(dists.std() + 1e-6)

        print(f"Ref Mahalanobis mean={self.ref_mean_dist:.3f}, "
              f"std={self.ref_std_dist:.3f}")

    def compute_score(self, img):
        """
        Returns (mahalanobis_distance, feats, probs)
        """
        feats, probs = self.model.infer(img)
        delta = feats - self.mu
        score = float(np.sqrt(delta @ self.inv_cov @ delta))
        return score, feats, probs

    def is_drift(self, score, thresh=None):
        """
        Use internal threshold if thresh is None:
          mu + thresh_std * sigma.
        """
        if self.ref_mean_dist is None:
            raise RuntimeError("Call ingest_reference() first")

        if thresh is None:
            thresh = self.ref_mean_dist + self.thresh_std * self.ref_std_dist
        return score > thresh

# ==============================================================================
# Evaluation Code
# ==============================================================================

def evaluate_stream(
    monitor,
    orchestrator,
    clean_imgs,
    train_dset,
    val_dset,
    total_frames=1000,
    drift_start=100,
    drift_end=300,
    N_CONSEC_FOR_DETECT=5,
    MIN_DRIFT_BUFFER=128,
    device="cpu",
):
    """
    Simulate a streaming evaluation with a single corruption episode.

    monitor       : DriftMonitor instance (Mahalanobis)
    orchestrator  : Orchestrator agent
    clean_imgs    : list[Tensor CxHxW] (clean training images)
    train_dset    : labeled MiniImageNet train dataset (for adaptation / ref)
    val_dset      : labeled MiniImageNet val dataset (for eval)
    total_frames  : length of stream
    drift_start   : index where corruption begins
    drift_end     : index where corruption ends (inclusive)
    N_CONSEC_FOR_DETECT : consecutive drift flags required for "confirmed drift"
    MIN_DRIFT_BUFFER    : min drift samples needed before calling LLM/adaptation
    device        : "cpu" or "cuda"
    """
    assert drift_start < drift_end < total_frames

    scores = []
    telemetry = []
    detected = False
    detection_index = None

    # Pick 1 random corruption function to use for the whole episode
    cname, corrupt_fn = random_corruption()
    print(f"[Streaming eval] Corruption selected: {cname}")

    n_flags = 0
    false_alarms = 0
    drift_buffer = []  # drifted images for adaptation
    scores_clean, scores_corr = [], []

    for i in tqdm(range(total_frames)):
        img_t = clean_imgs[i]  # torch [C,H,W], 0–1

        # convert to HWC numpy for corruption
        img_np = img_t.permute(1, 2, 0).numpy()  # [H,W,C] float

        # Apply corruption only inside drift window
        if drift_start <= i <= drift_end:
            img_np = corrupt_fn(img_np)  # all corrupt_* expect HWC

        # back to torch CHW
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float()

        # keep a copy of drifted images for adaptation
        if drift_start <= i <= drift_end:
            drift_buffer.append(img_t.detach().clone())

        # compute drift score
        score, feats, probs = monitor.compute_score(img_t)
        drift_flag = monitor.is_drift(score)
        print(drift_flag)

        scores.append(score)

        if i < drift_start:
            scores_clean.append(score)
        elif drift_start <= i <= drift_end:
            scores_corr.append(score)

        # simple false alarm counting (step-level)
        if drift_flag and not (drift_start <= i <= drift_end):
            false_alarms += 1

        # hysteresis for detection
        if drift_flag:
            n_flags += 1
        else:
            n_flags = 0

        # --- make telemetry JSON-serializable ---
        if isinstance(feats, np.ndarray):
            feats_serial = feats.tolist()
        else:
            try:
                feats_serial = np.array(feats).tolist()
            except Exception:
                feats_serial = None

        try:
            import torch as _torch
            is_tensor = isinstance(probs, _torch.Tensor)
        except Exception:
            is_tensor = False

        if is_tensor:
            probs_arr = probs.numpy()
            probs_serial = probs_arr.tolist()
        elif isinstance(probs, np.ndarray):
            probs_serial = probs.tolist()
        else:
            try:
                probs_serial = list(probs)
            except Exception:
                probs_serial = None

        telemetry.append({
            "t": int(i),
            "mahalanobis_score": float(score),
            # "feats": feats_serial,
            # "probs": probs_serial,
        })

        # ---- confirmed drift + adaptation trigger ----
        if (not detected
            and n_flags >= N_CONSEC_FOR_DETECT
            and len(drift_buffer) >= MIN_DRIFT_BUFFER):
            detected = True
            detection_index = i
            print(f"[DETECT] Confirmed drift at index {i} "
                  f"(n_flags={n_flags}, drift_buffer={len(drift_buffer)})")

            # Take last MIN_DRIFT_BUFFER samples as representative drift window
            recent_drift = list(drift_buffer[-MIN_DRIFT_BUFFER:])

            # Call LLM agent
            result = orchestrator.plan(
                telemetry=telemetry,
                shifted_examples=recent_drift,
                ref_examples=clean_imgs[:5],
                max_images_per_set=4,
            )

            # Apply actions safely (may adapt monitor.model)
            monitor, fix_accepted = apply_actions(
                monitor=monitor,
                actions_dict=result,
                clean_train_dataset=train_dset,
                clean_val_dataset=val_dset,
                drift_buffer=recent_drift,
                corruption_fn_for_eval=corrupt_fn,
                device=device,
            )

            # Reset drift tracking for post-fix period
            n_flags = 0
            drift_buffer.clear()
            # You could also reset telemetry if you want a fresh log
            # telemetry = []

        # end of loop over i

    print("clean mean/std: ", np.mean(scores_clean), np.std(scores_clean))
    print("corr  mean/std: ", np.mean(scores_corr), np.std(scores_corr))

    was_detected = detected
    detection_delay = (detection_index - drift_start) if detected else None

    return {
        "corruption": cname,
        "was_detected": was_detected,
        "detection_index": detection_index,
        "detection_delay": detection_delay,
        "false_alarms": false_alarms,
    }

def tent_like_adapt(
    monitor,
    drift_imgs,
    steps=50,
    batch_size=32,
    lr=5e-5,
    device="cpu",
):
    """
    TENT-like adaptation: update only normalization affine params to minimize entropy.
    More conservative: small lr, fewer steps, random subset of drift_imgs.
    """
    import random
    model_wrapper = monitor.model
    model = model_wrapper.model
    model.train()

    # Identify norm params
    norm_params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            if m.weight is not None:
                norm_params.append(m.weight)
            if m.bias is not None:
                norm_params.append(m.bias)

    if not norm_params:
        print("[TENT] No norm params found; skipping.")
        return

    optimizer = torch.optim.Adam(norm_params, lr=lr)
    softmax = nn.Softmax(dim=1)

    # Limit drift pool
    if len(drift_imgs) > 512:
        drift_imgs = random.sample(drift_imgs, 512)

    for step in range(steps):
        batch = random.sample(drift_imgs, min(batch_size, len(drift_imgs)))
        xb = torch.stack(batch)

        optimizer.zero_grad()
        x = model_wrapper.preprocess(xb)
        logits = model(x)
        probs = softmax(logits)

        ent = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
        loss = ent

        loss.backward()
        optimizer.step()

        if (step + 1) % 10 == 0:
            print(f"[TENT] step {step+1}/{steps}, loss={loss.item():.4f}")

    model.eval()
    # weights are mutated in-place via model_wrapper.model, so nothing to reassign


def strong_augment():
    # can tweak as needed
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
    ])

def full_ssl_adapt(
    monitor,
    drift_imgs,
    clean_imgs,
    steps=200,
    batch_size=32,
    lr=1e-4,
    lambda_consistency=1.0,
    lambda_distill=1.0,
    device="cpu",
):
    """
    Full-parameter adaptation using self-supervised-ish objectives:
      - Entropy minimisation on drift images
      - Consistency loss between two augmentations of drift images
      - Distillation on clean images toward original (pre-fix) model

    No GT labels used. Suitable for 'finetune_on_recent' / 'finetune_full'.
    """
    model_wrapper = monitor.model
    student = model_wrapper.model
    student.train()

    # Snapshot teacher (original model) for distillation
    teacher = copy.deepcopy(student)
    teacher.eval()

    softmax = nn.Softmax(dim=1)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)

    aug = strong_augment()

    # Reduce pools for sanity
    if len(drift_imgs) > 2048:
        drift_imgs = random.sample(drift_imgs, 2048)
    if len(clean_imgs) > 2048:
        clean_imgs = random.sample(clean_imgs, 2048)

    for step in tqdm(range(steps)):
        # ---- drift batch (unlabeled) ----
        xb_drift = random.sample(drift_imgs, min(batch_size, len(drift_imgs)))
        xb_drift = torch.stack(xb_drift)

        # two augmented views
        xa1 = torch.stack([aug(img) for img in xb_drift])
        xa2 = torch.stack([aug(img) for img in xb_drift])

        x1 = model_wrapper.preprocess(xa1)
        x2 = model_wrapper.preprocess(xa2)

        logits1 = student(x1)
        logits2 = student(x2)

        p1 = softmax(logits1)
        p2 = softmax(logits2)

        # entropy on one view
        ent = -torch.sum(p1 * torch.log(p1 + 1e-8), dim=1).mean()

        # symmetric KL for consistency
        kl1 = torch.sum(p1 * (torch.log(p1 + 1e-8) - torch.log(p2 + 1e-8)), dim=1)
        kl2 = torch.sum(p2 * (torch.log(p2 + 1e-8) - torch.log(p1 + 1e-8)), dim=1)
        cons = 0.5 * (kl1 + kl2).mean()

        # ---- clean batch for distillation ----
        xb_clean = random.sample(clean_imgs, min(batch_size, len(clean_imgs)))
        xb_clean = torch.stack(xb_clean)

        with torch.no_grad():
            x_clean_t = model_wrapper.preprocess(xb_clean)
            logits_teacher = teacher(x_clean_t)
            p_teacher = softmax(logits_teacher)

        x_clean_s = model_wrapper.preprocess(xb_clean)
        logits_student = student(x_clean_s)
        p_student = softmax(logits_student)

        # distillation: KL(student || teacher)
        distill = torch.sum(
            p_teacher * (torch.log(p_teacher + 1e-8) - torch.log(p_student + 1e-8)),
            dim=1,
        ).mean()

        loss = ent + lambda_consistency * cons + lambda_distill * distill

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 20 == 0:
            print(f"[FULL-SSL] step {step+1}/{steps}, "
                  f"ent={ent.item():.4f} cons={cons.item():.4f} dist={distill.item():.4f}")

    student.eval()
    model_wrapper.model = student
    monitor.model = model_wrapper


def eval_model_on_dataset(
    monitor,
    dataset,
    n_samples=500,
    corruption_fn=None,
    device="cpu",
):
    """
    Evaluate top-1 accuracy on first n_samples of a dataset.
    dataset: something like MiniImageNet where __getitem__ returns (img, label)
    corruption_fn: if not None, applied in HWC space before inference
    """
    model_wrapper = monitor.model
    model_wrapper.model
    model_wrapper.model.eval()
    softmax = nn.Softmax(dim=1)

    correct = 0
    total = 0

    for idx in range(min(n_samples, len(dataset))):
        img, label = dataset[idx]  # img: [C,H,W] tensor
        if corruption_fn is not None:
            # convert to HWC numpy -> corrupt -> back
            img_np = img.permute(1, 2, 0).numpy()
            img_np = corruption_fn(img_np)
            img = torch.from_numpy(img_np).permute(2, 0, 1).float()

        x = img.unsqueeze(0)
        x = model_wrapper.preprocess(x)
        with torch.no_grad():
            logits = model_wrapper.model(x)
            probs = softmax(logits)
            pred = probs.argmax(dim=1).item()

        if int(pred) == int(label):
            correct += 1
        total += 1

    acc = correct / max(total, 1)
    return acc


def apply_actions(
    monitor,
    actions_dict,
    clean_train_dataset,
    clean_val_dataset,
    drift_buffer,
    corruption_fn_for_eval=None,
    device="cpu",
):
    """
    Apply LLM-suggested actions in a safe, iterative way.

    For each action in order:
      - Clone current monitor => candidate_monitor
      - Apply JUST that action to candidate_monitor
      - Evaluate clean & shifted on val set
      - Accept if:
          - clean doesn't drop too much
          - shifted improves enough
        -> then rebaseline Mahalanobis stats on clean data
      - Else discard candidate and try next action.

    Returns:
      monitor (possibly updated), any_fix_accepted (bool)
    """

    corruption_type = actions_dict.get("corruption_type")
    builtin_actions = actions_dict.get("builtin_actions", [])
    code_patches = actions_dict.get("code_patches", [])

    print("LLM diagnosed corruption:", corruption_type)
    print("Builtin actions:", builtin_actions)
    print("Code patches:", [p["target"] for p in code_patches])

    # Baseline eval for the current monitor (starting point for the sequence)
    base_clean_acc = eval_model_on_dataset(
        monitor, clean_val_dataset,
        n_samples=300, device=device
    )
    base_shift_acc = eval_model_on_dataset(
        monitor, clean_val_dataset,
        n_samples=300,
        corruption_fn=corruption_fn_for_eval,
        device=device,
    )
    print(f"[BASELINE] clean acc={base_clean_acc:.3f}, "
          f"shifted acc={base_shift_acc:.3f}")

    CLEAN_TOL = 0.02          # allow up to 2% drop on clean
    MIN_SHIFT_IMPROVE = 0.03  # require at least 3% gain on shifted

    any_fix_accepted = False
    current_monitor = monitor
    current_clean_acc = base_clean_acc
    current_shift_acc = base_shift_acc

    # -------------- helper: gate + rebaseline --------------

    def try_candidate(candidate_monitor, action_name: str):
        nonlocal current_monitor, current_clean_acc, current_shift_acc, any_fix_accepted

        new_clean_acc = eval_model_on_dataset(
            candidate_monitor, clean_val_dataset,
            n_samples=300, device=device
        )
        new_shift_acc = eval_model_on_dataset(
            candidate_monitor, clean_val_dataset,
            n_samples=300,
            corruption_fn=corruption_fn_for_eval,
            device=device,
        )
        print(f"[POST-FIX:{action_name}] clean acc={new_clean_acc:.3f}, "
              f"shifted acc={new_shift_acc:.3f}")

        # if (new_clean_acc + CLEAN_TOL >= current_clean_acc) and \
        #    (new_shift_acc >= current_shift_acc + MIN_SHIFT_IMPROVE):
        if (new_clean_acc + CLEAN_TOL >= current_clean_acc) and \
           (new_shift_acc >= current_shift_acc):
            print(f"[GATE] Action '{action_name}' accepted.")
            # Re-baseline Mahalanobis stats on clean train data
            max_ref = min(len(clean_train_dataset), 1000)
            clean_ref_imgs = [clean_train_dataset[i][0] for i in range(max_ref)]
            candidate_monitor.ingest_reference(clean_ref_imgs)

            current_monitor = candidate_monitor
            current_clean_acc = new_clean_acc
            current_shift_acc = new_shift_acc
            any_fix_accepted = True
            return True
        else:
            print(f"[GATE] Action '{action_name}' rejected.")
            return False

    # -------------- Tier-1 actions, one by one --------------

    for act in builtin_actions:
        name = act["name"]
        params = act["params"]

        # We always start from the *current* accepted monitor
        candidate = copy.deepcopy(current_monitor)
        changed = False

        # 1) Fine-tuning / adaptation
        if name.startswith("finetune"):
            print(f"Applying adaptation action: {name} with {params}")
            drift_imgs = list(drift_buffer)
            if not drift_imgs:
                print("[ACTION] No drift_imgs available; skipping adaptation.")
                continue

            # map LLM-friendly names to behaviors
            if name == "finetune_light":
                tent_like_adapt(
                    candidate,
                    drift_imgs=drift_imgs,
                    steps=int(params.get("epochs", 5) * 10),  # e.g. 5 epochs → 50 steps
                    batch_size=32,
                    lr=float(params.get("lr", params.get("learning_rate", 5e-5))),
                    device=device,
                )
            else:  # 'finetune_on_recent', 'finetune_aggressive' etc.
                max_clean = min(len(clean_train_dataset), 2000)
                clean_imgs = [clean_train_dataset[i][0]
                              for i in range(max_clean)]

                full_ssl_adapt(
                    candidate,
                    drift_imgs=drift_imgs,
                    clean_imgs=clean_imgs,
                    steps=int(params.get("epochs", 10) * 20),
                    batch_size=32,
                    lr=float(params.get("lr", params.get("learning_rate", 1e-4))),
                    lambda_consistency=1.0,
                    lambda_distill=1.0,
                    device=device,
                )
            changed = True

        # 2) Confidence threshold tweaks
        elif name == "increase_confidence_threshold":
            delta = float(params.get("delta", 0.05))
            if not hasattr(candidate, "confidence_threshold"):
                candidate.confidence_threshold = 0.0
            candidate.confidence_threshold = max(
                0.0, min(1.0, candidate.confidence_threshold + delta)
            )
            print(f"[ACTION] confidence_threshold -> "
                  f"{candidate.confidence_threshold:.3f}")
            changed = True

        elif name == "decrease_confidence_threshold":
            delta = float(params.get("delta", 0.05))
            if not hasattr(candidate, "confidence_threshold"):
                candidate.confidence_threshold = 0.0
            candidate.confidence_threshold = max(
                0.0, min(1.0, candidate.confidence_threshold - delta)
            )
            print(f"[ACTION] confidence_threshold -> "
                  f"{candidate.confidence_threshold:.3f}")
            changed = True

        # 3) Reset model
        elif name == "reset_model":
            print("[ACTION] Resetting candidate monitor to fresh instance.")
            candidate = candidate.__class__()   # new DriftMonitor()
            changed = True

        # 4) Recompute normalization
        elif name == "recompute_normalization":
            print("[ACTION] Recomputing normalization from clean_train_dataset")
            recompute_normalization_from_dataset(
                model_wrapper=candidate.model,
                dataset=clean_train_dataset,
                max_samples=2000,
                device=device,
            )
            changed = True

        # 5) Change input resolution
        elif name == "change_input_resolution":
            size = int(params.get("size", 224))
            update_resize_in_preprocess(candidate.model, size)
            changed = True

        else:
            print("[ACTION] Unhandled builtin action:", name)

        if not changed:
            continue

        # Gate this single action
        accepted = try_candidate(candidate, name)
        # If rejected, we just move on to the next suggested action.
        # If accepted, current_monitor was updated.

    # -------------- Tier-2 code patches (also per-patch gating) --------------

    for patch in code_patches:
        ok, obj, msg = validate_and_build_model_patch(
            code=patch["code"],
            target=patch["target"],
            constraints=patch.get("constraints", {}),
        )
        print(f"[CODE PATCH] target={patch['target']} validation: {msg}")
        if not ok:
            continue

        candidate = copy.deepcopy(current_monitor)
        changed = False

        if patch["target"] == "model_arch":
            if isinstance(obj, nn.Module):
                apply_model_arch_patch(candidate, obj, device=device)
                changed = True
            else:
                print("[CODE PATCH] WARNING: model_arch object is not nn.Module; skipping.")

        elif patch["target"] == "preprocess":
            if callable(obj):
                apply_preprocess_patch(candidate, obj)
                changed = True
            else:
                print("[CODE PATCH] WARNING: preprocess object is not callable; skipping.")

        if not changed:
            continue

        accepted = try_candidate(candidate, f"code_patch:{patch['target']}")
        # if rejected, current_monitor unchanged; continue to next patch

    return current_monitor, any_fix_accepted

# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nref", type=int, default=1000)
    args = parser.parse_args()

    import torchvision.transforms as T
    tf = transforms.Compose([T.ToTensor()])

    from MLclf import MLclf
    tset, vset, dset = MLclf.miniimagenet_clf_dataset(
        ratio_train=0.6,
        ratio_val=0.2,
        seed_value=None,
        shuffle=True,
        transform=tf,
        save_clf_data=True,
    )

    clean_imgs = [dset[i][0] for i in range(args.nref)]

    monitor = DriftMonitor()
    orchestrator = agent.Orchestrator()

    # initial reference stats from clean images
    monitor.ingest_reference(clean_imgs)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = evaluate_stream(
        monitor=monitor,
        orchestrator=orchestrator,
        clean_imgs=clean_imgs,
        train_dset=tset,
        val_dset=vset,
        total_frames=1000,
        drift_start=100,
        drift_end=300,
        # N_CONSEC_FOR_DETECT=5,
        N_CONSEC_FOR_DETECT=1,
        MIN_DRIFT_BUFFER=64,
        device=device,
    )

    print(results)

if __name__ == "__main__":
    main()
