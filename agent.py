# agent.py (updated)
import inspect
import random
import base64
import copy
import json
import os
import io
import re

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from torchvision import models
from ood_eval import Model  # your local Model

from PIL import Image
import numpy as np

# Works for channel_swap, cutout

# TODO
# - neural network applications on edge devices...think

import ast
from typing import Any, Dict, List

# --------------------------------------------------------------------
# Tier-1 Actions: whitelisted, parameter-guarded "buttons"
# --------------------------------------------------------------------

SUPPORTED_ACTIONS: Dict[str, Dict[str, Any]] = {
    "recompute_normalization": {
        "params_schema": {}
    },
    "finetune_on_recent": {
        "params_schema": {
            "epochs": (1, 20),             # int range
            "lr": (1e-5, 5e-3),            # float range
            "max_steps": (10, 5000),
        }
    },
    "finetune_norm_only": {
        "params_schema": {
            "epochs": (1, 5),
            "lr": (1e-5, 5e-3),
        }
    },
    "finetune_full": {
        "params_schema": {
            "epochs": (5, 50),
            "lr": (1e-5, 1e-2),
        }
    },
    "increase_confidence_threshold": {
        "params_schema": {
            "delta": (0.01, 0.5),
        }
    },
    "decrease_confidence_threshold": {
        "params_schema": {
            "delta": (0.01, 0.5),
        }
    },
    "reset_model": {
        "params_schema": {}
    },
    # Add more opts...
    # Tier-2 hook: code patch (not executed here; just passed through and guarded)
    "code_patch": {
        "params_schema": {
            "target": ["model_arch", "preprocess"],
            "language": ["python"],
            "patch_type": ["full_definition"],  # could add "diff" later
            # "code" body is treated specially, not schema-clamped
        }
    },
}

def encode_image(img_or_path, format="JPEG", quality=85):
    """
    Accept either:
      - a filesystem path (str / Path)
      - a numpy array image in HWC, values in [0,1] or [0,255]
    Return: base64 string (no prefix).
    """
    if isinstance(img_or_path, (str, os.PathLike)):
        with open(img_or_path, "rb") as f:
            raw = f.read()
        return base64.b64encode(raw).decode("utf-8")

    # assume numpy array-like
    arr = np.asarray(img_or_path)
    if arr.dtype != np.uint8:
        # assume in [0,1] float
        if np.max(arr) <= 1.0:
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

    # Convert potential CHW -> HWC
    if arr.ndim == 3 and arr.shape[0] in (1,3) and arr.shape[0] != arr.shape[2]:
        # probably CHW
        arr = np.transpose(arr, (1,2,0))

    # PIL expects HxWxC
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format=format, quality=quality)
    b = buf.getvalue()
    return base64.b64encode(b).decode("utf-8")


class Orchestrator:
    """
    Given telemetry + example metadata or images, return actions from:
      - recompute_normalization
      - finetune_on_recent
      - finetune_light
      - finetune_aggressive
      - increase_confidence_threshold
      - decrease_confidence_threshold
      - reset_model
    """
    def __init__(self, vlm_model="gpt-4o"):
        # openai client (None if not configured)
        self.client = None
        if OpenAI is not None and os.getenv("OPENAI_API_KEY"):
            try:
                self.client = OpenAI()
            except Exception:
                self.client = None

        # keep your local Model instance
        self.model_inst = Model()

        # architecture/inference code for context
        self.model_arch = models.resnet18
        self.model_inf = self.model_inst.infer

        # string model id to send to OpenAI
        self.vlm_model = vlm_model
        self._file_cache = {}  # maps sha1 -> file_id

    def _upload_image(self, img_or_path):
        """
        Convert an image to a JPEG data URL for the VLM.

        Accepts:
        - numpy arrays / torch tensors: HWC or CHW
        - strings: http(s)/data URL or local path
        """

        import base64, os, io, hashlib
        import numpy as np
        from PIL import Image
        import torch

        # If it's already a URL, just return it
        if isinstance(img_or_path, (str, os.PathLike)):
            s = str(img_or_path)
            if s.startswith(("http://", "https://", "data:")):
                return s
            # else: treat as path
            with open(s, "rb") as f:
                data = f.read()
        else:
            # tensor / array
            if torch.is_tensor(img_or_path):
                arr = img_or_path.detach().cpu().numpy()
            else:
                arr = np.asarray(img_or_path)

            # Make sure it's HWC
            if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] != arr.shape[2]:
                arr = np.transpose(arr, (1, 2, 0))

            # Fix numeric range
            arr = arr.astype("float32")
            mn, mx = np.min(arr), np.max(arr)
            if mn >= -1.0 and mx <= 1.0:
                # assume normalized [-1,1] -> [0,255]
                arr = (arr * 0.5 + 0.5) * 255.0
            elif mn >= 0.0 and mx <= 1.0:
                # [0,1] -> [0,255]
                arr = arr * 255.0

            arr = np.clip(arr, 0, 255).astype("uint8")

            img = Image.fromarray(arr)

            # Upsample so the VLM can actually see blur
            min_size = 224
            if img.width < min_size or img.height < min_size:
                img = img.resize((min_size, min_size), resample=Image.NEAREST)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            data = buf.getvalue()

        # cache on hash
        h = hashlib.sha1(data).hexdigest()
        if not hasattr(self, "_file_cache"):
            self._file_cache = {}
        if h in self._file_cache:
            return self._file_cache[h]

        b64 = base64.b64encode(data).decode("ascii")
        url = f"data:image/jpeg;base64,{b64}"
        self._file_cache[h] = url
        return url


    # inside class Orchestrator:

    def _clamp_value(self, val, low, high):
        try:
            v = float(val)
        except Exception:
            return None
        return max(low, min(high, v))

    def _validate_builtin_actions(self, raw_actions: List[Dict[str, Any]]):
        """
        Filter & clamp actions to SUPPORTED_ACTIONS.
        Returns a list of safe Tier-1 actions (excluding 'code_patch').
        """
        safe = []
        for a in raw_actions:
            name = a.get("name")
            if name not in SUPPORTED_ACTIONS:
                continue
            if name == "code_patch":
                # handled elsewhere
                continue

            schema = SUPPORTED_ACTIONS[name]["params_schema"]
            raw_params = a.get("params", {}) if isinstance(a.get("params", {}), dict) else {}

            # --- NEW: normalize common synonyms from the LLM ---
            # finetune_*: learning_rate -> lr
            if "learning_rate" in raw_params and "lr" not in raw_params:
                raw_params["lr"] = raw_params["learning_rate"]
            # threshold actions: threshold -> delta
            if "threshold" in raw_params and "delta" not in raw_params:
                raw_params["delta"] = raw_params["threshold"]

            params: Dict[str, Any] = {}
            ok = True

            for key, spec in schema.items():
                if isinstance(spec, (list, tuple)) and all(isinstance(x, str) for x in spec):
                    # categorical
                    v = raw_params.get(key, spec[0])
                    if v not in spec:
                        v = spec[0]
                    params[key] = v
                elif isinstance(spec, tuple) and len(spec) == 2:
                    low, high = spec
                    v = raw_params.get(key, (low + high) / 2.0)
                    v_clamped = self._clamp_value(v, low, high)
                    if v_clamped is None:
                        ok = False
                        break
                    params[key] = int(v_clamped) if isinstance(v, int) else float(v_clamped)
                else:
                    ok = False
                    break

            if ok:
                safe.append({"name": name, "params": params})
        return safe[:3]

    def _extract_code_patches(self, raw_actions: List[Dict[str, Any]]):
        """
        Extract 'code_patch' actions and do basic schema validation.
        Return a list of dicts with target, code, constraints...
        """
        patches = []
        for a in raw_actions:
            if a.get("name") != "code_patch":
                continue
            raw_params = a.get("params", {}) if isinstance(a.get("params", {}), dict) else {}

            target = raw_params.get("target", None)
            language = raw_params.get("language", "python")
            patch_type = raw_params.get("patch_type", "full_definition")
            code = raw_params.get("code", "")

            # Check against SUPPORTED_ACTIONS schema
            schema = SUPPORTED_ACTIONS["code_patch"]["params_schema"]
            if target not in schema["target"]:
                continue
            if language not in schema["language"]:
                continue
            if patch_type not in schema["patch_type"]:
                continue
            if not isinstance(code, str) or len(code.strip()) == 0:
                continue

            # Optional user-specified constraints
            constraints = raw_params.get("constraints", {})
            patches.append({
                "target": target,
                "language": language,
                "patch_type": patch_type,
                "code": code,
                "constraints": constraints,
            })
        return patches
 
    def _make_prompt_blocks(self, telemetry, shifted_examples, ref_examples, max_images_per_set=4):
        """
        Build the `input` contents for responses.create with text blocks and image blocks.
        shifted_examples / ref_examples may be:
          - lists of numpy arrays (images)
          - lists of paths (strings)
          - lists of small metadata dicts
        We safely truncate images to `max_images_per_set`.
        """
        blocks = []

        # system prompt
        try:
            model_code = inspect.getsource(self.model_arch)
        except Exception:
            model_code = "Model definition unavailable."

        try:
            inference_code = inspect.getsource(self.model_inf)
        except Exception:
            inference_code = "Model inf. definition unavailable."
        
        # TODO update
        # sys_prompt = (
        #     "You are an IoT engineer.\n"
        #     "You are responsible for repairing an IoT-scale model for CIFAR10 under distribution drift.\n"
        #     "You are given reference vs drifted images; by comparing these, diagnose what is causing the drift and pick up to 3 actions from the provided available_actions."
        #     # "You are given telemetry (time-indexed scores, short feature vectors), small metadata summaries "
        #     # "for drifted vs reference images, the model architecture (ResNet18) and the inference function.\n\n"
        #     # "Diagnose what is going wrong (e.g., domain shift, normalization mismatch, under/overfitting) "
        #     # "and pick up to 3 actions from the provided available_actions. "
        #     "If you choose finetune actions, provide params: {'epochs': int, 'lr': float}.\n\n"
        #     "Respond ONLY with JSON of the form: "
        #     "{\"actions\":[{\"name\":str,\"params\":{...}}, ...], \"reasoning\": \"{your reasoning}\"} OR return updated code."
        # )
        sys_prompt = (
            "You are an IoT engineer operating on an image classifier deployed on edge devices.\n"
            "You are given:\n"
            "- reference (clean) images\n"
            "- shifted/corrupted images\n"
            "- short code context\n\n"
            "Your tasks:\n"
            "1) Compare the reference and shifted images and identify the most likely TYPE "
            "   of visual corruption affecting the shifted images. Choose one from:\n"
            "   ['blur', 'noise', 'brightness_change', 'contrast_change', 'color_shift', "
            "    'occlusion', 'translation', 'compression', 'other'].\n"
            "2) Diagnose what is going wrong with the model and pick up to 3 actions, in the order they should be applied, from available_actions.\n"
            "   Only use action names from available actions. When you use an action, set params within "
            "    ranges defined in the schema (e.g., epochs 1-20, learning rates 1e-5, 1e-2). If unsure, choose "
            "   conservative values. \n"
            "3) Optionally, you may propose a structured code patch using the 'code_patch' action with params:\n"
            "   {\n"
            "     \"target\": \"model_arch\" or \"preprocess\",\n"
            "     \"language\": \"python\",\n"
            "     \"patch_type\": \"full_definition\",\n"
            "     \"code\": \"...your nn.Module or preprocess code...\",\n"
            "     \"constraints\": {\"max_params\": int, \"max_flops\": int}\n"
            "   }\n"
            "   The code must be self-contained, use only PyTorch / torchvision operators, and respect the "
            "   constraints. It should define a drop-in replacement architecture or preprocessing.\n\n"
            "Respond ONLY with JSON of the form:\n"
            "{\n"
            "  \"corruption_type\": str,\n"
            "  \"actions\": [{\"name\": str, \"params\": {...}}, ...],\n"
            "  \"reasoning\": str\n"
            "}\n"
            "Do NOT include any free-form text outside this JSON.\n"
        )

        blocks.append({"type": "input_text", "text": sys_prompt})

        # # telemetry (stringified but compact)
        # try:
        #     telemetry_text = json.dumps(telemetry, default=str)
        # except Exception:
        #     telemetry_text = str(telemetry)
        # blocks.append({"type": "input_text", "text": "Telemetry (recent):"})
        # blocks.append({"type": "input_text", "text": telemetry_text})

        # include model / inference code as short references (may be long -- be cautious)
        # include only first N lines to avoid huge prompt
        def head_txt(s, nlines=60):
            return "\n".join((s or "").splitlines()[:nlines])

        blocks.append({"type": "input_text", "text": "Model architecture (head):"})
        blocks.append({"type": "input_text", "text": head_txt(model_code, nlines=40)})

        blocks.append({"type": "input_text", "text": "Inference function (head):"})
        blocks.append({"type": "input_text", "text": head_txt(inference_code, nlines=80)})

        # reference images / metadata
        blocks.append({"type": "input_text", "text": f"VISUAL TASK: Reference (clean) images/metadata:"})

        # detect whether ref_examples are images or metadata
        ref_to_send = ref_examples[:max_images_per_set]
        for item in ref_to_send:
            if isinstance(item, dict):
                blocks.append({"type":"input_text","text":json.dumps(item)})
                continue
            try:
                image_url = self._upload_image(item)
                blocks.append({"type":"input_image","image_url":image_url})
            except Exception:
                blocks.append({"type":"input_text","text":str(item)})

        # shifted / corrupted images
        blocks.append({"type": "input_text", "text": f"VISUAL TASK: Shifted/corrupted images/metadata:"})
        shifted_to_send = shifted_examples[:max_images_per_set]
        for item in shifted_to_send:
            if isinstance(item, dict):
                blocks.append({"type":"input_text","text":json.dumps(item)})
                continue
            try:
                image_url = self._upload_image(item)
                blocks.append({"type":"input_image","image_url":image_url})
            except Exception:
                blocks.append({"type":"input_text","text":str(item)})

        # available actions block
        avail = list(SUPPORTED_ACTIONS.keys())
        blocks.append({"type": "input_text", "text": "Available actions: " + json.dumps(avail)})

        return blocks

    def plan(self, telemetry, shifted_examples, ref_examples, max_images_per_set=4):
        """
        telemetry: list/dict (serializable)
        shifted_examples / ref_examples: lists of either numpy images, file paths, or metadata dicts
        """
        if self.client is None:
            # no client available -- fallback to a simple heuristic (safe)
            print("OpenAI client not configured; returning heuristic plan.")
            return [{"name": "recompute_normalization", "params": {}}]

        # Build blocks for the multimodal input; this will convert numpy arrays to base64 images
        blocks = self._make_prompt_blocks(telemetry, shifted_examples, ref_examples, max_images_per_set=max_images_per_set)

        # Wrap into the `input` array expected by the Responses API
        payload_input = [
            {"role": "system", "content": [{"type": "input_text", "text": "You are a helpful multimodal systems engineer."}]},
            {"role": "user", "content": blocks}
        ]

        try:
            resp = self.client.responses.create(
                model=self.vlm_model,
                input=payload_input,
            )
        except Exception as e:
            print("OpenAI call failed:", e)
            return []

        # Robustly extract text from response
        content_text = None
        # try output_text (SDK convenience)
        if hasattr(resp, "output_text") and resp.output_text:
            content_text = resp.output_text
        else:
            # try resp.output which is typically a list of items
            try:
                out = getattr(resp, "output", None)
                if out:
                    # collect text pieces
                    texts = []
                    for item in out:
                        # item may be dict-like with 'content'
                        try:
                            # content can be a list of elements
                            content = item.get("content", None) if isinstance(item, dict) else None
                        except Exception:
                            content = None
                        if content:
                            # join text pieces if present
                            if isinstance(content, list):
                                for c in content:
                                    if isinstance(c, dict) and c.get("type") == "output_text":
                                        texts.append(c.get("text", ""))
                                    elif isinstance(c, dict) and c.get("type") == "text":
                                        texts.append(c.get("text", ""))
                                    elif isinstance(c, str):
                                        texts.append(c)
                            elif isinstance(content, str):
                                texts.append(content)
                    if texts:
                        content_text = "\n".join(texts)
            except Exception:
                content_text = None

        # fallback
        if content_text is None:
            try:
                content_text = str(resp)
            except Exception:
                content_text = ""

        print(content_text)

        obj = {}
        # 1) Try direct JSON
        try:
            obj = json.loads(content_text)
        except Exception:
            # 2) Try fenced code block: ```json { ... } ```
            fenced = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```",
                content_text,
                re.DOTALL
            )
            if fenced:
                json_str = fenced.group(1)
                try:
                    obj = json.loads(json_str)
                except Exception:
                    obj = {}
            else:
                # 3) Fallback: take substring between first '{' and last '}'
                start = content_text.find("{")
                end = content_text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    candidate = content_text[start : end + 1]
                    try:
                        obj = json.loads(candidate)
                    except Exception:
                        obj = {}
                else:
                    obj = {}

        print(f"Object = {obj}")

        # Parse main fields
        corruption_type = obj.get("corruption_type", None)
        raw_actions = obj.get("actions", [])
        reasoning = obj.get("reasoning", "")

        builtin_actions = self._validate_builtin_actions(raw_actions)
        code_patches = self._extract_code_patches(raw_actions)

        return {
            "corruption_type": corruption_type,
            "builtin_actions": builtin_actions,
            "code_patches": code_patches,
            "reasoning": reasoning,
            "raw_response": content_text,
        }
