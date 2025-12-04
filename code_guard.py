# code_guard.py
import ast
import textwrap
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

ALLOWED_IMPORT_MODULES = {
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torchvision.models",
    "torchvision.transforms",
}

ALLOWED_NODE_TYPES = {
    ast.Module,
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.ClassDef,
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.Return,
    ast.Expr,
    ast.Call,
    ast.Attribute,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.arguments,
    ast.arg,
    ast.If,
    ast.For,
    ast.While,
    ast.BinOp,
    ast.UnaryOp,
    ast.Num,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Subscript,
    ast.Compare,
    ast.BoolOp,
    ast.IfExp,
    ast.With,
    ast.Try,
    # (you can tighten this further)
}

FORBIDDEN_NAMES = {
    "open", "exec", "eval", "__import__", "subprocess", "os", "sys", "shutil", "requests"
}

def _check_ast_safety(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if type(node) not in ALLOWED_NODE_TYPES:
            return False
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.name.split(".")[0] not in {m.split(".")[0] for m in ALLOWED_IMPORT_MODULES}:
                    return False
        if isinstance(node, ast.ImportFrom):
            if node.module is None:
                return False
            root = node.module.split(".")[0]
            if root not in {m.split(".")[0] for m in ALLOWED_IMPORT_MODULES}:
                return False
        if isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                return False
    return True

def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def validate_and_build_model_patch(
    code: str,
    target: str,
    constraints: Dict[str, Any],
) -> Tuple[bool, Optional[Any], str]:
    """
    Validate code patch and build a callable / class if safe.
    For target == 'model_arch', expect a nn.Module subclass called NewBackbone.
    For target == 'preprocess', expect a function build_preprocess() returning a transform.
    Returns: (ok, obj, message)
    """
    code = textwrap.dedent(code)
    try:
        tree = ast.parse(code)
    except Exception as e:
        return False, None, f"AST parse failed: {e}"

    if not _check_ast_safety(tree):
        return False, None, "AST safety check failed"

    # Execute in restricted globals
    safe_globals = {
        "__builtins__": {
            "range": range,
            "len": len,
            "print": print,
            "min": min,
            "max": max,
            "float": float,
            "int": int,
            "bool": bool,
            "list": list,
            "dict": dict,
            "zip": zip,
            "enumerate": enumerate,
            # add any other safe builtins you need
        },
        "torch": torch,
        "nn": nn,
    }
    import torchvision
    safe_globals["torchvision"] = torchvision
    safe_globals["models"] = torchvision.models
    safe_locals: Dict[str, Any] = {}

    try:
        exec(compile(tree, filename="<llm_patch>", mode="exec"), safe_globals, safe_locals)
    except Exception as e:
        return False, None, f"Execution failed: {e}"

    if target == "model_arch":
        NewBackbone = safe_locals.get("NewBackbone", None)
        if NewBackbone is None:
            return False, None, "NewBackbone class not found in code_patch"

        try:
            model = NewBackbone()
        except Exception as e:
            return False, None, f"Failed to instantiate NewBackbone: {e}"

        # Dummy forward to test compatibility
        try:
            x = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                out = model(x)
        except Exception as e:
            return False, None, f"Dummy forward failed: {e}"

        # Check param count
        max_params = int(constraints.get("max_params", 10_000_000))
        n_params = _count_params(model)
        if n_params > max_params:
            return False, None, f"Param count {n_params} > max_params {max_params}"

        return True, model, "OK"

    elif target == "preprocess":
        build_fn = safe_locals.get("build_preprocess", None)
        if build_fn is None:
            return False, None, "build_preprocess() not found in code_patch"

        try:
            preprocess = build_fn()
        except Exception as e:
            return False, None, f"Failed to build preprocess: {e}"

        return True, preprocess, "OK"

    else:
        return False, None, f"Unknown target {target}"
