from __future__ import annotations

import io
import tokenize
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NATIVE_PATH = ROOT / "train_gpt_sprint_native.py"
DEFAULT_OUTPUT_PATH = ROOT / "build" / "train_gpt_sprint_submit.py"

_FLASH_IMPORT_LINE = "from flash_attn_interface import flash_attn_func as flash_attn_3_func"
_FUTURE_IMPORT_LINE = "from __future__ import annotations"
_PRELUDE = """from __future__ import annotations
import os
os.environ.setdefault("BIGRAM_VOCAB_SIZE", "1536")
os.environ.setdefault("SUBMISSION_CODE_PATH", __file__)
try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
except Exception:
    import torch
    import torch.nn.functional as F
    def flash_attn_3_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **_):
        del dropout_p
        qh = q.permute(0, 2, 1, 3).contiguous()
        kh = k.permute(0, 2, 1, 3).contiguous()
        vh = v.permute(0, 2, 1, 3).contiguous()
        if qh.size(1) != kh.size(1):
            if qh.size(1) % kh.size(1) != 0:
                raise ValueError(
                    f"GQA fallback expected query heads to be divisible by KV heads, got {qh.size(1)} and {kh.size(1)}"
                )
            repeat = qh.size(1) // kh.size(1)
            kh = kh.repeat_interleave(repeat, dim=1)
            vh = vh.repeat_interleave(repeat, dim=1)
        out = F.scaled_dot_product_attention(
            qh,
            kh,
            vh,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
            scale=softmax_scale,
        )
        os.environ["SPRINT_FLASH_BACKEND"] = "sdpa_fallback"
        return out.permute(0, 2, 1, 3).contiguous()
else:
    os.environ.setdefault("SPRINT_FLASH_BACKEND", "flash_attn_interface")
"""


def _collapse_blank_lines(source: str) -> str:
    lines = source.splitlines()
    collapsed: list[str] = []
    previous_blank = False
    for line in lines:
        blank = not line.strip()
        if blank and previous_blank:
            continue
        collapsed.append(line.rstrip())
        previous_blank = blank
    return "\n".join(collapsed).strip() + "\n"


def _strip_comments(source: str) -> str:
    reader = io.StringIO(source).readline
    tokens = list(tokenize.generate_tokens(reader))
    pieces: list[tuple[int, str]] = []
    for token in tokens:
        token_type = token.type
        token_string = token.string
        if token_type == tokenize.COMMENT:
            continue
        pieces.append((token_type, token_string))
    return tokenize.untokenize(pieces)


def _native_submission_source(root: Path) -> str:
    source = (root / "train_gpt_sprint_native.py").read_text(encoding="utf-8")
    filtered_lines: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped == _FUTURE_IMPORT_LINE:
            continue
        if stripped == _FLASH_IMPORT_LINE:
            continue
        filtered_lines.append(line)
    stripped = _strip_comments("\n".join(filtered_lines) + "\n")
    return _collapse_blank_lines(stripped)


def build_submission_source(root: Path | None = None) -> str:
    repo_root = (root or ROOT).resolve()
    native_source = _native_submission_source(repo_root)
    return _PRELUDE + "\n" + native_source


def submission_output_path(root: Path | None = None) -> Path:
    repo_root = (root or ROOT).resolve()
    return repo_root / "build" / "train_gpt_sprint_submit.py"


def submission_code_bytes(root: Path | None = None) -> int:
    return len(build_submission_source(root).encode("utf-8"))


def ensure_submission_trainer(root: Path | None = None, out_path: Path | None = None) -> Path:
    repo_root = (root or ROOT).resolve()
    target = (out_path or submission_output_path(repo_root)).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    source = build_submission_source(repo_root)
    if not target.exists() or target.read_text(encoding="utf-8") != source:
        target.write_text(source, encoding="utf-8")
    return target
