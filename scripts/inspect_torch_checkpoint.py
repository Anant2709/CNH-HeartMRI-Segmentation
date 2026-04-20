#!/usr/bin/env python3
"""
Safely inspect a .pt file: PyTorch version, top-level keys, and tensor shapes.

Why cautious loading matters:
  Checkpoints can execute code when unpickled. Prefer weights_only=True when you only
  need tensors. Full training checkpoints may require weights_only=False — use that only
  for files you trust.

Usage:
  python scripts/inspect_torch_checkpoint.py path/to/model.pt
  python scripts/inspect_torch_checkpoint.py path/to/model.pt --unsafe-full-pickle
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a PyTorch .pt checkpoint structure.")
    parser.add_argument("checkpoint", type=Path, help="Path to .pt or .pth file")
    parser.add_argument(
        "--unsafe-full-pickle",
        action="store_true",
        help="Use weights_only=False (only for trusted files; security risk otherwise)",
    )
    args = parser.parse_args()
    path = args.checkpoint.expanduser().resolve()
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        import torch
    except ImportError:
        print("Install PyTorch: pip install torch", file=sys.stderr)
        sys.exit(1)

    print(f"PyTorch version: {torch.__version__}")
    print(f"File: {path} ({path.stat().st_size / (1024 * 1024):.2f} MB)")

    obj = None
    load_mode = None
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        load_mode = "weights_only=True"
    except Exception as e:
        print(f"\nweights_only=True failed: {type(e).__name__}: {e}")
        print(
            "Often means optimizer/config objects are in the file. "
            "If you trust it, re-run with --unsafe-full-pickle.\n"
        )

    if obj is None and args.unsafe_full_pickle:
        print("Loading with weights_only=False (trusted file only)...")
        obj = torch.load(path, map_location="cpu", weights_only=False)
        load_mode = "weights_only=False"

    if obj is None:
        sys.exit(1)

    print(f"\nLoad mode: {load_mode}")
    print(f"Top-level Python type: {type(obj)}")

    if isinstance(obj, dict):
        print("Top-level keys:")
        for k in sorted(obj.keys(), key=lambda x: str(x)):
            v = obj[k]
            line = f"  {k!r}: {type(v).__name__}"
            if hasattr(v, "shape"):
                line += f" shape={tuple(v.shape)} dtype={v.dtype}"
            print(line)
        for key in ("model", "state_dict", "net", "network"):
            if key in obj and hasattr(obj[key], "keys"):
                sd = obj[key]
                print(f"\nNested under {key!r} ({len(sd)} entries), first 15:")
                for i, (name, t) in enumerate(sd.items()):
                    if i >= 15:
                        print(f"  ... ({len(sd) - 15} more)")
                        break
                    if hasattr(t, "shape"):
                        print(f"    {name}: {tuple(t.shape)} {t.dtype}")
                    else:
                        print(f"    {name}: {type(t).__name__}")
    elif hasattr(obj, "state_dict"):
        print(
            "Object has .state_dict() — likely nn.Module; you need the Python class "
            "that defines the architecture to reconstruct the model."
        )
    else:
        print("Unexpected top-level type; manual inspection needed.")


if __name__ == "__main__":
    main()
