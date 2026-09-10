"""Run a Python entry point with one shared OpenMP runtime on macOS.

Example: python scripts/run_macos.py scripts/train_retriever.py --index-only
Also supports modules: python scripts/run_macos.py -m uvicorn src.api.app:app
"""
import importlib.util
import os
from pathlib import Path
import sys


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    env = os.environ.copy()
    if sys.platform == "darwin":
        spec = importlib.util.find_spec("torch")
        if spec is None or spec.origin is None:
            raise SystemExit("Install PyTorch in the active Python environment first")
        library_dir = Path(spec.origin).parent / "lib"
        if not (library_dir / "libomp.dylib").exists():
            raise SystemExit(f"Missing OpenMP runtime: {library_dir / 'libomp.dylib'}")
        # Set before process startup: dyld resolves libomp to a single library.
        previous = env.get("DYLD_LIBRARY_PATH", "")
        env["DYLD_LIBRARY_PATH"] = str(library_dir) + (os.pathsep + previous if previous else "")
        env.pop("KMP_DUPLICATE_LIB_OK", None)
    os.execve(sys.executable, [sys.executable, *sys.argv[1:]], env)


if __name__ == "__main__":
    main()
