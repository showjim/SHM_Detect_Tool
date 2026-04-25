"""
setup.py — Cython build entry point for shm-detect-tool.

setuptools automatically discovers this file when running `python -m build`.
It compiles core Python modules into native binary extensions (.so / .pyd)
so that source code is NOT included in the distributed PyPI wheel.

Files intentionally kept as plain Python (not compiled):
  - shm_detect_tool/cli.py            — registered console entry point, must stay importable
  - shm_detect_tool/__init__.py       — package marker

Local development:
    python setup.py build_ext --inplace   # compile in-place for testing
    pip install -e .                      # install in editable mode

CI / release:
    python -m build --wheel               # produces a platform-specific wheel
"""

import os
import glob
from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize
from setuptools.extension import Extension

# ---------------------------------------------------------------------------
# Files to keep as plain Python — NOT compiled
# ---------------------------------------------------------------------------
SKIP_FILES = {
    "__init__.py",
    "cli.py",
}


def find_extensions(src_dir: str = "shm_detect_tool") -> list[Extension]:
    """Walk shm_detect_tool/ and return one Extension per .py file that should be compiled."""
    extensions = []
    for root, dirs, files in os.walk(src_dir):
        # Never descend into __pycache__
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fname in sorted(files):
            if not fname.endswith(".py"):
                continue
            if fname in SKIP_FILES:
                continue
            rel_path = os.path.join(root, fname)
            # Convert  shm_detect_tool/Source/src_pytorch_public.py  →  shm_detect_tool.Source.src_pytorch_public
            module_name = rel_path.replace(os.sep, ".")[:-3]
            extensions.append(Extension(module_name, [rel_path]))
    return extensions


class build_ext(_build_ext):
    """Custom build_ext that removes .py source files (for compiled modules)
    from the build lib directory so they are NOT included in the wheel."""

    def run(self):
        super().run()
        # After compilation, remove .py source files that have been compiled
        # to .so/.pyd binaries. Keep files listed in SKIP_FILES.
        if self.build_lib:
            for root, dirs, files in os.walk(self.build_lib):
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                for fname in files:
                    if not fname.endswith(".py"):
                        continue
                    if fname in SKIP_FILES:
                        continue
                    full_path = os.path.join(root, fname)
                    # Check if a corresponding .so or .pyd exists
                    base = os.path.splitext(full_path)[0]
                    has_binary = (
                        glob.glob(base + ".*.so") or
                        glob.glob(base + ".*.pyd") or
                        glob.glob(base + ".so") or
                        glob.glob(base + ".pyd")
                    )
                    if has_binary:
                        os.remove(full_path)
                        print(f"  Removed source file from wheel: {fname}")


if __name__ == '__main__':
    ext_modules = cythonize(
        find_extensions(),
        compiler_directives={
            "language_level": "3",   # Python 3 semantics
            "binding": True,         # keeps docstrings accessible
        },
        nthreads=4,
    )

    setup(
        name="shm-detect-tool",
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
    )
