"""Execute the three EDA notebooks with this interpreter, preserving clean sources."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/eda-notebooks"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parent.parent
    for variable, directory in {
        "IPYTHONDIR": "ipython",
        "JUPYTER_RUNTIME_DIR": "jupyter-runtime",
        "JUPYTER_CONFIG_DIR": "jupyter-config",
        "MPLCONFIGDIR": "matplotlib",
    }.items():
        os.environ.setdefault(variable, str(project_root / ".cache" / directory))
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    for path in sorted((project_root / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        try:
            NotebookClient(
                notebook,
                timeout=180,
                km=manager,
                resources={"metadata": {"path": str(project_root)}},
            ).execute()
            nbformat.write(notebook, args.output / path.name)
            print(f"Executed {path.name}", flush=True)
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)


if __name__ == "__main__":
    main()
