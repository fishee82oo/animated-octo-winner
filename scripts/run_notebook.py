"""Execute the walkthrough with this interpreter and save checked outputs."""

import sys
from pathlib import Path
import tempfile
import json
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "notebooks/01_exploration.ipynb"
    nb = nbformat.read(path, as_version=4)
    with tempfile.TemporaryDirectory(prefix="power-spot-kernel-") as tmp:
        kernel = Path(tmp) / "power-spot"
        kernel.mkdir()
        (kernel / "kernel.json").write_text(
            json.dumps(
                dict(
                    argv=[
                        sys.executable,
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    display_name="Power Spot Python 3.11",
                    language="python",
                )
            )
        )
        manager = KernelSpecManager(kernel_dirs=[tmp])
        client = NotebookClient(
            nb,
            timeout=180,
            kernel_name="power-spot",
            resources={"metadata": {"path": str(ROOT)}},
            kernel_manager_class=__import__("jupyter_client").KernelManager,
        )
        client.create_kernel_manager()
        client.km.kernel_spec_manager = manager
        client.execute()
    nbformat.write(nb, path)
    count = sum(c.cell_type == "code" for c in nb.cells)
    print(f"Executed {count} code cells without errors: {path}")


if __name__ == "__main__":
    main()
