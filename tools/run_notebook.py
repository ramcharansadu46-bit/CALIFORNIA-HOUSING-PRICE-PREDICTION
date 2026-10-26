"""Execute a notebook headlessly:  python tools/run_notebook.py <in.ipynb> <out.ipynb>"""
import sys, nbformat
from nbclient import NotebookClient
src, dst = sys.argv[1], sys.argv[2]
nb = nbformat.read(src, as_version=4)
client = NotebookClient(nb, timeout=7200, kernel_name="python3", resources={"metadata": {"path": str(__import__('pathlib').Path(src).resolve().parent)}})
try:
    client.execute()
    status = "OK"
except Exception as e:
    status = f"FAILED: {type(e).__name__}: {str(e)[:3000]}"
nbformat.write(nb, dst)
print(status)
