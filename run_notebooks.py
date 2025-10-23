import nbformat
from nbclient import NotebookClient
from pathlib import Path
from datetime import datetime
import traceback
import time
import contextlib
import io
import re
import sys
import os

NOTEBOOK_DIR = Path("Notebooks")
LOG_FILE = Path("Logs/rerun_notebooks.log")
KERNEL_NAME = "python3"
STOP_ON_ERROR = False
KEYWORDS_TO_IGNORE = []

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


class StreamFilter(io.StringIO):
    """
    Filters out batch/progress noise from Lightning, tqdm, etc.
    Keeps your own print() output intact.
    """
    def write(self, s):
        # Suppress only lines that look like tqdm or Lightning progress bars
        if re.match(r"^Epoch\s+\d+|^\d+%|\|.*\||it/s|ETA|elapsed|v_num|step:", s):
            return
        # Suppress repeated whitespace or carriage return artifacts
        if s.strip() in ["", "\r", "\n"]:
            return
        super().write(s)


def run_notebooks():
    log(f"Notebook run started at {datetime.now()}\n")

    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if not notebooks:
        raise FileNotFoundError(f"No notebooks found in {NOTEBOOK_DIR.resolve()}")

    log(f"📁 Found {len(notebooks)} notebooks in {NOTEBOOK_DIR.resolve()}")
    log("⚙️  Running notebooks (silent training)...\n")

    os.environ["NB_AUTOMATED_RUN"] = "1"

    for nb_path in notebooks:
        start = time.time()
        log(f"🚀 Running {nb_path.name} ...")

        try:
            nb = nbformat.read(nb_path, as_version=4)
            client = NotebookClient(
                nb,
                timeout=None,
                kernel_name=KERNEL_NAME,
                resources={"metadata": {"path": nb_path.parent}},
            )

            filtered_out = StreamFilter()

            # Suppress BOTH stdout and stderr (tqdm, Lightning logs)
            with contextlib.redirect_stdout(filtered_out), contextlib.redirect_stderr(filtered_out):
                client.execute()

            log(filtered_out.getvalue())
            nbformat.write(nb, nb_path)

            elapsed = time.time() - start
            log(f"✅ Finished {nb_path.name} in {elapsed/60:.2f} min")

        except Exception as e:
            log(f"❌ Error in {nb_path.name}: {e}")
            log("".join(traceback.format_exception(e)))
            if STOP_ON_ERROR:
                break

    log("\n🎉 All notebooks processed.")
    log(f"Completed at {datetime.now()}")


if __name__ == "__main__":
    run_notebooks()


