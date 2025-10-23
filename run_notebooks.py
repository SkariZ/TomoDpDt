import nbformat
from nbclient import NotebookClient
from pathlib import Path
from datetime import datetime
import traceback
import time

# === CONFIGURATION ===
NOTEBOOK_DIR = Path("Notebooks")  # Folder containing your notebooks
KERNEL_NAME = "python3"           # Usually "python3" or the name shown in Jupyter
STOP_ON_ERROR = False             # Set True to stop at the first failure
LOG_FILE = "rerun_notebooks.log"  # Log file to store run info
KEYWORDS_TO_IGNORE = []          # List of keywords to ignore during execution
# ======================


def log(message: str):
    """Write message to both console and log file."""
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def run_notebooks():
    """Execute all notebooks in NOTEBOOK_DIR, overwriting them with fresh outputs."""
    # Create / reset the log file
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"Notebook run started at {datetime.now()}\n\n")

    # Collect all .ipynb files (alphabetical order)
    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))

    if not notebooks:
        raise FileNotFoundError(f"No notebooks found in {NOTEBOOK_DIR.resolve()}")

    # Remove notebooks that contain any of the ignore keywords in their filename
    if KEYWORDS_TO_IGNORE:
        notebooks = [
            nb for nb in notebooks
            if not any(keyword in nb.name for keyword in KEYWORDS_TO_IGNORE)
        ]
    
    log(f"📁 Found {len(notebooks)} notebooks in {NOTEBOOK_DIR.resolve()}")
    log("⚙️  Running notebooks (no timeout) and overwriting them with updated outputs...\n")

    for nb_path in notebooks:
        start_time = time.time()
        log(f"🚀 Running {nb_path.name} ...")

        try:
            # Load the notebook
            nb = nbformat.read(nb_path, as_version=4)

            # Create and execute a client with NO TIMEOUT
            client = NotebookClient(
                nb,
                timeout=None,  # <-- disables execution timeout
                kernel_name=KERNEL_NAME,
                resources={"metadata": {"path": nb_path.parent}},  # keeps notebook's folder as cwd
            )

            client.execute()  # Execute notebook

            # Overwrite the original notebook with executed version
            nbformat.write(nb, nb_path)

            elapsed = time.time() - start_time
            log(f"✅ Finished {nb_path.name} in {elapsed:.2f} seconds")

        except Exception as e:
            log(f"❌ Error while running {nb_path.name}: {e}")
            log("".join(traceback.format_exception(e)))
            if STOP_ON_ERROR:
                log("⛔ Stopping execution due to error.")
                break

    log("\n🎉 All notebooks processed.")
    log(f"Notebook run completed at {datetime.now()}")


if __name__ == "__main__":
    run_notebooks()
