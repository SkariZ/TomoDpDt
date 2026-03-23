import contextlib
import io
import os
import re
import time
import traceback
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat import from_dict

# === CONFIG ===
NOTEBOOK_DIR = Path("Notebooks")
LOG_FILE = Path("Logs/rerun_notebooks.log")
KERNEL_NAME = "python3"
STOP_ON_ERROR = False
KEYWORDS_TO_IGNORE = []
# ==============

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    """Log to console and file."""
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


SUPPRESS_PATTERNS = [
    r"Epoch\s+\d+",
    r"batch",
    r"\d+%|\|.*\||it/s|ETA|elapsed|v_num|step:",
    r"Training|Validating",
    r"loss",
    r"GPU available|TPU available|CUDA",
]


def _looks_noisy(text: str) -> bool:
    """Detect whether a line looks like training or progress noise."""
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in SUPPRESS_PATTERNS)


def scrub_notebook_outputs(nb, keep_last_lines=5, max_stream_chars=4000):
    """
    Compact or remove noisy training logs in notebook outputs.
    Keeps plots and useful text intact.
    """
    scrubbed_count = 0

    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue

        new_outputs = []
        for out in cell.get("outputs", []):
            if isinstance(out, dict):
                out = from_dict(out)
            elif not hasattr(out, "output_type"):
                continue

            output_type = getattr(out, "output_type", None)

            if output_type == "stream" and isinstance(out.get("text"), str):
                text = out["text"]
                if len(text) > max_stream_chars or _looks_noisy(text):
                    scrubbed_count += 1
                    lines = [line for line in text.splitlines() if not _looks_noisy(line)]
                    tail = "\n".join(lines[-keep_last_lines:]).strip()
                    summary = f"[suppressed training logs: {len(text):,} chars]\n"
                    compact = summary + (tail + "\n" if tail else "")
                    out = from_dict(
                        {
                            "output_type": "stream",
                            "name": out.get("name", "stdout"),
                            "text": compact,
                        }
                    )
                new_outputs.append(out)
                continue

            if output_type in ("display_data", "execute_result"):
                data = out.get("data", {})
                if any(kind in data for kind in ("image/png", "image/jpeg", "image/svg+xml")):
                    new_outputs.append(out)
                    continue

                text_plain = data.get("text/plain")
                if isinstance(text_plain, str) and (
                    _looks_noisy(text_plain) or len(text_plain) > max_stream_chars
                ):
                    scrubbed_count += 1
                    data["text/plain"] = "[suppressed display output]\n"
                    out["data"] = data
                new_outputs.append(out)
                continue

            new_outputs.append(out)

        if len(new_outputs) > 1:
            deduped = []
            for output in new_outputs:
                if not deduped or output.get("text") != deduped[-1].get("text"):
                    deduped.append(output)
            new_outputs = deduped

        cell["outputs"] = new_outputs
        if new_outputs:
            cell.setdefault("metadata", {}).setdefault("jupyter", {})["outputs_hidden"] = False

    if scrubbed_count:
        log(f"   Scrubbed {scrubbed_count} noisy outputs")
    return nb


class StreamFilter(io.StringIO):
    """Filter out Lightning or tqdm spam in real time."""

    def write(self, s):
        if not s.strip():
            return
        if any(re.search(pattern, s, re.IGNORECASE) for pattern in SUPPRESS_PATTERNS):
            return
        super().write(s)


def run_notebooks():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"Notebook run started at {datetime.now()}\n\n")

    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if not notebooks:
        raise FileNotFoundError(f"No notebooks found in {NOTEBOOK_DIR.resolve()}")

    if KEYWORDS_TO_IGNORE:
        notebooks = [nb for nb in notebooks if not any(keyword in nb.name for keyword in KEYWORDS_TO_IGNORE)]

    log(f"Found {len(notebooks)} notebooks in {NOTEBOOK_DIR.resolve()}")
    log("Running notebooks and cleaning outputs...\n")

    os.environ["NB_AUTOMATED_RUN"] = "1"

    for nb_path in notebooks:
        start_time = time.time()
        log(f"Running {nb_path.name} ...")
        try:
            nb = nbformat.read(nb_path, as_version=4)
            client = NotebookClient(
                nb,
                timeout=None,
                kernel_name=KERNEL_NAME,
                resources={"metadata": {"path": nb_path.parent}},
            )

            filtered = StreamFilter()
            with contextlib.redirect_stdout(filtered), contextlib.redirect_stderr(filtered):
                client.execute()

            filtered_text = filtered.getvalue().strip()
            if filtered_text:
                log(filtered_text)

            nb = scrub_notebook_outputs(nb, keep_last_lines=8, max_stream_chars=4000)
            nbformat.write(nb, nb_path)
            elapsed_minutes = (time.time() - start_time) / 60
            log(f"Finished {nb_path.name} in {elapsed_minutes:.2f} min\n")
        except Exception as exc:
            log(f"Error in {nb_path.name}: {exc}")
            log("".join(traceback.format_exception(exc)))
            if STOP_ON_ERROR:
                break

    log("\nAll notebooks processed.")
    log(f"Completed at {datetime.now()}")


if __name__ == "__main__":
    run_notebooks()
