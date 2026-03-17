import re
import os
import requests
import tarfile
from typing import Iterator
from collections import deque
import time
import sys
from pathlib import Path
from io import BytesIO

from pathlib import Path
import tempfile
import tarfile
import sys
import shutil

if sys.version_info >= (3, 14):
    from compression.zstd import ZstdFile
else:
    from zstandard import ZstdDecompressor


def extract_zstd(archive: Path, out_path: Path):
    """extract Zstandard .zst file
    works on Windows, Linux, MacOS, etc.
    Parameters
    ----------
    archive: pathlib.Path or str
      .zst file to extract
    out_path: pathlib.Path or str
      directory to extract files and directories to
    """

    archive = Path(archive).expanduser()
    out_path = Path(out_path).expanduser().resolve()
    # need .resolve() in case intermediate relative dir doesn't exist

    # we don't use the simpler "decompress" to allow arbitrarily large archives
    if sys.version_info >= (3, 14):
        with ZstdFile(archive, "rb") as f_in:
            with out_path.open("wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        _legacy_decompress_zstd_stream(archive, out_path)


def _legacy_decompress_zstd_stream(archive: Path, out_path: Path) -> None:

    with tempfile.TemporaryFile(suffix=".tar") as ofh:
        with archive.open("rb") as ifh:
            ZstdDecompressor().copy_stream(ifh, ofh)
        ofh.seek(0)
        with tarfile.open(fileobj=ofh) as z:
            z.extractall(out_path)

RV_BASE_CONTAINER_ROOT = "/tools/etiss_arch_riscv/rv_base"
HOST = os.environ.get("HOST", "localhost")
PORT = os.environ.get("PORT", "8080")
# USE_RICH = True
VERBOSE = True
# TODO MGCLIENT CDFGPASS

# if USE_RICH:
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn


assert len(sys.argv) >= 3
tag = sys.argv[1]
out_dir = sys.argv[2]
out_path = Path(out_dir)
out_path.mkdir(exist_ok=True)
# TODO: argparse


class RetargetClient:
    def __init__(self, api_url: str, token: str = None, console: Console = None):
        """
        api_url: base URL, e.g., "http://localhost:8080"
        token: optional API key / token if known
        """
        self.api_url = api_url.rstrip("/")
        self.token = token
        self.console = console

    def _rewrite_dsl(self, dsl_path: str) -> bytes:
        """
        Rewrites rv_base imports so they point to the container path.
        """

        dsl_filename = Path(dsl_path).name
        text = Path(dsl_path).read_text()

        import_re = re.compile(r'import\s+"([^"]+)"')

        def replace(match):
            path = match.group(1)

            if "rv_base" in path:
                new_path = path.split("rv_base", 1)[1].lstrip("/")
                return f'import "{RV_BASE_CONTAINER_ROOT}/{new_path}"'

            return match.group(0)

        rewritten = import_re.sub(replace, text)

        return dsl_filename, rewritten.encode("utf-8")

    def submit_job(self, tag: str, dsl_path: str, config_path: str):
        dsl_filename, rewritten_dsl = self._rewrite_dsl(dsl_path)
        files = {
            # "dsl": open(dsl_path, "rb"),
            "dsl": (dsl_filename, BytesIO(rewritten_dsl)),
            "config": open(config_path, "rb")
        }
        data = {
            "tag": tag,
        }
        resp = requests.post(f"{self.api_url}/jobs", files=files, data=data)
        resp.raise_for_status()
        data = resp.json()
        self.token = data["token"]
        return data["job_id"], data["token"]

    def get_status(self, job_id: str):
        resp = requests.get(f"{self.api_url}/jobs/{job_id}", params={"token": self.token})
        resp.raise_for_status()
        return resp.json()

    def download_artifact(self, job_id: str, dest_path: str):
        resp = requests.get(f"{self.api_url}/jobs/{job_id}/artifact", params={"token": self.token}, stream=True)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(1024*1024):
                f.write(chunk)

    def get_log_buf(self, job_id: str, n_lines: int = 50):
        resp = requests.get(f"{self.api_url}/jobs/{job_id}/logs", params={"token": self.token, "n_lines": n_lines, "stream": False})
        resp.raise_for_status()
        return resp.text

    def stream_logs(self, job_id: str, n_lines: int = 0) -> Iterator[str]:
        """
        Stream logs line by line (tail -f style).
        """
        with requests.get(
            f"{self.api_url}/jobs/{job_id}/logs",
            params={"token": self.token, "n_lines": n_lines, "stream": True},
            stream=True
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if line:
                    yield line

    def wait_for_completion(self, job_id: str, poll_interval: float = 5.0, verbose: bool = False):
        """Waits for job completion and prints live progress"""
        last_progress = -1
        n_lines = 50
        # if USE_RICH:
        # from rich import inspect
        # inspect(console)
        buffer = deque(maxlen=n_lines)
        buffer.append("Waiting for job to start... (queued)")
        # buffer = []
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TextColumn("Elapsed: {task.fields[elapsed_seconds]:.1f}s"),
            TextColumn("ETA: {task.fields[eta_seconds]:.1f}s"),
            # TimeRemainingColumn(),
        )

        task = progress.add_task(f"Running job {job_id}", total=100, elapsed_seconds=0, eta_seconds=0)

        layout = Layout()
        layout.split_column(
            Layout(progress, size=3),
            Layout(name="log")
        )
        def render() -> Layout:
            # console.print("render")
            layout["log"].update(
                Panel("\n".join(buffer), title=f"Last {n_lines} Lines")
            )
            return layout
        # TODO: add dummy context if disabled
        #  with Live(render, console=console, refresh_per_second=2):
        assert self.console is not None
        with Live(render(), console=self.console, refresh_per_second=2):
            log_stream = self.stream_logs(job_id, n_lines=n_lines)
            while True:
                status = self.get_status(job_id)
                # if verbose:
                #     assert USE_RICH
                #     if status["status"] in ("finished", "running"):
                #         # log_buf = self.get_log_buf(job_id, n_lines=n_lines)
                #         buffer = log_buf.splitlines()
                #         print("===!!!===")
                #         print("log_buf", log_buf)
                #         print("===???===")
                prog = status.get("progress", 0.0)
                # console.print("prog", prog)
                prog_percent = int(prog*100)
                # if prog_percent != last_progress:
                if prog_percent >= last_progress:
                    # if USE_RICH:
                    if True:
                        elapsed_seconds = status.get("elapsed_seconds", 0.0)
                        eta_seconds = status.get("eta_seconds", 0.0)
                        progress.update(task, completed=prog_percent, elapsed_seconds=elapsed_seconds, eta_seconds=eta_seconds)
                    else:
                        sys.stdout.write(f"\rJob {job_id} progress: {prog_percent}%")
                        sys.stdout.flush()
                    last_progress = prog_percent
                try:
                    # console.print("try")
                    # for _ in range(5):  # fetch a few lines at a time
                    # for _ in range(n_lines):  # fetch a few lines at a time
                    while True:
                        line = next(log_stream)
                        # console.print("line", line)
                        buffer.append(line)
                        # layout["log"].update(
                        #     Panel("\n".join(buffer), title=f"Last {n_lines} Lines")
                        # )
                except StopIteration:
                    # console.print("stop")
                    pass
                render()
                if status["status"] in ("finished", "failed"):
                    console.print(f"\nJob {job_id} completed with status: {status['status']}")
                    error = status.get("error")
                    if error:
                        console.print(f"ERROR: {error}")
                    break
                time.sleep(poll_interval)
        return status["status"]

    def download_logs(self, job_id: str, dest_path: str):
    
        resp = requests.get(
            f"{self.api_url}/jobs/{job_id}/logs",
            params={"token": self.token},
            stream=True
        )
    
        resp.raise_for_status()
    
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(1024 * 1024):
                f.write(chunk)


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    console = Console(force_terminal=True, force_interactive=True)
    client = RetargetClient(f"http://{HOST}:{PORT}", console=console)

    # Submit a job
    # job_id, token = client.submit_job("test_files/test.dsl", "test_files/config.yaml")
    # job_id, token = client.submit_job(tag, "test_files/XIsaac.splitted.core_desc", "test_files/config.yaml")
    cdsl_files = []
    cfg_files = []
    for arg in sys.argv[3:]:
       file_path = Path(arg)
       assert file_path.is_file(), f"Not a file: {arg}"
       suffix = file_path.suffix
       if suffix in [".core_desc", ".cdsl"]:
           cdsl_files.append(file_path)
       elif suffix in [".yaml", ".yml"]:
           cfg_files.append(file_path)
    assert len(cdsl_files) == 1
    cdsl_file = cdsl_files[0]
    if len(cfg_files) > 1:
        contents = []
        for file in cfg_files:
            with open(file, "r") as f:
                content = f.read()
                contents.append(content)
        merged_contents = "\n".join(contents)
        merged_contents = merged_contents.replace("---\n", "\n")

        # TODO: merge for real
        merged_cfg_file = out_path / "merged_config.yml"
        with open(merged_cfg_file, "w") as f:
            f.write(merged_contents)
        cfg_file = merged_cfg_file
    else:
        assert len(cfg_files) == 1
        cfg_file = cfg_files[0]
    job_id, token = client.submit_job(tag, str(cdsl_file), str(cfg_file))
    console.print(f"Submitted job {job_id} with token {token}")
    time.sleep(1)

    # Wait for completion while showing progress
    status = client.wait_for_completion(job_id, verbose=VERBOSE)
    # if status == "failed":
    log_file = out_path / f"{job_id}_logs.txt"
    console.print(f"Downloading logs to {log_file}")
    client.download_logs(job_id, str(log_file))
    if status == "failed":
        console.print("Artifact not available (job failed)")
    else:
        # Download artifact
        artifact_file = out_path / f"{job_id}_artifact.tar.zst"
        console.print(f"Downloading artifact to {artifact_file}")
        client.download_artifact(job_id, str(artifact_file))
        console.print("Download finished")
        console.print(f"Extracting artifact to {out_path}")
        extract_zstd(artifact_file, out_path)
        console.print("Extraction finished")
        console.print("Moving files")
        shutil.copytree(out_path / "output", out_path, dirs_exist_ok=True)
        shutil.rmtree(out_path / "output")
        console.print("Done")
        console.print(f"Removing archive {artifact_file}")
        os.remove(artifact_file)
        console.print("Done")
