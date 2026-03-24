#
# Copyright (c) 2026 TUM Department of Electrical and Computer Engineering.
#
# This file is part of ISAAC Toolkit.
# See https://github.com/tum-ei-eda/isaac-toolkit.git for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import re
import os
import sys
import time
import shutil
import argparse
from io import BytesIO
from pathlib import Path
from collections import deque
from abc import ABC, abstractmethod
from typing import Iterator, List

import requests
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import Progress, BarColumn, TextColumn

from isaac_toolkit.utils.compression import extract_zstd

RV_BASE_CONTAINER_ROOT = "/tools/etiss_arch_riscv/rv_base"
# TODO MGCLIENT CDFGPASS


class RetargetClient(ABC):

    def __init__(self, api_url: str, token: str = None, console: Console = None):
        """
        api_url: base URL, e.g., "http://localhost:8080"
        token: optional API key / token if known
        console: rich.Console instance for live progress
        """
        self.api_url = api_url.rstrip("/")
        self.token = token
        self.console = console

    def _rewrite_cdsl(self, cdsl_path: str) -> bytes:
        """
        Rewrites rv_base imports so they point to the container path.
        """

        cdsl_filename = Path(cdsl_path).name
        text = Path(cdsl_path).read_text()

        import_re = re.compile(r'import\s+"([^"]+)"')

        def replace(match):
            path = match.group(1)

            if "rv_base" in path:
                new_path = path.split("rv_base", 1)[1].lstrip("/")
                return f'import "{RV_BASE_CONTAINER_ROOT}/{new_path}"'

            return match.group(0)

        rewritten = import_re.sub(replace, text)

        return cdsl_filename, rewritten.encode("utf-8")

    def _submit_job(self, files: dict = None, data: dict = None):
        resp = requests.post(f"{self.api_url}/jobs", files=files, data=data)
        resp.raise_for_status()
        data = resp.json()
        self.token = data["token"]
        return data["job_id"], data["token"]

    @abstractmethod
    def submit_job(self, tag: str, *args, **kwargs):
        raise NotImplementedError

    def get_status(self, job_id: str):
        resp = requests.get(f"{self.api_url}/jobs/{job_id}", params={"token": self.token})
        resp.raise_for_status()
        return resp.json()

    def download_artifact(self, job_id: str, dest_path: str):
        resp = requests.get(f"{self.api_url}/jobs/{job_id}/artifact", params={"token": self.token}, stream=True)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(1024 * 1024):
                f.write(chunk)

    def get_log_buf(self, job_id: str, n_lines: int = 50):
        resp = requests.get(
            f"{self.api_url}/jobs/{job_id}/logs", params={"token": self.token, "n_lines": n_lines, "stream": False}
        )
        resp.raise_for_status()
        return resp.text

    def stream_logs(self, job_id: str, n_lines: int = 0) -> Iterator[str]:
        """
        Stream logs line by line (tail -f style).
        """
        with requests.get(
            f"{self.api_url}/jobs/{job_id}/logs",
            params={"token": self.token, "n_lines": n_lines, "stream": True},
            stream=True,
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
        layout.split_column(Layout(progress, size=3), Layout(name="log"))

        def render() -> Layout:
            # console.print("render")
            layout["log"].update(Panel("\n".join(buffer), title=f"Last {n_lines} Lines"))
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
                prog_percent = int(prog * 100)
                # if prog_percent != last_progress:
                if prog_percent >= last_progress:
                    # if USE_RICH:
                    if True:
                        elapsed_seconds = status.get("elapsed_seconds", 0.0)
                        eta_seconds = status.get("eta_seconds", 0.0)
                        progress.update(
                            task, completed=prog_percent, elapsed_seconds=elapsed_seconds, eta_seconds=eta_seconds
                        )
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
                    self.console.print(f"\nJob {job_id} completed with status: {status['status']}")
                    error = status.get("error")
                    if error:
                        self.console.print(f"ERROR: {error}")
                    break
                time.sleep(poll_interval)
        return status["status"]

    def download_logs(self, job_id: str, dest_path: str):

        resp = requests.get(f"{self.api_url}/jobs/{job_id}/logs", params={"token": self.token}, stream=True)

        resp.raise_for_status()

        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(1024 * 1024):
                f.write(chunk)


class Seal5RetargetClient(RetargetClient):

    def submit_job(self, tag: str, cdsl_path: str = None, config_path: str = None, **kwargs):
        assert cdsl_path is not None
        assert config_path is not None
        cdsl_filename, rewritten_cdsl = self._rewrite_cdsl(cdsl_path)
        files = {"cdsl": (dsl_filename, BytesIO(rewritten_cdsl)), "config": open(config_path, "rb")}
        data = {
            "tag": tag,
        }
        return self._submit_job(files=files, data=data)


class EtissRetargetClient(RetargetClient):

    def submit_job(self, tag: str, cdsl_path: str = None, **kwargs):
        assert cdsl_path is not None
        dsl_filename, rewritten_cdsl = self._rewrite_cdsl(dsl_path)
        files = {
            "cdsl": (dsl_filename, BytesIO(rewritten_cdsl)),
        }
        data = {
            "tag": tag,
        }
        return self._submit_job(files=files, data=data)


class EtissPerfRetargetClient(RetargetClient):

    def submit_job(self, tag: str, cdsl_path: str = None, cpdsl_path: str = None, **kwargs):
        assert cdsl_path is not None
        assert cpdsl_path is not None
        cdsl_filename, rewritten_cdsl = self._rewrite_cdsl(cdsl_path)
        cpdsl_filename = Path(cpdsl_path).name
        files = {
            "cdsl": (cdsl_filename, BytesIO(rewritten_cdsl)),
            "cpdsl": (cpdsl_filename, cpdsl_path),
        }
        data = {
            "tag": tag,
        }
        return self._submit_job(files=files, data=data)


def run_retargeting_service(
    files: List[str],
    out_dir: str = None,
    verbose: bool = False,
    host: str = "localhost",
    port: int = 8080,
    tag: str = None,
):

    assert out_dir is not None
    assert tag is not None
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)
    console = Console(force_terminal=True, force_interactive=True) if verbose else None

    client_lookup = {
        "llvm": Seal5RetargetClient,
        "iss": EtissRetargetClient,
        "perf": EtissPerfRetargetClient,
    }
    client_cls = client_lookup.get(tag)
    assert client_cls is not None, f"Unable to find client class for tag: {tag}"

    client = RetargetClient(f"http://{host}:{port}", console=console)

    # Separate input files
    cdsl_files: List[Path] = []
    cpdsl_files: List[Path] = []
    cfg_files: List[Path] = []
    for f in files:
        file_path = Path(f)
        if not file_path.is_file():
            raise ValueError(f"Not a file: {f}")
        if file_path.suffix in [".core_desc", ".cdsl"]:
            cdsl_files.append(file_path)
        elif file_path.suffix in [".core_perf_dsl"]:
            cpdsl_files.append(file_path)
        elif file_path.suffix in [".yaml", ".yml"]:
            cfg_files.append(file_path)
    if len(cdsl_files) != 1:
        raise ValueError(f"Expected exactly one .core_desc/.cdsl file, got {len(cdsl_files)}")
    cdsl_file = cdsl_files[0]
    cpdsl_file = None
    if len(cpdsl_files) > 0:
        if len(cpdsl_files) > 1:
            raise ValueError(f"Expected exactly one .core_perf_dsl file, got {len(cpdsl_files)}")
        cpdsl_file = cpdsl_files[0]

    # Merge configs if multiple
    if len(cfg_files) > 1:
        merged_contents = "\n".join([f.read_text() for f in cfg_files]).replace("---\n", "\n")
        merged_cfg_file = out_path / "merged_config.yml"
        merged_cfg_file.write_text(merged_contents)
        cfg_file = merged_cfg_file
    else:
        cfg_file = cfg_files[0]

    job_id, token = client.submit_job(tag, cdsl_path=str(cdsl_file), cfg_path=str(cfg_file), cpdsl_path=str(cpdsl_file))

    if console:
        console.print(f"Submitted job {job_id} with token {token}")
    time.sleep(1)

    # Wait for completion
    status = client.wait_for_completion(job_id) if console else None

    # Download logs
    log_file = out_path / f"{job_id}_logs.txt"
    if console:
        console.print(f"Downloading logs to {log_file}")
    client.download_logs(job_id, str(log_file))

    if status == "failed":
        if console:
            console.print("Artifact not available (job failed)")
    else:
        # Download artifact
        artifact_file = out_path / f"{job_id}_artifact.tar.zst"
        if console:
            console.print(f"Downloading artifact to {artifact_file}")
        client.download_artifact(job_id, str(artifact_file))
        if console:
            console.print("Download finished")
            console.print(f"Extracting artifact to {out_path}")
        extract_zstd(artifact_file, out_path)

        # Move files
        shutil.copytree(out_path / "output", out_path, dirs_exist_ok=True)
        shutil.rmtree(out_path / "output")
        if console:
            console.print("Done")
            console.print(f"Removing archive {artifact_file}")
        os.remove(artifact_file)
        if console:
            console.print("Done")


def handle(args):
    # assert args.session is not None
    sess = None
    if args.session is not None:
        session_dir = Path(args.session)
        assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
        sess = Session.from_dir(session_dir)
    set_log_level(console_level=args.log, file_level=args.log)
    run_retargeting_service(
        args.files,
        out_dir=args.out_dir,
        force=args.force,
        verbose=args.verbose,
        host=args.host,
        port=args.port,
        tag=args.tag,
    )
    if sess is not None:
        sess.save()


def get_parser():
    parser = argparse.ArgumentParser(description="Submit jobs to Retarget server and fetch results")
    parser.add_argument("files", nargs="+", help="Input files (.core_desc/.cdsl and config YAML)")
    parser.add_argument("--tag", required=True, help="Job tag")
    parser.add_argument("--output", "-o", dest="out_dir", help="Output directory")
    parser.add_argument("--session", "--sess", "-s", type=str, required=False)
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--host", default=os.environ.get("HOST", "localhost"), help="Server host override")
    parser.add_argument("--port", default=os.environ.get("PORT", "8080"), help="Server port override")
    parser.add_argument(
        "--log",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )  # TODO: move to defaults
    parser.add_argument("--verbose", action="store_true", help="Enable verbose console output")
    args = parser.parse_args()

    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
