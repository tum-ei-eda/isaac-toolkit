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
import sys
import logging
import argparse
import tempfile
import subprocess
from pathlib import Path
from math import log10, ceil
from typing import List, Optional, Union
from collections import defaultdict

import yaml
import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import filter_artifacts, ArtifactFlag, TableArtifact

logging.basicConfig(level=logging.DEBUG)  # TODO
logger = logging.getLogger(__name__)


def export_helper(df, dest_dir: Path, base_filename: str, chunk_rows: int = 10000, **kwargs):

    chunks = [df[i : i + chunk_rows].copy() for i in range(0, df.shape[0], chunk_rows)]
    num_chunks = len(chunks)
    # print("num_chunks", num_chunks)

    fillcnt = ceil(log10(num_chunks + 1)) + 1
    # print("fillcnt", fillcnt)

    for k, chunk in enumerate(chunks):
        out_filename = base_filename + "_" + str(k).zfill(fillcnt) + ".csv"
        out_path = dest_dir / out_filename
        chunk.to_csv(out_path, index=False, **kwargs)


def export_instr_trace(instr_trace_df, asm_trace_dir, chunk_rows: int = 10000):
    logger.info("Exporting ASM trace to %s...", asm_trace_dir)
    df = instr_trace_df[["pc", "instr"]].copy()
    df.rename(columns={"instr": "assembly"}, inplace=True)
    export_helper(df, asm_trace_dir, "asm_trace", chunk_rows=chunk_rows, sep=";", lineterminator=";\n")


def export_timing_trace(timing_trace_df, timing_trace_dir, uarch: str, chunk_rows: int = 10000):
    logger.info("Exporting timing trace to %s...", timing_trace_dir)
    base_filename = f"{uarch}_timing"
    export_helper(timing_trace_df, timing_trace_dir, base_filename, chunk_rows=chunk_rows)


def run_cmd(args, cwd=None, verbose=False):
    cmd_str = " ".join(args)
    logger.debug("> %s", cmd_str)

    try:
        if verbose:
            subprocess.run(
                args,
                check=True,
                cwd=cwd,
            )
        else:
            subprocess.run(
                args,
                check=True,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
    except subprocess.CalledProcessError as e:
        logger.error("Command failed with exit code %d: `%s`", e.returncode, cmd_str)
        if e.stdout:
            logger.error("Output:")
            logger.error(e.stdout.strip())
        sys.exit(e.returncode)


def flatten_metrics(data, parent_key="", sep="_"):
    flattened = {}

    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict):
            flattened.update(flatten_metrics(value, new_key, sep))
        else:
            flattened[new_key] = value

    return flattened


def run_trace_analyzer(
    sess: Session,
    output: Optional[Union[str, Path]] = None,
    ranges_yaml: Optional[Union[str, Path]] = None,
    uarch: str = "CV32E40P",
    force: bool = False,
    verbose: bool = False,
    views: bool = False,
    use_pkl: bool = False,
    gen_dfs: bool = True,
):
    artifacts = sess.artifacts

    instr_trace_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.INSTR_TRACE)
    assert len(instr_trace_artifacts) == 1
    instr_trace_artifact = instr_trace_artifacts[0]
    instr_trace_df = instr_trace_artifact.df

    timing_trace_artifacts = filter_artifacts(artifacts, lambda x: x.attrs.get("kind") == "timing_trace")
    assert len(timing_trace_artifacts) == 1
    timing_trace_artifact = timing_trace_artifacts[0]
    timing_trace_df = timing_trace_artifact.df

    assert ranges_yaml is not None
    ranges_yaml = Path(ranges_yaml)
    assert ranges_yaml.is_file()
    ranges_yaml = ranges_yaml.resolve()
    assert uarch is not None

    if output is None:
        output_dir = sess.directory / "output"
        output_dir.mkdir(exist_ok=True)
        trace_analyzer_output_dir = output_dir / "trace_analyzer"
        trace_analyzer_output_dir.mkdir(exist_ok=True)
    else:
        trace_analyzer_output_dir = Path(output)
    assert Path(trace_analyzer_output_dir).is_dir()
    logger.info("Creating output directory %s", trace_analyzer_output_dir)
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_dir = Path(tmpdirname)
        if use_pkl:
            # TODO: use compression?
            # TODO: make sure columns are as expected?
            asm_trace_dir = temp_dir / "asm_trace.pkl"
            instr_trace_df_ = instr_trace_df[["pc", "instr"]].copy()
            instr_trace_df_.rename(columns={"instr": "assembly"}, inplace=True)
            instr_trace_df_.to_pickle(asm_trace_dir)
            timing_trace_dir = temp_dir / "timing_trace.pkl"
            timing_trace_df.to_pickle(timing_trace_dir)
        else:
            asm_trace_dir = temp_dir / "asm_trace"
            asm_trace_dir.mkdir()
            export_instr_trace(instr_trace_df, asm_trace_dir)
            timing_trace_dir = temp_dir / "timing_trace"
            timing_trace_dir.mkdir()
            export_timing_trace(timing_trace_df, timing_trace_dir, uarch)
        # print("temp_dir", temp_dir)
        logger.info("Using temporary directory %s", temp_dir)
        model_name = "isaacModel"
        subprocess_kwargs = {}
        # TODO: error handling
        if not verbose:
            subprocess_kwargs["stdout"] = subprocess.DEVNULL
            subprocess_kwargs["stderr"] = subprocess.DEVNULL
        common_args = [
            "python3",
            "-m",
            "trace_analyzer.run",  # TODO: PYTHONPATH
        ]
        import_asm_trace_args = [
            *common_args,
            "import",
            model_name,
            f"-i={asm_trace_dir}",
            "-delim=;",
        ]
        # print("import_asm_trace_args", " ".join(import_asm_trace_args))
        logger.info("Importing ASM trace into %s model...", model_name)
        # subprocess.run(import_asm_trace_args, check=True, cwd=trace_analyzer_output_dir, **subprocess_kwargs)
        run_cmd(import_asm_trace_args, cwd=trace_analyzer_output_dir, verbose=verbose)
        extend_timing_trace_args = [
            *common_args,
            "load",
            model_name,
            "extend",
            f"-pt={timing_trace_dir}",
        ]
        # print("extend_timing_trace_args", " ".join(extend_timing_trace_args))
        logger.info("Extending %s model with timing trace...", model_name)
        run_cmd(extend_timing_trace_args, cwd=trace_analyzer_output_dir, verbose=verbose)
        extend_uarch_args = [
            *common_args,
            "load",
            model_name,
            "extend",
            f"-uarch={uarch}",
        ]
        # print("extend_uarch_args", " ".join(extend_uarch_args))
        logger.info("Extending %s model with %s uArch...", model_name, uarch)
        run_cmd(extend_uarch_args, cwd=trace_analyzer_output_dir, verbose=verbose)
        analyze_args = [
            *common_args,
            "load",
            model_name,
            "analyze",
            f"-r={ranges_yaml}",
            "-t",
            "-o",
            str(trace_analyzer_output_dir),
            "-y",
            str(trace_analyzer_output_dir),
        ]
        # print("analyze_args", " ".join(analyze_args))
        logger.info("Analyzing %s model... (Output directory: %s)", model_name, trace_analyzer_output_dir)
        run_cmd(analyze_args, cwd=trace_analyzer_output_dir, verbose=verbose)
        # TODO: parse metrics?
        gen_viewer_args = [
            *common_args,
            "load",
            model_name,
            "pipeline_viewer",
            f"-r={ranges_yaml}",
            "-o",
            str(trace_analyzer_output_dir),
        ]
        if gen_dfs:
            with open(ranges_yaml, "r") as f:
                ranges_data = yaml.safe_load(f)
            # print("ranges_data", ranges_data)
            ranges = []
            for i, entry in enumerate(ranges_data["ranges"]):
                range_name = entry[0]
                new = (i, range_name)
                ranges.append(new)
            # print("ranges", ranges)
            analysis_rows = defaultdict(list)
            for range_idx, range_name in ranges:
                fname = f"{model_name}_{range_name}_analysis.yaml"
                analysis_yaml_file = trace_analyzer_output_dir / fname
                assert analysis_yaml_file.is_file()
                with open(analysis_yaml_file, "r") as f:
                    analysis_data = yaml.safe_load(f)
                # print("analysis_data", analysis_data)
                for key, metrics in analysis_data.items():
                    # print("key", key)
                    if key == "metadata":
                        continue
                    flattened_metrics = flatten_metrics(metrics)
                    # print("flattened_metrics", flattened_metrics)
                    # TODO: add ranges df in addition to yaml?
                    row = {"range_name": range_name, **flattened_metrics}
                    analysis_rows[key].append(row)
            # print("analysis_rows", analysis_rows)
            for key, rows in analysis_rows.items():
                analysis_df = pd.DataFrame(rows)
                analysis_attrs = {
                    "instr_trace": instr_trace_artifact.name,
                    "timing_trace": timing_trace_artifact.name,
                    "kind": "analysis",
                    "group": key,
                    "by": __name__,
                }
                analysis_artifact = TableArtifact(f"analysis_{key}", analysis_df, attrs=analysis_attrs)
                sess.add_artifact(analysis_artifact, override=force)

        if views:
            # print("gen_viewer_args", " ".join(gen_viewer_args))
            logger.info(
                "Generating kanata files for %s model... (Output directory: %s)", model_name, trace_analyzer_output_dir
            )
            run_cmd(gen_viewer_args, cwd=trace_analyzer_output_dir, verbose=verbose)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    run_trace_analyzer(
        sess,
        output=args.output,
        force=args.force,
        verbose=args.verbose,
        ranges_yaml=args.ranges_yaml,
        uarch=args.uarch,
        views=args.views,
        use_pkl=args.use_pkl,
        gen_dfs=args.gen_dfs,
    )
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--views", action="store_true")
    parser.add_argument("--use-pkl", action="store_true")
    parser.add_argument("--gen-dfs", action="store_true")
    parser.add_argument("--ranges-yaml", type=str, required=True)
    parser.add_argument("--uarch", type=str, default="CV32E40P")
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
