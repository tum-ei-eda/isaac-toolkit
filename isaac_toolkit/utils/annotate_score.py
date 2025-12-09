import sys
import argparse

import yaml


def annotate_score(
    index,
    output=None,
    inplace=False,
    filtered=False,
    filtered2=False,
    prelim=False,
    final=False,
    util_weight=1.0,
    enc_weight=1.0,
):

    with open(index, "r") as f:
        combined_index_data = yaml.safe_load(f)
    candidates = combined_index_data["candidates"]
    # print("candidates", candidates)

    score_name = "score"

    if filtered:
        assert not (final or filtered2 or prelim)
        score_name = f"filtered_{score_name}"
    if filtered2:
        assert not (final or filtered or prelim)
        score_name = f"filtered2_{score_name}"
    if prelim:
        assert not (final or filtered2 or filtered)
        score_name = f"prelim_{score_name}"
    if final:
        assert not (prelim or filtered2 or filtered)
        score_name = f"final_{score_name}"

    for i, candidate in enumerate(candidates):
        # print("i", i)
        # name = candidate["properties"]["InstrName"]
        metrics = candidate.get("metrics", {})
        # print("metrics", metrics)
        # input("!")
        # runtime_reduction_rel = metrics["runtime_reduction_rel"]
        # code_size_reduction_rel = metrics["code_size_reduction_rel"]
        benefits = []
        costs = []
        util_score = metrics.get("util_score")
        if util_score is not None:
            assert util_score >= 0
            util_score = util_score * util_weight
            benefits.append(util_score)

        enc_weight = metrics.get("enc_weight")
        if enc_weight is not None:
            assert enc_weight >= 0
            enc_weight = enc_weight * enc_weight
            costs.append(enc_weight)

        assert len(benefits) > 0
        benefits_sum = sum(benefits)

        assert len(costs) > 0
        costs_sum = sum(costs)

        score = benefits_sum / costs_sum
        metrics[score_name] = score
        # print("metrics2", metrics)
        candidate["metrics"] = metrics

    if inplace:
        assert output is None
        out_file = index
    else:
        assert output is not None
        out_file = output

    combined_index_data["candidates"] = candidates

    with open(out_file, "w") as f:
        yaml.dump(combined_index_data, f)


def handle(args):
    annotate_score(
        args.index,
        output=args.output,
        inplace=args.inplace,
        filtered=args.filtered,
        filtered2=args.filtered2,
        prelim=args.prelim,
        final=args.final,
        util_weight=args.util_weight,
        env_weight=args.enc_weight,
    )


def get_parser():
    parser = argparse.ArgumentParser(description="TODO")
    parser.add_argument("index", help="Index yaml file")
    parser.add_argument("-o", "--output", default=None, help="Output yaml file")
    parser.add_argument("--inplace", action="store_true", help="TODO")
    parser.add_argument("--filtered", action="store_true", help="TODO")
    parser.add_argument("--filtered2", action="store_true", help="TODO")
    parser.add_argument("--prelim", action="store_true", help="TODO")
    parser.add_argument("--final", action="store_true", help="TODO")
    # parser.add_argument("--runtime-weight", type=float, default=1.0, help="TODO")
    # parser.add_argument("--code-size-weight", type=float, default=1.0, help="TODO")
    parser.add_argument("--util-weight", type=float, default=1.0, help="TODO")
    parser.add_argument("--enc-weight", type=float, default=1.0, help="TODO")
    parser.add_argument(
        "--log",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )  # TODO: move to defaults
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
