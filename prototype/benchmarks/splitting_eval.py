import argparse
from collections import Counter, defaultdict
from pathlib import Path

from src.deepscores.dataset import DeepScoresDataset
from src.symbols.evaluation import evaluate_split
from src.symbols.splitter import PrimitiveSplitter


DATA_DIR = Path(__file__).parent.parent / "data" / "ds2_dense"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--unknown-limit", type=int, default=100)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    args = parser.parse_args()

    dataset = DeepScoresDataset(DATA_DIR, split=args.split)
    splitter = PrimitiveSplitter()
    statuses = Counter()
    totals = defaultdict(lambda: defaultdict(float))
    count = 0

    for sample in dataset.iter_supported_crops(limit=args.limit):
        result = splitter.split(
            sample.staffless_mask,
            sample.staff_space,
            sample.staff_line,
            sample.original_mask,
        )
        statuses[str(result.status)] += 1
        for kind, score in evaluate_split(
            result, sample.truth_masks, sample.staffless_mask
        ).items():
            for field in ("precision", "recall", "f1", "iou"):
                totals[kind][field] += getattr(score, field)
        count += 1

    if count == 0:
        print("no supported crops found")
        return

    print(f"crops: {count}")
    print("status:", dict(statuses))
    print(f"{'class':>10} {'precision':>10} {'recall':>10} {'f1':>10} {'iou':>10}")
    for kind in ("notehead", "stem", "beam"):
        values = totals[kind]
        print(
            f"{kind:>10} {values['precision'] / count:>10.4f} "
            f"{values['recall'] / count:>10.4f} {values['f1'] / count:>10.4f} "
            f"{values['iou'] / count:>10.4f}"
        )

    unknown_statuses = Counter()
    unknown_total = defaultdict(float)
    unknown_count = 0
    for sample in dataset.iter_unknown_crops(limit=args.unknown_limit):
        result = splitter.split(
            sample.staffless_mask,
            sample.staff_space,
            sample.staff_line,
            sample.original_mask,
        )
        unknown_statuses[str(result.status)] += 1
        score = evaluate_split(result, sample.truth_masks, sample.staffless_mask)[
            "unknown"
        ]
        for field in ("precision", "recall", "f1", "iou"):
            unknown_total[field] += getattr(score, field)
        unknown_count += 1

    if unknown_count:
        print(f"unknown crops: {unknown_count}")
        print("unknown status:", dict(unknown_statuses))
        print(
            f"{'unknown':>10} {unknown_total['precision'] / unknown_count:>10.4f} "
            f"{unknown_total['recall'] / unknown_count:>10.4f} "
            f"{unknown_total['f1'] / unknown_count:>10.4f} "
            f"{unknown_total['iou'] / unknown_count:>10.4f}"
        )


if __name__ == "__main__":
    main()
