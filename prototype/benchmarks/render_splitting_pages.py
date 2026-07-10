import argparse
from pathlib import Path

from src.deepscores.dataset import DeepScoresDataset
from src.symbols.visualization import render_page_diagnostics
from src.symbols.splitter import PrimitiveSplitter


DATA_DIR = Path(__file__).parent.parent / "data" / "ds2_dense"
DEFAULT_OUTPUT = Path(__file__).parent.parent / "output" / "splitting_pages"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    dataset = DeepScoresDataset(DATA_DIR)
    splitter = PrimitiveSplitter()
    written = []
    page_number = 0
    for image_id in dataset.images:
        if page_number >= args.pages:
            break
        samples = dataset.page_compounds(image_id)
        if not samples:
            continue
        page_number += 1
        original, staffless, metadata = dataset.load_page(image_id)
        results = [
            (
                sample,
                splitter.split(
                    sample.staffless_mask,
                    sample.staff_space,
                    sample.staff_line,
                    sample.original_mask,
                ),
            )
            for sample in samples
        ]
        filename = f"page_{page_number:03d}.png"
        title = metadata["filename"]
        render_page_diagnostics(
            original, staffless, results, args.output / filename, title
        )
        written.append(args.output / filename)

    print(f"wrote {len(written)} full-page panels to {args.output}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
