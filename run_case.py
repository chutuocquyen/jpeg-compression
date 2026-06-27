from pathlib import Path
from PIL import Image

from encoder import writeOutput
import argparse

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    return parser.parse_args()

def main():
    args = parseArgs()

    input = Path(args.input)
    out_dir = Path("res")
    out_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(input) as im:
        arr = im.convert("RGB")

        for adaptive in (False, True):
            for subsampling in ("444", "422", "420", "gray"):
                out = out_dir / f"q50_{subsampling}_{'adaptive' if adaptive else 'default'}.jpg"

                writeOutput(
                    arr,
                    out,
                    quality=50,
                    subsampling=subsampling,
                    adaptive=adaptive
                )

if __name__ == "__main__":
    main()
