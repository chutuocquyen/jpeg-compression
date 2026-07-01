from pathlib import Path
import imageio.v3 as io

from encoder import writeOutput
import argparse

MODES = (1,)
PRECISIONS = (8,)
SUBSAMPLINGS = ("420",)
QUALITIES = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
OUT_DIR = Path("tmp")

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    return parser.parse_args()

def main():
    args = parseArgs()

    input = Path(args.input)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    im = io.imread(input)

    for mode in MODES:
        for precision in PRECISIONS:
            for subsampling in SUBSAMPLINGS:
                for adaptive in (True,):
                    for quality in QUALITIES:
                        if precision == 12 and not adaptive:
                            continue
                        if mode == 0 and precision != 8:
                            continue

                        out = OUT_DIR / f"m{mode}_q{quality}_{subsampling}_{'adaptive' if adaptive else 'default'}_{precision}.jpg"

                        writeOutput(
                            im,
                            out,
                            quality=quality,
                            subsampling=subsampling,
                            adaptive=adaptive,
                            mode=mode,
                            precision=precision
                        )

if __name__ == "__main__":
    main()
