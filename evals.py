import numpy as np
from PIL import Image

from pathlib import Path
import argparse

def getImages(input, output):
    with Image.open(input) as im:
        with Image.open(output) as out:
            a, b = np.asarray(im.convert("YCbCr")).astype(np.float64), np.asarray(out if out.mode == "L" else out.convert("YCbCr")).astype(np.float64)
            if b.ndim == 2:
                a = a[:, :, 0]
            return a, b

def snr(input, output):
    signal = np.mean(input**2)
    noise = np.mean((input - output)**2)
    return 10 * np.log10(signal / noise)

def psnr(input, output):
    rmse = np.sqrt(np.mean((input - output) ** 2))
    if rmse == 0:
        return float("inf")
    return 20 * np.log10(255 / rmse)

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    return parser.parse_args()

if __name__ == "__main__":
    args = parseArgs()
    input, output = getImages(Path(args.input), Path(args.output))
    print(f"PSNR: {psnr(input, output):.3f}")
    # print(f"SNR: {snr(input, output):.3f}")

    if output.ndim == 3:
        channels = ("Y", "Cb", "Cr")
        for i, channel in enumerate(channels):
            print(f"PSNR {channel}: {psnr(input[:, :, i], output[:, :, i]):.3f}")
