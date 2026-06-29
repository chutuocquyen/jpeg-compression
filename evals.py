import numpy as np
import imageio.v3 as io

from pathlib import Path
import argparse

from decoder import decodeInput
from encoder import RGB2YCbCr

def getImages(input, output):
    a = io.imread(input).astype(np.float64)
    b, precision = decodeInput(output)

    if b.ndim == 2:
        a = RGB2YCbCr(a, precision)[0]
        if precision != 8:
            a *= ((1 << precision) - 1) / 255
        return a, b, precision

    if precision != 8:
        a *= ((1 << precision) - 1) / 255
    a = np.stack(RGB2YCbCr(a, precision), axis=2)
    b = np.stack(RGB2YCbCr(b, precision), axis=2)
    return a, b, precision

def snr(input, output):
    signal = np.mean(input**2)
    noise = np.mean((input - output)**2)
    return 10 * np.log10(signal / noise)

def psnr(input, output, precision):
    rmse = np.sqrt(np.mean((input - output) ** 2))
    if rmse == 0:
        return float("inf")
    return 20 * np.log10(((1 << precision) - 1) / rmse)

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    return parser.parse_args()

if __name__ == "__main__":
    args = parseArgs()
    input, output, precision = getImages(Path(args.input), Path(args.output))
    print(f"PSNR: {psnr(input, output, precision):.3f}")
    # print(f"SNR: {snr(input, output):.3f}")

    if output.ndim == 3:
        channels = ("Y", "Cb", "Cr")
        for i, channel in enumerate(channels):
            print(f"PSNR {channel}: {psnr(input[:, :, i], output[:, :, i], precision):.3f}")
