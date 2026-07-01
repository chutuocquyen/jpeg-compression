import numpy as np
import imageio.v3 as io

from pathlib import Path
import argparse

from decoder import decodeInput
from encoder import RGB2YCbCr

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("QtAgg")

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

def getImages(input, output):
    a = io.imread(input).astype(np.float64)
    b, precision = decodeInput(output)

    if b.ndim == 2:
        a = np.stack(RGB2YCbCr(a, precision), axis=2)

        cb = np.full_like(b, 1 << (precision - 1))
        cr = np.full_like(b, 1 << (precision - 1))
        b = np.stack([b, cb, cr], axis=2)

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

def _psnr(input, output, precision):
    rmse = np.sqrt(np.mean((input - output) ** 2))
    if rmse == 0:
        return float("inf")
    return float(20 * np.log10(((1 << precision) - 1) / rmse))

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    # parser.add_argument("output")
    return parser.parse_args()

def main():
    args = parseArgs()
    input, output, precision = getImages(Path(args.input), Path(args.output))
    print(f"PSNR: {_psnr(input, output, precision):.3f}")
    # print(f"SNR: {snr(input, output):.3f}")

    if output.ndim == 3:
        channels = ("Y", "Cb", "Cr")
        for i, channel in enumerate(channels):
            print(f"PSNR {channel}: {_psnr(input[:, :, i], output[:, :, i], precision):.3f}")

def psnr(output):
    args = parseArgs()
    input, output, precision = getImages(Path(args.input), Path(output))
    # print(f"PSNR: {psnr(input, output, precision):.3f}")
    # print(f"SNR: {snr(input, output):.3f}")

    # if output.ndim == 3:
    #     channels = ("Y", "Cb", "Cr")
    #     for i, channel in enumerate(channels):
    #         print(f"PSNR {channel}: {psnr(input[:, :, i], output[:, :, i], precision):.3f}")
    return _psnr(input, output, precision)

if __name__ == "__main__":
    OUTPUTS = ("res/m1_q10_444_adaptive_8.jpg",
               "res/m1_q20_444_adaptive_8.jpg",
               "res/m1_q30_444_adaptive_8.jpg",
               "res/m1_q40_444_adaptive_8.jpg",
               "res/m1_q50_444_adaptive_8.jpg",
               "res/m1_q60_444_adaptive_8.jpg",
               "res/m1_q70_444_adaptive_8.jpg",
               "res/m1_q80_444_adaptive_8.jpg",
               "res/m1_q90_444_adaptive_8.jpg",
               "res/m1_q100_444_adaptive_8.jpg",
               "res/m1_q10_422_adaptive_8.jpg",
               "res/m1_q20_422_adaptive_8.jpg",
               "res/m1_q30_422_adaptive_8.jpg",
               "res/m1_q40_422_adaptive_8.jpg",
               "res/m1_q50_422_adaptive_8.jpg",
               "res/m1_q60_422_adaptive_8.jpg",
               "res/m1_q70_422_adaptive_8.jpg",
               "res/m1_q80_422_adaptive_8.jpg",
               "res/m1_q90_422_adaptive_8.jpg",
               "res/m1_q100_422_adaptive_8.jpg",
               "res/m1_q10_420_adaptive_8.jpg",
               "res/m1_q20_420_adaptive_8.jpg",
               "res/m1_q30_420_adaptive_8.jpg",
               "res/m1_q40_420_adaptive_8.jpg",
               "res/m1_q50_420_adaptive_8.jpg",
               "res/m1_q60_420_adaptive_8.jpg",
               "res/m1_q70_420_adaptive_8.jpg",
               "res/m1_q80_420_adaptive_8.jpg",
               "res/m1_q90_420_adaptive_8.jpg",
               "res/m1_q100_420_adaptive_8.jpg",
               "res/m1_q10_gray_adaptive_8.jpg",
               "res/m1_q20_gray_adaptive_8.jpg",
               "res/m1_q30_gray_adaptive_8.jpg",
               "res/m1_q40_gray_adaptive_8.jpg",
               "res/m1_q50_gray_adaptive_8.jpg",
               "res/m1_q60_gray_adaptive_8.jpg",
               "res/m1_q70_gray_adaptive_8.jpg",
               "res/m1_q80_gray_adaptive_8.jpg",
               "res/m1_q90_gray_adaptive_8.jpg",
               "res/m1_q100_gray_adaptive_8.jpg")

    plt.figure(dpi=200)
    LABELS = (r"4:4:4", r"4:2:2", r"4:2:0", r"Grayscale")
    for i in range(len(OUTPUTS) // 10):
        tmp = OUTPUTS[10 * i:10 * i + 10]
        y = []
        for a in tmp:
            y.append(psnr(a))

        x = np.linspace(10, 100, 10)
        # print(y)
        plt.plot(x, y, 'o-', linewidth=1, label=LABELS[i])
        plt.legend(loc="upper left")

    plt.ylim(20, 65)
    plt.xlabel(r"Compression quality")
    plt.ylabel(r"PSNR (dB)")
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig("res/subsampling_comparison.png")
