import numpy as np
import imageio.v3 as io

from typing import Optional, Union, Literal, Dict, Tuple, List
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass
import math
import struct
import sys
import argparse

Subsampling = Literal["gray", "444", "422", "420"]

LUMA_QUANT_TABLE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
    ])
CHROMA_QUANT_TABLE = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
    ])
ZIGZAG = np.array([
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
    ])

LUMA_DC_BITS = [0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01,     # [T.81] - Table K.3
                0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
CHROMA_DC_BITS = [0x00, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,   # [T.81] - Table K.4
                  0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]
DC_VAL = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B]

LUMA_AC_BITS = [0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03,     # [T.81] - Table K.5
                0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D]
LUMA_AC_VAL = [0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,0x13,0x51,0x61,0x07,
               0x22,0x71,0x14,0x32,0x81,0x91,0xA1,0x08,0x23,0x42,0xB1,0xC1,0x15,0x52,0xD1,0xF0,
               0x24,0x33,0x62,0x72,0x82,0x09,0x0A,0x16,0x17,0x18,0x19,0x1A,0x25,0x26,0x27,0x28,
               0x29,0x2A,0x34,0x35,0x36,0x37,0x38,0x39,0x3A,0x43,0x44,0x45,0x46,0x47,0x48,0x49,
               0x4A,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5A,0x63,0x64,0x65,0x66,0x67,0x68,0x69,
               0x6A,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7A,0x83,0x84,0x85,0x86,0x87,0x88,0x89,
               0x8A,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9A,0xA2,0xA3,0xA4,0xA5,0xA6,0xA7,
               0xA8,0xA9,0xAA,0xB2,0xB3,0xB4,0xB5,0xB6,0xB7,0xB8,0xB9,0xBA,0xC2,0xC3,0xC4,0xC5,
               0xC6,0xC7,0xC8,0xC9,0xCA,0xD2,0xD3,0xD4,0xD5,0xD6,0xD7,0xD8,0xD9,0xDA,0xE1,0xE2,
               0xE3,0xE4,0xE5,0xE6,0xE7,0xE8,0xE9,0xEA,0xF1,0xF2,0xF3,0xF4,0xF5,0xF6,0xF7,0xF8,
               0xF9,0xFA]
CHROMA_AC_BITS = [0x00, 0x02, 0x01, 0x02, 0x04, 0x04, 0x03, 0x04,   # [T.81] - Table K.6
                  0x07, 0x05, 0x04, 0x04, 0x00, 0x01, 0x02, 0x77]
CHROMA_AC_VAL = [0x00,0x01,0x02,0x03,0x11,0x04,0x05,0x21,0x31,0x06,0x12,0x41,0x51,0x07,0x61,0x71,
                 0x13,0x22,0x32,0x81,0x08,0x14,0x42,0x91,0xA1,0xB1,0xC1,0x09,0x23,0x33,0x52,0xF0,
                 0x15,0x62,0x72,0xD1,0x0A,0x16,0x24,0x34,0xE1,0x25,0xF1,0x17,0x18,0x19,0x1A,0x26,
                 0x27,0x28,0x29,0x2A,0x35,0x36,0x37,0x38,0x39,0x3A,0x43,0x44,0x45,0x46,0x47,0x48,
                 0x49,0x4A,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5A,0x63,0x64,0x65,0x66,0x67,0x68,
                 0x69,0x6A,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7A,0x82,0x83,0x84,0x85,0x86,0x87,
                 0x88,0x89,0x8A,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9A,0xA2,0xA3,0xA4,0xA5,
                 0xA6,0xA7,0xA8,0xA9,0xAA,0xB2,0xB3,0xB4,0xB5,0xB6,0xB7,0xB8,0xB9,0xBA,0xC2,0xC3,
                 0xC4,0xC5,0xC6,0xC7,0xC8,0xC9,0xCA,0xD2,0xD3,0xD4,0xD5,0xD6,0xD7,0xD8,0xD9,0xDA,
                 0xE2,0xE3,0xE4,0xE5,0xE6,0xE7,0xE8,0xE9,0xEA,0xF2,0xF3,0xF4,0xF5,0xF6,0xF7,0xF8,
                 0xF9,0xFA]

@dataclass(frozen=True)
class Component:
    cid: int
    name: str
    v: int      # Sampling factor
    h: int
    qid: int    # 0 - Luminance, 1 - Chrominance
    dcid: int
    acid: int

@dataclass
class PreprocessedImage:
    height: int
    width: int
    subsampling: Subsampling
    components: List[Component]
    planes: Dict[str, np.ndarray]
    qtables: Dict[int, np.ndarray]
    max_v: int
    max_h: int
    mcu_h: int
    mcu_w: int
    precision: int

class BitWriter:
    def __init__(self):
        self.out = bytearray()
        self.acc = 0
        self.nbits = 0

    def write(self, code: int, length: int):
        if length < 0 or length > 16:
            raise ValueError("Invalid bit length")
        if length == 0: return

        if code < 0 or code >= (1 << length):
            raise ValueError("Invalid code word")

        for pos in range(length - 1, -1, -1):
            self.acc = (self.acc << 1) | ((code >> pos) & 1)
            self.nbits += 1
            if self.nbits == 8:
                self.out.append(self.acc)
                if self.acc == 0xFF: self.out.append(0x00)      # Byte stuffing
                self.acc = 0
                self.nbits = 0

    def flush(self):
        if self.nbits:
            pad = 8 - self.nbits
            self.acc = (self.acc << pad) | ((1 << pad) - 1)
            self.out.append(self.acc)
            if self.acc == 0xFF: self.out.append(0x00)
            self.acc = 0
            self.nbits = 0
        return bytes(self.out)

# Process image
def normalizeImage(im, precision):
    arr = np.asarray(im)

    if arr.ndim == 2:
        pass
    elif arr.ndim == 3 and arr.shape[2] >= 3:   # RGBA
        arr = arr[:, :, :3]

    if np.issubdtype(arr.dtype, np.floating):
        arr = arr * ((1 << precision) - 1) if np.max(arr) <= 1 else arr
    elif precision == 12 and arr.dtype == np.uint8:
        arr = arr.astype(np.float64) * 4095 / 255
    return np.clip(np.rint(arr), 0, (1 << precision) - 1).astype(np.uint8 if precision == 8 else np.uint16)

def RGB2YCbCr(rgb: np.ndarray, precision: int = 8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RGB to YCbCr conversion
    Four decimal position accuracy
    [T.871]
    """
    x = rgb.astype(np.float64)
    r, g, b = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + (1 << (precision - 1))
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + (1 << (precision - 1))
    return y, cb, cr

def getTargetSize(n: int, m: int) -> int:
    return -(-n // m) * m

def padEdge(plane, target_h, target_w):
    h, w = plane.shape
    if target_h < h or target_w < w:
        raise ValueError("Invalid target size")
    return np.pad(plane, ((0, target_h - h), (0, target_w - w)), mode="edge")

def downsample(plane, factor_y, factor_x):
    if factor_y == 1 and factor_x == 1: return plane.copy()

    h, w = plane.shape
    if h % factor_y or w % factor_x:
        raise ValueError("Invalid downsampling factor")
    return plane.reshape(h // factor_y, factor_y, w // factor_x, factor_x).mean(axis=(1, 3)) # (y, pool_y, x, pool_x)

def preprocessImage(im,
                    quality: int,
                    subsampling: Subsampling,
                    precision: int) -> PreprocessedImage:
    arr = normalizeImage(im, precision)
    h0, w0 = arr.shape[:2]
    if h0 == 0 or w0 == 0:
        raise ValueError("Invalid image input")
    if h0 > 65535 or w0 > 65535:
        raise ValueError("Invalid image dimensions")

    qtableY = generateQuantTable(LUMA_QUANT_TABLE, quality, precision)
    qtableC = generateQuantTable(CHROMA_QUANT_TABLE, quality, precision)

    if arr.ndim == 2 or subsampling == "gray":
        y = arr if arr.ndim == 2 else RGB2YCbCr(arr, precision)[0]
        target_h, target_w = getTargetSize(h0, 8), getTargetSize(w0, 8)
        components = [Component(1, "Y", 1, 1, 0, 0, 0)]
        planes = {"Y": padEdge(y, target_h, target_w)}
        return PreprocessedImage(h0, w0, "gray", components, planes, {0: qtableY}, 1, 1, 8, 8, precision)

    if subsampling not in ("444", "422", "420"):
        raise ValueError("Invalid subsampling pattern")

    y, cb, cr = RGB2YCbCr(arr, precision)

    if subsampling == "444":
        v_samplingFactor, h_samplingFactor = 1, 1
        ds_y, ds_x = 1, 1
        mcu_h, mcu_w = 8, 8
    elif subsampling == "422":
        v_samplingFactor, h_samplingFactor = 1, 2
        ds_y, ds_x = 1, 2
        mcu_h, mcu_w = 8, 16
    elif subsampling == "420":
        v_samplingFactor, h_samplingFactor = 2, 2
        ds_y, ds_x = 2, 2
        mcu_h, mcu_w = 16, 16

    target_h, target_w = getTargetSize(h0, mcu_h), getTargetSize(w0, mcu_w)
    y_pad = padEdge(y, target_h, target_w)
    cb_pad = padEdge(cb, target_h, target_w)
    cr_pad = padEdge(cr, target_h, target_w)

    cb_ds = downsample(cb_pad, ds_y, ds_x)
    cr_ds = downsample(cr_pad, ds_y, ds_x)

    components = [
            Component(1, "Y", v_samplingFactor, h_samplingFactor, 0, 0, 0),
            Component(2, "Cb", 1, 1, 1, 1, 1),
            Component(3, "Cr", 1, 1, 1, 1, 1)
            ]
    planes = {"Y": y_pad, "Cb": cb_ds, "Cr": cr_ds}
    return PreprocessedImage(h0, w0, subsampling, components, planes, {0: qtableY, 1: qtableC}, v_samplingFactor, h_samplingFactor, mcu_h, mcu_w, precision)

# DCT & Quantization
def generateDCT():
    """
    8x8 orthonormal DCT of Type-II
    """
    t = np.zeros((8, 8), dtype=np.float64)
    for i in range(8):
        a = math.sqrt(1 / 8) if i == 0 else math.sqrt(2 / 8)
        for j in range(8):
            t[i, j] = a * math.cos(((2 * j + 1) * i * math.pi) / 16)
    return t

DCT = generateDCT()
DCT_T = DCT.T

def generateQuantTable(base: np.ndarray, quality: int, precision: int = 8) -> np.ndarray:
    """
    libjpeg
    """
    if quality < 1: quality = 1
    if quality > 100: quality = 100
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    return np.clip(np.rint((base.astype(np.float64) * scale + 50) / 100).astype(np.int32), 1, (1 << precision) - 1)

def quantizeBlock(block, qtable, precision: int = 8):
    shifted = block.astype(np.float64) - (1 << (precision - 1))
    coeff = np.rint(DCT @ shifted @ DCT_T / qtable).astype(np.int32)
    return coeff.ravel()[ZIGZAG]

def buildSymbolMap(bits, values):
    if len(bits) != 16:
        raise ValueError("Invalid count table")
    if sum(bits) != len(values):
        raise ValueError("Mismatched symbol count")

    map = {}    # Dict[int, Tuple[int, int]]
    code = 0
    pos = 0

    for length in range(1, 17):
        for _ in range(bits[length - 1]):
            run_size = int(values[pos])
            map[run_size] = (code, length)
            code += 1
            pos += 1
        code <<= 1
    return map

def countSymbols(image, qtables, precision):
    dc_freq = defaultdict(lambda: defaultdict(int))
    ac_freq = defaultdict(lambda: defaultdict(int))
    pred = {c.name: 0 for c in image.components}

    for c, block in getMCUs(image):
        zigzag = quantizeBlock(block, qtables[c.qid], precision)

        # DC
        dc = int(zigzag[0])
        dc_freq[c.dcid][category(dc - int(pred[c.name]))] += 1
        pred[c.name] = dc

        # AC
        run = 0
        for value in (int(i) for i in zigzag[1:]):
            if value == 0:
                run += 1
                if run == 16:
                    ac_freq[c.acid][0xF0] += 1
                    run = 0
                continue

            size = category(value)
            run_size = (run << 4) | size
            ac_freq[c.acid][run_size] += 1
            run = 0
        if run: ac_freq[c.acid][0] += 1

    return dict(dc_freq), dict(ac_freq)

def buildAdaptiveTables(freqs) -> Dict[Tuple[int, int], Tuple[list, list]]:
    tables = {}
    for dcid, freq in freqs[0].items():
        tables[(0, dcid)] = _buildAdaptiveTable(freq)
    for acid, freq in freqs[1].items():
        tables[(1, acid)] = _buildAdaptiveTable(freq)
    return tables

def _buildAdaptiveTable(freq: dict) -> Tuple[list, list]:
    MAX_SYMBOLS = 257   # 256 + 1
    arr = [0] * MAX_SYMBOLS
    for symbol, count in freq.items():
        if symbol < 0 or symbol > 255: raise ValueError("Invalid symbol")
        arr[symbol] = count
    arr[-1] = 1
    codesize = [0] * MAX_SYMBOLS
    others = [-1] * MAX_SYMBOLS

    # codesize
    while True:
        v1 = v2 = -1
        for i in range(MAX_SYMBOLS):
            if arr[i] == 0: continue
            if v1 == -1 or arr[i] <= arr[v1]:
                v1, v2 = i, v1
            elif v2 == -1 or arr[i] <= arr[v2]:
                v2 = i
        if v2 == -1: break

        arr[v1] += arr[v2]
        arr[v2] = 0

        codesize[v1] += 1
        while others[v1] != -1:
            v1 = others[v1]
            codesize[v1] += 1
        others[v1] = v2

        codesize[v2] += 1
        while others[v2] != -1:
            v2 = others[v2]
            codesize[v2] += 1

    # BITS
    bits = [0] * 33
    for i in range(MAX_SYMBOLS):
        if codesize[i]: bits[codesize[i]] += 1

    # Code length limiting
    i = 32
    while i > 16:
        while bits[i] > 0:
            j = i - 2
            while j > 0 and bits[j] == 0: j -= 1
            if j == 0: raise ValueError("Invalid code length")
            bits[i] -= 2
            bits[i - 1] += 1
            bits[j + 1] += 2
            bits[j] -= 1
        i -= 1

    while bits[i] == 0: i -= 1
    bits[i] -= 1
    bits = bits[1:17]

    # Sorting
    vals = []
    for i in range(1, 33):
        for j in range(MAX_SYMBOLS - 1):
            if codesize[j] == i:
                vals.append(j)

    return bits, vals

# Entropy coding
def signedBits(value: int, size: int) -> int:
    value = int(value)
    if size == 0:
        return 0
    if value >= 0:
        return value
    return (1 << size) - 1 + value

def category(value: int) -> int:
    t = abs(int(value))
    return 0 if t == 0 else t.bit_length()

def getEntropyData(image, lookup,
                   precision: int = 8,
                   Ss: int = 0,
                   Se: int = 63,
                   components: Optional[List[str]] = None):
    if Ss > Se:
        raise ValueError("Invalid spectral selectors")

    if components is None:
        components = [c.name for c in image.components]

    bw = BitWriter()
    if len(components) == 1:
        iter = getMCUs(image, components[0])
    else:
        iter = getMCUs(image)

    if Ss == 0:
        pred = {c: 0 for c in components}
        for c, block in iter:
            pred[c.name] = encodeBlock(block, image.qtables[c.qid], pred[c.name],
                                       c.dcid, c.acid, bw, lookup, precision, Ss, Se)
    else:
        for c, block in iter:
            encodeBlock(block, image.qtables[c.qid], None,
                        None, c.acid, bw, lookup, precision, Ss, Se)
    return bw.flush()

def getMCUs(image, component: Optional[str]=None):
    if component:
        c = next((c for c in image.components if c.name == component), None)
        if c is None:
            raise ValueError("Invalid component")
        nblocks_y = (image.height * c.v - 1) // (8 * image.max_v) + 1
        nblocks_x = (image.width * c.h - 1) // (8 * image.max_h) + 1
        for y in range(nblocks_y):
            for x in range(nblocks_x):
                y0, x0 = y * 8, x * 8
                yield c, image.planes[c.name][y0:y0 + 8, x0:x0 + 8]
    else:
        for my in range(0, image.planes["Y"].shape[0], image.mcu_h):
            for mx in range(0, image.planes["Y"].shape[1], image.mcu_w):
                for c in image.components:
                    y = (my * c.v) // image.max_v
                    x = (mx * c.h) // image.max_h
                    for vy in range(c.v):
                        for hx in range(c.h):
                            y0 = y + vy * 8
                            x0 = x + hx * 8
                            yield c, image.planes[c.name][y0:y0 + 8, x0:x0 + 8]

def encodeBlock(block,
                qtable,
                pred: Optional[int],
                dc_table_id: Optional[int],
                ac_table_id: int,
                bw: BitWriter,
                lookup,
                precision,
                Ss: int,
                Se: int):
    zigzag = quantizeBlock(block, qtable, precision)

    # DC
    if Ss == 0:
        dc_table = lookup[(0, dc_table_id)]
        dc = int(zigzag[0])
        diff = dc - int(pred)
        ssss = category(diff)
        dc_code, dc_length = dc_table[ssss]
        bw.write(dc_code, dc_length)                # Huffman code
        bw.write(signedBits(diff, ssss), ssss)      # DIFF as signed bits

    # AC
    if Se > 0:
        ac_table = lookup[(1, ac_table_id)]
        run = 0     # Zero count
        for value in (int(i) for i in zigzag[max(1, Ss):Se + 1]):
            if value == 0:
                run += 1
                if run == 16:
                    code, length = ac_table[0xF0]
                    bw.write(code, length)
                    run = 0
                continue

            size = category(value)
            run_size = (run << 4) | size
            code, length = ac_table[run_size]
            bw.write(code, length)
            bw.write(signedBits(value, size), size)
            run = 0
        if run:     # EOB
            code, length = ac_table[0x00]
            bw.write(code, length)

    if Ss == 0:
        return dc

# Marker
def u16_BE(x: int) -> bytes:
    return struct.pack(">H", x)

def buildMarker(code: int, payload: bytes = b"") -> bytes:
    prefix = bytes([0xFF, code])
    if code in (0xD8, 0xD9):    # SOI, EOI
        if payload:
            raise ValueError("Payload in SOI/EOI")
        return prefix
    # [T.81] - B.1.1.4 - prefix + 2 byte length parameter + data
    return prefix + u16_BE(len(payload) + 2) + payload

def getAPP0_JFIF():
    payload = (b"JFIF\x00" + bytes([1, 1]) + bytes([0]) + u16_BE(1) + u16_BE(1) + bytes([0, 0]))
    return buildMarker(0xE0, payload)

def getDQT(qtables, precision: int = 8):
    # [T.81] - Table B.4
    payload = bytearray()
    Pq = 0 if precision == 8 else 1
    for i in qtables:
        payload.append((Pq << 4) | (i & 0x0F))    # Lossy
        if Pq == 0:
            for a in qtables[i].ravel()[ZIGZAG]:
                payload.append(int(a) & 0xFF)
        else:
            for a in qtables[i].ravel()[ZIGZAG]:
                payload.extend(u16_BE(int(a) & 0xFFFF))
    return buildMarker(0xDB, bytes(payload))

def getSOF(image: PreprocessedImage, n: int, precision: int = 8):
    # [T.81] - Table B.2
    if n < 0 or n > 15:
        raise ValueError("Invalid frame marker")
    payload = bytearray()
    payload.append(precision)
    payload.extend(u16_BE(image.height))
    payload.extend(u16_BE(image.width))
    payload.append(len(image.components))
    for c in image.components:
        payload.append(c.cid)
        payload.append(((c.h & 0x0F) << 4) | (c.v & 0x0F))
        payload.append(c.qid)
    return buildMarker(0xC0 | n, bytes(payload))

def getHuffmanTables(tables: Dict, color: bool):
    out = bytearray()
    out.extend(_getHuffmanTable(0, 0, *tables[(0, 0)]))
    out.extend(_getHuffmanTable(1, 0, *tables[(1, 0)]))
    if color:
        out.extend(_getHuffmanTable(0, 1, *tables[(0, 1)]))
        out.extend(_getHuffmanTable(1, 1, *tables[(1, 1)]))
    return bytes(out)

def  _getHuffmanTable(c, id, bits, values):
    # [T.81] - Table B.5
    payload = bytearray()
    payload.append(((c & 0x0F) << 4) | (id & 0x0F))
    payload.extend(x & 0xFF for x in bits)
    payload.extend(x & 0xFF for x in values)
    return buildMarker(0xC4, bytes(payload))

def getScanHeader(components: List[Component],
                  Ss: int = 0, Se: int = 63,
                  Ah: int = 0, Al: int = 0):
    # [T.81] - Table B.3
    payload = bytearray()
    payload.append(len(components))
    for c in components:
        payload.append(c.cid)
        if Se == 0:
            payload.append((c.dcid & 0x0F) << 4)
        elif Ss > 0:
            payload.append(c.acid & 0x0F)
        else:
            payload.append(((c.dcid & 0x0F) << 4) | (c.acid & 0x0F))
    payload.extend([Ss, Se, (Ah << 4 | Al)])
    return buildMarker(0xDA, bytes(payload))

# Encoder API
def encodeOutput(im: PreprocessedImage,
                 lookup: Dict,
                 tables: Dict,
                 mode: int,
                 precision: int) -> bytes:
    out = bytearray()
    components = im.components

    out.extend(buildMarker(0xD8))       # SOI - FFD8
    out.extend(getAPP0_JFIF())          # [T.871] - 6.3
    out.extend(getDQT(im.qtables, precision if mode else 8))
    out.extend(getHuffmanTables(tables, color=len(components) == 3))

    if mode == 0:       # Baseline
        out.extend(getSOF(im, 0))
        out.extend(getScanHeader(components))
        out.extend(getEntropyData(im, lookup))

    elif mode == 1:     # Extended
        out.extend(getSOF(im, 1, precision))
        out.extend(getScanHeader(components))
        out.extend(getEntropyData(im, lookup, precision))

    elif mode == 2:     # Progressive
        out.extend(getSOF(im, 2, precision))
        # DC
        out.extend(getScanHeader(components, 0, 0))
        out.extend(getEntropyData(im, lookup, precision, 0, 0))
        # AC
        out.extend(getScanHeader([components[0]], 1, 2))
        out.extend(getEntropyData(im, lookup, precision, 1, 2, [components[0].name]))
        out.extend(getScanHeader([components[0]], 3, 63))
        out.extend(getEntropyData(im, lookup, precision, 3, 63, [components[0].name]))
        for c in components[1:]:
            out.extend(getScanHeader([c], 1, 63))
            out.extend(getEntropyData(im, lookup, precision, 1, 63, [c.name]))
    else:
        raise ValueError("Unsupported operation mode")

    out.extend(buildMarker(0xD9))        # EOI - FFD9
    return bytes(out)

def writeOutput(im,
                output: Union[str, Path],
                quality: int = 50,
                subsampling: Subsampling = "420",
                adaptive: bool = False,
                mode: int = 0,
                precision: int = 8) -> None:
    p = preprocessImage(im, quality, subsampling, precision if mode else 8)

    if adaptive:
        rawTables = buildAdaptiveTables(countSymbols(p, p.qtables, precision))
    else:
        rawTables = {(0, 0): (LUMA_DC_BITS, DC_VAL),    # (coeff, channel)
                     (1, 0): (LUMA_AC_BITS, LUMA_AC_VAL),
                     (0, 1): (CHROMA_DC_BITS, DC_VAL),
                     (1, 1): (CHROMA_AC_BITS, CHROMA_AC_VAL)}
    HUFFMAN_TABLES = {key: buildSymbolMap(bits, val) for key, (bits, val) in rawTables.items()}
    data = encodeOutput(p, HUFFMAN_TABLES, rawTables, mode, precision)
    Path(output).write_bytes(data)

# CLI
def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--quality", "-q", type=int, default=50)
    parser.add_argument("--subsampling", "-s", choices=["gray", "444", "422", "420"], default="420")
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--mode", "-m", type=int, default=0)
    parser.add_argument("--precision", "-p", type=int, default=8)

    return parser.parse_args()

def main():
    args = parseArgs()

    im = io.imread(args.input)
    writeOutput(im, args.output, quality=args.quality, subsampling=args.subsampling, adaptive=args.adaptive, mode=args.mode, precision=args.precision)

if __name__ == "__main__":
    # pass
    sys.exit(not main())
