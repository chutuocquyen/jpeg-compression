import numpy as np

from typing import Union, Dict, Tuple, List
from pathlib import Path
from dataclasses import dataclass
import struct
import argparse

import matplotlib
import matplotlib.pyplot as plt
from typing_extensions import Optional
matplotlib.use("QtAgg")

from encoder import ZIGZAG, DCT, DCT_T

@dataclass
class Component:
    cid: int
    name: str
    v: int
    h: int
    qid: int
    dcid: int = 0
    acid: int = 0

@dataclass
class Scan:
    components: List[Component]
    Ss: int
    Se: int
    Ah: int
    Al: int
    data: bytes

@dataclass
class ParsedImage:
    height: int
    width: int
    components: List[Component]
    qtables: Dict[int, np.ndarray]
    htables: Dict[Tuple[int, int], Dict[Tuple[int, int], int]]
    scans: List[Scan]

class BitReader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.acc = 0
        self.nbits = 0

    def read(self, n: int):
        value = 0
        for _ in range(n):
            value = (value << 1) | self.readBit()
        return value

    def readBit(self):
        if self.nbits == 0:
            self.acc = self.data[self.pos]
            self.pos += 1
            if self.acc == 0xFF:
                if self.data[self.pos] == 0x00:
                    self.pos += 1   # Byte destuffing
                else: raise ValueError("Unexpected marker")
            self.nbits = 8

        self.nbits -= 1
        return (self.acc >> self.nbits) & 1

def as_u16(data: bytes, i: int) -> int:
    return struct.unpack(">H", data[i:i + 2])[0]

def inverseDCT(zigzag: np.ndarray, qtable: np.ndarray) -> np.ndarray:
    zz = np.zeros(64, dtype=np.float64)
    zz[ZIGZAG] = zigzag

    coeff = zz.reshape(8, 8)
    inv = DCT_T @ (coeff * qtable) @ DCT + 128
    return np.clip(inv, 0, 255)

def readSegment(data: bytes, i: int) -> Tuple[bytes, int]:
    length = as_u16(data, i)
    return data[i + 2:i + length], i + length

def readMarker(data: bytes, i: int) -> Tuple[int, int]:
    while i < len(data) and data[i] != 0xFF: i += 1
    if i >= len(data):
        raise ValueError("Marker not found")
    while i < len(data) and data[i] == 0xFF: i += 1
    if i >= len(data):
        raise ValueError("Marker symbol not found")
    return data[i], i + 1

def readScanData(data: bytes, pos: int):
    i0 = pos

    while pos < len(data):
        if data[pos] != 0xFF:
            pos += 1
            continue

        if data[pos + 1] == 0x00:
            pos += 2
            continue

        if data[pos + 1] in (0xC4, 0xD9, 0xDA):
            return data[i0:pos], pos

        raise ValueError("Unexpected marker")
    raise ValueError("Missing EOI marker")

# Parser
def parseImage(source: Union[str, Path, bytes]) -> ParsedImage:
    if isinstance(source, (str, Path)):
        data = Path(source).read_bytes()
    else:
        data = source

    if len(data) < 2 or data[:2] != b"\xFF\xD8":
        raise ValueError("Missing SOI marker")

    height, width = None, None
    components = None
    qtables, htables = {}, {}
    scans = []

    pos = 2
    while pos < len(data):
        marker, pos = readMarker(data, pos)
        if marker == 0xD9:      # EOI
            break

        payload, pos = readSegment(data, pos)

        if marker == 0xDA:      # SOS
            if components is None:
                raise ValueError("SOS before SOF")
            scan_components, Ss, Se, Ah, Al = parseSOS(payload, components)
            scan_data, pos = readScanData(data, pos)
            scans.append(Scan(scan_components, Ss, Se, Ah, Al, scan_data))

        elif marker == 0xDB:    # DQT
            parseDQT(payload, qtables)
        elif marker == 0xC4:    # DHT
            parseHuffmanTables(payload, htables)
        elif marker == 0xC0 or marker == 0xC2:
            height, width, components = parseSOF(payload)

    if None in (height, width, components):
        raise ValueError("Missing SOF0")
    if not scans:
        raise ValueError("Missing scan data")

    return ParsedImage(height, width, components, qtables, htables, scans)

def parseDQT(payload, qtables) -> None:
    pos = 0
    while pos < len(payload):
        # precision = payload[pos] >> 4
        qid = payload[pos] & 0x0F
        pos += 1

        zz = np.zeros(64, dtype=np.float64)
        zz[ZIGZAG] = np.asarray(bytearray(payload[pos:pos + 64]), dtype=np.uint8).astype(np.float64)
        qtables[qid] = zz.reshape(8, 8)
        pos += 64

def parseHuffmanTables(payload, htables) -> None:
    pos = 0
    while pos < len(payload):
        table_class = payload[pos] >> 4
        table_id = payload[pos] & 0x0F
        if table_class not in (0, 1):
            raise ValueError("Invalid table class")
        pos += 1

        bits = list(payload[pos:pos + 16])
        count = sum(bits)
        pos += 16

        values = list(payload[pos:pos + count])
        pos += count

        htables[(table_class, table_id)] = _buildDecodeMap(bits, values)

def _buildDecodeMap(bits, values):
    map = {}
    code = 0
    pos = 0

    for length in range(1, 17):
        for _ in range(bits[length - 1]):
            map[(code, length)] = int(values[pos])
            code += 1
            pos += 1
        code <<= 1
    return map

def parseSOF(payload: bytes) -> Tuple[int, int, List[Component]]:
    height, width = as_u16(payload, 1), as_u16(payload, 3)
    Nf = payload[5]

    pos = 6
    components = []
    for _ in range(Nf):
        cid = payload[pos]
        size = payload[pos + 1]
        qid = payload[pos + 2]

        h, v = size >> 4, size & 0x0F
        name = {1: "Y", 2: "Cb", 3: "Cr"}.get(cid)
        components.append(Component(cid, name, v, h, qid))

        pos += 3

    return height, width, components

def parseSOS(payload: bytes, components: List[Component]) -> Tuple[List[Component], int, int, int, int]:
    Ns = payload[0]
    if Ns > 4:
        raise ValueError("Invalid number of image components")

    pos = 1
    scan_components = []
    for _ in range(Ns):
        cid = payload[pos]
        pos += 1

        c = components[cid - 1]
        if c.cid != cid:
            raise ValueError("Unmatched component ID")
        c.dcid = payload[pos] >> 4
        c.acid = payload[pos] & 0x0F
        scan_components.append(c)
        pos += 1

    Ss, Se = payload[pos], payload[pos + 1]
    Ah, Al = payload[pos + 2] >> 4, payload[pos + 2] & 0x0F

    return scan_components, Ss, Se, Ah, Al

# Decoder
def decodePlanes(im: ParsedImage) -> Dict[str, np.ndarray]:
    max_v = max(c.v for c in im.components)
    max_h = max(c.h for c in im.components)

    mcu_h, mcu_w = max_v * 8, max_h * 8
    mcu_rows = (im.height + mcu_h - 1) // mcu_h
    mcu_cols = (im.width + mcu_w - 1) // mcu_w

    coeffs = {}
    for c in im.components:
        coeffs[c.name] = np.zeros((mcu_rows * c.v, mcu_cols * c.h, 64), dtype=np.float64)

    for scan in im.scans:
        br = BitReader(scan.data)

        if scan.Ss == 0:
            pred = {c.cid: 0 for c in scan.components}
            for my in range(mcu_rows):
                for mx in range(mcu_cols):
                    for c in scan.components:
                        dc_table = im.htables[(0, c.dcid)]
                        ac_table = im.htables[(1, c.acid)] if scan.Se > 0 else None

                        for vy in range(c.v):
                            for hx in range(c.h):
                                by = my * c.v + vy
                                bx = mx * c.h + hx

                                coeffs[c.name][by, bx], pred[c.cid] = decodeBlock(br, dc_table, ac_table, pred[c.cid], scan.Ss, scan.Se)

        else:
            c = scan.components[0]
            ac_table = im.htables[(1, c.acid)]
            nblocks_y = (im.height * c.v + 8 * max_v - 1) // (8 * max_v)
            nblocks_x = (im.width  * c.h + 8 * max_h - 1) // (8 * max_h)
            for by in range(nblocks_y):
                for bx in range(nblocks_x):
                    coeffs[c.name][by, bx][scan.Ss:scan.Se + 1] = decodeBlock(br, None, ac_table, None, scan.Ss, scan.Se)[scan.Ss:scan.Se + 1]

    planes = {}
    for c in im.components:
        plane = np.zeros((mcu_rows * c.v * 8, mcu_cols * c.h * 8), dtype=np.float64)
        for y in range(mcu_rows * c.v):
            for x in range(mcu_cols * c.h):
                y0, x0 = y * 8, x * 8
                plane[y0:y0 + 8, x0:x0 + 8] = inverseDCT(coeffs[c.name][y, x], im.qtables[c.qid])
        planes[c.name] = plane

    return planes

def decodeBlock(br: BitReader,
                dc_table,
                ac_table,
                pred: Optional[int],
                Ss: int = 0,
                Se: int = 63):
    block = np.zeros(64, dtype=np.int32)

    if Ss == 0:
        ssss = readCodeword(br, dc_table)
        diff = getCoeff(br, ssss)
        dc = pred + diff
        block[0] = dc
        if Se == 0:
            return block, dc

    i = max(1, Ss)
    while i < Se + 1:
        run_size = readCodeword(br, ac_table)

        if run_size == 0x00:    # EOB
            break
        if run_size == 0xF0:
            i += 16
            continue

        run = run_size >> 4
        size = run_size & 0x0F

        i += run
        block[i] = getCoeff(br, size)
        i += 1

    if Ss == 0:
        return block, dc
    return block

def readCodeword(br: BitReader, table) -> int:
    code = 0
    for length in range(1, 17):
        code = (code << 1) | br.readBit()

        symbol = table.get((code, length))
        if symbol is not None:
            return symbol
    raise ValueError("Invalid code word")

def getCoeff(br: BitReader, size: int) -> int:
    if size == 0:
        return 0

    diff = br.read(size)

    if diff < (1 << (size - 1)):
        diff -= (1 << size) - 1
    return diff

def upsample(plane: np.ndarray,
             component: Component,
             max_v: int,
             max_h: int,
             target_h: int,
             target_w: int) -> np.ndarray:
    out = np.repeat(np.repeat(plane, max_v // component.v, axis=0), max_h // component.h, axis=1)

    return out[:target_h, :target_w]

def YCbCr2RGB(y, cb, cr) -> np.ndarray:
    """
    YCbCr to RGB conversion
    Four decimal position accuracy
    [T.871]
    """
    cb -= 128
    cr -= 128

    r = y + 1.402 * cr
    g = y - 0.3441 * cb - 0.7141 * cr
    b = y + 1.772 * cb

    rgb = np.stack([r, g, b], axis=2)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)

def decodeInput(source: Union[str, Path, bytes]) -> np.ndarray:
    im = parseImage(source)
    planes = decodePlanes(im)

    max_v = max(c.v for c in im.components)
    max_h = max(c.h for c in im.components)

    y = planes["Y"]

    if len(im.components) == 1:
        return np.rint(y[:im.height, :im.width]).astype(np.uint8)

    y = y[:im.height, :im.width]
    cb = upsample(planes["Cb"],
                  im.components[1],
                  max_v, max_h,
                  im.height, im.width)
    cr = upsample(planes["Cr"],
                  im.components[2],
                  max_v, max_h,
                  im.height, im.width)

    return YCbCr2RGB(y, cb, cr)

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    return parser.parse_args()

if __name__ == "__main__":
    args = parseArgs()
    im = decodeInput(args.input)

    fig = plt.figure(dpi=200)

    if im.ndim == 2:
        plt.imshow(im, cmap="gray", vmin=0, vmax=255)
    else:
        plt.imshow(im)

    plt.axis("off")
    plt.tight_layout()
    plt.show()
