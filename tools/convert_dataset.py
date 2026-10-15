"""Convert the Barbara et al. EOG dataset from MATLAB v5 .mat into a flat binary.

The C pipeline cannot read .mat files, and the dataset ships in MATLAB's v5
format. This reads that format directly using only the standard library --
struct for the record headers and zlib for the compressed elements -- so the
conversion needs no scipy, numpy or MATLAB install.

Each subject folder yields one .eog file laid out as:

    offset  type         field
    0       char[8]      "EOGDATA1"
    8       uint32       n_samples
    12      float32      sample_rate_hz
    16      float32[n]   horizontal EOG
    ...     float32[n]   vertical EOG
    ...     uint8[n]     control signal

The control signal labels each sample 0 for none, 1 for a forward saccade,
2 for a return saccade and 3 for a blink, as the dataset's description defines
them. It travels with the signal so the C side can score its output against
ground truth without a second file to keep in step.

Usage:
    python convert_dataset.py <dataset_dir> <output_dir>

where <dataset_dir> holds the S1..S6 subject folders.
"""

import os
import struct
import sys
import zlib

# MAT-file data types that appear in this dataset
MI_INT8, MI_UINT8, MI_INT16, MI_UINT16 = 1, 2, 3, 4
MI_INT32, MI_UINT32, MI_SINGLE, MI_DOUBLE = 5, 6, 7, 9
MI_MATRIX, MI_COMPRESSED = 14, 15

# struct format and width for each stored type
NUMERIC = {
    MI_INT8:   ("b", 1),
    MI_UINT8:  ("B", 1),
    MI_INT16:  ("h", 2),
    MI_UINT16: ("H", 2),
    MI_INT32:  ("i", 4),
    MI_UINT32: ("I", 4),
    MI_SINGLE: ("f", 4),
    MI_DOUBLE: ("d", 8),
}

SAMPLE_RATE_HZ = 256.0   # stated in the dataset description

MAGIC = b"EOGDATA1"


def read_tag(buf, off):
    """Return (data_type, num_bytes, header_size) for the element at `off`.

    MAT-files use a compact form for elements of at most four bytes, where the
    byte count is packed into the high half of the first word instead of
    occupying a word of its own.
    """
    word, = struct.unpack_from("<I", buf, off)
    if word >> 16:
        return word & 0xFFFF, word >> 16, 4
    dtype, nbytes = struct.unpack_from("<II", buf, off)
    return dtype, nbytes, 8


def element_size(nbytes, header_size):
    """Total bytes an element occupies, including its tag and padding."""
    if header_size == 4:
        return 8                       # compact form is always exactly 8 bytes
    return 8 + ((nbytes + 7) // 8) * 8  # payload padded to an 8-byte boundary


def read_numeric(buf, off):
    """Read one numeric element, returning (values, bytes_consumed)."""
    dtype, nbytes, hs = read_tag(buf, off)
    if dtype not in NUMERIC:
        raise ValueError("unsupported element type %d" % dtype)
    fmt, width = NUMERIC[dtype]
    count = nbytes // width
    values = struct.unpack_from("<%d%s" % (count, fmt), buf, off + hs)
    return values, element_size(nbytes, hs)


def parse_matrix(buf):
    """Parse a miMATRIX body into (name, dims, values).

    Only the real numeric arrays this dataset contains are handled; complex,
    sparse, cell and struct arrays are rejected rather than mis-read.
    """
    off = 0

    # Array flags: class in the low byte, complex/global/logical as bit flags
    dtype, nbytes, hs = read_tag(buf, off)
    flags = struct.unpack_from("<II", buf, off + hs)
    mx_class = flags[0] & 0xFF
    is_complex = bool(flags[0] & 0x0800)
    off += element_size(nbytes, hs)
    if is_complex:
        raise ValueError("complex arrays are not supported")
    if mx_class in (1, 2, 3, 5):       # cell, struct, object, sparse
        raise ValueError("array class %d is not supported" % mx_class)

    dims, consumed = read_numeric(buf, off)
    off += consumed

    dtype, nbytes, hs = read_tag(buf, off)
    name = buf[off + hs: off + hs + nbytes].decode("ascii", "replace")
    off += element_size(nbytes, hs)

    values, _ = read_numeric(buf, off)
    return name, tuple(dims), values


def load_mat(path):
    """Return {variable_name: (dims, values)} for a MATLAB v5 file."""
    with open(path, "rb") as handle:
        data = handle.read()

    if not data[:6] == b"MATLAB":
        raise ValueError("%s is not a MATLAB file" % path)
    if data[:19].startswith(b"MATLAB 7.3"):
        raise ValueError("%s is v7.3 (HDF5), which this reader does not handle"
                         % path)

    out = {}
    off = 128                          # fixed-size file header
    while off + 8 <= len(data):
        dtype, nbytes, hs = read_tag(data, off)
        body = data[off + hs: off + hs + nbytes]
        if dtype == MI_COMPRESSED:
            inner = zlib.decompress(body)
            idtype, inbytes, ihs = read_tag(inner, 0)
            if idtype == MI_MATRIX:
                name, dims, values = parse_matrix(inner[ihs:ihs + inbytes])
                out[name] = (dims, values)
        elif dtype == MI_MATRIX:
            name, dims, values = parse_matrix(body)
            out[name] = (dims, values)
        off += element_size(nbytes, hs)
    return out


def column_major(values, rows, cols, row):
    """Extract one row from a MATLAB array, which stores columns consecutively."""
    return [values[col * rows + row] for col in range(cols)]


def convert_subject(subject_dir, out_path):
    """Convert one SX folder, returning a summary dict."""
    eog = load_mat(os.path.join(subject_dir, "EOG.mat"))
    ctrl = load_mat(os.path.join(subject_dir, "ControlSignal.mat"))

    eog_dims, eog_values = eog["EOG"]
    rows, cols = eog_dims[0], eog_dims[1]
    if rows != 2:
        raise ValueError("expected 2 EOG rows, found %d" % rows)

    horizontal = column_major(eog_values, rows, cols, 0)
    vertical = column_major(eog_values, rows, cols, 1)

    ctrl_dims, ctrl_values = ctrl["ControlSignal"]
    if len(ctrl_values) != cols:
        raise ValueError("control signal length %d does not match %d samples"
                         % (len(ctrl_values), cols))
    control = [int(v) & 0xFF for v in ctrl_values]

    with open(out_path, "wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<I", cols))
        handle.write(struct.pack("<f", SAMPLE_RATE_HZ))
        handle.write(struct.pack("<%df" % cols, *horizontal))
        handle.write(struct.pack("<%df" % cols, *vertical))
        handle.write(struct.pack("<%dB" % cols, *control))

    counts = {label: control.count(label) for label in (0, 1, 2, 3)}
    return {
        "samples": cols,
        "seconds": cols / SAMPLE_RATE_HZ,
        "counts": counts,
        "h_range": (min(horizontal), max(horizontal)),
        "v_range": (min(vertical), max(vertical)),
    }


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2

    dataset_dir, out_dir = argv[1], argv[2]
    os.makedirs(out_dir, exist_ok=True)

    subjects = sorted(
        name for name in os.listdir(dataset_dir)
        if name.startswith("S") and
        os.path.isdir(os.path.join(dataset_dir, name))
    )
    if not subjects:
        print("no SX subject folders found in %s" % dataset_dir)
        return 1

    print("Converting %d subjects from %s" % (len(subjects), dataset_dir))
    print()
    print("  subject   samples    seconds   forward   return    blink")
    print("  " + "-" * 56)

    for subject in subjects:
        out_path = os.path.join(out_dir, subject + ".eog")
        info = convert_subject(os.path.join(dataset_dir, subject), out_path)
        counts = info["counts"]
        print("  %-8s %8d  %9.1f  %8d %8d %8d" % (
            subject, info["samples"], info["seconds"],
            counts[1], counts[2], counts[3]))

    print()
    print("Wrote .eog files to %s" % out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
