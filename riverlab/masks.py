"""Explicit source label decoding without mutating the original annotations."""
import numpy as np
from PIL import Image
from .io import read_json, resolve


def decode_mask(image, mapping=None):
    array = np.asarray(image)
    if not mapping:
        if array.ndim != 2 or not set(np.unique(array)) <= {0, 1, 255}:
            raise ValueError("Expected indexed 0/1/255 mask; RGB masks require an explicit mask_mapping")
        return array.astype(np.uint8)
    if array.ndim == 2:
        entries = [(int(key), int(value)) for key, value in mapping.get("indexed", {}).items()]
    elif array.ndim == 3 and array.shape[2] == 3:
        entries = [(tuple(int(x) for x in key.split(",")), int(value))
                   for key, value in mapping.get("rgb", {}).items()]
    else:
        raise ValueError("Unsupported mask shape: {}".format(array.shape))
    output = np.full(array.shape[:2], 255, dtype=np.uint8)
    known = np.zeros(array.shape[:2], dtype=bool)
    for key, value in entries:
        if value not in {0, 1, 255}:
            raise ValueError("Mapping output must be 0, 1 or 255")
        match = (array == key).all(2) if array.ndim == 3 else array == key
        output[match], known[match] = value, True
    if not known.all():
        colors = np.unique(array[~known], axis=0).tolist()
        raise ValueError("Unmapped annotation values: {}".format(colors[:10]))
    return output


def read_mask(row, predicted=False):
    path = row["prediction_path"] if predicted else row["mask_path"]
    mapping_path = row.get("mask_mapping") if not predicted else None
    mapping = read_json(mapping_path) if mapping_path else None
    with Image.open(resolve(path)) as image:
        return decode_mask(image, mapping)
