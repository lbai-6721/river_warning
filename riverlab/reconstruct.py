"""Reassemble existing row/column patch IDs into the original analysis ROI."""
from collections import defaultdict

import numpy as np
from PIL import Image

from .io import portable, resolve, write_csv, write_json
from .masks import read_mask


def reconstruct(rows, output, columns=5, nrows=2, layout="yx"):
    if layout not in {"xy", "yx"}:
        raise ValueError("Patch layout must be xy or yx")
    def suffix(x, y):
        return "{}{}".format(x, y) if layout == "xy" else "{}{}".format(y, x)
    groups = defaultdict(list)
    for row in rows:
        groups[row["parent_id"]].append(row)
    directory = resolve(output)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "JPEGImages").mkdir()
    (directory / "SegmentationClass").mkdir()
    result = []
    for i, (pid, patches) in enumerate(sorted(groups.items())):
        expected = {suffix(x, y) for x in range(columns) for y in range(nrows)}
        lookup = {r["sample_id"].rsplit("_", 1)[-1]: r for r in patches}
        if set(lookup) != expected or len(patches) != len(expected):
            raise ValueError("Incomplete/ambiguous patch grid for {}".format(pid))
        if len({r.get("split", "") for r in patches}) != 1:
            raise ValueError("Parent has patches crossing splits")
        image_rows, mask_rows = [], []
        for y in range(nrows):
            images, masks = [], []
            for x in range(columns):
                r = lookup[suffix(x, y)]
                with Image.open(resolve(r["image_path"])) as im:
                    images.append(np.array(im.convert("RGB")))
                masks.append(read_mask(r))
            image_rows.append(np.concatenate(images, axis=1))
            mask_rows.append(np.concatenate(masks, axis=1))
        image, mask = np.concatenate(image_rows), np.concatenate(mask_rows)
        # PNG avoids another JPEG encoding. This is the cropped analysis ROI,
        # not a reconstruction of discarded top/bottom pixels.
        ip = directory / "JPEGImages" / "{:06d}.png".format(i)
        mp = directory / "SegmentationClass" / "{:06d}.png".format(i)
        Image.fromarray(image).save(ip)
        Image.fromarray(mask).save(mp)
        row = dict(patches[0], sample_id=pid, image_path=portable(ip), mask_path=portable(mp),
                   width=image.shape[1], height=image.shape[0], image_sha256="", mask_mapping="")
        result.append(row)
    write_csv(directory / "manifest.csv", result)
    write_json(directory / "manifest.meta.json", {
        "source": "reconstructed_roi", "grid": [columns, nrows], "layout": layout,
        "note": "Source patches may already contain resizing/JPEG loss; discarded original pixels are not recovered"})
    return {"images": len(result), "manifest": portable(directory / "manifest.csv")}
