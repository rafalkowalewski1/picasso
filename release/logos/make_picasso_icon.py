"""Build the Picasso app icon (picasso.icns, picasso.ico, picasso.png)
from the docs logo, for the one-click installers.

The line-art logo sits on a white rounded tile (macOS Big Sur grid:
824/1024 tile, ~22% corner radius) so it stays visible on dark docks and
taskbars. Strokes are thickened per output size so they never drop below
~1.4 px.

Usage (macOS, needs ``iconutil``), from this folder::

    python make_picasso_icon.py ../../docs/_static/picasso-logo.png .
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

SS = 4  # supersampling for the tile edge


def thicken(alpha, radius):
    """Dilate the stroke mask by ``radius`` source px, antialiased."""
    if radius <= 0:
        return alpha
    mask = alpha > 127
    dist = ndimage.distance_transform_edt(~mask)
    grown = np.clip(radius + 0.5 - dist, 0, 1) * 255
    return np.maximum(alpha, grown).astype(np.uint8)


def render(logo_alpha, stroke_src, size):
    tile = 824 / 1024 * size
    radius = 0.2237 * tile
    off = (size - tile) / 2
    # logo fills 88% of the tile width
    h0, w0 = logo_alpha.shape
    w = 0.88 * tile
    h = w * h0 / w0
    # target stroke in output px
    native = stroke_src * w / w0
    target = max(native * 1.6, 1.4)
    r_src = (target * w0 / w - stroke_src) / 2
    alpha = thicken(logo_alpha, r_src)
    glyph = Image.fromarray(alpha).resize((round(w), round(h)), Image.LANCZOS)

    big = size * SS
    tile_mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(tile_mask).rounded_rectangle(
        [off * SS, off * SS, (off + tile) * SS - 1, (off + tile) * SS - 1],
        radius=radius * SS,
        fill=255,
    )
    tile_mask = tile_mask.resize((size, size), Image.LANCZOS)

    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    white = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    img.paste(white, (0, 0), tile_mask)
    black = Image.new("RGBA", glyph.size, (0, 0, 0, 255))
    img.paste(
        black,
        (round((size - w) / 2), round((size - h) / 2)),
        glyph,
    )
    return img


def main(logo_path, out_dir):
    logo = Image.open(logo_path).convert("RGBA")
    a = np.array(logo.getchannel("A"))
    # crop to content
    ys, xs = np.nonzero(a > 0)
    a = a[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    # stroke width ~ 2x the largest inscribed radius along the line
    edt = ndimage.distance_transform_edt(a > 127)
    stroke_src = 2 * np.percentile(edt[edt > 0], 99)
    print(f"source stroke ~{stroke_src:.1f} px")

    sizes = [16, 24, 32, 48, 64, 128, 256, 512, 1024]
    imgs = {s: render(a, stroke_src, s) for s in sizes}
    imgs[1024].save(os.path.join(out_dir, "picasso.png"))

    with tempfile.TemporaryDirectory() as tmp:
        iconset = os.path.join(tmp, "picasso.iconset")
        os.mkdir(iconset)
        for s in [16, 32, 128, 256, 512]:
            imgs[s].save(os.path.join(iconset, f"icon_{s}x{s}.png"))
            imgs[2 * s].save(os.path.join(iconset, f"icon_{s}x{s}@2x.png"))
        subprocess.run(
            [
                "iconutil",
                "-c",
                "icns",
                iconset,
                "-o",
                os.path.join(out_dir, "picasso.icns"),
            ],
            check=True,
        )

    ico_sizes = [16, 24, 32, 48, 64, 128, 256]
    imgs[256].save(
        os.path.join(out_dir, "picasso.ico"),
        format="ICO",
        sizes=[(s, s) for s in ico_sizes],
        append_images=[imgs[s] for s in ico_sizes[:-1]],
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
