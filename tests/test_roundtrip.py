import numpy as np
from pathlib import Path
from ambi.codec import encode_image, decode_image
from ambi.io import save_image_rgb, load_image_rgb

def test_roundtrip(tmp_path: Path):
    h, w = 64, 96
    img = np.random.rand(h, w, 3).astype(np.float32)
    src = tmp_path / "in.png"
    outa = tmp_path / "out.ambi"
    outp = tmp_path / "rec.png"
    save_image_rgb(src, img)
    encode_image(src, outa, {"encoder":{"block_size":32,"default_q":12,"default_K":5,"default_H":8},"policy":{"algorithm":"fixed_margin","score_margin_thresh":0.06},"prior":{"type":"deterministic","K":5},"format":{"magic":"AMBI","version":1}})
    decode_image(outa, outp)
    rec = load_image_rgb(outp)
    assert rec.shape == img.shape