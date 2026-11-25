import numpy as np
import pytest
from skimage import data
from ambi_core import CodecConfig, rgb_to_ycc, ycc_to_rgb
from ambi_core.bitstream import (encode_image, decode_image, read_info, pad_to_tiles, tile_grid,
                                 parse_header)
from ambi_core.errors import AmbiFormatError
from ambi_core.oracle import solve_tile, reconstruct_tile
from ambi_core.profile import default_profile, Profile
from ambi_core.entropy import EntropyModel


@pytest.fixture(scope="module")
def prof():
    return default_profile(CodecConfig(tile=16, max_block=16, aspect_cap=4.0))


def expected_recon(prof, rgb, lam, mu=0.0):
    """What the encoder believes the decoder will output (float32 search path)."""
    cfg = prof.cfg
    H, W = rgb.shape[:2]
    ycc = pad_to_tiles(rgb_to_ycc(rgb), cfg.tile)
    ty, tx = tile_grid(H, W, cfg.tile)
    canvas = np.zeros_like(ycc)
    T = cfg.tile
    for j in range(ty):
        for i in range(tx):
            t = np.ascontiguousarray(ycc[:, j*T:(j+1)*T, i*T:(i+1)*T])
            res = solve_tile(cfg, prof.model, t, lam, mu)
            rc, _ = reconstruct_tile(cfg, prof.model, t, res.leaves)
            canvas[:, j*T:(j+1)*T, i*T:(i+1)*T] = rc
    return ycc_to_rgb(canvas[:, :H, :W])


@pytest.mark.parametrize("shape", [(16, 16), (50, 61), (33, 17), (1, 1), (7, 40)])
def test_roundtrip_matches_encoder_expectation(prof, shape):
    img = data.astronaut()[200:200 + shape[0], 150:150 + shape[1]]
    r = encode_image(img, prof, 0.03)
    out = decode_image(r.data, prof, deblock=False)
    assert out.shape == img.shape and out.dtype == np.uint8
    exp = expected_recon(prof, img, 0.03)
    diff = np.abs(out.astype(int) - exp.astype(int))
    assert diff.max() <= 1
    assert (diff > 0).mean() < 2e-3


def test_real_size_matches_estimate(prof):
    img = data.astronaut()[100:164, 200:264]
    r = encode_image(img, prof, 0.03)
    header = read_info(r.data)["header_bytes"]
    assert 0 <= (r.nbytes - header) - r.est_bits / 8 <= 8 + 0.002 * r.est_bits / 8


def test_deterministic(prof):
    img = data.astronaut()[100:140, 100:140]
    a = encode_image(img, prof, 0.05).data
    b = encode_image(img, prof, 0.05).data
    assert a == b
    assert np.array_equal(decode_image(a, prof), decode_image(a, prof))


def test_flat_image_is_tiny_and_exact(prof):
    img = np.full((32, 48, 3), 120, dtype=np.uint8)
    r = encode_image(img, prof, 0.05)
    assert r.nbytes < 80
    assert np.abs(decode_image(r.data, prof).astype(int) - 120).max() <= 1


def test_target_lambda_trades_size_for_quality(prof):
    img = data.astronaut()[100:148, 200:248]
    small = encode_image(img, prof, 0.3)
    big = encode_image(img, prof, 0.003)
    assert small.nbytes < big.nbytes


def test_corrupt_inputs_raise_cleanly(prof):
    img = data.astronaut()[100:132, 100:132]
    good = encode_image(img, prof, 0.03).data
    with pytest.raises(AmbiFormatError):
        decode_image(b"NOPE" + good[4:], prof)
    with pytest.raises(AmbiFormatError):
        decode_image(good[:8], prof)
    with pytest.raises(AmbiFormatError):
        decode_image(good[:len(good) // 2], prof)
    other = Profile("other", 0, prof.cfg, EntropyModel(prof.cfg))
    counts = other.model.new_counts(); counts["cbf"][0, 0, 0] += np.array([1, 1000]); other.model.refit(counts)
    with pytest.raises(AmbiFormatError):
        decode_image(good, other)
    bad_version = bytearray(good); bad_version[4] = 99
    with pytest.raises(AmbiFormatError):
        decode_image(bytes(bad_version), prof)
    rng = np.random.default_rng(0)
    for _ in range(20):
        junk = bytearray(good)
        for k in rng.integers(13, len(junk), 5):
            junk[k] = int(rng.integers(0, 256))
        try:
            decode_image(bytes(junk), prof)
        except AmbiFormatError:
            pass


def test_profile_save_load_roundtrip(tmp_path, prof):
    prof.save(tmp_path / "p.ambiprofile")
    p2 = Profile.load(tmp_path / "p.ambiprofile")
    assert p2.fingerprint == prof.fingerprint and p2.cfg == prof.cfg


def test_fingerprint_ignores_encoder_only_knobs():
    a = default_profile(CodecConfig(tile=16, max_block=16))
    b = default_profile(CodecConfig(tile=16, max_block=16, alpha=0.9, mse_scale=10.0, rounding_offset=0.3))
    assert a.fingerprint == b.fingerprint
    c = default_profile(CodecConfig(tile=16, max_block=16, qsteps=(4.0, 10.0, 24.0, 60.0)))
    assert a.fingerprint != c.fingerprint
