import numpy as np
from ambi.features import block_features

def test_block_features_shape():
    x = np.random.rand(16,16,3).astype(np.float32)
    f = block_features(x)
    assert f.shape[0] == 18
    assert np.isfinite(f).all()