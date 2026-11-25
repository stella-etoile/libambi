import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ambi_core import CodecConfig, EntropyModel
from ambi_core.profile import Profile

ap = argparse.ArgumentParser()
ap.add_argument("model_npz"); ap.add_argument("out")
ap.add_argument("--name", default="photo-demo"); ap.add_argument("--id", type=int, default=1)
ap.add_argument("--mse-scale", type=float, default=353.43)
a = ap.parse_args()
cfg = CodecConfig(mse_scale=a.mse_scale)
Profile(a.name, a.id, cfg, EntropyModel.load(cfg, a.model_npz)).save(a.out)
print("wrote", a.out)
