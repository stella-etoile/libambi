import argparse
import yaml
from pathlib import Path
from ambi.codec import encode_image
from ambi.utils import set_seed

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--config", default=None)
    args = p.parse_args()
    cfg = {}
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed", 1004))
    encode_image(Path(args.input), Path(args.output), cfg)

if __name__ == "__main__":
    main()