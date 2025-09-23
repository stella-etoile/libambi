import argparse
from pathlib import Path
from ambi.codec import decode_image

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--workers", type=int, default=None, help="decode workers (overrides env)")
    p.add_argument("--chunk", type=int, default=None, help="blocks per task (overrides default)")
    args = p.parse_args()
    decode_image(Path(args.input), Path(args.output),
                 num_workers=args.workers, chunk_size=args.chunk)

if __name__ == "__main__":
    main()