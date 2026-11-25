import argparse, hashlib, pathlib, random, re, shutil, sys

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


def natural(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def split_files(files, val_frac, block, seed):
    blocks = [files[i:i + block] for i in range(0, len(files), block)]
    n_val = int(val_frac * len(blocks) + 0.5)
    val_idx = set(random.Random(seed).sample(range(len(blocks)), min(n_val, len(blocks))))
    train = [f for i, b in enumerate(blocks) if i not in val_idx for f in b]
    val = [f for i, b in enumerate(blocks) if i in val_idx for f in b]
    return train, val


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder of images (subfolders included)")
    ap.add_argument("--out", required=True, help="creates OUT\\train and OUT\\val")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--block", type=int, default=8, help="consecutive pictures that always stay together")
    ap.add_argument("--order", choices=["time", "name"], default="time",
                    help="time = file modification time (capture time for screenshots); name = natural name order")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--move", action="store_true", help="move the files instead of copying them")
    ap.add_argument("--force", action="store_true", help="allow a non-empty OUT\\train or OUT\\val")
    a = ap.parse_args()

    src, out = pathlib.Path(a.src), pathlib.Path(a.out)
    files = [f for f in src.rglob("*") if f.is_file() and f.suffix.lower() in EXTS]
    if not files:
        sys.exit(f"no images in {src}")
    files.sort(key=(lambda f: (f.stat().st_mtime, natural(f.name))) if a.order == "time" else (lambda f: natural(str(f.relative_to(src)))))

    seen, unique, dupes = {}, [], 0
    for f in files:
        h = hashlib.sha1(f.read_bytes()).hexdigest()
        if h in seen:
            dupes += 1
            continue
        seen[h] = f
        unique.append(f)
    train, val = split_files(unique, a.val_frac, max(1, a.block), a.seed)

    for sub in ("train", "val"):
        d = out / sub
        if d.exists() and any(d.iterdir()) and not a.force:
            sys.exit(f"{d} is not empty; pick another --out or pass --force")
        d.mkdir(parents=True, exist_ok=True)
    op = shutil.move if a.move else shutil.copy2
    for sub, group in (("train", train), ("val", val)):
        for f in group:
            dest = out / sub / f.name
            k = 1
            while dest.exists():
                dest = out / sub / f"{f.stem}_{k}{f.suffix}"
                k += 1
            op(str(f), str(dest))
    print(f"{len(files)} images found, {dupes} exact duplicates skipped, {len(unique)} used")
    print(f"train {len(train)} -> {out / 'train'}")
    print(f"val   {len(val)} -> {out / 'val'}   (whole blocks of {a.block}, {len(val) * 100 // max(1, len(unique))}% of the pictures)")


if __name__ == "__main__":
    main()
