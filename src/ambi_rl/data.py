from __future__ import annotations

import collections
import hashlib
import pathlib
import time

import numpy as np
import torch

from ambi_core import rgb_to_ycc
from ambi_core.rects import RectSpace


class TileStore:
    def __init__(self, mm, device, cache_tiles=256):
        self.mm, self.device, self.cache_tiles = mm, device, cache_tiles
        self.shape = tuple(mm.shape)
        self._cache = collections.OrderedDict()

    def _tile(self, i):
        t = self._cache.get(i)
        if t is None:
            t = torch.from_numpy(np.array(self.mm[i])).to(self.device)
            self._cache[i] = t
            if len(self._cache) > self.cache_tiles:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(i)
        return t

    def __getitem__(self, key):
        t, l, r = key
        t = torch.as_tensor(t)
        uniq, inv = torch.unique(t.reshape(-1).cpu(), return_inverse=True)
        block = torch.stack([self._tile(int(i)) for i in uniq.tolist()])
        return block[inv.reshape(t.shape).to(self.device), l, r]

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self.mm)


def _stamp(files, fingerprint, limit):
    h = hashlib.sha1(repr((fingerprint, limit)).encode())
    for f in files:
        st = f.stat()
        h.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()


class OracleData:
    def __init__(self, ycc, best, leafc, lams, space: RectSpace, device="cpu", src=None, root_opt=None):
        self.lams = torch.as_tensor(np.asarray(lams), dtype=torch.float64, device=device)
        self.ycc = torch.as_tensor(np.asarray(ycc), dtype=torch.float32, device=device)
        self.device = device
        if isinstance(best, TileStore):
            assert best.shape[2] == space.n_rects and best.shape == leafc.shape and root_opt is not None
            self.best, self.leafc = best, leafc
            self.root_opt = torch.as_tensor(np.asarray(root_opt), dtype=torch.float64, device=device)
        else:
            assert best.shape[2] == space.n_rects and best.shape == leafc.shape
            self.best = torch.as_tensor(np.asarray(best), dtype=torch.float32, device=device)
            self.leafc = torch.as_tensor(np.asarray(leafc), dtype=torch.float32, device=device)
            self.root_opt = self.best[:, :, space.root].double()
        self.src = src or [str(i) for i in range(len(self.ycc))]
        self.n, self.n_lams = self.best.shape[0], self.best.shape[1]

    @classmethod
    def load(cls, directory, space: RectSpace, device="cpu", fingerprint: int | None = None, limit: int | None = None,
             store: str = "auto", gpu_budget_gb: float = 6.0):
        files = sorted(f for f in pathlib.Path(directory).glob("tile_*.npz") if not f.name.endswith(".tmp.npz"))
        if limit:
            files = files[:limit]
        if not files:
            raise FileNotFoundError(f"no tile_*.npz files in {directory}")
        if store == "auto":
            z = np.load(files[0], allow_pickle=False)
            gb = len(files) * (z["best"].nbytes + z["leafc"].nbytes) / 2**30
            store = "disk" if gb > gpu_budget_gb else "gpu"
        if store == "disk":
            return cls._load_disk(files, space, device, fingerprint, limit)
        ycc, best, leafc, src = [], [], [], []
        lams = None
        for f in files:
            z = np.load(f, allow_pickle=False)
            cls._check(f, z, fingerprint, lams)
            lams = z["lams"] if lams is None else lams
            ycc.append(rgb_to_ycc(z["rgb"])); best.append(z["best"]); leafc.append(z["leafc"]); src.append(str(z["src"]))
        return cls(np.stack(ycc), np.stack(best), np.stack(leafc), lams, space, device, src)

    @staticmethod
    def _check(f, z, fingerprint, lams):
        if fingerprint is not None and int(z["fingerprint"]) != fingerprint:
            raise ValueError(f"{f.name} was generated with a different profile (fingerprint mismatch)")
        if lams is not None and not np.array_equal(lams, z["lams"]):
            raise ValueError(f"{f.name} has a different lambda grid")

    @classmethod
    def _load_disk(cls, files, space, device, fingerprint, limit):
        d = files[0].parent
        idx, fb, fl = d / "_index.npz", d / "_cache_best.npy", d / "_cache_leafc.npy"
        stamp = _stamp(files, fingerprint, limit)
        if idx.exists() and fb.exists() and fl.exists():
            with np.load(idx, allow_pickle=False) as z:
                if str(z["stamp"]) == stamp:
                    ycc, root_opt, lams, src = z["ycc"], z["root_opt"], z["lams"], [str(x) for x in z["src"]]
                    return cls(ycc, TileStore(np.load(fb, mmap_mode="r"), device), TileStore(np.load(fl, mmap_mode="r"), device),
                               lams, space, device, src, root_opt)
        z0 = np.load(files[0], allow_pickle=False)
        L, R = z0["best"].shape
        print(f"building the disk cache for {len(files)} tiles in {d} ({len(files) * L * R * 8 / 2**30:.1f} GB; done once) ...", flush=True)
        idx.unlink(missing_ok=True)                                         # an index only ever exists next to a complete cache
        mb = np.lib.format.open_memmap(fb, mode="w+", dtype=np.float32, shape=(len(files), L, R))
        ml = np.lib.format.open_memmap(fl, mode="w+", dtype=np.float32, shape=(len(files), L, R))
        ycc, root_opt, src, lams, t0 = [], [], [], None, time.time()
        for i, f in enumerate(files):
            z = np.load(f, allow_pickle=False)
            cls._check(f, z, fingerprint, lams)
            lams = z["lams"] if lams is None else lams
            mb[i], ml[i] = z["best"], z["leafc"]
            ycc.append(rgb_to_ycc(z["rgb"])); root_opt.append(z["best"][:, space.root].astype(np.float64)); src.append(str(z["src"]))
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(files)} tiles, {time.time() - t0:.0f}s", flush=True)
        mb.flush(); ml.flush(); del mb, ml
        ycc, root_opt = np.stack(ycc).astype(np.float32), np.stack(root_opt)
        tmp = d / "_index.tmp.npz"
        np.savez(tmp, ycc=ycc, root_opt=root_opt, lams=lams, src=np.array(src), stamp=np.array(stamp))
        tmp.replace(idx)
        return cls(ycc, TileStore(np.load(fb, mmap_mode="r"), device), TileStore(np.load(fl, mmap_mode="r"), device),
                   lams, space, device, src, root_opt)

    def __len__(self):
        return self.n
