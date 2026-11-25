from __future__ import annotations

import json
import os
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import CodecConfig
from .entropy import EntropyModel, table_shapes


@dataclass
class Profile:
    name: str
    profile_id: int
    cfg: CodecConfig
    model: EntropyModel

    @property
    def fingerprint(self) -> int:
        blob = json.dumps(self.cfg.decoder_dict(), sort_keys=True).encode()
        for k in sorted(self.model.freqs):
            blob += k.encode() + np.ascontiguousarray(self.model.freqs[k], dtype="<i4").tobytes()
        return zlib.crc32(blob) & 0xFFFFFFFF

    def save(self, path) -> None:
        meta = json.dumps({"name": self.name, "profile_id": self.profile_id,
                           "cfg": self.cfg.to_dict(), "format": 0})
        with open(path, "wb") as f:
            np.savez_compressed(f, meta=np.array(meta),
                                **{f"f_{k}": v for k, v in self.model.freqs.items()})

    @classmethod
    def load(cls, path) -> "Profile":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        cfg = CodecConfig.from_dict(meta["cfg"])
        model = EntropyModel(cfg, {k: z[f"f_{k}"] for k in table_shapes(cfg)})
        return cls(meta["name"], int(meta["profile_id"]), cfg, model)


def default_profile(cfg: CodecConfig | None = None) -> Profile:
    cfg = cfg or CodecConfig()
    return Profile("default-prior", 0, cfg, EntropyModel(cfg))


def profile_search_dirs(extra=()) -> list[Path]:
    dirs = [Path(d) for d in extra]
    if os.environ.get("AMBI_PROFILE_DIR"):
        dirs.append(Path(os.environ["AMBI_PROFILE_DIR"]))
    dirs.append(Path.cwd() / "profiles")
    dirs.append(Path(__file__).resolve().parents[2] / "profiles")
    return dirs


def list_profiles(extra=()) -> list[Profile]:
    seen, out = set(), []
    for d in profile_search_dirs(extra):
        if d.is_dir():
            for f in sorted(d.glob("*.ambiprofile")):
                if f.resolve() in seen:
                    continue
                seen.add(f.resolve())
                out.append(Profile.load(f))
    return out


def find_profile(spec, extra=()) -> Profile:
    if isinstance(spec, Profile):
        return spec
    if isinstance(spec, (str, Path)) and Path(spec).is_file():
        return Profile.load(spec)
    profs = list_profiles(extra)
    for p in profs:
        if (isinstance(spec, int) and p.profile_id == spec) or p.name == spec:
            return p
    if spec in (0, "default-prior", None):
        return default_profile()
    raise FileNotFoundError(f"profile {spec!r} not found (searched {[str(d) for d in profile_search_dirs(extra)]})")
