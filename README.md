# AMBI

Image codec built on **guillotine partitions**: each 64×64 tile is recursively cut into
rectangles (any width/height that is a multiple of 4, 4..64, leaf aspect ratio ≤ 8:1), and
each leaf is coded with the cheapest of three modes. The partition is meant to be chosen by
an RL policy; this repository contains the exact optimum (the oracle), a first learned
policy trained to imitate it (milestone 4: **works end to end, but not yet close to the
oracle**; see "Learned partition policy"), and everything around them.

## What exists (milestones 1-4)

| Module | Purpose |
|---|---|
| `config.py` | `CodecConfig`: geometry (tile, grid, min/max block, aspect cap), quantizer steps, entropy/distortion settings |
| `color.py` | RGB ↔ YCbCr, full-resolution 4:4:4 |
| `dct.py` | Orthonormal DCT-II for any block size; frequency-band map shared across shapes |
| `entropy.py` | **Integer-CDF** entropy model (tables sum to 2^16), prior, count-based refit, save/load |
| `rangecoder.py` | Real range coder (encoder + decoder) driven by those integer CDFs |
| `modes.py` | Leaf modes: `dct` (4 quantizer levels), `flat` (solid colour), `ts` (transform skip, 4 levels), and with intra prediction `intra` (predictor x 4 levels) |
| `intra.py` | **Intra predictors** (dc, v, h, tm, dr, drv, drh) on any rectangle, from the already-decoded top/left neighbours |
| `distortion.py` | `α·c·MSE + (1−α)·(1−SSIM_local)`, block pasted into original context |
| `oracle.py` | Exact DP over all guillotine partitions of a tile; iterative entropy-table fitting |
| `blockio.py` | Reference block serializer/deserializer on the real range coder (validates rate estimates) |
| `viz.py` | Partition overlay images |
| `bitstream.py` | **.ambi container + whole-image encoder/decoder** (header, tile trees, edge padding, one range-coder stream) |
| `profile.py` | `Profile` = config + entropy tables (+ id and CRC fingerprint); save/load, profile directory lookup |
| `deblock.py` | **Deblocking post-filter** across leaf boundaries; the encoder picks a per-image level (stored in the header) |
| `search.py` | Per-tile cost tables and lambda search: hit a **max file size** or **target PSNR**, verified by a real decode |
| `metrics.py` | PSNR / SSIM(Y) for reports |
| `errors.py` | `AmbiFormatError` (bad magic, wrong profile, truncated or corrupt data) |
| `rects.py` | `RectSpace`: all 18,496 rectangles of a tile, the 31 actions, child/cost tables, tree building from actions |
| `ambi_rl/` | **Policy network** (`model.py`), rollouts, oracle-data loader, imitation + regret training, `PolicyPartitioner` (needs PyTorch) |
| `ambi_cli/` | The `ambi` command: `encode`, `decode`, `info`, `profiles` (`encode --policy FILE` uses a trained policy) |

## Rules baked in

- Cut positions are multiples of 4 px; leaf side ∈ [4, 64]; aspect cap 8:1 applies to **leaves**
  (intermediate nodes may be any shape, since a long strip can always be cut further).
- Probabilities are integers everywhere, so encoder and decoder agree bit-exactly.
- Leaf mode and quantizer are chosen by exhaustive search on `J = D + λR` (not learned).
- Side information (leaf flag, cut axis, cut position, mode id) is charged in bits, so the
  optimizer cannot over-split for free. Optional `mu_bits` adds a per-leaf bias toward larger blocks.

## File format (version 1)

```
"AMBI" | version u8 | profile_id u8 | fingerprint u32 | width varint | height varint | colorspace u8 | deblock level u8 | payload
```

Version 1 added the deblock-level byte (cost: 1 byte per file). Version 0 files from milestone 2
are not readable by this build; the format is a pre-release and may still change.

The payload is a single range-coder stream: tiles in raster order; each tile is its partition
tree (leaf flag, cut axis, cut position, mode id, all as uniform symbols; the mode id is entropy coded in intra profiles) with each leaf's
coefficients inline. Sizes that are not multiples of 64 are padded by edge replication and
cropped by the decoder. Tile size, block limits, aspect cap, quantizer steps and the entropy
tables live in the **profile**; the fingerprint stops a file being decoded with the wrong one.
The decoder reconstructs in float64 so output does not depend on float32 rounding.

## Deblocking

After decoding, a filter smooths the seams between neighbouring leaves. Unlike a fixed JPEG grid,
AMBI's leaf boundaries are placed by the encoder where the content changes, so many of them sit on
real edges; the textbook H.264-style thresholds therefore *lost* about 0.3 dB on photos. The shipped
filter is much stricter: it only touches an edge when both sides are nearly flat and the step is a
fraction of the quantizer step, never changes a pixel by more than `deblock_tc * step`, and the
**encoder tries levels 0..4 per image and stores the best one** (`--deblock auto`, the default;
`--deblock 0` turns it off, `ambi decode --no-deblock` skips it for A/B comparison). The level scales
how permissive the edge test is (smooth images tolerate a high level, textured ones a low one).
The thresholds live in the profile (`deblock_*` fields, covered by the fingerprint).

Measured with the encoder's own level choice (photo-demo profile, 4 lambdas from 0.01 to 0.3):

| set | mean PSNR change | mean SSIM(Y) change | levels > 0 |
|---|---|---|---|
| tuning crops (coffee, rocket, hubble) | +0.17 dB | +0.0020 | 12/12 |
| held-out crops (astronaut, cat) | +0.17 dB | +0.0013 | 8/8 |
| sharp content (text, page; safety check) | +0.06 dB | +0.0015 | 6/8, never negative PSNR |

That is roughly 2% bitrate at equal PSNR, up to +0.5 dB on smooth content at low bitrates. An oracle
that switched the filter on or off per edge (needs side information, not built) would reach about
+0.3 dB on the tuning crops, so per-edge flags are a possible later refinement. The partition
search does not know about the filter (distortion is measured before filtering), so it is slightly
conservative; making the oracle/RL reward filter-aware is a later step. `scripts/deblock_experiments.py`
and `scripts/deblock_report.py` reproduce the tuning and the table above.

## Intra prediction (opt-in per profile)

A leaf can be **predicted from the pixels already decoded above and to its left**, and only the residual is DCT-coded. It is a
profile option (`cfg.intra_preds`), so profiles, `.ambi` files and oracle data made without it stay valid and unchanged.

- **Predictors** (`intra.py`): `dc` (mean of the neighbours), `v` (copy the row above), `h` (copy the left column), `tm` (left + above -
  corner), `dr` (down-right diagonal), `drv` / `drh` (steeper / flatter diagonals). Each predictor is combined with each of the 4
  quantizer levels, so 5 predictors add 20 modes. All predictions are integers, so encoder and decoder agree bit for bit.
- **Why the neighbours exist.** Tiles are decoded in raster order and each tree depth first, first child first. For any leaf, the pixels
  directly above, directly left and up-left are always decoded before it (a separating cut always puts them in the first child);
  up-right and down-left are not, so no predictor uses them. Pixels outside the picture are substituted (128, or the one side that exists).
- **The encoder is a closed loop.** The oracle's dynamic programme stays exact for a *surrogate*: it predicts every rectangle from the
  ORIGINAL neighbouring pixels (the tile edge counts as the picture edge when a tile is coded alone, as in `gen_oracle_data.py` and
  during profile fitting; `plan_image` and `TileTables` give each tile its real top/left border). The partition it picks is then coded in decoding order, and every leaf's mode is chosen
  with the neighbours the decoder will really have (the reconstruction), so the reported bits and quality are real. The gap between the
  surrogate and the closed loop is where a learned (PPO) policy can later beat the oracle.
- **Entropy tables.** Intra residuals have their own tables (`ir_ac_mag`, `ir_dc_cat`, `ir_cbf`) and the mode id is entropy coded (table
  `mode`, by block area) instead of uniform; both exist only in intra profiles. Fit them with `fit_profile.py --intra dc v h tm dr`.
- **Cost.** The oracle evaluates every predictor on every rectangle, so it is roughly 1 time unit per extra predictor slower: 5
  predictors make fitting and oracle-data generation about 5x slower (the policy at encode time is unaffected apart from the
  closed-loop pass, which is cheap). Use fewer predictors (`dc v h tm`) for a cheaper start.
- **Compare it fairly** with `scripts\compare_profiles.py`: it encodes the same pictures with two profiles at several lambdas and
  compares sizes at equal PSNR and equal SSIM(Y), since the same lambda is a different quality in different profiles. Fit the two
  profiles with `--base-profile` so they differ only in intra prediction.
- **Measured so far (small test, no gain yet).** Two profiles fitted on the same 20 stock-photo tiles (`--base-profile`, 2 iterations),
  exact oracle partitioner, 4 held-out 128x128 crops: the intra profile (dc v h tm dr) was **1-6% larger at equal PSNR** (median about
  +2%, smaller on about half of the crops). It chooses intra for 75-90% of the leaf area but makes about 4x as many leaves (465 vs 114
  at lambda 0.03 on one crop), and the mode ids alone were about 18% of that file. With only 20 fitting tiles the new tables (`ir_*`,
  `mode`) are mostly their priors, so this says "not worth it at this scale", not "does not work". The real test is a CLIC fit:
  `fit_profile.py --base-profile profiles\clic.ambiprofile --intra dc v h tm dr` and then `compare_profiles.py` on held-out pictures.
  If it stays flat, the first thing to try is coding the mode id conditioned on the previous leaf's mode (not done yet).
- **Existing policies still work.** A policy only chooses the partition, so an old `.ambipolicy` can drive an intra profile; for the best
  result generate oracle data with the intra profile and train again (the partitions that are optimal change).

## Lambda-map probe (is a per-tile lambda worth it?)

`scripts\lambda_map_probe.py` answers "would letting the encoder use a different lambda in each tile beat one global lambda?" without
changing the codec. Lambda is never stored in the file, so a tile coded at another lambda is an ordinary tile. For profiles without
intra prediction (tiles are independent) the probe encodes the picture at a ladder of constant lambdas, composes any per-tile map from
those results, searches for the map with the best whole-picture MS-SSIM for its size, then encodes the winning maps with the real
encoder, decodes the files, and compares sizes with the constant-lambda curve at EQUAL quality (MS-SSIM, SSIM(Y), PSNR).

    python scripts\lambda_map_probe.py --image-dir <val images> --profile profiles\clic-base.ambiprofile --crop 256 --max-images 8 --lams 0.02 0.08

Expect PSNR to get slightly worse (a constant lambda is already optimal for additive MSE); any gain is in MS-SSIM / SSIM. The map is
chosen knowing the original, so the saving is an upper bound for a learned predictor of the map. If it is under about 2%, do not build
the RL part. MS-SSIM needs pictures of at least 176 px, so use `--crop 256` or more.

### Seeing it: `lambda_map_side_by_side.py`

Draws one sheet per crop from REAL files (every picture is a decoded `.ambi`): the original, a constant-lambda file searched until its
quality equals the map's (the fair size comparison), the lambda-map file, the constant file at the map's starting lambda, a picture of
the per-tile multipliers (blue = finer than the global lambda, orange = coarser, labelled x0.5 / x2 ...) and the difference between the
two compared files. The two compared `.ambi` files and the original crop are saved next to the sheet; the map file decodes with the
ordinary decoder.

    python scripts\lambda_map_side_by_side.py --image-dir <val images> --profile profiles\clic-base.ambiprofile --crop 256 --max-images 3 --lam 0.02 --zoom 64
    python scripts\lambda_map_side_by_side.py --image-dir <val images> --profile profiles\clic-base.ambiprofile --map-json maps.json --lam 0.02

`--match msssim|ssim|psnr` chooses the quality the two files are matched on (default: what the map was optimised for; on `psnr` the map is
expected to lose). Maps come from a fresh search (about 2 minutes per 256x256 crop, mostly the cost tables) or from a `--json` written by
`lambda_map_probe.py` (`--map-json`). The same `--seed`, `--crop`, `--crops-per-image` and `--max-images` as a probe run pick the same crops.
Sheets go to `<image folder>_analysis\<name>_lambdamap`, the files to `<name>_lambdamap_files` (override with `--analysis-dir`).

## Comparing profiles by eye: `side_by_side_profiles.py`

`compare_profiles.py` gives the numbers (bytes at equal PSNR / SSIM); this draws the pictures. For every crop and every lambda one
block of two rows: the PNG original and one AMBI panel per profile on top, and under each AMBI panel a JPEG of the same file size
(`--jpeg psnr`: the same PSNR instead). Each panel shows size, bits per pixel, PSNR, SSIM(Y) and the compression (% of the PNG and
"N times smaller"); the AMBI panels after the first also show their difference to the first profile, the JPEG panels their difference
to the AMBI panel above. The files are real: `.ambi` and `.jpg` are written next to the sheets and the AMBI panel is the file decoded
back from disk. The crops are the ones `compare_profiles.py` picks with the same `--seed`, `--crop`, `--crops-per-image` and
`--max-images`. All profiles share one lambda, so their sizes differ a little; each JPEG is matched to its own AMBI panel.

```powershell
python scripts\side_by_side_profiles.py --image-dir I:\dataset\ambi\clic2024_split\val --oracle --crop 256 --max-images 8 --lam 0.02 `
    --profiles profiles\clic-base.ambiprofile profiles\clic-intra.ambiprofile profiles\clic-base-2k.ambiprofile profiles\clic-intra-2k.ambiprofile `
    --labels "old base" "old intra" "new base" "new intra" --zoom 64 --json profile_sheets.json
```

Sheets go to `...\val_analysis\val_profiles`, the converted files to `...\val_analysis\val_profiles_files`. It ends with a pooled table
(compression and quality per profile and lambda, next to the same-size JPEG). Time: the cost tables dominate, about the same per profile
as in `compare_profiles.py`, so four profiles take roughly twice one A-versus-B run per crop.

**Using the cores.** Both scripts run on ONE core unless you pass `--workers N` (with `--oracle`): each crop-and-profile pair then runs
in its own process, the slowest (intra) jobs first, with one status line for the whole run. Results, `.ambi` files and JSON are identical to a
single-process run. There are `2 x crops` jobs in `compare_profiles.py` (`profiles x crops` in the sheet script), so more workers than
that do not help; each worker needs roughly 0.3 GB for a 256 px crop. For 8 crops: `--workers 16`.

## Learned partition policy (milestone 4)

**What it is.** For every rectangle on the way down, the network scores 31 actions: keep as a leaf,
cut vertically at one of 15 positions, cut horizontally at one of 15 positions. The tile is
partitioned by applying the best legal action to the root, then to each child, and so on. A trunk
of stride-4 convolutions produces a 16×16 feature map; each rectangle is described by pooled
features plus exact statistics (variance, gradient energy, the jump across every candidate cut
line, variance between the two halves), and the score heads also see log λ.

**How it is trained (stage 1, imitation of the exact optimum).** The oracle DP already computes the
best cost of every rectangle, so the cost of *every* action at *every* rectangle is known without
extra oracle calls. For a tree problem, `J(policy) − J(optimum)` is exactly the sum, over the
rectangles the policy visits, of (cost of the chosen action − cost of the best action). So the loss
is a soft cross-entropy over action costs plus the expected regret itself, evaluated on three
state sources: the oracle's path, random shapes, and the policy's own rollouts (DAgger-style).
PPO fine-tuning is the planned next stage; it matters mostly once neighbour prediction makes block
costs depend on each other, because only then can it beat the oracle (today the oracle is exact).

Step-by-step instructions for training on your own photos with the RTX 4070 are in the next section.
A policy file decodes with any decoder: the bitstream does not depend on how the partition was chosen.

**Result of the demo run (CPU sandbox, 1,500 steps, 400 training tiles from 3 stock photos,
60 held-out tiles from 2 other photos).** Cost J above the exact optimum on held-out tiles:

| λ | 0.005 | 0.02 | 0.08 | 0.32 |
|---|---|---|---|---|
| learned policy, mean | 12.1% | 14.3% | 17.7% | 19.1% |
| learned policy, median | 11.1% | 12.1% | 12.8% | 14.2% |
| best fixed grid (chosen per tile with hindsight) | 16.8% | 18.4% | 19.2% | 21.4% |

Speed: **7 ms per tile** (CPU, batched) against about 4,000 ms for the oracle, roughly 500× faster.
On training tiles the gap is 7-10%, so the network is data-limited: it overfits 400 tiles.
Real files (held-out photos, same λ, filter on): policy files are 7-24% larger than oracle files,
with PSNR within about ±0.5 dB (mixed sign).

**Honest reading.** The policy is a working, fast, valid partitioner that beats a fixed grid by a few
points, but it is nowhere near the oracle, and these numbers come from a tiny CPU run that says
little about what the 4070 can reach. Most of the remaining error is choosing a cut at the wrong
position or axis when it does want to cut (about 9-12 of the gap's points), followed by
under-splitting and over-splitting. Adding exact edge-strength features at the candidate cut lines
did not change the held-out gap (16.0% either way), which points to data volume and model size
rather than missing inputs. Things to try on real data: 20-50× more tiles, width 96 / hidden 192,
and using the policy as a pruner for a local DP (keep the top few cuts per rectangle, solve exactly
inside that).

## Training your own policy (RTX 4070, Windows)

Everything below is PowerShell, run from the repo folder. Budget: about an hour of setup and a
few hours of mostly unattended CPU time for data, then the GPU run. The GPU code path could only
be run on CPU where this was written, so if the first run fails, send the traceback.

**0. Pick the images.** Put photos in two folders that share no images: `D:\wild\train` and
`D:\wild\val`. Split by *image*, never by tile, or the validation number is flattering. Aim for
at least 300 training images (more variety beats more tiles per image) and 40 or more validation
images. Tiles are cut at each image's native resolution, so use the resolution you intend to
encode: downscale your originals first if you will mostly compress downscaled copies.

*Screenshots or other pictures with big blank areas.* Most 64 px tiles of a screenshot are blank background, which teaches
the policy almost nothing and wastes oracle time. Pass `--flat-keep 0.15` to `fit_profile.py` and `gen_oracle_data.py` so a
blank tile is picked 15% as often as a tile with content (1 = uniform, the default). Screenshots taken seconds apart are
near-identical, so split with `scripts\split_images.py --src SHOTS --out SPLIT --val-frac 0.15 --block 8`: it keeps blocks of
consecutive pictures together (a random split would leak) and skips exact duplicates. The oracle data folders contain the
raw 64x64 pixel tiles, so keep them private if the screenshots are; the `.ambiprofile` and `.ambipolicy` files hold no pixels.

**1. Install PyTorch with CUDA, then the project.**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
# take the exact command from pytorch.org (Stable / Windows / Pip / CUDA), for example:
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # expect True and the 4070
```

**2. Fit a profile on your photos** (about 10 minutes). The oracle data is tied to the profile, so
do this once, first. The `--id` is stored in every .ambi file; keep it unique (the demo uses 1).

```powershell
python scripts\fit_profile.py --image-dir D:\wild\train --name wildlife --id 2 --out profiles\wildlife.ambiprofile
```

**3. Generate the oracle data** (the slow, CPU-bound part). It costs about 5 CPU-seconds per tile for
all 7 lambdas, so 4,000 tiles is roughly 5.5 CPU-hours: with `--workers 8` about 40 minutes. It is
resumable (rerun the same command after a Ctrl+C). On a rerun it skips every tile whose file is complete and was made for the same tile, profile and lambdas; a file cut short by the Ctrl+C is redone. The ETA counts only tiles computed in the current run, and shows `ETA ...` for the first few seconds. It takes about 0.9 MB of disk per tile.

```powershell
python scripts\gen_oracle_data.py --image-dir D:\wild\train --profile profiles\wildlife.ambiprofile --out data\wild_train --per-image 12 --workers 8
python scripts\gen_oracle_data.py --image-dir D:\wild\val   --profile profiles\wildlife.ambiprofile --out data\wild_val   --per-image 8 --workers 8 --extra-lams 0.03 0.12
```

Leave one or two cores free (`--workers` below your core count). Training loads the whole dataset
into RAM (about twice its size at the peak) and then into GPU memory: 4,000 tiles is about 3.5 GB of
VRAM and 7 GB of RAM. If memory is tight, pass `--limit-tiles 2500` to `train_policy.py`.

**Big data sets (`--store`).** Each tile costs about 1 MB (7 lambdas x 18,496 rectangles x 2 arrays, float32), so a 12 GB GPU holds
only about 8,000 tiles. With `--store auto` (the default of `train_policy.py`) a set that would need more than `--gpu-budget-gb`
(default 6) of device memory is read from disk instead: the first run builds two cache files next to the tiles
(`_cache_best.npy`, `_cache_leafc.npy`, plus `_index.npz`; about 1 MB per tile again, about 9 ms per tile to build, reused until the
tile files change) and every step copies only the 8 or so tiles it needs to the GPU (about 3 ms). 20,000 tiles then train fine;
the limits are disk space and the page cache. The numbers are identical to `--store gpu` (tested). `--store gpu` forces the old
behaviour, `--store disk` forces the cache even for a small set. The validation set is also checked against the budget, so a
1,000-tile set stays on the GPU.

**Would a pruned exact search work? (`topk_diag.py`).** The oracle data holds the exact value of every rectangle, so a search that only
considers the policy's k best actions per rectangle can be simulated without any encoder:

    python scripts\topk_diag.py --data-dir data\clic2k_val --profile profiles\clic-base-2k.ambiprofile --policy policies\clic2k_big.ambipolicy --device cuda

It prints, for k = 1, 2, 3, 5, 8, 12 (`--ks`), the aggregate J gap per lambda (k = 1 is the policy's own greedy tree and matches the
training log; k = all is the oracle), the share of the 18,496 rectangles such a search would have to visit (the oracle visits all of
them), and how often the oracle's own choice is inside the policy's top k on the oracle's optimal tree. It uses the first
`--limit-tiles` (150) tiles; lower `--batch` / `--chunk` if the GPU runs out of memory.

**Pruned exact search (`--prune-k K`).** The policy alone trusts its first choice at every rectangle. With `--prune-k K` the exact DP runs
over the policy's K best actions per rectangle instead: only the rectangles those candidates lead to are looked at, only those where
"leaf" is a candidate are priced as a leaf (the slow part of the oracle), and the cheapest combination wins. K = 1 is the policy's own
tree, K = every action is the exact oracle, and in between the result is never worse than the policy's tree.

    python scripts\compare_sizes.py ... --policy policies\clic2k_big.ambipolicy --prune-k 3 --device cuda
    python -m ambi_cli encode in.png -p clic-base-2k --lambda 0.03 --policy policies\clic2k_big.ambipolicy --prune-k 3 --device cuda

`compare_sizes.py` and `compare_profiles.py` take `--prune-k` too, and print how much of each tile was looked at and where the time went.
`scripts/topk_diag.py` predicts the gap for each K from the oracle data before you encode anything. In a 4-tile test with a weak demo policy
(12-15% J gap) the file size above the oracle fell from +19% (policy alone) to +6.9% (K=3), +2.4% (K=5) and +0.9% (K=8), at 2.7 s, 5.1 s and
8.0 s per image against 16 s for the oracle (2 CPU cores). `--mu` is not passed to the partitioner, so with `--prune-k` (as with the policy
alone) the partition is chosen as if `--mu` were 0.

**Using the cores (`--workers N`, with `--prune-k`).** Nearly all of the pruned search's time is pricing the candidate leaves, which is plain
CPU work, tile by tile. With `--workers N` (`ambi encode` and `compare_sizes.py`) the policy network stays on `--device` in the main process and
N worker processes price and solve the tiles while the device already works on the next batch; the files are byte-identical to the
single-process ones (tested). One process uses one core, and the 7900X3D has 12 physical cores, so try 12 and 24:

    python scripts\compare_sizes.py ... --policy policies\clic2k_big.ambipolicy --prune-k 5 --device cuda --workers 24
    python -m ambi_cli encode in.png -p clic-base-2k --lambda 0.03 --policy policies\clic2k_big.ambipolicy --prune-k 5 --device cuda --workers 24

The workers are started on the first tile and live until the encode ends. The summary line prints the CPU time per tile and, with workers,
the wall time per tile.

**4. Train.** Do a short trial first to see the speed and make sure nothing breaks, then the real run.

```powershell
python scripts\train_policy.py --train-dir data\wild_train --val-dir data\wild_val --profile profiles\wildlife.ambiprofile --out policies\trial.ambipolicy --device cuda --steps 100 --eval-every 100 --warmup-steps 20 --tiles-per-step 32
python scripts\train_policy.py --train-dir data\wild_train --val-dir data\wild_val --profile profiles\wildlife.ambiprofile --out policies\wild.ambipolicy --device cuda --steps 4000 --warmup-steps 200 --eval-every 250 --tiles-per-step 32 --width 96 --hidden 192
```

The log prints the validation J gap every 250 steps and saves a checkpoint each time, so you can stop
any time and use the last one. If you run out of GPU memory, lower `--tiles-per-step` to 16 and
`--uniform-per` to 32. Speed on the 4070 was not measured; use the trial run to extrapolate.

**5. Read the result.** The number to watch is the held-out **aggregate** J gap: the total cost lost
across all tiles, relative to the total optimal cost (the `val: aggregate J gap` line and the last column
of the final tables). It is what an image's total cost actually loses. The *mean of per-tile percentages*
can be wildly larger because near-empty tiles (flat sky, blurred background) have a tiny optimal cost, so a
tiny absolute error is a huge percentage; look at the median tile and the aggregate instead. The demo run
(400 stock tiles) ended at about 11-16% aggregate, depending on lambda. A big difference between the train and val gaps means more
images are needed, not more steps.

```powershell
python scripts\eval_policy.py policies\wild.ambipolicy --profile profiles\wildlife.ambiprofile --val-dir data\wild_val --images photo.png
ambi encode photo.png --profile profiles\wildlife.ambiprofile --policy policies\wild.ambipolicy --lambda 0.03
ambi encode photo.png --profile profiles\wildlife.ambiprofile --lambda 0.03      # oracle, for comparison
```

**6. Compare file sizes.** `scripts\compare_sizes.py` encodes crops (or whole images with `--crop 0`) and reports AMBI's size
against the same pixels stored as lossless PNG, together with the PSNR and SSIM AMBI reaches (PNG keeps every pixel and
AMBI does not, so read the two together). By default it also re-encodes the pixels with JPEG (`--vs png jpeg`) and compares sizes at equal PSNR and at equal SSIM;
add `webp` (`--vs png jpeg webp`) for lossy WebP too; if your photos are themselves WebP files that flatters WebP, so use it on
images that were never WebP-compressed. `--oracle` uses the exact partitioner (slow) instead of a policy.

```powershell
python scripts\compare_sizes.py --image-dir I:\dataset\val --profile profiles\clic.ambiprofile --policy policies\clic_best.ambipolicy --max-images 30
python scripts\compare_sizes.py --image-dir I:\dataset\val --profile profiles\clic.ambiprofile --policy policies\clic_best.ambipolicy --max-images 20 --crop 0 --vs both
```

**7. See the pictures side by side.** `scripts\side_by_side.py` picks PNGs (`--image-dir ... --n 3`, or one `--image`), encodes
each to a real `.ambi` file, decodes it back and writes one image per picture: the lossless PNG, the decoded AMBI picture
and a JPEG of the same pixels, with each panel's size, bpp, lambda, PSNR (dB), SSIM, % of the PNG and encode/decode time
printed under it. By default the JPEG quality is searched to match the AMBI file's size (`--jpeg psnr` matches PSNR instead,
`--jpeg-quality 75` gives one fixed-quality JPEG, `--jpeg none` drops it). Several `--lam` values give several AMBI+JPEG
pairs; `--zoom 128` adds a magnified patch under each panel. Results go next to the pictures: for `--image-dir D` the side-by-side
pictures go to `D_analysis\D_compared\` and the converted files (`.ambi`, `.jpg`, cropped originals) to `D_analysis\D_converted\`
(for `kodak`: `kodak_analysis\kodak_compared` and `kodak_analysis\kodak_converted`).
Change the base with `--analysis-dir`, or each folder with `--compare-dir` / `--convert-dir`. Mixed folders are fine (PNG, JPEG, WebP,
grayscale, palette, 16-bit, transparent); one bad file is reported and skipped.

```powershell
python scripts\side_by_side.py --image-dir I:\dataset\val --n 3 --lam 0.03 0.3 --zoom 128 --profile profiles\clic.ambiprofile --policy policies\clic_best.ambipolicy
```

**Troubleshooting.** `torch.cuda.is_available()` is False: update the NVIDIA driver, or reinstall
torch with the CUDA command from pytorch.org. "fingerprint mismatch": the data was made with a
different profile, so regenerate it. `ambi decode` says "profile 2 not found": the profile must be
in the `profiles` folder (or pass `--profile-dir`). The profile and the policy are separate files:
the profile is needed to decode, the policy only to encode.

## Using the CLI

```powershell
ambi profiles
ambi encode photo.png                         # default lambda 0.03 -> photo.ambi
ambi encode photo.png --lambda 0.1            # higher lambda = smaller file, lower quality
ambi encode photo.png --max-bytes 20000       # best quality that fits
ambi encode photo.png --target-psnr 36        # smallest file that reaches 36 dB
ambi encode photo.png --deblock 0             # filter off (default: encoder picks the level)
ambi encode photo.png --policy p.ambipolicy   # fast learned partitioner (not with --max-bytes/--target-psnr)
ambi encode photo.png --profile path\to\my.ambiprofile --json
ambi info photo.ambi
ambi decode photo.ambi -o photo_decoded.png
ambi decode photo.ambi --no-deblock           # skip the post filter, for comparison
```

`encode` prints savings vs the input file and vs raw size, bits per pixel, and PSNR / SSIM(Y)
measured by decoding the written file. Profiles are looked up in `--profile-dir`, `$AMBI_PROFILE_DIR`,
`.\profiles` and the repo's `profiles` folder. The included `photo-demo` profile was fitted on only
24 tiles from three stock photos: it is a demo, not a trained wildlife codec.

## One picture against JPEG, WebP, AVIF and AMBI: `codec_compare.py`

Takes one PNG (or a folder of them) and draws two separate pictures:

* `NAME.rd_curves.png` - the graph: PSNR (RGB) and SSIM (Y) against bits per pixel, one line per codec, every point a real file decoded
  again. The x-axis is zoomed to the range AMBI covers, and the points used in the second picture are ringed.
* `NAME.comparison.png` - the side-by-side telemetry: columns Original / JPEG / WebP / AVIF / AMBI, each with the whole picture and a
  detail window at native pixels (the busiest area, or `--box X Y`), then file size, size against the original file, bitrate, PSNR, SSIM,
  encode and decode time. The best value in each row is highlighted.

The Pillow codecs are searched to a *matched point* so the comparison is fair: by default each gets the highest quality whose file is no
larger than AMBI's (`--match size`); `--match psnr` instead gives each the smallest file that reaches AMBI's PSNR. A quality that cannot
reach the target is printed as such, not hidden. Next to the pictures it writes the real files (`NAME.jpg/.webp/.avif/.ambi`) and
`NAME.telemetry.csv` with every encode that was made.

```powershell
pip install matplotlib
python scripts\codec_compare.py photo.png -o out\codecs
python scripts\codec_compare.py photo.png -o out\codecs --match psnr --crop 512 --lams 0.02 0.08 0.32 --lambda 0.08   # quicker
```

It uses `profiles\clic-base-2k.ambiprofile` and `policies\clic2k_big.ambipolicy` with `--prune-k 5` by default (`--oracle` for the exact
search instead). The AMBI sweep (`--lams`, default 0.005 to 0.32) dominates the run time, since every lambda is a full encode; JPEG, WebP
and AVIF take well under a second each. JPEG is 4:2:0 with optimised tables, WebP is lossy at method 6, AVIF is lossy at speed 6
(`--webp-method`, `--avif-speed`). AVIF needs Pillow 11.3 or newer, or `pip install pillow-avif-plugin`; without it the script says so and
leaves AVIF out. If the input is itself a JPEG/WebP/AVIF the script warns: its errors are part of the "original", which flatters JPEG and
WebP, so use a PNG made from a lossless source. WebP and AVIF are here because the comparison was asked for; the usual reports compare
against PNG and JPEG only.

## The window (`ambi_gui`)

An encoder and a viewer in one window. It is a thin front end over the same code the CLI uses (`ambi_core` and `ambi_rl`); it adds no
codec logic.

```powershell
pip install -e ".[rl,gui]"
python -m ambi_gui                  # or: ambi-gui
python -m ambi_gui photo.ambi       # open a picture or an .ambi file straight away
```

- **Encode**: open a picture (button, File menu, or drop it on the window), choose what to encode for, press Encode.
  - *Quality*: a 0-100 slider (0 = smallest file, 100 = best picture; it maps onto lambda 0.32 .. 0.005 on a log scale, and the label
    shows the lambda).
  - *File size*: the best quality that fits under a size in KB, MB, or as a percentage of the original file.
  - *PSNR*: the smallest file that reaches a PSNR.
  - *Pruning*: policy only / k = 3 / k = 5 (default) / k = 8 / exact oracle. See `claude/pruned-search.md`: more candidates per rectangle
    means a smaller file and a slower encode. *Workers* is the number of processes for the pruned search (the default is half the logical
    cores: 12 on a 7900X3D; more than the physical core count does not help).
  - With a policy, *File size* and *PSNR* are found by real encodes at a few lambdas (2-4 are typical, at most 7), so they take that many times
    longer than a *Quality* encode. The exact oracle uses its own cost-table search instead. Cancel stops at the next progress step: within a tile or two while pruning, but the network pass of a policy-only encode is not interruptible.
- **View**: the picture shows the original and the decoded file with a draggable swipe divider (or one of them alone). Wheel zooms, drag
  pans, double-click or F toggles fit / 100%, Space flips between the two.
- **Results** under the picture: size (and how much smaller than the original file and than raw pixels), bits per pixel, PSNR and SSIM(Y),
  the PNG size of the same pixels, the JPEG size at the same PSNR, what the pruning looked at, and the times.
- **Open an .ambi**: the window finds the profile with the file's id *and* fingerprint in the profile folder and decodes it. There is
  no original in the file, so it shows size, bits per pixel, profile and decode time; "Compare the opened .ambi with its original..."
  loads the picture it was made from and adds PSNR and SSIM.
- **Files**: profile and policy are chosen from two folders (default: `.\profiles` and `.\policies`, then the repo's). The window
  prefers `clic-base-2k` and `clic2k_big` when they are there. The window does not check that a policy was trained for the chosen
  profile.

The encode runs on a background thread, so the window stays responsive. The worker processes of the pruned search are started with
`spawn` (Windows), which is why `ambi_gui/__main__.py` has the `if __name__ == "__main__"` guard and `freeze_support()`.

### One program instead of Python (`scripts\build_exe.ps1`)

PyInstaller packs the window, Python, numpy, PySide6 and PyTorch into a Windows program. It cannot cross-compile, so build on your
Windows machine, in the virtual environment that already runs the window:

```powershell
pip install -e ".[rl,gui]"
.\scripts\build_exe.ps1                # dist\ambi-gui\ambi-gui.exe: a folder that starts fast (zip the folder to share it)
.\scripts\build_exe.ps1 -OneFile       # dist\ambi-gui.exe: a single file, but it unpacks itself on every launch
.\scripts\build_exe.ps1 -NoTorch       # leave PyTorch out: small, but only the exact-oracle option works
.\scripts\build_exe.ps1 -NoData        # do not pack .\profiles and .\policies inside
```

- **Size is the catch.** The CUDA build of PyTorch brings its CUDA libraries with it: a test build with CUDA PyTorch 2.15 came to about 5 GB
  as a folder (the sandbox python also had unrelated packages installed, so yours may be smaller). A single file of that size unpacks to the
  temp folder on each launch, which takes a while. For a smaller program, build from a venv with the CPU-only torch
  (`pip install torch --index-url https://download.pytorch.org/whl/cpu`): the network then runs on the CPU. The pruned search itself runs on the CPU
  workers either way.
- **Profiles and policies** are packed inside (unless `-NoData`), and the window looks for `profiles` and `policies` folders next to the
  .exe, then in the working folder, then at the packed copy. Put your own folders next to the .exe to override the packed ones.
- **Errors** from the windowed program (it has no console) go to `%LOCALAPPDATA%\AMBI\ambi-gui.log`.
- Drag a picture or an .ambi file onto the .exe, or run `ambi-gui.exe photo.ambi`.
- The build writes `build\`, `dist\` and `ambi-gui.spec` into the repo root; delete them freely. Antivirus tools sometimes flag
  PyInstaller programs; that is a known false positive of the packaging, not of AMBI.
- Tested in the Linux sandbox: a folder build with PyTorch ran a pruned encode with spawned workers, the exact oracle, and a decode that
  matched, and a one-file build started its worker processes. Not yet built or run on Windows.

## Setup (Windows, PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,gui]"
pytest
python scripts\demo_oracle.py --out out        # fit tables, sweep lambda, compare with JPEG
python scripts\real_files_report.py            # real .ambi sizes vs real JPEG sizes
python scripts\make_rd_plot.py                 # rate-distortion chart
python scripts\fit_profile.py --help           # build your own profile (see the training section)
python scripts\deblock_experiments.py build   # cache crops, then: python scripts\deblock_report.py
```

PyTorch is only needed for `ambi_rl`, the policy scripts and `--policy` (`pip install -e ".[rl]"`;
for the RTX 4070 install the CUDA build of torch from pytorch.org first). Everything else runs without it.

## Known limitations

- Intra prediction is opt-in and new: the surrogate oracle (partition chosen against the original neighbours) and the closed loop (modes
  chosen against the reconstruction) can disagree, and oracle data and profile fitting code each tile alone with its edge treated as the
  picture edge, so leaves on the top and left edge of a tile cannot use their real neighbours there (`TileTables`/`plan_image` do). Profiles without intra still
  code every block's DC from scratch, and flat/TS means cost 8 raw bits per channel.
- Tree flags use uniform-cost side information (no context modelling).
- Rate model is per-tile: no context carried across tiles; tile borders use reflected context for SSIM.
- The decoder uses float64, but a cross-platform guarantee still needs an integer DCT before the format is frozen.
- Padding pixels on images that are not multiples of 64 are coded and scored like real pixels.
- No metadata (EXIF, colour profile) or alpha is stored; the format is a pre-release (version 1) and may change.
- The encoder is slow: about 4 s per 64×64 tile for one lambda, 5 s per tile to build cost tables (CPU, 2 cores).
  Large images use a 12-tile sample to estimate the target-size/quality curve.
- Deblocking is modest: seams soften but strong block edges remain at low bitrates (see `out/compare_deblock.png`).
- Python-loop bitstream writer: fine for tests and small images, slow for large ones.
- The learned policy is 12-19% above the oracle's cost on held-out tiles (demo run); `--policy` cannot be combined with `--max-bytes` / `--target-psnr`, which need the oracle's cost tables.
- The bundled demo policy was trained on 400 tiles from 3 stock photos; it is a pipeline check, not a product.
