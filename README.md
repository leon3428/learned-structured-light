# End-to-End Learning of Projection Patterns for Structured Light Systems

Code for the paper by Leon Stjepan Uroić and Damir Seršić (University of Zagreb).

Structured-light patterns are learned jointly with a U-Net depth decoder. Each training
scene has a light transport matrix (LTM) `T`, so the camera image for pattern `p` is
`c = T sigmoid(p)`: a sparse matrix product that gradients flow through back to the
projector pixels. The LTMs are rendered with a [modified Mitsuba 3](https://github.com/leon3428/mitsuba3)
and published as the
[ABC Light Transport 256](https://huggingface.co/datasets/leon3428/abc-light-transport-256)
dataset (100,000 scenes each for diffuse and metallic objects).

## Repository layout

```
configs/                 one TOML file per trained model in the paper (Tables 2-4)
src/learned_sl/
  data.py                Parquet reader for the Hugging Face dataset, validation split
  model.py               DepthNet (patterns -> LTM -> camera images -> U-Net)
  losses.py              masked RMSE, masked SSIM, pattern-variance loss
  train.py, evaluate.py  training and evaluation (single GPU or torchrun)
  baselines.py           classical FTP and PSP baselines
  patterns.py            fixed sinusoidal patterns, export of learned patterns
  figures/               metric plots (Fig. 3) and mesh renders (Figs. 4-5)
  generation/            dataset generation from ABC meshes (needs the Mitsuba fork)
  splits/                ids of the 9,000 validation samples per subset
external/mitsuba         Mitsuba 3 fork with the LTM renderer (git submodule)
```

## Installation

Python 3.12+ and an NVIDIA GPU are needed for training. With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/leon3428/learned-structured-light.git
cd learned-structured-light
uv sync                    # add --extra wandb to log to Weights & Biases
source .venv/bin/activate
```

or `pip install -e .` in a virtual environment.

## Data

```bash
sl-download --subset diffuse                 # ~94 GB into data/abc-light-transport-256
sl-download --subset metalic                 # ~108 GB
sl-download --subset diffuse --splits test   # test split only (~10 GB), for evaluation
```

Training and evaluation read the Parquet shards directly; there is no conversion step.
Pass `--data-root` to the other commands if the data is elsewhere. The metallic subset
is named `metalic` on the Hub; the configs call its models `metallic_*`.

The paper splits each subset into 81,000 training, 9,000 validation and 10,000 test
samples. Train and test are the dataset's own splits. The 9,000 validation samples are
held out of the training split, and their ids are in `src/learned_sl/splits/`.

## Reproducing the paper

### Training

Each config in `configs/` is one model from the paper:

| Paper | Configs |
|---|---|
| Table 2 (diffuse), Learned / Fixed | `diffuse_learned_{1,3,5}`, `diffuse_fixed_{1,3,5}` |
| Table 3 (metallic), Learned / Fixed | `metallic_learned_{1,3,5}`, `metallic_fixed_{1,3,5}` |
| Table 4 (ablations, 3 learned diffuse) | `diffuse_learned_3` (full loss), `ablation_no_variance`, `ablation_no_ssim`, `ablation_no_ssim_no_variance` |

```bash
sl-train configs/diffuse_learned_3.toml                                          # one GPU
torchrun --nproc-per-node 2 -m learned_sl.train configs/diffuse_learned_3.toml   # two GPUs
```

`batch_size` in the configs is the total over all GPUs: 16, as in the paper, where it
was 8 per GPU on two RTX 2080 Ti. A run takes about 9 hours on two 2080 Ti. Each run
writes `runs/<name>/` with the config, `metrics.jsonl` (training loss and validation
metrics per epoch), `last.pth`, and `best.pth`, the checkpoint with the lowest
validation MAE. The paper reports test metrics for that checkpoint.

### Evaluation

```bash
sl-evaluate runs/diffuse_learned_3 --paper-rounding --results-csv results/my_results.csv
```

This evaluates `best.pth` on the 10,000 test samples and writes
`runs/diffuse_learned_3/test_metrics.json`. With `--results-csv`, it also appends a row
to a CSV that `sl-plot-metrics` reads. Metrics cover the pixels that the projector
lights in the ground truth.

`--paper-rounding` reproduces how the paper's tables were computed. The decoder runs in
mixed precision and outputs float16 depth, and the original evaluation converted that to
millimeters in float16, which rounds predictions to 0.5 mm steps near 1 m. Without the
flag, errors are computed in float64. For the 3-pattern learned diffuse model, this
changes MAE by about 0.002 mm and BAD-2 by about 0.7 percentage points.

### Classical baselines

```bash
sl-baseline --subset diffuse --methods ftp psp3 psp5 --results-csv results/my_results.csv
```

FTP projects one OpenCV fringe pattern and decodes it by Fourier demodulation. PSP
projects OpenCV's three-step patterns, or five-step patterns generated in
`baselines.py`, and decodes them by least-squares phase shifting. Correspondences are
triangulated with the known synthetic geometry (`geometry.py`).

### Figures

```bash
sl-plot-metrics --input-csv results/my_results.csv                  # Fig. 3
sl-render-meshes --subset diffuse --samples 0 --baselines psp5 \
    --runs runs/diffuse_fixed_{1,3,5} runs/diffuse_learned_{1,3,5}  # Figs. 4-5
sl-patterns learned runs/diffuse_learned_3/best.pth --out-dir figures/patterns  # Figs. 6-7
sl-patterns fixed --count 3 --out-dir figures/fixed_patterns        # Fixed baseline
```

### Expected variation

Each configuration in the paper was trained once. Training now reads samples from
Parquet row groups of 64, so the shuffle differs from the original per-sample shuffle,
and runs are not bitwise reproducible. Expect retrained models to differ somewhat from
the paper's tables, most visibly where the differences between methods are small.

## Regenerating the dataset

This is only needed to render new data, and it requires the Mitsuba fork:

```bash
git submodule update --init --recursive
sudo apt install libhdf5-dev        # the LTM writer links the HDF5 C++ library
cmake -S external/mitsuba -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
# add "cuda_mono" to "enabled" in build/mitsuba.conf, then re-run the cmake command
ninja -C build
source build/setpath.sh             # puts the mitsuba module on PYTHONPATH
uv sync --extra generation
```

Then:

```bash
sl-abc-urls data/abc_raw            # lists the ABC OBJ chunk URLs; download them there
sl-abc-prepare -i data/abc_raw -o data/abc_meshes
sl-generate-preview --dataset-path data/abc_meshes             # sanity check
sl-generate -i data/abc_meshes -o data/abc_diffuse --object-material diffuse
sl-generate -i data/abc_meshes -o data/abc_metallic --object-material metallic
sl-split --dataset data/abc_diffuse
```

`sl-generate` renders one chunk of meshes per call (see `chunks.json`). The published
dataset rendered the first 100,000 ABC meshes and was packed into Parquet with the
scripts in its [Hugging Face repository](https://huggingface.co/datasets/leon3428/abc-light-transport-256/tree/main/scripts).
Object poses are random and the mesh-to-sample mapping was not recorded, so a
regenerated dataset will not match the published one.

## Tests

```bash
uv run pytest
```

## Acknowledgments

This work was supported by the Croatian Government project NPOO.C3.2.R3-I1.04.0033
"Scalable System of Cameras and Optical Filters for Industrial Applications (SKORPI)"
and the Croatian Science Foundation project DOK-2025-02-8664.
