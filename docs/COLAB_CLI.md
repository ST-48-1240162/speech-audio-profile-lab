# Colab CLI

Run install, tests, and ablation on a **remote Colab VM** via [google-colab-cli](https://github.com/googlecolab/google-colab-cli).

## Install (once)

```bash
uv tool install google-colab-cli
colab auth
```

## Verify (recommended)

From a local clone:

```bash
git clone https://github.com/ST-48-1240162/speech-audio-profile-lab.git
cd speech-audio-profile-lab
./scripts/colab_run_bundle.sh
```

Uploads a tarball, runs `unittest`, mock ablation, prints sample AudioProfile JSON.

## Full ablation (GPU)

```bash
./scripts/colab_run_ablation.sh
CORPUS=librispeech MAX_UTT=100 ./scripts/colab_run_ablation.sh
```

Downloads `results/ABLATION.md` and `results/ablation_table.csv` when finished.

## Git clone on VM

```bash
colab run scripts/colab_verify.py -- --source git
colab run --gpu T4 scripts/colab_verify.py -- --source git --real-hf
```

## Drive sync (optional)

If the repo lives on Google Drive:

```bash
colab run --timeout 600 scripts/colab_verify.py -- \
  --source drive --drive-subpath "path/under/MyDrive/speech-audio-profile-lab"
```

## Env overrides

| Variable | Default | Purpose |
|----------|---------|---------|
| `PROFILE_LAB_SOURCE` | auto (`bundle` if tarball present, else `git`) | `bundle`, `git`, or `drive` |
| `PROFILE_LAB_DRIVE_SUBPATH` | `speech-audio-profile-lab` | Path under `MyDrive` when using `--source drive` |
| `COLAB_RUN_TIMEOUT` | `900` | Exec timeout (seconds) |
| `COLAB_GPU` | `T4` | GPU for `colab_run_ablation.sh` |
