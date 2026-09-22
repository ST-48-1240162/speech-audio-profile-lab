# speech-audio-profile-lab

Voice-agent **turn metadata** as versioned **AudioProfile JSON**, with three analysis backends and telephony degradation lanes.

Compare backends on **clean vs telephony** audio with a small ablation runner (`scripts/run_ablation.py` → `results/ABLATION.md`).

## Quick start (Colab CLI)

```bash
uv tool install google-colab-cli
colab auth   # once
./scripts/colab_run_bundle.sh          # unit tests + mock ablation
./scripts/colab_run_ablation.sh        # full ablation (GPU T4 default)
```

See **`docs/COLAB_CLI.md`**.

## API

| Route | Params | Output |
|-------|--------|--------|
| `POST /profile` | `backend`, `lane`, WAV file | AudioProfile v1 |
| `GET /demo/profile` | `backend`, `lane` | Synthetic → AudioProfile |
| `POST /analyze` | WAV | Legacy flat dict |
| `GET /demo` | — | Legacy chain dict |

### Backends (`backend=`)

| ID | Role |
|----|------|
| `wav2vec2-heads` | Wav2Vec2 embedding + MFCC/sklearn proxies |
| `whisper-base` | Whisper ASR (`openai/whisper-base`) |
| `sensevoice-small` | FunASR SenseVoice → lang + ASR |

### Lanes (`lane=`)

| ID | Transform |
|----|-----------|
| `clean` | Passthrough |
| `telephony` | 8 kHz μ-law round-trip |
| `noise_telecom` | telephony + additive noise |

## Ablation

```bash
python scripts/run_ablation.py --mock --max-utt 8     # CI / smoke
python scripts/run_ablation.py --corpus synthetic --max-utt 100
python scripts/run_ablation.py --corpus librispeech --max-utt 100  # GPU
```

Outputs: `results/ABLATION.md`, `results/ablation_table.csv`.

## Agent orchestration

`src/audio_ml_jd_lab/graph.py` — **LangGraph** `StateGraph` with checkpointed stages:

`preprocess → infer → validate → package` (mirrors field-audio-tools stage graph).

```python
from audio_ml_jd_lab.graph import run_profile_agent
```

## Stack

Hugging Face Transformers · FunASR SenseVoice · LangChain · **LangGraph** · scikit-learn · FastAPI · jiwer

## Docker

```bash
docker build -f deploy/Dockerfile -t speech-audio-profile-lab .
docker run --rm -p 8080:8080 speech-audio-profile-lab
```
