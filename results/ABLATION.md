# speech-audio-profile-lab · ablation table

Backend × lane on a small utterance subset. Telephony lane = 8 kHz μ-law round-trip (+ noise for `noise_telecom`).

| Backend | Lane | n | WER ↓ | lang acc ↑ | JSON valid | p50 ms | p95 ms |
|---------|------|---|-------|------------|------------|--------|--------|
| wav2vec2-heads | clean | 8 | — | — | 1.000 | 596.5 | 1873.3 |
| wav2vec2-heads | telephony | 8 | — | — | 1.000 | 575.3 | 612.7 |
| whisper-base | clean | 8 | 1.000 | — | 1.000 | 3615.8 | 13018.0 |
| whisper-base | telephony | 8 | 1.000 | — | 1.000 | 3364.1 | 4473.1 |
| sensevoice-small | clean | 8 | 1.000 | 1.000 | 1.000 | 419.7 | 11276.3 |
| sensevoice-small | telephony | 8 | 1.000 | 1.000 | 1.000 | 433.0 | 653.0 |
