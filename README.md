## Audio to Daw
Project based on the work of [Ben Hayes](https://github.com/ben-hayes/synth-permutations/tree/main)

## 06/10
- Working dataset generation with working parallelization for surge XT
- WIP: parameters selection
---
- Added surge_full and surge_simple params from original paper 2.6 s/it/core
> 4s/it/core -> 900 samples/h. w/ 40 cores -> 36k samples/h
---
- Fixed parameters (midi/velocity/min_loudness) to match the original experiment.
- Added config file for original experiment
> When using min_loudness = -55, about 1/3rd of the samples will be regenerated -> important for time calculations.
---
- Chunked dataset for performance (RAM consumption)
- Added spectrogram