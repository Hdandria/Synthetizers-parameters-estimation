# CLAP Distillation Test Run Instructions

The Feature Distillation using CLAP has been implemented. To verify it works, you can run a training experiment with the `distill` flag enabled.

## 1. Quick Verification (Fast Dev Run)

Run `train.py` with `distill=True` and `fast_dev_run=True` to quickly ensure the pipeline works (data loading, CLAP forward pass, and loss computation) without running a full epoch.

```bash
python src/train.py experiment=flow_multi/base_full data.distill=True model.distill=True trainer.fast_dev_run=True
```

**What to look for:**

- The process should start without errors.
- In the logs, look for `train/clap_loss`. It should be logged.
- The `SynthDataset` should report it is loading audio (`read_audio=True`).

Note: You need to set both `data.distill=True` (to load audio) and `model.distill=True` (to enable loss). Since we haven't created a dedicated config file for this yet, command line overrides are the best way.

## 2. Full Training Verification

To start a real training run with distillation enabling:

```bash
python src/train.py experiment=flow_multi/base_full data.distill=True model.distill=True model.distill_weight=10.0
```

You can adjust `model.distill_weight` to control the influence of the distillation loss. Default is 10.0.

## 3. Configuration Details

The following parameters have been added:

### `src/data/synth_datamodule.py`: `SynthDataModule`

- `distill` (bool, default=False): If True, forces `read_audio=True` for train/val/test datasets in `setup()`.

### `src/models/surge_flow_matching_module.py`: `SurgeFlowMatchingModule`

- `distill` (bool, default=False): If True, loads the CLAP model (frozen) and calculates the additional loss.
- `distill_weight` (float, default=10.0): Weight of the CLAP distillation loss added to the total loss.
