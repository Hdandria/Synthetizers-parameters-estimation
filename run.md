# Dataset Generation and Training Commands

## 1. Generate 20,000 Sample Dataset

```bash
LOGURU_LEVEL=ERROR python src/data/vst/generate_vst_dataset.py \
    datasets/experiment_1/train.h5 \
    20000 \
    --plugin_path "vsts/Surge XT.vst3" \
    --preset_path "presets/surge-simple.vstpreset" \
    --sample_rate 44100 \
    --channels 2 \
    --velocity 100 \
    --signal_duration_seconds 4.0 \
    --min_loudness -55.0 \
    --param_spec surge_xt \
    --sample_batch_size 64 \
    --num_workers 20
```

## 2. Train Model on Generated Dataset

```bash
python src/train.py \
    data=surge \
    data.dataset_root=datasets/experiment_1 \
    data.batch_size=128 \
    data.num_workers=11 \
    trainer.devices=[2] \
    trainer.accelerator=gpu
```

## Notes

- **High-Performance Generation**: Each worker writes to its own file, then files are merged
- Dataset generation uses 20 worker processes (half of your 40 cores)
- Training uses GPU with ID 2
- Only loguru errors and tqdm progress bars will be shown in console
- Generated dataset will be saved to `datasets/experiment_1/train.h5`
- Training configuration uses the surge datamodule with default parameters
- **Expected Performance**: Should be ~20x faster than single-threaded generation
