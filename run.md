# Dataset Generation and Training Commands

## 1. Generate Train/Val/Test Dataset Splits

### Generate Training Set (16,000 samples)
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
    --param_spec surge_simple \
    --sample_batch_size 64 \
    --num_workers 20
```

### Generate Validation Set (2,000 samples)
```bash
LOGURU_LEVEL=ERROR python src/data/vst/generate_vst_dataset.py \
    datasets/experiment_1/val.h5 \
    2000 \
    --plugin_path "vsts/Surge XT.vst3" \
    --preset_path "presets/surge-simple.vstpreset" \
    --sample_rate 44100 \
    --channels 2 \
    --velocity 100 \
    --signal_duration_seconds 4.0 \
    --min_loudness -55.0 \
    --param_spec surge_simple \
    --sample_batch_size 64 \
    --num_workers 20
```

### Generate Test Set (2,000 samples)
```bash
LOGURU_LEVEL=ERROR python src/data/vst/generate_vst_dataset.py \
    datasets/experiment_1/test.h5 \
    2000 \
    --plugin_path "vsts/Surge XT.vst3" \
    --preset_path "presets/surge-simple.vstpreset" \
    --sample_rate 44100 \
    --channels 2 \
    --velocity 100 \
    --signal_duration_seconds 4.0 \
    --min_loudness -55.0 \
    --param_spec surge_simple \
    --sample_batch_size 64 \
    --num_workers 20
```

## 2. Compute Dataset Statistics

```bash
python scripts/get_dataset_stats.py datasets/experiment_1/train.h5
python scripts/get_dataset_stats.py datasets/experiment_1/val.h5
python scripts/get_dataset_stats.py datasets/experiment_1/test.h5
```

## 3. Train Model on Generated Dataset

```bash
python src/train.py \
    data=surge \
    model=surge_ffn \
    data.dataset_root=datasets/experiment_1 \
    data.batch_size=128 \
    data.num_workers=11 \
    trainer.devices=[2] \
    trainer.accelerator=gpu
```

## 4. Evaluate Model

```bash
python src/eval.py \
    data=surge \
    model=surge_ffn \
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
- Generated datasets will be saved to `datasets/experiment_1/{train,val,test}.h5`
- Training configuration uses the surge datamodule with default parameters
- **Expected Performance**: Should be ~20x faster than single-threaded generation

- **Required Files**: The SurgeDataModule expects `train.h5`, `val.h5`, and `test.h5` files in the dataset root directory
- **Statistics**: Run `get_dataset_stats.py` on each split to compute mean/std for normalization

