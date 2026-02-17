#!/bin/bash
set -e

# =============================================================================
# Reshard an existing Vital dataset on S3
#
# Downloads each shard, splits it into smaller sub-shards, uploads the new
# shards with temporary names, then deletes the old shards and renames the
# new ones to the conventional shard_0.h5, shard_1.h5, ... naming scheme.
#
# Usage:
#   ./scripts/reshard_vital_dataset.sh [OPTIONS]
#
# Options (override defaults via environment or flags):
#   --s3-prefix <prefix>    S3 prefix of the dataset (default: datasets/vital_uniform_1M)
#   --first-shard <n>       Index of the first old shard (default: 0)
#   --last-shard <n>        Index of the last old shard, inclusive (default: 9)
#   --splits <n>            How many sub-shards to create per old shard (default: 2)
#   --work-dir <dir>        Local working directory (default: /tmp/reshard_work)
# =============================================================================

# Locate .env file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"

# Load environment variables for AWS CLI commands
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from $ENV_FILE"
    set -a
    source <(grep -v '^#' "$ENV_FILE" | grep -v '^$' | sed 's/#.*$//')
    set +a
else
    echo "Error: .env file not found at $ENV_FILE"
    exit 1
fi

# --------------- Defaults ---------------
S3_PREFIX="datasets/vital_uniform_1M"
FIRST_SHARD=0
LAST_SHARD=9
SPLITS=2
WORK_DIR="/tmp/reshard_work"

# --------------- Parse CLI args ---------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --s3-prefix)   S3_PREFIX="$2"; shift 2;;
        --first-shard) FIRST_SHARD="$2"; shift 2;;
        --last-shard)  LAST_SHARD="$2"; shift 2;;
        --splits)      SPLITS="$2"; shift 2;;
        --work-dir)    WORK_DIR="$2"; shift 2;;
        -h|--help)
            sed -n '/^# Usage/,/^# =====/p' "$0" | head -n -1
            exit 0;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

# S3 Configuration (from .env)
S3_ENDPOINT="${AWS_ENDPOINT_URL}"
S3_REGION="${AWS_DEFAULT_REGION}"
S3_BASE="s3://${S3_BUCKET}/${S3_PREFIX}"

NUM_OLD_SHARDS=$(( LAST_SHARD - FIRST_SHARD + 1 ))
TOTAL_NEW_SHARDS=$(( NUM_OLD_SHARDS * SPLITS ))

echo "=============================================="
echo "  Reshard Vital Dataset"
echo "=============================================="
echo "S3 location:      $S3_BASE"
echo "Old shards:        $NUM_OLD_SHARDS  (shard_${FIRST_SHARD}.h5 .. shard_${LAST_SHARD}.h5)"
echo "Splits per shard:  $SPLITS"
echo "New shard count:   $TOTAL_NEW_SHARDS"
echo "Work directory:    $WORK_DIR"
echo "=============================================="

mkdir -p "$WORK_DIR"

# Helper: run aws through uv
aws_s3() {
    uv run aws s3 "$@" --endpoint-url "$S3_ENDPOINT" --region "$S3_REGION"
}

# =========================================================================
# Phase 1: Download each old shard, split, upload sub-shards with temp names
# =========================================================================
echo ""
echo "=== Phase 1: Split and upload ==="

NEW_SHARD_IDX=0

for OLD_IDX in $(seq $FIRST_SHARD $LAST_SHARD); do
    OLD_FILE="$WORK_DIR/shard_${OLD_IDX}.h5"
    S3_OLD="$S3_BASE/shard_${OLD_IDX}.h5"

    echo ""
    echo "--- Processing old shard $OLD_IDX ---"

    # Download
    echo "Downloading $S3_OLD ..."
    aws_s3 cp "$S3_OLD" "$OLD_FILE"

    # Split into $SPLITS sub-shards using Python
    echo "Splitting into $SPLITS sub-shards ..."
    uv run python - "$OLD_FILE" "$WORK_DIR" "$SPLITS" "$NEW_SHARD_IDX" <<'PYEOF'
import sys, h5py, hdf5plugin, numpy as np, math

old_path  = sys.argv[1]
work_dir  = sys.argv[2]
n_splits  = int(sys.argv[3])
start_idx = int(sys.argv[4])

with h5py.File(old_path, "r") as old_f:
    # Discover all dataset keys
    ds_keys = [k for k in old_f.keys() if isinstance(old_f[k], h5py.Dataset)]
    n_samples = old_f[ds_keys[0]].shape[0]
    samples_per_split = math.ceil(n_samples / n_splits)

    # Collect attributes from the audio dataset (if present)
    audio_attrs = {}
    if "audio" in old_f:
        for attr_name, attr_val in old_f["audio"].attrs.items():
            audio_attrs[attr_name] = attr_val

    for split_i in range(n_splits):
        lo = split_i * samples_per_split
        hi = min(lo + samples_per_split, n_samples)
        actual = hi - lo
        new_idx = start_idx + split_i
        out_path = f"{work_dir}/new_shard_{new_idx}.h5"

        print(f"  Writing {out_path}  (samples {lo}..{hi-1}, count={actual})")
        with h5py.File(out_path, "w") as new_f:
            for key in ds_keys:
                src_ds = old_f[key]
                new_shape = (actual,) + src_ds.shape[1:]
                new_ds = new_f.create_dataset(
                    key,
                    shape=new_shape,
                    dtype=src_ds.dtype,
                    **hdf5plugin.Blosc2(),
                )
                # Copy data in chunks to limit memory usage
                CHUNK = 1000
                for c_start in range(0, actual, CHUNK):
                    c_end = min(c_start + CHUNK, actual)
                    new_ds[c_start:c_end] = src_ds[lo + c_start : lo + c_end]

                # Preserve audio attrs on the audio dataset
                if key == "audio":
                    for attr_name, attr_val in audio_attrs.items():
                        new_ds.attrs[attr_name] = attr_val

print("Split complete.")
PYEOF

    # Remove downloaded old shard to free space
    rm "$OLD_FILE"
    echo "Removed local copy of old shard $OLD_IDX."

    # Upload each new sub-shard with temp name
    for SPLIT_I in $(seq 0 $((SPLITS - 1))); do
        SUB_IDX=$((NEW_SHARD_IDX + SPLIT_I))
        SUB_FILE="$WORK_DIR/new_shard_${SUB_IDX}.h5"
        S3_NEW="$S3_BASE/new_shard_${SUB_IDX}.h5"

        echo "Uploading new_shard_${SUB_IDX}.h5 ..."
        aws_s3 cp "$SUB_FILE" "$S3_NEW"
        rm "$SUB_FILE"
        echo "✓ Uploaded and cleaned up new_shard_${SUB_IDX}.h5"
    done

    NEW_SHARD_IDX=$((NEW_SHARD_IDX + SPLITS))
done

echo ""
echo "=== Phase 1 complete: all new shards uploaded with temp names ==="

# =========================================================================
# Phase 2: Delete all old shards
# =========================================================================
echo ""
echo "=== Phase 2: Delete old shards ==="

for OLD_IDX in $(seq $FIRST_SHARD $LAST_SHARD); do
    S3_OLD="$S3_BASE/shard_${OLD_IDX}.h5"
    echo "Deleting $S3_OLD ..."
    aws_s3 rm "$S3_OLD"
done

echo "✓ All old shards deleted."

# =========================================================================
# Phase 3: Rename new_shard_X.h5 -> shard_X.h5
# =========================================================================
echo ""
echo "=== Phase 3: Rename new shards ==="

for IDX in $(seq 0 $((TOTAL_NEW_SHARDS - 1))); do
    S3_SRC="$S3_BASE/new_shard_${IDX}.h5"
    S3_DST="$S3_BASE/shard_${IDX}.h5"
    echo "Renaming new_shard_${IDX}.h5 -> shard_${IDX}.h5 ..."
    aws_s3 mv "$S3_SRC" "$S3_DST"
done

echo ""
echo "=============================================="
echo "  Reshard complete!"
echo "  $NUM_OLD_SHARDS old shards (shard_${FIRST_SHARD}..shard_${LAST_SHARD}) -> $TOTAL_NEW_SHARDS new shards"
echo "  Location: $S3_BASE/shard_0.h5 .. shard_$((TOTAL_NEW_SHARDS - 1)).h5"
echo "=============================================="

# Cleanup work dir
rmdir "$WORK_DIR" 2>/dev/null || true
