# OVH Permission Issues - Fixed ✅

## Problem

Jobs were crashing on OVH AI Training with permission errors:
```
PermissionError: [Errno 13] Permission denied: '/workspace/logs/train'
PermissionError: [Errno 13] Permission denied: '/workspace/.config/matplotlib'
```

## Root Cause

1. **Directory ownership mismatch**: Dockerfile created directories with `trainer` user, but OVH runs with different permissions
2. **Mounted volumes override filesystem**: The `/workspace/outputs` S3 volume mount replaced local directories
3. **Unmounted paths fail**: `/workspace/logs` wasn't mounted, so the container couldn't create it

## Solutions Applied

### 1. Dockerfile Changes (`Dockerfile`)

**Before:**
```dockerfile
RUN mkdir -p /workspace/logs /workspace/outputs /workspace/.config
USER trainer  # Switched to non-root user
ENV MPLCONFIGDIR=/workspace/.config/matplotlib
```

**After:**
```dockerfile
# Use /tmp for matplotlib (always writable)
ENV MPLCONFIGDIR=/tmp/matplotlib

# Create entrypoint to handle runtime directory creation
ENTRYPOINT ["/entrypoint.sh"]
```

**Why:** 
- Let OVH handle user permissions (don't force `USER trainer`)
- Use `/tmp` for matplotlib (always writable on any system)
- Entrypoint creates directories at runtime with correct permissions

### 2. New Paths Config (`configs/paths/ovh.yaml`)

**Created:**
```yaml
# Use mounted outputs volume for logs (avoid permission issues)
log_dir: ${paths.root_dir}/outputs/logs/
```

**Why:** 
- Logs go into the mounted `/workspace/outputs` volume (has write permissions)
- Avoids trying to create `/workspace/logs` (which fails)

### 3. Launch Script Update (`launch.sh`)

**Before:**
```bash
-- python src/train.py experiment="${EXPERIMENT_CONFIG}"
```

**After:**
```bash
-- python src/train.py experiment="${EXPERIMENT_CONFIG}" ++paths=ovh
```

**Why:** 
- Uses the OVH-specific paths config automatically
- `++` is a force override that takes precedence over experiment config defaults
- Required because experiments have `- override /paths: docker` in their defaults

## Verification

To test the fix:

```bash
# Rebuild and push image
./launch.sh surge/base

# Check job logs
./scripts/logs.sh <job-id>

# Should see:
# ✅ No permission errors
# ✅ Logs writing to /workspace/outputs/logs/
# ✅ Matplotlib using /tmp/matplotlib
```

## File Structure

```
/workspace/
├── datasets/          # Mounted S3 (read-only)
├── outputs/           # Mounted S3 (read-write)
│   ├── logs/          # ← Logs now go here
│   └── checkpoints/
├── src/               # Code from Docker image
└── configs/           # Config from Docker image
```

## Key Takeaways

1. **Don't create directories in Dockerfile** for paths that will be mounted
2. **Use mounted volumes** for all writable paths
3. **Use /tmp** for temp files (matplotlib, cache, etc.)
4. **Runtime entrypoints** handle dynamic directory creation
5. **Environment-specific configs** (e.g., `paths=ovh`) for cloud platforms
