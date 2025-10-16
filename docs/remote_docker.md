```bash
# Locally
docker build -t synth-param-estimation:latest .
```

```bash
# Tag for Docker Hub
docker tag synth-param-estimation:latest benjamindupuis/synth-param-estimation:latest

# Push to Docker Hub
docker push benjamindupuis/synth-param-estimation:latest
```
   
server setup:

- connect to server vis ssh

```bash
# Install docker if not installed
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

configuration:

```bash
cat > .env << EOF
# .env
WANDB_API_KEY=wandb_api_key_here
PROJECT_ROOT=/workspace
S3_BUCKET=uniform-100k
S3_DATASET_PATH=/datasets
S3_PLUGIN_PATH=/plugins
AWS_ACCESS_KEY_ID=ovh_access_key_here
AWS_SECRET_ACCESS_KEY=ovh_secret_key_here
AWS_DEFAULT_REGION=gra
AWS_ENDPOINT_URL=https://s3.gra.io.cloud.ovh.netEOF
```

checker comment configuer s3 (awscli ?)
```bash
pip install ovh
ovh config
# Enter OVH credentials
```	

```bash
docker pull benjamindupuis/synth-param-estimation:latest
docker tag benjamindupuis/synth-param-estimation:latest synth-param-estimation:latest
```

- Launch experiments (-d detached)
```bash
docker run -d \
  --name synth-param-estimation-run \
  --gpus all \
  --env-file .env \
  benjamindupuis/synth-param-estimation:latest \
  bash -c "./scripts/launch_flow_multi.sh"
```

```bash
# Check containers
docker ps

# Check W&B dashboard
# https://wandb.ai/your-entity/synth-prediction
```