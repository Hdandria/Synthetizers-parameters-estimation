# Remote training on OVH GPU server

## First time setup

```bash
# 1. SSH and install prerequisites
ssh <user>@<server-ip>

curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
sudo apt-get install -y unzip
unzip awscliv2.zip
sudo ./aws/install

sudo apt update
sudo apt install -y nvidia-driver-580-server
sudo reboot

# 2. After reboot, copy launcher script and .env
mkdir -p ~/synth-launch/scripts
nano ~/synth-launch/scripts/launch_flow_multi.sh  # Paste script content
nano ~/synth-launch/.env  # Paste credentials below
chmod +x ~/synth-launch/scripts/launch_flow_multi.sh

# 3. Pull Docker image
docker pull benjamindupuis/synth-param-estimation:latest
```

**.env file template:**
```bash
WANDB_API_KEY=your_wandb_key
PROJECT_ROOT=/workspace
S3_BUCKET=uniform-datasets
S3_PLUGIN_PATH=/plugins
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=gra
AWS_ENDPOINT_URL=https://s3.gra.io.cloud.ovh.net
```

---

## Every time you launch experiments

```bash
# 1. SSH to server
ssh <user>@<server-ip>

# 2. Update Docker image (if code/configs changed)
docker pull benjamindupuis/synth-param-estimation:latest

# 3. Update launcher script if needed
cd ~/synth-launch
nano scripts/launch_flow_multi.sh  # Optional: modify EXPERIMENTS list

# 4. Run experiments
set -a; source .env; set +a
bash ./scripts/launch_flow_multi.sh
```

Experiments run sequentially. Monitor GPU: `nvidia-smi -l 5`  
Check W&B: https://wandb.ai/paindespistes-t-l-com-paris/synth-prediction

```bash
# Bonus: Clean up old containers
docker stop $(docker ps -q)
docker rm $(docker ps -a -q)
```

```bash
# Bonus: Check on running containers
docker ps
docker logs <container_id>
```
---

## To rebuild and push new Docker image (local machine)

```bash
docker build -t synth-param-estimation:latest .
docker tag synth-param-estimation:latest benjamindupuis/synth-param-estimation:latest
docker push benjamindupuis/synth-param-estimation:latest
```