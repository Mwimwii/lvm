#!/bin/bash
# Run inside a Lightning CPU Job to build & push the LVM Docker image.
# Requires env vars: DOCKER_USER, DOCKER_PASS
set -e

echo "=== Installing Docker CE ==="
apt-get update -qq && apt-get install -y -qq ca-certificates curl git
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" > /etc/apt/sources.list.d/docker.list
apt-get update -qq && apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin

echo "=== Starting Docker daemon ==="
dockerd &
sleep 5
until docker info >/dev/null 2>&1; do sleep 2; done
echo "Docker daemon ready."

echo "=== Cloning lvm repo ==="
git clone https://github.com/Mwimwii/lvm.git /tmp/lvm
cd /tmp/lvm

echo "=== Building image (~15-20 min) ==="
docker build -t "${DOCKER_USER}/lvm-processor:latest" .

echo "=== Pushing to Docker Hub ==="
echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin
docker push "${DOCKER_USER}/lvm-processor:latest"

echo "=== Done ==="
