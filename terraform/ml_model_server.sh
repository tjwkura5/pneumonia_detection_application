#!/bin/bash

# Redirect stdout and stderr to a log file
exec > /var/log/user-data.log 2>&1

# Ensure all commands are run with superuser privileges
echo "Running as user: $(whoami)"

# Write the .pem file
echo "${my_key}" > /home/ubuntu/.ssh/mykey.pem
chmod 400 /home/ubuntu/.ssh/mykey.pem
sudo chown ubuntu:ubuntu /home/ubuntu/.ssh/mykey.pem

# ************* Installing Node Exporter *****************************

# Install necessary packages as root
apt-get update && apt-get upgrade -y
apt-get install -y wget

# Download and install Node Exporter as root
wget https://github.com/prometheus/node_exporter/releases/download/v1.6.0/node_exporter-1.6.0.linux-amd64.tar.gz
tar xvfz node_exporter-1.6.0.linux-amd64.tar.gz
sudo mv node_exporter-1.6.0.linux-amd64/node_exporter /usr/local/bin/
rm -rf node_exporter-1.6.0.linux-amd64*

# Create a systemd service for Node Exporter to run as 'ubuntu'
cat <<EOL | sudo tee /etc/systemd/system/node_exporter.service
[Unit]
Description=Node Exporter

[Service]
User=ubuntu
ExecStart=/usr/local/bin/node_exporter

[Install]
WantedBy=multi-user.target
EOL

# Start and enable Node Exporter as root
sudo systemctl daemon-reload
sudo systemctl start node_exporter
sudo systemctl enable node_exporter

# Update and install basic packages
apt-get install -y python3-pip 
apt-get install -y python3-venv 
apt-get install -y git 
apt-get install -y build-essential 
apt-get install -y unzip

# Install NVIDIA drivers and CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt update
apt -y install cuda-drivers
apt install -y cuda-toolkit-12-6 cudnn-cuda-12

# Set up CUDA environment variables for immediate use in this script
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Set up CUDA environment variables
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /home/ubuntu/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> /home/ubuntu/.bashrc
#source ~/.bashrc

# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Download dataset from S3
mkdir -p /home/ubuntu/chest_xray
aws s3 cp s3://x-raysbucket/chest_xray/ /home/ubuntu/chest_xray --recursive --no-sign-request
sudo chown -R ubuntu:ubuntu /home/ubuntu/chest_xray

# Clone repository
cd /home/ubuntu
git clone https://github.com/tjwkura5/pneumonia_detection_application.git /home/ubuntu/CNN_deploy

# Set permissions on the repo
sudo chown -R ubuntu:ubuntu /home/ubuntu/CNN_deploy