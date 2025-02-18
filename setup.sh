#!/bin/bash

# Clone repository
git clone https://github.com/shirley-wu/text_to_table.git

# Change directory
cd /content/

# Download necessary data
gdown "https://drive.google.com/uc?id=1zTfDFCl1nf_giX7IniY5WbXi9tAuEHDn"

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Extract pre-trained model (BART)
tar -xzvf /content/drive/MyDrive/bart.base.tar.gz -C /content/

# Unzip dataset
unzip /content/data_release.zip -d /content/text_to_table/

# Install Python 3.8 and required dependencies
sudo apt-get update -y
sudo apt-get install -y python3.8 python3.8-distutils python3.8-dev

# Set Python 3.8 as the default
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
update-alternatives --config python3

# Install pip
sudo apt-get install python3-pip
python3 -m pip install --upgrade pip --user

# Install virtualenv and create a Python 3.8 environment
pip install virtualenv
virtualenv -p python3.8 py38_env

# Install specific pip version
pip install pip==23.2.1

# Install dependencies from the project
pip install -r /content/text_to_table/requirements.txt

# Clone Fairseq v0.10.2
git clone --branch v0.10.2 https://github.com/facebookresearch/fairseq.git

# Install Fairseq
cd /content/fairseq/
pip install --editable .
cd /content

# Move new datasets and preprocess scripts to the project folder
mv /content/new_datasets/* /content/text_to_table/data_preprocessing/
mv /content/new_preprocess/* /content/text_to_table/data_preprocessing/

# Make script executable
chmod +x setup.sh

echo "Setup completed successfully!"
