#!/bin/bash

# Create datasets directory if it doesn't exist
mkdir -p datasets

# Change to the datasets directory
cd datasets || exit

# Download the dataset from Kaggle
echo "Downloading weather dataset..."
curl -L -o weather-dataset.zip \
  https://www.kaggle.com/api/v1/datasets/download/jehanbhathena/weather-dataset

# Check if download was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to download the dataset"
    exit 1
fi

# Extract the dataset
echo "Extracting dataset..."
unzip -q weather-dataset.zip -d .

# Remove the zip file after extraction
rm weather-dataset.zip

echo "Dataset downloaded and extracted successfully to the 'datasets' directory."