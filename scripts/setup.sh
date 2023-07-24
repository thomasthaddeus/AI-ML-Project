#!/bin/bash

# Define the directories to be created
directories=(
"data/raw"
"data/interim"
"data/processed"
"data/external"
"data/models"
"notebooks"
"src/data"
"src/features"
"src/models"
"src/visualization"
"tests"
)

# Create the directories
for dir in "${directories[@]}"; do
  mkdir -p "$dir"
done

# Initialize README
echo "# My ML Project with Computer Vision

This project uses the following directory structure:" > README.md

for dir in "${directories[@]}"; do
  echo "- $dir" >> README.md
done

# Initialize requirements.txt
echo "# Generic Computer Vision Libraries
numpy
scipy
matplotlib
pandas
opencv-python
pillow
scikit-learn
tensorflow
keras
torch
torchvision" > requirements.txt

echo "Project initialized successfully!"
