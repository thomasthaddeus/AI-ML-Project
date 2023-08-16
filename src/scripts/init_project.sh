#!/bin/bash

# Call the first script to initialize project structure
./setup.sh

# Define the Python files to be created
declare -A py_files=(
["src/data/make_dataset.py"]="# TODO: This script is used to download, generate, clean, and process data"
["src/features/build_features.py"]="# TODO: This script is used to turn raw data into features for modeling"
["src/models/predict_model.py"]="# TODO: This script is used to make predictions using trained models"
["src/models/train_model.py"]="# TODO: This script is used to train models"
["src/visualization/visualize.py"]="# TODO: This script is used to create exploratory and results oriented visualizations"
["tests/test_basic.py"]="# TODO: This file should contain test cases for your project"
)

# Create the Python files
for file in "${!py_files[@]}"; do
  echo "${py_files[$file]}" > "$file"
done

# Initialize Dockerfile
echo "# TODO: Specify the Docker image and setup here" > Dockerfile

# Initialize setup.py
echo "# TODO: This file makes your project pip installable" > setup.py

echo "Python files and Dockerfile initialized successfully!"
