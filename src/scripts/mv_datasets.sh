#!/bin/bash

# Create the main Train and Test data folders
mkdir -p "Train data"
mkdir -p "Test data"

# Move all 'Train data' folders from each 'Dataset 2 (Complex)-*' directory to the main 'Train data' folder
for train_dir in Dataset\ 2\ \(Complex\)-*/Dataset\ 2\ \(Complex\)/Train\ data/*; do
    mv "$train_dir" "Train data/"
done

# Move all 'Test data' folders from each 'Dataset 2 (Complex)-*' directory to the main 'Test data' folder
for test_dir in Dataset\ 2\ \(Complex\)-*/Dataset\ 2\ \(Complex\)/Test\ data/*; do
    mv "$test_dir" "Test data/"
done

# If there's a standalone 'Negative data' directory at the root, move it to the 'Train data' folder
if [ -d "Negative data" ]; then
    mv "Negative data" "Train data/"
fi

# Remove all 'Dataset 2 (Complex)-*' directories
rm -rf Dataset\ 2\ \(Complex\)-*

echo "Data has been consolidated into 'Train data' and 'Test data' folders."
