import pandas as pd

def parse_line(line):
    """Parse a single line from the dataset."""
    parts = line.strip().split()
    image_path = parts[0].replace('.bmp', '.jpg')  # Adjust the file extension
    num_boxes = int(parts[1])
    boxes = [{'x': int(parts[i]), 'y': int(parts[i+1]), 'width': int(parts[i+2]), 'height': int(parts[i+3])}
             for i in range(2, 2 + 4 * num_boxes, 4)]
    return {
        'image_path': image_path,
        'boxes': boxes
    }

def read_dataset(dataset):
    """Read the dataset file and return a DataFrame."""
    with open(file=dataset, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    data = [parse_line(line) for line in lines]
    return pd.DataFrame(data)

# Usage
df = read_dataset('')
print(df.head())
