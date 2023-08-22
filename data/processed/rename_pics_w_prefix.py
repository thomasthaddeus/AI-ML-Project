# import os
# import csv

# def rename_pictures_with_prefix(folder_path):
#     # List of common image extensions. You can add more if needed.
#     image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

#     # Iterate over each file in the folder
#     for filename in os.listdir(folder_path):
#         # Check if the file is an image
#         if any(filename.lower().endswith(ext) for ext in image_extensions):
#             # Rename the file by adding "ds" prefix
#             new_filename = f"ds_{filename}"
#             old_file_path = os.path.join(folder_path, filename)
#             new_file_path = os.path.join(folder_path, new_filename)
#             os.rename(old_file_path, new_file_path)

# Example usage:
# rename_pictures_with_prefix('/path/to/your/folder')
# rename_pictures_with_prefix('D:/Python/AI/AI-ML-Project/data/processed/ds01-1/ds01')


import os
import csv

def consolidate_yolo_to_csv(folder_path, output_csv_path):
    # List of all txt files in the folder
    txt_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.txt')]

    # Open the CSV file for writing
    with open(output_csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        # Write the header for the CSV file
        csvwriter.writerow(['filename', 'class_id', 'x_center', 'y_center', 'width', 'height'])

        # Iterate over each txt file and read its content
        for txt_file in txt_files:
            with open(os.path.join(folder_path, txt_file), 'r') as file:
                lines = file.readlines()
                for line in lines:
                    # Split the line to get individual values
                    class_id, x_center, y_center, width, height = line.strip().split()

                    # Write to CSV
                    csvwriter.writerow([txt_file, class_id, x_center, y_center, width, height])

# Example usage:
consolidate_yolo_to_csv('D:/Python/AI/AI-ML-Project/data/processed/ds01-1/ds01', 'D:/Python/AI/AI-ML-Project/data/processed/ds01-1/output.csv')
