import cv2  # To get image dimensions

class ModifiedAnnotationParser:
    def __init__(self):
        # Default values (can be overridden by config)
        self.OUTPUT = './data/processed/annotations.json'
        self.FOLDER_IN = './data/processed'
        self.DS_TRAIN = "./data/processed/trn"
        self.DS_TEST = "./data/processed/test"
        self.XFILE_DIR = './data/processed/xml_data/'

    def load_config(self, config_file):
        try:
            config = configparser.ConfigParser()
            config.read(config_file)

            # Update attributes based on config
            if 'DEFAULT' in config:
                for key in config['DEFAULT']:
                    setattr(self, key.upper(), config['DEFAULT'][key])
        except Exception as e:
            print(f"Error loading config: {e}")

    def process_all_datasets(self):
        datasets = [self.DS_TRAIN, self.DS_TEST]
        for dataset in datasets:
            try:
                df = self.dataset_to_dataframe(dataset)
                json_filename = dataset.replace(".txt", ".json").replace(
                    "/processed/", "/processed/json/"
                )
                df.to_json(json_filename, orient="records", lines=True)
                print(f"Processed {dataset} and saved to {json_filename}")
            except Exception as e:
                print(f"Error processing dataset {dataset}: {e}")

    def dataset_to_dataframe(self, dataset):
        try:
            with open(file=dataset, mode="r", encoding="utf-8") as f:
                lines = f.readlines()
            data = [self.from_line(line) for line in lines]
            return pd.DataFrame(data)
        except Exception as e:
            print(f"Error converting dataset to dataframe: {e}")

    def from_line(self, line):
        # Placeholder method, to be implemented based on the line format.
        return {}

    def get_image_dimensions(self, image_path):
        try:
            img = cv2.imread(image_path)
            height, width, _ = img.shape
            return width, height
        except Exception as e:
            print(f"Error getting dimensions for {image_path}: {e}")
            return None, None

    def parse_yolo_line(self, line, image_dir):
        parts = line.strip().split()
        image_path = os.path.join(image_dir, parts[0])
        width, height = self.get_image_dimensions(image_path)
        if width and height:
            annotations = [{
                'class': 'pothole',
                'xmin': int((float(parts[1]) - float(parts[3])/2) * width),
                'ymin': int((float(parts[2]) - float(parts[4])/2) * height),
                'xmax': int((float(parts[1]) + float(parts[3])/2) * width),
                'ymax': int((float(parts[2]) + float(parts[4])/2) * height)
            }]
            return {
                'image_path': image_path,
                'width': width,
                'height': height,
                'annotations': annotations
            }
        else:
            return {}
