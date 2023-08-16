# Datset Notes

## Dataset-1: A Fully Annotated Image Dataset for Pothole Detection

- The directory `annotated-images` contains the images having pothole and their respective annotations (as XML file).
- The file `splits.json` contains the annotation filenames (.xml) of the **training** (80%) and **test** (20%) dataset in following format---

```javascript
{
  "train": ["img-110.xml", "img-578.xml", "img-455.xml", ...],
  "test": ["img-565.xml", "img-498.xml", "img-143.xml", ...]
}
```

### Important Notes

- The `<path>` tag in the xml annotations may not match with your environment, therefore, consider `<filename>` tag.
- The file `splits.json` was generated randomly taking 20% of the total dataset as **test** and the rest as **train**.
- The dataset is collected from the web and annotated manually. The images are of diverse regions and the annotations are done by multiple annotators. Therefore, the annotations may not be perfect.

## Dataset-2: Pothole Detection Dataset

Here we provide a dataset of 1,243 pothole images which have been annotated as per the YOLO labeling format.

We provide the image and the corresponding labeling in the dataset. We have collected the images of potholes from the web consisting of diverse regions. Here is the directory structure for the dataset:

### Dataset Directory Structure

```plaintext
Pothole Dataset
├── README.md
├── img-1.jpg
├── img-1.txt
├── img-2.jpg
├── img-2.txt
├── ...
├── img-1243.jpg
└── img-1243.txt
```

### Citation

If you consider using our work then please cite using:

```openbib
@INPROCEEDINGS{9290547,  author={Chitale, Pranjal A. and Kekre, Kaustubh Y. and Shenai, Hrishikesh R. and Karani, Ruhina and Gala, Jay P.},  booktitle={2020 35th International Conference on Image and Vision Computing New Zealand (IVCNZ)},   title={Pothole Detection and Dimension Estimation System using Deep Learning (YOLO) and Image Processing},   year={2020},  volume={},  number={},  pages={1-6},  doi={10.1109/IVCNZ51579.2020.9290547}}
```

## Dataset-3: Simplex

```xml
<image_path> <number_of_boxes> <x1> <y1> <width1> <height1> <x2> <y2> <width2> <height2> ...
```

## Dataset-4: Complex
