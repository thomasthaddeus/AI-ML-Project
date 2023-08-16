# A Fully Annotated Image Dataset for Pothole Detection

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



- The directory `annotated-images` contains the images having pothole and their respective annotations (as XML file).
- The file `splits.json` contains the annotation filenames (.xml) of the **training** (80%) and **test** (20%) dataset in following format---

```javascript
{
  "train": ["img-110.xml", "img-578.xml", "img-455.xml", ...],
  "test": ["img-565.xml", "img-498.xml", "img-143.xml", ...]
}
```

## Directory 

```bash
Folder PATH listing for volume Windows
Volume serial number is 520B-286B
C:.
\---RDD2022
    +---China_Drone
    |   \---train
    |       +---annotations
    |       |   \---xmls
    |       \---images
    +---China_MotorBike
    |   +---test
    |   |   \---images
    |   \---train
    |       +---annotations
    |       |   \---xmls
    |       \---images
    +---Czech
    |   +---test
    |   |   \---images
    |   \---train
    |       +---annotations
    |       |   \---xmls
    |       \---images
    +---India
    |   +---test
    |   |   \---images
    |   \---train
    |       +---annotations
    |       |   \---xmls
    |       \---images
    +---Japan
    |   +---test
    |   |   \---images
    |   \---train
    |       +---annotations
    |       |   \---xmls
    |       \---images
    +---Norway
    |   +---test
    |   |   \---images
    |   \---train
    |       +---annotations
    |       |   \---xmls
    |       \---images
    \---United_States
        +---test
        |   \---images
        \---train
            +---annotations
            |   \---xmls
            \---images
```

## Important Notes

- The `<path>` tag in the xml annotations may not match with your environment, therefore, consider `<filename>` tag.
- The file `splits.json` was generated randomly taking 20% of the total dataset as **test** and the rest as **train**.

## Dataset Description

Pothole dataset compiled at Electrical and Electronic Department, Stelllenbosch University, 2015

The entire dataset consists of two different sets, one was considered to be simple and the other more complex.
These datasets do share some files and there are a few instances where two different images would have the same name.
Therefore, the appropriate measures need to be taken if the data is combined into one larger dataset.

Each of the datasets contain folders containing the training (positive and negative) images as well as a set of positive test images.

## References

Please cite the following papers if you wish to use the dataset:
[1] S. Nienaber, M.J. Booysen, R.S. Kroon, �Detecting potholes using simple image processing techniques and real-world footage�, SATC, July 2015, Pretoria, South Africa.
[2] S. Nienaber, R.S. Kroon, M.J. Booysen , �A Comparison of Low-Cost Monocular Vision Techniques for Pothole Distance Estimation�, IEEE CIVTS, December 2015, Cape Town, South Africa.

The pothole detection task was found to be much easier if only the region in the image that contained the road was cropped and then used in detection such
as in the example pseudocode below.

%%%Pseudocode
%%%Crop rectangular region

*Downsample input image with a factor of two
*Rect roadSection(0,(1272/2+100)-150,(3680/2),(500/2+50))  //Example values


