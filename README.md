# AI-ML-Project

**Description:**
This repository is dedicated to building a computer vision model focused on detecting potholes. The model is trained on various datasets and is designed to assist urban municipalities in improving road maintenance and repair procedures.

**Key Features:**
- **Data Management:** Raw data is stored immutably, ensuring a fallback in case of processing errors. Scripts for data downloading, cleaning, and processing are organized systematically.
- **Model Training:** Scripts for model training and prediction are maintained separately for clarity.
- **Exploratory Data Analysis:** Jupyter notebooks are used for data exploration and are kept separate from the main source code.
- **Visualization:** Scripts for generating figures, plots, and visualizations are included.
- **Testing:** Unit tests are provided to ensure code functionality.
- **Dependencies:** A 'requirements.txt' file lists all project dependencies for easy replication.

**Dataset Sources:**

- annotated-potholes-dataset "Location 2 on Kaggle
- RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547 "RDD2022 - The multi-national Road Damage Dataset released through CRDDC'2022
- road-pothole-images-for-pothole-detection "Location 1 on Kaggle

**Workflow:**
1. **Data Collection and Annotation:** road imagery data collection includes various pothole types and conditions.
2. **Data Preprocessing:** Image resizing, normalization, and dataset splitting.
3. **Model Selection:** Evaluation of object detection models like Faster R-CNN, YOLO, and SSD.
4. **Model Training:** Training using TensorFlow with techniques like early stopping and data augmentation.
5. **Model Evaluation:** Performance assessment using precision, recall, and IoU metrics.
6. **Deployment:** Real-world deployment for real-time pothole detection.
7. **Post-deployment Monitoring:** Continuous performance monitoring and potential retraining.
8. **Iterative Improvement:** Feedback-based model enhancement.

## Dataset

1. [Kaggle][1]
2. [Another Kaggle Dataset][2]
3. [International Dataset][3]

**Languages Used:**
- Python (73.7%)
- Jupyter Notebook (21.9%)
- Shell (3.4%)
- Dockerfile (1.0%)

## License: [MIT](./LICENSE)

[1]: https://www.kaggle.com/datasets/sovitrath/road-pothole-images-for-pothole-detection "Location 1 on Kaggle"
[2]: https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset "Location 2 on Kaggle"
[3]: https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547 "RDD2022 - The multi-national Road Damage Dataset released through CRDDC'2022"
