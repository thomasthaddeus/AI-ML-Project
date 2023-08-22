# AI-ML-Project

## **MMD Diagram**

[![](https://mermaid.ink/img/pako:eNrtmVGvmjAUx79K08U9qYEWdPCwREGzJVv2cB-WLLx0UrQZFgNl9zrjdx_QcmfViyS7jmQhGuWc_uR_oKek_3iAqySk0IWDwSHgADDOhAuqwwCKDd3SAKq4zPACzk4yZW5maokyFbE4LpMBfENIZJAwgMPT8UykyQ-qCIzx1eHRIwvFRkIm2j0FsGaOwxN11KQeOX8t_bIy7kzZ6kzZ7kp53thlUUjup4w6U8adKVtdKXvNT5N7KqPOlHFnylZXyn5n69nvbD37na3nReM0R_d7bC-bhJ1V9LrC8qDWDyAN1xfblZBGJI_FRV2a9GQyaSetMSRNk8cNLbc7brVZSrckvn5fvlK23ogMvAVzRrKzIi_KiV6zHI2JyXeqpuOypKs7gJVgP4lgCQfLnK_Kgxu3MvoHtV-t6rwtyq_io3wPBgEPeCaIoD4j65RsJVNlivN95LtcgE9kT9MAApKBYpN7A0C3AHwLsG4BdnXdGvSBhSHlkgKm5OZmCwa1YHALxmpgkGQ8swWDWjC4BdNUD5aMb7ZgUAvmrJ4vuTibsYUOLBknMZCYBJZlG5bIzASj0fvnqatDVcUM66MqxKoZZpaigQsuVrFGYD1UN6torNNwLsU90wXX1pVkkGL0EOmhkptjFQI9rvUs_ceWNurJaupJ85AKX7zWmlDinhTzkRL3LBXroaJ9qbYADdfuoxYMbmKel_GiopYFBWRzJLI5OBzCLS2eeSwsvHlvx3s73tvx3o73dry3470d7-14b8d7O_7_2PFiq0tykTzs-Qq6Is3pEOa78I8zh25E4qzI7giH7gE-QdeajrGDbdvBeOo4pm3YQ7iHLsZjhKZoamDbmDjG1DwO4a8kKc5gjB2jeuGJ-Q47yLGq032rBqUmDZlI0s_y37DqT7Hjbys5JdI?type=png)](https://mermaid.live/edit#pako:eNrtmVGvmjAUx79K08U9qYEWdPCwREGzJVv2cB-WLLx0UrQZFgNl9zrjdx_QcmfViyS7jmQhGuWc_uR_oKek_3iAqySk0IWDwSHgADDOhAuqwwCKDd3SAKq4zPACzk4yZW5maokyFbE4LpMBfENIZJAwgMPT8UykyQ-qCIzx1eHRIwvFRkIm2j0FsGaOwxN11KQeOX8t_bIy7kzZ6kzZ7kp53thlUUjup4w6U8adKVtdKXvNT5N7KqPOlHFnylZXyn5n69nvbD37na3nReM0R_d7bC-bhJ1V9LrC8qDWDyAN1xfblZBGJI_FRV2a9GQyaSetMSRNk8cNLbc7brVZSrckvn5fvlK23ogMvAVzRrKzIi_KiV6zHI2JyXeqpuOypKs7gJVgP4lgCQfLnK_Kgxu3MvoHtV-t6rwtyq_io3wPBgEPeCaIoD4j65RsJVNlivN95LtcgE9kT9MAApKBYpN7A0C3AHwLsG4BdnXdGvSBhSHlkgKm5OZmCwa1YHALxmpgkGQ8swWDWjC4BdNUD5aMb7ZgUAvmrJ4vuTibsYUOLBknMZCYBJZlG5bIzASj0fvnqatDVcUM66MqxKoZZpaigQsuVrFGYD1UN6torNNwLsU90wXX1pVkkGL0EOmhkptjFQI9rvUs_ceWNurJaupJ85AKX7zWmlDinhTzkRL3LBXroaJ9qbYADdfuoxYMbmKel_GiopYFBWRzJLI5OBzCLS2eeSwsvHlvx3s73tvx3o73dry3470d7-14b8d7O_7_2PFiq0tykTzs-Qq6Is3pEOa78I8zh25E4qzI7giH7gE-QdeajrGDbdvBeOo4pm3YQ7iHLsZjhKZoamDbmDjG1DwO4a8kKc5gjB2jeuGJ-Q47yLGq032rBqUmDZlI0s_y37DqT7Hjbys5JdI)

## **Description:**

This repository is dedicated to building a computer vision model focused on detecting potholes. The model is trained on various datasets and is designed to assist urban municipalities in improving road maintenance and repair procedures.

## **Key Features:**

- **Data Management:** Raw data is stored immutably, ensuring a fallback in case of processing errors. Scripts for data downloading, cleaning, and processing are organized systematically.
- **Model Training:** Scripts for model training and prediction are maintained separately for clarity.
- **Exploratory Data Analysis:** Jupyter notebooks are used for data exploration and are kept separate from the main source code.
- **Visualization:** Scripts for generating figures, plots, and visualizations are included.
- **Testing:** Unit tests are provided to ensure code functionality.
- **Dependencies:** A 'requirements.txt' file lists all project dependencies for easy replication.

## **Dataset Sources:**

- annotated-potholes-dataset "Location 2 on Kaggle
- RDD2022\_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547 "RDD2022 - The multi-national Road Damage Dataset released through CRDDC'2022
- road-pothole-images-for-pothole-detection "Location 1 on Kaggle

## **Workflow:**

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

## **Languages Used:**

- Python (73.7%)
- Jupyter Notebook (21.9%)
- Shell (3.4%)
- Dockerfile (1.0%)

## License: [MIT](./LICENSE)

[1]: https://www.kaggle.com/datasets/sovitrath/road-pothole-images-for-pothole-detection "Location 1 on Kaggle"
[2]: https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset "Location 2 on Kaggle"
[3]: https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547 "RDD2022 - The multi-national Road Damage Dataset released through CRDDC'2022"
