# AI-ML-Project

initial folder structure

├── data
│   ├── raw                    # Raw data, immutable
│   ├── interim                # Extracted and cleaned data
│   ├── processed              # Final data used for modeling
│   ├── external               # Data from third party sources
│   └── models                 # Trained and serialized models, model predictions, or model summaries
├── notebooks                  # Jupyter notebooks
├── src                        # Source code
│   ├── __init__.py            # Makes src a Python module
│   ├── data                   # Scripts to download, generate, clean, and process data
│   │   └── make_dataset.py
│   ├── features               # Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   ├── models                 # Scripts to train models and then use trained models to make predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   └── visualization          # Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
├── tests                      # Test cases for your project
│   └── test_basic.py
├── .gitignore
├── Dockerfile
├── requirements.txt           # The dependencies we need to reproduce the environment, libraries etc.
├── setup.py                   # Makes project pip installable (pip install -e .) so src can be imported
└── README.md                  # Project description

