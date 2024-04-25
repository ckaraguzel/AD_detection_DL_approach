# Alzheimer's Disease Prediction from Structural Brain MRI

## Goal
The goal of this project is to predict Alzheimer's disease from structural brain MRI (Magnetic Resonance Imaging) scans. We aim to classify early Alzheimer's disease into two classes: Non-Demented and Very Mild Demented.

## Dataset
The dataset used in this project consists of structural brain MRI scans. The dataset is preprocessed and split into training and test sets.

## Installation
1. Clone the repository:
git clone https://github.com/ckaraguzel/AD_detection_DL_approach.git
2. Install the required dependencies: 
pip install -r requirements.txt 
3. Download data from [Alzheimer's Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu/data-samples/access-data/). This data is publicly available, however it requires an application to access the data.

## Preprocessing
The preprocessing step involves the following tasks:
1. Remove the dark frame surrounding the MRI scans.
2. Remove the skull from the MRI scans.
3. Split the preprocessed dataset into training and test folders with a ratio of 0.2.

## Model Training
We train two models for Alzheimer's disease prediction:
1. Convolutional Neural Network (CNN):
   - The CNN model is implemented using PyTorch.
   - The architecture of the CNN model is defined in `notebooks/model.py`.

2. CNN + XGBoost:
   - The CNN model is used as a feature extractor, and the extracted features are then fed into an XGBoost classifier.
   - The CNN model is trained using PyTorch, and the XGBoost classifier is trained using the XGBoost library.
   - The training notebook for the CNN + XGBoost model: `notebooks/XGBoosted_ADNI.ipynb`.

## Evaluation
The trained models are evaluated on the test set using AUC(Area Under the Curve) as the evaluation metric.

## Results
The results of the model evaluation will be added here once available.

## Conclusion
This project demonstrates the use of deep learning and machine learning techniques for predicting Alzheimer's disease from structural brain MRI scans. The combination of CNN and XGBoost models shows promising results in accurately classifying early Alzheimer's disease.

## Future Work
- Explore additional preprocessing techniques to enhance the quality of the MRI scans.
- Experiment with different CNN architectures and hyperparameter tuning to improve model performance.
- Investigate the effectiveness of other machine learning algorithms in combination with CNN features.
- Expand the dataset to include more diverse samples and stages of Alzheimer's disease.


## References
- [Alzheimer's Disease Neuroimaging Initiative (ADNI)](http://adni.loni.usc.edu/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
