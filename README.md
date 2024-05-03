# Alzheimer's Disease Prediction from Structural Brain MRI

## Goal
The goal of this project is to predict Alzheimer's disease from structural brain MRI (Magnetic Resonance Imaging) scans. We aim to classify early Alzheimer's disease into two classes: Cognitively Normal and Mild Cognitive Impairment.

## Dataset
The dataset used in this project consists of structural brain MRI scans. The dataset is preprocessed and split into training, validation and test sets.

## Installation
1. Clone the repository:
git clone https://github.com/ckaraguzel/AD_detection_DL_approach.git
2. Install the required dependencies: 
pip install -r requirements.txt 
3. Data is from [Alzheimer's Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu/data-samples/access-data/), more specifically ADNI3 study. This data is publicly available, however it requires an application to access the data.

## Preprocessing
The preprocessing step involves the following tasks:
1. Keep one MRI session per subject.
2. Order the slices in each session and filter only middle 65 slices.  
3. Split the preprocessed dataset into training, validation and test folders.

## Model Training
We train two models for Alzheimer's disease prediction:
1. Convolutional Neural Network (CNN):
   - The CNN model is implemented using PyTorch.
   - The architecture of the CNN model is defined in `notebooks/model1.py`.
   - Weighted loss function, L1 regularization and Dropout is used to avoid overfitting.

2. CNN + XGBoost:
   - The CNN model is used as a feature extractor, and the extracted features are then fed into an XGBoost classifier.
   - The CNN model is trained using PyTorch, and the XGBoost classifier is trained using the XGBoost library.
   - The training notebook for the CNN + XGBoost model: `notebooks/model2.py`.
   - Weighted loss function is used to avoid overfitting in addition to the pretrained CNN part.

## Evaluation
The trained models are evaluated on the test set using AUC(Area Under the Curve) as the evaluation metric.

## Results
- Model 1 (CNN): AUC: 0.58 from MRI images only.
- Model 2 (CNN + XGBoost): AUC: 0.61 from MRI images, age and sex data.

## Conclusion
This project demonstrates the use of deep learning and machine learning techniques for predicting Alzheimer's disease from structural brain MRI scans. The combination of CNN and XGBoost models shows promising results in accurately classifying early Alzheimer's disease.

## Future Work
- Explore additional preprocessing techniques to enhance the quality of the MRI scans.
- Experiment with different CNN architectures and hyperparameter tuning to improve model performance.
- Investigate the effectiveness of other machine learning algorithms in combination with CNN features.
- Expand the dataset to include more diverse samples and stages of Alzheimer's disease.
- Explore a multi-modal approach by incorporating other types of data, such as clinical information, genetic data, or cognitive test results, to enhance the predictive power of the models. This can provide a more comprehensive understanding of Alzheimer's disease and potentially improve the accuracy of the predictions.


## References
- [Alzheimer's Disease Neuroimaging Initiative (ADNI) database, ADNI3 study](http://adni.loni.usc.edu/) (https://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
