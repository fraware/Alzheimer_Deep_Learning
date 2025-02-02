## Deep Learning for Early Alzheimer's

### Overview

This project leverages machine learning techniques to detect early signs of Alzheimer's disease using MRI data. The dataset used is sourced from the Open Access Series of Imaging Studies (OASIS): https://sites.wustl.edu/oasisbrains/, which provides longitudinal MRI scans of 150 subjects aged 60 to 96.

### Dataset

- Each subject was scanned at least once.
- All subjects are right-handed.
- 72 subjects were classified as 'Nondemented' throughout the study.
- 64 subjects were classified as 'Demented' from the beginning.
- 14 subjects transitioned from 'Nondemented' to 'Demented' and were categorized as 'Converted'.

### Methodology

We employ various machine learning models to analyze the MRI data and predict early signs of Alzheimer's. The models used include:

- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest Classifier
- AdaBoost

### Cross-Validation

To optimize the performance of each model, we conduct 5-fold cross-validation, selecting hyperparameters that maximize accuracy. Performance metrics such as accuracy, recall, and the area under the receiver operating characteristic curve (AUC) are evaluated.

### Performance Measures

Our primary evaluation metric is AUC, which is crucial in medical diagnostics where early detection is essential. A higher true positive rate ensures that Alzheimer's cases are detected early, while minimizing false positives to prevent misdiagnosis.

### Key Goals

- Develop a robust model for early Alzheimer's detection.
- Improve diagnostic accuracy using MRI biomarkers.
- Compare the effectiveness of different machine learning techniques.

### Installation

To set up the project, install the required dependencies using:

```bash
pip install -r requirements.txt
```

### Usage

Run the main notebook or script to preprocess the dataset and train the models:

```bash
python main.py
```

### Results

A comparison of model performance based on AUC, accuracy, and recall will be available in the final evaluation.

### Acknowledgments

This research is based on publicly available MRI data from the OASIS project. Special thanks to the contributors of the dataset and the research community working on neurodegenerative disease detection.
