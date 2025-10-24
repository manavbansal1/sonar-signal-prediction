# Sonar Signal Prediction

A machine learning binary classifier that distinguishes between rocks and mines using sonar signals. Built with Python and scikit-learn.

## Overview

This project uses **Logistic Regression** to classify underwater objects detected by sonar as either rocks or mines based on 60 frequency features. The model achieves **~78% cross-validation accuracy** with robust evaluation metrics.

## Dataset

- **208 samples** (111 mines, 97 rocks)
- **60 sonar frequency features** (normalized 0-1 range)
- **No missing values** - Clean dataset ready for training
- Source: UCI Machine Learning Repository

## Tech Stack

- **Python 3.7+**
- **scikit-learn** - Model training and evaluation
- **pandas** - Data manipulation
- **numpy** - Numerical operations

## Key Features

- StandardScaler preprocessing for optimal performance
- 5-fold cross-validation for reliable metrics
- Model persistence with pickle
- Interactive CLI prediction system
- Comprehensive evaluation metrics (confusion matrix, classification report)

## Performance

```
Training Accuracy: 91.98%
Test Accuracy: 76.19%
Cross-Validation: 78.05% (±2.80%)

Precision: 75-78% for both classes
Recall: 70-82% for both classes
F1-Score: 74-78% balanced performance
```

### Confusion Matrix
```
              Predicted
           Mine  Rock
Actual Mine  9     2
       Rock  3     7
```

## Installation & Usage

```bash
# Clone repository
git clone https://github.com/manavbansal1/sonar-signal-prediction.git
cd sonar-signal-prediction

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas scikit-learn

# Run the model
python rock_vs_mine_prediction.py
```

## Technical Highlights

- **Stratified train-test split** (90/10) to maintain class balance
- **Feature scaling** to prevent data leakage (fit on train, transform on test)
- **5-fold cross-validation** for robust performance estimation
- **Model serialization** for deployment-ready predictions
- **Interactive prediction system** with confidence scores

## Project Structure

```
├── rock_vs_mine_prediction.py  # Main script
├── sonar_data.csv              # Dataset
├── README.md
└── .gitignore
```

## Future Enhancements

- Experiment with Random Forest and SVM algorithms
- Hyperparameter tuning with GridSearchCV
- Web interface with Streamlit or Flask
- Feature importance visualization
- Increase dataset size for better generalization


---

⭐ Star this repo if you find it useful!