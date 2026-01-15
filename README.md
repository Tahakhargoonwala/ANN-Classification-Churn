# 🔮 Customer Churn Prediction System

A machine learning application that predicts customer churn probability using an Artificial Neural Network (ANN) model. This project combines deep learning with an interactive Streamlit web interface for real-time churn risk assessment.

## 📋 Project Overview

Customer churn prediction is critical for businesses to identify at-risk customers and take proactive retention measures. This project uses customer behavioral and demographic data to predict the likelihood of a customer leaving a service.

**Key Objective:** Classify customers as likely to churn or not based on their profile and engagement metrics.

## 🎯 Features

- **Deep Learning Model**: Trained ANN model using TensorFlow/Keras
- **Interactive Web Interface**: Streamlit-based dashboard for real-time predictions
- **Data Preprocessing**: Automatic encoding and scaling of inputs
- **Visual Analytics**: 
  - Churn probability metrics
  - Risk level indicators with progress bars
  - Color-coded predictions (success/error alerts)
  - Expandable prediction summary
  - Multi-column layout for organized input

## 📊 Dataset

**Source**: Churn Modelling Dataset (`Churn_Modelling.csv`)

**Features**:
- **Customer Information**: CreditScore, Age, Geography, Gender
- **Account Details**: Tenure, Balance, NumOfProducts
- **Service Usage**: HasCrCard, IsActiveMember
- **Financial**: EstimatedSalary
- **Target**: Exited (Churn - 0/1)

## 🛠️ Technology Stack

- **ML Framework**: TensorFlow/Keras
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Web Interface**: Streamlit
- **Model Storage**: HDF5 format
- **Preprocessing**: StandardScaler, LabelEncoder, OneHotEncoder
- **Environment**: Python 3.x with virtual environment

## 📁 Project Structure

```
ANNProject/
├── app.py                      # Streamlit web application
├── model.h5                    # Trained neural network model
├── requirement.txt             # Project dependencies
├── Churn_Modelling.csv         # Training dataset
├── experiments.ipynb           # Model training & experiments
├── prediction.ipynb            # Prediction examples & testing
├── scalers.pickle              # StandardScaler for feature scaling
├── label_encoder_gender.pkl    # Gender encoder
├── onehot_encoder_geo.pkl      # Geography encoder
├── scaler.pkl                  # Additional scaler
└── logs/                       # TensorFlow event logs for monitoring
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Activate Virtual Environment**
```bash
cd ANNProject
myenv\Scripts\activate  # Windows
source myenv/bin/activate  # Linux/Mac
```

2. **Install Dependencies**
```bash
pip install -r requirement.txt
```

3. **Run the Application**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 💡 Usage

1. **Input Customer Information**: Fill in customer details across organized sections:
   - Customer Information (Geography, Gender, Age, Tenure)
   - Financial Information (Credit Score, Balance, Salary)
   - Product & Services (Number of Products)
   - Membership Status (Credit Card, Active Member)

2. **View Prediction Results**:
   - Churn probability percentage
   - Risk level indicator (Low Risk ✅ or High Risk ⚠️)
   - Visual progress bar showing risk intensity
   - Expandable summary with input details

## 🧠 Model Architecture

The ANN model is trained on the churn dataset with:
- Input preprocessing (scaling, encoding)
- Multiple dense layers with activation functions
- Dropout for regularization
- Binary classification output (churn probability)
- Saved weights in `model.h5`

## 📈 Performance Monitoring

- Training logs stored in `logs/` directory
- TensorFlow event files for visualization with TensorBoard
- Validation metrics tracked during model training

## 🔄 Workflow

1. **Data Preparation** → `experiments.ipynb`
2. **Model Training** → `experiments.ipynb`
3. **Model Evaluation** → `prediction.ipynb`
4. **Deployment** → `app.py` (Streamlit interface)

## 📦 Dependencies

Main packages:
- `tensorflow` - Deep learning framework
- `streamlit` - Web interface
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning utilities
- `pickle` - Model/encoder serialization

See `requirement.txt` for complete list with versions.

## 🎓 Model Insights

**Input Features**: 10 customer attributes
**Output**: Binary classification (0: No Churn, 1: Churn)
**Prediction Method**: Probability-based (threshold: 0.5)

## 📝 Notes

- All input values are validated and preprocessed automatically
- The model uses the same preprocessing pipeline as training data
- Predictions are displayed as probability percentages for interpretability
- Risk categorization: >50% = High Risk, ≤50% = Low Risk

## 🤝 Contributing

To improve the model:
1. Run `experiments.ipynb` for retraining
2. Experiment with different architectures and hyperparameters
3. Validate predictions in `prediction.ipynb`
4. Update `model.h5` after improvements

## 📄 License

This project is for educational and business analytics purposes.

---

**Last Updated**: January 2026
**Status**: Active & Running ✅
