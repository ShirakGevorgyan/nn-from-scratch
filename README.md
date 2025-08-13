# Heart Disease Prediction with Neural Networks

This project involves building a neural network entirely from scratch using Python. It focuses on training the model to predict the risk of heart disease. Additionally, the project includes deploying the model through a FastAPI application and packaging the entire solution in a Docker container for easy deployment and scalability.

### Neural Network Implementation
As part of this project, a custom neural network was implemented entirely from scratch, without relying on external libraries.

The neural network architecture includes:
- **Input Layer**: Number of features from the dataset (11 input features).
- **Hidden Layers**: 3 layers with 32, 64, and 32 neurons, using ReLU as the activation function.
- **Output Layer**: Single neuron with a sigmoid activation function for binary classification (presence or absence of heart disease).
- **Optimization**: Custom implementation of Adam Optimizer.
- **Loss Function**: Binary Cross-Entropy Loss.

### Dataset

**Clinical heart disease dataset** sourced from [Kaggle](https://www.kaggle.com/code/gkhnakar/heart-disease-risk-prediction/notebook), containing 1048 records with 12 features was used for training the neural network.

The dataset includes the following variables:

- **age**: Age of the person in years.
- **sex**: Gender of the individual (1 = Male, 0 = Female).
- **cp**: Chest pain type (1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic).
- **trestbps**: Resting blood pressure (in mm Hg).
- **chol**: Serum cholesterol level (in mg/dl).
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false).
- **restecg**: Resting electrocardiographic results (0 = normal, 1 = ST-T wave abnormality, 2 = probable left ventricular hypertrophy).
- **thalach**: Maximum heart rate achieved.
- **exang**: Exercise-induced angina (1 = yes, 0 = no).
- **oldpeak**: ST depression induced by exercise relative to rest.
- **slope**: Slope of the peak exercise ST segment (1 = upsloping, 2 = flat, 3 = downsloping).
- **target**: Outcome (0 = no heart disease, 1 = heart disease).


**Exploratory Data Analysis (EDA)** was performed to understand the dataset’s structure and distribution and analyze relationships between features and the target variable. Visualizations such as histograms, scatter plots, and count plots were used to uncover patterns and potential imbalances in the data. 

### Data Preprocessing
To prepare the dataset for training, the following preprocessing steps were applied:

 **Encoding**:
   - Features such as `sex`, `cp`, `fbs`, `restecg`, `exang`, and `slope` were encoded using **Target Encoding**. 

 **Scaling**:
   - Features such as `age`, `trestbps`, `chol`, and `thalach` were standardized using the **Standard Scaler**.
   - The `oldpeak` feature was scaled using the **MinMax Scaler**.

### Dataset Splitting

The dataset was split into:
- **Training Set**: 70% of the data used for training the neural network.
- **Validation Set**: 15% of the data used to tune the model and monitor performance during training.
- **Testing Set**: 15% of the data held out for final model evaluation.

The model was initially trained and evaluated on the training and validation sets. After confirming the model's performance, the training and validation sets were combined to retrain the model. The final evaluation was done on unseen test data to check how well the model performs on new data.

### Model Evaluation

Given the nature of healthcare data, where false negatives (failing to identify patients with heart disease) can have severe consequences, the evaluation of the model was focused on the following metrics:

1. **Recall (Sensitivity)**: Prioritized to ensure the model identifies as many positive cases (patients at risk) as possible, minimizing the chances of missed diagnoses.
2. **Matthews Correlation Coefficient (MCC)**: MCC was used to evaluate the model’s overall performance. It considers all elements of the confusion matrix and provides a balanced evaluation.


---

## Project Structure

```
├── data/
│   └── heart_disease.csv              
├── model/
│   ├── model.pkl                      
│   └── preprocessing.pkl              
├── notebook/
│   └── heart_disease_analysis.ipynb   
├── src/
│   ├── api.py                         
│   └── nn_from_scratch.py 
│   └── schemas.py                
├── venv/                              
├── Dockerfile                        
├── requirements.txt                   
```

---

## Tech Stack

- **Programming Language**: Python 3.11
- **Libraries**:
  - Core: `numpy`, `pandas`, `mypy`, `FastAPI`, `Pydantic`
  - Neural Network: Custom-built, no external frameworks
  - Containerization: Docker

---

**Pydantic** was used to validate input data in the FastAPI application. The `schemas.py` file ensures that all input values are of the correct type and within valid ranges. If invalid data is provided the API returns error messages.


## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<ElenTerteryan>/nn-from-scratch.git
cd nn-from-scratch
```

### 2. Create and Activate Virtual Environment
```bash
py -3.11 -m venv venv
venv\Scripts\activate # for Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Neural Network
Run the training notebook located at `notebook/heart_disease_analysis.ipynb`. This will save the trained model and preprocessing artifacts.

---

## Usage

### Start the FastAPI Application
```bash
uvicorn src.api:app --reload
```

### API Docs
The FastAPI application availale at http://localhost:8000/docs.

---

## Run with Docker 

```bash
docker build -t heart-disease-app .
docker run -p 8000:8000 heart-disease-app
```



