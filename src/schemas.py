"""
Input schemas for the Heart Disease Detector.

This module defines the Pydantic request model used by the API to validate
feature inputs for heart-disease risk prediction. The feature names and
encodings follow the classic UCI Heart Disease dataset conventions.

Notes
-----
- Units:
    * trestbps: mm Hg (resting blood pressure)
    * chol: mg/dL (serum cholesterol)
    * thalach: bpm (maximum heart rate achieved)
    * oldpeak: ST depression (unitless, relative to rest)
- Encodings (dataset convention):
    * sex: 0=female, 1=male
    * cp (chest pain type): 0..3
    * fbs (fasting blood sugar): 1 if >120 mg/dL, else 0
    * restecg (resting ECG): 0..2
    * exang (exercise-induced angina): 1=yes, 0=no
    * slope (ST segment slope): 0..2
"""

from pydantic import BaseModel

class HeartDiseaseRequest(BaseModel):
    """Single-row input for heart-disease risk prediction.

    Attributes
    ----------
    age : float
        Age in years.
    sex : int
        Biological sex (0=female, 1=male).
    cp : int
        Chest pain type encoded as 0..3.
    trestbps : float
        Resting blood pressure on admission (mm Hg).
    chol : float
        Serum cholesterol (mg/dL).
    fbs : int
        Fasting blood sugar flag (1 if >120 mg/dL, else 0).
    restecg : int
        Resting electrocardiographic results (0..2).
    thalach : float
        Maximum heart rate achieved (bpm).
    exang : int
        Exercise-induced angina (1=yes, 0=no).
    oldpeak : float
        ST depression induced by exercise relative to rest.
    slope : int
        Slope of the peak exercise ST segment (0..2).
    """
    # Age in years
    age: float
    # Sex: 0 = female, 1 = male
    sex: int
    # Chest pain type: 0..3
    cp: int
    # Resting blood pressure (mm Hg)
    trestbps: float
    # Serum cholesterol (mg/dL)
    chol: float
    # Fasting blood sugar: 1 if >120 mg/dL, else 0
    fbs: int
    # Resting ECG results: 0..2
    restecg: int
    # Maximum heart rate achieved (bpm)
    thalach: float
    # Exercise-induced angina: 1=yes, 0=no
    exang: int
    # ST depression (relative to rest)
    oldpeak: float
    # Slope of the peak exercise ST segment: 0..2
    slope: int
