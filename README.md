# Smart Health Risk Prediction API

This project is a Machine Learning powered REST API that predicts health risk levels based on patient data.

## Features
- Flask REST API
- Random Forest Classifier
- JSON input/output
- Deployed on Railway

## Input Features
- Age
- BMI
- Blood Pressure
- Glucose Level
- Smoking Status

## API Endpoint

POST /predict

### Request Body:
```json
{
  "age": 45,
  "bmi": 28,
  "bp": 140,
  "glucose": 110,
  "smoking": 1
}