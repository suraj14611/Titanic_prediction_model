# Titanic Survival Prediction Model 🚢

A Machine Learning web application that predicts whether a passenger would have survived the Titanic disaster based on passenger information. The project uses a trained classification model built with Scikit-learn and provides an easy-to-use web interface using Flask.

---

## 📌 Project Overview

The Titanic Survival Prediction Model is a beginner-friendly machine learning project that demonstrates the complete workflow of building, training, and deploying a predictive model.

The project includes:
- Data preprocessing
- Feature selection
- Model training
- Model serialization
- Flask web application for predictions

---

## 🛠️ Tech Stack

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML
- Pickle

---

## 📂 Project Structure

```
Titanic_prediction_model/
│
├── templates/
│   └── index.html
│
├── app.py                 # Flask web application
├── model_training.py      # Model training script
├── train.csv              # Training dataset
├── test.csv               # Testing dataset
├── titanic_model.pkl      # Saved trained model
└── README.md
```

---

## ⚙️ Features

- Predicts passenger survival based on user input
- Trained using the Titanic dataset
- Simple and interactive Flask interface
- Machine learning model saved using Pickle
- Easy to understand project structure for beginners

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/suraj14611/Titanic_prediction_model.git
```

Move into the project directory

```bash
cd Titanic_prediction_model
```

Install the required packages

```bash
pip install flask pandas numpy scikit-learn
```

Run the application

```bash
python app.py
```

The application will start on

```
http://127.0.0.1:5000/
```

---

## 📊 Dataset

This project uses the famous **Titanic Dataset**, which contains passenger details such as:

- Passenger Class (Pclass)
- Sex
- Age
- Number of Siblings/Spouses (SibSp)
- Number of Parents/Children (Parch)
- Fare
- Embarked Port

These features are used to predict passenger survival.

---

## 🧠 Machine Learning Workflow

1. Load the dataset
2. Clean missing values
3. Encode categorical variables
4. Select important features
5. Train the machine learning model
6. Save the trained model using Pickle
7. Deploy using Flask

---

## 🎯 Future Improvements

- Improve prediction accuracy
- Deploy the application online
- Add better UI styling
- Support multiple machine learning models
- Visualize prediction probabilities

---



---
