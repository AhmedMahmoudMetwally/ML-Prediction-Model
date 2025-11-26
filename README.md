Below is the **professional English README** — clean, structured, and ready to paste directly into **GitHub / LinkedIn**.

---

# 🚲 Bike Sales Payment Method Prediction

### Machine Learning Model + Tkinter Desktop Application

This project builds an end-to-end machine learning pipeline to **predict the payment method** a customer is likely to use when purchasing a bike.
It includes **data preprocessing, model training, evaluation, model saving**, and a fully functional **Tkinter desktop GUI application** for real-time predictions.

---

## 📌 **Project Overview**

The goal of this project is to analyze a bike sales dataset and develop a system that predicts whether a customer will pay using:

* 💳 Credit Card
* 💵 Cash
* 🧾 Installments
* Or any other payment method included in the dataset

The system uses a **Random Forest Classifier** and provides predictions through a simple and user-friendly **Tkinter graphical interface**.

---

## 📂 **Dataset Description**

The dataset contains **100,000+ bike sales records**, including:

| Column            | Description                         |
| ----------------- | ----------------------------------- |
| `Customer_Age`    | Customer age                        |
| `Bike_Model`      | Model of the purchased bike         |
| `Price`           | Price of the bike                   |
| `Quantity`        | Number of units purchased           |
| `Store_Location`  | Store branch                        |
| `Customer_Gender` | Male/Female                         |
| `Payment_Method`  | Target variable                     |
| `Date`            | Transaction date                    |
| `Revenue`         | Computed feature (Price × Quantity) |

Categorical features are encoded using **LabelEncoder**, and date formatting issues are handled safely.

---

## 🤖 **Machine Learning Pipeline**

### **1. Data Cleaning & Preprocessing**

* Converted dates to standard datetime format
* Encoded categorical columns
* Generated a new feature: **Revenue**
* Selected relevant features for training

### **2. Model Training**

* Algorithm used: **Random Forest Classifier**
* 80/20 train-test split
* Cross-validation applied for accurate scoring

### **3. Model Evaluation**

Includes:

* Accuracy score
* Classification report
* Confusion Matrix (visualized using Seaborn)
* Cross-validation mean accuracy

The trained model and encoders are saved as:

```
final_model.joblib
label_encoders.joblib
```

---

## 🖥️ **Tkinter Desktop Application**

A user-friendly GUI allows real-time prediction based on input fields:

* Customer Age
* Bike Model
* Bike Price
* Quantity
* Store Location
* Customer Gender

When the user clicks **Predict Payment Method**, the model returns the most probable payment type.

✔ No need for coding
✔ Instant prediction
✔ Uses the saved ML model

---

## 📦 **Project Structure**

```
📁 Bike-Payment-Prediction
│── bike_sales_prediction_app.py       # Complete ML + GUI script
│── final_model.joblib                 # Trained Random Forest model
│── label_encoders.joblib              # Encoders for categorical features
│── bike_sales_100k.csv                # Dataset
│── README.md                          # Documentation
```

---

## 🛠️ **Technologies Used**

* **Python 3.9+**
* **Pandas**, **NumPy**
* **Matplotlib**, **Seaborn**
* **Scikit-Learn**
* **Tkinter**
* **Joblib**

---

## 🚀 **How to Run the Application**

### **1. Install Dependencies**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### **2. Run the App**

```bash
python bike_sales_prediction_app.py
```

The Tkinter application will launch instantly.

---

## 📈 **Model Performance (Example)**

| Metric                                   | Score |
| ---------------------------------------- | ----- |
| Accuracy                                 | ~0.92 |
| Cross-validation Accuracy                | ~0.90 |
| Very balanced performance across classes |       |

