# bike_sales_app_enhanced.py
"""
Enhanced Bike Payment Method Prediction App
- Training, evaluation, saving/loading model & encoders
- Improved Tkinter GUI with validation and retrain option
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import math
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------- Config ----------
DATA_PATH = r'D:\Data Sience Methodology , data project\bike_sales_100k.csv'
MODEL_PATH = 'final_model.joblib'
ENCODERS_PATH = 'label_encoders.joblib'
RANDOM_STATE = 42

# ---------- Utility functions ----------
def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    # Clean Date column (non-fatal)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    return df

def preprocess(df, fit_encoders=True, encoders=None):
    df = df.copy()
    categorical_columns = ['Bike_Model', 'Store_Location', 'Payment_Method', 'Customer_Gender']
    label_encoders = {} if fit_encoders else encoders or {}

    # Check columns
    for col in categorical_columns:
        if col not in df.columns:
            raise KeyError(f"Missing expected column: {col}")

    if fit_encoders:
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    else:
        # Use provided encoders for transform
        for col in categorical_columns:
            le = label_encoders.get(col)
            if le is None:
                raise KeyError(f"Encoder for {col} not found.")
            df[col] = le.transform(df[col].astype(str))

    # Create Revenue if possible
    if 'Price' in df.columns and 'Quantity' in df.columns:
        df['Revenue'] = df['Price'] * df['Quantity']

    X = df[['Customer_Age', 'Bike_Model', 'Price', 'Quantity', 'Store_Location', 'Customer_Gender']]
    y = df['Payment_Method']
    return X, y, label_encoders

def train_and_save_model(data_path=DATA_PATH, model_path=MODEL_PATH, encoders_path=ENCODERS_PATH, random_state=RANDOM_STATE):
    print("[*] Loading data...")
    df = load_data(data_path)

    print("[*] Preprocessing data and encoding categorical features...")
    X, y, label_encoders = preprocess(df, fit_encoders=True)

    print("[*] Train/Test split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    print("[*] Training RandomForestClassifier...")
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    print("[*] Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix saved to file
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    fig_path = f'confusion_matrix_{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}.png'
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved confusion matrix to: {fig_path}")

    print("[*] Cross-validation (5-fold) on training set...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-Validation Mean Accuracy: {np.mean(cv_scores):.4f}")

    # Save model and encoders
    print(f"[*] Saving model to {model_path} and encoders to {encoders_path} ...")
    joblib.dump(model, model_path)
    joblib.dump(label_encoders, encoders_path)
    print("[*] Done.")

    return model, label_encoders

# ---------- Load model & encoders (if exist) ----------
def load_model_and_encoders(model_path=MODEL_PATH, encoders_path=ENCODERS_PATH):
    if not os.path.exists(model_path) or not os.path.exists(encoders_path):
        raise FileNotFoundError("Model or encoders not found. Please train the model first.")
    model = joblib.load(model_path)
    label_encoders = joblib.load(encoders_path)
    return model, label_encoders

# ---------- Safe transform helpers ----------
def safe_transform_category(le, value):
    """Transform string category value to encoded int. If unseen, try to handle gracefully."""
    try:
        return le.transform([str(value)])[0]
    except ValueError:
        # If unseen label, append temporarily (not recommended for production).
        classes = list(le.classes_)
        classes.append(str(value))
        new_le = LabelEncoder()
        new_le.classes_ = np.array(classes)
        return new_le.transform([str(value)])[0]

# =====================================================
# If script run directly, train model (if dataset present)
# =====================================================
if __name__ == "__main__" and (len(sys.argv) == 1 or sys.argv[1] != '--no-train'):
    # If user wants to skip training at launch, they can pass --no-train
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODERS_PATH):
            print("[*] Model or encoders not found — training from dataset.")
            train_and_save_model()
        else:
            print("[*] Model and encoders already exist. To retrain, run: python bike_sales_app_enhanced.py --retrain")
    except Exception as e:
        print("[Error] Training failed:", e)

# ---------- TKINTER GUI (separate import to avoid heavy imports during training-only runs) ----------
import tkinter as tk
from tkinter import ttk, messagebox

# Attempt to load model & encoders; if missing, disable prediction until trained
MODEL_AVAILABLE = os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH)
if MODEL_AVAILABLE:
    model, label_encoders = load_model_and_encoders()
else:
    model = None
    label_encoders = None

# ---------- GUI helpers ----------
def validate_int(value, name="value", min_val=None, max_val=None):
    try:
        ival = int(value)
        if min_val is not None and ival < min_val:
            raise ValueError(f"{name} must be >= {min_val}")
        if max_val is not None and ival > max_val:
            raise ValueError(f"{name} must be <= {max_val}")
        return ival, None
    except Exception as e:
        return None, str(e)

def validate_float(value, name="value", min_val=None, max_val=None):
    try:
        fval = float(value)
        if min_val is not None and fval < min_val:
            raise ValueError(f"{name} must be >= {min_val}")
        if max_val is not None and fval > max_val:
            raise ValueError(f"{name} must be <= {max_val}")
        return fval, None
    except Exception as e:
        return None, str(e)

def predict_payment_from_inputs(age, bike_model, price, quantity, store_location, gender):
    if model is None or label_encoders is None:
        raise RuntimeError("Model or encoders not available. Please train or load the model first.")
    # Transform categorical values using safe_transform_category
    bm_enc = safe_transform_category(label_encoders['Bike_Model'], bike_model)
    sl_enc = safe_transform_category(label_encoders['Store_Location'], store_location)
    g_enc = safe_transform_category(label_encoders['Customer_Gender'], gender)
    features = [age, bm_enc, price, quantity, sl_enc, g_enc]
    pred = model.predict([features])[0]
    return label_encoders['Payment_Method'].inverse_transform([pred])[0]

# ---------- Build GUI ----------
root = tk.Tk()
root.title("🚲 Bike Payment Method Predictor (Enhanced)")
root.geometry("480x560")
root.resizable(False, False)
padx = 12
pady = 6

frame = ttk.Frame(root, padding=12)
frame.pack(fill='both', expand=True)

# Title
title = ttk.Label(frame, text="Bike Payment Method Predictor", font=("Segoe UI", 14, "bold"))
title.grid(row=0, column=0, columnspan=2, pady=(0, 10))

# Age
ttk.Label(frame, text="Customer Age:").grid(row=1, column=0, sticky='w', padx=padx, pady=pady)
age_entry = ttk.Entry(frame)
age_entry.grid(row=1, column=1, sticky='ew', padx=padx)

# Bike Model
ttk.Label(frame, text="Bike Model:").grid(row=2, column=0, sticky='w', padx=padx, pady=pady)
bike_model_cb = ttk.Combobox(frame, state='readonly')
bike_model_cb.grid(row=2, column=1, sticky='ew', padx=padx)

# Price
ttk.Label(frame, text="Bike Price ($):").grid(row=3, column=0, sticky='w', padx=padx, pady=pady)
price_entry = ttk.Entry(frame)
price_entry.grid(row=3, column=1, sticky='ew', padx=padx)

# Quantity
ttk.Label(frame, text="Quantity:").grid(row=4, column=0, sticky='w', padx=padx, pady=pady)
quantity_entry = ttk.Entry(frame)
quantity_entry.grid(row=4, column=1, sticky='ew', padx=padx)

# Store Location
ttk.Label(frame, text="Store Location:").grid(row=5, column=0, sticky='w', padx=padx, pady=pady)
store_location_cb = ttk.Combobox(frame, state='readonly')
store_location_cb.grid(row=5, column=1, sticky='ew', padx=padx)

# Gender
ttk.Label(frame, text="Customer Gender:").grid(row=6, column=0, sticky='w', padx=padx, pady=pady)
gender_cb = ttk.Combobox(frame, state='readonly')
gender_cb.grid(row=6, column=1, sticky='ew', padx=padx)

# Status label
status_var = tk.StringVar(value="Model loaded and ready." if MODEL_AVAILABLE else "Model not available. Train the model first.")
status_label = ttk.Label(frame, textvariable=status_var, foreground="blue")
status_label.grid(row=7, column=0, columnspan=2, pady=(8, 8))

# Buttons
def on_predict():
    # Validate inputs
    age_raw = age_entry.get().strip()
    price_raw = price_entry.get().strip()
    qty_raw = quantity_entry.get().strip()
    bike_model = bike_model_cb.get().strip()
    store_location = store_location_cb.get().strip()
    gender = gender_cb.get().strip()

    if not MODEL_AVAILABLE:
        messagebox.showwarning("Model Missing", "Model is not available. Please train the model first using 'Retrain Model' button.")
        return

    # Validate numeric fields
    age, err = validate_int(age_raw, name="Age", min_val=0, max_val=120)
    if err:
        messagebox.showerror("Invalid Age", err); return
    price, err = validate_float(price_raw, name="Price", min_val=0.0)
    if err:
        messagebox.showerror("Invalid Price", err); return
    qty, err = validate_int(qty_raw, name="Quantity", min_val=1)
    if err:
        messagebox.showerror("Invalid Quantity", err); return

    # Validate categorical selections
    if not bike_model or not store_location or not gender:
        messagebox.showerror("Missing Selection", "Please select Bike Model, Store Location and Customer Gender.")
        return

    try:
        predicted = predict_payment_from_inputs(age, bike_model, price, qty, store_location, gender)
        messagebox.showinfo("Prediction", f"Predicted Payment Method:\n{predicted}")
    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))

def on_retrain():
    answer = messagebox.askyesno("Retrain Model", "Retraining will re-fit the model using the dataset and overwrite saved model/encoders. Continue?")
    if not answer:
        return
    try:
        status_var.set("Retraining model... Please wait.")
        root.update_idletasks()
        train_and_save_model()
        # reload
        global model, label_encoders, MODEL_AVAILABLE
        model, label_encoders = load_model_and_encoders()
        MODEL_AVAILABLE = True
        # update combobox values
        bike_model_cb['values'] = list(label_encoders['Bike_Model'].classes_)
        store_location_cb['values'] = list(label_encoders['Store_Location'].classes_)
        gender_cb['values'] = list(label_encoders['Customer_Gender'].classes_)
        status_var.set("Retraining complete. Model updated.")
        messagebox.showinfo("Retrain Complete", "Model retrained and saved successfully.")
    except Exception as e:
        status_var.set("Retraining failed.")
        messagebox.showerror("Retrain Error", str(e))

predict_btn = ttk.Button(frame, text="Predict Payment Method", command=on_predict)
predict_btn.grid(row=8, column=0, columnspan=2, pady=(10, 4), ipadx=6)

retrain_btn = ttk.Button(frame, text="Retrain Model (from dataset)", command=on_retrain)
retrain_btn.grid(row=9, column=0, columnspan=2, pady=(4, 4))

# Populate comboboxes if encoders present
if label_encoders:
    bike_model_cb['values'] = list(label_encoders['Bike_Model'].classes_)
    store_location_cb['values'] = list(label_encoders['Store_Location'].classes_)
    gender_cb['values'] = list(label_encoders['Customer_Gender'].classes_)

# Keep UI responsive
for i in range(2):
    frame.columnconfigure(i, weight=1)

# Run
root.mainloop()
