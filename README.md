# 🎬 Movie Success & Revenue Prediction System

*A Data Mining Project for CIS 593 – Cleveland State University*  
**Advised by Dr. Sunnie Chung**

## 👥 Team Members
- **Dharmik Kurlawala** – CSU ID: 2886995
- **Niteesh Singh** – CSU ID: 2886321

---

## 📌 Overview

This project builds an end-to-end machine learning system that predicts a movie’s box office **success (hit/flop)** and **revenue** using features known **before release**—including cast, crew, genres, budget, and overview text.

## 🎯 Goals

- **Classify** a movie as Success or Flop (Success = Revenue > 1.5 × Budget)
- **Predict** log-transformed box office revenue (via regression)
- **Deploy** a Flask API that serves predictions and provides visual explanations

---

## 📂 Project Structure

```
├── Dataset_Creation.py         # Merge raw TMDB datasets
├── Filtered_Columns.py         # Clean, label, and log-transform data
├── Data_Processing.py          # Feature engineering & scaling
├── Visualization.py            # Feature histograms & correlation plots
├── Train_Model.py              # ANN classifier & XGBoost regression
├── Evaluate_Model.py           # Evaluation metrics, ROC, PR, Calibration
├── INFERENCE_API.py            # Flask app for movie input/prediction
├── evaluation_plots/           # Confusion matrix, PR, ROC, calibration
├── debug_inputs/               # Stores raw user inputs from web form
├── form.html                   # HTML frontend form
├── movie_features.csv          # Final dataset with features & targets
└── *.pkl / .parquet            # Models, encoders, feature lists
```

---

## 🧠 Feature Engineering Highlights

- **Star Power**: Custom metric using past box-office, ratings, and popularity of cast/crew
- **Overview Sentiment**: Sentiment polarity from TextBlob
- **Is Sequel / Big Studio / Holiday Release**: Binary flags
- **Genres & Languages**: Multi-hot encodings (~100 total binary features)
- **TF-IDF**: Top 50 terms from overview text, downscaled to preserve balance

---

## 🧪 Model Design

### 1. ANN Classifier
- 3 hidden layers with dropout and L2 regularization
- Calibrated using **Isotonic Regression**
- Outputs probability of success

### 2. XGBoost Regressor
- Input includes engineered features + ANN’s success probability
- Predicts **log(revenue)**; exponentiated for dollar estimate

---

## 📊 Evaluation Results

| Metric        | Classifier  | Regressor      |
|---------------|-------------|----------------|
| Accuracy      | 83%         | —              |
| F1-Score      | 0.87        | —              |
| ROC-AUC       | 0.92        | —              |
| MAE           | —           | ~$33M USD      |
| R² Score      | —           | ~0.66          |

---

## 📈 Visual Outputs

Plots generated via `Evaluate_Model.py`:
- ![roc_curve](https://github.com/user-attachments/assets/6f178afc-0d1c-4597-8303-53a5c5f2cf78)

- ![pr_curve](https://github.com/user-attachments/assets/db9101a7-4419-43c3-8f6d-b2716a925582)

- ![confusion_matrix](https://github.com/user-attachments/assets/7da2ac1d-ad53-4580-b41f-ae75911a078a)

- ![calibration_curve](https://github.com/user-attachments/assets/3654c1f3-2a09-431f-b31f-d74cf7f2189e)


---

## 🌐 Run the Web API

```bash
python INFERENCE_API.py
```

Then visit: localhost  
Submit movie details via the form to receive:
- Success probability
- Revenue prediction
- SHAP visual explanations

---

## 🛠 Requirements

```
flask
pandas
numpy
scikit-learn
tensorflow
xgboost
textblob
shap
matplotlib
seaborn
unidecode
```

---

## 🧪 How to Run

1. **Data Setup**
    ```bash
    python Dataset_Creation.py
    python Filtered_Columns.py
    ```
2. **Feature Engineering**
    ```bash
    python Data_Processing.py
    ```

3. **Model Training**
    ```bash
    python Train_Model.py
    ```

4. **Evaluation**
    ```bash
    python Evaluate_Model.py
    ```

5. **API Deployment**
    ```bash
    python INFERENCE_API.py
    ```

---

## 📚 Academic Context

This work was submitted as the final project for the **CIS 593: Data Mining and Machine Learning** course at **Cleveland State University**, Spring 2025.
