# 🌾 Crop Recommendation System

A machine learning project that recommends the most suitable crop to grow based on soil nutrients and weather conditions. Five classification algorithms are trained and compared, and the best-performing model (Random Forest) is saved for reuse.

## 📌 Overview

Choosing the right crop for a given piece of land depends on soil composition and climate. This project uses a supervised learning approach: given the soil's **N, P, K** content, **pH**, and local **temperature, humidity and rainfall**, the model predicts which of **22 crops** is the best fit.

## 📊 Dataset

- **File:** `Crop_recommendation.csv`
- **Size:** 2,200 rows × 8 columns
- **Balance:** 100 samples for each of the 22 crops (perfectly balanced)
- **Missing values:** none observed in the notebook

| Feature | Description |
|---|---|
| `N` | Nitrogen content in soil |
| `P` | Phosphorus content in soil |
| `K` | Potassium content in soil |
| `temperature` | Temperature in °C |
| `humidity` | Relative humidity in % |
| `ph` | Soil pH value |
| `rainfall` | Rainfall in mm |
| `label` | **Target**: recommended crop |

**Crops covered (22):** rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil, pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute, coffee.

## 🛠️ Tech Stack

- Python 3
- pandas, NumPy
- scikit-learn
- matplotlib, seaborn
- Jupyter / Google Colab

## 🔬 Methodology

1. **Data exploration:** inspected shape, data types, class distribution and a feature correlation heatmap.
2. **Feature/target split:** `N, P, K, temperature, humidity, ph, rainfall` as features; `label` as target.
3. **Train/test split:** 80% train / 20% test (`test_size=0.2`, `random_state=2`), giving 440 test samples.
4. **Model training and evaluation:** five classifiers, each evaluated with accuracy, a per-class classification report, and 5-fold cross-validation.
5. **Model comparison:** accuracy bar chart across all algorithms.
6. **Model export:** the Random Forest model is serialized with `pickle`.
7. **Inference:** sample predictions using the trained model.

### Models and settings

| Model | Key settings |
|---|---|
| Decision Tree | `criterion="entropy"`, `max_depth=5`, `random_state=2` |
| Gaussian Naive Bayes | defaults |
| SVM | `kernel="poly"`, `degree=3`, `C=1`, with Min-Max scaling |
| Logistic Regression | `random_state=2` |
| Random Forest | `n_estimators=20`, `random_state=0` |

## 📈 Results

| Model | Test Accuracy | 5-Fold CV Accuracy (mean) |
|---|---|---|
| Decision Tree | 90.00% | 91.41% |
| Logistic Regression | 95.23% | 95.45% |
| SVM (poly kernel) | 97.95% | 98.27% |
| Gaussian Naive Bayes | **99.09%** | **99.50%** |
| Random Forest | **99.09%** | 99.45% |

**Key observations**

- Gaussian Naive Bayes and Random Forest are the top performers, both reaching 99.09% test accuracy.
- The depth-limited Decision Tree performs worst. It fails on `kidneybeans` and `mothbeans` (precision and recall of 0.00 for both), which drags down its overall score.
- Most remaining errors across the stronger models come from the `rice` and `jute` classes, which have similar nutrient and climate profiles.
- Random Forest was chosen for deployment because it is robust and consistent across folds.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Add the dataset

Place `Crop_recommendation.csv` in the project folder, then update the path in the notebook:

```python
df = pd.read_csv('Crop_recommendation.csv')
```

### 4. Run the notebook

```bash
jupyter notebook Crop_Recommendation_Model.ipynb
```

## 💡 Usage

Load the saved model and predict a crop. Input order is `[N, P, K, temperature, humidity, ph, rainfall]`.

```python
import pickle
import numpy as np

with open('models/RandomForest.pkl', 'rb') as f:
    model = pickle.load(f)

sample = np.array([[104, 18, 30, 23.6, 60.3, 6.7, 140.9]])
print(model.predict(sample))   # e.g. ['coffee']
```

## 📁 Project Structure

```
.
├── Crop_Recommendation_Model.ipynb   # Full analysis, training and evaluation
├── Crop_recommendation.csv           # Dataset
├── models/
│   └── RandomForest.pkl              # Saved Random Forest model
└── README.md
```

## 🔮 Future Improvements

- Hyperparameter tuning with `GridSearchCV` or `RandomizedSearchCV`
- Compare additional models such as XGBoost, KNN and Gradient Boosting
- Feature importance analysis to see which factors drive recommendations
- Show top-3 crop suggestions with probabilities instead of a single prediction
- Build a web app (Streamlit or Flask) for farmers to enter values and get recommendations
- Validate on real-world regional soil and weather data

## 🤝 Contributing

Suggestions and improvements are welcome. Feel free to open an issue or submit a pull request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
