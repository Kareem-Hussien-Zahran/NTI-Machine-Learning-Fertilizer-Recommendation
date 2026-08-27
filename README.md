# 🌱 Fertilizer Recommendation using Machine Learning

## 📌 Project Overview

This project is about predicting the **best fertilizer** for a specific agricultural situation.

The model uses information about the soil, crop, weather, irrigation, previous crop, and previous fertilizer usage to recommend a fertilizer.

The dataset contains **10,000 records** and the target variable is:

`Recommended_Fertilizer`

---

## 🎯 Project Goal

The main goal is to build a classification model that can predict the suitable fertilizer based on different agricultural features.

The target contains 7 fertilizer classes:

- Urea
- DAP
- MOP
- Compost
- Zinc Sulphate
- NPK
- SSP

The classes are not balanced. For example, Urea and DAP have much more samples than SSP.

---

## 📊 Dataset Features

The dataset contains different types of information:

### Soil Information

- Soil Type
- Soil pH
- Soil Moisture
- Organic Carbon
- Electrical Conductivity

### Nutrient Information

- Nitrogen Level
- Phosphorus Level
- Potassium Level

### Weather Information

- Temperature
- Humidity
- Rainfall

### Crop Information

- Crop Type
- Crop Growth Stage
- Previous Crop
- Season

### Farm Information

- Irrigation Type
- Region
- Fertilizer Used Last Season
- Yield Last Season

### Target

- Recommended Fertilizer

These features can be seen directly in the notebook dataset.

---

## 🔎 Data Preparation

First, the dataset is loaded using Pandas.

```python
df = pd.read_csv("fertilizer_recommendation.csv")
```

Then the data is divided into:

```python
X = df.drop("Recommended_Fertilizer", axis=1)
Y = df["Recommended_Fertilizer"]
```

So:

- `X` → input features
- `Y` → target variable

---

## 🔤 Categorical Features

The dataset contains several categorical columns.

The notebook automatically finds them:

```python
cat_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()
```

The categorical columns are:

- Soil_Type
- Crop_Type
- Crop_Growth_Stage
- Season
- Irrigation_Type
- Previous_Crop
- Region

They are converted to Pandas categorical type.

This is useful because **CatBoost and LightGBM can work with categorical features directly**.

---

## 🎯 Target Encoding

The target variable is converted from fertilizer names into numerical class labels using `LabelEncoder`.

```python
encoder = LabelEncoder()

Y = encoder.fit_transform(Y)
```

This is done because the classification models work with numerical class labels.

---

# 🤖 Models

In this project, different boosting models are tested.

## 1. CatBoost

CatBoost is trained with:

- 500 iterations
- Learning rate = 0.01
- Depth = 3
- Random seed = 42
- Balanced class weights

```python
CatBoostClassifier(
    iterations=500,
    learning_rate=0.01,
    depth=3,
    random_seed=42,
    auto_class_weights="Balanced"
)
```

The balanced class weights are used because the target classes are imbalanced.

---

## 2. LightGBM

LightGBM is also used for the classification problem.

The model uses:

```python
LGBMClassifier(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=5,
    random_state=42,
    min_child_samples=1,
    class_weight="balanced"
)
```

The `class_weight="balanced"` option helps the model deal with the imbalanced target classes.

LightGBM also receives the categorical features directly.

---

# 📈 Evaluation Metrics

The models are evaluated using several metrics:

- Accuracy
- Precision
- Recall
- Macro F1 Score
- Classification Report

The notebook also measures the **training time** of the models.

### Why Macro F1?

The dataset is imbalanced, so accuracy alone is not enough.

Macro F1 gives equal importance to each class, including classes with fewer samples.

---

# 📊 Confusion Matrix

A confusion matrix is used to see how the model predicts each fertilizer class.

For LightGBM:

```python
cm = confusion_matrix(Y, y_predict_LGB)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot(
    xticks_rotation=45
)
```

This helps us see which fertilizer classes are predicted correctly and which classes are confused with each other.

---

# 🏆 Results

The notebook currently shows the following LightGBM results:

| Metric | LightGBM |
|---|---:|
| Training Time | 7.87 seconds |
| Accuracy | 93.43% |
| Macro F1 | 86.95% |
| Precision | 86.65% |
| Recall | 93.57% |

These results are from the notebook's current evaluation.

CatBoost results shown in the notebook:

| Metric | CatBoost |
|---|---:|
| Training Time | 87.43 seconds |
| Accuracy | 87.91% |
| Macro F1 | 80.22% |
| Precision | 86.07% |
| Recall | 85.01% |



Based on these results, **LightGBM performed better than CatBoost in this experiment** and was also much faster.

---

# 🛠️ Libraries Used

```text
Python
Pandas
NumPy
Scikit-learn
CatBoost
LightGBM
XGBoost
Matplotlib
Seaborn
```

The notebook imports the main ML and evaluation libraries and installs CatBoost and LightGBM when needed. 
---

# 📁 Project Structure

```text
Fertilizer-Recommendation/
│
├── fertilizer_recommendation.csv
├── fertilizer_recommendation.ipynb
└── README.md
```

---

# 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fertilizer-recommendation.git
```

### 2. Install the required libraries

```bash
pip install pandas numpy scikit-learn
pip install catboost lightgbm xgboost
pip install matplotlib seaborn
```

### 3. Open the notebook

```bash
jupyter notebook
```

or open it using **Google Colab**.

### 4. Run the notebook cells

Run the cells from top to bottom to load the dataset, prepare the data, train the models, and evaluate the results.

---

# 💡 What I Learned

Through this project, I practiced:

- Data loading and exploration
- Separating features and target
- Working with categorical features
- Target encoding
- Handling imbalanced classification
- CatBoost
- LightGBM
- XGBoost
- Classification metrics
- Confusion Matrix
- Model training time comparison
- Comparing different ML models

---

# 🔮 Future Improvements

Some possible improvements for the project:

- Use `train_test_split` and evaluate on unseen test data
- Add cross-validation
- Tune the model hyperparameters
- Compare XGBoost, LightGBM and CatBoost using the same test set
- Add feature importance visualization
- Add normalized confusion matrix
- Add ROC-AUC evaluation
- Build a simple web application for fertilizer prediction

---

## 👨‍💻 Author

**Kareem Hussien**

AI & Data Science Student

Interested in:

- Data Analysis
- Machine Learning
- Artificial Intelligence
- Generative AI
- Agentic AI