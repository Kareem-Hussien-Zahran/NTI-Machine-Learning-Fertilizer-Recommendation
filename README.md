# 🌱 Fertilizer Recommendation System

A Machine Learning project that recommends the most suitable fertilizer
based on soil conditions, crop characteristics, environmental
conditions, and previous-season agricultural information.

## Problem

Farmers need to select the appropriate fertilizer according to several
interacting factors. Soil properties, crop type and growth stage,
weather conditions, irrigation, previous crops, and previous fertilizer
usage can all influence the fertilizer recommendation.

Choosing a suitable fertilizer manually can therefore be challenging.
The objective of this project is to build a **Machine Learning
classification system** that learns from historical agricultural data
and predicts the most suitable fertilizer for a new set of conditions.

The target variable is **Recommended_Fertilizer**, which contains seven
fertilizer classes.

## Dataset

The dataset contains **10,000 samples** and combines numerical and
categorical agricultural features.

### Soil Information

-   Soil Type
-   Soil pH
-   Soil Moisture
-   Organic Carbon
-   Electrical Conductivity
-   Nitrogen Level
-   Phosphorus Level
-   Potassium Level

### Crop & Agricultural Information

-   Crop Type
-   Crop Growth Stage
-   Previous Crop
-   Irrigation Type
-   Fertilizer Used Last Season
-   Yield Last Season

### Environmental Information

-   Temperature
-   Humidity
-   Rainfall
-   Season
-   Region

### Target

-   Recommended Fertilizer

## Data Preparation

The dataset was explored to understand its structure, feature types, and
target distribution.

A complete missing-value analysis was performed. No missing values were
found across the dataset, so no missing-value imputation was required.

The target variable was separated from the input features. Categorical
features were identified, including soil type, crop type, crop growth
stage, season, irrigation type, previous crop, and region.

Different preprocessing approaches were considered according to the
requirements of each Machine Learning model. The target variable was
encoded for model training, while categorical input features were
prepared for models that required numerical representations. CatBoost
was also evaluated using its ability to work directly with categorical
features.

The dataset was divided into **80% training data and 20% testing data**,
with stratification used to preserve the distribution of fertilizer
classes.

## Class Imbalance

The distribution of the fertilizer classes was analyzed before model
training because the classes were not perfectly balanced.

To reduce the impact of class imbalance, appropriate class-weighting
approaches were used during model training.

Because accuracy alone can be misleading when classes are imbalanced,
the models were evaluated using **Accuracy, Macro Precision, Macro
Recall, and Macro F1** in addition to training time.

## 📊 Model Comparison

Four Machine Learning classification models were trained and evaluated:

| 🧠 Model | 🎯 Accuracy | 🎯 Precision | 🔄 Recall | ⭐ Macro F1 | ⏱️ Training Time |
|:---|---:|---:|---:|---:|---:|
| **XGBoost** | **87.20%** 🥇 | 75.64% | 74.05% | 74.56% | **4.86 sec** ⚡ |
| **LightGBM** | 87.15% 🥈 | 77.67% | 79.33% | 76.35% | 9.83 sec |
| **Random Forest** | 86.80% | 86.32% | 83.47% | 79.27% | 5.85 sec |
| **CatBoost** | 86.55% | **86.56%** 🥇 | **83.77%** 🥇 | **79.34%** 🥇 | 47.58 sec |

### 🏆 Performance Summary

| Metric | Best Model | Result |
|:---|:---|---:|
| **Highest Accuracy** | XGBoost | **87.20%** |
| **Highest Precision** | CatBoost | **86.56%** |
| **Highest Recall** | CatBoost | **83.77%** |
| **Highest Macro F1** | CatBoost | **79.34%** |
| **Fastest Training** | XGBoost | **4.86 sec** |

> **Key observation:** XGBoost achieved the highest overall accuracy, while CatBoost provided the strongest balance across Precision, Recall, and Macro F1. This was particularly important because the fertilizer classes were not perfectly balanced.

## Final Model

Although **XGBoost achieved the highest accuracy at 87.20%**, CatBoost
achieved the highest **Macro F1 score at 79.34%**, together with the
highest Macro Precision and Macro Recall among the evaluated models.

CatBoost was selected as the final model for the recommendation
application because of its strong overall classification performance and
its suitability for datasets containing categorical agricultural
features.

The trained model is used to generate fertilizer recommendations for
previously unseen input data.

## Interactive Interface

The project is not limited to model training and evaluation. The trained
model was integrated into a **Streamlit interactive web interface**.

The interface allows any user to enter new agricultural information,
including:

-   Soil characteristics
-   Crop information
-   Environmental conditions
-   Irrigation and regional information
-   Previous-season fertilizer usage
-   Previous-season yield

After entering the required information, the user can submit the new
data and receive a **fertilizer recommendation generated by the trained
Machine Learning model**.

### 🚀 Live Interface

**Try the deployed application:**\
`STREAMLIT_APP_URL`

https://nti-machine-learning-fertilizer-recommendation-kareem-hussien.streamlit.app/ with the deployed Streamlit
> application URL.

This makes the project usable as a real prediction application rather
than only an offline Machine Learning experiment.

## About Me

### Kareem Hussien Abdelmonm Tawfik

I am a Computer Science and Artificial Intelligence student with a focus
on **AI & Data Science** and an interest in building practical Machine
Learning and Data-driven applications.

My learning journey covers areas including **Data Analysis, Machine
Learning, Deep Learning, Generative AI, Agentic AI, and Business
Intelligence**. I also work with technologies such as Python, Pandas,
Scikit-learn, CatBoost, LightGBM, XGBoost, SQL, Excel, and Power BI.

I enjoy turning raw data into useful insights and building Machine
Learning solutions that can be integrated into practical applications.

This project represents my work in applying the complete Machine
Learning workflow, from understanding and preparing the data to
comparing models, selecting a final model, and deploying it through an
interactive interface.

## Project Workflow

**Problem Definition → Data Exploration → Data Quality Check → Feature
Preparation → Class Imbalance Analysis → Model Training → Model
Comparison → Evaluation → Model Selection → Deployment → Interactive
Prediction**

------------------------------------------------------------------------

**🌱 Fertilizer Recommendation System**\
*Machine Learning Project by Kareem Hussien Abdelmonm Tawfik*
