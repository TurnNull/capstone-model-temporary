# 📊 Predictive Analytics Module

This project utilizes Linear Regression, Random Forest, and XGBoost models to predict rice yield based on climate and production data, with XGBoost identified as the superior model for its stability and robustness. As summarized in the evaluation table, XGBoost achieved the best performance on validation data with an RMSE of 25,687.45 and an $R^2$ score of 0.9857, effectively outperforming the other algorithms while demonstrating the most reliable generalization capabilities.

# 🛠 Tech Stack

<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white"/> <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white"/> <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white"/> <img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIj48cGF0aCBkPSJNMyAzdjE4aDE4Ii8+PHBhdGggZD0iTTcgMTRsMy0zIDQgNCA1LTYiLz48L3N2Zz4=&logoColor=white"/> <img src="https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxlbGxpcHNlIGN4PSI2IiBjeT0iMTIiIHJ4PSIyIiByeT0iOCIvPjxlbGxpcHNlIGN4PSIxMiIgY3k9IjEyIiByeD0iMyIgcnk9IjkiLz48ZWxsaXBzZSBjeD0iMTgiIGN5PSIxMiIgcng9IjIiIHJ5PSI3Ii8+PC9zdmc+&logoColor=white"/> <img src="https://img.shields.io/badge/Statsmodels-3B7DDD?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIj48cGF0aCBkPSJNMyAyMGgxOE0zIDIwQzMgMTAgNyA0IDEyIDRzOSA2IDkgMTYiLz48L3N2Zz4=&logoColor=white"/> <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/> <img src="https://img.shields.io/badge/XGBoost-006ACC?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxjaXJjbGUgY3g9IjEyIiBjeT0iNCIgcj0iMiIvPjxjaXJjbGUgY3g9IjYiIGN5PSIxMiIgcj0iMiIvPjxjaXJjbGUgY3g9IjE4IiBjeT0iMTIiIHI9IjIiLz48Y2lyY2xlIGN4PSIzIiBjeT0iMjAiIHI9IjEuNSIvPjxjaXJjbGUgY3g9IjkiIGN5PSIyMCIgcj0iMS41Ii8+PGNpcmNsZSBjeD0iMTUiIGN5PSIyMCIgcj0iMS41Ii8+PGNpcmNsZSBjeD0iMjEiIGN5PSIyMCIgcj0iMS41Ii8+PHBhdGggZD0iTTEyIDZMNiAxME0xMiA2bDYgNE02IDE0bC0zIDRNNiAxNGwzIDRNNiAxNGwzIDRNMTggMTRsLTMgNE0xOCAxNGwzIDQiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMS41IiBmaWxsPSJub25lIi8+PC9zdmc+&logoColor=white"/> <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white"/> <img src="https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white"/> <img src="https://img.shields.io/badge/Joblib-1F77B4?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik0xNyAzSDVjLTEuMTEgMC0yIC45LTIgMnYxNGMwIDEuMS44OSAyIDIgMmgxNGMxLjEgMCAyLS45IDItMlY3bC00LTR6bS01IDE2Yy0xLjY2IDAtMy0xLjM0LTMtM3MxLjM0LTMgMy0zIDMgMS4zNCAzIDMtMS4zNCAzLTMgM3ptMy0xMEg1VjVoMTB2NHoiLz48L3N2Zz4=&logoColor=white"/>

# 📂 Data Pipeline
| Stage | Process | Description |
| :--- | :--- | :--- |
| **1. Acquisition** | Data Source | Data gathered from [BPS Indonesia](https://drive.google.com/drive/folders/1pkZYGE-BFCTxWRFDj6iGC-lc8rEWunfz?usp=drive_link) [NASA POWES](https://drive.google.com/drive/folders/1cQ_a_sCR0P-Sl6VrhRNRoKtOZndseEIz?usp=drive_link). |
| **2. Cleaning** | Data Preprocessing  | Handling missing values, removing outliers, and normalizing column names. |
| **3. Engineering** | Feature Extraction | Transforming variables to enhance the predictive power of the model. |
| **4. Training** | Model Development | Training multiple models (LR, RF, XGB) with 5-fold cross-validation. |
| **5. Evaluation** | Performance Analysis | Generating metrics, logs, and artifacts (.joblib) for deployment. |

# ⚙️ Models

### Dataset Overview & Splitting
The dataset consists of 2,529 total entries, which were split to ensure robust model training and evaluation:
- The dataset comprises 2,529 entries
   - 2,023 rows (80%) for training
   - 506 rows (20%) for validation
   - 506 rows (20%) for final testing.

This project utilizes three supervised learning algorithms to predict rice yield. The models were evaluated using 5-fold cross-validation to ensure reliable performance and to minimize the risk of overfitting.

| Model | Purpose |
| :--- | :--- |
| Linear Regression | Baseline Prediction |
| Random Forest | Non-linear Modeling |
| XGBoost | High Performance |

# 🚀 Usage

To run this project on your local machine, please follow these steps:
### 1. Clone the Repository

First, download the project to your local computer:
```
git clone https://github.com/TurnNull/capstone-model-temporary/
cd your-project-repo
```
### 2. Install Dependencies

```
pip install -r requirements.txt
```
### 3. Running the Experiments (Jupyter Notebook)

All exploration and modeling processes are located in the notebooks/ folder. You can open them using Jupyter Notebook:
- exploratory_data analyisis/ (Exploratory Data Analysis)
   - eda_bps.ipynb
   - eda_nasa.ipynb
   - merge_data.ipynb
- linear_regression.ipynb (Baseline Linear Regression Model Tuning)
- random_forest.ipynb (Random Forest Tuning)
- xgboost.ipynb (XGBoost Tuning)
- model_comparison.ipynb (Model Comparison and Evaluation)

# 📈 Results

### Model Performance Summary
The following table provides a comparison of the evaluation metrics (MAE, RMSE, and $R^2$) on the validation data:
| Model | Train MAE | Val MAE | Val RMSE | Val $R^2$ |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | 18.419,16 | 18.484,60 | 32.594,62 | 0,9770 |
| **Random Forest** | 4.937,11 | 13.242,58 | 27.241,72 | 0,9840 |
| **XGBoost** | 8.187,66 | 13.704,89 | 25.687,45 | 0,9857 |

### Analysis
- Top Performing Model: XGBoost was selected as the primary model, achieving the lowest RMSE (25.68) and the highest $R^2$ score (0.9857) on the validation data, which demonstrates excellent stability and predictive accuracy.
- Overfitting Assessment: While the Random Forest model showed a significant performance gap between the training and validation sets indicating overfitting the XGBoost model demonstrated more consistent results, proving to be more robust and better suited for making predictions on new, unseen data.