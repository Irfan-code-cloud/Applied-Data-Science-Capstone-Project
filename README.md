# Applied-Data-Science-Capstone-Project
# 🚀 SpaceX Falcon 9 First Stage Landing Prediction

## Executive Summary

This project develops a machine learning pipeline to **predict the successful landing of the SpaceX Falcon 9 first stage booster**. The ability to predict a successful landing is crucial, as first-stage reusability is the primary factor driving SpaceX's significantly lower launch costs (\$62 million vs. competitors' >\$165 million). This predictive model can be used by alternate launch providers to better understand and bid against SpaceX.

## Project Goal

The primary goal is to determine, with high accuracy, whether a given Falcon 9 launch will result in a successful first stage landing, based on mission characteristics such as **Payload Mass**, **Orbit Type**, **Launch Site**, and other technical specifications.

## Data and Methodology

The project follows a standard machine learning methodology:

### 1. Data Processing
* **Data Collection & Wrangling:** Gathering historical Falcon 9 launch data, including features related to the launch vehicle and mission.
* **Feature Engineering & Formatting:** Transforming categorical features into numerical formats (e.g., using one-hot encoding).
* **Standardization:** Scaling numerical features ($\mathbf{X}$) using `StandardScaler` to ensure all features contribute equally to the model training process.
* **Target Variable ($\mathbf{Y}$):** Creating a binary `Class` column where $\mathbf{1}$ indicates a successful landing and $\mathbf{0}$ indicates a failure (or planned ocean splashdown).

### 2. Exploratory Data Analysis (EDA)
* Interactive data visualization was performed to explore correlations between various launch features (e.g., Payload Mass, Orbit Type) and the final mission outcome.
* **Key Finding:** Visualization confirmed that certain features, like **Orbit Type** (e.g., Polar, LEO, ISS) and **Payload Mass**, show a clear correlation with the positive landing rate.

### 3. Model Training & Selection
Four primary machine learning classification algorithms were used:

| Algorithm | Tuning Strategy |
| :--- | :--- |
| **Logistic Regression** | Tuned for optimal regularization parameter $\mathbf{C}$ |
| **Support Vector Machine (SVM)** | Tuned for best $\mathbf{kernel}$ (Linear, RBF, Poly, Sigmoid) and $\mathbf{C}$ |
| **Decision Tree Classifier** | Tuned for $\mathbf{max\_depth}$, $\mathbf{criterion}$ (gini/entropy), and other parameters |
| **K-Nearest Neighbors (KNN)** | Tuned for optimal number of neighbors $\mathbf{K}$ |

All models were tuned using **`GridSearchCV`** with **10-fold cross-validation** on the training data to find the best set of hyperparameters.

## 🏆 Results and Conclusion

The performance of all optimized models was evaluated on an independent **test set** (20% of the total data) to determine their real-world predictive ability.

Based on the test data analysis, the **Decision Tree Classifier** emerged as the **best-performing model**.

| Model | Cross-Validation Score (Training Data) | Test Data Accuracy |
| :--- | :--- | :--- |
| Logistic Regression | $\approx 0.847$ | **(Varies)** |
| Support Vector Machine (SVM) | $\approx 0.875$ | **(Varies)** |
| **Decision Tree Classifier** | **(Varies)** | $\approx \mathbf{0.8333}$ |
| K-Nearest Neighbors (KNN) | **(Varies)** | **(Varies)** |

The high accuracy of the Decision Tree model confirms its effectiveness in learning the complex, non-linear patterns within the Falcon 9 launch data.

## 🛠️ Technologies Used

* **Python**
* **Pandas** and **NumPy** for data manipulation
* **Matplotlib** and **Seaborn** for data visualization
* **Scikit-learn** (`sklearn`) for machine learning:
    * `StandardScaler`
    * `train_test_split`
    * `GridSearchCV`
    * `LogisticRegression`, `SVC`, `DecisionTreeClassifier`, `KNeighborsClassifier`
