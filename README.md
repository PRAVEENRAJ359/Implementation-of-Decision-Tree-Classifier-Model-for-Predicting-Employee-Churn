# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load employee data and split it into training and testing sets.
2.Train a Decision Tree classifier using entropy as the split criterion.
3.Evaluate the model using accuracy, confusion matrix, and classification report.
4.Use the trained model to predict whether a new employee will stay or leave.
## Program:
```PYTHON
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

# 1) Load dataset
# Replace the path below with the correct path for your environment if needed.
csv_path = "Employee.csv"   # <-- change if file is somewhere else
try:
    data = pd.read_csv(csv_path)
except FileNotFoundError:
    raise FileNotFoundError(f"File not found at {csv_path}. Update 'csv_path' to the correct location.")

# 2) Quick inspection
print("Data shape:", data.shape)
display(data.head())
print("\nInfo:")
display(data.info())
print("\nMissing values per column:\n", data.isnull().sum())
print("\nTarget distribution (left):")
display(data["left"].value_counts())

# 3) Handle missing values (simple approach: drop rows with NA)
if data.isnull().any().any():
    print("\nDropping rows with missing values...")
    data = data.dropna()
    print("New shape after dropna:", data.shape)

# 4) Encode categorical columns
# We encode 'salary'. If you have other categorical columns (e.g., 'department'), consider encoding or one-hot.
if 'salary' in data.columns:
    le = LabelEncoder()
    data['salary'] = le.fit_transform(data['salary'].astype(str))
    print("\nSalary classes (label encoding mapping):")
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(mapping)

# 5) Define features (X) and target (y)
feature_cols = [
    "satisfaction_level", "last_evaluation", "number_project",
    "average_montly_hours", "time_spend_company",
    "Work_accident", "promotion_last_5years", "salary"
]

# Validate that all feature columns exist
missing_feats = [c for c in feature_cols if c not in data.columns]
if missing_feats:
    raise ValueError(f"Missing expected feature columns in dataset: {missing_feats}")

X = data[feature_cols]
y = data["left"]

display(X.head())

# 6) Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=100, stratify=y if len(y.unique())>1 else None
)
print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

# 7) Initialize and train Decision Tree
dt = DecisionTreeClassifier(criterion="entropy", random_state=42)
dt.fit(X_train, y_train)

# 8) Predict on test data
y_pred = dt.predict(X_test)

# 9) Evaluation
acc = metrics.accuracy_score(y_test, y_pred)
print(f"\nAccuracy on test set: {acc:.4f}")

print("\nConfusion Matrix:")
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(metrics.classification_report(y_test, y_pred, digits=4))

# 10) Example: predict for a new employee
# Format: [satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, Work_accident, promotion_last_5years, salary]
# NOTE: salary must use the same encoding as used above (see mapping printed earlier)
new_employee = [[0.5, 0.8, 9, 260, 6, 0, 1, 2]]  # example values
pred = dt.predict(new_employee)
print(f"\nPrediction for new employee {new_employee[0]}: {pred} ->", "left" if pred[0]==1 else "stayed")

# 11) Visualize the decision tree
plt.figure(figsize=(14,10))
plot_tree(
    dt,
    feature_names=feature_cols,
    class_names=['stayed','left'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree for Employee Churn")
plt.show()

# Optionally save the tree figure
# plt.savefig("employee_churn_tree.png", bbox_inches="tight", dpi=200)

# 12) (Optional) show feature importances
importances = pd.Series(dt.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importances:")
display(importances)
Developed by: PRAVEEN RAJ R
RegisterNumber:  212224230207

```

## Output:
<img width="1225" height="692" alt="image" src="https://github.com/user-attachments/assets/a43aed54-cb8c-4e43-87aa-7722943daa98" />
<img width="993" height="922" alt="image" src="https://github.com/user-attachments/assets/cbe62477-9709-4cdc-8187-fc9d904d81c3" />
<img width="1353" height="856" alt="image" src="https://github.com/user-attachments/assets/071c6ff7-bb7f-4a9e-96ff-30945c2e531d" />
<img width="972" height="296" alt="image" src="https://github.com/user-attachments/assets/fdd17883-1f9a-4040-b368-24e6e4878f8d" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
