import json

file_path = r'd:\Downloads\CEEE\CEEE\Day-4\AI_Assisted_coding.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Task 1
notebook['cells'][2]['source'] = [
    "from sklearn.datasets import load_iris\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "# 1. Load the Iris dataset\n",
    "iris = load_iris()\n",
    "X = iris.data\n",
    "y = iris.target\n",
    "\n",
    "# 2. Split into train/test sets (80% train, 20% test)\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# 3. Train a Random Forest Classifier\n",
    "model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# 4. Predict and print the accuracy\n",
    "predictions = model.predict(X_test)\n",
    "accuracy = accuracy_score(y_test, predictions)\n",
    "print(f'Accuracy: {accuracy * 100:.2f}%')\n"
]

# Task 2
notebook['cells'][5]['source'] = [
    "**What did the AI say?** (Double click to edit)\n",
    "- Answer: This is a one-liner function that generates a list of prime numbers up to `x` (exclusive). It iterates through numbers `i` from 2 to `x-1`, and for each `i`, checks if it is divisible by any number `j` from 2 up to the square root of `i`. If `all()` the remainders are not zero, the number `i` is prime and added to the list."
]

# Task 3
notebook['cells'][8]['source'] = [
    "# FIXED CODE: Split first, then scale!\n",
    "import pandas as pd\n",
    "from sklearn.datasets import load_breast_cancer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "data = load_breast_cancer()\n",
    "X = data.data\n",
    "y = data.target\n",
    "\n",
    "# 1. Split the data FIRST to prevent data leakage\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# 2. Fit the scaler ONLY on the training data, then transform both\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "# 3. Train and predict\n",
    "model = LogisticRegression()\n",
    "model.fit(X_train_scaled, y_train)\n",
    "preds = model.predict(X_test_scaled)\n",
    "\n",
    "print(f'Fixed Model Accuracy: {accuracy_score(y_test, preds):.4f}')\n"
]

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)
print('Updated Day-4/AI_Assisted_coding.ipynb')
