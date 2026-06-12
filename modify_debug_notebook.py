import json

file_path = r'd:\Downloads\CEEE\CEEE\Day-4\AI_Assisted_debugging.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Task 4 Fixed Code
notebook['cells'][3]['source'] = [
    "# Write your fixed code here:\n",
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

# AI Prompt Documentation
notebook['cells'][4]['source'] = [
    "### AI Prompt Documentation\n",
    "Double-click this cell and paste the exact prompt you used with ChatGPT, and a 2-sentence summary of what it taught you about Data Leakage.\n",
    "\n",
    "**My Prompt:** What is wrong with the data scaling in this Machine Learning code? How is it causing data leakage?\n",
    "\n",
    "**What I Learned:** Data leakage occurs when the model is inadvertently exposed to test data during the training phase. In this case, scaling the entire dataset before splitting calculated the mean and variance across all data, meaning information about the test set leaked into the training set's scaled features."
]

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)
print('Updated Day-4/AI_Assisted_debugging.ipynb')
