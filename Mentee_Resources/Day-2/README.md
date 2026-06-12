# Day 2: Traditional Machine Learning Pipelines

This session provides a comprehensive examination of traditional Machine Learning utilizing the `scikit-learn` framework. The curriculum focuses on Classification—the process of assigning categorical labels to data based on learned features—and covers the complete model lifecycle from data ingestion to hyperparameter optimization.

## Conceptual Overview: The ML Pipeline

Developing a robust machine learning model requires a structured pipeline:
1.  **Exploratory Data Analysis (EDA):** Assessing data distribution, identifying missing values, and observing class imbalances.
2.  **Preprocessing & Scaling:** Standardizing feature ranges. Algorithms relying on distance metrics (e.g., K-Nearest Neighbors) are highly sensitive to unscaled data where one feature's magnitude overshadows another.
3.  **Model Training:** Fitting mathematical algorithms to the training data.
4.  **Evaluation & Tuning:** Validating the model's generalized performance using cross-validation and hyperparameter tuning frameworks.

## 1. Decision Boundary Visualization

Different algorithms mathematically partition feature space in distinct ways. The `DecisionBoundary.py` script visualizes these boundaries on a synthetic 2D dataset.

**Execution:**
```bash
python DecisionBoundary.py
```

**Key Architectural Differences:**
- **Logistic Regression:** Computes linear (straight-line) separating hyperplanes. It is computationally efficient but struggles with complex, non-linear relationships.
- **K-Nearest Neighbors (KNN):** A non-parametric model that classifies points based on the majority vote of the nearest localized data points, producing highly complex, non-linear boundaries.
- **Decision Trees:** Sequentially partitions data using orthogonal, step-wise rules, often prone to overfitting if tree depth is unconstrained.

## 2. Advanced Evaluation Metrics

Relying solely on "Accuracy" can be dangerously misleading, especially in imbalanced datasets (e.g., predicting a rare disease). The provided notebook introduces advanced metrics:
- **Confusion Matrix:** A matrix detailing True Positives, True Negatives, False Positives, and False Negatives.
- **Precision:** Of all positive predictions made by the model, what fraction were correct?
- **Recall (Sensitivity):** Of all actual positive cases in the data, what fraction did the model successfully identify?
- **F1-Score:** The harmonic mean of Precision and Recall, providing a single metric for model robustness.

## 3. Cross-Validation and Hyperparameter Tuning

- **K-Fold Cross-Validation:** A single train/test split can yield lucky or unlucky subsets. K-Fold validation partitions the data into *K* chunks, training and testing the model *K* times to compute a robust average accuracy.
- **GridSearchCV:** Machine learning models contain configuration variables called hyperparameters (e.g., the `max_depth` of a tree). Grid Search automates the process of testing combinations of hyperparameters using cross-validation to algorithmically determine the optimal model configuration.

## Lab Exercises
1. Execute `python DecisionBoundary.py` and observe the geometric differences between model architectures.
2. Launch `jupyter notebook` and open `Classification.ipynb`.
3. Execute the notebook sequentially, observing the Data Preprocessing steps and the generation of the Seaborn Confusion Matrix.
4. Analyze the `GridSearchCV` output to understand how the optimal Decision Tree depth was programmatically selected.
