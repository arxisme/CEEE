import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def plot_decision_boundaries():
    print("Generating Decision Boundary Visualizations for Class...")
    
    # 1. Create a toy 2D dataset
    X, y = make_moons(noise=0.3, random_state=42)
    X = StandardScaler().fit_transform(X)

    # 2. Define the models to compare
    models = {
        "Logistic Regression": LogisticRegression(),
        "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
        "Decision Tree (depth=5)": DecisionTreeClassifier(max_depth=5)
    }

    # 3. Setup the plotting grid
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Color maps
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    cm_bg = ListedColormap(['#FFAAAA', '#AAAAFF'])

    # 4. Plot each model
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, (name, model) in zip(axes, models.items()):
        # 1. Train the model on the existing data
        model.fit(X, y)
        
        # 2. Predict over the entire imaginary grid to draw the boundary
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # 3. Plot the contour boundary based on the predictions
        ax.contourf(xx, yy, Z, cmap=cm_bg, alpha=0.8)
        
        # Plot the actual data points
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors='k')
        
        ax.set_title(name, fontsize=14)
        ax.set_xticks(())
        ax.set_yticks(())
        
    plt.suptitle("How Different Models Separate Data", fontsize=18)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_decision_boundaries()