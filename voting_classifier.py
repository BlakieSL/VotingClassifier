import numpy as np  # Import NumPy library as np, used for numerical computations
import matplotlib.pyplot as plt  # Import pyplot module from Matplotlib, used for plotting graphs
from sklearn.datasets import make_moons  # Import make_moons function from scikit-learn, generates data with two opposing moon shapes
from sklearn.model_selection import train_test_split  # Import train_test_split function from scikit-learn, used to split data into training and test sets
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier class from scikit-learn, used for classification with random forests
from sklearn.linear_model import LogisticRegression  # Import LogisticRegression class from scikit-learn, used for classification with logistic regression
from sklearn.svm import SVC  # Import SVC (Support Vector Classifier) class from scikit-learn, used for classification with Support Vector Machine (SVM)
from sklearn.ensemble import VotingClassifier  # Import VotingClassifier class from scikit-learn, used to combine multiple classifiers into one group
from sklearn.metrics import accuracy_score  # Import accuracy_score function from scikit-learn, used to calculate classification accuracy

def plot_decision_boundary(clf, X, X_train, y_train, X_test, y_test, ax, title):
    # Create grid for decision boundary plot
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
        np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
    )# np.linspace creates evenly spaced values over a specified range, x[:, 0] means all rows from first column, .min() minimum value from feature 1

    # Predict for each grid point
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot data points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='.', label='Train')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', label='Test')

    # Draw decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3)

    # Set plot settings
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(title)

# Step 1: Create dataset
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)  # Generate dataset using make_moons function

# Step 2: Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split dataset into training and test sets

# Step 3: Create classifiers
svm_clf = SVC(random_state=42)
log_reg_clf = LogisticRegression(random_state=42)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 4: Combine classifiers into VotingClassifier
voting_clf = VotingClassifier(
    estimators=[
        ('svm', svm_clf),
        ('log_reg', log_reg_clf),
        ('rf', rf_clf)
    ],
    voting='hard'
)

# Step 5: Train VotingClassifier model
voting_clf.fit(X_train, y_train)   # Train VotingClassifier model on training data

# Step 6: Evaluate model
train_accuracy = accuracy_score(y_train, voting_clf.predict(X_train))
test_accuracy = accuracy_score(y_test, voting_clf.predict(X_test))

# Step 7: Plot results
fig, ax = plt.subplots(figsize=(10, 10))  # Set plot size

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='.', label='Train')  # Plot training data points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', label='Test')  # Plot test data points

xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),  # Create grid for decision boundary plot
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))

Z = voting_clf.predict(np.c_[xx.ravel(), yy.ravel()])  # Make predictions across entire grid
Z = Z.reshape(xx.shape)  # Reshape prediction results to original grid shape

plt.contourf(xx, yy, Z, alpha=0.3)  # Draw decision boundary based on VotingClassifier predictions
plt.xlabel('Feature 1')  # Set x-axis label
plt.ylabel('Feature 2')  # Set y-axis label
plt.title(f"VotingClassifier\nTrain acc:{train_accuracy:.4f}, Test acc:{test_accuracy:.4f}")  # Set plot title


plt.legend()  # Add legend
plt.tight_layout()  # Adjust plot layout
plt.show()  # Display plot
plt.savefig("03_voting_classifier.png")  # Save plot to file