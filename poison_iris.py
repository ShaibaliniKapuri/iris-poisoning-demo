import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def poison_data(X_train, y_train, poisoning_level):
    """
    Poisons a fraction of the training data.

    Args:
        X_train (np.array): Original training features.
        y_train (np.array): Original training labels.
        poisoning_level (float): The fraction of data to poison (e.g., 0.1 for 10%).

    Returns:
        A tuple of (X_poisoned, y_poisoned).
    """
    if poisoning_level == 0:
        return X_train.copy(), y_train.copy()

    # Determine the number of samples to poison
    num_to_poison = int(len(X_train) * poisoning_level)
    if num_to_poison == 0:
        return X_train.copy(), y_train.copy()

    # Make copies to avoid modifying original data
    X_poisoned = X_train.copy()
    y_poisoned = y_train.copy()

    # Get the number of features and classes
    num_features = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    # Randomly select indices to poison
    indices_to_poison = np.random.choice(len(X_train), num_to_poison, replace=False)

    # Inject poison: replace features with random noise and flip labels
    # The random noise is generated in a range far from the typical Iris data values.
    poison_features = np.random.rand(num_to_poison, num_features) * 100
    X_poisoned[indices_to_poison] = poison_features

    # Flip labels to a different class
    y_poisoned[indices_to_poison] = (y_poisoned[indices_to_poison] + 1) % num_classes

    print(f"Poisoned {num_to_poison} out of {len(X_train)} training samples.")
    return X_poisoned, y_poisoned


# 1. Load and Split Data
iris = load_iris()
X, y = iris.data, iris.target
# We create a clean, untouched validation set to properly evaluate the model
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# 2. Define Poisoning Levels
poisoning_levels = [0.0, 0.05, 0.10, 0.50]


# 3. Train and Evaluate for Each Level
print("--- Starting Data Poisoning Experiment on Iris Dataset ---")
for level in poisoning_levels:
    print(f"\n----- TESTING POISONING LEVEL: {level*100:.0f}% -----")

    # Poison the training data
    X_train_poisoned, y_train_poisoned = poison_data(X_train, y_train, level)

    # Initialize and train a Support Vector Machine (SVM) model
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train_poisoned, y_train_poisoned)

    # Evaluate the model on the CLEAN validation set
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    print(f"==> Validation Accuracy: {accuracy:.2%}")

print("\n--- Experiment Complete ---")
