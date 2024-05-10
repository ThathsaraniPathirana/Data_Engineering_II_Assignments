from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split

# Fetching the dataset
print("Fetching the dataset...")
data = fetch_covtype()
X = data.data
y = data.target
print("Dataset fetched successfully.")

# Splitting the dataset into training and testing
print("Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Dataset split successfully.")

# Creating the RandomForest model with default parameters
rf = RandomForestClassifier()
print("RandomForest model created with default parameters.")

# Exploring parameters
print("Exploring the parameters of the model...")
params = rf.get_params()
for param in params:
    print(f"{param}: {params[param]}")

# Training the model
print("Training the model...")
rf.fit(X_train, y_train)
print("Model training completed.")

# Optionally, evaluate the model on the test set
score = rf.score(X_test, y_test)
print(f"Accuracy of the RandomForest with default parameters on the test set: {score:.2%}")
