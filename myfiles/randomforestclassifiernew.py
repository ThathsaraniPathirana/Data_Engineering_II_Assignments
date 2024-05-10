import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, cross_val_score

# Start the timer
start_time = time.time()

print("Fetching data...")
data = fetch_covtype()
X = data.data
y = data.target
print("Data fetched successfully.")

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Data split into train and test sets.")

# Creating the RandomForest model with default parameters
rf = RandomForestClassifier()
print("RandomForest model created with default parameters.")

# Checking parameters
print("Parameters currently in use:\n")
print(rf.get_params())

# Training the model
print("Training the model...")
rf.fit(X_train, y_train)
print("Model training completed.")

# Evaluating the model
score = rf.score(X_test, y_test)
print(f"Accuracy of the RandomForest with default parameters: {score:.2%}")

# Cross-validation score
print("Performing cross-validation...")
cv_scores = cross_val_score(rf, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average cross-validation score: {cv_scores.mean():.2%}")

# Stop the timer
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")
