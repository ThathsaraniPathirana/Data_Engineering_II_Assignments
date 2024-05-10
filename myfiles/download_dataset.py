from sklearn.datasets import fetch_covtype

# Download the forest cover type dataset
print("Downloading forest cover type dataset...")
covtype = fetch_covtype()
print("Download completed.")

# Access the features and target labels
X, y = covtype.data, covtype.target
