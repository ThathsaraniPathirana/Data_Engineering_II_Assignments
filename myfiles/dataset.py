from sklearn.datasets import fetch_covtype
cov_type = fetch_covtype()
cov_type.data.shape
cov_type.target.shape
# Let's check the 4 first feature names
cov_type.feature_names[:4]
