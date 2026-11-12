from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier  # or Regressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
""""
Cheatsheet generated with DeepSeek that helped me understand how to build a pipeline
with sklearn. 
"""
# 1. Split features/target
iris = load_iris()
X = iris.data.astype(int)
y = iris.target

# 2. Identify column types (only works with pandas)
# numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
# categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

# 3. Numeric preprocessing
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler()),])

# 4. Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),])

# 5. Combine into a ColumnTransformer
# preprocessor = ColumnTransformer(transformers=[
#     ("num", numeric_transformer, numeric_features),
#     ("cat", categorical_transformer, categorical_features),])

preprocessor = ColumnTransformer(transformers=["num", numeric_transformer, iris.feature_names])

# 6. Full pipeline: preprocessing + model
pipeline = Pipeline(steps=[("classifier", RandomForestClassifier(random_state=42)),])

# 7. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 , stratify=y )
# drop stratify for regression
# 8. Fit
pipeline.fit(X_train, y_train)

# 9. Evaluate
score = pipeline.score(X_test, y_test)
print(f"Test score: {score:.3f}")

# 10. Cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print(f"CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Grid search
param_grid = {"classifier__n_estimators": [100, 300],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_leaf": [1, 5],}

grid_search = GridSearchCV(
    pipeline,param_grid,cv=5,
    scoring=None,  # When none is used, it uses the model preferred one
    n_jobs=-1,verbose=1,)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

best_pipeline = grid_search.best_estimator_