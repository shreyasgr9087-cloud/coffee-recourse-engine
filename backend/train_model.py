import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight
import joblib

df = pd.read_csv("coffee_dataset.csv")

le = LabelEncoder()
df['roast_level_encoded'] = le.fit_transform(df['roast_level'])

feature_names = ['roast_level_encoded', 'grind_size_microns', 'water_temp_c', 'brew_time_seconds', 'water_ratio']
X = df[feature_names]
y = df['taste_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Running Grid Search with Sample Weights...")

# CLAUDE'S FIX: Compute weights so the model doesn't just guess "Bitter"
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4]
}

gb = GradientBoostingClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=gb, 
    param_grid=param_grid, 
    cv=3, 
    n_jobs=-1,
    scoring='f1_macro' # CLAUDE'S FIX: Use F1 Macro instead of raw accuracy
)

grid_search.fit(X_train, y_train, sample_weight=sample_weights)

best_model = grid_search.best_estimator_
best_pred = best_model.predict(X_test)
best_acc = accuracy_score(y_test, best_pred)

print(f"Best Parameters Found: {grid_search.best_params_}")
print(f"Champion Model Accuracy: {best_acc * 100:.2f}%\n")
print(classification_report(y_test, best_pred))

feature_bounds = {
    f: (float(X_train[f].min()), float(X_train[f].max()), float(X_train[f].std()))
    for f in feature_names
}

joblib.dump({
    'model': best_model, 
    'encoder': le,
    'feature_names': feature_names,
    'feature_bounds': feature_bounds
}, "model.pkl")

print("Elite model package saved to 'model.pkl'!")