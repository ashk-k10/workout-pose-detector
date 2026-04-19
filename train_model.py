import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

csv_file = 'data/training_data.csv'

if not os.path.exists(csv_file):
    print("ERROR: data/training_data.csv not found!")
    print("Run extract_data.py first.")
    exit()

df = pd.read_csv(csv_file)
print(f"Total frames loaded: {len(df)}")
print(f"Classes found: {list(df['label'].unique())}")

X = df.drop('label', axis=1).values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTraining model... please wait...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

with open('workout_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved as workout_classifier.pkl")
print("You are ready to deploy!")