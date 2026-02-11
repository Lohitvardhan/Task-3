import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib

# Generate sample review data
np.random.seed(42)
reviews = [
    "great product love it", "terrible quality waste money", "good value works well",
    "absolutely horrible never buy", "excellent service fast delivery", "poor performance"
] * 500

sentiments = [1, 0, 1, 0, 1, 0] * 500
data = pd.DataFrame({'review': reviews, 'sentiment': sentiments})

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    data['review'], data['sentiment'], test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
    ('clf', LogisticRegression(random_state=42))
])

pipeline.fit(X_train, y_train)
train_acc = pipeline.score(X_train, y_train)
test_acc = pipeline.score(X_test, y_test)

# Save model
joblib.dump(pipeline, 'sentiment_model.pkl')
print(f"Model saved. Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
