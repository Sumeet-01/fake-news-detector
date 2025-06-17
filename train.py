import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
import os

# Load the data
fake_df = pd.read_csv(r"D:\Moto Edge 50\Projects\Data science projects\fake-news-detector\archive\Fake.csv")
true_df = pd.read_csv(r"D:\Moto Edge 50\Projects\Data science projects\fake-news-detector\archive\True.csv")

# Add labels
fake_df['label'] = 0
true_df['label'] = 1

# Combine and shuffle
data = pd.concat([fake_df, true_df], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# Preprocessing
X = data['text']
y = data['label']

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_tfidf = tfidf.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.25, random_state=42)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model/fake_news_model.pkl", "wb"))
pickle.dump(tfidf, open("model/tfidf_vectorizer.pkl", "wb"))

print("✅ Model and vectorizer saved in 'model/' folder.")
