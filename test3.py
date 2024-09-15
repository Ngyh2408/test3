import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import string

# Download stopwords
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('Dataset-SA.csv')

# Preprocessing function
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Apply preprocessing
df['Review'] = df['Review'].apply(preprocess_text)

# Prepare data for modeling
X = df['Review']
y = df['Sentiment']

# TF-IDF transformation
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

# Initialize classifiers
models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

results = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    results[name] = {
        "accuracy": accuracy,
        "report": report
    }

# Display results for each model
for name, result in results.items():
    print(f"### {name} Statistics:")
    print(f"*Accuracy:* {result['accuracy'] * 100:.2f}%")
    print("Classification Report:")
    for label, metrics in result['report'].items():
        if label != 'accuracy':  # Skip 'accuracy' report as it's already displayed
            print(f"Label: {label}")
            print(f"  Precision: {metrics['precision']:.2f}")
            print(f"  Recall: {metrics['recall']:.2f}")
            print(f"  F1-Score: {metrics['f1-score']:.2f}")
    print("\n")
