import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
import string
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

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

# Split data
X = df['Review']
y = df['Sentiment']

# TF-IDF transformation
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit app
st.title('Sentiment Analysis on Product Reviews')

st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Bar Chart for Sentiment Distribution
st.subheader("Sentiment Distribution (Predicted vs Actual)")

# Create a DataFrame for visualization
sentiment_counts = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
actual_counts = sentiment_counts['Actual'].value_counts().reset_index().rename(columns={'index': 'Sentiment', 'Actual': 'Count'})
predicted_counts = sentiment_counts['Predicted'].value_counts().reset_index().rename(columns={'index': 'Sentiment', 'Predicted': 'Count'})

# Merge actual and predicted counts
sentiment_comparison = pd.merge(actual_counts, predicted_counts, on='Sentiment', how='outer', suffixes=('_Actual', '_Predicted')).fillna(0)

# Plot bar chart
fig, ax = plt.subplots(figsize=(8, 6))
sentiment_comparison.plot(kind='bar', x='Sentiment', ax=ax, color=['skyblue', 'orange'])
plt.title('Actual vs Predicted Sentiment Counts')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=45)
st.pyplot(fig)

# Bar Chart for Confusion Matrix as True/False Predictions
st.subheader("True/False Predictions Distribution")

# Calculate True Positives, False Positives, True Negatives, and False Negatives
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Bar chart for TP, FP, TN, FN
fig2, ax2 = plt.subplots(figsize=(8, 6))
categories = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
values = [tn, fp, fn, tp]

ax2.bar(categories, values, color=['green', 'red', 'red', 'green'])
plt.title('Prediction Outcomes')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.xticks(rotation=45)
st.pyplot(fig2)

# Predict sentiment
def predict_sentiment(user_comment):
    processed_comment = preprocess_text(user_comment)
    user_comment_tfidf = tfidf.transform([processed_comment])
    prediction = model.predict(user_comment_tfidf)
    return prediction[0]

user_comment = st.text_input("Enter your product review:")

if user_comment:
    sentiment = predict_sentiment(user_comment)
    st.write(f"The sentiment of the comment is: {sentiment}")
