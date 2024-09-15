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
import streamlit as st
import matplotlib.pyplot as plt

# Download stopwords
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('Dataset-SA.csv')

# Count the number of reviews in the dataset
total_reviews = len(df)

# Streamlit app header
st.title('Sentiment Analysis on Product Reviews')

# Display the total number of reviews before preprocessing
st.write(f"*Total Number of Reviews before Preprocessing:* {total_reviews}")

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
    report = classification_report(y_test, y_pred)
    
    results[name] = {
        "accuracy": accuracy,
        "report": report
    }

# Display results for each model
for name, result in results.items():
    st.write(f"### {name} Classification Report:")
    st.text(result["report"])
    st.write(f"*Accuracy:* {result['accuracy'] * 100:.2f}%")

# Plot sentiment distribution
st.write("### Sentiment Distribution (Post-Processing):")
sentiment_distribution = df['Sentiment'].value_counts()
sentiment_labels = sentiment_distribution.index
sentiment_sizes = sentiment_distribution.values

# Define colors for sentiment categories
colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightskyblue']

# Calculate percentages
sentiment_percentages = sentiment_sizes / sentiment_sizes.sum() * 100

# Plot pie chart
st.write("### Sentiment Distribution Pie Chart:")
fig, ax = plt.subplots(figsize=(8, 6))
ax.pie(sentiment_percentages, labels=sentiment_labels, autopct='%1.1f%%', startangle=140, colors=colors[:len(sentiment_labels)])
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig)

# Display the review count table
review_count_table = pd.DataFrame({'Sentiment': sentiment_labels, 'Review Count': sentiment_sizes})
st.write("### Review Count Table:")
st.table(review_count_table)

# Predict sentiment
def predict_sentiment(user_comment):
    processed_comment = preprocess_text(user_comment)
    user_comment_tfidf = tfidf.transform([processed_comment])
    prediction = models["Multinomial Naive Bayes"].predict(user_comment_tfidf)
    return prediction[0]

# User input for predicting sentiment
st.write("### Predict Sentiment from Your Review")
user_comment = st.text_input("Enter your product review:")

if user_comment:
    sentiment = predict_sentiment(user_comment)
    st.write(f"*The sentiment of the comment is:* {sentiment}")
