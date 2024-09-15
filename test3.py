# Saving the provided code to a Python file named "sentiment_analysis.py"
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string

# Download stopwords if needed
nltk.download('stopwords')

# Load the dataset (adjust the path to your dataset)
df = pd.read_csv('Dataset-SA.csv')

# Check the columns in your dataset to confirm column names
print(df.columns)

# Preprocessing: Remove stopwords and punctuation from the 'Review' column
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Apply the preprocess function to the 'Review' column
df['Review'] = df['Review'].apply(preprocess_text)

# Visualize sentiment distribution in the dataset
plt.figure(figsize=(6, 4))
sns.countplot(df['Sentiment'], palette="viridis")
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Split data into features (X) and labels (y)
X = df['Review']  # Using the 'Review' column as the feature
y = df['Sentiment']  # Using the 'Sentiment' column as the label

# Convert text to TF-IDF features
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Generate and visualize the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Sentiment')
plt.ylabel('True Sentiment')
plt.show()

# Function to predict sentiment for new user input
def predict_sentiment(user_comment):
    processed_comment = preprocess_text(user_comment)
    user_comment_tfidf = tfidf.transform([processed_comment])
    prediction = model.predict(user_comment_tfidf)
    user_sentiments.append(prediction[0])
    return prediction[0]

# Visualize user sentiment predictions
user_sentiments = []

def visualize_user_sentiments():
    plt.figure(figsize=(6, 4))
    sns.countplot(user_sentiments, palette="viridis")
    plt.title('User Sentiment Predictions')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

# Get user input and predict sentiment in a loop
while True:
    user_comment = input("Enter your product review (or type 'exit' to quit): ")
    if user_comment.lower() == 'exit':
        break
    sentiment = predict_sentiment(user_comment)
    print(f"The sentiment of the comment is: {sentiment}")
    visualize_user_sentiments()
