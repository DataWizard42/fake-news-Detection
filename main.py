import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report



# Assuming you have a CSV file named 'fake_news_dataset.csv'
df = pd.read_csv(r"C:\Users\ayyhh\OneDrive\Desktop\fake news Detection\archive (1)\True.csv\True.csv")


# Assuming your dataset has a 'text' column containing the news articles and a 'label' column indicating whether it's fake or not
X = df['text']
y = df['title']

# Convert text to numerical features using TF-IDF (Term Frequency-Inverse Document Frequency)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_tfidf = tfidf_vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print('Classification Report:')
print(classification_report(y_test, y_pred))

