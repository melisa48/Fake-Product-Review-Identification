import pandas as pd
import numpy as np
import os
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

# Download NLTK stopwords and WordNet data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenize text
    tokens = word_tokenize(text)

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in tokens if word.isalpha()])

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text


def train_model():
    print("Current working directory:", os.getcwd())
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("Script directory:", script_dir)

    # Construct the full path to the CSV file
    csv_path = os.path.join(script_dir, 'data', 'reviews_dataset.csv')
    print("Looking for CSV file at:", csv_path)

    # Check if the file exists
    if os.path.exists(csv_path):
        print("CSV file found!")
    else:
        print("CSV file not found!")
        print("Contents of the 'data' directory:")
        data_dir = os.path.join(script_dir, 'data')
        if os.path.exists(data_dir):
            print(os.listdir(data_dir))
        else:
            print("'data' directory not found!")
        return  # Exit the function if the file is not found

    # Load your dataset from the data folder
    try:
        df = pd.read_csv(csv_path)
        print("CSV file loaded successfully!")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return  # Exit the function if there's an error loading the file

    # Assuming your dataset has 'review_text' and 'is_fake' columns
    X = df['review_text']
    y = df['is_fake']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Apply preprocessing
    X_train_processed = X_train.apply(preprocess_text)
    X_test_processed = X_test.apply(preprocess_text)

    # Extract features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_features = vectorizer.fit_transform(X_train_processed)
    X_test_features = vectorizer.transform(X_test_processed)

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30]
    }
    clf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_features, y_train)
    clf = grid_search.best_estimator_

    # Train the model with the best hyperparameters
    clf.fit(X_train_features, y_train)

    # Make predictions
    y_pred = clf.predict(X_test_features)

    # Print evaluation metrics
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    # Plot ROC curve
    plot_roc_curve(clf, X_test_features, y_test)

    # Save the model and vectorizer
    with open(os.path.join(script_dir, 'fake_review_model.pkl'), 'wb') as f:
        pickle.dump(clf, f)
    with open(os.path.join(script_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)

    print("Model and vectorizer saved successfully!")


def load_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, 'fake_review_model.pkl'), 'rb') as f:
        clf = pickle.load(f)
    with open(os.path.join(script_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    return clf, vectorizer


def predict_fake_review(review_text, clf, vectorizer):
    processed_text = preprocess_text(review_text)
    features = vectorizer.transform([processed_text])
    prediction = clf.predict(features)
    probability = clf.predict_proba(features)
    return "Fake" if prediction[0] == 1 else "Genuine", probability[0][1]


def plot_roc_curve(clf, X_test, y_test):
    y_score = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    # Train the model
    train_model()

    try:
        # Load the trained model
        clf, vectorizer = load_model()

        # Example usage
        while True:
            review = input("Enter a review (or 'quit' to exit): ")
            if review.lower() == 'quit':
                break
            result, prob = predict_fake_review(review, clf, vectorizer)
            print(f"The review is predicted to be: {result}")
            print(f"Probability of being fake: {prob:.2f}")
    except FileNotFoundError:
        print("Error: Model files not found. Make sure to train the model first.")
    except Exception as e:
        print(f"An error occurred: {e}")

