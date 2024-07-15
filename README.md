# Fake Product Review Identification
This project aims to identify fake product reviews using machine learning techniques. The system processes textual reviews and predicts whether a review is genuine or fake based on its content.

## Features
- Preprocessing: Text preprocessing techniques such as tokenization, lemmatization, and removal of stopwords are applied to clean the review text.
- Feature Extraction: TF-IDF (Term Frequency-Inverse Document Frequency) is used to extract features from the preprocessed text data.
- Model Training: A RandomForestClassifier is trained using GridSearchCV for hyperparameter tuning to classify reviews as genuine or fake.
- Model Evaluation: Performance metrics including precision, recall, F1-score, and accuracy are computed to evaluate the model's effectiveness.
- Prediction: Users can input a review to get predictions on whether it is fake or genuine, along with the probability score.
## Dataset
- The dataset used (reviews_dataset.csv) contains labeled examples of product reviews, where each review is labeled as either genuine (0) or fake (1).

## Requirements
- Python 3.x
- Libraries: pandas, numpy, nltk, scikit-learn, matplotlib
- Install the required libraries using pip:
- pip install pandas numpy nltk scikit-learn matplotlib

## Usage
1. Training the Model:
- Run python `fake_review_detector.py` to train the model.
- The script will preprocess the data, train the classifier, evaluate its performance, and save the trained model (`fake_review_model.pkl`) and TF-IDF vectorizer (`tfidf_vectorizer.pkl`).
2. Making Predictions:
- After training, the model files (`fake_review_model.pkl` and `tfidf_vectorizer.pkl`) will be saved in the project directory.
- You can input reviews interactively using the command line to predict whether they are fake or genuine.
- python fake_review_detector.py
3. Exiting the Prediction Mode:
- Type `quit` when prompted to exit the prediction mode.
- `data/reviews_dataset.csv`: CSV file containing the dataset of reviews and their labels.

## Author
- [Melisa Sever]

## Contributing
- Contributions are welcome! Please fork the repository and submit pull requests with your enhancements.
