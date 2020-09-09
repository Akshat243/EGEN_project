import numpy as np
import re
import nltk
import pandas as pd
nltk.download('stopwords')
from nltk.corpus import stopwords
import joblib
from sklearn.pipeline import make_pipeline

data = pd.read_csv(r"amazon_alexa.csv", encoding = "ISO-8859-1")
data.head()

reviews = data["verified_reviews"].tolist()
labels = data["feedback"].values

processed_reviews = []

for text in range(0, len(reviews)):
    text = re.sub(r'\W', ' ', str(reviews[text]))
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r'^b\s+', '', text)
    processed_reviews.append(text)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(processed_reviews, labels, test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.75, stop_words=stopwords.words('english'))
X_train1 = vectorizer.fit_transform(X_train).toarray()
X_test1 = vectorizer.transform(X_test).toarray()

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200, random_state=42)
rfc.fit(X_train1, y_train)

pipe = make_pipeline(TfidfVectorizer(max_features=2000, min_df=5, max_df=0.75, stop_words=stopwords.words('english')), RandomForestClassifier(n_estimators=200, random_state=42))
pipe.fit(X_train, y_train)
joblib.dump(pipe, 'alexamodel.pkl')

y_pred = rfc.predict(X_test1)

from sklearn.metrics import  accuracy_score
print(accuracy_score(y_test, y_pred))

import joblib
filename = 'alexa_model.sav'
joblib.dump(rfc, filename)

