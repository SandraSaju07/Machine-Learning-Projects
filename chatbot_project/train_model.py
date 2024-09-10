# For training the chatbot model

import os
import nltk
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')

# Load intents
with open(os.path.join('data','intents.json')) as file:
    data = json.load(file)

# Preapre data for training
tags = []
patterns = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Vectorize patterns and train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)
y = tags

# Train Logistic Regression model
clf = LogisticRegression(random_state=0, max_iter=10000)
clf.fit(X,y)

# Save model and vectorizer
with open(os.path.join('model','chatbot_model.pkl'),'wb') as model_file:
    pickle.dump(clf,model_file)
with open(os.path.join('model','vectorizer.pkl'),'wb') as vec_file:
    pickle.dump(vectorizer,vec_file)

print('Model and vectorizer saved successfully!')
