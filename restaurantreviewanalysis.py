import numpy as np
import pandas as pd
dataset = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/RestaurantReviews.tsv', delimiter = '\t')
dataset.head()
from google.colab import drive
drive.mount('/content/drive')
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


nltk.download('stopwords')


corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()

    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    review = [ps.stem(word)
              for word in review if not word in set(all_stopwords)]

    review = ' '.join(review)
    corpus.append(review)

# Use TF-IDF Vectorizer instead of CountVectorizer
tfidf = TfidfVectorizer(max_features=1000)

X = tfidf.fit_transform(corpus).toarray()

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, dataset['Liked'][:1000], test_size=0.1, random_state=43)

# Reduce the number of trees and set appropriate hyperparameters for RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=10, criterion='entropy', random_state=60)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calculate and print the training accuracy and test accuracy
train_acc = round(accuracy_score(y_train, model.predict(X_train)) * 100, 2)
test_acc = round(accuracy_score(y_test, y_pred) * 100, 2)

print("Training Accuracy: {}%".format(train_acc))
print("Test Accuracy: {}%".format(test_acc))
---------------------------------------------END OF CODE-----------------------------------------------------------------------------------------
              
#After running these code we get our 
#             1)Test Accuracy
#              2)Training Accuracy
#              [nltk_data] Downloading package stopwords to /root/nltk_data...
#[nltk_data]   Package stopwords is already up-to-date!
#Training Accuracy: 86.78%
#Test Accuracy: 82.0% 
#
