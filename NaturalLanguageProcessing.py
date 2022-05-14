import pandas as pd
import nltk
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


df = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names=["label", "text"])
sw = set(nltk.corpus.stopwords.words('english'))
cl = {'ham': 1, 'spam': 0}
df['label'] = df['label'].map(cl)


corpus = []
for i in range(0, 5572):
    text = re.sub('[^a-zA-Z]', ' ', df['text'][i])
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    corpus.append(text)

cv = CountVectorizer(max_features = 2000)
x = cv.fit_transform(corpus).toarray()
cl = df['label'].values

x_train, x_test, y_train, y_test = train_test_split(x, cl, test_size = 0.3, random_state = 12345)

# Regressione Logistica

lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)


print(confusion_matrix(y_test, lr_pred))
print(classification_report(y_test, lr_pred))
