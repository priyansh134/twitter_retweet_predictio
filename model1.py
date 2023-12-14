
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv('Final_data.csv')


# Assuming you have a DataFrame 'df' containing the numeric columns
# Extract the numeric columns from the DataFrame
numeric_columns = df[['Hashtag count', 'Punctuation count', 'Character count', 'word_count', 'Mentions Count',  'Acronym Count', 'URL Count']]

# Convert the numeric columns to a sparse matrix


# Slice the numeric matrix to retain only the first 70900 rows

# Stack the TF-IDF matrix and the modified numeric matrix horizontally


X=numeric_columns
y = df['retweet']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
X_train

#y_train
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to the training data
rf_classifier.fit(X_train, y_train)


svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)

# Fit the classifier to the training data
svm_classifier.fit(X_train, y_train)
pickle.dump(rf_classifier, open('model1.pkl','wb'))

model = pickle.load(open('model1.pkl','rb'))








