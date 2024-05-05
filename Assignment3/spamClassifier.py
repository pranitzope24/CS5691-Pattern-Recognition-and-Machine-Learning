import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import re
import unidecode
import contractions
from nltk.stem import PorterStemmer
import spacy
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from wordcloud import WordCloud

nlp = spacy.load('en_core_web_sm')
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')



df=pd.read_csv('dataset.csv')
df.head(10)
df=df.drop(['Unnamed: 0'], axis=1)
df = df.sample(frac = 1,random_state=200) #shuffle database

def remove_punctuation(text):
    punctuationfree = re.sub(r'[^\w\s]|_',' ',text)
    return punctuationfree

def replace_numbers(text):
    numbers_free = re.sub(r'\d+(\.\d+)?',' ', text)
    return numbers_free

def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output

def replace_url(text):
    result = re.sub(r'http\S+', ' Links ', text)
    return result

def limit_characters(text):
    output = re.sub(r'(.)\1{2,}', r'\1\1',text)
    return output

def replace_currency_signs(text):
    replaced_text=re.sub(r'£|\$', ' ruppees ',text)
    return replaced_text

def replace_mailid(text):
    replacement= re.sub(r'^.+@[^\.].*\.[a-z]{2,}$',' ',text)
    return replacement

def accented_to_ascii(text):
    text = unidecode.unidecode(text)
    text = re.sub(r"[^a-zA-Z0-9]+", " ",text)
    return text

def lemmatize(text):
    doc = nlp(text)
    lemmatized_sentence = " ".join([token.lemma_ for token in doc])
    return [token.lemma_ for token in doc]

def stem(token_array):
    ps = PorterStemmer()
    post_stemming=[]
    for token in token_array:
        post_stemming.append(ps.stem(token))
    return post_stemming

def remove_common_word(text):
    removed = re.sub(r'(\s*)subject(\s*)|(\s*)enron(\s*)'," ",text)
    return removed

# This Preprocesses the data in 14 steps and makes it suitable and easy for feature extraction
def preprocess_body(df):
    print("Preprocessing Body. May take a while (upto 10 mins)")
    df['Body']=df['Body'].apply(lambda x: str(x).lower())            
    df['Body']=df['Body'].apply(lambda x: contractions.fix(str(x)))  
    df['Body']=df['Body'].apply(lambda x:replace_numbers(str(x)))    
    df['Body']=df['Body'].apply(lambda x: replace_url(str(x)))       
    df['Body']=df['Body'].apply(lambda x: replace_mailid(str(x)))    
    df['Body']=df['Body'].apply(lambda x: replace_currency_signs(str(x)))
    df['Body']=df['Body'].apply(lambda x: remove_punctuation(str(x))) 
    df['Body']=df["Body"].apply(lambda x: accented_to_ascii(str(x))) 
    df['Body']=df['Body'].apply(lambda x: limit_characters(str(x)))  
    df['Body']=df['Body'].apply(lambda x: remove_common_word(str(x)))
    df['Body']=df['Body'].apply(lambda x:' '.join([w for w in str(x).split() if len(w)>1]))   
    df['Body']=df['Body'].apply(lambda x:lemmatize(str(x)))          
    df['Body']=df['Body'].apply(lambda x:stem(x))                    
    df['Body']= df['Body'].apply(lambda x:remove_stopwords(x))       
    df['Body']= df['Body'].apply(lambda x:' '.join(x))
    return df


df=preprocess_body(df)

spam_corpus = []
for mail in df[df['Label'] == 1]['Body']:
    for word in mail.split(' '):
        spam_corpus.append(word)
print(len(spam_corpus),len(set(spam_corpus)) )

ham_corpus = []
for mail in df[df['Label'] == 0]['Body']:
    for word in mail.split(' '):
        ham_corpus.append(word)
print(len(ham_corpus),len(set(ham_corpus)) )

# Plotting  a wordcloud to see the words in spam and ham corpus
wc = WordCloud(width=1000, height=500, min_font_size=10, background_color='white')
spam_wc = wc.generate(" ".join(spam_corpus))
plt.figure(figsize=(20,8))
plt.imshow(spam_wc)
plt.axis("off")
plt.title("Spam Corpus Word Cloud")
plt.show()
wc = WordCloud(width=1000, height=500, min_font_size=10, background_color='white')
ham_wc = wc.generate(" ".join(ham_corpus))
plt.figure(figsize=(20,8))
plt.imshow(ham_wc)
plt.axis("off")
plt.title("Ham Corpus Word Cloud")
plt.show()

X = df.Body
Y = df.Label
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=225, stratify=Y)
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#### SVM MODEL TRAINING ####
def train_test_SVC_model(regularization,kernel_,df):
    X = df.Body
    Y = df.Label
    model = SVC(C=regularization,kernel=kernel_)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 225,stratify=Y)
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    model.fit(X_train ,Y_train)
    X_test=vectorizer.transform(X_test)
    y_pred=model.predict(X_test)
    return accuracy_score(y_pred,Y_test)

kernels=['linear', 'rbf']
regularization_c = np.linspace(0.5,7,15)

score_kernels=[]
for kernel_ in kernels:
    score_regul=[]
    for regularization in regularization_c:
        score_regul.append(train_test_SVC_model(regularization,kernel_,df))
    score_kernels.append(score_regul)


print(regularization_c.shape)
plt.figure(figsize=(12,8))
plt.plot(np.linspace(0.5,7,15),score_kernels[0],marker='o',label='linear')
plt.plot(np.linspace(0.5,7,15),score_kernels[1],marker='o',label='rbf')
plt.legend()
plt.grid()
plt.xlabel('Regularization Constant')
plt.ylabel('Accuracy')
plt.title('Plot of accuracy of different kernels with different regularization constants')
plt.show()


idx_max_score=np.argmax(np.array(score_kernels))

if idx_max_score < len(np.linspace(0.5,7,15)):
    regularization=np.linspace(0.5,7,15)[idx_max_score]
    kernel_='linear'

else:
    regularization=np.linspace(0.5,7,15)[idx_max_score-len(np.linspace(0.5,7,15))]
    kernel_='rbf'

def test_model(test_mail,model):
    X_test=vectorizer.transform(test_mail)
    y_pred=svm_classifier.predict(X_test)
    return y_pred

svm_classifier = SVC(C=regularization,kernel=kernel_)
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
svm_classifier.fit(X_train_tfidf ,Y_train)
X_test_tfidf = vectorizer.transform(X_test)
y_pred = test_model(X_test,svm_classifier)

print("Model: SVM", "\n\tRegularization Constant: ",regularization,"\n\tKernel: ",kernel_)
print('Confusion Matrix:\n',confusion_matrix(y_pred,Y_test))
print("Accuracy\t:\t", accuracy_score(y_pred,Y_test))
print("Precision\t:\t", precision_score(y_pred,Y_test, average = 'weighted'))
print("Recall   \t:\t", recall_score(y_pred,Y_test, average = 'weighted'))


#### LOGISTIC REGRESSION MODEL TRAINING ####
class LogisticRegressionClassifer:
    def __init__(self):
        self.weights = None
        self.bias = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, X_train, Y_train):
        n = X_train.shape[0]
        m = X_train.shape[1]
        self.weights = np.zeros(m)
        self.bias = 0
        for _ in range(500):
            z = np.dot(X_train, self.weights) + self.bias
            predictions = self.sigmoid(z)
            dw = (1 / n) * np.dot(X_train.T, (predictions - Y_train))
            db = (1 / n) * np.sum(predictions - Y_train)
            self.weights -= 0.01 * dw
            self.bias -= 0.01 * db

    def predict(self, X_test):
        z = np.dot(X_test, self.weights) + self.bias
        predictions = self.sigmoid(z)
        return (predictions > 0.5).astype(int)


logistic_regression_classifier = LogisticRegressionClassifer()
logistic_regression_classifier.train(X_train_tfidf.toarray(),Y_train)

lr_train_pred = logistic_regression_classifier.predict(X_train_tfidf.toarray())
lr_test_pred = logistic_regression_classifier.predict(X_test_tfidf.toarray())

print("Logistic Regression Model:")
print("\tTraining Accuracy:", accuracy_score(lr_train_pred, Y_train))
print('Confusion Matrix:\n',confusion_matrix(lr_train_pred,Y_train))
print("Precision\t:\t", precision_score(lr_train_pred,Y_train, average = 'weighted'))
print("Recall   \t:\t", recall_score(lr_train_pred,Y_train, average = 'weighted'))

print("\tTesting Accuracy:", accuracy_score(lr_test_pred, Y_test))
print('Confusion Matrix:\n',confusion_matrix(lr_test_pred,Y_test))
print("Precision\t:\t", precision_score(lr_test_pred,Y_test, average = 'weighted'))
print("Recall   \t:\t", recall_score(lr_test_pred,Y_test, average = 'weighted'))



#### NAIVE BAYES MODEL TRAINING ####
class NaiveBayesClassifier:
    def __init__(self):
        self.pr_class = None
        self.pr_f = None

    def fit(self, X_train, Y_train):
        self.pr_class = np.zeros(2)
        self.pr_f = np.zeros((2, X_train.shape[1]))
        class_counts = np.bincount(Y_train)
        self.pr_class[0] = class_counts[0] / len(Y_train)
        self.pr_class[1] = 1 - self.pr_class[0]
        for c in [0, 1]:
            X_class = X_train[Y_train == c]
            self.pr_f[c] = (np.sum(X_class, axis=0) + 1) / (np.sum(X_class) + X_train.shape[1])

    def predict(self, X_test):
        posteriors = np.zeros((X_test.shape[0], 2))
        for c in [0, 1]:
            prior = np.log(self.pr_class[c])
            likelihood = np.sum(np.log(self.pr_f[c]) * X_test, axis=1)
            posteriors[:, c] = prior + likelihood
        predictions = np.argmax(posteriors, axis=1)
        return predictions


naive_bayes_classifier = NaiveBayesClassifier()
naive_bayes_classifier.fit(X_train_tfidf.toarray(),Y_train)

nb_train_predictions = naive_bayes_classifier.predict(X_train_tfidf.toarray())
nb_test_predictions = naive_bayes_classifier.predict(X_test_tfidf.toarray())

print("Naive Bayes Model:\n")
print("Training Accuracy:", accuracy_score(nb_train_predictions, Y_train))
print('Confusion Matrix:\n',confusion_matrix(nb_train_predictions,Y_train))
print("Precision\t:\t", precision_score(nb_train_predictions,Y_train, average = 'weighted'))
print("Recall   \t:\t", recall_score(nb_train_predictions,Y_train, average = 'weighted'))

print("Testing Accuracy:", accuracy_score(nb_test_predictions, Y_test))
print('Confusion Matrix:\n',confusion_matrix(nb_test_predictions,Y_test))
print("Precision\t:\t", precision_score(nb_test_predictions,Y_test, average = 'weighted'))
print("Recall   \t:\t", recall_score(nb_test_predictions,Y_test, average = 'weighted'))



#### PREDICTING LABELS FOR GIVEN CHALLENGE DATA ####
directory = "test"
test_df = pd.DataFrame(columns=['File', 'Body'])
sorted_filenames = sorted(os.listdir(directory), key=lambda x: int(x.split('.')[0][5:]))
for fname in sorted_filenames:
    if fname.endswith(".txt"):
        file_name = os.path.splitext(fname)[0]
        with open(os.path.join(directory, fname), 'r') as file:
            file_content = file.read()
        test_df = pd.concat([test_df, pd.DataFrame.from_records([{ 'File': fname, 'Body': file_content}])])
test_df = preprocess_body(test_df)
test_df.reset_index(drop=True, inplace=True)


X_test_eval = test_df.Body
X_test_eval_tfidf = vectorizer.transform(test_df.Body)

nb_pred_fin = naive_bayes_classifier.predict(X_test_eval_tfidf.toarray())
lr_pred_fin = logistic_regression_classifier.predict(X_test_eval_tfidf.toarray())
svm_pred_fin = svm_classifier.predict(X_test_eval_tfidf)

stacked_predictions = np.vstack((nb_pred_fin, lr_pred_fin, svm_pred_fin))
maj_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=stacked_predictions)


import sys
sys.stdout = open('output.txt', 'w')
for i in range(len(test_df)):
    print(f"{test_df['File'][i]:<15} {maj_pred[i]} {'(spam)' if maj_pred[i] == 1 else '(not spam)'}")

sys.stdout = sys.__stdout__

test_df['Prediction'] = maj_pred
final_df = test_df.copy()
final_df.drop(columns=['Body'],inplace=True)
final_df.to_csv("final_labelling.csv", index=False)

# Data (file name, label) is saved in csv format as well as a text output