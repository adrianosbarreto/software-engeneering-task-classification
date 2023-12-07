import os

import numpy as np
import pandas as pd
import preprocessing as pp
from sklearn.metrics.pairwise import pairwise_distances
# importing libraries
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.layers import GRU, Bidirectional
from keras.optimizers import SGD
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from keras.preprocessing.text import Tokenizer
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
#
# def lstm(X_train, y_train):
#     # Initialising the model
#     modelLSTM = Sequential()
#
#     # Adding LSTM layers
#     modelLSTM.add(LSTM(50,
#                            return_sequences=True,
#                            input_shape=(X_train.shape[1], 1)))
#     modelLSTM.add(LSTM(50,
#                            return_sequences=False))
#     modelLSTM.add(Dense(25))
#
#     # Adding the output layer
#     modelLSTM.add(Dense(1))
#
#     # Compiling the model
#     modelLSTM.compile(optimizer='adam',
#                           loss='mean_squared_error',
#                           metrics=["accuracy"])
#
#     # Fitting the model
#     modelLSTM.fit(X_train,
#                       y_train,
#                       batch_size=1,
#                       epochs=12)
#     modelLSTM.summary()


def save_tokenizer(dataframe, column, name):
    tokenizer = Tokenizer()

    # Ajustar o Tokenizer com o texto da coluna 'Texto'
    tokenizer.fit_on_texts(dataframe[column])

    # Salvar o Tokenizer em um arquivo binário
    with open(name, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def concat_dataset(folder):
    data_join = pd.DataFrame()
    for dirname, _, filenames in os.walk(folder):
        for filename in filenames:
            # print(os.path.join(dirname, filename))
            dataframe = pd.read_csv(os.path.join(dirname, filename))
            data_join = pd.concat([data_join, dataframe])
    return data_join

def pre_processing(dataset):
    # print(dataset['short_description'])
    # print(dataset['long_description'])
    dataset['short_description'] = dataset['short_description'].apply(pp.preprocessar)
    dataset['long_description'] = dataset['long_description'].apply(exec_preprocessing)

    # print(dataset['short_description'])
    # print(dataset['long_description'])

    return dataset

def time_category(valor):
    if valor == 0:
        return 'muito simples'
    elif 1 >= valor <= 5:
        return "simples"
    elif 5 > valor <= 15:
        return "medio"
    elif 15 > valor <= 30:
        return "complexo"
    else:
        return "muito complexo"

def time_category_to_numeric(valor):
    if valor == 'muito simples':
        return 0
    elif valor == 'simples':
        return 1
    elif valor == 'medio':
        return 2
    elif valor == 'complexo':
        return 3
    else:
        return 4  # 'muito complexo'

def exec_preprocessing(data):
    text = data.lower().strip() if data != '' and data is not None and not type(data) == float else ''
    return pp.preprocessar(text)

def only_text_and_label(dataset):
    data = pre_processing(dataset)

    return data

def format_time_in_class_label(dataset):
    bug = dataset['bug_fix_time']
    dataset['label'] = dataset['bug_fix_time'].apply(time_category)

    plt.hist(bug, 200, rwidth=0.9)
    plt.show()

    print(bug.value_counts()[:30])
    return dataset


def exec(data, target):
    # data = dataset.iloc[:, :-1]
    # target = dataset.iloc[:, -1]

    print(data, target)

    #Normalização de dados
    # scaler = MinMaxScaler()

    # data.iloc[:, :] = scaler.fit_transform(data.iloc[:, :])

    #Separação para treino e teste
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3, random_state=0)

    KNN = KNeighborsClassifier(n_neighbors=3)
    SVM = svm.SVC()
    NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2))

    classifiers = {'knn': KNN}#, 'svm': SVM, 'nn': NN}

    #Treino dos classificadores
    for cls in classifiers:
        print(cls, classifiers[cls])
        classifiers[cls].fit(data_train.values, target_train.values)

    predictions = {}

    #Predição
    for cls in classifiers:
        predictions[cls] = classifiers[cls].predict(data_test.values)
        print(cls, classifiers[cls])


    accuracy = {}

    for cls in predictions:
        accuracy[cls] = accuracy_score(target_test, predictions[cls])
        print(cls, accuracy[cls])


def reduced_dataset(dataset):
    return dataset[['short_description', 'long_description', 'label']]


def jaccard_distance(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return 1 - (intersection / union)

def run_LSTM(texts, labels):
    max_words = 1000  # Número máximo de palavras a serem consideradas no vocabulário
    max_len = 10000  # Comprimento máximo da sequência

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=max_len)
    y = np.array(labels)

    # Divisão dos dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Construção do modelo LSTM
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=1000, input_length=max_len))
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))

    # Adding the output layer
    model.add(Dense(1))

    # Compilação do modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Treinamento do modelo
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Avaliação do modelo
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    dataset = concat_dataset(folder='teste')
    print(dataset.head())
    dataset = only_text_and_label(dataset)
    dataset = format_time_in_class_label(dataset)

    print('Long description', dataset['long_description'].values)
    print('Lable', dataset['label'].values)

    # tfidf_vectorizer = TfidfVectorizer()
    # X = tfidf_vectorizer.fit_transform(dataset['long_description'].values)
    # print('X', X.toarray())
    #
    # dataset_exec = pd.DataFrame(X.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    #
    # print(dataset_exec['label'])

    # exec(dataset_exec, dataset['label'])


    run_LSTM(dataset['long_description'], dataset['label'].apply(time_category_to_numeric))

