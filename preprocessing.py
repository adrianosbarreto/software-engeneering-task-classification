import collections
import copy
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import snowballstemmer as sbs
from unicodedata import normalize


def chr_remove(old, to_remove):
    new_string = old
    for x in to_remove:
        new_string = new_string.replace(x, '')
    return new_string

def remover_caracter_especial(texto):
    texto = chr_remove(texto, '1234567890')
    texto = chr_remove(texto, '/.,?;:!@#$%¨<>|&*()_+-\'\\')
    return re.sub('u[^a-zA-Z \\\]', '', texto)

def remover_acentos(txt):
    return normalize('NFKD', txt).encode('ASCII','ignore').decode('ASCII')

def lower(texto):
    vetor = []
    for i in texto:
        vetor.append(i.lower())
    return vetor

def bigrama(texto):
    bigram = list(nltk.bigrams(texto))
    #print(bigram)
    return bigram
    
def trigrama(texto):
    trigram = list(nltk.trigrams(texto))
    #print(trigram)
    return trigram
    
def unigrama(texto):
    # v = []
    # for i in texto:
    #     #print(i)
    #     v.append(i)
    # return v

    return ' '.join(texto)
    
def tokenize(texto):
    return texto.split()

def stemmer(texto): 
    stemmer = sbs.stemmer("english")
    x = stemmer.stemWords(texto)
    #print(x)
    return x
    
def remove_stop_words(texto):
    #texto = texto.split()
    list_stop_words = stopwords.words('english')
    return [palavra for palavra in texto if palavra not in list_stop_words]
    
def frequencia(token):
    c = collections.Counter(token)
    return c.most_common()

    
def dice(vetor1, vetor2):
    #vet1 = copy.deepcopy(vetor1)
    #vet2 = copy.deepcopy(vetor2)
    x = len( intersecao(vetor1, vetor2) )
    #print("x",x)
    y = (len(vetor1)+len(vetor2))
    #print("y", y)
    return (2*x)/y
    
def cosseno(vetor1, vetor2):
    return len(intersecao(vetor1, vetor2))/np.sqrt(len(vetor1)*len(vetor2))

def mergeVetor(vetor1, vetor2):
    merge = [[], []]
    #maior, menor = [], []
    if len(vetor1) > len(vetor2):
        maior = frequencia(vetor1)
        menor = frequencia(vetor2)

    else:
        maior = frequencia(vetor2)
        menor = frequencia(vetor1)
    tam = len(maior)
    #tam = max(len(maior), len(menor))
    #print(menor)
    #print(maior)
    dicionario = {}
    dicionario.update(list(menor))
    #print(dicionario)
    #menor =
    
    for i in range(tam):
        #print(i, len(maior), len(menor), tam)
        #print(maior[i])
        key, _= maior[i]
        x = dicionario.get(key)
        #print(x)
        _, y = maior[i]
        if x is not None:
            merge[0].append(x)
            merge[1].append(y)
        else:
            merge[0].append(0)
            merge[1].append(y)       
    #print(merge)
    return merge     
    

def cosseno_df(vetor1, vetor2):
    cosseno = 0
    merge = mergeVetor(vetor1, vetor2)
    v1, v2 = np.array(merge[0]), np.array(merge[1])
    #print("v1 * v2", v1 * v2)
    soma_produto = np.sum(v1 * v2)
    soma2_v1 = np.sum(v1 * v1)
    soma2_v2 = np.sum(v2 * v2)
    #print(soma_produto, soma2_v1, soma2_v2)
    if np.sqrt(soma2_v1)*np.sqrt(soma2_v2) == 0:
        cosseno = 0
    else:
        cosseno = soma_produto/(np.sqrt(soma2_v1)*np.sqrt(soma2_v2))
    return cosseno

def intersecao(vet1, vet2):
    vetor1 = copy.deepcopy(vet1)
    vetor2 = copy.deepcopy(vet2)
    intersecao = []
    i = 0
    while i < len(vetor1):
        j = 0  
        while j < len(vetor2):
            if vetor1[i] == vetor2[j]:
                intersecao.append(vetor1[i])
                del vetor1[i]
                del vetor2[j]
                i-=1
                break
            j+=1
        i+=1
    #print(intersecao)
    return intersecao
                     
def preprocessar(vetor,u=1):
    x = remover_acentos(vetor)
    x = remover_caracter_especial(x)
    x = tokenize(x)
    x = remove_stop_words(x)
    x = lower(x)
    x = stemmer(x)
    if u == 1: 
        x = unigrama(x)
    elif u == 2:
        x = bigrama(x)
    elif u == 3:
        x = trigrama(x)
    return x

    
    
    
    
    
    
    
    
    
    
    
    