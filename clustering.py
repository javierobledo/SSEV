from sklearn.cluster.bicluster import SpectralBiclustering
import numpy as np
from sklearn.datasets import samples_generator as sg
from sklearn.datasets import make_checkerboard
from sklearn.metrics import consensus_score
from matplotlib import pyplot as plt
import os,pandas,re,time
from numpy import float32,save,load
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.file import *

def convert_pubdate(date):
    p = re.compile("\d{4}")
    date_as_string = str(date)
    match = p.search(date_as_string)
    if match:
        year_as_string = match.group(0)
    return int(year_as_string)

def obtain_full_corpus(headers_file,corpus_file):
    headers = pandas.read_csv(headers_file)
    corpus = pandas.read_csv(corpus_file)
    headers.pubdate = headers.pubdate.apply(convert_pubdate)
    headers = headers.set_index('id')
    corpus = corpus.set_index('id')
    return headers.join(corpus)

def obtain_file_name_from_dataset(dataset_name,preprocessing):
    headerfilename = os.path.join(*[os.path.dirname(os.path.realpath(__file__)),"tmp",dataset_name,dataset_name+"_header.csv"])
    if preprocessing == 'original' or preprocessing == 'random':
        corpusfilename = os.path.join(*[os.path.dirname(os.path.realpath(__file__)),"tmp",dataset_name,dataset_name+"_corpus.csv"])
    elif preprocessing == 'postagged':
        corpusfilename = os.path.join(
            *[os.path.dirname(os.path.realpath(__file__)), "tmp", dataset_name, dataset_name + "_corpus_tagged.csv"])
    return headerfilename,corpusfilename

def get_directory_dataset(dataset_name):
    return os.path.join(*[os.path.dirname(os.path.realpath(__file__)), "tmp", dataset_name, "spectral"])

def spectral_directory_exists(dataset_name):
    return os.path.exists(get_directory_dataset(dataset_name))

def create_spectral_directory(dataset_name):
    os.makedirs(get_directory_dataset(dataset_name))

def tfidf_exists(dataset_name,preprocessing):
    directory = get_directory_dataset(dataset_name)
    filenametfidf = "{ds}_{pr}_tfidf".format(ds=dataset_name, pr=preprocessing) + ".h5"
    filenamedocuments = "{ds}_{pr}_documents".format(ds=dataset_name, pr=preprocessing)+".npy"
    filenameterms = "{ds}_{pr}_terms".format(ds=dataset_name, pr=preprocessing)+".npy"
    fullpathtfidf = os.path.join(directory, filenametfidf)
    fullpathdocuments = os.path.join(directory, filenamedocuments)
    fullpathterms = os.path.join(directory, filenameterms)
    return os.path.exists(fullpathtfidf) and os.path.exists(fullpathdocuments) and os.path.exists(fullpathterms)

def create_tfidf(corpus,min_df_value,min_n,max_n):
    vectorizer = TfidfVectorizer(min_df=min_df_value, dtype=float32, ngram_range=(min_n, max_n))
    X = vectorizer.fit_transform(corpus)
    return X,vectorizer

def store_data(dataset_name,preprocessing,tfidf,documents,terms):
    directory = get_directory_dataset(dataset_name)
    filenametfidf = "{ds}_{pr}_tfidf".format(ds=dataset_name, pr=preprocessing) + ".h5"
    filenamedocuments = "{ds}_{pr}_documents".format(ds=dataset_name, pr=preprocessing)
    filenameterms = "{ds}_{pr}_terms".format(ds=dataset_name, pr=preprocessing)
    fullpathtfidf = os.path.join(directory, filenametfidf)
    fullpathdocuments = os.path.join(directory, filenamedocuments)
    fullpathterms = os.path.join(directory, filenameterms)
    store_sparse_mat(tfidf, "tfidf", fullpathtfidf)
    save(fullpathdocuments,documents)
    save(fullpathterms, terms)
    
def load_data(dataset_name,preprocessing):
    directory = get_directory_dataset(dataset_name)
    filenametfidf = "{ds}_{pr}_tfidf".format(ds=dataset_name, pr=preprocessing) + ".h5"
    filenamedocuments = "{ds}_{pr}_documents".format(ds=dataset_name, pr=preprocessing)
    filenameterms = "{ds}_{pr}_terms".format(ds=dataset_name, pr=preprocessing)
    fullpathtfidf = os.path.join(directory, filenametfidf)
    fullpathdocuments = os.path.join(directory, filenamedocuments)
    fullpathterms = os.path.join(directory, filenameterms)
    tfidf = load_sparse_mat("tfidf",fullpathtfidf).astype(float32)
    documents = load(fullpathdocuments+".npy")
    terms = load(fullpathterms + ".npy")
    return tfidf,documents,terms

def save_clasification(directory,dataset_name,preprocessing,mindf,k1,k2,ngram_min,ngram_max,model):
    filenamedoc = "{ds}_{pr}_{df}_{k1}_{k2}_{mi}_{ma}_spectral_documents".format(
        ds=dataset_name, pr=preprocessing,df=mindf,k1=k1,k2=k2,mi=ngram_min,ma=ngram_max)
    fullpathdoc = os.path.join(directory, filenamedoc)
    filenameterm = "{ds}_{pr}_{df}_{k1}_{k2}_{mi}_{ma}_spectral_terms".format(
        ds=dataset_name, pr=preprocessing, df=mindf, k1=k1, k2=k2, mi=ngram_min, ma=ngram_max)
    fullpathterm = os.path.join(directory, filenameterm)
    save(fullpathdoc, model.row_labels_.astype(float32))
    save(fullpathterm, model.column_labels_.astype(float32))

#def spectral_exists(dataset_naame)
    

def spectral(dataset_name,full,preprocessing,mindf,k1,k2,ngram_min,ngram_max):
    if not spectral_directory_exists(dataset_name):
        create_spectral_directory(dataset_name)
    if full:
        h, c = obtain_file_name_from_dataset(dataset_name, preprocessing)
        corpus = obtain_full_corpus(h, c)
        texts = corpus.text.values
        docnames = corpus.text.index.values
        print("full process")
        if not tfidf_exists(dataset_name,preprocessing):
            print("tfidf full not exists")
            X,v = create_tfidf(texts,mindf,ngram_min,ngram_max)
            words = v.get_feature_names()
            store_data(dataset_name,preprocessing,X,docnames,words)
        tfidf, documents, terms = load_data(dataset_name, preprocessing)
        #TODO Verify if spectral classification exists
        start = time.time()
        model = SpectralBiclustering(n_clusters=(k1, k2), random_state=0)
        model.fit(tfidf)
        end = time.time()
        print("Biclustering process takes", int(round(end - start)), "seconds")
        save_clasification(get_directory_dataset(dataset_name),dataset_name,preprocessing,mindf,k1,k2,ngram_min,ngram_max,model)
        print(model.get_indices(0))
    else:
        print("not full")
        #TODO Separate corpus in time periods
        #TODO For each period, create tfidf,terms and documents files (If not exists)
        #TODO Make classification
    #TODO Store clasification in files
    # n_clusters = (k1,k2)
    # data, rows, columns = make_checkerboard(
    #      shape=(300, 300), n_clusters=n_clusters, noise=10,
    #      shuffle=False, random_state=0)
    # print(data)
    # print(rows)
    # print(columns)
    # plt.matshow(data, cmap=plt.cm.Blues)
    # plt.title("Original dataset")
    #
    # data, row_idx, col_idx = sg._shuffle(data, random_state=0)
    # plt.matshow(data, cmap=plt.cm.Blues)
    # plt.title("Shuffled dataset")
    # model = SpectralBiclustering(n_clusters=n_clusters, method='log',
    #                              random_state=0)
    # model.fit(data)
    # score = consensus_score(model.biclusters_,
    #                         (rows[:, row_idx], columns[:, col_idx]))
    #
    # print("consensus score: {:.1f}".format(score))
    #
    # fit_data = data[np.argsort(model.row_labels_)]
    # fit_data = fit_data[:, np.argsort(model.column_labels_)]
    #
    # plt.matshow(fit_data, cmap=plt.cm.Blues)
    # plt.title("After biclustering; rearranged to show biclusters")
    #
    # plt.matshow(np.outer(np.sort(model.row_labels_) + 1,
    #                      np.sort(model.column_labels_) + 1),
    #             cmap=plt.cm.Blues)
    # plt.title("Checkerboard structure of rearranged data")
    #
    # plt.show()
