from sklearn.cluster.bicluster import SpectralBiclustering
import numpy as np
from sklearn.datasets import samples_generator as sg
from sklearn.datasets import make_checkerboard
from sklearn.metrics import consensus_score
from matplotlib import pyplot as plt
import os,pandas,re,time
from numpy import float32,save,load
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from utils.file import *
from sklearn.decomposition import LatentDirichletAllocation

def obtain_time_slices(start,end,portions):
    jump = int((end - start)/portions)
    first = list(range(start+1,end,jump))
    last = list(range(start+jump,end+jump,jump))
    return list(zip(first,last))

def split_data_in_time_slices(dataframe,start,end,portions):
    slices = obtain_time_slices(start,end,portions)
    return {(s,e) : dataframe.query(str(s)+' <= pubdate <=' + str(e)) for (s,e) in slices}

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

def get_directory_dataset_periods(dataset_name,preprocessing,start,end,n):
    directory = get_directory_dataset(dataset_name)
    newdirectory = "{p}_{s}-{e}_{n}".format(p=preprocessing,s=start,e=end,n=n)
    return os.path.join(directory,newdirectory)

def tfidf_periods_exists(dataset_name,preprocessing,start,end,n):
    directory = get_directory_dataset_periods(dataset_name,preprocessing,start,end,n)
    return os.path.exists(directory)

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

def store_data_periods(dataset_name,preprocessing,start,end,n,i,j,tfidf,documents,terms):
    directory = get_directory_dataset_periods(dataset_name,preprocessing,start,end,n)
    filenametfidf = "{i}-{j}_tfidf".format(i=i,j=j) + ".h5"
    filenamedocuments = "{i}-{j}_documents".format(i=i,j=j)
    filenameterms = "{i}-{j}_terms".format(i=i,j=j)
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

def load_data_periods(dataset_name,preprocessing,start,end,n,i,j):
    directory = get_directory_dataset_periods(dataset_name,preprocessing,start,end,n)
    filenametfidf = "{i}-{j}_tfidf".format(i=i,j=j) + ".h5"
    filenamedocuments = "{i}-{j}_documents".format(i=i,j=j)
    filenameterms = "{i}-{j}_terms".format(i=i,j=j)
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

def save_clasification_periods(dataset_name,preprocessing,mindf,k1,k2,ngram_min,ngram_max,model,start,end,n,i,j):
    directory = get_directory_dataset_periods(dataset_name,preprocessing,start,end,n)
    filenamedoc = "{i}-{j}_{df}_{k1}_{k2}_{mi}_{ma}_spectral_documents".format(i=i,j=j,
        df=mindf,k1=k1,k2=k2,mi=ngram_min,ma=ngram_max)
    fullpathdoc = os.path.join(directory, filenamedoc)
    filenameterm = "{i}-{j}_{df}_{k1}_{k2}_{mi}_{ma}_spectral_terms".format(i=i,j=j,
        df=mindf, k1=k1, k2=k2, mi=ngram_min, ma=ngram_max)
    fullpathterm = os.path.join(directory, filenameterm)
    save(fullpathdoc, model.row_labels_.astype(float32))
    save(fullpathterm, model.column_labels_.astype(float32))

def spectral_exists(directory,dataset_name,preprocessing,mindf,k1,k2,ngram_min,ngram_max):
    filenamedoc = "{ds}_{pr}_{df}_{k1}_{k2}_{mi}_{ma}_spectral_documents".format(
        ds=dataset_name, pr=preprocessing, df=mindf, k1=k1, k2=k2, mi=ngram_min, ma=ngram_max)
    fullpathdoc = os.path.join(directory, filenamedoc)+".npy"
    filenameterm = "{ds}_{pr}_{df}_{k1}_{k2}_{mi}_{ma}_spectral_terms".format(
        ds=dataset_name, pr=preprocessing, df=mindf, k1=k1, k2=k2, mi=ngram_min, ma=ngram_max)+".npy"
    fullpathterm = os.path.join(directory, filenameterm)
    return os.path.exists(fullpathdoc) and os.path.exists(fullpathterm)

def spectral_periods_exists(dataset_name,preprocessing,mindf,k1,k2,ngram_min,ngram_max,start,end,n,i,j):
    directory = get_directory_dataset_periods(dataset_name, preprocessing, start, end, n)
    filenamedoc = "{i}-{j}_{df}_{k1}_{k2}_{mi}_{ma}_spectral_documents".format(i=i,j=j,
        ds=dataset_name, pr=preprocessing, df=mindf, k1=k1, k2=k2, mi=ngram_min, ma=ngram_max)
    fullpathdoc = os.path.join(directory, filenamedoc)+".npy"
    filenameterm = "{i}-{j}_{df}_{k1}_{k2}_{mi}_{ma}_spectral_terms".format(i=i,j=j,
        ds=dataset_name, pr=preprocessing, df=mindf, k1=k1, k2=k2, mi=ngram_min, ma=ngram_max)+".npy"
    fullpathterm = os.path.join(directory, filenameterm)
    return os.path.exists(fullpathdoc) and os.path.exists(fullpathterm)

def load_classification(dataset_name,preprocessing,mindf,k1,k2,ngram_min,ngram_max):
    directory = get_directory_dataset(dataset_name)
    filenamedoc = "{ds}_{pr}_{df}_{k1}_{k2}_{mi}_{ma}_spectral_documents".format(
        ds=dataset_name, pr=preprocessing, df=mindf, k1=k1, k2=k2, mi=ngram_min, ma=ngram_max)+".npy"
    fullpathdoc = os.path.join(directory, filenamedoc)
    filenameterm = "{ds}_{pr}_{df}_{k1}_{k2}_{mi}_{ma}_spectral_terms".format(
        ds=dataset_name, pr=preprocessing, df=mindf, k1=k1, k2=k2, mi=ngram_min, ma=ngram_max)+".npy"
    fullpathterm = os.path.join(directory, filenameterm)
    return load(fullpathdoc),load(fullpathterm)

def load_classification_periods(dataset_name,preprocessing,mindf,k1,k2,ngram_min,ngram_max,start,end,n,i,j):
    directory = get_directory_dataset_periods(dataset_name, preprocessing, start, end, n)
    filenamedoc = "{i}-{j}_{df}_{k1}_{k2}_{mi}_{ma}_spectral_documents".format(i=i, j=j,
                                                                               ds=dataset_name, pr=preprocessing,
                                                                               df=mindf, k1=k1, k2=k2, mi=ngram_min,
                                                                               ma=ngram_max)
    fullpathdoc = os.path.join(directory, filenamedoc) + ".npy"
    filenameterm = "{i}-{j}_{df}_{k1}_{k2}_{mi}_{ma}_spectral_terms".format(i=i, j=j,
                                                                            ds=dataset_name, pr=preprocessing, df=mindf,
                                                                            k1=k1, k2=k2, mi=ngram_min,
                                                                            ma=ngram_max) + ".npy"
    fullpathterm = os.path.join(directory, filenameterm)
    return load(fullpathdoc), load(fullpathterm)

def spectral(dataset_name,full,preprocessing,mindf,k1,k2,ngram_min,ngram_max,start,end,n):
    if not spectral_directory_exists(dataset_name):
        create_spectral_directory(dataset_name)
    h, c = obtain_file_name_from_dataset(dataset_name, preprocessing)
    corpus = obtain_full_corpus(h, c)
    if full:
        texts = corpus.text.values
        docnames = corpus.text.index.values
        if not tfidf_exists(dataset_name,preprocessing):
            X,v = create_tfidf(texts,mindf,ngram_min,ngram_max)
            words = v.get_feature_names()
            store_data(dataset_name,preprocessing,X,docnames,words)
        tfidf, documents, terms = load_data(dataset_name, preprocessing)
        if not spectral_exists(get_directory_dataset(dataset_name),dataset_name,preprocessing,mindf,k1,k2,ngram_min,ngram_max):
            start = time.time()
            model = SpectralBiclustering(n_clusters=(k1, k2), random_state=0)
            model.fit(tfidf)
            end = time.time()
            print("Biclustering process takes", int(round(end - start)), "seconds")
            save_clasification(get_directory_dataset(dataset_name),dataset_name,preprocessing,mindf,k1,k2,ngram_min,ngram_max,model)
    else:
        time_corpus = split_data_in_time_slices(corpus, start, end, n)
        if not tfidf_periods_exists(dataset_name,preprocessing,start,end,n):
            os.makedirs(get_directory_dataset_periods(dataset_name,preprocessing,start,end,n))
            for (s, e), corp in time_corpus.items():
                texts = corp.text.values
                docnames = corp.text.index.values
                X, v = create_tfidf(texts, mindf, ngram_min, ngram_max)
                words = v.get_feature_names()
                store_data_periods(dataset_name,preprocessing,start,end,n,s,e,X,docnames,words)
        for s,e in time_corpus:
            tfidf, documents, terms = load_data_periods(dataset_name, preprocessing,start,end,n,s,e)
            if not spectral_periods_exists(dataset_name,preprocessing,mindf,k1,k2,ngram_min,ngram_max,start,end,n,s,e):
                st = time.time()
                model = SpectralBiclustering(n_clusters=(k1, k2), random_state=0)
                model.fit(tfidf)
                ed = time.time()
                print("Biclustering process takes", int(round(ed - st)), "seconds")
                save_clasification_periods(dataset_name,preprocessing,mindf,k1,k2,ngram_min,ngram_max,model,start,end,n,s,e)


def get_directory_lda_dataset(dataset_name):
    return os.path.join(*[os.path.dirname(os.path.realpath(__file__)), "tmp", dataset_name, "lda"])

def lda_directory_exists(dataset_name):
    return os.path.exists(get_directory_lda_dataset(dataset_name))

def create_lda_directory(dataset_name):
    os.makedirs(get_directory_lda_dataset(dataset_name))

#TODO in lda the clasification is documents and terms?
def lda_exists(dataset_name,preprocessing,mindf,k,ngram_min,ngram_max):
    directory = get_directory_lda_dataset(dataset_name)
    filenamedoc = "{ds}_{pr}_{df}_{k}_{mi}_{ma}_lda_documents".format(
        ds=dataset_name, pr=preprocessing, df=mindf, k=k, mi=ngram_min, ma=ngram_max)
    fullpathdoc = os.path.join(directory, filenamedoc)+".npy"
    filenameterm = "{ds}_{pr}_{df}_{k}_{mi}_{ma}_lda_terms".format(
        ds=dataset_name, pr=preprocessing, df=mindf, k=k, mi=ngram_min, ma=ngram_max)+".npy"
    fullpathterm = os.path.join(directory, filenameterm)
    return os.path.exists(fullpathdoc) and os.path.exists(fullpathterm)

def corpus_vektorizer_exists(dataset_name,preprocessing):
    directory = get_directory_lda_dataset(dataset_name)
    filenametfidf = "{ds}_{pr}_count_vektorizer".format(ds=dataset_name, pr=preprocessing) + ".h5"
    filenamedocuments = "{ds}_{pr}_documents_vektorizer".format(ds=dataset_name, pr=preprocessing) + ".npy"
    filenameterms = "{ds}_{pr}_terms_vektorizer".format(ds=dataset_name, pr=preprocessing) + ".npy"
    fullpathtfidf = os.path.join(directory, filenametfidf)
    fullpathdocuments = os.path.join(directory, filenamedocuments)
    fullpathterms = os.path.join(directory, filenameterms)
    return os.path.exists(fullpathtfidf) and os.path.exists(fullpathdocuments) and os.path.exists(fullpathterms)

def create_corpus_vektorizer(corpus,min_df_value,min_n,max_n):
    vectorizer = CountVectorizer(min_df=min_df_value, dtype=float32, ngram_range=(min_n, max_n))
    X = vectorizer.fit_transform(corpus)
    return X,vectorizer

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


def store_vekt_data(dataset_name,preprocessing,tfidf,documents,terms):
    directory = get_directory_lda_dataset(dataset_name)
    filenametfidf = "{ds}_{pr}_count_vektorizer".format(ds=dataset_name, pr=preprocessing) + ".h5"
    filenamedocuments = "{ds}_{pr}_documents_vektorizer".format(ds=dataset_name, pr=preprocessing)
    filenameterms = "{ds}_{pr}_terms_vektorizer".format(ds=dataset_name, pr=preprocessing)
    fullpathtfidf = os.path.join(directory, filenametfidf)
    fullpathdocuments = os.path.join(directory, filenamedocuments)
    fullpathterms = os.path.join(directory, filenameterms)
    store_sparse_mat(tfidf, "count", fullpathtfidf)
    save(fullpathdocuments,documents)
    save(fullpathterms, terms)

def load_vekt_data(dataset_name,preprocessing):
    directory = get_directory_lda_dataset(dataset_name)
    filenametfidf = "{ds}_{pr}_count_vektorizer".format(ds=dataset_name, pr=preprocessing) + ".h5"
    filenamedocuments = "{ds}_{pr}_documents_vektorizer".format(ds=dataset_name, pr=preprocessing)
    filenameterms = "{ds}_{pr}_terms_vektorizer".format(ds=dataset_name, pr=preprocessing)
    fullpathtfidf = os.path.join(directory, filenametfidf)
    fullpathdocuments = os.path.join(directory, filenamedocuments)
    fullpathterms = os.path.join(directory, filenameterms)
    tfidf = load_sparse_mat("count", fullpathtfidf).astype(float32)
    documents = load(fullpathdocuments + ".npy")
    terms = load(fullpathterms + ".npy")
    return tfidf, documents, terms

def from_lda_to_clusters():
    return

def lda(dataset_name,full,preprocessing,mindf,k,ngram_min,ngram_max,start,end,n):
    if not lda_directory_exists(dataset_name):
        create_lda_directory(dataset_name)
    h, c = obtain_file_name_from_dataset(dataset_name, preprocessing)
    corpus = obtain_full_corpus(h, c)
    if full:
        texts = corpus.text.values
        docnames = corpus.text.index.values
        if not corpus_vektorizer_exists(dataset_name,preprocessing):
            X,v = create_corpus_vektorizer(texts,mindf,ngram_min,ngram_max)
            words = v.get_feature_names()
            store_vekt_data(dataset_name, preprocessing, X, docnames, words)
        countvekt, documents, terms = load_vekt_data(dataset_name, preprocessing)
        lda = LatentDirichletAllocation(n_topics=k, max_iter=5, learning_method='online',
                                        learning_offset=50., random_state=0).fit(countvekt)
        return lda

    #    if not lda_exists(dataset_name,preprocessing,mindf,k,ngram_min,ngram_max):
    #         start = time.time()
    #         lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary)
    #         lda_model.fit(tfidf)
    #         end = time.time()
    #         print("LDA process takes", int(round(end - start)), "seconds")
    #         save_clasification(get_directory_lda_dataset(dataset_name),dataset_name,preprocessing,mindf,k1,k2,ngram_min,ngram_max,model)
    # else:
    #     time_corpus = split_data_in_time_slices(corpus, start, end, n)
    #     if not tfidf_periods_exists(dataset_name,preprocessing,start,end,n):
    #         os.makedirs(get_directory_dataset_periods(dataset_name,preprocessing,start,end,n))
    #         for (s, e), corp in time_corpus.items():
    #             texts = corp.text.values
    #             docnames = corp.text.index.values
    #             X, v = create_tfidf(texts, mindf, ngram_min, ngram_max)
    #             words = v.get_feature_names()
    #             store_data_periods(dataset_name,preprocessing,start,end,n,s,e,X,docnames,words)
    #     for s,e in time_corpus:
    #         tfidf, documents, terms = load_data_periods(dataset_name, preprocessing,start,end,n,s,e)
    #         if not spectral_periods_exists(dataset_name,preprocessing,mindf,k1,k2,ngram_min,ngram_max,start,end,n,s,e):
    #             st = time.time()
    #             model = SpectralBiclustering(n_clusters=(k1, k2), random_state=0)
    #             model.fit(tfidf)
    #             ed = time.time()
    #             print("Biclustering process takes", int(round(ed - st)), "seconds")
    #             save_clasification_periods(dataset_name,preprocessing,mindf,k1,k2,ngram_min,ngram_max,model,start,end,n,s,e)
