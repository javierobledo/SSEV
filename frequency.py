import re,pandas,os
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import float32,multiply,inner

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


def obtain_time_slices(start,end,portions):
    jump = int((end - start)/portions)
    first = list(range(start+1,end,jump))
    last = list(range(start+jump,end+jump,jump))
    return list(zip(first,last))

def split_data_in_time_slices(dataframe,start,end,portions):
    slices = obtain_time_slices(start,end,portions)
    return {(s,e) : dataframe.query(str(s)+' <= pubdate <=' + str(e)) for (s,e) in slices}

def tf(corpus,min_n,max_n):
    vectorizer = TfidfVectorizer(dtype=float32,ngram_range=(min_n,max_n),use_idf=False,smooth_idf=False,sublinear_tf=False,norm=None)
    X = vectorizer.fit_transform(corpus)
    y = vectorizer.get_feature_names()
    N = X.sum()
    X = multiply(X.sum(0).A1,1/N)
    term_freq = list(zip(y,X))
    #term_freq_sorted = sorted(term_freq, key=lambda tup: tup[1])
    #term_freq_sorted.reverse()
    return dict(term_freq)

def df(corpus,min_n,max_n):
    vectorizer = TfidfVectorizer(dtype=float32,ngram_range=(min_n,max_n),binary=True,use_idf=False,smooth_idf=False,norm=None)
    X = vectorizer.fit_transform(corpus)
    N,m = X.shape
    y = vectorizer.get_feature_names()
    X = X.sum(0).A1
    X = multiply(X,1/N)
    doc_freq = list(zip(y,X))
    #doc_freq_sorted = sorted(doc_freq, key=lambda tup: tup[1])
    #doc_freq_sorted.reverse()
    return dict(doc_freq)

def obtain_file_name_from_dataset(dataset_name,preprocessing):
    headerfilename = os.path.join(*[os.path.dirname(os.path.realpath(__file__)),"tmp",dataset_name,dataset_name+"_header.csv"])
    if preprocessing == 'original':
        corpusfilename = os.path.join(*[os.path.dirname(os.path.realpath(__file__)),"tmp",dataset_name,dataset_name+"_corpus.csv"])
    elif preprocessing == 'postagged':
        corpusfilename = os.path.join(
            *[os.path.dirname(os.path.realpath(__file__)), "tmp", dataset_name, dataset_name + "_corpus_tagged.csv"])
    return headerfilename,corpusfilename

def file_exist(filename):
    return os.path.exists(filename)

def obtain_frequency_index_in_register(df,row):
    return df.loc[(df.analysis_type == row["analysis_type"]) & (df.dataset_name == row["dataset_name"]) & (df.ngram_min == row["ngram_min"])
           & (df.ngram_max == row["ngram_max"]) & (df.preprocessing == row["preprocessing"]) & (df.start == row["start"]) & (df.end == row["end"])
           & (df.n == row["n"])]

#TODO
def frequency_data_in_register(df,row):
    x = obtain_frequency_index_in_register(df,row)
    return not x.empty



def register_frequency(dataset_name,start,end,n,ngram_min,ngram_max,analysis_type,preprocessing):
    filename = os.path.join(*[os.path.dirname(os.path.realpath(__file__)),"tmp",dataset_name,dataset_name+"_frequency.csv"])
    directory = os.path.join(*[os.path.dirname(os.path.realpath(__file__)),"tmp",dataset_name,"fq"])
    if not os.path.exists(directory):
        os.makedirs(directory)
    newfilename = "{ds}_{s}_{e}_{n}_{ngmin}_{ngmax}_{at}_{pr}".format(ds=dataset_name,s=start,e=end,n=n,ngmin=ngram_min,ngmax=ngram_max,at=analysis_type,pr=preprocessing)+".csv"
    #newfile = os.path.join(directory,newfilename)
    row = {"dataset_name": dataset_name, "start": start, "end": end, "n": n, "ngram_min": ngram_min,
           "ngram_max": ngram_max, "analysis_type": analysis_type, "preprocessing": preprocessing, "file": newfilename}
    #print(row)
    if not file_exist(filename):

        fq = pandas.DataFrame(row,index=[0])
        fq.to_csv(filename,index=False)
        return row
    else:
        nfq = pandas.DataFrame(row,index=[0])
        fq = pandas.read_csv(filename,index_col=0)
        if frequency_data_in_register(fq,row):
            return False
        else:
            x = pandas.concat([nfq, fq], ignore_index=True)
            x.to_csv(filename,index=False)
            return row

def create_frequency_file(directory,corpus,row):
    time_corpus = split_data_in_time_slices(corpus, row["start"], row["end"], row["n"])
    d = {}
    if row["analysis_type"] == 'tf':
        for (s,e),corp in time_corpus.items():
            f = tf(corp.text, row["ngram_min"], row["ngram_max"])
            data = pandas.Series(f)
            d[str(s)+"-"+str(e)]=data
        fdata = pandas.DataFrame(d)
        fdata.to_csv(os.path.join(directory, row["file"]), encoding="utf-8")
    elif row["analysis_type"] == 'df':
        for (s,e),corp in time_corpus.items():
            f = df(corp.text, row["ngram_min"], row["ngram_max"])
            data = pandas.Series(f)
            d[str(s)+"-"+str(e)]=data
        fdata = pandas.DataFrame(d)
        fdata.to_csv(os.path.join(directory,row["file"]),encoding="utf-8")

def load_frequency(directory,dataset_name,start,end,n,ngram_min,ngram_max,analysis_type,preprocessing):
    filename = "{ds}_{s}_{e}_{n}_{ngmin}_{ngmax}_{at}_{pr}".format(ds=dataset_name,s=start,e=end,n=n,ngmin=ngram_min,ngmax=ngram_max,at=analysis_type,pr=preprocessing)+".csv"
    df = pandas.DataFrame.from_csv(os.path.join(directory,filename))
    df.astype(float32)
    return df

def load_d(dataframe):
    return (dataframe>0)*1

def use_index(dataset_name,start,end,ngram_min,ngram_max,analysis_type,n,preprocessing):
    directory = os.path.join(*[os.path.dirname(os.path.realpath(__file__)), "tmp", dataset_name, "fq"])
    filename = "{ds}_{s}_{e}_{n}_{ngmin}_{ngmax}_{at}_{pr}".format(ds=dataset_name, s=start, e=end, n=n,
                                                                   ngmin=ngram_min, ngmax=ngram_max, at=analysis_type,
                                                                   pr=preprocessing) + ".csv"
    filefullpath = os.path.join(directory,filename)
    if not os.path.exists(filefullpath):
        f = load_frequency(directory,dataset_name,start,end,n,ngram_min,ngram_max,"tf",preprocessing)
        u = f.sum(axis=1)
        u.to_csv(filefullpath,encoding="utf-8")

def adoption_index(dataset_name,start,end,ngram_min,ngram_max,analysis_type,n,preprocessing):
    directory = os.path.join(*[os.path.dirname(os.path.realpath(__file__)), "tmp", dataset_name, "fq"])
    filename = "{ds}_{s}_{e}_{n}_{ngmin}_{ngmax}_{at}_{pr}".format(ds=dataset_name, s=start, e=end, n=n,
                                                                   ngmin=ngram_min, ngmax=ngram_max, at=analysis_type,
                                                                   pr=preprocessing) + ".csv"
    filefullpath = os.path.join(directory,filename)
    if not os.path.exists(filefullpath):
        f = load_frequency(directory,dataset_name,start,end,n,ngram_min,ngram_max,"tf",preprocessing).fillna(0)
        d = (f>0)*1
        n = len(f.columns)
        adoption_i={f.iloc[l].name : adop_index_int(f,d,l,n) for l in range(len(f.index))}
        a = pandas.Series(adoption_i)
        a.to_csv(filefullpath,encoding="utf-8")

def adop_index_int(f,d,k,n):
    f_w = f.iloc[k,]
    d_w = d.iloc[k,]
    return sum([(f_w.iloc[i] - f_w.iloc[j]) * d_w[i] * d_w[j] for i in range(n) for j in range(i + 1, n)])

def adop_index(f,d,w):
    f_w = f.loc[w,]
    d_w = d.loc[w,]
    n = len(d_w)
    return sum([(f_w.iloc[i] - f_w.iloc[j]) * d_w[i] * d_w[j] for i in range(n) for j in range(i+1,n)])

def frequency(dataset_name,start,end,n,ngram_min,ngram_max,analysis_type,nperiods,preprocessing):
    directory = os.path.join(*[os.path.dirname(os.path.realpath(__file__)), "tmp", dataset_name, "fq"])
    h,c = obtain_file_name_from_dataset(dataset_name,preprocessing)
    corpus = obtain_full_corpus(h,c)
    b = register_frequency(dataset_name,start,end,n,ngram_min,ngram_max,analysis_type,preprocessing)
    if b != False:
        create_frequency_file(directory,corpus,b)
