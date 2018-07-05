import os

def status(dataset_name):
    print("Dataset\t\tCorpus\t\tTagged")
    if existe_dataset(dataset_name):
        if existe_tagset(dataset_name):
            print(dataset_name+"\t"+"OK\t\tOK")
        else:
            print(dataset_name + "\t" + "OK\t\t-")
    else:
        print(dataset_name + "\t" + "-\t\t-")

def existe_dataset(dataset_name):
    f1 = os.path.join(*[os.path.dirname(os.path.realpath(__file__)),"tmp",dataset_name,dataset_name+"_corpus.csv"])
    f2 = os.path.join(*[os.path.dirname(os.path.realpath(__file__)),"tmp",dataset_name,dataset_name+"_header.csv"])
    return os.path.exists(f1) and os.path.exists(f2)

def existe_tagset(dataset_name):
    f = os.path.join(*[os.path.dirname(os.path.realpath(__file__)), "tmp", dataset_name, dataset_name + "_corpus_tagged.csv"])
    return os.path.exists(f)