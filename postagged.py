from utils.Postagger import Postagger
import os
from status import existe_dataset

def postag(dataset_name,postagger):
    directory = os.path.join(os.getcwd(), "tmp")
    out = os.path.join(*[os.path.dirname(os.path.realpath(__file__)),"tmp",dataset_name,dataset_name+"_corpus_tagged.csv"])
    if existe_dataset(dataset_name):
        postag = Postagger(dataset_name, directory, postagger)
        inp = os.path.join(*[os.path.dirname(os.path.realpath(__file__)),"tmp",dataset_name,dataset_name+"_corpus.csv"])
        postag.write_csv(out, postag.postagging_corpus(postag.read_csv(inp)))