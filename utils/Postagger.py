import nltk, csv, os, sys
from nltk.tag import UnigramTagger,BigramTagger,TrigramTagger
from nltk.corpus import treebank
from nltk.tokenize import sent_tokenize, word_tokenize
import codecs

class Postagger:
    csv.field_size_limit(sys.maxsize)

    def __init__(self,dataset,directory,tagger):
        self.dataset = dataset
        self.directory = directory
        train_sents = treebank.tagged_sents()
        if tagger == "unigram":
            self.tagger = UnigramTagger(train_sents)
        elif tagger == "bigram":
            self.tagger = BigramTagger(train_sents)
        elif tagger == "trigram":
            self.tagger = TrigramTagger(train_sents)

    def read_csv(self,filename):
        l = []
        with open(filename, encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                l.append(row)
            return l
    def write_csv(self,filename, data):
        fieldnames = ["id","text"]
        with open(filename,'w') as csvfile:
            writer = csv.DictWriter(csvfile,fieldnames,quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for row in data:
                writer.writerow(row)

    def postagging_text(self,text):
        l = []
        for sentence in sent_tokenize(text):
            tagged = self.tagger.tag(word_tokenize(sentence))
            l.append(tagged)
        return l


    def from_postag_to_text(self,postaged_text):
        text = ""
        for postagged in postaged_text:
            line = []
            for w,t in postagged:
                line.append(str(t))
            text += " "+ " ".join(line)+" "
        return text

    def postagging_corpus(self,corpus):
        l = []
        for text in corpus:
            postagged_text = self.postagging_text(text["text"])
            text_postag = self.from_postag_to_text(postagged_text)
            l.append({"id":str(text["id"]),"text":text_postag})
        return l

class UnicodeDictReader( object ):
    def __init__( self, *args, **kw ):
        self.encoding= kw.pop('encoding', 'mac_roman')
        self.reader= csv.DictReader( *args, **kw )
    def __iter__( self ):
        decode= codecs.getdecoder( self.encoding )
        for row in self.reader:
            t= dict( (k,decode(row[k])[0]) for k in row )
            yield t
