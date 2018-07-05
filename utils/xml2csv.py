#!/usr/bin/env python

from bs4 import BeautifulSoup
import codecs
import os, csv, pandas
from os.path import basename

out_rows = ["id", "text"]

def file2row(f):
    doc = BeautifulSoup(f, "xml")

    txt = doc.find("TEXT")
    if txt == None:
        txt = doc.find("text")
        if txt == None:
            raise Exception("Couldn't find <text> element")

    return {"id":os.path.splitext(basename(f.name))[0], "text": txt.get_text()
        .replace("\n", " ").replace('"','')}

def text_filenames(directory):
    l = []
    for file in os.listdir(directory):
        if file.endswith(".xml"):
            l.append(os.path.join(directory,file))
    return l

def corpus_to_csv(dir,dataset):
    corpusname = dataset+"_corpus.csv"
    datasetdir = os.path.join(dir, dataset)
    with open(os.path.join(datasetdir, corpusname), 'w') as csvfile:
        out = csv.DictWriter(csvfile, out_rows, quoting=csv.QUOTE_ALL)
        filenames = text_filenames(datasetdir)
        out.writeheader()
        def proc(fn):
            with codecs.open(fn, "r") as f:
                line = file2row(f)
                out.writerow(line)

        for fn in filenames:
            if os.path.isdir(fn):
                for f in os.listdir(fn):
                    proc(os.path.join(fn, f))
            else:
                proc(fn)

if __name__ == "__main__":
    import sys

    out = csv.DictWriter(sys.stdout, out_rows, quoting=csv.QUOTE_ALL)

    if sys.argv[1] == "-h":
        print("id\ttext")
        filenames = sys.argv[2:]
    else:
        filenames = sys.argv[1:]

    def proc(fn):
        with codecs.open(fn, "r", encoding="utf-8") as f:
            line = file2row(f)
            print(line.encode("utf-8"))

    for fn in filenames:
        if os.path.isdir(fn):
            for f in os.listdir(fn):
                proc(os.path.join(fn, f))
        else:
            proc(fn)
