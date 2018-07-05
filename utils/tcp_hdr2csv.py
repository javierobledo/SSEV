#!/usr/bin/env python

from bs4 import BeautifulSoup
import re
import csv
import codecs
import os
from os.path import basename

author_dates_pat = re.compile(r"(\sill.\s)?(\sfl.\s)?(\sca.\s)?(\sd.\s)??\d+\??( or \d+)?(-\d+)?\??[\.,]?")
filenames = []

out_rows = ["id", "dlps", "title", "author", "dates",
        "pubplace", "pub", "pubdate"]


def process_file(f, out):
    doc = BeautifulSoup(f, "xml")

    row = dict()
    fdesc = doc.find("FILEDESC")

    row["id"] = os.path.splitext(basename(f.name))[0]

    # TODO fails on EEBO, headers are lc and different
    # ECCO entries have IDNO types: TCP, DLPS, ESTC, DocNo
    # EEBO entries have IDNO types: DLPS, stc, estc, eebo citation, proquest, vid
    try:
        row["dlps"] = fdesc.find("IDNO", TYPE="DLPS").string
    except AttributeError:
        pass

    src = doc.find("BIBLFULL")
    try:
        row["title"] = src.TITLESTMT.TITLE.string
    except AttributeError:
        pass

    try:
        a_str = src.TITLESTMT.AUTHOR.string
        row["author"] = author_dates_pat.sub("", a_str)
        row["dates"] = author_dates_pat.search(a_str).group(0)
    except AttributeError:
        pass

    try:
        row["pubplace"] = src.PUBLICATIONSTMT.PUBPLACE.string
    except AttributeError:
        pass

    try:
        row["pub"] = src.PUBLICATIONSTMT.PUBLISHER.string
    except AttributeError:
        pass

    try:
        row["pubdate"] = src.PUBLICATIONSTMT.DATE.string
    except AttributeError:
        pass
    # unicode-ify
    out.writerow({ k: v for k, v in row.items() })

def header_filenames(directory):
    l = []
    for file in os.listdir(directory):
        if file.endswith(".hdr"):
            l.append(os.path.join(directory,file))
    return l

def headers_to_csv(dir,dataset):
    headername = dataset+"_header.csv"
    datasetdir = os.path.join(dir,dataset)
    with open(os.path.join(datasetdir,headername), 'w') as csvfile:
        out = csv.DictWriter(csvfile, out_rows, quoting=csv.QUOTE_ALL)
        filenames = header_filenames(datasetdir)
        out.writeheader()

        def proc(fn):
            with codecs.open(fn, "r") as f:
                process_file(f, out)

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
        out.writeheader()
        filenames = sys.argv[2:]
    else:
        filenames = sys.argv[1:]

    def proc(fn):
        with codecs.open(fn, "r", encoding="utf-8") as f:
            process_file(f, out)

    for fn in filenames:
        if os.path.isdir(fn):
            for f in os.listdir(fn):
                proc(os.path.join(fn, f))
        else:
            proc(fn)

