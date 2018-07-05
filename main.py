import argparse
from download import download
from status import status
from postagged import postag
from frequency import frequency,use_index,adoption_index
#"http://www.lib.umich.edu/tcp/docs/texts/ecco/"
def download_function(args):
    dataset_name = args.dataset_name
    dataset_url = args.dataset_url
    download(dataset_name,dataset_url)

def status_function(args):
    dataset_name = args.dataset_name
    status(dataset_name)

def postagger_function(args):
    dataset_name = args.dataset_name
    postagger_type = args.postagger_type
    postag(dataset_name,postagger_type)

def frequency_function(args):
    dataset_name = args.dataset_name
    ngram_min = args.ngram_min
    ngram_max = args.ngram_max
    analysis_type = args.analysis_type
    year_start = args.year_start
    year_end = args.year_end
    n = args.n
    preprocessing = args.preprocessing
    if analysis_type == "tf" or analysis_type == "df":
        frequency(dataset_name,year_start,year_end,n,ngram_min,ngram_max,analysis_type,n,preprocessing)
    elif analysis_type == "u":
        use_index(dataset_name,year_start,year_end,ngram_min,ngram_max,analysis_type,n,preprocessing)
    elif analysis_type == "a":
        adoption_index(dataset_name, year_start, year_end, ngram_min, ngram_max, analysis_type, n, preprocessing)

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

downloadparser = subparsers.add_parser("download")
downloadparser.add_argument('--dataset-name', action='store', default='ecco-tcp', dest='dataset_name',
                    help='Dataset Name')
downloadparser.add_argument('--dataset-url', action='store', default='http://localhost:8000/',
                    dest='dataset_url', help='Dataset Url')
downloadparser.set_defaults(func=download_function)

statusparser = subparsers.add_parser("status")
statusparser.add_argument('--dataset-name', action='store', default='ecco-tcp', dest='dataset_name',
                    help='Dataset Name')
statusparser.set_defaults(func=status_function)

postagparser = subparsers.add_parser("postagged")
postagparser.add_argument('--dataset-name', action='store', default='ecco-tcp', dest='dataset_name',
                    help='Dataset Name')
postagparser.add_argument('--postagger-type', action='store', default='unigram', dest='postagger_type',
                    help='Postagger Type')
postagparser.set_defaults(func=postagger_function)

frequencyparser = subparsers.add_parser("frequency")
frequencyparser.add_argument('--dataset-name', action='store', default='ecco-tcp', dest='dataset_name',
                    help='Dataset Name')
frequencyparser.add_argument('--ngram-min', action='store', default=1, dest='ngram_min',type=int)
frequencyparser.add_argument('--ngram-max', action='store', default=1, dest='ngram_max',type=int)
frequencyparser.add_argument('--year-start', action='store', default=1700, dest='year_start',type=int)
frequencyparser.add_argument('--year-end', action='store', default=1800, dest='year_end',type=int)
frequencyparser.add_argument('--analysis-type', action='store', default='tf', dest='analysis_type',
                    help='Term frequency (tf), Document frequency (df), User Index (u) or Adoption Index (a)')
frequencyparser.add_argument('--number-time-periods', action='store', default=10, dest='n', type=int,
                    help='Number of portions of time to consider. If 10, this means 10 time periods: 1701 to 1710, 1711 to 1720 and so on')
frequencyparser.add_argument('--preprocessing', action='store', default='original', dest='preprocessing',
                    help='Determine if the original, tagged or random corpus is used')
frequencyparser.set_defaults(func=frequency_function)

#options = ["download","--dataset-name","ecco-tcp","--dataset-url","http://localhost:8000/"]
args = parser.parse_args()
if hasattr(args, 'func'):
    args.func(args)