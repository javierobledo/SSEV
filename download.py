import sys, os, re, requests, shutil
from bs4 import BeautifulSoup
from zipfile import ZipFile
import urllib.request as ul
from utils.tcp_hdr2csv import headers_to_csv
from utils.xml2csv import corpus_to_csv

def files_actually_unzipped(origin_directory, destination_directory):
    """
    Verifies if the origin_directory with zipped files are actually unzipped. If files are unzipped, return True.
    Returns False otherwise. If the origin_directory doesn't exist an IOError is raised.
    :param origin_directory: The directory's path where ECCO-TCP zip files are stored
    :param destination_directory: The directory's path where the ECCO-TCP dataset will be unzipped
    :return: True or False
    """
    names = []
    if not os.path.exists(origin_directory):
        raise IOError("The dataset directory doesn't exist")
    if not os.path.exists(destination_directory):
        os.mkdir(destination_directory)
    for file in os.listdir(origin_directory):
        if file.endswith(".zip"):
            names += ZipFile(os.path.join(origin_directory, file)).namelist()
    other_names = os.listdir(destination_directory)
    return len(set(names) & set(other_names)) == len(set(names))


def unzip_ecco_tcp_xmls(origin_directory, destination_directory):
    """
    Create, if doesn't exist, the destination_directory and unzip all zip files contained in origin_directory.
    If the origin_directory doesn't exist, an exception is raised.
    :param origin_directory: The directory's path where ECCO-TCP zip files are stored
    :param destination_directory: The directory's path where the ECCO-TCP dataset will be unzipped
    :return: None
    """
    if not files_actually_unzipped(origin_directory, destination_directory):
            for file in os.listdir(origin_directory):
                if file.endswith(".zip"):
                    ZipFile(os.path.join(origin_directory, file)).extractall(destination_directory)

def get_all_data(dataseturl):
    soup = BeautifulSoup(requests.get(dataseturl).text, "lxml")
    #for a in soup.find('table').find_all('a'):
    for a in soup.find('body').find_all('a'):
        link = a['href']
        if re.match(r'^xml.*\.zip', link) or 'headers.ecco.zip' in link:
            yield dataseturl + link, link

def reporthook(blocknum, blocksize, totalsize):
    read = blocknum * blocksize
    total = totalsize // blocksize
    if totalsize > 0:
        percent = read * 100 // totalsize
        printProgress(percent, 100,prefix='Progress:',suffix='Complete', barLength=50)


def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(barLength * iteration // total)
    bar = fill * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def erase_all_files_with_extension(directory,dataset,extension):
    dir = os.path.join(directory, dataset)
    filesnames = os.listdir(dir)

    for file in filesnames:
        if file.endswith(extension):
            os.remove(os.path.join(dir, file))



def download(dataset_name,dataset_url):
    """returns True if download occurs successfully and False otherwise.

                Download the dataset given from the url, associate the name to it, and create two dataframes, one for the metadata
                and other for the documents itself.

                **Important**:
                 - Only ECCO-TCP dataset works.

                :param dataset_name: the first value
                :param dataset_url: the first value
                :type dataset_name: int, float,...
                :type dataset_url: int, float,...
                :returns: True or False
                :rtype: bool

                :Example:

                >>> import download
                >>> a = download("ECCO-TCP","")
                >>> a
                True
                """
    directory = "tmp"
    if not os.path.exists(os.path.join(directory,dataset_name)):
        os.makedirs(os.path.join(directory,dataset_name))
    for url, filename in get_all_data(dataset_url):
        if not os.path.exists(os.path.join(directory,dataset_name,filename)):
            print("Downloading "+filename+":",)
            ul.urlretrieve(url,os.path.join(directory,dataset_name,filename),reporthook)
    unzip_ecco_tcp_xmls(os.path.join(directory, dataset_name), os.path.join(directory, dataset_name + "_unzipped"))
    shutil.rmtree(os.path.join(directory, dataset_name))
    shutil.move(os.path.join(directory, dataset_name + "_unzipped"), os.path.join(directory, dataset_name))
    headers_to_csv(directory, dataset_name)
    corpus_to_csv(directory, dataset_name)
    erase_all_files_with_extension(directory, dataset_name, ".hdr")
    erase_all_files_with_extension(directory, dataset_name, ".xml")
