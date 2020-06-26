import requests, zipfile, io

def download_glove():
    url = 'http://nlp.stanford.edu/data/glove.6B.zip'
    print('downloading Glove word Embeddings')
    r = requests.get(url, stream =True)
    check = zipfile.is_zipfile(io.BytesIO(r.content))
    while not check:
        r = requests.get(url, stream =True)
        check = zipfile.is_zipfile(io.BytesIO(r.content))
    else:
        print("embedding downloaded!")
        print("Extracting embedding to the files folder")
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall('glove')
