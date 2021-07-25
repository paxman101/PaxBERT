import re

import fasttext
import numpy as np
from modin import pandas as pd
from bs4 import BeautifulSoup
import ray

ray.init()

PRETRAINED_MODEL_PATH = 'lid.176.bin'
model = fasttext.load_model(PRETRAINED_MODEL_PATH)


def clean_data(elem):
    if not isinstance(elem, str):
        return np.nan
    elem = BeautifulSoup(elem, "lxml").text
    sub = re.compile(r"\\")
    elem = re.sub(sub, '', elem)
    pred = model.predict(elem.lower())
    language = pred[0][0].replace('__label__', '')
    if language != 'en':
        return np.nan
    return elem


chunk = 0
for df in pd.read_csv('amazon_reviews_us_Books_v1_02.tsv', delimiter='\t', error_bad_lines=False, chunksize=50000):
    df['review_body'] = [clean_data(x) for x in df['review_body']]
    df.dropna(axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv('books_v1_02_cleaned_{}.csv'.format(chunk))
    chunk += 1
