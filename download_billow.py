import hashlib

import requests
import os
from tqdm import tqdm
import pandas as pd


def download(url, pathname, filename = None, verbose=False):
    """
    Downloads a file given an URL and puts it in the folder `pathname`
    """
    # if path doesn't exist, make that path dir
    if not os.path.isdir(pathname):
        os.makedirs(pathname)
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get("Content-Length", 0))
    if filename is None:
        filename = hashlib.sha1(response.content).hexdigest()[:10] + '.jpg'
    filename = os.path.join(pathname,filename)

    # progress bar, changing the unit to bytes instead of iteration (default by tqdm)
    progress = tqdm(response.iter_content(1024), f"Downloading {filename}",
                    total=file_size, unit="B", unit_scale=True, unit_divisor=1024, disable = ~verbose)
    with open(filename, "wb") as f:
        for data in progress.iterable:
            # write data read to the file
            f.write(data)
            # update the progress bar manually
            progress.update(len(data))



save_path = 'billow_illustrations'

imgs = pd.read_csv('download_list.csv')

# to download all set n_download = 300000
n_download = 10


for index, img in tqdm(imgs.iterrows()):
    if index > n_download:
        break
    save_dir = os.path.join(save_path, img['name'])
    filename = str(img['data-asset-id'])+'.jpg'
    
    if not os.path.isfile(os.path.join(save_dir,filename)):
        # for each image, download it
        download(img['url'], save_dir, filename = filename)