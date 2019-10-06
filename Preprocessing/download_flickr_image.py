# First install flickrapi
# pip install flickrapi
import flickrapi
import wget
import os
import pandas as pd
from functions.data import get_label_to_remap

out_path = '../Data/Flickr'

done = ['']
l2r = get_label_to_remap()

keywords = sorted(['mat'])

for keyword in keywords:
    # Flickr api access key
    flickr=flickrapi.FlickrAPI('<your_key>', '<your_key>', cache=True)
    remap = l2r[keyword]
    photos = flickr.walk(text=remap, tag_mode='all', tags=[remap], extras='url_c', per_page=100, sort='relevance')
    urls = []
    try:
        for i, photo in enumerate(photos):    
            url = photo.get('url_c')
            if url is not None:
                urls.append(url)
            if i > 1500:
                break
    except Exception as e:
        print(e)
    print('{0:s} : {1:d}'.format(keyword, len(urls)))
    os.system('rm -r {0:s}/{1:s}'.format(out_path, keyword))
    os.system('mkdir -p {0:s}/{1:s}'.format(out_path, keyword))
    # Download image from the url and save it to '00001.jpg'
    for idx, url in enumerate(urls):
        try:
            if not os.path.exists('{0:s}/{1:s}/{2:d}.jpg'.format(out_path, keyword, idx)):
                wget.download(url, '{0:s}/{1:s}/{2:d}.jpg'.format(out_path, keyword, idx))
        except Exception as e:
            print(e)
