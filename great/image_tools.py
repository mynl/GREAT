# image quilt from directory of pictures- general purpose!
# google book cover images quilt
# from telos\python\learning

from PIL import Image
# import bs4  ## beautiful soup
# import copy
import hashlib
import json
# import lxml
from lxml import html
import numpy as np
import os
import requests
from pathlib import Path 
from IPython.display import Image as ipImage

HEADER = {
    "Accept": "text/html,application/xhtml+xml,application/xml; "
              "q=0.9,image/webp,*/*;q=0.8",
    "Accept-Encoding": "text/html",
    "Accept-Language": "en-US,en",
    "Content-Type": "application/x-www-form-urlencoded",
    "User-Agent": "Mozilla/5.0"
}

def quilt(search_term, n, WW=1600, HH=1200, verbose=False, display=False):
    print('searching')
    dd = bookSearch(search_term, n)
    print(f'found {len(dd)} items\ndownloading images')
    p = Path(f'./{search_term}')
    saveImages(dd, p, verbose=verbose)
    print('making quit')
    out_file = Path(f'./{search_term}.jpg')
    makeQuilt(p, out_file, WW=WW, HH=HH)
    print('complete')
    if display:
        ipImage(filename=out_file, retina=True)
    return out_file

def makeQuilt(dir_path, out_path, *, img_file_filter='*.jpg', WW=1600, HH=1200, pad=0,
              img_width=128, h_tol=10, min_colors=0, display_image=False):
    '''
    128 is the natural image width
    most are 190 high

    :param dir_pame: Path directory or list of directories for input images
    :param out_path: Path or filename for output image
    :param img_filter: file filter sent to glob to find images
    :param WW:
    :param HH:
    :param pad:
    :param img_width: width of individual image in quilt
    :param h_tol: how close to end before starting a new row
    :param min_colors: only images with >= colors included; if =0 no filtering applied
    :param display_image: show image if done...requires Jupyter
    :return:
    '''
    ASPECT = 190/128

    if isinstance(dir_path, Path):
        dir_path = [dir_path]

    # a list of dir paths
    bimgs = []
    for p in dir_path:
        bimgs.extend(list(p.glob(img_file_filter)))
    print(f'Starting with {len(bimgs)} images...')

    # filter out the images
    if min_colors > 0:
        print('...filtering for min colors...')
        imgs = []
        for f in bimgs:
            img = Image.open(f)
            colors = img.getcolors(1024000) #put a higher value if there are many colors in your image
            max_occurence, most_present = 0, 0
            try:
                for c in colors:
                    if c[0] > max_occurence:
                        (max_occurence, most_present) = c
            except TypeError:
                # print(most_present, max_occurence)
                raise Exception("Too many colors in the image")
            if len(colors) >= min_colors:
                imgs.append(f)
    else:
        # no filtering if
        imgs = bimgs

    print(f'...found {len(imgs)} to use in quilt...')

    # repeat the images, if seems necessary or figure out size
    if img_width==0:
        # WW*HH is area of canvas; area of image roughly 1.6 * img_width**2 so need
        img_width = int(1.1 * np.sqrt(WW*HH/(1.6*len(imgs))))
        print(f'Computed image width = {img_width}')

    # assemble quilt
    print('...assembling quilt')
    blank_image = Image.new("RGB", (WW, HH))
    # padding around individual images
    wpad = pad
    hpad = pad
    icount = 0
    w = wpad
    h = hpad
    n_img = 0
    img_iter = iter(np.random.permutation(imgs))
    while True:
        try:
            fn = next(img_iter)
        except StopIteration:
            print('refreshing image iterator')
            img_iter = iter(np.random.permutation(imgs))
            fn = next(img_iter)

        icount += 1
        if icount > 1000:
            break
        # print(f'{icount:4d} {fn}')
        with Image.open(fn) as i:
            sz = i.size
            new_sz = (img_width, int(float(img_width)/sz[0]*sz[1]))
            j = i.resize(new_sz)
            blank_image.paste(j, (w,h))
            n_img += 1
            h += j.size[1] + hpad
            if h > HH - h_tol:
                h = hpad
                w += img_width + wpad
                print(f'starting new column, current width={w}')
                if w >= WW:
                    print(f'image width exceeded {w} > {WW}...breaking')
                    break
    blank_image.save(out_path)
    if display_image is True:
        ipImage(filename=out_path)

    return n_img


def makeQuilt_old(dir_path, out_path, *, img_file_filter='*.jpg', WW=1600, HH=1200, pad=0,
              img_width=128, min_colors=0, display_image=False):
    '''
    128 is the natural image width
    most are 190 high

    :param dir_pame: Path directory or list of directories for input images
    :param out_path: Path or filename for output image
    :param img_filter: file filter sent to glob to find images
    :param WW:
    :param HH:
    :param pad:
    :param img_width: width of individual image in quilt
    :param min_colors: only images with >= colors included; if =0 no filtering applied
    :param display_image: show image if done...requires Jupyter
    :return:
    '''
    ASPECT = 190/128

    if isinstance(dir_path, Path):
        dir_path = [dir_path]

    # a list of dir paths
    bimgs = []
    for p in dir_path:
        bimgs.extend(list(p.glob(img_file_filter)))
    print(f'Starting with {len(bimgs)} images...')

    # filter out the images
    if min_colors > 0:
        print('...filtering for min colors...')
        imgs = []
        for f in bimgs:
            img = Image.open(f)
            colors = img.getcolors(1024000) #put a higher value if there are many colors in your image
            max_occurence, most_present = 0, 0
            try:
                for c in colors:
                    if c[0] > max_occurence:
                        (max_occurence, most_present) = c
            except TypeError:
                # print(most_present, max_occurence)
                raise Exception("Too many colors in the image")
            if len(colors) >= min_colors:
                imgs.append(f)
    else:
        # no filtering if
        imgs = bimgs

    print(f'...found {len(imgs)} to use in quilt...')

    # repeat the images, if seems necessary or figure out size
    if img_width==0:
        # WW*HH is area of canvas; area of image roughly 1.6 * img_width**2 so need
        img_width = int(1.1 * np.sqrt(WW*HH/(1.6*len(imgs))))
        print(f'Computed image width = {img_width}')
    else:
        if len(imgs) < WW * HH / (img_width ** 2 * ASPECT):
            # pad images
            imgs = imgs * (1 * 
                    int( (WW * HH / (img_width**2 * ASPECT) / len(imgs))))

    # shuffle
    print(f'shuffling {len(imgs)} after padding ')
    imgs = np.random.permutation(imgs)

    # assemble quilt
    print('...assembling quilt')
    blank_image = Image.new("RGB", (WW, HH))
    w = 0
    h = 0
    n_img = 0
    # padding around individual images
    wpad = pad
    hpad = pad
    hTol = 100
    for fn in imgs:
        with Image.open(fn) as i:
            sz = i.size
            new_sz = (img_width, int(float(img_width)/sz[0]*sz[1]))
            j = i.resize(new_sz)
            blank_image.paste(j, (w,h))
            n_img += 1
            h += j.size[1] + hpad
            if h > HH - hTol:
                h = hpad
                w += img_width + wpad
                if w > WW:
                    print('image width exceeded {w} > {WW}...breaking')
                    break
    blank_image.save(out_path)
    if display_image is True:
        ipImage(filename=out_path)

    return n_img

def makeQuilt_very_old(dirName, outFileName, WW=3200, HH=2400, pad=0, img_width=128):
    '''
    128 is the natural image width
    most are 190 high

    :param dirName:
    :param outFileName:
    :param WW:
    :param HH:
    :param pad:
    :param img_width:
    :return:
    '''
    ASPECT = 190/128

    # padding around individual images
    wpad = pad
    hpad = pad

    hTol = 100
    imgs = []
    for f in os.listdir(dirName):
        fn = os.path.join(dirName, f)
        imgs.append(fn)

    # repeat the images, if seems necessary or figure out size
    if img_width==0:
        # WW*HH is area of canvas; area of image roughly 1.6 * img_width**2 so need
        img_width = int(1.1 * np.sqrt(WW*HH/(1.6*len(imgs))))
        print('Computed image width = {:}',format(img_width))
    else:
        if len(imgs) < WW*HH/(img_width**2*ASPECT):
            imgs = imgs * (1*int(WW*HH/(img_width**2*ASPECT/len(imgs))))

    # shuffle
    imgs = np.random.permutation(imgs)

    # assemble quilt
    blank_image = Image.new("RGB", (WW, HH))
    w = 0
    h = 0
    n_img = 0
    for fn in imgs:
        with Image.open(fn) as i:
            sz = i.size
            new_sz = (img_width, int(float(img_width)/sz[0]*sz[1]))
            j = i.resize(new_sz)
            # print(fn[-20:], i.size, j.size)
            blank_image.paste(j, (w,h))
            n_img += 1
            h+=j.size[1]+hpad
            # print(w, h)
            if h>HH-hTol:
                h=hpad
                w+=img_width+wpad
                if w > WW:
                    print("too many images...")
                    break
    blank_image.save(outFileName)
    return n_img

def qSearch(searchTerm):
    # requests for humans:
    url = 'https://www.googleapis.com/books/v1/volumes'
    params = {'q': searchTerm, 'maxResults': 40, 'key': 'AIzaSyBOl70s9noqg61i4yaHaF-o7BjWPe2f5Sw'}
    req = requests.get(url, params=params)
    d = json.loads(req.text)
    return [b['volumeInfo'] for b in d['items']]  ## list of dictionaries of the answer


def bookSearch(searchTerm, numItems=40):
    ans = list()
    qstr = 'https://www.googleapis.com/books/v1/volumes'
    args = dict()
    args['q'] = searchTerm

    MAXITEMS = 40
    itop = int(numItems / MAXITEMS)
    if itop * MAXITEMS != numItems:
        itop += 1

    for i in range(0, itop):
        args['startIndex'] = i * MAXITEMS
        if (i + 1) * MAXITEMS > numItems:
            args['maxResults'] = numItems - args['startIndex']
        else:
            args['maxResults'] = MAXITEMS
        # do the request
        # print arg_dict
        j = requests.get(qstr, params=args).json()
        try:
            for vol in j['items']:
                vi = vol['volumeInfo']
                d = dict()
                for i in ['authors', 'pageCount', 'publisher', 'title', 'subtitle', 'publishedDate', 'printType',
                          'description', 'imageLinks', 'infoLink', 'categories', 'canonicalVolumeLink']:
                    # for i in ['authors', 'title' ]:
                    d[i] = vi.get(i)
                ans.append(d)
        except:
            print("Unexpected error:", sys.exc_info()[0])
    return ans

## words in the titles
def wordList(a1):
    w = list()
    for i in a1:
        s = i['title'].lower().split(' ')
        w.extend(s)

    w.sort()
    ws = set(w)
    d = dict()
    for k in ws:
        d[k] = w.count(k)
    return d

def cleanName(str):
    s = str.replace('.', '')
    s = s.replace(', Jr', ' Jr')
    s = s.replace('Mr ', '')
    s = s.replace('Dr ', '')
    s = s.replace('Prof ', '')
    s = s.replace('Professor ', '')
    w = s.split(', ')
    if len(w) == 2:
        s = w[1] + ' ' + w[0]
    return s

def authorList(a1):
    ## authors
    w = list()
    for i in a1:
        if i.get('authors', None) is not None:
            for s in i['authors']:
                w.append(cleanName(s))
    w.sort()
    ws = set(w)
    d = dict()
    for k in ws:
        d[k] = w.count(k)
    return d

# def getImage(url, fname):
#     img = urlopen(url)
#     with open(fname, 'wb') as localFile:
#         localFile.write(img.read())


def quickHash(str):
    hasher = hashlib.md5()
    hasher.update(str.encode('utf-8'))
    return hasher.hexdigest()


def saveImages(a, base_path, verbose=False):
    # a = json object from search
    base_path.mkdir(parents=True, exist_ok=True)
    i = 0 
    for b in a:
        l1 = b.get('imageLinks')
        if l1 is not None:
            i += 1
            imgurl = l1.get('thumbnail')
            fn = base_path / f'{quickHash(imgurl)}.jpg'
            authors = b.get('authors')
            if authors is not None:
                authors = authors[0]
            else:
                authors = ''
            message = f"{i:3d} {authors:30s}\t{b.get('title')}"
            if verbose:
                print(message)
            fn.open('wb').write(requests.get(imgurl).content)

            
def authorSearch(searchTerm, authors, maxitems=10, depth=1):  
    # authors = set, answers are added to it
    url = 'https://www.googleapis.com/books/v1/volumes' # ?q=' + searchTerm.replace(' ', '+')
    page = requests.get(url, params = {'q' : searchTerm, 'maxitems' : maxitems})
    d = page.json()
    for b in d['items']:
        try:
            for a in b['volumeInfo']['authors']:
                authors.add(a)
        except:
            pass
    print('Search for ' + searchTerm + ' completed...currently found {:} terms'.format(len(authors)))
    if depth > 0:
        for aa in list(authors):
            authorSearch(aa, authors, maxitems, depth - 1)
    return len(authors)


def links_on_page(url, link_set):
    page = requests.get(url)
    tree = html.fromstring(page.text)
    links = tree.xpath('//a/@href')
    for l in links:
        if l[0:4] == 'http':
            u2 = l
        else:
            u2 = url + "/" + l
        link_set.add(u2)
    return link_set


def recursive_proc(url, link_set, depth):
    page_links = links_on_page(url, set())
    link_set = link_set.union(page_links)
    if depth > 0:
        for l in page_links:
            link_set = link_set.union(recursive_proc(l, link_set, depth - 1))
    return link_set


