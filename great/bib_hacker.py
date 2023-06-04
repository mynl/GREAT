

import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase
from pathlib import Path
import pickle
import re

p = Path('.')
bib = Path('\\s\\telos\\biblio\\')
# p = Path('c:\\temp\\sp')


def load_pickle():
    b = pickle.load((bib/'bibtex.pickle').open('rb'))
    return b


def refresh_pickle():
    with (bib/'library.bib').open('r', encoding='utf-8') as bibtex_file:
        bdb = bibtexparser.load(bibtex_file)
    pickle.dump(bdb, (bib/'bibtex.pickle').open('wb'))


def extract_refs_make_local_bib():
    # read all rst files, find cite tag pull out reference
    ans = []
    for fn in p.glob('**/*.rst'):
        txt = fn.read_text(encoding='utf-8')
        ans.extend(re.findall(":cite:?.?:`([^`]*)`", txt))
    # unique elements
    # deal with comma separated list
    ans = [k for j in [i.split(',') for i in ans]  for k in j ]
    # reduce to unique values
    ans = set(ans)
    print('refs found', sorted(ans))

    # pull out those entries
    bdb = load_pickle()
    out = []
    for k in ans:
        if k in bdb.entries_dict:
            out.append(bdb.entries_dict[k])
        else:
            print('not found ', k)
    # create new database
    db = BibDatabase()
    db.entries = out
    # write local file
    writer = BibTexWriter()
    with (p/'extract.bib').open('w', encoding='utf-8') as bibfile:
        bibfile.write(writer.write(db))


def extract_refs_make_local_bib_md():
    # read all rst files, find cite tag pull out reference
    ans = []
    for fn in p.glob('**/*.md'):
        txt = fn.read_text(encoding='utf-8')
        ans.extend(re.findall("@([A-Za-z0-9]+)", txt))
    # unique elements
    ans = set(ans)
    print('refs found', sorted(ans))

    # pull out those entries
    bdb = load_pickle()
    out = []
    for k in ans:
        if k in bdb.entries_dict:
            out.append(bdb.entries_dict[k])
        else:
            print('not found ', k)
    # create new database
    db = BibDatabase()
    db.entries = out
    # write local file
    writer = BibTexWriter()
    with (p/'extract.bib').open('w', encoding='utf-8') as bibfile:
        bibfile.write(writer.write(db))

if __name__ == '__main__':
    print('Updating local bib file extract.bib')
    print(f'Current directory = {p.resolve()}')
    extract_refs_make_local_bib()
