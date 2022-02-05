# Source: originated in meta_reddit hack, when creating pure word list and
# word clouds for the book
# Dec 2021
#
from abc import ABC

import matplotlib as mpl
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
import numpy as np
import pandas as pd
from PIL import Image

import re
from pathlib import Path
import html
from collections import OrderedDict
from textwrap import fill
import logging
from difflib import Differ
from pprint import pprint
from IPython.display import display, SVG

# from IPython.display import Markdown, HTML

from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer
from nltk.corpus import stopwords

logger = logging.getLogger('textanalyzer')
logger.setLevel(logging.DEBUG)

if len(logger.handlers) == 0:
    # to stderr
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter(
        '%(lineno) 5d | %(funcName)-15s | %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)


class TextAnalyzer(object):

    def __init__(self, fn):
        """
        Take in a text blob, tidy it up, e.g., remove TeX and Markdown markup (TODO HTML)

        Split, stem, and summarize. Produce various iterables and statistics output.

        fn = None allows for debug mode

        :param fn:   Path of input file

        """
        self.fn = fn
        self.txt_in = ''
        # the current state...all work is done on self.txt
        self.txt = ''
        self._stop_words = None
        self._stemmer = None
        # frequency DataFrame
        self._sf = None
        if fn is not None:
            self.read()

    def diff(self, top=-1, width=0):
        """
        TODO: this doesn't work at all!
        report diffs between in and out

        This may not work so well because of line splitting...
        :return:
        """
        logger.error("NYI "*20)
        d = Differ()
        if width:
            t1 = fill(self.txt_in, width).splitlines(keepends=True)
            t2 = fill(self.txt, width).splitlines(keepends=True)
        else:
            t1 = self.txt_in
            t2 = self.txt
        if top > 0:
            t1 = t1[:80*top]
            t2 = t2[:80*top]
        result = list(d.compare(t1, t2))
        pprint(result[:top])

    def read(self):
        """
        read in the text file, invalidate other variables

        :return:
        """
        assert isinstance(self.fn, Path) and self.fn.exists()
        with self.fn.open('r', encoding='utf-8') as f:
            self.txt_in = f.read()
        self.txt = self.txt_in
        logger.info(f'read {self.fn}, {len(self.txt_in):,d} characters.')

    def write(self, prefix='STRIPPED', width=0):
        """
        write out current state, if different from input

        :param prefix:   prefix for out file name
        :param width:    optional, wrap
        :return:
        """
        if self.txt != self.txt_in:
            out_fn = self.fn.with_name(f'{prefix}_{self.fn.name}')
            with out_fn.open('w', encoding='utf-8') as f:
                if width > 0:
                    f.write(fill(self.txt, width))
                else:
                    # one giant blob
                    f.write(self.txt)
            logger.info(f'Written to {out_fn}, {len(self.txt):,d} characters.')
        else:
            logger.info('No changes, nothing writtten.')

    @staticmethod
    def _tex_patterns():
        """
        regular expressions to clean up TeX

        :return:
        """
        # regular expression flags
        fl = re.MULTILINE+re.IGNORECASE

        # patterns to apply, not order matters
        patterns = OrderedDict()
        # pattern, verbose replacement, debug replacement
        patterns['comment'] = [re.compile(r'<!--(.|\r|\n)*?-->', flags=fl), 'COMMENT', ' ']
        patterns['md_table'] = [re.compile(r'^\|?.+\|$', flags=fl), 'MD_TABLE', '']
        # macro definitions need special handling
        # convenient to handle one and multiline macro definitions separately
        patterns['macro_one'] = \
            [re.compile(r'^\\(?:provide|renew|new)command.*\}\n', flags=fl), ' MACRO ', '']
        patterns['macro_multi'] = \
            [re.compile(r'^\\(?:provide|renew|new)command(?:.|\n)+?\n\}\n', flags=fl), ' MACRO ', '']
        # \text command, which appears within equations and has dollars messes things up, handle separately
        patterns['text_comand'] = [re.compile(r'\\text\{[^}]+\}', flags=fl), r'TEXT_CMD', ' ']
        patterns['equation'] = \
            [re.compile(r'([ |\r(]{0,1})(\$\$?)[^$]+\2([-) .,}:;|\n])', flags=fl), r'\1 EQN \3', ' ']
        patterns['environment'] = \
            [re.compile(r'\\begin(\{[a-z*]+\})(.|\r|\n)*?\\end\1', flags=fl), r'ENV[\1]', ' ']
        patterns['tex_command'] = \
            [re.compile(r'\\([a-z]+)(\{[^}]+\})*', flags=fl), r'CMD[\1]', ' ']
        patterns['markdown'] = [re.compile('[*#]+', flags=fl), r'MARKDOWN', '']
        patterns['images'] = [re.compile(r'!\[.*?\]\(.*?\)({.*?})?'), '', '']
        patterns['link'] = [re.compile(r'\[.*?\]({.*?})'), '', '']
        patterns['newlines'] = [re.compile('\n+', flags=fl), ' ', ' ']
        patterns['dashes_spaces'] = [re.compile(' +|--+|\ ?- ', flags=fl), ' ', ' ']
        patterns['possessives'] = [re.compile("'s", flags=fl), '', '']
        patterns['punctuation'] = [re.compile('[?!.,:;(){}\[\]`="\'_%<>/$&]', flags=fl), ' ', ' ']
        patterns['reference'] = [re.compile('@([a-z]+)[0-9a-z]{2,5}', flags=fl), r'\1', r'\1']
        patterns['multi_spaces'] = [re.compile(r'(\s)\s+', flags=fl), r'\1', r'\1']
        return patterns

    def remove_yaml(self):
        """
        Strip out yaml at top of file
        This starts from scratch so it can be applied multiple times

        :return:
        """
        len_in = len(self.txt_in)
        stxt = re.split('^---$', self.txt_in, flags=re.MULTILINE)
        assert len(stxt) == 3
        self.txt = stxt[2]
        logger.info(f'reset string and removed yaml. Deleted {len_in - len(self.txt):,d} characters; '
                    f'current length = {len(self.txt):,d}.')

    def tex_cleanup(self, *, remove_yaml=True, verbose=True, watch='',
                    num_char=0, omit=None, test_string=''):
        """
        remove all TeX and markdown

        if verbose - show progress and optionally track the next num_char characters after
        an input watch expression.

        watch and num_char only apply in verbose mode.

        :param fn:           pathlib.Path object with input file
        :param remove_yaml:
        :param verbose:      provide more feedback
        :param watch:        optional: character string to watch for
        :param num_char:     number of characters beyond watch string to report (verbose only)
        :param omit:         names of patterns to omit from the standard list
        :param test_string:  run on test string rather than self.txt and turn debug on
        :return:             adjusted string with all replacements, total number of replacements,
                             dictionary with number of replacements by type
        """

        if test_string != '':
            verbose = True
            txt = test_string
            logger.info('test mode!')
        else:
            if remove_yaml:
                self.remove_yaml()
            # local var
            txt = self.txt

        logger.info(f'base input text len = {len(txt):,d}')

        # unescape HTML like &amp;
        txt = html.unescape(txt)
        logger.info(f'escaped html, len = {len(txt):,d}')

        # collect the patterns TODO handle other types of markup
        patterns = self._tex_patterns()
        if omit is not None:
            for k in omit:
                del patterns[k]
        logger.info(f'applying {len(patterns)} TeX patterns')
        logger.debug(patterns.keys())

        repl_dict = OrderedDict()
        test_dict = OrderedDict()
        last_test = ''
        total_repl = 0
        last_verbose_blob = ''
        if verbose:
            for k, v in patterns.items():
                pat, repl, nodb = v
                # special for debug
                if 0 and k in ('macro_one',  'macro_multi'):
                    logger.info('~'*60)
                    logger.info(f"{'<<'*10} {k} {'>>'*10}")
                    for _ in pat.findall(txt):
                        logger.info(f'CHUNK len={len(_):,d} found')
                        # if len(_) < 1000:
                        #     logger.info(_)
                        # else:
                        #     logger.info(_[:500], '.'*20, _[-500:])
                    logger.info('~'*60)
                if test_string != '':
                    repl = nodb
                txt, n = pat.subn(repl, txt)
                if test_string != '':
                    if txt != last_test:
                        last_test = txt
                        test_dict[k] = txt
                        logger.info(f'Key: {k:12s}    | {txt}')
                repl_dict[k] = n
                total_repl +=n
                if n and test_string == '':
                    logger.info(f'Key: {k:12s}    | {n:8,d} replacements | len={len(txt):12,d}')
                if watch != '':
                    i = txt.find(watch)
                    if num_char:
                        j = num_char
                    else:
                        j = txt[i+1:].find('\n')
                    if txt[i:i+j] != last_verbose_blob:
                        # only report changes to the watched string
                        logger.info('-'*80)
                        last_verbose_blob = txt[i:i+j]
                        logger.info(last_verbose_blob)
                        logger.info('-'*80)
        else:
            for k, v in patterns.items():
                pat, _, repl = v
                # replace with production string
                txt, n = pat.subn(repl, txt)
                repl_dict[k] = n
                total_repl +=n

        if test_string == '':
            self.txt = txt

        logger.info(f'completed length {len(txt):,d} after {total_repl:,d} substitutions.')
        return total_repl, repl_dict, test_dict

    def show_stop_words(self):
        print(fill(' '.join(sorted(self._stop_words)), 80))

    def frequency(self, stemmer='snowball'):
        """
        make frequency DataFrame

        stemmer = porter, lancaster (very coarse), snowball (best)

        Comparison on words Porter stems to gener

        word           | snowball     | lancaster
        general        | general      | gen
        generality     | general      | gen
        generalization | general      | gen
        generalize     | general      | gen
        generalized    | general      | gen
        generalizes    | general      | gen
        generally      | general      | gen
        generate       | genera**t**  | gen
        generated      | generat      | gen
        generates      | generat      | gen
        generating     | generat      | gen
        generation     | generat      | gen
        generator      | generat      | gen
        generic        | gener**ic**  | gen

        https://towardsdatascience.com/stemming-corpus-with-nltk-7a6a6d02d3e5

        :return:
        """
        if self._stop_words is None:
            self._stop_words = stopwords.words('english')
            # john words
            self._stop_words.extend(['section', 'ins', 'one', 'two', 'three', 'first','second','third',
                                     'let', 'may', 'must', 'since', 'however', 'therefore', 'see', 'show',
                                     'determine', 'new', 'called', 'call', 'right', 'left'])
            if stemmer == 'snowball':
                self._stemmer = SnowballStemmer('english')
            elif stemmer == 'porter':
                self._stemmer = PorterStemmer()
            else:
                self._stemmer = LancasterStemmer()

        # None (the default value) means split according to any whitespace,
        # and discard empty strings from the result (!)
        # then strip out numbers that mysteriously appear
        # compiling first is about twice as fast
        digit = re.compile('[0-9]')
        short_dash = re.compile('[a-z]-[a-z]')
        other_remove = re.compile('\\\\')
        all_words = [i for i in self.txt.lower().split() if digit.search(i) is None
                     and i!='-' and short_dash.match(i) is None and
                     other_remove.match(i) is None]
        # not i.replace('.', '', 1).isdigit()]
        # set_all_words= set(all_words).difference(set(self._stop_words))
        logger.info(f'split into {len(all_words):,d} words')

        df = pd.DataFrame({'word': all_words, 'frequency': 1})
        df['stem'] = [self._stemmer.stem(i) for i in df.word]
        df['ordinal'] = range(len(df))
        logger.info('added stems')
        df['stop'] = 0
        df = df.set_index('word')
        df.loc[set(self._stop_words).intersection(set(df.index)), 'stop'] = 1
        logger.info('added stopword indicator')

        df_words = df.reset_index(drop=False).\
            groupby(['stop', 'stem', 'word'])['frequency'].sum().\
            sort_index().to_frame()

        df_stem = df.groupby(['stop', 'stem'])['frequency'].\
            sum().sort_index().to_frame()

        self._sf = df_stem.loc[0]
        self._sf['len'] = self._sf.index.str.len()

        return df, df_words, df_stem

        # stem_freq.merge(df[['stem', 'stop']], on='stem')
        # logger.info('created stem freuqency dataframe')
        # dd = df[['stem', 'stop']].drop_duplicates()
        # bit = stem_freq.merge(dd, left_index=True, right_on='stem')
        # logger.info('created dd and bit')
        # # for the word cloud
        # self._sf = bit.query('stop==0')[['frequency']]
        # self._sf['len'] = self._sf.index.str.len()
        # # or stemmed:
        # # self._sf = stem_freq
        # # self._sf['len'] = stem_freq.index.str.len()
        # return df, stem_freq, dd, bit

    def sf(self, min_length=4):
        return self._sf.query(f'len >= {min_length}')['frequency']

    @staticmethod
    def read_bibtexfile(fn):
        """
        read a bibtex file and return as a data frame
        Note, this is surpisingly slow
        :param fn:
        :return:
        """
        import bibtexparser
        if not isinstance(fn, Path):
            fn = Path(fn)
        logger.warning('Reading bibtex file...can be slow.')
        with fn.open('r', encoding='utf-8') as bibtex_file:
            db = bibtexparser.load(bibtex_file)
        logger.warning('Bibtex file read.')
        df = pd.DataFrame(db.entries)
        df = df.set_index('ID')
        return df

    def cited_authors(self, df):
        """
        df includes column authors and index ID
        output dictionary id=>{authors} with names processed nicely

        the output of this function is suitable to input into a wordcloud

        :param df: from read_bibtexfile
        :return:
        """
        # pull out all citations
        ref_re = re.compile('@([a-z]+[0-9a-z]{2,5})', flags=re.IGNORECASE+re.MULTILINE)
        refs = list(ref_re.findall(self.txt))

        # tidy up names
        d = {ID: TextAnalyzer._parse(x) for ID, x in df.iterrows() }
        d = {k: v for k, v in d.items() if v is not None}

        all_authors = []
        for citation in refs:
            try:
                all_authors.append(d[citation])
            except KeyError:
                logger.error(citation)
        flat_all_authors = [i for j in all_authors for i in j]

        # final tidying up---very ad hoc
        def gsr(w):
            if w.find('Cherny') >= 0:
                return 'Alexander S. Cherny'
            if w.find('Tsanakas') >= 0:
                return 'Andreas Tsanakas'
            if w == 'David J. Cummins':
                return 'J. David Cummins'
            if w.find('Jouini') >= 0:
                return 'Elyès Jouini'
            if w.find('Grundl') >= 0:
                return "Helmut Gründl"
            if w.find('S. Wang') >= 0:
                return 'Shaun Wang'
            if w.find('Young') >= 0:
                return 'Virginia R. Young'
            if w.find('Mildenhall') >= 0:
                return 'Stephen J. Mildenhall'
            w = w.replace('Follmer', 'Föllmer')
            return w
        dff = pd.DataFrame({'word': flat_all_authors, 'freq': 1})
        dff['new'] = list(map(gsr, dff.word))
        g = dff.groupby('new').freq.count().sort_values(ascending=False)
        return g

    @staticmethod
    def _parse(x):
        """
        clean up a name: initials, tex to HTML for accents etc.
        x has a author attribute

        :param nm:
        :return:
        """
        try:
            a = x.author.split(" and ")
            a = [TextAnalyzer._clean_name(f'{j[1]} {j[0]}') if len(j)>1 else TextAnalyzer._clean_name(j[0])
                 for j in [i.split(', ') for i in a]]
            return a
        except AttributeError:
            logger.error(f'ERROR: {x.name} {x.author}')
            return None

    @staticmethod
    def _clean_name(n):
        """
        tex to html to unicode conversion
        """
        def f(m):
            x = "&" + m[2] + {"'":'acute;', '^':'circ;', '`':'grave;', '"': 'uml;'}[m[1]]
            return html.unescape(x)

        reg = re.compile(r"\{\\(['`^\"])\{([a-zA-Z])\}\}")
        n1, i = reg.subn(f, n)
        n1 = re.sub(r'\{\\o\}', 'ø', n1)
        n1 = re.sub(r'[{}]', '', n1)
        n1 = TextAnalyzer._capitalize_initials(n1)
        return n1

    @staticmethod
    def _capitalize_initials(n):
        def f(m):
            return '. '.join(list(m[1])) + f'. {m[2]}'
        return re.sub(r'([A-Z]+) (.*)\b', f, n)


class WordClouder(WordCloud):

    def __init__(self, colormap='cividis',  width=1200,  height=800, scale=1, max_words=2000,
                 max_font_size=120, min_font_size=4, relative_scaling='auto', font='Times New Roman',
                 background='aliceblue',  prefer_horizontal=0.8, contour_width=1,
                 colorer="", mask=None, dpi=100):
        """
        WordCloud wrapper

        There are issues with name collisions between WordClouder and the parent WordCloud object

        By and large, WordClouder tries to use properties with similar but slighlty different names

            easy_font
            colorer     --> colorer function
            mask_image  --> mask...this has too many options and is set with set_mask_image

        """
        self._font_name = ''
        self._easy_font_name = 'times'
        self._fm = None
        self._mask_image_fn = ''
        self._colorer_fn = ''
        self._cloud = None
        self._word_freq_iterable = None
        self._min_word_length = 4
        self.dpi = dpi
        self.f = None

        super().__init__(
                        width=width,
                        height=height,
                        margin=2,
                        prefer_horizontal=prefer_horizontal,
                        scale=scale,
                        max_words=max_words,
                        min_font_size=min_font_size,
                        max_font_size=max_font_size,
                        font_step=1,
                        mode='RGB',
                        relative_scaling=relative_scaling,
                        colormap=colormap,
                        contour_width=contour_width,
                        contour_color='black'
        )

        # properties that set through to the undelying WordCloud object...set after
        # it is initialized
        self.background = background
        self.font = font
        self.mask = mask
        self.colorer = colorer
        self._word_dict = None
        self.color_function = None

        # self.colormap = colormap
        # self.width = width
        # self.height = height
        # self.scale = scale
        # self.max_words = max_words
        # self.max_font_size = max_font_size
        # self.min_font_size = min_font_size
        # self.relative_scaling = relative_scaling
        # self.prefer_horizontal = prefer_horizontal

    @property
    def word_freq(self):
        if self._word_freq_iterable is None:
            raise ValueError('Must set word frequecy iterable before generating word cloud')
        return self._word_freq_iterable[self._word_freq_iterable >= self._min_word_length]

    @word_freq.setter
    def word_freq(self, value):
        try:
            v1, min_word_length = value
        except:
            v1 = value
            min_word_length = 4
        self._word_freq_iterable = v1.copy()
        self._min_word_length = min_word_length

    def generate_cloud(self):
        logger.info('Generating cloud')
        self._cloud = self.generate_from_frequencies(self.word_freq)

    def recolor_cloud(self):
        if self._cloud is None:
            self.generate_cloud()
        logger.info('Recoloring cloud')
        return self._cloud.recolor(color_func=self.color_function, colormap=self.colormap)

    @property
    def cloud(self):
        if self._cloud is None:
            self.generate_cloud()
        if self.color_function is not None:
            self.recolor_cloud()
        return self._cloud

    def show(self, ax=None, usefig=False):
        """
        Show the wordcloud. Optionally pass in an axis.
        :param ax:
        :param usefig: force creation of new figure
        :return:
        """
        if ax is None and usefig is False:
            s = self.to_svg()
            display(SVG(s))
            return

        if ax is None and usefig:
            self.f = plt.figure(figsize=(self.width/self.dpi, self.height/self.dpi))
            ax = self.f.add_axes((0,0,1,1))
        ax.imshow(self.cloud)
        ax.axis('off')

    def save(self, fn):
        """
        fn = filename with extension
        Preferred: fn is .svg type

        If not svg then tries to save the latest figure

        :param fn:
        :return:
        """

        p = Path(fn)
        if p.suffix.lower() == '.svg':
            s = self.to_svg()
            with p.open('w', encoding='utf-8') as f:
                f.write(s)
            logger.info(f'Written {len(s)} characters to svg file.')
        else:
            assert self.f is not None
            self.f.set_facecolor('white')
            self.f.savefig(fn, dpi=self.dpi)
            logger.info('Saved from figure. Consider using .svg file for smaller, higher quality file.')

    @property
    def font(self):
        return self._font_name

    @font.setter
    def font(self, font_name):
        """
        Set the font by name, and find the relevant ttf file
        :param font_name:
        :return:
        """
        # fm will find any installed windows font by name
        if self._fm is None:
            self._fm = mpl.font_manager.FontManager()
        self.font_path = self._fm.findfont(font_name)
        self._font_name = font_name

    @property
    def background(self):
        """
        return background color code

        :return:
        """
        return self.background_color

    @background.setter
    def background(self, value):
        """
        set background color by name

        :param value:
        :return:
        """
        # actually wants a rgb string...else to_svg does not work properly
        self.background_color = 'rgb({}, {}, {})'.format(
            *tuple(int(256 * i) for i in mpl.colors.to_rgb(mpl.colors.CSS4_COLORS[value])))

    @property
    def mask_image(self):
        return self._mask_image_fn

    def set_mask_image(self, fn, channel=0, threshold=100, *, invert=False, transpose=False):
        """
        channel = 0,1,2 RGB used to make mask or 'avg' (NYI)

        transpose = rotate image

        invert = use threshold <= rather than > threshold

        """
        if self._mask_image_fn == fn:
            logger.info(f'{fn} is current mask...no change required.')
            return
        self._mask_image_fn == fn
        if fn == '':
            self.mask = None
        else:
            img = self._imread(fn)
            if type(channel) == str and channel in ('all', 'mean', 'avg', 'average'):
                temp = np.mean(img, axis=2)
            else:
                temp = img[:, :, channel]
            # All white (#FF or #FFFFFF) entries will be considerd “masked out”
            # while other entries will be free to draw on.
            if invert:
                idx = temp <= threshold
            else:
                idx = temp > threshold
            temp[~idx] = 255
            if transpose:
                self.mask = temp.T
            else:
                self.mask = temp
            # invalidate current cloud
            self._cloud = None

    def show_mask(self, ax=None, w=6, cmap='gray'):
        if self.mask is None:
            return
        mw, mh = self.mask.shape
        h = int(mh / mw * w)
        if ax is None:
            f = plt.figure(figsize=(w, h))
            ax = f.add_axes((0,0,1,1))
        ax.imshow(self.mask, cmap=cmap)
        ax.axis('off')

    @property
    def colorer(self):
        return self._colorer_fn

    @colorer.setter
    def colorer(self, fn):
        """
        create a color function from image with filename fn
        must exist
        """

        if self._colorer_fn == fn:
            return
        self._colorer_fn = fn
        if fn is not None and fn != '':
            img = self._imread(fn)
            self.color_function = ImageColorGenerator(img)
            self.recolor_cloud()
        elif fn is None:
            self.color_function = None

    @property
    def easy_font(self):
        return self._easy_font_name

    @easy_font.setter
    def easy_font(self, value):
        """
        value = script handwriting classic smc ...

        bradely     'Bradley Hand ITC'          not great
        comic       'Comic Sans MS'             good
        elegant     'Trebuchet MS'
        hand        'Lucida Handwriting'        good
        mono        'Ubuntu Mono'
        ravie       'Ravie'                     whacky
        script      'Brush Script MT'

        """
        value = value.lower()
        if value == self._easy_font_name:
            return
        if value == 'bradley':
            # doesn't work great
            self.font = 'Bradley Hand ITC'
        elif value == 'bahn':
            self.font = 'Bahnschrift'
        elif value == 'chiller':
            self.font = 'chiller'
        elif value == 'comic':
            self.font = 'Comic Sans MS'
        elif value == 'elegant':
            self.font = 'Trebuchet MS'
        elif value == 'garamond':
            self.font = 'Garamond'
        elif value == 'handwriting' or value == 'hand':
            self.font = 'Lucida Handwriting'
        elif value == 'heiti':
            self.font = 'Adobe Heiti Std'
        elif value == 'mono':
            self.font = 'Ubuntu Mono'
        elif value == 'ravie':
            self.font = 'Ravie'
        elif value == 'script':
            self.font = 'Brush Script MT'
        elif value == 'times':
            self.font = 'Times New Roman'
        elif value == 'xkcd':
            self.font = 'xkcd'
        else:
            raise ValueError(f'Unknown font {value} passed to easy_font')
        # new font...invalidate
        self._cloud = None

    def _imread(self, fn, preserve_aspect=True):
        """
        read fn image and reshape to be at least w x h, preserving aspect as requested
        """
        img = Image.open(fn)
        iw, ih = img.size
        w = self.width
        h = self.height
        if preserve_aspect is True:
            wscale = w / iw
            hscale = h / ih
            if wscale >= hscale:
                h = int(wscale * ih)
            else:
                w = int(hscale * iw)
        img_rs = img.resize((w, h), Image.BICUBIC)
        return np.asarray(img_rs)



# def example(az, el, alpha, shade):
#     """
#     example using tri_surf
#
#     :param az:
#     :param el:
#     :param alpha:
#     :param shade:
#     :return:
#     """
#     f = plt.figure(figsize=(10, 10))
#     ax = f.add_axes((0,0,1,1), projection='3d')
#
#     r = 1
#     polar = np.array([0, 2 * np.pi / 3, 2 * np.pi / 3, 2 * np.pi / 3])
#     azimith = np.array([0, np.pi / 2, 7 * np.pi / 6, 11 * np.pi / 6])
#
#     x = r * np.sin(polar) * np.cos(azimith)
#     y = r * np.sin(polar) * np.sin(azimith)
#     z = r * np.cos(polar)
#
#     n = len(x)
#     norm = mpl.colors.Normalize(0, n, clip=False)
#     cmappable = mpl.cm.ScalarMappable(norm=norm, cmap='hsv')
#     mapper = cmappable.to_rgba
#
#     sc = 1.1
#     for xx, yy, zz, tt in zip(x,y,z,['comments', 'crossposts', 'ups', 'downs']):
#         ha = 'left' if tt == 'crossposts' else 'center'
#         ax.text(sc*xx, sc*yy, sc*zz, tt, fontsize=12, ha=ha, va='bottom')
#         ax.plot([0, xx], [0, yy], [0, zz], lw=.5, c='k')
#
#     for s in np.linspace(0,1,22)[1:-1]:
#         ax.scatter(s*x, s*y, s*z, marker='o', c=[mapper(s*n)]*n)
#         # print(s, s*x, mapper(s*n))
#
#     ax.set(xlim=[min(x), max(x)], ylim=[min(y), max(y)], zlim=[min(z), .5], #max(z)],
#           xlabel='x', ylabel='y', zlabel='z')
#
#     for i, j in zip((0,0,0,1,1,2), (1,2,3,2,3,3)):
#         # ax.plot_trisurf([1e-5, x[i], x[j]], [0, y[i], y[j]], [0, z[i], z[j]], color='k', alpha=.2)
#         ax.plot_trisurf([1e-5, x[i], x[j]], [0, y[i], y[j]], [0, z[i], z[j]], alpha=alpha, color=mapper((i+j)/5),
#             shade=shade, lightsource=mpl.colors.LightSource(az, el))
#
#
#     ax.set_axis_off()
#
#     # ax.xaxis.set_major_locator(plt.NullLocator())
#     # ax.yaxis.set_major_locator(plt.NullLocator())
#     # ax.zaxis.set_major_locator(plt.NullLocator())
#     # ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
#
#     # ax.spines['right'].set_visible(False)
#     # ax.spines['bottom'].set_visible(False)
#     # ax.spines['left'].set_visible(False)
#
#     # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#
#     # ax.patch.set_facecolor(self.figure_bg_color)
#
#
#     # ax.legend()

import copy

## ported
# def meta_plot_triangle(self, ax, quad, logger, n=20):
#     """
#     diamond version of meta_plot_square -> doesn't work because lines overlap...
#     hence...turned into meta_plot_triangle that just shows up/down/comments and
#     not crossposts

#     :param ax:
#     :param quad:
#     :param n_expected: expected number of entries per side of square usually 20, was n "typical length"
#     :return:
#     """

#     quad = copy.deepcopy(quad)
#     del quad['crossposts']

#     theta1 = 11 * np.pi / 6
#     theta2 = 7 * np.pi / 6
#     # diamond version
#     coords = {
#         #              label beg            label end                                   ha
#         'comments':   [np.array((0., 0.)),  np.array((0., 1.)),                         'center'],
#         'ups':        [np.array((0., 0.)),  np.array((np.cos(theta2), np.sin(theta1))), 'left' ],
#         'downs':      [np.array((0., 0.)),  np.array((np.cos(theta1), np.sin(theta2))), 'right' ]
#     }

#     # this is usually 3n
#     n_actual = sum([len(v) for v in quad.values()])
#     if n_actual != 3 * n:
#         logger.info(f'Square plot has {n_actual} < {3 * n} entries!')

#     bigd = {k: dict() for k in coords.keys()}

#     vall = set()
#     for k, v in quad.items():
#         vall = vall.union(set(v.index))

#     connection_counter = dict.fromkeys(vall, 0)

#     # determines coordinates, but doesn't write anything
#     for k, v in quad.items():
#         beg, end, ha = copy.deepcopy(coords[k])

#         logger.info(f'n={n}, len={len(v.index)}')

#         coords_x = np.linspace(beg[0], end[0], n + 2)
#         coords_y = np.linspace(beg[1], end[1], n + 2)
#         for i, t in enumerate(v[::-1].index):
#             bigd[k][t] = np.array([coords_x[i + 1], coords_y[i + 1]])

#     # determine connection counts draw and count lines
#     for k, v in quad.items():
#         for ok, ov in quad.items():
#             if k > ok:
#                 s = set(v.index).intersection(set(ov.index))
#                 for t in s:
#                     connection_counter[t] += 1

#     # WTF?
#     # remap counts 1 line is counted once, 2 lines has 3 nodes, 3 lines has 6 counted
#     for k, v in connection_counter.items():
#         connection_counter[k] = {0: 0, 1: 1, 3: 2, 6: 3}[v]

#     # now we can determine the colors - another loop
#     n1 = len(set([i for i, v in connection_counter.items() if v == 1]))
#     n2 = len(set([i for i, v in connection_counter.items() if v == 2]))
#     n3 = len(set([i for i, v in connection_counter.items() if v == 3]))
#     n0 = n_actual - 2 * n1 - 3 * n2 - 4 * n3
#     n0a = len(set([i for i, v in connection_counter.items() if v == 0]))
#     logger.info(f'n0={n0a}, vs computed n0={n0}')
#     if n0a + 2 * n1 + 3 * n2 + 4 * n3 != n_actual:
#         logger.error('Unexpected math making the square...')

#     # colormaper
#     node_colors = {}
#     logger.info(f'setting up with {n1 + n2 + n3} colored nodes')
#     norm = mpl.colors.Normalize(0, n1 + n2 + n3, clip=True)
#     cmappable = mpl.cm.ScalarMappable(norm=norm, cmap=self.plot_colormap_name)
#     mapper = cmappable.to_rgba
#     node_number = 0

#     # everything is colored by the end of downs, need to force the right order
#     for k in ['ups', 'comments']:
#         v = quad[k]
#         for t in v[::-1].index:
#             if connection_counter[t] > 0 and t not in node_colors:
#                 node_colors[t] = mapper(node_number)
#                 node_number += 1

#     # now know colors, can draw lines
#     for k, v in quad.items():
#         for ok, ov in quad.items():
#             if k > ok:
#                 s = set(v.index).intersection(set(ov.index))
#                 for t in s:
#                     ff = bigd[k][t]
#                     tt = bigd[ok][t]
#                     c = node_colors[t]
#                     ax.plot([ff[0], tt[0]], [ff[1], tt[1]], lw=2, alpha=1, c=c)

#     # write labels and markers
#     marker = 'o'
#     for k, v in quad.items():
#         beg, end, ha = copy.deepcopy(coords[k])
#         coords_x = np.linspace(beg[0], end[0], n + 2)
#         coords_y = np.linspace(beg[1], end[1], n + 2)
#         for i, t in enumerate(v[::-1].index):
#             if t in node_colors:
#                 col = node_colors[t]
#             else:
#                 col = 'r'
#             ax.plot([coords_x[i + 1]], [coords_y[i + 1]], marker=marker, c=col, ms=5)

#     # how many common elements
#     s = set([i for i, v in connection_counter.items() if v > 0])
#     common = {}
#     for k, v in quad.items():
#         i = set(v.index).intersection(s)
#         common[k] = len(i)

#     self.meta_meta['square'] = dict(n0=n0, n1=n1, n2=n2, n3=n3, n80=n_actual,
#                                     common=common,
#                                     ups=len(quad['ups']),
#                                     downs=len(quad['downs']),
#                                     comments=len(quad['comments']))

#     # lines in each direction
#     sc = 1.05
#     for k, v in coords.items():
#         beg, end, ha = v
#         ax.plot([beg[0], end[0]], [beg[1], end[1]], lw=.5, c='k')
#         if k != 'comments':
#             beg *= sc
#             end *= sc
#         ax.text(*end, f'{k.title()} ({len(quad[k]) - common[k]})', fontsize=12, ha=ha, va='bottom')

#     # sz controls the size of the fig -s to s
#     sz = 1
#     dy = 0.03
#     base_x = -sz + 0.01
#     base_y = -0.8*sz + 3 * dy
#     ax.text(base_x, base_y + dy, f'{n_actual:d} nodes on triad',
#             ha='left', va='bottom', fontsize='large', c='k')
#     ax.text(base_x, base_y, f'{n0:d} orphans appear on only one branch',
#             ha='left', va='bottom', fontsize='large', c='r')
#     ax.text(base_x, base_y - dy, f'{n1:d} appear on two',
#             ha='left', va='bottom', fontsize='large')
#     ax.text(base_x, base_y - 2 * dy, f'{n2:d} appear on three',
#             ha='left', va='bottom', fontsize='large')
#     ax.text(base_x, base_y - 3 * dy, f'{n0 + n1 + n2 + n3:d} distinct items in all',
#             ha='left', va='bottom', fontsize='large')

#     # turn off all frills
#     ax.set(xlim=[-sz, sz], ylim=[-0.8*sz, sz], aspect=1)
#     ax.patch.set_facecolor(self.figure_bg_color)
#     self.standardize_plot(ax, hide_axes=True)


# def _make_coords(beg, end, n):
#     """
#     make coordinates, b=(bx, by, bz) etc. and n points

#     """
#     coords_x = np.linspace(beg[0], end[0], n + 2)
#     coords_y = np.linspace(beg[1], end[1], n + 2)
#     coords_z = np.linspace(beg[2], end[2], n + 2)
#     return coords_x, coords_y, coords_z

# def meta_plot_tetrahedron(self, quad, logger, n=20, alpha=.1, elev=20, azim=300, rot=0):
#     """
#     tetrahedron version of meta_plot_square
#     text graphic showing which posts are common to top up/down votes, comments and cross post
#     n:  number of elements in each list can be determined from quad

#     DRAW A PICTURE to see what is going on with coords!

#     Update: posts can have the same names, so this works with keys throughout

#     :param ax:
#     :param quad:
#     :param n_expected: expected number of entries per side of square usually 20, was n "typical length"
#     :return:
#     """

#     f = plt.figure(figsize=(10, 10))
#     ax = f.add_axes((0,0,1,1), projection='3d')
#     ax0 = f.add_axes((0,0,1,1))

#     # avoid weirdness
#     quad = copy.deepcopy(quad)

#     # vertex points
#     r = 1
#     polar = np.array([0, 2 * np.pi / 3, 2 * np.pi / 3, 2 * np.pi / 3])
#     azimith = np.array([0, np.pi / 2, 7 * np.pi / 6, 11 * np.pi / 6])
#     if rot != 0:
#         theta = rot / 360 * 2 * np.pi
#         rot = np.array([0, theta, theta, theta])
#         azimith += rot

#     x = r * np.sin(polar) * np.cos(azimith)
#     y = r * np.sin(polar) * np.sin(azimith)
#     z = r * np.cos(polar)
#     origin = np.array((0., 0., 0.))

#     if alpha > 0:
#         for i, j in zip((0,0,0,1,1,2), (1,2,3,2,3,3)):
#             ax.plot_trisurf([1e-5, x[i], x[j]], [0, y[i], y[j]], [0, z[i], z[j]], color='w', alpha=alpha)


#     # tetrahedron version
#     coords = {
#         #              label beg            label end            kind     rot     ha
#         'comments':   [origin,   np.array((x[0], y[0], z[0])),   'x',      45,    'left'  ],
#         'crossposts': [origin,   np.array((x[1], y[1], z[1])),   'y',     -45,    'left'  ],
#         'ups':        [origin,   np.array((x[2], y[2], z[2])),   'y',     -45,    'right' ],
#         'downs':      [origin,   np.array((x[3], y[3], z[3])),   'x',      45,    'right' ]
#     }

#     # this is usually 4n
#     n_actual = sum([len(v) for v in quad.values()])
#     if n_actual != 4 * n:
#         logger.info(f'Square plot has {n_actual} < {4 * n} entries!')


#     bigd = {k: dict() for k in coords.keys()}

#     vall = set()
#     for k, v in quad.items():
#         vall = vall.union(set(v.index))

#     connection_counter = dict.fromkeys(vall, 0)

#     # determines coordinates, but doesn't write anything
#     for k, v in quad.items():
#         beg, end, kind, rot, ha = copy.deepcopy(coords[k])
#         coords_x, coords_y, coords_z = _make_coords(beg, end, n)
#         for i, t in enumerate(v[::-1].index):
#             bigd[k][t] = np.array([coords_x[i + 1], coords_y[i + 1], coords_z[i + 1]])

#     # determine connection counts draw and count lines
#     for k, v in quad.items():
#         for ok, ov in quad.items():
#             if k > ok:
#                 s = set(v.index).intersection(set(ov.index))
#                 for t in s:
#                     connection_counter[t] += 1

#     # remap counts 1 line is counted once, 2 lines has 3 nodes, 3 lines has 6 counted
#     for k, v in connection_counter.items():
#         connection_counter[k] = {0: 0, 1: 1, 3: 2, 6: 3}[v]

#     # now we can determine the colors - another loop
#     n1 = len(set([i for i, v in connection_counter.items() if v == 1]))
#     n2 = len(set([i for i, v in connection_counter.items() if v == 2]))
#     n3 = len(set([i for i, v in connection_counter.items() if v == 3]))
#     n0 = n_actual - 2 * n1 - 3 * n2 - 4 * n3
#     n0a = len(set([i for i, v in connection_counter.items() if v == 0]))
#     logger.info(f'n0={n0a}, vs computed n0={n0}')
#     if n0a + 2 * n1 + 3 * n2 + 4 * n3 != n_actual:
#         logger.error('Unexpected math making the square...')

#     # colormaper
#     node_colors = {}
#     # norm1 = mpl.colors.Normalize(-np.pi, np.pi, clip=True)
#     logger.info(f'setting up with {n1 + n2 + n3} colored nodes')
#     norm = mpl.colors.Normalize(0, n1 + n2 + n3, clip=True)
#     cmappable = mpl.cm.ScalarMappable(norm=norm, cmap=self.plot_colormap_name)
#     mapper = cmappable.to_rgba
#     node_number = 0

#     # everything is colored by the end of downs, need to force the right order
#     for k in ['comments', 'ups', 'downs']:
#         v = quad[k]
#         for t in v[::-1].index:
#             if connection_counter[t] > 0 and t not in node_colors:
#                 # logger.info(f'setting node {t} (# {node_number}) to color {mapper(node_number)}')
#                 node_colors[t] = mapper(node_number)
#                 node_number += 1

#     # now know colors, can draw the connecting lines
#     for k, v in quad.items():
#         for ok, ov in quad.items():
#             if k > ok:
#                 s = set(v.index).intersection(set(ov.index))
#                 for t in s:
#                     ff = bigd[k][t]
#                     tt = bigd[ok][t]
#                     c = node_colors[t]
#                     ax.plot([ff[0], tt[0]], [ff[1], tt[1]], [ff[2], tt[2]], lw=1, alpha=1, c=c)

#     # write labels and markers
#     va = 'center_baseline'
#     marker = 'o'
#     for k, v in quad.items():
#         beg, end, kind, rot, ha = copy.deepcopy(coords[k])
#         coords_x, coords_y, coords_z = _make_coords(beg, end, n)
#         for i, t in enumerate(v[::-1].index):
#             if t in node_colors:
#                 col = node_colors[t]
#             else:
#                 col = 'r'
#             ax.scatter([coords_x[i + 1]], [coords_y[i + 1]], [coords_z[i + 1]], marker=marker, color=col, s=10)

#     ax.view_init(elev=elev, azim=azim)
#     # ax.set(xlim=[min(x), max(x)], ylim=[min(y), max(y)], zlim=[min(z), .5], #max(z)],
#     #       xlabel='x', ylabel='y', zlabel='z')
#     s = 0.8
#     ax.set(xlim=[-s, s], ylim=[-s, s], zlim=[-s, s],
#           xlabel='x', ylabel='y', zlabel='z')
#     ax.patch.set_facecolor(self.figure_bg_color)
#     ax.set_axis_off()
#     # better initial view

#     # how many common elements
#     s = set([i for i, v in connection_counter.items() if v > 0])
#     common = {}
#     for k, v in quad.items():
#         i = set(v.index).intersection(s)
#         common[k] = len(i)

#     self.meta_meta['square'] = dict(n0=n0, n1=n1, n2=n2, n3=n3, n80=n_actual,
#                                     common=common,
#                                     ups=len(quad['ups']),
#                                     downs=len(quad['downs']),
#                                     comments=len(quad['comments']),
#                                     crossposts=len(quad['crossposts']))

#     # lines in each direction
#     sc = 1.1
#     for xx, yy, zz, k in zip(x, y, z, coords.keys()):
#         ha = 'left' if k == 'crossposts' else 'center'
#         ax.text(sc*xx, sc*yy, sc*zz, f'{k.title()} ({len(quad[k]) - common[k]})', fontsize=12, ha=ha, va='bottom')
#         ax.plot([0, xx], [0, yy], [0, zz], lw=.5, c='k')


#     dy = 0.02
#     base_x = 0.025
#     base_y = 4 * dy
#     ax0.text(base_x, base_y + dy, f'{n_actual:d} nodes on tetrahedron',
#             ha='left', va='bottom', fontsize='large', c='k')
#     ax0.text(base_x, base_y, f'{n0:d} orphans appear on only one branch',
#             ha='left', va='bottom', fontsize='large', c='r')
#     ax0.text(base_x, base_y - dy, f'{n1:d} appear on two',
#             ha='left', va='bottom', fontsize='large')
#     ax0.text(base_x, base_y - 2 * dy, f'{n2:d} appear on three',
#             ha='left', va='bottom', fontsize='large')
#     ax0.text(base_x, base_y - 3 * dy, f'{n3:d} appear on all four branches',
#             ha='left', va='bottom', fontsize='large', fontweight='bold')
#     ax0.text(base_x, base_y - 4 * dy, f'{n0 + n1 + n2 + n3:d} distinct items in all',
#             ha='left', va='bottom', fontsize='large')
#     ax0.set_axis_off()
#     ax0.patch.set_facecolor((0,0,0,0))

#     def animate(i):
#         ax.view_init(elev=30, azim=i*3)
#         return ax.artists

#     return f, animate


# def make_animation(fn, f, animate):
#     """
#     # Animated square

#     f, animate = hack.meta_plot_tetrahedron(self, quad, logger, alpha=0, azim=30)
#     make_animation('c:\\temp\\fn.gif', f, animate)

#     """
#     anim = mpl.animation.FuncAnimation(f, animate, frames=120, blit=True)
#     writergif = mpl.animation.PillowWriter(fps=20)
#     anim.save(fn, writer=writergif)
