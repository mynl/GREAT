"""
PresentationManager: persist exhibits and tables and make overall presentation

Builds off doc_maker

    * Only makes presentations
    * Appendix and TOC
    * Removes unused options
    * Integrated into magics


Usage example

```
    test_df = pd.DataFrame(dict(x=range(100), y=range(100,200)))
    f, ax = plt.subplots(1,3)
    f = test_df.plot(ax = ax[2])

    doc = DocMaker.DocMaker('notes/test1.md')
    doc.section('Here is the main title ')
    doc.subsection('A dataframe section')
    doc.table(test_df.head(10), 'df1', caption='Here is a dataframe')
    doc.subsection('A Figure section')
    doc.figure(f, 'x_y', caption='Here is a 1x3 figure ', size="height=30%")
    doc.text('Some closing thoughts. and a  bunch more stuff.')

    doc.write_markdown()

    doc.process()
```


v 1.0 Dec 2020 created PresentationManager from DocMaker

"""

import os
import sys
from io import StringIO
from pathlib import Path
from matplotlib.pyplot import Figure
import pandas as pd
from .markdown_make import markdown_make_main
from .utils import logger
import re
import numpy as np
import unicodedata
from pandas.io.formats.format import EngFormatter
import json
from collections import OrderedDict
# import subprocess
# from platform import platform
from IPython.display import Markdown, display
from IPython.core.magic import Magics, magics_class, line_magic, cell_magic, line_cell_magic
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
from IPython import get_ipython
from titlecase import titlecase
import pypandoc
from pprint import PrettyPrinter
import datetime
import base64
import hashlib

# import sys
# import re
# import subprocess
# import pathlib
# import struct
# import numpy as np
# import glob
# import string
# import unicodedata


# Size = dict(TINY = T = 300
#     VSMALL = VS = 400
#     SMALL = S = 500
#     MEDIUM = M = 600
#     LARGE = L = 700
#     XLARGE = XL = 800


class PresentationManager(object):
    """
    Class level doc string...

    """

#     __yaml__ = """---
# title: "{title}"
# subtitle: "{subtitle}"
# author: Stephen J. Mildenhall
# date: "{created}"
# fontsize: {font_size}pt
# outertheme : metropolis
# innertheme : metropolis
# fonttheme  : structurebold
# colortheme : orchid
# institute: \\convexmark
# classoption: t
# numbersection: true
# toc: false
# filter: pandoc-citeproc
# bibliography: /S/TELOS/biblio/library.bib
# csl: /S/TELOS/biblio/journal-of-finance.csl
# link-citations: true
# header-includes:
#     - \\input{{general2021.tex}}
#     - \\geometry{{paper=legalpaper, landscape, mag=2000, truedimen}}
# cla: --top-level-division=section
# cla: --slide-level=2
# cla: --standalone
# cla: -f markdown+smart+yaml_metadata_block+citations{pdf_engine}
# cla: -o pdf
# cla: --filter pandoc-citeproc
# cla: -t beamer
# debug: true
# ---
#
# """

    def __init__(self, config_file):
        """
        builds_id -> specification of portfolio, key, base_dir etc.
        tops_id = None -> make contents etc.


        Originally:
        title, subtitle, key, base_dir, *,
                 top_name='', top_value='',
                 file_name='',
                 fig_format='pdf', output_style='with_table',
                 default_float_fmt=None, tidy=False,
                 tacit_override=False,
                 pdf_engine='')

        file_name of output, including extension

                 pdf_engine='cla: --pdf-engine=lualatex'):

        base_dir should be notes (but you need to state that to avoid screw ups)

        key = short vignette name prepended to name of all tables and image files; goes with title... uber level
        file_name can be a/b/c/file.md
        all intermediate dirs created
        images in a/b/c/img

        tidy: search through the subfolders of the base output and delete all files key-*.*

        depending on how you name the key the existing pdf file may be deleted.
        key = vig-{vig}-something and file = key.md then it will NOT be because only key-*.* files
        are deleted. This is the recommended approach and is the default if key is ''.

        if tacit_override then override tacit command and show anyway (tables and blobs only)


        title
        key  becomes key-top_value
        top_name  Module, Part etc.
        top_value  A, B  or I, II or whatever
        subtitle ==> top_name top_value subtitle
        filename ==> key top_value (upper case)  [can ne over ridden]


        top_name: Part | Module etc.
        top_value: A, B, I, II etc.

        :param file_name:
        :param base_dir: where to store output, default ./notes
        :param key: (becomes key-top_value, prepended to file names to cluster them. Should be unique to the doc.
        :param title:
        :param fig_format:
        :param output_style: caption (caption in main doc just numbers in include),
                    in-line (all in main doc) or with_table (all in include file)
        :param default_float_fmt:

        """

        if config_file is not None:
            p = Path(config_file)
            if p.exists():
                self.config_all = json.load(Path(config_file).open('r', encoding='utf-8'))
                self.config_file = config_file
            else:
                raise ValueError(f'Error: file {config_file} does not exist.')
        else:
            self.config_all = None
            self.config_file = None

        # get all the other variables defined in init:
        self.builds_id = None
        self.temp_file = None
        # other class variables related to config file
        self.config = None
        self.tops_id = ''
        self.raw_top_value = ''
        self.option_id = ''

        # read in builds level global variables
        self.title = ''
        self.subtitle = ''

        # files are named key-top_value-...
        self.key = ''
        self.top_name = ''
        self.top_value = ''

        # file specific, set manually
        self.base_dir = None
        self.file_name = ''
        self.file = None
        self.out_dir = None
        self.figure_dir = None
        self.table_dir = None

        # admin
        self.fig_format = 'pdf'
        self.output_style = 'with_table'
        self.pdf_engine = ''
        self.default_float_fmt = PresentationManager.default_float_format

        self.unit = 1000
        self.capital_standard = 0.996

        # unless over-ridden, start counting at 1, appendices count negative
        self.section = 0
        self.appendix = 0

        # output
        self.toc = []
        self._slides = set()
        self.sios = None

        # starts in base self.config: active and not in debug mode
        self._tacit_override = False
        self._active = True
        self._debug = False
        self._debug_state = False
        # avoid using dm because updated gets in a loop on the dates...
        self.debug_dm_file = None

    def activate(self, title, subtitle, key, top_name, raw_top_value, top_value, base_dir, filestem=''):
        """
        set all the options to work independently of a config file

        key = pirc
        top_value = Ia  (drives file name and fig/table saved name)
        raw_top_value I excludes options (for section names and TOC)
        top_name = chapter, module, etc.

        for files the name is {key}-{top_value}-...
        the main file is callled {key}-{top_value}.md unless different filestem provided

        :return:
        """
        self.title = title
        self.subtitle = subtitle
        self.key = key
        self.top_name = top_name
        self.top_value = top_value
        self.base_dir = Path(base_dir)
        self.file_name = f'{key}-{top_value}.md' if filestem == '' else f'{filestem}.md'
        self._complete_setup()

    def activate_build(self, builds_id):
        """
        Spec for portfolio etc.

        :param builds_id:
        :return:
        """
        self.builds_id = builds_id

        # specification of course, portfolio etc.
        self.config = self.config_all['builds'][builds_id]
        self.top_name = self.config['top_name']
        self.fig_format = self.config['fig_format']
        self.output_style = self.config['output_style']
        self.pdf_engine = self.config['pdf_engine']
        self.unit = self.config['unit']
        self.base_dir = Path(self.config['base_dir'])
        self.capital_standard = self.config['capital_standard']

        # read in builds level global variables
        self.title = self.config['title']
        self.key = self.config['key']

    def activate_top(self, tops_id, option_id, tidy=False):
        """
        Activate presentation for a particular top level part, chapter, module (the top_name)
        Corresponds to the different ipymd/md files that do the work to make the presentations.

        Output focus; the book of business is defined in builds

        option_id is appended to top_value for a distinguishing key

        :param tops_id:
        :param option_id:
        :param tidy:
        :return:
        """
        # for this specific document
        self.tops_id = tops_id
        top = self.config_all['tops'][tops_id]
        self.subtitle = top['subtitle']
        self.raw_top_value = top['top_value']
        self.option_id = option_id
        self.top_value = self.raw_top_value + self.option_id

        # no choice of filename
        self.file_name = f'{self.key}-{self.top_value}.md'

        self._complete_setup()
        # tidy up
        if tidy:
            self.tidy_up()

    def _complete_setup(self):
        """
        admin around files etc.
        ensures all directories exist
        :return:
        """
        # file specific, set manually
        # main file for output and ancillary directories
        self.file = self.base_dir / self.file_name
        logger.info(f'base_dir = {self.base_dir.resolve()}')
        logger.info(f'output file = {self.file.resolve()}')
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = self.base_dir / 'pdf'
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir = self.base_dir / 'img'
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir = self.base_dir / 'table'
        self.table_dir.mkdir(parents=True, exist_ok=True)
        # for debug output
        (self.table_dir / 'pdf').mkdir(parents=True, exist_ok=True)

        # unless over-ridden, start counting at 1, appendices count negative
        self.section = 0
        self.appendix = 0

        # output
        self.toc = []
        self._slides = set()
        self.sios = OrderedDict(front=StringIO(), contents=StringIO(), summary=StringIO(), body=StringIO(),
                                appendix=StringIO())

        # starts in base self.config: active and not in debug mode
        self._tacit_override = False
        self._active = True
        self._debug = False
        # avoid using dm because updated gets in a loop on the dates...
        self.debug_dm_file = self.base_dir / 'dm2.md'

        # start by writing the toc header
        self._write_buffer_('contents', '## Contents\n')

    def __getitem__(self, item):
        if self.config and item in self.config:
            return self.config[item]
        else:
            raise ValueError(f'Error, {self} has no item {item}.')

    @property
    def builds(self):
        """
        list builds in the config file
        :return:
        """
        s = ''
        for k, v in self.config_all['builds'].items():
            s += f'* `{k}` {v["description"]}{", **active**" if k==self.builds_id else ""}\n'
        return Markdown(s)

    @property
    def tops(self):
        """
        list tops in the config file
        :return:
        """
        s = '| top_value | key | Subtitle | File |\n' \
            '|:----:|:------|:--------------------------------------|:--------------------|\n'
        for k, v in self.config_all['tops'].items():
            s += f'| `{v["top_value"]}` | {k} | {v["subtitle"]} | {v["file"]}|\n'
        return Markdown(s)

    def temp_write(self, *lines):
        """
        write to temporory file

        :param argv:
        :return:
        """
        if self.temp_file is None:
            f = (self.base_dir / "dm_temp_file.md")
            if f.exists():
                f.unlink()
            self.temp_file = f.open('a', encoding='UTF-8')
        lines = [l.replace('\\\\', '\\') for l in lines]
        self.temp_file.writelines(lines)

    def temp_close(self):
        if self.temp_file is not None:
            self.temp_file.close()
        self.temp_file = None

    def pretty(self, kind='all'):
        """
        pretty print the config file
        :param kind:
        :return:
        """
        pp = PrettyPrinter(indent=2, width=100, compact=False, sort_dicts=False).pprint
        if kind in ['a',  'all']:
            pp(self.config_all)
        elif kind in ['b', 'build', 'builds']:
            pp(self.config)
        elif kind in ['t', 'top', 'tops']:
            pp(self.config_all['tops'])
        elif kind in ['gross', 'g']:
            pp(self.config['gross_portfolio'])
        elif kind in ['net', 'n']:
            pp(self.config['net_portfolio'])


    def reset(self):
        """
        reset all buffers etc.
        helpful but lazy

        :return:
        """
        self.activate_top(self.tops_id, self.option_id)
        self.debug = False

    def describe_portfolio(self):
        """
        return markdown description of portfolio
        :return:
        """
        s = ''
        for k, (ln, d) in self['line_descriptions'].items():
            s += f'* `{k}`: {ln}{d}\n'
        return s

    @property
    def debug(self):
        return self._debug

    @debug.setter
    def debug(self, value):
        assert value in ('on', 'append', 'off', True, False)
        # when this changes to on delete the old file; also acts as a reset
        if value in ('on', 'append', 'off', True):
            if self.debug_dm_file.exists():
                self.debug_dm_file.unlink()
        self._debug = value

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, value):
        self._active = value

    @property
    def tacit_override(self):
        return self._tacit_override

    @tacit_override.setter
    def tacit_override(self, value):
        self._tacit_override = value

    def section_number(self, buf):
        """
        get the number / letter for the next section

        return the prefix string for the toc and slide (allows summary not to be numbered)
        :param buf:
        :return:
        """
        prefix = ''
        if self.top_name != '':
            # prefix = f'{self.top_name} {self.top_value}'
            prefix = f'{self.raw_top_value}.'
        # not sure this is a good idea...with versions it intros trivial differences
        # prefix = ''
        if buf == 'body':
            self.section += 1
            return f'{prefix}{self.section:0>2d}. ', self.section
        elif buf == 'summary':
            return '', 0
        elif buf == 'appendix':
            self.appendix -= 1
            # c = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[self.appendix]
            c = int_to_roman(-self.appendix)
            return f'Appendix {prefix}{c}. ', self.appendix
        else:
            raise ValueError('Hopeless confusion!')

    def text(self, txt, buf='body', tacit=False):
        """
        add text to buffer

        only text function can create new sections

        """
        if not self.active:
            return

        stxt = txt.split('\n')

        for i, ln in enumerate(stxt):
            m = re.findall('^# (.*)', ln)
            if m:
                s, s_no = self.section_number(buf)
                tl = self.make_title(m[0])
                ln = f"# {s} {tl}"
                stxt[i] = ln
                self.toc.append((self.raw_top_value, s_no, tl))
            else:
                # title case ## headings too
                m2 = re.findall('^## (.*)', ln)
                if m2:
                    stxt[i] = '\n##  ' + titlecase(ln[3:])

        txt = '\n'.join(stxt) + '\n\n'
        self._write_buffer_(buf, txt)
        if self.tacit_override or not tacit:
            display(Markdown(txt))

    # aliases
    write = text
    blob = text

    def figure(self, f, label, buf='body', caption="", height="", new_slide=True, tacit=False, promise=False,
               option=True, **kwargs):
        """
        add a figure

        if f is a Figure it is used directly
        else call .get_figure
        else ValueError
        label = used for filename too

        :param f:
        :param label:
        :param buf:
        :param caption:
        :param height:
        :param new_slide:
        :param tacit:
        :param promise:  promise mode: just save the figure with appropriate name
        :param option:
        :param kwargs:
        """
        if not self.active:
            return

        if isinstance(f, Figure):
            pass
        else:
            try:
                f = f.get_figure()
            except AttritbuteError as ae:
                logger.warning(f'Cannot coerce input object {f} into a figure...ignoring')
                raise ae

        slide_caption = self.make_title(label)
        label = self.make_safe_label(label)
        if option:
            fig_file = self.figure_dir / f'{self.key}-{self.top_value}-{label}.{self.fig_format}'
        else:
            fig_file = self.figure_dir / f'{self.key}-{self.raw_top_value}-{label}.{self.fig_format}'
        fig_file_local = f'img/{fig_file.name}'
        if fig_file.exists():
            logger.warning(f'File {fig_file} already exists...over-writing.')
        if self.fig_format == 'png' and 'dpi' not in kwargs:
            kwargs['dpi'] = 600
            logger.info("adding 600 dpi")
        f.savefig(fig_file,  **kwargs)

        # not the caption, that can contain tex
        if not self.clean_underscores(caption):
            logger.warning('Caption contains unescaped underscores _. You must ensure they are in dollars.')

        if caption != '':
            fig_text = f'![{caption} '
        else:
            fig_text = '!['
        if label != '':
            fig_text += f'\\label{{fig:{label}}}'
        fig_text += f']({fig_file_local})'
        if height != '':
            fig_text += f'{{height={height}%}}'
        if caption == '':
            # suppresses "Figure" in pandoc
            # https://stackoverflow.com/questions/45030895/how-to-add-an-image-with-alt-tag-but-without-caption-in-pandoc
            fig_text += '  \\'
        fig_text += '\n\n'
        if new_slide and not promise:
            # need to write in one row so that debug mode write correctly to dm.md
            text_out = f'\n## {slide_caption}\n\n{fig_text}'
        elif not promise:
            # not new slide, not promise
            text_out = fig_text
        elif promise:
            # don't write anything
            return fig_text
        else:
            raise ValueError('Unexpected ')

        # actually write output
        self._write_buffer_(buf, text_out)

        if self.tacit_override or not tacit:
            display(Markdown(text_out))

    def save(self, dir, filestem, body, option):
        """
        just save body string to filename
        return the relevant include file

        :param self:
        :param filename:
        :param body:
        :return:
        """
        filestem = self.make_safe_label(filestem)
        if option:
            filename = f'{dir}/{self.key}-{self.top_value}-{filestem}.md'
        else:
            filename = f'{dir}/{self.key}-{self.raw_top_value}-{filestem}.md'
        file = self.base_dir / filename

        with file.open('w', encoding='utf-8') as f:
            f.write(body)

        return f'@@@include {filename}'

    def tikz_table(self, df, label, *, caption="", buf='body',
                   new_slide=True, tacit=False, promise=False,
                   float_format=None, tabs=None,
                   show_index=True, scale=0.717,
                   figure='figure', hrule=None,
                   equal=False, option=True,
                   vrule=None, sparsify=1):
        """

        Add a table using TikZ formatter

        label used as file name
        force_float = convert input to float first (makes a copy) for col alignment

        output_style as table
            with_table : all output in @@@ file and include in main md file; use when caption is generic
            caption:   puts caption text in the main markdown file, use when caption will be edited
            inline: all output in md file directly (not recommended)

        label and columns have _ escaped for TeX but the caption is not - so beware!

        fn_out=None, float_format=None, tabs=None,
                show_index=True, scale=0.717, height=1.5, column_sep=2, row_sep=0,
                figure='figure', color='blue!0', extra_defs='', lines=None,
                post_process='', label='', caption=''

        """

        assert not promise or self.output_style == 'with_table'

        if not self.active:
            if not tacit:
                display(df)
            return

        if float_format is None:
            float_format = self.default_float_fmt

        df = df.copy()
        df = df.astype(float, errors='ignore')

        # have to handle column names that may include _
        # For now assume there are not TeX names
        slide_caption = self.make_title(label)
        label = self.make_safe_label(label)
        label = self.clean_name(label)

        # check the caption, that can contain tex
        if not self.clean_underscores(caption):
            logger.warning('Caption contains unescaped underscores _. You must ensure they are in dollars.')

        # do the work
        s_table = df_to_tikz(df, label=label, caption=caption,
                             float_format=float_format, tabs=tabs,
                             show_index=show_index, scale=scale, equal=equal,
                             figure=figure, hrule=hrule, vrule=vrule, sparsify=sparsify, clean_index=True)

        # added a local bufer so output works better with debug model: need a single call to ._write_buffer_
        sio_temp = StringIO()
        if option:
            table_file = self.table_dir / f'{self.key}-{self.top_value}-{label}.md'
        else:
            table_file = self.table_dir / f'{self.key}-{self.raw_top_value}-{label}.md'
        if table_file.exists():
            logger.debug(f'File {table_file} exists, over-writing.')
        table_file_local = f'table/{table_file.name}'

        if new_slide and not promise:
            sio_temp.write(f'\n## {slide_caption}\n\n')
            # self._write_buffer_(buf, f'## {slide_caption}\n\n')
            if not tacit:
                display(Markdown(f'## {slide_caption}'))

        if (self.output_style == 'inline') or (self.output_style == 'in-line'):
            sio_temp.write(s_table)
            # self._write_buffer_(buf, s_table)

        elif self.output_style == 'with_table':
            with table_file.open('w', encoding='utf-8') as f:
                f.write(s_table)
            if promise:
                return f'@@@include {table_file_local}\n\n'
            else:
                sio_temp.write(f'@@@include {table_file_local}\n\n')
                # self._write_buffer_(buf, f'@@@include {table_file_local}\n\n')

        else:
            raise ValueError(f'Unknown option {self.output_style} for output_style passed to table.')

        # actually do the write
        self._write_buffer_(buf, sio_temp.getvalue())

        if self.tacit_override or not tacit:
            display(df.style.format(float_format))
            if caption != '':
                display(Markdown(caption))

    @staticmethod
    def default_float_format(x, neng=3):
        """
        the endless quest for the perfect float formatter...

        tester:

            for x in 1.123123982398324723947 * 10.**np.arange(-23, 23):
                print(default_float_format(x))

        :param x:
        :return:
        """
        ef = EngFormatter(neng, True)
        try:
            if x == 0:
                ans = '0'
            elif 1e-3 <= abs(x) < 1e6:
                if abs(x) <= 10:
                    ans = f'{x:.3g}'
                elif abs(x) < 100:
                    ans = f'{x:,.2f}'
                elif abs(x) < 1000:
                    ans = f'{x:,.1f}'
                else:
                    ans = f'{x:,.0f}'
            else:
                ans = ef(x)
            return ans
        except:
            return x

    @staticmethod
    def make_title(s):
        """
        make s into a slide title (label/title interoperability)

        :param s:
        :return:
        """
        # https://pypi.org/project/titlecase/#description
        s = titlecase(s)  # .replace('-', ' '))
        # from / to
        # for f, t in zip(['Var ', 'Tvar ', 'Cv ', 'Vs ', 'Epd ', 'Lr ', 'Roe ', 'Cle ', 'Gdp', 'Gnp'],
        #                 ['VaR ', 'TVaR ', 'CV ', 'vs. ', 'EPD ', 'LR ', 'ROE ', 'CLE ', 'GDP', 'GNP']):
        #     s = s.replace(f, t)
        return s

    @staticmethod
    def make_safe_label(label):

        """
        convert label to suitable for tex lable and file name statements

        You can look at the Django framework for how they create a "slug" from arbitrary
        text. A slug is URL- and filename- friendly.

        The Django text utils define a function, slugify(), that's probably the gold standard
        for this kind of thing. Essentially, their code is the following.

        Normalizes string, converts to lowercase, removes non-alpha characters,
        and converts spaces to hyphens.

        https://stackoverflow.com/questions/295135/turn-a-string-into-a-valid-filename
        """
        value = unicodedata.normalize('NFKD', label)
        value = re.sub(r'[^\w\s-]', '', value).strip().lower()
        value = re.sub(r'[-\s]+', '-', value)
        return value

    def _write_buffer_(self, buf, text='\n'):
        """
        single point to write to buffer, allows overloading of other write related functions, such as tracking toc
        and making a tree diagram of the document

        not to be called externally, hence name

        :param buf:
        :param text:
        :return:
        """
        # find the sections and add labels
        sections = re.findall('^(## .+)', text, re.MULTILINE)
        for slide_heading in sections:
            if slide_heading not in self._slides:
                # label the first occurrence of each slide name
                self._slides.add(slide_heading)
                new_text = f'{slide_heading} \\label{{slide:{self.make_safe_label(slide_heading[3:])}}}\n'
                # count = 1: only replace the first occurrence
                text = text.replace(slide_heading, new_text, 1)
                logger.info(f'New section>> {new_text[:-1]}')

        # actually write the text
        self.sios[buf].write(text)

        # optionally write debug file
        if self.debug in ('on', 'append', True):
            logger.info(f'In debug mode...writing {self.debug_dm_file}')
            mode = 'a' if self.debug == 'append' else 'w'
            self.debug_dm_file.open(mode, encoding='utf-8').write(text)

    @property
    def slides(self):
        return self._slides

    def tidy_up(self):
        """
        tidy up all project related files

        :return:
        """
        # tidying up
        existing = list(self.base_dir.glob(f'**/{self.key}-*.*'))
        if len(existing):
            logger.info(f"Deleting {len(existing)} existing file(s)...")

        for f in existing:
            f.unlink()

        # just on principle
        logger.info('Checking TMP files...')
        for f in self.base_dir.parent.glob(f'TMP_*.md'):
            logger.info(f'Deleting {f}')
            f.unlink()

    @staticmethod
    def date():
        return "Created {date:%Y-%m-%d %H:%M:%S}". \
            format(date=datetime.datetime.now())

    def dump_toc(self):
        """
        save the toc as a json file
        :return:
        """
        p = self.file.with_name(f'{self.key}-{self.top_value}-toc.json')
        json.dump(self.toc, p.open('w', encoding='utf-8'), indent=4)
        logger.info(f'Writing toc to {p.resolve()}')

    def load_toc(self):
        p = self.file.with_name(f'{self.key}-{self.top_value}-toc.json')
        try:
            toc = json.load(p.open('r', encoding='utf-8'))
            logger.info(f'Read toc from {p.resolve()}')
        except FileNotFoundError as e:
            logger.warning(f'TOC file {self.key}-{self.top_value}toc.json not found.')
            toc = None
        return toc

    def buffer(self, decorate=''):
        """
        assemble all the parts
        :return:
        """
        # update toc
        self.make_toc(decorate)
        # assemble parts
        s = '\n\n'.join(buf.getvalue().strip() for buf in self.sios.values())
        # sublime text-esque removal of white space
        s = re.sub(' *\n(\n?)\n*', r'\n\1', s, flags=re.MULTILINE)
        return s

    def make_toc(self, decorate=''):
        """
        sort the toc: Summary, Sections, Appendix
        if decorate != '' it will bold thaat row (for burst mode)
        :return:
        """
        # new SIO each time for contents...
        self.sios['contents'] = StringIO()

        if len(self.toc) == 0:
            return

        # there are no sections in the summary (section is # level)
        # toc_summary = sorted([i for i in self.toc if i[0].split()[2].strip() not in ('Appendix', 'Section')],
        #                      key=sort_fun)
        toc_body = sorted([i for i in self.toc if i[1] > 0], key=lambda x : x[1])
        toc_appendix = sorted([i for i in self.toc if i[1] < 0], key=lambda x : x[1])
        self.sios['contents'].write(f'# Table of Contents\n## {self.top_name} {self.raw_top_value} Contents\n\n')
        appendix_spacer_needed = True
        for t, s, c in toc_body + toc_appendix:
            s_ = int_to_roman(-s) if s < 0 else s
            if s < 0 and appendix_spacer_needed:
                self.sios['contents'].write('\n')
                appendix_spacer_needed = False
            if decorate == s:
                # annotate row
                self.sios['contents'].write(
                    f'\n\\grttocentry{{\\bf {"Section" if s > 0 else "Appendix"} {t}.{s_}}}{{\\bf {c}}}')
            else:
                self.sios['contents'].write(
                    f'\n\\grttocentry{{{"Section" if s > 0 else "Appendix"} {t}.{s_}}}{{{c}}}')

    def buffer_display(self):
        display(Markdown(self.buffer()))

    def buffer_persist(self, font_size=9, tacit=False, mode='single', debug=False, debug_filename='dm2.md'):
        """
        persist and write out the StringIO cache to a physical file

        if debug just write to dm.md and exit

        :param font_size:
        :param mode:   burst (separate files per section or all together or both
        :param toc:    for burst mode: read toc to number sections ignored if it doesn't exist
        :param tacit:  suppress markdown output of persisted file

        """

        if debug or mode in ('single', 'both'):
            subtitle = f'{self.top_name} {self.top_value}: {self.subtitle}'
            y = self['yaml'].format(
                    title=self.title, subtitle=subtitle, created=self.date(),
                    font_size=font_size, pdf_engine=self.pdf_engine)
            if debug:
                debug_file = self.base_dir / debug_filename
                with debug_file.open('w', encoding='UTF-8') as f:
                    f.write(y)
                    f.write(self.buffer())
                    logger.info(f'Writing to file {debug_file.resolve()}...and exiting.')
                return

            with self.file.open('w', encoding='UTF-8') as f:
                f.write(y)
                f.write(self.buffer())
                logger.info(f'Writing to file {self.file.resolve()}')
            # in all mode, dump out the toc for future reference
            self.dump_toc()
            if not tacit:
                display(Markdown(self.file.open('r').read()))

        if mode in ('burst', 'both'):
            toc = self.load_toc()
            if toc is None:
                toc = self.toc
            # full section name -> (top number, section number)
            toc_mapper = {k: (i, j) for i, j, k in toc}
            # put in the correct section numbers according to the master toc
            self.toc = [(*toc_mapper[k], k) for i, j, k in self.toc]
            logger.info(f'Adjusted toc = {self.toc}')

            # read buffers
            summary = self.sios['summary'].getvalue()

            body = self.sios['body'].getvalue().strip()
            sbody = re.split('^(# .+)$', body, flags=re.MULTILINE)
            body_dict = { i.split('  ')[1]: j for i, j in zip(sbody[1::2], sbody[2::2])}
            if sbody[0] != '':
                logger.warning(f'Pre first section not empty...Adding to summary.\n{sbody[0][:200]}')
                summary = summary + sbody[0]

            appendix = self.sios['appendix'].getvalue().strip()
            if appendix != '':
                sappendix = re.split('^(# .+)$', appendix, flags=re.MULTILINE)
                appendix_dict = { i.split('  ')[1]: j for i, j in zip(sappendix[1::2], sappendix[2::2])}
                if appendix[0] != '':
                    logger.warning(f'Pre first content before Appendix: not expected..ignoring\n{appendix[0][:100]}.')
            else:
                appendix_dict = {}

            for d in [body_dict, appendix_dict]:
                for section, content in d.items():
                    top_no, sec_no = toc_mapper[section]
                    logger.info(f'Writing {top_no}.{sec_no}: {section}')
                    s_ = f'{sec_no:0>2d}' if sec_no > 0 else int_to_roman(-sec_no)
                    fsec_no = f'{top_no}.{s_}'
                    self.make_toc(decorate=sec_no)
                    y = self['yaml'].format(
                            title=f'{self.top_name} {self.top_value}: {self.subtitle}',
                            subtitle=section, created=self.date(),
                            font_size=font_size, pdf_engine=self.pdf_engine)
                    with (self.base_dir / f'{self.key}-{top_no}-{s_}-burst.md').open("w", encoding='utf-8') as f:
                        f.write(y)
                        f.write('\n')
                        f.write(self.sios['contents'].getvalue().strip())
                        if summary != '':
                            # summary attached to toc, comes before the body content
                            f.write(summary.strip())
                            f.write('\n\n')
                        f.write(f'\n\n# {top_no}.{sec_no}. {section.strip()}\n\n')
                        f.write(content)

    def finish(self):
        """
        json driven last steps, reads font size and mode (single, both burst) from the config file

        :return:
        """
        font_size = self.config['font_size']
        mode = self.config['mode']
        self.buffer_persist(font_size, tacit=True, mode=mode)

    def process(self, font_size=9, tacit=True):
        """
        Write out and run pandoc to convert
        Can set font_size and margin here, default 10 and 1 inches.

        Set over_write = False and an existing markdown file will not be overwritten.

        """
        if not self.active:
            logger.warning('Currently inactive...returning')
            return

        self.buffer_persist(font_size, tacit)
        self.make_beamer(self.file)

    def process_burst(self):
        """
        pandoc all the burst files
        :return:
        """
        for f in self.base_dir.glob(f'{self.key}-burst-*.md'):
            # parallelize would be nice...
            logger.info(f'Processing {f}')
            self.make_beamer(f)

    @staticmethod
    def clean_name(n):
        """
        escape underscores for using a name in a DataFrame index

        :param n:
        :return:
        """
        try:
            if type(n) == str:
                return n.replace('_', r'\_')
            else:
                return n
        except:
            return n

    @staticmethod
    def clean_underscores(s):
        """
        check s for unescaped _s
        returns true if all _ escaped else false
        :param s:
        :return:
        """
        return np.all([s[x.start() - 1] == '\\' for x in re.finditer('_', s)])

    @staticmethod
    def clean_index(df):
        """
        escape _ for columns and index
        whether multi or not

        !!! you can do this with a renamer...

        :param df:
        :return:
        """

        idx_names = df.index.names
        col_names = df.columns.names

        if isinstance(df.columns, pd.core.indexes.multi.MultiIndex):
            df.columns = PresentationManager.clean_mindex_work(df.columns)
        else:
            df.columns = map(PresentationManager.clean_name, df.columns)

        # index
        if isinstance(df.index, pd.core.indexes.multi.MultiIndex):
            df.index = PresentationManager.clean_mindex_work(df.index)
        else:
            df.index = map(PresentationManager.clean_name, df.index)
        df.index.names = idx_names
        df.columns.names = col_names
        return df

    @staticmethod
    def clean_mindex_work(idx):
        for i, lv in enumerate(idx.levels):
            if lv.dtype == 'object':
                repl = map(PresentationManager.clean_name, lv)
                idx = idx.set_levels(repl, level=i)
        return idx

    def make_beamer(self, path_file):
        """
        make path_file = Path file object if possible

        :param path_file:
        :return:
        """
        # make it if windows
        if str(Path.home()) == '/home/steve':
            logger.warning('Linux system...cannot make file')
        else:
            logger.info('making file')
            cwd = Path.cwd()
            os.chdir(self.base_dir)
            markdown_make_main("", path_file.name, path_file.stem)
            os.chdir(cwd)

    def course_contents(self):
        """
        List of modules and links
        :return:
        """
        sio = StringIO()
        p = Path('../generated/pdf')
        i = 65
        for k, v in self.config_all['tops'].items():
            fn = f'pirc-{v["top_value"]}.pdf'
            print((p/fn).resolve())
            if (p/fn).exists():
                sio.write(f'* [{chr(i)}. {v["subtitle"]}](pirc/{fn})\n')
            else:
                sio.write(f'* {chr(i)}. {v["subtitle"]} (coming soon)\n')
            i += 1
        md = sio.getvalue()
        html = pypandoc.convert(md, 'html', 'markdown')
        with Path('../generated/html/pirc-course-contents.html').open('w', encoding='utf-8') as f:
            f.write(html)

        return md, html

    def summarize(self):
        """
        Summarize from config file

        based on markdown_make.md_summary()

        Stephen J. Mildenhall (c) 2021
        """

        # sio = full summary
        # short = short summary that goes to website
        sio = StringIO()
        short = StringIO()
        git = StringIO()
        # start by pretty printing the config file
        sio.write(f'# **{self.title}** Configuration File\n')

        sio.write('## Project Level Parameters\n\n')

        for k, v in self.config.items():
            if k == 'tops': break
            sio.write(f'* `{k}` {v}\n')

        top_name = self.config['top_name']
        sio.write(f'\n### {top_name} Contents\n')
        tops = self.config_all['tops']
        for k, v in tops.items():
            tv = v['top_value']
            st = v['subtitle']
            sio.write(f'* `{tv}` {st}\n')

        sio.write(f'\n### Portfolio Specification\n')
        sio.write(f'\n#### Gross\n')
        sio.write(f'```\n{self.config["gross_portfolio"].strip()}\n```\n\n')
        sio.write(f'\n#### Net\n')
        sio.write(f'```\n{self.config["net_portfolio"].strip()}\n```\n\n')

        sio.write('## Individual Module Contents\n')

        regexp = re.compile("^(#[#]*) (.*)$|^(```)")
        tab = '\t'

        # keep track of subtitles and section numbering
        dir_path = Path('../generated')
        last = ['','','','','','','','']
        nos = [0,0,0,0,0,0,0]
        last_level = 0
        # cut out duplicates
        last_slide = ''
        for v in tops.values():
            k = v['top_value']
            f = dir_path / f"pirc-{k}.md"
            fn = v['file']
            st = v['subtitle']
            logger.info(f.name)
            if f.exists():
                # fourth argument returns before doing any real work
                logger.info(f'processing {f}')
                resolved_f = markdown_make_main('', str(f), f.stem, None)
                resolved_f = Path(resolved_f)
                content_f = dir_path / f"pdf/pirc-{k}.pdf"
                if content_f.exists():
                    updated = f'{datetime.datetime.fromtimestamp(content_f.stat().st_mtime):%Y-%m-%d %H:%M:%S%z} UTZ'
                else:
                    updated = 'not created'
                with resolved_f.open('r', encoding='utf-8') as fh:
                    linked_st = f'[{st}](pirc-{k}.pdf)'
                    git_linked_st = f'[{st}](https://github.com/mynl/PIRC/blob/main/Python/{fn})'
                    sio.write(f'\n### {k}. {linked_st}\n\n')
                    short.write(f'* {k}. {linked_st}\n')
                    git.write(f'* {k}. {git_linked_st}\n')
                    # do not start off in a code block
                    in_code_block = False
                    for line in fh:
                        for match in regexp.finditer(line):
                            if not in_code_block and match.group(1) != None:
                                # matched a section: how many #s?; note match groups are 1-based lists
                                l = len(match.group(1))
                                if last[l] == match.group(2):
                                    # repeat, do nothing
                                    pass
                                else:
                                    # get rid of labels
                                    s = re.sub(r'\\label.*$', r'', match.group(2).strip())
                                    if l == 1:
                                        # presentation section level, no duplicates
                                        nos[l] += 1
                                        sio.write(f"* **{s:s}**\n")
                                        if s != 'Table of Contents' and s[:8] != 'Appendix':
                                            short.write(f'\t* {s}\n')
                                            git.write(f'\t* {s}\n')
                                    elif l == 2:
                                        # slide give numbers
                                        if s.strip() == last_slide:
                                            pass
                                        else:
                                            nos[l] += 1
                                            last_slide = s.strip()
                                            sio.write(f"\t{nos[l]:d}. {s:s}\n")
                                    else:
                                        # bullets
                                        nos[l] += 1
                                        sio.write(f"{tab*(l-1)}* {match.group(2):s}\n")
                                        # if l == 3:
                                        #     short.write(f"{tab*(l-1)}* {match.group(2):s}\n")
                                    last[l] = match.group(2)
                                if l < last_level:
                                    # reset deeper levels
                                    for j in range(l+1, last_level+1):
                                        nos[j] = 0
                                        last[j] = ''
                                last_level = l
                            elif match.group(3) != None and match.group(3) == '```':
                                ## in out code block
                                in_code_block = not in_code_block
                    short.write(f'\t* Last pdf update {updated}\n')
                # be tidy
                resolved_f.unlink()
            else:
                sio.write(f'\n## {k}. {st} (file does not exist)\n\n')
                short.write(f'* {k}. {st} (coming soon)\n')
                git.write(f'* {k}. {st} (coming soon)\n')

        long_md = sio.getvalue()
        short_md = short.getvalue()
        short_html = pypandoc.convert(short_md, 'html', 'markdown')
        git_md = git.getvalue()
        # git_html = pypandoc.convert(git_md, 'html', 'markdown')
        stripper = 'Individual Module Contents'
        long_html = pypandoc.convert(long_md[long_md.find(stripper) + len(stripper):], 'html', 'markdown')

        with Path('../generated/html/pirc-short-contents.html').open('w', encoding='utf-8') as f:
            f.write(short_html)
        with Path('../generated/html/pirc-long-contents.html').open('w', encoding='utf-8') as f:
            f.write(long_html)

        return long_md, short_md, git_md

        # process output to HTML
        # out_file_html = dir_path / f'{out_name}.html'
        # args = ( f"/home/steve/anaconda3/bin/pandoc --standalone --metadata title:'{out_name}' "
        #     "-V author:'Stephen J. Mildenhall' "
        #     "-f markdown+smart -t html --css=../css/github.css "
        #     f"-o  {out_file_md.resolve()} {out_file_md.resolve()}")
        # print(args)
        # subprocess.Popen(args)

    def uber_deck(self, subtitle, font_size=9, pdf_engine=''):
        """
        make a deck combining all available markdpown files
        try cla: --pdf-engine=lualatex if needed
        :return:
        """
        sio = StringIO()
        # generally need a beefier engine to build this deck...
        y = self['yaml'].format(
                title=self.title, subtitle=subtitle, created=self.date(),
                font_size=font_size, pdf_engine=pdf_engine)
        sio.write(f'{y}\n')
        for fn in sorted(self.base_dir.glob(f'{self.key}-?.md')):
            with fn.open('r', encoding='utf-8') as f:
                txt = f.read()
                lines = txt.split('\n')
                in_text = False
                for i, ln in enumerate(lines[1:]):
                    if in_text:
                        sio.write(f'{ln}\n')
                    elif ln == '---':
                        in_text = True
                    elif ln[:10] == 'subtitle: ':
                        print(f'Found section: {ln[11:-1]}')
                        sio.write(f'\\grttopdivider{{{ln[11:-1]}}}\n')
                    else:
                        # yaml line, ignored
                        pass
        out = self.base_dir / f'{self.key}-uber.md'
        with out.open('w', encoding='utf-8') as f:
            f.write(sio.getvalue())

    @staticmethod
    def short_hash(x, n=10):
        hasher = hashlib.sha1(x.encode('utf-8'))
        return base64.b32encode(hasher.digest())[:n].decode()

    def combine(self):
        """
        break pages of all files in PM into tops, sections and slides
        return as nice DataFrame

        self = PM object

        """

        # keep track of subtitles and section numbering
        dir_path = self.base_dir
        tops = self.config_all['tops']

        regexp = re.compile(r'^(# .*)$|^(## .*)$|^(---)$', flags=re.MULTILINE)
        # three groups here, plus what is between the dividers
        ngroups = 4
        dfs = []
        for v in tops.values():
            k = v['top_value']
            f = dir_path / f"pirc-{k}.md"
            logger.debug(f.name)

            if f.exists():
                # fourth argument returns before doing any real work
                logger.debug(f'processing {f}')
                resolved_f = markdown_make_main('', str(f), f.stem, None)
                # this is a TMP file
                logger.debug(resolved_f)
                resolved_f = Path(resolved_f)
                with resolved_f.open('r', encoding='utf-8') as fh:
                    txt = fh.read()
                    stxt = regexp.split(txt)
                    split = [stxt[1:][j * ngroups:j * ngroups + ngroups]
                             for j in range((len(stxt) - 1) // ngroups)]
                    if split[0][2] == '---':
                        split = split[2:]
                    # set the index to correspond to page numbers in the PDF (actual pages, not on-slide pages; no pauses)
                    df = pd.DataFrame(split,
                                      columns=['section', 'slide', 'dash', 'txt'],
                                      index=range(2, len(split) + 2))
                    dfs.append(df)
                # be tidy
                resolved_f.unlink()
            else:
                df = pd.DataFrame(columns=['section', 'slide', 'dash', 'txt'])
                df.loc[2] = ['# Table of Contents', None, None, f'Top {k} Coming Soon']
                dfs.append(df)

        dfall = pd.concat(dfs, keys=[i['top_value'] for i in self.config_all['tops'].values()], names=['top', 'idx'])
        dfall['ref'] = dfall.apply(lambda x: x.iloc[:-1].str.cat(), axis=1)
        dfall['content'] = dfall.apply(lambda x: x.iloc[:-1].str.cat(), axis=1)
        dfall['length'] = [len(i) for i in dfall.content]
        dfall[['kind', 'title']] = dfall.ref.str.extract('(##*) (.*)')
        dfall = dfall[['kind', 'title', 'content', 'length']].copy()
        dfall['hash'] = [self.short_hash(i, 4) for i in dfall.content]
        for k, v in tops.items():
            dfall.loc[(v['top_value'], 1), :] = ['.', v['subtitle'], '', 0, v['top_value']*4]
        dfall = dfall.sort_index()
        dfall['title'] = [self.remove_label(i) for i in dfall['title']]
        return dfall

    @staticmethod
    def combine_generic(dir_path, pattern, columns, hash_length=50, expand_includes=False, exclude_yaml=True):
        """
        generic version of combine that will apply to anything (e.g., the book!)

        for PIRC
        columns = ['section', 'slide', 'dash', 'txt']

        for the book
        columns = ['chapter', 'section', 'subsection', 'dash', 'txt']

        :param dir_path:  where the files live
        :param pattern: glob pattern for files to combine
        :return:
        """

        # keep track of subtitles and section numbering
        # going to assume there are no comments in code...

        # make the search string
        s = ''
        for i, t in enumerate(columns[:-2]):
            s += f'^({"#" * (i+1)} .*)$|'
        s +=  "^(---)$"
        logger.info(f'Using regex pattern {s}')
        regexp = re.compile(s, flags=re.MULTILINE)
        ngroups = len(columns)
        dfs = []
        fs = []

        cwd = os.getcwd()
        if dir_path.is_file():
            search = [dir_path]
            os.chdir(dir_path.parent)
        else:
            os.chdir(dir_path)
            search = dir_path.glob(pattern)

        for f in search:
            # fourth argument returns before doing any real work
            logger.info(f'processing {f}')
            fs.append(f.stem)
            if expand_includes:
                resolved_f = markdown_make_main('', str(f), f.stem, None)
                # this is a TMP file
                logger.info(f'Expanded to {resolved_f}')
                resolved_f = Path(resolved_f)
            else:
                resolved_f = f
            with resolved_f.open('r', encoding='utf-8') as fh:
                txt = fh.read()
                stxt = regexp.split(txt)
                split = [stxt[1:][j * ngroups:j * ngroups + ngroups]
                         for j in range((len(stxt) - 1) // ngroups)]
                if exclude_yaml and split[0][2] == '---':
                    split = split[2:]
                # set the index to correspond to page numbers in the PDF (actual pages, not on-slide pages; no pauses)
                start = 2 if exclude_yaml else 1
                df = pd.DataFrame(split,
                                  columns=columns,
                                  index=range(start, len(split) + start))
                dfs.append(df)
            # be tidy
            if expand_includes and resolved_f is not f:
                resolved_f.unlink()

        dfall = pd.concat(dfs, keys=fs, names=['top', 'idx'])
        dfall['ref'] = dfall.apply(lambda x: x.iloc[:-1].str.cat(), axis=1)
        dfall['content'] = dfall.apply(lambda x: x.iloc[:-1].str.cat(), axis=1)
        dfall['length'] = [len(i) for i in dfall.content]
        dfall[['kind', 'title']] = dfall.ref.str.extract('(##*) (.*)')
        dfall = dfall[['kind', 'title', 'content', 'length']].copy()
        dfall['level'] = dfall.kind.str.len().fillna(0).astype(int)
        dfall['hash'] = [PresentationManager.short_hash(i, hash_length) for i in dfall.content]
        idx = (dfall.level > 0)
        indent = '    '
        dfall['ititle'] = ''
        dfall.loc[idx, 'ititle'] = [ indent * lvl + t for _, (lvl, t) in dfall.loc[idx, ['level', 'title']].iterrows()]
        dfall = dfall.sort_index()
        dfall['title'] = [PresentationManager.remove_label(i) for i in dfall['title']]

        os.chdir(cwd)

        return dfall

    @staticmethod
    def remove_label(s):
        try:
            return re.sub(r'\\label{[^}]*}$', '', s)
        except:
            return s

    def make_accordion(self, df):
        """
        code for website
        """

        sio = StringIO()

        sio.write('''
    <div id="accordion-toc">
        <form action='/'>''')

        for top, g in df.groupby('top'):
            if len(g) > 1:
                top_contents = g.apply(lambda x: f'''
                    <div class="form-group form-check">
                        <label class="form-check-label">
                            <input class="form-check-input" type="checkbox">{x.title} ({x.hash})</input>
                        </label>
                    </div>''' if x.kind == '#' else ( f'''
                        <div class="form-group form-check"  style="text-indent: 2em">
                            <label class="form-check-label">
                                <input class="form-check-input" type="checkbox">[{x.hash}] {self.remove_label(x.title)} </input>
                            </label>
                        </div>'''
                        if x.kind=='##' else ''),
                    axis=1).str.cat()
            else:
                top_contents = '\n                <li> Coming soon. </li>'
            s = f'''
            <div class="card">
                <div class="card-header">
                  <a class="card-link" data-toggle="collapse" href="#{top}">
                    {g.title[0]} ({g.hash[0]})
                  </a>
                </div>
                <div id="{top}" class="collapse show" data-parent="#accordion-toc">
                  <div class="card-body">
                    <div class="form-group form-check">
                        <label class="form-check-label">
                            <input class="form-check-input" type="checkbox">Select All</input>
                        </label>
                    </div>{top_contents}
                  </div>
                </div>
            </div>'''
            sio.write(s)

        sio.write('''
        </form>
    </div>
    ''')

        with (self.base_dir / 'html/pirc-short-contents.html').open('w', encoding='utf-8') as f:
            f.write(sio.getvalue())

        return sio.getvalue()

    def make_deck(self, df, title, subtitle, handout, filestem, *hash_list):
        """

        create a doc from a list of slide hashes
        df = result of running self.combine()

        call either *list_of_hashes or hash1, hash2, ....

        """

        file = self.base_dir / f'{filestem}.md'
        with file.open('w', encoding='utf-8') as f:
            yaml = self.config['yaml'].format(
                title=title,
                subtitle=subtitle,
                created=self.date(),
                font_size=9,
                pdf_engine=''
            )
            yaml = yaml.replace('toc: false', 'toc: true')
            if not handout:
                yaml = yaml.replace(',handout', '')
            f.write(yaml)
            dfl = df.copy().set_index('hash')
            f.write(dfl.loc[hash_list, :].content.str.cat())

    def toggle_debug(self, onoff):
        """
        onoff == on - set debug=True but remember previous state
        onoff == off - restore previous state
        :param onoff:
        :return:
        """

        if onoff == "on":
            self._debug_state = self.debug
            self.debug = True

        elif onoff == 'off':
            self.debug = self._debug_state


@magics_class
class PresentationManagerMagic(Magics):
    """
    description: implements magics to help using PresentationManager class

    pmt = pres maker text (blob, write)

    """

    @line_cell_magic
    @magic_arguments('%pmb')
    @argument('-a', '--appendix', action='store_true', help='Mark as appendix material.')
    @argument('-c', '--code', action='store_true', help='Enclose in triple quotes as Python code.')
    @argument('-d', '--debug', action='store_true', help='Turn debug on temporarilty .')
    @argument('-f', '--fstring', action='store_true', help='Convert cell into f string and evaluate.')
    @argument('--front', action='store_true', help='Mark as front matter, before the toc.')
    @argument('-i', '--ignore', action='store_true', help='Ignore the cell, turns it into a comment')
    @argument('-t', '--tacit', action='store_true', help='Tacit: suppress output as Markdown')
    @argument('-s', '--summary', action='store_true', help='Mark as summary material (exlusive with appendix).')
    def pmb(self, line='', cell=None):
        """
        PresentationManager blob (text/write) line/cell magic

        %pmt line  -> written to body
        %%pmt -s -a -m  (s show; a=appendix, m=suMmary)

        """
        self.shell.ex('if PM.debug: logger.warning(f"PM debug mode set to {PM.debug}")')
        if cell is None:
            # defaults, tacit for everything except sections
            if line.strip()[0:2] == '# ':
                tacit = False
            else:
                tacit = True
            self.shell.ev(f'PM.text("\\n{line}", buf="body", tacit={tacit})')
        else:
            args = parse_argstring(self.pmb, line)
            if args.ignore:
                return
            logger.debug(args)
            buf = 'body'
            if args.appendix:
                buf = 'appendix'
            if args.summary:
                buf = 'summary'
            if args.front:
                buf = 'front'
            if args.fstring:
                logger.debug('evaluating as f string')
                if args.code:
                    temp = f'f"""\\\\\\\\footnotesize\n\n```python\n{cell}\n```\n\\\\\\\\normalsize"""'
                else:
                    temp = f'f"""{cell}"""'
                cell = self.shell.ev(temp)
            if args.debug:
                self.shell.ev('PM.toggle_debug("on")')
            self.shell.ev(f'PM.text("""{cell}""", buf="{buf}", tacit={args.tacit})')
            if args.debug:
                self.shell.ev('PM.toggle_debug("off")')

    @line_cell_magic
    @magic_arguments('%pmf')
    @argument('-a', '--appendix', action='store_true', help='Mark as appendix material.')
    @argument('-c', '--command', action='store_true', help='Load the underlying command into the current cell')
    @argument('-d', '--debug', action='store_true', help='Turn debug on temporarilty .')
    @argument('-f', '--fstring', action='store_true', help='Convert caption into f string and evaluate.')
    @argument('-h', '--height', type=float, default=0.0, help='Vertical size, generates height=height clause.')
    @argument('-i', '--ignore', action='store_true', help='Ignore the cell, turns into a comment')
    @argument('-n', '--new_slide', action='store_false', help='Set to suppress new slide.')
    @argument('-o', '--option', action='store_true',
              help='Set option for tables that vary with options, uses decorated top_value filename.')
    @argument('-p', '--promise', action='store_true', help='Promise: write file but append nothing to stream')
    @argument('-s', '--summary', action='store_true', help='Mark as summary material (exlusive with appendix).')
    @argument('-t', '--tacit', action='store_true', help='tacit: suppress output as Markdown')
    def pmf(self, line='', cell=None):
        """
        PresentationManager figure utility. Add a new figure to the stream, format and save as self.fig_format file.

        In line mode

            %pmf f @ label
            %pmf label         # assumes figure called f

        In cell mode

            %%pmf [options]
            f
            label text
            many lines of caption text
            caption continues.
        """
        self.shell.ex('if PM.debug: logger.warning(f"PM debug mode set to {PM.debug}")')
        if cell:
            args = parse_argstring(self.pmf, line)
            if args.ignore:
                return
            logger.debug(args)
            buf = 'body'
            if args.appendix: buf = 'appendix'
            if args.summary: buf = 'summary'

            # get rid of multiple new lines
            stxt = re.sub('\n+', '\n', cell, re.MULTILINE).strip().split('\n')
            assert len(stxt) >= 2
            f = stxt[0].strip()
            label = stxt[1].strip()
            if len(stxt) > 2:
                caption = '\n'.join(i.strip() for i in stxt[2:])
                if args.fstring:
                    logger.info('evaluating caption as f string')
                    temp = f'f"""{caption}"""'
                    caption = self.shell.ev(temp)
            else:
                caption = ""

            logger.info(caption)
            s = f'promise = PM.figure({f}, "{label}", buf="{buf}", caption="""{caption}""", ' \
                f'new_slide={args.new_slide}, tacit={args.tacit}, promise={args.promise}, option={args.option}'
            if args.height:
                s += f', height={args.height}'
            s += ')'
        else:
            # called as line magic: [fig] @ label or just label, uses f
            if line.find('@') >= 0:
                sline = line.split('@')
                f = sline[0].strip()
                label = sline[1].strip()
            else:
                f = 'f'
                label = line.strip()
            s = f'promise = PM.figure({f}, "{label}", tacit=True, promise=True)'
        logger.debug(s)
        if args.command:
            self.load_cell(s, line, cell)
        else:
            if args.debug:
                self.shell.ev('PM.toggle_debug("on")')
            self.shell.ex(s)
            if args.debug:
                self.shell.ev('PM.toggle_debug("off")')

    @cell_magic
    @magic_arguments('%pmsave')
    @argument('-f', '--fstring', action='store_true', help='Convert caption into f string and evaluate.')
    @argument('-i', '--ignore', action='store_true', help='Ignore the cell, turns into a comment')
    @argument('-o', '--option', action='store_true',
              help='Set option for tables that vary with options, uses decorated top_value filename.')
    @argument('-v', '--variable', type=str, default='', help='Variable name for output, default promise')
    def pmsave(self, line, cell):
        """
        save the contents of the cell into a file
        e.g. with diagram get tikz blobs of text, write them to a file and return the promise
        %%pmsave
        filename: e.g. dir stem --> dir/KEY-stem.md
        contents (either f string or not)

        sets promise = the relevant include string
        nothing is written to any buffer....that's up to you
        :param line:
        :param cell:
        :return:
        """
        args = parse_argstring(self.pmsave, line)
        if args.ignore:
            return
        # manipulation common to both engines
        logger.debug(args)
        stxt = re.sub('\n+', '\n', cell, re.MULTILINE).strip().split('\n')
        assert len(stxt) >= 2
        dir, *filestem = stxt[0].strip().split(' ')
        filestem = ' '.join(filestem)
        print(dir, filestem)
        body = stxt[1:]

        if args.fstring:
            logger.info('evaluating f strings')
            for i, s in enumerate(body):
                body[i] = self.shell.ev(f'f"""{s}"""')
        body = '\n'.join(body)
        var = args.variable if args.variable != '' else 'promise'

        s = f'{var} = PM.save("{dir}", "{filestem}", """{body}""", {args.option})'
        logger.debug(s[:100])
        self.shell.ex(s)

            #
            # if args.fstring:
            #     logger.info('evaluating as f string')
            #     if args.code:
            #         temp = f'f"""\\\\\\\\footnotesize\n\n```python\n{cell}\n```\n\\\\\\\\normalsize"""'
            #     else:
            #         temp = f'f"""{cell}"""'
            #     cell = self.shell.ev(temp)
            # self.shell.ev(f'PM.text("""{cell}""", buf="{buf}", tacit={args.tacit})')


    @line_cell_magic
    @magic_arguments('%pmt')
    @argument('-a', '--appendix', action='store_true', help='Mark as appendix material.')
    @argument('-b', '--tabs', type=str, default='', help='Set tabs.')
    @argument('-c', '--command', action='store_true', help='Load the underlying command into the current cell')
    @argument('-d', '--debug', action='store_true', help='Turn debug on temporarilty .')
    @argument('-f', '--fstring', action='store_true', help='Convert caption into f string and evaluate.')
    @argument('-h', '--hrule', type=str, default=None, help='Horizontal rule locations eg 1,3,-1')
    @argument('-i', '--ignore', action='store_true', help='Ignore the cell, turns into a comment')
    @argument('-m', '--summary', action='store_true', help='Mark as summary material (exlusive with appendix).')
    @argument('-n', '--new_slide', action='store_false', help='Set to suppress new slide.')
    @argument('-o', '--option', action='store_true',
              help='Set option for tables that vary with options, uses decorated top_value filename.')
    @argument('-p', '--promise', action='store_true', help='Promise: write file but append nothing to stream')
    @argument('-q', '--equal', action='store_true', help='Hint the column widths should be equal')
    @argument('-w', '--wide', type=int, default=0, help='Use wide table mode, WIDE number of columns')
    @argument('-r', '--format', action='store_true', help='Indicates the last line of input is a float_format function')
    @argument('-s', '--scale', type=float, default="M", help='Scale for tikz')
    @argument('-t', '--tacit', action='store_true', help='tacit: suppress output as Markdown.')
    @argument('-u', '--underscore', action='store_true', help='Apply de_underscore prior to formatting.')
    @argument('-v', '--vrule', type=str, default=None, help='Vertical rule locations, to left of col no')
    def pmt(self, line, cell=None):
        """
        PresentationManager table utility. Add a new table to the stream, format and save as TeX file.

        In line mode

            %pmt df @ label
            %pmt label         # assumes table called df

        In cell mode

            %%pmt [options
            df
            label text
            many lines of caption text
            caption continues

        tabs.


        """
        self.shell.ex('if PM.debug: logger.warning(f"PM debug mode set to {PM.debug}")')
        if cell:
            args = parse_argstring(self.pmt, line)
            if args.ignore:
                return
            # manipulation common to both engines
            logger.debug(args)
            buf = 'body'
            if args.appendix: buf = 'appendix'
            if args.summary: buf = 'summary'
            # get rid of multiple new lines
            stxt = re.sub('\n+', '\n', cell, re.MULTILINE).strip().split('\n')
            assert len(stxt) >= 2
            df = stxt[0].strip()
            label = stxt[1].strip()
            if args.format:
                ff = stxt.pop().strip()
            if len(stxt) > 2:
                caption = '\n'.join(i.strip() for i in stxt[2:])
                if args.fstring:
                    logger.info('evaluating caption as f string')
                    temp = f'f"""{caption}"""'
                    caption = self.shell.ev(temp)
            else:
                caption = ""
            logger.debug(caption)
            hrule = args.hrule
            vrule = args.vrule
            equal = args.equal
            option = args.option
            tabs = args.tabs
            if tabs != '':
                tabs = [float(i) for i in tabs.split(',')]
            if hrule:
                hrule = [int(i) for i in hrule.split(',') if i.isnumeric()]
            if vrule:
                vrule = [int(i) for i in vrule.split(',') if i.isnumeric()]
            try:
                scale = float(args.scale)
            except ValueError:
                logger.info(f'Error converting args.scale {args.scale} to float; valid options')
                scale = 0.5
            if args.underscore:
                s = f'promise = PM.tikz_table(grt.de_underscore({df}), '
            else:
                s = f'promise = PM.tikz_table({df}, '
            s += (  f'"{label}", buf="{buf}", caption="""{caption}""", '
                    f'new_slide={args.new_slide}, ' 
                    f'tacit={args.tacit}, promise={args.promise}, ' 
                    f'hrule={hrule}, vrule={vrule}, scale={scale}, ' 
                    f'equal={equal}, '
                    f'option={option}, '
                    'sparsify=1, figure="table"' )
            if type(tabs) == list:
                s += f', tabs={tabs} '
            if args.format:
                s += f''', float_format={ff}'''
            s += ')'
        else:
            # called as line magic: [table] @ label or just label, uses bit
            if line.find('@') >= 0:
                sline = line.split('@')
                df = sline[0].strip()
                label = sline[1].strip()
            else:
                df = 'bit'
                label = line.strip()
            s = f'promise = PM.table({df}, "{label}", tacit=True, promise=True)'

        logger.debug(f'command:\n\n{s}\n\n')
        if args.command:
            self.load_cell(s, line, cell)
        else:
            if args.debug:
                self.shell.ev('PM.toggle_debug("on")')
            self.shell.ex(s)
            if args.debug:
                self.shell.ev('PM.toggle_debug("off")')

    def load_cell(self, s, line, cell):
        """
        load s into the cell in a nicely formatted manner

        :param s:
        :return:
        """
        sps = s.split(',')
        n = sps[0].find('(') + 1
        cmd = sps[0][:n]
        space = ' ' * n
        s = f'{cmd}\n{space} {sps[0][n:]},\n{space}'
        s += (f',\n{space}').join(sps[1:])
        contents = f'# %%pmt {line}\n# ' + '\n# '.join(cell.strip().split('\n')) + '\n' + s
        self.shell.set_next_input(contents, replace=True)

    @line_magic
    def pmbuf(self, line=''):
        self.shell.ev('print(PM.buffer())')

    @cell_magic
    def qc(self, line, cell):
        """
        quick calculation: skip calculation if the variables listed in line
        exist in globals.

        use responsibly to avoid recomputing expensive items!
        """
        if line:
            sline = line.split(' ')
            status = [c in self.shell.user_ns for c in sline]
            if np.all(status):
                logger.info(f'Skipping recalc: all of {line} variables exist.')
            else:
                logger.info('Recomputing cell: missing variables.')
                self.shell.ex(cell)


def _sparsify(col):
    """
    sparsify col values, col a pd.Series or dict, with items and accessor
    column results from a reset_index so has index 0,1,2... this is relied upon.
    """
    last = col[0]
    new_col = col.copy()
    rules = []
    for k, v in col[1:].items():
        if v == last:
            new_col[k] = ''
        else:
            last = v
            rules.append(k-1)
            new_col[k] = v
    return new_col, rules


def _sparsify_mi(mi):
    """
    as above for a multi index level, without the benefit of the index...
    really all should use this function
    :param mi:
    :return:
    """
    last = mi[0]
    new_col = list(mi)
    rules = []
    for k, v in enumerate(new_col[1:]):
        if v == last:
            new_col[k+1] = ''
        else:
            last = v
            rules.append(k+1)
            new_col[k+1] = v
    return new_col, rules


def df_to_tikz(df, *, fn_out=None, float_format=None, tabs=None,
               show_index=True, scale=0.717, column_sep=3/8, row_sep=1/8,
               figure='figure', extra_defs='', hrule=None, equal=False,
               vrule=None, post_process='', label='', caption='',
               sparsify=1, clean_index=False):
    """
    Write DataFrame to custom tikz matrix to allow greater control of
    formatting and insertion of horizontal divider lines

    Estimates tabs from text width of fields (not so great if includes TeX);
    manual override available. Tabs gives the widths of each field in
    em (width of M)

    Standard row height = 1.5em seems to work - set in meta

    first and last thick rules by default
    others below (Python, zero-based) row number, excluding title row

    keyword arguments : value (no newlines in value) escape back slashes!
    #keyword... rows ignored
    passed in as a string to facilitate using them with %%pmt?

    Rules:
    hrule at i means below row i of the table. (1-based) Top, bottom and below index lines
    are inserted automatically. Top and bottom lines are thicker.

    vrule at i means to the left of table column i (1-based); there will never be a rule to the far
    right...it looks plebby; remember you must include the index columns!

    sparsify  number of cols of multi index to sparsify

    Issue: colun with floats and spaces or missing causess problems (VaR, TVaR, EPD, mean and CV table)

    keyword args:
        scale           scale applied to whole table - default 0.717
        height          row height, rec. 1 (em)
        column_sep      col sep in em
        row_sep         row sep in em
        figure          table, figure or sidewaysfigure
        color           color for text boxes (helps debugging)
        extra_defs      TeX defintions and commands put at top of table,
                        e.g., \\centering
        lines           lines below these rows, -1 for next to last row etc.; list of ints
        post_process    e.g., non-line commands put at bottom of table
        label
        caption         text for caption

    Original version see: C:\\S\\TELOS\\CAS\\AR_Min_Bias\\cvs_to_md.py

    :param df:
    :param fn_out:
    :param float_format:
    :param tabs:
    :param show_index:
    :param scale:
    :param column_sep:
    :param row_sep:
    :param figure:
    :param color:
    :param extra_defs:
    :param lines:
    :param post_process:
    :param label:
    :param caption:
    :return:


    """

    header = """
\\begin{{{figure}}}
\\centering
{extra_defs}
\\begin{{tikzpicture}}[
    auto,
    transform shape,
    nosep/.style={{inner sep=0}},
    table/.style={{
        matrix of nodes,
        row sep={row_sep}em,
        column sep={column_sep}em,
        nodes in empty cells,
        nodes={{rectangle, scale={scale}, text badly ragged}},
"""
    # put draw=blue!10 or so in nodes to see the node


    footer = """
{post_process}

\\end{{tikzpicture}}
{caption}
\\end{{{figure}}}
"""

    # make a safe float format
    if float_format is None:
        wfloat_format = PresentationManager.default_float_format
    else:
        # If you pass in a lambda function it won't have error handling
        def _ff(x):
            try:
                return float_format(x)
            except:
                return x
        wfloat_format = _ff

    if clean_index:
        # dont use the index
        # but you do use the columns, this does both
        # logger.debug(df.columns)
        df = PresentationManager.clean_index(df)
        # logger.debug(df.columns)

    # index
    if show_index:
        if isinstance(df.index, pd.MultiIndex):
            nc_index = len(df.index.levels)
            # df = df.copy().reset_index(drop=False, col_level=df.columns.nlevels - 1)
        else:
            nc_index = 1
        df = df.copy().reset_index(drop=False, col_level=df.columns.nlevels - 1)
        if sparsify:
            if hrule is None:
                hrule = set()
        for i in range(sparsify):
            df.iloc[:, i], rules = _sparsify(df.iloc[:, i])
            # print(rules, len(rules), len(df))
            # don't want lines everywhere
            if len(rules) < len(df) - 1:
                hrule = set(hrule).union(rules)
    else:
        nc_index = 0

    # this is handled by the caller
    # df = df.astype(float, errors='ignore')

    if nc_index:
        if vrule is None:
            vrule = set()
        else:
            vrule = set(vrule)
        # to the left of... +1
        vrule.add(nc_index + 1)

    if isinstance(df.columns, pd.MultiIndex):
        nr_columns = len(df.columns.levels)
    else:
        nr_columns = 1
    logger.debug(f'rows in columns {nr_columns}, cols in index {nc_index}')

    # internal TeX code
    matrix_name = hex(abs(hash(str(df))))

    # note this happens AFTER you have reset the index...need to pass number of index columns
    colw, mxmn, tabs = guess_column_widths(df, nc_index=nc_index, float_format=wfloat_format, tabs=tabs,
                                           scale=scale, equal=equal)
    # print(colw, tabs)
    logger.debug(f'tabs: {tabs}')

    # alignment dictionaries
    ad = {'l': 'left', 'r': 'right', 'c': 'center'}
    ad2 = {'l': '<', 'r': '>', 'c': '^'}
    # guess alignments: TODO add dates?
    align = []
    for n, i in zip(df.columns, df.dtypes):
        x, n = mxmn[n]
        if x == n and len(align) == 0:
            align.append('l')
        elif i == object and x == n:
            align.append('c')
        elif i == object:
            align.append('l')
        else:
            align.append('r')
    logger.debug(align)

    # start writing
    sio = StringIO()
    sio.write(header.format(figure=figure, extra_defs=extra_defs, scale=scale, column_sep=column_sep, row_sep=row_sep))

    # table header
    # title rows, start with the empty spacer row
    i = 1
    sio.write(f'\trow {i}/.style={{nodes={{text=black, anchor=north, inner ysep=0, text height=0, text depth=0}}}},\n')
    for i in range(2, nr_columns+2):
        sio.write(f'\trow {i}/.style={{nodes={{text=black, anchor=south, inner ysep=.2em, minimum height=1.3em, font=\\bfseries}}}},\n')

    # write column spec
    for i, w, al in zip(range(1, len(align)+1), tabs, align):
        # average char is only 0.48 of M
        # https://en.wikipedia.org/wiki/Em_(gtypography)
        if i == 1:
            # first column sets row height for entire row
            sio.write(f'\tcolumn {i:>2d}/.style={{' 
                  f'nodes={{align={ad[al]:<6s}}}, text height=0.9em, text depth=0.2em, '
                  f'inner xsep={column_sep}em, inner ysep=0, '
                  f'text width={max(2, 0.6 * w):.2f}em}},\n')
        else:
            sio.write(f'\tcolumn {i:>2d}/.style={{' 
                  f'nodes={{align={ad[al]:<6s}}}, nosep, text width={max(2, 0.6 * w):.2f}em}},\n')
    # extra col to right which enforces row height
    sio.write(f'\tcolumn {i+1:>2d}/.style={{text height=0.9em, text depth=0.2em, nosep, text width=0em}}')
    sio.write('\t}]\n')

    sio.write("\\matrix ({matrix_name}) [table, ampersand replacement=\\&]{{\n".format(matrix_name=matrix_name))

    # body of table, starting with the column headers
    # spacer row
    nl = ''
    for cn, al in zip(df.columns, align):
        s = f'{nl} {{cell:{ad2[al]}{colw[cn]}s}} '
        nl = '\\&'
        sio.write(s.format(cell=' '))
    # include the blank extra last column
    sio.write('\\& \\\\\n')
    # write header rows  (again, issues with multi index)
    mi_vrules = {}
    sparse_columns = {}
    if isinstance(df.columns, pd.MultiIndex):
        for lvl in range(len(df.columns.levels)):
            nl = ''
            sparse_columns[lvl], mi_vrules[lvl] = _sparsify_mi(df.columns.get_level_values(lvl))
            for cn, c, al in zip(df.columns, sparse_columns[lvl], align):
                c = wfloat_format(c)
                s = f'{nl} {{cell:{ad2[al]}{colw[cn]}s}} '
                nl = '\\&'
                sio.write(s.format(cell=c + '\\grtspacer'))
            # include the blank extra last column
            sio.write('\\& \\\\\n')
    else:
        nl = ''
        for c, al in zip(df.columns, align):
            c = wfloat_format(c)
            s = f'{nl} {{cell:{ad2[al]}{colw[c]}s}} '
            nl = '\\&'
            sio.write(s.format(cell=c + '\\grtspacer'))
        sio.write('\\& \\\\\n')

    # write table entries
    for idx, row in df.iterrows():
        nl = ''
        for c, cell, al in zip(df.columns, row, align):
            cell = wfloat_format(cell)
            s = f'{nl} {{cell:{ad2[al]}{colw[c]}s}} '
            nl = '\\&'
            sio.write(s.format(cell=cell))
            # if c=='p':
            #     print('COLp', cell, type(cell), s, s.format(cell=cell))
        sio.write('\\& \\\\\n')
    sio.write(f'}};\n\n')

    # decorations and post processing - horizontal and vertical lines
    nr, nc = df.shape
    # add for the index and the last row plus 1 for the added spacer row at the top
    nr += nr_columns + 1
    # always include top and bottom
    # you input a table row number and get a line below it; it is implemented as a line ABOVE the next row
    # function to convert row numbers to TeX table format (edge case on last row -1 is nr and is caught, -2
    # is below second to last row = above last row)
    # shift down extra 1 for the spacer row at the top
    python_2_tex = lambda x: x + nr_columns + 2 if x >= 0 else nr + x + 3
    tb_rules = [nr_columns + 1, python_2_tex(-1)]
    if hrule:
        hrule = set(map(python_2_tex, hrule)).union(tb_rules)
    else:
        hrule = list(tb_rules)
    logger.debug(f'hlines: {hrule}')

    # why
    yshift = row_sep / 2
    xshift = -column_sep / 2
    descender_proportion = 0.25

    # top rule is special
    ls = 'thick'
    ln = 1
    sio.write(f'\\path[draw, {ls}] ({matrix_name}-{ln}-1.south west)  -- ({matrix_name}-{ln}-{nc+1}.south east);\n')

    for ln in hrule:
        ls = 'thick' if ln == nr + nr_columns + 1 else ('semithick' if ln == 1 + nr_columns else 'very thin')
        if ln < nr:
            # line above TeX row ln+1 that exists
            sio.write(f'\\path[draw, {ls}] ([yshift={-yshift}em]{matrix_name}-{ln}-1.south west)  -- '
                      f'([yshift={-yshift}em]{matrix_name}-{ln}-{nc+1}.south east);\n')
        else:
            # line above row below bottom = line below last row
            # descenders are 200 to 300 below baseline
            ln = nr
            sio.write(f'\\path[draw, thick] ([yshift={-descender_proportion-yshift}em]{matrix_name}-{ln}-1.base west)  -- '
                      f'([yshift={-descender_proportion-yshift}em]{matrix_name}-{ln}-{nc+1}.base east);\n')

    # if multi index put in lines within the index TODO make this better!
    if nr_columns > 1:
        for ln in range(2, nr_columns+1):
            sio.write(f'\\path[draw, very thin] ([xshift={xshift}em, yshift={-yshift}em]'
                      f'{matrix_name}-{ln}-{nc_index+1}.south west)  -- '
                      f'([yshift={-yshift}em]{matrix_name}-{ln}-{nc+1}.south east);\n')

    written = set(range(1, nc_index+1))
    if vrule:
        # to left of col, 1 based, includes index
        # write these first
        # TODO fix madness vrule is to the left, mi_vrules are to the right...
        ls = 'very thin'
        for cn in vrule:
            if cn not in written:
                sio.write(f'\\path[draw, {ls}] ([xshift={xshift}em]{matrix_name}-1-{cn}.south west)  -- '
                      f'([yshift={-descender_proportion-yshift}em, xshift={xshift}em]{matrix_name}-{nr}-{cn}.base west);\n')
                written.add(cn-1)

    if len(mi_vrules) > 0:
        logger.debug(f'Generated vlines {mi_vrules}; already written {written}')
        # vertical rules for the multi index
        # these go to the RIGHT of the relevant column and reflect the index columns already
        # mi_vrules = {level of index: [list of vrule columns]
        # written keeps track of which vrules have been done already; start by cutting out the index columns
        ls = 'ultra thin'
        for k, cols in mi_vrules.items():
            # don't write the lowest level
            if k == len(mi_vrules) - 1:
                break
            for cn in cols:
                if cn in written:
                    pass
                else:
                    written.add(cn)
                    top = k + 1
                    if top == 0:
                        sio.write(f'\\path[draw, {ls}] ([xshift={-xshift}em]{matrix_name}-{top}-{cn}.south east)  -- '
                          f'([yshift={-descender_proportion-yshift}em, xshift={-xshift}em]{matrix_name}-{nr}-{cn}.base east);\n')
                    else:
                        sio.write(f'\\path[draw, {ls}] ([xshift={-xshift}em, yshift={-yshift}em]{matrix_name}-{top}-{cn}.south east)  -- '
                          f'([yshift={-descender_proportion-yshift}em, xshift={-xshift}em]{matrix_name}-{nr}-{cn}.base east);\n')

    if label == '':
        lt = ''
        label = '}  % no label'
    else:
        lt = label
        label = f'\\label{{tab:{label}}}'
    if caption == '':
        if label != '':
            logger.warning(f'You have a label but no caption; the label {label} will be ignored.')
        caption = '% caption placeholder'
    else:
        caption = f'\\caption{{{caption} {label}}}'
    sio.write(footer.format(figure=figure, post_process=post_process, caption=caption))

    if isinstance(fn_out, Path):
        fout = fn_out
    elif fn_out is not None:
        fout = Path(fn_out)
    else:
        fout = None

    if fout:
        with fout.open('w', encoding='utf-8') as f:
            f.write(sio.getvalue())

    return sio.getvalue()


def guess_column_widths(df, nc_index, float_format, tabs=None, scale=1, equal=False):
    """
    estimate sensible column widths for the dataframe [in what units?]

    :param df:
    :param nc_index: number of columns in the index...these are not counted as "data columns"
    :param float_format:
    :param tabs:
    :return:
        colw   affects how the table is printed in the md file (actual width of data elements)
        mxmn   affects aligmnent: are all columns the same width?
        tabs   affecets the actual output
    """

    # this
    # tabs from _tabs, an estimate column widths, determines the size of the table columns as displayed
    colw = dict.fromkeys(df.columns,  0)
    headw = dict.fromkeys(df.columns,  0)
    _tabs = []
    mxmn = {}
    nl = nc_index
    for i, c in enumerate(df.columns):
        # figure width of the column labels; if index c= str, if MI then c = tuple
        # cw is the width of the column header/title
        if type(c) == str:
            if i < nl:
                cw = len(c)
            else:
                # for data columns look at words rather than whole phrase
                cw = max(map(len, c.split(' ')))
                # logger.info(f'leng col = {len(c)}, longest word = {cw}')
        else:
            # could be float etc.
            try:
                cw = max(map(lambda x: len(float_format(x)), c))
            except TypeError:
                # not a MI, float or something
                cw = len(str(c))
        headw[c] = cw
        # now figure the width of the elements in the column
        # mxmn is used to determine whether to center the column (if all the same size)
        if df.dtypes.iloc[i] == object:
            # wierdness here were some objects actually contain floats, str evaluates to NaN
            # and picks up width zero
            try:
                # _ = list(map(lambda x: len(float_format(x)), df.iloc[:, i]))
                _ = df.iloc[:, i].map(lambda x: len(float_format(x)))
                colw[c] = _.max()
                mxmn[c] = (_.max(), _.min())
            except:
                e = sys.exc_info()[0]
                print(c, 'ERROR', e)
                logger.error(f'{c} error {e} DO SOMETHING ABOUT THIS...if it never occurs dont need the if')
                colw[c] = df[c].str.len().max()
                mxmn[c] = (df[c].str.len().max(), df[c].str.len().min())
        else:
            # _ = list(map(lambda x: len(float_format(x)), df[c]))
            _ = df.iloc[:, i].map(lambda x: len(float_format(x)))
            colw[c] = _.max()
            mxmn[c] = (_.max(), _.min())
        # debugging grief
        # if c == 'p':
        #     print(c, df[c], colw[c], mxmn[c], list(map(len, list(map(float_format, df[c])))))
    if tabs is None:
        # now know all column widths...decide what to do
        # are all the columns about the same width?
        data_cols = np.array([colw[k] for k in df.columns[nl:]])
        same_size = (data_cols.std() <= 0.1 * data_cols.mean())
        common_size = 0
        if same_size:
            common_size = int(data_cols.mean() + data_cols.std())
            logger.info(f'data cols appear same size = {common_size}')
        for i, c in enumerate(df.columns):
            if i < nl or not same_size:
                # index columns
                _tabs.append(int(max(colw[c], headw[c])))
            else:
                # data all seems about the same width
                _tabs.append(common_size)
        logger.info(f'Determined tab spacing: {_tabs}')
        if equal:
            # see if equal widths makes sense
            dt = _tabs[nl:]
            if max(dt) / sum(dt) < 4 / 3:
                _tabs = _tabs[:nl] + [max(dt)] * (len(_tabs) - nl)
                logger.info(f'Taking equal width hint: {_tabs}')
            else:
                logger.info(f'Rejecting equal width hint')
        # look to rescale, shoot for width of 150 on 100 scale basis
        data_width = sum(_tabs[nl:])
        index_width = sum(_tabs[:nl])
        target_width = 150 * scale - index_width
        if data_width / target_width < 0.9:
            # don't rescale above 1:1 - don't want too large
            rescale = min(1/scale, target_width / data_width)
            _tabs = [w if i < nl else w * rescale for i, w in enumerate(_tabs)]
            logger.info(f'Rescale {rescale} applied; tabs = {_tabs}')

        tabs = _tabs

    return colw, mxmn, tabs


def de_underscore(df, which='b'):
    """
    remove underscores from index and / or columns, replacing with space
    Note, default behavior is to quote underscores

    :param df:
    :param which: row column b[oth]
    :return:
    """
    # workers
    def de_(n):
        """
        replace underscores with space
        """
        try:
            if type(n) == str:
                return n.replace('_', ' ')
            else:
                return n
        except:
            return n

    def multi_de_(idx):
        """ de_ for multiindex """
        for i, lv in enumerate(idx.levels):
            if lv.dtype == 'object':
                repl = map(de_, lv)
                idx = idx.set_levels(repl, level=i)
        return idx

    # work
    df = df.copy()
    idx_names = df.index.names
    col_names = df.columns.names

    if which in ['both', 'b', 'columns']:
        if isinstance(df.columns, pd.core.indexes.multi.MultiIndex):
            df.columns = multi_de_(df.columns)
        else:
            df.columns = map(de_, df.columns)

    if which in ['both', 'b', 'index']:
        if isinstance(df.index, pd.core.indexes.multi.MultiIndex):
            df.index = multi_de_(df.index)
        else:
            df.index = map(de_, df.index)

    df.index.names = [de_(i) for i in idx_names]
    df.columns.names = [de_(i) for i in col_names]
    return df


def int_to_roman(num):
    val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4,
        1
        ]
    syb = [
        "M", "CM", "D", "CD",
        "C", "XC", "L", "XL",
        "X", "IX", "V", "IV",
        "I"
        ]
    roman_num = ''
    i = 0
    while num > 0:
        for _ in range(num // val[i]):
            roman_num += syb[i]
            num -= val[i]
        i += 1
    return roman_num


ip = get_ipython()
ip.register_magics(PresentationManagerMagic)

