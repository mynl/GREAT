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
from io import StringIO
from pathlib import Path
from matplotlib.pyplot import Figure
import pandas as pd
import datetime
from .markdown_make import markdown_make_main
from .utils import logger, filename
import re
import numpy as np
import unicodedata
from pandas.io.formats.format import EngFormatter
import json
from collections import OrderedDict
import subprocess
from platform import platform
from IPython.display import Markdown, display
from IPython.core.magic import Magics, magics_class, line_magic, cell_magic, line_cell_magic
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
from IPython import get_ipython
from titlecase import titlecase

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

    __yaml__ = """---
title: "{title}"
subtitle: "{subtitle}"
author: Stephen J. Mildenhall
date: "{created}"
fontsize: {font_size}pt
outertheme : metropolis
innertheme : metropolis
fonttheme  : structurebold
colortheme : orchid
institute: \\convexmark
classoption: t
numbersection: true 
toc: false
filter: pandoc-citeproc
bibliography: /S/TELOS/biblio/library.bib
csl: /S/TELOS/biblio/journal-of-finance.csl
link-citations: true
header-includes: 
    - \\input{{/s/telos/common/general2021.tex}}
    - \\geometry{{paper=legalpaper, landscape, mag=2000, truedimen}}
cla: --top-level-division=section
cla: --slide-level=2
cla: --standalone
cla: -f markdown+smart+yaml_metadata_block+citations{pdf_engine}
cla: -o pdf
cla: --filter pandoc-citeproc
cla: -t beamer
debug: true
---

"""

    def __init__(self, config_file, tops_id, tidy=False):
        """

        Originally:
        title, subtitle, key, base_dir, *,
                 top_name='', top_value='',
                 file_name='',
                 fig_format='pdf', output_style='with_table',
                 default_float_fmt=None, tidy=False,
                 default_table_font_size=0.15, tacit_override=False,
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
        :param default_table_font_size: float or zero, used as a custom font size

        """

        config = json.load(Path(f'{config_file}.config').open('r', encoding='utf-8'))
        self.tops_id = tops_id

        self.title = config['title']
        self.key = config['key']
        # tured to Path below
        base_dir = config['base_dir']
        self.top_name = config['top_name']
        self.fig_format = config['fig_format']
        self.output_style = config['output_style']
        self.pdf_engine = config['pdf_engine']

        # for this specific document
        top = config['tops'][tops_id]
        self.subtitle = top['subtitle']
        self.top_value = top['top_value']

        # no choice of filename
        self.file_name = f'{self.key}-{self.top_value}.md'

        # file specific, set manually
        self.default_float_fmt = PresentationManager.default_float_format

        # deprecated
        self.default_table_font_size = 0.15

        # main file for output and ancillary directories
        self.base_dir = Path(base_dir)
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

        # tidy up
        if tidy:
            self.tidy_up()

        # output
        self.toc = []
        self._slides = set()
        self.sios = OrderedDict(contents=StringIO(), summary=StringIO(), body=StringIO(), appendix=StringIO())

        # starts in base config: active and not in debug mode
        self._tacit_override = False
        self._active = True
        self._debug = False
        self.debug_dm_file = self.base_dir / 'dm.md'

        # start by writing the toc header
        self._write_buffer_('contents', '## Contents\n')

        # store for future use
        self.config = config
        self.config_file = config_file

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
            prefix = f'{self.top_value}.'
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
                self.toc.append((self.top_value, s_no, tl))
            else:
                # title case ## headings too
                m2 = re.findall('^## (.*)', ln)
                if m2:
                    stxt[i] = '##  ' + titlecase(ln[3:])

        txt = '\n'.join(stxt) + '\n\n'
        self._write_buffer_(buf, txt)

        if self.tacit_override or not tacit:
            display(Markdown(txt))

    # aliases
    write = text
    blob = text

    def figure(self, f, label, buf='body', caption="", height="", new_slide=True, tacit=False, promise=False, **kwargs):
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
        fig_file = self.figure_dir / f'{self.key}-{label}.{self.fig_format}'
        fig_file_local = f'img/{self.key}-{label}.{self.fig_format}'
        if fig_file.exists():
            logger.warning(f'File {fig_file} already exists...over-writing.')
        f.savefig(fig_file, **kwargs)

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
            self._write_buffer_(buf, f'## {slide_caption}\n\n')
            if not tacit:
                display(Markdown(f'## {slide_caption}'))
        if promise:
            return fig_text
        else:
            self._write_buffer_(buf, fig_text)

        if self.tacit_override or not tacit:
            display(Markdown(fig_text))

    def table(self, df, label, *, caption="", buf='body', float_format=None, fill_nan='', font_size=0.0,
              sparsify=False, force_float=False, multipart=0, new_slide=True, tacit=False, promise=False, **kwargs):
        r"""

        fontsize tiny scriptsize or 0.15 or nothing etc.
        multipart => need parts to have different file names, appended for >0 parth

        Add a tablea
        label used as file name
        stuff table in clipboard latex table, save to file...add caption, labels etc.
        rational formatting

        force_float = convert input to float first (makes a copy) for col alignment

            def sticb(self, df, float_format=None, fill_nan='', caption='caption', label='label', file_name='',
                  here='', font_size=False, sideways=False, sparsify=False, **kwargs):

        args passed to pandas to_latex
        From CEA project and Monograph SRM_Examples

        output_style
            with_table : all output in @@@ file and include in main md file; use when caption is generic
            caption:   puts caption text in the main markdown file, use when caption will be edited
            inline: all output in md file directly (not recommended)

        font_size = scriptsize, footnotesize etc.

        label and columns have _ escaped for TeX but the caption is not - so beware!

        multipart=True for widetable, then skip the beamer caption

        Test Cases
        ==========

            import great as grt
            from great.doc_maker import DocMaker
            df = grt.test_df()

            DM = doc.DocMaker(f'notes\\vig-n-tester-example.md',
                              key=f'vig-n',
                              title='All Different Table Options',
                              tidy=True)

            DM.text('Writing test examples of all combinations to file. ')
            j = 0
            sideways = False
            for fs in ['normalsize', 'footnotesize', 'scriptsize', 'tiny']:
                    for output_style in ['with_table', 'inline', 'caption']:
                        j += 1
                        DM.table(df, f'test-{j}',
                                 f'Caption 1 with settings font-size={fs}, sideways={sideways}, output-style={output_style.replace("_", "-")}',
                                 font_size=fs, sideways=sideways, output_style=output_style)

            sideways = True
            df = grt.test_df(ncols=20)

            # reveals there is  no tiny!
            for fs in ['normalsize', 'footnotesize', 'scriptsize', 'tiny']:
                j += 1
                DM.table(df, f'test-{j}',
                         f'Caption 1 with settings font-size={fs}, sideways={sideways}',
                         font_size=fs, sideways=sideways)


            DM.process()

        Parameters
        ==========

        :param promise:
        :param new_slide:
        :param buf:
        :param tacit:
        :param multipart:
        :param df:
        :param label:
        :param caption:
        :param float_format:
        :param fill_nan:
        :param here:
        :param font_size:
        :param custom_font_size:  e.g.  input the size 0.15, second size will
            be scaled up appropriately. Overrides font_size. \fontsize{0.15cm}{0.170cm}\selectfont
        :param sideways:
        :param sparsify:
        :param force_float:
        :param output_style:
        Can be None (default) or first, mid, last.
        :param kwargs:
        :return:

        """

        assert not promise or self.output_style == 'with_table'

        if not self.active:
            if not tacit:
                display(df)
            return

        if float_format is None:
            float_format = self.default_float_fmt

        df = df.copy()
        if force_float:
            df = df.astype(float, errors='ignore')

        # have to handle column names that may include _
        # For now assume there are not TeX names
        slide_caption = self.make_title(label)
        label = self.make_safe_label(label)
        label = self.clean_name(label)

        # check the caption, that can contain tex
        if not self.clean_underscores(caption):
            logger.warning('Caption contains unescaped underscores _. You must ensure they are in dollars.')

        df = self.clean_index(df)

        s = df.to_latex(float_format=float_format, sparsify=sparsify, escape=False, **kwargs)
        s = s.replace('nan', fill_nan)

        if type(font_size) in [int, float, np.int, np.float]:
            if font_size:
                font_size = f'\\fontsize{{{font_size}cm}}{{{font_size}cm}}\\selectfont\n'
                ends_font_size = '\\normalsize\n'
            else:
                font_size = f'\\fontsize{{{self.default_table_font_size}cm}}{{{self.default_table_font_size}cm}}\\selectfont\n'
                ends_font_size = '\\normalsize\n'
        elif font_size != '':
            font_size = f'\\{font_size}\n'
            ends_font_size = '\\normalsize\n'
        else:
            ends_font_size = ''

        s_md_pre = f'\n\\begin{{table}}\n{font_size} %\n'
        if caption != '':
            s_md_pre += f'\\caption{{{caption}}}\n'
        if label != '':
            s_md_pre += f'\\label{{tab:{label}}}\n'

        s_table = f'\\medskip\n\\centering\n{s}'
        s_md_post = f'{ends_font_size}\\end{{table}}\n\n'

        suffix = f'-{multipart}' if multipart else ''
        table_file = self.table_dir / f'{self.key}-{label}{suffix}.md'
        table_file_local = f'table/{self.key}-{label}{suffix}.md'

        if new_slide and not promise:
            self._write_buffer_(buf, f'## {slide_caption}\n\n')
            if not tacit:
                display(Markdown(f'## {slide_caption}'))

        if self.output_style == 'caption':
            with table_file.open('w', encoding='utf-8') as f:
                self._write_buffer_(buf, '\n')
                self._write_buffer_(buf, s_table)
            self._write_buffer_(buf, s_md_pre)
            self._write_buffer_(buf, f'\n@@@include {table_file_local}\n\n')
            self._write_buffer_(buf, s_md_post)

        elif (self.output_style == 'inline') or (self.output_style == 'in-line'):
            self._write_buffer_(buf, s_md_pre)
            self._write_buffer_(buf, s_table)
            self._write_buffer_(buf, s_md_post)

        elif self.output_style == 'with_table':
            with table_file.open('w', encoding='utf-8') as f:
                f.write(s_md_pre)
                f.write(s_table)
                f.write(s_md_post)
            if promise:
                return f'@@@include {table_file_local}\n\n'
            else:
                self._write_buffer_(buf, f'@@@include {table_file_local}\n\n')

        else:
            raise ValueError(f'Unknown option {self.output_style} for output_style passed to table.')

        if self.tacit_override or not tacit:
            display(df.style.format(float_format))
            if caption != '':
                display(Markdown(caption))

    def wide_table(self, df, label, *, buf='body', nparts=2,
                   float_format=None, fill_nan='', font_size=0.0,
                   sparsify=False, force_float=False, new_slide=True, new_slide_each_part=False,
                   tacit=False, promise=False, **kwargs):
        """

        nparts = split the table into nparts,
        new_slide for each part

        Splits df and passes to table to do the work.

        Large Tables
        ============

        Makes some attempt to split up very wide tables
        You handle long tables yourself.

        ```

        :return:

        """

        cols = df.shape[1]
        cols_per_part = cols // nparts
        if nparts * cols_per_part < cols:
            cols_per_part += 1

        ans = []
        for i, b in enumerate(range(0, cols, cols_per_part)):
            bit = df.iloc[:, b:b + cols_per_part]
            #  no captions for wide tables
            ans.append(self.table(bit, label, caption='', buf=buf, float_format=float_format, fill_nan=fill_nan,
                       font_size=font_size, sparsify=sparsify, force_float=force_float, multipart=i,
                       new_slide=(new_slide_each_part if i else new_slide), tacit=tacit, promise=promise, **kwargs))

        if promise:
            return '\n'.join(ans)

    def tikz_table(self, df, label, *, caption="", buf='body',
                   new_slide=True, tacit=False, promise=False,
                   float_format=None, tabs=None,
                   show_index=True, scale=0.717,
                   figure='figure', hrule=None,
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
                             show_index=show_index, scale=scale,
                             figure=figure, hrule=hrule, vrule=vrule, sparsify=sparsify, clean_index=True)

        table_file = self.table_dir / f'{self.key}-{label}.md'
        if table_file.exists():
            logger.warning(f'File {table_file} exists, over-wriring.')
        table_file_local = f'table/{self.key}-{label}.md'

        if new_slide and not promise:
            self._write_buffer_(buf, f'## {slide_caption}\n\n')
            if not tacit:
                display(Markdown(f'## {slide_caption}'))

        if (self.output_style == 'inline') or (self.output_style == 'in-line'):
            self._write_buffer_(buf, s_table)

        elif self.output_style == 'with_table':
            with table_file.open('w', encoding='utf-8') as f:
                f.write(s_table)
            if promise:
                return f'@@@include {table_file_local}\n\n'
            else:
                self._write_buffer_(buf, f'@@@include {table_file_local}\n\n')

        else:
            raise ValueError(f'Unknown option {self.output_style} for output_style passed to table.')

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
            logger.info('In debug mode...writing dm.md')
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
        return '\n\n'.join(buf.getvalue() for buf in self.sios.values())

    def make_toc(self, decorate=''):
        """
        sort the toc: Summary, Sections, Appendix
        if decorate != '' it will bold thaat row (for burst mode)
        :return:
        """
        # new SIO each time for contents...
        self.sios['contents'] = StringIO()
        # there are no sections in the summary (section is # level)
        # toc_summary = sorted([i for i in self.toc if i[0].split()[2].strip() not in ('Appendix', 'Section')],
        #                      key=sort_fun)
        toc_body = sorted([i for i in self.toc if i[1] > 0], key=lambda x : x[1])
        toc_appendix = sorted([i for i in self.toc if i[1] < 0], key=lambda x : x[1])
        self.sios['contents'].write(f'# Table of Contents\n## {self.top_name} {self.top_value} Contents\n\n')
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

    def buffer_persist(self, font_size=9, tacit=False, mode='single'):
        """
        persist and write out the StringIO cache to a physical file

        :param font_size:
        :param mode:   burst (separate files per section or all together or both
        :param toc:    for burst mode: read toc to number sections ignored if it doesn't exist
        :param tacit:  suppress markdown output of persisted file

        """
        if mode in ('single', 'both'):
            subtitle = f'{self.top_name} {self.top_value}: {self.subtitle}'
            y = self.__yaml__.format(
                    title=self.title, subtitle=subtitle, created=self.date(),
                    font_size=font_size, pdf_engine=self.pdf_engine)
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
                    y = self.__yaml__.format(
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

    def format_source(self, md_file, to='html', title=''):
        """
        Print out the source file to md_file and Pandoc if installed

        md_file is the name of the md (JupyText linked!) that creates the presentation.

        MUST use md linking through JupyText

        :param to: output format: html or pdf
        :param md_file:
        :return:
        """

        header = """---
title: Source Code
author: Stephen J. Mildenhall
date: created
geometry: hmargin=1in, vmargin=.75in
fontsize: 9pt
classoption: a4paper, landscape
header-includes: \\input{{/s/telos/common/general.tex}}
cla: --standalone
cla: -f markdown+smart+yaml_metadata_block+citations
cla: --pdf-engine=xelatex
cla: -o pdf
cla: --filter pandoc-citeproc
cla: -t latex
cla: --template=/S/TELOS/Common/sjm-doc-template.tex
debug: true
---

"""
        if isinstance(md_file, Path):
            p = md_file
        else:
            p = Path(md_file)
        # only working with markdown files
        if p.suffix != '.md':
            raise ValueError('format_source only works with markdown .md files')

        if to == 'html':
            CREATE_NO_WINDOW = 0x08000000
            source_output_file = self.base_dir / f'{self.key}-source.html'
            logger.info(f'Making {source_output_file}')
            css = filename('S/TELOS/PIRC/css/github.css')
            if title == "":
                title = p.stem
            command = f'pandoc -f markdown -t html -o {source_output_file.resolve()} -s ' \
                        f'--metadata title="{title}" '  \
                        f'--css={css} {p.resolve()}'
            logger.info(f'Executing: {command}')
            if platform()[:5] == 'Linux':
                p = subprocess.Popen(command, stdout=subprocess.PIPE)
            else:
                p = subprocess.Popen(command, creationflags=CREATE_NO_WINDOW, stdout=subprocess.PIPE)
            p.communicate()

        elif to == 'pdf':
            with p.open('r', encoding='utf-8') as f:
                txt = f.read()
            source_output_file = self.base_dir / f'{self.key}-source.md'
            with source_output_file.open('w', encoding='utf-8') as f:
                f.write(header + txt)
            self.make_beamer(source_output_file)
        else:

            raise ValueError(f'Unknown option to = {to}; pdf and html are valid options.')

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
    @argument('-f', '--fstring', action='store_true', help='Convert cell into f string and evaluate.')
    @argument('-t', '--tacit', action='store_true', help='Tacit: suppress output as Markdown')
    @argument('-s', '--summary', action='store_true', help='Mark as summary material (exlusive with appendix).')
    def pmb(self, line='', cell=None):
        """
        PresentationManager blob (text/write) line/cell magic

        %pmt line  -> written to body
        %%pmt -s -a -m  (s show; a=appendix, m=suMmary)

        """
        if cell is None:
            # defaults, tacit for everything except sections
            if line.strip()[0:2] == '# ':
                tacit = False
            else:
                tacit = True
            self.shell.ev(f'PM.text("{line}", buf="body", tacit={tacit})')
        else:
            args = parse_argstring(self.pmb, line)
            logger.info(args)
            buf = 'body'
            if args.appendix: buf = 'appendix'
            if args.summary: buf = 'summary'
            if args.fstring:
                logger.info('evaluating as f string')
                if args.code:
                    temp = f'f"""\\\\\\\\footnotesize\n\n```python\n{cell}\n```"""'
                else:
                    temp = f'f"""{cell}"""'
                cell = self.shell.ev(temp)
            self.shell.ev(f'PM.text("""{cell}""", buf="{buf}", tacit={args.tacit})')

    @line_cell_magic
    @magic_arguments('%pmf')
    @argument('-a', '--appendix', action='store_true', help='Mark as appendix material.')
    @argument('-f', '--fstring', action='store_true', help='Convert caption into f string and evaluate.')
    @argument('-h', '--height', type=float, default=0.0, help='Vertical size, generates height=height clause.')
    @argument('-n', '--new_slide', action='store_false', help='Set to suppress new slide.')
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

        if cell:
            args = parse_argstring(self.pmf, line)
            logger.info(args)
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
            s = f'promise = PM.figure({f}, "{label}", buf="{buf}", caption="""{caption}""", new_slide={args.new_slide}, ' \
                f'tacit={args.tacit}, promise={args.promise}'
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
        logger.info(s)
        self.shell.ex(s)

    @line_cell_magic
    @magic_arguments('%pmt')
    @argument('-a', '--appendix', action='store_true', help='Mark as appendix material.')
    @argument('-e', '--engine', type=str, default='tikz', help='Use ENGINE to covert to md: latex or tikz (default')
    @argument('-f', '--fstring', action='store_true', help='Convert caption into f string and evaluate.')
    @argument('-h', '--hrule', type=str, default=None, help='Horizontal rule locations eg 1,3,-1')
    @argument('-m', '--summary', action='store_true', help='Mark as summary material (exlusive with appendix).')
    @argument('-n', '--new_slide', action='store_false', help='Set to suppress new slide.')
    @argument('-p', '--promise', action='store_true', help='Promise: write file but append nothing to stream')
    @argument('-w', '--wide', type=int, default=0, help='Use wide table mode, WIDE number of columns')
    @argument('-r', '--format', action='store_true', help='Indicates the last line of input is a float_format function')
    @argument('-s', '--scale', type=float, default="M", help='Scale for tikz')
    @argument('-t', '--tacit', action='store_true', help='tacit: suppress output as Markdown.')
    @argument('-u', '--underscore', action='store_true', help='Apply de_underscore prior to formatting.')
    @argument('-v', '--vrule', type=str, default=None, help='Vertical rule locations, to left of col no')
    @argument('-z', '--fontsize', type=str, default='', help='Fontsize: tiny, scriptsize, footnotesize, or number '
                                                         '(e.g., 0.15) for custom font size etc.')
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
            caption continues.


        """
        if cell:
            args = parse_argstring(self.pmt, line)
            # manipulation common to both engines
            logger.info(args)
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
            logger.info(caption)
            #
            # Engine specific
            if args.engine == 'latex':
                if args.wide:
                    # wide tables don't have captions.
                    s = f'promise = PM.wide_table({df}, "{label}", buf="{buf}", ' \
                            f'new_slide={args.new_slide}, nparts={args.wide}, '  \
                            f'tacit={args.tacit}, promise={args.promise}'
                else:
                    s = f'promise = PM.table({df}, "{label}", buf="{buf}", caption="""{caption}""", ' \
                            f'new_slide={args.new_slide}, ' \
                            f'tacit={args.tacit}, promise={args.promise}'
                if args.fontsize != '':
                    if np.all([i in '0123456789.' for i in args.fontsize]):
                        s += f', fontsize={args.fontsize}'
                    else:
                        s += f', fontsize="{args.font_size}"'
                if args.format:
                    s += f''', float_format={ff}'''
                s += ')'
            elif args.engine == 'tikz':
                hrule = args.hrule
                vrule = args.vrule
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
                s += f'"{label}", buf="{buf}", caption="""{caption}""", '\
                     f'new_slide={args.new_slide}, ' \
                     f'tacit={args.tacit}, promise={args.promise}, ' \
                     f'hrule={hrule}, vrule={vrule}, scale={scale}, ' \
                     'sparsify=1, figure="table" '
                if args.format:
                    s += f''', float_format={ff}'''
                s += ')'
            else:
                raise ValueError(f'Inadmissible enging {args.engine}: options tikz or latex')
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

        logger.info(s)
        self.shell.ex(s)

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
                logger.info(f'All of {line} variables exist...skipping recalc.')
            else:
                logger.info('Missing variables..recomputing cell.')
                self.shell.ex(cell)



def _sparsify(col):
    """
    sparsify col values, col a pd.Series
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
            rules.append(k)
            new_col[k] = v
    return new_col, rules


def df_to_tikz(df, *, fn_out=None, float_format=None, tabs=None,
               show_index=True, scale=0.717, height=1, column_sep=1/2, row_sep=1/8,
               figure='figure', color='blue!0', extra_defs='', hrule=None,
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
    :param height:
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
    table/.style={{
        matrix of nodes,
        row sep={row_sep}em,
        column sep={column_sep}em,
        nodes in empty cells,
        nodes={{draw={color}, rectangle, scale={scale}, text height={height}em, text badly ragged}},
"""

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
        else:
            nc_index = 1
        df = df.copy().reset_index(drop=False)
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

    # this affects how the table is printed in the md file
    # tabs from _tabs, an estimate column widths, determines the size of the table columns as displayed
    colw = dict.fromkeys(df.columns,  0)
    headw = dict.fromkeys(df.columns,  0)
    _tabs = []
    mxmn = {}
    nl = df.index.nlevels
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
                cw = max(map(lambda x: len(wfloat_format(x)), c))
            except TypeError:
                # not a MI, float or something
                cw = len(str(c))
        headw[c] = cw
        # now figure the width of the elements in the column
        # mxmn is used to determine whether to center the column (if all the same size)
        if df.dtypes[c] == object:
            # 1.2 allows more capitals in the heading
            # colw[c] = max(0, max(df[c].str.len().max(), cw))
            colw[c] = df[c].str.len().max()
            mxmn[c] = (df[c].str.len().max(), df[c].str.len().min())
        else:
            _ = list(map(lambda x: len(wfloat_format(x)), df[c]))
            # colw[c] = max(0, max(max(_), cw))
            colw[c] = max(_)
            mxmn[c] = (max(_), min(_))

    if tabs is None:
        # now know all column widths...decide what to do
        # are all the columns about the same width?
        data_cols = np.array([colw[k] for k in df.columns[nl:]])
        same_size = (data_cols.std() <= 0.1 * data_cols.mean())
        common_size = 0
        if same_size:
            common_size = data_cols.mean() + data_cols.std()
            logger.info(f'data cols appear same size = {common_size}')
        for i, c in enumerate(df.columns):
            if i < nl or not same_size:
                # index columns
                _tabs.append(max(colw[c], headw[c]))
            else:
                # data all seems about the same width
                _tabs.append(common_size)
        print(_tabs)
        tabs = _tabs

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
    sio.write(header.format(figure=figure, extra_defs=extra_defs, color=color,
                            scale=scale, height=height, column_sep=column_sep,
                            row_sep=row_sep))

    # table header
    # title rows
    for i in range(1, nr_columns+1):
        sio.write(f'\trow {i}/.style={{nodes={{text=black, font=\\bfseries}}}},\n')

    # write column spec
    for i, w, al in zip(range(1, len(align)+1), tabs, align):
        # average char is only 0.48 of M
        # https://en.wikipedia.org/wiki/Em_(gtypography)
        sio.write(f'\tcolumn {i}/.style={{' # baseline, '
                  f'nodes={{align={ad[al]:<6s}}}, text width={max(2, 0.6 * w):.2f}em, '
                  f'inner xsep=0, inner ysep=0}},\n')
    sio.write('\t}]\n')

    sio.write("\\matrix ({matrix_name}) [table, ampersand replacement=\\&]{{\n".format(matrix_name=matrix_name))

    # body of table, starting with the column headers
    # write header rows  (again, issues with multi index)
    if isinstance(df.columns, pd.MultiIndex):
        for lvl in range(len(df.columns.levels)):
            nl = ''
            for cn, c, al in zip(df.columns, df.columns.get_level_values(lvl), align):
                c = wfloat_format(c)
                s = f'{nl} {{cell:{ad2[al]}{colw[cn]}s}} '
                nl = '\\&'
                sio.write(s.format(cell=c))
            sio.write('\\\\\n')
    else:
        nl = ''
        for c, al in zip(df.columns, align):
            c = wfloat_format(c)
            s = f'{nl} {{cell:{ad2[al]}{colw[c]}s}} '
            nl = '\\&'
            sio.write(s.format(cell=c))
        sio.write('\\\\\n')

    # write table entries
    for idx, row in df.iterrows():
        nl = ''
        for c, cell, al in zip(df.columns, row, align):
            cell = wfloat_format(cell)
            s = f'{nl} {{cell:{ad2[al]}{colw[c]}s}} '
            nl = '\\&'
            sio.write(s.format(cell=cell))
        sio.write('\\\\\n')
    sio.write(f'}};\n\n')

    # decorations and post processing - horizontal and vertical lines
    nr, nc = df.shape
    # add for the index and the last row
    nr += nr_columns
    # always include top and bottom
    # you input a table row number and get a line below it; it is implemented as a line ABOVE the next row
    # function to convert row numbers to TeX table format (edge case on last row -1 is nr and is caught, -2
    # is below second to last row = above last row)
    python_2_tex = lambda x: x + nr_columns + 1 if x >= 0 else nr + x + 2
    tb_rules = [nr_columns + 1, python_2_tex(-1)]
    if hrule:
        hrule = set(map(python_2_tex, hrule)).union(tb_rules)
    else:
        hrule = list(tb_rules)
    logger.info(f'hlines: {hrule}')

    # why
    yshift = row_sep / 2
    xshift = -column_sep / 2
    descender_proportion = 0.25

    # top rule is special
    ls = 'thick'
    ln = 1
    sio.write(f'\\path[draw, {ls}] ([yshift={yshift}em]{matrix_name}-{ln}-1.north west)  -- '
              f'([yshift={yshift}em]{matrix_name}-{ln}-{nc}.north east);\n')

    for ln in hrule:
        ls = 'thick' if ln == nr + nr_columns + 1 else ('semithick' if ln == 1 + nr_columns else 'very thin')
        if ln <= nr:
            # line above TeX row ln+1 that exists
            sio.write(f'\\path[draw, {ls}] ([yshift={yshift}em]{matrix_name}-{ln}-1.north west)  -- '
                      f'([yshift={yshift}em]{matrix_name}-{ln}-{nc}.north east);\n')
        else:
            # line above row below bottom = line below last row
            # descenders are 200 to 300 below baseline
            ln = nr
            sio.write(f'\\path[draw, thick] ([yshift={-height*descender_proportion-yshift}em]{matrix_name}-{ln}-1.base west)  -- '
                      f'([yshift={-height*descender_proportion-yshift}em]{matrix_name}-{ln}-{nc}.base east);\n')

    if vrule:
        # to left of col, 1 based, includes index
        ls = 'very thin'
        for cn in vrule:
            sio.write(f'\\path[draw, {ls}] ([yshift={yshift}em, xshift={xshift}em]{matrix_name}-1-{cn}.north west)  -- '
                      f'([yshift={-height*descender_proportion-yshift}em, xshift={xshift}em]{matrix_name}-{nr}-{cn}.base west);\n')

    if label == '':
        lt = ''
        label = '}  % no label'
    else:
        lt = label
        label = f'\\label{{tab:{label}}}'
    if caption == '':
        if label != '':
            logger.warning(f'You have a lable but no caption; the label {label} will be ignored.')
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
            # for debug only
            if lt != '':
                f.write(f'\n\n# A section\nSome text. And a reference \\cref{{{lt}}}')

    return sio.getvalue()


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

    df.index.names = idx_names
    df.columns.names = col_names
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

