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
import logging
import re
import numpy as np
import unicodedata
from pandas.io.formats.format import EngFormatter

from IPython.display import Markdown, display
from IPython.core.magic import Magics, magics_class, line_magic, cell_magic, line_cell_magic
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
from IPython import get_ipython

logger = logging.getLogger('aggregate')

# import sys
# import re
# import subprocess
# import pathlib
# import struct
# import numpy as np
# import glob
# import string
# import unicodedata


class PresentationManager(object):

    __yaml__ = """---
title: "{title}"
author: Stephen J. Mildenhall
date: "{created}"
fontsize: {font_size}pt
outertheme : metropolis
innertheme : metropolis
fonttheme  : structurebold
colortheme : orchid
institute: \\convexmark
classoption: t
toc: false
filter: pandoc-citeproc
bibliography: /S/TELOS/biblio/library.bib
csl: /S/TELOS/biblio/journal-of-finance.csl
link-citations: true
header-includes:
    - \\input{{/s/telos/common/newgeneral.tex}}
cla: --standalone
cla: -f markdown+smart+yaml_metadata_block+citations
cla: --pdf-engine=xelatex
cla: -o pdf
cla: --filter pandoc-citeproc
cla: -t beamer
debug: true
---

"""

    def __init__(self, title, key, *, file_name='',
                 fig_format='pdf', output_style='with_table',
                 default_float_fmt=None,
                 default_table_font_size=0.15):
        """
        file_name of output, including extension

        key = short vignette name prepended to name of all tables and image files
        file_name can be a/b/c/file.md
        all intermediate dirs created
        images in a/b/c/img

        tidy: search through the subfolders of the base output and delete all files key-*.*

        depending on how you name the key the existing pdf file may be deleted.
        key = vig-{vig}-something and file = key.md then it will NOT be because only key-*.* files
        are deleted. This is the recommended approach and is the default if key is ''.

        :param file_name:
        :param key:  prepended to file names to cluster them. Should be unique to the doc.
        :param title:
        :param fig_format:
        :param output_style: caption (caption in main doc just numbers in include),
                    in-line (all in main doc) or with_table (all in include file)
        :param default_float_fmt:
        :param default_table_font_size: float or zero, used as a custom font size

        """

        self.title = title
        self.key = key
        if file_name == '':
            self.file_name = self.make_safe_label(title) + '.md'
        else:
            self.file_name = file_name + '.md'
        self.fig_format = fig_format
        self.output_style = output_style
        if default_float_fmt is None:
            self.default_float_fmt = PresentationManager.default_float_format
        else:
            self.default_float_fmt = default_float_fmt
        self.default_table_font_size = default_table_font_size

        # main file for output and ancillary directories
        self.base_dir = Path('notes')
        self.file = self.base_dir / self.file_name
        logger.info(f'using base dir {self.base_dir.absolute()}')
        logger.info(f'using filename {self.file}')
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = self.base_dir / 'pdf'
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir = self.base_dir / 'img'
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir = self.base_dir / 'table'
        self.table_dir.mkdir(parents=True, exist_ok=True)
        self.section = 0
        self.appendix = -1

        # tidy up
        self.tidy_up()

        # output
        self.toc = []
        self.sios = {'contents': StringIO(), 'summary': StringIO(), 'body': StringIO(), 'appendix' : StringIO()}

        self.sios['contents'].write('## Contents\n')

        # starts active
        self._active = True


    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, value):
        self._active = value

    def section_number(self, buf):
        """
        get the number / letter for the next section

        return the prefix string for the toc and slide (allows summary not to be numbered)
        :param buf:
        :return:
        """
        if buf == 'body':
            self.section += 1
            return f'Section {self.section: 2d}. '
        elif buf == 'summary':
            return ''
        elif buf == 'appendix':
            self.appendix += 1
            c = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[self.appendix]
            return f'Appendix {c}. '
        else:
            raise ValueError('Hopeless confusion!')

    def text(self, txt, buf='body', tacit=False):
        """
        add text to buffer

        """
        if not self.active:
            return

        stxt = txt.split('\n')

        for i, ln in enumerate(stxt):
            m = re.findall('^# (.*)', ln)
            if m:
                s = self.section_number(buf)
                ln = f"# {s} {self.make_title(m[0])}"
                toc_ln = f"* {s} {self.make_title(m[0])}"
                stxt[i] = ln
                self.toc.append(toc_ln)

        txt = '\n'.join(stxt)
        self.sios[buf].write(txt)
        self.sios[buf].write('\n\n')
        if not tacit:
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
            self.sios[buf].write(f'## {slide_caption}\n\n')
            if not tacit:
                display(Markdown(f'## {slide_caption}'))
        if promise:
            return fig_text
        else:
            self.sios[buf].write(fig_text)

        if not tacit:
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

        assert not promise or self.output_style=='with_table'

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

        # pick up the correct output buffer
        sio = self.sios[buf]

        if new_slide and not promise:
            sio.write(f'## {slide_caption}\n\n')
            if not tacit:
                display(Markdown(f'## {slide_caption}'))

        if self.output_style == 'caption':
            with table_file.open('w', encoding='utf-8') as f:
                f.write('\n')
                f.write(s_table)
            sio.write(s_md_pre)
            sio.write(f'\n@@@include {table_file_local}\n\n')
            sio.write(s_md_post)

        elif (self.output_style == 'inline') or (self.output_style == 'in-line'):
            sio.write(s_md_pre)
            sio.write(s_table)
            sio.write(s_md_post)

        elif self.output_style == 'with_table':
            with table_file.open('w', encoding='utf-8') as f:
                f.write(s_md_pre)
                f.write(s_table)
                f.write(s_md_post)
            if promise:
                return f'@@@include {table_file_local}\n\n'
            else:
                sio.write(f'@@@include {table_file_local}\n\n')

        else:
            raise ValueError(f'Unknown option {self.output_style} for output_style passed to table.')

        if not tacit:
            display(df.style.format(float_format))

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
            if abs(x) >= 1e-3 and abs(x) < 1e6:
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
        s = s.replace('-', ' ').title()
        # from / to
        for f, t in zip(['Var', 'Tvar', 'Cv', 'Vs', 'Epd', 'Lr', 'Roe'],
                        ['VaR', 'TVaR', 'CV', 'vs.', 'EPD', 'LR', 'ROE']):
            s = s.replace(f, t)
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
        value = re.sub('[^\w\s-]', '', value).strip().lower()
        value = re.sub('[-\s]+', '-', value)
        return value

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

    def buffer(self):
        """
        assemble all the parts
        :return:
        """
        # new SIO each time for contents...
        self.sios['contents'] = StringIO()
        toc_summary = sorted([i for i in self.toc if i[2:].split()[0].strip() not in ('Appendix', 'Section')])
        toc_body = sorted([i for i in self.toc if i[2:].split()[0].strip() == 'Section'])
        toc_appendix = sorted([i for i in self.toc if i[2:].split()[0].strip() == 'Appendix'])
        self.sios['contents'].write('# Table of Conents\n## Contents\n\\footnotesize\n\n')
        self.sios['contents'].write('\n'.join(toc_summary + toc_body + toc_appendix))
        return '\n\n'.join(buf.getvalue() for buf in self.sios.values())

    def buffer_display(self):
        display(Markdown(self.buffer()))

    def buffer_persist(self, font_size=12, tacit=False):
        """
        persist and write out the StringIO cache to a physical file
        """
        yaml = self.__yaml__.format(
                title=self.title, created=self.date(),
                font_size=font_size)

        with self.file.open('w', encoding='UTF-8') as f:
            f.write(yaml)
            f.write(self.buffer())

        if not tacit:
            display(Markdown(self.file.open('r').read()))

    def process(self, font_size=10, tacit=True):
        """
        Write out and run pandoc to convert
        Can set font_size and margin here, default 10 and 1 inches.

        Set over_write = False and an existing markdown file will not be overwritten.

        """
        # need to CD into appropriate directory
        # do NOT overwrite if just using to create the exhibits and you are happy with the md "holder"
        # file
        if not self.active:
            print('Currently inactive...returning')
            return

        self.buffer_persist(font_size, tacit)
        if str(Path.home()) != '/home/steve':
            # not in Linux
            cwd = Path.cwd()
            print(str(self.base_dir))
            os.chdir(self.base_dir)
            markdown_make_main("", self.file.name, str(self.file.stem))
            os.chdir(cwd)

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

    # TODO SORT OUT below here?!
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
        # columns
        if isinstance(df.columns, pd.core.indexes.multi.MultiIndex):
            df.columns = PresentationManager.clean_mindex_work(df.columns)
        else:
            df.columns = map(PresentationManager.clean_name, df.columns)

        # index
        if isinstance(df.index, pd.core.indexes.multi.MultiIndex):
            df.index = PresentationManager.clean_mindex_work(df.index)
        else:
            df.index = map(PresentationManager.clean_name, df.index)

        return df

    @staticmethod
    def clean_mindex_work(idx):
        for i, lv in enumerate(idx.levels):
            if lv.dtype == 'object':
                repl = map(PresentationManager.clean_name, lv)
                idx = idx.set_levels(repl, level=i)
        return idx


@magics_class
class PresentationManagerMagic(Magics):
    """
    description: implements magics to help using PresentationManager class

    pmt = pres maker text (blob, write)

    """

    @line_cell_magic
    @magic_arguments('%pmb')
    @argument('-t', '--tacit', action='store_true', help='Tacit: suppress output as Markdown')
    @argument('-a', '--appendix', action='store_true', help='Mark as appendix material.')
    @argument('-s', '--summary', action='store_true', help='Mark as summary material (exlusive with appendix).')
    @argument('-f', '--fstring', action='store_true', help='Convert cell into f string and evaluate.')
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
                temp = f'f"""{cell}"""'
                cell = self.shell.ev(temp)
            self.shell.ev(f'PM.text("""{cell}""", buf="{buf}", tacit={args.tacit})')

    @line_cell_magic
    @magic_arguments('%pmf')
    @argument('-t', '--tacit', action='store_true', help='tacit: suppress output as Markdown')
    @argument('-a', '--appendix', action='store_true', help='Mark as appendix material.')
    @argument('-s', '--summary', action='store_true', help='Mark as summary material (exlusive with appendix).')
    @argument('-n', '--new_slide', action='store_false', help='Set to suppress new slide.')
    @argument('-f', '--fstring', action='store_true', help='Convert caption into f string and evaluate.')
    @argument('-p', '--promise', action='store_true', help='Promise: write file but append nothing to stream')
    @argument('-h', '--height', type=float, default=0.0, help='Vertical size, generates height=height clause.')
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
            optional: last line is format_function, e.g., as lambda function

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
    @argument('-t', '--tacit', action='store_true', help='tacit: suppress output as Markdown')
    @argument('-p', '--promise', action='store_true', help='Promise: write file but append nothing to stream')
    @argument('-a', '--appendix', action='store_true', help='Mark as appendix material.')
    @argument('-m', '--summary', action='store_true', help='Mark as summary material (exlusive with appendix).')
    @argument('-n', '--new_slide', action='store_false', help='Set to suppress new slide.')
    @argument('-f', '--fstring', action='store_true', help='Convert caption into f string and evaluate.')
    @argument('-w', '--wide', type=int, default=0, help='Use wide table mode, WIDE number of columns')
    @argument('-z', '--size', type=str, default='', help='Fontsize: tiny, scriptsize, footnotesize, or number (e.g., 0.15) for custom font size etc.')
    @argument('-r', '--format', action='store_true', help='Indicates the last line of input is a float_format function')
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
            if args.wide:
                # wide tables don't have captions.
                s = f'promise = PM.wide_table({df}, "{label}", buf="{buf}", ' \
                        f'new_slide={args.new_slide}, nparts={args.wide}, '  \
                        f'tacit={args.tacit}, promise={args.promise}'
            else:
                s = f'promise = PM.table({df}, "{label}", buf="{buf}", caption="""{caption}""", ' \
                        f'new_slide={args.new_slide}, ' \
                        f'tacit={args.tacit}, promise={args.promise}'
            if args.size != '':
                if np.all([i in '0123456789.' for i in args.size]):
                    s += f', font_size={args.size}'
                else:
                    s += f', font_size="{args.size}"'
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


ip = get_ipython()
ip.register_magics(PresentationManagerMagic)

