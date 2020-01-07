"""
DocMaker: persist exhibits and tables and make overall document

Uses sticb from CEA project

Usage example

```
	test_df = pd.DataFrame(dict(x=range(100), y=range(100,200)))
	fg, ax = plt.subplots(1,3)
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


v 1.0 Dec 2019

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

logger = logging.getLogger('aggdev.log')

# import sys
# import re
# import subprocess
# import pathlib
# import struct
# import numpy as np
# import glob
# import string
# import unicodedata


class DocMaker(object):
    __yaml__ = """---
title: "{title}"
author: Stephen J. Mildenhall
date: "{created}"
fontsize: {font_size}pt
geometry: margin={margin}in
numbersections: {number_sections}
filter: pandoc-citeproc
bibliography: /S/TELOS/biblio/library.bib
csl: /S/TELOS/biblio/journal-of-finance.csl
link-citations: {link_citations}
header-includes:
    - \\input{{/s/teaching/common/general.tex}}
cla: --standalone
cla: -f markdown+smart+yaml_metadata_block+citations
cla: --pdf-engine=xelatex
cla: -t {to_format}
cla: -o {file_format}
cla: --filter pandoc-citeproc
debug: true
---

"""

    def __init__(self, file_name, key, title='DocMaker Document', to_format='latex',
                 file_format='pdf', output_style='caption', tidy=True):
        """
        file_name of output, including extension

        key = short vignette name prepended to name of all tables and image files
        file_name can be a/b/c/file.md
        all intermediate dirs created
        images in a/b/c/img

        tidy: search through the subfolders of the base output and delete all files key-*.*

        to_format can be latex or beamer (later: add the style elements)

        :param file_name:
        :param key:
        :param title:
        :param to_format:
        :param file_format:
        :param output_style: caption (caption in main doc just numbers in include),
                    in-line (all in main doc) or with_table (all in include file)
        :param tidy:

        """

        self.title = title
        self.key = key
        self.output_style = output_style

        self.file_format = file_format
        self.to_format = to_format
        self.file_name = file_name
        self.file = Path(file_name)

        existing = list(self.file.parent.glob(f'**/{key}-*.*'))
        if len(existing):
            print(f"{'Deleting' if tidy else 'There are'} {len(existing)} existing file(s)...")
        for f in existing:
            print(f)

        if tidy:
            for f in existing:
                f.unlink()

        # just on principle
        print('Checking for TMP files...')
        for f in self.file.parent.glob(f'TMP_*.md'):
            print(f'Deleing {f}')
            f.unlink()

        self.file.parent.mkdir(parents=True, exist_ok=True)

        self.out_dir = self.file.cwd() / self.file.parent / 'pdf'
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.out_file = self.out_dir / f'{self.file.parent.stem}.{file_format}'

        self.to_format = to_format
        self.figure_no = 0
        self.figure_dir = self.file.parent / 'img'
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir = self.file.parent / 'table'
        self.table_dir.mkdir(parents=True, exist_ok=True)
        self.sio = StringIO()
        self._off = False

    @property
    def on(self):
        return not self._off

    @on.setter
    def on(self, value):
        self._off = not value

    @property
    def off(self):
        return self._off

    @staticmethod
    def _date_():
        return "Created {date:%Y-%m-%d %H:%M:%S}". \
            format(date=datetime.datetime.now())

    def get_buffer(self):
        return self.sio.getvalue()

    def print_buffer(self):
        print(self.get_buffer())

    def write_markdown(self, font_size, margin):
        """
        persist and write out the StringIO cache to a physical file
        """
        yaml = self.__yaml__.format(
            title=self.title, created=self._date_(),
            font_size=font_size, margin=margin,
            number_sections="true", link_citations="true",
            to_format=self.to_format, file_format=self.file_format)

        with self.file.open('w', encoding='UTF-8') as f:
            f.write(yaml)
            f.write(self.sio.getvalue())

    def process(self, font_size=10, margin=0.5):
        """
        Write out and run pandoc to convert
        Can set font_size and margin here, default 10 and 0.5 inches.
        """
        # need to CD into appropriate directory
        # do NOT overwrite if just using to create the exhibits and you are happy with the md "holder"
        # file
        if self._off:
            print('Currently in OFF mode...returning')
            return

        if not self.file.exists():
            self.write_markdown(font_size, margin)
        cwd = Path.cwd()
        print(str(self.file))
        os.chdir(self.file.parent)
        markdown_make_main("", self.file.name, str(self.file.stem))
        os.chdir(cwd)

    def write(self, *txtv, connector=' '):
        """
        add stuff and return it to get into console
        """
        self.sio.write(connector.join([str(i) for i in txtv]))
        self.sio.write('\n\n')
        return txtv

    def text(self, txt):
        """
        just add text
        """
        self.sio.write(txt)
        self.sio.write('\n\n')

    def section(self, txt):
        self.sio.write(f'# {txt}\n\n')

    def subsection(self, txt):
        self.sio.write(f'## {txt}\n\n')

    def figure(self, f, label, caption="", size="", **kwargs):
        """
        add a figure
        if f is a Figure it is used directly
        else call .get_figure
        else ValueError
        label = used for filename too
        """
        if self._off: return
        if isinstance(f, Figure):
            pass
        else:
            try:
                f = f.get_figure()
            except AttritbuteError as ae:
                print(f'Cannot coerce input object {f} into a figure...ignoring')
                raise ae

        fig_file = self.figure_dir / f'{self.key}-{label}.pdf'
        fig_file_local = f'img/{self.key}-{label}.pdf'
        f.savefig(fig_file, **kwargs)

        # clean label
        label = self.clean_name(label)
        # not the caption, that can contain tex
        if not self.escaped_(caption):
            logger.warning('Caption contains unescaped underscores _. You must ensure they are in dollars.')

        if caption != '':
            fig_text = f'![{caption} '
        else:
            fig_text = '!['
        if label != '':
            fig_text += f'\\label{{fig:{label}}}'
        fig_text += f']({fig_file_local})'
        if size != '':
            fig_text += f'{{{size}}}'
        fig_text += '\n\n'
        self.sio.write(fig_text)

    def table(self, df, label, caption="",
              float_format=None, fill_nan='',
              here='', font_size='', sideways=False,
              sparsify=False, force_float=False, output_style='default', **kwargs):
        """
        Add a table
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

        :param df:
        :param label:
        :param caption:
        :param float_format:
        :param fill_nan:
        :param here:
        :param font_size:
        :param sideways:
        :param sparsify:
        :param force_float:
        :param output_style:
        :param kwargs:
        :return:

        """
        if self._off:
            return df

        # make some sensible choices
        ncols = len(df.columns)
        if ncols >= 12:
            sideways = True
        # do not try to guess font size...just input it sensibly!

        if output_style == 'default':
            output_style = self.output_style

        # will want a much better approach!
        if float_format is None:
            float_format = lambda x: f'{x:.5g}'

        df = df.copy()
        if force_float:
            df = df.astype(float, errors='ignore')

        # have to handle column names that may include _
        # For now assume there are not TeX names
        label = self.clean_name(label)
        # not the caption, that can contain tex
        if not self.escaped_(caption):
            logger.warning('Caption contains unescaped underscores _. You must ensure they are in dollars.')

        df = self.clean_index(df)

        s = df.to_latex(float_format=float_format, sparsify=sparsify, escape=False, **kwargs)
        s = s.replace('nan', fill_nan)
        if here:
            here = f'[{here}]'
        else:
            here = ''
        if font_size != '':
            font_size = f'\\{font_size}\n'
            ends_font_size = '\\normalsize\n'
        else:
            ends_font_size = ''

        # table type
        if sideways:
            tt = 'sidewaystable'
        else:
            tt = 'table'

        s_md_pre = f'\n\\begin{{{tt}}}{here}\n{font_size}\\caption{{{caption}}}\n\\label{{tab:{label}}}\n'
        s_table = f'\\medskip\n\\centering\n{s}'
        s_md_post = f'{ends_font_size}\\end{{{tt}}}\n\n'

        table_file = self.table_dir / f'{self.key}-{label}.md'
        table_file_local = f'table/{self.key}-{label}.md'

        if output_style == 'with_table':
            with table_file.open('w', encoding='utf-8') as f:
                f.write('\n')
                f.write(s_table)
            self.sio.write(s_md_pre)
            self.sio.write(f'\n@@@include {table_file_local}\n\n')
            self.sio.write(s_md_post)

        elif (output_style == 'inline') or (output_style == 'in-line'):
            self.sio.write(s_md_pre)
            self.sio.write(s_table)
            self.sio.write(s_md_post)

        elif output_style == 'caption':
            with table_file.open('w', encoding='utf-8') as f:
                f.write(s_md_pre)
                f.write(s_table)
                f.write(s_md_post)
            self.sio.write(f'@@@include {table_file_local}\n\n')

        else:
            raise ValueError(f'Unknown option {output_style} for output_style passed to table.')

        return df.style.format(float_format)

    @staticmethod
    def clean_name(n):
        try:
            if type(n) == str:
                return n.replace('_', r'\_')
            else:
                return n
        except:
            return n

    @staticmethod
    def escaped_(s):
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
            df.columns = DocMaker.clean_mindex_work(df.columns)
        else:
            df.columns = map(DocMaker.clean_name, df.columns)

        # index
        if isinstance(df.index, pd.core.indexes.multi.MultiIndex):
            df.index = DocMaker.clean_mindex_work(df.index)
        else:
            df.index = map(DocMaker.clean_name, df.index)

        return df

    @staticmethod
    def clean_mindex_work(idx):
        for i, lv in enumerate(idx.levels):
            if lv.dtype == 'object':
                repl = map(DocMaker.clean_name, lv)
                idx = idx.set_levels(repl, level=i)
        return idx

