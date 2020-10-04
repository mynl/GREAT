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
import shutil

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
cla: -o {file_format}
cla: --filter pandoc-citeproc
cla: -t {to_format}
debug: true
---

"""

    def __init__(self, file_name, key='', title='DocMaker Document', to_format='latex',
                 file_format='pdf', fig_format='pdf', output_style='with_table', tidy=False, back_up=True,
                 label_prefix='', figure_dir='', table_dir='', unique_mode=False):
        """
        file_name of output, including extension

        key = short vignette name prepended to name of all tables and image files
        file_name can be a/b/c/file.md
        all intermediate dirs created
        images in a/b/c/img

        tidy: search through the subfolders of the base output and delete all files key-*.*

        to_format can be latex or beamer (later: add the style elements)

        depending on how you name the key the existing pdf file may be deleted.
        key = vig-{vig}-something and file = key.md then it will NOT be because only key-*.* files
        are deleted. This is the recommended approach and is the default if key is ''.

        :param file_name:
        :param key:           defaults to filename excl. the extension
        :param title:
        :param to_format:
        :param file_format:
        :param output_style: caption (caption in main doc just numbers in include),
                    in-line (all in main doc) or with_table (all in include file)
        :param tidy: delete old working files, default True
        :param back_up: make a back up of the last output file, default True
        :param unique_mode: make output file names unique
        :param label_prefix: prefix to add to labels for sub part vigs, these are only used in the docs for TeX
                    accessed labels; they are not used to generate the filename - since these already refer
                    to the vig name
                    figure_dir
                    table_dir over-ride defaults for figure and table outputs
        """

        self.unique_mode = unique_mode
        self.title = title
        self.output_style = output_style
        self.file_format = file_format
        self.fig_format = fig_format
        self.to_format = to_format
        self.file_name = file_name
        self.file = Path(file_name)
        self.label_prefix = f'{label_prefix}-' if label_prefix else ''
        # just ability to generate a unique number so do not have to be careful labels are unique
        self._number = iter(range(10000))
        self.labels = []
        if key == '':
            key = self.key = self.file.stem
        else:
            self.key = key


        existing = list(self.file.parent.glob(f'**/{key}-*.*'))
        if len(existing):
            print(f"{'Deleting' if tidy else 'There are'} {len(existing)} existing file(s)...")
        # for f in existing:
        #     print(f)

        if tidy:
            for f in existing:
                f.unlink()

        # just on principle
        print('Checking for TMP files...')
        for f in self.file.parent.glob(f'TMP_*.md'):
            print(f'Deleting {f}')
            f.unlink()

        self.file.parent.mkdir(parents=True, exist_ok=True)

        self.out_dir = self.file.cwd() / self.file.parent / 'pdf'
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # note this isn't actually used in markdown_make - that creates the filename itself
        self.out_file = self.out_dir / f'{self.file.stem}.{file_format}'

        # however it is used here in make a backup
        if self.out_file.exists() and back_up:
            backup = self.out_dir / f'backup/{self.out_file.name}'
            print(f"Creating back up of {self.out_file} to {backup}")
            shutil.copy(str(self.out_file), str(backup))

        self.to_format = to_format
        self.figure_no = 0
        if figure_dir == '':
            self.figure_dir = self.file.parent / 'img'
        else:
            self.figure_dir = Path(figure_dir)
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        if table_dir == '':
            self.table_dir = self.file.parent / 'table'
        else:
            self.table_dir = Path(table_dir)
        self.table_dir.mkdir(parents=True, exist_ok=True)
        self.sio = StringIO()
        self._off = False

    def make_unique(self, label):
        """ for labelling exhibits uniquely """
        if self.unique_mode:
            if label in self.labels:
                label = f'{label}-{next(self._number)}'
            else:
                self.labels.append(label)
        return label

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

    def include_source(self, file_name):
        """
        file_name = name of ipynb file without the extension

        :param file_name:
        :return:
        """
        self.text(f'''

\\clearpage

\\scriptsize

```python

@@@include code/{file_name}.py

```

\\normalsize

''')

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

    def process(self, font_size=10, margin=1, over_write=True):
        """
        Write out and run pandoc to convert
        Can set font_size and margin here, default 10 and 1 inches.

        Set over_write = False and an existing markdown file will not be overwritten.

        """
        # need to CD into appropriate directory
        # do NOT overwrite if just using to create the exhibits and you are happy with the md "holder"
        # file
        if self._off:
            print('Currently in OFF mode...returning')
            return

        if self.file.exists() == False or over_write:
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

        # do not want the prefix to be used in the file name, just the internal lable
        label = self.make_unique(label)
        fig_file = self.figure_dir / f'{self.key}-{label}.{self.fig_format}'
        fig_file_local = f'img/{self.key}-{label}.{self.fig_format}'
        f.savefig(fig_file, **kwargs)
        label = f'{self.label_prefix}{label}'

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

    def wide_table(self, df, label, caption="", nparts=2,
              float_format=None, fill_nan='',
              here='', font_size='', custom_font_size=0.0,
              sparsify=False, force_float=False, output_style='default', **kwargs):
        """

        nparts = split the table into nparts

        Splits df and passes to table to do the work.

        Large Tables
        ============

        Makes some attempt to split up very large tables


        Example

        ```python

        import great as grt

        DM = grt.DocMaker(f'notes\\dm.md',
                  key=f'dm-test',
                  title=f"Test of DocMaker",
                  tidy=True, back_up=True)

        t = grt.test_df(10, 20)
        DM.table(t, 'all-in-one', 'Trying to put the whole thing togeher. ')
        DM.wide_table(t, 'wide-table', 'Splitting into 3 parts. ', nparts=3)
        DM.process(12, 1)
        ```

        ```tex
            \\begin{sidewaystable}
             \\caption{Table One}\\label{tab:one}
             \\centering
             \\begin{tabular}{*{4}{c}} \\toprule
               Table Head & Table Head & Table Head & Table Head \\\\ \\midrule
               Some Values & Some Values & Some Values & Some Values \\\\
               Some Values & Some Values & Some Values & Some Values \\\\
               Some Values & Some Values & Some Values & Some Values \\\\
               Some Values & Some Values & Some Values & Some Values \\\\ \\bottomrule
             \\end{tabular}

            \\vspace{2\\baselineskip}
            \\caption{Table Two}\\label{tab:two}
             \\centering
             \\begin{tabular}{*{4}{c}} \\toprule
               Table Head & Table Head & Table Head & Table Head \\\\ \\midrule
               Some Values & Some Values & Some Values & Some Values \\\\
               Some Values & Some Values & Some Values & Some Values \\\\
               Some Values & Some Values & Some Values & Some Values \\\\
               Some Values & Some Values & Some Values & Some Values \\\\ \\bottomrule
             \\end{tabular}
            \\end{sidewaystable}
        ```

        :return:

        """

        cols = df.shape[1]
        cols_per_part = cols // nparts
        if nparts * cols_per_part < cols:
            cols_per_part += 1

        for i, b in enumerate(range(0, cols, cols_per_part)):
            bit = df.iloc[:, b:b + cols_per_part]
            lbl = f'{self.label_prefix}{label}-{i}-of-{nparts}'
            if i == 0:
                c = f"{caption} Part {i+1} of {nparts}."
            else:
                c = f"Table {label.replace('-', ' ')} continued, part {i+1} of {nparts}"
            display(
                self.table(bit, lbl,
                        caption=c,
                        float_format=float_format, fill_nan=fill_nan,
                        here=here, font_size=font_size, sideways=True, custom_font_size=custom_font_size,
                        sparsify=False, force_float=False, output_style='with_table', **kwargs)
                     )

    def table(self, df, label, caption="",
              float_format=None, fill_nan='',
              here='', font_size='', sideways=False, custom_font_size=0.0,
              sparsify=False, force_float=False, output_style='default', **kwargs):
        r"""
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
        :param custom_font_size:  e.g.  input the size 0.15, second size will be scaled up appropriately. Overrides font_size
        \fontsize{0.15cm}{0.170cm}\selectfont
        :param sideways:
        :param sparsify:
        :param force_float:
        :param output_style:
        Can be None (default) or first, mid, last.
        :param kwargs:
        :return:

        """
        if self._off:
            return df

        # make some sensible choices: this is not really helpful...too much like driver assist
        # ncols = len(df.columns)
        # if ncols >= 12:
        #     sideways = True
        # do not try to guess font size...just input it sensibly!

        if output_style == 'default':
            output_style = self.output_style

        # will want a much better approach!
        def default_ff(x):
            try:
                return f'{x:.4g}'
            except:
                return x

        if float_format is None:
            float_format = default_ff

        df = df.copy()
        if force_float:
            df = df.astype(float, errors='ignore')

        # have to handle column names that may include _
        # For now assume there are not TeX names
        label = self.make_unique(label)
        label_fn = self.clean_name(label)
        label = f'{self.label_prefix}{label_fn}'

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
        if custom_font_size:
            font_size = f'\\fontsize{{{custom_font_size}cm}}{{{18/15*custom_font_size}cm}}\\selectfont\n'
            ends_font_size = '\\normalsize\n'
        else:
            ends_font_size = ''

        # table type
        if sideways:
            tt = 'sidewaystable'
        else:
            tt = 'table'

        s_md_pre = f'\n\\begin{{{tt}}}{here}\n{font_size}\\caption{{{caption}}}\n\\label{{tab:{label}}}\n'
        s_table = f'\medskip\n\\centering\n{s}'
        s_md_post = f'{ends_font_size}\\end{{{tt}}}\n\n'

        table_file = self.table_dir / f'{self.key}-{label_fn}.md'
        table_file_local = f'tables/{self.key}-{label_fn}.md'

        if output_style == 'caption':
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

        elif output_style == 'with_table':
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
        """
        escape underscores

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

