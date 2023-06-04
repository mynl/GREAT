"""
SimpleManager: simple version of PresentationManager aimed at simple
HTML files and short LaTeX docs. No section numbers etc. No config file.

Based on PresentationManager as of 2022-01-25.

    * Only makes presentations
    * No Appendix nor TOC
    * Removes unused options
    * Integrated into alternative set of magics magics

v 1.0 Dec 2022 SimpleManager from PresentationManager

"""

import sys
import logging
import re
from collections import OrderedDict
from io import StringIO
from pathlib import Path

import numpy as np
from IPython import get_ipython
from IPython.core.magic import Magics, magics_class, cell_magic
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
from IPython.display import Markdown, display
from matplotlib.pyplot import Figure

from .pres_maker import df_to_tikz
from .maker import ManagerBase

logger = logging.getLogger(__name__)


class SimpleManager(ManagerBase):
    """
    SimpleManager class

    """

    def __init__(self, title, prefix, option_id='', base_dir='.', figure_dir='img',
        table_dir='table_data'):
        """

        :param title:
        :param prefix: for prefixing figure and table files
        :param option_id: when there are scenarios
        :param base_dir: where to store output
        :param figure_dir:
        :param table_dir: where CSV snapshots of data get stored; all tables are in-line!
        :param figure_format:
        :param dpi:
        """

        # read in builds level global variables
        self.title = title
        self.prefix = prefix
        self.option_id = option_id

        # file specific, set manually
        if isinstance(base_dir, Path):
            self.base_dir = base_dir
        else:
            self.base_dir = Path(base_dir)

        self.file_name = title.replace(' ', '-') + '.md'
        self.file = self.base_dir / self.file_name
        self.figure_dir = self.base_dir / figure_dir
        self.figure_dir_name = figure_dir
        self.table_dir = self.base_dir / table_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir.mkdir(parents=True, exist_ok=True)

        # admin
        self.default_float_fmt = SimpleManager.default_float_format

        # starts in base self.config: active and not in debug mode
        self._tacit_override = False
        self._active = True
        self._debug = False
        self._debug_state = False
        # avoid using dm because updated gets in a loop on the dates...
        self.debug_dm_file = None

        # may want to add, e.g., tags later
        self.sios = OrderedDict(body=StringIO())

        # avoid using dm because updated gets in a loop on the dates...
        self.debug_dm_file = self.base_dir / 'dm2.md'

        # start by writing the header
        self._write_buffer_('body', f'# {self.title.title()}\n')

    def text(self, txt, buf='body', tacit=False):
        """
        add text to buffer

        """
        if not self.active:
            return

        self._write_buffer_(buf, txt)
        if self.tacit_override or not tacit:
            display(Markdown(txt))

    def figure(self, f, label, buf='body', caption="", width="", new_slide=False,
               tacit=False, promise=False, option=True, figure_format='jpg',
               dpi=600, **kwargs):
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
        :param width:
        :param new_slide:
        :param tacit:
        :param promise:  promise mode: just save the figure with appropriate name
        :param option:   different versions of the same figure - decorate name with option
        :param kwargs:
        """
        if not self.active:
            return

        if isinstance(f, Figure):
            pass
        else:
            try:
                f = f.get_figure()
            except AttributeError as ae:
                logger.warning(f'Cannot coerce input object {f} into a figure...ignoring')
                raise ae

        slide_caption = self.make_title(label)
        # adds the option id if option is true
        label = self.make_label(label, option)
        # label = self.make_safe_label(label)
        if self.prefix != '':
            fig_file = self.figure_dir / f'{self.prefix}-{label}.{figure_format}'
        else:
            fig_file = self.figure_dir / f'{label}.{figure_format}'
        fig_file_local = f'{self.figure_dir_name}/{fig_file.name}'
        if fig_file.exists():
            logger.warning(f'File {fig_file} already exists...over-writing.')
        f.savefig(fig_file, dpi=dpi,  **kwargs)
        logger.info(f'{dpi} dpi fig written to {fig_file}')
        # not the caption, that can contain tex
        if not self.clean_underscores(caption):
            logger.warning('Caption contains unescaped underscores _. You must ensure they are in dollars.')

        if caption != '':
            fig_text = f'![{caption} '
        else:
            fig_text = '!['
        fig_text += f']({fig_file_local})'
        if width != '':
            fig_text += f'{{width={width}%}}'
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

    def make_label(self, label, option):
        # prepend option id on label to avoid collisions
        if option:
            label = f'{self.option_id}-{label}'
        label = self.make_safe_label(label)
        label = self.clean_name(label)
        return label

    def table(self, df, label, *, caption="", buf='body',
                   new_slide=True, tacit=False, promise=False,
                   float_format=None, show_index=True, sparsify=1, option=False):
        """
        poor man's table based on df.to_html

        """
        if not self.active:
            if not tacit:
                display(df)
            return

        # store the table
        temp = self.make_label(label, option)
        data_file = self.table_dir / f'{self.prefix}-{temp}.csv'
        df.to_csv(data_file)

        if float_format is None:
            float_format = self.default_float_fmt

        df = df.copy()
        df = df.astype(float, errors='ignore')
        slide_caption = self.make_title(label)

        s_table = df.to_html(buf=None, index=show_index, float_format=float_format, sparsify=sparsify)

        # write it out, with table, no separate files - don't want @@@ complications
        if new_slide and not promise:
            self._write_buffer_(buf, f'\n## {slide_caption}\n\n')
            # self._write_buffer_(buf, f'## {slide_caption}\n\n')
            if not tacit:
                display(Markdown(f'## {slide_caption}'))

        # actually do the write
        self._write_buffer_(buf, s_table)
        self._write_buffer_(buf, '\n\n')

        if self.tacit_override or not tacit:
            display(df.style.format(float_format))
            if caption != '':
                display(Markdown(caption))

    # LIKELY WON'T BE USED IN THIS SIMPLE VERSION...but left
    def tikz_table(self, df, label, *, caption="", buf='body',
                   new_slide=True, tacit=False, promise=False,
                   float_format=None, tabs=None,
                   show_index=True, scale=0.717,
                   figure='figure', hrule=None,
                   equal=False, option=True, latex=None,
                   vrule=None, sparsify=1):
        """
        Add a table using TikZ formatter

        label used as file name
        force_float = convert input to float first (makes a copy) for col alignment

        output_style as table

        1. with_table : all output in @@@ file and include in main md file; use when caption is generic
        1. caption:   puts caption text in the main markdown file, use when caption will be edited
        1. inline: all output in md file directly (not recommended)

        label and columns have _ escaped for TeX but the caption is not - so beware!::

            fn_out=None, float_format=None, tabs=None,
                show_index=True, scale=0.717, height=1.5, column_sep=2, row_sep=0,
                figure='figure', color='blue!0', extra_defs='', lines=None,
                post_process='', label='', caption=''

        :param df:
        :param label:
        :param caption:
        :param buf:
        :param new_slide:
        :param tacit:
        :param promise:
        :param float_format:
        :param tabs:
        :param show_index:
        :param scale:
        :param figure:
        :param hrule:
        :param equal:
        :param option:
        :param latex:
        :param vrule:
        :param sparsify:
        :return:
        """

        if not self.active:
            if not tacit:
                display(df)
            return

        # store the table
        temp = self.make_label(label, option)
        data_file = self.table_dir / f'data/{self.prefix}-{temp}.csv'
        df.to_csv(data_file)

        df = df.copy()
        df = df.astype(float, errors='ignore')

        # have to handle column names that may include _
        # For now assume there are not TeX names
        slide_caption = self.make_title(label)
        label = self.make_label(label, option)

        # check the caption, that can contain tex
        if not self.clean_underscores(caption):
            logger.warning('Caption contains unescaped underscores _. You must ensure they are in dollars.')

        # added a local buffer so output works better with debug model: need a single call to ._write_buffer_
        if float_format is None:
            float_format = self.default_float_fmt

        # do the work
        s_table = df_to_tikz(df, label=label, caption=caption,
                             float_format=float_format, tabs=tabs,
                             show_index=show_index, scale=scale, equal=equal,
                             figure=figure, hrule=hrule, vrule=vrule, sparsify=sparsify,
                             latex=latex, clean_index=True)


        sio_temp = StringIO()
        table_file = self.table_dir / f'{self.prefix}-{temp}.md'
        if table_file.exists():
            logger.debug(f'File {table_file} exists, over-writing.')
        table_file_local = f'{self.table_dir.stem}/{table_file.name}'

        if new_slide and not promise:
            sio_temp.write(f'\n## {slide_caption}\n\n')
            # self._write_buffer_(buf, f'## {slide_caption}\n\n')
            if not tacit:
                display(Markdown(f'## {slide_caption}'))

        with table_file.open('w', encoding='utf-8') as f:
            f.write(s_table)
        if promise:
            return f'@@@include {table_file_local}\n\n'
        else:
            _ = f'@@@include {table_file_local}\n\n'
            sio_temp.write(_)
            logger.info(_)

        # actually do the write
        self._write_buffer_(buf, sio_temp.getvalue())

        if self.tacit_override or not tacit:
            display(df.style.format(float_format))
            if caption != '':
                display(Markdown(caption))

    def _write_buffer_(self, buf, text='\n'):
        """
        single point to write to buffer, allows overloading of other write related functions,
        such as tracking toc and making a tree diagram of the document

        not to be called externally, hence name

        :param buf:
        :param text:
        :return:
        """
        # actually write the text
        self.sios[buf].write(text)

        # optionally write debug file
        if self.debug in ('on', 'append', True):
            logger.info(f'In debug mode...writing {self.debug_dm_file}')
            mode = 'a' if self.debug == 'append' else 'w'
            self.debug_dm_file.open(mode, encoding='utf-8').write(text)

    def buffer_persist(self, tacit=False, debug=False, debug_filename='dm2.md'):
        """
        persist and write out the StringIO cache to a physical file

        if debug just write to dm.md and exit

        :param font_size:
        :param toc:    for burst mode: read toc to number sections ignored if it doesn't exist
        :param tacit:  suppress markdown output of persisted file

        """

        if debug:
            debug_file = self.base_dir / debug_filename
            with debug_file.open('w', encoding='UTF-8') as f:
                f.write(self.buffer())
                logger.info(f'Writing to file {debug_file.resolve()}...and exiting.')
            return

        with self.file.open('w', encoding='UTF-8') as f:
            f.write(self.buffer())
            logger.info(f'Writing to file {self.file.resolve()}')
        if not tacit:
            display(Markdown(self.file.open('r').read()))

    def buffer_to_blog(self, tacit=True, dry_run=True, source_dir=None, out_dir=None):
        """
        Post the buffer as a blog entry

        Remember to add the tags!

        :param tacit: passed to buffer_persist
        :param dry_run: bassed to BlogPublisher routine
        :param source_dir:  default is .
        :param out_dir:  defualut is Path.hom() / 'S/websites/new_mynl/blog'
        :return:
        """

        sys.path.append(str((Path.home() / 'S/TELOS/blog/Python').resolve()))
        from blog_tools import BlogPublisher

        if source_dir is None:
            source_dir = Path('.').resolve()
        if out_dir is None:
            out_dir = Path.home() / 'S/Websites/new_mynl/blog'

        # make blog poster object
        bp = BlogPublisher(source_dir, out_dir, update=True, dry_run=dry_run)

        # create file of buffer
        self.buffer_persist(tacit=tacit)

        logger.info(f'Publishing {self.file}')

        # publish it
        bp.publish_file(self.file)


@magics_class
class SimpleManagerMagic(Magics):
    """
    description: implements magics to help using SimpleManager class

    expects a SM object (where PresentationManagerMagic uses PM)

    Very similar to PresentationManagerMagic (not ideal)

    * Deleted some unused options
    * All cell magics, not line/cell

    """

    @cell_magic
    @magic_arguments('%smb')
    @argument('-c', '--code', action='store_true', help='Enclose in triple quotes as Python code.')
    @argument('-d', '--debug', action='store_true', help='Turn debug on temporarily .')
    @argument('-f', '--fstring', action='store_true', help='Convert cell into f string and evaluate.')
    @argument('-i', '--ignore', action='store_true', help='Ignore the cell, turns it into a comment')
    @argument('-t', '--tacit', action='store_true', help='Tacit: suppress output as Markdown')
    def smb(self, line='', cell=None):
        """
        SimpleManager blob (text/write) line/cell magic

        %pmt line  -> written to body
        %%pmt -s -a -m  (s show; a=appendix, m=suMmary)

        """
        self.shell.ex('if SM.debug: logger.warning(f"SM debug mode set to {SM.debug}")')
        args = parse_argstring(self.smb, line)
        if args.ignore:
            return
        logger.debug(args)
        buf = 'body'
        if args.fstring:
            logger.debug('evaluating as f string')
            if args.code:
                temp = f'f"""\n\n```python\n{cell.strip()}\n```\n\n"""'
            else:
                temp = f'f"""{cell}"""'
            cell = self.shell.ev(temp)
        if args.debug:
            self.shell.ev('SM.toggle_debug("on")')
        self.shell.ev(f'SM.text("""{cell}""", buf="{buf}", tacit={args.tacit})')
        if args.debug:
            self.shell.ev('SM.toggle_debug("off")')

    @cell_magic
    @magic_arguments('%smf')
    @argument('-c', '--command', action='store_true', help='Load the underlying command into the current cell')
    @argument('-d', '--debug', action='store_true', help='Turn debug on temporarily .')
    @argument('--dpi', type=int, default=600, help='DPI for savefig. Default 600.')
    @argument('-f', '--fstring', action='store_true', help='Convert caption into f string and evaluate.')
    @argument('-i', '--ignore', action='store_true', help='Ignore the cell, turns into a comment')
    @argument('-m', '--format', type=str, default='jpg', help='Save file format, default jpg')
    @argument('-n', '--new_slide', action='store_true', help='Set to use label as second level heading. '
                                                             'Note difference to pmt')
    @argument('-o', '--option', action='store_true',
              help='Set option for tables that vary with options, uses decorated top_value filename.')
    @argument('-p', '--promise', action='store_true', help='Promise: write file but append nothing to stream')
    @argument('-t', '--tacit', action='store_true', help='tacit: suppress output as Markdown')
    @argument('-w', '--width', type=int, default=100, help='Horizontal size, generates width=width clause.')
    def smf(self, line='', cell=None):
        """
        SimpleManager figure utility. Add a new figure to the stream, format and save as format file.

        @argument('-h', '--height', type=float, default=0.0, help='Vertical size, generates height=height clause.')

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
        self.shell.ex('if SM.debug: logger.warning(f"SM debug mode set to {SM.debug}")')

        args = parse_argstring(self.smf, line)
        if args.ignore:
            return
        logger.debug(args)
        buf = 'body'

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
        s = f'promise = SM.figure({f}, "{label}", buf="{buf}", caption="""{caption}""", ' \
            f'new_slide={args.new_slide}, tacit={args.tacit}, promise={args.promise}, ' \
            f'option={args.option}, figure_format="{args.format}", dpi={args.dpi}'
        if args.width:
            s += f', width={args.width}'
        s += ')'

        logger.debug(s)
        if args.command:
            self.load_cell(s, line, cell)
        else:
            if args.debug:
                self.shell.ev('SM.toggle_debug("on")')
            self.shell.ex(s)
            if args.debug:
                self.shell.ev('SM.toggle_debug("off")')


    @cell_magic
    @magic_arguments('%smt')
    @argument('-b', '--tabs', type=str, default='', help='Set tabs.')
    @argument('-c', '--command', action='store_true', help='Load the underlying command into the current cell')
    @argument('-d', '--debug', action='store_true', help='Turn debug on temporarily .')
    @argument('-f', '--fstring', action='store_true', help='Convert caption into f string and evaluate.')
    @argument('-h', '--hrule', type=str, default=None, help='Horizontal rule locations e.g., 1,3,-1')
    @argument('-i', '--ignore', action='store_true', help='Ignore the cell, turns into a comment')
    @argument('-l', '--latex', type=str, default='', help='latex switches applied to the float container, '
                                                           'h is the most useful')
    @argument('-m', '--summary', action='store_true', help='Mark as summary material (exclusive with appendix).')
    @argument('-n', '--new_slide', action='store_false', help='Set to suppress new slide.')
    @argument('-o', '--option', action='store_true',
              help='Set option for tables that vary with options, uses decorated top_value filename.')
    @argument('-p', '--promise', action='store_true', help='Promise: write file but append nothing to stream')
    @argument('-q', '--equal', action='store_true', help='Hint the column widths should be equal')
    @argument('-w', '--wide', type=int, default=0, help='Use wide table mode, WIDE number of columns')
    @argument('-r', '--format', action='store_true', help='Indicates the last line of input is a float_format function')
    @argument('-s', '--scale', type=float, default="0.8", help='Scale for tikz')
    @argument('-t', '--tacit', action='store_true', help='tacit: suppress output as Markdown.')
    @argument('-u', '--underscore', action='store_true', help='Apply de_underscore prior to formatting.')
    @argument('-v', '--vrule', type=str, default=None, help='Vertical rule locations, to left of col no')
    @argument('-z', '--tikz', action='store_true', help='Use tikz table converter. Default uses Pandas to_html.')
    def smt(self, line, cell):
        """
        SimpleManager table utility. Add a new table to the stream, format and save as TeX file.

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
        self.shell.ex('if SM.debug: logger.warning(f"SM debug mode set to {SM.debug}")')

        args = parse_argstring(self.smt, line)
        if args.ignore:
            return
        # manipulation common to both engines
        logger.debug(args)
        buf = 'body'
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
        option = args.option

        if args.tikz:
            logger.warning(f'Making tikz {args.tikz}')
            hrule = args.hrule
            vrule = args.vrule
            equal = args.equal
            tabs = args.tabs
            latex = args.latex
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
                s = f'promise = SM.tikz_table(grt.de_underscore({df}), '
            else:
                s = f'promise = SM.tikz_table({df}, '
            s += (  f'"{label}", buf="{buf}", caption="""{caption}""", '
                    f'new_slide={args.new_slide}, '
                    f'tacit={args.tacit}, promise={args.promise}, '
                    f'hrule={hrule}, vrule={vrule}, scale={scale}, '
                    f'equal={equal}, '
                    f'option={option}, '
                    f'latex="{latex}", '
                    'sparsify=1, figure="table"' )
            if type(tabs) == list:
                s += f', tabs={tabs} '
        else:
            s = f'promise = SM.table({df}, '
            # just html table options
            s += (  f'"{label}", buf="{buf}", caption="""{caption}""", '
                f'new_slide={args.new_slide}, '
                f'tacit={args.tacit}, promise={args.promise}, '
                f'option={option}, '
                'sparsify=1, ' )
        if args.format:
            s += f''', float_format={ff}'''
        s += ')'

        logger.debug(f'command:\n\n{s}\n\n')
        if args.command:
            self.load_cell(s, line, cell)
        else:
            if args.debug:
                self.shell.ev('SM.toggle_debug("on")')
            self.shell.ex(s)
            if args.debug:
                self.shell.ev('SM.toggle_debug("off")')

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


ip = get_ipython()
if ip is not None:
    ip.register_magics(SimpleManagerMagic)

