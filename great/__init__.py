"""
GREAT for Python

"""

import logging

logging.basicConfig(format='%(asctime)s | l. %(lineno) 5d | %(funcName)-22s | %(message)s',
                    datefmt='%H:%M:%S')

from .crypto import SFile

from .doc_maker import DocMaker

from .image_tools import quilt

from .markdown_make import markdown_make_main, md_summary

from .pres_maker import PresentationManager, de_underscore, df_to_tikz, guess_column_widths

from .simple_maker import SimpleManager

from .scripts import *

from .utils import checksum, float_to_binary, test_df, SimpleAxes, FigureManager, now, filename, binary

from .watcher import Watcher

from .word_tools import TextAnalyzer, WordClouder
