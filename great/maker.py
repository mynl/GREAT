"""
Base class for PresentationManager and SimpleManager

"""

import datetime
import re

import numpy as np
import pandas as pd
import unicodedata
from pandas.io.formats.format import EngFormatter
from titlecase import titlecase
from IPython.display import Markdown, display


class ManagerBase(object):
    """
    Common methods for SimpleManager and PresentationManager
    """

    def buffer(self, decorate=''):
        """
        assemble all the parts
        :return:
        """
        # update toc
        if 'contents' in self.sios:
            self.make_toc(decorate)
        # assemble parts
        s = '\n\n'.join(buf.getvalue().strip() for buf in self.sios.values())
        # sublime text-esque removal of white space
        s = re.sub(' *\n(\n?)\n*', r'\n\1', s, flags=re.MULTILINE)
        return s

    def buffer_display(self):
        display(Markdown(self.buffer()))

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

    def make_label(self, label, option):
        # prepend option id on label to avoid collisions
        if option:
            label = f'{self.option_id}-{label}'
        label = self.make_safe_label(label)
        label = self.clean_name(label)
        return label

    @staticmethod
    def default_float_format(x, neng=3):
        """
        the endless quest for the perfect float formatter...

        tester::

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

    @staticmethod
    def date():
        return "Created {date:%Y-%m-%d %H:%M:%S}". \
            format(date=datetime.datetime.now())

    @staticmethod
    def clean_name(n):
        """
        escape underscores for using a name in a DataFrame index

        :param n:
        :return:
        """
        try:
            if type(n) == str:
                # quote underscores that are not in dollars
                return '$'.join((i if n % 2 else i.replace('_', '\\_') for n, i in enumerate(n.split('$'))))
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
            df.columns = ManagerBase.clean_mindex_work(df.columns)
        else:
            df.columns = map(ManagerBase.clean_name, df.columns)

        # index
        if isinstance(df.index, pd.core.indexes.multi.MultiIndex):
            df.index = ManagerBase.clean_mindex_work(df.index)
        else:
            df.index = map(ManagerBase.clean_name, df.index)
        df.index.names = idx_names
        df.columns.names = col_names
        return df

    @staticmethod
    def clean_mindex_work(idx):
        for i, lv in enumerate(idx.levels):
            if lv.dtype == 'object':
                repl = map(ManagerBase.clean_name, lv)
                idx = idx.set_levels(repl, level=i)
        return idx

    def toggle_debug(self, onoff):
        """
        ``onoff == on``  set debug=True but remember previous state
        ``onoff == off`` restore previous state
        :param onoff:
        :return:
        """

        if onoff == "on":
            self._debug_state = self.debug
            self.debug = True

        elif onoff == 'off':
            self.debug = self._debug_state

