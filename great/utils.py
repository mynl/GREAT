"""
Misc. utilities

"""

import hashlib
import pandas as pd
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
from cycler import cycler
from itertools import product
import math
from numpy import floor

# for magics
from IPython import get_ipython
from IPython.core import magic_arguments
from IPython.core.magic import line_magic, cell_magic, line_cell_magic, Magics, magics_class


def checksum(ob):
    """
    make the checksum of an object in a reasonable way
    """
    hasher = hashlib.sha256()
    if isinstance(ob, pd.core.frame.DataFrame):
        bob = str(ob.head(20).T).encode('utf-8')
    else:
        bob = str(ob).encode('utf-8')
    hasher.update(bob)
    return hasher.hexdigest()


def test_df(nrows=10, ncols=3, multi_index=False):
    """
    make a dummy test dataframe

    """
    if multi_index: ncols += 2
    colnames = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    assert ncols < 26
    df = pd.DataFrame(np.random.rand(nrows, ncols), columns=colnames[:ncols])
    df.columns.name = 'col_name'
    df.index.name = 'idx_name'
    if multi_index:
        df.iloc[:, -2] = range(nrows)
        df.iloc[:, -1] = np.random.choice(colnames, nrows)
        df = df.set_index(list(df.columns[-2:]))
        df.index.names = ['l1_name', 'l2_name']
    return df


def float_to_binary(num):
    """
    Certain floats like 0.1 have very awkward binary reps and lead to floating point issues,
    and hence indexing issues. Best just to avoid. E.g. don't use 0.1 use 0.125 as a bs. etc.

    This function lets you see the binary expansion of the float.

    Struct can provide us with the float packed into bytes. The '!' ensures that
    it's in network byte order (big-endian) and the 'f' says that it should be
    packed as a float. Alternatively, for double-precision, you could use 'd'.

    https://stackoverflow.com/questions/16444726/binary-representation-of-float-in-python-bits-not-hex

    """
    packed = struct.pack('!f', num)
    # print('Packed: %s' % repr(packed))

    # For each integer, we'll convert it to its binary representation.
    binaries = [bin(i) for i in packed]
    # print ('Binaries: %s' % binaries)

    # Now strip off the '0b' from each of these
    stripped_binaries = [s.replace('0b', '') for s in binaries]
    # print ('Stripped: %s' % stripped_binaries)

    # Pad each byte's binary representation's with 0's to make sure it has all 8 bits:
    #
    # ['00111110', '10100011', '11010111', '00001010']
    padded = [s.rjust(8, '0') for s in stripped_binaries]
    # print( 'Padded: %s' % padded)

    # At this point, we have each of the bytes for the network byte ordered float
    # in an array as binary strings. Now we just concatenate them to get the total
    # representation of the float:
    return ''.join(padded)


# general utilities
class SimpleAxes():
    def __init__(self, n, nc=3, aspect=1.5, sm_height=2.0, lg_height=4, **kwargs):
        """
        make a reasonable grid of n axes nc per row with given height and aspect ratio
        returns the figure, the axs and ax iterator
        sm_height uses for two or more rows
        lg_height used for just one row

        kwargs passed to subplots, e.g. sharex sharey
        """
        nc = min(nc, n)
        nr = n // nc
        if n % nc:
            nr += 1
        if nr == 1:
            height = sm_height
        else:
            height = lg_height
        w = nc * height * aspect
        h = nr * height
        sc = min(w, 8) / w
        w *= sc
        h *= sc
        w = min(w, 8)
        self.f, self.axs = plt.subplots(nr, nc, figsize=(w, h), squeeze=False,
                                        constrained_layout=True, **kwargs)
        self.axit = iter(self.axs.flatten())
        self._ax = None

    def __next__(self):
        self._ax = next(self.axit)
        return self._ax

    def get_figure(self):
        return self.f

    def tidy(self):
        """
        remove all unused plots
        """
        try:
            while 1:
                self.f.delaxes(next(self.axit))
        except StopIteration:
            return

    @property
    def ax(self):
        if self._ax is None:
            self._ax = next(self.axit)
        return self._ax

# Great Formatter
class GreatFormatter(ticker.ScalarFormatter):
    def __init__(self, sci=True, power_range=(-3,3), offset=True, mathText=True):
        super().__init__(useOffset=offset,useMathText=mathText)
        self.set_powerlimits(power_range)
        self.set_scientific(sci)

    def _set_order_of_magnitude(self):
        super()._set_order_of_magnitude()
        self.orderOfMagnitude = int(3 * np.floor(self.orderOfMagnitude / 3))


# Another figure/plotter manager: manages cycles for color/black and white
class FigureManager():
    """


    """
    def __init__(self, cycle='c', lw=1.5, color_mode='mono', k=0.8, font_size=12,
                 legend_font='', use_tex=False, default_figsize=(5, 3.5)):
        """
        Font size was 9 and legend was x-small

        Create figure with common defaults

        cycle = cws
            c - cycle colors
            w - cycle widths
            s - cycle styles
            o - styles x colors, implies csw and w=single number (produces 8 series)

        lw = default line width or [lws] of length 4

        smaller k overall darker lines; colors are equally spaced between 0 and k
        k=0.8 is a reasonable range for four colors (0, k/3, 2k/3, k)

        https://matplotlib.org/3.1.1/tutorials/intermediate/color_cycle.html

        https://matplotlib.org/3.1.1/users/dflt_style_changes.html#colors-in-default-property-cycle

        https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html

        https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html

        https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
        """

        assert len(cycle) > 0

        # this sets a much smaller base fontsize
        # plt.rcParams.update({'font.size': 12})
        # plt.rcParams.update({'axes.titlesize': 'large'})
        # plt.rcParams.update({'axes.labelsize': 'small'})
        # list(map(plt.rcParams.get, ('axes.titlesize', 'font.size')))
        plt.rcParams.update({'font.size': font_size})
        self.default_figsize = default_figsize

        if legend_font != '':
            plt.rcParams['legend.fontsize'] = legend_font

        plt.rc('font', family='serif')
        plt.rc('text', usetex=use_tex)

        if color_mode == 'mono':
            # https://stackoverflow.com/questions/20118258/matplotlib-coloring-line-plots-by-iteration-dependent-gray-scale
            # default_colors = ['black', 'grey', 'darkgrey', 'lightgrey']
            default_colors = [(i*k, i*k, i*k) for i in [0, 1/3, 2/3, 1]]
            default_ls = ['solid', 'dashed', 'dotted', 'dashdot']

        else:
            # print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
            default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
                              '#7f7f7f', '#bcbd22', '#17becf']
            default_ls = ['solid', 'dashed', 'dotted', 'dashdot', (0, (5, 1))] * 2

        props = []
        if 'o' in cycle:
            n = len(default_colors) // 2
            if color_mode == 'mono':
                cc = [i[1] for i in product(default_ls, default_colors[::2])]
            else:
                cc = [i[1] for i in product(default_ls, default_colors[:n])]
            lsc = [i[0] for i in product(default_ls, default_colors[:n])]
            props.append(cycler('color', cc))
            props.append(cycler('linewidth', [lw] * (len(default_colors) * len(default_ls) // 2)))
            props.append(cycler('linestyle', lsc))
        else:
            if 'c' in cycle:
                props.append(cycler('color', default_colors))
            else:
                props.append(cycler('color', [default_colors[0]] * len(default_ls)))
            if 'w' in cycle:
                if type(ls) == int:
                    props.append(cycler('linewidth', [lw] * len(default_colors)))
                else:
                    props.append(cycler('linewidth', lw))
            if 's' in cycle:
                props.append(cycler('linestyle', default_ls))

        # combine all cyclers
        cprops = props[0]
        for c in props[1:]:
            cprops += c

        mpl.rcParams['axes.prop_cycle'] = cycler(cprops)
        # for a in axs.flatten():
        #     a.set_prop_cycle(cprops)
        self.last_fig = None

    def make_fig(self, nr=1, nc=1, figsize=None, xfmt='great', yfmt='great',
                places=None, power_range=(-3,3), sep='', unit='', sci=True,
                mathText=True, offset=True, **kwargs):
        """

        make grid of axes
        apply format to xy axes

        xfmt='d' for default axis formatting, n=nice, e=engineering, s=scientific, g=great
        great = engineering with power of three exponents

        """

        if figsize is None:
            figsize = self.default_figsize

        f, axs = plt.subplots(nr, nc, figsize=figsize, constrained_layout=True, squeeze=False, **kwargs)
        for ax in axs.flat:
            if xfmt[0] != 'd':
                FigureManager.easy_formatter(ax, which='x', kind=xfmt, places=places,
                    power_range=power_range, sep=sep, unit=unit, sci=sci, mathText=mathText, offset=offset)
            if yfmt[0] != 'default':
                FigureManager.easy_formatter(ax, which='y', kind=yfmt, places=places,
                    power_range=power_range, sep=sep, unit=unit, sci=sci, mathText=mathText, offset=offset)

        if nr * nc == 1:
            axs = axs[0, 0]

        self.last_fig = f
        return f, axs

    __call__ = make_fig

    def save_fig(self, fn, **kwargs):
        """
        Save the last figure at fn

        Mostly will be saved by the docmaker

        :param fn:
        :return:
        """
        if self.last_fig:
            self.last_fig.savefig(fn, **kwargs)

    @staticmethod
    def easy_formatter(ax, which, kind, places=None, power_range=(-3,3), sep='', unit='', sci=True,
        mathText=False, offset=True):
        """
        set which (x, y, b, both) to kind = sci, eng, nice
        nice = engineering but uses e-3, e-6 etc.
        see docs for ScalarFormatter and EngFormatter


        """
        def make_fmt(kind, places, power_range, sep, unit):
            if kind=='sci' or kind[0]=='s':
                fm = ticker.ScalarFormatter()
                fm.set_powerlimits(power_range)
                fm.set_scientific(True)
            elif kind=='eng' or kind[0]=='e':
                fm = ticker.EngFormatter(unit=unit, places=places, sep=sep)
            elif kind=='great' or kind[0]=='g':
                fm = GreatFormatter(sci=sci, power_range=power_range, offset=offset, mathText=mathText)
            elif kind=='nice' or kind[0]=='n':
                fm = ticker.EngFormatter(unit=unit, places=places, sep=sep)
                fm.ENG_PREFIXES = { i: f'e{i}' if i else '' for i in range(-24, 25, 3)}
            else:
                raise ValueError(f'Passed {kind}, expected sci or eng')
            return fm

        # what to set
        if which=='b' or which=='both':
            which = ['xaxis', 'yaxis']
        elif which=='x':
            which = ['xaxis']
        else:
            which = ['yaxis']

        for l in which:
            fm = make_fmt(kind, places, power_range, sep, unit)
            getattr(ax, l).set_major_formatter(fm)

    @staticmethod
    def plot_examples():
        x = np.linspace(0, 2 * np.pi)
        offsets = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        # Create array with shifted-sine curve along each column
        yy = np.transpose([np.sin(x + phi) for phi in offsets])

        plt.rc('lines', linewidth=4)
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) +
                                   cycler('linestyle', ['-', '--', ':', '-.'])))
        fig, (ax0, ax1) = plt.subplots(nrows=2)
        ax0.plot(yy)
        ax0.set_title('Set default color cycle to rgby')

        ax1.set_prop_cycle(cycler('color', ['c', 'm', 'y', 'k']) +
                           cycler('lw', [1, 2, 3, 4]) + cycler('linestyle', ['-', '--', ':', '-.']))
        ax1.plot(yy)
        ax1.set_title('Set axes color cycle to cmyk')

        # Tweak spacing between subplots to prevent labels from overlapping
        fig.subplots_adjust(hspace=0.3)

    @staticmethod
    def formatter_examples():
        smfig = FigureManager(color_mode='col')

        f, ax = smfig(1,3, (12,4))
        a1, a2, a3 = ax.flat
        xs = np.linspace(-1e-2, 1e-2, 10000)
        a1.plot(xs, xs * xs * np.sin(1/xs), lw=0.25)
        a1.grid(lw=0.25)

        FigureManager.easy_formatter(a2, 'b', 'eng')
        a2.plot(xs, xs * np.sin(1/xs), lw=0.25)
        a2.grid(lw=0.25)

        FigureManager.easy_formatter(a3, 'b', 'sci', power_range=(-1,1))
        a3.plot(xs, np.sin(1/xs), lw=0.25)
        a3.grid(lw=0.25)

        return f, ax


@magics_class
class GreatMagics(Magics):
    """
    Note: GreatMagics looks for a global variable smfig = FigureManager()

    https://mlexplained.com/2017/12/28/creating-custom-magic-commands-in-jupyter/

    """

    @cell_magic
#     @magic_arguments.magic_arguments()
#     @magic_arguments.argument('--verbose', '-v',
#           help='Print expanded text results')
#     @magic_arguments.argument('-h',
#           action='store_true',
#           help='specify individual height')
#     @magic_arguments.argument('-w',
#           action='store_true',
#           help='specify individual width')
    def sf(self, line='', cell=None):
        """
        s[m]f[ig] 1 2 -w
        """
#         args = magic_arguments.parse_argstring(self.sf, line)
#         if args.h is None:
#             hh = args.h
#         else:
#             hh = 3.25

#         if args.w is None:
#             ww = 4.0
#         else:
#             ww = args.w

        ww, hh = 4, 3.25
        verbose = False
        if line != '':
            if line.find('v') >= 0:
                verbose = True
                line = line.replace('v', '').strip()
        if line != '':
            ls = line.replace(',', ' ').split()
            nr = int(ls[0])
            nc = int(ls[1])
            if len(ls) > 2:
                ww = float(ls[2])
                hh = float(ls[3])
        else:
            nr, nc = 1, 1

        w = nc * ww
        h = nr * hh

        if nr * nc == 1:
            ax = 'ax = ax0 = axs\n'
        else:
            ax = ",".join([f'ax{i}' for i in range(nr*nc)]) + ' = axs.flat\n'

        s = f'f, axs = smfig({nr}, {nc}, ({w}, {h}))\n{ax}{cell}'
        if verbose:
            print(s)

        self.shell.ex(s)

ip = get_ipython()
ip.register_magics(GreatMagics)
