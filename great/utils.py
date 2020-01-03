"""
Misc. utilities

"""

import hashlib
import pandas as pd
import struct
import numpy as np


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
