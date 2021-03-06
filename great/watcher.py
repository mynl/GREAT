from pathlib import Path
from datetime import datetime
from time import sleep
from .markdown_make import markdown_make_main
# import argparse
import os


class Watcher(object):

    # __slots__ = ('watched_paths', 'wait')

    def __init__(self, *argv):
        """
        add file to list of watched files

        :param argv: list of file names
        :return:
        """
        self.watched_paths = {}
        self.wait = 1
        for fn in argv:
            if type(fn) == str:
                p = Path(fn)
            else:
                p = fn
            if p.exists():
                self.watched_paths[p] = datetime.fromtimestamp(p.stat().st_mtime)
                print(f'Watching file {fn}')
            else:
                print(f'File {fn} does not exist...will watch for its future creation')
                self.watched_paths[p] = datetime.now()

    def watch(self, wait):
        """
        watch for changes in watched_path files

        :param wait:
        :return:
        """
        count = 0
        updates = 0
        if len(self.watched_paths):
            print(f'Initiating watching on {len(self.watched_paths)} files.')
        else:
            print('No files...exiting')
            return
        while 1:
            sleep(wait)
            count += 1
            if count % 300 == 0:
                print(f'Still monitoring...{updates} updates performed.')
            for k, v in self.watched_paths.items():
                if k.exists():
                    udt = datetime.fromtimestamp(k.stat().st_mtime)
                    if udt > v:
                        updates += 1
                        self.watched_paths[k] = udt
                        self.safe_make(k, udt)

    @staticmethod
    def safe_make(path, udt):
        """
        safe make path file
        :param path:
        :return:
        """
        cwd = Path.cwd()
        os.chdir(path.parent)
        print(f'Re-making {path.name} at {udt}', path.parent.resolve(), path.name, path.stem)
        markdown_make_main("", path.name, path.stem)
        os.chdir(cwd)


#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Watch files and make on change')
#     parser.add_argument('-f', '--file_name', nargs='+',
#                         action="store", type=str, dest="file_name", required=False,
#                         help='File name to watch')
#     parser.add_argument('-g', '--glob',
#                         action="store", type=str, required=False,
#                         help='File name to watch')
#     parser.add_argument('-w', '--wait',
#                         action="store", type=int, dest="wait", default=1, required=False,
#                         help='Wait time between checks (seconds), default=1.')
#     # parse args
#     args = parser.parse_args()
#     if args.file_name is None:
#         files = []
#     else:
#         files = args.file_name
#     if args.glob is not None and args.glob != '':
#         for f in Path().glob(args.glob):
#             files.append(f)
#     add_watch(*files)
#     watch(args.wait)