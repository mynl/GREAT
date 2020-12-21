"""

MarkdownMake

all based on /s/tbd/python/markdown-make.py
see there for previous version
May 2018: removed extraneous code and tidied up
Dec 2019: possible issue with repeated cla's that need to be stripped from YAML
MOVED here to DocMaker for future development

v 1.0 Dec 2019

"""

import os
import hashlib
import sys
import re
import subprocess
import glob
import string
import unicodedata
import datetime
import pathlib
import shutil
import re

# for MarkdownMake
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_CONSOLE = 0x00000010
TEMP_FILE_NAME = 'dm-temp'
BEGIN_STRING = '\\begin{document}'
END_STRING = '\\end{document}'
LOCAL_YAML = 'local_build.yaml'  # name of default yaml files for empty headers


def markdown_make_main(*argv):
    """
    When called from ST3 argv will be like
    ['\\s\\tbd\\python\\Markdown_Make.py',
    'C:\\S\\Teaching\\2018-01\\ACT3349\\Notes\\Final.md', 'Final']

    When called internally (as a burst) may also include ydic which is then
    used as the spec for all builds first arg is the name of this script
    second arg is the full name of the file being processes
    third arg is the stripped name of the file.

    This is specified in the build file:
    "cmd": ["python", "\\s\\tbd\\python\\Markdown_Make.py", "$file", "$file_base_name"]
    so there are other options.

    This batch file will open the md file, read the yaml and proceed accordingly.
    YAML can set variables...that's it
    This batch file is responsible for setting command line arg_dict
    Args you want set are prefixed with clarg:

    Pass through command line arg_dict specified in yaml with cla
    (command line argument) xcla will be ignored

    format files: will be specified in the templates...so there will be a
    sjm-beamer-template which will refer to the right fmt file etc.
    to and from formats are specified in the cla

    there is no include-header - all these options are in the base yaml

    there will be a header-includes: but that is a standard env variable

    will keep debug

    burst: true/false, build burst true/false, output directory default is c
    temp burst (which has img and pdf)
    creating link: > mklink /j img c:\\s\\telos\\notes\\img

    if there is no YAML at the top looks up the directory tree for a file
    called LOCAL_YAML and uses that in place.

    mode = quick: run pandoc and then pdflatex separately (avoids multiple
    runs); if there is a template
    it is used via a first line %&/s/telos/common/sjm-doc-template taken from
    the YAML

    :param argv:
    :return:
    """

    debug = 0
    def dp(x):
        if debug: print(x)

    print('RUNNING DOCMAKER VERSION ' * 5)


    if len(argv) == 0:
        argv = sys.argv

    ydic = {}

    # update biblio file if we appear to be processing the book...
    book = pathlib.Path(argv[1])
    making_book = False
    if str(book.parent.absolute()).find('spectral_risk_measures_monograph') >= 0 or \
       str(book.parent.absolute()).find('book-hack'):
        making_book = True
        update_book_bibfile()

    # fix the date
    insert_date(argv[1])

    PATTERN = re.compile(r'''((?:[^:"']|"[^"]*"|'[^']*')+)''')
    yaml = False
    with open(argv[1], 'r', encoding='UTF-8') as g:
        txt = g.read()
        g.seek(0)
        # read in (when processing burst pages ydic is passed in)
        for l in g:
            if l == '---\n':
                yaml = not yaml
            if not yaml:
                break
            # need to only find the first :, beware : in quotes...
            s = PATTERN.split(l)[1::2]
            # beware here: this fucks up if there is a colon on the right...change PATTERN?
            if len(s) == 2:
                # here we are going to catch -o XXX because easier than doing later
                ans = s[1].strip()
                if ans[0:3] == '-o ':
                    ans = '-o pdf/{:}.{:}'.format(argv[2], ans[3:])
                if s[0] in ydic.keys():
                    ydic[s[0]] += ans.split(' ')
                else:
                    ydic[s[0]] = ans.split(' ')

    # pandoc started complaining about multiple cla args in YAML so take them out
    # print(ydic)
    # fa = re.findall(r'^cla[^\n]+\n', txt, re.MULTILINE)
    # print(fa)
    txt, n = re.subn(r'^cla[ ]*:[^\n]+\n', '', txt, flags=re.MULTILINE)
    print(f'Removed {n} cla clauses from txt')

    # put in defaults if there is no YAML - allows easy creation of slides
    add_non_cla = False
    if len(ydic) == 0:
        ydic = get_default_yaml(argv[1])
        add_non_cla = True

    # insert_date function does NOT work for local-yaml builds
    dt = '"Created {date:%Y-%m-%d %H:%M:%S.%f}"'.format(date=datetime.datetime.now()).rstrip('0')
    if ydic.get('date', None):
        # print('Date tester: ', ydic['date'], ydic['date'][0])
        if ydic['date'][0].lower() in ['now', 'created', 'insert date', 'date']:
            # may or may not be in the text...
            if add_non_cla:
                print('Replacing date: now with (no yaml) ', dt)
                ydic['date'] = [dt]
            else:
                # this is a little fragile...
                dt = 'date: "Created {date:%Y-%m-%d %H:%M:%S.%f}"'.format(date=datetime.datetime.now())
                print('Replacing date: now with ', dt)
                txt = txt.replace('date: now', dt)

    # process @@@include statements
    dn = os.path.split(argv[1])[0]
    n_includes = 0
    if txt.find('@@@') > 0:
        # have work to do
        # first, substitute for all NNN specs
        # note, obvious issue if the first three chars do not
        # uniquely identify file
        # assumes you are in the current directory
        file_map = {i[0:3]: i for i in glob.glob("*.md")}
        color_includes = ydic.get('color-includes', False)
        if color_includes:
            color_includes = color_includes[0]
        txt, n_includes = process_includes(txt, dn, color_includes, n_includes, file_map)
        # print('processing includes', n_includes)

    if ydic.get('burst', '') != '':
        # burst mode: burst: [do burst t/f], [do build t/f], outdir
        cmd = ' '.join(ydic.get('burst', '')).split(',')
        do_burst = cmd[0].strip().lower()
        do_build = cmd[1].strip().lower()
        if len(cmd) == 2:
            base_dir = 'c:\\temp\\burst'
        else:
            base_dir = cmd[2:]
        # tidy base_dir
        # for f in glob.glob(os.path.join(base_dir, '*.md')):
        #     print(f'removing {f}')
        #     os.remove(f)
        if do_burst == 'true':
            print(f'bursting to {base_dir}')
            burst_file(txt, base_dir)
        print(do_build)
        if do_build == 'true':
            print('building burst...')
            cwd = os.getcwd()
            os.chdir(base_dir)
            proc_files(base_dir)
            os.chdir(cwd)
        return 0

    # create new file for processing in all cases because need to strip cla clauses
    comments = (ydic.get('comments', [1])[0] == 'show')
    fn = hashlib.md5(txt.encode('utf-8')).hexdigest()
    nfn = os.path.join(dn, f'TMP_{fn}.md')
    if comments:
        print('Revealing comments...')
        txt = show_comments(txt)
    with open(nfn, 'w', encoding='utf-8') as f:
        if add_non_cla:
            f.write('---\n')
            for k, v in ydic.items():
                if k not in ['cla', 'xcla', 'debug', 'xdebug']:
                    f.write('{:}:\t{:}\n'.format(k, ' '.join(v)))
            f.write('---\n')
        f.write(txt)

    # build the argument list
    args = ["\\users\\{:}\\appdata\\local\\pandoc\\pandoc.exe".format(os.getlogin())]

    # append all cla's from the yaml
    args += ydic.get('cla', [])

    # add the name of the file being processes!
    args += [nfn]

    data_dir = ''
    aux_dir = ''
    if 'cla' in ydic:
        for c in ydic['cla']:
            if c[0:10] == '--data-dir':
                data_dir = c.split('=')[1]
                aux_dir = os.path.join(data_dir, 'templates')
    if data_dir == '':
        data_dir = '/s/telos/common'
        aux_dir = '/s/telos/common/'

    debug_mode = ydic.get('debug', ['no'])[0].lower()
    # print(debug_mode)
    dp(f'ready to execute\n{debug_mode}\n{ydic}\n{args}')

    if making_book:
        # create a batch file of the build, often handy to have
        with open('make_last.bat', 'w', encoding='utf-8') as f:
            f.write('REM Last build\n')
            f.write(f'REM {dt}\n')
            out = ' '.join(args)
            f.write(out)
            f.write(f'\nREM TeX Output: uncomment next line\n')
            f.write('REM ')
            f.write(re.sub(r'\\pdf\\(.*)\.pdf', r'\1.tex', out))

    if debug_mode not in ('quick', 'maketexformat'):
        print('EXECUTING MARKDOWN BUILD\n\n{:}\n'.format(' '.join(args)))
        # print('RAW args')
        # print(args)
    if debug_mode in ('no run', 'norun'):
        print('NO RUN MODE...no further processing\n\n')
        print(ydic['cla'])
        # print(ydic)
        print('\ndata-dir', data_dir, '\naux_dir', aux_dir)
    elif debug_mode in ('quick'):
        # ASSUMES: building to beamer or tex making a pdf...this will work by making a tex file and then pdflatex'ing it
        # run pandoc
        args, new_filename = adjust_output_file(args, 'temp')
        print('EXECUTING pandoc\n\n{:}\n'.format(' '.join(args)))
        p = subprocess.Popen(args, creationflags=CREATE_NO_WINDOW, stdout=subprocess.PIPE)
        p.communicate()
        # then run latex
        tex_args = ['pdflatex', '-output-directory=pdf', '-aux-directory=temp', new_filename]
        print('EXECUTING tex\n\n{:}\n'.format(' '.join(tex_args)))
        p = subprocess.Popen(tex_args, creationflags=CREATE_NO_WINDOW, stdout=subprocess.PIPE)
        p.communicate()
    elif debug_mode == 'maketexformat':
        # make tex format file based on this file
        # finds tex filename name looking in the pandoc template file
        # creates a tex file for the fmt by stripping the contents out of the calling program
        print('making tex format...')
        args, template_name = remove_template(args, TEMP_FILE_NAME)
        dp(f'Temp filename: {TEMP_FILE_NAME}')
        print('EXECUTING pandoc\n\n{:}\n'.format(' '.join(args)))

        p = subprocess.Popen(args, creationflags=CREATE_NO_WINDOW, stdout=subprocess.PIPE)
        p.communicate()

        # gamble that you do NOT actually need to strip out the doc...the latexformat appears to indicate that...
        # for now will actually edit it
        out_file_name = os.path.join(data_dir, '{:}.tex'.format(TEMP_FILE_NAME)).replace('\\',
                                                                                         '/')  # this is a throwaway file
        template_args = ['pdflatex', '-aux-directory={:}'.format(aux_dir), '-ini', '-jobname="' + template_name + '"',
                         '"&pdflatex"',
                         "mylatexformat.ltx", out_file_name]
        print('EXECUTING tex format build\n\n' + ' '.join(template_args))
        print('\n')
        p = subprocess.Popen(' '.join(template_args), creationflags=CREATE_NO_WINDOW, stdout=subprocess.PIPE)
        p.communicate()

    elif debug_mode in ('true', 'yes', 'on', 'debug'):
        p = subprocess.Popen(args, creationflags=CREATE_NO_WINDOW, stdout=subprocess.PIPE)
        p.communicate()
    else:
        subprocess.Popen(args, creationflags=CREATE_NO_WINDOW)

    return 0


def process_includes(txt, dn, color_includes, n_includes, file_map):
    """
    Iterative processing of include files
    file_map looks for nnn_something.md files in the current directory
    dn = directory name
    """
    # changed pattern to allow importing nonmarkdown files, e.g. py sourcefiles
    # but, WTF, is this re??
    includes = re.findall(r'@@@include (\.\./)?([0-9]{3}|[0-9A-Za-z])([^\n]+\.[a-z]+)?', txt)
    # print(includes)
    for res_ in includes:
        original_match = res = ''.join(res_)
        # print(res_, file_map)
        # res_[1] looks for nnn type files and tries to find them in file_map
        if res_[2] == '':
            res = file_map[res_[1]]
            # print(f'REPLACING {res_} with {res}')
        else:
            res = original_match
            # print(f'using {"".join(res_)} as {res}')
        print(f'Importing {res}')
        n_includes += 1
        try:
            with open(os.path.join(dn, res), 'r', encoding='utf-8') as f:
                repl = f.read()
                repl, n_includes = process_includes(repl, dn, False, n_includes, file_map)
                if color_includes:
                    repl = f'''
\\color{{{color_includes}}}
{repl}
\\color{{black}}
'''
                txt = txt.replace(f'@@@include {original_match}', repl)
        except FileNotFoundError:
            print(f'WARNING: File {res} not found...ignoring')
    return txt, n_includes


def adjust_output_file(args, dir_name, new_output_name=''):
    # swap pdf/xx.pdf output for dir_name/xx.tex
    new_filename = ''
    for i in args[::-1]:
        if i == '-o':
            args.remove(i)
        if i[0:4] == 'pdf/':
            args.remove(i)
            if new_output_name == '':
                new_output_name = i[4:-4]
            new_filename = '{:}/{:}.tex'.format(dir_name, new_output_name)
    args.append('-o')
    args.append(new_filename)
    return args, new_filename


def remove_template(args, output_file_name):
    # remove the arg to use template
    template_name = ''
    dir_name = ''
    template_found = False
    for i in args:
        if i[0:10] == '--template':
            # print('FOUND')
            args.remove(i)
            # i ='--template=/s/telos/common/sjm-doc-template.tex'
            eq_loc = i.find("=") + 1
            dir_end = len(i) - i[::-1].find('/') - 1
            dir_name = i[eq_loc:dir_end]
            template_name = i[dir_end + 1:-4]
            template_found = True
            # print(template_name)
            break
    if not template_found:
        raise ValueError('\n\n\nERROR: making template, need cla: --template=/template name... command line option!\nAborting.\n\n')
        return
    args, trash = adjust_output_file(args, dir_name, output_file_name)
    return args, template_name


# from pandoc_debug.py
# test burst
# base_dir = 'c:\\temp\\burst'
# with open('c:\\s\\telos\\notes\\32.022.Verisk_blockchain.md') as f:
#     txt = f.read()
#
# burst_file(txt, base_dir)

def burst_file(txt, base_dir):
    """
    burst file into subfiles in base_dir
    name given by page no

    :param txt:
    :param base_dir:
    :return:
    """

    page_split = re.split('(\n##?[ \n]|\n\-\-[\-]+\n)', txt)

    # is there YAML to split off?
    if page_split[0] == '':
        # does not start with yaml
        start_at = 1
    else:
        # starts with yaml
        start_at = 3

    for i, (tag, p) in enumerate(zip(page_split[start_at::2], page_split[start_at + 1::2])):
        # print(i, tag, p.strip())
        if p[0:3] != '@@@':
            # do not bother with slides that are includes (could have multi includes on a page
            # ignore that for now)
            # find a possible name
            pn = p.strip().split('\n')[0]
            if pn[0] not in ('\n', '\\'):
                fn = clean_filename(pn) + '.md'
            else:
                fn = f'page_{i:03d}.md'
            # print(f'possible name: {fn}')
            # fn = f'page_{i:03d}.md'
            ffn = os.path.join(base_dir, fn)
            with open(ffn, 'w', encoding='UTF-8') as f:
                if tag[0:4] == '\n---':
                    f.write(p)
                else:
                    f.write(tag + p)


def clean_filename(filename):
    """
    file name maker: removes unacceptable filename characers such as * and ?

    :return:
    """
    validFilenameChars = f"-_.() {string.ascii_letters}{string.digits}"
    cleanedFilename = unicodedata.normalize('NFKD', filename)
    return ''.join(c for c in cleanedFilename if c in validFilenameChars).title().replace(' ', '_')


def proc_files(base_dir):
    """

    :param base_dir:
    :param ydic:    dictionary of pandoc args
    :return:
    """

    ffns = glob.glob(os.path.join(base_dir, '*.md'))
    for ffn in ffns:
        fn = os.path.split(ffn)[1][:-3]
        # print(f'calling main {ffn}, {fn}')
        main("", ffn, fn)


def get_default_yaml(fn):
    """
    Find and read the default yaml file for fn, fn = full path to markdown file
    as passed to markdown make as argv[1]

    First it finds a 'local_build.yaml' file by looking in the directory containing
    fn and then iterating up all its parent directories. Stops the first file it
    finds.

    Opens the yaml and performs a standard markdown make parse. Removes the abstract.

    Puts in the -o file name with appropriate extension

    Returns populated ydic

    """
    ydic = {}
    PATTERN = re.compile(r'''((?:[^:"']|"[^"]*"|'[^']*')+)''')
    yaml = False
    p = pathlib.Path(fn)
    fn_ = p.stem
    dn = p.parents[0]

    # look for local build file
    for path in p.parents:
        pl = path / LOCAL_YAML
        if pl.exists():
            print('Using default yaml {:}'.format(pl))
            break

    with open(pl, 'r', encoding='UTF-8') as g:
        # read in (when processing burst pages ydic is passed in)
        for l in g:
            if l == '---\n':
                yaml = not yaml
            if not yaml:
                break
            # need to only find the first :, beware : in quotes...
            s = PATTERN.split(l)[1::2]
            if len(s) == 2:
                # here we are going to catch -o XXX because easier than doing later
                ans = s[1].strip()
                if ans[0:3] == '-o ':
                    ans = '-o {:}{:}'.format(p.parents[0] / 'pdf' / p.stem, '.pdf')
                if s[0] in ydic.keys():
                    ydic[s[0]] += ans.split(' ')
                else:
                    ydic[s[0]] = ans.split(' ')
    # certain things we ignore
    for k in ['abstract']:
        if k in ydic:
            del ydic[k]
    # print(ydic)
    ydic['title'] = [p.stem.replace('_', ' ').title()]
    return ydic


def insert_date(fn):
    """
    If there is YAML put the date in to line 3 (---, title, author, date)
    """
    DATE_LINE = 3
    p = pathlib.Path(fn)

    with open(p, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if lines[0] == '---\n':
            dt = 'date: "Created {date:%Y-%m-%d %H:%M:%S.%f}"\n'.format(
                date=datetime.datetime.now()).rstrip('0')
            if lines[DATE_LINE].find('Created') >= 0 or lines[DATE_LINE].find('now') >= 0:
                lines[DATE_LINE] = dt
                print('Replacing date: now with ', dt)
            else:
                print("Retaining original date")
            with open(p, 'w', encoding='utf-8') as f:
                f.writelines(lines)


def show_comments(txt):
    """
    show all <!-- xxx --> comments
    """

    start = '\\color{blue}\n\\footnotesize\n\\begingroup\n\\leftskip8em\n\\rightskip\\leftskip'
    end = '\n\\endgroup\n\\color{black}\n\\normalsize '
    lstart = '\\color{blue} \['
    lend = '\] \\color{black}'

    stext = txt.split('\n')

    out = []

    open_groups = 0

    for i, l in enumerate(stext):
        if l == '<!--':
            l = start
            open_groups += 1
            # print('open group...')
        elif l == '-->':
            if open_groups == 1:
                open_groups = 0
                # print('...close group')
            else:
                raise ValueError('Close para group without open at line {}\nPrevious lines {}'.format(l, stext[i-3:i]))
            l = end
        else:
            if (l.find('<!--') >= 0 or l.find('-->') >= 0) and open_groups:
                ValueError('Apparent {} nested groups at line {}'.format(open_groups, l))
            l = l.replace('<!--', lstart).replace('-->', lend)
        out.append(l)

    txt = '\n'.join(out)

    return txt

def update_book_bibfile():
    print('Updating bibliography file...')
    p_from = pathlib.Path('/s/telos/biblio/library.bib')
    p_to = pathlib.Path('/s/telos/spectral_risk_measures_monograph/docs/library.bib')
    shutil.copy(p_from, p_to)

if __name__ == '__main__':
    markdown_make_main()
