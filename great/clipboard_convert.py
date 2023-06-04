# convert clipboard from markdown to rst

from pyperclip import copy, paste
from pypandoc import convert_text

out = convert_text(paste(), to='rst', format='markdown')

copy(out)
