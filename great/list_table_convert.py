# convert markdown pipe table into rst list table
# result will still need editing but will be pretty good

from pyperclip import copy, paste
from pypandoc import convert_text
import re

txt = paste()

stxt = txt.strip().split('\n')
# drop first row
stxt.pop(1)

ans = []

ans.append('''.. list-table:: Frozen Delights!
\t:widths: 25 25 25 25
\t:header-rows: 1\n\n''')

for r in stxt:
    sr = r.strip().split('|')[1:-1]
    if len(sr):
        sr = [convert_text(i.strip().replace('\\E', '\\mathsf E'), to='rst', format='markdown') for i in sr]
        ans.append(f'\t* - {sr[0]}'.replace('\r\n', ''))
        ans.extend([f'\t  - {i}'.replace('\r\n', '') for i in sr[1:]])

copy('\n'.join(ans))


# | Code                             | Distribution | Meaning         |
# |:---------------------------------|:-------------|:----------------|
# | `sev lognorm 10 cv 3`          | lognormal    | mean 10, cv 0.3 |
# | `sev 10 * lognorm 1.75`        | lognormal    | $10X$, $X$ lognormal($\mu=0,\sigma=1.75$)  |
# | `sev 10 * lognorm 1.75 + 20`   | lognormal    | $10X + 20$      |
# | `sev 10 * lognorm 1 cv 3 + 50` | lognormal    | $10Y + 50$, $Y$ lognormal mean 1 cv 3 |
# | `sev 100 * pareto 1.3 - 100`   | Pareto       | Pareto, survival $(100/(100+x))^{1.3}$    |
# | `sev 50 * normal + 100 `       | normal       | mean 100, std dev 50  |
# | `sev 5 * expon`                | exponential  | mean 5          |
# | `sev 5 * uniform + 1`          | uniform      | uniform between 1 and 6|
# | `sev 50 * beta 2 3`            | beta         | $50Z$, $Z$ beta parameters 2, 3 |
