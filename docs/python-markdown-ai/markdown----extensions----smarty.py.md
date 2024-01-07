# `markdown\markdown\extensions\smarty.py`

```

# Smarty extension for Python-Markdown
# ====================================

# Adds conversion of ASCII dashes, quotes and ellipses to their HTML
# entity equivalents.

# See https://Python-Markdown.github.io/extensions/smarty
# for documentation.

# Author: 2013, Dmitry Shachnev <mitya57@gmail.com>

# All changes Copyright 2013-2014 The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

# SmartyPants license:

#    Copyright (c) 2003 John Gruber <https://daringfireball.net/>
#    All rights reserved.

#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:

#    *  Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.

#    *  Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the
#       distribution.

#    *  Neither the name "SmartyPants" nor the names of its contributors
#       may be used to endorse or promote products derived from this
#       software without specific prior written permission.

#    This software is provided by the copyright holders and contributors "as
#    is" and any express or implied warranties, including, but not limited
#    to, the implied warranties of merchantability and fitness for a
#    particular purpose are disclaimed. In no event shall the copyright
#    owner or contributors be liable for any direct, indirect, incidental,
#    special, exemplary, or consequential damages (including, but not
#    limited to, procurement of substitute goods or services; loss of use,
#    data, or profits; or business interruption) however caused and on any
#    theory of liability, whether in contract, strict liability, or tort
#    (including negligence or otherwise) arising in any way out of the use
#    of this software, even if advised of the possibility of such damage.


# `smartypants.py` license:

#    `smartypants.py` is a derivative work of SmartyPants.
#    Copyright (c) 2004, 2007 Chad Miller <http://web.chad.org/>

#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:

#    *  Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.

#    *  Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the
#       distribution.

#    This software is provided by the copyright holders and contributors "as
#    is" and any express or implied warranties, including, but not limited
#    to, the implied warranties of merchantability and fitness for a
#    particular purpose are disclaimed. In no event shall the copyright
#    owner or contributors be liable for any direct, indirect, incidental,
#    special, exemplary, or consequential damages (including, but not
#    limited to, procurement of substitute goods or services; loss of use,
#    data, or profits; or business interruption) however caused and on any
#    theory of liability, whether in contract, strict liability, or tort
#    (including negligence or otherwise) arising in any way out of the use
#    of this software, even if advised of the possibility of such damage.

"""
Adds conversion of ASCII dashes, quotes and ellipses to their HTML
entity equivalents.

See the [documentation](https://Python-Markdown.github.io/extensions/smarty)
for details.
"""

from __future__ import annotations

from . import Extension
from ..inlinepatterns import HtmlInlineProcessor, HTML_RE
from ..treeprocessors import InlineProcessor
from ..util import Registry
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from markdown import Markdown
    from .. import inlinepatterns
    import re
    import xml.etree.ElementTree as etree

# Constants for quote education.
punctClass = r"""[!"#\$\%'()*+,-.\/:;<=>?\@\[\\\]\^_`{|}~]"""
endOfWordClass = r"[\s.,;:!?)]"
closeClass = r"[^\ \t\r\n\[\{\(\-\u0002\u0003]"

openingQuotesBase = (
    r'(\s'               # a  whitespace char
    r'|&nbsp;'           # or a non-breaking space entity
    r'|--'               # or dashes
    r'|–|—'              # or Unicode
    r'|&[mn]dash;'       # or named dash entities
    r'|&#8211;|&#8212;'  # or decimal entities
    r')'
)

substitutions = {
    'mdash': '&mdash;',
    'ndash': '&ndash;',
    'ellipsis': '&hellip;',
    'left-angle-quote': '&laquo;',
    'right-angle-quote': '&raquo;',
    'left-single-quote': '&lsquo;',
    'right-single-quote': '&rsquo;',
    'left-double-quote': '&ldquo;',
    'right-double-quote': '&rdquo;',
}


# Special case if the very first character is a quote
# followed by punctuation at a non-word-break. Close the quotes by brute force:
singleQuoteStartRe = r"^'(?=%s\B)" % punctClass
doubleQuoteStartRe = r'^"(?=%s\B)' % punctClass

# Special case for double sets of quotes, e.g.:
#   <p>He said, "'Quoted' words in a larger quote."</p>
doubleQuoteSetsRe = r""""'(?=\w)"""
singleQuoteSetsRe = r"""'"(?=\w)"""

# Special case for decade abbreviations (the '80s):
decadeAbbrRe = r"(?<!\w)'(?=\d{2}s)"

# Get most opening double quotes:
openingDoubleQuotesRegex = r'%s"(?=\w)' % openingQuotesBase

# Double closing quotes:
closingDoubleQuotesRegex = r'"(?=\s)'
closingDoubleQuotesRegex2 = '(?<=%s)"' % closeClass

# Get most opening single quotes:
openingSingleQuotesRegex = r"%s'(?=\w)" % openingQuotesBase

# Single closing quotes:
closingSingleQuotesRegex = r"(?<=%s)'(?!\s|s\b|\d)" % closeClass
closingSingleQuotesRegex2 = r"'(\s|s\b)"

# All remaining quotes should be opening ones
remainingSingleQuotesRegex = r"'"
remainingDoubleQuotesRegex = r'"'

HTML_STRICT_RE = HTML_RE + r'(?!\>)'


class SubstituteTextPattern(HtmlInlineProcessor):
    def __init__(self, pattern: str, replace: Sequence[int | str | etree.Element], md: Markdown):
        """ Replaces matches with some text. """
        HtmlInlineProcessor.__init__(self, pattern)
        self.replace = replace
        self.md = md

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[str, int, int]:
        result = ''
        for part in self.replace:
            if isinstance(part, int):
                result += m.group(part)
            else:
                result += self.md.htmlStash.store(part)
        return result, m.start(0), m.end(0)


def makeExtension(**kwargs):  # pragma: no cover
    return SmartyExtension(**kwargs)

```