# `markdown\markdown\extensions\fenced_code.py`

```

# Fenced Code Extension for Python Markdown
# =========================================

# This extension adds Fenced Code Blocks to Python-Markdown.

# See https://Python-Markdown.github.io/extensions/fenced_code_blocks
# for documentation.

# Original code Copyright 2007-2008 [Waylan Limberg](http://achinghead.com/).

# All changes Copyright 2008-2014 The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

"""
This extension adds Fenced Code Blocks to Python-Markdown.

See the [documentation](https://Python-Markdown.github.io/extensions/fenced_code_blocks)
for details.
"""

from __future__ import annotations

from textwrap import dedent
from . import Extension
from ..preprocessors import Preprocessor
from .codehilite import CodeHilite, CodeHiliteExtension, parse_hl_lines
from .attr_list import get_attrs, AttrListExtension
from ..util import parseBoolValue
from ..serializers import _escape_attrib_html
import re
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:  # pragma: no cover
    from markdown import Markdown


class FencedCodeExtension(Extension):
    def __init__(self, **kwargs):
        self.config = {
            'lang_prefix': ['language-', 'Prefix prepended to the language. Default: "language-"']
        }
        """ Default configuration options. """
        super().__init__(**kwargs)

    def extendMarkdown(self, md):
        """ Add `FencedBlockPreprocessor` to the Markdown instance. """
        md.registerExtension(self)

        md.preprocessors.register(FencedBlockPreprocessor(md, self.getConfigs()), 'fenced_code_block', 25)


def makeExtension(**kwargs):  # pragma: no cover
    return FencedCodeExtension(**kwargs)

In this code, we have a Python Markdown extension for adding Fenced Code Blocks. The comments provide information about the purpose of the extension, its documentation, original copyright, and license. The code itself defines a class `FencedCodeExtension` that extends the `Extension` class and includes methods for initializing the extension and extending the Markdown instance. The `makeExtension` function is also defined to create an instance of the `FencedCodeExtension` class. Overall, the code is well-documented and organized.
```