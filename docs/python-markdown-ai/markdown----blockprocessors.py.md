# `markdown\markdown\blockprocessors.py`

```py

# Python Markdown

# A Python implementation of John Gruber's Markdown.

# Documentation: https://python-markdown.github.io/
# GitHub: https://github.com/Python-Markdown/markdown/
# PyPI: https://pypi.org/project/Markdown/

# Started by Manfred Stienstra (http://www.dwerg.net/).
# Maintained for a few years by Yuri Takhteyev (http://www.freewisdom.org).
# Currently maintained by Waylan Limberg (https://github.com/waylan),
# Dmitry Shachnev (https://github.com/mitya57) and Isaac Muse (https://github.com/facelessuser).

# Copyright 2007-2023 The Python Markdown Project (v. 1.7 and later)
# Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
# Copyright 2004 Manfred Stienstra (the original version)

# License: BSD (see LICENSE.md for details).

"""
A block processor parses blocks of text and adds new elements to the ElementTree. Blocks of text,
separated from other text by blank lines, may have a different syntax and produce a differently
structured tree than other Markdown. Block processors excel at handling code formatting, equation
layouts, tables, etc.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any
from . import util
from .blockparser import BlockParser

if TYPE_CHECKING:  # pragma: no cover
    from markdown import Markdown

logger = logging.getLogger('MARKDOWN')


def build_block_parser(md: Markdown, **kwargs: Any) -> BlockParser:
    """ Build the default block parser used by Markdown. """
    parser = BlockParser(md)
    parser.blockprocessors.register(EmptyBlockProcessor(parser), 'empty', 100)
    parser.blockprocessors.register(ListIndentProcessor(parser), 'indent', 90)
    parser.blockprocessors.register(CodeBlockProcessor(parser), 'code', 80)
    parser.blockprocessors.register(HashHeaderProcessor(parser), 'hashheader', 70)
    parser.blockprocessors.register(SetextHeaderProcessor(parser), 'setextheader', 60)
    parser.blockprocessors.register(HRProcessor(parser), 'hr', 50)
    parser.blockprocessors.register(OListProcessor(parser), 'olist', 40)
    parser.blockprocessors.register(UListProcessor(parser), 'ulist', 30)
    parser.blockprocessors.register(BlockQuoteProcessor(parser), 'quote', 20)
    parser.blockprocessors.register(ReferenceProcessor(parser), 'reference', 15)
    parser.blockprocessors.register(ParagraphProcessor(parser), 'paragraph', 10)
    return parser


# The following classes are block processors that handle different types of Markdown blocks.

class CodeBlockProcessor(BlockProcessor):
    """ Process code blocks. """

    def test(self, parent: etree.Element, block: str) -> bool:
        return block.startswith(' '*self.tab_length)

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        sibling = self.lastChild(parent)
        block = blocks.pop(0)
        theRest = ''
        if (sibling is not None and sibling.tag == "pre" and
           len(sibling) and sibling[0].tag == "code"):
            # The previous block was a code block. As blank lines do not start
            # new code blocks, append this block to the previous, adding back
            # line breaks removed from the split into a list.
            code = sibling[0]
            block, theRest = self.detab(block)
            code.text = util.AtomicString(
                '{}\n{}\n'.format(code.text, util.code_escape(block.rstrip()))
            )
        else:
            # This is a new code block. Create the elements and insert text.
            pre = etree.SubElement(parent, 'pre')
            code = etree.SubElement(pre, 'code')
            block, theRest = self.detab(block)
            code.text = util.AtomicString('%s\n' % util.code_escape(block.rstrip()))
        if theRest:
            # This block contained unindented line(s) after the first indented
            # line. Insert these lines as the first block of the master blocks
            # list for future processing.
            blocks.insert(0, theRest)


# The other block processor classes (BlockQuoteProcessor, UListProcessor, HashHeaderProcessor, SetextHeaderProcessor,
# HRProcessor, EmptyBlockProcessor, ReferenceProcessor, and ParagraphProcessor) follow a similar structure, with a
# test method to determine if the block should be processed, and a run method to actually process the block.

# The build_block_parser function creates a default block parser used by Markdown, and registers the different block
# processors to handle various types of Markdown blocks.

# The entire code is a Python implementation of John Gruber's Markdown, and it provides a way to parse blocks of text
# and add new elements to the ElementTree. It handles different types of Markdown blocks such as code blocks, blockquotes,
# headers, horizontal rules, empty blocks, reference blocks, and paragraphs.

```