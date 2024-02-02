# `markdown\markdown\extensions\toc.py`

```py

# Table of Contents Extension for Python-Markdown
# ===============================================

# See https://Python-Markdown.github.io/extensions/toc
# for documentation.

# Original code Copyright 2008 [Jack Miller](https://codezen.org/)

# All changes Copyright 2008-2014 The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

"""
Add table of contents support to Python-Markdown.

See the [documentation](https://Python-Markdown.github.io/extensions/toc)
for details.
"""

from __future__ import annotations

from . import Extension  # 导入扩展模块
from ..treeprocessors import Treeprocessor  # 导入树处理器模块
from ..util import code_escape, parseBoolValue, AMP_SUBSTITUTE, HTML_PLACEHOLDER_RE, AtomicString  # 导入工具模块
from ..treeprocessors import UnescapeTreeprocessor  # 导入树处理器模块
import re  # 导入正则表达式模块
import html  # 导入 HTML 模块
import unicodedata  # 导入 Unicode 数据模块
import xml.etree.ElementTree as etree  # 导入 XML 树模块
from typing import TYPE_CHECKING, Any, Iterator, MutableSet  # 导入类型检查模块

if TYPE_CHECKING:  # pragma: no cover
    from markdown import Markdown  # 如果是类型检查，导入 Markdown 模块


def slugify(value: str, separator: str, unicode: bool = False) -> str:
    """ Slugify a string, to make it URL friendly. """
    if not unicode:
        # Replace Extended Latin characters with ASCII, i.e. `žlutý` => `zluty`
        value = unicodedata.normalize('NFKD', value)
        value = value.encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[{}\s]+'.format(separator), separator, value)


def slugify_unicode(value: str, separator: str) -> str:
    """ Slugify a string, to make it URL friendly while preserving Unicode characters. """
    return slugify(value, separator, unicode=True)


IDCOUNT_RE = re.compile(r'^(.*)_([0-9]+)$')


def unique(id: str, ids: MutableSet[str]) -> str:
    """ Ensure id is unique in set of ids. Append '_1', '_2'... if not """
    while id in ids or not id:
        m = IDCOUNT_RE.match(id)
        if m:
            id = '%s_%d' % (m.group(1), int(m.group(2))+1)
        else:
            id = '%s_%d' % (id, 1)
    ids.add(id)
    return id


def get_name(el: etree.Element) -> str:
    """Get title name."""

    text = []
    for c in el.itertext():
        if isinstance(c, AtomicString):
            text.append(html.unescape(c))
        else:
            text.append(c)
    return ''.join(text).strip()


def stashedHTML2text(text: str, md: Markdown, strip_entities: bool = True) -> str:
    """ Extract raw HTML from stash, reduce to plain text and swap with placeholder. """
    def _html_sub(m: re.Match[str]) -> str:
        """ Substitute raw html with plain text. """
        try:
            raw = md.htmlStash.rawHtmlBlocks[int(m.group(1))]
        except (IndexError, TypeError):  # pragma: no cover
            return m.group(0)
        # Strip out tags and/or entities - leaving text
        res = re.sub(r'(<[^>]+>)', '', raw)
        if strip_entities:
            res = re.sub(r'(&[\#a-zA-Z0-9]+;)', '', res)
        return res

    return HTML_PLACEHOLDER_RE.sub(_html_sub, text)


def unescape(text: str) -> str:
    """ Unescape escaped text. """
    c = UnescapeTreeprocessor()
    return c.unescape(text)


def nest_toc_tokens(toc_list):
    """Given an unsorted list with errors and skips, return a nested one.

        [{'level': 1}, {'level': 2}]
        =>
        [{'level': 1, 'children': [{'level': 2, 'children': []}]}]

    A wrong list is also converted:

        [{'level': 2}, {'level': 1}]
        =>
        [{'level': 2, 'children': []}, {'level': 1, 'children': []}]
    """

    ordered_list = []
    if len(toc_list):
        # Initialize everything by processing the first entry
        last = toc_list.pop(0)
        last['children'] = []
        levels = [last['level']]
        ordered_list.append(last)
        parents = []

        # Walk the rest nesting the entries properly
        while toc_list:
            t = toc_list.pop(0)
            current_level = t['level']
            t['children'] = []

            # Reduce depth if current level < last item's level
            if current_level < levels[-1]:
                # Pop last level since we know we are less than it
                levels.pop()

                # Pop parents and levels we are less than or equal to
                to_pop = 0
                for p in reversed(parents):
                    if current_level <= p['level']:
                        to_pop += 1
                    else:  # pragma: no cover
                        break
                if to_pop:
                    levels = levels[:-to_pop]
                    parents = parents[:-to_pop]

                # Note current level as last
                levels.append(current_level)

            # Level is the same, so append to
            # the current parent (if available)
            if current_level == levels[-1]:
                (parents[-1]['children'] if parents
                 else ordered_list).append(t)

            # Current level is > last item's level,
            # So make last item a parent and append current as child
            else:
                last['children'].append(t)
                parents.append(last)
                levels.append(current_level)
            last = t

    return ordered_list


class TocExtension(Extension):

    TreeProcessorClass = TocTreeprocessor

    def __init__(self, **kwargs):
        self.config = {
            'marker': [
                '[TOC]',
                'Text to find and replace with Table of Contents. Set to an empty string to disable. '
                'Default: `[TOC]`.'
            ],
            'title': [
                '', 'Title to insert into TOC `<div>`. Default: an empty string.'
            ],
            'title_class': [
                'toctitle', 'CSS class used for the title. Default: `toctitle`.'
            ],
            'toc_class': [
                'toc', 'CSS class(es) used for the link. Default: `toclink`.'
            ],
            'anchorlink': [
                False, 'True if header should be a self link. Default: `False`.'
            ],
            'anchorlink_class': [
                'toclink', 'CSS class(es) used for the link. Defaults: `toclink`.'
            ],
            'permalink': [
                0, 'True or link text if a Sphinx-style permalink should be added. Default: `False`.'
            ],
            'permalink_class': [
                'headerlink', 'CSS class(es) used for the link. Default: `headerlink`.'
            ],
            'permalink_title': [
                'Permanent link', 'Title attribute of the permalink. Default: `Permanent link`.'
            ],
            'permalink_leading': [
                False,
                'True if permalinks should be placed at start of the header, rather than end. Default: False.'
            ],
            'baselevel': ['1', 'Base level for headers. Default: `1`.'],
            'slugify': [
                slugify, 'Function to generate anchors based on header text. Default: `slugify`.'
            ],
            'separator': ['-', 'Word separator. Default: `-`.'],
            'toc_depth': [
                6,
                'Define the range of section levels to include in the Table of Contents. A single integer '
                '(b) defines the bottom section level (<h1>..<hb>) only. A string consisting of two digits '
                'separated by a hyphen in between (`2-5`) defines the top (t) and the bottom (b) (<ht>..<hb>). '
                'Default: `6` (bottom).'
            ],
        }
        """ Default configuration options. """

        super().__init__(**kwargs)

    def extendMarkdown(self, md):
        """ Add TOC tree processor to Markdown. """
        md.registerExtension(self)
        self.md = md
        self.reset()
        tocext = self.TreeProcessorClass(md, self.getConfigs())
        md.treeprocessors.register(tocext, 'toc', 5)

    def reset(self) -> None:
        self.md.toc = ''
        self.md.toc_tokens = []


def makeExtension(**kwargs):  # pragma: no cover
    return TocExtension(**kwargs)

```