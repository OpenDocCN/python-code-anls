# `markdown\markdown\util.py`

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
This module contains various contacts, classes and functions which get referenced and used
throughout the code base.
"""

from __future__ import annotations  # Importing annotations from the future

import re  # Importing regular expression module
import sys  # Importing system-specific parameters and functions
import warnings  # Importing warning functions
from functools import wraps, lru_cache  # Importing functions for higher-order functions and caching
from itertools import count  # Importing count function for creating an iterator
from typing import TYPE_CHECKING, Generic, Iterator, NamedTuple, TypeVar, TypedDict, overload  # Importing type hints for static type checking

if TYPE_CHECKING:  # pragma: no cover
    from markdown import Markdown  # Importing Markdown class for type hinting
    import xml.etree.ElementTree as etree  # Importing ElementTree module for XML support

_T = TypeVar('_T')  # Creating a type variable

"""
Constants you might want to modify
-----------------------------------------------------------------------------
"""

BLOCK_LEVEL_ELEMENTS: list[str] = [  # List of block-level HTML tags
    # Elements which are invalid to wrap in a `<p>` tag.
    # See https://w3c.github.io/html/grouping-content.html#the-p-element
    'address', 'article', 'aside', 'blockquote', 'details', 'div', 'dl',
    'fieldset', 'figcaption', 'figure', 'footer', 'form', 'h1', 'h2', 'h3',
    'h4', 'h5', 'h6', 'header', 'hgroup', 'hr', 'main', 'menu', 'nav', 'ol',
    'p', 'pre', 'section', 'table', 'ul',
    # Other elements which Markdown should not be mucking up the contents of.
    'canvas', 'colgroup', 'dd', 'body', 'dt', 'group', 'html', 'iframe', 'li', 'legend',
    'math', 'map', 'noscript', 'output', 'object', 'option', 'progress', 'script',
    'style', 'summary', 'tbody', 'td', 'textarea', 'tfoot', 'th', 'thead', 'tr', 'video'
]
"""
List of HTML tags which get treated as block-level elements. Same as the `block_level_elements`
attribute of the [`Markdown`][markdown.Markdown] class. Generally one should use the
attribute on the class. This remains for compatibility with older extensions.
"""

# Placeholders
STX = '\u0002'  # "Start of Text" marker for placeholder templates
ETX = '\u0003'  # "End of Text" marker for placeholder templates
INLINE_PLACEHOLDER_PREFIX = STX+"klzzwxh:"  # Prefix for inline placeholder template
INLINE_PLACEHOLDER = INLINE_PLACEHOLDER_PREFIX + "%s" + ETX  # Placeholder template for stashed inline text
INLINE_PLACEHOLDER_RE = re.compile(INLINE_PLACEHOLDER % r'([0-9]+)')  # Regular Expression which matches inline placeholders
AMP_SUBSTITUTE = STX+"amp"+ETX  # Placeholder template for HTML entities
HTML_PLACEHOLDER = STX + "wzxhzdk:%s" + ETX  # Placeholder template for raw HTML
HTML_PLACEHOLDER_RE = re.compile(HTML_PLACEHOLDER % r'([0-9]+)')  # Regular expression which matches HTML placeholders
TAG_PLACEHOLDER = STX + "hzzhzkh:%s" + ETX  # Placeholder template for tags

# Constants you probably do not need to change
# -----------------------------------------------------------------------------

RTL_BIDI_RANGES = (  # Right-to-left bidi ranges
    ('\u0590', '\u07FF'),  # Hebrew, Arabic, Syriac, Arabic supplement, Thaana, Nko
    ('\u2D30', '\u2D7F')  # Tifinagh
)


# AUXILIARY GLOBAL FUNCTIONS
# =============================================================================

@lru_cache(maxsize=None)
def get_installed_extensions():
    """ Return all entry_points in the `markdown.extensions` group. """
    if sys.version_info >= (3, 10):  # Checking Python version
        from importlib import metadata  # Importing metadata from importlib
    else:  # `<PY310` use backport
        import importlib_metadata as metadata  # Importing metadata from importlib_metadata
    # Only load extension entry_points once.
    return metadata.entry_points(group='markdown.extensions')  # Returning entry points for markdown extensions


def deprecated(message: str, stacklevel: int = 2):
    """
    Raise a [`DeprecationWarning`][] when wrapped function/method is called.

    Usage:

    ```python
    @deprecated("This method will be removed in version X; use Y instead.")
    def some_method():
        pass
    ```py
    """
    def wrapper(func):
        @wraps(func)
        def deprecated_func(*args, **kwargs):
            warnings.warn(  # Warning about deprecation
                f"'{func.__name}' is deprecated. {message}",
                category=DeprecationWarning,
                stacklevel=stacklevel
            )
            return func(*args, **kwargs)
        return deprecated_func
    return wrapper


def parseBoolValue(value: str | None, fail_on_errors: bool = True, preserve_none: bool = False) -> bool | None:
    """Parses a string representing a boolean value. If parsing was successful,
       returns `True` or `False`. If `preserve_none=True`, returns `True`, `False`,
       or `None`. If parsing was not successful, raises `ValueError`, or, if
       `fail_on_errors=False`, returns `None`."""
    if not isinstance(value, str):  # Checking if value is not a string
        if preserve_none and value is None:  # Checking if preserve_none is True and value is None
            return value
        return bool(value)  # Converting value to boolean
    elif preserve_none and value.lower() == 'none':  # Checking if preserve_none is True and value is 'none'
        return None
    elif value.lower() in ('true', 'yes', 'y', 'on', '1'):  # Checking if value is a truthy string
        return True
    elif value.lower() in ('false', 'no', 'n', 'off', '0', 'none'):  # Checking if value is a falsy string
        return False
    elif fail_on_errors:  # Checking if fail_on_errors is True
        raise ValueError('Cannot parse bool value: %r' % value)  # Raising ValueError if parsing was not successful


def code_escape(text: str) -> str:
    """HTML escape a string of code."""
    if "&" in text:  # Checking if '&' is in the text
        text = text.replace("&", "&amp;")  # Replacing '&' with HTML entity
    if "<" in text:  # Checking if '<' is in the text
        text = text.replace("<", "&lt;")  # Replacing '<' with HTML entity
    if ">" in text:  # Checking if '>' is in the text
        text = text.replace(">", "&gt;")  # Replacing '>' with HTML entity
    return text  # Returning the escaped text


def _get_stack_depth(size: int = 2) -> int:
    """Get current stack depth, performantly.
    """
    frame = sys._getframe(size)  # Getting the current frame

    for size in count(size):  # Looping through the count
        frame = frame.f_back  # Moving to the previous frame
        if not frame:  # Checking if frame is None
            return size  # Returning the stack depth


def nearing_recursion_limit() -> bool:
    """Return true if current stack depth is within 100 of maximum limit."""
    return sys.getrecursionlimit() - _get_stack_depth() < 100  # Checking if current stack depth is within 100 of maximum limit


# MISC AUXILIARY CLASSES
# =============================================================================

class AtomicString(str):
    """A string which should not be further processed."""
    pass


class Processor:
    """ The base class for all processors.

    Attributes:
        Processor.md: The `Markdown` instance passed in an initialization.

    Arguments:
        md: The `Markdown` instance this processor is a part of.

    """
    def __init__(self, md: Markdown | None = None):
        self.md = md


if TYPE_CHECKING:  # pragma: no cover
    class TagData(TypedDict):
        tag: str
        attrs: dict[str, str]
        left_index: int
        right_index: int


class HtmlStash:
    """
    This class is used for stashing HTML objects that we extract
    in the beginning and replace with place-holders.
    """

    def __init__(self):
        """ Create an `HtmlStash`. """
        self.html_counter = 0  # Counter for inline html segments
        self.rawHtmlBlocks: list[str | etree.Element] = []  # List to store raw HTML blocks
        self.tag_counter = 0  # Counter for tags
        self.tag_data: list[TagData] = []  # List to store tag data in the order tags appear

    def store(self, html: str | etree.Element) -> str:
        """
        Saves an HTML segment for later reinsertion.  Returns a
        placeholder string that needs to be inserted into the
        document.

        Keyword arguments:
            html: An html segment.

        Returns:
            A placeholder string.

        """
        self.rawHtmlBlocks.append(html)  # Appending the HTML segment to the list
        placeholder = self.get_placeholder(self.html_counter)  # Getting the placeholder for the HTML segment
        self.html_counter += 1  # Incrementing the counter
        return placeholder  # Returning the placeholder

    def reset(self) -> None:
        """ Clear the stash. """
        self.html_counter = 0  # Resetting the counter
        self.rawHtmlBlocks = []  # Clearing the list

    def get_placeholder(self, key: int) -> str:
        return HTML_PLACEHOLDER % key  # Returning the placeholder for the given key

    def store_tag(self, tag: str, attrs: dict[str, str], left_index: int, right_index: int) -> str:
        """Store tag data and return a placeholder."""
        self.tag_data.append({'tag': tag, 'attrs': attrs,
                              'left_index': left_index,
                              'right_index': right_index})  # Appending tag data to the list
        placeholder = TAG_PLACEHOLDER % str(self.tag_counter)  # Getting the placeholder for the tag
        self.tag_counter += 1  # Incrementing the counter
        return placeholder  # Returning the placeholder


# Used internally by `Registry` for each item in its sorted list.
# Provides an easier to read API when editing the code later.
# For example, `item.name` is more clear than `item[0]`.
class _PriorityItem(NamedTuple):
    name: str
    priority: float

```