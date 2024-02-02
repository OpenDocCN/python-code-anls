title: Release Notes for v2.6

# Python-Markdown 2.6 Release Notes

We are pleased to release Python-Markdown 2.6 which adds a few new features
and fixes various bugs. See the list of changes below for details.

Python-Markdown version 2.6 supports Python versions 2.7, 3.2, 3.3, and 3.4 as
well as PyPy.

## Backwards-incompatible Changes

### `safe_mode` Deprecated

Both `safe_mode` and the associated `html_replacement_text` keywords are
deprecated in version 2.6 and will raise a **`DeprecationWarning`**. The
`safe_mode` and `html_replacement_text` keywords will be ignored in the next
release. The so-called "safe mode" was never actually "safe" which has resulted
in many people having a false sense of security when using it. As an
alternative, the developers of Python-Markdown recommend that any untrusted
content be passed through an HTML sanitizer (like [Bleach]) after being
converted to HTML by markdown. In fact, [Bleach Whitelist] provides a curated
list of tags, attributes, and styles suitable for filtering user-provided HTML
using bleach.

If your code previously looked like this:

```py
html = markdown.markdown(text, safe_mode=True)
```

Then it is recommended that you change your code to read something like this:

```py
import bleach
from bleach_whitelist import markdown_tags, markdown_attrs
html = bleach.clean(markdown.markdown(text), markdown_tags, markdown_attrs)
```

If you are not interested in sanitizing untrusted text, but simply desire to
escape raw HTML, then that can be accomplished through an extension which
removes HTML parsing:

```py
from markdown.extensions import Extension

class EscapeHtml(Extension):
    def extendMarkdown(self, md, md_globals):
        del md.preprocessors['html_block']
        del md.inlinePatterns['html']

html = markdown.markdown(text, extensions=[EscapeHtml()])
```

As the HTML would not be parsed with the above Extension, then the serializer
will escape the raw HTML, which is exactly what happens now when
`safe_mode="escape"`.

[Bleach]: https://bleach.readthedocs.io/
[Bleach Whitelist]: https://github.com/yourcelf/bleach-whitelist

### Positional Arguments Deprecated

Positional arguments on the `markdown.Markdown()` class are deprecated as are
all except the `text` argument on the `markdown.markdown()` wrapper function.
Using positional arguments will raise a **`DeprecationWarning`** in 2.6 and an
error in the next release. Only keyword arguments should be used. For example,
if your code previously looked like this:

```py
html = markdown.markdown(text, [SomeExtension()])
```

Then it is recommended that you change it to read something like this:

```py
html = markdown.markdown(text, extensions=[SomeExtension()])
```

!!! Note
    This change is being made as a result of deprecating `"safe_mode"` as the
    `safe_mode` argument was one of the positional arguments. When that argument
    is removed, the two arguments following it will no longer be at the correct
    position. It is recommended that you always use keywords when they are
    supported for this reason.

### "Shortened" Extension Names Deprecated

In previous versions of Python-Markdown, the built-in extensions received
special status and did not require the full path to be provided. Additionally,
third party extensions whose name started with `"mdx_"` received the same
special treatment. This behavior is deprecated and will raise a
**`DeprecationWarning`** in version 2.6 and an error in the next release. Ensure
that you always use the full path to your extensions. For example, if you
previously did the following:

```py
markdown.markdown(text, extensions=['extra'])
```

You should change your code to the following:

```py
markdown.markdown(text, extensions=['markdown.extensions.extra'])
```

The same applies to the command line:

```py
python -m markdown -x markdown.extensions.extra input.txt
```

Similarly, if you have used a third party extension (for example `mdx_math`),
previously you might have called it like this:

```py
markdown.markdown(text, extensions=['math'])
```

As the `"mdx"` prefix will no longer be appended, you will need to change your
code as follows (assuming the file `mdx_math.py` is installed at the root of
your PYTHONPATH):

```py
markdown.markdown(text, extensions=['mdx_math'])
```

Extension authors will want to update their documentation to reflect the new
behavior.

See the [documentation](../reference.md#extensions) for a full explanation
of the current behavior.

### Extension Configuration as Part of Extension Name Deprecated

The previously documented method of appending the extension configuration
options as a string to the extension name is deprecated and will raise a
**`DeprecationWarning`** in version 2.6 and an error in 2.7. The
[`extension_configs`](../reference.md#extension_configs) keyword should be used
instead. See the [documentation](../reference.md#extension-configs) for a full
explanation of the current behavior.

### HeaderId Extension Pending Deprecation

The HeaderId Extension is pending deprecation and will raise a
**`PendingDeprecationWarning`** in version 2.6. The extension will be deprecated
in the next release and raise an error in the release after that. Use the [Table
of Contents][TOC] Extension instead, which offers most of the features of the
HeaderId Extension and more (support for meta data is missing).

Extension authors who have been using the `slugify` and `unique` functions
defined in the HeaderId Extension should note that those functions are now
defined in the Table of Contents extension and should adjust their import
statements accordingly (`from markdown.extensions.toc import slugify, unique`).

### The `configs` Keyword is Deprecated

Positional arguments and the `configs` keyword on the
`markdown.extension.Extension` class (and its subclasses) are deprecated. Each
individual configuration option should be passed to the class as a keyword/value
pair. For example. one might have previously initiated an extension subclass
like this:

```py
ext = SomeExtension(configs={'somekey': 'somevalue'})
```

That code should be updated to pass in the options directly:

```py
ext = SomeExtension(somekey='somevalue')
```

Extension authors will want to note that this affects the `makeExtension`
function as well. Previously it was common for the function to be defined as
follows:

```py
def makeExtension(configs=None):
    return SomeExtension(configs=configs)
```

Extension authors will want to update their code to the following instead:

```py
def makeExtension(**kwargs):
    return SomeExtension(**kwargs)
```

Failing to do so will result in a **`DeprecationWarning`** and will raise an
error in the next release. See the [Extension API][mext] documentation for more
information.

In the event that an `markdown.extension.Extension` subclass overrides the
`__init__` method and implements its own configuration handling, then the above
may not apply. However, it is recommended that the subclass still calls the
parent `__init__` method to handle configuration options like so:

```py
class SomeExtension(markdown.extension.Extension):
    def __init__(**kwargs):
        # Do pre-config stuff here
        # Set config defaults
        self.config = {
            'option1' : ['value1', 'description1'],
            'option2' : ['value2', 'description2']
        }
        # Set user defined configs
        super(MyExtension, self).__init__(**kwargs)
        # Do post-config stuff here
```

Note the call to `super` to get the benefits of configuration handling from the
parent class. See the [documentation][config] for more information.

[config]: ../extensions/api.md#configsettings
[mext]: ../extensions/api.md#makeextension

## What's New in Python-Markdown 2.6

### Official Support for PyPy

Official support for [PyPy] has been added. While Python-Markdown has most
likely worked on PyPy for some time, it is now officially supported and tested
on PyPy.

[PyPy]: https://pypy.org/

### YAML Style Meta-Data

<del>The [Meta-Data] Extension now includes optional support for [YAML] style
meta-data.</del> By default, the YAML deliminators are recognized, however, the
actual data is parsed as previously. This follows the syntax of [MultiMarkdown],
which inspired this extension.

<del>Alternatively, if the `yaml` option is set, then the data is parsed as
YAML.</del> <ins>As the `yaml` option was buggy, it was removed in 2.6.1. It is
suggested that a preprocessor (like [docdata]) or a third party extension be
used if you want true YAML support. See [Issue #390][#390] for a full
explanation.</ins>

[MultiMarkdown]: https://fletcherpenney.net/multimarkdown/#metadata
[Meta-Data]: ../extensions/meta_data.md
[YAML]: https://yaml.org/
[#390]: https://github.com/Python-Markdown/markdown/issues/390
[docdata]: https://github.com/waylan/docdata

### Table of Contents Extension Refactored

The [Table of Contents][TOC] Extension has been refactored and some new features
have been added. See the documentation for a full explanation of each feature
listed below:

* The extension now assigns the Table of Contents to the `toc` attribute of
  the Markdown class regardless of whether a "marker" was found in the
  document. Third party frameworks no longer need to insert a "marker," run
  the document through Markdown, then extract the Table of Contents from the
  document.

* The Table of Contents Extension is now a "registered extension." Therefore,
  when the `reset` method of the Markdown class is called, the `toc` attribute
  on the Markdown class is cleared (set to an empty string).

* When the `marker` configuration option is set to an empty string, the parser
  completely skips the process of searching the document for markers. This
  should save parsing time when the Table of Contents Extension is being used
  only to assign ids to headers.

* A `separator` configuration option has been added allowing users to override
  the separator character used by the slugify function.

* A `baselevel` configuration option has been added allowing users to set the
  base level of headers in their documents (h1-h6). This allows the header
  levels to be automatically adjusted to fit within the hierarchy of an HTML
  template.

[TOC]: ../extensions/toc.md

### Pygments can now be disabled

The [CodeHilite][ch] Extension has gained a new configuration option:
`use_pygments`. The option is `True` by default, however, it allows one to turn
off Pygments code highlighting (set to `False`) while preserving the language
detection features of the extension. Note that Pygments language guessing is not
used as that would 'use Pygments'. If a language is defined for a code block, it
will be assigned to the `<code>` tag as a class in the manner suggested by the
[HTML5 spec][spec] (alternate output will not be entertained) and could
potentially be used by a JavaScript library in the browser to highlight the code
block.

[ch]: ../extensions/code_hilite.md
[spec]: https://www.w3.org/TR/html5/text-level-semantics.html#the-code-element

### Miscellaneous

Test coverage has been improved including running [flake8]. While those changes
will not directly effect end users, the code is being better tested which will
benefit everyone.

[flake8]: https://flake8.readthedocs.io/en/latest/

Various bug fixes have been made.  See the
[commit log](https://github.com/Python-Markdown/markdown/commits/master)
for a complete history of the changes.
