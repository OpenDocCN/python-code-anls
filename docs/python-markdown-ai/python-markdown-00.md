# PythonMarkdown源码解析 0

# Contributor Code of Conduct

As contributors and maintainers of this project, and in the interest of
fostering an open and welcoming community, we pledge to respect all people who
contribute through reporting issues, posting feature requests, updating
documentation, submitting pull requests or patches, and other activities.

We are committed to making participation in this project a harassment-free
experience for everyone, regardless of level of experience, gender, gender
identity and expression, sexual orientation, disability, personal appearance,
body size, race, ethnicity, age, religion, or nationality.

Examples of unacceptable behavior by participants include:

* The use of sexualized language or imagery
* Personal attacks
* Trolling or insulting/derogatory comments
* Public or private harassment
* Publishing other's private information, such as physical or electronic
  addresses, without explicit permission
* Other unethical or unprofessional conduct

Project maintainers have the right and responsibility to remove, edit, or
reject comments, commits, code, wiki edits, issues, and other contributions
that are not aligned to this Code of Conduct, or to ban temporarily or
permanently any contributor for other behaviors that they deem inappropriate,
threatening, offensive, or harmful.

By adopting this Code of Conduct, project maintainers commit themselves to
fairly and consistently applying these principles to every aspect of managing
this project. Project maintainers who do not follow or enforce the Code of
Conduct may be permanently removed from the project team.

This Code of Conduct applies both within project spaces and in public spaces
when an individual is representing the project or its community.

Instances of abusive, harassing, or otherwise unacceptable behavior may be
reported by contacting a project maintainer at python.markdown@gmail.com. All
complaints will be reviewed and investigated and will result in a response that
is deemed necessary and appropriate to the circumstances. Maintainers are
obligated to maintain confidentiality with regard to the reporter of an
incident.

This Code of Conduct is adapted from the [Contributor Covenant][homepage],
version 1.3.0, available at
[https://www.contributor-covenant.org/version/1/3/0/][version]

[homepage]: https://www.contributor-covenant.org/
[version]: https://www.contributor-covenant.org/version/1/3/0/


Installing Python-Markdown
==========================

As an Admin/Root user on your system do:

    pip install markdown

Or for more specific instructions, view the documentation in `docs/install.md`
or on the website at <https://Python-Markdown.github.io/install/>.


Copyright 2007, 2008 The Python Markdown Project (v. 1.7 and later)
Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
Copyright 2004 Manfred Stienstra (the original version)

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of the Python Markdown Project nor the
  names of its contributors may be used to endorse or promote products
  derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE PYTHON MARKDOWN PROJECT ''AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL ANY CONTRIBUTORS TO THE PYTHON MARKDOWN PROJECT
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.


[Python-Markdown][]
===================

[![Build Status][build-button]][build]
[![Coverage Status][codecov-button]][codecov]
[![Latest Version][mdversion-button]][md-pypi]
[![Python Versions][pyversion-button]][md-pypi]
[![BSD License][bsdlicense-button]][bsdlicense]
[![Code of Conduct][codeofconduct-button]][Code of Conduct]

[build-button]: https://github.com/Python-Markdown/markdown/workflows/CI/badge.svg?event=push
[build]: https://github.com/Python-Markdown/markdown/actions?query=workflow%3ACI+event%3Apush
[codecov-button]: https://codecov.io/gh/Python-Markdown/markdown/branch/master/graph/badge.svg
[codecov]: https://codecov.io/gh/Python-Markdown/markdown
[mdversion-button]: https://img.shields.io/pypi/v/Markdown.svg
[md-pypi]: https://pypi.org/project/Markdown/
[pyversion-button]: https://img.shields.io/pypi/pyversions/Markdown.svg
[bsdlicense-button]: https://img.shields.io/badge/license-BSD-yellow.svg
[bsdlicense]: https://opensource.org/licenses/BSD-3-Clause
[codeofconduct-button]: https://img.shields.io/badge/code%20of%20conduct-contributor%20covenant-green.svg?style=flat-square
[Code of Conduct]: https://github.com/Python-Markdown/markdown/blob/master/CODE_OF_CONDUCT.md

This is a Python implementation of John Gruber's [Markdown][].
It is almost completely compliant with the reference implementation,
though there are a few known issues. See [Features][] for information
on what exactly is supported and what is not. Additional features are
supported by the [Available Extensions][].

[Python-Markdown]: https://Python-Markdown.github.io/
[Markdown]: https://daringfireball.net/projects/markdown/
[Features]: https://Python-Markdown.github.io#Features
[Available Extensions]: https://Python-Markdown.github.io/extensions

Documentation
-------------

```pybash
pip install markdown
```
```pypython
import markdown
html = markdown.markdown(your_text_string)
```

For more advanced [installation] and [usage] documentation, see the `docs/` directory
of the distribution or the project website at <https://Python-Markdown.github.io/>.

[installation]: https://python-markdown.github.io/install/
[usage]: https://python-markdown.github.io/reference/

See the change log at <https://Python-Markdown.github.io/change_log>.

Support
-------

You may report bugs, ask for help, and discuss various other issues on the [bug tracker][].

[bug tracker]: https://github.com/Python-Markdown/markdown/issues

Code of Conduct
---------------

Everyone interacting in the Python-Markdown project's code bases, issue trackers,
and mailing lists is expected to follow the [Code of Conduct].


title: Authors

Primary Authors
===============

* __[Waylan Limberg](https://github.com/waylan)__

    @waylan is the current maintainer of the code and has written much of the
    current code base, including a complete refactor of the core for version 2.0.
    He started out by authoring many of the available extensions and later was
    asked to join Yuri, where he began fixing numerous bugs, adding
    documentation and making general improvements to the existing code base.

* __[Dmitry Shachnev](https://github.com/mitya57)__

    @mitya57 joined the team after providing a number of helpful patches and has
    been assisting with maintenance, reviewing pull requests and ticket triage.

* __[Isaac Muse](https://github.com/facelessuser)__

    @facelessuser joined the team after providing a number of helpful patches
    and has been assisting with maintenance, reviewing pull requests and ticket
    triage.

* __[Yuri Takhteyev](https://github.com/yuri)__

    Yuri wrote most of the code found in version 1.x while procrastinating his
    Ph.D. Various pieces of his code still exist, most notably the basic
    structure.

* __Manfed Stienstra__

    Manfed wrote the original version of the script and is responsible for
    various parts of the existing code base.

* __Artem Yunusov__

    Artem, who as part of a 2008 GSoC project, refactored inline patterns,
    replaced the NanoDOM with ElementTree support and made various other
    improvements.

* __David Wolever__

    David refactored the extension API and made other improvements
    as he helped to integrate Markdown into Dr.Project.

Other Contributors
==================

The incomplete list of individuals below have provided patches or otherwise
contributed to the project prior to the project being hosted on GitHub. See the
GitHub commit log for a list of recent contributors. We would like to thank
everyone who has contributed to the project in any way.

* Eric Abrahamsen
* Jeff Balogh
* Sergej Chodarev
* Chris Clark
* Tiago Cogumbreiro
* Kjell Magne Fauske
* G. Clark Haynes
* Daniel Krech
* Steward Midwinter
* Jack Miller
* Neale Pickett
* Paul Stansifer
* John Szakmeister
* Malcolm Tredinnick
* Ben Wilson
* and many others who helped by reporting bugs


title: Changelog
toc_depth: 2

# Python-Markdown Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [Contributing Guide](contributing.md) for details.

## [unreleased]

### Fixed

* Fix type annotations for `convertFile` - it accepts only bytes-based buffers.
  Also remove legacy checks from Python 2 (#1400)
* Improve and expand type annotations in the code base (#1401).

## [3.5.1] -- 2023-10-31

### Fixed

* Fix a performance problem with HTML extraction where large HTML input could
  trigger quadratic line counting behavior (#1392).
* Improve and expand type annotations in the code base (#1394).

## [3.5] -- 2023-10-06

### Added

#### Add `permalink_leading` configuration option to the toc extension (#1339)

A new boolean option `permalink_leading` controls the position of the permanent
link anchors generated with `permalink`. Setting `permalink_leading` to `True`
will cause the links to be inserted at the start of the header, before any other
header content. The default behavior for `permalink` is to append permanent
links to the header, placing them after all other header content.

### Changed

* Add support for cPython version 3.12 (and PyPy 3.10) and drop support for
  Python version 3.7 (#1357).
* Refactor changelog to use the format defined at <https://keepachangelog.com/>.
* Update the list of empty HTML tags (#1353).
* Add customizable TOC title class to TOC extension (#1293).
* Add API documentation of the code base which is generated by
  [mkdocstrings](https://mkdocstrings.github.io/) (#1220).

### Fixed

* Fix a corner case in admonitions where if an indented code block was provided
  as the first block, the output would be malformed (#1329).

## [3.4.4] -- 2023-07-25

### Fixed

* Add a special case for initial `'s` to smarty extension (#1305).
* Unescape any backslash escaped inline raw HTML (#1358).
* Unescape backslash escaped TOC token names (#1360).

## [3.4.3] -- 2023-03-23

### Fixed

* Restore console script (#1327).

## [3.4.2] -- 2023-03-22

### Fixed
* Officially support Python 3.11.
* Improve standalone * and _ parsing (#1300).
* Consider `<html>` HTML tag a block-level element (#1309).
* Switch from `setup.py` to `pyproject.toml`.

## [3.4.1] -- 2022-07-15

### Fixed

* Fix an import issue with `importlib.util` (#1274).

## [3.4] -- 2022-07-15

### Changed

#### The `tables` extension now uses a `style` attribute instead of an `align` attribute for alignment.

The [HTML4 spec](https://www.w3.org/TR/html4/present/graphics.html#h-15.1.2)
specifically deprecates the use of the `align` attribute and it does not appear
at all in the [HTML5
spec](https://www.w3.org/TR/html53/tabular-data.html#attributes-common-to-td-and-th-elements).
Therefore, by default, the [tables](extensions/tables.md) extension will now use
the `style` attribute (setting just the `text-align` property) in `td` and `th`
blocks.

The former behavior is available by setting the `use_align_attribute`
configuration option to `True` when enabling the extension.

For example, to configure the old `align` behavior:

```pypython
from markdown.extensions.tables import TableExtension

markdown.markdown(src, extensions=[TableExtension(use_align_attribute=True)])
```

#### Backslash unescaping moved to Treeprocessor (#1131).

Unescaping backslash escapes has been moved to a Treeprocessor, which  enables
proper HTML escaping during serialization. However, it is recognized that
various third-party extensions may be calling the old class at
`postprocessors.UnescapePostprocessor`. Therefore, the old class remains in the
code base, but has been deprecated and will be removed in a future release. The
new class `treeprocessors.UnescapeTreeprocessor` should be used instead.

#### Previously deprecated objects have been removed

Various objects were deprecated in version 3.0 and began raising deprecation
warnings (see the [version 3.0 release notes](#30-2018-09-21) for details). Any of those objects
which remained in version 3.3 have been removed from the code base in version 3.4
and will now raise errors. The relevant objects are listed below.


| Deprecated Object                      | Replacement Object                  |
|----------------------------------------|-------------------------------------|
| `markdown.version`                     | `markdown.__version__`              |
| `markdown.version_info`                | `markdown.__version_info__`         |
| `markdown.util.etree`                  | `xml.etree.ElementTree`             |
| `markdown.util.string_type`            | `str`                               |
| `markdown.util.text_type`              | `str`                               |
| `markdown.util.int2str`                | `chr`                               |
| `markdown.util.iterrange`              | `range`                             |
| `markdown.util.isBlockLevel`           | `markdown.Markdown().is_block_level`|
| `markdown.util.Processor().markdown`   | `markdown.util.Processor().md`      |
| `markdown.util.Registry().__setitem__` | `markdown.util.Registry().register` |
| `markdown.util.Registry().__delitem__` |`markdown.util.Registry().deregister`|
| `markdown.util.Registry().add`         | `markdown.util.Registry().register` |

In addition, the `md_globals` parameter of
`Markdown.extensions.Extension.extendMarkdown()` is no longer recognized as a
valid parameter and will raise an error if provided.

### Added

* Some new configuration options have been added to the
  [footnotes](extensions/footnotes.md) extension (#1218):

    * Small refactor of the `BACKLINK_TITLE` option; The use of `format()`
      instead of "old" `%d` formatter allows one to specify text without the
      need to have the number of the footnote in it (like footnotes on
      Wikipedia for example). The modification is backward compatible so no
      configuration change is required.

    * Addition of a new option `SUPERSCRIPT_TEXT` that allows one to specify a
      custom placeholder for the footnote itself in the text.
      Ex: `[{}]` will give `<sup>[1]</sup>`, `({})` will give `<sup>(1)</sup>`,
      or by default, the current behavior: `<sup>1</sup>`.

* The [Table of Contents](extensions/toc.md) extension now accepts a
  `toc_class` parameter which can be used to set the CSS class(es) on the
  `<div>` that contains the Table of Contents (#1224).

* The CodeHilite extension now supports a `pygments_formatter` option that can
  be set to a custom formatter class (#1187).

    - If `pygments_formatter` is set to a string (ex: `'html'`), Pygments'
      default formatter by that name is used.
    - If `pygments_formatter` is set to a formatter class (or any callable
      which returns a formatter instance), then an instance of that class is
      used.

    The formatter class is now passed an additional option, `lang_str`, to
    denote the language of the code block (#1258). While Pygments' built-in
    formatters will ignore the option, a custom formatter assigned to the
    `pygments_formatter` option can make use of the `lang_str` to include the
    code block's language in the output.

### Fixed

* Extension entry-points are only loaded if needed (#1216).
* Added additional checks to the `<pre><code>` handling of
  `PrettifyTreeprocessor` (#1261, #1263).
* Fix XML deprecation warnings.

## [3.3.7] -- 2022-05-05

### Fixed

* Disallow square brackets in reference link ids (#1209).
* Retain configured `pygments_style` after first code block (#1240).
* Ensure fenced code attributes are properly escaped (#1247).

## [3.3.6] -- 2021-11-17

### Fixed

* Fix a dependency issue (#1195, #1196).

## [3.3.5] -- 2021-11-16

### Fixed

* Make the `slugify_unicode` function not remove diacritical marks (#1118).
* Fix `[toc]` detection when used with `nl2br` extension (#1160).
* Re-use compiled regex for block level checks (#1169).
* Don't process shebangs in fenced code blocks when using CodeHilite (#1156).
* Improve email address validation for Automatic Links (#1165).
* Ensure `<summary>` tags are parsed correctly (#1079).
* Support Python 3.10 (#1124).

## [3.3.4] -- 2021-02-24

### Fixed

* Properly parse unclosed tags in code spans (#1066).
* Properly parse processing instructions in md_in_html (#1070).
* Properly parse code spans in md_in_html (#1069).
* Preserve text immediately before an admonition (#1092).
* Simplified regex for HTML placeholders (#928) addressing (#932).
* Ensure `permalinks` and `anchorlinks` are not restricted by `toc_depth` (#1107).
* Fix corner cases with lists under admonitions (#1102).

## [3.3.3] -- 2020-10-25

### Fixed

* Unify all block-level tags (#1047).
* Fix issue where some empty elements would have text rendered as `None` when using `md_in_html` (#1049).
* Avoid catastrophic backtracking in `hr` regex (#1055).
* Fix `hr` HTML handling (#1053).

## [3.3.2] -- 2020-10-19

### Fixed

* Properly parse inline HTML in md_in_html (#1040 & #1045).
* Avoid crashing when md_in_html fails (#1040).

## [3.3.1] -- 2020-10-12

### Fixed

* Correctly parse raw `script` and `style` tags (#1036).
* Ensure consistent class handling by `fenced_code` and `codehilite` (#1032).

## [3.3] -- 2020-10-06

### Changed

#### The prefix `language-` is now prepended to all language classes by default on code blocks.

The [HTML5
spec](https://www.w3.org/TR/html5/text-level-semantics.html#the-code-element)
recommends that the class defining the language of a code block be prefixed with
`language-`. Therefore, by default, both the
[fenced_code](extensions/fenced_code_blocks.md) and
[codehilite](extensions/code_hilite.md) extensions now prepend the prefix when
code highlighting is disabled.

If you have previously been including the prefix manually in your fenced code blocks, then you will not want a second
instance of the prefix. Similarly, if you are using a third party syntax highlighting tool which does not recognize
the prefix, or requires a different prefix, then you will want to redefine the prefix globally using the `lang_prefix`
configuration option of either the `fenced_code` or `codehilite` extensions.

For example, to configure `fenced_code` to not apply any prefix (the previous behavior), set the option to an empty string:

```pypython
from markdown.extensions.fenced_code import FencedCodeExtension

markdown.markdown(src, extensions=[FencedCodeExtension(lang_prefix='')])
```

!!! note
    When code highlighting is
    [enabled](extensions/fenced_code_blocks.md#enabling-syntax-highlighting),
    the output from Pygments is used unaltered. Currently, Pygments does not
    provide an option to include the language class in the output, let alone
    prefix it. Therefore, any language prefix is only applied when syntax
    highlighting is disabled.

#### Attribute Lists are more strict (#898).

Empty curly braces are now completely ignored by the [Attribute List](extensions/attr_list.md) extension. Previously, the extension would
recognize them as attribute lists and remove them from the document. Therefore, it is no longer necessary to backslash
escape a set of curly braces which are empty or only contain whitespace.

Despite not being documented, previously an attribute list could be defined anywhere within a table cell and get
applied to the cell (`<td>` element). Now the attribute list must be defined at the end of the cell content and must
be separated from the rest of the content by at least one space. This makes it easy to differentiate between attribute
lists defined on inline elements within a cell and the attribute list for the cell itself. It is also more consistent
with how attribute lists are defined on other types of elements.

The extension has also added support for defining attribute lists on table header cells (`<th>` elements) in the same
manner as data cells (`<td>` elements).

In addition, the documentation for the extensions received an overhaul. The features (#987) and limitations (#965) of the extension are now fully documented.

### Added

* All Pygments' options are now available for syntax highlighting (#816).
    - The [Codehilite](extensions/code_hilite.md) extension now accepts any options
      which Pygments supports as global configuration settings on the extension.
    - [Fenced Code Blocks](extensions/fenced_code_blocks.md) will accept any of the
      same options on individual code blocks.
    - Any of the previously supported aliases to Pygments' options continue to be
      supported at this time. However, it is recommended that the Pygments option names
      be used directly to ensure continued compatibility in the future.

* [Fenced Code Blocks](extensions/fenced_code_blocks.md) now work with
  [Attribute Lists](extensions/attr_list.md) when syntax highlighting is disabled.
  Any random HTML attribute can be defined and set on the `<code>` tag of fenced code
  blocks when the `attr_list` extension is enabled (#816).

* The HTML parser has been completely replaced. The new HTML parser is built on Python's
  [`html.parser.HTMLParser`](https://docs.python.org/3/library/html.parser.html), which
  alleviates various bugs and simplify maintenance of the code (#803, #830).

* The [Markdown in HTML](extensions/md_in_html.md) extension has been rebuilt on the
  new HTML Parser, which drastically simplifies it. Note that raw HTML elements with a
  `markdown` attribute defined are now converted to ElementTree Elements and are rendered
  by the serializer. Various bugs have been fixed (#803, #595, #780, and #1012).

* Link reference parsing, abbreviation reference parsing and footnote reference parsing
  has all been moved from `preprocessors` to `blockprocessors`, which allows them to be
  nested within other block level elements. Specifically, this change was necessary to
  maintain the current behavior in the rebuilt Markdown in HTML extension. A few random
  edge-case bugs (see the included tests) were resolved in the process (#803).

* An alternate function `markdown.extensions.headerid.slugify_unicode` has been included
  with the [Table of Contents](extensions/toc.md) extension which supports Unicode
  characters in table of contents slugs. The old `markdown.extensions.headerid.slugify`
  method which removes non-ASCII characters remains the default. Import and pass
  `markdown.extensions.headerid.slugify_unicode` to the `slugify` configuration option
  to use the new behavior.

* Support was added for Python 3.9 and dropped for Python 3.5.

### Fixed

* Document how to pass configuration options to Extra (#1019).
* Fix HR which follows strong em (#897).
* Support short reference image links (#894).
* Avoid a `RecursionError` from deeply nested blockquotes (#799).
* Fix issues with complex emphasis (#979).
* Fix unescaping of HTML characters `<>` in CodeHilite (#990).
* Fix complex scenarios involving lists and admonitions (#1004).
* Fix complex scenarios with nested ordered and unordered lists in a definition list (#918).

## [3.2.2] -- 2020-05-08

### Fixed

* Add `checklinks` tox environment to ensure all links in documentation are good.
* Refactor extension API documentation (#729).
* Load entry_points (for extensions) only once using `importlib.metadata`.
* Do not double escape entities in TOC.
* Correctly report if an extension raises a `TypeError` (#939).
* Raise a `KeyError` when attempting to delete a nonexistent key from the
  extension registry (#939).
* Remove import of `packaging` (or `pkg_resources` fallback) entirely.
* Remove `setuptools` as a run-time dependency (`install_required`).

## [3.2.1] -- 2020-02-12

### Fixed

* The `name` property in `toc_tokens` from the TOC extension now
  escapes HTML special characters (`<`, `>`, and `&`).

## [3.2] -- 2020-02-07

### Changed

#### Drop support for Python 2.7

Python 2.7 reaches end-of-life on 2020-01-01 and Python-Markdown 3.2 has dropped
support for it. Please upgrade to Python 3, or use Python-Markdown 3.1.

#### `em` and `strong` inline processor changes

In order to fix issue #792, `em`/`strong` inline processors were refactored. This
translated into removing many of the existing inline processors that handled this
logic:

* `em_strong`
* `strong`
* `emphasis`
* `strong2`
* `emphasis`

These processors were replaced with two new ones:

* `em_strong`
* `em_strong2`

The [`legacy_em`](extensions/legacy_em.md) extension was also modified with new,
refactored logic and simply overrides the `em_strong2` inline processor.

#### CodeHilite now always wraps with `<code>` tags

Before, the HTML generated by CodeHilite looked like:
- `<pre><code>foo = 'bar'</code></pre>` if you **were not** using Pygments.
- `<pre>foo = 'bar'</pre>`  if you **were** using Pygments.

To make the cases more consistent (and adhere to many Markdown specifications and
HTML code block markup suggestions), CodeHilite will now always additionally wrap
code with `<code>` tags. See #862 for more details.

This change does not alter the Python-Markdown API, but users relying on the old
markup will find their output now changed.

Internally, this change relies on the Pygments 2.4, so you must be using at least
that version to see this effect. Users with earlier Pygments versions will
continue to see the old behavior.

#### `markdown.util.etree` deprecated

Previously, Python-Markdown was using either the `xml.etree.cElementTree` module
or the `xml.etree.ElementTree` module, based on their availability. In modern
Python versions, the former is a deprecated alias for the latter. Thus, the
compatibility layer is deprecated and extensions are advised to use
`xml.etree.ElementTree` directly. Importing `markdown.util.etree` will raise
a `DeprecationWarning` beginning in version 3.2 and may be removed in a future
release.

Therefore, extension developers are encouraged to replace
`from markdown.util import etree` with
`import xml.etree.ElementTree as etree` in their code.

### Added

* Some new configuration options have been added to the [toc](extensions/toc.md)
  extension:

    * The `anchorlink_class` and `permalink_class` options allow class(es) to be
      assigned to the `anchorlink` and `permalink` respectively. This allows using
      icon fonts from CSS for the links. Therefore, an empty string passed to
      `permalink` now generates an empty `permalink`. Previously no `permalink`
      would have been generated. (#776)

    * The `permalink_title` option allows the title attribute of a `permalink` to be
      set to something other than the default English string `Permanent link`. (#877)

* Document thread safety (#812).

* Markdown parsing in HTML has been exposed via a separate extension called
  [`md_in_html`](extensions/md_in_html.md).

* Add support for Python 3.8.

### Fixed

* HTML tag placeholders are no longer included in  `.toc_tokens` (#899).
* Unescape backslash-escaped characters in TOC ids (#864).
* Refactor bold and italic logic in order to solve complex nesting issues (#792).
* Always wrap CodeHilite code in `code` tags (#862).

## [3.1.1] -- 2019-05-20

### Fixed

* Fixed import failure in `setup.py` when the source directory is not
  on `sys.path` (#823).
* Prefer public `packaging` module to pkg_resources' private copy of
  it (#825).

## [3.1] -- 2019-03-25

### Changed

#### `markdown.version` and `markdown.version_info` deprecated

Historically, version numbers were acquired via the attributes
`markdown.version` and `markdown.version_info`. As of 3.0, a more standardized
approach is being followed and versions are acquired via the
`markdown.__version__` and `markdown.__version_info__` attributes.  As of 3.1
the legacy attributes will raise a `DeprecationWarning` if they are accessed. In
a future release the legacy attributes will be removed.

### Added

* A [Contributing Guide](contributing.md) has been added (#732).

* A new configuration option to set the footnote separator has been added. Also,
  the `rel` and `rev` attributes have been removed from footnotes as they are
  not valid in HTML5. The `refs` and `backrefs` classes already exist and
  serve the same purpose (#723).

* A new option for `toc_depth` to set not only the bottom section level,
  but also the top section level. A string consisting of two digits
  separated by a hyphen in between (`"2-5"`), defines the top (`t`) and the
  bottom (`b`) (`<ht>..<hb>`). A single integer still defines the bottom
  section level (`<h1>..<hb>`) only. (#787).

### Fixed

* Update CLI to support PyYAML 5.1.
* Overlapping raw HTML matches no longer leave placeholders behind (#458).
* Emphasis patterns now recognize newline characters as whitespace (#783).
* Version format had been updated to be PEP 440 compliant (#736).
* Block level elements are defined per instance, not as class attributes
  (#731).
* Double escaping of block code has been eliminated (#725).
* Problems with newlines in references has been fixed (#742).
* Escaped `#` are now handled in header syntax (#762).

## [3.0.1] -- 2018-09-28

### Fixed

* Brought back the `version` and `version_info` variables (#709).
* Added support for hexadecimal HTML entities (#712).

## [3.0] -- 2018-09-21

### Changed

#### `enable_attributes` keyword deprecated

The `enable_attributes` keyword is deprecated in version 3.0 and will be
ignored. Previously the keyword was `True` by default and enabled an
undocumented way to define attributes on document elements. The feature has been
removed from version 3.0. As most users did not use the undocumented feature, it
should not affect most users. For the few who did use the feature, it can be
enabled by using the [Legacy Attributes](extensions/legacy_attrs.md)
extension.

#### `smart_emphasis` keyword and `smart_strong` extension deprecated

The `smart_emphasis` keyword is deprecated in version 3.0 and will be ignored.
Previously the keyword was `True` by default and caused the parser to ignore
middle-word emphasis. Additionally, the optional `smart_strong` extension
provided the same behavior for strong emphasis. Both of those features are now
part of the default behavior, and the [Legacy
Emphasis](extensions/legacy_em.md) extension is available to disable that
behavior.

#### `output_formats` simplified to `html` and `xhtml`.

The `output_formats` keyword now only accepts two options: `html` and `xhtml`
Note that if `(x)html1`, `(x)html4` or `(x)html5` are passed in, the number is
stripped and ignored.

#### `safe_mode` and `html_replacement_text` keywords deprecated

Both `safe_mode` and the associated `html_replacement_text` keywords are
deprecated in version 3.0 and will be ignored. The so-called "safe mode" was
never actually "safe" which has resulted in many people having a false sense of
security when using it. As an alternative, the developers of Python-Markdown
recommend that any untrusted content be passed through an HTML sanitizer (like
[Bleach](https://bleach.readthedocs.io/)) after being converted to HTML by
markdown. In fact, [Bleach
Whitelist](https://github.com/yourcelf/bleach-whitelist) provides a curated list
of tags, attributes, and styles suitable for filtering user-provided HTML using
bleach.

If your code previously looked like this:

```pypython
html = markdown.markdown(text, safe_mode=True)
```

Then it is recommended that you change your code to read something like this:

```pypython
import bleach
from bleach_whitelist import markdown_tags, markdown_attrs
html = bleach.clean(markdown.markdown(text), markdown_tags, markdown_attrs)
```

If you are not interested in sanitizing untrusted text, but simply desire to
escape raw HTML, then that can be accomplished through an extension which
removes HTML parsing:

```pypython
from markdown.extensions import Extension

class EscapeHtml(Extension):
    def extendMarkdown(self, md):
        md.preprocessors.deregister('html_block')
        md.inlinePatterns.deregister('html')

html = markdown.markdown(text, extensions=[EscapeHtml()])
```

As the HTML would not be parsed with the above Extension, then the serializer
will escape the raw HTML, which is exactly what happened in previous versions
with `safe_mode="escape"`.

#### Positional arguments deprecated

Positional arguments on the `markdown.Markdown()` class are deprecated as are
all except the `text` argument on the `markdown.markdown()` wrapper function.
Using positional arguments will raise an error. Only keyword arguments should be
used. For example, if your code previously looked like this:

```pypython
html = markdown.markdown(text, [SomeExtension()])
```

Then it is recommended that you change it to read something like this:

```pypython
html = markdown.markdown(text, extensions=[SomeExtension()])
```

!!! Note
    This change is being made as a result of deprecating `"safe_mode"` as the
    `safe_mode` argument was one of the positional arguments. When that argument
    is removed, the two arguments following it will no longer be at the correct
    position. It is recommended that you always use keywords when they are
    supported for this reason.

#### Extension name behavior has changed

In previous versions of Python-Markdown, the built-in extensions received
special status and did not require the full path to be provided. Additionally,
third party extensions whose name started with `"mdx_"` received the same
special treatment. This is no longer the case.

Support has been added for extensions to define an [entry
point](extensions/api.md#entry_point). An entry point is a string name which
can be used to point to an `Extension` class. The built-in extensions now have
entry points which match the old short names. And any third-party extensions
which define entry points can now get the same behavior. See the documentation
for each specific extension to find the assigned name.

If an extension does not define an entry point, then the full path to the
extension must be used. See the [documentation](reference.md#extensions) for
a full explanation of the current behavior.

#### Extension configuration as part of extension name deprecated

The previously documented method of appending the extension configuration
options as a string to the extension name is deprecated and will raise an error.
The [`extension_configs`](reference.md#extension_configs) keyword should be
used instead. See the [documentation](reference.md#extension_configs) for a
full explanation of the current behavior.

#### HeaderId extension deprecated

The HeaderId Extension is deprecated and will raise an error if specified. Use
the [Table of Contents](extensions/toc.md) Extension instead, which offers
most of the features of the HeaderId Extension and more (support for meta data
is missing).

Extension authors who have been using the `slugify` and `unique` functions
defined in the HeaderId Extension should note that those functions are now
defined in the Table of Contents extension and should adjust their import
statements accordingly (`from markdown.extensions.toc import slugify, unique`).

####  Homegrown `OrderedDict` has been replaced with a purpose-built `Registry`

All processors and patterns now get "registered" to a
[Registry](extensions/api.md#registry). A backwards compatible shim is
included so that existing simple extensions should continue to work.
A `DeprecationWarning` will be raised for any code which calls the old API.

#### Markdown class instance references.

Previously, instances of the `Markdown` class were represented as any one of
`md`, `md_instance`, or `markdown`. This inconsistency made it difficult when
developing extensions, or just maintaining the existing code. Now, all instances
are consistently represented as `md`.

The old attributes on class instances still exist, but raise a
`DeprecationWarning` when accessed. Also on classes where the instance was
optional, the attribute always exists now and is simply `None` if no instance
was provided (previously the attribute would not exist).

#### `markdown.util.isBlockLevel` deprecated

The `markdown.util.isBlockLevel` function is deprecated and will raise a
`DeprecationWarning`. Instead, extensions should use the `isBlockLevel` method
of the `Markdown` class instance. Additionally, a list of block level elements
is defined in the `block_level_elements` attribute of the `Markdown` class which
extensions can access to alter the list of elements which are treated as block
level elements.

#### `md_globals` keyword deprecated from extension API

Previously, the `extendMarkdown` method of a `markdown.extensions.Extension`
subclasses accepted an `md_globals` keyword, which contained the value returned
by Python's `globals()` built-in function. As all of the configuration is now
held within the `Markdown` class instance, access to the globals is no longer
necessary and any extensions which expect the keyword will raise a
`DeprecationWarning`. A future release will raise an error.

#### `markdown.version` and `markdown.version_info` deprecated

Historically, version numbers were acquired via the attributes
`markdown.version` and `markdown.version_info`. Moving forward, a more
standardized approach is being followed and versions are acquired via the
`markdown.__version__` and `markdown.__version_info__` attributes.  The legacy
attributes are still available to allow distinguishing versions between the
legacy Markdown 2.0 series and the Markdown 3.0 series, but in the future the
legacy attributes will be removed.

#### Added new, more flexible `InlineProcessor` class

A new `InlineProcessor` class handles inline processing much better and allows
for more flexibility. The new `InlineProcessor` classes no longer utilize
unnecessary pretext and post-text captures. New class can accept the buffer that
is being worked on and manually process the text without regular expressions and
return new replacement bounds. This helps us to handle links in a better way and
handle nested brackets and logic that is too much for regular expression.

### Added

* A new [testing framework](test_tools.md) is included as a part of the
  Markdown library, which can also be used by third party extensions.

* A new `toc_depth` parameter has been added to the
  [Table of Contents Extension](extensions/toc.md).

* A new `toc_tokens` attribute has been added to the Markdown class by the
  [Table of Contents Extension](extensions/toc.md), which contains the raw
  tokens used to build the Table of Contents. Users can use this to build their
  own custom Table of Contents rather than needing to parse the HTML available
  on the `toc` attribute of the Markdown class.

* When the [Table of Contents Extension](extensions/toc.md) is used in
  conjunction with the [Attribute Lists Extension](extensions/attr_list.md)
  and a `data-toc-label` attribute is defined on a header, the content of the
  `data-toc-label` attribute is now used as the content of the Table of Contents
  item for that header.

* Additional CSS class names can be appended to
  [Admonitions](extensions/admonition.md).

## Previous Releases

For information on prior releases, see their changelogs:

* [2.x and earlier](change_log/index.md)


title: Command Line

Using Python-Markdown on the Command Line
=========================================

While Python-Markdown is primarily a python library, a command line script is
included as well. While there are many other command line implementations
of Markdown, you may not have them installed, or you may prefer to use
Python-Markdown's various extensions.

Generally, you will want to have the Markdown library fully installed on your
system to run the command line script. See the
[Installation instructions](install.md) for details.

Python-Markdown's command line script takes advantage of Python's `-m` flag.
Therefore, assuming the python executable is on your system path, use the
following format:

```pybash
python -m markdown [options] [args]
```

That will run the module as a script with the options and arguments provided.

At its most basic usage, one would simply pass in a file name as the only argument:

```pybash
python -m markdown input_file.txt
```

Piping input and output (on `STDIN` and `STDOUT`) is fully supported as well.
For example:

```pybash
echo "Some **Markdown** text." | python -m markdown > output.html
```

Use the `--help` option for a list all available options and arguments:

```pybash
python -m markdown --help
```

If you don't want to call the python executable directly (using the `-m` flag),
follow the instructions below to use a wrapper script:

Setup
-----

Upon installation, the `markdown_py` script will have been copied to
your Python "Scripts" directory. Different systems require different methods to
ensure that any files in the Python "Scripts" directory are on your system
path.

* **Windows**:

    Assuming a default install of Python on Windows, your "Scripts" directory
    is most likely something like `C:\\Python37\Scripts`. Verify the location
    of your "Scripts" directory and add it to you system path.

    Calling `markdown_py` from the command line will call the wrapper batch
    file `markdown_py.bat` in the `"Scripts"` directory created during install.

* __*nix__ (Linux, OSX, BSD, Unix, etc.):

    As each \*nix distribution is different and we can't possibly document all
    of them here, we'll provide a few helpful pointers:

    * Some systems will automatically install the script on your path. Try it
      and see if it works. Just run `markdown_py` from the command line.

    * Other systems may maintain a separate "Scripts" ("bin") directory which
      you need to add to your path. Find it (check with your distribution) and
      either add it to your path or make a symbolic link to it from your path.

    * If you are sure `markdown_py` is on your path, but it still is not being
      found, check the permissions of the file and make sure it is executable.

    As an alternative, you could just `cd` into the directory which contains
    the source distribution, and run it from there. However, remember that your
    markdown text files will not likely be in that directory, so it is much
    more convenient to have `markdown_py` on your path.

!!!Note
    Python-Markdown uses `"markdown_py"` as a script name because the Perl
    implementation has already taken the more obvious name "markdown".
    Additionally, the default Python configuration on some systems would cause a
    script named `"markdown.py"` to fail by importing itself rather than the
    markdown library. Therefore, the script has been named `"markdown_py"` as a
    compromise. If you prefer a different name for the script on your system, it
    is suggested that you create a symbolic link to `markdown_py` with your
    preferred name.

Usage
-----

To use `markdown_py` from the command line, run it as

```pybash
markdown_py input_file.txt
```

or

```pybash
markdown_py input_file.txt > output_file.html
```

For a complete list of options, run

```pybash
markdown_py --help
```

Using Extensions
----------------

To load a Python-Markdown extension from the command line use the `-x`
(or `--extension`) option. The extension module must be on your `PYTHONPATH`
(see the [Extension API](extensions/api.md) for details). The extension can
then be invoked by the name assigned to an entry point or using Python's dot
notation to point to an extension

For example, to load an extension with the assigned entry point name `myext`,
run the following command:

```pybash
python -m markdown -x myext input.txt
```

And to load an extension with Python's dot notation:

```pybash
python -m markdown -x path.to.module:MyExtClass input.txt
```

To load multiple extensions, specify an `-x` option for each extension:

```pybash
python -m markdown -x myext -x path.to.module:MyExtClass input.txt
```

If the extension supports configuration options (see the documentation for the
extension you are using to determine what settings it supports, if any), you
can pass them in as well:

```pybash
python -m markdown -x myext -c config.yml input.txt
```

The `-c` (or `--extension_configs`) option accepts a file name. The file must be
in either the [YAML] or [JSON] format and contain YAML or JSON data that would
map to a Python Dictionary in the format required by the
[`extension_configs`][ec] keyword of the `markdown.Markdown` class. Therefore,
the file `config.yaml` referenced in the above example might look like this:

```pyyaml
myext:
    option1: 'value1'
    option2: True
```

Similarly, a JSON configuration file might look like this:

```pyjson
{
  "myext":
  {
    "option1": "value1",
    "option2": "value2"
  }
}
```

Note that while the `--extension_configs` option does specify the
`myext` extension, you still need to load the extension with the `-x` option,
or the configuration for that extension will be ignored. Further, if an
extension requires a value that cannot be parsed in JSON (for example a
reference to a function), one has to use a YAML configuration file.

The `--extension_configs` option will only support YAML configuration files if
[PyYAML] is installed on your system. JSON should work with no additional
dependencies. The format of your configuration file is automatically detected.

[ec]: reference.md#extension_configs
[YAML]: https://yaml.org/
[JSON]: https://json.org/
[PyYAML]: https://pyyaml.org/
[2.5 release notes]: change_log/release-2.5.md


# Contributing to Python-Markdown

The following is a set of guidelines for contributing to Python-Markdown and its
extensions, which are hosted in the [Python-Markdown Organization] on GitHub.
These are mostly guidelines, not rules. Use your best judgment, and feel free to
propose changes to this document in a pull request.

## Code of Conduct

This project and everyone participating in it is governed by the
[Python-Markdown Code of Conduct]. By participating, you are expected to uphold
this code. Please report unacceptable behavior to <python.markdown@gmail.com>.

## Project Organization

The core Python-Markdown code base and any built-in extensions are hosted in the
[Python-Markdown/markdown] project on GitHub. Other extensions maintained by the
Python-Markdown project may be hosted as separate repositories in the
[Python-Markdown Organization] on GitHub and must follow best practices for
third-party extensions.

The [Python-Markdown/markdown] project is organized as follows:

* Branch `master` should generally be stable and release-ready at all times.
* Version branches should be used for bug-fixes back-ported to the most recent
  PATCH release.
* No other branches should be created. Any other branches which exist are
  preserved for historical reasons only.

## Issues

Feature requests, bug reports, usage questions, and other issues can all be
raised on the GitHub [issue tracker].

When describing issues try to phrase your ticket in terms of the behavior you
think needs to change rather than the code you think needs to change.

Make sure you're running the latest version of Python-Markdown before reporting
an issue.

Search the issue list first for related items. Be sure to check closed issues
and pull requests. GitHub's search only checks open issues by default.

You may want to check the [syntax rules] and/or [Babelmark] to confirm that your
expectations align with the rules and/or other implementations of Markdown.

If reporting a syntax bug, you must provide the minimal input which exhibits the
behavior, the actual output and the output you expected. All three items must be
provided as textual code blocks (screen-shots are not helpful). It may also be
helpful to point to the [syntax rules] which specifically address the area of
concern.

Feature requests will often be closed with a recommendation that they be
implemented as third party extensions outside of the core Python-Markdown
library. Keeping new feature requests implemented as third party extensions
allows us to keep the maintenance overhead of Python-Markdown to a minimum, so
that the focus can be on continued stability, bug fixes, and documentation.

If you intend to submit a fix for your bug or provide an implementation of your
feature request, it is not necessary to first open an issue. You can report a
bug or make a feature request as part of a pull request. Of course, if you want
to receive feedback on how to implement a bug-fix or feature before submitting
a solution, then it would be appropriate to open an issue first and ask your
questions there.

Having your issue closed does not necessarily mean the end of a discussion. If
you believe your issue has been closed incorrectly, explain why and we'll
consider if it needs to be reopened.

## Pull Requests

A pull request often represents the start of a discussion, and does not
necessarily need to be the final, finished submission. In fact, if you discover
an issue and intend to provide a fix for it, there is no need to open an issue
first. You can report the issue and provide the fix together in a pull request.

All pull requests should be made from your personal fork of the library hosted
in your personal GitHub account. Do not create branches on the
[Python-Markdown/markdown] project for pull requests. All pull requests should
be implemented in a new branch with a unique name. Remember that if you have an
outstanding pull request, pushing new commits to the related branch of your
GitHub repository will also automatically update the pull request. It may help
to review GitHub's documentation on [Creating a pull request from a fork].

If you are providing a fix for a previously reported issue, you must reference
the issue in your commit message. Be sure to prefix the reference with one of
GitHub's [action words] which will automatically close the issue when the pull
request is merged. For example, `fixes #42` and `closes #42` would be
acceptable, whereas `ref #42` would not. Of course, if merging a pull request
should not cause an issue to be closed, then the action word should not be
included when referencing that issue.

Before being accepted, each pull request must include the applicable code, new
tests of all new features, updated tests for any changed features, documentation
updates, and an appropriate update to the changelog. All changes must follow
the applicable style guides. Failure to meet any one of the requirements is
likely to delay any serious consideration of your pull request and may even
cause it to be closed. Of course, if you are in the early stages of development,
you may include a note in the pull request acknowledging that it is incomplete
along with a request for feedback.

Pull requests will generally not be accepted if any tests are failing.
Therefore, it is recommended that you run the tests before submitting your pull
request. After making a pull request, check the build status in the
GitHub interface to ensure that all tests are running as expected. If any checks
fail, you may push additional commits to your branch. GitHub will add those
commits to the pull request and rerun the checks.

It is generally best not to squash multiple commits and force-push your changes
to a pull request. Instead, the maintainers would like to be able to follow the
series of commits along with the discussion about those changes as they
progress over time. If your pull request is accepted, it will be squashed at
that time if deemed appropriate.

## Style Guides

In an effort to maintain consistency, Python-Markdown adheres to the following
style guides in its code and documentation. A pull request may be rejected if it
fails to match the relevant style guides.

### Code Style Guide

Except as noted below, all pull requests should follow Python's standard [PEP8
Style Guide] and are run through [Flake8] to ensure that the style guide is
followed.

Legacy code which does not follow the guidelines should only be updated if and
when other changes (bug fix, feature addition, etc.) are being made to that
section of code. While new features should be given names that follow modern
Python naming conventions, existing names should be preserved to avoid backward
incompatible changes.

Line length is limited to a maximum of 119 characters.

When a line of code does not fit within the line length limit, continuation
lines should align elements wrapped inside parentheses, brackets and braces
using a *hanging indent*. When using a hanging indent there should be no
arguments on the first line and further indentation should be used to clearly
distinguish itself as a continuation line. The closing parenthesis, bracket or
brace should be on a line by itself and should line up under the first character
of the line that starts the multi-line construct.

```pypython
my_list = [
    1, 2, 3,
    4, 5, 6,
]
result = some_function_that_takes_arguments(
    'a', 'b', 'c',
    'd', 'e', 'f',
)
```

When the conditional part of an `if`-statement is long enough to require that it
be written across multiple lines, extra indentation should be included on the
conditional continuation line.

```pypython
if (this_is_one_thing
        and that_is_another_thing):
    do_something()
```

### Documentation Style Guide

Documentation should be in American English. The tone of the documentation
should be simple, plain, objective and well-balanced where possible.

Keep paragraphs reasonably short.

With the exception of code blocks, limit line length to 79 characters. You may
want to use your editor's tools to automatically hard wrap lines of text.

Don't use abbreviations such as 'e.g.' but instead use the long form, such as
'For example'.

The documentation is built from the [Markdown] source files in the [`docs`
directory][docs directory] by the [MkDocs] static site generator. In addition to
the basic Markdown syntax, the following extensions are supported: [extra],
[admonition], [smarty], [codehilite], and [toc].

There are a few conventions you should follow when working on the
documentation.

#### Headers

Headers should use the hash style. For example:

```pymd
## Some important topic
```

The underline style should not be used. Don't do this:

```pymd
Some important topic
====================
```

#### Links

Links should always use the reference style, with the referenced hyperlinks kept
at the end of the document.

```pymd
Here is a link to [some other thing][other-thing].

More text...

[other-thing]: http://example.com/other/thing
```

This style helps keep the documentation source consistent and readable.

If you are linking to another document within Python-Markdown's documentation,
you should use a relative link, and link to the `.md` suffix. If applicable, it
is preferred that the link includes a hash fragment pointing to the specific
section of the page. For example:

```pymd
[authentication]: reference.md#Markdown
```

Linking in this style ensures that the links work when browsing the
documentation on GitHub. If your Markdown editor makes links clickable, they
will work there as well. When the documentation is built, these links will be
converted into regular links which point to the built HTML pages.

#### Notes and Warnings

If you want to draw attention to a note or warning, use the syntax defined in
Python-Markdown's [Admonition Extension]:

```pymd
!!! note

    This is the content of the note.
```

#### Changelog

Any commit/pull request which changes the behavior of the Markdown library in
any way must include an entry in the changelog. If a change only alters the
documentation or tooling for the project, then an entry in the changelog is
not necessary. The current changelog can be found at `docs/changelog.md`.

The current changelog follows the format defined at
[keepachangelog.com](https://keepachangelog.com/en/1.1.0/). The description of
each change should include a reference to the relevant GitHub issue in the
format `#123` (where `123` is the issue number).

Edits to the changelog should generally add entries to the `[unreleased]`
version at the top of the log. A pull request should not alter an entry for a
previously released version, unless it is editing an error in the notes for
that version, or is otherwise expressly deemed appropriate by the project
maintainers.

The current changelog should only document the changes for one MAJOR release and
its various MINOR and PATCH releases (see [Versions](#versions) for an
explanation of MAJOR, MINOR, and PATCH releases). Older versions from previous
series of releases can be found in the archive at `docs/change_log/` and may
follow a different format. Note that the archived changelogs are not in the site
navigation and are only linked from the [Previous
Releases](changelog.md#previous-releases) section of the current changelog.

### Commit Message Style Guide

Use the present tense ("Add feature" not "Added feature").

Use the imperative mood ("Move item to..." not "Moves item to...").

Limit the first line to 72 characters or less.

Reference issues and pull requests liberally after the first line. Include a
summary of the changes/additions made without replicating the content of the
documentation or changelog. This is where an explanation of the choices made
should be found. References to issues and pull requests should only provide the
context in which a choice was made. However, the commit should be able to stand
on its own.

## Development Environment

To start developing on Python-Markdown is it best to create a [fork] of the
project on GitHub. After [cloning your fork] to your local system, you will want
to [configure a remote] that points to the upstream repository so that you can
[sync changes] made in the original repository with your fork.

It is recommended that all development be done from within a Python [virtual
environment], which isolates any experimental code from the general system. To
create a virtual environment, use the following command from the root of the
local working copy of your GitHub fork:

```pysh
virtualenv venv
```

That creates a virtual environment which is contained in the `venv` directory
within your local working copy. Note that the repository is configured so that
git will ignore any files within a directory named `venv` or `ENV` for this
very reason.

On Posix systems (Linux, BSD, MacOS, etc.), use the following command to
activate the environment:

```pysh
source venv/bin/activate
```

On Windows, use this command instead:

```pysh
venv/Scripts/activate
```

See the [User Guide] for more information on using virtual environments.

To be able to run the Markdown library directly while working on it, install the
working copy into the environment in [Development Mode] after activating the
virtual environment for the first time:

```pysh
pip install -e .
```

Now any saved changes will immediately be available within the virtual
environment.

You can run the command line script with the following command:

```pysh
python -m markdown
```

Before building the documentation for the first time, you will need to install
some optional dependencies with the command:

```pysh
pip install -e .[docs]
```

To build the documentation and serve it locally on a development server, run:

```pysh
mkdocs serve
```

Then point your browser at `http://127.0.0.1:8000/`. For a complete list of
options available, view MkDocs' help with the command:

```pysh
mkdocs --help
```

Before running tests for the first time, you will need to install some optional
dependencies with the command:

```pysh
pip install -e .[testing]
```

And you can directly run the tests with:

```pysh
python -m unittest discover tests
```

To get a coverage report after running the tests, use these commands instead:

```pysh
coverage run --source=markdown -m unittest discover tests
coverage report --show-missing
```

!!! note

    Some tests require the [PyTidyLib] library, which depends on the [HTML Tidy]
    library. If you do not have PyTidyLib installed, the tests which depend upon
    it will be skipped. Given the difficulty in installing the HTML Tidy library
    on many systems, you may choose to leave both libraries uninstalled and
    depend on the continuous integration server to run those tests when you
    submit a pull request.

The above setup will only run tests against the code in one version of Python.
However,  Python-Markdown supports multiple versions of Python. Therefore, a
[tox] configuration is included in the repository, which includes test
environments for all supported Python versions, a [Flake8] test environment, and
a spellchecker for the documentation. While it is generally fine to leave those
tests for the continuous integration server to run when a pull request is
submitted, for more advanced changes, you may want to run those tests locally.
To do so, simply install tox:

```pysh
pip install tox
```

Then, to run all configured test environments, simply call the command `tox`
with no arguments. See help (`tox -h`) for more options.

!!! note

    The tox environments expect that some dependencies are already installed on
    your system. For example, by default, any Python version specific
    environment will fail if that version of Python is not installed.
    Additionally, the tox environments assume that the [HTML Tidy] library is
    installed and may fail when attempting to install [PyTidyLib] if it is not.
    Finally, the `spellchecker` environment requires [aspell] and the
    `aspell-en` dictionary to be installed. Unfortunately, installing those
    dependencies may differ significantly from system to system and is outside
    the scope of this guide.

!!! seealso "See Also"

    Python-Markdown provides [test tools] which simply test Markdown syntax.
    Understanding those tools will often help in understanding why a test may be
    failing.

## Versions

Python-Markdown follows [Semantic Versioning] and uses the
`MAJOR.MINOR.PATCH[.dev#|a#|b#|rc#]` format for identifying releases. The status
of the `master` branch should always be identified in the `__version_info__`
tuple defined in [`markdown/__meta__.py`][markdown/__meta__.py]. The contents of
that tuple will automatically be converted into a normalized version which
conforms to [PEP 440]. Each time the version is changed, the continuous
integration server will run a test to ensure that the current version is in a
valid normalized format.

### Version Status

A MAJOR version is in development status when the MINOR version is `0`, the
PATCH version is `0`, and the version includes a `dev` segment.

A MINOR version is in development status when the MINOR version is not `0`, the
PATCH version is `0`, and the version includes a `dev` segment.

At all other times, the code is considered stable and release-ready.

MAJOR and MINOR releases may or may not get pre-releases (alpha, beta, release
candidate, etc.) at the discretion of the project maintainers.

### Version Workflow

Bug fixes may be merged from a pull request to the `master` branch at any time
so long as all tests pass, including one or more new tests which would have
failed prior to the change.

New features and backward incompatible changes may only be merged to the
`master` branch when the MAJOR and/or MINOR version is in development status
pursuant to [Semantic Versioning].

A separate commit to the `master` branch should be made to bump up the MAJOR
and/or MINOR version and set development status. Only then will any pull
requests implementing new features or backward incompatible changes be accepted.

If a bug fix is deemed to be important and the `master` branch is in development
status, a back-port of the fix should be committed to a version branch. If the
appropriate version branch does not exist, then it should be created and a pull
request back-porting the fix made against that branch. The version branch should
be named with the most recently released MINOR version. For example, if the
`master` branch is at `3.1.dev0` and the most recent MINOR release was `3.0.4`,
then the version branch would be named `3.0` and any releases from that branch
would increment the PATCH version only (`3.0.5`, `3.0.6`...).

## Release Process

When a new release is being prepared, the release manager should follow the
following steps:

1. Verify that all outstanding issues and pull requests related to the release
   have been resolved.

2. Confirm that the changelog has been updated and indicate the date and
   version of the new release.

3. Update the version defined in [`markdown/__meta__.py`][markdown/__meta__.py].

4. Build a local copy of the documentation, browse through the pages and
   confirm that no obvious issues exist with the documentation.

5. Create a pull request with a commit message in the following format:

        Bump version to X.X.X

6. After all checks have passed, merge the pull request.

7. Create a git tag with the new version as the tag name and push to the
   [Python-Markdown/markdown] repository. The new tag should trigger a GitHub
   workflow which will automatically deploy the release to PyPI and update the
   documentation.

    In the event that the deployment fails, the following steps can be taken to
    deploy manually:

    - Deploy the release to [PyPI] with the command `make deploy`.

    - Deploy an update to the documentation using [MkDocs]. The following example
      assumes that local clones of the [Python-Markdown/markdown] and
      [`Python-Markdown/Python-Markdown.github.io`][Python-Markdown/Python-Markdown.github.io]
      repositories are in sibling directories named `markdown` and `Python-Markdown.github.io`
      respectively.

            cd Python-Markdown.github.io
            mkdocs gh-deploy --config-file ../markdown/mkdocs.yml --remote-branch master

## Issue and Pull Request Labels

Below are the labels used to track and manages issues and pull requests. The
labels are loosely grouped by their purpose, but it is not necessary for every
issue to have a label from every group, and an issue may have more than one
label from the same group.

### Type of Issue or Pull Request

| Label name                   | Description      |
| ---------------------------- | ---------------- |
| `bug`{ .label .bug }         | Bug report.      |
| `feature`{ .label .feature } | Feature request. |
| `support`{ .label .support } | Support request. |
| `process`{ .label .process } | Discussions regarding policies and development process. |

### Category of Issue or Pull Request

| Label name                       | Description                              |
| -------------------------------- | ---------------------------------------- |
| `core`{ .label .core }           | Related to the core parser code.                   |
| `extension`{ .label .extension } | Related to one or more of the included extensions. |
| `docs`{ .label .docs }           | Related to the project documentation.              |

### Status of Issue

| Label name                              | Description                       |
| --------------------------------------- | --------------------------------- |
| `more-info-needed`{ .label .pending }   | More information needs to be provided.              |
| `needs-confirmation`{ .label .pending } | The alleged behavior needs to be confirmed.         |
| `needs-decision`{ .label .pending }     | A decision needs to be made regarding request.      |
| `confirmed`{ .label .approved }         | Confirmed bug report or approved feature request.   |
| `someday-maybe`{ .label .low }          | Approved **low priority** request.                  |
| `duplicate`{ .label .rejected }         | The issue has been previously reported.             |
| `wontfix`{ .label .rejected }           | The issue will not be fixed for the stated reasons. |
| `invalid`{ .label .rejected }           | Invalid report (user error, upstream issue, etc).   |
| `3rd-party`{ .label .rejected }         | Should be implemented as a third party extension.   |


### Status of Pull Request

| Label name                            | Description                         |
| ------------------------------------- | ----------------------------------- |
| `work-in-progress`{ .label .pending } | A partial solution. More changes will be coming.     |
| `needs-review`{ .label .pending }     | Needs to be reviewed and/or approved.                |
| `requires-changes`{ .label .pending } | Awaiting updates after a review.                     |
| `approved`{ .label .approved }        | The pull request is ready to be merged.              |
| `rejected`{ .label .rejected }        | The pull request is rejected for the stated reasons. |

[Python-Markdown Organization]: https://github.com/Python-Markdown
[Python-Markdown Code of Conduct]: https://github.com/Python-Markdown/markdown/blob/master/CODE_OF_CONDUCT.md
[Python-Markdown/markdown]: https://github.com/Python-Markdown/markdown
[issue tracker]: https://github.com/Python-Markdown/markdown/issues
[syntax rules]: https://daringfireball.net/projects/markdown/syntax
[Babelmark]: https://johnmacfarlane.net/babelmark2/
[Creating a pull request from a fork]: https://help.github.com/articles/creating-a-pull-request-from-a-fork/
[action words]: https://help.github.com/articles/closing-issues-using-keywords/
[PEP8 Style Guide]: https://www.python.org/dev/peps/pep-0008/
[Flake8]: http://flake8.pycqa.org/en/latest/index.html
[Markdown]: https://daringfireball.net/projects/markdown/basics
[docs directory]: https://github.com/Python-Markdown/markdown/tree/master/docs
[MkDocs]: https://www.mkdocs.org/
[extra]: extensions/extra.md
[admonition]: extensions/admonition.md
[smarty]: extensions/smarty.md
[codehilite]: extensions/code_hilite.md
[toc]: extensions/toc.md
[Admonition Extension]: extensions/admonition.md#syntax
[fork]: https://help.github.com/articles/about-forks
[cloning your fork]: https://help.github.com/articles/cloning-a-repository/
[configure a remote]: https://help.github.com/articles/configuring-a-remote-for-a-fork
[sync changes]: https://help.github.com/articles/syncing-a-fork
[virtual environment]: https://virtualenv.pypa.io/en/stable/
[User Guide]: https://virtualenv.pypa.io/en/stable/user_guide.html
[Development Mode]: https://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode
[PyTidyLib]: https://countergram.github.io/pytidylib/
[HTML Tidy]: https://www.html-tidy.org/
[tox]: https://tox.readthedocs.io/en/latest/
[aspell]: http://aspell.net/
[test tools]: test_tools.md
[Semantic Versioning]: https://semver.org/spec/v2.0.0.html
[markdown/__meta__.py]: https://github.com/Python-Markdown/markdown/blob/master/markdown/__meta__.py#L29
[PEP 440]: https://www.python.org/dev/peps/pep-0440/
[PyPI]: https://pypi.org/project/Markdown/
[Python-Markdown/Python-Markdown.github.io]: https://github.com/Python-Markdown/Python-Markdown.github.io

<style type="text/css">
    /* GitHub Label Styles */

    code.label {
        color: #000000;
        font-weight: 600;
        line-height: 15px;
        display: inline-block;
        padding: 4px 6px;
    }
    code.bug {
        background-color: #c45b46;
    }
    code.feature {
        background-color: #7b17d8;
        color: #ffffff;
    }
    code.support {
        background-color: #efbe62;
    }
    code.process {
        background-color: #eec9ff;
    }
    code.core {
        background-color: #0b02e1;
        color: #ffffff;
    }
    code.extension {
        background-color: #709ad8;
    }
    code.docs {
        background-color: #b2ffeb;
    }
    code.approved {
        background-color: #beed6d;
    }
    code.low {
        background-color: #dddddd;
    }
    code.pending {
        background-color: #f0f49a;
    }
    code.rejected {
        background-color: #f7c7be;
    }
</style>
