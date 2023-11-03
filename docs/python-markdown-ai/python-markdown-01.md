# PythonMarkdown源码解析 1

title: Python-Markdown

Python-Markdown
===============

This is a Python implementation of John Gruber's
[Markdown](https://daringfireball.net/projects/markdown/).
It is almost completely compliant with the reference implementation,
though there are a few very minor [differences](#differences). See John's
[Syntax Documentation](https://daringfireball.net/projects/markdown/syntax)
for the syntax rules.

To get started, see the [installation instructions](install.md), the [library
reference](reference.md), and the [command line interface](cli.md).

Goals
-----

The Python-Markdown project is developed with the following goals in mind:

* Maintain a Python library (with an optional CLI wrapper) suited to use in web
  server environments (never raise an exception, never write to stdout, etc.) as
  an implementation of the markdown parser that follows the
  [syntax rules][] and the behavior of the original (markdown.pl)
  implementation as reasonably as possible (see [differences](#differences) for
  a few exceptions).

* Provide an [Extension API](extensions/api.md) which makes it possible
  to change and/or extend the behavior of the parser.

!!! Note

    *This is not a CommonMark implementation*; nor is it trying to be!
    Python-Markdown was developed long before the CommonMark specification was
    released and has always (mostly) followed the [syntax rules][] and behavior
    of the original reference implementation. No accommodations have been made
    to address the changes which CommonMark has suggested. It is recommended
    that you look elsewhere if you want an implementation which follows the
    CommonMark specification.

Features
--------

In addition to the basic markdown syntax, Python-Markdown supports the following
features:

* __International Input__

    Python-Markdown will accept [input](reference.md#text) in any language
    supported by Unicode including bi-directional text. In fact the test suite
    includes documents written in Russian and Arabic.

* __Extensions__

    Various [extensions](extensions/index.md) are provided (including
    [extra](extensions/extra.md)) to change and/or extend the base syntax.
    Additionally, a public [Extension API](extensions/api.md) is available
    to write your own extensions.

* __Output Formats__

    Python-Markdown can output documents with either HTML or XHTML style tags.
    See the [Library Reference](reference.md#output_format) for details.

* __Command Line Interface__

    In addition to being a Python Library, a
    [command line script](cli.md) is available for your convenience.

Differences
-----------

While Python-Markdown strives to fully implement markdown as described in the
[syntax rules](https://daringfireball.net/projects/markdown/syntax), the rules
can be interpreted in different ways and different implementations
occasionally vary in their behavior (see the
[Babelmark FAQ](https://johnmacfarlane.net/babelmark2/faq.html#what-are-some-examples-of-interesting-divergences-between-implementations)
for some examples). Known and intentional differences found in Python-Markdown
are summarized below:

* __Middle-Word Emphasis__

    Python-Markdown defaults to ignoring middle-word emphasis (and strong
    emphasis). In other words, `some_long_filename.txt` will not become
    `some<em>long</em>filename.txt`. This can be switched off if desired. See
    the [Legacy EM Extension](extensions/legacy_em.md) for details.

* __Indentation/Tab Length__

    The [syntax rules](https://daringfireball.net/projects/markdown/syntax#list)
    clearly state that when a list item consists of multiple paragraphs, "each
    subsequent paragraph in a list item **must** be indented by either 4 spaces
    or one tab" (emphasis added). However, many implementations do not enforce
    this rule and allow less than 4 spaces of indentation. The implementers of
    Python-Markdown consider it a bug to not enforce this rule.

    This applies to any block level elements nested in a list, including
    paragraphs, sub-lists, blockquotes, code blocks, etc. They **must** always
    be indented by at least four spaces (or one tab) for each level of nesting.

    In the event that one would prefer different behavior,
    [tab_length](reference.md#tab_length) can be set to whatever length is
    desired. Be warned however, as this will affect indentation for all aspects
    of the syntax (including root level code blocks). Alternatively, a
    [third party extension] may offer a solution that meets your needs.

* __Consecutive Lists__

    While the syntax rules are not clear on this, many implementations (including
    the original) do not end one list and start a second list when the list marker
    (asterisks, pluses, hyphens, and numbers) changes. For consistency,
    Python-Markdown maintains the same behavior with no plans to change in the
    foreseeable future. That said, the [Sane List Extension](extensions/sane_lists.md)
    is available to provide a less surprising behavior.

Support
-------

You may report bugs, ask for help, and discuss various other issues on the [bug tracker][].

[third party extension]: https://github.com/Python-Markdown/markdown/wiki/Third-Party-Extensions
[syntax rules]: https://daringfireball.net/projects/markdown/syntax
[bug tracker]: https://github.com/Python-Markdown/markdown/issues


title: Installation

# Installing Python-Markdown

## The Easy Way

The easiest way to install Python-Markdown is simply to type the
following command from the command line:

```pybash
pip install markdown
```

That's it! You're ready to [use](reference.md) Python-Markdown. Enjoy!

For more detailed instructions on installing Python packages, see the
[Installing Packages] tutorial in the [Python Packaging User Guide].

[Installing Packages]: https://packaging.python.org/tutorials/installing-packages/
[Python Packaging User Guide]: https://packaging.python.org/

## Using the Git Repository {: #git }

If you're the type that likes to live on the edge, you may want to keep up with
the latest additions and bug fixes in the repository between releases.
Python-Markdown is maintained in a Git repository on GitHub.com. To
get a copy of Python-Markdown from the repository do the following from the
command line:

```pybash
pip install git+https://github.com/Python-Markdown/markdown.git
```


title: Library Reference

# Using Markdown as a Python Library

First and foremost, Python-Markdown is intended to be a python library module
used by various projects to convert Markdown syntax into HTML.

## The Basics

To use markdown as a module:

```pypython
import markdown
html = markdown.markdown(your_text_string)
```

## The Details

Python-Markdown provides two public functions ([`markdown.markdown`](#markdown)
and [`markdown.markdownFromFile`](#markdownFromFile)) both of which wrap the
public class [`markdown.Markdown`](#Markdown). If you're processing one
document at a time, these functions will serve your needs. However, if you need
to process multiple documents, it may be advantageous to create a single
instance of the `markdown.Markdown` class and pass multiple documents through
it. If you do use a single instance though, make sure to call the `reset`
method appropriately ([see below](#convert)).

### markdown.markdown(text [, **kwargs]) {: #markdown data-toc-label='markdown.markdown' }

The following options are available on the `markdown.markdown` function:

__text__{: #text }

:   The source Unicode string. (required)

    !!! note "Important"
        Python-Markdown expects a **Unicode** string as input (some simple ASCII binary strings *may* work only by
        coincidence) and returns output as a Unicode string. Do not pass binary strings to it! If your input is
        encoded, (e.g. as UTF-8), it is your responsibility to decode it.  For example:

            :::python
            with open("some_file.txt", "r", encoding="utf-8") as input_file:
                text = input_file.read()
            html = markdown.markdown(text)

        If you want to write the output to disk, you *must* encode it yourself:

            :::python
            with open("some_file.html", "w", encoding="utf-8", errors="xmlcharrefreplace") as output_file:
                output_file.write(html)

__extensions__{: #extensions }

:   A list of extensions.

    Python-Markdown provides an [API](extensions/api.md) for third parties to
    write extensions to the parser adding their own additions or changes to the
    syntax. A few commonly used extensions are shipped with the markdown
    library. See the [extension documentation](extensions/index.md) for a
    list of available extensions.

    The list of extensions may contain instances of extensions and/or strings
    of extension names.

        :::python
        extensions=[MyExtClass(), 'myext', 'path.to.my.ext:MyExtClass']

    !!! note
        The preferred method is to pass in an instance of an extension. Strings
        should only be used when it is impossible to import the Extension Class
        directly (from the command line or in a template).

    When passing in extension instances, each class instance must be a subclass
    of `markdown.extensions.Extension` and any configuration options should be
    defined when initiating the class instance rather than using the
    [`extension_configs`](#extension_configs) keyword. For example:

        :::python
        from markdown.extensions import Extension
        class MyExtClass(Extension):
            # define your extension here...

        markdown.markdown(text, extensions=[MyExtClass(option='value')])

    If an extension name is provided as a string, the string must either be the
    registered entry point of any installed extension or the importable path
    using Python's dot notation.

    See the documentation specific to an extension for the string name assigned
    to an extension as an entry point.  Simply include the defined name as
    a string in the list of extensions. For example, if an extension has the
    name `myext` assigned to it and the extension is properly installed, then
    do the following:

        :::python
        markdown.markdown(text, extensions=['myext'])

    If an extension does not have a registered entry point, Python's dot
    notation may be used instead. The extension must be installed as a
    Python module on your PYTHONPATH. Generally, a class should be specified in
    the name. The class must be at the end of the name and be separated by a
    colon from the module.

    Therefore, if you were to import the class like this:

        :::python
        from path.to.module import MyExtClass

    Then load the extension as follows:

        :::python
        markdown.markdown(text, extensions=['path.to.module:MyExtClass'])

    If only one extension is defined within a module and the module includes a
    `makeExtension` function which returns an instance of the extension, then
    the class name is not necessary. For example, in that case one could do
    `extensions=['path.to.module']`. Check the documentation for a specific
    extension to determine if it supports this feature.

    When loading an extension by name (as a string), you can only pass in
    configuration settings to the extension by using the
    [`extension_configs`](#extension_configs) keyword.

    !!! seealso "See Also"
        See the documentation of the [Extension API](extensions/api.md) for
        assistance in creating extensions.

__extension_configs__{: #extension_configs }

:   A dictionary of configuration settings for extensions.

    Any configuration settings will only be passed to extensions loaded by name
    (as a string). When loading extensions as class instances, pass the
    configuration settings directly to the class when initializing it.

    !!! Note
        The preferred method is to pass in an instance of an extension, which
        does not require use of the `extension_configs` keyword at all.
        See the [extensions](#extensions) keyword for details.

    The dictionary of configuration settings must be in the following format:

        :::python
        extension_configs = {
            'extension_name_1': {
                'option_1': 'value_1',
                'option_2': 'value_2'
            },
            'extension_name_2': {
                'option_1': 'value_1'
            }
        }

    When specifying the extension name, be sure to use the exact same
    string as is used in the [extensions](#extensions) keyword to load the
    extension. Otherwise, the configuration settings will not be applied to
    the extension. In other words, you cannot use the entry point in on
    place and Python dot notation in the other. While both may be valid for
    a given extension, they will not be recognized as being the same
    extension by Markdown.

    See the documentation specific to the extension you are using for help in
    specifying configuration settings for that extension.

__output_format__{: #output_format }:

:   Format of output.

    Supported formats are:

    * `"xhtml"`: Outputs XHTML style tags. **Default**.
    * `"html"`: Outputs HTML style tags.

    The values can be in either lowercase or uppercase.

__tab_length__{: #tab_length }:

: Length of tabs in the source. Default: 4

### `markdown.markdownFromFile (**kwargs)` {: #markdownFromFile data-toc-label='markdown.markdownFromFile' }

With a few exceptions, `markdown.markdownFromFile` accepts the same options as
`markdown.markdown`. It does **not** accept a `text` (or Unicode) string.
Instead, it accepts the following required options:

__input__{: #input } (required)

:   The source text file.

    `input` may be set to one of three options:

    * a string which contains a path to a readable file on the file system,
    * a readable file-like object,
    * or `None` (default) which will read from `stdin`.

__output__{: #output }

:   The target which output is written to.

    `output` may be set to one of three options:

    * a string which contains a path to a writable file on the file system,
    * a writable file-like object,
    * or `None` (default) which will write to `stdout`.

__encoding__{: #encoding }

:   The encoding of the source text file.

    Defaults to `"utf-8"`. The same encoding will always be used for input and output.
    The `xmlcharrefreplace` error handler is used when encoding the output.

    !!! Note
        This is the only place that decoding and encoding of Unicode
        takes place in Python-Markdown. If this rather naive solution does not
        meet your specific needs, it is suggested that you write your own code
        to handle your encoding/decoding needs.

### markdown.Markdown([**kwargs]) {: #Markdown data-toc-label='markdown.Markdown' }

The same options are available when initializing the `markdown.Markdown` class
as on the [`markdown.markdown`](#markdown) function, except that the class does
**not** accept a source text string on initialization. Rather, the source text
string must be passed to one of two instance methods.

!!! warning

    Instances of the `markdown.Markdown` class are only thread safe within
    the thread they were created in. A single instance should not be accessed
    from multiple threads.

#### Markdown.convert(source) {: #convert data-toc-label='Markdown.convert' }

The `source` text must meet the same requirements as the [`text`](#text)
argument of the [`markdown.markdown`](#markdown) function.

You should also use this method if you want to process multiple strings
without creating a new instance of the class for each string.

```pypython
md = markdown.Markdown()
html1 = md.convert(text1)
html2 = md.convert(text2)
```

Depending on which options and/or extensions are being used, the parser may
need its state reset between each call to `convert`.

```pypython
html1 = md.convert(text1)
md.reset()
html2 = md.convert(text2)
```

To make this easier, you can also chain calls to `reset` together:

```pypython
html3 = md.reset().convert(text3)
```

#### Markdown.convertFile(**kwargs) {: #convertFile data-toc-label='Markdown.convertFile' }

The arguments of this method are identical to the arguments of the same
name on the `markdown.markdownFromFile` function ([`input`](#input),
[`output`](#output), and [`encoding`](#encoding)). As with the
[`convert`](#convert) method, this method should be used to
process multiple files without creating a new instance of the class for
each document. State may need to be `reset` between each call to
`convertFile` as is the case with `convert`.


title: Test Tools

# Test Tools

Python-Markdown provides some testing tools which simplify testing actual
Markdown output against expected output. The tools are built on the Python
standard  library [`unittest`][unittest]. Therefore, no additional libraries are
required. While Python-Markdown uses the tools for its own tests, they were
designed and built so that third party extensions could use them as well.
Therefore, the tools are importable from `markdown.test_tools`.

The test tools include two different `unittest.TestCase` subclasses:
`markdown.test_tools.TestCase` and `markdown.test_tools.LegacyTestCase`.

## `markdown.test_tools.TestCase`

The `markdown.test_tools.TestCase` class is a `unittest.TestCase` subclass with
a few additional helpers to make testing Markdown output easier.

Properties
: `default_kwargs`: A `dict` of keywords to pass to Markdown for each
test. The defaults can be overridden on individual tests.

Methods
: `assertMarkdownRenders`: accepts the source text, the expected output, an optional
  dictionary of `expected_attrs`, and any keywords to pass to Markdown. The
  `default_kwargs` defined on the class are used except where overridden by
  keyword arguments. The output and expected output are passed to
  `TestCase.assertMultiLineEqual`. An `AssertionError` is raised with a diff
  if the actual output does not equal the expected output. The optional
  keyword `expected_attrs` accepts a dictionary of attribute names as keys with
  expected values. Each value is checked against the attribute of that
  name on the instance of the `Markdown` class using `TestCase.assertEqual`. An
  `AssertionError` is raised if any value does not match the expected value.

: `dedent`: Dedent triple-quoted strings.

In all other respects, `markdown.test_tools.TestCase` behaves as
`unittest.TestCase`. In fact, `assertMarkdownRenders` tests could be mixed with
other `unittest` style tests within the same test class.

An example Markdown test might look like this:

```pypython
from markdown.test_tools import TestCase

class TestHr(TestCase):
    def test_hr_before_paragraph(self):
        self.assertMarkdownRenders(
            # The Markdown source text used as input
            self.dedent(
                """
                ***
                An HR followed by a paragraph with no blank line.
                """
            ),
            # The expected HTML output
            self.dedent(
                """
                <hr>
                <p>An HR followed by a paragraph with no blank line.</p>
                """
            ),
            # Other keyword arguments to pass to `markdown.markdown`
            output_format='html'
        )
```

## `markdown.test_tools.LegacyTestCase`

In the past Python-Markdown exclusively used file-based tests. Many of those
tests still exist in Python-Markdown's test suite, including the test files from
the [reference implementation][perl] (`markdown.pl`) and [PHP Markdown][PHP].
Each test consists of a matching pair of text and HTML files. The text file
contains a snippet of Markdown source text formatted for a specific syntax
feature and the HTML file contains the expected HTML output of that snippet.
When the test suite is run, each text file is run through Markdown and the
output is compared with the HTML file as a separate unit test. When a test
fails, the error report includes a diff of the expected output compared to the
actual output to easily identify any problems.

A separate `markdown.test_tools.LegacyTestCase` subclass must be created for
each directory of test files. Various properties can be defined within the
subclass to point to a directory of text-based test files and define various
behaviors/defaults for those tests. The following properties are supported:

* `location`: A path to the directory of test files. An absolute path is
  preferred.
* `exclude`: A list of tests to skip. Each test name should comprise of a
  file name without an extension.
* `normalize`: A boolean value indicating if the HTML should be normalized.
  Default: `False`. Note: Normalization of HTML requires that [PyTidyLib] be
  installed on the system. If PyTidyLib is not installed and `normalize` is set
  to `True`, then the test will be skipped, regardless of any other settings.
* `input_ext`: A string containing the file extension of input files.
  Default: `.txt`.
* `output_ext`: A string containing the file extension of expected output files.
  Default: `html`.
* `default_kwargs`: A `markdown.test_tools.Kwargs` instance which stores the
  default set of keyword arguments for all test files in the directory.

In addition, properties can be defined for each individual set of test files
within the directory. The property should be given the name of the file without
the file extension. Any spaces and dashes in the file name should be replaced
with underscores. The value of the property should be a
`markdown.test_tools.Kwargs` instance which contains the keyword arguments that
should be passed to `markdown.markdown` for that test file. The keyword
arguments will "update" the `default_kwargs`.

When the class instance is created during a test run, it will walk the given
directory and create a separate unit test for each set of test files using the
naming scheme: `test_filename`. One unit test will be run for each set of input
and output files.

The definition of an example set of tests might look like this:

```pypython
from markdown.test_tools import LegacyTestCase, Kwargs
import os

# Get location of this file and use to find text file dirs.
parent_test_dir = os.path.abspath(os.path.dirname(__file__))


class TestFoo(LegacyTestCase):
    # Define location of text file directory. In this case, the directory is
    # named "foo" and is in the same parent directory as this file.
    location = os.path.join(parent_test_dir, 'foo')
    # Define default keyword arguments. In this case, unless specified
    # differently, all tests should use the output format "html".
    default_kwargs = Kwargs(output_format='html')

    # The "xhtml" test should override the output format and use "xhtml".
    xhtml = Kwargs(output_format='xhtml')

    # The "toc" test should use the "toc" extension with a custom permalink
    # setting.
    toc = Kwargs(
        extensions=['markdown.extensions.toc'],
        extension_configs={'markdown.extensions.toc': {'permalink': "[link]"}}
    )
```

Note that in the above example, the text file directory may contain many more
text-based test files than `xhtml` (`xhtml.txt` and `xhtml.html`) and `toc`
(`toc.txt` and `toc.html`). As long as each set of files exists as a pair, a
test will be created and run for each of them. Only the `xhtml` and `toc` tests
needed to be specifically identified as they had specific, non-default settings
which needed to be defined.

## Running Python-Markdown's Tests

As all of the tests for the `markdown` library are unit tests, standard
`unittest` methods of calling tests can be used. For example, to run all of
Python-Markdown's tests, from the root of the git repository, run the following
command:

```pysh
python -m unittest discover tests
```

That simple command will search everything in the `tests` directory and it's
sub-directories and run all `unittest` tests that it finds, including
`unittest.TestCase`, `markdown.test_tools.TestCase`, and
`markdown.test_tools.LegacyTestCase` subclasses. Normal [unittest] discovery
rules apply.

!!! seealso "See Also"

    See the [Contributing Guide] for instructions on setting up a
    [development environment] for running the tests.

[unittest]: https://docs.python.org/3/library/unittest.html
[Perl]: https://daringfireball.net/projects/markdown/
[PHP]: http://michelf.com/projects/php-markdown/
[PyTidyLib]: http://countergram.github.io/pytidylib/
[Contributing Guide]: contributing.md
[development environment]: contributing.md#development-environment


title: Change Log

Python-Markdown Change Log
=========================

!!! note

    This is an archive of the changelog prior to the release of version 3.0. See the [current changelog](../changelog.md) for up-to-date details.

Jan 4, 2018: Released version 2.6.11 (a bug-fix release). Added a new
`BACKLINK-TITLE` option to the footnote extension so that non-English
users can provide a custom title to back links (see #610).

Dec 7, 2017: Released version 2.6.10 (a documentation update).

Aug 17, 2017: Released version 2.6.9 (a bug-fix release).

Jan 25, 2017: Released version 2.6.8 (a bug-fix release).

Sept 23, 2016: Released version 2.6.7 (a bug-fix release).

Mar 20, 2016: Released version 2.6.6 (a bug-fix release).

Nov 24, 2015: Released version 2.6.5 (a bug-fix release).

Nov 6, 2015: Released version 2.6.4 (a bug-fix release).

Oct 26, 2015: Released version 2.6.3 (a bug-fix release).

Apr 20, 2015: Released version 2.6.2 (a bug-fix release).

Mar 8, 2015: Released version 2.6.1 (a bug-fix release). The (new)
`yaml` option has been removed from the Meta-Data Extension as it was buggy
(see [#390](https://github.com/Python-Markdown/markdown/issues/390)).

Feb 19, 2015: Released version 2.6 ([Notes](release-2.6.md)).

Nov 19, 2014: Released version 2.5.2 (a bug-fix release).

Sept 26, 2014: Released version 2.5.1 (a bug-fix release).

Sept 12, 2014: Released version 2.5.0 ([Notes](release-2.5.md)).

Feb 16, 2014: Released version 2.4.0 ([Notes](release-2.4.md)).

Mar 22, 2013: Released version 2.3.1 (a bug-fix release).

Mar 14, 2013: Released version 2.3.0 ([Notes](release-2.3.md))

Nov 4, 2012: Released version 2.2.1 (a bug-fix release).

Jul 5, 2012: Released version 2.2.0 ([Notes](release-2.2.md)).

Jan 22, 2012: Released version 2.1.1 (a bug-fix release).

Nov 24, 2011: Released version 2.1.0 ([Notes](release-2.1.md)).

Oct 7, 2009: Released version 2.0.3. (a bug-fix release).

Sept 28, 2009: Released version 2.0.2 (a bug-fix release).

May 20, 2009: Released version 2.0.1 (a bug-fix release).

Mar 30, 2009: Released version 2.0 ([Notes](release-2.0.md)).

Mar 8, 2009: Release Candidate 2.0-rc-1.

Feb 2009: Added support for multi-level lists to new Blockprocessors.

Jan 2009: Added HTML 4 output as an option (thanks Eric Abrahamsen)

Nov 2008: Added Definition List ext. Replaced old core with Blockprocessors.
Broken up into multiple files.

Oct 2008: Changed logging behavior to work better with other systems.
Refactored tree traversing. Added `treap` implementation, then replaced with
OrderedDict. Renamed various processors to better reflect what they actually
do. Refactored footnote ext to match PHP Extra's output.

Sept 2008: Moved `prettifyTree` to a Postprocessor, replaced WikiLink ext
with WikiLinks (note the s) ext (uses bracketed links instead of CamelCase)
and various bug fixes.

August 18 2008: Reorganized directory structure. Added a 'docs' directory
and moved all extensions into a 'markdown-extensions' package.
Added additional documentation and a few bug fixes. (v2.0-beta)

August 4 2008: Updated included extensions to `ElementTree`. Added a
separate command line script. (v2.0-alpha)

July 2008: Switched from home-grown `NanoDOM` to `ElementTree` and
various related bugs (thanks Artem Yunusov).

June 2008: Fixed issues with nested inline patterns and cleaned
up testing framework (thanks Artem Yunusov).

May 2008: Added a number of additional extensions to the
distribution and other minor changes. Moved repository to git from svn.

Mar 2008: Refactored extension API to accept either an
extension name (as a string) or an instance of an extension
(Thanks David Wolever). Fixed various bugs and added doc strings.

Feb 2008: Various bug-fixes mostly regarding extensions.

Feb 18, 2008: Version 1.7.

Feb 13, 2008: A little code cleanup and better documentation
and inheritance for Preprocessors/Postprocessors.

Feb 9, 2008: Double-quotes no longer HTML escaped and raw HTML
honors `<?foo>`, `<@foo>`, and `<%foo>` for those who run markdown on
template syntax.

Dec 12, 2007: Updated docs. Removed encoding argument from Markdown
and markdown as per list discussion. Clean up in prep for 1.7.

Nov 29, 2007: Added support for images inside links. Also fixed
a few bugs in the footnote extension.

Nov 19, 2007: `message` now uses python's logging module. Also removed
limit imposed by recursion in `_process_section()`. You can now parse as
long of a document as your memory can handle.

Nov 5, 2007: Moved `safe_mode` code to a `textPostprocessor` and added
escaping option.

Nov 3, 2007: Fixed convert method to accept empty strings.

Oct 30, 2007: Fixed `BOM` removal (thanks Malcolm Tredinnick). Fixed
infinite loop in bracket regular expression for inline links.

Oct 11, 2007: `LineBreaks` is now an `inlinePattern`. Fixed `HR` in
blockquotes. Refactored `_processSection` method (see tracker #1793419).

Oct 9, 2007: Added `textPreprocessor` (from 1.6b).

Oct 8, 2008: Fixed Lazy Blockquote. Fixed code block on first line.
Fixed empty inline image link.

Oct 7, 2007: Limit recursion on inline patterns. Added a 'safe' tag
to `htmlStash`.

March 18, 2007: Fixed or merged a bunch of minor bugs, including
multi-line comments and markup inside links. (Tracker #s: 1683066,
1671153, 1661751, 1627935, 1544371, 1458139.) -> v. 1.6b

Oct 10, 2006: Fixed a bug that caused some text to be lost after
comments.  Added "safe mode" (user's HTML tags are removed).

Sept 6, 2006: Added exception for PHP tags when handling HTML blocks.

August 7, 2006: Incorporated Sergej Chodarev's patch to fix a problem
with ampersand normalization and HTML blocks.

July 10, 2006: Switched to using `optparse`.  Added proper support for
Unicode.

July 9, 2006: Fixed the `<!--@address.com>` problem (Tracker #1501354).

May 18, 2006: Stopped catching unquoted titles in reference links.
Stopped creating blank headers.

May 15, 2006: A bug with lists, recursion on block-level elements,
run-in headers, spaces before headers, Unicode input (thanks to Aaron
Swartz). Sourceforge tracker #s: 1489313, 1489312, 1489311, 1488370,
1485178, 1485176. (v. 1.5)

Mar. 24, 2006: Switched to a not-so-recursive algorithm with
`_handleInline`.  (Version 1.4)

Mar. 15, 2006: Replaced some instance variables with class variables
(a patch from Stelios Xanthakis).  Chris Clark's new regexps that do
not trigger mid-word underlining.

Feb. 28, 2006: Clean-up and command-line handling by Stewart
Midwinter. (Version 1.3)

Feb. 24, 2006: Fixed a bug with the last line of the list appearing
again as a separate paragraph.  Incorporated Chris Clark's "mail-to"
patch.  Added support for `<br />` at the end of lines ending in two or
more spaces.  Fixed a crashing bug when using `ImageReferencePattern`.
Added several utility methods to `Nanodom`.  (Version 1.2)

Jan. 31, 2006: Added `hr` and `hr/` to BLOCK_LEVEL_ELEMENTS and
changed `<hr/>` to `<hr />`.  (Thanks to Sergej Chodarev.)

Nov. 26, 2005: Fixed a bug with certain tabbed lines inside lists
getting wrapped in `<pre><code>`.  (v. 1.1)

Nov. 19, 2005: Made `<!...`, `<?...`, etc. behave like block-level
HTML tags.

Nov. 14, 2005: Added entity code and email auto-link fix by Tiago
Cogumbreiro.  Fixed some small issues with backticks to get 100%
compliance with John's test suite.  (v. 1.0)

Nov. 7, 2005: Added an `unlink` method for documents to aid with memory
collection (per Doug Sauder's suggestion).

Oct. 29, 2005: Restricted a set of HTML tags that get treated as
block-level elements.

Sept. 18, 2005: Refactored the whole script to make it easier to
customize it and made footnote functionality into an extension.
(v. 0.9)

Sept. 5, 2005: Fixed a bug with multi-paragraph footnotes.  Added
attribute support.

Sept. 1, 2005: Changed the way headers are handled to allow inline
syntax in headers (e.g. links) and got the lists to use p-tags
correctly (v. 0.8)

Aug. 29, 2005: Added flexible tabs, fixed a few small issues, added
basic support for footnotes.  Got rid of `xml.dom.minidom` and added
pretty-printing. (v. 0.7)

Aug. 13, 2005: Fixed a number of small bugs in order to conform to the
test suite.  (v. 0.6)

Aug. 11, 2005: Added support for inline HTML and entities, inline
images, auto-links, underscore emphasis. Cleaned up and refactored the
code, added some more comments.

Feb. 19, 2005: Rewrote the handling of high-level elements to allow
multi-line list items and all sorts of nesting.

Feb. 3, 2005: Reference-style links, single-line lists, backticks,
escape, emphasis in the beginning of the paragraph.

Nov. 2004: Added links, blockquotes, HTML blocks to Manfred
Stienstra's code

Apr. 2004: Manfred's version at `http://www.dwerg.net/projects/markdown/`


title: Release Notes for v2.0

Python-Markdown 2.0 Release Notes
=================================

We are happy to release Python-Markdown 2.0, which has been over a year in the
making. We have rewritten significant portions of the code, dramatically
extending the extension API, increased performance, and added numerous
extensions to the distribution (including an extension that mimics PHP Markdown
Extra), all while maintaining backward compatibility with the end user API in
version 1.7.

Python-Markdown supports Python versions 2.3, 2.4, 2.5, and 2.6. We have even
released a version converted to Python 3.0!

Backwards-incompatible Changes
------------------------------

While Python-Markdown has experienced numerous internal changes, those changes
should only affect extension authors. If you have not written your own
extensions, then you should not need to make any changes to your code.
However, you may want to ensure that any third party extensions you are using
are compatible with the new API.

The new extension API is fully [documented](../extensions/api.md) in the docs.
Below is a summary of the significant changes:

* The old home-grown NanoDOM has been replaced with ElementTree. Therefore all
  extensions must use ElementTree rather than the old NanoDOM.
* The various processors and patterns are now stored with OrderedDicts rather
  than lists. Any code adding processors and/or patterns into Python-Markdown
  will need to be adjusted to use the new API using OrderedDicts.
* The various types of processors available have been either combined, added,
  or removed. Ensure that your processors match the currently supported types.

What's New in Python-Markdown 2.0
---------------------------------

Thanks to the work of Artem Yunusov as part of GSoC 2008, Python-Markdown uses
ElementTree internally to build the (X)HTML document from markdown source text.
This has resolved various issues with the older home-grown NanoDOM and made
notable increases in performance.

Artem also refactored the Inline Patterns to better support nested patterns
which has resolved many inconsistencies in Python-Markdown's parsing of the
markdown syntax.

The core parser had been completely rewritten, increasing performance and, for
the first time, making it possible to override/add/change the way block level
content is parsed.

Python-Markdown now parses markdown source text more closely to the other
popular implementations (Perl, PHP, etc.) than it ever has before. With the
exception of a few minor insignificant differences, any difference should be
considered a bug, rather than a limitation of the parser.

The option to return HTML4 output as apposed to XHTML has been added. In
addition, extensions should be able to easily add additional output formats.

As part of implementing markdown in the Dr. Project project (a Trac fork), among
other things, David Wolever refactored the "extension" keyword so that it
accepts either the extension names as strings or instances of extensions. This
makes it possible to include multiple extensions in a single module.

Numerous extensions are included in the distribution by default. See
[available_extensions](../extensions/index.md) for a complete list.

See the [Change Log](index.md) for a full list of changes.



title: Release Notes for v2.1

Python-Markdown 2.1 Release Notes
=================================

We are pleased to release Python-Markdown 2.1 which makes many
improvements on 2.0. In fact, we consider 2.1 to be what 2.0 should have been.
While 2.1 consists mostly of bug fixes, bringing Python-Markdown more inline
with other implementations, some internal improvements were made to the parser,
a few new built-in extensions were added, and HTML5 support was added.

Python-Markdown supports Python versions 2.4, 2.5, 2.6, 2.7, 3.1, and 3.2 out
of the box. In fact, the same code base installs on Python 3.1 and 3.2 with no
extra work by the end user.

Backwards-incompatible Changes
------------------------------

While Python-Markdown has received only minor internal changes since the last
release, there are a few backward-incompatible changes to note:

* Support had been dropped for Python 2.3. No guarantees are made that the
  library will work in any version of Python lower than 2.4. Additionally, while
  the library had been tested with Python 2.4, consider Python 2.4 support to be
  depreciated. It is not likely that any future versions will continue to
  support any version of Python less than 2.5. Note that Python 3.0 is not
  supported due to a bug in its 2to3 tool. If you must use Python-Markdown with
  Python 3.0, it is suggested you manually use Python 3.1's 2to3 tool to do a
  conversion.

* Python-Markdown previously accepted positional arguments on its class and
  wrapper methods. It now expects keyword arguments. Currently, the positional
  arguments should continue to work, but the solution feels hacky and may be
  removed in a future version. All users are encouraged to use keyword arguments
  as documented in the [Library Reference](../reference.md).

* Past versions of Python-Markdown provided module level Global variables which
  controlled the behavior of a few different aspects of the parser. Those global
  variables have been replaced with attributes on the Markdown class.
  Additionally, those attributes are settable as keyword arguments when
  initializing a class instance. Therefore, if you were editing the global
  variables (either by editing the source or by overriding them in your code),
  you should now set them on the class. See the [Library
  Reference](../reference.md) for the options available.

* If you have been using the HeaderId extension to
  define custom ids on headers, you will want to switch to using the new
  [Attribute List](../extensions/attr_list.md) extension. The HeaderId extension
  now only auto-generates ids on headers which have not already had ids defined.
  Note that the [Extra](../extensions/extra.md) extension has been switched to
  use Attribute Lists instead of HeaderId as it did previously.

* Some code was moved into the `markdown.util` namespace which was previously in
  the `markdown` namespace. Extension authors may need to adjust a few import
  statements in their extensions to work with the changes.

* The command line script name was changed to `markdown_py`. The previous name
  (`markdown`) was conflicting with people (and Linux package systems) who also
  had markdown.pl installed on there system as markdown.pl's command line script
  was also named `markdown`. Be aware that installing Python-Markdown 2.1 will
  not remove the old versions of the script with different names. You may want
  to remove them yourself as they are unlikely to work properly.

What's New in Python-Markdown 2.1
---------------------------------

Three new extensions were added. [Attribute Lists](../extensions/attr_list.md),
which was inspired by Maruku's feature of the same name, [Newline to
Break](../extensions/nl2br.md), which was inspired by GitHub Flavored Markdown,
and Smart Strong, which fills a hole in the Extra extension.

HTML5 is now supported. All this really means is that new block level elements
introduced in the HTML5 spec are now properly recognized as raw HTML. As
valid  HTML5 can consist of either HTML4 or XHTML1, there is no need to add a
new HTML5  serializers. That said, `html5` and `xhtml5` have been added as
aliases of the `html4` and `xhtml1` serializers respectively.

An XHTML serializer has been added. Previously, ElementTree's XML serializer
was being used for XHTML output. With the new serializer we are able to avoid
more invalid output like empty elements (i.e., `<p />`) which can choke
browsers.

Improved support for Python 3.x. Now when running `setupy.py install` in
Python 3.1 or greater the 2to3 tool is run automatically. Note that Python 3.0
is not supported due to a bug in its 2to3 tool. If you must use Python-Markdown
with Python 3.0, it is suggested you manually use Python 3.1's 2to3 tool to
do a conversion.

Methods on instances of the Markdown class that do not return results can now
be changed allowing one to do `md.reset().convert(moretext)`.

The Markdown class was refactored so that a subclass could define its own
`build_parser` method which would build a completely different parser. In
other words, one could use the basic machinery in the markdown library to
build a parser of a different markup language without the overhead of building
the markdown parser and throwing it away.

Import statements within markdown have been improved so that third party
libraries can embed the markdown library if they desire (licensing permitting).

Added support for Python's `-m` command line option. You can run the markdown
package as a command line script. Do `python -m markdown [options] [args]`.
Note that this is only fully supported in Python 2.7+. Python 2.5 & 2.6
require you to call the module directly (`markdown.__main__`) rather than
the package (`markdown`). This does not work in Python 2.4.

The command line script has been renamed to `markdown_py` which avoids all the
various problems we had with previous names.  Also improved the command line
script to accept input on `stdin`.

The testing framework has been completely rebuilt using the Nose testing
framework. This provides a number of benefits including the ability to better
test the built-in extensions and other options available to change the parsing
behavior. See the Test Suite documentation for details.

Various bug fixes have been made, which are too numerous to list here. See the
[commit log](https://github.com/Python-Markdown/markdown/commits/master) for a
complete history of the changes.


title: Release Notes for v2.2

Python-Markdown 2.2 Release Notes
=================================

We are pleased to release Python-Markdown 2.2 which makes improvements on 2.1.
While 2.2 is primarily a bug fix release, some internal improvements were made
to the parser, and a few security issues were resolved.

Python-Markdown supports Python versions 2.5, 2.6, 2.7, 3.1, and 3.2 out
of the box.

Backwards-incompatible Changes
------------------------------

While Python-Markdown has received only minor internal changes since the last
release, there are a few backward-incompatible changes to note:

* Support had been dropped for Python 2.4. No guarantees are made that the
  library will work in any version of Python lower than 2.5. Additionally, while
  the library had been tested with Python 2.5, consider Python 2.5 support to be
  depreciated. It is not likely that any future versions will continue to
  support any version of Python less than 2.6.

* For many years Python-Markdown has identified `<ins>` and `<del>` tags in raw
  HTML input as block level tags. As they are actually inline level tags, this
  behavior has been changed. This may result in slightly different output. While
  in most cases, the new output is more correct, there may be a few edge cases
  where a document author has relied on the previous incorrect behavior. It is
  likely that a few adjustments may need to be made to those documents.

* The behavior of the `enable_attributes` keyword has been slightly altered. If
  authors have been using attributes in documents with `safe_mode` on, those
  attributes will no longer be parsed unless `enable_attributes` is explicitly
  set to `True`. This change was made to prevent untrusted authors from
  injecting potentially harmful JavaScript in documents. This change had no
  effect when not in `safe_mode`.

What's New in Python-Markdown 2.2
---------------------------------

The docs were refactored and can now be found at
`http://packages.python.org/Markdown/`. The docs are now maintained in the
Repository and are generated by the `setup.py build_docs` command.

The [Sane_Lists](../extensions/sane_lists.md)
extension was added. The Sane Lists Extension alters the behavior of the
Markdown List syntax to be less surprising by not allowing the mixing of list
types. In other words, an ordered list will not continue when an unordered list
item is encountered and vice versa.

Markdown now excepts a full path to an extension module. In other words, your
extensions no longer need to be in the primary namespace (and start with `mdx_`)
for Markdown to find them. Just do `Markdown(extension=['path.to.some.module'])`.
As long as the provided module contains a compatible extension, the extension
will be loaded.

The BlockParser API was slightly altered to allow `blockprocessor.run` to return
`True` or `False` which provides more control to the block processor loop from
within any Blockprocessor instance.

Various bug fixes have been made. See the
[commit log](https://github.com/Python-Markdown/markdown/commits/master)
for a complete history of the changes.


title: Release Notes for v2.3

Python-Markdown 2.3 Release Notes
=================================

We are pleased to release Python-Markdown 2.3 which adds one new extension,
removes a few old (obsolete) extensions, and now runs on both Python 2 and
Python 3 without running the 2to3 conversion tool. See the list of changes
below for details.

Python-Markdown supports Python versions 2.6, 2.7, 3.1, 3.2, and 3.3.

Backwards-incompatible Changes
------------------------------

* Support has been dropped for Python 2.5. No guarantees are made that the
  library will work in any version of Python lower than 2.6. As all supported
  Python versions include the ElementTree library, Python-Markdown will no
  longer try to import a third-party installation of ElementTree.

* All classes are now "new-style" classes. In other words, all classes subclass
  from 'object'. While this is not likely to affect most users, extension
  authors may need to make a few minor adjustments to their code.

* "safe_mode" has been further restricted. Markdown formatted links must be of a
  known white-listed scheme when in "safe_mode" or the URL is discarded. The
  white-listed schemes are: 'HTTP', 'HTTPS', 'FTP', 'FTPS', 'MAILTO', and
  'news'. Schemeless URLs are also permitted, but are checked in other ways - as
  they have been for some time.

* The ids assigned to footnotes now contain a dash (`-`) rather than a colon
  (`:`) when `output_format` it set to `"html5"` or `"xhtml5"`. If you are
  making reference to those ids in your JavaScript or CSS and using the HTML5
  output, you will need to update your code accordingly. No changes are
  necessary if you are outputting XHTML (the default) or HTML4.

* The `force_linenos` configuration setting of the CodeHilite extension has been
  marked as **Pending Deprecation** and a new setting `linenums` has been added
  to replace it. See documentation for the [CodeHilite Extension] for an
  explanation of the new `linenums` setting. The new setting will honor the old
  `force_linenos` if it is set, but it will raise a `PendingDeprecationWarning`
  and will likely be removed in a future version of Python-Markdown.

[CodeHilite Extension]: ../extensions/code_hilite.md

* The "RSS" extension has been removed and no longer ships with Python-Markdown.
  If you would like to continue using the extension (not recommended), it is
  archived on [GitHub](https://gist.github.com/waylan/4773365).

* The "HTML Tidy" Extension has been removed and no longer ships with
  Python-Markdown. If you would like to continue using the extension (not
  recommended), it is archived on
  [GitHub](https://gist.github.com/waylan/5152650). Note that the underlying
  library, uTidylib, is not Python 3 compatible. Instead, it is recommended that
  the newer [PyTidyLib] (version 0.2.2+ for Python 3 comparability - install
  from GitHub not PyPI) be used. As the API for that library is rather simple,
  it is recommended that the output of Markdown be wrapped in a call to
  PyTidyLib rather than using an extension (for example:
  `tidylib.tidy_fragment(markdown.markdown(source), options={...})`).

[PyTidyLib]: http://countergram.github.io/pytidylib/

What's New in Python-Markdown 2.3
---------------------------------

* The entire code base now universally runs in Python 2 and Python 3 without any
  need for running the 2to3 conversion tool. This not only simplifies testing,
  but by using Unicode_literals, results in more consistent behavior across
  Python versions. Additionally, the relative imports (made possible in Python 2
  via absolute_import) allows the entire library to more easily be embedded in a
  sub-directory of another project. The various files within the library will
  still import each other properly even though 'markdown' may not be in Python's
  root namespace.

* The [Admonition Extension] has been added, which implements [rST-style][rST]
  admonitions in the Markdown syntax. However, be warned that this extension is
  experimental and the syntax and behavior is still subject to change. Please
  try it out and report bugs and/or improvements.

[Admonition Extension]: ../extensions/admonition.md
[rST]: http://docutils.sourceforge.net/docs/ref/rst/directives.html#specific-admonitions

* Various bug fixes have been made. See the [commit
  log](https://github.com/Python-Markdown/markdown/commits/master) for a
  complete history of the changes.


title:      Release Notes for v2.4

Python-Markdown 2.4 Release Notes
=================================

We are pleased to release Python-Markdown 2.4 which adds one new extension
and fixes various bugs. See the list of changes below for details.

Python-Markdown supports Python versions 2.6, 2.7, 3.1, 3.2, and 3.3.

Backwards-incompatible Changes
------------------------------

* The `force_linenos` configuration setting of the CodeHilite extension has been
  marked as **Deprecated**. It had previously been marked as "Pending
  Deprecation" in version 2.3 when a new setting `linenums` was added to replace
  it. See documentation for the [CodeHilite Extension] for an explanation of the
  new `linenums` setting. The new setting will honor the old `force_linenos` if
  it is set, but `force_linenos` will raise a `DeprecationWarning` and will
  likely be removed in a future version of Python-Markdown.

[CodeHilite Extension]: ../extensions/code_hilite.md

* URLs are no longer percent-encoded. This improves compatibility with the
  original (written in Perl) Markdown implementation. Please percent-encode your
  URLs manually when needed.

What's New in Python-Markdown 2.4
---------------------------------

* Thanks to the hard work of [Dmitry Shachnev] the [Smarty Extension] has been
  added, which implements [SmartyPants] using Python-Markdown's Extension API.
  This offers a few benefits over a third party script. The HTML does not need
  to be "tokenized" twice, no hacks are required to combine SmartyPants and code
  highlighting, and we get markdown's escaping feature for free. Please try it
  out and report bugs and/or improvements.

[Dmitry Shachnev]: https://github.com/mitya57
[Smarty Extension]: ../extensions/smarty.md
[SmartyPants]: https://daringfireball.net/projects/smartypants/

* The [Table of Contents Extension] now supports new `permalink` option for
  creating [Sphinx]-style anchor links.

[Table of Contents Extension]: ../extensions/toc.md
[Sphinx]: http://sphinx-doc.org/

* It is now possible to enable Markdown formatting inside HTML blocks by
  appending `markdown=1` to opening tag attributes. See [Markdown Inside HTML
  Blocks] section for details. Thanks to [ryneeverett] for implementing this
  feature.

[Markdown Inside HTML Blocks]: ../extensions/extra.md#nested-markdown-inside-html-blocks
[ryneeverett]: https://github.com/ryneeverett

* The code blocks now support emphasizing some of the code lines. To use this
  feature, specify `hl_lines` option after language name, for example (using the
  [Fenced Code Extension]):

        ```py.python hl_lines="1 3"
        # This line will be emphasized.
        # This one won't.
        # This one will be also emphasized.
        ```

    Thanks to [A. Jesse Jiryu Davis] for implementing this feature.

[Fenced Code Extension]: ../extensions/fenced_code_blocks.md
[A. Jesse Jiryu Davis]: https://github.com/ajdavis

* Various bug fixes have been made. See the [commit
  log](https://github.com/Python-Markdown/markdown/commits/master) for a
  complete history of the changes.


title:      Release Notes for v2.5

Python-Markdown 2.5 Release Notes
=================================

We are pleased to release Python-Markdown 2.5 which adds a few new features
and fixes various bugs. See the list of changes below for details.

Python-Markdown version 2.5 supports Python versions 2.7, 3.2, 3.3, and 3.4.

Backwards-incompatible Changes
------------------------------

* Python-Markdown no longer supports Python version 2.6. You must be using Python
  versions 2.7, 3.2, 3.3, or 3.4.

[importlib]: https://pypi.org/project/importlib/

* The `force_linenos` configuration key on the [CodeHilite Extension] has been **deprecated**
  and will raise a `KeyError` if provided. In the previous release (2.4), it was
  issuing a `DeprecationWarning`. The [`linenums`][linenums] keyword should be used
  instead, which provides more control of the output.

[CodeHilite Extension]: ../extensions/code_hilite.md
[linenums]: ../extensions/code_hilite.md#usage

* Both `safe_mode` and the associated `html_replacement_text` keywords will be
  deprecated in version 2.6 and will raise a **`PendingDeprecationWarning`** in
  2.5. The so-called "safe mode" was never actually "safe" which has resulted in
  many people having a false sense of security when using it. As an alternative,
  the developers of Python-Markdown recommend that any untrusted content be
  passed through an HTML sanitizer (like [Bleach]) after being converted to HTML
  by markdown.

    If your code previously looked like this:

        html = markdown.markdown(text, same_mode=True)

    Then it is recommended that you change your code to read something like this:

        import bleach
        html = bleach.clean(markdown.markdown(text))

    If you are not interested in sanitizing untrusted text, but simply desire to
    escape raw HTML, then that can be accomplished through an extension which
    removes HTML parsing:

        from markdown.extensions import Extension

        class EscapeHtml(Extension):
            def extendMarkdown(self, md, md_globals):
                del md.preprocessors['html_block']
                del md.inlinePatterns['html']

        html = markdown.markdown(text, extensions=[EscapeHtml()])

    As the HTML would not be parsed with the above Extension, then the
    serializer will escape the raw HTML, which is exactly what happens now when
    `safe_mode="escape"`.

[Bleach]: https://bleach.readthedocs.io/

* Positional arguments on the `markdown.Markdown()` are pending deprecation as are
  all except the `text` argument on the `markdown.markdown()` wrapper function.
  Only keyword arguments should be used. For example, if your code previously
  looked like this:

         html = markdown.markdown(text, ['extra'])

    Then it is recommended that you change it to read something like this:

        html = markdown.markdown(text, extensions=['extra'])

    !!! Note
        This change is being made as a result of deprecating `"safe_mode"` as the
        `safe_mode` argument was one of the positional arguments. When that argument
        is removed, the two arguments following it will no longer be at the correct
        position. It is recommended that you always use keywords when they are supported
        for this reason.

* In previous versions of Python-Markdown, the built-in extensions received
  special status and did not require the full path to be provided. Additionally,
  third party extensions whose name started with `"mdx_"` received the same
  special treatment. This behavior will be deprecated in version 2.6 and will
  raise a **`PendingDeprecationWarning`** in 2.5. Ensure that you always use the
  full path to your extensions. For example, if you previously did the
  following:

        markdown.markdown(text, extensions=['extra'])

    You should change your code to the following:

        markdown.markdown(text, extensions=['markdown.extensions.extra'])

    The same applies to the command line:

        $ python -m markdown -x markdown.extensions.extra input.txt

    See the [documentation](../reference.md#extensions) for a full explanation
    of the current behavior.

* The previously documented method of appending the extension configuration as
  a string to the extension name will be deprecated in Python-Markdown
  version 2.6 and will raise a **`PendingDeprecationWarning`** in 2.5. The
  [`extension_configs`](../reference.md#extension_configs) keyword should
  be used instead. See the [documentation](../reference.md#extension-configs)
  for a full explanation of the current behavior.

What's New in Python-Markdown 2.5
---------------------------------

* The [Smarty Extension] has had a number of additional configuration settings
  added, which allows one to define their own substitutions to better support
  languages other than English. Thanks to [Martin Altmayer] for implementing this
  feature.

[Smarty Extension]: ../extensions/smarty.md
[Martin Altmayer]:https://github.com/MartinAltmayer

* Named Extensions (strings passed to the [`extensions`][ex] keyword of
  `markdown.Markdown`) can now point to any module and/or Class on your
  PYTHONPATH. While dot notation was previously supported, a module could not
  be at the root of your PYTHONPATH. The name had to contain at least one dot
  (requiring it to be a sub-module). This restriction no longer exists.

    Additionally, a Class may be specified in the name. The class must be at the
    end of the name (which uses dot notation from PYTHONPATH) and be separated
    by a colon from the module.

    Therefore, if you were to import the class like this:

        from path.to.module import SomeExtensionClass

    Then the named extension would comprise this string:

        "path.to.module:SomeExtensionClass"

    This allows multiple extensions to be implemented within the same module and
    still accessible when the user is not able to import the extension directly
    (perhaps from a template filter or the command line).

    This also means that extension modules are no longer required to include the
    `makeExtension` function which returns an instance of the extension class.
    However, if the user does not specify the class name (she only provides
    `"path.to.module"`) the extension will fail to load without the
    `makeExtension` function included in the module. Extension authors will want
    to document carefully what is required to load their extensions.

[ex]: ../reference.md#extensions

* The Extension Configuration code has been refactored to make it a little
  easier for extension authors to work with configuration settings. As a
  result, the [`extension_configs`][ec] keyword now accepts a dictionary
  rather than requiring a list of tuples. A list of tuples is still supported
  so no one needs to change their existing code. This should also simplify the
  learning curve for new users.

    Extension authors are encouraged to review the new methods available on the
    `markdown.extnesions.Extension` class for handling configuration and adjust
    their code going forward. The included extensions provide a model for best
    practices. See the [API] documentation for a full explanation.

[ec]: ../reference.md#extension_configs
[API]: ../extensions/api.md#configsettings

* The [Command Line Interface][cli] now accepts a `--extensions_config` (or
  `-c`) option which accepts a file name and passes the parsed content of a
  [YAML] or [JSON] file to the [`extension_configs`][ec] keyword of the
  `markdown.Markdown` class. The contents of the YAML or JSON must map to a
  Python Dictionary which matches the format required by the
  `extension_configs` keyword. Note that [PyYAML] is required to parse YAML
  files.

[cli]: ../cli.md#using-extensions
[YAML]: https://yaml.org/
[JSON]: https://json.org/
[PyYAML]: https://pyyaml.org/

* The [Admonition Extension][ae] is no longer considered "experimental."

[ae]: ../extensions/admonition.md

* There have been various refactors of the testing framework. While those
  changes will not directly effect end users, the code is being better tested
  which will benefit everyone.

* Various bug fixes have been made. See the [commit
  log](https://github.com/Python-Markdown/markdown/commits/master) for a
  complete history of the changes.


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

### "Shortened" Extension Names Deprecated

In previous versions of Python-Markdown, the built-in extensions received
special status and did not require the full path to be provided. Additionally,
third party extensions whose name started with `"mdx_"` received the same
special treatment. This behavior is deprecated and will raise a
**`DeprecationWarning`** in version 2.6 and an error in the next release. Ensure
that you always use the full path to your extensions. For example, if you
previously did the following:

```pypython
markdown.markdown(text, extensions=['extra'])
```

You should change your code to the following:

```pypython
markdown.markdown(text, extensions=['markdown.extensions.extra'])
```

The same applies to the command line:

```pypython
python -m markdown -x markdown.extensions.extra input.txt
```

Similarly, if you have used a third party extension (for example `mdx_math`),
previously you might have called it like this:

```pypython
markdown.markdown(text, extensions=['math'])
```

As the `"mdx"` prefix will no longer be appended, you will need to change your
code as follows (assuming the file `mdx_math.py` is installed at the root of
your PYTHONPATH):

```pypython
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

```pypython
ext = SomeExtension(configs={'somekey': 'somevalue'})
```

That code should be updated to pass in the options directly:

```pypython
ext = SomeExtension(somekey='somevalue')
```

Extension authors will want to note that this affects the `makeExtension`
function as well. Previously it was common for the function to be defined as
follows:

```pypython
def makeExtension(configs=None):
    return SomeExtension(configs=configs)
```

Extension authors will want to update their code to the following instead:

```pypython
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

```pypython
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
