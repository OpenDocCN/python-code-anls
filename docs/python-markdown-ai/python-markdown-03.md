# PythonMarkdown源码解析 3

title: Definition Lists Extension

Definition Lists
================

Summary
-------

The Definition Lists extension adds the ability to create definition lists in
Markdown documents.

This extension is included in the standard Markdown library.

Syntax
------

Definition lists are defined using the syntax established in
[PHP Markdown Extra][php].

[php]: http://www.michelf.com/projects/php-markdown/extra/#def-list

Thus, the following text (taken from the above referenced PHP documentation):

```pymd
Apple
:   Pomaceous fruit of plants of the genus Malus in
    the family Rosaceae.

Orange
:   The fruit of an evergreen tree of the genus Citrus.
```

will be rendered as:

```pyhtml
<dl>
<dt>Apple</dt>
<dd>Pomaceous fruit of plants of the genus Malus in
the family Rosaceae.</dd>

<dt>Orange</dt>
<dd>The fruit of an evergreen tree of the genus Citrus.</dd>
</dl>
```

Usage
-----

See [Extensions](index.md) for general extension usage. Use `def_list` as the
name of the extension.

This extension does not accept any special configuration options.

A trivial example:

```pypython
markdown.markdown(some_text, extensions=['def_list'])
```


title: Extra Extension

# Python-Markdown Extra

## Summary

A compilation of various Python-Markdown extensions that (mostly) imitates
[PHP Markdown Extra](http://michelf.com/projects/php-markdown/extra/).

The supported extensions include:

* [Abbreviations](abbreviations.md)
* [Attribute Lists](attr_list.md)
* [Definition Lists](definition_lists.md)
* [Fenced Code Blocks](fenced_code_blocks.md)
* [Footnotes](footnotes.md)
* [Tables](tables.md)
* [Markdown in HTML](md_in_html.md)

See each individual extension for syntax documentation. Extra and all its
supported extensions are included in the standard Markdown library.

## Usage

From the Python interpreter:

```pypycon
>>> import markdown
>>> html = markdown.markdown(text, extensions=['extra'])
```

To pass configuration options to the extensions included with Extra, they must be passed to Extra, with the
underlying extension identified as well. In that way Extra will have access to the options and can pass them on to
the appropriate underlying extension.

```pypython
config = {
    'extra': {
        'footnotes': {
            'UNIQUE_IDS': True
        },
        'fenced_code': {
            'lang_prefix': 'lang-'
        }
    },
    'toc': {
        'permalink': True
    }
}

html = markdown.markdown(text, extensions=['extra', 'toc'], extension_configs=config)
```

Note that in the above example, `footnotes` and `fenced_code` are both nested under the `extra` key as those
extensions are included with Extra. However, the `toc` extension is not included with `extra` and therefore its
configuration options are not nested under the `extra` key.

See each individual extension for a list of supported configuration options.

There are many other [extensions](index.md) which are distributed with Python-Markdown that are not included here in
Extra. The features of those extensions are not part of PHP Markdown Extra, and therefore, not part of Python-Markdown
Extra.


title: Fenced Code Blocks Extension

# Fenced Code Blocks

## Summary

The Fenced Code Blocks extension adds a secondary way to define code blocks, which overcomes a few limitations of
indented code blocks.

This extension is included in the standard Markdown library.

## Syntax

Fenced Code Blocks are defined using the syntax originally established in [PHP Markdown Extra][php] and popularized by
[GitHub Flavored Markdown][gfm].

Fenced code blocks begin with three or more backticks (` ```py `) or tildes (`~~~`) on a line by themselves and end with
a matching set of backticks or tildes on a line by themselves. The closing set must contain the same number and type
of characters as the opening set. It is recommended that a blank line be placed before and after the code block.

````md
A paragraph before the code block.

```py
a one-line code block
```

A paragraph after the code block.
```py`

While backticks seem to be more popular among users, tildes may be used as well.

````md
~~~
a one-line code block
~~~
```py`

To include a set of backticks (or tildes) within a code block, use a different number of backticks for the
delimiters.

`````md
```py`
```
```py`
`````

Fenced code blocks can have a blank line as the first and/or last line of the code block and those lines will be
preserved.

```py`md
```

a three-line code block

```py
````

Unlike indented code blocks, a fenced code block can immediately follow a list item without becoming
part of the list.

```py`md
* A list item.

```
not part of the list
```py
````

!!! warning

    Fenced Code Blocks are only supported at the document root level. Therefore, they cannot be nested inside lists or
    blockquotes. If you need to nest fenced code blocks, you may want to try the third party extension [SuperFences]
    instead.

### Attributes

Various attributes may be defined on a per-code-block basis by defining them immediately following the opening
deliminator. The attributes should be wrapped in curly braces `{}` and be on the same line as the deliminator. It is
generally best to separate the attribute list from the deliminator with a space. Attributes within the list must be
separated by a space.

```py`md
``` { attributes go here }
a code block with attributes
```py
````

How those attributes will affect the output will depend on various factors as described below.

#### Language

The language of the code within a code block can be specified for use by syntax highlighters, etc. The language should
be prefixed with a dot and not contain any whitespace (`.language-name`).

```py`md
``` { .html }
<p>HTML Document</p>
```py
````

So long as the language is the only option specified, the curly brackets and/or the dot may be excluded:

```py`md
``` html
<p>HTML Document</p>
```py
````

Either of the above examples will output the following HTML:

```pyhtml
<pre><code class="language-html">&lt;p&gt;HTML Document&lt;/p&gt;
</code></pre>
```

Note that the language name has been prefixed with `language-` and it has been assigned to the `class` attribute on
the `<code>` tag, which is the format suggested by the [HTML 5 Specification][html5] (see the second "example" in the
Specification). While `language` is the default prefix, the prefix may be overridden using the
[`lang_prefix`](#lang_prefix) configuration option.

#### Classes

In addition to the language, additional classes may be defined by prefixing them with a dot, just like the language.

```py`md
``` { .html .foo .bar }
<p>HTML Document</p>
```py
````

When defining multiple classes, only the first class will be used as the "language" for the code block. All others are
assigned to the `<pre>` tag unaltered. Additionally, the curly braces and dot are required for all classes, including
the language class if more than one class is defined.

The above example will output the following HTML:

```pyhtml
<pre class="foo bar"><code class="language-html">&lt;p&gt;HTML Document&lt;/p&gt;
</code></pre>
```

#### ID

An `id` can be defined for a code block, which would allow a link to point directly to the code block using a URL
hash. IDs must be prefixed with a hash character (`#`) and only contain characters permitted in HTML `id` attributes.

```py`md
``` { #example }
A linkable code block
```py
````

The `id` attribute is assigned to the `<pre>` tag of the output. The above example will output the following HTML:

```pyhtml
<pre id="example"><code>A linkable code block
</code></pre>
```

From elsewhere within the same document, one could link to the code block with `[link](#example)`.

IDs may be defined along with the language, other classes, or any other supported attributes. The order of items does
not matter.

```py`md
``` { #example .lang .foo .bar }
A linkable code block
```py
````

#### Key/Value Pairs

If the `fenced_code` and [`attr_list`][attr_list] extensions are both enabled, then key/value pairs can be defined in
the attribute list. So long as code highlighting is not enabled (see below), the key/value pairs will be assigned as
attributes on the `<code>` tag in the output. Key/value pairs must be defined using the syntax documented for the
`attr_list` extension (for example, values with whitespace must be wrapped in quotes).

```py`md
``` { .lang #example style="color: #333; background: #f8f8f8;" }
A code block with inline styles. Fancy!
```py
````

The above example will output the following HTML:

```pyhtml
<pre id="example"><code class="language-lang"  style="color: #333; background: #f8f8f8;">A code block with inline styles. Fancy!
</code></pre>
```

If the `attr_list` extension is not enabled, then the key/value pairs will be ignored.

#### Syntax Highlighting

If the `fenced_code` extension and syntax highlighting are both enabled, then the [`codehilite`][codehilite]
extension will be used for syntax highlighting the contents of the code block. The language defined in the attribute
list will be passed to `codehilite` to ensure that the correct language is used. If no language is specified and
language guessing is not disabled for the `codehilite` extension, then the language will be guessed.

The `codehilite` extension uses the [Pygments] engine to do syntax highlighting. Any valid Pygments options can be
defined as key/value pairs in the attribute list and will be passed on to Pygments.

```py`md
``` { .lang linenos=true linenostart=42 hl_lines="43-44 50" title="An Example Code Block" }`
A truncated code block...
```py
````

Valid options include any option accepted by Pygments' [`HTMLFormatter`][HTMLFormatter] except for the `full` option,
as well as any options accepted by the relevant [lexer][lexer] (each language has its own lexer). While most lexers
don't have options that are all that useful in this context, there are a few important exceptions. For example, the
[PHP lexer's] `startinline` option eliminates the need to start each code fragment with `<?php`.

!!! note

    The `fenced_code` extension does not alter the output provided by Pygments. Therefore, only options which Pygments
    provides can be utilized. As Pygments does not currently provide a way to define an ID, any ID defined in an
    attribute list will be ignored when syntax highlighting is enabled. Additionally, any key/value pairs which are not Pygments options will be ignored, regardless of whether the `attr_list` extension is enabled.

##### Enabling Syntax Highlighting

To enable syntax highlighting, the [`codehilite`][codehilite] extension must be enabled and the `codehilite`
extension's `use_pygments` option must be set to `True` (the default).

Alternatively, so long as the `codehilite` extension is enabled, you can override a global `use_pygments=False`
option for an individual code block by including `use_pygments=true` in the attribute list. While the `use_pygments`
key/value pair will not be included in the output, all other attributes will behave as they would if syntax
highlighting was enabled only for that code block.

Conversely, to disable syntax highlighting on an individual code block, include `use_pygments=false` in the attribute
list. While the `use_pygments` key/value pair will not be included in the output, all other attributes will behave as
they would if syntax highlighting was disabled for that code block regardless of any global setting.

!!! seealso "See Also"

    You will need to properly install and setup Pygments for syntax highlighting to work. See the `codehilite`
    extension's [documentation][setup] for details.

## Usage

See [Extensions] for general extension usage. Use `fenced_code` as the name of the extension.

See the [Library Reference] for information about configuring extensions.

The following option is provided to configure the output:

* **`lang_prefix`**{#lang_prefix}:
    The prefix prepended to the language class assigned to the HTML `<code>` tag. Default: `language-`.

A trivial example:

```pypython
markdown.markdown(some_text, extensions=['fenced_code'])
```

[php]: http://www.michelf.com/projects/php-markdown/extra/#fenced-code-blocks
[gfm]: https://help.github.com/en/github/writing-on-github/creating-and-highlighting-code-blocks
[SuperFences]: https://facelessuser.github.io/pymdown-extensions/extensions/superfences/
[html5]: https://html.spec.whatwg.org/#the-code-element
[attr_list]: ./attr_list.md
[codehilite]: ./code_hilite.md
[Pygments]: http://pygments.org/
[HTMLFormatter]: https://pygments.org/docs/formatters/#HtmlFormatter
[lexer]: https://pygments.org/docs/lexers/
[PHP lexer's]: https://pygments.org/docs/lexers/#lexers-for-php-and-related-languages
[setup]: ./code_hilite.md#setup
[Extensions]: ./index.md
[Library Reference]: ../reference.md#extensions


title: Footnotes Extension

Footnotes
=========

Summary
-------

The Footnotes extension adds syntax for defining footnotes in Markdown
documents.

This extension is included in the standard Markdown library.

Syntax
------

Python-Markdown's Footnote syntax follows the generally accepted syntax of the
Markdown community at large and almost exactly matches [PHP Markdown Extra][]'s
implementation of footnotes. The only differences involve a few subtleties in
the output.

[PHP Markdown Extra]: http://michelf.com/projects/php-markdown/extra/#footnotes

Example:

```pymd
Footnotes[^1] have a label[^@#$%] and the footnote's content.

[^1]: This is a footnote content.
[^@#$%]: A footnote on the label: "@#$%".
```

A footnote label must start with a caret `^` and may contain any inline text
(including spaces) between a set of square brackets `[]`. Only the first
caret has any special meaning.

A footnote content must start with the label followed by a colon and at least
one space. The label used to define the content must exactly match the label used
in the body (including capitalization and white space). The content would then
follow the label either on the same line or on the next line. The content may
contain multiple lines, paragraphs, code blocks, blockquotes and most any other
markdown syntax. The additional lines must be indented one level (four spaces or
one tab).

When working with multiple blocks, it may be helpful to start the content on a
separate line from the label which defines the content. This way the entire block
is indented consistently and any errors are more easily discernible by the author.

```pymd
[^1]:
    The first paragraph of the definition.

    Paragraph two of the definition.

    > A blockquote with
    > multiple lines.

        a code block

    A final paragraph.
```

Usage
-----

See [Extensions](index.md) for general extension usage. Use `footnotes` as the
name of the extension.

See the [Library Reference](../reference.md#extensions) for information about
configuring extensions.

The following options are provided to configure the output:

* **`PLACE_MARKER`**:
    A text string used to mark the position where the footnotes are rendered.
    Defaults to `///Footnotes Go Here///`.

    If the place marker text is not found in the document, the footnote
    definitions are placed at the end of the resulting HTML document.

* **`UNIQUE_IDS`**:
    Whether to avoid collisions across multiple calls to `reset()`. Defaults to
    `False`.

* **`BACKLINK_TEXT`**:
    The text string that links from the footnote definition back to the position
    in the document. Defaults to `&#8617;`.

* **`SUPERSCRIPT_TEXT`**:
    The text string that links from the position in the document to the footnote
    definition. Defaults to `{}`, i.e. only the footnote's number.

* **`BACKLINK_TITLE`**:
    The text string for the `title` HTML attribute of the footnote definition link.
    The placeholder `{}` will be replaced by the footnote number. Defaults to
    `Jump back to footnote {} in the text`.

* **`SEPARATOR`**:
    The text string used to set the footnote separator. Defaults to `:`.

A trivial example:

```pypython
markdown.markdown(some_text, extensions=['footnotes'])
```

Resetting Instance State
-----

Footnote definitions are stored within the  `markdown.Markdown` class instance between
multiple runs of the class.  This allows footnotes from all runs to be included in
output, with  links and references that are unique, even though the class has been
called multiple times.

However, if needed, the definitions can be cleared between runs by calling `reset`.

For instance, the home page of a blog might include the content from multiple documents.
By not calling `reset`, all of the footnotes will be rendered, and they will all have
unique links and references.

On the other hand, individual blog post pages might need the content from only one
document, and should have footnotes pertaining only to that page. By calling `reset`
between runs, the footnote definitions from the first document will be cleared before
the second document is rendered.

An example of calling `reset`:

```pypython
md = markdown.Markdown(extensions=['footnotes'])
html1 = md.convert(text_with_footnote)
md.reset()
html2 = md.convert(text_without_footnote)
```


title: Extensions

# Extensions

Python Markdown offers a flexible extension mechanism, which makes it possible
to change and/or extend the behavior of the parser without having to edit the
actual source files.

To use an extension, pass it to markdown with the `extensions` keyword.

```pypython
markdown.markdown(some_text, extensions=[MyExtClass(), 'myext', 'path.to.my.ext:MyExtClass'])
```

See the [Library Reference](../reference.md#extensions) for more details.

From the command line, specify an extension with the `-x` option.

```pybash
python -m markdown -x myext -x path.to.module:MyExtClass input.txt > output.html
```

See the [Command Line docs](../cli.md) or use the `--help` option for more details.

!!! seealso "See Also"
    If you would like to write your own extensions, see the
    [Extension API](api.md) for details.

Officially Supported Extensions
-------------------------------

The extensions listed below are included with (at least) the most recent release
and are officially supported by Python-Markdown. Any documentation is
maintained here and all bug reports should be made to the project. If you
have a typical install of Python-Markdown, these extensions are already
available to you using the "Entry Point" name listed in the second column below.

Extension                          | Entry Point    | Dot Notation
---------------------------------- | -------------- | ------------
[Extra]                            | `extra`        | `markdown.extensions.extra`
&nbsp; &nbsp; [Abbreviations]      | `abbr`         | `markdown.extensions.abbr`
&nbsp; &nbsp; [Attribute Lists]    | `attr_list`    | `markdown.extensions.attr_list`
&nbsp; &nbsp; [Definition Lists]   | `def_list`     | `markdown.extensions.def_list`
&nbsp; &nbsp; [Fenced Code Blocks] | `fenced_code`  | `markdown.extensions.fenced_code`
&nbsp; &nbsp; [Footnotes]          | `footnotes`    | `markdown.extensions.footnotes`
&nbsp; &nbsp; [Markdown in HTML]   | `md_in_html`   | `markdown.extensions.md_in_html`
&nbsp; &nbsp; [Tables]             | `tables`       | `markdown.extensions.tables`
[Admonition]                       | `admonition`   | `markdown.extensions.admonition`
[CodeHilite]                       | `codehilite`   | `markdown.extensions.codehilite`
[Legacy Attributes]                | `legacy_attrs`  | `markdown.extensions.legacy_attrs`
[Legacy Emphasis]                  | `legacy_em`    | `markdown.extensions.legacy_em`
[Meta-Data]                        | `meta`         | `markdown.extensions.meta`
[New Line to Break]                | `nl2br`        | `markdown.extensions.nl2br`
[Sane Lists]                       | `sane_lists`   | `markdown.extensions.sane_lists`
[SmartyPants]                      | `smarty`       | `markdown.extensions.smarty`
[Table of Contents]                | `toc`          | `markdown.extensions.toc`
[WikiLinks]                        | `wikilinks`    | `markdown.extensions.wikilinks`

[Extra]: extra.md
[Abbreviations]: abbreviations.md
[Attribute Lists]: attr_list.md
[Definition Lists]: definition_lists.md
[Fenced Code Blocks]: fenced_code_blocks.md
[Footnotes]: footnotes.md
[Tables]: tables.md
[Admonition]: admonition.md
[CodeHilite]: code_hilite.md
[Legacy Attributes]: legacy_attrs.md
[Legacy Emphasis]: legacy_em.md
[Meta-Data]: meta_data.md
[New Line to Break]: nl2br.md
[Markdown in HTML]: md_in_html.md
[Sane Lists]: sane_lists.md
[SmartyPants]: smarty.md
[Table of Contents]: toc.md
[WikiLinks]: wikilinks.md

Third Party Extensions
----------------------

Various individuals and/or organizations have developed extensions which they
have made available to the public. A [list of third party extensions][list]
is maintained on the wiki for your convenience. The Python-Markdown team
offers no official support for these extensions. Please see the developer of
each extension for support.

[list]: https://github.com/Python-Markdown/markdown/wiki/Third-Party-Extensions


title: Legacy Attributes Extension

# Legacy Attributes

## Summary

The Legacy Attributes extension restores Python-Markdown's original attribute
setting syntax. Older versions of Python Markdown (prior to 3.0) included
built-in and undocumented support for defining attributes on elements. Most
users have never made use of the syntax and it has been deprecated in favor of
[Attribute Lists](attr_list.md). This extension restores the legacy behavior for
users who have existing documents which use the syntax.

## Syntax

Attributes are defined by including the following within the element you wish to
assign the attributes to:

```pymd
{@key=value}
```

For example, to define a class to a paragraph:

```pymd
A paragraph with the attribute defined {@class=foo}anywhere within.
```

Which results in the following output:

```pyhtml
<p class="foo">A paragraph with the attribute defined anywhere within.</p>
```

The same applies for inline elements:

```pymd
Some *emphasized{@id=bar}* text.
```

```pyhtml
<p>Some <em id="bar">emphasized</em> text.</p>
```

You can also define attributes in images:

```pymd
![Alt text{@id=baz}](path/to/image.jpg)
```

```pyhtml
<p><img alt="Alt text" id="baz" src="path/to/image.jpg" /></p>
```

## Usage

See [Extensions](index.md) for general extension usage. Use `legacy_attrs` as the
name of the extension.

This extension does not accept any special configuration options.

A trivial example:

```pypython
markdown.markdown(some_text, extensions=['legacy_attrs'])
```


title: Legacy EM Extension

# Legacy EM

## Summary

The Legacy EM extension restores Markdown's original behavior for emphasis and
strong syntax when using underscores.

By default Python-Markdown treats `_connected_words_` intelligently by
recognizing that mid-word underscores should not be used for emphasis. In other
words, by default, that input would result in this output:
`<em>connected_words</em>`.

However, that behavior is not consistent with the original rules or the behavior
of the reference implementation. Therefore, this extension can be used to better
match the reference implementation. With the extension enabled, the above input
would result in this output: `<em>connected</em>words_`.

## Usage

See [Extensions](index.md) for general extension usage. Use `legacy_em` as the
name of the extension.

This extension does not accept any special configuration options.

A trivial example:

```pypython
markdown.markdown(some_text, extensions=['legacy_em'])
```


title: Markdown in HTML Extension

# Markdown in HTML

## Summary

An extension that parses Markdown inside of HTML tags.

## Syntax

By default, Markdown ignores any content within a raw HTML block-level element. With the `md-in-html` extension
enabled, the content of a raw HTML block-level element can be parsed as Markdown by including  a `markdown` attribute
on the opening tag. The `markdown` attribute will be stripped from the output, while all other attributes will be
preserved.

The `markdown` attribute can be assigned one of three values: [`"1"`](#1), [`"block"`](#block), or [`"span"`](#span).

!!! note

    The expressions "block-level" and "span-level" as used in this document refer to an element's designation
    according to the HTML specification. Whereas the `"span"` and `"block"` values assigned to the `markdown`
    attribute refer to the Markdown parser's behavior.

### `markdown="1"` { #1 }

When the `markdown` attribute is set to `"1"`, then the parser will use the default behavior for that specific tag.

The following tags have the `block` behavior by default: `article`, `aside`, `blockquote`, `body`, `colgroup`,
`details`, `div`, `dl`, `fieldset`, `figcaption`, `figure`, `footer`, `form`, `group`, `header`, `hgroup`, `hr`,
`iframe`,  `main`, `map`, `menu`, `nav`, `noscript`, `object`, `ol`, `output`, `progress`, `section`, `table`,
`tbody`, `tfoot`, `thead`, `tr`,  `ul` and `video`.

For example, the following:

```py
<div markdown="1">
This is a *Markdown* Paragraph.
</div>
```

... is rendered as:

```py html
<div>
<p>This is a <em>Markdown</em> Paragraph.</p>
</div>
```

The following tags have the `span` behavior by default: `address`, `dd`, `dt`, `h[1-6]`, `legend`, `li`, `p`, `td`,
and `th`.

For example, the following:

```py
<p markdown="1">
This is not a *Markdown* Paragraph.
</p>
```

... is rendered as:

```py html
<p>
This is not a <em>Markdown</em> Paragraph.
</p>
```

### `markdown="block"` { #block }

When the `markdown` attribute is set to `"block"`, then the parser will force the `block` behavior on the contents of
the element so long as it is one of the `block` or `span` tags.

The content of a `block` element is parsed into block-level content. In other words, the text is rendered as
paragraphs, headers, lists, blockquotes, etc. Any inline syntax within those elements is processed as well.

For example, the following:

```py
<section markdown="block">
# A header.

A *Markdown* paragraph.

* A list item.
* A second list item.

</section>
```

... is rendered as:

```py html
<section>
<h1>A header.</h1>
<p>A <em>Markdown</em> paragraph.</p>
<ul>
<li>A list item.</li>
<li>A second list item.</li>
</ul>
</section>
```

!!! warning

    Forcing elements to be parsed as `block` elements when they are not by default could result in invalid HTML.
    For example, one could force a `<p>` element to be nested within another `<p>` element. In most cases it is
    recommended to use the default behavior of `markdown="1"`. Explicitly setting `markdown="block"` should be
    reserved for advanced users who understand the HTML specification and how browsers parse and render HTML.

### `markdown="span"` { #span }

When the `markdown` attribute is set to `"span"`, then the parser will force the `span` behavior on the contents
of the element so long as it is one of the `block` or `span` tags.

The content of a `span` element is not parsed into block-level content. In other words, the content will not be
rendered as paragraphs, headers, etc. Only inline syntax will be rendered, such as links, strong, emphasis, etc.

For example, the following:

```py
<div markdown="span">
# *Not* a header
</div>
```

... is rendered as:

```py html
<div>
# <em>Not</em> a header
</div>
```

### Ignored Elements

The following tags are always ignored, regardless of any `markdown` attribute: `canvas`, `math`, `option`, `pre`,
`script`, `style`, and `textarea`. All other raw HTML tags are treated as span-level tags and are not affected by this
extension.

### Nesting

When nesting multiple levels of raw HTML elements, a `markdown` attribute must be defined for each block-level
element. For any block-level element which does not have a `markdown` attribute, everything inside that element is
ignored, including child elements with `markdown` attributes.

For example, the following:

```py
<article id="my-article" markdown="1">
# Article Title

A Markdown paragraph.

<section id="section-1" markdown="1">
## Section 1 Title

<p>Custom raw **HTML** which gets ignored.</p>

</section>

<section id="section-2" markdown="1">
## Section 2 Title

<p markdown="1">**Markdown** content.</p>

</section>

</article>
```

... is rendered as:

```pyhtml
<article id="my-article">
<h1>Article Title</h1>
<p>A Markdown paragraph.</p>
<section id="section-1">
<h2>Section 1 Title</h2>
<p>Custom raw **HTML** which gets ignored.</p>
</section>
<section id="section-2">
<h2>Section 2 Title</h2>
<p><strong>Markdown</strong> content.</p>
</section>
</article>
```

When the value of an element's `markdown` attribute is more permissive that its parent, then the parent's stricter
behavior is enforced. For example, a `block` element nested within a `span` element will be parsed using the `span`
behavior. However, if the value of an element's `markdown` attribute is the same as, or more restrictive than, its
parent, the the child element's behavior is observed. For example, a `block` element may contain either `block`
elements or `span` elements as children and each element will be parsed using the specified behavior.

### Tag Normalization

While the default behavior is for Markdown to not alter raw HTML, as this extension is parsing the content of raw HTML elements, it will do some normalization of the tags of block-level elements. For example, the following raw HTML:

```py
<div markdown="1">
<p markdown="1">A Markdown paragraph with *no* closing tag.
<p>A raw paragraph with *no* closing tag.
</div>
```

... is rendered as:

```py html
<div>
<p>A Markdown paragraph with <em>no</em> closing tag.
</p>
<p>A raw paragraph with *no* closing tag.
</p>
</div>
```

Notice that the parser properly recognizes that an unclosed  `<p>` tag ends when another `<p>` tag begins or when the
parent element ends. In both cases, a closing `</p>` was added to the end of the element, regardless of whether a
`markdown` attribute was assigned to the element.

To avoid any normalization, an element must not be a descendant of any block-level element which has a `markdown`
attribute defined.

!!! warning

    The normalization behavior is only documented here so that document authors are not surprised when their carefully
    crafted raw HTML is altered by Markdown. This extension should not be relied on to normalize and generate valid
    HTML. For the best results, always include valid raw HTML (with both opening and closing tags) in your Markdown
    documents.

## Usage

From the Python interpreter:

```py pycon
>>> import markdown
>>> html = markdown.markdown(text, extensions=['md_in_html'])
```


title: Meta-Data Extension

Meta-Data
=========

Summary
-------

The Meta-Data extension adds a syntax for defining meta-data about a document.
It is inspired by and follows the syntax of [MultiMarkdown][]. Currently,
this extension does not use the meta-data in any way, but simply provides it as
a `Meta` attribute of a Markdown instance for use by other extensions or
directly by your python code.

This extension is included in the standard Markdown library.

[MultiMarkdown]: https://fletcherpenney.net/multimarkdown/#metadata

Syntax
------

Meta-data consists of a series of keywords and values defined at the beginning
of a markdown document like this:

```pymd
Title:   My Document
Summary: A brief description of my document.
Authors: Waylan Limberg
         John Doe
Date:    October 2, 2007
blank-value:
base_url: http://example.com

This is the first paragraph of the document.
```

The keywords are case-insensitive and may consist of letters, numbers,
underscores and dashes and must end with a colon. The values consist of
anything following the colon on the line and may even be blank.

If a line is indented by 4 or more spaces, that line is assumed to be an
additional line of the value for the previous keyword. A keyword may have as
many lines as desired.

The first blank line ends all meta-data for the document. Therefore, the first
line of a document must not be blank.

Alternatively, You may use YAML style delimiters to mark the start and/or end
of your meta-data. When doing so, the first line of your document must be `---`.
The meta-data ends at the first blank line or the first line containing an end
deliminator (either `---` or `...`), whichever comes first. Even though YAML
delimiters are supported, meta-data is not parsed as YAML.

All meta-data is stripped from the document prior to any further processing
by Markdown.

Usage
-----

See [Extensions](index.md) for general extension usage. Use `meta` as the name
of the extension.

A trivial example:

```pypython
markdown.markdown(some_text, extensions=['meta'])
```

Accessing the Meta-Data
-----------------------

The meta-data is made available as a python Dict in the `Meta` attribute of an
instance of the Markdown class. For example, using the above document:

```pypycon
>>> md = markdown.Markdown(extensions = ['meta'])
>>> html = md.convert(text)
>>> # Meta-data has been stripped from output
>>> print html
<p>This is the first paragraph of the document.</p>

>>> # View meta-data
>>> print md.Meta
{
'title' : ['My Document'],
'summary' : ['A brief description of my document.'],
'authors' : ['Waylan Limberg', 'John Doe'],
'date' : ['October 2, 2007'],
'blank-value' : [''],
'base_url' : ['http://example.com']
}
```

Note that the keys are all lowercase and the values consist of a list of
strings where each item is one line for that key. This way, one could preserve
line breaks if desired. Or the items could be joined where appropriate. No
assumptions are made regarding the data. It is simply passed as found to the
`Meta` attribute.

Perhaps the meta-data could be passed into a template system, or used by
various Markdown extensions. The possibilities are left to the imagination of
the developer.

Compatible Extensions
---------------------

The following extensions are currently known to work with the Meta-Data
extension. The keywords they are known to support are also listed.

* [WikiLinks](wikilinks.md)
    * `wiki_base_url`
    * `wiki_end_url`
    * `wiki_html_class`


title: New Line to Break Extension

New-Line-to-Break Extension
===========================

Summary
-------

The New-Line-to-Break (`nl2br`) Extension will cause newlines to be treated as
hard breaks; like StackOverflow and [GitHub][] flavored Markdown do.

[Github]: https://github.github.com/github-flavored-markdown/

Example
-------

```pypycon
>>> import markdown
>>> text = """
... Line 1
... Line 2
... """
>>> html = markdown.markdown(text, extensions=['nl2br'])
>>> print html
<p>Line 1<br />
Line 2</p>
```

Usage
-----

See [Extensions](index.md) for general extension usage. Use `nl2br` as the name
of the extension.

This extension does not accept any special configuration options.

A trivial example:

```pypython
markdown.markdown(some_text, extensions=['nl2br'])
```


title: Sane Lists Extension

Sane Lists
==========

Summary
-------

The Sane Lists extension alters the behavior of the Markdown List syntax
to be less surprising.

This extension is included in the standard Markdown library.

Syntax
------

Sane Lists do not allow the mixing of list types. In other words, an ordered
list will not continue when an unordered list item is encountered and
vice versa. For example:

```pymd
1. Ordered item 1
2. Ordered item 2

* Unordered item 1
* Unordered item 2
```

will result in the following output:

```pyhtml
<ol>
  <li>Ordered item 1</li>
  <li>Ordered item 2</li>
</ol>

<ul>
  <li>Unordered item 1</li>
  <li>Unordered item 2</li>
</ul>
```

Whereas the default Markdown behavior would be to generate an unordered list.

Note that, unlike the default Markdown behavior, if a blank line is not
included between list items, the different list type is ignored completely.
This corresponds to the behavior of paragraphs. For example:

```pymd
A Paragraph.
* Not a list item.

1. Ordered list item.
* Not a separate list item.
```

With this extension the above will result in the following output:

```pyhtml
<p>A Paragraph.
* Not a list item.</p>

<ol>
  <li>Ordered list item.
  * Not a separate list item.</li>
</ol>
```

Sane lists also recognize the number used in ordered lists. Given the following
list:

```pymd
4. Apples
5. Oranges
6. Pears
```

By default markdown will ignore the fact that the first line started
with item number "4" and the HTML list will start with a number "1".
This extension will result in the following HTML output:

```pyhtml
<ol start="4">
  <li>Apples</li>
  <li>Oranges</li>
  <li>Pears</li>
</ol>
```

In all other ways, Sane Lists should behave as normal Markdown lists.

Usage
-----

See [Extensions](index.md) for general extension usage. Use `sane_lists` as the
name of the extension.

This extension does not accept any special configuration options.

A trivial example:

```pypython
markdown.markdown(some_text, extensions=['sane_lists'])
```


title: SmartyPants Extension

SmartyPants
===========

Summary
-------

The SmartyPants extension converts ASCII dashes, quotes and ellipses to
their HTML entity equivalents.

ASCII symbol | Replacements    | HTML Entities       | Substitution Keys
------------ | --------------- | ------------------- | ----------------------------------------
`'`          | &lsquo; &rsquo; | `&lsquo;` `&rsquo;` | `'left-single-quote'`, `'right-single-quote'`
`"`          | &ldquo; &rdquo; | `&ldquo;` `&rdquo;` | `'left-double-quote'`, `'right-double-quote'`
`<< >>`      | &laquo; &raquo; | `&laquo;` `&raquo;` | `'left-angle-quote'`, `'right-angle-quote'`
`...`        | &hellip;        | `&hellip;`          | `'ellipsis'`
`--`         | &ndash;         | `&ndash;`           | `'ndash'`
`---`        | &mdash;         | `&mdash;`           | `'mdash'`

Using the configuration option 'substitutions' you can overwrite the
default substitutions. Just pass a dict mapping (a subset of) the
keys to the substitution strings.

For example, one might use the following configuration to get correct quotes for
the German language:

```pypython
extension_configs = {
    'smarty': {
        'substitutions': {
            'left-single-quote': '&sbquo;', # sb is not a typo!
            'right-single-quote': '&lsquo;',
            'left-double-quote': '&bdquo;',
            'right-double-quote': '&ldquo;'
        }
    }
}
```

!!! note
    This extension re-implements the Python [SmartyPants]
    library by integrating it into the markdown parser.
    While this does not provide any additional features,
    it does offer a few advantages. Notably, it will not
    try to work on highlighted code blocks (using the
    [CodeHilite] Extension) like the third party library
    has been known to do.

[SmartyPants]: https://pythonhosted.org/smartypants/
[CodeHilite]: code_hilite.md

Usage
-----

See [Extensions](index.md) for general extension usage. Use `smarty` as the
name of the extension.

See the [Library Reference](../reference.md#extensions) for information about
configuring extensions.

The following options are provided to configure the output:

Option                | Default value | Description
------                | ------------- | -----------
`smart_dashes`        | `True`        | whether to convert dashes
`smart_quotes`        | `True`        | whether to convert straight quotes
`smart_angled_quotes` | `False`       | whether to convert angled quotes
`smart_ellipses`      | `True`        | whether to convert ellipses
`substitutions`       | `{}`          | overwrite default substitutions

A trivial example:

```pypython
markdown.markdown(some_text, extensions=['smarty'])
```

Further reading
---------------

SmartyPants extension is based on the original SmartyPants implementation
by John Gruber. Please read its [documentation][1] for details.

[1]: https://daringfireball.net/projects/smartypants/


title: Tables Extension

Tables
======

Summary
-------

The Tables extension adds the ability to create tables in Markdown documents.

This extension is included in the standard Markdown library.

Syntax
------

Tables are defined using the syntax established in [PHP Markdown Extra][php].

[php]: http://www.michelf.com/projects/php-markdown/extra/#table

Thus, the following text (taken from the above referenced PHP documentation):

```pymd
First Header  | Second Header
------------- | -------------
Content Cell  | Content Cell
Content Cell  | Content Cell
```

will be rendered as:

```pyhtml
<table>
  <thead>
    <tr>
      <th>First Header</th>
      <th>Second Header</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Content Cell</td>
      <td>Content Cell</td>
    </tr>
    <tr>
      <td>Content Cell</td>
      <td>Content Cell</td>
    </tr>
  </tbody>
</table>
```

!!! seealso "See Also"
    The [Attribute Lists](./attr_list.md) extension includes support for defining attributes on table cells.

Usage
-----

See [Extensions](index.md) for general extension usage. Use `tables` as the
name of the extension.

See the [Library Reference](../reference.md#extensions) for information about
configuring extensions.

The following options are provided to change the default behavior:

* **`use_align_attribute`**: Set to `True` to use `align` instead of an appropriate `style` attribute

    Default: `'False'`


A trivial example:

```pypython
markdown.markdown(some_text, extensions=['tables'])
```

### Examples

For an example, let us suppose that alignment should be controlled by the legacy `align`
attribute.

```pypycon
>>> from markdown.extensions.tables import TableExtension
>>> html = markdown.markdown(text,
...                 extensions=[TableExtension(use_align_attribute=True)]
... )
```


title: Table of Contents Extension

Table of Contents
=================

Summary
-------

The Table of Contents extension generates a Table of Contents from a Markdown
document and adds it into the resulting HTML document.

This extension is included in the standard Markdown library.

Syntax
------

By default, all headers will automatically have unique `id` attributes
generated based upon the text of the header. Note this example, in which all
three headers would have the same `id`:

```pymd
#Header
#Header
#Header
```

Results in:

```pyhtml
<h1 id="header">Header</h1>
<h1 id="header_1">Header</h1>
<h1 id="header_2">Header</h1>
```

Place a marker in the document where you would like the Table of Contents to
appear. Then, a nested list of all the headers in the document will replace the
marker. The marker defaults to `[TOC]` so the following document:

```pymd
[TOC]

# Header 1

## Header 2
```

would generate the following output:

```pyhtml
<div class="toc">
  <ul>
    <li><a href="#header-1">Header 1</a></li>
      <ul>
        <li><a href="#header-2">Header 2</a></li>
      </ul>
  </ul>
</div>
<h1 id="header-1">Header 1</h1>
<h2 id="header-2">Header 2</h2>
```

Regardless of whether a `marker` is found in the document (or disabled), the
Table of Contents is available as an attribute (`toc`) on the Markdown class.
This allows one to insert the Table of Contents elsewhere in their page
template. For example:

```pypycon
>>> md = markdown.Markdown(extensions=['toc'])
>>> html = md.convert(text)
>>> page = render_some_template(context={'body': html, 'toc': md.toc})
```

The `toc_tokens` attribute is also available on the Markdown class and contains
a nested list of dict objects. For example, the above document would result in
the following object at `md.toc_tokens`:

```pypython
[
    {
        'level': 1,
        'id': 'header-1',
        'name': 'Header 1',
        'children': [
            {'level': 2, 'id': 'header-2', 'name': 'Header 2', 'children':[]}
        ]
    }
]
```

Note that the `level` refers to the `hn` level. In other words, `<h1>` is level
`1` and `<h2>` is level `2`, etc. Be aware that improperly nested levels in the
input may result in odd nesting of the output.

### Custom Labels

In most cases, the text label in the Table of Contents should match the text of
the header. However, occasionally that is not desirable. In that case, if this
extension is used in conjunction with the [Attribute Lists Extension] and a
`data-toc-label` attribute is defined on the header, then the contents of that
attribute will be used as the text label for the item in the Table of Contents.
For example, the following Markdown:

[Attribute Lists Extension]: attr_list.md

```pymd
[TOC]

# Functions

## `markdown.markdown(text [, **kwargs])` { #markdown data-toc-label='markdown.markdown' }
```
would generate the following output:

```pyhtml
<div class="toc">
  <ul>
    <li><a href="#functions">Functions</a></li>
      <ul>
        <li><a href="#markdown">markdown.markdown</a></li>
      </ul>
  </ul>
</div>
<h1 id="functions">Functions</h1>
<h2 id="markdown"><code>markdown.markdown(text [, **kwargs])</code></h2>
```

Notice that the text in the Table of Contents is much cleaner and easier to read
in the context of a Table of Contents. The `data-toc-label` is not included in
the HTML header element. Also note that the ID was manually defined in the
attribute list to provide a cleaner URL when linking to the header. If the ID is
not manually defined, it is always derived from the text of the header, never
from the `data-toc-label` attribute.

Usage
-----

See [Extensions](index.md) for general extension usage. Use `toc` as the name
of the extension.

See the [Library Reference](../reference.md#extensions) for information about
configuring extensions.

The following options are provided to configure the output:

* **`marker`**:
    Text to find and replace with the Table of Contents. Defaults to `[TOC]`.

    Set to an empty string to disable searching for a marker, which may save
    some time, especially on long documents.

* **`title`**:
    Title to insert in the Table of Contents' `<div>`. Defaults to `None`.

* **`title_class`**:
    CSS class used for the title contained in the Table of Contents. Defaults to `toctitle`.

* **`toc_class`**:
    CSS class(es) used for the `<div>` containing the Table of Contents. Defaults to `toc`.

* **`anchorlink`**:
    Set to `True` to cause all headers to link to themselves. Default is `False`.

* **`anchorlink_class`**:
    CSS class(es) used for the link. Defaults to `toclink`.

* **`permalink`**:
    Set to `True` or a string to generate permanent links at the end of each header.
    Useful with Sphinx style sheets.

    When set to `True` the paragraph symbol (&para; or "`&para;`") is used as
    the link text. When set to a string, the provided string is used as the link
    text.

* **`permalink_class`**:
    CSS class(es) used for the link. Defaults to `headerlink`.

* **`permalink_title`**:
    Title attribute of the permanent link. Defaults to `Permanent link`.

* **`permalink_leading`**:
    Set to `True` if the extension should generate leading permanent links.
    Default is `False`.

    Leading permanent links are placed at the start of the header tag,
    before any header content. The default `permalink` behavior (when
    `permalink_leading` is unset or set to `False`) creates trailing
    permanent links, which are placed at the end of the header content.

* **`baselevel`**:
    Base level for headers. Defaults to `1`.

    The `baselevel` setting allows the header levels to be automatically
    adjusted to fit within the hierarchy of your HTML templates. For example,
    suppose the Markdown text for a page should not contain any headers higher
    than level 3 (`<h3>`). The following will accomplish that:

        :::pycon
        >>>  text = '''
        ... #Some Header
        ... ## Next Level'''
        >>> from markdown.extensions.toc import TocExtension
        >>> html = markdown.markdown(text, extensions=[TocExtension(baselevel=3)])
        >>> print html
        <h3 id="some_header">Some Header</h3>
        <h4 id="next_level">Next Level</h4>'

* **`slugify`**:
    Callable to generate anchors.

    Default: `markdown.extensions.toc.slugify`

    In order to use a different algorithm to define the id attributes, define  and
    pass in a callable which takes the following two arguments:

    * `value`: The string to slugify.
    * `separator`: The Word Separator.

    The callable must return a string appropriate for use in HTML `id` attributes.

    An alternate version of the default callable supporting Unicode strings is also
    provided as `markdown.extensions.toc.slugify_unicode`.

* **`separator`**:
    Word separator. Character which replaces white space in id. Defaults to "`-`".

* **`toc_depth`**
    Define the range of section levels to include in the Table of Contents.
    A single integer (`b`) defines the bottom section level (`<h1>..<hb>`) only.
    A string consisting of two digits separated by a hyphen in between (`"2-5"`),
    define the top (`t`) and the bottom (`b`) (`<ht>..<hb>`). Defaults to `6` (bottom).

    When used with conjunction with `baselevel`, this parameter will not
    take the fitted hierarchy from `baselevel` into account. That is, if
    both `toc_depth` and `baselevel` are `3`, then only the highest level
    will be present in the table. If you set `baselevel` to `3` and
    `toc_depth` to `"2-6"`, the *first* headline will be `<h3>` and so still
    included in the Table of Contents. To exclude this first level, you
    have to set `toc_depth` to `"4-6"`.

A trivial example:

```pypython
markdown.markdown(some_text, extensions=['toc'])
```
