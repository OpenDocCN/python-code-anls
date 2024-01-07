# `markdown\tests\test_extensions.py`

```

# 导入 unittest 和 markdown 模块
import unittest
import markdown

# 定义一个测试类，继承自 unittest.TestCase
class TestCaseWithAssertStartsWith(unittest.TestCase):

    # 定义一个自定义的断言方法，用于检查字符串是否以指定前缀开头
    def assertStartsWith(self, expectedPrefix, text, msg=None):
        if not text.startswith(expectedPrefix):
            if len(expectedPrefix) + 5 < len(text):
                text = text[:len(expectedPrefix) + 5] + '...'
            standardMsg = '{} not found at the start of {}'.format(repr(expectedPrefix),
                                                                   repr(text))
            self.fail(self._formatMessage(msg, standardMsg))

# 定义一个测试类，用于测试 markdown.extensions.Extension
class TestExtensionClass(unittest.TestCase):

    # 在测试之前设置环境
    def setUp(self):
        # 定义一个测试用的 markdown 扩展类
        class TestExtension(markdown.extensions.Extension):
            # 定义配置信息
            config = {
                'foo': ['bar', 'Description of foo'],
                'bar': ['baz', 'Description of bar']
            }

        # 实例化测试用的扩展类
        self.ext = TestExtension()
        self.ExtKlass = TestExtension

    # 测试获取配置信息
    def testGetConfig(self):
        self.assertEqual(self.ext.getConfig('foo'), 'bar')

    # 测试获取默认配置信息
    def testGetConfigDefault(self):
        self.assertEqual(self.ext.getConfig('baz'), '')
        self.assertEqual(self.ext.getConfig('baz', default='missing'), 'missing')

    # 测试获取所有配置信息
    def testGetConfigs(self):
        self.assertEqual(self.ext.getConfigs(), {'foo': 'bar', 'bar': 'baz'})

    # 测试获取配置信息的描述
    def testGetConfigInfo(self):
        self.assertEqual(
            dict(self.ext.getConfigInfo()),
            dict([
                ('foo', 'Description of foo'),
                ('bar', 'Description of bar')
            ])
        )

    # 测试设置配置信息
    def testSetConfig(self):
        self.ext.setConfig('foo', 'baz')
        self.assertEqual(self.ext.getConfigs(), {'foo': 'baz', 'bar': 'baz'})

    # 测试设置错误的配置信息
    def testSetConfigWithBadKey(self):
        # 期望设置错误的配置信息会引发 KeyError
        self.assertRaises(KeyError, self.ext.setConfig, 'bad', 'baz')

    # 测试在初始化时使用配置信息
    def testConfigAsKwargsOnInit(self):
        ext = self.ExtKlass(foo='baz', bar='blah')
        self.assertEqual(ext.getConfigs(), {'foo': 'baz', 'bar': 'blah'})

# 定义一个测试类，用于测试 abbr 扩展
class TestAbbr(unittest.TestCase):

    # 在测试之前设置环境
    def setUp(self):
        # 实例化一个 markdown.Markdown 对象，指定使用 abbr 扩展
        self.md = markdown.Markdown(extensions=['abbr'])

    # 测试简单的缩写
    def testSimpleAbbr(self):
        text = 'Some text with an ABBR and a REF. Ignore REFERENCE and ref.' + \
               '\n\n*[ABBR]: Abbreviation\n' + \
               '*[REF]: Abbreviation Reference'
        self.assertEqual(
            self.md.convert(text),
            '<p>Some text with an <abbr title="Abbreviation">ABBR</abbr> '
            'and a <abbr title="Abbreviation Reference">REF</abbr>. Ignore '
            'REFERENCE and ref.</p>'
        )

    # 测试嵌套的缩写
    def testNestedAbbr(self):
        text = '[ABBR](/foo) and _ABBR_\n\n' + \
               '*[ABBR]: Abbreviation'
        self.assertEqual(
            self.md.convert(text),
            '<p><a href="/foo"><abbr title="Abbreviation">ABBR</abbr></a> '
            'and <em><abbr title="Abbreviation">ABBR</abbr></em></p>'
        )

# 定义一个测试类，用于测试 MetaData 扩展
class TestMetaData(unittest.TestCase):

    # 在测试之前设置环境
    def setUp(self):
        # 实例化一个 markdown.Markdown 对象，指定使用 meta 扩展
        self.md = markdown.Markdown(extensions=['meta'])

    # 测试基本的元数据
    def testBasicMetaData(self):
        text = '''Title: A Test Doc.
Author: Waylan Limberg
        John Doe
Blank_Data:

The body. This is paragraph one.'''
        self.assertEqual(
            self.md.convert(text),
            '<p>The body. This is paragraph one.</p>'
        )
        self.assertEqual(
            self.md.Meta, {
                'author': ['Waylan Limberg', 'John Doe'],
                'blank_data': [''],
                'title': ['A Test Doc.']
            }
        )

    # 测试指定简单 YAML 格式的元数据
    def testYamlMetaData(self):
        text = '''---
Title: A Test Doc.
Author: [Waylan Limberg, John Doe]
Blank_Data:
---

The body. This is paragraph one.'''
        self.assertEqual(
            self.md.convert(text),
            '<p>The body. This is paragraph one.</p>'
        )
        self.assertEqual(
            self.md.Meta, {
                'author': ['[Waylan Limberg, John Doe]'],
                'blank_data': [''],
                'title': ['A Test Doc.']
            }
        )

    # 测试没有元数据的文档
    def testMissingMetaData(self):
        text = '    Some Code - not extra lines of meta data.'
        self.assertEqual(
            self.md.convert(text),
            '<pre><code>Some Code - not extra lines of meta data.\n'
            '</code></pre>'
        )
        self.assertEqual(self.md.Meta, {})

    # 测试没有换行符的元数据
    def testMetaDataWithoutNewline(self):
        text = 'title: No newline'
        self.assertEqual(self.md.convert(text), '')
        self.assertEqual(self.md.Meta, {'title': ['No newline']})

    # 测试重置元数据
    def testMetaDataReset(self):
        text = '''Title: A Test Doc.
Author: Waylan Limberg
        John Doe
Blank_Data:

The body. This is paragraph one.'''
        self.md.convert(text)

        self.md.reset()
        self.assertEqual(self.md.Meta, {})

# 定义一个测试类，用于测试 Wikilinks 扩展
class TestWikiLinks(unittest.TestCase):

    # 在测试之前设置环境
    def setUp(self):
        # 实例化一个 markdown.Markdown 对象，指定使用 wikilinks 扩展
        self.md = markdown.Markdown(extensions=['wikilinks'])
        self.text = "Some text with a [[WikiLink]]."

    # 测试基本的 Wikilinks
    def testBasicWikilinks(self):
        self.assertEqual(
            self.md.convert(self.text),
            '<p>Some text with a '
            '<a class="wikilink" href="/WikiLink/">WikiLink</a>.</p>'
        )

    # 测试 Wikilinks 中的空白字符
    def testWikilinkWhitespace(self):
        self.assertEqual(
            self.md.convert('[[ foo bar_baz ]]'),
            '<p><a class="wikilink" href="/foo_bar_baz/">foo bar_baz</a></p>'
        )
        self.assertEqual(
            self.md.convert('foo [[ ]] bar'),
            '<p>foo  bar</p>'
        )

    # 测试简单的设置
    def testSimpleSettings(self):
        self.assertEqual(markdown.markdown(
            self.text, extensions=[
                markdown.extensions.wikilinks.WikiLinkExtension(
                    base_url='/wiki/',
                    end_url='.html',
                    html_class='foo')
                ]
            ),
            '<p>Some text with a '
            '<a class="foo" href="/wiki/WikiLink.html">WikiLink</a>.</p>')

    # 测试复杂的设置
    def testComplexSettings(self):
        md = markdown.Markdown(
            extensions=['wikilinks'],
            extension_configs={
                'wikilinks': [
                    ('base_url', 'http://example.com/'),
                    ('end_url', '.html'),
                    ('html_class', '')
                ]
            },
            safe_mode=True
        )
        self.assertEqual(
            md.convert(self.text),
            '<p>Some text with a '
            '<a href="http://example.com/WikiLink.html">WikiLink</a>.</p>'
        )

    # 测试带有 Wikilinks 扩展的 MetaData
    def testWikilinksMetaData(self):
        text = """wiki_base_url: http://example.com/
wiki_end_url:   .html
wiki_html_class:

Some text with a [[WikiLink]]."""
        md = markdown.Markdown(extensions=['meta', 'wikilinks'])
        self.assertEqual(
            md.convert(text),
            '<p>Some text with a '
            '<a href="http://example.com/WikiLink.html">WikiLink</a>.</p>'
        )

        # MetaData 不应该传递到下一个文档：
        self.assertEqual(
            md.convert("No [[MetaData]] here."),
            '<p>No <a class="wikilink" href="/MetaData/">MetaData</a> '
            'here.</p>'
        )

    # 测试自定义 URL 构建器
    def testURLCallback(self):
        from markdown.extensions.wikilinks import WikiLinkExtension

        def my_url_builder(label, base, end):
            return '/bar/'

        md = markdown.Markdown(extensions=[WikiLinkExtension(build_url=my_url_builder)])
        self.assertEqual(
            md.convert('[[foo]]'),
            '<p><a class="wikilink" href="/bar/">foo</a></p>'
        )

# 定义一个测试类，用于测试 Admonition 扩展
class TestAdmonition(unittest.TestCase):

    # 在测试之前设置环境
    def setUp(self):
        # 实例化一个 markdown.Markdown 对象，指定使用 admonition 扩展
        self.md = markdown.Markdown(extensions=['admonition'])

    # 测试正则表达式
    def testRE(self):
        RE = self.md.parser.blockprocessors['admonition'].RE
        tests = [
            ('!!! note', ('note', None)),
            ('!!! note "Please Note"', ('note', 'Please Note')),
            ('!!! note ""', ('note', '')),
        ]
        for test, expected in tests:
            self.assertEqual(RE.match(test).groups(), expected)

# 定义一个测试类，用于测试 Smarty 扩展
class TestSmarty(unittest.TestCase):

    # 在测试之前设置环境
    def setUp(self):
        config = {
            'smarty': [
                ('smart_angled_quotes', True),
                ('substitutions', {
                    'ndash': '\u2013',
                    'mdash': '\u2014',
                    'ellipsis': '\u2026',
                    'left-single-quote': '&sbquo;',  # `sb` is not a typo!
                    'right-single-quote': '&lsquo;',
                    'left-double-quote': '&bdquo;',
                    'right-double-quote': '&ldquo;',
                    'left-angle-quote': '[',
                    'right-angle-quote': ']',
                }),
            ]
        }
        # 实例化一个 markdown.Markdown 对象，指定使用 smarty 扩展，并传入配置信息
        self.md = markdown.Markdown(
            extensions=['smarty'],
            extension_configs=config
        )

    # 测试自定义替换
    def testCustomSubstitutions(self):
        text = """<< The "Unicode char of the year 2014"
is the 'mdash': ---
Must not be confused with 'ndash'  (--) ... >>
"""
        correct = """<p>[ The &bdquo;Unicode char of the year 2014&ldquo;
is the &sbquo;mdash&lsquo;: \u2014
Must not be confused with &sbquo;ndash&lsquo;  (\u2013) \u2026 ]</p>"""
        self.assertEqual(self.md.convert(text), correct)

```