# BasicComputerGames源码解析 8

# `00_Common/dotnet/Games.Common.Test/IO/TokenTests.cs`

This is a unit test for the `Token` class in the `Games.Common.IO` namespace. The `Token` class is used to represent a simple file input stream that has a specified value, whether it is a string or a number, and optionally provides information about whether the value is a number or not.

The `TokenTestCases` property is a collection of test cases that are used to validate the properties of the `Token` class. The tests cover different scenarios, such as a string value, a number value, a non-number value, and a float number.

The `Ctor_PopulatesProperties`测试用例 checks that the properties of the `Token` class are set correctly when a new instance of the class is created. It does this by creating a new instance of the class with the specified value and then compares the properties of the new instance to an expected object.


```
using FluentAssertions;
using Xunit;

namespace Games.Common.IO;

public class TokenTests
{
    [Theory]
    [MemberData(nameof(TokenTestCases))]
    public void Ctor_PopulatesProperties(string value, bool isNumber, float number)
    {
        var expected = new { String = value, IsNumber = isNumber, Number = number };

        var token = new Token(value);

        token.Should().BeEquivalentTo(expected);
    }

    public static TheoryData<string, bool, float> TokenTestCases() => new()
    {
        { "", false, float.NaN },
        { "abcde", false, float.NaN },
        { "123  ", true, 123 },
        { "+42  ", true, 42 },
        { "-42  ", true, -42 },
        { "+3.14159  ", true, 3.14159F },
        { "-3.14159  ", true, -3.14159F },
        { "   123", false, float.NaN },
        { "1.2e4", true, 12000 },
        { "2.3e-5", true, 0.000023F },
        { "1e100", true, float.MaxValue },
        { "-1E100", true, float.MinValue },
        { "1E-100", true, 0 },
        { "-1e-100", true, 0 },
        { "100abc", true, 100 },
        { "1,2,3", true, 1 },
        { "42,a,b", true, 42 },
        { "1.2.3", true, 1.2F },
        { "12e.5", false, float.NaN },
        { "12e0.5", true, 12 }
    };
}

```

# `00_Common/dotnet/Games.Common.Test/IO/TextIOTests/NumberFormatTests.cs`

This is a test code for the MediaWare谷控制器，目的是在WriteFloatTestCases和WriteLineIntTestCases方法中编写测试用例，测试能否正确生成指定格式的理论数据。

首先，我们可以看到这两个方法都使用了命名参数，即WriteFloatTestCases和WriteLineIntTestCases，这意味着它们可以接受一个名称参数和一个基本字符串参数。

然后，在WriteFloatTestCases方法中，我们使用了一个包含多个不同浮点数的对象，这个对象代表了测试要覆盖的不同的数值类型。

接下来，在WriteLineIntTestCases方法中，我们同样使用了一个包含多个不同整数的对象，这个对象代表了测试要覆盖的不同整数类型。

在这些测试用例中，我们都是通过创建一个StringWriter对象来写入测试用例的字符串，这个StringWriter对象在后面会被用来检查生成的理论数据是否正确。

最后，在WriteFloatTestCases和WriteLineIntTestCases方法的代码中，我们都是通过将值和基本字符串参数传递给IO.Write函数来写入测试用例的值。这个Write函数会在生成的字符串中包含我们传递的值和基本字符串。

总的来说，这两个方法在写入测试用例的字符串时都使用了相同的逻辑，因此它们的实现是相似的。


```
using System;
using System.IO;
using FluentAssertions;
using Xunit;

namespace Games.Common.IO.TextIOTests;

public class NumberFormatTests
{
    [Theory]
    [MemberData(nameof(WriteFloatTestCases))]
    public void Write_Float_FormatsNumberSameAsBasic(float value, string basicString)
    {
        var outputWriter = new StringWriter();
        var io = new TextIO(new StringReader(""), outputWriter);

        io.Write(value);

        outputWriter.ToString().Should().BeEquivalentTo(basicString);
    }

    [Theory]
    [MemberData(nameof(WriteFloatTestCases))]
    public void WriteLine_Float_FormatsNumberSameAsBasic(float value, string basicString)
    {
        var outputWriter = new StringWriter();
        var io = new TextIO(new StringReader(""), outputWriter);

        io.WriteLine(value);

        outputWriter.ToString().Should().BeEquivalentTo(basicString + Environment.NewLine);
    }

    public static TheoryData<float, string> WriteFloatTestCases()
        => new()
        {
            { 1000F, " 1000 " },
            { 3.1415927F, " 3.1415927 " },
            { 1F, " 1 " },
            { 0F, " 0 " },
            { -1F, "-1 " },
            { -3.1415927F, "-3.1415927 " },
            { -1000F, "-1000 " },
        };

    [Theory]
    [MemberData(nameof(WriteIntTestCases))]
    public void Write_Int_FormatsNumberSameAsBasic(int value, string basicString)
    {
        var outputWriter = new StringWriter();
        var io = new TextIO(new StringReader(""), outputWriter);

        io.Write(value);

        outputWriter.ToString().Should().BeEquivalentTo(basicString);
    }

    [Theory]
    [MemberData(nameof(WriteIntTestCases))]
    public void WriteLine_Int_FormatsNumberSameAsBasic(int value, string basicString)
    {
        var outputWriter = new StringWriter();
        var io = new TextIO(new StringReader(""), outputWriter);

        io.WriteLine(value);

        outputWriter.ToString().Should().BeEquivalentTo(basicString + Environment.NewLine);
    }

    public static TheoryData<int, string> WriteIntTestCases()
        => new()
        {
            { 1000, " 1000 " },
            { 1, " 1 " },
            { 0, " 0 " },
            { -1, "-1 " },
            { -1000, "-1000 " },
        };
}

```

# `00_Common/dotnet/Games.Common.Test/IO/TextIOTests/ReadMethodTests.cs`

该代码是一个测试类，它包括对几个头文件的引用，这些头文件定义了两个字符串、两个浮点数和一个四元组类型的类。

首先，该代码使用System命名空间中的System类，该命名空间包含许多有用的类和函数，如System.Array、System.Collections.Generic、System.IO、System.Linq等。

接下来，该代码使用System.Collections.Generic命名空间中的系统类，定义了一个名为TwoStrings的类，该类包含两个字符串类型。

然后，该代码使用System.Collections.Generic命名空间中的系统类，定义了一个名为TwoNumbers的类，该类包含两个浮点数类型。

接着，该代码使用System.Collections.Generic命名空间中的系统类，定义了一个名为ThreeNumbers的类，该类包含三个浮点数类型。

最后，该代码使用System.Collections.Generic命名空间中的系统类，定义了一个名为FourNumbers的类，该类包含四个浮点数类型。

该代码的其他部分包括对AssertXunit、Assertion和AssertReflection介质的引用，这些介质用于断言属性和方法的行为。


```
using System;
using System.Collections.Generic;
using System.IO;
using FluentAssertions;
using FluentAssertions.Execution;
using Xunit;

using TwoStrings = System.ValueTuple<string, string>;
using TwoNumbers = System.ValueTuple<float, float>;
using ThreeNumbers = System.ValueTuple<float, float, float>;
using FourNumbers = System.ValueTuple<float, float, float, float>;

using static System.Environment;
using static Games.Common.IO.Strings;

```

It looks like this is a C# class that contains two test methods: `Read4NumbersTestCases` and `ReadNumbersTestCases`.

The `Read4NumbersTestCases` method takes an `IReadWrite` and an input string `prompt`, and returns a `Func<IReadWrite, FourNumbers>` that reads the input string and returns a `FourNumbers` object.

The `ReadNumbersTestCases` method takes an `IReadWrite` and an input string `prompt`, and returns a `TheoryData` object that contains a function that reads the input string and returns a list of `float` values.

Both methods read input values and return objects that contain the input values. However, the return type of the `Read4NumbersTestCases` method is `Func<IReadWrite, FourNumbers>`, while the return type of the `ReadNumbersTestCases` method is `TheoryData<Func<IReadWrite, IReadOnlyList<float>>, string, string, float[]>`.


```
namespace Games.Common.IO.TextIOTests;

public class ReadMethodTests
{
    [Theory]
    [MemberData(nameof(ReadStringTestCases))]
    [MemberData(nameof(Read2StringsTestCases))]
    [MemberData(nameof(ReadNumberTestCases))]
    [MemberData(nameof(Read2NumbersTestCases))]
    [MemberData(nameof(Read3NumbersTestCases))]
    [MemberData(nameof(Read4NumbersTestCases))]
    [MemberData(nameof(ReadNumbersTestCases))]
    public void ReadingValuesHasExpectedPromptsAndResults<T>(
        Func<IReadWrite, T> read,
        string input,
        string expectedOutput,
        T expectedResult)
    {
        var inputReader = new StringReader(input + Environment.NewLine);
        var outputWriter = new StringWriter();
        var io = new TextIO(inputReader, outputWriter);

        var result = read.Invoke(io);
        var output = outputWriter.ToString();

        using var _ = new AssertionScope();
        output.Should().Be(expectedOutput);
        result.Should().BeEquivalentTo(expectedResult);
    }

    [Fact]
    public void ReadNumbers_ArrayEmpty_ThrowsArgumentException()
    {
        var io = new TextIO(new StringReader(""), new StringWriter());

        Action readNumbers = () => io.ReadNumbers("foo", Array.Empty<float>());

        readNumbers.Should().Throw<ArgumentException>()
            .WithMessage("'values' must have a non-zero length.*")
            .WithParameterName("values");
    }

    public static TheoryData<Func<IReadWrite, string>, string, string, string> ReadStringTestCases()
    {
        static Func<IReadWrite, string> ReadString(string prompt) => io => io.ReadString(prompt);

        return new()
        {
            { ReadString("Name"), "", "Name? ", "" },
            { ReadString("prompt"), " foo  ,bar", $"prompt? {ExtraInput}{NewLine}", "foo" }
        };
    }

    public static TheoryData<Func<IReadWrite, TwoStrings>, string, string, TwoStrings> Read2StringsTestCases()
    {
        static Func<IReadWrite, TwoStrings> Read2Strings(string prompt) => io => io.Read2Strings(prompt);

        return new()
        {
            { Read2Strings("2 strings"), ",", "2 strings? ", ("", "") },
            {
                Read2Strings("Input please"),
                $"{NewLine}x,y",
                $"Input please? ?? {ExtraInput}{NewLine}",
                ("", "x")
            }
        };
    }

    public static TheoryData<Func<IReadWrite, float>, string, string, float> ReadNumberTestCases()
    {
        static Func<IReadWrite, float> ReadNumber(string prompt) => io => io.ReadNumber(prompt);

        return new()
        {
            { ReadNumber("Age"), $"{NewLine}42,", $"Age? {NumberExpected}{NewLine}? {ExtraInput}{NewLine}", 42 },
            { ReadNumber("Guess"), "3,4,5", $"Guess? {ExtraInput}{NewLine}", 3 }
        };
    }

    public static TheoryData<Func<IReadWrite, TwoNumbers>, string, string, TwoNumbers> Read2NumbersTestCases()
    {
        static Func<IReadWrite, TwoNumbers> Read2Numbers(string prompt) => io => io.Read2Numbers(prompt);

        return new()
        {
            { Read2Numbers("Point"), "3,4,5", $"Point? {ExtraInput}{NewLine}", (3, 4) },
            {
                Read2Numbers("Foo"),
                $"x,4,5{NewLine}4,5,x",
                $"Foo? {NumberExpected}{NewLine}? {ExtraInput}{NewLine}",
                (4, 5)
            }
        };
    }

    public static TheoryData<Func<IReadWrite, ThreeNumbers>, string, string, ThreeNumbers> Read3NumbersTestCases()
    {
        static Func<IReadWrite, ThreeNumbers> Read3Numbers(string prompt) => io => io.Read3Numbers(prompt);

        return new()
        {
            { Read3Numbers("Point"), "3.2, 4.3, 5.4, 6.5", $"Point? {ExtraInput}{NewLine}", (3.2F, 4.3F, 5.4F) },
            {
                Read3Numbers("Bar"),
                $"x,4,5{NewLine}4,5,x{NewLine}6,7,8,y",
                $"Bar? {NumberExpected}{NewLine}? {NumberExpected}{NewLine}? {ExtraInput}{NewLine}",
                (6, 7, 8)
            }
        };
    }

    public static TheoryData<Func<IReadWrite, FourNumbers>, string, string, FourNumbers> Read4NumbersTestCases()
    {
        static Func<IReadWrite, FourNumbers> Read4Numbers(string prompt) => io => io.Read4Numbers(prompt);

        return new()
        {
            { Read4Numbers("Point"), "3,4,5,6,7", $"Point? {ExtraInput}{NewLine}", (3, 4, 5, 6) },
            {
                Read4Numbers("Baz"),
                $"x,4,5,6{NewLine} 4, 5 , 6,7  ,x",
                $"Baz? {NumberExpected}{NewLine}? {ExtraInput}{NewLine}",
                (4, 5, 6, 7)
            }
        };
    }

    public static TheoryData<Func<IReadWrite, IReadOnlyList<float>>, string, string, float[]> ReadNumbersTestCases()
    {
        static Func<IReadWrite, IReadOnlyList<float>> ReadNumbers(string prompt) =>
            io =>
            {
                var numbers = new float[6];
                io.ReadNumbers(prompt, numbers);
                return numbers;
            };

        return new()
        {
            { ReadNumbers("Primes"), "2, 3, 5, 7, 11, 13", $"Primes? ", new float[] { 2, 3, 5, 7, 11, 13 } },
            {
                ReadNumbers("Qux"),
                $"42{NewLine}3.141, 2.718{NewLine}3.0e8, 6.02e23{NewLine}9.11E-28",
                $"Qux? ?? ?? ?? ",
                new[] { 42, 3.141F, 2.718F, 3.0e8F, 6.02e23F, 9.11E-28F }
            }
        };
    }
}

```

# `00_Common/javascript/WebTerminal/HtmlTerminal.js`

This is a JavaScript class called `Write`, which is a simple text-to-the-screen utility class that can write text to the console, insert a newline at the end, or handle multiple lines of text.

The class has the following methods:

* `write(text)`: writes the given text to the console. Can handle multiple lines of text and insert a newline at the end.
* `writeln(text)`: writes the given text to the console and inserts a newline at the end.
* `input(callback)`: prompts the user to enter text, which is then added to the input callback. The input ends with a linebreak.

The `@public` annotation indicates that these methods are public, which means they can be used by other classes or scripts outside of the class's scope.


```
/**
 * @class HtmlTerminal
 * 
 * This class is a very basic implementation of a "terminal" in the browser.
 * It provides simple functions like "write" and an "input" Callback.
 * 
 * @license AGPL-2.0
 * @author Alexaner Wunschik <https://github.com/mojoaxel>
 */
class HtmlTerminal {

  /**
   * Input callback.
   * If the prompt is activated by calling the input function
   * a callback is defined. If this member is not set this means
   * the prompt is not active.
   * 
   * @private
   * @type {function}
   */
  #inputCallback = undefined;

  /**
   * A html element to show a "prompt".
   * 
   * @private
   * @type {HTMLElement}
   */
  #$prompt;
  
  /**
   * Constructor
   * Creates a basic terminal simulation on the provided HTMLElement.
   * 
   * @param {HTMLElement} $output - a dom element
   */
  constructor($output) {
    // Store the output DOM element in a local variable.
    this.$output = $output;

    // Clear terminal.
    this.clear();

    // Add the call "terminal" to the $output element.
    this.$output.classList.add('terminal');

    // Create a prompt element.
    // This element gets added if input is needed.
    this.#$prompt = document.createElement("input");
    this.#$prompt.setAttribute("id", "prompt");
    this.#$prompt.setAttribute("type", "text");
    this.#$prompt.setAttribute("length", "50");
    this.#$prompt.addEventListener("keydown", this.#handleKey.bind(this));

    // Force focus on the promt on each click.
    // This is needed for mobile support.
    document.body.addEventListener('click', () => {
      this.#$prompt.focus();
    });
  }

  /**
   * Creates a new HTMLElement with the given text content.
   * This element than gets added to the $output as a new "line".
   * 
   * @private
   * @memberof MinimalTerminal
   * @param {String} text - text that should be displayed in the new "line".
   * @returns {HTMLElement} return a new DOM Element <pre class="line"></pre>
   */
  #newLine(text) {
    const $lineNode = document.createElement("pre");
    $lineNode.classList.add("line");
    $lineNode.innerText = text;
    return $lineNode;
  }

  /**
   * TODO
   * 
   * @private
   * @param {*} e 
   */
  #handleKey(e) {
    // if no input-callback is defined just return
    if (!this.#inputCallback) {
      return;
    }

    if (e.keyCode == 13) {
      const text = this.#$prompt.value;
      this.#$prompt.value = '';
      this.#$prompt.remove();
      this.#inputCallback(text + '\n');
    }
  }

  /**
   * Clear the terminal.
   * Remove all lines.
   * 
   * @public
   */
  clear() {
    this.$output.innerText = "";
  }

  /**
   * Create a new div and add html content.
   * 
   * @public
   * @param {*} htmlContent 
   */
  inserHtml(htmlContent) {
    const $htmlNode = document.createElement("div");
    $htmlNode.innerHTML = htmlContent;
    this.$output.appendChild($htmlNode);
    document.body.scrollTo(0, document.body.scrollHeight);
  }

  /**
   * Write a text to the terminal.
   * By default there is no linebreak at the end of a new line
   * except the line ensd with a "\n".
   * If the given text has multible linebreaks, multibe lines are inserted.
   * 
   * @public
   * @param {string} text 
   */
  write(text) {
    if (!text || text.length <= 0) {
      // empty line
      this.$output.appendChild(document.createElement("br"));
    } else if (text.endsWith("\n")) {
      // single line with linebrank
      const $lineNode = this.#newLine(text);
      this.$output.appendChild(this.#newLine(text));
      this.$output.appendChild(document.createElement("br"));
    } else if (text.includes("\n")) {
      // multible lines
      const lines = text.split("\n");
      lines.forEach((line) => {
        this.write(line);
      });
    } else {
      // single line
      this.$output.appendChild(this.#newLine(text));
    }

    // scroll to the buttom of the page
    document.body.scrollTo(0, document.body.scrollHeight);
  }

  /**
   * Like "write" but with a newline at the end.
   * 
   * @public
   * @param {*} text 
   */
  writeln(text) {
    this.write(text + "\n");
  }

  /**
   * Query from user input.
   * This is done by adding a input-element at the end of the terminal,
   * that showes a prompt and a blinking cursor.
   * If a key is pressed the input is added to the prompt element.
   * The input ends with a linebreak.
   * 
   * @public
   * @param {*} callback 
   */
  input(callback) {
    // show prompt with a blinking prompt
    this.#inputCallback = callback;
    this.$output.appendChild(this.#$prompt);
    this.#$prompt.focus();
  }
}

```

# `00_Utilities/build-index.js`

该脚本的作用是创建一个名为 "index.html" 的文件，并将其放在目录的根目录中。然后在脚本目录下使用 `fs.writeFileSync` 函数将当前目录下的 "index.html" 文件复制到 "index.html" 文件中。

具体来说，该脚本需要使用 `fs` 和 `path` 模块，通过 `require` 函数导入它们。然后通过 `const fs = require('fs');` 导入 `fs` 模块，通过 `const path = require('path');` 导入 `path` 模块。接下来，通过 `const TITLE = 'BASIC Computer Games';` 定义一个名为 "TITLE" 的常量，通过 `const JAVASCRIPT_FOLDER = 'javascript';` 定义一个名为 "JAVASCRIPT_FOLDER" 的常量。最后，通过 `const fs = require('fs');` 和 `const path = require('path');` 调用 `fs.writeFileSync` 函数将 "index.html" 文件复制到 "index.html" 文件中，并将 "TITLE" 和 "JAVASCRIPT_FOLDER" 赋值给 `TITLE` 和 `JAVASCRIPT_FOLDER`， respectively。


```
#!/usr/bin/env node
/**
 * This script creates an "index.html" file in the root of the directory.
 * 
 * Call this from the root of the project with 
 *  `node ./00_Utilities/build-index.js`
 * 
 * @author Alexander Wunschik <https://github.com/mojoaxel>
 */

const fs = require('fs');
const path = require('path');

const TITLE = 'BASIC Computer Games';
const JAVASCRIPT_FOLDER = 'javascript';
```

该代码的作用是创建一个游戏链接列表，用于将游戏的各个文件提供给用户。具体来说，它实现了以下功能：

1. 遍历游戏文件的列表。
2. 对于每个游戏文件，根据其文件类型创建一个链接元素，如HTML或JavaScript文件。
3. 如果游戏文件是JavaScript文件，它将在链接元素中添加一个包含该文件的链接。
4. 如果游戏文件是HTML文件，它将在链接元素中添加一个包含该文件的链接。
5. 如果某个游戏文件无法被处理（例如，文件不存在或文件类型不匹配），则会抛出一个错误。
6. 在文件列表的末尾，如果游戏文件列表的长度大于1，它将在列表中添加一个包含所有文件的链接。
7. 在游戏文件列表的顶部，添加游戏名称。
8. 在每个链接元素中，使用`<li>`标签来表示链接。
9. 在每个链接元素中，使用`<a>`标签来表示链接。
10. 在`<a>`标签的`href`属性中，使用`path.basename(file)`来获取文件的完整路径，而不是`file`，以确保正确处理JavaScript文件。
11. 在处理游戏文件时，如果文件以`.html`或`.mjs`结尾，它将使用`path.basename(file)`而不是`file`来获取文件名，以确保正确处理HTML和JavaScript文件。
12. 如果在处理游戏文件时遇到错误，它将在栈中捕获并抛出错误。
13. 最后，如果游戏文件列表的长度大于1，它将在链接列表的末尾添加一个包含所有文件的链接。


```
const IGNORE_FOLDERS_START_WITH = ['.', '00_', 'buildJvm', 'Sudoku'];
const IGNORE_FILES = [
	// "84 Super Star Trek"  has it's own node/js implementation (using xterm)
	'cli.mjs', 'superstartrek.mjs'
];

function createGameLinks(game) {
	const creatFileLink = (file, name = path.basename(file)) => {
		if (file.endsWith('.html')) {
			return `
				<li><a href="${file}">${name.replace('.html', '')}</a></li>
			`;
		} else if (file.endsWith('.mjs')) {
			return `
				<li><a href="./00_Common/javascript/WebTerminal/terminal.html#${file}">${name.replace('.mjs', '')} (node.js)</a></li>
			`;
		} else {
			throw new Error(`Unknown file-type found: ${file}`);
		}
	}

	if (game.files.length > 1) {
		const entries = game.files.map(file => {
			return creatFileLink(file);
		});
		return `
			<li>
				<span>${game.name}</span>
				<ul>${entries.map(e => `\t\t\t${e}`).join('\n')}</ul>
			</li>
		`;
	} else {
		return creatFileLink(game.files[0], game.name);
	}
}

```

该函数的作用是创建一个包含游戏列表的 HTML 文件头和主体内容。具体来说，它将游戏列表中的每个游戏链接进行处理，创建一个带标题和游戏名称的列表项，并将它们组合成一个 HTML 文件。函数的输入参数包括游戏列表和标题，游戏列表是一个数组的数组，而标题是一个字符串。函数将返回一个 HTML 文件，以将游戏列表和标题显示在 HTML 文件中。


```
function createIndexHtml(title, games) {
	const listEntries = games.map(game => 
		createGameLinks(game)
	).map(entry => `\t\t\t${entry}`).join('\n');

	const head = `
		<head>
			<meta charset="UTF-8">
			<title>${title}</title>
			<link rel="stylesheet" href="./00_Utilities/javascript/style_terminal.css" />
		</head>
	`;

	const body = `
		<body>
			<article id="output">
				<header>
					<h1>${title}</h1>
				</header>
				<main>
					<ul>
						${listEntries}
					</ul>
				</main>
			</article>
		</body>
	`;

	return `
		<!DOCTYPE html>
		<html lang="en">
		${head}
		${body}
		</html>
	`.trim().replace(/\s\s+/g, '');
}

```

该函数的作用是查找指定文件夹中所有的JavaScript文件，并返回它们的路径。函数的实现包括以下步骤：

1. 首先检查指定的文件夹是否包含一个名为 "javascript" 的子文件夹。如果不包含，则抛出一个错误。
2. 如果包含子文件夹，则递归地检查其中所有的文件。
3. 过滤出所有的JavaScript文件，只允许以".html"结尾的文件通过。
4. 过滤出所有的JavaScript文件，允许名为".mjs"的文件通过。
5. 通过递归地检查".html"和".mjs"的文件，过滤出所有的文件。
6. 如果过滤出的文件数量为0，则抛出一个错误。
7. 返回过滤出的文件的路径，使用path.join()方法将文件夹和文件名连接起来。


```
function findJSFilesInFolder(folder) {
	// filter folders that do not include a subfolder called "javascript"
	const hasJavascript = fs.existsSync(`${folder}/${JAVASCRIPT_FOLDER}`);
	if (!hasJavascript) {
		throw new Error(`Game "${folder}" is missing a javascript folder`);
	}

	// get all files in the javascript folder
	const files = fs.readdirSync(`${folder}/${JAVASCRIPT_FOLDER}`);

	// filter files only allow .html files
	const htmlFiles = files.filter(file => file.endsWith('.html'));
	const mjsFiles = files.filter(file => file.endsWith('.mjs'));
	const entries = [
		...htmlFiles,
		...mjsFiles
	].filter(file => !IGNORE_FILES.includes(file));

	if (entries.length == 0) {
		throw new Error(`Game "${folder}" is missing a HTML or node.js file in the folder "${folder}/${JAVASCRIPT_FOLDER}"`);
	}

	return entries.map(file => path.join(folder, JAVASCRIPT_FOLDER, file));
}

```

该代码是一个 Node.js 函数，主要目的是列出当前目录下的所有文件夹，并对文件夹进行筛选，最后生成一个包含所有游戏信息并保存为 index.html 的文件。

具体来说，代码的功能如下：

1. 使用 fs.readdirSync 函数获取当前目录下的所有文件夹。
2. 对获取到的文件夹进行过滤，仅保留文件夹和其子文件夹。
3. 对文件夹进行过滤，仅允许以点号（.）开头的文件夹。
4. 对文件夹进行排序，按照文件夹名称字母顺序排序。
5. 使用 findJSFilesInFolder 函数查找文件夹内的所有 JavaScript 文件，并过滤掉文件夹名称以点号（.）开头的文件夹。
6. 对上一步得到的文件夹进行过滤，仅保留其中的游戏文件夹。
7. 根据游戏文件夹创建一个 index.html 文件，并将其内容保存到当前目录下。
8. 在主函数中输出 "index.html" 文件的成功创建。


```
function main() {
	// Get the list of all folders in the current director
	let folders = fs.readdirSync(process.cwd());

	// filter files only allow folders
	folders = folders.filter(folder => fs.statSync(folder).isDirectory());

	// filter out the folders that start with a dot or 00_
	folders = folders.filter(folder => {
		return !IGNORE_FOLDERS_START_WITH.some(ignore => folder.startsWith(ignore));
	});

	// sort the folders alphabetically (by number)
	folders = folders.sort();

	// get name and javascript file from folder
	const games = folders.map(folder => {
		const name = folder.replace('_', ' ');
		let files;
		
		try {
			files = findJSFilesInFolder(folder);
		} catch (error) {
			console.warn(`Game "${name}" is missing a javascript implementation: ${error.message}`);
			return null;
		}

		return {
			name,
			files
		}
	}).filter(game => game !== null);

	// create a index.html file with a list of all games
	const htmlContent = createIndexHtml(TITLE, games);
	fs.writeFileSync('index.html', htmlContent);
	console.log(`index.html successfully created!`);
}

```

这行代码是一个 C 语言程序的主函数入口，它告诉编译器在程序运行时从这里开始执行代码。通常在程序注释的前面，而且程序从此处开始执行，所以也可以称为程序的起点。


```
main();
```

# `00_Utilities/find-missing-implementations.js`

这段代码的主要作用是查找给定语言中未找到解决方案的游戏。它通过遍历游戏文件夹和语言文件夹，并在每个语言文件夹中查找至少一个符合预期扩展名的文件。如果找到了这样的文件，它将添加到一個選定的文件夾中，以供後續擴展。


```
/**
 * Program to find games that are missing solutions in a given language
 *
 * Scan each game folder, check for a folder for each language, and also make
 * sure there's at least one file of the expected extension and not just a
 * readme or something
 */

const fs = require("fs");
const glob = require("glob");

// relative path to the repository root
const ROOT_PATH = "../.";

const languages = [
  { name: "csharp", extension: "cs" },
  { name: "java", extension: "java" },
  { name: "javascript", extension: "html" },
  { name: "pascal", extension: "pas" },
  { name: "perl", extension: "pl" },
  { name: "python", extension: "py" },
  { name: "ruby", extension: "rb" },
  { name: "vbnet", extension: "vb" },
];

```

This is a JavaScript file that provides a solution to a problem of finding all puzzles in a language-rich environment and counting the number of solutions for each language. It uses the `fs` and `esm` modules to perform file I/O and ES5-style arrow functions.

The `getFilesRecursive` function is defined to retrieve all files with a given language extension in the root directory. The solution is stored in an object that maps puzzles to a list of languages that have solutions for that puzzle. The languages are stored in an object that maps each language to its index in the puzzle mapping object.

The main function reads the puzzle directories and counts the number of solutions for each language in the game. It then outputs the number of missing solutions and the number of missing languages for each game.

The `languages` array includes all the known programming languages. The puzzle directories are read from the root directory and are stored in an object. The puzzle directories are then processed one at a time to count the solutions for each language and output the number of missing solutions and the number of missing languages for each game.


```
const getFilesRecursive = async (path, extension) => {
  return new Promise((resolve, reject) => {
    glob(`${path}/**/*.${extension}`, (err, matches) => {
      if (err) {
        reject(err);
      }
      resolve(matches);
    });
  });
};

const getPuzzleFolders = () => {
  return fs
    .readdirSync(ROOT_PATH, { withFileTypes: true })
    .filter((dirEntry) => dirEntry.isDirectory())
    .filter(
      (dirEntry) =>
        ![".git", "node_modules", "00_Utilities"].includes(dirEntry.name)
    )
    .map((dirEntry) => dirEntry.name);
};

(async () => {
  let missingGames = {};
  let missingLanguageCounts = {};
  languages.forEach((l) => (missingLanguageCounts[l.name] = 0));
  const puzzles = getPuzzleFolders();
  for (const puzzle of puzzles) {
    for (const { name: language, extension } of languages) {
      const files = await getFilesRecursive(
        `${ROOT_PATH}/${puzzle}/${language}`,
        extension
      );
      if (files.length === 0) {
        if (!missingGames[puzzle]) missingGames[puzzle] = [];

        missingGames[puzzle].push(language);
        missingLanguageCounts[language]++;
      }
    }
  }
  const missingCount = Object.values(missingGames).flat().length;
  if (missingCount === 0) {
    console.log("All games have solutions for all languages");
  } else {
    console.log(`Missing ${missingCount} implementations:`);

    Object.entries(missingGames).forEach(
      ([p, ls]) => (missingGames[p] = ls.join(", "))
    );

    console.log(`\nMissing languages by game:`);
    console.table(missingGames);
    console.log(`\nBy language:`);
    console.table(missingLanguageCounts);
  }
})();

```

这是一条单行注释，意味着这段代码只是一个简单的回调函数声明，不会执行实际的代码。单行注释通常用于在代码中描述某种特殊的含义或提供一个简短的说明。


```
return;

```

# `00_Utilities/find-unimplemented.js`

这是一个Node.js脚本，它的作用是找出未实现的游戏，并输出这些游戏的代码。它使用了JavaScript核心模块（require）和全局模块（glob）来查找各自项目文件夹下的.js文件。通过循环遍历这些文件，发现未实现的游戏后，将它们的内容输出到控制台。

脚本会将所有找到的未实现游戏按照其所在的语言名称进行分组，并在每种语言下面打印一个数组。例如，如果你运行这个脚本并且存在名为 "lang1.js" 和 "lang2.js" 的文件，那么它将输出：
```scss
[
 "Lang1",
 "Lang2"
]
```
如果你想要进一步限定筛选语言，可以在运行时使用"-L"选项，如下：
```scss
find-unimplemented.js -L lang1 -L lang2 ...
```
这将只输出特定语言中未实现的游戏，例如：
```scss
[  "Lang1",  "Lang2"]
```


```
/**
 * Program to show unimplemented games by language, optionally filtered by
 * language
 *
 * Usage: node find-unimplemented.js [[[lang1] lang2] ...]
 *
 * Adapted from find-missing-implementtion.js
 */

const fs = require("fs");
const glob = require("glob");

// relative path to the repository root
const ROOT_PATH = "../.";

```

该代码定义了一个名为 "languages" 的数组，其中包含多个对象，每个对象都包含一个 "name" 属性和一个 "extension" 属性。

接着，定义了一个名为 "getFilesRecursive" 的函数，该函数接受两个参数：一个 "path" 参数和一个 "extension" 参数。函数使用 "glob" 函数来查找指定路径下的所有文件，并返回一个 Promise 对象，其 "resolve" 方法返回一个由匹配的文件名组成的数组，其 "reject" 方法接受一个错误对象。

最后，该代码将数组 "languages" 的每个对象都赋值给一个名为 "languages" 的变量，并将该变量存储在一个名为 "const languages" 的常量中。


```
let languages = [
  { name: "csharp", extension: "cs" },
  { name: "java", extension: "java" },
  { name: "javascript", extension: "html" },
  { name: "pascal", extension: "pas" },
  { name: "perl", extension: "pl" },
  { name: "python", extension: "py" },
  { name: "ruby", extension: "rb" },
  { name: "vbnet", extension: "vb" },
];

const getFilesRecursive = async (path, extension) => {
  return new Promise((resolve, reject) => {
    glob(`${path}/**/*.${extension}`, (err, matches) => {
      if (err) {
        reject(err);
      }
      resolve(matches);
    });
  });
};

```

这段代码的作用是实现了一个程序，用于处理一些二进制文件的夹。它接受一个参数，用于指定二进制文件夹的根目录。代码首先使用fs.readdirSync函数读取指定根目录下的所有二进制文件夹，然后使用filter函数过滤出那些包含".git"、"node_modules"、"00_Utilities"或"buildJvm"等文件夹的目录。接下来，代码使用map函数将二进制文件夹名称存储到一个平数组中。

接下来，代码会接受一个或多个命令行参数，用于指定要分析的语言列表。对于每个语言，代码会使用过滤函数从其名称中删除".git"、".node_modules"、".00_Utilities"和".buildJvm"等文件夹，并将它们存储到一个平数组中。最后，代码会将每个语言的二进制文件夹存储到一个结果数组中。


```
const getPuzzleFolders = () => {
  return fs
    .readdirSync(ROOT_PATH, { withFileTypes: true })
    .filter((dirEntry) => dirEntry.isDirectory())
    .filter(
      (dirEntry) =>
        ![".git", "node_modules", "00_Utilities", "buildJvm"].includes(dirEntry.name)
    )
    .map((dirEntry) => dirEntry.name);
};

(async () => {
  const result = {};
  if (process.argv.length > 2) {
    languages = languages.filter((language) => process.argv.slice(2).includes(language.name));
  }
  for (const { name: language } of languages) {
    result[language] = [];
  }

  const puzzleFolders = getPuzzleFolders();
  for (const puzzleFolder of puzzleFolders) {
    for (const { name: language, extension } of languages) {
      const files = await getFilesRecursive(
        `${ROOT_PATH}/${puzzleFolder}/${language}`, extension
      );
      if (files.length === 0) {
        result[language].push(puzzleFolder);
      }
    }
  }
  console.log('Unimplementation by language:')
  console.dir(result);
})();

```

这是一条函数指针，它返回一个函数表达式。函数表达式是一个可以调用一个函数的代码块，通常包括函数的名称、参数和返回类型。

这条代码的作用是将函数指针置为空，即不指向任何函数的代码块。这意味着函数表达式不会调用任何函数，无论程序员如何调用它，它都不会产生任何输出或产生任何错误。

这条代码通常用于编写不输出函数内部代码块的函数，或者在函数被继承时，防止子类意外地调用了父类的实现。


```
return;

```

# `00_Utilities/markdown_todo.py`

The script appears to be processing and outlining the contents of a directory tree. It starts by setting the initial contents of the `strings_done` list to an empty list and the initial checklist to an empty list.

It then loops through the `os.walk` function, which is a part of the Python standard library that generates the contents of a directory tree. For each directory it encounters, it checks if the directory name contains any of a predefined list of keywords and, if so, joins the `checklist` with the directory name. If not, it adds the directory name to the `checklist`.

Next, it checks if the directory name is a new directory and, if it is not, it splits the name into its component parts and joins them with `/`. It then checks the language of the directory name and, if the language is not in a predefined list of ignore folders, it adds the directory name to the `strings_done` list and the language to the `checklist`, with a comment that mentions the language.

If the directory name is a new directory or contains a directory that contains `.git`, it adds the directory name to the `strings_done` list and the language to the `checklist`.

Finally, it checks if there is a `buildJvm` or `htmlcov` file in the directory and, if there is, it prints a comment with the contents of the file.

It is unclear from the code provided what the `os.walk` function is supposed to do, but it appears to be some kind of directory traversal function that generates the contents of a directory tree.


```
import os
from typing import Dict, List


def has_implementation(lang: str, file_list: List[str], subdir_list: List[str]) -> bool:
    if lang == "csharp":
        return any(file.endswith(".cs") for file in file_list)
    elif lang == "vbnet":
        return any(file.endswith(".vb") for file in file_list)
    else:
        return len(file_list) > 1 or len(subdir_list) > 0


def get_data(checklist_orig: List[str], root_dir: str = "..") -> List[List[str]]:
    """

    Parameters
    ----------
    root_dir : str
        The root directory you want to start from.
    """
    lang_pos: Dict[str, int] = {
        lang: i for i, lang in enumerate(checklist_orig[1:], start=1)
    }
    strings_done: List[List[str]] = []

    ignore_folders = [
        ".git",
        "00_Utilities",
        ".github",
        ".mypy_cache",
        ".pytest_cache",
        "00_Alternate_Languages",
        "00_Common",
        "buildJvm",
        "htmlcov",
    ]

    prev_game = ""

    empty_boxes = ["⬜️" for _ in checklist_orig]
    checklist = empty_boxes[:]

    for dir_name, subdir_list, file_list in sorted(os.walk(root_dir)):
        # split_dir[1] is the game
        # split_dir[2] is the language
        split_dir = dir_name.split(os.path.sep)

        if len(split_dir) == 2 and split_dir[1] not in ignore_folders:
            if prev_game == "":
                prev_game = split_dir[1]
                checklist[0] = f"{split_dir[1]:<30}"

            if prev_game != split_dir[1]:
                # it's a new dir
                strings_done.append(checklist)
                checklist = [
                    f"{f'[{split_dir[1]}](../{split_dir[1]})':<30}",
                ] + empty_boxes[1:]
                prev_game = split_dir[1]
        elif (
            len(split_dir) == 3 and split_dir[1] != ".git" and split_dir[2] in lang_pos
        ):
            out = (
                "✅"
                if has_implementation(split_dir[2], file_list, subdir_list)
                else "⬜️"
            )
            if split_dir[2] not in lang_pos or lang_pos[split_dir[2]] >= len(checklist):
                print(f"Could not find {split_dir[2]}: {dir_name}")
                checklist[lang_pos[split_dir[2]]] = "⬜️"
                continue
            checklist[lang_pos[split_dir[2]]] = out
    return strings_done


```

这段代码定义了一个名为 `write_file` 的函数，它接受一个字符串 `path`、一个包含多个语言名称的字符串 `languages` 和一个包含多个字符串完成情况的列表 `strings_done` 作为参数。这个函数的主要目的是将所给参数写入到一个名为 `path/_game.txt` 的文件中，并在文件中添加一些游戏信息。

具体来说，这段代码执行以下操作：

1. 将给定的 `languages` 字列表中的每个语言名称转换为一个空格，并在列表末尾添加一个 `---`。
2. 将 `languages` 列表中的所有语言名称连接成一个字符串，并将其转换为一个空格。
3. 将上面得到的两个字符串连接成一个字符串，并将其转换为一个空格。
4. 给定一个字符串 `write_string`，其中包含游戏信息和一些其他信息（如游戏总数）。
5. 对 `write_string` 进行一些修改，包括：
  a. 在开头添加一个 "# TODO" 标签，表示这是一个待完成的工作。
  b. 在游戏信息字符串中，将游戏总数 `nb_games` 的值转换为一个带有百分号的字符串。
  c. 在 `strings_done` 中的每个元素都是一个包含多个语言名称的列表，每个元素对应一个完成情况。我们将这些列表中的所有元素连接成一个空格，并在空格中添加一个 `---`。
  d. 对于每个完成情况，我们将完成情况信息连接到游戏信息上。
6. 使用 `with` 语句打开一个名为 `_game.txt` 的文件，并将 `write_string` 中的内容写入文件中。

这段代码的作用是将所给的 `languages`、`strings_done` 和 `write_file` 参数写入到一个名为 `_game.txt` 的文件中，其中 `languages` 参数定义了游戏名称，`strings_done` 参数定义了每个完成情况的字符串，`write_file` 函数包含了游戏信息和一些其他信息。


```
def write_file(path: str, languages: List[str], strings_done: List[List[str]]) -> None:
    dashes_arr = ["---"] * (len(languages) + 1)
    dashes_arr[0] = "-" * 30
    dashes = " | ".join(dashes_arr)
    write_string = f"# TODO list\n {'game':<30}| {' | '.join(languages)}\n{dashes}\n"
    sorted_strings = list(
        map(lambda l: " | ".join(l) + "\n", sorted(strings_done, key=lambda x: x[0]))
    )
    write_string += "".join(sorted_strings)
    write_string += f"{dashes}\n"
    language_indices = range(1, len(languages) + 1)
    nb_games = len(strings_done)
    write_string += (
        f"{f'Sum of {nb_games}':<30} | "
        + " | ".join(
            [
                f"{sum(row[lang] == '✅' for row in strings_done)}"
                for lang in language_indices
            ]
        )
        + "\n"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(write_string)


```

这段代码的作用是检查当前脚本是否作为主程序运行，如果不是，则执行以下操作：

1. 定义了一个名为`languages`的字典，包含了各种编程语言的名称和相应的缩写。
2. 调用了一个名为`get_data`的函数，该函数的作用是从一个包含列表和语言名称的列表中获取数据并将它们存储在变量中。
3. 创建了一个名为`write_file`的函数，该函数的作用是 write_file 文件("TODO.md")。
4. 将上面两个函数的结果进行绑定，并打印出所有语言名称和其相应的缩写。


```
if __name__ == "__main__":
    languages = {
        "csharp": "C#",
        "java": "Java",
        "javascript": "JS",
        "kotlin": "Kotlin",
        "lua": "Lua",
        "perl": "Perl",
        "python": "Python",
        "ruby": "Ruby",
        "rust": "Rust",
        "vbnet": "VB.NET",
    }
    strings_done = get_data(["game"] + list(languages.keys()))
    write_file("TODO.md", list(languages.values()), strings_done)

```

#### Utilities

These are global helper / utility programs to assist us in maintaining all the ports. 

# TODO list
 game                          | C# | Java | JS | Kotlin | Lua | Perl | Python | Ruby | Rust | VB.NET
------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
01_Acey_Ducey                  | ✅ | ✅ | ✅ | ⬜️ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅
[02_Amazing](../02_Amazing)    | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ✅ | ✅
[03_Animal](../03_Animal)      | ✅ | ✅ | ✅ | ✅ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[04_Awari](../04_Awari)        | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[05_Bagels](../05_Bagels)      | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⬜️
[06_Banner](../06_Banner)      | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ✅
[07_Basketball](../07_Basketball) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️
[08_Batnum](../08_Batnum)      | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅
[09_Battle](../09_Battle)      | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[10_Blackjack](../10_Blackjack) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️
[11_Bombardment](../11_Bombardment) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[12_Bombs_Away](../12_Bombs_Away) | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️
[13_Bounce](../13_Bounce)      | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️
[14_Bowling](../14_Bowling)    | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[15_Boxing](../15_Boxing)      | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[16_Bug](../16_Bug)            | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️
[17_Bullfight](../17_Bullfight) | ✅ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[18_Bullseye](../18_Bullseye)  | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️
[19_Bunny](../19_Bunny)        | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[20_Buzzword](../20_Buzzword)  | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[21_Calendar](../21_Calendar)  | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[22_Change](../22_Change)      | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ✅ | ⬜️
[23_Checkers](../23_Checkers)  | ✅ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️
[24_Chemist](../24_Chemist)    | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️
[25_Chief](../25_Chief)        | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[26_Chomp](../26_Chomp)        | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️
[27_Civil_War](../27_Civil_War) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[28_Combat](../28_Combat)      | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[29_Craps](../29_Craps)        | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[30_Cube](../30_Cube)          | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️
[31_Depth_Charge](../31_Depth_Charge) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[32_Diamond](../32_Diamond)    | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[33_Dice](../33_Dice)          | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ✅ | ✅
[34_Digits](../34_Digits)      | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[35_Even_Wins](../35_Even_Wins) | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️
[36_Flip_Flop](../36_Flip_Flop) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️
[37_Football](../37_Football)  | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[38_Fur_Trader](../38_Fur_Trader) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[39_Golf](../39_Golf)          | ✅ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[40_Gomoko](../40_Gomoko)      | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️
[41_Guess](../41_Guess)        | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ✅ | ⬜️
[42_Gunner](../42_Gunner)      | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[43_Hammurabi](../43_Hammurabi) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[44_Hangman](../44_Hangman)    | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[45_Hello](../45_Hello)        | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[46_Hexapawn](../46_Hexapawn)  | ✅ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[47_Hi-Lo](../47_Hi-Lo)        | ✅ | ✅ | ✅ | ✅ | ⬜️ | ✅ | ✅ | ✅ | ✅ | ⬜️
[48_High_IQ](../48_High_IQ)    | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[49_Hockey](../49_Hockey)      | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️ | ⬜️ | ⬜️ | ⬜️ | ⬜️
[50_Horserace](../50_Horserace) | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[51_Hurkle](../51_Hurkle)      | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[52_Kinema](../52_Kinema)      | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[53_King](../53_King)          | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[54_Letter](../54_Letter)      | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ✅ | ⬜️
[55_Life](../55_Life)          | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[56_Life_for_Two](../56_Life_for_Two) | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️ | ⬜️ | ⬜️ | ⬜️ | ⬜️
[57_Literature_Quiz](../57_Literature_Quiz) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️
[58_Love](../58_Love)          | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[59_Lunar_LEM_Rocket](../59_Lunar_LEM_Rocket) | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ✅ | ⬜️
[60_Mastermind](../60_Mastermind) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ✅ | ⬜️
[61_Math_Dice](../61_Math_Dice) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ✅ | ⬜️
[62_Mugwump](../62_Mugwump)    | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️
[63_Name](../63_Name)          | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[64_Nicomachus](../64_Nicomachus) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️
[65_Nim](../65_Nim)            | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️
[66_Number](../66_Number)      | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ✅ | ⬜️
[67_One_Check](../67_One_Check) | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️
[68_Orbit](../68_Orbit)        | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[69_Pizza](../69_Pizza)        | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[70_Poetry](../70_Poetry)      | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[71_Poker](../71_Poker)        | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ⬜️ | ⬜️ | ⬜️ | ⬜️
[72_Queen](../72_Queen)        | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️
[73_Reverse](../73_Reverse)    | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ✅
[74_Rock_Scissors_Paper](../74_Rock_Scissors_Paper) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ✅ | ⬜️
[75_Roulette](../75_Roulette)  | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️
[76_Russian_Roulette](../76_Russian_Roulette) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[77_Salvo](../77_Salvo)        | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[78_Sine_Wave](../78_Sine_Wave) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ✅ | ⬜️
[79_Slalom](../79_Slalom)      | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[80_Slots](../80_Slots)        | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[81_Splat](../81_Splat)        | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️
[82_Stars](../82_Stars)        | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ✅ | ⬜️
[83_Stock_Market](../83_Stock_Market) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[84_Super_Star_Trek](../84_Super_Star_Trek) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[85_Synonym](../85_Synonym)    | ✅ | ✅ | ✅ | ✅ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[86_Target](../86_Target)      | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[87_3-D_Plot](../87_3-D_Plot)  | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[88_3-D_Tic-Tac-Toe](../88_3-D_Tic-Tac-Toe) | ✅ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[89_Tic-Tac-Toe](../89_Tic-Tac-Toe) | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ✅ | ✅ | ⬜️ | ✅ | ⬜️
[90_Tower](../90_Tower)        | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️
[91_Train](../91_Train)        | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[92_Trap](../92_Trap)          | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[93_23_Matches](../93_23_Matches) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[94_War](../94_War)            | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ✅ | ✅ | ✅ | ✅ | ⬜️
[95_Weekday](../95_Weekday)    | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ✅ | ⬜️
------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
Sum of 95                      | 63 | 80 | 95 | 7 | 2 | 60 | 92 | 49 | 20 | 6


# `00_Utilities/DotnetUtils/DotnetUtils/Extensions.cs`

这段代码是一个自定义的 Extensions 类，它提供了选缓和器的功能，用于在给定的 IEnumerable<T> 对象中选择符合给定函数表达式的元素。

具体来说，这个自定义的 Extensions 类提供以下几个方法：

- SelectT：用于在给定的 IEnumerable<(T1, T2)> 对象中选择符合给定函数表达式的元素，该函数表达式是通过 lambda 表达式实现的。
- SelectT：用于在给定的 IEnumerable<(T1, T2, T3)> 对象中选择符合给定函数表达式的元素，该函数表达式是通过 lambda 表达式实现的。
- WithIndex：用于在给定的 IEnumerable<(T1, T2)> 对象中选择符合给定函数表达式的元素，该函数表达式是通过 lambda 表达式实现的，其中传递一个整数表达式和一个 lambda 表达式。
- None：用于检查给定的 IEnumerable<T> 对象中是否存在匹配空格条件的元素，如果给定的函数表达式为空，则返回 false，否则返回 true。
- IsNullOrWhitespace：用于检查给定的字符串是否为空格或 null。
- RelativePath：用于在给定的路径中返回相对于给定根路径的相对路径，如果给定的根路径为空，则返回路径本身。该方法通过计算根路径和当前路径的相对路径来实现在根路径中查找给定路径。


```
﻿using System.Diagnostics.CodeAnalysis;
using static System.IO.Path;

namespace DotnetUtils;

public static class Extensions {
    public static IEnumerable<TResult> SelectT<T1, T2, TResult>(this IEnumerable<(T1, T2)> src, Func<T1, T2, TResult> selector) =>
        src.Select(x => selector(x.Item1, x.Item2));
    public static IEnumerable<TResult> SelectT<T1, T2, T3, TResult>(this IEnumerable<(T1, T2, T3)> src, Func<T1, T2, T3, TResult> selector) =>
        src.Select(x => selector(x.Item1, x.Item2, x.Item3));
    public static IEnumerable<(T1, T2, int)> WithIndex<T1, T2>(this IEnumerable<(T1, T2)> src) => src.Select((x, index) => (x.Item1, x.Item2, index));

    public static bool None<T>(this IEnumerable<T> src, Func<T, bool>? predicate = null) =>
        predicate is null ?
            !src.Any() :
            !src.Any(predicate);

    public static bool IsNullOrWhitespace([NotNullWhen(false)] this string? s) => string.IsNullOrWhiteSpace(s);

    [return: NotNullIfNotNull("path")]
    public static string? RelativePath(this string? path, string? rootPath) {
        if (
            path.IsNullOrWhitespace() ||
            rootPath.IsNullOrWhitespace()
        ) { return path; }

        path = path.TrimEnd('\\'); // remove trailing backslash, if present
        return GetRelativePath(rootPath, path.TrimEnd('\\'));
    }
}

```

# `00_Utilities/DotnetUtils/DotnetUtils/Functions.cs`

这段代码是一个C#类，包含一些函数，下面分别对每个函数进行解释：

1. `getValue(string path, params string[] names)`：该函数接收一个字符串参数`path`以及一个或多个字符串参数`names`。函数的作用是获取`path`中指定的`names`属性的值。如果`names`参数为空字符串，则会抛出一个`InvalidOperationException`。函数的实现使用`XDocument.Load(path).Element("Project")?.Element("PropertyGroup")`读取`path`中的`PropertyGroup`元素，然后递归调用`getValue`函数。

2. `getValue(XElement? parent, params string[] names)`：该函数与上面函数类似，不同之处在于该函数接收一个`XElement`类型的参数`parent`以及一个或多个字符串参数`names`。函数的作用是获取`parent`中指定的`names`属性的值。如果`names`参数为空字符串，则会抛出一个`InvalidOperationException`。函数的实现与上面函数类似，使用`XElement? elem`遍历`parent`元素，并尝试获取`name`属性的值。如果遍历完成后仍然没有找到指定名称的`name`属性，则返回`null`。

3. `getChoice(int maxValue)`：该函数接收一个整数参数`maxValue`。函数的作用是获取一个随机整数，并将其作为答案。函数实现简单，直接使用`Math.Random()`生成一个随机整数，然后将其作为答案。

4. `getChoice(int minValue, int maxValue)`：该函数与上面函数类似，但是允许用户指定最大值。该函数接收两个整数参数`minValue`和`maxValue`，用于获取用户输入的选择随机整数。函数的实现与上面函数类似，使用`Math.Random()`生成一个随机整数，然后将其作为答案。不过，函数会提示用户输入最小值和最大值，并要求用户输入有效的整数。


```
﻿using System.Xml.Linq;
using static System.Console;

namespace DotnetUtils;

public static class Functions {
    public static string? getValue(string path, params string[] names) {
        if (names.Length == 0) { throw new InvalidOperationException(); }
        var parent = XDocument.Load(path).Element("Project")?.Element("PropertyGroup");
        return getValue(parent, names);
    }

    public static string? getValue(XElement? parent, params string[] names) {
        if (names.Length == 0) { throw new InvalidOperationException(); }
        XElement? elem = null;
        foreach (var name in names) {
            elem = parent?.Element(name);
            if (elem != null) { break; }
        }
        return elem?.Value;
    }

    public static int getChoice(int maxValue) => getChoice(0, maxValue);

    public static int getChoice(int minValue, int maxValue) {
        int result;
        do {
            Write("? ");
        } while (!int.TryParse(ReadLine(), out result) || result < minValue || result > maxValue);
        //WriteLine();
        return result;
    }


}

```

# `00_Utilities/DotnetUtils/DotnetUtils/Globals.cs`

该代码是一个名为"DotnetUtils"的命名空间，其中包含一个名为"Globals"的类。

该类中包含一个名为"LangData"的受保护的属性，该属性是一个名为"Dictionary<string, (string codefileExtension, string projExtension)>"的类型。

LangData中包含两个键值对，分别映射为"csharp"和"vbnet"两种编程语言。每个键值对包含两个属性，一个是代码文件扩展名，另一个是项目文件扩展名。

该代码的作用是定义了一个名为"Globals"的命名空间，其中包含一个名为"LangData"的受保护的属性，该属性包含两个键值对，分别映射为"csharp"和"vbnet"两种编程语言。每个键值对包含两个属性，一个是代码文件扩展名，另一个是项目文件扩展名。


```
﻿namespace DotnetUtils;

public static class Globals {
    public static readonly Dictionary<string, (string codefileExtension, string projExtension)> LangData = new() {
        { "csharp", ("cs", "csproj") },
        { "vbnet", ("vb", "vbproj") }
    };
}

```

# `00_Utilities/DotnetUtils/DotnetUtils/Methods.cs`

This is a class written in C# that implements the `IProcessStartInfo` interface, which is a part of the `System.Diagnostics` namespace.

This class appears to be used to start a new process, gather input from its standard input and/or standard output, and display any error messages it receives, as well as redirect its standard output and/or standard input.

The class includes several methods for doing this, as well as several instance variables.

The `RunProcessAsync` method takes a `Process` object and an optional `input` parameter, and starts a new process and waits for it to complete. It returns a `Task<ProcessResult>` object, which represents the result of the process.

The `RedirectStandardOutput` and `RedirectStandardError` methods redirect the standard output and standard error of the process, respectively.

The `Start` method starts the process, and the `Exited` method is called when the process has finished.

The `OutputDataReceived` and `ErrorDataReceived` methods are called when the standard input receives any data.

The `Error` property is a variable that is assigned in the `OutputDataReceived` method if the standard error of the process receives any data.

The `RedirectStandardInput` method redirects the standard input of the process.

Note that this class uses `System.Diagnostics`命名空間， which is part of the `System` namespace. This means that it is considered part of the `System` namespace, and is therefore accessible throughout your code using the `using System.Diagnostics` statement.


```
﻿using System.Diagnostics;

namespace DotnetUtils;

public static class Methods {
    public static ProcessResult RunProcess(string filename, string arguments) {
        var process = new Process() {
            StartInfo = {
                FileName = filename,
                Arguments = arguments,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            },
            EnableRaisingEvents = true
        };
        return RunProcess(process);
    }

    public static ProcessResult RunProcess(Process process, string input = "") {
        var (output, error) = ("", "");
        var (redirectOut, redirectErr) = (
            process.StartInfo.RedirectStandardOutput,
            process.StartInfo.RedirectStandardError
        );
        if (redirectOut) {
            process.OutputDataReceived += (s, ea) => output += ea.Data + "\n";
        }
        if (redirectErr) {
            process.ErrorDataReceived += (s, ea) => error += ea.Data + "\n";
        }

        if (!process.Start()) {
            throw new InvalidOperationException();
        };

        if (redirectOut) { process.BeginOutputReadLine(); }
        if (redirectErr) { process.BeginErrorReadLine(); }
        if (!string.IsNullOrEmpty(input)) {
            process.StandardInput.WriteLine(input);
            process.StandardInput.Close();
        }
        process.WaitForExit();
        return new ProcessResult(process.ExitCode, output, error);
    }

    public static Task<ProcessResult> RunProcessAsync(Process process, string input = "") {
        var tcs = new TaskCompletionSource<ProcessResult>();
        var (output, error) = ("", "");
        var (redirectOut, redirectErr) = (
            process.StartInfo.RedirectStandardOutput,
            process.StartInfo.RedirectStandardError
        );

        process.Exited += (s, e) => tcs.SetResult(new ProcessResult(process.ExitCode, output, error));

        if (redirectOut) {
            process.OutputDataReceived += (s, ea) => output += ea.Data + "\n";
        }
        if (redirectErr) {
            process.ErrorDataReceived += (s, ea) => error += ea.Data + "\n";
        }

        if (!process.Start()) {
            // what happens to the Exited event if process doesn't start successfully?
            throw new InvalidOperationException();
        }

        if (redirectOut) { process.BeginOutputReadLine(); }
        if (redirectErr) { process.BeginErrorReadLine(); }
        if (!string.IsNullOrEmpty(input)) {
            process.StandardInput.WriteLine(input);
            process.StandardInput.Close();
        }

        return tcs.Task;
    }
}

```

这段代码定义了一个名为 `ProcessResult` 的自定义数据类，它具有三个字段：`ExitCode`、`StdOut` 和 `StdErr`。

这个数据类的 `ToString()` 方法返回它的字段表达式的字符串表示形式。在这个例子中，`ToString()` 返回一个字符串，它由以下几部分组成：

1. 如果 `StdOut` 不是 `null` 或 empty，并且 `ExitCode` 大于0，那么输出一个换行符，并在其后跟上 `ExitCode` 的字符串表示形式。
2. 如果 `StdOut` 是 `null` 或empty，并且 `ExitCode` 大于0，那么输出 `ExitCode` 的字符串表示形式。
3. 如果 `ExitCode` 不大于0，那么输出 `ExitCode` 的字符串表示形式，并在其后跟 `StdErr` 的字符串表示形式。

例如，如果 `ExitCode` 是10，`StdOut` 是"Hello World"，`StdErr` 是"Error: no space left for栋格"，那么 `ToString()` 返回以下字符串：
```
10
Error: no space left for  world
```


```
public sealed record ProcessResult(int ExitCode, string StdOut, string StdErr) {
    public override string? ToString() =>
        StdOut +
        (StdOut is not (null or "") && ExitCode > 0 ? "\n" : "") +
        (ExitCode != 0 ?
            $"{ExitCode}\n{StdErr}" :
            "");
}

```