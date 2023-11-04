# BasicComputerGames源码解析 7

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript war.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "war"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:
```
	miniscript weekday.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:
```
	load "weekday"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:
```
	miniscript word.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:
```
	load "word"
	run
```

# Common Library

## Purpose

The primary purpose of this library is to implement common behaviours of the BASIC interpreter that impact gameplay, to
free coders porting the games to concentrate on the explicit game logic.

The behaviours implemented by this library are:

* Complex interactions involved in text input.
* Formatting of number in text output.
* Behaviour of the BASIC `RND(float)` PRNG function.

A secondary purpose is to provide common services that, with dependency injection, would allow a ported game to be
driven programmatically to permit full-game acceptance tests to be written.

The library is **NOT** intended to be:

* a repository for common game logic that is implemented in the BASIC code of the games; or
* a DSL allowing the BASIC code to be compiled in a different language environment with minimal changes. This implies
  that implementations of the above behaviours should use method names, etc, that are idiomatic of the specific
  language's normal routines for that behaviour.

## Text Input

The behaviour of the BASIC interpreter when accepting text input from the user is a major element of the original
gameplay experience that is seen to be valuable to maintain. This behaviour is complex and non-trivial to implement, and
is better implemented once for other developers to use so they can concentrate on the explicit game logic.

The text input/output behaviour can be investigated using a basic program such as:

**`BASIC_Tests/InputTest.bas`**

```basic
10 INPUT "Enter 3 numbers";A,B,C
20 PRINT "You entered: ";A;B;C
30 PRINT "--------------------------"
40 GOTO 10
```

The following transcript shows the use of this program, and some interesting behaviours of the BASIC interpreter INPUT
routine. There are some other behaviours which can be seen in the unit tests for the C# library implementation.

```dos
Enter 3 numbers? -1,2,3.141             <-- multiple numbers are separated by commas
You entered: -1  2  3.141
--------------------------
Enter 3 numbers? 1                      <-- ... or entered on separate lines
?? 2
?? 3
You entered:  1  2  3
--------------------------
Enter 3 numbers? 1,2                    <-- ... or both
?? 3
You entered:  1  2  3
--------------------------
Enter 3 numbers? 1,-2,3,4               <-- Extra input is ignore with a warning
!EXTRA INPUT IGNORED
You entered:  1 -2  3
--------------------------
Enter 3 numbers?   5  , 6.7, -8   ,10   <-- Whitespace around values is ignored
!EXTRA INPUT IGNORED
You entered:  5  6.7 -8
--------------------------
Enter 3 numbers? abcd,e,f               <-- Non-numeric entries must be retried
!NUMBER EXPECTED - RETRY INPUT LINE
? 1,2,abc                               <-- A single non-numeric invalidates the whole line
!NUMBER EXPECTED - RETRY INPUT LINE
? 1de,2.3f,10k,abcde                    <-- ... except for trailing non-digit chars  and extra input
!EXTRA INPUT IGNORED
You entered:  1  2.3  10
--------------------------
Enter 3 numbers? 1,"2,3",4              <-- Double-quotes enclose a single parsed value
You entered:  1  2  4
--------------------------
Enter 3 numbers? 1,2,"3                 <-- An unmatched double-quote crashes the interpreter
vintbas.exe: Mismatched inputbuf in inputVars
CallStack (from HasCallStack):
  error, called at src\Language\VintageBasic\Interpreter.hs:436:21 in main:Language.VintageBasic.Interpreter
```

I propose to ignore this last behaviour - the interpreter crash - and instead treat the end of the input line as the end
of a quoted value.  There are some additional behaviours to those shown above which can be seen in the unit tests for
the C# implementation of the library.

Note also that BASIC numeric variables store a single-precision floating point value, so numeric input functions should
return a value of that type.

### Implementation Notes

The usage of the `INPUT` command in the BASIC code of the games was analysed, with the variables used designated as
`num`, for a numeric variable (eg. `M1`), or `str`, for a string variable (eg. `C$`). The result of this usage analysis
across all game programs is:

Variable number and type|Count
---|---
str|137
str  str|2
num|187
num  num|27
num  num  num|7
num  num  num  num|1
num  num  num  num  num  num  num  num  num  num|1

The usage count is interesting, but not important. What is important is the variable type and number for each usage.
Implementers if the above behaviours do not need to cater for mixed variable types in their input routines (although the
BASIC interpreter does support this). Input routines also need to cater only for 1 or 2 string values, and 1, 2, 3, 4,
or 10 numeric values.

## Numbers in Text Output

As seen in the transcript above, the BASIC interpreter has some particular rules for formatting numbers in text output.
Negative numbers are printed with a leading negative sign (`-`) and a trailing space. Positive numbers are printed also
with the trailing space, but with a leading space in place of the sign character.

Additional formatting rules can be seen by running this program:

**`BASIC_Tests/OutputTest.bas`**

```basic
10 A=1: B=-2: C=0.7: D=123456789: E=-0.0000000001
20 PRINT "|";A;"|";B;"|";C;"|";D;"|";E;"|"
```

The output is:

```dos
| 1 |-2 | .7 | 1.2345679E+8 |-1.E-10 |
```

This clearly shows the leading and trailing spaces, but also shows that:

* numbers without an integer component are printed without a leading zero to the left of the decimal place;
* numbers above and below a certain magnitude are printed in scientific notation.

<!-- markdownlint-disable MD024 -->
### Implementation Notes
<!-- markdownlint-enable MD024 -->

I think the important piece of this number formatting behaviour, in terms of its impact on replicating the text output
of the original games, is the leading and trailing spaces. This should be the minimum behaviour supported for numeric
output. Additional formatting behaviours may be investigated and supported by library implementers as they choose.

## The BASIC `RND(float)` function

The [Vintage BASIC documentation](http://vintage-basic.net/downloads/Vintage_BASIC_Users_Guide.html) describes the
behaviour of the `RND(float)` function:

> Psuedorandom number generator. The behavior is different depending on the value passed. If the value is positive, the
> result will be a new random value between 0 and 1 (including 0 but not 1). If the value is zero, the result will be a
> repeat of the last random number generated. If the value is negative, it will be rounded down to the nearest integer
> and used to reseed the random number generator. Pseudorandom sequences can be repeated by reseeding with the same
> number.

This behaviour can be shown by the program:

**`BASIC_Tests/RndTest.bas`**

```basic
10 PRINT "1: ";RND(1);RND(1);RND(0);RND(0);RND(1)
20 PRINT "2: ";RND(-2);RND(1);RND(1);RND(1)
30 PRINT "3: ";RND(-5);RND(1);RND(1);RND(1)
40 PRINT "4: ";RND(-2);RND(1);RND(1);RND(1)
```

The output of this program is:

```dos
1:  .97369444  .44256502  .44256502  .44256502  .28549057    <-- Repeated values due to RND(0)
2:  .4986506  4.4510484E-2  .96231  .35997057
3:  .8113741  .13687313  6.1034977E-2  .7874807
4:  .4986506  4.4510484E-2  .96231  .35997057                <-- Same sequence as line 2 due to same seed
```

<!-- markdownlint-disable MD024 -->
### Implementation Notes
<!-- markdownlint-enable MD024 -->

While the BASIC `RND(x)` function always returns a number between 0 (inclusive) and 1 (exclusive) for positive non-zero
values of `x`, game porters would find it convenient for the library to include functions returning a random float or
integer in a range from an inclusive minimum to an exclusive maximum.

As one of the games, "Football", makes use of `RND(0)` with different scaling applied than the previous use of `RND(1)`,
a common library implementation should always generate a value between 0 and 1, and scale that for a function with a
range, so that a call to the equivalent of the `RND(0)` function can return the previous value between 0 and 1.


# Games.Common

This is the common library for C# and VB.Net ports of the games.

## Overview

### Game Input/Output

* `TextIO` is the main class which manages text input and output for a game. It take a `TextReader` and a `TextWriter` in
its constructor so it can be wired up in unit tests to test gameplay scenarios.
* `ConsoleIO` derives from `TextIO` and binds it to `System.Console.In` and `System.Console.Out`.
* `IReadWrite` is an interface implemented by `TextIO` which may be useful in some test scenarios.

```csharp
public interface IReadWrite
{
    // Reads a float value from input.
    float ReadNumber(string prompt);

    // Reads 2 float values from input.
    (float, float) Read2Numbers(string prompt);

    // Reads 3 float values from input.
    (float, float, float) Read3Numbers(string prompt);

    // Reads 4 float values from input.
    (float, float, float, float) Read4Numbers(string prompt);

    // Read numbers from input to fill an array.
    void ReadNumbers(string prompt, float[] values);

    // Reads a string value from input.
    string ReadString(string prompt);

    // Reads 2 string values from input.
    (string, string) Read2Strings(string prompt);

    // Writes a string to output.
    void Write(string message);

    // Writes a string to output, followed by a new-line.
    void WriteLine(string message = "");

    // Writes a float to output, formatted per the BASIC interpreter, with leading and trailing spaces.
    void Write(float value);

    // Writes a float to output, formatted per the BASIC interpreter, with leading and trailing spaces,
    // followed by a new-line.
    void WriteLine(float value);

    // Writes the contents of a Stream to output.
    void Write(Stream stream);}
```

### Random Number Generation

* `IRandom` is an interface that provides basic methods that parallel the 3 uses of BASIC's `RND(float)` function.
* `RandomNumberGenerator` is an implementation of `IRandom` built around `System.Random`.
* `IRandomExtensions` provides convenience extension methods for obtaining random numbers as `int` and also within a
  given range.

```csharp
public interface IRandom
{
    // Like RND(1), gets a random float such that 0 <= n < 1.
    float NextFloat();

    // Like RND(0), Gets the float returned by the previous call to NextFloat.
    float PreviousFloat();

    // Like RND(-x), Reseeds the random number generator.
    void Reseed(int seed);
}
```

Extension methods on `IRandom`:

```csharp
// Gets a random float such that 0 <= n < exclusiveMaximum.
float NextFloat(this IRandom random, float exclusiveMaximum);

// Gets a random float such that inclusiveMinimum <= n < exclusiveMaximum.
float NextFloat(this IRandom random, float inclusiveMinimum, float exclusiveMaximum);

// Gets a random int such that 0 <= n < exclusiveMaximum.
int Next(this IRandom random, int exclusiveMaximum);

// Gets a random int such that inclusiveMinimum <= n < exclusiveMaximum.
int Next(this IRandom random, int inclusiveMinimum, int exclusiveMaximum);

// Gets the previous unscaled float (between 0 and 1) scaled to a new range:
// 0 <= x < exclusiveMaximum.
float PreviousFloat(this IRandom random, float exclusiveMaximum);

// Gets the previous unscaled float (between 0 and 1) scaled to a new range:
// inclusiveMinimum <= n < exclusiveMaximum.
float PreviousFloat(this IRandom random, float inclusiveMinimum, float exclusiveMaximum);

// Gets the previous unscaled float (between 0 and 1) scaled to an int in a new range:
// 0 <= n < exclusiveMaximum.
int Previous(this IRandom random, int exclusiveMaximum);

// Gets the previous unscaled float (between 0 and 1) scaled to an int in a new range:
// inclusiveMinimum <= n < exclusiveMaximum.
int Previous(this IRandom random, int inclusiveMinimum, int exclusiveMaximum);
```

## C\# Usage

### Add Project Reference

Add the `Games.Common` project as a reference to the game project. For example, here's the reference from the C\# port
of `86_Target`:

```xml
<ItemGroup>
  <ProjectReference Include="..\..\00_Common\dotnet\Games.Common\Games.Common.csproj" />
</ItemGroup>
```

### C# Game Input/Output usage

A game can be encapsulated in a class which takes a `TextIO` instance in it's constructor:

```csharp
public class Game
{
    private readonly TextIO _io;

    public Game(TextIO io) => _io = io;

    public void Play()
    {
        var name = _io.ReadString("What is your name");
        var (cats, dogs) = _io.Read2Number($"Hello, {name}, how many pets do you have (cats, dogs)");
        _io.WriteLine($"So, {cats + dogs} pets in total, huh?");
    }
}
```

Then the entry point of the game program would look something like:

```csharp
var game = new Game(new ConsoleIO());
game.Play();
```

### C# Random Number Generator usage

```csharp
var io = new ConsoleIO();
var rng = new RandomNumberGenerator();
io.WriteLine(rng.NextFloat());           // 0.1234, for example
io.WriteLine(rng.NextFloat());           // 0.6, for example
io.WriteLine(rng.PreviousFloat());       // 0.6, repeats previous
io.WriteLine(rng.PreviousFloat(0, 10));  // 6,   repeats previous value, but scaled to new range
```

### C# Unit Test usage

`TextIO` can be initialised with a `StringReader` and `StringWriter` to enable testing. For example, given the `Game`
class above:

```csharp
var reader = new StringReader("Joe Bloggs\r\n4\n\r5");
var writer = new StringWriter();
var game = new Game(new TextIO(reader, writer))

game.Play();

writer.ToString().Should().BeEquivalentTo(
    "What is your name? Hello, Joe Bloggs, how many pets do you have (cats, dogs)? ?? So, 9 pets in total, huh?");
```

Note the lack of line breaks in the expected output, because during game play the line breaks come from the text input.

Of course, `IReadWrite` can also be mocked for simple test scenarios.

## VB.Net Usage

*To be provided*


# `00_Common/dotnet/Games.Common/_InternalsVisibleTo.cs`

这段代码是一个 Common.Assembly 类型的印章，表示它是一个独立的 assembly 单元，可以被其他程序或 Ferdinand划分为一个整顿，供给调试，但不被调试器公开。它使用 System.Runtime.CompilerServices命名空间中的特性，以确保在命令行构造器中包含右锁链的导入，以便于在秋天进行骨架复制。

具体而言，该代码允许子程序在需要时动态加载 Common.Assembly 类型，允许其在程序中使用，但不允许修改其内容或以任何方式暴露其实现。这个Assembly还包含一个静态构造函数，在程序加载时执行，确保其内部VisibleTo("Games.Common.Test") 声明，使"Games.Common.Test"类中的静态成员在程序中可见。


```
using System.Runtime.CompilerServices;

[assembly:InternalsVisibleTo("Games.Common.Test")]

```

# `00_Common/dotnet/Games.Common/IO/ConsoleIO.cs`

这段代码是一个名为 `ConsoleIO` 的类，实现了 `IReadWrite` 接口，用于从标准输入(即 `Console.In`)中读取输入，并将输入输出到标准输出(即 `Console.Out`)。

具体来说，该类继承自 `TextIO` 类，它提供了 `base` 方法，使用了 `Console.In` 和 `Console.Out` 两个方法来获取输入和输出。另外，该类还重写了 `ReadCharacter` 方法，这个方法从标准输入中读取一个字符，并返回它的 ASCII 码。

由于该类是 `sealed` 的，因此它的实例化必须通过 `new` 关键字来传递一个实例。这个实例将继承自 `TextIO` 类，并且可以使用 `TextIO` 的常用方法来读写输入输出。


```
using System;

namespace Games.Common.IO;

/// <summary>
/// An implementation of <see cref="IReadWrite" /> with input begin read for STDIN and output being written to
/// STDOUT.
/// </summary>
public sealed class ConsoleIO : TextIO
{
    public ConsoleIO()
        : base(Console.In, Console.Out)
    {
    }

    public override char ReadCharacter() => Console.ReadKey(intercept: true).KeyChar;
}

```

# `00_Common/dotnet/Games.Common/IO/InsufficientInputException.cs`



这段代码定义了一个名为 "InsufficientInputException" 的异常类，继承自 Exception 类。该异常类有两个构造函数，一个是无参构造函数，另一个是在构造函数中传入一个字符串参数。

该异常类的字符串参数 "Insufficient input was supplied" 是用来描述异常情况的字符串，会在异常类被创建时根据构造函数的顺序出现在异常类的第一行。

该异常类的方法是空的方法，不会执行任何逻辑。

使用该异常类的条件是：在代码中使用 "throw new InsufficientInputException();" 语句，就可以抛出该异常，让程序更加健壮。


```
namespace Games.Common.IO;

public class InsufficientInputException : Exception
{
    public InsufficientInputException()
        : base("Insufficient input was supplied")
    {
    }
}

```

# `00_Common/dotnet/Games.Common/IO/IReadWrite.cs`



This is a code snippet written in C# that defines a class called `WriteColor` that writes color values to an `WriteColor` object.

The `WriteColor` class has several overloads for the `WriteColor` method, which takes a single parameter named `color` of type `Color`. The first overload accepts a `Color` object and writes it to the output. The second overload accepts a `Color` object and a `Number` object and writes the color value to the output. The third overload accepts a `string` object and writes the specified format to the output.

The `WriteColor` class also has a fourth overload called `WriteFormattedString`, which accepts a `string` object and a format string, and writes the specified format to the output.

It is important to note that the `WriteColor` class cannot be used as a regular class member because it requires you to create it on the spot and instance it, but you can use the `Color` class as a class member.

Please let me know if you have any questions about this or need further clarification.


```
using Games.Common.Numbers;

namespace Games.Common.IO;

/// <summary>
/// Provides for input and output of strings and numbers.
/// </summary>
public interface IReadWrite
{
    /// <summary>
    /// Reads a character from input.
    /// </summary>
    /// <returns>The character read.</returns>
    char ReadCharacter();

    /// <summary>
    /// Reads a <see cref="float" /> value from input.
    /// </summary>
    /// <param name="prompt">The text to display to prompt for the value.</param>
    /// <returns>A <see cref="float" />, being the value entered.</returns>
    float ReadNumber(string prompt);

    /// <summary>
    /// Reads 2 <see cref="float" /> values from input.
    /// </summary>
    /// <param name="prompt">The text to display to prompt for the values.</param>
    /// <returns>A <see cref="ValueTuple{float, float}" />, being the values entered.</returns>
    (float, float) Read2Numbers(string prompt);

    /// <summary>
    /// Reads 3 <see cref="float" /> values from input.
    /// </summary>
    /// <param name="prompt">The text to display to prompt for the values.</param>
    /// <returns>A <see cref="ValueTuple{float, float, float}" />, being the values entered.</returns>
    (float, float, float) Read3Numbers(string prompt);

    /// <summary>
    /// Reads 4 <see cref="float" /> values from input.
    /// </summary>
    /// <param name="prompt">The text to display to prompt for the values.</param>
    /// <returns>A <see cref="ValueTuple{float, float, float, float}" />, being the values entered.</returns>
    (float, float, float, float) Read4Numbers(string prompt);

    /// <summary>
    /// Read numbers from input to fill an array.
    /// </summary>
    /// <param name="prompt">The text to display to prompt for the values.</param>
    /// <param name="values">A <see cref="float[]" /> to be filled with values from input.</param>
    void ReadNumbers(string prompt, float[] values);

    /// <summary>
    /// Reads a <see cref="string" /> value from input.
    /// </summary>
    /// <param name="prompt">The text to display to prompt for the value.</param>
    /// <returns>A <see cref="string" />, being the value entered.</returns>
    string ReadString(string prompt);

    /// <summary>
    /// Reads 2 <see cref="string" /> values from input.
    /// </summary>
    /// <param name="prompt">The text to display to prompt for the values.</param>
    /// <returns>A <see cref="ValueTuple{string, string}" />, being the values entered.</returns>
    (string, string) Read2Strings(string prompt);

    /// <summary>
    /// Writes a <see cref="string" /> to output.
    /// </summary>
    /// <param name="message">The <see cref="string" /> to be written.</param>
    void Write(string message);

    /// <summary>
    /// Writes a <see cref="string" /> to output, followed by a new-line.
    /// </summary>
    /// <param name="message">The <see cref="string" /> to be written.</param>
    void WriteLine(string message = "");

    /// <summary>
    /// Writes a <see cref="Number" /> to output.
    /// </summary>
    /// <param name="value">The <see cref="Number" /> to be written.</param>
    void Write(Number value);

    /// <summary>
    /// Writes a <see cref="Number" /> to output.
    /// </summary>
    /// <param name="value">The <see cref="Number" /> to be written.</param>
    void WriteLine(Number value);

    /// <summary>
    /// Writes an <see cref="object" /> to output.
    /// </summary>
    /// <param name="value">The <see cref="object" /> to be written.</param>
    void Write(object value);

    /// <summary>
    /// Writes an <see cref="object" /> to output.
    /// </summary>
    /// <param name="value">The <see cref="object" /> to be written.</param>
    void WriteLine(object value);

    /// <summary>
    /// Writes a formatted string to output.
    /// </summary>
    /// <param name="format">The format <see cref="string" /> to be written.</param>
    /// <param name="value">The values to be inserted into the format.</param>
    void Write(string format, params object[] values);

    /// <summary>
    /// Writes a formatted string to output followed by a new-line.
    /// </summary>
    /// <param name="format">The format <see cref="string" /> to be written.</param>
    /// <param name="value">The values to be inserted into the format.</param>
    void WriteLine(string format, params object[] values);

    /// <summary>
    /// Writes the contents of a <see cref="Stream" /> to output.
    /// </summary>
    /// <param name="stream">The <see cref="Stream" /> to be written.</param>
    void Write(Stream stream, bool keepOpen = false);
}

```

# `00_Common/dotnet/Games.Common/IO/Strings.cs`

这段代码定义了一个名为 "Games.Common.IO.Strings" 的命名空间，其中包含一个名为 "Strings" 的内部类。

这个内部类定义了两个字符串变量，一个名为 "NumberExpected"，另一个名为 "ExtraInput"。

"NumberExpected" 的值为 "!Number expected - retry input line"，其中的感叹号表示这个值是一个错误提示信息，而不是一个预期的输入。

"ExtraInput" 的值为 "!Extra input ignored"，其中的感叹号表示这个值是一个错误提示信息，而不是一个预期的输入。

这个内部类可以被用来在应用程序中处理输入字符串，确保它们符合预期的格式。例如，在应用程序中，可以通过这个内部类来检查用户输入是否为数字，或者是否包含 "!Extra" 这样的前缀。


```
namespace Games.Common.IO;

internal static class Strings
{
    internal const string NumberExpected = "!Number expected - retry input line";
    internal const string ExtraInput = "!Extra input ignored";
}

```

# `00_Common/dotnet/Games.Common/IO/TextIO.cs`

This code appears to be a classes library for a command-line tool. The library includes several methods for reading input from the user, such as reading a string, two strings, or a series of strings. The methods all take a string prompt as the input, and return the value entered by the user or an ArgumentOutOfRangeException if the value is negative or the prompt is empty.

The library also includes a method for reading input from the user's console, and several methods for writing output to the console or a file. The `Write` method can take a variety of arguments, including a number, a string, or an object.

The `ReadString` method reads a line of input from the user and returns the first part of the line. The `Read2Strings` method reads two strings from the user and returns them.

The `throw new ArgumentOutOfRangeException` method is used to handle negative values.

It is recommended to use the `using` statement to close the resources, if you plan to keep the class created by the `CreateObject` method, for example:
```
using System;
using System.IO;
using System.Text;

public class MyClass
{
   public MyClass()
   {
       // initialize the resources
   }
   public void WriteOutput(string prompt)
   {
       // code to write the prompt to the console or a file
   }
}
```
It is also recommended to add a `、` before the variable name, for example:
```
using System;
using System.IO;
using System.Text;

public class MyClass
{
   public string WriteOutput(string prompt)
   {
       // code to write the prompt to the console or a file
   }
}
```
It is also good practice to close the resources, when you are done with them, for example:
```
using System;
using System.IO;
using System.Text;

public class MyClass
{
   public string WriteOutput(string prompt)
   {
       // code to write the prompt to the console or a file
   }
}

public class Program
{
   public static void Main(string[] args)
   {
       MyClass myClass = new MyClass();
       myClass.WriteOutput("This is a prompt");
       myClass.WriteOutput("I am a test");
       myClass.WriteOutput(new int { Message = "Hello World" });
   }
}
```
It is also a good practice to remove unnecessary code, like `if (!keepOpen)` from the `Write` method, it is not needed in the examples provided.


```
using Games.Common.Numbers;

namespace Games.Common.IO;

/// <inheritdoc />
/// <summary>
/// Implements <see cref="IReadWrite" /> with input read from a <see cref="TextReader" /> and output written to a
/// <see cref="TextWriter" />.
/// </summary>
/// <remarks>
/// This implementation reproduces the Vintage BASIC input experience, prompting multiple times when partial input
/// supplied, rejecting non-numeric input as needed, warning about extra input being ignored, etc.
/// </remarks>
public class TextIO : IReadWrite
{
    private readonly TextReader _input;
    private readonly TextWriter _output;
    private readonly TokenReader _stringTokenReader;
    private readonly TokenReader _numberTokenReader;

    public TextIO(TextReader input, TextWriter output)
    {
        _input = input ?? throw new ArgumentNullException(nameof(input));
        _output = output ?? throw new ArgumentNullException(nameof(output));
        _stringTokenReader = TokenReader.ForStrings(this);
        _numberTokenReader = TokenReader.ForNumbers(this);
    }

    public virtual char ReadCharacter()
    {
        while(true)
        {
            var ch = _input.Read();
            if (ch != -1) { return (char)ch; }
        }
    }

    public float ReadNumber(string prompt) => ReadNumbers(prompt, 1)[0];

    public (float, float) Read2Numbers(string prompt)
    {
        var numbers = ReadNumbers(prompt, 2);
        return (numbers[0], numbers[1]);
    }

    public (float, float, float) Read3Numbers(string prompt)
    {
        var numbers = ReadNumbers(prompt, 3);
        return (numbers[0], numbers[1], numbers[2]);
    }

    public (float, float, float, float) Read4Numbers(string prompt)
    {
        var numbers = ReadNumbers(prompt, 4);
        return (numbers[0], numbers[1], numbers[2], numbers[3]);
    }

    public void ReadNumbers(string prompt, float[] values)
    {
        if (values.Length == 0)
        {
            throw new ArgumentException($"'{nameof(values)}' must have a non-zero length.", nameof(values));
        }

        var numbers = _numberTokenReader.ReadTokens(prompt, (uint)values.Length).Select(t => t.Number).ToArray();
        numbers.CopyTo(values.AsSpan());
    }

    private IReadOnlyList<float> ReadNumbers(string prompt, uint quantity) =>
        (quantity > 0)
            ? _numberTokenReader.ReadTokens(prompt, quantity).Select(t => t.Number).ToList()
            : throw new ArgumentOutOfRangeException(
                nameof(quantity),
                $"'{nameof(quantity)}' must be greater than zero.");

    public string ReadString(string prompt)
    {
        return ReadStrings(prompt, 1)[0];
    }

    public (string, string) Read2Strings(string prompt)
    {
        var values = ReadStrings(prompt, 2);
        return (values[0], values[1]);
    }

    private IReadOnlyList<string> ReadStrings(string prompt, uint quantityRequired) =>
        _stringTokenReader.ReadTokens(prompt, quantityRequired).Select(t => t.String).ToList();

    internal string ReadLine(string prompt)
    {
        Write(prompt + "? ");
        return _input.ReadLine() ?? throw new InsufficientInputException();
    }

    public void Write(string value) => _output.Write(value);

    public void WriteLine(string value = "") => _output.WriteLine(value);

    public void Write(Number value) => _output.Write(value.ToString());

    public void WriteLine(Number value) => _output.WriteLine(value.ToString());

    public void Write(object value) => _output.Write(value.ToString());

    public void WriteLine(object value) => _output.WriteLine(value.ToString());

    public void Write(string format, params object[] values) => _output.Write(format, values);

    public void WriteLine(string format, params object[] values) => _output.WriteLine(format, values);

    public void Write(Stream stream, bool keepOpen = false)
    {
        using var reader = new StreamReader(stream);
        while (!reader.EndOfStream)
        {
            _output.WriteLine(reader.ReadLine());
        }

        if (!keepOpen) { stream?.Dispose(); }
    }

    private string GetString(float value) => value < 0 ? $"{value} " : $" {value} ";
}

```

# `00_Common/dotnet/Games.Common/IO/Token.cs`



这段代码是一个字符串工具类，名为Token，其作用是验证一个字符串是否为数字，如果为数字，则可以将其转换为浮点数，并返回该数字。

具体来说，代码首先定义了一个内部类Token，其中包含一个字符串变量Value，一个表示数字是否可读的布尔变量IsNumber，以及一个浮点数变量Number。

Token类中，定义了一个正则表达式pattern，用于匹配字符串中的数字。这个正则表达式要求数字必须以数字符号开始，然后可能包含一个小数点，最后可能包含一个或两个数字。

Token类的Builder类，用于在构建Token对象时执行某些操作，例如在构建Token对象之前，可能需要先清空Builder对象的一些成员变量，例如IsQuoted和TrailingWhiteSpaceCount。

使用Token的示例代码，可以将其作为一个可读的数学表达式，用于计算数学中的表达式，例如将一个复杂的数学表达式拆分为简单的数学表达式，并将它们代入求值，然后将结果合并。


```
using System.Text;
using System.Text.RegularExpressions;

namespace Games.Common.IO;

internal class Token
{
    private static readonly Regex _numberPattern = new(@"^[+\-]?\d*(\.\d*)?([eE][+\-]?\d*)?");

    internal Token(string value)
    {
        String = value;

        var match = _numberPattern.Match(String);

        IsNumber = float.TryParse(match.Value, out var number);
        Number = (IsNumber, number) switch
        {
            (false, _) => float.NaN,
            (true, float.PositiveInfinity) => float.MaxValue,
            (true, float.NegativeInfinity) => float.MinValue,
            (true, _) => number
        };
    }

    public string String { get; }
    public bool IsNumber { get; }
    public float Number { get; }

    public override string ToString() => String;

    internal class Builder
    {
        private readonly StringBuilder _builder = new();
        private bool _isQuoted;
        private int _trailingWhiteSpaceCount;

        public Builder Append(char character)
        {
            _builder.Append(character);

            _trailingWhiteSpaceCount = char.IsWhiteSpace(character) ? _trailingWhiteSpaceCount + 1 : 0;

            return this;
        }

        public Builder SetIsQuoted()
        {
            _isQuoted = true;
            return this;
        }

        public Token Build()
        {
            if (!_isQuoted) { _builder.Length -= _trailingWhiteSpaceCount; }
            return new Token(_builder.ToString());
        }
    }
}

```

# `00_Common/dotnet/Games.Common/IO/Tokenizer.cs`

This appears to be a description of a JSON parsing pipeline that uses the Jim我问學的算法 (G juice) to parse JSON data. Jim我问学是一个基于解析器的Java库，它可以在短时间内处理大量JSON数据。

具体来说，这段代码描述了从给定的输入字符串开始，按照指定的分隔符、引号或空白符，将输入字符流转换为相应的Token结构，并将这些Token添加到一个输出字符串中。

在这个过程中，G juice提供了四个主要的类：LookForStartOfTokenState、InQuotedTokenState、InTokenState和ExpectSeparatorState。这些类实现了ITokenizerState接口，提供了对输入字符流的处理和相应的Token结构。在处理输入字符流时，这些类会根据指定的分隔符、引号或空白符，将输入字符流转换为相应的Token结构，并将这些Token添加到输出字符串中。


```
using System;
using System.Collections.Generic;

namespace Games.Common.IO;

/// <summary>
/// A simple state machine which parses tokens from a line of input.
/// </summary>
internal class Tokenizer
{
    private const char Quote = '"';
    private const char Separator = ',';

    private readonly Queue<char> _characters;

    private Tokenizer(string input) => _characters = new Queue<char>(input);

    public static IEnumerable<Token> ParseTokens(string input)
    {
        if (input is null) { throw new ArgumentNullException(nameof(input)); }

        return new Tokenizer(input).ParseTokens();
    }

    private IEnumerable<Token> ParseTokens()
    {
        while (true)
        {
            var (token, isLastToken) = Consume(_characters);
            yield return token;

            if (isLastToken) { break; }
        }
    }

    public (Token, bool) Consume(Queue<char> characters)
    {
        var tokenBuilder = new Token.Builder();
        var state = ITokenizerState.LookForStartOfToken;

        while (characters.TryDequeue(out var character))
        {
            (state, tokenBuilder) = state.Consume(character, tokenBuilder);
            if (state is AtEndOfTokenState) { return (tokenBuilder.Build(), false); }
        }

        return (tokenBuilder.Build(), true);
    }

    private interface ITokenizerState
    {
        public static ITokenizerState LookForStartOfToken { get; } = new LookForStartOfTokenState();

        (ITokenizerState, Token.Builder) Consume(char character, Token.Builder tokenBuilder);
    }

    private struct LookForStartOfTokenState : ITokenizerState
    {
        public (ITokenizerState, Token.Builder) Consume(char character, Token.Builder tokenBuilder) =>
            character switch
            {
                Separator => (new AtEndOfTokenState(), tokenBuilder),
                Quote => (new InQuotedTokenState(), tokenBuilder.SetIsQuoted()),
                _ when char.IsWhiteSpace(character) => (this, tokenBuilder),
                _ => (new InTokenState(), tokenBuilder.Append(character))
            };
    }

    private struct InTokenState : ITokenizerState
    {
        public (ITokenizerState, Token.Builder) Consume(char character, Token.Builder tokenBuilder) =>
            character == Separator
                ? (new AtEndOfTokenState(), tokenBuilder)
                : (this, tokenBuilder.Append(character));
    }

    private struct InQuotedTokenState : ITokenizerState
    {
        public (ITokenizerState, Token.Builder) Consume(char character, Token.Builder tokenBuilder) =>
            character == Quote
                ? (new ExpectSeparatorState(), tokenBuilder)
                : (this, tokenBuilder.Append(character));
    }

    private struct ExpectSeparatorState : ITokenizerState
    {
        public (ITokenizerState, Token.Builder) Consume(char character, Token.Builder tokenBuilder) =>
            character == Separator
                ? (new AtEndOfTokenState(), tokenBuilder)
                : (new IgnoreRestOfLineState(), tokenBuilder);
    }

    private struct IgnoreRestOfLineState : ITokenizerState
    {
        public (ITokenizerState, Token.Builder) Consume(char character, Token.Builder tokenBuilder) =>
            (this, tokenBuilder);
    }

    private struct AtEndOfTokenState : ITokenizerState
    {
        public (ITokenizerState, Token.Builder) Consume(char character, Token.Builder tokenBuilder) =>
            throw new InvalidOperationException();
    }
}

```

# `00_Common/dotnet/Games.Common/IO/TokenReader.cs`

This is a class written in C# that reads a sequence of tokens from the user. The tokens are read one at a time and only the valid tokens are added to the list of tokens. The method `ReadValidTokens` reads a maximum number of tokens, up to `maxCount`, and returns them. The method `ReadLineOfTokens` reads up to `maxCount` tokens from an input line.


```
using System;
using System.Collections.Generic;
using System.Linq;
using static Games.Common.IO.Strings;

namespace Games.Common.IO;

/// <summary>
/// Reads from input and assembles a given number of values, or tokens, possibly over a number of input lines.
/// </summary>
internal class TokenReader
{
    private readonly TextIO _io;
    private readonly Predicate<Token> _isTokenValid;

    private TokenReader(TextIO io, Predicate<Token> isTokenValid)
    {
        _io = io;
        _isTokenValid = isTokenValid ?? (t => true);
    }

    /// <summary>
    /// Creates a <see cref="TokenReader" /> which reads string tokens.
    /// </summary>
    /// <param name="io">A <see cref="TextIO" /> instance.</param>
    /// <returns>The new <see cref="TokenReader" /> instance.</returns>
    public static TokenReader ForStrings(TextIO io) => new(io, t => true);

    /// <summary>
    /// Creates a <see cref="TokenReader" /> which reads tokens and validates that they can be parsed as numbers.
    /// </summary>
    /// <param name="io">A <see cref="TextIO" /> instance.</param>
    /// <returns>The new <see cref="TokenReader" /> instance.</returns>
    public static TokenReader ForNumbers(TextIO io) => new(io, t => t.IsNumber);

    /// <summary>
    /// Reads valid tokens from one or more input lines and builds a list with the required quantity.
    /// </summary>
    /// <param name="prompt">The string used to prompt the user for input.</param>
    /// <param name="quantityNeeded">The number of tokens required.</param>
    /// <returns>The sequence of tokens read.</returns>
    public IEnumerable<Token> ReadTokens(string prompt, uint quantityNeeded)
    {
        if (quantityNeeded == 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(quantityNeeded),
                $"'{nameof(quantityNeeded)}' must be greater than zero.");
        }

        var tokens = new List<Token>();

        while (tokens.Count < quantityNeeded)
        {
            tokens.AddRange(ReadValidTokens(prompt, quantityNeeded - (uint)tokens.Count));
            prompt = "?";
        }

        return tokens;
    }

    /// <summary>
    /// Reads a line of tokens, up to <paramref name="maxCount" />, and rejects the line if any are invalid.
    /// </summary>
    /// <param name="prompt">The string used to prompt the user for input.</param>
    /// <param name="maxCount">The maximum number of tokens to read.</param>
    /// <returns>The sequence of tokens read.</returns>
    private IEnumerable<Token> ReadValidTokens(string prompt, uint maxCount)
    {
        while (true)
        {
            var tokensValid = true;
            var tokens = new List<Token>();
            foreach (var token in ReadLineOfTokens(prompt, maxCount))
            {
                if (!_isTokenValid(token))
                {
                    _io.WriteLine(NumberExpected);
                    tokensValid = false;
                    prompt = "";
                    break;
                }

                tokens.Add(token);
            }

            if (tokensValid) { return tokens; }
        }
    }

    /// <summary>
    /// Lazily reads up to <paramref name="maxCount" /> tokens from an input line.
    /// </summary>
    /// <param name="prompt">The string used to prompt the user for input.</param>
    /// <param name="maxCount">The maximum number of tokens to read.</param>
    /// <returns></returns>
    private IEnumerable<Token> ReadLineOfTokens(string prompt, uint maxCount)
    {
        var tokenCount = 0;

        foreach (var token in Tokenizer.ParseTokens(_io.ReadLine(prompt)))
        {
            if (++tokenCount > maxCount)
            {
                _io.WriteLine(ExtraInput);
                break;
            }

            yield return token;
        }
    }
}

```

# `00_Common/dotnet/Games.Common/Numbers/Number.cs`

这段代码定义了一个名为Number的结构体，其成员为单精度浮点数_value，用于表示一个数值。

Number构造函数接受一个float类型的参数value，将其赋值给_value，构造Number实例。

Number包含两个名为operator float和operator Number的成员函数，分别用于将Number实例转换为float类型和将float类型转换为Number实例。

Number还包含一个名为ToString的成员函数，用于将Number实例转换为字符串格式，并使用三引号将结果返回。这个函数的实现比较简单，直接将_value转换为字符串，并加入"("和")"来表示括号。

最后，在命名空间中定义了Number命名空间，以便在需要使用这个命名空间的情况下可以轻松地使用它。


```
namespace Games.Common.Numbers;

/// <summary>
/// A single-precision floating-point number with string formatting equivalent to the BASIC interpreter.
/// </summary>
public struct Number
{
    private readonly float _value;

    public Number (float value)
    {
        _value = value;
    }

    public static implicit operator float(Number value) => value._value;

    public static implicit operator Number(float value) => new Number(value);

    public override string ToString() => _value < 0 ? $"{_value} " : $" {_value} ";
}

```

# `00_Common/dotnet/Games.Common/Randomness/IRandom.cs`

这段代码定义了一个 namespace Games.Common.Randomness，其中定义了一个 interface IRandom，描述了一个 random number generator。

这个 interface 的 NextFloat() 方法接受一个 float 类型的参数，并返回一个 0 到 1 之间的随机 float 值。

这个 interface 的 PreviousFloat() 方法返回 previous random number，即先前的 random number。

这个 interface 的 Reseed() 方法接受一个 integer 类型的参数，表示随机数种子，并使用该种子重启随机数生成器的种子。

通过实现这个 interface，可以访问这个 random number generator，生成 random number 值，并在需要时重启生成器。


```
namespace Games.Common.Randomness;

/// <summary>
/// Provides access to a random number generator
/// </summary>
public interface IRandom
{
    /// <summary>
    /// Gets a random <see cref="float" /> such that 0 &lt;= n &lt; 1.
    /// </summary>
    /// <returns>The random number.</returns>
    float NextFloat();

    /// <summary>
    /// Gets the <see cref="float" /> returned by the previous call to <see cref="NextFloat" />.
    /// </summary>
    /// <returns>The previous random number.</returns>
    float PreviousFloat();

    /// <summary>
    /// Reseeds the random number generator.
    /// </summary>
    /// <param name="seed">The seed.</param>
    void Reseed(int seed);
}

```

# `00_Common/dotnet/Games.Common/Randomness/IRandomExtensions.cs`

This is a C# class that includes several methods for scaling random numbers to different ranges.

The `Scale` method takes two parameters: `zeroToOne`, which represents the range between 0 and 1 that the random number will be scaled to, and `exclusiveMaximum`, which is the exclusive maximum value (i.e., the largest number that can be within the specified range). This method returns the scaled random number.

The `Previous` method takes three parameters: `random`, `inclusiveMinimum`, and `exclusiveMaximum`, which represent the current random number, the minimum inclusive value, and the maximum inclusive value, respectively. This method returns the previous unscaled random number (i.e., the random number before applying the scaling).

The `Previous` method can be overridden by another method with the same name that takes only two parameters, with the same behavior as `Previous`.

The `ToInt` method is a helper method for converting a random number to its corresponding integer value.

Overall, this class provides a way to generate random numbers within a specified range by scaling them to different values.


```
using System;

namespace Games.Common.Randomness;

/// <summary>
/// Provides extension methods to <see cref="IRandom" /> providing random numbers in a given range.
/// </summary>
/// <value></value>
public static class IRandomExtensions
{
    /// <summary>
    /// Gets a random <see cref="float" /> such that 0 &lt;= n &lt; exclusiveMaximum.
    /// </summary>
    /// <returns>The random number.</returns>
    public static float NextFloat(this IRandom random, float exclusiveMaximum) =>
        Scale(random.NextFloat(), exclusiveMaximum);

    /// <summary>
    /// Gets a random <see cref="float" /> such that inclusiveMinimum &lt;= n &lt; exclusiveMaximum.
    /// </summary>
    /// <returns>The random number.</returns>
    public static float NextFloat(this IRandom random, float inclusiveMinimum, float exclusiveMaximum) =>
        Scale(random.NextFloat(), inclusiveMinimum, exclusiveMaximum);

    /// <summary>
    /// Gets a random <see cref="int" /> such that 0 &lt;= n &lt; exclusiveMaximum.
    /// </summary>
    /// <returns>The random number.</returns>
    public static int Next(this IRandom random, int exclusiveMaximum) => ToInt(random.NextFloat(exclusiveMaximum));

    /// <summary>
    /// Gets a random <see cref="int" /> such that inclusiveMinimum &lt;= n &lt; exclusiveMaximum.
    /// </summary>
    /// <returns>The random number.</returns>
    public static int Next(this IRandom random, int inclusiveMinimum, int exclusiveMaximum) =>
        ToInt(random.NextFloat(inclusiveMinimum, exclusiveMaximum));

    /// <summary>
    /// Gets the previous unscaled <see cref="float" /> (between 0 and 1) scaled to a new range:
    /// 0 &lt;= x &lt; <paramref name="exclusiveMaximum" />.
    /// </summary>
    /// <returns>The random number.</returns>
    public static float PreviousFloat(this IRandom random, float exclusiveMaximum) =>
        Scale(random.PreviousFloat(), exclusiveMaximum);

    /// <summary>
    /// Gets the previous unscaled <see cref="float" /> (between 0 and 1) scaled to a new range:
    /// <paramref name="inclusiveMinimum" /> &lt;= n &lt; <paramref name="exclusiveMaximum" />.
    /// </summary>
    /// <returns>The random number.</returns>
    public static float PreviousFloat(this IRandom random, float inclusiveMinimum, float exclusiveMaximum) =>
        Scale(random.PreviousFloat(), inclusiveMinimum, exclusiveMaximum);

    /// <summary>
    /// Gets the previous unscaled <see cref="float" /> (between 0 and 1) scaled to an <see cref="int" /> in a new
    /// range: 0 &lt;= n &lt; <paramref name="exclusiveMaximum" />.
    /// </summary>
    /// <returns>The random number.</returns>
    public static int Previous(this IRandom random, int exclusiveMaximum) =>
        ToInt(random.PreviousFloat(exclusiveMaximum));

    /// <summary>
    /// Gets the previous unscaled <see cref="float" /> (between 0 and 1) scaled to an <see cref="int" /> in a new
    /// range: <paramref name="inclusiveMinimum" /> &lt;= n &lt; <paramref name="exclusiveMaximum" />.
    /// <returns>The random number.</returns>
    public static int Previous(this IRandom random, int inclusiveMinimum, int exclusiveMaximum) =>
        ToInt(random.PreviousFloat(inclusiveMinimum, exclusiveMaximum));

    private static float Scale(float zeroToOne, float exclusiveMaximum)
    {
        if (exclusiveMaximum <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(exclusiveMaximum), "Must be greater than 0.");
        }

        return Scale(zeroToOne, 0, exclusiveMaximum);
    }

    private static float Scale(float zeroToOne, float inclusiveMinimum, float exclusiveMaximum)
    {
        if (exclusiveMaximum <= inclusiveMinimum)
        {
            throw new ArgumentOutOfRangeException(nameof(exclusiveMaximum), "Must be greater than inclusiveMinimum.");
        }

        var range = exclusiveMaximum - inclusiveMinimum;
        return zeroToOne * range + inclusiveMinimum;
    }

    private static int ToInt(float value) => (int)Math.Floor(value);
}

```

# `00_Common/dotnet/Games.Common/Randomness/RandomNumberGenerator.cs`

这段代码是一个自定义的随机数生成器，实现了`IRandom`接口。它包含一个私有变量`_random`和一个私有变量`_previous`，分别用于存储当前随机数和上一次随机数。

在`RandomNumberGenerator`类中，`_random`变量在构造函数中生成一个基于当前时间的随机数，时间间隔为1秒。然后，`_previous`变量保存了上一次随机数，用于在每次生成随机数时减少随机数的不确定性。

接着，`NextFloat()`方法返回了上一次随机数，用于下一次生成随机数时的起点。`PreviousFloat()`方法返回了上一次随机数，用于在生成随机数时减少上一次随机数对当前随机数的影响。`Reseed()`方法用于在需要时重新生成随机数，它接受一个整数作为参数，表示要重新生成随机数的种子。

总的来说，这段代码提供了一个简单的随机数生成器，可以在需要时产生一个随机的随机数，用于各种随机数生成场景。


```
using System;

namespace Games.Common.Randomness;

/// <inheritdoc />
public class RandomNumberGenerator : IRandom
{
    private Random _random;
    private float _previous;

    public RandomNumberGenerator()
    {
        // The BASIC RNG is seeded based on time with a 1 second resolution
        _random = new Random((int)(DateTime.UtcNow.Ticks / TimeSpan.TicksPerSecond));
    }

    public float NextFloat() => _previous = (float)_random.NextDouble();

    public float PreviousFloat() => _previous;

    public void Reseed(int seed) => _random = new Random(seed);
}

```

# `00_Common/dotnet/Games.Common.Test/IO/TokenizerTests.cs`

该测试代码旨在通过 `Tokenizer.ParseTokens` 函数将给定的输入字符串分割成预期输出的一行或多行。为了验证分割是否正确，测试代码使用 `TheoryData` 类型来定义一组测试用例，其中包含输入和预期输出两者的组合。每个测试用例都包含输入字符串、预期输出字符串以及预期输出是否与输入字符串分割后的结果相等。

具体来说，测试代码中的 `TokenizerTests` 类包含了 `ParseTokens_SplitsStringIntoExpectedTokens` 方法，该方法接受输入字符串和预期输出字符串作为参数。对于每个输入字符串，该方法都会使用 `Tokenizer.ParseTokens` 函数来获取相应的输出字符串，然后使用 LINQ 中的 `Select` 方法将输出字符串转换为字符串数组，并使用 `Should` 方法来验证预期输出是否与输入字符串分割后的结果相等。如果分割正确，该测试用例将会通过 `System.Object` 这个名字空間的 `TheoryData` 类型，否则将产生一个 `System.InvalidOperationException` 异常。

在 `TokenizerTestCases` 类中，定义了一系列的测试用例，包括输入为空字符串、输入只有一行字符串、输入多行字符串等不同情况。每个测试用例都包含输入和预期输出两者的组合，这些组合代表了不同的输入情况。


```
using System.Linq;
using FluentAssertions;
using Xunit;

namespace Games.Common.IO;

public class TokenizerTests
{
    [Theory]
    [MemberData(nameof(TokenizerTestCases))]
    public void ParseTokens_SplitsStringIntoExpectedTokens(string input, string[] expected)
    {
        var result = Tokenizer.ParseTokens(input);

        result.Select(t => t.ToString()).Should().BeEquivalentTo(expected);
    }

    public static TheoryData<string, string[]> TokenizerTestCases() => new()
    {
        { "", new[] { "" } },
        { "aBc", new[] { "aBc" } },
        { "  Foo   ", new[] { "Foo" } },
        { "  \" Foo  \"  ", new[] { " Foo  " } },
        { "  \" Foo    ", new[] { " Foo    " } },
        { "\"\"abc", new[] { "" } },
        { "a\"\"bc", new[] { "a\"\"bc" } },
        { "\"\"", new[] { "" } },
        { ",", new[] { "", "" } },
        { " foo  ,bar", new[] { "foo", "bar" } },
        { "\"a\"bc,de", new[] { "a" } },
        { "a\"b,\" c,d\", f ,,g", new[] { "a\"b", " c,d", "f", "", "g" } }
    };
}

```

# `00_Common/dotnet/Games.Common.Test/IO/TokenReaderTests.cs`

It looks like you have written code for a `ReadTokensTestCases` class and a `ReadNumericTokensTestCases` class, but they are not defined in the code snippet you provided.

If you have defined these classes elsewhere in your program, I would recommend looking for the definitions for these tests in your code organization or documentation.

If not, I would suggest looking at the contents of the code snippet you provided, and trying to understand what it is intended to demonstrate. Based on the contents of the code, it is difficult to provide a meaningful answer to the question you asked.


```
using System;
using System.IO;
using System.Linq;
using FluentAssertions;
using FluentAssertions.Execution;
using Xunit;

using static System.Environment;
using static Games.Common.IO.Strings;

namespace Games.Common.IO;

public class TokenReaderTests
{
    private readonly StringWriter _outputWriter;

    public TokenReaderTests()
    {
        _outputWriter = new StringWriter();
    }

    [Fact]
    public void ReadTokens_QuantityNeededZero_ThrowsArgumentException()
    {
        var sut = TokenReader.ForStrings(new TextIO(new StringReader(""), _outputWriter));

        Action readTokens = () => sut.ReadTokens("", 0);

        readTokens.Should().Throw<ArgumentOutOfRangeException>()
            .WithMessage("'quantityNeeded' must be greater than zero.*")
            .WithParameterName("quantityNeeded");
    }


    [Theory]
    [MemberData(nameof(ReadTokensTestCases))]
    public void ReadTokens_ReadingValuesHasExpectedPromptsAndResults(
        string prompt,
        uint tokenCount,
        string input,
        string expectedOutput,
        string[] expectedResult)
    {
        var sut = TokenReader.ForStrings(new TextIO(new StringReader(input + NewLine), _outputWriter));

        var result = sut.ReadTokens(prompt, tokenCount);
        var output = _outputWriter.ToString();

        using var _ = new AssertionScope();
        output.Should().Be(expectedOutput);
        result.Select(t => t.String).Should().BeEquivalentTo(expectedResult);
    }

    [Theory]
    [MemberData(nameof(ReadNumericTokensTestCases))]
    public void ReadTokens_Numeric_ReadingValuesHasExpectedPromptsAndResults(
        string prompt,
        uint tokenCount,
        string input,
        string expectedOutput,
        float[] expectedResult)
    {
        var sut = TokenReader.ForNumbers(new TextIO(new StringReader(input + NewLine), _outputWriter));

        var result = sut.ReadTokens(prompt, tokenCount);
        var output = _outputWriter.ToString();

        using var _ = new AssertionScope();
        output.Should().Be(expectedOutput);
        result.Select(t => t.Number).Should().BeEquivalentTo(expectedResult);
    }

    public static TheoryData<string, uint, string, string, string[]> ReadTokensTestCases()
    {
        return new()
        {
            { "Name", 1, "Bill", "Name? ", new[] { "Bill" } },
            { "Names", 2, " Bill , Bloggs ", "Names? ", new[] { "Bill", "Bloggs" } },
            { "Names", 2, $" Bill{NewLine}Bloggs ", "Names? ?? ", new[] { "Bill", "Bloggs" } },
            {
                "Foo",
                6,
                $"1,2{NewLine}\" a,b \"{NewLine},\"\"c,d{NewLine}d\"x,e,f",
                $"Foo? ?? ?? ?? {ExtraInput}{NewLine}",
                new[] { "1", "2", " a,b ", "", "", "d\"x" }
            }
        };
    }

    public static TheoryData<string, uint, string, string, float[]> ReadNumericTokensTestCases()
    {
        return new()
        {
            { "Age", 1, "23", "Age? ", new[] { 23F } },
            { "Constants", 2, " 3.141 , 2.71 ", "Constants? ", new[] { 3.141F, 2.71F } },
            { "Answer", 1, $"Forty-two{NewLine}42 ", $"Answer? {NumberExpected}{NewLine}? ", new[] { 42F } },
            {
                "Foo",
                6,
                $"1,2{NewLine}\" a,b \"{NewLine}3, 4  {NewLine}5.6,7,a, b",
                $"Foo? ?? {NumberExpected}{NewLine}? ?? {ExtraInput}{NewLine}",
                new[] { 1, 2, 3, 4, 5.6F, 7 }
            }
        };
    }
}

```