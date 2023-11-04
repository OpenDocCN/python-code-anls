# BasicComputerGames源码解析 38

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/) by Alex Conconi

---

### Porting notes for Lua

- This is a straightfoward port with only minor modifications for input
validation and text formatting.

- The "Try again?" question accepts 'y', 'yes', 'n', 'no' (case insensitive),
whereas the original BASIC version defaults to no unless 'YES' is typed.

- The "How many rolls?" question presents a more user friendly message
in case of invalid input.


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


# `33_Dice/python/dice.py`

这段代码是一个用于模拟掷骰子的程序，它使用了一个 while 循环来模拟多次掷骰子。每次循环，程序会询问用户输入掷骰子的次数，然后使用循环内部的变量来跟踪所需的骰子次数和相应的频率分布。

具体来说，这段代码的作用是模拟投掷一定次数的骰子，并输出它们的频率分布。它有趣的结论是，当投掷的次数越多时，接近理论分布的骰子数量将越来越接近他们的理论频


```
"""
Dice

From: BASIC Computer Games (1978)
      Edited by David H. Ahl

"Not exactly a game, this program simulates rolling
 a pair of dice a large number of times and prints out
 the frequency distribution.  You simply input the
 number of rolls.  It is interesting to see how many
 rolls are necessary to approach the theoretical
 distribution:

 2  1/36  2.7777...%
 3  2/36  5.5555...%
 4  3/36  8.3333...%
   etc.

```

This program simulates the rolling of a pair of dice. The user is prompted to enter the number of times they want the computer to roll the dice. After each roll, the program tracks the counts of roll outcomes in a 13-element list. The first two indices (0 & 1) are ignored, leaving just the indices that match the roll values (2 through 12).

The program starts by creating a list called `freq` to store the counts of roll outcomes. The counts are established by randomly rolling a pair of dice, and the frequency of each number is incremented.

The program then displays an intro text with the program's name and an explanation of its purpose.

The user is then prompted to enter the number of rolls they want to simulate. If the user enters a number of 5000 or more, a long response message is displayed.

The user is then entered into a game loop that keeps prompting the user to enter the number of rolls they want to simulate until they choose to exit the game loop or until they enter a number that is not a large number.

The program also has a function called `display_final_results` that displays the final results of the game by displaying the counts of roll outcomes in the `freq` list.

Overall, the program simulates the rolling of a pair of dice, allowing the user to enter the number of rolls they want to simulate and see the results of each roll.


```
"Daniel Freidus wrote this program while in the
 seventh grade at Harrison Jr-Sr High School,
 Harrison, New York."

Python port by Jeff Jetton, 2019
"""

import random


def main() -> None:
    # We'll track counts of roll outcomes in a 13-element list.
    # The first two indices (0 & 1) are ignored, leaving just
    # the indices that match the roll values (2 through 12).
    freq = [0] * 13

    # Display intro text
    print("\n                   Dice")
    print("Creative Computing  Morristown, New Jersey")
    print("\n\n")
    # "Danny Freidus"
    print("This program simulates the rolling of a")
    print("pair of dice.")
    print("You enter the number of times you want the computer to")
    print("'roll' the dice.   Watch out, very large numbers take")
    print("a long time.  In particular, numbers over 5000.")

    still_playing = True
    while still_playing:
        print()
        n = int(input("How many rolls? "))

        # Roll the dice n times
        for _ in range(n):
            die1 = random.randint(1, 6)
            die2 = random.randint(1, 6)
            roll_total = die1 + die2
            freq[roll_total] += 1

        # Display final results
        print("\nTotal Spots   Number of Times")
        for i in range(2, 13):
            print(" %-14d%d" % (i, freq[i]))

        # Keep playing?
        print()
        response = input("Try again? ")
        if len(response) > 0 and response.upper()[0] == "Y":
            # Clear out the frequency list
            freq = [0] * 13
        else:
            # Exit the game loop
            still_playing = False


```

这段代码是一个Python程序，其中包含一个if语句。if语句的语法是在Python 2中使用的，而在Python 3中，则需要将if语句中的“__name__”改为“is”。

if __name__ == "__main__":
   main()

if语句的作用是判断当前程序是否作为主程序运行。如果是，那么程序将进入__main__函数，否则程序将继续执行。在这个例子中，如果__name__不等于 "__main__"，那么程序将进入一个else语句，其中包含main()函数。

if __name__ == "__main__":
   main()

这段代码的作用是允许您在Python程序中使用if语句，而无需在if语句前加上“__main__”。在if语句中，您可以使用Python 2和Python 3中的语法。


```
if __name__ == "__main__":
    main()

########################################################
#
# Porting Notes
#
#   A fairly straightforward port.  The only change is
#   in the handling of the user's "try again" response.
#   The original program only continued if the user
#   entered "YES", whereas this version will continue
#   if any word starting with "Y" or "y" is given.
#
#   The instruction text--which, like all these ports,
#   was taken verbatim from the original listing--is
```

这段代码是一个警告，告知在创建大规模随机 roll 操作时，设置 roll 数量过高可能会导致时间过长。当前时间在使用的计算机上，5000 卷的 roll 通常会花费不到 1/10 秒。

作者建议进行修改，具体如下：

1. 为结果添加第三个列，显示每个计数所代表的百分比。
2. 更好地说，使用低位级 bar graph 使用行波浪线表示相对值，每个波浪线代表一％。

对于这个警告，作者提醒说在创建大规模随机 roll 操作时，应该注意不要设置得太高，以免影响程序的性能。同时，提供了可行的修改建议，让用户可以选择使用哪个方案。


```
#   charmingly quaint in its dire warning against
#   setting the number of rolls too high.  At the time
#   of this writing, on a fairly slow computer, a
#   5000-roll run typically clocks in at well under
#   1/10 of a second!
#
#
# Ideas for Modifications
#
#   Have the results include a third column showing
#   the percent of rolls each count represents.  Or
#   (better yet) print a low-fi bar graph using
#   rows of asterisks to represent relative values,
#   with each asterisk representing one percent,
#   for example.
```

这段代码是一个简单的 Python 程序，用于计算不同数量投掷骰子所得到的理论预期百分比。这个程序有一个名为“roll_dice.py”的文件名。

程序的主要部分如下：

```python
# 导出 rolling.py 函数
from rolling import roll

# 定义结果字符串
result_str = "理论上预期的百分比\n"

# 初始化关系，一个系列0次成功投掷的比率是我们已经知晓的
num_successes = 0
count_of_rolls = 0

# 如果是3个，4个，或者5个骰子
if num_successes == 3:
   rolls = [1, 2, 3]
elif num_successes == 4:
   rolls = [1, 2, 3, 4]
elif num_successes == 5:
   rolls = [1, 2, 3, 4, 5]
else:
   rolls = [1, 2, 3, 4]

# 计算期望值
for roll in rolls:
   percent_success = roll / 6 * 100
   num_successes += 1
   count_of_rolls += 1
   result_str += f"在 {num_successes} 次成功投掷中，{percent_success} 的期望值为 {percent_success:.2f}%"
   result_str += f"，总投掷次数为 {count_of_rolls} 次。"
   result_str += f"，成功次数为 {num_successes} 次。"
   print(result_str)
```

这段代码的作用是计算成功投掷一个骰子（3个，4个或5个）时的预期百分比。它将成功投掷的次数、期望值和总投掷次数存储在一个字典中，然后输出结果。

值得注意的是，这个程序仅作为一个简单的示例，实际应用中可能需要对数据进行更复杂的处理，例如对不同次数的成功投掷进行加权平均以得出更精确的期望值。


```
#
#   Add a column showing the theoretically expected
#   percentage, for comparison.
#
#   Keep track of how much time the series of rolls
#   takes and add that info to the final report.
#
#   What if three (or four, or five...) dice were
#   rolled each time?
#
########################################################

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Digits

The player writes down a set of 30 numbers (0, 1, or 2) at random prior to playing the game. The computer program, using pattern recognition techniques, attempts to guess the next number in your list.

The computer asks for 10 numbers at a time. It always guesses first and then examines the next number to see if it guessed correctly. By pure luck (or chance or probability), the computer ought to be right 10 times. It is uncanny how much better it generally does than that!

This program originated at Dartmouth; original author unknown.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=58)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=73)


Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

- The program contains a lot of mysterious and seemingly arbitrary constants.  It's not clear there is any logic or rationality behind it.
- The key equation involved in the guess (line 700) involves a factor of `A`, but `A` is always 0, making that term meaningless.  As a result, all the work to build and update array K and value Z2 appear to be meaningless, too.


# `34_Digits/csharp/Game.cs`



这段代码定义了一个名为 "GameSeries" 的类，用于呈现一个带有三项游戏的游戏序列。这个游戏序列可以通过输入想要尝试的游戏数量来控制，每次游戏开始时，游戏会随机选择一个游戏进行展示。

GameSeries 类包含三个私有字段：

- IReadOnlyList<int> _weights: 这是一个只读的整数列表，包含三个游戏所需的权重。
- IReadWrite _io: 这是一个用于从玩家输入中读取字节的 IReadWrite 类。
- IRandom _random: 这是一个用于从玩家输入中读取随机整数的 IRandom 类。

GameSeries 类包含一个名为 "Play" 的内部方法，这个方法会从玩家输入中读取想要尝试的游戏数量，然后循环运行一个带有三个游戏的游戏实例，每次游戏开始时，游戏会随机选择一个游戏进行展示。

最后，GameSeries 类还包含一个名为 "Game" 的内部类，用于呈现具体的游戏实例。


```
namespace Digits;

internal class GameSeries
{
    private readonly IReadOnlyList<int> _weights = new List<int> { 0, 1, 3 }.AsReadOnly();

    private readonly IReadWrite _io;
    private readonly IRandom _random;

    public GameSeries(IReadWrite io, IRandom random)
    {
        _io = io;
        _random = random;
    }

    internal void Play()
    {
        _io.Write(Streams.Introduction);

        if (_io.ReadNumber(Prompts.ForInstructions) != 0)
        {
            _io.Write(Streams.Instructions);
        }

        do
        {
            new Game(_io, _random).Play();
        } while (_io.ReadNumber(Prompts.WantToTryAgain) == 1);

        _io.Write(Streams.Thanks);
    }
}

```



This code defines an internal class `Game` that has a constructor that takes in two parameters: `IReadWrite` for the input stream and an instance of `IRandom`. The purpose of these parameters is not specified in this code, but it is possible to infer some possible functionality from the code.

The `Game` class has a single public method called `Play`, which seems to be the entry point of the program. This method reads in a number of digits from the input stream using the `IReadWrite` parameter, then calls a private method called `GuessDigits` and finally writes out the result of the game.

The `GuessDigits` method takes in an `IEnumerable<int>` of digits, the number of correct guesses, and the current digit being guessed at. It reads the current digit from the input stream using the `IRandom` parameter, then calls the `_guesser.GuessNextDigit()` method to get the predicted digit for the current round. If the predicted digit matches the current digit, the method increments the `correctGuesses` counter. The method then writes out the result of the game, indicating whether the player won, it's a tie or the player lost.

The `Play` method also has a `Streams.YouWin` output if the player wins, an output of `Streams.ItsATie` if it's a tie and an output of `Streams.IWin` if the player wins.


```
internal class Game
{
    private readonly IReadWrite _io;
    private readonly Guesser _guesser;

    public Game(IReadWrite io, IRandom random)
    {
        _io = io;
        _guesser = new Guesser(random);
    }

    public void Play()
    {
        var correctGuesses = 0;

        for (int round = 0; round < 3; round++)
        {
            var digits = _io.Read10Digits(Prompts.TenNumbers, Streams.TryAgain);

            correctGuesses = GuessDigits(digits, correctGuesses);
        }

        _io.Write(correctGuesses switch
        {
            < 10 => Streams.YouWin,
            10 => Streams.ItsATie,
            > 10 => Streams.IWin
        });
    }

    private int GuessDigits(IEnumerable<int> digits, int correctGuesses)
    {
        _io.Write(Streams.Headings);

        foreach (var digit in digits)
        {
            var guess = _guesser.GuessNextDigit();
            if (guess == digit) { correctGuesses++; }

            _io.WriteLine(Formats.GuessResult, guess, digit, guess == digit ? "Right" : "Wrong", correctGuesses);

            _guesser.ObserveActualDigit(digit);
        }

        return correctGuesses;
    }
}

```

# `34_Digits/csharp/Guesser.cs`

这段代码定义了一个名为"Guesser"的内部类，其作用是让用户猜测下一个数字，并从数字0到9中进行猜测。

具体来说，Guesser类包含一个内部成员变量IRandom，用于生成随机数，以及一个内部成员变量Guesser，用于存储当前猜测的数字。

Guesser类还包含一个GuessNextDigit方法，该方法使用IRandom生成随机数，并计算当前猜测的数字加上当前数字乘以每个数字的权重(即当前数字在数字集中出现的次数)的和，然后将和与当前猜测的数字进行比较。如果和比当前猜测的数字大或者使用IRandom随机生成的随机数大于50%，就认为当前猜测的数字是正确的，否则继续猜测。

此外，Guesser类还包括一个ObserveActualDigit方法，该方法接受一个整数参数，表示当前猜测的数字，然后使用IRandom生成一个随机数，与当前猜测的数字进行比较，如果生成的随机数大于50%，就将当前猜测的数字更新为生成的随机数。通过多次调用ObserveActualDigit方法，用户可以观察到数字集中每个数字的出现次数，从而提高猜测的准确性。


```
namespace Digits;

internal class Guesser
{
    private readonly Memory _matrices = new();
    private readonly IRandom _random;

    public Guesser(IRandom random)
    {
        _random = random;
    }

    public int GuessNextDigit()
    {
        var currentSum = 0;
        var guess = 0;

        for (int i = 0; i < 3; i++)
        {
            var sum = _matrices.GetWeightedSum(i);
            if (sum > currentSum || _random.NextFloat() >= 0.5)
            {
                currentSum = sum;
                guess = i;
            }
        }

        return guess;
    }

    public void ObserveActualDigit(int digit) => _matrices.ObserveDigit(digit);
}

```

# `34_Digits/csharp/IOExtensions.cs`



这段代码是一个名为"IOExtensions"的内部类，包含一个名为"Read10Digits"的静态方法，它的参数为"IReadWrite"类型，类型为"int"和"string"类型。

该方法的作用是读取10个数字，可以有0、1或2，从用户输入中读取。如果读取成功，则返回数字。如果任何一个数字在循环中没有被读取到，则会执行循环，并尝试从文件中读取内容到Stream对象中。

从代码中可以看出，该方法使用的是"Read10Digits"这个名字，但并没有定义可读性或可访问性。因此，如果要在代码中使用该方法，需要创建一个IOExtensions类，并在其中添加一个可读性或可访问性较低的名称，以便其他开发人员更容易地使用该方法。


```
namespace Digits;

internal static class IOExtensions
{
    internal static IEnumerable<int> Read10Digits(this IReadWrite io, string prompt, Stream retryText)
    {
        while (true)
        {
            var numbers = new float[10];
            io.ReadNumbers(prompt, numbers);

            if (numbers.All(n => n == 0 || n == 1 || n == 2))
            {
                return numbers.Select(n => (int)n);
            }    

            io.Write(retryText);
        }
    }
}
```

# `34_Digits/csharp/Matrix.cs`

这段代码定义了一个名为Matrix的内部类，表示一个2D矩阵。该类包含一个私有变量_weight表示矩阵的权重，一个私有变量_values一个3D数组，表示矩阵中的元素。

该类有一个构造函数，接受3个参数：矩阵的宽度、矩阵中每个元素的权重以及一个名为seedFactory的函数，用于生成随机整数。构造函数首先初始化_values数组，然后从0到_weight-1循环遍历矩阵的每个元素，使用seedFactory生成随机整数并将其存储在_values数组中。

该类还有一个名为Index的私有变量，用于跟踪矩阵中每个元素的行。

该类有一个GetWeightedValue方法，用于获取指定行元素的权乘和，即_weight * _values[Index, row]。

该类还有一个IncrementValue方法，用于增加指定行元素的值，即_values[Index, row]自增1。

总之，这段代码定义了一个用于表示和组织矩阵数据的类，该类可以使用多种方式操作矩阵，包括生成随机数、访问元素和行、以及计算元素的和。


```
namespace Digits;

internal class Matrix
{
    private readonly int _weight;
    private readonly int[,] _values;

    public Matrix(int width, int weight, Func<int, int, int> seedFactory)
    {
        _weight = weight;
        _values = new int[width, 3];
        
        for (int i = 0; i < width; i++)
        for (int j = 0; j < 3; j++)
        {
            _values[i, j] = seedFactory.Invoke(i, j);
        }

        Index = width - 1;
    }

    public int Index { get; set; }

    public int GetWeightedValue(int row) => _weight * _values[Index, row];

    public int IncrementValue(int row) => _values[Index, row]++;
}
```

# `34_Digits/csharp/Memory.cs`



这段代码定义了一个名为 "Memory" 的类，其包含一个私有的 "Matrix" 数组，以及一个公共的 "GetWeightedSum" 方法和一个名为 "ObserveDigit" 的公共方法。

在 "Memory" 类中，首先创建了一个包含两个整型变量 "row" 和 "digit"，以及一个包含三个整型变量 "27"、"9" 和 "3" 的数组。这个数组是为了解决 Matrix 类的，通过给定的行号计算每个元素的值。

在 "GetWeightedSum" 方法中，通过调用 Matrix 类的 "GetWeightedValue" 方法，来获取每个元素在给定行号下的值，然后将这些值求和得到结果。

在 "ObserveDigit" 方法中，对每个 "digit" 进行操作，包括对每个矩阵元素进行步进，以及将该数字的索引存储在 "Index" 属性中，以便在需要时可以访问该数字的对齐位置。


```
namespace Digits;

public class Memory
{
    private readonly Matrix[] _matrices;

    public Memory()
    {
        _matrices = new[] 
        {
            new Matrix(27, 3, (_, _) => 1),
            new Matrix(9, 1, (i, j) => i == 4 * j ? 2 : 3),
            new Matrix(3, 0, (_, _) => 9)
        };
    }

    public int GetWeightedSum(int row) => _matrices.Select(m => m.GetWeightedValue(row)).Sum();

    public void ObserveDigit(int digit)
    {
        for (int i = 0; i < 3; i++)
        {
            _matrices[i].IncrementValue(digit);
        }

        _matrices[0].Index = _matrices[0].Index % 9 * 3 + digit;
        _matrices[1].Index = _matrices[0].Index % 9;
        _matrices[2].Index = digit;
    }
}
```

# `34_Digits/csharp/Program.cs`

这段代码的作用是创建一个名为 "GameSeries" 的类，其中包含三个来自 "Games.Common.IO"、"Games.Common.Randomness" 和 "Digits.Resources.Resource" 的命名常量。它还包含一个构造函数和一个名为 "newGameSeries" 的静态方法，该方法接受两个参数，一个是 "ConsoleIO" 类的实例，另一个是 "RandomNumberGenerator" 类的实例。

具体来说，这段代码会创建一个 "GameSeries" 类的新实例，该实例使用 "ConsoleIO" 和 "RandomNumberGenerator" 类从标准输入和随机数生成器中获取资源，然后使用这些资源来创建和播放游戏序列。最后，"newGameSeries" 方法将开始玩这个游戏序列。


```
global using Digits;
global using Games.Common.IO;
global using Games.Common.Randomness;
global using static Digits.Resources.Resource;

new GameSeries(new ConsoleIO(), new RandomNumberGenerator()).Play();
```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `34_Digits/csharp/Resources/Resource.cs`



这段代码是一个自定义的数字计算游戏的资源文件夹，其中包含了游戏中的各种资源，包括输入输出流、提示信息、游戏格式等。

具体来说，代码中定义了一个Streams类，其中包含了多个静态的Stream类型的成员变量，分别代表了游戏中的介绍、指令、再试、胜利、失败、谢谢、标题等元素。这些Stream类的成员函数使用了GetStream()方法，该方法根据传入的名称来加载游戏资源文件中的对应Stream对象，然后返回给用户。

代码中还定义了一个Prompts类，其中包含了多个静态的GetString类型的成员变量，分别代表了游戏中的输入输出提示信息，如ForInstructions、TenNumbers、WantToTryAgain等。

代码中还定义了一个Formats类，其中包含了一个静态的GetString类型的成员函数，代表了游戏中的胜利/失败提示信息。

最后，代码中还有一位绕组的GetString类型的成员函数GetString()，该函数使用了GetStream()方法来加载游戏资源文件中的对应Stream对象，然后返回给用户一个字符串。

该代码的作用是定义了一个游戏中的资源文件夹，包含了游戏中的各种元素，包括输入输出流、提示信息、游戏格式等，可以被游戏中的程序通过调用GetStream()方法来加载游戏资源文件中的对应元素。


```
using System.Reflection;
using System.Runtime.CompilerServices;

namespace Digits.Resources;

internal static class Resource
{
    internal static class Streams
    {
        public static Stream Introduction => GetStream();
        public static Stream Instructions => GetStream();
        public static Stream TryAgain => GetStream();
        public static Stream ItsATie => GetStream();
        public static Stream IWin => GetStream();
        public static Stream YouWin => GetStream();
        public static Stream Thanks => GetStream();
        public static Stream Headings => GetStream();
    }

    internal static class Prompts
    {
        public static string ForInstructions => GetString();
        public static string TenNumbers => GetString();
        public static string WantToTryAgain => GetString();
    }

    internal static class Formats
    {
        public static string GuessResult => GetString();
    }

    private static string GetString([CallerMemberName] string? name = null)
    {
        using var stream = GetStream(name);
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }


    private static Stream GetStream([CallerMemberName] string? name = null) =>
        Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")
            ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
}
```

# `34_Digits/java/Digits.java`

This is a Java program that reads a piece of paper with numbers in it. The program then prints instructions on how to play the game, and then prompts the user to guess the numbers. If the user makes a correct guess, the program will tell the user the number of digits that they predicted. If the user makes a wrong guess, the program will tell the user whether their guess was too far off or too close. The program will then ask the user if they want to continue guessing until they correctly guess all of the numbers.

The program also has a printInstructionChoice function that will print the instructions to the user. And a printIntro function that will print the introduction to the game.

It is important to note that this program is not very functional and has a lot of issues, such as the part where it does not check if the user entered a number and it does not handle the exceptions.


```
import java.util.Arrays;
import java.util.InputMismatchException;
import java.util.Scanner;

/**
 * DIGITS
 * <p>
 * Converted from BASIC to Java by Aldrin Misquitta (@aldrinm)
 */
public class Digits {

	public static void main(String[] args) {
		printIntro();
		Scanner scan = new Scanner(System.in);

		boolean showInstructions = readInstructionChoice(scan);
		if (showInstructions) {
			printInstructions();
		}

		int a = 0, b = 1, c = 3;
		int[][] m = new int[27][3];
		int[][] k = new int[3][3];
		int[][] l = new int[9][3];

		boolean continueGame = true;
		while (continueGame) {
			for (int[] ints : m) {
				Arrays.fill(ints, 1);
			}
			for (int[] ints : k) {
				Arrays.fill(ints, 9);
			}
			for (int[] ints : l) {
				Arrays.fill(ints, 3);
			}

			l[0][0] = 2;
			l[4][1] = 2;
			l[8][2] = 2;

			int z = 26, z1 = 8, z2 = 2, runningCorrect = 0;

			for (int t = 1; t <= 3; t++) {
				boolean validNumbers = false;
				int[] numbers = new int[0];
				while (!validNumbers) {
					System.out.println();
					numbers = read10Numbers(scan);
					validNumbers = true;
					for (int number : numbers) {
						if (number < 0 || number > 2) {
							System.out.println("ONLY USE THE DIGITS '0', '1', OR '2'.");
							System.out.println("LET'S TRY AGAIN.");
							validNumbers = false;
							break;
						}
					}
				}

				System.out.printf("\n%-14s%-14s%-14s%-14s", "MY GUESS", "YOUR NO.", "RESULT", "NO. RIGHT");
				for (int number : numbers) {
					int s = 0;
					int myGuess = 0;
					for (int j = 0; j <= 2; j++) {
						//What did the original author have in mind ? The first expression always results in 0 because a is always 0
						int s1 = a * k[z2][j] + b * l[z1][j] + c * m[z][j];
						if (s < s1) {
							s = s1;
							myGuess = j;
						} else if (s1 == s) {
							if (Math.random() >= 0.5) {
								myGuess = j;
							}
						}
					}

					String result;
					if (myGuess != number) {
						result = "WRONG";
					} else {
						runningCorrect++;
						result = "RIGHT";
						m[z][number] = m[z][number] + 1;
						l[z1][number] = l[z1][number] + 1;
						k[z2][number] = k[z2][number] + 1;
						z = z - (z / 9) * 9;
						z = 3 * z + number;
					}
					System.out.printf("\n%-14d%-14d%-14s%-14d", myGuess, number, result, runningCorrect);

					z1 = z - (z / 9) * 9;
					z2 = number;
				}
			}

			//print summary report
			System.out.println();
			if (runningCorrect > 10) {
				System.out.println();
				System.out.println("I GUESSED MORE THAN 1/3 OF YOUR NUMBERS.");
				System.out.println("I WIN.\u0007");
			} else if (runningCorrect < 10) {
				System.out.println("I GUESSED LESS THAN 1/3 OF YOUR NUMBERS.");
				System.out.println("YOU BEAT ME.  CONGRATULATIONS *****");
			} else {
				System.out.println("I GUESSED EXACTLY 1/3 OF YOUR NUMBERS.");
				System.out.println("IT'S A TIE GAME.");
			}

			continueGame = readContinueChoice(scan);
		}

		System.out.println("\nTHANKS FOR THE GAME.");
	}

	private static boolean readContinueChoice(Scanner scan) {
		System.out.print("\nDO YOU WANT TO TRY AGAIN (1 FOR YES, 0 FOR NO) ? ");
		int choice;
		try {
			choice = scan.nextInt();
			return choice == 1;
		} catch (InputMismatchException ex) {
			return false;
		} finally {
			scan.nextLine();
		}
	}

	private static int[] read10Numbers(Scanner scan) {
		System.out.print("TEN NUMBERS, PLEASE ? ");
		int[] numbers = new int[10];

		for (int i = 0; i < numbers.length; i++) {
			boolean validInput = false;
			while (!validInput) {
				try {
					int n = scan.nextInt();
					validInput = true;
					numbers[i] = n;
				} catch (InputMismatchException ex) {
					System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE");
				} finally {
					scan.nextLine();
				}
			}
		}

		return numbers;
	}

	private static void printInstructions() {
		System.out.println("\n");
		System.out.println("PLEASE TAKE A PIECE OF PAPER AND WRITE DOWN");
		System.out.println("THE DIGITS '0', '1', OR '2' THIRTY TIMES AT RANDOM.");
		System.out.println("ARRANGE THEM IN THREE LINES OF TEN DIGITS EACH.");
		System.out.println("I WILL ASK FOR THEN TEN AT A TIME.");
		System.out.println("I WILL ALWAYS GUESS THEM FIRST AND THEN LOOK AT YOUR");
		System.out.println("NEXT NUMBER TO SEE IF I WAS RIGHT. BY PURE LUCK,");
		System.out.println("I OUGHT TO BE RIGHT TEN TIMES. BUT I HOPE TO DO BETTER");
		System.out.println("THAN THAT *****");
		System.out.println();
	}

	private static boolean readInstructionChoice(Scanner scan) {
		System.out.print("FOR INSTRUCTIONS, TYPE '1', ELSE TYPE '0' ? ");
		int choice;
		try {
			choice = scan.nextInt();
			return choice == 1;
		} catch (InputMismatchException ex) {
			return false;
		} finally {
			scan.nextLine();
		}
	}

	private static void printIntro() {
		System.out.println("                                DIGITS");
		System.out.println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
		System.out.println("\n\n");
		System.out.println("THIS IS A GAME OF GUESSING.");
	}

}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `34_Digits/javascript/digits.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。它们的主要作用如下：

1. `print` 函数将一个字符串转换为 Unicode 编码后，将其插入到页面上某个元素的文本节点中。这个字符串可以包含任意的 Unicode 字符。

2. `input` 函数允许用户在网页上输入一行文本。它将接收用户输入的字符串，并将其存储在 `input_str` 变量中。然后，它等待一个关键事件，即用户按下回车键，然后将 `input_str` 打印到页面上并将其从文档中删除。然后，它再次询问用户输入字符串，并将结果打印到页面上并将其从文档中删除。

在这两个函数中，都使用了 Document Object Model（DOM）和 JavaScript 中的 `document.getElementById` 和 `document.createTextNode` 函数来获取和设置页面上元素的引用和文本节点内容。


```
// DIGITS
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       input_element = document.createElement("INPUT");

                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      input_str = input_element.value;
                                                      document.getElementById("output").removeChild(input_element);
                                                      print(input_str);
                                                      print("\n");
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

```

In this game, the player is asked to guess the numbers that are already known, but the number they are asked to guess is not revealed until the player tries to guess. The player is given a limited number of attempts to guess the numbers, and upon each guess, the game tells the player if the guess was too high or too low, or if the game is over and the player has won. The game ends when the player either guesses correctly or runs out of attempts.



```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// Main program
async function main()
{
    print(tab(33) + "DIGITS\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("THIS IS A GAME OF GUESSING.\n");
    print("FOR INSTRUCTIONS, TYPE '1', ELSE TYPE '0'");
    e = parseInt(await input());
    if (e != 0) {
        print("\n");
        print("PLEASE TAKE A PIECE OF PAPER AND WRITE DOWN\n");
        print("THE DIGITS '0', '1', OR '2' THIRTY TIMES AT RANDOM.\n");
        print("ARRANGE THEM IN THREE LINES OF TEN DIGITS EACH.\n");
        print("I WILL ASK FOR THEN TEN AT A TIME.\n");
        print("I WILL ALWAYS GUESS THEM FIRST AND THEN LOOK AT YOUR\n");
        print("NEXT NUMBER TO SEE IF I WAS RIGHT. BY PURE LUCK,\n");
        print("I OUGHT TO BE RIGHT TEN TIMES. BUT I HOPE TO DO BETTER\n");
        print("THAN THAT *****\n");
        print("\n");
        print("\n");
    }
    a = 0;
    b = 1;
    c = 3;
    m = [];
    k = [];
    l = [];
    n = [];
    while (1) {
        for (i = 0; i <= 26; i++) {
            m[i] = [];
            for (j = 0; j <= 2; j++) {
                m[i][j] = 1;
            }
        }
        for (i = 0; i <= 2; i++) {
            k[i] = [];
            for (j = 0; j <= 2; j++) {
                k[i][j] = 9;
            }
        }
        for (i = 0; i <= 8; i++) {
            l[i] = [];
            for (j = 0; j <= 2; j++) {
                l[i][j] = 3;
            }
        }
        l[0][0] = 2;
        l[4][1] = 2;
        l[8][2] = 2;
        z = 26;
        z1 = 8;
        z2 = 2;
        x = 0;
        for (t = 1; t <= 3; t++) {
            while (1) {
                print("\n");
                print("TEN NUMBERS, PLEASE");
                str = await input();
                for (i = 1; i <= 10; i++) {
                    n[i] = parseInt(str);
                    j = str.indexOf(",");
                    if (j >= 0) {
                        str = str.substr(j + 1);
                    }
                    if (n[i] < 0 || n[i] > 2)
                        break;
                }
                if (i <= 10) {
                    print("ONLY USE THE DIGITS '0', '1', OR '2'.\n");
                    print("LET'S TRY AGAIN.\n");
                } else {
                    break;
                }
            }
            print("\n");
            print("MY GUESS\tYOUR NO.\tRESULT\tNO. RIGHT\n");
            print("\n");
            for (u = 1; u <= 10; u++) {
                n2 = n[u];
                s = 0;
                for (j = 0; j <= 2; j++) {
                    s1 = a * k[z2][j] + b * l[z1][j] + c * m[z][j];
                    if (s > s1)
                        continue;
                    if (s < s1 || Math.random() >= 0.5) {
                        s = s1;
                        g = j;
                    }
                }
                print("  " + g + "\t\t   " + n[u] + "\t\t");
                if (g == n[u]) {
                    x++;
                    print(" RIGHT\t " + x + "\n");
                    m[z][n2]++;
                    l[z1][n2]++;
                    k[z2][n2]++;
                    z = z % 9;
                    z = 3 * z + n[u];
                } else {
                    print(" WRONG\t " + x + "\n");
                }
                z1 = z % 9;
                z2 = n[u];
            }
        }
        print("\n");
        if (x > 10) {
            print("I GUESSED MORE THAN 1/3 OF YOUR NUMBERS.\n");
            print("I WIN.\n");
        } else if (x == 10) {
            print("I GUESSED EXACTLY 1/3 OF YOUR NUMBERS.\n");
            print("IT'S A TIE GAME.\n");
        } else {
            print("I GUESSED LESS THAN 1/3 OF YOUR NUMBERS.\n");
            print("YOU BEAT ME.  CONGRATULATIONS *****\n");
        }
        print("\n");
        print("DO YOU WANT TO TRY AGAIN (1 FOR YES, 0 FOR NO)");
        x = parseInt(await input());
        if (x != 1)
            break;
    }
    print("\n");
    print("THANKS FOR THE GAME.\n");
}

```

这道题目没有给出代码，只是说明了一个名为`main()`的函数。根据函数的名称，我们可以猜测它可能是用来编写程序的入口点。但是，为了确保我们的猜测，我们需要看到完整的程序。


```
main();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


# `34_Digits/python/Digits.py`

这段代码是一个Python程序，它的主要目的是让用户猜测一个1到100之间的随机整数。它使用了两个功能模块：random和typing。

1. print_intro()是一个函数，用于打印游戏介绍信息。函数内部使用print函数输出了字符串"DIGITS"，"CREATIVE COMPUTING"和"MORRISTOWN, NEW JERSEY"。然后又输出了两行字符串"THIS IS A GAME OF GUESSING。"。

2. read_instruction_choice()是一个函数，用于读取用户输入的指令选择。函数内部使用print函数询问用户"FOR INSTRUCTIONS, TYPE '1', ELSE TYPE '0' ?"。它会提示用户输入一个数字，如果用户输入数字1，则返回True，否则返回False。

这段代码的主要目的是让用户在得到一个随机的1到100之间的整数的同时，猜测这个整数。通过猜测这个整数，用户可以锻炼他们的数学能力和逻辑思维能力。


```
import random
from typing import List


def print_intro() -> None:
    print("                                DIGITS")
    print("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print("\n\n")
    print("THIS IS A GAME OF GUESSING.")


def read_instruction_choice() -> bool:
    print("FOR INSTRUCTIONS, TYPE '1', ELSE TYPE '0' ? ")
    try:
        choice = int(input())
        return choice == 1
    except (ValueError, TypeError):
        return False


```

这段代码定义了两个函数，`print_instructions()` 是 `print_instructions()` 的回调函数，而 `read_10_numbers()` 是 `read_10_numbers()` 的回调函数。这两个函数的作用如下：

1. `print_instructions()`：这个函数用于打印一段文字，包括一些说明。它的作用是告诉用户按照要求做某事，然后随机生成三个数字，并将它们打印出来。具体来说，这个函数会输出：
```
Please take a piece of paper and write down
THE DIGITS '0', '1', or '2' thirty times at random.

Array them in three lines of ten digits each.
I will ask for ten at a time.
I will always guess them first and then look at your
next number to see if I was right.
I ought to be right ten times.
But I hope to do better than that.
```
2. `read_10_numbers()`：这个函数用于读取十个数字，并返回一个列表。具体来说，这个函数会输出：
```
TEN NUMBERS, PLEASE ? 
```
然后它会等待用户输入这十个数字，并将它们存储在一个列表中。如果用户输入的数字有误，或不是数字，函数会输出：
```
!NUMBER EXPECTED - RETRY INPUT LINE
```
如果用户输入的数字正确，函数就会返回这个数字列表。


```
def print_instructions() -> None:
    print("\n")
    print("PLEASE TAKE A PIECE OF PAPER AND WRITE DOWN")
    print("THE DIGITS '0', '1', OR '2' THIRTY TIMES AT RANDOM.")
    print("ARRANGE THEM IN THREE LINES OF TEN DIGITS EACH.")
    print("I WILL ASK FOR THEN TEN AT A TIME.")
    print("I WILL ALWAYS GUESS THEM FIRST AND THEN LOOK AT YOUR")
    print("NEXT NUMBER TO SEE IF I WAS RIGHT. BY PURE LUCK,")
    print("I OUGHT TO BE RIGHT TEN TIMES. BUT I HOPE TO DO BETTER")
    print("THAN THAT *****")
    print()


def read_10_numbers() -> List[int]:
    print("TEN NUMBERS, PLEASE ? ")
    numbers = []

    for _ in range(10):
        valid_input = False
        while not valid_input:
            try:
                n = int(input())
                valid_input = True
                numbers.append(n)
            except (TypeError, ValueError):
                print("!NUMBER EXPECTED - RETRY INPUT LINE")

    return numbers


```

这两段代码的作用如下：

1. `read_continue_choice()` 函数的作用是询问用户是否要再次尝试，函数内部会先打印一段提示信息，然后提示用户输入一个数字，如果用户输入 1，表示他们猜对了，返回 True；如果用户输入 0，表示他们猜错了或者输入不是数字，返回 False。
2. `print_summary_report()` 函数的作用是打印游戏中的得分报告，函数内部会根据用户的得分情况打印不同的消息。如果用户的得分大于等于 10，那么会打印两行消息；如果用户的得分小于 10，那么会打印一行消息；如果用户的得分正好是 1/3，那么会打印一行消息。


```
def read_continue_choice() -> bool:
    print("\nDO YOU WANT TO TRY AGAIN (1 FOR YES, 0 FOR NO) ? ")
    try:
        choice = int(input())
        return choice == 1
    except (ValueError, TypeError):
        return False


def print_summary_report(running_correct: int) -> None:
    print()
    if running_correct > 10:
        print()
        print("I GUESSED MORE THAN 1/3 OF YOUR NUMBERS.")
        print("I WIN.\u0007")
    elif running_correct < 10:
        print("I GUESSED LESS THAN 1/3 OF YOUR NUMBERS.")
        print("YOU BEAT ME.  CONGRATULATIONS *****")
    else:
        print("I GUESSED EXACTLY 1/3 OF YOUR NUMBERS.")
        print("IT'S A TIE GAME.")


```

It looks like you're trying to create a program that simulates a game of Hangman where the user has to guess a number within a certain number of guesses. The program should keep track of the number of valid guesses, the number of runs made by the user, and the correct number of times the user has guessed the number.

The program starts by explaining to the user that they have successfully guessed a number and how many runs they made. It then sets a variable called valid\_numbers to a False and breaks out of the while loop that makes sure the user doesn't guess the number.

The program then loops through the numbers until the user runs out of guesses. Each time the program makes a guess, it uses a combination of the original author's method, the number of runs made by the user, and the random number to determine which number to guess. If the user chooses a number that is different from the original author's method, it prints out that the user has guessed incorrectly. If the user chooses a number that is the same as the original author's method, it increments the running\_correct counter and updates the number in the machine file.

The program also updates the number of valid guesses and continues the game loop until the user chooses to continue or a valid number is guessed. Finally, it prints out a summary of the runs made by the user and continues to the main menu.


```
def main() -> None:
    print_intro()
    if read_instruction_choice():
        print_instructions()

    a = 0
    b = 1
    c = 3

    m = [[1] * 3 for _ in range(27)]
    k = [[9] * 3 for _ in range(3)]
    l = [[3] * 3 for _ in range(9)]  # noqa: E741

    continue_game = True
    while continue_game:
        l[0][0] = 2
        l[4][1] = 2
        l[8][2] = 2
        z: float = 26
        z1: float = 8
        z2 = 2
        running_correct = 0

        for _round in range(1, 4):
            valid_numbers = False
            numbers = []
            while not valid_numbers:
                print()
                numbers = read_10_numbers()
                valid_numbers = True
                for number in numbers:
                    if number < 0 or number > 2:
                        print("ONLY USE THE DIGITS '0', '1', OR '2'.")
                        print("LET'S TRY AGAIN.")
                        valid_numbers = False
                        break

            print(
                "\n%-14s%-14s%-14s%-14s"
                % ("MY GUESS", "YOUR NO.", "RESULT", "NO. RIGHT")
            )

            for number in numbers:
                s = 0
                my_guess = 0
                for j in range(0, 3):
                    # What did the original author have in mind ?
                    # The first expression always results in 0 because a is always 0
                    s1 = a * k[z2][j] + b * l[int(z1)][j] + c * m[int(z)][j]
                    if s < s1:
                        s = s1
                        my_guess = j
                    elif s1 == s and random.random() >= 0.5:
                        my_guess = j

                result = ""

                if my_guess != number:
                    result = "WRONG"
                else:
                    running_correct += 1
                    result = "RIGHT"
                    m[int(z)][number] = m[int(z)][number] + 1
                    l[int(z1)][number] = l[int(z1)][number] + 1
                    k[int(z2)][number] = k[int(z2)][number] + 1
                    z = z - (z / 9) * 9
                    z = 3 * z + number
                print(
                    "\n%-14d%-14d%-14s%-14d"
                    % (my_guess, number, result, running_correct)
                )

                z1 = z - (z / 9) * 9
                z2 = number

        print_summary_report(running_correct)
        continue_game = read_continue_choice()

    print("\nTHANKS FOR THE GAME.")


```

这段代码是一个if语句，它会判断当前脚本是否被名为"__main__"的模块初始化。如果当前脚本被初始化为模块，那么就会执行if语句块内的内容。

在这个例子中，if语句块内只有一行内容，即输出字符串"main()"。这个输出会在程序运行时打印出来，无论当前脚本是否被正常初始化。

if __name__ == "__main__":
   main()

这段代码的作用是判断当前脚本是否被名为"__main__"的模块初始化，如果是，就执行main()函数，否则不做任何操作。


```
if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Even Wins

This is a game between you and the computer. To play, an odd number of objects (marbles, chips, matches) are placed in a row. You take turns with the computer picking up between one and four objects each turn. The game ends when there are no objects left, and the winner is the one with an even number of objects picked up.

Two versions of this game are included. While to the player they appear similar, the programming approach is quite different. EVEN WINS, the first version, is deterministic — i.e., the computer plays by fixed, good rules and is impossible to beat if you don’t know how to play the game. It always starts with 27 objects, although you may change this.

The second version, GAME OF EVEN WINS, is much more interesting because the computer starts out only knowing the rules of the game. Using simple techniques of artificial intelligence (cybernetics), the computer gradually learns to play this game from its mistakes until it plays a very good game. After 20 games, the computer is a challenge to beat. Variation in the human’s style of play seems to make the computer learn more quickly. If you plot the learning curve of this program, it closely resembles classical human learning curves from psychological experiments.

Eric Peters at DEC wrote the GAME OF EVEN WINS. The original author of EVEN WINS is unknown.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=60)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=75)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `35_Even_Wins/javascript/evenwins.js`

该代码将JavaScript函数转换为BASIC代码，并实现了两个函数：print()和input()。

print()函数将一个字符串打印到网页上的一个元素上，该元素在代码中是通过调用document.getElementById("output").appendChild(document.createTextNode(str))实现的。

input()函数允许用户输入一个字符串，并将其存储在变量input_str中。该函数通过调用document.getElementById("output").appendChild(input_element)和input_element.focus()实现了与用户交互。当用户按下键盘上的13时，input()函数会捕获到该事件，并打印从用户输入的值到网页上指定的元素中的字符串。


```
// EVEN WINS
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       input_element = document.createElement("INPUT");

                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      input_str = input_element.value;
                                                      document.getElementById("output").removeChild(input_element);
                                                      print(input_str);
                                                      print("\n");
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

```

This is a program that plays a game of Connect-the-Dotts where two players take turns picking up marbles until one of them has picked up all the marbles or their opponent wins. The game ends when one player has won or if no winner has been determined.
The program starts by explaining the rules and how the game will be played.
It then goes through the game of connect-the-dots, assigning a number of marbles based on the number of dots and how many of each color of marbles the player has.
It then checks if the player has won and if not, it gives the other player an opportunity to win by picking up some marbles.
It then continues to ask the player if they want to continue the game until one of them wins or if they want to exit the game.

It is important to note that the game is not complete and some of the statements in the program may not be implemented correctly.


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var ma = [];
var ya = [];

// Main program
async function main()
{
    print(tab(31) + "EVEN WINS\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    y1 = 0;
    m1 = 0;
    print("     THIS IS A TWO PERSON GAME CALLED 'EVEN WINS.'\n");
    print("TO PLAY THE GAME, THE PLAYERS NEED 27 MARBLES OR\n");
    print("OTHER OBJECTS ON A TABLE.\n");
    print("\n");
    print("\n");
    print("     THE 2 PLAYERS ALTERNATE TURNS, WITH EACH PLAYER\n");
    print("REMOVING FROM 1 TO 4 MARBLES ON EACH MOVE.  THE GAME\n");
    print("ENDS WHEN THERE ARE NO MARBLES LEFT, AND THE WINNER\n");
    print("IS THE ONE WITH AN EVEN NUMBER OF MARBLES.\n");
    print("\n");
    print("\n");
    print("     THE ONLY RULES ARE THAT (1) YOU MUST ALTERNATE TURNS,\n");
    print("(2) YOU MUST TAKE BETWEEN 1 AND 4 MARBLES EACH TURN,\n");
    print("AND (3) YOU CANNOT SKIP A TURN.\n");
    print("\n");
    print("\n");
    print("\n");
    while (1) {
        print("     TYPE A '1' IF YOU WANT TO GO FIRST, AND TYPE\n");
        print("A '0' IF YOU WANT ME TO GO FIRST.\n");
        c = parseInt(await input());
        print("\n");
        if (c != 0) {
            t = 27;
            print("\n");
            print("\n");
            print("\n");
            print("TOTAL= " + t + "\n");
            print("\n");
            print("\n");
            print("WHAT IS YOUR FIRST MOVE");
            m = 0;
        } else {
            t = 27;
            m = 2;
            print("\n");
            print("TOTAL= " + t + "\n");
            print("\n");
            m1 += m;
            t -= m;
        }
        while (1) {
            if (m) {
                print("I PICK UP " + m + " MARBLES.\n");
                if (t == 0)
                    break;
                print("\n");
                print("TOTAL= " + t + "\n");
                print("\n");
                print("     AND WHAT IS YOUR NEXT MOVE, MY TOTAL IS " + m1 + "\n");
            }
            while (1) {
                y = parseInt(await input());
                print("\n");
                if (y < 1 || y > 4) {
                    print("\n");
                    print("THE NUMBER OF MARBLES YOU MUST TAKE BE A POSITIVE\n");
                    print("INTEGER BETWEEN 1 AND 4.\n");
                    print("\n");
                    print("     WHAT IS YOUR NEXT MOVE?\n");
                    print("\n");
                } else if (y > t) {
                    print("     YOU HAVE TRIED TO TAKE MORE MARBLES THAN THERE ARE\n");
                    print("LEFT.  TRY AGAIN.\n");
                } else {
                    break;
                }
            }

            y1 += y;
            t -= y;
            if (t == 0)
                break;
            print("TOTAL= " + t + "\n");
            print("\n");
            print("YOUR TOTAL IS " + y1 + "\n");
            if (t < 0.5)
                break;
            r = t % 6;
            if (y1 % 2 != 0) {
                if (t >= 4.2) {
                    if (r <= 3.4) {
                        m = r + 1;
                        m1 += m;
                        t -= m;
                    } else if (r < 4.7 || r > 3.5) {
                        m = 4;
                        m1 += m;
                        t -= m;
                    } else {
                        m = 1;
                        m1 += m;
                        t -= m;
                    }
                } else {
                    m = t;
                    t -= m;
                    print("I PICK UP " + m + " MARBLES.\n");
                    print("\n");
                    print("TOTAL = 0\n");
                    m1 += m;
                    break;
                }
            } else {
                if (r < 1.5 || r > 5.3) {
                    m = 1;
                    m1 += m;
                    t -= m;
                } else {
                    m = r - 1;
                    m1 += m;
                    t -= m;
                    if (t < 0.2) {
                        print("I PICK UP " + m + " MARBLES.\n");
                        print("\n");
                        break;
                    }
                }
            }
        }
        print("THAT IS ALL OF THE MARBLES.\n");
        print("\n");
        print(" MY TOTAL IS " + m1 + ", YOUR TOTAL IS " + y1 +"\n");
        print("\n");
        if (m1 % 2 != 0) {
            print("     YOU WON.  DO YOU WANT TO PLAY\n");
        } else {
            print("     I WON.  DO YOU WANT TO PLAY\n");
        }
        print("AGAIN?  TYPE 1 FOR YES AND 0 FOR NO.\n");
        a1 = parseInt(await input());
        if (a1 == 0)
            break;
        m1 = 0;
        y1 = 0;
    }
    print("\n");
    print("OK.  SEE YOU LATER\n");
}

```

这道题是一个简单的编程题目，要求我们解释以下代码的作用，但不输出源代码。

代码如下：

```
main();
```

根据我对编程的基本了解，这个代码应该是用来启动一个程序或者脚本的。程序或脚本可以是一段代码，也可以是一个完整的应用程序。

但是，由于这道题没有给出具体的程序或脚本，我也无法给出具体的解释。所以，我的答案就是：这段代码的作用未知，可能是用来启动一个程序或脚本的。


```
main();

```