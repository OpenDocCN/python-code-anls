# BasicComputerGames源码解析 62

# `66_Number/javascript/number.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。它们的作用如下：

1. `print` 函数的作用是在文档中的某个元素（例如，id 为 `output` 的元素）的 `appendChild` 方法添加一个新的文本节点（例如，将字符串 `"str"` 添加到 `output` 元素的 `appendChild` 方法中）。
2. `input` 函数的作用是在用户输入一个字符串后，将其存储在变量 `input_str` 中，并将输入的字符串显示在文档中的某个元素上（例如，将用户输入的字符串显示在 `document.getElementById("output").appendChild` 方法中的 `appendChild` 方法中）。

`input` 函数中使用的是 Promise 类型的 Promise，其中的回调函数使用的是 document.createElement("INPUT") 创建一个输入元素，并使用 print() 函数将用户输入的字符串添加到该元素上。在回调函数中，使用 document.getElementById("output").appendChild(input_element) 将创建的输入元素添加到文档中的 "output" 元素上。然后使用 focus() 方法将输入元素聚焦，并监听 input 元素上的 keydown 事件，以便在用户按下回车键时接收输入。

当用户输入字符串并点击回车键时，将调用 `input` 函数，并将用户输入的字符串作为参数传递给它。函数会将输入的字符串显示在 "output" 元素上，并使用 print() 函数将其添加到文档中的 "output" 元素上。此外，函数还会将一个新的字符串添加到 "output" 元素上，以便在将来的输入中使用。


```
// NUMBER
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

This is a program that allows the user to guess numbers from 1 to 5, and the computer will randomly select one of the numbers. The user will earn or lose points based on how close they get to the guessed number, and they can earn a jackpot of double points if they get 500 points. The user can continue guessing until they either get 500 points or hit the jackpot.



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
    print(tab(33) + "NUMBER\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("YOU HAVE 100 POINTS.  BY GUESSING NUMBERS FROM 1 TO 5, YOU\n");
    print("CAN GAIN OR LOSE POINTS DEPENDING UPON HOW CLOSE YOU GET TO\n");
    print("A RANDOM NUMBER SELECTED BY THE COMPUTER.\n");
    print("\n");
    print("YOU OCCASIONALLY WILL GET A JACKPOT WHICH WILL DOUBLE(!)\n");
    print("YOUR POINT COUNT.  YOU WIN WHEN YOU GET 500 POINTS.\n");
    print("\n");
    p = 0;
    while (1) {
        do {
            print("GUESS A NUMBER FROM 1 TO 5");
            g = parseInt(await input());
        } while (g < 1 || g > 5) ;
        r = Math.floor(5 * Math.random() + 1);
        s = Math.floor(5 * Math.random() + 1);
        t = Math.floor(5 * Math.random() + 1);
        u = Math.floor(5 * Math.random() + 1);
        v = Math.floor(5 * Math.random() + 1);
        if (g == r) {
            p -= 5;
        } else if (g == s) {
            p += 5;
        } else if (g == t) {
            p += p;
            print("YOU HIT THE JACKPOT!!!\n");
        } else if (g == u) {
            p += 1;
        } else if (g == v) {
            p -= p * 0.5;
        }
        if (p <= 500) {
            print("YOU HAVE " + p + " POINTS.\n");
            print("\n");
        } else {
            print("!!!!YOU WIN!!!! WITH " + p + " POINTS.\n");
            break;
        }
    }
}

```

这道题是一个简单的 Python 代码，包含一个名为 `main()` 的函数。然而，这个函数并没有函数体，只是一个名为 `main()` 的空函数。空函数在 Python 中有什么作用呢？它们可以作为其他函数的参数，让我们在调用它们的时候提供具体的参数。所以，这道题的作用就是让我们提供一个空函数作为参数，让我们在调用这个函数的时候提供具体的参数。


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


# `66_Number/python/number.py`

这段代码是一个 Python 编写的基于赌博游戏（number guessing game）的程序。其目的是指导用户如何进行数字猜测游戏。

程序首先导入了 random 模块，以便从计算机的随机数生成器中获取随机整数。

然后定义了一个名为 print_instructions 的函数，该函数用于输出游戏说明。函数在游戏开始时被调用，随后在游戏过程中继续被调用，以便用户在猜测数字时了解他们当前的得分和奖金。

函数内部使用 print 函数输出游戏说明，然后使用 printf 函数获取用户的输入。函数还打印出奖金的详细信息，以及告诉用户何时可以获得奖金。最后，函数还告诉用户游戏何时结束，以及如何计算他们的得分。


```
"""
NUMBER

A number guessing (gambling) game.

Ported by Dave LeCompte
"""

import random


def print_instructions() -> None:
    print("YOU HAVE 100 POINTS.  BY GUESSING NUMBERS FROM 1 TO 5, YOU")
    print("CAN GAIN OR LOSE POINTS DEPENDING UPON HOW CLOSE YOU GET TO")
    print("A RANDOM NUMBER SELECTED BY THE COMPUTER.")
    print()
    print("YOU OCCASIONALLY WILL GET A JACKPOT WHICH WILL DOUBLE(!)")
    print("YOUR POINT COUNT.  YOU WIN WHEN YOU GET 500 POINTS.")
    print()


```

这段代码定义了两个函数，分别是 `fnr()` 和 `main()`。

`fnr()` 函数接受一个整数参数，并使用 `random.randint()` 函数生成一个介于 1 和 5 之间的随机整数，然后将其返回。这个函数的作用是产生一个随机的整数，用于主程序中的 `guess` 变量。

`main()` 函数的主要作用是让用户猜测 1 到 5 之间的随机整数，并在猜测正确或输出的边界值时给予提示。当用户猜测正确的点数时，程序会根据猜测结果奖励或扣除相应的分数，并输出当前的分数。当用户猜测错误的点数时，程序会根据猜测结果提示用户正确的答案，并根据题目描述中的规则进行相应的数值处理。

整段代码的主要作用是提供一个简单的游戏，让用户猜测 1 到 5 之间的随机整数，并通过不同的规则给予用户相应的奖励或扣除相应的分数。


```
def fnr() -> int:
    return random.randint(1, 5)


def main() -> None:
    print(" " * 33 + "NUMBER")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")

    print_instructions()

    points: float = 100

    while points <= 500:
        print("GUESS A NUMBER FROM 1 TO 5")
        guess = int(input())

        if (guess < 1) or (guess > 5):
            continue

        r = fnr()
        s = fnr()
        t = fnr()
        u = fnr()
        v = fnr()

        if guess == r:
            # lose 5
            points -= 5
        elif guess == s:
            # gain 5
            points += 5
        elif guess == t:
            # double!
            points += points
            print("YOU HIT THE JACKPOT!!!")
        elif guess == u:
            # gain 1
            points += 1
        elif guess == v:
            # lose half
            points = points - (points * 0.5)

        print(f"YOU HAVE {points} POINTS.")
        print()
    print(f"!!!!YOU WIN!!!! WITH {points} POINTS.")


```

这段代码是一个Python程序中的一个if语句。if语句可以用于在程序运行时检查文件是否符合某些特定的条件。如果文件符合条件，if语句会执行if语句内部的代码，否则将跳过if语句内部的代码。

在这段代码中，if __name__ == "__main__"：是一个非常简单的if语句，它的作用是判断当前程序是否处于__main__模块中。如果当前程序正处于__main__模块中，那么程序将跳过if语句内部的代码并继续执行程序的其余部分。否则，if语句内部的代码将被执行。

在if语句内部的代码中，有一个名为main的函数，它将定义一个函数体，但是不会输出任何东西。main函数体可能是用于执行某些操作或者输出一些信息，但是在这段代码中，它的作用是没有任何特定的行为。


```
if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by Anthony Rubick [AnthonyMichaelTDM](https://github.com/AnthonyMichaelTDM)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### One Check

In this game or puzzle, 48 checkers are placed on the two outside spaces of a standard 64-square checkerboard as shown:

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
| ● | ● | ● | ● | ● | ● | ● | ● |
| ● | ● | ● | ● | ● | ● | ● | ● |
| ● | ● |   |   |   |   | ● | ● |
| ● | ● |   |   |   |   | ● | ● |
| ● | ● |   |   |   |   | ● | ● |
| ● | ● |   |   |   |   | ● | ● |
| ● | ● | ● | ● | ● | ● | ● | ● |
| ● | ● | ● | ● | ● | ● | ● | ● |

The object is to remove as many checkers as possible by diagonal jumps (as in standard checkers).

It is easy to remove 30 to 39 checkers, a challenge to remove 40 to 44, and a substantial feat to remove 45 to 47.

The program was created and written by David Ahl.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=122)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=137)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `67_One_Check/csharp/Board.cs`

This is a C# class that represents a game of Connect-F Wildup. It includes a class for each piece, a class for each move, and a class for each player.

The `Piece` class represents a piece on the board, with properties for its position, owner, and type.

The `Move` class represents a move that can be made by a player, with properties for the piece it involves, the position of the piece, and the color of the piece.

The `Player` class represents a player in the game, with properties for their index, name, and the piece they play.

The `Game` class is the main class for the game, and includes methods for reporting the state of the game, and for displaying the board.

The `IReadWrite` interface is used to communicate between the game and the player's input.

The `Move` class has a `TryMove` method that checks if the move is valid, and a `GetReport` method that returns a report stringifying the moves made.

The `Piece` class has a `IsInRange` method that checks if the piece is within the range of the board, and a `IsTwoSpacesDiagonally` method that checks if the piece is on the diagonal of two spaces.

The `Game` class has a `PlayMove` method that prompts the player to enter a move, and a `TryMove` method that checks if the move is valid.

Note that this is just one possible implementation of a Connect-F Wildup game in C#, and that different implementations may have different properties and methods.


```
namespace OneCheck;

internal class Board
{
    private readonly bool[][] _checkers;
    private int _pieceCount;
    private int _moveCount;

    public Board()
    {
        _checkers = 
            Enumerable.Range(0, 8)
                .Select(r => Enumerable.Range(0, 8)
                    .Select(c => r <= 1 || r >= 6 || c <= 1 || c >= 6).ToArray())
                .ToArray();
        _pieceCount = 48;
    }

    private bool this[int index]
    {
        get => _checkers[index / 8][index % 8];
        set => _checkers[index / 8][index % 8] = value;
    }

    public bool PlayMove(IReadWrite io)
    {
        while (true)
        {
            var from = (int)io.ReadNumber(Prompts.From);
            if (from == 0) { return false; }

            var move = new Move { From = from - 1, To = (int)io.ReadNumber(Prompts.To) - 1 };

            if (TryMove(move)) 
            { 
                _moveCount++;
                return true; 
            }

            io.Write(Streams.IllegalMove);
        }
    }

    public bool TryMove(Move move)
    {
        if (move.IsInRange && move.IsTwoSpacesDiagonally && IsPieceJumpingPieceToEmptySpace(move))
        {
            this[move.From] = false;
            this[move.Jumped] = false;
            this[move.To] = true;
            _pieceCount--;
            return true;
        }

        return false;
    }

    private bool IsPieceJumpingPieceToEmptySpace(Move move) => this[move.From] && this[move.Jumped] && !this[move.To];

    public string GetReport() => string.Format(Formats.Results, _moveCount, _pieceCount);

    public override string ToString() => 
        string.Join(Environment.NewLine, _checkers.Select(r => string.Join(" ", r.Select(c => c ? " 1" : " 0"))));
}

```

# `67_One_Check/csharp/Game.cs`

这段代码是一个名为 "OneCheck.Game" 的类，它是一个游戏。这个类的构造函数接受一个 "IReadWrite" 类型的参数，表示游戏需要读写。

在 "Play" 方法中，首先创建一个名为 "Board" 的类，这个类可能包含游戏的所有状态和操作。

然后进入一个 do-while 循环，每次执行游戏的一个步骤。首先，游戏创建一个 "Board" 类的实例，并调用 "PlayMove" 方法来处理玩家的下一步行动。

在 "do" 块中，游戏读取一行玩家的棋步，并将其输出到控制台。然后，它再次调用 "PlayMove" 方法来获取下一行玩家的棋步。

在 "while" 块中，游戏等待玩家是否再次确认继续游戏。如果玩家确认，则游戏继续流程，否则它结束游戏。

最后，游戏输出一个 "bye" 字符串来表示游戏结束。


```
namespace OneCheck;

internal class Game
{
    private readonly IReadWrite _io;

    public Game(IReadWrite io)
    {
        _io = io;
    }

    public void Play()
    {
        _io.Write(Streams.Introduction);
        
        do
        {
            var board = new Board();
            do
            {
                _io.WriteLine(board);
                _io.WriteLine();
            } while (board.PlayMove(_io));

            _io.WriteLine(board.GetReport());
        } while (_io.ReadYesNo(Prompts.TryAgain) == "yes");

        _io.Write(Streams.Bye);
    }
}

```



该代码是一个名为 `IOExtensions` 的内部类，包含一个名为 `ReadYesNo` 的静态方法，该方法接受一个 `IReadWrite` 类型的参数 `io` 和一个字符串参数 `prompt`。

该方法的作用是在不间断地读取字符串 `prompt` 并输入 "是" 或 "否" 之后，返回用户输入的确认。如果用户输入 "是" 或 "否"，则会输出确认消息并继续等待下一步操作。如果用户在输入过程中按任意键，则会中断读取操作并返回一个随机字符串。

例如，当调用 `ReadYesNo` 方法并传入参数 `"Are you sure?"` 时，该方法将返回 "是"。如果用户输入 "否"，则会中断读取操作并返回一个随机字符串。如果用户在输入过程中按任意键，则会中断读取操作。


```
internal static class IOExtensions
{
    internal static string ReadYesNo(this IReadWrite io, string prompt)
    {
        while (true)
        {
            var response = io.ReadString(prompt).ToLower();

            if (response == "yes" || response == "no") { return response; }

            io.Write(Streams.YesOrNo);
        }
    }
}

```

# `67_One_Check/csharp/Move.cs`

这段代码定义了一个名为"Move"的内部类，用于计算两个数字之间的跳转。

具体来说，这个类包含两个整型成员变量"From"和"To"，分别表示要计算的起始和终止数字。还包含一个计算跳跃的算术运算符"/ 2"，用于将起始和终止数字除以2得到跳跃后的中间数字。

另外，这个类还包括两个判断条件：

- "IsInRange"使用"From >= 0 && From <= 63"的条件判断，说明只有当起始和终止数字在0到63的范围内时，这个条件才为真，也就是整个表达式"From >= 0 && From <= 63 && To >= 0 && To <= 63"为真时，"IsInRange"返回true。
- "IsTwoSpacesDiagonally"使用"RowDelta == 2 && ColumnDelta == 2"的条件判断，说明只有当行和列的差值都为2时，这个条件才为真，也就是"RowDelta"为2,"ColumnDelta"为2时，"IsTwoSpacesDiagonally"返回true。

最后，这个类的实例还有一个"RowDelta"和"ColumnDelta"成员变量，用于计算起始和终止数字相对于网格的行列偏移量。


```
namespace OneCheck;

internal class Move
{
    public int From { get; init; }
    public int To { get; init; }
    public int Jumped => (From + To) / 2;

    public bool IsInRange => From >= 0 && From <= 63 && To >= 0 && To <= 63;
    public bool IsTwoSpacesDiagonally => RowDelta == 2 && ColumnDelta == 2;
    private int RowDelta => Math.Abs(From / 8 - To / 8);
    private int ColumnDelta => Math.Abs(From % 8 - To % 8);
}
```

# `67_One_Check/csharp/Program.cs`

这段代码的作用是创建一个名为 "Game" 的类，该类使用 OneCheck.Resources.Resource 中的资源创建游戏窗口并输出游戏玩法。

具体来说，以下步骤是关键：

1. 在全局区域中使用 Games.Common.IO；
2. 在全局区域中使用 using OneCheck.Resources.Resource；
3. 在 Game 类中创建一个名为 "newGame" 的方法，该方法接收一个空字符串作为参数，并使用该字符串创建一个名为 "Game" 的类实例；
4. 在 "newGame" 方法的体内，使用 using OneCheck.Resources.Resource；
5. 在 "newGame" 方法中，调用 Games.Common.IO.Create；
6. 创建一个新的 Game 类实例后，使用该实例的 Play 方法播放游戏。


```
global using Games.Common.IO;
global using static OneCheck.Resources.Resource;
using OneCheck;

new Game(new ConsoleIO()).Play();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `67_One_Check/csharp/Resources/Resource.cs`



这段代码是一个自定义的序列化类，名为 `Resource`。它包含了多个静态类，每个类都是为了在需要时获取一些特定的字符串或格式而设计的。

具体来说，这段代码定义了以下几个序列化类：

- `Streams` 类：包含了四个静态的 `Stream` 类，分别是 Introduction、IllegalMove、YesOrNo 和 Bye，这些字符串用于显示不同类型的提示信息给用户。
- `Formats` 类：定义了一个静态的字符串类 `string`，用于存储格式化的结果。
- `Prompts` 类：定义了一个静态的字符串类 `string`，用于在用户需要重新输入数据时提供的提示信息。
- `Strings` 类：定义了一个静态的字符串类 `string`，包含了两个字符串常量 `TooManyColumns` 和 `TooManyRows`，用于在用户输入数据时判断是否达到了错误的界限。

`Resource` 类中包含了一些方法，用于获取不同类型的资源，包括 `GetStream`、`GetString`、`GetFormattedString` 和 `TryAgain`。其中，`GetStream` 方法可以获取指定名称的资源文件，并返回一个 `Stream` 对象，这个 `Stream` 对象可以被用于读取和写入资源文件的内容。

对于每个 `Stream`,`GetString` 方法返回给定的 `name` 参数的值，这个值已经被声明为 `string` 类型，在 `Streams` 类中定义了静态的 `Stream` 类。

另外，`Assembly.GetExecutingAssembly().GetManifestResourceStream` 方法用于获取应用程序中资源文件的位置，如果指定的名称或路径不存在，则会抛出异常。


```
using System.Reflection;
using System.Runtime.CompilerServices;

namespace OneCheck.Resources;

internal static class Resource
{
    internal static class Streams
    {
        public static Stream Introduction => GetStream();
        public static Stream IllegalMove => GetStream();
        public static Stream YesOrNo => GetStream();
        public static Stream Bye => GetStream();
    }

    internal static class Formats
    {
        public static string Results => GetString();
    }

    internal static class Prompts
    {
        public static string From => GetString();
        public static string To => GetString();
        public static string TryAgain => GetString();
    }

    internal static class Strings
    {
        public static string TooManyColumns => GetString();
        public static string TooManyRows => GetString();
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

# `67_One_Check/java/OneCheck.java`

这段代码是一个名为 "One Check" 的计算机游戏，基于 1970 年代 的 BASIC 游戏。它将 Basic 语言中的游戏逻辑与 Java 语言实现。这个游戏的主要目的是让用户通过输入字符与计算机进行交互，并尝试通过计算得到最高得分。

该代码的作用是提供一个计算机游戏，让用户通过输入字符进行交互，并计算得分。这个游戏的核心玩法是用户通过猜数字来猜测下一个数，每次猜测后，计算机会告诉用户猜测的数字是高还是低，直到用户猜中为止。


```
import java.util.Arrays;
import java.util.Scanner;

/**
 * Game of One Check
 * <p>
 * Based on the BASIC game of One Check here
 * https://github.com/coding-horror/basic-computer-games/blob/main/67%20One%20Check/onecheck.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 *
 * Converted from BASIC to Java by Darren Cardenas.
 */

```

This is a Java program that simulates a game of chess. The program uses a while loop to repeatedly ask the user to play a move or to retry a previously made move. The user can choose to try again to continue the game, or they can exit the game after one try.

The program also has a method called "printBoard" that prints the current state of the game board, which is updated each time the program is run.

Note that this program is a very basic implementation and does not include any features that would be considered modern or robust, such as chess piecing or move prioritization.


```
public class OneCheck {

  private final Scanner scan;  // For user input

  private enum Step {
    SHOW_INSTRUCTIONS, SHOW_BOARD, GET_MOVE, GET_SUMMARY, QUERY_RETRY
  }

  public OneCheck() {

    scan = new Scanner(System.in);

  }  // End of constructor OneCheck

  public void play() {

    showIntro();
    startGame();

  }  // End of method play

  private static void showIntro() {

    System.out.println(" ".repeat(29) + "ONE CHECK");
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    System.out.println("\n\n");

  }  // End of method showIntro

  private void startGame() {

    int fromSquare = 0;
    int numJumps = 0;
    int numPieces = 0;
    int square = 0;
    int startPosition = 0;
    int toSquare = 0;

    // Move legality test variables
    int fromTest1 = 0;
    int fromTest2 = 0;
    int toTest1 = 0;
    int toTest2 = 0;

    int[] positions = new int[65];

    Step nextStep = Step.SHOW_INSTRUCTIONS;

    String lineContent = "";
    String userResponse = "";

    // Begin outer while loop
    while (true) {

      // Begin switch
      switch (nextStep) {

        case SHOW_INSTRUCTIONS:

          System.out.println("SOLITAIRE CHECKER PUZZLE BY DAVID AHL\n");
          System.out.println("48 CHECKERS ARE PLACED ON THE 2 OUTSIDE SPACES OF A");
          System.out.println("STANDARD 64-SQUARE CHECKERBOARD.  THE OBJECT IS TO");
          System.out.println("REMOVE AS MANY CHECKERS AS POSSIBLE BY DIAGONAL JUMPS");
          System.out.println("(AS IN STANDARD CHECKERS).  USE THE NUMBERED BOARD TO");
          System.out.println("INDICATE THE SQUARE YOU WISH TO JUMP FROM AND TO.  ON");
          System.out.println("THE BOARD PRINTED OUT ON EACH TURN '1' INDICATES A");
          System.out.println("CHECKER AND '0' AN EMPTY SQUARE.  WHEN YOU HAVE NO");
          System.out.println("POSSIBLE JUMPS REMAINING, INPUT A '0' IN RESPONSE TO");
          System.out.println("QUESTION 'JUMP FROM ?'\n");
          System.out.println("HERE IS THE NUMERICAL BOARD:\n");

          nextStep = Step.SHOW_BOARD;
          break;

        case SHOW_BOARD:

          // Begin loop through all squares
          for (square = 1; square <= 57; square += 8) {

            lineContent = String.format("% -4d%-4d%-4d%-4d%-4d%-4d%-4d%-4d", square, square + 1, square + 2,
                                        square + 3, square + 4, square + 5, square + 6, square + 7);
            System.out.println(lineContent);

          }  // End loop through all squares

          System.out.println("");
          System.out.println("AND HERE IS THE OPENING POSITION OF THE CHECKERS.");
          System.out.println("");

          Arrays.fill(positions, 1);

          // Begin generating start positions
          for (square = 19; square <= 43; square += 8) {

            for (startPosition = square; startPosition <= square + 3; startPosition++) {

              positions[startPosition] = 0;

            }
          }  // End generating start positions

          numJumps = 0;

          printBoard(positions);

          nextStep = Step.GET_MOVE;
          break;

        case GET_MOVE:

          System.out.print("JUMP FROM? ");
          fromSquare = scan.nextInt();
          scan.nextLine();  // Discard newline

          // User requested summary
          if (fromSquare == 0) {
            nextStep = Step.GET_SUMMARY;
            break;
          }

          System.out.print("TO? ");
          toSquare = scan.nextInt();
          scan.nextLine();  // Discard newline
          System.out.println("");

          // Check legality of move
          fromTest1 = (int) Math.floor((fromSquare - 1.0) / 8.0);
          fromTest2 = fromSquare - 8 * fromTest1;
          toTest1 = (int) Math.floor((toSquare - 1.0) / 8.0);
          toTest2 = toSquare - 8 * toTest1;

          if ((fromTest1 > 7) ||
              (toTest1 > 7) ||
              (fromTest2 > 8) ||
              (toTest2 > 8) ||
              (Math.abs(fromTest1 - toTest1) != 2) ||
              (Math.abs(fromTest2 - toTest2) != 2) ||
              (positions[(toSquare + fromSquare) / 2] == 0) ||
              (positions[fromSquare] == 0) ||
              (positions[toSquare] == 1)) {

            System.out.println("ILLEGAL MOVE.  TRY AGAIN...");
            nextStep = Step.GET_MOVE;
            break;
          }

          positions[toSquare] = 1;
          positions[fromSquare] = 0;
          positions[(toSquare + fromSquare) / 2] = 0;
          numJumps++;

          printBoard(positions);

          nextStep = Step.GET_MOVE;
          break;

        case GET_SUMMARY:

          numPieces = 0;

          // Count remaining pieces
          for (square = 1; square <= 64; square++) {
            numPieces += positions[square];
          }

          System.out.println("");
          System.out.println("YOU MADE " + numJumps + " JUMPS AND HAD " + numPieces + " PIECES");
          System.out.println("REMAINING ON THE BOARD.\n");

          nextStep = Step.QUERY_RETRY;
          break;

        case QUERY_RETRY:

          while (true) {
            System.out.print("TRY AGAIN? ");
            userResponse = scan.nextLine();
            System.out.println("");

            if (userResponse.toUpperCase().equals("YES")) {
              nextStep = Step.SHOW_BOARD;
              break;
            }
            else if (userResponse.toUpperCase().equals("NO")) {
              System.out.println("O.K.  HOPE YOU HAD FUN!!");
              return;
            }
            else {
              System.out.println("PLEASE ANSWER 'YES' OR 'NO'.");
            }
          }
          break;

        default:
          System.out.println("INVALID STEP");
          nextStep = Step.QUERY_RETRY;
          break;

      }  // End of switch

    }  // End outer while loop

  }  // End of method startGame

  public void printBoard(int[] positions) {

    int column = 0;
    int row = 0;
    String lineContent = "";

    // Begin loop through all rows
    for (row = 1; row <= 57; row += 8) {

      // Begin loop through all columns
      for (column = row; column <= row + 7; column++) {

        lineContent += " " + positions[column];

      }  // End loop through all columns

      System.out.println(lineContent);
      lineContent = "";

    }  // End loop through all rows

    System.out.println("");

  }  // End of method printBoard

  public static void main(String[] args) {

    OneCheck game = new OneCheck();
    game.play();

  }  // End of method main

}  // End of class OneCheck

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `67_One_Check/javascript/onecheck.js`

这段代码的作用是向网页上的一个 id 为“output”的 text 元素中输入用户输入的字符串，并在用户点击回车时将其输出到控制台。

具体来说，该代码包含两个函数：`print()` 和 `input()`。

`print()` 函数将一个字符串作为参数，将其添加到网页上的“output”元素的文本内容中。这个“output”元素在网页上可能来自于用户的表单或者其他 text 元素，具体来源未知。

`input()` 函数是一个 Promise 函数，它接受一个回调函数作为参数。该函数的作用是获取用户输入的字符串，并在用户点击回车时将其作为参数传递给 `print()` 函数，并将其添加到“output”元素的文本内容中。

在 `input()` 函数中，首先创建一个 `INPUT` 元素，设置其 `type` 属性为“text”，并设置其 `length` 属性为“50”，这可能是为了确保该元素可以接收一个最大长度为 50 的字符串。然后将该元素添加到网页上的“output”元素中，并将其设置为不可编辑，以便用户无法修改其中的内容。

接着，该函数监听 `INPUT` 元素的 `keydown` 事件，当用户按下回车键时，该事件会触发。在事件处理程序中，通过 `event.keyCode` 获取用户输入的键码，如果键码为 13（即回车键），则获取用户输入的字符串，并将其添加到“output”元素的文本内容中。然后，该函数会将该字符串输出到控制台，并将其中的换行符 \n 也一并输出。

最后，该函数调用 `print()` 函数，并将用户输入的字符串作为参数传递给它，将其添加到“output”元素的文本内容中。


```
// ONE CHECK
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

This is a program written in JavaScript that allows the user to play a game of Connect-the-Dot on a chessboard. The game starts with a checkered board and the player is given 20 moves to make, trying to connect any non-adjacent dots on the board. If the player is unable to make a valid move after 20 attempts, the game is over and the player is given a summary of the number of moves made and the result of the game.

The code uses the `AI` algorithm, which thinks like a human player, to connect the dots on the board. The algorithm uses a `2048` puzzle as the algorithm's base and generates a new puzzle every time the player makes a move. The algorithm has the advantage of not having limited resources, but the disadvantage of not being able to think like a human player.


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var a = [];

// Main program
async function main()
{
    print(tab(30) + "ONE CHECK\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    for (i = 0; i <= 64; i++)
        a[i] = 0;
    print("SOLITAIRE CHECKER PUZZLE BY DAVID AHL\n");
    print("\n");
    print("48 CHECKERS ARE PLACED ON THE 2 OUTSIDE SPACES OF A\n");
    print("STANDARD 64-SQUARE CHECKERBOARD.  THE OBJECT IS TO\n");
    print("REMOVE AS MANY CHECKERS AS POSSIBLE BY DIAGONAL JUMPS\n");
    print("(AS IN STANDARD CHECKERS).  USE THE NUMBERED BOARD TO\n");
    print("INDICATE THE SQUARE YOU WISH TO JUMP FROM AND TO.  ON\n");
    print("THE BOARD PRINTED OUT ON EACH TURN '1' INDICATES A\n");
    print("CHECKER AND '0' AN EMPTY SQUARE.  WHEN YOU HAVE NO\n");
    print("POSSIBLE JUMPS REMAINING, INPUT A '0' IN RESPONSE TO\n");
    print("QUESTION 'JUMP FROM ?'\n");
    print("\n");
    print("HERE IS THE NUMERICAL BOARD:\n");
    print("\n");
    while (1) {
        for (j = 1; j <= 57; j += 8) {
            str = "";
            for (i = 0; i <= 7; i++) {
                while (str.length < 4 * i)
                    str += " ";
                str += " " + (j + i);
            }
            print(str + "\n");
        }
        print("\n");
        print("AND HERE IS THE OPENING POSITION OF THE CHECKERS.\n");
        print("\n");
        for (j = 1; j <= 64; j++)
            a[j] = 1;
        for (j = 19; j <= 43; j += 8)
            for (i = j; i <= j + 3; i++)
                a[i] = 0;
        m = 0;
        while (1) {
            // Print board
            for (j = 1; j <= 57; j += 8) {
                str = "";
                for (i = j; i <= j + 7; i++) {
                    str += " " + a[i] + " ";
                }
                print(str + "\n");
            }
            print("\n");
            while (1) {
                print("JUMP FROM");
                f = parseInt(await input());
                if (f == 0)
                    break;
                print("TO");
                t = parseInt(await input());
                print("\n");
                // Check legality of move
                f1 = Math.floor((f - 1) / 8);
                f2 = f - 8 * f1;
                t1 = Math.floor((t - 1) / 8);
                t2 = t - 8 * t1;
                if (f1 > 7 || t1 > 7 || f2 > 8 || t2 > 8 || Math.abs(f1 - t1) != 2 || Math.abs(f2 - t2) != 2 || a[(t + f) / 2] == 0 || a[f] == 0 || a[t] == 1) {
                    print("ILLEGAL MOVE.  TRY AGAIN...\n");
                    continue;
                }
                break;
            }
            if (f == 0)
                break;
            // Update board
            a[t] = 1;
            a[f] = 0;
            a[(t + f) / 2] = 0;
            m++;
        }
        // End game summary
        s = 0;
        for (i = 1; i <= 64; i++)
            s += a[i];
        print("\n");
        print("YOU MADE " + m + " JUMPS AND HAD " + s + " PIECES\n");
        print("REMAINING ON THE BOARD.\n");
        print("\n");
        while (1) {
            print("TRY AGAIN");
            str = await input();
            if (str == "YES")
                break;
            if (str == "NO")
                break;
            print("PLEASE ANSWER 'YES' OR 'NO'.\n");
        }
        if (str == "NO")
            break;
    }
    print("\n");
    print("O.K.  HOPE YOU HAD FUN!!\n");
}

```

这道题目没有提供代码，因此无法给出具体的解释。一般来说，在编程中，`main()` 函数是程序的入口点，当程序运行时，首先会执行这个函数。`main()` 函数可以是程序中的任何函数或代码片段，因此它的作用取决于程序的具体结构。在某些程序中，`main()` 函数可能没有特定的含义，而仅仅是一个程序计数器，程序运行结束后，计数器就会被重置。


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


# `67_One_Check/python/onecheck.py`

Initial instructions for the Solitaire Checker game:

1. The game is called "SOLITAIRE CHECKER PUZZLE BY DAVID AHL".
2. The objective is to remove as many checkers as possible by diagonal jumps.
3. The game is played on a standard 64-square checkerboard.
4. The checkers are placed on the 2 outside spaces of a standard 64-square checkerboard.
5. The player's objective is to remove as many checkers as possible by diagonal jumps.
6. The board is printed out on each turn.
7. The numbers on the board indicate the square the player wants to jump from.
8. The player can input the number of the square they want to jump from and the number of moves they want to make on their first turn.
9. The player will make moves based on the numbers on the board.
10. The player will continue until they have either eaten all the checkers or cannot make any more moves.
11. If the player has eaten all the checkers, they will advance to the next level.
12. If the player cannot make any more moves, they will be taken to the end of the game.

Note: There is no paper for the game, just the instructions on how to play the game.


```
"""
ONE CHECK

Port to Python by imiro
"""

from typing import Tuple


def main() -> None:
    # Initial instructions
    print(" " * 30 + "ONE CHECK")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")
    print("SOLITAIRE CHECKER PUZZLE BY DAVID AHL\n")
    print("48 CHECKERS ARE PLACED ON THE 2 OUTSIDE SPACES OF A")
    print("STANDARD 64-SQUARE CHECKERBOARD.  THE OBJECT IS TO")
    print("REMOVE AS MANY CHECKERS AS POSSIBLE BY DIAGONAL JUMPS")
    print("(AS IN STANDARD CHECKERS).  USE THE NUMBERED BOARD TO")
    print("INDICATE THE SQUARE YOU WISH TO JUMP FROM AND TO.  ON")
    print("THE BOARD PRINTED OUT ON EACH TURN '1' INDICATES A")
    print("CHECKER AND '0' AN EMPTY SQUARE.  WHEN YOU HAVE NO")
    print("POSSIBLE JUMPS REMAINING, INPUT A '0' IN RESPONSE TO")
    print("QUESTION 'JUMP FROM ?'\n")
    print("HERE IS THE NUMERICAL BOARD:\n")

    while True:
        for j in range(1, 64, 8):
            for i in range(j, j + 7):
                print(i, end=(" " * (3 if i < 10 else 2)))
            print(j + 7)
        print("\nAND HERE IS THE OPENING POSITION OF THE CHECKERS.\n")

        (jumps, left) = play_game()

        print()
        print(f"YOU MADE {jumps} JUMPS AND HAD {left} PIECES")
        print("REMAINING ON THE BOARD.\n")

        if not (try_again()):
            break

    print("\nO.K.  HOPE YOU HAD FUN!!")


```

This is a Python implementation of a board game where the player can "jump" to a specific position on the board. The board is indexed in 1-based indexing, with the numbers 1-70 for the numbers 1-14, and the numbers 71-72 for the numbers 14-28. The numbers on the board range from 1 to 44 for the players to jump from, and the numbers 45 to 72 for the players to jump to.

The code first defines the board as a 2D list with 70 elements, with 71 elements for the row number and 72 elements for the column number. The player is able to move 8 squares in one step. The code then generates a board with all elements initialized to zero.

The code then generates a board with the specified squares being 1 and all others being 0. The jumps made by the player is then counted and stored in the variable "jumps".

The code then allows the player to "jump" to a specific position by the player. The player is first prompted to choose the number from which they want to jump, and then the number is added to the board. The code then checks if the move is legal, meaning if the move is within the legal range of the game (0-44 for the row, 0-3 for the column). If the move is legal, the board is updated accordingly.

The code also updates the display of the board, printing the current state of the board for each square.

The code then enters a loop that prints the board, until the user breaks the loop by pressing "C".


```
def play_game() -> Tuple[str, str]:
    # Initialize board
    # Give more than 64 elements to accomodate 1-based indexing
    board = [1] * 70
    for j in range(19, 44, 8):
        for i in range(j, j + 4):
            board[i] = 0
    jumps = 0
    while True:
        # print board
        for j in range(1, 64, 8):
            for i in range(j, j + 7):
                print(board[i], end=" ")
            print(board[j + 7])
        print()

        while True:
            print("JUMP FROM", end=" ")
            f_str = input()
            f = int(f_str)
            if f == 0:
                break
            print("TO", end=" ")
            t_str = input()
            t = int(t_str)
            print()

            # Check legality of move
            f1 = (f - 1) // 8
            f2 = f - 8 * f1
            t1 = (t - 1) // 8
            t2 = t - 8 * t1
            if (
                f1 > 7
                or t1 > 7
                or f2 > 8
                or t2 > 8
                or abs(f1 - t1) != 2
                or abs(f2 - t2) != 2
                or board[(t + f) // 2] == 0
                or board[f] == 0
                or board[t] == 1
            ):
                print("ILLEGAL MOVE.  TRY AGAIN...")
                continue
            break

        if f == 0:
            break
        board[t] = 1
        board[f] = 0
        board[(t + f) // 2] = 0
        jumps = jumps + 1

    left = 0
    for i in range(1, 64 + 1):
        left = left + board[i]
    return (str(jumps), str(left))


```

这段代码是一个 Python 函数 `try_again()`，它尝试从用户那里获取一个答案（可能是 "是" 或 "否"）。如果用户给出的答案是 "是"，那么函数将返回 `True`，否则返回 `False`。

函数内部首先打印 "TRY AGAIN" 并等待用户输入答案。然后，它将把用户输入的答案转换为大写并将其存储在变量 `answer` 中。

接下来，函数将检查用户输入的答案是否为 "是"。如果是，函数将返回 `True`，否则返回 `False`。最后，函数将打印一条消息，要求用户再次尝试或重新输入他们的答案。然后，它将调用自身并继续等待用户输入。

在 `if __name__ == "__main__":` 语句中，我们创建了一个主函数，该函数调用 `try_again()` 函数。如果 `try_again()` 返回 `True`，主函数将打印 "恭喜你，你答对啦！"。如果返回 `False`，主函数将打印 "很遗憾，答错了。请再试一次！"。


```
def try_again() -> bool:
    print("TRY AGAIN", end=" ")
    answer = input().upper()
    if answer == "YES":
        return True
    elif answer == "NO":
        return False
    print("PLEASE ANSWER 'YES' OR 'NO'.")
    return try_again()


if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Orbit

ORBIT challenges you to visualize spacial positions in polar coordinates. The object is to detonate a Photon explosive within a certain distance of a germ laden Romulan spaceship. This ship is orbiting a planet at a constant altitude and orbital rate (degrees/hour). The location of the ship is hidden by a device that renders the ship invisible, but after each bomb you are told how close to the enemy ship your bomb exploded. The challenge is to hit an invisible moving target with a limited number of shots.

The planet can be replaced by a point at its center (called the origin); then the ship’s position can be given as a distance form the origin and an angle between its position and the eastern edge of the planet.

```
direction
of orbit    <       ^ ship
              \     ╱
                \  ╱ <
                 |╱   \
                 ╱      \
                ╱         \
               ╱           | angle
              ╱           /
             ╱          /
            ╱         /
           ╱——————————————————— E

```

The distance of the bomb from the ship is computed using the law of cosines. The law of cosines states:

```
D = SQUAREROOT( R**2 + D1**2 - 2*R*D1*COS(A-A1) )
```

Where D is the distance between the ship and the bomb, R is the altitude of the ship, D1 is the altitude of the bomb, and A-A1 is the angle between the ship and the bomb.


```
                 bomb  <
                        ╲                   ^ ship
                         ╲                  ╱
                          ╲                ╱ <
                           ╲              ╱   \
                        D1  ╲            ╱      \
                             ╲        R ╱         \
                              ╲   A1   ╱           | A
                               ╲⌄——— ◝╱           /
                                ╲    ╱ \        /
                                 ╲  ╱   \      /
                                  ╲╱───────────────────── E

```

ORBIT was originally called SPACE WAR and was written by Jeff Lederer of Project SOLO Pittsburgh, Pennsylvania.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=124)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=139)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `68_Orbit/csharp/program.cs`

This appears to be a game programmed in C# that is simulating a hypothetical first contact scenario between humanity and the Romulan Empire. It is written for the Unity game engine and appears to include elements such as a显示出你拥有技能等级的界面上、一个球形的地图以及一个展示你距离目标的距离的 HUD。游戏中的玩家可以操纵一个拥有不同技能的游戏角色，在游戏过程中可以打击敌人，或者使用各种武器来获得优势。


```
using System.Text;

namespace Orbit
{
    class Orbit
    {
        private void DisplayIntro()
        {
            Console.WriteLine();
            Console.WriteLine("ORBIT".PadLeft(23));
            Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine("");
            Console.WriteLine("SOMEWHERE ABOVE YOUR PLANET IS A ROMULAN SHIP.");
            Console.WriteLine();
            Console.WriteLine("THE SHIP IS IN A CONSTANT POLAR ORBIT.  ITS");
            Console.WriteLine("DISTANCE FROM THE CENTER OF YOUR PLANET IS FROM");
            Console.WriteLine("10,000 TO 30,000 MILES AND AT ITS PRESENT VELOCITY CAN");
            Console.WriteLine("CIRCLE YOUR PLANET ONCE EVERY 12 TO 36 HOURS.");
            Console.WriteLine();
            Console.WriteLine("UNFORTUNATELY, THEY ARE USING A CLOAKING DEVICE SO");
            Console.WriteLine("YOU ARE UNABLE TO SEE THEM, BUT WITH A SPECIAL");
            Console.WriteLine("INSTRUMENT YOU CAN TELL HOW NEAR THEIR SHIP YOUR");
            Console.WriteLine("PHOTON BOMB EXPLODED.  YOU HAVE SEVEN HOURS UNTIL THEY");
            Console.WriteLine("HAVE BUILT UP SUFFICIENT POWER IN ORDER TO ESCAPE");
            Console.WriteLine("YOUR PLANET'S GRAVITY.");
            Console.WriteLine();
            Console.WriteLine("YOUR PLANET HAS ENOUGH POWER TO FIRE ONE BOMB AN HOUR.");
            Console.WriteLine();
            Console.WriteLine("AT THE BEGINNING OF EACH HOUR YOU WILL BE ASKED TO GIVE AN");
            Console.WriteLine("ANGLE (BETWEEN 0 AND 360) AND A DISTANCE IN UNITS OF");
            Console.WriteLine("100 MILES (BETWEEN 100 AND 300), AFTER WHICH YOUR BOMB'S");
            Console.WriteLine("DISTANCE FROM THE ENEMY SHIP WILL BE GIVEN.");
            Console.WriteLine();
            Console.WriteLine("AN EXPLOSION WITHIN 5,000 MILES OF THE ROMULAN SHIP");
            Console.WriteLine("WILL DESTROY IT.");
            Console.WriteLine();
            Console.WriteLine("BELOW IS A DIAGRAM TO HELP YOU VISUALIZE YOUR PLIGHT.");
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine("                          90");
            Console.WriteLine("                    0000000000000");
            Console.WriteLine("                 0000000000000000000");
            Console.WriteLine("               000000           000000");
            Console.WriteLine("             00000                 00000");
            Console.WriteLine("            00000    XXXXXXXXXXX    00000");
            Console.WriteLine("           00000    XXXXXXXXXXXXX    00000");
            Console.WriteLine("          0000     XXXXXXXXXXXXXXX     0000");
            Console.WriteLine("         0000     XXXXXXXXXXXXXXXXX     0000");
            Console.WriteLine("        0000     XXXXXXXXXXXXXXXXXXX     0000");
            Console.WriteLine("180<== 00000     XXXXXXXXXXXXXXXXXXX     00000 ==>0");
            Console.WriteLine("        0000     XXXXXXXXXXXXXXXXXXX     0000");
            Console.WriteLine("         0000     XXXXXXXXXXXXXXXXX     0000");
            Console.WriteLine("          0000     XXXXXXXXXXXXXXX     0000");
            Console.WriteLine("           00000    XXXXXXXXXXXXX    00000");
            Console.WriteLine("            00000    XXXXXXXXXXX    00000");
            Console.WriteLine("             00000                 00000");
            Console.WriteLine("               000000           000000");
            Console.WriteLine("                 0000000000000000000");
            Console.WriteLine("                    0000000000000");
            Console.WriteLine("                         270");
            Console.WriteLine();
            Console.WriteLine("X - YOUR PLANET");
            Console.WriteLine("O - THE ORBIT OF THE ROMULAN SHIP");
            Console.WriteLine();
            Console.WriteLine("ON THE ABOVE DIAGRAM, THE ROMULAN SHIP IS CIRCLING");
            Console.WriteLine("COUNTERCLOCKWISE AROUND YOUR PLANET.  DON'T FORGET THAT");
            Console.WriteLine("WITHOUT SUFFICIENT POWER THE ROMULAN SHIP'S ALTITUDE");
            Console.WriteLine("AND ORBITAL RATE WILL REMAIN CONSTANT.");
            Console.WriteLine();
            Console.WriteLine("GOOD LUCK.  THE FEDERATION IS COUNTING ON YOU.");
       }

        private bool PromptYesNo(string Prompt)
        {
            bool Success = false;

            while (!Success)
            {
                Console.Write(Prompt);
                string LineInput = Console.ReadLine().Trim().ToLower();

                if (LineInput.Equals("yes"))
                    return true;
                else if (LineInput.Equals("no"))
                    return false;
                else
                    Console.WriteLine("Yes or No");
            }

            return false;
        }

        private int PromptForNumber(string Prompt)
        {
            bool InputSuccess = false;
            int ReturnResult = 0;

            while (!InputSuccess)
            {
                Console.Write(Prompt);
                string Input = Console.ReadLine().Trim();
                InputSuccess = int.TryParse(Input, out ReturnResult);
                if (!InputSuccess)
                    Console.WriteLine("*** Please enter a valid number ***");
            }   

            return ReturnResult;
        }

        private void PlayOneRound()
        {
            Random rand = new Random();
            string Prompt = "";

            int A_AngleToShip = 0;
            int D_DistanceFromBombToShip = 0;
            int R_DistanceToShip = 0;
            int H_Hour = 0;
            int A1_Angle = 0;
            int D1_DistanceForDetonation = 0;
            int T = 0;
            double C_ExplosionDistance = 0;

            A_AngleToShip = Convert.ToInt32(360 * rand.NextDouble());
            D_DistanceFromBombToShip = Convert.ToInt32(200 * rand.NextDouble()) + 200;
            R_DistanceToShip = Convert.ToInt32(20 * rand.NextDouble()) + 10;

            while (H_Hour < 7)
            {
                H_Hour++;

                Console.WriteLine();
                Console.WriteLine();
                Prompt = "This is hour " + H_Hour.ToString() + ", at what angle do you wish to send\nyour photon bomb? ";
                A1_Angle = PromptForNumber(Prompt);

                D1_DistanceForDetonation = PromptForNumber("How far out do you wish to detonate it? ");

                Console.WriteLine();
                Console.WriteLine();

                A_AngleToShip += R_DistanceToShip;
                if (A_AngleToShip >= 360)
                    A_AngleToShip -= 360;

                T = Math.Abs(A_AngleToShip = A1_Angle);
                if (T >= 180)
                    T = 360 - T;

                C_ExplosionDistance = Math.Sqrt(D_DistanceFromBombToShip * D_DistanceFromBombToShip + D1_DistanceForDetonation * 
                                                D1_DistanceForDetonation - 2 * D_DistanceFromBombToShip * D1_DistanceForDetonation * 
                                                Math.Cos(T * 3.14159 / 180));
                
                Console.WriteLine("Your photon bomb exploded {0:N3}*10^2 miles from the", C_ExplosionDistance);
                Console.WriteLine("Romulan ship.");

                if (C_ExplosionDistance <= 50)
                {
                    Console.WriteLine("You have successfully completed your mission.");
                    return;
                }
            }

            Console.WriteLine("You allowed the Romulans to escape.");
            return;
 
        }

        public void Play()
        {
            bool ContinuePlay = true;

            DisplayIntro();

            do 
            {
                PlayOneRound();

                Console.WriteLine("Another Romulan ship has gone in to orbit.");
                ContinuePlay = PromptYesNo("Do you wish to try to destroy it? ");
            }
            while (ContinuePlay);
            
            Console.WriteLine("Good bye.");
        }
    }
    class Program
    {
        static void Main(string[] args)
        {

            new Orbit().Play();

        }
    }
}
```