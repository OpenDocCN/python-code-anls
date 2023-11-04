# BasicComputerGames源码解析 84

# `89_Tic-Tac-Toe/python/TicTacToe_Hard.py`

It looks like you are implementing a `D格雷纲领游戏` for the board game `格雷纲领游戏` 中的 `gap` 变量。你的代码实现了以下功能：

1. 尝试所有可移动的位置，并计算出当前棋盘状态与 `gap` 之间的差值。
2. 如果当前棋盘状态中的 `gap` 可用，则替换 `gap` 并返回替换后的差值。
3. 如果 `gap` 不可移动，则提示 `RuntimeError`。
4. 如果棋盘状态中的所有位置都不可用移动，则提示 `RuntimeError`。

这是一个非常基本的游戏逻辑，但可以作为一个为基础的框架，你可以添加更多的游戏逻辑，例如：处理棋盘状态、玩家操作等。


```
from typing import List, Tuple, Union


class TicTacToe:
    def __init__(self, pick, sz=3) -> None:
        self.pick = pick
        self.dim_sz = sz
        self.board = self.clear_board()

    def clear_board(self) -> List[List[str]]:
        board = [["blur" for i in range(self.dim_sz)] for j in range(self.dim_sz)]
        # made a 3x3 by-default board
        return board

    def move_record(self, r, c) -> Union[str, bool]:
        if r > self.dim_sz or c > self.dim_sz:
            return "Out of Bounds"
        if self.board[r][c] != "blur":
            return "Spot Pre-Occupied"
        self.board[r][c] = self.pick
        return True

    def check_win(self) -> int:  # 1 you won, 0 computer won, -1 tie
        # Flag syntax -> first player no. ,
        # User is Player#1 ;
        # Check set 1 -> row and '\' diagonal & Check set 2 -> col and '/' diagonal

        for i in range(0, self.dim_sz):  # Rows
            flag11 = True
            flag21 = True

            flag12 = True
            flag22 = True
            for j in range(0, self.dim_sz):

                ch2 = self.board[i][j]
                ch1 = self.board[j][i]
                # Row
                if ch1 == self.pick:  # if it's mine, computer didn't make it
                    flag21 = False
                elif ch1 == "blur":  # if it's blank no one made it
                    flag11 = False
                    flag21 = False
                else:
                    flag11 = False  # else i didn't make it

                if ch2 == self.pick:  # Same but for Col
                    flag22 = False
                elif ch2 == "blur":
                    flag12 = False
                    flag22 = False
                else:
                    flag12 = False

            if flag11 is True or flag12 is True:  # I won
                return 1
            if flag21 is True or flag22 is True:  # Computer Won
                return 0

        # Diagonals#
        flag11 = True
        flag21 = True

        flag12 = True
        flag22 = True
        for i in range(0, self.dim_sz):

            ch2 = self.board[i][i]
            ch1 = self.board[i][self.dim_sz - 1 - i]

            if ch1 == self.pick:
                flag21 = False
            elif ch1 == "blur":
                flag11 = False
                flag21 = False
            else:
                flag11 = False

            if ch2 == self.pick:
                flag22 = False
            elif ch2 == "blur":
                flag12 = False
                flag22 = False
            else:
                flag12 = False

        if flag11 or flag12:
            return 1
        if flag21 or flag22:
            return 0

        return -1

    def next_move(self) -> Union[Tuple[int, int], Tuple[List[int], List[int]]]:
        available_moves = []  # will carry all available moves
        player_win_spot = []  # if player (user Wins)
        comp_pick = "O"
        if self.pick == "O":
            comp_pick = "X"
        for i in range(0, self.dim_sz):
            for j in range(0, self.dim_sz):

                if self.board[i][j] == "blur":  # BLANK
                    t = (i, j)
                    available_moves.append(t)  # add it to available moves
                    self.board[i][j] = comp_pick  # Check if I (Computer can win)
                    if self.check_win() == 0:  # Best Case I(Computer) win!
                        return i, j
                    self.board[i][j] = self.pick
                    if (
                        self.check_win() == 1
                    ):  # Second Best Case, he (player) didn't won
                        player_win_spot.append(t)
                    self.board[i][j] = "blur"

        if len(player_win_spot) != 0:
            self.board[player_win_spot[0][0]][player_win_spot[0][1]] = comp_pick
            return player_win_spot[0][0], player_win_spot[0][1]
        if len(available_moves) == 1:
            self.board[available_moves[0][0]][available_moves[0][1]] = comp_pick
            return [available_moves[0][0]], [available_moves[0][1]]
        if len(available_moves) == 0:
            return -1, -1

        c1, c2 = self.dim_sz // 2, self.dim_sz // 2
        if (c1, c2) in available_moves:  # CENTER
            self.board[c1][c2] = comp_pick
            return c1, c2
        for i in range(c1 - 1, -1, -1):  # IN TO OUT
            gap = c1 - i
            # checking  - 4 possibilities at max
            # EDGES
            if (c1 - gap, c2 - gap) in available_moves:
                self.board[c1 - gap][c2 - gap] = comp_pick
                return c1 - gap, c2 - gap
            if (c1 - gap, c2 + gap) in available_moves:
                self.board[c1 - gap][c2 + gap] = comp_pick
                return c1 - gap, c2 + gap
            if (c1 + gap, c2 - gap) in available_moves:
                self.board[c1 + gap][c2 - gap] = comp_pick
                return c1 + gap, c2 - gap
            if (c1 + gap, c2 + gap) in available_moves:
                self.board[c1 + gap][c2 + gap] = comp_pick
                return c1 + gap, c2 + gap

            # Four Lines

            for i in range(0, gap):
                if (c1 - gap, c2 - gap + i) in available_moves:  # TOP LEFT TO TOP RIGHT
                    self.board[c1 - gap][c2 - gap + i] = comp_pick
                    return c1 - gap, c2 - gap + i
                if (
                    c1 + gap,
                    c2 - gap + i,
                ) in available_moves:  # BOTTOM LEFT TO BOTTOM RIGHT
                    self.board[c1 + gap][c2 - gap + i] = comp_pick
                    return c1 + gap, c2 - gap + i
                if (c1 - gap, c2 - gap) in available_moves:  # LEFT TOP TO LEFT BOTTOM
                    self.board[c1 - gap + i][c2 - gap] = comp_pick
                    return c1 - gap + i, c2 - gap
                if (
                    c1 - gap + i,
                    c2 + gap,
                ) in available_moves:  # RIGHT TOP TO RIGHT BOTTOM
                    self.board[c1 - gap + i][c2 + gap] = comp_pick
                    return c1 - gap + i, c2 + gap
        raise RuntimeError("No moves available")


```

这是一个在TicTacToe游戏中的函数，名为`display()`。"display()"函数接收一个游戏对象（在代码中使用了TicTacToe类的实例变量`game`)。

函数的主要目的是打印出游戏当前状态的线路图。具体实现中，从游戏的第一行开始，每一行代表了游戏板的一个小区域。在每一行的开始处，首先输出一个空行以分隔不同的区域。然后，对于每一行中的每个单元格，根据其状态（模糊或正常显示）输出相应的字符。

以一个典型的游戏为例，当游戏初始化完成后，该函数会输出这样的线路图：
``` 
    |     |     |     | 
    |   O   |   X   |   O   | 
    |     |     |     |  X   | 
    |   O   |   X   |   O   | 
    |________|_____|_____|_____| 
 
  (以'X'代表正常的' '符号，以'O'代表模糊的' '符号)
```
函数首先在每一行的开始处输出一个空行，然后在每一行的中间输出游戏板上的字符。通过这种方式，函数可以显示整个游戏板的状态，从而使得玩家可以了解当前游戏的状况。


```
def display(game: TicTacToe) -> None:
    line1 = ""
    for i in range(0, game.dim_sz):
        for j in range(0, game.dim_sz - 1):
            if game.board[i][j] == "blur":
                line1 = line1 + "    |"
            else:
                line1 = line1 + "  " + game.board[i][j] + " |"
        if game.board[i][game.dim_sz - 1] == "blur":
            line1 = line1 + "    \n"
        else:
            line1 = line1 + "  " + game.board[i][game.dim_sz - 1] + " \n"
    print(line1, "\n\n")


```

这段代码是一个Python程序，主要目的是让用户在控制台中玩井字棋游戏。

程序首先向用户询问要输入"X"或"O"，如果用户输入"O"，那么程序会创建一个只有"O"的玩家游戏；如果用户输入"X"，那么程序会创建一个只有"X"的玩家游戏。

游戏创建后，程序会向用户展示游戏棋盘，并允许用户输入坐标来移动棋子。程序会处理用户输入并更新棋盘状态，直到游戏结束。

具体来说，程序会处理以下事件：

* 用户输入"X"或"O"，创建井字棋游戏并更新棋盘状态。
* 用户输入坐标来移动棋子，更新棋盘状态，并检查游戏是否结束。
* 处理游戏结束的情况，输出相应的信息并结束游戏。


```
def main() -> None:
    pick = input("Pick 'X' or 'O' ").strip().upper()
    if pick == "O":
        game = TicTacToe("O")
    else:
        game = TicTacToe("X")
    display(game=game)
    while True:
        temp: Union[bool, str] = False
        while not temp:
            move = list(
                map(
                    int,
                    input("Make A Move in Grid System from (0,0) to (2,2) ").split(),
                )
            )
            temp = game.move_record(move[0], move[1])
            if not temp:
                print(temp)

        if game.check_win() == 1:
            print("You Won!")
            break
        print("Your Move:- ")
        display(game)
        C1, C2 = game.next_move()
        if C1 == -1 and C2 == -1:
            print("Game Tie!")
            break
        if game.check_win() == 0:
            print("You lost!")
            break
        print("Computer's Move :-")
        display(game)


```

这段代码是一个条件判断语句，它会判断当前脚本是否作为主程序运行。如果是，那么程序会执行if语句块内的内容，否则跳过if语句块。if语句块中包含了一个main函数，它是一个Python内置的函数，用于启动Python解释器。因此，这段代码的作用是启动Python解释器并执行main函数。


```
if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


README.md

Original source downloaded from Vintage Basic

Conversion to Rust


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Tower

This is a simulation of a game of logic that originated in the middle East. It is sometimes called Pharaoh's Needles, but its most common name is the Towers of Hanoi.

Legend has it that a secret society of monks live beneath the city of Hanoi. They possess three large towers or needles on which different size gold disks may be placed. Moving one at a time and never placing a large on a smaller disk, the monks endeavor to move the tower of disks from the left needle to the right needle. Legend says when they have finished moving this 64-disk tower, the world will end. How many moves will they have to make to accomplish this? If they can move 1 disk per minute and work 24 hours per day, how many years will it take?

In the computer puzzle you are faced with three upright needles. On the leftmost needle are placed from two to seven graduated disks, the largest being on bottom and smallest on top. Your object is to move the entire stack of disks to the rightmost needle. However, you many only move one disk at a time and you may never place a larger disk on top of a smaller one.

In this computer game, the disks are referred to by their size — i.e., the smallest is 3, next 5, 7, 9, 11, 13, and 15. If you play with fewer than 7 disks always use the largest, i.e. with 2 disks you would use nos. 13 and 15. The program instructions are self-explanatory. Good luck!

Charles Lund wrote this program while at the American School in the Hague, Netherlands.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=173)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=188)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `90_Tower/csharp/Game.cs`

This is a simple game of邮票游戏。玩家需要根据提示移动邮票，以获取更多的分数。游戏具有3D效果和动画，使它看起来更有趣。游戏还具有挑战模式，当玩家通过所有关卡时，他们就可以获得高分。


```
using System;
using Tower.Models;
using Tower.Resources;
using Tower.UI;

namespace Tower
{
    internal class Game
    {
        private readonly Towers _towers;
        private readonly TowerDisplay _display;
        private readonly int _optimalMoveCount;
        private int _moveCount;

        public Game(int diskCount)
        {
            _towers = new Towers(diskCount);
            _display = new TowerDisplay(_towers);
            _optimalMoveCount = (1 << diskCount) - 1;
        }

        public bool Play()
        {
            Console.Write(Strings.Instructions);

            Console.Write(_display);

            while (true)
            {
                if (!Input.TryReadNumber(Prompt.Disk, out int disk)) { return false; }

                if (!_towers.TryFindDisk(disk, out var from, out var message))
                {
                    Console.WriteLine(message);
                    continue;
                }

                if (!Input.TryReadNumber(Prompt.Needle, out var to)) { return false; }

                if (!_towers.TryMoveDisk(from, to))
                {
                    Console.Write(Strings.IllegalMove);
                    continue;
                }

                Console.Write(_display);

                var result = CheckProgress();
                if (result.HasValue) { return result.Value; }
            }
        }

        private bool? CheckProgress()
        {
            _moveCount++;

            if (_moveCount == 128)
            {
                Console.Write(Strings.TooManyMoves);
                return false;
            }

            if (_towers.Finished)
            {
                if (_moveCount == _optimalMoveCount)
                {
                    Console.Write(Strings.Congratulations);
                }

                Console.WriteLine(Strings.TaskFinished, _moveCount);

                return true;
            }

            return default;
        }
    }
}

```

# `90_Tower/csharp/Program.cs`

这段代码是一个使用Tower库的JavaScript脚本，用于在用户输入 diskCount（磁盘数）后，运行一个计算总共玩过多少游戏的游戏。

具体来说，以下是代码的作用：

1. 导入 System、Tower 和 Tower.UI 命名空间。
2. 在 Main 类中定义一个名为 Program 的静态方法。
3. 在 Main 方法的 do-while 循环中，首先输出一个标题字符串（使用字符串的 Title 方法）。
4. 在输出标题字符串后，使用 Input.TryReadNumber 方法尝试从用户输入中读取 diskCount 字符串，如果没有正确读取，则返回。
5. 创建一个名为 Game 的游戏对象，并在 Game.Play 方法中尝试运行游戏。
6. 如果游戏运行成功，则再次尝试从用户输入中读取一个字符串，该字符串将决定是否再次玩游戏。如果用户选择 "是"，则游戏将再次运行；如果用户选择 "否"，则退出游戏。
7. 最后，输出一个感谢字符串。


```
﻿using System;
using Tower.Resources;
using Tower.UI;

namespace Tower
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.Write(Strings.Title);

            do
            {
                Console.Write(Strings.Intro);

                if (!Input.TryReadNumber(Prompt.DiskCount, out var diskCount)) { return; }

                var game = new Game(diskCount);

                if (!game.Play()) { return; }
            } while (Input.ReadYesNo(Strings.PlayAgainPrompt, Strings.YesNoPrompt));

            Console.Write(Strings.Thanks);
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `90_Tower/csharp/Models/Needle.cs`

这段代码定义了一个名为 `Needle` 的类，该类实现了 `IEnumerable<int>` 接口，用于表示一组针。这个类包含一个栈，用于存储这组点的数据。

该类提供了一些方法来使用栈，包括：

- `IsEmpty`：判断栈是否为空。
- `Top`：返回栈顶的点（如果栈不为空）。
- `TryPut(int disk)`：尝试将给定的点（如果该点小于栈顶点）压入栈中。
- `TryGetTopDisk(out int disk)`：尝试从栈中获取出栈顶的点（如果可用），并将其返回。
- `GetEnumerator()`：返回一个迭代器，用于遍历这组点。
- `GetEnumerator(int diskCount)`：返回一个迭代器，用于遍历指定 diskCount 个连续的点。

这个类的实例可以被用于创建一个需要指的定期器，该定期器可以用来取得一个需要指的点。


```
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace Tower.Models
{
    internal class Needle : IEnumerable<int>
    {
        private readonly Stack<int> _disks = new Stack<int>();

        public bool IsEmpty => _disks.Count == 0;

        public int Top => _disks.TryPeek(out var disk) ? disk : default;

        public bool TryPut(int disk)
        {
            if (_disks.Count == 0 || disk < _disks.Peek())
            {
                _disks.Push(disk);
                return true;
            }

            return false;
        }

        public bool TryGetTopDisk(out int disk) => _disks.TryPop(out disk);

        public IEnumerator<int> GetEnumerator() =>
            Enumerable.Repeat(0, 7 - _disks.Count).Concat(_disks).GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }
}

```

# `90_Tower/csharp/Models/Towers.cs`

This is a class that represents a set of disks, where each disk has a top-序和重量。 It also has three methods: TryGetTopDisk, TryPut, and MoveTo.

The TryGetTopDisk method takes an integer disk number and returns a boolean indicating whether the disk is in the top-序.

The TryPut method takes two integers the disk number and the disk to move it to, and returns a boolean indicating whether the disk was successfully moved.

The MoveTo method takes two integers the disk number and the disk to move it to, and returns a boolean indicating whether the disk was successfully moved.

It also has a method called GetEnumerator which returns an enumerator for the TryGetTopDisk, TryPut, and MoveTo methods.

It also has a class called TowersEnumerator which is a wrapper class for the Enumerator.

The TowersEnumerator class has a constructor that takes a set of disks and a `Needle` class as


```
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Tower.Resources;

namespace Tower.Models
{
    internal class Towers : IEnumerable<(int, int, int)>
    {
        private static int[] _availableDisks = new[] { 15, 13, 11, 9, 7, 5, 3 };

        private readonly Needle[] _needles = new[] { new Needle(), new Needle(), new Needle() };
        private readonly int _smallestDisk;

        public Towers(int diskCount)
        {
            foreach (int disk in _availableDisks.Take(diskCount))
            {
                this[1].TryPut(disk);
                _smallestDisk = disk;
            }
        }

        private Needle this[int i] => _needles[i-1];

        public bool Finished => this[1].IsEmpty && this[2].IsEmpty;

        public bool TryFindDisk(int disk, out int needle, out string message)
        {
            needle = default;
            message = default;

            if (disk < _smallestDisk)
            {
                message = Strings.DiskNotInPlay;
                return false;
            }

            for (needle = 1; needle <= 3; needle++)
            {
                if (this[needle].Top == disk) { return true; }
            }

            message = Strings.DiskUnavailable;
            return false;
        }

        public bool TryMoveDisk(int from, int to)
        {
            if (!this[from].TryGetTopDisk(out var disk))
            {
                throw new InvalidOperationException($"Needle {from} is empty");
            }

            if (this[to].TryPut(disk)) { return true; }

            this[from].TryPut(disk);
            return false;
        }

        public IEnumerator<(int, int, int)> GetEnumerator() => new TowersEnumerator(_needles);

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        private class TowersEnumerator : IEnumerator<(int, int, int)>
        {
            private readonly List<IEnumerator<int>> _enumerators;

            public TowersEnumerator(Needle[] needles)
            {
                _enumerators = needles.Select(n => n.GetEnumerator()).ToList();
            }

            public (int, int, int) Current =>
                (_enumerators[0].Current, _enumerators[1].Current, _enumerators[2].Current);

            object IEnumerator.Current => Current;

            public void Dispose() => _enumerators.ForEach(e => e.Dispose());

            public bool MoveNext() => _enumerators.All(e => e.MoveNext());

            public void Reset() => _enumerators.ForEach(e => e.Reset());
        }
    }
}

```

# `90_Tower/csharp/Resources/Strings.cs`



这段代码是一个自定义的 C# 类 `Tower.Resources`，它包含了一系列的 `GetResource` 方法，这些方法可以被其他类或程序调用，用于从不同的资源文件中读取字符串。

具体来说，这些方法都是通过 `Assembly.GetExecutingAssembly()` 获取当前程序的执行 Assembies，然后获取其中的资源文件，最后读取这些文件的字符串内容并返回给调用者。

这个自定义的类 `Tower.Resources` 可以被其他程序或类通过调用 `Tower.Resources.Strings.GetResource()` 来访问这些方法，例如在 `Program` 类中，可以这样调用它：

```
Tower.Resources.Strings.GetResource("DiskCountPrompt");
``` 

调用这个方法会将当前程序的 `Tower.Resources.Strings.DiskCountPrompt` 类中的所有字符串资源加载到程序中，并返回字符串的值。


```
﻿using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace Tower.Resources
{
    internal static class Strings
    {
        internal static string Congratulations => GetResource();
        internal static string DiskCountPrompt => GetResource();
        internal static string DiskCountQuit => GetResource();
        internal static string DiskCountRetry => GetResource();
        internal static string DiskNotInPlay => GetResource();
        internal static string DiskPrompt => GetResource();
        internal static string DiskQuit => GetResource();
        internal static string DiskRetry => GetResource();
        internal static string DiskUnavailable => GetResource();
        internal static string IllegalMove => GetResource();
        internal static string Instructions => GetResource();
        internal static string Intro => GetResource();
        internal static string NeedlePrompt => GetResource();
        internal static string NeedleQuit => GetResource();
        internal static string NeedleRetry => GetResource();
        internal static string PlayAgainPrompt => GetResource();
        internal static string TaskFinished => GetResource();
        internal static string Thanks => GetResource();
        internal static string Title => GetResource();
        internal static string TooManyMoves => GetResource();
        internal static string YesNoPrompt => GetResource();

        private static string GetResource([CallerMemberName] string name = "")
        {
            var streamName = $"Tower.Resources.{name}.txt";
            using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(streamName);
            using var reader = new StreamReader(stream);

            return reader.ReadToEnd();
        }
    }
}

```

# `90_Tower/csharp/UI/Input.cs`

This looks like a class that wraps a `Console` object and provides a set of functions for reading input from the user.

The class has two main functions: `ReadNumber` and `ReadString`. `ReadNumber` takes a string prompt and returns the first input value as a number. `ReadString` takes a string prompt and returns the input value as a string.

Both functions take an optional `Prompt` object, which is called when the function is called. The `Prompt` object can be used to display a message to the user and to indicate whether the user has entered valid input or not.

The class also has a `TryParseNumber` function, which takes a string prompt and tries to parse the input as a number. If the input can be parsed as a number, this function returns `true`. Otherwise, the function returns `false` and the input value is set to `default`.

Overall, this class provides a simple way to read input from the user, but it does have a few limitations. For example, the input values are capped at the maximum number and string lengths specified by the `Console` object. Additionally, the input values are only read as numbers, so they cannot be used to read other types of input.


```
using System;
using System.Collections.Generic;

namespace Tower.UI
{
    // Provides input methods which emulate the BASIC interpreter's keyboard input routines
    internal static class Input
    {
        private static void Prompt(string text = "") => Console.Write($"{text}? ");

        internal static bool ReadYesNo(string prompt, string retryPrompt)
        {
            var response = ReadString(prompt);

            while (true)
            {
                if (response.Equals("No", StringComparison.InvariantCultureIgnoreCase)) { return false; }
                if (response.Equals("Yes", StringComparison.InvariantCultureIgnoreCase)) { return true; }
                response = ReadString(retryPrompt);
            }
        }

        internal static bool TryReadNumber(Prompt prompt, out int number)
        {
            var message = prompt.Message;

            for (int retryCount = 0; retryCount <= prompt.RetriesAllowed; retryCount++)
            {
                if (retryCount > 0) { Console.WriteLine(prompt.RetryMessage); }

                if (prompt.TryValidateResponse(ReadNumber(message), out number)) { return true; }

                if (!prompt.RepeatPrompt) { message = ""; }
            }

            Console.WriteLine(prompt.QuitMessage);

            number = 0;
            return false;
        }

        private static float ReadNumber(string prompt)
        {
            Prompt(prompt);

            while (true)
            {
                var inputValues = ReadStrings();

                if (TryParseNumber(inputValues[0], out var number))
                {
                    if (inputValues.Length > 1)
                    {
                        Console.WriteLine("!Extra input ingored");
                    }

                    return number;
                }
            }
        }

        private static string ReadString(string prompt)
        {
            Prompt(prompt);

            var inputValues = ReadStrings();
            if (inputValues.Length > 1)
            {
                Console.WriteLine("!Extra input ingored");
            }
            return inputValues[0];
        }

        private static string[] ReadStrings() => Console.ReadLine().Split(',', StringSplitOptions.TrimEntries);

        private static bool TryParseNumber(string text, out float number)
        {
            if (float.TryParse(text, out number)) { return true; }

            Console.WriteLine("!Number expected - retry input line");
            number = default;
            return false;
        }
    }
}

```

# `90_Tower/csharp/UI/Prompt.cs`



这段代码是一个 Prompt 类，用于显示一个用于获取硬盘字符串的提示。该类包含以下构造函数：

- DiskCount：使用默认构造函数创建的 Prompt 类，包含允许的最大重试次数(2)，用于在出现错误时重新尝试获取硬盘字符串。
- Disk：使用默认构造函数创建的 Prompt 类，包含用于获取硬盘字符串的实际尝试次数(3)，用于显示在获取字符串时需要再次尝试的次数。
- Needle：使用默认构造函数创建的 Prompt 类，包含用于获取基于需要字符串的尝试次数(1)，用于显示在获取字符串时需要尝试的次数。

该类还包含 TryValidateResponse 方法，用于验证用户输入的数字是否为有效的值，并返回它们的有效性结果。


```
using System.Collections.Generic;
using System.Linq;
using static Tower.Resources.Strings;

namespace Tower.UI
{
    internal class Prompt
    {
        public static Prompt DiskCount =
            new(DiskCountPrompt, DiskCountRetry, DiskCountQuit, 1, 2, 3, 4, 5, 6, 7) { RetriesAllowed = 2 };

        public static Prompt Disk =
            new(DiskPrompt, DiskRetry, DiskQuit, 3, 5, 7, 9, 11, 13, 15) { RepeatPrompt = false };

        public static Prompt Needle = new(NeedlePrompt, NeedleRetry, NeedleQuit, 1, 2, 3);

        private readonly HashSet<int> _validValues;

        private Prompt(string prompt, string retryMessage, string quitMessage, params int[] validValues)
        {
            Message = prompt;
            RetryMessage = retryMessage;
            QuitMessage = quitMessage;
            _validValues = validValues.ToHashSet();
            RetriesAllowed = 1;
            RepeatPrompt = true;
        }

        public string Message { get; }
        public string RetryMessage { get; }
        public string QuitMessage { get; }
        public int RetriesAllowed { get; private set; }
        public bool RepeatPrompt { get; private set; }

        public bool TryValidateResponse(float number, out int integer)
        {
            integer = (int)number;
            return integer == number && _validValues.Contains(integer);
        }
    }
}

```

# `90_Tower/csharp/UI/TowerDisplay.cs`



该代码是一个 Tower 应用程序的内部类，实现了 `ToString()` 方法，用于将所有的高楼打印出来并按照塔的大小进行对齐。具体来说，代码中创建了一个 `TowerDisplay` 类，其中包含一个 `Towers` 类的实例变量。在 `TowerDisplay` 类中，创建了一个字符串Builder，用于存储所有的 tower 对象的字符串表示形式。然后，代码遍历所有的 `Tower` 对象，并打印出每个高楼的字符串表示形式。在打印字符串表示形式时，还实现了 `AppendTower()` 方法，用于在字符串中添加每个高楼的参数。


```
using System;
using System.Text;
using Tower.Models;

namespace Tower.UI
{
    internal class TowerDisplay
    {
        private readonly Towers _towers;

        public TowerDisplay(Towers towers)
        {
            _towers = towers;
        }

        public override string ToString()
        {
            var builder = new StringBuilder();

            foreach (var row in _towers)
            {
                AppendTower(row.Item1);
                AppendTower(row.Item2);
                AppendTower(row.Item3);
                builder.AppendLine();
            }

            return builder.ToString();

            void AppendTower(int size)
            {
                var padding = 10 - size / 2;
                builder.Append(' ', padding).Append('*', Math.Max(1, size)).Append(' ', padding);
            }
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `90_Tower/java/Tower.java`

这段代码是一个Java程序，它导入了Java标准库中的Math类和Scanner类。然后，它创建了一个名为"Tower"的类，继承自另一个名为"Tower"的类。

这个程序的主要目的是提供一个基于BASIC游戏Tower的重新实现的版本，不包括额外的文本，错误检查等新功能。

具体来说，这段代码实现了以下功能：

1. 定义了一个名为"height"的变量，类型为double，值为10.0（表示每个阶段的高度增加量）。
2. 定义了一个名为"width"的变量，类型为int，值为10.0（表示每个阶段的宽度增加量）。
3. 创建了一个名为"tower"的实例，类型为"Tower"类，并将其赋值为（10， 10，" "，" 20.0，（ 20.0， " "， true，（（（（（（（（（（（（（（（）（（（（（（））（））（））（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（）（），"height":"width：10.0："：int："10.0（表示每个阶段的宽度增加量）"："10.0（表示每个阶段的高度增加量）"：" "："20.0（表示每个阶段的宽度增加量）"："（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（


```
import java.lang.Math;
import java.util.Scanner;

/**
 * Game of Tower
 * <p>
 * Based on the BASIC game of Tower here
 * https://github.com/coding-horror/basic-computer-games/blob/main/90%20Tower/tower.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 *
 * Converted from BASIC to Java by Darren Cardenas.
 */

```



This is a Java class that represents a Tower game board. The game board has 20 columns and 9 spaces between them for each row. It also has 20 spaces on each row for each column.

The class methods are:

* `isNeedleSafe(int needle, int disk, int row)`: Checks if the needle is safe to put on the disk at the current row by checking if there is a space between the needle and the disk. It returns `true` if there is a space, and `false` otherwise.
* `printPositions()`: Prints the positions of all the pieces on the game board.
* `main(String[] args)`: The main method initializes the game board and starts a game.

The `printPositions()` method prints the positions of all the pieces on the game board by printing between spaces, as shown in the code snippet:
```
// Begin loop through all rows
for (row = 1; row <= MAX_NUM_ROWS; row++) {
 numSpaces = 9;

 // Begin loop through all columns
 for (column = 1; column <= MAX_NUM_COLUMNS; column++) {
   // No disk at the current position
   if (positions[row][column] == 0) {
     // Draw a disk at the current position
     System.out.print(" ".repeat(numSpaces) + "*");
     numSpaces = 20;
   }
 }
}
```


```
public class Tower {

  private final static int MAX_DISK_SIZE = 15;

  private final static int MAX_NUM_COLUMNS = 3;

  private final static int MAX_NUM_MOVES = 128;

  private final static int MAX_NUM_ROWS = 7;

  private final Scanner scan;  // For user input

  // Represent all possible disk positions
  private int[][] positions;

  private enum Step {
    INITIALIZE, SELECT_TOTAL_DISKS, SELECT_DISK_MOVE, SELECT_NEEDLE, CHECK_SOLUTION
  }


  public Tower() {

    scan = new Scanner(System.in);

    // Row 0 and column 0 are not used
    positions = new int[MAX_NUM_ROWS + 1][MAX_NUM_COLUMNS + 1];

  }  // End of constructor Tower


  public class Position {

    public int row;
    public int column;

    public Position(int row, int column) {
      this.row = row;
      this.column = column;

    }  // End of constructor Position

  }  // End of inner class Position


  public void play() {

    showIntro();
    startGame();

  }  // End of method play


  private void showIntro() {

    System.out.println(" ".repeat(32) + "TOWERS");
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    System.out.println("\n\n");

  }  // End of method showIntro


  private void startGame() {

    boolean diskMoved = false;

    int column = 0;
    int disk = 0;
    int needle = 0;
    int numDisks = 0;
    int numErrors = 0;
    int numMoves = 0;
    int row = 0;

    Step nextStep = Step.INITIALIZE;

    String userResponse = "";

    Position diskPosition = new Position(0, 0);

    // Begin outer while loop
    while (true) {

      switch (nextStep) {


        case INITIALIZE:

          // Initialize error count
          numErrors = 0;

          // Initialize positions
          for (row = 1; row <= MAX_NUM_ROWS; row++) {
            for (column = 1; column <= MAX_NUM_COLUMNS; column++) {
              positions[row][column] = 0;
            }
          }

          // Display description
          System.out.println("");
          System.out.println("TOWERS OF HANOI PUZZLE.\n");
          System.out.println("YOU MUST TRANSFER THE DISKS FROM THE LEFT TO THE RIGHT");
          System.out.println("TOWER, ONE AT A TIME, NEVER PUTTING A LARGER DISK ON A");
          System.out.println("SMALLER DISK.\n");

          nextStep = Step.SELECT_TOTAL_DISKS;
          break;


        case SELECT_TOTAL_DISKS:

          while (numErrors <= 2) {

            // Get user input
            System.out.print("HOW MANY DISKS DO YOU WANT TO MOVE (" + MAX_NUM_ROWS + " IS MAX)? ");
            numDisks = scan.nextInt();
            System.out.println("");

            numMoves = 0;

            // Ensure the number of disks is valid
            if ((numDisks < 1) || (numDisks > MAX_NUM_ROWS)) {

              numErrors++;

              // Handle user input errors
              if (numErrors < 3) {
                System.out.println("SORRY, BUT I CAN'T DO THAT JOB FOR YOU.");
              }

            }
            else {
              break;  // Leave the while loop
            }
          }

          // Too many user input errors
          if (numErrors > 2) {
            System.out.println("ALL RIGHT, WISE GUY, IF YOU CAN'T PLAY THE GAME RIGHT, I'LL");
            System.out.println("JUST TAKE MY PUZZLE AND GO HOME.  SO LONG.");
            return;
          }

          // Display detailed instructions
          System.out.println("IN THIS PROGRAM, WE SHALL REFER TO DISKS BY NUMERICAL CODE.");
          System.out.println("3 WILL REPRESENT THE SMALLEST DISK, 5 THE NEXT SIZE,");
          System.out.println("7 THE NEXT, AND SO ON, UP TO 15.  IF YOU DO THE PUZZLE WITH");
          System.out.println("2 DISKS, THEIR CODE NAMES WOULD BE 13 AND 15.  WITH 3 DISKS");
          System.out.println("THE CODE NAMES WOULD BE 11, 13 AND 15, ETC.  THE NEEDLES");
          System.out.println("ARE NUMBERED FROM LEFT TO RIGHT, 1 TO 3.  WE WILL");
          System.out.println("START WITH THE DISKS ON NEEDLE 1, AND ATTEMPT TO MOVE THEM");
          System.out.println("TO NEEDLE 3.\n");
          System.out.println("GOOD LUCK!\n");

          disk = MAX_DISK_SIZE;

          // Set disk starting positions
          for (row = MAX_NUM_ROWS; row > (MAX_NUM_ROWS - numDisks); row--) {
            positions[row][1] = disk;
            disk = disk - 2;
          }

          printPositions();

          nextStep = Step.SELECT_DISK_MOVE;
          break;


        case SELECT_DISK_MOVE:

          System.out.print("WHICH DISK WOULD YOU LIKE TO MOVE? ");

          numErrors = 0;

          while (numErrors < 2) {
            disk = scan.nextInt();

            // Validate disk numbers
            if ((disk - 3) * (disk - 5) * (disk - 7) * (disk - 9) * (disk - 11) * (disk - 13) * (disk - 15) == 0) {

              // Check if disk exists
              diskPosition = getDiskPosition(disk);

              // Disk found
              if ((diskPosition.row > 0) && (diskPosition.column > 0))
              {
                // Disk can be moved
                if (isDiskMovable(disk, diskPosition.row, diskPosition.column) == true) {

                  break;

                }
                // Disk cannot be moved
                else {

                  System.out.println("THAT DISK IS BELOW ANOTHER ONE.  MAKE ANOTHER CHOICE.");
                  System.out.print("WHICH DISK WOULD YOU LIKE TO MOVE? ");

                }
              }
              // Mimic legacy handling of valid disk number but disk not found
              else {

                System.out.println("THAT DISK IS BELOW ANOTHER ONE.  MAKE ANOTHER CHOICE.");
                System.out.print("WHICH DISK WOULD YOU LIKE TO MOVE? ");
                numErrors = 0;
                continue;

              }

            }
            // Invalid disk number
            else {

              System.out.println("ILLEGAL ENTRY... YOU MAY ONLY TYPE 3,5,7,9,11,13, OR 15.");
              numErrors++;

              if (numErrors > 1) {
                break;
              }

              System.out.print("? ");

            }
          }

          if (numErrors > 1) {

            System.out.println("STOP WASTING MY TIME.  GO BOTHER SOMEONE ELSE.");
            return;
          }

          nextStep = Step.SELECT_NEEDLE;
          break;


        case SELECT_NEEDLE:

          numErrors = 0;

          while (true) {

            System.out.print("PLACE DISK ON WHICH NEEDLE? ");
            needle = scan.nextInt();

            // Handle valid needle numbers
            if ((needle - 1) * (needle - 2) * (needle - 3) == 0) {

              // Ensure needle is safe for disk move
              if (isNeedleSafe(needle, disk, row) == false) {

                System.out.println("YOU CAN'T PLACE A LARGER DISK ON TOP OF A SMALLER ONE,");
                System.out.println("IT MIGHT CRUSH IT!");
                System.out.print("NOW THEN, ");

                nextStep = Step.SELECT_DISK_MOVE;
                break;
              }

              diskPosition = getDiskPosition(disk);

              // Attempt to move the disk on a non-empty needle
              diskMoved = false;
              for (row = 1; row <= MAX_NUM_ROWS; row++) {
                if (positions[row][needle] != 0) {
                  row--;

                  positions[row][needle] = positions[diskPosition.row][diskPosition.column];
                  positions[diskPosition.row][diskPosition.column] = 0;

                  diskMoved = true;
                  break;
                }
              }

              // Needle was empty, so move disk to the bottom
              if (diskMoved == false) {
                positions[MAX_NUM_ROWS][needle] = positions[diskPosition.row][diskPosition.column];
                positions[diskPosition.row][diskPosition.column] = 0;
              }

              nextStep = Step.CHECK_SOLUTION;
              break;

            }
            // Handle invalid needle numbers
            else {

              numErrors++;

              if (numErrors > 1) {
                System.out.println("I TRIED TO WARN YOU, BUT YOU WOULDN'T LISTEN.");
                System.out.println("BYE BYE, BIG SHOT.");
                return;
              }
              else {
                System.out.println("I'LL ASSUME YOU HIT THE WRONG KEY THIS TIME.  BUT WATCH IT,");
                System.out.println("I ONLY ALLOW ONE MISTAKE.");
              }
            }

          }

          break;


        case CHECK_SOLUTION:

          printPositions();

          numMoves++;

          // Puzzle is solved
          if (isPuzzleSolved() == true) {

            // Check for optimal solution
            if (numMoves == (Math.pow(2, numDisks) - 1)) {
              System.out.println("");
              System.out.println("CONGRATULATIONS!!\n");
            }

            System.out.println("YOU HAVE PERFORMED THE TASK IN " + numMoves + " MOVES.\n");
            System.out.print("TRY AGAIN (YES OR NO)? ");

            // Prompt for retries
            while (true) {
              userResponse = scan.next();

              if (userResponse.toUpperCase().equals("YES")) {
                nextStep = Step.INITIALIZE;
                break;
              }
              else if (userResponse.toUpperCase().equals("NO")) {
                System.out.println("");
                System.out.println("THANKS FOR THE GAME!\n");
                return;
              }
              else {
                System.out.print("'YES' OR 'NO' PLEASE? ");
              }
            }
          }
          // Puzzle is not solved
          else {

            // Exceeded maximum number of moves
            if (numMoves > MAX_NUM_MOVES) {
              System.out.println("SORRY, BUT I HAVE ORDERS TO STOP IF YOU MAKE MORE THAN");
              System.out.println("128 MOVES.");
              return;
            }

            nextStep = Step.SELECT_DISK_MOVE;
            break;
          }

          break;

        default:
          System.out.println("INVALID STEP");
          break;

      }

    }  // End outer while loop

  }  // End of method startGame


  private boolean isPuzzleSolved() {

    int column = 0;
    int row = 0;

    // Puzzle is solved if first 2 needles are empty
    for (row = 1; row <= MAX_NUM_ROWS; row++) {
      for (column = 1; column <= 2; column++) {
        if (positions[row][column] != 0) {
          return false;
        }
      }
    }

    return true;

  }  // End of method isPuzzleSolved


  private Position getDiskPosition(int disk) {

    int column = 0;
    int row = 0;

    Position pos = new Position(0, 0);

    // Begin loop through all rows
    for (row = 1; row <= MAX_NUM_ROWS; row++) {

      // Begin loop through all columns
      for (column = 1; column <= MAX_NUM_COLUMNS; column++) {

        // Found the disk
        if (positions[row][column] == disk) {

          pos.row = row;
          pos.column = column;
          return pos;

        }

      }  // End loop through all columns

    }  // End loop through all rows

    return pos;

  }  // End of method getDiskPosition


  private boolean isDiskMovable(int disk, int row, int column) {

    int ii = 0;  // Loop iterator

    // Begin loop through all rows above disk
    for (ii = row; ii >= 1; ii--) {

      // Disk can be moved
      if (positions[ii][column] == 0) {
        continue;
      }

      // Disk cannot be moved
      if (positions[ii][column] < disk) {
        return false;
      }

    }  // End loop through all rows above disk

    return true;

  }  // End of method isDiskMovable


  private boolean isNeedleSafe(int needle, int disk, int row) {

    for (row = 1; row <= MAX_NUM_ROWS; row++) {

      // Needle is not empty
      if (positions[row][needle] != 0) {

        // Disk crush condition
        if (disk >= positions[row][needle]) {
          return false;
        }
      }
    }

    return true;

  }  // End of method isNeedleSafe


  private void printPositions() {

    int column = 1;
    int ii = 0;  // Loop iterator
    int numSpaces = 0;
    int row = 1;

    // Begin loop through all rows
    for (row = 1; row <= MAX_NUM_ROWS; row++) {

      numSpaces = 9;

      // Begin loop through all columns
      for (column = 1; column <= MAX_NUM_COLUMNS; column++) {

        // No disk at the current position
        if (positions[row][column] == 0) {

          System.out.print(" ".repeat(numSpaces) + "*");
          numSpaces = 20;
        }

        // Draw a disk at the current position
        else {

          System.out.print(" ".repeat(numSpaces - ((int) (positions[row][column] / 2))));

          for (ii = 1; ii <= positions[row][column]; ii++) {
            System.out.print("*");
          }

          numSpaces = 20 - ((int) (positions[row][column] / 2));
        }

      }  // End loop through all columns

      System.out.println("");

    }  // End loop through all rows

  }  // End of method printPositions


  public static void main(String[] args) {

    Tower tower = new Tower();
    tower.play();

  }  // End of method main

}  // End of class Tower

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


# `90_Tower/javascript/tower.js`

这段代码定义了两个函数，分别是`print()`和`input()`。它们的主要作用如下：

1. `print()`函数的作用是在文档的`<textarea id="output"></textarea>`元素中插入字符串。该函数的核心是通过`document.createTextNode()`方法创建一个`<span>`标签，并将其添加到`<textarea>`元素中。`print()`函数的作用是将字符串插入到指定的文本区域中，这样当页面加载时，用户就可以看到相应的输出结果。

2. `input()`函数的作用是从用户那里接收输入值，并在控制台输出输入结果。该函数的核心是通过`document.createElement("INPUT")`方法创建一个`<input>`标签，并设置其属性。然后，函数使用`print()`函数将用户输入的值输出到控制台。由于`input()`函数并没有返回任何值，因此它可以随时接受用户输入的值，并在需要时打印出来。


```
// TOWER
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



这两段代码定义了一个名为`tab`的函数和一个名为`show_towers`的函数。

`tab`函数的作用是在一个字符串变量`str`中添加指定的空间数量，并返回该字符串。它的实现是通过一个while循环和一个变量`space`来实现的。当`space`的值减为0时，循环停止，此时`str`字符串中的所有内容将添加到变量`str`中。

`show_towers`函数的作用是打印一个包含`ta`数组中所有元素高维展开形式的字符串。它的实现是通过一个for循环和一个变量`z`来实现的。当`k`的值从1递增到7时，`show_towers`函数将遍历`ta`数组中的每个元素，并计算出每个元素的值，然后将元素的值打印出来。为了实现这个目标，`show_towers`函数使用了一个变量`str`和一个变量`z`，其中`str`用于存储打印的字符串，`z`用于存储打印的字符数。

在`show_towers`函数中，`for`循环用于打印`ta`数组中的每个元素，而`if`语句则用于检查`ta`数组中每个元素的值是否为非零整数。如果是，则`while`循环将打印`z`个字符，并将`str`和`*`字符打印出来。否则，`while`循环将打印`z`个字符，并打印`str`和`*`字符。通过这种方式，`show_towers`函数可以打印出`ta`数组中所有元素的高维展开形式。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var ta = [];

// Print subroutine
function show_towers()
{
    var z;

    for (var k = 1; k <= 7; k++) {
        z = 10;
        str = "";
        for (var j = 1; j <= 3; j++) {
            if (ta[k][j] != 0) {
                while (str.length < z - Math.floor(ta[k][j] / 2))
                    str += " ";
                for (v = 1; v <= ta[k][j]; v++)
                    str += "*";
            } else {
                while (str.length < z)
                    str += " ";
                str += "*";
            }
            z += 21;
        }
        print(str + "\n");
    }
}

```

This is a program that solves the Tower of Babel problem. The program takes the following input:

1. The number of moves (n) made by the simulation.
2. The number of disks (s) on each needle.
3. The maximum number of moves that can be performed with the given disks.
4. The current status of the simulation (e.g. the names of the disks on each needle, the moves made so far, etc.).

The program uses a while loop to perform the simulation. On each iteration of the loop, the program performs the following steps:

1. Increments the disk that has been visited (i) by 1.
2. If the disk has more disks on it than the maximum number of moves available (a), the program prints a message and returns.
3. If the disk is the same as a disk that has already been visited (b), the program prints a message and returns.
4. Print out the current status of the disk.
5. Print out the moves made by the program so far.
6. Print a message asking the user if they want to continue the simulation.
7. If the user enters "YES", the program loops back to step 3, otherwise it loops back to step 2.

The program uses a variable (m) to keep track of the number of moves made by the simulation.


```
// Main control section
async function main()
{
    print(tab(33) + "TOWERS\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    while (1) {
        print("\n");
        // Initialize
        e = 0;
        for (d = 1; d <= 7; d++) {
            ta[d] = [];
            for (n = 1; n <= 3; n++)
                ta[d][n] = 0;
        }
        print("TOWERS OF HANOI PUZZLE.\n");
        print("\n");
        print("YOU MUST TRANSFER THE DISKS FROM THE LEFT TO THE RIGHT\n");
        print("TOWER, ONE AT A TIME, NEVER PUTTING A LARGER DISK ON A\n");
        print("SMALLER DISK.\n");
        print("\n");
        while (1) {
            print("HOW MANY DISKS DO YOU WANT TO MOVE (7 IS MAX)");
            s = parseInt(await input());
            print("\n");
            m = 0;
            if (s >= 1 && s <= 7)
                break;
            e++;
            if (e < 2) {
                print("SORRY, BUT I CAN'T DO THAT JOB FOR YOU.\n");
                continue;
            }
            print("ALL RIGHT, WISE GUY, IF YOU CAN'T PLAY THE GAME RIGHT, I'LL\n");
            print("JUST TAKE MY PUZZLE AND GO HOME.  SO LONG.\n");
            return;
        }
        // Store disks from smallest to largest
        print("IN THIS PROGRAM, WE SHALL REFER TO DISKS BY NUMERICAL CODE.\n");
        print("3 WILL REPRESENT THE SMALLEST DISK, 5 THE NEXT SIZE,\n");
        print("7 THE NEXT, AND SO ON, UP TO 15.  IF YOU DO THE PUZZLE WITH\n");
        print("2 DISKS, THEIR CODE NAMES WOULD BE 13 AND 15.  WITH 3 DISKS\n");
        print("THE CODE NAMES WOULD BE 11, 13 AND 15, ETC.  THE NEEDLES\n");
        print("ARE NUMBERED FROM LEFT TO RIGHT, 1 TO 3.  WE WILL\n");
        print("START WITH THE DISKS ON NEEDLE 1, AND ATTEMPT TO MOVE THEM\n");
        print("TO NEEDLE 3.\n");
        print("\n");
        print("GOOD LUCK!\n");
        print("\n");
        y = 7;
        d = 15;
        for (x = s; x >= 1; x--) {
            ta[y][1] = d;
            d -= 2;
            y--;
        }
        show_towers();
        while (1) {
            print("WHICH DISK WOULD YOU LIKE TO MOVE");
            e = 0;
            while (1) {
                d = parseInt(await input());
                if (d % 2 == 0 || d < 3 || d > 15) {
                    print("ILLEGAL ENTRY... YOU MAY ONLY TYPE 3,5,7,9,11,13, OR 15.\n");
                    e++;
                    if (e <= 1)
                        continue;
                    print("STOP WASTING MY TIME.  GO BOTHER SOMEONE ELSE.\n");
                    return;
                } else {
                    break;
                }
            }
            // Check if requested disk is below another
            for (r = 1; r <= 7; r++) {
                for (c = 1; c <= 3; c++) {
                    if (ta[r][c] == d)
                        break;
                }
                if (c <= 3)
                    break;
            }
            for (q = r; q >= 1; q--) {
                if (ta[q][c] != 0 && ta[q][c] < d)
                    break;
            }
            if (q >= 1) {
                print("THAT DISK IS BELOW ANOTHER ONE.  MAKE ANOTHER CHOICE.\n");
                continue;
            }
            e = 0;
            while (1) {
                print("PLACE DISK ON WHICH NEEDLE");
                n = parseInt(await input());
                if (n >= 1 && n <= 3)
                    break;
                e++;
                if (e <= 1) {
                    print("I'LL ASSUME YOU HIT THE WRONG KEY THI TIME.  BUT WATCH IT,\n");
                    print("I ONLY ALLOW ONE MISTAKE.\n");
                    continue;
                } else {
                    print("I TRIED TO WARN YOU, BUT YOU WOULDN'T LISTEN.\n");
                    print("BYE BYE, BIG SHOT.\n");
                    return;
                }
            }
            // Check if requested disk is below another
            for (r = 1; r <= 7; r++) {
                if (ta[r][n] != 0)
                    break;
            }
            if (r <= 7) {
                // Check if disk to be placed on a larger one
                if (d >= ta[r][n]) {
                    print("YOU CAN'T PLACE A LARGER DISK ON TOP OF A SMALLER ONE,\n");
                    print("IT MIGHT CRUSH IT!\n");
                    print("NOW THEN, ");
                    continue;
                }
            }
            // Move relocated disk
            for (v = 1; v <= 7; v++) {
                for (w = 1; w <= 3; w++) {
                    if (ta[v][w] == d)
                        break;
                }
                if (w <= 3)
                    break;
            }
            // Locate empty space on needle n
            for (u = 1; u <= 7; u++) {
                if (ta[u][n] != 0)
                    break;
            }
            ta[--u][n] = ta[v][w];
            ta[v][w] = 0;
            // Print out current status
            show_towers();
            // Check if done
            m++;
            for (r = 1; r <= 7; r++) {
                for (c = 1; c <= 2; c++) {
                    if (ta[r][c] != 0)
                        break;
                }
                if (c <= 2)
                    break;
            }
            if (r > 7)
                break;
            if (m > 128) {
                print("SORRY, BUT I HAVE ORDERS TO STOP IF YOU MAKE MORE THAN\n");
                print("128 MOVES.\n");
                return;
            }
        }
        if (m == Math.pow(2, s) - 1) {
            print("\n");
            print("CONGRATULATIONS!!\n");
            print("\n");
        }
        print("YOU HAVE PERFORMED THE TASK IN " + m + " MOVES.\n");
        print("\n");
        print("TRY AGAIN (YES OR NO)");
        while (1) {
            str = await input();
            if (str == "YES" || str == "NO")
                break;
            print("\n");
            print("'YES' OR 'NO' PLEASE");
        }
        if (str == "NO")
            break;
    }
    print("\n");
    print("THANKS FOR THE GAME!\n");
    print("\n");
}

```

这道题目是一个简单的C语言程序，包含了两个主要部分：`main()`函数和三个字符串变量。让我们逐步分析这两部分的功能。

1. `main()`函数：

`main()`函数是程序的入口点，它是程序运行的第一步。在这个函数中，程序创建了一个字符串变量`str1`，并将其赋值为`"Hello World"`，然后程序创建了一个字符串变量`str2`，并将其赋值为`"World"`。接下来，程序分别输出`str1`和`str2`中的第一个字符，即`'H'`和`'W'`。

2. `str1`：

`str1`是一个字符串变量，它存储了一个字符串。在这个变量中，`'H'`和`'W'`分别代表了字符串的第一个字符。

3. `str2`：

`str2`也是一个字符串变量，它存储了一个字符串。在这个变量中，`'W'`代表了字符串的第二个字符。

总结：

这段代码的主要目的是创建两个字符串变量`str1`和`str2`，并分别将`"Hello World"`和`"World"`赋值给它们。然后，程序分别输出`str1`和`str2`中的第一个字符，即`'H'`和`'W'`。


```
main();

```