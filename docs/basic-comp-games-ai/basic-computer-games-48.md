# BasicComputerGames源码解析 48

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


# `45_Hello/python/hello.py`

这段代码是一个简单的 "chat" 机器人，用于询问用户是否愿意与提示。它由两个条件判断和一个函数组成。

函数 `get_yes_or_no` 接受一个字符串参数，并返回一个元组，其中第一个元素是布尔值，表示用户是否输入了 "是"，第二个元素是布尔值，表示机器人给出的建议是否正确，第三个元素是提示用户输入的内容。

`get_yes_or_no` 函数会首先提示用户输入一个字符串，然后将其转换为小写并获取字母大小写。接下来，它通过 `upper()` 方法将字符串转换为大写，并检查用户是否输入 "是"。如果是，函数返回 `True`，`get_yes_or_no` 的第一个元素设置为 `True`，表示用户输入了正确的字符串；如果不是，函数返回 `False`，`get_yes_or_no` 的第二个元素设置为 `True`，表示机器人给出的建议正确；同时，函数的第三个元素保留输入的内容，作为提示给用户。

最后，代码中还有一行 `Warning, the advice given here is bad.`，这是一种警告，告知开发人员代码中存在问题，因为 `get_yes_or_no` 函数没有处理输入为 "是" 的用户。


```
"""
HELLO

A very simple "chat" bot.

Warning, the advice given here is bad.

Ported by Dave LeCompte
"""

import time
from typing import Optional, Tuple


def get_yes_or_no() -> Tuple[bool, Optional[bool], str]:
    msg = input()
    if msg.upper() == "YES":
        return True, True, msg
    elif msg.upper() == "NO":
        return True, False, msg
    else:
        return False, None, msg


```

这段代码是一个 Python 函数，它通过向用户询问一个 "你喜欢这里吗？" 的问题来寻求用户的反馈。如果用户回答 "是"，那么函数会输出一条欢迎消息并结束询问，否则它会再次询问用户是否喜欢这里。

函数的核心部分是询问用户是否喜欢这里，然后根据用户回答的结果做出相应的调整。如果用户回答 "是"，那么函数会结束询问并输出一条欢迎消息；如果用户回答 "否"，那么函数会再次询问用户是否喜欢这里，并提示他们可以 "开心点" 这里。


```
def ask_enjoy_question(user_name: str) -> None:
    print(f"HI THERE, {user_name}, ARE YOU ENJOYING YOURSELF HERE?")

    while True:
        valid, value, msg = get_yes_or_no()

        if valid:
            if value:
                print(f"I'M GLAD TO HEAR THAT, {user_name}.")
                print()
            else:
                print(f"OH, I'M SORRY TO HEAR THAT, {user_name}. MAYBE WE CAN")
                print("BRIGHTEN UP YOUR VISIT A BIT.")
            break
        else:
            print(f"{user_name}, I DON'T UNDERSTAND YOUR ANSWER OF '{msg}'.")
            print("PLEASE ANSWER 'YES' OR 'NO'.  DO YOU LIKE IT HERE?")


```



这两函数是Python语言中的函数，用于提示用户输入问题类型。

第一个函数 `prompt_for_problems(user_name: str) -> str` 提示用户输入一个字符串，然后返回一个字符串，表示用户输入的问题类型。函数的作用是通过 `print` 函数输出一个欢迎消息，然后提示用户输入问题类型，最后根据用户输入的问题类型返回一个字符串。

第二个函数 `prompt_too_much_or_too_little() -> Tuple[bool, Optional[bool]]` 提示用户输入一个字符串，然后返回一个元组，表示用户输入的问题是否太多或太少，并返回一个布尔值表示问题的严重程度。函数的作用是通过 `input` 函数获取用户输入，然后使用 `upper` 函数将输入的字符串转换为大写形式，最后根据用户输入的严重程度返回一个布尔值，表示问题的严重程度是太多或太少。


```
def prompt_for_problems(user_name: str) -> str:
    print()
    print(f"SAY, {user_name}, I CAN SOLVE ALL KINDS OF PROBLEMS EXCEPT")
    print("THOSE DEALING WITH GREECE.  WHAT KIND OF PROBLEMS DO")
    print("YOU HAVE? (ANSWER SEX, HEALTH, MONEY, OR JOB)")

    problem_type = input().upper()
    return problem_type


def prompt_too_much_or_too_little() -> Tuple[bool, Optional[bool]]:
    answer = input().upper()
    if answer == "TOO MUCH":
        return True, True
    elif answer == "TOO LITTLE":
        return True, False
    return False, None


```

这段代码是一个函数，它试图解决一个经典的谜题，即“Is your problem too much or too little?”，该谜题通常以一种幽默和有趣的方式提出。

函数中包含一个无限循环，该循环将一直重复直到用户输入有效的答案。在每次循环中，函数会首先调用一个名为“prompt_too_much_or_too_little()”的函数，该函数会提示用户输入问题是否太简单或太复杂，如果用户输入的答案是正确的，函数将返回一个True值，否则将返回一个False值。

如果函数的True值被返回，那么函数将首先打印一些幽默和有趣的话，然后打印用户的姓名并建议他们尝试一些实际的问题。如果函数的False值被返回，函数将打印一些更加严肃和有意义的话，并再次强调用户应该认真对待问题。

总的来说，这段代码就是试图让用户回答一个经典的谜题，以此来解决问题并取悦用户。


```
def solve_sex_problem(user_name: str) -> None:
    print("IS YOUR PROBLEM TOO MUCH OR TOO LITTLE?")
    while True:
        valid, too_much = prompt_too_much_or_too_little()
        if valid:
            if too_much:
                print("YOU CALL THAT A PROBLEM?!!  I SHOULD HAVE SUCH PROBLEMS!")
                print(f"IF IT BOTHERS YOU, {user_name}, TAKE A COLD SHOWER.")
            else:
                print(f"WHY ARE YOU HERE IN SUFFERN, {user_name}?  YOU SHOULD BE")
                print("IN TOKYO OR NEW YORK OR AMSTERDAM OR SOMEPLACE WITH SOME")
                print("REAL ACTION.")
            return
        else:
            print(f"DON'T GET ALL SHOOK, {user_name}, JUST ANSWER THE QUESTION")
            print("WITH 'TOO MUCH' OR 'TOO LITTLE'.  WHICH IS IT?")


```

这段代码定义了三个函数，分别是解决金钱问题、健康问题和职业问题。每个函数以不同的方式向用户传达建议。

1. solve_money_problem函数：当用户请求金钱问题时，这个函数会打印一段幽默的话，并提供一些解决金钱问题可能的建议。

2. solve_health_problem函数：当用户请求健康问题时，这个函数会打印一段建议，告诉用户应该注意什么来保持健康。

3. solve_job_problem函数：当用户请求职业问题时，这个函数会打印一段与职业相关建议，告诉用户他们需要付出更多努力来获得更好的工作体验。


```
def solve_money_problem(user_name: str) -> None:
    print(f"SORRY, {user_name}, I'M BROKE TOO.  WHY DON'T YOU SELL")
    print("ENCYCLOPEADIAS OR MARRY SOMEONE RICH OR STOP EATING")
    print("SO YOU WON'T NEED SO MUCH MONEY?")


def solve_health_problem(user_name: str) -> None:
    print(f"MY ADVICE TO YOU {user_name} IS:")
    print("     1.  TAKE TWO ASPRIN")
    print("     2.  DRINK PLENTY OF FLUIDS (ORANGE JUICE, NOT BEER!)")
    print("     3.  GO TO BED (ALONE)")


def solve_job_problem(user_name: str) -> None:
    print(f"I CAN SYMPATHIZE WITH YOU {user_name}.  I HAVE TO WORK")
    print("VERY LONG HOURS FOR NO PAY -- AND SOME OF MY BOSSES")
    print(f"REALLY BEAT ON MY KEYBOARD.  MY ADVICE TO YOU, {user_name},")
    print("IS TO OPEN A RETAIL COMPUTER STORE.  IT'S GREAT FUN.")


```

这段代码定义了两个函数，函数1 `alert_unknown_problem_type`，函数2 `ask_question_loop`。

函数1 `alert_unknown_problem_type`的作用是打印一条消息，用于告知用户他们输入的类型问题类型是一个未知的错误。这个错误类型是在 `problem_type` 变量中传递给函数的。

函数2 `ask_question_loop` 的作用是不断地询问用户他们想要解决的问题类型。它使用了 `while` 循环，并且在循环条件中使用了 `valid` 和 `value` 变量。`valid` 变量用于存储用户是否输入了有效的答案，`value` 变量用于存储用户输入的答案。如果 `valid` 并且 `value` 的值为 `True`，那么函数将打印进一步的消息，以帮助用户更好地解决问题。如果 `valid` 并且 `value` 的值为 `False`，那么函数将打印一条消息，告知用户输入的问题类型是一个未知的错误。

在这两个函数中，函数1 `alert_unknown_problem_type` 的作用是告知用户一个未知的问题类型，而函数2 `ask_question_loop` 的作用是帮助用户更好地解决问题。


```
def alert_unknown_problem_type(user_name: str, problem_type: str) -> None:
    print(f"OH, {user_name}, YOUR ANSWER OF {problem_type} IS GREEK TO ME.")


def ask_question_loop(user_name: str) -> None:
    while True:
        problem_type = prompt_for_problems(user_name)
        if problem_type == "SEX":
            solve_sex_problem(user_name)
        elif problem_type == "HEALTH":
            solve_health_problem(user_name)
        elif problem_type == "MONEY":
            solve_money_problem(user_name)
        elif problem_type == "JOB":
            solve_job_problem(user_name)
        else:
            alert_unknown_problem_type(user_name, problem_type)

        while True:
            print()
            print(f"ANY MORE PROBLEMS YOU WANT SOLVED, {user_name}?")

            valid, value, msg = get_yes_or_no()
            if valid:
                if value:
                    print("WHAT KIND (SEX, MONEY, HEALTH, JOB)")
                    break
                else:
                    return
            print(f"JUST A SIMPLE 'YES' OR 'NO' PLEASE, {user_name}.")


```

这段代码的作用是向用户询问是否已经留下了5美元，如果没有留下，则提供一个负面消息，如果已经留下了，则不会产生任何费用。

具体来说，代码首先会输出一段文本，告诉用户这是一个收费咨询，费用为5美元，让用户留下一些钱。然后，代码会等待4秒钟，以确保用户有足够的时间来考虑。

接下来，代码会再次输出一段文本，询问用户是否已经留下了钱。如果用户已经留下了钱，则代码会输出一些负面消息，告诉用户是一个欺骗行为，费用不会退还。如果没有留下钱，则代码会再次输出一些负面消息，告诉用户需要支付咨询费用，以确保自己不会被欺骗。

整段代码的输出结果为：

```
THAT WILL BE $5.00 FOR THE ADVICE, user_name。
PLEASE LEAVE THE MONEY ON THE TERMINAL。
THIS IS A RIP OFF, user_name!!!
DO NOT PAY THE MONEY, user_name。
```

```
THIS IS A RIP OFF, user_name!!!
HEY, user_name, YOU LEFT NO MONEY AT ALL!
YOU ARE CHEATING ME OUT OF MY HARD-EARNED LIVING。
```

```
THAT'S HONEST, user_name, BUT HOW DO YOU EXPECT
ME TO GO ON WITH MY PSYCHOLOGY STUDIES IF MY PATIENTS
DON'T PAY THEIR BILLS?
```


```
def ask_for_fee(user_name: str) -> None:
    print()
    print(f"THAT WILL BE $5.00 FOR THE ADVICE, {user_name}.")
    print("PLEASE LEAVE THE MONEY ON THE TERMINAL.")
    time.sleep(4)
    print()
    print()
    print()
    print("DID YOU LEAVE THE MONEY?")

    while True:
        valid, value, msg = get_yes_or_no()
        if valid:
            if value:
                print(f"HEY, {user_name}, YOU LEFT NO MONEY AT ALL!")
                print("YOU ARE CHEATING ME OUT OF MY HARD-EARNED LIVING.")
                print()
                print(f"WHAT A RIP OFF, {user_name}!!!")
                print()
            else:
                print(f"THAT'S HONEST, {user_name}, BUT HOW DO YOU EXPECT")
                print("ME TO GO ON WITH MY PSYCHOLOGY STUDIES IF MY PATIENTS")
                print("DON'T PAY THEIR BILLS?")
            return
        else:
            print(f"YOUR ANSWER OF '{msg}' CONFUSES ME, {user_name}.")
            print("PLEASE RESPOND WITH 'YES' or 'NO'.")


```

这段代码定义了两个函数，一个是`unhappy_goodbye`，另一个是`happy_goodbye`。这两个函数都是用于与用户交互的，但是它们在传递给用户的信息上有所不同。

`unhappy_goodbye`函数的目的是让用户感觉不舒服，它通过打印一些不愉快的话来达到这个目的。具体来说，它打印了一些常见的不愉快用语，如"TAKE A WALK, XX。"这里的`XX`是用户的名字。这个函数还会打印一些清除屏幕的额外行，以便让用户有足够的空间进行他们的反应。

`happy_goodbye`函数的目的是给用户带来愉悦的体验，它通过打印一些愉快的话来达到这个目的。具体来说，它打印了一些常见的好心情用语，如"NICE MEETING YOU, XX,"和"HAVE A NICE DAY。"这里的`XX`是用户的名字。这个函数还会打印一些有用的信息，如`\n`，`\r`和`\t`，以便在用户输入时能够正确地排版。

在`main`函数中，首先会打印一些欢迎信息，然后让用户输入他们的名字。接着，会进入一个询问用户是否想要继续交互的循环。如果用户选择继续，则会调用`ask_enjoy_question`函数来询问用户他们想问的问题。如果用户选择结束这个交互，则会调用`happy_goodbye`函数来向用户传达一个愉快的信息。否则，则会调用`unhappy_goodbye`函数，让用户感到不舒服。


```
def unhappy_goodbye(user_name: str) -> None:
    print()
    print(f"TAKE A WALK, {user_name}.")
    print()
    print()


def happy_goodbye(user_name: str) -> None:
    print(f"NICE MEETING YOU, {user_name}, HAVE A NICE DAY.")


def main() -> None:
    print(" " * 33 + "HELLO")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")
    print("HELLO.  MY NAME IS CREATIVE COMPUTER.\n\n")
    print("WHAT'S YOUR NAME?")
    user_name = input()
    print()

    ask_enjoy_question(user_name)

    ask_question_loop(user_name)

    ask_for_fee(user_name)

    if False:
        happy_goodbye(user_name)
    else:
        unhappy_goodbye(user_name)


```

这段代码是一个Python程序中的一个if语句，它的作用是判断当前程序是否作为主程序运行。如果当前程序作为主程序运行，那么程序会执行if语句内部的代码，否则跳过if语句内部的代码。

在这段if语句中，有两个表达式，分别是__name__和__main__。其中，__name__表示当前程序的文件名，而__main__表示当前程序的入口函数。

if __name__ == "__main__":
   main()

这句话的意思是，如果当前程序的文件名和入口函数名相等，那么程序会执行if语句内部的代码，否则就跳过if语句内部的代码。而main()函数就是Python中的一内置函数，用于输出"Hello, World!"。因此，这段代码的作用就是输出"Hello, World!"。


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


### Hexapawn

The game of Hexapawn and a method to learn a strategy for playing the game was described in Martin Gardner’s “Mathematical Games” column in the March 1962 issue of _Scientific American_. The method described in the article was for a hypothetical learning machine composed of match boxes and colored beads. This has been generalized in the program HEX.

The program learns by elimination of bad moves. All positions encountered by the program and acceptable moves from them are stored in an array. When the program encounters an unfamiliar position, the position and all legal moves from it are added to the list. If the program loses a game, it erases the move that led to defeat. If it hits a position from which all moves have been deleted (they all led to defeat), it erases the move that got it there and resigns. Eventually, the program learns to play extremely well and, indeed, is unbeatable. The learning strategy could be adopted to other simple games with a finite number of moves (tic-tac-toe, small board checkers, or other chess-based games).

The original version of this program was written by R.A. Kaapke. It was subsequently modified by Jeff Dalton and finally by Steve North of Creative Computing.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=83)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=98)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Known Bugs

- There are valid board positions that will cause the program to print "ILLEGAL BOARD PATTERN" and break.  For example: human 8,5; computer 1,5; human 9,5; computer 3,5; human 7,5.  This is a valid game-over pattern, but it is not detected as such because of incorrect logic in lines 240-320 (intended to detect whether the computer has any legal moves).

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `46_Hexapawn/csharp/Board.cs`

This is a class definition for a `Board` that represents a chessboard. It has a `Pawn` array for each cell, and provides methods for getting and setting the value of each cell. It also defines a reflected board which is a copy of the board, and an enumeration for the board's elements.

The `Board` class implements the `IEnumerable<Pawn>` and `IEquatable<Board>` interfaces, which allow it to be used as a list of `Pawn`s and as an object for equality.

The `Board` class has a constructor that initializes the cells of the board with the default values, a constructor that initializes the board with a specified array of `Pawn`, and a constructor that initializes the board with a specified chessboard.

The `Board` class has several methods for getting and setting the value of each cell, including the `this[int index]` method which returns the cell at the given index, and the `GetEnumerator` method which returns an enumerator for the cells of the board.

The `Board` class also defines a `Reflected` method which returns a reflected board, and a `GetHashCode` method which returns the hash code of the board.


```
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using static Hexapawn.Pawn;

namespace Hexapawn;

internal class Board : IEnumerable<Pawn>, IEquatable<Board>
{
    private readonly Pawn[] _cells;

    public Board()
    {
        _cells = new[]
        {
            Black, Black, Black,
            None,  None,  None,
            White, White, White
        };
    }

    public Board(params Pawn[] cells)
    {
        _cells = cells;
    }

    public Pawn this[int index]
    {
        get => _cells[index - 1];
        set => _cells[index - 1] = value;
    }

    public Board Reflected => new(Cell.AllCells.Select(c => this[c.Reflected]).ToArray());

    public IEnumerator<Pawn> GetEnumerator() => _cells.OfType<Pawn>().GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    public override string ToString()
    {
        var builder = new StringBuilder().AppendLine();
        for (int row = 0; row < 3; row++)
        {
            builder.Append("          ");
            for (int col = 0; col < 3; col++)
            {
                builder.Append(_cells[row * 3 + col]);
            }
            builder.AppendLine();
        }
        return builder.ToString();
    }

    public bool Equals(Board other) => other?.Zip(this).All(x => x.First == x.Second) ?? false;

    public override bool Equals(object obj) => Equals(obj as Board);

    public override int GetHashCode()
    {
        var hash = 19;

        for (int i = 0; i < 9; i++)
        {
            hash = hash * 53 + _cells[i].GetHashCode();
        }

        return hash;
    }
}

```

# `46_Hexapawn/csharp/Cell.cs`



这段代码定义了一个名为Cell的类，用于表示一个细胞，该细胞包含一个整数编号，以及一个关于board中 middle列的镜像的引用。

Cell类包含一个静态的Cell数组，以及一个静态的Cell反射数组。Cell类的方法包括：

- AllCells：返回board中所有细胞的集合。
- TryCreate：尝试创建一个Cell引用，并返回它是否成功。
- Reflected：返回关于middle列的cell引用对象的 reflection。

除了以上方法外，Cell类还重写了toString方法，用于在toString方法中返回细胞对象的字符串表示形式。

从代码中可以看出，该代码的目的是提供一个数字表示的细胞对象，该对象可以用来在游戏中的各个位置进行操作，并支持使用反射操作获取到该细胞在board上的镜像。


```
using System;
using System.Collections.Generic;

namespace Hexapawn;

// Represents a cell on the board, numbered 1 to 9, with support for finding the reflection of the reference around
// the middle column of the board.
internal class Cell
{
    private static readonly Cell[] _cells = new Cell[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    private static readonly Cell[] _reflected = new Cell[] { 3, 2, 1, 6, 5, 4, 9, 8, 7 };
    private readonly int _number;
    private Cell(int number)
    {
        if (number < 1 || number > 9)
        {
            throw new ArgumentOutOfRangeException(nameof(number), number, "Must be from 1 to 9");
        }
        _number = number;
    }
    // Facilitates enumerating all the cells.
    public static IEnumerable<Cell> AllCells => _cells;
    // Takes a value input by the user and attempts to create a Cell reference
    public static bool TryCreate(float input, out Cell cell)
    {
        if (IsInteger(input) && input >= 1 && input <= 9)
        {
            cell = (int)input;
            return true;
        }
        cell = default;
        return false;
        static bool IsInteger(float value) => value - (int)value == 0;
    }
    // Returns the reflection of the cell reference about the middle column of the board.
    public Cell Reflected => _reflected[_number - 1];
    // Allows the cell reference to be used where an int is expected, such as the indexer in Board.
    public static implicit operator int(Cell c) => c._number;
    public static implicit operator Cell(int number) => new(number);
    public override string ToString() => _number.ToString();
}

```

# `46_Hexapawn/csharp/Computer.cs`

This looks like a Java implementation of a simple chess game where the player has a list of potential moves and the computer has a list of potential moves that reflect the previous move of the computer. The player can use the TryGetMoves and TrySelectMove methods to try to get a move, and if the move is found, the TryGetMoves method will return true and the move will be included in the moves list. If the move is not found in the moves list, the computer will have no legal moves and the player wins.


```
using System;
using System.Collections.Generic;
using System.Linq;
using Games.Common.IO;
using Games.Common.Randomness;
using static Hexapawn.Pawn;

namespace Hexapawn;

/// <summary>
/// Encapsulates the logic of the computer player.
/// </summary>
internal class Computer
{
    private readonly TextIO _io;
    private readonly IRandom _random;
    private readonly Dictionary<Board, List<Move>> _potentialMoves;
    private (List<Move>, Move) _lastMove;
    public Computer(TextIO io, IRandom random)
    {
        _io = io;
        _random = random;

        // This dictionary implements the data in the original code, which encodes board positions for which the
        // computer has a legal move, and the list of possible moves for each position:
        //   900 DATA -1,-1,-1,1,0,0,0,1,1,-1,-1,-1,0,1,0,1,0,1
        //   905 DATA -1,0,-1,-1,1,0,0,0,1,0,-1,-1,1,-1,0,0,0,1
        //   910 DATA -1,0,-1,1,1,0,0,1,0,-1,-1,0,1,0,1,0,0,1
        //   915 DATA 0,-1,-1,0,-1,1,1,0,0,0,-1,-1,-1,1,1,1,0,0
        //   920 DATA -1,0,-1,-1,0,1,0,1,0,0,-1,-1,0,1,0,0,0,1
        //   925 DATA 0,-1,-1,0,1,0,1,0,0,-1,0,-1,1,0,0,0,0,1
        //   930 DATA 0,0,-1,-1,-1,1,0,0,0,-1,0,0,1,1,1,0,0,0
        //   935 DATA 0,-1,0,-1,1,1,0,0,0,-1,0,0,-1,-1,1,0,0,0
        //   940 DATA 0,0,-1,-1,1,0,0,0,0,0,-1,0,1,-1,0,0,0,0
        //   945 DATA -1,0,0,-1,1,0,0,0,0
        //   950 DATA 24,25,36,0,14,15,36,0,15,35,36,47,36,58,59,0
        //   955 DATA 15,35,36,0,24,25,26,0,26,57,58,0
        //   960 DATA 26,35,0,0,47,48,0,0,35,36,0,0,35,36,0,0
        //   965 DATA 36,0,0,0,47,58,0,0,15,0,0,0
        //   970 DATA 26,47,0,0,47,58,0,0,35,36,47,0,28,58,0,0,15,47,0,0
        //
        // The original code loaded this data into two arrays.
        //   40 FOR I=1 TO 19: FOR J=1 TO 9: READ B(I,J): NEXT J: NEXT I
        //   45 FOR I=1 TO 19: FOR J=1 TO 4: READ M(I,J): NEXT J: NEXT I
        //
        // When finding moves for the computer the first array was searched for the current board position, or the
        // reflection of it, and the resulting index was used in the second array to get the possible moves.
        // With this dictionary we can just use the current board as the index, and retrieve a list of moves for
        // consideration by the computer.
        _potentialMoves = new()
        {
            [new(Black, Black, Black, White, None,  None,  None,  White, White)] = Moves((2, 4), (2, 5), (3, 6)),
            [new(Black, Black, Black, None,  White, None,  White, None,  White)] = Moves((1, 4), (1, 5), (3, 6)),
            [new(Black, None,  Black, Black, White, None,  None,  None,  White)] = Moves((1, 5), (3, 5), (3, 6), (4, 7)),
            [new(None,  Black, Black, White, Black, None,  None,  None,  White)] = Moves((3, 6), (5, 8), (5, 9)),
            [new(Black, None,  Black, White, White, None,  None,  White, None)]  = Moves((1, 5), (3, 5), (3, 6)),
            [new(Black, Black, None,  White, None,  White, None,  None,  White)] = Moves((2, 4), (2, 5), (2, 6)),
            [new(None,  Black, Black, None,  Black, White, White, None,  None)]  = Moves((2, 6), (5, 7), (5, 8)),
            [new(None,  Black, Black, Black, White, White, White, None,  None)]  = Moves((2, 6), (3, 5)),
            [new(Black, None,  Black, Black, None,  White, None,  White, None)]  = Moves((4, 7), (4, 8)),
            [new(None,  Black, Black, None,  White, None,  None,  None,  White)] = Moves((3, 5), (3, 6)),
            [new(None,  Black, Black, None,  White, None,  White, None,  None)]  = Moves((3, 5), (3, 6)),
            [new(Black, None,  Black, White, None,  None,  None,  None,  White)] = Moves((3, 6)),
            [new(None,  None,  Black, Black, Black, White, None,  None,  None)]  = Moves((4, 7), (5, 8)),
            [new(Black, None,  None,  White, White, White, None,  None,  None)]  = Moves((1, 5)),
            [new(None,  Black, None,  Black, White, White, None,  None,  None)]  = Moves((2, 6), (4, 7)),
            [new(Black, None,  None,  Black, Black, White, None,  None,  None)]  = Moves((4, 7), (5, 8)),
            [new(None,  None,  Black, Black, White, None,  None,  None,  None)]  = Moves((3, 5), (3, 6), (4, 7)),
            [new(None,  Black, None,  White, Black, None,  None,  None,  None)]  = Moves((2, 8), (5, 8)),
            [new(Black, None,  None,  Black, White, None,  None,  None,  None)]  = Moves((1, 5), (4, 7))
        };
    }

    // Try to make a move. We first try to find a legal move for the current board position.
    public bool TryMove(Board board)
    {
        if (TryGetMoves(board, out var moves, out var reflected) &&
            TrySelectMove(moves, out var move))
        {
            // We've found a move, so we record it as the last move made, and then announce and make the move.
            _lastMove = (moves, move);
            // If we found the move from a reflacted match of the board we need to make the reflected move.
            if (reflected) { move = move.Reflected; }
            _io.WriteLine($"I move {move}");
            move.Execute(board);
            return true;
        }
        // We haven't found a move for this board position, so remove the previous move that led to this board
        // position from future consideration. We don't want to make that move again, because we now know it's a
        // non-winning move.
        ExcludeLastMoveFromFuturePlay();
        return false;
    }

    // Looks up the given board and its reflection in the potential moves dictionary. If it's found then we have a
    // list of potential moves. If the board is not found in the dictionary then the computer has no legal moves,
    // and the human player wins.
    private bool TryGetMoves(Board board, out List<Move> moves, out bool reflected)
    {
        if (_potentialMoves.TryGetValue(board, out moves))
        {
            reflected = false;
            return true;
        }
        if (_potentialMoves.TryGetValue(board.Reflected, out moves))
        {
            reflected = true;
            return true;
        }
        reflected = default;
        return false;
    }

    // Get a random move from the list. If the list is empty, then we've previously eliminated all the moves for
    // this board position as being non-winning moves. We therefore resign the game.
    private bool TrySelectMove(List<Move> moves, out Move move)
    {
        if (moves.Any())
        {
            move = moves[_random.Next(moves.Count)];
            return true;
        }
        _io.WriteLine("I resign.");
        move = null;
        return false;
    }

    private void ExcludeLastMoveFromFuturePlay()
    {
        var (moves, move) = _lastMove;
        moves.Remove(move);
    }

    private static List<Move> Moves(params Move[] moves) => moves.ToList();

    public bool IsFullyAdvanced(Board board) =>
        board[9] == Black || board[8] == Black || board[7] == Black;
}

```

# `46_Hexapawn/csharp/Game.cs`

这段代码是一个游戏的主要类，负责处理游戏的逻辑。

它首先引入了两个外部类，TextIO 和 Board。TextIO 类用于从控制台读取输入，Board 类用于在棋盘上落子。

然后，它定义了一个 Play 方法，用于让玩家进行一次操作，并返回玩家的类型。

在内部，游戏实例化一个 Board 对象，并使用 TextIO 类从控制台读取棋盘信息。然后，游戏开始一个循环，每一轮玩家可以移动棋子，然后处理 AI 的动作，检查棋子的状态，并输出结果。

如果玩家没有移动棋子，AI 也没有动作，游戏将循环继续。如果玩家移动棋子，AI 会检查是否允许移动，并输出结果。如果玩家移动棋子后 AI 没有动作，游戏将循环继续。如果玩家移动棋子后 AI 允许移动，游戏将更新棋盘并继续循环。

总结起来，这段代码定义了一个游戏，可以进行 AI 控制，玩家可以移动棋子，并且可以在 AI 控制下一步棋。


```
using System;
using Games.Common.IO;

namespace Hexapawn;

// A single game of Hexapawn
internal class Game
{
    private readonly TextIO _io;
    private readonly Board _board;

    public Game(TextIO io)
    {
        _board = new Board();
        _io = io;
    }

    public object Play(Human human, Computer computer)
    {
        _io.WriteLine(_board);
        while(true)
        {
            human.Move(_board);
            _io.WriteLine(_board);
            if (!computer.TryMove(_board))
            {
                return human;
            }
            _io.WriteLine(_board);
            if (computer.IsFullyAdvanced(_board) || human.HasNoPawns(_board))
            {
                return computer;
            }
            if (!human.HasLegalMove(_board))
            {
                _io.Write("You can't move, so ");
                return computer;
            }
        }
    }
}

```

# `46_Hexapawn/csharp/GameSeries.cs`



This code appears to be a game where two players take turns playing a series of games. The game is controlled by one player (the `Human` player), while the other player is the computer (the `Computer` player). 

The `GameSeries` class seems to keep track of the current game state, including the current score for each player, the winner of each game, and the overall score for the series.

The `Play` method of the `GameSeries` class seems to start by displaying the title of the game and asking the player if they want instructions. If the player chooses 'Y', the instructions for the game are displayed.

Then, the game is played using the `Game` class. This class seems to be responsible for generating the game play and keeping track of the winner.

Finally, the game win/loss record is updated and the final score is displayed.


```
using System.Collections.Generic;
using System.Linq;
using Games.Common.IO;
using Games.Common.Randomness;
using Hexapawn.Resources;

namespace Hexapawn;

// Runs series of games between the computer and the human player
internal class GameSeries
{
    private readonly TextIO _io;
    private readonly Computer _computer;
    private readonly Human _human;
    private readonly Dictionary<object, int> _wins;

    public GameSeries(TextIO io, IRandom random)
    {
        _io = io;
        _computer = new(io, random);
        _human = new(io);
        _wins = new() { [_computer] = 0, [_human] = 0 };
    }

    public void Play()
    {
        _io.Write(Resource.Streams.Title);

        if (_io.GetYesNo("Instructions") == 'Y')
        {
            _io.Write(Resource.Streams.Instructions);
        }

        while (true)
        {
            var game = new Game(_io);

            var winner = game.Play(_human, _computer);
            _wins[winner]++;
            _io.WriteLine(winner == _computer ? "I win." : "You win.");

            _io.Write($"I have won {_wins[_computer]} and you {_wins[_human]}");
            _io.WriteLine($" out of {_wins.Values.Sum()} games.");
            _io.WriteLine();
        }
    }
}

```

# `46_Hexapawn/csharp/Human.cs`



该代码是一个用于模拟简单棋游戏的程序。主要作用是控制玩家的移动操作。以下是代码的作用解释：

1. 引入System命名空间：使用System命名空间可以方便地引用一些通用的类和接口，提高程序的可读性和可维护性。

2. 引入Games.Common.IO：该命名空间中包含了一些与文件操作有关的接口和类，包括文件读取、写入等操作。

3. 定义Human类：该类是所有游戏的玩家类，用于处理玩家在游戏中的操作。

4. 移动函数Move：该函数用于让玩家从当前位置移动到指定位置。玩家在每次移动前需要先检查游戏 board 中是否存在合法移动。

5. 检查函数HasLegalMove：该函数用于检查玩家输入的移动是否合法，包括检查行、列、棋是否可以移动到该位置等。

6. 移动函数TryExecute：该函数用于在玩家输入非法移动后，尝试执行移动操作。如果移动合法，则返回True，否则返回False。

7. 构造函数Human：该函数用于创建一个Human实例，需要一个TextIO实例作为参数。

8. 行移事件程序行移：该事件程序让玩家从当前位置开始向指定方向移动。

9. 列移事件程序列移：该事件程序让玩家从当前位置开始向指定列移动。

10. 棋移动事件程序：该事件程序允许玩家在指定位置放置棋子。

11. 检查函数AllCells：该函数用于检查游戏 board 中所有细胞是否都是白色。

12. 检查函数HasNoPawns：该函数用于检查游戏 board 中是否所有细胞都是棋子（即没有玩家位置的细胞）。

13. 移动函数Execute：该函数用于在指定位置放置棋子。

14. 文件读取事件：该事件程序用于从文件中读取玩家输入的行或列。

15. 文件写入事件：该事件程序用于将指定行或列的棋子移动到指定位置。


```
using System;
using System.Linq;
using Games.Common.IO;
using static Hexapawn.Cell;
using static Hexapawn.Move;
using static Hexapawn.Pawn;

namespace Hexapawn;

internal class Human
{
    private readonly TextIO _io;

    public Human(TextIO io)
    {
        _io = io;
    }

    public void Move(Board board)
    {
        while (true)
        {
            var move = _io.ReadMove("Your move");

            if (TryExecute(board, move)) { return; }

            _io.WriteLine("Illegal move.");
        }
    }

    public bool HasLegalMove(Board board)
    {
        foreach (var from in AllCells.Where(c => c > 3))
        {
            if (board[from] != White) { continue; }

            if (HasLegalMove(board, from))
            {
                return true;
            }
        }

        return false;
    }

    private bool HasLegalMove(Board board, Cell from) =>
        Right(from).IsRightDiagonalToCapture(board) ||
        Straight(from).IsStraightMoveToEmptySpace(board) ||
        from > 4 && Left(from).IsLeftDiagonalToCapture(board);

    public bool HasNoPawns(Board board) => board.All(c => c != White);

    public bool TryExecute(Board board, Move move)
    {
        if (board[move.From] != White) { return false; }

        if (move.IsStraightMoveToEmptySpace(board) ||
            move.IsLeftDiagonalToCapture(board) ||
            move.IsRightDiagonalToCapture(board))
        {
            move.Execute(board);
            return true;
        }

        return false;
    }
}

```

# `46_Hexapawn/csharp/IReadWriteExtensions.cs`

这段代码是一个自定义的 C# 类，提供了三种输入方法，包括：

1. `GetYesNo` 方法：该方法接受一个字符串 prompt，并返回一个字符。它通过循环读取一个 "是（Y）还是（N）" 的输入，如果循环中检测到任何一个字符，则返回相应的字符。

2. `ReadMove` 方法：该方法接受一个字符串 prompt，并返回一个移动（例如步骤，方向，速度等）。它通过循环读取两个整数（from 和 to），然后使用 `TryCreate` 方法从输入中创建移动对象，如果循环成功，则返回该移动对象。如果循环失败，则输出 "Illegal Coordinates。"

3. `WriteToConsole` 方法：该方法接受一个字符串 prompt，并将其输出到控制台。它使用 `WriteLine` 方法将提示信息输出到控制台。


```
using System;
using System.Linq;
using Games.Common.IO;

namespace Hexapawn;

// Provides input methods which emulate the BASIC interpreter's keyboard input routines
internal static class IReadWriteExtensions
{
    internal static char GetYesNo(this IReadWrite io, string prompt)
    {
        while (true)
        {
            var response = io.ReadString($"{prompt} (Y-N)").FirstOrDefault();
            if ("YyNn".Contains(response))
            {
                return char.ToUpperInvariant(response);
            }
        }
    }

    // Implements original code:
    //   120 PRINT "YOUR MOVE";
    //   121 INPUT M1,M2
    //   122 IF M1=INT(M1)AND M2=INT(M2)AND M1>0 AND M1<10 AND M2>0 AND M2<10 THEN 130
    //   123 PRINT "ILLEGAL CO-ORDINATES."
    //   124 GOTO 120
    internal static Move ReadMove(this IReadWrite io, string prompt)
    {
        while(true)
        {
            var (from, to) = io.Read2Numbers(prompt);

            if (Move.TryCreate(from, to, out var move))
            {
                return move;
            }

            io.WriteLine("Illegal Coordinates.");
        }
    }
}

```

# `46_Hexapawn/csharp/Move.cs`

This is a class written in C# that represents a chessboard. It has a `Board` class that stores the current state of the board, and it has methods for executing a move, reflection, and producing a mirror image of the board.

The `Board` class has a `From` and `To` property that track the current cell moved from and to, respectively. The class has a `Reflected` property that is a tuple of two integers that represents the reflected move, from bottom left to top right.

The class has two methods for executing a move and reflecting the move of another cell. The class also has a method `TryCreate` that takes two floating point coordinates as input and tries to create a `Move` object.

The `Move` class has a constructor that takes two integer arguments, `From` and `To`, and creates a new `Move` object with the current cell from and the new cell to.

The `Move` class also has methods for creating mirror images of the board, getting the current cell moved from, and getting the cell to the right of the central column of the board.


```
using static Hexapawn.Pawn;

namespace Hexapawn;

/// <summary>
/// Represents a move which may, or may not, be legal.
/// </summary>
internal class Move
{
    private readonly Cell _from;
    private readonly Cell _to;
    private readonly int _metric;

    public Move(Cell from, Cell to)
    {
        _from = from;
        _to = to;
        _metric = _from - _to;
    }

    public void Deconstruct(out Cell from, out Cell to)
    {
        from = _from;
        to = _to;
    }

    public Cell From => _from;

    // Produces the mirror image of the current moved, reflected around the central column of the board.
    public Move Reflected => (_from.Reflected, _to.Reflected);

    // Allows a tuple of two ints to be implicitly converted to a Move.
    public static implicit operator Move((int From, int To) value) => new(value.From, value.To);

    // Takes floating point coordinates, presumably from keyboard input, and attempts to create a Move object.
    public static bool TryCreate(float input1, float input2, out Move move)
    {
        if (Cell.TryCreate(input1, out var from) &&
            Cell.TryCreate(input2, out var to))
        {
            move = (from, to);
            return true;
        }

        move = default;
        return false;
    }

    public static Move Right(Cell from) => (from, from - 2);
    public static Move Straight(Cell from) => (from, from - 3);
    public static Move Left(Cell from) => (from, from - 4);

    public bool IsStraightMoveToEmptySpace(Board board) => _metric == 3 && board[_to] == None;

    public bool IsLeftDiagonalToCapture(Board board) => _metric == 4 && _from != 7 && board[_to] == Black;

    public bool IsRightDiagonalToCapture(Board board) =>
        _metric == 2 && _from != 9 && _from != 6 && board[_to] == Black;

    public void Execute(Board board)
    {
        board[_to] = board[_from];
        board[_from] = None;
    }

    public override string ToString() => $"from {_from} to {_to}";
}

```

# `46_Hexapawn/csharp/Pawn.cs`

这段代码定义了一个名为"Pawn"的类，用于表示棋盘上的细胞。这个类包含三个成员变量，分别是代表黑色、白色和未标记的细胞的三种不同字符类型的实例。

接着，在类中定义了一个名为"Pawn"的静态构造函数，该构造函数接收一个字符类型的参数，用于初始化细胞的符号。

另外，还定义了一个名为"ToString"的静态方法，用于将细胞的符号字符串化并返回。

最后，在类中没有做任何其他明显的操作，因此，除了定义类的结构和成员变量之外，没有其他的代码行为。


```
namespace Hexapawn;

// Represents the contents of a cell on the board
internal class Pawn
{
    public static readonly Pawn Black = new('X');
    public static readonly Pawn White = new('O');
    public static readonly Pawn None = new('.');

    private readonly char _symbol;

    private Pawn(char symbol)
    {
        _symbol = symbol;
    }

    public override string ToString() => _symbol.ToString();
}


```

# `46_Hexapawn/csharp/Program.cs`

这段代码使用了三个自定义的库：Games.Common.IO、Games.Common.Randomness和Hexapawn，实现了创建一个基于ConsoleIO和RandomNumberGenerator的GameSeries游戏实例，并用它来运行一个测试游戏。

GameSeries是一个创建IO、RandomNumberGenerator和Hexapawn的类，提供了常见的文件操作、随机数生成和游戏逻辑功能。

新创建的GameSeries实例接收两个参数，一个是ConsoleIO，另一个是RandomNumberGenerator。ConsoleIO用于输出游戏中的信息和信息，RandomNumberGenerator用于生成随机数。

接下来，用这两个参数创建一个GameSeries实例后，用它来调用其的Play()方法运行游戏。这里的"游戏系列"应该是用来运行Hexapawn游戏的。


```
﻿using Games.Common.IO;
using Games.Common.Randomness;
using Hexapawn;

new GameSeries(new ConsoleIO(), new RandomNumberGenerator()).Play();


```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `46_Hexapawn/csharp/Resources/Resource.cs`

这段代码是一个名为 `Resource` 的类，其中包含一个名为 `Instructions` 的静态字段和一个名为 `Title` 的静态字段。

`Instructions` 和 `Title` 字段都使用 `GetStream` 方法获取资源文件中的内容。这个方法使用了 `[CallerMemberName]` 的注解，用于将获取资源的名称指定给调用者。

另外，还有一段使用 `Assembly.GetExecutingAssembly()` 和 `GetManifestResourceStream` 方法获取指定资源文件内容的方法，这个方法将会在程序运行时动态获取指定的资源文件内容，并返回一个 `Stream` 类型的变量。


```
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace Hexapawn.Resources;

internal static class Resource
{
    internal static class Streams
    {
        public static Stream Instructions => GetStream();
        public static Stream Title => GetStream();
    }

    private static Stream GetStream([CallerMemberName] string name = null)
        => Assembly.GetExecutingAssembly().GetManifestResourceStream($"Hexapawn.Resources.{name}.txt");
}
```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `46_Hexapawn/javascript/hexapawn.js`

这段代码定义了两个函数，分别是`print()`和`input()`。

`print()`函数的作用是打印字符串`str`到页面上，并将其添加到页面上指定的元素中。这个字符串是由`console.log()`函数打印出来的，而`console.log()`函数将字符串转换为字符数组，然后将它们添加到`document.getElementById()`函数返回的元素中。

`input()`函数的作用是接收用户的输入，它将接收用户在输入框中输入的字符串，并在输入框中创建一个事件监听器，以便在用户按下回车键时接收输入。在监听器中，使用`document.getElementById()`函数创建一个输入框元素，设置其`type`属性为`text`，`length`属性为`50`，并将它添加到页面上指定的元素中。然后，将输入框元素的`focus`属性设置为`true`，以便获取用户的输入。接下来，当用户按下回车键时，将接收到的输入存储在`input_str`变量中，并将其添加到页面上指定的元素中，同时将打印机中的内容也添加到页面中。


```
// HEXAPAWN
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

这是一段 JavaScript 代码，定义了一个名为 "tab" 的函数，接受一个名为 "space" 的参数。函数的主要目的是在给定的字符串空间内添加指定的字符，以创建一个指定的输出字符串。

在函数体中，首先定义了一个名为 "str" 的字符串变量，并使用 while 循环从给定的空间值中逐个删除字符，每次添加一个空格。当 space 的值减少到 0 时，循环结束，并返回字符串 "str"，即最终的结果。

此外，还定义了一个名为 "ba" 的数组变量，用于存储给定的字符串。这些数组元素使用逗号分隔，指定不同的空格数量，以便在循环中使用。

最后，在代码的最后部分，定义了一个名为 "tab" 的函数，接受一个空字符串作为参数，使用 while 循环和移位运算符（如 "space--"）来创建一个包含指定字符的输出字符串。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var ba = [,
          [,-1,-1,-1,1,0,0,0,1,1],
          [,-1,-1,-1,0,1,0,1,0,1],
          [,-1,0,-1,-1,1,0,0,0,1],
          [,0,-1,-1,1,-1,0,0,0,1],
          [,-1,0,-1,1,1,0,0,1,0],
          [,-1,-1,0,1,0,1,0,0,1],
          [,0,-1,-1,0,-1,1,1,0,0],
          [,0,-1,-1,-1,1,1,1,0,0],
          [,-1,0,-1,-1,0,1,0,1,0],
          [,0,-1,-1,0,1,0,0,0,1],
          [,0,-1,-1,0,1,0,1,0,0],
          [,-1,0,-1,1,0,0,0,0,1],
          [,0,0,-1,-1,-1,1,0,0,0],
          [,-1,0,0,1,1,1,0,0,0],
          [,0,-1,0,-1,1,1,0,0,0],
          [,-1,0,0,-1,-1,1,0,0,0],
          [,0,0,-1,-1,1,0,0,0,0],
          [,0,-1,0,1,-1,0,0,0,0],
          [,-1,0,0,-1,1,0,0,0,0]];
```

这段代码创建了一个包含多行数据的二维数组`ma`，每一行有四个元素，分别代表不同的数据类型。

数组中的元素如下：
1. `[, ,24,25,36,0],` - 这是一个包含四个元素的行，每个元素都是一个数值，代表第二行。
2. `[,14,15,36,0],` - 这是一个包含四个元素的行，每个元素都是一个数值，代表第一行。
3. `[,15,35,36,47],` - 这是一个包含四个元素的行，每个元素都是一个数值，代表第三行。
4. `[,36,58,59,0],` - 这是一个包含四个元素的行，每个元素都是一个数值，代表第四行。
5. `[,15,35,36,0],` - 这是一个包含四个元素的行，每个元素都是一个数值，代表第二行。
6. `[,26,57,58,0],` - 这是一个包含四个元素的行，每个元素都是一个数值，代表第三行。
7. `[,26,35,0,0],` - 这是一个包含四个元素的行，每个元素都是一个数值，代表第四行。
8. `[,47,48,0,0],` - 这是一个包含四个元素的行，每个元素都是一个数值，代表第三行。
9. `[,35,36,0,0],` - 这是一个包含四个元素的行，每个元素都是一个数值，代表第二行。
10. `[,35,36,0,0],` - 这是一个包含四个元素的行，每个元素都是一个数值，代表第四行。
11. `[,36,0,0,0],` - 这是一个包含四个元素的行，每个元素都是一个数值，代表第二行。
12. `[,47,58,0,0],` - 这是一个包含四个元素的行，每个元素都是一个数值，代表第一行。
13. `[,15,0,0,0],` - 这是一个包含四个元素的行，每个元素都是一个数值，代表第二行。
14. `[,26,47,0,0],` - 这是一个包含四个元素的行，每个元素都是一个数值，代表第三行。
15. `[,47,58,0,0],` - 这是一个包含四个元素的行，每个元素都是一个数值，代表第一行。
16. `[,35,36,47,0],` - 这是一个包含四个元素的行，每个元素都是一个数值，代表第三行。
17. `[,28,58,0,0],` - 这是一个包含四个元素的行，每个元素都是一个数值，代表第三行。
18. `[,15,47,0,0],` - 这是一个包含四个元素的行，每个元素都是一个数值，代表第二行。


```
var ma = [,
          [,24,25,36,0],
          [,14,15,36,0],
          [,15,35,36,47],
          [,36,58,59,0],
          [,15,35,36,0],
          [,24,25,26,0],
          [,26,57,58,0],
          [,26,35,0,0],
          [,47,48,0,0],
          [,35,36,0,0],
          [,35,36,0,0],
          [,36,0,0,0],
          [,47,58,0,0],
          [,15,0,0,0],
          [,26,47,0,0],
          [,47,58,0,0],
          [,35,36,47,0],
          [,28,58,0,0],
          [,15,47,0,0]];
```

这段代码定义了一个名为 `show_board` 的函数，它使用了三个变量 `s`、`t` 和 `ps`。函数的作用是打印一个 3x3 的矩阵，并在矩阵中打印出一些指定的字符。

具体来说，代码中首先定义了一个包含六个整数的数组 `s`，一个包含七个字符的数组 `t`，和一个字符串 `ps`。接着，定义了一个名为 `show_board` 的函数。

函数体中，首先打印一个 10 字节的制表，用来分隔不同的输入行。然后，从 `i` 循环到 `3`，从 `j` 循环到 `3`，打印出 `ps` 数组中 `s[(i - 1) * 3 + j] + 1` 对应的字符。这里的 `i` 和 `j` 是循环变量，它们分别从 `1` 递增到 `3`。

最后，在循环结束后，再次打印一个 10 字节的制表，并输出一些指定的字符，占一行。


```
var s = [];
var t = [];
var ps = "X.O";

function show_board()
{
    print("\n");
    for (var i = 1; i <= 3; i++) {
        print(tab(10));
        for (var j = 1; j <= 3; j++) {
            print(ps[s[(i - 1) * 3 + j] + 1]);
        }
        print("\n");
    }
}

```

这段代码定义了一个名为 `mirror` 的函数 `mirror(x)`，它接受一个整数参数 `x`。

函数的逻辑是基于一系列 `if` 语句，如果 `x` 的值等于 1，那么返回 3；如果 `x` 的值等于 3，那么返回 1；如果 `x` 的值等于 6，那么返回 4；如果 `x` 的值等于 4，那么返回 6；如果 `x` 的值等于 9，那么返回 7；如果 `x` 的值等于 7，那么返回 9。如果以上所有条件都不符合，那么函数返回 `x`。

简而言之，这个函数的作用是返回给定的整数 `x` 的下一个整数，如果给定的整数 `x` 在一个数列中，函数将返回该数列中的下一个整数；如果给定的整数 `x` 在一个数列中，函数将返回给定的整数 `x`。


```
function mirror(x)
{
    if (x == 1)
        return 3;
    if (x == 3)
        return 1;
    if (x == 6)
        return 4;
    if (x == 4)
        return 6;
    if (x == 9)
        return 7;
    if (x == 7)
        return 9;
    return x;
}

```

This appears to be a computer program written in Java, with some elements of Python and C++ mixed in as well. It is designed to play a game of chess, where one player (the computer) takes turns moving pieces on a 8x8 chessboard and the other player (the human) plays the game.

The program starts by initializing the chessboard and setting the computer's first move to be a move to the d7 square. It then runs a while loop that continues until one of the players has won (either by reaching a specific condition or by having the game end because one of the players left the board), or the game is a draw.

Within the while loop, the program uses a variety of functions to handle the different moves that the computer can make. These functions include moving the computer's pieces, checking for legal moves for the human player, and displaying the board after the game is over.

The program also includes a function that prints out the final board after the game is over, as well as a function that prints out the winner (either the computer or the human) and the number of games that were played.

Note: This program is based on the assumption that the game is played on a standard 8x8 chessboard, with white pieces on a1-8 square and black pieces on a9-8 square. The exact look of the chessboard may vary depending on the size of the board and the specific implementation.


```
// Main program
async function main()
{
    print(tab(32) + "HEXAPAWN\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // HEXAPAWN:  INTERPRETATION OF HEXAPAWN GAME AS PRESENTED IN
    // MARTIN GARDNER'S "THE UNEXPECTED HANGING AND OTHER MATHEMATIC-
    // AL DIVERSIONS", CHAPTER EIGHT:  A MATCHBOX GAME-LEARNING MACHINE
    // ORIGINAL VERSION FOR H-P TIMESHARE SYSTEM BY R.A. KAAPKE 5/5/76
    // INSTRUCTIONS BY JEFF DALTON
    // CONVERSION TO MITS BASIC BY STEVE NORTH
    for (i = 0; i <= 9; i++) {
        s[i] = 0;
    }
    w = 0;
    l = 0;
    do {
        print("INSTRUCTIONS (Y-N)");
        str = await input();
        str = str.substr(0, 1);
    } while (str != "Y" && str != "N") ;
    if (str == "Y") {
        print("\n");
        print("THIS PROGRAM PLAYS THE GAME OF HEXAPAWN.\n");
        print("HEXAPAWN IS PLAYED WITH CHESS PAWNS ON A 3 BY 3 BOARD.\n");
        print("THE PAWNS ARE MOVED AS IN CHESS - ONE SPACE FORWARD TO\n");
        print("AN EMPTY SPACE OR ONE SPACE FORWARD AND DIAGONALLY TO\n");
        print("CAPTURE AN OPPOSING MAN.  ON THE BOARD, YOUR PAWNS\n");
        print("ARE 'O', THE COMPUTER'S PAWNS ARE 'X', AND EMPTY \n");
        print("SQUARES ARE '.'.  TO ENTER A MOVE, TYPE THE NUMBER OF\n");
        print("THE SQUARE YOU ARE MOVING FROM, FOLLOWED BY THE NUMBER\n");
        print("OF THE SQUARE YOU WILL MOVE TO.  THE NUMBERS MUST BE\n");
        print("SEPERATED BY A COMMA.\n");
        print("\n");
        print("THE COMPUTER STARTS A SERIES OF GAMES KNOWING ONLY WHEN\n");
        print("THE GAME IS WON (A DRAW IS IMPOSSIBLE) AND HOW TO MOVE.\n");
        print("IT HAS NO STRATEGY AT FIRST AND JUST MOVES RANDOMLY.\n");
        print("HOWEVER, IT LEARNS FROM EACH GAME.  THUS, WINNING BECOMES\n");
        print("MORE AND MORE DIFFICULT.  ALSO, TO HELP OFFSET YOUR\n");
        print("INITIAL ADVANTAGE, YOU WILL NOT BE TOLD HOW TO WIN THE\n");
        print("GAME BUT MUST LEARN THIS BY PLAYING.\n");
        print("\n");
        print("THE NUMBERING OF THE BOARD IS AS FOLLOWS:\n");
        print(tab(10) + "123\n");
        print(tab(10) + "456\n");
        print(tab(10) + "789\n");
        print("\n");
        print("FOR EXAMPLE, TO MOVE YOUR RIGHTMOST PAWN FORWARD,\n");
        print("YOU WOULD TYPE 9,6 IN RESPONSE TO THE QUESTION\n");
        print("'YOUR MOVE ?'.  SINCE I'M A GOOD SPORT, YOU'LL ALWAYS\n");
        print("GO FIRST.\n");
        print("\n");
    }
    while (1) {
        x = 0;
        y = 0;
        s[4] = 0;
        s[5] = 0;
        s[6] = 0;
        s[1] = -1;
        s[2] = -1;
        s[3] = -1;
        s[7] = 1;
        s[8] = 1;
        s[9] = 1;
        show_board();
        while (1) {
            while (1) {
                print("YOUR MOVE");
                str = await input();
                m1 = parseInt(str);
                m2 = parseInt(str.substr(str.indexOf(",") + 1));
                if (m1 > 0 && m1 < 10 && m2 > 0 && m2 < 10) {
                    if (s[m1] != 1 || s[m2] == 1 || (m2 - m1 != -3 && s[m2] != -1) || (m2 > m1) || (m2 - m1 == -3 && s[m2] != 0) || (m2 - m1 < -4) || (m1 == 7 && m2 == 3))
                        print("ILLEGAL MOVE.\n");
                    else
                        break;
                } else {
                    print("ILLEGAL CO-ORDINATES.\n");
                }
            }

            // Move player's pawn
            s[m1] = 0;
            s[m2] = 1;
            show_board();

            // Find computer pawns
            for (i = 1; i <= 9; i++) {
                if (s[i] == -1)
                    break;
            }
            // If none or player reached top then finish
            if (i > 9 || s[1] == 1 || s[2] == 1 || s[3] == 1) {
                computer = false;
                break;
            }
            // Find computer pawns with valid move
            for (i = 1; i <= 9; i++) {
                if (s[i] != -1)
                    continue;
                if (s[i + 3] == 0
                 || (mirror(i) == i && (s[i + 2] == 1 || s[i + 4] == 1))
                 || (i <= 3 && s[5] == 1)
                 || s[8] == 1)
                    break;
            }
            if (i > 9) {  // Finish if none possible
                computer = false;
                break;
            }
            for (i = 1; i <= 19; i++) {
                for (j = 1; j <= 3; j++) {
                    for (k = 3; k >= 1; k--) {
                        t[(j - 1) * 3 + k] = ba[i][(j - 1) * 3 + 4 - k];
                    }
                }
                for (j = 1; j <= 9; j++) {
                    if (s[j] != ba[i][j])
                        break;
                }
                if (j > 9) {
                    r = 0;
                    break;
                }
                for (j = 1; j <= 9; j++) {
                    if (s[j] != t[j])
                        break;
                }
                if (j > 9) {
                    r = 1;
                    break;
                }
            }
            if (i > 19) {
                print("ILLEGAL BOARD PATTERN\n");
                break;
            }
            x = i;
            for (i = 1; i <= 4; i++) {
                if (ma[x][i] != 0)
                    break;
            }
            if (i > 4) {
                print("I RESIGN.\n");
                computer = false;
                break;
            }
            // Select random move from possibilities
            do {
                y = Math.floor(Math.random() * 4 + 1);
            } while (ma[x][y] == 0) ;
            // Announce move
            if (r == 0) {
                print("I MOVE FROM " + Math.floor(ma[x][y] / 10) + " TO " + ma[x][y] % 10 + "\n");
                s[Math.floor(ma[x][y] / 10)] = 0;
                s[ma[x][y] % 10] = -1;
            } else {
                print("I MOVE FROM " + mirror(Math.floor(ma[x][y] / 10)) + " TO " + mirror(ma[x][y]) % 10 + "\n");
                s[mirror(Math.floor(ma[x][y] / 10))] = 0;
                s[mirror(ma[x][y] % 10)] = -1;
            }
            show_board();
            // Finish if computer reaches bottom
            if (s[7] == -1 || s[8] == -1 || s[9] == -1) {
                computer = true;
                break;
            }
            // Finish if no player pawns
            for (i = 1; i <= 9; i++) {
                if (s[i] == 1)
                    break;
            }
            if (i > 9) {
                computer = true;
                break;
            }
            // Finish if player cannot move
            for (i = 1; i <= 9; i++) {
                if (s[i] != 1)
                    continue;
                if (s[i - 3] == 0)
                    break;
                if (mirror(i) != i) {
                    if (i >= 7) {
                        if (s[5] == -1)
                            break;
                    } else {
                        if (s[2] == -1)
                            break;
                    }
                } else {
                    if (s[i - 2] == -1 || s[i - 4] == -1)
                        break;
                }

            }
            if (i > 9) {
                print("YOU CAN'T MOVE, SO ");
                computer = true;
                break;
            }
        }
        if (computer) {
            print("I WIN.\n");
            w++;
        } else {
            print("YOU WIN\n");
            ma[x][y] = 0;
            l++;
        }
        print("I HAVE WON " + w + " AND YOU " + l + " OUT OF " + (l + w) + " GAMES.\n");
        print("\n");
    }
}

```

这道题目缺少上下文，无法给出具体的解释。不过，如果您能提供更多信息，我会尽力帮助您理解代码的作用。


```
main();

```