# BasicComputerGames源码解析 14

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `06_Banner/javascript/banner.js`

这段代码是一个 JavaScript 函数，具有以下功能：

1. `print()` 函数用于将文本内容输出到页面上。它接收一个字符串参数，并将其作为文本节点添加到页面上元素的 `appendChild()` 方法中。
2. `input()` 函数用于获取用户输入的字符串。它接收一个空字符串作为参数，并返回一个Promise对象。在Promise对象中，它使用户可以访问输入元素，并在输入元素的关键事件上监听事件处理程序。当用户按下键盘上的13键时，它会将输入的字符串作为参数传递给`print()`函数，并将其输出到页面上。


```
// BANNER
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

It looks like a valid Unix Shell script. The script appears to be a preparation script for a network installation. The script is written in Bash shell script language and it is designed to run on Linux or UNIX-based operating systems.

The script contains a series of commands that are used for different tasks, such as installing the necessary software, copying files, and creating a installation directory. The script also sets up a systemd service for the迎合 web server.



```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var letters = [" ",0,0,0,0,0,0,0,
               "A",505,37,35,34,35,37,505,
               "G",125,131,258,258,290,163,101,
               "E",512,274,274,274,274,258,258,
               "T",2,2,2,512,2,2,2,
               "W",256,257,129,65,129,257,256,
               "L",512,257,257,257,257,257,257,
               "S",69,139,274,274,274,163,69,
               "O",125,131,258,258,258,131,125,
               "N",512,7,9,17,33,193,512,
               "F",512,18,18,18,18,2,2,
               "K",512,17,17,41,69,131,258,
               "B",512,274,274,274,274,274,239,
               "D",512,258,258,258,258,131,125,
               "H",512,17,17,17,17,17,512,
               "M",512,7,13,25,13,7,512,
               "?",5,3,2,354,18,11,5,
               "U",128,129,257,257,257,129,128,
               "R",512,18,18,50,82,146,271,
               "P",512,18,18,18,18,18,15,
               "Q",125,131,258,258,322,131,381,
               "Y",8,9,17,481,17,9,8,
               "V",64,65,129,257,129,65,64,
               "X",388,69,41,17,41,69,388,
               "Z",386,322,290,274,266,262,260,
               "I",258,258,258,512,258,258,258,
               "C",125,131,258,258,258,131,69,
               "J",65,129,257,257,257,129,128,
               "1",0,0,261,259,512,257,257,
               "2",261,387,322,290,274,267,261,
               "*",69,41,17,512,17,41,69,
               "3",66,130,258,274,266,150,100,
               "4",33,49,41,37,35,512,33,
               "5",160,274,274,274,274,274,226,
               "6",194,291,293,297,305,289,193,
               "7",258,130,66,34,18,10,8,
               "8",69,171,274,274,274,171,69,
               "9",263,138,74,42,26,10,7,
               "=",41,41,41,41,41,41,41,
               "!",1,1,1,384,1,1,1,
               "0",57,69,131,258,131,69,57,
               ".",1,1,129,449,129,1,1];

```

这段代码的作用是：

1. 创建三个空数组f、j和s。
2. 通过三个输入函数分别获取用户输入的字符数组ls、字符串as和整数数组as中的元素个数。
3. 通过条件语句判断输入的字符串中是否有所有字符，如果是，则打印字符并输出到控制台。
4. 通过循环和字符串比较判断，在输出到控制台的字符串中，是否找到了相应的元素，并将其打印到控制台。
5. 通过循环输出所有元素后，等待用户按回车键，此时不输出任何东西。
6. 通过循环输出所有元素后，等待用户按回车键，此时不输出任何东西。


```
f = [];
j = [];
s = [];

// Main program
async function main()
{
    print("HORIZONTAL");
    x = parseInt(await input());
    print("VERTICAL");
    y = parseInt(await input());
    print("CENTERED");
    ls = await input();
    g1 = 0;
    if (ls > "P")
        g1 = 1;
    print("CHARACTER (TYPE 'ALL' IF YOU WANT CHARACTER BEING PRINTED)");
    ms = await input();
    print("STATEMENT");
    as = await input();
    print("SET PAGE");	// This means to prepare printer, just press Enter
    os = await input();

    for (t = 0; t < as.length; t++) {
        ps = as.substr(t, 1);
        for (o = 0; o < 50 * 8; o += 8) {
            if (letters[o] == ps) {
                for (u = 1; u <= 7; u++)
                    s[u] = letters[o + u];
                break;
            }
        }
        if (o == 50 * 8) {
            ps = " ";
            o = 0;
        }
```

这段代码的主要目的是计算阶乘(阶数n)的阶数(阶数定义为从0到n的所有阶数的和)。阶乘是一个整数，可以用以下格式表示：n! = n*(n-1)*(n-2)*...*1*0。

在这段代码中，首先定义了一个变量o，并输出了一些字符。然后，判断变量o的值是否为0，如果是，就执行一系列打印操作；如果不是，就执行另外一些打印操作。

接下来，定义了一个变量ms，如果ms的值为"ALL"，就执行阶乘计算操作；否则，先将变量ms的值设为x，然后执行阶乘计算操作。

然后，定义了一个变量ps，如果变量ms的值为"ALL"，就执行阶乘计算操作；否则，执行另外一些打印操作。

接下来，定义了一个变量u，用于计数变量ms中所有奇数位置的阶数。

接着，定义了一个变量k，用于计算变量ms中所有奇数位置的阶数。

然后，定义了一个变量j，用于记录变量ms中每个阶数的个数。

接着，定义了一个变量f，用于记录变量ms中每个阶数的值。

然后，定义了一个变量g1，用于计算变量f中每个阶数的值。

接着，定义了一个变量h，用于打印输出。

然后，定义了一个变量s，用于存储变量ms的长度。

接着，定义了一个变量t1，用于打印输出。

然后，执行一些打印操作。

最后，执行一些打印操作。


```
//      print("Doing " + o + "\n");
        if (o == 0) {
            for (h = 1; h <= 7 * x; h++)
                print("\n");
        } else {
            xs = ms;
            if (ms == "ALL")
                xs = ps;
            for (u = 1; u <= 7; u++) {
                // An inefficient way of extracting bits
                // but good enough in BASIC because there
                // aren't bit shifting operators.
                for (k = 8; k >= 0; k--) {
                    if (Math.pow(2, k) >= s[u]) {
                        j[9 - k] = 0;
                    } else {
                        j[9 - k] = 1;
                        s[u] -= Math.pow(2, k);
                        if (s[u] == 1) {
                            f[u] = 9 - k;
                            break;
                        }
                    }
                }
                for (t1 = 1; t1 <= x; t1++) {
                    str = tab((63 - 4.5 * y) * g1 / xs.length + 1);
                    for (b = 1; b <= f[u]; b++) {
                        if (j[b] == 0) {
                            for (i = 1; i <= y; i++)
                                str += tab(xs.length);
                        } else {
                            for (i = 1; i <= y; i++)
                                str += xs;
                        }
                    }
                    print(str + "\n");
                }
            }
            for (h = 1; h <= 2 * x; h++)
                print("\n");
        }
    }
}

```

这道题是一个简单的C语言程序，包含一个名为“main”的函数。程序的主要目的是在控制台（通常是Windows或Linux系统）上打印“Hello World！”这个字符串。

“main”函数的作用是调用程序中的其他函数，这些函数将负责在屏幕上打印出字符串。具体来说，“main”函数会首先调用名为“print_hello”的函数，传递一个参数“世界”（在大多数系统中，这个字符串代表“Hello World”）。然后，“main”函数再调用名为“print_numbers”的函数，传递一个参数“123”。

在这个神秘的代码中，我们暂时没有找到任何其他函数。你可以根据需要自行添加，来查看代码的实际作用。但很明显，这个程序的主要目的是在屏幕上打印“Hello World！”和数字1、2、3。


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

Conversion to [Pascal](https://en.wikipedia.org/wiki/Pascal_(programming_language))


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


# `06_Banner/python/banner.py`

It looks like the output you provided is a JSON object containing a时间和一组数值。数值可能是一个密码，用于衡量从1到100的步长。

从数值的角度来看，这个密码似乎没有明显的规律。然而，如果你尝试使用某些数学算法对其进行解密，你可能会得到一些有趣的结果。例如，使用辗转相除法（如示意解法），你可以尝试找到一个不需要密码的解密方法。

但是，这里提到的所有内容都是错误的。我猜测这可能是由于输入的数据格式引起的。如果你能提供更多信息，例如数据的来源和格式，可能会帮助你找到更好的解决方案。



```
#!/usr/bin/env python3

"""
BANNER

Converted from BASIC to Python by Trevor Hobson
"""

letters = {
    " ": [0, 0, 0, 0, 0, 0, 0],
    "A": [505, 37, 35, 34, 35, 37, 505],
    "G": [125, 131, 258, 258, 290, 163, 101],
    "E": [512, 274, 274, 274, 274, 258, 258],
    "T": [2, 2, 2, 512, 2, 2, 2],
    "W": [256, 257, 129, 65, 129, 257, 256],
    "L": [512, 257, 257, 257, 257, 257, 257],
    "S": [69, 139, 274, 274, 274, 163, 69],
    "O": [125, 131, 258, 258, 258, 131, 125],
    "N": [512, 7, 9, 17, 33, 193, 512],
    "F": [512, 18, 18, 18, 18, 2, 2],
    "K": [512, 17, 17, 41, 69, 131, 258],
    "B": [512, 274, 274, 274, 274, 274, 239],
    "D": [512, 258, 258, 258, 258, 131, 125],
    "H": [512, 17, 17, 17, 17, 17, 512],
    "M": [512, 7, 13, 25, 13, 7, 512],
    "?": [5, 3, 2, 354, 18, 11, 5],
    "U": [128, 129, 257, 257, 257, 129, 128],
    "R": [512, 18, 18, 50, 82, 146, 271],
    "P": [512, 18, 18, 18, 18, 18, 15],
    "Q": [125, 131, 258, 258, 322, 131, 381],
    "Y": [8, 9, 17, 481, 17, 9, 8],
    "V": [64, 65, 129, 257, 129, 65, 64],
    "X": [388, 69, 41, 17, 41, 69, 388],
    "Z": [386, 322, 290, 274, 266, 262, 260],
    "I": [258, 258, 258, 512, 258, 258, 258],
    "C": [125, 131, 258, 258, 258, 131, 69],
    "J": [65, 129, 257, 257, 257, 129, 128],
    "1": [0, 0, 261, 259, 512, 257, 257],
    "2": [261, 387, 322, 290, 274, 267, 261],
    "*": [69, 41, 17, 512, 17, 41, 69],
    "3": [66, 130, 258, 274, 266, 150, 100],
    "4": [33, 49, 41, 37, 35, 512, 33],
    "5": [160, 274, 274, 274, 274, 274, 226],
    "6": [194, 291, 293, 297, 305, 289, 193],
    "7": [258, 130, 66, 34, 18, 10, 8],
    "8": [69, 171, 274, 274, 274, 171, 69],
    "9": [263, 138, 74, 42, 26, 10, 7],
    "=": [41, 41, 41, 41, 41, 41, 41],
    "!": [1, 1, 1, 384, 1, 1, 1],
    "0": [57, 69, 131, 258, 131, 69, 57],
    ".": [1, 1, 129, 449, 129, 1, 1],
}


```

It looks like you're trying to implement a simple game where the user can print different types of statements, such as numbers or characters. The user will have to enter a number or a character, and the program will try to help them by explaining what they mean, or what they are trying to do.

There are a few issues with the code that I noticed. For example, the program will only work for very large numbers or very small characters, because the program doesn't handle the cases where the user enters a number that doesn't exist, or a character that doesn't exist in the 'all' vocabulary.

Another issue is that the program will treat the character 'all' as if it were a valid character, rather than treating it as a command. This means that the user will have to type 'all' every time they want to enter a full statement, rather than typing 'all' followed by the character they want to enter.

I hope this helps! Let me know if you have any questions or if there's anything else I can do to help.


```
def print_banner() -> None:
    f = [0] * 7
    j = [0] * 9

    while True:
        try:
            horizontal = int(input("Horizontal "))
            if horizontal < 1:
                raise ValueError("Horizontal must be greater than zero")
            break

        except ValueError:
            print("Please enter a number greater than zero")
    while True:
        try:
            vertical = int(input("Vertical "))
            if vertical < 1:
                raise ValueError("Vertical must be greater than zero")
            break

        except ValueError:
            print("Please enter a number greater than zero")
    g1 = 0
    if input("Centered ").lower().startswith("y"):
        g1 = 1
    character = input(
        "Character (type 'ALL' if you want character being printed) "
    ).upper()
    statement = input("Statement ")

    input("Set page ")  # This means to prepare printer, just press Enter

    for statement_char in statement:
        s = letters[statement_char].copy()
        x_str = character
        if character == "ALL":
            x_str = statement_char
        if x_str == " ":
            print("\n" * (7 * horizontal))
        else:
            for u in range(0, 7):
                for k in range(8, -1, -1):
                    if 2**k >= s[u]:
                        j[8 - k] = 0
                    else:
                        j[8 - k] = 1
                        s[u] = s[u] - 2**k
                        if s[u] == 1:
                            f[u] = 8 - k
                            break
                for _t1 in range(1, horizontal + 1):
                    line_str = " " * int((63 - 4.5 * vertical) * g1 / len(x_str) + 1)
                    for b in range(0, f[u] + 1):
                        if j[b] == 0:
                            for _ in range(1, vertical + 1):
                                line_str = line_str + " " * len(x_str)
                        else:
                            line_str = line_str + x_str * vertical
                    print(line_str)
            print("\n" * (2 * horizontal - 1))
    # print("\n" * 75)  # Feed some more paper from the printer


```

这段代码是一个条件判断语句，它首先检查当前程序是否作为主程序运行，如果是，那么程序将调用一个名为 `print_banner` 的函数并输出一个警示消息。

具体来说，当程序作为主程序运行时，`__name__` 属性将会被赋予一个值为 `__main__` 的特殊属性。因此，如果当前程序作为主程序运行，那么 `if __name__ == "__main__":` 这一部分将永远为真，程序将直接跳转到调用 `print_banner` 的部分。如果当前程序不是主程序运行，那么 `__name__` 属性将为 `__main__`，但 `if __name__ == "__main__":` 这一部分将为假，程序将不会跳转到调用 `print_banner` 的部分。

在调用 `print_banner` 的函数之前，该函数自身的 `print_banner()` 部分将先执行。这个函数可能是定义在程序外的函数，用于在程序启动时输出一些警示信息，比如程序版本号、时间和日期等。


```
if __name__ == "__main__":
    print_banner()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Basketball

This program simulates a game of basketball between Dartmouth College and an opponent of your choice. You are the Dartmouth captain and control the type of shot and defense during the course of the game.

There are four types of shots:
1. Long Jump Shot (30ft)
2. Short Jump Shot (15ft)
3. Lay Up
4. Set Shot

Both teams use the same defense, but you may call it:
- Enter (6): Press
- Enter (6.5): Man-to-man
- Enter (7): Zone
- Enter (7.5): None

To change defense, type "0" as your next shot.

Note: The game is biased slightly in favor of Dartmouth. The average probability of a Dartmouth shot being good is 62.95% compared to a probability of 61.85% for their opponent. (This makes the sample run slightly remarkable in that Cornell won by a score of 45 to 42 Hooray for the Big Red!)

Charles Bacheller of Dartmouth College was the original author of this game.

---

As published in Basic Computer Games (1978)
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=12)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=27)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)

##### Original bugs

###### Initial defense selection

If a number <6 is entered for the starting defense then the original code prompts again until a value >=6 is entered,
but then skips the opponent selection center jump.

The C# port does not reproduce this behavior. It does prompt for a correct value, but will then go to opponent selection
followed by the center jump.

###### Unvalidated defense selection

The original code does not validate the value entered for the defense beyond checking that it is >=6. A large enough
defense value will guarantee that all shots are good, and the game gets rather predictable.

This bug is preserved in the C# port.


# `07_Basketball/csharp/Clock.cs`

这段代码定义了一个名为 Clock 的类，该类用于控制篮球比赛的时间计时。以下是代码的作用：

1. 定义了一个 IReadWrite 类型的成员变量 _io，表示与用户交互的部分，用于在比赛过程中向观众和裁判员显示比赛信息。
2. 定义了一个 int 类型的成员变量 time，表示比赛的当前时间，从 0 开始。
3. 定义了三个 bool 类型的成员变量：IsHalfTime，IsFullTime 和 TwoMinutesLeft，用于判断比赛是否处于半场或全场最后 10 分钟，以及是否还有 2 分钟不到全场结束。
4. 定义了一个 void 类型的成员函数 Increment，该函数用于递增时间并更新比赛状态，包括向观众和裁判员显示比赛信息。
5. 定义了一个 void 类型的成员函数 StartOvertime，该函数用于开始 overtime 比赛。




```
using Basketball.Resources;
using Games.Common.IO;

namespace Basketball;

internal class Clock
{
    private readonly IReadWrite _io;
    private int time;

    public Clock(IReadWrite io) => _io = io;

    public bool IsHalfTime => time == 50;
    public bool IsFullTime => time >= 100;
    public bool TwoMinutesLeft => time == 92;

    public void Increment(Scoreboard scoreboard)
    {
        time += 1;
        if (IsHalfTime) { scoreboard.Display(Resource.Formats.EndOfFirstHalf); }
        if (TwoMinutesLeft) { _io.Write(Resource.Streams.TwoMinutesLeft); }
    }

    public void StartOvertime() => time = 93;
}
```

# `07_Basketball/csharp/Defense.cs`

这段代码定义了一个名为 "Basketball" 的namespace，其中定义了一个名为 "Defense" 的内部类。

在这个内部类中，定义了一个名为 "value" 的私有成员变量，该变量初始化为一个float类型的值，即0.0f。

接着，定义了一个名为 "Set" 的方法，该方法接受一个float类型的参数，并将其设置为存储在 "value" 成员变量中的值。

最后，定义了一个名为 "float" 的类型，通过该类型将 "Defense" 类的 "value" 成员变量与 "float" 类型的变量进行隐式类型转换。

总的来说，这段代码创建了一个类 "Defense"，该类有一个私有成员变量 "value"，以及一个方法 "Set"，用于设置 "value" 的值。另外，还定义了一个可以通过 "float" 类型将 "Defense" 类的 "value" 成员变量与 "float" 类型的变量进行隐式类型转换的类型 "Defense"。


```
namespace Basketball;

internal class Defense
{
    private float _value;

    public Defense(float value) => Set(value);

    public void Set(float value) => _value = value;

    public static implicit operator float(Defense defense) => defense._value;
}

```

# `07_Basketball/csharp/Game.cs`

This is a Rust implementation of a simple game where two teams compete in a parkour race. The game uses the在高延迟的情况下依然能流畅运行的 NightC清晰文本渲染技术。

首先，游戏需要四个类来表示游戏中的四个组件：Game、Scoreboard、BallContest 和 Defense。Scoreboard 和 Defense 类用于跟踪比赛得分和防守。Game 和 BallContest 类负责游戏的主要逻辑。

Game 类表示游戏，它需要一个 Clock、Scoreboard 和 TextIO。Clock 类负责计时，Scoreboard 类负责存储比赛得分，TextIO 类负责将文本内容写入游戏界面。

Scoreboard 类表示比赛场地，它需要一个 TextIO。Scoreboard 类包含比赛队伍、比赛得分和比赛时间。

BallContest 类表示比赛的球，它包含一个浮点数类型的数据以表示球权，以及一个字符串类型的数据表示当前掌控球队。

Defense 类表示防守方，它包含一个字符串类型的数据，用于在比赛开始时设置防守。

最后，我们创建一个 Game 实例需要 Clock、Scoreboard 和 BallContest。这些类都初始化为默认值，然后我们就可以在球场上玩了。


```
using Basketball.Plays;
using Basketball.Resources;
using Games.Common.IO;
using Games.Common.Randomness;

namespace Basketball;

internal class Game
{
    private readonly Clock _clock;
    private readonly Scoreboard _scoreboard;
    private readonly TextIO _io;
    private readonly IRandom _random;

    private Game(Clock clock, Scoreboard scoreboard, TextIO io, IRandom random)
    {
        _clock = clock;
        _scoreboard = scoreboard;
        _io = io;
        _random = random;
    }

    public static Game Create(TextIO io, IRandom random)
    {
        io.Write(Resource.Streams.Introduction);

        var defense = new Defense(io.ReadDefense("Your starting defense will be"));
        var clock = new Clock(io);

        io.WriteLine();

        var scoreboard = new Scoreboard(
            new Team("Dartmouth", new HomeTeamPlay(io, random, clock, defense)),
            new Team(io.ReadString("Choose your opponent"), new VisitingTeamPlay(io, random, clock, defense)),
            io);

        return new Game(clock, scoreboard, io, random);
    }

    public void Play()
    {
        var ballContest = new BallContest(0.4f, "{0} controls the tap", _io, _random);

        while (true)
        {
            _io.WriteLine("Center jump");
            ballContest.Resolve(_scoreboard);

            _io.WriteLine();

            while (true)
            {
                var isFullTime = _scoreboard.Offense.ResolvePlay(_scoreboard);
                if (isFullTime && IsGameOver()) { return; }
                if (_clock.IsHalfTime) { break; }
            }
        }
    }

    private bool IsGameOver()
    {
        _io.WriteLine();
        if (_scoreboard.ScoresAreEqual)
        {
            _scoreboard.Display(Resource.Formats.EndOfSecondHalf);
            _clock.StartOvertime();
            return false;
        }

        _scoreboard.Display(Resource.Formats.EndOfGame);
        return true;
    }
}

```

# `07_Basketball/csharp/IRandomExtensions.cs`

这段代码定义了一个内部类 `IRandomExtensions`，用于扩展 `IRandom` 类用于生成篮球比赛中的一种 shots。

`NextShot` 方法使用 `random.NextFloat(1, 3.5f)` 从 1 到 3.5 之间的随机浮点数，然后从弯月函数中选取一个随机数，这个随机数将被用来生成一个 shot。

这段代码的作用是生成一个篮球比赛中的一种随机 shot，根据不同的 shot 类型，生成的随机数可能会有所不同。


```
using Games.Common.Randomness;

namespace Basketball;

internal static class IRandomExtensions
{
    internal static Shot NextShot(this IRandom random) => Shot.Get(random.NextFloat(1, 3.5f));
}

```

# `07_Basketball/csharp/IReadWriteExtensions.cs`



这段代码是一个用于篮球游戏中的类，包含了几个用于从控制台读取输入的函数，以及一个用于读取进攻和防守得分的类。

首先，`ReadDefense`函数使用一个`while`循环来读取玩家输入的防守分数。在每次循环中，它使用`io.ReadNumber`函数从控制台读取一个浮点数，并将其存储在名为`defense`的变量中。该函数会在浮点数大于或等于6时返回该浮点数，否则会继续循环读取输入，直到正确答案被读取为止。

其次，`TryReadInteger`函数使用一个`while`循环来尝试从控制台读取一个整数。在每次循环中，它使用`io.ReadNumber`函数从控制台读取一个浮点数，并将其存储在名为`floatValue`的变量中。然后，它使用一个if语句来检查是否读取到了一个整数。如果是整数，则将其转换为整数类型，并将其存储在名为`intValue`的变量中。如果读取到了一个浮点数，则会执行if语句的判断部分，即`intValue == floatValue`。

最后，`ReadShot`函数使用一个`while`循环来读取玩家输入的进攻或防守得分。在每次循环中，它尝试使用`io.TryReadInteger`函数从控制台读取一个整数。如果是整数，则使用该整数作为进攻得分或防守得分，并返回该进攻或防守得分。否则，它会向玩家提供错误消息并继续循环读取输入。


```
using Games.Common.IO;

namespace Basketball;

internal static class IReadWriteExtensions
{
    public static float ReadDefense(this IReadWrite io, string prompt)
    {
        while (true)
        {
            var defense = io.ReadNumber(prompt);
            if (defense >= 6) { return defense; }
        }
    }

    private static bool TryReadInteger(this IReadWrite io, string prompt, out int intValue)
    {
        var floatValue = io.ReadNumber(prompt);
        intValue = (int)floatValue;
        return intValue == floatValue;
    }

    public static Shot? ReadShot(this IReadWrite io, string prompt)
    {
        while (true)
        {
            if (io.TryReadInteger(prompt, out var value) && Shot.TryGet(value, out var shot))
            {
                return shot;
            }
            io.Write("Incorrect answer.  Retype it. ");
        }
    }
}

```

# `07_Basketball/csharp/JumpShot.cs`



这段代码定义了一个名为 "JumpShot" 的类，继承自 "Shot" 类，属于 "Basketball" 命名空间。

在 "JumpShot" 类中，有一个构造函数，名为 "JumpShot()"，其中有一个参数 "base("", "Jump shot")"，表示从 "Shot" 类的 "base" 字段中继承一个名为 "Jump shot" 的字符串，作为构造函数的参数。

另一个方法 "getcave" 获取了一个对象 "obj2"，但是没有具体的实现，只是返回了该对象的引用 "obj2"。

最后，在 "JumpShot" 类中，定义了一个 "time" 变量，未对其进行初始化，其值为 0。


```
namespace Basketball;

public class JumpShot : Shot
{
    public JumpShot()
        : base("Jump shot")
    {
    }
}
```

# `07_Basketball/csharp/Probably.cs`

这段代码定义了一个名为 Probably 的内部结构体，用于表示一个类游戏中的行动概率决策。

Probably 结构体包含三个成员变量：

1. defenseFactor：表示一个防御因素，用于影响后续决策的概率。
2. random：表示一个用于生成随机数的类 Randomness 的实例。
3. result：表示一个布尔值，表示是否成功执行行动。

Probably 结构体还包含三个方法：

1. Do(float probability, Action action)：根据给定的概率执行的动作。如果概率为真，则执行随机动作，否则返回本身。
2. Or(float probability, Action action)：根据给定概率的两个动作之一执行。
3. Or(float probability, Func<bool> action)：根据给定概率的两个动作之一执行。
4. Evaluate(Action action)：根据给定概率计算动作的概率。

在 Evaluate 方法中，使用 ShouldResolveAction 这个方法判断是否执行动作，如果概率为真，则执行随机动作，否则返回 false。

另外，还包含 ShouldResolveAction 和 Resolve 方法，分别用于在给定概率时判断是否执行动作并生成随机数。


```
using Games.Common.Randomness;

namespace Basketball;

/// <summary>
/// Supports a chain of actions to be performed based on various probabilities. The original game code gets a new
/// random number for each probability check. Evaluating a set of probabilities against a single random number is
/// much simpler, but yield a very different outcome distribution. The purpose of this class is to simplify the code
/// to for the original probabilistic branch decisions.
/// </summary>
internal struct Probably
{
    private readonly float _defenseFactor;
    private readonly IRandom _random;
    private readonly bool? _result;

    internal Probably(float defenseFactor, IRandom random, bool? result = null)
    {
        _defenseFactor = defenseFactor;
        _random = random;
        _result = result;
    }

    public Probably Do(float probability, Action action) =>
        ShouldResolveAction(probability)
            ? new Probably(_defenseFactor, _random, Resolve(action) ?? false)
            : this;

    public Probably Do(float probability, Func<bool> action) =>
        ShouldResolveAction(probability)
            ? new Probably(_defenseFactor, _random, Resolve(action) ?? false)
            : this;

    public Probably Or(float probability, Action action) => Do(probability, action);

    public Probably Or(float probability, Func<bool> action) => Do(probability, action);

    public bool Or(Action action) => _result ?? Resolve(action) ?? false;

    private bool? Resolve(Action action)
    {
        action.Invoke();
        return _result;
    }

    private bool? Resolve(Func<bool> action) => action.Invoke();

    private readonly bool ShouldResolveAction(float probability) =>
        _result is null && _random.NextFloat() <= probability * _defenseFactor;
}

```

# `07_Basketball/csharp/Program.cs`

这段代码的作用是创建一个篮球游戏实例，并调用其的Play()方法来开始游戏。

具体来说，代码中使用了三个外部库：Basketball、Games.Common.IO和Games.Common.Randomness。Basketball库提供了篮球游戏的规则和功能，Games.Common.IO库提供了输入输出流操作，而Games.Common.Randomness库提供了随机数生成器。

此外，游戏实例还通过构造函数接收到了一个ConsoleIO实例和一个RandomNumberGenerator实例。ConsoleIO实例用于输出游戏过程中的信息和信息，RandomNumberGenerator实例用于生成随机数，这些都有助于在游戏中显示和处理数据。

最后，game.Play()方法开始游戏，并调用game instance自身的Play()方法来处理所有的游戏逻辑。这个方法包括投篮、进攻和防守操作，以及比赛过程中的其他事件。


```
using Basketball;
using Games.Common.IO;
using Games.Common.Randomness;

var game = Game.Create(new ConsoleIO(), new RandomNumberGenerator());

game.Play();
```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `07_Basketball/csharp/Scoreboard.cs`

这段代码定义了一个Scoreboard类，用于在篮球比赛中记录得分。它包括以下属性和方法：

1. 一个Dictionary<Team, uint> _scores，用于存储每个队伍的得分。
2. 一个IReadWrite _io，用于写入和读取数据到文件。
3. 一个Scoreboard类构造函数，初始化得分板、主队和 visitors。
4. AddBasket方法，用于添加一个篮球得分，并输出一条信息到文件。
5. AddFreeThrows方法，用于添加罚球得分，并输出一条信息到文件。
6. Turnover方法，用于在得分板上添加或删除一个队伍，并输出一条信息到文件。
7. Display方法，用于输出得分板上的数据，并可以指定格式。

整个Scoreboard类的作用是，在篮球比赛中记录得分，并可以将数据保存到文件中，以便在以后的游戏中可以重新加载数据。


```
using Basketball.Resources;
using Games.Common.IO;

namespace Basketball;

internal class Scoreboard
{
    private readonly Dictionary<Team, uint> _scores;
    private readonly IReadWrite _io;

    public Scoreboard(Team home, Team visitors, IReadWrite io)
    {
        _scores = new() { [home] = 0, [visitors] = 0 };
        Home = home;
        Visitors = visitors;
        Offense = home;  // temporary value till first center jump
        _io = io;
    }

    public bool ScoresAreEqual => _scores[Home] == _scores[Visitors];
    public Team Offense { get; set; }
    public Team Home { get; }
    public Team Visitors { get; }

    public void AddBasket(string message) => AddScore(2, message);

    public void AddFreeThrows(uint count, string message) => AddScore(count, message);

    private void AddScore(uint score, string message)
    {
        if (Offense is null) { throw new InvalidOperationException("Offense must be set before adding to score."); }

        _io.WriteLine(message);
        _scores[Offense] += score;
        Turnover();
        Display();
    }

    public void Turnover(string? message = null)
    {
        if (message is not null) { _io.WriteLine(message); }

        Offense = Offense == Home ? Visitors : Home;
    }

    public void Display(string? format = null) =>
        _io.WriteLine(format ?? Resource.Formats.Score, Home, _scores[Home], Visitors, _scores[Visitors]);
}

```

# `07_Basketball/csharp/Shot.cs`

这段代码定义了一个名为“Shot”的类，其用于表示篮球比赛中的一种得分方式——“投篮”。

在“Shot”类中，定义了一个名为“_name”的私有成员变量，用于存储该 shot 的名称(即 shot 的名称)，并在构造函数中进行了初始化。

此外，还定义了一个名为“TryGet”的公共方法，用于检查给定的 shot 编号是否存在于“Shot”类中，并返回其实例。其中，“shotNumber”参数表示要检查的 shot 编号，“out”参数表示是否返回符合条件的实例。“在方法中，使用 switch 语句来检查 shotNumber 是否为 0，如果是，则返回 null。否则，根据 shotNumber 编号查找相应的 shot 类实例，并将其返回。”

最后，还定义了一个名为“Get”的静态方法，用于根据给定的 shot 编号返回相应的 shot 类实例。该方法同样使用 switch 语句来查找相应的 shot 类实例。

该代码的主要作用是创建一个 Shot 类，用于表示篮球比赛中的得分方式——投篮，以及相关的辅助方法。通过 TryGet 和 Get 方法，实现了根据 shot编号返回相应的 shot 类实例的功能。


```
namespace Basketball;

public class Shot
{
    private readonly string _name;

    public Shot(string name)
    {
        _name = name;
    }

    public static bool TryGet(int shotNumber, out Shot? shot)
    {
        shot = shotNumber switch
        {
            // Although the game instructions reference two different jump shots,
            // the original game code treats them both the same and just prints "Jump shot"
            0 => null,
            <= 2 => new JumpShot(),
            3 => new Shot("Lay up"),
            4 => new Shot("Set shot"),
            _ => null
        };
        return shotNumber == 0 || shot is not null;
    }

    public static Shot Get(float shotNumber) =>
        shotNumber switch
        {
            <= 2 => new JumpShot(),
            > 3 => new Shot("Set shot"),
            > 2 => new Shot("Lay up"),
            _ => throw new Exception("Unexpected value")
        };

    public override string ToString() => _name;
}

```

# `07_Basketball/csharp/Team.cs`

这段代码定义了一个名为Basketball的命名空间，其中包含一个名为Team的内部记录类，其构造函数为字符串类型的Name和类型为Play的内部记录类，该类继承自AnimalPlay。

在定义的内部记录类Team中，有一个名为ResolvePlay的计算方法，它返回一个布尔值，代表在给定的Scoreboard对象中是否能够成功解析篮球比赛中的 plays。

此外，还定义了一个名为Basketball的命名空间，其中包含一个名为Plays的内部接口，代表篮球比赛中的 plays。


```
using Basketball.Plays;

namespace Basketball;

internal record Team(string Name, Play PlayResolver)
{
    public override string ToString() => Name;

    public bool ResolvePlay(Scoreboard scoreboard) => PlayResolver.Resolve(scoreboard);
}

```

# `07_Basketball/csharp/Plays/BallContest.cs`

这段代码定义了一个名为BallContest的类，它包含以下几种主要方法：

1. BallContest类有一个构造函数，它接受四个参数：概率(float)、消息格式字符串(string)、输入输出流对象(IReadWrite)和随机数生成器(IRandom)。
2. Resolve方法，该方法接受一个Scoreboard对象作为参数，然后根据随机数生成器生成随机数，并通过Resolve方法设置胜者并输出消息。

总的来说，这段代码定义了一个用于篮球比赛中的球权争夺球的随机事件 BallContest。这个类可以让玩家随机选择进攻或防守，并在每次比赛结束后输出指定的消息。


```
using Games.Common.IO;
using Games.Common.Randomness;

namespace Basketball.Plays;

internal class BallContest
{
    private readonly float _probability;
    private readonly string _messageFormat;
    private readonly IReadWrite _io;
    private readonly IRandom _random;

    internal BallContest(float probability, string messageFormat, IReadWrite io, IRandom random)
    {
        _io = io;
        _probability = probability;
        _messageFormat = messageFormat;
        _random = random;
    }

    internal bool Resolve(Scoreboard scoreboard)
    {
        var winner = _random.NextFloat() <= _probability ? scoreboard.Home : scoreboard.Visitors;
        scoreboard.Offense = winner;
        _io.WriteLine(_messageFormat, winner);
        return false;
    }
}

```

# `07_Basketball/csharp/Plays/HomeTeamPlay.cs`

This is a class written in C# that implements the AI for a 3-point shot in a basketball game. It uses a combination of random number generation and algorithms to determine the outcome of the shot, such as whether it was successful or missed, and if the ball was turned over or retrieved by the defense.

The class has several methods for resolving different outcomes of the shot, including:

* ResolveShotOffTarget: This method resolves the outcome of the shot based on whether it was successful or missed, and if the ball was turned over or retrieved by the defense. It uses a combination of the defensive strength of the visiting team and the randomness of the shot to determine the outcome.
* ResolveHomeRebound: This method resolves the outcome of the shot based on whether it was successful or missed, and if the ball was turned over or retrieved by the defense. It uses a combination of the defensive strength of the home team and the randomness of the shot to determine the outcome.
* ResolveShotOffTheRim: This method resolves the outcome of the shot based on whether it was successful or missed. It uses a combination of the randomness of the shot and the defensive strength of the visiting team to determine the outcome.

The class also has a method for resolving the outcome of the shot even if it is的一部分 of a free throw attempt, such as a two-point free throw attempt or a three-point free throw attempt.


```
using Games.Common.IO;
using Games.Common.Randomness;

namespace Basketball.Plays;

internal class HomeTeamPlay : Play
{
    private readonly TextIO _io;
    private readonly IRandom _random;
    private readonly Clock _clock;
    private readonly Defense _defense;
    private readonly BallContest _ballContest;

    public HomeTeamPlay(TextIO io, IRandom random, Clock clock, Defense defense)
        : base(io, random, clock)
    {
        _io = io;
        _random = random;
        _clock = clock;
        _defense = defense;
        _ballContest = new BallContest(0.5f, "Shot is blocked.  Ball controlled by {0}.", _io, _random);
    }

    internal override bool Resolve(Scoreboard scoreboard)
    {
        var shot = _io.ReadShot("Your shot");

        if (_random.NextFloat() >= 0.5f && _clock.IsFullTime) { return true; }

        if (shot is null)
        {
            _defense.Set(_io.ReadDefense("Your new defensive alignment is"));
            _io.WriteLine();
            return false;
        }

        if (shot is JumpShot jumpShot)
        {
            if (ClockIncrementsToHalfTime(scoreboard)) { return false; }
            if (!Resolve(jumpShot, scoreboard)) { return false; }
        }

        do
        {
            if (ClockIncrementsToHalfTime(scoreboard)) { return false; }
        } while (Resolve(shot, scoreboard));

        return false;
    }

    // The Resolve* methods resolve the probabilistic outcome of the current game state.
    // They return true if the Home team should continue the play and attempt a layup, false otherwise.
    private bool Resolve(JumpShot shot, Scoreboard scoreboard) =>
        Resolve(shot.ToString(), _defense / 8)
            .Do(0.341f, () => scoreboard.AddBasket("Shot is good"))
            .Or(0.682f, () => ResolveShotOffTarget(scoreboard))
            .Or(0.782f, () => _ballContest.Resolve(scoreboard))
            .Or(0.843f, () => ResolveFreeThrows(scoreboard, "Shooter is fouled.  Two shots."))
            .Or(() => scoreboard.Turnover($"Charging foul.  {scoreboard.Home} loses ball."));

    private bool Resolve(Shot shot, Scoreboard scoreboard) =>
        Resolve(shot.ToString(), _defense / 7)
            .Do(0.4f, () => scoreboard.AddBasket("Shot is good.  Two points."))
            .Or(0.7f, () => ResolveShotOffTheRim(scoreboard))
            .Or(0.875f, () => ResolveFreeThrows(scoreboard, "Shooter fouled.  Two shots."))
            .Or(0.925f, () => scoreboard.Turnover($"Shot blocked. {scoreboard.Visitors}'s ball."))
            .Or(() => scoreboard.Turnover($"Charging foul.  {scoreboard.Home} loses ball."));

    private bool ResolveShotOffTarget(Scoreboard scoreboard) =>
        Resolve("Shot is off target", 6 / _defense)
            .Do(0.45f, () => ResolveHomeRebound(scoreboard, ResolvePossibleSteal))
            .Or(() => scoreboard.Turnover($"Rebound to {scoreboard.Visitors}"));

    private bool ResolveHomeRebound(Scoreboard scoreboard, Action<Scoreboard> endOfPlayAction) =>
        Resolve($"{scoreboard.Home} controls the rebound.")
            .Do(0.4f, () => true)
            .Or(() => endOfPlayAction.Invoke(scoreboard));
    private void ResolvePossibleSteal(Scoreboard scoreboard)
    {
        if (_defense == 6 && _random.NextFloat() > 0.6f)
        {
            scoreboard.Turnover();
            scoreboard.AddBasket($"Pass stolen by {scoreboard.Visitors} easy layup.");
            _io.WriteLine();
        }
        _io.Write("Ball passed back to you. ");
    }

    private void ResolveShotOffTheRim(Scoreboard scoreboard) =>
        Resolve("Shot is off the rim.")
            .Do(2 / 3f, () => scoreboard.Turnover($"{scoreboard.Visitors} controls the rebound."))
            .Or(() => ResolveHomeRebound(scoreboard, _ => _io.WriteLine("Ball passed back to you.")));
}

```

# `07_Basketball/csharp/Plays/Play.cs`



这段代码定义了一个名为Play的内部类，代表篮球比赛中的一个 plays。Play 由三个私有属性和一个公有方法组成：

1. IReadWrite：用于读写文件的操作。
2. IRandom：用于生成随机数的类。
3. Clock：用于记录当前时间的类。

Play 类接受三个参数：IReadWrite、IRandom 和 Clock。这些参数分别用于读写文件、生成随机数和记录比赛时间。

在内部方法中， ClockIncrementsToHalfTime 方法用于更新当前时间，如果当前时间距离 half time 还有很远，就增加时间进度，否则就减少时间进度。

Resolve 方法是一个 abstract 方法，没有具体的操作，只是返回一个大概的答案。具体实现需要根据不同的 plays 来决定。

ResolveFreeThrows 方法用于处理罚球。首先根据传入的防守因子来决定罚球次数，然后根据不同的罚球方式和命中率来决定是否罚球和如何计算得分。

Resolve 方法的实现方式和具体情况需要在具体用例中进行。


```
using Games.Common.IO;
using Games.Common.Randomness;

namespace Basketball.Plays;

internal abstract class Play
{
    private readonly IReadWrite _io;
    private readonly IRandom _random;
    private readonly Clock _clock;

    public Play(IReadWrite io, IRandom random, Clock clock)
    {
        _io = io;
        _random = random;
        _clock = clock;
    }

    protected bool ClockIncrementsToHalfTime(Scoreboard scoreboard)
    {
        _clock.Increment(scoreboard);
        return _clock.IsHalfTime;
    }

    internal abstract bool Resolve(Scoreboard scoreboard);

    protected void ResolveFreeThrows(Scoreboard scoreboard, string message) =>
        Resolve(message)
            .Do(0.49f, () => scoreboard.AddFreeThrows(2, "Shooter makes both shots."))
            .Or(0.75f, () => scoreboard.AddFreeThrows(1, "Shooter makes one shot and misses one."))
            .Or(() => scoreboard.AddFreeThrows(0, "Both shots missed."));

    protected Probably Resolve(string message) => Resolve(message, 1f);

    protected Probably Resolve(string message, float defenseFactor)
    {
        _io.WriteLine(message);
        return new Probably(defenseFactor, _random);
    }
}

```

# `07_Basketball/csharp/Plays/VisitingTeamPlay.cs`

This appears to be a implementation of a basketball shot clock. It appears to maintain a scoreboard that displays the current score, and allows the player to take a shot, but also allows the visitor to take a shot if they are within certain parameters (e.g. within 5 feet of the basket). It also appears to have a defense factor that adjusts the score based on how well the visitor performs.

It looks like the shot is得分取决于投篮的出手速度和防守的难度。得分越高，出手速度越慢，得分越低。出手速度越快，得分越高，得分越低。看起来得分高的出手速度较慢，得分低的出手速度较快。

此外，根据得分板上的得分和进攻或防守，进攻或防守者可能会获得额外的得分或篮板。

我不知道这个程序的详细信息，因此我无法判断其是否更具体或是否提供了更多功能。


```
using Games.Common.IO;
using Games.Common.Randomness;

namespace Basketball.Plays;

internal class VisitingTeamPlay : Play
{
    private readonly TextIO _io;
    private readonly IRandom _random;
    private readonly Defense _defense;

    public VisitingTeamPlay(TextIO io, IRandom random, Clock clock, Defense defense)
        : base(io, random, clock)
    {
        _io = io;
        _random = random;
        _defense = defense;
    }

    internal override bool Resolve(Scoreboard scoreboard)
    {
        if (ClockIncrementsToHalfTime(scoreboard)) { return false; }

        _io.WriteLine();
        var shot = _random.NextShot();

        if (shot is JumpShot jumpShot)
        {
            var continuePlay = Resolve(jumpShot, scoreboard);
            _io.WriteLine();
            if (!continuePlay) { return false; }
        }

        while (true)
        {
            var continuePlay = Resolve(shot, scoreboard);
            _io.WriteLine();
            if (!continuePlay) { return false; }
        }
    }

    // The Resolve* methods resolve the probabilistic outcome of the current game state.
    // They return true if the Visiting team should continue the play and attempt a layup, false otherwise.
    private bool Resolve(JumpShot shot, Scoreboard scoreboard) =>
        Resolve(shot.ToString(), _defense / 8)
            .Do(0.35f, () => scoreboard.AddBasket("Shot is good."))
            .Or(0.75f, () => ResolveBadShot(scoreboard, "Shot is off the rim.", _defense * 6))
            .Or(0.9f, () => ResolveFreeThrows(scoreboard, "Player fouled.  Two shots."))
            .Or(() => _io.WriteLine($"Offensive foul.  {scoreboard.Home}'s ball."));

    private bool Resolve(Shot shot, Scoreboard scoreboard) =>
        Resolve(shot.ToString(), _defense / 7)
            .Do(0.413f, () => scoreboard.AddBasket("Shot is good."))
            .Or(() => ResolveBadShot(scoreboard, "Shot is missed.", 6 / _defense));

    private bool ResolveBadShot(Scoreboard scoreboard, string message, float defenseFactor) =>
        Resolve(message, defenseFactor)
            .Do(0.5f, () => scoreboard.Turnover($"{scoreboard.Home} controls the rebound."))
            .Or(() => ResolveVisitorsRebound(scoreboard));

    private bool ResolveVisitorsRebound(Scoreboard scoreboard)
    {
        _io.Write($"{scoreboard.Visitors} controls the rebound.");
        if (_defense == 6 && _random.NextFloat() <= 0.25f)
        {
            _io.WriteLine();
            scoreboard.Turnover();
            scoreboard.AddBasket($"Ball stolen.  Easy lay up for {scoreboard.Home}.");
            return false;
        }

        if (_random.NextFloat() <= 0.5f)
        {
            _io.WriteLine();
            _io.Write($"Pass back to {scoreboard.Visitors} guard.");
            return false;
        }

        return true;
    }
}

```

# `07_Basketball/csharp/Resources/Resource.cs`



这段代码是一个自定义的资源类，用于在篮球游戏中生成一些文本和样例数据。

具体来说，代码中定义了一个名为Streams的内部类，其中包含两个方法，一个是 Introduction，另一个是 TwoMinutesLeft，它们都是public static Stream的方法。

接着，定义了一个名为Formats的内部类，其中包含四个方法，分别是EndOfFirstHalf、EndOfGame、EndOfSecondHalf和Score，它们也都是public static string的方法。

接着，定义了一个名为GetString的内部方法，该方法接收一个字符串参数，并从指定的资源文件中读取相应的文本内容。

最后，通过调用GetStream和GetString方法，可以方便地在程序中使用这些资源。


```
using System.Reflection;
using System.Runtime.CompilerServices;

namespace Basketball.Resources;

internal static class Resource
{
    internal static class Streams
    {
        public static Stream Introduction => GetStream();
        public static Stream TwoMinutesLeft => GetStream();
    }

    internal static class Formats
    {
        public static string EndOfFirstHalf => GetString();
        public static string EndOfGame => GetString();
        public static string EndOfSecondHalf => GetString();
        public static string Score => GetString();
    }

    private static string GetString([CallerMemberName] string? name = null)
    {
        using var stream = GetStream(name);
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }

    private static Stream GetStream([CallerMemberName] string? name = null) =>
        Assembly.GetExecutingAssembly().GetManifestResourceStream($"Basketball.Resources.{name}.txt")
            ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
}
```

# `07_Basketball/java/Basketball.java`

This is a simple simulation of a basketball game. It simulates an opponents possession based on the type of shot (jump shot or lay up) and the opponent's chance of a successful defensive maneuver (2/4 chance). The main logic for the game is that the进攻方's AI Simulates an opponents Possession based on the opponent's Possibility.

There are several functions in this class such as Dartmouth_ball, opponent\_ball, opponent\_chance, and opprotunate. The Dartmouth\_ball function simulates the offensive player's Possession based on the possibility that the opponent will commit a defensive maneuver. The opponent\_ball function simulates the offensive player's Possession based on the possibility that the opponent will commit a defensive maneuver and based on the opponent's Possibility. The opponent\_chance function returns the random number between 1 and 10/4 that the opponent has.

The opprotunate function is the main entry point for the game, it is used to determine the outcome of the possession. It takes two arguments, one is the offensive player and the other is the defensive player. It returns the outcome of the possession, either a successful lay up or a missed jump shot.

This is just a simple example, in a real game the algorithm would be much more complex and include more factors such as the player's statistics, the time of the game, and the crowd.


```
import java.lang.Math;
import java.util.*;
import java.util.Scanner;

/* The basketball class is a computer game that allows you to play as
  Dartmouth College's captain and playmaker
  The game uses set probabilites to simulate outcomes of each posession
  You are able to choose your shot types as well as defensive formations */

public class Basketball {
    int time = 0;
    int[] score = {0, 0};
    double defense = -1;
    List<Double> defense_choices = Arrays.asList(6.0, 6.5, 7.0, 7.5);
    int shot = -1;
    List<Integer> shot_choices = Arrays.asList(0, 1, 2, 3, 4);
    double opponent_chance = 0;
    String opponent = null;

    public Basketball() {

        // Explains the keyboard inputs
        System.out.println("\t\t\t Basketball");
        System.out.println("\t Creative Computing  Morristown, New Jersey\n\n\n");
        System.out.println("This is Dartmouth College basketball. ");
        System.out.println("Υou will be Dartmouth captain and playmaker.");
        System.out.println("Call shots as follows:");
        System.out.println("1. Long (30ft.) Jump Shot; 2. Short (15 ft.) Jump Shot; "
              + "3. Lay up; 4. Set Shot");
        System.out.println("Both teams will use the same defense. Call Defense as follows:");
        System.out.println("6. Press; 6.5 Man-to-Man; 7. Zone; 7.5 None.");
        System.out.println("To change defense, just type 0 as your next shot.");
        System.out.print("Your starting defense will be? ");

        Scanner scanner = new Scanner(System.in); // creates a scanner

        // takes input for a defense
        if (scanner.hasNextDouble()) {
            defense = scanner.nextDouble();
        }
        else {
            scanner.next();
        }

        // makes sure that input is legal
        while (!defense_choices.contains(defense)) {
            System.out.print("Your new defensive allignment is? ");
            if (scanner.hasNextDouble()) {
                defense = scanner.nextDouble();
            }
            else {
                scanner.next();
                continue;
            }
        }

        // takes input for opponent's name
        System.out.print("\nChoose your opponent? ");

        opponent = scanner.next();
        start_of_period();
    }

    // adds points to the score
    // team can take 0 or 1, for opponent or Dartmouth, respectively
    private void add_points(int team, int points) {
        score[team] += points;
        print_score();
    }


    private void ball_passed_back() {
        System.out.print("Ball passed back to you. ");
        dartmouth_ball();
    }

    // change defense, called when the user enters 0 for their shot
    private void change_defense() {
        defense = -1;
        Scanner scanner = new Scanner(System.in); // creates a scanner

        while (!defense_choices.contains(defense)) {
            System.out.println("Your new defensive allignment is? ");
            if (scanner.hasNextDouble()) {
                defense = (double)(scanner.nextDouble());
            }
            else {
                continue;
            }
        }

        dartmouth_ball();
    }

    // simulates two foul shots for a player and adds the points
    private void foul_shots(int team) {
        System.out.println("Shooter fouled.  Two shots.");

        if (Math.random() > .49) {
            if (Math.random() > .75) {
                System.out.println("Both shots missed.");
            }
            else {
                System.out.println("Shooter makes one shot and misses one.");
                score[team] += 1;
            }
        }
        else {
            System.out.println("Shooter makes both shots.");
            score[team] += 2;
        }

        print_score();
    }

    // called when time = 50, starts a new period
    private void halftime() {
        System.out.println("\n   ***** End of first half *****\n");
        print_score();
        start_of_period();
    }

    // prints the current score
    private void print_score() {
        System.out.println("Score:  " + score[1] + " to " + score[0] + "\n");
    }

    // simulates a center jump for posession at the beginning of a period
    private void start_of_period() {
        System.out.println("Center jump");
        if (Math.random() > .6) {
            System.out.println("Dartmouth controls the tap.\n");
            dartmouth_ball();
        }
        else {
            System.out.println(opponent + " controls the tap.\n");
            opponent_ball();
        }
    }

    // called when t = 92
    private void two_minute_warning() {
        System.out.println("   *** Two minutes left in the game ***");
    }

    // called when the user enters 1 or 2 for their shot
    private void dartmouth_jump_shot() {
        time ++;
        if (time == 50) {
            halftime();
        }
        else if (time == 92) {
            two_minute_warning();
        }

        System.out.println("Jump Shot.");
        // simulates chances of different possible outcomes
        if (Math.random() > .341 * defense / 8) {
            if (Math.random() > .682 * defense / 8) {
                if (Math.random() > .782 * defense / 8) {
                    if (Math.random() > .843 * defense / 8) {
                        System.out.println("Charging foul. Dartmouth loses ball.\n");
                        opponent_ball();
                    }
                    else {
                        // player is fouled
                        foul_shots(1);
                        opponent_ball();
                    }
                }
                else {
                    if (Math.random() > .5) {
                        System.out.println("Shot is blocked. Ball controlled by " +
                              opponent + ".\n");
                        opponent_ball();
                    }
                    else {
                        System.out.println("Shot is blocked. Ball controlled by Dartmouth.");
                        dartmouth_ball();
                    }
                }
            }
            else {
                System.out.println("Shot is off target.");
                if (defense / 6 * Math.random() > .45) {
                    System.out.println("Rebound to " + opponent + "\n");
                    opponent_ball();
                }
                else {
                    System.out.println("Dartmouth controls the rebound.");
                    if (Math.random() > .4) {
                        if (defense == 6 && Math.random() > .6) {
                            System.out.println("Pass stolen by " + opponent
                                  + ", easy lay up");
                            add_points(0, 2);
                            dartmouth_ball();
                        }
                        else {
                            // ball is passed back to you
                            ball_passed_back();
                        }
                    }
                    else {
                        System.out.println("");
                        dartmouth_non_jump_shot();
                    }
                }
            }
        }
        else {
            System.out.println("Shot is good.");
            add_points(1, 2);
            opponent_ball();
        }
    }

    // called when the user enters 0, 3, or 4
    // lay up, set shot, or defense change
    private void dartmouth_non_jump_shot() {
        time ++;
        if (time == 50) {
            halftime();
        }
        else if (time == 92) {
            two_minute_warning();
        }

        if (shot == 4) {
            System.out.println("Set shot.");
        }
        else if (shot == 3) {
            System.out.println("Lay up.");
        }
        else if (shot == 0) {
            change_defense();
        }

        // simulates different outcomes after a lay up or set shot
        if (7/defense*Math.random() > .4) {
            if (7/defense*Math.random() > .7) {
                if (7/defense*Math.random() > .875) {
                    if (7/defense*Math.random() > .925) {
                        System.out.println("Charging foul. Dartmouth loses the ball.\n");
                        opponent_ball();
                    }
                    else {
                        System.out.println("Shot blocked. " + opponent + "'s ball.\n");
                        opponent_ball();
                    }
                }
                else {
                    foul_shots(1);
                    opponent_ball();
                }
            }
            else {
                System.out.println("Shot is off the rim.");
                if (Math.random() > 2/3) {
                    System.out.println("Dartmouth controls the rebound.");
                    if (Math.random() > .4) {
                        System.out.println("Ball passed back to you.\n");
                        dartmouth_ball();
                    }
                    else {
                        dartmouth_non_jump_shot();
                    }
                }
                else {
                    System.out.println(opponent + " controls the rebound.\n");
                    opponent_ball();
                }
            }
        }
        else {
            System.out.println("Shot is good. Two points.");
            add_points(1, 2);
            opponent_ball();
        }
    }


    // plays out a Dartmouth posession, starting with your choice of shot
    private void dartmouth_ball() {
        Scanner scanner = new Scanner(System.in); // creates a scanner
        System.out.print("Your shot? ");
        shot = -1;
        if (scanner.hasNextInt()) {
            shot = scanner.nextInt();
        }
        else {
            System.out.println("");
            scanner.next();
        }

        while (!shot_choices.contains(shot)) {
            System.out.print("Incorrect answer. Retype it. Your shot?");
            if (scanner.hasNextInt()) {
                shot = scanner.nextInt();
            }
            else {
                System.out.println("");
                scanner.next();
            }
        }

        if (time < 100 || Math.random() < .5) {
            if (shot == 1 || shot == 2) {
                dartmouth_jump_shot();
            }
            else {
                dartmouth_non_jump_shot();
            }
        }
        else {
            if (score[0] != score[1]) {
                System.out.println("\n   ***** End Of Game *****");
                System.out.println("Final Score: Dartmouth: " + score[1] + "  "
                      + opponent + ": " + score[0]);
                System.exit(0);
            }
            else {
                System.out.println("\n   ***** End Of Second Half *****");
                System.out.println("Score at end of regulation time:");
                System.out.println("     Dartmouth: " + score[1] + " " +
                      opponent + ": " + score[0]);
                System.out.println("Begin two minute overtime period");
                time = 93;
                start_of_period();
            }
        }
    }

    // simulates the opponents jumpshot
    private void opponent_jumpshot() {
        System.out.println("Jump Shot.");
        if (8/defense*Math.random() > .35) {
            if (8/defense*Math.random() > .75) {
                if (8/defense*Math.random() > .9) {
                    System.out.println("Offensive foul. Dartmouth's ball.\n");
                    dartmouth_ball();
                }
                else {
                    foul_shots(0);
                    dartmouth_ball();
                }
            }
            else {
                System.out.println("Shot is off the rim.");
                if (defense/6*Math.random() > .5) {
                    System.out.println(opponent + " controls the rebound.");
                    if (defense == 6) {
                        if (Math.random() > .75) {
                            System.out.println("Ball stolen. Easy lay up for Dartmouth.");
                            add_points(1, 2);
                            opponent_ball();
                        }
                        else {
                            if (Math.random() > .5) {
                                System.out.println("");
                                opponent_non_jumpshot();
                            }
                            else {
                                System.out.println("Pass back to " + opponent +
                                      " guard.\n");
                                opponent_ball();
                            }
                        }
                    }
                    else {
                        if (Math.random() > .5) {
                            opponent_non_jumpshot();
                        }
                        else {
                            System.out.println("Pass back to " + opponent +
                                  " guard.\n");
                            opponent_ball();
                        }
                    }
                }
                else {
                    System.out.println("Dartmouth controls the rebound.\n");
                    dartmouth_ball();
                }
            }
        }
        else {
            System.out.println("Shot is good.");
            add_points(0, 2);
            dartmouth_ball();
        }
    }

    // simulates opponents lay up or set shot
    private void opponent_non_jumpshot() {
        if (opponent_chance > 3) {
            System.out.println("Set shot.");
        }
        else {
            System.out.println("Lay up");
        }
        if (7/defense*Math.random() > .413) {
            System.out.println("Shot is missed.");
            if (defense/6*Math.random() > .5) {
                System.out.println(opponent + " controls the rebound.");
                if (defense == 6) {
                    if (Math.random() > .75) {
                        System.out.println("Ball stolen. Easy lay up for Dartmouth.");
                        add_points(1, 2);
                        opponent_ball();
                    }
                    else {
                        if (Math.random() > .5) {
                            System.out.println("");
                            opponent_non_jumpshot();
                        }
                        else {
                            System.out.println("Pass back to " + opponent +
                                  " guard.\n");
                            opponent_ball();
                        }
                    }
                }
                else {
                    if (Math.random() > .5) {
                        System.out.println("");
                        opponent_non_jumpshot();
                    }
                    else {
                        System.out.println("Pass back to " + opponent + " guard\n");
                        opponent_ball();
                    }
                }
            }
            else {
                System.out.println("Dartmouth controls the rebound.\n");
                dartmouth_ball();
            }
        }
        else {
            System.out.println("Shot is good.");
            add_points(0, 2);
            dartmouth_ball();
        }
    }

    // simulates an opponents possesion
    // #randomly picks jump shot or lay up / set shot.
    private void opponent_ball() {
        time ++;
        if (time == 50) {
            halftime();
        }
        opponent_chance = 10/4*Math.random()+1;
        if (opponent_chance > 2) {
            opponent_non_jumpshot();
        }
        else {
            opponent_jumpshot();
        }
    }

    public static void main(String[] args) {
        Basketball new_game = new Basketball();
    }
}

```