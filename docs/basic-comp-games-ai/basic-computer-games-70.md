# BasicComputerGames源码解析 70

# `76_Russian_Roulette/javascript/russianroulette.js`

这段代码定义了两个函数，分别是`print()`和`input()`。

`print()`函数的作用是接收一个字符串参数（str），将其显示在页面上，并将其插入到页面上一个具有特定 id 的 div 元素中。

`input()`函数的作用是接收一个字符串参数（userInput），将其存储在变量`userInput`中，并返回一个Promise。该函数使用户可以输入一个字符串，并在用户输入后将其存储在变量中。然后，该函数将Promise解决为包含用户输入的字符串。


```
// RUSSIAN ROULETTE
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

这段代码是一个 HTML 游戏，名为“RUSSIAN ROULETTE”。它通过不同的动画效果和游戏规则向玩家展示了俄罗斯轮盘赌的过程。游戏中有两种操作：通过点击按钮让“俄罗斯轮盘赌”开始，或者通过点击按钮让游戏继续。在游戏过程中，通过计算机会产生一系列随机的操作，包括轮子旋转、枪声响起、裁判读出得分等。游戏的胜利条件是让 21 个轮子都转完，或者在 21 个轮子转完前达到指定的分数。


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
    print(tab(28) + "RUSSIAN ROULETTE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("THIS IS A GAME OF >>>>>>>>>>RUSSIAN ROULETTE.\n");
    restart = true;
    while (1) {
        if (restart) {
            restart = false;
            print("\n");
            print("HERE IS A REVOLVER.\n");
        }
        print("TYPE '1' TO SPIN CHAMBER AND PULL TRIGGER.\n");
        print("TYPE '2' TO GIVE UP.\n");
        print("GO");
        n = 0;
        while (1) {
            i = parseInt(await input());
            if (i == 2) {
                print("     CHICKEN!!!!!\n");
                break;
            }
            n++;
            if (Math.random() > 0.833333) {
                print("     BANG!!!!!   YOU'RE DEAD!\n");
                print("CONDOLENCES WILL BE SENT TO YOUR RELATIVES.\n");
                break;
            }
            if (n > 10) {
                print("YOU WIN!!!!!\n");
                print("LET SOMEONE ELSE BLOW HIS BRAINS OUT.\n");
                restart = true;
                break;
            }
            print("- CLICK -\n");
            print("\n");
        }
        print("\n");
        print("\n");
        print("\n");
        print("...NEXT VICTIM...\n");
    }
}

```

这是 C 语言中的一个程序，名为 "main"。程序的作用是启动一个 C 语言程序，并将其提交给计算机运行。当你运行这个程序时，程序将读取用户输入的多行字符，并将其存储在一个数组 "userInputs" 中。然后，程序将遍历数组，检查输入是否包含 "quit"，如果是，就退出程序。如果输入不是 "quit"，程序将打印 "Hello World" 消息，并等待用户输入新的字符。


```
main();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


# `76_Russian_Roulette/python/russianroulette.py`

这段代码是一个文本版的俄罗斯轮盘赌游戏。在游戏中，你将有机会在一个带有1个子弹和5个空弹夹的 revolver上进行游戏。你通过输入“1”来决定继续游戏，或者以“2”来退出。游戏规则是，你需要在游戏中赢得10次游戏并保持生命。

这个程序是由Tom Adametx在1978年的Curtis Jr. High School中编写而成的。


```
"""
Russian Roulette

From Basic Computer Games (1978)

   In this game, you are given by the computer a
  revolver loaded with one bullet and five empty
  chambers. You spin the chamber and pull the trigger
  by inputting a "1", or, if you want to quit, input
  a "2". You win if you play ten times and are still
  alive.
   Tom Adametx wrote this program while a student at
  Curtis Jr. High School in Sudbury, Massachusetts.
"""


```

这段代码是一个Python程序，它的主要目的是让用户参与一个俄罗斯轮盘赌游戏。程序中定义了一个名为"initial_message"的函数，用于在游戏开始时打印一些介绍信息。接下来定义了一个名为"NUMBER_OF_ROUNDS"的变量，表示游戏轮数，其值为9。

此外，程序中定义了一个名为"parse_input"的函数，用于接收用户输入，并返回一个整数。在此函数中，程序会提示用户输入一个数字，然后判断输入是否为数字。如果输入不是数字，程序会输出一个错误消息并提示用户重新输入数字。


```
from random import random

NUMBER_OF_ROUNDS = 9


def initial_message() -> None:
    print(" " * 28 + "Russian Roulette")
    print(" " * 15 + "Creative Computing  Morristown, New Jersey\n\n\n")
    print("This is a game of >>>>>>>>>>Russian Roulette.\n")
    print("Here is a Revolver.")


def parse_input() -> int:
    while True:
        try:
            i = int(input("? "))
            return i
        except ValueError:
            print("Number expected...")


```

这段代码是一个简单的 Python 游戏，玩家需要根据提示在有限次数内在一个虚拟的迷宫中找到出口。

游戏的主要逻辑在 `main` 函数中，首先会调用一个名为 `initial_message` 的函数，这个函数会在游戏开始时输出一段欢迎消息，然后进入一个无限循环，等待玩家输入。

循环内部有一个条件 `dead` 和一个变量 `n`,`dead` 表示是否发现了一个敌人，`n` 表示当前循环的迭代次数。玩家需要在每次循环中点击屏幕上的一个按钮，这个按钮会随机改变一个值，如果是 2 的话，游戏就会结束。每次循环结束后，游戏会随机生成一个数字 `i`，如果 `i` 的值大于游戏中的 `NUMBER_OF_ROUNDS` 的话，游戏就会结束，否则游戏会继续。

如果 `i` 的值是 2，游戏就会结束，否则游戏会继续，并且会生成一个新的敌人。新敌人会在同样的条件下生成，直到游戏结束。

游戏结束时，游戏会输出一些信息，包括输出敌人的位置，提示玩家可以继续游戏，以及输出一些游戏相关的信息，如“你赢了”或“你死了”。


```
def main() -> None:
    initial_message()
    while True:
        dead = False
        n = 0
        print("Type '1' to Spin chamber and pull trigger")
        print("Type '2' to Give up")
        print("Go")
        while not dead:
            i = parse_input()

            if i == 2:
                break

            if random() > 0.8333333333333334:
                dead = True
            else:
                print("- CLICK -\n")
                n += 1

            if n > NUMBER_OF_ROUNDS:
                break
        if dead:
            print("BANG!!!!!   You're Dead!")
            print("Condolences will be sent to your relatives.\n\n\n")
            print("...Next victim...")
        else:
            if n > NUMBER_OF_ROUNDS:
                print("You win!!!!!")
                print("Let someone else blow his brain out.\n")
            else:
                print("     Chicken!!!!!\n\n\n")
                print("...Next victim....")


```

这段代码是一个用于检测游戏是否已成功结束的程序。它首先检查当前脚本是否是主程序，如果是，就执行主程序中的代码。接下来，它将调用一个名为“main”的函数，这个函数可能是包含游戏逻辑的函数。最后，它将输出一条信息，表明游戏是否已成功结束，如果是，该信息将包含在游戏中。


```
if __name__ == "__main__":
    main()

########################################################
# Porting Notes
#
#    Altough the description says that accepts "1" or "2",
#   the original game accepts any number as input, and
#   if it's different of "2" the program considers
#   as if the user had passed "1". That feature was
#   kept in this port.
#    Also, in the original game you must "pull the trigger"
#   11 times instead of 10 in orden to win,
#   given that N=0 at the beginning and the condition to
#   win is "IF N > 10 THEN  80". That was fixed in this
```

这段代码的作用是询问用户输入触发器（trigger）的编号，用户需要输入10次才能确认。该代码定义了一个名为trigger的常量，其值为NUMBER_OF_ROUNDS。这是一段输入输出代码，具体作用是询问用户输入触发器编号以进行轮询。


```
#   port, asking the user to pull the trigger only ten
#   times, tough the number of round can be set changing
#   the constant NUMBER_OF_ROUNDS.
#
########################################################

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Salvo

The rules are _not_ explained by the program, so read carefully this description by Larry Siegel, the program author.

SALVO is played on a 10x10 grid or board using an x,y coordinate system. The player has 4 ships:
- battleship (5 squares)
- cruiser (3 squares)
- two destroyers (2 squares each)

The ships may be placed horizontally, vertically, or diagonally and must not overlap. The ships do not move during the game.

As long as any square of a battleship still survives, the player is allowed three shots, for a cruiser 2 shots, and for each destroyer 1 shot. Thus, at the beginning of the game the player has 3+2+1+1=7 shots. The players enters all of his shots and the computer tells what was hit. A shot is entered by its grid coordinates, x,y. The winner is the one who sinks all of the opponents ships.

Important note: Your ships are located and the computer’s ships are located on 2 _separate_ 10x10 boards.

Author of the program is Lawrence Siegel of Shaker Heights, Ohio.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=142)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=157)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

The program does no validation of ship positions; your ship coordinates may be scattered around the board in any way you like.  (Computer ships will not do this, but they may be placed diagonally in such a way that they cross each other.)  Scattering your ships in this way probably defeats whatever all that spaghetti-code logic the computer is using to pick its moves, which is based on the assumption of contiguous ships.

Moreover: as per the analysis in

https://forums.raspberrypi.com/viewtopic.php?p=1997950#p1997950

see also the earlier post

https://forums.raspberrypi.com/viewtopic.php?p=1994961#p1994961

in the same thread, there is a typo in later published versions of the SALVO Basic source code compared to the original edition of 101 Basic Computer Games.

This typo is interesting because it causes the program to play by a much weaker strategy while exhibiting no other obvious side effects. I would recommend changing the line 3970 in the Basic program back to the original

`3970 K(R,S)=K(R,S)+E(U)-2*INT(H(U)+.5)`

and to change the JavaScript program accordingly.  (And note that some ports — looking at you, Python — do not implement the original strategy at all, but merely pick random unshot locations for every shot.)



# `77_Salvo/csharp/Coordinate.cs`

这段代码定义了一个名为 Coordinate 的结构体，它包含一个整数类型的变量 Value，以及 Min 和 Max 两个常量。

Coordinate 的结构体定义了一些方法：

1. Range：返回 1 到 10 之间的整数序列，然后对每个整数调用 Create 方法，将其转换为 Coordinate 类型。
2. IsInRange：判断 Value 是否在 Min 和 Max 之间，如果是，则返回 true，否则返回 false。
3. Create：根据给定的 float 值返回一个新的 Coordinate 对象。
4. TryCreateValid：尝试根据给定的 float 值返回一个 Coordinate 对象，如果返回 false，则返回默认的 Coordinate 对象。这个方法使用了 Create 方法，如果 Create 方法返回 false，则说明给定的 float 值无法创建有效的 Coordinate 对象，从而返回 default。
5. BringIntoRange：返回一个将 Value 移动到 Min 或 Max 范围内的 Coordinate 对象。这个方法使用了 switch 语句，根据给定的 Min 和 Max 值，返回一个合适的 Coordinate 对象。
6. operator +：重载了加号运算符，用于在 Coordinate 之间进行加法操作。
7. operator -：重载了减号运算符，用于在 Coordinate 之间进行减法操作。
8. ToString：重载了字符串格式化运算符，用于将 Coordinate 对象转换为字符串并输出。

除了上述方法外，还有几个辅助方法：

1. MinValue 和 MaxValue：定义了两个常量，分别表示 Min 和 Max 两个整数值。
2. Range：返回 1 到 10 之间的整数序列，用于 Coordinate 的 range 属性的实现。
3. Create：帮助用户返回一个 Coordinate 对象。
4. TryCreateValid：尝试根据给定的 float 值返回一个 Coordinate 对象，如果返回 false，则返回 default，这个方法的作用与之前的 Create 方法类似，但是会默认调用此方法返回 default。


```
namespace Salvo;

internal record struct Coordinate(int Value)
{
    public const int MinValue = 1;
    public const int MaxValue = 10;

    public static IEnumerable<Coordinate> Range => Enumerable.Range(1, 10).Select(v => new Coordinate(v));

    public bool IsInRange => Value is >= MinValue and <= MaxValue;

    public static Coordinate Create(float value) => new((int)value);

    public static bool TryCreateValid(float value, out Coordinate coordinate)
    {
        coordinate = default;
        if (value != (int)value) { return false; }

        var result = Create(value);

        if (result.IsInRange)
        {
            coordinate = result;
            return true;
        }

        return false;
    }

    public Coordinate BringIntoRange(IRandom random)
        => Value switch
        {
            < MinValue => new(MinValue + (int)random.NextFloat(2.5F)),
            > MaxValue => new(MaxValue - (int)random.NextFloat(2.5F)),
            _ => this
        };

    public static implicit operator Coordinate(float value) => Create(value);
    public static implicit operator int(Coordinate coordinate) => coordinate.Value;

    public static Coordinate operator +(Coordinate coordinate, int offset) => new(coordinate.Value + offset);
    public static int operator -(Coordinate a, Coordinate b) => a.Value - b.Value;

    public override string ToString() => $" {Value} ";
}

```

# `77_Salvo/csharp/Fleet.cs`

This is a simple implementation of a battleship fleet in the game Men of War. It has the basic logic to add a battleship to the fleet, and then it tries to position the battleship in a shot.

It has two methods one for the random version and one for the Coordinate version, in the random version it will try to add the battleship to the fleet by trying to position it in a shot, if it succeeds it will return. In the coordinate version, it will add the battleship to the fleet by adding it to a list and then it will try to position it in a shot.

It also has a method called "ReceiveShots" that takes a random shot position and an action to report the hit.

Note that this is a simple implementation, the game has many features and it is not an complete implementation of the game.


```
using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;

namespace Salvo;

internal class Fleet
{
    private readonly List<Ship> _ships;

    internal Fleet(IReadWrite io)
    {
        io.WriteLine(Prompts.Coordinates);
        _ships = new()
        {
            new Battleship(io),
            new Cruiser(io),
            new Destroyer("A", io),
            new Destroyer("B", io)
        };
    }

    internal Fleet(IRandom random)
    {
        _ships = new();
        while (true)
        {
            _ships.Add(new Battleship(random));
            if (TryPositionShip(() => new Cruiser(random)) &&
                TryPositionShip(() => new Destroyer("A", random)) &&
                TryPositionShip(() => new Destroyer("B", random)))
            {
                return;
            } 
            _ships.Clear();
        }

        bool TryPositionShip(Func<Ship> shipFactory)
        {
            var shipGenerationAttempts = 0;
            while (true)
            {
                var ship = shipFactory.Invoke();
                shipGenerationAttempts++;
                if (shipGenerationAttempts > 25) { return false; }
                if (_ships.Min(ship.DistanceTo) >= 3.59)
                {
                    _ships.Add(ship);
                    return true; 
                }
            }
        }
    }

    internal IEnumerable<Ship> Ships => _ships.AsEnumerable();

    internal void ReceiveShots(IEnumerable<Position> shots, Action<Ship> reportHit)
    {
        foreach (var position in shots)
        {
            var ship = _ships.FirstOrDefault(s => s.IsHit(position));
            if (ship == null) { continue; }
            if (ship.IsDestroyed) { _ships.Remove(ship); }
            reportHit(ship);
        }
    }
}

```

# `77_Salvo/csharp/Game.cs`



这段代码是一个简单的游戏，玩家需要通过选择角色并选择行动来与其他玩家或电脑进行交互。游戏通过两个私有变量来跟踪游戏的行为：IReadWrite和一个IRandom。

Game类有两个构造函数，分别接受一个IReadWrite和一个IRandom对象。这些构造函数分别初始化Game对象所需要的资源和随机数种子。

Play方法是游戏的主要方法，它使用IReadWrite和IRandom对象来写入游戏标题并从玩家那里获取选择。然后，它创建一个TurnHandler对象，该对象使用_io和_random来跟踪游戏进展。接下来，它循环等待玩家选择并调用turnHandler的PlayTurn方法来获取结果。如果结果是Computer，那么游戏已经结束，输出YouWon。否则，游戏还在继续，输出IWon。

最后，代码中还有一些注释，说明了代码的一些用途，例如说明游戏如何与玩家交互，如何处理玩家选择，如何处理游戏结果等等。


```
namespace Salvo;

internal class Game 
{
    private readonly IReadWrite _io;
    private readonly IRandom _random;

    public Game(IReadWrite io, IRandom random)
    {
        _io = io;
        _random = random;
    }

    internal void Play()
    {
        _io.Write(Streams.Title);

        var turnHandler = new TurnHandler(_io, _random);
        _io.WriteLine();

        Winner? winner;
        do 
        {
            winner = turnHandler.PlayTurn();
        } while (winner == null);

        _io.Write(winner == Winner.Computer ? Streams.IWon : Streams.YouWon);
    }
}

```

# `77_Salvo/csharp/Offset.cs`

这段代码定义了一个名为`Offset`的结构体，它包含两个整数字段`X`和`Y`，以及一个内部软件引用来获取或设置偏移量的字段`scale`。

在`Offset`结构体的`operator *`方法中，通过将`offset`和`scale`乘以同一个整数，来创建一个新的`Offset`实例。

`operator []`方法返回一个可迭代的`Offset`实例，用于表示单位长度内的点。

另外，`Units`方法使用一个for循环，遍历平面上的所有点，并创建相应的`Offset`实例，然后将它们添加到输出流中。


```
namespace Salvo;

internal record struct Offset(int X, int Y)
{
    public static readonly Offset Zero = 0;

    public static Offset operator *(Offset offset, int scale) => new(offset.X * scale, offset.Y * scale);

    public static implicit operator Offset(int value) => new(value, value);

    public static IEnumerable<Offset> Units
    {
        get
        {
            for (int x = -1; x <= 1; x++)
            {
                for (int y = -1; y <= 1; y++)
                {
                    var offset = new Offset(x, y);
                    if (offset != Zero) { yield return offset; }
                }
            }
        }
    }
}

```

# `77_Salvo/csharp/Position.cs`



这段代码定义了一个名为Position的结构体，用于表示二维平面上的一个点。这个结构体有两个域：一个布尔类型的IsInRange域，表示点是否在指定的范围内，还有一个布尔类型的IsOnDiagonal域，表示点是否在垂直于X轴的直线上。

除了这两个域，还有两个静态方法：Create和TryCreateValid。Create方法接受两个参数，表示位置的坐标，并返回一个表示该位置的Position对象。TryCreateValid方法接受两个参数，表示位置的坐标，并返回一个表示该位置是否有效的布尔值。

All方法接受一个位置，并返回一个IEnumerable<Position>对象，表示该位置的所有邻居。Neighbours方法接受一个位置和一个Offset对象，表示该位置的所有邻居。

DistanceTo方法计算两个位置之间的距离，并返回一个表示该距离的浮点数。BringIntoRange方法尝试将一个位置移动到指定的区间，并返回一个新的Position对象。

operator+方法接受两个Position对象和一個Offset对象，并返回一个新的Position对象。

is operator==operator ju容貌粉末文字long long int，可以构造 bone INF带宽折合金56430健康小吃何首乌林忆 Sweep对眼睛有益的情歌 end，想表达。就是想问一个属性可以有多个值，返回多个值。

除了上述方法之外，还可以看到一些其他的代码，包括一个用于创建指定位置的函数，一个用于获取指定位置的所有邻居的函数，一个用于计算指定位置与给定位置之间的距离的函数，一个用于将一个位置移动到指定区间的函数等等。


```
namespace Salvo;

internal record struct Position(Coordinate X, Coordinate Y)
{
    public bool IsInRange => X.IsInRange && Y.IsInRange;
    public bool IsOnDiagonal => X == Y;

    public static Position Create((float X, float Y) coordinates) => new(coordinates.X, coordinates.Y);

    public static bool TryCreateValid((float X, float Y) coordinates, out Position position)
    {
        if (Coordinate.TryCreateValid(coordinates.X, out var x) && Coordinate.TryCreateValid(coordinates.Y, out var y))
        {
            position = new(x, y);
            return true;
        }

        position = default;
        return false;
    }

    public static IEnumerable<Position> All
        => Coordinate.Range.SelectMany(x => Coordinate.Range.Select(y => new Position(x, y)));

    public IEnumerable<Position> Neighbours
    {
        get
        {
            foreach (var offset in Offset.Units)
            {
                var neighbour = this + offset;
                if (neighbour.IsInRange) { yield return neighbour; }
            }
        }
    }

    internal float DistanceTo(Position other)
    {
        var (deltaX, deltaY) = (X - other.X, Y - other.Y);
        return (float)Math.Sqrt(deltaX * deltaX + deltaY * deltaY);
    }

    internal Position BringIntoRange(IRandom random)
        => IsInRange ? this : new(X.BringIntoRange(random), Y.BringIntoRange(random));

    public static Position operator +(Position position, Offset offset) 
        => new(position.X + offset.X, position.Y + offset.Y);

    public static implicit operator Position(int value) => new(value, value);

    public override string ToString() => $"{X}{Y}";
}

```

# `77_Salvo/csharp/Program.cs`

这段代码的作用是创建并运行一个名为“MyGame”的游戏。它引入了多个外部库，包括System、Games.Common.IO、Games.Common.Randomness和Salvo，以及定义了一些静态类Salvo.Resources.Resource。

具体来说，这段代码创建了一个名为“MyGame”的主类，该类继承自Salvo.Ships.Game类，然后使用new关键字创建了一个新的Game实例，并传入一个ConsoleIO对象和一个DataRandom对象，这些对象用于与游戏交互。最后，调用该Game实例的Play()方法来运行游戏。


```
global using System;
global using Games.Common.IO;
global using Games.Common.Randomness;
global using Salvo;
global using Salvo.Ships;
global using static Salvo.Resources.Resource;

//new Game(new ConsoleIO(), new RandomNumberGenerator()).Play();
new Game(new ConsoleIO(), new DataRandom()).Play();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `77_Salvo/csharp/TurnHandler.cs`

This looks like a implementation of the AI for a Connect-F营养物质配送 game. It appears to handle turns taken by the human or the computer, and displays the results of the shots taken.

The `PlayComputerTurn` method appears to take the computer's turn, while the `PlayHumanTurn` method appears to take the human's turn. Both methods call the `PlayShot` method to determine which shots to take, and then write a message to the console indicating the outcome of the shot.

The `PlayShot` method appears to be responsible for determining which shots the computer or human should take, based on the number of shots remaining and the game's rules. It does this by calling the `GetShots` method of the `ComputerShotSelector` class, passing in the current turn number and returning the number of shots that can be used for a shot. If there are more shots available, the method returns `Winner.Human`, indicating that the computer should take a shot. If there are no more shots available, the method returns `Winner.Computer`, indicating that the computer should not take a shot.

The `IHaveShots` method is a helper method that returns a message indicating the number of shots the computer has, based on the game's rules. It uses the `GetShots` method of the `ComputerShotSelector` class to determine the number of shots available, and writes a message accordingly.

The `IHaveMoreShotsThanSquares` message is also a helper method that returns a message indicating whether the computer has more shots than squares remaining in the game. It uses the `GetShots` method of the `ComputerShotSelector` class to determine the number of shots available, and writes a message accordingly.


```
using Salvo.Targetting;

namespace Salvo;

internal class TurnHandler
{
    private readonly IReadWrite _io;
    private readonly Fleet _humanFleet;
    private readonly Fleet _computerFleet;
    private readonly bool _humanStarts;
    private readonly HumanShotSelector _humanShotSelector;
    private readonly ComputerShotSelector _computerShotSelector;
    private readonly Func<Winner?> _turnAction;
    private int _turnNumber;

    public TurnHandler(IReadWrite io, IRandom random)
    {
        _io = io;
        _computerFleet = new Fleet(random);
        _humanFleet = new Fleet(io);
        _turnAction = AskWhoStarts()
            ? () => PlayHumanTurn() ?? PlayComputerTurn()
            : () => PlayComputerTurn() ?? PlayHumanTurn();
        _humanShotSelector = new HumanShotSelector(_humanFleet, io);
        _computerShotSelector = new ComputerShotSelector(_computerFleet, random, io);
    }

    public Winner? PlayTurn()
    {
        _io.Write(Strings.Turn(++_turnNumber));
        return _turnAction.Invoke();
    }

    private bool AskWhoStarts()
    {
        while (true)
        {
            var startResponse = _io.ReadString(Prompts.Start);
            if (startResponse.Equals(Strings.WhereAreYourShips, StringComparison.InvariantCultureIgnoreCase))
            {
                foreach (var ship in _computerFleet.Ships)
                {
                    _io.WriteLine(ship);
                }
            }
            else
            {
                return startResponse.Equals("yes", StringComparison.InvariantCultureIgnoreCase);
            }
        }
    }

    private Winner? PlayComputerTurn()
    {
        var numberOfShots = _computerShotSelector.NumberOfShots;
        _io.Write(Strings.IHaveShots(numberOfShots));
        if (numberOfShots == 0) { return Winner.Human; }
        if (_computerShotSelector.CanTargetAllRemainingSquares)
        {
            _io.Write(Streams.IHaveMoreShotsThanSquares);
            return Winner.Computer;
        }

        _humanFleet.ReceiveShots(
            _computerShotSelector.GetShots(_turnNumber),
            ship =>
            { 
                _io.Write(Strings.IHit(ship.Name));
                _computerShotSelector.RecordHit(ship, _turnNumber);
            });

        return null;
    }

    private Winner? PlayHumanTurn()
    {
        var numberOfShots = _humanShotSelector.NumberOfShots;
        _io.Write(Strings.YouHaveShots(numberOfShots));
        if (numberOfShots == 0) { return Winner.Computer; }
        if (_humanShotSelector.CanTargetAllRemainingSquares) 
        { 
            _io.WriteLine(Streams.YouHaveMoreShotsThanSquares);
            return Winner.Human;
        }
        
        _computerFleet.ReceiveShots(
            _humanShotSelector.GetShots(_turnNumber), 
            ship => _io.Write(Strings.YouHit(ship.Name)));
        
        return null;
    }
}

```

# `77_Salvo/csharp/Winner.cs`

这段代码定义了一个内部枚举类型`Winner`，该枚举类型有两个枚举值`Human`和`Computer`。

`internal`关键字表示该枚举类型是内部使用的，意味着它只能在国内使用，不能在公共代码中使用。

`Winner`是一个命名枚举类型，它定义了两个枚举值`Human`和`Computer`，分别表示人类和计算机。

枚举类型是一种数据类型，它可以用来定义某些可重复的值，这些值通常与特定类型相关联。在这里，`Winner`枚举类型定义了两个值，`Human`和`Computer`，用于表示人类和计算机。

`enum`关键字表示该段代码是一个枚举类型，`Winner`是该枚举类型的名称，`Human`和`Computer`是该枚举类型的两个枚举值。


```
namespace Salvo;

internal enum Winner
{
    Human,
    Computer
}

```

# `77_Salvo/csharp/Extensions/IOExtensions.cs`

这段代码是一个名为 "Games.Common.IO" 的命名空间中包含的类 "IOExtensions"。

这个类中包含了一些方法来操作文件输入和输出。下面是每个方法的说明：

- `ReadPosition(this IReadWrite io)` 返回一个表示输入位置的 `Position` 对象。这个方法的作用是读取一个字符串，并尝试在字符串中找到指定的数字，然后返回位置对象。如果没有找到数字，则会返回 `Position.Zero`。

- `ReadValidPosition(this IReadWrite io)` 返回一个表示输入位置的 `Position` 对象。这个方法会尝试不断地向字符串中读取数字，直到找到有效的位置为止。如果找到了有效位置，则返回该位置对象。如果一直无法找到有效位置，则会返回 `Position.Zero`。

- `ReadPositions(this IReadWrite io, string shipName, int shipSize)` 返回一个包含指定船只中所有位置的 `IEnumerable<Position>` 对象。这个方法的作用是读取指定船只中所有位置，并返回一个 `IEnumerable<Position>` 对象，每个位置都是根据指定的船只、名称和大小来确定的。


```
namespace Games.Common.IO;

internal static class IOExtensions
{
    internal static Position ReadPosition(this IReadWrite io) => Position.Create(io.Read2Numbers(""));

    internal static Position ReadValidPosition(this IReadWrite io)
    {
        while (true)
        {
            if (Position.TryCreateValid(io.Read2Numbers(""), out var position)) 
            { 
                return position; 
            }
            io.Write(Streams.Illegal);
        }
    }

    internal static IEnumerable<Position> ReadPositions(this IReadWrite io, string shipName, int shipSize)
    {
        io.WriteLine(shipName);
        for (var i = 0; i < shipSize; i++)
        {
             yield return io.ReadPosition();
        }
    }
}

```

# `77_Salvo/csharp/Extensions/RandomExtensions.cs`

这段代码是一个名为"Games.Common.Randomness"命名空间下的内部类，主要定义了一些扩展类来处理随机数。

首先，定义了一个名为"RandomExtensions"的内部类，以及一个内部静态函数"NextShipPosition"，该函数接受一个IRandom类型的参数，返回一个包含起降位置的元组。

接着，定义了一个名为"NextCoordinate"的内部静态函数，该函数接受一个IRandom类型的参数，返回一个坐标值。

然后，定义了一个名为"NextOffset"的内部静态函数，该函数接受一个IRandom类型的参数，返回一个随机整数。

接下来，定义了一个名为"GetRandomShipPositionInRange"的内部静态函数，该函数接受一个IRandom类型的参数以及一个整数参数，返回一个包含起降位置的元组。函数内部通过使用nextShipPosition函数生成随机位置，并对结果进行判断，如果生成的位置在指定范围内，则返回该元组，否则继续尝试。

最后，在RandomExtensions命名空间中定义了一些额外的静态属性和函数，用于在其他类中使用。


```
namespace Games.Common.Randomness;

internal static class RandomExtensions
{
    internal static (Position, Offset) NextShipPosition(this IRandom random)
    {
        var startX = random.NextCoordinate();
        var startY = random.NextCoordinate();
        var deltaY = random.NextOffset();
        var deltaX = random.NextOffset();
        return (new(startX, startY), new(deltaX, deltaY));
    }

    private static Coordinate NextCoordinate(this IRandom random)
        => random.Next(Coordinate.MinValue, Coordinate.MaxValue + 1);

    private static int NextOffset(this IRandom random) => random.Next(-1, 2);

    internal static (Position, Offset) GetRandomShipPositionInRange(this IRandom random, int shipSize)
    {
        while (true)
        {
            var (start, delta) = random.NextShipPosition();
            var shipSizeLessOne = shipSize - 1;
            var end = start + delta * shipSizeLessOne;
            if (delta != 0 && end.IsInRange) 
            {
                return (start, delta);
            }
        }
    }
}

```

# `77_Salvo/csharp/Resources/Resource.cs`



这段代码是一个自定义的 .NET 类，名为 `Resource`。它包含了许多与游戏“文明”游戏相关的资源文件和文本，如游戏中的船的位置、船的归属、射击数、胜利条件等等。

具体来说，这段代码：

1. 定义了一个 `Streams` 类，它包含了一些与游戏中的进度相关的流。例如，`Title` 是一个 `Stream`，代表游戏中的主菜单。`YouHaveMoreShotsThanSquares`、`YouWon` 等也是 `Stream`，代表游戏中的胜利条件和其他进度条件。

2. 定义了一个 `Strings` 类，它包含了一些与游戏中的字符串相关的字符串常量。例如，`WhereAreYourShips`、`YouHaveShots` 等字符串常量分别代表游戏中的船的位置、船的归属等信息。

3. 定义了一个 `Prompts` 类，它包含了一些与游戏中的提示相关的字符串常量。例如，`Coordinates`、`Start` 等字符串常量分别代表游戏中的坐标、游戏开始提示等信息。

4. 定义了一个 `GetStream` 方法，它接受一个字符串参数和一个可选的 `name` 参数。这个方法返回游戏引擎中与指定的资源文件相关的输入流。

5. 定义了一个 `Format` 方法，它接受一个 `T` 参数和一个字符串参数 `name`。这个方法使用 `GetString` 方法获取与指定资源文件相关的字符串，并使用格式化字符串的方法将 `T` 类型的值转换为字符串。

6. 定义了一个 `GetString` 方法，它接受一个字符串参数 `name`。这个方法从游戏引擎中的输入流中读取与指定资源文件相关的字符串，并将它们返回。


```
using System.Reflection;
using System.Runtime.CompilerServices;

namespace Salvo.Resources;

internal static class Resource
{
    internal static class Streams
    {
        public static Stream Title => GetStream();
        public static Stream YouHaveMoreShotsThanSquares => GetStream();
        public static Stream YouWon => GetStream();
        public static Stream IHaveMoreShotsThanSquares => GetStream();
        public static Stream IWon => GetStream();
        public static Stream Illegal => GetStream();
    }

    internal static class Strings
    {
        public static string WhereAreYourShips => GetString();
        public static string YouHaveShots(int number) => Format(number);
        public static string IHaveShots(int number) => Format(number);
        public static string YouHit(string shipName) => Format(shipName);
        public static string IHit(string shipName) => Format(shipName);
        public static string ShotBefore(int turnNumber) => Format(turnNumber);
        public static string Turn(int number) => Format(number);
    }

    internal static class Prompts
    {
        public static string Coordinates => GetString();
        public static string Start => GetString();
        public static string SeeShots => GetString();
    }

    private static string Format<T>(T value, [CallerMemberName] string? name = null) 
        => string.Format(GetString(name), value);

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

# `77_Salvo/csharp/Ships/Battleship.cs`

这段代码定义了一个名为Battleship的内部类，继承自另一个内部类Ship。

Battleship类有两个构造函数，一个是使用IReadWrite接口的构造函数，另一个是使用IRandom接口的构造函数。这两个构造函数都继承自Ship类的构造函数，因此都会调用Ship类的构造函数，并将Ship类的参数传递给内部变量。

Battleship类还重写了两个方法：

- public interface Iterator<out TResponse>
{
   bool hasNext();
   TResponse next();
   bool isDone();
}

- public class Car<out TResponse> : Iterator<TResponse>
{
   private readonly Ship _ship;
   private int _position;

   public Car(Ship ship, int position)
   {
       _ship = ship;
       _position = position;
   }

   public TResponse current => _ship.Shots[_position];
   public bool hasNext() => _ship.Size > 0;
   public TResponse next() => _ship.Shots[_position];
   public bool isDone() => _position == _ship.Size - 1;
}

注意，由于代码中使用了内部类，因此需要声明一个内部类的外部接口类来定义该内部类的行为。同时，由于该代码是为游戏设计而写的，因此其中可能会包含游戏逻辑，需要确保该代码的安全性和正确性。


```
namespace Salvo.Ships;

internal sealed class Battleship : Ship
{
    internal Battleship(IReadWrite io) 
        : base(io) 
    { 
    }

    internal Battleship(IRandom random)
        : base(random)
    {
    }

    internal override int Shots => 3;
    internal override int Size => 5;
}

```

# `77_Salvo/csharp/Ships/Cruiser.cs`



这段代码定义了一个名为Cruiser的内部类，属于名为Ship的类。Cruiser继承了Ship类，并实现了两个覆盖了Shots和Size属性的方法，即：

1. 构造函数，用于初始化Cruiser的输入输出流对象(IReadWrite io)，并复制该对象，以便在Cruiser的运行时行为中使用。

2. 析构函数，用于在Cruiser的运行时行为中清空Cruiser的输入输出流对象，以避免内存泄漏。

3. 两个覆盖了Shots属性的方法，分别实现了Cruiser的炮弹数量和尺寸。

4. 创建Cruiser实例的两种不同方式，分别使用了IReadWrite和IRandom构造函数。其中，IReadWrite构造函数的参数在创建对象时从输入流中读取数据，而IRandom构造函数的参数在创建对象时从随机数生成器中获取随机数。


```
namespace Salvo.Ships;

internal sealed class Cruiser : Ship
{
    internal Cruiser(IReadWrite io) 
        : base(io) 
    { 
    }
    
    internal Cruiser(IRandom random)
        : base(random)
    {
    }

    internal override int Shots => 2;
    internal override int Size => 3;
}

```

# `77_Salvo/csharp/Ships/Destroyer.cs`

这段代码定义了一个名为 "Destroyer" 的内部类，它继承自 "Ship" 类，并在类中重写了 "Shots" 和 "Size" 方法。

具体来说，这段代码的作用是创建和初始化一个名为 "Destroyer" 的船只，并设置其名为 "Destroyer" 且为第 "1" 种船型的船只，同时设置其拥有 "1" 支 "1" 支流的炮弹，每艘船最大可容纳 "2" 名人。


```
namespace Salvo.Ships;

internal sealed class Destroyer : Ship
{
    internal Destroyer(string nameIndex, IReadWrite io)
        : base(io, $"<{nameIndex}>")
    {
    }

    internal Destroyer(string nameIndex, IRandom random)
        : base(random, $"<{nameIndex}>")
    {
    }

    internal override int Shots => 1;
    internal override int Size => 2;
}

```

# `77_Salvo/csharp/Ships/Ship.cs`



这段代码定义了一个名为Ship的内部抽象类，用于创建和处理船的位置数据。

内部抽象类Ship继承自非内部类的Ship，因此可以推断出Ship是Ssalvo.Ships命名空间中的一部分。

Ship类包含一个抽象类型的Shots成员和一个Size成员变量，这些成员类型都是内部抽象类型，因此可以推断出Ship的具体实现类需要实现这些成员类型。

Ship类还包含一个 abstract类型的IsDamaged和IsDestroyed成员变量。IsDamaged用于判断船是否受到损坏，而IsDestroyed用于判断船是否被摧毁。

Ship类还包含一个ToElement方法，用于计算船与指定元素之间的最短距离。

最重要的是，Ship类继承自Ship，并实现了IReadWrite和IRandom接口，因此可以安全地读取和生成游戏中的位置数据。

因此，这段代码的作用是创建一个用于创建和处理船的位置数据的内部抽象类，其具体实现类需要实现IReadWrite和IRandom接口，以便读取和生成游戏中的位置数据。


```
namespace Salvo.Ships;

internal abstract class Ship
{
    private readonly List<Position> _positions = new();

    protected Ship(IReadWrite io, string? nameSuffix = null)
    {
        Name = GetType().Name + nameSuffix;
        _positions = io.ReadPositions(Name, Size).ToList();
    }

    protected Ship(IRandom random, string? nameSuffix = null)
    {
        Name = GetType().Name + nameSuffix;

        var (start, delta) = random.GetRandomShipPositionInRange(Size);
        for (var i = 0; i < Size; i++)
        {
            _positions.Add(start + delta * i);
        }
    }

    internal string Name { get; }
    internal abstract int Shots { get; }
    internal abstract int Size { get; }
    internal bool IsDamaged => _positions.Count > 0 && _positions.Count < Size;
    internal bool IsDestroyed => _positions.Count == 0;

    internal bool IsHit(Position position) => _positions.Remove(position);

    internal float DistanceTo(Ship other)
        => _positions.SelectMany(a => other._positions.Select(b => a.DistanceTo(b))).Min();

    public override string ToString() 
        => string.Join(Environment.NewLine, _positions.Select(p => p.ToString()).Prepend(Name));
}

```

# `77_Salvo/csharp/Targetting/ComputerShotSelector.cs`



这段代码定义了一个名为 ComputerShotSelector 的内部类，它继承自 ShotSelector 类，用于在游戏内部查找并选中棋子。

ComputerShotSelector 类的内部成员包括：

- _knownHitsStrategy：一个已知击中伤害的策略，用于在搜索过程中查找棋子。
- _searchPatternStrategy：一个搜索模式策略，用于在搜索过程中按照某种搜索模式查找棋子。
- _io：一个 IReadWrite 接口，用于从玩家或游戏控制台获取输入。
- _showShots：一个布尔值，用于控制是否在控制台输出选中的棋子。

ComputerShotSelector 类的一个 Constructor 方法接收三个参数：源游戏实例、随机数生成器和 IReadWrite 接口。这些参数用于初始化 ComputerShotSelector 类的实例。

ComputerShotSelector 类的 GetShots 方法返回选中棋子的位置数组。如果 _showShots 为 true，则使用控制台输出选中的棋子位置。

ComputerShotSelector 类的 RecordHit 方法用于记录选中棋子的伤害，其中仅在 _knownHitsStrategy.KnowsOfDamagedShips 为 true 时执行。

GetSelectionStrategy 的方法根据 _knownHitsStrategy 和 _searchPatternStrategy 是否可用来获取选中棋子的策略。如果 _knownHitsStrategy 存在，则使用它来获取棋子，否则使用 _searchPatternStrategy 来获取。


```
namespace Salvo.Targetting;

internal class ComputerShotSelector : ShotSelector
{
    private readonly KnownHitsShotSelectionStrategy _knownHitsStrategy;
    private readonly SearchPatternShotSelectionStrategy _searchPatternStrategy;
    private readonly IReadWrite _io;
    private readonly bool _showShots;

    internal ComputerShotSelector(Fleet source, IRandom random, IReadWrite io) 
        : base(source)
    {
        _knownHitsStrategy = new KnownHitsShotSelectionStrategy(this);
        _searchPatternStrategy = new SearchPatternShotSelectionStrategy(this, random);
        _io = io;
        _showShots = io.ReadString(Prompts.SeeShots).Equals("yes", StringComparison.InvariantCultureIgnoreCase);
    }

    protected override IEnumerable<Position> GetShots()
    {
        var shots = GetSelectionStrategy().GetShots(NumberOfShots).ToArray();
        if (_showShots)
        {
            _io.WriteLine(string.Join(Environment.NewLine, shots));
        }
        return shots;
    }

    internal void RecordHit(Ship ship, int turn) => _knownHitsStrategy.RecordHit(ship, turn);

    private ShotSelectionStrategy GetSelectionStrategy()
        => _knownHitsStrategy.KnowsOfDamagedShips ? _knownHitsStrategy : _searchPatternStrategy;
}

```

# `77_Salvo/csharp/Targetting/HumanShotSelector.cs`

这段代码定义了一个名为HumanShotSelector的内部类，继承自ShotSelector类，位于名为Salvo.Targetting的命名空间中。

该类的作用是管理一个人群射击游戏中的射击操作。在HumanShotSelector中，我们创建了一个private的成员变量_io，表示一个有效的碘量（即游戏中的资源），以及一个内部方法GetShots，用于从源代理中获取射击位置。

GetShots方法的基本逻辑是：首先检查是否可以在当前位置射击，如果允许，则从源代理中读取一个位置，并将其存储在内部位置数组中。如果当前位置之前已经被选择过，则显示“你之前已经在这里射击了”，并跳过该位置。

另外，还重写了基类ShotSelector的GetShots方法，以便在需要时可以调用。


```
namespace Salvo.Targetting;

internal class HumanShotSelector : ShotSelector
{
    private readonly IReadWrite _io;

    internal HumanShotSelector(Fleet source, IReadWrite io) 
        : base(source)
    {
        _io = io;
    }

    protected override IEnumerable<Position> GetShots()
    {
        var shots = new Position[NumberOfShots];
        
        for (var i = 0; i < shots.Length; i++)
        {
            while (true)
            {
                var position = _io.ReadValidPosition();
                if (WasSelectedPreviously(position, out var turnTargeted)) 
                { 
                    _io.WriteLine($"YOU SHOT THERE BEFORE ON TURN {turnTargeted}");
                    continue;
                }
                shots[i] = position;
                break;
            }
        }

        return shots;
    }
}

```

# `77_Salvo/csharp/Targetting/KnownHitsShotSelectionStrategy.cs`

This code appears to be a script for a game in which players take turns rolling a die and marking certain positions on a grid with a "1" or an "X". The script keeps track of which ships have been hit by the die, and updates the internal state of the game based on the results of these hits.

The script has two main methods:

* `GetShots` takes an integer `numberOfShots` and returns an array of `Position` objects representing the positions on the grid where the hit has occurred.
* `GetShotLegend` returns a dictionary of the legend of the grid, where the keys are the names of the positions on the grid and the values are the corresponding position objects.

The `GetShots` method first initializes a `tempGrid` dictionary with the position objects of all the shots, and then iterates through the rows and columns of the grid to update the `tempGrid` based on the results of the hits. The method then returns the modified `tempGrid` as an array of `Position` objects.

The `GetShotLegend` method returns a dictionary of the names of the positions on the grid, where the keys are the names of the positions and the values are the corresponding position objects.

Note that the script uses a variable `_damagedShips` to keep track of which ships have been hit by the die. This variable is initialized to new `Ship` objects for each turn, and is updated each turn with the results of the hits.


```
namespace Salvo.Targetting;

internal class KnownHitsShotSelectionStrategy : ShotSelectionStrategy
{
    private readonly List<(int Turn, Ship Ship)> _damagedShips = new();

    internal KnownHitsShotSelectionStrategy(ShotSelector shotSelector)
        : base(shotSelector)
    {
    }

    internal bool KnowsOfDamagedShips => _damagedShips.Any();

    internal override IEnumerable<Position> GetShots(int numberOfShots)
    {
        var tempGrid = Position.All.ToDictionary(x => x, _ => 0);
        var shots = Enumerable.Range(1, numberOfShots).Select(x => new Position(x, x)).ToArray();

        foreach (var (hitTurn, ship) in _damagedShips)
        {
            foreach (var position in Position.All)
            {
                if (WasSelectedPreviously(position))
                {  
                    tempGrid[position]=-10000000;
                    continue;
                }

                foreach (var neighbour in position.Neighbours)    
                {
                    if (WasSelectedPreviously(neighbour, out var turn) && turn == hitTurn)
                    {
                        tempGrid[position] += hitTurn + 10 - position.Y * ship.Shots;
                    }
                }
            }
        }

        foreach (var position in Position.All)
        {
            var Q9=0;
            for (var i = 0; i < numberOfShots; i++)
            {
                if (tempGrid[shots[i]] < tempGrid[shots[Q9]]) 
                { 
                    Q9 = i;
                }
            }
            if (position.X <= numberOfShots && position.IsOnDiagonal) { continue; }
            if (tempGrid[position]<tempGrid[shots[Q9]]) { continue; }
            if (!shots.Contains(position))
            {
                shots[Q9] = position;
            }
        }

        return shots;
    } 

    internal void RecordHit(Ship ship, int turn)
    {
        if (ship.IsDestroyed) 
        {
            _damagedShips.RemoveAll(x => x.Ship == ship);
        }
        else
        {
            _damagedShips.Add((turn, ship));
        }
    }
}

```

# `77_Salvo/csharp/Targetting/SearchPattern.cs`



这段代码定义了一个名为 `SearchPattern` 的内部类，用于在给定数据集中进行查找操作。

首先，它创建了一个名为 `_offsets` 的不可变数组，其元素个数为 6，每个元素都是一个 `Offset` 类。该数组的元素朝一个方向移动，从 1 开始，方向可以是 1 或者 -1。

接着，它定义了一个名为 `_nextIndex` 的整数变量，用于跟踪数组中下一个元素的位置。

然后，它定义了一个名为 `TryGetOffset` 的方法，用于返回一个指向指定位置的 `Offset` 对象的引用或 `false`。该方法首先检查 `_nextIndex` 是否已经大于 `_offsets` 数组长度，如果是，则返回 `false`。否则，它将遍历 `_offsets` 数组，并返回第一个元素的位置 `_nextIndex`。

最后，它定义了一个名为 `Reset` 的方法，用于重置 `_nextIndex` 变量为 0。

该代码的作用是定义一个用于在给定数据集中进行查找操作的类，该类可以接受一个可变的数据集，并返回一个指向指定位置的 `Offset` 对象的引用或 `false`。


```
using System.Collections.Immutable;

namespace Salvo.Targetting;

internal class SearchPattern
{
    private static readonly ImmutableArray<Offset> _offsets =
        ImmutableArray.Create<Offset>(new(1, 1), new(-1, 1), new(1, -3), new(1, 1), new(0, 2), new(-1, 1));

    private int _nextIndex;

    internal bool TryGetOffset(out Offset offset)
    {
        offset = default;
        if (_nextIndex >= _offsets.Length) { return false; }
        
        offset = _offsets[_nextIndex++];
        return true;
    }

    internal void Reset() => _nextIndex = 0;
}
```

# `77_Salvo/csharp/Targetting/SearchPatternShotSelector.cs`

This is a class that implements the `SearchPatternShotSelectionStrategy` interface. This interface represents a strategy for selecting shots in a game.

The class takes in two arguments in the constructor: `shotSelector` and `random`. The `shotSelector` is used to determine which shots to select and `random` is used to generate random numbers for the shot selection process.

The class has a `MaxSearchPatternAttempts` field which is the maximum number of attempts to find a valid shot pattern, and a `_searchPattern` field which is used to store the current state of the shot pattern. It also has a list of `Position` objects which store the shots that have already been made.

The `GetShots` method returns an iteration of the shots that have been found, starting with the first shot when the strategy is used for the first time.

The `SearchFrom` method is used to search for shots. It takes in the number of shots to attempt, and the seed of the random number generator. It attempts to find a valid shot pattern by calling the `TryGetOffset` method of the `_searchPattern`, and if it does, it adds the ship position and move it to the next position. If it fails to find a valid shot pattern, it will return early.

The `FindValidShots` method is used to search for valid shots. It takes in the number of shots to attempt, and is used to keep track of the ship positions that have already been processed. It attempts to find a valid shot by checking if the shot is within the range and if it is not already in the list of shots. If the shot is found, it is added to the list of shots. If the shot could be in range, it is also trying to find the best possible offset for the shot by calling the `TryGetOffset` method of the `_searchPattern` and adding the ship position to the next position.

The class also has a `IsValidShot` method which checks if a shot is valid, by checking if it is within the range and if it is not already in the list of shots.


```
namespace Salvo.Targetting;

internal class SearchPatternShotSelectionStrategy : ShotSelectionStrategy
{
    private const int MaxSearchPatternAttempts = 100;
    private readonly IRandom _random;
    private readonly SearchPattern _searchPattern = new();
    private readonly List<Position> _shots = new();

    internal SearchPatternShotSelectionStrategy(ShotSelector shotSelector, IRandom random) 
        : base(shotSelector)
    {
        _random = random;
    }

    internal override IEnumerable<Position> GetShots(int numberOfShots)
    {
        _shots.Clear();
        while(_shots.Count < numberOfShots)
        {
            var (seed, _) = _random.NextShipPosition();
            SearchFrom(numberOfShots, seed);
        }
        return _shots;
    }

    private void SearchFrom(int numberOfShots, Position candidateShot)
    {
        var attemptsLeft = MaxSearchPatternAttempts;
        while (true)
        {
            _searchPattern.Reset();
            if (attemptsLeft-- == 0) { return; }
            candidateShot = candidateShot.BringIntoRange(_random);
            if (FindValidShots(numberOfShots, ref candidateShot)) { return; }
        }
    }

    private bool FindValidShots(int numberOfShots, ref Position candidateShot)
    {
        while (true)
        {
            if (IsValidShot(candidateShot))
            {
                _shots.Add(candidateShot);
                if (_shots.Count == numberOfShots) { return true; }
            }
            if (!_searchPattern.TryGetOffset(out var offset)) { return false; }
            candidateShot += offset;
        }
    }

    private bool IsValidShot(Position candidate)
        => candidate.IsInRange && !WasSelectedPreviously(candidate) && !_shots.Contains(candidate);
}
```

# `77_Salvo/csharp/Targetting/ShotSelectionStrategy.cs`



这段代码定义了一个内部抽象类 ShotSelectionStrategy，用于处理玩家选择射击目标时的情况。

在这个内部类中，包含一个 ShotSelector 类型的成员变量，用于存储选择目标的函数，以及一个内部函数 wasSelectedPreviously，用于记录每个位置是否已经被选择过和 turn 是否已经旋转。

GetShots 函数用于获取从当前位置可用的所有位置，并返回一个射击目标的有序列表。WasSelectedPreviously函数用于检查给定的位置是否已经被选择过，如果已经选择过，则返回 true，否则返回 false。WasSelectedPreviously函数有两个重载版本，第一个重载版本没有参数 out 类型，表示需要手动设置 turn 变量；第二个重载版本则有 out 参数，表示可以通过 turn 变量来标记是否已经选择过该位置。

整段代码的作用是定义了一个用于处理选择射击目标的状态机，可以记录每个位置是否已经被选择过，并提供一个方法来获取从当前位置可用的所有位置。


```
namespace Salvo.Targetting;

internal abstract class ShotSelectionStrategy
{
    private readonly ShotSelector _shotSelector;
    protected ShotSelectionStrategy(ShotSelector shotSelector)
    {
        _shotSelector = shotSelector;
    }

    internal abstract IEnumerable<Position> GetShots(int numberOfShots);

    protected bool WasSelectedPreviously(Position position) => _shotSelector.WasSelectedPreviously(position);

    protected bool WasSelectedPreviously(Position position, out int turn)
        => _shotSelector.WasSelectedPreviously(position, out turn);
}

```