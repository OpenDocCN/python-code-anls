# BasicComputerGames源码解析 37

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `31_Depth_Charge/javascript/depthcharge.js`

这段代码定义了两个函数，分别是`print`函数和`input`函数。这两个函数都在执行以下操作：

1. 将字符串`str`通过`print`函数将其添加到页面上一个名为`output`的元素中。
2. 通过`input`函数，接收用户输入的字符串`input_str`。该函数使用`input_element`变量来存储用户输入的字符串，然后将其添加到页面上的元素中，并设置元素的`type`属性为`text`，`length`属性为`50`，以确保输入框可以接收长文本。
3. 使用`input_element`的`addEventListener`函数来监听用户输入的键盘事件。当用户按下键盘上的`13`键时，将获取到用户输入的字符串`input_str`，将其添加到页面上的元素中，并使用`print`函数将其输出到页面上。
4. 使用`input`函数的`Promise`构造函数来处理用户输入的回调函数。该构造函数使用`resolve`方法来接收并返回用户输入的字符串，使用`print`函数将其添加到页面上。


```
// DEPTH CHARGE
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

It looks like this is a simple game where the player must navigate through a grid of rooms, each of which has a door and a danger value (either "NORTH", "SOUTH", "EAST", or "WEST"). The player must always navigate to the next room that has a non-danger value in order to progress. The danger value is determined by a random number that is generated each time the player enters a new room.

The `.random()` function is used to generate a random number, and the modulo `g` (the maximum value that the random number can take on) is used to ensure that the random number stays within the range of 0-99.

The `Math.abs()` function is used to calculate the absolute value of the danger value, since the comparison in the if statement `if (Math.abs(x - a) + Math.abs(y - b) + Math.abs(z - c) == 0)` uses the sum of the absolute values.

The game also includes a catch-all label that prints if the player has entered a room with a danger value of "WEST".

It is not clear if the catch-all label is intended to be used as a hint to the player, or if it is part of the game.


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
    print(tab(30) + "DEPTH CHARGE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("DIMENSION OF THE SEARCH AREA");
    g = Math.floor(await input());
    n = Math.floor(Math.log(g) / Math.log(2)) + 1;
    print("YOU ARE THE CAPTAIN OF THE DESTROYER USS COMPUTER\n");
    print("AN ENEMY SUB HAS BEEN CAUSING YOU TROUBLE.  YOUR\n");
    print("MISSION IS TO DESTROY IT.  YOU HAVE " + n + " SHOTS.\n");
    print("SPECIFY DEPTH CHARGE EXPLOSION POINT WITH A\n");
    print("TRIO OF NUMBERS -- THE FIRST TWO ARE THE\n");
    print("SURFACE COORDINATES; THE THIRD IS THE DEPTH.\n");
    do {
        print("\n");
        print("GOOD LUCK !\n");
        print("\n");
        a = Math.floor(Math.random() * g);
        b = Math.floor(Math.random() * g);
        c = Math.floor(Math.random() * g);
        for (d = 1; d <= n; d++) {
            print("\n");
            print("TRIAL #" + d + " ");
            str = await input();
            x = parseInt(str);
            y = parseInt(str.substr(str.indexOf(",") + 1));
            z = parseInt(str.substr(str.lastIndexOf(",") + 1));
            if (Math.abs(x - a) + Math.abs(y - b) + Math.abs(z - c) == 0)
                break;
            if (y > b)
                print("NORTH");
            if (y < b)
                print("SOUTH");
            if (x > a)
                print("EAST");
            if (x < a)
                print("WEST");
            if (y != b || x != a)
                print(" AND");
            if (z > c)
                print(" TOO LOW.\n");
            if (z < c)
                print(" TOO HIGH.\n");
            if (z == c)
                print(" DEPTH OK.\n");
            print("\n");
        }
        if (d <= n) {
            print("\n");
            print("B O O M ! ! YOU FOUND IT IN " + d + " TRIES!\n");
        } else {
            print("\n");
            print("YOU HAVE BEEN TORPEDOED!  ABANDON SHIP!\n");
            print("THE SUBMARINE WAS AT " + a + "," + b + "," + c + "\n");
        }
        print("\n");
        print("\n");
        print("ANOTHER GAME (Y OR N)");
        str = await input();
    } while (str.substr(0, 1) == "Y") ;
    print("OK.  HOPE YOU ENJOYED YOURSELF.\n");
}

```

这是C++中的一个标准的main函数，其作用是程序的入口点。在main函数中，程序开始执行，所有任务和代码都在一个统一的位置开始。所以，main函数是C++应用程序的第一个入口点。


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

## Conversion

Not a difficult conversion - but a chance to throw in a few ways
Perl makes life easy.

 * To get the sub permission which is a random location in the g x g x g grid we can use:
   * assigning multiple variables in list form ($a,$b,$c) = (?,?,?)
   * where the list on the right hand side is generated with a map function

 * We use ternarys to generate the message if you miss the sub.
   * We use join to stitch the pieces of the string together.
   * If we have a ternary where we don't want to return anything we return an empty list rather than an empty string - if you return the latter you still get the padding spaces.


# `31_Depth_Charge/python/depth_charge.py`

这段代码是一个用于显示欢迎消息的 Python 函数。它的目的是在游戏开始时显示一段欢迎消息，告诉玩家这是一款游戏，并提供了如何与控制器交互。

具体来说，这段代码实现了以下功能：

1. 导入了 `math` 和 `random` 模块，这些模块对数学和随机数进行操作。
2. 定义了一个名为 `show_welcome` 的函数，这个函数接受一个空括号作为参数。
3. 在函数内部，使用了 `print` 函数来输出文本字符串，使用了 `chr` 函数来输出控制台字符。`27` 表示控制台上的 `ESC` 键，`"Erase in Display"` 表示 `ERASING_SCREEN` 游戏内输出窗口，`"Creative Computing"` 表示游戏制作公司。
4. 使用了 `print` 函数的中心化函数 `center`，将文本字符串 `"DEPTH CHARGE"` 居中显示。
5. 最后，`print` 函数中的字符串 `"Creative Computing  Morristown, New Jersey"` 是游戏内输出窗口中的公司名称。

整个函数的作用是在游戏开始时输出欢迎消息，告诉玩家这是一款游戏，并提供了如何与控制器交互。


```
"""
Original BASIC version as published in Basic Computer Games (1978)
https://www.atariarchives.org/basicgames/showpage.php?page=55

Converted to Python by Anson VanDoren in 2021
"""

import math
import random
from typing import Tuple


def show_welcome() -> None:
    # Clear screen. chr(27) is `Esc`, and the control sequence is
    # initiated by Ctrl+[
    # `J` is "Erase in Display" and `2J` means clear the entire screen
    print(chr(27) + "[2J")

    # Show the intro text, centered
    print("DEPTH CHARGE".center(45))
    print("Creative Computing  Morristown, New Jersey\n\n".center(45))


```

这段代码是一个 Python 函数，名为 `get_num_charges`，它返回一个元组 `(search_area, num_charges)`。函数的作用是用于计算一个搜索区域的深度，然后返回这个搜索区域的尺寸和深度。

函数首先输出一段游戏提示，然后进入一个无限循环。在循环中，函数要求用户输入搜索区域的形状（比如 "请输入一个正方形的边长" ）。如果用户输入的形状不是数字，函数会提示并重新请求输入。如果用户输入的形状是数字，函数会将搜索区域的尺寸设置为用户输入的值，并返回这个值。

函数内部使用了一个有趣的技巧，将搜索区域的尺寸计算为 log2(search\_area) + 1。这是因为，对于一个搜索区域，它的对数一定是一个搜索区域的大小（不包括边界）。因此，将搜索区域的尺寸计算为 log2(search\_area) 确保了计算的正确性。此外，由于搜索区域的尺寸必须是一个整数，因此我们需要在计算时进行整除，以确保得到一个整数结果。


```
def get_num_charges() -> Tuple[int, int]:
    print("Depth Charge game\n")
    while True:
        search_area_str = input("Dimensions of search area? ")

        # Make sure the input is an integer
        try:
            search_area = int(search_area_str)
            break
        except ValueError:
            print("Must enter an integer number. Please try again...")

    num_charges = int(math.log2(search_area)) + 1
    return search_area, num_charges


```

这道代码定义了两个函数，分别是 `ask_for_new_game()` 和 `show_shot_result()`。这两个函数的作用是询问用户是否想要开始新的游戏，以及在游戏过程中检测 shot（一个包含两个坐标点的列表，通常表示位置）是否在指定位置的左边或右边。

具体来说，`ask_for_new_game()` 函数的作用是获取用户输入，并判断输入是否为 "Y" 或 "N"。如果是 "Y"，则进入游戏主函数，否则输出 "OK. Hope you enjoyed yourself" 并退出游戏。

`show_shot_result()` 函数的作用是在 shot 的位置检测是否在指定位置的左边或右边。具体来说，该函数会检测 shot 中的两个坐标点是否在指定位置的左边或右边。如果是左边，函数会输出 "too low"；如果是右边，函数会输出 "too high"。如果 shot 中的两个坐标点不在指定位置的左边或右边，函数会输出 "depth OK"，并在结果中添加 "and"。最后，函数会输出结果，并返回结果。


```
def ask_for_new_game() -> None:
    answer = input("Another game (Y or N): ")
    if answer.lower().strip()[0] == "y":
        main()
    else:
        print("OK. Hope you enjoyed yourself")
        exit()


def show_shot_result(shot, location) -> None:
    result = "Sonar reports shot was "
    if shot[1] > location[1]:  # y-direction
        result += "north"
    elif shot[1] < location[1]:  # y-direction
        result += "south"
    if shot[0] > location[0]:  # x-direction
        result += "east"
    elif shot[0] < location[0]:  # x-direction
        result += "west"
    if shot[1] != location[1] or shot[0] != location[0]:
        result += " and "

    if shot[2] > location[2]:
        result += "too low."
    elif shot[2] < location[2]:
        result += "too high."
    else:
        result += "depth OK."
    print(result)
    return


```

这段代码定义了一个名为 `get_shot_input` 的函数，其返回值为三个整数类型的 Tuple。函数会在一个无限循环中等待用户输入坐标，如果输入的格式不正确，函数会提示用户并尝试纠正错误。如果用户输入的坐标格式正确，函数会返回对应的整数类型的值。

函数的逻辑主要分为以下几个步骤：

1. 接收用户输入的坐标，并将输入的字符串按照空格分隔。
2. 将输入的字符串转换成整数类型。
3. 在尝试将整数值转换为坐标的情况下，如果发生错误，函数会提示用户并尝试纠正错误。
4. 返回转换后的坐标值。

函数的作用是获取用户输入的坐标并返回它们，以便在程序中进行后续的处理和使用。


```
def get_shot_input() -> Tuple[int, int, int]:
    while True:
        raw_guess = input("Enter coordinates: ")
        try:
            xyz = raw_guess.split()
        except ValueError:
            print("Please enter coordinates separated by spaces")
            print("Example: 3 2 1")
            continue
        try:
            x, y, z = (int(num) for num in xyz)
            return x, y, z
        except ValueError:
            print("Please enter whole numbers only")


```

这段代码是一个 Python 函数，名为 `play_game`，它用于在给定的深度区域和攻击次数下进行一次简单的游戏。在这个游戏中，玩家扮演美国海军的电脑角色，任务是摧毁敌方潜艇。

函数首先向玩家描述了情境，然后询问玩家想要进行多少轮射击。每次射击后，函数会生成一个新的随机位置，用于下一次射击。如果玩家在指定轮数内找到了潜艇，游戏结束，玩家将获得胜利。否则，游戏继续进行，直到玩家无法在指定轮数内摧毁潜艇。

请注意，这段代码中存在一些问题。例如，玩家只能进行一次射击，而游戏会在玩家未能在指定轮数内摧毁潜艇时结束，这意味着玩家可能在游戏中失败。此外，游戏还会在每次射击后清空所有位置，这意味着玩家将需要重新生成它们。


```
def play_game(search_area, num_charges) -> None:
    print("\nYou are the captain of the destroyer USS Computer.")
    print("An enemy sub has been causing you trouble. Your")
    print(f"mission is to destroy it. You have {num_charges} shots.")
    print("Specify depth charge explosion point with a")
    print("trio of numbers -- the first two are the")
    print("surface coordinates; the third is the depth.")
    print("\nGood luck!\n")

    # Generate position for submarine
    a, b, c = (random.randint(0, search_area) for _ in range(3))

    # Get inputs until win or lose
    for i in range(num_charges):
        print(f"\nTrial #{i+1}")
        x, y, z = get_shot_input()

        if (x, y, z) == (a, b, c):
            print(f"\nB O O M ! ! You found it in {i+1} tries!\n")
            ask_for_new_game()
        else:
            show_shot_result((x, y, z), (a, b, c))

    # out of shots
    print("\nYou have been torpedoed! Abandon ship!")
    print(f"The submarine was at {a} {b} {c}")
    ask_for_new_game()


```

这段代码是一个Python程序，名为“main”。它定义了一个函数“main”，函数内包含一个空括号“())”，这表示函数可以接受不带参数的执行。

函数体内部，首先通过一个名为“get_num_charges”的函数取得充电区域数量和充电次数，然后将这些数据存储在变量“search_area”和“num_charges”中。

接下来，调用一个名为“play_game”的函数，该函数接收“search_area”和“num_charges”作为参数。这个函数可能执行与游戏相关的操作，例如移动地图上的游戏角色或者处理用户输入的事件。

最后，程序进入了主函数“__main__”，这意味着如果不带参数运行程序，它将调用上面定义的函数。然而，在这个特殊的运行时环境中，程序不会输出任何值或进行任何实际的操作。


```
def main() -> None:
    search_area, num_charges = get_num_charges()
    play_game(search_area, num_charges)


if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Diamond

This program fills an 8.5x11 piece of paper with diamonds (plotted on a hard-copy terminal, of course). The program asks for an odd number to be input in the range 5 to 31. The diamonds printed will be this number of characters high and wide. The number of diamonds across the page will vary from 12 for 5-character wide diamonds to 1 for a diamond 31-characters wide. You can change the content of the pattern if you wish.

The program was written by David Ahl of Creative Computing.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=56)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=71)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `32_Diamond/csharp/Pattern.cs`



该代码是一个钻石（Diamond）模式的类。通过使用钻石模式（Diamond Pattern）来创建和绘制图形。以下是代码的功能和用途：

1. 构造函数：创建一个新的钻石模式实例，将一个 `IReadWrite` 类型的数据源（可能是文件或用户输入）包装到 `Pattern` 类的实例中。

2. `draw` 方法：绘制钻石模式，将钻石的大小存储在 `diamondSize` 变量中，并输出钻石的轮廓。然后，根据钻石大小计算出钻石线数，将钻石线存储在 `diamondLines` 列表中。最后，使用循环来遍历钻石线并输出。

3. `GetDiamondLines` 方法：根据钻石的大小，使用 `for` 循环来生成钻石线，并将钻石线存储在 `patternLines` 列表中。

4. `patternLines` 的生成：该方法根据钻石的大小，通过调整钻石线的长度来生成钻石线。首先，计算出钻石线的长度 `patternLinesCount`。然后，对于每个钻石线 `patternLines[diamondRow]`，生成以其长度为基数的字符串。最后，将这些字符串添加到 `patternLines` 列表中。

5. `Diamond` 类：该类实现了钻石模式，提供了生成和绘制钻石模式的方法。其构造函数接收一个 `IReadWrite` 类型的数据源，并设置钻石模式的数据源。`draw` 方法绘制钻石模式，然后生成并输出钻石线。`GetDiamondLines` 方法用于生成钻石线列表，`patternLines` 方法用于生成钻石线，`patternLinesCount` 方法用于确定钻石线的长度。


```
using System.Text;
using static Diamond.Resources.Resource;

namespace Diamond;

internal class Pattern
{
    private readonly IReadWrite _io;

    public Pattern(IReadWrite io)
    {
        _io = io;
        io.Write(Streams.Introduction);
    }

    public void Draw()
    {
        var diamondSize = _io.ReadNumber(Prompts.TypeNumber);
        _io.WriteLine();

        var diamondCount = (int)(60 / diamondSize);

        var diamondLines = new List<string>(GetDiamondLines(diamondSize)).AsReadOnly();

        for (int patternRow = 0; patternRow < diamondCount; patternRow++)
        {
            for (int diamondRow = 0; diamondRow < diamondLines.Count; diamondRow++)
            {
                var line = new StringBuilder();
                for (int patternColumn = 0; patternColumn < diamondCount; patternColumn++)
                {
                    line.PadToLength((int)(patternColumn * diamondSize)).Append(diamondLines[diamondRow]);
                }
                _io.WriteLine(line);
            }
        }
    }

    public static IEnumerable<string> GetDiamondLines(float size)
    {
        for (var i = 1; i <= size; i += 2)
        {
            yield return GetLine(i);
        }

        for (var i = size - 2; i >= 1; i -= 2)
        {
            yield return GetLine(i);
        }

        string GetLine(float i) =>
            string.Concat(
                new string(' ', (int)(size - i) / 2),
                new string('C', Math.Min((int)i, 2)),
                new string('!', Math.Max(0, (int)i - 2)));
    }
}

```

# `32_Diamond/csharp/Program.cs`

这段代码的作用是创建一个自定义的输入输出类（Pattern），然后使用该类的一个实例来输出在游戏窗口上绘制一个模式（draw）的形状。

具体来说，这段代码包含以下几个部分：

1. 使用 `Games.Common.IO` 命名空间中的 `new` 关键字来创建一个自定义类 `Pattern`，该类中包含一个 `draw` 方法。

2. 在 `Pattern` 类的 `draw` 方法中，使用 `new` 关键字来创建一个 `ConsoleIO` 的实例，然后将其传递给 `Pattern` 类的 `draw` 方法中，这样就可以使用 `console.out` 流来输出图形数据。

3. 在 `Pattern` 类的 `draw` 方法中，使用 `Pattern` 类的实例来创建一个新的 `Pattern` 对象，并在其 `draw` 方法的结束部分使用 `.` 符号来指定要在哪些位置绘制图形。

4. 最后，使用 `Pattern` 类的实例来调用其 `draw` 方法，这样就可以在游戏窗口上绘制一个模式。


```
global using Games.Common.IO;
using Diamond;

new Pattern(new ConsoleIO()).Draw();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `32_Diamond/csharp/StringBuilderExtensions.cs`



这段代码是一个名为`StringBuilderExtensions`的类，其目的是扩展`StringBuilder`类以实现向指定长度的字符串进行填充。

具体来说，这个类包含一个名为`PadToLength`的内部静态方法，其接受一个表示要填充的长度的参数。这个方法使用`builder.Append(' ', length - builder.Length)`来在字符串的结尾向左填充空格，直到长度达到目标长度。最后，这个方法返回一个新的`StringBuilder`对象，其中包含填充后的字符串。

这个代码示例可以用来在一个需要填充字符串的场景中使用，例如在写文章时，可以使用`PadToLength`来确保文章的长度不会超过所需的最大长度。


```
using System.Text;

namespace Diamond;

internal static class StringBuilderExtensions
{
    internal static StringBuilder PadToLength(this StringBuilder builder, int length) => 
        builder.Append(' ', length - builder.Length);
}
```

# `32_Diamond/csharp/Resources/Resource.cs`



这段代码是一个自定义的 Diamond.Resources 命名空间中的类，其中包括两个内部类 Streams 和 Prompts，以及一个私有类 Resource。

Streams 类包含一个 Introduction 字段，它是一个静态的内部类，这个类也被称为 Stream，它提供了一个字符串的输入，然后返回一个 Stream 对象。

Prompts 类包含一个 TypeNumber 字段，它是一个静态的内部类，它提供了一个字符串的输入，然后返回一个字符串。

Resource 类包含一个内部类 StreamReader，它实现了 IServiceProvider，它使用 C# 8 的特性，通过服务提供程序提供服务。它包含一个静态的 GetString 方法，它使用传入的 name 参数，从定义中查找一个名为 Stream 的资源，并返回它的入口点。

最后，Assembly.GetExecutingAssembly().GetManifestResourceStream方法用于获取应用程序的资源文件，如果指定的名称包含 "." 那么将返回应用程序的默认根目录。


```
using System.Reflection;
using System.Runtime.CompilerServices;

namespace Diamond.Resources;

internal static class Resource
{
    internal static class Streams
    {
        public static Stream Introduction => GetStream();
    }

    internal static class Prompts
    {
        public static string TypeNumber => GetString();
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

# `32_Diamond/java/Diamond.java`

This is a Java program that simulates a game of diamond. Diamond is a card game where two to eight players, each of whom tries to佩顿钻石， is played with 22 cards.

The program starts by defining a Diamond object named `diamond`. Then, it executes a while loop that increments the diamond's body by either 2 or -2, depending on whether it is the top or bottom half of the diamond, and updates the user's turn to the next half.

Next, it loops through each row of diamonds, starting with the top row, and fills in the钻石's body with the appropriate characters. For the bottom row, it adds spaces instead of filling in the diamond's body.

After the while loop, it prints out the final result of the game, taking into account whether the user won or lost.

The program also includes code that prints out the top half of each diamond, starting from the user's turn. This is done using a for loop that fills in the diamond's body with the appropriate characters for the top half, and then prints out the result.

Overall, this program simulates the game of diamond, allowing the user to decide on their turn and the size of the diamond, and outputs the final result of the game.


```
import java.util.Scanner;

/**
 * Game of Diamond
 * <p>
 * Based on the BASIC game of Diamond here
 * https://github.com/coding-horror/basic-computer-games/blob/main/32%20Diamond/diamond.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 *
 * Converted from BASIC to Java by Darren Cardenas.
 */

public class Diamond {

  private static final int LINE_WIDTH = 60;

  private static final String PREFIX = "CC";

  private static final char SYMBOL = '!';

  private final Scanner scan;  // For user input


  public Diamond() {

    scan = new Scanner(System.in);

  }  // End of constructor Diamond


  public void play() {

    showIntro();
    startGame();

  }  // End of method play


  private void showIntro() {

    System.out.println(" ".repeat(32) + "DIAMOND");
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    System.out.println("\n\n");

  }  // End of method showIntro


  private void startGame() {

    int body = 0;
    int column = 0;
    int end = 0;
    int fill = 0;
    int increment = 2;
    int numPerSide = 0;
    int prefixIndex = 0;
    int row = 0;
    int start = 1;
    int userNum = 0;

    String lineContent = "";

    // Get user input
    System.out.println("FOR A PRETTY DIAMOND PATTERN,");
    System.out.print("TYPE IN AN ODD NUMBER BETWEEN 5 AND 21? ");
    userNum = scan.nextInt();
    System.out.println("");

    // Calcuate number of diamonds to be drawn on each side of screen
    numPerSide = (int) (LINE_WIDTH / userNum);

    end = userNum;

    // Begin loop through each row of diamonds
    for (row = 1; row <= numPerSide; row++) {

      // Begin loop through top and bottom halves of each diamond
      for (body = start; increment < 0 ? body >= end : body <= end; body += increment) {

        lineContent = "";

        // Add whitespace
        while (lineContent.length() < ((userNum - body) / 2)) {
          lineContent += " ";
        }

        // Begin loop through each column of diamonds
        for (column = 1; column <= numPerSide; column++) {

          prefixIndex = 1;

          // Begin loop that fills each diamond with characters
          for (fill = 1; fill <= body; fill++) {

            // Right side of diamond
            if (prefixIndex > PREFIX.length()) {

              lineContent += SYMBOL;

            }
            // Left side of diamond
            else {

              lineContent += PREFIX.charAt(prefixIndex - 1);
              prefixIndex++;

            }

          }  // End loop that fills each diamond with characters

          // Column finished
          if (column == numPerSide) {

            break;

          }
          // Column not finishd
          else {

            // Add whitespace
            while (lineContent.length() < (userNum * column + (userNum - body) / 2)) {
              lineContent += " ";
            }

          }

        }  // End loop through each column of diamonds

        System.out.println(lineContent);

      }  // End loop through top and bottom half of each diamond

      if (start != 1) {

        start = 1;
        end = userNum;
        increment = 2;

      }
      else {

        start = userNum - 2;
        end = 1;
        increment = -2;
        row--;

      }

    }  // End loop through each row of diamonds

  }  // End of method startGame


  public static void main(String[] args) {

    Diamond diamond = new Diamond();
    diamond.play();

  }  // End of method main

}  // End of class Diamond

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `32_Diamond/javascript/diamond.js`

这段代码定义了两个函数，分别是`print()`和`input()`。

`print()`函数的作用是在页面上打印一段字符串，将字符串打印到页面上一个有一个`<textarea>`元素的`output`元素中。这个 `<textarea>` 元素在网页上是一个文本域，当用户输入文本并点击提交按钮时，将文本发送到服务器进行处理。

`input()`函数的作用是从用户那里获取一个字符串，用于在页面上显示。它通过与用户交互来获取字符串，然后将其存储在变量`input_str`中。函数通过使用`document.getElementById()`和`setAttribute()`函数获取用户输入的值，然后使用`addEventListener()`函数来监听用户输入的键盘事件。当用户按下键盘上的13时，函数会捕获到键盘事件并打印字符串到页面上，然后将其从变量中删除并打印出来。


```
// DIAMOND
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



这段代码是一个名为`tab`的函数，它会将一个字符串`str`中的空格去掉，并返回去掉空格后的字符串。

函数体中，首先定义了一个字符串变量`str`，然后使用了一个`while`循环和一个变量`space`。`space`从0开始递增，每次递增1，当`space`大于0时，循环会继续执行。在循环中，`str`字符串会逐渐添加，每次添加一个空格。当`space`为0时，循环停止。

函数返回字符串`str`。

main函数中，首先调用了一个名为`tab`的函数，并传递了一个参数`33`，它代表需要打印的字符串长度。函数返回一个字符串，打印出来。然后，又调用了另一个名为`tab`的函数，传递了一个参数`15`，同样代表需要打印的字符串长度。函数返回一个字符串，打印出来。接下来，打印了两个横线。最后，又调用了一个名为`tab`的函数，传递了一个参数`33`，代表需要打印的字符串长度。函数返回一个字符串，打印出来。

这段代码的作用是打印出一些用于制作钻石图案的字符串，其中每个字符串由多个空格和数字组成。每个数字代表了字符串中出现的字符数量，数字的范围在5到21之间。


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
    print(tab(33) + "DIAMOND\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("FOR A PRETTY DIAMOND PATTERN,\n");
    print("TYPE IN AN ODD NUMBER BETWEEN 5 AND 21");
    r = parseInt(await input());
    q = Math.floor(60 / r);
    as = "CC"
    x = 1;
    y = r;
    z = 2;
    for (l = 1; l <= q; l++) {
        for (n = x; z < 0 ? n >= y : n <= y; n += z) {
            str = "";
            while (str.length < (r - n) / 2)
                str += " ";
            for (m = 1; m <= q; m++) {
                c = 1;
                for (a = 1; a <= n; a++) {
                    if (c > as.length)
                        str += "!";
                    else
                        str += as[c++ - 1];
                }
                if (m == q)
                    break;
                while (str.length < r * m + (r - n) / 2)
                    str += " ";
            }
            print(str + "\n");
        }
        if (x != 1) {
            x = 1;
            y = r;
            z = 2;
        } else {
            x = r - 2;
            y = 1;
            z = -2;
            l--;
        }
    }
}

```

这道题目是一个不完整的程序，缺少了程序的具体内容。它包含了一个名为 "main" 的函数，但缺乏函数体。因此，我们无法得知它将要执行哪些操作。通常，一个完整的程序会包含一系列的函数，每个函数实现特定的功能。在没有函数体的情况下，我们无法确定 main 函数将会执行哪些操作。


```
main();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/), structure inspired from [the Java port](https://github.com/coding-horror/basic-computer-games/blob/main/32_Diamond/java/Diamond.java).

### How to Run
1. Install [kotlin command line](https://kotlinlang.org/docs/command-line.html) compiler from JetBrains.
2. Compile with `kotlinc diamond.kt -include-runtime -d diamond.jar`
3. Run with `java -jar diamond.jar`

### Changes from Original
This version validates that user input is correct.



Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


# `32_Diamond/python/diamond.py`

这段代码定义了一个名为`print_diamond`的函数，它接受五个参数：

1. `begin_width`：打印开始的位置，单位为宽度（从0开始计算）。
2. `end_width`：打印结束的位置，单位为宽度（从0开始计算）。
3. `step`：步长，单位为宽度（从1开始计算，例如2表示从0开始向右平移两个宽度）。
4. `width`：要打印的文本字符串的宽度（不包括结束字符串）。
5. `count`：重复的边缘字符的数量。

函数内部首先定义了一个`edge_string`变量，然后定义了一个`fill`变量。接着，从`begin_width`开始循环，每次循环计算出要打印的行数`line_buffer`，然后打印该行字符串。在循环外，还需要打印一个空行。然后从`end_width`开始循环，每次循环打印一个`!"`字符。循环内部，如果当前`a`的值大于`edge_string`的长度，则需要打印`fill`字符，否则需要打印`edge_string`字符。最后，循环结束后，将`n`加到`end_width`上，以便在打印下一行时从正确的位置开始。


```
"""
DIAMOND

Prints pretty diamond patterns to the screen.

Ported by Dave LeCompte
"""


def print_diamond(begin_width, end_width, step, width, count) -> None:
    edge_string = "CC"
    fill = "!"

    n = begin_width
    while True:
        line_buffer = " " * ((width - n) // 2)
        for across in range(count):
            for a in range(n):
                if a >= len(edge_string):
                    line_buffer += fill
                else:
                    line_buffer += edge_string[a]
            line_buffer += " " * (
                (width * (across + 1) + (width - n) // 2) - len(line_buffer)
            )
        print(line_buffer)
        if n == end_width:
            return
        n += step


```

这段代码是一个Python程序，主要目的是输出一个美钻图案，并接收用户输入的大小。

具体来说，程序首先定义了一个名为main的函数，它接受一个空括号作为参数，并返回一个空括号。

接着，程序输出一个空的9英寸（约合23厘米）宽的页面，然后输出一个钻石形状的图案，其中有几行几列的钻石形状。接着程序提示用户输入美钻图案的大小，然后根据输入的大小计算出需要输出的行数或列数，并循环打印钻石形状的每一行。

程序输出的钻石图案是一个具有对称性的多行多列图案，其中钻石形状的每个正方形边长从5到21不等。程序通过循环来逐步输出钻石形状的每一行，并在输出的最后两行添加额外的钻石形状，以使图案显得更加完美。


```
def main() -> None:
    print(" " * 33, "DIAMOND")
    print(" " * 15, "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")
    print("FOR A PRETTY DIAMOND PATTERN,")
    print("TYPE IN AN ODD NUMBER BETWEEN 5 AND 21")
    width = int(input())
    print()

    PAGE_WIDTH = 60

    count = int(PAGE_WIDTH / width)

    for _down in range(count):
        print_diamond(1, width, 2, width, count)
        print_diamond(width - 2, 1, -2, width, count)

    print()
    print()


```

这段代码是一个if语句，它的作用是判断当前脚本是否被情悬挂给了脚本模块(即是否作为了一个module)。如果当前脚本被情悬挂给了模块，那么该if语句将跳转到__main__函数中执行。

换句话说，这段代码会检查当前脚本是否是一个真正的模块，如果是，那么将跳转到__main__函数中执行该脚本。__main__函数是Python中的一个特殊函数，只有在当前脚本被情悬挂给了模块时才会被执行。

因此，这段代码的作用是用于确保脚本只能在__main__函数中执行，而不能在常规的模块中执行。


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


### Dice

Not exactly a game, this program simulates rolling a pair of dice a large number of times and prints out the frequency distribution. You simply input the number of rolls. It is interesting to see how many rolls are necessary to approach the theoretical distribution:

|   |      |            |
|---|------|------------|
| 2 | 1/36 | 2.7777...% |
| 3 | 2/36 | 5.5555...% |
| 4 | 3/36 | 8.3333...% |
etc.

Daniel Freidus wrote this program while in the seventh grade at Harrison Jr-Sr High School, Harrison, New York.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=57)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=72)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `33_Dice/csharp/Game.cs`

This is a C# class that appears to be the controller for a game of乾隆八戒。 It has a method called `TryAgain()`, which prompts the player to try again if they haven't finished yet, and a method called `CountRolls()`, which returns the number of rolls made by the player so far.

The `TryAgain()` method displays a message asking the player if they want to try again, and then prompts the player to press either 'Y' or 'N' to confirm their decision. If the player presses 'Y', the method returns `true`, otherwise it returns `false`.

The `CountRolls()` method reads the key pressed by the player, converts it to uppercase, and checks if it's the key for trying again. If it's not, the method continues with the looping process. If it is the key for trying again, the method returns `true`, otherwise it returns `false`.

Overall, this class seems to be setting up a simple game of乾隆八戒， where the player can try again to continue playing after a roll.


```
﻿using System;
using System.Linq;

namespace BasicComputerGames.Dice
{
	public class Game
	{
		private readonly RollGenerator _roller = new RollGenerator();

		public void GameLoop()
		{
			DisplayIntroText();

			// RollGenerator.ReseedRNG(1234);		// hard-code seed for repeatabilty during testing

			do
			{
				int numRolls = GetInput();
				var counter = CountRolls(numRolls);
				DisplayCounts(counter);
			} while (TryAgain());
		}

		private void DisplayIntroText()
		{
			Console.ForegroundColor = ConsoleColor.Yellow;
			Console.WriteLine("Dice");
			Console.WriteLine("Creating Computing, Morristown, New Jersey."); Console.WriteLine();

			Console.ForegroundColor = ConsoleColor.DarkGreen;
			Console.WriteLine("Original code by Danny Freidus.");
			Console.WriteLine("Originally published in 1978 in the book 'Basic Computer Games' by David Ahl.");
			Console.WriteLine("Modernized and converted to C# in 2021 by James Curran (noveltheory.com).");
			Console.WriteLine();

			Console.ForegroundColor = ConsoleColor.Gray;
			Console.WriteLine("This program simulates the rolling of a pair of dice.");
			Console.WriteLine("You enter the number of times you want the computer to");
			Console.WriteLine("'roll' the dice. Watch out, very large numbers take");
			Console.WriteLine("a long time. In particular, numbers over 10 million.");
			Console.WriteLine();

			Console.ForegroundColor = ConsoleColor.Yellow;
			Console.WriteLine("Press any key start the game.");
			Console.ReadKey(true);
		}

		private int GetInput()
		{
			int num = -1;
			Console.WriteLine();
			do
			{
				Console.WriteLine();
				Console.Write("How many rolls? ");
			} while (!Int32.TryParse(Console.ReadLine(), out num));

			return num;
		}

		private  void DisplayCounts(int[] counter)
		{
			Console.WriteLine();
			Console.WriteLine($"\tTotal\tTotal Number");
			Console.WriteLine($"\tSpots\tof Times");
			Console.WriteLine($"\t===\t=========");
			for (var n = 1; n < counter.Length; ++n)
			{
				Console.WriteLine($"\t{n + 1,2}\t{counter[n],9:#,0}");
			}
			Console.WriteLine();
		}

		private  int[] CountRolls(int x)
		{
			var counter = _roller.Rolls().Take(x).Aggregate(new int[12], (cntr, r) =>
			{
				cntr[r.die1 + r.die2 - 1]++;
				return cntr;
			});
			return counter;
		}
		/// <summary>
		/// Prompt the player to try again, and wait for them to press Y or N.
		/// </summary>
		/// <returns>Returns true if the player wants to try again, false if they have finished playing.</returns>
		private bool TryAgain()
		{
			Console.ForegroundColor = ConsoleColor.White;
			Console.WriteLine("Would you like to try again? (Press 'Y' for yes or 'N' for no)");

			Console.ForegroundColor = ConsoleColor.Yellow;
			Console.Write("> ");

			char pressedKey;
			// Keep looping until we get a recognised input
			do
			{
				// Read a key, don't display it on screen
				ConsoleKeyInfo key = Console.ReadKey(true);
				// Convert to upper-case so we don't need to care about capitalisation
				pressedKey = Char.ToUpper(key.KeyChar);
				// Is this a key we recognise? If not, keep looping
			} while (pressedKey != 'Y' && pressedKey != 'N');
			// Display the result on the screen
			Console.WriteLine(pressedKey);

			// Return true if the player pressed 'Y', false for anything else.
			return (pressedKey == 'Y');
		}
	}
}

```

# `33_Dice/csharp/Program.cs`

这段代码定义了一个名为 "BasicComputerGames.Dice" 的命名空间，其中包含一个名为 "Program" 的类，该类包含一个名为 "Main" 的方法。

在 "Main" 方法中，使用 new 关键字创建了一个名为 "game" 的对象，并将其赋值为一个名为 "Game" 的类实例。

然后，使用 Game 类的 "GameLoop" 函数来让游戏不断运行，直到玩家选择退出为止。


```
﻿namespace BasicComputerGames.Dice
{
	public class Program
	{
		public static void Main(string[] args)
		{
			// Create an instance of our main Game class
			Game game = new Game();

			// Call its GameLoop function. This will play the game endlessly in a loop until the player chooses to quit.
			game.GameLoop();
		}
	}
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/) by James Curran (http://www.noveltheory.com)


# `33_Dice/csharp/RollGenerator.cs`

这段代码定义了一个名为 `RollGenerator` 的类，用于生成随机结果。

首先，在类中定义了一个名为 `_rnd` 的静态变量，其初始值设置为一个随机数生成器类中的 `Random` 类实例。

接着，定义了一个名为 `ReseedRNG` 的静态方法，该方法接受一个整数参数 `seed`，用于重新生成随机数种子。

然后，定义了一个名为 `Rolls` 的静态方法，该方法使用一个 while 循环，每次生成两张牌的随机结果，并将结果 yield 出来。在循环内部，使用 `_rnd.Next(1, 7)` 生成一个 1 到 7 之间的随机整数，用于选择牌的类型。然后，再次使用 `_rnd.Next(1, 7)` 生成一个 1 到 7 之间的随机整数，用于选择牌的数量。

最后，在 `Rolls` 方法的循环体内，使用 `yield return` 方法将生成结果返回，每次返回两张牌的结果，即 `(int die1, int die2)`。


```
﻿using System;
using System.Collections.Generic;

namespace BasicComputerGames.Dice
{
	public class RollGenerator
	{
		static Random _rnd = new Random();

		public static void ReseedRNG(int seed) => _rnd = new Random(seed);

		public IEnumerable<(int die1, int die2)> Rolls()
		{
			while (true)
			{
				yield return (_rnd.Next(1, 7), _rnd.Next(1, 7));
			}
		}
	}
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `33_Dice/java/src/Dice.java`

**Update:**

This code appears to be a Java class named `QuestionPlayer`. It contains methods for displaying a message on the screen, accepting input from the keyboard, converting input to an integer, and checking whether the player entered "Y" or "YES" in response to a question.

The `displayTextAndGetNumber()` method displays a message on the screen and accepts input from the keyboard. It then converts the input to an integer using the `Integer.parseInt()` method.

The `displayTextAndGetInput()` method displays a message on the screen, accepts input from the keyboard, and returns the input as a `String`.

The `yesEntered()` method checks whether the player entered "Y" or "YES" in response to a question. It does this by checking whether the input is equal to any of the values passed as an array.

The `stringIsAnyValue()` method checks whether a given string is equal to one of a variable number of values. It uses the `Arrays.stream()` method to iterate over the values and the `anyMatch()` method to check for a match.


```
import java.util.Arrays;
import java.util.Scanner;

/**
 * Game of Dice
 * <p>
 * Based on the Basic game of Dice here
 * https://github.com/coding-horror/basic-computer-games/blob/main/33%20Dice/dice.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Dice {

    // Used for keyboard input
    private final Scanner kbScanner;

    private enum GAME_STATE {
        START_GAME,
        INPUT_AND_CALCULATE,
        RESULTS,
        GAME_OVER
    }

    // Current game state
    private GAME_STATE gameState;

    private int[] spots;

    public Dice() {
        kbScanner = new Scanner(System.in);

        gameState = GAME_STATE.START_GAME;
    }

    /**
     * Main game loop
     */
    public void play() {

        do {
            switch (gameState) {

                case START_GAME:
                    intro();
                    spots = new int[12];
                    gameState = GAME_STATE.INPUT_AND_CALCULATE;
                    break;

                case INPUT_AND_CALCULATE:

                    int howManyRolls = displayTextAndGetNumber("HOW MANY ROLLS? ");
                    for (int i = 0; i < howManyRolls; i++) {
                        int diceRoll = (int) (Math.random() * 6 + 1) + (int) (Math.random() * 6 + 1);
                        // save dice roll in zero based array
                        spots[diceRoll - 1]++;
                    }
                    gameState = GAME_STATE.RESULTS;
                    break;

                case RESULTS:
                    System.out.println("TOTAL SPOTS" + simulateTabs(8) + "NUMBER OF TIMES");
                    for (int i = 1; i < 12; i++) {
                        // show output using zero based array
                        System.out.println(simulateTabs(5) + (i + 1) + simulateTabs(20) + spots[i]);
                    }
                    System.out.println();
                    if (yesEntered(displayTextAndGetInput("TRY AGAIN? "))) {
                        gameState = GAME_STATE.START_GAME;
                    } else {
                        gameState = GAME_STATE.GAME_OVER;
                    }
                    break;
            }
        } while (gameState != GAME_STATE.GAME_OVER);
    }

    private void intro() {
        System.out.println(simulateTabs(34) + "DICE");
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("THIS PROGRAM SIMULATES THE ROLLING OF A");
        System.out.println("PAIR OF DICE.");
        System.out.println("YOU ENTER THE NUMBER OF TIMES YOU WANT THE COMPUTER TO");
        System.out.println("'ROLL' THE DICE.  WATCH OUT, VERY LARGE NUMBERS TAKE");
        System.out.println("A LONG TIME.  IN PARTICULAR, NUMBERS OVER 5000.");
    }

    /*
     * Print a message on the screen, then accept input from Keyboard.
     * Converts input to an Integer
     *
     * @param text message to be displayed on screen.
     * @return what was typed by the player.
     */
    private int displayTextAndGetNumber(String text) {
        return Integer.parseInt(displayTextAndGetInput(text));
    }

    /*
     * Print a message on the screen, then accept input from Keyboard.
     *
     * @param text message to be displayed on screen.
     * @return what was typed by the player.
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next();
    }

    /**
     * Checks whether player entered Y or YES to a question.
     *
     * @param text player string from kb
     * @return true of Y or YES was entered, otherwise false
     */
    private boolean yesEntered(String text) {
        return stringIsAnyValue(text, "Y", "YES");
    }

    /**
     * Check whether a string equals one of a variable number of values
     * Useful to check for Y or YES for example
     * Comparison is case insensitive.
     *
     * @param text   source string
     * @param values a range of values to compare against the source string
     * @return true if a comparison was found in one of the variable number of strings passed
     */
    private boolean stringIsAnyValue(String text, String... values) {

        return Arrays.stream(values).anyMatch(str -> str.equalsIgnoreCase(text));
    }

    /**
     * Simulate the old basic tab(xx) command which indented text by xx spaces.
     *
     * @param spaces number of spaces required
     * @return String with number of spaces
     */
    private String simulateTabs(int spaces) {
        char[] spacesTemp = new char[spaces];
        Arrays.fill(spacesTemp, ' ');
        return new String(spacesTemp);
    }
}

```

# `33_Dice/java/src/DiceGame.java`

这段代码定义了一个名为 DiceGame 的类，其中包含一个名为 main 的方法。

在 main 方法中，使用 new 关键字创建了一个 Dice 对象，并调用其的 play 方法。Dice 对象在整个程序中会被用来操作骰子的。

Dice 对象可能具有轮转、点数等功能，而 play 方法则是让 Dice 对象投掷出骰子，并显示结果。具体实现可能会因程序修改而有所不同。


```
public class DiceGame {
    public static void main(String[] args) {
        Dice dice = new Dice();
        dice.play();
    }
}

```

# `33_Dice/javascript/dice.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。

`print` 函数的作用是在页面上打印一段字符串，将字符串作为参数传递给 `print` 函数，函数会在页面上创建一个包含该字符串的文本节点，并将其添加到页面的 `<textarea>` 元素中。

`input` 函数的作用是从用户那里获取输入，函数会创建一个包含输入字段的 `<input>` 元素，并设置其 `type` 属性为 `text`，设置其 `length` 属性为 `50`。函数会将 `input` 元素的 `focus` 事件处理程序设置为 `document.getElementById("output").appendChild(input_element)`，当用户点击页面上的 `INPUT` 元素时，函数会将 `input` 元素的值存储在 `input_str` 变量中，并将该值输出到页面上。函数还会监听 `keydown` 事件处理程序，当用户按下键盘上的 `SPACE` 键时，函数会将 `input_str` 的值打印到页面上，并将其中的 `\n` 换行。


```
// DICE
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

This code defines a JavaScript function called "tab" that takes one argument "space", which is an integer. The function builds a string of spaces based on the number of times the argument "space" is greater than zero, and returns the resulting string.

The main program prints out the result of calling the "tab" function with different arguments. For example, calling "tab(34)" will print out a string of spaces with 34 spaces in it.


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
    print(tab(34) + "DICE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    f = [];
    // Danny Freidus
    print("THIS PROGRAM SIMULATES THE ROLLING OF A\n");
    print("PAIR OF DICE.\n");
    print("YOU ENTER THE NUMBER OF TIMES YOU WANT THE COMPUTER TO\n");
    print("'ROLL' THE DICE.  WATCH OUT, VERY LARGE NUMBERS TAKE\n");
    print("A LONG TIME.  IN PARTICULAR, NUMBERS OVER 5000.\n");
    do {
        for (q = 1; q <= 12; q++)
            f[q] = 0;
        print("\n");
        print("HOW MANY ROLLS");
        x = parseInt(await input());
        for (s = 1; s <= x; s++) {
            a = Math.floor(Math.random() * 6 + 1);
            b = Math.floor(Math.random() * 6 + 1);
            r = a + b;
            f[r]++;
        }
        print("\n");
        print("TOTAL SPOTS\tNUMBER OF TIMES\n");
        for (v = 2; v <= 12; v++) {
            print("\t" + v + "\t" + f[v] + "\n");
        }
        print("\n");
        print("\n");
        print("TRY AGAIN");
        str = await input();
    } while (str.substr(0, 1) == "Y") ;
}

```

这是 C 语言中的一个标准函数，名为 `main()`，它是程序的入口点。当程序运行时，首先会进入 `main()` 函数，这个函数内部会首先输出 "Hello World!"，然后关闭程序并结束执行。

这里 `main()` 函数是一个左结合的函数，也就是说当函数内部有分支语句时，分支语句的左右部分都会被视为 `main()` 函数的一部分。


```
main();

```