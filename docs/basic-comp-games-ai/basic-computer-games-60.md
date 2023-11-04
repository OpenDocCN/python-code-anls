# BasicComputerGames源码解析 60

# `61_Math_Dice/python/mathdice.py`

This code is using the `random` module to generate two random integers, which will be used to create images of two dice. The program will print out a description of the game, including a dice with an equals sign followed by a question mark, and a prompt to the user to type their answer when they have finished playing. The program will then end when the user types 0.


```
from random import randint

print("Math Dice")
print("https://github.com/coding-horror/basic-computer-games")
print()
print(
    """This program generates images of two dice.
When two dice and an equals sign followed by a question
mark have been printed, type your answer, and hit the ENTER
key.
To conclude the program, type 0.
"""
)


```

这段代码使用了Python的函数传值特性，定义了两个函数print_0和print_2，分别打印一个“|”形状和一个星号形状的图案。接着定义了一个主函数print_dice，该函数包含两个条件判断，判断n是否为4、5或6，如果是，则执行print_2函数，否则执行print_0函数。print_0函数打印一个星号形状的图案，print_2函数打印一个星号形状的图案和一个“*”形状的图案，根据星号和数字的不同组合，会打印不同的图案。最终输出一个带有两个“|”形状图案的结果。


```
def print_dice(n: int) -> None:
    def print_0() -> None:
        print("|     |")

    def print_2() -> None:
        print("| * * |")

    print(" ----- ")

    if n in [4, 5, 6]:
        print_2()
    elif n in [2, 3]:
        print("| *   |")
    else:
        print_0()

    if n in [1, 3, 5]:
        print("|  *  |")
    elif n in [2, 4]:
        print_0()
    else:
        print_2()

    if n in [4, 5, 6]:
        print_2()
    elif n in [2, 3]:
        print("|   * |")
    else:
        print_0()

    print(" ----- ")


```

这段代码是一个Python程序，名为“main”。程序的主要目的是让用户猜两个随机整数的和，并在给出提示后让用户尝试计算答案。如果用户在两次尝试之间猜中了答案，程序会打印出“Correct!”的消息。如果用户在两次尝试之间未猜中答案，程序会根据提示告诉用户哪些地方出了问题，并让他们重新尝试。


```
def main() -> None:

    while True:
        d1 = randint(1, 6)
        d2 = randint(1, 6)
        guess = 13

        print_dice(d1)
        print("   +")
        print_dice(d2)
        print("   =")

        tries = 0
        while guess != (d1 + d2) and tries < 2:
            if tries == 1:
                print("No, count the spots and give another answer.")
            try:
                guess = int(input())
            except ValueError:
                print("That's not a number!")
            if guess == 0:
                exit()
            tries += 1

        if guess != (d1 + d2):
            print(f"No, the answer is {d1 + d2}!")
        else:
            print("Correct!")

        print("The dice roll again....")


```

这段代码是一个if语句，它的作用是判断当前脚本是否被作为主程序运行。如果当前脚本被作为主程序运行，那么程序会执行if语句中的代码块。

在这个例子中，if语句块中只有一行代码，即“main()”。这条代码会引发一个名为“__main__”的函数，因此if语句块中的代码实际上是在定义并调用一个名为“__main__”的函数。

简单来说，这段代码就是一个if语句，它判断当前脚本是否作为主程序，如果是，则执行if语句中的代码块，即调用一个名为“__main__”的函数。


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

If you wish to give the user more than 2 attempts to get the number, change value assigned to the num_tries variable at the start of the main function


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Mugwump

Your objective in this game is to find the four Mugwumps hiding on various squares of a 10 by 10 grid. Homebase (lower left) is position (0,0) and a guess is a pair of whole numbers (0 to 9), separated by commas. The first number is the number of units to the right of homebase and the second number is the distance above homebase.

You get ten guesses to locate the four Mugwumps; after each guess, the computer tells you how close you are to each Mugwump. Playing the game with the aid of graph paper and a compass should allow you to find all the Mugwumps in six or seven moves using triangulation similar to Loran radio navigation.

If you want to make the game somewhat more difficult, you can print the distance to each Mugwump either rounded or truncated to the nearest integer.

This program was modified slightly by Bob Albrecht of People’s Computer Company. It was originally written by students of Bud Valenti of Project SOLO in Pittsburg, Pennsylvania.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=114)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=129)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `62_Mugwump/csharp/Distance.cs`

这段代码定义了一个名为 "Distance" 的内部结构体 "Distance"，它包含一个私有成员变量 "value"，以及两个私有成员函数 "deltaX" 和 "deltaY"，这两个函数分别获取 "value" 的偏移量。

"Distance" 的构造函数接受两个浮点数参数 "deltaX" 和 "deltaY"，并使用它们计算出 "value" 的值，其中 "deltaX * deltaX" 和 "deltaY * deltaY" 分别代表 "value" 的平方项。

"Distance" 的 "ToString" 方法返回 "value" 的字符串表示形式，其中 "0.0" 是 "value" 的默认前缀，表示小数点后有两位数字。

这段代码的主要目的是定义一个计算两点之间距离的类，并提供了距离的表示方式和计算方法。这个类可以被用于计算图形、地图等数据中的距离，从而实现三维空间中的距离计算。


```
namespace Mugwump;

internal struct Distance
{
    private readonly float _value;

    public Distance(float deltaX, float deltaY)
    {
        _value = (float)Math.Sqrt(deltaX * deltaX + deltaY * deltaY);
    }

    public override string ToString() => _value.ToString("0.0");
}

```

# `62_Mugwump/csharp/Game.cs`



这段代码是一个游戏类，它使用反射和随机数生成器来实现游戏玩法。以下是代码的作用说明：

1. 游戏类包含两个私有变量：`_io` 和 `_random`，分别用于输出游戏界面信息和生成随机数。

2. 构造函数接受两个参数：`TextIO` 和 `IRandom`，用于获取游戏界面的 `TextIO` 实例和生成随机数。

3. `Play` 函数用于玩游戏，它包含以下步骤：

  a. 显示游戏介绍并等待玩家按任意键继续游戏。
  
  b. 循环玩一次游戏，每次游戏会生成一个包含四个随机位置的 `Grid` 对象，其中每个位置有一个包含 `_io` 和 `_random` 访问器的属性 `_position`。
  
  c. 在每个游戏循环中，使用 `_io.WriteLine` 输出游戏结果并暴露一些隐藏的位置，游戏也就结束啦。

4. `DisplayIntro` 函数用于在游戏开始时显示一段游戏介绍，它从游戏资源文件中读取一个 `Mugwump.Strings.Intro.txt` 文件的内容并将其写入 `TextIO`。

5. `Play` 函数用于生成随机游戏，它包含以下步骤：

   a. 循环 `4` 次，每次生成一个包含四个随机位置的 `Grid` 对象。

   b. 在每个游戏循环中，使用 `_io.ReadGuess` 获取玩家的猜测并检查它是否正确。

   c. 如果猜测正确，游戏就胜利啦，游戏也就结束了。

   d. 如果猜测不正确，游戏会提示玩家继续尝试，玩家就可以再试一次。


```
using System.Reflection;

namespace Mugwump;

internal class Game
{
    private readonly TextIO _io;
    private readonly IRandom _random;

    internal Game(TextIO io, IRandom random)
    {
        _io = io;
        _random = random;
    }

    internal void Play(Func<bool> playAgain = null)
    {
        DisplayIntro();

        while (playAgain?.Invoke() ?? true)
        {
            Play(new Grid(_io, _random));

            _io.WriteLine();
            _io.WriteLine("That was fun! Let's play again.......");
            _io.WriteLine("Four more mugwumps are now in hiding.");
        }
    }

    private void DisplayIntro()
    {
        using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream("Mugwump.Strings.Intro.txt");

        _io.Write(stream);
    }

    private void Play(Grid grid)
    {
        for (int turn = 1; turn <= 10; turn++)
        {
            var guess = _io.ReadGuess($"Turn no. {turn} -- what is your guess");

            if (grid.Check(guess))
            {
                _io.WriteLine();
                _io.WriteLine($"You got them all in {turn} turns!");
                return;
            }
        }

        _io.WriteLine();
        _io.WriteLine("Sorry, that's 10 tries.  Here is where they're hiding:");
        grid.Reveal();
    }
}

```

# `62_Mugwump/csharp/Grid.cs`



这段代码是一个Mugwump类，它包含一个Grid构造函数和一个Reveal方法。Grid构造函数接受两个参数，一个TextIO实例和一个随机数生成器。在构造函数中，使用Enumerable.Range函数生成一个包含1到4个随机整数的列表，然后将其中的每个整数转换为Mugwump类。Mugwump类包含一个查找类，用于查找给定的猜测位置的Mugwump。如果该位置在Mugwamp列表中找到了Mugwamp，该函数将在控制台输出到Mugwamp列表中。如果位置没有找到Mugwamp，该函数将输出到控制台并删除Mugwamp列表中的所有元素。

Reveal方法与Grid构造函数类似，只不过它返回Mugwamp类的Reveal方法，该方法将Mugwamp对象作为字符串输出到控制台。


```
using System.Collections.Generic;
using System.Linq;

namespace Mugwump;

internal class Grid
{
    private readonly TextIO _io;
    private readonly List<Mugwump> _mugwumps;

    public Grid(TextIO io, IRandom random)
    {
        _io = io;
        _mugwumps = Enumerable.Range(1, 4).Select(id => new Mugwump(id, random.NextPosition(10, 10))).ToList();
    }

    public bool Check(Position guess)
    {
        foreach (var mugwump in _mugwumps.ToList())
        {
            var (found, distance) = mugwump.FindFrom(guess);

            _io.WriteLine(found ? $"You have found {mugwump}" : $"You are {distance} units from {mugwump}");
            if (found)
            {
                _mugwumps.Remove(mugwump);
            }
        }

        return _mugwumps.Count == 0;
    }

    public void Reveal()
    {
        foreach (var mugwump in _mugwumps)
        {
            _io.WriteLine(mugwump.Reveal());
        }
    }
}

```

# `62_Mugwump/csharp/IRandomExtensions.cs`

这段代码是一个namespace级别的IRandomExtensions类，其中包含一个NextPosition方法。

作用：

该NextPosition方法接收一个IRandom实例和一个最大X和最大Y的整数，它返回一个新位置，其x坐标和y坐标分别基于当前随机数生成，并且不会超出最大X和最大Y的范围。

换句话说，该方法使用随机数生成器生成一个平滑的位置，用于在地图或其他游戏世界上生成位置。这个位置可以使用它来在地图上绘制或者在游戏世界上使用。


```
namespace Mugwump;

internal static class IRandomExtensions
{
    internal static Position NextPosition(this IRandom random, int maxX, int maxY) =>
        new(random.Next(maxX), random.Next(maxY));
}

```

# `62_Mugwump/csharp/Mugwump.cs`

这段代码定义了一个名为Mugwump的类，其中包含一个私有变量_id和一个私有变量_position，以及一些公共方法，如FindFrom和Reveal。

FindFrom方法接收一个参数guess，并返回两个布尔值，第一个表示guess是否等于Mugwump的position，第二个表示guess与Mugwump的position之间的距离。

Reveal方法返回Mugwump对象的字符串表示形式，其中包含其id和position信息。

Mugwump类还实现了toString方法，用于将其作为字符串返回。

这段代码的作用是定义了一个Mugwump类，用于表示一只被命名为“Mugwump”的生物，它可以被找到并描述其位置和状态。


```
namespace Mugwump;

internal class Mugwump
{
    private readonly int _id;
    private readonly Position _position;

    public Mugwump(int id, Position position)
    {
        _id = id;
        _position = position;
    }

    public (bool, Distance) FindFrom(Position guess) => (guess == _position, guess - _position);

    public string Reveal() => $"{this} is at {_position}";

    public override string ToString() => $"Mugwump {_id}";
}

```

# `62_Mugwump/csharp/Position.cs`

这段代码定义了一个名为 "Mugwump" 的命名空间，其中包含一个名为 "Position" 的内部结构体，该结构体具有两个浮点型成员变量 X 和 Y，以及一个重写自 "ToString()" 的 "public override" 的方法。

在 "public override" 的方法内部，使用 "namespace" 关键字定义了一个名为 "Distance" 的内部结构体，该结构体具有一个名为 "operator -" 的方法，其参数为两个 "Position" 类型的变量 p1 和 p2，返回值为一个新的 "Position" 类型的变量 result。

通过 "operator -" 方法，可以计算两个 "Position" 之间的距离，即使它们是相对于彼此的坐标。该方法实际上计算了两个 "Position" 之间的欧几里得距离，因此名称可能有点误导人。

此外，定义 "Position" 结构体时使用了 "public static" 的访问权限，这意味着该结构体可以直接从 "public" 命名空间访问，因此可以被定义为公共的、通用的。


```
namespace Mugwump;

internal record struct Position(float X, float Y)
{
    public override string ToString() => $"( {X} , {Y} )";

    public static Distance operator -(Position p1, Position p2) => new(p1.X - p2.X, p1.Y - p2.Y);
}

```

# `62_Mugwump/csharp/Program.cs`

这段代码是一个使用Mugwump库的游戏引擎，实现了创建一个游戏对象、设置游戏参数、生成随机数、输出游戏结果等功能。

具体来说，代码的作用如下：

1. 引入三个外部库：System、Games.Common.IO和Games.Common.Randomness，这些库提供了游戏开发中常用的功能和工具。

2. 创建一个RandomNumberGenerator类，用于生成随机数。

3. 创建一个ConsoleIO类，用于读写游戏日志到控制台。

4. 创建一个Game类，该类实现了IGame接口，用于管理游戏对象和运行游戏。

5. 调用Game类的Play()方法来运行游戏，其中参数是一个实现了IGame接口的对象，用于在游戏中接收和处理用户输入。

6. 在游戏运行结束时，输出游戏结果，以便用户了解游戏的得分和排名等信息。


```
﻿global using System;
global using Games.Common.IO;
global using Games.Common.Randomness;

using Mugwump;

var random = new RandomNumberGenerator();
var io = new ConsoleIO();

var game = new Game(io, random);

game.Play();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `62_Mugwump/csharp/TextIOExtensions.cs`

这段代码是一个C#类，名为Mugwump.TextIOExtensions。它是一个接口，名为TextIOExtensions，提供了输入方法，以模拟BASIC interpreter的键盘输入例行。

具体来说，这个接口包括一个名为ReadGuess的内部静态方法，它接受一个字符串参数prompt，并返回一个Position类型的位置数组。这个方法的作用是，当You need to read input from user，比如需要用户输入一个密码时，BASIC interpreter的键盘输入例行（也就是用户输入密码时，用户按下的键盘上的键）可以用这个方法来获取。

实际上，这个方法并没有对输入数据做任何处理，只是一个简单的读取输入数据并返回的接口。如果你需要对输入数据做处理，可以在这个方法上进行添加，比如获取用户输入的长度，或者检查输入是否符合某些规则。


```
namespace Mugwump;

// Provides input methods which emulate the BASIC interpreter's keyboard input routines
internal static class TextIOExtensions
{
    internal static Position ReadGuess(this TextIO io, string prompt)
    {
        io.WriteLine();
        io.WriteLine();
        var (x, y) = io.Read2Numbers(prompt);
        return new Position(x, y);
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `62_Mugwump/java/src/Mugwump.java`

This is a Java program that simulates a game of Connect-the-Dot where the player tries to match each dot to its corresponding number.

The program has several key features:

1. It uses a main method that initializes the game.
2. It has a distanceOfHomebase method that takes a string and a number and returns the number of spaces to the right of the homebase and above it.
3. It has a getDelimitedValue method that takes a string and a position and returns the integer representation of the value.
4. It has a displayTextAndGetInput method that displays a message on the screen and then accepts input from the keyboard.
5. It has a addSpaces method that adds spaces of a given number to a string.

The program uses a Mugwump class that implements the Icon class and has a method to play the game.

The program starts by instantiating the Mugwump class and then calls its play method, which simulates the game.


```
import java.util.Arrays;
import java.util.Scanner;

/**
 * Game of Mugwump
 * <p>
 * Based on the Basic game of Mugwump here
 * https://github.com/coding-horror/basic-computer-games/blob/main/62%20Mugwump/mugwump.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */

public class Mugwump {

    public static final int NUMBER_OF_MUGWUMPS = 4;

    public static final int MAX_TURNS = 10;

    public static final int FOUND = -1;

    // Used for keyboard input
    private final Scanner kbScanner;

    private enum GAME_STATE {
        INIT,
        GAME_START,
        PLAY_TURN
    }

    // Current game state
    private GAME_STATE gameState;

    int[][] mugwumpLocations;

    int turn;

    public Mugwump() {
        kbScanner = new Scanner(System.in);
        gameState = GAME_STATE.INIT;
    }

    /**
     * Main game loop
     */
    public void play() {

        do {
            switch (gameState) {

                case INIT:
                    intro();
                    gameState = GAME_STATE.GAME_START;

                    break;

                case GAME_START:

                    turn = 0;

                    // initialise all array elements with 0
                    mugwumpLocations = new int[NUMBER_OF_MUGWUMPS][2];

                    // Place 4 mugwumps
                    for (int i = 0; i < NUMBER_OF_MUGWUMPS; i++) {
                        for (int j = 0; j < 2; j++) {
                            mugwumpLocations[i][j] = (int) (Math.random() * 10);
                        }
                    }
                    gameState = GAME_STATE.PLAY_TURN;
                    break;

                case PLAY_TURN:
                    turn++;
                    String locations = displayTextAndGetInput("TURN NO." + turn + " -- WHAT IS YOUR GUESS? ");
                    int distanceRightGuess = getDelimitedValue(locations, 0);
                    int distanceUpGuess = getDelimitedValue(locations, 1);

                    int numberFound = 0;
                    for (int i = 0; i < NUMBER_OF_MUGWUMPS; i++) {

                        if (mugwumpLocations[i][0] == FOUND) {
                            numberFound++;
                        }

                        int right = mugwumpLocations[i][0];
                        int up = mugwumpLocations[i][1];

                        if (right == distanceRightGuess && up == distanceUpGuess) {
                            if (right != FOUND) {
                                System.out.println("YOU HAVE FOUND MUGWUMP " + (i + 1));
                                mugwumpLocations[i][0] = FOUND;
                            }
                            numberFound++;
                        } else {
                            // Not found so show distance
                            if (mugwumpLocations[i][0] != FOUND) {
                                double distance = Math.sqrt((Math.pow(right - distanceRightGuess, 2.0d))
                                        + (Math.pow(up - distanceUpGuess, 2.0d)));

                                System.out.println("YOU ARE " + (int) ((distance * 10) / 10) + " UNITS FROM MUGWUMP");
                            }
                        }
                    }

                    if (numberFound == NUMBER_OF_MUGWUMPS) {
                        System.out.println("YOU GOT THEM ALL IN " + turn + " TURNS!");
                        gameState = GAME_STATE.GAME_START;
                    } else if (turn >= MAX_TURNS) {
                        System.out.println("SORRY, THAT'S " + MAX_TURNS + " TRIES.  HERE IS WHERE THEY'RE HIDING");
                        for (int i = 0; i < NUMBER_OF_MUGWUMPS; i++) {
                            if (mugwumpLocations[i][0] != FOUND) {
                                System.out.println("MUGWUMP " + (i + 1) + " IS AT ("
                                        + mugwumpLocations[i][0] + "," + mugwumpLocations[i][1] + ")");
                            }
                        }
                        gameState = GAME_STATE.GAME_START;
                    }

                    // Game ended?
                    if (gameState != GAME_STATE.PLAY_TURN) {
                        System.out.println("THAT WAS FUN! LET'S PLAY AGAIN.......");
                        System.out.println("FOUR MORE MUGWUMPS ARE NOW IN HIDING.");
                    }
            }
            // Infinite loop - based on original basic version
        } while (true);
    }

    private void intro() {
        System.out.println(addSpaces(33) + "MUGWUMP");
        System.out.println(addSpaces(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("THE OBJECT OF THIS GAME IS TO FIND FOUR MUGWUMPS");
        System.out.println("HIDDEN ON A 10 BY 10 GRID.  HOMEBASE IS POSITION 0,0.");
        System.out.println("ANY GUESS YOU MAKE MUST BE TWO NUMBERS WITH EACH");
        System.out.println("NUMBER BETWEEN 0 AND 9, INCLUSIVE.  FIRST NUMBER");
        System.out.println("IS DISTANCE TO RIGHT OF HOMEBASE AND SECOND NUMBER");
        System.out.println("IS DISTANCE ABOVE HOMEBASE.");
        System.out.println();
        System.out.println("YOU GET 10 TRIES.  AFTER EACH TRY, I WILL TELL");
        System.out.println("YOU HOW FAR YOU ARE FROM EACH MUGWUMP.");
    }

    /**
     * Accepts a string delimited by comma's and returns the pos'th delimited
     * value (starting at count 0).
     *
     * @param text - text with values separated by comma's
     * @param pos  - which position to return a value for
     * @return the int representation of the value
     */
    private int getDelimitedValue(String text, int pos) {
        String[] tokens = text.split(",");
        return Integer.parseInt(tokens[pos]);
    }

    /*
     * Print a message on the screen, then accept input from Keyboard.
     *
     * @param text message to be displayed on screen.
     * @return what was typed by the player.
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.nextLine();
    }

    /**
     * Return a string of x spaces
     *
     * @param spaces number of spaces required
     * @return String with number of spaces
     */
    private String addSpaces(int spaces) {
        char[] spacesTemp = new char[spaces];
        Arrays.fill(spacesTemp, ' ');
        return new String(spacesTemp);
    }

    public static void main(String[] args) {

        Mugwump mugwump = new Mugwump();
        mugwump.play();
    }
}

```

# `62_Mugwump/javascript/mugwump.js`

这段代码定义了两个函数，分别是`print()`和`input()`。

`print()`函数的作用是打印一段字符串到网页的输出区域（通常是DOM元素）。该函数调用了`document.getElementById()`获取输出元素，并创建了一个包含该元素的文字节点，然后将其添加到输出区域中。

`input()`函数的作用是接收用户输入的字符串。它通过创建一个`<INPUT>`元素，设置了其`type`属性的值为`text`，并设置了其`length`属性的值为`50`（表示最大输入字符数为50个）。然后将该元素添加到输出区域中，并注册了一个`keydown`事件监听器，以便在用户按下键盘上的键时接收输入的字符串。当用户按下了回车键时，函数会将输入的字符串打印出来，并将其添加到输出区域中。


```
// MUGWUMP
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

This is a program that uses the Google Sheets API to generate a solution to the popular game, Mugwump. The game is played with four players, each of whom is given a list of the的位置 of the other three players. The first player is given a unique letter (represented by a 2D array) that is used to represent their position. The position of each player is represented by a 3D array, where each element of the array corresponds to one of the three players.

The program uses a while loop to keep playing the game until the user tells it to stop. The loop is breaked when the user enters a number greater than or equal to 10. When the user enters a number greater than 10, the program displays the number of tries it has tried, and then displays the positions of the four players in the game.

The program also uses a for loop to calculate the distance between each player's position and the center of the circle. The distance is represented by a variable of type Math.sqrt, which issqrt(2) \* Math.sqrt(3). The program then uses this distance to display the distance to the nearest player, using the Math.floor function to round the distance down to the nearest whole number.

The program also uses a variable of type Print to display information about the game, such as the names of the players, their positions, and the number of tries it has taken.

It is important to note that the program is using the Google Sheets API to access the data, so it is important to have the permission to access the sheets.



```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var p = [];

// Main program
async function main()
{
    print(tab(33) + "MUGWUMP\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // Courtesy People's Computer Company
    print("THE OBJECT OF THIS GAME IS TO FIND FOUR MUGWUMPS\n");
    print("HIDDEN ON A 10 BY 10 GRID.  HOMEBASE IS POSITION 0,0.\n");
    print("ANY GUESS YOU MAKE MUST BE TWO NUMBERS WITH EACH\n");
    print("NUMBER BETWEEN 0 AND 9, INCLUSIVE.  FIRST NUMBER\n");
    print("IS DISTANCE TO RIGHT OF HOMEBASE AND SECOND NUMBER\n");
    print("IS DISTANCE ABOVE HOMEBASE.\n");
    print("\n");
    print("YOU GET 10 TRIES.  AFTER EACH TRY, I WILL TELL\n");
    print("YOU HOW FAR YOU ARE FROM EACH MUGWUMP.\n");
    print("\n");
    while (1) {
        for (i = 1; i <= 4; i++) {
            p[i] = [];
            for (j = 1; j <= 2; j++) {
                p[i][j] = Math.floor(10 * Math.random());
            }
        }
        t = 0;
        do {
            t++;
            print("\n");
            print("\n");
            print("TURN NO. " + t + " -- WHAT IS YOUR GUESS");
            str = await input();
            m = parseInt(str);
            n = parseInt(str.substr(str.indexOf(",") + 1));
            for (i = 1; i <= 4; i++) {
                if (p[i][1] == -1)
                    continue;
                if (p[i][1] == m && p[i][2] == n) {
                    p[i][1] = -1;
                    print("YOU HAVE FOUND MUGWUMP " + i + "\n");
                } else {
                    d = Math.sqrt(Math.pow(p[i][1] - m, 2) + Math.pow(p[i][2] - n, 2));
                    print("YOU ARE " + Math.floor(d * 10) / 10 + " UNITS FROM MUGWUMP " + i + "\n");
                }
            }
            for (j = 1; j <= 4; j++) {
                if (p[j][1] != -1)
                    break;
            }
            if (j > 4) {
                print("\n");
                print("YOU GOT THEM ALL IN " + t + " TURNS!\n");
                break;
            }
        } while (t < 10) ;
        if (t == 10) {
            print("\n");
            print("SORRY, THAT'S 10 TRIES.  HERE IS WHERE THEY'RE HIDING:\n");
            for (i = 1; i <= 4; i++) {
                if (p[i][1] != -1)
                    print("MUGWUMP " + i + " IS AT (" + p[i][1] + "," + p[i][2] + ")\n");
            }
        }
        print("\n");
        print("THAT WAS FUN! LET'S PLAY AGAIN.......\n");
        print("FOUR MORE MUGWUMPS ARE NOW IN HIDING.\n");
    }
}

```

这道题是一个简单的编程题目，主要目的是让我们了解 main() 函数在程序中的作用。main() 函数是 Python 应用程序的入口点，程序从此处开始执行。当程序运行时，首先会执行 main() 函数，然后是程序中的代码。main() 函数可以确保程序按照预期运行，即使赋予不同的输入参数，程序也会给出相应的结果。


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


# `62_Mugwump/python/mugwump.py`

Here's a breakdown of the code:

1. The first line imports the `math` module and the `random` module.
2. The third line imports the `typing` module.
3. The `introduction()` function is the main entry point of


```
from math import sqrt
from random import randint
from typing import List, Tuple


def introduction() -> None:
    print(
        """The object of this game is to find 4 mugwumps
hidden on a 10*10 grid.  Homebase is position 0,0.
Any guess you make must be two numbers with each
number between 0 and 9 inclusive.  First number
is distance to right of homebase, and second number
is the distance above homebase."""
    )

    print(
        """You get 10 tries.  After each try, I will tell
```

这段代码是一个 Python 程序，它的主要目的是生成一个包含 n 个随机整数的列表，并输出其中一些列表的 "隐藏" 位置。它包含两个函数：generate_mugwumps 和 reveal_mugwumps。下面分别对这两个函数进行解释：

1. generate_mugwumps：

这个函数接收一个整数参数 n，然后生成一个包含 n 个随机整数的列表。函数内部使用 for 循环来遍历生成过程，每次生成一个包含两个整数的列表，然后将这个列表添加到 mugwumps 列表中。

2. reveal_mugwumps：

这个函数接收一个包含随机整数的列表 mugwumps，然后输出这些列表中的所有位置。函数内部使用 for 循环来遍历 mugwumps 列表，然后对于每个位置，程序输出该位置的 "隐藏" 位置。


```
you how far you are from each mugwump."""
    )


def generate_mugwumps(n: int = 4) -> List[List[int]]:
    mugwumps = []
    for _ in range(n):
        current = [randint(0, 9), randint(0, 9)]
        mugwumps.append(current)
    return mugwumps


def reveal_mugwumps(mugwumps: List[List[int]]) -> None:
    print("Sorry, that's 10 tries.  Here's where they're hiding.")
    for idx, mugwump in enumerate(mugwumps, 1):
        if mugwump[0] != -1:
            print(f"Mugwump {idx} is at {mugwump[0]},{mugwump[1]}")


```

这段代码定义了两个函数，分别是 `calculate_distance` 和 `play_round`。它们的作用分别是计算玩家猜测的距离和玩家在游戏中寻找隐藏的 mugwumps。

具体来说，`calculate_distance` 函数接受一个 tuple 类型的参数 `guess` 和一个列表类型的参数 `mugwumps`。它使用 Pythonic 的 `**` 运算符展开这两个参数，然后计算它们之间的差的平方根，并将结果返回。这个函数的作用是帮助玩家计算他们距离目标位置有多远。

`play_round` 函数则接受两个参数：一个是用于存储已经找到的 mugwumps 的列表，另一个是用于存储每个 mugwump 距离的列表。它使用一个 while 循环来不断询问玩家猜测下一个隐藏的 mugwump，并在找到后更新 score。当玩家猜测错误的次数达到 4 时，它将揭示所有的隐藏 mugwumps。在每一次玩家猜测后，`turns` 变量会增加 1，以便统计玩家的猜测次数。

这两个函数都在一个名为 `play_again` 的函数中。这个函数首先让玩家再次猜测，如果猜测正确的，就退出游戏；否则，继续让玩家猜测，直到他们猜对为止。


```
def calculate_distance(guess: Tuple[int, int], mugwump: List[int]) -> float:
    d = sqrt(((mugwump[0] - guess[0]) ** 2) + ((mugwump[1] - guess[1]) ** 2))
    return d


def play_again() -> None:
    print("THAT WAS FUN! LET'S PLAY AGAIN.......")
    choice = input("Press Enter to play again, any other key then Enter to quit.")
    if choice == "":
        print("Four more mugwumps are now in hiding.")
    else:
        exit()


def play_round() -> None:
    mugwumps = generate_mugwumps()
    turns = 1
    score = 0
    while turns <= 10 and score != 4:
        m = -1
        while m == -1:
            try:
                m, n = map(int, input(f"Turn {turns} - what is your guess? ").split())
            except ValueError:
                m = -1
        for idx, mugwump in enumerate(mugwumps):
            if m == mugwump[0] and n == mugwump[1]:
                print(f"You found mugwump {idx + 1}")
                mugwumps[idx][0] = -1
                score += 1
            if mugwump[0] == -1:
                continue
            print(
                f"You are {calculate_distance((m, n), mugwump):.1f} units from mugwump {idx + 1}"
            )
        turns += 1
    if score == 4:
        print(f"Well done! You got them all in {turns} turns.")
    else:
        reveal_mugwumps(mugwumps)


```

这段代码是一个Python程序，主要作用是一个交互式的游戏。这里有一个while循环，只要程序还在运行，就会一直循环执行下面的introduction()函数和play_round()函数。

introduction()函数的作用是输出一个介绍，告诉用户这个程序是一个什么东西，以及如何使用。这里没有实现具体的输出，只是简单地输出了一行欢迎消息。

play_round()函数的作用是让用户进行一次游戏。它会随机生成两个0到9的整数，然后让用户判断自己的数字是否比对手的数字大。如果用户胜了，play_round()函数会再次生成随机数，并且继续让用户判断，直到有一方失败为止。

整个程序的逻辑就是通过while循环一直调用introduction()函数和play_round()函数，让用户进行多次游戏，直到程序结束。


```
if __name__ == "__main__":
    introduction()
    while True:
        play_round()
        play_again()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Name

NAME is a silly little ice-breaker to get a relationship going between a computer and a shy human. The sorting algorithm used is highly inefficient — as any reader of _Creative Computing_ will recognize, this is the worst possible sort for speed. But the program is good fun and that’s what counts here.

NAME was originally written by Geoffrey Chase of the Abbey, Portsmouth, Rhode Island.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=116)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=131)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `63_Name/csharp/Program.cs`

This appears to be a simple program that outputs a series of text elements centered at the top of the screen, in a Hello, followed by its name, a space, and then a request to input its name. The name is then reversed and then compared to the input name, and finally the program outputs whether the input name is liked or not. The program uses the `Console.WriteLine` method to output the text elements to the console.


```
﻿using System;

namespace Name
{
    public class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("NAME".CentreAlign());
            Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY".CentreAlign());
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine("HELLO.");
            Console.WriteLine("MY NAME IS CREATIVE COMPUTER.");
            Console.Write("WHAT'S YOUR NAME (FIRST AND LAST? ");
            var name = Console.ReadLine();
            Console.WriteLine();
            Console.WriteLine($"THANK YOU, {name.Reverse()}.");
            Console.WriteLine("OOPS!  I GUESS I GOT IT BACKWARDS.  A SMART");
            Console.WriteLine("COMPUTER LIKE ME SHOULDN'T MAKE A MISTAKE LIKE THAT!");
            Console.WriteLine();
            Console.WriteLine("BUT I JUST NOTICED YOUR LETTERS ARE OUT OF ORDER.");
            Console.WriteLine($"LET'S PUT THEM IN ORDER LIKE THIS: {name.Sort()}");
            Console.WriteLine();
            Console.Write("DON'T YOU LIKE THAT BETTER? ");
            var like = Console.ReadLine();
            Console.WriteLine();

            if (like.ToUpperInvariant() == "YES")
            {
                Console.WriteLine("I KNEW YOU'D AGREE!!");
            }
            else
            {
                Console.WriteLine("I'M SORRY YOU DON'T LIKE IT THAT WAY.");
            }

            Console.WriteLine();
            Console.WriteLine($"I REALLY ENJOYED MEETING YOU {name}.");
            Console.WriteLine("HAVE A NICE DAY!");
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `63_Name/csharp/StringExtensions.cs`

以上代码是一个名为 "Name.StringExtensions" 的命名类，其中包含以下三个方法：

1. CentreAlign: 该方法接收一个字符串参数，并返回一个左对齐、右对齐、宽度为 consoleWidth 字符串。

2. Reverse: 该方法接收一个字符串参数，并返回一个逆序的字符串。

3. Sort: 该方法接收一个字符串参数，并返回一个按升序或降序排列的字符串。

此代码使用了 C# 的命名约定，包括驼峰命名法、单引号括起类名、命名参数等。


```
﻿using System;

namespace Name
{
    public static class StringExtensions
    {
        private const int ConsoleWidth = 120; // default console width

        public static string CentreAlign(this string value)
        {
            int spaces = ConsoleWidth - value.Length;
            int leftPadding = spaces / 2 + value.Length;

            return value.PadLeft(leftPadding).PadRight(ConsoleWidth);
        }

        public static string Reverse(this string value)
        {
            if (value is null)
            {
                return null;
            }

            char[] characterArray = value.ToCharArray();
            Array.Reverse(characterArray);
            return new String(characterArray);
        }

        public static string Sort(this string value)
        {
            if (value is null)
            {
                return null;
            }

            char[] characters = value.ToCharArray();
            Array.Sort(characters);
            return new string(characters);
        }
    }
}

```

# `63_Name/java/Name.java`



This is a Java program that prints out a series of names and a prompt asking for the name of a person to sort. Here's a step-by-step explanation of the program:

1. Define a print() method that takes a String parameter representing the name to print.
2. Define a print() method that takes a String parameter representing the name to print in reverse order.
3. Define a print() method that takes a String parameter representing the name to print.
4. Define a main() method that is the entry point of the program.
5. Call the print() methods with different arguments, such as a name and a reversed name.
6. Call the print() method with a prompt asking for a name to sort.
7. Print a newline character (System.out.println(" ")) at the end of each printed name.
8. Print a newline character (System.out.println()) at the end of each printed name.
9. Print the name of the person whose name is being requested to sort.
10. Print "I guessed I got it backwards."
11. Print "OOPS! I guess I got it backwards."
12. Print "But I just notic


```
import java.util.Arrays;
import java.util.Scanner;

public class Name {

    public static void printempty() { System.out.println(" "); }

    public static void print(String toprint) { System.out.println(toprint); }

    public static void main(String[] args) {
        print("                                          NAME");
        print("                         CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        printempty();
        printempty();
        print("HELLO.");
        print("MY NAME iS CREATIVE COMPUTER.");
        print("WHATS YOUR NAME? (FIRST AND LAST)");

        Scanner namesc = new Scanner(System.in);
        String name = namesc.nextLine();

        String namereversed = new StringBuilder(name).reverse().toString();

        char namesorted[] = name.toCharArray();
        Arrays.sort(namesorted);

        printempty();
        print("THANK YOU, " + namereversed);
        printempty();
        print("OOPS!  I GUESS I GOT IT BACKWARDS.  A SMART");
        print("COMPUTER LIKE ME SHOULDN'T MAKE A MISTAKE LIKE THAT!");
        printempty();
        printempty();
        print("BUT I JUST NOTICED YOUR LETTERS ARE OUT OF ORDER.");

        print("LET'S PUT THEM IN ORDER LIKE THIS: " + new String(namesorted));
        printempty();
        printempty();

        print("DON'T YOU LIKE THAT BETTER?");
        printempty();

        Scanner agreementsc = new Scanner(System.in);
        String agreement = agreementsc.nextLine();

        if (agreement.equalsIgnoreCase("yes")) {
            print("I KNEW YOU'D AGREE!!");
        } else {
            print("I'M SORRY YOU DON'T LIKE IT THAT WAY.");
            printempty();
            print("I REALLY ENJOYED MEETING YOU, " + name);
            print("HAVE A NICE DAY!");
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `63_Name/javascript/name.js`

这两段代码是在 JavaScript 中实现的。它们的作用是解释如何将文本输入框中的用户输入转换为字符串并将其显示在屏幕上，以及对输入的文本进行输入。

具体来说，第一段代码 `print()` 是一个函数，用于将给定的字符串输出到网页上的一个元素中。该函数通过调用 `document.getElementById()` 获取到网页上的元素（在这里是 `output` 元素），然后创建一个 `createTextNode()` 方法的字符对象，将字符串转换为 `document.createTextNode()` 方法返回的节点。最后，将这个节点添加到指定的元素上并设置其样式。

第二段代码 `input()` 是一个 Promise 函数，用于获取用户输入的文本。该函数创建了一个新的输入元素（在这里是一个 `<INPUT>` 元素），设置了其样式，并添加到页面上。然后，函数开始监听输入元素的事件，当用户按下键盘上的 13 号键（通常是回车键）时，它将获取输入元素中的文本并将其存储在 `input_str` 变量中。最后，函数将 `input_str` 字符串中的换行符 `\n` 添加到字符串中，并将其添加到指定的元素上。


```
// NAME
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

的创作计算机模拟程序。这个程序会读取用户的输入，并输出一个类似于下面这样的文本：

```
Hello,
MY NAME IS CREATIVE COMPUTER.
WHAT'S YOUR NAME (FIRST AND LAST)?
```

然后，程序会读取用户的输入，并尝试将用户的输入存储在一个数组中。如果用户输入的格式正确，程序会尝试输出用户的输入。如果格式错误，程序会输出一条错误消息。

例如，如果用户输入下面这个格式：

```
JOAN
```

程序会尝试输出下面这个文本：

```
OH No!  That's not a valid input.
```

如果用户输入下面这个格式：

```
JOHN
JOAN
```

程序会尝试输出下面这个文本：

```
OH No!  That's not a valid input.
JOHN
```

除了输出正确的输入外，程序还会输出一些常见文本，如：

```
HELLO.
MY NAME IS CREATIVE COMPUTER.
```

```
WHAT'S YOUR NAME (FIRST AND LAST)?
```

```
HELLO,
MY NAME IS CREATIVE COMPUTER.
WHAT'S YOUR NAME (FIRST AND LAST)?
```

```
OOPS!  I GUESS I GOT IT BACKWARDS.  A SMART
COMPUTER LIKE ME SHOULDN'T MAKE A MISTAKE LIKE THAT!
```

```
I'M SORRY YOUdon't like it that way.
```

```
I REALLY ENJOYED MEETING YOU today.
HAVE A NICE DAY!
```

这些文本会在程序中一直循环，直到用户关闭程序并退出。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var str;
var b;

// Main program
async function main()
{
    print(tab(34) + "NAME\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("HELLO.\n");
    print("MY NAME IS CREATIVE COMPUTER.\n");
    print("WHAT'S YOUR NAME (FIRST AND LAST)");
    str = await input();
    l = str.length;
    print("\n");
    print("THANK YOU, ");
    for (i = l; i >= 1; i--)
        print(str[i - 1]);
    print(".\n");
    print("OOPS!  I GUESS I GOT IT BACKWARDS.  A SMART\n");
    print("COMPUTER LIKE ME SHOULDN'T MAKE A MISTAKE LIKE THAT!\n");
    print("\n");
    print("BUT I JUST NOTICED YOUR LETTERS ARE OUT OF ORDER.\n");
    print("LET'S PUT THEM IN ORDER LIKE THIS: ");
    b = [];
    for (i = 1; i <= l; i++)
        b[i - 1] = str.charCodeAt(i - 1);
    b.sort();
    for (i = 1; i <= l; i++)
        print(String.fromCharCode(b[i - 1]));
    print("\n");
    print("\n");
    print("DON'T YOU LIKE THAT BETTER");
    ds = await input();
    if (ds == "YES") {
        print("\n");
        print("I KNEW YOU'D AGREE!!\n");
    } else {
        print("\n");
        print("I'M SORRY YOU DON'T LIKE IT THAT WAY.\n");
    }
    print("\n");
    print("I REALLY ENJOYED MEETING YOU " + str + ".\n");
    print("HAVE A NICE DAY!\n");
}

```

这道题目没有给出代码，只有一行 main();。根据常规程序结构，这一行标志着程序的主函数，也就是程序的入口。

main() 是许多编程语言中的一个函数，用于将程序执行的重心从代码的其他部分转向程序的输入和输出。程序的输入通常来自标准输入（通常是键盘），而程序的输出通常显示在屏幕或输出设备上。

在 main() 函数中，程序可以读取用户的输入并对其进行处理，然后根据用户的需求执行相应的操作。不同的编程语言可能对 main() 函数有不同的含义，但它们共同的作用是引导程序的执行过程。


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


# `63_Name/python/name.py`

这段代码定义了一个名为 `is_yes_ish` 的函数，该函数接收一个字符串参数 `answer`，并返回一个布尔值。函数的作用是判断给定的字符串是否为 "是" 或 "yes"，因为函数的描述是简单地对这个字符串进行处理。

具体来说，这段代码实现了一个字符串 manipulation，即对字符串进行清洗（去除空格、strip）和转换为大写，然后判断得到的结果是否为 "Y" 或 "yes"。如果结果在这两个字符串中，函数返回 `True`，否则返回 `False`。


```
"""
NAME

simple string manipulations on the user's name

Ported by Dave LeCompte
"""


def is_yes_ish(answer: str) -> bool:
    cleaned = answer.strip().upper()
    if cleaned in ["Y", "YES"]:
        return True
    return False


```

这段代码是一个Python程序，主要目的是获取用户输入的名字，然后将其存储在一个列表中，并使用sorted函数对列表进行排序。排序后的名字存储在一个新的列表中，然后打印出来，最后向用户提供一个评价，告诉他们是否喜欢这个结果。


```
def main() -> None:
    print(" " * 34 + "NAME")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")
    print("HELLO.")
    print("MY NAME iS CREATIVE COMPUTER.")
    name = input("WHAT'S YOUR NAME (FIRST AND LAST)?")
    print()
    name_as_list = list(name)
    reversed_name = "".join(name_as_list[::-1])
    print(f"THANK YOU, {reversed_name}.\n")
    print("OOPS!  I GUESS I GOT IT BACKWARDS.  A SMART")
    print("COMPUTER LIKE ME SHOULDN'T MAKE A MISTAKE LIKE THAT!\n\n")
    print("BUT I JUST NOTICED YOUR LETTERS ARE OUT OF ORDER.")

    sorted_name = "".join(sorted(name_as_list))
    print(f"LET'S PUT THEM IN ORDER LIKE THIS: {sorted_name}\n\n")

    print("DON'T YOU LIKE THAT BETTER?")
    like_answer = input()
    print()
    if is_yes_ish(like_answer):
        print("I KNEW YOU'D AGREE!!")
    else:
        print("I'M SORRY YOU DON'T LIKE IT THAT WAY.")
    print()
    print(f"I REALLY ENJOYED MEETING YOU, {name}.")
    print("HAVE A NICE DAY!")


```

这段代码是一个if语句，它会判断当前脚本是否作为主程序运行。如果当前脚本作为主程序运行，那么if语句块内的代码将会被执行。

在这个if语句块内，定义了一个名为“__main__”的常量，它的值为“__main__”。这个常量在Python中非常重要，表示当前脚本是否作为主程序运行。如果当前脚本作为主程序运行，那么if语句块内的代码将会被执行。否则，if语句块内的代码将不会被执行。

在这个if语句块内，使用了一个名为“main”的函数作为if语句的判断条件。这个函数可能是定义在当前脚本中，也可能是定义在其他文件中，它会在当前脚本中执行if语句块内的代码。如果当前脚本作为主程序运行，那么函数中的代码将会被执行。否则，函数中的代码将不会被执行。


```
if __name__ == "__main__":
    main()

```