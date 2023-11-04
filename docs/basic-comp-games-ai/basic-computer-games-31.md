# BasicComputerGames源码解析 31

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by Anthony Rubick [AnthonyMichaelTDM](https://github.com/AnthonyMichaelTDM)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Chief

In the words of the program author, John Graham, “CHIEF is designed to give people (mostly kids) practice in the four operations (addition, multiplication, subtraction, and division).

It does this while giving people some fun. And then, if the people are wrong, it shows them how they should have done it.

CHIEF was written by John Graham of Upper Brookville, New York.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=43)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=58)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `25_Chief/csharp/Game.cs`



这段代码是一个Chief框架的游戏，通过控制台或游戏画面进行交互。

具体来说，代码中定义了一个Game类，其中的Play方法进行游戏的逻辑。在该方法中，首先通过一个IReadWrite的实例来读取玩家的输入。然后，通过调用一些方法来获取游戏中的提示信息，并根据玩家的输入来决定游戏进程。

具体来说，代码中定义了以下方法：

- DoIntroduction：用于初始化游戏，包括显示游戏标题、询问玩家是否准备好了游戏等操作。
- Play：用于实际游戏的核心逻辑，包括读取玩家的输入，并根据玩家的不同选择来执行不同的操作。
- Answer：用于获取玩家输入的问题，以便进行输入验证。
- Original：用于获取玩家输入的原话，以便在游戏结束时显示。
- Bye：用于表示游戏结束，并输出一个向下的箭头。
-迷信：用于显示一个神秘的象征，以表明游戏已经结束。
- Lightening：用于在游戏结束时模拟雷声，以增加游戏的神秘感。

最后，在Play方法的最后，使用_io.ReadNumber读取玩家的输入，并使用Prompts.Answer获取玩家的答案。然后，使用Math.CalculateOriginal方法计算原始输入，并使用Prompts.Bet获取另一个输入。最后，根据玩家的输入，执行相应的操作，并输出一些信息来表明游戏已经结束。


```
using static Chief.Resources.Resource;

namespace Chief;

internal class Game
{
    private readonly IReadWrite _io;

    public Game(IReadWrite io)
    {
        _io = io;
    }

    internal void Play()
    {
        DoIntroduction();

        var result = _io.ReadNumber(Prompts.Answer);

        if (_io.ReadYes(Formats.Bet, Math.CalculateOriginal(result)))
        {
            _io.Write(Streams.Bye);
            return;
        }

        var original = _io.ReadNumber(Prompts.Original);

        _io.WriteLine(Math.ShowWorking(original));

        if (_io.ReadYes(Prompts.Believe))
        {
            _io.Write(Streams.Bye);
            return;
        }

        _io.Write(Streams.Lightning);
    }

    private void DoIntroduction()
    {
        _io.Write(Streams.Title);
        if (!_io.ReadYes(Prompts.Ready))
        {
            _io.Write(Streams.ShutUp);
        }

        _io.Write(Streams.Instructions);
    }
}

```

# `25_Chief/csharp/IReadWriteExtensions.cs`

这段代码是一个名为“Chief. Internal Static Class IReadWriteExtensions”的命名空间。内部静态类“IReadWriteExtensions”提供了一些与文件读写相关的扩展方法。

具体来说，这段代码定义了一个名为“ReadYes”的静态方法，该方法接受一个实际的“IReadWrite”接口类型的参数“io”，一个格式字符串“format”，和一个表示要读取的数据“value”。方法返回一个布尔值，表示“io”是否成功读取到格式化后的数据“value”。

另外，还定义了一个名为“ReadYes”的静态方法，该方法接受一个格式字符串“prompt”，表示一个用户输入的问题。该方法返回一个布尔值，表示用户是否输入了“是”（用ASCII编码为“Yes”）。

这两个静态方法可以直接在需要时通过调用接口实体的形式来使用。


```
namespace Chief;

internal static class IReadWriteExtensions
{
    internal static bool ReadYes(this IReadWrite io, string format, Number value) =>
        io.ReadYes(string.Format(format, value));
    internal static bool ReadYes(this IReadWrite io, string prompt) =>
        io.ReadString(prompt).Equals("Yes", StringComparison.InvariantCultureIgnoreCase);
}
```

# `25_Chief/csharp/Math.cs`

这段代码是一个C#类中定义了一个名为Math的公共类，其中包含两个方法，一个是CalculateOriginal，另一个是ShowWorking。

1. CalculateOriginal方法接收一个float类型的参数result，并计算出result加1再减去5的结果，然后将这个结果乘以5除以8得到的结果，最后再将这个结果减去3。所以，CalculateOriginal方法的目的是为了计算一个指定参数的原始值，使其加上一个固定的值后再进行计算。

2. ShowWorking方法接收一个Number类型的参数value，并使用string.Format方法输出一个字符串，这个字符串包含了多个占位符，用于将value的值逐步输出。首先将value的值加3，然后将value除以5，再将value乘以8，将value加5，将value除以5加5，最后将value的值输出。所以，ShowWorking方法的目的是为了输出一个字符串，显示Math类中定义的计算结果。


```
using static Chief.Resources.Resource;

namespace Chief;

public static class Math
{
    public static float CalculateOriginal(float result) => (result + 1 - 5) * 5 / 8 * 5 - 3;

    public static string ShowWorking(Number value) =>
        string.Format(
            Formats.Working,
            value,
            value += 3,
            value /= 5,
            value *= 8,
            value = value / 5 + 5,
            value - 1);
}
```

# `25_Chief/csharp/Program.cs`

这段代码使用了两个全局变量，一个是 using Games.Common.IO，另一个是 using Chief。

using Games.Common.IO 是一个从 Games.Common.IO 命名空间中继承的 I/O 类。这个类的成员可能包括与文件、文件夹、网络连接等相关的操作。

using Chief 是一个从 Chief 命名空间中继承的类。这个类可能与程序的启动、配置、日志等功能相关。

new Game(new ConsoleIO()).Play() 创建了一个 Game 对象，并调用其的 Play() 方法来启动游戏。

具体来说，new Game(new ConsoleIO()).Play() 创建了一个新的 Game 对象，其中 new ConsoleIO() 创建了一个新的 ConsoleIO 对象，这个对象可能是一个新的游戏客户端或者是一个用于与游戏服务器通信的客户端。然后，Game 对象中的 new ConsoleIO().Play() 方法启动了游戏客户端，并开始输出游戏日志到 Console。


```
global using Games.Common.IO;
global using Chief;

new Game(new ConsoleIO()).Play();
```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `25_Chief/csharp/Resources/Resource.cs`



这段代码是一个自定义的 .NET 类，名为 `Resource`。它包含一个 `Streams` 类和一个 `Formats` 类，以及一个 `Prompts` 类。

`.NET` 中的 `System.Reflection` 命名空间包含许多与程序的资源相关的类和接口，例如 `Assembly` 和 `Stream` 类。

`Resource` 类中的 `Streams` 类定义了 5 个 `Stream` 类型，分别为 `Bye`、`Instructions`、`Lightning`、`ShutUp` 和 `Title`，这些 Stream 类型没有明确的含义，但根据其名称可以猜测它们可能与程序的界面、日志或其他信息相关。

`Formats` 类中的 `string` 类型定义了 2 个字符串类型，分别为 `Bet` 和 `Working`，但同样没有明确的含义。

`Prompts` 类中的 `string` 类型定义了 4 个字符串类型，分别为 `Answer`、`Believe`、`Original` 和 `Ready`，同样没有明确的含义。

`GetStream` 方法是一个私有方法，它使用一个 `string` 类型和一个可选的 `name` 参数，返回一个与传给它的 `name` 有关联的 `Stream` 对象。

`GetString` 方法是一个私有方法，它使用上面定义的 `Stream` 对象和一个可选的 `name` 参数，返回一个字符串的 RFC 8223 编码形式的资源资源名称。


```
using System.Reflection;
using System.Runtime.CompilerServices;

namespace Chief.Resources;

internal static class Resource
{
    internal static class Streams
    {
        public static Stream Bye => GetStream();
        public static Stream Instructions => GetStream();
        public static Stream Lightning => GetStream();
        public static Stream ShutUp => GetStream();
        public static Stream Title => GetStream();
    }

    internal static class Formats
    {
        public static string Bet => GetString();
        public static string Working => GetString();
    }

    internal static class Prompts
    {
        public static string Answer => GetString();
        public static string Believe => GetString();
        public static string Original => GetString();
        public static string Ready => GetString();
    }

    private static string GetString([CallerMemberName] string? name = null)
    {
        using var stream = GetStream(name);
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }

    private static Stream GetStream([CallerMemberName] string? name = null)
        => Assembly.GetExecutingAssembly().GetManifestResourceStream($"Chief.Resources.{name}.txt")
            ?? throw new ArgumentException($"Resource stream {name} does not exist", nameof(name));
}
```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `25_Chief/java/src/Chief.java`

This is a Java program that appears to simulate a simplified version of the game 2048. The program has several main classes including Chief, Chief's Numbers, and Main.

The Chief class has a constructor that initializes the Chief's Numbers to be a large integer value and a method int() that returns a large integer value. The method does some processing with numbers and then returns it.

The Main class has a constructor that initializes the Chief's Numbers to be a large integer value, a string of large numbers, and a boolean flag indicating whether to display the numbers on the screen or not. It also has a displayTextAndGetInput method that displays a message on the screen and then accepts input from the keyboard.

The program also has a StringIsAnyValue class that appears to check if a given string is equal to one of the values specified in the constructor's init() method. It has a yesEntered method that returns true if the string is equal to one of the values.

Overall, this program appears to be a simple game 2048 simulation that allows the user to interact with the game by displaying numbers on the screen or not, and accepts user input to continue the game.


```
import java.util.Arrays;
import java.util.Scanner;

/**
 * Game of Chief
 * <p>
 * Based on the Basic game of Hurkle here
 * https://github.com/coding-horror/basic-computer-games/blob/main/25%20Chief/chief.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Chief {

    private enum GAME_STATE {
        STARTING,
        READY_TO_START,
        ENTER_NUMBER,
        CALCULATE_AND_SHOW,
        END_GAME,
        GAME_OVER
    }

    private GAME_STATE gameState;

    // The number the computer determines to be the players starting number
    private double calculatedNumber;

    // Used for keyboard input
    private final Scanner kbScanner;

    public Chief() {

        gameState = GAME_STATE.STARTING;

        // Initialise kb scanner
        kbScanner = new Scanner(System.in);
    }

    /**
     * Main game loop
     */
    public void play() {

        do {
            switch (gameState) {

                // Show an introduction the first time the game is played.
                case STARTING:
                    intro();
                    gameState = GAME_STATE.READY_TO_START;
                    break;

                // show an message to start
                case READY_TO_START:
                    if (!yesEntered(displayTextAndGetInput("ARE YOU READY TO TAKE THE TEST YOU CALLED ME OUT FOR? "))) {
                        System.out.println("SHUT UP, PALE FACE WITH WISE TONGUE.");
                    }

                    instructions();
                    gameState = GAME_STATE.ENTER_NUMBER;
                    break;

                // Enter the number to be used to calculate
                case ENTER_NUMBER:
                    double playerNumber = Double.parseDouble(
                            displayTextAndGetInput(" WHAT DO YOU HAVE? "));

                    // Exact same formula used in the original game to calculate the players original number
                    calculatedNumber = (playerNumber + 1 - 5) * 5 / 8 * 5 - 3;

                    gameState = GAME_STATE.CALCULATE_AND_SHOW;
                    break;

                // Enter the number to be used to calculate
                case CALCULATE_AND_SHOW:
                    if (yesEntered(
                            displayTextAndGetInput("I BET YOUR NUMBER WAS " + calculatedNumber
                                    + ". AM I RIGHT? "))) {
                        gameState = GAME_STATE.END_GAME;

                    } else {
                        // Player did not agree, so show the breakdown
                        double number = Double.parseDouble(
                                displayTextAndGetInput(" WHAT WAS YOUR ORIGINAL NUMBER? "));
                        double f = number + 3;
                        double g = f / 5;
                        double h = g * 8;
                        double i = h / 5 + 5;
                        double j = i - 1;
                        System.out.println("SO YOU THINK YOU'RE SO SMART, EH?");
                        System.out.println("NOW WATCH.");
                        System.out.println(number + " PLUS 3 EQUALS " + f + ". DIVIDED BY 5 EQUALS " + g);
                        System.out.println("TIMES 8 EQUALS " + h + ". IF WE DIVIDE BY 5 AND ADD 5,");
                        System.out.println("WE GET " + i + ", WHICH, MINUS 1, EQUALS " + j + ".");
                        if (yesEntered(displayTextAndGetInput("NOW DO YOU BELIEVE ME? "))) {
                            gameState = GAME_STATE.END_GAME;
                        } else {
                            // Time for a lightning bolt.
                            System.out.println("YOU HAVE MADE ME MAD!!!");
                            System.out.println("THERE MUST BE A GREAT LIGHTNING BOLT!");
                            System.out.println();
                            for (int x = 30; x >= 22; x--) {
                                System.out.println(tabbedSpaces(x) + "X X");
                            }
                            System.out.println(tabbedSpaces(21) + "X XXX");
                            System.out.println(tabbedSpaces(20) + "X   X");
                            System.out.println(tabbedSpaces(19) + "XX X");
                            for (int y = 20; y >= 13; y--) {
                                System.out.println(tabbedSpaces(y) + "X X");
                            }
                            System.out.println(tabbedSpaces(12) + "XX");
                            System.out.println(tabbedSpaces(11) + "X");
                            System.out.println(tabbedSpaces(10) + "*");
                            System.out.println();
                            System.out.println("#########################");
                            System.out.println();
                            System.out.println("I HOPE YOU BELIEVE ME NOW, FOR YOUR SAKE!!");
                            gameState = GAME_STATE.GAME_OVER;
                        }

                    }
                    break;

                // Sign off message for cases where the Chief is not upset
                case END_GAME:
                    System.out.println("BYE!!!");
                    gameState = GAME_STATE.GAME_OVER;
                    break;

                // GAME_OVER State does not specifically have a case
            }
        } while (gameState != GAME_STATE.GAME_OVER);
    }

    /**
     * Simulate tabs by building up a string of spaces
     *
     * @param spaces how many spaces are there to be
     * @return a string with the requested number of spaces
     */
    private String tabbedSpaces(int spaces) {
        char[] repeat = new char[spaces];
        Arrays.fill(repeat, ' ');
        return new String(repeat);
    }

    private void instructions() {
        System.out.println(" TAKE A NUMBER AND ADD 3. DIVIDE NUMBER BY 5 AND");
        System.out.println("MULTIPLY BY 8. DIVIDE BY 5 AND ADD THE SAME. SUBTRACT 1.");
    }

    /**
     * Basic information about the game
     */
    private void intro() {
        System.out.println("CHIEF");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("I AM CHIEF NUMBERS FREEK, THE GREAT INDIAN MATH GOD.");
    }

    /**
     * Returns true if a given string is equal to at least one of the values specified in the call
     * to the stringIsAnyValue method
     *
     * @param text string to search
     * @return true if string is equal to one of the varargs
     */
    private boolean yesEntered(String text) {
        return stringIsAnyValue(text, "Y", "YES");
    }

    /**
     * Returns true if a given string contains at least one of the varargs (2nd parameter).
     * Note: Case insensitive comparison.
     *
     * @param text   string to search
     * @param values varargs of type string containing values to compare
     * @return true if one of the varargs arguments was found in text
     */
    private boolean stringIsAnyValue(String text, String... values) {

        // Cycle through the variable number of values and test each
        for (String val : values) {
            if (text.equalsIgnoreCase(val)) {
                return true;
            }
        }

        // no matches
        return false;
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

}

```

# `25_Chief/java/src/ChiefGame.java`

这段代码定义了一个名为“ChiefGame”的类，该类包含一个名为“main”的静态方法，其参数为一个字符串数组“args”。

在“main”方法中，首先创建了一个名为“chief”的 instance变量，该变量继承自名为“Chief”的类，但并未定义任何方法。

接着，使用“new”关键字，调用了“Chief”类中名为“play”的静态方法，该方法的具体实现未在代码中给出。


```
public class ChiefGame {

    public static void main(String[] args) {

        Chief chief = new Chief();
        chief.play();
    }
}

```

# `25_Chief/javascript/chief.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。它们的作用如下：

1. `print` 函数的作用是在文档中的一个元素（id 为 `output` 的元素）中添加文本内容，内容由参数 `str` 指定。这个文本内容将以 `<br>` 标签换行并添加到 `output` 元素中。

2. `input` 函数的作用是获取用户输入的字符串 `input_str`。它首先创建一个 `<input>` 元素，设置其 `type` 属性为 `text`，设置其 `length` 属性为 `50`，然后将该元素添加到文档中的一个元素中（这个元素在 `print` 函数中创建），并为该元素添加一个 `keydown` 事件监听器。当用户按下键盘上的 `13` 键时，该函数会将 `input_str` 赋值给变量 `input_str`，并将其添加到文档中的一个元素中（这个元素在 `print` 函数中创建），最后将 `input_str` 打印出来。


```
// CHIEF
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

The function `parseFloat(await input())` is likely to return the result of the `input()` function, which prompts the user to enter a number. The function `f = k + 3;` sets the variable `f` to the result of adding `3` to the variable `k`. The variable `g` is then set to `f / 5`, and the variable `h` is set to `g * 8`. The variable `i` is set to `h / 5 + 5`, and the variable `j` is set to `i - 1`. The code then prints a series of equations and statements, which are likely intended to demonstrate some kind of problem-solving scenario. It is not clear what the function does, as it does not have any specific implementation.



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
    print(tab(30) + "CHIEF\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("I AM CHIEF NUMBERS FREEK, THE GREAT INDIAN MATH GOD.\n");
    print("ARE YOU READY TO TAKE THE TEST YOU CALLED ME OUT FOR");
    a = await input();
    if (a.substr(0, 1) != "Y")
        print("SHUT UP, PALE FACE WITH WIE TONGUE.\n");
    print(" TAKE A NUMBER AND ADD 3. DIVIDE THIS NUMBER BY 5 AND\n");
    print("MULTIPLY BY 8. DIVIDE BY 5 AND ADD THE SAME. SUBTRACT 1.\n");
    print("  WHAT DO YOU HAVE");
    b = parseFloat(await input());
    c = (b + 1 - 5) * 5 / 8 * 5 - 3;
    print("I BET YOUR NUMBER WAS " + Math.floor(c + 0.5) + ". AM I RIGHT");
    d = await input();
    if (d.substr(0, 1) != "Y") {
        print("WHAT WAS YOUR ORIGINAL NUMBER");
        k = parseFloat(await input());
        f = k + 3;
        g = f / 5;
        h = g * 8;
        i = h / 5 + 5;
        j = i - 1;
        print("SO YOU THINK YOU'RE SO SMART, EH?\n");
        print("NOW WATCH.\n");
        print(k + " PLUS 3 EQUALS " + f + ". THIS DIVIDED BY 5 EQUALS " + g + ";\n");
        print("THIS TIMES 8 EQUALS " + h + ". IF WE DIVIDE BY 5 AND ADD 5,\n");
        print("WE GET " + i + ", WHICH, MINUS 1, EQUALS " + j + ".\n");
        print("NOW DO YOU BELIEVE ME");
        z = await input();
        if (z.substr(0, 1) != "Y") {
            print("YOU HAVE MADE ME MAD!!!\n");
            print("THERE MUST BE A GREAT LIGHTNING BOLT!\n");
            print("\n");
            print("\n");
            for (x = 30; x >= 22; x--)
                print(tab(x) + "X X\n");
            print(tab(21) + "X XXX\n");
            print(tab(20) + "X   X\n");
            print(tab(19) + "XX X\n");
            for (y = 20; y >= 13; y--)
                print(tab(y) + "X X\n");
            print(tab(12) + "XX\n");
            print(tab(11) + "X\n");
            print(tab(10) + "*\n");
            print("\n");
            print("#########################\n");
            print("\n");
            print("I HOPE YOU BELIEVE ME NOW, FOR YOUR SAKE!!\n");
            return;
        }
    }
    print("BYE!!!\n");
}

```

这道题目是一个简单的C语言程序，其中包含了一个主函数（main function）。程序的主要作用是输出"Hello World!"。

在C语言中，main函数是程序的入口点，当程序运行时，首先会执行main函数。所以，在main函数中，程序可以做一些初始化操作，也可以做一些输出操作，以此开始程序的执行。

但是，在这道题目中，main函数没有任何语句，所以它无法对程序进行任何操作。因此，这道题目的答案就是：main函数没有对程序进行任何操作，因此它的作用就是输出"Hello World!"。


```
main();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/) by Alex Conconi

---

### Lua porting notes

- I did not like the old Western movie language style in the game introduction
and decided to tone it down, even if this deviates from the original BASIC
version.

- The `craps_game` function contains the main game logic: it
  - prints the game credits and presents the intro question;
  - asks for the end result and computes the original numer;
  - calls `explain_solution` to print the various steps of the computation;
  - presents the outro question and prints a `bolt` if necessary.

- Added basic input validation to accept only valid integers for numeric input.

- Minor formatting edits (lowercase, punctuation).

- Any answer to a "yes or no" question is regarded as "yes" if the input line
starts with 'y' or 'Y', else no.


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


# `25_Chief/python/chief.py`

这段代码是一个 Python 函数，名为 `print_lightning_bolt()`。它没有定义任何变量，直接输出一个带箭头的闪电符号，并在符号前面输出一系列的星号，当符号数量减至 1 时，输出一条感叹号。符号的数量随着函数调用不断递减，每次输出一个带箭头的闪电符号，符号数量减少 1；然后在符号前面输出一系列星号，符号数量随着函数调用也减少 1；接着输出一系列带箭头的闪电符号，符号数量再减少 1；不断重复这个过程，直到符号数量减少到 8，输出一个感叹号，符号数量减少到 7，输出一个带箭头的闪电符号，符号数量减少到 6，输出一个带箭头的闪电符号，符号数量减少到 5，输出一个带箭头的闪电符号，符号数量减少到 4，输出一个带箭头的闪电符号，符号数量减少到 3，输出一个带箭头的闪电符号，符号数量减少到 2，输出一个带箭头的闪电符号，符号数量减少到 1，输出一个带箭头的闪电符号，符号数量减少到 0，输出一个带箭头的闪电符号，符号数量减少到 -1，并输出一条感叹号，符号数量减少到 -2，输出一个带箭头的闪电符号，符号数量减少到 -3，输出一个带箭头的闪电符号，符号数量减少到 -4，输出一个带箭头的闪电符号，符号数量减少到 -5，输出一个带箭头的闪电符号，符号数量减少到 -6，输出一个带箭头的闪电符号，符号数量减少到 -7，并输出一条感叹号，符号数量减少到 -8。


```
def print_lightning_bolt() -> None:
    print("*" * 36)
    n = 24
    while n > 16:
        print(" " * n + "x x")
        n -= 1
    print(" " * 16 + "x xxx")
    print(" " * 15 + "x   x")
    print(" " * 14 + "xxx x")
    n -= 1
    while n > 8:
        print(" " * n + "x x")
        n -= 1
    print(" " * 8 + "xx")
    print(" " * 7 + "x")
    print("*" * 36)


```

This is a program written in Python that gives you a number and asks you to perform a series of mathematical operations with that number. It then compares your answers to the correct solutions and gives you feedback on whether you are correct or not.

Here is a summary of the program:

1. The program asks you to take a number and add 3 to it.
2. It then asks you to divide that number by 5 and multiply the result by 8.
3. Finally, it asks you to divide the result by 5 and add the same number.
4. The program then gives you a number and asks you to guess it.
5. If you think you have the correct answer, it will tell you.
6. If not, it will give you the correct solution to the original number.
7. If you still think you know the correct answer, it will tell you to try again.
8. If you continue to make incorrect guesses, it will tell you that you have made it very bad and that you should calm down.

It is important to note that this program is not very good at math and it is not meant for serious people.


```
def print_solution(n: float) -> None:
    print(f"\n{n} plus 3 gives {n + 3}. This Divided by 5 equals {(n + 3) / 5}")
    print(f"This times 8 gives {((n + 3) / 5) * 8}. If we divide 5 and add 5.")
    print(
        f"We get {(((n + 3) / 5) * 8) / 5 + 5}, "
        f"which, minus 1 equals {((((n + 3) / 5) * 8) / 5 + 5) - 1}"
    )


def game() -> None:
    print("\nTake a Number and ADD 3. Now, Divide this number by 5 and")
    print("multiply by 8. Now, Divide by 5 and add the same. Subtract 1")

    you_have = float(input("\nWhat do you have? "))
    comp_guess = (((you_have - 4) * 5) / 8) * 5 - 3
    first_guess_right = input(
        f"\nI bet your number was {comp_guess} was I right(Yes or No)? "
    )

    if first_guess_right.lower() == "yes":
        print("\nHuh, I Knew I was unbeatable")
        print("And here is how i did it")
        print_solution(comp_guess)
        input()
    else:
        original_number = float(input("\nHUH!! what was you original number? "))

        if original_number == comp_guess:
            print("\nThat was my guess, AHA i was right")
            print(
                "Shamed to accept defeat i guess, don't worry you can master mathematics too"
            )
            print("Here is how i did it")
            print_solution(comp_guess)
            input()
        else:
            print("\nSo you think you're so smart, EH?")
            print("Now, Watch")
            print_solution(original_number)

            believe_me = input("\nNow do you believe me? ")

            if believe_me.lower() == "yes":
                print("\nOk, Lets play again sometime bye!!!!")
                input()
            else:
                print("\nYOU HAVE MADE ME VERY MAD!!!!!")
                print("BY THE WRATH OF THE MATHEMATICS AND THE RAGE OF THE GODS")
                print("THERE SHALL BE LIGHTNING!!!!!!!")
                print_lightning_bolt()
                print("\nI Hope you believe me now, for your own sake")
                input()


```

这段代码是一个 Python 程序，它的作用是回答一个用户输入的问题：是愿意参加一个数学考试，还是不参加。如果用户输入 "是"，那么程序会调用一个名为 "game" 的函数，这个函数还没有被定义。如果用户输入 "否"，那么程序会输出一段文本，然后 Prompt用户再次输入。


```
if __name__ == "__main__":
    print("I am CHIEF NUMBERS FREEK, The GREAT INDIAN MATH GOD.")
    play = input("\nAre you ready to take the test you called me out for(Yes or No)? ")
    if play.lower() == "yes":
        game()
    else:
        print("Ok, Nevermind. Let me go back to my great slumber, Bye")
        input()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by [Jadi](https://github.com/jadijadi)

Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Chomp

This program is an adaptation of a mathematical game originally described by Martin Gardner in the January 1973 issue of _Scientific American_. Up to a 9x9 grid is set up by you with the upper left square in a poison square. This grid is the cookie. Players alternately chomp away at the cookie from the lower right. To take a chomp, input a row and column number of one of the squares remaining on the cookie. All of the squares below and to the right of that square, including that square, disappear.

Any number of people can play — the computer is only the moderator; it is not a player. Two-person strategies are interesting to work out but strategies when three or more people are playing are the real challenge.

The computer version of the game was written by Peter Sessions of People’s Computer Company.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=44)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=59)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `26_Chomp/csharp/Cookie.cs`

This is a class that represents a cookie that can be clicked to execute code in a Chrome browser. It has a row and column count, as well as an array of bits that represent the state of the cookie (0 for not run, 1 for run). It has a tryChomp method that takes a row and column number and tries to execute the code in that location, and a toString method that returns a string representation of the cookie.


```
using System.Text;

namespace Chomp;

internal class Cookie
{
    private readonly int _rowCount;
    private readonly int _columnCount;
    private readonly char[][] _bits;

    public Cookie(int rowCount, int columnCount)
    {
        _rowCount = rowCount;
        _columnCount = columnCount;

        // The calls to Math.Max here are to duplicate the original behaviour
        // when negative values are given for the row or column count.
        _bits = new char[Math.Max(_rowCount, 1)][];
        for (int row = 0; row < _bits.Length; row++)
        {
            _bits[row] = Enumerable.Repeat('*', Math.Max(_columnCount, 1)).ToArray();
        }
        _bits[0][0] = 'P';
    }

    public bool TryChomp(int row, int column, out char chomped)
    {
        if (row < 1 || row > _rowCount || column < 1 || column > _columnCount || _bits[row - 1][column - 1] == ' ')
        {
            chomped = default;
            return false;
        }

        chomped = _bits[row - 1][column - 1];

        for (int r = row; r <= _rowCount; r++)
        {
            for (int c = column; c <= _columnCount; c++)
            {
                _bits[r - 1][c - 1] = ' ';
            }
        }

        return true;
    }

    public override string ToString()
    {
        var builder = new StringBuilder().AppendLine("       1 2 3 4 5 6 7 8 9");
        for (int row = 1; row <= _bits.Length; row++)
        {
            builder.Append(' ').Append(row).Append("     ").AppendLine(string.Join(' ', _bits[row - 1]));
        }
        return builder.ToString();
    }
}
```

# `26_Chomp/csharp/Game.cs`

这段代码是一个名为“Chomp”的namespace内部类“Game”。

这个Game类的作用是玩一个简单的游戏，玩家需要输入“Do you want the rules (1=Yes, 0=No!)”来选择游戏规则，否则游戏继续进行。

游戏开始时，它会输出“Introduction”字符串，然后等待玩家输入是否要开始游戏。如果玩家选择“Yes”，它会输出“Rules”字符串，否则继续等待下一次玩家操作。

在游戏的循环中，它先输出“HereWeGo”字符串，然后等待玩家的输入。每次玩家输入一个坐标值（row和column），它调用一个名为“Chomp”的内部函数来处理玩家的行动。如果“Chomp”函数返回 true，说明玩家成功攻击了，否则游戏继续进行。

如果玩家在游戏中使用“Chomp”函数时，参数是一个有效的坐标（row和column），它尝试攻击另一个玩家，并输出“YouLose”字符串。如果玩家再次选择“Yes”，游戏循环将停止，否则继续游戏。


```
namespace Chomp;

internal class Game
{
    private readonly IReadWrite _io;

    public Game(IReadWrite io)
    {
        _io = io;
    }

    internal void Play()
    {
        _io.Write(Resource.Streams.Introduction);
        if (_io.ReadNumber("Do you want the rules (1=Yes, 0=No!)") != 0)
        {
            _io.Write(Resource.Streams.Rules);
        }

        while (true)
        {
            _io.Write(Resource.Streams.HereWeGo);

            var (playerCount, rowCount, columnCount) = _io.ReadParameters();

            var loser = Play(new Cookie(rowCount, columnCount), new PlayerNumber(playerCount));

            _io.WriteLine(string.Format(Resource.Formats.YouLose, loser));

            if (_io.ReadNumber("Again (1=Yes, 0=No!)") != 1) { break; }
        }
    }

    private PlayerNumber Play(Cookie cookie, PlayerNumber player)
    {
        while (true)
        {
            _io.WriteLine(cookie);

            var poisoned = Chomp(cookie, player);

            if (poisoned) { return player; }

            player++;
        }
    }

    private bool Chomp(Cookie cookie, PlayerNumber player)
    {
        while (true)
        {
            _io.WriteLine(string.Format(Resource.Formats.Player, player));

            var (row, column) = _io.Read2Numbers(Resource.Prompts.Coordinates);

            if (cookie.TryChomp((int)row, (int)column, out char chomped))
            {
                return chomped == 'P';
            }

            _io.Write(Resource.Streams.NoFair);
        }
    }
}

```

# `26_Chomp/csharp/IOExtensions.cs`

这段代码是一个名为 "IOExtensions" 的命名空间类，其目的是扩展了 "ReadParameters" 和 "ReadNumberWithMax" 方法的功能。

具体来说，这个类包含了一个名为 "ReadParameters" 的静态方法，其接收一个 "IReadWrite" 类型的参数 "io"。这个方法返回三个整数参数，分别是一个整数类型的变量、一个浮点数类型的变量和一个整数类型的变量，它们分别代表着玩家数量、最大行数和最大列数。这个方法使用了 "Resource.Prompts" 类中的 "HowManyPlayers"、"HowManyRows" 和 "HowManyColumns" 方法获取用户输入。

另外，这个类还包含一个名为 "ReadNumberWithMax" 的静态方法，它也接收一个 "IReadWrite" 类型的参数 "io" 和一个字符串类型的参数 "initialPrompt"。这个方法在初始化之后，会一直循环读取用户输入，直到输入值不小于 9 时才会停止。在循环中，每次会获取用户输入的一个整数，并更新 "initialPrompt" 变量的值。

总结起来，这个代码的作用是帮助开发者处理用户输入并返回所需的参数，使得代码更加易于读取和维护。


```
namespace Chomp;

internal static class IOExtensions
{
    public static (float, int, int) ReadParameters(this IReadWrite io)
        => (
            (int)io.ReadNumber(Resource.Prompts.HowManyPlayers),
            io.ReadNumberWithMax(Resource.Prompts.HowManyRows, 9, Resource.Strings.TooManyRows),
            io.ReadNumberWithMax(Resource.Prompts.HowManyColumns, 9, Resource.Strings.TooManyColumns)
        );

    private static int ReadNumberWithMax(this IReadWrite io, string initialPrompt, int max, string reprompt)
    {
        var prompt = initialPrompt;

        while (true)
        {
            var response = io.ReadNumber(prompt);
            if (response <= 9) { return (int)response; }

            prompt = $"{reprompt} {initialPrompt.ToLowerInvariant()}";
        }
    }
}
```

# `26_Chomp/csharp/PlayerNumber.cs`

这段代码定义了一个名为 `PlayerNumber` 的内部类，该类用于跟踪游戏中玩家的数量。

该类包含三个私有成员变量：`_playerCount`、`_counter` 和 `_number`。其中，`_playerCount` 是一个浮点数变量，用于跟踪玩家的数量；`_counter` 是一个整数变量，用于跟踪 `_playerCount` 的值递增；`_number` 是一个浮点数变量，用于存储当前玩家的数量。

该类包含一个名为 `Increment()` 的方法，用于递增 `_number` 的值。该方法的实现与原始程序相同，但是对 `_playerCount` 的值进行了检查，以避免除以零的异常。

该类还包含一个名为 `operator ++()` 的重载运算符方法，用于递增 `_number` 的值并返回 `this`。

在 `toString()` 方法中，将 `_number` 的值格式化为字符串，并在字符串前面加上 `<br />`。

总之，该代码定义了一个用于跟踪游戏中玩家数量的对象，并提供了增加玩家数量的方法。


```
namespace Chomp;

internal class PlayerNumber
{
    private readonly float _playerCount;
    private int _counter;
    private float _number;

    // The original code does not constrain playerCount to be an integer
    public PlayerNumber(float playerCount)
    {
        _playerCount = playerCount;
        _number = 0;
        Increment();
    }

    public static PlayerNumber operator ++(PlayerNumber number) => number.Increment();

    private PlayerNumber Increment()
    {
		if (_playerCount == 0) { throw new DivideByZeroException(); }

        // The increment logic here is the same as the original program, and exhibits
        // interesting behaviour when _playerCount is not an integer.
        _counter++;
        _number = _counter - (float)Math.Floor(_counter / _playerCount) * _playerCount;
        if (_number == 0) { _number = _playerCount; }
        return this;
    }

    public override string ToString() => (_number >= 0 ? " " : "") + _number.ToString();
}
```

# `26_Chomp/csharp/Program.cs`

这段代码的作用是在游戏引擎中打开一个名为“新游戏”的窗口，并在其中运行一个名为“common.io”的文件。

具体来说，以下代码会执行以下操作：

1. 全局变量“using Games.Common.IO”和“using Chomp.Resources”被创建。
2. 全局变量“new Game(new ConsoleIO()).Play()”创建了一个名为“Game”的类对象，并调用了一个名为“Play”的接口方法， passing in a new“console.IO”类型的实例作为参数。
3. “new Game(new ConsoleIO()).Play()”返回了一个“Game”类对象，这个对象被赋值给全局变量“gameObject”。
4. “gameObject”这个全局变量被用来创建一个新的“ConsoleGame”窗口，并在其中显示新的游戏。


```
global using Games.Common.IO;
global using Chomp.Resources;
using Chomp;

new Game(new ConsoleIO()).Play();
```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `26_Chomp/csharp/Resources/Resource.cs`



该代码是一个自定义的程序集，名为“Chomp.Resources”。它包含了一些定义，包括Streams、Formats、Prompts和Strings类，以及它们的静态属性。

作用是创建一个名为“Chomp.Resources”的程序集，其中包含一些定义类和静态属性，用于在游戏中的信息和提示。

具体来说，该程序集定义了以下几种资源类型：

- Streams类：包含4个静态方法，分别用于获取当前游戏中的流，介绍，规则和公平局的流。
- Formats类：包含2个静态方法，用于生成玩家输出的字符串。
- Prompts类：包含4个静态方法，分别用于获取当前游戏中的坐标，玩家数量，棋盘数量和棋盘深度。
- Strings类：包含2个静态方法，分别用于获取当前游戏中过多的列和行，以及过度的行和列。

此外，还包含GetString方法，用于从游戏资源文件中读取字符串，并使用Assembly.GetExecutingAssembly().GetManifestResourceStream方法获取资源文件的信息。

最后，定义了一个名为“Chomp.Resources”的类，通过调用GetString、GetStream等方法，来获取游戏中的信息和提示。


```
using System.Reflection;
using System.Runtime.CompilerServices;

namespace Chomp.Resources;

internal static class Resource
{
    internal static class Streams
    {
        public static Stream HereWeGo => GetStream();
        public static Stream Introduction => GetStream();
        public static Stream Rules => GetStream();
        public static Stream NoFair => GetStream();
    }

    internal static class Formats
    {
        public static string Player => GetString();
        public static string YouLose => GetString();
    }

    internal static class Prompts
    {
        public static string Coordinates => GetString();
        public static string HowManyPlayers => GetString();
        public static string HowManyRows => GetString();
        public static string HowManyColumns => GetString();
        public static string TooManyColumns => GetString();
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

# `26_Chomp/java/Chomp.java`

This is a Java program that simulates a game of Connect-the-Dots for two players. The program uses a grid-based interface to display the Connect-the-Dot board, as well as a scanner to accept the player's input.

The program starts by creating a Connect-the-Dot board, which is displayed with a prompt asking the player to enter the coordinates of a space they want to click on. The program then enters a while loop that continues until the player clicks on an empty space, at which point it displays an error message and prompts the player to enter a valid coordinate.

If the player enters a valid coordinate, the program displays the grid-based Connect-the-Dot board and updates the player's position by incrementing their turn number by 1. The program then prints out the Connect-the-Dot board and updates the board's state by changing the value of the space the player clicked on.

The program also has a move function that allows the player to move to another space on the board by taking the turn number of the space they clicked on.

The program has a main method that initializes the Connect-the-Dot board and starts a game loop that displays the board and waits for the player to make a move.

Note: This program assumes that the Connect-the-Dot board has a maximum size of the numberOfPlayers.


```
import java.util.Scanner;
public class Chomp{
	int rows;
	int cols;
	int numberOfPlayers;
	int []board;
	Scanner scanner;
	Chomp(){
		System.out.println("\t\t\t\tCHOMP");
		System.out.println("\t\tCREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
		System.out.println("THIS IS THE GAME OF CHOMP (SCIENTIFIC AMERICAN, JAN 1973)");
		System.out.print("Do you want the rules (1=Yes, 0=No!)  ");

		scanner = new Scanner(System.in);
		int choice = scanner.nextInt();
		if(choice != 0){
			System.out.println("Chomp is for 1 or more players (Humans only).\n");
			System.out.println("Here's how a board looks (This one is 5 by 7):");
			System.out.println("\t1 2 3 4 5 6 7");
			System.out.println(" 1     P * * * * * *\n 2     * * * * * * *\n 3     * * * * * * *\n 4     * * * * * * *\n 5     * * * * * * *");
			System.out.println("\nThe board is a big cookie - R rows high and C columns \nwide. You input R and C at the start. In the upper left\ncorner of the cookie is a poison square (P). The one who\nchomps the poison square loses. To take a chomp, type the\nrow and column of one of the squares on the cookie.\nAll of the squares below and to the right of that square\n(Including that square, too) disappear -- CHOMP!!\nNo fair chomping squares that have already been chomped,\nor that are outside the original dimensions of the cookie.\n");
			System.out.println("Here we go...\n");
		}
		startGame();
	}

	private void startGame(){
		System.out.print("How many players ");
		numberOfPlayers = scanner.nextInt();
		while(numberOfPlayers < 2){
			System.out.print("How many players ");
                	numberOfPlayers = scanner.nextInt();
		}
		System.out.print("How many rows ");
		rows = scanner.nextInt();
		while(rows<=0 || rows >9){
			if(rows <= 0){
				System.out.println("Minimun 1 row is required !!");
			}
			else{
				System.out.println("Too many rows(9 is maximum). ");
			}
			System.out.print("How many rows ");
			rows = scanner.nextInt();
		}
		System.out.print("How many columns ");
                cols = scanner.nextInt();
                while(cols<=0 || cols >9){
                        if(cols <= 0){
                                System.out.println("Minimun 1 column is required !!");
                        }
                        else{
                                System.out.println("Too many columns(9 is maximum). ");
                        }
                        System.out.print("How many columns ");
                        cols = scanner.nextInt();
                }
		board = new int[rows];
		for(int i=0;i<rows;i++){
			board[i]=cols;
		}
		printBoard();
		scanner.nextLine();
		move(0);
	}

	private void printBoard(){
		System.out.print("        ");
		for(int i=0;i<cols;i++){
			System.out.print(i+1);
			System.out.print(" ");
		}
		for(int i=0;i<rows;i++){
			System.out.print("\n ");
			System.out.print(i+1);
			System.out.print("      ");
			for(int j=0;j<board[i];j++){
				if(i == 0 && j == 0){
					System.out.print("P ");
				}
				else{
					System.out.print("* ");
				}
			}
		}
		System.out.println("");
	}

	private void move(int player){
		System.out.println(String.format("Player %d",(player+1)));

		String input;
		String [] coordinates;
		int x=-1,y=-1;
		while(true){
			try{
				System.out.print("Coordinates of chomp (Row, Column) ");
				input = scanner.nextLine();
				coordinates = input.split(",");
				x = Integer.parseInt(coordinates[0]);
				y = Integer.parseInt(coordinates[1]);
				break;
			}
			catch(Exception e){
				System.out.println("Please enter valid coordinates.");
				continue;
			}
		}

		while(x>rows || x <1 || y>cols || y<1 || board[x-1]<y){
			System.out.println("No fair. You're trying to chomp on empty space!");
	                while(true){
                        	try{
					System.out.print("Coordinates of chomp (Row, Column) ");
                	                input = scanner.nextLine();
        	                        coordinates = input.split(",");
	                                x = Integer.parseInt(coordinates[0]);
                        	        y = Integer.parseInt(coordinates[1]);
                	                break;
        	                }
	                        catch(Exception e){
                        	        System.out.println("Please enter valid coordinates.");
                	                continue;
        	                }
	                }
		}

		if(x == 1 && y == 1){
			System.out.println("You lose player "+(player+1));
			int choice = -1 ;
			while(choice != 0 && choice != 1){
				System.out.print("Again (1=Yes, 0=No!) ");
				choice = scanner.nextInt();
			}
			if(choice == 1){
				startGame();
			}
			else{
				System.exit(0);
			}
		}
		else{
			for(int i=x-1;i<rows;i++){
				if(board[i] >= y){
					board[i] = y-1;
				}
			}
			printBoard();
			move((player+1)%numberOfPlayers);
		}
	}


	public static void main(String []args){
		new Chomp();
	}
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `26_Chomp/javascript/chomp.js`

这段代码定义了两个函数，分别是`print()`和`input()`。

`print()`函数的作用是接收一个字符串参数，将其显示在页面上。函数内部创建了一个`<textarea>`元素，将其添加到文档中的一个`<output>`元素下，然后将字符串内容追加到该元素中。

`input()`函数的作用是从用户接收一个字符串输入。函数内部创建了一个`<input>`元素，设置了其`type`属性为"text"，`length`属性为"50"。该元素被添加到文档中的一个`<output>`元素下，并设置了一个键盘事件监听器，以便在用户按下回车键时接收输入。当用户点击该元素时，函数内部的`focus()`方法将获得焦点，并且将`input`元素的值赋给`input_str`变量。然后函数内部将`input_str`的内容输出，并输出两个空行。


```
// CHOMP
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

这段代码定义了一个名为 "tab" 的函数，用于在 2x2 的棋盘上打印出 " space " 变量所表示的字符。

在函数内部，首先定义了一个名为 "str" 的字符串变量，用于存储打印出来的字符串。然后使用 while 循环，将 "space" 变量所表示的字符一个一个地添加到 "str" 字符串的末尾，并在每次添加后减少 "space" 变量的值。最后，将 "str" 字符串返回。

该函数的作用是打印出一个 2x2 的棋盘，并在中央的单元格中添加一个空格，使得棋盘看起来像一个 " "。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var a = [];
var r;
var c;

function init_board()
{
    for (i = 1; i <= r; i++)
        for (j = 1; j <= c; j++)
            a[i][j] = 1;
    a[1][1] = -1;
}

```



这段代码定义了一个名为 `show_board` 的函数，其作用是输出一个 7x7 的表格，并在表格中输出一行行由数字和星号组成的字符串。

具体来说，函数首先输出一个空行，然后输出一个由 7 个空格和 7 个星号组成的字符串，该字符串代表表格的行边。接下来，函数使用 for 循环来遍历 1 到 7 的整数，并将每个整数与 6 进行比较，如果该整数与矩阵中的某个元素相等，则该元素对应的星号前面的空格要被替换为星号，否则该元素对应的星号前面的空格不被替换。最后，函数使用 for 循环输出字符串中的所有星号，并在字符串的每行结束后输出一个空行，使得输出结果更加美观。


```
function show_board()
{
    print("\n");
    print(tab(7) + "1 2 3 4 5 6 7 8 9\n");
    for (i = 1; i <= r; i++) {
        str = i + tab(6);
        for (j = 1; j <= c; j++) {
            if (a[i][j] == -1)
                str += "P ";
            else if (a[i][j] == 0)
                break;
            else
                str += "* ";
        }
        print(str + "\n");
    }
    print("\n");
}

```

This is a program that plays a board game called "Tic Tac Toe" on a 3x3 grid. It starts by printing the initial board, then enters a while loop that allows the player to choose which columns and rows to fill with their own "X" mark. After the player has filled in their chosen columns and rows, the program displays the board again and then enters another while loop that prints the board and checks for any unfair attempts by the player to add their own "X" marks. If the player successfully adds all their own "X" marks, the program ends and the player wins. If not, the program prints a message and enters a loop that prints the board again until the game is over.



```
// Main program
async function main()
{
    print(tab(33) + "CHOMP\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    for (i = 1; i <= 10; i++)
        a[i] = [];
    // *** THE GAME OF CHOMP *** COPYRIGHT PCC 1973 ***
    print("\n");
    print("THIS IS THE GAME OF CHOMP (SCIENTIFIC AMERICAN, JAN 1973)\n");
    print("DO YOU WANT THE RULES (1=YES, 0=NO!)");
    r = parseInt(await input());
    if (r != 0) {
        f = 1;
        r = 5;
        c = 7;
        print("CHOMP IS FOR 1 OR MORE PLAYERS (HUMANS ONLY).\n");
        print("\n");
        print("HERE'S HOW A BOARD LOOKS (THIS ONE IS 5 BY 7):\n");
        init_board();
        show_board();
        print("\n");
        print("THE BOARD IS A BIG COOKIE - R ROWS HIGH AND C COLUMNS\n");
        print("WIDE. YOU INPUT R AND C AT THE START. IN THE UPPER LEFT\n");
        print("CORNER OF THE COOKIE IS A POISON SQUARE (P). THE ONE WHO\n");
        print("CHOMPS THE POISON SQUARE LOSES. TO TAKE A CHOMP, TYPE THE\n");
        print("ROW AND COLUMN OF ONE OF THE SQUARES ON THE COOKIE.\n");
        print("ALL OF THE SQUARES BELOW AND TO THE RIGHT OF THAT SQUARE\n");
        print("INCLUDING THAT SQUARE, TOO) DISAPPEAR -- CHOMP!!\n");
        print("NO FAIR CHOMPING SQUARES THAT HAVE ALREADY BEEN CHOMPED,\n");
        print("OR THAT ARE OUTSIDE THE ORIGINAL DIMENSIONS OF THE COOKIE.\n");
        print("\n");
    }
    while (1) {
        print("HERE WE GO...\n");
        f = 0;
        for (i = 1; i <= 10; i++) {
            a[i] = [];
            for (j = 1; j <= 10; j++) {
                a[i][j] = 0;
            }
        }
        print("\n");
        print("HOW MANY PLAYERS");
        p = parseInt(await input());
        i1 = 0;
        while (1) {
            print("HOW MANY ROWS");
            r = parseInt(await input());
            if (r <= 9)
                break;
            print("TOO MANY ROWS (9 IS MAXIMUM). NOW ");
        }
        while (1) {
            print("HOW MANY COLUMNS");
            c = parseInt(await input());
            if (c <= 9)
                break;
            print("TOO MANY COLUMNS (9 IS MAXIMUM). NOW ");
        }
        print("\n");
        init_board();
        while (1) {
            // Print the board
            show_board();
            // Get chomps for each player in turn
            i1++;
            p1 = i1 - Math.floor(i1 / p) * p;
            if (p1 == 0)
                p1 = p;
            while (1) {
                print("PLAYER " + p1 + "\n");
                print("COORDINATES OF CHOMP (ROW,COLUMN)");
                str = await input();
                r1 = parseInt(str);
                c1 = parseInt(str.substr(str.indexOf(",") + 1));
                if (r1 >= 1 && r1 <= r && c1 >= 1 && c1 <= c && a[r1][c1] != 0)
                    break;
                print("NO FAIR. YOU'RE TRYING TO CHOMP ON EMPTY SPACE!\n");
            }
            if (a[r1][c1] == -1)
                break;
            for (i = r1; i <= r; i++)
                for (j = c1; j <= c; j++)
                    a[i][j] = 0;
        }
        // End of game detected
        print("YOU LOSE, PLAYER " + p1 + "\n");
        print("\n");
        print("AGAIN (1=YES, 0=NO!)");
        r = parseInt(await input());
        if (r != 1)
            break;
    }
}

```

这是C++程序的main函数，它负责程序的启动和执行。在C++中，所有程序都必须包含一个main函数，程序在启动时会首先执行这个函数。

main函数中可以包含程序中所有需要执行的代码，包括输入输出、加载资源、定义变量等等。因此，main函数是C++程序的核心部分，也是程序和用户交互的唯一途径。

程序棒喹苟有限公司。


```
main();

```