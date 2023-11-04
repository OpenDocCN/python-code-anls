# BasicComputerGames源码解析 36

# `30_Cube/csharp/Program.cs`

这段代码的作用是创建一个游戏引擎，并在游戏中使用随机数生成器和 Cube.Resources.Resource 中的资源。

首先，它导入了 Games.Common.IO 和 Games.Common.Randomness，这两个库提供了从控制台输入和生成随机数的功能。

然后，它导入了 Cube.Resources.Resource，这个库可能用于在游戏中加载和加载资源。

接着，它创建了一个名为 Games 的类，可能是游戏的主类。它还创建了一个名为 args 的参数，用于接收玩家传给游戏的主参数。

在 args 参数中，如果包含了 "--non-random" 参数，那么就会创建一个名为 ZerosGenerator 的类，并使用该类生成随机数。否则，就会创建一个名为 RandomNumberGenerator 的类，并使用该类生成随机数。

接下来，它创建了一个名为 Game 的类，可能是游戏的主类。它使用了 args 参数中的 RandomNumberGenerator 类来生成随机数，然后创建了一个 ConsoleIO 类的实例作为游戏输出的渠道。

最后，它调用了 Games.main 方法，这个方法可能是游戏的主方法，用于启动游戏。


```
global using Games.Common.IO;
global using Games.Common.Randomness;

global using static Cube.Resources.Resource;

using Cube;

IRandom random = args.Contains("--non-random") ? new ZerosGenerator() : new RandomNumberGenerator();

new Game(new ConsoleIO(), random).Play();

```

# `30_Cube/csharp/RandomExtensions.cs`

这段代码是一个名为 "RandomExtensions" 的命名空间，其中包含一个内部类 "RandomExtensions"。

这个内部类实现了三个方法，分别是：

1. "NextLocation"，该方法接收一个 IRandom 类型的随机数，以及一个包含三个整数的 bias 参数。它返回一个由 random.NextCoordinate 方法生成的位置坐标。

2. "NextCoordinate"，该方法与 "NextLocation" 类似，但只接收一个整数的 bias 参数。它返回 random.Next 方法生成的随机浮点数。

3. "RandomExtensions.NextCoordinate"，该方法实现了 "NextCoordinate" 内部类中的方法，用于生成指定偏移量的随机浮点数。

从代码中可以看出，这个命名空间的主要目的是为了在需要时生成随机数，特别是在涉及位置坐标等概念时。通过使用 NextLocation 方法，可以确保生成位置坐标的随机性，而通过 NextCoordinate 方法，则可以确保生成浮点数的随机性。


```
namespace Cube;

internal static class RandomExtensions
{
    internal static (float, float, float) NextLocation(this IRandom random, (int, int, int) bias)
        => (random.NextCoordinate(bias.Item1), random.NextCoordinate(bias.Item2), random.NextCoordinate(bias.Item3));

    private static float NextCoordinate(this IRandom random, int bias)
    {
        var value = random.Next(3);
        if (value == 0) { value = bias; }
        return value;
    }
}
```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)

#### Execution

As noted in the main Readme file, the randomization code in the BASIC program has a switch (the variable `X`) that
allows the game to be run in a deterministic (non-random) mode.

Running the C# port without command-line parameters will play the game with random mine locations.

Running the port with a `--non-random` command-line switch will run the game with non-random mine locations.


# `30_Cube/csharp/ZerosGenerator.cs`



这段代码定义了一个名为 ZerosGenerator 的内部类，它继承自名为 IRandom 的接口。这个内部类的作用是生成随机浮点数。

在 ZerosGenerator 类中，有两个方法，NextFloat 和 PreviousFloat，它们都返回 0。第一个方法用来生成一个新的浮点数，第二个方法用来将上一个浮点数清零。

此外，还有一个名为 Reseed 的方法，它接受一个整数参数 seed，用于在生成随机数时使用种子，从而使每次生成的随机数都不同。

综上所述，这段代码定义了一个用于生成随机浮点数的类，通过NextFloat和PreviousFloat方法可以生成新的随机数，而Reseed方法则用于在生成随机数时指定种子。


```
namespace Cube;

internal class ZerosGenerator : IRandom
{
    public float NextFloat() => 0;

    public float PreviousFloat() => 0;

    public void Reseed(int seed) { }
}
```

# `30_Cube/csharp/Resources/Resource.cs`



这段代码是一个名为 `Resource` 的类，其中包含多个内部类，包括 `Streams`、`Prompts` 和 `Formats` 三个内部类。

`Streams` 类包含了一系列的静态方法，每个方法返回一个 `Stream` 对象，这些方法的名称与 Stream 对象的名称相对应，例如 `GetStream()` 方法的名称就是 `Introduction`,`Instructions` 和 `Wager` 方法的名称。这些方法的实现是通过调用 `GetStream()` 方法来获取的。

`Prompts` 类包含了一系列的静态方法，每个方法的返回类型都是一个字符串，例如 `GetString()` 方法的返回类型就是 `string` 类型。这些方法的实现是通过调用 `GetString()` 方法来获取的。

`Formats` 类包含了一个名为 `Balance` 的静态方法，返回类型也是一个字符串。

在 `GetStream()` 方法中，通过调用 `Assembly.GetExecutingAssembly().GetManifestResourceStream()` 方法，来获取一个 `Resource.Stream` 对象。如果这个方法返回一个 `null` 值，那么会抛出一个异常，并指定一个字符串作为引起这个异常的原因。

最后，在 `Main` 类的 `Run()` 方法中，创建了一个 `Resource` 类的实例，并调用 `Setup()` 和 `Cleanup()` 方法来设置和清除游戏中的资源和变量。


```
using System.Reflection;
using System.Runtime.CompilerServices;

namespace Cube.Resources;

internal static class Resource
{
    internal static class Streams
    {
        public static Stream Introduction => GetStream();
        public static Stream Instructions => GetStream();
        public static Stream Wager => GetStream();
        public static Stream IllegalMove => GetStream();
        public static Stream Bang => GetStream();
        public static Stream Bust => GetStream();
        public static Stream Congratulations => GetStream();
        public static Stream Goodbye => GetStream();
    }

    internal static class Prompts
    {
        public static string HowMuch => GetString();
        public static string BetAgain => GetString();
        public static string YourMove => GetString();
        public static string NextMove => GetString();
        public static string TryAgain => GetString();
    }

    internal static class Formats
    {
        public static string Balance => GetString();
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

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `30_Cube/java/src/Cube.java`

This is a Java class that represents a simple text-based game where the player has to choose between moving up, down, left, and right, and then enter the destination point.

The `Location` class represents the point that the player is at and the point they wish to move to. The `Location` class has an `equals()` method that checks if two `Location` objects are the same, and a `hashCode()` method that provides a unique integer identifier for the `Location` object.

The `isMoveValid()` method checks if the player's move is valid, i.e., if the player is moving only one location in any direction (up, down, left, or right).

The `readParsedBoolean()` method reads the boolean value entered by the player.

The main method initializes the game and reads the player's input. If the input is a valid boolean value, it is parsed and returned. If not, it returns `false`.

Then, it checks if the input is a valid move, and if so, it checks if the move is valid.

After that, it is used to set the initial position of the player to the origin point and the destination point to the move direction.

Then, it checks if the player is at an valid position to move, and if not, it returns `false`.

Finally, it is used to navigate the player to the destination point, and if the move was successful, it returns `true`.


```
import java.io.PrintStream;
import java.util.HashSet;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;

/**
 * Game of Cube
 * <p>
 * Based on game of Cube at:
 * https://github.com/coding-horror/basic-computer-games/blob/main/30_Cube/cube.bas
 *
 *
 */
public class Cube {

    //Current player location
    private Location playerLocation;

    //Current list of mines
    private Set<Location> mines;

    //System input / output objects
    private PrintStream out;
    private Scanner scanner;

    //Player's current money
    private int money;

    /**
     * Entry point, creates a new Cube object and calls the play method
     * @param args Java execution arguments, not used in application
     */
    public static void main(String[] args) {
        new Cube().play();
    }

    public Cube() {
        out = System.out;
        scanner = new Scanner(System.in);
        money = 500;
        mines = new HashSet<>(5);
    }

    /**
     * Clears mines and places 5 new mines on the board
     */
    private void placeMines() {
        mines.clear();
        Random random = new Random();
        for(int i = 0; i < 5; i++) {
            int x = random.nextInt(1,4);
            int y = random.nextInt(1,4);
            int z = random.nextInt(1,4);
            mines.add(new Location(x,y,z));
        }
    }

    /**
     * Runs the entire game until the player runs out of money or chooses to stop
     */
    public void play() {
        out.println("DO YOU WANT TO SEE INSTRUCTIONS? (YES--1,NO--0)");
        if(readParsedBoolean()) {
            printInstructions();
        }
        do {
            placeMines();
            out.println("WANT TO MAKE A WAGER?");
            int wager = 0 ;

            if(readParsedBoolean()) {
                out.println("HOW MUCH?");
                do {
                    wager = Integer.parseInt(scanner.nextLine());
                    if(wager > money) {
                        out.println("TRIED TO FOOL ME; BET AGAIN");
                    }
                } while(wager > money);
            }

            playerLocation = new Location(1,1,1);
            while(playerLocation.x + playerLocation.y + playerLocation.z != 9) {
                out.println("\nNEXT MOVE");
                String input = scanner.nextLine();

                String[] stringValues = input.split(",");

                if(stringValues.length < 3) {
                    out.println("ILLEGAL MOVE, YOU LOSE.");
                    return;
                }

                int x = Integer.parseInt(stringValues[0]);
                int y = Integer.parseInt(stringValues[1]);
                int z = Integer.parseInt(stringValues[2]);

                Location location = new Location(x,y,z);

                if(x < 1 || x > 3 || y < 1 || y > 3 || z < 1 || z > 3 || !isMoveValid(playerLocation,location)) {
                    out.println("ILLEGAL MOVE, YOU LOSE.");
                    return;
                }

                playerLocation = location;

                if(mines.contains(location)) {
                    out.println("******BANG******");
                    out.println("YOU LOSE!\n\n");
                    money -= wager;
                    break;
                }
            }

            if(wager > 0) {
                out.printf("YOU NOW HAVE %d DOLLARS\n",money);
            }

        } while(money > 0 && doAnotherRound());

        out.println("TOUGH LUCK!");
        out.println("\nGOODBYE.");
    }

    /**
     * Queries the user whether they want to play another round
     * @return True if the player decides to play another round,
     * False if the player would not like to play again
     */
    private boolean doAnotherRound() {
        if(money > 0) {
            out.println("DO YOU WANT TO TRY AGAIN?");
            return readParsedBoolean();
        } else {
            return false;
        }
    }

    /**
     * Prints the instructions to the game, copied from the original code.
     */
    public void printInstructions() {
        out.println("THIS IS A GAME IN WHICH YOU WILL BE PLAYING AGAINST THE");
        out.println("RANDOM DECISION OF THE COMPUTER. THE FIELD OF PLAY IS A");
        out.println("CUBE OF SIDE 3. ANY OF THE 27 LOCATIONS CAN BE DESIGNATED");
        out.println("BY INPUTTING THREE NUMBERS SUCH AS 2,3,1. AT THE START");
        out.println("YOU ARE AUTOMATICALLY AT LOCATION 1,1,1. THE OBJECT OF");
        out.println("THE GAME IS TO GET TO LOCATION 3,3,3. ONE MINOR DETAIL:");
        out.println("THE COMPUTER WILL PICK, AT RANDOM, 5 LOCATIONS AT WHICH");
        out.println("IT WILL PLANT LAND MINES. IF YOU HIT ONE OF THESE LOCATIONS");
        out.println("YOU LOSE. ONE OTHER DETAIL: YOU MAY MOVE ONLY ONE SPACE");
        out.println("IN ONE DIRECTION EACH MOVE. FOR  EXAMPLE: FROM 1,1,2 YOU");
        out.println("MAY MOVE TO 2,1,2 OR 1,1,3. YOU MAY NOT CHANGE");
        out.println("TWO OF THE NUMBERS ON THE SAME MOVE. IF YOU MAKE AN ILLEGAL");
        out.println("MOVE, YOU LOSE AND THE COMPUTER TAKES THE MONEY YOU MAY");
        out.println("\n");
        out.println("ALL YES OR NO QUESTIONS WILL BE ANSWERED BY A 1 FOR YES");
        out.println("OR A 0 (ZERO) FOR NO.");
        out.println();
        out.println("WHEN STATING THE AMOUNT OF A WAGER, PRINT ONLY THE NUMBER");
        out.println("OF DOLLARS (EXAMPLE: 250)  YOU ARE AUTOMATICALLY STARTED WITH");
        out.println("500 DOLLARS IN YOUR ACCOUNT.");
        out.println();
        out.println("GOOD LUCK!");
    }

    /**
     * Waits for the user to input a boolean value. This could either be (true,false), (1,0), (y,n), (yes,no), etc.
     * By default, it will return false
     * @return Parsed boolean value of the user input
     */
    private boolean readParsedBoolean() {
        String in = scanner.nextLine();
        try {
            return in.toLowerCase().charAt(0) == 'y' || Boolean.parseBoolean(in) || Integer.parseInt(in) == 1;
        } catch(NumberFormatException exception) {
            return false;
        }
    }

    /**
     * Checks if a move is valid
     * @param from The point that the player is at
     * @param to The point that the player wishes to move to
     * @return True if the player is only moving, at most, 1 location in any direction, False if the move is invalid
     */
    private boolean isMoveValid(Location from, Location to) {
        return Math.abs(from.x - to.x) + Math.abs(from.y - to.y) + Math.abs(from.z - to.z) <= 1;
    }

    public class Location {
        int x,y,z;

        public Location(int x, int y, int z) {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        /*
        For use in HashSet and checking if two Locations are the same
         */
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            Location location = (Location) o;

            if (x != location.x) return false;
            if (y != location.y) return false;
            return z == location.z;
        }

        /*
        For use in the HashSet to accordingly index the set
         */
        @Override
        public int hashCode() {
            int result = x;
            result = 31 * result + y;
            result = 31 * result + z;
            return result;
        }
    }
}

```

# `30_Cube/javascript/cube.js`

这段代码定义了两个函数，分别是`print`函数和`input`函数。

`print`函数的作用是在网页上打印输出一个字符串，接收一个字符串参数。该函数将接收的字符串通过`document.getElementById`获取到的`output`元素添加到文档中，然后输出到该元素上。

`input`函数的作用是从用户接收一个字符串，并尝试将其解析为整数。该函数首先创建一个`<INPUT>`元素，设置其`type`属性为`text`，`length`属性为`50`，并将`INPUT`元素添加到网页上的一个`<div>`元素中，该元素的ID为`output`。然后，函数将该元素的`addEventListener`方法设置为监听键事件，当该事件接收到`keydown`事件时，函数将捕获到事件，并获取到用户输入的字符串。接下来，函数将从用户输入的字符串中提取一个整数，将其存储在变量`input_str`中，并将其打印输出到页面中，然后将该元素从文档中删除。


```
// CUBE
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

This is a program written in JavaScript that simulates a game of chess. It starts by asking the player to enter their next move, and then performs a check to make sure it is legal. If the move is not legal, it will print an error message and the player will lose the game. If the move is合法， it will print a victory message and the player will win the game.

The game also supports the option to try again and it will reset the game to the initial state.

It is important to mention that this program is not complete and may have bugs or lack functionality.


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
    print(tab(33) + "CUBE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("DO YOU WANT TO SEE THE INSTRUCTIONS? (YES--1,NO--0)");
    b7 = parseInt(await input());
    if (b7 != 0) {
        print("THIS IS A GAME IN WHICH YOU WILL BE PLAYING AGAINST THE\n");
        print("RANDOM DECISION OF THE COMPUTER. THE FIELD OF PLAY IS A\n");
        print("CUBE OF SIDE 3. ANY OF THE 27 LOCATIONS CAN BE DESIGNATED\n");
        print("BY INPUTING THREE NUMBERS SUCH AS 2,3,1. AT THE START,\n");
        print("YOU ARE AUTOMATICALLY AT LOCATION 1,1,1. THE OBJECT OF\n");
        print("THE GAME IS TO GET TO LOCATION 3,3,3. ONE MINOR DETAIL:\n");
        print("THE COMPUTER WILL PICK, AT RANDOM, 5 LOCATIONS AT WHICH\n");
        print("IT WILL PLANT LAND MINES. IF YOU HIT ONE OF THESE LOCATIONS\n");
        print("YOU LOSE. ONE OTHER DETAIL: YOU MAY MOVE ONLY ONE SPACE \n");
        print("IN ONE DIRECTION EACH MOVE. FOR  EXAMPLE: FROM 1,1,2 YOU\n");
        print("MAY MOVE TO 2,1,2 OR 1,1,3. YOU MAY NOT CHANGE\n");
        print("TWO OF THE NUMBERS ON THE SAME MOVE. IF YOU MAKE AN ILLEGAL\n");
        print("MOVE, YOU LOSE AND THE COMPUTER TAKES THE MONEY YOU MAY\n");
        print("HAVE BET ON THAT ROUND.\n");
        print("\n");
        print("\n");
        print("ALL YES OR NO QUESTIONS WILL BE ANSWERED BY A 1 FOR YES\n");
        print("OR A 0 (ZERO) FOR NO.\n");
        print("\n");
        print("WHEN STATING THE AMOUNT OF A WAGER, PRINT ONLY THE NUMBER\n");
        print("OF DOLLARS (EXAMPLE: 250)  YOU ARE AUTOMATICALLY STARTED WITH\n");
        print("500 DOLLARS IN YOUR ACCOUNT.\n");
        print("\n");
        print("GOOD LUCK!\n");
    }
    a1 = 500;
    while (1) {
        a = Math.floor(3 * Math.random());
        if (a == 0)
            a = 3;
        b = Math.floor(3 * Math.random());
        if (b == 0)
            b = 2;
        c = Math.floor(3 * Math.random());
        if (c == 0)
            c = 3;
        d = Math.floor(3 * Math.random());
        if (d == 0)
            d = 1;
        e = Math.floor(3 * Math.random());
        if (e == 0)
            e = 3;
        f = Math.floor(3 * Math.random());
        if (f == 0)
            f = 3;
        g = Math.floor(3 * Math.random());
        if (g == 0)
            g = 3;
        h = Math.floor(3 * Math.random());
        if (h == 0)
            h = 3;
        i = Math.floor(3 * Math.random());
        if (i == 0)
            i = 2;
        j = Math.floor(3 * Math.random());
        if (j == 0)
            j = 3;
        k = Math.floor(3 * Math.random());
        if (k == 0)
            k = 2;
        l = Math.floor(3 * Math.random());
        if (l == 0)
            l = 3;
        m = Math.floor(3 * Math.random());
        if (m == 0)
            m = 3;
        n = Math.floor(3 * Math.random());
        if (n == 0)
            n = 1;
        o = Math.floor(3 * Math.random());
        if (o == 0)
            o = 3;
        print("WANT TO MAKE A WAGER?");
        z = parseInt(await input());
        if (z != 0) {
            print("HOW MUCH ");
            while (1) {
                z1 = parseInt(await input());
                if (a1 < z1) {
                    print("TRIED TO FOOL ME; BET AGAIN");
                } else {
                    break;
                }
            }
        }
        w = 1;
        x = 1;
        y = 1;
        print("\n");
        print("IT'S YOUR MOVE:  ");
        while (1) {
            str = await input();
            p = parseInt(str);
            q = parseInt(str.substr(str.indexOf(",") + 1));
            r = parseInt(str.substr(str.lastIndexOf(",") + 1));
            if (p > w + 1 || q > x + 1 || r > y + 1 || (p == w + 1 && (q >= x + 1 || r >= y + 1)) || (q == x + 1 && r >= y + 1)) {
                print("\n");
                print("ILLEGAL MOVE, YOU LOSE.\n");
                break;
            }
            w = p;
            x = q;
            y = r;
            if (p == 3 && q == 3 && r == 3) {
                won = true;
                break;
            }
            if (p == a && q == b && r == c
             || p == d && q == e && r == f
             || p == g && q == h && r == i
             || p == j && q == k && r == l
             || p == m && q == n && r == o) {
                print("******BANG******");
                print("YOU LOSE!");
                print("\n");
                print("\n");
                won = false;
                break;
            }
            print("NEXT MOVE: ");
        }
        if (won) {
            print("CONGRATULATIONS!\n");
            if (z != 0) {
                z2 = a1 + z1;
                print("YOU NOW HAVE " + z2 + " DOLLARS.\n");
                a1 = z2;
            }
        } else {
            if (z != 0) {
                print("\n");
                z2 = a1 - z1;
                if (z2 <= 0) {
                    print("YOU BUST.\n");
                    break;
                } else {
                    print(" YOU NOW HAVE " + z2 + " DOLLARS.\n");
                    a1 = z2;
                }
            }
        }
        print("DO YOU WANT TO TRY AGAIN ");
        s = parseInt(await input());
        if (s != 1)
            break;
    }
    print("TOUGH LUCK!\n");
    print("\n");
    print("GOODBYE.\n");
}

```

这道题是一个简单的 Python 代码，包含一个名为 "main()" 的函数。函数内部没有包含任何其他代码，因此它只是一个空函数，不会执行任何操作。

在 Python 中，所有程序都必须包含一个名为 "main()" 的函数。当程序运行时，Python 解释器会首先查找 main() 函数，如果找不到，会默认生成一个名为 "print()" 的函数作为默认出口，导致程序无休止地打印 "print()"。

因此，这段代码只是一个简单的程序，包含一个空函数 main()。


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


# `30_Cube/python/cube.py`

这段代码是一个Python脚本，它实现了将CUBE游戏中的坐标从BASIC语义转换为Python浮点数坐标的函数。具体来说，该函数返回一个三元组(x, y, z)，分别代表在CUBE游戏中的位置。

函数的实现包括以下步骤：

1. 使用Python标准库中的`random`模块，从1到3的随机整数生成一个随机整数，用于表示在CUBE游戏中的行。
2. 使用步骤1生成的随机整数，从1到3的随机整数生成一个随机整数，用于表示在CUBE游戏中的列。
3. 使用步骤2和步骤3生成的随机整数，生成一个随机整数，用于表示在CUBE游戏中的位置。

函数的实现使用了`typing.Tuple`类，它可以表示一个元组，由多个元素组成。在这里，该类用于将生成的随机整数组合成一个三元组，以返回CUBE游戏中对应的位置。


```
#!/usr/bin/env python3

"""
CUBE

Converted from BASIC to Python by Trevor Hobson
"""

import random
from typing import Tuple


def mine_position() -> Tuple[int, int, int]:
    return (random.randint(1, 3), random.randint(1, 3), random.randint(1, 3))


```

This is a game of 21, where the player must decide whether to fold or
prompt the dealer's hand. The player is shown two cards and has the option
to either fold, ask the dealer to deal a third card, or try to guess the
dealer's hand. If the player thinks they can guess the dealer's hand,
they can try to match each of the dealer's cards by ordering
descending quantities. If the player does not think they can guess
the dealer's hand, they must fold and lose half their money. The
dealer's hand is dealt, and the player's money is updated. If the
player wins, they print a message and their money is updated. If
the player loses, they print a message and their money is not
updated, but they keep their money.

Here is the code for the game:
```
import random

def try_to_guess:
   wager = int(input("How much do you want to wager? "))
   if not 0 <= wager <= money:
       print("Tried to fool me; bet again")
   else:
       print("Congratulations!")
       money = money + wager

def make_move:
   move = (-1, -1, -1)
   while True:
       try:
           move = parse_move(input("It's your move: "))
       except (ValueError, IndexError):
           print("Please enter valid coordinates.")
       if (
           abs(move[0] - position[0])
           + abs(move[1] - position[1])
           + abs(move[2] - position[2])
       ) > 1:
           print("\nIllegal move. You lose")
           money = money - wager
           break
       elif move in mines:
           print("\n******BANG******")
           print("You lose!")
           money = money - wager
           break
       else:
           position = move
           print("Next move: ")
       return move

def handle_player_input(input_str):
   move = (-1, -1, -1)
   while True:
       try:
           move = parse_move(input_str)
       except (ValueError, IndexError):
           print("Please enter valid coordinates.")
       if (
           abs(move[0] - position[0])
           + abs(move[1] - position[1])
           + abs(move[2] - position[2])
       ) > 1:
           print("\nIllegal move. You lose")
           money = money - wager
           break
       elif move == 3:
           print("\nCongratulations!")
           money = money + wager
           break
       elif move in mines:
           print("\n******BANG******")
           print("You lose!")
           money = money - wager
           break
       else:
           position = move
           print("Next move: ")
       return move

def handle_dealer_input(input_str):
   move = try_to_guess()
   return move

def get_money():
   return money

def set_money(new_money):
   money = new_money
   return money

def game_over():
   print("\n21 game over")
   print("The player wins!")
   return True

def play_game():
   money = 2100
   cards = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11",
              "12", "有利牌", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110", "111", "112", "113", "114", "115", "116", "117", "118", "119", "120", "121", "122", "123", "124", "125", "126", "127", "128", "129", "130", "131", "132", "133", "134", "135", "136", "137", "138", "139", "140", "141", "142", "143", "144", "145", "146", "147", "148", "149", "150", "151", "152", "153", "154", "155", "156", "157", "158", "159", "160", "161", "162", "163", "164", "165", "


```
def parse_move(move: str) -> Tuple[int, int, int]:
    coordinates = [int(item) for item in move.split(",")]
    if len(coordinates) == 3:
        return tuple(coordinates)  # type: ignore
    raise ValueError


def play_game() -> None:
    """Play one round of the game"""

    money = 500
    print("\nYou have", money, "dollars.")
    while True:
        mines = []
        for _ in range(5):
            while True:
                mine = mine_position()
                if not (mine in mines or mine == (1, 1, 1) or mine == (3, 3, 3)):
                    break
            mines.append(mine)
        wager = -1
        while wager == -1:
            try:
                wager = int(input("\nHow much do you want to wager? "))
                if not 0 <= wager <= money:
                    wager = -1
                    print("Tried to fool me; bet again")
            except ValueError:
                print("Please enter a number.")
        prompt = "\nIt's your move: "
        position = (1, 1, 1)
        while True:
            move = (-1, -1, -1)
            while move == (-1, -1, -1):
                try:
                    move = parse_move(input(prompt))
                except (ValueError, IndexError):
                    print("Please enter valid coordinates.")
            if (
                abs(move[0] - position[0])
                + abs(move[1] - position[1])
                + abs(move[2] - position[2])
            ) > 1:
                print("\nIllegal move. You lose")
                money = money - wager
                break
            elif (
                move[0] not in [1, 2, 3]
                or move[1] not in [1, 2, 3]
                or move[2] not in [1, 2, 3]
            ):
                print("\nIllegal move. You lose")
                money = money - wager
                break
            elif move == (3, 3, 3):
                print("\nCongratulations!")
                money = money + wager
                break
            elif move in mines:
                print("\n******BANG******")
                print("You lose!")
                money = money - wager
                break
            else:
                position = move
                prompt = "\nNext move: "
        if money > 0:
            print("\nYou now have", money, "dollars.")
            if not input("Do you want to try again ").lower().startswith("y"):
                break
        else:
            print("\nYou bust.")
    print("\nTough luck")
    print("\nGoodbye.")


```

这段代码是一个函数，名为`print_instructions`，它用于在游戏开始时向玩家介绍游戏的规则。以下是该函数的作用：

1. 打印游戏说明字符串
2. 打印游戏场地大小
3. 提示玩家输入三个数字来选择场地的位置
4. 在玩家选择好场地后，打印游戏开始时玩家的位置
5. 提示玩家计算机将在随机位置放置五颗地雷
6. 提示玩家只能在一个方向移动
7. 提示玩家不得更改选择场地的方式
8. 如果玩家移动违法，游戏失败，玩家失去所选场地
9. 如果玩家成功到达目的地，游戏胜利
10. 在游戏开始时，打印可以选择赢得的赌注金额
11. 在游戏开始时，打印可用的游戏资金


```
def print_instructions() -> None:
    print("\nThis is a game in which you will be playing against the")
    print("random decisions of the computer. The field of play is a")
    print("cube of side 3. Any of the 27 locations can be designated")
    print("by inputing three numbers such as 2,3,1. At the start,")
    print("you are automatically at location 1,1,1. The object of")
    print("the game is to get to location 3,3,3. One minor detail:")
    print("the computer will pick, at random, 5 locations at which")
    print("it will plant land mines. If you hit one of these locations")
    print("you lose. One other detail: You may move only one space")
    print("in one direction each move. For example: From 1,1,2 you")
    print("may move to 2,1,2 or 1,1,3. You may not change")
    print("two of the numbers on the same move. If you make an illegal")
    print("move, you lose and the computer takes the money you may")
    print("have bet on that round.\n")
    print("When stating the amount of a wager, print only the number")
    print("of dollars (example: 250) you are automatically started with")
    print("500 dollars in your account.\n")
    print("Good luck!")


```

这段代码是一个Python程序，它的作用是运行游戏。程序中包含两个函数，一个是`print_instructions()`，另一个是`play_game()`。这两个函数都会在程序中被调用。

`print_instructions()`函数的作用是打印游戏说明。它通过计算34个空格和15个星号，然后加上"CUBE"和"CREATIVE COMPUTING"。结果是在屏幕上打印出"CUBE"和"CREATIVE COMPUTING"。

`play_game()`函数的作用是让用户继续玩游戏。它先打印出"Do you want to see the instructions?"，然后等待用户输入。如果用户输入的是"y"，那么它就会打印出游戏中的说明。否则，它就会让用户继续玩游戏，并再次提示用户输入。这样一直循环下去，直到用户不想玩了，程序才会结束。


```
def main() -> None:
    print(" " * 34 + "CUBE")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")
    if input("Do you want to see the instructions ").lower().startswith("y"):
        print_instructions()

    keep_playing = True
    while keep_playing:
        play_game()
        keep_playing = input("\nPlay again? (yes or no) ").lower().startswith("y")


if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Depth Charge

In this program you are captain of the destroyer USS Computer. An enemy submarine has been causing trouble and your mission is to destroy it. You may select the seize of the “cube” of water you wish to search in. The computer then determines how many depth charges you get to destroy the submarine.

Each depth charge is exploded by you specifying a trio of numbers; the first two are the surface coordinates (X,Y), the third is the depth. After each depth charge, your sonar observer will tell you where the explosion was relative to the submarine.

Dana Noftle wrote this program while a student at Acton High School, Acton, Massachusetts.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=55)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=70)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `31_Depth_Charge/csharp/Controller.cs`

This is a C# class that appears to define a game interface for a text-based adventure game.

The class defines several methods, including `View.PromptDimension()` which prompts the user for the desired dimension number.

The class also defines methods for receiving input from the user, such as `View.ShowInvalidNumber()` and `View.ShowInvalidDimension()` which return an error message if the user enters an invalid number or dimension.

The class has methods for getting user input in the form of `View.PromptGuess()` and `View.PromptPlayAgain()`.

The last class also defines a `InputCoordinates()` class which retrieves a set of user input coordinates from the user.

It appears that the game interface is for a game where the user is prompted to enter a number and an optional depth, and then provided with a set of coordinates to continue the game.


```
﻿using System;

namespace DepthCharge
{
    /// <summary>
    /// Contains functions for reading input from the user.
    /// </summary>
    static class Controller
    {
        /// <summary>
        /// Retrives a dimension for the play area from the user.
        /// </summary>
        /// <remarks>
        /// Note that the original BASIC version would allow dimension values
        /// of 0 or less.  We're doing a little extra validation here in order
        /// to avoid strange behaviour.
        /// </remarks>
        public static int InputDimension()
        {
            View.PromptDimension();

            while (true)
            {
                if (!Int32.TryParse(Console.ReadLine(), out var dimension))
                    View.ShowInvalidNumber();
                else
                if (dimension < 1)
                    View.ShowInvalidDimension();
                else
                    return dimension;
            }
        }

        /// <summary>
        /// Retrieves a set of coordinates from the user.
        /// </summary>
        /// <param name="trailNumber">
        /// The current trail number.
        /// </param>
        public static (int x, int y, int depth) InputCoordinates(int trailNumber)
        {
            View.PromptGuess(trailNumber);

            while (true)
            {
                var coordinates = Console.ReadLine().Split(',');

                if (coordinates.Length < 3)
                    View.ShowTooFewCoordinates();
                else
                if (coordinates.Length > 3)
                    View.ShowTooManyCoordinates();
                else
                if (!Int32.TryParse(coordinates[0], out var x) ||
                    !Int32.TryParse(coordinates[1], out var y) ||
                    !Int32.TryParse(coordinates[2], out var depth))
                    View.ShowInvalidNumber();
                else
                    return (x, y, depth);
            }
        }

        /// <summary>
        /// Retrieves the user's intention to play again (or not).
        /// </summary>
        public static bool InputPlayAgain()
        {
            View.PromptPlayAgain();

            while (true)
            {
                switch (Console.ReadLine())
                {
                    case "Y":
                        return true;
                    case "N":
                        return false;
                    default:
                        View.ShowInvalidYesOrNo();
                        break;
                }
            }
        }
    }
}

```

# `31_Depth_Charge/csharp/Program.cs`

这段代码是一个基于控制台的应用程序，名为“DepthCharge”。它的主要目的是让用户猜测一个2D游戏中的潜艇位置，每次猜测需要输入三个坐标值，然后显示猜测结果、游戏结果以及最后的告别。

具体来说，这段代码执行以下操作：

1. 创建一个名为“Random”的随机类，用于生成随机数。
2. 创建一个名为“Program”的类，作为程序的入口点。
3. 创建一个名为“Main”的静态方法，作为程序的执行入口点。
4. 在“Main”方法中，创建一个名为“random”的随机对象，并将其赋值为一个新的随机数生成器。
5. 创建一个名为“View”的类，用于显示游戏的相关信息，包括游戏标题、游戏提示、提示消息以及游戏结果。
6. 创建一个名为“Submarine”的类，用于处理游戏规则，包括生成潜艇位置的功能。
7. 在“Submarine”类中，使用随机数生成器生成新的坐标值，并将其作为潜艇位置。
8. 在“Main”方法中，创建一个名为“Controller”的类，用于处理用户输入，包括获取游戏输入并将其传递给游戏中的精灵。
9. 在“Controller”类中，创建一个名为“View”的静态方法，用于显示猜测结果、游戏结果以及最后的告别。
10. 在“Main”方法中，创建一个名为“Program”的类，将其余所有类都静态化，并调用“View.ShowBanner”方法来显示游戏标题。
11. 调用“Controller.InputDimension”方法来获取游戏中的维度，并将其存储在变量中。
12. 调用“CalculateMaximumGuesses”方法来计算出猜测的最大值，并将其存储在变量中。
13. 在“Main”方法中，创建一个名为“do...while”循环，用于让用户一直猜测，直到他们选择“退出”。
14. 在“do...while”循环中，调用“Controller.InputPlayAgain”方法来检查用户是否还想要继续游戏。
15. 如果用户选择“退出”，则显示“再见”消息。
16. 如果用户在游戏过程中点击“游戏结果”按钮，则显示游戏结果，包括猜测的位置、猜测次数以及游戏结果。
17. 如果用户在游戏过程中点击“猜潜艇”按钮，则调用“Controller.InputCoordinates”方法来获取用户输入的三个坐标值，并将其存储在变量中。
18. 如果猜测位置与当前潜艇位置不同，则显示“猜测位置”消息，并将其存储在变量中。


```
﻿using System;

namespace DepthCharge
{
    class Program
    {
        static void Main(string[] args)
        {
            var random = new Random();

            View.ShowBanner();

            var dimension = Controller.InputDimension();
            var maximumGuesses = CalculateMaximumGuesses();

            View.ShowInstructions(maximumGuesses);

            do
            {
                View.ShowStartGame();

                var submarineCoordinates = PlaceSubmarine();
                var trailNumber = 1;
                var guess = (0, 0, 0);

                do
                {
                    guess = Controller.InputCoordinates(trailNumber);
                    if (guess != submarineCoordinates)
                        View.ShowGuessPlacement(submarineCoordinates, guess);
                }
                while (guess != submarineCoordinates && trailNumber++ < maximumGuesses);

                View.ShowGameResult(submarineCoordinates, guess, trailNumber);
            }
            while (Controller.InputPlayAgain());

            View.ShowFarewell();

            int CalculateMaximumGuesses() =>
                (int)Math.Log2(dimension) + 1;

            (int x, int y, int depth) PlaceSubmarine() =>
                (random.Next(dimension), random.Next(dimension), random.Next(dimension));
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `31_Depth_Charge/csharp/View.cs`

This is a class that defines a `CdrSubmarine` class with several methods for displaying information about the submarine's location and the user's guess.

The `CdrSubmarine` class has several methods of the same name, `ShowFinalGuess`, `ShowInt depth), `ShowTrialNumber`, `ShowInvalidNumber`, `ShowInvalidDimension`, `ShowTooFewCoordinates`, `ShowTooManyCoordinates`, and `ShowInvalidYesOrNo`. These methods all take in different inputs and display the appropriate message or information.

The last class also has a `ShowFarewell` method and a `ShowInvalidNumber` method.

It is important to note that this class is not meant to be used in any real-world application and it is not implemented with any real-world functionality.


```
﻿using System;

namespace DepthCharge
{
    /// <summary>
    /// Contains methods for displaying information to the user.
    /// </summary>
    static class View
    {
        public static void ShowBanner()
        {
            Console.WriteLine("                             DEPTH CHARGE");
            Console.WriteLine("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
        }

        public static void ShowInstructions(int maximumGuesses)
        {
            Console.WriteLine("YOU ARE THE CAPTAIN OF THE DESTROYER USS COMPUTER");
            Console.WriteLine("AN ENEMY SUB HAS BEEN CAUSING YOU TROUBLE.  YOUR");
            Console.WriteLine($"MISSION IS TO DESTROY IT.  YOU HAVE {maximumGuesses} SHOTS.");
            Console.WriteLine("SPECIFY DEPTH CHARGE EXPLOSION POINT WITH A");
            Console.WriteLine("TRIO OF NUMBERS -- THE FIRST TWO ARE THE");
            Console.WriteLine("SURFACE COORDINATES; THE THIRD IS THE DEPTH.");
            Console.WriteLine();
        }

        public static void ShowStartGame()
        {
            Console.WriteLine("GOOD LUCK !");
            Console.WriteLine();
        }

        public static void ShowGuessPlacement((int x, int y, int depth) actual, (int x, int y, int depth) guess)
        {
            Console.Write("SONAR REPORTS SHOT WAS ");
            if (guess.y > actual.y)
                Console.Write("NORTH");
            if (guess.y < actual.y)
                Console.Write("SOUTH");
            if (guess.x > actual.x)
                Console.Write("EAST");
            if (guess.x < actual.x)
                Console.Write("WEST");
            if (guess.y != actual.y || guess.x != actual.y)
                Console.Write(" AND");
            if (guess.depth > actual.depth)
                Console.Write (" TOO LOW.");
            if (guess.depth < actual.depth)
                Console.Write(" TOO HIGH.");
            if (guess.depth == actual.depth)
                Console.Write(" DEPTH OK.");

            Console.WriteLine();
        }

        public static void ShowGameResult((int x, int y, int depth) submarineLocation, (int x, int y, int depth) finalGuess, int trailNumber)
        {
            Console.WriteLine();

            if (submarineLocation == finalGuess)
            {
                Console.WriteLine($"B O O M ! ! YOU FOUND IT IN {trailNumber} TRIES!");
            }
            else
            {
                Console.WriteLine("YOU HAVE BEEN TORPEDOED!  ABANDON SHIP!");
                Console.WriteLine($"THE SUBMARINE WAS AT {submarineLocation.x}, {submarineLocation.y}, {submarineLocation.depth}");
            }
        }

        public static void ShowFarewell()
        {
            Console.WriteLine ("OK.  HOPE YOU ENJOYED YOURSELF.");
        }

        public static void ShowInvalidNumber()
        {
            Console.WriteLine("PLEASE ENTER A NUMBER");
        }

        public static void ShowInvalidDimension()
        {
            Console.WriteLine("PLEASE ENTER A VALID DIMENSION");
        }

        public static void ShowTooFewCoordinates()
        {
            Console.WriteLine("TOO FEW COORDINATES");
        }

        public static void ShowTooManyCoordinates()
        {
            Console.WriteLine("TOO MANY COORDINATES");
        }

        public static void ShowInvalidYesOrNo()
        {
            Console.WriteLine("PLEASE ENTER Y OR N");
        }

        public static void PromptDimension()
        {
            Console.Write("DIMENSION OF SEARCH AREA? ");
        }

        public static void PromptGuess(int trailNumber)
        {
            Console.WriteLine();
            Console.Write($"TRIAL #{trailNumber}? ");
        }

        public static void PromptPlayAgain()
        {
            Console.WriteLine();
            Console.Write("ANOTHER GAME (Y OR N)? ");
        }
    }
}

```

# `31_Depth_Charge/java/DepthCharge.java`

这段代码是一个名为 "Depth Charge" 的游戏，基于 1970 年代的游戏 Depth Charge。这个游戏是一个简单的文本冒险游戏，玩家需要通过输入命令来控制游戏中的角色，通过探索世界并与其他角色互动来完成游戏。

该代码的作用是实现一个基于 Depth Charge 的文本冒险游戏，没有新增的功能和错误检查。该游戏已经被转换为 Java 语言，以便于人们更好地理解和维护这个游戏。


```
import java.util.Scanner;
import java.lang.Math;

/**
 * Game of Depth Charge
 * <p>
 * Based on the BASIC game of Depth Charge here
 * https://github.com/coding-horror/basic-computer-games/blob/main/31%20Depth%20Charge/depthcharge.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 *
 * Converted from BASIC to Java by Darren Cardenas.
 */

```

Based on the provided code, it appears that this is a game where the player is潜水员，任务是到达一个目标点并返回。玩家需要通过 sonar 声纳来避开水雷并到达目标点。如果玩家的潜水艇遇到水雷，则会炸毁并弹回，如果玩家的潜水艇成功到达目标点，则会显示 "YOUR win!"。

需要注意的是，这个游戏似乎存在一些逻辑错误，例如在处理不同层的目标点时，有时候会只显示一个目标点，而忽略了其他目标点。此外，这个游戏的地图也没有显示出来，玩家需要通过 sonar 声纳来感知周围的环境。



```
public class DepthCharge {

  private final Scanner scan;  // For user input

  public DepthCharge() {

    scan = new Scanner(System.in);

  }  // End of constructor DepthCharge

  public void play() {

    showIntro();
    startGame();

  }  // End of method play

  private static void showIntro() {

    System.out.println(" ".repeat(29) + "DEPTH CHARGE");
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    System.out.println("\n\n");

  }  // End of method showIntro

  private void startGame() {

    int searchArea = 0;
    int shotNum = 0;
    int shotTotal = 0;
    int shotX = 0;
    int shotY = 0;
    int shotZ = 0;
    int targetX = 0;
    int targetY = 0;
    int targetZ = 0;
    int tries = 0;
    String[] userCoordinates;
    String userResponse = "";

    System.out.print("DIMENSION OF SEARCH AREA? ");
    searchArea = Integer.parseInt(scan.nextLine());
    System.out.println("");

    shotTotal = (int) (Math.log10(searchArea) / Math.log10(2)) + 1;

    System.out.println("YOU ARE THE CAPTAIN OF THE DESTROYER USS COMPUTER");
    System.out.println("AN ENEMY SUB HAS BEEN CAUSING YOU TROUBLE.  YOUR");
    System.out.println("MISSION IS TO DESTROY IT.  YOU HAVE " + shotTotal + " SHOTS.");
    System.out.println("SPECIFY DEPTH CHARGE EXPLOSION POINT WITH A");
    System.out.println("TRIO OF NUMBERS -- THE FIRST TWO ARE THE");
    System.out.println("SURFACE COORDINATES; THE THIRD IS THE DEPTH.");

    // Begin outer while loop
    while (true) {

      System.out.println("");
      System.out.println("GOOD LUCK !");
      System.out.println("");

      targetX = (int) ((searchArea + 1) * Math.random());
      targetY = (int) ((searchArea + 1) * Math.random());
      targetZ = (int) ((searchArea + 1) * Math.random());

      // Begin loop through all shots
      for (shotNum = 1; shotNum <= shotTotal; shotNum++) {

        // Get user input
        System.out.println("");
        System.out.print("TRIAL # " + shotNum + "? ");
        userResponse = scan.nextLine();

        // Split on commas
        userCoordinates = userResponse.split(",");

        // Assign to integer variables
        shotX = Integer.parseInt(userCoordinates[0].trim());
        shotY = Integer.parseInt(userCoordinates[1].trim());
        shotZ = Integer.parseInt(userCoordinates[2].trim());

        // Win condition
        if (Math.abs(shotX - targetX) + Math.abs(shotY - targetY)
            + Math.abs(shotZ - targetZ) == 0) {

          System.out.println("B O O M ! ! YOU FOUND IT IN" + shotNum + " TRIES!");
          break;

        }

        this.getReport(targetX, targetY, targetZ, shotX, shotY, shotZ);

        System.out.println("");

      }  // End loop through all shots

      if (shotNum > shotTotal) {

        System.out.println("");
        System.out.println("YOU HAVE BEEN TORPEDOED!  ABANDON SHIP!");
        System.out.println("THE SUBMARINE WAS AT " + targetX + "," + targetY + "," + targetZ);
      }

      System.out.println("");
      System.out.println("");
      System.out.print("ANOTHER GAME (Y OR N)? ");
      userResponse = scan.nextLine();

      if (!userResponse.toUpperCase().equals("Y")) {
        System.out.print("OK.  HOPE YOU ENJOYED YOURSELF.");
        return;
      }

    }  // End outer while loop

  }  // End of method startGame

  public void getReport(int a, int b, int c, int x, int y, int z) {

    System.out.print("SONAR REPORTS SHOT WAS ");

    // Handle y coordinate
    if (y > b) {

      System.out.print("NORTH");

    } else if (y < b) {

      System.out.print("SOUTH");
    }

    // Handle x coordinate
    if (x > a) {

      System.out.print("EAST");

    } else if (x < a) {

      System.out.print("WEST");
    }

    if ((y != b) || (x != a)) {

      System.out.print(" AND");
    }

    // Handle depth
    if (z > c) {

      System.out.println(" TOO LOW.");

    } else  if (z < c) {

      System.out.println(" TOO HIGH.");

    } else {

      System.out.println(" DEPTH OK.");
    }

    return;

  }  // End of method getReport

  public static void main(String[] args) {

    DepthCharge game = new DepthCharge();
    game.play();

  }  // End of method main

}  // End of class DepthCharge

```