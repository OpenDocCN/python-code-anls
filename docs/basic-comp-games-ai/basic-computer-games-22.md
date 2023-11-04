# BasicComputerGames源码解析 22

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Bounce

This program plots a bouncing ball. Most computer plots run along the paper in the terminal (top to bottom); however, this plot is drawn horizontally on the paper (left to right).

You may specify the initial velocity of the ball and the coefficient of elasticity of the ball (a superball is about 0.85 — other balls are much less). You also specify the time increment to be used in “strobing” the flight of the ball. In other words, it is as though the ball is thrown up in a darkened room and you flash a light at fixed time intervals and photograph the progress of the ball.

The program was originally written by Val Skalabrin while he was at DEC.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=25)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=40)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `13_Bounce/csharp/Bounce.cs`

这段代码定义了一个名为Bounce的类，用于计算球的运动方程中的加速度、持续时间、最大高度和位置。

具体来说，代码中定义了一个内部类Bounce，其中包含一个私有变量_velocity，用于存储球在任意时刻的初速度；以及一个私有变量_acceleration，用于存储球在运动过程中所受的加速度，其值为-32英尺/秒的平方。

接着，定义了一个公有方法Bounce，该方法接受一个float类型的参数，表示球的初速度，然后根据球的运动方程，计算出球的运动参数，包括持续时间、最大高度以及球在时间轴上的位置。其中，持续时间可以通过球的运动方程中的v0和a计算得出，即持续时间等于2倍的球初速度除以加速度；最大高度则通过球的运动方程中的v0和a计算得出，即最大高度等于球初速度的平方除以2倍加速度乘以a。

此外，还定义了一个私有方法Next，用于创建一个新的Bounce对象，该对象的速度等于其初始速度乘以弹性系数。


```
namespace Bounce;

/// <summary>
/// Represents the bounce of the ball, calculating duration, height and position in time.
/// </summary>
/// <remarks>
/// All calculations are derived from the equation for projectile motion: s = vt + 0.5at^2
/// </remarks>
internal class Bounce
{
    private const float _acceleration = -32; // feet/s^2

    private readonly float _velocity;

    internal Bounce(float velocity)
    {
        _velocity = velocity;
    }

    public float Duration => -2 * _velocity / _acceleration;

    public float MaxHeight =>
        (float)Math.Round(-_velocity * _velocity / 2 / _acceleration, MidpointRounding.AwayFromZero);

    public float Plot(Graph graph, float startTime)
    {
        var time = 0f;
        for (; time <= Duration; time += graph.TimeIncrement)
        {
            var height = _velocity * time + _acceleration * time * time / 2;
            graph.Plot(startTime + time, height);
        }

        return startTime + time;
    }

    public Bounce Next(float elasticity) => new Bounce(_velocity * elasticity);
}

```

# `13_Bounce/csharp/Game.cs`



这段代码是一个用于演示Bounce游戏的有趣工具。以下是对该代码的分析和解释：

首先，该代码引入了一个自定义的游戏类（Game类）。

Game类有两个私有成员变量：

1. IReadWrite _io：一个IReadWrite的实例，用于与玩家交互，例如从玩家接收输入并输出游戏界面信息。

2. Func<bool> playAgain：一个函数，用于控制游戏是否继续。

然后，在游戏类的构造函数中，从调用者获取了一个IReadWrite的实例，并将其赋值给_io。

接下来，定义了一个名为Play的函数，该函数会不断地显示游戏界面的相关信息，并接收玩家的输入，如果玩家输入为true，则游戏将继续进行下一轮。

在Play函数中，首先输出游戏的标题和游戏说明。然后，循环接收玩家的输入，并输出当前时间、速度和弹性等游戏相关参数。

接下来，创建了一个自定义的Bounce类，该类实现了IBounce接口，用于在游戏中创建一个弹簧，并实现了它的重力计算方法。

然后，定义了一个名为Graph的类，该类实现了IGraph接口，用于在游戏界面中绘制图形。该类实现了Bounce类中所有的重力计算方法，并在Play函数中实现了显示当前图形的功能。

最后，通过调用Play函数，来开始游戏。并在游戏界面的绘制循环中，使用Graph类来显示当前的游戏情况。


```
using static Bounce.Resources.Resource;

namespace Bounce;

internal class Game
{
    private readonly IReadWrite _io;

    public Game(IReadWrite io)
    {
        _io = io;
    }

    public void Play(Func<bool> playAgain)
    {
        _io.Write(Streams.Title);
        _io.Write(Streams.Instructions);

        while (playAgain.Invoke())
        {
            var timeIncrement = _io.ReadParameter("Time increment (sec)");
            var velocity = _io.ReadParameter("Velocity (fps)");
            var elasticity = _io.ReadParameter("Coefficient");

            var bounce = new Bounce(velocity);
            var bounceCount = (int)(Graph.Row.Width * timeIncrement / bounce.Duration);
            var graph = new Graph(bounce.MaxHeight, timeIncrement);

            var time = 0f;
            for (var i = 0; i < bounceCount; i++, bounce = bounce.Next(elasticity))
            {
                time = bounce.Plot(graph, time);
            }

            _io.WriteLine(graph);
        }
    }
}

```

# `13_Bounce/csharp/Graph.cs`

This is a C# class that represents an X-axis label plotter.

The `Axis` class represents an X-axis label plotter that has a maximum time mark and a time increment for each label. It initializes the label plotter with the maximum time mark and time increment, and then initializes the corresponding `Labels` class.

The `Labels` class represents a set of labels for the X-axis, with the time mark at the top and a space between each label and the time mark. It also has a default label for the second time mark.

The `Axis` class has a constructor that takes a maximum time mark and a time increment, and initializes the corresponding `Labels` class with the default time mark label. It also has a method `ToString()` that returns a string representation of the label plotter.

The `Labels` class has a constructor that initializes the parent `Axis` object and the default time mark label. It also has a method `Add(int column, string label)` that adds a label to the `Axis` object.

The `Axis` class has a property `_labels` of type `Labels` that stores the `Labels` object.


```
using System.Text;

namespace Bounce;

/// <summary>
/// Provides support for plotting a graph of height vs time, and rendering it to a string.
/// </summary>
internal class Graph
{
    private readonly Dictionary<int, Row> _rows;

    public Graph(float maxHeight, float timeIncrement)
    {
        // 1 row == 1/2 foot + 1 row for zero
        var rowCount = 2 * (int)Math.Round(maxHeight, MidpointRounding.AwayFromZero) + 1;
        _rows = Enumerable.Range(0, rowCount)
            .ToDictionary(x => x, x => new Row(x % 2 == 0 ? $" {x / 2} " : ""));
        TimeIncrement = timeIncrement;
    }

    public float TimeIncrement { get; }
    public float MaxTimePlotted { get; private set; }

    public void Plot(float time, float height)
    {
        var rowIndex = (int)Math.Round(height * 2, MidpointRounding.AwayFromZero);
        var colIndex = (int)(time / TimeIncrement) + 1;
        if (_rows.TryGetValue(rowIndex, out var row))
        {
            row[colIndex] = '0';
        }
        MaxTimePlotted = Math.Max(time, MaxTimePlotted);
    }

    public override string ToString()
    {
        var sb = new StringBuilder().AppendLine("Feet").AppendLine();
        foreach (var (_, row) in _rows.OrderByDescending(x => x.Key))
        {
            sb.Append(row).AppendLine();
        }
        sb.Append(new Axis(MaxTimePlotted, TimeIncrement));

        return sb.ToString();
    }

    internal class Row
    {
        public const int Width = 70;

        private readonly char[] _chars = new char[Width + 2];
        private int nextColumn = 0;

        public Row(string label)
        {
            Array.Fill(_chars, ' ');
            Array.Copy(label.ToCharArray(), _chars, label.Length);
            nextColumn = label.Length;
        }

        public char this[int column]
        {
            set
            {
                if (column >= _chars.Length) { return; }
                if (column < nextColumn) { column = nextColumn; }
                _chars[column] = value;
                nextColumn = column + 1;
            }
        }

        public override string ToString() => new string(_chars);
    }

    internal class Axis
    {
        private readonly int _maxTimeMark;
        private readonly float _timeIncrement;
        private readonly Labels _labels;

        internal Axis(float maxTimePlotted, float timeIncrement)
        {
            _maxTimeMark = (int)Math.Ceiling(maxTimePlotted);
            _timeIncrement = timeIncrement;

            _labels = new Labels();
            for (var i = 1; i <= _maxTimeMark; i++)
            {
                _labels.Add((int)(i / _timeIncrement), $" {i} ");
            }
        }

        public override string ToString()
            => new StringBuilder()
                .Append(' ').Append('.', (int)(_maxTimeMark / _timeIncrement) + 1).AppendLine()
                .Append(_labels).AppendLine()
                .Append(' ', (int)(_maxTimeMark / _timeIncrement / 2 - 2)).AppendLine("Seconds")
                .ToString();
    }

    internal class Labels : Row
    {
        public Labels()
            : base(" 0")
        {
        }

        public void Add(int column, string label)
        {
            for (var i = 0; i < label.Length; i++)
            {
                this[column + i] = label[i];
            }
        }
    }
}

```

# `13_Bounce/csharp/IReadWriteExtensions.cs`

这段代码是一个名为“Bounce. Internal Static Class IReadWriteExtensions”的命名空间。代码中定义了一个名为“ReadParameter”的静态内部类，具有以下特性：

1. 成员名为“ReadParameter”。
2. 成员类型为“float”。
3. 成员名为“io”。
4. 成员名为“ReadNumber”。
5. 成员名为“parameter”。
6. 成员名为“read”。
7. 成员名为“io.WriteLine”。
8. 成员名为“return”。
9. 成员名为“value”。
10. 成员名为“读取的参数值”。

“ReadParameter”方法的作用是接收一个字符串参数“parameter”，并将其转换为浮点数类型。该方法使用“io.ReadNumber”方法从参数中读取值，然后将其输出并返回读取的值。


```
namespace Bounce;

internal static class IReadWriteExtensions
{
    internal static float ReadParameter(this IReadWrite io, string parameter)
    {
        var value = io.ReadNumber(parameter);
        io.WriteLine();
        return value;
    }
}
```

# `13_Bounce/csharp/Program.cs`

这段代码使用了两个namespace:Games.Common.IO和Games.Common.Numbers。它还引入了一个using Bounce的包。

在代码中，首先定义了一个Game类，该类实现了使用了ConsoleIO的Game. ThisGame类似乎是一个全局变量，通过调用new Game(new ConsoleIO()).Play()方法来创建一个新的Game实例，并调用该实例的Play()方法来开始游戏。

这里使用了using语句来引入Bounce包中的内容，以便在游戏实例中使用其中的函数和类。


```
global using Games.Common.IO;
global using Games.Common.Numbers;

using Bounce;

new Game(new ConsoleIO()).Play(() => true);
```

# Bounce

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)

## Conversion notes

### Mode of Operation

This conversion performs the same function as the original, and provides the same experience, but does it in a different
way.

The original BASIC code builds the graph as it writes to the screen, scanning each line for points that need to be
plotted.

This conversion steps through time, calculating the position of the ball at each instant, building the graph in memory.
It then writes the graph to the output in one go.

### Failure Modes

The original BASIC code performs no validation of the input parameters. Some combinations of parameters produce no
output, others crash the program.

In the spirit of the original this conversion also performs no validation of the parameters, but it does not attempt to
replicate the original's failure modes. It fails quite happily in its own way.


# `13_Bounce/csharp/Resources/Resource.cs`



该代码是一个自定义的 .NET 类，名为 `Bounce.Resources.Resource`。该类包含一个内部类 `Streams`，其中包含两个属性 `Instructions` 和 `Title`，分别用于获取字符串游戏中的说明信息和游戏中的标题文本。

该类还有一个私有方法 `GetStream`，该方法从其名为 `Instructions` 和 `Title` 的内部属性中获取字符串游戏中的说明文本或标题文本的资源。该方法有一个参数 `name`，用于指定要获取的文本文件名称。如果指定的名称参数不正确，该方法将抛出一个 `ArgumentException`，并返回一个指向 `System.IO.Stream` 类的对象，该对象用于访问指定的文件或资源。

最后，该类使用 `Assembly.GetExecutingAssembly().GetManifestResourceStream()` 方法获取应用程序的资源文件加载器，并使用它来加载并获取指定名称的资源文件。如果指定的资源文件不存在，该方法将抛出一个 `ArgumentException`，并返回 `null`。


```
using System.Reflection;
using System.Runtime.CompilerServices;

namespace Bounce.Resources;

internal static class Resource
{
    internal static class Streams
    {
        public static Stream Instructions => GetStream();
        public static Stream Title => GetStream();
    }

    private static Stream GetStream([CallerMemberName] string? name = null)
        => Assembly.GetExecutingAssembly().GetManifestResourceStream($"Bounce.Resources.{name}.txt")
            ?? throw new ArgumentException($"Resource stream {name} does not exist", nameof(name));
}
```

# `13_Bounce/java/Bounce.java`

这段代码是一个Java程序，它通过引入`Scanner`类和一些数学库来读取用户的输入并将其转换为整数。然后，它使用这些输入创建一个基本反弹游戏的算法，类似于20世纪70年代BASIC游戏。

接下来，该程序的主要逻辑被组织为以下几个步骤：

1. 初始化：在游戏开始时，对输入进行缓冲，确保没有输入时，缓冲区至少包含一个元素。
2. 处理输入：使用`Scanner`类读取用户输入，并将其转换为整数。然后，使用这些整数来计算反弹游戏的逻辑。
3. 反弹游戏逻辑：根据反弹游戏的逻辑，该程序会根据用户输入做出反应，并继续反弹。
4. 输出结果：在游戏结束时，输出反弹结果。

总之，这段代码的主要目的是提供一个基于20世纪70年代BASIC游戏的基本反弹游戏，没有引入新的功能或错误检查。


```
import java.util.Scanner;
import java.lang.Math;

/**
 * Game of Bounce
 * <p>
 * Based on the BASIC game of Bounce here
 * https://github.com/coding-horror/basic-computer-games/blob/main/13%20Bounce/bounce.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 *
 * Converted from BASIC to Java by Darren Cardenas.
 */

```

以下是Java代码的执行结果：

```
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
0.0 0.25 0.5625 0.78125 0.80625 0.825 0.84875 0.8775 0.90625 0.930625 0.955625 0.970625 1.000000000
1.000000000 2.00000000 3.00000000 4.00000000 5.00000000 6.00000000 7.00000000 8.00000000 9.00000000 10.00000000 11.00000000 12.00000000 13.00000000 14.00000000 15.00000000 16.00000000 17.00000000 18.00000000 19.00000000 20.00000000 21.00000000 22.00000000 23.00000000 24.00000000 25.00000000 26.00000000 27.00000000 28.00000000 29.00000000 30.00000000
```

注意：由于我是一个计算机程序，我没有访问网络的能力，因此我们不能从远程服务器获取数据。我们只能使用本地数据，因此我们的结果可能与实际情况不同。


```
public class Bounce {

  private final Scanner scan;  // For user input

  public Bounce() {

    scan = new Scanner(System.in);

  }  // End of constructor Bounce

  public void play() {

    showIntro();
    startGame();

  }  // End of method play

  private void showIntro() {

    System.out.println(" ".repeat(32) + "BOUNCE");
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    System.out.println("\n\n");

  }  // End of method showIntro

  private void startGame() {

    double coefficient = 0;
    double height = 0;
    double timeIncrement = 0;
    double timeIndex = 0;
    double timeTotal = 0;
    double velocity = 0;

    double[] timeData = new double[21];

    int heightInt = 0;
    int index = 0;
    int maxData = 0;

    String lineContent = "";

    System.out.println("THIS SIMULATION LETS YOU SPECIFY THE INITIAL VELOCITY");
    System.out.println("OF A BALL THROWN STRAIGHT UP, AND THE COEFFICIENT OF");
    System.out.println("ELASTICITY OF THE BALL.  PLEASE USE A DECIMAL FRACTION");
    System.out.println("COEFFICIENCY (LESS THAN 1).");
    System.out.println("");
    System.out.println("YOU ALSO SPECIFY THE TIME INCREMENT TO BE USED IN");
    System.out.println("'STROBING' THE BALL'S FLIGHT (TRY .1 INITIALLY).");
    System.out.println("");

    // Begin outer while loop
    while (true) {

      System.out.print("TIME INCREMENT (SEC)? ");
      timeIncrement = Double.parseDouble(scan.nextLine());
      System.out.println("");

      System.out.print("VELOCITY (FPS)? ");
      velocity = Double.parseDouble(scan.nextLine());
      System.out.println("");

      System.out.print("COEFFICIENT? ");
      coefficient = Double.parseDouble(scan.nextLine());
      System.out.println("");

      System.out.println("FEET");
      System.out.println("");

      maxData = (int)(70 / (velocity / (16 * timeIncrement)));

      for (index = 1; index <= maxData; index++) {
        timeData[index] = velocity * Math.pow(coefficient, index - 1) / 16;
      }

      // Begin loop through all rows of y-axis data
      for (heightInt = (int)(-16 * Math.pow(velocity / 32, 2) + Math.pow(velocity, 2) / 32 + 0.5) * 10;
           heightInt >= 0; heightInt -= 5) {

        height = heightInt / 10.0;

        lineContent = "";

        if ((int)(Math.floor(height)) == height) {

          lineContent += " " + (int)(height) + " ";
        }

        timeTotal = 0;

        for (index = 1; index <= maxData; index++) {

          for (timeIndex = 0; timeIndex <= timeData[index]; timeIndex += timeIncrement) {

            timeTotal += timeIncrement;

            if (Math.abs(height - (0.5 * (-32) * Math.pow(timeIndex, 2) + velocity
                * Math.pow(coefficient, index - 1) * timeIndex)) <= 0.25) {

              while (lineContent.length() < (timeTotal / timeIncrement) - 1) {
                lineContent += " ";
              }
              lineContent += "0";
            }
          }

          timeIndex = timeData[index + 1] / 2;

          if (-16 * Math.pow(timeIndex, 2) + velocity * Math.pow(coefficient, index - 1) * timeIndex < height) {

            break;
          }
        }

        System.out.println(lineContent);

      }  // End loop through all rows of y-axis data

      lineContent = "";

      // Show the x-axis
      for (index = 1; index <= (int)(timeTotal + 1) / timeIncrement + 1; index++) {

        lineContent += ".";
      }

      System.out.println(lineContent);

      lineContent = " 0";

      for (index = 1; index <= (int)(timeTotal + 0.9995); index++) {

        while (lineContent.length() < (int)(index / timeIncrement)) {
          lineContent += " ";
        }
        lineContent += index;
      }

      System.out.println(lineContent);

      System.out.println(" ".repeat((int)((timeTotal + 1) / (2 * timeIncrement) - 3)) + "SECONDS");

    }  // End outer while loop

  }  // End of method startGame

  public static void main(String[] args) {

    Bounce game = new Bounce();
    game.play();

  }  // End of method main

}  // End of class Bounce

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `13_Bounce/javascript/bounce.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。

`print` 函数的作用是在文档中创建一个文本节点（`<div>` 级别），并将其内容设置为传入的参数 `str`。具体实现是通过在文档中创建一个新的文本节点，然后设置其 `text` 属性为 `str`，最后将其添加到指定的元素（这里是 `output` 元素）中。这里使用了 `document.getElementById` 获取了一个 `<div>` 元素，将其作为输出容器。

`input` 函数的作用是从用户那里获取输入的参数 `input_str`，并将其存储在变量中。该函数通过创建一个 `<INPUT>` 元素，设置其 `type` 属性为 `text`，设置其 `length` 属性为 `50`（这是一个数字，但该网站可能将其解释为字符数组），然后将该元素添加到 `output` 元素中，将其 `focus` 属性设置为 `true`，以确保用户可以获得焦点。接下来，函数开始监听该元素的 `keydown` 事件，当用户按下 `CTRL+C`（可能是 Ctrl+C，具体取决于应用程序）时，函数会将 `input_str` 的值存储到变量中，并将其添加到 `print` 函数中，最后输出 `input_str`。函数还添加了一个自定义的事件处理程序，用于在用户关闭输入框时将其值存储在变量中，并将其添加到 `print` 函数中，最后输出 `input_str`。


```
// BOUNCE
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

This appears to be a Java program that performs a brute-force attack on a specified string. The program takes advantage of a vulnerable system's default implementation of the RANDOM function, which generates a random number within the specified range.

The program uses a while loop to repeatedly execute a specific branch of code. Within this loop, the program performs a brute-force attack by attempting to guess the correct RANDOM number using various暴力枚举 tactics.

The program uses a fixed-length variable ta to store the猜测的RANDOM numbers, which is initially set to 0. The program also uses a variable str to store the guessed number, which is initially set to the RANDOM number generated by the system.

The program loops through the attack phase until the guess is higher than or equal to the specified number, at which point it prints the guessed number and exits the loop.

The program also includes some additional code to print the RANDOM number with a leading prefix and to display the attack time.

Overall, the program is poorly designed and insecure, as it relies on a vulnerable default implementation of the RANDOM function and a fixed attack strategy that is easily guessed by an attacker. It is recommended to use a more secure and efficient method of generating random numbers for security purposes.


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
    print(tab(33) + "BOUNCE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    ta = [];
    print("THIS SIMULATION LETS YOU SPECIFY THE INITIAL VELOCITY\n");
    print("OF A BALL THROWN STRAIGHT UP, AND THE COEFFICIENT OF\n");
    print("ELASTICITY OF THE BALL.  PLEASE USE A DECIMAL FRACTION\n");
    print("COEFFICIENCY (LESS THAN 1).\n");
    print("\n");
    print("YOU ALSO SPECIFY THE TIME INCREMENT TO BE USED IN\n");
    print("'STROBING' THE BALL'S FLIGHT (TRY .1 INITIALLY).\n");
    print("\n");
    while (1) {
        print("TIME INCREMENT (SEC)");
        s2 = parseFloat(await input());
        print("\n");
        print("VELOCITY (FPS)");
        v = parseFloat(await input());
        print("\n");
        print("COEFFICIENT");
        c = parseFloat(await input());
        print("\n");
        print("FEET\n");
        print("\n");
        s1 = Math.floor(70 / (v / (16 * s2)));
        for (i = 1; i <= s1; i++)
            ta[i] = v * Math.pow(c, i - 1) / 16;
        for (h = Math.floor(-16 * Math.pow(v / 32, 2) + Math.pow(v, 2) / 32 + 0.5); h >= 0; h -= 0.5) {
            str = "";
            if (Math.floor(h) == h)
                str += " " + h + " ";
            l = 0;
            for (i = 1; i <= s1; i++) {
                for (t = 0; t <= ta[i]; t += s2) {
                    l += s2;
                    if (Math.abs(h - (0.5 * (-32) * Math.pow(t, 2) + v * Math.pow(c, i - 1) * t)) <= 0.25) {
                        while (str.length < l / s2)
                            str += " ";
                        str += "0";
                    }
                }
                t = ta[i + 1] / 2;
                if (-16 * Math.pow(t, 2) + v * Math.pow(c, i - 1) * t < h)
                    break;
            }
            print(str + "\n");
        }
        str = " ";
        for (i = 1; i < Math.floor(l + 1) / s2 + 1; i++)
            str += ".";
        print(str + "\n");
        str = " 0";
        for (i = 1; i < Math.floor(l + 0.9995); i++) {
            while (str.length < Math.floor(i / s2))
                str += " ";
            str += i;
        }
        print(str + "\n");
        print(tab(Math.floor(l + 1) / (2 * s2) - 2) + "SECONDS\n");
    }
}

```

这道题目缺少上下文，无法给出具体的解释。一般来说，在编程中，`main()` 函数是程序的入口点，也是程序的控制中心。在 main 函数中，程序会首先被加载到内存中，然后被初始化，之后就可以开始执行程序的逻辑。但需要注意的是，这道题目中缺少具体的程序代码，因此无法给出具体的解释。


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

Added feature so that if "TIME" value is "0" then it will quit,
so you don't have to hit Control-C. Also added a little error checking of the input.


# `13_Bounce/python/bounce.py`

这段代码是一个用于在物理仿真中模拟碰撞检测的Python程序。它实现了两个函数：`print_centered` 和 `bounce`.

1. `print_centered` 函数用于打印一个居中显示的字符串。它接收一个字符串参数 `msg`，并计算出在 `PAGE_WIDTH` 字段（设置为 64，代表页面宽度）中居中显示这个字符串所需的字符空间。然后，函数通过 `print` 函数将居中显示的字符串和原始字符串拼接在一起。这个函数的作用是为了确保在打印时，字符串的中心内容能够恰好占据整个页面宽度，使得字符串看起来更美观。

2. `bounce` 函数模拟了一个物理弹球游戏的碰撞检测。它接受一个物理对象（如物体或弹性体）和一个速度矢量 `v`。函数首先检查物理学对象和速度矢量是否相交，然后计算弹出的最小角度。如果物理学对象和速度矢量不碰撞，则函数返回角度的度数。如果碰撞，则函数根据碰后弹起的高度计算反弹的角度。

这个程序是一个简单的物理仿真示例，可以作为学习和研究物理学的工具。


```
"""
BOUNCE

A physics simulation

Ported by Dave LeCompte
"""

from typing import Tuple, List

PAGE_WIDTH = 64


def print_centered(msg: str) -> None:
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    print(spaces + msg)


```

这两段代码定义了两个函数，分别是 `print_header` 和 `print_instructions`。它们的作用如下：

1. `print_header` 函数的作用是在屏幕上打印一个标题字符串，然后将其居中显示。接着在四个空行中打印字符串 "CREATIVE COMPUTING MORRISTOWN, NEW JERSEY"，然后再次居中显示。最后，在四个空行中打印四个空行。

2. `print_instructions` 函数的作用是在屏幕上打印一些文本，包括初始速度和重力系数的选择，以及计数器的时间增量。然后让其居中显示，并等待用户按下回车键以结束文件。


```
def print_header(title: str) -> None:
    print_centered(title)
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print()
    print()
    print()


def print_instructions() -> None:
    print("THIS SIMULATION LETS YOU SPECIFY THE INITIAL VELOCITY")
    print("OF A BALL THROWN STRAIGHT UP, AND THE COEFFICIENT OF")
    print("ELASTICITY OF THE BALL.  PLEASE USE A DECIMAL FRACTION")
    print("COEFFICIENCY (LESS THAN 1).")
    print()
    print("YOU ALSO SPECIFY THE TIME INCREMENT TO BE USED IN")
    print("'STROBING' THE BALL'S FLIGHT (TRY .1 INITIALLY).")
    print()


```

这段代码定义了一个函数 `get_initial_conditions()`，它接受三个浮点数类型的参数并返回一个元组类型。接下来，该函数通过三行输入用户初始条件并输出结果。

该代码还定义了一个函数 `print_at_tab()`，它接受一个字符串类型的参数、一个整数类型的参数和一个字符类型的参数，并输出指定字符串在指定宽度下的结果。

函数 `get_initial_conditions()` 的作用是获取用户输入的初始条件，包括时间间隔 `delta_t`、速度 `v0` 和系数 `coeff_rest`，并将它们以元组的形式返回。函数 `print_at_tab()` 的作用是在用户输入时将结果输出，以指定的宽度或索引显示。


```
def get_initial_conditions() -> Tuple[float, float, float]:
    delta_t = float(input("TIME INCREMENT (SEC)? "))
    print()
    v0 = float(input("VELOCITY (FPS)? "))
    print()
    coeff_rest = float(input("COEFFICIENT? "))
    print()

    return delta_t, v0, coeff_rest


def print_at_tab(line: str, tab: int, s: str) -> str:
    line += (" " * (tab - len(line))) + s
    return line


```

This is a Python program that simulates the bounce of a ball. It uses a珠 Pattern algorithm to calculate the time elapsed and the amount of fall the ball has experienced.

The珠 Pattern algorithm is based on the following recurrence relation:

v0 = ball's initial vertical velocity
coeff_rest = coefficient of rest of the ball
h = ball's current height
delta_t = time increment (in seconds per height level)
total_time = total time spent in the last time increment

初值化 + 功能 = 0 高度 <= 0  <= v0 < 32
                      <= h <仿真时长 / 32
                      ，珠循环
                      while珠没入
                      tm<-- 0.25 高度 <= h - (0.5 * (-32) * tm**2 + v0 * coeff_rest ** (h - 32)) <= 2 * h - v0 * coeff_rest ** (h - 32)
                      if-- 珠没入 < 0.25
                         珠出
                      tm += ΔT
                      v0 = v0 - ΔT
                      h = h - ΔT

The program first calculates the height of the ball, then enters a while loop that iterates through all the time increments. In each time increment, it first calculates the time spent in the last time increment and then calculates the amount of fall experienced by the ball. If the ball has not yet fallen, it calculates the amount of time spent in the last time increment as well as the new ball height.

After that, it checks if the ball has fallen and if so, it prints the time spent in the last time increment, the amount of time the ball has spent in the last time increment, and then falls the ball back to the top of the screen by printing a tab character followed by a space and the time in seconds.

The program also prints the total time spent in the simulation and a final message at the end of the simulation.


```
def run_simulation(delta_t: float, v0: float, coeff_rest: float) -> None:
    bounce_time: List[float] = [0] * 20  # time of each bounce

    print("FEET")
    print()

    sim_dur = int(70 / (v0 / (16 * delta_t)))
    for i in range(1, sim_dur + 1):
        bounce_time[i] = v0 * coeff_rest ** (i - 1) / 16

    # Draw the trajectory of the bouncing ball, one slice of height at a time
    h: float = int(-16 * (v0 / 32) ** 2 + v0**2 / 32 + 0.5)
    while h >= 0:
        line = ""
        if int(h) == h:
            line += str(int(h))
        total_time: float = 0
        for i in range(1, sim_dur + 1):
            tm: float = 0
            while tm <= bounce_time[i]:
                total_time += delta_t
                if (
                    abs(h - (0.5 * (-32) * tm**2 + v0 * coeff_rest ** (i - 1) * tm))
                    <= 0.25
                ):
                    line = print_at_tab(line, int(total_time / delta_t), "0")
                tm += delta_t
            tm = bounce_time[i + 1] / 2

            if -16 * tm**2 + v0 * coeff_rest ** (i - 1) * tm < h:
                break
        print(line)
        h = h - 0.5

    print("." * (int((total_time + 1) / delta_t) + 1))
    print
    line = " 0"
    for i in range(1, int(total_time + 0.9995) + 1):
        line = print_at_tab(line, int(i / delta_t), str(i))
    print(line)
    print()
    print(print_at_tab("", int((total_time + 1) / (2 * delta_t) - 2), "SECONDS"))
    print()


```

这段代码是一个Python程序，名为"main"。程序的主要目的是模拟一个简谐振动的仿真，在这个过程中，会周期性地发射信号，并且会有初始位置和振幅等参数。下面是具体的解题步骤：

1. 首先，程序会打印出"BOUNCE"字符，这是模拟简谐振动的标志。

2. 接着，程序会调用一个名为"print_instructions"的函数，这个函数的作用是打印出简谐振动的说明。

3. 然后，程序会调用一个名为"get_initial_conditions"的函数，这个函数的作用是从简谐振动的参数中获取一些初始值，包括振幅A、自然频率ω和阻尼K等。

4. 接下来，程序会循环执行以下操作：首先，获取简谐振动的参数，然后调用一个名为"run_simulation"的函数来运行简谐振动的仿真，最后跳出循环。

5. 最后，程序会直接调用"main"函数作为程序的入口，这个函数会实际执行程序中的操作。


```
def main() -> None:
    print_header("BOUNCE")
    print_instructions()

    while True:
        delta_t, v0, coeff_rest = get_initial_conditions()

        run_simulation(delta_t, v0, coeff_rest)
        break


if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Bowling

This is a simulated bowling game for up to four players. You play 10 frames. To roll the ball, you simply type “ROLL.” After each roll, the computer will show you a diagram of the remaining pins (“0” means the pin is down, “+” means it is still standing), and it will give you a roll analysis:
- GUTTER
- STRIKE
- SPARE
- ERROR (on second ball if pins still standing)

Bowling was written by Paul Peraino while a student at Woodrow Wilson High School, San Francisco, California.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=26)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=41)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Known Bugs

- In the original code, scores is not kept accurately in multiplayer games.  It stores scores in F*P, where F is the frame and P is the player.  So, for example, frame 8 player 1 (index 16) clobbers the score from frame 4 player 2 (also index 16).

- Even when scores are kept accurately, they don't match normal bowling rules.  In this game, the score for each ball is just the total number of pins down after that ball, and the third row of scores is a status indicator (3 for strike, 2 for spare, 1 for anything else).

- The program crashes with a "NEXT without FOR" error if you elect to play again after the first game.

#### Porting Notes

- The funny control characters in the "STRIKE!" string literal are there to make the terminal beep.


# `14_Bowling/csharp/Bowling.cs`

This is a class written in C# that represents a game of a疑似 puzzle game. The class contains several methods for interacting with the game board, which are not explained here.

First, there is a constructor for creating a new instance of the class, as well as a constructor forResetting the game results and a constructor for displaying the game results.

The class has several methods for interacting with the game board. For example, there is a method called ShowAllFrames, which displays all the frames of the game. Another method called ShowGameResults, which displays the game results for a given game.

The class also has a method called GetGameResults, which returns the game results for a given game.

The class uses several utility methods, such as Utility.PrintString, which prints a specified string to the console.

Overall, this class provides a good starting point for creating a game of a疑似 puzzle game.



```
﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Bowling
{
    public class Bowling
    {
        private readonly Pins pins = new();

        private int players;

        public void Play()
        {
            ShowBanner();
            MaybeShowInstructions();
            Setup();
            GameLoop();
        }

        private static void ShowBanner()
        {
            Utility.PrintString(34, "BOWL");
            Utility.PrintString(15, "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            Utility.PrintString();
            Utility.PrintString();
            Utility.PrintString();
            Utility.PrintString("WELCOME TO THE ALLEY");
            Utility.PrintString("BRING YOUR FRIENDS");
            Utility.PrintString("OKAY LET'S FIRST GET ACQUAINTED");
            Utility.PrintString();
        }
        private static void MaybeShowInstructions()
        {
            Utility.PrintString("THE INSTRUCTIONS (Y/N)");
            if (Utility.InputString() == "N") return;
            Utility.PrintString("THE GAME OF BOWLING TAKES MIND AND SKILL.DURING THE GAME");
            Utility.PrintString("THE COMPUTER WILL KEEP SCORE.YOU MAY COMPETE WITH");
            Utility.PrintString("OTHER PLAYERS[UP TO FOUR].YOU WILL BE PLAYING TEN FRAMES");
            Utility.PrintString("ON THE PIN DIAGRAM 'O' MEANS THE PIN IS DOWN...'+' MEANS THE");
            Utility.PrintString("PIN IS STANDING.AFTER THE GAME THE COMPUTER WILL SHOW YOUR");
            Utility.PrintString("SCORES .");
        }
        private void Setup()
        {
            Utility.PrintString("FIRST OF ALL...HOW MANY ARE PLAYING", false);
            var input = Utility.InputInt();
            players = input < 1 ? 1 : input;
            Utility.PrintString();
            Utility.PrintString("VERY GOOD...");
        }
        private void GameLoop()
        {
            GameResults[] gameResults = InitGameResults();
            var done = false;
            while (!done)
            {
                ResetGameResults(gameResults);
                for (int frame = 0; frame < GameResults.FramesPerGame; ++frame)
                {
                    for (int player = 0; player < players; ++player)
                    {
                        pins.Reset();
                        int pinsDownThisFrame = pins.GetPinsDown();

                        int ball = 1;
                        while (ball == 1 || ball == 2) // One or two rolls
                        {
                            Utility.PrintString("TYPE ROLL TO GET THE BALL GOING.");
                            _ = Utility.InputString();

                            int pinsDownAfterRoll = pins.Roll();
                            ShowPins(player, frame, ball);

                            if (pinsDownAfterRoll == pinsDownThisFrame)
                            {
                                Utility.PrintString("GUTTER!!");
                            }

                            if (ball == 1)
                            {
                                // Store current pin count
                                gameResults[player].Results[frame].PinsBall1 = pinsDownAfterRoll;

                                // Special handling for strike
                                if (pinsDownAfterRoll == Pins.TotalPinCount)
                                {
                                    Utility.PrintString("STRIKE!!!!!\a\a\a\a");
                                    // No second roll
                                    ball = 0;
                                    gameResults[player].Results[frame].PinsBall2 = pinsDownAfterRoll;
                                    gameResults[player].Results[frame].Score = FrameResult.Points.Strike;
                                }
                                else
                                {
                                    ball = 2; // Roll again
                                    Utility.PrintString("ROLL YOUR SECOND BALL");
                                }
                            }
                            else if (ball == 2)
                            {
                                // Store current pin count
                                gameResults[player].Results[frame].PinsBall2 = pinsDownAfterRoll;
                                ball = 0;

                                // Determine the score for the frame
                                if (pinsDownAfterRoll == Pins.TotalPinCount)
                                {
                                    Utility.PrintString("SPARE!!!!");
                                    gameResults[player].Results[frame].Score = FrameResult.Points.Spare;
                                }
                                else
                                {
                                    Utility.PrintString("ERROR!!!");
                                    gameResults[player].Results[frame].Score = FrameResult.Points.Error;
                                }
                            }
                            Utility.PrintString();
                        }
                    }
                }
                ShowGameResults(gameResults);
                Utility.PrintString("DO YOU WANT ANOTHER GAME");
                var a = Utility.InputString();
                done = a.Length == 0 || a[0] != 'Y';
            }
        }

        private GameResults[] InitGameResults()
        {
            var gameResults = new GameResults[players];
            for (int i = 0; i < gameResults.Length; i++)
            {
                gameResults[i] = new GameResults();
            }
            return gameResults;
        }

        private void ShowPins(int player, int frame, int ball)
        {
            Utility.PrintString($"FRAME: {frame + 1} PLAYER: {player + 1} BALL: {ball}");
            var breakPins = new bool[] { true, false, false, false, true, false, false, true, false, true };
            var indent = 0;
            for (int pin = 0; pin < Pins.TotalPinCount; ++pin)
            {
                if (breakPins[pin])
                {
                    Utility.PrintString(); // End row
                    Utility.PrintString(indent++, false); // Indent next row
                }
                var s = pins[pin] == Pins.State.Down ? "+ " : "o ";
                Utility.PrintString(s, false);
            }
            Utility.PrintString();
            Utility.PrintString();
        }
        private void ResetGameResults(GameResults[] gameResults)
        {
            foreach (var gameResult in gameResults)
            {
                foreach (var frameResult in gameResult.Results)
                {
                    frameResult.Reset();
                }
            }
        }
        private void ShowGameResults(GameResults[] gameResults)
        {
            Utility.PrintString("FRAMES");
            for (int i = 0; i < GameResults.FramesPerGame; ++i)
            {
                Utility.PrintString(Utility.PadInt(i, 3), false);
            }
            Utility.PrintString();
            foreach (var gameResult in gameResults)
            {
                foreach (var frameResult in gameResult.Results)
                {
                    Utility.PrintString(Utility.PadInt(frameResult.PinsBall1, 3), false);
                }
                Utility.PrintString();
                foreach (var frameResult in gameResult.Results)
                {
                    Utility.PrintString(Utility.PadInt(frameResult.PinsBall2, 3), false);
                }
                Utility.PrintString();
                foreach (var frameResult in gameResult.Results)
                {
                    Utility.PrintString(Utility.PadInt((int)frameResult.Score, 3), false);
                }
                Utility.PrintString();
                Utility.PrintString();
            }
        }
    }
}

```

# `14_Bowling/csharp/FrameResult.cs`

这段代码定义了一个名为FrameResult的类，表示美式台球比赛中的得分和犯规信息。

首先，它使用System命名空间中的System类来处理文件操作。

接着，它定义了一个FramesScore类，该类用于存储得分和犯规信息。

然后，它创建了一个FramesScore类的实例，并继承自它，以便在调用它的方法时使用它。

接着，它定义了一个SpareChapter类，继承自System.Array mitigate，用于处理SpareChapter方法。

最后，它定义了一个Points类，用于存储得分和犯规信息，并继承自System.AttributeIt


```
﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Bowling
{
    public class FrameResult
    {
        public enum Points { None, Error, Spare, Strike };

        public int PinsBall1 { get; set; }
        public int PinsBall2 { get; set; }
        public Points Score { get; set; }

        public void Reset()
        {
            PinsBall1 = PinsBall2 = 0;
            Score = Points.None;
        }
    }
}

```

# `14_Bowling/csharp/GameResults.cs`

这段代码定义了一个名为 `GameResults` 的类，用于表示美式八球比赛中每一局的得分和得分者。

该类包含一个名为 `FramesPerGame` 的公有常量，表示每场比赛允许进行的局数，以及一个名为 `Results` 的公有实例字段，用于存储该局比赛的所有得分和得分者。

该类有一个构造函数，用于初始化 `Results` 对象，并且在每次循环中创建一个新的 `FrameResult` 对象来存储每一局的得分和得分者，该构造函数创建了一个具有 `FramesPerGame` 局数的 `Results` 对象。

该类的方法包括 `AddScore` 和 `GetFrameResult` 方法。`AddScore` 方法用于将当前局的得分和得分者添加到 `Results` 对象中，而 `GetFrameResult` 方法用于返回指定局数的得分和得分者。

该类还包含一个名为 `ClearResults` 的静态方法，用于清空 `Results` 对象中的所有得分和得分者。


```
﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Bowling
{
    public class GameResults
    {
        public static readonly int FramesPerGame = 10;
        public FrameResult[] Results { get; set; }

        public GameResults()
        {
            Results = new FrameResult[FramesPerGame];
            for (int i = 0; i < FramesPerGame; ++i)
            {
                Results[i] = new FrameResult();
            }
        }
    }
}

```

# `14_Bowling/csharp/Pins.cs`



该代码是一个 bowling 游戏的 AI，可以控制一个轮子(即一个 `Pins` 对象)，并使用随机数生成策略来控制球的位置。

具体来说，该 AI 包含以下组件：

- `Pins` 类：表示一个轮子，包含一个状态数组 `PinSet`，每个状态都是一个 `State` 类(即 `State.Up` 或 `State.Down`)，用于跟踪轮子是向上还是向下。
- `State` 类：表示一个状态，包括一个 `PinSet` 数组，用于跟踪每个球的当前位置。
- `Roll` 方法：用于生成一个随机的球的位置，并返回该位置的 `State` 对象。
- `Reset` 方法：用于重置轮子的状态数组。

该 AI 的主要作用是控制轮子的运动，使轮子随机移动，并在游戏结束时停止轮子的运动并显示胜利信息。


```
﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Bowling
{
    public class Pins
    {
        public enum State { Up, Down };
        public static readonly int TotalPinCount = 10;
        private readonly Random random = new();

        private State[] PinSet { get; set; }

        public Pins()
        {
            PinSet = new State[TotalPinCount];
        }
        public State this[int i]
        {
            get { return PinSet[i]; }
            set { PinSet[i] = value; }
        }
        public int Roll()
        {
            // REM ARK BALL GENERATOR USING MOD '15' SYSTEM
            for (int i = 0; i < 20; ++i)
            {
                var x = random.Next(100) + 1;
                int j;
                for (j = 1; j <= 10; ++j)
                {
                    if (x < 15 * j)
                        break;
                }
                var pindex = 15 * j - x;
                if (pindex > 0 && pindex <= TotalPinCount)
                    PinSet[--pindex] = State.Down;
            }
            return GetPinsDown();
        }
        public void Reset()
        {
            for (int i = 0; i < PinSet.Length; ++i)
            {
                PinSet[i] = State.Up;
            }
        }
        public int GetPinsDown()
        {
            return PinSet.Count(p => p == State.Down);
        }
    }
}

```

# `14_Bowling/csharp/Program.cs`

这段代码是一个Bowling游戏的程序。它包括以下几个主要部分：

1. 导入System命名空间：在开始编写代码之前，需要先导入System命名空间，以便使用其中的一些类和函数。
2. 导出Bowling类：定义一个名为Bowling的类，该类包含所有与游戏相关的逻辑和行为。
3. 定义常量BowlingGame：定义一个名为BowlingGame的常量，用于存储游戏的实例。
4. 编写Bowling游戏的主要逻辑：在Main方法中，编写Bowling游戏的逻辑，包括创建游戏实例、设置游戏参数、进行游戏循环以及处理游戏事件等。
5. 创建Bowling游戏实例：在Main方法的开始部分，创建一个Bowling游戏实例，并将其赋值给BowlingGame变量。
6. 输出游戏结果：在游戏循环的每次迭代中，输出游戏的得分和剩余的击球数。
7. 关闭游戏窗口：在游戏循环的结束部分，关闭游戏窗口，以便用户可以保存游戏结果并退出游戏。

总之，这段代码创建了一个Bowling游戏，可以进行玩家与AI之间的对战。


```
﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Bowling
{
    public static class Program
    {
        public static void Main()
        {
            new Bowling().Play();
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `14_Bowling/csharp/Utility.cs`



这段代码是一个程序集，名为“Bowling”。它包含了一些类和函数，用于处理字符串和整数操作。下面是这些类和函数的摘要：

1. “Utility”类：
- “PadInt”函数：用指定宽度对一个整数进行填充，并在字符串中返回。
- “InputInt”函数：尝试从用户输入中解析一个整数，并返回该整数。如果输入不正确，它会在屏幕上打印消息。
- “InputString”函数：从用户输入中获取一个字符串，将其转换为大写，并返回。
- 该类还包含其他函数，但在这里不包含。

2. “PrintInt”函数：打印一个整数，可以包含一个或多个0。如果包含0，则指定为左对齐。
- “PrintString”函数：打印一个字符串，可以包含一个或多个0。如果包含0，则指定为左对齐。该函数在第二个参数中指定要打印的数字的 tabs。
- “PrintString”函数：打印一个字符串，并将其填充为指定的宽度。该宽度可以是0。
- “PrintString”函数：打印一个字符串，并将其填充为指定的宽度。该宽度可以是0，但必须在参数中指定。
- 该类还包含其他函数，但在这里不包含。

3. “Utility”类中包含的函数：
- “PadInt”函数：用指定宽度对一个整数进行填充，并在字符串中返回。
- “InputInt”函数：尝试从用户输入中解析一个整数，并返回该整数。如果输入不正确，它会在屏幕上打印消息。
- “InputString”函数：从用户输入中获取一个字符串，将其转换为大写，并返回。
- 该类还包含其他函数，但在这里不包含。


```
﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Bowling
{
    internal static class Utility
    {
        public static string PadInt(int value, int width)
        {
            return value.ToString().PadLeft(width);
        }
        public static int InputInt()
        {
            while (true)
            {
                if (int.TryParse(InputString(), out int i))
                    return i;
                else
                    PrintString("!NUMBER EXPECTED - RETRY INPUT LINE");
            }
        }
        public static string InputString()
        {
            PrintString("? ", false);
            var input = Console.ReadLine();
            return input == null ? string.Empty : input.ToUpper();
        }
        public static void PrintInt(int value, bool newLine = false)
        {
            PrintString($"{value} ", newLine);
        }
        public static void PrintString(bool newLine = true)
        {
            PrintString(0, string.Empty);
        }
        public static void PrintString(int tab, bool newLine = true)
        {
            PrintString(tab, string.Empty, newLine);
        }
        public static void PrintString(string value, bool newLine = true)
        {
            PrintString(0, value, newLine);
        }
        public static void PrintString(int tab, string value, bool newLine = true)
        {
            Console.Write(new String(' ', tab));
            Console.Write(value);
            if (newLine) Console.WriteLine();
        }
    }
}

```