# BasicComputerGames源码解析 85

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


# `90_Tower/python/tower.py`

这段代码定义了一个名为 Disk 的类，用于管理磁盘驱动器的大小。

首先，在类中定义了一个构造函数，该函数接收一个参数 `size`，表示磁盘驱动器的容量，容量必须是整数类型。

然后，定义了两个方法，一个用于获取磁盘驱动器的大小，另一个用于打印有关磁盘驱动器的信息。

接下来，在 `main` 函数中创建了一个 Disk 类的实例，并传递了磁盘驱动器的大小，例如 10 和 20。

最后，使用 `print` 函数打印有关磁盘驱动器的信息，特别是驱动器的大小。


```
import sys
from typing import List, Optional


class Disk:
    def __init__(self, size: int) -> None:
        self.__size = size

    def size(self) -> int:
        return self.__size

    def print(self) -> None:
        print("[ %s ]" % self.size())


```

这段代码定义了一个名为Tower的类，具有以下属性和方法：

- `__init__`：初始化方法，创建一个空列表并将其设置为`self.__disks`。
- `empty`：判断列表是否为空，返回真时为空，否则为满。
- `top`：返回列表的最后一个 disk，如果列表为空，则返回None。
- `add`：向列表末尾添加一个新的disk，前提条件是列表不为空。如果列表已经为空，则会引发异常。
- `pop`：删除列表的最后一个disk，并返回它。如果列表为空，则会引发异常。
- `print`：打印列表中所有disk的大小，每个disk之间用逗号分隔。

Tower类创建了一种形状特殊的列表，可以在其中添加、删除和打印disk。例如，以下代码创建了一个Tower实例，并向其中添加了一个大型的Disk:

```
t = Tower()
t.add(Disk(10))
t.add(Disk(20))
t.add(Disk(30))
t.print()
```

这将输出类似以下内容的结果：

```
Needle: [10.0, 20.0, 30.0]
```

注意，`add`方法在添加新的disk时，会检查列表是否为空。如果是，则会引发`Exception`。这是因为在代码中，我们假设列表总是满的，所以如果没有添加任何disk，程序可能会崩溃。


```
class Tower:
    def __init__(self) -> None:
        self.__disks: List[Disk] = []

    def empty(self) -> bool:
        return len(self.__disks) == 0

    def top(self) -> Optional[Disk]:
        if self.empty():
            return None
        else:
            return self.__disks[-1]

    def add(self, disk: Disk) -> None:
        if not self.empty():
            t = self.top()
            assert t is not None  # cannot happen as it's not empty
            if disk.size() > t.size():
                raise Exception(
                    "YOU CAN'T PLACE A LARGER DISK ON TOP OF A SMALLER ONE, IT MIGHT CRUSH IT!"
                )
        self.__disks.append(disk)

    def pop(self) -> Disk:
        if self.empty():
            raise Exception("empty pop")
        return self.__disks.pop()

    def print(self) -> None:
        r = "Needle: [%s]" % (", ".join([str(x.size()) for x in self.__disks]))
        print(r)


```

This is a simple game where the player must choose a disk from a group of disks, each of which has a unique top number that increases by one each turn. The player can also turn the disk they are holding onto another disk, which may be valid or not. The valid turns are displayed with a message and the number of valid moves the player is allowed to make is displayed with a message. If the player runs out of valid moves, the game ends and the player is given the opportunity to choose a different disk.


```
class Game:
    def __init__(self) -> None:
        # use fewer sizes to make debugging easier
        # self.__sizes = [3, 5, 7]  # ,9,11,13,15]
        self.__sizes = [3, 5, 7, 9, 11, 13, 15]

        self.__sizes.sort()

        self.__towers = []
        self.__moves = 0
        self.__towers = [Tower(), Tower(), Tower()]
        self.__sizes.reverse()
        for size in self.__sizes:
            disk = Disk(size)
            self.__towers[0].add(disk)

    def winner(self) -> bool:
        return self.__towers[0].empty() and self.__towers[1].empty()

    def print(self) -> None:
        for t in self.__towers:
            t.print()

    def moves(self) -> int:
        return self.__moves

    def which_disk(self) -> int:
        w = int(input("WHICH DISK WOULD YOU LIKE TO MOVE\n"))
        if w in self.__sizes:
            return w
        raise Exception()

    def pick_disk(self) -> Optional[Tower]:
        which = None
        while which is None:
            try:
                which = self.which_disk()
            except Exception:
                print("ILLEGAL ENTRY... YOU MAY ONLY TYPE 3,5,7,9,11,13, OR 15.\n")

        valids = [t for t in self.__towers if t.top() and t.top().size() == which]
        assert len(valids) in (0, 1)
        if not valids:
            print("THAT DISK IS BELOW ANOTHER ONE.  MAKE ANOTHER CHOICE.\n")
            return None
        else:
            assert valids[0].top().size() == which
            return valids[0]

    def which_tower(self) -> Optional[Tower]:
        try:
            needle = int(input("PLACE DISK ON WHICH NEEDLE\n"))
            tower = self.__towers[needle - 1]
        except Exception:
            print(
                "I'LL ASSUME YOU HIT THE WRONG KEY THIS TIME.  BUT WATCH IT,\nI ONLY ALLOW ONE MISTAKE.\n"
            )
            return None
        else:
            return tower

    def take_turn(self) -> None:
        from_tower = None
        while from_tower is None:
            from_tower = self.pick_disk()

        to_tower = self.which_tower()
        if not to_tower:
            to_tower = self.which_tower()

        if not to_tower:
            print("I TRIED TO WARN YOU, BUT YOU WOULDN'T LISTEN.\nBYE BYE, BIG SHOT.\n")
            sys.exit(0)

        disk = from_tower.pop()
        try:
            to_tower.add(disk)
            self.__moves += 1
        except Exception as err:
            print(err)
            from_tower.add(disk)


```

这段代码是一个Python程序，它解释了一个有趣的谜题。谜题是在一个有限的时间内，将最小的3个盘子移动到编号为1的盘子。通过编号为2的盘子，可以看到下一个盘子是编号为3的盘子，然后继续通过这种方式推算出编号为4的盘子，以此类推，直到编号为15的盘子。

程序的作用是让用户通过交互式的方式，让其在有限的时间内尝试将编号为1到3的盘子移动到编号为3的盘子。当用户成功地将盘子移动到正确的位置时，程序会显示恭喜信息。当用户在尝试移动盘子之后，程序会检查是否达到了128个移动次数的限制，如果不是，则程序会显示一个错误消息并退出。如果用户在移动盘子过程中超过了128个，则程序会显示一个错误消息并退出。


```
def main() -> None:
    print(
        """
    IN THIS PROGRAM, WE SHALL REFER TO DISKS BY NUMERICAL CODE.
    3 WILL REPRESENT THE SMALLEST DISK, 5 THE NEXT SIZE,
    7 THE NEXT, AND SO ON, UP TO 15.  IF YOU DO THE PUZZLE WITH
    2 DISKS, THEIR CODE NAMES WOULD BE 13 AND 15.  WITH 3 DISKS
    THE CODE NAMES WOULD BE 11, 13 AND 15, ETC.  THE NEEDLES
    ARE NUMBERED FROM LEFT TO RIGHT, 1 TO 3.  WE WILL
    START WITH THE DISKS ON NEEDLE 1, AND ATTEMPT TO MOVE THEM
    TO NEEDLE 3.

    GOOD LUCK!

    """
    )

    game = Game()
    while True:
        game.print()

        game.take_turn()

        if game.winner():
            print(
                "CONGRATULATIONS!!\nYOU HAVE PERFORMED THE TASK IN %s MOVES.\n"
                % game.moves()
            )
            while True:
                yesno = input("TRY AGAIN (YES OR NO)\n")
                if yesno.upper() == "YES":
                    game = Game()
                    break
                elif yesno.upper() == "NO":
                    print("THANKS FOR THE GAME!\n")
                    sys.exit(0)
                else:
                    print("'YES' OR 'NO' PLEASE\n")
        elif game.moves() > 128:
            print("SORRY, BUT I HAVE ORDERS TO STOP IF YOU MAKE MORE THAN 128 MOVES.")
            sys.exit(0)


```

这段代码是一个Python程序中的一个if语句。if语句是Python中的一種条件判斷語句，用于決定程序的執行流程。

if __name__ == "__main__":
```是if语句的伪代码，用于指定if语句的条件。如果当前程序是作为程序的主要入口点（即`__main__`文件），则程序会执行if语句内部的代码。

"__main__"是一个特殊的字符串，用于标识Python解释器是否需要执行if语句内部的代码。如果当前程序是作为程序的主要入口点，则程序会执行if语句内部的代码，否则不会执行。

因此，这段代码的作用是用于定义程序的执行流程，只有在程序作为主要入口点时才会执行if语句内部的代码。


```
if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Train

TRAIN is a program which uses the computer to generate problems with random initial conditions to teach about the time-speed-distance relationship (distance = rate x time). You then input your answer and the computer verifies your response.

TRAIN is merely an example of a student-generated problem. Maximum fun (and benefit) comes more from _writing_ programs like this as opposed to solving the specific problem posed. Exchange your program with others—you solve their problem and let them solve yours.

TRAIN was originally written in FOCAL by one student for use by others in his class. It was submitted to us by Walt Koetke, Lexington High School, Lexington, Mass.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=175)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=190)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `91_Train/csharp/Train/TrainGame.cs`

This is a class written in C# that embarks on a train journey simulation. The class provides functionality to calculate the duration of the journey, calculate the percentage difference if the user provides some information about the journey, and generate random numbers.

The `TrainJourney` class has several methods:

* `public static double CalculateCarJourneyDuration(double carSpeed, double timeDifference, double trainSpeed)`: This method calculates the duration of the journey based on the car speed, time difference, and train speed.
* `public static double GenerateRandomNumber(int baseSpeed, int multiplier)`: This method generates a random number within the specified base speed and multiplier.
* `public static bool IsWithinAllowedDifference(int percentageDifference, int allowedDifference)`: This method checks if the difference is within the specified allowed difference.
* `public static int CalculatePercentageDifference(double userInputCarJourneyDuration, double carJourneyDuration)`: This method calculates the percentage difference if the user provides some information about the journey.
* `public static void DisplayIntroText()`: This method displays an introduction message.

The `TrainJourney` class also has a `TryAgain()` method, which attempts to display a helpful message to the user if the initial input is not correct. If the user enters "NO", the `TryAgain()` method is called again, and if the user enters "YES", the current `TrainJourney` instance will generate a random number and display it as a reminder.


```
﻿using System;
using System.Linq;

namespace Train
{
    public class TrainGame
    {
        private Random Rnd { get; } = new Random();
        private readonly int ALLOWED_PERCENTAGE_DIFFERENCE = 5;

        static void Main()
        {
            TrainGame train = new TrainGame();
            train.GameLoop();
        }

        public void GameLoop()
        {
            DisplayIntroText();

            do
            {
                PlayGame();
            } while (TryAgain());
        }

        private void PlayGame()
        {
            int carSpeed = (int)GenerateRandomNumber(40, 25);
            int timeDifference = (int)GenerateRandomNumber(5, 15);
            int trainSpeed = (int)GenerateRandomNumber(20, 19);

            Console.WriteLine($"A CAR TRAVELING {carSpeed} MPH CAN MAKE A CERTAIN TRIP IN");
            Console.WriteLine($"{timeDifference} HOURS LESS THAN A TRAIN TRAVELING AT {trainSpeed} MPH");
            Console.WriteLine("HOW LONG DOES THE TRIP TAKE BY CAR?");

            double userInputCarJourneyDuration = double.Parse(Console.ReadLine());
            double actualCarJourneyDuration = CalculateCarJourneyDuration(carSpeed, timeDifference, trainSpeed);
            int percentageDifference = CalculatePercentageDifference(userInputCarJourneyDuration, actualCarJourneyDuration);

            if (IsWithinAllowedDifference(percentageDifference, ALLOWED_PERCENTAGE_DIFFERENCE))
            {
                Console.WriteLine($"GOOD! ANSWER WITHIN {percentageDifference} PERCENT.");
            }
            else
            {
                Console.WriteLine($"SORRY.  YOU WERE OFF BY {percentageDifference} PERCENT.");
            }
            Console.WriteLine($"CORRECT ANSWER IS {actualCarJourneyDuration} HOURS.");
        }

        public static bool IsWithinAllowedDifference(int percentageDifference, int allowedDifference)
        {
            return percentageDifference <= allowedDifference;
        }

        private static int CalculatePercentageDifference(double userInputCarJourneyDuration, double carJourneyDuration)
        {
            return (int)(Math.Abs((carJourneyDuration - userInputCarJourneyDuration) * 100 / userInputCarJourneyDuration) + .5);
        }

        public static double CalculateCarJourneyDuration(double carSpeed, double timeDifference, double trainSpeed)
        {
            return timeDifference * trainSpeed / (carSpeed - trainSpeed);
        }

        public double GenerateRandomNumber(int baseSpeed, int multiplier)
        {
            return multiplier * Rnd.NextDouble() + baseSpeed;
        }

        private bool TryAgain()
        {
            Console.WriteLine("ANOTHER PROBLEM (YES OR NO)? ");
            return IsInputYes(Console.ReadLine());
        }

        public static bool IsInputYes(string consoleInput)
        {
            var options = new string[] { "Y", "YES" };
            return options.Any(o => o.Equals(consoleInput, StringComparison.CurrentCultureIgnoreCase));
        }

        private void DisplayIntroText()
        {
            Console.WriteLine("TRAIN");
            Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            Console.WriteLine();
            Console.WriteLine("TIME - SPEED DISTANCE EXERCISE");
            Console.WriteLine();
        }
    }
}

```

# `91_Train/csharp/TrainTests/TrainGameTests.cs`



以上代码是用于测试TrainGame类的功能，其中包括：

1. `MinimumRandomNumber()`测试用例，验证game.GenerateRandomNumber(10, 10)的最小值是否大于等于10。
2. `MaximumRandomNumber()`测试用例，验证game.GenerateRandomNumber(10, 10)的最大值是否小于等于110。
3. `IsInputYesWhenY()`测试用例，验证TrainGame.IsInputYes("y")是否为真。
4. `IsInputYesWhenNotY()`测试用例，验证TrainGame.IsInputYes("a")是否为真。
5. `CarDurationTest()`测试用例，验证TrainGame.CalculateCarJourneyDuration(30, 1, 15)的结果是否为1。
6. `IsWithinAllowedDifference()`测试用例，验证TrainGame.IsWithinAllowedDifference(5, 5)的结果是否为真，即5到5之间的距离是否在允许的范围内。
7. `IsNotWithinAllowedDifference()`测试用例，验证TrainGame.IsWithinAllowedDifference(6, 5)的结果是否为假，即6到5之间的距离是否超出了允许的范围。


```
using Train;
using Xunit;

namespace TrainTests
{
    public class TrainGameTests
    {
        [Fact]
        public void MiniumRandomNumber()
        {
            TrainGame game = new TrainGame();
            Assert.True(game.GenerateRandomNumber(10, 10) >= 10);
        }

        [Fact]
        public void MaximumRandomNumber()
        {
            TrainGame game = new TrainGame();
            Assert.True(game.GenerateRandomNumber(10, 10) <= 110);
        }

        [Fact]
        public void IsInputYesWhenY()
        {
            Assert.True(TrainGame.IsInputYes("y"));
        }

        [Fact]
        public void IsInputYesWhenNotY()
        {
            Assert.False(TrainGame.IsInputYes("a"));
        }

        [Fact]
        public void CarDurationTest()
        {
            Assert.Equal(1, TrainGame.CalculateCarJourneyDuration(30, 1, 15) );
        }

        [Fact]
        public void IsWithinAllowedDifference()
        {
            Assert.True(TrainGame.IsWithinAllowedDifference(5,5));
        }


        [Fact]
        public void IsNotWithinAllowedDifference()
        {
            Assert.False(TrainGame.IsWithinAllowedDifference(6, 5));
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `91_Train/java/src/Train.java`

In this version of the code, the condition for the while loop has been modified to check for a "Y" or "yes" input instead of just a "PROBLEM". If a "Y" or "yes" input is entered, the game over variable is set to true, causing the while loop to exit.

However, I'm not sure if this is the desired behavior for the game. It's generally a good practice to provide clear and concise documentation about the purpose and effects of any changes made to a program.

In addition, there are a few issues with the code that I found:

1. missing dependency: The code assumes that the Keyboard is being used to receive input from the player. If this is not the case, you may need to modify the code to use a different input method.
2. null pointer: In the displayTextAndGetInput method, the variable kbScanner is being initialized to null. If kbScanner is null, the method will throw a java.util.NullPointerException.
3. checking for a null value: In the yesEntered method, the variable text is being compared to a null value. It should be checking if the text is equal to a string rather than a value.
4. formatting issue: In the println statement, the format string "%.2f" is being used to print the speed distance exercise. However, this format string is only available in Java 8 and later versions. If you are using a older version of Java, the format string may not work correctly.
5. unused variable: In the goodbye method, the variable train is being initialized to null. However, you are using the train variable in the main method. It would be better to initialize the train variable to null in the main method instead.


```
import java.util.Arrays;
import java.util.Scanner;

/**
 * Train
 * <p>
 * Based on the Basic program Train here
 * https://github.com/coding-horror/basic-computer-games/blob/main/91%20Train/train.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic program in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Train {

    private final Scanner kbScanner;

    public Train() {
        kbScanner = new Scanner(System.in);
    }

    public void process() {

        intro();

        boolean gameOver = false;

        do {
            double carMph = (int) (25 * Math.random() + 40);
            double hours = (int) (15 * Math.random() + 5);
            double train = (int) (19 * Math.random() + 20);

            System.out.println(" A CAR TRAVELING " + (int) carMph + " MPH CAN MAKE A CERTAIN TRIP IN");
            System.out.println((int) hours + " HOURS LESS THAN A TRAIN TRAVELING AT " + (int) train + " MPH.");

            double howLong = Double.parseDouble(displayTextAndGetInput("HOW LONG DOES THE TRIP TAKE BY CAR? "));

            double hoursAnswer = hours * train / (carMph - train);
            int percentage = (int) (Math.abs((hoursAnswer - howLong) * 100 / howLong) + .5);
            if (percentage > 5) {
                System.out.println("SORRY.  YOU WERE OFF BY " + percentage + " PERCENT.");
            } else {
                System.out.println("GOOD! ANSWER WITHIN " + percentage + " PERCENT.");
            }
            System.out.println("CORRECT ANSWER IS " + hoursAnswer + " HOURS.");

            System.out.println();
            if (!yesEntered(displayTextAndGetInput("ANOTHER PROBLEM (YES OR NO)? "))) {
                gameOver = true;
            }

        } while (!gameOver);


    }

    private void intro() {
        System.out.println("TRAIN");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("TIME - SPEED DISTANCE EXERCISE");
        System.out.println();
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
     * Program startup.
     *
     * @param args not used (from command line).
     */
    public static void main(String[] args) {
        Train train = new Train();
        train.process();
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


# `91_Train/javascript/train.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。

`print` 函数的作用是在文档中创建一个输出框（output 元素），然后将一个字符串（str）添加到该输出框中。

`input` 函数的作用是获取用户输入的字符串（inputStr），并将其存储在变量中。该函数通过创建一个输入元素（inputElement），然后将一个字符串（inputStr）和输入框（inputElement）的属性设置，将输入框的类型设置为文本，长度设置为 50。接着将创建好的输入元素（inputElement）添加到文档中的输出框（output）中，并将其聚焦（focus）。然后输入框（inputElement）的 `keydown` 事件监听器（event listener）监听用户按键，当事件监听器检测到按下的是回车键（13键）时，将输入框（inputElement）的值（inputStr）存储到变量中，并输出字符串（print）。

在该函数中，还定义了一个 Promise 类型的 resolve 函数，用于在用户点击回车键时，将输入框（inputElement）的值（inputStr）存储到变量中，并输出字符串（print）。该函数通过调用 `Promise` 的 `resolve` 方法，将输入框（inputElement）的值（inputStr）存储到变量中，并输出字符串（print）。


```
// TRAIN
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

这段代码是一个 JavaScript 函数，名为 `tab`，它的作用是打印出在给定速度下，一个汽车行驶与一个火车旅行所需时间的差异。

具体来说，这段代码执行以下操作：

1. 首先，在函数中定义了一个名为 `str` 的字符串变量，用于存储在给定速度下，汽车与火车分别行驶所需的时间。
2. 在 while 循环中，调用 `str += " "` 的方法，逐个在字符串末尾添加空格，使得字符串的格式为：在给定速度下，汽车与火车分别行驶所需时间的格式。
3. 调用 `return str` 方法，返回字符串变量 `str`，即汽车与火车分别行驶所需时间的字符串。

代码中并没有做任何错误检查，因此可能会存在一些问题，例如输入不是数字时，程序可能会出现错误。此外，如果输入的数字不是合理的速度，也会导致程序出现错误。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// Main control section
async function main()
{
    print(tab(33) + "TRAIN\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("TIME - SPEED DISTANCE EXERCISE\n");
    print("\n ");
    while (1) {
        c = Math.floor(25 * Math.random()) + 40;
        d = Math.floor(15 * Math.random()) + 5;
        t = Math.floor(19 * Math.random()) + 20;
        print(" A CAR TRAVELING " + c + " MPH CAN MAKE A CERTAIN TRIP IN\n");
        print(d + " HOURS LESS THAN A TRAIN TRAVELING AT " + t + " MPH.\n");
        print("HOW LONG DOES THE TRIP TAKE BY CAR");
        a = parseFloat(await input());
        v = d * t / (c - t);
        e = Math.floor(Math.abs((v - a) * 100 / a) + 0.5);
        if (e > 5) {
            print("SORRY.  YOU WERE OFF BY " + e + " PERCENT.\n");
        } else {
            print("GOOD! ANSWER WITHIN " + e + " PERCENT.\n");
        }
        print("CORRECT ANSWER IS " + v + " HOURS.\n");
        print("\n");
        print("ANOTHER PROBLEM (YES OR NO)\n");
        str = await input();
        print("\n");
        if (str.substr(0, 1) != "Y")
            break;
    }
}

```

这道题目要求解释以下代码的作用，不要输出源代码。从代码中可以看出，只有一个名为 main 的函数，且该函数没有参数输出，因此该函数的作用是整理解释并运行程序。


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


# `91_Train/python/train.py`

这段代码是一个Python脚本，主要作用是解释并输出一个基于BASIC语言的游戏。这个游戏可以让玩家在两种不同速度的交通工具之间做出选择，并输出在选择不同的交通工具时所需要的时间。

具体来说，这段代码实现了一个play_game函数，它接受一个参数，代表玩家输入的选择。然后通过使用random库函数，分别从高到低生成一个40到65之间的随机整数表示汽车的行驶速度，生成一个5到20之间的随机整数表示火车行驶的速度。

接着，这段代码输出一个类似于这样：
```
A car travelling 55.56 MPH can make a certain trip in 5.0 hours less than a train travelling at 34.99 MPH
A car travelling 55.56 MPH can make a certain trip in 5.0 hours less than a train travelling at 34.99 MPH
```
输出结果类似于上面的描述，描述了汽车和火车在不同速度下行驶相同距离所需的时间差异。

最后，通过一个while循环，让玩家输入所需的时间，然后计算出汽车行驶所需的时间，最后计算出错误百分比，并输出正确答案。


```
#!/usr/bin/env python3
# TRAIN
#
# Converted from BASIC to Python by Trevor Hobson

import random


def play_game() -> None:
    """Play one round of the game"""
    car_speed = random.randint(40, 65)
    time_difference = random.randint(5, 20)
    train_speed = random.randint(20, 39)
    print("\nA car travelling", car_speed, "MPH can make a certain trip in")
    print(time_difference, "hours less than a train travelling at", train_speed, "MPH")
    time_answer: float = 0
    while time_answer == 0:
        try:
            time_answer = float(input("How long does the trip take by car "))
        except ValueError:
            print("Please enter a number.")
    car_time = time_difference * train_speed / (car_speed - train_speed)
    error_percent = int(abs((car_time - time_answer) * 100 / time_answer) + 0.5)
    if error_percent > 5:
        print("Sorry. You were off by", error_percent, "percent.")
        print("Correct answer is", round(car_time, 6), "hours")
    else:
        print("Good! Answer within", error_percent, "percent.")


```

这段代码定义了一个名为 `main` 的函数，它返回一个 `None` 类型的值。

函数内部首先打印出 " " 33 个字符，再加一个 "TRAIN"字符，最后打印出 " " 15 个字符，以及 "MORRISTOWN, NEW JERSEY" 的字符。接着打印 "Time - speed distance exercise" 字符串。

然后定义了一个名为 `keep_playing` 的布尔变量，并将其初始化为 `True`。

进入一个无限循环，每次循环函数内部先调用 `play_game` 函数，然后调用 `keep_playing` 函数，让它判断用户是否输入了 "y"。如果是，则退出循环，否则继续执行下一次调用 `play_game` 函数的操作。

最后在程序外部，调用 `main` 函数，并传入参数 `None`，表示函数可以接受一个参数，但实际上不需要使用这个参数。


```
def main() -> None:
    print(" " * 33 + "TRAIN")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")
    print("Time - speed distance exercise")

    keep_playing = True
    while keep_playing:
        play_game()
        keep_playing = input("\nAnother problem (yes or no) ").lower().startswith("y")


if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by Anthony Rubick [AnthonyMichaelTDM](https://github.com/AnthonyMichaelTDM)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Trap

This is another in the family of “guess the mystery number” games. In TRAP the computer selects a random number between 1 and 100 (or other limit set). Your object is to find the number. On each guess, you enter 2 numbers trying to trap the mystery number between your two trap numbers. The computer will tell you if you have trapped the number.

To win the game, you must guess the mystery number by entering it as the same value for both of your trap numbers. You get 6 guesses (this should be changed if you change the guessing limit).

After you have played GUESS, STARS, and TRAP, compare the guessing strategy you have found best for each game. Do you notice any similarities? What are the differences? Can you write a new guessing game with still another approach?

TRAP was suggested by a 10-year-old when he was playing GUESS. It was originally programmed by Steve Ullman and extensively modified into its final form by Bob Albrecht of People’s Computer Co.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=176)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=191)


Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `92_Trap/csharp/Program.cs`

这段代码是一个基于Trap算法的抓猜游戏。它由一个位于纽约州 Morristown的软件开发组织开发。该程序通过使用随机数生成器来随机生成一个1到100之间的整数，然后让玩家进行猜测，直到最多6次猜测为止。

程序的主要功能如下：

1. 输出游戏的陷阱信息，包括 "TRAP" 和 "CREATIVE COMPUTING"。
2. 输出游戏的提示信息，例如 "MY NUMBER IS LARGER THAN YOUR TRAP NUMBERS" 或 "MY NUMBER IS SMALLER THAN YOUR TRAP NUMBERS"。
3. 通过玩家输入的数字，程序会生成一个随机数，并让玩家进行猜测。根据玩家的猜测，程序会输出 "MY NUMBER IS LARGER THAN YOUR TRAP NUMBERS" 或 "MY NUMBER IS SMALLER THAN YOUR TRAP NUMBERS"。如果程序生成的数字与玩家输入的数字相同，那么程序会输出 "YOU GOT IT！"。
4. 如果程序生成的数字比玩家输入的数字大，那么程序会输出 "MY NUMBER IS LARGER THAN YOUR TRAP NUMBERS"。如果程序生成的数字比玩家输入的数字小，那么程序会输出 "MY NUMBER IS SMALLER THAN YOUR TRAP NUMBERS"。如果程序生成的数字与玩家输入的数字相同，那么程序会输出 "YOU HAD YOUR NUMBER PAIRED WITH ANOTHER TRAP"。
5. 如果程序在6次猜测之后仍然无法猜出正确答案，那么程序会输出 "SORTS OF MADNESS"，然后停止游戏。

程序的主要目的是让玩家通过猜测，尽可能快地找到正确答案。通过使用随机数生成器，程序保证了每个猜测都是独立的，并且可以让玩家在猜测过程中学习到游戏策略，进一步提高猜中正确答案的概率。


```
﻿using System;

namespace trap_cs
{
  class Program
  {
    const int maxGuesses = 6;
    const int maxNumber = 100;
    static void Main(string[] args)
    {
      int lowGuess  = 0;
      int highGuess = 0;

      Random randomNumberGenerator = new ();

      Print("TRAP");
      Print("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
      Print();
      Print();
      Print();

      PrintInstructions();

      int numberToGuess = randomNumberGenerator.Next(1, maxNumber);

      for (int nGuess = 1; nGuess <= maxGuesses + 1; nGuess++)
      {
        if (nGuess > maxGuesses)
        {
          Print(string.Format("SORRY, THAT'S {0} GUESSES. THE NUMBER WAS {1}", maxGuesses, numberToGuess));
          Print();
          break;
        }

        GetGuesses(nGuess, ref lowGuess, ref highGuess);

        if(lowGuess == highGuess && lowGuess == numberToGuess)
        {
          Print("YOU GOT IT!!!");
          Print();
          Print("TRY AGAIN.");
          Print();
          break;
        }
        if (highGuess < numberToGuess)
        {
          Print("MY NUMBER IS LARGER THAN YOUR TRAP NUMBERS.");
        }
        else if (lowGuess > numberToGuess)
        {
          Print("MY NUMBER IS SMALLER THAN YOUR TRAP NUMBERS.");
        }
        else
        {
          Print("YOU HAVE TRAPPED MY NUMBER.");
        }
      }
    }

```

This is a class written in C# that simulates a game of猜数字。 The class has several methods:

* `Initialize`: This method is called when the game is first started. It sets the initial values of the low and high guesses to 0, and the maximum number of guesses to 5.
* `Solve`: This method takes a guess for each of the two numbers and tries to determine if the guess is correct, whether the first number is larger than or smaller than the second number, and finally updates the values of the low and high guesses. If the guess is correct, the low guess is updated to be the higher guess. If the guess is incorrect, the loop continues.
* `Print`: This method prints a message to the console.
* `GetGuesses`: This method takes the number of guesses and prints them to the console.
* `GetIntegerFromConsole`: This method takes a prompt message from the console and returns the user's response as an integer.
* `private int maxGuesses = 5;`
* `private int lowGuess;`
* `private int highGuess;`
* `private int maxNumber;`
* `private int minGuess;`
* `private int trapNumber;`

This class could be further extended to include more methods, such as checking for duplicate guesses or giving the user feedback after each guess.


```
// TRAP
// REM - STEVE ULLMAN, 8 - 1 - 72
    static void PrintInstructions()
    {
      Print("INSTRUCTIONS ?");

      char response = Console.ReadKey().KeyChar;
      if (response == 'Y')
      {
        Print(string.Format("I AM THINKING OF A NUMBER BETWEEN 1 AND {0}", maxNumber));
        Print("TRY TO GUESS MY NUMBER. ON EACH GUESS,");
        Print("YOU ARE TO ENTER 2 NUMBERS, TRYING TO TRAP");
        Print("MY NUMBER BETWEEN THE TWO NUMBERS. I WILL");
        Print("TELL YOU IF YOU HAVE TRAPPED MY NUMBER, IF MY");
        Print("NUMBER IS LARGER THAN YOUR TWO NUMBERS, OR IF");
        Print("MY NUMBER IS SMALLER THAN YOUR TWO NUMBERS.");
        Print("IF YOU WANT TO GUESS ONE SINGLE NUMBER, TYPE");
        Print("YOUR GUESS FOR BOTH YOUR TRAP NUMBERS.");
        Print(string.Format("YOU GET {0} GUESSES TO GET MY NUMBER.", maxGuesses));
      }
    }
    static void Print(string stringToPrint)
    {
      Console.WriteLine(stringToPrint);
    }
    static void Print()
    {
      Console.WriteLine();
    }
    static void GetGuesses(int nGuess, ref int lowGuess, ref int highGuess)
    {
      Print();
      Print(string.Format("GUESS #{0}", nGuess));

      lowGuess  = GetIntFromConsole("Type low guess");
      highGuess = GetIntFromConsole("Type high guess");

      if(lowGuess > highGuess)
      {
        int tempGuess = lowGuess;

        lowGuess = highGuess;
        highGuess = tempGuess;
      }
    }
    static int GetIntFromConsole(string prompt)
    {

      Console.Write( prompt + " > ");
      string intAsString = Console.ReadLine();

      if(int.TryParse(intAsString, out int intValue) ==false)
      {
        intValue = 1;
      }

      return intValue;
    }
  }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `92_Trap/java/src/Trap.java`

This is a Java class that uses multiple Gradle features to generate a random number.

First, it imports the required packages, which include:

* ` java.util.Arrays` for handling arrays
* ` java.util.Collections` for enhancing the `Collections` class
* ` java.util.Comparator` for implementing the `Comparator` interface
* ` java.util.HashMap` for storing the random numbers
* ` java.util.LinkedList` for storing the random numbers
* ` java.util.Map` for storing the random numbers
* ` java.util.Random` for generating random numbers
* ` java.util.Scanner` for reading input from the keyboard

The class then defines the required methods:

* `kbScanner.next()` generates a random number within the specified range (0-256).
* `Arrays.stream(values).anyMatch(str -> str.equalsIgnoreCase(text))` checks whether the entered string matches one of the expected strings.
* `displayTextAndGetInput()` displays a message on the screen and then accepts input from the keyboard. It returns what was typed by the player.
* `randomNumber()` generates a random single digit number.
* `generateRandomNumber()` generates a random single digit number within the specified range (0-256).


```
import java.util.Arrays;
import java.util.Scanner;

/**
 * Game of Trap
 * <p>
 * Based on the Basic game of Trap here
 * https://github.com/coding-horror/basic-computer-games/blob/main/92%20Trap/trap.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Trap {

    public static final int HIGH_NUMBER_RANGE = 100;
    public static final int MAX_GUESSES = 6;

    private enum GAME_STATE {
        STARTING,
        START_GAME,
        GUESSING,
        PLAY_AGAIN,
        GAME_OVER
    }

    // Used for keyboard input
    private final Scanner kbScanner;

    // Current game state
    private GAME_STATE gameState;

    // Players guess count;
    private int currentPlayersGuess;

    // Computers random number
    private int computersNumber;

    public Trap() {

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

                // Show an introduction and optional instructions the first time the game is played.
                case STARTING:
                    intro();
                    if (yesEntered(displayTextAndGetInput("INSTRUCTIONS? "))) {
                        instructions();
                    }
                    gameState = GAME_STATE.START_GAME;
                    break;

                // Start new game
                case START_GAME:
                    computersNumber = randomNumber();
                    currentPlayersGuess = 1;
                    gameState = GAME_STATE.GUESSING;
                    break;

                // Player guesses the number until they get it or run out of guesses
                case GUESSING:
                    System.out.println();
                    String playerRangeGuess = displayTextAndGetInput("GUESS # " + currentPlayersGuess + "? ");
                    int startRange = getDelimitedValue(playerRangeGuess, 0);
                    int endRange = getDelimitedValue(playerRangeGuess, 1);

                    // Has the player won?
                    if (startRange == computersNumber && endRange == computersNumber) {
                        System.out.println("YOU GOT IT!!!");
                        System.out.println();
                        gameState = GAME_STATE.PLAY_AGAIN;
                    } else {
                        // show where the guess is at
                        System.out.println(showGuessResult(startRange, endRange));
                        currentPlayersGuess++;
                        if (currentPlayersGuess > MAX_GUESSES) {
                            System.out.println("SORRY, THAT'S " + MAX_GUESSES + " GUESSES. THE NUMBER WAS "
                                    + computersNumber);
                            gameState = GAME_STATE.PLAY_AGAIN;
                        }
                    }
                    break;

                // Play again, or exit game?
                case PLAY_AGAIN:
                    System.out.println("TRY AGAIN");
                    gameState = GAME_STATE.START_GAME;
            }
        } while (gameState != GAME_STATE.GAME_OVER);
    }

    /**
     * Show the players guess result
     *
     * @param start start range entered by player
     * @param end   end range
     * @return text to indicate their progress.
     */
    private String showGuessResult(int start, int end) {

        String status;
        if (start <= computersNumber && computersNumber <= end) {
            status = "YOU HAVE TRAPPED MY NUMBER.";
        } else if (computersNumber < start) {
            status = "MY NUMBER IS SMALLER THAN YOUR TRAP NUMBERS.";
        } else {
            status = "MY NUMBER IS LARGER THAN YOUR TRAP NUMBERS.";
        }

        return status;
    }

    private void instructions() {
        System.out.println("I AM THINKING OF A NUMBER BETWEEN 1 AND " + HIGH_NUMBER_RANGE);
        System.out.println("TRY TO GUESS MY NUMBER. ON EACH GUESS,");
        System.out.println("YOU ARE TO ENTER 2 NUMBERS, TRYING TO TRAP");
        System.out.println("MY NUMBER BETWEEN THE TWO NUMBERS. I WILL");
        System.out.println("TELL YOU IF YOU HAVE TRAPPED MY NUMBER, IF MY");
        System.out.println("NUMBER IS LARGER THAN YOUR TWO NUMBERS, OR IF");
        System.out.println("MY NUMBER IS SMALLER THAN YOUR TWO NUMBERS.");
        System.out.println("IF YOU WANT TO GUESS ONE SINGLE NUMBER, TYPE");
        System.out.println("YOUR GUESS FOR BOTH YOUR TRAP NUMBERS.");
        System.out.println("YOU GET " + MAX_GUESSES + " GUESSES TO GET MY NUMBER.");
    }

    private void intro() {
        System.out.println("TRAP");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println();
    }

    /**
     * Accepts a string delimited by comma's and returns the nth delimited
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
     * Generate random number
     * Used as a single digit of the computer player
     *
     * @return random number
     */
    private int randomNumber() {
        return (int) (Math.random()
                * (HIGH_NUMBER_RANGE) + 1);
    }
}

```

# `92_Trap/java/src/TrapGame.java`

这段代码定义了一个名为 TrapGame 的类，其中包含一个名为 main 的方法。在 main 方法中，使用 new 关键字创建了一个 Trap 类的实例，并调用该实例的 play 方法。

Trap 类是一个类，其中包含一个名为 trap 的实例变量。traps 变量被声明为 public，这意味着该变量可以被任何类访问。此外，traps 变量被声明为 final，这意味着无论从哪个类继承，traps 变量始终指向同一个对象。

play() 方法是一个名为 trap 的方法，其中包含游戏玩法中 trap 的行为。由于没有提供具体的 trap 行为，因此无法确定 trap 类中包含什么。但是，通常情况下，traps 类会被设计为捕获和响应游戏中的陷阱，例如让游戏者失去生命值或者让游戏无法继续进行下一步。


```
public class TrapGame {

    public static void main(String[] args) {

        Trap trap = new Trap();
        trap.play();
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


# `92_Trap/javascript/trap.js`

这是一个 JavaScript 代码片段，它实现了将文本输入框中的输入转换为字符串并将其打印到页面上，同时在输入框中添加了一个键盘事件监听器，以便在按下回车键时将输入的字符串打印到页面上。

具体来说，代码中的 `input` 函数接收一个输入元素（即文本输入框）和一个字符串作为参数，返回一个 Promise 对象。在 Promise 对象中，有一个 `resolve` 方法和一个 `reject` 方法，分别用于将输入的字符串和错误信息返回给调用者。

在 `input` 函数中，首先创建了一个新的输入元素，设置其样式，并将其添加到页面上。然后，使用 `document.getElementById("output").appendChild(document.createTextNode(str))` 将输入的字符串打印到页面上。

接着，在输入框中添加了一个键盘事件监听器，监听 `keydown` 事件。当输入框中的字符被按下回车键时，监听器会收到一个事件，检查事件类型是否为 `keydown` 事件。如果是，则获取输入框中的字符串，并将其打印到页面上。同时，在页面上打印一个换行符，以便新输入的信息能够正确显示在页面上。

最后，在 `input` 函数中，使用 `return` 返回一个 Promise 对象，并使用 `Promise.all` 方法等待输入字符串和打印操作的完成。


```
// TRAP
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

As an AI language model, I cannot guess your number. However, I can provide you with some tips to try when trying to guess a number.

1. Try to identify a pattern between the two trap numbers.
2. Make a range of possible numbers for the trap numbers.
3. Try to guess a number that is significantly larger or smaller than the trap numbers.
4. If you think the trap numbers are too close to guess, try guessing a number that is interpolated between the two trap numbers.

Remember, there is no guarantee to guessing the number, but by following these tips, you can increase your chances.


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// Main control section
async function main()
{
    print(tab(34) + "TRAP\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    g = 6;
    n = 100;
    // Trap
    // Steve Ullman, Aug/01/1972
    print("INSTRUCTIONS");
    str = await input();
    if (str.substr(0, 1) == "Y") {
        print("I AM THINKING OF A NUMBER BETWEEN 1 AND " + n + "\n");
        print("TRY TO GUESS MY NUMBER. ON EACH GUESS,\n");
        print("YOU ARE TO ENTER 2 NUMBERS, TRYING TO TRAP\n");
        print("MY NUMBER BETWEEN THE TWO NUMBERS. I WILL\n");
        print("TELL YOU IF YOU HAVE TRAPPED MY NUMBER, IF MY\n");
        print("NUMBER IS LARGER THAN YOUR TWO NUMBERS, OR IF\n");
        print("MY NUMBER IS SMALLER THAN YOUR TWO NUMBERS.\n");
        print("IF YOU WANT TO GUESS ONE SINGLE NUMBER, TYPE\n");
        print("YOUR GUESS FOR BOTH YOUR TRAP NUMBERS.\n");
        print("YOU GET " + g + " GUESSES TO GET MY NUMBER.\n");
    }
    while (1) {
        x = Math.floor(n * Math.random()) + 1;
        for (q = 1; q <= g; q++) {
            print("\n");
            print("GUESS #" + q + " ");
            str = await input();
            a = parseInt(str);
            b = parseInt(str.substr(str.indexOf(",") + 1));
            if (a == b && x == a) {
                print("YOU GOT IT!!!\n");
                break;
            }
            if (a > b) {
                r = a;
                a = b;
                b = r;
            }
            if (a <= x && x <= b) {
                print("YOU HAVE TRAPPED MY NUMBER.\n");
            } else if (x >= a) {
                print("MY NUMBER IS LARGER THAN YOUR TRAP NUMBERS.\n");
            } else {
                print("MY NUMBER IS SMALLER THAN YOUR TRAP NUMBERS.\n");
            }
        }
        print("\n");
        print("TRY AGAIN.\n");
        print("\n");
    }
}

```

这是经典的 "Hello, World!" 程序，用于在 Unix 和类 Unix 的操作系统中启动一个新程序。

具体来说，这个程序会读取一个从标准输入(通常是键盘)输入一个或多个字符，并输出一个短的 "Hello, World!" 消息来欢迎用户。

例如，如果你在终端中运行这个程序，它可能会输出类似这样的消息：

```
$ main
Hello, World!
``` 

如果你在一个可读写文件中运行这个程序，它可能会读取文件中的内容并输出到文件末尾。

总的来说，这个程序是一个非常重要的工具，几乎每个程序员都会在他们的程序中使用它来开始他们的程序。


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


# `92_Trap/python/trap.py`

这段代码是一个Python program，名为“TRAP”，功能是通过玩一个数独游戏来吸引用户的注意。数独游戏是一种有趣的智力游戏，通过在空白方格中填写数字，让数独变成一个完整的数字，通过解决每一道数字，让游戏更具挑战性。

这段代码的作用是：

1. 导入 random 库，用于生成随机数。
2. 定义了一个名为 play_game 的函数，用于执行数独游戏。
3. 在 play_game 函数中，引入了之前的功能。
4. 通过 random.randint(1, number_max) 生成一个 1 到 number_max 之间的随机整数，得到一个数字。
5. 通过 turn 计数器记录当前的轮数，每次增加 1，达到猜测最大值时，游戏结束。
6. 循环处理用户的输入，计算两个数字之和是否等于 2。
7. 如果相等，就得到了一个正确的猜测，输出轮数并跳出循环。
8. 如果一个数字小于另一个数字，则输出提示信息。
9. 如果一个数字大于另一个数字，则输出提示信息。
10. 如果轮数达到猜测最大值，游戏结束并输出提示信息。




```
#!/usr/bin/env python3
# TRAP
#
# STEVE ULLMAN, 8-1-72
# Converted from BASIC to Python by Trevor Hobson

import random

number_max = 100
guess_max = 6


def play_game() -> None:
    """Play one round of the game"""

    number_computer = random.randint(1, number_max)
    turn = 0
    while True:
        turn += 1
        user_guess = [-1, -1]
        while user_guess == [-1, -1]:
            try:
                user_input = [
                    int(item)
                    for item in input("\nGuess # " + str(turn) + " ? ").split(",")
                ]
                if len(user_input) == 2:
                    if sum(1 < x < number_max for x in user_input) == 2:
                        user_guess = user_input
                    else:
                        raise ValueError
                else:
                    raise ValueError
            except (ValueError, IndexError):
                print("Please enter a valid guess.")
        if user_guess[0] > user_guess[1]:
            user_guess[0], user_guess[1] = user_guess[1], user_guess[0]
        if user_guess[0] == user_guess[1] == number_computer:
            print("You got it!!!")
            break
        elif user_guess[0] <= number_computer <= user_guess[1]:
            print("You have trapped my number.")
        elif number_computer < user_guess[0]:
            print("My number is smaller than your trap numbers.")
        else:
            print("My number is larger than your trap numbers.")
        if turn == guess_max:
            print("That's", turn, "guesses. The number was", number_computer)
            break


```

这段代码是一个Python程序，主要目的是让玩家猜数字并 trap 出数字。程序内部分为两个主要部分：

1. 打印一些装饰性的文本，包括一个有空格 34 行和一些字母组成的有空格 15 行。
2. 如果用户输入 "y"，程序将进入一个猜测数字的游戏。程序将随机生成一个数字，并提示用户猜测两个数字，尝试 trap 出目标数字。程序将在猜测失败时提示用户，并给出其猜测的最大猜测次数。
3. 如果用户在猜测过程中成功 trap 出了目标数字，程序会输出一条消息并继续游戏。
4. 如果用户在猜测过程中未能 trap 出目标数字，程序将提示用户重新猜测，并继续游戏。
5. 程序将一直保持猜测状态，直到用户明确表示不想继续游戏为止。

总的来说，这段代码的主要目的是让用户在给定的次数内猜出两个数字，并 trap 出其中的一个数字。


```
def main() -> None:
    print(" " * 34 + "TRAP")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")
    if input("Instructions ").lower().startswith("y"):
        print("\nI am thinking of a number between 1 and", number_max)
        print("try to guess my number. On each guess,")
        print("you are to enter 2 numbers, trying to trap")
        print("my number between the two numbers. I will")
        print("tell you if you have trapped my number, if my")
        print("number is larger than your two numbers, or if")
        print("my number is smaller than your two numbers.")
        print("If you want to guess one single number, type")
        print("your guess for both your trap numbers.")
        print("You get", guess_max, "guesses to get my number.")

    keep_playing = True
    while keep_playing:
        play_game()
        keep_playing = input("\nTry again. ").lower().startswith("y")


```

这段代码是一个Python程序中的一个if语句。if语句可以用于在程序运行时检查文件是否符合某些特定的条件，如果条件为真，则执行if语句内部的代码块。

在这段代码中，if __name__ == "__main__"：是一个特殊的作用域，用于确保当程序是作为独立的主机脚本运行时，而不是作为操作系统交互式 shell 运行时，如果当前目录中包含该脚本，则执行if语句内部的代码块。 "__main__"是一个特殊常量，用于判断当前目录是否包含程序的主入口文件，即程序的入口点。

因此，如果当前目录中包含该脚本，则会执行if语句内部的代码块，否则程序将不会执行if语句内部的代码块，也不会创建一个名为 "__main__" 的常量。


```
if __name__ == "__main__":
    main()

```