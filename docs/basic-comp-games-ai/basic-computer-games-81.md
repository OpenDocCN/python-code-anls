# BasicComputerGames源码解析 81

# `86_Target/csharp/RandomExtensions.cs`

这段代码是一个静态类，名为 "RandomExtensions"，旨在提供一些用于生成随机数的实用工具。

内部 static class RandomExtensions 表示该类是一个自定义的基类，可以从现在定义的继承类中使用静态内部类来扩展方法。

NextPosition 方法是一个静态方法，它使用角度和旋转来生成一个点的位置，这个点的位置可以是目标的中心点。这个方法使用了 Games.Common.Randomness 命名空间中提供的 IRandom 类型来生成随机数。

这个方法的作用是生成一个实数类型的随机点，坐标为 [目标中心， 目标中心， 1] ，其中 1 表示该点在目标中心的垂直距离。生成的随机点的位置是使用 angle 和 rotation 函数来计算的，这些函数使用的是 rnd.NextFloat() 方法来生成一个介于 0 和 1 之间的浮点数，再乘以 360/240 得到一个旋转角度，单位是弧度。接着，这个角度再乘以 0.2，就可以得到一个水平方向的偏移量，单位是像素。最后，生成的随机点的位置就是 [水平偏移量， 垂直偏移量， 1] 。

这个方法可以用于许多生成随机数的场景，比如在游戏中生成玩家位置的时候。


```
using Games.Common.Randomness;

namespace Target
{
    internal static class RandomExtensions
    {
        public static Point NextPosition(this IRandom rnd) => new (
            Angle.InRotations(rnd.NextFloat()),
            Angle.InRotations(rnd.NextFloat()),
            100000 * rnd.NextFloat() + rnd.NextFloat());
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `86_Target/java/Target.java`

This is a Java program that simulates a virtual reality game where the player is given a task to shoot at a target. The program takes input from the user to determine the position and orientation of the target and the distance between the target and the player.

The program first defines a `TargetAttempt` class that represents the data structure holding the target information. This class has four instance variables: `xDeviation`, `zDeviation`, `distance`, and `validInput`. The `xDeviation` and `zDeviation` variables represent the position of the target in the x and z axes, respectively. The `distance` variable represents the distance between the target and the player, and the `validInput` variable is a boolean flag that indicates whether the input is valid.

The program then defines a `Shooter` class that extends the `TargetAttempt` class. This class has a method called `shoot()` that calculates the position and distance of the target and returns the `TargetAttempt` object.

The main method of the program is responsible for initializing the game and simulating the shooting process. It first calls the `printIntro()` method to display the game introduction screen. Then it calls the `shoot()` method for the `Shooter` object to simulate the shooting process.

The `printIntro()` method displays the game introduction screen with some information about the game. The `shoot()` method then simulates the shooting process by calculating the position and distance of the target and returning the `TargetAttempt` object.

Note that this program assumes a 3D coordinate system and that the x, y, and z axes are 0.1 degree apart. This can be changed by modifying the constant values in the program.


```
import java.util.Scanner;

/**
 * TARGET
 * <p>
 * Converted from BASIC to Java by Aldrin Misquitta (@aldrinm)
 */
public class Target {

	private static final double RADIAN = 180 / Math.PI;

	public static void main(String[] args) {
		Scanner scan = new Scanner(System.in);

		printIntro();

		//continue till the user aborts
		while (true) {
			int numberShots = 0;

			final double xAxisInRadians = Math.random() * 2 * Math.PI;
			final double yAxisInRadians = Math.random() * 2 * Math.PI;
			System.out.printf("RADIANS FROM X AXIS = %.7f     FROM Z AXIS = %.7f\n", xAxisInRadians, yAxisInRadians);

			final double p1 = 100000 * Math.random() + Math.random();
			final double x = Math.sin(yAxisInRadians) * Math.cos(xAxisInRadians) * p1;
			final double y = Math.sin(yAxisInRadians) * Math.sin(xAxisInRadians) * p1;
			final double z = Math.cos(yAxisInRadians) * p1;
			System.out.printf("TARGET SIGHTED: APPROXIMATE COORDINATES:  X=%.3f  Y=%.3f  Z=%.3f\n", x, y, z);
			boolean targetOrSelfDestroyed = false;
			while (!targetOrSelfDestroyed) {
				numberShots++;
				int estimatedDistance = 0;
				switch (numberShots) {
					case 1:
						estimatedDistance = (int) (p1 * .05) * 20;
						break;
					case 2:
						estimatedDistance = (int) (p1 * .1) * 10;
						break;
					case 3:
						estimatedDistance = (int) (p1 * .5) * 2;
						break;
					case 4:
					case 5:
						estimatedDistance = (int) (p1);
						break;
				}

				System.out.printf("     ESTIMATED DISTANCE: %s\n\n", estimatedDistance);

				final TargetAttempt targetAttempt = readInput(scan);
				if (targetAttempt.distance < 20) {
					System.out.println("YOU BLEW YOURSELF UP!!");
					targetOrSelfDestroyed = true;
				} else {
					final double a1 = targetAttempt.xDeviation / RADIAN;
					final double b1 = targetAttempt.zDeviation / RADIAN;
					System.out.printf("RADIANS FROM X AXIS = %.7f  FROM Z AXIS = %.7f\n", a1, b1);

					final double x1 = targetAttempt.distance * Math.sin(b1) * Math.cos(a1);
					final double y1 = targetAttempt.distance * Math.sin(b1) * Math.sin(a1);
					final double z1 = targetAttempt.distance * Math.cos(b1);

					double distance = Math.sqrt((x1 - x) * (x1 - x) + (y1 - y) * (y1 - y) + (z1 - z) * (z1 - z));
					if (distance > 20) {
						double X2 = x1 - x;
						double Y2 = y1 - y;
						double Z2 = z1 - z;
						if (X2 < 0) {
							System.out.printf("SHOT BEHIND TARGET %.7f KILOMETERS.\n", -X2);
						} else {
							System.out.printf("SHOT IN FRONT OF TARGET %.7f KILOMETERS.\n", X2);
						}
						if (Y2 < 0) {
							System.out.printf("SHOT TO RIGHT OF TARGET %.7f KILOMETERS.\n", -Y2);
						} else {
							System.out.printf("SHOT TO LEFT OF TARGET %.7f KILOMETERS.\n", Y2);
						}
						if (Z2 < 0) {
							System.out.printf("SHOT BELOW TARGET %.7f KILOMETERS.\n", -Z2);
						} else {
							System.out.printf("SHOT ABOVE TARGET %.7f KILOMETERS.\n", Z2);
						}
						System.out.printf("APPROX POSITION OF EXPLOSION:  X=%.7f   Y=%.7f   Z=%.7f\n", x1, y1, z1);
						System.out.printf("     DISTANCE FROM TARGET =%.7f\n\n\n\n", distance);
					} else {
						System.out.println(" * * * HIT * * *   TARGET IS NON-FUNCTIONAL");
						System.out.printf("DISTANCE OF EXPLOSION FROM TARGET WAS %.5f KILOMETERS.\n", distance);
						System.out.printf("MISSION ACCOMPLISHED IN %s SHOTS.\n", numberShots);
						targetOrSelfDestroyed = true;
					}
				}
			}
			System.out.println("\n\n\n\n\nNEXT TARGET...\n");
		}
	}

	private static TargetAttempt readInput(Scanner scan) {
		System.out.println("INPUT ANGLE DEVIATION FROM X, DEVIATION FROM Z, DISTANCE ");
		boolean validInput = false;
		TargetAttempt targetAttempt = new TargetAttempt();
		while (!validInput) {
			String input = scan.nextLine();
			final String[] split = input.split(",");
			try {
				targetAttempt.xDeviation = Float.parseFloat(split[0]);
				targetAttempt.zDeviation = Float.parseFloat(split[1]);
				targetAttempt.distance = Float.parseFloat(split[2]);
				validInput = true;
			} catch (NumberFormatException nfe) {
				System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE\n? ");
			}

		}
		return targetAttempt;
	}

	private static void printIntro() {
		System.out.println("                                TARGET");
		System.out.println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
		System.out.println("\n\n");
		System.out.println("YOU ARE THE WEAPONS OFFICER ON THE STARSHIP ENTERPRISE");
		System.out.println("AND THIS IS A TEST TO SEE HOW ACCURATE A SHOT YOU");
		System.out.println("ARE IN A THREE-DIMENSIONAL RANGE.  YOU WILL BE TOLD");
		System.out.println("THE RADIAN OFFSET FOR THE X AND Z AXES, THE LOCATION");
		System.out.println("OF THE TARGET IN THREE DIMENSIONAL RECTANGULAR COORDINATES,");
		System.out.println("THE APPROXIMATE NUMBER OF DEGREES FROM THE X AND Z");
		System.out.println("AXES, AND THE APPROXIMATE DISTANCE TO THE TARGET.");
		System.out.println("YOU WILL THEN PROCEED TO SHOOT AT THE TARGET UNTIL IT IS");
		System.out.println("DESTROYED!");
		System.out.println("\nGOOD LUCK!!\n\n");
	}

	/**
	 * Represents the user input
	 */
	private static class TargetAttempt {

		double xDeviation;
		double zDeviation;
		double distance;
	}
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


# `86_Target/javascript/target.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。

1. `print` 函数的作用是在文档中的一个元素（此处为 `output` 元素）中添加字符串。这个字符串由 `input` 函数获取的用户输入组成。

2. `input` 函数的作用是获取用户输入的字符串，并在输入框中添加用户输入的字符串。它还监听输入框中的 `keydown` 事件，当用户按下回车键时，将获取到的字符串打印到控制台。

总之，这两个函数共同构成了一个简单的用户输入/输出交互应用。


```
// TARGET
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

It looks like you are trying to calculate the distance of an explosion from a target, based on the position of the explosion and the distance between the explosion and the target. The target is assumed to be a non-functional value, which means its position cannot be used to calculate the damage of the explosion.

The code appears to use a fixed number of shots (5) to calculate the damage of the explosion, based on the distance between the explosion and the target. The distance of the explosion from the target is first calculated, and then used as a factor in the calculation of the damage.

If the distance between the explosion and the target is less than or equal to 20 kilometers, the code prints a message and the distance of the explosion from the target. Otherwise, it prints a message indicating that the mission was not successful and then calculates the position of the explosion and prints the distance of the explosion from the target.

It is not clear from the code how the damage of the explosion is calculated, or how the shots are fired. It is also not clear what the variable "r" represents.


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
    print(tab(33) + "TARGET\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    r = 0;  // 1 in original
    r1 = 57.296;
    p = Math.PI;
    print("YOU ARE THE WEAPONS OFFICER ON THE STARSHIP ENTERPRISE\n");
    print("AND THIS IS A TEST TO SEE HOW ACCURATE A SHOT YOU\n");
    print("ARE IN A THREE-DIMENSIONAL RANGE.  YOU WILL BE TOLD\n");
    print("THE RADIAN OFFSET FOR THE X AND Z AXES, THE LOCATION\n");
    print("OF THE TARGET IN THREE DIMENSIONAL RECTANGULAR COORDINATES,\n");
    print("THE APPROXIMATE NUMBER OF DEGREES FROM THE X AND Z\n");
    print("AXES, AND THE APPROXIMATE DISTANCE TO THE TARGET.\n");
    print("YOU WILL THEN PROCEEED TO SHOOT AT THE TARGET UNTIL IT IS\n");
    print("DESTROYED!\n");
    print("\n");
    print("GOOD LUCK!!\n");
    print("\n");
    print("\n");
    while (1) {
        a = Math.random() * 2 * p;
        b = Math.random() * 2 * p;
        q = Math.floor(a * r1);
        w = Math.floor(b * r1);
        print("RADIANS FROM X AXIS = " + a + "   FROM Z AXIS = " + b + "\n");
        p1 = 100000 * Math.random() + Math.random();
        x = Math.sin(b) * Math.cos(a) * p1;
        y = Math.sin(b) * Math.sin(a) * p1;
        z = Math.cos(b) * p1;
        print("TARGET SIGHTED: APPROXIMATE COORDINATES:  X=" + x + "  Y=" + y + "  Z=" + z + "\n");
        while (1) {
            r++;
            switch (r) {
                case 1:
                    p3 = Math.floor(p1 * 0.05) * 20;
                    break;
                case 2:
                    p3 = Math.floor(p1 * 0.1) * 10;
                    break;
                case 3:
                    p3 = Math.floor(p1 * 0.5) * 2;
                    break;
                case 4:
                    p3 = Math.floor(p1);
                    break;
                case 5:
                    p3 = p1;
                    break;
            }
            print("     ESTIMATED DISTANCE: " + p3 + "\n");
            print("\n");
            print("INPUT ANGLE DEVIATION FROM X, DEVIATION FROM Z, DISTANCE");
            str = await input();
            a1 = parseInt(str);
            b1 = parseInt(str.substr(str.indexOf(",") + 1));
            p2 = parseInt(str.substr(str.lastIndexOf(",") + 1));
            print("\n");
            if (p2 < 20) {
                print("YOU BLEW YOURSELF UP!!\n");
                break;
            }
            a1 /= r1;
            b1 /= r1;
            print("RADIANS FROM X AXIS = " + a1 + "  ");
            print("FROM Z AXIS = " + b1 + "\n");
            x1 = p2 * Math.sin(b1) * Math.cos(a1);
            y1 = p2 * Math.sin(b1) * Math.sin(a1);
            z1 = p2 * Math.cos(b1);
            d = Math.sqrt((x1 - x) * (x1 - x) + (y1 - y) * (y1 - y) + (z1 - z) * (z1 - z));
            if (d <= 20) {
                print("\n");
                print(" * * * HIT * * *   TARGET IS NON-FUNCTIONAL\n");
                print("\n");
                print("DISTANCE OF EXPLOSION FROM TARGET WAS " + d + " KILOMETERS.");
                print("\n");
                print("MISSION ACCOMPLISHED IN " + r + " SHOTS.\n");
                r = 0;
                for (i = 1; i <= 5; i++)
                    print("\n");
                print("NEXT TARGET...\n");
                print("\n");
                break;
            }
            x2 = x1 - x;
            y2 = y1 - y;
            z2 = z1 - z;
            if (x2 >= 0)
                print("SHOT IN FRONT OF TARGET " + x2 + " KILOMETERS.\n");
            else
                print("SHOT BEHIND TARGET " + -x2 + " KILOMETERS.\n");
            if (y2 >= 0)
                print("SHOT TO LEFT OF TARGET " + y2 + " KILOMETERS.\n");
            else
                print("SHOT TO RIGHT OF TARGET " + -y2 + " KILOMETERS.\n");
            if (z2 >= 0)
                print("SHOT ABOVE TARGET " + z2 + " KILOMETERS.\n");
            else
                print("SHOT BELOW TARGET " + -z2 + " KILOMETERS.\n");
            print("APPROX POSITION OF EXPLOSION:  X=" + x1 + "   Y=" + y1 + "   Z=" + z1 + "\n");
            print("     DISTANCE FROM TARGET = " + d + "\n");
            print("\n");
            print("\n");
            print("\n");
        }
    }
}

```

这道题目缺少上下文，无法得知代码的具体作用。一般来说，`main()` 函数是程序的入口点，程序从此处开始执行。在执行之前，可能会对程序进行初始化，比如加载资源、设置变量等。在 `main()` 函数中，程序会尽力去运行所有的代码，直到遇到错误或者程序结束。


```
main();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)

Modified so that if the user enters "quit" or "stop" for the input, the program will exit.
This way the user doesn't have to enter Contorl-C to quit.

Target values can be space and/or comma separated, so "1 2 3" is valid, as is "1,2,3" or even "1, 2, 3".
I believe the original Basic program wanted "1,2,3" or else each on a separate line.


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


# `86_Target/python/target.py`

这段代码是一个Python程序，它的目的是创建一个名为“TARGET”的类，该类用于模拟瞄准射击游戏/3D trigonometry练习。它由以下几个部分组成：

1. 导入了math和random模块，这些模块用于数学计算和生成随机数。
2. 定义了一个PAGE_WIDTH变量，将其赋值为64，这意味着页面的宽度为64像素。
3. 在不输出源代码的情况下，定义了一个名为“Weapon”的类，该类用于表示武器对象。
4. 在Weapon类中，定义了一个构造函数，用于初始化武器对象的参数。
5. 在Weapon类中，定义了一个“shoot”方法，用于向目标点发送一束射线，并计算出射线与目标点之间的角度。
6. 在Weapon类中，定义了一个“update”方法，用于在每一帧更新武器对象的位置和状态。
7. 在最后，程序导入了typing模块，该模块允许将Weapon对象作为列表推入列表中。


```
"""
TARGET

Weapon targeting simulation / 3d trigonometry practice

Ported by Dave LeCompte
"""

import math
import random
from typing import List

PAGE_WIDTH = 64


```



这些函数的主要目的是打印一些文本，并在页面上居中显示它们。

第一个函数 `print_centered` 接收一个字符串参数 `msg`，并返回其本身。这个函数的作用是打印一个字符串，并将其置于页面的中心位置，使得字符串在页面上看起来更整齐。在这个函数中，`PAGE_WIDTH` 是一个变量，表示整个页面的宽度。它通过 `len(msg)` 计算出需要多少内联的行距，然后将其除以 2，得到行距的数量。最后，在 `print` 函数中，使用 `spaces` 变量中的字符来填充内联行距，并在字符串两侧加上空格，使得内联后的字符串看起来更整齐。

第二个函数 `print_header` 接收一个字符串参数 `title`，并返回其本身。这个函数的作用是打印一个字符串，并将其置于页面的顶部，这样在页面上显示标题时，文本会显得更加突出。在这个函数中，`print_centered` 函数用来打印标题，并将其打印到页面的顶部。

第三个函数 `print_instructions` 接收一个字符串参数 `instructions`，并返回其本身。这个函数的作用是打印一些文本，并将其置于页面上。在这个函数中，`print_header` 函数用来打印标题，`print_centered` 函数用来打印文本，然后将它们打印在页面上。在这个函数中，`instructions` 变量中的字符被打印出来，形成一个短文本，告诉用户该如何操作。


```
def print_centered(msg: str) -> None:
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    print(spaces + msg)


def print_header(title: str) -> None:
    print_centered(title)
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print()
    print()
    print()


def print_instructions() -> None:
    print("YOU ARE THE WEAPONS OFFICER ON THE STARSHIP ENTERPRISE")
    print("AND THIS IS A TEST TO SEE HOW ACCURATE A SHOT YOU")
    print("ARE IN A THREE-DIMENSIONAL RANGE.  YOU WILL BE TOLD")
    print("THE RADIAN OFFSET FOR THE X AND Z AXES, THE LOCATION")
    print("OF THE TARGET IN THREE DIMENSIONAL RECTANGULAR COORDINATES,")
    print("THE APPROXIMATE NUMBER OF DEGREES FROM THE X AND Z")
    print("AXES, AND THE APPROXIMATE DISTANCE TO THE TARGET.")
    print("YOU WILL THEN PROCEEED TO SHOOT AT THE TARGET UNTIL IT IS")
    print("DESTROYED!")
    print()
    print("GOOD LUCK!!")
    print()
    print()


```

这段代码是一个Python函数，主要作用是 prompt 和 next_target 函数。

prompt 函数通过输入连续三个整数来计算客户端的角度偏差，如果客户端输入的不是三个连续的数字，那么该函数就会跳过继续计算。函数返回一个包含三个浮点数的列表，即计算得到的角度偏差。

next_target 函数用于计算客户端下一个目标点的位置，通过连续打印五行文本，然后再次打印 "NEXT TARGET..." 消息，在客户端应用程序中等待用户输入位置偏移量。函数会连续打印五个目标点的位置，然后等待用户输入偏移量。函数返回 None，这意味着它不会执行后续计算，而是返回一个 None 对象，用于告诉 Python 程序不需要等待用户输入偏移量。


```
def prompt() -> List[float]:
    while True:
        response = input("INPUT ANGLE DEVIATION FROM X, DEVIATION FROM Z, DISTANCE? ")
        if "," not in response:
            continue

        terms = response.split(",")
        if len(terms) != 3:
            continue

        return [float(t) for t in terms]


def next_target() -> None:
    for _ in range(5):
        print()
    print("NEXT TARGET...")
    print()


```

这段代码定义了一个名为 "describe_miss" 的函数，它接受四个参数：一个二维的 xyz 坐标系中的 x, y, z 值，以及一个距离目标点（d）的值。

函数的作用是计算出从不同方向距离目标点的最短距离，并输出这些距离。

函数首先检查 x, y, z 值是否小于零，如果是，则输出 "SHOT BEHIND TARGET {-x:.2f} KILOMETERS."，如果不是，则输出 "SHOT IN FRONT OF TARGET {x:.2f} KILOMETERS."。同样的，如果是 y < 0，则输出 "SHOT TO RIGHT OF TARGET {-y:.2f} KILOMETERS"，如果不是，则输出 "SHOT TO LEFT OF TARGET {y:.2f} KILOMETERS。"；如果是 z < 0，则输出 "SHOT BELOW TARGET {-z:.2f} KILOMETERS"，如果不是，则输出 "SHOT ABOVE TARGET {z:.2f} KILOMETERS。"。

接下来，函数会计算出目标点到各个方向的距离，并输出这些距离。然后，函数会输出目标点与各个方向的距离的平方，并再次输出一个空行，以便下一行输出。


```
def describe_miss(x, y, z, x1, y1, z1, d) -> None:
    x2 = x1 - x
    y2 = y1 - y
    z2 = z1 - z

    if x2 < 0:
        print(f"SHOT BEHIND TARGET {-x2:.2f} KILOMETERS.")
    else:
        print(f"SHOT IN FRONT OF TARGET {x2:.2f} KILOMETERS.")

    if y2 < 0:
        print(f"SHOT TO RIGHT OF TARGET {-y2:.2f} KILOMETERS.")
    else:
        print(f"SHOT TO LEFT OF TARGET {y2:.2f} KILOMETERS.")

    if z2 < 0:
        print(f"SHOT BELOW TARGET {-z2:.2f} KILOMETERS.")
    else:
        print(f"SHOT ABOVE TARGET {z2:.2f} KILOMETERS.")

    print(f"APPROX POSITION OF EXPLOSION:  X={x1:.4f}   Y={y1:.4f}   Z={z1:.4f}")
    print(f"     DISTANCE FROM TARGET = {d:.2f}")
    print()
    print()
    print()


```

This is a Python program that simulates an explosion based on the distance received by an explosion from a target. The program takes in the angles of the explosion and the distance between the explosion and the target, as well as the number of shots fired.

The program first checks if the number of shots fired is greater than or equal to 1. If it is, it calculates the blast radius based on the shot type and then takes the player's position and the distance to the target, and calculate the blast distance.

If the number of shots fired is less than 1, the program will not blast and will instead display a message.

The program also checks if the distance to the target is less than or equal to 20, and if it is, it will display the distance of the explosion from the target and a message indicating that the mission was accomplished.

The program also has a function to describe the blast.

Note that this program is a simulation and it does not reflect any real-world scenario and the function it has is not safe for any use.


```
def do_shot_loop(p1, x, y, z) -> None:
    shot_count = 0
    while True:
        shot_count += 1
        if shot_count == 1:
            p3 = int(p1 * 0.05) * 20
        elif shot_count == 2:
            p3 = int(p1 * 0.1) * 10
        elif shot_count == 3:
            p3 = int(p1 * 0.5) * 2
        elif shot_count == 4:
            p3 = int(p1)
        else:
            p3 = p1

        if p3 == int(p3):
            print(f"     ESTIMATED DISTANCE: {p3}")
        else:
            print(f"     ESTIMATED DISTANCE: {p3:.2f}")
        print()
        a1, b1, p2 = prompt()

        if p2 < 20:
            print("YOU BLEW YOURSELF UP!!")
            return

        a1 = math.radians(a1)
        b1 = math.radians(b1)
        show_radians(a1, b1)

        x1 = p2 * math.sin(b1) * math.cos(a1)
        y1 = p2 * math.sin(b1) * math.sin(a1)
        z1 = p2 * math.cos(b1)

        distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2 + (z1 - z) ** 2)

        if distance <= 20:
            print()
            print(" * * * HIT * * *   TARGET IS NON FUNCTIONAL")
            print()
            print(f"DISTANCE OF EXPLOSION FROM TARGET WAS {distance:.4f} KILOMETERS")
            print()
            print(f"MISSION ACCOMPLISHED IN {shot_count} SHOTS.")

            return
        else:
            describe_miss(x, y, z, x1, y1, z1, distance)


```

这段代码定义了两个函数，`show_radians()` 函数接受两个参数 `a` 和 `b`，并输出这两个角度的弧度值。`play_game()` 函数则是一个无限循环，每次会生成一个随机的角度 `a` 和一个随机的角度 `b`，然后调用 `show_radians()` 函数输出这两个角度的弧度值。接着调用 `do_shot_loop()` 函数，这个函数没有定义，因此不会执行。最后调用 `next_target()` 函数，这个函数也没有定义，因此不会执行。


```
def show_radians(a, b) -> None:
    print(f"RADIANS FROM X AXIS = {a:.4f}   FROM Z AXIS = {b:.4f}")


def play_game() -> None:
    while True:
        a = random.uniform(0, 2 * math.pi)  # random angle
        b = random.uniform(0, 2 * math.pi)  # random angle

        show_radians(a, b)

        p1 = random.uniform(0, 100000) + random.uniform(0, 1)
        x = math.sin(b) * math.cos(a) * p1
        y = math.sin(b) * math.sin(a) * p1
        z = math.cos(b) * p1
        print(
            f"TARGET SIGHTED: APPROXIMATE COORDINATES:  X={x:.1f}  Y={y:.1f}  Z={z:.1f}"
        )

        do_shot_loop(p1, x, y, z)
        next_target()


```

这段代码是一个Python程序，名为“main”。在程序中，首先定义了一个名为“main”的函数，该函数返回一个名为“None”的类型。

函数体中，先打印一个名为“TARGET”的输出，然后打印一系列的指令（informations）。接着调用一个名为“play_game”的函数，最后输出一条消息，表明程序成功运行。

程序的最后，使用一个名为“if __name__ == "__main__":”的语句来确保程序在运行时会按照以下顺序执行“main”函数内的内容。如果程序成功运行，则程序将进入一个无限循环，调用“main”函数内的“play_game”函数。


```
def main() -> None:
    print_header("TARGET")
    print_instructions()

    play_game()


if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### 3-D Plot

3-D PLOT will plot the family of curves of any function. The function Z is plotted as “rising” out of the x-y plane with x and y inside a circle of radius 30. The resultant plot looks almost 3-dimensional.

You set the function you want plotted in line 5. As with any mathematical plot, some functions come out “prettier” than others.

The author of this amazingly clever program is Mark Bramhall of DEC.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=167)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=182)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `87_3-D_Plot/csharp/Function.cs`



这段代码定义了一个名为 `Function` 的内部类，它内部定义了一个名为 `GetRows` 的方法，以及一个名为 `GetValues` 的内部方法。

`GetRows` 方法的作用是获取一个浮点数序列(也就是一个二维数组)，该序列的每个元素都是实数轴上某个位置的值，用于绘制曲线。该方法包含一个循环，变量 `x` 从-30到30，每次增加1.5。

`GetValues` 方法的作用是获取一个浮点数序列(也就是一个二维数组)，该序列的每个元素都是在实数轴上某个位置的值，用于绘制曲线上的点。该方法包含一个循环，变量 `x` 从-30到30，每次增加1.5，变量 `y` 从-yLimit到-yLimit，每次减少5。在循环内部，使用 `GetValue` 方法获取点(x,y)的值，其中 `GetValue` 方法接收一个浮点数参数 x 和一个浮点数参数 y，并计算出 y 轴的值。如果当前点的值大于之前计算得到的值，则更新该值，否则继续向上循环。

最后，`Function` 类可能还包含其他方法，但是上述代码提供了对函数的主要定义。


```
using System;
using System.Collections.Generic;

namespace Plot
{
    internal static class Function
    {
        internal static IEnumerable<IEnumerable<int>> GetRows()
        {
            for (var x = -30f; x <= 30f; x += 1.5f)
            {
                yield return GetValues(x);
            }
        }

        private static IEnumerable<int> GetValues(float x)
        {
            var zPrevious = 0;
            var yLimit = 5 * (int)(Math.Sqrt(900 - x * x) / 5);

            for (var y = yLimit; y >= -yLimit; y -= 5)
            {
                var z = GetValue(x, y);

                if (z > zPrevious)
                {
                    zPrevious = z;
                    yield return z;
                }
            }
        }

        private static int GetValue(float x, float y)
        {
            var r = (float)Math.Sqrt(x * x + y * y);
            return (int)(25 + 30 * Math.Exp(-r * r / 100) - 0.7f * y);
        }
    }
}

```

# `87_3-D_Plot/csharp/Program.cs`



这段代码是一个用于生成3D图形的Python程序。程序的主要作用是指导如何使用不同的函数和类来创建一个自定义的、非标准的三维函数图形。下面是程序的详细解释：

1. 程序导入了三个命名空间：System、Plot和Function。

2. 在Main函数中定义了一个程序的主要实例。

3. 在main函数中调用了一个名为FunctionGetRows的函数，这个函数获取了一个包含行索引的二维数组。

4. 在Main函数中循环遍历每一行，并且循环变量z包含从0到数组长度的值。

5. 在循环内部，使用Console.WriteLine函数输出一个星号(*)来表示当前正在绘制的点。

6. 在循环内部，使用Console.SetCursorPosition函数设置当前绘图位置为当前行的索引值，以便输出更多的点。

7. 程序还定义了一个名为Plot的类，在其中包含一个名为plot的方法，这个方法接受一个整数参数，代表当前要绘制的点。

8. 在Plot方法中，程序使用Console.Write函数来输出当前点的位置，然后使用Console.SetCursorPosition函数来设置绘图位置。

9. 最后，程序还定义了一个名为PrintTitle的静态函数，这个函数使用Console.WriteLine函数来输出标题信息。


```
﻿using System;

namespace Plot
{
    class Program
    {
        static void Main(string[] args)
        {
            PrintTitle();

            foreach (var row in Function.GetRows())
            {
                foreach (var z in row)
                {
                    Plot(z);
                }
                Console.WriteLine();
            }
        }

        private static void PrintTitle()
        {
            Console.WriteLine("                                3D Plot");
            Console.WriteLine("               Creative Computing  Morristown, New Jersey");
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
        }

        private static void Plot(int z)
        {
            var x = Console.GetCursorPosition().Top;
            Console.SetCursorPosition(z, x);
            Console.Write("*");
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html)

Converted to [D](https://dlang.org/) by [Bastiaan Veelo](https://github.com/veelo).

## Running the code

Assuming the reference [dmd](https://dlang.org/download.html#dmd) compiler:
```shell
dmd -dip1000 -run threedeeplot.d
```

[Other compilers](https://dlang.org/download.html) also exist.

## On rounding floating point values to integer values

The D equivalent of Basic `INT` is [`floor`](https://dlang.org/phobos/std_math_rounding.html#.floor),
which rounds towards negative infinity. If you change occurrences of `floor` to
[`lrint`](https://dlang.org/phobos/std_math_rounding.html#.lrint), you'll see that the plots show a bit more detail,
as is done in the bonus below.

## Bonus: Self-writing programs

With a small modification to the source, the program can be extended to **plot a random function**, and **print its formula**.

```shell
rdmd -dip1000 threedeeplot_random.d
```
(`rdmd` caches the executable, which results in speedy execution when the source does not change.)

### Example output
```
                                    3D Plot
              (After Creative Computing  Morristown, New Jersey)


                           f(z) = 30 * sin(z / 10.0)

                             *
                      *      *    * *
                *         *      *    * *
                    *         *      *    * *
            *           *        *       *   *  *
               *           *         *      *   *  *
                  *           *         *     *    * **
       *             *           *        *      *   *  *
         *              *           *       *     *   *  **
            *              *          *       *    *   *  * *
              *              *          *      *   *   *  *  *
                *              *          *     *  *  *   *   **
                  *              *         *    * *  *   *    * *
   *                *             *        *    ** *    *     * *
    *                *             *        *  **     *      *   *
     *                 *            *       * *     *       *    *
      *                 *            *      * *   *         *    *
       *                *             *     ** *           *     **
        *                *            *     **            *      **
        *                *            *     *            *       **
        *                *            *     *            *       **
        *                *            *     *            *       **
        *                *            *     **            *      **
       *                *             *     ** *           *     **
      *                 *            *      * *   *         *    *
     *                 *            *       * *     *       *    *
    *                *             *        *  **     *      *   *
   *                *             *        *    ** *    *     * *
                  *              *         *    * *  *   *    * *
                *              *          *     *  *  *   *   **
              *              *          *      *   *   *  *  *
            *              *          *       *    *   *  * *
         *              *           *       *     *   *  **
       *             *           *        *      *   *  *
                  *           *         *     *    * **
               *           *         *      *   *  *
            *           *        *       *   *  *
                    *         *      *    * *
                *         *      *    * *
                      *      *    * *
                             *
```

### Breakdown of differences

Have a look at the relevant differences between `threedeeplot.d` and `threedeeplot_random.d`.
This is the original function with the single expression that is evaluated for the plot:
```d
    static float fna(float z)
    {
        return 30.0 * exp(-z * z / 100.0);
    }
```
Here `static` means that the nested function does not need acces to its enclosing scope.

Now, by inserting the following:
```d
    enum functions = ["30.0 * exp(-z * z / 100.0)",
                      "sqrt(900.01 - z * z) * .9 - 2",
                      "30 * (cos(z / 16.0) + .5)",
                      "30 - 30 * sin(z / 18.0)",
                      "30 * exp(-cos(z / 16.0)) - 30",
                      "30 * sin(z / 10.0)"];

    size_t index = uniform(0, functions.length);
    writeln(center("f(z) = " ~ functions[index], width), "\n");
```
and changing the implementation of `fna` to
```d
    float fna(float z)
    {
        final switch (index)
        {
            static foreach (i, f; functions)
                case i:
                    mixin("return " ~ f ~ ";");
        }
    }
```
we unlock some very special abilities of D. Let's break it down:

```d
    enum functions = ["30.0 * exp(-z * z / 100.0)", /*...*/];
```
This defines an array of strings, each containing a mathematical expression. Due to the `enum` keyword, this is an
array that really only exists at compile-time.

```d
    size_t index = uniform(0, functions.length);
```
This defines a random index into the array. `functions.length` is evaluated at compile-time, due to D's compile-time
function evaluation (CTFE).

```d
    writeln(center("f(z) = " ~ functions[index], width), "\n");
```
Unmistakenly, this prints the formula centered on a line. What happens behind the scenes is that `functions` (which
only existed at compile-time before now) is pasted in, so that an instance of that array actually exists at run-time
at this spot, and is instantly indexed.

```d
    float fna(float z)
    {
        final switch (index)
        {
            // ...
        }
    }
```
`static` has been dropped from the nested function because we want to evaluate `index` inside it. The function contains
an ordinary `switch`, with `final` providing some extra robustness. It disallows a `default` case and produces an error
when the switch doesn't handle all cases. The `switch` body is where the magic happens and consists of these three
lines:
```d
            static foreach (i, f; functions)
                case i:
                    mixin("return " ~ f ~ ";");
```
The `static foreach` iterates over `functions` at compile-time, producing one `case` for every element in `functions`.
`mixin` takes a string, which is constructed at compile-time, and pastes it right into the source.

In effect, the implementation of `float fna(float z)` unrolls itself into
```d
    float fna(float z)
    {
        final switch (index)
        {
            case 0:
                return 30.0 * exp(-z * z / 100.0);
            case 1:
                return sqrt(900.01 - z * z) * .9 - 2;
            case 2:
                return 30 * (cos(z / 16.0) + .5);
            case 3:
                return 30 - 30 * sin(z / 18.0);
            case 4:
                return 30 * exp(-cos(z / 16.0)) - 30;
            case 5:
                return 30 * sin(z / 10.0)";
        }
    }
```

So if you feel like adding another function, all you need to do is append it to the `functions` array, and the rest of
the program *rewrites itself...*


# `87_3-D_Plot/java/Plot3D.java`

这段代码是一个名为 "3D Plot" 的 Java 类，它导入了 Math 包，然后定义了一个名为 "main" 的方法。

在 "main" 方法中，该类没有做任何特定的事情，但是在这个包中定义了一个 "Game of 3-D Plot"，根据basic游戏3D图形的原理，创建了一个3D图形的类。


```
import java.lang.Math;

/**
 * Game of 3-D Plot
 * <p>
 * Based on the BASIC game of 3-D Plot here
 * https://github.com/coding-horror/basic-computer-games/blob/main/87%203-D%20Plot/3dplot.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 *
 * Converted from BASIC to Java by Darren Cardenas.
 */

// Java class names cannot begin with a letter, so class name 3dplot cannot be used
```

This is a Java class that simulates a 3D plot using the `Plot3D` class. The `startGame` method displays a 3D plot of Morristown, NJ. The `func` method is a simple function that calculates the value of the `x` and `y` coordinates based on the `z` coordinate, which is passed in as an argument.

The `main` method is the starting point of the application and initializes the `Plot3D` object and displays the plot using the `play` method.

Note that the `Plot3D` class is not defined in this code, so you will need to import it somewhere in your code. You can add the following import statement at the top of your class file to import it:
```python
import org.tmatesh.荏苒.plot3d.Plot3D;
```


```
public class Plot3D {


  public void play() {

    showIntro();
    startGame();

  }  // End of method play


  private void showIntro() {

    System.out.println(" ".repeat(31) + "3D PLOT");
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    System.out.println("\n\n\n");

  }  // End of method showIntro


  private void startGame() {

    float row = 0;
    int column = 0;
    int limit = 0;
    int plotVal = 0;
    int root = 0;

    String lineContent = "";

    // Begin loop through all rows
    for (row = -30; row <= 30; row += 1.5) {

      limit = 0;

      root = 5 * (int) Math.floor((Math.sqrt(900 - row * row) / 5));

      // Begin loop through all columns
      for (column = root; column >= -root; column += -5) {

        plotVal = 25 + (int) Math.floor(func(Math.sqrt(row * row + column * column)) - 0.7 * column);

        if (plotVal > limit) {

          limit = plotVal;

          // Add whitespace
          while (lineContent.length() < (plotVal-1)) {
            lineContent += " ";
          }

          lineContent += "*";

        }

      }  // End loop through all columns

      System.out.println(lineContent);

      lineContent = "";

    }  // End loop through all rows

  }  // End of method startGame


  // Function to be plotted
  public double func(double inputVal) {

    return (30 * Math.exp(-inputVal * inputVal / 100));

  }


  public static void main(String[] args) {

    Plot3D plot = new Plot3D();
    plot.play();

  }  // End of method main

}  // End of class Plot3D

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `87_3-D_Plot/javascript/3dplot.js`

这是一段用于在网页上生成3D图表数据的JavaScript代码。以下是代码的作用：

1. `print` 函数的作用是将要打印的字符串添加到页面上，并将其显示为文本节点。具体来说，它通过 `document.getElementById("output").appendChild(document.createTextNode(str))` 将一个字符串添加到页面上，其中 `getElementById("output")` 选择器定位到具有 "output" id 的元素，并将其内容设置为 `str` 字符串。`appendChild` 方法将 `str` 字符串插入到元素中，并返回一个新的文本节点。

2. `tab` 函数的作用是在指定空间数量（以空格为间隔）的字符串中打印字符。具体来说，它通过创建一个空字符串，并使用 `while` 循环来逐个生成字符。在每次循环中，字符的数量递增，然后在生成字符串中的每个字符时，将其空间数量减少1。最后，它将生成的字符串返回。


```
// 3D PLOT
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//
function print(str)
{
	document.getElementById("output").appendChild(document.createTextNode(str));
}

function tab(space)
{
	var str = "";
	while (space-- > 0)
		str += " ";
	return str;
}

```

这段代码定义了一个名为 equation 的函数，它接受一个整数参数 input。函数内部使用 Math.exp 函数计算输入参数的平方根，再乘以 30，最终结果保留一位小数。

接下来两行代码输出了一些字符，用于在文本中显示输出结果。

第三行代码使用 for 循环遍历变量 x 的取值范围，对于每个 x 的值，函数内部的 l 变量将被初始化为 0。

在循环内部，使用 Math.sqrt 函数计算变量 y 的值。通过调用 equation 函数计算 z 值，其中变量 y 将根据方程 x * x + y * y 进行替换。如果 z 值大于 l 值，则 l 值将更新为 z，否则在循环中字符串将添加 z。最后，在循环结束后，将字符串输出，并在字符串中添加星号。

第四行代码在循环内部使用 str += " "; 将字符串添加到了 l 的后面。

第五行代码输出了一些字符，用于在文本中显示输出结果。


```
function equation(input)
{
	return 30 * Math.exp(-input * input / 100);
}

print(tab(32) + "3D PLOT\n");
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");

for (x = -30; x <= 30; x += 1.5) {
	l = 0;
	y1 = 5 * Math.floor(Math.sqrt(900 - x * x) / 5);
	str = "";
	for (y = y1; y >= -y1; y -= 5) {
		z = Math.floor(25 + equation(Math.sqrt(x * x + y * y)) - .7 * y);
		if (z > l) {
			l = z;
			while (str.length < z)
				str += " ";
			str += "*";
		}
	}
	print(str + "\n");
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


# `87_3-D_Plot/python/3dplot.py`

这段代码是一个用于绘制3D曲线的Python程序。它将输入的x值经过一个数学方程转换为三维空间中的坐标，并使用这些坐标计算出每个y坐标的值。最终，程序将输出由星号组成的3D曲线。

具体来说，代码中定义了一个方程函数`equation(x)`，它接受一个浮点数x作为输入，并返回一个浮点数。这个方程使用单点乘积法(如取对数)将x乘以自己，再乘以30，最后对结果进行指数函数Exponential(自然对数)的幂次方，从而得到一个在指定区间内的复杂函数。

主函数`main()`中包含以下步骤：

1. 输出程序的名称和版权信息。
2. 输出一条消息，其中包含曲线将使用负数x轴坐标，最大x轴坐标为315，每个横坐标的最大值将用于绘制曲线。
3. 循环使用浮点数变量`x`，将其除以10并取平方根得到每个横坐标的中心值。
4. 循环使用数学方程`equation(sqrt(x * x))`计算每个横坐标对应的y值。
5. 循环从每个横坐标的最大值开始，将计算出的y值存储在一个列表中，并在对应的横坐标上绘制星号。
6. 最后，程序将循环中的所有y值打印出来，以便观察3D曲线。


```
#!/usr/bin/env python3

# 3D PLOT
#
# Converted from BASIC to Python by Trevor Hobson

from math import exp, floor, sqrt


def equation(x: float) -> float:
    return 30 * exp(-x * x / 100)


def main() -> None:
    print(" " * 32 + "3D PLOT")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n\n")

    for x in range(-300, 315, 15):
        x1 = x / 10
        max_column = 0
        y1 = 5 * floor(sqrt(900 - x1 * x1) / 5)
        y_plot = [" "] * 80

        for y in range(y1, -(y1 + 5), -5):
            column = floor(25 + equation(sqrt(x1 * x1 + y * y)) - 0.7 * y)
            if column > max_column:
                max_column = column
                y_plot[column] = "*"
        print("".join(y_plot))


```

这段代码是一个if语句，它的判断条件是(__name__ == "__main__")。这个if语句的作用是在程序被调用时执行main()函数。

if语句是一种布尔运算，它的值只有两种可能：True或False。如果条件为True，则if语句会执行if语句内部的代码，否则不会执行。在这个例子中，if(__name__ == "__main__") 的条件永远为False，因此if语句不会执行if语句内部的代码。

但是，由于if(__name__ == "__main__") 的判断条件是程序被调用时，因此它的代码实际上永远也不会被执行。所以，这段代码几乎没有任何作用，它的存在只是为了让程序在编译时看起来更完整。


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


### 3-D Tic-Tac-Toe

3-D TIC-TAC-TOE is a game of tic-tac-toe in a 4x4x4 cube. You must get 4 markers in a row or diagonal along any 3-dimensional plane in order to win.

Each move is indicated by a 3-digit number (digits not separated by commas), with each digit between 1 and 4 inclusive. The digits indicate the level, column, and row, respectively, of the move. You can win if you play correctly; although, it is considerably more difficult than standard, two-dimensional 3x3 tic-tac-toe.

This version of 3-D TIC-TAC-TOE is from Dartmouth College.

### Conversion notes

The AI code for TicTacToe2 depends quite heavily on the non-structured GOTO (I can almost hear Dijkstra now) and translation is quite challenging. This code relies very heavily on GOTOs that bind the code tightly together. Comments explain where that happens in the original.

There are at least two bugs from the original BASIC:

1. Code should only allow player to input valid 3D coordinates where every digit is between 1 and 4, but the original code allows any value between 111 and 444 (such as 297, for instance).
2. If the player moves first and the game ends in a draw, the original program will still prompt the player for a move instead of calling for a draw.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=168)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=183)


Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `88_3-D_Tic-Tac-Toe/csharp/Program.cs`

这段代码是一个C#程序，定义了一个名为"ThreeDTicTacToe"namespace，包含一个名为"Program"类，以及一个名为"Main"的静态方法。

在Main方法的内部，创建了一个新的类名为"Qubic"，且没有定义任何成员。

该代码的作用是编译并运行一个名为"Qubic.cs"的文件，即定义了一个名为"Qubic"的类，但没有定义任何成员，编译运行后，会弹出一个应用程序窗口，显示游戏"3D Tic Tac Toe"的随机游戏结果。


```
﻿namespace ThreeDTicTacToe
{
    class Program
    {
        static void Main()
        {
            new Qubic().Run();
        }
    }
}

```

# `88_3-D_Tic-Tac-Toe/csharp/Qubic.cs`

This is a summary of a class that has methods for checking for a draw in a chess game, clearing the board of potential moves, and clearing the board of all spaces. The class is based on the C# programming language and uses a board game class called "BoardGame". The "BoardGame" class has methods for getting and setting the current position of the chess board, and for checking whether a given piece is currently on the board. The "DrawChecker" class contains the code for the methods in the "BoardGame" class.


```
﻿using System.Text;

namespace ThreeDTicTacToe
{
    /// <summary>
    /// Qubic is a 3D Tic-Tac-Toe game played on a 4x4x4 cube. This code allows
    ///  a player to compete against a deterministic AI that is surprisingly
    ///  difficult to beat.
    /// </summary>
    internal class Qubic
    {
        // The Y variable in the original BASIC.
        private static readonly int[] CornersAndCenters = QubicData.CornersAndCenters;
        // The M variable in the original BASIC.
        private static readonly int[,] RowsByPlane = QubicData.RowsByPlane;

        // Board spaces are filled in with numeric values. A space could be:
        //
        //  - EMPTY: no one has moved here yet.
        //  - PLAYER: the player moved here.
        //  - MACHINE: the machine moved here.
        //  - POTENTIAL: the machine, in the middle of its move,
        //      might fill a space with a potential move marker, which
        //      prioritizes the space once it finally chooses where to move.
        //
        // The numeric values allow the program to determine what moves have
        //  been made in a row by summing the values in a row. In theory, the
        //  individual values could be any positive numbers that satisfy the
        //  following:
        //
        //  - EMPTY = 0
        //  - POTENTIAL * 4 < PLAYER
        //  - PLAYER * 4 < MACHINE
        private const double PLAYER = 1.0;
        private const double MACHINE = 5.0;
        private const double POTENTIAL = 0.125;
        private const double EMPTY = 0.0;

        // The X variable in the original BASIC. This is the Qubic board,
        //  flattened into a 1D array.
        private readonly double[] Board = new double[64];

        // The L variable in the original BASIC. There are 76 unique winning rows
        //  in the board, so each gets an entry in RowSums. A row sum can be used
        //  to check what moves have been made to that row in the board.
        //
        // Example: if RowSums[i] == PLAYER * 4, the player won with row i!
        private readonly double[] RowSums = new double[76];

        public Qubic() { }

        /// <summary>
        /// Run the Qubic game.
        ///
        /// Show the title, prompt for instructions, then begin the game loop.
        /// </summary>
        public void Run()
        {
            Title();
            Instructions();
            Loop();
        }

        /***********************************************************************
        /* Terminal Text/Prompts
        /**********************************************************************/
        #region TerminalText

        /// <summary>
        /// Display title and attribution.
        ///
        /// Original BASIC: 50-120
        /// </summary>
        private static void Title()
        {
            Console.WriteLine(
                "\n" +
                "                                 QUBIC\n\n" +
                "               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n"
            );
        }

        /// <summary>
        /// Prompt user for game instructions.
        ///
        /// Original BASIC: 210-313
        /// </summary>
        private static void Instructions()
        {
            Console.Write("DO YOU WANT INSTRUCTIONS? ");
            var yes = ReadYesNo();

            if (yes)
            {
                Console.WriteLine(
                    "\n" +
                    "THE GAME IS TIC-TAC-TOE IN A 4 X 4 X 4 CUBE.\n" +
                    "EACH MOVE IS INDICATED BY A 3 DIGIT NUMBER, WITH EACH\n" +
                    "DIGIT BETWEEN 1 AND 4 INCLUSIVE.  THE DIGITS INDICATE THE\n" +
                    "LEVEL, ROW, AND COLUMN, RESPECTIVELY, OF THE OCCUPIED\n" +
                    "PLACE.\n" +
                    "\n" +
                    "TO PRINT THE PLAYING BOARD, TYPE 0 (ZERO) AS YOUR MOVE.\n" +
                    "THE PROGRAM WILL PRINT THE BOARD WITH YOUR MOVES INDI-\n" +
                    "CATED WITH A (Y), THE MACHINE'S MOVES WITH AN (M), AND\n" +
                    "UNUSED SQUARES WITH A ( ).  OUTPUT IS ON PAPER.\n" +
                    "\n" +
                    "TO STOP THE PROGRAM RUN, TYPE 1 AS YOUR MOVE.\n\n"
                );
            }
        }

        /// <summary>
        /// Prompt player for whether they would like to move first, or allow
        ///  the machine to make the first move.
        ///
        /// Original BASIC: 440-490
        /// </summary>
        /// <returns>true if the player wants to move first</returns>
        private static bool PlayerMovePreference()
        {
            Console.Write("DO YOU WANT TO MOVE FIRST? ");
            var result = ReadYesNo();
            Console.WriteLine();
            return result;
        }

        /// <summary>
        /// Run the Qubic program loop.
        /// </summary>
        private void Loop()
        {
            // The "retry" loop; ends if player quits or chooses not to retry
            // after game ends.
            while (true)
            {
                ClearBoard();
                var playerNext = PlayerMovePreference();

                // The "game" loop; ends if player quits, player/machine wins,
                // or game ends in draw.
                while (true)
                {
                    if (playerNext)
                    {
                        // Player makes a move.
                        var playerAction = PlayerMove();
                        if (playerAction == PlayerAction.Move)
                        {
                            playerNext = !playerNext;
                        }
                        else
                        {
                            return;
                        }
                    }
                    else
                    {
                        // Check for wins, if any.
                        RefreshRowSums();
                        if (CheckPlayerWin() || CheckMachineWin())
                        {
                            break;
                        }

                        // Machine makes a move.
                        var machineAction = MachineMove();
                        if (machineAction == MachineAction.Move)
                        {
                            playerNext = !playerNext;
                        }
                        else if (machineAction == MachineAction.End)
                        {
                            break;
                        }
                        else
                        {
                            throw new Exception("unreachable; machine should always move or end game in game loop");
                        }
                    }
                }

                var retry = RetryPrompt();

                if (!retry)
                {
                    return;
                }
            }
        }

        /// <summary>
        /// Prompt the user to try another game.
        ///
        /// Original BASIC: 1490-1560
        /// </summary>
        /// <returns>true if the user wants to play again</returns>
        private static bool RetryPrompt()
        {
            Console.Write("DO YOU WANT TO TRY ANOTHER GAME? ");
            return ReadYesNo();
        }

        /// <summary>
        /// Read a yes/no from the terminal. This method accepts anything that
        ///  starts with N/n as no and Y/y as yes.
        /// </summary>
        /// <returns>true if the player answered yes</returns>
        private static bool ReadYesNo()
        {
            while (true)
            {
                var response = Console.ReadLine() ?? " ";
                if (response.ToLower().StartsWith("y"))
                {
                    return true;
                }
                else if (response.ToLower().StartsWith("n"))
                {
                    return false;
                }
                else
                {
                    Console.Write("INCORRECT ANSWER.  PLEASE TYPE 'YES' OR 'NO'. ");
                }
            }
        }

        #endregion

        /***********************************************************************
        /* Player Move
        /**********************************************************************/
        #region PlayerMove

        /// <summary>
        /// Possible actions player has taken after ending their move. This
        ///  replaces the `GOTO` logic that allowed the player to jump out of
        ///  the game loop and quit.
        /// </summary>
        private enum PlayerAction
        {
            /// <summary>
            /// The player ends the game prematurely.
            /// </summary>
            Quit,
            /// <summary>
            /// The player makes a move on the board.
            /// </summary>
            Move,
        }

        /// <summary>
        /// Make the player's move based on their input.
        ///
        /// Original BASIC: 500-620
        /// </summary>
        /// <returns>Whether the player moved or quit the program.</returns>
        private PlayerAction PlayerMove()
        {
            // Loop until a valid move is inputted.
            while (true)
            {
                var move = ReadMove();
                if (move == 1)
                {
                    return PlayerAction.Quit;
                }
                else if (move == 0)
                {
                    ShowBoard();
                }
                else
                {
                    ClearPotentialMoves();
                    if (TryCoordToIndex(move, out int moveIndex))
                    {
                        if (Board[moveIndex] == EMPTY)
                        {
                            Board[moveIndex] = PLAYER;
                            return PlayerAction.Move;
                        }
                        else
                        {
                            Console.WriteLine("THAT SQUARE IS USED, TRY AGAIN.");
                        }
                    }
                    else
                    {
                        Console.WriteLine("INCORRECT MOVE, TRY AGAIN.");
                    }
                }
            }
        }

        /// <summary>
        /// Read a player move from the terminal. Move can be any integer.
        ///
        /// Original BASIC: 510-520
        /// </summary>
        /// <returns>the move inputted</returns>
        private static int ReadMove()
        {
            Console.Write("YOUR MOVE? ");
            return ReadInteger();
        }

        /// <summary>
        /// Read an integer from the terminal.
        ///
        /// Original BASIC: 520
        ///
        /// Unlike the basic, this code will not accept any string that starts
        ///  with a number; only full number strings are allowed.
        /// </summary>
        /// <returns>the integer inputted</returns>
        private static int ReadInteger()
        {
            while (true)
            {
                var response = Console.ReadLine() ?? " ";

                if (int.TryParse(response, out var move))
                {
                    return move;

                }
                else
                {
                    Console.Write("!NUMBER EXPECTED - RETRY INPUT LINE--? ");
                }
            }
        }

        /// <summary>
        /// Display the board to the player. Spaces taken by the player are
        ///  marked with "Y", while machine spaces are marked with "M".
        ///
        /// Original BASIC: 2550-2740
        /// </summary>
        private void ShowBoard()
        {
            var s = new StringBuilder(new string('\n', 9));

            for (int i = 1; i <= 4; i++)
            {
                for (int j = 1; j <= 4; j++)
                {
                    s.Append(' ', 3 * (j + 1));
                    for (int k = 1; k <= 4; k++)
                    {
                        int q = (16 * i) + (4 * j) + k - 21;
                        s.Append(Board[q] switch
                        {
                            EMPTY or POTENTIAL => "( )      ",
                            PLAYER => "(Y)      ",
                            MACHINE => "(M)      ",
                            _ => throw new Exception($"invalid space value {Board[q]}"),
                        });
                    }
                    s.Append("\n\n");
                }
                s.Append("\n\n");
            }

            Console.WriteLine(s.ToString());
        }

        #endregion

        /***********************************************************************
        /* Machine Move
        /**********************************************************************/
        #region MachineMove

        /// <summary>
        /// Check all rows for a player win.
        ///
        /// A row indicates a player win if its sum = PLAYER * 4.
        ///
        /// Original BASIC: 720-780
        /// </summary>
        /// <returns>whether the player won in any row</returns>
        private bool CheckPlayerWin()
        {
            for (int row = 0; row < 76; row++)
            {
                if (RowSums[row] == (PLAYER * 4))
                {
                    // Found player win!
                    Console.WriteLine("YOU WIN AS FOLLOWS");
                    DisplayRow(row);
                    return true;
                }
            }

            // No player win found.
            return false;
        }

        /// <summary>
        /// Check all rows for a row that the machine could move to to win
        ///  immediately.
        ///
        /// A row indicates a player could win immediately if it has three
        ///  machine moves already; that is, sum = MACHINE * 3.
        ///
        /// Original Basic: 790-920
        /// </summary>
        /// <returns></returns>
        private bool CheckMachineWin()
        {
            for (int row = 0; row < 76; row++)
            {
                if (RowSums[row] == (MACHINE * 3))
                {
                    // Found a winning row!
                    for (int space = 0; space < 4; space++)
                    {
                        int move = RowsByPlane[row, space];
                        if (Board[move] == EMPTY)
                        {
                            // Found empty space in winning row; move there.
                            Board[move] = MACHINE;
                            Console.WriteLine($"MACHINE MOVES TO {IndexToCoord(move)} , AND WINS AS FOLLOWS");
                            DisplayRow(row);
                            return true;
                        }
                    }
                }
            }

            // No winning row available.
            return false;
        }

        /// <summary>
        /// Display the coordinates of a winning row.
        /// </summary>
        /// <param name="row">index into RowsByPlane data</param>
        private void DisplayRow(int row)
        {
            for (int space = 0; space < 4; space++)
            {
                Console.Write($" {IndexToCoord(RowsByPlane[row, space])} ");
            }
            Console.WriteLine();
        }

        /// <summary>
        /// Possible actions machine can take in a move. This helps replace the
        ///  complex GOTO logic from the original BASIC, which allowed the
        ///  program to jump from the machine's action to the end of the game.
        /// </summary>
        private enum MachineAction
        {
            /// <summary>
            /// Machine did not take any action.
            /// </summary>
            None,
            /// <summary>
            /// Machine made a move.
            /// </summary>
            Move,
            /// <summary>
            /// Machine either won, conceded, or found a draw.
            /// </summary>
            End,
        }

        /// <summary>
        /// Machine decides where to move on the board, and ends the game if
        ///  appropriate.
        ///
        /// The machine's AI tries to take the following actions (in order):
        ///
        ///  1. If the player has a row that will get them the win on their
        ///     next turn, block that row.
        ///  2. If the machine can trap the player (create two different rows
        ///     with three machine moves each that cannot be blocked by only a
        ///     single player move, create such a trap.
        ///  3. If the player can create a similar trap for the machine on
        ///     their next move, block the space where that trap would be
        ///     created.
        ///  4. Find a plane in the board that is well-populated by player
        ///     moves, and take a space in the first such plane.
        ///  5. Find the first open corner or center and move there.
        ///  6. Find the first open space and move there.
        ///
        /// If none of these actions are possible, then the board is entirely
        ///  full, and the game results in a draw.
        ///
        /// Original BASIC: start at 930
        /// </summary>
        /// <returns>the action the machine took</returns>
        private MachineAction MachineMove()
        {
            // The actions the machine attempts to take, in order.
            var actions = new Func<MachineAction>[]
            {
                BlockPlayer,
                MakePlayerTrap,
                BlockMachineTrap,
                MoveByPlane,
                MoveCornerOrCenter,
                MoveAnyOpenSpace,
            };

            foreach (var action in actions)
            {
                // Try each action, moving to the next if nothing happens.
                var actionResult = action();
                if (actionResult != MachineAction.None)
                {
                    // Not in original BASIC: check for draw after each machine
                    // move.
                    if (CheckDraw())
                    {
                        return DrawGame();
                    }
                    return actionResult;
                }
            }

            // If we got here, all spaces are taken. Draw the game.
            return DrawGame();
        }

        /// <summary>
        /// Block a row with three spaces already taken by the player.
        ///
        /// Original BASIC: 930-1010
        /// </summary>
        /// <returns>
        /// Move if the machine blocked,
        /// None otherwise
        /// </returns>
        private MachineAction BlockPlayer()
        {
            for (int row = 0; row < 76; row++)
            {
                if (RowSums[row] == (PLAYER * 3))
                {
                    // Found a row to block on!
                    for (int space = 0; space < 4; space++)
                    {
                        if (Board[RowsByPlane[row, space]] == EMPTY)
                        {
                            // Take the remaining empty space.
                            Board[RowsByPlane[row, space]] = MACHINE;
                            Console.WriteLine($"NICE TRY. MACHINE MOVES TO {IndexToCoord(RowsByPlane[row, space])}");
                            return MachineAction.Move;
                        }
                    }
                }
            }

            // Didn't find a row to block on.
            return MachineAction.None;
        }

        /// <summary>
        /// Create a trap for the player if possible. A trap can be created if
        ///  moving to a space on the board results in two different rows having
        ///  three MACHINE spaces, with the remaining space not shared between
        ///  the two rows. The player can only block one of these traps, so the
        ///  machine will win.
        ///
        /// If a player trap is not possible, but a row is found that is
        ///  particularly advantageous for the machine to move to, the machine
        ///  will try and move to a plane edge in that row.
        ///
        /// Original BASIC: 1300-1480
        ///
        /// Lines 1440/50 of the BASIC call 2360 (MovePlaneEdge). Because it
        ///  goes to this code only after it has found an open space marked as
        ///  potential, it cannot reach line 2440 of that code, as that is only
        ///  reached if an open space failed to be found in the row on which
        ///  that code was called.
        /// </summary>
        /// <returns>
        /// Move if a trap was created,
        /// End if the machine conceded,
        /// None otherwise
        /// </returns>
        private MachineAction MakePlayerTrap()
        {
            for (int row = 0; row < 76; row++)
            {
                // Refresh row sum, since new POTENTIALs might have changed it.
                var rowSum = RefreshRowSum(row);

                // Machine has moved in this row twice, and player has not moved
                // in this row.
                if (rowSum >= (MACHINE * 2) && rowSum < (MACHINE * 2) + 1)
                {
                    // Machine has no potential moves yet in this row.
                    if (rowSum == (MACHINE * 2))
                    {
                        for (int space = 0; space < 4; space++)
                        {
                            // Empty space can potentially be used to create a
                            // trap.
                            if (Board[RowsByPlane[row, space]] == EMPTY)
                            {
                                Board[RowsByPlane[row, space]] = POTENTIAL;
                            }
                        }
                    }
                    // Machine has already found a potential move in this row,
                    // so a trap can be created with another row.
                    else
                    {
                        return MakeOrBlockTrap(row);
                    }
                }
            }

            // No player traps can be made.
            RefreshRowSums();

            for (int row = 0; row < 76; row++)
            {
                // A row may be particularly advantageous for the machine to
                // move to at this point; this is the case if a row is entirely
                // filled with POTENTIAL or has one MACHINE and others
                // POTENTIAL. Such rows may help set up trapping opportunities.
                if (RowSums[row] == (POTENTIAL * 4) || RowSums[row] == MACHINE + (POTENTIAL * 3))
                {
                    // Try moving to a plane edge in an advantageous row.
                    return MovePlaneEdge(row, POTENTIAL);
                }
            }

            // No spaces found that are particularly advantageous to machine.
            ClearPotentialMoves();
            return MachineAction.None;
        }

        /// <summary>
        /// Block a trap that the player could create for the machine on their
        ///  next turn.
        ///
        /// If there are no player traps to block, but a row is found that is
        ///  particularly advantageous for the player to move to, the machine
        ///  will try and move to a plane edge in that row.
        ///
        /// Original BASIC: 1030-1190
        ///
        /// Lines 1160/1170 of the BASIC call 2360 (MovePlaneEdge). As with
        ///  MakePlayerTrap, because it goes to this code only after it has
        ///  found an open space marked as potential, it cannot reach line 2440
        ///  of that code, as that is only reached if an open space failed to be
        ///  found in the row on which that code was called.
        /// </summary>
        /// <returns>
        /// Move if a trap was created,
        /// End if the machine conceded,
        /// None otherwise
        /// </returns>
        private MachineAction BlockMachineTrap()
        {
            for (int i = 0; i < 76; i++)
            {
                // Refresh row sum, since new POTENTIALs might have changed it.
                var rowSum = RefreshRowSum(i);

                // Player has moved in this row twice, and machine has not moved
                // in this row.
                if (rowSum >= (PLAYER * 2) && rowSum < (PLAYER * 2) + 1)
                {
                    // Machine has no potential moves yet in this row.
                    if (rowSum == (PLAYER * 2))
                    {
                        for (int j = 0; j < 4; j++)
                        {
                            if (Board[RowsByPlane[i, j]] == EMPTY)
                            {
                                Board[RowsByPlane[i, j]] = POTENTIAL;
                            }
                        }
                    }
                    // Machine has already found a potential move in this row,
                    // so a trap can be created with another row by the player.
                    // Move to block.
                    else
                    {
                        return MakeOrBlockTrap(i);
                    }
                }
            }

            // No player traps to block found.
            RefreshRowSums();

            for (int row = 0; row < 76; row++)
            {
                // A row may be particularly advantageous for the player to move
                // to at this point, indicated by a row containing all POTENTIAL
                // moves or one PLAYER and rest POTENTIAL. Such rows may aid in
                // in the later creation of traps.
                if (RowSums[row] == (POTENTIAL * 4) || RowSums[row] == PLAYER + (POTENTIAL * 3))
                {
                    // Try moving to a plane edge in an advantageous row.
                    return MovePlaneEdge(row, POTENTIAL);
                }
            }

            // No spaces found that are particularly advantageous to the player.
            return MachineAction.None;
        }

        /// <summary>
        /// Either make a trap for the player or block a trap the player could
        ///  create on their next turn.
        ///
        /// Unclear how this method could possibly end with a concession; it
        ///  seems it can only be called if the row contains a potential move.
        ///
        /// Original BASIC: 2230-2350
        /// </summary>
        /// <param name="row">the row containing the space to move to</param>
        /// <returns>
        /// Move if the machine moved,
        /// End if the machine conceded
        /// </returns>
        private MachineAction MakeOrBlockTrap(int row)
        {
            for (int space = 0; space < 4; space++)
            {
                if (Board[RowsByPlane[row, space]] == POTENTIAL)
                {
                    Board[RowsByPlane[row, space]] = MACHINE;

                    // Row sum indicates we're blocking a player trap.
                    if (RowSums[row] < MACHINE)
                    {
                        Console.Write("YOU FOX.  JUST IN THE NICK OF TIME, ");
                    }
                    // Row sum indicates we're completing a machine trap.
                    else
                    {
                        Console.Write("LET'S SEE YOU GET OUT OF THIS:  ");
                    }

                    Console.WriteLine($"MACHINE MOVES TO {IndexToCoord(RowsByPlane[row, space])}");

                    return MachineAction.Move;
                }
            }

            // Unclear how this can be reached.
            Console.WriteLine("MACHINE CONCEDES THIS GAME.");
            return MachineAction.End;
        }

        /// <summary>
        /// Find a satisfactory plane on the board and move to one if that
        ///  plane's plane edges.
        ///
        /// A plane on the board is satisfactory if it meets the following
        ///  conditions:
        ///     1. Player has made exactly 4 moves on the plane.
        ///     2. Machine has made either 0 or one moves on the plane.
        ///  Such a plane is one that the player could likely use to form traps.
        ///
        /// Original BASIC: 1830-2020
        ///
        /// Line 1990 of the original basic calls 2370 (MovePlaneEdge). Only on
        ///  this call to MovePlaneEdge can line 2440 of that method be reached,
        ///  which surves to help this method iterate through the rows of a
        ///  plane.
        /// </summary>
        /// <returns>
        /// Move if a move in a plane was found,
        /// None otherwise
        /// </returns>
        private MachineAction MoveByPlane()
        {
            // For each plane in the cube...
            for (int plane = 1; plane <= 18; plane++)
            {
                double planeSum = PlaneSum(plane);

                // Check that plane sum satisfies condition.
                const double P4 = PLAYER * 4;
                const double P4_M1 = (PLAYER * 4) + MACHINE;
                if (
                    (planeSum >= P4 && planeSum < P4 + 1) ||
                    (planeSum >= P4_M1 && planeSum < P4_M1 + 1)
                )
                {
                    // Try to move to plane edges in each row of plane
                    // First, check for plane edges marked as POTENTIAL.
                    for (int row = (4 * plane) - 4; row < (4 * plane); row++)
                    {
                        var moveResult = MovePlaneEdge(row, POTENTIAL);
                        if (moveResult != MachineAction.None)
                        {
                            return moveResult;
                        }
                    }

                    // If no POTENTIAL plane edge found, look for an EMPTY one.
                    for (int row = (4 * plane) - 4; row < (4 * plane); row++)
                    {
                        var moveResult = MovePlaneEdge(row, EMPTY);
                        if (moveResult != MachineAction.None)
                        {
                            return moveResult;
                        }
                    }
                }
            }

            // No satisfactory planes with open plane edges found.
            ClearPotentialMoves();
            return MachineAction.None;
        }

        /// <summary>
        /// Given a row, move to the first space in that row that:
        ///  1. is a plane edge, and
        ///  2. has the given value in Board
        ///
        /// Plane edges are any spaces on a plane with one face exposed. The AI
        ///  prefers to move to these spaces before others, presumably
        ///  because they are powerful moves: a plane edge is contained on 3-4
        ///  winning rows of the cube.
        ///
        /// Original BASIC: 2360-2490
        ///
        /// In the original BASIC, this code is pointed to from three different
        ///  locations by GOTOs:
        ///  - 1440/50, or MakePlayerTrap;
        ///  - 1160/70, or BlockMachineTrap; and
        ///  - 1990, or MoveByPlane.
        /// At line 2440, this code jumps back to line 2000, which is in
        ///  MoveByPlane. This makes it appear as though calling MakePlayerTrap
        ///  or BlockPlayerTrap in the BASIC could jump into the middle of the
        ///  MoveByPlane method; were this to happen, not all of MoveByPlane's
        ///  variables would be defined! However, the program logic prevents
        ///  this from ever occurring; see each method's description for why
        ///  this is the case.
        /// </summary>
        /// <param name="row">the row to try to move to</param>
        /// <param name="spaceValue">
        /// what value the space to move to should have in Board
        /// </param>
        /// <returns>
        /// Move if a plane edge piece in the row with the given spaceValue was
        /// found,
        /// None otherwise
        /// </returns>
        private MachineAction MovePlaneEdge(int row, double spaceValue)
        {
            // Given a row, we want to find the plane edge pieces in that row.
            // We know that each row is part of a plane, and that the first
            // and last rows of the plane are on the plane edge, while the
            // other two rows are in the middle. If we know whether a row is an
            // edge or middle, we can determine which spaces in that row are
            // plane edges.
            //
            // Below is a birds-eye view of a plane in the cube, with rows
            // oriented horizontally:
            //
            //   row 0: ( ) (1) (2) ( )
            //   row 1: (0) ( ) ( ) (3)
            //   row 2: (0) ( ) ( ) (3)
            //   row 3: ( ) (1) (2) ( )
            //
            // The plane edge pieces have their row indices marked. The pattern
            // above shows that:
            //
            //  if row == 0 | 3, plane edge spaces = [1, 2]
            //  if row == 1 | 2, plane edge spaces = [0, 3]

            // The below condition replaces the following BASIC code (2370):
            //
            //  I-(INT(I/4)*4)>1
            //
            // which in C# would be:
            //
            //
            // int a = i - (i / 4) * 4 <= 1)
            //     ? 1
            //     : 2;
            //
            // In the above, i is the one-indexed row in RowsByPlane.
            //
            // This condition selects a different a value based on whether the
            // given row is on the edge or middle of its plane.
            int a = (row % 4) switch
            {
                0 or 3 => 1,  // row is on edge of plane
                1 or 2 => 2,  // row is in middle of plane
                _ => throw new Exception($"unreachable ({row % 4})"),
            };

            // Iterate through plane edge pieces of the row.
            //
            //  if a = 1 (row is edge), iterate through [0, 3]
            //  if a = 2 (row is middle), iterate through [1, 2]
            for (int space = a - 1; space <= 4 - a; space += 5 - (2 * a))
            {
                if (Board[RowsByPlane[row, space]] == spaceValue)
                {
                    // Found a plane edge to take!
                    Board[RowsByPlane[row, space]] = MACHINE;
                    Console.WriteLine($"MACHINE TAKES {IndexToCoord(RowsByPlane[row, space])}");
                    return MachineAction.Move;
                }
            }

            // No valid corner edge to take.
            return MachineAction.None;
        }

        /// <summary>
        /// Find the first open corner or center in the board and move there.
        ///
        /// Original BASIC: 1200-1290
        ///
        /// This is the only place where the Z variable from the BASIC code is
        ///  used; here it is implied in the for loop.
        /// </summary>
        /// <returns>
        /// Move if an open corner/center was found and moved to,
        /// None otherwise
        /// </returns>
        private MachineAction MoveCornerOrCenter()
        {
            foreach (int space in CornersAndCenters)
            {
                if (Board[space] == EMPTY)
                {
                    Board[space] = MACHINE;
                    Console.WriteLine($"MACHINE MOVES TO {IndexToCoord(space)}");
                    return MachineAction.Move;
                }
            }

            return MachineAction.None;
        }

        /// <summary>
        /// Find the first open space in the board and move there.
        ///
        /// Original BASIC: 1720-1800
        /// </summary>
        /// <returns>
        /// Move if an open space was found and moved to,
        /// None otherwise
        /// </returns>
        private MachineAction MoveAnyOpenSpace()
        {
            for (int row = 0; row < 64; row++)
            {
                if (Board[row] == EMPTY)
                {
                    Board[row] = MACHINE;
                    Console.WriteLine($"MACHINE LIKES {IndexToCoord(row)}");
                    return MachineAction.Move;
                }
            }
            return MachineAction.None;
        }

        /// <summary>
        /// Draw the game in the event that there are no open spaces.
        ///
        /// Original BASIC: 1810-1820
        /// </summary>
        /// <returns>End</returns>
        private MachineAction DrawGame()
        {
            Console.WriteLine("THIS GAME IS A DRAW.");
            return MachineAction.End;
        }

        #endregion

        /***********************************************************************
        /* Helpers
        /**********************************************************************/
        #region Helpers

        /// <summary>
        /// Attempt to transform a cube coordinate to an index into Board.
        ///
        /// A valid cube coordinate is a three-digit number, where each digit
        ///  of the number X satisfies 1 <= X <= 4.
        ///
        /// Examples:
        ///  111 -> 0
        ///  444 -> 63
        ///  232 -> 35
        ///
        /// If the coord provided is not valid, the transformation fails.
        ///
        /// The conversion from coordinate to index is essentially a conversion
        ///  between base 4 and base 10.
        ///
        /// Original BASIC: 525-580
        ///
        /// This method fixes a bug in the original BASIC (525-526), which only
        ///  checked whether the given coord satisfied 111 <= coord <= 444. This
        ///  allows invalid coordinates such as 199 and 437, whose individual
        ///  digits are out of range.
        /// </summary>
        /// <param name="coord">cube coordinate (e.g. "111", "342")</param>
        /// <param name="index">trasnformation output</param>
        /// <returns>
        /// true if the transformation was successful, false otherwise
        /// </returns>
        private static bool TryCoordToIndex(int coord, out int index)
        {
            // parse individual digits, subtract 1 to get base 4 number
            var hundreds = (coord / 100) - 1;
            var tens = ((coord % 100) / 10) - 1;
            var ones = (coord % 10) - 1;

            // bounds check for each digit
            foreach (int digit in new int[] { hundreds, tens, ones })
            {
                if (digit < 0 || digit > 3)
                {
                    index = -1;
                    return false;
                }
            }

            // conversion from base 4 to base 10
            index = (16 * hundreds) + (4 * tens) + ones;
            return true;
        }

        /// <summary>
        /// Transform a Board index into a valid cube coordinate.
        ///
        /// Examples:
        ///  0 -> 111
        ///  63 -> 444
        ///  35 -> 232
        ///
        /// The conversion from index to coordinate is essentially a conversion
        ///  between base 10 and base 4.
        ///
        /// Original BASIC: 1570-1610
        /// </summary>
        /// <param name="index">Board index</param>
        /// <returns>the corresponding cube coordinate</returns>
        private static int IndexToCoord(int index)
        {
            // check that index is valid
            if (index < 0 || index > 63)
            {
                // runtime exception; all uses of this method are with
                // indices provided by the program, so this should never fail
                throw new Exception($"index {index} is out of range");
            }

            // convert to base 4, add 1 to get cube coordinate
            var hundreds = (index / 16) + 1;
            var tens = ((index % 16) / 4) + 1;
            var ones = (index % 4) + 1;

            // concatenate digits
            int coord = (hundreds * 100) + (tens * 10) + ones;
            return coord;
        }

        /// <summary>
        /// Refresh the values in RowSums to account for any changes.
        ///
        /// Original BASIC: 1640-1710
        /// </summary>
        private void RefreshRowSums()
        {
            for (var row = 0; row < 76; row++)
            {
                RefreshRowSum(row);
            }
        }

        /// <summary>
        /// Refresh a row in RowSums to reflect changes.
        /// </summary>
        /// <param name="row">row in RowSums to refresh</param>
        /// <returns>row sum after refresh</returns>
        private double RefreshRowSum(int row)
        {
            double rowSum = 0;
            for (int space = 0; space < 4; space++)
            {
                rowSum += Board[RowsByPlane[row, space]];
            }
            RowSums[row] = rowSum;
            return rowSum;
        }

        /// <summary>
        /// Calculate the sum of spaces in one of the 18 cube planes in RowSums.
        ///
        /// Original BASIC: 1840-1890
        /// </summary>
        /// <param name="plane">the desired plane</param>
        /// <returns>sum of spaces in plane</returns>
        private double PlaneSum(int plane)
        {
            double planeSum = 0;
            for (int row = (4 * (plane - 1)); row < (4 * plane); row++)
            {
                for (int space = 0; space < 4; space++)
                {
                    planeSum += Board[RowsByPlane[row, space]];
                }
            }
            return planeSum;
        }

        /// <summary>
        /// Check whether the board is in a draw state, that is all spaces are
        ///  full and neither the player nor the machine has won.
        ///
        /// The original BASIC contains a bug that if the player moves first, a
        ///  draw will go undetected. An example series of player inputs
        ///  resulting in such a draw (assuming player goes first):
        ///
        ///  114, 414, 144, 444, 122, 221, 112, 121,
        ///  424, 332, 324, 421, 231, 232, 244, 311,
        ///  333, 423, 331, 134, 241, 243, 143, 413,
        ///  142, 212, 314, 341, 432, 412, 431, 442
        /// </summary>
        /// <returns>whether the game is a draw</returns>
        private bool CheckDraw()
        {
            for (var i = 0; i < 64; i++)
            {
                if (Board[i] != PLAYER && Board[i] != MACHINE)
                {
                    return false;
                }
            }

            RefreshRowSums();

            for (int row = 0; row < 76; row++)
            {
                var rowSum = RowSums[row];
                if (rowSum == PLAYER * 4 || rowSum == MACHINE * 4)
                {
                    return false;
                }
            }


            return true;
        }

        /// <summary>
        /// Reset POTENTIAL spaces in Board to EMPTY.
        ///
        /// Original BASIC: 2500-2540
        /// </summary>
        private void ClearPotentialMoves()
        {
            for (var i = 0; i < 64; i++)
            {
                if (Board[i] == POTENTIAL)
                {
                    Board[i] = EMPTY;
                }
            }
        }

        /// <summary>
        /// Reset all spaces in Board to EMPTY.
        ///
        /// Original BASIC: 400-420
        /// </summary>
        private void ClearBoard()
        {
            for (var i = 0; i < 64; i++)
            {
                Board[i] = EMPTY;
            }
        }

        #endregion
    }
}

```