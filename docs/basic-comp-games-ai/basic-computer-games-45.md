# BasicComputerGames源码解析 45

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Gunner

GUNNER allows you to adjust the fire of a field artillery weapon to hit a stationary target. You specify the number of degrees of elevation of your weapon; 45 degrees provides maximum range with values under or over 45 degrees providing less range.

You get up to five shots to destroy the enemy before he destroys you. Gun range varies between 20,000 and 60,000 yards and burst radius is 100 yards. You must specify elevation within approximately 0.2 degrees to get a hit.

Tom Kloos of the Oregon Museum of Science and Industry in Portland, Oregon originally wrote GUNNER. Extensive modifications were added by David Ahl.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=77)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=92)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `42_Gunner/csharp/Program.cs`

This is a class written in C# that appears to be a simple game where the player is given an order to place a projectile on a target at a certain elevation and then it will determine if the target has been destroyed or not.

The class has several static methods, including one that calculates the difference between the target and impact based on the elevation, another that prints the return to base, and another that prints the initial introduction.

The class also has two methods, one that gets the elevation from the player and the other that prints the order to the player.

It is also worth noting that the class uses a variable named "weirdNumber" which is calculated as 2 multiplied by the elevation divided by 57.3, this number is used to calculate the distance to the target, and it also uses "Math.Sin()" function to calculate the distance shot.


```
﻿namespace Gunner
{
    class Program
    {
        static void Main(string[] args)
        {
            PrintIntro();

            string keepPlaying = "Y";

            while (keepPlaying == "Y") {
                PlayGame();
                Console.WriteLine("TRY AGAIN (Y OR N)");
                keepPlaying = Console.ReadLine();
            }
        }

        static void PlayGame()
        {
            int totalAttempts = 0;
            int amountOfGames = 0;

            while (amountOfGames < 4) {

                int maximumRange = new Random().Next(0, 40000) + 20000;
                Console.WriteLine($"MAXIMUM RANGE OF YOUR GUN IS {maximumRange} YARDS." + Environment.NewLine + Environment.NewLine + Environment.NewLine);

                int distanceToTarget = (int) (maximumRange * (0.1 + 0.8 * new Random().NextDouble()));
                Console.WriteLine($"DISTANCE TO THE TARGET IS {distanceToTarget} YARDS.");

                (bool gameWon, int attempts) = HitTheTarget(maximumRange, distanceToTarget);

                if(!gameWon) {
                    Console.WriteLine(Environment.NewLine + "BOOM !!!!   YOU HAVE JUST BEEN DESTROYED" + Environment.NewLine +
                        "BY THE ENEMY." + Environment.NewLine + Environment.NewLine + Environment.NewLine
                    );
                    PrintReturnToBase();
                    break;
                } else {
                    amountOfGames += 1;
                    totalAttempts += attempts;

                    Console.WriteLine($"TOTAL ROUNDS EXPENDED WERE:{totalAttempts}");

                    if (amountOfGames < 4) {
                        Console.WriteLine("THE FORWARD OBSERVER HAS SIGHTED MORE ENEMY ACTIVITY...");
                    } else {
                        if (totalAttempts > 18) {
                            PrintReturnToBase();
                        } else {
                            Console.WriteLine($"NICE SHOOTING !!");
                        }
                    }
                }
            }
        }

        static (bool, int) HitTheTarget(int maximumRange, int distanceToTarget)
        {
            int attempts = 0;

            while (attempts < 6)
            {
                int elevation = GetElevation();

                int differenceBetweenTargetAndImpact = CalculateDifferenceBetweenTargetAndImpact(maximumRange, distanceToTarget, elevation);

                if (Math.Abs(differenceBetweenTargetAndImpact) < 100)
                {
                    Console.WriteLine($"*** TARGET DESTROYED *** {attempts} ROUNDS OF AMMUNITION EXPENDED.");
                    return (true, attempts);
                }
                else if (differenceBetweenTargetAndImpact > 100)
                {
                    Console.WriteLine($"OVER TARGET BY {Math.Abs(differenceBetweenTargetAndImpact)} YARDS.");
                }
                else
                {
                    Console.WriteLine($"SHORT OF TARGET BY {Math.Abs(differenceBetweenTargetAndImpact)} YARDS.");
                }

                attempts += 1;
            }
            return (false, attempts);
        }

        static int CalculateDifferenceBetweenTargetAndImpact(int maximumRange, int distanceToTarget, int elevation)
        {
            double weirdNumber = 2 * elevation / 57.3;
            double distanceShot = maximumRange * Math.Sin(weirdNumber);
            return (int)distanceShot - distanceToTarget;
        }

        static void PrintReturnToBase()
        {
            Console.WriteLine("BETTER GO BACK TO FORT SILL FOR REFRESHER TRAINING!");
        }

        static int GetElevation()
        {
            Console.WriteLine("ELEVATION");
            int elevation = int.Parse(Console.ReadLine());
            if (elevation > 89) {
                Console.WriteLine("MAXIMUM ELEVATION IS 89 DEGREES");
                return GetElevation();
            }
            if (elevation < 1) {
                Console.WriteLine("MINIMUM ELEVATION IS 1 DEGREE");
                return GetElevation();
            }
            return elevation;
        }

        static void PrintIntro()
        {
            Console.WriteLine(new String(' ', 30) + "GUNNER");
            Console.WriteLine(new String(' ', 15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY" + Environment.NewLine + Environment.NewLine + Environment.NewLine);
            Console.WriteLine("YOU ARE THE OFFICER-IN-CHARGE, GIVING ORDERS TO A GUN");
            Console.WriteLine("CREW, TELLING THEM THE DEGREES OF ELEVATION YOU ESTIMATE");
            Console.WriteLine("WILL PLACE A PROJECTILE ON TARGET.  A HIT WITHIN 100 YARDS");
            Console.WriteLine("OF THE TARGET WILL DESTROY IT." + Environment.NewLine);
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `42_Gunner/java/Gunner.java`

This appears to be a Java program that simulates a game of "Warcraft III" where players are dropped into a map and must fight their way through waves of enemies to reach a target point. The player is provided with a limited amount of ammunition to fight against the enemy, and each time the player is hit, the program simulates the damage the player has taken and the number of rounds of ammunition used.

The program has several functions, including a loop that allows the player to continue fighting until they have been defeated or reached the target. The program also has a function that calculates the error distance between the player's position and the target, and a function that prints out the number of rounds of ammunition used.

The program also has a println() function which prints out a string of text, this is used to display the player's progress in the game.


```
import java.util.Random;
import java.util.Scanner;

public class Gunner {

    public static final int MAX_ROUNDS = 6;
    public static final int MAX_ENEMIES = 4;
    public static final int ERROR_DISTANCE = 100;

    private static Scanner scanner = new Scanner(System.in);
    private static Random random = new Random();

    public static void main(String[] args) {
        println("                              GUNNER");
        println("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        println();
        println();
        println();
        println("YOU ARE THE OFFICER-IN-CHARGE, GIVING ORDERS TO A GUN");
        println("CREW, TELLING THEM THE DEGREES OF ELEVATION YOU ESTIMATE");
        println("WILL PLACE A PROJECTILE ON TARGET.  A HIT WITHIN " + ERROR_DISTANCE + " YARDS");
        println("OF THE TARGET WILL DESTROY IT.");
        println();
        while (true) {
            int maxRange = random.nextInt(40000) + 20000;
            int enemyCount = 0;
            int totalRounds = 0;
            println("MAXIMUM RANGE OF YOUR GUN IS " + maxRange + " YARDS.\n");

            while (true) {
                int rounds = fightEnemy(maxRange);
                totalRounds += rounds;

                if (enemyCount == MAX_ENEMIES || rounds >= MAX_ROUNDS) {
                    if (rounds < MAX_ROUNDS) {
                        println("\n\n\nTOTAL ROUNDS EXPENDED WERE:" + totalRounds);
                    }
                    if (totalRounds > 18 || rounds >= MAX_ROUNDS) {
                        println("BETTER GO BACK TO FORT SILL FOR REFRESHER TRAINING!");
                    } else {
                        println("NICE SHOOTING !!");
                    }
                    println("\nTRY AGAIN (Y OR N)");
                    String tryAgainResponse = scanner.nextLine();
                    if ("Y".equals(tryAgainResponse) || "y".equals(tryAgainResponse)) {
                        break;
                    }
                    println("\nOK.  RETURN TO BASE CAMP.");
                    return;
                }
                enemyCount++;
                println("\nTHE FORWARD OBSERVER HAS SIGHTED MORE ENEMY ACTIVITY...");
            }
        }
    }

    private static int fightEnemy(int maxRange) {
        int rounds = 0;
        long target = Math.round(maxRange * (random.nextDouble() * 0.8 + 0.1));
        println("      DISTANCE TO THE TARGET IS " + target + " YARDS.");

        while (true) {
            println("\nELEVATION?");
            double elevation = Double.parseDouble(scanner.nextLine());
            if (elevation > 89.0) {
                println("MAXIMUM ELEVATION IS 89 DEGREES.");
                continue;
            }
            if (elevation < 1.0) {
                println("MINIMUM ELEVATION IS ONE DEGREE.");
                continue;
            }
            rounds++;
            if (rounds >= MAX_ROUNDS) {
                println("\nBOOM !!!!   YOU HAVE JUST BEEN DESTROYED ");
                println("BY THE ENEMY.\n\n\n");
                break;
            }

            long error = calculateError(maxRange, target, elevation);
            if (Math.abs(error) < ERROR_DISTANCE) {
                println("*** TARGET DESTROYED ***  " + rounds + " ROUNDS OF AMMUNITION EXPENDED.");
                break;
            } else if (error > ERROR_DISTANCE) {
                println("SHORT OF TARGET BY " + Math.abs(error) + " YARDS.");
            } else {
                println("OVER TARGET BY " + Math.abs(error) + " YARDS.");
            }

        }
        return rounds;
    }

    private static long calculateError(int maxRange, long target, double elevationInDegrees) {
        double elevationInRadians = Math.PI * elevationInDegrees / 90.0; //convert degrees to radians
        double impact = maxRange * Math.sin(elevationInRadians);
        double error = target - impact;
        return Math.round(error);
    }

    private static void println(String s) {
        System.out.println(s);
    }

    private static void println() {
        System.out.println();
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `42_Gunner/javascript/gunner.js`

这段代码定义了两个函数：`print()` 和 `input()`。

`print()` 函数的作用是在文档中创建一个输出元素（一个 `<textarea>` 元素），然后将其内容（一个字符串）添加到该元素中。这个字符串内容将以文本的形式在页面上显示。

`input()` 函数的作用是接收用户输入的字符串（输入框）。首先，它创建了一个新的输入元素（一个 `<input>` 元素），并将其设置为只能输入文本。然后，它将其添加到文档中的一个元素（可能是页面上的一个空文本框）。接着，它将设置一个 `keyup` 事件处理程序，以便在用户按下键盘上的数字键时接收输入。当用户点击数字键时，函数会获取当前输入的字符串，并以文本的形式在页面上显示。


```
// GUNNER
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

这段代码定义了一个名为 `tab` 的函数，它会接收一个参数 `space`，并返回一个字符串，该字符串由 `space` 变量中的数字组成，数字从 0 开始递增。函数中包含一个 while 循环，该循环从 `space` 变量中减去 1，当 `space` 的值大于 0 时，循环会继续执行。在循环体内，使用空格填充字符串，每次填充一个空格后，将 `space` 的值减去 1，直到 `space` 的值等于 0，循环停止。

在这段代码中，有四个调用 `tab` 函数的语句。这些语句分别传入不同的参数，第一个参数是 `30`，第二个参数是 `15`，第三个参数是空字符串，第四个参数是 `"YOU ARE THE OFFICER-IN-CHARGE, GIVING ORDERS TO A GUN"`。函数返回的字符串分别为 `"SPACE IS MAGICAL，喋亦有曼妙，\n"`、`"GUNNER IS A LAZY，慢惰的，\n"` 和空字符串 `""`。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

print(tab(30) + "GUNNER\n");
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
print("\n");
print("\n");
print("\n");
print("YOU ARE THE OFFICER-IN-CHARGE, GIVING ORDERS TO A GUN\n");
print("CREW, TELLING THEM THE DEGREES OF ELEVATION YOU ESTIMATE\n");
```

This appears to be a script for a video game where the player is a soldier fighting in a war. It appears to be a first-person shooter and the player is issueg a任务 to destroy enemy targets that havebeen identified. The player must maintain a certain minimum target completion rate, or the objective will not be achieved, and the player will be transported back to the base camp for treatment.
The player is given a limited amount of supplies, and must use them wisely to repair or upgrade their weapons and equipment. The player is also given a limited amount of health, and must use it wisely to heal themselves when necessary.
The player can also call in support from other players, and can use a variety of different weapons, including sniper rifles and machine guns.


```
print("WILL PLACE A PROJECTILE ON TARGET.  A HIT WITHIN 100 YARDS\n");
print("OF THE TARGET WILL DESTROY IT.\n");
print("\n");

// Main control section
async function main()
{
    while (1) {
        r = Math.floor(40000 * Math.random() + 20000);
        print("MAXIMUM RANGE OF YOUR GUN IS " + r + " YARDS.\n");
        z = 0;
        print("\n");
        s1 = 0;
        while (1) {
            t = Math.floor(r * (0.1 + 0.8 * Math.random()));
            s = 0;
            print("DISTANCE TO THE TARGET IS " + t + " YARDS.\n");
            print("\n");

            while (1) {
                print("\n");
                print("ELEVATION");
                b = parseFloat(await input());
                if (b > 89) {
                    print("MAXIMUM ELEVATION IS 89 DEGREES.\n");
                    continue;
                }
                if (b < 1) {
                    print("MINIMUM ELEVATION IS ONE DEGREE.\n");
                    continue;
                }
                if (++s >= 6) {
                    print("\n");
                    print("BOOM !!!!   YOU HAVE JUST BEEN DESTROYED BY THE ENEMY.\n");
                    print("\n");
                    print("\n");
                    print("\n");
                    e = 0;
                    break;
                }
                b2 = 2 * b / 57.3;
                i = r * Math.sin(b2);
                x = t - i;
                e = Math.floor(x);
                if (true) { //Math.abs(e) < 100) {
                    e = 1;
                    break;
                }
                if (e > 100) {
                    print("SHORT OF TARGET BY " + Math.abs(e) + " YARDS.\n");
                } else {
                    print("OVER TARGET BY " + Math.abs(e) + " YARDS.\n");
                }
            }
            if (e == 1) {
                print("*** TARGET DESTROYED *** " + s + " ROUNDS OF AMMUNITION EXPENDED.\n");
                s1 += s;
                if (z == 4) {
                    print("\n");
                    print("\n");
                    print("TOTAL ROUND EXPENDED WERE: " + s1 + "\n");
                    break;
                } else {
                    z++;
                    print("\n");
                    print("THE FORWARD OBSERVER HAS SIGHTED MORE ENEMY ACTIVITY...\n");
                }
            } else {
                s1 = 19;
                break;
            }
        }
        if (s1 > 18) {
            print("BETTER GO BACK TO FORT SILL FOR REFRESHER TRAINING!\n");
        } else {
            print("NICE SHOOTING !!");
        }
        print("\n");
        print("TRY AGAIN (Y OR N)");
        str = await input();
        if (str.substr(0, 1) != "Y")
            break;
    }
    print("\n");
    print("OK.  RETURN TO BASE CAMP.\n");
}

```

这道题的代码是 `main()`，它是 Python 应用程序的主函数。在一个 Python 应用程序中，`main()` 函数通常是 Python 脚本中 export 函数的前面部分。`export` 函数用于告诉 Python 解释器，哪些函数可以从该应用程序中导出，以便其他程序可以调用它们。

`main()` 函数的作用是，当这个脚本被独立地运行时，`export` 函数列出的函数将作为一个独立的模块可以被 Python 应用程序和模块调用。所以，当脚本被运行时，它将导出函数 `hello()`、`world()` 和 `sayhello()`。


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


# `42_Gunner/python/gunner.py`

It looks like you have provided the code for an update event in the game少女幻执事 Online. When this event is triggered, it checks the player's current elevation and shotgun range, and then it attempts to fire the gun at the enemy. If the player is able to successfully fire the gun, it will calculate the impact of the bullet on the enemy and calculate the distance to the target. Based on the distance to the target, it will determine if the enemy was destroyed or not. If the enemy is not destroyed, it will check if the player has any more ammunition, and if not, it will return to the fort.

It's important to note that this code is not complete and may not work as intended. For example, it does not check if the player has the necessary permissions to fire the gun, or if the enemy is not human. It also does not handle the case where the player does not have any more ammunition and the enemy is human.


```
#!/usr/bin/env python3
#
# Ported to Python by @iamtraction

from math import sin
from random import random


def gunner() -> None:
    gun_range = int(40000 * random() + 20000)

    print("\nMAXIMUM RANGE OF YOUR GUN IS", gun_range, "YARDS.")

    killed_enemies = 0
    S1 = 0

    while True:
        target_distance = int(gun_range * (0.1 + 0.8 * random()))
        shots = 0

        print("\nDISTANCE TO THE TARGET IS", target_distance, "YARDS.")

        while True:
            elevation = float(input("\n\nELEVATION? "))

            if elevation > 89:
                print("MAXIMUM ELEVATION IS 89 DEGREES.")
                continue

            if elevation < 1:
                print("MINIMUM ELEVATION IS ONE DEGREE.")
                continue

            shots += 1

            if shots < 6:
                B2 = 2 * elevation / 57.3
                shot_impact = gun_range * sin(B2)
                shot_proximity = target_distance - shot_impact
                shot_proximity_int = int(shot_proximity)

                if abs(shot_proximity_int) < 100:
                    print(
                        "*** TARGET DESTROYED *** ",
                        shots,
                        "ROUNDS OF AMMUNITION EXPENDED.",
                    )
                    S1 += shots
                    if killed_enemies == 4:
                        print("\n\nTOTAL ROUNDS EXPENDED WERE: ", S1)
                        if S1 > 18:
                            print("BETTER GO BACK TO FORT SILL FOR REFRESHER TRAINING!")
                            return
                        else:
                            print("NICE SHOOTING !!")
                            return
                    else:
                        killed_enemies += 1
                        print(
                            "\nTHE FORWARD OBSERVER HAS SIGHTED MORE ENEMY ACTIVITY..."
                        )
                        break
                else:
                    if shot_proximity_int > 100:
                        print("SHORT OF TARGET BY", abs(shot_proximity_int), "YARDS.")
                    else:
                        print("OVER TARGET BY", abs(shot_proximity_int), "YARDS.")
            else:
                print("\nBOOM !!!!   YOU HAVE JUST BEEN DESTROYED BY THE ENEMY.\n\n\n")
                print("BETTER GO BACK TO FORT SILL FOR REFRESHER TRAINING!")
                return


```

这段代码是一个Python程序，它定义了一个名为`main`的函数，该函数返回一个`None`。

在函数内部，首先打印了一些字符，然后又打印了一些字符，接着又打印了一些字符，接着又打印了一些字符，再接着又打印了一些字符，接着又打印了一些字符，最后又打印了一些字符。这些字符都是用` "`字符进行的，然后在字符串的末尾增加了一个`\n`字符。

接下来，定义了一个名为`gunner`的函数，该函数的作用是在控制台输出一些字符，并等待用户输入字符，如果用户输入的不是`Y`，则打印一些信息并返回。

在主函数中，首先调用`gunner`函数，然后在循环中再次调用`gunner`函数，并等待用户输入。如果用户输入的不是`Y`，则退出循环，打印一些信息并返回。如果用户输入的是`Y`，则返回到`main`函数的内部。


```
def main() -> None:
    print(" " * 33 + "GUNNER")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print("\n\n\n")
    print("YOU ARE THE OFFICER-IN-CHARGE, GIVING ORDERS TO A GUN")
    print("CREW, TELLING THEM THE DEGREES OF ELEVATION YOU ESTIMATE")
    print("WILL PLACE A PROJECTILE ON TARGET.  A HIT WITHIN 100 YARDS")
    print("OF THE TARGET WILL DESTROY IT.")

    while True:
        gunner()

        not_again = input("TRY AGAIN (Y OR N)? ").upper() != "Y"
        if not_again:
            print("\nOK.  RETURN TO BASE CAMP.")
            break


```

这段代码是一个if语句，它的作用是判断当前脚本是否作为主程序运行。如果当前脚本作为主程序运行，那么程序会执行if语句中的代码块。

在这个if语句中，有一个名叫__main__的常量，它是一个特殊的常量，只有在程序作为主程序运行时才会被赋予值。如果当前脚本作为主程序运行，那么__main__的值就是`"__main__"`，否则__main__的值就是`"__default__"`。

if语句块中的代码是程序的入口点，也就是程序从哪里开始执行。在这种情况下，如果当前脚本作为主程序运行，那么程序将从if语句块开始执行。如果当前脚本不是主程序，那么if语句块就不会被执行，程序将直接退出。


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


### Hammurabi

In this game you direct the administrator of Sumeria, Hammurabi, how to manage the city. The city initially has 1,000 acres, 100 people and 3,000 bushels of grain in storage.

You may buy and sell land with your neighboring city-states for bushels of grain — the price will vary between 17 and 26 bushels per acre. You also must use grain to feed your people and as seed to plant the next year’s crop.

You will quickly find that a certain number of people can only tend a certain amount of land and that people starve if they are not fed enough. You also have the unexpected to contend with such as a plague, rats destroying stored grain, and variable harvests.

You will also find that managing just the few resources in this game is not a trivial job over a period of say ten years. The crisis of population density rears its head very rapidly.

This program was originally written in Focal at DEC; author unknown. David Ahl converted it to BASIC and added the 10-year performance assessment. If you wish to change any of the factors, the extensive remarks in the program should make modification fairly straightforward.

Note for trivia buffs: somewhere along the line an m was dropped out of the spelling of Hammurabi in hte Ahl version of the computer program. This error has spread far and wide until a generation of students now think that Hammurabi is the incorrect spelling.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=78)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=93)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

- Though the file name and README both spell "Hammurabi" with two M's, the program itself consistently uses only one M.

#### External Links
 - C: https://github.com/beyonddream/hamurabi
 - Rust: https://github.com/beyonddream/hamurabi.rs


# `43_Hammurabi/csharp/ActionResult.cs`

这段代码定义了一个名为ActionResult的枚举类型，用于表示游戏中的不同行动结果。枚举类型中包含四个成员变量，分别表示行动成功、城市没有足够的粮食、城市没有足够的土地、城市没有足够的人口。每个成员变量都有一个默认的名称，分别为人性、粮食、土地和人口。

通过枚举类型的定义，这段代码可以被看作是一个描述不同行动结果的框架，用于在游戏代码中进行行动结果的描述和分类。例如，当玩家执行一个成功的 action 时，可以使用ActionResult.Success来描述这个结果；当玩家执行一个缺乏粮食的 action 时，可以使用ActionResult.InsufficientStores来描述这个结果。


```
﻿namespace Hammurabi
{
    /// <summary>
    /// Enumerates the different possible outcomes of attempting the various
    /// actions in the game.
    /// </summary>
    public enum ActionResult
    {
        /// <summary>
        /// The action was a success.
        /// </summary>
        Success,

        /// <summary>
        /// The action could not be completed because the city does not have
        /// enough bushels of grain.
        /// </summary>
        InsufficientStores,

        /// <summary>
        /// The action could not be completed because the city does not have
        /// sufficient acreage.
        /// </summary>
        InsufficientLand,

        /// <summary>
        /// The action could not be completed because the city does not have
        /// sufficient population.
        /// </summary>
        InsufficientPopulation,

        /// <summary>
        /// The requested action offended the city steward.
        /// </summary>
        Offense
    }
}

```

# `43_Hammurabi/csharp/Controller.cs`

这段代码是一个面向Hammurabi游戏世界的控制器类，其目的是提供给玩家一个数字，使游戏继续进行。在代码中，使用System命名空间中的类和函数，以及Hammurabi命名空间中的GameState类。

具体来说，这段代码的作用是：

1. 它提供了一个方法来读取玩家输入的数字，直到玩家输入了正确的数字，并且更新了游戏状态。这就意味着，在每次播放游戏时，玩家都必须输入一个数字，直到他们输入了正确的数字，才能继续进行游戏。如果玩家输入的不是数字，程序将显示错误消息并继续等待玩家的输入。

2. 它提供了一个Rule函数，用于invoke规则，在接收到玩家输入数字并获取其GameState后，使用该函数处理获取到的GameState。

3. 它包含一个while循环，该循环将一直提示玩家输入数字，直到玩家正确输入数字并更新游戏状态。

4. 在while循环中，首先执行prompt方法，这个方法会在屏幕上显示一个消息，要求玩家输入数字。然后，它尝试从玩家输入中解析Int32类型的数字，并使用try-catch语句来捕获可能发生的异常情况。如果玩家输入的不是数字，它将调用View.ShowInvalidNumber方法来显示消息并继续等待玩家的输入。

5. 如果玩家正确输入数字，将调用rule函数，传递给该函数的参数包括GameState、number和ActionResult类型的变量。然后，它根据结果采取不同的行动：如果结果是ActionResult.InsufficientLand，它将调用View.ShowInsufficientLand方法来显示消息并继续等待玩家的输入。如果结果是ActionResult.InsufficientPopulation或ActionResult.InsufficientStores，它将调用View.ShowInsufficientPopulation或View.ShowInsufficientStores方法来显示消息并继续等待玩家的输入。如果结果是ActionResult.Offense，它将调用GreatOffence方法，但需要小心，因为这个方法在70年代可能没有意义。

6. 在整个过程完成后，它将返回更新后的游戏状态。


```
﻿using System;

namespace Hammurabi
{
    /// <summary>
    /// Provides methods for reading input from the user.
    /// </summary>
    public static class Controller
    {
        /// <summary>
        /// Continuously prompts the user to enter a number until he or she
        /// enters a valid number and updates the game state.
        /// </summary>
        /// <param name="state">
        /// The current game state.
        /// </param>
        /// <param name="prompt">
        /// Action that will display the prompt to the user.
        /// </param>
        /// <param name="rule">
        /// The rule to invoke once input is retrieved.
        /// </param>
        /// <returns>
        /// The updated game state.
        /// </returns>
        public static GameState UpdateGameState(
            GameState state,
            Action prompt,
            Func<GameState, int, (GameState newState, ActionResult result)> rule)
        {
            while (true)
            {
                prompt();

                if (!Int32.TryParse(Console.ReadLine(), out var amount))
                {
                    View.ShowInvalidNumber();
                    continue;
                }

                var (newState, result) = rule(state, amount);

                switch (result)
                {
                    case ActionResult.InsufficientLand:
                        View.ShowInsufficientLand(state);
                        break;
                    case ActionResult.InsufficientPopulation:
                        View.ShowInsufficientPopulation(state);
                        break;
                    case ActionResult.InsufficientStores:
                        View.ShowInsufficientStores(state);
                        break;
                    case ActionResult.Offense:
                        // Not sure why we have to blow up the game here...
                        // Maybe this made sense in the 70's.
                        throw new GreatOffence();
                    default:
                        return newState;
                }
            }
        }
    }
}

```

# `43_Hammurabi/csharp/GameResult.cs`

这段代码是一个Namespace，其中定义了一个名为GameResult的类，这个类用于存储游戏的最终结果。

这个类的定义了一个名为PerformanceRating的属性，用于获取玩家的表现评分。

它还定义了名为AcresPerPerson和FinalStarvation的属性，用于获取每个城市最终年份的人口数量和最终年份出现过的贫困人口数量。

然后，这个类定义了一个名为TotalStarvation和AverageStarvationRate的属性，用于获取最终年份出现过的贫困人口数量和平均每年贫困人口数量百分比。

最后，它定义了一个名为Assassins的属性，用于获取想要暗杀玩家的玩家数量。

此外，它还定义了一个名为WasPlayerImpeached的属性，用于获取玩家是否因 Starved too many people（饿死了太多人）而受到弹劾的标志。


```
﻿namespace Hammurabi
{
    /// <summary>
    /// Stores the final game result.
    /// </summary>
    public record GameResult
    {
        /// <summary>
        /// Gets the player's performance rating.
        /// </summary>
        public PerformanceRating Rating { get; init; }

        /// <summary>
        /// Gets the number of acres in the city per person.
        /// </summary>
        public int AcresPerPerson { get; init; }

        /// <summary>
        /// Gets the number of people who starved the final year in office.
        /// </summary>
        public int FinalStarvation { get; init; }

        /// <summary>
        /// Gets the total number of people who starved.
        /// </summary>
        public int TotalStarvation { get; init; }

        /// <summary>
        /// Gets the average starvation rate per year (as a percentage
        /// of population).
        /// </summary>
        public int AverageStarvationRate { get; init; }

        /// <summary>
        /// Gets the number of people who want to assassinate the player.
        /// </summary>
        public int Assassins { get; init; }

        /// <summary>
        /// Gets a flag indicating whether the player was impeached for
        /// starving too many people.
        /// </summary>
        public bool WasPlayerImpeached { get; init; }
    }
}

```

# `43_Hammurabi/csharp/GameState.cs`



<script src="https://www.google.com/search?q=verapper+api&btn=0&cx=Y2MPbZoRQJQ7k8J5Mwdg3w8&弱点=0&江吃着pride+in+the+last+blight+period+of+the+1970s&hl=en&meta=urefresh&link=table&query=verapper+api&btn=0&cx=N28hoiFhaW&ie=UTF-8&aqp=65&oq=98TJX2pU10E95:10P&电器=1&杜绝=1&sa=X&武帝=0&好奇心=0&lat=39.9012&lc=3&ad=0&md=0& bigtpush=1& graphic=1& linkshare=1& trash=1&对你的恐惧是无穷的&义无反顾的杂糅的批判的现在的确实不寻常的，但是您的法律援助线上还有一位美国的45岁的女士在闷声读《数字化生存》小说，她是从前美国SBA区域信贷办公室的官员，对区里几个小镇的破产倒闭非常熟悉，&绪申请表上都有"Iures familiar with the Resolution Address Service (RAS)"。 基于此，我认为您今天的回答是正确的。




```
﻿namespace Hammurabi
{
    /// <summary>
    /// Stores the state of the game.
    /// </summary>
    public record GameState
    {
        /// <summary>
        /// Gets the current game year.
        /// </summary>
        public int Year { get; init; }

        /// <summary>
        /// Gets the city's population.
        /// </summary>
        public int Population { get; init; }

        /// <summary>
        /// Gets the population increase this year.
        /// </summary>
        public int PopulationIncrease { get; init; }

        /// <summary>
        /// Gets the number of people who starved.
        /// </summary>
        public int Starvation { get; init; }

        /// <summary>
        /// Gets the city's size in acres.
        /// </summary>
        public int Acres { get; init; }

        /// <summary>
        /// Gets the price for an acre of land (in bushels).
        /// </summary>
        public int LandPrice { get; init; }

        /// <summary>
        /// Gets the number of bushels of grain in the city stores.
        /// </summary>
        public int Stores { get; init; }

        /// <summary>
        /// Gets the amount of food distributed to the people.
        /// </summary>
        public int FoodDistributed { get; init; }

        /// <summary>
        /// Gets the number of acres that were planted.
        /// </summary>
        public int AcresPlanted { get; init; }

        /// <summary>
        /// Gets the number of bushels produced per acre.
        /// </summary>
        public int Productivity { get; init; }

        /// <summary>
        /// Gets the amount of food lost to rats.
        /// </summary>
        public int Spoilage { get; init; }

        /// <summary>
        /// Gets a flag indicating whether the current year is a plague year.
        /// </summary>
        public bool IsPlagueYear { get; init; }

        /// <summary>
        /// Gets a flag indicating whether the player has been impeached.
        /// </summary>
        public bool IsPlayerImpeached { get; init; }
    }
}

```

# `43_Hammurabi/csharp/GreatOffence.cs`

这段代码定义了一个名为“GreatOffence”的类，并继承自“InvalidOperationException”类。

在这个类的内部，使用了一个名为“GreatOffence”的类，它的摘要中说明这个类是用来表示玩家极度不端和/或不负责任的态度，使得游戏无法继续。

该类有一个内部事件“GreatOffenceEvent”，这个事件被用来在玩家犯下重大过错时触发，比如在玩家未能在游戏中尊重其他玩家或未能完成游戏目标等情况下。

总结起来，这段代码定义了一个用于表示玩家不良行为的类，该类在发生重大过错时会引发一个自定义的异常事件。


```
﻿using System;

namespace Hammurabi
{
    /// <summary>
    /// Indicates that the game cannot continue due to the player's extreme
    /// incompetance and/or unserious attitude!
    /// </summary>
    public class GreatOffence : InvalidOperationException
    {
    }
}

```

# `43_Hammurabi/csharp/PerformanceRating.cs`

这段代码定义了一个名为`PerformanceRating`的枚举类型，包含了四种不同的性能评级，分别为Disgraceful、Bad、Ok和Terrific。这个枚举类型可以被用于表示玩家在不同游戏中的表现，评级越高，表示玩家表现越差。

枚举类型通常用于在代码中描述具有特定含义的常量，可以帮助开发人员更好地理解和维护代码。在这个例子中，`PerformanceRating`枚举类型可以被用来表示游戏中的四种不同的难度水平，例如游戏的评分系统可以根据玩家的表现评定不同的评级。


```
﻿namespace Hammurabi
{
    /// <summary>
    /// Enumerates the different performance ratings that the player can
    /// achieve.
    /// </summary>
    public enum PerformanceRating
    {
        Disgraceful,
        Bad,
        Ok,
        Terrific
    }
}

```

# `43_Hammurabi/csharp/Program.cs`

这段代码是一个Hammurabi游戏的程序。它的主要作用是解释这个游戏的玩法和逻辑。

首先，它定义了一个名为Program的类，其中包含了一些常量和一些方法。

然后，它有一个名为Main的方法，它是这个程序的入口点。

在Main方法中，它首先创建了一个随机数生成器random，以及一个空的历史列表。

然后，它循环来尝试运行游戏，直到玩家被击败。在每次运行游戏时，它首先从rules.cs中加载游戏的状态，然后显示所选城市的概述。

接下来，它显示所选土地的价格，然后尝试从玩家那里获得更多土地。

然后，它显示一些分离的符号，然后尝试更新游戏状态。

接下来，它显示所选土地的详细信息，然后尝试从玩家那里获得更多食物。

最后，在每次循环结束后，它将所选的游戏状态添加到游戏历史的列表中。

然后，它从rules.cs中获取游戏的结局，并显示给玩家。

如果程序在运行过程中捕获到了GreatOffence异常，它将显示一个游戏结束的画面。


```
﻿using System;
using System.Collections.Immutable;

namespace Hammurabi
{
    public static class Program
    {
        public const int GameLength = 10;

        public static void Main(string[] args)
        {
            var random  = new Random() ;
            var state   = Rules.BeginGame();
            var history = ImmutableList<GameState>.Empty;

            View.ShowBanner();

            try
            {
                while (!state.IsPlayerImpeached)
                {
                    state = Rules.BeginTurn(state, random);
                    View.ShowCitySummary(state);

                    if (state.Year > GameLength)
                        break;

                    View.ShowLandPrice(state);
                    var newState = Controller.UpdateGameState(state, View.PromptBuyLand, Rules.BuyLand);
                    state = newState.Acres != state.Acres ?
                        newState : Controller.UpdateGameState(state, View.PromptSellLand, Rules.SellLand);

                    View.ShowSeparator();
                    state = Controller.UpdateGameState(state, View.PromptFeedPeople, Rules.FeedPeople);

                    View.ShowSeparator();
                    state = Controller.UpdateGameState(state, View.PromptPlantCrops, Rules.PlantCrops);

                    state = Rules.EndTurn(state, random);
                    history = history.Add(state);
                }

                var result = Rules.GetGameResult(history, random);
                View.ShowGameResult(result);
            }
            catch (GreatOffence)
            {
                View.ShowGreatOffence();
            }

            View.ShowFarewell();
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `43_Hammurabi/csharp/Rules.cs`

This is a class that aggregates the game state history to calculate the final rating of a player. The rating is based on various factors such as the average starvation rate, the number of kills by the community, and the player's own rating. The final rating can be either a PerformanceRating.Disgraceful, PerformanceRating.Bad, PerformanceRating.Ok, or PerformanceRating.Terrific. The assassins' number is also included in the rating if the player's rating is PerformanceRating.Ok.

The class takes two parameters, history and random, which are both required to be of type GameState. The history is a sequence of GameState objects, and the random is used to generate random numbers for the calculations. The method GetGameResult takes two parameters, history and random, and returns a new GameResult object.

The method starts by calculating the average starvation rate and the total starvation of all the states in the history. It then takes the history and the random and calculates the final rating, based on the user's rating and other factors like the number of kills by the community. Finally, if the player is injured and the final rating is PerformanceRating.Bad, the method will return PerformanceRating.Disgraceful, as it represents a poor game experience.

Note: The max rating for a player can be limited by the game or the community, it is not guaranteed to happen in every game or every community.


```
﻿using System;
using System.Collections.Generic;
using System.Linq;

namespace Hammurabi
{
    public static class Rules
    {
        /// <summary>
        /// Creates the initial state for a new game.
        /// </summary>
        public static GameState BeginGame() =>
            new GameState
            {
                Year                = 0,
                Population          = 95,
                PopulationIncrease  = 5,
                Starvation          = 0,
                Acres               = 1000,
                Stores              = 0,
                AcresPlanted        = 1000,
                Productivity        = 3,
                Spoilage            = 200,
                IsPlagueYear        = false,
                IsPlayerImpeached   = false
            };

        /// <summary>
        /// Updates the game state to start a new turn.
        /// </summary>
        public static GameState BeginTurn(GameState state, Random random) =>
            state with
            {
                Year            = state.Year + 1,
                Population      = (state.Population + state.PopulationIncrease - state.Starvation) / (state.IsPlagueYear ? 2 : 1),
                LandPrice       = random.Next(10) + 17,
                Stores          = state.Stores + (state.AcresPlanted * state.Productivity) - state.Spoilage,
                AcresPlanted    = 0,
                FoodDistributed = 0
            };

        /// <summary>
        /// Attempts to purchase the given number of acres.
        /// </summary>
        /// <returns>
        /// The updated game state and action result.
        /// </returns>
        public static (GameState newState, ActionResult result) BuyLand(GameState state, int amount)
        {
            var price = state.LandPrice * amount;

            if (price < 0)
                return (state, ActionResult.Offense);
            else
            if (price > state.Stores)
                return (state, ActionResult.InsufficientStores);
            else
                return (state with { Acres = state.Acres + amount, Stores = state.Stores - price }, ActionResult.Success);
        }

        /// <summary>
        /// Attempts to sell the given number of acres.
        /// </summary>
        /// <returns>
        /// The updated game state and action result.
        /// </returns>
        public static (GameState newState, ActionResult result) SellLand(GameState state, int amount)
        {
            var price = state.LandPrice * amount;

            if (price < 0)
                return (state, ActionResult.Offense);
            else
            if (amount >= state.Acres)
                return (state, ActionResult.InsufficientLand);
            else
                return (state with { Acres = state.Acres - amount, Stores = state.Stores + price }, ActionResult.Success);
        }

        /// <summary>
        /// Attempts to feed the people the given number of buschels.
        /// </summary>
        /// <returns>
        /// <returns>
        /// The updated game state and action result.
        /// </returns>
        public static (GameState newState, ActionResult result) FeedPeople(GameState state, int amount)
        {
            if (amount < 0)
                return (state, ActionResult.Offense);
            else
            if (amount > state.Stores)
                return (state, ActionResult.InsufficientStores);
            else
                return (state with { Stores = state.Stores - amount, FoodDistributed = state.FoodDistributed + amount }, ActionResult.Success);
        }

        /// <summary>
        /// Attempts to plant crops on the given number of acres.
        /// </summary>
        /// <returns>
        /// The updated game state and action result.
        /// </returns>
        public static (GameState newState, ActionResult result) PlantCrops(GameState state, int amount)
        {
            var storesRequired = amount / 2;
            var maxAcres       = state.Population * 10;

            if (amount < 0)
                return (state, ActionResult.Offense);
            else
            if (amount > state.Acres)
                return (state, ActionResult.InsufficientLand);
            else
            if (storesRequired > state.Stores)
                return (state, ActionResult.InsufficientStores);
            else
            if ((state.AcresPlanted + amount) > maxAcres)
                return (state, ActionResult.InsufficientPopulation);
            else
                return (state with
                {
                    AcresPlanted = state.AcresPlanted + amount,
                    Stores       = state.Stores - storesRequired,
                }, ActionResult.Success);
        }

        /// <summary>
        /// Ends the current turn and returns the updated game state.
        /// </summary>
        public static GameState EndTurn(GameState state, Random random)
        {
            var productivity = random.Next(1, 6);
            var harvest = productivity * state.AcresPlanted;

            var spoilage = random.Next(1, 6) switch
            {
                2 => state.Stores / 2,
                4 => state.Stores / 4,
                _ => 0
            };

            var populationIncrease= (int)((double)random.Next(1, 6) * (20 * state.Acres + state.Stores + harvest - spoilage) / state.Population / 100 + 1);

            var plagueYear = random.Next(20) < 3;

            var peopleFed  = state.FoodDistributed / 20;
            var starvation = peopleFed < state.Population ? state.Population - peopleFed : 0;
            var impeached  = starvation > state.Population * 0.45;

            return state with
            {
                Productivity       = productivity,
                Spoilage           = spoilage,
                PopulationIncrease = populationIncrease,
                Starvation         = starvation,
                IsPlagueYear       = plagueYear,
                IsPlayerImpeached  = impeached
            };
        }

        /// <summary>
        /// Examines the game's history to arrive at the final result.
        /// </summary>
        public static GameResult GetGameResult(IEnumerable<GameState> history, Random random)
        {
            var (_, averageStarvationRate, totalStarvation, finalState) = history.Aggregate(
                (count: 0, starvationRate: 0, totalStarvation: 0, finalState: default(GameState)),
                (stats, state) =>
                (
                    stats.count + 1,
                    ((stats.starvationRate * stats.count) + (state.Starvation * 100 / state.Population)) / (stats.count + 1),
                    stats.totalStarvation + state.Starvation,
                    state
                ));

            var acresPerPerson = finalState.Acres / finalState.Population;

            var rating = finalState.IsPlayerImpeached ?
                PerformanceRating.Disgraceful :
                (averageStarvationRate, acresPerPerson) switch
                {
                    (> 33, _) => PerformanceRating.Disgraceful,
                    (_, < 7)  => PerformanceRating.Disgraceful,
                    (> 10, _) => PerformanceRating.Bad,
                    (_, < 9)  => PerformanceRating.Bad,
                    (> 3, _)  => PerformanceRating.Ok,
                    (_, < 10) => PerformanceRating.Ok,
                    _         => PerformanceRating.Terrific
                };

            var assassins = rating == PerformanceRating.Ok ?
                random.Next(0, (int)(finalState.Population * 0.8)) : 0;

            return new GameResult
            {
                Rating                = rating,
                AcresPerPerson        = acresPerPerson,
                FinalStarvation       = finalState.Starvation,
                TotalStarvation       = totalStarvation,
                AverageStarvationRate = averageStarvationRate,
                Assassins             = assassins,
                WasPlayerImpeached    = finalState.IsPlayerImpeached
            };
        }
    }
}

```

# `43_Hammurabi/csharp/View.cs`

This is a class called `PerformanceRating` that has several methods for assigning a performance rating to a person. These methods include `Terrific`, `Perfect`, `Fantastic`, ` Awkward`, ` Smart`, ` Evil`, ` Giggle`, `Silly`, `Crazy`, `Joyful`, `Sad`, `Boring`, `Amusing`, `Interesting`, `Laughing`, `Laughing squares`, `Thieving`, `Joyful squares`, `Square squaring`, `Square squaring squares`, `Square two squares`, `Square two squares and gaps`, `Square three squares`, `Square three squares and gaps`, `Square four squares`, `Square four squares and gaps`, `Square five squares`, and `Square five squares and gaps`.

The `Terrific` method assigns a performance rating of `5`, the `Perfect` method assigns a performance rating of `10`, the `Fantastic` method assigns a performance rating of `7`, the `Awkward` method assigns a performance rating of `6`, the `Smart` method assigns a performance rating of `8`, the `Evil` method assigns a performance rating of `9`, the `Giggle` method assigns a performance rating of `10`, the `Silly` method assigns a performance rating of `1`, the `Crazy` method assigns a performance rating of `11`, the `Joyful` method assigns a performance rating of `11`, the `Sad` method assigns a performance rating of `11`, the `Boring` method assigns a performance rating of `1`, the `Amusing` method assigns a performance rating of `1`, the `Interesting` method assigns a performance rating of `1`, the `Laughing` method assigns a performance rating of `1`, the `Thieving` method assigns a performance rating of `1`, the `Joyful squares` method assigns a performance rating of `1`, the `Square squaring` method assigns a performance rating of `1`, the `Square squaring squares` method assigns a performance rating of `1`, the `Square two squares` method assigns a performance rating of `1`, the `Square two squares and gaps` method assigns a performance rating of `1`, the `Square three squares` method assigns a performance rating of `1`, the `Square three squares and gaps` method assigns a performance rating of `1`, the `Square four squares` method assigns a performance rating of `1`, the `Square four squares and gaps` method assigns a performance rating of `1`, the `Square five squares` method assigns a performance rating of `1`, and the `Square five squares and gaps` method assigns a performance rating of `1`.

Note that these methods are just examples and do not necessarily represent all possible performance ratings.


```
﻿using System;

namespace Hammurabi
{
    /// <summary>
    /// Provides various methods for presenting information to the user.
    /// </summary>
    public static class View
    {
        /// <summary>
        /// Shows the introductory banner to the player.
        /// </summary>
        public static void ShowBanner()
        {
            Console.WriteLine("                                HAMURABI");
            Console.WriteLine("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine("TRY YOUR HAND AT GOVERNING ANCIENT SUMERIA");
            Console.WriteLine("FOR A TEN-YEAR TERM OF OFFICE.");
        }

        /// <summary>
        /// Shows a summary of the current state of the city.
        /// </summary>
        public static void ShowCitySummary(GameState state)
        {
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine("HAMURABI:  I BEG TO REPORT TO YOU,");
            Console.WriteLine($"IN YEAR {state.Year}, {state.Starvation} PEOPLE STARVED, {state.PopulationIncrease} CAME TO THE CITY,");

            if (state.IsPlagueYear)
            {
                Console.WriteLine("A HORRIBLE PLAGUE STRUCK!  HALF THE PEOPLE DIED.");
            }

            Console.WriteLine($"POPULATION IS NOW {state.Population}");
            Console.WriteLine($"THE CITY NOW OWNS {state.Acres} ACRES.");
            Console.WriteLine($"YOU HARVESTED {state.Productivity} BUSHELS PER ACRE.");
            Console.WriteLine($"THE RATS ATE {state.Spoilage} BUSHELS.");
            Console.WriteLine($"YOU NOW HAVE {state.Stores} BUSHELS IN STORE.");
            Console.WriteLine();
        }

        /// <summary>
        /// Shows the current cost of land.
        /// </summary>
        /// <param name="state"></param>
        public static void ShowLandPrice(GameState state)
        {
            Console.WriteLine ($"LAND IS TRADING AT {state.LandPrice} BUSHELS PER ACRE.");
        }

        /// <summary>
        /// Displays a section separator.
        /// </summary>
        public static void ShowSeparator()
        {
            Console.WriteLine();
        }

        /// <summary>
        /// Inform the player that he or she has entered an invalid number.
        /// </summary>
        public static void ShowInvalidNumber()
        {
            Console.WriteLine("PLEASE ENTER A VALID NUMBER");
        }

        /// <summary>
        /// Inform the player that he or she has insufficient acreage.
        /// </summary>
        public static void ShowInsufficientLand(GameState state)
        {
            Console.WriteLine($"HAMURABI:  THINK AGAIN.  YOU OWN ONLY {state.Acres} ACRES.  NOW THEN,");
        }

        /// <summary>
        /// Inform the player that he or she has insufficient population.
        /// </summary>
        public static void ShowInsufficientPopulation(GameState state)
        {
            Console.WriteLine($"BUT YOU HAVE ONLY {state.Population} PEOPLE TO TEND THE FIELDS!  NOW THEN,");
        }

        /// <summary>
        /// Inform the player that he or she has insufficient grain stores.
        /// </summary>
        public static void ShowInsufficientStores(GameState state)
        {
            Console.WriteLine("HAMURABI:  THINK AGAIN.  YOU HAVE ONLY");
            Console.WriteLine($"{state.Stores} BUSHELS OF GRAIN.  NOW THEN,");
        }

        /// <summary>
        /// Show the player that he or she has caused great offence.
        /// </summary>
        public static void ShowGreatOffence()
        {
            Console.WriteLine();
            Console.WriteLine("HAMURABI:  I CANNOT DO WHAT YOU WISH.");
            Console.WriteLine("GET YOURSELF ANOTHER STEWARD!!!!!");
        }

        /// <summary>
        /// Shows the game's final result to the user.
        /// </summary>
        public static void ShowGameResult(GameResult result)
        {
            if (!result.WasPlayerImpeached)
            {
                Console.WriteLine($"IN YOUR 10-YEAR TERM OF OFFICE, {result.AverageStarvationRate} PERCENT OF THE");
                Console.WriteLine("POPULATION STARVED PER YEAR ON THE AVERAGE, I.E. A TOTAL OF");
                Console.WriteLine($"{result.TotalStarvation} PEOPLE DIED!!");

                Console.WriteLine("YOU STARTED WITH 10 ACRES PER PERSON AND ENDED WITH");
                Console.WriteLine($"{result.AcresPerPerson} ACRES PER PERSON.");
                Console.WriteLine();
            }

            switch (result.Rating)
            {
                case PerformanceRating.Disgraceful:
                    if (result.WasPlayerImpeached)
                        Console.WriteLine($"YOU STARVED {result.FinalStarvation} PEOPLE IN ONE YEAR!!!");

                    Console.WriteLine("DUE TO THIS EXTREME MISMANAGEMENT YOU HAVE NOT ONLY");
                    Console.WriteLine("BEEN IMPEACHED AND THROWN OUT OF OFFICE BUT YOU HAVE");
                    Console.WriteLine("ALSO BEEN DECLARED NATIONAL FINK!!!!");
                    break;
                case PerformanceRating.Bad:
                    Console.WriteLine("YOUR HEAVY-HANDED PERFORMANCE SMACKS OF NERO AND IVAN IV.");
                    Console.WriteLine("THE PEOPLE (REMAINING) FIND YOU AN UNPLEASANT RULER, AND,");
                    Console.WriteLine("FRANKLY, HATE YOUR GUTS!!");
                    break;
                case PerformanceRating.Ok:
                    Console.WriteLine("YOUR PERFORMANCE COULD HAVE BEEN SOMEWHAT BETTER, BUT");
                    Console.WriteLine($"REALLY WASN'T TOO BAD AT ALL. {result.Assassins} PEOPLE");
                    Console.WriteLine("WOULD DEARLY LIKE TO SEE YOU ASSASSINATED BUT WE ALL HAVE OUR");
                    Console.WriteLine("TRIVIAL PROBLEMS.");
                    break;
                case PerformanceRating.Terrific:
                    Console.WriteLine("A FANTASTIC PERFORMANCE!!!  CHARLEMANGE, DISRAELI, AND");
                    Console.WriteLine("JEFFERSON COMBINED COULD NOT HAVE DONE BETTER!");
                    break;
            }
        }

        /// <summary>
        /// Shows a farewell message to the user.
        /// </summary>
        public static void ShowFarewell()
        {
            Console.WriteLine("SO LONG FOR NOW.");
            Console.WriteLine();
        }

        /// <summary>
        /// Prompts the user to buy land.
        /// </summary>
        public static void PromptBuyLand()
        {
            Console.Write("HOW MANY ACRES DO YOU WISH TO BUY? ");
        }

        /// <summary>
        /// Prompts the user to sell land.
        /// </summary>
        public static void PromptSellLand()
        {
            Console.Write("HOW MANY ACRES DO YOU WISH TO SELL? ");
        }

        /// <summary>
        /// Prompts the user to feed the people.
        /// </summary>
        public static void PromptFeedPeople()
        {
            Console.Write("HOW MANY BUSHELS DO YOU WISH TO FEED YOUR PEOPLE? ");
        }

        /// <summary>
        /// Prompts the user to plant crops.
        /// </summary>
        public static void PromptPlantCrops()
        {
            Console.Write("HOW MANY ACRES DO YOU WISH TO PLANT WITH SEED? ");
        }
    }
}

```