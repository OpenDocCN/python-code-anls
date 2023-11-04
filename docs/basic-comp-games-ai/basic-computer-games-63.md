# BasicComputerGames源码解析 63

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `68_Orbit/java/Orbit.java`

这段代码的主要作用是创建一个基于1970年代BASIC游戏“轨道”的Java版本。该游戏的主要功能在代码中并未添加任何额外的文本或错误检查等新功能。

具体来说，这段代码实现了以下几个关键功能：

1. 导入Java Integer类、Math类和Scanner类，这些类用于处理游戏的输入和数学计算。
2. 定义了一个名为“轨道”的类，该类包含游戏的主要逻辑。
3. 通过Scanner类从用户接收输入，包括让用户通过输入猜测未知数轨道中各个点的坐标。
4. 通过Math类中的Math.random()函数生成一个0到1之间的随机数，用于计算轨道上每个点的坐标。
5. 通过Math.abs()函数计算点P的极坐标变化，即计算点P在以原点为中心、半径为1的圆上绕着原点旋转的角度。
6. 通过Scanner类再次接收用户的输入，让用户继续猜测轨道中下一个点的坐标。
7. 程序具有主循环，无限循环中让用户猜测下一个点的坐标，直到用户输入一个不是数字的字符串，程序才结束。

尽管这段代码的游戏玩法基本上与原始BASIC游戏相同，但是它缺乏很多现代Java程序中常见的错误检查和提示，因此对于现代开发者来说，在学习和使用这段代码时需要特别注意。


```
import java.lang.Integer;
import java.lang.Math;
import java.util.Scanner;

/**
 * Game of Orbit
 * <p>
 * Based on the BASIC game of Orbit here
 * https://github.com/coding-horror/basic-computer-games/blob/main/68%20Orbit/orbit.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 *
 * Converted from BASIC to Java by Darren Cardenas.
 */

```

In this scenario, the player is the Rimilian ship, trying to prevent the Romulan ship from escaping from Earth by exploding its nuclear weapon. The Rimilian ship is equipped with a bomb to destroy the Romulan ship, and must navigate through the Romulan ship to reach its target.

The player can control the Rimilian ship using the arrow keys, and the goal is to destroy the Romulan ship by hitting it with the bomb. The player must also avoid hitting the Earth or the other ships. If the player hits the Earth, the game is over, and if the player hits any other ship, the game is also over.

The game progresses through seven hours, with the player facing increasingly difficult challenges. At the end of the game, if the player has successfully destroyed the Romulan ship, the game is over and the player wins. If the player has allowed the Romulan ship to escape, the game is also over.


```
public class Orbit {

  private final Scanner scan;  // For user input

  public Orbit() {

    scan = new Scanner(System.in);

  }  // End of constructor Orbit

  public void play() {

    showIntro();
    startGame();

  }  // End of method play

  private static void showIntro() {

    System.out.println(" ".repeat(32) + "ORBIT");
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    System.out.println("\n\n");

    System.out.println("SOMEWHERE ABOVE YOUR PLANET IS A ROMULAN SHIP.");
    System.out.println("");
    System.out.println("THE SHIP IS IN A CONSTANT POLAR ORBIT.  ITS");
    System.out.println("DISTANCE FROM THE CENTER OF YOUR PLANET IS FROM");
    System.out.println("10,000 TO 30,000 MILES AND AT ITS PRESENT VELOCITY CAN");
    System.out.println("CIRCLE YOUR PLANET ONCE EVERY 12 TO 36 HOURS.");
    System.out.println("");
    System.out.println("UNFORTUNATELY, THEY ARE USING A CLOAKING DEVICE SO");
    System.out.println("YOU ARE UNABLE TO SEE THEM, BUT WITH A SPECIAL");
    System.out.println("INSTRUMENT YOU CAN TELL HOW NEAR THEIR SHIP YOUR");
    System.out.println("PHOTON BOMB EXPLODED.  YOU HAVE SEVEN HOURS UNTIL THEY");
    System.out.println("HAVE BUILT UP SUFFICIENT POWER IN ORDER TO ESCAPE");
    System.out.println("YOUR PLANET'S GRAVITY.");
    System.out.println("");
    System.out.println("YOUR PLANET HAS ENOUGH POWER TO FIRE ONE BOMB AN HOUR.");
    System.out.println("");
    System.out.println("AT THE BEGINNING OF EACH HOUR YOU WILL BE ASKED TO GIVE AN");
    System.out.println("ANGLE (BETWEEN 0 AND 360) AND A DISTANCE IN UNITS OF");
    System.out.println("100 MILES (BETWEEN 100 AND 300), AFTER WHICH YOUR BOMB'S");
    System.out.println("DISTANCE FROM THE ENEMY SHIP WILL BE GIVEN.");
    System.out.println("");
    System.out.println("AN EXPLOSION WITHIN 5,000 MILES OF THE ROMULAN SHIP");
    System.out.println("WILL DESTROY IT.");
    System.out.println("");
    System.out.println("BELOW IS A DIAGRAM TO HELP YOU VISUALIZE YOUR PLIGHT.");
    System.out.println("");
    System.out.println("");
    System.out.println("                          90");
    System.out.println("                    0000000000000");
    System.out.println("                 0000000000000000000");
    System.out.println("               000000           000000");
    System.out.println("             00000                 00000");
    System.out.println("            00000    XXXXXXXXXXX    00000");
    System.out.println("           00000    XXXXXXXXXXXXX    00000");
    System.out.println("          0000     XXXXXXXXXXXXXXX     0000");
    System.out.println("         0000     XXXXXXXXXXXXXXXXX     0000");
    System.out.println("        0000     XXXXXXXXXXXXXXXXXXX     0000");
    System.out.println("180<== 00000     XXXXXXXXXXXXXXXXXXX     00000 ==>0");
    System.out.println("        0000     XXXXXXXXXXXXXXXXXXX     0000");
    System.out.println("         0000     XXXXXXXXXXXXXXXXX     0000");
    System.out.println("          0000     XXXXXXXXXXXXXXX     0000");
    System.out.println("           00000    XXXXXXXXXXXXX    00000");
    System.out.println("            00000    XXXXXXXXXXX    00000");
    System.out.println("             00000                 00000");
    System.out.println("               000000           000000");
    System.out.println("                 0000000000000000000");
    System.out.println("                    0000000000000");
    System.out.println("                         270");
    System.out.println("");
    System.out.println("X - YOUR PLANET");
    System.out.println("O - THE ORBIT OF THE ROMULAN SHIP");
    System.out.println("");
    System.out.println("ON THE ABOVE DIAGRAM, THE ROMULAN SHIP IS CIRCLING");
    System.out.println("COUNTERCLOCKWISE AROUND YOUR PLANET.  DON'T FORGET THAT");
    System.out.println("WITHOUT SUFFICIENT POWER THE ROMULAN SHIP'S ALTITUDE");
    System.out.println("AND ORBITAL RATE WILL REMAIN CONSTANT.");
    System.out.println("");
    System.out.println("GOOD LUCK.  THE FEDERATION IS COUNTING ON YOU.");

  }  // End of method showIntro

  private void startGame() {

    double bombDistance = 0;
    int bombAltitude = 0;
    int bombAngle = 0;
    int deltaAngle = 0;
    int hour = 0;
    int shipAltitude = 0;
    int shipAngle = 0;
    int shipRate = 0;
    String userResponse = "";

    // Begin outer while loop
    while (true) {
      shipAngle = (int) (361 * Math.random());
      shipAltitude = (int) (201 * Math.random() + 200);
      shipRate = (int) (21 * Math.random() + 10);

      hour = 0;

      // Begin time limit loop
      while (hour < 7) {

        System.out.println("");
        System.out.println("");
        System.out.println("THIS IS HOUR " + (hour + 1) + ", AT WHAT ANGLE DO YOU WISH TO SEND");
        System.out.print("YOUR PHOTON BOMB? ");
        bombAngle = Integer.parseInt(scan.nextLine());

        System.out.print("HOW FAR OUT DO YOU WISH TO DETONATE IT? ");
        bombAltitude = Integer.parseInt(scan.nextLine());

        System.out.println("");
        System.out.println("");

        // Update ship position
        shipAngle += shipRate;

        // Handle full revolutions
        if (shipAngle >= 360) {
          shipAngle -= 360;
        }

        deltaAngle = Math.abs(shipAngle - bombAngle);

        // Keep angle in upper quadrants
        if (deltaAngle >= 180) {
          deltaAngle = 360 - deltaAngle;
        }

        bombDistance = Math.sqrt(shipAltitude * shipAltitude + bombAltitude * bombAltitude - 2 * shipAltitude
                       * bombAltitude * Math.cos(deltaAngle * Math.PI / 180));

        System.out.format("YOUR PHOTON BOMB EXPLODED " + "%.5f" + "*10^2 MILES FROM THE\n", bombDistance);
        System.out.println("ROMULAN SHIP.");

        // Win condition
        if (bombDistance <= 50) {
          System.out.println("YOU HAVE SUCCESSFULLY COMPLETED YOUR MISSION.");
          break;
        }

        hour++;

      }  // End time limit loop

      // Lose condition
      if (hour == 7) {
        System.out.println("YOU HAVE ALLOWED THE ROMULANS TO ESCAPE.");
      }

      System.out.println("ANOTHER ROMULAN SHIP HAS GONE INTO ORBIT.");
      System.out.print("DO YOU WISH TO TRY TO DESTROY IT? ");
      userResponse = scan.nextLine();

      if (!userResponse.toUpperCase().equals("YES")) {
        System.out.println("GOOD BYE.");
        break;
      }

    }  // End outer while loop

  }  // End of method startGame

  public static void main(String[] args) {

    Orbit game = new Orbit();
    game.play();

  }  // End of method main

}  // End of class Orbit

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `68_Orbit/javascript/orbit.js`

这段代码定义了两个函数，分别是 `print()` 和 `input()`。

`print()` 函数的作用是接收一个字符串参数，将其添加到页面上某个元素的一个文本节点中。

`input()` 函数的作用是接收一个字符串参数，并返回一个 Promise 对象。该函数通过创建一个带有 `INPUT` 标签和 `type="text"` 属性的元素，来模拟一个简单的输入框。然后，函数将焦点添加到该元素上，并监听键盘事件 `keyup`。当事件处理程序捕获到 `keydown` 事件时，如果键盘事件是 `Control+Enter`，那么将 `input()` 函数返回的字符串参数作为输入并将其添加到页面上；如果键盘事件是 `KeyDown` 事件，那么将 `input()` 函数返回的字符串参数作为输入并将其添加到文档的 `输出` 元素中。


```
// ORBIT
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

This is a program that simulates the launch of a photon bomb. The player is prompted to input the coordinates of the expected impact point (the "angle" or "pitch" of the bomb), the distance of the expected impact point from the launch point, and the distance of the explosion (the "sqrt" of the distance^2 + distance^2). The player then sends the coordinates and distance to the bomb, and the program simulates the explosion and, based on the distance and the strength of the explosion, shows the player's success or failure.
It is also worth noting that, this is not a real life simulation and it does not reflect any kind of real world data or events.


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var a = [];

// Main program
async function main()
{
    print(tab(33) + "ORBIT\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("SOMEWHERE ABOVE YOUR PLANET IS A ROMULAN SHIP.\n");
    print("\n");
    print("THE SHIP IS IN A CONSTANT POLAR ORBIT.  ITS\n");
    print("DISTANCE FROM THE CENTER OF YOUR PLANET IS FROM\n");
    print("10,000 TO 30,000 MILES AND AT ITS PRESENT VELOCITY CAN\n");
    print("CIRCLE YOUR PLANET ONCE EVERY 12 TO 36 HOURS.\n");
    print("\n");
    print("UNFORTUNATELY, THEY ARE USING A CLOAKING DEVICE SO\n");
    print("YOU ARE UNABLE TO SEE THEM, BUT WITH A SPECIAL\n");
    print("INSTRUMENT YOU CAN TELL HOW NEAR THEIR SHIP YOUR\n");
    print("PHOTON BOMB EXPLODED.  YOU HAVE SEVEN HOURS UNTIL THEY\n");
    print("HAVE BUILT UP SUFFICIENT POWER IN ORDER TO ESCAPE\n");
    print("YOUR PLANET'S GRAVITY.\n");
    print("\n");
    print("YOUR PLANET HAS ENOUGH POWER TO FIRE ONE BOMB AN HOUR.\n");
    print("\n");
    print("AT THE BEGINNING OF EACH HOUR YOU WILL BE ASKED TO GIVE AN\n");
    print("ANGLE (BETWEEN 0 AND 360) AND A DISTANCE IN UNITS OF\n");
    print("100 MILES (BETWEEN 100 AND 300), AFTER WHICH YOUR BOMB'S\n");
    print("DISTANCE FROM THE ENEMY SHIP WILL BE GIVEN.\n");
    print("\n");
    print("AN EXPLOSION WITHIN 5,000 MILES OF THE ROMULAN SHIP\n");
    print("WILL DESTROY IT.\n");
    print("\n");
    print("BELOW IS A DIAGRAM TO HELP YOU VISUALIZE YOUR PLIGHT.\n");
    print("\n");
    print("\n");
    print("                          90\n");
    print("                    0000000000000\n");
    print("                 0000000000000000000\n");
    print("               000000           000000\n");
    print("             00000                 00000\n");
    print("            00000    XXXXXXXXXXX    00000\n");
    print("           00000    XXXXXXXXXXXXX    00000\n");
    print("          0000     XXXXXXXXXXXXXXX     0000\n");
    print("         0000     XXXXXXXXXXXXXXXXX     0000\n");
    print("        0000     XXXXXXXXXXXXXXXXXXX     0000\n");
    print("180<== 00000     XXXXXXXXXXXXXXXXXXX     00000 ==>0\n");
    print("        0000     XXXXXXXXXXXXXXXXXXX     0000\n");
    print("         0000     XXXXXXXXXXXXXXXXX     0000\n");
    print("          0000     XXXXXXXXXXXXXXX     0000\n");
    print("           00000    XXXXXXXXXXXXX    00000\n");
    print("            00000    XXXXXXXXXXX    00000\n");
    print("             00000                 00000\n");
    print("               000000           000000\n");
    print("                 0000000000000000000\n");
    print("                    0000000000000\n");
    print("                         270\n");
    print("\n");
    print("X - YOUR PLANET\n");
    print("O - THE ORBIT OF THE ROMULAN SHIP\n");
    print("\n");
    print("ON THE ABOVE DIAGRAM, THE ROMULAN SHIP IS CIRCLING\n");
    print("COUNTERCLOCKWISE AROUND YOUR PLANET.  DON'T FORGET THAT\n");
    print("WITHOUT SUFFICIENT POWER THE ROMULAN SHIP'S ALTITUDE\n");
    print("AND ORBITAL RATE WILL REMAIN CONSTANT.\n");
    print("\n");
    print("GOOD LUCK.  THE FEDERATION IS COUNTING ON YOU.\n");
    while (1) {
        a = Math.floor(360 * Math.random());
        d = Math.floor(200 * Math.random() + 200);
        r = Math.floor(20 * Math.random() + 10);
        h = 0;
        while (h < 7) {
            print("\n");
            print("\n");
            print("THIS IS HOUR " + (h + 1) + ", AT WHAT ANGLE DO YOU WISH TO SEND\n");
            print("YOUR PHOTON BOMB");
            a1 = parseFloat(await input());
            print("HOW FAR OUT DO YOU WISH TO DETONATE IT");
            d1 = parseFloat(await input());
            print("\n");
            print("\n");
            a += r;
            if (a >= 360)
                a -= 360;
            t = Math.abs(a - a1);
            if (t >= 180)
                t = 360 - t;
            c = Math.sqrt(d * d + d1 * d1 - 2 * d * d1 * Math.cos(t * Math.PI / 180));
            print("YOUR PHOTON BOMB EXPLODED " + c + "*10^2 MILES FROM THE\n");
            print("ROMULAN SHIP.\n");
            if (c <= 50)
                break;
            h++;
        }
        if (h == 7) {
            print("YOU HAVE ALLOWED THE ROMULANS TO ESCAPE.\n");
        } else {
            print("YOU HAVE SUCCESSFULLY COMPLETED YOUR MISSION.\n");
        }
        print("ANOTHER ROMULAN SHIP HAS GONE INTO ORBIT.\n");
        print("DO YOU WISH TO TRY TO DESTROY IT");
        str = await input();
        if (str != "YES")
            break;
    }
    print("GOOD BYE.\n");
}

```

这是经典的 "Hello, World!" 程序，用于在 C 语言环境中启动一个新程序并输出 "Hello, World!" 消息。

在 C 语言中，`main()` 函数是程序的入口点，当程序运行时，首先执行的就是这个函数。因此，`main()` 函数也被视为程序的主函数。

这个代码片段没有其他代码，所以它不能执行任何实际的操作或处理任何输入数据。它只是一个简单的程序入口点，用于告诉计算机程序从这里开始执行。


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

This Perl script is a port of orbit, which is the 68th entry in Basic
Computer Games.

In this game you are a planetary defense gunner trying to shoot down a
cloaked Romulan ship before it can escape.

This is pretty much a straight port of the BASIC into idiomatic Perl.


# `68_Orbit/python/orbit.py`

这段代码是一个Python文件中的函数，函数名为ORBIT，定义在名为orbit_computer.py的文件中。

这个函数的作用是打印一个关于Orbital mechanics simulation的文本，其中包括一个关于这个计算机的说明。

具体来说，这个函数首先导入了两个函数：math.random和math.print。然后定义了一个名为PAGE_WIDTH的变量，值为64，这样就可以控制打印内容的最大宽度。

函数内部包括两个定义：print_centered和Orbital。print_centered函数打印一个带有中心对齐的字符串，使用了输入的msg参数。这个函数通过计算字符串长度与PAGE_WIDTH之间的关系，来确定在哪个位置插入spaces字符，使得字符串对齐到PAGE_WIDTH的奇数倍。Orbital函数没有定义具体的内容，因此无法知道它做了什么。

最后，在函数内部，使用了print函数打印出ORBIT计算机的说明，这个说明包括一个关于这个计算机的描述和一个ORBIT计算机的图标。


```
"""
ORBIT

Orbital mechanics simulation

Port by Dave LeCompte
"""

import math
import random

PAGE_WIDTH = 64


def print_centered(msg: str) -> None:
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    print(spaces + msg)


```

这段代码是一个函数，名为`print_instructions()`，它返回一个`None`对象。

函数体中，首先打印出一段关于星际迷航共和国（ROMULAN SHIP）的说明，指出这艘船处于一个常驻极地轨道上，距离中心星系的距离在10,000至30,000英里之间，当前速度可以使其绕行地球一次12至36小时。接下来，提到不幸的是，他们正在使用一种隐身装置，因此无法看到，但只要有一个特殊的仪器，可以用它来确定临近的战舰。最后，给了一个具体的的方向，使得可以在7小时内足够它们建造足够的能源逃离行星的引力。


```
def print_instructions() -> None:
    print(
        """SOMEWHERE ABOVE YOUR PLANET IS A ROMULAN SHIP.

THE SHIP IS IN A CONSTANT POLAR ORBIT.  ITS
DISTANCE FROM THE CENTER OF YOUR PLANET IS FROM
10,000 TO 30,000 MILES AND AT ITS PRESENT VELOCITY CAN
CIRCLE YOUR PLANET ONCE EVERY 12 TO 36 HOURS.

UNFORTUNATELY, THEY ARE USING A CLOAKING DEVICE SO
YOU ARE UNABLE TO SEE THEM, BUT WITH A SPECIAL
INSTRUMENT YOU CAN TELL HOW NEAR THEIR SHIP YOUR
PHOTON BOMB EXPLODED.  YOU HAVE SEVEN HOURS UNTIL THEY
HAVE BUILT UP SUFFICIENT POWER IN ORDER TO ESCAPE
YOUR PLANET'S GRAVITY.

```

这段代码是一个简单的游戏文本，描述了一个关于如何在敌人船上放置炸弹并摧毁敌舰的故事。游戏文本显示，每个小时开始时，玩家需要给出一个角度（0到360度）和距离（100到300英里），然后敌人的距离从敌人船只的位置得到更新。在游戏过程中，如果玩家放置的炸弹在敌人船只5000海里内，就会摧毁敌人船只并结束游戏。


```
YOUR PLANET HAS ENOUGH POWER TO FIRE ONE BOMB AN HOUR.

AT THE BEGINNING OF EACH HOUR YOU WILL BE ASKED TO GIVE AN
ANGLE (BETWEEN 0 AND 360) AND A DISTANCE IN UNITS OF
100 MILES (BETWEEN 100 AND 300), AFTER WHICH YOUR BOMB'S
DISTANCE FROM THE ENEMY SHIP WILL BE GIVEN.

AN EXPLOSION WITHIN 5,000 MILES OF THE ROMULAN SHIP
WILL DESTROY IT.

BELOW IS A DIAGRAM TO HELP YOU VISUALIZE YOUR PLIGHT.


                          90
                    0000000000000
                 0000000000000000000
               000000           000000
             00000                 00000
            00000    XXXXXXXXXXX    00000
           00000    XXXXXXXXXXXXX    00000
          0000     XXXXXXXXXXXXXXX     0000
         0000     XXXXXXXXXXXXXXXXX     0000
        0000     XXXXXXXXXXXXXXXXXXX     0000
```

这段代码是一个32位无符号整数，由180个0组成。这个数字在二进制中代表了一个由0和255组成的八位数，每一位都是1，整个数是0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000


```
180<== 00000     XXXXXXXXXXXXXXXXXXX     00000 ==>0
        0000     XXXXXXXXXXXXXXXXXXX     0000
         0000     XXXXXXXXXXXXXXXXX     0000
          0000     XXXXXXXXXXXXXXX     0000
           00000    XXXXXXXXXXXXX    00000
            00000    XXXXXXXXXXX    00000
             00000                 00000
               000000           000000
                 0000000000000000000
                    0000000000000
                         270

X - YOUR PLANET
O - THE ORBIT OF THE ROMULAN SHIP

```

这段代码是一个海盗游戏画面，展示了联邦制计数船围绕地球counterclockwise旋转的情况。在没有足够的电力时，船的俯仰角和轨道速率将保持不变。最后提醒玩家输入“YES”或“NO”来选择是否继续游戏。


```
ON THE ABOVE DIAGRAM, THE ROMULAN SHIP IS CIRCLING
COUNTERCLOCKWISE AROUND YOUR PLANET.  DON'T FORGET THAT
WITHOUT SUFFICIENT POWER THE ROMULAN SHIP'S ALTITUDE
AND ORBITAL RATE WILL REMAIN CONSTANT.

GOOD LUCK.  THE FEDERATION IS COUNTING ON YOU.
"""
    )


def get_yes_or_no() -> bool:
    while True:
        response = input().upper()
        if response == "YES":
            return True
        elif response == "NO":
            return False
        else:
            print("PLEASE TYPE 'YES' OR 'NO'")


```

这是一个Python的函数，名为`game_over`和`play_game`。

`game_over`函数用于判断玩家是否成功完成任务。如果玩家成功，函数会输出一条消息，表示任务成功完成。否则，函数会输出一条消息，表示允许 Romulans 逃脱，并输出一条消息，询问玩家是否希望继续攻击 Romulans 的船。最后，函数返回一个布尔值，表示玩家是否成功或未成功完成任务。

`play_game`函数用于实际玩游戏。函数会生成两个随机数，rom_angle 和 bomb_angle，分别表示romulans 和 RomulanBot的当前角度。函数会询问玩家是否继续发送 Photon Bomb，或者发送 RomulanBot 的角度。然后，函数计算两个角度之间的差异，并使用数学公式计算出一个距离，然后比较这个距离与50（如果超过50则认为是未在目标范围内）之间的值。如果距离小于或等于50，则函数判断已成功在目标范围内摧毁 Romulan，并返回 True；否则，函数判断已超过50，并返回 False。最后，函数根据结果返回一个布尔值，表示玩家是否成功或未成功完成任务。


```
def game_over(is_success: bool) -> bool:
    if is_success:
        print("YOU HAVE SUCCESSFULLY COMPLETED YOUR MISSION.")
    else:
        print("YOU HAVE ALLOWED THE ROMULANS TO ESCAPE.")
    print("ANOTHER ROMULAN SHIP HAS GONE INTO ORBIT.")
    print("DO YOU WISH TO TRY TO DESTROY IT?")

    return get_yes_or_no()


def play_game() -> bool:
    rom_angle = random.randint(0, 359)
    rom_distance = random.randint(100, 300)
    rom_angular_velocity = random.randint(10, 30)
    hour = 0
    while hour < 7:
        hour += 1
        print()
        print()
        print(f"THIS IS HOUR {hour}, AT WHAT ANGLE DO YOU WISH TO SEND")
        print("YOUR PHOTON BOMB?")

        bomb_angle = float(input())
        print("HOW FAR OUT DO YOU WISH TO DETONATE IT?")
        bomb_distance = float(input())
        print()
        print()

        rom_angle = (rom_angle + rom_angular_velocity) % 360
        angular_difference = rom_angle - bomb_angle
        c = math.sqrt(
            rom_distance**2
            + bomb_distance**2
            - 2
            * rom_distance
            * bomb_distance
            * math.cos(math.radians(angular_difference))
        )

        print(f"YOUR PHOTON BOMB EXPLODED {c:.4f}*10^2 MILES FROM THE")
        print("ROMULAN SHIP.")

        if c <= 50:
            # Destroyed the Romulan
            return True

    # Ran out of time
    return False


```

这段代码是一个Python程序，名为“main”。程序的主要目的是提供游戏玩家一个有趣的游戏体验。

程序中包含以下主要函数：

1. `print_centered` 函数：用于在屏幕上打印字符。`centered` 函数的作用是将字符以中心方式居中打印。

2. `print_instructions` 函数：用于在屏幕上打印游戏说明。该函数的作用是在屏幕上打印游戏规则和说明。

3. `play_game` 函数：用于让玩家玩游戏。该函数会不断地尝试连接服务器，并返回一个布尔值，表示游戏是否成功。

4. `game_over` 函数：用于判断游戏是否结束。该函数会根据玩家是否在游戏中或服务器是否恢复正常来判断游戏是否结束。

5. `if` 语句：用于判断条件是否成立，如果成立则执行括号内的语句。

6. `return` 语句：用于返回一个值给调用者。


```
def main() -> None:
    print_centered("ORBIT")
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")

    print_instructions()

    while True:
        success = play_game()
        again = game_over(success)
        if not again:
            return


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


### Pizza

In this game, you take orders for pizzas from people living in Hyattsville. Armed with a map of the city, you must then tell your delivery boy the address where the pizza is to be delivered. If the pizza is delivered to the correct address, the customer phones you and thanks you; if not, you must give the driver the correct address until the pizza gets delivered.

Some interesting modifications suggest themselves for this program such as pizzas getting cold after two incorrect delivery attempts or taking three or more orders at a time and figuring out the shortest delivery route. Send us your modifications!

This program seems to have surfaced originally at the University of Georgia in Athens, Georgia. The author is unknown.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=126)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=141)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Known Bugs

- The program does no validation of its input, and crashes if you enter coordinates outside the valid range.  (Ports may choose to improve on this, for example by repeating the prompt until valid coordinates are given.)

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `69_Pizza/csharp/CustomerMap.cs`

This code appears to be a class that defines a `CustomerMap` and several methods related to it.

The `CustomerMap` class appears to be a 2D array that maps the X and Y coordinates of each customer in a map to a string representing the customer name.

The `GetCustomerMap` method appears to return a 2D array of customer maps, each of the same size as the map that it is derived from.

The `GetCustomerNames` method appears to return a list of customer names in the format `"A"`, `"B"`, `"C"`, etc.

The `AppendXLine` method appears to take a `mapToDisplay` string and a `horizontalSpace` integer as input, and appends a line with the X coordinates of the customer nodes in the map to the `mapToDisplay`.

The `AppendCustomerInfo` method appears to take a `CustomerMap` and a string array as input, and appends the customer names of the nodes in the map to the given customer names array.

It looks like the class also defines a `CustomerSegment` class, which appears to be a base class for all customer segments, with a `GetCustomerMap` method that returns a reference to a 2D array of the same size as the map that it is derived from, and a `GetCustomerNames` method that returns a list of customer names in the format `"A"`, `"B"`, `"C"`, etc.

It is worth noting that the code also defines a `CustomerMap2D` class, which appears to be a class that derives from `CustomerMap`, and defines a `GetX` method that returns a 2D array of the X coordinates of the customer nodes in the map, and a `GetY` method that returns a 2D array of the Y coordinates of the customer nodes in the map. It is unclear from the code how this class is intended to be used.



```
﻿using System.Text;

namespace Pizza
{
    internal class CustomerMap
    {
        private readonly int _mapSize;
        private readonly string[,] _customerMap;

        public CustomerMap(int mapSize)
        {
            _mapSize = mapSize;
            _customerMap = GenerateCustomerMap();
        }

        /// <summary>
        /// Gets customer on position X, Y.
        /// </summary>
        /// <param name="x">Represents X position.</param>
        /// <param name="y">Represents Y position.</param>
        /// <returns>If positions is valid then returns customer name otherwise returns empty string.</returns>
        public string GetCustomerOnPosition(int x, int y)
        {
            if(IsPositionOutOfRange(x, y))
            {
                return string.Empty;
            }

            return _customerMap[y, x];
        }

        /// <summary>
        /// Overridden ToString for getting text representation of customers map.
        /// </summary>
        /// <returns>Text representation of customers map.</returns>
        public override string ToString()
        {
            int verticalSpace = 4;
            int horizontalSpace = 5;

            var mapToDisplay = new StringBuilder();

            AppendXLine(mapToDisplay, horizontalSpace);

            for (int i = _customerMap.GetLength(0) - 1; i >= 0; i--)
            {
                mapToDisplay.AppendLine("-", verticalSpace);
                mapToDisplay.Append($"{i + 1}");
                mapToDisplay.Append(' ', horizontalSpace);

                for (var j = 0; j < _customerMap.GetLength(1); j++)
                {
                    mapToDisplay.Append($"{_customerMap[i, j]}");
                    mapToDisplay.Append(' ', horizontalSpace);
                }

                mapToDisplay.Append($"{i + 1}");
                mapToDisplay.Append(' ', horizontalSpace);
                mapToDisplay.Append(Environment.NewLine);
            }

            mapToDisplay.AppendLine("-", verticalSpace);

            AppendXLine(mapToDisplay, horizontalSpace);

            return mapToDisplay.ToString();
        }

        /// <summary>
        /// Checks if position is out of range or not.
        /// </summary>
        /// <param name="x">Represents X position.</param>
        /// <param name="y">Represents Y position.</param>
        /// <returns>True if position is out of range otherwise false.</returns>
        private bool IsPositionOutOfRange(int x, int y)
        {
            return
                x < 0 || x > _mapSize - 1 ||
                y < 0 || y > _mapSize - 1;
        }

        /// <summary>
        /// Generates array which represents customers map.
        /// </summary>
        /// <returns>Returns customers map.</returns>
        private string[,] GenerateCustomerMap()
        {
            string[,] customerMap = new string[_mapSize, _mapSize];
            string[] customerNames = GetCustomerNames(_mapSize * _mapSize);
            int currentCustomerNameIndex = 0;

            for (int i = 0; i < customerMap.GetLength(0); i++)
            {
                for (int j = 0; j < customerMap.GetLength(1); j++)
                {
                    customerMap[i, j] = customerNames[currentCustomerNameIndex++].ToString();
                }
            }

            return customerMap;
        }

        /// <summary>
        /// Generates customer names. Names are represented by alphanumerics from 'A'. Name of last customer depends on passed parameter.
        /// </summary>
        /// <param name="numberOfCustomers">How many customers need to be generated.</param>
        /// <returns>List of customer names.</returns>
        private static string[] GetCustomerNames(int numberOfCustomers)
        {
            // returns ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P"];
            return Enumerable.Range(65, numberOfCustomers).Select(c => ((Char)c).ToString()).ToArray();
        }

        /// <summary>
        /// Appends line with X coordinates.
        /// </summary>
        /// <param name="mapToDisplay">Current map where a new line will be appended.</param>
        /// <param name="horizontalSpace">Number of horizontal delimiters which will be added between each coordination.</param>
        private void AppendXLine(StringBuilder mapToDisplay, int horizontalSpace)
        {
            mapToDisplay.Append(' ');
            mapToDisplay.Append('-', horizontalSpace);
            for (var i = 0; i < _customerMap.GetLength(0); i++)
            {
                mapToDisplay.Append($"{i + 1}");
                mapToDisplay.Append('-', horizontalSpace);
            }
            mapToDisplay.Append(Environment.NewLine);
        }
    }
}

```

# `69_Pizza/csharp/PizzaGame.cs`

This is a class that contains methods for interacting with the player input.
The class has two methods, `GetPlayerInput` and `AskQuestionWithYesNoResponse`, both of which take a question as a parameter and return a boolean value indicating the player's input.
The `GetPlayerInput` method displays a question in the console and then prompts the player to enter a response by either typing "YES", "Y", or "NO". The response will be stored in the `userInput` variable, and the method returns `true` if the player entered "YES" or "NO", and `false` otherwise.
The `AskQuestionWithYesNoResponse` method displays the same question in the console as the one passed in as a parameter, and then prompts the player to enter a response by typing either "YES", "Y", or "NO". The method returns the first positive response, or `false` if no response is entered.
Both methods contain code that reads input from the player, and丁类 also have a `WriteEmptyLine` method that writes an empty line to the console.


```
﻿namespace Pizza
{
    internal class PizzaGame
    {
        private const int CustomerMapSize = 4;
        private readonly CustomerMap _customerMap = new CustomerMap(CustomerMapSize);

        /// <summary>
        /// Starts game. Main coordinator for pizza game.
        /// It is responsible for showing information, getting data from user and starting to delivery pizza.
        /// </summary>
        public void Play()
        {
            ShowHeader();

            string playerName = GetPlayerName();

            ShowIntroduction(playerName);
            ShowMap();

            if (AskForMoreDirections())
            {
                ShowMoreDirections(playerName);

                var playerUnderstands = AskIfPlayerUnderstand();
                if (!playerUnderstands)
                {
                    return;
                }
            }

            StartDelivery(playerName);
            EndDelivery(playerName);
        }

        /// <summary>
        /// Starts with pizza delivering to customers.
        /// Every 5 deliveries it is asking user whether want to continue in delivering.
        /// </summary>
        /// <param name="playerName">Player name which was filled by user.</param>
        private void StartDelivery(string playerName)
        {
            var numberOfDeliveredPizzas = 0;
            while (true)
            {
                numberOfDeliveredPizzas++;
                string deliverPizzaToCustomer = GetRandomCustomer();

                WriteEmptyLine();
                Console.WriteLine($"HELLO {playerName}'S PIZZA.  THIS IS {deliverPizzaToCustomer}.");
                Console.WriteLine("\tPLEASE SEND A PIZZA.");

                DeliverPizzaByPlayer(playerName, deliverPizzaToCustomer);

                if (numberOfDeliveredPizzas % 5 == 0)
                {
                    bool playerWantToDeliveryMorePizzas = AskQuestionWithYesNoResponse("DO YOU WANT TO DELIVER MORE PIZZAS?");
                    if (!playerWantToDeliveryMorePizzas)
                    {
                        WriteEmptyLine();
                        break;
                    }
                }
            }
        }

        /// <summary>
        /// Gets random customer for which pizza should be delivered.
        /// </summary>
        /// <returns>Customer name.</returns>
        private string GetRandomCustomer()
        {
            int randomPositionOnX = Random.Shared.Next(0, CustomerMapSize);
            int randomPositionOnY = Random.Shared.Next(0, CustomerMapSize);

            return _customerMap.GetCustomerOnPosition(randomPositionOnX, randomPositionOnY);
        }

        /// <summary>
        /// Delivers pizza to customer by player. It verifies whether player was delivering pizza to correct customer.
        /// </summary>
        /// <param name="playerName">Player name which was filled by user.</param>
        /// <param name="deliverPizzaToCustomer">Customer name which order pizza.</param>
        private void DeliverPizzaByPlayer(string playerName, string deliverPizzaToCustomer)
        {
            while (true)
            {
                string userInput = GetPlayerInput($"\tDRIVER TO {playerName}:  WHERE DOES {deliverPizzaToCustomer} LIVE?");
                var deliveredToCustomer = GetCustomerFromPlayerInput(userInput);
                if (string.IsNullOrEmpty(deliveredToCustomer))
                {
                    deliveredToCustomer = "UNKNOWN CUSTOMER";
                }

                if (deliveredToCustomer.Equals(deliverPizzaToCustomer))
                {
                    Console.WriteLine($"HELLO {playerName}.  THIS IS {deliverPizzaToCustomer}, THANKS FOR THE PIZZA.");
                    break;
                }

                Console.WriteLine($"THIS IS {deliveredToCustomer}.  I DID NOT ORDER A PIZZA.");
                Console.WriteLine($"I LIVE AT {userInput}");
            }
        }

        /// <summary>
        /// Gets customer name by user input with customer coordinations.
        /// </summary>
        /// <param name="userInput">Input from users - it should represent customer coordination separated by ','.</param>
        /// <returns>If coordinations are correct and customer exists then returns true otherwise false.</returns>
        private string GetCustomerFromPlayerInput(string userInput)
        {
            var pizzaIsDeliveredToPosition = userInput?
                .Split(',')
                .Select(i => int.TryParse(i, out var customerPosition) ? (customerPosition - 1) : -1)
                .Where(i => i != -1)
                .ToArray() ?? Array.Empty<int>();
            if (pizzaIsDeliveredToPosition.Length != 2)
            {
                return string.Empty;
            }

            return _customerMap.GetCustomerOnPosition(pizzaIsDeliveredToPosition[0], pizzaIsDeliveredToPosition[1]);
        }

        /// <summary>
        /// Shows game header in console.
        /// </summary>
        private void ShowHeader()
        {
            Console.WriteLine("PIZZA".PadLeft(22));
            Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            WriteEmptyLine(3);
            Console.WriteLine("PIZZA DELIVERY GAME");
            WriteEmptyLine();
        }

        /// <summary>
        /// Asks user for name which will be used in game.
        /// </summary>
        /// <returns>Player name.</returns>
        private string GetPlayerName()
        {
            return GetPlayerInput("WHAT IS YOUR FIRST NAME:");
        }

        /// <summary>
        /// Shows game introduction in console
        /// </summary>
        /// <param name="playerName">Player name which was filled by user.</param>
        private void ShowIntroduction(string playerName)
        {
            Console.WriteLine($"HI, {playerName}.  IN THIS GAME YOU ARE TO TAKE ORDERS");
            Console.WriteLine("FOR PIZZAS.  THEN YOU ARE TO TELL A DELIVERY BOY");
            Console.WriteLine("WHERE TO DELIVER THE ORDERED PIZZAS.");
            WriteEmptyLine(2);
        }

        /// <summary>
        /// Shows customers map in console. In this method is used overridden method 'ToString' for getting text representation of customers map.
        /// </summary>
        private void ShowMap()
        {
            Console.WriteLine("MAP OF THE CITY OF HYATTSVILLE");
            WriteEmptyLine();

            Console.WriteLine(_customerMap.ToString());

            Console.WriteLine("THE OUTPUT IS A MAP OF THE HOMES WHERE");
            Console.WriteLine("YOU ARE TO SEND PIZZAS.");
            WriteEmptyLine();
            Console.WriteLine("YOUR JOB IS TO GIVE A TRUCK DRIVER");
            Console.WriteLine("THE LOCATION OR COORDINATES OF THE");
            Console.WriteLine("HOME ORDERING THE PIZZA.");
            WriteEmptyLine();
        }

        /// <summary>
        /// Asks user if needs more directions.
        /// </summary>
        /// <returns>True if user need more directions otherwise false.</returns>
        private bool AskForMoreDirections()
        {
            var playerNeedsMoreDirections = AskQuestionWithYesNoResponse("DO YOU NEED MORE DIRECTIONS?");
            WriteEmptyLine();

            return playerNeedsMoreDirections;
        }

        /// <summary>
        /// Shows more directions.
        /// </summary>
        /// <param name="playerName">Player name which was filled by user.</param>
        private void ShowMoreDirections(string playerName)
        {
            Console.WriteLine("SOMEBODY WILL ASK FOR A PIZZA TO BE");
            Console.WriteLine("DELIVERED.  THEN A DELIVERY BOY WILL");
            Console.WriteLine("ASK YOU FOR THE LOCATION.");
            Console.WriteLine("\tEXAMPLE:");
            Console.WriteLine("THIS IS J.  PLEASE SEND A PIZZA.");
            Console.WriteLine($"DRIVER TO {playerName}.  WHERE DOES J LIVE?");
            Console.WriteLine("YOUR ANSWER WOULD BE 2,3");
        }

        /// <summary>
        /// Asks user if understands to instructions.
        /// </summary>
        /// <returns>True if user understand otherwise false.</returns>
        private bool AskIfPlayerUnderstand()
        {
            var playerUnderstands = AskQuestionWithYesNoResponse("UNDERSTAND?");
            if (!playerUnderstands)
            {
                Console.WriteLine("THIS JOB IS DEFINITELY TOO DIFFICULT FOR YOU. THANKS ANYWAY");
                return false;
            }

            WriteEmptyLine();
            Console.WriteLine("GOOD.  YOU ARE NOW READY TO START TAKING ORDERS.");
            WriteEmptyLine();
            Console.WriteLine("GOOD LUCK!!");
            WriteEmptyLine();

            return true;
        }

        /// <summary>
        /// Shows message about ending delivery in console.
        /// </summary>
        /// <param name="playerName">Player name which was filled by user.</param>
        private void EndDelivery(string playerName)
        {
            Console.WriteLine($"O.K. {playerName}, SEE YOU LATER!");
            WriteEmptyLine();
        }

        /// <summary>
        /// Gets input from user.
        /// </summary>
        /// <param name="question">Question which is displayed in console.</param>
        /// <returns>User input.</returns>
        private string GetPlayerInput(string question)
        {
            Console.Write($"{question} ");

            while (true)
            {
                var userInput = Console.ReadLine();
                if (!string.IsNullOrWhiteSpace(userInput))
                {
                    return userInput;
                }
            }
        }

        /// <summary>
        /// Asks user with required resposne 'YES', 'Y, 'NO', 'N'.
        /// </summary>
        /// <param name="question">Question which is displayed in console.</param>
        /// <returns>True if user write 'YES', 'Y'. False if user write 'NO', 'N'.</returns>
        private static bool AskQuestionWithYesNoResponse(string question)
        {
            var possitiveResponse = new string[] { "Y", "YES" };
            var negativeResponse = new string[] { "N", "NO" };
            var validUserInputs = possitiveResponse.Concat(negativeResponse);

            Console.Write($"{question} ");

            string? userInput;
            while (true)
            {
                userInput = Console.ReadLine();
                if (!string.IsNullOrWhiteSpace(userInput) && validUserInputs.Contains(userInput.ToUpper()))
                {
                    break;
                }

                Console.Write($"'YES' OR 'NO' PLEASE, NOW THEN, {question} ");
            }

            return possitiveResponse.Contains(userInput.ToUpper());
        }

        /// <summary>
        /// Writes empty line in console.
        /// </summary>
        /// <param name="numberOfEmptyLines">Number of empty lines which will be written in console. Parameter is optional and default value is 1.</param>
        private void WriteEmptyLine(int numberOfEmptyLines = 1)
        {
            for (int i = 0; i < numberOfEmptyLines; i++)
            {
                Console.WriteLine();
            }
        }
    }
}

```

# `69_Pizza/csharp/Program.cs`

这是一个面向对象编程语言（例如Java、C#等）中的一个示例代码。这段代码定义了一个名为PizzaGame的类，并在该类中实现了一个名为Main的方法。

Main方法的实现创建了一个名为PizzaGame的实例，然后调用其的Play方法来执行游戏的主要逻辑。但在此示例中，省略了PizzaGame类的方法调用的详细说明。

总之，这段代码的目的是创建一个PizzaGame的实例，并调用其的Play方法以在主函数中执行游戏的主要逻辑。


```
﻿namespace Pizza
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var pizzaGame = new PizzaGame();
            pizzaGame.Play();
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `69_Pizza/csharp/StringBuilderExtensions.cs`

这段代码是一个名为`StringBuilderExtensions`的类，其目的是扩展`StringBuilder`类以提供添加新行和特定值的功能。

具体来说，这个类包含一个名为`AppendLine`的方法，它接收一个`StringBuilder`对象、一个要添加的值`value`以及一个添加行数的`int`参数`numberOfLines`。这个方法会在`StringBuilder`对象中添加`value`重复`numberOfLines`次，每次添加一个新的行到字符串中。

例如，如果你创建一个字符串变量`myString`并将其赋值为`"Hello"`、将`AppendLine`方法应用于`myString.AppendLine("第二", 5)`将把`"Hello第二"`添加到字符串中。


```
﻿using System.Text;

namespace Pizza
{
    internal static class StringBuilderExtensions
    {
        /// <summary>
        /// Extensions for adding new lines of specific value.
        /// </summary>
        /// <param name="stringBuilder">Extended class.</param>
        /// <param name="value">Value which will be repeated.</param>
        /// <param name="numberOfLines">Number of lines that will be appended.</param>
        public static void AppendLine(this StringBuilder stringBuilder, string value, int numberOfLines)
        {
            for (int i = 0; i < numberOfLines; i++)
            {
                stringBuilder.AppendLine(value);
            }
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `69_Pizza/java/src/Pizza.java`

This is a Java class that appears to implement a simple text-based game where the player is prompted to enter a series of choices. The game has different levels of difficulty and the player must use the Enter key to enter their choices.

The class contains several methods:

* `getDelimitedValue(String text, int pos)`: This method takes a string and a position in the string and returns the value at that position.
* `yesEntered(String text)`: This method takes a string and returns true if the string is equal to one of the values specified in the call to `stringIsAnyValue`.
* `yesOrNoEntered(String text)`: This method takes a string and returns true if the string contains at least one of the values specified in the call to `stringIsAnyValue`, case-insensitive.
* `stringIsAnyValue(String text, String... values)`: This method takes a string and some values and returns true if any of the values match the specified ones.
* `displayTextAndGetInput(String text)`: This method displays a message on the screen and then accepts input from the keyboard. It returns what was typed by the player.

The class also contains several instance variables:

* `text`: A string representing the current level of the game.
* `values`: A list of strings representing the different levels of the game.
* `kbScanner`: An instance of the `键盘` class, which is responsible for reading input from the keyboard.

The `键盘` class has several methods:

* `next()`: This method reads a single character from the keyboard and returns it.

The class also has a `游戏难度`常量， which is initialized in the constructor:

* `游戏难度 = 5`: This means the game will have 5 levels of difficulty, and the player will have to press the Enter key 5 times to complete each level.


```
import java.util.Scanner;

/**
 * Game of Pizza
 * <p>
 * Based on the Basic game of Hurkle here
 * https://github.com/coding-horror/basic-computer-games/blob/main/69%20Pizza/pizza.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Pizza {

    private final int MAX_DELIVERIES = 5;

    private enum GAME_STATE {
        STARTING,
        ENTER_NAME,
        DRAW_MAP,
        MORE_DIRECTIONS,
        START_DELIVER,
        DELIVER_PIZZA,
        TOO_DIFFICULT,
        END_GAME,
        GAME_OVER
    }

    // houses that can order pizza
    private final char[] houses = new char[]{'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
            'J', 'K', 'L', 'M', 'N', 'O', 'P'};

    // size of grid
    private final int[] gridPos = new int[]{1, 2, 3, 4};

    private GAME_STATE gameState;

    private String playerName;

    // How many pizzas have been successfully delivered
    private int pizzaDeliveryCount;

    // current house that ordered a pizza
    private int currentHouseDelivery;

    // Used for keyboard input
    private final Scanner kbScanner;

    public Pizza() {

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
                    init();
                    intro();
                    gameState = GAME_STATE.ENTER_NAME;
                    break;

                // Enter the players name
                case ENTER_NAME:
                    playerName = displayTextAndGetInput("WHAT IS YOUR FIRST NAME? ");
                    System.out.println("HI " + playerName + ". IN GAME YOU ARE TO TAKE ORDERS");
                    System.out.println("FOR PIZZAS.  THEN YOU ARE TO TELL A DELIVERY BOY");
                    System.out.println("WHERE TO DELIVER THE ORDERED PIZZAS.");
                    System.out.println();
                    gameState = GAME_STATE.DRAW_MAP;
                    break;

                // Draw the map
                case DRAW_MAP:
                    drawMap();
                    gameState = GAME_STATE.MORE_DIRECTIONS;
                    break;

                // need more directions (how to play) ?
                case MORE_DIRECTIONS:
                    extendedIntro();
                    String moreInfo = displayTextAndGetInput("DO YOU NEED MORE DIRECTIONS? ");
                    if (!yesOrNoEntered(moreInfo)) {
                        System.out.println("'YES' OR 'NO' PLEASE, NOW THEN,");
                    } else {
                        // More instructions selected
                        if (yesEntered(moreInfo)) {
                            displayMoreDirections();
                            // Player understand now?
                            if (yesEntered(displayTextAndGetInput("UNDERSTAND? "))) {
                                System.out.println("GOOD.  YOU ARE NOW READY TO START TAKING ORDERS.");
                                System.out.println();
                                System.out.println("GOOD LUCK!!");
                                System.out.println();
                                gameState = GAME_STATE.START_DELIVER;
                            } else {
                                // Not understood, essentially game over
                                gameState = GAME_STATE.TOO_DIFFICULT;
                            }
                        } else {
                            // no more directions were needed, start delivering pizza
                            gameState = GAME_STATE.START_DELIVER;
                        }
                    }

                    break;

                // Too difficult to understand, game over!
                case TOO_DIFFICULT:
                    System.out.println("JOB IS DEFINITELY TOO DIFFICULT FOR YOU. THANKS ANYWAY");
                    gameState = GAME_STATE.GAME_OVER;
                    break;

                // Delivering pizza
                case START_DELIVER:
                    // select a random house and "order" a pizza for them.
                    currentHouseDelivery = (int) (Math.random()
                            * (houses.length) + 1) - 1; // Deduct 1 for 0-based array

                    System.out.println("HELLO " + playerName + "'S PIZZA.  THIS IS "
                            + houses[currentHouseDelivery] + ".");
                    System.out.println("  PLEASE SEND A PIZZA.");
                    gameState = GAME_STATE.DELIVER_PIZZA;
                    break;

                // Try and deliver the pizza
                case DELIVER_PIZZA:

                    String question = "  DRIVER TO " + playerName + ":  WHERE DOES "
                            + houses[currentHouseDelivery] + " LIVE ? ";
                    String answer = displayTextAndGetInput(question);

                    // Convert x,y entered by player to grid position of a house
                    int x = getDelimitedValue(answer, 0);
                    int y = getDelimitedValue(answer, 1);
                    int calculatedPos = (x + (y - 1) * 4) - 1;

                    // Did the player select the right house to deliver?
                    if (calculatedPos == currentHouseDelivery) {
                        System.out.println("HELLO " + playerName + ".  THIS IS " + houses[currentHouseDelivery]
                                + ", THANKS FOR THE PIZZA.");
                        pizzaDeliveryCount++;

                        // Delivered enough pizza?

                        if (pizzaDeliveryCount > MAX_DELIVERIES) {
                            gameState = GAME_STATE.END_GAME;
                        } else {
                            gameState = GAME_STATE.START_DELIVER;
                        }
                    } else {
                        System.out.println("THIS IS " + houses[calculatedPos] + ".  I DID NOT ORDER A PIZZA.");
                        System.out.println("I LIVE AT " + x + "," + y);
                        // Don't change gameState so state is executed again
                    }

                    break;

                // Sign off message for cases where the Chief is not upset
                case END_GAME:
                    if (yesEntered(displayTextAndGetInput("DO YOU WANT TO DELIVER MORE PIZZAS? "))) {
                        init();
                        gameState = GAME_STATE.START_DELIVER;
                    } else {
                        System.out.println();
                        System.out.println("O.K. " + playerName + ", SEE YOU LATER!");
                        System.out.println();
                        gameState = GAME_STATE.GAME_OVER;
                    }
                    break;

                // GAME_OVER State does not specifically have a case
            }
        } while (gameState != GAME_STATE.GAME_OVER);
    }

    private void drawMap() {

        System.out.println("MAP OF THE CITY OF HYATTSVILLE");
        System.out.println();
        System.out.println(" -----1-----2-----3-----4-----");
        int k = 3;
        for (int i = 1; i < 5; i++) {
            System.out.println("-");
            System.out.println("-");
            System.out.println("-");
            System.out.println("-");

            System.out.print(gridPos[k]);
            int pos = 16 - 4 * i;
            System.out.print("     " + houses[pos]);
            System.out.print("     " + houses[pos + 1]);
            System.out.print("     " + houses[pos + 2]);
            System.out.print("     " + houses[pos + 3]);
            System.out.println("     " + gridPos[k]);
            k = k - 1;
        }
        System.out.println("-");
        System.out.println("-");
        System.out.println("-");
        System.out.println("-");
        System.out.println(" -----1-----2-----3-----4-----");
    }

    /**
     * Basic information about the game
     */
    private void intro() {
        System.out.println("PIZZA");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println();
        System.out.println("PIZZA DELIVERY GAME");
        System.out.println();
    }

    private void extendedIntro() {
        System.out.println("THE OUTPUT IS A MAP OF THE HOMES WHERE");
        System.out.println("YOU ARE TO SEND PIZZAS.");
        System.out.println();
        System.out.println("YOUR JOB IS TO GIVE A TRUCK DRIVER");
        System.out.println("THE LOCATION OR COORDINATES OF THE");
        System.out.println("HOME ORDERING THE PIZZA.");
        System.out.println();
    }

    private void displayMoreDirections() {
        System.out.println();
        System.out.println("SOMEBODY WILL ASK FOR A PIZZA TO BE");
        System.out.println("DELIVERED.  THEN A DELIVERY BOY WILL");
        System.out.println("ASK YOU FOR THE LOCATION.");
        System.out.println("     EXAMPLE:");
        System.out.println("THIS IS J.  PLEASE SEND A PIZZA.");
        System.out.println("DRIVER TO " + playerName + ".  WHERE DOES J LIVE?");
        System.out.println("YOUR ANSWER WOULD BE 2,3");
        System.out.println();
    }

    private void init() {
        pizzaDeliveryCount = 1;
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
     * returns true if Y, YES, N, or NO was the compared value in text
     * case-insensitive
     *
     * @param text search string
     * @return true if one of the varargs was found in text
     */
    private boolean yesOrNoEntered(String text) {
        return stringIsAnyValue(text, "Y", "YES", "N", "NO");
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