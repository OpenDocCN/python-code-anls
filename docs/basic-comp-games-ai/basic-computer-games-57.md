# BasicComputerGames源码解析 57

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


# `58_Love/python/love.py`

这段代码是一个批处理程序，用于在命令行界面上输出 "LOVE" 字符组成的图案，每隔 60 个字符换行。这段代码的作用是打印出罗伯特·印第安纳（Robert Indiana）创作的著名作品《爱》（Love）中的 60 个字符，并在需要时换行。


```
"""
LOVE

From: BASIC Computer Games (1978)
      Edited by David H. Ahl

"This program is designed to reproduce Robert Indiana's great art
 work 'Love' with a message of your choice up to 60 characters long.

"The [DATA variable is] an alternating count of the number
 of characters and blanks which form the design.  These data give
 the correct proportions for a standard 10 character per inch
 Teletype or line printer.

"The LOVE program was created by David Ahl."


```

It seems like there might be a mistake in the data you're providing. The numbers don't add up to equal 252, and the sum of the even numbers is 224, which is significantly less than the sum of the odd numbers. Additionally, the data starts from 1, which seems to be a pivot point, and the numbers are all multiples of 1.

If you have accurate data, please provide more of it to help me understand what you're trying to achieve.


```
Python port by Jeff Jetton, 2019
"""


# Image data. Each top-level element is a row. Each row element
# contains alternating character and blank run lengths.
DATA = [
    [
        60,
    ],
    [1, 12, 26, 9, 12],
    [3, 8, 24, 17, 8],
    [4, 6, 23, 21, 6],
    [4, 6, 22, 12, 5, 6, 5],
    [4, 6, 21, 11, 8, 6, 4],
    [4, 6, 21, 10, 10, 5, 4],
    [4, 6, 21, 9, 11, 5, 4],
    [4, 6, 21, 8, 11, 6, 4],
    [4, 6, 21, 7, 11, 7, 4],
    [4, 6, 21, 6, 11, 8, 4],
    [4, 6, 19, 1, 1, 5, 11, 9, 4],
    [4, 6, 19, 1, 1, 5, 10, 10, 4],
    [4, 6, 18, 2, 1, 6, 8, 11, 4],
    [4, 6, 17, 3, 1, 7, 5, 13, 4],
    [4, 6, 15, 5, 2, 23, 5],
    [1, 29, 5, 17, 8],
    [1, 29, 9, 9, 12],
    [1, 13, 5, 40, 1],
    [1, 13, 5, 40, 1],
    [4, 6, 13, 3, 10, 6, 12, 5, 1],
    [5, 6, 11, 3, 11, 6, 14, 3, 1],
    [5, 6, 11, 3, 11, 6, 15, 2, 1],
    [6, 6, 9, 3, 12, 6, 16, 1, 1],
    [6, 6, 9, 3, 12, 6, 7, 1, 10],
    [7, 6, 7, 3, 13, 6, 6, 2, 10],
    [7, 6, 7, 3, 13, 14, 10],
    [8, 6, 5, 3, 14, 6, 6, 2, 10],
    [8, 6, 5, 3, 14, 6, 7, 1, 10],
    [9, 6, 3, 3, 15, 6, 16, 1, 1],
    [9, 6, 3, 3, 15, 6, 15, 2, 1],
    [10, 6, 1, 3, 16, 6, 14, 3, 1],
    [10, 10, 16, 6, 12, 5, 1],
    [11, 8, 13, 27, 1],
    [11, 8, 13, 27, 1],
    [
        60,
    ],
]


```

这段代码的主要作用是接收用户输入的一条消息，并在消息中放置随机生成的随机的表情符号（sic）。然后，它会将消息重复多次，直到收到至少一行用户输入的消息。最后，它会输出一条消息，该消息是基于所接收到的消息和消息长度计算出来的。


```
# Assume that the total length of the first element
# is the line length used by every row
ROW_LEN = sum(DATA[0])


def main() -> None:
    # Display intro text
    print("\n                  Love")
    print("Creative Computing  Morristown, New Jersey")
    print("\n\n")
    print("A tribute to the great American artist, Robert Indiana.")
    print("His great work will be reproduced with a message of")
    print("your choice up to 60 characters.  If you can't think of")
    print("a message, simple type the word 'love'\n")  # (sic)

    # Get message from user
    message = input("Your message, please? ")
    if message == "":
        message = "LOVE"

    # Repeat the message until we get at least one line's worth
    while len(message) < ROW_LEN:
        message += message

    # Display image
    print("\n" * 9)
    for row in DATA:
        print_message = True
        position = 0
        line_text = ""
        for length in row:
            if print_message:
                text = message[position : (position + length)]
                print_message = False
            else:
                text = " " * length
                print_message = True
            line_text += text
            position += length
        print(line_text)

    print()


```

这段代码是一个Python程序中的一个if语句，其作用是判断当前程序是否作为主程序运行。如果是主程序运行，则会执行if语句内部的代码。

if __name__ == "__main__":
   main()

这段代码的作用是判断当前程序是否作为主程序运行。如果当前程序作为主程序运行，则会执行if语句内部的代码。这里的if语句内部包含了一个名为main的函数，它可能是程序中的一个主函数，也可能是其他的作用程序。如果当前程序不是主程序，则会直接跳过if语句，执行if语句内部的代码。


```
if __name__ == "__main__":
    main()


######################################################################
#
# Porting Notes
#
#   Not too different from the original, logic-wise. The image was
#   originally encoded as a series of BASIC "DATA" lines. Here,
#   we've converted it to a more Pythonic nested list structure.
#   Other changes include reducing some of the vertical spacing
#   (since we'll probably be showing this on a screen rather than
#   the sort of tractor-feed printer the program was written for)
#   and having the message default to LOVE when no input is given.
```

这段代码的主要目的是使用简单的run-length编码对一个大写字母图像压缩，实现压缩比率高达8.5-1的效果。

run-length编码是一种对重复字符进行编码的技术。在这个例子中，该编码使用从图像中提取的60个重复的字符('T'和' ')来表示不同的字符。通过对图像进行分割，该编码将每个字符的计数器清零，然后在需要表示该字符时，将计数器增加一步。当图像中的字符重复出现时，计数器会增加，并在计数器中保存该字符的计数信息。最后，图像被压缩为252个数据值，其中包含有60个字符的计数信息，因此压缩比率为8.5-1。

该程序还实现了两个功能：

1. 对用户的消息输入进行处理，从输入中删除空格，并将字符转换为大写形式。
2. 允许用户选择要显示的消息，即使这些消息很大。


```
#
#   This program uses a simple version of run-length encoding to
#   compress a 60 x 36 image (2,160 characters) into just 252 DATA
#   values.  That's about an 8.5-to-1 data compression ratio,
#   which is pretty good!
#
#
# Ideas for Modifications
#
#   Process the user's message input to remove spaces and change
#   to uppercase.
#
#   Encode other images in a similar fashion and let the user choose
#   which one they'd like to use to display their message.
#
```

这段代码是一个用于读取包含类似于字符艺术（*, **,***）的文本文件，并生成包含正确嵌套列表值的 Python 代码的程序。这个程序的输入文件可以包含任意数量和类型的字符艺术。

程序的核心部分是一个无限循环，它将读取文件中的每一行字符艺术，并将其转换成元组类型。然后，程序会遍历这些元组，尝试将其拆分成两个元组，如果成功，就将其添加到正确的列表中。如果尝试失败，就继续循环读取。

在程序的最后，它会将生成的列表输出到控制台，这样用户就可以看到生成的结果。


```
#   To help with the above step, create a program that reads in a
#   text file of any sort of similar character/space art and produces
#   the Python code to initialize the correct nested list of values.
#
#   For example, if the input file were:
#
#     *****
#     *  **
#     **  *
#
#   Your program would output:
#
#    ((5, ), (1, 1, 2), (2, 1, 1))
#
######################################################################

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by Jadi.


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Lunar LEM Rocket

This game in its many different versions and names (ROCKET, LUNAR, LEM, and APOLLO) is by far and away the single most popular computer game. It exists in various versions that start you anywhere from 500 feet to 200 miles away from the moon, or other planets, too. Some allow the control of directional stabilization rockets and/or the retro rocket. The three versions presented here represent the most popular of the many variations.

In most versions of this game, the temptation is to slow up too soon and then have no fuel left for the lower part of the journey. This, of course, is disastrous (as you will find out when you land your own capsule)!

LUNAR was originally in FOCAL by Jim Storer while a student at Lexington High School and subsequently converted to BASIC by David Ahl. ROCKET was written by Eric Peters at DEC and LEM by William Labaree II of Alexandria, Virginia.

In this program, you set the burn rate of the retro rockets (pounds of fuel per second) every 10 seconds and attempt to achieve a soft landing on the moon. 200 lbs/sec really puts the brakes on, and 0 lbs/sec is free fall. Ignition occurs a 8 lbs/sec, so _do not_ use burn rates between 1 and 7 lbs/sec. To make the landing more of a challenge, but more closely approximate the real Apollo LEM capsule, you should make the available fuel at the start (N) equal to 16,000 lbs, and the weight of the capsule (M) equal to 32,500 lbs.

#### LEM
This is the most comprehensive of the three versions and permits you to control the time interval of firing, the thrust, and the attitude angle. It also allows you to work in the metric or English system of measurement. The instructions in the program dialog are very complete, so you shouldn’t have any trouble.

#### ROCKET
In this version, you start 500 feet above the lunar surface and control the burn rate in 1-second bursts. Each unit of fuel slows your descent by 1 ft/sec. The maximum thrust of your engine is 30 ft/sec/sec.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=106)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=121)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Known Bugs

### lem.bas

- The input validation on the thrust value (displayed as P, stored internally as F) appears to be incorrect.  It allows negative values up up to -95, but at -96 or more balks and calls it negative.  I suspect the intent was to disallow any value less than 0 (in keeping with the instructions), *or* nonzero values less than 10.

- The physics calculations seem very sus.  If you enter "1000,0,0" (i.e. no thrust at all, integrating 1000 seconds at a time) four times in a row, you first fall, but then mysteriously gain vertical speed, and end up being lost in space.  This makes no sense.  A similar result happened when just periodically applying 10% thrust in an attempt to hover.


#### Porting Notes

(please note any difficulties or challenges in porting here)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `59_Lunar_LEM_Rocket/javascript/lem.js`

这段代码定义了两个函数，分别是`print`函数和`input`函数。

`print`函数的作用是在屏幕上将文本内容输出到指定的元素中，其接受一个字符串参数，并将其附加给一个由`document.getElementById("output")`创建的文本节点，然后将其插入到文档中。其实现方式是通过`document.createTextNode`创建一个节点，并将其添加到指定的元素中，然后将其插入到文档中，最后将其分割为字符串，创建新的节点并将其插入到元素中。

`input`函数的作用是从用户接收输入的一个字符串，并将其存储在变量`input_str`中，该函数通过在屏幕上显示一个`INPUT`元素，要求用户输入内容并返回一个包含用户输入的字符串。该函数通过使用`document.getElementById("input")`创建一个`INPUT`元素对象，并将其设置为`type="text"`和`length="50"`，然后将`INPUT`元素对象添加到指定的元素中，并设置其`focus`属性。接着，函数监听`keydown`事件，当用户按下了`ESC`键时，函数会获取到当前输入的字符串，并将其存储在`input_str`变量中。最后，函数使用`print`函数将用户输入的字符串输出到屏幕上，并将其记录在`print`函数中。


```
// LEM
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

This is a program that simulates a space program, which is a game where players can control a spaceship to complete various tasks. The program has different levels of difficulty based on the player's performance, and the difficulty level is increased each time the player tries to achieve a higher score.

The program starts by explaining the rules of the game and how to control the spaceship. It then enters a loop where the player tries to control the spaceship to avoid obstacles and reach the goal. The player can move the spaceship left and right using the arrow keys, and they can rotate the spaceship using the up arrow key.

The program also checks for various events that can affect the player's得分. For example, if the player collects too many enemies, the program will tell them to "GO OUTSIDE THE MODULE FOR AN E.V.A." or "IF YOU WANT TO SPIN AROUND, GO OUTSIDE THE MODULE". If the player's spaceship hits an object or another spaceship, they will be given a score. If the player's spaceship hits the ground, the program will tell them to "NOTHING TO SAY".

The program has different levels of difficulty, and the difficulty level is increased each time the player tries to achieve a higher score. The program also has a space program mode, where the player can try to achieve higher scores. The program ends when the player's spaceship hits an object or another spaceship, or when the space program mode is ended.


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
    print(tab(34) + "LEM\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    // ROCKT2 is an interactive game that simulates a lunar
    // landing is similar to that of the Apollo program.
    // There is absolutely no chance involved
    zs = "GO";
    b1 = 1;
    while (1) {
        m = 17.95;
        f1 = 5.25;
        n = 7.5;
        r0 = 926;
        v0 = 1.29;
        t = 0;
        h0 = 60;
        r = r0 + h0;
        a = -3,425;
        r1 = 0;
        a1 = 8.84361e-4;
        r3 = 0;
        a3 = 0;
        m1 = 7.45;
        m0 = m1;
        b = 750;
        t1 = 0;
        f = 0;
        p = 0;
        n = 1;
        m2 = 0;
        s = 0;
        c = 0;
        if (zs == "YES") {
            print("\n");
            print("OK, DO YOU WANT THE COMPLETE INSTRUCTIONS OR THE INPUT -\n");
            print("OUTPUT STATEMENTS?\n");
            while (1) {
                print("1=COMPLETE INSTRUCTIONS\n");
                print("2=INPUT-OUTPUT STATEMENTS\n");
                print("3=NEITHER\n");
                b1 = parseInt(await input());
                qs = "NO";
                if (b1 == 1)
                    break;
                qs = "YES";
                if (b1 == 2 || b1 == 3)
                    break;
            }
        } else {
            print("\n");
            print("LUNAR LANDING SIMULATION\n");
            print("\n");
            print("HAVE YOU FLOWN AN APOLLO/LEM MISSION BEFORE");
            while (1) {
                print(" (YES OR NO)");
                qs = await input();
                if (qs == "YES" || qs == "NO")
                    break;
                print("JUST ANSWER THE QUESTION, PLEASE, ");
            }
        }
        if (qs == "YES") {
            print("\n");
            print("INPUT MEASUREMENT OPTION NUMBER");
        } else {
            print("\n");
            print("WHICH SYSTEM OF MEASUREMENT DO YOU PREFER?\n");
            print(" 1=METRIC     0=ENGLISH\n");
            print("ENTER THE APPROPRIATE NUMBER");
        }
        while (1) {
            k = parseInt(await input());
            if (k == 0 || k == 1)
                break;
            print("ENTER THE APPROPRIATE NUMBER");
        }
        if (k == 1) {
            z = 1852.8;
            ms = "METERS";
            g3 = 3.6;
            ns = " KILOMETERS";
            g5 = 1000;
        } else {
            z = 6080;
            ms = "FEET";
            g3 = 0.592;
            ns = "N.MILES";
            g5 = z;
        }
        if (b1 != 3) {
            if (qs != "YES") {
                print("\n");
                print("  YOU ARE ON A LUNAR LANDING MISSION.  AS THE PILOT OF\n");
                print("THE LUNAR EXCURSION MODULE, YOU WILL BE EXPECTED TO\n");
                print("GIVE CERTAIN COMMANDS TO THE MODULE NAVIGATION SYSTEM.\n");
                print("THE ON-BOARD COMPUTER WILL GIVE A RUNNING ACCOUNT\n");
                print("OF INFORMATION NEEDED TO NAVIGATE THE SHIP.\n");
                print("\n");
                print("\n");
                print("THE ATTITUDE ANGLE CALLED FOR IS DESCRIBED AS FOLLOWS.\n");
                print("+ OR -180 DEGREES IS DIRECTLY AWAY FROM THE MOON\n");
                print("-90 DEGREES IS ON A TANGENT IN THE DIRECTION OF ORBIT\n");
                print("+90 DEGREES IS ON A TANGENT FROM THE DIRECTION OF ORBIT\n");
                print("0 (ZERO) DEGREES IS DIRECTLY TOWARD THE MOON\n");
                print("\n");
                print(tab(30) + "-180|+180\n");
                print(tab(34) + "^\n");
                print(tab(27) + "-90 < -+- > +90\n");
                print(tab(34) + "!\n");
                print(tab(34) + "0\n");
                print(tab(21) + "<<<< DIRECTION OF ORBIT <<<<\n");
                print("\n");
                print(tab(20) + "------ SURFACE OF MOON ------\n");
                print("\n");
                print("\n");
                print("ALL ANGLES BETWEEN -180 AND +180 DEGREES ARE ACCEPTED.\n");
                print("\n");
                print("1 FUEL UNIT = 1 SEC. AT MAX THRUST\n");
                print("ANY DISCREPANCIES ARE ACCOUNTED FOR IN THE USE OF FUEL\n");
                print("FOR AN ATTITUDE CHANGE.\n");
                print("AVAILABLE ENGINE POWER: 0 (ZERO) AND ANY VALUE BETWEEN\n");
                print("10 AND 100 PERCENT.\n");
                print("\n");
                print("NEGATIVE THRUST OR TIME IS PROHIBITED.\n");
                print("\n");
            }
            print("\n");
            print("INPUT: TIME INTERVAL IN SECONDS ------ (T)\n");
            print("       PERCENTAGE OF THRUST ---------- (P)\n");
            print("       ATTITUDE ANGLE IN DEGREES ----- (A)\n");
            print("\n");
            if (qs != "YES") {
                print("FOR EXAMPLE:\n");
                print("T,P,A? 10,65,-60\n");
                print("TO ABORT THE MISSION AT ANY TIME, ENTER 0,0,0\n");
                print("\n");
            }
            print("OUTPUT: TOTAL TIME IN ELAPSED SECONDS\n");
            print("        HEIGHT IN " + ms + "\n");
            print("        DISTANCE FROM LANDING SITE IN " + ms + "\n");
            print("        VERTICAL VELOCITY IN " + ms + "/SECOND\n");
            print("        HORIZONTAL VELOCITY IN " + ms + "/SECOND\n");
            print("        FUEL UNITS REMAINING\n");
            print("\n");
        }
        while (1) {
            for (i = 1; i <= n; i++) {
                if (m1 != 0) {
                    m1 -= m2;
                    if (m1 <= 0) {
                        f = f * (1 + m1 / m2);
                        m2 = m1 + m2;
                        print("YOU ARE OUT OF FUEL.\n");
                        m1 = 0;
                    }
                } else {
                    f = 0;
                    m2 = 0;
                }
                m = m - 0.5 * m2;
                r4 = r3;
                r3 = -0.5 * r0 * Math.pow(v0 / r, 2) + r * a1 * a1;
                r2 = (3 * r3 - r4) / 2 + 0.00526 * f1 * f * c / m;
                a4 = a3;
                a3 = -2 * r1 * a1 / r;
                a2 = (3 * a3 - a4) / 2 + 0.0056 * f1 * f * s / (m * r);
                x = r1 * t1 + 0.5 * r2 * t1 * t1;
                r = r + x;
                h0 = h0 + x;
                r1 = r1 + r2 * t1;
                a = a + a1 * t1 + 0.5 * a2 * t1 * t1;
                a1 = a1 + a2 * t1;
                m = m - 0.5 * m2;
                t = t + t1;
                if (h0 < 3.287828e-4)
                    break;
            }
            h = h0 * z;
            h1 = r1 * z;
            d = r0 * a * z;
            d1 = r * a1 * z;
            t2 = m1 * b / m0;
            print(" " + t + "\t" + h + "\t" + d + "\t" + h1 + "\t" + d1 + "\t" + t2 + "\n");
            if (h0 < 3.287828e-4) {
                if (r1 < -8.21957e-4 || Math.abs(r * a1) > 4.93174e-4 || h0 < -3.287828e-4) {
                    print("\n");
                    print("CRASH !!!!!!!!!!!!!!!!\n");
                    print("YOUR IMPACT CREATED A CRATER " + Math.abs(h) + " " + ms + " DEEP.\n");
                    x1 = Math.sqrt(d1 * d1 + h1 * h1) * g3;
                    print("AT CONTACT YOU WERE TRAVELING " + x1 + " " + ns + "/HR\n");
                    break;
                }
                if (Math.abs(d) > 10 * z) {
                    print("YOU ARE DOWN SAFELY - \n");
                    print("\n");
                    print("BUT MISSED THE LANDING SITE BY " + Math.abs(d / g5) + " " + ns + ".\n");
                    break;
                }
                print("\n");
                print("TRANQUILITY BASE HERE -- THE EAGLE HAS LANDED.\n");
                print("CONGRATULATIONS -- THERE WAS NO SPACECRAFT DAMAGE.\n");
                print("YOU MAY NOW PROCEED WITH SURFACE EXPLORATION.\n");
                break;
            }
            if (r0 * a > 164.474) {
                print("\n");
                print("YOU HAVE BEEN LOST IN SPACE WITH NO HOPE OF RECOVERY.\n");
                break;
            }
            if (m1 > 0) {
                while (1) {
                    print("T,P,A");
                    str = await input();
                    t1 = parseFloat(str);
                    f = parseFloat(str.substr(str.indexOf(",") + 1));
                    p = parseFloat(str.substr(str.lastIndexOf(",") + 1));
                    f = f / 100;
                    if (t1 < 0) {
                        print("\n");
                        print("THIS SPACECRAFT IS NOT ABLE TO VIOLATE THE SPACE-");
                        print("TIME CONTINUUM.\n");
                        print("\n");
                    } else if (t1 == 0) {
                        break;
                    } else if (Math.abs(f - 0.05) > 1 || Math.abs(f - 0.05) < 0.05) {
                        print("IMPOSSIBLE THRUST VALUE ");
                        if (f < 0) {
                            print("NEGATIVE\n");
                        } else if (f - 0.05 < 0.05) {
                            print("TOO SMALL\n");
                        } else {
                            print("TOO LARGE\n");
                        }
                        print("\n");
                    } else if (Math.abs(p) > 180) {
                        print("\n");
                        print("IF YOU WANT TO SPIN AROUND, GO OUTSIDE THE MODULE\n");
                        print("FOR AN E.V.A.\n");
                        print("\n");
                    } else {
                        break;
                    }
                }
                if (t1 == 0) {
                    print("\n");
                    print("MISSION ABENDED\n");
                    break;
                }
            } else {
                t1 = 20;
                f = 0;
                p = 0;
            }
            n = 20;
            if (t1 >= 400)
                n = t1 / 20;
            t1 = t1 / n;
            p = p * 3.14159 / 180;
            s = Math.sin(p);
            c = Math.cos(p);
            m2 = m0 * t1 * f / b;
            r3 = -0.5 * r0 * Math.pow(v0 / r, 2) + r * a1 * a1;
            a3 = -2 * r1 * a1 / r;
        }
        print("\n");
        while (1) {
            print("DO YOU WANT TO TRY IT AGAIN (YES/NO)?\n");
            zs = await input();
            if (zs == "YES" || zs == "NO")
                break;
        }
        if (zs != "YES")
            break;
    }
    print("\n");
    print("TOO BAD, THE SPACE PROGRAM HATES TO LOSE EXPERIENCED\n");
    print("ASTRONAUTS.\n");
}

```

这道题的代码是一个C语言的主函数（main function），也就是程序的入口点。在main函数中，程序会首先被调用，然后进入一个循环，如果有满足特定条件的语句，就会执行这些语句。

对于这道题，虽然没有给出具体的代码，但是我们可以根据常见的main函数作用来推测它的作用。

通常情况下，main函数以下面的形式：

```c
int main(int argc, char **argv) {
   // 定义函数的作用域
   // 在main函数中可以定义局部变量，但是不推荐
   // 函数内最好不要使用全局变量
   // 可以根据需要使用系统提供的函数，如system()
   // 可以根据需要输出帮助信息，例如：printf("Hello World!")
   // 返回0，表示程序成功运行，也可以返回其他值，如1、-1等
   return 0;
}
```

从这个规范来看，这道题的代码可能是为了输出 "Hello World!" 这样的信息，告诉用户程序成功运行。当然，具体的作用还需要根据上下文来判断。


```
main();

```

# `59_Lunar_LEM_Rocket/javascript/lunar.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。

`print` 函数的作用是接收一个字符串参数，将其中的文本添加到页面上一个有一个自定义 id 为 "output" 的 div 元素中。

`input` 函数的作用是接收一个字符串参数，返回一个 Promise 对象。该函数使用户可以输入字符，并在输入框中保存输入的字符串，之后可以将其提取出来并输出。该函数会在用户输入时监控键盘事件，当用户按下回车键时，将提取的字符串打印到页面上并将其从 input 元素中移除。


```
// LUNAR
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

这段代码定义了一个名为 "tab" 的函数，接受一个参数 "space"。这个函数的作用是在字符串中打印出指定数量的空白字符，直到指定的空间位置没有空白字符为止。

在函数内部，首先定义了一个名为 "str" 的字符串变量，并使用一个 while 循环来打印出指定数量的空白字符。每次循环，将一个空白字符打印到 "str" 字符串的末尾，并将 "space" 变量减少 1。

在代码的最后，通过 return 语句返回了 "str" 字符串，但没有对其进行任何修改。

在代码的后面，定义了几个变量 l、t、m、s、k、a、v，但没有对它们进行使用。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var l;
var t;
var m;
var s;
var k;
var a;
var v;
```

这段代码定义了一个名为"formula_set_1"的函数。

函数内部定义了五个变量：i、j、q、g、z和一个名为d的变量。

接着在函数体内部，对这五个变量进行了修改，分别加入了它们与一个名为s的变量的乘积。

然后，对五个变量再次进行了修改，分别减少了它们与一个名为k的变量的乘积，并加入了一个名为i的变量。

最后，返回了修改后的一组变量。


```
var i;
var j;
var q;
var g;
var z;
var d;

function formula_set_1()
{
    l = l + s;
    t = t - s;
    m = m - s * k;
    a = i;
    v = j;
}

```

这两个函数的主要目的是计算两个变量 q 和 s 的值，其中 q 是 s 的某个函数，s 是一个正整数。

函数 formula_set_2() 的作用是计算 q 的值，公式较为复杂，但可以较为精确地计算出 q 的值。

函数 formula_set_3() 的作用是计算给定整数 s 下的第 5 行方程的解，即先计算出前 4 行方程的解，第 5 行方程的解。

具体来说，函数 formula_set_2() 的实现过程如下：

1. 根据给定的公式，计算出 s 的值。
2. 根据第 3 行的公式，计算出 v 的值。
3. 根据第 2 行的公式，计算出 z 的值。
4. 根据第 1 行的公式，计算出 a、g 和 m 的值。
5. 根据第 1 行的公式，带入计算出 k 和 m 的值。
6. 根据第 3 行的公式，带入计算出 d 的值。
7. 根据第 2 行的公式，带入计算出 s 的值。
8. 根据第 4 行的公式，带入计算出 v 的值。

函数 formula_set_3() 的实现过程较为复杂，但保证了在给定整数 s 的情况下，第 5 行方程有唯一解。


```
function formula_set_2()
{
    q = s * k / m;
    j = v + g * s + z * (-q - q * q / 2 - Math.pow(q, 3) / 3 - Math.pow(q, 4) / 4 - Math.pow(q, 5) / 5);
    i = a - g * s * s / 2 - v * s + z * s * (q / 2 + Math.pow(q, 2) / 6 + Math.pow(q, 3) / 12 + Math.pow(q, 4) / 20 + Math.pow(q, 5) / 30);
}

function formula_set_3()
{
    while (s >= 5e-3) {
        d = v + Math.sqrt(v * v + 2 * a * (g - z * k / m));
        s = 2 * a / d;
        formula_set_2();
        formula_set_1();
    }
}

```

This is a code snippet for a game where players are given the task of landing on the moon. The player must use a mecanism to control the lander's velocity and apply enough force to achieve a stable landing. The game has different landing scenarios based on the player's choices and the actual landing distance.



```
// Main program
async function main()
{
    print(tab(33) + "LUNAR\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("THIS IS A COMPUTER SIMULATION OF AN APOLLO LUNAR\n");
    print("LANDING CAPSULE.\n");
    print("\n");
    print("\n");
    print("THE ON-BOARD COMPUTER HAS FAILED (IT WAS MADE BY\n");
    print("XEROX) SO YOU HAVE TO LAND THE CAPSULE MANUALLY.\n");
    while (1) {
        print("\n");
        print("SET BURN RATE OF RETRO ROCKETS TO ANY VALUE BETWEEN\n");
        print("0 (FREE FALL) AND 200 (MAXIMUM BURN) POUNDS PER SECOND.\n");
        print("SET NEW BURN RATE EVERY 10 SECONDS.\n");
        print("\n");
        print("CAPSULE WEIGHT 32,500 LBS; FUEL WEIGHT 16,000 LBS.\n");
        print("\n");
        print("\n");
        print("\n");
        print("GOOD LUCK\n");
        l = 0;
        print("\n");
        print("SEC\tMI + FT\t\tMPH\tLB FUEL\tBURN RATE\n");
        print("\n");
        a = 120;
        v = 1;
        m = 32500;
        n = 16500;
        g = 1e-3;
        z = 1.8;
        while (1) {
            print(l + "\t" + Math.floor(a) + " + " + Math.floor(5280 * (a - Math.floor(a))) + " \t" + Math.floor(3600 * v * 100) / 100 + "\t" + (m - n) + "\t");
            k = parseFloat(await input());
            t = 10;
            should_exit = false;
            while (1) {
                if (m - n < 1e-3)
                    break;
                if (t < 1e-3)
                    break;
                s = t;
                if (m < n + s * k)
                    s = (m - n) / k;
                formula_set_2();
                if (i <= 0) {
                    formula_set_3();
                    should_exit = true;
                    break;
                }
                if (v > 0) {
                    if (j < 0) {
                        do {
                            w = (1 - m * g / (z * k)) / 2;
                            s = m * v / (z * k * (w + Math.sqrt(w * w + v / z))) + 0.05;
                            formula_set_2();
                            if (i <= 0) {
                                formula_set_3();
                                should_exit = true;
                                break;
                            }
                            formula_set_1();
                            if (j > 0)
                                break;
                        } while (v > 0) ;
                        if (should_exit)
                            break;
                        continue;
                    }
                }
                formula_set_1();
            }
            if (should_exit)
                break;
            if (m - n < 1e-3) {
                print("FUEL OUT AT " + l + " SECOND\n");
                s = (-v * Math.sqrt(v * v + 2 * a * g)) / g;
                v = v + g * s;
                l = l + s;
                break;
            }
        }
        w = 3600 * v;
        print("ON MOON AT " + l + " SECONDS - IMPACT VELOCITY " + w + " MPH\n");
        if (w <= 1.2) {
            print("PERFECT LANDING!\n");
        } else if (w <= 10) {
            print("GOOD LANDING (COULD BE BETTER)\n");
        } else if (w <= 60) {
            print("CRAFT DAMAGE... YOU'RE STRANDED HERE UNTIL A RESCUE\n");
            print("PARTY ARRIVES. HOPE YOU HAVE ENOUGH OXYGEN!\n");
        } else {
            print("SORRY THERE WERE NO SURVIVORS. YOU BLEW IT!\n");
            print("IN FACT, YOU BLASTED A NEW LUNAR CRATER " + (w * 0.227) + " FEET DEEP!\n");
        }
        print("\n");
        print("\n");
        print("\n");
        print("TRY AGAIN??\n");
    }
}

```

这是C++中的一个标准函数，名为`main()`。在C++程序中，`main()`函数是程序的入口点，也是程序开始执行的地方。无论程序是否成功，`main()`函数的返回码（即0或1）都将决定程序是成功还是失败。

函数体中没有输出语句，所以不会输出任何东西。


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


# `59_Lunar_LEM_Rocket/python/lunar.py`

这段代码是一个字符串，表示一个关于月球着陆模拟器的介绍。没有实际的代码实现。


```
"""
LUNAR

Lunar landing simulation

Ported by Dave LeCompte
"""

import math
from dataclasses import dataclass
from typing import Any, NamedTuple

PAGE_WIDTH = 64

COLUMN_WIDTH = 2
```

这段代码定义了几个常量和变量，用于计算飞行中不同的宽度参数。

SECONDS_WIDTH = 4，表示每个秒的宽度为4。

MPH_WIDTH = 6，表示每小时前进的宽度为6。

ALT_MI_WIDTH = 6，表示每分钟的宽度为6，单位为米。

ALT_FT_WIDTH = 4，表示每英尺的宽度为4，单位为英尺。

MPH_WIDTH = 6，表示每小时前进的宽度为6。

FUEL_WIDTH = 8，表示每个燃油单位的宽度为8。

BURN_WIDTH = 10，表示每个燃烧单位的宽度为10。

SECONDS_LEFT = 0，表示每分钟的左侧时间步长为0。

SECONDS_RIGHT = SECONDS_LEFT + SECONDS_WIDTH，表示每分钟的右侧时间步长为每分钟的左侧时间步长加上每分钟的宽度。

ALT_LEFT = SECONDS_RIGHT + COLUMN_WIDTH，表示每分钟的左侧位置为每分钟的右侧位置加上每分钟的宽度，单位为米。

ALT_MI_RIGHT = ALT_LEFT + ALT_MI_WIDTH，表示每分钟的米位位置为每分钟的左侧位置加上每分钟的宽度，单位为米。

ALT_FT_RIGHT = ALT_MI_RIGHT + COLUMN_WIDTH + ALT_FT_WIDTH，表示每分钟的米英尺位置为每分钟的米位位置加上每分钟的宽度加上每英尺的宽度，单位为英尺。

MPH_LEFT = ALT_FT_RIGHT + COLUMN_WIDTH，表示每分钟的左侧位置为每分钟的右侧位置加上每分钟的宽度，单位为米。

MPH_RIGHT = MPH_LEFT + MPH_WIDTH，表示每分钟的右侧位置为每分钟的右侧位置加上每分钟的宽度，单位为米。


```
SECONDS_WIDTH = 4
MPH_WIDTH = 6
ALT_MI_WIDTH = 6
ALT_FT_WIDTH = 4
MPH_WIDTH = 6
FUEL_WIDTH = 8
BURN_WIDTH = 10

SECONDS_LEFT = 0
SECONDS_RIGHT = SECONDS_LEFT + SECONDS_WIDTH
ALT_LEFT = SECONDS_RIGHT + COLUMN_WIDTH
ALT_MI_RIGHT = ALT_LEFT + ALT_MI_WIDTH
ALT_FT_RIGHT = ALT_MI_RIGHT + COLUMN_WIDTH + ALT_FT_WIDTH
MPH_LEFT = ALT_FT_RIGHT + COLUMN_WIDTH
MPH_RIGHT = MPH_LEFT + MPH_WIDTH
```

这段代码定义了三个变量：FUEL_LEFT，FUEL_RIGHT 和 BURN_LEFT，BURN_RIGHT。它们都是燃料相关的变量。然后，该代码定义了一个名为 PhysicalState 的类，该类包含燃料的速率（velocity）和高度（altitude）。

接下来，代码实现了一个 print_centered 函数，该函数接受一个字符串参数 msg，并在页面的中间部分居中打印该消息。在实际应用中，这个函数可以用来在显示窗口中显示重要信息，如按钮提示或者标签。


```
FUEL_LEFT = MPH_RIGHT + COLUMN_WIDTH
FUEL_RIGHT = FUEL_LEFT + FUEL_WIDTH
BURN_LEFT = FUEL_RIGHT + COLUMN_WIDTH
BURN_RIGHT = BURN_LEFT + BURN_WIDTH


class PhysicalState(NamedTuple):
    velocity: float
    altitude: float


def print_centered(msg: str) -> None:
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    print(spaces + msg)


```

这段代码定义了两个函数，分别是 `print_header` 和 `add_rjust`。它们的作用是：

1. `print_header` 函数的作用是在屏幕上打印一个标题，然后将其左右居中。接着打印一行垂直居中、然后又向上居中的行。最后没有返回值。
2. `add_rjust` 函数的作用是在一个给定的行字符串 `line` 和一个给定的偏移量 `pos` 之间添加一个新的字段，将该字段右 justification 到行结束的位置。它返回一个新的字符串，其中新字段的内容等于 `line` 中的内容加上给定偏移量的字符串 `s`。

`print_header` 函数中的 `print_centered` 函数的作用是在屏幕上打印字符，并将其左右居中。这个函数的实现可能是在 Unix 传统的 ASCII 字符集中查找的，因为这些字符有特定的字符集中的字符来实现左右居中。在 Windows 平台上，可以使用 `coreprint` 库来实现。

`add_rjust` 函数的作用是在一个给定的行字符串 `line` 和一个给定的偏移量 `pos` 之间添加一个新的字段，将该字段右 justification 到行结束的位置。它返回一个新的字符串，其中新字段的内容等于 `line` 中的内容加上给定偏移量的字符串 `s`。


```
def print_header(title: str) -> None:
    print_centered(title)
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")


def add_rjust(line: str, s: Any, pos: int) -> str:
    """Add a new field to a line right justified to end at pos"""
    s_str = str(s)
    slen = len(s_str)
    if len(line) + slen > pos:
        new_len = pos - slen
        line = line[:new_len]
    if len(line) + slen < pos:
        spaces = " " * (pos - slen - len(line))
        line = line + spaces
    return line + s_str


```

这段代码是一个函数，名为 `add_ljust`，它接受三个参数 `line`、`s` 和 `pos`。函数的功能是在一条左对齐的线路上，将一个字符串 `s` 添加到 `line` 的起始位置 `pos` 处，并将结果返回。

函数首先将参数 `s` 转换成字符串类型，然后检查 `line` 长度是否大于 `pos`。如果是，则将 `line` 的前 `pos` 个字符截去，然后将 `s` 和截去的部分拼接在一起，得到一个新的字符串，将起始位置 `pos` 和 `s` 拼接在一起，并将结果返回。如果是，则需要在 `spaces` 变量中填入 `"_ "`，然后将 `pos` 和 `-len(line)` 的字符串连接起来，再将 `s` 和连接的字符填充 `spaces`，最后将 `spaces` 和 `pos` 之间的字符串替换为空格，得到一个新的字符串，将起始位置 `pos` 和 `s` 拼接在一起，并将结果返回。

函数还定义了一个名为 `print_instructions` 的函数，该函数打印出一些文字说明。


```
def add_ljust(line: str, s: str, pos: int) -> str:
    """Add a new field to a line left justified starting at pos"""
    s = str(s)
    if len(line) > pos:
        line = line[:pos]
    if len(line) < pos:
        spaces = " " * (pos - len(line))
        line = line + spaces
    return line + s


def print_instructions() -> None:
    """Somebody had a bad experience with Xerox."""
    print("THIS IS A COMPUTER SIMULATION OF AN APOLLO LUNAR")
    print("LANDING CAPSULE.\n\n")
    print("THE ON-BOARD COMPUTER HAS FAILED (IT WAS MADE BY")
    print("XEROX) SO YOU HAVE TO LAND THE CAPSULE MANUALLY.\n")


```

这段代码定义了一个名为 print_intro 的函数，该函数不会输出任何值，但是会输出一些信息来帮助调试。函数的作用是在程序运行时设置火箭发动机的燃料和 burn 速率。

具体来说，函数中首先输出了一系列关于火箭发动机设置的信息，包括燃速范围、新燃速设置以及燃料重量。接着，函数会输出一些关于如何编写报告的信息，包括如何将数据格式化等。

然后，函数定义了一个名为 format_line_for_report 的函数，该函数接收输入的一些参数，包括当前火箭发动机的燃速、燃料重量、报告格式等。函数的作用是将输入的信息格式化并输出，以便用户更容易地阅读和理解。例如，函数会根据输入的燃速来输出不同的格式，包括如何显示燃速、燃料重量等。


```
def print_intro() -> None:
    print("SET BURN RATE OF RETRO ROCKETS TO ANY VALUE BETWEEN")
    print("0 (FREE FALL) AND 200 (MAXIMUM BURN) POUNDS PER SECOND.")
    print("SET NEW BURN RATE EVERY 10 SECONDS.\n")
    print("CAPSULE WEIGHT 32,500 LBS; FUEL WEIGHT 16,000 LBS.\n\n\n")
    print("GOOD LUCK\n")


def format_line_for_report(
    t: Any,
    miles: Any,
    feet: Any,
    velocity: Any,
    fuel: Any,
    burn_rate: str,
    is_header: bool,
) -> str:
    line = add_rjust("", t, SECONDS_RIGHT)
    line = add_rjust(line, miles, ALT_MI_RIGHT)
    line = add_rjust(line, feet, ALT_FT_RIGHT)
    line = add_rjust(line, velocity, MPH_RIGHT)
    line = add_rjust(line, fuel, FUEL_RIGHT)
    if is_header:
        line = add_rjust(line, burn_rate, BURN_RIGHT)
    else:
        line = add_ljust(line, burn_rate, BURN_LEFT)
    return line


```

This code appears to be a simplified physics engine that simulates an aircraft's motion. It has a fuel system that provides a limited amount of fuel per second, which is used to calculate the velocity and altitude of the aircraft. The aircraft can also make Euler's Method calculations of its motion, and can predict the altitude and velocity of the aircraft based on its current position and velocity. The code includes a number of functions for managing the aircraft's fuel, making predictions, and displaying information about the aircraft's state. It also includes a prompt for the user to input a burn amount, which is used to calculate the aircraft's thrust.


```
class SimulationClock:
    def __init__(self, elapsed_time: float, time_until_next_prompt: float) -> None:
        self.elapsed_time = elapsed_time
        self.time_until_next_prompt = time_until_next_prompt

    def time_for_prompt(self) -> bool:
        return self.time_until_next_prompt < 1e-3

    def advance(self, delta_t: float) -> None:
        self.elapsed_time += delta_t
        self.time_until_next_prompt -= delta_t


@dataclass
class Capsule:
    altitude: float = 120  # in miles above the surface
    velocity: float = 1  # downward
    m: float = 32500  # mass_with_fuel
    n: float = 16500  # mass_without_fuel
    g: float = 1e-3
    z: float = 1.8
    fuel_per_second: float = 0

    def remaining_fuel(self) -> float:
        return self.m - self.n

    def is_out_of_fuel(self) -> bool:
        return self.remaining_fuel() < 1e-3

    def update_state(
        self, sim_clock: SimulationClock, delta_t: float, new_state: PhysicalState
    ) -> None:
        sim_clock.advance(delta_t)
        self.m = self.m - delta_t * self.fuel_per_second
        self.altitude = new_state.altitude
        self.velocity = new_state.velocity

    def fuel_time_remaining(self) -> float:
        # extrapolates out how many seconds we have at the current fuel burn rate
        assert self.fuel_per_second > 0
        return self.remaining_fuel() / self.fuel_per_second

    def predict_motion(self, delta_t: float) -> PhysicalState:
        # Perform an Euler's Method numerical integration of the equations of motion.

        q = delta_t * self.fuel_per_second / self.m

        # new velocity
        new_velocity = (
            self.velocity
            + self.g * delta_t
            + self.z * (-q - q**2 / 2 - q**3 / 3 - q**4 / 4 - q**5 / 5)
        )

        # new altitude
        new_altitude = (
            self.altitude
            - self.g * delta_t**2 / 2
            - self.velocity * delta_t
            + self.z
            * delta_t
            * (q / 2 + q**2 / 6 + q**3 / 12 + q**4 / 20 + q**5 / 30)
        )

        return PhysicalState(altitude=new_altitude, velocity=new_velocity)

    def make_state_display_string(self, sim_clock: SimulationClock) -> str:
        seconds = sim_clock.elapsed_time
        miles = int(self.altitude)
        feet = int(5280 * (self.altitude - miles))
        velocity = int(3600 * self.velocity)
        fuel = int(self.remaining_fuel())
        burn_rate = " ? "

        return format_line_for_report(
            seconds, miles, feet, velocity, fuel, burn_rate, False
        )

    def prompt_for_burn(self, sim_clock: SimulationClock) -> None:
        msg = self.make_state_display_string(sim_clock)

        self.fuel_per_second = float(input(msg))
        sim_clock.time_until_next_prompt = 10


```

这段代码是一个名为 `show_landing` 的函数，它接受两个参数 `sim_clock` 和 `capsule`。函数的作用是在模拟时间 `sim_clock`  elapsed 时间后输出一些信息，然后根据 `w` 的值输出不同的消息。

具体来说，函数首先计算了胶囊 `capsule` 的速度 `w`，然后根据 `w` 的值输出不同的消息。如果 `w` 小于 1.2，那么输出 "ON MOON AT {sim_clock.elapsed_time:.2f} SECONDS - IMPACT VELOCITY {w:.2f} MPH"，表示胶囊已经触地，但是速度较慢。如果 `w` 小于 10，那么输出 "GOOD LANDING (COULD BE BETTER)"，表示胶囊已经触地，但是速度较快，而且还有可能继续飞行。如果 `w` 在 1 到 60 之间，那么输出 "CRAFT DAMAGE... YOU'RE STRANDED HERE UNTIL A RESCUE"，表示胶囊已经坠毁，可能有一些损坏，需要等待救援。如果 `w` 大于 60，那么输出 "SORRY THERE WERE NO SURVIVORS. YOU BLEW IT！"，表示胶囊已经爆炸，所有人都不幸死亡。最后，函数调用 `end_sim()` 函数来结束模拟。


```
def show_landing(sim_clock: SimulationClock, capsule: Capsule) -> None:
    w = 3600 * capsule.velocity
    print(
        f"ON MOON AT {sim_clock.elapsed_time:.2f} SECONDS - IMPACT VELOCITY {w:.2f} MPH"
    )
    if w < 1.2:
        print("PERFECT LANDING!")
    elif w < 10:
        print("GOOD LANDING (COULD BE BETTER)")
    elif w <= 60:
        print("CRAFT DAMAGE... YOU'RE STRANDED HERE UNTIL A RESCUE")
        print("PARTY ARRIVES. HOPE YOU HAVE ENOUGH OXYGEN!")
    else:
        print("SORRY THERE WERE NO SURVIVORS. YOU BLEW IT!")
        print(f"IN FACT, YOU BLASTED A NEW LUNAR CRATER {w*.227:.2f} FEET DEEP!")
    end_sim()


```



该代码定义了两个函数：`show_out_of_fuel` 和 `process_final_tick`。这两个函数都是与胶囊对象(Capsule)相关的。

`show_out_of_fuel` 函数的作用是在模拟时间(SimulationClock)的流逝时间内打印出“FUEL OUT”的消息，并显示当前胶囊的速度。

`process_final_tick` 函数的作用是在每次模拟时间(SimulationClock)的流逝过程中，更新胶囊对象(Capsule)的状态。这个函数会根据胶囊的速度和 delta_t 的大小来决定是否进行位置更新。如果 delta_t 小于 5e-3，则直接调用 `show_landing` 函数进行位置更新，并返回。否则，函数会计算出平均速度，并更新胶囊对象的状态。


```
def show_out_of_fuel(sim_clock: SimulationClock, capsule: Capsule) -> None:
    print(f"FUEL OUT AT {sim_clock.elapsed_time} SECONDS")
    delta_t = (
        -capsule.velocity
        + math.sqrt(capsule.velocity**2 + 2 * capsule.altitude * capsule.g)
    ) / capsule.g
    capsule.velocity += capsule.g * delta_t
    sim_clock.advance(delta_t)
    show_landing(sim_clock, capsule)


def process_final_tick(
    delta_t: float, sim_clock: SimulationClock, capsule: Capsule
) -> None:
    # When we extrapolated our position based on our velocity
    # and delta_t, we overshot the surface. For better
    # accuracy, we will back up and do shorter time advances.

    while True:
        if delta_t < 5e-3:
            show_landing(sim_clock, capsule)
            return
        # line 35
        average_vel = (
            capsule.velocity
            + math.sqrt(
                capsule.velocity**2
                + 2
                * capsule.altitude
                * (capsule.g - capsule.z * capsule.fuel_per_second / capsule.m)
            )
        ) / 2
        delta_t = capsule.altitude / average_vel
        new_state = capsule.predict_motion(delta_t)
        capsule.update_state(sim_clock, delta_t, new_state)


```

这段代码是一个名为`handle_flyaway`的函数，它用于在模拟中处理胶囊的起飞和着陆。

在代码中，首先定义了一个胶囊对象`capsule`，该对象包含了胶囊的初始位置、速度、燃料量以及重力。

接着定义了一个名为`sim_clock`的模拟时钟`SimulationClock`，该对象用于控制模拟的时间步长。

在`handle_flyaway`函数中，首先判断胶囊是否已经着陆，如果已着陆，则返回`True`；如果模拟还没有结束，则继续模拟。

模拟过程中，先计算胶囊在当前时间步的加速度，然后再根据当前加速度和胶囊的初始速度以及燃料量计算出胶囊在预测时间步的位置。

接着判断胶囊是否预测到达最高点，如果到达最高点则返回`True`，否则继续模拟。

最后，如果新的速度为正或者为负，则说明胶囊已经恢复到原来的状态，返回`False`。

总的来说，这段代码主要目的是处理胶囊在模拟中的起飞和着陆，确保胶囊在模拟结束时能够正确判断并返回相应的结果。


```
def handle_flyaway(sim_clock: SimulationClock, capsule: Capsule) -> bool:
    """
    The user has started flying away from the moon. Since this is a
    lunar LANDING simulation, we wait until the capsule's velocity is
    positive (downward) before prompting for more input.

    Returns True if landed, False if simulation should continue.
    """

    while True:
        w = (1 - capsule.m * capsule.g / (capsule.z * capsule.fuel_per_second)) / 2
        delta_t = (
            capsule.m
            * capsule.velocity
            / (
                capsule.z
                * capsule.fuel_per_second
                * math.sqrt(w**2 + capsule.velocity / capsule.z)
            )
        ) + 0.05

        new_state = capsule.predict_motion(delta_t)

        if new_state.altitude <= 0:
            # have landed
            return True

        capsule.update_state(sim_clock, delta_t, new_state)

        if (new_state.velocity > 0) or (capsule.velocity <= 0):
            # return to normal sim
            return False


```

这段代码是一个人工智能实验室中常见的研究者与模拟器交互的一个函数。

end_sim()函数是结束模拟的函数，它通过调用try again来重新进行模拟。

run_simulation()函数是一个模拟研究者进行实验的函数。

在这个函数中，首先会输出一些信息，包括一些报告和数据的格式，这些信息对于后续的实验研究非常有用。

然后创建了一个模拟钟，这个钟可以模拟研究者进行实验的时间。

接下来，创建了一个燃料盒，这个盒子里可以存储燃料，这个盒子的容量和燃料的用量是通过prompt_for_burn函数进行管理的。

然后，prompt_for_burn函数会向模拟钟询问研究者是否有足够的燃料在盒子里，然后继续进行实验。

在while循环中，会检查capsule对象中是否还有足够的燃料。

如果capsule.is_out_of_fuel()：会输出缺少燃料的警告信息，并返回尝试之前。

如果sim_clock.time_for_prompt()：会询问模拟器是否有足够的时间来进行实验，如果足够，就继续进行。

如果sim_clock.time_until_next_prompt()：会询问模拟器下一次可以进行实验的时间，如果足够，就继续进行。

如果capsule.fuel_per_second > 0：会根据当前时间，向模拟器提供燃料，并让模拟器继续进行实验。

否则，就尝试使用之前存储的燃料。

接着，new_state对象则是根据当前时间，根据capsule.predict_motion函数所预测出来的新的状态。

如果新的状态中的altitude <= 0，就表示研究者的飞船已经脱离了月球的引力，然后执行一些额外的操作。

如果新的状态.velocity > 0 和 new_state.velocity < 0，表示研究者的飞船正在向着月球返航。

如果研究者的飞船正在移动，而且新的状态中的altitude > 0，那么就表示研究者的飞船已经着落在了月球表面，然后执行一些额外的操作。

最后，handle_flyaway函数会在研究者的飞船脱离月球的引力时执行一些额外的操作。


```
def end_sim() -> None:
    print("\n\n\nTRY AGAIN??\n\n\n")


def run_simulation() -> None:
    print()
    print(
        format_line_for_report("SEC", "MI", "FT", "MPH", "LB FUEL", "BURN RATE", True)
    )

    sim_clock = SimulationClock(0, 10)
    capsule = Capsule()

    capsule.prompt_for_burn(sim_clock)

    while True:
        if capsule.is_out_of_fuel():
            show_out_of_fuel(sim_clock, capsule)
            return

        if sim_clock.time_for_prompt():
            capsule.prompt_for_burn(sim_clock)
            continue

        # clock advance is the shorter of the time to the next prompt,
        # or when we run out of fuel.
        if capsule.fuel_per_second > 0:
            delta_t = min(
                sim_clock.time_until_next_prompt, capsule.fuel_time_remaining()
            )
        else:
            delta_t = sim_clock.time_until_next_prompt

        new_state = capsule.predict_motion(delta_t)

        if new_state.altitude <= 0:
            process_final_tick(delta_t, sim_clock, capsule)
            return

        if capsule.velocity > 0 and new_state.velocity < 0:
            # moving away from the moon

            landed = handle_flyaway(sim_clock, capsule)
            if landed:
                process_final_tick(delta_t, sim_clock, capsule)
                return

        else:
            capsule.update_state(sim_clock, delta_t, new_state)


```

这段代码是一个Python程序，名为“main”。它导入了两个函数：“print_header”和“print_instructions”，以及一个名为“run_simulation”的函数。程序的主要作用是让用户运行程序，并在程序结束时输出一些信息。

具体来说，这段代码的作用如下：

1. 首先，程序会调用一个名为“print_header”的函数，这个函数的作用是输出一个“LUNAR”的标题。

2. 接着，程序会调用一个名为“print_instructions”的函数，这个函数的作用是输出一些关于如何使用这个程序的说明信息。

3. 在循环中，程序会调用一个名为“print_intro”的函数，这个函数的作用是在循环开始时输出一些信息。

4. 然后，程序会调用一个名为“run_simulation”的函数，这个函数的作用是运行一个模拟程序。

5. 在循环的每一次迭代中，程序会先调用“print_intro”函数，然后在调用“run_simulation”函数之前输出一些信息。

6. 循环会一直运行，直到程序被手动中断。


```
def main() -> None:
    print_header("LUNAR")
    print_instructions()
    while True:
        print_intro()
        run_simulation()


if __name__ == "__main__":
    main()

```