# BasicComputerGames源码解析 43

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)

There are 2 compiled executables in the compiled/ directory (windows and linux) that you can play right away!

Program.cs contains the C# source code.
It has been written for .NET Core 3.1

The source code is well documented.


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `39_Golf/javascript/golf.js`

这段代码实现了两个函数：`print()` 和 `input()`。

1. `print()` 函数的作用是将一个字符串 `str` 输出到网页上的一个元素中，该元素具有一个名为 "output" 的 id。

2. `input()` 函数的作用是从用户的输入中获取一个字符串 `input_str`，并将其存储在一个变量中。该函数通过调用 `print()` 函数将输入的字符串输出，并将其存储在 `input_str` 变量中。

函数 `input()` 的实现较为复杂，但主要目的是从用户的输入中获取一个字符串，并将其输出到网页上。它通过创建一个带有输入字段的输入元素，然后监听该元素的 `keydown` 事件，以便在用户按下回车键时获取输入的字符串。当获取到字符串后，函数调用 `print()` 函数将其输出，并将其存储在 `input_str` 变量中。最后，函数将 `input_str` 内容输出，并在调用 `print()` 函数后自动将其从内存中清除。


```
// GOLF
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

这段代码定义了一个名为 `tab` 的函数，接受一个参数 `space`，用于指定输出字符的数量。函数内部，使用一个变量 `str` 来存储输入参数 `space` 减1的结果，然后使用 while 循环从 `space` 开始，每次循环将一个空格添加到 `str` 的开头，并将 `space` 减少1。循环一直执行到 `space` 等于0为止。

代码中还定义了几个变量 `la`、`f`、`s1`、`g2`、`g3` 和 `x`，但它们并没有在函数中使用，或者含义与函数本身无关。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var la = [];
var f;
var s1;
var g2;
var g3;
var x;

```

这段代码定义了一个名为`show_obstacle`的函数，它接受一个整数数组`hole_data`作为参数。

在函数内部，首先通过一个名为`la`的整数变量，获取当前函数执行的索引。然后，通过switch语句判断当前索引对应的障碍物类型，并输出相应的提示信息。

具体来说，当`la[x]`的值为1时，输出"FAIRWAY."；当值为2时，输出"ROUGH."；当值为3时，输出"TREES."；当值为4时，输出"ADJACENT FAIRWAY."；当值为5时，输出"TRAP."；当值为6时，输出"WATER."。


```
var hole_data = [
    361,4,4,2,389,4,3,3,206,3,4,2,500,5,7,2,
    408,4,2,4,359,4,6,4,424,4,4,2,388,4,4,4,
    196,3,7,2,400,4,7,2,560,5,7,2,132,3,2,2,
    357,4,4,4,294,4,2,4,475,5,2,3,375,4,4,2,
    180,3,6,2,550,5,6,6,
];

function show_obstacle()
{
    switch (la[x]) {
        case 1:
            print("FAIRWAY.\n");
            break;
        case 2:
            print("ROUGH.\n");
            break;
        case 3:
            print("TREES.\n");
            break;
        case 4:
            print("ADJACENT FAIRWAY.\n");
            break;
        case 5:
            print("TRAP.\n");
            break;
        case 6:
            print("WATER.\n");
            break;
    }
}

```

In this program, the code is trying to determine who will win a game of cards based on the values of the cards. The program has different routines for different situations, and each routine has a series of commands that the user must follow in order to choose their card and make their move.

The program starts by explaining the rules of the game, and then prompting the user to choose their card by printing a list of available cards and asking them to choose one. After the user has chosen their card, the program checks whether their card is powerful enough to win the game, and if not, then the user must choose a different card.

If the user chooses a card that is powerful enough to win the game, the program enters a loop where the user makes their moves and the computer checks if their opponent's cards are powerful enough to match them. The program also has different routines for different situations, such as when the user's opponent is playing from the driver's seat or when the user's opponent is playing from the mechanic's seat.

The program also has some error handling built-in to try to ensure that the user is not telling the computer to choose a card that is not powerful enough to win the game.

Overall, the program is designed to be a simple and informative tool for determining who will win the game of cards.



```
function show_score()
{
    g2 += s1;
    print("TOTAL PAR FOR " + (f - 1) + " HOLES IS " + g3 + "  YOUR TOTAL IS " + g2 + "\n");
}

// Main program
async function main()
{
    print(tab(34) + "GOLF\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("WELCOME TO THE CREATIVE COMPUTING COUNTRY CLUB,\n");
    print("AN EIGHTEEN HOLE CHAMPIONSHIP LAYOUT LOCATED A SHORT\n");
    print("DISTANCE FROM SCENIC DOWNTOWN MORRISTOWN.  THE\n");
    print("COMMENTATOR WILL EXPLAIN THE GAME AS YOU PLAY.\n");
    print("ENJOY YOUR GAME; SEE YOU AT THE 19TH HOLE...\n");
    print("\n");
    print("\n");
    next_hole = 0;
    g1 = 18;
    g2 = 0;
    g3 = 0;
    a = 0;
    n = 0.8;
    s2 = 0;
    f = 1;
    while (1) {
        print("WHAT IS YOUR HANDICAP");
        h = parseInt(await input());
        print("\n");
        if (h < 0 || h > 30) {
            print("PGA HANDICAPS RANGE FROM 0 TO 30.\n");
        } else {
            break;
        }
    }
    do {
        print("DIFFICULTIES AT GOLF INCLUDE:\n");
        print("0=HOOK, 1=SLICE, 2=POOR DISTANCE, 4=TRAP SHOTS, 5=PUTTING\n");
        print("WHICH ONE (ONLY ONE) IS YOUR WORST");
        t = parseInt(await input());
        print("\n");
    } while (t > 5) ;
    s1 = 0;
    first_routine = true;
    while (1) {
        if (first_routine) {
            la[0] = 0;
            j = 0;
            q = 0;
            s2++;
            k = 0;
            if (f != 1) {
                print("YOUR SCORE ON HOLE " + (f - 1) + " WAS " + s1 + "\n");
                show_score();
                if (g1 == f - 1)    // Completed all holes?
                    return;         // Exit game
                if (s1 > p + 2) {
                    print("KEEP YOUR HEAD DOWN.\n");
                } else if (s1 == p) {
                    print("A PAR.  NICE GOING.\n");
                } else if (s1 == p - 1) {
                    print("A BIRDIE.\n");
                } else if (s1 == p - 2) {
                    if (p != 3)
                        print("A GREAT BIG EAGLE.\n");
                    else
                        print("A HOLE IN ONE.\n");
                }
            }
            if (f == 19) {
                print("\n");
                show_score();
                if (g1 == f - 1)
                    return;
            }
            s1 = 0;
            print("\n");
            if (s1 != 0 && la[0] < 1)
                la[0] = 1;
        }
        if (s1 == 0) {
            d = hole_data[next_hole++];
            p = hole_data[next_hole++];
            la[1] = hole_data[next_hole++];
            la[2] = hole_data[next_hole++];
            print("\n");
            print("YOU ARE AT THE TEE OFF HOLE " + f + " DISTANCE " + d + " YARDS, PAR " + p + "\n");
            g3 += p;
            print("ON YOUR RIGHT IS ");
            x = 1;
            show_obstacle();
            print("ON YOUR LEFT IS ");
            x = 2
            show_obstacle();
        } else {
            x = 0;
            if (la[0] > 5) {
                if (la[0] > 6) {
                    print("YOUR SHOT WENT OUT OF BOUNDS.\n");
                } else {
                    print("YOUR SHOT WENT INTO THE WATER.\n");
                }
                s1++;
                print("PENALTY STROKE ASSESSED.  HIT FROM PREVIOUS LOCATION.\n");
                j++;
                la[0] = 1;
                d = b;
            } else {
                print("SHOT WENT " + d1 + " YARDS.  IT'S " + d2 + " YARDS FROM THE CUP.\n");
                print("BALL IS " + Math.floor(o) + " YARDS OFF LINE... IN ");
                show_obstacle();
            }
        }

        while (1) {
            if (a != 1) {
                print("SELECTION OF CLUBS\n");
                print("YARDAGE DESIRED                       SUGGESTED CLUBS\n");
                print("200 TO 280 YARDS                           1 TO 4\n");
                print("100 TO 200 YARDS                          19 TO 13\n");
                print("  0 TO 100 YARDS                          29 TO 23\n");
                a = 1;
            }
            print("WHAT CLUB DO YOU CHOOSE");
            c = parseInt(await input());
            print("\n");
            if (c >= 1 && c <= 29 && (c < 5 || c >= 12)) {
                if (c > 4)
                    c -= 6;
                if (la[0] <= 5 || c == 14 || c == 23) {
                    s1++;
                    w = 1;
                    if (c <= 13) {
                        if (f % 3 == 0 && s2 + q + (10 * (f - 1) / 18) < (f - 1) * (72 + ((h + 1) / 0.85)) / 18) {
                            q++;
                            if (s1 % 2 != 0 && d >= 95) {
                                print("BALL HIT TREE - BOUNCED INTO ROUGH " + (d - 75) + " YARDS FROM HOLE.\n");
                                d -= 75;
                                continue;
                            }
                            print("YOU DUBBED IT.\n");
                            d1 = 35;
                            second_routine = 1;
                            break;
                        } else if (c < 4 && la[0] == 2) {
                            print("YOU DUBBED IT.\n");
                            d1 = 35;
                            second_routine = 1;
                            break;
                        } else {
                            second_routine = 0;
                            break;
                        }
                    } else {
                        print("NOW GAUGE YOUR DISTANCE BY A PERCENTAGE (1 TO 100)\n");
                        print("OF A FULL SWING");
                        w = parseInt(await input());
                        w /= 100;
                        print("\n");
                        if (w <= 1) {
                            if (la[0] == 5) {
                                if (t == 3) {
                                    if (Math.random() <= n) {
                                        n *= 0.2;
                                        print("SHOT DUBBED, STILL IN TRAP.\n");
                                        continue;
                                    }
                                    n = 0.8;
                                }
                                d2 = 1 + (3 * Math.floor((80 / (40 - h)) * Math.random()));
                                second_routine = 2;
                                break;
                            }
                            if (c != 14)
                                c -= 10;
                            second_routine = 0;
                            break;
                        }
                        s1--;
                        // Fall through to THAT CLUB IS NOT IN THE BAG.
                    }
                }
            }
            print("THAT CLUB IS NOT IN THE BAG.\n");
            print("\n");
        }
        if (second_routine == 0) {
            if (s1 > 7 && d < 200) {
                d2 = 1 + (3 * Math.floor((80 / (40 - h)) * Math.random()));
                second_routine = 2;
            } else {
                d1 = Math.floor(((30 - h) * 2.5 + 187 - ((30 - h) * 0.25 + 15) * c / 2) + 25 * Math.random());
                d1 = Math.floor(d1 * w);
                if (t == 2)
                    d1 = Math.floor(d1 * 0.85);
            }
        }
        if (second_routine <= 1) {
            o = (Math.random() / 0.8) * (2 * h + 16) * Math.abs(Math.tan(d1 * 0.0035));
            d2 = Math.floor(Math.sqrt(Math.pow(o, 2) + Math.pow(Math.abs(d - d1), 2)));
            if (d - d1 < 0) {
                if (d2 >= 20)
                    print("TOO MUCH CLUB, YOU'RE PAST THE HOLE.\n");
            }
            b = d;
            d = d2;
            if (d2 > 27) {
                if (o < 30 || j > 0) {
                    la[0] = 1;
                } else {
                    if (t <= 0) {
                        s9 = (s2 + 1) / 15;
                        if (Math.floor(s9) == s9) {
                            print("YOU SLICED- ");
                            la[0] = la[1];
                        } else {
                            print("YOU HOOKED- ");
                            la[0] = la[2];
                        }
                    } else {
                        s9 = (s2 + 1) / 15;
                        if (Math.floor(s9) == s9) {
                            print("YOU HOOKED- ");
                            la[0] = la[2];
                        } else {
                            print("YOU SLICED- ");
                            la[0] = la[1];
                        }
                    }
                    if (o > 45)
                        print("BADLY.\n");
                }
                first_routine = false;
            } else if (d2 > 20) {
                la[0] = 5;
                first_routine = false;
            } else if (d2 > 0.5) {
                la[0] = 8;
                d2 = Math.floor(d2 * 3);
                second_routine = 2;
            } else {
                la[0] = 9;
                print("YOU HOLED IT.\n");
                print("\n");
                f++;
                first_routine = true;
            }
        }
        if (second_routine == 2) {
            while (1) {
                print("ON GREEN, " + d2 + " FEET FROM THE PIN.\n");
                print("CHOOSE YOUR PUTT POTENCY (1 TO 13):");
                i = parseInt(await input());
                s1++;
                if (s1 + 1 - p <= (h * 0.072) + 2 && k <= 2) {
                    k++;
                    if (t == 4)
                        d2 -= i * (4 + 1 * Math.random()) + 1;
                    else
                        d2 -= i * (4 + 2 * Math.random()) + 1.5;
                    if (d2 < -2) {
                        print("PASSED BY CUP.\n");
                        d2 = Math.floor(-d2);
                        continue;
                    }
                    if (d2 > 2) {
                        print("PUTT SHORT.\n");
                        d2 = Math.floor(d2);
                        continue;
                    }
                }
                print("YOU HOLED IT.\n");
                print("\n");
                f++;
                break;
            }
            first_routine = true;
        }
    }
}

```

这道题目缺少上下文，无法给出具体的解释。不过一般来说，在大多数编程语言中，`main()`函数是程序的入口点。在函数内，程序会执行一些初始化操作，然后开始执行用户输入的指令。对于这段代码而言，可能是为了确保所有用户输入的参数都被正确读取并保存到实参中。但需要根据具体的代码和背景来给出详细的解释。


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


# `39_Golf/python/golf.py`

这段代码是一个基于Python的程序，它模拟了一个高尔夫球场的环境。尽管这是一个文本游戏，但代码使用简单的几何原理来模拟球场的外观。

该程序模拟了一个40码宽的高尔夫球场，周围被5码的 rough 环境包围。绿色区域是一个10码的圆形，在杯子的位置(0,0)。

程序中包含以下主要步骤：

1. 创建一个包含高尔夫球场元素（如球场、四块草地、边缘的 rough）的画布。
2. 设置球场的大小（40码×10码）。
3. 创建一个圆形（绿色）对象，并将其位置设置为(0,0)。
4. 创建四个方形（代表四个球场），每个方形都是40码×10码的面积，并且周围有一层5码的 rough。
5. 将这四个方形设置为场景中的一个子对象。
6. 循环处理每一帧（每一帧代表一次游戏操作，例如移动球员）。
7. 检查球员是否在球场上。
8. 检查球员是否接近绿色。
9. 如果球员在球场上且靠近绿色，那么将球员位置设置为(球员位置， 0)。

总之，该代码使用简单的几何原理创建了一个高尔夫球场的环境，并模拟了玩家的移动和球场的状态。


```
'''
        8""""8 8"""88 8     8""""
        8    " 8    8 8     8
        8e     8    8 8e    8eeee
        88  ee 8    8 88    88
        88   8 8    8 88    88
        88eee8 8eeee8 88eee 88

GOLF


Despite being a text based game, the code uses simple geometry to simulate a course.
Fairways are 40 yard wide rectangles, surrounded by 5 yards of rough around the perimeter.
The green is a circle of 10 yards radius around the cup.
The cup is always at point (0,0).

```

这段代码是一个使用基本三角函数来绘制高尔夫球位置的程序。通过计算击球距离球场的远近以及球的偏差角度，来确定球的位置。这个程序基于真实世界高尔夫球场的平均值，并且受到很多随机因素，例如球场的难易程度、球员的技能水平等等。

在代码中，我们定义了三个变量：courseInfo、clubs 和 scoreCard，它们都包含一个空对象。我们可以从1开始对它们进行索引。就像所有好的程序员一样，我们也会从0开始计数，但在这个上下文中，将0作为变量索引的起点会更加自然，因为通常情况下高尔夫球的球洞号是从1开始的。

在函数内，我们创建了一个名为 " rough" 的空对象，并将其计数器设置为 1。我们也创建了名为 "businessRules" 和 "luck" 的空对象，并将它们的内容都设置为 "0"。接下来，我们在函数中执行了以下操作：

1. 创建一个名为 "pins" 的空对象，并将其计数器设置为调用者提供的 "courseInfo"、"clubs" 和 "scoreCard" 数组中的第一个对象的计数器值。
2. 通过调用 "nearest 500" 函数来获取球场的最近500码的球洞编号，并将其存储在 "pins.courseInfo.length" 变量中。
3. 通过调用 "findNearest green" 函数来获取球场内最近绿色球洞的球洞编号，并将其存储在 "pins.courseInfo.length" 变量中。
4. 通过调用 "angle 225" 函数来获取球场内最近绿色球洞的偏差角度，并将其存储在 "pins.courseInfo.length" 变量中。
5. 创建一个名为 " fairway" 的空对象，并将其计数器设置为调用者提供的 "courseInfo"、"clubs" 和 "scoreCard" 数组中的第二个对象的计数器值。
6. 通过调用 "nearest 350" 函数来获取球场最近350码的球洞编号，并将其存储在 "fairway.courseInfo.length" 变量中。
7. 通过调用 "angle 60" 函数来获取球场内最近绿色球洞的偏差角度，并将其存储在 "fairway.courseInfo.length" 变量中。
8. 创建一个名为 "green" 的空对象，并将其计数器设置为调用者提供的 "courseInfo"、"clubs" 和 "scoreCard" 数组中的第三个对象的计数器值。
9. 通过调用 "nearest 250" 函数来获取球场最近250码的球洞编号，并将其存储在 "green.courseInfo.length" 变量中。
10. 通过调用 "angle 30" 函数来获取球场内最近绿色球洞的偏差角度，并将其存储在 "green.courseInfo.length" 变量中。
11. 将 "挥杆距离"、"球的偏差角度" 和 "击球点" 变量设置为 0，以便在计算球的位置时进行初始化。
12. 最后，我们调用了一个名为 "plot球的位置" 的函数，并传入高尔夫球场的


```
Using basic trigonometry we can plot the ball's location using the distance of the stroke and
and the angle of deviation (hook/slice).

The stroke distances are based on real world averages of different club types.
Lots of randomization, "business rules", and luck influence the game play.
Probabilities are commented in the code.

note: 'courseInfo', 'clubs', & 'scoreCard' arrays each include an empty object so indexing
can begin at 1. Like all good programmers we count from zero, but in this context,
it's more natural when hole number one is at index one


    |-----------------------------|
    |            rough            |
    |   ----------------------    |
    |   |                     |   |
    | r |        =  =         | r |
    | o |     =        =      | o |
    | u |    =    .     =     | u |
    | g |    =   green  =     | g |
    | h |     =        =      | h |
    |   |        =  =         |   |
    |   |                     |   |
    |   |                     |   |
    |   |      Fairway        |   |
    |   |                     |   |
    |   |               ------    |
    |   |            --        -- |
    |   |           --  hazard  --|
    |   |            --        -- |
    |   |               ------    |
    |   |                     |   |
    |   |                     |   |   out
    |   |                     |   |   of
    |   |                     |   |   bounds
    |   |                     |   |
    |   |                     |   |
    |            tee              |


```

这段代码是一个高尔夫球场计算器，用于计算高尔夫球的位置。它根据球的击球点(x,y)距离球杆的距离(d)以及球的角度(theta)来计算高尔夫球的新位置。

首先，它定义了绿色(Typical green size: 20-30 yards)和高尔夫球场内(Typical golf course fairways are 35 to 45 yards wide)高尔夫球场内的典型宽度。

然后，它计算球的新的位置，基于球的击球点(x,y)，球杆的距离(d)和球的角度(theta)。对于右手持球者，球的切削角度(theta)为正值，表示球会朝右切。对于左手持球者，球的切削角度(theta)为负值，表示球会朝左切。

接着，它定义了球的初始位置(Inital position)为球的中心点(0,0)，即球的坐标为(0,0)。同时，它还定义了球的杯(Cup)始终指向(0,0)的意思。

最后，它使用了`atan2`函数来计算球的杯(Cup)和球(Golf ball)之间的角度。将球的杯的向量设置为(0,-1)相当于将球心从坐标(0,0)移动到坐标(0,-1)位置，即球心偏离了(0,-1)。


```
Typical green size: 20-30 yards
Typical golf course fairways are 35 to 45 yards wide
Our fairway extends 5 yards past green
Our rough is a 5 yard perimeter around fairway

We calculate the new position of the ball given the ball's point, the distance
of the stroke, and degrees off line (hook or slice).

Degrees off (for a right handed golfer):
Slice: positive degrees = ball goes right
Hook: negative degrees = left goes left

The cup is always at point: 0,0.
We use atan2 to compute the angle between the cup and the ball.
Setting the cup's vector to 0,-1 on a 360 circle is equivalent to:
```

这段代码计算了一个球在另一个杯子上的反向角度，并输出了一些角度和相应的球面时间(以度、分和秒为单位)。

代码首先定义了四个角度单位为o'clock的刻度，分别对应于12 o'clock、3 o'clock、6 o'clock和9 o'clock。然后，代码使用余弦定理计算出这些刻度与球之间的逆时针角度。

具体来说，代码使用余弦函数(即：cscenario(角度， hyp))计算每个角度的余弦值，然后将其转换为弧度。然后，代码使用球的位置(即球心角)和逆时针角度计算出球在另一个杯子上的位置。最后，代码输出了一些角度和相应的球面时间。


```
0 deg = 12 o'clock;  90 deg = 3 o'clock;  180 deg = 6 o'clock;  270 = 9 o'clock
The reverse angle between the cup and the ball is a difference of PI (using radians).

Given the angle and stroke distance (hypotenuse), we use cosine to compute
the opposite and adjacent sides of the triangle, which, is the ball's new position.

        0
        |
270 - cup - 90
        |
        180


        cup
        |
        |
        | opp
        |-----* new position
        |    /
        |   /
    adj  |  /
        | /  hyp
        |/
        tee

```

This code appears to be a script for a game or algorithm that involves golf stroke tracking. The script uses a technique called "bitwise masking" to describe the results of a large number of combinations.

Here's a breakdown of how the code works:

1. The formula for the bitwise masking algorithm is given: "01110001" (OR 5 bits) OR "10101011" (AND 5 bits) OR "11110000" (XOR 5 bits).
2. The code then creates a function called "hook" that takes a slice (a group of 5 bits) as input and returns a modified slice that includes all the bits from the hook bitwise mask.
3. The code defines a function called "slice" that takes a whole number (a 32-bit integer) and returns a slice that includes all the bits from the start of the integer to the end of the bitwise mask.
4. Finally, the code calls the "hook" function with the "slice" function, and prints the result (a modified slice).

The purpose of this code is unclear without more context, but it appears to be a tool for tracking golf strokes or球的位置， using bitwise masking.


```
<- hook    slice ->


Given the large number of combinations needed to describe a particular stroke / ball location,
we use the technique of "bitwise masking" to describe stroke results.
With bit masking, multiple flags (bits) are combined into a single binary number that can be
tested by applying a mask. A mask is another binary number that isolates a particular bit that
you are interested in. You can then apply your language's bitwise opeartors to test or
set a flag.

Game design by Jason Bonthron, 2021
www.bonthron.com
for my father, Raymond Bonthron, an avid golfer

Inspired by the 1978 "Golf" from "Basic Computer Games"
```

这段代码是一个名为“GolfGame”的类，它是一个模仿高尔夫球游戏的电子游戏。这个游戏可能是由一个匿名作者创作的，但是它的原作者已经被知情人修改过。现在，这个游戏已经被移植到了Python语言中，由一位名为马丁·托马斯的程序员进行了修改。

在这个游戏中，玩家需要点击按钮来打高尔夫球，每个按钮都会让高尔夫球飞行的距离产生一个随机的变化。游戏还会随机生成一些鸟儿，它们可能会分散在球场的不同位置，并且每个鸟儿的位置都会随着时间的推移产生变化。高尔夫球和鸟儿的运动轨迹都是受到物理效果影响的，包括重力、风力和碰撞检测等。

总的来说，这段代码的主要目的是提供一个有趣和具有挑战性的高尔夫球游戏，让玩家可以享受游戏带来的乐趣。


```
by Steve North, who modified an existing golf game by an unknown author

Ported in 2022 to Python by Martin Thoma
'''


import enum
import math
import random
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, NamedTuple, Tuple


```

这段代码定义了一个名为 `clear_console` 的函数和一个名为 `Point` 的类，以及一个名为 `GameObjType` 的枚举类型。

函数 `clear_console` 的作用是清除控制台并返回 `None`，具体实现是通过 `print` 函数来实现，并使用 `end` 参数来指定输出结束的位置。这里使用了 `033[H]` 和 `033[J]`，它们是 `PS1` 控制台显示模式，分别显示 `H` 和 `J` 两个字符，相当于 `↑` 和 `↓` 方向键。将它们并在一起，就清空了整个控制台并刷新了屏幕。

类 `Point` 是一个命名元组类型，包含两个整型变量 `x` 和 `y`，用于表示平面直角坐标系中的点。

枚举类型 `GameObjType` 定义了 9 个枚举值，分别为 `BALL`、`CUP`、`GREEN`、`FAIRWAY`、`ROUGH`、`TREES`、`WATER` 和 `SAND`，用于表示不同类型的地形。


```
def clear_console() -> None:
    print("\033[H\033[J", end="")


class Point(NamedTuple):
    x: int
    y: int


class GameObjType(enum.Enum):
    BALL = enum.auto()
    CUP = enum.auto()
    GREEN = enum.auto()
    FAIRWAY = enum.auto()
    ROUGH = enum.auto()
    TREES = enum.auto()
    WATER = enum.auto()
    SAND = enum.auto()


```

这段代码定义了两个类，一个是CircleGameObj，另一个是RectGameObj。这两个类都属于NamedTuple类型，表示它们拥有一个命名成员tuple。

CircleGameObj类包含以下成员变量：
- CircleGameObj(NamedTuple): 定义了类中心点的x和y坐标，以及半径，同时继承了GameObjType类。

RectGameObj类包含以下成员变量：
- RectGameObj(NamedTuple): 定义了类上下左右两边的边界，以及宽度，同时继承了GameObjType类。

两个类都是从GameObjType类继承而来的，因此它们都实现了GameObj类。


```
class CircleGameObj(NamedTuple):
    # center point
    X: int
    Y: int
    Radius: int
    Type: GameObjType


class RectGameObj(NamedTuple):
    # Upper left corner
    X: int
    Y: int
    Width: int
    Length: int
    Type: GameObjType


```

这段代码定义了两个类，一个是`HoleInfo`类，另一个是`HoleGeometry`类。

`HoleInfo`类表示一个圆孔信息，包括圆孔的ID、所在的球场、圆孔的草坪长度和角度、以及圆孔周围的危险障碍物。

`HoleGeometry`类表示一个圆孔的几何信息，包括圆孔的草坪、球杆、障碍物，以及球场的具体位置和朝向。

这两个类都继承自`CircleGameObj`类，表示它们都属于圆孔类物体。


```
Ball = CircleGameObj
Hazard = CircleGameObj


class HoleInfo(NamedTuple):
    hole: int
    yards: int
    par: int
    hazards: List[Hazard]
    description: str


class HoleGeometry(NamedTuple):
    cup: CircleGameObj
    green: CircleGameObj
    fairway: RectGameObj
    rough: RectGameObj
    hazards: List[Hazard]


```

这段代码定义了一个名为 Plot 的类，它有四个成员变量：x、y、offline，分别表示该点在水平方向、垂直方向和侧面的距离，以及该点是否在定义了一个圆柱游戏对象(CircleGameObj)和一个矩形游戏对象(RectGameObj)的矩形内部。

此外，还定义了一个名为 get_distance 的函数，用于计算两个点之间的距离，该函数接收两个点坐标并返回一个浮点数。

以及一个名为 is_in_rectangle 的函数，用于判断一个点是否在矩形内部，该函数接收一个圆柱游戏对象和一个矩形游戏对象，并返回一个布尔值。


```
@dataclass
class Plot:
    x: int
    y: int
    offline: int


def get_distance(pt1: Point, pt2: Point) -> float:
    """distance between 2 points"""
    return math.sqrt(math.pow((pt2.x - pt1.x), 2) + math.pow((pt2.y - pt1.y), 2))


def is_in_rectangle(pt: CircleGameObj, rect: RectGameObj) -> bool:
    # only true if its completely inside
    return (
        (pt.X > rect.X)
        and (pt.X < rect.X + rect.Width)
        and (pt.Y > rect.Y)
        and (pt.Y < rect.Y + rect.Length)
    )


```



这三段代码分别是：

1. `to_radians(angle: float) -> float` 是一个函数，接受一个角度参数 `angle`，返回这个角度的弧度值。它的实现是通过乘以 `math.pi` / `180.0` 来将角度转换为弧度。

2. `to_degrees_360(angle: float) -> float` 是一个函数，接受一个角度参数 `angle`，返回这个角度的度数。它的实现是通过将弧度数转换为度数，然后再将度数转换为弧度。

3. `odds(x: int) -> bool` 是一个函数，接受一个整数参数 `x`，返回一个布尔值。它的实现是使用 `random.randint(1, 101)` 来生成一个随机整数，然后判断这个整数是否小于或等于 `x`。


```
def to_radians(angle: float) -> float:
    return angle * (math.pi / 180.0)


def to_degrees_360(angle: float) -> float:
    """radians to 360 degrees"""
    deg = angle * (180.0 / math.pi)
    if deg < 0.0:
        deg += 360.0
    return deg


def odds(x: int) -> bool:
    # chance an integer is <= the given argument
    # between 1-100
    return random.randint(1, 101) <= x


```

This appears to be a list of `HoleInfo` objects, each of which describes a location with one or more hazards. Each `HoleInfo` object has a `latitude` and a `longitude` value, as well as a `description` and a `gameObjectType` value. The `gameObjectType` value indicates the type of hazard located at the location, such as `TREES` for a tree hazard, `WATER` for water hazards, or `SAND` for sand hazards. Additionally, some `HoleInfo` objects have additional hazards, such as `Hazard(-20, 200, 10, GameObjType.TREES)` or `Hazard(14, 12, 8, GameObjType.SAND)`.


```
# THE COURSE
CourseInfo = [
    HoleInfo(0, 0, 0, [], ""),  # include a blank so index 1 == hole 1
    # -------------------------------------------------------- front 9
    HoleInfo(
        1,
        361,
        4,
        [
            Hazard(20, 100, 10, GameObjType.TREES),
            Hazard(-20, 80, 10, GameObjType.TREES),
            Hazard(-20, 100, 10, GameObjType.TREES),
        ],
        "There are a couple of trees on the left and right.",
    ),
    HoleInfo(
        2,
        389,
        4,
        [Hazard(0, 160, 20, GameObjType.WATER)],
        "There is a large water hazard across the fairway about 150 yards.",
    ),
    HoleInfo(
        3,
        206,
        3,
        [
            Hazard(20, 20, 5, GameObjType.WATER),
            Hazard(-20, 160, 10, GameObjType.WATER),
            Hazard(10, 12, 5, GameObjType.SAND),
        ],
        "There is some sand and water near the green.",
    ),
    HoleInfo(
        4,
        500,
        5,
        [Hazard(-14, 12, 12, GameObjType.SAND)],
        "There's a bunker to the left of the green.",
    ),
    HoleInfo(
        5,
        408,
        4,
        [
            Hazard(20, 120, 20, GameObjType.TREES),
            Hazard(20, 160, 20, GameObjType.TREES),
            Hazard(10, 20, 5, GameObjType.SAND),
        ],
        "There are some trees to your right.",
    ),
    HoleInfo(
        6,
        359,
        4,
        [Hazard(14, 0, 4, GameObjType.SAND), Hazard(-14, 0, 4, GameObjType.SAND)],
        "",
    ),
    HoleInfo(
        7,
        424,
        5,
        [
            Hazard(20, 200, 10, GameObjType.SAND),
            Hazard(10, 180, 10, GameObjType.SAND),
            Hazard(20, 160, 10, GameObjType.SAND),
        ],
        "There are several sand traps along your right.",
    ),
    HoleInfo(8, 388, 4, [Hazard(-20, 340, 10, GameObjType.TREES)], ""),
    HoleInfo(
        9,
        196,
        3,
        [Hazard(-30, 180, 20, GameObjType.TREES), Hazard(14, -8, 5, GameObjType.SAND)],
        "",
    ),
    # -------------------------------------------------------- back 9
    HoleInfo(
        hole=10,
        yards=400,
        par=4,
        hazards=[
            Hazard(-14, -8, 5, GameObjType.SAND),
            Hazard(14, -8, 5, GameObjType.SAND),
        ],
        description="",
    ),
    HoleInfo(
        11,
        560,
        5,
        [
            Hazard(-20, 400, 10, GameObjType.TREES),
            Hazard(-10, 380, 10, GameObjType.TREES),
            Hazard(-20, 260, 10, GameObjType.TREES),
            Hazard(-20, 200, 10, GameObjType.TREES),
            Hazard(-10, 180, 10, GameObjType.TREES),
            Hazard(-20, 160, 10, GameObjType.TREES),
        ],
        "Lots of trees along the left of the fairway.",
    ),
    HoleInfo(
        12,
        132,
        3,
        [
            Hazard(-10, 120, 10, GameObjType.WATER),
            Hazard(-5, 100, 10, GameObjType.SAND),
        ],
        "There is water and sand directly in front of you. A good drive should clear both.",
    ),
    HoleInfo(
        13,
        357,
        4,
        [
            Hazard(-20, 200, 10, GameObjType.TREES),
            Hazard(-10, 180, 10, GameObjType.TREES),
            Hazard(-20, 160, 10, GameObjType.TREES),
            Hazard(14, 12, 8, GameObjType.SAND),
        ],
        "",
    ),
    HoleInfo(14, 294, 4, [Hazard(0, 20, 10, GameObjType.SAND)], ""),
    HoleInfo(
        15,
        475,
        5,
        [Hazard(-20, 20, 10, GameObjType.WATER), Hazard(10, 20, 10, GameObjType.SAND)],
        "Some sand and water near the green.",
    ),
    HoleInfo(16, 375, 4, [Hazard(-14, -8, 5, GameObjType.SAND)], ""),
    HoleInfo(
        17,
        180,
        3,
        [
            Hazard(20, 100, 10, GameObjType.TREES),
            Hazard(-20, 80, 10, GameObjType.TREES),
        ],
        "",
    ),
    HoleInfo(
        18,
        550,
        5,
        [Hazard(20, 30, 15, GameObjType.WATER)],
        "There is a water hazard near the green.",
    ),
]


```

这段代码使用了 bitwise flags，也就是二进制位运算中的标志位。这些 flags 用于表示各种状态，例如 dub 表示 tempting，hook 表示 hearing，slice_ 表示 going，passed_cup 表示 hit，in_cup 表示 in，on_fairway 表示 on，on_green 表示 on，in_rough 表示 in，in_trees 表示 in，in_water 表示 in，out_of_bounds 表示 out_of_bounds，luck 表示 random，ace 表示 attack。

这段代码的具体作用是用于 bitwise 算法的状态判断和计数。其中，dub 表示所有 flag 中最左边的 flag，hook 表示当前已经听到但还没有计算过的 flag，slice_ 表示当前正在进行的 slice 操作，passed_cup 表示已经 hit过的 flag，in_cup 表示当前正在输入的 flag，on_fairway 表示当前正在计算的 flag，on_green 表示当前正在播放的 flag，in_rough 表示当前正在输入的 rough 信息，in_trees 表示当前正在使用的 trees 信息，in_water 表示当前正在使用的 water 信息，out_of_bounds 表示当前已经超出范围，luck 表示随机的 flag，ace 表示当前已经计算过的 flag。


```
# -------------------------------------------------------- bitwise Flags
dub = 0b00000000000001
hook = 0b00000000000010
slice_ = 0b00000000000100
passed_cup = 0b00000000001000
in_cup = 0b00000000010000
on_fairway = 0b00000000100000
on_green = 0b00000001000000
in_rough = 0b00000010000000
in_sand = 0b00000100000000
in_trees = 0b00001000000000
in_water = 0b00010000000000
out_of_bounds = 0b00100000000000
luck = 0b01000000000000
ace = 0b10000000000000


```

This is a implementation of a game where the player can choose from a list of options, and the app will provide the player with information about the different options, as well as the outcome of their choice. The game has different modes, such as "quit", "bag", and "game over", which are described below:

* `quit` mode: The player is asked to leave the game.
* `bag` mode: The player is asked to评价他们的球袋， this will affect their score and the score will be added to the game over score.
* `game over` mode: The player is informed of their game score, and then given the option to visit the pro shop.

The game also has a `game_over` mode which is not used in the game but it could be added to track the score.


```
class Golf:
    ball: Ball
    hole_num: int = 0
    stroke_num: int = 0
    handicap: int = 0
    player_difficulty: int = 0
    hole_geometry: HoleGeometry

    # all fairways are 40 yards wide, extend 5 yards beyond the cup, and
    # have 5 yards of rough around the perimeter
    fairway_width: int = 40
    fairway_extension: int = 5
    rough_amt: int = 5

    # ScoreCard records the ball position after each stroke
    # a new list for each hole
    # include a blank list so index 1 == hole 1
    score_card: List[List[Ball]] = [[]]

    # YOUR BAG
    clubs: List[Tuple[str, int]] = [
        ("", 0),
        # name, average yardage
        ("Driver", 250),
        ("3 Wood", 225),
        ("5 Wood", 200),
        ("Hybrid", 190),
        ("4 Iron", 170),
        ("7 Iron", 150),
        ("9 Iron", 125),
        ("Pitching wedge", 110),
        ("Sand wedge", 75),
        ("Putter", 10),
    ]

    def __init__(self) -> None:
        print(" ")
        print('          8""""8 8"""88 8     8"""" ')
        print('          8    " 8    8 8     8     ')
        print("          8e     8    8 8e    8eeee ")
        print("          88  ee 8    8 88    88    ")
        print("          88   8 8    8 88    88    ")
        print("          88eee8 8eeee8 88eee 88    ")
        print(" ")
        print("Welcome to the Creative Computing Country Club,")
        print("an eighteen hole championship layout located a short")
        print("distance from scenic downtown Lambertville, New Jersey.")
        print("The game will be explained as you play.")
        print("Enjoy your game! See you at the 19th hole...")
        print(" ")
        print("Type QUIT at any time to leave the game.")
        print("Type BAG at any time to review the clubs in your bag.")
        print(" ")

        input("Press any key to continue.")
        clear_console()
        self.start_game()

    def start_game(self) -> None:
        print(" ")
        print("              YOUR BAG")
        self.review_bag()
        print("Type BAG at any time to review the clubs in your bag.")
        print(" ")

        input("Press any key to continue.")
        clear_console()
        self.ask_handicap()

    def ask_handicap(self) -> None:
        print(" ")

        self.ask(
            "PGA handicaps range from 0 to 30.\nWhat is your handicap?",
            0,
            30,
            self.set_handicap_ask_difficulty,
        )

    def set_handicap_ask_difficulty(self, i: int) -> None:
        self.handicap = i
        print(" ")

        self.ask(
            (
                "Common difficulties at golf include:\n"
                "1=Hook, 2=Slice, 3=Poor Distance, 4=Trap Shots, 5=Putting\n"
                "Which one is your worst?"
            ),
            1,
            5,
            self.set_difficulty_and_hole,
        )

    def set_difficulty_and_hole(self, j: int) -> None:
        self.player_difficulty = j
        clear_console()
        self.new_hole()

    def new_hole(self) -> None:
        self.hole_num += 1
        self.stroke_num = 0

        info: HoleInfo = CourseInfo[self.hole_num]

        yards: int = info.yards
        # from tee to cup
        cup = CircleGameObj(0, 0, 0, GameObjType.CUP)
        green = CircleGameObj(0, 0, 10, GameObjType.GREEN)

        fairway = RectGameObj(
            0 - int(self.fairway_width / 2),
            0 - (green.Radius + self.fairway_extension),
            self.fairway_width,
            yards + (green.Radius + self.fairway_extension) + 1,
            GameObjType.FAIRWAY,
        )

        rough = RectGameObj(
            fairway.X - self.rough_amt,
            fairway.Y - self.rough_amt,
            fairway.Width + (2 * self.rough_amt),
            fairway.Length + (2 * self.rough_amt),
            GameObjType.ROUGH,
        )

        self.ball = Ball(0, yards, 0, GameObjType.BALL)

        self.score_card_start_new_hole()

        self.hole_geometry = HoleGeometry(cup, green, fairway, rough, info.hazards)

        print(f"                |> {self.hole_num}")
        print("                |        ")
        print("                |        ")
        print("          ^^^^^^^^^^^^^^^")

        print(
            f"Hole #{self.hole_num}. You are at the tee. Distance {info.yards} yards, par {info.par}."
        )
        print(info.description)

        self.tee_up()

    def set_putter_and_stroke(self, strength: float) -> None:
        putter = self.clubs[self.putt]
        self.stroke((putter[1] * (strength / 10.0)), self.putt)

    def ask_gauge(self, c: int) -> None:
        self.club = self.clubs[c]

        print(" ")
        print(f"[{self.club[0].upper()}: average {self.club[1]} yards]")

        foo = partial(self.make_stroke, c=c)

        self.ask(
            "Now gauge your distance by a percentage of a full swing. (1-10)",
            1,
            10,
            foo,
        )

    def make_stroke(self, strength: float, c: int) -> None:
        self.stroke((self.club[1] * (strength / 10.0)), c)

    def tee_up(self) -> None:
        # on the green? automatically select putter
        # otherwise Ask club and swing strength
        if self.is_on_green(self.ball) and not self.is_in_hazard(
            self.ball, GameObjType.SAND
        ):
            self.putt = 10
            print("[PUTTER: average 10 yards]")
            if odds(20):
                msg = "Keep your head down.\n"
            else:
                msg = ""

            self.ask(
                msg + "Choose your putt potency. (1-10)",
                1,
                10,
                self.set_putter_and_stroke,
            )
        else:
            self.ask("What club do you choose? (1-10)", 1, 10, self.ask_gauge)

    def stroke(self, club_amt: float, club_index: int) -> None:
        self.stroke_num += 1

        flags = 0b000000000000

        # fore! only when driving
        if (self.stroke_num == 1) and (club_amt > 210) and odds(30):
            print('"...Fore !"')

        # dub
        if odds(5):
            # there's always a 5% chance of dubbing it
            flags |= dub

        # if you're in the rough, or sand, you really should be using a wedge
        if (
            (
                self.is_in_rough(self.ball)
                or self.is_in_hazard(self.ball, GameObjType.SAND)
            )
            and not (club_index == 8 or club_index == 9)
            and odds(40)
        ):
            flags |= dub

        # trap difficulty
        if (
            self.is_in_hazard(self.ball, GameObjType.SAND)
            and self.player_difficulty == 4
        ) and odds(20):
            flags |= dub

        # hook/slice
        # There's 10% chance of a hook or slice
        # if it's a known player_difficulty then increase chance to 30%
        # if it's a putt & putting is a player_difficulty increase to 30%

        rand_hook_slice: bool
        if (
            self.player_difficulty == 1
            or self.player_difficulty == 2
            or (self.player_difficulty == 5 and self.is_on_green(self.ball))
        ):
            rand_hook_slice = odds(30)
        else:
            rand_hook_slice = odds(10)

        if rand_hook_slice:
            if self.player_difficulty == 1:
                if odds(80):
                    flags |= hook
                else:
                    flags |= slice_
            elif self.player_difficulty == 2:
                if odds(80):
                    flags |= slice_
                else:
                    flags |= hook
            else:
                if odds(50):
                    flags |= hook
                else:
                    flags |= slice_

        # beginner's luck !
        # If handicap is greater than 15, there's a 10% chance of avoiding all errors
        if (self.handicap > 15) and (odds(10)):
            flags |= luck

        # ace
        # there's a 10% chance of an Ace on a par 3
        if CourseInfo[self.hole_num].par == 3 and odds(10) and self.stroke_num == 1:
            flags |= ace

        # distance:
        # If handicap is < 15, there a 50% chance of reaching club average,
        # a 25% of exceeding it, and a 25% of falling short
        # If handicap is > 15, there's a 25% chance of reaching club average,
        # and 75% chance of falling short
        # The greater the handicap, the more the ball falls short
        # If poor distance is a known player_difficulty, then reduce distance by 10%

        distance: float
        rnd = random.randint(1, 101)

        if self.handicap < 15:
            if rnd <= 25:
                distance = club_amt - (club_amt * (self.handicap / 100.0))
            elif rnd > 25 and rnd <= 75:
                distance = club_amt
            else:
                distance = club_amt + (club_amt * 0.10)
        else:
            if rnd <= 75:
                distance = club_amt - (club_amt * (self.handicap / 100.0))
            else:
                distance = club_amt

        if self.player_difficulty == 3 and odds(80):  # poor distance
            distance = distance * 0.80

        if (flags & luck) == luck:
            distance = club_amt

        # angle
        # For all strokes, there's a possible "drift" of 4 degrees
        # a hooks or slice increases the angle between 5-10 degrees,
        # hook uses negative degrees
        angle = random.randint(0, 5)
        if (flags & slice_) == slice_:
            angle = random.randint(5, 11)
        if (flags & hook) == hook:
            angle = 0 - random.randint(5, 11)
        if (flags & luck) == luck:
            angle = 0

        plot = self.plot_ball(self.ball, distance, angle)
        # calculate a new location
        if (flags & luck) == luck and plot.y > 0:
            plot.y = 2

        flags = self.find_ball(
            Ball(plot.x, plot.y, plot.offline, GameObjType.BALL), flags
        )

        self.interpret_results(plot, flags)

    def plot_ball(self, ball: Ball, stroke_distance: float, degrees_off: float) -> Plot:
        cup_vector = Point(0, -1)
        rad_from_cup = math.atan2(ball.Y, ball.X) - math.atan2(
            cup_vector.y, cup_vector.x
        )
        rad_from_ball = rad_from_cup - math.pi

        hypotenuse = stroke_distance
        adjacent = math.cos(rad_from_ball + to_radians(degrees_off)) * hypotenuse
        opposite = math.sqrt(math.pow(hypotenuse, 2) - math.pow(adjacent, 2))

        new_pos: Point
        if to_degrees_360(rad_from_ball + to_radians(degrees_off)) > 180:
            new_pos = Point(int(ball.X - opposite), int(ball.Y - adjacent))
        else:
            new_pos = Point(int(ball.X + opposite), int(ball.Y - adjacent))

        return Plot(new_pos.x, new_pos.y, int(opposite))

    def interpret_results(self, plot: Plot, flags: int) -> None:
        cup_distance: int = int(
            get_distance(
                Point(plot.x, plot.y),
                Point(self.hole_geometry.cup.X, self.hole_geometry.cup.Y),
            )
        )
        travel_distance: int = int(
            get_distance(Point(plot.x, plot.y), Point(self.ball.X, self.ball.Y))
        )

        print(" ")

        if (flags & ace) == ace:
            print("Hole in One! You aced it.")
            self.score_card_record_stroke(Ball(0, 0, 0, GameObjType.BALL))
            self.report_current_score()
            return

        if (flags & in_trees) == in_trees:
            print("Your ball is lost in the trees. Take a penalty stroke.")
            self.score_card_record_stroke(self.ball)
            self.tee_up()
            return

        if (flags & in_water) == in_water:
            if odds(50):
                msg = "Your ball has gone to a watery grave."
            else:
                msg = "Your ball is lost in the water."
            print(msg + " Take a penalty stroke.")
            self.score_card_record_stroke(self.ball)
            self.tee_up()
            return

        if (flags & out_of_bounds) == out_of_bounds:
            print("Out of bounds. Take a penalty stroke.")
            self.score_card_record_stroke(self.ball)
            self.tee_up()
            return

        if (flags & dub) == dub:
            print("You dubbed it.")
            self.score_card_record_stroke(self.ball)
            self.tee_up()
            return

        if (flags & in_cup) == in_cup:
            if odds(50):
                msg = "You holed it."
            else:
                msg = "It's in!"
            print(msg)
            self.score_card_record_stroke(Ball(plot.x, plot.y, 0, GameObjType.BALL))
            self.report_current_score()
            return

        if ((flags & slice_) == slice_) and not ((flags & on_green) == on_green):
            if (flags & out_of_bounds) == out_of_bounds:
                bad = "badly"
            else:
                bad = ""
            print(f"You sliced{bad}: {plot.offline} yards offline.")

        if ((flags & hook) == hook) and not ((flags & on_green) == on_green):
            if (flags & out_of_bounds) == out_of_bounds:
                bad = "badly"
            else:
                bad = ""
            print(f"You hooked{bad}: {plot.offline} yards offline.")

        if self.stroke_num > 1:
            prev_ball = self.score_card_get_previous_stroke()
            d1 = get_distance(
                Point(prev_ball.X, prev_ball.Y),
                Point(self.hole_geometry.cup.X, self.hole_geometry.cup.Y),
            )
            d2 = cup_distance
            if d2 > d1:
                print("Too much club.")

        if (flags & in_rough) == in_rough:
            print("You're in the rough.")

        if (flags & in_sand) == in_sand:
            print("You're in a sand trap.")

        if (flags & on_green) == on_green:
            if cup_distance < 4:
                pd = str(cup_distance * 3) + " feet"
            else:
                pd = f"{cup_distance} yards"
            print(f"You're on the green. It's {pd} from the pin.")

        if ((flags & on_fairway) == on_fairway) or ((flags & in_rough) == in_rough):
            print(
                f"Shot went {travel_distance} yards. "
                f"It's {cup_distance} yards from the cup."
            )

        self.score_card_record_stroke(Ball(plot.x, plot.y, 0, GameObjType.BALL))

        self.ball = Ball(plot.x, plot.y, 0, GameObjType.BALL)

        self.tee_up()

    def report_current_score(self) -> None:
        par = CourseInfo[self.hole_num].par
        if len(self.score_card[self.hole_num]) == par + 1:
            print("A bogey. One above par.")
        if len(self.score_card[self.hole_num]) == par:
            print("Par. Nice.")
        if len(self.score_card[self.hole_num]) == (par - 1):
            print("A birdie! One below par.")
        if len(self.score_card[self.hole_num]) == (par - 2):
            print("An Eagle! Two below par.")
        if len(self.score_card[self.hole_num]) == (par - 3):
            print("Double Eagle! Unbelievable.")

        total_par: int = 0
        for i in range(1, self.hole_num + 1):
            total_par += CourseInfo[i].par

        print(" ")
        print("-----------------------------------------------------")
        if self.hole_num > 1:
            hole_str = "holes"
        else:
            hole_str = "hole"
        print(
            f" Total par for {self.hole_num} {hole_str} is: {total_par}. "
            f"Your total is: {self.score_card_get_total()}."
        )
        print("-----------------------------------------------------")
        print(" ")

        if self.hole_num == 18:
            self.game_over()
        else:
            time.sleep(2)
            self.new_hole()

    def find_ball(self, ball: Ball, flags: int) -> int:
        if self.is_on_fairway(ball) and not self.is_on_green(ball):
            flags |= on_fairway
        if self.is_on_green(ball):
            flags |= on_green
        if self.is_in_rough(ball):
            flags |= in_rough
        if self.is_out_of_bounds(ball):
            flags |= out_of_bounds
        if self.is_in_hazard(ball, GameObjType.WATER):
            flags |= in_water
        if self.is_in_hazard(ball, GameObjType.TREES):
            flags |= in_trees
        if self.is_in_hazard(ball, GameObjType.SAND):
            flags |= in_sand

        if ball.Y < 0:
            flags |= passed_cup

        # less than 2, it's in the cup
        d = get_distance(
            Point(ball.X, ball.Y),
            Point(self.hole_geometry.cup.X, self.hole_geometry.cup.Y),
        )
        if d < 2:
            flags |= in_cup

        return flags

    def is_on_fairway(self, ball: Ball) -> bool:
        return is_in_rectangle(ball, self.hole_geometry.fairway)

    def is_on_green(self, ball: Ball) -> bool:
        d = get_distance(
            Point(ball.X, ball.Y),
            Point(self.hole_geometry.cup.X, self.hole_geometry.cup.Y),
        )
        return d < self.hole_geometry.green.Radius

    def hazard_hit(self, h: Hazard, ball: Ball, hazard: GameObjType) -> bool:
        d = get_distance(Point(ball.X, ball.Y), Point(h.X, h.Y))
        result = False
        if (d < h.Radius) and h.Type == hazard:
            result = True
        return result

    def is_in_hazard(self, ball: Ball, hazard: GameObjType) -> bool:
        result: bool = False
        for h in self.hole_geometry.hazards:
            result = result and self.hazard_hit(h, ball, hazard)
        return result

    def is_in_rough(self, ball: Ball) -> bool:
        return is_in_rectangle(ball, self.hole_geometry.rough) and (
            not is_in_rectangle(ball, self.hole_geometry.fairway)
        )

    def is_out_of_bounds(self, ball: Ball) -> bool:
        return (not self.is_on_fairway(ball)) and (not self.is_in_rough(ball))

    def score_card_start_new_hole(self) -> None:
        self.score_card.append([])

    def score_card_record_stroke(self, ball: Ball) -> None:
        clone = Ball(ball.X, ball.Y, 0, GameObjType.BALL)
        self.score_card[self.hole_num].append(clone)

    def score_card_get_previous_stroke(self) -> Ball:
        return self.score_card[self.hole_num][len(self.score_card[self.hole_num]) - 1]

    def score_card_get_total(self) -> int:
        total: int = 0
        for h in self.score_card:
            total += len(h)
        return total

    def ask(
        self, question: str, min_: int, max_: int, callback: Callable[[int], Any]
    ) -> None:
        # input from console is always an integer passed to a callback
        # or "quit" to end game
        print(question)
        i = input().strip().lower()
        if i == "quit":
            self.quit_game()
            return
        if i == "bag":
            self.review_bag()

        try:
            n = int(i)
            success = True
        except Exception:
            success = False
            n = 0

        if success:
            if n >= min_ and n <= max_:
                callback(n)
            else:
                self.ask(question, min_, max_, callback)
        else:
            self.ask(question, min_, max_, callback)

    def review_bag(self) -> None:
        print(" ")
        print("  #     Club      Average Yardage")
        print("-----------------------------------")
        print("  1    Driver           250")
        print("  2    3 Wood           225")
        print("  3    5 Wood           200")
        print("  4    Hybrid           190")
        print("  5    4 Iron           170")
        print("  6    7 Iron           150")
        print("  7    9 Iron           125")
        print("  8    Pitching wedge   110")
        print("  9    Sand wedge        75")
        print(" 10    Putter            10")
        print(" ")

    def quit_game(self) -> None:
        print("\nLooks like rain. Goodbye!\n")
        return

    def game_over(self) -> None:
        net = self.score_card_get_total() - self.handicap
        print("Good game!")
        print(f"Your net score is: {net}")
        print("Let's visit the pro shop...")
        print(" ")
        return


```

这段代码是一个 if 语句，它的作用是在程序运行时（而非编译时）检查是否执行了程序的main函数。如果程序的执行环境（通常是浏览器或命令行）中已经定义了__main__函数，则程序将直接执行if语句中的语句，否则程序将无法执行该语句。

简单地说，这段代码确保在程序运行时会执行程序的main函数，即使程序是在运行时而不是编译时定义的。


```
if __name__ == "__main__":
    Golf()

```