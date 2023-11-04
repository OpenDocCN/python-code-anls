# BasicComputerGames源码解析 15

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `07_Basketball/javascript/basketball.js`

这段代码定义了两个函数，分别是`print`函数和`input`函数。

`print`函数的作用是在网页上打印一段字符串，将字符串作为参数传递给`document.getElementById("output").appendChild(document.createTextNode(str))`，这里使用了JavaScript中的`document.getElementById`获取到了网页上的一个元素，并将其设置为文本，并将其添加到了文档中的某个位置。

`input`函数的作用是从用户那里获取输入的值，将用户的输入存储在`input_str`变量中，这里使用了JavaScript中的`document.getElementById("input")`获取到了网页上某个元素的引用，并将其设置为用户输入的值。这里还添加了一个事件监听器，当用户按下键盘上的13号键时，会将用户输入的值存储到`input_str`变量中，并将其打印到页面上。


```
// BASKETBALL
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

这段代码定义了一个名为 "tab" 的函数，它会将一个字符串变量 "str" 中的所有空格都删除，并返回该字符串。

在代码中，首先创建了一个名为 "s" 的二维数组，用于存储游戏中的玩家步数。接着定义了一个名为 "z" 的变量，用于存储游戏中的计数器，用于记录玩家已经完成了多少步。定义了一个名为 "d" 的变量，用于存储游戏中的障碍物数量。定义了一个名为 "p" 的变量，用于存储玩家当前所处的位置。定义了一个名为 "your_turn" 的变量，用于标记当前是否为玩家操作时间，即玩家可以移动或跳跃的时间。定义了一个名为 "game_restart" 的变量，用于存储是否可以重新开始游戏或是否已经开始了游戏。

在函数 "tab" 中，首先创建了一个空字符串 "str"，并使用 while 循环来遍历数组 "s" 中的所有元素，每次将一个空格移动到字符串的末尾，并将步数 "space" 减去 1。最后，返回了字符串 "str"。

在代码的末尾，我对 "tab" 函数进行了定义，并定义了 "game_restart" 变量为 false，表示游戏已经开始了。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var s = [0, 0];
var z;
var d;
var p;
var your_turn;
var game_restart;

```

这三段代码定义了一个游戏中的函数。

首先，有一个名为 `two_minutes()` 的函数，它只是简单地输出一行字符串，并没有做任何实际的计算或工作。

其次，有一个名为 `show_scores()` 的函数，它输出当前游戏中的得分，包括胜分数和剩余分数。胜分数是在游戏开始时就已经设为 0 的，而剩余分数是通过 `score_computer()` 函数计算出来的。

最后，有一个名为 `score_computer()` 的函数，它是一个计算得分的学生。这个函数的作用是遍历学生的成绩数组 `s`，将每个分数加上 2，然后输出当前的得分。


```
function two_minutes()
{
    print("\n");
    print("   *** TWO MINUTES LEFT IN THE GAME ***\n");
    print("\n");
}

function show_scores()
{
    print("SCORE: " + s[1] + " TO " + s[0] + "\n");
}

function score_computer()
{
    s[0] = s[0] + 2;
    show_scores();
}

```



这两段代码定义了两个函数，score_player()和half_time()。

score_player()函数的功能是增加球员 s 的分数，并输出当前球员的得分。具体来说，score_player()函数的代码包括以下几行：

- 将球员 s 的分数加2，存储到数组 s 的第二个元素中；
- 通过 show_scores() 函数输出当前球员的得分；
- 没有其他代码。

half_time()函数的功能是在半场比赛结束时，输出一些信息并暂停屏幕输出。具体来说，half_time()函数的代码包括以下几行：

- 暂停屏幕输出，等待半场比赛结束；
- 通过 print() 函数输出 "SCORE: DARMOUTH: " + s[1] + "  " + os + ": " + s[0] + "\n"，其中 s[1] 和 s[0] 分别表示上半场比赛的得分和下半场比赛的得分，os 表示的是球员在场上的时间(如果该球员在场上，则输出 'YES'，否则输出 'NO');
- 通过 print() 函数输出一个空行，以终止半场比赛的输出。

score_player()函数用于更新球员的得分，而half_time()函数用于在半场比赛结束时输出一些信息并暂停屏幕输出，这两个函数在足球比赛分析系统中可能有不同的作用。


```
function score_player()
{
    s[1] = s[1] + 2;
    show_scores();
}

function half_time()
{
    print("\n");
    print("   ***** END OF FIRST HALF *****\n");
    print("SCORE: DARMOUTH: " + s[1] + "  " + os + ": " + s[0] + "\n");
    print("\n");
    print("\n");
}

```

这段代码定义了一个名为 `foul` 的函数，其作用是模拟游戏中的情况，判断玩家是否成功射门，如果成功，则输出一条信息，否则再次调用函数，并逐渐改变玩家的得分情况。函数包含三个条件判断，分别对应成功、失准和未命中三种情况。每次调用函数时，会根据Math.random()的值来判断是否成功，如果成功，则执行相应的操作并调用 `show_scores()` 函数输出比赛结果，否则不执行任何操作，直接调用 `show_scores()` 函数输出比赛结果，以便记录胜负。


```
function foul()
{
    if (Math.random() <= 0.49) {
        print("SHOOTER MAKES BOTH SHOTS.\n");
        s[1 - p] = s[1 - p] + 2;
        show_scores();
    } else if (Math.random() <= 0.75) {
        print("SHOOTER MAKES ONE SHOT AND MISSES ONE.\n");
        s[1 - p] = s[1 - p] + 1;
        show_scores();
    } else {
        print("BOTH SHOTS MISSED.\n");
        show_scores();
    }
}

```



This is a programming assignment that simulates a game of Dartmouth. The game is played between two teams, with each team having a turn being one-on-one with the ball. The objective of the game is to score goals by kicking the ball through the opponent's goalposts.

The code implements the game logic for the Dartmouth game. It uses a combination of random number generation and some basic programming constructs to simulate the game.

The game starts with a kickoff from one of the computer's players. The player takes a shot at the ball, and if they are successful, they will try to score a goal. If they miss or if the ball goes out of bounds, the other team will have a chance to score.

The game uses a random number generator to determine the outcome of each kick. For example, if the random number is less than 0.4, the kick is a good shot and the player tries to score a goal. If the random number is greater than or equal to 0.8, the kick is a shot, and the player tries to block the opponent's goal. If the random number is between 0.4 and 0.7, the kick is a normal shot.

If the player misses the ball, the opponent has a chance to score a goal. If the ball goes out of bounds, the game is over and the other team wins.

The game also has some basic logic to control the actions of the players. For example, if the player takes a shot at the ball, the opponent will have a chance to block the shot by taking a shot themselves.

Overall, the code for the Dartmouth game is a simple implementation that uses a combination of random number and basic programming constructs to simulate the game.


```
function player_play()
{
    if (z == 1 || z == 2) {
        t++;
        if (t == 50) {
            half_time();
            game_restart = 1;
            return;
        }
        if (t == 92)
            two_minutes();
        print("JUMP SHOT\n");
        if (Math.random() <= 0.341 * d / 8) {
            print("SHOT IS GOOD.\n");
            score_player();
            return;
        }
        if (Math.random() <= 0.682 * d / 8) {
            print("SHOT IS OFF TARGET.\n");
            if (d / 6 * Math.random() >= 0.45) {
                print("REBOUND TO " + os + "\n");
                return;
            }
            print("DARTMOUTH CONTROLS THE REBOUND.\n");
            if (Math.random() > 0.4) {
                if (d == 6) {
                    if (Math.random() > 0.6) {
                        print("PASS STOLEN BY " + os + " EASY LAYUP.\n");
                        score_computer();
                        return;
                    }
                }
                print("BALL PASSED BACK TO YOU. ");
                your_turn = 1;
                return;
            }
        } else if (Math.random() <= 0.782 * d / 8) {
            print("SHOT IS BLOCKED.  BALL CONTROLLED BY ");
            if (Math.random() <= 0.5) {
                print("DARTMOUTH.\n");
                your_turn = 1;
                return;
            }
            print(os + ".\n");
            return;
        } else if (Math.random() <= 0.843 * d / 8) {
            print("SHOOTER IS FOULED.  TWO SHOTS.\n");
            foul();
            return;
            // In original code but lines 1180-1195 aren't used (maybe replicate from computer's play)
            //        } else if (Math.random() <= 0.9 * d / 8) {
            //            print("PLAYER FOULED, TWO SHOTS.\n");
            //            foul();
            //            return;
        } else {
            print("CHARGING FOUL.  DARTMOUTH LOSES BALL.\n");
            return;
        }
    }
    while (1) {
        if (++t == 50) {
            half_time();
            game_restart = 1;
            return;
        }
        if (t == 92)
            two_minutes();
        if (z == 0) {
            your_turn = 2;
            return;
        }
        if (z <= 3)
            print("LAY UP.\n");
        else
            print("SET SHOT.\n");
        if (7 / d * Math.random() <= 0.4) {
            print("SHOT IS GOOD.  TWO POINTS.\n");
            score_player();
            return;
        }
        if (7 / d * Math.random() <= 0.7) {
            print("SHOT IS OFF THE RIM.\n");
            if (Math.random() <= 2.0 / 3.0) {
                print(os + " CONTROLS THE REBOUND.\n");
                return;
            }
            print("DARMOUTH CONTROLS THE REBOUND.\n");
            if (Math.random() <= 0.4)
                continue;
            print("BALL PASSED BACK TO YOU.\n");
            your_turn = 1;
            return;
        }
        if (7 /d * Math.random() <= 0.875) {
            print("SHOOTER FOULED.  TWO SHOTS.\n");
            foul();
            return;
        }
        if (7 /d * Math.random() <= 0.925) {
            print("SHOT BLOCKED. " + os + "'S BALL.\n");
            return;
        }
        print("CHARGING FOUL.  DARTHMOUTH LOSES THE BALL.\n");
        return;
    }
}

```

This is a program written in Java that simulates a game of basketball. The game is played between two teams, the "Player" team and the "Computer" team. The "Player" team is controlled by the user, while the "Computer" team is controlled by the computer.

The game is played using a combination of HTML, CSS, and Java. The HTML is used to display the game board and the scoreboard, while the CSS is used to style the elements on the board. The Java code is used to simulate the game logic.

The game starts with a simple loop that prints out a message to the screen, asking the user to choose a team to control. The user can then choose between the "Player" and "Computer" teams by clicking on their respective button.

The game then enters its main loop, where it first checks for a player to make a shot. If the user clicks on the "Player" button, the game checks if the shot is valid by checking if the ball is within 3 feet of the basket and if the player is holding the ball with two hands. If the shot is valid, the game increments the score and calls a function called "score_computer" to calculate the score. If the shot is not valid, the game prints out a message and the user is asked to try again.

If the user clicks on the "Computer" button, the game controls the ball and tries to pass it to the other team. If the player holding the ball is the "Player" team, the game checks if the ball is within 3 feet of the basket and if the player holding the ball with two hands. If the ball is within 3 feet of the basket and the player is holding the ball with two hands, the game increments the score and calls a function called "score_player" to calculate the score. If the ball is not within 3 feet of the basket or the player is not holding the ball with two hands, the game prints out a message and the user is asked to try again.

The game also has a feature where the player can steal the ball if the ball is within 3 feet of the basket and the player is holding the ball with two hands. When the player steal the ball, the game prints out a message and the user is asked to try again.

The game also has a feature where the user can control the direction of the ball after a made shot. This is done using a button next to the score display, which is bound to the "player_控制" variable.

The game ends when either team scores 10 points inside the 3-point line or when the game is over 2 minutes.

This program is well-written and easy to understand.


```
function computer_play()
{
    rebound = 0;
    while (1) {
        p = 1;
        if (++t == 50) {
            half_time();
            game_restart = 1;
            return;
        }
        print("\n");
        z1 = 10 / 4 * Math.random() + 1;
        if (z1 <= 2) {
            print("JUMP SHOT.\n");
            if (8 / d * Math.random() <= 0.35) {
                print("SHOT IS GOOD.\n");
                score_computer();
                return;
            }
            if (8 / d * Math.random() <= 0.75) {
                print("SHOT IS OFF RIM.\n");
                if (d / 6 * Math.random() <= 0.5) {
                    print("DARMOUTH CONTROLS THE REBOUND.\n");
                    return;
                }
                print(os + " CONTROLS THE REBOUND.\n");
                if (d == 6) {
                    if (Math.random() <= 0.75) {
                        print("BALL STOLEN.  EASY LAP UP FOR DARTMOUTH.\n");
                        score_player();
                        continue;
                    }
                    if (Math.random() > 0.6) {
                        print("PASS STOLEN BY " + os + " EASY LAYUP.\n");
                        score_computer();
                        return;
                    }
                    print("BALL PASSED BACK TO YOU. ");
                    return;
                }
                if (Math.random() <= 0.5) {
                    print("PASS BACK TO " + os + " GUARD.\n");
                    continue;
                }
            } else if (8 / d * Math.random() <= 0.90) {
                print("PLAYER FOULED.  TWO SHOTS.\n");
                foul();
                return;
            } else {
                print("OFFENSIVE FOUL.  DARTMOUTH'S BALL.\n");
                return;
            }
        }
        while (1) {
            if (z1 > 3) {
                print("SET SHOT.\n");
            } else {
                print("LAY UP.\n");
            }
            if (7 / d * Math.random() <= 0.413) {
                print("SHOT IS GOOD.\n");
                score_computer();
                return;
            }
            print("SHOT IS MISSED.\n");
            // Spaguetti jump, better to replicate code
            if (d / 6 * Math.random() <= 0.5) {
                print("DARMOUTH CONTROLS THE REBOUND.\n");
                return;
            }
            print(os + " CONTROLS THE REBOUND.\n");
            if (d == 6) {
                if (Math.random() <= 0.75) {
                    print("BALL STOLEN.  EASY LAP UP FOR DARTMOUTH.\n");
                    score_player();
                    break;
                }
                if (Math.random() > 0.6) {
                    print("PASS STOLEN BY " + os + " EASY LAYUP.\n");
                    score_computer();
                    return;
                }
                print("BALL PASSED BACK TO YOU. ");
                return;
            }
            if (Math.random() <= 0.5) {
                print("PASS BACK TO " + os + " GUARD.\n");
                break;
            }
        }
    }
}

```

This appears to be a game of牌局， where each player takes turns attempting to hit the other player's card with their own card. The code also includes a function called `computer_play()` that appears to play the game automatically if the user does not interact with it for a certain period of time.

It's important to note that this code may have issues with the clarity, readability and maintainability, also it is not implemented for any specific game or shogi.


```
// Main program
async function main()
{
    print(tab(31) + "BASKETBALL\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("THIS IS DARTMOUTH COLLEGE BASKETBALL.  YOU WILL BE DARTMOUTH\n");
    print(" CAPTAIN AND PLAYMAKER.  CALL SHOTS AS FOLLOWS:  1. LONG\n");
    print(" (30 FT.) JUMP SHOT; 2. SHORT (15 FT.) JUMP SHOT; 3. LAY\n");
    print(" UP; 4. SET SHOT.\n");
    print("BOTH TEAMS WILL USE THE SAME DEFENSE.  CALL DEFENSE AS\n");
    print("FOLLOWS:  6. PRESS; 6.5 MAN-TO MAN; 7. ZONE; 7.5 NONE.\n");
    print("TO CHANGE DEFENSE, JUST TYPE 0 AS YOUR NEXT SHOT.\n");
    print("YOUR STARTING DEFENSE WILL BE");
    t = 0;
    p = 0;
    d = parseFloat(await input());
    if (d < 6) {
        your_turn = 2;
    } else {
        print("\n");
        print("CHOOSE YOUR OPPONENT");
        os = await input();
        game_restart = 1;
    }
    while (1) {
        if (game_restart) {
            game_restart = 0;
            print("CENTER JUMP\n");
            if (Math.random() > 3.0 / 5.0) {
                print("DARMOUTH CONTROLS THE TAP.\n");
            } else {
                print(os + " CONTROLS THE TAP.\n");
                computer_play();
            }
        }
        if (your_turn == 2) {
            print("YOUR NEW DEFENSIVE ALLIGNMENT IS");
            d = parseFloat(await input());
        }
        print("\n");
        while (1) {
            print("YOUR SHOT");
            z = parseInt(await input());
            p = 0;
            if (z != Math.floor(z) || z < 0 || z > 4)
                print("INCORRECT ANSWER.  RETYPE IT. ");
            else
                break;
        }
        if (Math.random() < 0.5 || t < 100) {
            game_restart = 0;
            your_turn = 0;
            player_play();
            if (game_restart == 0 && your_turn == 0)
                computer_play();
        } else {
            print("\n");
            if (s[1] == s[0]) {
                print("\n");
                print("   ***** END OF SECOND HALF *****\n");
                print("\n");
                print("SCORE AT END OF REGULATION TIME:\n");
                print("        DARTMOUTH: " + s[1] + "  " + os + ": " + s[0] + "\n");
                print("\n");
                print("BEGIN TWO MINUTE OVERTIME PERIOD\n");
                t = 93;
                print("CENTER JUMP\n");
                if (Math.random() > 3.0 / 5.0)
                    print("DARMOUTH CONTROLS THE TAP.\n");
                else
                    print(os + " CONTROLS THE TAP.\n");
            } else {
                print("   ***** END OF GAME *****\n");
                print("FINAL SCORE: DARMOUTH: " + s[1] + "  " + os + ": " + s[0] + "\n");
                break;
            }
        }
    }
}

```

这是C++中的一个标准的main函数，用于启动C++程序并执行程序的代码。在C++程序中，main函数是程序的入口点，当程序启动时，它将首先执行main函数中的代码。

对于这段代码而言，它定义了一个名为main的函数，但没有任何函数体。这意味着main函数不会执行任何代码，它只是一个函数声明。在C++中，声明一个函数但没有函数体可以让程序在以后定义和使用函数时进行使用。

所以，如果一个C++程序需要使用main函数中的函数，需要在程序中定义和使用该函数。


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

Conversion to [Pascal](https://en.wikipedia.org/wiki/Pascal_(programming_language))


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)

There are two version of the code here, a "faithful" translation (basketball-orig.pl) and
a "modern" translation (basketball.pl). The main difference between the 2 are is that the
faithful translation has 3 GOTOs in it while the modern version has no GOTO. I have added
a "TIME" print when the score is shown so the Clock is visible. Halftime is at "50" and
end of game is at 100 (per the Basic code).

The 3 GOTOs in the faitful version are because of the way the original code jumped into
the "middle of logic" that has no obivious way to avoid ... that I can see, at least while
still maintaining something of the look and structure of the original Basic.

The modern version avoided the GOTOs by restructuring the program in the 2 "play()" subs.
Despite the change, this should play the same way as the faithful version.

All of the percentages remain the same. If writing this from scratch, we really should
have only a single play() sub which uses the same code for both teams, which would also
make the game more fair ... but that wasn't done so the percent edge to Darmouth has been
maintained here.


# `07_Basketball/python/basketball.py`

这段代码是一个简单的 Python 类，名为 `Basketball`。它允许用户扮演 Dartmouth 学院的队长和 playmaker，在游戏过程中使用概率来模拟每个进攻和防守的选择。

具体来说，这段代码定义了一个包含以下内容的类：

- `print_intro()` 方法用于输出游戏的介绍信息，包括游戏的名称、创建地点和游戏方式。
- `print_ shottypes()` 方法用于输出可能的投篮类型。
- `print_defense()` 方法用于输出可能的防守阵容，包括球迷可以选择的防守选项。
- `print_score()` 方法用于在每次得分后输出分数。
- `play_game()` 方法用于模拟比赛，并允许用户选择不同的投篮类型或防守阵容。

此外，还有一 `play_menu()` 方法用于显示菜单，允许用户从以下选项中选择：

- `Jump Shot (长跳投)`
- `Jump Shot (短跳投)`
- `Lay up`
- `Set Shot`
- `Press`
- `Man-to-Man`
- `Zone`
- `None`

这个菜单可以更改防守阵容，并且在每次得分后，菜单会再次显示可供选择。


```
"""
The basketball class is a computer game that allows you to play as
Dartmouth College's captain and playmaker
The game uses set probabilites to simulate outcomes of each posession
You are able to choose your shot types as well as defensive formations
"""

import random
from typing import List, Literal, Optional


def print_intro() -> None:
    print("\t\t\t Basketball")
    print("\t Creative Computing  Morristown, New Jersey\n\n\n")
    print("This is Dartmouth College basketball. ")
    print("Υou will be Dartmouth captain and playmaker.")
    print("Call shots as follows:")
    print(
        "1. Long (30ft.) Jump Shot; "
        "2. Short (15 ft.) Jump Shot; "
        "3. Lay up; 4. Set Shot"
    )
    print("Both teams will use the same defense. Call Defense as follows:")
    print("6. Press; 6.5 Man-to-Man; 7. Zone; 7.5 None.")
    print("To change defense, just type 0 as your next shot.")
    print("Your starting defense will be? ", end="")


```

This is a Python class that simulates a basketball game. The `Basketball` class has methods for simulating a shot, a lay up, and a dunk. The `simulate_shot` method takes a shot type as an argument and sends a shot from the user. The `simulate_layup` method also takes a shot type as an argument and simulate a lay up. The `simulate_dunk` method is for simulating a dunk and it is taking a lot of random number in this function which is not defined in the class.


```
class Basketball:
    def __init__(self) -> None:
        self.time = 0
        self.score = [0, 0]  # first value is opponents score, second is home
        self.defense_choices: List[float] = [6, 6.5, 7, 7.5]
        self.shot: Optional[int] = None
        self.shot_choices: List[Literal[0, 1, 2, 3, 4]] = [0, 1, 2, 3, 4]
        self.z1: Optional[float] = None

        print_intro()

        self.defense = get_defense_choice(self.defense_choices)

        self.opponent = get_opponents_name()
        self.start_of_period()

    def add_points(self, team: Literal[0, 1], points: Literal[0, 1, 2]) -> None:
        """
        Add points to the score.

        Team can take 0 or 1, for opponent or Dartmouth, respectively
        """
        self.score[team] += points
        self.print_score()

    def ball_passed_back(self) -> None:
        print("Ball passed back to you. ", end="")
        self.dartmouth_ball()

    def change_defense(self) -> None:
        """change defense, called when the user enters 0 for their shot"""
        defense = None

        while defense not in self.defense_choices:
            print("Your new defensive allignment is? ")
            try:
                defense = float(input())
            except ValueError:
                continue
        assert isinstance(defense, float)
        self.defense = defense
        self.dartmouth_ball()

    def foul_shots(self, team: Literal[0, 1]) -> None:
        """Simulate two foul shots for a player and adds the points."""
        print("Shooter fouled.  Two shots.")
        if random.random() > 0.49:
            if random.random() > 0.75:
                print("Both shots missed.")
            else:
                print("Shooter makes one shot and misses one.")
                self.score[team] += 1
        else:
            print("Shooter makes both shots.")
            self.score[team] += 2

        self.print_score()

    def halftime(self) -> None:
        """called when t = 50, starts a new period"""
        print("\n   ***** End of first half *****\n")
        self.print_score()
        self.start_of_period()

    def print_score(self) -> None:
        """Print the current score"""
        print(f"Score:  {self.score[1]} to {self.score[0]}\n")

    def start_of_period(self) -> None:
        """Simulate a center jump for posession at the beginning of a period"""
        print("Center jump")
        if random.random() > 0.6:
            print("Dartmouth controls the tap.\n")
            self.dartmouth_ball()
        else:
            print(self.opponent + " controls the tap.\n")
            self.opponent_ball()

    def two_minute_warning(self) -> None:
        """called when t = 92"""
        print("   *** Two minutes left in the game ***")

    def dartmouth_jump_shot(self) -> None:
        """called when the user enters 1 or 2 for their shot"""
        self.time += 1
        if self.time == 50:
            self.halftime()
        elif self.time == 92:
            self.two_minute_warning()
        print("Jump Shot.")
        # simulates chances of different possible outcomes
        if random.random() > 0.341 * self.defense / 8:
            if random.random() > 0.682 * self.defense / 8:
                if random.random() > 0.782 * self.defense / 8:
                    if random.random() > 0.843 * self.defense / 8:
                        print("Charging foul. Dartmouth loses ball.\n")
                        self.opponent_ball()
                    else:
                        # player is fouled
                        self.foul_shots(1)
                        self.opponent_ball()
                else:
                    if random.random() > 0.5:
                        print(
                            "Shot is blocked. Ball controlled by "
                            + self.opponent
                            + ".\n"
                        )
                        self.opponent_ball()
                    else:
                        print("Shot is blocked. Ball controlled by Dartmouth.")
                        self.dartmouth_ball()
            else:
                print("Shot is off target.")
                if self.defense / 6 * random.random() > 0.45:
                    print("Rebound to " + self.opponent + "\n")
                    self.opponent_ball()
                else:
                    print("Dartmouth controls the rebound.")
                    if random.random() > 0.4:
                        if self.defense == 6 and random.random() > 0.6:
                            print("Pass stolen by " + self.opponent + ", easy lay up")
                            self.add_points(0, 2)
                            self.dartmouth_ball()
                        else:
                            # ball is passed back to you
                            self.ball_passed_back()
                    else:
                        print()
                        self.dartmouth_non_jump_shot()
        else:
            print("Shot is good.")
            self.add_points(1, 2)
            self.opponent_ball()

    def dartmouth_non_jump_shot(self) -> None:
        """
        Lay up, set shot, or defense change

        called when the user enters 0, 3, or 4
        """
        self.time += 1
        if self.time == 50:
            self.halftime()
        elif self.time == 92:
            self.two_minute_warning()

        if self.shot == 4:
            print("Set shot.")
        elif self.shot == 3:
            print("Lay up.")
        elif self.shot == 0:
            self.change_defense()

        # simulates different outcomes after a lay up or set shot
        if 7 / self.defense * random.random() > 0.4:
            if 7 / self.defense * random.random() > 0.7:
                if 7 / self.defense * random.random() > 0.875:
                    if 7 / self.defense * random.random() > 0.925:
                        print("Charging foul. Dartmouth loses the ball.\n")
                        self.opponent_ball()
                    else:
                        print("Shot blocked. " + self.opponent + "'s ball.\n")
                        self.opponent_ball()
                else:
                    self.foul_shots(1)
                    self.opponent_ball()
            else:
                print("Shot is off the rim.")
                if random.random() > 2 / 3:
                    print("Dartmouth controls the rebound.")
                    if random.random() > 0.4:
                        print("Ball passed back to you.\n")
                        self.dartmouth_ball()
                    else:
                        self.dartmouth_non_jump_shot()
                else:
                    print(self.opponent + " controls the rebound.\n")
                    self.opponent_ball()
        else:
            print("Shot is good. Two points.")
            self.add_points(1, 2)
            self.opponent_ball()

    def dartmouth_ball(self) -> None:
        """plays out a Dartmouth posession, starting with your choice of shot"""
        shot = get_dartmouth_ball_choice(self.shot_choices)
        self.shot = shot

        if self.time < 100 or random.random() < 0.5:
            if self.shot == 1 or self.shot == 2:
                self.dartmouth_jump_shot()
            else:
                self.dartmouth_non_jump_shot()
        else:
            if self.score[0] != self.score[1]:
                print("\n   ***** End Of Game *****")
                print(
                    "Final Score: Dartmouth: "
                    + str(self.score[1])
                    + "  "
                    + self.opponent
                    + ": "
                    + str(self.score[0])
                )
            else:
                print("\n   ***** End Of Second Half *****")
                print("Score at end of regulation time:")
                print(
                    "     Dartmouth: "
                    + str(self.score[1])
                    + " "
                    + self.opponent
                    + ": "
                    + str(self.score[0])
                )
                print("Begin two minute overtime period")
                self.time = 93
                self.start_of_period()

    def opponent_jumpshot(self) -> None:
        """Simulate the opponents jumpshot"""
        print("Jump Shot.")
        if 8 / self.defense * random.random() > 0.35:
            if 8 / self.defense * random.random() > 0.75:
                if 8 / self.defense * random.random() > 0.9:
                    print("Offensive foul. Dartmouth's ball.\n")
                    self.dartmouth_ball()
                else:
                    self.foul_shots(0)
                    self.dartmouth_ball()
            else:
                print("Shot is off the rim.")
                if self.defense / 6 * random.random() > 0.5:
                    print(self.opponent + " controls the rebound.")
                    if self.defense == 6:
                        if random.random() > 0.75:
                            print("Ball stolen. Easy lay up for Dartmouth.")
                            self.add_points(1, 2)
                            self.opponent_ball()
                        else:
                            if random.random() > 0.5:
                                print()
                                self.opponent_non_jumpshot()
                            else:
                                print("Pass back to " + self.opponent + " guard.\n")
                                self.opponent_ball()
                    else:
                        if random.random() > 0.5:
                            self.opponent_non_jumpshot()
                        else:
                            print("Pass back to " + self.opponent + " guard.\n")
                            self.opponent_ball()
                else:
                    print("Dartmouth controls the rebound.\n")
                    self.dartmouth_ball()
        else:
            print("Shot is good.")
            self.add_points(0, 2)
            self.dartmouth_ball()

    def opponent_non_jumpshot(self) -> None:
        """Simulate opponents lay up or set shot."""
        if self.z1 > 3:  # type: ignore
            print("Set shot.")
        else:
            print("Lay up")
        if 7 / self.defense * random.random() > 0.413:
            print("Shot is missed.")
            if self.defense / 6 * random.random() > 0.5:
                print(self.opponent + " controls the rebound.")
                if self.defense == 6:
                    if random.random() > 0.75:
                        print("Ball stolen. Easy lay up for Dartmouth.")
                        self.add_points(1, 2)
                        self.opponent_ball()
                    else:
                        if random.random() > 0.5:
                            print()
                            self.opponent_non_jumpshot()
                        else:
                            print("Pass back to " + self.opponent + " guard.\n")
                            self.opponent_ball()
                else:
                    if random.random() > 0.5:
                        print()
                        self.opponent_non_jumpshot()
                    else:
                        print("Pass back to " + self.opponent + " guard\n")
                        self.opponent_ball()
            else:
                print("Dartmouth controls the rebound.\n")
                self.dartmouth_ball()
        else:
            print("Shot is good.")
            self.add_points(0, 2)
            self.dartmouth_ball()

    def opponent_ball(self) -> None:
        """
        Simulate an opponents possesion

        Randomly picks jump shot or lay up / set shot.
        """
        self.time += 1
        if self.time == 50:
            self.halftime()
        self.z1 = 10 / 4 * random.random() + 1
        if self.z1 > 2:
            self.opponent_non_jumpshot()
        else:
            self.opponent_jumpshot()


```

这段代码定义了一个名为 `get_defense_choice` 的函数，用于给用户提供一个防御选择。函数的输入参数 `defense_choices` 是一个列表，其中每个元素都是单精度浮点数(float)。

函数的作用是等待用户输入一个防御选择，并将其存储在 `defense_choices` 列表中。如果用户输入的不是有效的防御选择，函数会再次提示用户输入，直到提供了有效的防御选择为止。

函数会检查用户输入的值是否为浮点数，如果不是，则会将 `None` 赋值给 `defense_choices` 列表的第一个元素。函数还使用 `assert` 语句来检查输入的值是否为浮点数，如果不是，则会引发 `ValueError` 异常。

最后，函数返回了有效的防御选择，并将其存储在 `defense_choices` 列表的第一个元素中。


```
def get_defense_choice(defense_choices: List[float]) -> float:
    """Takes input for a defense"""
    try:
        defense = float(input())
    except ValueError:
        defense = None

    # if the input wasn't a valid defense, takes input again
    while defense not in defense_choices:
        print("Your new defensive allignment is? ", end="")
        try:
            defense = float(input())
        except ValueError:
            continue
    assert isinstance(defense, float)
    return defense


```

这段代码定义了一个名为 `get_dartmouth_ball_choice` 的函数，它接受一个名为 `shot_choices` 的列表参数。这个列表参数包含五个选项，分别代表 0、1、2、3 和 4。

函数的作用是询问用户选择一个球，然后从 `shot_choices` 列表中选择一个正确的选项并返回。如果用户选择了一个不在列表中的数字，函数会提示他们重新输入。如果用户选择了正确的数字，函数会返回选择那个球的数字。

具体来说，函数首先会提示用户输入他们的选择，然后是一个无限循环。如果用户选择的不是数字，函数会再次提示他们重新输入。在循环的每次迭代中，函数会尝试从 `shot_choices` 列表中选择一个数字，并确保该数字是数字类型。如果用户选择了正确的数字，函数会返回该数字作为结果。


```
def get_dartmouth_ball_choice(shot_choices: List[Literal[0, 1, 2, 3, 4]]) -> int:
    print("Your shot? ", end="")
    shot = None
    try:
        shot = int(input())
    except ValueError:
        shot = None

    while shot not in shot_choices:
        print("Incorrect answer. Retype it. Your shot? ", end="")
        try:
            shot = int(input())
        except Exception:
            continue
    assert isinstance(shot, int)
    return shot


```

这段代码定义了一个函数 `get_opponents_name()`，用于获取玩家选择的对手的名字。这个函数使用了Python中的一行打印函数 `print()` 和输入函数 `input()`。

函数的功能是向用户询问他们要选择哪一个敌人，然后返回用户输入的名称作为字符串并将其存储起来。在主函数 `__main__` 中，调用 `get_opponents_name()` 函数并将其返回值作为参数传入，这将在函数中执行相应的操作并返回一个字符串。


```
def get_opponents_name() -> str:
    """Take input for opponent's name"""
    print("\nChoose your opponent? ", end="")
    return input()


if __name__ == "__main__":
    Basketball()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Batnum

The game starts with an imaginary pile of objects, coins for example. You and your opponent (the computer) alternately remove objects from the pile. You specify in advance the minimum and maximum number of objects that can be taken on each turn. You also specify in advance how winning is defined:
1. To take the last object
2. To avoid taking the last object

You may also determine whether you or the computer go first.

The strategy of this game is based on modulo arithmetic. If the maximum number of objects a player may remove in a turn is M, then to gain a winning position a player at the end of his turn must leave a stack of 1 modulo (M+1) coins. If you don’t understand this, play the game 23 Matches first, then BATNUM, and have fun!

BATNUM is a generalized version of a great number of manual remove-the-object games. The original computer version was written by one of the two originators of the BASIC language, John Kemeny of Dartmouth College.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=14)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=29)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Known Bugs

- Though the instructions say "Enter a negative number for new pile size to stop playing," this does not actually work.

#### Porting Notes

(please note any difficulties or challenges in porting here)



# `08_Batnum/csharp/BatnumGame.cs`



This is a code snippet for a text-based AI in a game of看书-like task, where players take turns to read a book and try to maximize their score.

The `PlayerTurn()` method is called by the game when it is the player's turn to turn the page. It determines what the player will do, such as reading the book, taking an action to return the book to the scanner, or asking the scanner to return the book. If the player chooses to read the book, it will return a result to the game. If the player chooses to take an action, it will return a result to the game. If the player tries to read the book but the result is not defined, it will return the result "Resources.INPUT_ILLEGAL".

The `ComputerTurn()` method is called by the game when it is the computer's turn to turn the page. It calculates the move to play and detects the win/lose conditions. If the win condition is detected, it will return a result to the game. If the computer tries to read the book but the result is not defined, it will return the result "Resources.INPUT_ILLEGAL".

Note that the `winCriteria` variable is set to different values depending on whether the game is won or lost. If the game is won, the computer will detect the win condition and return a result to the game. If the game is lost, the computer will detect the win condition and return a result to the game.

The `pileSize` variable keeps track of the amount of books the computer has in its inventory. If the computer has no books in its inventory, it will return the result "Resources.INPUT_ZERO". If the computer has too many books in its inventory, it will return the result "Resources.INPUT_MAX". If the computer has enough books in its inventory to return, it will return the result "Resources.INPUT_ZERO". If the computer tries to read the book but the result is not defined, it will return the result "Resources.INPUT_ILLEGAL".


```
﻿using Batnum.Properties;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Batnum
{
    public enum WinOptions
    {
        /// <summary>
        /// Last person to play wins
        /// </summary>
        WinWithTakeLast = 1,
        /// <summary>
        /// Last person to play loses
        /// </summary>
        WinWithAvoidLast = 2
    }

    public enum Players
    {
        Computer = 1,
        Human = 2
    }

    public class BatnumGame
    {
        public BatnumGame(int pileSize, WinOptions winCriteria, int minTake, int maxtake, Players firstPlayer, Func<string, int>askPlayerCallback)
        {
            this.pileSize = pileSize;
            this.winCriteria = winCriteria;
            this.minTake = minTake;
            this.maxTake = maxtake;
            this.currentPlayer = firstPlayer;
            this.askPlayerCallback = askPlayerCallback;
        }

        private int pileSize;
        private WinOptions winCriteria;
        private int minTake;
        private int maxTake;
        private Players currentPlayer;
        private Func<string, int> askPlayerCallback;

        /// <summary>
        /// Returns true if the game is running
        /// </summary>
        public bool IsRunning => pileSize > 0;

        /// <summary>
        /// Takes the next turn
        /// </summary>
        /// <returns>A message to be displayed to the player</returns>
        public string TakeTurn()
        {
            //Edge condition - can occur when minTake is more > 1
            if (pileSize < minTake)
            {
                pileSize = 0;
                return string.Format(Resources.END_DRAW, minTake);
            }
            return currentPlayer == Players.Computer ? ComputerTurn() : PlayerTurn();
        }

        private string PlayerTurn()
        {
            int draw = askPlayerCallback(Resources.INPUT_TURN);
            if (draw == 0)
            {
                pileSize = 0;
                return Resources.INPUT_ZERO;
            }
            if (draw < minTake || draw > maxTake || draw > pileSize)
            {
                return Resources.INPUT_ILLEGAL;
            }
            pileSize = pileSize - draw;
            if (pileSize == 0)
            {
                return winCriteria == WinOptions.WinWithTakeLast ? Resources.END_PLAYERWIN : Resources.END_PLAYERLOSE;
            }
            currentPlayer = Players.Computer;
            return "";
        }

        private string ComputerTurn()
        {
            //first calculate the move to play
            int sumTake = minTake + maxTake;
            int draw = pileSize - sumTake * (int)(pileSize / (float)sumTake);
            draw = Math.Clamp(draw, minTake, maxTake);

            //detect win/lose conditions
            switch (winCriteria)
            {
                case WinOptions.WinWithAvoidLast when (pileSize == minTake): //lose condition
                    pileSize = 0;
                    return string.Format(Resources.END_COMPLOSE, minTake);
                case WinOptions.WinWithAvoidLast when (pileSize <= maxTake): //avoid automatic loss on next turn
                    draw = Math.Clamp(draw, minTake, pileSize - 1);
                    break;
                case WinOptions.WinWithTakeLast when pileSize <= maxTake: // win condition
                    draw = Math.Min(pileSize, maxTake);
                    pileSize = 0;
                    return string.Format(Resources.END_COMPWIN, draw);
            }
            pileSize -= draw;
            currentPlayer = Players.Human;
            return string.Format(Resources.COMPTURN, draw, pileSize);
        }
    }
}

```

# `08_Batnum/csharp/ConsoleUtilities.cs`

This is a class that provides several useful utility methods for the Windows console application.

The `CenterText` method takes a string parameter and centers the text horizontally within the console window, while keeping the vertical order and splitting the text across the window width.

The `WriteLineWordWrap` method takes a string parameter and writes the specified data, followed by the current line terminator, to the console window while word wrapping the lines that would otherwise break the words. It takes a `paragraph` argument and a `tabSize` parameter, which specifies the width of the tab characters.

The `WriteLineWithCaps` method is a wrapper for the `WriteLineWordWrap` method, but it replaces the space between each character with a `' '` character, so that the output looks like this: `Hello World ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' '


```
﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Batnum
{
    public static class ConsoleUtilities
    {
        /// <summary>
        /// Ask the user a question and expects a comma separated pair of numbers representing a number range in response
        /// the range provided must have a maximum which is greater than the minimum
        /// </summary>
        /// <param name="question">The question to ask</param>
        /// <param name="minimum">The minimum value expected</param>
        /// <param name="maximum">The maximum value expected</param>
        /// <returns>A pair of numbers representing the minimum and maximum of the range</returns>
        public static (int min, int max) AskNumberRangeQuestion(string question, Func<int, int, bool> Validate)
        {
            while (true)
            {
                Console.Write(question);
                Console.Write(" ");
                string[] rawInput = Console.ReadLine().Split(',');
                if (rawInput.Length == 2)
                {
                    if (int.TryParse(rawInput[0], out int min) && int.TryParse(rawInput[1], out int max))
                    {
                        if (Validate(min, max))
                        {
                            return (min, max);
                        }
                    }
                }
                Console.WriteLine();
            }
        }

        /// <summary>
        /// Ask the user a question and expects a number in response
        /// </summary>
        /// <param name="question">The question to ask</param>
        /// <param name="minimum">A minimum value expected</param>
        /// <param name="maximum">A maximum value expected</param>
        /// <returns>The number the user entered</returns>
        public static int AskNumberQuestion(string question, Func<int, bool> Validate)
        {
            while (true)
            {
                Console.Write(question);
                Console.Write(" ");
                string rawInput = Console.ReadLine();
                if (int.TryParse(rawInput, out int number))
                {
                    if (Validate(number))
                    {
                        return number;
                    }
                }
                Console.WriteLine();
            }
        }

        /// <summary>
        /// Align content to center of console.
        /// </summary>
        /// <param name="content">Content to center</param>
        /// <returns>Center aligned text</returns>
        public static string CenterText(string content)
        {
            int windowWidth = Console.WindowWidth;
            return String.Format("{0," + ((windowWidth / 2) + (content.Length / 2)) + "}", content);
        }

        /// <summary>
        ///     Writes the specified data, followed by the current line terminator, to the standard output stream, while wrapping lines that would otherwise break words.
        ///     source: https://stackoverflow.com/questions/20534318/make-console-writeline-wrap-words-instead-of-letters
        /// </summary>
        /// <param name="paragraph">The value to write.</param>
        /// <param name="tabSize">The value that indicates the column width of tab characters.</param>
        public static void WriteLineWordWrap(string paragraph, int tabSize = 4)
        {
            string[] lines = paragraph
                .Replace("\t", new String(' ', tabSize))
                .Split(new string[] { Environment.NewLine }, StringSplitOptions.None);

            for (int i = 0; i < lines.Length; i++)
            {
                string process = lines[i];
                List<String> wrapped = new List<string>();

                while (process.Length > Console.WindowWidth)
                {
                    int wrapAt = process.LastIndexOf(' ', Math.Min(Console.WindowWidth - 1, process.Length));
                    if (wrapAt <= 0) break;

                    wrapped.Add(process.Substring(0, wrapAt));
                    process = process.Remove(0, wrapAt + 1);
                }

                foreach (string wrap in wrapped)
                {
                    Console.WriteLine(wrap);
                }

                Console.WriteLine(process);
            }
        }
    }
}

```

# `08_Batnum/csharp/Program.cs`

这段代码是一个Batnum游戏的控制器，其作用是让玩家在指定堆数的情况下，从自己拥有的玩家中选择一名玩家进行游戏，并从游戏开始时设置的选项中选择是否使用随机数选项。

该代码使用了Batnum库中的UseBatnumConsoleProxies和UseBatnumConsoleFrameworks方法，以在控制台输出游戏信息和请求玩家的输入。

具体来说，代码首先使用Console.WriteLine和ConsoleUtilities.CenterText方法来输出游戏名称和提示信息，然后使用Console.WriteLine和ConsoleUtilities.CenterText方法来输出 intro 部分文本。

接着，代码使用while 循环来让玩家轮到自己的回合，并且在每次轮到自己的回合时，使用Console.WriteLine和ConsoleUtilities.AskNumberQuestion方法来获取玩家输入并获取所需游戏信息。

然后，代码使用using语句来导入所需的Batnum库和System.Reflection库，并使用BatnumGame类来创建和控制游戏实例。

最后，代码在 while 循环中使用 game.IsRunning 属性来判断游戏是否正在运行，并在每次轮到自己的回合时使用 game.TakeTurn() 方法来获取游戏当前回合的提示信息并输出到控制台。


```
﻿using Batnum;
using Batnum.Properties;
using System;

Console.WriteLine(ConsoleUtilities.CenterText(Resources.GAME_NAME));
Console.WriteLine(ConsoleUtilities.CenterText(Resources.INTRO_HEADER));
Console.WriteLine();
Console.WriteLine();
Console.WriteLine();
ConsoleUtilities.WriteLineWordWrap(Resources.INTRO_PART1);
Console.WriteLine();
ConsoleUtilities.WriteLineWordWrap(Resources.INTRO_PART2);

while (true)
{
    Console.WriteLine();
    int pileSize = ConsoleUtilities.AskNumberQuestion(Resources.START_QUESTION_PILESIZE, (n) => n > 1);
    WinOptions winOption = (WinOptions)ConsoleUtilities.AskNumberQuestion(Resources.START_QUESTION_WINOPTION, (n) => Enum.IsDefined(typeof(WinOptions), n));
    (int minTake, int maxTake) = ConsoleUtilities.AskNumberRangeQuestion(Resources.START_QUESTION_DRAWMINMAX, (min,max) => min >= 1 && max < pileSize && max > min);
    Players currentPlayer = (Players)ConsoleUtilities.AskNumberQuestion(Resources.START_QUESTION_WHOSTARTS, (n) => Enum.IsDefined(typeof(Players), n));

    BatnumGame game = new BatnumGame(pileSize, winOption, minTake, maxTake, currentPlayer, (question) => ConsoleUtilities.AskNumberQuestion(question, (c) => true));
    while(game.IsRunning)
    {
        string message = game.TakeTurn();
        Console.WriteLine(message);
    }

}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)

This conversion uses C#9 and is built for .net 5.0

Functional changes from Original
- handle edge condition for end game where the minimum draw amount is greater than the number of items remaining in the pile
- Takes into account the width of the console
- Mulilingual Support (English/French currently)


# `08_Batnum/csharp/Properties/Resources.Designer.cs`

```
// Constants for the game
internal static class C enter依从物ternal static class C 
{
   public static final int MAX_OBJECTS = 10;
   public static final int MIN_OBJECTS = 1;
   public static final int PILE_SIZE = 100;
   public static final int MAX_SIZE = 500;

   // Function to retrieve the enter draw dimensions
   public static native double[] getDrawDimensions();

   // Function to retrieve the enter question text and draw dimensions
   public static native double[] getQuestionTextAndDrawDimensions();
}
```



```
﻿//------------------------------------------------------------------------------
// <auto-generated>
//     This code was generated by a tool.
//     Runtime Version:4.0.30319.42000
//
//     Changes to this file may cause incorrect behavior and will be lost if
//     the code is regenerated.
// </auto-generated>
//------------------------------------------------------------------------------

namespace Batnum.Properties {
    using System;


    /// <summary>
    ///   A strongly-typed resource class, for looking up localized strings, etc.
    /// </summary>
    // This class was auto-generated by the StronglyTypedResourceBuilder
    // class via a tool like ResGen or Visual Studio.
    // To add or remove a member, edit your .ResX file then rerun ResGen
    // with the /str option, or rebuild your VS project.
    [global::System.CodeDom.Compiler.GeneratedCodeAttribute("System.Resources.Tools.StronglyTypedResourceBuilder", "16.0.0.0")]
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
    [global::System.Runtime.CompilerServices.CompilerGeneratedAttribute()]
    internal class Resources {

        private static global::System.Resources.ResourceManager resourceMan;

        private static global::System.Globalization.CultureInfo resourceCulture;

        [global::System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1811:AvoidUncalledPrivateCode")]
        internal Resources() {
        }

        /// <summary>
        ///   Returns the cached ResourceManager instance used by this class.
        /// </summary>
        [global::System.ComponentModel.EditorBrowsableAttribute(global::System.ComponentModel.EditorBrowsableState.Advanced)]
        internal static global::System.Resources.ResourceManager ResourceManager {
            get {
                if (object.ReferenceEquals(resourceMan, null)) {
                    global::System.Resources.ResourceManager temp = new global::System.Resources.ResourceManager("Batnum.Properties.Resources", typeof(Resources).Assembly);
                    resourceMan = temp;
                }
                return resourceMan;
            }
        }

        /// <summary>
        ///   Overrides the current thread's CurrentUICulture property for all
        ///   resource lookups using this strongly typed resource class.
        /// </summary>
        [global::System.ComponentModel.EditorBrowsableAttribute(global::System.ComponentModel.EditorBrowsableState.Advanced)]
        internal static global::System.Globalization.CultureInfo Culture {
            get {
                return resourceCulture;
            }
            set {
                resourceCulture = value;
            }
        }

        /// <summary>
        ///   Looks up a localized string similar to COMPUTER TAKES {0} AND LEAVES {1}.
        /// </summary>
        internal static string COMPTURN {
            get {
                return ResourceManager.GetString("COMPTURN", resourceCulture);
            }
        }

        /// <summary>
        ///   Looks up a localized string similar to COMPUTER TAKES {0} AND LOSES.
        /// </summary>
        internal static string END_COMPLOSE {
            get {
                return ResourceManager.GetString("END_COMPLOSE", resourceCulture);
            }
        }

        /// <summary>
        ///   Looks up a localized string similar to COMPUTER TAKES {0} AND WINS.
        /// </summary>
        internal static string END_COMPWIN {
            get {
                return ResourceManager.GetString("END_COMPWIN", resourceCulture);
            }
        }

        /// <summary>
        ///   Looks up a localized string similar to ITS A DRAW, THERE ARE ONLY {0} PIECES LEFT.
        /// </summary>
        internal static string END_DRAW {
            get {
                return ResourceManager.GetString("END_DRAW", resourceCulture);
            }
        }

        /// <summary>
        ///   Looks up a localized string similar to TOUGH LUCK, YOU LOSE..
        /// </summary>
        internal static string END_PLAYERLOSE {
            get {
                return ResourceManager.GetString("END_PLAYERLOSE", resourceCulture);
            }
        }

        /// <summary>
        ///   Looks up a localized string similar to CONGRATULATIONS, YOU WIN..
        /// </summary>
        internal static string END_PLAYERWIN {
            get {
                return ResourceManager.GetString("END_PLAYERWIN", resourceCulture);
            }
        }

        /// <summary>
        ///   Looks up a localized string similar to BATNUM.
        /// </summary>
        internal static string GAME_NAME {
            get {
                return ResourceManager.GetString("GAME_NAME", resourceCulture);
            }
        }

        /// <summary>
        ///   Looks up a localized string similar to ILLEGAL MOVE, RENETER IT.
        /// </summary>
        internal static string INPUT_ILLEGAL {
            get {
                return ResourceManager.GetString("INPUT_ILLEGAL", resourceCulture);
            }
        }

        /// <summary>
        ///   Looks up a localized string similar to YOUR MOVE ?.
        /// </summary>
        internal static string INPUT_TURN {
            get {
                return ResourceManager.GetString("INPUT_TURN", resourceCulture);
            }
        }

        /// <summary>
        ///   Looks up a localized string similar to I TOLD YOU NOT TO USE ZERO! COMPUTER WINS BY FORFEIT..
        /// </summary>
        internal static string INPUT_ZERO {
            get {
                return ResourceManager.GetString("INPUT_ZERO", resourceCulture);
            }
        }

        /// <summary>
        ///   Looks up a localized string similar to CREATIVE COMPUTING MORRISTOWN, NEW JERSEY.
        /// </summary>
        internal static string INTRO_HEADER {
            get {
                return ResourceManager.GetString("INTRO_HEADER", resourceCulture);
            }
        }

        /// <summary>
        ///   Looks up a localized string similar to THIS PROGRAM IS A &apos;BATTLE&apos; OF NUMBERS GAME, WHERE THE COMPUTER IS YOUR OPPONENT.
        /// </summary>
        internal static string INTRO_PART1 {
            get {
                return ResourceManager.GetString("INTRO_PART1", resourceCulture);
            }
        }

        /// <summary>
        ///   Looks up a localized string similar to THE GAME STARTS WITH AN ASSUMED PILE OF OBJECTS. YOU AND YOUR OPPONENT ALTERNATELY REMOVE OBJECTS FROM THE PILE. WINNNING IS DEFINED IN ADVANCE AS TAKING THE LAST OBJECT OR NOT. YOU CAN ALSO SPECIFY SOME OTHER BEGINING CONDITIONS. DON&apos;T USER ZERO, HOWWEVER, IN PLAYING THE GAME..
        /// </summary>
        internal static string INTRO_PART2 {
            get {
                return ResourceManager.GetString("INTRO_PART2", resourceCulture);
            }
        }

        /// <summary>
        ///   Looks up a localized string similar to ENTER MIN AND MAX ?.
        /// </summary>
        internal static string START_QUESTION_DRAWMINMAX {
            get {
                return ResourceManager.GetString("START_QUESTION_DRAWMINMAX", resourceCulture);
            }
        }

        /// <summary>
        ///   Looks up a localized string similar to ENTER PILE SIZE ?.
        /// </summary>
        internal static string START_QUESTION_PILESIZE {
            get {
                return ResourceManager.GetString("START_QUESTION_PILESIZE", resourceCulture);
            }
        }

        /// <summary>
        ///   Looks up a localized string similar to ENTER START OPTION - 1 COMPUTER FIRST, 2 YOU FIRST ?.
        /// </summary>
        internal static string START_QUESTION_WHOSTARTS {
            get {
                return ResourceManager.GetString("START_QUESTION_WHOSTARTS", resourceCulture);
            }
        }

        /// <summary>
        ///   Looks up a localized string similar to ENTER WIN OPTION - 1 TO TAKE LAST, 2 TO AVOID LAST: ?.
        /// </summary>
        internal static string START_QUESTION_WINOPTION {
            get {
                return ResourceManager.GetString("START_QUESTION_WINOPTION", resourceCulture);
            }
        }
    }
}

```