# BasicComputerGames源码解析 23

# `14_Bowling/java/Bowling.java`

该代码的主要作用是实现一个基于1970年代BASIC游戏的简单游戏。该游戏在Java中实现，主要的区别是缺乏一些BASIC游戏中的文本错误检查等新特性。

具体来说，该代码实现了以下功能：

1. 定义了一个名为"GameOfBowling"的类，该类包含以下成员变量：
	* 一个名为"玩家猜拳赢的拳数"的整型变量"playerPunch"；
	* 一个名为"玩家猜拳输的拳数"的整型变量"playerLoss"；
	* 一个名为"游戏得分"的整型变量"score"。

2. 实现了以下方法：
	* 玩家猜拳方法："玩家猜拳赢"；
	* 玩家猜拳方法："玩家猜拳输"。

3. 程序的主循环，该主循环在玩家猜拳时执行。首先，程序会提示玩家猜拳，如果猜对了，就显示游戏得分，并给出提示信息，告诉玩家实际得分比玩家猜的拳数多了多少。如果猜错了，就显示玩家猜的拳数比实际得分多了多少，并提示玩家再猜一次。程序会一直循环执行，直到玩家猜了27个不同的拳数，即BASIC游戏中的全部18个拳数。

该代码的实现主要依赖于基本输入输出流（Scanner）和基本数据类型（int整型）。


```
import java.util.Scanner;
import java.lang.Math;

/**
 * Game of Bowling
 * <p>
 * Based on the BASIC game of Bowling here
 * https://github.com/coding-horror/basic-computer-games/blob/main/14%20Bowling/bowling.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 *
 * Converted from BASIC to Java by Darren Cardenas.
 */

```

以上是一个简单的Bowling游戏的示例，该游戏实现了基本游戏流程，包括初始化、规则显示、游戏过程、得分统计等功能。玩家可以通过输入“Y”来开始游戏或“NO”来退出游戏。游戏中的 scores 变量保存了每个球的得分，在游戏过程中每当玩家得分， scores 数组会相应更新。游戏中的界面设计较为简单，仅能提供游戏规则的基本信息。

注意：这个示例中的 scores 数组初始化为 0，实际应用中可以考虑提供一个默认的分数值或者提供一个用户界面让玩家输入自定义分数。同时，这个示例没有错误处理机制，如在尝试连接到游戏服务器时如果网络连接异常，应该给出合适的错误提示。


```
public class Bowling {

  private final Scanner scan;  // For user input

  public Bowling() {

    scan = new Scanner(System.in);

  }  // End of constructor Bowling

  public void play() {

    showIntro();
    startGame();

  }  // End of method play

  private static void showIntro() {

    System.out.println(" ".repeat(33) + "BOWL");
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    System.out.println("\n\n");

  }  // End of method showIntro

  private void startGame() {

    int ball = 0;
    int bell = 0;
    int frame = 0;
    int ii = 0;  // Loop iterator
    int jj = 0;  // Loop iterator
    int kk = 0;  // Loop iterator
    int numPlayers = 0;
    int pinsDownBefore = 0;
    int pinsDownNow = 0;
    int player = 0;
    int randVal = 0;
    int result = 0;

    int[] pins = new int[16];

    int[][] scores = new int[101][7];

    String userResponse = "";

    System.out.println("WELCOME TO THE ALLEY");
    System.out.println("BRING YOUR FRIENDS");
    System.out.println("OKAY LET'S FIRST GET ACQUAINTED");
    System.out.println("");
    System.out.println("THE INSTRUCTIONS (Y/N)");
    System.out.print("? ");

    userResponse = scan.nextLine();

    if (userResponse.toUpperCase().equals("Y")) {
      printRules();
    }

    System.out.print("FIRST OF ALL...HOW MANY ARE PLAYING? ");
    numPlayers = Integer.parseInt(scan.nextLine());

    System.out.println("");
    System.out.println("VERY GOOD...");

    // Begin outer while loop
    while (true) {

      for (ii = 1; ii <= 100; ii++) {
        for (jj = 1; jj <= 6; jj++) {
          scores[ii][jj] = 0;
        }
      }

      frame = 1;

      // Begin frame while loop
      while (frame < 11) {

        // Begin loop through all players
        for (player = 1; player <= numPlayers; player++) {

          pinsDownBefore = 0;
          ball = 1;
          result = 0;

          for (ii = 1; ii <= 15; ii++) {
            pins[ii] = 0;
          }

          while (true) {

            // Ball generator using mod '15' system

            System.out.println("TYPE ROLL TO GET THE BALL GOING.");
            System.out.print("? ");
            scan.nextLine();

            kk = 0;
            pinsDownNow = 0;

            for (ii = 1; ii <= 20; ii++) {

              randVal = (int)(Math.random() * 100) + 1;

              for (jj = 1; jj <= 10; jj++) {

                if (randVal < 15 * jj) {
                  break;
                }
              }
              pins[15 * jj - randVal] = 1;
            }

            // Pin diagram

            System.out.println("PLAYER: " + player + " FRAME: " + frame + " BALL: " + ball);

            for (ii = 0; ii <= 3; ii++) {

              System.out.println("");

              System.out.print(" ".repeat(ii));

              for (jj = 1; jj <= 4 - ii; jj++) {

                kk++;

                if (pins[kk] == 1) {

                  System.out.print("O ");

                } else {

                  System.out.print("+ ");
                }
              }
            }

            System.out.println("");

            // Roll analysis

            for (ii = 1; ii <= 10; ii++) {
              pinsDownNow += pins[ii];
            }

            if (pinsDownNow - pinsDownBefore == 0) {
              System.out.println("GUTTER!!");
            }

            if (ball == 1 && pinsDownNow == 10) {
              System.out.println("STRIKE!!!!!");

              // Ring bell
              for (bell = 1; bell <= 4; bell++) {
                System.out.print("\007");
                try {
                  Thread.sleep(500);
                } catch (InterruptedException e) {
                  Thread.currentThread().interrupt();
                }
              }
              result = 3;
            }

            if (ball == 2 && pinsDownNow == 10) {
              System.out.println("SPARE!!!!");
              result = 2;
            }

            if (ball == 2 && pinsDownNow < 10) {
              System.out.println("ERROR!!!");
              result = 1;
            }

            if (ball == 1 && pinsDownNow < 10) {
              System.out.println("ROLL YOUR 2ND BALL");
            }

            // Storage of the scores

            System.out.println("");

            scores[frame * player][ball] = pinsDownNow;

            if (ball != 2) {
              ball = 2;
              pinsDownBefore = pinsDownNow;

              if (result != 3) {
                scores[frame * player][ball] = pinsDownNow - pinsDownBefore;
                if (result == 0) {
                  continue;
                }
              } else {
                scores[frame * player][ball] = pinsDownNow;
              }

            }
            break;
          }

          scores[frame * player][3] = result;

        }  // End loop through all players

        frame++;

      }  // End frame while loop

      System.out.println("FRAMES");

      System.out.print(" ");
      for (ii = 1; ii <= 10; ii++) {
        System.out.print(ii + " ");
      }

      System.out.println("");

      for (player = 1; player <= numPlayers; player++) {
        for (ii = 1; ii <= 3; ii++) {
          System.out.print(" ");
          for (jj = 1; jj <= 10; jj++) {
            System.out.print (scores[jj * player][ii] + " ");
          }
          System.out.println("");
        }
        System.out.println("");
      }

      System.out.println("DO YOU WANT ANOTHER GAME");
      System.out.print("? ");

      userResponse = scan.nextLine();

      if (!String.valueOf(userResponse.toUpperCase().charAt(0)).equals("Y")) {
        break;
      }

    }  // End outer while loop

  }  // End of method startGame

  public static void printRules() {

    System.out.println("THE GAME OF BOWLING TAKES MIND AND SKILL.DURING THE GAME");
    System.out.println("THE COMPUTER WILL KEEP SCORE.YOU MAY COMPETE WITH");
    System.out.println("OTHER PLAYERS[UP TO FOUR].YOU WILL BE PLAYING TEN FRAMES");
    System.out.println("ON THE PIN DIAGRAM 'O' MEANS THE PIN IS DOWN...'+' MEANS THE");
    System.out.println("PIN IS STANDING.AFTER THE GAME THE COMPUTER WILL SHOW YOUR");
    System.out.println("SCORES .");

  }  // End of method printRules

  public static void main(String[] args) {

    Bowling game = new Bowling();
    game.play();

  }  // End of method main

}  // End of class Bowling

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `14_Bowling/javascript/bowling.js`

这段代码的作用是创建了一个 Web 级别的输入框，当用户点击后，它会询问用户输入字符并将其存储在变量 `input_str` 中。然后，它将使用 `print` 函数将用户输入的字符打印到页面上。

具体来说，该代码由两部分组成：

1. `print` 函数，它将接收一个字符串参数 `str`，并将其作为文本添加到页面上元素的 `appendChild` 方法中。这个函数的作用是将文本添加到页面上，以便用户可以看到他们在输入框中输入的字符。
2. `input` 函数，它是一个 Promise  Promise，它返回一个函数，当用户点击输入框时，它将弹出一个输入框，询问用户输入字符。当用户点击时，它将创建一个输入元素，设置其样式和属性，并将输入框的 `addEventListener` 方法与 `keydown` 事件绑定在一起。当用户按下键盘上的数字 13 时，它将在不省略输入框内容的情况下读取用户输入的字符，并将其存储在 `input_str` 变量中。

之后，它会使用 `print` 函数将 `input_str` 变量中的字符打印到页面上。此外，它还使用 `print` 函数打印一个换行符，以便在将字符打印到页面上时可以将其与其他字符分隔开来。


```
// BOWLING
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

It looks like there is a bug in the original game. The bug is that when the game reaches the 2610 score, it doesn't reset the score and continues jumping to that score without restarting the game.

The fix to this bug is to when the score reaches the maximum score (2610), the game should reset to the score 1, and then restart the game with a new score.

Here's the updated code for how to do this:
```
print("STRIKE!!!\n");
print("MAX SCORE attained so far is 2610\n");
q = 3;

while (q <= 11) {
 print("STRIKE again...\n");
 q = 3;
}

print("MAX SCORE attained so far is 2610\n");
print("FRAMES\n");
for (i = 1; i <= 10; i++)
 print(" " + i + " ");
print("\n");

for (p = 1; p <= r; p++) {
 for (i = 1; i <= 3; i++) {
   print(" " + a[j * p][i] + " ");
 }
 print("\n");
}

print("DO YOU WANT ANOTHER GAME");
str = await input();
if (str.substr(0, 1) != "Y")
 break;

if (q == 11) {
 print("MAX SCORE attained so far is 2610\n");
 print("DO YOU WANT TO RESTART THE GAME");
 str = await input();
 if (str.substr(0, 1) != "Y")
   break;
 print("MAX SCORE attained so far is 1\n");
 q = 3;
}
```


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
    print(tab(34) + "BOWL\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    c = [];
    a = [];
    for (i = 0; i <= 15; i++)
        c[i] = 0;
    print("WELCOME TO THE ALLEY\n");
    print("BRING YOUR FRIENDS\n");
    print("OKAY LET'S FIRST GET ACQUAINTED\n");
    print("\n");
    print("THE INSTRUCTIONS (Y/N)\n");
    str = await input();
    if (str.substr(0, 1) == "Y") {
        print("THE GAME OF BOWLING TAKES MIND AND SKILL.DURING THE GAME\n");
        print("THE COMPUTER WILL KEEP SCORE.YOU MAY COMPETE WITH\n");
        print("OTHER PLAYERS[UP TO FOUR].YOU WILL BE PLAYING TEN FRAMES\n");
        print("ON THE PIN DIAGRAM 'O' MEANS THE PIN IS DOWN...'+' MEANS THE\n");
        print("PIN IS STANDING.AFTER THE GAME THE COMPUTER WILL SHOW YOUR\n");
        print("SCORES .\n");
    }
    print("FIRST OF ALL...HOW MANY ARE PLAYING");
    r = parseInt(await input());
    while (1) {
        print("\n");
        print("VERY GOOD...\n");
        for (i = 1; i <= 100; i++) {
            a[i] = [];
            for (j = 1; j <= 6; j++)
                a[i][j] = 0;
        }
        f = 1;
        do {
            for (p = 1; p <= r; p++) {
                // m = 0; // Repeated in original
                b = 1;
                m = 0;
                q = 0;
                for (i = 1; i <= 15; i++)
                    c[i] = 0;
                while (1) {
                    // Ball generator using mod '15' system
                    print("TYPE ROLL TO GET THE BALL GOING.\n");
                    ns = await input();
                    k = 0;
                    d = 0;
                    for (i = 1; i <= 20; i++) {
                        x = Math.floor(Math.random() * 100);
                        for (j = 1; j <= 10; j++)
                            if (x < 15 * j)
                                break;
                        c[15 * j - x] = 1;
                    }
                    // Pin diagram
                    print("PLAYER: " + p + " FRAME: " + f + " BALL: " + b + "\n");
                    print("\n");
                    for (i = 0; i <= 3; i++) {
                        str = "";
                        for (j = 1; j <= 4 - i; j++) {
                            k++;
                            while (str.length < i)
                                str += " ";
                            if (c[k] == 1)
                                str += "O ";
                            else
                                str += "+ ";
                        }
                        print(str + "\n");
                    }
                    // Roll analysis
                    for (i = 1; i <= 10; i++)
                        d += c[i];
                    if (d - m == 0)
                        print("GUTTER!!\n");
                    if (b == 1 && d == 10) {
                        print("STRIKE!!!!!\n");
                        q = 3;
                    }
                    if (b == 2 && d == 10) {
                        print("SPARE!!!!\n");
                        q = 2;
                    }
                    if (b == 2 && d < 10) {
                        print("ERROR!!!\n");
                        q = 1;
                    }
                    if (b == 1 && d < 10) {
                        print("ROLL YOUR 2ND BALL\n");
                    }
                    // Storage of the scores
                    print("\n");
                    a[f * p][b] = d;
                    if (b != 2) {
                        b = 2;
                        m = d;
                        if (q == 3) {
                            a[f * p][b] = d;
                        } else {
                            a[f * p][b] = d - m;
                            if (q == 0) // ROLL
                                continue;
                        }
                    }
                    break;
                }
                a[f * p][3] = q;
            }
        } while (++f < 11) ;
        print("FRAMES\n");
        for (i = 1; i <= 10; i++)
            print(" " + i + " ");
        print("\n");
        for (p = 1; p <= r; p++) {
            for (i = 1; i <= 3; i++) {
                for (j = 1; j <= 10; j++) {
                    print(" " + a[j * p][i] + " ");
                }
                print("\n");
            }
            print("\n");
        }
        print("DO YOU WANT ANOTHER GAME");
        str = await input();
        if (str.substr(0, 1) != "Y")
            break;
        // Bug in original game, jumps to 2610, without restarting P variable
    }
}

```

这是经典的 "Hello, World!" 程序，用于在计算机屏幕上输出 "Hello, World!" 消息。

在 C 语言中，`main()` 函数是程序的入口点，当程序运行时，首先执行的就是这个函数。因此，`main()` 函数可以被视为程序的 "门面"，因为它负责将程序与用户交互，并提供程序执行所需的资源。

在没有其他代码的情况下，`main()` 函数通常包含程序的入口点，也就是程序开始执行的地方。因此，当我们向用户输出 "Hello, World!" 时，就是通过 `main()` 函数来实现的。


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

###Bowling program in Perl

Run normally, this is a fairly faithful translation of the Basic game.
The only real differences are a few trivial fix-ups on the prints to make it
look better, and the player/frame/ball line was put before the "get the ball
going" line to make it more obvious who's turn it is.

However, if you run it with "-a" on the command line, it will go into
"advanced" mode, which means that "." is used to show pin down and "!" for
pin up, current running scores are shown at the end of each frame, and the
scoring also looks more normal at the end. This is all done because I think it
looks better and I wanted to see a score. Having a flag says you can play
whichever version of the game you like.

Note, the original code doesn't do the 10th frame correctly, in that it will
never do more than 2 balls, so the best score you can get is a 290.
This is true in both modes. That being said, it will always give you a mediocre
game; I don't think I've ever seen a score over 140.


# `14_Bowling/python/bowling.py`

这段代码定义了两个函数，simulate_roll() 和 calculate_score()。这两个函数的输入参数都是一个整数列表 pins，它们的作用是模拟掷骰子和计算得分。

simulate_roll() 的作用是模拟掷骰子。函数的输入参数是一个整数列表 pins，它表示骰子上的点数。在函数内部，使用 for 循环来生成一个 20 次的随机整数，如果生成的随机整数小于列表 pins 中的元素个数，那么将 pins 中的元素设置为 1。这样，模拟掷骰子的过程就可以在每次生成随机整数后，随机选择一个点数来改变骰子状态。

calculate_score() 的作用是计算得分。函数的输入参数是一个整数列表 rolls，它表示每次掷骰子的结果。函数内部首先计算当前轮次所有点数的和，如果当前轮次的点数和等于 10，就认为已经赢得了比赛，得分就是当前轮次的点数数。否则，就重新开始计算得分。在计算得分的过程中，使用一个变量 b 来记录当前轮次的点数和是否已经赢得了比赛，如果 b 的值为 2，就说明当前轮次已经开始了新的比赛。

总的来说，这两个函数都是对给定点数列表进行操作，以模拟掷骰子和计算得分。


```
import random
from typing import List


def simulate_roll(pins: List[int]) -> None:
    for _ in range(20):
        x = random.randint(0, 14)
        if x < len(pins):
            pins[x] = 1


def calculate_score(rolls: List[int]) -> int:
    score = 0
    frame = 1
    b = 1
    for index, pins in enumerate(rolls):
        score += pins
        if b == 1:
            if pins == 10:  # strike
                score += sum(rolls[index + 1 : index + 3])
                frame += 1
            else:
                b = 2
        else:
            if sum(rolls[index - 1 : index + 1]) == 10:  # spare
                score += rolls[index + 1]
            b = 1
            frame += 1
        if frame > 10:
            break

    return score


```

This appears to be a Python class that simulates a game of skittles. It has a `show` method that takes a list of pins and prints them out in a 4x4 grid. The `rolls` attribute is a list of the number of pins toppled on each roll. The class has a `calculate_score` method that takes the `rolls` list and returns the total score, based on the number of pins toppled in each roll. The `extra` attribute is a variable that determines the number of extra rolls to be added at the end of the game, if the score is still less than 10.


```
class Player:
    def __init__(self, name: str) -> None:
        self.name = name
        self.rolls: List[int] = []

    def play_frame(self, frame: int) -> None:
        extra = 0
        prev_score = 0
        pins = [0] * 10  # reset the pins
        for ball in range(2):
            simulate_roll(pins)
            score = sum(pins)
            self.show(pins)
            pin_count = score - prev_score
            self.rolls.append(pin_count)  # log the number of pins toppled this roll
            print(f"{pin_count} for {self.name}")
            if score - prev_score == 0:
                print("GUTTER!!!")
            if ball == 0:
                if score == 10:
                    print("STRIKE!!!")
                    extra = 2
                    break  # cannot roll more than once in a frame
                else:
                    print(f"next roll {self.name}")
            else:
                if score == 10:
                    print("SPARE!")
                    extra = 1

            prev_score = score  # remember previous pins to distinguish ...
        if frame == 9 and extra > 0:
            print(f"Extra rolls for {self.name}")
            pins = [0] * 10  # reset the pins
            score = 0
            for _ball in range(extra):
                if score == 10:
                    pins = [0] * 10
                simulate_roll(pins)
                score = sum(pins)
                self.rolls.append(score)

    def __str__(self) -> str:
        return f"{self.name}: {self.rolls}, total:{calculate_score(self.rolls)}"

    def show(self, pins: List[int]) -> None:
        pins_iter = iter(pins)
        print()
        for row in range(4):
            print(" " * row, end="")
            for _ in range(4 - row):
                p = next(pins_iter)
                print("O " if p else "+ ", end="")
            print()


```

这段代码定义了一个名为 `centre_text` 的函数，用于将文本居中显示。该函数接受两个参数：`text` 和 `width`。函数首先计算文本的长度 `t`，然后返回一个字符串，该字符串在 `width` 像素宽度内居中显示文本。接下来是函数的主函数：

```
def main():
   print(centre_text("Bowl", 80))
   print(centre_text("CREATIVE COMPUTING MORRISTOWN, NEW JERSEY", 80))
   print()
   print("WELCOME TO THE ALLEY.")
   print("BRING YOUR FRIENDS.")
   print("OKAY LET'S FIRST GET ACQUAINTED.")

   while True:
       print()
       if input("THE INSTRUCTIONS (Y/N)? ") in "yY":
           print("THE GAME OF BOWLING TAKES MIND AND SKILL. DURING THE GAME")
           print("THE COMPUTER WILL KEEP SCORE. YOU MAY COMPETE WITH")
           print("OTHER PLAYERS[UP TO FOUR]. YOU WILL BE PLAYING TEN FRAMES.")
           print("ON THE PIN DIAGRAM 'O' MEANS THE PIN IS DOWN...'+' MEANS THE")
           print("PIN IS STANDING. AFTER THE GAME THE COMPUTER WILL SHOW YOUR")
           print("SCORES.")

       total_players = int(input("FIRST OF ALL...HOW MANY ARE PLAYING? "))
       player_names = []
       print()
       print("VERY good...")
       for index in range(total_players):
           player_names.append(Player(input(f"Enter name for player {index + 1}: ")))

       for frame in range(10):
           for player in player_names:
               player.play_frame(frame)

       for player in player_names:
           print(player)

       if input("DO YOU WANT ANOTHER GAME? ") not in "yY":
           break
```

这段代码的主要目的是让玩家在获得与朋友一起参加棒球游戏时，可以在棒球比赛的过程中通过与计算机或其他玩家进行竞争来获得胜利。游戏开始时，程序会询问玩家是否想要开始游戏，如果玩家选择同意，那么程序会进入一个循环，该循环将提示玩家输入游戏中的各种参数，包括球员姓名、分数等。在循环中，玩家还可以通过输入来告诉计算机他们想要移动的步数，如果输入为数字，计算机将尝试让球移动到指定方向，否则，如果玩家想要结束游戏，程序将结束并显示分数。


```
def centre_text(text: str, width: int) -> str:
    t = len(text)
    return (" " * ((width - t) // 2)) + text


def main() -> None:
    print(centre_text("Bowl", 80))
    print(centre_text("CREATIVE COMPUTING MORRISTOWN, NEW JERSEY", 80))
    print()
    print("WELCOME TO THE ALLEY.")
    print("BRING YOUR FRIENDS.")
    print("OKAY LET'S FIRST GET ACQUAINTED.")

    while True:
        print()
        if input("THE INSTRUCTIONS (Y/N)? ") in "yY":
            print("THE GAME OF BOWLING TAKES MIND AND SKILL. DURING THE GAME")
            print("THE COMPUTER WILL KEEP SCORE. YOU MAY COMPETE WITH")
            print("OTHER PLAYERS[UP TO FOUR]. YOU WILL BE PLAYING TEN FRAMES.")
            print("ON THE PIN DIAGRAM 'O' MEANS THE PIN IS DOWN...'+' MEANS THE")
            print("PIN IS STANDING. AFTER THE GAME THE COMPUTER WILL SHOW YOUR")
            print("SCORES.")

        total_players = int(input("FIRST OF ALL...HOW MANY ARE PLAYING? "))
        player_names = []
        print()
        print("VERY GOOD...")
        for index in range(total_players):
            player_names.append(Player(input(f"Enter name for player {index + 1}: ")))

        for frame in range(10):
            for player in player_names:
                player.play_frame(frame)

        for player in player_names:
            print(player)

        if input("DO YOU WANT ANOTHER GAME? ") not in "yY":
            break


```

这段代码是一个用于将文本游戏转换为Python程序的脚本。它首先检查当前脚本是否为__main__函数，如果是，则执行main函数。

在这段注释中，开发人员对程序进行了修改，以使其更易于理解和维护。具体来说，他们添加了一个Player类来存储玩家数据，解决了许多原来程序中存在的计算索引问题。他们也增加了新的功能，例如允许玩家在最后一个框框中得分高于10分，以及允许玩家在最后一轮中进行平局。


```
if __name__ == "__main__":
    main()


############################################################################################
#
# This is a fairly straight conversion to python with some exceptions.
# I have kept most of the upper case text that the program prints.
# I have added the feature of giving names to players.
# I have added a Player class to store player data in.
# This last change works around the problems in the original storing data in a matrix.
# The original had bugs in calculating indexes which meant that the program
# would overwrite data in the matrix, so the results printed out contained errors.
# The last change is to do with the strict rules which allow extra rolls if the player
# scores a spare or strike in the last frame.
```

这段代码是一个C++程序，它的作用是允许用户进行额外的掷骰子和计算得分。它包含两个主要函数：`roll()`和`score()`。

1. `roll()`函数用于让用户输入额外的骰子个数，并掷骰子。函数接受两个整数参数，一个表示骰子的面数，另一个表示骰子面的计分规则（如是否使用奇怪的得分计算方式）。

2. `score()`函数使用用户输入的骰子个数和得分计分规则计算得分。它将计算得到的总分数存储在变量`score`中。

由于在代码中没有输出语句，因此程序不会输出给用户进行输入。用户输入的额外骰子个数将用于计算得分，得分将根据指定的得分计分规则计算。


```
# This program allows these extra rolls and also calculates the proper score.
#
############################################################################################

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Boxing

This program simulates a three-round Olympic boxing match. The computer coaches one of the boxers and determines his punches and defences, while you do the same for your boxer. At the start of the match, you may specify your man’s best punch and his vulnerability.

There are approximately seven major punches per round, although this may be varied. The best out of three rounds wins.

Jesse Lynch of St. Paul, Minnesota created this program.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=28)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=43)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Known Bugs

- The code that handles player punch type 1 checks for opponent weakness type 4; this is almost certainly a mistake.

- Line breaks or finishing messages are omitted in various cases.  For example, if the player does a hook, and that's the opponent's weakness, then 7 points are silently awarded without outputting any description or line break, and the next sub-round will begin on the same line.

- When the opponent selects a hook, control flow falls through to the uppercut case.  Perhaps related, a player weakness of type 2 (hook) never has any effect on the game.

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `15_Boxing/csharp/AttackStrategy.cs`



这个代码定义了一个抽象类 AttackStrategy，它继承自 Boxer 类，用于处理攻击策略。

在 AttackStrategy 类中，我们定义了一些常量，包括 KnockoutDamageThreshold，它代表 knockout 伤害阈值，以及 Stack 类，用于在 Stack 上添加动作。

我们还定义了一个 Attack 方法，在 method 中，我们首先获取最佳的攻击拳击，然后对攻击伤害进行增加，并使用 work Stack 中的动作来添加到 Attach 的工作流中。

接着，我们定义了一些 abstract 方法 FullSwing,Hook,Uppercut 和 Jab，这些方法重写了 AttributePunch 类中的抽象方法，用于生成不同的攻击拳击。

最后，我们定义了一个 RegisterKnockout 方法，用于在击倒拳击后输出一条消息，并清除 Stack。

这个抽象类定义了一个可以攻击敌人的攻击策略，通过不同的拳击来对敌人造成不同的伤害，并使用 Work Stack 中的动作来添加到 Attach 的工作流中，以使攻击流更加流畅。


```
﻿namespace Boxing;

public abstract class AttackStrategy
{
    protected const int KnockoutDamageThreshold = 35;
    protected readonly Boxer Other;
    protected readonly Stack<Action> Work;
    private readonly Action _notifyGameEnded;

    public AttackStrategy(Boxer other, Stack<Action> work, Action notifyGameEnded)
    {
        Other = other;
        Work = work;
        _notifyGameEnded = notifyGameEnded;
    }

    public void Attack()
    {
        var punch = GetPunch();
        if (punch.IsBestPunch)
        {
            Other.DamageTaken += 2;
        }

        Work.Push(punch.Punch switch
        {
            Punch.FullSwing => FullSwing,
            Punch.Hook => Hook,
            Punch.Uppercut => Uppercut,
            _ => Jab
        });
    }

    protected abstract AttackPunch GetPunch();
    protected abstract void FullSwing();
    protected abstract void Hook();
    protected abstract void Uppercut();
    protected abstract void Jab();

    protected void RegisterKnockout(string knockoutMessage)
    {
        Work.Clear();
        _notifyGameEnded();
        Console.WriteLine(knockoutMessage);
    }

    protected record AttackPunch(Punch Punch, bool IsBestPunch);
}

```

# `15_Boxing/csharp/Boxer.cs`

这段代码定义了一个名为Boxer的类，用于实现拳击比赛中的数据结构。这个类包含了一些拳击比赛中常见的属性和方法，如拳击手、拳击手法、伤害值、胜利判断等等。

具体来说，这个类包含以下属性和方法：

- _wins：表示拳击手在比赛中的获胜次数，初始值为0。
- Name：用于存储拳击手的姓名，初始值为一个空字符串。
- BestPunch：包含拳击手法，是一个Punch类型的成员变量，用于记录当前拳击手最有效的拳击手法。
- Vulnerability：包含攻击力，是一个Punch类型的成员变量，用于记录当前拳击手最弱的攻击手法。
- SetName：用于设置拳击手的姓名，使用Console.WriteLine方法将提示字符串打印出来，并要求用户输入。
- DamageTaken：表示拳击手在上一回合中的总伤害值，用于记录拳击手上一回合造成的伤害值。
- ResetForNewRound：用于清空DamageTaken，使它重新归零，以便在每一回合的开始时重新计算。
- RecordWin：使用ResetForNewRound方法，将DamageTaken归零，并增加获胜次数。
- IsWinner：判断拳击手是否获胜，如果获胜次数大于或等于2，则判断为获胜者。
- ToString：重写toString方法，用于将拳击手姓名打印出来。

这个类还实现了三个默认方法：Punch.Boxing、Punch.Clermont和Punch.F次年。


```
﻿namespace Boxing;

public class Boxer
{
    private int _wins;

    private string Name { get; set; } = string.Empty;

    public Punch BestPunch { get; set; }

    public Punch Vulnerability { get; set; }

    public void SetName(string prompt)
    {
        Console.WriteLine(prompt);
        string? name;
        do
        {
            name = Console.ReadLine();
        } while (string.IsNullOrWhiteSpace(name));
        Name = name;
    }

    public int DamageTaken { get; set; }

    public void ResetForNewRound() => DamageTaken = 0;

    public void RecordWin() => _wins += 1;

    public bool IsWinner => _wins >= 2;

    public override string ToString() => Name;
}

```

这段代码的作用是创建一个名为Opponent的类，该类继承自Boxer类，代表拳击比赛中的对手。在这个类中，有一个名为SetRandomPunches的静态方法，用于设置随机 punch(拳击)技能的攻击力。

该方法的实现是通过使用GameUtils类中的Roll方法来生成一个4面骰子，其中B1表示最佳拳击，D1表示脆弱性。在循环中，使用BestPunch和Vulnerability变量更新随机生成的攻击力和脆弱性。

这里使用Do-while循环语句，其中的判断条件是BestPunch == Vulnerability，表示在每次循环中，首先随机生成一个攻击力和脆弱性，然后比较它们是否相等。如果相等，则循环将继续进行下一次循环，直到生成不同的值为止。这样，就可以在每次循环中更新随机生成的攻击力和脆弱性，使得对手的攻击力和防御力更加随机化。


```
public class Opponent : Boxer
{
    public void SetRandomPunches()
    {
        do
        {
            BestPunch = (Punch) GameUtils.Roll(4); // B1
            Vulnerability = (Punch) GameUtils.Roll(4); // D1
        } while (BestPunch == Vulnerability);
    }
}

```

# `15_Boxing/csharp/OpponentAttackStrategy.cs`

This is a code snippet for a tabletop game where players are taking turns playing different moves. It appears to be a Java programming language implementation.

The game is featuring a couple of different moves: uppercut and jab. Uppercut appears to be a high-risk attack that can inflict damage on an opponent if they are not prepared, while jab is a low-risk attack that can cause damage if it misses.

It appears that the game is also featuring a vulnerability called punch. punch appears to be a critical hit that can cause a significant amount of damage if the attacker is a certain type of character.

It looks like there are different rules for each move and that damage is scored based on the outcome of the move. It also appears that there are certain thresholds for certain moves to be considered valid, such as a threshold for the user to successfully block an attack.

Overall, it looks like this is a complex game with many different mechanics and rules, and it would be challenging to fully understand without seeing the code for yourself.


```
﻿using static Boxing.GameUtils;
using static System.Console;

namespace Boxing;

public class OpponentAttackStrategy : AttackStrategy
{
    private readonly Opponent _opponent;

    public OpponentAttackStrategy(Opponent opponent, Boxer player,  Action notifyGameEnded, Stack<Action> work) : base(player, work, notifyGameEnded)
    {
        _opponent = opponent;
    }

    protected override AttackPunch GetPunch()
    {
        var punch = (Punch)Roll(4);
        return new AttackPunch(punch, punch == _opponent.BestPunch);
    }

    protected override void FullSwing() // 720
    {
        Write($"{_opponent}  TAKES A FULL SWING AND");
        if (Other.Vulnerability == Punch.FullSwing)
        {
            ScoreFullSwing();
        }
        else
        {
            if (RollSatisfies(60, x => x < 30))
            {
                WriteLine(" IT'S BLOCKED!");
            }
            else
            {
                ScoreFullSwing();
            }
        }

        void ScoreFullSwing()
        {
            WriteLine(" POW!!!!! HE HITS HIM RIGHT IN THE FACE!");
            if (Other.DamageTaken > KnockoutDamageThreshold)
            {
                Work.Push(RegisterOtherKnockedOut);
            }
            Other.DamageTaken += 15;
        }
    }

    protected override void Hook() // 810
    {
        Write($"{_opponent} GETS {Other} IN THE JAW (OUCH!)");
        Other.DamageTaken += 7;
        WriteLine("....AND AGAIN!");
        Other.DamageTaken += 5;
        if (Other.DamageTaken > KnockoutDamageThreshold)
        {
            Work.Push(RegisterOtherKnockedOut);
        }
    }

    protected override void Uppercut() // 860
    {
        Write($"{Other} IS ATTACKED BY AN UPPERCUT (OH,OH)...");
        if (Other.Vulnerability == Punch.Uppercut)
        {
            ScoreUppercut();
        }
        else
        {
            if (RollSatisfies(200, x => x > 75))
            {
                WriteLine($" BLOCKS AND HITS {_opponent} WITH A HOOK.");
                _opponent.DamageTaken += 5;
            }
            else
            {
                ScoreUppercut();
            }
        }

        void ScoreUppercut()
        {
            WriteLine($"AND {_opponent} CONNECTS...");
            Other.DamageTaken += 8;
        }
    }

    protected override void Jab() // 640
    {
        Write($"{_opponent}  JABS AND ");
        if (Other.Vulnerability == Punch.Jab)
        {
            ScoreJab();
        }
        else
        {
            if (RollSatisfies(7, x => x > 4))
            {
                WriteLine("BLOOD SPILLS !!!");
                ScoreJab();
            }
            else
            {
                WriteLine("IT'S BLOCKED!");
            }
        }

        void ScoreJab() => Other.DamageTaken += 5;
    }

    private void RegisterOtherKnockedOut()
        => RegisterKnockout($"{Other} IS KNOCKED COLD AND {_opponent} IS THE WINNER AND CHAMP!");
}

```

# `15_Boxing/csharp/PlayerAttackStrategy.cs`

This appears to be a character-based game where one player is "Punch" and the other is "Other," and the objective is for the player Punch to hit the player Other and cause damage. The game has various rules and scoring systems for hits, blocks, and uppercuts.

There are also various abilities for each character, such as the ability to score a higher cut if a certain vulnerability is hit, the ability to block a certain number of hits, and the ability to score a hook if the opponent is within certain ranges.

The game also has a score system, where the player with the highest score at the end of each round is the winner.


```
﻿using static Boxing.GameUtils;
using static System.Console;
namespace Boxing;

public class PlayerAttackStrategy : AttackStrategy
{
    private readonly Boxer _player;

    public PlayerAttackStrategy(Boxer player, Opponent opponent, Action notifyGameEnded, Stack<Action> work)
        : base(opponent, work, notifyGameEnded) => _player = player;

    protected override AttackPunch GetPunch()
    {
        var punch = GameUtils.GetPunch($"{_player}'S PUNCH");
        return new AttackPunch(punch, punch == _player.BestPunch);
    }

    protected override void FullSwing() // 340
    {
        Write($"{_player} SWINGS AND ");
        if (Other.Vulnerability == Punch.FullSwing)
        {
            ScoreFullSwing();
        }
        else
        {
            if (RollSatisfies(30, x => x < 10))
            {
                ScoreFullSwing();
            }
            else
            {
                WriteLine("HE MISSES");
            }
        }

        void ScoreFullSwing()
        {
            WriteLine("HE CONNECTS!");
            if (Other.DamageTaken > KnockoutDamageThreshold)
            {
                Work.Push(() => RegisterKnockout($"{Other} IS KNOCKED COLD AND {_player} IS THE WINNER AND CHAMP!"));
            }
            Other.DamageTaken += 15;
        }
    }

    protected override void Uppercut() // 520
    {
        Write($"{_player} TRIES AN UPPERCUT ");
        if (Other.Vulnerability == Punch.Uppercut)
        {
            ScoreUpperCut();
        }
        else
        {
            if (RollSatisfies(100, x => x < 51))
            {
                ScoreUpperCut();
            }
            else
            {
                WriteLine("AND IT'S BLOCKED (LUCKY BLOCK!)");
            }
        }

        void ScoreUpperCut()
        {
            WriteLine("AND HE CONNECTS!");
            Other.DamageTaken += 4;
        }
    }

    protected override void Hook() // 450
    {
        Write($"{_player} GIVES THE HOOK... ");
        if (Other.Vulnerability == Punch.Hook)
        {
            ScoreHookOnOpponent();
        }
        else
        {
            if (RollSatisfies(2, x => x == 1))
            {
                WriteLine("BUT IT'S BLOCKED!!!!!!!!!!!!!");
            }
            else
            {
                ScoreHookOnOpponent();
            }
        }

        void ScoreHookOnOpponent()
        {
            WriteLine("CONNECTS...");
            Other.DamageTaken += 7;
        }
    }

    protected override void Jab()
    {
        WriteLine($"{_player} JABS AT {Other}'S HEAD");
        if (Other.Vulnerability == Punch.Jab)
        {
            ScoreJabOnOpponent();
        }
        else
        {
            if (RollSatisfies(8, x => x < 4))
            {
                WriteLine("IT'S BLOCKED.");
            }
            else
            {
                ScoreJabOnOpponent();
            }
        }

        void ScoreJabOnOpponent() => Other.DamageTaken += 3;
    }
}

```

# `15_Boxing/csharp/Program.cs`

这段代码使用了Boxing库来实现输出信息的功能。

首先，使用new string('\t', 33) + "BOXING"创建了一个输出字符串，其中33表示新宽度格式下的33字符，该字符串将输出"BOXING"。

其次，使用new string('\t', 15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"创建了一个输出字符串，其中15表示新宽度格式下的15字符，该字符串将输出"CREATIVE COMPUTING"。

接着，使用WriteLine方法输出变量opponent的名称"WHAT IS YOUR OPPONENT'S NAME"和变量player的名称"INPUT YOUR MAN'S NAME"。

使用Environment.NewLine方法输出变量Environment的名称"(3 ROUNDS -- 2 OUT OF 3 WINS)".

然后，使用Boxing.GameUtils.PrintPunchDescription方法调用PrintPunchDescription方法输出对手的攻击描述。

最后，使用GetPunch方法获取对手的最佳攻击，并将其输出。


```
﻿using Boxing;
using static Boxing.GameUtils;
using static System.Console;

WriteLine(new string('\t', 33) + "BOXING");
WriteLine(new string('\t', 15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
WriteLine("{0}{0}{0}BOXING OLYMPIC STYLE (3 ROUNDS -- 2 OUT OF 3 WINS){0}", Environment.NewLine);

var opponent = new Opponent();
opponent.SetName("WHAT IS YOUR OPPONENT'S NAME"); // J$
var player = new Boxer();
player.SetName("INPUT YOUR MAN'S NAME"); // L$

PrintPunchDescription();
player.BestPunch = GetPunch("WHAT IS YOUR MANS BEST"); // B
```

这段代码的主要目的是让玩家与对手进行拳击游戏。玩家通过输入 "WHAT IS HIS VULNERABILITY" 来选择一个攻击，而对手则随机生成反击。游戏开始时，玩家和对手进行一分钟的拳击，然后循环进行直到游戏结束。

在循环中，玩家会选择一个攻击回合，然后编写一个名为 round 的环形进行游戏。每次游戏开始时，round 会向对手发送一个攻击，并等待对手做出反应。如果 round 中的游戏结束，循环将会退出，这样就可以让玩家继续选择攻击，直到他们选择了一个攻击或者对手生成了超过两个攻击。

在这段代码中，还输出了一些信息来描述比赛情况。在每次游戏结束后，程序会输出 "{0}{0}AND NOW GOODBYE FROM THE OLYMPIC ARENA." 来表示玩家和对手的比赛结果。


```
player.Vulnerability = GetPunch("WHAT IS HIS VULNERABILITY"); // D
opponent.SetRandomPunches();
WriteLine($"{opponent}'S ADVANTAGE IS {opponent.BestPunch.ToFriendlyString()} AND VULNERABILITY IS SECRET.");


for (var i = 1; i <= 3; i ++) // R
{
    var round = new Round(player, opponent, i);
    round.Start();
    round.CheckOpponentWin();
    round.CheckPlayerWin();
    if (round.GameEnded) break;
}
WriteLine("{0}{0}AND NOW GOODBYE FROM THE OLYMPIC ARENA.{0}", Environment.NewLine);

```

# `15_Boxing/csharp/Punch.cs`

这段代码定义了一个枚举类型Punch，包含了四种不同的拳击方式，数字值从1到4分别对应着FullSwing、Hook、Uppercut和Jab。

枚举类型是一种数据类型，它可以用来表示具有特定含义的一组数字或字符。在这里，Punch枚举类型用数字来表示不同的拳击方式。

在实际应用中，Punch枚举类型可以用来定义一个序列或一组可重复使用的代码块，这样就可以更方便地引用和扩展这个功能。


```
namespace Boxing;

public enum Punch
{
    FullSwing = 1,
    Hook = 2,
    Uppercut = 3,
    Jab = 4
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `15_Boxing/csharp/Round.cs`

This is a class written in C# that defines a game between two players. The game has a default round duration of 3 minutes. The game has a main method which initializes the game and a method for checking if the player or opponent has won.

The main method first checks if the game is already over and sets the GameEnded variable to true if it is. It then waits for 300 milliseconds and then checks if the player or opponent has won using the CheckRoundWinner method. If the player has won, it prints a message and sets the GameEnded variable to true. If the opponent has won, it also prints a message and sets the GameEnded variable to true.

The CheckRoundWinner method checks if the opponent took damage in a given round and then either the player or opponent wins the round. It does this by calling the DecisionWhoAttacks method which is responsible for deciding which player attacks in a given round.

The DecisionWhoAttacks method rolls a random number between 1 and 10 and then calls either the Attack method of the _opponentAttackStrategy object or the Attack method of the _playerAttackStrategy object, depending on whether the roll is greater than or equal to 5.

The Attack method of each strategy object is called to determine which player attacks in a given round.


```
﻿namespace Boxing;

class Round
{

    private readonly Boxer _player;
    private readonly Boxer _opponent;
    private readonly int _round;
    private Stack<Action> _work = new();
    private readonly PlayerAttackStrategy _playerAttackStrategy;
    private readonly OpponentAttackStrategy _opponentAttackStrategy;

    public bool GameEnded { get; private set; }

    public Round(Boxer player, Opponent opponent, int round)
    {
        _player = player;
        _opponent = opponent;
        _round = round;
        _work.Push(ResetPlayers);
        _work.Push(CheckOpponentWin);
        _work.Push(CheckPlayerWin);

        void NotifyGameEnded() => GameEnded = true;
        _playerAttackStrategy = new PlayerAttackStrategy(player, opponent, NotifyGameEnded, _work);
        _opponentAttackStrategy = new OpponentAttackStrategy(opponent, player, NotifyGameEnded, _work);
    }

    public void Start()
    {
        while (_work.Count > 0)
        {
            var action = _work.Pop();
            // This delay does not exist in the VB code but it makes a bit easier to follow the game.
            // I assume the computers at the time were slow enough
            // so that they did not need this delay...
            Thread.Sleep(300);
            action();
        }
    }

    public void CheckOpponentWin()
    {
        if (_opponent.IsWinner)
        {
            Console.WriteLine($"{_opponent} WINS (NICE GOING, {_opponent}).");
            GameEnded = true;
        }
    }

    public void CheckPlayerWin()
    {
        if (_player.IsWinner)
        {
            Console.WriteLine($"{_player}  AMAZINGLY WINS!!");
            GameEnded = true;
        }
    }

    private void ResetPlayers()
    {
        _player.ResetForNewRound();
        _opponent.ResetForNewRound();
        _work.Push(RoundBegins);
    }

    private void RoundBegins()
    {
        Console.WriteLine();
        Console.WriteLine($"ROUND {_round} BEGINS...");
        _work.Push(CheckRoundWinner);
        for (var i = 0; i < 7; i++)
        {
            _work.Push(DecideWhoAttacks);
        }
    }

    private void CheckRoundWinner()
    {
        if (_opponent.DamageTaken > _player.DamageTaken)
        {
            Console.WriteLine($"{_player} WINS ROUND {_round}");
            _player.RecordWin();
        }
        else
        {
            Console.WriteLine($"{_opponent} WINS ROUND {_round}");
            _opponent.RecordWin();
        }
    }

    private void DecideWhoAttacks()
    {
        _work.Push( GameUtils.RollSatisfies(10, x => x > 5) ? _opponentAttackStrategy.Attack : _playerAttackStrategy.Attack );
    }
}

```

# `15_Boxing/csharp/Utils.cs`

这段代码是一个用于描述不同拳击招式的类，主要实现了以下功能：

1. `PrintPunchDescription`方法用于打印出不同拳击招式的描述字符串。

2. `GetPunch`方法接受一个字符串参数，表示要描述的拳击招式，然后程序会枚举所有可用的拳击招式，直到找到与传入的参数匹配的或者没有匹配的拳击招式。

3. `Roll`方法接受一个上下文参数，然后使用`Rnd`生成器生成一个随机整数，将其作为参数传递给`upperLimit`方法，然后返回生成的随机整数。

4. `RollSatisfies`方法接受一个上下文参数和一个判断条件，然后使用`Roll`方法生成随机整数，并将其与给定的判断条件比较，如果返回true，则返回生成的随机整数的字符串表示法。

5. `ToFriendlyString`方法接受一个`Punch`对象，返回该拳击招式的友好字符串表示法，根据不同的拳击招式类型，该方法返回不同的字符串。


```
﻿namespace Boxing;
public static class GameUtils
{
    private static readonly Random Rnd = new((int) DateTime.UtcNow.Ticks);
    public static void PrintPunchDescription() =>
        Console.WriteLine($"DIFFERENT PUNCHES ARE: {PunchDesc(Punch.FullSwing)}; {PunchDesc(Punch.Hook)}; {PunchDesc(Punch.Uppercut)}; {PunchDesc(Punch.Jab)}.");

    private static string PunchDesc(Punch punch) => $"({(int)punch}) {punch.ToFriendlyString()}";

    public static Punch GetPunch(string prompt)
    {
        Console.WriteLine(prompt);
        Punch result;
        while (!Enum.TryParse(Console.ReadLine(), out result) || !Enum.IsDefined(typeof(Punch), result))
        {
            PrintPunchDescription();
        }
        return result;
    }

    public static Func<int, int> Roll { get;  } =  upperLimit => (int) (upperLimit * Rnd.NextSingle()) + 1;

    public static bool RollSatisfies(int upperLimit, Predicate<int> predicate) => predicate(Roll(upperLimit));

    public static string ToFriendlyString(this Punch punch)
        => punch switch
        {
            Punch.FullSwing => "FULL SWING",
            Punch.Hook => "HOOK",
            Punch.Uppercut => "UPPERCUT",
            Punch.Jab => "JAB",
            _ => throw new ArgumentOutOfRangeException(nameof(punch), punch, null)
        };

}

```