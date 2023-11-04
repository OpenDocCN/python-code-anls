# BasicComputerGames源码解析 40

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Football

Football is probably the most popular simulated sports game. I have seen some people play to elect to play computerized football in preference to watching a football game on television.

Two versions of football are presented. The first is somewhat “traditional” in that you, the player, are playing against the computer. You have a choice of seven offensive plays. On defense the computer seems to play a zone defence, but you have no choice of plays. The computer program presents the necessary rules as you play, and it is also the referee and determines penalties when an infraction is committed. FTBALL was written by John Kemeny at Dartmouth.

IN the second version of football, the computer referees a game played between two human players. Each player gets a list of twenty plays with a code value for each one. This list should be kept confidential from your opponent. The codes can be changes in data. All twenty plays are offensive; a defensive play is specified by defending against a type of offensive play. A defense is good for other similar types of plays, for example, a defense against a flare pass is very good against a screen pass but much less good against a half-back option.

This game was originally written by Raymond Miseyka of Butler, Pennsylvania.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=64)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=79)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `37_Football/javascript/football.js`

这段代码是一个Javascript脚本，它将文本输入框中的用户输入转换为BASIC代码并运行在浏览器中。

具体来说，这个脚本会将输入框中的文本内容（存储在变量input_str中）打印到页面上，并在页面上显示一个新的空行。

input()函数通过向输入框添加一个INPUT元素，获取用户输入的文本内容，并在页面上显示一个新的空行。它还监听INPUT元素的keydown事件，以便在用户按下回车键时获取输入的文本内容并打印到页面上。

input()函数的语法如下：
```css
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
                      return resolve(input_str);
                   });
           }
       }
   });
}
```


```
// FOOTBALL
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

这段代码定义了一个名为 `tab` 的函数，它接受一个参数 `space`，表示要输出多少个空格。函数内部，使用一个字符串变量 `str`，并使用 while 循环来循环输入 `space` 次字符。每次循环，将一个空格添加到 `str` 的开头，并将 `space` 的值减 1。当循环结束时，返回 `str` 中的所有字符，它们将组成一个空格字符串。

接下来，定义了一个包含多个数组的 `player_data` 变量。这些数组存储了玩家在游戏中的数据，包括他们的生命值和分数等。然后，定义了四个变量 `aa`、`ba`、`ca` 和 `ha`，它们都包含一个数组，但具体内容暂时没有定义。

最后，在函数内部创建了一个包含多个数组的 `ta` 变量，但同样没有具体的内容定义。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var player_data = [17,8,4,14,19,3,10,1,7,11,15,9,5,20,13,18,16,2,12,6,
                   20,2,17,5,8,18,12,11,1,4,19,14,10,7,9,15,6,13,16,3];
var aa = [];
var ba = [];
var ca = [];
var ha = [];
var ta = [];
```

这段代码定义了一个包含9个数字的数组wa、xa、ya和za，以及一个包含10个字符串的数组ms和一个包含11个字符串的数组da。同时，定义了一个字符串数组ps，用于存储投手技能的名称。

然后，定义了一个变量p和两个变量t，似乎没有具体的赋值。

最后，有一个函数field_headers，该函数没有具体的实现，只是通过打印了一些字符串，然后没有返回任何值。


```
var wa = [];
var xa = [];
var ya = [];
var za = [];
var ms = [];
var da = [];
var ps = [, "PITCHOUT","TRIPLE REVERSE","DRAW","QB SNEAK","END AROUND",
          "DOUBLE REVERSE","LEFT SWEEP","RIGHT SWEEP","OFF TACKLE",
          "WISHBONE OPTION","FLARE PASS","SCREEN PASS",
          "ROLL OUT OPTION","RIGHT CURL","LEFT CURL","WISHBONE OPTION",
          "SIDELINE PASS","HALF-BACK OPTION","RAZZLE-DAZZLE","BOMB!!!!"];
var p;
var t;

function field_headers()
{
    print("TEAM 1 [0   10   20   30   40   50   60   70   80   90");
    print("   100] TEAM 2\n");
    print("\n");
}

```

这三个人人函数分别是：分离字符串、展示气球得分和展示比赛得分。

1. separator()函数的作用是创建一个空字符串str，然后使用for循环从1到72遍历，每次循环在字符串中添加"+"号，最后输出str。

2. show_ball()函数的作用是在tab()函数的基础上，展示某支队伍的得分。首先计算da[t]+5+p/2的值，然后使用tab()函数将结果居中显示，同时输出该得分与球场的单位数。

3. show_scores()函数的作用是展示比赛得分。首先打印出各个队伍得分，然后根据公式ha[t]>=e来判断比赛是否结束。如果比赛结束，则打印"TEAM "+t+""，并返回true；否则返回false。


```
function separator()
{
    str = "";
    for (x = 1; x <= 72; x++)
        str += "+";
    print(str + "\n");
}

function show_ball()
{
    print(tab(da[t] + 5 + p / 2) + ms[t] + "\n");
    field_headers();
}

function show_scores()
{
    print("\n");
    print("TEAM 1 SCORE IS " + ha[1] + "\n");
    print("TEAM 2 SCORE IS " + ha[2] + "\n");
    print("\n");
    if (ha[t] >= e) {
        print("TEAM " + t + " WINS*******************");
        return true;
    }
    return false;
}

```



这两函数的主要作用是向用户输出有关足球比赛的数据，包括比赛中的得分、射门和接触球等。

第一个函数 `loss_posession()` 的作用是告诉用户比赛结果，胜利的球队名称和失败球队的名称。

第二个函数 `touchdown()` 的作用是告诉用户哪个球队在何时何地踢入了一个进球，包括进球的球队名称，得分，以及是否是有效进球。


```
function loss_posession() {
    print("\n");
    print("** LOSS OF POSSESSION FROM TEAM " + t + " TO TEAM " + ta[t] + "\n");
    print("\n");
    separator();
    print("\n");
    t = ta[t];
}

function touchdown() {
    print("\n");
    print("TOUCHDOWN BY TEAM " + t + " *********************YEA TEAM\n");
    q = 7;
    g = Math.random();
    if (g <= 0.1) {
        q = 6;
        print("EXTRA POINT NO GOOD\n");
    } else {
        print("EXTRA POINT GOOD\n");
    }
    ha[t] = ha[t] + q;
}

```

This is a program that simulates a football game. It allows you to simulate different situations
and outcomes, including
scheduling decisions, player movements, and


```
// Main program
async function main()
{
    print(tab(32) + "FOOTBALL\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("PRESENTING N.F.U. FOOTBALL (NO FORTRAN USED)\n");
    print("\n");
    print("\n");
    while (1) {
        print("DO YOU WANT INSTRUCTIONS");
        str = await input();
        if (str == "YES" || str == "NO")
            break;
    }
    if (str == "YES") {
        print("THIS IS A FOOTBALL GAME FOR TWO TEAMS IN WHICH PLAYERS MUST\n");
        print("PREPARE A TAPE WITH A DATA STATEMENT (1770 FOR TEAM 1,\n");
        print( "1780 FOR TEAM 2) IN WHICH EACH TEAM SCRAMBLES NOS. 1-20\n");
        print("THESE NUMBERS ARE THEN ASSIGNED TO TWENTY GIVEN PLAYS.\n");
        print("A LIST OF NOS. AND THEIR PLAYS IS PROVIDED WITH\n");
        print("BOTH TEAMS HAVING THE SAME PLAYS. THE MORE SIMILAR THE\n");
        print("PLAYS THE LESS YARDAGE GAINED.  SCORES ARE GIVEN\n");
        print("WHENEVER SCORES ARE MADE. SCORES MAY ALSO BE OBTAINED\n");
        print("BY INPUTTING 99,99 FOR PLAY NOS. TO PUNT OR ATTEMPT A\n");
        print("FIELD GOAL, INPUT 77,77 FOR PLAY NUMBERS. QUESTIONS WILL BE\n");
        print("ASKED THEN. ON 4TH DOWN, YOU WILL ALSO BE ASKED WHETHER\n");
        print("YOU WANT TO PUNT OR ATTEMPT A FIELD GOAL. IF THE ANSWER TO\n");
        print("BOTH QUESTIONS IS NO IT WILL BE ASSUMED YOU WANT TO\n");
        print("TRY AND GAIN YARDAGE. ANSWER ALL QUESTIONS YES OR NO.\n");
        print("THE GAME IS PLAYED UNTIL PLAYERS TERMINATE (CONTROL-C).\n");
        print("PLEASE PREPARE A TAPE AND RUN.\n");
    }
    print("\n");
    print("PLEASE INPUT SCORE LIMIT ON GAME");
    e = parseInt(await input());
    for (i = 1; i <= 40; i++) {
        if (i <= 20) {
            aa[player_data[i - 1]] = i;
        } else {
            ba[player_data[i - 1]] = i - 20;
        }
        ca[i] = player_data[i - 1];
    }
    l = 0;
    t = 1;
    do {
        print("TEAM " + t + " PLAY CHART\n");
        print("NO.      PLAY\n");
        for (i = 1; i <= 20; i++) {
            str = "" + ca[i + l];
            while (str.length < 6)
                str += " ";
            str += ps[i];
            print(str + "\n");
        }
        l += 20;
        t = 2;
        print("\n");
        print("TEAR OFF HERE----------------------------------------------\n");
        for (x = 1; x <= 11; x++)
            print("\n");
    } while (l == 20) ;
    da[1] = 0;
    da[2] = 3;
    ms[1] = "--->";
    ms[2] = "<---";
    ha[1] = 0;
    ha[2] = 0;
    ta[1] = 2;
    ta[2] = 1;
    wa[1] = -1;
    wa[2] = 1;
    xa[1] = 100;
    xa[2] = 0;
    ya[1] = 1;
    ya[2] = -1;
    za[1] = 0;
    za[2] = 100;
    p = 0;
    field_headers();
    print("TEAM 1 DEFEND 0 YD GOAL -- TEAM 2 DEFENDS 100 YD GOAL.\n");
    t = Math.floor(2 * Math.random() + 1);
    print("\n");
    print("THE COIN IS FLIPPED\n");
    routine = 1;
    while (1) {
        if (routine <= 1) {
            p = xa[t] - ya[t] * 40;
            separator();
            print("\n");
            print("TEAM " + t + " RECEIVES KICK-OFF\n");
            k = Math.floor(26 * Math.random() + 40);
        }
        if (routine <= 2) {
            p = p - ya[t] * k;
        }
        if (routine <= 3) {
            if (wa[t] * p >= za[t] + 10) {
                print("\n");
                print("BALL WENT OUT OF ENDZONE --AUTOMATIC TOUCHBACK--\n");
                p = za[t] - wa[t] * 20;
                if (routine <= 4)
                    routine = 5;
            } else {
                print("BALL WENT " + k + " YARDS.  NOW ON " + p + "\n");
                show_ball();
            }
        }
        if (routine <= 4) {
            while (1) {
                print("TEAM " + t + " DO YOU WANT TO RUNBACK");
                str = await input();
                if (str == "YES" || str == "NO")
                    break;
            }
            if (str == "YES") {
                k = Math.floor(9 * Math.random() + 1);
                r = Math.floor(((xa[t] - ya[t] * p + 25) * Math.random() - 15) / k);
                p = p - wa[t] * r;
                print("\n");
                print("RUNBACK TEAM " + t + " " + r + " YARDS\n");
                g = Math.random();
                if (g < 0.25) {
                    loss_posession();
                    routine = 4;
                    continue;
                } else if (ya[t] * p >= xa[t]) {
                    touchdown();
                    if (show_scores())
                        return;
                    t = ta[t];
                    routine = 1;
                    continue;
                } else if (wa[t] * p >= za[t]) {
                    print("\n");
                    print("SAFETY AGAINST TEAM " + t + " **********************OH-OH\n");
                    ha[ta[t]] = ha[ta[t]] + 2;
                    if (show_scores())
                        return;
                    print("TEAM " + t + " DO YOU WANT TO PUNT INSTEAD OF A KICKOFF");
                    str = await input();
                    p = za[t] - wa[t] * 20;
                    if (str == "YES") {
                        print("\n");
                        print("TEAM " + t + " WILL PUNT\n");
                        g = Math.random();
                        if (g < 0.25) {
                            loss_posession();
                            routine = 4;
                            continue;
                        }
                        print("\n");
                        separator();
                        k = Math.floor(25 * Math.random() + 35);
                        t = ta[t];
                        routine = 2;
                        continue;
                    }
                    touchdown();
                    if (show_scores())
                        return;
                    t = ta[t];
                    routine = 1;
                    continue;
                } else {
                    routine = 5;
                    continue;
                }
            } else if (str == "NO") {
                if (wa[t] * p >= za[t])
                    p = za[t] - wa[t] * 20;
            }
        }
        if (routine <= 5) {
            d = 1;
            s = p;
        }
        if (routine <= 6) {
            str = "";
            for (i = 1; i <= 72; i++)
                str += "=";
            print(str + "\n");
            print("TEAM " + t + " DOWN " + d + " ON " + p + "\n");
            if (d == 1) {
                if (ya[t] * (p + ya[t] * 10) >= xa[t])
                    c = 8;
                else
                    c = 4;
            }
            if (c != 8) {
                print(tab(27) + (10 - (ya[t] * p - ya[t] * s)) + " YARDS TO 1ST DOWN\n");
            } else {
                print(tab(27) + (xa[t] - ya[t] * p) + " YARDS\n");
            }
            show_ball();
            if (d == 4)
                routine = 8;
        }
        if (routine <= 7) {
            u = Math.floor(3 * Math.random() - 1);
            while (1) {
                print("INPUT OFFENSIVE PLAY, DEFENSIVE PLAY");
                str = await input();
                if (t == 1) {
                    p1 = parseInt(str);
                    p2 = parseInt(str.substr(str.indexOf(",") + 1));
                } else {
                    p2 = parseInt(str);
                    p1 = parseInt(str.substr(str.indexOf(",") + 1));
                }
                if (p1 == 99) {
                    if (show_scores())
                        return;
                    if (p1 == 99)
                        continue;
                }
                if (p1 < 1 || p1 > 20 || p2 < 1 || p2 > 20) {
                    print("ILLEGAL PLAY NUMBER, CHECK AND\n");
                    continue;
                }
                break;
            }
        }
        if (d == 4 || p1 == 77) {
            while (1) {
                print("DOES TEAM " + t + " WANT TO PUNT");
                str = await input();
                if (str == "YES" || str == "NO")
                    break;
            }
            if (str == "YES") {
                print("\n");
                print("TEAM " + t + " WILL PUNT\n");
                g = Math.random();
                if (g < 0.25) {
                    loss_posession();
                    routine = 4;
                    continue;
                }
                print("\n");
                separator();
                k = Math.floor(25 * Math.random() + 35);
                t = ta[t];
                routine = 2;
                continue;
            }
            while (1) {
                print("DOES TEAM " + t + " WANT TO ATTEMPT A FIELD GOAL");
                str = await input();
                if (str == "YES" || str == "NO")
                    break;
            }
            if (str == "YES") {
                print("\n");
                print("TEAM " + t + " WILL ATTEMPT A FIELD GOAL\n");
                g = Math.random();
                if (g < 0.025) {
                    loss_posession();
                    routine = 4;
                    continue;
                } else {
                    f = Math.floor(35 * Math.random() + 20);
                    print("\n");
                    print("KICK IS " + f + " YARDS LONG\n");
                    p = p - wa[t] * f;
                    g = Math.random();
                    if (g < 0.35) {
                        print("BALL WENT WIDE\n");
                    } else if (ya[t] * p >= xa[t]) {
                        print("FIELD GOLD GOOD FOR TEAM " + t + " *********************YEA");
                        q = 3;
                        ha[t] = ha[t] + q;
                        if (show_scores())
                            return;
                        t = ta[t];
                        routine = 1;
                        continue;
                    }
                    print("FIELD GOAL UNSUCCESFUL TEAM " + t + "-----------------TOO BAD\n");
                    print("\n");
                    separator();
                    if (ya[t] * p < xa[t] + 10) {
                        print("\n");
                        print("BALL NOW ON " + p + "\n");
                        t = ta[t];
                        show_ball();
                        routine = 4;
                        continue;
                    } else {
                        t = ta[t];
                        routine = 3;
                        continue;
                    }
                }
            } else {
                routine = 7;
                continue;
            }
        }
        y = Math.floor(Math.abs(aa[p1] - ba[p2]) / 19 * ((xa[t] - ya[t] * p + 25) * Math.random() - 15));
        print("\n");
        if (t == 1 && aa[p1] < 11 || t == 2 && ba[p2] < 11) {
            print("THE BALL WAS RUN\n");
        } else if (u == 0) {
            print("PASS INCOMPLETE TEAM " + t + "\n");
            y = 0;
        } else {
            g = Math.random();
            if (g <= 0.025 && y > 2) {
                print("PASS COMPLETED\n");
            } else {
                print("QUARTERBACK SCRAMBLED\n");
            }
        }
        p = p - wa[t] * y;
        print("\n");
        print("NET YARDS GAINED ON DOWN " + d + " ARE " + y + "\n");

        g = Math.random();
        if (g <= 0.025) {
            loss_posession();
            routine = 4;
            continue;
        } else if (ya[t] * p >= xa[t]) {
            touchdown();
            if (show_scores())
                return;
            t = ta[t];
            routine = 1;
            continue;
        } else if (wa[t] * p >= za[t]) {
            print("\n");
            print("SAFETY AGAINST TEAM " + t + " **********************OH-OH\n");
            ha[ta[t]] = ha[ta[t]] + 2;
            if (show_scores())
                return;
            print("TEAM " + t + " DO YOU WANT TO PUNT INSTEAD OF A KICKOFF");
            str = await input();
            p = za[t] - wa[t] * 20;
            if (str == "YES") {
                print("\n");
                print("TEAM " + t + " WILL PUNT\n");
                g = Math.random();
                if (g < 0.25) {
                    loss_posession();
                    routine = 4;
                    continue;
                }
                print("\n");
                separator();
                k = Math.floor(25 * Math.random() + 35);
                t = ta[t];
                routine = 2;
                continue;
            }
            touchdown();
            if (show_scores())
                return;
            t = ta[t];
            routine = 1;
        } else if (ya[t] * p - ya[t] * s >= 10) {
            routine = 5;
        } else {
            d++;
            if (d != 5) {
                routine = 6;
            } else {
                print("\n");
                print("CONVERSION UNSUCCESSFUL TEAM " + t + "\n");
                t = ta[t];
                print("\n");
                separator();
                routine = 5;
            }
        }
    }
}

```

这道题目没有提供代码，因此无法解释代码的作用。一般来说，在编程中， `main()` 函数是程序的入口点，也是程序的控制中心。在 `main()` 函数中，程序会执行一系列的指令，这些指令可能会包括用户输入的数据、文件读写操作等。程序运行的结果，也可能会与用户的输入有关。


```
main();

```

# `37_Football/javascript/ftball.js`

这段代码是一个 JavaScript 函数，它的作用是向一个网页上的一个 div 元素（id 为 "output"）添加文本内容，并允许用户输入字符。

具体来说，它由两个函数组成：`print` 和 `input`。

`print` 函数的作用是接收一个字符串（比如从用户输入中获得），将其添加到目标 div 元素中的文本内容中。它的实现是通过创建一个文本节点，将其添加到目标元素中，然后设置文本内容、类型、长度等属性，最后绑定一个 `keyup` 事件以便监听用户输入的事件。当用户输入字符时，函数会将输入的字符串添加到 `output` 元素中的文本内容中，并输出该字符串。当用户输入回空格时，函数会将 `output` 元素中的文本内容清除，并输出一个换行符，使得输入的文本内容不会显示在页面上。

`input` 函数的作用是获取用户输入的字符串，并将其作为参数传递给 `print` 函数。它创建了一个带输入字段的 INPUT 元素，然后向该元素添加事件监听器，以便监听用户输入的事件。当用户输入字符时，函数会将其存储在 `input_str` 变量中，然后将其作为参数传递给 `print` 函数。当用户输入回空格时，函数会将 `input_str` 中的字符串作为参数传递给 `print` 函数，并输出一个换行符，使得输入的文本内容不会显示在页面上。


```
// FTBALL
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



这段代码定义了一个名为 `tab` 的函数，它会将传入的参数 `space` 中的值打印出来，直到 `space` 变量为 0 时停止。

在函数内部，首先定义了一个字符串变量 `str`，并使用一个 while 循环来打印字符串中的空格，直到 `space` 变量减少到 0。

然后，定义了一个包含足球比赛中的常见术语的数组 `ls`，以及一个包含字符串键值的数组 `os` 和 `sa`。

接着，通过 `eval` 函数将定义好的术语字符串赋值给 `ls.join(", ")` 表达式，并将结果保存回给 `ls` 数组。

最后，通过 `console.log` 函数将打印好的字符串输出，同时也将 `ls` 数组输出到终端。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var os = [];
var sa = [];
var ls = [, "KICK","RECEIVE"," YARD ","RUN BACK FOR ","BALL ON ",
          "YARD LINE"," SIMPLE RUN"," TRICKY RUN"," SHORT PASS",
          " LONG PASS","PUNT"," QUICK KICK "," PLACE KICK"," LOSS ",
          " NO GAIN","GAIN "," TOUCHDOWN "," TOUCHBACK ","SAFETY***",
          "JUNK"];
```

这段代码定义了三个变量p、x和x1，以及一个名为fnf的函数和一个名为fng的函数。fnf函数接受一个参数x，并返回1减去2乘以p。fng函数接受一个参数z，并包含两个嵌套的函数调用：一个名为x1的函数和一个名为x的函数。x1函数接受一个参数x，并返回x1减去x；x函数接受一个参数x，并返回x减去x1。

该代码还定义了一个名为show_score的函数，该函数没有具体的实现，只是简单地将三个变量的值打印出来。

最后，该代码没有输出任何函数的源代码，也没有对任何变量进行初始化。


```
var p;
var x;
var x1;

function fnf(x)
{
    return 1 - 2 * p;
}

function fng(z)
{
    return p * (x1 - x) + (1 - p) * (x - x1);
}

function show_score()
{
    print("\n");
    print("SCORE:  " + sa[0] + " TO " + sa[1] + "\n");
    print("\n");
    print("\n");
}

```

这两段代码定义了两个函数 `show_position` 和 `offensive_td`。

1. `show_position` 函数的作用是打印在某个位置上的文字，它基于 `x` 是否小于或等于 50 来决定打印哪个内容。具体地，它首先检查 `x` 是否小于或等于 50，如果是，就打印 `ls[5]`、`os[0]`、`x` 和 `ls[6]`。否则，就打印 `ls[5]`、`os[1]`、`(100 - x)` 和 `ls[6]`。代码中使用的是 `os` 数组，它包含了 `p` 变量，因此这个函数实际执行的是 `os[p]`。

2. `offensive_td` 函数的作用是在 MVP 游戏里进行进攻操作，它随机选择一个玩家，对选中的玩家进行攻击，然后显示攻击结果。攻击结果有三种可能：KICK IS GOOD、KICK IS OFF TO THE SIDE 和 INSIDE PUNCH。具体地，它首先打印选中玩家的名字，然后根据 `Math.random()` 的值来决定是否进行攻击操作。如果 `Math.random()` 的值在 0 到 0.8 之间，就执行攻击操作，并显示“KICK IS GOOD”。否则，就显示“KICK IS OFF TO THE SIDE”。每次攻击后，它都会显示选中玩家的名字并计算剩余的点数。然后，它将更新 `sa` 数组，以便在下一次攻击时使用。


```
function show_position()
{
    if (x <= 50) {
        print(ls[5] + os[0] + " " + x + " " + ls[6] + "\n");
    } else {
        print(ls[5] + os[1] + " " + (100 - x) + " " + ls[6] + "\n");
    }
}

function offensive_td()
{
    print(ls[17] + "***\n");
    if (Math.random() <= 0.8) {
        sa[p] = sa[p] + 7;
        print("KICK IS GOOD.\n");
    } else {
        print("KICK IS OFF TO THE SIDE\n");
        sa[p] = sa[p] + 6;
    }
    show_score();
    print(os[p] + " KICKS OFF\n");
    p = 1 - p;
}

```

This appears to be a SimPy game where the objective is for one team to score points and the other team to prevent them from scoring. The game has different rules for each of the four cases listed, which seems to be based on which team is playing offensive or defensive.

In the last case, the offensive team is trying to score a touchdown and the defensive team is trying to prevent a touchdown. If the offensive team scores a touchdown, the defensive team will try to prevent the offensive team from getting the ball again. If the defensive team prevents the offensive team from getting the ball again, the offensive team will have to start a new play.

It's not clear what the values for the different arguments of the `p * 4` are or what other game rules there are for the offensive and defensive teams.


```
// Main program
async function main()
{
    print(tab(33) + "FTBALL\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("THIS IS DARTMOUTH CHAMPIONSHIP FOOTBALL.\n");
    print("\n");
    print("YOU WILL QUARTERBACK DARTMOUTH. CALL PLAYS AS FOLLOWS:\n");
    print("1= SIMPLE RUN; 2= TRICKY RUN; 3= SHORT PASS;\n");
    print("4= LONG PASS; 5= PUNT; 6= QUICK KICK; 7= PLACE KICK.\n");
    print("\n");
    print("CHOOSE YOUR OPPONENT");
    os[1] = await input();
    os[0] = "DARMOUTH";
    print("\n");
    sa[0] = 0;
    sa[1] = 0;
    p = Math.floor(Math.random() * 2);
    print(os[p] + " WON THE TOSS\n");
    if (p != 0) {
        print(os[1] + " ELECTS TO RECEIVE.\n");
        print("\n");
    } else {
        print("DO YOU ELECT TO KICK OR RECEIVE");
        while (1) {
            str = await input();
            print("\n");
            if (str == ls[1] || str == ls[2])
                break;
            print("INCORRECT ANSWER.  PLEASE TYPE 'KICK' OR 'RECEIVE'");
        }
        e = (str == ls[1]) ? 1 : 2;
        if (e == 1)
            p = 1;
    }
    t = 0;
    start = 1;
    while (1) {
        if (start <= 1) {
            x = 40 + (1 - p) * 20;
        }
        if (start <= 2) {
            y = Math.floor(200 * Math.pow((Math.random() - 0.5), 3) + 55);
            print(" " + y + " " + ls[3] + " KICKOFF\n");
            x = x - fnf(1) * y;
            if (Math.abs(x - 50) >= 50) {
                print("TOUCHBACK FOR " + os[p] + ".\n");
                x = 20 + p * 60;
                start = 4;
            } else {
                start = 3;
            }
        }
        if (start <= 3) {
            y = Math.floor(50 * Math.pow(Math.random(), 2)) + (1 - p) * Math.floor(50 * Math.pow(Math.random(), 4));
            x = x + fnf(1) * y;
            if (Math.abs(x - 50) < 50) {
                print(" " + y + " " + ls[3] + " RUNBACK\n");
            } else {
                print(ls[4]);
                offensive_td();
                start = 1;
                continue;
            }
        }
        if (start <= 4) {
            // First down
            show_position();
        }
        if (start <= 5) {
            x1 = x;
            d = 1;
            print("\n");
            print("FIRST DOWN " + os[p] + "***\n");
            print("\n");
            print("\n");
        }
        // New play
        t++;
        if (t == 30) {
            if (Math.random() <= 1.3) {
                print("GAME DELAYED.  DOG ON FIELD.\n");
                print("\n");
            }
        }
        if (t >= 50 && Math.random() <= 0.2)
            break;
        if (p != 1) {
            // Opponent's play
            if (d <= 1) {
                z = Math.random() > 1 / 3 ? 1 : 3;
            } else if (d != 4) {
                if (10 + x - x1 < 5 || x < 5) {
                    z = Math.random() > 1 / 3 ? 1 : 3;
                } else if (x <= 10) {
                    a = Math.floor(2 * Math.random());
                    z = 2 + a;
                } else if (x <= x1 || d < 3 || x < 45) {
                    a = Math.floor(2 * Math.random());
                    z = 2 + a * 2;
                } else {
                    if (Math.random() > 1 / 4)
                        z = 4;
                    else
                        z = 6;
                }
            } else {
                if (x <= 30) {
                    z = 5;
                } else if (10 + x - x1 < 3 || x < 3) {
                    z = Math.random() > 1 / 3 ? 1 : 3;
                } else {
                    z = 7;
                }
            }
        } else {
            print("NEXT PLAY");
            while (1) {
                z = parseInt(await input());
                if (Math.abs(z - 4) <= 3)
                    break;
                print("ILLEGAL PLAY NUMBER, RETYPE");
            }
        }
        f = 0;
        print(ls[z + 6] + ".  ");
        r = Math.random() * (0.98 + fnf(1) * 0.02);
        r1 = Math.random();
        switch (z) {
            case 1: // Simple run
            case 2: // Tricky run
                if (z == 1) {
                    y = Math.floor(24 * Math.pow(r - 0.5, 3) + 3);
                    if (Math.random() >= 0.05) {
                        routine = 1;
                        break;
                    }
                } else {
                    y = Math.floor(20 * r - 5);
                    if (Math.random() > 0.1) {
                        routine = 1;
                        break;
                    }
                }
                f = -1;
                x3 = x;
                x = x + fnf(1) * y;
                if (Math.abs(x - 50) < 50) {
                    print("***  FUMBLE AFTER ");
                    routine = 2;
                    break;
                } else {
                    print("***  FUMBLE.\n");
                    routine = 4;
                    break;
                }
            case 3: // Short pass
            case 4: // Long pass
                if (z == 3) {
                    y = Math.floor(60 * Math.pow(r1 - 0.5, 3) + 10);
                } else {
                    y = Math.floor(160 * Math.pow((r1 - 0.5), 3) + 30);
                }
                if (z == 3 && r < 0.05 || z == 4 && r < 0.1) {
                    if (d != 4) {
                        print("INTERCEPTED.\n");
                        f = -1;
                        x = x + fnf(1) * y;
                        if (Math.abs(x - 50) >= 50) {
                            routine = 4;
                            break;
                        }
                        routine = 3;
                        break;
                    } else {
                        y = 0;
                        if (Math.random() < 0.3) {
                            print("BATTED DOWN.  ");
                        } else {
                            print("INCOMPLETE.  ");
                        }
                        routine = 1;
                        break;
                    }
                } else if (z == 4 && r < 0.3) {
                    print("PASSER TACKLED.  ");
                    y = -Math.floor(15 * r1 + 3);
                    routine = 1;
                    break;
                } else if (z == 3 && r < 0.15) {
                    print("PASSER TACLKED.  ");
                    y = -Math.floor(10 * r1);
                    routine = 1;
                    break;
                } else if (z == 3 && r < 0.55 || z == 4 && r < 0.75) {
                    y = 0;
                    if (Math.random() < 0.3) {
                        print("BATTED DOWN.  ");
                    } else {
                        print("INCOMPLETE.  ");
                    }
                    routine = 1;
                    break;
                } else {
                    print("COMPLETE.  ");
                    routine = 1;
                    break;
                }
            case 5:  // Punt
            case 6:  // Quick kick
                y = Math.floor(100 * Math.pow((r - 0.5), 3) + 35);
                if (d != 4)
                    y = Math.floor(y * 1.3);
                print(" " + y + " " + ls[3] + " PUNT\n");
                if (Math.abs(x + y * fnf(1) - 50) < 50 && d >= 4) {
                    y1 = Math.floor(Math.pow(r1, 2) * 20);
                    print(" " + y1 + " " + ls[3] + " RUN BACK\n");
                    y = y - y1;
                }
                f = -1;
                x = x + fnf(1) * y;
                if (Math.abs(x - 50) >= 50) {
                    routine = 4;
                    break;
                }
                routine = 3;
                break;
            case 7: // Place kick
                y = Math.floor(100 * Math.pow((r - 0.5), 3) + 35);
                if (r1 <= 0.15) {
                    print("KICK IS BLOCKED  ***\n");
                    x = x - 5 * fnf(1);
                    p = 1 - p;
                    start = 4;
                    continue;
                }
                x = x + fnf(1) * y;
                if (Math.abs(x - 50) >= 60) {
                    if (r1 <= 0.5) {
                        print("KICK IS OFF TO THE SIDE.\n");
                        print(ls[18] + "\n");
                        p = 1 - p;
                        x = 20 + p * 60;
                        start = 4;
                        continue;
                    } else {
                        print("FIELD GOAL ***\n");
                        sa[p] = sa[p] + 3;
                        show_score();
                        print(os[p] + " KICKS OFF\n");
                        p = 1 - p;
                        start = 1;
                        continue;
                    }
                } else {
                    print("KICK IS SHORT.\n");
                    if (Math.abs(x - 50) >= 50) {
                        // Touchback
                        print(ls[18] + "\n");
                        p = 1 - p;
                        x = 20 + p * 60;
                        start = 4;
                        continue;
                    }
                    p = 1 - p;
                    start = 3;
                    continue;
                }

        }
        // Gain or loss
        if (routine <= 1) {
            x3 = x;
            x = x + fnf(1) * y;
            if (Math.abs(x - 50) >= 50) {
                routine = 4;
            }
        }
        if (routine <= 2) {
            if (y != 0) {
                print(" " + Math.abs(y) + " " + ls[3]);
                if (y < 0)
                    yt = -1;
                else if (y > 0)
                    yt = 1;
                else
                    yt = 0;
                print(ls[15 + yt]);
                if (Math.abs(x3 - 50) <= 40 && Math.random() < 0.1) {
                    // Penalty
                    p3 = Math.floor(2 * Math.random());
                    print(os[p3] + " OFFSIDES -- PENALTY OF 5 YARDS.\n");
                    print("\n");
                    print("\n");
                    if (p3 != 0) {
                        print("DO YOU ACCEPT THE PENALTY");
                        while (1) {
                            str = await input();
                            if (str == "YES" || str == "NO")
                                break;
                            print("TYPE 'YES' OR 'NO'");
                        }
                        if (str == "YES") {
                            f = 0;
                            d = d - 1;
                            if (p != p3)
                                x = x3 + fnf(1) * 5;
                            else
                                x = x3 - fnf(1) * 5;
                        }
                    } else {
                        // Opponent's strategy on penalty
                        if ((p != 1 && (y <= 0 || f < 0 || fng(1) < 3 * d - 2))
                            || (p == 1 && ((y > 5 && f >= 0) || d < 4 || fng(1) >= 10))) {
                            print("PENALTY REFUSED.\n");
                        } else {
                            print("PENALTY ACCEPTED.\n");
                            f = 0;
                            d = d - 1;
                            if (p != p3)
                                x = x3 + fnf(1) * 5;
                            else
                                x = x3 - fnf(1) * 5;
                        }
                    }
                    routine = 3;
                }
            }
        }
        if (routine <= 3) {
            show_position();
            if (f != 0) {
                p = 1 - p;
                start = 5;
                continue;
            } else if (fng(1) >= 10) {
                start = 5;
                continue;
            } else if (d == 4) {
                p = 1 - p;
                start = 5;
                continue;
            } else {
                d++;
                print("DOWN: " + d + "     ");
                if ((x1 - 50) * fnf(1) >= 40) {
                    print("GOAL TO GO\n");
                } else {
                    print("YARDS TO GO: " + (10 - fng(1)) + "\n");
                }
                print("\n");
                print("\n");
                start = 6;
                continue;
            }
        }
        if (routine <= 4) {
            // Ball in end-zone
            e = (x >= 100) ? 1 : 0;
            switch (1 + e - f * 2 + p * 4) {
                case 1:
                case 5:
                    // Safety
                    sa[1 - p] = sa[1 - p] + 2;
                    print(ls[19] + "\n");
                    show_score();
                    print(os[p] + " KICKS OFF FROM ITS 20 YARD LINE.\n");
                    x = 20 + p * 60;
                    p = 1 - p;
                    start = 2;
                    continue;
                case 3:
                case 6:
                    // Defensive TD
                    print(ls[17] + "FOR " + os[1 - p] + "***\n");
                    p = 1 - p;
                    // Fall-thru
                case 2:
                case 8:
                    // Offensive TD
                    print(ls[17] + "***\n");
                    if (Math.random() <= 0.8) {
                        sa[p] = sa[p] + 7;
                        print("KICK IS GOOD.\n");
                    } else {
                        print("KICK IS OFF TO THE SIDE\n");
                        sa[p] = sa[p] + 6;
                    }
                    show_score();
                    print(os[p] + " KICKS OFF\n");
                    p = 1 - p;
                    start = 1;
                    continue;
                case 4:
                case 7:
                    // Touchback
                    print(ls[18] + "\n");
                    p = 1 - p;
                    x = 20 + p * 60;
                    start = 4;
                    continue;
            }
        }
    }
    print("END OF GAME  ***\n");
    print("FINAL SCORE:  " + os[0] + ": " + sa[0] + "  " + os[1] + ": " + sa[1] + "\n");
}

```

这道题是一个简单的 Python 代码，包含一个名为 "main()" 的函数。但为了更清楚地理解其作用，我们可以分析一下 "main()" 函数内包含的代码。

"main()" 函数内并没有显式地定义任何参数，也没有做其他的事情，它仅仅是一个函数名。所以 "main()" 函数本身并不能完成任何操作。

然而，这段代码作为程序的入口点（就是 "main()" 函数），当程序运行时会首先进入这个函数。所以，如果你在 "main()" 函数外编写代码，它们就无法直接被程序识别。而如果你在 "main()" 函数内编写代码，那么这段代码就会被执行。


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


# `37_Football/python/football.py`

这段代码是一个自定义的 Python 游戏，名为 "FOOTBALL"。它使用了 Python 的 json 模块、math 模块以及 random 模块，同时使用了 Pathlib 库。

具体来说，这段代码实现了一个简单的猜数字游戏。游戏需要玩家输入一个数字，程序会在 1 到 100 之间随机选择一个数字作为答案。如果输入的数字正确，游戏会输出 "恭喜你！"，否则游戏会输出 "很遗憾，你猜的数字是错误的！"。

程序的主要逻辑在以下几行：
```python
import json
from math import floor
from pathlib import Path
from random import randint, random
from typing import List, Tuple

# 定义一个列表，用于存储已经猜测过的数字
num_猜测 = [0] * 100

# 随机生成一个 1 到 100 的整数作为答案
answer = random.randint(1, 100)

# 如果猜测的数字和答案一致，输出 "恭喜你！"
if answer == num_猜测[0]:
   print("恭喜你！")

# 如果猜测的数字和答案不一致，输出 "很遗憾，你猜的数字是错误的！"
else:
   print("很遗憾，你猜的数字是错误的！")
```
这段代码的主要作用是提供一个简单而娱乐的游戏，让玩家猜测一个 1 到 100 之间的整数，如果猜对了游戏会开心地恭喜玩家，否则会通知玩家猜测的数字是错误的。


```
"""
FOOTBALL

A game.

Ported to Python by Martin Thoma in 2022.
The JavaScript version by Oscar Toledo G. (nanochess) was used
"""
# NOTE: The newlines might be wrong

import json
from math import floor
from pathlib import Path
from random import randint, random
from typing import List, Tuple

```

这段代码的主要作用是读取一个名为 "data.json" 的 JSON 文件，并将其内容存储在变量 "data" 中。然后，它从 "data" 字典中的 "players" 键中获取了所有玩家的编号，并将其存储在列表 "player_data" 中。接下来，它从 "actions" 字典中获取了所有动作，并将其存储在列表 "actions" 中。

最后，它创建了三个列表 "aa"、"ba" 和 "ca"，每个列表中都包含数字 -100，分别代表三个方向的分数变化。同时，它还创建了一个名为 "score" 的列表，其中包含两个元素，分别代表得分和副分。此外，它还创建了两个名为 "ta" 和 "wa" 的元组，分别包含一个整数和一个整数，分别表示击败和一个逆向移动的分数变化。最后，它还创建了两个名为 "xa" 和 "ya" 的元组，分别包含一个整数和一个整数，分别表示使用和一个卸载的分数变化。


```
with open(Path(__file__).parent / "data.json") as f:
    data = json.load(f)

player_data = [num - 1 for num in data["players"]]
actions = data["actions"]


aa: List[int] = [-100 for _ in range(20)]
ba: List[int] = [-100 for _ in range(20)]
ca: List[int] = [-100 for _ in range(40)]
score: List[int] = [0, 0]
ta: Tuple[int, int] = (1, 0)
wa: Tuple[int, int] = (-1, 1)
xa: Tuple[int, int] = (100, 0)
ya: Tuple[int, int] = (1, -1)
```

这段代码定义了一个函数 `ask_bool()`，该函数接受一个字符串参数 `prompt`，并返回一个布尔值。

在函数内部，有一个循环，该循环会不断地从用户那里获取输入，并将其存储在变量 `answer` 中。然后，该函数会使用 `input()` 函数来获取用户输入的 lowercase 版本，并将其存储在变量 `answer` 中。

接下来，该函数会使用条件语句来判断用户输入是否为 "yes" 或 "y"，如果是，则返回 `True`，否则返回 `False`。


```
za: Tuple[int, int] = (0, 100)
marker: Tuple[str, str] = ("--->", "<---")
t: int = 0
p: int = 0
winning_score: int


def ask_bool(prompt: str) -> bool:
    while True:
        answer = input(prompt).lower()
        if answer in ["yes", "y"]:
            return True
        elif answer in ["no", "n"]:
            return False


```

这两段代码的功能是询问用户输入是进攻还是防守，并在用户输入后返回相应的得分。

第一段代码 `ask_int` 是一个内部函数，它使用一个 while 循环来不断询问用户输入进攻或防守，直到用户输入正确为止。函数内部使用 `input` 函数获取用户输入，并使用 `try` 语句尝试将输入转换为整数，如果转换成功，则返回输入值。如果转换失败，则返回一个 None 对象。

第二段代码 `get_offense_defense` 是一个内部函数，它也是一个 while 循环，不断询问用户输入进攻或防守，直到用户输入正确为止。函数内部使用 `input` 函数获取用户输入，并使用 `split` 方法将输入的字符串分割成两个整数，并使用 `return` 语句返回它们。函数内部使用 `except` 语句来捕获异常情况，例如输入不是字符串或输入不合法的值等。


```
def ask_int(prompt: str) -> int:
    while True:
        answer = input(prompt)
        try:
            int_answer = int(answer)
            return int_answer
        except Exception:
            pass


def get_offense_defense() -> Tuple[int, int]:
    while True:
        input_str = input("INPUT OFFENSIVE PLAY, DEFENSIVE PLAY: ")
        try:
            p1, p2 = (int(n) for n in input_str.split(","))
            return p1, p2
        except Exception:
            pass


```

这段代码定义了三个函数，每个函数的作用如下：

1. `field_headers()`：该函数用于打印比赛场上的标题信息，包括两个队伍、每个队伍得分和比赛得分。函数返回 `None`。

2. `separator()`：该函数用于在输出的同时插入一个分隔符，使得结果看起来更加整齐。函数返回 `None`。

3. `show_ball()`：该函数接收一个二维列表 `da`，表示得分统计信息，包括得分和比赛场次。函数首先计算 `p` 的一半，然后打印标记和 `da[t]` 的和，接着打印 `" " * ((da[t] + 5) + int(p / 2))` 的内容，最后调用 `field_headers()` 函数。函数返回 `None`。


```
def field_headers() -> None:
    print("TEAM 1 [0   10   20   30   40   50   60   70   80   90   100] TEAM 2")
    print("\n\n")


def separator() -> None:
    print("+" * 72 + "\n")


def show_ball() -> None:
    da: Tuple[int, int] = (0, 3)
    print(" " * (da[t] + 5 + int(p / 2)) + marker[t] + "\n")
    field_headers()


```

这段代码定义了两个函数，show_scores()和loss_posession()。

show_scores()函数的作用是输出比赛得分情况。它首先打印一个空行，然后打印两个带有数字的行，分别是比赛中的第一得分和第二得分。接着，它又打印一个空行，并在其中包含一个条件语句，根据比赛得分是否大于获胜得分来判断比赛结果是否为胜。最后，如果条件成立，它将输出获胜的球队名称，并返回True。

loss_posession()函数的作用是计算失球情况。它首先获取一个名为t的整数，该整数表示被失球的球队。然后，它打印一个空行，并输出一个带有两个字符串的行，分别是失球前两个球队的名称。接着，它又打印一个空行，并在其中包含一个条件语句，判断是否失球。如果条件成立，它将输出一个带有三个字符串的行，分别是失球球队名称、失球时间和失球分数。最后，它将返回None，表明它没有做任何其他事情。


```
def show_scores() -> bool:
    print()
    print(f"TEAM 1 SCORE IS {score[0]}")
    print(f"TEAM 2 SCORE IS {score[1]}")
    print()
    if score[t] >= winning_score:
        print(f"TEAM {t+1} WINS*******************")
        return True
    return False


def loss_posession() -> None:
    global t
    print()
    print(f"** LOSS OF POSSESSION FROM TEAM {t+1} TO TEAM {ta[t]+1}")
    print()
    separator()
    print()
    t = ta[t]


```

这两函数是Python中的函数，函数1是“着陆点（touchdown）”，函数2是“打印顶部标题（print_header）”。

函数1的作用是输出一些关于足球比赛的信息，包括队伍得分、球员得分和比赛得分等。函数2的作用是在比赛开始前输出一些关于足球比赛的描述，包括比赛场地、球队数量等。

在对函数1进行进一步解析后，我们可以看到该函数使用了Morristown中等球队的语言，来输出信息。函数1使用了print函数来输出文本字符串，然后使用了f-string来打印其中的变量t。函数1还使用了q作为参数，来判断随机数g是否小于0.1。如果是，就重新设置为6；否则，就打印“EXTRA POINT NO GOOD”。最后，函数1还使用了score数据结构来存储各队伍的得分。

函数2的作用是在比赛开始前输出一些关于足球比赛的描述。函数2使用了print函数来输出文本字符串，然后使用了f-string来打印其中的变量t。函数2还使用了print函数来输出“FOOTBALL”和“CREATIVE COMPUTING”，来描述足球比赛和比赛的目的。


```
def touchdown() -> None:
    print()
    print(f"TOUCHDOWN BY TEAM {t+1} *********************YEA TEAM")
    q = 7
    g = random()
    if g <= 0.1:
        q = 6
        print("EXTRA POINT NO GOOD")
    else:
        print("EXTRA POINT GOOD")
    score[t] = score[t] + q


def print_header() -> None:
    print(" " * 32 + "FOOTBALL")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")
    print("PRESENTING N.F.U. FOOTBALL (NO FORTRAN USED)\n\n")


```

这段代码定义了一个名为 `print_instructions` 的函数，其返回类型为 `None`。函数内部打印出一段描述足球游戏的说明，包括比赛双方人数、每支队伍比赛的号码范围、每场比赛得分情况等。

具体来说，这段代码实现了一个游戏 scores 数据类型的列表，包含了 330 个分数值。每个分数值由两个整数表示，分别是球队编号和比赛号码。这段代码还实现了两个辅助函数，分别是 `pyt LeAgAp` 和 `dg`，它们可以用来查询关于足球比赛的相关信息。


```
def print_instructions() -> None:
    print(
        """THIS IS A FOOTBALL GAME FOR TWO TEAMS IN WHICH PLAYERS MUST
PREPARE A TAPE WITH A DATA STATEMENT (1770 FOR TEAM 1,
1780 FOR TEAM 2) IN WHICH EACH TEAM SCRAMBLES NOS. 1-20
THESE NUMBERS ARE THEN ASSIGNED TO TWENTY GIVEN PLAYS.
A LIST OF NOS. AND THEIR PLAYS IS PROVIDED WITH
BOTH TEAMS HAVING THE SAME PLAYS. THE MORE SIMILAR THE
PLAYS THE LESS YARDAGE GAINED.  SCORES ARE GIVEN
WHENEVER SCORES ARE MADE. SCORES MAY ALSO BE OBTAINED
BY INPUTTING 99,99 FOR PLAY NOS. TO PUNT OR ATTEMPT A
FIELD GOAL, INPUT 77,77 FOR PLAY NUMBERS. QUESTIONS WILL BE
ASKED THEN. ON 4TH DOWN, YOU WILL ALSO BE ASKED WHETHER
YOU WANT TO PUNT OR ATTEMPT A FIELD GOAL. IF THE ANSWER TO
BOTH QUESTIONS IS NO IT WILL BE ASSUMED YOU WANT TO
```

It appears that this is a Python program that is meant to simulate a football game. The program has several different functions that are called in order to simulate different aspects of the game, such as the conversion of a kickoff to a pass play or a play on special teams.

The program also includes a function called show\_scores, which is intended to display the score of the game at the end of each play. However, as the program is only printing out the scores for the first 5 games, it appears that this function is not being used in its entirety.

There are also several different parameters that can be passed to the program, such as the number of simulations, the number of simulations per game, and the probability of a fumble. These parameters can be adjusted at runtime to simulate different game settings.


```
TRY AND GAIN YARDAGE. ANSWER ALL QUESTIONS YES OR NO.
THE GAME IS PLAYED UNTIL PLAYERS TERMINATE (CONTROL-C).
PLEASE PREPARE A TAPE AND RUN.
"""
    )


def main() -> None:
    global winning_score
    print_header()
    want_instructions = ask_bool("DO YOU WANT INSTRUCTIONS? ")
    if want_instructions:
        print_instructions()
    print()
    winning_score = ask_int("PLEASE INPUT SCORE LIMIT ON GAME: ")
    for i in range(40):
        index = player_data[i - 1]
        if i < 20:
            aa[index] = i
        else:
            ba[index] = i - 20
        ca[i] = index
    offset = 0
    for t in [0, 1]:
        print(f"TEAM {t+1} PLAY CHART")
        print("NO.      PLAY")
        for i in range(20):
            input_str = f"{ca[i + offset]}"
            while len(input_str) < 6:
                input_str += " "
            input_str += actions[i]
            print(input_str)
        offset += 20
        t = 1
        print()
        print("TEAR OFF HERE----------------------------------------------")
        print("\n" * 10)

    field_headers()
    print("TEAM 1 DEFEND 0 YD GOAL -- TEAM 2 DEFENDS 100 YD GOAL.")
    t = randint(0, 1)
    print()
    print("THE COIN IS FLIPPED")
    routine = 1
    while True:
        if routine <= 1:
            p = xa[t] - ya[t] * 40
            separator()
            print(f"TEAM {t+1} RECEIVES KICK-OFF")
            k = floor(26 * random() + 40)
        if routine <= 2:
            p = p - ya[t] * k
        if routine <= 3:
            if wa[t] * p >= za[t] + 10:
                print("BALL WENT OUT OF ENDZONE --AUTOMATIC TOUCHBACK--")
                p = za[t] - wa[t] * 20
                if routine <= 4:
                    routine = 5
            else:
                print(f"BALL WENT {k} YARDS.  NOW ON {p}")
                show_ball()

        if routine <= 4:
            want_runback = ask_bool(f"TEAM {t+1} DO YOU WANT TO RUNBACK? ")

            if want_runback:
                k = floor(9 * random() + 1)
                r = floor(((xa[t] - ya[t] * p + 25) * random() - 15) / k)
                p = p - wa[t] * r
                print(f"RUNBACK TEAM {t+1} {r} YARDS")
                g = random()
                if g < 0.25:
                    loss_posession()
                    routine = 4
                    continue
                elif ya[t] * p >= xa[t]:
                    touchdown()
                    if show_scores():
                        return
                    t = ta[t]
                    routine = 1
                    continue
                elif wa[t] * p >= za[t]:
                    print(f"SAFETY AGAINST TEAM {t+1} **********************OH-OH")
                    score[ta[t]] = score[ta[t]] + 2
                    if show_scores():
                        return

                    p = za[t] - wa[t] * 20
                    want_punt = ask_bool(
                        f"TEAM {t+1} DO YOU WANT TO PUNT INSTEAD OF A KICKOFF? "
                    )
                    if want_punt:
                        print(f"TEAM {t+1} WILL PUNT")
                        g = random()
                        if g < 0.25:
                            loss_posession()
                            routine = 4
                            continue

                        print()
                        separator()
                        k = floor(25 * random() + 35)
                        t = ta[t]
                        routine = 2
                        continue

                    touchdown()
                    if show_scores():
                        return
                    t = ta[t]
                    routine = 1
                    continue
                else:
                    routine = 5
                    continue

            else:
                if wa[t] * p >= za[t]:
                    p = za[t] - wa[t] * 20

        if routine <= 5:
            d = 1
            s = p

        if routine <= 6:
            print("=" * 72 + "\n")
            print(f"TEAM {t+1} DOWN {d} ON {p}")
            if d == 1:
                if ya[t] * (p + ya[t] * 10) >= xa[t]:
                    c = 8
                else:
                    c = 4

            if c != 8:
                yards = 10 - (ya[t] * p - ya[t] * s)
                print(" " * 27 + f"{yards} YARDS TO 1ST DOWN")
            else:
                yards = xa[t] - ya[t] * p
                print(" " * 27 + f"{yards} YARDS")

            show_ball()
            if d == 4:
                routine = 8

        if routine <= 7:
            u = floor(3 * random() - 1)
            while True:
                p1, p2 = get_offense_defense()
                if t != 1:
                    p2, p1 = p1, p2

                if p1 == 99:
                    if show_scores():
                        return
                    if p1 == 99:
                        continue

                if p1 < 1 or p1 > 20 or p2 < 1 or p2 > 20:
                    print("ILLEGAL PLAY NUMBER, CHECK AND ", end="")
                    continue

                break
            p1 -= 1
            p2 -= 1

        if d == 4 or p1 == 77:
            want_punt = ask_bool(f"DOES TEAM {t+1} WANT TO PUNT? ")

            if want_punt:
                print()
                print(f"TEAM {t+1} WILL PUNT")
                g = random()
                if g < 0.25:
                    loss_posession()
                    routine = 4
                    continue

                print()
                separator()
                k = floor(25 * random() + 35)
                t = ta[t]
                routine = 2
                continue

            attempt_field_goal = ask_bool(
                f"DOES TEAM {t+1} WANT TO ATTEMPT A FIELD GOAL? "
            )

            if attempt_field_goal:
                print()
                print(f"TEAM {t+1} WILL ATTEMPT A FIELD GOAL")
                g = random()
                if g < 0.025:
                    loss_posession()
                    routine = 4
                    continue
                else:
                    f = floor(35 * random() + 20)
                    print()
                    print(f"KICK IS {f} YARDS LONG")
                    p = p - wa[t] * f
                    g = random()
                    if g < 0.35:
                        print("BALL WENT WIDE")
                    elif ya[t] * p >= xa[t]:
                        print(
                            f"FIELD GOLD GOOD FOR TEAM {t+1} *********************YEA"
                        )
                        q = 3
                        score[t] = score[t] + q
                        if show_scores():
                            return
                        t = ta[t]
                        routine = 1
                        continue

                    print(f"FIELD GOAL UNSUCCESFUL TEAM {t+1}-----------------TOO BAD")
                    print()
                    separator()
                    if ya[t] * p < xa[t] + 10:
                        print()
                        print(f"BALL NOW ON {p}")
                        t = ta[t]
                        show_ball()
                        routine = 4
                        continue
                    else:
                        t = ta[t]
                        routine = 3
                        continue

            else:
                routine = 7
                continue

        y = floor(
            abs(aa[p1] - ba[p2]) / 19 * ((xa[t] - ya[t] * p + 25) * random() - 15)
        )
        print()
        if t == 1 and aa[p1] < 11 or t == 2 and ba[p2] < 11:
            print("THE BALL WAS RUN")
        elif u == 0:
            print(f"PASS INCOMPLETE TEAM {t+1}")
            y = 0
        else:
            g = random()
            if g <= 0.025 and y > 2:
                print("PASS COMPLETED")
            else:
                print("QUARTERBACK SCRAMBLED")

        p = p - wa[t] * y
        print()
        print(f"NET YARDS GAINED ON DOWN {d} ARE {y}")

        g = random()
        if g <= 0.025:
            loss_posession()
            routine = 4
            continue
        elif ya[t] * p >= xa[t]:
            touchdown()
            if show_scores():
                return
            t = ta[t]
            routine = 1
            continue
        elif wa[t] * p >= za[t]:
            print()
            print(f"SAFETY AGAINST TEAM {t+1} **********************OH-OH")
            score[ta[t]] = score[ta[t]] + 2
            if show_scores():
                return
            p = za[t] - wa[t] * 20
            want_punt = ask_bool(
                f"TEAM {t+1} DO YOU WANT TO PUNT INSTEAD OF A KICKOFF? "
            )
            if want_punt:
                print()
                print(f"TEAM {t+1} WILL PUNT")
                g = random()
                if g < 0.25:
                    loss_posession()
                    routine = 4
                    continue

                print()
                separator()
                k = floor(25 * random() + 35)
                t = ta[t]
                routine = 2
                continue

            touchdown()
            if show_scores():
                return
            t = ta[t]
            routine = 1
        elif ya[t] * p - ya[t] * s >= 10:
            routine = 5
        else:
            d += 1
            if d != 5:
                routine = 6
            else:
                print()
                print(f"CONVERSION UNSUCCESSFUL TEAM {t+1}")
                t = ta[t]
                print()
                separator()
                routine = 5


```

这段代码是一个条件判断语句，它检查当前脚本是否作为主程序运行。如果是，那么程序会执行 main() 函数内部的代码。简单来说，这段代码的作用是判断当前脚本是否作为主程序运行，如果是，就执行 main() 函数内部的代码。


```
if __name__ == "__main__":
    main()

```