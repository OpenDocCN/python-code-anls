# BasicComputerGames源码解析 66

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `71_Poker/javascript/poker.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。

`print` 函数的作用是在文档中创建一个文本节点（`<br>` 标签），并将用户输入的字符串作为文本内容插入到该节点中。插入后，文本节点将添加到文档的 `<textarea>` 元素中，可以通过某种方式选择并打印出来。

`input` 函数的作用是接收用户输入的字符串，并在输入框中添加一个打叉号（`<br>` 标签）。打叉号将用户的输入作为参数传递给 `print` 函数，每次仅接收一个字符串。函数会在输入框中添加打叉号，然后通过某种方式提示用户输入字符串，例如通过 `Promise` 中的 `then` 方法，打印出用户输入的字符串，并在打叉号之后添加内容。


```
// POKER
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

这段代码定义了一个名为 `tab` 的函数，接受一个参数 `space`。这个函数的主要目的是输出一个由空格组成的字符串，其中的空格数量与传入的 `space` 参数相关。下面是这个函数的实现细节：

1. 定义了一个字符串变量 `str`，用于存储输出字符串中的所有字符。
2. 定义了一个变量 `space`，用于存储当前循环中的剩余空格数量。
3. 在一个循环中，从 `0` 初始化 `str` 字符串，并使用 `while` 循环来不断地添加字符 ` " "` 到 `str` 中。
4. 在循环中，使用 `space` 变量递减，直到剩余的空格数量为 `0`。
5. 返回 `str` 字符串，即 由空格组成的字符串。

该函数的作用是输出一个由空格组成的字符串，可以用于某些程序中的需求，例如在控制台输出中输出一个空行。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var aa = [];
var ba = [];
var b;
var c;
var d;
var g;
var i;
```

这段代码定义了多个变量k,m,n,p,s,u,v,x,z,hs,is,js,ks，以及一个函数fna。

函数fna的作用是返回一个介于0到9的随机整数。


```
var k;
var m;
var n;
var p;
var s;
var u;
var v;
var x;
var z;
var hs;
var is;
var js;
var ks;

function fna(x)
{
    return Math.floor(10 * Math.random());
}

```



这段代码定义了三个函数，其中第一个函数 `fnb(x)` 是一个输入函数，它将 `x` 随机整数并将其除以 100，然后将商输出。第二个函数 `im_busted()` 是一个输出函数，它输出 "I'M BUSTED. CONGRATULATIONS!"。第三个函数 `deal_card()` 是一个无限循环函数，它使用 `Math.random()` 生成一个 4 位数的随机整数，然后将其除以 100，以确保不是无效的牌型。如果生成的数字大于 3，或者数字不是 1-99 中的一个，那么函数将停止循环。在循环中，如果发现出现了重复的牌，那么函数将停止循环并继续执行下一次。如果 `z` 不等于 1，那么函数将遍历 `z-1` 次，每次将当前的牌型设置为之前已经设置过的牌型，然后将之前设置过的牌型设置为当前牌型的两倍，这样可以确保在每次循环中，都将会遇到之前设置过的牌型。


```
function fnb(x)
{
    return x % 100;
}

function im_busted()
{
    print("I'M BUSTED.  CONGRATULATIONS!\n");
}

// 1740
function deal_card()
{
    while (1) {
        aa[z] = 100 * Math.floor(4 * Math.random()) + Math.floor(100 * Math.random());
        if (Math.floor(aa[z] / 100) > 3)    // Invalid suit
            continue;
        if (aa[z] % 100 > 12) // Invalid number
            continue;
        if (z != 1) {
            for (k = 1; k <= z - 1; k++) {
                if (aa[z] == aa[k])
                    break;
            }
            if (k <= z - 1) // Repeated card
                continue;
            if (z > 10) {
                n = aa[u];
                aa[u] = aa[z];
                aa[z] = n;
            }
        }
        return;
    }
}

```

这段代码定义了一个名为 show_cards 的函数，其功能是输出一副扑克牌（包括大小王）的点数。

函数内部通过 for 循环遍历所有牌的点数（从 0 到 A），然后使用 fnb 函数输出这个点数。接着调用 show_number 和 show_suit 函数分别显示这个点数的点数和花色。如果当前循环的点数是偶数，则输出一个换行符。

最终，函数会在循环结束后输出一个包含所有牌点数的表格。


```
// 1850
function show_cards()
{
    for (z = n; z <= n + 4; z++) {
        print(" " + z + "--  ");
        k = fnb(aa[z]);
        show_number();
        print(" OF");
        k = Math.floor(aa[z] / 100);
        show_suit();
        if (z % 2 == 0)
            print("\n");
    }
    print("\n");
}

```

这段代码是一个 JavaScript 函数，名为 `show_number`，功能是输出一个数学计数器的视图。

具体来说，当 `k` 等于 9、10、11 或 12 时，函数会输出 "JACK"、"QUEEN" 或 "KING"，分别对应数字 9、10、11 和 12。当 `k` 小于 9 时，函数会先输出 " "（空格），然后输出 `k + 2`，即数字 `k` 在计数器中所在的位置（从 0 开始计数）。

输出结果如下：

```
JACK 
QUEEN 
KING 
ACE 
 7 
 8 
 9 
``` 

注意，这段代码会根据 `k` 的值选择不同的输出字符，而不会修改 `show_number` 函数本身。


```
// 1950
function show_number()
{
    if (k == 9)
        print("JACK");
    if (k == 10)
        print("QUEEN");
    if (k == 11)
        print("KING");
    if (k == 12)
        print("ACE");
    if (k < 9)
        print(" " + (k + 2));
}

```

It appears that the code is trying to determine if a statement is true or false. The code is using a combination of strategies such


```
// 2070
function show_suit()
{
    if (k == 0)
        print(" CLUBS\t");
    if (k == 1)
        print(" DIAMONDS\t");
    if (k == 2)
        print(" HEARTS\t");
    if (k == 3)
        print(" SPADES\t");
}

// 2170
function evaluate_hand()
{
    u = 0;
    for (z = n; z <= n + 4; z++) {
        ba[z] = fnb(aa[z]);
        if (z != n + 4) {
            if (Math.floor(aa[z] / 100) == Math.floor(aa[z + 1] / 100))
                u++;
        }
    }
    if (u == 4) {
        x = 11111;
        d = aa[n];
        hs = "A FLUS";
        is = "H IN";
        u = 15;
        return;
    }
    for (z = n; z <= n + 3; z++) {
        for (k = z + 1; k <= n + 4; k++) {
            if (ba[z] > ba[k]) {
                x = aa[z];
                aa[z] = aa[k];
                ba[z] = ba[k];
                aa[k] = x;
                ba[k] = aa[k] - 100 * Math.floor(aa[k] / 100);
            }
        }
    }
    x = 0;
    for (z = n; z <= n + 3; z++) {
        if (ba[z] == ba[z + 1]) {
            x = x + 11 * Math.pow(10, z - n);
            d = aa[z];
            if (u < 11) {
                u = 11;
                hs = "A PAIR";
                is = " OF ";
            } else if (u == 11) {
                if (ba[z] == ba[z - 1]) {
                    hs = "THREE";
                    is = " ";
                    u = 13;
                } else {
                    hs = "TWO P";
                    is = "AIR, ";
                    u = 12;
                }
            } else if (u == 12) {
                u = 16;
                hs = "FULL H";
                is = "OUSE, ";
            } else if (ba[z] == ba[z - 1]) {
                u = 17;
                hs = "FOUR";
                is = " ";
            } else {
                u = 16;
                hs = "FULL H";
                is = "OUSE. ";
            }
        }
    }
    if (x == 0) {
        if (ba[n] + 3 == ba[n + 3]) {
            x = 1111;
            u = 10;
        }
        if (ba[n + 1] + 3 == ba[n + 4]) {
            if (u == 10) {
                u = 14;
                hs = "STRAIG";
                is = "HT";
                x = 11111;
                d = aa[n + 4];
                return;
            }
            u = 10;
            x = 11110;
        }
    }
    if (u < 10) {
        d = aa[n + 4];
        hs = "SCHMAL";
        is = "TZ, ";
        u = 9;
        x = 11000;
        i = 6;
        return;
    }
    if (u == 10) {
        if (i == 1)
            i = 6;
        return;
    }
    if (u > 12)
        return;
    if (fnb(d) > 6)
        return;
    i = 6;
}

```



这个代码定义了一个名为 `get_prompt` 的函数和一个名为 `player_low_in_money` 的函数。

`get_prompt` 函数的作用是在屏幕上显示一个警告，并询问用户是否想要回答问题。如果用户输入了 "是的"，则会执行 `get_prompt` 函数，并将用户输入的字符串作为参数返回。如果用户输入了 "否"，则直接返回。

`player_low_in_money` 函数的作用是检查玩家是否在金钱方面陷入困境，并决定是否向用户显示他们当前的金钱状况。

函数首先打印一个警告信息，告诉用户他们不能赌博，然后询问用户是否想要继续。如果用户选择 "是"，则会尝试通过 `get_prompt` 函数来获取用户是否想要出售他们的问题。然后，函数会再次调用 `get_prompt` 函数来获取玩家是否想要出售他们的问题，并继续循环检查。

如果两次 `get_prompt` 函数的返回值都不是 "是"，则说明用户不想出售他们的东西，函数就会返回 `false`。否则，函数就会继续循环检查，直到第三次调用 `get_prompt` 函数，此时函数就会返回 `true`。


```
function get_prompt(question, def)
{
    var str;

    str = window.prompt(question, def);
    print(question + "? " + str + "\n");
    return str;
}

function player_low_in_money()
{
    print("\n");
    print("YOU CAN'T BET WITH WHAT YOU HAVEN'T GOT.\n");
    str = "N";
    if (o % 2 != 0) {
        str = get_prompt("WOULD YOU LIKE TO SELL YOUR WATCH", "YES");
        if (str.substr(0, 1) != "N") {
            if (fna(0) < 7) {
                print("I'LL GIVE YOU $75 FOR IT.\n");
                s += 75;
            } else {
                print("THAT'S A PRETTY CRUMMY WATCH - I'LL GIVE YOU $25.\n");
                s += 25;
            }
            o *= 2;
        }
    }
    if (o % 3 == 0 && str.substr(0, 1) == "N") {
        str = get_prompt("WILL YOU PART WITH THAT DIAMOND TIE TACK", "YES");
        if (str.substr(0, 1) != "N") {
            if (fna(0) < 6) {
                print("YOU ARE NOW $100 RICHER.\n");
                s += 100;
            } else {
                print("IT'S PASTE.  $25.\n");
                s += 25;
            }
            o *= 3;
        }
    }
    if (str.substr(0,1) == "N") {
        print("YOUR WAD IS SHOT.  SO LONG, SUCKER!\n");
        return true;
    }
    return false;
}

```

这段代码是一个名为`computer_low_in_money`的函数，它的作用是判断一个人是否可以在有限的预算内购买一台新的电脑。

该函数首先检查给定的预算`c`、零钱`g`和剩余的钱`v`是否满足购买电脑的条件，如果满足，则返回`true`，否则返回`false`。

如果预算`g`为零，则将`v`设置为`c`，并将`o`除以`2`，这样当`c`减去`g`时，如果`c`是负数，`print`会输出一条消息，然后将`k`设置为`g`，将`s`设置为`s-g`，将`c`设置为`c-k`，并将`p`设置为`p+g+k`，然后再次尝试购买。

如果预算`c`、`g`和`o`都为负数，但是`js`为正数，那么函数会再次尝试购买，并提示用户购买完后缀名为`TIE_TAK`的领带，售价为`50美元`。

如果预算`c`、`g`和`o`都为负数，但是`js`为负数，那么函数会提示用户他们的数学成绩很差，然后返回`false`，表明他们无法在这个预算内购买电脑。


```
function computer_low_in_money()
{
    if (c - g - v >= 0)
        return false;
    if (g == 0) {
        v = c;
        return false;
    }
    if (c - g < 0) {
        print("I'LL SEE YOU.\n");
        k = g;
        s = s - g;
        c = c - k;
        p = p + g + k;
        return false;
    }
    js = "N";
    if (o % 2 == 0) {
        js = get_prompt("WOULD YOU LIKE TO BUY BACK YOUR WATCH FOR $50", "YES");
        if (js.substr(0, 1) != "N") {
            c += 50;
            o /= 2;
        }
    }
    if (js.substr(0, 1) == "N" && o % 3 == 0) {
        js = get_prompt("WOULD YOU LIKE TO BUY BACK YOUR TIE TACK FOR $50", "YES");
        if (js.substr(0, 1) != "N") {
            c += 50;
            o /= 3;
        }
    }
    if (js.substr(0, 1) == "N") {
        print("I'M BUSTED.  CONGRATULATIONS!\n");
        return true;
    }
    return false;
}

```



这是一个牌类游戏中的函数，负责向玩家询问是否要下注。具体作用如下：

1. 如果玩家的手牌中没有筹码，或者当前牌面点数和小于所下注的牌面点数，函数会提示玩家“NO SMALL CHANGE, PLEASE.”并返回0。

2. 如果玩家手中的牌点数减去所下的注数后仍然小于25，函数会提示玩家“IF YOU CAN'T SEE MY BET, THEN FOLD.”并返回0。

3. 如果当前牌面点数等于25，且玩家没有下注，则会询问计算机是否会跟注。计算机的回答是“IF YOUR PLAYER HAS NO BET, THE COMPUTER WILL CONSIDER IT.”

4. 如果玩家下注，则会根据所下的注数计算出牌面点数，并将结果通知玩家。例如，如果玩家下注10点，则会计算出牌面点数为35，并输出“I'LL SEE YOU, AND RAISE YOU 25。”

5. 如果玩家和计算机都选择了跟注，则会根据玩家的牌面点数和手牌点数计算出牌面点数，并尝试通知玩家“WELL, IT'S TIME TO CHECK OUT.”。


```
function ask_for_bet()
{
    var forced;

    if (t != Math.floor(t)) {
        if (k != 0 || g != 0 || t != 0.5) {
            print("NO SMALL CHANGE, PLEASE.\n");
            return 0;
        }
        return 1;
    }
    if (s - g - t < 0) {
        if (player_low_in_money())
            return 2;
        return 0;
    }
    if (t == 0) {
        i = 3;
    } else if (g + t < k) {
        print("IF YOU CAN'T SEE MY BET, THEN FOLD.\n");
        return 0;
    } else {
        g += t;
        if (g != k) {
            forced = false;
            if (z != 1) {
                if (g <= 3 * z)
                    forced = true;
            } else {
                if (g <= 5) {
                    if (z < 2) {
                        v = 5;
                        if (g <= 3 * z)
                            forced = true;
                    }
                } else {
                    if (z == 1 || t > 25) {
                        i = 4;
                        print("I FOLD.\n");
                        return 1;
                    }
                }
            }
            if (forced || z == 2) {
                v = g - k + fna(0);
                if (computer_low_in_money())
                    return 2;
                print("I'LL SEE YOU, AND RAISE YOU " + v + "\n");
                k = g + v;
                return 0;
            }
            print("I'LL SEE YOU.\n");
            k = g;
        }
    }
    s -= g;
    c -= k;
    p += g + k;
    return 1;
}

```

这段代码是一个名为 `check_for_win` 的函数，用于在玩家游戏胜利或玩家失败时输出一些信息。

具体来说，函数有三种情况，根据玩家的游戏类型（0 代表普通玩，1 代表大神玩），会输出不同的信息。在每种情况下，函数会输出一次 `print` 函数，然后分别将 $c 和 $s 累加到变量中，最后输出一些信息，并返回 1，表示游戏胜利。

例如，如果玩的是大神玩，函数会输出：
```
I WIN.
YOU WIN.
NOW I HAVE 10 AND YOU HAVE 20
```
如果玩的是普通玩，函数会输出：
```
I WIN.
YOU WIN.
I LOSE.
```
总的来说，这段代码就是输出一些游戏胜利或失败时的信息。


```
function check_for_win(type)
{
    if (type == 0 && i == 3 || type == 1) {
        print("\n");
        print("I WIN.\n");
        c += p;
    } else if (type == 0 && i == 4 || type == 2) {
        print("\n");
        print("YOU WIN.\n");
        s += p;
    } else {
        return 0;
    }
    print("NOW I HAVE $" + c + " AND YOU HAVE $" + s + "\n");
    return 1;
}

```

这段代码定义了一个名为 `show_hand` 的函数，其作用是输出一个牌的三个元素(分别为牌面数字和小卒的点数)以及一个名为 `HS` 的数字，基于当前牌面数字。

具体来说，函数的实现过程如下：

1. 如果牌面数字为 `A`，则执行以下操作：
   a. 计算 `k / 100` 的整数部分，结果存储在 `k` 中。
   b. 输出 `k`。
   c. 输出一行字符 `\n`。
   d. 调用 `show_suit` 函数。
   e. 调用 `show_number` 函数。
   f. 如果 `HS` 是 `"A FLUS"`，则输出 `"HIGH"`。
   g. 否则输出 `"'S'"`。

2. 如果牌面数字为 `B`,`C`,`D`,`E`,`F`,`G`,`H`,`J`,`K`,`L`,`M`,`N`,`O`,`P`,`Q`,`R`,`S`,`T`,`U`,`V`,`W`,`X`,`Y`,`Z`,`a`,`b`,`c`,`d`,`e`,`f`,`g`,`h`,`i`,`j`,`k`,`l`,`m`,`n`,`o`,`p`,`q`,`r`,`s`,`t`,`u`,`v`,`w`,`x`,`y`,`z`,`a`,`b`,`c`,`d`,`e`,`f`,`g`,`h`,`i`,`j`,`k`,`l`,`m`,`n`,`o`,`p`,`q`,`r`,`s`,`t`,`u`,`v`,`w`,`x`,`y`,`z`,`a`,`b`,`c`,`d`,`e`,`f`,`g`,`h`,`i`,`j`,`k`,`l`,`m`,`n`,`o`,`p`,`q`,`r`,`s`,`t`,`u`,`v`,`w`,`x`,`y`,`z`,`a`,`b`,`c`,`d`,`e`,`f`,`g`,`h`,`i`,`j`,`k`,`l`,`m`,`n`,`o`,`p`,`q`,`r`,`s`,`t`,`u`,`v`,`w`,`x`,`y`,`z`,`a`,`b`,`c`,`d`,`e`,`f`,`g`,`h`,`i`,`j`,`k`,`l`,`m`,`n`,`o`,`p`,`q`,`r`,`s`,`t`,`u`,`v`,`w`,`x`,`y`,`z`,`a`,`b`,`c`,`d`,`e`,`f`,`g`,`h`,`i`,`j`,`k`,`l`,`m`,`n`,`o`,`p`,`q`,`r`,`s`,`t`,`u`,`v`,`w`,`x`,`y`,`z`,`a`,`b`,`c`,`d`,`e`,`f`,`g`,`h`,`i`,`j`,`k`,`l`,`m`,`n`,`o`,`p`,`q`,`r`,`s`,`t`,`u`,`v`,`w`,`x`,`y`,`z`,`a`,`b`,`c`,`d`,`e`,`f`,`g`,`h`,`i`,`j`,`k`,`l`,`m`,`n`,`o`,`p`,`q`,`r`,`s`,`t`,`u`,`v`,`w`,`x`,`y`,`z`,`a`,`b`,`c`,`d`,`e`,`f`,`g`,`h`,`i`,`j`,`k`,`l`,`m`,`n`,`o`,`p`,`q`,`r`,`s`,`t`,`u`,`v`,`w`,`


```
function show_hand()
{
    print(hs + is);
    if (hs == "A FLUS") {
        k = Math.floor(k / 100);
        print("\n");
        show_suit();
        print("\n");
    } else {
        k = fnb(k);
        show_number();
        if (hs == "SCHMAL" || hs == "STRAIG")
            print(" HIGH\n");
        else
            print("'S\n");
    }
}

```

This is a program written in JavaScript that allows you to compare two hands. You can choose to have a function that prints the cards in one of the hands and a function that compares the cards. The program also has a function that evaluates one hand and compares the other hand to the cards.

It starts by printing "NOW WE COMPARE HANDS:" and then it prompts the user to input the name of the hand. The program then compares the two hands. If the first hand is not a list, it will be treated as a random hand and the user will be prompted to input a name.

The program has a while loop that keeps asking the user if they want to continue until they type "YES" or "NO". If they type "YES", the program will continue and if "NO", the program will end.

If the user wants to continue, the program will ask for the name of the hand and compare the two hands. If the first hand is a list, the program will compare the first hand to the cards and if the hand is a random hand, it will be treated as a comparison and the program will compare the second hand to the cards.

The program also has a function that prints the name of the hand and compares the two hands. If the first hand is not a list, it will be treated as a random hand and the user will be prompted to input a name.

The program also has a function that evaluates one hand and compares the other hand to the cards.

It's important to note that the program is not checking for any potential issues with the way the hands are being compared, like if one hand has more cards than the other or if the comparison is not being done correctly.


```
// Main program
async function main()
{
    print(tab(33) + "POKER\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("WELCOME TO THE CASINO.  WE EACH HAVE $200.\n");
    print("I WILL OPEN THE BETTING BEFORE THE DRAW; YOU OPEN AFTER.\n");
    print("TO FOLD BET 0; TO CHECK BET .5.\n");
    print("ENOUGH TALK -- LET'S GET DOWN TO BUSINESS.\n");
    print("\n");
    o = 1;
    c = 200;
    s = 200;
    z = 0;
    while (1) {
        p = 0;
        //
        print("\n");
        if (c <= 5) {
            im_busted();
            return;
        }
        print("THE ANTE IS $5, I WILL DEAL:\n");
        print("\n");
        if (s <= 5) {
            if (player_low_in_money())
                return;
        }
        p += 10;
        s -= 5;
        c -= 5;
        for (z = 1; z <= 10; z++)
            deal_card();
        print("YOUR HAND:\n");
        n = 1;
        show_cards();
        n = 6;
        i = 2;
        evaluate_hand();
        print("\n");
        first = true;
        if (i == 6) {
            if (fna(0) > 7) {
                x = 11100;
                i = 7;
                z = 23;
            } else if (fna(0) > 7) {
                x = 11110;
                i = 7;
                z = 23;
            } else if (fna(0) < 2) {
                x = 11111;
                i = 7;
                z = 23;
            } else {
                z = 1;
                k = 0;
                print("I CHECK.\n");
                first = false;
            }
        } else {
            if (u < 13) {
                if (fna(0) < 2) {
                    i = 7;
                    z = 23;
                } else {
                    z = 0;
                    k = 0;
                    print("I CHECK.\n");
                    first = false;
                }
            } else if (u > 16) {
                z = 2;
                if (fna(0) < 1)
                    z = 35;
            } else {
                z = 35;
            }
        }
        if (first) {
            v = z + fna(0);
            g = 0;
            if (computer_low_in_money())
                return;
            print("I'LL OPEN WITH $" + v + "\n");
            k = v;
        }
        g = 0;
        do {
            print("\nWHAT IS YOUR BET");
            t = parseFloat(await input());
            status = ask_for_bet();
        } while (status == 0) ;
        if (status == 2)
            return;
        status = check_for_win(0);
        if (status == 1) {
            while (1) {
                print("DO YOU WISH TO CONTINUE");
                hs = await input();
                if (hs == "YES") {
                    status = 1;
                    break;
                }
                if (hs == "NO") {
                    status = 2;
                    break;
                }
                print("ANSWER YES OR NO, PLEASE.\n");
            }
        }
        if (status == 2)
            return;
        if (status == 1) {
            p = 0;
            continue;
        }
        print("\n");
        print("NOW WE DRAW -- HOW MANY CARDS DO YOU WANT");
        while (1) {
            t = parseInt(await input());
            if (t != 0) {
                z = 10;
                if (t >= 4) {
                    print("YOU CAN'T DRAW MORE THAN THREE CARDS.\n");
                    continue;
                }
                print("WHAT ARE THEIR NUMBERS:\n");
                for (q = 1; q <= t; q++) {
                    u = parseInt(await input());
                    z++;
                    deal_card();
                }
                print("YOUR NEW HAND:\n");
                n = 1;
                show_cards();
            }
            break;
        }
        z = 10 + t;
        for (u = 6; u <= 10; u++) {
            if (Math.floor(x / Math.pow(10, u - 6)) != 10 * Math.floor(x / Math.pow(10, u - 5)))
                break;
            z++;
            deal_card();
        }
        print("\n");
        print("I AM TAKING " + (z - 10 - t) + " CARD");
        if (z != 11 + t) {
            print("S");
        }
        print("\n");
        n = 6;
        v = i;
        i = 1;
        evaluate_hand();
        b = u;
        m = d;
        if (v == 7) {
            z = 28;
        } else if (i == 6) {
            z = 1;
        } else {
            if (u < 13) {
                z = 2;
                if (fna(0) == 6)
                    z = 19;
            } else if (u < 16) {
                z = 19;
                if (fna(0) == 8)
                    z = 11;
            } else {
                z = 2;
            }
        }
        k = 0;
        g = 0;
        do {
            print("\nWHAT IS YOUR BET");
            t = parseFloat(await input());
            status = ask_for_bet();
        } while (status == 0) ;
        if (status == 2)
            return;
        if (t == 0.5) {
            if (v != 7 && i == 6) {
                print("I'LL CHECK\n");
            } else {
                v = z + fna(0);
                if (computer_low_in_money())
                    return;
                print("I'LL BET $" + v + "\n");
                k = v;
                do {
                    print("\nWHAT IS YOUR BET");
                    t = parseFloat(await input());
                    status = ask_for_bet();
                } while (status == 0) ;
                if (status == 2)
                    return;
                status = check_for_win(0);
                if (status == 1) {
                    while (1) {
                        print("DO YOU WISH TO CONTINUE");
                        hs = await input();
                        if (hs == "YES") {
                            status = 1;
                            break;
                        }
                        if (hs == "NO") {
                            status = 2;
                            break;
                        }
                        print("ANSWER YES OR NO, PLEASE.\n");
                    }
                }
                if (status == 2)
                    return;
                if (status == 1) {
                    p = 0;
                    continue;
                }
            }
        } else {
            status = check_for_win(0);
            if (status == 1) {
                while (1) {
                    print("DO YOU WISH TO CONTINUE");
                    hs = await input();
                    if (hs == "YES") {
                        status = 1;
                        break;
                    }
                    if (hs == "NO") {
                        status = 2;
                        break;
                    }
                    print("ANSWER YES OR NO, PLEASE.\n");
                }
            }
            if (status == 2)
                return;
            if (status == 1) {
                p = 0;
                continue;
            }
        }
        print("\n");
        print("NOW WE COMPARE HANDS:\n");
        js = hs;
        ks = is;
        print("MY HAND:\n");
        n = 6;
        show_cards();
        n = 1;
        evaluate_hand();
        print("\n");
        print("YOU HAVE ");
        k = d;
        show_hand();
        hs = js;
        is = ks;
        k = m;
        print("AND I HAVE ");
        show_hand();
        status = 0;
        if (b > u) {
            status = 1;
        } else if (u > b) {
            status = 2;
        } else {
            if (hs != "A FLUS") {
                if (fnb(m) < fnb(d))
                    status = 2;
                else if (fnb(m) > fnb(d))
                    status = 1;
            } else {
                if (fnb(m) > fnb(d))
                    status = 1;
                else if (fnb(d) > fnb(m))
                    status = 2;
            }
            if (status == 0) {
                print("THE HAND IS DRAWN.\n");
                print("ALL $" + p + " REMAINS IN THE POT.\n");
                continue;
            }
        }
        status = check_for_win(status);
        if (status == 1) {
            while (1) {
                print("DO YOU WISH TO CONTINUE");
                hs = await input();
                if (hs == "YES") {
                    status = 1;
                    break;
                }
                if (hs == "NO") {
                    status = 2;
                    break;
                }
                print("ANSWER YES OR NO, PLEASE.\n");
            }
        }
        if (status == 2)
            return;
        if (status == 1) {
            p = 0;
            continue;
        }
    }
}

```

这道题目要求解释以下代码的作用，但是我不清楚你指的是哪段代码。如果你可以提供更多上下文或者详细解释其中的一部分，我将非常乐意帮助你。


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


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Queen

This game is based on the permissible moves of the chess queen — i.e., along any vertical, horizontal, or diagonal. In this game, the queen can only move to the left, down, and diagonally down to the left.

The object of the game is to place the queen (one only) in the lower left-hand square (no. 158), by alternating moves between you and the computer. The one to place the queen there wins.

You go first and place the queen in any one of the squares on the top row or the right-hand column. That is your first move. The computer is beatable, but it takes some figuring. See if you can devise a winning strategy.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=133)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=148)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `72_Queen/csharp/Computer.cs`

这段代码定义了一个名为Queen的命名空间类，包含一个名为Computer的内部类。

在这个内部类中，定义了一个名为_randomiseFrom的静态HashSet，包含6个随机数。还定义了一个名为_desirable的静态HashSet，包含4个随机数。还定义了一个名为_random的静态变量，其值是一个从0到让位的IRandom。

在Computer的构造函数中，将_random赋值给Computer的实例变量。

在GetMove方法中，从Computer的实例变量中获取随机数种子，然后使用该随机数种子从_randomiseFrom和_desirable中获取一个随机数，并将其作为从对象的位置。如果从对象中查找位置是有效的，则返回该位置。如果从对象中查找位置是无效的，则使用随机数生成一个位置。

在IsOptimal方法中，从7个选项中选择一个方向，然后使用回溯法查找该方向是否是最优的。

在代码的后面部分，还定义了一个名为Queen的类，但没有实现任何方法。


```
namespace Queen;

internal class Computer
{
    private static readonly HashSet<Position> _randomiseFrom = new() { 41, 44, 73, 75, 126, 127 };
    private static readonly HashSet<Position> _desirable = new() { 73, 75, 126, 127, 158 };
    private readonly IRandom _random;

    public Computer(IRandom random)
    {
        _random = random;
    }

    public Position GetMove(Position from)
        => from + (_randomiseFrom.Contains(from) ? _random.NextMove() : FindMove(from));

    private Move FindMove(Position from)
    {
        for (int i = 7; i > 0; i--)
        {
            if (IsOptimal(Move.Left, out var move)) { return move; }
            if (IsOptimal(Move.Down, out move)) { return move; }
            if (IsOptimal(Move.DownLeft, out move)) { return move; }

            bool IsOptimal(Move direction, out Move move)
            {
                move = direction * i;
                return _desirable.Contains(from + move);
            }
        }

        return _random.NextMove();
    }
}

```

# `72_Queen/csharp/Game.cs`

This is a simple game where the player has to choose between collaborating with a computer or competing with it. The game has a unique starting position, where the player has to decide if they want to be the computer or the human.

The `Computer` class is responsible for generating the move for the computer, while the `Game` class is responsible for displaying the game board, handling user input, and checking the validity of the moves.

The `PlaySeries` method is called when the user clicks the "Play Series" button, which will display the title of the game, Instructions to play, and then ask the user if they want to proceed or if they want to reset the game. After that, the game will loop through the moves, checking if the human or the computer wins, and displaying the appropriate message in the `_io` object.

The `PlayGame` method is responsible for generating the moves for the computer and human, based on the current position of the game. It uses the `GetMove` method of the `Computer` class to get the move for the human, and then displays the computer's move by calling the `Strings.ComputerMove` method, and displays the appropriate message for the human or the computer if they win or if the game is a draw.


```
namespace Queen;

internal class Game
{
    private readonly IReadWrite _io;
    private readonly IRandom _random;
    private readonly Computer _computer;

    public Game(IReadWrite io, IRandom random)
    {
        _io = io;
        _random = random;
        _computer = new Computer(random);
    }

    internal void PlaySeries()
    {
        _io.Write(Streams.Title);
        if (_io.ReadYesNo(Prompts.Instructions)) { _io.Write(Streams.Instructions); }

        while (true)
        {
            var result = PlayGame();
            _io.Write(result switch
            {
                Result.HumanForfeits => Streams.Forfeit,
                Result.HumanWins => Streams.Congratulations,
                Result.ComputerWins => Streams.IWin,
                _ => throw new InvalidOperationException($"Unexpected result {result}")
            });

            if (!_io.ReadYesNo(Prompts.Anyone)) { break; }
        }

        _io.Write(Streams.Thanks);
    }

    private Result PlayGame()
    {
        _io.Write(Streams.Board);
        var humanPosition = _io.ReadPosition(Prompts.Start, p => p.IsStart, Streams.IllegalStart, repeatPrompt: true);
        if (humanPosition.IsZero) { return Result.HumanForfeits; }

        while (true)
        {
            var computerPosition = _computer.GetMove(humanPosition);
            _io.Write(Strings.ComputerMove(computerPosition));
            if (computerPosition.IsEnd) { return Result.ComputerWins; }

            humanPosition = _io.ReadPosition(Prompts.Move, p => (p - computerPosition).IsValid, Streams.IllegalMove);
            if (humanPosition.IsZero) { return Result.HumanForfeits; }
            if (humanPosition.IsEnd) { return Result.HumanWins; }
        }
    }

    private enum Result { ComputerWins, HumanWins, HumanForfeits };
}

```

# `72_Queen/csharp/IOExtensions.cs`

这段代码是一个名为Queen的命名空间中包含的内部类IOExtensions，其作用是帮助IO读写流处理输入数据。

具体来说，代码中定义了两个方法，一个是ReadYesNo，用于读取用户输入的"是"或"否"的答案，另一个是ReadPosition，用于读取指定输入的指定位置。

ReadYesNo方法中，通过循环不断从IO流中读取用户的输入，并将其转换为字符串，然后使用ToLower()方法将其转换为lowercase字符串。接着，代码中使用while循环不断询问用户输入是"是"还是"否"，如果用户输入为"是"，则返回true，否则返回false。最后，代码将一个"是"或"否"的字符串发送回给IO流，并在每次循环结束后将其写入到Streams.YesOrNo流中。

ReadPosition方法中，首先使用while循环不断读取用户输入，并将其转换为整数。然后，根据所读取的整数，创建一个Position对象，并检查其是否等于所读取的整数以及isValid函数的返回值。如果所创建的Position对象等于所读取的整数，并且isValid函数返回true，则返回该Position对象。

如果所创建的Position对象不等于所读取的整数，或者isValid函数返回false，则代码将错误信息写入到Error流中，并将重复Prompt设置为false，以便在下一次读取输入之前重新询问用户输入。


```
namespace Queen;

internal static class IOExtensions
{
    internal static bool ReadYesNo(this IReadWrite io, string prompt)
    {
        while (true)
        {
            var answer = io.ReadString(prompt).ToLower();
            if (answer == "yes") { return true; }
            if (answer == "no") { return false; }

            io.Write(Streams.YesOrNo);
        }
    }

    internal static Position ReadPosition(
        this IReadWrite io,
        string prompt,
        Predicate<Position> isValid,
        Stream error,
        bool repeatPrompt = false)
    {
        while (true)
        {
            var response = io.ReadNumber(prompt);
            var number = (int)response;
            var position = new Position(number);
            if (number == response && (position.IsZero || isValid(position)))
            {
                return position;
            }

            io.Write(error);
            if (!repeatPrompt) { prompt = ""; }
        }
    }
}

```

# `72_Queen/csharp/Move.cs`

这段代码定义了一个名为 Queen 的命名空间，其中包含一个名为 Move 的内部记录类型，表示移动的方向和步数。

Move 记录类型包含四个成员变量：左右方向(IsLeft 和 IsDown)，步数(Row)以及一个布尔值，表示移动是否有效。

Move 类还包含三个成员函数：Left,DownLeft 和 Down，这些函数返回一个 Move 对象，分别表示向左、向下和向下的移动。

移动函数 operator * 的实现比较复杂，它将移动参量 move 乘以一个整数 scale，并返回一个新的 Move 对象。这个新的移动对象包含移动参量 move 的左右方向和步数分别乘以 scale，而 row 属性则乘以 scale。

整数 scale 只能为 1 或 2。在 isValid 变量中，条件判断是否满足以下两个条件之一：Diagonal > 0 或者 IsLeft 或 IsDown 或 IsDownLeft。如果满足条件，isValid 变量将永远为 true，这样移动函数 operator * 将返回一个新的移动对象。


```
namespace Queen;

internal record struct Move(int Diagonal, int Row)
{
    public static readonly Move Left = new(1, 0);
    public static readonly Move DownLeft = new(2, 1);
    public static readonly Move Down = new(1, 1);

    public bool IsValid => Diagonal > 0 && (IsLeft || IsDown || IsDownLeft);
    private bool IsLeft => Row == 0;
    private bool IsDown => Row == Diagonal;
    private bool IsDownLeft => Row * 2 == Diagonal;

    public static Move operator *(Move move, int scale) => new(move.Diagonal * scale, move.Row * scale);
}
```

# `72_Queen/csharp/Position.cs`

这段代码定义了一个结构体 `Position`，表示二维游戏地图中的位置。这个结构体有两个成员变量：`Diagonal` 和 `Row`，分别表示纵横坐标。另外，还有一个成员变量 `Zero`，表示坐标为 0 的位置。还有一个 `ToString` 方法，用于将 `Position` 对象转换为字符串形式。

接着，定义了一个内部类 `Position`，这个类继承自 `Position` 结构体，并添加了一些方法。例如， `IsZero` 方法用于判断是否是坐标为 0 的位置； `IsStart` 方法用于判断是否是游戏地图的起点或终点； `IsEnd` 方法用于判断是否是游戏地图的起点或终点，并且这两个坐标的位置关系为行 8，列 15。

接着，定义了一个 `operator+` 方法，用于将两个 `Position` 对象相加，并返回一个新的 `Position` 对象。这个方法接收两个参数：一个 `Move` 对象，表示从第二个位置向目标位置移动。

接着，定义了一个 `operator-` 方法，用于将两个 `Position` 对象相减，并返回一个新的 `Position` 对象。这个方法同样接收两个参数：一个 `Move` 对象，表示从目标位置向原始位置移动。


```
namespace Queen;

internal record struct Position(int Diagonal, int Row)
{
    public static readonly Position Zero = new(0);

    public Position(int number)
        : this(Diagonal: number / 10, Row: number % 10)
    {
    }

    public bool IsZero => Row == 0 && Diagonal == 0;
    public bool IsStart => Row == 1 || Row == Diagonal;
    public bool IsEnd => Row == 8 && Diagonal == 15;

    public override string ToString() => $"{Diagonal}{Row}";

    public static implicit operator Position(int value) => new(value);

    public static Position operator +(Position position, Move move)
        => new(Diagonal: position.Diagonal + move.Diagonal, Row: position.Row + move.Row);
    public static Move operator -(Position to, Position from)
        => new(Diagonal: to.Diagonal - from.Diagonal, Row: to.Row - from.Row);
}

```

# `72_Queen/csharp/Program.cs`

这段代码的作用是创建一个名为 "Game" 的类，它继承自 "Queen.Resources.Resource" 类。它使用 "Games.Common.IO" 和 "Games.Common.Randomness" 两个命名空间中的类来导入输入输出流和随机数生成器的接口。

然后，它引入了 "Queen" 命名空间中的类，创建了一个新的 "Game" 类，并使用 "new ConsoleIO()" 和 "new RandomNumberGenerator()" 方法来创建一个输出流和一个随机数生成器实例。

接着，它使用 "using Queen.Resources.Resource;` 引入了 "Queen.Resources.Resource" 命名空间中的类，这样就可以使用该类中的资源了。

最后，它创建了一个新的 "Game" 实例，使用 "new Game(new ConsoleIO(), new RandomNumberGenerator()).PlaySeries()" 的方法来初始化游戏并开始播放测试系列。


```
global using Games.Common.IO;
global using Games.Common.Randomness;
global using static Queen.Resources.Resource;

using Queen;

new Game(new ConsoleIO(), new RandomNumberGenerator()).PlaySeries();
```

# `72_Queen/csharp/RandomExtensions.cs`

这段代码是一个名为Queen的namespace内部类，定义了一个名为RandomExtensions的internal静态类。

这个类实现了一个名为NextMove的函数，接受一个IRandom类型的参数。这个函数返回一个Move类型的对象，表示随机下一步的位置相对于当前位置的偏移量。

NextMove函数使用了一个switch语句，根据输入随机数的大小，返回不同的移动方向。具体来说，当随机数大于0.6F时，返回Move.Down方向，当随机数大于0.3F时，返回Move.DownLeft方向，否则返回Move.Left方向。


```
namespace Queen;

internal static class RandomExtensions
{
    internal static Move NextMove(this IRandom random)
        => random.NextFloat() switch
        {
            > 0.6F => Move.Down,
            > 0.3F => Move.DownLeft,
            _ => Move.Left
        };
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `72_Queen/csharp/Resources/Resource.cs`



这段代码是一个程序集类，名为`Queen.Resources`，其目的是在游戏`Queen`中提供一些与游戏交互的资源。具体来说，它包含以下几个类：

- `Streams`类，包含各种类型的输出流，例如游戏窗口中的标题、进度条、提示信息等等。
- `Prompts`类，包含一些用于游戏玩家输入的提示信息。
- `Strings`类，包含一些可以用来在游戏界面上显示的文本字符串。
- `GetStream`方法，用于获取指定名称的输出流。
- `GetPrompt`方法，根据指定名称返回一个提示信息。
- `Assembly`类，用于获取执行当前程序的程序集，并从其资源中获取资源。

由于该程序集是使用`GetExecutingAssembly`方法获取执行程序的程序集，然后使用`GetManifestResourceStream`方法获取资源文件中的内容，因此它可以在运行时动态地获取，而无需事先将所有资源加载到内存中。


```
using System.Reflection;
using System.Runtime.CompilerServices;

namespace Queen.Resources;

internal static class Resource
{
    internal static class Streams
    {
        public static Stream Title => GetStream();
        public static Stream Instructions => GetStream();
        public static Stream YesOrNo => GetStream();
        public static Stream Board => GetStream();
        public static Stream IllegalStart => GetStream();
        public static Stream IllegalMove => GetStream();
        public static Stream Forfeit => GetStream();
        public static Stream IWin => GetStream();
        public static Stream Congratulations => GetStream();
        public static Stream Thanks => GetStream();
    }

    internal static class Prompts
    {
        public static string Instructions => GetPrompt();
        public static string Start => GetPrompt();
        public static string Move => GetPrompt();
        public static string Anyone => GetPrompt();
    }

    internal static class Strings
    {
        public static string ComputerMove(Position position) => string.Format(GetString(), position);
    }

    private static string GetPrompt([CallerMemberName] string? name = null) => GetString($"{name}Prompt");

    private static string GetString([CallerMemberName] string? name = null)
    {
        using var stream = GetStream(name);
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }

    private static Stream GetStream([CallerMemberName] string? name = null) =>
        Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")
            ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
}
```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `72_Queen/javascript/queen.js`

该代码是将BASIC语言中的函数转换为JavaScript后生成的代码。主要目的是创建一个名为"console"的2D画布，并在其中画出一个输入框和两个按钮。当点击按钮时，将接收用户输入的文本并将其附加到"console"画布中的第二个文本框中，然后将"console"画布中的文本打印出来并将其删除。

具体来说，代码中的两个主要函数是"input"和"print"。

"input"函数从用户接收输入的文本，并在2D画布上创建一个带有输入框的按钮。该函数将弹出一个包含输入框的窗口，允许用户输入文本并将其保存在变量"input_str"中。当用户单击按钮时，该函数将接收输入并将其附加到"console"画布中的第二个文本框中。

"print"函数将接收一个字符串参数，并在"console"画布中的第二个文本框中将其打印出来。


```
// QUEEN
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

这段代码定义了一个名为 `tab` 的函数，该函数接收一个整数参数 `space`。函数内部创建了一个空字符串 `str`，并使用 while 循环来遍历 `space` 次减一，每次将一个空格添加到 `str` 的末尾。

在代码的最后，定义了一个包含 1 到 12 的数字数组 `sa`，用于存储数字。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var sa = [,81,  71,  61,  51,  41,  31,  21,  11,
           92,  82,  72,  62,  52,  42,  32,  22,
          103,  93,  83,  73,  63,  53,  43,  33,
          114, 104,  94,  84,  74,  64,  54,  44,
          125, 115, 105,  95,  85,  75,  65,  55,
          136, 126, 116, 106,  96,  86,  76,  66,
          147, 137, 127, 117, 107,  97,  87,  77,
          158, 148, 138, 128, 118, 108,  98,  88];

```

这段代码定义了一个名为 show_instructions 的函数，该函数用于在屏幕上输出游戏棋盘的移动说明。

在函数内部，首先声明了四个变量：m、m1、u 和 t，分别代表棋盘、棋子、玩家 1 和玩家的棋子。

接着，定义了一个名为 t1 的变量，用于存储玩家 1 的棋子。

在 show_instructions 函数的显示中，首先向玩家介绍了游戏规则，然后描述了玩家可以进行的操作。

接下来是具体的游戏过程。在每一次玩家移动棋子时，函数会输出相应的提示信息，然后提示玩家输入新的移动，并确保玩家在每一次回复后按回键确认。

当玩家完成其移动操作时，函数会输出一条消息，然后继续等待下一次移动操作。

总的来说，这段代码定义了一个用于在屏幕上输出游戏棋盘移动说明的函数，以便玩家更好地了解游戏规则和操作方式。


```
var m;
var m1;
var u;
var t;
var u1;
var t1;

function show_instructions()
{
    print("WE ARE GOING TO PLAY A GAME BASED ON ONE OF THE CHESS\n");
    print("MOVES.  OUR QUEEN WILL BE ABLE TO MOVE ONLY TO THE LEFT,\n");
    print("DOWN, OR DIAGONALLY DOWN AND TO THE LEFT.\n");
    print("\n");
    print("THE OBJECT OF THE GAME IS TO PLACE THE QUEEN IN THE LOWER\n");
    print("LEFT HAND SQUARE BY ALTERNATING MOVES BETWEEN YOU AND THE\n");
    print("COMPUTER.  THE FIRST ONE TO PLACE THE QUEEN THERE WINS.\n");
    print("\n");
    print("YOU GO FIRST AND PLACE THE QUEEN IN ANY ONE OF THE SQUARES\n");
    print("ON THE TOP ROW OR RIGHT HAND COLUMN.\n");
    print("THAT WILL BE YOUR FIRST MOVE.\n");
    print("WE ALTERNATE MOVES.\n");
    print("YOU MAY FORFEIT BY TYPING '0' AS YOUR MOVE.\n");
    print("BE SURE TO PRESS THE RETURN KEY AFTER EACH RESPONSE.\n");
    print("\n");
    print("\n");
}

```

这段代码定义了一个名为 `show_map` 的函数，它的作用是打印一个 8x8 的网格。

具体来说，这段代码首先输出一个空行，然后使用两个 for 循环来遍历这个网格的每个单元格。在循环变量 `a` 被设置为 0 时，`b` 被设置为 1，这样外层循环将遍历 8 次，而内层循环将遍历 8 次。

在内层循环中，通过 `i = 8 * a + b` 计算当前单元格的行数，然后使用 `print(" " + sa[i] + " ");` 在当前单元格中输出字符串 `" " + sa[i] + " "`，其中 `sa` 可能是一个数组，这里我们无法得知具体内容。最后，内层循环执行完毕后，使用 `print("\n")` 在每个内层循环行末尾输出一个换行符。

整个外层循环执行完毕后，又执行了三个内层循环，所以最终输出的结果是一个 8x8 的网格，每个单元格中都有 " " 字符。


```
function show_map()
{
    print("\n");
    for (var a = 0; a <= 7; a++) {
        for (var b = 1; b <= 8; b++) {
            i = 8 * a + b;
            print(" " + sa[i] + " ");
        }
        print("\n");
        print("\n");
        print("\n");
    }
    print("\n");
}

```

这两函数Check if the number is equal to some fixed number is called as "move" function.
This move function takes two arguments, first a number m and second a variable t.
It checks whether the value of m is equal to some of the fixed number which are 158, 127, 126, or 75 or 73.
If the value of m is equal to any of those fixed numbers, it will return true else it will return false.

This function is using if else statement.
It first checks the value of m whether it is equal to any of the fixed numbers or not.
If it is not equal to any of the fixed numbers then it will set the variable t to 0 and u to random number between 0 to 100.
Then it will calculate the value of m = 10*t + u.

This function is useful for testing if the move function is working as expected or not.


```
function test_move()
{
    m = 10 * t + u;
    if (m == 158 || m == 127 || m == 126 || m == 75 || m == 73)
        return true;
    return false;
}

function random_move()
{
    // Random move
    z = Math.random();
    if (z > 0.6) {
        u = u1 + 1;
        t = t1 + 1;
    } else if (z > 0.3) {
        u = u1 + 1;
        t = t1 + 2;
    } else {
        u = u1;
        t = t1 + 1;
    }
    m = 10 * t + u;
}

```



该代码定义了一个名为 computer_move 的函数，其作用是让游戏角色(未定义)进行移动。

函数体中包含一系列条件判断，判断移动是否成功，如果成功则执行 random_move 函数，否则不执行该函数。因此，该代码的作用是让游戏角色在尝试移动时，如果移动成功，则随机选择移动的方向，否则禁止移动。


```
function computer_move()
{
    if (m1 == 41 || m1 == 44 || m1 == 73 || m1 == 75 || m1 == 126 || m1 == 127) {
        random_move();
        return;
    }
    for (k = 7; k >= 1; k--) {
        u = u1;
        t = t1 + k;
        if (test_move())
            return;
        u += k;
        if (test_move())
            return;
        t += k;
        if (test_move())
            return;
    }
    random_move();
}

```

This is a program written in JavaScript that allows the user to play a game of chess. The program will play a game against the computer using a different move each time, or it will ask the user to make a move. The user can win the game by offering to forfeit or by making a valid move. The program will also print the moves made by the computer and the user.

The program starts by asking the user if they want to play a game or if they want to exit. If the user wants to play a game, the program will ask the user to make a move. The program will then start making moves for the user. If the user wants to forfeit, the program will print a message and end the game. If the user makes a valid move, the program will print a message and continue making moves for the user.

The program will continue to do this until the user quits the game. The program will also print a message at the end of the game to thank the user for playing.


```
// Main program
async function main()
{
    print(tab(33) + "QUEEN\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");

    while (1) {
        print("DO YOU WANT INSTRUCTIONS");
        str = await input();
        if (str == "YES" || str == "NO")
            break;
        print("PLEASE ANSWER 'YES' OR 'NO'.\n");
    }
    if (str == "YES")
        show_instructions();
    while (1) {
        show_map();
        while (1) {
            print("WHERE WOULD YOU LIKE TO START");
            m1 = parseInt(await input());
            if (m1 == 0) {
                print("\n");
                print("IT LOOKS LIKE I HAVE WON BY FORFEIT.\n");
                print("\n");
                break;
            }
            t1 = Math.floor(m1 / 10);
            u1 = m1 - 10 * t1;
            if (u1 == 1 || u1 == t1)
                break;
            print("PLEASE READ THE DIRECTIONS AGAIN.\n");
            print("YOU HAVE BEGUN ILLEGALLY.\n");
            print("\n");
        }
        while (m1) {
            if (m1 == 158) {
                print("\n");
                print("C O N G R A T U L A T I O N S . . .\n");
                print("\n");
                print("YOU HAVE WON--VERY WELL PLAYED.\n");
                print("IT LOOKS LIKE I HAVE MET MY MATCH.\n");
                print("THANKS FOR PLAYING--I CAN'T WIN ALL THE TIME.\n");
                print("\n");
                break;
            }
            computer_move();
            print("COMPUTER MOVES TO SQUARE " + m + "\n");
            if (m == 158) {
                print("\n");
                print("NICE TRY, BUT IT LOOKS LIKE I HAVE WON.\n");
                print("THANKS FOR PLAYING.\n");
                print("\n");
                break;
            }
            print("WHAT IS YOUR MOVE");
            while (1) {
                m1 = parseInt(await input());
                if (m1 == 0)
                    break;
                t1 = Math.floor(m1 / 10);
                u1 = m1 - 10 * t1;
                p = u1 - u;
                l = t1 - t;
                if (m1 <= m || p == 0 && l <= 0 || p != 0 && l != p && l != 2 * p) {
                    print("\n");
                    print("Y O U   C H E A T . . .  TRY AGAIN");
                    continue;
                }
                break;
            }
            if (m1 == 0) {
                print("\n");
                print("IT LOOKS LIKE I HAVE WON BY FORFEIT.\n");
                print("\n");
                break;
            }
        }
        while (1) {
            print("ANYONE ELSE CARE TO TRY");
            str = await input();
            print("\n");
            if (str == "YES" || str == "NO")
                break;
            print("PLEASE ANSWER 'YES' OR 'NO'.\n");
        }
        if (str != "YES")
            break;
    }
    print("\n");
    print("OK --- THANKS AGAIN.\n");
}

```

这是C++中的一个标准的main函数，用于启动程序的运行。在调用main函数之前，必须保证程序中定义了所有需要的函数，包括main函数本身。

main函数的实现是程序的入口点，当程序运行时，首先会执行这个函数。程序中定义的所有函数，包括main函数，都是在main函数内部定义的。

main函数可以执行程序中的代码，但不会返回任何值。程序的代码可以在main函数中进行初始化，也可以在main函数中进行执行。

对于这段代码，由于缺少程序的具体内容，无法解释它的具体作用。


```
main();

```