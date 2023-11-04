# BasicComputerGames源码解析 42

# `38_Fur_Trader/java/src/FurTraderGame.java`



该代码创建了一个名为FurTraderGame的公共类，该类具有一个名为main的静态方法，其参数为字符串数组args，表示该方法的输入参数。

在main方法的内部，使用new关键字创建了一个FurTrader对象，并调用该对象的play()方法。由于题目没有提供FurTrader类的具体实现，因此无法提供play()方法的详细说明。

可以想象，该代码的作用是启动一个FurTrader游戏，并运行该游戏的主程序。


```
public class FurTraderGame {
    public static void main(String[] args) {

        FurTrader furTrader = new FurTrader();
        furTrader.play();
    }
}

```

# `38_Fur_Trader/java/src/Pelt.java`

这段代码定义了一个名为 "Pelt" 的类，用于存储玩家在这种奇特毛发上的绒毛数量。绒毛数量由一个名为 "name" 的私有整数和另一个名为 "number" 的私有整数组成。

在 "Pelt" 类中，构造函数在玩家使用这种绒毛时初始化这两个私有变量。设置绒毛数量(name)的函数将更新 "number" 变量。获取绒毛数量(number)的函数将返回 "name" 变量的引用。获取并更新玩家名字的函数将在玩家失去一个或多个绒毛时调用，将 "number" 变量设置为 0。


```
/**
 * Pelt object - tracks the name and number of pelts the player has for this pelt type
 */
public class Pelt {

    private final String name;
    private int number;

    public Pelt(String name, int number) {
        this.name = name;
        this.number = number;
    }

    public void setPeltCount(int pelts) {
        this.number = pelts;
    }

    public int getNumber() {
        return this.number;
    }

    public String getName() {
        return this.name;
    }

    public void lostPelts() {
        this.number = 0;
    }
}

```

# `38_Fur_Trader/javascript/furtrader.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。

`print` 函数的作用是在文档中创建一个 `appendChild` 事件，将一个 `createTextNode` 对象（可能是从浏览器的 `textContent` 属性获取到的字符串）追加到文档中的 `output` 元素（可能是 `document` 对象的一个元素）。

`input` 函数的作用是接收一个输入字段（可能是浏览器中的 `INPUT` 元素）和一个字符串（用户输入的文本）。它将接收到的用户输入的文本存储在一个变量 `input_str` 中，然后使用 Promise 对象的一个子函数来处理输入。

Promise 对象的一个子函数 `addEventListener` 用于监听 `input_str` 事件，当用户输入的字符串触发事件时，该事件处理程序会将 `input_str` 的值存储在 `input_str` 变量中，并将其打印到页面上，然后打印一个换行符，最后将 `input_str` 的值存储回来。


```
// FUR TRADER
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

这是一个 JavaScript 函数，名为 `tab`，它的作用是打印出给定字符数组中的元素。函数接受一个参数 `space`，它代表需要在字符串中打印的空格数量。函数内部使用了一个 while 循环，该循环从 0 到 `space`-1（也就是字符数组的长度减一）进行遍历，每次循环将一个空格添加到字符串的末尾。

该函数的作用是打印出给定字符数组中的元素，而不是为该数组创建一个新的字符串。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var f = [];
var bs = [, "MINK", "BEAVER", "ERMINE", "FOX"];

function reset_stats()
{
    for (var j = 1; j <= 4; j++)
        f[j] = 0;
}

```

This appears to be a game where you make a character that can transform into different animals and the goal is to navigate through a series of challenges and trading posts to obtain the best possible fur for your animal. The fur is used to buy supplies, such as food, drink, and eventually other animals. The player must also manage their animal's resources, such as their strength, defense, and特殊 abilities, as well as their budget for trading.

The game features a wide range of animal transformations, including beavers, foxes, ermines, mink, and even an owl. Each animal has different abilities and strengths, and the player must weigh these against their needs to decide which animal to choose.

The player must also manage the finances of their character, deciding how to use their earnings from the challenges and trading posts to buy or sell their animal's furs. Their choices will affect their animal's ability to survive and thrive.

The game also includes a New York City market where the player can trade their animal's furs for supplies, such as food, drink, and eventually other animals. The player must also manage their budget and trade decisions to ensure they are making the best choices for their character.

The game is well-written and has an engaging story line, with a variety of challenging obstacles and interesting characters to keep the player engaged. The player will need to think critically and strategically to navigate the different paths and make the best decisions for their character.


```
// Main program
async function main()
{
    print(tab(31) + "FUR TRADER\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    first_time = true;
    while (1) {
        if (first_time) {
            print("YOU ARE THE LEADER OF A FRENCH FUR TRADING EXPEDITION IN \n");
            print("1776 LEAVING THE LAKE ONTARIO AREA TO SELL FURS AND GET\n");
            print("SUPPLIES FOR THE NEXT YEAR.  YOU HAVE A CHOICE OF THREE\n");
            print("FORTS AT WHICH YOU MAY TRADE.  THE COST OF SUPPLIES\n");
            print("AND THE AMOUNT YOU RECEIVE FOR YOUR FURS WILL DEPEND\n");
            print("ON THE FORT THAT YOU CHOOSE.\n");
            i = 600;
            print("DO YOU WISH TO TRADE FURS?\n");
            first_time = false;
        }
        print("ANSWER YES OR NO\t");
        str = await input();
        if (str == "NO")
            break;
        print("\n");
        print("YOU HAVE $" + i + " SAVINGS.\n");
        print("AND 190 FURS TO BEGIN THE EXPEDITION.\n");
        e1 = Math.floor((0.15 * Math.random() + 0.95) * Math.pow(10, 2) + 0.5) / Math.pow(10, 2);
        b1 = Math.floor((0.25 * Math.random() + 1.00) * Math.pow(10, 2) + 0.5) / Math.pow(10, 2);
        print("\n");
        print("YOUR 190 FURS ARE DISTRIBUTED AMONG THE FOLLOWING\n");
        print("KINDS OF PELTS: MINK, BEAVER, ERMINE AND FOX.\n");
        reset_stats();
        for (j = 1; j <= 4; j++) {
            print("\n");
            print("HOW MANY " + bs[j] + " PELTS DO YOU HAVE\n");
            f[j] = parseInt(await input());
            f[0] = f[1] + f[2] + f[3] + f[4];
            if (f[0] == 190)
                break;
            if (f[0] > 190) {
                print("\n");
                print("YOU MAY NOT HAVE THAT MANY FURS.\n");
                print("DO NOT TRY TO CHEAT.  I CAN ADD.\n");
                print("YOU MUST START AGAIN.\n");
                break;
            }
        }
        if (f[0] > 190) {
            first_time = true;
            continue;
        }
        print("YOU MAY TRADE YOUR FURS AT FORT 1, FORT 2,\n");
        print("OR FORT 3.  FORT 1 IS FORT HOCHELAGA (MONTREAL)\n");
        print("AND IS UNDER THE PROTECTION OF THE FRENCH ARMY.\n");
        print("FORT 2 IS FORT STADACONA (QUEBEC) AND IS UNDER THE\n");
        print("PROTECTION OF THE FRENCH ARMY.  HOWEVER, YOU MUST\n");
        print("MAKE A PORTAGE AND CROSS THE LACHINE RAPIDS.\n");
        print("FORT 3 IS FORT NEW YORK AND IS UNDER DUTCH CONTROL.\n");
        print("YOU MUST CROSS THROUGH IROQUOIS LAND.\n");
        do {
            print("ANSWER 1, 2, OR 3.\n");
            b = parseInt(await input());
            if (b == 1) {
                print("YOU HAVE CHOSEN THE EASIEST ROUTE.  HOWEVER, THE FORT\n");
                print("IS FAR FROM ANY SEAPORT.  THE VALUE\n");
                print("YOU RECEIVE FOR YOUR FURS WILL BE LOW AND THE COST\n");
                print("OF SUPPLIES HIGHER THAN AT FORTS STADACONA OR NEW YORK.\n");
            } else if (b == 2) {
                print("YOU HAVE CHOSEN A HARD ROUTE.  IT IS, IN COMPARSION,\n");
                print("HARDER THAN THE ROUTE TO HOCHELAGA BUT EASIER THAN\n");
                print("THE ROUTE TO NEW YORK.  YOU WILL RECEIVE AN AVERAGE VALUE\n");
                print("FOR YOUR FURS AND THE COST OF YOUR SUPPLIES WILL BE AVERAGE.\n");
            } else {
                print("YOU HAVE CHOSEN THE MOST DIFFICULT ROUTE.  AT\n");
                print("FORT NEW YORK YOU WILL RECEIVE THE HIGHEST VALUE\n");
                print("FOR YOUR FURS.  THE COST OF YOUR SUPPLIES\n");
                print("WILL BE LOWER THAN AT ALL THE OTHER FORTS.\n");
            }
            if (b >= 1 && b <= 3) {
                print("DO YOU WANT TO TRADE AT ANOTHER FORT?\n");
                print("ANSWER YES OR NO\t");
                str = await input();
                if (str == "YES") {
                    b = 0;
                }
            }
        } while (b < 1 || b > 3) ;
        show_beaver = true;
        show_all = true;
        if (b == 1) {
            i -= 160;
            print("\n");
            m1 = Math.floor((0.2 * Math.random() + 0.7) * Math.pow(10, 2) + 0.5) / Math.pow(10, 2);
            e1 = Math.floor((0.2 * Math.random() + 0.65) * Math.pow(10, 2) + 0.5) / Math.pow(10, 2);
            b1 = Math.floor((0.2 * Math.random() + 0.75) * Math.pow(10, 2) + 0.5) / Math.pow(10, 2);
            d1 = Math.floor((0.2 * Math.random() + 0.8) * Math.pow(10, 2) + 0.5) / Math.pow(10, 2);
            print("SUPPLIES AT FORT HOCHELAGA COST $150.00.\n");
            print("YOUR TRAVEL EXPENSES TO HOCHELAGA WERE $10.00.\n");
        } else if (b == 2) {
            i -= 140;
            print("\n");
            m1 = Math.floor((0.3 * Math.random() + 0.85) * Math.pow(10, 2) + 0.5) / Math.pow(10, 2);
            e1 = Math.floor((0.15 * Math.random() + 0.8) * Math.pow(10, 2) + 0.5) / Math.pow(10, 2);
            b1 = Math.floor((0.2 * Math.random() + 0.9) * Math.pow(10, 2) + 0.5) / Math.pow(10, 2);
            p = Math.floor(10 * Math.random()) + 1;
            if (p <= 2) {
                f[2] = 0;
                print("YOUR BEAVER WERE TOO HEAVY TO CARRY ACROSS\n");
                print("THE PORTAGE.  YOU HAD TO LEAVE THE PELTS, BUT FOUND\n");
                print("THEM STOLEN WHEN YOU RETURNED.\n");
                show_beaver = false;
            } else if (p <= 6) {
                print("YOU ARRIVED SAFELY AT FORT STADACONA.\n");
            } else if (p <= 8) {
                reset_stats();
                print("YOUR CANOE UPSET IN THE LACHINE RAPIDS.  YOU\n");
                print("LOST ALL YOUR FURS.\n");
                show_all = false;
            } else if (p <= 10) {
                f[4] = 0;
                print("YOUR FOX PELTS WERE NOT CURED PROPERLY.\n");
                print("NO ONE WILL BUY THEM.\n");
            }
            print("SUPPLIES AT FORT STADACONA COST $125.00.\n");
            print("YOUR TRAVEL EXPENSES TO STADACONA WERE $15.00.\n");

            d1 = Math.floor((0.2 * Math.random() + 0.8) * Math.pow(10, 2) + 0.5) / Math.pow(10, 2);
        } else if (b == 3) {
            i -= 105;
            print("\n");
            m1 = Math.floor((0.15 * Math.random() + 1.05) * Math.pow(10, 2) + 0.5) / Math.pow(10, 2);
            d1 = Math.floor((0.25 * Math.random() + 1.1) * Math.pow(10, 2) + 0.5) / Math.pow(10, 2);
            p = Math.floor(10 * Math.random()) + 1;
            if (p <= 2) {
                print("YOU WERE ATTACKED BY A PARTY OF IROQUOIS.\n");
                print("ALL PEOPLE IN YOUR TRADING GROUP WERE\n");
                print("KILLED.  THIS ENDS THE GAME.\n");
                break;
            } else if (p <= 6) {
                print("YOU WERE LUCKY.  YOU ARRIVED SAFELY\n");
                print("AT FORT NEW YORK.\n");
            } else if (p <= 8) {
                reset_stats();
                print("YOU NARROWLY ESCAPED AN IROQUOIS RAIDING PARTY.\n");
                print("HOWEVER, YOU HAD TO LEAVE ALL YOUR FURS BEHIND.\n");
                show_all = false;
            } else if (p <= 10) {
                b1 /= 2;
                m1 /= 2;
                print("YOUR MINK AND BEAVER WERE DAMAGED ON YOUR TRIP.\n");
                print("YOU RECEIVE ONLY HALF THE CURRENT PRICE FOR THESE FURS.\n");
            }
            print("SUPPLIES AT NEW YORK COST $80.00.\n");
            print("YOUR TRAVEL EXPENSES TO NEW YORK WERE $25.00.\n");
        }
        print("\n");
        if (show_all) {
            if (show_beaver)
                print("YOUR BEAVER SOLD FOR $" + b1 * f[2] + " ");
            print("YOUR FOX SOLD FOR $" + d1 * f[4] + "\n");
            print("YOUR ERMINE SOLD FOR $" + e1 * f[3] + " ");
            print("YOUR MINK SOLD FOR $" + m1 * f[1] + "\n");
        }
        i += m1 * f[1] + b1 * f[2] + e1 * f[3] + d1 * f[4];
        print("\n");
        print("YOU NOW HAVE $" + i + " INCLUDING YOUR PREVIOUS SAVINGS\n");
        print("\n");
        print("DO YOU WANT TO TRADE FURS NEXT YEAR?\n");
    }
}

```

这道题目缺少上下文，无法给出具体的解释。通常来说，在编程中， `main()` 函数是程序的入口点，程序从此处开始执行。它的作用是启动程序，告诉操作系统程序要开始执行哪些代码。对于命令行程序，用户通过命令行指定程序的入口点，例如 `python myprogram.py`，其中 `myprogram.py` 是程序的文件名， `main()` 函数就是程序的入口点。


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

You can answer yes/no questions in lower case if desired.


# `38_Fur_Trader/python/furtrader.py`

这段代码是一个Python脚本，主要作用是让玩家拥有货币和商品。具体来说，它实现了以下功能：

1. 导入random和sys模块，用于生成随机数和系统功能。
2. 从typing模块中创建一个列表类型，用于存储玩家的资金和商品。
3. 在全局变量中，声明了玩家资金和商品列表的头三个元素为0，以便初始化。
4. 通过FUR_MINK、FUR_BEAVER和FUR_ERMINE常量，对货币和商品进行类型转换，以便与其他玩家进行交易时使用。
5. 在游戏主循环中，提取玩家的资金，并将商品列表中的元素分别乘以3，以便在交易时也能进行3倍增长。
6. 创建了一个货币交易类，实现了玩家可以用货币购买商品或者向其他玩家出售商品获得货币的功能。
7. 在游戏循环中，让玩家选择购买商品或者出售商品，并随机从玩家资金中选择货币，实现了货币系统和随机购买商品功能。


```
#!/usr/bin/env python3

import random  # for generating random numbers
import sys  # for system function, like exit()
from typing import List

# global variables for storing player's status
player_funds: float = 0  # no money
player_furs = [0, 0, 0, 0]  # no furs


# Constants
FUR_MINK = 0
FUR_BEAVER = 1
FUR_ERMINE = 2
```

这段代码定义了四个变量，以及一个函数 `show_introduction()`。函数的作用是输出一段介绍性的信息，向玩家介绍 French fur trading expedition 的情况。

变量包括：

- `FUR_FOX`：表示 fur 宠物的名称，这里包括 "MINK"、"BEAVER"、"ERMINE" 和 "FOX"。
- `MAX_FURS`：表示可以购买的最大 fur 宠物的数量，这里设置为 190。
- `FUR_NAMES`：表示 fur 宠物的列表，这里仅包含 "MINK"、"BEAVER"、"ERMINE" 和 "FOX"。
- `FORT_MONTREAL`、`FORT_QUEBEC` 和 `FORT_NEWYORK`：表示可以交易的四个堡垒，它们的名称为 "HOCHELAGA (MONTREAL)"、"STADACONA (QUEBEC)" 和 "NEW YORK"。

函数 `show_introduction()` 的作用是显示上述介绍性的信息，向玩家介绍 fur trading expedition 的情况。函数没有其他明显的功能。


```
FUR_FOX = 3
MAX_FURS = 190
FUR_NAMES = ["MINK", "BEAVER", "ERMINE", "FOX"]

FORT_MONTREAL = 1
FORT_QUEBEC = 2
FORT_NEWYORK = 3
FORT_NAMES = ["HOCHELAGA (MONTREAL)", "STADACONA (QUEBEC)", "NEW YORK"]


def show_introduction() -> None:
    """Show the player the introductory message"""
    print("YOU ARE THE LEADER OF A FRENCH FUR TRADING EXPEDITION IN ")
    print("1776 LEAVING THE LAKE ONTARIO AREA TO SELL FURS AND GET")
    print("SUPPLIES FOR THE NEXT YEAR.  YOU HAVE A CHOICE OF THREE")
    print("FORTS AT WHICH YOU MAY TRADE.  THE COST OF SUPPLIES")
    print("AND THE AMOUNT YOU RECEIVE FOR YOUR FURS WILL DEPEND")
    print("ON THE FORT THAT YOU CHOOSE.")
    print()


```

这段代码是一个名为 `get_fort_choice()` 的函数，用于向玩家显示在 Fort 中可用的选择，并等待玩家输入。如果玩家的输入有效且为 1、2 或 3，函数将返回所选的选项。否则，函数将继续提示玩家。

具体来说，函数会在屏幕上循环多次输出不同 Fort 设施的选择提示，包括 Fort 1、Fort 2 和 Fort 3。每次输出时，函数将不同的选择与对应的 Fort 设施和转移方式打印出来。在玩家输入有效选项后，函数将读取玩家的输入并尝试将其转换为整数。如果转换成功，函数将返回所选选项的整数表示。否则，函数将继续循环提示玩家。


```
def get_fort_choice() -> int:
    """Show the player the choices of Fort, get their input, if the
    input is a valid choice (1,2,3) return it, otherwise keep
    prompting the user."""
    result = 0
    while result == 0:
        print()
        print("YOU MAY TRADE YOUR FURS AT FORT 1, FORT 2,")
        print("OR FORT 3.  FORT 1 IS FORT HOCHELAGA (MONTREAL)")
        print("AND IS UNDER THE PROTECTION OF THE FRENCH ARMY.")
        print("FORT 2 IS FORT STADACONA (QUEBEC) AND IS UNDER THE")
        print("PROTECTION OF THE FRENCH ARMY.  HOWEVER, YOU MUST")
        print("MAKE A PORTAGE AND CROSS THE LACHINE RAPIDS.")
        print("FORT 3 IS FORT NEW YORK AND IS UNDER DUTCH CONTROL.")
        print("YOU MUST CROSS THROUGH IROQUOIS LAND.")
        print("ANSWER 1, 2, OR 3.")

        player_choice = input(">> ")  # get input from the player

        # try to convert the player's string input into an integer
        try:
            result = int(player_choice)  # string to integer
        except Exception:
            # Whatever the player typed, it could not be interpreted as a number
            pass

    return result


```

这段代码是一个名为 `show_fort_comment` 的函数，它接受一个参数 `which_fort`。函数内部通过 `print` 函数输出关于该 Fort 的描述，然后根据 `which_fort` 的值，在描述中补充关于不同 Fort 的信息。

具体来说，如果 `which_fort` 是 `FORTCOMMAND`，那么函数将输出类似以下内容的信息：
```yaml
YOU HAVE CHOSEN THE EASIEST ROUTE.  HOWEVER, THE FORT IS FAR FROM ANY SEAPORT.  THE VALUE YOU RECEIVE FOR YOUR FURS WILL BE LOW AND THE COST OF SUPPLIES HIGHER THAN AT FORTS STADACONA OR NEW YORK.
```
如果 `which_fort` 是 `FORTCOMPATTROMETRY`，那么函数将输出类似以下内容的信息：
```makefile
YOU HAVE CHOSEN A HARD ROUTE.  IT IS, IN COMPLICATION, HARDER THAN THE ROUTE TO HOCHELAGA BUT EASIER THAN THE ROUTE TO NEW YORK.  YOU WILL RECEIVE AN AVERAGE VALUE FOR YOUR FURS AND THE COST OF YOUR SUPPLIES WILL BE AVERAGE.
```
如果 `which_fort` 是 `FORTCOMMAND`，那么函数将输出类似以下内容的信息：
```sql
YOU HAVE CHOSEN THE MOST DIFFICULAR ROUTE.  AT FORT NEW YORK YOU WILL RECEIVE THE HIGHEST VALUE FOR YOUR FURS.  THE COST OF YOUR SUPPLIES WILL BE LOWER THAN AT ALL THE OTHER FORTS.
```
然而，如果 `which_fort` 不存在，函数将通过 `print("Internal error #1, fort " + str(which_fort) + " does not exist")` 和 `sys.exit(1)` 来输出错误信息并退出程序。


```
def show_fort_comment(which_fort) -> None:
    """Print the description for the fort"""
    print()
    if which_fort == FORT_MONTREAL:
        print("YOU HAVE CHOSEN THE EASIEST ROUTE.  HOWEVER, THE FORT")
        print("IS FAR FROM ANY SEAPORT.  THE VALUE")
        print("YOU RECEIVE FOR YOUR FURS WILL BE LOW AND THE COST")
        print("OF SUPPLIES HIGHER THAN AT FORTS STADACONA OR NEW YORK.")
    elif which_fort == FORT_QUEBEC:
        print("YOU HAVE CHOSEN A HARD ROUTE.  IT IS, IN COMPARSION,")
        print("HARDER THAN THE ROUTE TO HOCHELAGA BUT EASIER THAN")
        print("THE ROUTE TO NEW YORK.  YOU WILL RECEIVE AN AVERAGE VALUE")
        print("FOR YOUR FURS AND THE COST OF YOUR SUPPLIES WILL BE AVERAGE.")
    elif which_fort == FORT_NEWYORK:
        print("YOU HAVE CHOSEN THE MOST DIFFICULT ROUTE.  AT")
        print("FORT NEW YORK YOU WILL RECEIVE THE HIGHEST VALUE")
        print("FOR YOUR FURS.  THE COST OF YOUR SUPPLIES")
        print("WILL BE LOWER THAN AT ALL THE OTHER FORTS.")
    else:
        print("Internal error #1, fort " + str(which_fort) + " does not exist")
        sys.exit(1)  # you have a bug
    print()


```

这段代码是一个函数 `get_yes_or_no()`，它接受一个字符串类型的参数并返回一个字符。

函数的作用是询问玩家输入 "YES" 或 "NO"，并不断 prompt，直到收到有效的输入为止。在玩家输入时，函数会检查输入是否以 "Y" 或 "N" 开头，如果是，则返回相应的结果。如果未输入，则一直 prompt。

函数的实现较为简单，主要实现了输入的有效性校验，以及结果的返回。


```
def get_yes_or_no() -> str:
    """Prompt the player to enter 'YES' or 'NO'. Keep prompting until
    valid input is entered.  Accept various spellings by only
    checking the first letter of input.
    Return a single letter 'Y' or 'N'"""
    result = ""
    while result not in ("Y", "N"):
        print("ANSWER YES OR NO")
        player_choice = input(">> ")
        player_choice = player_choice.strip().upper()  # trim spaces, make upper-case
        if player_choice.startswith("Y"):
            result = "Y"
        elif player_choice.startswith("N"):
            result = "N"
    return result


```

这段代码的作用是提示玩家购买毛皮，并获取购买数量。它具有以下特点：

1.它会提示玩家输入毛皮种类，如Mink、Beaver、ermine 和 Fox。
2.在玩家输入错误数量时，它会再次提示玩家输入正确的数量，并重新循环以确保所有 fur 种类都被购买。
3.它返回购买的 fur 数量列表。


```
def get_furs_purchase() -> List[int]:
    """Prompt the player for how many of each fur type they want.
    Accept numeric inputs, re-prompting on incorrect input values"""
    results: List[int] = []

    print("YOUR " + str(MAX_FURS) + " FURS ARE DISTRIBUTED AMONG THE FOLLOWING")
    print("KINDS OF PELTS: MINK, BEAVER, ERMINE AND FOX.")
    print()

    while len(results) < len(FUR_NAMES):
        print(f"HOW MANY {FUR_NAMES[len(results)]} DO YOU HAVE")
        count_str = input(">> ")
        try:
            count = int(count_str)
            results.append(count)
        except Exception:  # invalid input, prompt again by re-looping
            pass
    return results


```

This is a Python module that simulates a trading game. The game allows the player to trade furs (beaver, fox, ermine, and mink) for money. The player can also calculate the value of their furs and keep track of their savings.

The `game_state` variable is set to "trading" at the beginning of the game. If the player wants to trade furs, the `should_trade` variable is set to `True` and the game loops to check if the player wants to trade. If the player does not want to trade, the game loops to calculate the player's savings.

If the player wants to trade furs, the game calculates the value of the furs and updates the player's savings accordingly. If the player runs out of furs, the game prints an error message and returns.

The `get_yes_or_no()` function returns a boolean value indicating whether the player wants to trade furs or not.

Note that this is a very basic game and could be improved in many ways, such as adding more game logic, creating a more realistic trading system, and allowing the player to choose from a wider range of furs.


```
def main() -> None:
    print(" " * 31 + "FUR TRADER")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print(" " * 15 + "(Ported to Python Oct 2012 krt@krt.com.au)")
    print("\n\n\n")

    game_state = "starting"
    fox_price = None  # sometimes this takes the "last" price (probably this was a bug)

    while True:

        if game_state == "starting":
            show_introduction()

            player_funds: float = 600  # Initial player start money
            player_furs = [0, 0, 0, 0]  # Player fur inventory

            print("DO YOU WISH TO TRADE FURS?")
            should_trade = get_yes_or_no()
            if should_trade == "N":
                sys.exit(0)  # STOP
            game_state = "trading"

        elif game_state == "trading":
            print()
            print("YOU HAVE $ %1.2f IN SAVINGS" % (player_funds))
            print("AND " + str(MAX_FURS) + " FURS TO BEGIN THE EXPEDITION")
            player_furs = get_furs_purchase()

            if sum(player_furs) > MAX_FURS:
                print()
                print("YOU MAY NOT HAVE THAT MANY FURS.")
                print("DO NOT TRY TO CHEAT.  I CAN ADD.")
                print("YOU MUST START AGAIN.")
                game_state = "starting"  # T/N: Wow, harsh.
            else:
                game_state = "choosing fort"

        elif game_state == "choosing fort":
            which_fort = get_fort_choice()
            show_fort_comment(which_fort)
            print("DO YOU WANT TO TRADE AT ANOTHER FORT?")
            change_fort = get_yes_or_no()
            if change_fort == "N":
                game_state = "travelling"

        elif game_state == "travelling":
            print()
            if which_fort == FORT_MONTREAL:
                mink_price = (
                    int((0.2 * random.random() + 0.70) * 100 + 0.5) / 100
                )  # INT((.2*RND(1)+.7)*10^2+.5)/10^2
                ermine_price = (
                    int((0.2 * random.random() + 0.65) * 100 + 0.5) / 100
                )  # INT((.2*RND(1)+.65)*10^2+.5)/10^2
                beaver_price = (
                    int((0.2 * random.random() + 0.75) * 100 + 0.5) / 100
                )  # INT((.2*RND(1)+.75)*10^2+.5)/10^2
                fox_price = (
                    int((0.2 * random.random() + 0.80) * 100 + 0.5) / 100
                )  # INT((.2*RND(1)+.8)*10^2+.5)/10^2

                print("SUPPLIES AT FORT HOCHELAGA COST $150.00.")
                print("YOUR TRAVEL EXPENSES TO HOCHELAGA WERE $10.00.")
                player_funds -= 160

            elif which_fort == FORT_QUEBEC:
                mink_price = (
                    int((0.30 * random.random() + 0.85) * 100 + 0.5) / 100
                )  # INT((.3*RND(1)+.85)*10^2+.5)/10^2
                ermine_price = (
                    int((0.15 * random.random() + 0.80) * 100 + 0.5) / 100
                )  # INT((.15*RND(1)+.8)*10^2+.5)/10^2
                beaver_price = (
                    int((0.20 * random.random() + 0.90) * 100 + 0.5) / 100
                )  # INT((.2*RND(1)+.9)*10^2+.5)/10^2
                fox_price = (
                    int((0.25 * random.random() + 1.10) * 100 + 0.5) / 100
                )  # INT((.25*RND(1)+1.1)*10^2+.5)/10^2
                event_picker = int(10 * random.random()) + 1

                if event_picker <= 2:
                    print("YOUR BEAVER WERE TOO HEAVY TO CARRY ACROSS")
                    print("THE PORTAGE.  YOU HAD TO LEAVE THE PELTS, BUT FOUND")
                    print("THEM STOLEN WHEN YOU RETURNED.")
                    player_furs[FUR_BEAVER] = 0
                elif event_picker <= 6:
                    print("YOU ARRIVED SAFELY AT FORT STADACONA.")
                elif event_picker <= 8:
                    print("YOUR CANOE UPSET IN THE LACHINE RAPIDS.  YOU")
                    print("LOST ALL YOUR FURS.")
                    player_furs = [0, 0, 0, 0]
                elif event_picker <= 10:
                    print("YOUR FOX PELTS WERE NOT CURED PROPERLY.")
                    print("NO ONE WILL BUY THEM.")
                    player_furs[FUR_FOX] = 0
                else:
                    print(
                        "Internal Error #3, Out-of-bounds event_picker"
                        + str(event_picker)
                    )
                    sys.exit(1)  # you have a bug

                print()
                print("SUPPLIES AT FORT STADACONA COST $125.00.")
                print("YOUR TRAVEL EXPENSES TO STADACONA WERE $15.00.")
                player_funds -= 140

            elif which_fort == FORT_NEWYORK:
                mink_price = (
                    int((0.15 * random.random() + 1.05) * 100 + 0.5) / 100
                )  # INT((.15*RND(1)+1.05)*10^2+.5)/10^2
                ermine_price = (
                    int((0.15 * random.random() + 0.95) * 100 + 0.5) / 100
                )  # INT((.15*RND(1)+.95)*10^2+.5)/10^2
                beaver_price = (
                    int((0.25 * random.random() + 1.00) * 100 + 0.5) / 100
                )  # INT((.25*RND(1)+1.00)*10^2+.5)/10^2
                if fox_price is None:
                    # Original Bug?  There is no Fox price generated for New York, it will use any previous "D1" price
                    # So if there was no previous value, make one up
                    fox_price = (
                        int((0.25 * random.random() + 1.05) * 100 + 0.5) / 100
                    )  # not in orginal code
                event_picker = int(10 * random.random()) + 1

                if event_picker <= 2:
                    print("YOU WERE ATTACKED BY A PARTY OF IROQUOIS.")
                    print("ALL PEOPLE IN YOUR TRADING GROUP WERE")
                    print("KILLED.  THIS ENDS THE GAME.")
                    sys.exit(0)
                elif event_picker <= 6:
                    print("YOU WERE LUCKY.  YOU ARRIVED SAFELY")
                    print("AT FORT NEW YORK.")
                elif event_picker <= 8:
                    print("YOU NARROWLY ESCAPED AN IROQUOIS RAIDING PARTY.")
                    print("HOWEVER, YOU HAD TO LEAVE ALL YOUR FURS BEHIND.")
                    player_furs = [0, 0, 0, 0]
                elif event_picker <= 10:
                    mink_price /= 2
                    fox_price /= 2
                    print("YOUR MINK AND BEAVER WERE DAMAGED ON YOUR TRIP.")
                    print("YOU RECEIVE ONLY HALF THE CURRENT PRICE FOR THESE FURS.")
                else:
                    print(
                        "Internal Error #4, Out-of-bounds event_picker"
                        + str(event_picker)
                    )
                    sys.exit(1)  # you have a bug

                print()
                print("SUPPLIES AT NEW YORK COST $85.00.")
                print("YOUR TRAVEL EXPENSES TO NEW YORK WERE $25.00.")
                player_funds -= 105

            else:
                print("Internal error #2, fort " + str(which_fort) + " does not exist")
                sys.exit(1)  # you have a bug

            # Calculate sales
            beaver_value = beaver_price * player_furs[FUR_BEAVER]
            fox_value = fox_price * player_furs[FUR_FOX]
            ermine_value = ermine_price * player_furs[FUR_ERMINE]
            mink_value = mink_price * player_furs[FUR_MINK]

            print()
            print("YOUR BEAVER SOLD FOR $%6.2f" % (beaver_value))
            print("YOUR FOX SOLD FOR    $%6.2f" % (fox_value))
            print("YOUR ERMINE SOLD FOR $%6.2f" % (ermine_value))
            print("YOUR MINK SOLD FOR   $%6.2f" % (mink_value))

            player_funds += beaver_value + fox_value + ermine_value + mink_value

            print()
            print(
                "YOU NOW HAVE $ %1.2f INCLUDING YOUR PREVIOUS SAVINGS" % (player_funds)
            )

            print()
            print("DO YOU WANT TO TRADE FURS NEXT YEAR?")
            should_trade = get_yes_or_no()
            if should_trade == "N":
                sys.exit(0)  # STOP
            else:
                game_state = "trading"


```

这段代码是一个Python程序中的一个if语句。if语句是Python中的一种条件语句，用于检查一个条件是否为真，如果是真，则执行if语句内部的代码，否则跳过if语句。

在这段if语句中，条件判断符为`__name__ == "__main__"`。这里，`__name__`是一个保留字，用于防止程序被其它文件中的同名函数或类所迷惑。它的作用是在程序运行时判断当前文件是否与`__main__.py`文件名相同。如果当前文件与`__main__.py`文件名相同，则执行if语句内部的代码，否则跳过if语句。

if语句内部的代码为`main()`，它表示一个函数，用于执行当前程序的main部分。因此，如果当前文件名为`__main__.py`，则程序将首先执行`main()`函数，然后才执行if语句外的代码。


```
if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)

##### Translator Notes:
I tried to preserve as much of the original layout and flow of the code
as possible.  However I did use quasi enumerated types for the Fort numbers
and Fur types.  I think this was certainly a change for the better, and
makes the code much easier to read.

I program in many different languages on a daily basis.  Most languages
require brackets around expressions, so I just cannot bring myself to
write an expression without brackets.  IMHO it makes the code easier to study,
but it does contravene the Python PEP-8 Style guide.

Interestingly the code seems to have a bug around the prices of Fox Furs.
The commodity-rate for these is stored in the variable `D1`, however some
paths through the code do not set this price.  So there was a chance of
using this uninitialised, or whatever the previous loop set.  I don't
think this was the original authors intent.  So I preserved the original flow
of the code (using the previous `D1` value), but also catching the
uninitialised path, and assigning a "best guess" value.

krt@krt.com.au 2020-10-10


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Golf

This is a single player golf game. In other words it’s you against the golf course (the computer). The program asks for your handicap (maximum of 30) and your area of difficulty. You have a bag of 29 clubs plus a putter. On the course you have to contend with rough, trees, on and off fairway, sand traps, and water hazards. In addition, you can hook, slice, go out of bounds, or hit too far. On putting, you determine the potency factor (or percent of swing). Until you get the swing of the game (no pun intended), you’ll probably was to use a fairly high handicap.

Steve North of Creative Computing modified the original version of this game, the author of which is unknown.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=71)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=86)


Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Known Bugs

- The weakness numbers printed in the original BASIC program are wrong.  It says 4=TRAP SHOTS, 5=PUTTING, but in the code, trap shots and putting are 3 and 4, respectively.

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `39_Golf/csharp/Program.cs`

这是一个使用ASCII艺术的代码片段，其中包含了多种字符和特殊字符。具体来说，这段代码是一个自定义的ASCII图形，包含了一个"GOLF"字符串，由"8"字符组成。

在代码中，第一个"8"字符代表一个空格，第二个"8"字符代表一个句号，第三个"8"字符代表一个引号，然后紧跟着的四个"8"字符再次代表一个空格。接下来的三个"8"字符和一个双引号中的"8"字符代表一个反引号，最后一个"8"字符代表一个感叹号。

总体而言，这段代码看起来像是一个简单的ASCII图形，可能是用于在命令行或控制台中显示某种信息。但是，由于缺乏上下文和具体目的，无法确定它的实际用途和意义。


```
﻿//
//          8""""8 8"""88 8     8""""
//          8    " 8    8 8     8
//          8e     8    8 8e    8eeee
//          88  ee 8    8 88    88
//          88   8 8    8 88    88
//          88eee8 8eeee8 88eee 88
//
// GOLF
//
// C#
// .NET Core
// TargetFramework: netcoreapp 3.1
//
// Run source:
```

这段代码是一个用于生成随机洞穴棋(Holesite Game)的代码。它使用了.NET framework，在命令行中运行了一个名为“INDEX”的出版工具。该出版工具提供了两种不同的编译选项，分别用于在Linux和Windows系统上生成游戏。

具体来说，该代码编译生成了两个可执行文件：“dotnet publish --self-contained -c Release -r linux-x64 /p:PublishSingleFile=true /p:PublishTrimmed=true”和“dotnet publish -r win-x64 -c Release /p:PublishSingleFile=true”。其中，第一个选项是在Linux系统上使用“Release”配置目标，第二个选项是在Windows系统上使用“Release”配置目标。

此外，该代码还包括一些用于生成游戏棋盘的函数，如“NewHole”、“TeeUp”和“Stroke”，以及用于生成随机洞穴棋的构造函数“HolesiteGame”。


```
// dotnet run
//
// Linux compile:
// dotnet publish --self-contained -c Release -r linux-x64 /p:PublishSingleFile=true /p:PublishTrimmed=true
//
// Windows compile:
// dotnet publish -r win-x64 -c Release /p:PublishSingleFile=true
//
//
// INDEX
// ----------------- methods
// constructor
// NewHole
// TeeUp
// Stroke
```

这段代码是一个高尔夫球场景中的AI（人工智能）脚本，帮助高尔夫球爱好者（或AI）进行高尔夫球挥杆和路径分析。以下是这段代码的一些关键部分的功能：

1. `InterpretResults`：这是一个结果解释函数，用于评估用户的高尔夫球挥杆表现。
2. `ReportCurrentScore`：这个函数将当前用户的得分报告给用户。
3. `ScoreCardNewHole`：创建一个新的得分卡（score card）记录用户的历史得分。
4. `ScoreCardRecordStroke`：记录用户最新的高尔夫球挥杆击球成绩（如：分数）。
5. `ScoreCardGetPreviousStroke`：获取用户之前的高尔夫球挥杆成绩（即：上次的得分）。
6. `ScoreCardGetTotal`：计算用户在高尔夫球挥杆过程中总共得分。
7. `Ask`：向用户询问是否要进行新的高尔夫球挥杆击球。
8. `Wait`：等待用户回答。


```
// PlotBall
// InterpretResults
// ReportCurrentScore
// FindBall
// IsOnFairway
// IsOnGreen
// IsInHazard
// IsInRough
// IsOutOfBounds
// ScoreCardNewHole
// ScoreCardRecordStroke
// ScoreCardGetPreviousStroke
// ScoreCardGetTotal
// Ask
// Wait
```

这是一个二进制数据类，名为"HoleGeometry"，可能用于在游戏引擎中模拟一个 golf 游戏的洞穴。

它包括以下几个类：

* "HoleInfo"，可能是用于记录一个洞穴信息的类。
* "CircleGameObj"，可能是用于表示一个洞穴对象的类。
* "RectGameObj"，可能是用于表示一个洞穴区域的类。
* "HoleGeometry"，可能是用于表示洞穴几何信息的类。
* "Plot"，可能是用于在游戏界面中显示洞穴信息的类。
* "GetDistance"，可能是用于计算两个点之间距离的类。
* "IsInRectangle"，可能是用于检查一个点是否在 rectangle 内的类。

该代码的作用是提供一个方便的方式来在游戏引擎中模拟洞穴。通过使用这些类，可以轻松地创建、操作和显示洞穴。


```
// ReviewBag
// Quit
// GameOver
// ----------------- DATA
// Clubs
// CourseInfo
// ----------------- classes
// HoleInfo
// CircleGameObj
// RectGameObj
// HoleGeometry
// Plot
// ----------------- helper methods
// GetDistance
// IsInRectangle
```

这段代码是一个基于简单几何的模拟程序，用于在文字 Based 游戏中模拟球赛。它模拟了球场的基本形状，包括外围绕成的 5 码 rough，以及位于球门前的一圆环形绿色。

球赛中的公平场地是画有白色界定的矩形区域，边长为 40 码。球门位于（0，0）的位置，总是以球形周围的 10 码为半径画一个圆形区域。

为了模拟不同类型高尔夫球场的实际情况，该程序使用了基于实际高尔夫球场数据统计的计算方法。因此，该代码中存在很多随机性，业务规则和运气成分，从而影响了游戏的玩法。


```
// ToRadians
// ToDegrees360
// Odds
//
//  Despite being a text based game, the code uses simple geometry to simulate a course.
//  Fairways are 40 yard wide rectangles, surrounded by 5 yards of rough around the perimeter.
//  The green is a circle of 10 yards radius around the cup.
//  The cup is always at point (0,0).
//
//  Using basic trigonometry we can plot the ball's location using the distance of the stroke and
//  and the angle of deviation (hook/slice).
//
//  The stroke distances are based on real world averages of different club types.
//  Lots of randomization, "business rules", and luck influence the game play.
//  Probabilities are commented in the code.
```

这段代码定义了一个名为 "note" 的字符串变量，包含三个字符串对象 "courseInfo"、"clubs" 和 "scoreCard"，每个对象都包含一个空对象 "{}"，具有索引号 1。

该代码使用三元运算符将三个字符串对象连接起来，生成一个新的字符串对象 "courseInfoClubsScoreCard"。这个新对象将包含三个空对象，分别代表 "courseInfo"、"clubs" 和 "scoreCard" 对象中的值。

最后，代码使用 "console.log()" 函数将新生成的字符串对象输出到控制台。


```
//
//  note: 'courseInfo', 'clubs', & 'scoreCard' arrays each include an empty object so indexing
//  can begin at 1. Like all good programmers we count from zero, but in this context,
//  it's more natural when hole number one is at index one
//
//
//     |-----------------------------|
//     |            rough            |
//     |   ----------------------    |
//     |   |                     |   |
//     | r |        =  =         | r |
//     | o |     =        =      | o |
//     | u |    =    .     =     | u |
//     | g |    =   green  =     | g |
//     | h |     =        =      | h |
```

这是一个 C 语言的代码，定义了一个名为 "Fairway" 的函数。函数的作用是计算给定整数 "x" 辗转赋值后的结果，即从初始值开始，对于每个步长 "i"，将变量 "x" 乘以 2 再加上 "i"，并输出计算出的结果。

代码中定义了一个名为 "|" 的宏，它的作用是将定义的变量 "x" 和宏 "i" 拼接成一个新的字符串，用于输出计算出的结果。

函数体中首先定义了一个名为 "|" 的宏，它的作用是将定义的变量 "x" 和宏 "i" 拼接成一个新的字符串，用于输出计算出的结果。然后计算变量 "x" 乘以 2 并加上 "i"，使用循环从 0 到给定的变量 "x" 长度 - 1 步长 "i" 进行循环，每次循环将变量 "x" 的值乘以 2 并加上 "i"。最后，将循环计算得到的计算结果输出。


```
//     |   |        =  =         |   |
//     |   |                     |   |
//     |   |                     |   |
//     |   |      Fairway        |   |
//     |   |                     |   |
//     |   |               ------    |
//     |   |            --        -- |
//     |   |           --  hazard  --|
//     |   |            --        -- |
//     |   |               ------    |
//     |   |                     |   |
//     |   |                     |   |   out
//     |   |                     |   |   of
//     |   |                     |   |   bounds
//     |   |                     |   |
```

这段代码是一个用于计算高尔夫球杆位置的程序。它包含以下输出：

```
//     |   |                     |   |
//     |          20-30 yards           |   |
//     |                                 |   |
//     |        5 yards past green          |   |
//     |                                     |   |
//     |           rough perimeter          |   |
//     |                                     |   |
//     |  -----------------------------------|   |
//     |                                         |   |
//     |          25 degrees off line           |   |
//     |                                         |   |
//     |    Hook: positive degrees           |   |
//     |---------------------------------------|   |
//     |   |                                     |   |
//     |   |                        5 yards      |   |
//     |   |                                     |   |
//     |   |          new position of the ball |   |
//     |   |---------------------------------------|   |
```

这段代码接受一个包含四个值的参数：球的位置（x, y, z，单位：英尺或码）、球与参考线（可以是东、南、西或北）和球的角度（以度为单位）。它计算出球在四氟乙烯高尔夫球场上的位置，然后返回该位置。


```
//     |   |                     |   |
//     |            tee              |
//
//
//  Typical green size: 20-30 yards
//  Typical golf course fairways are 35 to 45 yards wide
//  Our fairway extends 5 yards past green
//  Our rough is a 5 yard perimeter around fairway
//
//  We calculate the new position of the ball given the ball's point, the distance
//  of the stroke, and degrees off line (hook or slice).
//
//  Degrees off (for a right handed golfer):
//  Slice: positive degrees = ball goes right
//  Hook: negative degrees = left goes left
```

这段代码计算了一个球与一个杯子之间的角度，并给出了杯子与球的初始位置。同时，它还计算了杯子与球之间的直角三角形的两个角度，使用余弦函数计算了三角形中的两个边长，并根据这些边长计算了球的新位置。

具体来说，这段代码首先定义了一个常量 $hypotenuse$，它代表球与杯子之间的直角三角形的斜边长。然后，它计算了杯子与球之间的直角三角形的两个角度，分别使用 atan2 函数计算。这两个角度分别是杯子与球之间的余弦定理计算得到的。

接着，代码将杯子的向量设置为 $(0,-1)$，在 $360$ 圆周上这个向量代表 $0$ 度。通过将杯子的向量与 $hypotenuse$ 相乘，并使用余弦函数计算，代码得到了球与杯子之间的直角三角形的两个边长。然后，代码使用这些边长计算了球的新位置，并将这个新位置存储在变量 $x$ 中。

最后，代码还计算了杯子与球之间的直角三角形的第三个角度，使用余弦函数计算，并使用这个角度计算了球与杯子之间的最短距离 $d$。


```
//
//  The cup is always at point: 0,0.
//  We use atan2 to compute the angle between the cup and the ball.
//  Setting the cup's vector to 0,-1 on a 360 circle is equivalent to:
//  0 deg = 12 o'clock;  90 deg = 3 o'clock;  180 deg = 6 o'clock;  270 = 9 o'clock
//  The reverse angle between the cup and the ball is a difference of PI (using radians).
//
//  Given the angle and stroke distance (hypotenuse), we use cosine to compute
//  the opposite and adjacent sides of the triangle, which, is the ball's new position.
//
//           0
//           |
//    270 - cup - 90
//           |
//          180
```

这段代码使用了括号、作用域、引用和结构等概念，来描述一个三元操作符的赋值语句。

具体来说，这段代码首先定义了一个名为"cup"的函数，包含三个参数：一个代表"不包含任何元素的杯子"的常量，一个代表当前所在位置的引用，和一个代表"opp"常量的引用。然后，在函数内部，定义了一个名为"新位置"的局部变量，并使用"opp"常量作为除数，将"不包含任何元素的杯子"的常量除以"opp"常量，得到一个新的位置。最后，在函数内部，使用了"/"运算符和"hyp"常量，将当前所在位置的引用除以"新位置"局部变量，得到一个新的位置。最终，将"opp"常量的引用赋值给"新位置"局部变量，表示将杯子从当前位置移到了新位置。


```
//
//
//          cup
//           |
//           |
//           | opp
//           |-----* new position
//           |    /
//           |   /
//      adj  |  /
//           | /  hyp
//           |/
//          tee
//
//    <- hook    slice ->
```

这段代码是一个用于生成描述高尔夫球或击球位置组合的函数。它通过使用位图掩码技术来描述组合。具体来说，这段代码将多个位图（每个位置组合的比特数）组合成一个二进制数，并使用位运算符对它进行操作。然后，你可以使用你自己的语言位运算符来测试或设置这些位，以描述你所感兴趣的位置组合。

这段代码适用于需要描述大量高尔夫球或击球位置组合的情况。通过使用位图掩码技术，它能够高效地描述这些组合，而不会产生大量重复或冗长的输出。


```
//
//
//  Given the large number of combinations needed to describe a particular stroke / ball location,
//  we use the technique of "bitwise masking" to describe stroke results.
//  With bit masking, multiple flags (bits) are combined into a single binary number that can be
//  tested by applying a mask. A mask is another binary number that isolates a particular bit that
//  you are interested in. You can then apply your language's bitwise opeartors to test or
//  set a flag.
//
//  Game design by Jason Bonthron, 2021
//  www.bonthron.com
//  for my father, Raymond Bonthron, an avid golfer
//
//  Inspired by the 1978 "Golf" from "Basic Computer Games"
//  by Steve North, who modified an existing golf game by an unknown author
```

This looks like a C# class that defines some classes and methods for a game that involves rolling a die and moving a character around on a grid.

The `Character` class defines the properties of a character, such as its position on the grid and whether it is " online " (able to perform actions).

The `Game` class defines the game loop and some methods for interacting with the player, such as rolling a die and moving the character.

The `Plot` class defines the properties of a plot, such as its position and whether it is "offline" (able to perform actions).

The `GetDistance` method calculates the distance between two points.

The `IsInRectangle` method checks whether a point is within a given rectangle.

The `ToRadians` method converts an angle in radians to degrees.

The `ToDegrees360` method converts an angle in radians to degrees in a 360-degree range.

The `Odds` method is a helper method that generates a random number between 1 and 100.

Note that this is just one possible implementation of a game that involves rolling a die and moving a character around on a grid, and there are many other ways to design and implement such a game.


```
//
//

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Threading;


namespace Golf
{
    using Ball = Golf.CircleGameObj;
    using Hazard = Golf.CircleGameObj;

    // --------------------------------------------------------------------------- Program
    class Program
    {
        static void Main(string[] args)
        {
            Golf g = new Golf();
        }
    }


    // --------------------------------------------------------------------------- Golf
    public class Golf
    {
        Ball BALL;
        int HOLE_NUM = 0;
        int STROKE_NUM = 0;
        int Handicap = 0;
        int PlayerDifficulty = 0;
        HoleGeometry holeGeometry;

        // all fairways are 40 yards wide, extend 5 yards beyond the cup, and
        // have 5 yards of rough around the perimeter
        const int FairwayWidth = 40;
        const int FairwayExtension = 5;
        const int RoughAmt = 5;

        // ScoreCard records the ball position after each stroke
        // a new list for each hole
        // include a blank list so index 1 == hole 1

        List<List<Ball>> ScoreCard = new List<List<Ball>> { new List<Ball>() };

        static void w(string s) { Console.WriteLine(s); } // WRITE
        Random RANDOM = new Random();


        // --------------------------------------------------------------- constructor
        public Golf()
        {
            Console.Clear();
            w(" ");
            w("          8\"\"\"\"8 8\"\"\"88 8     8\"\"\"\" ");
            w("          8    \" 8    8 8     8     ");
            w("          8e     8    8 8e    8eeee ");
            w("          88  ee 8    8 88    88    ");
            w("          88   8 8    8 88    88    ");
            w("          88eee8 8eeee8 88eee 88    ");
            w(" ");
            w("Welcome to the Creative Computing Country Club,");
            w("an eighteen hole championship layout located a short");
            w("distance from scenic downtown Lambertville, New Jersey.");
            w("The game will be explained as you play.");
            w("Enjoy your game! See you at the 19th hole...");
            w(" ");
            w("Type QUIT at any time to leave the game.");
            w("Type BAG at any time to review the clubs in your bag.");
            w(" ");

            Wait((z) =>
            {
                w(" ");
                w("              YOUR BAG");
                ReviewBag();
                w("Type BAG at any time to review the clubs in your bag.");
                w(" ");

                Wait((zz) =>
                {
                    w(" ");

                    Ask("PGA handicaps range from 0 to 30.\nWhat is your handicap?", 0, 30, (i) =>
                    {
                        Handicap = i;
                        w(" ");

                        Ask("Common difficulties at golf include:\n1=Hook, 2=Slice, 3=Poor Distance, 4=Trap Shots, 5=Putting\nWhich one is your worst?", 1, 5, (j) =>
                        {
                            PlayerDifficulty = j;
                            Console.Clear();
                            NewHole();
                        });
                    });
                });
            });
        }


        // --------------------------------------------------------------- NewHole
        void NewHole()
        {
            HOLE_NUM++;
            STROKE_NUM = 0;

            HoleInfo info = CourseInfo[HOLE_NUM];

            int yards = info.Yards;  // from tee to cup
            int par = info.Par;
            var cup = new CircleGameObj(0, 0, 0, GameObjType.CUP);
            var green = new CircleGameObj(0, 0, 10, GameObjType.GREEN);

            var fairway = new RectGameObj(0 - (FairwayWidth / 2),
                                          0 - (green.Radius + FairwayExtension),
                                          FairwayWidth,
                                          yards + (green.Radius + FairwayExtension) + 1,
                                          GameObjType.FAIRWAY);

            var rough = new RectGameObj(fairway.X - RoughAmt,
                                        fairway.Y - RoughAmt,
                                        fairway.Width + (2 * RoughAmt),
                                        fairway.Length + (2 * RoughAmt),
                                        GameObjType.ROUGH);

            BALL = new Ball(0, yards, 0, GameObjType.BALL);

            ScoreCardStartNewHole();

            holeGeometry = new HoleGeometry(cup, green, fairway, rough, info.Hazard);

            w("                |> " + HOLE_NUM);
            w("                |        ");
            w("                |        ");
            w("          ^^^^^^^^^^^^^^^");

            Console.WriteLine("Hole #{0}. You are at the tee. Distance {1} yards, par {2}.", HOLE_NUM, info.Yards, info.Par);
            w(info.Description);

            TeeUp();
        }


        // --------------------------------------------------------------- TeeUp
        // on the green? automatically select putter
        // otherwise Ask club and swing strength

        void TeeUp()
        {
            if (IsOnGreen(BALL) && !IsInHazard(BALL, GameObjType.SAND))
            {
                var putt = 10;
                w("[PUTTER: average 10 yards]");
                var msg = Odds(20) ? "Keep your head down.\n" : "";

                Ask(msg + "Choose your putt potency. (1-10)", 1, 10, (strength) =>
                {
                    var putter = Clubs[putt];
                    Stroke(Convert.ToDouble((double)putter.Item2 * ((double)strength / 10.0)), putt);
                });
            }
            else
            {
                Ask("What club do you choose? (1-10)", 1, 10, (c) =>
                {
                    var club = Clubs[c];

                    w(" ");
                    Console.WriteLine("[{0}: average {1} yards]", club.Item1.ToUpper(), club.Item2);

                    Ask("Now gauge your distance by a percentage of a full swing. (1-10)", 1, 10, (strength) =>
                    {
                        Stroke(Convert.ToDouble((double)club.Item2 * ((double)strength / 10.0)), c);
                    });
                });
            };
        }


        // -------------------------------------------------------- bitwise Flags
        int dub         = 0b00000000000001;
        int hook        = 0b00000000000010;
        int slice       = 0b00000000000100;
        int passedCup   = 0b00000000001000;
        int inCup       = 0b00000000010000;
        int onFairway   = 0b00000000100000;
        int onGreen     = 0b00000001000000;
        int inRough     = 0b00000010000000;
        int inSand      = 0b00000100000000;
        int inTrees     = 0b00001000000000;
        int inWater     = 0b00010000000000;
        int outOfBounds = 0b00100000000000;
        int luck        = 0b01000000000000;
        int ace         = 0b10000000000000;


        // --------------------------------------------------------------- Stroke
        void Stroke(double clubAmt, int clubIndex)
        {
            STROKE_NUM++;

            var flags = 0b000000000000;

            // fore! only when driving
            if ((STROKE_NUM == 1) && (clubAmt > 210) && Odds(30)) { w("\"...Fore !\""); };

            // dub
            if (Odds(5)) { flags |= dub; }; // there's always a 5% chance of dubbing it

            // if you're in the rough, or sand, you really should be using a wedge
            if ((IsInRough(BALL) || IsInHazard(BALL, GameObjType.SAND)) &&
                !(clubIndex == 8 || clubIndex == 9))
            {
                if (Odds(40)) { flags |= dub; };
            };

            // trap difficulty
            if (IsInHazard(BALL, GameObjType.SAND) && PlayerDifficulty == 4)
            {
                if (Odds(20)) { flags |= dub; };
            }

            // hook/slice
            // There's 10% chance of a hook or slice
            // if it's a known playerDifficulty then increase chance to 30%
            // if it's a putt & putting is a playerDifficulty increase to 30%

            bool randHookSlice = (PlayerDifficulty == 1 ||
                                  PlayerDifficulty == 2 ||
                                  (PlayerDifficulty == 5 && IsOnGreen(BALL))) ? Odds(30) : Odds(10);

            if (randHookSlice)
            {
                if (PlayerDifficulty == 1)
                {
                    if (Odds(80)) { flags |= hook; } else { flags |= slice; };
                }
                else if (PlayerDifficulty == 2)
                {
                    if (Odds(80)) { flags |= slice; } else { flags |= hook; };
                }
                else
                {
                    if (Odds(50)) { flags |= hook; } else { flags |= slice; };
                };
            };

            // beginner's luck !
            // If handicap is greater than 15, there's a 10% chance of avoiding all errors
            if ((Handicap > 15) && (Odds(10))) { flags |= luck; };

            // ace
            // there's a 10% chance of an Ace on a par 3
            if (CourseInfo[HOLE_NUM].Par == 3 && Odds(10) && STROKE_NUM == 1) { flags |= ace; };

            // distance:
            // If handicap is < 15, there a 50% chance of reaching club average,
            // a 25% of exceeding it, and a 25% of falling short
            // If handicap is > 15, there's a 25% chance of reaching club average,
            // and 75% chance of falling short
            // The greater the handicap, the more the ball falls short
            // If poor distance is a known playerDifficulty, then reduce distance by 10%

            double distance;
            int rnd = RANDOM.Next(1, 101);

            if (Handicap < 15)
            {
                if (rnd <= 25)
                {
                    distance = clubAmt - (clubAmt * ((double)Handicap / 100.0));
                }
                else if (rnd > 25 && rnd <= 75)
                {
                    distance = clubAmt;
                }
                else
                {
                    distance = clubAmt + (clubAmt * 0.10);
                };
            }
            else
            {
                if (rnd <= 75)
                {
                    distance = clubAmt - (clubAmt * ((double)Handicap / 100.0));
                }
                else
                {
                    distance = clubAmt;
                };
            };

            if (PlayerDifficulty == 3)  // poor distance
            {
                if (Odds(80)) { distance = (distance * 0.80); };
            };

            if ((flags & luck) == luck) { distance = clubAmt; }

            // angle
            // For all strokes, there's a possible "drift" of 4 degrees
            // a hooks or slice increases the angle between 5-10 degrees, hook uses negative degrees
            int angle = RANDOM.Next(0, 5);
            if ((flags & slice) == slice) { angle = RANDOM.Next(5, 11); };
            if ((flags & hook) == hook) { angle = 0 - RANDOM.Next(5, 11); };
            if ((flags & luck) == luck) { angle = 0; };

            var plot = PlotBall(BALL, distance, Convert.ToDouble(angle));  // calculate a new location
            if ((flags & luck) == luck) { if(plot.Y > 0){ plot.Y = 2; }; };

            flags = FindBall(new Ball(plot.X, plot.Y, plot.Offline, GameObjType.BALL), flags);

            InterpretResults(plot, flags);
        }


        // --------------------------------------------------------------- plotBall
        Plot PlotBall(Ball ball, double strokeDistance, double degreesOff)
        {
            var cupVector = new Point(0, -1);
            double radFromCup = Math.Atan2((double)ball.Y, (double)ball.X) - Math.Atan2((double)cupVector.Y, (double)cupVector.X);
            double radFromBall = radFromCup - Math.PI;

            var hypotenuse = strokeDistance;
            var adjacent = Math.Cos(radFromBall + ToRadians(degreesOff)) * hypotenuse;
            var opposite = Math.Sqrt(Math.Pow(hypotenuse, 2) - Math.Pow(adjacent, 2));

            Point newPos;
            if (ToDegrees360(radFromBall + ToRadians(degreesOff)) > 180)
            {
                newPos = new Point(Convert.ToInt32(ball.X - opposite),
                                   Convert.ToInt32(ball.Y - adjacent));
            }
            else
            {
                newPos = new Point(Convert.ToInt32(ball.X + opposite),
                                   Convert.ToInt32(ball.Y - adjacent));
            }

            return new Plot(newPos.X, newPos.Y, Convert.ToInt32(opposite));
        }


        // --------------------------------------------------------------- InterpretResults
        void InterpretResults(Plot plot, int flags)
        {
            int cupDistance = Convert.ToInt32(GetDistance(new Point(plot.X, plot.Y),
                                                          new Point(holeGeometry.Cup.X, holeGeometry.Cup.Y)));
            int travelDistance = Convert.ToInt32(GetDistance(new Point(plot.X, plot.Y),
                                                             new Point(BALL.X, BALL.Y)));

            w(" ");

            if ((flags & ace) == ace)
            {
                w("Hole in One! You aced it.");
                ScoreCardRecordStroke(new Ball(0, 0, 0, GameObjType.BALL));
                ReportCurrentScore();
                return;
            };

            if ((flags & inTrees) == inTrees)
            {
                w("Your ball is lost in the trees. Take a penalty stroke.");
                ScoreCardRecordStroke(BALL);
                TeeUp();
                return;
            };

            if ((flags & inWater) == inWater)
            {
                var msg = Odds(50) ? "Your ball has gone to a watery grave." : "Your ball is lost in the water.";
                w(msg + " Take a penalty stroke.");
                ScoreCardRecordStroke(BALL);
                TeeUp();
                return;
            };

            if ((flags & outOfBounds) == outOfBounds)
            {
                w("Out of bounds. Take a penalty stroke.");
                ScoreCardRecordStroke(BALL);
                TeeUp();
                return;
            };

            if ((flags & dub) == dub)
            {
                w("You dubbed it.");
                ScoreCardRecordStroke(BALL);
                TeeUp();
                return;
            };

            if ((flags & inCup) == inCup)
            {
                var msg = Odds(50) ? "You holed it." : "It's in!";
                w(msg);
                ScoreCardRecordStroke(new Ball(plot.X, plot.Y, 0, GameObjType.BALL));
                ReportCurrentScore();
                return;
            };

            if (((flags & slice) == slice) &&
                !((flags & onGreen) == onGreen))
            {
                var bad = ((flags & outOfBounds) == outOfBounds) ? " badly" : "";
                Console.WriteLine("You sliced{0}: {1} yards offline.", bad, plot.Offline);
            };

            if (((flags & hook) == hook) &&
                !((flags & onGreen) == onGreen))
            {
                var bad = ((flags & outOfBounds) == outOfBounds) ? " badly" : "";
                Console.WriteLine("You hooked{0}: {1} yards offline.", bad, plot.Offline);
            };

            if (STROKE_NUM > 1)
            {
                var prevBall = ScoreCardGetPreviousStroke();
                var d1 = GetDistance(new Point(prevBall.X, prevBall.Y),
                                     new Point(holeGeometry.Cup.X, holeGeometry.Cup.Y));
                var d2 = cupDistance;
                if (d2 > d1) { w("Too much club."); };
            };

            if ((flags & inRough) == inRough) { w("You're in the rough."); };

            if ((flags & inSand) == inSand) { w("You're in a sand trap."); };

            if ((flags & onGreen) == onGreen)
            {
                var pd = (cupDistance < 4) ? ((cupDistance * 3) + " feet") : (cupDistance + " yards");
                Console.WriteLine("You're on the green. It's {0} from the pin.", pd);
            };

            if (((flags & onFairway) == onFairway) ||
                ((flags & inRough) == inRough))
            {
                Console.WriteLine("Shot went {0} yards. It's {1} yards from the cup.", travelDistance, cupDistance);
            };

            ScoreCardRecordStroke(new Ball(plot.X, plot.Y, 0, GameObjType.BALL));

            BALL = new Ball(plot.X, plot.Y, 0, GameObjType.BALL);

            TeeUp();
        }


        // --------------------------------------------------------------- ReportCurrentScore
        void ReportCurrentScore()
        {
            var par = CourseInfo[HOLE_NUM].Par;
            if (ScoreCard[HOLE_NUM].Count == par + 1) { w("A bogey. One above par."); };
            if (ScoreCard[HOLE_NUM].Count == par) { w("Par. Nice."); };
            if (ScoreCard[HOLE_NUM].Count == (par - 1)) { w("A birdie! One below par."); };
            if (ScoreCard[HOLE_NUM].Count == (par - 2)) { w("An Eagle! Two below par."); };
            if (ScoreCard[HOLE_NUM].Count == (par - 3)) { w("Double Eagle! Unbelievable."); };

            int totalPar = 0;
            for (var i = 1; i <= HOLE_NUM; i++) { totalPar += CourseInfo[i].Par; };

            w(" ");
            w("-----------------------------------------------------");
            Console.WriteLine(" Total par for {0} hole{1} is: {2}. Your total is: {3}.",
                              HOLE_NUM,
                              ((HOLE_NUM > 1) ? "s" : ""), //plural
                              totalPar,
                              ScoreCardGetTotal());
            w("-----------------------------------------------------");
            w(" ");

            if (HOLE_NUM == 18)
            {
                GameOver();
            }
            else
            {
                Thread.Sleep(2000);
                NewHole();
            };
        }


        // --------------------------------------------------------------- FindBall
        int FindBall(Ball ball, int flags)
        {
            if (IsOnFairway(ball) && !IsOnGreen(ball)) { flags |= onFairway; }
            if (IsOnGreen(ball)) { flags |= onGreen; }
            if (IsInRough(ball)) { flags |= inRough; }
            if (IsOutOfBounds(ball)) { flags |= outOfBounds; }
            if (IsInHazard(ball, GameObjType.WATER)) { flags |= inWater; }
            if (IsInHazard(ball, GameObjType.TREES)) { flags |= inTrees; }
            if (IsInHazard(ball, GameObjType.SAND))  { flags |= inSand;  }

            if (ball.Y < 0) { flags |= passedCup; }

            // less than 2, it's in the cup
            var d = GetDistance(new Point(ball.X, ball.Y),
                                new Point(holeGeometry.Cup.X, holeGeometry.Cup.Y));
            if (d < 2) { flags |= inCup; };

            return flags;
        }


        // --------------------------------------------------------------- IsOnFairway
        bool IsOnFairway(Ball ball)
        {
            return IsInRectangle(ball, holeGeometry.Fairway);
        }


        // --------------------------------------------------------------- IsOngreen
        bool IsOnGreen(Ball ball)
        {
            var d = GetDistance(new Point(ball.X, ball.Y),
                                new Point(holeGeometry.Cup.X, holeGeometry.Cup.Y));
            return d < holeGeometry.Green.Radius;
        }


        // --------------------------------------------------------------- IsInHazard
        bool IsInHazard(Ball ball, GameObjType hazard)
        {
            bool result = false;
            Array.ForEach(holeGeometry.Hazards, (Hazard h) =>
            {
                var d = GetDistance(new Point(ball.X, ball.Y), new Point(h.X, h.Y));
                if ((d < h.Radius) && h.Type == hazard) { result = true; };
            });
            return result;
        }


        // --------------------------------------------------------------- IsInRough
        bool IsInRough(Ball ball)
        {
            return IsInRectangle(ball, holeGeometry.Rough) &&
                (IsInRectangle(ball, holeGeometry.Fairway) == false);
        }


        // --------------------------------------------------------------- IsOutOfBounds
        bool IsOutOfBounds(Ball ball)
        {
            return (IsOnFairway(ball) == false) && (IsInRough(ball) == false);
        }


        // --------------------------------------------------------------- ScoreCardNewHole
        void ScoreCardStartNewHole()
        {
            ScoreCard.Add(new List<Ball>());
        }


        // --------------------------------------------------------------- ScoreCardRecordStroke
        void ScoreCardRecordStroke(Ball ball)
        {
            var clone = new Ball(ball.X, ball.Y, 0, GameObjType.BALL);
            ScoreCard[HOLE_NUM].Add(clone);
        }


        // ------------------------------------------------------------ ScoreCardGetPreviousStroke
        Ball ScoreCardGetPreviousStroke()
        {
            return ScoreCard[HOLE_NUM][ScoreCard[HOLE_NUM].Count - 1];
        }


        // --------------------------------------------------------------- ScoreCardGetTotal
        int ScoreCardGetTotal()
        {
            int total = 0;
            ScoreCard.ForEach((h) => { total += h.Count; });
            return total;
        }


        // --------------------------------------------------------------- Ask
        // input from console is always an integer passed to a callback
        // or "quit" to end game

        void Ask(string question, int min, int max, Action<int> callback)
        {
            w(question);
            string i = Console.ReadLine().Trim().ToLower();
            if (i == "quit") { Quit(); return; };
            if (i == "bag") { ReviewBag(); };

            int n;
            bool success = Int32.TryParse(i, out n);

            if (success)
            {
                if (n >= min && n <= max)
                {
                    callback(n);
                }
                else
                {
                    Ask(question, min, max, callback);
                }
            }
            else
            {
                Ask(question, min, max, callback);
            };
        }


        // --------------------------------------------------------------- Wait
        void Wait(Action<int> callback)
        {
            w("Press any key to continue.");

            ConsoleKeyInfo keyinfo;
            do { keyinfo = Console.ReadKey(true); }
            while (keyinfo.KeyChar < 0);
            Console.Clear();
            callback(0);
        }


        // --------------------------------------------------------------- ReviewBag
        void ReviewBag()
        {
            w(" ");
            w("  #     Club      Average Yardage");
            w("-----------------------------------");
            w("  1    Driver           250");
            w("  2    3 Wood           225");
            w("  3    5 Wood           200");
            w("  4    Hybrid           190");
            w("  5    4 Iron           170");
            w("  6    7 Iron           150");
            w("  7    9 Iron           125");
            w("  8    Pitching wedge   110");
            w("  9    Sand wedge        75");
            w(" 10    Putter            10");
            w(" ");
        }


        // --------------------------------------------------------------- Quit
        void Quit()
        {
            w("");
            w("Looks like rain. Goodbye!");
            w("");
            Wait((z) => { });
            return;
        }


        // --------------------------------------------------------------- GameOver
        void GameOver()
        {
            var net = ScoreCardGetTotal() - Handicap;
            w("Good game!");
            w("Your net score is: " + net);
            w("Let's visit the pro shop...");
            w(" ");
            Wait((z) => { });
            return;
        }


        // YOUR BAG
        // ======================================================== Clubs
        (string, int)[] Clubs = new (string, int)[] {
            ("",0),

                // name, average yardage
                ("Driver", 250),
                ("3 Wood", 225),
                ("5 Wood", 200),
                ("Hybrid", 190),
                ("4 Iron", 170),
                ("7 Iron", 150),
                ("9 Iron", 125),
                ("Pitching wedge", 110),
                ("Sand wedge", 75),
                ("Putter", 10)
                };


        // THE COURSE
        // ======================================================== CourseInfo

        HoleInfo[] CourseInfo = new HoleInfo[]{
            new HoleInfo(0, 0, 0, new Hazard[]{}, ""), // include a blank so index 1 == hole 1


            // -------------------------------------------------------- front 9
            // hole, yards, par, hazards, (description)

            new HoleInfo(1, 361, 4,
                         new Hazard[]{
                             new Hazard( 20, 100, 10, GameObjType.TREES),
                             new Hazard(-20,  80, 10, GameObjType.TREES),
                             new Hazard(-20, 100, 10, GameObjType.TREES)
                         },
                         "There are a couple of trees on the left and right."),

            new HoleInfo(2, 389, 4,
                         new Hazard[]{
                             new Hazard(0, 160, 20, GameObjType.WATER)
                         },
                         "There is a large water hazard across the fairway about 150 yards."),

            new HoleInfo(3, 206, 3,
                         new Hazard[]{
                             new Hazard( 20,  20,  5, GameObjType.WATER),
                             new Hazard(-20, 160, 10, GameObjType.WATER),
                             new Hazard( 10,  12,  5, GameObjType.SAND)
                         },
                         "There is some sand and water near the green."),

            new HoleInfo(4, 500, 5,
                         new Hazard[]{
                             new Hazard(-14, 12, 12, GameObjType.SAND)
                         },
                         "There's a bunker to the left of the green."),

            new HoleInfo(5, 408, 4,
                         new Hazard[]{
                             new Hazard(20, 120, 20, GameObjType.TREES),
                             new Hazard(20, 160, 20, GameObjType.TREES),
                             new Hazard(10,  20,  5, GameObjType.SAND)
                         },
                         "There are some trees to your right."),

            new HoleInfo(6, 359, 4,
                         new Hazard[]{
                             new Hazard( 14, 0, 4, GameObjType.SAND),
                             new Hazard(-14, 0, 4, GameObjType.SAND)
                         },
                         ""),

            new HoleInfo(7, 424, 5,
                         new Hazard[]{
                             new Hazard(20, 200, 10, GameObjType.SAND),
                             new Hazard(10, 180, 10, GameObjType.SAND),
                             new Hazard(20, 160, 10, GameObjType.SAND)
                         },
                         "There are several sand traps along your right."),

            new HoleInfo(8, 388, 4,
                         new Hazard[]{
                             new Hazard(-20, 340, 10, GameObjType.TREES)
                         },
                         ""),

            new HoleInfo(9, 196, 3,
                         new Hazard[]{
                             new Hazard(-30, 180, 20, GameObjType.TREES),
                             new Hazard( 14,  -8,  5, GameObjType.SAND)
                         },
                         ""),

            // -------------------------------------------------------- back 9
            // hole, yards, par, hazards, (description)

            new HoleInfo(10, 400, 4,
                         new Hazard[]{
                             new Hazard(-14, -8, 5, GameObjType.SAND),
                             new Hazard( 14, -8, 5, GameObjType.SAND)
                         },
                         ""),

            new HoleInfo(11, 560, 5,
                         new Hazard[]{
                             new Hazard(-20, 400, 10, GameObjType.TREES),
                             new Hazard(-10, 380, 10, GameObjType.TREES),
                             new Hazard(-20, 260, 10, GameObjType.TREES),
                             new Hazard(-20, 200, 10, GameObjType.TREES),
                             new Hazard(-10, 180, 10, GameObjType.TREES),
                             new Hazard(-20, 160, 10, GameObjType.TREES)
                         },
                         "Lots of trees along the left of the fairway."),

            new HoleInfo(12, 132, 3,
                         new Hazard[]{
                             new Hazard(-10, 120, 10, GameObjType.WATER),
                             new Hazard( -5, 100, 10, GameObjType.SAND)
                         },
                         "There is water and sand directly in front of you. A good drive should clear both."),

            new HoleInfo(13, 357, 4,
                         new Hazard[]{
                             new Hazard(-20, 200, 10, GameObjType.TREES),
                             new Hazard(-10, 180, 10, GameObjType.TREES),
                             new Hazard(-20, 160, 10, GameObjType.TREES),
                             new Hazard( 14,  12,  8, GameObjType.SAND)
                         },
                         ""),

            new HoleInfo(14, 294, 4,
                         new Hazard[]{
                             new Hazard(0, 20, 10, GameObjType.SAND)
                         },
                         ""),

            new HoleInfo(15, 475, 5,
                         new Hazard[]{
                             new Hazard(-20, 20, 10, GameObjType.WATER),
                             new Hazard( 10, 20, 10, GameObjType.SAND)
                         },
                         "Some sand and water near the green."),

            new HoleInfo(16, 375, 4,
                         new Hazard[]{
                             new Hazard(-14, -8, 5, GameObjType.SAND)
                         },
                         ""),

            new HoleInfo(17, 180, 3,
                         new Hazard[]{
                             new Hazard( 20, 100, 10, GameObjType.TREES),
                             new Hazard(-20,  80, 10, GameObjType.TREES)
                         },
                         ""),

            new HoleInfo(18, 550, 5,
                         new Hazard[]{
                             new Hazard(20, 30, 15, GameObjType.WATER)
                         },
                         "There is a water hazard near the green.")
        };


        // -------------------------------------------------------- HoleInfo
        class HoleInfo
        {
            public int Hole { get; }
            public int Yards { get; }
            public int Par { get; }
            public Hazard[] Hazard { get; }
            public string Description { get; }

            public HoleInfo(int hole, int yards, int par, Hazard[] hazard, string description)
            {
                Hole = hole;
                Yards = yards;
                Par = par;
                Hazard = hazard;
                Description = description;
            }
        }


        public enum GameObjType { BALL, CUP, GREEN, FAIRWAY, ROUGH, TREES, WATER, SAND }


        // -------------------------------------------------------- CircleGameObj
        public class CircleGameObj
        {
            public GameObjType Type { get; }
            public int X { get; }
            public int Y { get; }
            public int Radius { get; }

            public CircleGameObj(int x, int y, int r, GameObjType type)
            {
                Type = type;
                X = x;
                Y = y;
                Radius = r;
            }
        }


        // -------------------------------------------------------- RectGameObj
        public class RectGameObj
        {
            public GameObjType Type { get; }
            public int X { get; }
            public int Y { get; }
            public int Width { get; }
            public int Length { get; }

            public RectGameObj(int x, int y, int w, int l, GameObjType type)
            {
                Type = type;
                X = x;
                Y = y;
                Width = w;
                Length = l;
            }
        }


        // -------------------------------------------------------- HoleGeometry
        public class HoleGeometry
        {
            public CircleGameObj Cup { get; }
            public CircleGameObj Green { get; }
            public RectGameObj Fairway { get; }
            public RectGameObj Rough { get; }
            public Hazard[] Hazards { get; }

            public HoleGeometry(CircleGameObj cup, CircleGameObj green, RectGameObj fairway, RectGameObj rough, Hazard[] haz)
            {
                Cup = cup;
                Green = green;
                Fairway = fairway;
                Rough = rough;
                Hazards = haz;
            }
        }


        // -------------------------------------------------------- Plot
        public class Plot
        {
            public int X { get; }
            public int Y { get; set; }
            public int Offline { get; }

            public Plot(int x, int y, int offline)
            {
                X = x;
                Y = y;
                Offline = offline;
            }
        }


        // -------------------------------------------------------- GetDistance
        // distance between 2 points
        double GetDistance(Point pt1, Point pt2)
        {
            return Math.Sqrt(Math.Pow((pt2.X - pt1.X), 2) + Math.Pow((pt2.Y - pt1.Y), 2));
        }


        // -------------------------------------------------------- IsInRectangle
        bool IsInRectangle(CircleGameObj pt, RectGameObj rect)
        {
            return ((pt.X > rect.X) &&
                    (pt.X < rect.X + rect.Width) &&
                    (pt.Y > rect.Y) &&
                    (pt.Y < rect.Y + rect.Length));
        }


        // -------------------------------------------------------- ToRadians
        double ToRadians(double angle) { return angle * (Math.PI / 180.0); }


        // -------------------------------------------------------- ToDegrees360
        // radians to 360 degrees
        double ToDegrees360(double angle)
        {
            double deg = angle * (180.0 / Math.PI);
            if (deg < 0.0) { deg += 360.0; }
            return deg;
        }


        // -------------------------------------------------------- Odds
        // chance an integer is <= the given argument
        // between 1-100
        Random RND = new Random();

        bool Odds(int x)
        {
            return RND.Next(1, 101) <= x;
        }
    }
}

```