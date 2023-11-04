# BasicComputerGames源码解析 76

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


# `83_Stock_Market/javascript/stockmarket.js`

这段代码定义了两个函数，名为 `print` 和 `input`。

`print` 函数的作用是在页面上输出一个字符串，接收一个参数 `str`，将其转换为文本节点并添加到页面的 `<textarea>` 元素中，这里 `print` 函数会在页面上输出 `str`，然后将其包含在一个换行符中，并将其添加到 `<textarea>` 元素中。

`input` 函数的作用是从用户接收一个字符串，接收用户输入后返回一个 Promise 对象，这里 `input` 函数会创建一个 `<INPUT>` 元素，设置其 `type` 属性为 `text` 和 `length` 属性为 `50`，并将 `print` 函数中获取到的输出字符串作为 `value` 属性添加到 `<INPUT>` 元素中，然后将 `<INPUT>` 元素添加到页面上，设置 `input` 函数的 `addEventListener` 函数来监听 `keydown` 事件，当用户按下回车键时，将用户输入的字符串存储到 `input` 函数的参数中，并将其输出到页面上，并将其与之前的输出字符串进行拼接，最后输出结果。


```
// STOCKMARKET
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

这段代码定义了一个名为 `tab` 的函数，它会根据传入的参数 `space` 打印出一些字符。

首先，在函数内部，定义了一个字符串变量 `str`，并使用一个无限循环来打印字符，循环变量 `space` 是一个递减的整数，初始值为 0。循环中，使用 `console.log()` 函数将每个字符打印到 `str` 变量中，并在字符串的末尾添加一个空格。

然后，在循环外部，定义了四个变量 `sa`、`pa`、`za` 和 `ca`，它们都初始化为一个空数组。定义了一个整型变量 `i1`、一个整型变量 `n1` 和一个字符串变量 `e1`，它们的值都未定义。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var sa = [];
var pa = [];
var za = [];
var ca = [];
var i1;
var n1;
var e1;
```

This script appears to simulate the stock market over a period of 8 days, adjusting the trend sign and slope based on the random number generated.

It does this by first initializing the variables w3, e1, and e2 to 0. Then, it checks if the current day is within the range of 1 to 8, and if so, it adds 10 points to the stock's value and resets the variables.

If the current day is not within the range of 1 to 8, it subtracts 10 points from the stock's value and resets the variables.

It appears that the script also generates a random number between 0 and 1, and adds a certain percentage of that number to the stock's value based on whether the current day is a Monday or Friday.

Finally, after 8 days, it randomly changes the trend sign and slope, and also adjusts the stock's value based on the trend sign.


```
var i2;
var n2;
var e2;
var x1;
var w3;
var t8;
var a;
var s4;

// New stock values - subroutine
function randomize_initial()
{
    // RANDOMLY PRODUCE NEW STOCK VALUES BASED ON PREVIOUS
    // DAY'S VALUES
    // N1,N2 ARE RANDOM NUMBERS OF DAYS WHICH RESPECTIVELY
    // DETERMINE WHEN STOCK I1 WILL INCREASE 10 PTS. AND STOCK
    // I2 WILL DECREASE 10 PTS.
    // IF N1 DAYS HAVE PASSED, PICK AN I1, SET E1, DETERMINE NEW N1
    if (n1 <= 0) {
        i1 = Math.floor(4.99 * Math.random() + 1);
        n1 = Math.floor(4.99 * Math.random() + 1);
        e1 = 1;
    }
    // IF N2 DAYS HAVE PASSED, PICK AN I2, SET E2, DETERMINE NEW N2
    if (n2 <= 0) {
        i2 = Math.floor(4.99 * Math.random() + 1);
        n2 = Math.floor(4.99 * Math.random() + 1);
        e2 = 1;
    }
    // DEDUCT ONE DAY FROM N1 AND N2
    n1--;
    n2--;
    // LOOP THROUGH ALL STOCKS
    for (i = 1; i <= 5; i++) {
        x1 = Math.random();
        if (x1 < 0.25) {
            x1 = 0.25;
        } else if (x1 < 0.5) {
            x1 = 0.5;
        } else if (x1 < 0.75) {
            x1 = 0.75;
        } else {
            x1 = 0.0;
        }
        // BIG CHANGE CONSTANT:W3  (SET TO ZERO INITIALLY)
        w3 = 0;
        if (e1 >= 1 && Math.floor(i1 + 0.5) == Math.floor(i + 0.5)) {
            // ADD 10 PTS. TO THIS STOCK;  RESET E1
            w3 = 10;
            e1 = 0;
        }
        if (e2 >= 1 && Math.floor(i2 + 0.5) == Math.floor(i + 0.5)) {
            // SUBTRACT 10 PTS. FROM THIS STOCK;  RESET E2
            w3 -= 10;
            e2 = 0;
        }
        // C(I) IS CHANGE IN STOCK VALUE
        ca[i] = Math.floor(a * sa[i]) + x1 + Math.floor(3 - 6 * Math.random() + 0.5) + w3;
        ca[i] = Math.floor(100 * ca[i] + 0.5) / 100;
        sa[i] += ca[i];
        if (sa[i] <= 0) {
            ca[i] = 0;
            sa[i] = 0;
        } else {
            sa[i] = Math.floor(100 * sa[i] + 0.5) / 100;
        }
    }
    // AFTER T8 DAYS RANDOMLY CHANGE TREND SIGN AND SLOPE
    if (--t8 < 1) {
        // RANDOMLY CHANGE TREND SIGN AND SLOPE (A), AND DURATION
        // OF TREND (T8)
        t8 = Math.floor(4.99 * Math.random() + 1);
        a = Math.floor((Math.random() / 10) * 100 + 0.5) / 100;
        s4 = Math.random();
        if (s4 > 0.5)
            a = -a;
    }
}

```

This code appears to be a Java program that is meant to print out the portfolio for a user based on their initial investments and the number of stocks they have. It first sets the initial values for the portfolio and then loops through the stocks to calculate the value of each stock. The portfolio is then printed out with the stock name, the price/share, the holdings, and the current price change. The program also includes a bell ringing feature that generates a random signal for the user to see if their portfolio is being affected by market movements.

It is important to note that this program is a complete code and it will be deployed as a java application only. Also, it is not a real-world application and it is not meant to be a production-ready system.


```
// Main program
async function main()
{
    print(tab(30) + "STOCK MARKET\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // STOCK MARKET SIMULATION     -STOCK-
    // REVISED 8/18/70 (D. PESSEL, L. BRAUN, C. LOSIK)
    // IMP VRBLS: A-MRKT TRND SLP; B5-BRKRGE FEE; C-TTL CSH ASSTS;
    // C5-TTL CSH ASSTS (TEMP); C(I)-CHNG IN STK VAL; D-TTL ASSTS;
    // E1,E2-LRG CHNG MISC; I-STCK #; I1,I2-STCKS W LRG CHNG;
    // N1,N2-LRG CHNG DAY CNTS; P5-TTL DAYS PRCHSS; P(I)-PRTFL CNTNTS;
    // Q9-NEW CYCL?; S4-SGN OF A; S5-TTL DYS SLS; S(I)-VALUE/SHR;
    // T-TTL STCK ASSTS; T5-TTL VAL OF TRNSCTNS;
    // W3-LRG CHNG; X1-SMLL CHNG(<$1); Z4,Z5,Z6-NYSE AVE.; Z(I)-TRNSCT
    // SLOPE OF MARKET TREND:A  (SAME FOR ALL STOCKS)
    x = 1;
    a = Math.floor(Math.random() / 10 * 100 + 0.5) / 100;
    t5 = 0;
    x9 = 0;
    n1 = 0;
    n2 = 0;
    e1 = 0;
    e2 = 0;
    // INTRODUCTION
    print("DO YOU WANT THE INSTRUCTIONS (YES-TYPE 1, NO-TYPE 0)");
    z9 = parseInt(await input());
    print("\n");
    print("\n");
    if (z9 >= 1) {
        print("THIS PROGRAM PLAYS THE STOCK MARKET.  YOU WILL BE GIVEN\n");
        print("$10,000 AND MAY BUY OR SELL STOCKS.  THE STOCK PRICES WILL\n");
        print("BE GENERATED RANDOMLY AND THEREFORE THIS MODEL DOES NOT\n");
        print("REPRESENT EXACTLY WHAT HAPPENS ON THE EXCHANGE.  A TABLE\n");
        print("OF AVAILABLE STOCKS, THEIR PRICES, AND THE NUMBER OF SHARES\n");
        print("IN YOUR PORTFOLIO WILL BE PRINTED.  FOLLOWING THIS, THE\n");
        print("INITIALS OF EACH STOCK WILL BE PRINTED WITH A QUESTION\n");
        print("MARK.  HERE YOU INDICATE A TRANSACTION.  TO BUY A STOCK\n");
        print("TYPE +NNN, TO SELL A STOCK TYPE -NNN, WHERE NNN IS THE\n");
        print("NUMBER OF SHARES.  A BROKERAGE FEE OF 1% WILL BE CHARGED\n");
        print("ON ALL TRANSACTIONS.  NOTE THAT IF A STOCK'S VALUE DROPS\n");
        print("TO ZERO IT MAY REBOUND TO A POSITIVE VALUE AGAIN.  YOU\n");
        print("HAVE $10,000 TO INVEST.  USE INTEGERS FOR ALL YOUR INPUTS.\n");
        print("(NOTE:  TO GET A 'FEEL' FOR THE MARKET RUN FOR AT LEAST\n");
        print("10 DAYS)\n");
        print("-----GOOD LUCK!-----\n");
    }
    // GENERATION OF STOCK TABLE: INPUT REQUESTS
    // INITIAL STOCK VALUES
    sa[1] = 100;
    sa[2] = 85;
    sa[3] = 150;
    sa[4] = 140;
    sa[5] = 110;
    // INITIAL T8 - # DAYS FOR FIRST TREND SLOPE (A)
    t8 = Math.floor(4.99 * Math.random() + 1);
    // RANDOMIZE SIGN OF FIRST TREND SLOPE (A)
    if (Math.random() <= 0.5)
        a -= a;
    // RANDOMIZE INITIAL VALUES
    randomize_initial();
    // INITIAL PORTFOLIO CONTENTS
    for (i = 1; i <= 5; i++) {
        pa[i] = 0;
        za[i] = 0;
    }
    print("\n");
    print("\n");
    // INITIALIZE CASH ASSETS:C
    c = 10000;
    z5 = 0;
    // PRINT INITIAL PORTFOLIO
    print("STOCK\t \t\t\tINITIALS\tPRICE/SHARE\n");
    print("INT. BALLISTIC MISSILES\t\t  IBM\t\t" + sa[1] + "\n");
    print("RED CROSS OF AMERICA\t\t  RCA\t\t" + sa[2] + "\n");
    print("LICHTENSTEIN, BUMRAP & JOKE\t  LBJ\t\t" + sa[3] + "\n");
    print("AMERICAN BANKRUPT CO.\t\t  ABC\t\t" + sa[4] + "\n");
    print("CENSURED BOOKS STORE\t\t  CBS\t\t" + sa[5] + "\n");
    while (1) {
        print("\n");
        // NYSE AVERAGE:Z5; TEMP. VALUE:Z4; NET CHANGE:Z6
        z4 = z5;
        z5 = 0;
        t = 0;
        for (i = 1; i <= 5; i++) {
            z5 += sa[i];
            t += sa[i] * pa[i];
        }
        z5 = Math.floor(100 * (z5 / 5) + 0.5) / 100;
        z6 = Math.floor((z5 - z4) * 100 + 0.5) / 100;
        // TOTAL ASSETS:D
        d = t + c;
        if (x9 <= 0) {
            print("NEW YORK STOCK EXCHANGE AVERAGE: " + z5 + "\n");
        } else {
            print("NEW YORK STOCK EXCHANGE AVERAGE: " + z5 + " NET CHANGE " + z6 + "\n");
        }
        print("\n");
        t = Math.floor(100 * t + 0.5) / 100;
        print("TOTAL STOCK ASSETS ARE   $" + t + "\n");
        c = Math.floor(100 * c + 0.5) / 100;
        print("TOTAL CASH ASSETS ARE    $" + c + "\n");
        d = Math.floor(100 * d + 0.5) / 100;
        print("TOTAL ASSETS ARE         $" + d + "\n");
        print("\n");
        if (x9 != 0) {
            print("DO YOU WISH TO CONTINUE (YES-TYPE 1, NO-TYPE 0)");
            q9 = parseInt(await input());
            if (q9 < 1) {
                print("HOPE YOU HAD FUN!!\n");
                return;
            }
        }
        // INPUT TRANSACTIONS
        while (1) {
            print("WHAT IS YOUR TRANSACTION IN\n");
            print("IBM");
            za[1] = parseInt(await input());
            print("RCA");
            za[2] = parseInt(await input());
            print("LBJ");
            za[3] = parseInt(await input());
            print("ABC");
            za[4] = parseInt(await input());
            print("CBS");
            za[5] = parseInt(await input());
            print("\n");
            // TOTAL DAY'S PURCHASES IN $:P5
            p5 = 0;
            // TOTAL DAY'S SALES IN $:S5
            s5 = 0;
            for (i = 1; i <= 5; i++) {
                za[i] = Math.floor(za[i] + 0.5);
                if (za[i] > 0) {
                    p5 += za[i] * sa[i];
                } else {
                    s5 -= za[i] * sa[i];
                    if (-za[i] > pa[i]) {
                        print("YOU HAVE OVERSOLD A STOCK; TRY AGAIN.\n");
                        break;
                    }
                }
            }
            if (i <= 5)
                contine;
            // TOTAL VALUE OF TRANSACTIONS:T5
            t5 = p5 + s5;
            // BROKERAGE FEE:B5
            b5 = Math.floor(0.01 * t5 * 100 + 0.5) / 100;
            // CASH ASSETS=OLD CASH ASSETS-TOTAL PURCHASES
            // -BROKERAGE FEES+TOTAL SALES:C5
            c5 = c - p5 - b5 + s5;
            if (c5 < 0) {
                print("YOU HAVE USED $" + (-c5) + " MORE THAN YOU HAVE.\n");
                continue;
            }
            break;
        }
        c = c5;
        // CALCULATE NEW PORTFOLIO
        for (i = 1; i <= 5; i++) {
            pa[i] += za[i];
        }
        // CALCULATE NEW STOCK VALUES
        randomize_initial();
        // PRINT PORTFOLIO
        // BELL RINGING-DIFFERENT ON MANY COMPUTERS
        print("\n");
        print("**********     END OF DAY'S TRADING     **********\n");
        print("\n");
        print("\n");
        if (x9 >= 1) ;
        print("STOCK\tPRICE/SHARE\tHOLDINGS\tVALUE\tNET PRICE CHANGE\n");
        print("IBM\t" + sa[1] + "\t\t" + pa[1] + "\t\t" + sa[1] * pa[1] + "\t" + ca[1] + "\n");
        print("RCA\t" + sa[2] + "\t\t" + pa[2] + "\t\t" + sa[2] * pa[2] + "\t" + ca[2] + "\n");
        print("LBJ\t" + sa[3] + "\t\t" + pa[3] + "\t\t" + sa[3] * pa[3] + "\t" + ca[3] + "\n");
        print("ABC\t" + sa[4] + "\t\t" + pa[4] + "\t\t" + sa[4] * pa[4] + "\t" + ca[4] + "\n");
        print("CBS\t" + sa[5] + "\t\t" + pa[5] + "\t\t" + sa[5] * pa[5] + "\t" + ca[5] + "\n");
        x9 = 1;
        print("\n");
        print("\n");
    }
}

```

这是 C 语言中的一个程序，名为 "main"。程序的作用是运行在计算机的中央处理器（CPU）上，负责执行程序中定义的所有代码。

在 "main" 程序中，首先会定义一个名为 "main" 的函数。这个函数是程序执行的入口点，程序在调用这个函数之后才会开始执行。

通常情况下，在 "main" 函数中会包含程序的主要操作和代码。这些代码会根据用户的需求和输入，执行相应的操作，完成各种功能。例如，读取用户输入的文件信息、对输入的信息进行计算、生成随机数、显示图形等等。

具体来说，"main" 函数可能还会包含一些程序预设的代码，例如打印输出、关闭程序、检查是否支持某种特定功能等等。这些代码由程序开发者预先编写好，以便在程序运行时自动执行。


```
main();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


# `83_Stock_Market/python/Stock_Market.py`



This is a class definition for an `UpdateAssets` class. It appears to be a part of a larger program or script, and it creates a `UpdateAssets` object that keeps track of your cash and holding assets, and updates the values of these assets based on the current market prices and any new holdings.

The `UpdateAssets` class has several methods:

- `__init__(self, data)`: This method takes in a `data` argument, which is a dictionary of all the data held by the script.
- `holdings(self)`: This method returns the current value of the `Holdings` attribute for each stock in the `data` dictionary.
- `update_stock_assets(self)`: This method updates the value of the `Stock_assets` attribute by setting the current value of `Holdings` for each stock.
- `print_assets(self)`: This method prints out the current value of the `Stock_assets` attribute.
- `_check_transaction(self, new_holdings)`: This method checks if the script has enough funds to complete the transaction, and returns a boolean value indicating whether the transaction can be completed or not.
- `update_holdings(self, new_holdings)`: This method updates the value of the `Holdings` attribute for each stock in the `data` dictionary by setting the current value of `Holdings` for each stock.

It is important to note that this class is written in Python, and the code may have been specifically written for a specific program or script.


```
import random
from typing import Any, Dict, List


class Stock_Market:
    def __init__(self) -> None:
        # Hard Coded Names
        short_names = ["IBM", "RCA", "LBJ", "ABC", "CBS"]
        full_names = [
            "INT. BALLISTIC MISSLES",
            "RED CROSS OF AMERICA",
            "LICHTENSTEIN, BUMRAP & JOKE",
            "AMERICAN BANKRUPT CO.",
            "CENSURED BOOKS STORE",
        ]

        # Initializing Dictionary to hold all the information systematically
        self.data: Dict[str, Any] = {}
        for sn, fn in zip(short_names, full_names):
            # A dictionary for each stock
            temp = {"Name": fn, "Price": None, "Holdings": 0}
            # Nested outer dictionary for all stocks
            self.data[sn] = temp

        # Initializing Randomly generated initial prices
        for stock in self.data.values():
            stock["Price"] = round(random.uniform(80, 120), 2)  # Price b/w 60 and 120

        # Initialize Assets
        self.cash_assets = 10000
        self.stock_assets = 0

    def total_assets(self) -> float:
        return self.cash_assets + self.stock_assets

    def _generate_day_change(self) -> None:
        self.changes = []
        for _ in range(len(self.data)):
            self.changes.append(
                round(random.uniform(-5, 5), 2)
            )  # Random % Change b/w -5 and 5

    def update_prices(self) -> None:
        self._generate_day_change()
        for stock, change in zip(self.data.values(), self.changes):
            stock["Price"] = round(stock["Price"] + (change / 100) * stock["Price"], 2)

    def print_exchange_average(self) -> None:

        sum = 0
        for stock in self.data.values():
            sum += stock["Price"]

        print(f"\nNEW YORK STOCK EXCHANGE AVERAGE: ${sum / 5:.2f}")

    def get_average_change(self) -> float:
        sum: float = 0
        for change in self.changes:
            sum += change

        return round(sum / 5, 2)

    def print_first_day(self) -> None:

        print("\nSTOCK\t\t\t\t\tINITIALS\tPRICE/SHARE($)")
        for stock, data in self.data.items():
            if stock != "LBJ":
                print("{}\t\t\t{}\t\t{}".format(data["Name"], stock, data["Price"]))
            else:
                print("{}\t\t{}\t\t{}".format(data["Name"], stock, data["Price"]))

        self.print_exchange_average()
        self.print_assets()

    def take_inputs(self) -> List[str]:
        print("\nWHAT IS YOUR TRANSACTION IN")
        flag = False
        while not flag:
            new_holdings = []
            for stock in self.data.keys():
                try:
                    new_holdings.append(int(input(f"{stock}? ")))
                except Exception:
                    print("\nINVALID ENTRY, TRY AGAIN\n")
                    break
            if len(new_holdings) == 5:
                flag = self._check_transaction(new_holdings)

        return new_holdings  # type: ignore

    def print_trading_day(self) -> None:

        print("STOCK\tPRICE/SHARE\tHOLDINGS\tNET. Value\tPRICE CHANGE")
        for stock, data, change in zip(
            self.data.keys(), self.data.values(), self.changes
        ):
            value = data["Price"] * data["Holdings"]
            print(
                "{}\t{}\t\t{}\t\t{:.2f}\t\t{}".format(
                    stock, data["Price"], data["Holdings"], value, change
                )
            )

    def update_cash_assets(self, new_holdings) -> None:
        sell = 0
        buy = 0
        for stock, holding in zip(self.data.values(), new_holdings):
            if holding > 0:
                buy += stock["Price"] * holding

            elif holding < 0:
                sell += stock["Price"] * abs(holding)

        self.cash_assets = self.cash_assets + sell - buy

    def update_stock_assets(self) -> None:
        sum = 0
        for data in self.data.values():
            sum += data["Price"] * data["Holdings"]

        self.stock_assets = round(sum, 2)

    def print_assets(self) -> None:
        print(f"\nTOTAL STOCK ASSETS ARE: ${self.stock_assets:.2f}")
        print(f"TOTAL CASH ASSETS ARE: ${self.cash_assets:.2f}")
        print(f"TOTAL ASSETS ARE: ${self.total_assets():.2f}")

    def _check_transaction(self, new_holdings) -> bool:
        sum = 0
        for stock, holding in zip(self.data.values(), new_holdings):
            if holding > 0:
                sum += stock["Price"] * holding

            elif holding < 0:
                if abs(holding) > stock["Holdings"]:
                    print("\nYOU HAVE OVERSOLD SOME STOCKS, TRY AGAIN\n")
                    return False

        if sum > self.cash_assets:
            print(
                "\nYOU HAVE USED ${:.2f} MORE THAN YOU HAVE, TRY AGAIN\n".format(
                    sum - self.cash_assets
                )
            )
            return False

        return True

    def update_holdings(self, new_holdings) -> None:
        for stock, new_holding in zip(self.data.values(), new_holdings):
            stock["Holdings"] += new_holding


```

这段代码是一个函数，名为`print_instruction()`，它返回一个`None`对象。

函数体内部首先打印一段关于高风险高回报的金融市场提示，然后告诉用户他们拥有的可用股票、股票价格和所持有股票数量。接下来，函数会随机生成股票价格，并打印输出。

随机生成的股票价格是为了让现实金融市场的价格波动，不能完全代表实际交易。用户接下来可以进行买入或卖出操作，并注意所有交易将收取1%的佣金。


```
def print_instruction() -> None:

    print(
        """
THIS PROGRAM PLAYS THE STOCK MARKET.  YOU WILL BE GIVEN
$10,000 AND MAY BUY OR SELL STOCKS.  THE STOCK PRICES WILL
BE GENERATED RANDOMLY AND THEREFORE THIS MODEL DOES NOT
REPRESENT EXACTLY WHAT HAPPENS ON THE EXCHANGE.  A TABLE
OF AVAILABLE STOCKS, THEIR PRICES, AND THE NUMBER OF SHARES
IN YOUR PORTFOLIO WILL BE PRINTED.  FOLLOWING THIS, THE
INITIALS OF EACH STOCK WILL BE PRINTED WITH A QUESTION
MARK.  HERE YOU INDICATE A TRANSACTION.  TO BUY A STOCK
TYPE +NNN, TO SELL A STOCK TYPE -NNN, WHERE NNN IS THE
NUMBER OF SHARES.  A BROKERAGE FEE OF 1% WILL BE CHARGED
ON ALL TRANSACTIONS.  NOTE THAT IF A STOCK'S VALUE DROPS
```

这段代码是一个股票市场模拟程序，主要目的是让用户了解股票市场的运作方式。该程序由两个主要函数组成：`main`函数和辅助函数`print_instruction`。

主函数 `main` 函数负责处理用户输入的指令，并根据指令调用相应的函数。辅助函数 `print_instruction` 用于在开始交易之前向用户显示帮助信息。

程序的主要步骤如下：

1. 初始化股票市场，包括设置初始资金、股票数等。
2. 循环模拟一天的交易，包括更新股票价格、打印交易信息、计算平均汇率等。
3. 接收用户输入，根据指令更新股票市场数据。
4. 循环与用户交互，直到用户选择退出。
5. 在循环结束后，打印结果并等待用户输入新的指令。

代码中涉及了一些辅助函数和变量，但这些函数和变量在程序的主要功能中并不起关键作用。辅助函数的实现可能需要调用其他库或编写更多的代码。


```
TO ZERO IT MAY REBOUND TO A POSITIVE VALUE AGAIN.  YOU
HAVE $10,000 TO INVEST.  USE INTEGERS FOR ALL YOUR INPUTS.
(NOTE:  TO GET A 'FEEL' FOR THE MARKET RUN FOR AT LEAST
10 DAYS)
          ------------GOOD LUCK!------------\n
    """
    )


def main() -> None:
    print("\t\t      STOCK MARKET")
    help = input("\nDO YOU WANT INSTRUCTIONS(YES OR NO)? ")

    # Printing Instruction
    if help.lower() == "yes":
        print_instruction()

    # Initialize Game
    Game = Stock_Market()

    # Do first day
    Game.print_first_day()
    new_holdings = Game.take_inputs()
    Game.update_holdings(new_holdings)
    Game.update_cash_assets(new_holdings)
    print("\n------------END OF TRADING DAY--------------\n")

    response = 1
    while response == 1:

        # Simulate a DAY
        Game.update_prices()
        Game.print_trading_day()
        Game.print_exchange_average()
        Game.update_stock_assets()
        Game.print_assets()

        response = int(input("\nDO YOU WISH TO CONTINUE (YES-TYPE 1, NO-TYPE 0)? "))
        if response == 0:
            break

        new_holdings = Game.take_inputs()
        Game.update_holdings(new_holdings)
        Game.update_cash_assets(new_holdings)
        print("\n------------END OF TRADING DAY--------------\n")

    print("\nHOPE YOU HAD FUN!!!!")
    input()


```

这段代码是一个Python程序中的一个if语句。if语句是Python中的一种特殊类型的语句，它的作用是判断一个表达式的值是否为真，如果是真，则执行if语句内部的代码，否则跳过if语句继续执行。

在这段if语句中，表达式是`__name__ == "__main__"`，它表示的是当前程序的文件名是否与`__main__`完全相同。如果当前程序的文件名与`__main__`完全相同，那么程序会执行if语句内部的代码，否则跳过if语句继续执行。

因此，这段代码的作用是判断当前程序的文件名是否与`__main__`完全相同，如果是，则执行if语句内部的代码，否则跳过if语句继续执行。


```
if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Super Star Trek

#### Brief History
Many versions of Star Trek have been kicking around various college campuses since the late sixties. I recall playing one at Carnegie-Mellon Univ. in 1967 or 68, and a very different one at Berkeley. However, these were a far cry from the one written by Mike Mayfield of Centerline Engineering and/or Custom Data. This was written for an HP2000C and completed in October 1972. It became the “standard” Star Trek in February 1973 when it was put in the HP contributed program library and onto a number of HP Data Center machines.

In the summer of 1973, I converted the HP version to BASIC-PLUS for DEC’s RSTS-11 compiler and added a few bits and pieces while I was at it. Mary Cole at DEC contributed enormously to this task too. Later that year I published it under the name SPACWE (Space War — in retrospect, an incorrect name) in my book _101 Basic Computer Games_.It is difficult today to find an interactive computer installation that does not have one of these versions of Star Trek available.

#### Quadrant Nomenclature
Recently, certain critics have professed confusion as to the origin on the “quadrant” nomenclature used on all standard CG (Cartesian Galactic) maps. Naturally, for anyone with the remotest knowledge of history, no explanation is necessary; however, the following synopsis should suffice for the critics:

As everybody schoolboy knows, most of the intelligent civilizations in the Milky Way had originated galactic designations of their own choosing well before the Third Magellanic Conference, at which the so-called “2⁶ Agreement” was reached. In that historic document, the participant cultures agreed, in all two-dimensional representations of the galaxy, to specify 64 major subdivisions, ordered as an 8 x 8 matrix. This was partially in deference to the Earth culture (which had done much in the initial organization of the Federation), whose century-old galactic maps had landmarks divided into four “quadrants,” designated by ancient “Roman Numerals” (the origin of which has been lost).

To this day, the official logs of starships originating on near-Earth starbases still refer to the major galactic areas as “quadrants.”

The relation between the Historical and Standard nomenclatures is shown in the simplified CG map below.

|   | 1            | 2  | 3   | 4  | 5          | 6  | 7   | 8  |
|---|--------------|----|-----|----|------------|----|-----|----|
| 1 |    ANTARES   |    |     |    |   SIRIUS   |    |     |    |
|   | I            | II | III | IV | I          |    | III | IV |
| 2 |     RIGEL    |    |     |    |    DENEB   |    |     |    |
|   | I            | II | III | IV | I          | II | III | IV |
| 3 |    PROCYON   |    |     |    |   CAPELLA  |    |     |    |
|   | I            | II | III | IV | I          | II | III | IV |
| 4 | VEGA         |    |     |    | BETELGUESE |    |     |    |
|   | I            | II | III | IV | I          | II | III | IV |
| 5 |    CANOPUS   |    |     |    |  ALDEBARA  |    |     |    |
|   | I            | II | III | IV | I          | II | III | IV |
| 6 |    ALTAIR    |    |     |    |   REGULUS  |    |     |    |
|   | I            | II | III | IV | I          | II | III | IV |
| 7 | SAGITTARIOUS |    |     |    |  ARCTURUS  |    |     |    |
|   | I            | II | III | IV | I          | II | III | IV |
| 8 |    POLLUX    |    |     |    |    SPICA   |    |     |    |
|   | I            | II | III | IV | I          | II | III | IV |

#### Super Star Trek† Rules and Notes
1. OBJECTIVE: You are Captain of the starship “Enterprise”† with a mission to seek and destroy a fleet of Klingon† warships (usually about 17) which are menacing the United Federation of Planets.† You have a specified number of stardates in which to complete your mission. You also have two or three Federation Starbases† for resupplying your ship.

2. You will be assigned a starting position somewhere in the galaxy. The galaxy is divided into an 8 x 8 quadrant grid. The astronomical name of a quadrant is called out upon entry into a new region. (See “Quadrant Nomenclature.”) Each quadrant is further divided into an 8 x 8 section grid.

3. On a section diagram, the following symbols are used:
    - `<*>` Enterprise
    - `†††` Klingon
    - `>!<` Starbase
    - `*`   Star

4. You have eight commands available to you (A detailed description of each command is given in the program instructions.)
    - `NAV` Navigate the Starship by setting course and warp engine speed.
    - `SRS` Short-range sensor scan (one quadrant)
    - `LRS` Long-range sensor scan (9 quadrants)
    - `PHA` Phaser† control (energy gun)
    - `TOR` Photon torpedo control
    - `SHE` Shield control (protects against phaser fire)
    - `DAM` Damage and state-of-repair report
    - `COM` Call library computer

5. Library computer options are as follows (more complete descriptions are in program instructions):
    - `0` Cumulative galactic report
    - `1` Status report
    - `2` Photon torpedo course data
    - `3` Starbase navigation data
    - `4` Direction/distance calculator
    - `5` Quadrant nomenclature map

6. Certain reports on the ship’s status are made by officers of the Enterprise who appears on the original TV Show—Spock,† Scott,† Uhura,† Chekov,† etc.

7. Klingons are non-stationary within their quadrants. If you try to maneuver on them, they will move and fire on you.

8. Firing and damage notes:
    - Phaser fire diminishes with increased distance between combatants.
    - If a Klingon zaps you hard enough (relative to your shield strength) he will generally cause damage to some part of your ship with an appropriate “Damage Control” report resulting.
    - If you don’t zap a Klingon hard enough (relative to his shield strength) you won’t damage him at all. Your sensors will tell the story.
    - Damage control will let you know when out-of-commission devices have been completely repaired.

9. Your engines will automatically shut down if you should attempt to leave the galaxy, or if you should try to maneuver through a star, or Starbase, or—heaven help you—a Klingon warship.

10. In a pinch, or if you should miscalculate slightly, some shield control energy will be automatically diverted to warp engine control (if your shield are operational!).

11. While you’re docked at a Starbase, a team of technicians can repair your ship (if you’re willing for them to spend the time required—and the repairmen _always_ underestimate…)

12. If, to same maneuvering time toward the end of the game, you should cold-bloodedly destroy a Starbase, you get a nasty note from Starfleet Command. If you destroy your _last_ Starbase, you lose the game! (For those who think this is too a harsh penalty, delete line 5360-5390, and you’ll just get a “you dumdum!”-type message on all future status reports.)

13. End game logic has been “cleaned up” in several spots, and it is possible to get a new command after successfully completing your mission (or, after resigning your old one).

14. For those of you with certain types of CRT/keyboards setups (e.g. Westinghouse 1600), a “bell” character is inserted at appropriate spots to cause the following items to flash on and off on the screen:
    - The Phrase “\*RED\*” (as in Condition: Red)
    - The character representing your present quadrant in the cumulative galactic record printout.

15. This version of Star Trek was created for a Data General Nova 800 system with 32K or core. So that it would fit, the instructions are separated from the main program via a CHAIN. For conversion to DEC BASIC-PLUS, Statement 160 (Randomize) should be moved after the return from the chained instructions, say to Statement 245. For Altair BASIC, Randomize and the chain instructions should be eliminated.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=157)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=166)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

Instructions in this directory at
instructions.txt

#### Porting Notes

Many of the programs in this book and this collection have bugs in the original code.

@jkboyce has done a great job of discovering and fixing a number of bugs in the [original code](superstartrek.bas), as part of his [python implementation](python/superstartrek.py), which should be noted by other implementers:

- line `4410` : `D(7)` should be `D(6)`
- lines `8310`,`8330`,`8430`,`8450` : Division by zero is possible
- line `440` : `B9` should be initialised to 0, not 2


#### External Links
 - C++: https://www.codeproject.com/Articles/28399/The-Object-Oriented-Text-Star-Trek-Game-in-C


# `84_Super_Star_Trek/csharp/Game.cs`

This is a script for a Photon game. It appears to be a part of a Starbase in the Starbase Engine.

The script is using Photon's Unity game engine and Unity's Starbase module. It defines a ` PhotonHive` and a ` PhotonShieldControl` subclass, both of which are part of Photon's PhotonShieldProxy module. It also uses Photon's ` DamageControl` and ` LibraryComputer` modules.

The script is initialized in the `OnAddToScene` method and adds PhotonHive, PhotonShieldControl, DamageControl, and LibraryComputer to the scene.

The PhotonHive is configured to write commands to the console when it is ready to accept commands.

The script uses the `StartIn` method to start the enterprise in a specific quadrant, and the `BuildCurrentQuadrant` method to build a instance of the current quadrant.

It appears that the script is managing the enterprise's state, including its quadrant, stardate, and Galactic Record status. It also appears to be handling damage calculations and status reports.

Overall, it looks like the script is a part of a larger game and is responsible for managing various aspects of the game's state.


```
using System;
using Games.Common.IO;
using Games.Common.Randomness;
using SuperStarTrek.Objects;
using SuperStarTrek.Resources;
using SuperStarTrek.Space;
using SuperStarTrek.Systems;
using SuperStarTrek.Systems.ComputerFunctions;

namespace SuperStarTrek;

internal class Game
{
    private readonly TextIO _io;
    private readonly IRandom _random;

    private int _initialStardate;
    private int _finalStarDate;
    private float _currentStardate;
    private Coordinates _currentQuadrant;
    private Galaxy _galaxy;
    private int _initialKlingonCount;
    private Enterprise _enterprise;

    internal Game(TextIO io, IRandom random)
    {
        _io = io;
        _random = random;
    }

    internal float Stardate => _currentStardate;

    internal float StardatesRemaining => _finalStarDate - _currentStardate;

    internal void DoIntroduction()
    {
        _io.Write(Strings.Title);

        if (_io.GetYesNo("Do you need instructions", IReadWriteExtensions.YesNoMode.FalseOnN))
        {
            _io.Write(Strings.Instructions);

            _io.WaitForAnyKeyButEnter("to continue");
        }
    }

    internal void Play()
    {
        Initialise();
        var gameOver = false;

        while (!gameOver)
        {
            var command = _io.ReadCommand();

            var result = _enterprise.Execute(command);

            gameOver = result.IsGameOver || CheckIfStranded();
            _currentStardate += result.TimeElapsed;
            gameOver |= _currentStardate > _finalStarDate;
        }

        if (_galaxy.KlingonCount > 0)
        {
            _io.Write(Strings.EndOfMission, _currentStardate, _galaxy.KlingonCount);
        }
        else
        {
            _io.Write(Strings.Congratulations, CalculateEfficiency());
        }
    }

    private void Initialise()
    {
        _currentStardate = _initialStardate = _random.Next(20, 40) * 100;
        _finalStarDate = _initialStardate + _random.Next(25, 35);

        _currentQuadrant = _random.NextCoordinate();

        _galaxy = new Galaxy(_random);
        _initialKlingonCount = _galaxy.KlingonCount;

        _enterprise = new Enterprise(3000, _random.NextCoordinate(), _io, _random);
        _enterprise
            .Add(new WarpEngines(_enterprise, _io))
            .Add(new ShortRangeSensors(_enterprise, _galaxy, this, _io))
            .Add(new LongRangeSensors(_galaxy, _io))
            .Add(new PhaserControl(_enterprise, _io, _random))
            .Add(new PhotonTubes(10, _enterprise, _io))
            .Add(new ShieldControl(_enterprise, _io))
            .Add(new DamageControl(_enterprise, _io))
            .Add(new LibraryComputer(
                _io,
                new CumulativeGalacticRecord(_io, _galaxy),
                new StatusReport(this, _galaxy, _enterprise, _io),
                new TorpedoDataCalculator(_enterprise, _io),
                new StarbaseDataCalculator(_enterprise, _io),
                new DirectionDistanceCalculator(_enterprise, _io),
                new GalaxyRegionMap(_io, _galaxy)));

        _io.Write(Strings.Enterprise);
        _io.Write(
            Strings.Orders,
            _galaxy.KlingonCount,
            _finalStarDate,
            _finalStarDate - _initialStardate,
            _galaxy.StarbaseCount > 1 ? "are" : "is",
            _galaxy.StarbaseCount,
            _galaxy.StarbaseCount > 1 ? "s" : "");

        _io.WaitForAnyKeyButEnter("when ready to accept command");

        _enterprise.StartIn(BuildCurrentQuadrant());
    }

    private Quadrant BuildCurrentQuadrant() => new(_galaxy[_currentQuadrant], _enterprise, _random, _galaxy, _io);

    internal bool Replay() => _galaxy.StarbaseCount > 0 && _io.ReadExpectedString(Strings.ReplayPrompt, "Aye");

    private bool CheckIfStranded()
    {
        if (_enterprise.IsStranded) { _io.Write(Strings.Stranded); }
        return _enterprise.IsStranded;
    }

    private float CalculateEfficiency() =>
        1000 * (float)Math.Pow(_initialKlingonCount / (_currentStardate - _initialStardate), 2);
}

```

# `84_Super_Star_Trek/csharp/IRandomExtensions.cs`

这段代码是一个自定义的 `IRandomExtensions` 类，它提供了对 `IRandom` 类的扩展，用于生成更加随机的随机数。

首先，它实现了 `IRandom.Next1To8Inclusive` 方法，用于生成 1 到 8 之间的随机整数。它的实现是通过让 `random.NextFloat()` 生成一个介于 0 和 1 之间的浮点数，然后将其乘以 7.98 并加上 1.01，从而得到一个介于 1 到 8 之间的随机整数。这个方法生成了一个从 1 到 8 之间的随机整数，但稍微偏移了一些，因为它包含了一个轻微的偏差，远离了 1 和 8。

其次，它实现了 `IRandom.Next1To8Inclusive` 方法，用于生成 1 到 8 之间的随机整数。它的实现与 `IRandom.Next1To8Inclusive` 方法相同，但使用了另一个算法，这个算法更加精确地实现了从 1 到 8 之间的随机整数生成。这个算法通过让 `random.NextFloat()` 生成一个介于 0 和 1 之间的浮点数，然后将其乘以 7.98 并加上 0.98，再将结果四舍五入到整数。这个方法生成的整数更加精确，但是生成的随机数范围稍微偏移了一些。


```
using Games.Common.Randomness;
using SuperStarTrek.Space;

namespace SuperStarTrek;

internal static class IRandomExtensions
{
    internal static Coordinates NextCoordinate(this IRandom random) =>
        new Coordinates(random.Next1To8Inclusive() - 1, random.Next1To8Inclusive() - 1);

    // Duplicates the algorithm used in the original code to get an integer value from 1 to 8, inclusive:
    //     475 DEF FNR(R)=INT(RND(R)*7.98+1.01)
    // Returns a value from 1 to 8, inclusive.
    // Note there's a slight bias away from the extreme values, 1 and 8.
    internal static int Next1To8Inclusive(this IRandom random) => (int)(random.NextFloat() * 7.98 + 1.01);
}

```

# `84_Super_Star_Trek/csharp/IReadWriteExtensions.cs`

This is a context-aware command and prompt application written in C#. It allows the user to interact with the console in a Yes/No manner.

The `ReadCommand` method reads a command from the user and returns it. The user is prompted to enter a command one of the following:

* One of the specified commands.
* To exit the program.

The `ReadCourse` method reads a course from the user and returns it. The user is prompted to enter a direction one of the following:

* North.
* South.
* East.
* West.
* To exit the program.

The `TryReadCourse` method attempts to read a course from the user. If the user enters incorrect data, it will return `false`, otherwise it will return the course.

The `GetYesNo` method reads a Yes/No value from the user.

It is important to note that `TryReadNumberInRange` is a method that reads a number from a specified range. It requires a `ReadWrite` object to be provided to it, which is not provided in this code snippet.


```
using System;
using System.Linq;
using Games.Common.IO;
using SuperStarTrek.Commands;
using SuperStarTrek.Space;
using static System.StringComparison;

namespace SuperStarTrek;

internal static class IReadWriteExtensions
{
    internal static void WaitForAnyKeyButEnter(this IReadWrite io, string prompt)
    {
        io.Write($"Hit any key but Enter {prompt} ");
        while (io.ReadCharacter() == '\r');
    }

    internal static (float X, float Y) GetCoordinates(this IReadWrite io, string prompt) =>
        io.Read2Numbers($"{prompt} (X,Y)");

    internal static bool TryReadNumberInRange(
        this IReadWrite io,
        string prompt,
        float minValue,
        float maxValue,
        out float value)
    {
        value = io.ReadNumber($"{prompt} ({minValue}-{maxValue})");

        return value >= minValue && value <= maxValue;
    }

    internal static bool ReadExpectedString(this IReadWrite io, string prompt, string trueValue) =>
        io.ReadString(prompt).Equals(trueValue, InvariantCultureIgnoreCase);

    internal static Command ReadCommand(this IReadWrite io)
    {
        while(true)
        {
            var response = io.ReadString("Command");

            if (response.Length >= 3 &&
                Enum.TryParse(response.Substring(0, 3), ignoreCase: true, out Command parsedCommand))
            {
                return parsedCommand;
            }

            io.WriteLine("Enter one of the following:");
            foreach (var command in Enum.GetValues(typeof(Command)).OfType<Command>())
            {
                io.WriteLine($"  {command}  ({command.GetDescription()})");
            }
            io.WriteLine();
        }
    }

    internal static bool TryReadCourse(this IReadWrite io, string prompt, string officer, out Course course)
    {
        if (!io.TryReadNumberInRange(prompt, 1, 9, out var direction))
        {
            io.WriteLine($"{officer} reports, 'Incorrect course data, sir!'");
            course = default;
            return false;
        }

        course = new Course(direction);
        return true;
    }

    internal static bool GetYesNo(this IReadWrite io, string prompt, YesNoMode mode)
    {
        var response = io.ReadString($"{prompt} (Y/N)").ToUpperInvariant();

        return (mode, response) switch
        {
            (YesNoMode.FalseOnN, "N") => false,
            (YesNoMode.FalseOnN, _) => true,
            (YesNoMode.TrueOnY, "Y") => true,
            (YesNoMode.TrueOnY, _) => false,
            _ => throw new ArgumentOutOfRangeException(nameof(mode), mode, "Invalid value")
        };
    }

    internal enum YesNoMode
    {
        TrueOnY,
        FalseOnN
    }
}

```

# `84_Super_Star_Trek/csharp/Program.cs`

这段代码是一个24K内存的程序，它是一个基于《星际迷航》电视节目外观的虚拟星舰图。它由Mike Mayfield于1978年制作，并在1974年的《101基础游戏》中发布。这个程序最初是由Bob Leedom帮助编写的，并且经过调试和修改。


```
﻿// SUPER STARTREK - MAY 16,1978 - REQUIRES 24K MEMORY
//
// ****         **** STAR TREK ****        ****
// ****  SIMULATION OF A MISSION OF THE STARSHIP ENTERPRISE,
// ****  AS SEEN ON THE STAR TREK TV SHOW.
// ****  ORIGIONAL PROGRAM BY MIKE MAYFIELD, MODIFIED VERSION
// ****  PUBLISHED IN DEC'S "101 BASIC GAMES", BY DAVE AHL.
// ****  MODIFICATIONS TO THE LATTER (PLUS DEBUGGING) BY BOB
// ****  LEEDOM - APRIL & DECEMBER 1974,
// ****  WITH A LITTLE HELP FROM HIS FRIENDS . . .
// ****  COMMENTS, EPITHETS, AND SUGGESTIONS SOLICITED --
// ****  SEND TO:  R. C. LEEDOM
// ****            WESTINGHOUSE DEFENSE & ELECTRONICS SYSTEMS CNTR.
// ****            BOX 746, M.S. 338
// ****            BALTIMORE, MD  21203
```

该代码是一个用于将文本文件中的行转换为Microsoft C#代码的程序。它主要的作用是将保存的游戏文本文件中的每一行转换为等效的C#代码。转换后的代码会保存为同一个文件，但每个文件会保留原始文本行中的注释。

具体实现包括以下步骤：

1. 读取文件中的行并将其保存到一个变量中；
2. 使用游戏中的随机数生成器生成一个随机整数；
3. 如果生成的随机整数是奇数，则使用下一个随机整数将生成的代码行号增加1；否则，将生成的代码行号设置为当前随机整数；
4. 如果生成的随机整数是偶数，则使用当前随机整数将生成的代码行号增加1，并将前一个代码行号设置为生成的随机整数的下一个奇数；
5. 将生成的代码行号输出并保存到游戏中的随机数生成器中。

由于使用了多个语句，因此代码会变得更复杂。同时，由于使用了多重语句，因此需要使用“?”来代替“打印”。当程序生成过多的代码行时，每个文件可能会变得非常长，因此需要使用随机数生成器来控制生成的行数。


```
// ****
// ****  CONVERTED TO MICROSOFT 8 K BASIC 3/16/78 BY JOHN GORDERS
// ****  LINE NUMBERS FROM VERSION STREK7 OF 1/12/75 PRESERVED AS
// ****  MUCH AS POSSIBLE WHILE USING MULTIPLE STATEMENTS PER LINE
// ****  SOME LINES ARE LONGER THAN 72 CHARACTERS; THIS WAS DONE
// ****  BY USING "?" INSTEAD OF "PRINT" WHEN ENTERING LINES
// ****
// ****  CONVERTED TO MICROSOFT C# 2/20/21 BY ANDREW COOPER
// ****

using Games.Common.IO;
using Games.Common.Randomness;
using SuperStarTrek;

var io = new ConsoleIO();
```

这段代码的作用是创建一个随机数生成器实例，并将其赋值给变量random。然后，创建一个名为game的实例，该实例包含一个IO流和一个random变量。random变量用于生成随机数。

game instance上有一个Do-while循环，只要它仍在运行，就会重复执行game.DoIntroduction()和game.Play()方法。

game.DoIntroduction()方法可能是加载游戏数据或初始化游戏的代码，但并不包含游戏循环的逻辑。game.Play()方法则是游戏的main方法，其中包含游戏的所有核心逻辑。

在循环中，game.Replay()方法用于显示游戏中的所有事件和replay。这个方法可能会处理游戏循环中的一些异步操作，比如从玩家处读取游戏反馈，或在游戏完成时执行一些清理工作。但具体是什么操作，取决于game实例的实现方式。


```
var random = new RandomNumberGenerator();

var game = new Game(io, random);

game.DoIntroduction();

do
{
    game.Play();
} while (game.Replay());

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `84_Super_Star_Trek/csharp/StringExtensions.cs`

这段代码是一个名为`StringExtensions`的内部类，属于`SuperStarTrek`命名空间。该类包含一个名为`pluralize`的静态方法，该方法接受一个`string`类型的参数和一个`int`类型的参数（表示要复数的数量）。

`pluralize`方法的作用是将`string`类型的参数的单数形式（也就是`singular`）和`int`类型的参数（表示复数的数量）连接起来，形成一个新的字符串，如果复数的数量大于1，就在新字符串的末尾添加一个`s`。通过调用该方法，可以方便地将一个`string`类型和一个`int`类型的参数连接起来，形成一个新的字符串，该字符串表示指定数量的复数。例如：

```
String str = "T trek";
int quantity = 3;
String result = StringExtensions.pluralize(str, quantity);

System.out.println(result);  // 输出 " Trekkm"
```

在上述示例中，`StringExtensions.pluralize`方法将`"T trek"`和3连接起来，形成一个新的字符串"Trekkm"，其中`"s"`表示复数的数量为3，因此最终输出结果为"Trekkm"。


```
namespace SuperStarTrek;

internal static class StringExtensions
{
    internal static string Pluralize(this string singular, int quantity) => singular + (quantity > 1 ? "s" : "");
}

```

# `84_Super_Star_Trek/csharp/Commands/Command.cs`

这是一个定义了 Internal 枚举类型的代码。这个枚举类型定义了 9 个不同的命令，并且每个命令都有一个描述。但是，由于这个枚举类型是内部使用的，所以没有给出具体的枚举值。你可以按照需要自定义这个枚举类型。


```
using System.ComponentModel;

namespace SuperStarTrek.Commands;

internal enum Command
{
    [Description("To set course")]
    NAV,

    [Description("For short range sensor scan")]
    SRS,

    [Description("For long range sensor scan")]
    LRS,

    [Description("To fire phasers")]
    PHA,

    [Description("To fire photon torpedoes")]
    TOR,

    [Description("To raise or lower shields")]
    SHE,

    [Description("For damage control reports")]
    DAM,

    [Description("To call on library-computer")]
    COM,

    [Description("To resign your command")]
    XXX
}

```

# `84_Super_Star_Trek/csharp/Commands/CommandExtensions.cs`



这段代码是一个命令扩展类，目的是为了提供一种简便的方式来获取命令的描述信息。该命令扩展类使用反射来获取命令的类型，并使用该类型中定义的GetField方法获取命令的类型字段。然后，它使用GetCustomAttribute方法获取命令类型字段中定义的DescriptionAttribute类型，并使用该类型的description属性来获取命令的描述信息。最后，命令的类型字段将自动生成一个默认的DescriptionAttribute对象，如果还没有定义该类型的话。

因此，该代码的作用是提供一个方便的方式来获取命令的描述信息，从而使代码更加易于理解和维护。


```
using System.Reflection;
using System.ComponentModel;

namespace SuperStarTrek.Commands;

internal static class CommandExtensions
{
    internal static string GetDescription(this Command command) =>
        typeof(Command)
            .GetField(command.ToString())
            .GetCustomAttribute<DescriptionAttribute>()
            .Description;
}

```

# `84_Super_Star_Trek/csharp/Commands/CommandResult.cs`

这段代码定义了一个名为"Commands"的命名空间，其中定义了一个名为"CommandResult"的内部类。

"CommandResult"内部类包含两个公有成员变量，一个是"IsGameOver"，另一个是"TimeElapsed"。

"IsGameOver"变量是一个布尔值，初始值为false。它用于指示命令是否已经成功，如果为true，则表示游戏已结束，否则表示游戏还在进行中。

"TimeElapsed"变量是一个浮点数，用于记录命令执行所花费的时间。它用于在游戏中记录时间统计和记录游戏胜负等目的。

另外，还定义了一个名为"Elapsed"的静态方法，用于将指定的时间延迟后返回一个名为"CommandResult.Elapsed"的命令结果，延迟的时间由传入的"timeElapsed"参数指定。


```
namespace SuperStarTrek.Commands;

internal class CommandResult
{
    public static readonly CommandResult Ok = new(false);
    public static readonly CommandResult GameOver = new(true);

    private CommandResult(bool isGameOver)
    {
        IsGameOver = isGameOver;
    }

    private CommandResult(float timeElapsed)
    {
        TimeElapsed = timeElapsed;
    }

    public bool IsGameOver { get; }
    public float TimeElapsed { get; }

    public static CommandResult Elapsed(float timeElapsed) => new(timeElapsed);
}

```

# `84_Super_Star_Trek/csharp/Objects/Enterprise.cs`

This is a type of code that appears to define a method for controlling the movement of a shield, represented by the `ShieldControl` class. The method is called `MoveWithinQuadrant`, which takes a `Course` object and a distance in units of the quadrant (e.g. 1 unit = 100 meters), and returns either a tuple of the final quadrant coordinates and the current quadrant or a tuple of the coordinates of the destination quadrant.

The `MoveWithinQuadrant` method takes a `Course` object and a distance in units of the quadrant (e.g. 1 unit = 100 meters), and returns either a tuple of the final quadrant coordinates and the current quadrant or a tuple of the coordinates of the destination quadrant.

The `MoveBeyondQuadrant` method takes a `Course` object and a distance in units of the quadrant (e.g. 1 unit = 100 meters), and returns either the complete quadrant coordinates and the sector or the sector coordinates of the destination quadrant.

The `ShiftCompensate` method appears to take a `double[]` array of the original quadrant coordinates, the original distance in units of the quadrant, the new distance in units of the quadrant, and the new quadrant coordinates, and returns the transformed quadrant coordinates.

The `GetOverallTimeElapsed` method appears to return the time elapsed in seconds from the start of the maneuver, taking into account the time elapsed in each quadrant, the time elapsed in the warp engine, and the distance moved by the shield.


```
using System;
using System.Collections.Generic;
using System.Linq;
using Games.Common.IO;
using Games.Common.Randomness;
using SuperStarTrek.Commands;
using SuperStarTrek.Resources;
using SuperStarTrek.Space;
using SuperStarTrek.Systems;

namespace SuperStarTrek.Objects;

internal class Enterprise
{
    private readonly int _maxEnergy;
    private readonly IReadWrite _io;
    private readonly List<Subsystem> _systems;
    private readonly Dictionary<Command, Subsystem> _commandExecutors;
    private readonly IRandom _random;
    private Quadrant _quadrant;

    public Enterprise(int maxEnergy, Coordinates sector, IReadWrite io, IRandom random)
    {
        SectorCoordinates = sector;
        TotalEnergy = _maxEnergy = maxEnergy;

        _systems = new List<Subsystem>();
        _commandExecutors = new Dictionary<Command, Subsystem>();
        _io = io;
        _random = random;
    }

    internal Quadrant Quadrant => _quadrant;

    internal Coordinates QuadrantCoordinates => _quadrant.Coordinates;

    internal Coordinates SectorCoordinates { get; private set; }

    internal string Condition => GetCondition();

    internal LibraryComputer Computer => (LibraryComputer)_commandExecutors[Command.COM];

    internal ShieldControl ShieldControl => (ShieldControl)_commandExecutors[Command.SHE];

    internal float Energy => TotalEnergy - ShieldControl.ShieldEnergy;

    internal float TotalEnergy { get; private set; }

    internal int DamagedSystemCount => _systems.Count(s => s.IsDamaged);

    internal IEnumerable<Subsystem> Systems => _systems;

    internal PhotonTubes PhotonTubes => (PhotonTubes)_commandExecutors[Command.TOR];

    internal bool IsDocked => _quadrant.EnterpriseIsNextToStarbase;

    internal bool IsStranded => TotalEnergy < 10 || Energy < 10 && ShieldControl.IsDamaged;

    internal Enterprise Add(Subsystem system)
    {
        _systems.Add(system);
        _commandExecutors[system.Command] = system;

        return this;
    }

    internal void StartIn(Quadrant quadrant)
    {
        _quadrant = quadrant;
        quadrant.Display(Strings.StartText);
    }

    private string GetCondition() =>
        IsDocked switch
        {
            true => "Docked",
            false when _quadrant.HasKlingons => "*Red*",
            false when Energy / _maxEnergy < 0.1f => "Yellow",
            false => "Green"
        };

    internal CommandResult Execute(Command command)
    {
        if (command == Command.XXX) { return CommandResult.GameOver; }

        return _commandExecutors[command].ExecuteCommand(_quadrant);
    }

    internal void Refuel() => TotalEnergy = _maxEnergy;

    public override string ToString() => "<*>";

    internal void UseEnergy(float amountUsed)
    {
        TotalEnergy -= amountUsed;
    }

    internal CommandResult TakeHit(Coordinates sector, int hitStrength)
    {
        _io.WriteLine($"{hitStrength} unit hit on Enterprise from sector {sector}");
        ShieldControl.AbsorbHit(hitStrength);

        if (ShieldControl.ShieldEnergy <= 0)
        {
            _io.WriteLine(Strings.Destroyed);
            return CommandResult.GameOver;
        }

        _io.WriteLine($"      <Shields down to {ShieldControl.ShieldEnergy} units>");

        if (hitStrength >= 20)
        {
            TakeDamage(hitStrength);
        }

        return CommandResult.Ok;
    }

    private void TakeDamage(float hitStrength)
    {
        var hitShieldRatio = hitStrength / ShieldControl.ShieldEnergy;
        if (_random.NextFloat() > 0.6 || hitShieldRatio <= 0.02f)
        {
            return;
        }

        var system = _systems[_random.Next1To8Inclusive() - 1];
        system.TakeDamage(hitShieldRatio + 0.5f * _random.NextFloat());
        _io.WriteLine($"Damage Control reports, '{system.Name} damaged by the hit.'");
    }

    internal void RepairSystems(float repairWorkDone)
    {
        var repairedSystems = new List<string>();

        foreach (var system in _systems.Where(s => s.IsDamaged))
        {
            if (system.Repair(repairWorkDone))
            {
                repairedSystems.Add(system.Name);
            }
        }

        if (repairedSystems.Any())
        {
            _io.WriteLine("Damage Control report:");
            foreach (var systemName in repairedSystems)
            {
                _io.WriteLine($"        {systemName} repair completed.");
            }
        }
    }

    internal void VaryConditionOfRandomSystem()
    {
        if (_random.NextFloat() > 0.2f) { return; }

        var system = _systems[_random.Next1To8Inclusive() - 1];
        _io.Write($"Damage Control report:  {system.Name} ");
        if (_random.NextFloat() >= 0.6)
        {
            system.Repair(_random.NextFloat() * 3 + 1);
            _io.WriteLine("state of repair improved");
        }
        else
        {
            system.TakeDamage(_random.NextFloat() * 5 + 1);
            _io.WriteLine("damaged");
        }
    }

    internal float Move(Course course, float warpFactor, int distance)
    {
        var (quadrant, sector) = MoveWithinQuadrant(course, distance) ?? MoveBeyondQuadrant(course, distance);

        if (quadrant != _quadrant.Coordinates)
        {
            _quadrant = new Quadrant(_quadrant.Galaxy[quadrant], this, _random, _quadrant.Galaxy, _io);
        }
        _quadrant.SetEnterpriseSector(sector);
        SectorCoordinates = sector;

        TotalEnergy -= distance + 10;
        if (Energy < 0)
        {
            _io.WriteLine("Shield Control supplies energy to complete the maneuver.");
            ShieldControl.ShieldEnergy = Math.Max(0, TotalEnergy);
        }

        return GetTimeElapsed(quadrant, warpFactor);
    }

    private (Coordinates, Coordinates)? MoveWithinQuadrant(Course course, int distance)
    {
        var currentSector = SectorCoordinates;
        foreach (var (sector, index) in course.GetSectorsFrom(SectorCoordinates).Select((s, i) => (s, i)))
        {
            if (distance == 0) { break; }

            if (_quadrant.HasObjectAt(sector))
            {
                _io.WriteLine($"Warp engines shut down at sector {currentSector} dues to bad navigation");
                distance = 0;
                break;
            }

            currentSector = sector;
            distance -= 1;
        }

        return distance == 0 ? (_quadrant.Coordinates, currentSector) : null;
    }

    private (Coordinates, Coordinates) MoveBeyondQuadrant(Course course, int distance)
    {
        var (complete, quadrant, sector) = course.GetDestination(QuadrantCoordinates, SectorCoordinates, distance);

        if (!complete)
        {
            _io.Write(Strings.PermissionDenied, sector, quadrant);
        }

        return (quadrant, sector);
    }

    private float GetTimeElapsed(Coordinates finalQuadrant, float warpFactor) =>
        finalQuadrant == _quadrant.Coordinates
            ? Math.Min(1, (float)Math.Round(warpFactor, 1, MidpointRounding.ToZero))
            : 1;
}

```