# BasicComputerGames源码解析 75

# `82_Stars/java/src/StarsGame.java`

这段代码定义了一个名为 "StarsGame" 的类，其中包含一个名为 "main" 的方法。

在 "main" 方法中，使用 Java 语言内置的 "import" 关键字引入了三个外部类：

1. "java.lang.reflect" 包中的 "AnnotatedType" 类
2. "java.lang" 包中的 "Reflect" 类
3. "java.lang.reflect.metadata" 包中的 "Metadata" 类

在这三个类的引导下，定义了一个名为 "Stars" 的类，并使用 "new" 关键字创建了一个 "Stars" 的实例。

接下来，使用 "this.<class>()" 的形式获取 "Stars" 实例的 "play" 方法，并将其调用。由于没有提供参数，这个方法会默认执行不带参数的 "play" 方法。


```
import java.lang.reflect.AnnotatedType;

public class StarsGame {

    public static void main(String[] args) {
        Stars stars = new Stars();
        stars.play();
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


# `82_Stars/javascript/stars.js`

这段代码定义了两个函数，分别是`print()`和`input()`。这两个函数的主要作用如下：

1. `print()`函数的作用是在页面上打印输出一个字符串，接收一个字符串参数（`str`）。该函数将调用`document.getElementById()`获取一个元素（此处为`output`元素），然后使用`createTextNode()`创建一个字符串节点，并将其添加到指定的元素中。最后，将字符串输出到页面上。

2. `input()`函数的作用是接收一个输入字符（`input_str`），主要用途是监听用户在输入框中的点击事件。该函数会创建一个`<INPUT>`元素，设置其`type`属性为`text`，`length`属性为`50`（表示允许的最大字符数），并将元素添加到页面上。然后，函数会将输入的字符串存储在`input_str`变量中。

函数中使用了一个 Promise 对象（使用async/await语法实现），其作用是在用户点击输入框时，异步执行`input()`函数，并在函数完成时将结果（`input_str`）返回。


```
// STARS
//
// Converted from BASIC to Javascript by Qursch
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

This is a game where the player is prompted to guess a number that has been randomly generated. The game will output the number of guesses it takes for the player to guess the number correctly, or how many猜测是错误的， or how many星星失落。 The game will continue until the player decides to stop or if they have guessed the number correctly.


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var guesses = 7;
var limit = 100;

// Main program
async function main()
{
    print(tab(33) + "STARS\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n\n\n");

    // Instructions
    print("DO YOU WANT INSTRUCTIONS? (Y/N)");
    var instructions = await input();
    if(instructions.toLowerCase()[0] == "y") {
        print(`I AM THINKING OF A WHOLE NUMBER FROM 1 TO ${limit}\n`);
        print("TRY TO GUESS MY NUMBER.  AFTER YOU GUESS, I\n");
        print("WILL TYPE ONE OR MORE STARS (*).  THE MORE\n");
        print("STARS I TYPE, THE CLOSER YOU ARE TO MY NUMBER.\n");
        print("ONE STAR (*) MEANS FAR AWAY, SEVEN STARS (*******)\n");
        print(`MEANS REALLY CLOSE!  YOU GET ${guesses} GUESSES.\n\n\n`);
    }

    // Game loop
    while (true) {

        var randomNum = Math.floor(Math.random() * limit) + 1;
        var loss = true;

        print("\nOK, I AM THINKING OF A NUMBER, START GUESSING.\n\n");

        for(var guessNum=1; guessNum <= guesses; guessNum++) {

            // Input guess
            print("YOUR GUESS");
            var guess = parseInt(await input());

            // Check if guess is correct
            if(guess == randomNum) {
                loss = false;
                print("\n\n" + "*".repeat(50) + "!!!\n");
                print(`YOU GOT IT IN ${guessNum} GUESSES!!! LET'S PLAY AGAIN...\n`);
                break;
            }

            // Output distance in stars
            var dist = Math.abs(guess - randomNum);
            if(isNaN(dist)) print("*");
            else if(dist >= 64) print("*");
            else if(dist >= 32) print("**");
            else if(dist >= 16) print("***");
            else if(dist >= 8) print("****");
            else if(dist >= 4) print("*****");
            else if(dist >= 2) print("******");
            else print("*******")
            print("\n\n")
        }

        if(loss) {
            print(`SORRY, THAT'S ${guesses} GUESSES. THE NUMBER WAS ${randomNum}\n`);
        }
    }
}

```

这是 C 语言中的一个标准函数，名为 `main()`，它是程序的入口点。当程序运行时，首先会执行这个函数。

函数体中包含了一系列命令，按照从左到右的顺序执行。具体的执行顺序如下：
1. 首先，会输出字符串 `"Hello World"`，这是因为函数内部没有参数也没有返回值，直接输出一个字符串。
2. 然后，会计算一个整数表达式 `2 + 3 + 4`，结果为 9。并将这个结果赋值给变量 `i`，以便在后面的代码中使用。
3. 接着，会使用 `i` 的值作为参数，计算一个整数表达式 `i * i`，结果为 -1。将这个结果赋值给变量 `j`，以便在后面的代码中使用。
4. 然后，会使用变量 `j` 的值作为参数，计算一个整数表达式 `j / 5`，结果为 -1。将这个结果赋值给变量 `k`，以便在后面的代码中使用。
5. 接下来，会使用变量 `k` 的值作为参数，计算一个整数表达式 `k * k`，结果为 0。将这个结果赋值给变量 `l`，以便在后面的代码中使用。
6. 然后，会使用变量 `l` 的值作为参数，计算一个整数表达式 `l * 7`，结果为 49。
7. 接着，会使用变量 `i` 的值作为参数，计算一个整数表达式 `i--`，结果为 0。将这个结果赋值给变量 `p`，以便在后面的代码中使用。
8. 然后，会使用变量 `p` 的值作为参数，计算一个整数表达式 `p * p`，结果为 0。将这个结果赋值给变量 `q`，以便在后面的代码中使用。
9. 接下来，会使用变量 `j` 的值作为参数，计算一个整数表达式 `j <   5`，结果为 0。将这个结果赋值给变量 `r`，以便在后面的代码中使用。
10. 然后，会使用变量 `r` 的值作为参数，计算一个整数表达式 `r / 2`，结果为 0。将这个结果赋值给变量 `s`，以便在后面的代码中使用。
11. 接下来，会使用变量 `i` 的值作为参数，计算一个整数表达式 `i++`，结果为 1。将这个结果赋值给变量 `t`，以便在后面的代码中使用。
12. 然后，会使用变量 `t` 的值作为参数，计算一个整数表达式 `t > 2`，结果为 0。将这个结果赋值给变量 `u`，以便在后面的代码中使用。
13. 接下来，会使用变量 `u` 的值作为参数，计算一个整数表达式 `u < 6`，结果为 0。将这个结果赋值给变量 `v`，以便在后面的代码中使用。
14. 然后，会使用变量 `v` 的值作为参数，计算一个整数表达式 `v * 3`，结果为 0。将这个结果赋值给变量 `w`，以便在后面的代码中使用。
15. 接下来，会使用变量 `w` 的值作为参数，计算一个整数表达式 `w < 10`，结果为 0。将这个结果赋值给变量 `x`，以便在后面的代码中使用。
16. 然后，会使用变量 `x` 的值作为参数，计算一个整数表达式 `x + 1`，结果为 1。将这个结果赋值给变量 `z`，以便在后面的代码中使用。
17. 最后，会使用变量 `z` 的值作为参数，计算一个整数表达式 `z < 2`，结果为 0。

上述代码会输出一个字符串 `"Hello World"`，然后计算一个整数表达式 `i + j + k + l + p + q + r + s + t + u + v + w + x + z + y + 2`，最终结果为 0。


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


# `82_Stars/python/stars.py`

这段代码是一个BASIC程序，用于计算一个随机数。程序接受一个整数作为输入，然后从1到100（或其他你设定的最大值）中随机选择一个数字。玩家需要猜测这个数字，计算机会给出提示，告诉玩家猜大了还是猜小了。当玩家猜对数字时，获得一个星号，当猜错时则会获得一个七星。程序最多允许玩家猜测7次。

程序的主要目的是提供一个探索数字的游戏，通过猜测数字并尝试不同的策略，让学生了解随机数的概念和应用。


```
"""
Stars

From: BASIC Computer Games (1978)
      Edited by David H. Ahl

"In this game, the computer selects a random number from 1 to 100
 (or any value you set [for MAX_NUM]).  You try to guess the number
 and the computer gives you clues to tell you how close you're
 getting.  One star (*) means you're far away from the number; seven
 stars (*******) means you're really close.  You get 7  guesses.

"On the surface this game is very similar to GUESS; however, the
 guessing strategy is quite different.  See if you can come up with
 one or more approaches to finding the mystery number.

```

这段代码是一个Python程序，目的是指导玩家猜测一个整数，并给出了一些游戏提示。具体来说，程序有以下几个主要部分：

1. 定义了一个名为`MAX_NUM`的变量，其值为100。这是一个上限，表示玩家的猜测范围。

2. 定义了一个名为`MAX_GUESSES`的变量，其值为7。这是玩家猜测正确的最大次数。

3. 定义了一个名为`print_instructions`的函数，该函数用于打印游戏提示。函数在程序中会被反复调用，从而在游戏中逐步指导玩家。

4. 在游戏提示中，程序会告诉玩家他的数字是1到100之间的某个整数。然后，程序会提示玩家猜测他的数字。程序会随机猜测一个数字，并提示玩家最多可以猜7次。

5. 如果玩家猜中了数字，程序会打印一条消息，告诉玩家猜中了多少星星。如果玩家猜错了，程序会提示玩家他的猜测离数字还有多远，并给出一个星星的数量。如果玩家猜了7次，程序会提示玩家这是一个危险的猜测，并告知玩家他的数字是离他非常远的数字。

总之，这个程序是一个简单的游戏，旨在指导玩家猜测一个整数。程序通过输出游戏提示，使用随机数生成器来猜测数字，并根据玩家的猜测结果提供提示。通过多次练习，玩家可以学会如何猜测一个整数，并逐渐提高猜测的准确性。


```
"Bob Albrecht of People's Computer Company created this game."


Python port by Jeff Jetton, 2019
"""


import random

# Some contants
MAX_NUM = 100
MAX_GUESSES = 7


def print_instructions() -> None:
    """Instructions on how to play"""
    print("I am thinking of a whole number from 1 to %d" % MAX_NUM)
    print("Try to guess my number.  After you guess, I")
    print("will type one or more stars (*).  The more")
    print("stars I type, the closer you are to my number.")
    print("one star (*) means far away, seven stars (*******)")
    print("means really close!  You get %d guesses." % MAX_GUESSES)


```

这段代码定义了两个函数，`print_stars` 和 `get_guess`。

函数 `print_stars` 接收两个参数 `secret_number` 和 `guess`。这个函数的作用是打印出 `guess` 与 `secret_number` 之间的差异所对应的星星数量。如果 `guess` 与 `secret_number` 完全相等，则不会输出任何星星。

函数 `get_guess` 接收一个参数 `prompt`，这个参数是一个字符串。这个函数的作用是不断地从用户那里获取一个整数的猜测，并返回这个猜测。函数会一直运行，直到用户输入的字符串不符合要求（不是数字）。

总的来说，这两个函数的主要目的是为了让用户输入一个星星数量，并在用户输入正确的数字时输出相应的星星数量。


```
def print_stars(secret_number, guess) -> None:
    diff = abs(guess - secret_number)
    stars = ""
    for i in range(8):
        if diff < 2**i:
            stars += "*"
    print(stars)


def get_guess(prompt: str) -> int:
    while True:
        guess_str = input(prompt)
        if guess_str.isdigit():
            guess = int(guess_str)
            return guess


```

这段代码是一个Python程序，主要目的是让玩家猜测一个三位数的数字，并在猜测过程中提示玩家计算机在思考的数字、给出前三猜测的数字，并在猜测正确或超过三猜后结束游戏。

具体来说，代码首先会提示玩家输入是否需要说明，如果需要说明，则会调用一个名为 `print_instructions` 的函数，该函数会输出游戏的规则和玩法。

接着，代码会进入一个无限循环，每次循环都会随机生成一个 1 到 99 之间的整数（MAX_NUM），并提示玩家猜测自己猜的数字，然后循环体会提示玩家继续猜测，直到猜中为止。如果三猜都没有猜中，则会显示游戏结束的消息。如果猜中了，则游戏结束，输出一条消息并询问玩家是否还想继续玩。


```
def main() -> None:
    # Display intro text
    print("\n                   Stars")
    print("Creative Computing  Morristown, New Jersey")
    print("\n\n")
    # "*** Stars - People's Computer Center, MenloPark, CA"

    response = input("Do you want instructions? ")
    if response.upper()[0] == "Y":
        print_instructions()

    still_playing = True
    while still_playing:

        # "*** Computer thinks of a number"
        secret_number = random.randint(1, MAX_NUM)
        print("\n\nOK, I am thinking of a number, start guessing.")

        # Init/start guess loop
        guess_number = 0
        player_has_won = False
        while (guess_number < MAX_GUESSES) and not player_has_won:

            print()
            guess = get_guess("Your guess? ")
            guess_number += 1

            if guess == secret_number:
                # "*** We have a winner"
                player_has_won = True
                print("**************************************************!!!")
                print(f"You got it in {guess_number} guesses!!!")

            else:
                print_stars(secret_number, guess)

            # End of guess loop

        # "*** Did not guess in [MAX_GUESS] guesses"
        if not player_has_won:
            print(f"\nSorry, that's {guess_number} guesses, number was {secret_number}")

        # Keep playing?
        response = input("\nPlay again? ")
        if response.upper()[0] != "Y":
            still_playing = False


```

这段代码是一个Python程序的if语句，它会判断当前程序是否作为主程序运行。如果是主程序，那么程序会执行main函数中的代码。

在if语句的两侧，分别定义了一个函数main，它是一个空函数，里面什么也没有做。

在代码的最后部分，有一些注释，说明了这段代码的用途，即提供一个简单的游戏。这个游戏应该是一个迭代式的猜数字游戏，每次玩家猜测一个数字，程序会告诉玩家他们猜的数字是高了还是低了，直到他们猜中为止。

总之，这段代码是一个简单的用于演示迭代式猜数字游戏程序。


```
if __name__ == "__main__":
    main()

######################################################################
#
# Porting Notes
#
#   The original program never exited--it just kept playing rounds
#   over and over.  This version asks to continue each time.
#
#
# Ideas for Modifications
#
#   Let the player know how many guesses they have remaining after
#   each incorrect guess.
```

这段代码是一个简单的Python程序，它的主要目的是询问玩家在游戏开始时选择一个技能等级，这个技能等级会影响到MAX_NUM和MAX_GUESSES两个变量。MAX_NUM是一个变量，它存储了玩家猜测的最大数值，而MAX_GUESSES是一个变量，它存储了玩家猜测的最大次数。

根据玩家的选择，MAX_NUM和MAX_GUESSES的值会在1到50和1到100之间进行相应的调整，使得MAX_NUM和MAX_GUESSES更加符合玩家的技能水平。这样，玩家可以在游戏中通过不同的技能等级来挑战不同难度的关卡，而程序会根据玩家的技能水平来设置合理的猜测的最大值和次数，使得游戏更加有趣和挑战性更高。


```
#
#   Ask the player to select a skill level at the start of the game,
#   which will affect the values of MAX_NUM and MAX_GUESSES.
#   For example:
#
#       Easy   = 8 guesses, 1 to 50
#       Medium = 7 guesses, 1 to 100
#       Hard   = 6 guesses, 1 to 200
#
######################################################################

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/)


<<<<<<< HEAD
#STARS

From: BASIC Computer Games (1978), edited by David H. Ahl

In this game, the computer selects a random number from 1 to 100
(or any value you set [for MAX_NUM]).  You try to guess the number
and the computer gives you clues to tell you how close you're
getting.  One star (*) means you're far away from the number; seven
stars (*******) means you're really close.  You get 7  guesses.

On the surface this game is very similar to GUESS; however, the
guessing strategy is quite different.  See if you can come up with
one or more approaches to finding the mystery number.

Bob Albrecht of People's Computer Company created this game.

## NOTES

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by JW Bruce

thanks to Jeff Jetton for his Python port which provide inspiration
=======
Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/)
>>>>>>> 3e27c70ca800f5efbe6bc1a7d180211decf55b7d


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Stock Market

This program “plays” the stock market. You will be given $10,000 and may buy or sell stocks. Stock prices and trends are generated randomly; therefore, this model does not represent exactly what happens on the exchange. (Depending upon your point of view, you may feel this is quite a good representation!)

Every trading day, a table of stocks, their prices, and number of shares in your portfolio is printed. Following this, the initials of each stock are printed followed by a question mark. You indicate your transaction in number of shares — a positive number to buy, negative to sell, or 0 to do no trading. A brokerage fee of 1% is charges on all transactions (a bargain!). Note: Even if the value of a stock drops to zero, it may rebound again — then again, it may not.

This program was created by D. Pessel, L. Braun, and C. Losik of the Huntington Computer Project at SUNY, Stony Brook, N.Y.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=154)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=166)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `83_Stock_Market/csharp/Assets.cs`

这段代码定义了一个名为Assets的类，用于存储玩家拥有的资产。该类包含两个属性：Cash和Portfolio。

作用：该代码创建了一个名为Assets的类，用于在游戏引擎中代表玩家资产。Cash属性表示玩家拥有的现金金额，Portfolio属性是一个ImmutableArray，用于存储玩家拥有的每个公司的股票数量。这些属性可以用来在游戏中查看玩家拥有的资产和财富。


```
﻿using System.Collections.Immutable;

namespace Game
{
    /// <summary>
    /// Stores the player's assets.
    /// </summary>
    public record Assets
    {
        /// <summary>
        /// Gets the player's amount of cash.
        /// </summary>
        public double Cash { get; init; }

        /// <summary>
        /// Gets the number of stocks owned of each company.
        /// </summary>
        public ImmutableArray<int> Portfolio { get; init; }
    }
}

```

# `83_Stock_Market/csharp/Broker.cs`

This function appears to be a higher-level version of a set of stocks functions that takes an assets object, a list of stock transactions, and a list of company objects. The function appears to apply the transactions to the assets and return the new assets with the updated portfolio and a code indicating the result of the transaction.

The function takes a specific order of the assets, transactions and companies, first it applies the transactions to the assets and calculate the net cost and the transaction size. Then it calculate the brokerage fee and create the new assets object, if the new assets formation is successful it will return the new assets object and the transaction result.

It is worth noting that the function assumes the company have a SharePrice property, and also assumes the function is called on the company rather than the assets.

Please let me know if there is anything else I can help with.


```
﻿using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;

namespace Game
{
    /// <summary>
    /// Contains functions for exchanging assets.
    /// </summary>
    public static class Broker
    {
        /// <summary>
        /// Applies the given set of transactions to the given set of assets.
        /// </summary>
        /// <param name="assets">
        /// The assets to update.
        /// </param>
        /// <param name="transactions">
        /// The set of stocks to purchase or sell.  Positive values indicate
        /// purchaes and negative values indicate sales.
        /// </param>
        /// <param name="companies">
        /// The collection of companies.
        /// </param>
        /// <returns>
        /// Returns the sellers new assets and a code indicating the result
        /// of the transaction.
        /// </returns>
        public static (Assets newAssets, TransactionResult result) Apply(Assets assets, IEnumerable<int> transactions, IEnumerable<Company> companies)
        {
            var (netCost, transactionSize) = Enumerable.Zip(
                    transactions,
                    companies,
                    (amount, company) => (amount * company.SharePrice))
                .Aggregate(
                    (netCost: 0.0, transactionSize: 0.0),
                    (accumulated, amount) => (accumulated.netCost + amount, accumulated.transactionSize + Math.Abs(amount)));

            var brokerageFee = 0.01 * transactionSize;

            var newAssets = assets with
            {
                Cash      = assets.Cash - netCost - brokerageFee,
                Portfolio = ImmutableArray.CreateRange(Enumerable.Zip(
                    assets.Portfolio,
                    transactions,
                    (sharesOwned, delta) => sharesOwned + delta))
            };

            if (newAssets.Portfolio.Any(amount => amount < 0))
                return (newAssets, TransactionResult.Oversold);
            else
            if (newAssets.Cash < 0)
                return (newAssets, TransactionResult.Overspent);
            else
                return (newAssets, TransactionResult.Ok);
        }
    }
}

```

# `83_Stock_Market/csharp/Company.cs`

这段代码定义了一个名为 `Company` 的数据类，用于表示一家公司。这个类包含三个属性：`Name`、`StockSymbol` 和 `SharePrice`，分别表示公司的名字、股票符号和当前股票价格。这个类的定义在 `public record Company` 修饰下，意味着所有包含 `Company` 的声明都必须使用这个定义。

虽然这个类定义了公司的属性，但它并不代表一个实际的公司。要创建一家公司，需要提供足够的信息以初始化这个类的实例。


```
﻿namespace Game
{
    /// <summary>
    /// Represents a company.
    /// </summary>
    public record Company
    {
        /// <summary>
        /// Gets the company's name.
        /// </summary>
        public string Name { get; }

        /// <summary>
        /// Gets the company's three letter stock symbol.
        /// </summary>
        public string StockSymbol { get; }

        /// <summary>
        /// Gets the company's current share price.
        /// </summary>
        public double SharePrice { get; init; }

        /// <summary>
        /// Initializes a new Company record.
        /// </summary>
        public Company(string name, string stockSymbol, double sharePrice) =>
            (Name, StockSymbol, SharePrice) = (name, stockSymbol, sharePrice);
    }
}

```

# `83_Stock_Market/csharp/Controller.cs`

This is a class written in C# that appears to be for a trading application. It includes a method for applying a transaction to a company's assets, as well as a method for getting the transaction amount for a given company. The `UpdateAssets` method takes a collection of companies and an optional transaction amount as input and returns the updated assets. The `GetTransactionAmount` method takes a company as input and returns the number of shares to buy or sell.

The class also includes a `View` class that is responsible for displaying information to the user. The `PromptEnterTransactions` method displays a dialog prompt for the user to enter the transaction amount and the company to enter. The `Broker.Apply` method is a method for applying the transaction to the company's assets. It takes the transaction amount and the company as input and returns the updated assets or an error message if the transaction could not be completed successfully.

Overall, this class appears to be a part of a larger trading application that allows users to buy or sell stocks.


```
﻿using System;
using System.Collections.Generic;
using System.Linq;

namespace Game
{
    public static class Controller
    {
        /// <summary>
        /// Manages the initial interaction with the user.
        /// </summary>
        public static void StartGame()
        {
            View.ShowBanner();

            var showInstructions = GetYesOrNo(View.PromptShowInstructions);
            View.ShowSeparator();
            if (showInstructions)
                View.ShowInstructions();

            View.ShowSeparator();
        }

        /// <summary>
        /// Gets a yes or no answer from the user.
        /// </summary>
        /// <param name="prompt">
        /// Displays the prompt.
        /// </param>
        /// <returns>
        /// True if the user answered yes and false if he or she answered no.
        /// </returns>
        public static bool GetYesOrNo(Action prompt)
        {
            prompt();

            var response = default(char);
            do
            {
                response = Console.ReadKey(intercept: true).KeyChar;
            }
            while (response != '0' && response != '1');

            View.ShowChar(response);
            return response == '1';
        }

        /// <summary>
        /// Gets a transaction amount for each company in the given collection
        /// of companies and returns the updated assets.
        /// </summary>
        /// <param name="assets">
        /// The assets to update.
        /// </param>
        /// <param name="companies">
        /// The collection of companies.
        /// </param>
        /// <returns>
        /// The updated assets.
        /// </returns>
        public static Assets UpdateAssets(Assets assets, IEnumerable<Company> companies)
        {
            while (true)
            {
                View.PromptEnterTransactions();

                var result = Broker.Apply (
                    assets,
                    companies.Select(GetTransactionAmount).ToList(),
                    companies);

                switch (result)
                {
                    case (Assets newAssets, TransactionResult.Ok):
                        return newAssets;
                    case (_, TransactionResult.Oversold):
                        View.ShowOversold();
                        break;
                    case (Assets newAssets, TransactionResult.Overspent):
                        View.ShowOverspent(-newAssets.Cash);
                        break;
                }
            }
        }

        /// <summary>
        /// Gets a transaction amount for the given company.
        /// </summary>
        /// <param name="company">
        /// The company to buy or sell.
        /// </param>
        /// <returns>
        /// The number of shares to buy or sell.
        /// </returns>
        public static int GetTransactionAmount(Company company)
        {
            while (true)
            {
                View.PromptBuySellCompany(company);

                var input = Console.ReadLine();
                if (input is null)
                    Environment.Exit(0);
                else
                if (!Int32.TryParse(input, out var amount))
                    View.PromptValidInteger();
                else
                    return amount;
            }
        }
    }
}

```

# `83_Stock_Market/csharp/Program.cs`

这段代码是一个模拟金融市场的游戏。在这个游戏中，玩家需要从一个预定义的公司资产组合中购买和出售股票。这个资产组合包括不同类型的公司，例如军火公司、银行、飞机制造公司等等。这些公司的股票价格是浮动的，根据市场情况会有所变化。

在游戏中，玩家通过钱或者股票来购买资产。资产购买和出售是通过一个叫做`Controller`的类来实现的。这个类提供了一些方法，如`StartGame`、`Simulate`、`ShowCompanies`、`ShowTradeResults`、`ShowAssets`等方法来操作游戏界面和数据库。

另外，这个游戏还提供了一个`View`类，用于显示游戏中的信息，例如公司的股票价格、资产组合、交易信息等等。


```
﻿using System;
using System.Collections.Immutable;
using System.Linq;

namespace Game
{
    class Program
    {
        /// <summary>
        /// Defines the set of companies that will be simulated in the game.
        /// </summary>
        private readonly static ImmutableArray<Company> Companies = ImmutableArray.CreateRange(new[]
        {
            new Company("INT. BALLISTIC MISSILES",     "IBM", sharePrice:100),
            new Company("RED CROSS OF AMERICA",        "RCA", sharePrice:85 ),
            new Company("LICHTENSTEIN, BUMRAP & JOKE", "LBJ", sharePrice:150),
            new Company("AMERICAN BANKRUPT CO.",       "ABC", sharePrice:140),
            new Company("CENSURED BOOKS STORE",        "CBS", sharePrice:110)
        });

        static void Main()
        {
            var assets = new Assets
            {
                Cash      = 10000.0,
                Portfolio = ImmutableArray.CreateRange(Enumerable.Repeat(0, Companies.Length))
            };

            var previousDay = default(TradingDay);

            Controller.StartGame();

            foreach (var day in StockMarket.Simulate(Companies))
            {
                if (previousDay is null)
                    View.ShowCompanies(day.Companies);
                else
                    View.ShowTradeResults(day, previousDay, assets);

                View.ShowAssets(assets, day.Companies);

                if (previousDay is not null && !Controller.GetYesOrNo(View.PromptContinue))
                    break;

                assets      = Controller.UpdateAssets(assets, day.Companies);
                previousDay = day;
            }

            View.ShowFarewell();
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `83_Stock_Market/csharp/StockMarket.cs`

This is a summary of a solution for generating an infinite sequence of random company indexes and null values for a market trend. The solution defines a static class called `Trends` which uses a method called `PriceSpikes` to generate the sequence.

The `PriceSpikes` method takes three parameters: `random`, `companyCount`, and `minDays` and `maxDays` and returns an infinite sequence of random integer values. The sequence is generated by randomly selecting the number of days between each price spike and using the square root of that number to generate the random integer values.

The `Trends` class also defines a method called `GenerateTrend` which generates a random value for the trend.

Note that this code snippet assumes that there is already a sequence of random numbers generated and that the number of companies is known.


```
﻿using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Game.Extensions;

namespace Game
{
    /// <summary>
    /// Provides a method for simulating a stock market.
    /// </summary>
    public static class StockMarket
    {
        /// <summary>
        /// Simulates changes in the stock market over time.
        /// </summary>
        /// <param name="companies">
        /// The collection of companies that will participate in the market.
        /// </param>
        /// <returns>
        /// An infinite sequence of trading days.  Each day represents the
        /// state of the stock market at the start of that day.
        /// </returns>
        public static IEnumerable<TradingDay> Simulate(ImmutableArray<Company> companies)
        {
            var random = new Random();

            var cyclicParameters = EnumerableExtensions.Zip(
                Trends(random, 1, 5),
                PriceSpikes(random, companies.Length, 1, 5),
                PriceSpikes(random, companies.Length, 1, 5),
                (trend, company1, company2) => (trend, positiveSpike: company1, negativeSpike: company2));

            return cyclicParameters.SelectAndAggregate(
                new TradingDay
                {
                    Companies = companies
                },
                (parameters, previousDay) => previousDay with
                {
                    Companies = previousDay.Companies.Map(
                        (company, index) => AdjustSharePrice(
                            random,
                            company,
                            parameters.trend,
                            parameters.positiveSpike == index,
                            parameters.negativeSpike == index))
                });
        }

        /// <summary>
        /// Creates a copy of a company with a randomly adjusted share price,
        /// based on the given parameters.
        /// </summary>
        /// <param name="random">
        /// The random number generator.
        /// </param>
        /// <param name="company">
        /// The company to adjust.
        /// </param>
        /// <param name="trend">
        /// The slope of the overall market price trend.
        /// </param>
        /// <param name="positiveSpike">
        /// True if the function should simulate a positive spike in the
        /// company's share price.
        /// </param>
        /// <param name="negativeSpike">
        /// True if the function should simulate a negative spike in the
        /// company's share price.
        /// </param>
        /// <returns>
        /// The adjusted company.
        /// </returns>
        private static Company AdjustSharePrice(Random random, Company company, double trend, bool positiveSpike, bool negativeSpike)
        {
            var boost = random.Next(4) * 0.25;

            var spikeAmount = 0.0;

            if (positiveSpike)
                spikeAmount = 10;

            if (negativeSpike)
                spikeAmount = spikeAmount - 10;

            var priceChange = (int)(trend * company.SharePrice) + boost + (int)(3.5 - (6 * random.NextDouble())) + spikeAmount;

            var newPrice = company.SharePrice + priceChange;
            if (newPrice < 0)
                newPrice = 0;

            return company with { SharePrice = newPrice };
        }

        /// <summary>
        /// Generates an infinite sequence of market trends.
        /// </summary>
        /// <param name="random">
        /// The random number generator.
        /// </param>
        /// <param name="minDays">
        /// The minimum number of days each trend should last.
        /// </param>
        /// <param name="maxDays">
        /// The maximum number of days each trend should last.
        /// </param>
        public static IEnumerable<double> Trends(Random random, int minDays, int maxDays) =>
            random.Integers(minDays, maxDays + 1).SelectMany(daysInCycle => Enumerable.Repeat(GenerateTrend(random), daysInCycle));

        /// <summary>
        /// Generates a random value for the market trend.
        /// </summary>
        /// <param name="random">
        /// The random number generator.
        /// </param>
        /// <returns>
        /// A trend value in the range [-0.1, 0.1].
        /// </returns>
        private static double GenerateTrend(Random random) =>
            ((int)(random.NextDouble() * 10 + 0.5) / 100.0) * (random.Next(2) == 0 ? 1 : -1) ;

        /// <summary>
        /// Generates an infinite sequence of price spikes.
        /// </summary>
        /// <param name="random">
        /// The random number generator.
        /// </param>
        /// <param name="companyCount">
        /// The number of companies.
        /// </param>
        /// <param name="minDays">
        /// The minimum number of days in between price spikes.
        /// </param>
        /// <param name="maxDays">
        /// The maximum number of days in between price spikes.
        /// </param>
        /// <returns>
        /// An infinite sequence of random company indexes and null values.
        /// A non-null value means that the corresponding company should
        /// experience a price spike.
        /// </returns>
        private static IEnumerable<int?> PriceSpikes(Random random, int companyCount, int minDays, int maxDays) =>
            random.Integers(minDays, maxDays + 1)
                .SelectMany(
                    daysInCycle => Enumerable.Range(0, daysInCycle),
                    (daysInCycle, dayNumber) => dayNumber == 0 ? random.Next(companyCount) : default(int?));
    }
}

```

# `83_Stock_Market/csharp/TradingDay.cs`

这段代码定义了一个名为 TradingDay 的类，用于表示单个交易日。该类包含两个静态字段，分别是平均股票价格和包含该日所有公开上市公司的 ImmutableArray。

平均股票价格字段通过调用 Companies.Average 方法获取该市场中所有公司的平均股票价格，并将它们存储在 Companies 静态字段中。

Companies 字段是一个 ImmutableArray，用于存储该市场中所有公开上市公司的集合。该类在静态构造函数中被初始化，并在需要时进行默认值填充。

该代码的目的是提供一个简单地表示单个交易日的方法，该方法可以获取市场的平均股票价格和对其中所有公开上市公司的集合。


```
﻿using System.Collections.Immutable;
using System.Linq;

namespace Game
{
    /// <summary>
    /// Represents a single trading day.
    /// </summary>
    public record TradingDay
    {
        /// <summary>
        /// Gets the average share price of all companies in the market this
        /// day.
        /// </summary>
        public double AverageSharePrice =>
            Companies.Average (company => company.SharePrice);

        /// <summary>
        /// Gets the collection of public listed companies in the stock market
        /// this day.
        /// </summary>
        public ImmutableArray<Company> Companies { get; init; }
    }
}

```

# `83_Stock_Market/csharp/TransactionResult.cs`

这段代码定义了一个枚举类型TransactionResult，描述了在应用事务时可能出现的不同结果。这个枚举类型包含三个枚举常量，分别是Ok、Oversold和Overshot。

在枚举常量中，每个常量都有一个默认的枚举类型成员函数，分别是TransactionResult.Ok、TransactionResult.Oversold和TransactionResult.Overshot。这些成员函数定义了枚举常量的含义，表示事务的不同结果。

当创建这个枚举类型时，就可以在代码中使用它，比如可以这样写：
```
enum TransactionResult
{
   /// <summary>
   /// The transaction was successful.
   /// </summary>
   TransactionResult.Ok,

   /// <summary>
   /// The transaction failed because the seller tried to sell more shares
   /// than he or she owns.
   /// </summary>
   TransactionResult.Oversold,

   /// <summary>
   /// The transaction failed because the net cost was greater than the
   /// seller's available cash.
   /// </summary>
   TransactionResult.Overspent
}
```
在枚举类型的后面，定义了一个Transactions类型的类，它包含了枚举类型中的所有成员函数，可以像通常的类一样使用它们。


```
﻿namespace Game
{
    /// <summary>
    /// Enumerates the different possible outcomes of applying a transaction.
    /// </summary>
    public enum TransactionResult
    {
        /// <summary>
        /// The transaction was successful.
        /// </summary>
        Ok,

        /// <summary>
        /// The transaction failed because the seller tried to sell more shares
        /// than he or she owns.
        /// </summary>
        Oversold,

        /// <summary>
        /// The transaction failed because the net cost was greater than the
        /// seller's available cash.
        /// </summary>
        Overspent
    }
}

```

# `83_Stock_Market/csharp/View.cs`

This is a class called `StockPrinter` that provides a class for printing stock information for a company. This class contains several methods for displaying different aspects of the company's stock, such as the current stock price, the total amount of cash assets, the total amount of stock, and more. The class also has a method for displaying overexpired and overspent messages if the user has overstepped their limits.

The class also contains methods for displaying a separator and a farewell message. Additionally, there are two prompts for the user to input more information. The first prompt is for the user to confirm if they want to see more instructions, and the second prompt is for the user to continue or exit the program.

The class also has a method for prompting the user to enter a valid integer for the amount they want to invest.


```
﻿using System;
using System.Collections.Generic;
using System.Linq;
using Game.Extensions;

namespace Game
{
    /// <summary>
    /// Contains functions for displaying information to the user.
    /// </summary>
    public static class View
    {
        public static void ShowBanner()
        {
            Console.WriteLine("                             STOCK MARKET");
            Console.WriteLine("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
        }

        public static void ShowInstructions()
        {
            Console.WriteLine("THIS PROGRAM PLAYS THE STOCK MARKET.  YOU WILL BE GIVEN");
            Console.WriteLine("$10,000 AND MAY BUY OR SELL STOCKS.  THE STOCK PRICES WILL");
            Console.WriteLine("BE GENERATED RANDOMLY AND THEREFORE THIS MODEL DOES NOT");
            Console.WriteLine("REPRESENT EXACTLY WHAT HAPPENS ON THE EXCHANGE.  A TABLE");
            Console.WriteLine("OF AVAILABLE STOCKS, THEIR PRICES, AND THE NUMBER OF SHARES");
            Console.WriteLine("IN YOUR PORTFOLIO WILL BE PRINTED.  FOLLOWING THIS, THE");
            Console.WriteLine("INITIALS OF EACH STOCK WILL BE PRINTED WITH A QUESTION");
            Console.WriteLine("MARK.  HERE YOU INDICATE A TRANSACTION.  TO BUY A STOCK");
            Console.WriteLine("TYPE +NNN, TO SELL A STOCK TYPE -NNN, WHERE NNN IS THE");
            Console.WriteLine("NUMBER OF SHARES.  A BROKERAGE FEE OF 1% WILL BE CHARGED");
            Console.WriteLine("ON ALL TRANSACTIONS.  NOTE THAT IF A STOCK'S VALUE DROPS");
            Console.WriteLine("TO ZERO IT MAY REBOUND TO A POSITIVE VALUE AGAIN.  YOU");
            Console.WriteLine("HAVE $10,000 TO INVEST.  USE INTEGERS FOR ALL YOUR INPUTS.");
            Console.WriteLine("(NOTE:  TO GET A 'FEEL' FOR THE MARKET RUN FOR AT LEAST");
            Console.WriteLine("10 DAYS)");
            Console.WriteLine("-----GOOD LUCK!-----");
        }

        public static void ShowCompanies(IEnumerable<Company> companies)
        {
            var maxNameLength = companies.Max(company => company.Name.Length);

            Console.WriteLine($"{"STOCK".PadRight(maxNameLength)} INITIALS      PRICE/SHARE");
            foreach (var company in companies)
                Console.WriteLine($"{company.Name.PadRight(maxNameLength)}   {company.StockSymbol}          {company.SharePrice:0.00}");

            Console.WriteLine();
            Console.WriteLine($"NEW YORK STOCK EXCHANGE AVERAGE: {companies.Average(company => company.SharePrice):0.00}");
            Console.WriteLine();
        }

        public static void ShowTradeResults(TradingDay day, TradingDay previousDay, Assets assets)
        {
            var results = EnumerableExtensions.Zip(
                day.Companies,
                previousDay.Companies,
                assets.Portfolio,
                (company, previous, shares) =>
                (
                    stockSymbol: company.StockSymbol,
                    price: company.SharePrice,
                    shares,
                    value: shares * company.SharePrice,
                    change: company.SharePrice - previous.SharePrice
                )).ToList();

            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine("**********     END OF DAY'S TRADING     **********");
            Console.WriteLine();
            Console.WriteLine();

            Console.WriteLine("STOCK\tPRICE/SHARE\tHOLDINGS\tVALUE\tNET PRICE CHANGE");
            foreach (var result in results)
                Console.WriteLine($"{result.stockSymbol}\t{result.price}\t\t{result.shares}\t\t{result.value:0.00}\t\t{result.change:0.00}");

            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();

            var averagePrice = day.AverageSharePrice;
            var averagePriceChange = averagePrice - previousDay.AverageSharePrice;

            Console.WriteLine($"NEW YORK STOCK EXCHANGE AVERAGE: {averagePrice:0.00} NET CHANGE {averagePriceChange:0.00}");
            Console.WriteLine();
        }

        public static void ShowAssets(Assets assets, IEnumerable<Company> companies)
        {
            var totalStockValue = Enumerable.Zip(
                assets.Portfolio,
                companies,
                (shares, company) => shares * company.SharePrice).Sum();

            Console.WriteLine($"TOTAL STOCK ASSETS ARE   ${totalStockValue:0.00}");
            Console.WriteLine($"TOTAL CASH ASSETS ARE    ${assets.Cash:0.00}");
            Console.WriteLine($"TOTAL ASSETS ARE         ${totalStockValue + assets.Cash:0.00}");
            Console.WriteLine();
        }

        public static void ShowOversold()
        {
            Console.WriteLine();
            Console.WriteLine("YOU HAVE OVERSOLD A STOCK; TRY AGAIN.");
        }

        public static void ShowOverspent(double amount)
        {
            Console.WriteLine();
            Console.WriteLine($"YOU HAVE USED ${amount:0.00} MORE THAN YOU HAVE.");
        }

        public static void ShowFarewell()
        {
            Console.WriteLine("HOPE YOU HAD FUN!!");
        }

        public static void ShowSeparator()
        {
            Console.WriteLine();
            Console.WriteLine();
        }

        public static void ShowChar(char c)
        {
            Console.WriteLine(c);
        }

        public static void PromptShowInstructions()
        {
            Console.Write("DO YOU WANT THE INSTRUCTIONS (YES-TYPE 1, NO-TYPE 0)? ");
        }

        public static void PromptContinue()
        {
            Console.Write("DO YOU WISH TO CONTINUE (YES-TYPE 1, NO-TYPE 0)? ");
        }

        public static void PromptEnterTransactions()
        {
            Console.WriteLine("WHAT IS YOUR TRANSACTION IN");
        }

        public static void PromptBuySellCompany(Company company)
        {
            Console.Write($"{company.StockSymbol}? ");
        }

        public static void PromptValidInteger()
        {
            Console.WriteLine("PLEASE ENTER A VALID INTEGER");
        }
    }
}

```

# `83_Stock_Market/csharp/Extensions/EnumerableExtensions.cs`

This is a function that takes in three sequences, `first`, `second`, and `third`, and combines them into a single sequence `result`. The function uses the `Func<T1, T2, T3, TResult>` type to specify that the result of the combination should be of type `TResult`.

The function works identically to the `Enumerable.Zip` method, except that it combines three sequences instead of two. It is defined as an extension method for consistency with the similar LINQ methods in the `Enumerable` class.

The `Zip` extension method is used to combine the sequences into a single sequence that may be easily accessed and processed.


```
﻿using System;
using System.Collections.Generic;
using System.Linq;

namespace Game.Extensions
{
    /// <summary>
    /// Provides additional methods for the <see cref="IEnumerable{T}"/>
    /// interface.
    /// </summary>
    public static class EnumerableExtensions
    {
        /// <summary>
        /// Simultaneously projects each element of a sequence and applies
        /// the result of the previous projection.
        /// </summary>
        /// <typeparam name="TSource">
        /// The type of elements in the source sequence.
        /// </typeparam>
        /// <typeparam name="TResult">
        /// The type of elements in the result sequence.
        /// </typeparam>
        /// <param name="source">
        /// The source sequence.
        /// </param>
        /// <param name="seed">
        /// The seed value for the aggregation component.  This value is
        /// passed to the first call to <paramref name="selector"/>.
        /// </param>
        /// <param name="selector">
        /// The projection function.  This function is supplied with a value
        /// from the source sequence and the result of the projection on the
        /// previous value in the source sequence.
        /// </param>
        /// <returns>
        /// The resulting sequence.
        /// </returns>
        public static IEnumerable<TResult> SelectAndAggregate<TSource, TResult>(
            this IEnumerable<TSource> source,
            TResult seed,
            Func<TSource, TResult, TResult> selector)
        {
            foreach (var element in source)
            {
                seed = selector(element, seed);
                yield return seed;
            }
        }

        /// <summary>
        /// Combines the results of three distinct sequences into a single
        /// sequence.
        /// </summary>
        /// <typeparam name="T1">
        /// The element type of the first sequence.
        /// </typeparam>
        /// <typeparam name="T2">
        /// The element type of the second sequence.
        /// </typeparam>
        /// <typeparam name="T3">
        /// The element type of the third sequence.
        /// </typeparam>
        /// <typeparam name="TResult">
        /// The element type of the resulting sequence.
        /// </typeparam>
        /// <param name="first">
        /// The first source sequence.
        /// </param>
        /// <param name="second">
        /// The second source sequence.
        /// </param>
        /// <param name="third">
        /// The third source sequence.
        /// </param>
        /// <param name="resultSelector">
        /// Function that combines results from each source sequence into a
        /// final result.
        /// </param>
        /// <returns>
        /// A sequence of combined values.
        /// </returns>
        /// <remarks>
        /// <para>
        /// This function works identically to Enumerable.Zip except that it
        /// combines three sequences instead of two.
        /// </para>
        /// <para>
        /// We have defined this as an extension method for consistency with
        /// the similar LINQ methods in the <see cref="Enumerable"/> class.
        /// However, since there is nothing special about the first sequence,
        /// it is often more clear to call this as a regular function.  For
        /// example:
        /// </para>
        /// <code>
        /// EnumerableExtensions.Zip(
        ///     sequence1,
        ///     sequence2,
        ///     sequence3,
        ///     (a, b, c) => GetResult (a, b, c));
        /// </code>
        /// </remarks>
        public static IEnumerable<TResult> Zip<T1, T2, T3, TResult>(
            this IEnumerable<T1> first,
            IEnumerable<T2> second,
            IEnumerable<T3> third,
            Func<T1, T2, T3, TResult> resultSelector)
        {
            using var enumerator1 = first.GetEnumerator();
            using var enumerator2 = second.GetEnumerator();
            using var enumerator3 = third.GetEnumerator();

            while (enumerator1.MoveNext() && enumerator2.MoveNext() && enumerator3.MoveNext())
                yield return resultSelector(enumerator1.Current, enumerator2.Current, enumerator3.Current);
        }
    }
}

```

# `83_Stock_Market/csharp/Extensions/ImmutableArrayExtensions.cs`

这段代码定义了一个名为 `ImmutableArrayExtensions` 的类，提供了对 `ImmutableArray<TSource>` 类的额外方法。

该类包含一个名为 `Map` 的方法，它接受一个 `ImmutableArray<TSource>` 和一个函数 `selector`，将每个元素从源数组中根据指定的选择器函数映射到新的元素类型中。

具体来说，这段代码首先定义了一个名为 `Map` 的类，其构造函数接受一个 `ImmutableArray<TSource>` 和一个名为 `selector` 的函数。在 `Map` 方法中，使用 `ImmutableArray.CreateBuilder<TResult>` 方法创建一个带有 `TSource` 类型长度的新数组，然后使用循环遍历源数组中的每个元素，将每个元素根据指定的选择器函数映射到新的元素类型中（传递给 `selector` 函数的参数为 `source[i]` 和 `i`）。最后，返回新数组构建后通过 `移动到Immutable` 方法得到的 `ImmutableArray<TResult>`。


```
﻿using System;
using System.Collections.Immutable;

namespace Game.Extensions
{
    /// <summary>
    /// Provides additional methods for the <see cref="ImmutableArray{T}"/> class.
    /// </summary>
    public static class ImmutableArrayExtensions
    {
        /// <summary>
        /// Maps each element in an immutable array to a new value.
        /// </summary>
        /// <typeparam name="TSource">
        /// The type of elements in the source array.
        /// </typeparam>
        /// <typeparam name="TResult">
        /// The type of elements in the resulting array.
        /// </typeparam>
        /// <param name="source">
        /// The source array.
        /// </param>
        /// <param name="selector">
        /// Function which receives an element from the source array and its
        /// index and returns the resulting element.
        /// </param>
        public static ImmutableArray<TResult> Map<TSource, TResult>(this ImmutableArray<TSource> source, Func<TSource, int, TResult> selector)
        {
            var builder = ImmutableArray.CreateBuilder<TResult>(source.Length);

            for (var i = 0; i < source.Length; ++i)
                builder.Add(selector(source[i], i));

            return builder.MoveToImmutable();
        }
    }
}

```

# `83_Stock_Market/csharp/Extensions/RandomExtensions.cs`



这段代码是一个扩展库，提供了对随机类(Random)的额外方法。主要作用是生成一个无限长度的随机数序列，可以通过调用生成的随机数序列，来生成各种随机的序列操作。通过调用生成的随机数序列，可以轻松地生成随机种子，随机数种子可以用来确保每次运行代码时生成的结果都不同。


```
﻿using System;
using System.Collections.Generic;

namespace Game.Extensions
{
    /// <summary>
    /// Provides additional methods for the <see cref="Random"/> class.
    /// </summary>
    public static class RandomExtensions
    {
        /// <summary>
        /// Generates an infinite sequence of random numbers.
        /// </summary>
        /// <param name="random">
        /// The random number generator.
        /// </param>
        /// <param name="min">
        /// The inclusive lower bound of the range to generate.
        /// </param>
        /// <param name="max">
        /// The exclusive upper bound of the range to generate.
        /// </param>
        /// <returns>
        /// An infinite sequence of random integers in the range [min, max).
        /// </returns>
        /// <remarks>
        /// <para>
        /// We use an exclusive upper bound, even though it's a little
        /// confusing, for the sake of consistency with Random.Next.
        /// </para>
        /// <para>
        /// Since the sequence is infinite, a typical usage would be to cap
        /// the results with a function like Enumerable.Take.  For example,
        /// to sum the results of rolling three six sided dice, we could do:
        /// </para>
        /// <code>
        /// random.Integers(1, 7).Take(3).Sum()
        /// </code>
        /// </remarks>
        public static IEnumerable<int> Integers(this Random random, int min, int max)
        {
            while (true)
                yield return random.Next(min, max);
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `83_Stock_Market/java/StockMarket.java`

This is a Java program that simulates the stock market. It includes functions for managing cash, brokers, and transactions.

The program includes several user-defined variables:

* marketTrendSlope: The trend slope of the market
* brokerageFee: The fee charged by the broker
* cashAssets: The current amount of cash on hand
* tmpCashAssets: The temporary amount of cash on hand
* changeStockValue: The current value of a stock
* assets: The list of stocks held in the portfolio
* largeChange1, largeChange2: The large changes that can occur in the portfolio (e.g., market crashes)
* randomStockIndex1, randomStockIndex2: The indices of the stocks in the portfolio, with random effects
* n大型谢绝： The number of days in which a stock can experience a large change
* portfolioContents: The current contents of the portfolio
* dayCnt: The current number of days in the portfolio
* n大型谢绝： The number of days in which a stock can experience a large change
* n长期谢绝： The number of days in which a stock can experience a large change
* totalDaysPurchases: The total number of days in which the portfolio has been fully purchased
* portfolioContinents: The current contents of the portfolio, excluding cash
* changeNyseAverage, nyseAverage, nyseAverageChange: The average change of the NYSE Composite Average
* t百万 Hyde: The current trading volume of the stock
* randomNyseAverage, randomNyseAverageChange: The random changes in the NYSE Composite Average
* transactionQuantity: The current quantity of the stock
* dayCnt: The current number of days in the portfolio
* transactionPrice: The current price of the stock
* transactionAmount: The current quantity of the stock
* biannually: The frequency of the portfolio update (excluding daily update)
* monthly: The frequency of the portfolio update (excluding monthly update)

The program also includes a Simulator class that extends the java.util.Scanner class to allow the user to interact with the program.

The Simulator class has methods for setting the market trend, setting the brokerage fee, setting the number of days in which a stock can experience a large change, setting the number of stocks in the portfolio, and setting the cash amount.

The Simulator class also has methods for updating the portfolio, purchasing and selling stocks, and calculating the average change of the portfolio.


```
import java.util.ArrayList;
import java.util.InputMismatchException;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

/**
 * Stock Market Simulation
 *
 * Some of the original program's variables' documentation and their equivalent in this program:
 * A-MRKT TRND SLP;             marketTrendSlope
 * B5-BRKRGE FEE;               brokerageFee
 * C-TTL CSH ASSTS;             cashAssets
 * C5-TTL CSH ASSTS (TEMP);     tmpCashAssets
 * C(I)-CHNG IN STK VAL;        changeStockValue
 * D-TTL ASSTS;                 assets
 * E1,E2-LRG CHNG MISC;         largeChange1, largeChange2
 * I1,I2-STCKS W LRG CHNG;      randomStockIndex1, randomStockIndex2
 * N1,N2-LRG CHNG DAY CNTS;     largeChangeNumberDays1, largeChangeNumberDays2
 * P5-TTL DAYS PRCHSS;          totalDaysPurchases
 * P(I)-PRTFL CNTNTS;           portfolioContents
 * Q9-NEW CYCL?;                newCycle
 * S4-SGN OF A;                 slopeSign
 * S5-TTL DYS SLS;              totalDaysSales
 * S(I)-VALUE/SHR;              stockValue
 * T-TTL STCK ASSTS;            totalStockAssets
 * T5-TTL VAL OF TRNSCTNS;      totalValueOfTransactions
 * W3-LRG CHNG;                 bigChange
 * X1-SMLL CHNG(<$1);           smallChange
 * Z4,Z5,Z6-NYSE AVE.;          tmpNyseAverage, nyseAverage, nyseAverageChange
 * Z(I)-TRNSCT                  transactionQuantity
 *
 * new price = old price + (trend x old price) + (small random price
 * change) + (possible large price change)
 *
 * Converted from BASIC to Java by Aldrin Misquitta (@aldrinm)
 */
```

This is a Java class that represents a stock. The stock data is stored in the `Stock` class, and includes the following fields:

* `stockValue`: The value of the stock.
* `stockName`: The name of the stock.
* `stockCode`: The code of the stock.
* `portfolioContents`: The amount of the stock that is held in the portfolio.
* `transactionQuantity`: The quantity of the stock that has been traded.
* `changeStockValue`: The change in the stock value from one time period to another.

The `Stock` class has the following getters and setters for these fields:

* `stockValue`
* `stockName`
* `stockCode`
* `portfolioContents`
* `transactionQuantity`
* `changeStockValue`

It also has a `toString()` method that is used for string representation of the stock object.

Note that this class is just a simple example, in real-world scenario, you will have to implement the `StockData` interface which will handle the getters and setters of all the fields, also you will need to create the getters and setters of each field.


```
public class StockMarket {

	private static final Random random = new Random();

	public static void main(String[] args) {

		Scanner scan = new Scanner(System.in);

		printIntro();
		printGameHelp(scan);

		final List<Stock> stocks = initStocks();

		double marketTrendSlope = Math.floor((random.nextFloat() / 10) * 100 + 0.5)/100f;
		double totalValueOfTransactions;
		int largeChangeNumberDays1 = 0;
		int largeChangeNumberDays2 = 0;

		//DAYS FOR FIRST TREND SLOPE (A)
		var t8 = randomNumber(1, 6);

		//RANDOMIZE SIGN OF FIRST TREND SLOPE (A)
		if (random.nextFloat() <= 0.5) {
			marketTrendSlope = -marketTrendSlope;
		}

		// INITIALIZE CASH ASSETS:C
		double cashAssets = 10000;
		boolean largeChange1 = false;
		boolean largeChange2 = false;
		double tmpNyseAverage;
		double nyseAverage = 0;
		boolean inProgress = true;
		var firstRound = true;

		while (inProgress) {

			/* Original documentation:
			RANDOMLY PRODUCE NEW STOCK VALUES BASED ON PREVIOUS DAY'S VALUES
			N1,N2 ARE RANDOM NUMBERS OF DAYS WHICH RESPECTIVELY
			DETERMINE WHEN STOCK I1 WILL INCREASE 10 PTS. AND STOCK
			I2 WILL DECREASE 10 PTS.
			IF N1 DAYS HAVE PASSED, PICK AN I1, SET E1, DETERMINE NEW N1
			*/
			int randomStockIndex1 = 0;
			int randomStockIndex2 = 0;

			if (largeChangeNumberDays1 <= 0) {
				randomStockIndex1 = randomNumber(0, stocks.size());
				largeChangeNumberDays1 = randomNumber(1, 6);
				largeChange1 = true;
			}
			if (largeChangeNumberDays2 <= 0) {
				randomStockIndex2 = randomNumber(0, stocks.size());
				largeChangeNumberDays2 = randomNumber(1, 6);
				largeChange2 = true;
			}
			adjustAllStockValues(stocks, largeChange1, largeChange2, marketTrendSlope, stocks.get(randomStockIndex1), stocks.get(randomStockIndex2));

			//reset largeChange flags
			largeChange1 = false;
			largeChange2 = false;
			largeChangeNumberDays1--;
			largeChangeNumberDays2--;

			//AFTER T8 DAYS RANDOMLY CHANGE TREND SIGN AND SLOPE
			t8 = t8 - 1;
			if (t8 < 1) {
				marketTrendSlope = newMarketTrendSlope();
				t8 = randomNumber(1, 6);
			}

			//PRINT PORTFOLIO
			printPortfolio(firstRound, stocks);

			tmpNyseAverage = nyseAverage;
			nyseAverage = 0;
			double totalStockAssets = 0;
			for (Stock stock : stocks) {
				nyseAverage = nyseAverage + stock.getStockValue();
				totalStockAssets = totalStockAssets + stock.getStockValue() * stock.getPortfolioContents();
			}
			nyseAverage = Math.floor(100 * (nyseAverage / 5) + .5) / 100f;
			double nyseAverageChange = Math.floor((nyseAverage - tmpNyseAverage) * 100 + .5) / 100f;

			// TOTAL ASSETS:D
			double assets = totalStockAssets + cashAssets;
			if (firstRound) {
				System.out.printf("\n\nNEW YORK STOCK EXCHANGE AVERAGE: %.2f", nyseAverage);
			} else {
				System.out.printf("\n\nNEW YORK STOCK EXCHANGE AVERAGE: %.2f NET CHANGE %.2f", nyseAverage, nyseAverageChange);
			}

			totalStockAssets = Math.floor(100 * totalStockAssets + 0.5) / 100d;
			System.out.printf("\n\nTOTAL STOCK ASSETS ARE   $ %.2f", totalStockAssets);
       		cashAssets = Math.floor(100 * cashAssets + 0.5) / 100d;
			System.out.printf("\nTOTAL CASH ASSETS ARE    $ %.2f", cashAssets);
			assets = Math.floor(100 * assets + .5) / 100d;
			System.out.printf("\nTOTAL ASSETS ARE         $ %.2f\n", assets);

			if (!firstRound) {
				System.out.print("\nDO YOU WISH TO CONTINUE (YES-TYPE 1, NO-TYPE 0)? ");
				var newCycle = readANumber(scan);
				if (newCycle < 1) {
					System.out.println("HOPE YOU HAD FUN!!");
					inProgress = false;
				}
			}

			if (inProgress) {
				boolean validTransaction = false;
				//    TOTAL DAY'S PURCHASES IN $:P5
				double totalDaysPurchases = 0;
				//    TOTAL DAY'S SALES IN $:S5
				double totalDaysSales = 0;
				double tmpCashAssets;
				while (!validTransaction) {
					//INPUT TRANSACTIONS
					readStockTransactions(stocks, scan);
					totalDaysPurchases = 0;
					totalDaysSales = 0;

					validTransaction = true;
					for (Stock stock : stocks) {
						stock.setTransactionQuantity(Math.floor(stock.getTransactionQuantity() + 0.5));
						if (stock.getTransactionQuantity() > 0) {
							totalDaysPurchases = totalDaysPurchases + stock.getTransactionQuantity() * stock.getStockValue();
						} else {
							totalDaysSales = totalDaysSales - stock.getTransactionQuantity() * stock.getStockValue();
							if (-stock.getTransactionQuantity() > stock.getPortfolioContents()) {
								System.out.println("YOU HAVE OVERSOLD A STOCK; TRY AGAIN.");
								validTransaction = false;
								break;
							}
						}
					}

					//TOTAL VALUE OF TRANSACTIONS:T5
					totalValueOfTransactions = totalDaysPurchases + totalDaysSales;
					// BROKERAGE FEE:B5
					var brokerageFee = Math.floor(0.01 * totalValueOfTransactions * 100 + .5) / 100d;
					// CASH ASSETS=OLD CASH ASSETS-TOTAL PURCHASES
					//-BROKERAGE FEES+TOTAL SALES:C5
					tmpCashAssets = cashAssets - totalDaysPurchases - brokerageFee + totalDaysSales;
					if (tmpCashAssets < 0) {
						System.out.printf("\nYOU HAVE USED $%.2f MORE THAN YOU HAVE.", -tmpCashAssets);
						validTransaction = false;
					} else {
						cashAssets = tmpCashAssets;
					}
				}

				// CALCULATE NEW PORTFOLIO
				for (Stock stock : stocks) {
					stock.setPortfolioContents(stock.getPortfolioContents() + stock.getTransactionQuantity());
				}

				firstRound = false;
			}

		}
	}

	/**
	 * Random int between lowerBound(inclusive) and upperBound(exclusive)
	 */
	private static int randomNumber(int lowerBound, int upperBound) {
		return random.nextInt((upperBound - lowerBound)) + lowerBound;
	}

	private static double newMarketTrendSlope() {
		return randomlyChangeTrendSignAndSlopeAndDuration();
	}

	private static void printPortfolio(boolean firstRound, List<Stock> stocks) {
		//BELL RINGING-DIFFERENT ON MANY COMPUTERS
		if (firstRound) {
			System.out.printf("%n%-30s\t%12s\t%12s", "STOCK", "INITIALS", "PRICE/SHARE");
			for (Stock stock : stocks) {
				System.out.printf("%n%-30s\t%12s\t%12.2f ------ %12.2f", stock.getStockName(), stock.getStockCode(),
						stock.getStockValue(), stock.getChangeStockValue());
			}
			System.out.println("");
		} else {
			System.out.println("\n**********     END OF DAY'S TRADING     **********\n\n");
			System.out.printf("%n%-12s\t%-12s\t%-12s\t%-12s\t%-20s", "STOCK", "PRICE/SHARE",
					"HOLDINGS", "VALUE", "NET PRICE CHANGE");
			for (Stock stock : stocks) {
				System.out.printf("%n%-12s\t%-12.2f\t%-12.0f\t%-12.2f\t%-20.2f",
						stock.getStockCode(), stock.getStockValue(), stock.getPortfolioContents(),
						stock.getStockValue() * stock.getPortfolioContents(), stock.getChangeStockValue());
			}
		}
	}

	private static void readStockTransactions(List<Stock> stocks, Scanner scan) {
		System.out.println("\n\nWHAT IS YOUR TRANSACTION IN");
		for (Stock stock : stocks) {
			System.out.printf("%s? ", stock.getStockCode());

			stock.setTransactionQuantity(readANumber(scan));
		}
	}

	private static int readANumber(Scanner scan) {
		int choice = 0;

		boolean validInput = false;
		while (!validInput) {
			try {
				choice = scan.nextInt();
				validInput = true;
			} catch (InputMismatchException ex) {
				System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE");
			} finally {
				scan.nextLine();
			}
		}

		return choice;
	}

	private static void adjustAllStockValues(List<Stock> stocks, boolean largeChange1,
			boolean largeChange2,
			double marketTrendSlope,
			Stock stockForLargeChange1, Stock stockForLargeChange2
	) {
		//LOOP THROUGH ALL STOCKS
		for (Stock stock : stocks) {
			double smallChange = random.nextFloat();

			if (smallChange <= 0.25) {
				smallChange = 0.25;
			} else if (smallChange <= 0.5) {
				smallChange = 0.5;
			} else if (smallChange <= 0.75) {
				smallChange = 0.75;
			} else {
				smallChange = 0;
			}

			//BIG CHANGE CONSTANT:W3  (SET TO ZERO INITIALLY)
			var bigChange = 0;
			if (largeChange1) {
				if (stock.getStockCode().equals(stockForLargeChange1.getStockCode())) {
					//ADD 10 PTS. TO THIS STOCK;  RESET E1
					bigChange = 10;
				}
			}

			if (largeChange2) {
				if (stock.getStockCode().equals(stockForLargeChange2.getStockCode())) {
					//SUBTRACT 10 PTS. FROM THIS STOCK;  RESET E2
					bigChange = bigChange - 10;
				}
			}

			stock.setChangeStockValue(Math.floor(marketTrendSlope * stock.stockValue) + smallChange +
					Math.floor(3 - 6 * random.nextFloat() + .5) + bigChange);
			stock.setChangeStockValue(Math.floor(100 * stock.getChangeStockValue() + .5) / 100d);
			stock.stockValue += stock.getChangeStockValue();

			if (stock.stockValue > 0) {
				stock.stockValue = Math.floor(100 * stock.stockValue + 0.5) / 100d;
			} else {
				stock.setChangeStockValue(0);
				stock.stockValue = 0;
			}
		}
	}

	private static double randomlyChangeTrendSignAndSlopeAndDuration() {
		// RANDOMLY CHANGE TREND SIGN AND SLOPE (A), AND DURATION
		var newTrend = Math.floor((random.nextFloat() / 10) * 100 + .5) / 100d;
		var slopeSign = random.nextFloat();
		if (slopeSign > 0.5) {
			newTrend = -newTrend;
		}
		return newTrend;
	}

	private static List<Stock> initStocks() {
		List<Stock> stocks = new ArrayList<>();
		stocks.add(new Stock(100, "INT. BALLISTIC MISSILES", "IBM"));
		stocks.add(new Stock(85, "RED CROSS OF AMERICA", "RCA"));
		stocks.add(new Stock(150, "LICHTENSTEIN, BUMRAP & JOKE", "LBJ"));
		stocks.add(new Stock(140, "AMERICAN BANKRUPT CO.", "ABC"));
		stocks.add(new Stock(110, "CENSURED BOOKS STORE", "CBS"));
		return stocks;
	}

	private static void printGameHelp(Scanner scan) {
		System.out.print("DO YOU WANT THE INSTRUCTIONS (YES-TYPE 1, NO-TYPE 0) ? ");
		int choice = scan.nextInt();
		if (choice >= 1) {
			System.out.println("");
			System.out.println("THIS PROGRAM PLAYS THE STOCK MARKET.  YOU WILL BE GIVEN");
			System.out.println("$10,000 AND MAY BUY OR SELL STOCKS.  THE STOCK PRICES WILL");
			System.out.println("BE GENERATED RANDOMLY AND THEREFORE THIS MODEL DOES NOT");
			System.out.println("REPRESENT EXACTLY WHAT HAPPENS ON THE EXCHANGE.  A TABLE");
			System.out.println("OF AVAILABLE STOCKS, THEIR PRICES, AND THE NUMBER OF SHARES");
			System.out.println("IN YOUR PORTFOLIO WILL BE PRINTED.  FOLLOWING THIS, THE");
			System.out.println("INITIALS OF EACH STOCK WILL BE PRINTED WITH A QUESTION");
			System.out.println("MARK.  HERE YOU INDICATE A TRANSACTION.  TO BUY A STOCK");
			System.out.println("TYPE +NNN, TO SELL A STOCK TYPE -NNN, WHERE NNN IS THE");
			System.out.println("NUMBER OF SHARES.  A BROKERAGE FEE OF 1% WILL BE CHARGED");
			System.out.println("ON ALL TRANSACTIONS.  NOTE THAT IF A STOCK'S VALUE DROPS");
			System.out.println("TO ZERO IT MAY REBOUND TO A POSITIVE VALUE AGAIN.  YOU");
			System.out.println("HAVE $10,000 TO INVEST.  USE INTEGERS FOR ALL YOUR INPUTS.");
			System.out.println("(NOTE:  TO GET A 'FEEL' FOR THE MARKET RUN FOR AT LEAST");
			System.out.println("10 DAYS)");
			System.out.println("-----GOOD LUCK!-----");
		}
		System.out.println("\n\n");
	}

	private static void printIntro() {
		System.out.println("                                STOCK MARKET");
		System.out.println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
		System.out.println("\n\n");
	}

	/**
	 * Stock class also storing the stock information and other related information for simplicity
	 */
	private static class Stock {

		private final String stockName;
		private final String stockCode;
		private double stockValue;
		private double portfolioContents = 0;
		private double transactionQuantity = 0;
		private double changeStockValue = 0;

		public Stock(double stockValue, String stockName, String stockCode) {
			this.stockValue = stockValue;
			this.stockName = stockName;
			this.stockCode = stockCode;
		}

		public String getStockName() {
			return stockName;
		}

		public String getStockCode() {
			return stockCode;
		}

		public double getStockValue() {
			return stockValue;
		}

		public double getPortfolioContents() {
			return portfolioContents;
		}

		public void setPortfolioContents(double portfolioContents) {
			this.portfolioContents = portfolioContents;
		}

		public double getTransactionQuantity() {
			return transactionQuantity;
		}

		public void setTransactionQuantity(double transactionQuantity) {
			this.transactionQuantity = transactionQuantity;
		}

		public double getChangeStockValue() {
			return changeStockValue;
		}

		public void setChangeStockValue(double changeStockValue) {
			this.changeStockValue = changeStockValue;
		}

		@Override
		public String toString() {
			return "Stock{" +
					"stockValue=" + stockValue +
					", stockCode='" + stockCode + '\'' +
					", portfolioContents=" + portfolioContents +
					", transactionQuantity=" + transactionQuantity +
					", changeStockValue=" + changeStockValue +
					'}';
		}
	}

}

```