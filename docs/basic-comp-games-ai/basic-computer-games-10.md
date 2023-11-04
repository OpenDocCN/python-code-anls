# BasicComputerGames源码解析 10

# `01_Acey_Ducey/java/src/AceyDuceyGame.java`

这段代码定义了一个名为AceyDuceyGame的类，用于调用游戏。在main方法中，定义了一个布尔变量keepPlaying，用于判断是否继续玩游戏。然后，创建了一个AceyDucey类的实例游戏，并进入一个无限循环，每次循环都会调用游戏的play()方法来继续玩游戏。循环内部会不断输出结果，并在每次输出后继续循环。 keepPlaying变量会在循环开始时设置为true，以便在每次循环时调用游戏的playAgain()方法来使游戏继续。


```
/**
 * This class is used to invoke the game.
 *
 */
public class AceyDuceyGame {

    public static void main(String[] args) {

        boolean keepPlaying;
        AceyDucey game = new AceyDucey();

        // Keep playing game until infinity or the player loses
        do {
            game.play();
            System.out.println();
            System.out.println();
            System.out.println();
            keepPlaying = game.playAgain();
        } while (keepPlaying);
    }
}

```

# `01_Acey_Ducey/java/src/Card.java`

这段代码定义了一个名为 "Card" 的类，代表一张从一副牌中抽出的扑克牌。

在 "Card" 类的构造函数中，首先调用了一个名为 "init" 的内部方法，该方法接受一个整数参数 "value"，用于初始化扑克牌的值。

如果 "value" 的值在 2 到 9 之间，那么构造函数会使用 "String.valueOf" 方法来将 "value" 转换为字符串，并将其设置为扑克牌的名称。

如果 "value" 的值大于 9，那么构造函数会使用一个 switch 语句来查找对应于 "value" 值的名称，并将其设置为扑克牌的名称。

如果 "value" 的值无法用上述方法初始化，那么构造函数会执行一个默认操作，将其设置为 "Unknown"。

在 "Card" 类的 "getValue" 方法中，返回扑克牌的值。

在 "Card" 类的 "getName" 方法中，返回扑克牌的名称，即它的名称字符串表示。


```
/**
 * A card from a deck - the value is between 2-14 to cover
 * cards with a face value of 2-9 and then a Jack, Queen, King, and Ace
 */
public class Card {
    private int value;
    private String name;

    Card(int value) {
        init(value);
    }

    private void init(int value) {
        this.value = value;
        if (value < 11) {
            this.name = String.valueOf(value);
        } else {
            switch (value) {
                case 11:
                    this.name = "Jack";
                    break;
                case 12:
                    this.name = "Queen";
                    break;
                case 13:
                    this.name = "King";
                    break;
                case 14:
                    this.name = "Ace";
                    break;

                default:
                    this.name = "Unknown";
            }
        }
    }

    public int getValue() {
        return value;
    }

    public String getName() {
        return name;
    }
}

```

# `01_Acey_Ducey/javascript/aceyducey.js`

这段代码定义了一些工具变量。

首先，它检查了浏览器是否具有一个窗口对象。如果没有，它将输出一条消息，表明浏览器和 Node.js 支持。

然后，它定义了两个工具函数，`validLowerCaseYesStrings` 和 `validLowerCaseNoStrings`。这些函数用于检查输入字符串是否以 "yes" 或 "no" 开头。

最后，它定义了一个包含两个或更多个 "yes" 或 "no" 的字符串，称为 `validLowerCaseYesAndNoStrings`。


```
// UTILITY VARIABLES

// By default:
// — Browsers have a window object
// — Node.js does not
// Checking for an undefined window object is a loose check
// to enable browser and Node.js support
const isRunningInBrowser = typeof window !== 'undefined';

// To easily validate input strings with utility functions
const validLowerCaseYesStrings = ['yes', 'y'];
const validLowerCaseNoStrings = ['no', 'n'];
const validLowerCaseYesAndNoStrings = [
    ...validLowerCaseYesStrings,
    ...validLowerCaseNoStrings,
];
```

这段代码定义了两个函数，一个是 `getRandomCard()`，另一个是 `newGameCards()`。

`getRandomCard()` 函数用于生成一个 2-14 之间的随机整数（包括 2 和 14，但不包括 ACE，因为 ACE 的值被视为 14）。函数使用了 `Math.random()` 函数生成一个浮点数范围内的随机整数，然后将其转换为整数并返回。

`newGameCards()` 函数返回一副游戏牌中的三张牌。函数首先使用 `getRandomCard()` 函数生成三张随机牌，然后使用一系列条件判断来确保生成的牌符合游戏规则。具体来说，函数需要确保生成的牌中有两张不同的牌，且其中一张牌的点数要小于另一张牌的点数。

因此，这段代码的作用是生成一副游戏牌中的三张牌，其中两张牌的点数不同，且其中一张牌的点数要小于另一张牌的点数。


```
// UTILITY VARIABLES

// Function to get a random number (card) 2-14 (ACE is 14)
function getRandomCard() {
    // In our game, the value of ACE is greater than face cards;
    // instead of having the value of ACE be 1, we’ll have it be 14.
    // So, we want to shift the range of random numbers from 1-13 to 2-14
    let min = 2;
    let max = 14;
    // Return random integer between two values, inclusive
    return Math.floor(Math.random() * (max - min + 1) + min);
}

function newGameCards() {
    let cardOne = getRandomCard();
    let cardTwo = getRandomCard();
    let cardThree = getRandomCard();
    // We want:
    // 1. cardOne and cardTwo to be different cards
    // 2. cardOne to be lower than cardTwo
    // So, while cardOne is greater than or equal too cardTwo
    // we will continue to generate random cards.
    while (cardOne >= cardTwo) {
        cardOne = getRandomCard();
        cardTwo = getRandomCard();
    }
    return [cardOne, cardTwo, cardThree];
}

```

这段代码定义了一个名为 `getCardValue` 的函数，它接受一个名为 `card` 的参数。该函数的作用是获取一张扑克牌的点数（即牌面）并返回。

函数内部，首先定义了一个名为 `faceOrAce` 的对象，包含了一些扑克牌的点数名称。接着，代码使用 `let` 关键字定义了一个名为 `temp` 的变量，用于存储 `card` 参数所对应的扑克牌的点数。

接着，代码使用 `??` 运算符对 `temp` 和 `card` 进行比较。如果 `temp` 中的任意一个值与 `card` 相等，那么函数将返回 `temp` 中的值。否则，函数将返回一个默认值，使用 Nullish Coalescing Operator（??）来处理 `temp` 为 `null` 的情况。最后，函数返回 `temp` 和 `card` 两者中的第一个值。

整段代码的作用是提供一个函数，用于获取一张扑克牌的点数（牌面），并返回该点的数值。


```
// Function to get card value
function getCardValue(card) {
    let faceOrAce = {
        11: 'JACK',
        12: 'QUEEN',
        13: 'KING',
        14: 'ACE',
    };
    // If card value matches a key in faceOrAce, use faceOrAce value;
    // Else, return undefined and handle with the Nullish Coalescing Operator (??)
    // and default to card value.
    let cardValue = faceOrAce[card] ?? card;
    return cardValue;
}

```

It looks like the code is trying to create a game where the user is prompted to make a bet, and then the computer will draw a card and determine if the user won or lost.

There are a few things that could be improvements:

* It is not clear what the `getCardValue` function does, and it is not being used in the code. It would be helpful to know what the function is intended to do.
* The code only draws a single card and does not provide any information about the cards that are available. It would be interesting to see how the game could be expanded to include more cards and potentially different types of bets.
* The code uses the `async` and `await` keywords, but it does not provide any explanations or examples of how these keywords can be used. It would be helpful to see how these keywords are intended to be used in the game.

I hope this helps! Let me know if you have any questions.


```
print(spaces(26) + 'ACEY DUCEY CARD GAME');
print(spaces(15) + 'CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n');
print('ACEY-DUCEY IS PLAYED IN THE FOLLOWING MANNER');
print('THE DEALER (COMPUTER) DEALS TWO CARDS FACE UP');
print('YOU HAVE AN OPTION TO BET OR NOT BET DEPENDING');
print('ON WHETHER OR NOT YOU FEEL THE CARD WILL HAVE');
print('A VALUE BETWEEN THE FIRST TWO.');
print("IF YOU DO NOT WANT TO BET, INPUT '0'");

main();

async function main() {
    let bet;
    let availableDollars = 100;

    // Loop game forever
    while (true) {
        let [cardOne, cardTwo, cardThree] = newGameCards();

        print(`YOU NOW HAVE ${availableDollars} DOLLARS.\n`);

        print('HERE ARE YOUR NEXT TWO CARDS: ');
        print(getCardValue(cardOne));
        print(getCardValue(cardTwo));
        print('');

        // Loop until receiving a valid bet
        let validBet = false;
        while (!validBet) {
            print('\nWHAT IS YOUR BET? ');
            bet = parseInt(await input(), 10);
            let minimumRequiredBet = 0;
            if (bet >= minimumRequiredBet) {
                if (bet > availableDollars) {
                    print('SORRY, MY FRIEND, BUT YOU BET TOO MUCH.');
                    print(`YOU HAVE ONLY ${availableDollars} DOLLARS TO BET.`);
                } else {
                    validBet = true;
                }
            }
        }
        if (bet == 0)
        {
            // User chose not to bet.
            print('CHICKEN!!');
            print('');
            // Don't draw a third card, draw a new set of 2 cards.
            continue;
        }

        print('\n\nHERE IS THE CARD WE DREW: ');
        print(getCardValue(cardThree));

        // Determine if player won or lost
        if (cardThree > cardOne && cardThree < cardTwo) {
            print('YOU WIN!!!');
            availableDollars = availableDollars + bet;
        } else {
            print('SORRY, YOU LOSE');

            if (bet >= availableDollars) {
                print('');
                print('');
                print('SORRY, FRIEND, BUT YOU BLEW YOUR WAD.');
                print('');
                print('');
                print('TRY AGAIN (YES OR NO)');

                let tryAgainInput = await input();

                print('');
                print('');

                if (isValidYesString(tryAgainInput)) {
                    availableDollars = 100;
                } else {
                    print('O.K., HOPE YOU HAD FUN!');
                    break;
                }
            } else {
                availableDollars = availableDollars - bet;
            }
        }
    }
}

```

这段代码定义了三个函数 `isValidYesNoString`, `isValidYesString`, 和 `isValidNoString`，它们都使用了变量 `string` 并返回一个布尔值。

第一个函数 `isValidYesNoString` 接收一个字符串参数，并返回一个布尔值，表示它是否是一个有效的 "是" 或 "否" 字符串。这个函数的作用是检查给定的字符串是否符合 "是" 或 "否" 字符串的规则，如果符合，则返回 `true`，否则返回 `false`。

第二个函数 `isValidYesString` 接收一个字符串参数，并返回一个布尔值，表示它是否是一个有效的 "是" 字符串。这个函数的作用是检查给定的字符串是否符合 "是" 字符串的规则，如果符合，则返回 `true`，否则返回 `false`。

第三个函数 `isValidNoString` 接收一个字符串参数，并返回一个布尔值，表示它是否是一个有效的 "否" 字符串。这个函数的作用是检查给定的字符串是否符合 "否" 字符串的规则，如果符合，则返回 `true`，否则返回 `false`。

第四个函数 `print` 接收一个字符串参数，并将其输出到 HTML 页面上。它使用了 `document.getElementById('output')` 来获取一个元素，并将其内容设置为给定的字符串。如果当前运行环境是在浏览器中，它会将字符串添加一个尾部的换行符。


```
// UTILITY FUNCTIONS
function isValidYesNoString(string) {
    return validLowerCaseYesAndNoStrings.includes(string.toLowerCase());
}

function isValidYesString(string) {
    return validLowerCaseYesStrings.includes(string.toLowerCase());
}

function isValidNoString(string) {
    return validLowerCaseNoStrings.includes(string.toLowerCase());
}

function print(string) {
    if (isRunningInBrowser) {
        // Adds trailing newline to match console.log behavior
        document
            .getElementById('output')
            .appendChild(document.createTextNode(string + '\n'));
    } else {
        console.log(string);
    }
}

```



该函数 `input()` 是一个非阻塞函数，用于获取用户输入。函数接受两个参数：`isRunningInBrowser` 和 `outputElement`。

如果 `isRunningInBrowser` 为 `true`，则函数接受来自浏览器 DOM 元素的输入。函数会创建一个 Promise 对象，并在其中设置一个 `resolve` 函数，用于在获取到用户输入后打印结果并返回结果。然后，函数会创建一个 `inputElement` 元素，并将其添加到 `outputElement` 中。接着，函数会添加一个 `keydown` 事件到 `inputElement`，并在事件处理程序中编写一个简短的逻辑，以便在用户按下 "Enter" 键时捕获输入并打印结果。最后，函数会使用 `resolve` 函数来返回捕获到的用户输入。

如果 `isRunningInBrowser` 为 `false`，则函数接受来自命令行工具的输入。函数会使用 Node.js 中的 `readline` 模块来读取命令行中的输入。函数会将 `readline` 中的 `input` 和 `output` 选项设置为 `process.stdin` 和 `process.stdout`，并将 `readline` 实例化并运行在 `q()` 函数中。当 `q()` 函数返回用户输入时，函数会将输入存储在 `input` 变量中，然后使用 `resolve` 函数来返回输入。最后，函数会使用 `close` 函数来关闭 `readline` 实例。


```
function input() {
    if (isRunningInBrowser) {
        // Accept input from the browser DOM input
        return new Promise((resolve) => {
            const outputElement = document.querySelector('#output');
            const inputElement = document.createElement('input');
            outputElement.append(inputElement);
            inputElement.focus();

            inputElement.addEventListener('keydown', (event) => {
                if (event.key === 'Enter') {
                    const result = inputElement.value;
                    inputElement.remove();
                    print(result);
                    print('');
                    resolve(result);
                }
            });
        });
    } else {
        // Accept input from the command line in Node.js
        // See: https://nodejs.dev/learn/accept-input-from-the-command-line-in-nodejs
        return new Promise(function (resolve) {
            const readline = require('readline').createInterface({
                input: process.stdin,
                output: process.stdout,
            });
            readline.question('', function (input) {
                resolve(input);
                readline.close();
            });
        });
    }
}

```

这两函数脚本是一个流水线上的函数，其主要目的是为了解决前端与后端数据同步的问题。

第一个函数 `printInline` 的作用是打印字符串 `string`，并将它显示在页面上。具体来说，它通过 `/src` 路径获取了 `string` 并将其显示在 `#output` 元素中。同时，在 `isRunningInBrowser` 条件为 `true` 时，它还将在页面上添加 `string` 节点。

第二个函数 `spaces` 的作用是在控制台输出指定 `numberOfSpaces` 数量的 ` spaces`。它通过创建一个空字符数组并使用 `repeat` 方法来创建 `numberOfSpaces` 数量的 ` spaces` 字符串，然后将其输出到控制台。

这两个函数脚本是在 Node.js 的 Node.js 环境中运行的，而不是在浏览器环境中。因此，它们不包括在浏览器扩展程序的行为或用户交互中。


```
function printInline(string) {
    if (isRunningInBrowser) {
        document
            .getElementById('output')
            .appendChild(document.createTextNode(string));
    } else {
        process.stdout.write(string);
    }
}

function spaces(numberOfSpaces) {
    return ' '.repeat(numberOfSpaces);
}

// UTILITY FUNCTIONS

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


As published in Basic Computer Games (1978), as found at Annarchive:
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=17)


Conversion to Lua
- [Lua.org](https://www.lua.org)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


# `01_Acey_Ducey/python/acey_ducey.py`

这段代码是一个Python脚本，它实现了Acey-Ducey游戏。在这个游戏中，玩家需要根据自己选择的牌编号，在规定的时间内选择一张对应的牌，然后判断选择的牌是否为王。如果选择的牌是Jack，则游戏结束，输出当前牌编号，否则继续进行下一轮。


```
#!/usr/bin/env python3
"""
Play the Acey-Ducey game
https://www.atariarchives.org/basicgames/showpage.php?page=2
"""

import random


cards = {
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "10",
    11: "Jack",
    12: "Queen",
    13: "King",
    14: "Ace",
}


```

这段代码是一个简单的游戏，让用户在两个限制下进行抽牌。在游戏开始时，玩家已经拥有100美元的现金。然后，程序会持续显示玩家手上的现金，并抽出两张牌。接着，程序会要求玩家下注，如果注注大于0，就会从玩家的现金中扣除这个注数。如果这个注注小于0，或者玩家无法输入一个 positive 数字，程序会提示玩家。如果注注大于两张牌之间的最大值，程序会告诉玩家很遗憾他们输了。如果玩家成功赢得了注注，程序就会从玩家的现金中扣除这个注数的2倍，并告诉玩家他们赢了。否则，程序会告诉玩家他们输了。最后，如果玩家在游戏结束时仍然拥有100美元的现金，程序会告诉玩家很遗憾他们错过了这个机会。


```
def play_game() -> None:
    cash = 100
    while cash > 0:
        print(f"You now have {cash} dollars\n")
        print("Here are you next two cards")
        round_cards = list(cards.keys())  # gather cards from dictionary
        card_a = random.choice(round_cards)  # choose a card
        card_b = card_a  # clone the first card, so we avoid the same number for the second card
        while (card_a == card_b):  # if the cards are the same, choose another card
            card_b = random.choice(round_cards)
        card_c = random.choice(round_cards)  # choose last card
        if card_a > card_b:  # swap cards if card_a is greater than card_b
            card_a, card_b = card_b, card_a
        print(f" {cards[card_a]}")
        print(f" {cards[card_b]}\n")
        while True:
            try:
                bet = int(input("What is your bet? "))
                if bet < 0:
                    raise ValueError("Bet must be more than zero")
                if bet == 0:
                    print("CHICKEN!!\n")
                if bet > cash:
                    print("Sorry, my friend but you bet too much")
                    print(f"You only have {cash} dollars to bet")
                    continue
                cash -= bet
                break

            except ValueError:
                print("Please enter a positive number")
        print(f" {cards[card_c]}")
        if bet > 0:
            if card_a <= card_c <= card_b:
                print("You win!!!")
                cash += bet * 2
            else:
                print("Sorry, you lose")

    print("Sorry, friend, but you blew your wad")


```

这段代码的主要作用是让用户参与一个Acey-Ducey游戏。在这个游戏中，计算机先随机发两张牌，然后提示玩家可以选择是否继续游戏。如果玩家选择继续，计算机将随机发出一张牌，并提示玩家再次输入“y”或“n”，以决定是否继续游戏。如果玩家选择不继续游戏，游戏将结束，并输出“Ok hope you had fun”的消息。

在游戏过程中，如果玩家选择继续游戏，计算机将随机发出两张牌并提示玩家可以选择是否要下注。如果玩家决定下注，计算机将提示玩家输入他们的下注金额。如果玩家再次确定继续游戏，计算机将随机发出一张牌，并提示玩家再次输入“y”或“n”，以决定是否继续游戏。

在整个游戏中，计算机将重复发送两张牌，并随机发出一张牌作为第一轮下注。然后，玩家可以选择是否继续游戏或退出游戏。如果玩家选择退出游戏，游戏将结束并输出“Ok hope you had fun”的消息。


```
def main() -> None:
    print(
        """
Acey-Ducey is played in the following manner
The dealer (computer) deals two cards face up
You have an option to bet or not bet depending
on whether or not you feel the card will have
a value between the first two.
If you do not want to bet, input a 0
  """
    )
    keep_playing = True

    while keep_playing:
        play_game()
        keep_playing = input("Try again? (yes or no) ").lower().startswith("y")
    print("Ok hope you had fun")


```

这段代码是一个Python程序的if语句，其作用是当程序作为主函数(__main__)运行时执行。

具体来说，程序首先调用Python标准库中的random模块的seed()函数，这个函数用于随机生成一个随机数作为程序的种子(seed)，有助于让程序运行出更加随机的结果。

然后程序进入if语句的判断部分，判断当前程序是否作为主函数运行。如果当前程序作为主函数运行，那么程序将跳转到if语句的局部代码块，也就是执行if语句中的代码。

在这个if语句的局部代码块中，程序再次调用random模块的seed()函数，但是这次使用的是当前进程的随机数种子(process seed)，而不是程序的种子。这样做可以保证每次运行程序时生成的随机数都不同，从而保证程序运行结果的随机性。

最后，程序再次调用了一个名为main()的函数，这个函数可能是定义在程序外的函数，用于执行程序的其他部分。不过，在这个if语句的局部代码块中，程序并没有定义main()函数，因此这个函数的实际作用就等同于定义在了程序内部，但是程序还是可以正常运行。


```
if __name__ == "__main__":
    random.seed()
    main()

```

# `01_Acey_Ducey/python/acey_ducey_oo.py`

这段代码是一个用于生成随机游戏排名的 Python 程序。它使用了 `typing` 包中的 `List`、`Literal` 和 `NamedTuple` 类型来定义游戏中玩家的不同身份（如 "Suit" 代表 " clubs", "Rank" 代表 " rank"）。

具体来说，这段代码实现了一个名为 `generate_rankings` 的函数，它接受一个包含所有玩家身份列表的参数。函数首先从列表中随机选择一个身份，然后根据用户提供的参数，对选择的身份进行排序，最后返回排好序的身份列表。

函数的实现还可以看作是一个名为 `random_rankings` 的装饰函数，它接受一个列表作为参数，并返回一个排好序的身份列表。这个装饰函数使用了 `get_args` 函数来获取用户提供的参数，然后将其传递给 `random.sample` 函数，用于随机选择身份。


```
"""
AceyDuchy
From: BASIC Computer Games (1978)
      Edited by David Ahl
"The original BASIC program author was Bill Palmby
 of Prairie View, Illinois."
Python port by Aviyam Fischer, 2022
"""

from typing import List, Literal, NamedTuple, TypeAlias, get_args
import random

Suit: TypeAlias = Literal["\u2665", "\u2666", "\u2663", "\u2660"]
Rank: TypeAlias = Literal[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]


```

这段代码定义了一个名为 "Card" 的类，该类继承自 "NamedTuple" 类。这个类包含两个属性，一个是 "suit"，表示牌的性别，另一个是 "rank"，表示牌的排名。

在 "Card" 的 "__str__" 方法中，定义了一个字符串格式化操作，用于将 "rank" 属性转换为字符串并返回。

另外，定义了一个名为 "Deck" 的类，该类继承自 "List" 类。这个类包含一个 "cards" 列表，用于存储牌的数据，并且在 "__init__" 方法中调用 "build" 方法来将牌存储到 "cards" 列表中。

在 "build" 方法中，使用了 "get\_args" 函数获取所有可能的牌的排名，并循环遍历每种排名，将牌对象存储到 "cards" 列表中。

在 "shuffle" 方法中，使用 Python 的 "random.shuffle" 函数对 "cards" 列表进行随机重排。

在 "deal" 方法中，使用 "cards.pop" 方法从 "cards" 列表中取出牌，并将其返回。


```
class Card(NamedTuple):
    suit: Suit
    rank: Rank

    def __str__(self) -> str:
        r = str(self.rank)
        r = {"11": "J", "12": "Q", "13": "K", "14": "A"}.get(r, r)
        return f"{r}{self.suit}"


class Deck:
    def __init__(self) -> None:
        self.cards: List[Card] = []
        self.build()

    def build(self) -> None:
        for suit in get_args(Suit):
            for rank in get_args(Rank):
                self.cards.append(Card(suit, rank))

    def shuffle(self) -> None:
        random.shuffle(self.cards)

    def deal(self) -> Card:
        return self.cards.pop()


```

This is a Python implementation of a simple game where two players take turns drawing cards and trying to win the game by either having a higher rank card or using the "Bet" option to increase their chances of winning.

The game has a initial money value of 100 and a flag indicating whether the game is still going.

In each iteration of the game, the player selects one card each and compares it to the cards of the other player. If the card of the player with the highest rank wins the game, the player's money is increased by the bet amount. If the game is still going when all the players have played all their cards, a "Chicken" game is played, and the game is over if the player runs out of cards or if the player's money is reduced to zero.

The game also has a feature where the player can use the "Bet" option to increase their chances of winning. The player must then draw one more card and compare it to the remaining cards of the other player. If the player's card has a higher rank than the other player's card, they win the game, and the money is increased by the bet amount. If the player's card has a lower rank, they lose the game, and the money is reduced by the bet amount.


```
class Game:
    def __init__(self) -> None:
        self.deck = Deck()
        self.deck.shuffle()
        self.card_a = self.deck.deal()
        self.card_b = self.deck.deal()
        self.money = 100
        self.not_done = True

    def play(self) -> None:
        while self.not_done:
            while self.money > 0:
                card_a = self.card_a
                card_b = self.card_b

                if card_a.rank > card_b.rank:
                    card_a, card_b = card_b, card_a

                if card_a.rank == card_b.rank:
                    self.card_b = self.deck.deal()
                    card_b = self.card_b

                print(f"You have:\t ${self.money} ")
                print(f"Your cards:\t {card_a} {card_b}")

                bet = int(input("What is your bet? "))
                player_card = self.deck.deal()
                if 0 < bet <= self.money:

                    print(f"Your deal:\t {player_card}")
                    if card_a.rank <= player_card.rank <= card_b.rank:
                        print("You Win!")
                        self.money += bet
                    else:
                        print("You Lose!")
                        self.money -= bet
                        self.not_done = False
                else:
                    print("Chicken!")
                    print(f"Your deal should have been: {player_card}")
                    if card_a.rank < player_card.rank < card_b.rank:
                        print("You could have won!")
                    else:
                        print("You would lose, so it was wise of you to chicken out!")

                if len(self.deck.cards) <= 3:
                    print("You ran out of cards. Game over.")
                    self.not_done = False
                    break

                self.card_a = self.deck.deal()
                self.card_b = self.deck.deal()

        if self.money == 0:
            self.not_done = False


```

这段代码是一个游戏循环函数，它会在一个无限循环中重复执行游戏操作。具体来说，它实现了以下功能：

1. 初始化游戏变量 game，并让 game.play()，这意味着 game 变量会调用 game 类中的 play() 方法来执行游戏操作。
2. 在循环中，它会询问玩家是否想要继续游戏，如果玩家输入“y”，那么游戏循环将会终止，因为玩家已经决定继续游戏了。
3. 循环还会继续执行 game.play() 方法，这意味着游戏将再次运行，并且玩家将再次看到他们的金钱数量以及是否可以选择继续游戏或结束游戏。
4. 最后，如果玩家在循环中没有做出任何选择，游戏循环将会终止，并且游戏将结束，显示“Thanks for playing!” 消息。

总之，这段代码是一个让玩家玩游戏的游戏循环，玩家可以选择继续游戏或结束游戏。


```
def game_loop() -> None:
    game_over = False

    while not game_over:
        game = Game()
        game.play()
        print(f"You have ${game.money} left")
        print("Would you like to play again? (y/n)")
        if input() == "n":
            game_over = True


def main() -> None:
    print(
        """
    Acey Ducey is a card game where you play against the computer.
    The Dealer(computer) will deal two cards facing up.
    You have an option to bet or not bet depending on whether or not you
    feel the card will have a value between the first two.
    If you do not want to bet input a 0
    """
    )
    game_loop()
    print("\nThanks for playing!")


```

这段代码是一个if语句，它会判断当前脚本是否作为主程序运行。如果是，那么代码块内的语句将被执行。这里使用了Python中的random模块，通过设置random.seed()函数的种子号来让Python随机数生成器的行为与特定的程序运行时一致。通过这样的方式，这段代码会确保每次运行程序时生成的随机数都不同，从而可以进行各种可靠的测试和数据分析等应用。


```
if __name__ == "__main__":
    random.seed()
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)

Propose using pylint and black to format python files so that it conforms to some standards


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/) by Christopher Özbek [coezbek@github](https://github.com/coezbek).


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by Alex Kotov [mur4ik18@github](https://github.com/mur4ik18).

Further edits by

- Berker Şal [berkersal@github](https://github.com/berkersal)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Amazing

This program will print out a different maze every time it is run and guarantees only one path through. You can choose the dimensions of the maze — i.e. the number of squares wide and long.

The original program author was Jack Hauber of Windsor, Connecticut.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=3)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=18)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Known Bugs

- The input dimensions are checked for values of 1, but not for values of 0 or less.  Such inputs will cause the program to break.

#### Porting Notes

**2022-01-04:** patched original source in [#400](https://github.com/coding-horror/basic-computer-games/pull/400) to fix a minor bug where a generated maze may be missing an exit, particularly at small maze sizes.


# `02_Amazing/csharp/Amazing.cs`

This code looks like it is written in C# and it defines a class called `AmazingGame` that has a `Play` method.

The `AmazingGame` class defines several methods that are used to manipulate cells in a 2D array called `Cells`. These methods allow you to set the count of cells in a given row, column, or range to 0.

The `IsPrevRowSet`, `IsNextColSet`, `IsNextRowSet`, and `GetPrevCol`, `GetPrevRow`, `GetNextCol`, and `GetNextRow` methods are used to check if a row or column has a specified setting and return a reference to the cell before the current row or column.

The `GetFirstUnset` method is used to return the first cell in a row or column that has a count of 0.

The `Play` method is used to start the game.

Overall, this code appears to be a game class that could be used to simulate different types of games.


```
﻿using System;
using System.Collections.Generic;

namespace Amazing
{
    class AmazingGame
    {
        private const int FIRST_COL = 0;
        private const int FIRST_ROW = 0;
        private const int EXIT_UNSET = 0;
        private const int EXIT_DOWN = 1;
        private const int EXIT_RIGHT = 2;

        private static int GetDelimitedValue(String text, int pos)
        {
            String[] tokens = text.Split(",");

            int val;
            if (Int32.TryParse(tokens[pos], out val))
            {
                return val;
            }
            return 0;
        }

        private static String Tab(int spaces)
        {
            return new String(' ', spaces);
        }

        public static int Random(int min, int max)
        {
            Random random = new Random();
            return random.Next(max - min) + min;
        }

        public void Play()
        {
            Console.WriteLine(Tab(28) + "AMAZING PROGRAM");
            Console.WriteLine(Tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            Console.WriteLine();

            int width = 0;
            int length = 0;

            do
            {
                String range = DisplayTextAndGetInput("WHAT ARE YOUR WIDTH AND LENGTH");
                if (range.IndexOf(",") > 0)
                {
                    width = GetDelimitedValue(range, 0);
                    length = GetDelimitedValue(range, 1);
                }
            }
            while (width < 1 || length < 1);

            Grid grid = new Grid(length, width);
            int enterCol = grid.SetupEntrance();

            int totalWalls = width * length + 1;
            int count = 2;
            Cell cell = grid.StartingCell();

            while (count != totalWalls)
            {
                List<Direction> possibleDirs = GetPossibleDirs(grid, cell);

                if (possibleDirs.Count != 0)
                {
                    cell = SetCellExit(grid, cell, possibleDirs);
                    cell.Count = count++;
                }
                else
                {
                    cell = grid.GetFirstUnset(cell);
                }
            }
            grid.SetupExit();

            WriteMaze(width, grid, enterCol);
        }

        private Cell SetCellExit(Grid grid, Cell cell, List<Direction> possibleDirs)
        {
            Direction direction = possibleDirs[Random(0, possibleDirs.Count)];
            if (direction == Direction.GO_LEFT)
            {
                cell = grid.GetPrevCol(cell);
                cell.ExitType = EXIT_RIGHT;
            }
            else if (direction == Direction.GO_UP)
            {
                cell = grid.GetPrevRow(cell);
                cell.ExitType = EXIT_DOWN;
            }
            else if (direction == Direction.GO_RIGHT)
            {
                cell.ExitType = cell.ExitType + EXIT_RIGHT;
                cell = grid.GetNextCol(cell);
            }
            else if (direction == Direction.GO_DOWN)
            {
                cell.ExitType = cell.ExitType + EXIT_DOWN;
                cell = grid.GetNextRow(cell);
            }
            return cell;
        }

        private void WriteMaze(int width, Grid grid, int enterCol)
        {
            // top line
            for (int i = 0; i < width; i++)
            {
                if (i == enterCol) Console.Write(".  ");
                else Console.Write(".--");
            }
            Console.WriteLine(".");

            for (int i = 0; i < grid.Length; i++)
            {
                Console.Write("I");
                for (int j = 0; j < grid.Width; j++)
                {
                    if (grid.Cells[i,j].ExitType == EXIT_UNSET || grid.Cells[i, j].ExitType == EXIT_DOWN)
                        Console.Write("  I");
                    else Console.Write("   ");
                }
                Console.WriteLine();
                for (int j = 0; j < grid.Width; j++)
                {
                    if (grid.Cells[i,j].ExitType == EXIT_UNSET || grid.Cells[i, j].ExitType == EXIT_RIGHT)
                        Console.Write(":--");
                    else Console.Write(":  ");
                }
                Console.WriteLine(".");
            }
        }

        private List<Direction> GetPossibleDirs(Grid grid, Cell cell)
        {
            var possibleDirs = new List<Direction>();
            foreach (var val in Enum.GetValues(typeof(Direction)))
            {
                possibleDirs.Add((Direction)val);
            }

            if (cell.Col == FIRST_COL || grid.IsPrevColSet(cell))
            {
                possibleDirs.Remove(Direction.GO_LEFT);
            }
            if (cell.Row == FIRST_ROW || grid.IsPrevRowSet(cell))
            {
                possibleDirs.Remove(Direction.GO_UP);
            }
            if (cell.Col == grid.LastCol || grid.IsNextColSet(cell))
            {
                possibleDirs.Remove(Direction.GO_RIGHT);
            }
            if (cell.Row == grid.LastRow || grid.IsNextRowSet(cell))
            {
                possibleDirs.Remove(Direction.GO_DOWN);
            }
            return possibleDirs;
        }

        private String DisplayTextAndGetInput(String text)
        {
            Console.WriteLine(text);
            return Console.ReadLine();
        }


        private enum Direction
        {
            GO_LEFT,
            GO_UP,
            GO_RIGHT,
            GO_DOWN,
        }

        public class Cell
        {
            public int ExitType { get; set; }
            public int Count { get; set; }

            public int Col { get; set; }
            public int Row { get; set; }

            public Cell(int row, int col)
            {
                ExitType = EXIT_UNSET;
                Row = row;
                Col = col;
            }
        }


        public class Grid
        {
            public Cell[,] Cells { get; private set; }

            public int LastCol { get; set; }
            public int LastRow { get; set; }

            public int Width { get; private set; }
            public int Length { get; private set; }

            private int enterCol;

            public Grid(int length, int width)
            {
                LastCol = width - 1;
                LastRow = length - 1;
                Width = width;
                Length = length;

                Cells = new Cell[length,width];
                for (int i = 0; i < length; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        this.Cells[i,j] = new Cell(i, j);
                    }
                }
            }

            public int SetupEntrance()
            {
                this.enterCol = Random(0, Width);
                Cells[0, enterCol].Count = 1;
                return this.enterCol;
            }

            public void SetupExit()
            {
                int exit = Random(0, Width - 1);
                Cells[LastRow, exit].ExitType += 1;
            }

            public Cell StartingCell()
            {
                return Cells[0, enterCol];
            }

            public bool IsPrevColSet(Cell cell)
            {
                return 0 != Cells[cell.Row, cell.Col - 1].Count;
            }

            public bool IsPrevRowSet(Cell cell)
            {
                return 0 != Cells[cell.Row - 1, cell.Col].Count;
            }

            public bool IsNextColSet(Cell cell)
            {
                return 0 != Cells[cell.Row, cell.Col + 1].Count;
            }

            public bool IsNextRowSet(Cell cell)
            {
                return 0 != Cells[cell.Row + 1, cell.Col].Count;
            }

            public Cell GetPrevCol(Cell cell)
            {
                return Cells[cell.Row, cell.Col - 1];
            }

            public Cell GetPrevRow(Cell cell)
            {
                return Cells[cell.Row - 1, cell.Col];
            }

            public Cell GetNextCol(Cell cell)
            {
                return Cells[cell.Row, cell.Col + 1];
            }

            public Cell GetNextRow(Cell cell)
            {
                return Cells[cell.Row + 1, cell.Col];
            }

            public Cell GetFirstUnset(Cell cell)
            {
                int col = cell.Col;
                int row = cell.Row;
                Cell newCell;
                do
                {
                    if (col != this.LastCol)
                    {
                        col++;
                    }
                    else if (row != this.LastRow)
                    {
                        row++;
                        col = 0;
                    }
                    else
                    {
                        row = 0;
                        col = 0;
                    }
                }
                while ((newCell = Cells[row, col]).Count == 0);
                return newCell;
            }
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            new AmazingGame().Play();
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `02_Amazing/java/Amazing.java`

This appears to be a Java implementation of a simple game of长寿， where two players take turns being the "It" character, with the first player being "It" and the second player being the "Seeker". The "Seeker" trying to find the "It" character, but the "It" character can move around the board and click on different cells to choose their own move. The "It" character can also "smoke", which will randomly hide one cell (all cells with the value "I" will be hidden) without telling the "Seeker" which cell was hidden.

The game has several parameters, including the width and height of the game board, the number of cells per row and column, and the initial position of the "It" character. The "setupExit()" method is called when the game board is setup, and the "setup()" method is called when the game starts.

The "getPrevCol()", "getPrevRow()", "getNextCol()", and "getNextRow()" methods allow the "Seeker" to get the position of the "It" character, the "It" character's own position, and the position of the next cell click, respectively.

The "isPrevColSet()" and "isPrevRowSet()" methods check if the "Seeker" has already set the position of the "It" character.

The "isNextColSet()" and "isNextRowSet()" methods check if the "Seeker" has already set the position of the "It" character.

The "getFirstUnset()` method is used when the "Seeker" clicks on a cell that has not yet been hidden by the "It" character.

Overall, this implementation is quite simple, but it provides a basic starting point for a长寿 game.


```
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;
import static java.lang.System.in;
import static java.lang.System.out;

/**
 * Core algorithm copied from amazing.py
 */
public class Amazing {

    final static int FIRST_COL = 0;
    final static int FIRST_ROW = 0;
    final static int EXIT_UNSET = 0;
    final static int EXIT_DOWN = 1;
    final static int EXIT_RIGHT = 2;
    private final Scanner kbScanner;
    public Amazing() {
        kbScanner = new Scanner(in);
    }

    private static int getDelimitedValue(String text, int pos) {
        String[] tokens = text.split(",");
        try {
            return Integer.parseInt(tokens[pos]);
        } catch (Exception ex) {
            return 0;
        }
    }

    private static String tab(int spaces) {
        char[] spacesTemp = new char[spaces];
        Arrays.fill(spacesTemp, ' ');
        return new String(spacesTemp);
    }

    public static int random(int min, int max) {
        Random random = new Random();
        return random.nextInt(max - min) + min;
    }

    public void play() {
        out.println(tab(28) + "AMAZING PROGRAM");
        out.println(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        out.println();

        int width = 0;
        int length = 0;

        do {
            String range = displayTextAndGetInput("WHAT ARE YOUR WIDTH AND LENGTH");
            if (range.indexOf(",") > 0) {
                width = getDelimitedValue(range, 0);
                length = getDelimitedValue(range, 1);
            }
        } while (width < 1 || length < 1);

        Grid grid = new Grid(length, width);
        int enterCol = grid.setupEntrance();

        int totalWalls = width * length + 1;
        int count = 2;
        Cell cell = grid.startingCell();

        while (count != totalWalls) {
            ArrayList<Direction> possibleDirs = getPossibleDirs(grid, cell);

            if (possibleDirs.size() != 0) {
                cell = setCellExit(grid, cell, possibleDirs);
                cell.count = count++;
            } else {
                cell = grid.getFirstUnset(cell);
            }
        }
        grid.setupExit();

        writeMaze(width, grid, enterCol);
    }

    private Cell setCellExit(Grid grid, Cell cell, ArrayList<Direction> possibleDirs) {
        Direction direction = possibleDirs.get(random(0, possibleDirs.size()));
        if (direction == Direction.GO_LEFT) {
            cell = grid.getPrevCol(cell);
            cell.exitType = EXIT_RIGHT;
        } else if (direction == Direction.GO_UP) {
            cell = grid.getPrevRow(cell);
            cell.exitType = EXIT_DOWN;
        } else if (direction == Direction.GO_RIGHT) {
            cell.exitType = cell.exitType + EXIT_RIGHT;
            cell = grid.getNextCol(cell);
        } else if (direction == Direction.GO_DOWN) {
            cell.exitType = cell.exitType + EXIT_DOWN;
            cell = grid.getNextRow(cell);
        }
        return cell;
    }

    private void writeMaze(int width, Grid grid, int enterCol) {
        // top line
        for (int i = 0; i < width; i++) {
            if (i == enterCol) {
                out.print(".  ");
            } else {
                out.print(".--");
            }
        }
        out.println('.');

        for (Cell[] rows : grid.cells) {
            out.print("I");
            for (Cell cell : rows) {
                if (cell.exitType == EXIT_UNSET || cell.exitType == EXIT_DOWN) {
                    out.print("  I");
                } else {
                    out.print("   ");
                }
            }
            out.println();
            for (Cell cell : rows) {
                if (cell.exitType == EXIT_UNSET || cell.exitType == EXIT_RIGHT) {
                    out.print(":--");
                } else {
                    out.print(":  ");
                }
            }
            out.println(".");
        }
    }

    private ArrayList<Direction> getPossibleDirs(Grid grid, Cell cell) {
        ArrayList<Direction> possibleDirs = new ArrayList<>(Arrays.asList(Direction.values()));

        if (cell.col == FIRST_COL || grid.isPrevColSet(cell)) {
            possibleDirs.remove(Direction.GO_LEFT);
        }
        if (cell.row == FIRST_ROW || grid.isPrevRowSet(cell)) {
            possibleDirs.remove(Direction.GO_UP);
        }
        if (cell.col == grid.lastCol || grid.isNextColSet(cell)) {
            possibleDirs.remove(Direction.GO_RIGHT);
        }
        if (cell.row == grid.lastRow || grid.isNextRowSet(cell)) {
            possibleDirs.remove(Direction.GO_DOWN);
        }
        return possibleDirs;
    }

    private String displayTextAndGetInput(String text) {
        out.print(text);
        return kbScanner.next();
    }

    enum Direction {
        GO_LEFT,
        GO_UP,
        GO_RIGHT,
        GO_DOWN,
    }

    public static class Cell {
        int exitType = EXIT_UNSET;
        int count = 0;

        int col;
        int row;

        public Cell(int row, int col) {
            this.row = row;
            this.col = col;
        }
    }

    public static class Grid {
        Cell[][] cells;

        int lastCol;
        int lastRow;

        int width;
        int enterCol;

        public Grid(int length, int width) {
            this.lastCol = width - 1;
            this.lastRow = length - 1;
            this.width = width;

            this.cells = new Cell[length][width];
            for (int i = 0; i < length; i++) {
                this.cells[i] = new Cell[width];
                for (int j = 0; j < width; j++) {
                    this.cells[i][j] = new Cell(i, j);
                }
            }
        }

        public int setupEntrance() {
            this.enterCol = random(0, this.width);
            cells[0][this.enterCol].count = 1;
            return this.enterCol;
        }

        public void setupExit() {
            int exit = random(0, width - 1);
            cells[lastRow][exit].exitType += 1;
        }

        public Cell startingCell() {
            return cells[0][enterCol];
        }

        public boolean isPrevColSet(Cell cell) {
            return 0 != cells[cell.row][cell.col - 1].count;
        }

        public boolean isPrevRowSet(Cell cell) {
            return 0 != cells[cell.row - 1][cell.col].count;
        }

        public boolean isNextColSet(Cell cell) {
            return 0 != cells[cell.row][cell.col + 1].count;
        }

        public boolean isNextRowSet(Cell cell) {
            return 0 != cells[cell.row + 1][cell.col].count;
        }

        public Cell getPrevCol(Cell cell) {
            return cells[cell.row][cell.col - 1];
        }

        public Cell getPrevRow(Cell cell) {
            return cells[cell.row - 1][cell.col];
        }

        public Cell getNextCol(Cell cell) {
            return cells[cell.row][cell.col + 1];
        }

        public Cell getNextRow(Cell cell) {
            return cells[cell.row + 1][cell.col];
        }

        public Cell getFirstUnset(Cell cell) {
            int col = cell.col;
            int row = cell.row;
            Cell newCell;
            do {
                if (col != this.lastCol) {
                    col++;
                } else if (row != this.lastRow) {
                    row++;
                    col = 0;
                } else {
                    row = 0;
                    col = 0;
                }
            } while ((newCell = cells[row][col]).count == 0);
            return newCell;
        }
    }
}

```

# `02_Amazing/java/AmazingGame.java`

这段代码定义了一个名为AmazingGame的公共类，该类具有一个名为main的静态方法，其参数为字符串数组args，表示程序启动时传递给程序的命令行参数。在main方法中，创建了一个名为Amazing的类的一个实例，然后调用该实例的play()方法。

Amazing是一个抽象类，说明该类中有一个抽象方法，需要使用abstract关键字进行定义。从该代码中无法确定Amazing类中具体的实现，因为并没有定义任何方法或变量。

该代码的作用是创建一个Amazing类的实例，并调用其的play()方法。Amazing类中可能包含游戏逻辑或其他代码，但由于该代码缺少关键的实现细节，无法确定其具体的作用。


```
public class AmazingGame {
    public static void main(String[] args) {
        Amazing amazing = new Amazing();
        amazing.play();
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `02_Amazing/javascript/amazing.js`

该代码是一个 JavaScript 函数，具有以下功能：

1. `print` 函数将一个字符串打印到页面上，并在页面上生成了一个新的文本节点，通过 id 为 "output" 的 div。
2. `input` 函数允许用户输入一行字符。输入的字符串将存储在变量 `input_str` 中，并返回一个 Promise 对象。
3. `input` 函数使用户可以输入一行字符，并在字符串中插入一个输入框。用户输入的字符串将存储在变量 `input_str` 中。
4. `input` 函数还监听了一个名为 "keydown" 的事件，当用户按下键盘上的 13 键时，将获取到用户输入的字符串，并将其存储在 `input_str` 变量中。
5. `print` 函数将在页面上打印字符串，并将其打印到与 "output"  div 相同的 "output" div 中。


```
// AMAZING
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

这段代码定义了一个名为 tab 的函数，它会接收一个参数 space，并返回一个字符串。这个字符串是由 space 变量所减去的一个空格数组，也就是在 space 为 0 时返回一个空字符串。

在函数内部，使用 while 循环来创建一个空格字符串，并将其添加到 str 变量中。每次循环时，space 变量都会减少 1，因此循环将继续进行，直到 space 为 0。

最终，函数返回 str 变量，即一个由 28 个空格组成的字符串，其中第 28 个空格是一个换行符。

这段代码的作用是输出一个由多个空格组成的字符串，每个空格宽度逐渐增加。在输出时，tab 函数会根据传入的参数（空格的数量）输出不同的字符数。例如，当传递一个参数 28 时，输出将会有 28 个空格，字符串将会有一个明显的换行符。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

print(tab(28) + "AMAZING PROGRAM\n");
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
print("\n");
print("\n");
print("\n");
print("FOR EXAMPLE TYPE 10,10 AND PRESS ENTER\n");
print("\n");

```

This is a program that reads a 2D matrix of integers and a 2D array of strings. It starts by reading the matrix of integers. Next, it reads the 2D array of strings and then the matrix of integers again.

The program then reads the rows of the matrix of integers. For each row, it checks if the block of integers is already processed. If it is not, it reads the block and processes it. If it is already processed, it prints the block and continues to the next row.

The program also reads the columns of the matrix of integers. For each column, it reads the block from the left and processes it. If the block is already processed, it prints the block and continues to the next column.

Finally, the program prints the matrix of integers.


```
// Main program
async function main()
{
    while (1) {
        print("WHAT ARE YOUR WIDTH AND LENGTH");
        a = await input();
        h = parseInt(a);
        v2 = parseInt(a.substr(a.indexOf(",") + 1));
        if (h > 1 && v2 > 1)
            break;
        print("MEANINGLESS DIMENSIONS.  TRY AGAIN.\n");
    }
    w = [];
    v = [];
    for (i = 1; i <= h; i++) {
        w[i] = [];
        v[i] = [];
        for (j = 1; j <= v2; j++) {
            w[i][j] = 0;
            v[i][j] = 0;
        }
    }
    print("\n");
    print("\n");
    print("\n");
    print("\n");
    q = 0;
    z = 0;
    x = Math.floor(Math.random() * h + 1);
    for (i = 1; i <= h; i++) {
        if (i == x)
            print(".  ");
        else
            print(".--");
    }
    print(".\n");
    c = 1;
    w[x][1] = c;
    c++;
    r = x;
    s = 1;
    entry = 0;
    while (1) {
        if (entry == 2) {	// Search for a non-explored cell
            do {
                if (r < h) {
                    r++;
                } else if (s < v2) {
                    r = 1;
                    s++;
                } else {
                    r = 1;
                    s = 1;
                }
            } while (w[r][s] == 0) ;
        }
        if (entry == 0 && r - 1 > 0 && w[r - 1][s] == 0) {	// Can go left?
            if (s - 1 > 0 && w[r][s - 1] == 0) {	// Can go up?
                if (r < h && w[r + 1][s] == 0) {	// Can go right?
                    // Choose left/up/right
                    x = Math.floor(Math.random() * 3 + 1);
                } else if (s < v2) {
                    if (w[r][s + 1] == 0) {	// Can go down?
                        // Choose left/up/down
                        x = Math.floor(Math.random() * 3 + 1);
                        if (x == 3)
                            x = 4;
                    } else {
                        x = Math.floor(Math.random() * 2 + 1);
                    }
                } else if (z == 1) {
                    x = Math.floor(Math.random() * 2 + 1);
                } else {
                    q = 1;
                    x = Math.floor(Math.random() * 3 + 1);
                    if (x == 3)
                        x = 4;
                }
            } else if (r < h && w[r + 1][s] == 0) {	// Can go right?
                if (s < v2) {
                    if (w[r][s + 1] == 0) {	// Can go down?
                        // Choose left/right/down
                        x = Math.floor(Math.random() * 3 + 1);
                    } else {
                        x = Math.floor(Math.random() * 2 + 1);
                    }
                    if (x >= 2)
                        x++;
                } else if (z == 1) {
                    x = Math.floor(Math.random() * 2 + 1);
                    if (x >= 2)
                        x++;
                } else {
                    q = 1;
                    x = Math.floor(Math.random() * 3 + 1);
                    if (x >= 2)
                        x++;
                }
            } else if (s < v2) {
                if (w[r][s + 1] == 0) {	// Can go down?
                    // Choose left/down
                    x = Math.floor(Math.random() * 2 + 1);
                    if (x == 2)
                        x = 4;
                } else {
                    x = 1;
                }
            } else if (z == 1) {
                x = 1;
            } else {
                q = 1;
                x = Math.floor(Math.random() * 2 + 1);
                if (x == 2)
                    x = 4;
            }
        } else if (s - 1 > 0 && w[r][s - 1] == 0) {	// Can go up?
            if (r < h && w[r + 1][s] == 0) {
                if (s < v2) {
                    if (w[r][s + 1] == 0)
                        x = Math.floor(Math.random() * 3 + 2);
                    else
                        x = Math.floor(Math.random() * 2 + 2);
                } else if (z == 1) {
                    x = Math.floor(Math.random() * 2 + 2);
                } else {
                    q = 1;
                    x = Math.floor(Math.random() * 3 + 2);
                }
            } else if (s < v2) {
                if (w[r][s + 1] == 0) {
                    x = Math.floor(Math.random() * 2 + 2);
                    if (x == 3)
                        x = 4;
                } else {
                    x = 2;
                }
            } else if (z == 1) {
                x = 2;
            } else {
                q = 1;
                x = Math.floor(Math.random() * 2 + 2);
                if (x == 3)
                    x = 4;
            }
        } else if (r < h && w[r + 1][s] == 0) {	// Can go right?
            if (s < v2) {
                if (w[r][s + 1] == 0)
                    x = Math.floor(Math.random() * 2 + 3);
                else
                    x = 3;
            } else if (z == 1) {
                x = 3;
            } else {
                q = 1;
                x = Math.floor(Math.random() * 2 + 3);
            }
        } else if (s < v2) {
            if (w[r][s + 1] == 0) 	// Can go down?
                x = 4;
            else {
                entry = 2;	// Blocked!
                continue;
            }
        } else if (z == 1) {
            entry = 2;	// Blocked!
            continue;
        } else {
            q = 1;
            x = 4;
        }
        if (x == 1) {	// Left
            w[r - 1][s] = c;
            c++;
            v[r - 1][s] = 2;
            r--;
            if (c == h * v2 + 1)
                break;
            q = 0;
            entry = 0;
        } else if (x == 2) {	// Up
            w[r][s - 1] = c;
            c++;
            v[r][s - 1] = 1;
            s--;
            if (c == h * v2 + 1)
                break;
            q = 0;
            entry = 0;
        } else if (x == 3) {	// Right
            w[r + 1][s] = c;
            c++;
            if (v[r][s] == 0)
                v[r][s] = 2;
            else
                v[r][s] = 3;
            r++;
            if (c == h * v2 + 1)
                break;
            entry = 1;
        } else if (x == 4) {	// Down
            if (q != 1) {	// Only if not blocked
                w[r][s + 1] = c;
                c++;
                if (v[r][s] == 0)
                    v[r][s] = 1;
                else
                    v[r][s] = 3;
                s++;
                if (c == h * v2 + 1)
                    break;
                entry = 0;
            } else {
                z = 1;
                if (v[r][s] == 0) {
                    v[r][s] = 1;
                    q = 0;
                    r = 1;
                    s = 1;
                    while (w[r][s] == 0) {
                        if (r < h) {
                            r++;
                        } else if (s < v2) {
                            r = 1;
                            s++;
                        } else {
                            r = 1;
                            s = 1;
                        }
                    }
                    entry = 0;
                } else {
                    v[r][s] = 3;
                    q = 0;
                    entry = 2;
                }
            }
        }
    }
    for (j = 1; j <= v2; j++) {
        str = "I";
        for (i = 1; i <= h; i++) {
            if (v[i][j] < 2)
                str += "  I";
            else
                str += "   ";
        }
        print(str + "\n");
        str = "";
        for (i = 1; i <= h; i++) {
            if (v[i][j] == 0 || v[i][j] == 2)
                str += ":--";
            else
                str += ":  ";
        }
        print(str + ".\n");
    }
```

这段代码的作用是输出一个二维数组中元素的访问顺序。具体来说，代码会遍历这个二维数组，对于每个元素，会先将其存储在字符串中，然后遍历该元素所在的行，并将该元素值乘以宽度w[i][j]并将其添加到字符串中，最后输出该字符串。

换句话说，这段代码的作用是输出二维数组中元素从左上角到右下角、按照访问顺序依次被访问的行列。


```
// If you want to see the order of visited cells
//    for (j = 1; j <= v2; j++) {
//        str = "I";
//        for (i = 1; i <= h; i++) {
//            str += w[i][j] + " ";
//        }
//        print(str + "\n");
//    }
}

main();

```