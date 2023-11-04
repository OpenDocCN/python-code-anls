# BasicComputerGames源码解析 87

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


# `94_War/javascript/war.js`

这两行代码定义了两个函数，分别是 `print` 和 `tab`。

1. `print` 函数的作用是将一个字符串 `str` 打印到页面上一个叫做 `output` 的元素中。这个元素在代码中没有定义，它可能是页面上的一个空格或者是一个 HTML 元素，具体取决于代码的上下文。但在这个函数中，我们将 `str` 作为参数，使用 `document.getElementById` 获取到 `output` 元素，然后使用 `document.createTextNode` 创建一个字符节点，将 `str` 作为节点内容，最后将节点添加到 `output` 元素中。这样，当这个函数被调用时，它将在页面上打印出 `str` 这个字符串。

2. `tab` 函数的作用是返回一个字符串，这个字符串由若干个空格组成。这个函数比较简单，它只是通过循环来生成一些空格，然后返回包含这些空格的字符串。


```
// WAR
//
// Original conversion from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

function print(str) {
    document.getElementById("output").appendChild(document.createTextNode(str));
}

function tab(space) {
    let str = "";
    while (space-- > 0) {
        str += " ";
    }
    return str;
}

```



该函数的作用是在页面中接收用户输入的文本输入框，并等待用户输入后将其值存储在变量中，并将结果打印到页面上。

具体来说，该函数创建了一个新的Promise对象，并在其中定义了一个resolve函数，用来存储用户输入的文本。然后该函数创建了一个input元素，并将其添加到页面上。该元素设置了一个"type"属性的值为"text"，设置了一个"length"属性的值为"50"，这样它就可以接收用户输入的文本了。

该函数还添加了一个键盘事件监听器，当用户按下键盘上的13时，将接收到的值存储在input元素的值中，并打印到页面上。此外，当用户在输入框中更改内容时，还会触发该事件，函数将接收到的值存储在变量中，并打印到页面上。

最后，该函数通过调用resolve函数来存储用户输入的文本，并在调用结束后返回该变量的值。


```
function input() {
    return new Promise(function (resolve) {
        const input_element = document.createElement("INPUT");

        print("? ");
        input_element.setAttribute("type", "text");
        input_element.setAttribute("length", "50");
        document.getElementById("output").appendChild(input_element);
        input_element.focus();
        input_element.addEventListener("keydown", function (event) {
            if (event.keyCode == 13) {
                const input_str = input_element.value;
                document.getElementById("output").removeChild(input_element);
                print(input_str);
                print("\n");
                resolve(input_str);
            }
        });
    });
}

```

这段代码定义了一个名为 `askYesOrNo` 的函数，它会向用户提问 "Do you like it?" 并提示用户需要输入 "yes" 或 "no"。如果用户输入 "yes"，则函数会返回 true，否则返回 false。

函数内部使用了一个无限循环，会一直反复向用户提问，直到用户回答为止。循环中，函数会先输出问题，然后等待用户输入，再将用户输入转换成字符串并将其转换为小写。接着，函数会通过调用 `(await input()).toUpperCase()` 来获取用户输入，并检查它是否等于 "yes" 或 "no"。

如果用户输入 "yes"，则函数返回 `true`，否则返回 `false`。如果用户输入 "NO"，则函数会输出 "YES OR NO, PLEASE."。


```
async function askYesOrNo(question) {
    while (1) {
        print(question);
        const str = (await input()).toUpperCase();
        if (str === "YES") {
            return true;
        }
        else if (str === "NO") {
            return false;
        }
        else {
            print("YES OR NO, PLEASE.  ");
        }
    }
}

```



这是一个使用 JavaScript 编写的 Async/Await 函数，它的作用是询问玩家是否需要游戏建议，并基于玩家是否想要的回答返回一个游戏牌的牌面。

具体来说，它执行了以下操作：

1. 使用 `askYesOrNo` 函数询问玩家是否想要游戏建议。
2. 如果玩家选择了“是”，那么执行以下操作：
	1. 打印一段游戏规则提示信息，告诉玩家游戏结束了，无论他们是否选择继续或完成游戏。
	2. 打印两个感叹号，表明游戏已经结束。
	3. 输出两条换行符，清除之前的输出并开始输出新的牌面。
	4. 输出两个换行符，结束游戏并清除之前的输出。

另外，它还执行了以下操作：

1. 定义了一个名为 `createGameDeck` 的函数，该函数接收一个游戏牌的牌面数量和游戏大小作为参数。
2. `createGameDeck` 函数使用循环和 `Math.random()` 函数生成一个包含游戏牌面数量张随机牌面的游戏牌。
3. `askYesOrNo` 函数是一个静态函数，它的作用是询问玩家是否想要游戏建议。


```
async function askAboutInstructions() {
    const playerWantsInstructions = await askYesOrNo("DO YOU WANT DIRECTIONS");
    if (playerWantsInstructions) {
        print("THE COMPUTER GIVES YOU AND IT A 'CARD'.  THE HIGHER CARD\n");
        print("(NUMERICALLY) WINS.  THE GAME ENDS WHEN YOU CHOOSE NOT TO\n");
        print("CONTINUE OR WHEN YOU HAVE FINISHED THE PACK.\n");
    }
    print("\n");
    print("\n");
}

function createGameDeck(cards, gameSize) {
    const deck = [];
    const deckSize = cards.length;
    for (let j = 0; j < gameSize; j++) {
        let card;

        // Compute a new card index until we find one that isn't already in the new deck
        do {
            card = Math.floor(deckSize * Math.random());
        } while (deck.includes(card));
        deck.push(card);
    }
    return deck;
}

```

这些代码定义了三个函数，分别是 `computeCardValue`、`printGameOver` 和 `printTitle`。下面分别对这三个函数进行解释。

1. `computeCardValue(cardIndex)`：这是一个计算游戏牌价值的函数，它将传入的 `cardIndex` 除以 4，并取整。这个函数的作用是，如果只有一张牌，则将其返回；如果有两张牌，则将它们分别返回；如果有三张牌，则将它们分别返回。

2. `printGameOver(playerScore, computerScore)`：这是一个打印游戏结束信息的函数，它接收两个参数 `playerScore` 和 `computerScore`，并将它们打印出来。这个函数的作用是，在游戏结束后打印一些游戏结束信息，如 "WE HAVE RUN OUT OF CARDS.  FINAL SCORE:  YOU: ${playerScore}  THE COMPUTER: ${computerScore}" 等。

3. `printTitle()`：这是一个打印游戏标题的函数，它使用 `tab` 函数将字符串打印出来。这个函数的作用是，在游戏开始时打印一些游戏标题信息，如 "WARDPORT - CARD GAME OF WAR" 等。


```
function computeCardValue(cardIndex) {
    return Math.floor(cardIndex / 4);
}

function printGameOver(playerScore, computerScore) {
    print("\n");
    print("\n");
    print(`WE HAVE RUN OUT OF CARDS.  FINAL SCORE:  YOU: ${playerScore}  THE COMPUTER: ${computerScore}\n`);
    print("\n");
}

function printTitle() {
    print(tab(33) + "WAR\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("THIS IS THE CARD GAME OF WAR.  EACH CARD IS GIVEN BY SUIT-#\n");
    print("AS S-7 FOR SPADE 7.  ");
}

```

这段代码定义了一个名为 printCards 的函数，该函数接收两个参数：playerCard 和 computerCard。函数的作用是打印两张牌的名称，并输出到控制台。

在代码中，首先定义了一个名为 cards 的数组，该数组包含了多张牌。然后，在 printCards 函数内部，使用了循环来遍历 cards 数组中的每一张牌，并打印出了该张牌的名称。

该函数的作用是输出两张牌的名称，以便玩家和计算机可以查看它们。


```
function printCards(playerCard, computerCard) {
    print("\n");
    print(`YOU: ${playerCard}\tCOMPUTER: ${computerCard}\n`);
}

const cards = [
    "S-2", "H-2", "C-2", "D-2",
    "S-3", "H-3", "C-3", "D-3",
    "S-4", "H-4", "C-4", "D-4",
    "S-5", "H-5", "C-5", "D-5",
    "S-6", "H-6", "C-6", "D-6",
    "S-7", "H-7", "C-7", "D-7",
    "S-8", "H-8", "C-8", "D-8",
    "S-9", "H-9", "C-9", "D-9",
    "S-10", "H-10", "C-10", "D-10",
    "S-J", "H-J", "C-J", "D-J",
    "S-Q", "H-Q", "C-Q", "D-Q",
    "S-K", "H-K", "C-K", "D-K",
    "S-A", "H-A", "C-A", "D-A"
];

```

这段代码是一个名为 `main` 的函数，是这段代码的主控制部分。

这个函数内部async/await 并调用了一个名为 `askAboutInstructions` 的异步函数，这个函数会在程序中询问用户有关游戏规则的说明。

函数内部创建了一个 `计算机分数` 和一个 `玩家分数` 变量，分别用来记录游戏中的得分。

然后，它使用 `createGameDeck` 函数从游戏牌组中随机抽取了指定数量的牌，这些牌的数量是 `cards.length`。

接下来，它使用一个无限循环来让游戏继续进行，每次从牌组中随机抽取一张牌，然后打印出这张牌以及这张牌在玩家和电脑手中的得分。

如果牌组为空，函数会打印出游戏结束的消息以及两个得分，否则，它询问用户是否要继续游戏，如果用户选择继续，则重新循环回到执行 `askAboutInstructions` 函数这一步。

最后，函数在程序结束时打印一条消息，感谢用户参加游戏，并且输出一些信息，包括游戏已经进行了多长时间，以及两个得分的总和。


```
// Main control section
async function main() {
    printTitle();
    await askAboutInstructions();

    let computerScore = 0;
    let playerScore = 0;

    // Generate a random deck
    const gameSize = cards.length;    // Number of cards to shuffle into the game deck.  Can be <= cards.length.
    const deck = createGameDeck(cards, gameSize);
    let shouldContinuePlaying = true;

    while (deck.length > 0 && shouldContinuePlaying) {
        const playerCard = deck.shift();    // Take a card
        const computerCard = deck.shift();    // Take a card
        printCards(cards[playerCard], cards[computerCard]);

        const playerCardValue = computeCardValue(playerCard);
        const computerCardValue = computeCardValue(computerCard);
        if (playerCardValue < computerCardValue) {
            computerScore++;
            print("THE COMPUTER WINS!!! YOU HAVE " + playerScore + " AND THE COMPUTER HAS " + computerScore + "\n");
        } else if (playerCardValue > computerCardValue) {
            playerScore++;
            print("YOU WIN. YOU HAVE " + playerScore + " AND THE COMPUTER HAS " + computerScore + "\n");
        } else {
            print("TIE.  NO SCORE CHANGE.\n");
        }

        if (deck.length === 0) {
            printGameOver(playerScore, computerScore);
        }
        else {
            shouldContinuePlaying = await askYesOrNo("DO YOU WANT TO CONTINUE");
        }
    }
    print("THANKS FOR PLAYING.  IT WAS FUN.\n");
    print("\n");
}

```

这是C++程序的main函数，它被称为程序的入口点。在这个函数中，程序会开始执行一些必要的操作，包括初始化计算机硬件和加载操作系统。然后，程序会执行用户提供的输入或默认设置，并根据需要执行一些特定的代码。

对于这段代码而言，它定义了一个名为main的函数，但没有进行任何实际的初始化或执行任何操作。它只是声明了一个名为main的函数，但没有对其进行定义或赋值。因此，这段代码不能被程序解释器识别为代码，也不能执行任何操作。


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


# `94_War/python/war.py`

这段代码是一个Python脚本，主要实现了将 Basic 编程语言中的“战争纸牌”游戏数据导入到 Python 中的功能。具体来说，这段代码做了以下几件事情：

1. 导入 Basic 语言相关的包（import card_value, etc.）。这些包中定义了一些函数和变量，与战争纸牌游戏的数据有关。
2. 定义了一个名为“card_value”的函数，它接收一个输入字符串参数。这个函数的作用是将输入的字符串拆分成两个部分，然后根据它们在字符串中出现的顺序，返回一个相应的数字。这部分代码与战争纸牌游戏中不同牌型的值相对应。
3. 在主程序中调用“card_value”函数，并将基本输入数据（如游戏地图、目标牌、手牌等）作为参数传入。这样，每次调用函数时，它都会根据不同的参数返回不同的牌值。
4. 对于每个牌值，程序会将其显示在控制台。


```
#!/usr/bin/env python3

"""
WAR

Converted from BASIC to Python by Trevor Hobson
"""

import json
import random
from pathlib import Path
from typing import List


def card_value(input: str) -> int:
    return ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"].index(
        input.split("-")[1]
    )


```

这段代码是一个 Python 函数，名为 `play_game()`，它实现了的游戏只有一个回合。在这个函数中，定义了一个 `with` 语句打开一个名为 `cards.json` 的文件，并将其中的内容存储在一个名为 `cards` 的列表中。

接着，使用 `random.shuffle()` 函数对列表中的元素进行随机化，这将随机打乱列表中元素的位置。

然后，定义了两个变量 `score_you` 和 `score_computer`，它们用于存储游戏中的得分。另外，定义了一个变量 `cards_left`，用于跟踪剩余的卡片数量。

接下来，使用一个循环来模拟游戏的每一回合。在每次循环中，使用 `print()` 函数输出游戏中的信息，包括发牌人和牌面值，然后使用 `card_value()` 函数比较两张牌的大小，并更新游戏中的得分。

最后，在游戏结束时，输出游戏的得分，并在可能的条件下继续游戏。


```
def play_game() -> None:
    """Play one round of the game"""
    with open(Path(__file__).parent / "cards.json") as f:
        cards: List[str] = json.load(f)

    random.shuffle(cards)
    score_you = 0
    score_computer = 0
    cards_left = 52
    for round in range(26):
        print()
        card_you = cards[round]
        card_computer = cards[round * 2]
        print("You:", card_you, " " * (8 - len(card_you)) + "Computer:", card_computer)
        value_you = card_value(card_you)
        value_computer = card_value(card_computer)
        if value_you > value_computer:
            score_you += 1
            print(
                "You win. You have", score_you, "and the computer has", score_computer
            )
        elif value_computer > value_you:
            score_computer += 1
            print(
                "The computer wins!!! You have",
                score_you,
                "and the computer has",
                score_computer,
            )
        else:
            print("Tie. No score change.")
        cards_left -= 2
        if cards_left > 2 and input("Do you want to continue ").lower().startswith("n"):
            break
    if cards_left == 0:
        print(
            "\nWe have run out of cards. Final score: You:",
            score_you,
            "the computer:",
            score_computer,
        )
    print("\nThanks for playing. It was fun.")


```

这段代码是一个 Python 的函数，名为 `main()`。函数的作用是在屏幕上打印出一个卡片游戏（War）的规则，并接受玩家输入的游戏类型（是“进行游戏”还是“结束游戏”），如果玩家输入“进行游戏”，则函数会进行游戏的处理，否则结束游戏。

具体来说，这段代码的功能如下：

1. 在屏幕上打印出包含 "WAR" 和 "CREATIVE COMPUTING" 的文本，并在此文本后跟上 "MORRISTOWN, NEW JERSEY"。
2. 在屏幕上打印出 "This is the card game of war. Each card is given by suit-#" 的文本，说明这是一个卡片游戏，每张卡片上的花色编号。
3. 如果玩家输入 "y"，则函数会告诉玩家游戏已经结束，让玩家选择是否继续游戏或者结束游戏。如果玩家输入 "no"，则直接结束游戏。
4. 玩家在游戏中可以选择继续游戏，只要输入 "y"，然后再次运行函数。

这段代码的主要目的是提供一个卡片游戏，并在游戏中告诉玩家游戏的规则和如何进行游戏。


```
def main() -> None:
    print(" " * 33 + "WAR")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")
    print("This is the card game of war. Each card is given by suit-#")
    print("as S-7 for Spade 7.")
    if input("Do you want directions ").lower().startswith("y"):
        print("The computer gives you and it a 'card'. The higher card")
        print("(numerically) wins. The game ends when you choose not to")
        print("continue or when you have finished the pack.")

    keep_playing = True
    while keep_playing:
        play_game()
        keep_playing = input("\nPlay again? (yes or no) ").lower().startswith("y")


```

这段代码是一个Python程序中的一个if语句，它的作用是判断当前程序是否作为主程序运行。如果当前程序作为主程序运行，那么程序会执行if语句内部的代码。

if语句是一个布尔表达式，可以测试某个条件是否为真。如果条件为真，则if语句内部的代码会被执行，否则不会执行。

在这段代码中，if语句的判断条件是"`__name__` == "__main__"。这个条件的意思是：程序的名称是否和`__main__`相同。如果当前程序的名称和`__main__`相同，那么程序会执行if语句内部的代码。

由于`__name__`和`__main__`在Python中是相同的，因此if语句不会产生任何输出，也不会影响程序的执行流程。


```
if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by Anthony Rubick [AnthonyMichaelTDM](https://github.com/AnthonyMichaelTDM)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Weekday

This program gives facts about your date of birth (or some other day of interest). It is not prepared to give information on people born before the use of the current type of calendar, i.e. year 1582.

You merely enter today’s date in the form—month, day, year and your date of birth in the same form. The computer then tells you the day of the week of your birth date, your age, and how much time you have spent sleeping, eating, working, and relaxing.

This program was adapted from a GE timesharing program by Tom Kloos at the Oregon Museum of Science and Industry.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=179)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=194)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `95_Weekday/csharp/Program.cs`

This is a code for a program that calculates a person's lifespan based on their birthdate and the number of days they have lived. It assumes that the person has provided their birthdate and age in their profile, and it outputs the number of days they have lived and a phrase indicating whether they have lived a full life or not.

The program first calculates the person's expected age based on their birthdate, using the formula for someone of their age, and then calculates the number of days they have lived by subtracting


```
﻿using System.Text;

namespace Weekday
{
    class Weekday
    {
        private void DisplayIntro()
        {
            Console.WriteLine("");
            Console.WriteLine("SYNONYM".PadLeft(23));
            Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            Console.WriteLine("");
            Console.WriteLine("Weekday is a computer demonstration that");
            Console.WriteLine("gives facts about a date of interest to you.");
            Console.WriteLine("");
        }

        private bool ValidateDate(string InputDate, out DateTime ReturnDate)
        {
            // The expectation is that the input is in the format D,M,Y
            // but any valid date format (other than with commas) will work
            string DateString = InputDate.Replace(",", "/");

            return (DateTime.TryParse(DateString, out ReturnDate));
        }

        private DateTime PromptForADate(string Prompt)
        {
            bool Success = false;
            string LineInput = String.Empty;
            DateTime TodaysDate = DateTime.MinValue;

            // Get the date for input and validate it
            while (!Success)
            {
                Console.Write(Prompt);
                LineInput = Console.ReadLine().Trim().ToLower();

                Success = ValidateDate(LineInput, out TodaysDate);

                if (!Success)
                {
                    Console.WriteLine("*** Invalid date.  Please try again.");
                    Console.WriteLine("");
                }
            }

            return TodaysDate;
        }

        private void CalculateDateDiff(DateTime TodaysDate, DateTime BirthDate, Double Factor, out int AgeInYears, out int AgeInMonths, out int AgeInDays)
        {
            // leveraged Stack Overflow answer: https://stackoverflow.com/a/3055445

            // Convert to number of days since Birth Date, multiple by factor then store as new FactorDate
            TimeSpan TimeDiff = TodaysDate.Subtract(BirthDate);
            Double NumberOfDays = TimeDiff.Days * Factor;
            DateTime FactorDate = BirthDate.AddDays(NumberOfDays);

            // Compute difference between FactorDate (which is TodaysDate * Factor) and BirthDate
            AgeInMonths = FactorDate.Month - BirthDate.Month;
            AgeInYears = FactorDate.Year - BirthDate.Year;

            if (FactorDate.Day < BirthDate.Day)
            {
                AgeInMonths--;
            }

            if (AgeInMonths < 0)
            {
                AgeInYears--;
                AgeInMonths += 12;
            }

            AgeInDays = (FactorDate - BirthDate.AddMonths((AgeInYears * 12) + AgeInMonths)).Days;

        }

        private void WriteColumnOutput(string Message, int Years, int Months, int Days)
        {

            Console.WriteLine("{0,-25} {1,-10:N0} {2,-10:N0} {3,-10:N0}", Message, Years, Months, Days);

        }

        private void DisplayOutput(DateTime TodaysDate, DateTime BirthDate)
        {
            Console.WriteLine("");

            // Not allowed to play if the current year is before 1582
            if (TodaysDate.Year < 1582)
            {
                Console.WriteLine("Not prepared to give day of week prior to MDLXXXII.");
                return;
            }

            // Share which day of the week the BirthDate was on
            Console.Write(" {0} ", BirthDate.ToString("d"));

            string DateVerb = "";
            if (BirthDate.CompareTo(TodaysDate) < 0)
            {
                DateVerb = "was a ";
            }
            else if (BirthDate.CompareTo(TodaysDate) == 0)
            {
                DateVerb = "is a ";
            }
            else
            {
                DateVerb = "will be a ";
            }
            Console.Write("{0}", DateVerb);

            // Special warning if their birth date was on a Friday the 13th!
            if (BirthDate.DayOfWeek.ToString().Equals("Friday") && BirthDate.Day == 13)
            {
                Console.WriteLine("{0} the Thirteenth---BEWARE", BirthDate.DayOfWeek.ToString());
            }
            else
            {
                Console.WriteLine("{0}", BirthDate.DayOfWeek.ToString());
            }

            // If today's date is the same month & day as the birth date then wish them a happy birthday!
            if (BirthDate.Month == TodaysDate.Month && BirthDate.Day == TodaysDate.Day)
            {
                Console.WriteLine("");
                Console.Write("***Happy Birthday***");
            }

            Console.WriteLine("");

            // Only show the date calculations if BirthDate is before TodaysDate
            if (DateVerb.Trim().Equals("was a"))
            {

                Console.WriteLine("{0,-24} {1,-10} {2,-10} {3,-10}", " ", "Years", "Months", "Days");

                int TheYears = 0, TheMonths = 0, TheDays = 0;
                int FlexYears = 0, FlexMonths = 0, FlexDays = 0;

                CalculateDateDiff(TodaysDate, BirthDate, 1, out TheYears, out TheMonths, out TheDays);
                WriteColumnOutput("Your age if birthdate", TheYears, TheMonths, TheDays);

                FlexYears = TheYears;
                FlexMonths = TheMonths;
                FlexDays = TheDays;
                CalculateDateDiff(TodaysDate, BirthDate, .35, out FlexYears, out FlexMonths, out FlexDays);
                WriteColumnOutput("You have slept", FlexYears, FlexMonths, FlexDays);

                FlexYears = TheYears;
                FlexMonths = TheMonths;
                FlexDays = TheDays;
                CalculateDateDiff(TodaysDate, BirthDate, .17, out FlexYears, out FlexMonths, out FlexDays);
                WriteColumnOutput("You have eaten", FlexYears, FlexMonths, FlexDays);

                FlexYears = TheYears;
                FlexMonths = TheMonths;
                FlexDays = TheDays;
                CalculateDateDiff(TodaysDate, BirthDate, .23, out FlexYears, out FlexMonths, out FlexDays);
                string FlexPhrase = "You have played";
                if (TheYears > 3)
                    FlexPhrase = "You have played/studied";
                if (TheYears > 9)
                    FlexPhrase = "You have worked/played";
                WriteColumnOutput(FlexPhrase, FlexYears, FlexMonths, FlexDays);

                FlexYears = TheYears;
                FlexMonths = TheMonths;
                FlexDays = TheDays;
                CalculateDateDiff(TodaysDate, BirthDate, .25, out FlexYears, out FlexMonths, out FlexDays);
                WriteColumnOutput("You have relaxed", FlexYears, FlexMonths, FlexDays);

                Console.WriteLine("");
                Console.WriteLine("* You may retire in {0} *".PadLeft(38), BirthDate.Year + 65);
            }
        }

        public void PlayTheGame()
        {
            DateTime TodaysDate = DateTime.MinValue;
            DateTime BirthDate = DateTime.MinValue;

            DisplayIntro();

            TodaysDate = PromptForADate("Enter today's date in the form: 3,24,1978  ? ");
            BirthDate = PromptForADate("Enter day of birth (or other day of interest)? ");

            DisplayOutput(TodaysDate, BirthDate);

        }
    }
    class Program
    {
        static void Main(string[] args)
        {

            new Weekday().PlayTheGame();

        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `95_Weekday/java/Weekday.java`

This is a Java program that provides a simple weekday demonstration for a given date of interest. The program reads the input date from the user and validates that it is in the expected format (3,24,1979). The program then displays the weekday and the date itself.

Here's a brief explanation of the code:

1. The program defines a `WeekdayIsComputerDemonstration` class that overrides the `run` method.
2. In the `run` method, the program reads the input date from the user using a `Scanner` object and a `String[]` array.
3. The program then validates that the input date is in the expected format (3,24,1979) and stores the `month`, `day`, and `year` values in the corresponding fields of the `DateStruct` object.
4. The program displays the weekday (Monday, Tuesday, etc.) and the date in the appropriate format using `System.out.println`.

Overall, this program is designed to provide a simple weekday demonstration for a given date of interest.


```
import java.util.Scanner;

/**
 * WEEKDAY
 *
 * Converted from BASIC to Java by Aldrin Misquitta (@aldrinm)
 *
 */
public class Weekday {

	//TABLE OF VALUES FOR THE MONTHS TO BE USED IN CALCULATIONS.
	//Dummy value added at index 0, so we can reference directly by the month number value
	private final static int[] t = new int[]{-1, 0, 3, 3, 6, 1, 4, 6, 2, 5, 0, 3, 5};

	public static void main(String[] args) {
		printIntro();

		Scanner scanner = new Scanner(System.in);
		System.out.print("ENTER TODAY'S DATE IN THE FORM: 3,24,1979 ");
		DateStruct todaysDate = readDate(scanner);

		System.out.print("ENTER DAY OF BIRTH (OR OTHER DAY OF INTEREST) ");
		DateStruct dateOfInterest = readDate(scanner);

		int I1 = (dateOfInterest.year - 1500) / 100;
		//TEST FOR DATE BEFORE CURRENT CALENDAR.
		if ((dateOfInterest.year - 1582) >= 0) {
			int A = I1 * 5 + (I1 + 3) / 4;
			int I2 = (A - b(A) * 7);
			int Y2 = (dateOfInterest.year / 100);
			int Y3 = (dateOfInterest.year - Y2 * 100);
			A = Y3 / 4 + Y3 + dateOfInterest.day + t[dateOfInterest.month] + I2;
			calculateAndPrintDayOfWeek(I1, A, todaysDate, dateOfInterest, Y3);

			if ((todaysDate.year * 12 + todaysDate.month) * 31 + todaysDate.day
					== (dateOfInterest.year * 12 + dateOfInterest.month) * 31 + dateOfInterest.day) {
				return; //stop the program
			}

			int I5 = todaysDate.year - dateOfInterest.year;
			System.out.print("\n");
			int I6 = todaysDate.month - dateOfInterest.month;
			int I7 = todaysDate.day - dateOfInterest.day;
			if (I7 < 0) {
				I6 = I6 - 1;
				I7 = I7 + 30;
			}
			if (I6 < 0) {
				I5 = I5 - 1;
				I6 = I6 + 12;
			}
			if (I5 < 0) {
				return; //do nothing. end the program
			} else {
				if (I7 != 0) {
					printHeadersAndAge(I5, I6, I7);
				} else {
					if (I6 != 0) {
						printHeadersAndAge(I5, I6, I7);
					} else {
						System.out.println("***HAPPY BIRTHDAY***");
						printHeadersAndAge(I5, I6, I7);
					}
				}
			}

			int A8 = (I5 * 365) + (I6 * 30) + I7 + (I6 / 2);
			int K5 = I5;
			int K6 = I6;
			int K7 = I7;
			//CALCULATE RETIREMENT DATE.
			int E = dateOfInterest.year + 65;
			// CALCULATE TIME SPENT IN THE FOLLOWING FUNCTIONS.
			float F = 0.35f;
			System.out.printf("%-28s", "YOU HAVE SLEPT");
			DateStruct scratchDate = new DateStruct(K6, K7, K5); //K5 is a temp year, K6 is month, K7 is day
			printStatisticRow(F, A8, scratchDate);
			K5 = scratchDate.year;
			K6 = scratchDate.month;
			K7 = scratchDate.day;

			F = 0.17f;
			System.out.printf("%-28s", "YOU HAVE EATEN");

			scratchDate = new DateStruct(K6, K7, K5);
			printStatisticRow(F, A8, scratchDate);
			K5 = scratchDate.year;
			K6 = scratchDate.month;
			K7 = scratchDate.day;

			F = 0.23f;
			if (K5 > 3) {
				if (K5 > 9) {
					System.out.printf("%-28s", "YOU HAVE WORKED/PLAYED");
				} else {
					System.out.printf("%-28s", "YOU HAVE PLAYED/STUDIED");
				}
			} else {
				System.out.printf("%-28s", "YOU HAVE PLAYED");
			}

			scratchDate = new DateStruct(K6, K7, K5);
			printStatisticRow(F, A8, scratchDate);
			K5 = scratchDate.year;
			K6 = scratchDate.month;
			K7 = scratchDate.day;

			if (K6 == 12) {
				K5 = K5 + 1;
				K6 = 0;
			}
			System.out.printf("%-28s%14s%14s%14s%n", "YOU HAVE RELAXED", K5, K6, K7);
			System.out.printf("%16s***  YOU MAY RETIRE IN %s ***%n", " ", E);
			System.out.printf("%n%n%n%n%n");
		} else {
			System.out.println("NOT PREPARED TO GIVE DAY OF WEEK PRIOR TO MDLXXXII.");
		}
	}


	private static void printStatisticRow(float F, int A8, DateStruct scratchDate) {
		int K1 = (int) (F * A8);
		int I5 = K1 / 365;
		K1 = K1 - (I5 * 365);
		int I6 = K1 / 30;
		int I7 = K1 - (I6 * 30);
		int K5 = scratchDate.year - I5;
		int K6 = scratchDate.month - I6;
		int K7 = scratchDate.day - I7;
		if (K7 < 0) {
			K7 = K7 + 30;
			K6 = K6 - 1;
		}
		if (K6 <= 0) {
			K6 = K6 + 12;
			K5 = K5 - 1;
		}
		//to return the updated values of K5, K6, K7 we send them through the scratchDate
		scratchDate.year = K5;
		scratchDate.month = K6;
		scratchDate.day = K7;
		System.out.printf("%14s%14s%14s%n", I5, I6, I7);
	}

	private static void printHeadersAndAge(int I5, int I6, int I7) {
		System.out.printf("%14s%14s%14s%14s%14s%n", " ", " ", "YEARS", "MONTHS", "DAYS");
		System.out.printf("%14s%14s%14s%14s%14s%n", " ", " ", "-----", "------", "----");
		System.out.printf("%-28s%14s%14s%14s%n", "YOUR AGE (IF BIRTHDATE)", I5, I6, I7);
	}

	private static void calculateAndPrintDayOfWeek(int i1, int a, DateStruct dateStruct, DateStruct dateOfInterest, int y3) {
		int b = (a - b(a) * 7) + 1;
		if (dateOfInterest.month > 2) {
			printDayOfWeek(dateStruct, dateOfInterest, b);
		} else {
			if (y3 == 0) {
				int aa = i1 - 1;
				int t1 = aa - a(aa) * 4;
				if (t1 == 0) {
					if (b != 0) {
						b = b - 1;
						printDayOfWeek(dateStruct, dateOfInterest, b);
					} else {
						b = 6;
						b = b - 1;
						printDayOfWeek(dateStruct, dateOfInterest, b);
					}
				}
			}
		}
	}

	/**
	 * PRINT THE DAY OF THE WEEK THE DATE FALLS ON.
	 */
	private static void printDayOfWeek(DateStruct dateStruct, DateStruct dateOfInterest, int b) {
		if (b == 0) {
			b = 7;
		}
		if ((dateStruct.year * 12 + dateStruct.month) * 31
				+ dateStruct.day
				<
				(dateOfInterest.year * 12
						+ dateOfInterest.month) * 31 + dateOfInterest.day) {
			System.out.printf("%s / %s / %s WILL BE A ", dateOfInterest.month, dateOfInterest.day, dateOfInterest.year);
		} else if ((dateStruct.year * 12 + dateStruct.month) * 31
				+ dateStruct.day == (dateOfInterest.year * 12 + dateOfInterest.month)
				* 31 + dateOfInterest.day) {
			System.out.printf("%s / %s / %s IS A ", dateOfInterest.month, dateOfInterest.day, dateOfInterest.year);
		} else {
			System.out.printf("%s / %s / %s WAS A ", dateOfInterest.month, dateOfInterest.day, dateOfInterest.year);
		}
		switch (b) {
			case 1:
				System.out.println("SUNDAY.");
				break;
			case 2:
				System.out.println("MONDAY.");
				break;
			case 3:
				System.out.println("TUESDAY.");
				break;
			case 4:
				System.out.println("WEDNESDAY.");
				break;
			case 5:
				System.out.println("THURSDAY.");
				break;
			case 6:
				if (dateOfInterest.day == 13) {
					System.out.println("FRIDAY THE THIRTEENTH---BEWARE!");
				} else {
					System.out.println("FRIDAY.");
				}
				break;
			case 7:
				System.out.println("SATURDAY.");
				break;
		}
	}

	private static int a(int a) {
		return a / 4;
	}

	private static int b(int a) {
		return a / 7;
	}


	private static void printIntro() {
		System.out.println("                                WEEKDAY");
		System.out.println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
		System.out.println("\n\n\n");
		System.out.println("WEEKDAY IS A COMPUTER DEMONSTRATION THAT");
		System.out.println("GIVES FACTS ABOUT A DATE OF INTEREST TO YOU.");
		System.out.println("\n");
	}

	/**
	 * Read user input for a date, do some validation and return a simple date structure
	 */
	private static DateStruct readDate(Scanner scanner) {
		boolean done = false;
		int mm = 0, dd = 0, yyyy = 0;
		while (!done) {
			String input = scanner.next();
			String[] tokens = input.split(",");
			if (tokens.length < 3) {
				System.out.println("DATE EXPECTED IN FORM: 3,24,1979 - RETRY INPUT LINE");
			} else {
				try {
					mm = Integer.parseInt(tokens[0]);
					dd = Integer.parseInt(tokens[1]);
					yyyy = Integer.parseInt(tokens[2]);
					done = true;
				} catch (NumberFormatException nfe) {
					System.out.println("NUMBER EXPECTED - RETRY INPUT LINE");
				}
			}
		}
		return new DateStruct(mm, dd, yyyy);
	}

	/**
	 * Convenience date structure to hold user date input
	 */
	private static class DateStruct {
		int month;
		int day;
		int year;

		public DateStruct(int month, int day, int year) {
			this.month = month;
			this.day = day;
			this.year = year;
		}
	}

}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


# `95_Weekday/javascript/weekday.js`

这段代码定义了一个名为 "WEEKDAY" 的函数，该函数将获取用户输入的 week 字，并将其打印到网页上的一个元素 "output" 中。

具体来说，这段代码包含两个函数：

1. `print()` 函数，该函数接收一个字符串参数 `str`，将其打印到网页上的元素 "output" 中。
2. `getUserInput()` 函数，该函数从用户那里获取一个字符串，并返回一个Promise对象，该对象在承诺的时间内（通常在页面加载后）获取用户输入的字符串。

"WEEKDAY" 函数通过调用 `print()` 函数来完成其任务，并将从 "getUserInput()` 函数中获取的字符串作为参数传递给 `print()` 函数，这样就可以将用户输入的字符串打印到 "output" 元素中。


```
// WEEKDAY
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

/**
 * Print given string to the end of the "output" element.
 * @param str
 */
function print(str) {
    document.getElementById("output").appendChild(document.createTextNode(str));
}

/**
 * Obtain user input
 * @returns {Promise<String>}
 */
```



该函数的作用是获取用户输入的字符串，并将其存储在变量 `input_str` 中。用户可以通过点击输入框并输入字符来触发该函数。

函数内部使用了一个 Promise 对象，该对象表示异步操作的最终结果。在 Promise 对象中，有一个 resolve 方法和一个 reject 方法。resolve 方法接受一个参数，而 reject 方法则接受一个错误对象。

在函数代码中，首先创建了一个 "输入" 元素的 INPUT 标签，并设置其 type 属性和 length 属性的值。这样，用户输入的字符串就会被存储在 input 元素中。

然后，将创建好的 input 元素添加到页面上，并将其 focus 属性设置为输入框上。这样，当用户点击输入框并输入字符时，input 元素就会获取到用户的输入。

当用户点击 input 元素时，会触发一个 keydown 事件，该事件会监听 input 元素上的所有 keydown 事件。在该事件中，有一个 if 语句，用于检查用户点击的是否是回车键(即 13)。如果是回车键，则执行一些操作并将用户输入的字符串打印到页面上，并使用 resolve 方法来存储返回的结果。


```
function input() {
    return new Promise(function (resolve) {
        const input_element = document.createElement("INPUT");

        print("? ");
        input_element.setAttribute("type", "text");
        input_element.setAttribute("length", "50");
        document.getElementById("output").appendChild(input_element);
        input_element.focus();
        input_element.addEventListener("keydown", function (event) {
            if (event.keyCode === 13) {
                const input_str = input_element.value;
                document.getElementById("output").removeChild(input_element);
                print(input_str);
                print("\n");
                resolve(input_str);
            }
        });
    });
}

```



这段代码定义了一个名为 `tab` 的函数，接受一个参数 `spaceCount`，代表要返回的字符串中空格的数量。函数实现是通过在字符串中添加 `spaceCount` 个空格来构成字符串。最后，函数返回这个字符串。

该函数的作用是用于在给定的字符串中返回由指定的空格数量组成的一个字符串。这个字符串可以用于各种用途，例如在网页中的排版、数据输入等等。


```
/**
 * Create a string consisting of the given number of spaces
 * @param spaceCount
 * @returns {string}
 */
function tab(spaceCount) {
    let str = "";
    while (spaceCount-- > 0)
        str += " ";
    return str;
}

const MONTHS_PER_YEAR = 12;
const DAYS_PER_COMMON_YEAR = 365;
const DAYS_PER_IDEALISED_MONTH = 30;
```

This is a JavaScript class called `DateComp`, which contains a method called `getDayOfWeek()` that returns the day of the week for a given date. Here's how the code works:

1. The method first calculates an offset based on the century part of the year.
2. It then calculates an offset based on the shortened two-digit year, taking into account whether the year is a leap year.
3. Finally, it combines the year and month offsets with the day and month of the week to form a US-style string of the date.

Note that the code assumes that the input date is a valid date object and that the `commonsenseYearMonth()` function is defined elsewhere.


```
const MAXIMUM_DAYS_PER_MONTH = 31;
// In a common (non-leap) year the day of the week for the first of each month moves by the following amounts.
const COMMON_YEAR_MONTH_OFFSET = [0, 3, 3, 6, 1, 4, 6, 2, 5, 0, 3, 5];

/**
 * Date representation.
 */
class DateStruct {
    #year;
    #month;
    #day;

    /**
     * Build a DateStruct
     * @param {number} year
     * @param {number} month
     * @param {number} day
     */
    constructor(year, month, day) {
        this.#year = year;
        this.#month = month;
        this.#day = day;
    }

    get year() {
        return this.#year;
    }

    get month() {
        return this.#month;
    }

    get day() {
        return this.#day;
    }

    /**
     * Determine if the date could be a Gregorian date.
     * Be aware the Gregorian calendar was not introduced in all places at once,
     * see https://en.wikipedia.org/wiki/Gregorian_calendar
     * @returns {boolean} true if date could be Gregorian; otherwise false.
     */
    isGregorianDate() {
        let result = false;
        if (this.#year > 1582) {
            result = true;
        } else if (this.#year === 1582) {
            if (this.#month > 10) {
                result = true;
            } else if (this.#month === 10 && this.#day >= 15) {
                result = true;
            }
        }
        return result;
    }

    /**
     * The following performs a hash on the day parts which guarantees that
     * 1. different days will return different numbers
     * 2. the numbers returned are ordered.
     * @returns {number}
     */
    getNormalisedDay() {
        return (this.year * MONTHS_PER_YEAR + this.month) * MAXIMUM_DAYS_PER_MONTH + this.day;
    }

    /**
     * Determine the day of the week.
     * This calculation returns a number between 1 and 7 where Sunday=1, Monday=2, ..., Saturday=7.
     * @returns {number} Value between 1 and 7 representing Sunday to Saturday.
     */
    getDayOfWeek() {
        // Calculate an offset based on the century part of the year.
        const centuriesSince1500 = Math.floor((this.year - 1500) / 100);
        let centuryOffset = centuriesSince1500 * 5 + (centuriesSince1500 + 3) / 4;
        centuryOffset = Math.floor(centuryOffset % 7);

        // Calculate an offset based on the shortened two digit year.
        // January 1st moves forward by approximately 1.25 days per year
        const yearInCentury = this.year % 100;
        const yearInCenturyOffsets = yearInCentury / 4 + yearInCentury;

        // combine offsets with day and month
        let dayOfWeek = centuryOffset + yearInCenturyOffsets + this.day + COMMON_YEAR_MONTH_OFFSET[this.month - 1];

        dayOfWeek = Math.floor(dayOfWeek % 7) + 1;
        if (this.month <= 2 && this.isLeapYear()) {
            dayOfWeek--;
        }
        if (dayOfWeek === 0) {
            dayOfWeek = 7;
        }
        return dayOfWeek;
    }

    /**
     * Determine if the given year is a leap year.
     * @returns {boolean}
     */
    isLeapYear() {
        if ((this.year % 4) !== 0) {
            return false;
        } else if ((this.year % 100) !== 0) {
            return true;
        } else if ((this.year % 400) !== 0) {
            return false;
        }
        return true;
    }

    /**
     * Returns a US formatted date, i.e. Month/Day/Year.
     * @returns {string}
     */
    toString() {
        return this.#month + "/" + this.#day + "/" + this.#year;
    }
}

```

This is a JavaScript class called `Duration` which represents a formatted duration. It has several static methods: `equals()`, `hashCode()`, `toString()` and `fromDays()`.

`equals()` method compares two `Duration` objects and returns a string with the same format as `Duration.toString()`.

`hashCode()` method generates an hash code for the `Duration` object by counting the number of unique bytes in its constructor.

`toString()` method returns a formatted string representation of the `Duration` object, including the years, months, and days in the specified format.

`fromDays()` method takes two arguments: the total number of days, and a factor representing the number of days per common year. It then calculates the duration based on this factor and returns a `Duration` object.

The `fromDays()` method is a naive calculation which assumes all months are 30 days. It is used to convert a given number of days to a `Duration` object.

The `ween()` method is a utility method that calculates the duration between two `DateStruct` objects. It takes two arguments, the first date, and the second date.

The `we Years, months, days` is the formula to calculate the duration between the two dates.


```
/**
 * Duration representation.
 * Note: this class only handles positive durations well
 */
class Duration {
    #years;
    #months;
    #days;

    /**
     * Build a Duration
     * @param {number} years
     * @param {number} months
     * @param {number} days
     */
    constructor(years, months, days) {
        this.#years = years;
        this.#months = months;
        this.#days = days;
        this.#fixRanges();
    }

    get years() {
        return this.#years;
    }

    get months() {
        return this.#months;
    }

    get days() {
        return this.#days;
    }

    clone() {
        return new Duration(this.#years, this.#months, this.#days);
    }

    /**
     * Adjust Duration by removing years, months and days from supplied Duration.
     * This is a naive calculation which assumes all months are 30 days.
     * @param {Duration} timeToRemove
     */
    remove(timeToRemove) {
        this.#years -= timeToRemove.years;
        this.#months -= timeToRemove.months;
        this.#days -= timeToRemove.days;
        this.#fixRanges();
    }

    /**
     * Move days and months into expected range.
     */
    #fixRanges() {
        if (this.#days < 0) {
            this.#days += DAYS_PER_IDEALISED_MONTH;
            this.#months--;
        }
        if (this.#months < 0) {
            this.#months += MONTHS_PER_YEAR;
            this.#years--;
        }
    }

    /**
     * Computes an approximation of the days covered by the duration.
     * The calculation assumes all years are 365 days, months are 30 days each,
     * and adds on an extra bit the more months that have passed.
     * @returns {number}
     */
    getApproximateDays() {
        return (
            (this.#years * DAYS_PER_COMMON_YEAR)
            + (this.#months * DAYS_PER_IDEALISED_MONTH)
            + this.#days
            + Math.floor(this.#months / 2)
        );
    }

    /**
     * Returns a formatted duration with tab separated values, i.e. Years\tMonths\tDays.
     * @returns {string}
     */
    toString() {
        return this.#years + "\t" + this.#months + "\t" + this.#days;
    }

    /**
     * Determine approximate Duration between two dates.
     * This is a naive calculation which assumes all months are 30 days.
     * @param {DateStruct} date1
     * @param {DateStruct} date2
     * @returns {Duration}
     */
    static between(date1, date2) {
        let years = date1.year - date2.year;
        let months = date1.month - date2.month;
        let days = date1.day - date2.day;
        return new Duration(years, months, days);
    }

    /**
     * Calculate years, months and days as factor of days.
     * This is a naive calculation which assumes all months are 30 days.
     * @param dayCount Total day to convert to a duration
     * @param factor   Factor to apply when calculating the duration
     * @returns {Duration}
     */
    static fromDays(dayCount, factor) {
        let totalDays = Math.floor(factor * dayCount);
        const years = Math.floor(totalDays / DAYS_PER_COMMON_YEAR);
        totalDays -= years * DAYS_PER_COMMON_YEAR;
        const months = Math.floor(totalDays / DAYS_PER_IDEALISED_MONTH);
        const days = totalDays - (months * DAYS_PER_IDEALISED_MONTH);
        return new Duration(years, months, days);
    }
}

```

This appears to be a program that calculates the estimated days between a user's birth date and a given date, and then prints out information about how much time the user has spent on different activities and items.

It uses a class called `DifferenceBetweenDates` that appears to keep track of the difference between the two dates, as well as an approximate number of days between them. It also has methods for getting the difference in days, the number of days between the two dates, and the number of days spent on different activities.

The program then goes through the estimated days between the user's birth date and the given date, and calculates how much time the user has spent on different activities. For example, it calculates how much time the user has spent sleeping, eating, working, studying, playing, and relaxing.

It also calculates how much time the user has spent relaxing, and prints out information about how much time they have spent on different activities and what their estimated retirement age


```
// Main control section
async function main() {
    /**
     * Reads a date, and extracts the date information.
     * This expects date parts to be comma separated, using US date ordering,
     * i.e. Month,Day,Year.
     * @returns {Promise<DateStruct>}
     */
    async function inputDate() {
        let dateString = await input();
        const month = parseInt(dateString);
        const day = parseInt(dateString.substr(dateString.indexOf(",") + 1));
        const year = parseInt(dateString.substr(dateString.lastIndexOf(",") + 1));
        return new DateStruct(year, month, day);
    }

    /**
     * Obtain text for the day of the week.
     * @param {DateStruct} date
     * @returns {string}
     */
    function getDayOfWeekText(date) {
        const dayOfWeek = date.getDayOfWeek();
        let dayOfWeekText = "";
        switch (dayOfWeek) {
            case 1:
                dayOfWeekText = "SUNDAY.";
                break;
            case 2:
                dayOfWeekText = "MONDAY.";
                break;
            case 3:
                dayOfWeekText = "TUESDAY.";
                break;
            case 4:
                dayOfWeekText = "WEDNESDAY.";
                break;
            case 5:
                dayOfWeekText = "THURSDAY.";
                break;
            case 6:
                if (date.day === 13) {
                    dayOfWeekText = "FRIDAY THE THIRTEENTH---BEWARE!";
                } else {
                    dayOfWeekText = "FRIDAY.";
                }
                break;
            case 7:
                dayOfWeekText = "SATURDAY.";
                break;
        }
        return dayOfWeekText;
    }

    print(tab(32) + "WEEKDAY\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("WEEKDAY IS A COMPUTER DEMONSTRATION THAT\n");
    print("GIVES FACTS ABOUT A DATE OF INTEREST TO YOU.\n");
    print("\n");
    print("ENTER TODAY'S DATE IN THE FORM: 3,24,1979  ");
    const today = await inputDate();
    // This program determines the day of the week
    //  for a date after 1582
    print("ENTER DAY OF BIRTH (OR OTHER DAY OF INTEREST)");
    const dateOfBirth = await inputDate();
    print("\n");
    // Test for date before current calendar.
    if (!dateOfBirth.isGregorianDate()) {
        print("NOT PREPARED TO GIVE DAY OF WEEK PRIOR TO X.XV.MDLXXXII.\n");
    } else {
        const normalisedToday = today.getNormalisedDay();
        const normalisedDob = dateOfBirth.getNormalisedDay();

        let dayOfWeekText = getDayOfWeekText(dateOfBirth);
        if (normalisedToday < normalisedDob) {
            print(dateOfBirth + " WILL BE A " + dayOfWeekText + "\n");
        } else if (normalisedToday === normalisedDob) {
            print(dateOfBirth + " IS A " + dayOfWeekText + "\n");
        } else {
            print(dateOfBirth + " WAS A " + dayOfWeekText + "\n");
        }

        if (normalisedToday !== normalisedDob) {
            print("\n");
            let differenceBetweenDates = Duration.between(today, dateOfBirth);
            if (differenceBetweenDates.years >= 0) {
                if (differenceBetweenDates.days === 0 && differenceBetweenDates.months === 0) {
                    print("***HAPPY BIRTHDAY***\n");
                }
                print("                        \tYEARS\tMONTHS\tDAYS\n");
                print("                        \t-----\t------\t----\n");
                print("YOUR AGE (IF BIRTHDATE) \t" + differenceBetweenDates + "\n");

                const approximateDaysBetween = differenceBetweenDates.getApproximateDays();
                const unaccountedTime = differenceBetweenDates.clone();

                // 35% sleeping
                const sleepTimeSpent = Duration.fromDays(approximateDaysBetween, 0.35);
                print("YOU HAVE SLEPT \t\t\t" + sleepTimeSpent + "\n");
                unaccountedTime.remove(sleepTimeSpent);

                // 17% eating
                const eatenTimeSpent = Duration.fromDays(approximateDaysBetween, 0.17);
                print("YOU HAVE EATEN \t\t\t" + eatenTimeSpent + "\n");
                unaccountedTime.remove(eatenTimeSpent);

                // 23% working, studying or playing
                const workPlayTimeSpent = Duration.fromDays(approximateDaysBetween, 0.23);
                if (unaccountedTime.years <= 3) {
                    print("YOU HAVE PLAYED \t\t" + workPlayTimeSpent + "\n");
                } else if (unaccountedTime.years <= 9) {
                    print("YOU HAVE PLAYED/STUDIED \t" + workPlayTimeSpent + "\n");
                } else {
                    print("YOU HAVE WORKED/PLAYED \t\t" + workPlayTimeSpent + "\n");
                }
                unaccountedTime.remove(workPlayTimeSpent);

                // Remaining time spent relaxing
                print("YOU HAVE RELAXED \t\t" + unaccountedTime + "\n");

                const retirementYear = dateOfBirth.year + 65;
                print("\n");
                print(tab(16) + "***  YOU MAY RETIRE IN " + retirementYear + " ***\n");
                print("\n");
            }
        }
    }
    print("\n");
    print("\n");
    print("\n");
    print("\n");
    print("\n");
}

```

这道题是一个简单的编程题目，没有给出具体的代码，只是让我们解释 main() 的作用。main() 是编程语言中的一个函数，它是程序的入口点，也就是程序开始运行的地方。当程序运行时，首先会进入 main() 函数，然后就可以开始执行程序的其他部分。main() 函数可以包含程序的任何代码，但通常情况下，它会包含程序的主要函数和必要的初始化。所以，main() 的作用是控制程序的执行流程，并确保程序能够正常运行。


```
main();

```