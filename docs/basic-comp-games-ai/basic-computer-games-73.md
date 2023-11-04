# BasicComputerGames源码解析 73

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


# `79_Slalom/python/slalom.py`

这段代码定义了一个名为medals的Python字典，其中包含三种金属(金、银、 bronze)的奖牌数量，均为0。

定义了一个名为ask的函数，该函数使用Python的random模块从用户那里获取问题，并输出问题，然后等待用户输入问题，并返回用户输入的问题 uppercase。

定义了一个名为ask_int的函数，该函数与ask类似，但等待用户输入的是一个数字，而不是一个字符串。函数使用ask函数获取问题，并尝试将问题转换为数字，如果转换成功，函数返回数字，否则返回-1。


```
from random import random

medals = {
    "gold": 0,
    "silver": 0,
    "bronze": 0,
}


def ask(question: str) -> str:
    print(question, end="? ")
    return input().upper()


def ask_int(question: str) -> int:
    reply = ask(question)
    return int(reply) if reply.isnumeric() else -1


```

这段代码是一个用于比赛前检查用户输入的函数，其目的是在用户输入“INS”、“MAX”或“RUN”时，分别显示不同的信息。具体来说，当用户输入“INS”时，函数会显示比赛的相关信息，包括这是1976年冬季奥运会 giant slalom 项目的比赛，美国队希望取得胜利等等。当用户输入“MAX”时，函数会显示每个门的最大速度。当用户输入“RUN”时，函数会显示比赛开始的时间。


```
def pre_run(gates, max_speeds) -> None:
    print('\nType "INS" for instructions')
    print('Type "MAX" for approximate maximum speeds')
    print('Type "RUN" for the beginning of the race')
    cmd = ask("Command--")
    while cmd != "RUN":
        if cmd == "INS":
            print("\n*** Slalom: This is the 1976 Winter Olypic Giant Slalom.  You are")
            print("            the American team's only hope for a gold medal.\n")
            print("     0 -- Type this if you want to see how long you've taken.")
            print("     1 -- Type this if you want to speed up a lot.")
            print("     2 -- Type this if you want to speed up a little.")
            print("     3 -- Type this if you want to speed up a teensy.")
            print("     4 -- Type this if you want to keep going the same speed.")
            print("     5 -- Type this if you want to check a teensy.")
            print("     6 -- Type this if you want to check a little.")
            print("     7 -- Type this if you want to check a lot.")
            print("     8 -- Type this if you want to cheat and try to skip a gate.\n")
            print(" The place to use these options is when the Computer asks:\n")
            print("Option?\n")
            print("                Good Luck!\n")
            cmd = ask("Command--")
        elif cmd == "MAX":
            print("Gate Max")
            print(" # M.P.H.")
            print("----------")
            for i in range(0, gates):
                print(f" {i + 1}  {max_speeds[i]}")
            cmd = ask("Command--")
        else:
            cmd = ask(f'"{cmd}" is an illegal command--Retry')


```

It looks like you've implemented a simple traffic simulation game. The game has several rules and the player must follow them in order to progress.

The game has different speeds that the player can use, and the player must choose the speed that is lower than the maximum speed that is allowed to be used at that time. The player will have to wait until the traffic slows down to a stop before they can use the next speed.

The game also has different levels of difficulty, and the player must earn gold, silver, or bronze depending on how many wins they have. These levels are determined by the maximum speed that can be used at each time.

Overall, the game looks like it would be a fun and simple way to experiment with traffic. The player would have to balance the speed they want to use with the maximum speed allowed, as well as wait for the right opportunities to use it.


```
def run(gates, lvl, max_speeds) -> None:
    global medals
    print("The starter counts down...5...4...3...2...1...Go!")
    time: float = 0
    speed = int(random() * (18 - 9) + 9)
    print("You're off")
    for i in range(0, gates):
        while True:
            print(f"\nHere comes gate #{i + 1}:")
            print(f" {int(speed)} M.P.H.")
            old_speed = speed
            opt = ask_int("Option")
            while opt < 1 or opt > 8:
                if opt == 0:
                    print(f"You've taken {int(time)} seconds.")
                else:
                    print("What?")
                opt = ask_int("Option")

            if opt == 8:
                print("***Cheat")
                if random() < 0.7:
                    print("An official caught you!")
                    print(f"You took {int(time + random())} seconds.")
                    return
                else:
                    print("You made it!")
                    time += 1.5
            else:
                match opt:
                    case 1:
                        speed += int(random() * (10 - 5) + 5)

                    case 2:
                        speed += int(random() * (5 - 3) + 3)

                    case 3:
                        speed += int(random() * (4 - 1) + 1)

                    case 5:
                        speed -= int(random() * (4 - 1) + 1)

                    case 6:
                        speed -= int(random() * (5 - 3) + 3)

                    case 7:
                        speed -= int(random() * (10 - 5) + 5)
                print(f" {int(speed)} M.P.H.")
                if speed > max_speeds[i]:
                    if random() < ((speed - max_speeds[i]) * 0.1) + 0.2:
                        print(
                            f"You went over the maximum speed and {'snagged a flag' if random() < .5 else 'wiped out'}!"
                        )
                        print(f"You took {int(time + random())} seconds")
                        return
                    else:
                        print("You went over the maximum speed and made it!")
                if speed > max_speeds[i] - 1:
                    print("Close one!")
            if speed < 7:
                print("Let's be realistic, ok? Let's go back and try again...")
                speed = old_speed
            else:
                time += max_speeds[i] - speed + 1
                if speed > max_speeds[i]:
                    time += 0.5
                break
    print(f"\nYou took {int(time + random())} seconds.")
    avg = time / gates
    if avg < 1.5 - (lvl * 0.1):
        print("Yout won a gold medal!")
        medals["gold"] += 1
    elif avg < 2.9 - (lvl * 0.1):
        print("You won a silver medal!")
        medals["silver"] += 1
    elif avg < 4.4 - (lvl * 0.01):
        print("You won a bronze medal!")
        medals["bronze"] += 1


```

This appears to be a program written in Python that simulates an skiing race. It first asks the user how many gates the course has, and then it displays the course's speed limits based on the number of gates.

It then enters a while loop that runs until the user decides to exit. Inside this loop, it displays the user's current speed based on their input and the speed limits of the course.

It also has a while loop that runs until the user decides to exit, and inside this loop, it asks the user if they want to play again. If the user enters "NO", the loop breaks.

Finally, it prints out the user's speed and the number of gold, silver, and bronze medals they have won.


```
def main() -> None:
    print("Slalom".rjust(39))
    print("Creative Computing Morristown, New Jersey\n\n\n".rjust(57))

    max_speeds = [
        14,
        18,
        26,
        29,
        18,
        25,
        28,
        32,
        29,
        20,
        29,
        29,
        25,
        21,
        26,
        29,
        20,
        21,
        20,
        18,
        26,
        25,
        33,
        31,
        22,
    ]

    while True:
        gates = ask_int("How many gates does this course have (1 to 25)")
        if gates < 1:
            print("Try again,")
        else:
            if gates > 25:
                print("25 is the limit.")
            break

    pre_run(gates, max_speeds)

    while True:
        lvl = ask_int("Rate yourself as a skier, (1=Worst, 3=Best)")
        if lvl < 1 or lvl > 3:
            print("The bounds are 1-3.")
        else:
            break

    while True:
        run(gates, lvl, max_speeds)
        while True:
            answer = ask("Do you want to play again?")
            if answer == "YES" or answer == "NO":
                break
            else:
                print('Please type "YES" or "NO"')
        if answer == "NO":
            break

    print("Thanks for the race")
    if medals["gold"] > 0:
        print(f"Gold medals: {medals['gold']}")
    if medals["silver"] > 0:
        print(f"Silver medals: {medals['silver']}")
    if medals["bronze"] > 0:
        print(f"Bronze medals: {medals['bronze']}")


```

这段代码是一个条件判断语句，它会判断当前脚本是否作为主程序运行。如果是主程序运行，那么程序会执行if语句中的代码块。这里的判断依据是，如果当前脚本作为主程序运行，那么执行if语句中的代码块就是程序的真正意义。

if语句中的代码块是 main() 函数，它会在判断条件为真（也就是当前脚本作为主程序运行）的情况下执行。main() 函数可能是定义在脚本内部，也可能是定义在脚本外部。在这段代码中，我们无法确定 main() 函数的具体实现，因为它可能是开发者事先定义好的函数，也可能是运行时动态生成的函数。


```
if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Slots

The slot machine or one-arm bandit is a mechanical device that will absorb coins just about as fast as you can feed it. After inserting a coin, you pull a handle that sets three independent reels spinning. If the reels stop with certain symbols appearing in the pay line, you get a certain payoff. The original slot machine, called the Liberty Bell, was invented in 1895 by Charles Fey in San Francisco. Fey refused to sell or lease the manufacturing rights, so H.S. Mills in Chicago built a similar, but much improved machine called the Operators Bell. This has survived nearly unchanged to today.

On the Operators Bell and other standard slot machines, there are 20 symbols on each wheel but they are not distributed evenly among the objects (cherries, bar, apples, etc.). Of the 8,000 passible combinations, the expected payoff (to the player) is 7,049 or $89.11 for every $100.00 put in, one of the lowest expected payoffs in all casino games.

In the program here, the payoff is considerably more liberal; indeed it appears to favor the player by 11% — i.e., an expected payoff of $111 for each $100 bet.

The program was originally written by Fred Mirabella and Bob Harper.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=149)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=164)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)

This C# implementation of slots was done using a [C# script](https://github.com/filipw/dotnet-script).

# Required
[.NET Core SDK (i.e., .NET 6.0)](https://dotnet.microsoft.com/en-us/download)

Install dotnet-script.  On the command line run:
```
dotnet tool install -g dotnet-script
```

# Run
```
dotnet script .\slots.csx
```


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `80_Slots/java/src/Slots.java`

This code appears to implement a number of string and number-related functions.

The `stringIsAnyValue` function takes a `text` string and a variable number of `values` strings and returns `true` if any of the `values` strings compares to the `text` using a case-insensitive comparison.

The `simulateTabs` function takes an integer `spaces` and returns a formatted string with the specified number of spaces between the `spaces` characters.

The `stringIsCaseSensitive` function is a wrapper for the `stringIsAnyValue` function and is used by the `compareTo` method to determine if the `text` string is considered case-sensitive.

The `compareTo` method takes two `text` strings and returns `true` if any of the `values` strings compares to either of the `text` strings in a case-insensitive comparison.

The `compareTo` method is used by the `Arrays.stream` method in the `values` parameter of the `stringIsAnyValue` function.

The `winningSymbol` function takes three integers `reel1`, `reel2`, and `reel3` and returns the index of the first reel that matches another reel.

The `randomSymbol` function takes no arguments and returns a random integer between 0 and 5.

The `analyzeReel` function appears to be the main function that is responsible for comparing the `text` to the `values` and returning the index of the first match. It takes a `text` string and a variable number of `values` strings and returns the index of the first match.


```
import java.util.Arrays;
import java.util.Scanner;

/**
 * Game of Slots
 * <p>
 * Based on the Basic game of Slots here
 * https://github.com/coding-horror/basic-computer-games/blob/main/80%20Slots/slots.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Slots {

    public static final String[] SLOT_SYMBOLS = {"BAR", "BELL", "ORANGE", "LEMON", "PLUM", "CHERRY"};

    public static final int NUMBER_SYMBOLS = SLOT_SYMBOLS.length;

    // Jackpot symbol (BAR)
    public static final int BAR_SYMBOL = 0;

    // Indicator that the current spin won nothing
    public static final int NO_WINNER = -1;

    // Used for keyboard input
    private final Scanner kbScanner;

    private enum GAME_STATE {
        START_GAME,
        ONE_SPIN,
        RESULTS,
        GAME_OVER
    }

    // Current game state
    private GAME_STATE gameState;

    // Different types of spin results
    private enum WINNINGS {
        JACKPOT(100),
        TOP_DOLLAR(10),
        DOUBLE_BAR(5),
        REGULAR(2),
        NO_WIN(0);

        private final int multiplier;

        WINNINGS(int mult) {
            multiplier = mult;
        }

        // No win returns the negative amount of net
        // otherwise calculate winnings based on
        // multiplier
        public int calculateWinnings(int bet) {

            if (multiplier == 0) {
                return -bet;
            } else {
                // Return original bet plus a multipler
                // of the win type
                return (multiplier * bet) + bet;
            }
        }
    }

    private int playerBalance;

    public Slots() {

        kbScanner = new Scanner(System.in);
        gameState = GAME_STATE.START_GAME;
    }

    /**
     * Main game loop
     */
    public void play() {

        int[] slotReel = new int[3];

        do {
            // Results of a single spin
            WINNINGS winnings;

            switch (gameState) {

                case START_GAME:
                    intro();
                    playerBalance = 0;
                    gameState = GAME_STATE.ONE_SPIN;
                    break;

                case ONE_SPIN:

                    int playerBet = displayTextAndGetNumber("YOUR BET? ");

                    slotReel[0] = randomSymbol();
                    slotReel[1] = randomSymbol();
                    slotReel[2] = randomSymbol();

                    // Store which symbol (if any) matches at least one other reel
                    int whichSymbolWon = winningSymbol(slotReel[0], slotReel[1], slotReel[2]);

                    // Display the three randomly drawn symbols
                    StringBuilder output = new StringBuilder();
                    for (int i = 0; i < 3; i++) {
                        if (i > 0) {
                            output.append(" ");
                        }
                        output.append(SLOT_SYMBOLS[slotReel[i]]);
                    }

                    System.out.println(output);

                    // Calculate results

                    if (whichSymbolWon == NO_WINNER) {
                        // No symbols match = nothing won
                        winnings = WINNINGS.NO_WIN;
                    } else if (slotReel[0] == slotReel[1] && slotReel[0] == slotReel[2]) {
                        // Top dollar, 3 matching symbols
                        winnings = WINNINGS.TOP_DOLLAR;
                        if (slotReel[0] == BAR_SYMBOL) {
                            // All 3 symbols are BAR. Jackpot!
                            winnings = WINNINGS.JACKPOT;
                        }
                    } else {
                        // At this point the remaining options are a regular win
                        // or a double, since the rest (including not winning) have already
                        // been checked above.
                        // Assume a regular win
                        winnings = WINNINGS.REGULAR;

                        // But if it was the BAR symbol that matched, its a double bar
                        if (slotReel[0] == BAR_SYMBOL) {
                            winnings = WINNINGS.DOUBLE_BAR;
                        }

                    }

                    // Update the players balance with the amount won or lost on this spin
                    playerBalance += winnings.calculateWinnings(playerBet);

                    System.out.println();

                    // Output what happened on this spin
                    switch (winnings) {
                        case NO_WIN:
                            System.out.println("YOU LOST.");
                            break;

                        case REGULAR:
                            System.out.println("DOUBLE!!");
                            System.out.println("YOU WON!");
                            break;

                        case DOUBLE_BAR:
                            System.out.println("*DOUBLE BAR*");
                            System.out.println("YOU WON!");
                            break;

                        case TOP_DOLLAR:
                            System.out.println();
                            System.out.println("**TOP DOLLAR**");
                            System.out.println("YOU WON!");
                            break;

                        case JACKPOT:
                            System.out.println();
                            System.out.println("***JACKPOT***");
                            System.out.println("YOU WON!");
                            break;

                    }

                    System.out.println("YOUR STANDINGS ARE $" + playerBalance);

                    // If player does not elect to play again, show results of session
                    if (!yesEntered(displayTextAndGetInput("AGAIN? "))) {
                        gameState = GAME_STATE.RESULTS;

                    }
                    break;

                case RESULTS:
                    if (playerBalance == 0) {
                        System.out.println("HEY, YOU BROKE EVEN.");
                    } else if (playerBalance > 0) {
                        System.out.println("COLLECT YOUR WINNINGS FROM THE H&M CASHIER.");
                    } else {
                        // Lost
                        System.out.println("PAY UP!  PLEASE LEAVE YOUR MONEY ON THE TERMINAL.");
                    }

                    gameState = GAME_STATE.GAME_OVER;
                    break;
            }
        } while (gameState != GAME_STATE.GAME_OVER);
    }

    private void intro() {
        System.out.println(simulateTabs(30) + "SLOTS");
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("YOU ARE IN THE H&M CASINO,IN FRONT OF ONE OF OUR");
        System.out.println("ONE-ARM BANDITS. BET FROM $1 TO $100.");
        System.out.println("TO PULL THE ARM, PUNCH THE RETURN KEY AFTER MAKING YOUR BET.");
    }

    /*
     * Print a message on the screen, then accept input from Keyboard.
     * Converts input to an Integer
     *
     * @param text message to be displayed on screen.
     * @return what was typed by the player.
     */
    private int displayTextAndGetNumber(String text) {
        return Integer.parseInt(displayTextAndGetInput(text));
    }

    /*
     * Print a message on the screen, then accept input from Keyboard.
     *
     * @param text message to be displayed on screen.
     * @return what was typed by the player.
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next();
    }

    /**
     * Checks whether player entered Y or YES to a question.
     *
     * @param text player string from kb
     * @return true of Y or YES was entered, otherwise false
     */
    private boolean yesEntered(String text) {
        return stringIsAnyValue(text, "Y", "YES");
    }

    /**
     * Check whether a string equals one of a variable number of values
     * Useful to check for Y or YES for example
     * Comparison is case insensitive.
     *
     * @param text   source string
     * @param values a range of values to compare against the source string
     * @return true if a comparison was found in one of the variable number of strings passed
     */
    private boolean stringIsAnyValue(String text, String... values) {

        return Arrays.stream(values).anyMatch(str -> str.equalsIgnoreCase(text));
    }

    /**
     * Simulate the old basic tab(xx) command which indented text by xx spaces.
     *
     * @param spaces number of spaces required
     * @return String with number of spaces
     */
    private String simulateTabs(int spaces) {
        char[] spacesTemp = new char[spaces];
        Arrays.fill(spacesTemp, ' ');
        return new String(spacesTemp);
    }

    /**
     * Find the symbol that won this round i.e. the first reel that matched another reel
     *
     * @param reel1 reel1 spin result
     * @param reel2 reel2 spin result
     * @param reel3 reel3 spin result
     * @return NO_WINNER if no reels match otherwise an int 0-2 to indicate the reel that matches another
     */
    private int winningSymbol(int reel1, int reel2, int reel3) {
        if (reel1 == reel2) {
            return 0;
        } else if (reel1 == reel3) {
            return 0;
        } else if (reel2 == reel3) {
            return 1;
        } else {
            return NO_WINNER;
        }
    }

    /**
     * Random symbol for a slot wheel
     *
     * @return number between 0-5
     */
    private int randomSymbol() {
        return (int) (Math.random() * NUMBER_SYMBOLS);
    }
}

```

# `80_Slots/java/src/SlotsGame.java`



这段代码是一个名为SlotsGame的Java类，其中包含了一个名为main的方法。当这个方法被调用时，它将创建一个Slots对象，并调用Slots的play()方法来使游戏开始玩。

Slots是一个游戏引擎，它可以模拟一个带有不同旋转位置的3个纺纱孔，玩家需要在游戏中在3个纺纱孔中选择一个来获得这个位置。

在这段代码中，我们首先定义了一个名为Slots的类，但我们并没有在在这个类中编写任何实际的游戏逻辑。我们创建了一个Slots对象并将其提供给main方法，然后在main方法中创建了一个Slots对象并调用其play()方法来开始游戏。所以，这段代码的作用是创建一个Slots对象并开始游戏。


```
public class SlotsGame {
    public static void main(String[] args) {
        Slots slots = new Slots();
        slots.play();
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


# `80_Slots/javascript/slots.js`

这段代码定义了两个函数，分别是`print()`和`input()`。

`print()`函数的作用是接收一个字符串参数，将其显示在页面上，并将其添加到页面上一个`<textarea>`元素中，该元素有一个`<br />`换行符。具体实现是通过在文档中创建一个新的文本节点，并将其内容设置为传入的字符串，然后将其添加到指定的元素中。

`input()`函数的作用是接收用户的输入，将其存储在变量`input_str`中，并在其后面添加一个换行符。它返回一个Promise对象，该对象在Promise对象上执行一个输入操作，该操作会获取用户输入的字符串，并在输入的字符串后面添加一个换行符。


```
// SLOTS
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

This is a program that allows a user to play a game of H2O. The user is given a choice of two numbers and three numbers, and they have to predict whether the numbers are high or low. If the user predicts correctly, they win. If not, they lose. The program uses a random number generator to generate the numbers. The user's standings are displayed at the end of the game.

It is important to note that this program may encourage users to predict high numbers, as the higher the numbers, the more money the user wins. This can be a problem in a game where the goal is to predict low numbers.

Additionally, it is also important to note that the program does not provide any way to ensure the randomness of the numbers. It is up to the developer to implement a more secure random number generator.


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var figures = [, "BAR", "BELL", "ORANGE", "LEMON", "PLUM", "CHERRY"];

// Main program
async function main()
{
    print(tab(30) + "SLOTS\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // Produced by Fred Mirabelle and Bob Harper on Jan 29, 1973
    // It simulates the slot machine.
    print("YOU ARE IN THE H&M CASINO,IN FRONT ON ONE OF OUR\n");
    print("ONE-ARM BANDITS. BET FROM $1 TO $100.\n");
    print("TO PULL THE ARM, PUNCH THE RETURN KEY AFTER MAKING YOUR BET.\n");
    p = 0;
    while (1) {
        while (1) {
            print("\n");
            print("YOUR BET");
            m = parseInt(await input());
            if (m > 100) {
                print("HOUSE LIMITS ARE $100\n");
            } else if (m < 1) {
                print("MINIMUM BET IS $1\n");
            } else {
                break;
            }
        }
        // Not implemented: GOSUB 1270 ten chimes
        print("\n");
        x = Math.floor(6 * Math.random() + 1);
        y = Math.floor(6 * Math.random() + 1);
        z = Math.floor(6 * Math.random() + 1);
        print("\n");
        // Not implemented: GOSUB 1310 seven chimes after figure x and y
        print(figures[x] + " " + figures[y] + " " + figures[z] + "\n");
        lost = false;
        if (x == y && y == z) {  // Three figure
            print("\n");
            if (z != 1) {
                print("**TOP DOLLAR**\n");
                p += ((10 * m) + m);
            } else {
                print("***JACKPOT***\n");
                p += ((100 * m) + m);
            }
            print("YOU WON!\n");
        } else if (x == y || y == z || x == z) {
            if (x == y)
                c = x;
            else
                c = z;
            if (c == 1) {
                print("\n");
                print("*DOUBLE BAR*\n");
                print("YOU WON\n");
                p += ((5 * m) + m);
            } else if (x != z) {
                print("\n");
                print("DOUBLE!!\n");
                print("YOU WON!\n");
                p += ((2 * m) + m);
            } else {
                lost = true;
            }
        } else {
            lost = true;
        }
        if (lost) {
            print("\n");
            print("YOU LOST.\n");
            p -= m;
        }
        print("YOUR STANDINGS ARE $" + p + "\n");
        print("AGAIN");
        str = await input();
        if (str.substr(0, 1) != "Y")
            break;
    }
    print("\n");
    if (p < 0) {
        print("PAY UP!  PLEASE LEAVE YOUR MONEY ON THE TERMINAL.\n");
    } else if (p == 0) {
        print("HEY, YOU BROKE EVEN.\n");
    } else {
        print("COLLECT YOUR WINNINGS FROM THE H&M CASHIER.\n");
    }
}

```

这是C++中的一个标准库函数，名为“main()”。这个函数是程序的入口点，程序从这里开始执行。在main()函数中，会首先输出“欢迎来到C++编程语言”，然后关闭输出流，接着就是程序体。

main()函数是C++命令行程序或Windows应用程序的起点。当程序运行时，首先会进入main()函数，然后会执行程序的代码。

对于C++程序，如果没有显式调用main()函数，那么程序会在屏幕上输出“欢迎来到C++编程语言”。对于Windows应用程序，程序运行时会自动调用main()函数，如果没有显式调用，那么程序同样会在屏幕上输出“欢迎来到C++编程语言”。


```
main();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)

This Perl script is a port of slots, which is the 80th entry in Basic
Computer Games.

I know nothing about slot machines, and my research into them says to me
that the payout tables can be fairly arbitrary. But I have taken the
liberty of deeming the BASIC program's refusal to pay on LEMON CHERRY
LEMON a bug, and made that case a double.

My justification for this is that at the point where the BASIC has
detected the double in the first and third reels it has already detected
that there is no double in the first and second reels. After the check
for a bar (and therefore a double bar) fails it goes back and checks for
a double on the second and third reels. But we know this check will
fail, since the check for a double on the first and second reels failed.
So if a loss was intended at this point, why not just call it a loss?

To restore the original behavior, comment out the entire line commented
'# Bug fix?' (about line 75) and uncomment the line with the trailing
comment '# Bug?' (about line 83).


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


# `80_Slots/python/slots.py`

这段代码是一个用于模拟赌博机（即赌博游戏）的程序。它有三个变量，用于存储硬币、接收 coins 的赌注和相应的支付结果。

程序的主要部分是一个 if 语句，该语句根据特定的符号在支付线上出现。如果三个转子（indepent reels）停止旋转并出现特定的符号，那么它会根据支付线的符号来确定支付结果，例如 3 个硬币或 1 个硬币。

程序还包含一个 Slots 标签，说明它来源于 1978 年由查尔斯·费依（Charles Fey）发明的赌博机。


```
########################################################
#
# Slots
#
# From Basic Computer Games (1978)
#
#    "The slot machine or one-arm bandit is a mechanical
#   device that will absorb coins just about as fast as
#   you can feed it. After inserting a coin, you pull a
#   handle that sets three indepent reels spining. If the
#   reels stop with certain symbols appearing in the pay
#   line, you get a certain payoff. The original slot
#   machine, called the Liberty bell, was invented in 1895
#   by Charles Fey in San Francisco. Fey refused to sell
#   or lease the manufacturing rights, so H.S. Mills in
```

这段代码是一个关于芝加哥制造的一种类似但和改进过的操作员贝尔(Operators Bell)机器的描述。这个机器已经有近20个符号，但是它们不是平均地分配在8000种不同的组合中。在芝加哥制造的操作员贝尔和其他标准的赌场机器上，这些符号分布得不那么平均。对于每种可能的组合，这个机器的期望回报(to the player)是7049或者每个$100.00的赌注的$89.11中最低的预期回报。这个程序中的回报非常慷慨，事实上它似乎偏袒玩家，比赌场游戏中的其他机器高11%。


```
#   Chicago built a similar, but much improved, machine
#   called the Operators Bell. This has survived nearly
#   unchanged to today.
#     On the operators Bell and other standard slot
#   machines, there are 20 symbols on each wheel but they
#   are not distributed evenly among the objects(cherries,
#   bar, apples, etc). Of the 8000 possible combinations,
#   the expected payoff(to the player) is 7049 or $89.11
#   for every $100.00 put in, one of the lowest expected
#   payoffs of all casino games.
#     In the program here, the payoff is considerably more
#   liberal; indeed it appears to favor the player by 11%
#   -- i.e., an expected payoff of $111 for each $100 bet."
#     The program was originally written by Fred Mirabelle
#   and Bob Harper
```

这段代码是一个交互式程序，它引导用户进行一个模拟赌场的游戏。这个程序使用了Python标准库中的`collections`库来统计在0到99之间点数的出现次数，使用了`random`库中的`choices`函数来随机选择一个数字（在0到99之间），然后使用了`typing`库中的`List`函数来创建一个需要包含数字的列表。

具体来说，这个程序首先会输出一个欢迎消息，告诉玩家他们正在一个模拟赌场，并且要押注。然后会要求玩家输入一个数字，这个数字范围是从1到100，然后程序会统计在0到99之间出现次数最多的数字，也就是这个数字在游戏中的点数。如果玩家想要退出程序，就可以按下回车键。程序的逻辑就是根据玩家的输入来选择一个点数，并在选择后输出这个点数作为答案。


```
#
########################################################

import sys
from collections import Counter
from random import choices
from typing import List


def initial_message() -> None:
    print(" " * 30 + "Slots")
    print(" " * 15 + "Creative Computing Morrison, New Jersey")
    print("\n" * 3)
    print("You are in the H&M Casino, in front of one of our")
    print("one-arm Bandits. Bet from $1 to $100.")
    print("To pull the arm, punch the return key after making your bet.")


```

这段代码是一个函数，名为 `input_betting()`，它接受一个整数类型的参数并返回该参数。

函数体首先输出一行字符，然后定义了一个变量 `b`，并将其初始化为负无穷(即 -1)。

接下来，函数体使用一个 while 循环，该循环会在 b 小于 1 或大于 100 时执行一次。在每次循环中，函数尝试从用户输入中获取一个整数类型的赌注，并将其存储在变量 `b` 中。如果用户输入的值不是整数类型，函数会将其设置为 -1，以便在后续循环中正确处理这个值。

如果 b 的值大于 100，函数会输出 "House limits are $100"，意思是这个赌场接受的最高赌注是 100 美元。如果 b 的值小于 1 美元，函数会输出 "Minimum bet is $1"，意思是这个赌场接受的最小赌注是 1 美元。

最后，函数使用 `beeping()` 函数来模拟一些模拟噪音，以使代码更加随机化。

该函数的作用是获取用户输入的赌注，然后根据赌注的值提示用户最大和最小赌注的限制，并返回用户输入的赌注值。


```
def input_betting() -> int:
    print("\n")
    b = -1
    while b < 1 or b > 100:
        try:
            b = int(input("Your bet:"))
        except ValueError:
            b = -1
        if b > 100:
            print("House limits are $100")
        elif b < 1:
            print("Minium bet is $1")
    beeping()
    return int(b)


```



这是一个Python程序，包含两个函数，一个是`beeping()`，另一个是`spin_wheels()`。这两个函数的作用是不同的。

`beeping()`函数的作用是产生一个铃声，并持续5秒钟。它的实现方式是在程序中循环5次，每次循环时输出一条横杠(|)，然后输出一个flush()函数，这个函数将缓冲区中的所有内容输出到屏幕上，并刷新缓冲区。这样可以产生一个持续5秒钟的铃声。

`spin_wheels()`函数的作用是从给定的水果列表中选择3个，并输出选择的3个水果的字母。它的实现方式是使用Python内置的`choices()`函数，这个函数可以从一个或多个元素中选择一个元素，这个函数的参数`possible_fruits`是一个列表，其中包含了给定的水果列表。这个函数的返回值是一个元组，其中包含了3个选择的水果的字母，顺序不确定。

代码中还包含了一个`print(*wheel)`语句，这个语句的作用是输出选择的3个水果的字母，而不是一个列表。这里使用了`print()`函数的`*`语法，它可以将参数传递给`print()`函数，并将每个参数打印出来。这里的`*`表示将`wheel`列表中的每个元素打印出来，而不是将`轮播`字符串打印出来。


```
def beeping() -> None:
    # Function to produce a beep sound.
    # In the original program is the subroutine at line 1270
    for _ in range(5):
        sys.stdout.write("\a")
        sys.stdout.flush()


def spin_wheels() -> List[str]:
    possible_fruits = ["Bar", "Bell", "Orange", "Lemon", "Plum", "Cherry"]
    wheel = choices(possible_fruits, k=3)

    print(*wheel)
    beeping()

    return wheel


```

这段代码定义了一个名为 `adjust_profits` 的函数，它接受一个名为 `wheel` 的列表、一个名为 `m` 的整数和一个名为 `proights` 的整数作为参数。

首先，函数通过创建一个包含所有独特元素的安全集合（set）来去除 wheel 中的重复元素。

然后，函数根据传入的参数对 wheel 中的元素进行分类。如果只有一个元素，那么这个元素就是 "Bar"，函数会输出一个 jackpot 奖金，奖金数额为 $100 \times m$ 加上利润数。

如果传入的元素有两个，那么函数会尝试找到一个出现次数为两个的元素。如果是 "Bar"，奖金数额为 $5 \times m$ 加上利润数；否则，奖金数额为 $2 \times m$ 加上利润数。

如果传入的元素有三个或更多，那么函数会输出 "You Lost"，奖金数为 0。

最后，函数返回调整后的利润数。


```
def adjust_profits(wheel: List[str], m: int, profits: int) -> int:
    # we remove the duplicates
    s = set(wheel)

    if len(s) == 1:
        # the three fruits are the same
        fruit = s.pop()

        if fruit == "Bar":
            print("\n***Jackpot***")
            profits = ((100 * m) + m) + profits
        else:
            print("\n**Top Dollar**")
            profits = ((10 * m) + m) + profits

        print("You Won!")
    elif len(s) == 2:
        # two fruits are equal
        c = Counter(wheel)
        # we get the fruit that appears two times
        fruit = sorted(c.items(), key=lambda x: x[1], reverse=True)[0][0]

        if fruit == "Bar":
            print("\n*Double Bar*")
            profits = ((5 * m) + m) + profits
        else:
            print("\nDouble!!")
            profits = ((2 * m) + m) + profits

        print("You Won!")
    else:
        # three different fruits
        print("\nYou Lost.")
        profits = profits - m

    return profits


```

这段代码定义了一个名为 `final_message` 的函数，它接受一个整数参数 `profits`。函数根据 `profits` 的值输出不同的消息。

函数首先检查 `profits` 是否小于 0，如果是，那么输出 "Pay up!  Please leave your money on the terminal"。如果 `profits` 不等于 0，那么它判断 `profits` 是否与 0 相等。如果是，那么输出 "Hey, You broke even."。如果 `profits` 大于 0，那么输出 "Collect your winnings from the H&M cashier。"。

在函数的主干部分，我们创建了一个名为 `profits` 的整数变量，一个名为 `keep_betting` 的布尔变量，一个用于保持当前轮赌博注的布尔变量。我们调用了一个名为 `initial_message` 的函数，这个函数没有做任何事情，然后一个名为 `spin_wheels` 的函数和一个名为 `adjust_profits` 的函数。

`spin_wheels` 函数的作用是让用户轮赌博注，这个函数没有在代码中实现，我们不知道它具体是如何工作的。`adjust_profits` 函数的作用是调整 `profits` 的值，使其大于或等于 0。我们可以看到，如果 `keep_betting` 是 `True`，那么这个函数被调用，否则不会被调用。


```
def final_message(profits: int) -> None:
    if profits < 0:
        print("Pay up!  Please leave your money on the terminal")
    elif profits == 0:
        print("Hey, You broke even.")
    else:
        print("Collect your winings from the H&M cashier.")


def main() -> None:
    profits = 0
    keep_betting = True

    initial_message()
    while keep_betting:
        m = input_betting()
        w = spin_wheels()
        profits = adjust_profits(w, m, profits)

        print(f"Your standings are ${profits}")
        answer = input("Again?")

        try:
            if answer[0].lower() != "y":
                keep_betting = False
        except IndexError:
            keep_betting = False

    final_message(profits)


```

这段代码是一个 Python 程序，其目的是定义一个函数并输出它的作用。这个函数是 Python 标准库中的一个命名函数，被称为 `__main__` 函数，它的作用是在程序作为主函数(Main Function)运行时被调用。

在这段注释中，程序描述了一个通用的规则，即在程序作为主函数运行时，将调用函数 `main()`。这个函数内部可以编写程序需要执行的所有代码。

在这个例子中，函数 `main()` 可能被视为一个自定义函数，其目的是在程序运行时执行一些自定义逻辑。由于 `main()` 函数没有明确的定义，因此它的实现可能会因程序的需求而异。


```
if __name__ == "__main__":
    main()

######################################################################
#
# Porting notes
#
#   The selections of the fruits(Bar, apples, lemon, etc.) are made
#   with equal probability, accordingly to random.choices documentation.
#   It could be added a weights list to the function and therefore
#   adjust the expected payoff
#
######################################################################

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Splat

SPLAT simulates a parachute jump in which you try to open your parachute at the last possible moment without going splat! You may select your own terminal velocity or let the computer do it for you. You many also select the acceleration due to gravity or, again, let the computer do it in which case you might wind up on any of eight planets (out to Neptune), the moon, or the sun.

The computer then tells you the height you’re jumping from and asks for the seconds of free fall. It then divides your free fall time into eight intervals and gives you progress reports on your way down. The computer also keeps track of all prior jumps in the array A and lets you know how you compared with previous successful jumps. If you want to recall information from previous runs, then you should store array A in a disk or take file and read it before each run.

John Yegge created this program while at the Oak Ridge Associated Universities.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=151)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=166)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `81_Splat/csharp/Program.cs`

This is a game written in C# that involves an object that simulates a platformer jumping and landing. The game has a simple input system where the player can either click to jump or press the spacebar to land. The object also has amath system that calculates the forces applied to the player's character based on the jump and landing values. The game also has a need to check if the ground is hit and if the player has reached the top or bottom of the screen.

The game has different animation states for different states of the jump. For example, when the player jumps, the character will have a different animation than when they land. Additionally, the game has a need to check if the player's character has reached the ground and if they are going to be launched.

Overall, this game is well-written and easy to understand. It is a fun and interactive experience for the players.


```
﻿using System.Collections;
using System.Text;

namespace Splat
{
    class Splat
    {
        private ArrayList DistanceLog = new ArrayList();

        private string[][] AccelerationData =
        {
            new string[] {"Fine. You're on Mercury. Acceleration={0} ft/sec/sec", "12.2"},
            new string[] {"All right.  You're on Venus. Acceleration={0} ft/sec/sec", "28.3"},
            new string[] {"Then you're on Earth. Acceleration={0} ft/sec/sec", "32.16"},
            new string[] {"Fine. You're on the Moon. Acceleration={0} ft/sec/sec", "5.15"},
            new string[] {"All right. You're on Mars. Acceleration={0} ft/sec/sec", "12.5"},
            new string[] {"Then you're on Jupiter. Acceleration={0} ft/sec/sec", "85.2"},
            new string[] {"Fine. You're on Saturn. Acceleration={0} ft/sec/sec", "37.6"},
            new string[] {"All right. You're on Uranus. Acceleration={0} ft/sec/sec", "33.8"},
            new string[] {"Then you're on Neptune. Acceleration={0} ft/sec/sec", "39.6"},
            new string[] {"Fine. You're on the Sun. Acceleration={0} ft/sec/sec", "896"}
        };

        private void DisplayIntro()
        {
            Console.WriteLine("");
            Console.WriteLine("SPLAT".PadLeft(23));
            Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            Console.WriteLine("");
            Console.WriteLine("Welcome to 'Splat' -- the game that simulates a parachute");
            Console.WriteLine("jump.  Try to open your chute at the last possible");
            Console.WriteLine("moment without going splat.");
            Console.WriteLine("");
        }

        private bool PromptYesNo(string Prompt)
        {
            bool Success = false;

            while (!Success)
            {
                Console.Write(Prompt);
                string LineInput = Console.ReadLine().Trim().ToLower();

                if (LineInput.Equals("yes"))
                    return true;
                else if (LineInput.Equals("no"))
                    return false;
                else
                    Console.WriteLine("Yes or No");
            }

            return false;
        }

        private void WriteRandomBadResult()
        {
           string[] BadResults = {"Requiescat in pace.","May the Angel of Heaven lead you into paradise.",
                "Rest in peace.","Son-of-a-gun.","#$%&&%!$","A kick in the pants is a boost if you're headed right.",
                "Hmmm. Should have picked a shorter time.","Mutter. Mutter. Mutter.","Pushing up daisies.",
                "Easy come, easy go."};

            Random rand = new Random();

            Console.WriteLine(BadResults[rand.Next(BadResults.Length)]);
        }

        private void WriteColumnOutput(double Column1, double Column2)
        {

            Console.WriteLine("{0,-11:N3}    {1,-17:N2}", Column1, Column2);

        }

        private void WriteColumnOutput(double Column1, string Column2)
        {

            Console.WriteLine("{0,-11:N3}    {1,-17}", Column1, Column2);

        }

        private void WriteColumnOutput(string Column1, string Column2)
        {

            Console.WriteLine("{0,-11}    {1,-17}", Column1, Column2);

        }

        private void WriteSuccessfulResults(double Distance)
        {
            // Add new result
            DistanceLog.Add(Distance);

            // Sort by distance
            DistanceLog.Sort();

            int ArrayLength = DistanceLog.Count;

            // If 1st, 2nd, or 3rd jump then write a special message
            if (ArrayLength <= 3)
            {
                Console.Write("Amazing!!! Not bad for your ");
                if (ArrayLength == 1)
                    Console.Write("1st ");
                else if (ArrayLength == 2)
                    Console.Write("2nd ");
                else
                    Console.Write("3rd ");
                Console.WriteLine("successful jump!!!");
            }
            // Otherwise write a message based on where this jump falls in the list
            else
            {
                int JumpPosition = DistanceLog.IndexOf(Distance);


                if (ArrayLength - JumpPosition <= .1 * ArrayLength)
                {
                    Console.WriteLine("Wow! That's some jumping. Of the {0} successful jumps", ArrayLength);
                    Console.WriteLine("before yours, only {0} opened their chutes lower than", (ArrayLength - JumpPosition));
                    Console.WriteLine("you did.");
                }
                else if (ArrayLength - JumpPosition <= .25 * ArrayLength)
                {
                    Console.WriteLine("Pretty good! {0} successful jumps preceded yours and only", ArrayLength - 1);
                    Console.WriteLine("{0} of them got lower than you did before their chutes", (ArrayLength - 1 - JumpPosition));
                    Console.WriteLine("opened.");
                }
                else if (ArrayLength - JumpPosition <= .5 * ArrayLength)
                {
                    Console.WriteLine("Not bad. There have been  {0} successful jumps before yours.", ArrayLength - 1);
                    Console.WriteLine("You were beaten out by {0} of them.", (ArrayLength - 1 - JumpPosition));
                }
                else if (ArrayLength - JumpPosition <= .75 * ArrayLength)
                {
                    Console.WriteLine("Conservative aren't you? You ranked only {0} in the", (ArrayLength - JumpPosition));
                    Console.WriteLine("{0} successful jumps before yours.", ArrayLength - 1);
                }
                else if (ArrayLength - JumpPosition <= .9 * ArrayLength)
                {
                    Console.WriteLine("Humph! Don't you have any sporting blood? There were");
                    Console.WriteLine("{0} successful jumps before yours and you came in {1} jumps", ArrayLength - 1, JumpPosition);
                    Console.WriteLine("better than the worst. Shape up!!!");
                }
                else
                {
                    Console.WriteLine("Hey! You pulled the rip cord much too soon. {0} successful", ArrayLength - 1);
                    Console.WriteLine("jumps before yours and you came in number {0}! Get with it!", (ArrayLength - JumpPosition));
                }
            }

        }

        private void PlayOneRound()
        {
            bool InputSuccess = false;
            Random rand = new Random();
            double Velocity = 0;
            double TerminalVelocity = 0;
            double Acceleration = 0;
            double AccelerationInput = 0;
            double Altitude = ((9001 * rand.NextDouble()) + 1000);
            double SecondsTimer = 0;
            double Distance = 0;
            bool TerminalVelocityReached = false;

            Console.WriteLine("");

            // Determine the terminal velocity (user or system)
            if (PromptYesNo("Select your own terminal velocity (yes or no)? "))
            {
                // Prompt user to enter the terminal velocity of their choice
                while (!InputSuccess)
                {
                    Console.Write("What terminal velocity (mi/hr)? ");
                    string Input = Console.ReadLine().Trim();
                    InputSuccess = double.TryParse(Input, out TerminalVelocity);
                    if (!InputSuccess)
                        Console.WriteLine("*** Please enter a valid number ***");
                 }
            }
            else
            {
                TerminalVelocity = rand.NextDouble() * 1000;
                Console.WriteLine("OK.  Terminal Velocity = {0:N0} mi/hr", (TerminalVelocity));
            }

            // Convert Terminal Velocity to ft/sec
            TerminalVelocity = TerminalVelocity * 5280 / 3600;

            // Not sure what this calculation is
            Velocity = TerminalVelocity + ((TerminalVelocity * rand.NextDouble()) / 20) - ((TerminalVelocity * rand.NextDouble()) / 20);

            // Determine acceleration due to gravity (user or system)
            if (PromptYesNo("Want to select acceleration due to gravity (yes or no)? "))
            {
                 // Prompt user to enter the acceleration of their choice
                InputSuccess = false;
                while (!InputSuccess)
                {
                    Console.Write("What acceleration (ft/sec/sec)? ");
                    string Input = Console.ReadLine().Trim();
                    InputSuccess = double.TryParse(Input, out AccelerationInput);
                    if (!InputSuccess)
                        Console.WriteLine("*** Please enter a valid number ***");
                 }
            }
            else
            {
                // Choose a random acceleration entry from the data array
                int Index = rand.Next(0, AccelerationData.Length);
                Double.TryParse(AccelerationData[Index][1], out AccelerationInput);

                // Display the corresponding planet this acceleration exists on and the value
                Console.WriteLine(AccelerationData[Index][0], AccelerationInput.ToString());
            }

            Acceleration = AccelerationInput + ((AccelerationInput * rand.NextDouble()) / 20) - ((AccelerationInput * rand.NextDouble()) / 20);

            Console.WriteLine("");
            Console.WriteLine("    Altitude         = {0:N0} ft", Altitude);
            Console.WriteLine("    Term. Velocity   = {0:N3} ft/sec +/-5%", TerminalVelocity);
            Console.WriteLine("    Acceleration     = {0:N2} ft/sec/sec +/-5%", AccelerationInput);
            Console.WriteLine("Set the timer for your freefall.");

            // Prompt for how many seconds the fall should be before opening the chute
            InputSuccess = false;
            while (!InputSuccess)
            {
                Console.Write("How many seconds? ");
                string Input = Console.ReadLine().Trim();
                InputSuccess = double.TryParse(Input, out SecondsTimer);
                if (!InputSuccess)
                    Console.WriteLine("*** Please enter a valid number ***");
            }

            // Begin the drop!
            Console.WriteLine("Here we go.");
            Console.WriteLine("");

            WriteColumnOutput("Time (sec)", "Dist to Fall (ft)");
            WriteColumnOutput("==========", "=================");

            // Loop through the number of seconds stepping by 8 intervals
            for (double i = 0; i < SecondsTimer; i+=(SecondsTimer/8))
            {
                if (i > (Velocity / Acceleration))
                {
                    // Terminal Velocity achieved.  Only print out the warning once.
                    if (TerminalVelocityReached == false)
                        Console.WriteLine("Terminal velocity reached at T plus {0:N4} seconds.", (Velocity / Acceleration));

                    TerminalVelocityReached = true;
                }

                // Calculate distance dependent upon whether terminal velocity has been reached
                if (TerminalVelocityReached)
                {
                    Distance = Altitude - ((Math.Pow(Velocity,2) / (2 * Acceleration)) + (Velocity * (i - (Velocity / Acceleration))));
                }
                else
                {
                    Distance = Altitude - ((Acceleration / 2) * Math.Pow(i,2));
                }

                // Was the ground hit?  If so, then SPLAT!
                if (Distance <= 0)
                {
                    if (TerminalVelocityReached)
                    {
                        WriteColumnOutput((Velocity / Acceleration) + ((Altitude - (Math.Pow(Velocity,2) / (2 * Acceleration))) / Velocity).ToString(), "SPLAT");
                    }
                    else
                    {
                        WriteColumnOutput(Math.Sqrt(2 * Altitude / Acceleration), "SPLAT");
                    }

                    WriteRandomBadResult();

                    Console.WriteLine("I'll give you another chance.");
                    break;
                }
                else
                {
                    WriteColumnOutput(i, Distance);
                }
            }

            // If the number of seconds of drop ended and we are still above ground then success!
            if (Distance > 0)
            {
                // We made it!  Chutes open!
                Console.WriteLine("Chute Open");

                // Store succesful jump and write out a fun message
                WriteSuccessfulResults(Distance);
            }

        }

        public void PlayTheGame()
        {
            bool ContinuePlay = false;

            DisplayIntro();

            do
            {
                PlayOneRound();

                ContinuePlay = PromptYesNo("Do you want to play again? ");
                if (!ContinuePlay)
                    ContinuePlay = PromptYesNo("Please? ");
            }
            while (ContinuePlay);

            Console.WriteLine("SSSSSSSSSS.");

        }
    }
    class Program
    {
        static void Main(string[] args)
        {

            new Splat().PlayTheGame();

        }
    }
}

```