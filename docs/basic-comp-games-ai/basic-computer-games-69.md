# BasicComputerGames源码解析 69

# `75_Roulette/java/oop/Bet.java`

In this game, the odds are assigned based on the number and color of the die. The odds are as follows:

1-18, odds of 1
19-36, odds of 1
Even, odds of 1

Column 1 (red), odds of 1
Column 2 (black), odds of 1
Column 3 (white), odds of 1

Single zero, odds of 35
Double zero, odds of 35

The odds are determined by the `w.isNumber()` method, which returns true if the `w` object is a valid number. If the `w` object is not a valid number, the method throws a `RuntimeException`.

The odds are then applied based on the `w.number()` property, which is the number of the `w` object. For example, if `w` is a die with a number between 1 and 6, the odds will be 1. If `w` is a die with a number greater than 6, the odds will be 3. If `w` is a die with a number equal to 6, the odds will be 2.

The odds are also applied based on the `w.color()` property. If `w` is a die with a color of red, the odds will be 1. If `w` is a die with a color of black, the odds will be 1. If `w` does not have a color, the odds will be 1.


```
/* A bet has a target (the code entered, which is 1-36, or special values for
 * the various groups, zero and double-zero), and an amount in dollars
 */

public class Bet {
    public int target;
    public int amount;

    /* bet on a target, of an amount */
    public Bet(int on, int of) {
        target = on; amount = of;
    }

    /* check if this is a valid bet - on a real target and of a valid amount */
    public boolean isValid() {
        return ((target > 0) && (target <= 50) &&
                (amount >= 5) && (amount <= 500));
    }

    /* utility to return either the odds amount in the case of a win, or zero for a loss */
    private int m(boolean isWon, int odds) {
        return isWon? odds: 0;
    }

    /* look at the wheel to see if this bet won.
     * returns 0 if it didn't, or the odds if it did
     */
    public int winsOn(Wheel w) {
        if (target < 37) {
            // A number bet 1-36 wins at odds of 35 if it is the exact number
            return m(w.isNumber() && (w.number() == target), 35);
        } else
            switch (target) {
            case 37:   // 1-12, odds of 2
                return m(w.isNumber() && (w.number() <= 12), 2);
            case 38:   // 13-24, odds of 2
                return m(w.isNumber() && (w.number() > 12) && (w.number() <= 24), 2);
            case 39:   // 25-36, odds of 2
                return m(w.isNumber() && (w.number() > 24), 2);
            case 40:   // Column 1, odds of 2
                return m(w.isNumber() && ((w.number() % 3) == 1), 2);
            case 41:   // Column 2, odds of 2
                return m(w.isNumber() && ((w.number() % 3) == 2), 2);
            case 42:   // Column 3, odds of 2
                return m(w.isNumber() && ((w.number() % 3) == 0), 2);
            case 43:   // 1-18, odds of 1
                return m(w.isNumber() && (w.number() <= 18), 1);
            case 44:   // 19-36, odds of 1
                return m(w.isNumber() && (w.number() > 18), 1);
            case 45:   // even, odds of 1
                return m(w.isNumber() && ((w.number() %2) == 0), 1);
            case 46:   // odd, odds of 1
                return m(w.isNumber() && ((w.number() %2) == 1), 1);
            case 47:   // red, odds of 1
                return m(w.isNumber() && (w.color() == Wheel.BLACK), 1);
            case 48:   // black, odds of 1
                return m(w.isNumber() && (w.color() == Wheel.RED), 1);
            case 49: // single zero, odds of 35
                return m(w.value().equals("0"), 35);
            case 50: // double zero, odds of 35
                return m(w.value().equals("00"), 35);
            }
        throw new RuntimeException("Program Error - invalid bet");
    }
}

```

# `75_Roulette/java/oop/Roulette.java`

This is a program written in Java that simulates a game of Monte Carlo在中奖的情况下，中奖的概率是多少。这个游戏的规则很简单，就是用户轮流下注，然后程序随机决定是否中奖，中奖后根据下注的金额计算出中奖概率，最终比较概率和用户下的赌注，如果概率和用户下的赌注相同，就算平局，否则就算输掉所有的赌注。

程序的主要类有两个，一个是Wheel类，负责处理游戏中的轮子，另一个是Bets类，负责处理用户下的赌注。Wheel类中包含一个Spin方法和一个Value方法，分别负责让轮子转速增加和计算轮子下注值，而Bets类中包含一个WinsOn方法，用于计算用户下注是否中奖，同时包含一个Amount方法，用于计算用户下的赌注金额，另外还有一个Beta方法，用于计算下注金额和赢得的金额的乘积。

在Wheel类中，还包含一个Roll数组，用于记录过去用户下的赌注，一个Write数组，用于将用户的下注记录下来，还有一个House数组，用于记录用户的赌注金额，一个Player数组，用于记录用户的分数，最后一个一个是Long类型的Spinner，用于计算概率。

这个程序中，玩家最多可以下注一次，如果多次下注，那么最后一次下注的赌注金额就是他的总赌注金额，然后程序会计算他全部的赌注，如果胜利，那么就会打印出所有下注的金额，并且告诉他胜利了还是失败了，否则就不会再打印任何东西，这样就可以让玩家知道他的游戏情况。


```
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.time.format.FormatStyle;

public class Roulette {
    public static void main(String args[]) throws Exception {
        Roulette r = new Roulette();
        r.play();
    }

    private BufferedReader reader;
    private PrintStream writer;

    private int house;      // how much money does the house have
    private int player;     // how much money does the player have
    private Wheel wheel = new Wheel();

    public Roulette() {
        reader = new BufferedReader(new InputStreamReader(System.in));
        writer = System.out;
        house = 100000;
        player = 1000;
    }

    // for a test / cheat mode -- set the random number generator to a known value
    private void setSeed(long l) {
        wheel.setSeed(l);
    }

    public void play() {
        try {
            intro();
            writer.println("WELCOME TO THE ROULETTE TABLE\n" +
                           "DO YOU WANT INSTRUCTIONS");
            String instr = reader.readLine();
            if (!instr.toUpperCase().startsWith("N"))
                instructions();

            while (betAndSpin()) { // returns true if the game is to continue
            }

            if (player <= 0) {
                // player ran out of money
                writer.println("THANKS FOR YOUR MONEY.\nI'LL USE IT TO BUY A SOLID GOLD ROULETTE WHEEL");
            } else {
                // player has money -- print them a check
                writer.println("TO WHOM SHALL I MAKE THE CHECK");

                String payee = reader.readLine();

                writer.println("-".repeat(72));
                tab(50); writer.println("CHECK NO. " + (new Random().nextInt(100) + 1));
                writer.println();
                tab(40); writer.println(LocalDate.now().format(DateTimeFormatter.ofLocalizedDate(FormatStyle.LONG)));
                writer.println("\n\nPAY TO THE ORDER OF-----" + payee + "-----$ " + player);
                writer.print("\n\n");
                tab(10); writer.println("THE MEMORY BANK OF NEW YORK\n");
                tab(40); writer.println("THE COMPUTER");
                tab(40); writer.println("----------X-----\n");
                writer.println("-".repeat(72));
                writer.println("COME BACK SOON!\n");
            }
        }
        catch (IOException e) {
            // this should not happen
            System.err.println("System error:\n" + e);
        }
    }

    /* Write the starting introduction */
    private void intro() throws IOException {
        tab(32); writer.println("ROULETTE");
        tab(15); writer.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n");
    }

    /* Display the game instructions */
    private void instructions() {
        String[] instLines = new String[] {
            "THIS IS THE BETTING LAYOUT",
            "  (*=RED)",
            "" ,
            " 1*    2     3*",
            " 4     5*    6 ",
            " 7*    8     9*",
            "10    11    12*",
            "---------------",
            "13    14*   15 ",
            "16*   17    18*",
            "19*   20    21*",
            "22    23*   24 ",
            "---------------",
            "25*   26    27*",
            "28    29    30*",
            "31    32*   33 ",
            "34*   35    36*",
            "---------------",
            "    00    0    ",
            "" ,
            "TYPES OF BETS",
            ""  ,
            "THE NUMBERS 1 TO 36 SIGNIFY A STRAIGHT BET",
            "ON THAT NUMBER.",
            "THESE PAY OFF 35:1",
            ""  ,
            "THE 2:1 BETS ARE:",
            " 37) 1-12     40) FIRST COLUMN",
            " 38) 13-24    41) SECOND COLUMN",
            " 39) 25-36    42) THIRD COLUMN",
            ""  ,
            "THE EVEN MONEY BETS ARE:",
            " 43) 1-18     46) ODD",
            " 44) 19-36    47) RED",
            " 45) EVEN     48) BLACK",
            "",
            " 49)0 AND 50)00 PAY OFF 35:1",
            " NOTE: 0 AND 00 DO NOT COUNT UNDER ANY",
            "       BETS EXCEPT THEIR OWN.",
            "",
            "WHEN I ASK FOR EACH BET, TYPE THE NUMBER",
            "AND THE AMOUNT, SEPARATED BY A COMMA.",
            "FOR EXAMPLE: TO BET $500 ON BLACK, TYPE 48,500",
            "WHEN I ASK FOR A BET.",
            "",
            "THE MINIMUM BET IS $5, THE MAXIMUM IS $500.",
            "" };
        writer.println(String.join("\n", instLines));
    }

    /* Take a set of bets from the player, then spin the wheel and work out the winnings *
     * This returns true if the game is to continue afterwards
     */
    private boolean betAndSpin() throws IOException {
        int betCount = 0;

        while (betCount == 0) {   // keep asking how many bets until we get a good answer
            try {
                writer.println("HOW MANY BETS");
                String howMany = reader.readLine();
                betCount = Integer.parseInt(howMany.strip());

                if ((betCount < 1) || (betCount > 100)) betCount = 0; // bad -- set zero and ask again
            }
            catch (NumberFormatException e) {
                // this happens if the input is not a number
                writer.println("INPUT ERROR");
            }
        }

        HashSet<Integer> betsMade = new HashSet<>(); // Bet targets already made, so we can spot repeats
        ArrayList<Bet> bets = new ArrayList<>();     // All the bets for this round

        while (bets.size() < betCount) {
            Bet bet = new Bet(0, 0);                 // an invalid bet to hold the place
            while (!bet.isValid()) {                 // keep asking until it is valid
                try {
                    writer.println("NUMBER " + (bets.size() + 1));
                    String fields[] = reader.readLine().split(",");
                    if (fields.length == 2) {
                        bet = new Bet(Integer.parseInt(fields[0].strip()),
                                      Integer.parseInt(fields[1].strip()));
                    }
                }
                catch (NumberFormatException e) {
                    writer.println("INPUT ERROR");
                }
            }

            // Check if there is already a bet on the same target
            if (betsMade.contains(bet.target)) {
                writer.println("YOU MADE THAT BET ONCE ALREADY,DUM-DUM");
            } else {
                betsMade.add(bet.target); // note this target has now been bet on
                bets.add(bet);
            }
        }

        writer.println("SPINNING\n\n");

        wheel.spin(); // this deliberately takes some random amount of time

        writer.println(wheel.value());

        // go through the bets, and evaluate each one
        int betNumber = 1;
        for (Bet b : bets) {
            int multiplier = b.winsOn(wheel);
            if (multiplier == 0) {
                // lost the amount of the bet
                writer.println("YOU LOSE " + b.amount + " DOLLARS ON BET " + betNumber);
                house += b.amount;
                player -= b.amount;
            } else {
                // won the amount of the bet, multiplied by the odds
                int winnings = b.amount * multiplier;
                writer.println("YOU WIN " + winnings + " DOLLARS ON BET " + betNumber);
                house -= winnings;
                player += winnings;
            }
            ++betNumber;
        }

        writer.println("\nTOTALS:\tME\tYOU\n\t" + house + "\t" + player);

        if (player <= 0) {
            writer.println("OOPS! YOU JUST SPENT YOUR LAST DOLLAR");
            return false;     // do not repeat since the player has no more money
        }
        if (house <= 0) {
            writer.println("YOU BROKE THE HOUSE!");
            player = 101000;  // can't win more than the house started with
            return false;     // do not repeat since the house has no more money
        }

        // player still has money, and the house still has money, so ask the player
        // if they want to continue
        writer.println("AGAIN");
        String doContinue = reader.readLine();

        // repeat if the answer was not "n" or "no"
        return (!doContinue.toUpperCase().startsWith("N"));
    }

    // utility to print n spaces for formatting
    private void tab(int n) {
        writer.print(" ".repeat(n));
    }
}

```

# `75_Roulette/java/oop/Wheel.java`

This is a simple Python class that simulates a physical slot machine. It has methods to spin the wheel, check the result of the spin, and output the result in the form of a string. The slot machine uses a random number generator to determine the outcome of each spin.

The `Wheel` class has several static methods for setting up the machine, including a `setSeed` method for setting the seed of the random number generator.

The `spin` method checks the result of the spin by waiting for 1 second. If it can't catch an `InterruptedException`, it will continue to wait until it stops.

The `value` method returns the string representation of the number, either `1-36`, `0`, or `00`.

The `zero` method checks if either 0 or 00 is hit.

The `isNumber` method checks if the number is either 0 or 00.

The `number` method rolls the number and returns it, either `0` or `+INT_MAX`.

The `color` method returns the color of the number, either `ZERO`, `BLACK`, or `RED`.

The class also has methods for checking if either ZERO or BLACK is the result of the spin.


```
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;

// The roulette wheel
public class Wheel {
    // List the numbers which are black
    private HashSet<Integer> black = new HashSet<>(Arrays.asList(new Integer[] { 1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36 }));

    private Random random = new Random();
    private int pocket = 38;

    public static final int ZERO=0;
    public static final int BLACK=1;
    public static final int RED=2;

    // Set up a wheel. You call "spin", and then can check the result.
    public Wheel() {
    }

    // Cheat / test mode
    void setSeed(long l) {
        random.setSeed(l);
    }

    // Spin the wheel onto a new random value.
    public void spin() {
        // keep spinning for a while
        do {
            try {
                // 1 second delay. Where it stops, nobody knows
                Thread.sleep(1000);
            }
            catch (InterruptedException e) {}

            pocket = random.nextInt(38) + 1;
        } while (random.nextInt(4) > 0); // keep spinning until it stops
    }

    // The string representation of the number; 1-36, 0, or 00
    public String value() {
        if (pocket == 37) return "0";
        else if (pocket == 38) return "00";
        else return String.valueOf(pocket);
    }

    // True if either 0 or 00 is hit
    public boolean zero() {
        return (pocket > 36);
    }

    // True if anything other than 0 or 00 is hit
    public boolean isNumber() {
        return (pocket < 37);
    }

    // The number rolled
    public int number() {
        if (zero()) return 0;
        else return pocket;
    }

    // Either ZERO, BLACK, or RED
    public int color() {
        if (zero()) return ZERO;
        else if (black.contains(pocket)) return BLACK;
        else return RED;
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


# `75_Roulette/javascript/roulette.js`

这两位代码主要实现了一个简单的交互式输入，允许用户向一个('#output')的元素中输入字符，并在输入框中高亮显示输入的字符。

具体来说，`print()`函数的作用是将一个字符串打印到网页的#output元素中，这里使用了物联网开发工具（如 Nanochess）将 Basic 语言代码转换为 JavaScript 的功能。

而`input()`函数则是实现了这个输入交互功能。它接受一个包含输入字段和输入字段描述的 HTML 元素，返回一个 Promise 对象，其中的 `resolve()` 方法用于接受用户输入的字符串并将其打印到网页上，而 `input()` 函数本身则是一个回调函数，它接受一个输入事件，在事件处理程序中获取用户输入的字符串，并将其存储在 `input_str` 变量中。

输入框中输入的字符会被存储在 `input_str` 变量中，然后将其打印到网页上。当用户点击输入框外的回车时，`input()` 函数会阻止事件的默认行为，弹出一个警告框告诉用户已经输入了字符。

最后，`print()` 函数的作用是打印字符串，它会在字符串的末尾添加一个换行符，并将其插入到输入框的值中。


```
// ROULETTE
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

这段代码定义了一个名为 "tab" 的函数，用于将一个数字数组中的数字转换为加号(+)的形式。

具体来说，代码中定义了一个名为 "str" 的字符串变量，用于存储当前数字数组中的所有数字。然后，代码使用 while 循环，将一个空格字符(即没有一个数字的位置)减少1，直到变量 space 的值等于0为止。在循环体内，代码将一个空格字符的值存储在 str 字符串中，并用一个空格字符替换掉 space 减少的数值。这样，最终就得到了一个所有数字的加号形式。

接下来，代码创建了两个名为 "ba" 和 "ca" 的数组，用于存储不同的数字。然后，代码又创建了两个名为 "ta" 和 "xxa" 的数组，用于存储数字数组中的两个数字。

最后，代码定义了一个名为 "aa" 的数组，用于存储数字数组中的所有数字。然后，代码使用 for 循环，遍历数字数组中的所有数字，并将它们添加到变量 "aa" 数组中。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var ba = [];
var ca = [];
var ta = [];
var xa = [];
var aa = [];

var numbers = [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36];

```

This is an interactive program that allows the user to spend their money as they see fit. They can add money to a variable `d` and then add it to a variable `p`. The program will then give the user options for how they can spend the money, such as buying a solid gold rouette wheel or investing it in a money tree. If the user chooses to spend it all, the program will tell them they have $p left. If they spend less than $p, the program will tell them they have $d left. If they spend more than $p, the program will tell them they have $p - $d left. The program will also print the name of the money bank where the money was spent.

The program also uses a random number generator to generate a random 4-digit number, which the user can choose as the amount they want to add to `d`.

It is important to note that the program is not a real-world application, and the use of eval and dynamic typing is not recommended.


```
// Main program
async function main()
{
    print(tab(32) + "ROULETTE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // Roulette
    // David Joslin
    print("WELCOME TO THE ROULETTE TABLE\n");
    print("\n");
    print("DO YOU WANT INSTRUCTIONS");
    str = await input();
    if (str.substr(0, 1) != "N") {
        print("\n");
        print("THIS IS THE BETTING LAYOUT\n");
        print("  (*=RED)\n");
        print("\n");
        print(" 1*    2     3*\n");
        print(" 4     5*    6 \n");
        print(" 7*    8     9*\n");
        print("10    11    12*\n");
        print("---------------\n");
        print("13    14*   15 \n");
        print("16*   17    18*\n");
        print("19*   20    21*\n");
        print("22    23*   24 \n");
        print("---------------\n");
        print("25*   26    27*\n");
        print("28    29    30*\n");
        print("31    32*   33 \n");
        print("34*   35    36*\n");
        print("---------------\n");
        print("    00    0    \n");
        print("\n");
        print("TYPES OF BETS\n");
        print("\n");
        print("THE NUMBERS 1 TO 36 SIGNIFY A STRAIGHT BET\n");
        print("ON THAT NUMBER.\n");
        print("THESE PAY OFF 35:1\n");
        print("\n");
        print("THE 2:1 BETS ARE:\n");
        print(" 37) 1-12     40) FIRST COLUMN\n");
        print(" 38) 13-24    41) SECOND COLUMN\n");
        print(" 39) 25-36    42) THIRD COLUMN\n");
        print("\n");
        print("THE EVEN MONEY BETS ARE:\n");
        print(" 43) 1-18     46) ODD\n");
        print(" 44) 19-36    47) RED\n");
        print(" 45) EVEN     48) BLACK\n");
        print("\n");
        print(" 49)0 AND 50)00 PAY OFF 35:1\n");
        print(" NOTE: 0 AND 00 DO NOT COUNT UNDER ANY\n");
        print("       BETS EXCEPT THEIR OWN.\n");
        print("\n");
        print("WHEN I ASK FOR EACH BET, TYPE THE NUMBER\n");
        print("AND THE AMOUNT, SEPARATED BY A COMMA.\n");
        print("FOR EXAMPLE: TO BET $500 ON BLACK, TYPE 48,500\n");
        print("WHEN I ASK FOR A BET.\n");
        print("\n");
        print("THE MINIMUM BET IS $5, THE MAXIMUM IS $500.\n");
        print("\n");
    }
    // Program begins here
    // Type of bet(number) odds
    for (i = 1; i <= 100; i++) {
        ba[i] = 0;
        ca[i] = 0;
        ta[i] = 0;
    }
    for (i = 1; i <= 38; i++)
        xa[i] = 0;
    p = 1000;
    d = 100000;
    while (1) {
        do {
            print("HOW MANY BETS");
            y = parseInt(await input());
        } while (y < 1) ;
        for (i = 1; i <= 50; i++) {
            aa[i] = 0;
        }
        for (c = 1; c <= y; c++) {
            while (1) {
                print("NUMBER " + c + " ");
                str = await input();
                x = parseInt(str);
                z = parseInt(str.substr(str.indexOf(",") + 1));
                ba[c] = z;
                ta[c] = x;
                if (x < 1 || x > 50)
                    continue;
                if (z < 1)
                    continue;
                if (z < 5 || z > 500)
                    continue;
                if (aa[x] != 0) {
                    print("YOU MADE THAT BET ONCE ALREADY,DUM-DUM\n");
                    continue;
                }
                aa[x] = 1;
                break;
            }
        }
        print("SPINNING\n");
        print("\n");
        print("\n");
        do {
            s = Math.floor(Math.random() * 100);
        } while (s == 0 || s > 38) ;
        xa[s]++;    // Not used
        if (s > 37) {
            print("00\n");
        } else if (s == 37) {
            print("0\n");
        } else {
            for (i1 = 1; i1 <= 18; i1++) {
                if (s == numbers[i1 - 1])
                    break;
            }
            if (i1 <= 18)
                print(s + " RED\n");
            else
                print(s + " BLACK\n");
        }
        print("\n");
        for (c = 1; c <= y; c++) {
            won = 0;
            switch (ta[c]) {
                case 37:    // 1-12 (37) 2:1
                    if (s > 12) {
                        won = -ba[c];
                    } else {
                        won = ba[c] * 2;
                    }
                    break;
                case 38:    // 13-24 (38) 2:1
                    if (s > 12 && s < 25) {
                        won = ba[c] * 2;
                    } else {
                        won = -ba[c];
                    }
                    break;
                case 39:    // 25-36 (39) 2:1
                    if (s > 24 && s < 37) {
                        won = ba[c] * 2;
                    } else {
                        won = -ba[c];
                    }
                    break;
                case 40:    // First column (40) 2:1
                    if (s < 37 && s % 3 == 1) {
                        won = ba[c] * 2;
                    } else {
                        won = -ba[c];
                    }
                    break;
                case 41:    // Second column (41) 2:1
                    if (s < 37 && s % 3 == 2) {
                        won = ba[c] * 2;
                    } else {
                        won = -ba[c];
                    }
                    break;
                case 42:    // Third column (42) 2:1
                    if (s < 37 && s % 3 == 0) {
                        won = ba[c] * 2;
                    } else {
                        won = -ba[c];
                    }
                    break;
                case 43:    // 1-18 (43) 1:1
                    if (s < 19) {
                        won = ba[c];
                    } else {
                        won = -ba[c];
                    }
                    break;
                case 44:    // 19-36 (44) 1:1
                    if (s > 18 && s < 37) {
                        won = ba[c];
                    } else {
                        won = -ba[c];
                    }
                    break;
                case 45:    // Even (45) 1:1
                    if (s < 37 && s % 2 == 0) {
                        won = ba[c];
                    } else {
                        won = -ba[c];
                    }
                    break;
                case 46:    // Odd (46) 1:1
                    if (s < 37 && s % 2 != 0) {
                        won = ba[c];
                    } else {
                        won = -ba[c];
                    }
                    break;
                case 47:    // Red (47) 1:1
                    for (i = 1; i <= 18; i++) {
                        if (s == numbers[i - 1])
                            break;
                    }
                    if (i <= 18) {
                        won = ba[c];
                    } else {
                        won = -ba[c];
                    }
                    break;
                case 48:    // Black (48) 1:1
                    for (i = 1; i <= 18; i++) {
                        if (s == numbers[i - 1])
                            break;
                    }
                    if (i <= 18 || s > 36) {
                        won = -ba[c];
                    } else {
                        won = ba[c];
                    }
                    break;
                default:    // 1-36,0,00 (1-36,49,50) 35:1
                    if (ta[c] < 49 && ta[c] == s
                        || ta[c] == 49 && s == 37
                        || ta[c] == 50 && s == 38) {
                        won = ba[c] * 35;
                    } else {
                        won = -ba[c];
                    }
                    break;
            }
            d -= won;
            p += won;
            if (won < 0) {
                print("YOU LOSE " + -won + " DOLLARS ON BET " + c + "\n");
            } else {
                print("YOU WIN " + won + " DOLLARS ON BET " + c + "\n");
            }
        }
        print("\n");
        print("TOTALS:\tME\tYOU\n");
        print(" \t" + d + "\t" + p + "\n");
        if (p <= 0) {
            print("OOPS! YOU JUST SPENT YOUR LAST DOLLAR!\n");
            break;
        } else if (d <= 0) {
            print("YOU BROKE THE HOUSE!\n");
            p = 101000;
        }
        print("AGAIN");
        str = await input();
        if (str.substr(0, 1) != "Y")
            break;
    }
    if (p < 1) {
        print("THANKS FOR YOUR MONEY.\n");
        print("I'LL USE IT TO BUY A SOLID GOLD ROULETTE WHEEL\n");
    } else {
        print("TO WHOM SHALL I MAKE THE CHECK");
        str = await input();
        print("\n");
        for (i = 1; i <= 72; i++)
            print("-");
        print("\n");
        print(tab(50) + "CHECK NO. " + Math.floor(Math.random() * 100) + "\n");
        print("\n");
        print(tab(40) + new Date().toDateString());
        print("\n");
        print("\n");
        print("PAY TO THE ORDER OF-----" + str + "-----$ " + p + "\n");
        print("\n");
        print("\n");
        print(tab(10) + "\tTHE MEMORY BANK OF NEW YORK\n");
        print("\n");
        print(tab(40) + "\tTHE COMPUTER\n");
        print(tab(40) + "----------X-----\n");
        print("\n");
        for (i = 1; i <= 72; i++)
            print("-");
        print("\n");
        print("COME BACK SOON!\n");
    }
    print("\n");
}

```

这是 C 语言中的一个程序，包含一个名为 "main" 的函数，但不含任何函数体。

在 C 语言中，程序入口点由程序员定义。通常情况下，程序入口点是包含在 "main" 函数中，但是也可以定义在其他地方。如果没有定义程序入口点，则系统将随机选择一个函数作为入口点。

因此，这个代码片段定义了一个名为 "main" 的函数，但没有定义任何参数、返回值或其他代码逻辑。它可能被视为一个空的函数，或者是一个声明，表示程序将搜索其他函数来执行操作。


```
main();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)

This conversion consists of three files in `75_Roulette/perl/`:

- `roulette.pl` is the port of the BASIC to Perl;
- `roulette-test.t` is a Perl test for correctness of display and payout;
- `make-roulette-test.pl` generates roulette-test.t from roulette.bas.

The ported version of the game numbers the slots from 0 rather than 1, and uses a dispatch table to figure out the payout.

The Perl test loads `roulette.pl` and verifies the Perl slot display and payout logic against the BASIC for all combinations of slots and bets. If any tests fail that fact will be noted at the end of the output.

The test code is generated by reading the BASIC, retaining only the slot display and payout logic (based on line numbers), and wrapping this in code that generates all combinations of bet and spin result. The result is run, and the result is captured and parsed to produce `roulette-test.t`. `make-roulette-test.pl` has some command-line options that may be of interest. `--help` will display the documentation.


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


# `75_Roulette/python/roulette.py`

这段代码的作用是定义了一个名为`print_instructions`的函数，该函数将输出一段文字说明。函数调用的结果不会产生任何实际输出。

具体来说，这段代码：

1. 导入了`random`模块，以便从那里导入随机数生成函数。
2. 从`datetime`模块中导入`date`函数，用于将日期转换为`datetime.date`类型。
3. 从`typing`模块中导入`List`和`Tuple`类型，以便在函数中使用它们的类型注释。
4. 创建了一个名为`RED_NUMBERS`的列表，用于存储预定义的颜色编号，这些编号将在函数中用于生成随机数。
5. 定义了一个名为`print_instructions`的函数，该函数没有参数。
6. 在函数内部，使用`print`函数输出了一个字符串，其中包含一些描述性的文本。这个字符串将告诉用户如何使用生成的随机数。
7. 使用`random.sample`函数从`RED_NUMBERS`列表中随机选择一个或多个元素，用于生成随机整数。
8. 使用`date.today.strftime`函数获取当前日期，并将其格式化为字符串。
9. 最后，将`date.today.strftime`函数的返回值与`RED_NUMBERS`列表中的随机整数作为元组添加到两个新的元组中，一个包含当前日期，另一个包含生成的随机整数。

这段代码将生成一个随机整数，然后将其与当前日期组合，以便用户可以了解它是如何生成的。


```
import random
from datetime import date
from typing import List, Tuple

global RED_NUMBERS
RED_NUMBERS = [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36]


def print_instructions() -> None:
    print(
        """
THIS IS THE BETTING LAYOUT
  (*=RED)

 1*    2     3*
 4     5*    6
 7*    8     9*
```

这是一行代码，它是一个文本文件，里面存储了形如 "NUMS\_BETS\_[N]" 的三元组，每个三元组代表着一个 nested 列表，包含以下形式：

```
[0] 表示这个 nested 列表的根节点，是一个数字；
[1] 表示这个 nested 列表的子节点，是一个数字；
[2] 表示这个 nested 列表的根节点的下标。
```

所以，整个文件主要作用是存储了一个由数字组成的 nested 列表，每个 nested 列表表示一个三元组，包含了该三元的 nested 结构。


```
10    11    12*
---------------
13    14*   15
16*   17    18*
19*   20    21*
22    23*   24
---------------
25*   26    27*
28    29    30*
31    32*   33
34*   35    36*
---------------
    00    0

TYPES OF BETS

```

这段代码是一个Excel VBA宏，它的目的是计算并显示一个随机数在1到36范围内的 straight bet（即在指定的数字中，随机数和目标数字的差值的绝对值的最大值）。

具体而言，这段代码执行以下操作：

1. 通过使用 `THE NUMBERS 1 TO 36 SIGNIFY A STRAIGHT BET` 这一函数，从1到36的范围内随机生成一个数字。
2. 通过 `THESE PAY OFF 35:1` 这一函数，计算随机数和目标数字35之间的 straight bet 所包含的数目，结果为35。
3. 通过 `THE 2:1 BETS ARE` 这一函数，计算随机数和目标数字1之间的 straight bet 所包含的数目，结果为4。
4. 通过 `THE EVEN MONEY BETS ARE` 这一函数，计算随机数和目标数字0（即目标数字）之间的 straight bet 所包含的数目，结果为48。
5. 通过 `PAY OFF 35:1` 这一函数，计算随机数在1到35之间的情形，即随机数和目标数字的差值的绝对值的最大值。结果为35。
6. 通过 `PAY OFF 35:1` 这一函数，计算随机数在35到36之间的情形，即随机数和目标数字的差值的绝对值的最大值。结果为1。
7. 最后，通过 ` 49:0 AND 50:00 PAY OFF 35:1` 这一函数，计算随机数在35到36之间的情形，即随机数和目标数字的差值的绝对值的最大值。结果为1。


```
THE NUMBERS 1 TO 36 SIGNIFY A STRAIGHT BET
ON THAT NUMBER.
THESE PAY OFF 35:1

THE 2:1 BETS ARE:
37) 1-12     40) FIRST COLUMN
38) 13-24    41) SECOND COLUMN
39) 25-36    42) THIRD COLUMN

THE EVEN MONEY BETS ARE:
43) 1-18     46) ODD
44) 19-36    47) RED
45) EVEN     48) BLACK

 49)0 AND 50)00 PAY OFF 35:1
```

这段代码是一个Python类的函数，它的目的是接受用户输入的投注数量，并在忠诚度和最大金额的限制内，返回每个投注的ID和值。

代码的最开始提示用户可以最多输入500个投注，每次投注必须用空格分隔，而且每个投注只能是一个数字。当用户输入投注数量后，函数会返回一个包含每个投注ID和值的列表，用于在赌博游戏中的跟踪和比较。

该函数的核心是输入验证和条件判断。对于每个投注，函数会先检查该投注ID是否存在于已有的投注中。如果不存在，函数会继续检查当前投注ID是否在500美元以内，如果是，就返回该ID和对应的值。如果存在或者不在500美元以内，函数会继续为该投注计算一个值，该值将根据当前的投注ID来确定。

函数还使用了一个变量min_bet来跟踪最小投注金额，当用户输入的投注数量不足5个时，min_bet将设置为5。当用户输入的投注数量超过5个但小于等于10个时，min_bet将设置为10。当用户输入的投注数量超过10个但小于等于20个时，min_bet将设置为15。当用户输入的投注数量超过20个但小于等于30个时，min_bet将设置为20。当用户输入的投注数量超过30个时，min_bet将设置为25。

最后，函数还使用了一个变量max_bet来跟踪最大投注金额，当用户输入的投注数量不足10个时，max_bet将设置为500。当用户输入的投注数量超过10个但小于等于20个时，max_bet将设置为1000。当用户输入的投注数量超过20个但小于等于30个时，max_bet将设置为1500。当用户输入的投注数量超过30个时，max_bet将设置为2000。


```
NOTE: 0 AND 00 DO NOT COUNT UNDER ANY
   BETS EXCEPT THEIR OWN.

WHEN I ASK FOR EACH BET, TYPE THE NUMBER
AND THE AMOUNT, SEPARATED BY A COMMA.
FOR EXAMPLE: TO BET $500 ON BLACK, TYPE 48,500
WHEN I ASK FOR A BET.

THE MINIMUM BET IS $5, THE MAXIMUM IS $500.

    """
    )


def query_bets() -> Tuple[List[int], List[int]]:
    """Queries the user to input their bets"""
    bet_count = -1
    while bet_count <= 0:
        try:
            bet_count = int(input("HOW MANY BETS? "))
        except Exception:
            ...

    bet_ids = [-1] * bet_count
    bet_values = [0] * bet_count

    for i in range(bet_count):
        while bet_ids[i] == -1:
            try:
                in_string = input("NUMBER " + str(i + 1) + "? ").split(",")
                id_, val = int(in_string[0]), int(in_string[1])

                # check other bet_IDs
                for j in range(i):
                    if id_ != -1 and bet_ids[j] == id_:
                        id_ = -1
                        print("YOU ALREADY MADE THAT BET ONCE, DUM-DUM")
                        break

                if id_ > 0 and id_ <= 50 and val >= 5 and val <= 500:
                    bet_ids[i] = id_
                    bet_values[i] = val
            except Exception:
                pass
    return bet_ids, bet_values


```

This function appears to compute the total net winnings based on the `bet_ids` list and the `result` integer. It uses a `get_modifier` function to apply a modifier to the winnings based on the `id` of the bet and the `num` of the bet.

The function takes in the `bet_ids` list and returns the total net winnings.


```
def bet_results(bet_ids: List[int], bet_values: List[int], result) -> int:
    """Computes the results, prints them, and returns the total net winnings"""
    total_winnings = 0

    def get_modifier(id_: int, num: int) -> int:
        if (
            (id_ == 37 and num <= 12)
            or (id_ == 38 and num > 12 and num <= 24)
            or (id_ == 39 and num > 24 and num < 37)
            or (id_ == 40 and num < 37 and num % 3 == 1)
            or (id_ == 41 and num < 37 and num % 3 == 2)
            or (id_ == 42 and num < 37 and num % 3 == 0)
        ):
            return 2
        elif (
            (id_ == 43 and num <= 18)
            or (id_ == 44 and num > 18 and num <= 36)
            or (id_ == 45 and num % 2 == 0)
            or (id_ == 46 and num % 2 == 1)
            or (id_ == 47 and num in RED_NUMBERS)
            or (id_ == 48 and num not in RED_NUMBERS)
        ):
            return 1
        elif id_ < 37 and id_ == num:
            return 35
        else:
            return -1

    for i in range(len(bet_ids)):
        winnings = bet_values[i] * get_modifier(bet_ids[i], result)
        total_winnings += winnings

        if winnings >= 0:
            print("YOU WIN " + str(winnings) + " DOLLARS ON BET " + str(i + 1))
        else:
            print("YOU LOSE " + str(winnings * -1) + " DOLLARS ON BET " + str(i + 1))

    return winnings


```

这段代码是一个函数名为`print_check`的Python函数，用于打印一张收据，收据上包含了收款人姓名、收款金额、收款日期以及收款人的账户信息。

具体来说，代码会首先向用户询问要向谁打印收据，然后会生成一个72字符的打印头，接着打印出收款人的姓名、收款金额、收款日期以及收款人的账户信息。最后，会输出一个包含银行名称和标识的横线，以及输出计算机的名称。


```
def print_check(amount: int) -> None:
    """Print a check of a given amount"""
    name = input("TO WHOM SHALL I MAKE THE CHECK? ")

    print("-" * 72)
    print()
    print(" " * 40 + "CHECK NO. " + str(random.randint(0, 100)))
    print(" " * 40 + str(date.today()))
    print()
    print("PAY TO THE ORDER OF -----" + name + "----- $" + str(amount))
    print()
    print(" " * 40 + "THE MEMORY BANK OF NEW YORK")
    print(" " * 40 + "THE COMPUTER")
    print(" " * 40 + "----------X-----")
    print("-" * 72)


```

This is a simple role-playing game where players can bet on the outcome of a virtual roulette wheel. Players must choose whether to bet on the number or the color, and they must spin the wheel until they either win or lose their bet. The game balance is maintained by the host, which periodically updates the game state and displays the winnings for all players.

The game starts with a budget of $1000 and a starting host balance of $100000. The player must spin the wheel 38 times before they have the opportunity to make a bet. Their bet is randomly assigned a number or a color, and the game will end when they either win or lose their bet.

If the player wins, they receive their bet amount plus their winnings, and the host reduces their balance. If the player loses, their balance is reduced by the amount of their bet. The host keep track of their balance and displays it to all players.

The game ends when a player tries to spin the wheel again. If the player has not made a bet within that time, they will be prompted to enter their bet.


```
def main() -> None:
    player_balance = 1000
    host_balance = 100000

    print(" " * 32 + "ROULETTE")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print()
    print()
    print()

    if string_to_bool(input("DO YOU WANT INSTRUCTIONS? ")):
        print_instructions()

    while True:
        bet_ids, bet_values = query_bets()

        print("SPINNING")
        print()
        print()

        val = random.randint(0, 38)
        if val == 38:
            print("0")
        elif val == 37:
            print("00")
        elif val in RED_NUMBERS:
            print(str(val) + " RED")
        else:
            print(str(val) + " BLACK")

        print()
        total_winnings = bet_results(bet_ids, bet_values, val)
        player_balance += total_winnings
        host_balance -= total_winnings

        print()
        print("TOTALS:\tME\t\tYOU")
        print("\t\t" + str(host_balance) + "\t" + str(player_balance))

        if player_balance <= 0:
            print("OOPS! YOU JUST SPENT YOUR LAST DOLLAR!")
            break
        elif host_balance <= 0:
            print("YOU BROKE THE HOUSE!")
            player_balance = 101000
            break
        if not string_to_bool(input("PLAY AGAIN? ")):
            break

    if player_balance <= 0:
        print("THANKS FOR YOUR MONEY")
        print("I'LL USE IT TO BUY A SOLID GOLD ROULETTE WHEEL")
    else:
        print_check(player_balance)
    print("COME BACK SOON!")


```

这段代码定义了一个名为 `string_to_bool` 的函数，它接收一个字符串参数 `string`，并返回一个布尔值。

这个函数的实现是通过判断给定的字符串是否属于特定的字串列表中的一个来实现的。这个列表包括 "yes"、"y"、"true"、"t" 和 "yes"。这些字符串在这里被作为特殊值，用于判断给定字符串是否与它们中的任何一个匹配。

在 `__main__` 函数中，如果你直接调用 `string_to_bool` 函数，它会返回一个布尔值，因为传递给它的字符串 "string" 是 "yes"。如果你直接调用 `string_to_bool` 函数并传入一个字符串参数，它将返回 `False`，因为任何字符串都可以被转换为布尔值。


```
def string_to_bool(string: str) -> bool:
    """Converts a string to a bool"""
    return string.lower() in ("yes", "y", "true", "t", "yes")


if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Russian Roulette

In this game, you are given by the computer a revolver loaded with one bullet and five empty chambers. You spin the chamber and pull the trigger by inputting a “1,” or, if you want to quit, input a “2.” You win if you play ten times and are still alive.

Tom Adametx wrote this program while a student at Curtis Jr. High School in Sudbury, Massachusetts.

⚠️ This game includes EXPLICT references to suicide, and should not be included in most distributions, especially considering the extreme simplicity of the program.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=141)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=153)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `76_Russian_Roulette/csharp/Program.cs`

This appears to be a simple game where the player has to shoot饥饿胡同等僵尸，每击杀一个僵尸就会获得一个随机的结果，可能是死亡，也可能是胜利，还可能是成功解救出一个受害者。游戏共13个选项，包括“在非洲草原上野餐”、“制作火柴盒”等等。

游戏使用了Console.WriteLine()方法来输出信息，使用Console.ReadKey()方法来接收玩家的输入。其中，游戏的输入选项和游戏结果选项都存储在一个名为GameResult的类中。在游戏的循环中，首先会获取一个随机数，然后根据玩家的输入来判断游戏结果，最后输出信息。

需要注意的是，游戏中的“spinChamber”方法并没有在代码中实现，因此不知道具体的方法和实现。此外，游戏中的“give up”选项也没有具体的功能实现，可能是只是一个显示信息的提示，具体的作用需要根据游戏的设计来决定。


```
﻿using System;

namespace RussianRoulette
{
    public class Program
    {
        public static void Main(string[] args)
        {
            PrintTitle();

            var includeRevolver = true;
            while (true)
            {
                PrintInstructions(includeRevolver);
                switch (PlayGame())
                {
                    case GameResult.Win:
                        includeRevolver = true;
                        break;
                    case GameResult.Chicken:
                    case GameResult.Dead:
                        includeRevolver = false;
                        break;
                }
            }
        }

        private static void PrintTitle()
        {
            Console.WriteLine("           Russian Roulette");
            Console.WriteLine("Creative Computing  Morristown, New Jersey");
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine("This is a game of >>>>>>>>>>Russian Roulette.");
        }

        private static void PrintInstructions(bool includeRevolver)
        {
            Console.WriteLine();
            if (includeRevolver)
            {
                Console.WriteLine("Here is a revolver.");
            }
            else
            {
                Console.WriteLine();
                Console.WriteLine();
                Console.WriteLine("...Next Victim...");
            }
            Console.WriteLine("Type '1' to spin chamber and pull trigger.");
            Console.WriteLine("Type '2' to give up.");
        }

        private static GameResult PlayGame()
        {
            var rnd = new Random();
            var round = 0;
            while (true)
            {
                round++;
                Console.Write("Go: ");
                var input = Console.ReadKey().KeyChar;
                Console.WriteLine();
                if (input != '2')
                {
                    // Random.Next will retun a value that is the same or greater than the minimum and
                    // less than the maximum.
                    // A revolver has 6 rounds.
                    if (rnd.Next(1, 7) == 6)
                    {
                        Console.WriteLine("     Bang!!!!!   You're dead!");
                        Console.WriteLine("Condolences will be sent to your relatives.");
                        return GameResult.Dead;
                    }
                    else
                    {
                        if (round > 10)
                        {
                            Console.WriteLine("You win!!!!!");
                            Console.WriteLine("Let someone else blow their brains out.");
                            return GameResult.Win;
                        }
                        else
                        {
                            Console.WriteLine("- CLICK -");
                            Console.WriteLine();
                        }
                    }
                }
                else
                {
                    Console.WriteLine("     CHICKEN!!!!!");
                    return GameResult.Chicken;
                }
            }
        }

        private enum GameResult
        {
            Win,
            Chicken,
            Dead
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `76_Russian_Roulette/java/src/RussianRoulette.java`

This is a Java program that appears to simulate the game of Russian roulette. It starts with an infinite loop that plays the game until it is explicitly stopped.

The program has a main method that initializes the game state to the initial state (GAME_STATE.GAME_START), which is the same state as the initial state in the original Russian roulette game.

The program also has a intro method that prints some information about the game, including its name and the俄语 name.

The game logic is implemented in a separate method called play, which is called from the main method.

这个程序的主要作用是让玩家继续练习RUSSIAN ROULETTE游戏，它会不断地接受玩家的输入，直到有人挑战或停止游戏。


```
import java.util.Arrays;
import java.util.Scanner;

/**
 * Game of Russian Roulette Paper
 * <p>
 * Based on the Basic game of Russian Roulette here
 * https://github.com/coding-horror/basic-computer-games/blob/main/76%20Russian%20Roulette/russianroulette.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */

public class RussianRoulette {

    public static final int BULLETS_IN_CHAMBER = 10;
    public static final double CHANCE_OF_GETTING_SHOT = .833333d;

    // Used for keyboard input
    private final Scanner kbScanner;

    private enum GAME_STATE {
        INIT,
        GAME_START,
        FIRE_BULLET,
        NEXT_VICTIM
    }

    // Current game state
    private GAME_STATE gameState;

    int bulletsShot;

    public RussianRoulette() {
        kbScanner = new Scanner(System.in);
        gameState = GAME_STATE.INIT;
    }

    /**
     * Main game loop
     */
    public void play() {

        do {
            switch (gameState) {

                case INIT:
                    intro();
                    gameState = GAME_STATE.GAME_START;

                    break;

                case GAME_START:
                    bulletsShot = 0;
                    System.out.println();
                    System.out.println("HERE IS A REVOLVER.");
                    System.out.println("TYPE '1' TO SPIN CHAMBER AND PULL TRIGGER.");
                    System.out.println("TYPE '2' TO GIVE UP.");
                    gameState = GAME_STATE.FIRE_BULLET;
                    break;

                case FIRE_BULLET:

                    int choice = displayTextAndGetNumber("GO ");

                    // Anything but selecting give up = have a shot
                    if (choice != 2) {
                        bulletsShot++;
                        if (Math.random() > CHANCE_OF_GETTING_SHOT) {
                            System.out.println("     BANG!!!!!   YOU'RE DEAD!");
                            System.out.println("CONDOLENCES WILL BE SENT TO YOUR RELATIVES.");
                            gameState = GAME_STATE.NEXT_VICTIM;
                        } else if (bulletsShot > BULLETS_IN_CHAMBER) {
                            System.out.println("YOU WIN!!!!!");
                            System.out.println("LET SOMEONE ELSE BLOW HIS BRAINS OUT.");
                            gameState = GAME_STATE.GAME_START;
                        } else {
                            // Phew player survived this round
                            System.out.println("- CLICK -");
                        }
                    } else {
                        // Player gave up
                        System.out.println("     CHICKEN!!!!!");
                        gameState = GAME_STATE.NEXT_VICTIM;

                    }
                    break;

                case NEXT_VICTIM:
                    System.out.println("...NEXT VICTIM...");
                    gameState = GAME_STATE.GAME_START;
            }
            // Infinite loop - based on original basic version
        } while (true);
    }

    private void intro() {
        System.out.println(addSpaces(28) + "RUSSIAN ROULETTE");
        System.out.println(addSpaces(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("THIS IS A GAME OF >>>>>>>>>>RUSSIAN ROULETTE.");
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
        return kbScanner.nextLine();
    }

    /**
     * Return a string of x spaces
     *
     * @param spaces number of spaces required
     * @return String with number of spaces
     */
    private String addSpaces(int spaces) {
        char[] spacesTemp = new char[spaces];
        Arrays.fill(spacesTemp, ' ');
        return new String(spacesTemp);
    }

    public static void main(String[] args) {

        RussianRoulette russianRoulette = new RussianRoulette();
        russianRoulette.play();
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)
