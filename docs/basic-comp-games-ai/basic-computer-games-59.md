# BasicComputerGames源码解析 59

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `60_Mastermind/javascript/mastermind.js`

该代码定义了一个名为"MASTERMIND"的函数，它将输入的字符串打印到网页上的一个元素中。"print"函数将输入的字符串打印到该元素的"output"子元素中。"input"函数是一个Promise，它等待用户输入字符串，然后将输入的字符串打印到该元素中，并等待用户按回车键以接收结果。该函数使用了一个input元素和一个存储用户输入的字符串的变量input_str。"input_element"变量将存储用户输入的元素，而"input_str"变量将存储用户输入的字符串。"print"函数和"input"函数都使用了document.getElementById()方法和setAttribute()方法来获取和设置元素的属性和关注点。"input_element.addEventListener("keydown", function (event) {
if (event.keyCode == 13) {
input_str = input_element.value;
document.getElementById("output").removeChild(input_element);
print(input_str);
print("\n");
resolve(input_str);
} });"将event监听器存储在一个函数中，以便在用户按下回车键时调用它。"print"函数和"input"函数都使用了appendChild()方法来将元素添加到文档中的输出元素中。


```
// MASTERMIND
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

这段代码定义了一个名为 "tab" 的函数，它会接收一个名为 "space" 的整数参数。

在函数内部，定义了一个字符串变量 "str"，并使用 while 循环来遍历 "space" 变量，每次循环将一个空格添加到 "str" 字符串的末尾。

在循环的外部，使用了一个前置的 "var" 关键字，来声明了九个整数变量 p9、c9、b、w、f、m，以及一个浮点数变量 "var" 类型的变量 p9。

该函数的作用是输出一个由九个空格组成的字符串，其中第九个空格是终止循环的条件。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var p9;
var c9;
var b;
var w;
var f;
var m;

```



这段代码定义了四个变量 qa、sa、ss 和 as，以及一个函数 initialize_qa 和 increment_qa。

initialize_qa 的作用是在程序启动时对 qa 数组进行初始化，将其所有元素都设置为零。

increment_qa 的作用是每次循环中，对 qa 数组中的元素进行加一操作，并将最终结果存回数组中。这个循环会一直持续到 qa[1] 大于等于 10 时停止，因为在此时，这个循环会溢出到溢出数组上，导致不稳定的结果。

initialize_qa 和 increment_qa 函数内部都是 while 循环，一直在进行操作，直到 qa 数组中的元素个数达到了 9，也就是 limited_p 函数的返回值。initialize_qa 函数中，对于 qa[1] <= 0 这个条件，会执行一次循环，将 qa 数组中从 1 到 9 的元素都设置为 1，这样就可以使得 limited_p 函数在第一次调用时能够正确返回 9。而 increment_qa 函数中，如果 qa[1] <= c9，则直接返回，否则执行循环操作，将 qa 数组中从 1 到余下的元素都加 1。


```
var qa;
var sa;
var ss;
var as;
var gs;
var hs;

function initialize_qa()
{
    for (s = 1; s <= p9; s++)
        qa[s] = 0;
}

function increment_qa()
{
    if (qa[1] <= 0) {
        // If zero, this is our firt increment: make all ones
        for (s = 1; s <= p9; s++)
            qa[s] = 1;
    } else {
        q = 1;
        while (1) {
            qa[q] = qa[q] + 1;
            if (qa[q] <= c9)
                return;
            qa[q] = 1;
            q++;
        }
    }
}

```



这两段代码都旨在完成一个 QA 题目的输入输出，具体解释如下：

1. `convert_qa()` 函数的作用是将 `QA` 数组中的每个元素，除去输入字符串中的数字，再将输入字符串中的每个元素向后移一位，得到一个新的 `QA` 数组。

2. `get_number()` 函数的作用是读取一个长度为 `p9` 的字符串 `input_str`，返回其中所有出现过的字符的个数 `count`。具体实现过程中，先初始化变量 `b`、`w`、`f` 和 `gs` 都为 0。然后从 `input_str` 的第一个字符开始遍历，如果当前字符既不是已出现过的字符，也不是已计算过的字符，则执行以下操作：

  - 如果当前字符是已出现过的字符，将其个数 `b` 加一，并将该字符后面的所有字符向后移一位，得到一个新的字符串 `new_str`。然后将 `new_str` 中的字符 `f` 赋值给 `f`，并将 `f` 的值加 2。
  
  - 如果当前字符既不是已出现过的字符，也不是已计算过的字符，则执行以下操作：

      - 遍历当前字符周围的每个字符，找到已出现过的字符。如果找到已出现过的字符，则输出该字符的个数 `i`。如果遍历过程中没有找到已出现过的字符，则将 `w` 加一，并将 `w` 加到 `gs` 中。
   
   - 在循环过程中，将 `gs` 中所有已出现过的字符的个数 `i` 加一，得到最终结果 `count = i + b`，即字符出现的总个数。

注意，以上解释中未涉及到 `String.fromCharCode()` 方法，是因为该方法在题目中并未提及需要实现字符转码。


```
function convert_qa()
{
    for (s = 1; s <= p9; s++) {
        as[s] = ls.substr(qa[s] - 1, 1);
    }
}

function get_number()
{
    b = 0;
    w = 0;
    f = 0;
    for (s = 1; s <= p9; s++) {
        if (gs[s] == as[s]) {
            b++;
            gs[s] = String.fromCharCode(f);
            as[s] = String.fromCharCode(f + 1);
            f += 2;
        } else {
            for (t = 1; t <= p9; t++) {
                if (gs[s] == as[t] && gs[t] != as[t]) {
                    w++;
                    as[t] = String.fromCharCode(f);
                    gs[s] = String.fromCharCode(f + 1);
                    f += 2;
                    break;
                }
            }
        }
    }
}

```



这段代码的主要作用是执行QA游戏的打印函数。

具体来说，它实现了以下功能：

1. 定义了一个名为convert_qa_hs的函数，该函数的作用是将输入的QA游戏字符串数组转换成骨架状字符串数组。转换的步骤如下：

- 遍历输入字符串数组的每个元素s，从该元素开始遍历到字符串数组的最后一个元素p9。
- 对于每个元素s，从字符串数组的第s个元素开始，遍历到该元素之前的所有元素，并将它们存储到变量hs中。

2. 定义了一个名为copy_hs的函数，该函数的作用是将输入的字符串数组复制到输出字符串数组中。

3. 定义了一个名为board_printout的函数，该函数的作用是打印输出QA游戏的骨架状字符串数组。

4. 在board_printout函数中，使用了嵌套循环来遍历输出字符串数组中的每个元素，并打印输出。


```
function convert_qa_hs()
{
    for (s = 1; s <= p9; s++) {
        hs[s] = ls.substr(qa[s] - 1, 1);
    }
}

function copy_hs()
{
    for (s = 1; s <= p9; s++) {
        gs[s] = hs[s];
    }
}

function board_printout()
{
    print("\n");
    print("BOARD\n");
    print("MOVE     GUESS          BLACK     WHITE\n");
    for (z = 1; z <= m - 1; z++) {
        str = " " + z + " ";
        while (str.length < 9)
            str += " ";
        str += ss[z];
        while (str.length < 25)
            str += " ";
        str += sa[z][1];
        while (str.length < 35)
            str += " ";
        str += sa[z][2];
        print(str + "\n");
    }
    print("\n");
}

```

这两函数是 Python 中常见的函数，其作用是分别用于功能和Score的实现。

1. `quit()` 函数的作用是在程序正常退出时执行，它通过调用 `print()` 函数输出了一段字符串，表明这是一个简单的程序退出，然后调用 `convert_qa()` 函数来处理一些未定义的 QA 问题，最后在退出前输出 "GOOD BYE" 字符串。

2. `show_score()` 函数的作用是打印Score，其通过调用 `print()` 函数来输出 "SCORE：" 字符串，然后调用 `show_points()` 函数来处理一些未定义的点数问题。


```
function quit()
{
    print("QUITTER!  MY COMBINATION WAS: ");
    convert_qa();
    for (x = 1; x <= p9; x++) {
        print(as[x]);
    }
    print("\n");
    print("GOOD BYE\n");
}

function show_score()
{
    print("SCORE:\n");
    show_points();
}

```

In this program, the AI attempts to guess the secret word by trying different combinations of moves. The AI has a limited number of attempts, and if it fails to guess the word after a certain number of attempts, it will assume the word is "SHERIFF-SMOOTH," which is the word that the AI is trying to guess.

The AI works by using a combination of human input and the program's own internal mechanisms to try and guess the word. First, the AI will ask the user to think of a combination of letters, and then it will use this combination to generate all possible combinations of letters in the word.

The AI then uses a loop to try different combinations of letters to generate possible moves. It will keep track of the number of guesses it has made and the number of letters that have been used in each move. If the AI makes a guess and the user confirms that it is a valid move by pressing "RETURN," the AI will update its internal state to reflect that it has made the guess.

Finally, the AI will continue to generate combinations of letters until it has run out of possible moves, at which point it will assume that the word it was trying to guess is "SHERIFF-SMOOTH" and will end its program.

The AI also includes a feature that allows it to show its score, which is based on the number of valid moves it has made. The AI will also show its final score at the end of the game.


```
function show_points()
{
    print("     COMPUTER " + c + "\n");
    print("     HUMAN    " + h + "\n");
    print("\n");
}

var color = ["BLACK", "WHITE", "RED", "GREEN",
             "ORANGE", "YELLOW", "PURPLE", "TAN"];

// Main program
async function main()
{
    print(tab(30) + "MASTERMIND\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    //
    //  MASTERMIND II
    //  STEVE NORTH
    //  CREATIVE COMPUTING
    //  PO BOX 789-M MORRISTOWN NEW JERSEY 07960
    //
    //
    while (1) {
        print("NUMBER OF COLORS");
        c9 = parseInt(await input());
        if (c9 <= 8)
            break;
        print("NO MORE THAN 8, PLEASE!\n");
    }
    print("NUMBER OF POSITIONS");
    p9 = parseInt(await input());
    print("NUMBER OF ROUNDS");
    r9 = parseInt(await input());
    p = Math.pow(c9, p9);
    print("TOTAL POSSIBILITIES = " + p + "\n");
    h = 0;
    c = 0;
    qa = [];
    sa = [];
    ss = [];
    as = [];
    gs = [];
    ia = [];
    hs = [];
    ls = "BWRGOYPT";
    print("\n");
    print("\n");
    print("COLOR    LETTER\n");
    print("=====    ======\n");
    for (x = 1; x <= c9; x++) {
        str = color[x - 1];
        while (str.length < 13)
            str += " ";
        str += ls.substr(x - 1, 1);
        print(str + "\n");
    }
    print("\n");
    for (r = 1; r <= r9; r++) {
        print("\n");
        print("ROUND NUMBER " + r + " ----\n");
        print("\n");
        print("GUESS MY COMBINATION.\n");
        print("\n");
        // Get a combination
        a = Math.floor(p * Math.random() + 1);
        initialize_qa();
        for (x = 1; x <= a; x++) {
            increment_qa();
        }
        for (m = 1; m <= 10; m++) {
            while (1) {
                print("MOVE # " + m + " GUESS ");
                str = await input();
                if (str == "BOARD") {
                    board_printout();
                } else if (str == "QUIT") {
                    quit();
                    return;
                } else if (str.length != p9) {
                    print("BAD NUMBER OF POSITIONS.\n");
                } else {
                    // Unpack str into gs(1-p9)
                    for (x = 1; x <= p9; x++) {
                        y = ls.indexOf(str.substr(x - 1, 1));
                        if (y < 0) {
                            print("'" + str.substr(x - 1, 1) + "' IS UNRECOGNIZED.\n");
                            break;
                        }
                        gs[x] = str.substr(x - 1, 1);
                    }
                    if (x > p9)
                        break;
                }
            }
            // Now we convert qa(1-p9) into as(1-p9) [ACTUAL GUESS]
            convert_qa();
            // And get number of blacks and white
            get_number();
            if (b == p9) {
                print("YOU GUESSED IT IN " + m + " MOVES!\n");
                break;
            }
            //tell human results
            print("YOU HAVE " + b + " BLACKS AND " + w + " WHITES.")
            // Save all this stuff for board printout later
            ss[m] = str;
            sa[m] = [];
            sa[m][1] = b;
            sa[m][2] = w;
        }
        if (m > 10) {
            print("YOU RAN OUT OF MOVES!  THAT'S ALL YOU GET!\n");
        }
        h += m;
        show_score();

        //
        // Now computer guesses
        //
        for (x = 1; x <= p; x++)
            ia[x] = 1;
        print("NOW I GUESS.  THINK OF A COMBINATION.\n");
        print("HIT RETURN WHEN READY:");
        str = await input();
        for (m = 1; m <= 10; m++) {
            initialize_qa();
            // Find a guess
            g = Math.floor(p * Math.random() + 1);
            if (ia[g] != 1) {
                for (x = g; x <= p; x++) {
                    if (ia[x] == 1)
                        break;
                }
                if (x > p) {
                    for (x = 1; x <= g; x++) {
                        if (ia[x] == 1)
                            break;
                    }
                    if (x > g) {
                        print("YOU HAVE GIVEN ME INCONSISTENT INFORMATION.\n");
                        print("TRY AGAIN, AND THIS TIME PLEASE BE MORE CAREFUL.\n");
                        for (x = 1; x <= p; x++)
                            ia[x] = 1;
                        print("NOW I GUESS.  THINK OF A COMBINATION.\n");
                        print("HIT RETURN WHEN READY:");
                        str = await input();
                        m = 0;
                        continue;
                    }
                }
                g = x;
            }
            // Now we convert guess #g into gs
            for (x = 1; x <= g; x++) {
                increment_qa();
            }
            convert_qa_hs();
            print("MY GUESS IS: ");
            for (x = 1; x <= p9; x++) {
                print(hs[x]);
            }
            print("  BLACKS, WHITES ");
            str = await input();
            b1 = parseInt(str);
            w1 = parseInt(str.substr(str.indexOf(",") + 1));
            if (b1 == p9) {
                print("I GOT IT IN " + m + " MOVES!\n");
                break;
            }
            initialize_qa();
            for (x = 1; x <= p; x++) {
                increment_qa();
                if (ia[x] != 0) {
                    copy_hs();
                    convert_qa();
                    get_number();
                    if (b1 != b || w1 != w)
                        ia[x] = 0;
                }
            }
        }
        if (m > 10) {
            print("I USED UP ALL MY MOVES!\n");
            print("I GUESS MY CPU I JUST HAVING AN OFF DAY.\n");
        }
        c += m;
        show_score();
    }
    print("GAME OVER\n");
    print("FINAL SCORE:\n");
    show_points();
}

```

这道题是一个简单的编程题目，我们需要解释 main() 函数的作用，而不会输出具体的源代码。

作为一个 C 语言程序，main() 函数是程序的入口点，也是程序执行的第一步。在 main() 函数中，程序会执行一系列预先设置好的指令，这些指令将会导致程序的运行。

通常情况下，程序在 main() 函数中需要做些什么，取决于程序的设计。 main() 函数可以用来做一些通用的操作，比如初始化计算机硬件，或者加载软件依赖关系等。

总之，main() 函数是程序的核心部分，它决定了程序的执行流程，以及程序如何与计算机交互。


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

This is pretty much a re-implementation of the BASIC, taking advantage
of Perl's array functionality and working directly with the alphabetic
color codes.


# `60_Mastermind/python/mastermind.py`

这段代码定义了一个名为setup_game的函数，用于设置游戏参数。函数返回四个整数参数，表示游戏中有多少种不同的颜色、每个位置可以放置多少种不同的颜色以及游戏需要进行多少轮。

这个函数的实现是模块化的，使用了Python标准库中的random和sys模块。从函数的实现来看，它似乎没有对游戏本身做出任何修改，只是一个简单的数据收集和返回函数。


```
import random
import sys
from typing import List, Union, Tuple


#  define some parameters for the game which should not be modified.
def setup_game() -> Tuple[int, int, int, int]:
    print("""
                                  MASTERMIND
                   CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY



    """)
    # get user inputs for game conditions
    num_colors: int = len(COLOR_LETTERS) + 1
    while num_colors > len(COLOR_LETTERS):
        num_colors = int(input("Number of colors (max 8): "))  # C9 in BASIC
    num_positions = int(input("Number of positions: "))  # P9 in BASIC
    num_rounds = int(input("Number of rounds: "))  # R9 in BASIC
    possibilities = num_colors**num_positions

    print(f"Number of possibilities {possibilities}")
    print("Color\tLetter")
    print("=====\t======")
    for element in range(0, num_colors):
        print(f"{COLORS[element]}\t{COLORS[element][0]}")
    return num_colors, num_positions, num_rounds, possibilities


```

这段代码定义了一个全局变量 `NUM_COLORS`，该变量是一个包含 9 个颜色的列表。还定义了一个全局变量 `COLOR_LETTERS`，该变量是一个字符串，表示位置从 0 到 8 的颜色字母。接着，定义了 `setup_game()` 函数，可能是用于设置游戏主变的函数。在主函数中，使用了 `NUM_COLORS` 和 `COLOR_LETTERS`，定义了 `NUM_POSITIONS`, `NUM_ROUNDS` 和 `POSSIBILITIES`，这些变量有助于在游戏中生成随机颜色。然后，定义了两个本地变量 `human_score` 和 `computer_score`，用于跟踪玩家和电脑的得分。程序还使用了一个 while 循环，让玩家玩 9 轮游戏。循环内部，分别调用 `human_turn()` 和 `computer_turn()` 函数，用于玩家和电脑选择颜色或生成随机颜色。最后，调用 `print_score()` 函数来打印最终得分，并使用 `is_final_score=True` 参数来输出 "WINNING" 或 "LOSING"。


```
# Global variables
COLORS = ["BLACK", "WHITE", "RED", "GREEN", "ORANGE", "YELLOW", "PURPLE", "TAN"]
COLOR_LETTERS = "BWRGOYPT"
NUM_COLORS, NUM_POSITIONS, NUM_ROUNDS, POSSIBILITIES = setup_game()
human_score = 0
computer_score = 0


def main() -> None:
    current_round = 1
    while current_round <= NUM_ROUNDS:
        print(f"Round number {current_round}")
        human_turn()
        computer_turn()
        current_round += 1
    print_score(is_final_score=True)
    sys.exit()


```

It looks like you have a Python program that generates a random secret combination of 33 numbers and allows the user to guess the combination. Here are some suggestions on how you could improve the program:

1. Function for printing the board:
You can create a function for printing the board that displays the number of guesses left, the guesses so far, and the number of wins for each combination. This would make it easier for the user to see their progress.
2. Function for getting invalid letters:
You can create a function that removes any invalid letters from the user's guess and returns a list of valid letters. This would help prevent users from using special characters or symbols that are not valid in code.
3. Function for comparing two positions:
You can create a function that compares two guesses and returns the difference between the two guesses. This would make it easier for the user to know if their guess was correct or not.
4. Function for returning the secret combination:
You can create a function that generates a random secret combination of 33 numbers and returns the secret combination. This would be a more comprehensive solution and would allow users to compare their results with others.
5. some numbers are not valid, you can make a function for getting the valid numbers of a secret combination.
6. Need a way to return the user out of the game, you can use a variable to keep track of the number of wins and the number of losses for each user.
7. Add some comments describing what each section of the code is doing.
8. It would be good to have a way to keep track of the score, you can use a variable to keep track of the wins and the number of moves for each user.

This is just a starting point, and you can certainly make more improvements to the program based on your specific requirements.


```
def human_turn() -> None:
    global human_score
    num_moves = 1
    guesses: List[List[Union[str, int]]] = []
    print("Guess my combination ...")
    secret_combination = int(POSSIBILITIES * random.random())
    answer = possibility_to_color_code(secret_combination)
    while True:
        print(f"Move # {num_moves} Guess : ")
        user_command = input("Guess ")
        if user_command == "BOARD":
            print_board(guesses)  # 2000
        elif user_command == "QUIT":  # 2500
            print(f"QUITTER! MY COMBINATION WAS: {answer}")
            print("GOOD BYE")
            quit()
        elif len(user_command) != NUM_POSITIONS:  # 410
            print("BAD NUMBER OF POSITIONS")
        else:
            invalid_letters = get_invalid_letters(user_command)
            if invalid_letters > "":
                print(f"INVALID GUESS: {invalid_letters}")
            else:
                guess_results = compare_two_positions(user_command, answer)
                if guess_results[1] == NUM_POSITIONS:  # correct guess
                    print(f"You guessed it in {num_moves} moves!")
                    human_score = human_score + num_moves
                    print_score()
                    return  # from human turn, triumphant
                else:
                    print(
                        "You have {} blacks and {} whites".format(
                            guess_results[1], guess_results[2]
                        )
                    )
                    guesses.append(guess_results)
                    num_moves += 1

        if num_moves > 10:  # RAN OUT OF MOVES
            print("YOU RAN OUT OF MOVES! THAT'S ALL YOU GET!")
            print(f"THE ACTUAL COMBINATION WAS: {answer}")
            human_score = human_score + num_moves
            print_score()
            return  # from human turn, defeated


```

It looks like the code is meant to play a game of Connect-the-Dots where the user has to guess a code made up of colored dots. The code starts by displaying all possible combinations of dots and then when the user makes a guess it compares the computer's guess to some of the possible answers and if the two are not the same then the computer eliminates one of the possibilities. The code also keeps track of the number of moves made by the computer and if the user's guess is correct, the computer's score is updated and the user is given a hint. If the user makes a guess that is not a correct answer and the computer's score has not yet been updated, the user is given an opportunity to continue guessing.


```
def computer_turn() -> None:
    global computer_score
    while True:
        all_possibilities = [1] * POSSIBILITIES
        num_moves = 1
        print("NOW I GUESS. THINK OF A COMBINATION.")
        input("HIT RETURN WHEN READY: ")
        while True:
            possible_guess = find_first_solution_of(all_possibilities)
            if possible_guess < 0:  # no solutions left :(
                print("YOU HAVE GIVEN ME INCONSISTENT INFORMATION.")
                print("TRY AGAIN, AND THIS TIME PLEASE BE MORE CAREFUL.")
                break  # out of inner while loop, restart computer turn

            computer_guess = possibility_to_color_code(possible_guess)
            print(f"My guess is: {computer_guess}")
            blacks_str, whites_str = input(
                "ENTER BLACKS, WHITES (e.g. 1,2): "
            ).split(",")
            blacks = int(blacks_str)
            whites = int(whites_str)
            if blacks == NUM_POSITIONS:  # Correct guess
                print(f"I GOT IT IN {num_moves} MOVES")
                computer_score = computer_score + num_moves
                print_score()
                return  # from computer turn

            # computer guessed wrong, deduce which solutions to eliminate.
            for i in range(0, POSSIBILITIES):
                if all_possibilities[i] == 0:  # already ruled out
                    continue
                possible_answer = possibility_to_color_code(i)
                comparison = compare_two_positions(
                    possible_answer, computer_guess
                )
                if (blacks != comparison[1]) or (whites != comparison[2]):
                    all_possibilities[i] = 0

            if num_moves == 10:
                print("I USED UP ALL MY MOVES!")
                print("I GUESS MY CPU IS JUST HAVING AN OFF DAY.")
                computer_score = computer_score + num_moves
                print_score()
                return  # from computer turn, defeated.
            num_moves += 1


```



The code defines two functions, `find_first_solution_of` and `get_invalid_letters`.

`find_first_solution_of` takes a list of integers `all_possibilities` and returns the first non-zero integer that can be used to solve the puzzle. The function uses the因子图 algorithm to generate all possible solutions and returns the first solution that meets the condition.

`get_invalid_letters` takes a string `user_command` and returns a list of valid colors for the game configuration. The function uses a list comprehension to extract only the valid colors from the user command.


```
def find_first_solution_of(all_possibilities: List[int]) -> int:
    """Scan through all_possibilities for first remaining non-zero marker,
    starting from some random position and wrapping around if needed.
    If not found return -1."""
    start = int(POSSIBILITIES * random.random())
    for i in range(0, POSSIBILITIES):
        solution = (i + start) % POSSIBILITIES
        if all_possibilities[solution]:
            return solution
    return -1


# 470
def get_invalid_letters(user_command) -> str:
    """Makes sure player input consists of valid colors for selected game configuration."""
    valid_colors = COLOR_LETTERS[:NUM_COLORS]
    invalid_letters = ""
    for letter in user_command:
        if letter not in valid_colors:
            invalid_letters = invalid_letters + letter
    return invalid_letters


```

这是一个Python 2000-线的函数，定义了两个函数。

1. `print_board` 函数接收一个 `guesses` 列表，打印过去每一个guess的轮子信息，如：
```
Guess: 1  1  2  2  3  4  5  5  6  7  7  8  8  9  9  9  1  1  2  2  3  3  4  4  5  5  6  7  8  9 10
```
2. `possibility_to_color_code` 函数接收一个 `possibility` 整数，返回一个颜色代号，解释见下：
```
"ABCDEFGHIJKLMNOPQRSTUVWXYZ#0北欧#1北极#2南极#3冲绳#4牡丹#5 Methods#6 period#7 teamwork#8 solutions#9艺龙#10文武双馨"
```
换句话说，这个函数接受一个可能的一组密码（可以是0-9，A-Z）和一个数字，并返回一个颜色代号（使用 I - XZ）以表示可能的密钥。


```
# 2000
def print_board(guesses) -> None:
    """Print previous guesses within the round."""
    print("Board")
    print("Move\tGuess\tBlack White")
    for idx, guess in enumerate(guesses):
        print(f"{idx + 1}\t{guess[0]}\t{guess[1]}     {guess[2]}")


def possibility_to_color_code(possibility: int) -> str:
    """Accepts a (decimal) number representing one permutation in the realm of
    possible secret codes and returns the color code mapped to that permutation.
    This algorithm is essentially converting a decimal  number to a number with
    a base of #num_colors, where each color code letter represents a digit in
    that #num_colors base."""
    color_code: str = ""
    pos: int = NUM_COLORS ** NUM_POSITIONS  # start with total possibilities
    remainder = possibility
    for _ in range(NUM_POSITIONS - 1, 0, -1):  # process all but the last digit
        pos = pos // NUM_COLORS
        color_code += COLOR_LETTERS[remainder // pos]
        remainder = remainder % pos
    color_code += COLOR_LETTERS[remainder]  # last digit is what remains
    return color_code


```

这段代码定义了一个名为 `compare_two_positions` 的函数，它比较了一个候选人和参考人员在一个职位（以字符的形式表示）上的猜测和答案。它返回了一个包含黑色和白色（正确颜色）的列表。

具体来说，该函数首先初始化了一个名为 `guess` 的字符串变量和一个名为 `answer` 的字符串变量。然后，它进入一个循环，遍历所有的职位。对于每个职位，它比较猜测和答案是否匹配。如果是正确的颜色和位置，它增加了一个 `increment` 变量。否则，如果是正确的颜色和位置，它增加了 `whites` 变量，同时将答案的一部分和正确的颜色一起添加到了猜测中。最后，它返回了初始猜测、正确颜色列表和正确颜色计数。


```
# 4500
def compare_two_positions(guess: str, answer: str) -> List[Union[str, int]]:
    """Returns blacks (correct color and position) and whites (correct color
    only) for candidate position (guess) versus reference position (answer)."""
    increment = 0
    blacks = 0
    whites = 0
    initial_guess = guess
    for pos in range(0, NUM_POSITIONS):
        if guess[pos] != answer[pos]:
            for pos2 in range(0, NUM_POSITIONS):
                if not (
                    guess[pos] != answer[pos2] or guess[pos2] == answer[pos2]
                ):  # correct color but not correct place
                    whites = whites + 1
                    answer = answer[:pos2] + chr(increment) + answer[pos2 + 1:]
                    guess = guess[:pos] + chr(increment + 1) + guess[pos + 1:]
                    increment = increment + 2
        else:  # correct color and placement
            blacks = blacks + 1
            # THIS IS DEVIOUSLY CLEVER
            guess = guess[:pos] + chr(increment + 1) + guess[pos + 1:]
            answer = answer[:pos] + chr(increment) + answer[pos + 1:]
            increment = increment + 2
    return [initial_guess, blacks, whites]


```

这段代码是一个Python函数，名为`print_score`，功能是打印得分。它接受一个布尔类型的参数`is_final_score`，默认值为False，表示在每一轮结束后是否打印最终得分。

函数内部的逻辑是：

1. 如果`is_final_score`为True，那么打印"GAME OVER"和"FINAL SCORE："；
2. 否则，打印"SCORE："并输出当前电脑得分和人类得分。

最终，这段代码会在每一轮结束后打印当前的得分，包括每一轮的最终得分。


```
# 5000 + logic from 1160
def print_score(is_final_score: bool = False) -> None:
    """Print score after each turn ends, including final score at end of game."""
    if is_final_score:
        print("GAME OVER")
        print("FINAL SCORE:")
    else:
        print("SCORE:")
    print(f"     COMPUTER {computer_score}")
    print(f"     HUMAN    {human_score}")


if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by Anthony Rubick [AnthonyMichaelTDM](https://github.com/AnthonyMichaelTDM)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by Anthony Rubick [AnthonyMichaelTDM](https://github.com/AnthonyMichaelTDM)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Math Dice

The program presents pictorial drill on addition facts using printed dice with no reading involved. It is good for beginning addition, since the answer can be derived from counting spots on the dice as well as by memorizing math facts or awareness of number concepts. It is especially effective run on a CRT terminal.

It was originally written by Jim Gerrish, a teacher at the Bernice A. Ray School in Hanover, New Hampshire.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=113)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=128)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `61_Math_Dice/csharp/GameState.cs`

这段代码定义了一个名为 MathDice 的命名空间，其中定义了一个名为 GameState 的枚举类型，它有两个成员变量，分别为 FirstAttempt 和 SecondAttempt，它们的值为 0 和 1。

枚举类型是一种数据类型，它可以用来表示一组相关的值，其中每个值的含义在类型定义中已经明确指定。

在这个例子中，GameState 枚举类型定义了两种不同的游戏状态，FirstAttempt 和 SecondAttempt。通过使用枚举类型，可以方便地在程序中使用不同的状态表示法，使得代码更加清晰易懂。


```
﻿namespace MathDice
{
    public enum GameState
    {
        FirstAttempt = 0,
        SecondAttempt = 1,
    }
}

```

# `61_Math_Dice/csharp/Program.cs`

This is a class written in C# that outputs the game of Rock, Paper, Scissors. The class includes methods for rolling a fair die, displaying the game board, and giving the answer to the user. It also includes a helper method for displaying the left and right pip values of a 3-pip die.

The class has an instance variable called `gameState` which is a boolean indicating whether it's the first attempt or the second attempt. It also has instance variables for the top and bottom rows of the game board, as well as the die values for the die.

The class has two methods, `Roll` and `GetAnswer`. The `Roll` method takes a ref variable `die` and rolls a fair die. The `GetAnswer` method takes a ref variable and attempts to parse an integer input from the user.

The class also has a `DrawDie` method which displays the game board. This method displays the top row as `NO`, the middle row as `COUNT THE SPOTS`, and the bottom row as `NO`. It also displays the top row as `COUNT THE SPOTS` and the bottom row as `NO`.

The class also has a `drawDie` method which displays the game board. This method takes the number of pips as an argument, and then displays the appropriate section of the game board and the pips.

The class also has a helper method called `TwoPips` which displays the two pip values of a 3-pip die.


```
﻿using System;

namespace MathDice
{
    public static class Program
    {
        readonly static Random random = new Random();

        static int DieOne = 0;
        static int DieTwo = 0;

        private const string NoPips = "I     I";
        private const string LeftPip = "I *   I";
        private const string CentrePip = "I  *  I";
        private const string RightPip = "I   * I";
        private const string TwoPips = "I * * I";
        private const string Edge = " ----- ";

        static void Main(string[] args)
        {
            int answer;

            GameState gameState = GameState.FirstAttempt;

            Console.WriteLine("MATH DICE".CentreAlign());
            Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY".CentreAlign());
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine("THIS PROGRAM GENERATES SUCCESSIVE PICTURES OF TWO DICE.");
            Console.WriteLine("WHEN TWO DICE AND AN EQUAL SIGN FOLLOWED BY A QUESTION");
            Console.WriteLine("MARK HAVE BEEN PRINTED, TYPE YOUR ANSWER AND THE RETURN KEY.");
            Console.WriteLine("TO CONCLUDE THE LESSON, TYPE CONTROL-C AS YOUR ANSWER.");
            Console.WriteLine();
            Console.WriteLine();

            while (true)
            {
                if (gameState == GameState.FirstAttempt)
                {
                    Roll(ref DieOne);
                    Roll(ref DieTwo);

                    DrawDie(DieOne);
                    Console.WriteLine("   +");
                    DrawDie(DieTwo);
                }

                answer = GetAnswer();

                if (answer == DieOne + DieTwo)
                {
                    Console.WriteLine("RIGHT!");
                    Console.WriteLine();
                    Console.WriteLine("THE DICE ROLL AGAIN...");

                    gameState = GameState.FirstAttempt;
                }
                else
                {
                    if (gameState == GameState.FirstAttempt)
                    {
                        Console.WriteLine("NO, COUNT THE SPOTS AND GIVE ANOTHER ANSWER.");
                        gameState = GameState.SecondAttempt;
                    }
                    else
                    {
                        Console.WriteLine($"NO, THE ANSWER IS{DieOne + DieTwo}");
                        Console.WriteLine();
                        Console.WriteLine("THE DICE ROLL AGAIN...");
                        gameState = GameState.FirstAttempt;
                    }
                }
            }
        }

        private static int GetAnswer()
        {
            int answer;

            Console.Write("      =?");
            var input = Console.ReadLine();

            int.TryParse(input, out answer);

            return answer;
        }

        private static void DrawDie(int pips)
        {
            Console.WriteLine(Edge);
            Console.WriteLine(OuterRow(pips, true));
            Console.WriteLine(CentreRow(pips));
            Console.WriteLine(OuterRow(pips, false));
            Console.WriteLine(Edge);
            Console.WriteLine();
        }

        private static void Roll(ref int die) => die = random.Next(1, 7);

        private static string OuterRow(int pips, bool top)
        {
            return pips switch
            {
                1 => NoPips,
                var x when x == 2 || x == 3 => top ? LeftPip : RightPip,
                _ => TwoPips
            };
        }

        private static string CentreRow(int pips)
        {
            return pips switch
            {
                var x when x == 2 || x == 4 => NoPips,
                6 => TwoPips,
                _ => CentrePip
            };
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)

Conversion Notes

- There are minor spacing issues which have been preserved in this port.
- This implementation uses switch expressions to concisely place the dice pips in the right place.
- Random() is only pseudo-random but perfectly adequate for the purposes of simulating dice rolls.
- Console width is assumed to be 120 chars for the purposes of centrally aligned the intro text.


# `61_Math_Dice/csharp/StringExtensions.cs`

这段代码定义了一个名为 "MathDice" 的命名空间，其中包含一个名为 "StringExtensions" 的类。

在 "StringExtensions" 类中，定义了一个名为 " CentreAlign" 的方法，接受一个字符串参数 "value"。

该方法首先计算出该字符串 "value" 在控制台输出的默认宽度，即 120 个字符。然后，它计算出需要左对齐的额外字符数，这个字符数是控制台宽度减去字符串 "value" 的长度的一半再加上字符串 "value" 的长度。最后，该方法使用 "PadLeft" 和 "PadRight" 方法对字符串进行左对齐和右对齐，使得字符串输出时对齐到控制台宽度。

所以，该代码的作用是定义了一个字符串扩展方法 "CentreAlign"，用于将一个字符串 "value" 左对齐并使其输出对齐到控制台宽度。


```
﻿namespace MathDice
{
    public static class StringExtensions
    {
        private const int ConsoleWidth = 120; // default console width

        public static string CentreAlign(this string value)
        {
            int spaces = ConsoleWidth - value.Length;
            int leftPadding = spaces / 2 + value.Length;

            return value.PadLeft(leftPadding).PadRight(ConsoleWidth);
        }
    }
}

```

# `61_Math_Dice/java/Die.java`

This is a Java class called `Die` that simulates a physical die. It has a `getFaceValue()` method that returns the `faceValue`, which is the value of the die face (1, 2, or 3). It also has a `throwDie()` method that generates a new random number between 1 and the `sides` value (default is 6) to be stored in `faceValue`.

The `Die` class has a `main()` method that creates a new `Die` object with default sides (6), and then calls the `throwDie()` method to generate a random number to be stored in `faceValue`. Finally, it calls the `printDie()` method to print the face value.

The `printDie()` method generates a random number between 1 and the `sides` value and prints the appropriate string of characters depending on the value. It prints the face value first, then moves on to the next set of characters (e.g. 4-face die will print "| 3 2 |").

The `printTwo()` method prints the string "| *   |" for a 2-face die, and the string "|  *   |" for a 4-face die.

Note that this implementation assumes that the `sides` value is a power of 2.


```
import java.util.Random;

public class Die {
    private static final int DEFAULT_SIDES = 6;
    private int faceValue;
    private int sides;
    private Random generator = new Random();

    /**
     * Construct a new Die with default sides
     */
    public Die() {
        this.sides = DEFAULT_SIDES;
        this.faceValue = 1 + generator.nextInt(sides);
    }

    /**
     * Generate a new random number between 1 and sides to be stored in faceValue
     */
    private void throwDie() {
        this.faceValue = 1 + generator.nextInt(sides);
    }


    /**
     * @return the faceValue
     */
    public int getFaceValue() {
        return faceValue;
    }


    public void printDie() {
        throwDie();
        int x = this.getFaceValue();

        System.out.println(" ----- ");

        if(x==4||x==5||x==6) {
            printTwo();
        } else if(x==2||x==3) {
            System.out.println("| *   |");
        } else {
            printZero();
        }

        if(x==1||x==3||x==5) {
            System.out.println("|  *  |");
        } else if(x==2||x==4) {
            printZero();
        } else {
            printTwo();
        }

        if(x==4||x==5||x==6) {
            printTwo();
        } else if(x==2||x==3) {
            System.out.println("|   * |");
        } else {
            printZero();
        }

        System.out.println(" ----- ");
    }

    private void printZero() {
        System.out.println("|     |");
    }

    private void printTwo() {
        System.out.println("| * * |");
    }
}

```

# `61_Math_Dice/java/MathDice.java`

这段代码定义了一个名为 MathDice 的类，其作用是生成两个 dice 并输出基本计算机游戏。程序的主要功能是让用户轮流投掷两个 dice，然后根据用户输入的答案来判断是否正确。如果用户的答案是正确的，程序会输出 "Correct"，否则会输出 "No, the answer is XXX!" 并继续让学生尝试。如果用户在两次投掷之间没有提供答案，程序会继续等待下一次投掷并输出 "No, count the spots and give another answer."。


```
import java.util.Scanner;

public class MathDice {

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        Die dieOne = new Die();
        Die dieTwo = new Die();
        int guess = 1;
        int answer;

        System.out.println("Math Dice");
        System.out.println("https://github.com/coding-horror/basic-computer-games");
        System.out.println();
        System.out.print("This program generates images of two dice.\n"
                + "When two dice and an equals sign followed by a question\n"
                + "mark have been printed, type your answer, and hit the ENTER\n" + "key.\n"
                + "To conclude the program, type 0.\n");

        while (true) {
            dieOne.printDie();
            System.out.println("   +");
            dieTwo.printDie();
            System.out.println("   =");
            int tries = 0;
            answer = dieOne.getFaceValue() + dieTwo.getFaceValue();

            while (guess!=answer && tries < 2) {
                if(tries == 1)
                    System.out.println("No, count the spots and give another answer.");
                try{
                    guess = in.nextInt();
                } catch(Exception e) {
                    System.out.println("Thats not a number!");
                    in.nextLine();
                }

                if(guess == 0)
                    System.exit(0);

                tries++;
            }

            if(guess != answer){
                System.out.println("No, the answer is " + answer + "!");
            } else {
                System.out.println("Correct");
            }
            System.out.println("The dice roll again....");
        }
    }

}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `61_Math_Dice/javascript/mathdice.js`

这段代码定义了两个函数，分别是`print()`和`input()`。

`print()`函数的作用是在网页中创建一个输出框（output），将用户输入的字符串通过`document.getElementById()`获取的元素注入到输出框中。

`input()`函数的作用是获取用户输入的字符串，并将其存储在变量`input_str`中。然后，它通过调用`input_element.focus()`来获取输入框的焦点，以便用户可以编辑输入框的内容。接着，它使用`input_element.addEventListener()`来监听输入框的`keydown`事件，当用户按下键盘上的数字13时，会将用户输入的字符串存储到变量`input_str`中，并将其输出到网页上，以便用户查看他们的输入。


```
// MATH DICE
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

It looks like you are trying to print a series of numbers in a given order and then print a message for each set of numbers. The numbers are being generated using the `Math.random()` method, and the order in which the numbers are being generated is determined by the values of the `d` variable.

If you would like to modify this code to print a different message for each set of numbers, you can do so by using a `for` loop instead of a `while` loop. Here is an example of how you could modify the code to print a different message for each set of numbers:
```
TO CONCLUDE THE LESSON, TYPE ZERO AS YOUR ANSWER.

print("TO CONCLUDE THE LESSON, TYPE ZERO AS YOUR ANSWER.")
print("\n");
print("\n");
print("\n");
n = 0;
while (1) {
   n++;
   d = Math.floor(6 * Math.random() + 1);
   print(" ----- \n");
   if (d == 1)
       print("I     I\n");
   else if (d == 2 || d == 3)
       print("I *   I\n");
   else
       print("I * * I\n");
   if (d == 2 || d == 4)
       print("I     I\n");
   else if (d == 6)
       print("I * * I\n");
   else
       print("I  *  I\n");
   if (d == 1)
       print("I     I\n");
   else if (d == 2 || d == 3)
       print("I   * I\n");
   else
       print("I * * I\n");
   print(" ----- \n");
   print("\n");
   if (n != 2) {
       print("   +\n");
       print("\n");
       a = d;
       continue;
   }
   t = d + a;
   print("      =");
   t1 = parseInt(await input());
   if (t1 == 0)
       break;
   if (t1 != t) {
       print("NO, COUNT THE SPOTS AND GIVE ANOTHER ANSWER.\n");
       print("      =");
       t1 = parseInt(await input());
       if (t1 != t) {
           print("NO, THE ANSWER IS " + t + "\n");
       }
   }
   if (t1 == t) {
       print("RIGHT!\n");
   }
   print("\n");
   print("THE DICE ROLL AGAIN...\n");
   print("\n");
   n = 0;
}
```
You can also modify the code to print a different message for each set of numbers by using a `for` loop instead of a `while` loop.
```
TO CONCLUDE THE LESSON, TYPE ZERO AS YOUR ANSWER.

print("TO CONCLUDE THE LESSON, TYPE ZERO AS YOUR ANSWER.")
print("\n");
print("\n");
print("\n");
n = 0;

while (1) {
   n++;
   d = Math.floor(6 * Math.random() + 1);
   print(" ----- \n");
   if (d == 1)
       print("I     I\n");
   else if (d == 2 || d == 3)
       print("I *   I\n");
   else
       print("I * * I\n");
   if (d == 2 || d == 4)
       print("I     I\n");
   else if (d == 6)
       print("I * * I\n");
   else
       print("I  *  I\n");
   if (d == 1)
       print("I     I\n");
   else if (d == 2 || d == 3)
       print("I   *  I\n");
   else
       print("I * * I\n");
   print(" ----- \n");
   print("\n");
   if (n != 2) {
       print("   +\n");
       print("\n");
       a = d;
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
    print(tab(31) + "MATH DICE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("THIS PROGRAM GENERATES SUCCESSIVE PICTURES OF TWO DICE.\n");
    print("WHEN TWO DICE AND AN EQUAL SIGN FOLLOWED BY A QUESTION\n");
    print("MARK HAVE BEEN PRINTED, TYPE YOUR ANSWER AND THE RETURN KEY.\n"),
    print("TO CONCLUDE THE LESSON, TYPE ZERO AS YOUR ANSWER.\n");
    print("\n");
    print("\n");
    n = 0;
    while (1) {
        n++;
        d = Math.floor(6 * Math.random() + 1);
        print(" ----- \n");
        if (d == 1)
            print("I     I\n");
        else if (d == 2 || d == 3)
            print("I *   I\n");
        else
            print("I * * I\n");
        if (d == 2 || d == 4)
            print("I     I\n");
        else if (d == 6)
            print("I * * I\n");
        else
            print("I  *  I\n");
        if (d == 1)
            print("I     I\n");
        else if (d == 2 || d == 3)
            print("I   * I\n");
        else
            print("I * * I\n");
        print(" ----- \n");
        print("\n");
        if (n != 2) {
            print("   +\n");
            print("\n");
            a = d;
            continue;
        }
        t = d + a;
        print("      =");
        t1 = parseInt(await input());
        if (t1 == 0)
            break;
        if (t1 != t) {
            print("NO, COUNT THE SPOTS AND GIVE ANOTHER ANSWER.\n");
            print("      =");
            t1 = parseInt(await input());
            if (t1 != t) {
                print("NO, THE ANSWER IS " + t + "\n");
            }
        }
        if (t1 == t) {
            print("RIGHT!\n");
        }
        print("\n");
        print("THE DICE ROLL AGAIN...\n");
        print("\n");
        n = 0;
    }
}

```

这是经典的 "Hello, World!" 程序，用于在 C 语言环境中启动一个新程序。该程序由 John Griesemer 于 1953 年开发，是程序调试和学习的流行方式。

在这段代码中，没有任何实际的功能和数据，只是一个简单的程序入口点。这个程序不会产生任何输出，也不对任何变量或文件进行任何修改。但是，它总是作为一个 "null" 程序，即在开始时设置堆栈为程序的结束地址，因此所有子程序都将从该点开始执行。

该程序的主要目的是在开发过程中提供一个简单的开始点，以便将程序的输出结果输出到屏幕上，或者将程序的输出结果保存到一个文件中。


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
