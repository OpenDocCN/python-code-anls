# BasicComputerGames源码解析 50

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


# `47_Hi-Lo/python/hilo.py`

I'm sorry, but as an AI language model, I don't have the code for the game "HI LO". However, I can give you some tips on how to create a game of HI LO:

1. Write the story and objective of the game.
2. Create a character (e.g., a young boy named Jack) and a setting.
3. Decide on the rules of the game.
4. Write a code for the game.
5. Test the game and make any necessary modifications.
6. Share the game with your target audience.

If you have any questions or need further assistance, feel free to ask.


```
#!/usr/bin/env python3
import random

MAX_ATTEMPTS = 6
QUESTION_PROMPT = "? "


def main() -> None:
    print("HI LO")
    print("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")
    print("THIS IS THE GAME OF HI LO.\n")
    print("YOU WILL HAVE 6 TRIES TO GUESS THE AMOUNT OF MONEY IN THE")
    print("HI LO JACKPOT, WHICH IS BETWEEN 1 AND 100 DOLLARS.  IF YOU")
    print("GUESS THE AMOUNT, YOU WIN ALL THE MONEY IN THE JACKPOT!")
    print("THEN YOU GET ANOTHER CHANCE TO WIN MORE MONEY.  HOWEVER,")
    print("IF YOU DO NOT GUESS THE AMOUNT, THE GAME ENDS.\n\n")

    total_winnings = 0
    while True:
        print()
        secret = random.randint(1, 100)
        guessed_correctly = False

        for _attempt in range(MAX_ATTEMPTS):
            print("YOUR GUESS", end=QUESTION_PROMPT)
            guess = int(input())

            if guess == secret:
                print(f"GOT IT!!!!!!!!!!   YOU WIN {secret} DOLLARS.")
                guessed_correctly = True
                break
            elif guess > secret:
                print("YOUR GUESS IS TOO HIGH.")
            else:
                print("YOUR GUESS IS TOO LOW.")

        if guessed_correctly:
            total_winnings += secret
            print(f"YOUR TOTAL WINNINGS ARE NOW {total_winnings} DOLLARS.")
        else:
            print(f"YOU BLEW IT...TOO BAD...THE NUMBER WAS {secret}")

        print("\n")
        print("PLAY AGAIN (YES OR NO)", end=QUESTION_PROMPT)
        answer = input().upper()
        if answer != "YES":
            break

    print("\nSO LONG.  HOPE YOU ENJOYED YOURSELF!!!")


```

这段代码是一个if语句，判断当前脚本是否被命为__main__. 如果当前脚本被命为__main__，则会执行if语句块内的代码。

在这个例子中，if语句块内的代码是“main()”。这部分代码将会被解释器执行，也就是呼叫一个函数并让它自由地歌唱。如果当前脚本不是被命为__main__，那么if语句块内的代码将永远无法被执行。


```
if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/) by [R.T. Lechow](https://github.com/rtlechow)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### High IQ

This is a computerized version of an old European solitaire game of logic. The game starts with a pegboard shaped like a cross having pegs in every hole but the center. The object is to remove all 32 pegs, or as many as possible, by jumping into an empty hole, then removing the jumped peg.

There are several different winning strategies for playing, and of course, each strategy can be played eight different ways on the board. Can you find a consistent winner?

Charles Lund wrote this game while at The American School in The Hague, Netherlands.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=86)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=101)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html)

Converted to [D](https://dlang.org/) by [Bastiaan Veelo](https://github.com/veelo).

## Running the code

Assuming the reference [dmd](https://dlang.org/download.html#dmd) compiler:
```shell
dmd -dip1000 -run highiq.d
```

[Other compilers](https://dlang.org/download.html) also exist.


## Discussion

The original BASIC game code made use of calculus and clever choises of field IDs to determine the validity of moves.
This is the original layout of IDs over the board:

```
          13   14   15

          22   23   24

29   30   31   32   33   34   35

38   39   40   41   42   43   44

47   48   49   50   51   52   53

          58   59   60

          67   68   69
```

This seems not very logical, because, wouldn't it make much more sense to let columns increase with 1 and rows increase
with 10, so you'd get a consistent coordinate system? It seems that the original author's first step in validating
moves was to check that moves jumped from one field over another one onto the next. He did this by making sure that
adjacent IDs alter between even and odd horizontally *and* vertically. So a valid move was always from an even ID to an
even ID *or* from an odd ID to an odd ID. So one of the checks that the BASIC code made was that the sum of both IDs
was even. This is of course not a sufficient test, because moves that jump over three fields are illegal. Therefore the
IDs seem to have been carefully laid oud so that the IDs increase with 1 horizontally, and 9 vertically, everywhere. So
the only valid difference between IDs for a horizontal move was always 2, and the only valid difference for a vertical
move was always 18.

Fact of the matter is, however, that checking for difference is sufficient and the even sum rule is superfluous, so
there is no need for the peculiar distribution of field IDs. Therefore I have chosen the following more logical
distribution:

```
          13   14   15

          23   24   25

31   32   33   34   35   36   37

41   42   43   44   45   46   47

51   52   53   54   55   56   57

          63   64   65

          73   74   75
```

As a consequence, the implementation of the game code has become much simpler; Not alone due to one less check, but due
to the fact that conversions between IDs and board coordinates have become unnecessary and thus we can work with a single
representation of the board state.

This version makes a prettier print of the board than the BASIC original, with coordinates for every move, and explains
illegal moves.


## Demo

```
                      H-I-Q
(After Creative Computing  Morristown, New Jersey)


Fields are identified by 2-digit numbers, each
between 1 and 7. Example: the middle field is 44,
the bottom middle is 74.

      _1  _2  _3  _4  _5  _6  _7
            ┌───┬───┬───┐
 1_         │ ■ │ ■ │ ■ │
            ├───┼───┼───┤
 2_         │ ■ │ ■ │ ■ │
    ┌───┬───┼───┼───┼───┼───┬───┐
 3_ │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │
    ├───┼───┼───┼───┼───┼───┼───┤
 4_ │ ■ │ ■ │ ■ │   │ ■ │ ■ │ ■ │
    ├───┼───┼───┼───┼───┼───┼───┤
 5_ │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │
    └───┴───┼───┼───┼───┼───┴───┘
 6_         │ ■ │ ■ │ ■ │
            ├───┼───┼───┤
 7_         │ ■ │ ■ │ ■ │
            └───┴───┴───┘

Move which peg? 23
The peg at 23 has nowhere to go. Try again.

Move which peg? 24
To where? 34
Field 34 is occupied. Try again.
To where? 54
Field 54 is occupied. Try again.
To where? 44

      _1  _2  _3  _4  _5  _6  _7
            ┌───┬───┬───┐
 1_         │ ■ │ ■ │ ■ │
            ├───┼───┼───┤
 2_         │ ■ │   │ ■ │
    ┌───┬───┼───┼───┼───┼───┬───┐
 3_ │ ■ │ ■ │ ■ │   │ ■ │ ■ │ ■ │
    ├───┼───┼───┼───┼───┼───┼───┤
 4_ │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │
    ├───┼───┼───┼───┼───┼───┼───┤
 5_ │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │
    └───┴───┼───┼───┼───┼───┴───┘
 6_         │ ■ │ ■ │ ■ │
            ├───┼───┼───┤
 7_         │ ■ │ ■ │ ■ │
            └───┴───┴───┘

Move which peg? 14
The peg at 14 has nowhere to go. Try again.

Move which peg? 24
There is no peg at 24. Try again.

Move which peg? 44
The peg at 44 has nowhere to go. Try again.

Move which peg? 32
To where? 22
Field 22 is ouside the board. Try again.
To where? 33
Field 33 is occupied. Try again.
To where? 34

      _1  _2  _3  _4  _5  _6  _7
            ┌───┬───┬───┐
 1_         │ ■ │ ■ │ ■ │
            ├───┼───┼───┤
 2_         │ ■ │   │ ■ │
    ┌───┬───┼───┼───┼───┼───┬───┐
 3_ │ ■ │   │   │ ■ │ ■ │ ■ │ ■ │
    ├───┼───┼───┼───┼───┼───┼───┤
 4_ │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │
    ├───┼───┼───┼───┼───┼───┼───┤
 5_ │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │
    └───┴───┼───┼───┼───┼───┴───┘
 6_         │ ■ │ ■ │ ■ │
            ├───┼───┼───┤
 7_         │ ■ │ ■ │ ■ │
            └───┴───┴───┘

Move which peg? 44
To where? 33
You cannot move diagonally. Try again.
To where? 24

      _1  _2  _3  _4  _5  _6  _7
            ┌───┬───┬───┐
 1_         │ ■ │ ■ │ ■ │
            ├───┼───┼───┤
 2_         │ ■ │ ■ │ ■ │
    ┌───┬───┼───┼───┼───┼───┬───┐
 3_ │ ■ │   │   │   │ ■ │ ■ │ ■ │
    ├───┼───┼───┼───┼───┼───┼───┤
 4_ │ ■ │ ■ │ ■ │   │ ■ │ ■ │ ■ │
    ├───┼───┼───┼───┼───┼───┼───┤
 5_ │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │
    └───┴───┼───┼───┼───┼───┴───┘
 6_         │ ■ │ ■ │ ■ │
            ├───┼───┼───┤
 7_         │ ■ │ ■ │ ■ │
            └───┴───┴───┘

Move which peg? 36
To where? 33
You can't jump that far. Try again.
To where? 35
Field 35 is occupied. Try again.
To where? 34

      _1  _2  _3  _4  _5  _6  _7
            ┌───┬───┬───┐
 1_         │ ■ │ ■ │ ■ │
            ├───┼───┼───┤
 2_         │ ■ │ ■ │ ■ │
    ┌───┬───┼───┼───┼───┼───┬───┐
 3_ │ ■ │   │   │ ■ │   │   │ ■ │
    ├───┼───┼───┼───┼───┼───┼───┤
 4_ │ ■ │ ■ │ ■ │   │ ■ │ ■ │ ■ │
    ├───┼───┼───┼───┼───┼───┼───┤
 5_ │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │
    └───┴───┼───┼───┼───┼───┴───┘
 6_         │ ■ │ ■ │ ■ │
            ├───┼───┼───┤
 7_         │ ■ │ ■ │ ■ │
            └───┴───┴───┘

Move which peg? 46
To where? 36
You need to jump over another peg. Try again.
To where? down
Field 00 is ouside the board. Try again.
To where?
```


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `48_High_IQ/java/src/HighIQ.java`

This is a Java implementation of a simple game of rock-paper-scissors where two players take turns trying to rock, paper or scissors against each other.

It uses a `RockPaperScissors` class to represent the game board and has the following methods:

* `isStart(int from, int to)`: checks if the start position is valid for the game (i.e. it is an even position).
* `isFinished()`: checks if the game is finished and no more moves can be made.
* `play(int from, int to)`: checks if the move is valid and updates the game board accordingly.
* `printBoard()`: prints the game board.

It also has a `getChar()` method to retrieve the character of a given position.

Note that this implementation does not include any error checking for invalid input or out of range positions.


```
import java.io.PrintStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

/**
 * Game of HighIQ
 * <p>
 * Based on the Basic Game of HighIQ Here:
 * https://github.com/coding-horror/basic-computer-games/blob/main/48_High_IQ/highiq.bas
 *
 * No additional functionality has been added
 */
public class HighIQ {

    //Game board, as a map of position numbers to their values
    private final Map<Integer, Boolean> board;

    //Output stream
    private final PrintStream out;

    //Input scanner to use
    private final Scanner scanner;


    public HighIQ(Scanner scanner) {
        out = System.out;
        this.scanner = scanner;
        board = new HashMap<>();

        //Set of all locations to put initial pegs on
        int[] locations = new int[]{
                13, 14, 15, 22, 23, 24, 29, 30, 31, 32, 33, 34, 35, 38, 39, 40, 42, 43, 44, 47, 48, 49, 50, 51, 52, 53, 58, 59, 60, 67, 68, 69
        };

        for (int i : locations) {
            board.put(i, true);
        }

        board.put(41, false);
    }

    /**
     * Plays the actual game, from start to finish.
     */
    public void play() {
        do {
            printBoard();
            while (!move()) {
                out.println("ILLEGAL MOVE, TRY AGAIN...");
            }
        } while (!isGameFinished());

        int pegCount = 0;
        for (Integer key : board.keySet()) {
            if (board.getOrDefault(key, false)) {
                pegCount++;
            }
        }

        out.println("YOU HAD " + pegCount + " PEGS REMAINING");

        if (pegCount == 1) {
            out.println("BRAVO!  YOU MADE A PERFECT SCORE!");
            out.println("SAVE THIS PAPER AS A RECORD OF YOUR ACCOMPLISHMENT!");
        }
    }

    /**
     * Makes an individual move
     * @return True if the move was valid, false if the user made an error and the move is invalid
     */
    public boolean move() {
        out.println("MOVE WHICH PIECE");
        int from = scanner.nextInt();

        //using the getOrDefault, which will make the statement false if it is an invalid position
        if (!board.getOrDefault(from, false)) {
            return false;
        }

        out.println("TO WHERE");
        int to = scanner.nextInt();

        if (board.getOrDefault(to, true)) {
            return false;
        }

        //Do nothing if they are the same
        if (from == to) {
            return true;
        }

        //using the difference to check if the relative locations are valid
        int difference = Math.abs(to - from);
        if (difference != 2 && difference != 18) {
            return false;
        }

        //check if there is a peg between from and to
        if (!board.getOrDefault((to + from) / 2, false)) {
            return false;
        }

        //Actually move
        board.put(from,false);
        board.put(to,true);
        board.put((from + to) / 2, false);

        return true;
    }

    /**
     * Checks if the game is finished
     * @return True if there are no more moves, False otherwise
     */
    public boolean isGameFinished() {
        for (Integer key : board.keySet()) {
            if (board.get(key)) {
                //Spacing is either 1 or 9
                //Looking to the right and down from every point, checking for both directions of movement
                for (int space : new int[]{1, 9}) {
                    Boolean nextToPeg = board.getOrDefault(key + space, false);
                    Boolean hasMovableSpace = !board.getOrDefault(key - space, true) || !board.getOrDefault(key + space * 2, true);
                    if (nextToPeg && hasMovableSpace) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    public void printBoard() {
        for (int i = 0; i < 7; i++) {
            for (int j = 11; j < 18; j++) {
                out.print(getChar(j + 9 * i));
            }
            out.println();
        }
    }

    private char getChar(int position) {
        Boolean value = board.get(position);
        if (value == null) {
            return ' ';
        } else if (value) {
            return '!';
        } else {
            return 'O';
        }
    }
}

```

# `48_High_IQ/java/src/HighIQGame.java`



This is a Java program that simulates a game of Connect-the-Dot. It prompts the user to either play the game or exit. If the user chooses to play, it prints the instructions for the game and then starts playing.

The game is played by connecting a series of dots, with the goal of connecting five dots in a row to win the game. The user is prompted to input the dots to connect. Each input line has a space between it and the next line, so the user can separate each input with a space.

If the user wants to exit the game, they can simply type "YES".

This program is just one example of how Connect-the-Dot could be implemented in Java. There are many other ways to approach the same problem, and different programs may have different features and functionality.


```
import java.util.Scanner;

public class HighIQGame {
    public static void main(String[] args) {

        printInstructions();

        Scanner scanner = new Scanner(System.in);
        do {
            new HighIQ(scanner).play();
            System.out.println("PLAY AGAIN (YES OR NO)");
        } while(scanner.nextLine().equalsIgnoreCase("yes"));
    }

    public static void printInstructions() {
        System.out.println("\t\t\t H-I-Q");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println("HERE IS THE BOARD:");
        System.out.println("          !    !    !");
        System.out.println("         13   14   15\n");
        System.out.println("          !    !    !");
        System.out.println("         22   23   24\n");
        System.out.println("!    !    !    !    !    !    !");
        System.out.println("29   30   31   32   33   34   35\n");
        System.out.println("!    !    !    !    !    !    !");
        System.out.println("38   39   40   41   42   43   44\n");
        System.out.println("!    !    !    !    !    !    !");
        System.out.println("47   48   49   50   51   52   53\n");
        System.out.println("          !    !    !");
        System.out.println("         58   59   60\n");
        System.out.println("          !    !    !");
        System.out.println("         67   68   69");
        System.out.println("TO SAVE TYPING TIME, A COMPRESSED VERSION OF THE GAME BOARD");
        System.out.println("WILL BE USED DURING PLAY.  REFER TO THE ABOVE ONE FOR PEG");
        System.out.println("NUMBERS.  OK, LET'S BEGIN.");
    }
}

```

# `48_High_IQ/javascript/highiq.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。

函数 `print` 的作用是将一个字符串渲染到网页的输出区域（通常是 `<div>` 元素）中。这个字符串是通过对字符串进行 `.appendChild` 方法添加到 `document.getElementById("output")` 对象中的。

函数 `input` 的作用是获取用户输入的字符串（通常是问题或者是一个提示信息）。它通过调用 `input` 函数，并将获取到的字符串存储在 `input_str` 变量中，然后将其返回。

在这两个函数中，都使用了 `document.getElementById("output")` 获取到网页上某个元素的引用，并对其进行添加、删除或者修改。在 `input` 函数中，还使用了 `document.createElement("INPUT")` 创建了一个 `<input>` 元素，并设置其属性以获取用户输入。


```
// H-I-Q
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

这是一段 JavaScript 代码，定义了一个名为 "tab" 的函数，以及三个变量：b、t 和 m。

函数的作用是接收一个参数 "space"，并在该参数上输出字符串 "tab"，结尾有一个空格。

变量 b 是一个包含 11 个元素的数组，变量 t 是一个包含 11 个元素的数组，变量 m 是一个包含 22 个元素的数组，数组元素序列为 [13, 14, 15, ..., 67, 68, 69]。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var b = [];
var t = [];
var m = [,13,14,15,
          22,23,24,
    29,30,31,32,33,34,35,
    38,39,40,41,42,43,44,
    47,48,49,50,51,52,53,
          58,59,60,
          67,68,69];
```

这段代码定义了两个变量：变量 z 和变量 p，但没有对它们进行初始化。

接下来定义了一个名为 print_board 的函数，该函数用于在控制台上打印一个 9x9 的方格矩阵。函数内部也有两个变量：变量 z 和变量 p，同样没有进行初始化。

在 print_board 函数内部，使用两个嵌套循环来遍历方格矩阵中的每个元素。对于每个元素，首先创建一个空字符串并打印出来。然后，根据当前元素是 1、9 中的一个或者是 4、5、6 中的一个，将相应的字符添加到字符串中。如果是 5 的话，还会打印 "!" 感叹符号。否则，如果是 4 或 6，则添加 "O" 字符。最后，在循环结束后，打印出来的是整个字符串，中间用 "\n" 换行。


```
var z;
var p;

//
// Print board
//
function print_board()
{
    for (x = 1; x <= 9; x++) {
        str = "";
        for (y = 1; y <= 9; y++) {
            if (x == 1 || x == 9 || y == 1 || y == 9)
                continue;
            if (x == 4 || x == 5 || x == 6 || y == 4 || y == 5 || y == 6) {
                while (str.length < y * 2)
                    str += " ";
                if (t[x][y] == 5)
                    str += "!";
                else
                    str += "O";
            }
        }
        print(str + "\n");
    }
}

```

这段代码的作用是更新一个二维数组 `board` 中每个元素的值，使得经过数组中元素的平方后，能够被整除。

具体来说，代码首先初始化了一个变量 `c`，其值为 1。然后，用一个嵌套的循环遍历了数组 `board` 中所有的元素，每个元素的列和行都是从 1 到 9，计算并更新了每个元素的值。

在更新过程中，代码判断当前元素值是否等于它所在的平方数，如果是，则执行特定的操作(比如将该元素值设为 0，或者将该元素所在的行和列的元素值都设为 5)，从而实现了对数组元素值的修改。

此外，代码还做了一些额外的操作，比如将 `board` 中所有元素中，行列标识为 0 的元素的值都设为 0，以及将 `board` 中所有元素中，行列标识为奇数的元素的值都设为 -3。这些操作是为了让经过平方后，仍能被整除。


```
//
// Update board
//
function update_board()
{
    c = 1;
    for (var x = 1; x <= 9; x++) {
        for (var y = 1; y <= 9; y++, c++) {
            if (c != z)
                continue;
            if (c + 2 == p) {
                if (t[x][y + 1] == 0)
                    return false;
                t[x][y + 2] = 5;
                t[x][y + 1] = 0;
                b[c + 1] = -3;
            } else if (c + 18 == p) {
                if (t[x + 1][y] == 0)
                    return false;
                t[x + 2][y] = 5;
                t[x + 1][y] = 0;
                b[c + 9] = -3;
            } else if (c - 2 == p) {
                if (t[x][y - 1] == 0)
                    return false;
                t[x][y - 2] = 5;
                t[x][y - 1] = 0;
                b[c - 1] = -3;
            } else if (c - 18 == p) {
                if (t[x - 1][y] == 0)
                    return false;
                t[x - 2][y] = 5;
                t[x - 1][y] = 0;
                b[c - 9] = -3;
            } else {
                continue;
            }
            b[z] = -3;
            b[p] = -7;
            t[x][y] = 0;
            return true;
        }
    }
}

```

这段代码是一个名为`check_game_over`的函数，它的作用是检查游戏是否结束。游戏结束的条件是某一个位置的`T`没有出现，且该位置的`T`满足一定的条件。如果满足游戏结束的条件，则返回`false`，否则返回`true`。

代码中使用了两个嵌套循环，第一个循环遍历`r`从2到8的整数，第二个循环遍历`c`从2到8的整数。在内部循环中，判断当前`T`是否为5，如果不是5，则执行`continue`跳过当前行。接着判断当前`T`所在的行是否已经大于3，如果是，则需要检查`T`左上角和右下角是否都为5，如果是，则游戏结束，返回`false`。接着判断当前`T`所在的列是否已经大于3，如果是，则需要检查`T`左上角和右下角是否都为5，如果是，则游戏结束，返回`false`。接着判断当前`T`所在的行是否小于7，如果是，则需要检查`T`右上角和左下角是否都为5，如果是，则游戏结束，返回`false`。接着判断当前`T`所在的列是否小于7，如果是，则需要检查`T`左下角和右上角是否都为5，如果是，则游戏结束，返回`false`。

因此，该函数可以作为判断游戏是否结束的依据，如果游戏结束，则返回`false`，否则返回`true`。


```
//
// Check for game over
//
// Rewritten because original subroutine was buggy
//
function check_game_over()
{
    f = 0;
    for (r = 2; r <= 8; r++) {
        for (c = 2; c <= 8; c++) {
            if (t[r][c] != 5)
                continue;
            f++;
            if (r > 3 && t[r - 1][c] == 5 && t[r - 2][c] == 0)
                return false;
            if (c > 3 && t[r][c - 1] == 5 && t[r][c - 2] == 0)
                return false;
            if (r < 7 && t[r + 1][c] == 5 && t[r + 2][c] == 0)
                return false;
            if (c < 7 && t[r][c + 1] == 5 && t[r][c + 2] == 0)
                return false;
        }
    }
    return true;
}

```

This appears to be a board game where the player must take turns placing pieces on a 6x6 game board. The player can also enter moves for other players to make.

The game board is divided into two main parts: the primary board and the secondary board. The primary board displays the current state of the game, including the pieces that each player has on the board, as well as the number of pieces remaining for each player. The secondary board shows the history of moves made by each player, as well as the number of pieces remaining for each player.

The player can enter moves by using the "MOVE" command followed by the coordinates of the piece they want to move. For example, "MOVE 1 2".

The game also has a feature where players can change the number of pieces they have on the board. When a player places a piece, they can choose to give it to another player or keep it for themselves. When a player chooses to give it to another player, they must use the "CONTROL" command followed by the number of pieces they want to control. For example, "CONTROL 2 3".

There is also a feature where the player can "Reset" the game board to its original state. This can be done using the "RESTART" command.

Overall, this game appears to be a simple yet challenging piece placement game.


```
// Main program
async function main()
{
    print(tab(33) + "H-I-Q\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    for (r = 0; r <= 70; r++)
        b[r] = 0;
    print("HERE IS THE BOARD:\n");
    print("\n");
    print("          !    !    !\n");
    print("         13   14   15\n");
    print("\n");
    print("          !    !    !\n");
    print("         22   23   24\n");
    print("\n");
    print("!    !    !    !    !    !    !\n");
    print("29   30   31   32   33   34   35\n");
    print("\n");
    print("!    !    !    !    !    !    !\n");
    print("38   39   40   41   42   43   44\n");
    print("\n");
    print("!    !    !    !    !    !    !\n");
    print("47   48   49   50   51   52   53\n");
    print("\n");
    print("          !    !    !\n");
    print("         58   59   60\n");
    print("\n");
    print("          !    !    !\n");
    print("         67   68   69\n");
    print("\n");
    print("TO SAVE TYPING TIME, A COMPRESSED VERSION OF THE GAME BOARD\n");
    print("WILL BE USED DURING PLAY.  REFER TO THE ABOVE ONE FOR PEG\n");
    print("NUMBERS.  OK, LET'S BEGIN.\n");
    while (1) {
        // Set up board
        for (r = 1; r <= 9; r++) {
            t[r] = [];
            for (c = 1; c <= 9; c++) {
                if (r == 4 || r == 5 || r == 6 || c == 4 || c == 5 || c == 6 && (r != 1 && c != 1 && r != 9 && c != 9)) {
                    t[r][c] = 5;
                } else {
                    t[r][c] = -5;
                }
            }
        }
        t[5][5] = 0;
        print_board();
        // Init secondary board
        for (w = 1; w <= 33; w++) {
            b[m[w]] = -7;
        }
        b[41] = -3;
        // Input move and check on legality
        do {
            while (1) {
                print("MOVE WHICH PIECE");
                z = parseInt(await input());
                if (b[z] == -7) {
                    print("TO WHERE");
                    p = parseInt(await input());
                    if (p != z
                        && b[p] != 0
                        && b[p] != -7
                        && (z + p) % 2 == 0
                        && (Math.abs(z - p) - 2) * (Math.abs(z - p) - 18) == 0
                        && update_board())
                        break;
                }
                print("ILLEGAL MOVE, TRY AGAIN...\n");
            }
            print_board();
        } while (!check_game_over()) ;
        // Game is over
        print("THE GAME IS OVER.\n");
        print("YOU HAD " + f + " PIECES REMAINING.\n");
        if (f == 1) {
            print("BRAVO!  YOU MADE A PERFECT SCORE!\n");
            print("SAVE THIS PAPER AS A RECORD OF YOUR ACCOMPLISHMENT!\n");
        }
        print("\n");
        print("PLAY AGAIN (YES OR NO)");
        str = await input();
        if (str == "NO")
            break;
    }
    print("\n");
    print("SO LONG FOR NOW.\n");
    print("\n");
}

```

这道题目缺少上下文，无法得知代码的具体作用。一般来说，`main()` 函数是程序的入口点，程序从这里开始执行。在函数内，程序可以执行必要的初始化操作，并调用其他函数来完成具体任务。不同的编程语言和程序可能会有所不同，但是 `main()` 函数通常是程序执行的第一步。


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


# `48_High_IQ/python/High_IQ.py`

这段代码定义了一个名为 `new_board()` 的函数，它使用一个字典类型来存储一个游戏的棋盘。这个字典包含 20 个键值对，每个键表示棋盘上的一段，值为 `"!"`。

具体来说，这个字典的第一行定义了 20 个键值对，每个键都表示一段棋盘。其中，每个键都是合法的，这意味着你可以在游戏中使用这些键来表示不同的棋盘状态。

第二行定义了一个 `Dictionary` 对象，这个对象继承自 Python 的 built-in `Dict` 类型。通过创建这个对象，你可以集中访问上面定义的 20 个键值对，因此可以像操作一个简单的字典一样来访问棋盘的状态。

最后，定义了函数的返回值类型为 `Dict[int, str]`，这意味着这个函数返回的是一个包含所有合法棋盘状态的键值对字典。


```
from typing import Dict


def new_board() -> Dict[int, str]:
    """
    Using a dictionary in python to store the board,
    since we are not including all numbers within a given range.
    """
    return {
        13: "!",
        14: "!",
        15: "!",
        22: "!",
        23: "!",
        24: "!",
        29: "!",
        30: "!",
        31: "!",
        32: "!",
        33: "!",
        34: "!",
        35: "!",
        38: "!",
        39: "!",
        40: "!",
        42: "!",
        43: "!",
        44: "!",
        47: "!",
        48: "!",
        49: "!",
        50: "!",
        51: "!",
        52: "!",
        53: "!",
        58: "!",
        59: "!",
        60: "!",
        67: "!",
        68: "!",
        69: "!",
        41: "O",
    }


```

这段代码定义了一个名为 `print_instructions` 的函数，其返回类型为 `None`。函数体中包含一个打印字符串，其中显示了一个简单的图形化棋盘，棋盘中有 6x6 的方格，并在中心位置放置了 "X" 和 "O" 两个棋子。

具体来说，这段代码的作用是定义了一个函数，该函数会打印一个包含简单图形化棋盘的提示消息，然后调用该函数并传入一个参数 `None`，表示该函数没有返回值。


```
def print_instructions() -> None:
    print(
        """
HERE IS THE BOARD:

          !    !    !
         13   14   15

          !    !    !
         22   23   24

!    !    !    !    !    !    !
29   30   31   32   33   34   35

!    !    !    !    !    !    !
```

这段代码是一个用于保存游戏棋盘的压缩版本。它将棋盘上的所有格子存储在一个二维列表中，然后将其存储为字符串中的行和列的数量。最后，它还包含一个布尔值，表示棋盘是否已压缩。


```
38   39   40   41   42   43   44

!    !    !    !    !    !    !
47   48   49   50   51   52   53

          !    !    !
         58   59   60

          !    !    !
         67   68   69

TO SAVE TYPING TIME, A COMPRESSED VERSION OF THE GAME BOARD
WILL BE USED DURING PLAY.  REFER TO THE ABOVE ONE FOR PEG
NUMBERS.  OK, LET'S BEGIN.
    """
    )


```

这段代码定义了一个名为 `print_board` 的函数，它接收一个名为 `board` 的字典参数，并返回 None。

函数内部的逻辑是：首先，打印出两行垂直居中的空格，然后打印出传入的 `board` 参数中的所有元素。接着，打印出两行垂直居中的空格，然后打印出 `board` 参数中的所有元素。接着，打印出两行垂直居中的空格，然后打印出 `board` 参数中的所有元素。再接着，打印出两行垂直居中的空格，然后打印出 `board` 参数中的所有元素。最后，打印出两行垂直居中的空格，然后打印出 `board` 参数中的所有元素。


```
def print_board(board: Dict[int, str]) -> None:
    """Prints the boards using indexes in the passed parameter"""
    print(" " * 2 + board[13] + board[14] + board[15])
    print(" " * 2 + board[22] + board[23] + board[24])
    print(
        board[29]
        + board[30]
        + board[31]
        + board[32]
        + board[33]
        + board[34]
        + board[35]
    )
    print(
        board[38]
        + board[39]
        + board[40]
        + board[41]
        + board[42]
        + board[43]
        + board[44]
    )
    print(
        board[47]
        + board[48]
        + board[49]
        + board[50]
        + board[51]
        + board[52]
        + board[53]
    )
    print(" " * 2 + board[58] + board[59] + board[60])
    print(" " * 2 + board[67] + board[68] + board[69])


```

这段代码是一个用Python语言编写的游戏，其目的是让用户通过玩这个游戏来锻炼他们的Python编程技能。游戏中有两个主要函数，一个是`play_game()`，另一个是`move()`。

1. `play_game()`函数的作用是创建一个新的游戏棋盘，并让游戏持续运行，直到游戏结束。在游戏运行期间，该函数会不断地打印当前的游戏棋盘状态，并允许用户移动棋盘上的棋子。当游戏结束时，函数会打印用户的得分，并告诉用户如何保存游戏成绩。

2. `move()`函数的作用是让用户输入新的棋子位置，并尝试移动该棋子。如果用户输入的位置是不存在的，函数会打印错误消息并重新询问用户输入。如果用户输入的位置是有效的，函数会将棋子从该位置移动到该位置，并更新棋盘状态。

该游戏还使用了一个辅助函数`is_game_finished()`，该函数用于检查游戏是否已经结束。如果游戏没有结束，函数会继续运行游戏循环，否则游戏循环将停止。

该游戏还使用了一个变量`peg_count`，用于跟踪玩家在游戏中拥有的棋子数量。该变量基于玩家每次移动时，将加入或删除一个棋子。当玩家得分时，游戏会打印一条消息并告诉用户他们的得分。


```
def play_game() -> None:
    # Create new board
    board = new_board()

    # Main game loop
    while not is_game_finished(board):
        print_board(board)
        while not move(board):
            print("ILLEGAL MOVE! TRY AGAIN")

    # Check peg count and print the user's score
    peg_count = 0
    for key in board.keys():
        if board[key] == "!":
            peg_count += 1

    print("YOU HAD " + str(peg_count) + " PEGS REMAINING")

    if peg_count == 1:
        print("BRAVO! YOU MADE A PERFECT SCORE!")
        print("SAVE THIS PAPER AS A RECORD OF YOUR ACCOMPLISHMENT!")


```

以上代码是一个 Python 函数，其目的是在井字棋棋盘上移动一个棋子，并判断该移动是否成功。

函数接收一个名为 `board` 的字典，其中包含棋子的位置信息，以及一个字符类型的输入，表示要移动的棋子是“北边”还是“南边”。

函数首先检查输入是否合法，如果输入不是数字类型，则返回 `False`。

接下来，函数会获取用户输入的起始棋子位置，并将其转换成整数类型。然后，函数会遍历 `board` 中的所有棋子，检查起始位置是否在其中的某个位置，如果是，则返回移动是否成功。

如果起始位置不在 `board` 中，则函数会返回 `False`。如果移动成功，则函数会返回 `True`。


```
def move(board: Dict[int, str]) -> bool:
    """Queries the user to move. Returns false if the user puts in an invalid input or move, returns true if the move was successful"""
    start_input = input("MOVE WHICH PIECE? ")

    if not start_input.isdigit():
        return False

    start = int(start_input)

    if start not in board or board[start] != "!":
        return False

    end_input = input("TO WHERE? ")

    if not end_input.isdigit():
        return False

    end = int(end_input)

    if end not in board or board[end] != "O":
        return False

    difference = abs(start - end)
    center = int((end + start) / 2)
    if (
        (difference == 2 or difference == 18)
        and board[end] == "O"
        and board[center] == "!"
    ):
        board[start] = "O"
        board[center] = "O"
        board[end] = "!"
        return True
    else:
        return False


```

这段代码是一个Python程序，名为“main”。程序的主要目的是让用户进行一次有趣的猜数字游戏。下面是具体的游戏流程：

1. 输出游戏开始信息，包括游戏板的尺寸和一些说明。
2. 发送游戏开始前的说明信息。
3. 发送游戏开始，然后等待玩家的操作。
4. 发送游戏结束后的信息，包括游戏是否结束，以及是否可以重新开始等。
5. 判断游戏是否结束，如果游戏结束，那么打印一些提示信息，并返回True。否则，返回False。
6. 检查游戏是否可以在指定位置移动，如果可以，那么进行移动，并返回True；否则，返回False。

具体来说，这段代码实现了一个简单的游戏，可以猜测玩家输入的数字。游戏板是一个二维列表，包含了从1到9的数字，以及一个感叹号（!"）代表这个位置为不可移动。程序首先输出游戏开始信息，然后等待玩家的操作。在玩家操作后，程序会判断游戏是否结束，并输出相应的信息。


```
def main() -> None:
    print(" " * 33 + "H-I-Q")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print_instructions()
    play_game()


def is_game_finished(board) -> bool:
    """Check all locations and whether or not a move is possible at that location."""
    for pos in board.keys():
        if board[pos] == "!":
            for space in [1, 9]:
                # Checks if the next location has a peg
                next_to_peg = ((pos + space) in board) and board[pos + space] == "!"
                # Checks both going forward (+ location) or backwards (-location)
                has_movable_space = (
                    not ((pos - space) in board and board[pos - space] == "!")
                ) or (
                    not ((pos + space * 2) in board and board[pos + space * 2] == "!")
                )
                if next_to_peg and has_movable_space:
                    return False
    return True


```

这段代码是一个条件判断语句，它的作用是在程序运行时检查当前操作系统的名称是否为 "Windows"。"__name__" 是一个特殊的字符串，用于鉴定当前操作系统的名称。如果当前操作系统的名称是 "Windows"，则执行 "main()" 函数，否则程序不会执行 "main()" 函数。


```
if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)

[Implementation](./High_IQ.py) by [Thomas Kwashnak](https://github.com/LittleTealeaf)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Hockey

This is a simulation of a ice hockey game. The computer, in this case, moderates and referees the pay between two human opponents. Of course, one person could play both sides.

The program asks for team names, player names, and even the name of the referee. Four types of shot are permitted and a shot may be aimed at one of four areas. You are also asked about passing. The game is very comprehensive with lots of action, face offs, blocks, passes, 4 on 2 situations, and so on. Unfortunately there are no penalties.

The original author is Robert Puopolo; modifications by Steve North of Creative Computing.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=88)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=103)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Known Bugs

- An apparent missing line 430 causes the code to fall through from the "FLIPS A WRISTSHOT" case directly to the "BACKHANDS ONE" case.
- The author consistently misspells the verb "lets" (writing it like the contraction "let's"), while having no trouble with "leads", "gets", "hits", etc.

#### Porting Notes

(please note any difficulties or challenges in porting here)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `49_Hockey/javascript/hockey.js`

该代码是一个Javascript到BASIC的转换函数，允许将一个字符串打印到页面上，并允许用户输入一行字符。

具体来说，代码中的`print()`函数将一个字符串打印到页面上，并在页面上创建了一个新的文本节点，将该节点附加到`document.getElementById("output")`元素上。

`input()`函数是一个Promise，它等待用户输入一行字符。它创建了一个`<INPUT>`元素，设置其`type`属性为"text"，并设置其`length`属性为"50"。它将该元素添加到页面上，并为该元素添加了一个键盘事件监听器，以便在用户按下回车键时捕获输入的值。当用户输入一行字符时，该函数将其附加到打印出的字符串中，并打印到页面上。

因此，该代码的主要目的是创建一个允许用户输入一行字符并将其打印到页面上的人工智能助手。


```
// HOCKEY
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

这段代码定义了一个名为“tab”的函数，接受一个名为“space”的整数参数。这个函数的主要目的是返回一个由字符“ space ”和它前面的所有空格组成的字符串。

代码中定义了四个变量：str、bs、ha和ta。str变量用于存储返回的字符串，bs、ha和ta变量用于存储字符串中的每一项。这些变量都是在循环中创建的，通过while循环从0开始，每次增加1，当space变量的值大于0时进行循环。

在循环体内，str会通过添加“ space ”前的空格来构建字符串。每次循环结束后，str都会被赋值为“ ”（一个空字符串）。

最终，返回的字符串就是由space和它前面的所有空格组成的字符串。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var as = [];
var bs = [];
var ha = [];
var ta = [];
var t1 = [];
var t2 = [];
var t3 = [];

```

This appears to be a programming language for a basketball game. It parses a user's input and generates a scoring summary of the game.

The program starts by defining some constants for the initial score, which is not provided in the code but is likely to be determined by the game.

Then, the program enters a loop for each half of the game, printing out the score, goals, assists, and shots taken by the opposing team in that half of the game.

Finally, the program prints out the final score and a summary of the game.

Note: This is just an example program and may not be complete or functional for all basketball games.


```
// Main program
async function main()
{
    print(tab(33) + "HOCKEY\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // Robert Puopolo Alg. 1 140 McCowan 6/7/73 Hockey
    for (c = 0; c <= 20; c++)
        ha[c] = 0;
    for (c = 1; c <= 5; c++) {
        ta[c] = 0;
        t1[c] = 0;
        t2[c] = 0;
        t3[c] = 0;
    }
    x = 1;
    print("\n");
    print("\n");
    print("\n");
    while (1) {
        print("WOULD YOU LIKE THE INSTRUCTIONS");
        str = await input();
        print("\n");
        if (str == "YES" || str == "NO")
            break;
        print("ANSWER YES OR NO!!\n");
    }
    if (str == "YES") {
        print("\n");
        print("THIS IS A SIMULATED HOCKEY GAME.\n");
        print("QUESTION     RESPONSE\n");
        print("PASS         TYPE IN THE NUMBER OF PASSES YOU WOULD\n");
        print("             LIKE TO MAKE, FROM 0 TO 3.\n");
        print("SHOT         TYPE THE NUMBER CORRESPONDING TO THE SHOT\n");
        print("             YOU WANT TO MAKE.  ENTER:\n");
        print("             1 FOR A SLAPSHOT\n");
        print("             2 FOR A WRISTSHOT\n");
        print("             3 FOR A BACKHAND\n");
        print("             4 FOR A SNAP SHOT\n");
        print("AREA         TYPE IN THE NUMBER CORRESPONDING TO\n");
        print("             THE AREA YOU ARE AIMING AT.  ENTER:\n");
        print("             1 FOR UPPER LEFT HAND CORNER\n");
        print("             2 FOR UPPER RIGHT HAND CORNER\n");
        print("             3 FOR LOWER LEFT HAND CORNER\n");
        print("             4 FOR LOWER RIGHT HAND CORNER\n");
        print("\n");
        print("AT THE START OF THE GAME, YOU WILL BE ASKED FOR THE NAMES\n");
        print("OF YOUR PLAYERS.  THEY ARE ENTERED IN THE ORDER: \n");
        print("LEFT WING, CENTER, RIGHT WING, LEFT DEFENSE,\n");
        print("RIGHT DEFENSE, GOALKEEPER.  ANY OTHER INPUT REQUIRED WILL\n");
        print("HAVE EXPLANATORY INSTRUCTIONS.\n");
    }
    print("ENTER THE TWO TEAMS");
    str = await input();
    c = str.indexOf(",");
    as[7] = str.substr(0, c);
    bs[7] = str.substr(c + 1);
    print("\n");
    do {
        print("ENTER THE NUMBER OF MINUTES IN A GAME");
        t6 = parseInt(await input());
        print("\n");
    } while (t6 < 1) ;
    print("\n");
    print("WOULD THE " + as[7] + " COACH ENTER HIS TEAM\n");
    print("\n");
    for (i = 1; i <= 6; i++) {
        print("PLAYER " + i + " ");
        as[i] = await input();
    }
    print("\n");
    print("WOULD THE " + bs[7] + " COACH DO THE SAME\n");
    print("\n");
    for (t = 1; t <= 6; t++) {
        print("PLAYER " + t + " ");
        bs[t] = await input();
    }
    print("\n");
    print("INPUT THE REFEREE FOR THIS GAME");
    rs = await input();
    print("\n");
    print(tab(10) + as[7] + " STARTING LINEUP\n");
    for (t = 1; t <= 6; t++) {
        print(as[t] + "\n");
    }
    print("\n");
    print(tab(10) + bs[7] + " STARTING LINEUP\n");
    for (t = 1; t <= 6; t++) {
        print(bs[t] + "\n");
    }
    print("\n");
    print("WE'RE READY FOR TONIGHTS OPENING FACE-OFF.\n");
    print(rs + " WILL DROP THE PUCK BETWEEN " + as[2] + " AND " + bs[2] + "\n");
    s2 = 0;
    s3 = 0;
    for (l = 1; l <= t6; l++) {
        c = Math.floor(2 * Math.random()) + 1;
        if (c == 1)
            print(as[7] + " HAS CONTROL OF THE PUCK\n");
        else
            print(bs[7] + " HAS CONTROL.\n");
        do {

            print("PASS");
            p = parseInt(await input());
            for (n = 1; n <= 3; n++)
                ha[n] = 0;
        } while (p < 0 || p > 3) ;
        do {
            for (j = 1; j <= p + 2; j++)
                ha[j] = Math.floor(5 * Math.random()) + 1;
        } while (ha[j - 1] == ha[j - 2] || (p + 2 >= 3 && (ha[j - 1] == ha[j - 3] || ha[j - 2] == ha[j - 3]))) ;
        if (p == 0) {
            while (1) {
                print("SHOT");
                s = parseInt(await input());
                if (s >= 1 && s <= 4)
                    break;
            }
            if (c == 1) {
                print(as[ha[j - 1]]);
                g = ha[j - 1];
                g1 = 0;
                g2 = 0;
            } else {
                print(bs[ha[j - 1]]);
                g2 = 0;
                g2 = 0;
                g = ha[j - 1];
            }
            switch (s) {
                case 1:
                    print(" LET'S A BOOMER GO FROM THE RED LINE!!\n");
                    z = 10;
                    break;
                case 2:
                    print(" FLIPS A WRISTSHOT DOWN THE ICE\n");
                    // Probable missing line 430 in original
                case 3:
                    print(" BACKHANDS ONE IN ON THE GOALTENDER\n");
                    z = 25;
                    break;
                case 4:
                    print(" SNAPS A LONG FLIP SHOT\n");
                    z = 17;
                    break;
            }
        } else {
            if (c == 1) {
                switch (p) {
                    case 1:
                        print(as[ha[j - 2]] + " LEADS " + as[ha[j - 1]] + " WITH A PERFECT PASS.\n");
                        print(as[ha[j - 1]] + " CUTTING IN!!!\n");
                        g = ha[j - 1];
                        g1 = ha[j - 2];
                        g2 = 0;
                        z1 = 3;
                        break;
                    case 2:
                        print(as[ha[j - 2]] + " GIVES TO A STREAKING " + as[ha[j - 1]] + "\n");
                        print(as[ha[j - 3]] + " COMES DOWN ON " + bs[5] + " AND " + bs[4] + "\n");
                        g = ha[j - 3];
                        g1 = ha[j - 1];
                        g2 = ha[j - 2];
                        z1 = 2;
                        break;
                    case 3:
                        print("OH MY GOD!! A ' 4 ON 2 ' SITUATION\n");
                        print(as[ha[j - 3]] + " LEADS " + as[ha[j - 2]] + "\n");
                        print(as[ha[j - 2]] + " IS WHEELING THROUGH CENTER.\n");
                        print(as[ha[j - 2]] + " GIVES AND GOEST WITH " + as[ha[j - 1]] + "\n");
                        print("PRETTY PASSING!\n");
                        print(as[ha[j - 1]] + " DROPS IT TO " + as[ha[j - 4]] + "\n");
                        g = ha[j - 4];
                        g1 = ha[j - 1];
                        g2 = ha[j - 2];
                        z1 = 1;
                        break;
                }
            } else {
                switch (p) {
                    case 1:
                        print(bs[ha[j - 1]] + " HITS " + bs[ha[j - 2]] + " FLYING DOWN THE LEFT SIDE\n");
                        g = ha[j - 2];
                        g1 = ha[j - 1];
                        g2 = 0;
                        z1 = 3;
                        break;
                    case 2:
                        print("IT'S A ' 3 ON 2 '!\n");
                        print("ONLY " + as[4] + " AND " + as[5] + " ARE BACK.\n");
                        print(bs[ha[j - 2]] + " GIVES OFF TO " + bs[ha[j - 1]] + "\n");
                        print(bs[ha[j - 1]] + " DROPS TO " + bs[ha[j - 3]] + "\n");
                        g = ha[j - 3];
                        g1 = ha[j - 1];
                        g2 = ha[j - 2];
                        z1 = 2;
                        break;
                    case 3:
                        print(" A '3 ON 2 ' WITH A ' TRAILER '!\n");
                        print(bs[ha[j - 4]] + " GIVES TO " + bs[ha[j - 2]] + " WHO SHUFFLES IT OFF TO\n");
                        print(bs[ha[j - 1]] + " WHO FIRES A WING TO WING PASS TO \n");
                        print(bs[ha[j - 3]] + " AS HE CUTS IN ALONE!!\n");
                        g = ha[j - 3];
                        g1 = ha[j - 1];
                        g2 = ha[j - 2];
                        z1 = 1;
                        break;
                }
            }
            do {
                print("SHOT");
                s = parseInt(await input());
            } while (s < 1 || s > 4) ;
            if (c == 1)
                print(as[g]);
            else
                print(bs[g]);
            switch (s) {
                case 1:
                    print(" LET'S A BIG SLAP SHOT GO!!\n");
                    z = 4;
                    z += z1;
                    break;
                case 2:
                    print(" RIPS A WRIST SHOT OFF\n");
                    z = 2;
                    z += z1;
                    break;
                case 3:
                    print(" GETS A BACKHAND OFF\n");
                    z = 3;
                    z += z1;
                    break;
                case 4:
                    print(" SNAPS OFF A SNAP SHOT\n");
                    z = 2;
                    z += z1;
                    break;
            }
        }
        do {
            print("AREA");
            a = parseInt(await input());
        } while (a < 1 || a > 4) ;
        if (c == 1)
            s2++;
        else
            s3++;
        a1 = Math.floor(4 * Math.random()) + 1;
        if (a == a1) {
            while (1) {
                ha[20] = Math.floor(100 * Math.random()) + 1;
                if (ha[20] % z != 0)
                    break;
                a2 = Math.floor(100 * Math.random()) + 1;
                if (a2 % 4 == 0) {
                    if (c == 1)
                        print("SAVE " + bs[6] + " --  REBOUND\n");
                    else
                        print("SAVE " + as[6] + " --  FOLLOW up\n");
                    continue;
                } else {
                    a1 = a + 1;  // So a != a1
                }
            }
            if (ha[20] % z != 0) {
                if (c == 1) {
                    print("GOAL " + as[7] + "\n");
                    ha[9]++;
                } else {
                    print("SCORE " + bs[7] + "\n");
                    ha[8]++;
                }
                // Bells in origninal
                print("\n");
                print("SCORE: ");
                if (ha[8] <= ha[9]) {
                    print(as[7] + ": " + ha[9] + "\t" + bs[7] + ": " + ha[8] + "\n");
                } else {
                    print(bs[7] + ": " + ha[8] + "\t" + as[7] + ": " + ha[9] + "\n");
                }
                if (c == 1) {
                    print("GOAL SCORED BY: " + as[g] + "\n");
                    if (g1 != 0) {
                        if (g2 != 0) {
                            print(" ASSISTED BY: " + as[g1] + " AND " + as[g2] + "\n");
                        } else {
                            print(" ASSISTED BY: " + as[g1] + "\n");
                        }
                    } else {
                        print(" UNASSISTED.\n");
                    }
                    ta[g]++;
                    t1[g1]++;
                    t1[g2]++;
                    // 1540
                } else {
                    print("GOAL SCORED BY: " + bs[g] + "\n");
                    if (g1 != 0) {
                        if (g2 != 0) {
                            print(" ASSISTED BY: " + bs[g1] + " AND " + bs[g2] + "\n");
                        } else {
                            print(" ASSISTED BY: " + bs[g1] + "\n");
                        }
                    } else {
                        print(" UNASSISTED.\n");
                    }
                    t2[g]++;
                    t3[g1]++;
                    t3[g2]++;
                    // 1540
                }
            }
        }
        if (a != a1) {
            s1 = Math.floor(6 * Math.random()) + 1;
            if (c == 1) {
                switch (s1) {
                    case 1:
                        print("KICK SAVE AND A BEAUTY BY " + bs[6] + "\n");
                        print("CLEARED OUT BY " + bs[3] + "\n");
                        l--;
                        continue;
                    case 2:
                        print("WHAT A SPECTACULAR GLOVE SAVE BY " + bs[6] + "\n");
                        print("AND " + bs[6] + " GOLFS IT INTO THE CROWD\n");
                        break;
                    case 3:
                        print("SKATE SAVE ON A LOW STEAMER BY " + bs[6] + "\n");
                        l--;
                        continue;
                    case 4:
                        print("PAD SAVE BY " + bs[6] + " OFF THE STICK\n");
                        print("OF " + as[g] + " AND " + bs[6] + " COVERS UP\n");
                        break;
                    case 5:
                        print("WHISTLES ONE OVER THE HEAD OF " + bs[6] + "\n");
                        l--;
                        continue;
                    case 6:
                        print(bs[6] + " MAKES A FACE SAVE!! AND HE IS HURT\n");
                        print("THE DEFENSEMAN " + bs[5] + " COVERS UP FOR HIM\n");
                        break;
                }
            } else {
                switch (s1) {
                    case 1:
                        print("STICK SAVE BY " + as[6] +"\n");
                        print("AND CLEARED OUT BY " + as[4] + "\n");
                        l--;
                        continue;
                    case 2:
                        print("OH MY GOD!! " + bs[g] + " RATTLES ONE OFF THE POST\n");
                        print("TO THE RIGHT OF " + as[6] + " AND " + as[6] + " COVERS ");
                        print("ON THE LOOSE PUCK!\n");
                        break;
                    case 3:
                        print("SKATE SAVE BY " + as[6] + "\n");
                        print(as[6] + " WHACKS THE LOOSE PUCK INTO THE STANDS\n");
                        break;
                    case 4:
                        print("STICK SAVE BY " + as[6] + " AND HE CLEARS IT OUT HIMSELF\n");
                        l--;
                        continue;
                    case 5:
                        print("KICKED OUT BY " + as[6] + "\n");
                        print("AND IT REBOUNDS ALL THE WAY TO CENTER ICE\n");
                        l--;
                        continue;
                    case 6:
                        print("GLOVE SAVE " + as[6] + " AND HE HANGS ON\n");
                        break;
                }
            }
        }
        print("AND WE'RE READY FOR THE FACE-OFF\n");
    }
    // Bells chime
    print("THAT'S THE SIREN\n");
    print("\n");
    print(tab(15) + "FINAL SCORE:\n");
    if (ha[8] <= ha[9]) {
        print(as[7] + ": " + ha[9] + "\t" + bs[7] + ": " + ha[8] + "\n");
    } else {
        print(bs[7] + ": " + ha[8] + "\t" + as[7] + ": " + ha[9] + "\n");
    }
    print("\n");
    print(tab(10) + "SCORING SUMMARY\n");
    print("\n");
    print(tab(25) + as[7] + "\n");
    print("\tNAME\tGOALS\tASSISTS\n");
    print("\t----\t-----\t-------\n");
    for (i = 1; i <= 5; i++) {
        print("\t" + as[i] + "\t" + ta[i] + "\t" + t1[i] + "\n");
    }
    print("\n");
    print(tab(25) + bs[7] + "\n");
    print("\tNAME\tGOALS\tASSISTS\n");
    print("\t----\t-----\t-------\n");
    for (t = 1; t <= 5; t++) {
        print("\t" + bs[t] + "\t" + t2[t] + "\t" + t3[t] + "\n");
    }
    print("\n");
    print("SHOTS ON NET\n");
    print(as[7] + ": " + s2 + "\n");
    print(bs[7] + ": " + s3 + "\n");
}

```

这是 C 语言中的一个程序，名为 "main"。程序的作用是启动 C 语言编译器，然后编写并运行一个名为 "test.c" 的源程序。

"main" 函数是 C 语言中的一个全局函数，它提供了一个入口点，让程序从此处开始执行。当 "main" 函数被调用时，它将首先查找并运行 "test.c" 中的源程序。

"main" 函数通常会包含一些全局变量，这些变量在程序运行期间保存在内存中。这些变量以及程序的其他部分都可以被用户通过输入命令行参数进行设置。


```
main();

```