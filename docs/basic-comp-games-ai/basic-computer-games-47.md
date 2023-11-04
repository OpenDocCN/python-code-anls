# BasicComputerGames源码解析 47

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `44_Hangman/java/Hangman.java`

This appears to be a Java program that outputs a simple "hangman" picture of a person. The program has several functions that are called in the main method.

The hangman picture is created using a 2D array called "hangmanPicture". Each element of the array is either a space (" ") or a character代表一种鞋子。

The program also has two helper functions, printDiscoveredLetters and printLettersUsed. These functions are used to print the letters in the "hangmanPicture" array and the letters used by the user, respectively.

The main method starts by printing an intro message and then loops through each character in the "hangmanPicture" array.

It is not clear from the code what the purpose of each character in the "hangmanPicture" array is, or how it is being used. It is possible that the character is being used to form a valid "hangman" character, but without more information it is not possible to determine this.


```
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Scanner;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * HANGMAN
 *
 * Converted from BASIC to Java by Aldrin Misquitta (@aldrinm)
 */

public class Hangman {

	//50 word list
	private final static List<String> words = List.of(
			"GUM", "SIN", "FOR", "CRY", "LUG", "BYE", "FLY",
			"UGLY", "EACH", "FROM", "WORK", "TALK", "WITH", "SELF",
			"PIZZA", "THING", "FEIGN", "FIEND", "ELBOW", "FAULT", "DIRTY",
			"BUDGET", "SPIRIT", "QUAINT", "MAIDEN", "ESCORT", "PICKAX",
			"EXAMPLE", "TENSION", "QUININE", "KIDNEY", "REPLICA", "SLEEPER",
			"TRIANGLE", "KANGAROO", "MAHOGANY", "SERGEANT", "SEQUENCE",
			"MOUSTACHE", "DANGEROUS", "SCIENTIST", "DIFFERENT", "QUIESCENT",
			"MAGISTRATE", "ERRONEOUSLY", "LOUDSPEAKER", "PHYTOTOXIC",
			"MATRIMONIAL", "PARASYMPATHOMIMETIC", "THIGMOTROPISM");

	public static void main(String[] args) {
		Scanner scan = new Scanner(System.in);

		printIntro();

		int[] usedWords = new int[50];
		int roundNumber = 1;
		int totalWords = words.size();
		boolean continueGame = false;

		do {
			if (roundNumber > totalWords) {
				System.out.println("\nYOU DID ALL THE WORDS!!");
				break;
			}

			int randomWordIndex;
			do {
				randomWordIndex = ((int) (totalWords * Math.random())) + 1;
			} while (usedWords[randomWordIndex] == 1);
			usedWords[randomWordIndex] = 1;

			boolean youWon = playRound(scan, words.get(randomWordIndex - 1));
			if (!youWon) {
				System.out.print("\nYOU MISSED THAT ONE.  DO YOU WANT ANOTHER WORD? ");
			} else {
				System.out.print("\nWANT ANOTHER WORD? ");
			}
			final String anotherWordChoice = scan.next();

			if (anotherWordChoice.toUpperCase().equals("YES") || anotherWordChoice.toUpperCase().equals("Y")) {
				continueGame = true;
			}
			roundNumber++;
		} while (continueGame);

		System.out.println("\nIT'S BEEN FUN!  BYE FOR NOW.");
	}

	private static boolean playRound(Scanner scan, String word) {
		char[] letters;
		char[] discoveredLetters;
		int misses = 0;
		Set<Character> lettersUsed = new LinkedHashSet<>();//LinkedHashSet maintains the order of characters inserted

		String[][] hangmanPicture = new String[12][12];
		//initialize the hangman picture
		for (int i = 0; i < hangmanPicture.length; i++) {
			for (int j = 0; j < hangmanPicture[i].length; j++) {
				hangmanPicture[i][j] = " ";
			}
		}
		for (int i = 0; i < hangmanPicture.length; i++) {
			hangmanPicture[i][0] = "X";
		}
		for (int i = 0; i < 7; i++) {
			hangmanPicture[0][i] = "X";
		}
		hangmanPicture[1][6] = "X";

		int totalWordGuesses = 0; //guesses

		int len = word.length();
		letters = word.toCharArray();

		discoveredLetters = new char[len];
		Arrays.fill(discoveredLetters, '-');

		boolean validNextGuess = false;
		char guessLetter = ' ';

		while (misses < 10) {
			while (!validNextGuess) {
				printLettersUsed(lettersUsed);
				printDiscoveredLetters(discoveredLetters);

				System.out.print("WHAT IS YOUR GUESS? ");
				var tmpRead = scan.next();
				guessLetter = Character.toUpperCase(tmpRead.charAt(0));
				if (lettersUsed.contains(guessLetter)) {
					System.out.println("YOU GUESSED THAT LETTER BEFORE!");
				} else {
					lettersUsed.add(guessLetter);
					totalWordGuesses++;
					validNextGuess = true;
				}
			}

			if (word.indexOf(guessLetter) >= 0) {
				//replace all occurrences in D$ with G$
				for (int i = 0; i < letters.length; i++) {
					if (letters[i] == guessLetter) {
						discoveredLetters[i] = guessLetter;
					}
				}
				//check if the word is fully discovered
				boolean isWordDiscovered = true;
				for (char discoveredLetter : discoveredLetters) {
					if (discoveredLetter == '-') {
						isWordDiscovered = false;
						break;
					}
				}
				if (isWordDiscovered) {
					System.out.println("YOU FOUND THE WORD!");
					return true;
				}

				printDiscoveredLetters(discoveredLetters);
				System.out.print("WHAT IS YOUR GUESS FOR THE WORD? ");
				final String wordGuess = scan.next();
				if (wordGuess.toUpperCase().equals(word)) {
					System.out.printf("RIGHT!!  IT TOOK YOU %s GUESSES!", totalWordGuesses);
					return true;
				} else {
					System.out.println("WRONG.  TRY ANOTHER LETTER.");
				}
			} else {
				misses = misses + 1;
				System.out.println("\n\nSORRY, THAT LETTER ISN'T IN THE WORD.");
				drawHangman(misses, hangmanPicture);
			}
			validNextGuess = false;
		}

		System.out.printf("SORRY, YOU LOSE.  THE WORD WAS %s", word);
		return false;
	}

	private static void drawHangman(int m, String[][] hangmanPicture) {
		switch (m) {
			case 1:
				System.out.println("FIRST, WE DRAW A HEAD");
				hangmanPicture[2][5] = "-";
				hangmanPicture[2][6] = "-";
				hangmanPicture[2][7] = "-";
				hangmanPicture[3][4] = "(";
				hangmanPicture[3][5] = ".";
				hangmanPicture[3][7] = ".";
				hangmanPicture[3][8] = ")";
				hangmanPicture[4][5] = "-";
				hangmanPicture[4][6] = "-";
				hangmanPicture[4][7] = "-";
				break;
			case 2:
				System.out.println("NOW WE DRAW A BODY.");
				for (var i = 5; i <= 8; i++) {
					hangmanPicture[i][6] = "X";
				}
				break;
			case 3:
				System.out.println("NEXT WE DRAW AN ARM.");
				for (int i = 3; i <= 6; i++) {
					hangmanPicture[i][i - 1] = "\\";
				}
				break;
			case 4:
				System.out.println("THIS TIME IT'S THE OTHER ARM.");
				hangmanPicture[3][10] = "/";
				hangmanPicture[4][9] = "/";
				hangmanPicture[5][8] = "/";
				hangmanPicture[6][7] = "/";
				break;
			case 5:
				System.out.println("NOW, LET'S DRAW THE RIGHT LEG.");
				hangmanPicture[9][5] = "/";
				hangmanPicture[10][4] = "/";
				break;
			case 6:
				System.out.println("THIS TIME WE DRAW THE LEFT LEG.");
				hangmanPicture[9][7] = "\\";
				hangmanPicture[10][8] = "\\";
				break;
			case 7:
				System.out.println("NOW WE PUT UP A HAND.");
				hangmanPicture[2][10] = "\\";
				break;
			case 8:
				System.out.println("NEXT THE OTHER HAND.");
				hangmanPicture[2][2] = "/";
				break;
			case 9:
				System.out.println("NOW WE DRAW ONE FOOT");
				hangmanPicture[11][9] = "\\";
				hangmanPicture[11][10] = "-";
				break;
			case 10:
				System.out.println("HERE'S THE OTHER FOOT -- YOU'RE HUNG!!");
				hangmanPicture[11][2] = "-";
				hangmanPicture[11][3] = "/";
				break;
		}
		for (int i = 0; i <= 11; i++) {
			for (int j = 0; j <= 11; j++) {
				System.out.print(hangmanPicture[i][j]);
			}
			System.out.print("\n");
		}

	}

	private static void printDiscoveredLetters(char[] D$) {
		System.out.println(new String(D$));
		System.out.println("\n");
	}

	private static void printLettersUsed(Set<Character> lettersUsed) {
		System.out.println("\nHERE ARE THE LETTERS YOU USED:");
		System.out.println(lettersUsed.stream()
				.map(Object::toString).collect(Collectors.joining(",")));
		System.out.println("\n");
	}

	private static void printIntro() {
		System.out.println("                                HANGMAN");
		System.out.println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
		System.out.println("\n\n\n");
	}

}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `44_Hangman/javascript/hangman.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。

`print` 函数的作用是在文档中创建一个文本节点（`<li>` 标签），并将传入的参数（字符串 `str`）将其添加到该节点中。它的代码中，使用了 `document.getElementById` 获取了一个名为 "output" 的元素，将其作为文本节点添加到了文档中。

`input` 函数的作用是获取用户输入的字符串（`<input>` 标签的 `value` 属性），并将其存储在变量 `input_str` 中。它使用了 `document.createElement` 创建了一个 `<INPUT>` 元素，设置了其 `type` 属性为 `text`，`length` 属性为 `50`（表示最大输入字符数为 50），并将其添加到了文档中。然后，它绑定了 `keydown` 事件，当用户按下了键盘上的 `13` 键时，它将获取到输入框中的字符串，并将其存储在 `input_str` 变量中。最后，它通过调用 `print` 函数将 `input_str` 的字符串打印到页面上。

总的来说，这两个函数都是用来处理用户输入的，但它们有着不同的用途和作用方式。


```
// HANGMAN
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

这段代码定义了一个名为 `tab` 的函数，它接受一个参数 `space`，它会是一个整数，并且只有在 `space` 大于 0 时才会执行以下代码。

1. 在函数内部，定义了一个字符串变量 `str`，并使用一个变量 `space` 来跟踪剩余的空间数量。
2. 在循环中，使用 `str` 变量中的每个字符来填充字符串，每次增加一个额外的空格，直到将 `space` 变量减少到 0 时，循环结束。
3. 返回生成的字符串，覆盖了原字符串变量 `str`。
4. 在三个打印语句中，使用了 `print` 函数来输出字符串。第一个参数是 `tab(32)`，即将 `space` 变量设置为 32 时，函数生成的字符串，第二个参数添加了一个 `HANGMAN` 字符，并在其后面添加了一个空格。第二个和第三个打印语句的参数与第一个相同，但是将输出结果换行。
5. 在一个循环语句中，创建了一个名为 `pa` 的字符数组，但是没有给其赋值。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

print(tab(32) + "HANGMAN\n");
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
print("\n");
print("\n");
print("\n");

var pa = [];
```

这段代码定义了四个变量，分别命名为"words"、"da"、"na"和"ua"，它们都是字符串类型的数组。接着，定义了一个包含31个单词的字符串变量"words"。


```
var la = [];
var da = [];
var na = [];
var ua = [];

var words = ["GUM","SIN","FOR","CRY","LUG","BYE","FLY",
             "UGLY","EACH","FROM","WORK","TALK","WITH","SELF",
             "PIZZA","THING","FEIGN","FIEND","ELBOW","FAULT","DIRTY",
             "BUDGET","SPIRIT","QUAINT","MAIDEN","ESCORT","PICKAX",
             "EXAMPLE","TENSION","QUININE","KIDNEY","REPLICA","SLEEPER",
             "TRIANGLE","KANGAROO","MAHOGANY","SERGEANT","SEQUENCE",
             "MOUSTACHE","DANGEROUS","SCIENTIST","DIFFERENT","QUIESCENT",
             "MAGISTRATE","ERRONEOUSLY","LOUDSPEAKER","PHYTOTOXIC",
             "MATRIMONIAL","PARASYMPATHOMIMETIC","THIGMOTROPISM"];

```

In this program, the word to be guessed is stored in a 2D array called `pa`. The program uses several different approaches to try to guess the word, including checking for repeated characters, using positional indexing to check for words ending in a certain pattern, and asking the user to guess a word by input.

The first approach to guessing the word is to check for repeated characters. If a position in the array is repeated, the program will know that the word ends there.

The second approach is to check for words ending in a certain pattern. This is done by first checking the characters in the current word and then checking the characters in the surrounding characters. If any characters match the pattern, the program knows that the word ends there.

Finally, the program will ask the user to guess a word by input. The user will be prompted to enter a word, and the program will try to determine whether the word is correct or not.

Overall, this program is designed to guess a word by input, but it uses several different approaches to try to guess the word.


```
// Main control section
async function main()
{
    c = 1;
    n = 50;
    while (1) {
        for (i = 1; i <= 20; i++)
            da[i] = "-";
        for (i = 1; i <= n; i++)
            ua[i] = 0;
        m = 0;
        ns = "";
        for (i = 1; i <= 12; i++) {
            pa[i] = [];
            for (j = 1; j <= 12; j++) {
                pa[i][j] = " ";
            }
        }
        for (i = 1; i <= 12; i++) {
            pa[i][1] = "X";
        }
        for (i = 1; i <= 7; i++) {
            pa[1][i] = "X";
        }
        pa[2][7] = "X";
        if (c >= n) {
            print("YOU DID ALL THE WORDS!!\n");
            break;
        }
        do {
            q = Math.floor(n * Math.random()) + 1;
        } while (ua[q] == 1) ;
        ua[q] = 1;
        c++;
        t1 = 0;
        as = words[q - 1];
        l = as.length;
        for (i = 1; i <= as.length; i++)
            la[i] = as[i - 1];
        while (1) {
            while (1) {
                print("HERE ARE THE LETTERS YOU USED:\n");
                print(ns + "\n");
                print("\n");
                for (i = 1; i <= l; i++) {
                    print(da[i]);
                }
                print("\n");
                print("\n");
                print("WHAT IS YOUR GUESS");
                str = await input();
                if (ns.indexOf(str) != -1) {
                    print("YOU GUESSED THAT LETTER BEFORE!\n");
                } else {
                    break;
                }
            }
            ns += str;
            t1++;
            r = 0;
            for (i = 1; i <= l; i++) {
                if (la[i] == str) {
                    da[i] = str;
                    r++;
                }
            }
            if (r == 0) {
                m++;
                print("\n");
                print("\n");
                print("SORRY, THAT LETTER ISN'T IN THE WORD.\n");
                switch (m) {
                    case 1:
                        print("FIRST, WE DRAW A HEAD\n");
                        break;
                    case 2:
                        print("NOW WE DRAW A BODY.\n");
                        break;
                    case 3:
                        print("NEXT WE DRAW AN ARM.\n");
                        break;
                    case 4:
                        print("THIS TIME IT'S THE OTHER ARM.\n");
                        break;
                    case 5:
                        print("NOW, LET'S DRAW THE RIGHT LEG.\n");
                        break;
                    case 6:
                        print("THIS TIME WE DRAW THE LEFT LEG.\n");
                        break;
                    case 7:
                        print("NOW WE PUT UP A HAND.\n");
                        break;
                    case 8:
                        print("NEXT THE OTHER HAND.\n");
                        break;
                    case 9:
                        print("NOW WE DRAW ONE FOOT.\n");
                        break;
                    case 10:
                        print("HERE'S THE OTHER FOOT -- YOU'RE HUNG!!\n");
                        break;
                }
                switch (m) {
                    case 1:
                        pa[3][6] = "-";
                        pa[3][7] = "-";
                        pa[3][8] = "-";
                        pa[4][5] = "(";
                        pa[4][6] = ".";
                        pa[4][8] = ".";
                        pa[4][9] = ")";
                        pa[5][6] = "-";
                        pa[5][7] = "-";
                        pa[5][8] = "-";
                        break;
                    case 2:
                        for (i = 6; i <= 9; i++)
                            pa[i][7] = "X";
                        break;
                    case 3:
                        for (i = 4; i <= 7; i++)
                            pa[i][i - 1] = "\\";
                        break;
                    case 4:
                        pa[4][11] = "/";
                        pa[5][10] = "/";
                        pa[6][9] = "/";
                        pa[7][8] = "/";
                        break;
                    case 5:
                        pa[10][6] = "/";
                        pa[11][5] = "/";
                        break;
                    case 6:
                        pa[10][8] = "\\";
                        pa[11][9] = "\\";
                        break;
                    case 7:
                        pa[3][11] = "\\";
                        break;
                    case 8:
                        pa[3][3] = "/";
                        break;
                    case 9:
                        pa[12][10] = "\\";
                        pa[12][11] = "-";
                        break;
                    case 10:
                        pa[12][3] = "-";
                        pa[12][4] = "/";
                        break;
                }
                for (i = 1; i <= 12; i++) {
                    str = "";
                    for (j = 1; j <= 12; j++)
                        str += pa[i][j];
                    print(str + "\n");
                }
                print("\n");
                print("\n");
                if (m == 10) {
                    print("SORRY, YOU LOSE.  THE WORD WAS " + as + "\n");
                    print("YOU MISSED THAT ONE.  DO YOU ");
                    break;
                }
            } else {
                for (i = 1; i <= l; i++)
                    if (da[i] == "-")
                        break;
                if (i > l) {
                    print("YOU FOUND THE WORD!\n");
                    break;
                }
                print("\n");
                for (i = 1; i <= l; i++)
                    print(da[i]);
                print("\n");
                print("\n");
                print("WHAT IS YOUR GUESS FOR THE WORD");
                bs = await input();
                if (as == bs) {
                    print("RIGHT!!  IT TOOK YOU " + t1 + " GUESSES!\n");
                    break;
                }
                print("WRONG.  TRY ANOTHER LETTER.\n");
                print("\n");
            }
        }
        print("WANT ANOTHER WORD");
        str = await input();
        if (str != "YES")
            break;
    }
    print("\n");
    print("IT'S BEEN FUN!  BYE FOR NOW.\n");
    // Lines 620 and 990 unused in original
}

```

这是C++中的一个标准库函数，名为`main()`。`main()`函数是程序的入口点，当程序运行时，它首先执行这个函数。

这个代码片段没有函数体，它只是一个函数指针。函数指针是一种数据结构，它保存了一个函数的地址。当程序运行时，它将查找`main()`函数的地址并将其装入`main()`函数指针中。然后，程序将跳转到`main()`函数开始执行，因此，`main()`函数中的代码将有机会执行。

`main()`函数是C++中的一个辅助函数，它不返回任何值，但它是程序中所有函数的入口点。这个函数通常用于在程序运行时执行一些通用的操作，例如初始化计算机变量、检查输入参数等。


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


# `44_Hangman/python/hangman.py`

这段代码定义了一个名为 Canvas 的类，用于在图形用户界面（GUI）中绘制文本基型的图形。

Canvas 类包含以下方法：

1. __init__：初始化画布和颜色。
2. clear：清除画布。
3. render：打印画布中的内容，并返回其内容。
4. put：将字符串中的单个字符添加到画布中，指定在画布中的位置（x，y）。

Canvas 类的方法主要依赖于画布的初始化和绘图操作。通过初始化画布并不断将新的字符添加到画布中，Canvas 类可以绘制出文本基型的图形。通过调用 clear 和 render 方法，可以清空画布并将画布的内容打印出来。调用 put 方法可以在画布中绘制单个字符。


```
#!/usr/bin/env python3

"""
HANGMAN

Converted from BASIC to Python by Trevor Hobson and Daniel Piron
"""

import random
from typing import List


class Canvas:
    """For drawing text-based figures"""

    def __init__(self, width: int = 12, height: int = 12, fill: str = " ") -> None:
        self._buffer = []
        for _ in range(height):
            line = []
            for _ in range(width):
                line.append("")
            self._buffer.append(line)

        self.clear()

    def clear(self, fill: str = " ") -> None:
        for row in self._buffer:
            for x in range(len(row)):
                row[x] = fill

    def render(self) -> str:
        lines = []
        for line in self._buffer:
            # Joining by the empty string ("") smooshes all of the
            # individual characters together as one line.
            lines.append("".join(line))
        return "\n".join(lines)

    def put(self, s: str, x: int, y: int) -> None:
        # In an effort to avoid distorting the drawn image, only write the
        # first character of the given string to the buffer.
        self._buffer[y][x] = s[0]


```

这两段代码都是Python中的函数，它们的作用是初始化和绘制图形到画布上。

`init_gallows`函数的作用是在画布上绘制一个包含12个方格的X字形的图案，然后按照指定的顺序在方格中插入四个圆括号。

`draw_head`函数的作用是在画布上绘制一个包含9个星号(代表九个方向)的图案，然后在每个星号上插入一个圆括号。

这些函数都是使用`Canvas`类来操作画布的，分别通过不同的坐标和字符来绘制图形。


```
def init_gallows(canvas: Canvas) -> None:
    for i in range(12):
        canvas.put("X", 0, i)
    for i in range(7):
        canvas.put("X", i, 0)
    canvas.put("X", 6, 1)


def draw_head(canvas: Canvas) -> None:
    canvas.put("-", 5, 2)
    canvas.put("-", 6, 2)
    canvas.put("-", 7, 2)
    canvas.put("(", 4, 3)
    canvas.put(".", 5, 3)
    canvas.put(".", 7, 3)
    canvas.put(")", 8, 3)
    canvas.put("-", 5, 4)
    canvas.put("-", 6, 4)
    canvas.put("-", 7, 4)


```

这段代码定义了三个函数，分别是 draw_body、draw_right_arm 和 draw_left_arm。这些函数都是 Canvas 类的实例方法，用于在画布上绘制不同部位的身体。

具体来说，draw_body 函数会在画布上从左往右、从上往下的 5x1 网格中，向左偏移 6 个像素，也就是在横坐标为 6 到 8 的位置上画一条从左上往右下的虚线。draw_right_arm 函数则会在画布上从左往右、从上往下的 3x1 网格中，向右偏移 i-1 个像素，也就是在横坐标为 i-1 到 i+1 的位置上画一条从左上往右下的虚线。draw_left_arm 函数则是在 draw_body 函数的基础上，左右两边各多画了一条虚线，分别位于横坐标为 10 到 11、12 到 13 的位置上。


```
def draw_body(canvas: Canvas) -> None:
    for i in range(5, 9, 1):
        canvas.put("X", 6, i)


def draw_right_arm(canvas: Canvas) -> None:
    for i in range(3, 7):
        canvas.put("\\", i - 1, i)


def draw_left_arm(canvas: Canvas) -> None:
    canvas.put("/", 10, 3)
    canvas.put("/", 9, 4)
    canvas.put("/", 8, 5)
    canvas.put("/", 7, 6)


```

这是一个Python代码，它描述了绘图操作。def draw\_right\_leg(canvas: Canvas) -> None:
绘制了右臂的图案，将四个方括号分别绘制在(5, 9)、(4, 10)、(7, 9)、(8, 10)这四个位置。draw\_left\_leg(canvas: Canvas) -> None:
绘制了左臂的图案，将四个反斜杠分别绘制在(7, 9)、(8, 10)、(10, 2)、(11, 2)这四个位置。draw\_left\_hand(canvas: Canvas) -> None:
绘制了左手的图案，将四个反斜杠分别绘制在(10, 2)、(11, 2)、(12, 2)、(13, 2)这四个位置。draw\_right\_hand(canvas: Canvas) -> None:
绘制了右手的图案，将四个斜杠分别绘制在(2, 2)、(3, 2)、(4, 2)、(5, 2)这四个位置。


```
def draw_right_leg(canvas: Canvas) -> None:
    canvas.put("/", 5, 9)
    canvas.put("/", 4, 10)


def draw_left_leg(canvas: Canvas) -> None:
    canvas.put("\\", 7, 9)
    canvas.put("\\", 8, 10)


def draw_left_hand(canvas: Canvas) -> None:
    canvas.put("\\", 10, 2)


def draw_right_hand(canvas: Canvas) -> None:
    canvas.put("/", 2, 2)


```

这段代码定义了两个函数，`draw_left_foot` 和 `draw_right_foot`。这两个函数都是绘画函数，接收一个 `Canvas` 对象作为参数，并对其进行绘制操作。

具体来说，这两个函数分别向左和向右绘制足迹，然后按照给定的顺序绘制手臂和腿部。PHASES 常量是一个元组，其中包含绘制的每个 phase 的名称，这些phase按照顺序对应到绘制函数中。

draw_head函数将绘制一个头部，从左上角开始绘制，然后向右下角绘制一条横线，再向右上角绘制两条横线，最终得到一个完整的头部。

draw_body函数将绘制一个身体，从左上角开始绘制，然后向右下角绘制一条横线，再向左上角绘制一条横线，最终得到一个完整的身体。

draw_right_arm函数将绘制一个手臂，从左上角开始绘制，然后向右下角绘制一条横线，再向右上角绘制一条横线，最终得到一个完整的手臂。

draw_left_arm函数将绘制一个手臂，从左上角开始绘制，然后向右下角绘制一条横线，再向左上角绘制一条横线，最终得到一个完整的手臂。

draw_right_leg函数将绘制一个腿部，从左上角开始绘制，然后向右下角绘制一条横线，再向左上角绘制一条横线，最终得到一个完整的腿部。

draw_left_leg函数将绘制一个腿部，从左上角开始绘制，然后向右下角绘制一条横线，再向左上角绘制一条横线，最终得到一个完整的腿部。

draw_left_foot函数将绘制一个左脚，从左上角开始绘制，然后向右下角绘制一条横线，再向左上角绘制一条横线，最终得到一个完整的左脚。

draw_right_foot函数将绘制一个右脚，从左上角开始绘制，然后向右下角绘制一条横线，再向左上角绘制一条横线，最终得到一个完整的右脚。

最后，draw_right_hand函数将绘制一个手部，从左上角开始绘制，然后向右下角绘制一条横线，再向左上角绘制一条横线，最终得到一个完整的手部。

draw_left_hand函数将绘制一个手部，从左上角开始绘制，然后向右下角绘制一条横线，再向左上角绘制一条横线，最终得到一个完整的手部。

这些绘制函数按照特定的顺序执行，以创建一个完整的图形。


```
def draw_left_foot(canvas: Canvas) -> None:
    canvas.put("\\", 9, 11)
    canvas.put("-", 10, 11)


def draw_right_foot(canvas: Canvas) -> None:
    canvas.put("-", 2, 11)
    canvas.put("/", 3, 11)


PHASES = (
    ("First, we draw a head", draw_head),
    ("Now we draw a body.", draw_body),
    ("Next we draw an arm.", draw_right_arm),
    ("this time it's the other arm.", draw_left_arm),
    ("Now, let's draw the right leg.", draw_right_leg),
    ("This time we draw the left leg.", draw_left_leg),
    ("Now we put up a hand.", draw_left_hand),
    ("Next the other hand.", draw_right_hand),
    ("Now we draw one foot", draw_left_foot),
    ("Here's the other foot -- you're hung!!", draw_right_foot),
)


```

这段代码定义了一个名为 `words` 的列表，包含了21个单词。这些单词被用于某个程序或脚本，但具体用途并未在代码中明确说明。


```
words = [
    "GUM",
    "SIN",
    "FOR",
    "CRY",
    "LUG",
    "BYE",
    "FLY",
    "UGLY",
    "EACH",
    "FROM",
    "WORK",
    "TALK",
    "WITH",
    "SELF",
    "PIZZA",
    "THING",
    "FEIGN",
    "FIEND",
    "ELBOW",
    "FAULT",
    "DIRTY",
    "BUDGET",
    "SPIRIT",
    "QUAINT",
    "MAIDEN",
    "ESCORT",
    "PICKAX",
    "EXAMPLE",
    "TENSION",
    "QUININE",
    "KIDNEY",
    "REPLICA",
    "SLEEPER",
    "TRIANGLE",
    "KANGAROO",
    "MAHOGANY",
    "SERGEANT",
    "SEQUENCE",
    "MOUSTACHE",
    "DANGEROUS",
    "SCIENTIST",
    "DIFFERENT",
    "QUIESCENT",
    "MAGISTRATE",
    "ERRONEOUSLY",
    "LOUDSPEAKER",
    "PHYTOTOXIC",
    "MATRIMONIAL",
    "PARASYMPATHOMIMETIC",
    "THIGMOTROPISM",
]


```

It looks like you've implemented a guessing game where the user tries to guess a secret word by repeating it multiple times. The game keeps track of the number of guesses it takes the user to correctly guess the word, as well as the number of guesses it takes to make a guess that does not match any of the words in the word.

There are several things you could improve on in this game:

1. You're using the `indices` list to store the letters in the target word, but this assumes that the letters in the target word are stored in the same order as they appear in the game. This could be easily fixed by storing the indices in the `target_words` list instead, which would allow you to easily compare the actual letters in the target word to the ones the user is guessing.
2. The game currently has a hard-coded list of phases for when the user makes a wrong guess. This is not very flexible, as you could easily modify the phases to fit your needs.
3. The game currently has a maximum number of guesses at 10, which is quite strict. This could be easily adjusted to fit the needs of your game.
4. The game currently has a comments system for when the user makes a wrong guess, but this is not very informative. You could easily add a few lines of text to explain what the comment means, or even have the user be able to see the actual word they are trying to guess.

Overall, your game looks like it could be a fun and interesting way for people to guess a secret word.


```
def play_game(guess_target: str) -> None:
    """Play one round of the game"""
    wrong_guesses = 0
    guess_progress = ["-"] * len(guess_target)
    guess_list: List[str] = []

    gallows = Canvas()
    init_gallows(gallows)

    guess_count = 0
    while True:
        print("Here are the letters you used:")
        print(",".join(guess_list) + "\n")
        print("".join(guess_progress) + "\n")
        guess_letter = ""
        guess_word = ""
        while guess_letter == "":

            guess_letter = input("What is your guess? ").upper()[0]
            if not guess_letter.isalpha():
                guess_letter = ""
                print("Only letters are allowed!")
            elif guess_letter in guess_list:
                guess_letter = ""
                print("You guessed that letter before!")

        guess_list.append(guess_letter)
        guess_count += 1
        if guess_letter in guess_target:
            indices = [
                i for i, letter in enumerate(guess_target) if letter == guess_letter
            ]
            for i in indices:
                guess_progress[i] = guess_letter
            if "".join(guess_progress) == guess_target:
                print("You found the word!")
                break
            else:
                print("\n" + "".join(guess_progress) + "\n")
                while guess_word == "":
                    guess_word = input("What is your guess for the word? ").upper()
                    if not guess_word.isalpha():
                        guess_word = ""
                        print("Only words are allowed!")
                if guess_word == guess_target:
                    print("Right!! It took you", guess_count, "guesses!")
                    break
        else:
            comment, draw_bodypart = PHASES[wrong_guesses]

            print(comment)
            draw_bodypart(gallows)
            print(gallows.render())

            wrong_guesses += 1
            print("Sorry, that letter isn't in the word.")

            if wrong_guesses == 10:
                print("Sorry, you lose. The word was " + guess_target)
                break


```

这段代码的主要作用是向用户随机生成一些单词，并在用户继续时停止游戏。

具体来说，它首先定义了一个主函数（main），其中包含以下操作：

1. 输出一个游戏界面，使用空格和制字符填充32个字符。
2. 输出一段文字，使用空格和制字符填充15个字符。
3. 从单词列表中随机选择一个单词，并计数器加一。
4. 循环让用户选择是否继续游戏，如果用户选择“是”，循环将调用generate_words函数。
5. 如果当前单词数等于单词列表的长度，显示“你猜对了！”并停止游戏。
6. 循环结束后，输出“It's been fun! Bye for now.”（“游戏很有趣！再见。）”。

generate_words函数的作用是生成一个随机单词，并从给定的单词列表中随机选择一个。


```
def main() -> None:
    print(" " * 32 + "HANGMAN")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")

    random.shuffle(words)
    current_word = 0
    word_count = len(words)

    keep_playing = True
    while keep_playing:

        play_game(words[current_word])
        current_word += 1

        if current_word == word_count:
            print("You did all the words!!")
            keep_playing = False
        else:
            keep_playing = (
                input("Want another word? (yes or no) ").lower().startswith("y")
            )

    print("It's been fun! Bye for now.")


```

这段代码是一个条件判断语句，它的作用是判断当前脚本是否作为主程序运行。如果当前脚本作为主程序运行，那么程序会执行 if 语句块内的语句，否则程序会跳过 if 语句块并继续执行 next 语句。

具体来说，这段代码的意义是：如果当前脚本作为主程序运行，那么执行 if 语句块内的语句；否则执行 next 语句。这里的 if 语句块是一个空语句，无论条件真假，都不会执行其中的语句。而 __name__ 是一个特殊字符，它在这里作为 if 语句的判断条件，如果当前脚本作为主程序运行，那么 __name__ 的值为 "__main__"，程序会执行 if 语句块内的语句；否则程序会跳过 if 语句块并继续执行 next 语句。


```
if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Hello

This is a sample of one of the great number of conversational programs. In a sense, it is like a CAI program except that its responses are just good fun. Whenever a computer is exhibited at a convention or conference with people that have not used a computer before, the conversational programs seem to get the first activity.

In this particular program, the computer dispenses advice on various problems such as sex, health, money, or job.

David Ahl is the author of HELLO.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=82)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=97)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `45_Hello/java/Hello.java`

Hello,

I'm sorry, but it appears that there is a mistake in the method name. It should be `public static void main(String[] args)` instead of `public static void startGame(String[] args)`. Please correct this and I'll be happy to help.


```
import java.util.Scanner;

/**
 * Game of Hello
 * <p>
 * Based on the BASIC game of Hello here
 * https://github.com/coding-horror/basic-computer-games/blob/main/45%20Hello/hello.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 *
 * Converted from BASIC to Java by Darren Cardenas.
 */

public class Hello {

  private static final int MONEY_WAIT_MS = 3000;

  private final boolean goodEnding = false;

  private final Scanner scan;  // For user input

  public Hello() {

    scan = new Scanner(System.in);

  }  // End of constructor Hello

  public void play() {

    showIntro();
    startGame();

  }  // End of method play

  private static void showIntro() {

    System.out.println(" ".repeat(32) + "HELLO");
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    System.out.println("\n\n");

  }  // End of method showIntro

  private void startGame() {

    boolean moreProblems = true;

    String userCategory = "";
    String userName = "";
    String userResponse = "";

    // Name question
    System.out.println("HELLO.  MY NAME IS CREATIVE COMPUTER.\n\n");
    System.out.print("WHAT'S YOUR NAME? ");
    userName = scan.nextLine();
    System.out.println("");

    // Enjoyment question
    System.out.print("HI THERE, " + userName + ", ARE YOU ENJOYING YOURSELF HERE? ");

    while (true) {
      userResponse = scan.nextLine();
      System.out.println("");

      if (userResponse.toUpperCase().equals("YES")) {
        System.out.println("I'M GLAD TO HEAR THAT, " + userName + ".\n");
        break;
      }
      else if (userResponse.toUpperCase().equals("NO")) {
        System.out.println("OH, I'M SORRY TO HEAR THAT, " + userName + ". MAYBE WE CAN");
        System.out.println("BRIGHTEN UP YOUR VISIT A BIT.");
        break;
      }
      else {
        System.out.println(userName + ", I DON'T UNDERSTAND YOUR ANSWER OF '" + userResponse + "'.");
        System.out.print("PLEASE ANSWER 'YES' OR 'NO'.  DO YOU LIKE IT HERE? ");
      }
    }

    // Category question
    System.out.println("");
    System.out.println("SAY, " + userName + ", I CAN SOLVE ALL KINDS OF PROBLEMS EXCEPT");
    System.out.println("THOSE DEALING WITH GREECE.  WHAT KIND OF PROBLEMS DO");
    System.out.print("YOU HAVE (ANSWER SEX, HEALTH, MONEY, OR JOB)? ");

    while (moreProblems) {
      userCategory = scan.nextLine();
      System.out.println("");

      // Sex advice
      if (userCategory.toUpperCase().equals("SEX")) {
        System.out.print("IS YOUR PROBLEM TOO MUCH OR TOO LITTLE? ");
        userResponse = scan.nextLine();
        System.out.println("");

        while (true) {
          if (userResponse.toUpperCase().equals("TOO MUCH")) {
            System.out.println("YOU CALL THAT A PROBLEM?!!  I SHOULD HAVE SUCH PROBLEMS!");
            System.out.println("IF IT BOTHERS YOU, " + userName + ", TAKE A COLD SHOWER.");
            break;
          }
          else if (userResponse.toUpperCase().equals("TOO LITTLE")) {
            System.out.println("WHY ARE YOU HERE IN SUFFERN, " + userName + "?  YOU SHOULD BE");
            System.out.println("IN TOKYO OR NEW YORK OR AMSTERDAM OR SOMEPLACE WITH SOME");
            System.out.println("REAL ACTION.");
            break;
          }
          else {
            System.out.println("DON'T GET ALL SHOOK, " + userName + ", JUST ANSWER THE QUESTION");
            System.out.print("WITH 'TOO MUCH' OR 'TOO LITTLE'.  WHICH IS IT? ");
            userResponse = scan.nextLine();
          }
        }
      }
      // Health advice
      else if (userCategory.toUpperCase().equals("HEALTH")) {
        System.out.println("MY ADVICE TO YOU " + userName + " IS:");
        System.out.println("     1.  TAKE TWO ASPRIN");
        System.out.println("     2.  DRINK PLENTY OF FLUIDS (ORANGE JUICE, NOT BEER!)");
        System.out.println("     3.  GO TO BED (ALONE)");
      }
      // Money advice
      else if (userCategory.toUpperCase().equals("MONEY")) {
        System.out.println("SORRY, " + userName + ", I'M BROKE TOO.  WHY DON'T YOU SELL");
        System.out.println("ENCYCLOPEADIAS OR MARRY SOMEONE RICH OR STOP EATING");
        System.out.println("SO YOU WON'T NEED SO MUCH MONEY?");
      }
      // Job advice
      else if (userCategory.toUpperCase().equals("JOB")) {
        System.out.println("I CAN SYMPATHIZE WITH YOU " + userName + ".  I HAVE TO WORK");
        System.out.println("VERY LONG HOURS FOR NO PAY -- AND SOME OF MY BOSSES");
        System.out.println("REALLY BEAT ON MY KEYBOARD.  MY ADVICE TO YOU, " + userName + ",");
        System.out.println("IS TO OPEN A RETAIL COMPUTER STORE.  IT'S GREAT FUN.");
      }
      else {
        System.out.println("OH, " + userName + ", YOUR ANSWER OF " + userCategory + " IS GREEK TO ME.");
      }

      // More problems question
      while (true) {
        System.out.println("");
        System.out.print("ANY MORE PROBLEMS YOU WANT SOLVED, " + userName + "? ");
        userResponse = scan.nextLine();
        System.out.println("");

        if (userResponse.toUpperCase().equals("YES")) {
          System.out.print("WHAT KIND (SEX, MONEY, HEALTH, JOB)? ");
          break;
        }
        else if (userResponse.toUpperCase().equals("NO")) {
          moreProblems = false;
          break;
        }
        else {
          System.out.println("JUST A SIMPLE 'YES' OR 'NO' PLEASE, " + userName + ".");
        }
      }
    }

    // Payment question
    System.out.println("");
    System.out.println("THAT WILL BE $5.00 FOR THE ADVICE, " + userName + ".");
    System.out.println("PLEASE LEAVE THE MONEY ON THE TERMINAL.");

    // Pause
    try {
      Thread.sleep(MONEY_WAIT_MS);
    } catch (Exception e) {
      System.out.println("Caught Exception: " + e.getMessage());
    }

    System.out.println("\n\n");

    while (true) {
      System.out.print("DID YOU LEAVE THE MONEY? ");
      userResponse = scan.nextLine();
      System.out.println("");

      if (userResponse.toUpperCase().equals("YES")) {
        System.out.println("HEY, " + userName + "??? YOU LEFT NO MONEY AT ALL!");
        System.out.println("YOU ARE CHEATING ME OUT OF MY HARD-EARNED LIVING.");
        System.out.println("");
        System.out.println("WHAT A RIP OFF, " + userName + "!!!\n");
        break;
      }
      else if (userResponse.toUpperCase().equals("NO")) {
        System.out.println("THAT'S HONEST, " + userName + ", BUT HOW DO YOU EXPECT");
        System.out.println("ME TO GO ON WITH MY PSYCHOLOGY STUDIES IF MY PATIENTS");
        System.out.println("DON'T PAY THEIR BILLS?");
        break;
      }
      else {
        System.out.println("YOUR ANSWER OF '" + userResponse + "' CONFUSES ME, " + userName + ".");
        System.out.println("PLEASE RESPOND WITH 'YES' OR 'NO'.");
      }
    }

    // Legacy included unreachable code
    if (goodEnding) {
      System.out.println("NICE MEETING YOU, " + userName + ", HAVE A NICE DAY.");
    }
    else {
      System.out.println("");
      System.out.println("TAKE A WALK, " + userName + ".\n");
    }

  }  // End of method startGame

  public static void main(String[] args) {

    Hello hello = new Hello();
    hello.play();

  }  // End of method main

}  // End of class Hello

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `45_Hello/javascript/hello.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。

`print` 函数的作用是在文档的输出区域（也就是浏览器地址栏下方区域）中添加一个新的文本节点，并将用户输入的参数作为文本内容插入到该节点中。

`input` 函数的作用是获取用户输入的字符串，并在用户输入后将其存储在变量 `input_str` 中。该函数还监听用户输入字符中的 `keydown` 事件，当用户按下键盘上的 `13` 键时，将 `input_str` 存储到变量中，并将其显示在页面上。

输入字符串后，函数会将其存储在 `input_str` 变量中，并使用 `print` 函数将其添加到页面的输出区域中，以便用户可以看到输入的字符串。


```
// HELLO
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

It seems like you're trying to use a natural language input to determine what kind of problem you need help with, but it looks like your input formatting is a little bit messed up.

Anyway, if you're going to ask for money, I will assume that you meant to ask for job or health problems, and I will provide you with the appropriate advice.

If you're asking for sex or money problems, I'm sorry, but I don't have any information to provide in that regard.

If you're asking for job problems, I suggest that you try to get a job or look for new opportunities.

If you're asking for health problems, I suggest that you consult a doctor or a health clinic.

If you're asking for sex problems, I'm sorry, but I don't have any information to provide in that regard.

If you're asking for money problems, I suggest that you try to ask for help from your family, or consider looking for a job that pays better.

If you're asking for general problems, I suggest that you try to find a therapist or counselor.

If you're asking for advice on your job, I suggest that you ask for feedback from your supervisor or colleagues.

If you're asking for advice on your health, I suggest that you ask for help from a doctor or a health clinic.

If you're asking for advice on your money, I suggest that you ask for help from a financial advisor or a credit counselor.


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// Main control section
async function main()
{
    print(tab(33) + "HELLO\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("HELLO.  MY NAME IS CREATIVE COMPUTER.\n");
    print("\n");
    print("\n");
    print("WHAT'S YOUR NAME");
    ns = await input();
    print("\n");
    print("HI THERE, " + ns + ", ARE YOU ENJOYING YOURSELF HERE");
    while (1) {
        bs = await input();
        print("\n");
        if (bs == "YES") {
            print("I'M GLAD TO HEAR THAT, " + ns + ".\n");
            print("\n");
            break;
        } else if (bs == "NO") {
            print("OH, I'M SORRY TO HEAR THAT, " + ns + ". MAYBE WE CAN\n");
            print("BRIGHTEN UP YOUR VISIT A BIT.\n");
            break;
        } else {
            print("PLEASE ANSWER 'YES' OR 'NO'.  DO YOU LIKE IT HERE");
        }
    }
    print("\n");
    print("SAY, " + ns + ", I CAN SOLVED ALL KINDS OF PROBLEMS EXCEPT\n");
    print("THOSE DEALING WITH GREECE.  WHAT KIND OF PROBLEMS DO\n");
    print("YOU HAVE (ANSWER SEX, HEALTH, MONEY, OR JOB)");
    while (1) {
        cs = await input();
        print("\n");
        if (cs != "SEX" && cs != "HEALTH" && cs != "MONEY" && cs != "JOB") {
            print("OH, " + ns + ", YOUR ANSWER OF " + cs + " IS GREEK TO ME.\n");
        } else if (cs == "JOB") {
            print("I CAN SYMPATHIZE WITH YOU " + ns + ".  I HAVE TO WORK\n");
            print("VERY LONG HOURS FOR NO PAY -- AND SOME OF MY BOSSES\n");
            print("REALLY BEAT ON MY KEYBOARD.  MY ADVICE TO YOU, " + ns + ",\n");
            print("IS TO OPEN A RETAIL COMPUTER STORE.  IT'S GREAT FUN.\n");
        } else if (cs == "MONEY") {
            print("SORRY, " + ns + ", I'M BROKE TOO.  WHY DON'T YOU SELL\n");
            print("ENCYCLOPEADIAS OR MARRY SOMEONE RICH OR STOP EATING\n");
            print("SO YOU WON'T NEED SO MUCH MONEY?\n");
        } else if (cs == "HEALTH") {
            print("MY ADVICE TO YOU " + ns + " IS:\n");
            print("     1.  TAKE TWO ASPRIN\n");
            print("     2.  DRINK PLENTY OF FLUIDS (ORANGE JUICE, NOT BEER!)\n");
            print("     3.  GO TO BED (ALONE)\n");
        } else {
            print("IS YOUR PROBLEM TOO MUCH OR TOO LITTLE");
            while (1) {
                ds = await input();
                print("\n");
                if (ds == "TOO MUCH") {
                    print("YOU CALL THAT A PROBLEM?!!  I SHOULD HAVE SUCH PROBLEMS!\n");
                    print("IF IT BOTHERS YOU, " + ns + ", TAKE A COLD SHOWER.\n");
                    break;
                } else if (ds == "TOO LITTLE") {
                    print("WHY ARE YOU HERE IN SUFFERN, " + ns + "?  YOU SHOULD BE\n");
                    print("IN TOKYO OR NEW YORK OR AMSTERDAM OR SOMEPLACE WITH SOME\n");
                    print("REAL ACTION.\n");
                    break;
                } else {
                    print("DON'T GET ALL SHOOK, " + ns + ", JUST ANSWER THE QUESTION\n");
                    print("WITH 'TOO MUCH' OR 'TOO LITTLE'.  WHICH IS IT");
                }
            }
        }
        print("\n");
        print("ANY MORE PROBLEMS YOU WANT SOLVED, " + ns);
        es = await input();
        print("\n");
        if (es == "YES") {
            print("WHAT KIND (SEX, MONEY, HEALTH, JOB)");
        } else if (es == "NO") {
            print("THAT WILL BE $5.00 FOR THE ADVICE, " + ns + ".\n");
            print("PLEASE LEAVE THE MONEY ON THE TERMINAL.\n");
            print("\n");
```

这段代码的主要目的是模拟用户与AI程序进行交流的过程。程序内部先获取了一个当前日期（用Java的Date类获取）的值，将其赋值给变量d。然后在while循环中，程序会不断地获取用户输入的一个字符串gs，并将其与之前获取的变量d进行比较。

如果gs字符串为"YES"，程序会打印"HEY, " + ns + "???You Left No Money At All!"，然后询问用户ns的姓名并输出ns的联系方式。接着程序会输出"YOU ARE CHEATING ME OUT OF MY HARD-EARNED LIVING."，表明用户欺诈行为。然后程序会询问用户是否还有其他问题，用户回答"NO"，程序会继续输出"THAT'S HONEST, " + ns + ", BUT HOW DO YOU EXPECT\nYOUR TO GO ON WITH YOUR PSYCHOLOGY STUDIES IF YOUR PATIENT\nDON'T PAY THEIR BILLS?\n"类似的信息。

如果gs字符串为"NO"，程序会先输出"THAT'S HONEST, " + ns + ", BUT HOW DO YOU EXPECT\nYOU TO GO ON WITH YOUR PSYCHOLOGY STUDIES IF YOUR PATIENT\nDON'T PAY THEIR BILLS?"询问用户是否还有其他问题，如果没有问题，就会执行break；跳出while循环。如果用户回答"NO"，程序会继续输出"YOUR ANSWER OF '" + gs + "' CONFUSES ME, " + ns + "\n".


```
//            d = new Date().valueOf();
//            while (new Date().valueOf() - d < 2000) ;
            print("\n");
            print("\n");
            while (1) {
                print("DID YOU LEAVE THE MONEY");
                gs = await input();
                print("\n");
                if (gs == "YES") {
                    print("HEY, " + ns + "??? YOU LEFT NO MONEY AT ALL!\n");
                    print("YOU ARE CHEATING ME OUT OF MY HARD-EARNED LIVING.\n");
                    print("\n");
                    print("WHAT A RIP OFF, " + ns + "!!!\n");
                    print("\n");
                    break;
                } else if (gs == "NO") {
                    print("THAT'S HONEST, " + ns + ", BUT HOW DO YOU EXPECT\n");
                    print("ME TO GO ON WITH MY PSYCHOLOGY STUDIES IF MY PATIENT\n");
                    print("DON'T PAY THEIR BILLS?\n");
                    break;
                } else {
                    print("YOUR ANSWER OF '" + gs + "' CONFUSES ME, " + ns + ".\n");
                    print("PLEASE RESPOND WITH 'YES' OR 'NO'.\n");
                }
            }
            break;
        }
    }
    print("\n");
    print("TAKE A WALK, " + ns + ".\n");
    print("\n");
    print("\n");
    // Line 390 not used in original
}

```

这道题目是一个不完整的C语言代码，缺少了程序的具体实现。它包含了两个主要部分：`main()`函数和两个函数声明。这两个部分一起定义了一个程序，但它们各自承担不同的功能。

`main()`函数是程序的入口点，程序从这里开始执行。在这个函数中，用户可以输入一些命令，然后系统会根据用户的要求来调用程序的其他部分。

第一个函数声明定义了一个名为`changeBed`的函数，它接受两个整数参数，一个是`bEDge`，另一个是`side`。这个函数的作用是判断给定的边（`bEDge` 和 `side`）是否可以翻转（`true`或`false`）。

第二个函数声明定义了一个名为`computeFiveThirty`的函数，它接受一个整数参数，代表当前的计数器值（`int`）。这个函数的作用是计算53个连续奇数的和。

总之，`main()`函数作为程序的入口点，负责处理用户输入并调用其他函数。`changeBed()`函数用于处理用户关于边界条件的指定，而`computeFiveThirty()`函数用于计算奇数计数器的值。


```
main();

```