# BasicComputerGames源码解析 13

# `05_Bagels/csharp/Game.cs`

This is a class written in C# that simulates a game of bag桥， where players take turns rolling a die and targeting a number that they believe is their goal. The simulation includes different levels of difficulty and various methods for printing messages, such as displaying a numbered introduction message or printing the背包el validation.
The main method, DisplayIntroText, prints out a series of informative messages to the console when the program starts and also displays a numbered introduction message.
The second, BagelValidation,


```
﻿using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace BasicComputerGames.Bagels
{
	public class Game : GameBase
	{
		public void GameLoop()
		{
			DisplayIntroText();
			int points = 0;
			do
			{
				var result =PlayRound();
				if (result)
					++points;
			} while (TryAgain());

			Console.WriteLine();
			Console.WriteLine($"A {points} point Bagels buff!!");
			Console.WriteLine("Hope you had fun. Bye.");
		}

		private const int Length = 3;
		private const int MaxGuesses = 20;

		private bool  PlayRound()
		{
			var secret = BagelNumber.CreateSecretNumber(Length);
			Console.WriteLine("O.K. I have a number in mind.");
			for (int guessNo = 1; guessNo <= MaxGuesses; ++guessNo)
			{
				string strGuess;
				BagelValidation isValid;
				do
				{
					Console.WriteLine($"Guess #{guessNo}");
					strGuess = Console.ReadLine();
					isValid = BagelNumber.IsValid(strGuess, Length);
					PrintError(isValid);
				} while (isValid != BagelValidation.Valid);

				var guess = new BagelNumber(strGuess);
				var fermi = 0;
				var pico = 0;
				(pico, fermi) = secret.CompareTo(guess);
				if(pico + fermi == 0)
					Console.Write("BAGELS!");
				else if (fermi == Length)
				{
					Console.WriteLine("You got it!");
					return true;
				}
				else
				{
					PrintList("Pico ", pico);
					PrintList("Fermi ", fermi);
				}
				Console.WriteLine();
			}

			Console.WriteLine("Oh, well.");
			Console.WriteLine($"That's {MaxGuesses} guesses.  My Number was {secret}");

			return false;

		}

		private void PrintError(BagelValidation isValid)
		{
			switch (isValid)
			{
				case BagelValidation.NonDigit:
					Console.WriteLine("What?");
					break;

				case BagelValidation.NotUnique:
					Console.WriteLine("Oh, I forgot to tell you that the number I have in mind has no two digits the same.");
					break;

				case BagelValidation.WrongLength:
					Console.WriteLine($"Try guessing a {Length}-digit number.");
					break;

				case BagelValidation.Valid:
					break;
			}
		}

		private void PrintList(string msg, int repeat)
		{
			for(int i=0; i<repeat; ++i)
				Console.Write(msg);
		}

		private void DisplayIntroText()
		{
			Console.ForegroundColor = ConsoleColor.Yellow;
			Console.WriteLine("Bagels");
			Console.WriteLine("Creating Computing, Morristown, New Jersey.");
			Console.WriteLine();

			Console.ForegroundColor = ConsoleColor.DarkGreen;
			Console.WriteLine(
				"Original code author unknow but suspected to be from Lawrence Hall of Science, U.C. Berkley");
			Console.WriteLine("Originally published in 1978 in the book 'Basic Computer Games' by David Ahl.");
			Console.WriteLine("Modernized and converted to C# in 2021 by James Curran (noveltheory.com).");
			Console.WriteLine();

			Console.ForegroundColor = ConsoleColor.Gray;
			Console.WriteLine("I am thinking of a three-digit number.  Try to guess");
			Console.WriteLine("my number and I will give you clues as follows:");
			Console.WriteLine("   pico   - One digit correct but in the wrong position");
			Console.WriteLine("   fermi  - One digit correct and in the right position");
			Console.WriteLine("   bagels - No digits correct");
			Console.WriteLine();

			Console.ForegroundColor = ConsoleColor.Yellow;
			Console.WriteLine("Press any key start the game.");
			Console.ReadKey(true);
		}
	}
}

```

# `05_Bagels/csharp/GameBase.cs`

这段代码是一个名为 "BasicComputerGames.Bagels" 的命名空间中包含的一个名为 "GameBase" 的类。这个类定义了一个名为 "TryAgain" 的方法，用于询问玩家是否想要再次尝试。

在 "TryAgain" 方法中，首先设置了一个 "Random" 类型的变量 "Rnd"，用于生成随机数。然后定义了一个字符串 "Press Y or N"，并将其赋值为 "Press Y or N"。接着，使用 "Console.ForegroundColor" 方法将背景颜色设置为白色，然后使用 "Console.WriteLine" 方法在屏幕上显示这个字符串。

接下来，将字符串颜色设置为黄色，然后等待玩家的输入。在循环中，使用 "Console.ReadKey" 方法来获取玩家的输入，无论是 "Y" 还是 "N"。然后将所获取的输入转换为大写，并检查这个输入是否与 "Press Y or N" 完全匹配。如果匹配，则 "TryAgain" 方法返回真，即玩家想要再次尝试。否则，返回 false。

最后，还有一些辅助方法，如 "GameBase" 类中定义的 "PressAnyKey" 方法，用于等待玩家按任意键来退出游戏。


```
﻿using System;

namespace BasicComputerGames.Bagels
{
	public class GameBase
	{
		protected Random Rnd { get; } = new Random();

		/// <summary>
		/// Prompt the player to try again, and wait for them to press Y or N.
		/// </summary>
		/// <returns>Returns true if the player wants to try again, false if they have finished playing.</returns>
		protected bool TryAgain()
		{
			Console.ForegroundColor = ConsoleColor.White;
			Console.WriteLine("Would you like to try again? (Press 'Y' for yes or 'N' for no)");

			Console.ForegroundColor = ConsoleColor.Yellow;
			Console.Write("> ");

			char pressedKey;
			// Keep looping until we get a recognised input
			do
			{
				// Read a key, don't display it on screen
				ConsoleKeyInfo key = Console.ReadKey(true);
				// Convert to upper-case so we don't need to care about capitalisation
				pressedKey = Char.ToUpper(key.KeyChar);
				// Is this a key we recognise? If not, keep looping
			} while (pressedKey != 'Y' && pressedKey != 'N');
			// Display the result on the screen
			Console.WriteLine(pressedKey);

			// Return true if the player pressed 'Y', false for anything else.
			return (pressedKey == 'Y');
		}

	}
}

```

# `05_Bagels/csharp/Program.cs`

这段代码是一个 C# 类的程序，定义了一个名为 "BasicComputerGames.Bagels" 的命名空间，其中包含一个名为 "Program" 的类。这个类的定义了一个名为 "Main" 的方法，它的参数列表使用了 "string[] args" 类型，表示这个方法可以接受任意数量的字符串参数。

在 "Main" 方法中，首先创建了一个名为 "game" 的对象，这个对象从 "BasicComputerGames.Bagels" 命名空间中继承而来。然后，代码调用了 "game" 对象的一个名为 "GameLoop" 的方法，这个方法的参数是一个空括号，表示这个方法可以接受任意数量的匿名类型参数。

由于这个方法的名称和参数列表类似于 JavaScript 中的 "setInterval" 函数，它会创建一个无限循环，这个循环会在每个匿名类型参数被赋值之后重新执行。因此，这个方法会在游戏循环中不断地调用自己，直到游戏的玩家选择退出为止。


```
﻿namespace BasicComputerGames.Bagels
{
	public class Program
	{
		public static void Main(string[] args)
		{
			// Create an instance of our main Game class
			var game = new Game();

			// Call its GameLoop function. This will play the game endlessly in a loop until the player chooses to quit.
			game.GameLoop();
		}
	}
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `05_Bagels/java/BagelGame.java`

这段代码定义了一个名为Bagels的游戏，它使用Java来包装一个单人纸牌游戏的所有状态和游戏逻辑。它主要用于教育目的，帮助学生理解如何编写游戏。在实际应用中，通常会有更多的人力和时间成本，因此需要更复杂的游戏引擎和更高级的图形用户界面。


```
/******************************************************************************
*
* Encapsulates all the state and game logic for one single game of Bagels
*
* Used by Bagels.java
*
* Jeff Jetton, 2020
*
******************************************************************************/

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
```

This is a Java method that takes a string s, which is the response from the game, and a string t, which is the secret number that the game is trying to solve. It returns a response string based on how well the game was solved and sets the game state accordingly.

The method first converts the string t to an integer list and then loops through the characters in the string t. It then compares the characters in the string t to the characters in the correct order for a correct answer, which is then displayed in the response string.

If the game was not solved correctly, the method increments the guess counter and checks if the player has exceeded the maximum number of guesses. If the player has exceeded the maximum number of guesses, the game is considered lost and the response string is displayed.

The method also checks if the response string is equal to "BAGELS", which is the response for a game that could not be solved. If the response string is equal to "BAGELS", the game is considered un solved and the response string is displayed.


```
import java.util.Set;

public class BagelGame {

  public static final String CORRECT = "FERMI FERMI FERMI";
  public static final int MAX_GUESSES = 20;

  enum GameState {
      RUNNING,
      WON,
      LOST
    }

  private GameState      state = GameState.RUNNING;
  private List<Integer>  secretNum;
  private int            guessNum = 1;

  public BagelGame() {
    // No-arg constructor for when you don't need to set the seed
    this(new Random());
  }

  public BagelGame(long seed) {
    // Setting the seed as a long value
    this(new Random(seed));
  }

  public BagelGame(Random rand) {
    // This is the "real" constructor, which expects an instance of
    // Random to use for shuffling the digits of the secret number.

    // Since the digits cannot repeat in our "number", we can't just
    // pick three random 0-9 integers. Instead, we'll treat it like
    // a deck of ten cards, numbered 0-9.
    List<Integer> digits = new ArrayList<Integer>(10);
    // The 10 specified initial allocation, not actual size,
    // which is why we add rather than set each element...
    for (int i = 0; i < 10; i++) {
      digits.add(i);
    }
    // Collections offers a handy-dandy shuffle method. Normally it
    // uses a fresh Random class PRNG, but we're supplying our own
    // to give us controll over whether or not we set the seed
    Collections.shuffle(digits, rand);

    // Just take the first three digits
    secretNum = digits.subList(0, 3);
  }

  public boolean isOver() {
    return state != GameState.RUNNING;
  }

  public boolean isWon() {
    return state == GameState.WON;
  }

  public int getGuessNum() {
    return guessNum;
  }

  public String getSecretAsString() {
    // Convert the secret number to a three-character string
    String secretString = "";
    for (int n : secretNum) {
      secretString += n;
    }
    return secretString;
  }

  @Override
  public String toString() {
    // Quick report of game state for debugging purposes
    String s = "Game is " + state + "\n";
    s += "Current Guess Number: " + guessNum + "\n";
    s += "Secret Number: " + secretNum;
    return s;
  }

  public String validateGuess(String guess) {
    // Checks the passed string and returns null if it's a valid guess
    // (i.e., exactly three numeric characters)
    // If not valid, returns an "error" string to display to user.
    String error = "";

    if (guess.length() == 3) {
      // Correct length. Are all the characters numbers?
      try {
        Integer.parseInt(guess);
      } catch (NumberFormatException ex) {
        error = "What?";
      }
      if (error == "") {
        // Check for unique digits by placing each character in a set
        Set<Character> uniqueDigits = new HashSet<Character>();
        for (int i = 0; i < guess.length(); i++){
          uniqueDigits.add(guess.charAt(i));
        }
        if (uniqueDigits.size() != guess.length()) {
          error = "Oh, I forgot to tell you that the number I have in mind\n";
          error += "has no two digits the same.";
        }
      }
    } else {
      error = "Try guessing a three-digit number.";
    }

    return error;
  }

  public String makeGuess(String s) throws IllegalArgumentException {
    // Processes the passed guess string (which, ideally, should be
    // validated by previously calling validateGuess)
    // Return a response string (PICO, FERMI, etc.) if valid
    // Also sets game state accordingly (sets win state or increments
    // number of guesses)

    // Convert string to integer list, just to keep things civil
    List<Integer> guess = new ArrayList<Integer>(3);
    for (int i = 0; i < 3; i++) {
      guess.add((int)s.charAt(i) - 48);
    }

    // Build response string...
    String response = "";
    // Correct digit, but in wrong place?
    for (int i = 0; i < 2; i++) {
      if (secretNum.get(i) == guess.get(i+1)) {
        response += "PICO ";
      }
      if (secretNum.get(i+1) == guess.get(i)) {
        response += "PICO ";
      }
    }
    if (secretNum.get(0) == guess.get(2)) {
      response += "PICO ";
    }
    if (secretNum.get(2) == guess.get(0)) {
      response += "PICO ";
    }
    // Correct digits in right place?
    for (int i = 0; i < 3; i++) {
      if (secretNum.get(i) == guess.get(i)) {
        response += "FERMI ";
      }
    }
    // Nothin' right?
    if (response == "") {
      response = "BAGELS";
    }
    // Get rid of any space that might now be at the end
    response = response.trim();
    // If correct, change state
    if (response.equals(CORRECT)) {
      state = GameState.WON;
    } else {
      // If not, increment guess counter and check for game over
      guessNum++;
      if (guessNum > MAX_GUESSES) {
        state = GameState.LOST;
      }
    }
    return response;
  }

}

```

# `05_Bagels/java/Bagels.java`

这段代码是一个简单的Bagels游戏。这个程序的作用是让用户猜一个3位数的密码，并且提供了20次猜测的机会。在每次猜测后，程序会给出一个提示，告诉用户猜大了还是猜小了。用户最多可以猜20次，如果20次都没猜对，那么程序就会告诉用户密码是什么。


```
/******************************************************************************
*
* Bagels
*
* From: BASIC Computer Games (1978)
*       Edited by David H. Ahl
*
* "In this game, the computer picks a 3-digit secret number using
*  the digits 0 to 9 and you attempt to guess what it is.  You are
*  allowed up to twenty guesses.  No digit is repeated.  After
*  each guess the computer will give you clues about your guess
*  as follows:
*
*  PICO     One digit is correct, but in the wrong place
*  FERMI    One digit is in the correct place
```

这段代码是一个用于玩数字谜题的游戏，玩家需要根据提示从谜题中推断出正确的数字，并尝试不断改进自己的策略。程序允许最多20次猜测，如果使用正确的策略，猜测时间不会超过8次。

这段代码的具体实现可能还有所不同，但大致的功能和流程就是这样。


```
*  BAGELS   No digit is correct
*
* "You will learn to draw inferences from the clues and, with
*  practice, you'll learn to improve your score.  There are several
*  good strategies for playing Bagels.  After you have found a good
*  strategy, see if you can improve it.  Or try a different strategy
*  altogether and see if it is any better.  While the program allows
*  up to twenty guesses, if you use a good strategy it should not
*  take more than eight guesses to get any number.
*
* "The original authors of this program are D. Resek and P. Rowe of
*  the Lawrence Hall of Science, Berkeley, California."
*
* Java port by Jeff Jetton, 2020, based on an earlier Python port
*
```

This is a Java program that uses a game of BagelBoss. The program has two main parts, the game logic and the game interface. The game logic is responsible for keeping track of the game state and checking if the game is over or not. The game interface is responsible for displaying the game state to the player and taking user input.

The program has several features such as the ability to play again, to display the winning number of guesses and the secret number once the game is over.

Overall, the program provides a simple game of BagelBoss that can be played by the user.


```
******************************************************************************/

import java.util.Scanner;

public class Bagels {

  public static void main(String[] args) {

    int gamesWon = 0;

    // Intro text
    System.out.println("\n\n                Bagels");
    System.out.println("Creative Computing  Morristown, New Jersey");
    System.out.println("\n\n");
    System.out.print("Would you like the rules (Yes or No)? ");

    // Need instructions?
    Scanner scan = new Scanner(System.in);
    String s = scan.nextLine();
    if (s.length() == 0 || s.toUpperCase().charAt(0) != 'N') {
      System.out.println();
      System.out.println("I am thinking of a three-digit number.  Try to guess");
      System.out.println("my number and I will give you clues as follows:");
      System.out.println("   PICO   - One digit correct but in the wrong position");
      System.out.println("   FERMI  - One digit correct and in the right position");
      System.out.println("   BAGELS - No digits correct");
    }

    // Loop for playing multiple games
    boolean stillPlaying = true;
    while(stillPlaying) {

      // Set up a new game
      BagelGame game = new BagelGame();
      System.out.println("\nO.K.  I have a number in mind.");

      // Loop guess and responsses until game is over
      while (!game.isOver()) {
        String guess = getValidGuess(game);
        String response = game.makeGuess(guess);
        // Don't print a response if the game is won
        if (!game.isWon()) {
          System.out.println(response);
        }
      }

      // Game is over. But did we win or lose?
      if (game.isWon()) {
        System.out.println("You got it!!!\n");
        gamesWon++;
      } else {
        System.out.println("Oh well");
        System.out.print("That's " + BagelGame.MAX_GUESSES + " guesses.  ");
        System.out.println("My number was " + game.getSecretAsString());
      }

      stillPlaying = getReplayResponse();
    }

    // Print goodbye message
    if (gamesWon > 0) {
      System.out.println("\nA " + gamesWon + " point Bagels buff!!");
    }
    System.out.println("Hope you had fun.  Bye.\n");
  }

  private static String getValidGuess(BagelGame game) {
    // Keep asking for a guess until valid
    Scanner scan = new Scanner(System.in);
    boolean valid = false;
    String guess = "";
    String error;

    while (!valid) {
      System.out.print("Guess # " + game.getGuessNum() + "     ? ");
      guess = scan.nextLine().trim();
      error = game.validateGuess(guess);
      if (error == "") {
        valid = true;
      } else {
        System.out.println(error);
      }
    }
    return guess;
  }

  private static boolean getReplayResponse() {
    // keep asking for response until valid
    Scanner scan = new Scanner(System.in);
    // Keep looping until a non-zero-length string is entered
    while (true) {
      System.out.print("Play again (Yes or No)? ");
      String response = scan.nextLine().trim();
      if (response.length() > 0) {
        return response.toUpperCase().charAt(0) == 'Y';
      }
    }
  }

}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `05_Bagels/javascript/bagels.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。

1. `print` 函数的作用是在页面上输出一个字符串。它接受一个字符串参数 `str`，并将其转换为文本节点添加到页面上元素 `output` 的 innerHTML 属性中。具体实现是通过创建一个 text 节点，设置其样式，然后将 `str` 字符串作为其 textContent 属性值进行添加。

2. `input` 函数的作用是从用户接收输入一行字符串。它接收一个 `str` 参数，用于存储用户输入的字符串。函数会在页面上添加一个输入框，并将其样式设置为允许用户输入一行字符。当用户点击输入框时，函数会将 `str` 参数的值存储在 `input` 变量中，然后使用户可以输入一行字符。函数会在用户输入后自动将 `str` 参数打印出来，并将输入框内容保留以便后续处理。

注意：该代码存在一些问题，如：未对 `print` 函数进行文档说明，未定义 `nanochess` 函数


```
// BAGELS
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

这两行代码定义了一个名为 `tab` 的函数，其作用是打印出指定字符数组 `space` 中的内容。

在函数中，首先创建一个空字符串 `str`，然后使用一个 while 循环，该循环从 `space` 变量的初始值(即 0)开始，每次减 1，当 `space` 大于 0 时循环停止。在循环体内，使用一个空格 `space-- > 0` 将字符添加到字符串的末尾，并使用 `str += " "` 将一个空格字符串中的空白添加到 `str` 中。

最后，函数返回字符串 `str`，并在调用该函数时分别打印出两个字符串，每个字符串由空格分隔。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

print(tab(33) + "BAGELS\n");
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");

// *** Bagles number guessing game
// *** Original source unknown but suspected to be
// *** Lawrence Hall of Science, U.C. Berkeley

```

It looks like you have written a program that takes a number input from the user and checks if it is a palindrome. If it is, it prints a special message and then gives the user feedback. If it is not a palindrome, it prints a different message and then gives the user feedback based on how many guesses they made.

The program first checks if the input is a two-digit number. If it is not a palindrome, it then checks if the input is a two-digit number by comparing the first and last digits to see if they are the same. If they are not the same, the program checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number by comparing each digit to see if it is the same. If it is not a palindrome, the program then checks if the input is a two-digit number


```
a1 = [0,0,0,0];
a = [0,0,0,0];
b = [0,0,0,0];

y = 0;
t = 255;

print("\n");
print("\n");
print("\n");

// Main program
async function main()
{
    while (1) {
        print("WOULD YOU LIKE THE RULES (YES OR NO)");
        str = await input();
        if (str.substr(0, 1) != "N") {
            print("\n");
            print("I AM THINKING OF A THREE-DIGIT NUMBER.  TRY TO GUESS\n");
            print("MY NUMBER AND I WILL GIVE YOU CLUES AS FOLLOWS:\n");
            print("   PICO   - ONE DIGIT CORRECT BUT IN THE WRONG POSITION\n");
            print("   FERMI  - ONE DIGIT CORRECT AND IN THE RIGHT POSITION\n");
            print("   BAGELS - NO DIGITS CORRECT\n");
        }
        for (i = 1; i <= 3; i++) {
            do {
                a[i] = Math.floor(Math.random() * 10);
                for (j = i - 1; j >= 1; j--) {
                    if (a[i] == a[j])
                        break;
                }
            } while (j >= 1) ;
        }
        print("\n");
        print("O.K.  I HAVE A NUMBER IN MIND.\n");
        for (i = 1; i <= 20; i++) {
            while (1) {
                print("GUESS #" + i);
                str = await input();
                if (str.length != 3) {
                    print("TRY GUESSING A THREE-DIGIT NUMBER.\n");
                    continue;
                }
                for (z = 1; z <= 3; z++)
                    a1[z] = str.charCodeAt(z - 1);
                for (j = 1; j <= 3; j++) {
                    if (a1[j] < 48 || a1[j] > 57)
                        break;
                    b[j] = a1[j] - 48;
                }
                if (j <= 3) {
                    print("WHAT?");
                    continue;
                }
                if (b[1] == b[2] || b[2] == b[3] || b[3] == b[1]) {
                    print("OH, I FORGOT TO TELL YOU THAT THE NUMBER I HAVE IN MIND\n");
                    print("HAS NO TWO DIGITS THE SAME.\n");
                    continue;
                }
                break;
            }
            c = 0;
            d = 0;
            for (j = 1; j <= 2; j++) {
                if (a[j] == b[j + 1])
                    c++;
                if (a[j + 1] == b[j])
                    c++;
            }
            if (a[1] == b[3])
                c++;
            if (a[3] == b[1])
                c++;
            for (j = 1; j <= 3; j++) {
                if (a[j] == b[j])
                    d++;
            }
            if (d == 3)
                break;
            for (j = 0; j < c; j++)
                print("PICO ");
            for (j = 0; j < d; j++)
                print("FERMI ");
            if (c + d == 0)
                print("BAGELS");
            print("\n");
        }
        if (i <= 20) {
            print("YOU GOT IT!!!\n");
            print("\n");
        } else {
            print("OH WELL.\n");
            print("THAT'S A TWENTY GUESS.  MY NUMBER WAS " + a[1] + a[2] + a[3]);
        }
        y++;
        print("PLAY AGAIN (YES OR NO)");
        str = await input();
        if (str.substr(0, 1) != "Y")
            break;
    }
    if (y == 0)
        print("HOPE YOU HAD FUN.  BYE.\n");
    else
        print("\nA " + y + " POINT BAGELS BUFF!!\n");

}

```

这是C++程序的main函数，是程序的控制中心。程序从这里开始执行，因此所有后续代码都在这里。main函数的实现是程序启动的第一步，通常会负责加载资源、初始化数据和设置计数器等。对于C++程序，如果没有其他文件包含main函数，那么程序将无法运行。


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


# `05_Bagels/python/bagels.py`

这段代码是一个BASIC程序，用于生成一个3位数的随机密码。用户被要求在最多20次尝试后猜测一个3位数密码，而程序会给出每次猜测的提示，以帮助用户更好地理解猜想的正确性。

具体来说，这段代码可以分为以下几个部分：

1. 一个继承自`Song`类的`Bagels`类，用于生成密码和给出提示。
2. 一个包含三个方法：`make_guess()`，`get_clues()`和`游戏次数()`。
3. `make_guess()`方法，用于生成一个新的提示，用于帮助用户理解猜想的正确性。
4. `get_clues()`方法，用于获取系统给出的提示，以便用户了解谜题的难度。
5. `游戏次数()`方法，用于计算游戏失败的用户次数。
6. 在主程序部分，首先创建一个`Bagels`实例，然后调用`make_guess()`方法生成提示，等待用户输入猜测的密码。如果用户在20次尝试后仍未猜中密码，程序会调用`游戏次数()`方法来计算失败的用户次数，并继续生成新的提示。如此循环，直到用户猜中密码，程序才会停止生成提示，并显示成功猜中的信息。


```
"""
Bagels

From: BASIC Computer Games (1978)
      Edited by David H. Ahl

"In this game, the computer picks a 3-digit secret number using
 the digits 0 to 9 and you attempt to guess what it is.  You are
 allowed up to twenty guesses.  No digit is repeated.  After
 each guess the computer will give you clues about your guess
 as follows:

 PICO     One digit is correct, but in the wrong place
 FERMI    One digit is in the correct place
 BAGELS   No digit is correct

```

这段代码是一个用于解释如何玩纸牌的程序，它可以从给出的提示中学会推断，并通过练习来提高得分。它介绍了几种有效的纸牌游戏策略，包括斗地主、斗拱等。在游戏的初始阶段，它允许用户进行最多20次的猜测。

这段代码的作用是向用户介绍如何通过观察提示来推断出牌的策略，并通过练习来提高得分。它鼓励用户尝试不同的策略，以提高他们的得分。


```
"You will learn to draw inferences from the clues and, with
 practice, you'll learn to improve your score.  There are several
 good strategies for playing Bagels.  After you have found a good
 strategy, see if you can improve it.  Or try a different strategy
 altogether and see if it is any better.  While the program allows
 up to twenty guesses, if you use a good strategy it should not
 take more than eight guesses to get any number.

"The original authors of this program are D. Resek and P. Rowe of
 the Lawrence Hall of Science, Berkeley, California."


Python port by Jeff Jetton, 2019
"""


```

这段代码的作用是让用户猜测一个三位数，并在猜测错误时给出提示。当用户猜对数字时，程序会返回一个由该数字的每个数字组成的字符串列表。

具体来说，代码中定义了一个名为`MAX_GUESSES`的变量，用于确定最多可以猜测多少次。接下来，定义了一个名为`print_rules`的函数，该函数用于打印出所有的规则，包括如何猜测、如何提供提示等。

然后，定义了一个名为`pick_number`的函数，该函数返回一个由数字0到9组成的随机列表，或者一个由数字0到9组成的字符串列表。

最后，在程序的主循环中，调用`print_rules`函数来打印出所有的规则，然后调用`pick_number`函数来生成一个随机的三位数字，并将其打印出来。接下来，程序将随机等待用户的猜测，直到猜对或猜错20次为止。如果猜对，程序将返回该数字的每个数字的字符串表示形式。


```
import random
from typing import List

MAX_GUESSES = 20


def print_rules() -> None:
    print("\nI am thinking of a three-digit number.  Try to guess")
    print("my number and I will give you clues as follows:")
    print("   PICO   - One digit correct but in the wrong position")
    print("   FERMI  - One digit correct and in the right position")
    print("   BAGELS - No digits correct")


def pick_number() -> List[str]:
    # Note that this returns a list of individual digits
    # as separate strings, not a single integer or string
    numbers = list(range(10))
    random.shuffle(numbers)
    num = numbers[0:3]
    num_str = [str(i) for i in num]
    return num_str


```

这段代码定义了一个名为 `get_valid_guess` 的函数，它接受一个整数参数 `guesses`。函数的作用是帮助用户猜测一个有效的数字，它通过不断地询问用户直到提供一个有效的数字来达到这个目的。

函数首先判断一个前提条件：所有输入的数字必须是一个的三位整数。然后，它询问用户输入数字，并检查该数字是否由三个不同的数字组成。如果是，那么函数会跳过询问用户输入的数字，这意味着它已经得到了一个有效的数字。否则，函数会提示用户猜测一个三位数，并要求他们再次输入数字。

如果用户在两次输入之间没有提供有效的数字，那么函数会提示用户重新猜测。如果用户最终提供了有效的数字，那么函数会返回它。


```
def get_valid_guess(guesses: int) -> str:
    valid = False
    while not valid:
        guess = input(f"Guess # {guesses}     ? ")
        guess = guess.strip()
        # Guess must be three characters
        if len(guess) == 3:
            # And they should be numbers
            if guess.isnumeric():
                # And the numbers must be unique
                if len(set(guess)) == 3:
                    valid = True
                else:
                    print("Oh, I forgot to tell you that the number I have in mind")
                    print("has no two digits the same.")
            else:
                print("What?")
        else:
            print("Try guessing a three-digit number.")

    return guess


```

这段代码定义了一个名为 `build_result_string` 的函数，它接受两个参数，一个是数字列表 `num`，另一个是猜测字符串 `guess`。函数返回一个字符串，表示结果。

函数内部的逻辑如下：

1. 首先初始化结果字符串 `result` 为空字符串。
2. 遍历 `num` 列表中的前两个元素，检查它们是否与 `guess` 中的两个元素相同。如果是，将 `"PICO"` 字符添加到 `result` 字符串的末尾。
3. 如果已经遍历完全部 `num` 列表，但是还没有遍历到 `guess` 列表的剩余元素，将 `"PICO"` 字符添加到 `result` 字符串的末尾。
4. 如果 `num` 列表中的第一个元素与 `guess` 列表中的第二个元素相同，将 `"PICO"` 字符添加到 `result` 字符串的末尾。
5. 如果 `num` 和 `guess` 列表中的所有元素都不相同，将 `"BAGELS"` 字符添加到 `result` 字符串的末尾，表示没有正确匹配到任何数。

最终，函数返回的结果字符串将取决于 `num` 和 `guess` 参数。


```
def build_result_string(num: List[str], guess: str) -> str:
    result = ""

    # Correct digits in wrong place
    for i in range(2):
        if num[i] == guess[i + 1]:
            result += "PICO "
        if num[i + 1] == guess[i]:
            result += "PICO "
    if num[0] == guess[2]:
        result += "PICO "
    if num[2] == guess[0]:
        result += "PICO "

    # Correct digits in right place
    for i in range(3):
        if num[i] == guess[i]:
            result += "FERMI "

    # Nothing right?
    if result == "":
        result = "BAGELS"

    return result


```

这段代码是一个Python程序，主要作用是让玩家猜测一个两位数的密码，如果猜测正确，就会显示游戏胜利，否则会显示游戏失败并重新开始游戏。

具体来说，程序首先会生成一个包含两个字符的提示信息，告诉玩家这是一道密码题，然后让玩家输入是玩还是不玩。接着程序会生成一个数字，用于提示玩家要猜的密码，并且让玩家开始猜测。

在每次猜测后，程序会告诉玩家猜大了还是猜小了，并且如果猜对了密码，就会显示游戏胜利，否则重新开始游戏。如果玩家在规定的猜测次数内没有猜对密码，程序会显示游戏失败并重新开始游戏。

最后，程序会根据猜测的密码是否正确来决定是否显示游戏胜利，并且程序会在程序结束时输出一些鼓励的话语。


```
def main() -> None:
    # Intro text
    print("\n                Bagels")
    print("Creative Computing  Morristown, New Jersey\n\n")

    # Anything other than N* will show the rules
    response = input("Would you like the rules (Yes or No)? ")
    if len(response) > 0:
        if response.upper()[0] != "N":
            print_rules()
    else:
        print_rules()

    games_won = 0
    still_running = True
    while still_running:

        # New round
        num = pick_number()
        num_str = "".join(num)
        guesses = 1

        print("\nO.K.  I have a number in mind.")
        guessing = True
        while guessing:

            guess = get_valid_guess(guesses)

            if guess == num_str:
                print("You got it!!!\n")
                games_won += 1
                guessing = False
            else:
                print(build_result_string(num, guess))
                guesses += 1
                if guesses > MAX_GUESSES:
                    print("Oh well")
                    print(f"That's {MAX_GUESSES} guesses.  My number was {num_str}")
                    guessing = False

        valid_response = False
        while not valid_response:
            response = input("Play again (Yes or No)? ")
            if len(response) > 0:
                valid_response = True
                if response.upper()[0] != "Y":
                    still_running = False

    if games_won > 0:
        print(f"\nA {games_won} point Bagels buff!!")

    print("Hope you had fun.  Bye.\n")


```

这段代码是一个用于验证程序功能的if语句，它仅在程序作为主函数（__main__）运行时执行一次。程序的主要功能是确保用户输入的正确性，并允许用户输入最大数量（MAX_NUM）的值。

当程序作为主函数运行时，if语句将检查输入是否符合预期。如果用户输入的值不符合预期，if语句将输出错误信息并暂停程序。如果输入符合预期，if语句将允许程序继续运行。

在此代码中，第二行注释（# Porting Notes）指出程序已经验证了玩家输入的正确性，并复制了原始程序的功能。

关于建议的修改，该程序已经很完善，没有太多可修改的地方。但是，可以在程序中增加一些提示信息，以提高代码的可读性和可维护性。


```
if __name__ == "__main__":
    main()

######################################################################
#
# Porting Notes
#
#   The original program did an unusually good job of validating the
#   player's input (compared to many of the other programs in the
#   book). Those checks and responses have been exactly reproduced.
#
#
# Ideas for Modifications
#
#   It should probably mention that there's a maximum of MAX_NUM
```

这段代码是一个用于创建猜数字程序的基本实现。它定义了一个名为guesses的变量，但没有为其指定任何值。接着，它考虑了一个猜测范围的问题，即使用0到9中的任意数字。如果猜测的范围不同，它会对程序进行修改，以支持从2到6位数字的猜测。

在程序的其他部分，没有对guesses变量进行使用，因此它是一个可能的输入，可以根据需要进行更改。


```
#   guesses in the instructions somewhere, shouldn't it?
#
#   Could this program be written to use anywhere from, say 2 to 6
#   digits in the number to guess? How would this change the routine
#   that creates the "result" string?
#
######################################################################

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


## BAGELS

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/) by [Tom Armitage](https://github.com/infovore)

## Translator's notes:

This is a highly imperative port. As such, it's very much, in the spirit of David Ahl's original version, and also highly un-Rubyish.

A few decisions I made:

* the main loop is a 'while' loop. Most games are a main loop that runs until it doesn't, and I felt that "while the player wished to keep playing, the game should run" was an appropriate structure.
* lots of puts and gets; that feels appropriate to the Ahl implementation. No clever cli or curses libraries here.
* the number in question, and the player's answer, are stored as numbers. They're only converted into arrays for the purpose of `puts_clue_for` - ie, when comparison is need. The original game stored them as arrays, which made sense, but given the computer says "I have a number in mind", I decided to store what was in its 'mind' as a number.
* the `String#center` method from Ruby 2.5~ sure is handy.


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by Till Klister [tikste@github](https://github.com/tikste).


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Banner

This program creates a large banner on a terminal of any message you input. The letters may be any dimension of you wish although the letter height plus distance from left-hand side should not exceed 6 inches. Experiment with the height and width until you get a pleasing effect on whatever terminal you are using.

This program was written by Leonard Rosendust of Brooklyn, New York.

---

As published in Basic Computer Games (1978)
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=10)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=25)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

- The "SET PAGE" input, stored in `O$`, has no effect.  It was probably meant as an opportunity for the user to set their pin-feed printer to the top of the page before proceeding.

- The data values for each character are the bit representation of each horizontal row of the printout (vertical column of a character), plus one.  Perhaps because of this +1, the original code (and some of the ports here) are much more complicated than they need to be.

(please note any difficulties or challenges in porting here)



# `06_Banner/csharp/banner.cs`

This is a class called `Banner` that prints a banner-style text based on the parameters passed to it. It uses a `for` loop to iterate through each character in the `Statement` field, which is the text to be printed.

The `PrintBanner` method prints the entire banner based on the parameters passed to it. It does this by first adding a spacer for each character (2 spaces for each character) and then printing the character by character.

It also includes a `PrintChar` method that prints a single character at a time. This method takes two arguments: the character to be printed and the background color for the character.

In the `Play` method, the `Banner` class takes a `Statement` field, which is the text to be printed. It then calls the `PrintBanner` method, passing in the `Statement` as a parameter.

In the `Main` method, a new `Banner` object is created and called the `Play` method. This method prints the banner to the console.


```
﻿using System;
using System.Collections.Generic;
using System.Linq;

namespace banner
{
    class Banner
    {
        private int Horizontal { get; set; }
        private int Vertical { get; set; }
        private bool Centered { get; set; }
        private string Character { get; set; }
        private string Statement { get; set; }

        // This provides a bit-ended representation of each symbol
        // that can be output.  Each symbol is defined by 7 parts -
        // where each part is an integer value that, when converted to
        // the binary representation, shows which section is filled in
        // with values and which are spaces.  i.e., the 'filled in'
        // parts represent the actual symbol on the paper.
        Dictionary<char, int[]> letters = new Dictionary<char, int[]>()
        {
            {' ', new int[] { 0, 0, 0, 0, 0, 0, 0 } },
            {'A', new int[] {505, 37, 35, 34, 35, 37, 505} },
            {'B', new int[] {512, 274, 274, 274, 274, 274, 239} },
            {'C', new int[] {125, 131, 258, 258, 258, 131, 69} },
            {'D', new int[] {512, 258, 258, 258, 258, 131, 125} },
            {'E', new int[] {512, 274, 274, 274, 274, 258, 258} },
            {'F', new int[] {512, 18, 18, 18, 18, 2, 2} },
            {'G', new int[] {125, 131, 258, 258, 290, 163, 101} },
            {'H', new int[] {512, 17, 17, 17, 17, 17, 512} },
            {'I', new int[] {258, 258, 258, 512, 258, 258, 258} },
            {'J', new int[] {65, 129, 257, 257, 257, 129, 128} },
            {'K', new int[] {512, 17, 17, 41, 69, 131, 258} },
            {'L', new int[] {512, 257, 257, 257, 257, 257, 257} },
            {'M', new int[] {512, 7, 13, 25, 13, 7, 512} },
            {'N', new int[] {512, 7, 9, 17, 33, 193, 512} },
            {'O', new int[] {125, 131, 258, 258, 258, 131, 125} },
            {'P', new int[] {512, 18, 18, 18, 18, 18, 15} },
            {'Q', new int[] {125, 131, 258, 258, 322, 131, 381} },
            {'R', new int[] {512, 18, 18, 50, 82, 146, 271} },
            {'S', new int[] {69, 139, 274, 274, 274, 163, 69} },
            {'T', new int[] {2, 2, 2, 512, 2, 2, 2} },
            {'U', new int[] {128, 129, 257, 257, 257, 129, 128} },
            {'V', new int[] {64, 65, 129, 257, 129, 65, 64} },
            {'W', new int[] {256, 257, 129, 65, 129, 257, 256} },
            {'X', new int[] {388, 69, 41, 17, 41, 69, 388} },
            {'Y', new int[] {8, 9, 17, 481, 17, 9, 8} },
            {'Z', new int[] {386, 322, 290, 274, 266, 262, 260} },
            {'0', new int[] {57, 69, 131, 258, 131, 69, 57} },
            {'1', new int[] {0, 0, 261, 259, 512, 257, 257} },
            {'2', new int[] {261, 387, 322, 290, 274, 267, 261} },
            {'3', new int[] {66, 130, 258, 274, 266, 150, 100} },
            {'4', new int[] {33, 49, 41, 37, 35, 512, 33} },
            {'5', new int[] {160, 274, 274, 274, 274, 274, 226} },
            {'6', new int[] {194, 291, 293, 297, 305, 289, 193} },
            {'7', new int[] {258, 130, 66, 34, 18, 10, 8} },
            {'8', new int[] {69, 171, 274, 274, 274, 171, 69} },
            {'9', new int[] {263, 138, 74, 42, 26, 10, 7} },
            {'?', new int[] {5, 3, 2, 354, 18, 11, 5} },
            {'*', new int[] {69, 41, 17, 512, 17, 41, 69} },
            {'=', new int[] {41, 41, 41, 41, 41, 41, 41} },
            {'!', new int[] {1, 1, 1, 384, 1, 1, 1} },
            {'.', new int[] {1, 1, 129, 449, 129, 1, 1} }
        };


        /// <summary>
        /// This displays the provided text on the screen and then waits for the user
        /// to enter a integer value greater than 0.
        /// </summary>
        /// <param name="DisplayText">Text to display on the screen asking for the input</param>
        /// <returns>The integer value entered by the user</returns>
        private int GetNumber(string DisplayText)
        {
            Console.Write(DisplayText);
            string TempStr = Console.ReadLine();

            Int32.TryParse(TempStr, out int TempInt);

            if (TempInt <= 0)
            {
                throw new ArgumentException($"{DisplayText} must be greater than zero");
            }

            return TempInt;
        }

        /// <summary>
        /// This displays the provided text on the screen and then waits for the user
        /// to enter a Y or N.  It cheats by just looking for a 'y' and returning that
        /// as true.  Anything else that the user enters is returned as false.
        /// </summary>
        /// <param name="DisplayText">Text to display on the screen asking for the input</param>
        /// <returns>Returns true or false</returns>
        private bool GetBool(string DisplayText)
        {
            Console.Write(DisplayText);
            return (Console.ReadLine().StartsWith("y", StringComparison.InvariantCultureIgnoreCase));
        }

        /// <summary>
        /// This displays the provided text on the screen and then waits for the user
        /// to enter an arbitrary string.  That string is then returned 'as-is'.
        /// </summary>
        /// <param name="DisplayText">Text to display on the screen asking for the input</param>
        /// <returns>The string entered by the user.</returns>
        private string GetString(string DisplayText)
        {
            Console.Write(DisplayText);
            return (Console.ReadLine().ToUpper());
        }

        /// <summary>
        /// This queries the user for the various inputs needed by the program.
        /// </summary>
        private void GetInput()
        {
            Horizontal = GetNumber("Horizontal ");
            Vertical = GetNumber("Vertical ");
            Centered = GetBool("Centered ");
            Character = GetString("Character (type 'ALL' if you want character being printed) ");
            Statement = GetString("Statement ");
            // We don't care about what the user enters here.  This is just telling them
            // to set the page in the printer.
            _ = GetString("Set page ");
        }

        /// <summary>
        /// This prints out a single character of the banner - adding
        /// a few blanks lines as a spacer between characters.
        /// </summary>
        private void PrintChar(char ch)
        {
            // In the trivial case (a space character), just print out the spaces
            if (ch.Equals(' '))
            {
                Console.WriteLine(new string('\n', 7 * Horizontal));
                return;
            }

            // If a specific character to be printed was provided by the user,
            // then user that as our ouput character - otherwise take the
            // current character
            char outCh = Character == "ALL" ? ch : Character[0];
            int[] letter = new int[7];
            try
            {
                letters[outCh].CopyTo(letter, 0);
            }
            catch (KeyNotFoundException)
            {
                throw new KeyNotFoundException($"The provided letter {outCh} was not found in the letters list");
            }

            // This iterates through each of the parts that make up
            // each letter.  Each part represents 1 * Horizontal lines
            // of actual output.
            for (int idx = 0; idx < 7; idx++)
            {
                // New int array declarations default to zeros
                // numSections decides how many 'sections' need to be printed
                // for a given line of each character
                int[] numSections = new int[7];
                // fillInSection decides whether each 'section' of the
                // character gets filled in with the character or with blanks
                int[] fillInSection = new int[9];

                // This uses the value in each part to decide which
                // sections are empty spaces in the letter or filled in
                // spaces.  For each section marked with 1 in fillInSection,
                // that will correspond to 1 * Vertical characters actually
                // being output.
                for (int exp = 8; exp >= 0; exp--)
                {
                    if (Math.Pow(2, exp) < letter[idx])
                    {
                        fillInSection[8 - exp] = 1;
                        letter[idx] -= (int)Math.Pow(2, exp);
                        if (letter[idx] == 1)
                        {
                            // Once we've exhausted all of the sections
                            // defined in this part of the letter, then
                            // we marked that number and break out of this
                            // for loop.
                            numSections[idx] = 8 - exp;
                            break;
                        }
                    }
                }

                // Now that we know which sections of this part of the letter
                // are filled in or spaces, we can actually create the string
                // to print out.
                string lineStr = "";

                if (Centered)
                    lineStr += new string(' ', (int)(63 - 4.5 * Vertical) * 1 / 1 + 1);

                for (int idx2 = 0; idx2 <= numSections[idx]; idx2++)
                {
                    lineStr = lineStr + new string(fillInSection[idx2] == 0 ? ' ' : outCh, Vertical);
                }

                // Then we print that string out 1 * Horizontal number of times
                for (int lineidx = 1; lineidx <= Horizontal; lineidx++)
                {
                    Console.WriteLine(lineStr);
                }
            }

            // Finally, add a little spacer after each character for readability.
            Console.WriteLine(new string('\n', 2 * Horizontal - 1));
        }

        /// <summary>
        /// This prints the entire banner based in the parameters
        /// the user provided.
        /// </summary>
        private void PrintBanner()
        {
            // Iterate through each character in the statement
            foreach (char ch in Statement)
            {
                PrintChar(ch);
            }

            // In the original version, it would print an additional 75 blank
            // lines in order to feed the printer paper...don't really need this
            // since we're not actually printing.
            // Console.WriteLine(new string('\n', 75));
        }

        /// <summary>
        /// Main entry point into the banner class and handles the main loop.
        /// </summary>
        public void Play()
        {
            GetInput();
            PrintBanner();
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            new Banner().Play();
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `06_Banner/java/Banner.java`

这段代码实现了一个简单的BASIC游戏（Banner游戏）版，用于在控制台界面接收玩家输入并输出游戏结果。具体来说，这段代码实现了以下功能：

1. 引入了Scanner类，用于从控制台接收输入；
2. 创建了一个HashMap，用于存储游戏结果，例如胜/负/平局的字段；
3. 通过HashMap实现了Map接口，将游戏结果存储为键值对（即胜/负/平局，键类型为String，值类型为boolean）形式；
4. 实现了BASIC游戏中的规则，包括：
   - 初始化游戏，将胜/负/平局字段设置为false，地图的行数和列数初始化为游戏地图的行数和列数；
   - 显示游戏地图，每一行按照“0 1 2 3 ...”的格式输出，每一列按照“0 1 2 3 ...”的格式输出，如果当前行/列有数值则输出对应的数值；
   - 接收玩家的输入，判断输入是否为'A'或'D'，如果是则交换当前行/列的值，否则不进行交换；
   - 实现了游戏循环，当玩家不输入时，继续执行游戏循环，直到游戏结束；
   - 通过HashMap实现了记忆化功能，当游戏重新加载时，根据地图初始化情况直接返回，避免了重复的输入/输出操作。


```
import java.util.Scanner;
import java.util.HashMap;
import java.util.Map;
import java.lang.Math;

/**
 * Game of Banner
 * <p>
 * Based on the BASIC game of Banner here
 * https://github.com/coding-horror/basic-computer-games/blob/main/06%20Banner/banner.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 *
 * Converted from BASIC to Java by Darren Cardenas.
 */

```

This is a Java class that represents a game of Hangman where the user must choose a word and then guess letters until they have correctly guessed the entire word.

The Hangman game has several features such as the ability to add extra spaces for中心和包含 words, a feature to add a letter if it's not already used, and the ability to resend the game if the user loses.

The game also uses a simple display to show the correctly guessed letters while the incorrect guesses are hidden.

The class also has a control to add a banner at the top with the title "Hangman" and the subtitle "You are currently playing hangman"

It's important to note that the game is not Implemented with the best practices and it's always a good idea to add more comments and informations to the code.


```
public class Banner {

  private final Scanner scan;  // For user input

  public Banner() {

    scan = new Scanner(System.in);

  }  // End of constructor Banner

  public void play() {

    int bitIndex = 0;
    int centerFlag = 0;
    int dataIndex = 0;
    int hIndex = 0;
    int horizontal = 0;
    int index = 0;
    int letterIndex = 0;
    int tempVal = 0;
    int vertical = 0;
    int vIndex = 0;
    int writeIndex = 0;

    int[] writerMap = new int[10];
    int[] writeLimit = new int[10];

    String centerResponse = "";
    String characters = "";
    String letter = "";
    String lineContent = "";
    String setPage = "";
    String statement = "";
    String token = "";  // Print token

    Map<String, int[]> symbolData = new HashMap<String, int[]>();
    symbolData.put(" ", new int[]{0,0,0,0,0,0,0,0              });
    symbolData.put("A", new int[]{0,505,37,35,34,35,37,505     });
    symbolData.put("G", new int[]{0,125,131,258,258,290,163,101});
    symbolData.put("E", new int[]{0,512,274,274,274,274,258,258});
    symbolData.put("T", new int[]{0,2,2,2,512,2,2,2            });
    symbolData.put("W", new int[]{0,256,257,129,65,129,257,256 });
    symbolData.put("L", new int[]{0,512,257,257,257,257,257,257});
    symbolData.put("S", new int[]{0,69,139,274,274,274,163,69  });
    symbolData.put("O", new int[]{0,125,131,258,258,258,131,125});
    symbolData.put("N", new int[]{0,512,7,9,17,33,193,512      });
    symbolData.put("F", new int[]{0,512,18,18,18,18,2,2        });
    symbolData.put("K", new int[]{0,512,17,17,41,69,131,258    });
    symbolData.put("B", new int[]{0,512,274,274,274,274,274,239});
    symbolData.put("D", new int[]{0,512,258,258,258,258,131,125});
    symbolData.put("H", new int[]{0,512,17,17,17,17,17,512     });
    symbolData.put("M", new int[]{0,512,7,13,25,13,7,512       });
    symbolData.put("?", new int[]{0,5,3,2,354,18,11,5          });
    symbolData.put("U", new int[]{0,128,129,257,257,257,129,128});
    symbolData.put("R", new int[]{0,512,18,18,50,82,146,271    });
    symbolData.put("P", new int[]{0,512,18,18,18,18,18,15      });
    symbolData.put("Q", new int[]{0,125,131,258,258,322,131,381});
    symbolData.put("Y", new int[]{0,8,9,17,481,17,9,8          });
    symbolData.put("V", new int[]{0,64,65,129,257,129,65,64    });
    symbolData.put("X", new int[]{0,388,69,41,17,41,69,388     });
    symbolData.put("Z", new int[]{0,386,322,290,274,266,262,260});
    symbolData.put("I", new int[]{0,258,258,258,512,258,258,258});
    symbolData.put("C", new int[]{0,125,131,258,258,258,131,69 });
    symbolData.put("J", new int[]{0,65,129,257,257,257,129,128 });
    symbolData.put("1", new int[]{0,0,0,261,259,512,257,257    });
    symbolData.put("2", new int[]{0,261,387,322,290,274,267,261});
    symbolData.put("*", new int[]{0,69,41,17,512,17,41,69      });
    symbolData.put("3", new int[]{0,66,130,258,274,266,150,100 });
    symbolData.put("4", new int[]{0,33,49,41,37,35,512,33      });
    symbolData.put("5", new int[]{0,160,274,274,274,274,274,226});
    symbolData.put("6", new int[]{0,194,291,293,297,305,289,193});
    symbolData.put("7", new int[]{0,258,130,66,34,18,10,8      });
    symbolData.put("8", new int[]{0,69,171,274,274,274,171,69  });
    symbolData.put("9", new int[]{0,263,138,74,42,26,10,7      });
    symbolData.put("=", new int[]{0,41,41,41,41,41,41,41       });
    symbolData.put("!", new int[]{0,1,1,1,384,1,1,1            });
    symbolData.put("0", new int[]{0,57,69,131,258,131,69,57    });
    symbolData.put(".", new int[]{0,1,1,129,449,129,1,1        });

    System.out.print("HORIZONTAL? ");
    horizontal = Integer.parseInt(scan.nextLine());

    System.out.print("VERTICAL? ");
    vertical = Integer.parseInt(scan.nextLine());

    System.out.print("CENTERED? ");
    centerResponse = scan.nextLine().toUpperCase();

    centerFlag = 0;

    // Lexicographical comparison
    if (centerResponse.compareTo("P") > 0) {
      centerFlag = 1;
    }

    System.out.print("CHARACTER (TYPE 'ALL' IF YOU WANT CHARACTER BEING PRINTED)? ");
    characters = scan.nextLine().toUpperCase();

    System.out.print("STATEMENT? ");
    statement = scan.nextLine().toUpperCase();

    // Initiates the print
    System.out.print("SET PAGE? ");
    setPage = scan.nextLine();

    // Begin loop through statement letters
    for (letterIndex = 1; letterIndex <= statement.length(); letterIndex++) {

      // Extract a letter
      letter = String.valueOf(statement.charAt(letterIndex - 1));

      // Begin loop through all symbol data
      for (String symbolString: symbolData.keySet()) {

        // Begin letter handling
        if (letter.equals(" ")) {
          for (index = 1; index <= (7 * horizontal); index++) {
            System.out.println("");
          }
          break;

        } else if (letter.equals(symbolString)) {
          token = characters;

          if (characters.equals("ALL")) {
            token = symbolString;
          }

          for (dataIndex = 1; dataIndex <= 7; dataIndex++) {

            // Avoid overwriting symbol data
            tempVal = symbolData.get(symbolString)[dataIndex];

            for (bitIndex = 8; bitIndex >= 0; bitIndex--) {

              if (Math.pow(2, bitIndex) < tempVal) {
                writerMap[9 - bitIndex] = 1;
                tempVal -= Math.pow(2, bitIndex);

                if (tempVal == 1) {
                  writeLimit[dataIndex] = 9 - bitIndex;
                  break;
                }

              } else {

                writerMap[9 - bitIndex] = 0;
              }
            }  // End of bitIndex loop

            for (hIndex = 1; hIndex <= horizontal; hIndex++) {

              // Add whitespace for centering
              lineContent = " ".repeat((int)((63 - 4.5 * vertical) * centerFlag / token.length()));

              for (writeIndex = 1; writeIndex <= writeLimit[dataIndex]; writeIndex++) {

                if (writerMap[writeIndex] == 0) {

                  for (vIndex = 1; vIndex <= vertical; vIndex++) {

                    for (index = 1; index <= token.length(); index++) {
                      lineContent += " ";
                    }
                  }

                } else {

                  for (vIndex = 1; vIndex <= vertical; vIndex++) {
                    lineContent += token;
                  }
                }

              }  // End of writeIndex loop

              System.out.println(lineContent);

            }  // End of hIndex loop

          }  // End of dataIndex loop

          // Add padding between letters
          for (index = 1; index <= 2 * horizontal; index++) {
            System.out.println("");
          }

        }  // End letter handling

      }  // End loop through all symbol data

    }  // End loop through statement letters

    // Add extra length to the banner
    for (index = 1; index <= 75; index++) {
      System.out.println("");
    }

  }  // End of method play

  public static void main(String[] args) {

    Banner game = new Banner();
    game.play();

  }  // End of method main

}  // End of class Banner

```