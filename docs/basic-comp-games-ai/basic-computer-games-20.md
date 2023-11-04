# BasicComputerGames源码解析 20

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by Anthony Rubick [AnthonyMichaelTDM](https://github.com/AnthonyMichaelTDM)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Bombardment

BOMBARDMENT is played on two, 5x5 grids or boards with 25 outpost locations numbered 1 to 25. Both you and the computer have four platoons of troops that can be located at any four outposts on your respective grids.

At the start of the game, you locate (or hide) your four platoons on your grid. The computer does the same on its grid. You then take turns firing missiles or bombs at each other’s outposts trying to destroy all four platoons. The one who finds all four opponents’ platoons first, wins.

This program was slightly modified from the original written by Martin Burdash of Parlin, New Jersey.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=22)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=37)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Known Bugs

- Though the instructions say you can't place two platoons on the same outpost, the code does not enforce this.  So the player can "cheat" and guarantee a win by entering the same outpost number two or more times.

#### Porting Notes

- To ensure the instructions don't scroll off the top of the screen, we may want to insert a "(Press Return)" or similar prompt before printing the tear-off matrix.

(please note any difficulties or challenges in porting here)


# `11_Bombardment/csharp/Bombardment.cs`

This is a class in a game. It has methods for generating a random number for the computer to make its guess, handling turns for the computer and human, and printing the results of the game. It also has a Startup method for printing a starting message and setting up the initial positions of the computer and human players.
It also has a PrintStartingMessage method for printing a starting message for the game and a PlaceComputerPlatoons method for placing the computer platoons.
It also has a StoreHumanPositions method for storing the positions of the human players.


```
using System;
using System.Collections.Generic;

namespace Bombardment
{
    // <summary>
    // Game of Bombardment
    // Based on the Basic game of Bombardment here
    // https://github.com/coding-horror/basic-computer-games/blob/main/11%20Bombardment/bombardment.bas
    // Note:  The idea was to create a version of the 1970's Basic game in C#, without introducing
    // new features - no additional text, error checking, etc has been added.
    // </summary>
    internal class Bombardment
    {
        private static int MAX_GRID_SIZE = 25;
        private static int MAX_PLATOONS = 4;
        private static Random random = new Random();
        private List<int> computerPositions = new List<int>();
        private List<int> playerPositions = new List<int>();
        private List<int> computerGuesses = new List<int>();

        private void PrintStartingMessage()
        {
            Console.WriteLine("{0}BOMBARDMENT", new string(' ', 33));
            Console.WriteLine("{0}CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY", new string(' ', 15));
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();

            Console.WriteLine("YOU ARE ON A BATTLEFIELD WITH 4 PLATOONS AND YOU");
            Console.WriteLine("HAVE 25 OUTPOSTS AVAILABLE WHERE THEY MAY BE PLACED.");
            Console.WriteLine("YOU CAN ONLY PLACE ONE PLATOON AT ANY ONE OUTPOST.");
            Console.WriteLine("THE COMPUTER DOES THE SAME WITH ITS FOUR PLATOONS.");
            Console.WriteLine();
            Console.WriteLine("THE OBJECT OF THE GAME IS TO FIRE MISSLES AT THE");
            Console.WriteLine("OUTPOSTS OF THE COMPUTER.  IT WILL DO THE SAME TO YOU.");
            Console.WriteLine("THE ONE WHO DESTROYS ALL FOUR OF THE ENEMY'S PLATOONS");
            Console.WriteLine("FIRST IS THE WINNER.");
            Console.WriteLine();
            Console.WriteLine("GOOD LUCK... AND TELL US WHERE YOU WANT THE BODIES SENT!");
            Console.WriteLine();
            Console.WriteLine("TEAR OFF MATRIX AND USE IT TO CHECK OFF THE NUMBERS.");

            // As an alternative to repeating the call to WriteLine(),
            // we can print the new line character five times.
            Console.Write(new string('\n', 5));

            // Print a sample board (presumably the game was originally designed to be
            // physically printed on paper while played).
            for (var i = 1; i <= 25; i += 5)
            {
                // The token replacement can be padded by using the format {tokenPosition, padding}
                // Negative values for the padding cause the output to be left-aligned.
                Console.WriteLine("{0,-3}{1,-3}{2,-3}{3,-3}{4,-3}", i, i + 1, i + 2, i + 3, i + 4);
            }

            Console.WriteLine("\n");
        }

        // Generate 5 random positions for the computer's platoons.
        private void PlaceComputerPlatoons()
        {
            do
            {
                var nextPosition = random.Next(1, MAX_GRID_SIZE);
                if (!computerPositions.Contains(nextPosition))
                {
                    computerPositions.Add(nextPosition);
                }

            } while (computerPositions.Count < MAX_PLATOONS);
        }

        private void StoreHumanPositions()
        {
            Console.WriteLine("WHAT ARE YOUR FOUR POSITIONS");

            // The original game assumed that the input would be five comma-separated values, all on one line.
            // For example: 12,22,1,4,17
            var input = Console.ReadLine();
            var playerPositionsAsStrings = input.Split(",");
            foreach (var playerPosition in playerPositionsAsStrings) {
                playerPositions.Add(int.Parse(playerPosition));
            }
        }

        private void HumanTurn()
        {
            Console.WriteLine("WHERE DO YOU WISH TO FIRE YOUR MISSLE");
            var input = Console.ReadLine();
            var humanGuess = int.Parse(input);

            if(computerPositions.Contains(humanGuess))
            {
                Console.WriteLine("YOU GOT ONE OF MY OUTPOSTS!");
                computerPositions.Remove(humanGuess);

                switch(computerPositions.Count)
                {
                    case 3:
                        Console.WriteLine("ONE DOWN, THREE TO GO.");
                        break;
                    case 2:
                        Console.WriteLine("TWO DOWN, TWO TO GO.");
                        break;
                    case 1:
                        Console.WriteLine("THREE DOWN, ONE TO GO.");
                        break;
                    case 0:
                        Console.WriteLine("YOU GOT ME, I'M GOING FAST.");
                        Console.WriteLine("BUT I'LL GET YOU WHEN MY TRANSISTO&S RECUP%RA*E!");
                        break;
                }
            }
            else
            {
                Console.WriteLine("HA, HA YOU MISSED. MY TURN NOW:");
            }
        }

        private int GenerateComputerGuess()
        {
            int computerGuess;
            do
            {
                computerGuess = random.Next(1, 25);
            }
            while(computerGuesses.Contains(computerGuess));
            computerGuesses.Add(computerGuess);

            return computerGuess;
        }

        private void ComputerTurn()
        {
            var computerGuess = GenerateComputerGuess();

            if (playerPositions.Contains(computerGuess))
            {
                Console.WriteLine("I GOT YOU. IT WON'T BE LONG NOW. POST {0} WAS HIT.", computerGuess);
                playerPositions.Remove(computerGuess);

                switch(playerPositions.Count)
                {
                    case 3:
                        Console.WriteLine("YOU HAVE ONLY THREE OUTPOSTS LEFT.");
                        break;
                    case 2:
                        Console.WriteLine("YOU HAVE ONLY TWO OUTPOSTS LEFT.");
                        break;
                    case 1:
                        Console.WriteLine("YOU HAVE ONLY ONE OUTPOST LEFT.");
                        break;
                    case 0:
                        Console.WriteLine("YOU'RE DEAD. YOUR LAST OUTPOST WAS AT {0}. HA, HA, HA.", computerGuess);
                        Console.WriteLine("BETTER LUCK NEXT TIME.");
                        break;
                }
            }
            else
            {
                Console.WriteLine("I MISSED YOU, YOU DIRTY RAT. I PICKED {0}. YOUR TURN:", computerGuess);
            }
        }

        public void Play()
        {
            PrintStartingMessage();
            PlaceComputerPlatoons();
            StoreHumanPositions();

            while (playerPositions.Count > 0 && computerPositions.Count > 0)
            {
                HumanTurn();

                if (computerPositions.Count > 0)
                {
                    ComputerTurn();
                }
            }
        }
    }
}

```

# `11_Bombardment/csharp/Program.cs`

这是一个使用C#编写的Bombardment应用程序类，其中包含一个静态的、名为“Main”的函数。

Main函数是应用程序的入口点，当程序运行时，它将首先执行Main函数中的代码。

在Main函数中，我们创建了一个名为“bombardment”的类，该类并未定义任何成员变量或方法。

接着，我们使用Main函数的参数数组参数传递给创建“bombardment”对象的操作，这里并没有传递任何参数。

最后，我们调用了名为“Play”的静态函数，该函数可能是一个包含Bombardment类中方法组成的集合。由于我们没有提供具体的函数实现，所以无法确定“Play”函数会执行什么操作。


```
﻿using System;

namespace Bombardment
{
    class Program
    {
        static void Main(string[] args)
        {
            var bombardment = new Bombardment();
            bombardment.Play();
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `11_Bombardment/java/src/Bombardment.java`

This is a Java class, `Nopera`, which simulates a virtual game of chess. It has several methods for initializing the game, displaying messages on the screen, and accepting player input through the keyboard.

The `init` method initializes the game by creating four locations for the computer's platoons and a set of players' platoons.

The `displayTextAndGetInput` method displays a message on the screen and accepts the player's input. It returns the input as a string.

The `generateRandomNumber` method generates a random number.

You can use this class to simulate a game of chess with multiple players.


```
import java.util.HashSet;
import java.util.Scanner;

/**
 * Game of Bombardment
 * <p>
 * Based on the Basic game of Bombardment here
 * https://github.com/coding-horror/basic-computer-games/blob/main/11%20Bombardment/bombardment.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Bombardment {

    public static final int MAX_GRID_SIZE = 25;
    public static final int PLATOONS = 4;

    private enum GAME_STATE {
        STARTING,
        DRAW_BATTLEFIELD,
        GET_PLAYER_CHOICES,
        PLAYERS_TURN,
        COMPUTER_TURN,
        PLAYER_WON,
        PLAYER_LOST,
        GAME_OVER
    }

    private GAME_STATE gameState;

    public static final String[] PLAYER_HIT_MESSAGES = {"ONE DOWN, THREE TO GO.",
            "TWO DOWN, TWO TO GO.", "THREE DOWN, ONE TO GO."};

    public static final String[] COMPUTER_HIT_MESSAGES = {"YOU HAVE ONLY THREE OUTPOSTS LEFT.",
            "YOU HAVE ONLY TWO OUTPOSTS LEFT.", "YOU HAVE ONLY ONE OUTPOST LEFT."};

    private HashSet<Integer> computersPlatoons;
    private HashSet<Integer> playersPlatoons;

    private HashSet<Integer> computersGuesses;

    // Used for keyboard input
    private final Scanner kbScanner;

    public Bombardment() {

        gameState = GAME_STATE.STARTING;

        // Initialise kb scanner
        kbScanner = new Scanner(System.in);
    }

    /**
     * Main game loop
     */
    public void play() {

        do {
            switch (gameState) {

                // Show an introduction the first time the game is played.
                case STARTING:
                    init();
                    intro();
                    gameState = GAME_STATE.DRAW_BATTLEFIELD;
                    break;

                // Enter the players name
                case DRAW_BATTLEFIELD:
                    drawBattlefield();
                    gameState = GAME_STATE.GET_PLAYER_CHOICES;
                    break;

                // Get the players 4 locations for their platoons
                case GET_PLAYER_CHOICES:
                    String playerChoices = displayTextAndGetInput("WHAT ARE YOUR FOUR POSITIONS? ");

                    // Store the 4 player choices that were entered separated with commas
                    for (int i = 0; i < PLATOONS; i++) {
                        playersPlatoons.add(getDelimitedValue(playerChoices, i));
                    }

                    gameState = GAME_STATE.PLAYERS_TURN;
                    break;

                // Players turn to pick a location
                case PLAYERS_TURN:

                    int firePosition = getDelimitedValue(
                            displayTextAndGetInput("WHERE DO YOU WISH TO FIRE YOUR MISSILE? "), 0);

                    if (didPlayerHitComputerPlatoon(firePosition)) {
                        // Player hit a player platoon
                        int hits = updatePlayerHits(firePosition);
                        // How many hits has the player made?
                        if (hits != PLATOONS) {
                            showPlayerProgress(hits);
                            gameState = GAME_STATE.COMPUTER_TURN;
                        } else {
                            // Player has obtained 4 hits, they win
                            gameState = GAME_STATE.PLAYER_WON;
                        }
                    } else {
                        // Player missed
                        System.out.println("HA, HA YOU MISSED. MY TURN NOW:");
                        System.out.println();
                        gameState = GAME_STATE.COMPUTER_TURN;
                    }

                    break;

                // Computers time to guess a location
                case COMPUTER_TURN:

                    // Computer takes a guess of a location
                    int computerFirePosition = uniqueComputerGuess();
                    if (didComputerHitPlayerPlatoon(computerFirePosition)) {
                        // Computer hit a player platoon
                        int hits = updateComputerHits(computerFirePosition);
                        // How many hits has the computer made?
                        if (hits != PLATOONS) {
                            showComputerProgress(hits, computerFirePosition);
                            gameState = GAME_STATE.PLAYERS_TURN;
                        } else {
                            // Computer has obtained 4 hits, they win
                            System.out.println("YOU'RE DEAD. YOUR LAST OUTPOST WAS AT " + computerFirePosition
                                    + ". HA, HA, HA.");
                            gameState = GAME_STATE.PLAYER_LOST;
                        }
                    } else {
                        // Computer missed
                        System.out.println("I MISSED YOU, YOU DIRTY RAT. I PICKED " + computerFirePosition
                                + ". YOUR TURN:");
                        System.out.println();
                        gameState = GAME_STATE.PLAYERS_TURN;
                    }

                    break;

                // The player won
                case PLAYER_WON:
                    System.out.println("YOU GOT ME, I'M GOING FAST. BUT I'LL GET YOU WHEN");
                    System.out.println("MY TRANSISTO&S RECUP%RA*E!");
                    gameState = GAME_STATE.GAME_OVER;
                    break;

                case PLAYER_LOST:
                    System.out.println("BETTER LUCK NEXT TIME.");
                    gameState = GAME_STATE.GAME_OVER;
                    break;

                // GAME_OVER State does not specifically have a case
            }
        } while (gameState != GAME_STATE.GAME_OVER);
    }

    /**
     * Calculate computer guess.  Make that the computer does not guess the same
     * location twice
     *
     * @return location of the computers guess that has not been guessed previously
     */
    private int uniqueComputerGuess() {

        boolean validGuess = false;
        int computerGuess;
        do {
            computerGuess = randomNumber();

            if (!computersGuesses.contains(computerGuess)) {
                validGuess = true;
            }
        } while (!validGuess);

        computersGuesses.add(computerGuess);

        return computerGuess;
    }

    /**
     * Create four unique platoons locations for the computer
     * We are using a hashset which guarantees uniqueness so
     * all we need to do is keep trying to add a random number
     * until all four are in the hashset
     *
     * @return 4 locations of computers platoons
     */
    private HashSet<Integer> computersChosenPlatoons() {

        // Initialise contents
        HashSet<Integer> tempPlatoons = new HashSet<>();

        boolean allPlatoonsAdded = false;

        do {
            tempPlatoons.add(randomNumber());

            // All four created?
            if (tempPlatoons.size() == PLATOONS) {
                // Exit when we have created four
                allPlatoonsAdded = true;
            }

        } while (!allPlatoonsAdded);

        return tempPlatoons;
    }

    /**
     * Shows a different message for each number of hits
     *
     * @param hits total number of hits by player on computer
     */
    private void showPlayerProgress(int hits) {

        System.out.println("YOU GOT ONE OF MY OUTPOSTS!");
        showProgress(hits, PLAYER_HIT_MESSAGES);
    }

    /**
     * Shows a different message for each number of hits
     *
     * @param hits total number of hits by computer on player
     */
    private void showComputerProgress(int hits, int lastGuess) {

        System.out.println("I GOT YOU. IT WON'T BE LONG NOW. POST " + lastGuess + " WAS HIT.");
        showProgress(hits, COMPUTER_HIT_MESSAGES);
    }

    /**
     * Prints a message from the passed array based on the value of hits
     *
     * @param hits     - number of hits the player or computer has made
     * @param messages - an array of string with messages
     */
    private void showProgress(int hits, String[] messages) {
        System.out.println(messages[hits - 1]);
    }

    /**
     * Update a player hit - adds a hit the player made on the computers platoon.
     *
     * @param fireLocation - computer location that got hit
     * @return number of hits the player has inflicted on the computer in total
     */
    private int updatePlayerHits(int fireLocation) {

        // N.B. only removes if present, so its redundant to check if it exists first
        computersPlatoons.remove(fireLocation);

        // return number of hits in total
        return PLATOONS - computersPlatoons.size();
    }

    /**
     * Update a computer hit - adds a hit the computer made on the players platoon.
     *
     * @param fireLocation - player location that got hit
     * @return number of hits the player has inflicted on the computer in total
     */
    private int updateComputerHits(int fireLocation) {

        // N.B. only removes if present, so its redundant to check if it exists first
        playersPlatoons.remove(fireLocation);

        // return number of hits in total
        return PLATOONS - playersPlatoons.size();
    }

    /**
     * Determine if the player hit one of the computers platoons
     *
     * @param fireLocation the players choice of location to fire on
     * @return true if a computer platoon was at that position
     */
    private boolean didPlayerHitComputerPlatoon(int fireLocation) {
        return computersPlatoons.contains(fireLocation);
    }

    /**
     * Determine if the computer hit one of the players platoons
     *
     * @param fireLocation the computers choice of location to fire on
     * @return true if a players platoon was at that position
     */
    private boolean didComputerHitPlayerPlatoon(int fireLocation) {
        return playersPlatoons.contains(fireLocation);
    }

    /**
     * Draw the battlefield grid
     */
    private void drawBattlefield() {
        for (int i = 1; i < MAX_GRID_SIZE + 1; i += 5) {
            System.out.printf("%-2s %-2s %-2s %-2s %-2s %n", i, i + 1, i + 2, i + 3, i + 4);
        }
    }

    /**
     * Basic information about the game
     */
    private void intro() {
        System.out.println("BOMBARDMENT");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("YOU ARE ON A BATTLEFIELD WITH 4 PLATOONS AND YOU");
        System.out.println("HAVE 25 OUTPOSTS AVAILABLE WHERE THEY MAY BE PLACED.");
        System.out.println("YOU CAN ONLY PLACE ONE PLATOON AT ANY ONE OUTPOST.");
        System.out.println("THE COMPUTER DOES THE SAME WITH ITS FOUR PLATOONS.");
        System.out.println();
        System.out.println("THE OBJECT OF THE GAME IS TO FIRE MISSILES AT THE");
        System.out.println("OUTPOSTS OF THE COMPUTER.  IT WILL DO THE SAME TO YOU.");
        System.out.println("THE ONE WHO DESTROYS ALL FOUR OF THE ENEMY'S PLATOONS");
        System.out.println("FIRST IS THE WINNER.");
        System.out.println();
        System.out.println("GOOD LUCK... AND TELL US WHERE YOU WANT THE BODIES SENT!");
        System.out.println();
        System.out.println("TEAR OFF MATRIX AND USE IT TO CHECK OFF THE NUMBERS.");
        System.out.println();
        System.out.println();
    }

    private void init() {

        // Create four locations for the computers platoons.
        computersPlatoons = computersChosenPlatoons();

        // Players platoons.
        playersPlatoons = new HashSet<>();

        computersGuesses = new HashSet<>();
    }

    /**
     * Accepts a string delimited by comma's and returns the nth delimited
     * value (starting at count 0).
     *
     * @param text - text with values separated by comma's
     * @param pos  - which position to return a value for
     * @return the int representation of the value
     */
    private int getDelimitedValue(String text, int pos) {
        String[] tokens = text.split(",");
        return Integer.parseInt(tokens[pos]);
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
     * Generate random number
     *
     * @return random number
     */
    private int randomNumber() {
        return (int) (Math.random()
                * (MAX_GRID_SIZE) + 1);
    }
}

```

# `11_Bombardment/java/src/BombardmentGame.java`

这段代码定义了一个名为 "BombardmentGame" 的类，其中包含一个名为 "main" 的方法。

在 "main" 方法中，使用 "new" 关键字创建了一个名为 "bombardment" 的对象，然后调用该对象的 "play" 方法。

"Bombardment" 是一个类，但在这个代码中没有定义它的具体行为。它可能是用来创建一个游戏背景，或者做一些其他事情，但是它没有任何具体的行为被实现。


```
public class BombardmentGame {

    public static void main(String[] args) {

        Bombardment bombardment = new Bombardment();
        bombardment.play();
    }
}

```

# `11_Bombardment/javascript/bombardment.js`

这段代码定义了两个函数，分别是`print()`和`input()`。

`print()`函数的作用是在页面上打印一段字符串，接收一个字符串参数，并将该参数附加给一个元素，该元素是一个具有文本内容的标签（`<div>`）。

`input()`函数的作用是接收用户输入的字符串，返回一个Promise对象。该函数创建了一个带有提示信息的消息框，要求用户输入字符串，并在用户输入后监听键盘事件，当用户按下回车键时，将用户输入的字符串打印到页面上，并返回用户输入的字符串。

这两个函数的功能是一起工作的，用户可以在页面中输入字符串，并点击按钮将其显示在页面上，也可以通过简单的HTML和CSS样式在页面上显示输出。


```
// BOMBARDMENT
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

This appears to be a game of wacky Santa, where the player must hit a button (represented by the number 1-9) to take a turn. The player will have to navigate through different levels, each of which ends with a random event.

The game starts with a brief explanation of the rules, and then the player is prompted to choose a character (1-9) to represent them in the game. The player will then be taken through the different levels of the game, each of which ends with a random event.

The levels seem to be structured differently each time the game is run, but the basic concept of hitting the button to take a turn remains the same. As the player progresses through the levels, they will encounter a variety of different challenges and random events, including the loss of their character, the capture by a rival character (with a brief explanation of why they are capturing the player), and the death of their character (with a brief explanation of why they are now dead).

It is not possible to determine the outcome of each random event without know the number that was selected by the player, but based on the explanation provided, it seems like the game is intended to be a source of entertainment and fun for the players.



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
    print(tab(33) + "BOMBARDMENT\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("YOU ARE ON A BATTLEFIELD WITH 4 PLATOONS AND YOU\n");
    print("HAVE 25 OUTPOSTS AVAILABLE WHERE THEY MAY BE PLACED.\n");
    print("YOU CAN ONLY PLACE ONE PLATOON AT ANY ONE OUTPOST.\n");
    print("THE COMPUTER DOES THE SAME WITH ITS FOUR PLATOONS.\n");
    print("\n");
    print("THE OBJECT OF THE GAME IS TO FIRE MISSILES AT THE\n");
    print("OUTPOSTS OF THE COMPUTER.  IT WILL DO THE SAME TO YOU.\n");
    print("THE ONE WHO DESTROYS ALL FOUR OF THE ENEMY'S PLATOONS\n");
    print("FIRST IS THE WINNER.\n");
    print("\n");
    print("GOOD LUCK... AND TELL US WHERE YOU WANT THE BODIES SENT!\n");
    print("\n");
    // "TEAR OFF" because it supposed this to be printed on a teletype
    print("TEAR OFF MATRIX AND USE IT TO CHECK OFF THE NUMBERS.\n");
    for (r = 1; r <= 5; r++)
        print("\n");
    ma = [];
    for (r = 1; r <= 100; r++)
        ma[r] = 0;
    p = 0;
    q = 0;
    z = 0;
    for (r = 1; r <= 5; r++) {
        i = (r - 1) * 5 + 1;
        print(i + "\t" + (i + 1) + "\t" + (i + 2) + "\t" + (i + 3) + "\t" + (i + 4) + "\n");
    }
    for (r = 1; r <= 10; r++)
        print("\n");
    c = Math.floor(Math.random() * 25) + 1;
    do {
        d = Math.floor(Math.random() * 25) + 1;
        e = Math.floor(Math.random() * 25) + 1;
        f = Math.floor(Math.random() * 25) + 1;
    } while (c == d || c == e || c == f || d == e || d == f || e == f) ;
    print("WHAT ARE YOUR FOUR POSITIONS");
    str = await input();
    g = parseInt(str);
    str = str.substr(str.indexOf(",") + 1);
    h = parseInt(str);
    str = str.substr(str.indexOf(",") + 1);
    k = parseInt(str);
    str = str.substr(str.indexOf(",") + 1);
    l = parseInt(str);
    print("\n");
    // Another "bug" your outpost can be in the same position as a computer outpost
    // Let us suppose both live in a different matrix.
    while (1) {
        // The original game didn't limited the input to 1-25
        do {
            print("WHERE DO YOU WISH TO FIRE YOUR MISSLE");
            y = parseInt(await input());
        } while (y < 0 || y > 25) ;
        if (y == c || y == d || y == e || y == f) {

            // The original game has a bug. You can shoot the same outpost
            // several times. This solves it.
            if (y == c)
                c = 0;
            if (y == d)
                d = 0;
            if (y == e)
                e = 0;
            if (y == f)
                f = 0;
            q++;
            if (q == 1) {
                print("ONE DOWN. THREE TO GO.\n");
            } else if (q == 2) {
                print("TWO DOWN. TWO TO GO.\n");
            } else if (q == 3) {
                print("THREE DOWN. ONE TO GO.\n");
            } else {
                print("YOU GOT ME, I'M GOING FAST. BUT I'LL GET YOU WHEN\n");
                print("MY TRANSISTO&S RECUP%RA*E!\n");
                break;
            }
        } else {
            print("HA, HA YOU MISSED. MY TURN NOW:\n");
        }
        print("\n");
        print("\n");
        do {
            m = Math.floor(Math.random() * 25 + 1);
            p++;
            n = p - 1;
            for (t = 1; t <= n; t++) {
                if (m == ma[t])
                    break;
            }
        } while (t <= n) ;
        x = m;
        ma[p] = m;
        if (x == g || x == h || x == l || x == k) {
            z++;
            if (z < 4)
                print("I GOT YOU. IT WON'T BE LONG NOW. POST " + x + " WAS HIT.\n");
            if (z == 1) {
                print("YOU HAVE ONLY THREE OUTPOSTS LEFT.\n");
            } else if (z == 2) {
                print("YOU HAVE ONLY TWO OUTPOSTS LEFT.\n");
            } else if (z == 3) {
                print("YOU HAVE ONLY ONE OUTPOST LEFT.\n");
            } else {
                print("YOU'RE DEAD. YOUR LAST OUTPOST WAS AT " + x + ". HA, HA, HA.\n");
                print("BETTER LUCK NEXT TIME.\n");
            }
        } else {
            print("I MISSED YOU, YOU DIRTY RAT. I PICKED " + m + ". YOUR TURN:\n");
        }
        print("\n");
        print("\n");
    }
}

```

这道题的代码是 `main()`，这是一个程序的入口函数。在大部分程序中，`main()` 函数是必须的，因为它定义了程序的起点。程序从这里开始执行，它可能会做一些初始化工作，然后处理命令行输入或用户交互。

然而，对于这道题，我没有看到任何需要执行的程序。因此，这个代码本身并没有明确的作用。你需要提供更多信息，以便我能够更好地解释这段代码。


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


# `11_Bombardment/python/bombardment.py`

这段代码是一个用于在控制台界面上打印游戏介绍信息的脚本。它使用了Python 3和Python标准库中的random和functools库。

具体来说，它首先定义了一个名为print_intro的函数，该函数会在屏幕上打印出一个具有33行15列的文本，其中包括游戏中的各种信息和游戏规则。函数使用了print()函数来输出这个文本。

然后，函数创建了一个名为tft外星人的游戏地图，它是一个具有四个据点和25个外据点的棋盘。这个地图是由游戏中的电脑生成的，与玩家控制的四支军队中的两支军队拥有相同的据点。

接着，函数在地图上打印出了它的游戏目标，即攻击敌人的据点并摧毁它们。游戏还向玩家提供了如何使用计算机来攻击敌人据点的提示。最后，函数询问了玩家想要发送多少个身体，然后停止输出并开始打印一个带有数字的矩阵，用于玩家检查他们的军队是否击败了所有的敌人据点。


```
#!/usr/bin/env python3
import random
from functools import partial
from typing import Callable, List, Set


def print_intro() -> None:
    print(" " * 33 + "BOMBARDMENT")
    print(" " * 15 + " CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print("\n\n")
    print("YOU ARE ON A BATTLEFIELD WITH 4 PLATOONS AND YOU")
    print("HAVE 25 OUTPOSTS AVAILABLE WHERE THEY MAY BE PLACED.")
    print("YOU CAN ONLY PLACE ONE PLATOON AT ANY ONE OUTPOST.")
    print("THE COMPUTER DOES THE SAME WITH ITS FOUR PLATOONS.")
    print()
    print("THE OBJECT OF THE GAME IS TO FIRE MISSLES AT THE")
    print("OUTPOSTS OF THE COMPUTER.  IT WILL DO THE SAME TO YOU.")
    print("THE ONE WHO DESTROYS ALL FOUR OF THE ENEMY'S PLATOONS")
    print("FIRST IS THE WINNER.")
    print()
    print("GOOD LUCK... AND TELL US WHERE YOU WANT THE BODIES SENT!")
    print()
    print("TEAR OFF MATRIX AND USE IT TO CHECK OFF THE NUMBERS.")
    print("\n" * 4)


```



这段代码定义了三个函数，各自有不同的作用：

1. `display_field()`函数用于打印一个包含5行数据的表格，每一行都有5列，每列打印一个字符串。

2. `positions_list()`函数返回一个包含1到25的整数组成的列表，即所谓的“位置列表”。

3. `generate_enemy_positions()`函数用于生成一个随机的4个“位置”中的位置列表，即随机从1到25中的数字。这个列表将包含4个不同的位置，而且每个位置只出现一次。


```
def display_field() -> None:
    for row in range(5):
        initial = row * 5 + 1
        print("\t".join([str(initial + column) for column in range(5)]))
    print("\n" * 9)


def positions_list() -> List[int]:
    return list(range(1, 26, 1))


def generate_enemy_positions() -> Set[int]:
    """Randomly choose 4 'positions' out of a range of 1 to 25"""
    positions = positions_list()
    random.shuffle(positions)
    return set(positions[:4])


```

这段代码定义了一个名为 `is_valid_position` 的函数，它接收一个整数参数 `pos`，并返回一个布尔值，表示该位置是否在已知位置列表中。

接下来定义了一个名为 `prompt_for_player_positions` 的函数，它使用一个循环来获取玩家在他的四个位置。在循环中，它通过 `input` 函数获取一个字符串，其中包含四个数字，这些数字被分割为位置列表。接下来，它验证用户输入，确保列表中有四个不同的位置，并且所有位置都处于整数范围内。最后，它返回经过验证的位置列表。


```
def is_valid_position(pos: int) -> bool:
    return pos in positions_list()


def prompt_for_player_positions() -> Set[int]:
    while True:
        raw_positions = input("WHAT ARE YOUR FOUR POSITIONS? ")
        positions = {int(pos) for pos in raw_positions.split()}
        # Verify user inputs (for example, if the player gives a
        # a position for 26, the enemy can never hit it)
        if len(positions) != 4:
            print("PLEASE ENTER 4 UNIQUE POSITIONS\n")
            continue
        elif any(not is_valid_position(pos) for pos in positions):
            print("ALL POSITIONS MUST RANGE (1-25)\n")
            continue
        else:
            return positions


```

这段代码是一个用于向玩家询问目标位置并攻击目标位置的程序。

首先，程序通过一个无限循环来不断向玩家询问目标位置，直到玩家输入一个有效的位置为止。

如果输入的位置不在有效的位置范围内，程序会输出一条错误消息并继续等待玩家的输入。

一旦输入的有效位置被确认，程序会向目标位置发射一束光束，并输出一条攻击成功和一条攻击失败的提示消息。在攻击成功后，程序还会输出一条攻击进度消息，用于显示攻击进度和目标位置的剩余距离。

如果程序在一段时间内没有检测到玩家在控制的有效位置，它就会停止攻击并等待玩家的下一个输入。

攻击函数会检测目标位置是否在位置集合中，如果是，就输出一条攻击消息并从位置集合中移除目标位置。如果不是，就输出一条错误消息。


```
def prompt_player_for_target() -> int:

    while True:
        target = int(input("WHERE DO YOU WISH TO FIRE YOUR MISSLE? "))
        if not is_valid_position(target):
            print("POSITIONS MUST RANGE (1-25)\n")
            continue

        return target


def attack(
    target: int,
    positions: Set[int],
    hit_message: str,
    miss_message: str,
    progress_messages: str,
) -> bool:
    """Performs attack procedure returning True if we are to continue."""

    if target in positions:
        print(hit_message.format(target))
        positions.remove(target)
        print(progress_messages[len(positions)].format(target))
    else:
        print(miss_message.format(target))

    return len(positions) > 0


```

这段代码定义了一个名为 `init_enemy` 的函数，它接受一个匿名函数作为参数，并返回一个实现了 `Callable[[], int]]` 接口的对象。

函数的作用是生成一个可以选择随机位置的函数，避免了选择相同位置两次的问题。具体实现是先从一组可用位置中随机获取一组位置，然后随机洗牌该组位置，接着从洗牌后的位置中选择一个位置，最后返回该位置。

函数返回的 `choose` 函数是一个简单的迭代器，用于生成下一个随机位置。由于使用了 `random.shuffle` 函数对位置列表进行随机洗牌，因此每次生成的位置都是随机的。


```
def init_enemy() -> Callable[[], int]:
    """
    Return a closure analogous to prompt_player_for_target.

    Will choose from a unique sequence of positions to avoid picking the
    same position twice.
    """

    position_sequence = positions_list()
    random.shuffle(position_sequence)
    position = iter(position_sequence)

    def choose() -> int:
        return next(position)

    return choose


```

这段代码定义了两个字符串变量PLAYER_PROGRESS_MESSAGES和ENEMY_PROGRESS_MESSAGES。这两个变量都包含四个字符串元素，它们描述了玩家在不同游戏阶段的进展情况。PLAYER_PROGRESS_MESSAGES包含了玩家在游戏过程中的三个可用点数（3, 2, 1, 0），而ENEMY_PROGRESS_MESSAGES包含了敌人当前游戏阶段的进展情况。


```
# Messages correspond to outposts remaining (3, 2, 1, 0)
PLAYER_PROGRESS_MESSAGES = (
    "YOU GOT ME, I'M GOING FAST. BUT I'LL GET YOU WHEN\nMY TRANSISTO&S RECUP%RA*E!",
    "THREE DOWN, ONE TO GO.\n\n",
    "TWO DOWN, TWO TO GO.\n\n",
    "ONE DOWN, THREE TO GO.\n\n",
)


ENEMY_PROGRESS_MESSAGES = (
    "YOU'RE DEAD. YOUR LAST OUTPOST WAS AT {}. HA, HA, HA.\nBETTER LUCK NEXT TIME.",
    "YOU HAVE ONLY ONE OUTPOST LEFT.\n\n",
    "YOU HAVE ONLY TWO OUTPOSTS LEFT.\n\n",
    "YOU HAVE ONLY THREE OUTPOSTS LEFT.\n\n",
)


```

这段代码是一个Python程序，主要目的是让两个敌人进行对抗。两个函数 `attack` 是攻击函数，需要一个目标位置（在函数内部以 `positions=enemy_positions`形式传递）。攻击函数会根据目标位置尝试进行攻击，并输出相应的消息。

另外，两个函数 `partial(attack, positions=player_positions)` 和 `partial(attack, positions=enemy_positions)` 是辅助攻击函数，只需传入一个位置参数即可。这些函数可以看做是攻击函数的实例，可以保证在函数调用时只有传入同一个参数。

函数 `print_intro()` 和 `display_field()` 似乎没有实际作用，可能是在程序运行时打印一些提示信息。

另外， `generate_enemy_positions()` 和 `prompt_for_player_positions()` 似乎也没有在代码中定义，也没有在函数中使用。因此，我无法解释它们具体做什么。


```
def main() -> None:
    print_intro()
    display_field()

    enemy_positions = generate_enemy_positions()
    player_positions = prompt_for_player_positions()

    # Build partial functions only requiring the target as input
    player_attacks = partial(
        attack,
        positions=enemy_positions,
        hit_message="YOU GOT ONE OF MY OUTPOSTS!",
        miss_message="HA, HA YOU MISSED. MY TURN NOW:\n\n",
        progress_messages=PLAYER_PROGRESS_MESSAGES,
    )

    enemy_attacks = partial(
        attack,
        positions=player_positions,
        hit_message="I GOT YOU. IT WON'T BE LONG NOW. POST {} WAS HIT.",
        miss_message="I MISSED YOU, YOU DIRTY RAT. I PICKED {}. YOUR TURN:\n\n",
        progress_messages=ENEMY_PROGRESS_MESSAGES,
    )

    enemy_position_choice = init_enemy()

    # Play as long as both player_attacks and enemy_attacks allow to continue
    while player_attacks(prompt_player_for_target()) and enemy_attacks(
        enemy_position_choice()
    ):
        pass


```

这段代码是一个Python程序中的一个if语句，它的作用是判断当前程序是否作为主程序运行。如果当前程序作为主程序运行，那么程序会执行if语句中的内容，也就是main()函数。

if语句是一种条件语句，它的语法为if __name__ == "__main__":，其中"__main__"是一个特殊的字符串，表示一个可执行文件的路径。当程序作为主程序运行时，它会检查当前目录是否包含名为"__main__"的可执行文件，如果存在，则执行该文件中的内容。

因此，这段代码的作用是用于确保程序在作为主程序运行时能够正常执行。如果没有其他程序在运行，或者当前目录中没有名为"__main__"的可执行文件，那么程序将无法执行if语句中的内容，也就是无法启动程序。


```
if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by [ugurkupeli](https://github.com/ugurkupeli)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Bombs Away

In this program, you fly a World War II bomber for one of the four protagonists of the war. You then pick your target or the type of plane you are flying. Depending on your flying experience and the quality of enemy defenders, you then may accomplish your mission, get shot down, or make it back through enemy fire. In any case, you get a chance to fly again.

David Ahl modified the original program which was created by David Sherman while a student at Curtis Jr. High School, Sudbury, Massachusetts.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=24)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=39)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Known Bugs

- If you play as Japan and say it is not your first mission, it is impossible to complete your mission; the only possible outcomes are "you made it through" or "boom".  Moreover, the odds of each outcome depend on a variable (R) that is only set if you played a previous mission as a different side.  It's possible this is an intentional layer of complexity meant to encourage repeat play, but it's more likely just a logical error.

#### Porting Notes

(please note any difficulties or challenges in porting here)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `12_Bombs_Away/csharp/BombsAwayConsole/ConsoleUserInterface.cs`

This is a class that reads key information from the console in Python. It has several methods including `ReadKey()` which reads a key from the console and returns it. Another method `ChooseYesOrNo()` allows the user to choose between 'Y' or 'N' from the console. The last method `InputInteger()` reads an integer from the console. All of these methods take in a message as a parameter.


```
﻿namespace BombsAwayConsole;

/// <summary>
/// Implements <see cref="BombsAwayGame.IUserInterface"/> by writing to and reading from <see cref="Console"/>.
/// </summary>
internal class ConsoleUserInterface : BombsAwayGame.IUserInterface
{
    /// <summary>
    /// Write message to console.
    /// </summary>
    /// <param name="message">Message to display.</param>
    public void Output(string message)
    {
        Console.WriteLine(message);
    }

    /// <summary>
    /// Write choices with affixed indexes, allowing the user to choose by index.
    /// </summary>
    /// <param name="message">Message to display.</param>
    /// <param name="choices">Choices to display.</param>
    /// <returns>Choice that user picked.</returns>
    public int Choose(string message, IList<string> choices)
    {
        IEnumerable<string> choicesWithIndexes = choices.Select((choice, index) => $"{choice}({index + 1})");
        string choiceText = string.Join(", ", choicesWithIndexes);
        Output($"{message} -- {choiceText}");

        ISet<ConsoleKey> allowedKeys = ConsoleKeysFromList(choices);
        ConsoleKey? choice;
        do
        {
            choice = ReadChoice(allowedKeys);
            if (choice is null)
            {
                Output("TRY AGAIN...");
            }
        }
        while (choice is null);

        return ListIndexFromConsoleKey(choice.Value);
    }

    /// <summary>
    /// Convert the given list to its <see cref="ConsoleKey"/> equivalents. This generates keys that map
    /// the first element to <see cref="ConsoleKey.D1"/>, the second element to <see cref="ConsoleKey.D2"/>,
    /// and so on, up to the last element of the list.
    /// </summary>
    /// <param name="list">List whose elements will be converted to <see cref="ConsoleKey"/> equivalents.</param>
    /// <returns><see cref="ConsoleKey"/> equivalents from <paramref name="list"/>.</returns>
    private ISet<ConsoleKey> ConsoleKeysFromList(IList<string> list)
    {
        IEnumerable<int> indexes = Enumerable.Range((int)ConsoleKey.D1, list.Count);
        return new HashSet<ConsoleKey>(indexes.Cast<ConsoleKey>());
    }

    /// <summary>
    /// Convert the given console key to its list index equivalent. This assumes the key was generated from
    /// <see cref="ConsoleKeysFromList(IList{string})"/>
    /// </summary>
    /// <param name="key">Key to convert to its list index equivalent.</param>
    /// <returns>List index equivalent of key.</returns>
    private int ListIndexFromConsoleKey(ConsoleKey key)
    {
        return key - ConsoleKey.D1;
    }

    /// <summary>
    /// Read a key from the console and return it if it is in the given allowed keys.
    /// </summary>
    /// <param name="allowedKeys">Allowed keys.</param>
    /// <returns>Key read from <see cref="Console"/>, if it is in <paramref name="allowedKeys"/>; null otherwise./></returns>
    private ConsoleKey? ReadChoice(ISet<ConsoleKey> allowedKeys)
    {
        ConsoleKeyInfo keyInfo = ReadKey();
        return allowedKeys.Contains(keyInfo.Key) ? keyInfo.Key : null;
    }

    /// <summary>
    /// Read key from <see cref="Console"/>.
    /// </summary>
    /// <returns>Key read from <see cref="Console"/>.</returns>
    private ConsoleKeyInfo ReadKey()
    {
        ConsoleKeyInfo result = Console.ReadKey(intercept: false);
        // Write a blank line to the console so the displayed key is on its own line.
        Console.WriteLine();
        return result;
    }

    /// <summary>
    /// Allow user to choose 'Y' or 'N' from <see cref="Console"/>.
    /// </summary>
    /// <param name="message">Message to display.</param>
    /// <returns>True if user chose 'Y', false if user chose 'N'.</returns>
    public bool ChooseYesOrNo(string message)
    {
        Output(message);
        ConsoleKey? choice;
        do
        {
            choice = ReadChoice(new HashSet<ConsoleKey>(new[] { ConsoleKey.Y, ConsoleKey.N }));
            if (choice is null)
            {
                Output("ENTER Y OR N");
            }
        }
        while (choice is null);

        return choice.Value == ConsoleKey.Y;
    }

    /// <summary>
    /// Get integer by reading a line from <see cref="Console"/>.
    /// </summary>
    /// <returns>Integer read from <see cref="Console"/>.</returns>
    public int InputInteger()
    {
        bool resultIsValid;
        int result;
        do
        {
            string? integerText = Console.ReadLine();
            resultIsValid = int.TryParse(integerText, out result);
            if (!resultIsValid)
            {
                Output("PLEASE ENTER A NUMBER");
            }
        }
        while (!resultIsValid);

        return result;
    }
}

```

# `12_Bombs_Away/csharp/BombsAwayConsole/Program.cs`

这段代码的作用是在用户愿意再次玩游戏的情况下，重复地创建并玩同一个游戏。

具体来说，代码首先导入了BombsAwayConsole和BombsAwayGame，然后定义了一个名为PlayGameWhileUserWantsTo的函数，该函数接收一个ConsoleUserInterface类型的参数ui。

在函数内部，代码使用do-while循环，在每次循环中，创建并开始玩一个名为Game的游戏对象，然后使用该游戏对象的方法Play()来开始玩游戏。在游戏的每次迭代中，代码使用一个名为UserWantsToPlayAgain的函数来获取用户是否愿意再次玩游戏，如果用户不想再玩游戏，则循环将停止。如果用户愿意再次玩游戏，则游戏将继续进行，否则循环将停止。

因此，这段代码的作用是在用户愿意再次玩游戏的情况下，重复地创建并玩同一个游戏。


```
﻿using BombsAwayConsole;
using BombsAwayGame;

/// Create and play <see cref="Game"/>s using a <see cref="ConsoleUserInterface"/>.
PlayGameWhileUserWantsTo(new ConsoleUserInterface());

void PlayGameWhileUserWantsTo(ConsoleUserInterface ui)
{
    do
    {
        new Game(ui).Play();
    }
    while (UserWantsToPlayAgain(ui));
}

```

这段代码是一个条件判断语句，名为UserWantsToPlayAgain，它接受一个UI用户界面对象ui作为参数。

代码首先定义了一个bool类型的变量result，并将其初始化为假（false）。然后，代码使用ui.ChooseYesOrNo方法询问用户是否想要再次玩游戏，并将结果存储在result变量中。

如果result为真（true），那么代码将使用Console.WriteLine方法输出"CHICKEN !!!"，这意味着用户不会再次玩游戏，因为他们在之前的游戏中显示了一个警告。

如果result为假（false），则不会执行代码中的任何操作。


```
bool UserWantsToPlayAgain(IUserInterface ui)
{
    bool result = ui.ChooseYesOrNo("ANOTHER MISSION (Y OR N)?");
    if (!result)
    {
        Console.WriteLine("CHICKEN !!!");
    }

    return result;
}

```

# `12_Bombs_Away/csharp/BombsAwayGame/AlliesSide.cs`

这段代码定义了一个名为 `AlliesSide` 的类，继承自 `MissionSide` 类。这个类表示游戏中的一个盟友角色，可以搭乘解放者、B-29、B-17 或者兰克纳星机的任务。

该类的构造函数接收一个 `IUserInterface` 类型的参数，用于在游戏UI中显示任务信息。

该类重写了 `ChooseMissionMessage` 方法，用于在选择任务时显示消息。这个消息提示玩家选择哪种战斗机进行任务。

该类还重写了 `AllMissions` 方法，用于返回可供玩家选择的任务列表。这些任务包括在游戏中起飞不同类型的炸弹来完成任务。


```
﻿namespace BombsAwayGame;

/// <summary>
/// Allies protagonist. Can fly missions in a Liberator, B-29, B-17, or Lancaster.
/// </summary>
internal class AlliesSide : MissionSide
{
    public AlliesSide(IUserInterface ui)
        : base(ui)
    {
    }

    protected override string ChooseMissionMessage => "AIRCRAFT";

    protected override IList<Mission> AllMissions => new Mission[]
    {
        new("LIBERATOR", "YOU'VE GOT 2 TONS OF BOMBS FLYING FOR PLOESTI."),
        new("B-29", "YOU'RE DUMPING THE A-BOMB ON HIROSHIMA."),
        new("B-17", "YOU'RE CHASING THE BISMARK IN THE NORTH SEA."),
        new("LANCASTER", "YOU'RE BUSTING A GERMAN HEAVY WATER PLANT IN THE RUHR.")
    };
}

```

# `12_Bombs_Away/csharp/BombsAwayGame/EnemyArtillery.cs`

这段代码定义了一个名为"EnemyArtillery"的类，用于表示游戏中的敌人火炮。

该类包含两个成员变量，一个字符串类型的"Name"变量和一个整数类型的"Accuracy"变量。其中，"Name"变量用于存储火炮类型，最大长度没有限制，而"Accuracy"变量则是"T"变量在原始BASIC游戏中的值，用于表示该火炮的精度。

该类还包含一个内部结构体类型的"EnemyArtillery"类，用于存储敌方火炮的属性，但是不包括名称。

最后，该类定义了一个名为"BombsAwayGame"的外部命名空间，用于包含该类的所有成员和子类。


```
﻿namespace BombsAwayGame;

/// <summary>
/// Represents enemy artillery.
/// </summary>
/// <param name="Name">Name of artillery type.</param>
/// <param name="Accuracy">Accuracy of artillery. This is the `T` variable in the original BASIC.</param>
internal record class EnemyArtillery(string Name, int Accuracy);

```

# `12_Bombs_Away/csharp/BombsAwayGame/Game.cs`

This code defines a `Game` class that simulates a World War II fighter plane game. The `Game` class has a `Play` method that allows the player to choose a side to play and a `ChooseSide` method that prompts the player to choose a side. The `AllSideDescriptors` method returns an array of all side descriptors (e.g. names of sides) that can be used for the `ChooseSide` method.


```
﻿namespace BombsAwayGame;

/// <summary>
/// Plays the Bombs Away game using a supplied <see cref="IUserInterface"/>.
/// </summary>
public class Game
{
    private readonly IUserInterface _ui;

    /// <summary>
    /// Create game instance using the given UI.
    /// </summary>
    /// <param name="ui">UI to use for game.</param>
    public Game(IUserInterface ui)
    {
        _ui = ui;
    }

    /// <summary>
    /// Play game. Choose a side and play the side's logic.
    /// </summary>
    public void Play()
    {
        _ui.Output("YOU ARE A PILOT IN A WORLD WAR II BOMBER.");
        Side side = ChooseSide();
        side.Play();
    }

    /// <summary>
    /// Represents a <see cref="Side"/>.
    /// </summary>
    /// <param name="Name">Name of side.</param>
    /// <param name="CreateSide">Create instance of side that this descriptor represents.</param>
    private record class SideDescriptor(string Name, Func<Side> CreateSide);

    /// <summary>
    /// Choose side and return a new instance of that side.
    /// </summary>
    /// <returns>New instance of side that was chosen.</returns>
    private Side ChooseSide()
    {
        SideDescriptor[] sides = AllSideDescriptors;
        string[] sideNames = sides.Select(a => a.Name).ToArray();
        int index = _ui.Choose("WHAT SIDE", sideNames);
        return sides[index].CreateSide();
    }

    /// <summary>
    /// All side descriptors.
    /// </summary>
    private SideDescriptor[] AllSideDescriptors => new SideDescriptor[]
    {
        new("ITALY", () => new ItalySide(_ui)),
        new("ALLIES", () => new AlliesSide(_ui)),
        new("JAPAN", () => new JapanSide(_ui)),
        new("GERMANY", () => new GermanySide(_ui)),
    };
}

```