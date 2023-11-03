# BasicComputerGames源码解析 6

### Life

The Game of Life was originally described in _Scientific American_, October 1970, in an article by Martin Gardner. The game itself was originated by John Conway of Gonville and Caius College, University of Cambridge England.

In the “manual” game, organisms exist in the form of counters (chips or checkers) on a large checkerboard and die or reproduce according to some simple genetic rules. Conway’s criteria for choosing his genetic laws were carefully delineated as follows:
1. There should be no initial pattern for which there is a simple proof that the population can grow without limit.
2. There should be simple initial patterns that apparently do grow without limit.
3. There should be simple initial patterns that grow and change for a considerable period of time before coming to an end in three possible ways:
    1. Fading away completely (from overcrowding or from becoming too sparse)
    2. Settling into a stable configuration that remains unchanged thereafter
    3. Entering an oscillating phase in which they repeat an endless cycle of two or more periods

In brief, the rules should be such as to make the behavior of the population relatively unpredictable. Conway’s genetic laws are delightfully simple. First note that each cell of the checkerboard (assumed to be an infinite plane) has eight neighboring cells, four adjacent orthogonally, four adjacent diagonally. The rules are:
1. Survivals. Every counter with two or three neighboring counters survives for the next generation.
2. Deaths. Each counter with four or more neighbors dies (is removed) from overpopulation. Every counter with one neighbor or none dies from isolation.
3. Births. Each empty cell adjacent to exactly three neighbors — no more — is a birth cell. A counter is placed on it at the next move.

It is important to understand that all births and deaths occur simultaneously. Together they constitute a single generation or, as we shall call it, a “move” in the complete “life history” of the initial configuration.

You will find the population constantly undergoing unusual, sometimes beautiful and always unexpected change. In a few cases the society eventually dies out (all counters vanishing), although this may not happen until after a great many generations. Most starting patterns either reach stable figures — Conway calls them “still lifes” — that cannot change or patterns that oscillate forever. Patterns with no initial symmetry tend to become symmetrical. Once this happens the symmetry cannot be lost, although it may increase in richness.

Conway used a DEC PDP-7 with a graphic display to observe long-lived populations. You’ll probably find this more enjoyable to watch on a CRT than a hard-copy terminal.

Since MITS 8K BASIC does not have LINE INPUT, to enter leading blanks in the patter, type a “.” at the start of the line. This will be converted to a space by BASIC, but it permits you to type leading spaces. Typing DONE indicates that you are finished entering the pattern. See sample run.

Clark Baker of Project DELTA originally wrote this version of LIFE which was further modified by Steve North of Creative Computing.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=100)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=115)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html


#### Porting Notes

- To make sense of the code, it's important to understand what the values in the A(X,Y) array mean:
  - 0: dead cell
  - 1: live cell
  - 2: currently live, but dead next cycle
  - 3: currently dead, but alive next cycle


(please note any difficulties or challenges in porting here)


# Life

An implementation of John Conway's popular cellular automaton, also know as **Conway's Game of Life**. The original source was downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html).

Ported by Dyego Alekssander Maas.

## How to run

This program requires you to install [.NET 6 SDK](https://dotnet.microsoft.com/en-us/download/dotnet/6.0). After installed, you just need to run `dotnet run` from this directory in the terminal.

## Know more about Conway's Game of Life

You can find more about Conway's Game of Life on this page of the [Cornell Math Explorers' Club](http://pi.math.cornell.edu/~lipa/mec/lesson6.html), alongside many examples of patterns you can try.

### Optional parameters

Optionally, you can run this program with the `--wait 1000` argument, the number being the time in milliseconds
that the application will pause between each iteration. This is enables you to watch the simulation unfolding. By default, there is no pause between iterations.

The complete command would be `dotnet run --wait 1000`.

## Entering patterns

Once running the game, you are expected to enter a pattern. This pattern consists of multiple lines of text with either **spaces** or **some character**, usually an asterisk (`*`).

Spaces represent empty cells. Asterisks represent alive cells.

After entering the pattern, you need to enter the word "DONE". It is not case sensitive. An example of pattern would be:

```
 *
***
DONE
```

### Some patterns you could try

```
 *
***
```

```
*
***
```

```
**
**
```

```
  *
 *
*
```

This one is known as **glider**:

```
***
*
 *
```

## Instructions to the port

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# Game of Life - Java version

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)

## Requirements

* Requires Java 17 (or later)

## Notes

The Java version of Game of Life tries to mimics the behaviour of the BASIC version.
However, the Java code does not have much in common with the original.

**Differences in behaviour:**
* Input supports the ```.``` character, but it's optional.
* Evaluation of ```DONE``` input string is case insensitive.
* Run with the ```-s``` command line argument to halt the program after each generation, and continue when ```ENTER``` is pressed.


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


# Conway's Life

Original from David Ahl's _Basic Computer Games_, downloaded from http://www.vintage-basic.net/games.html.

Ported to Rust by Jon Fetter-Degges

Developed and tested on Rust 1.64.0

## How to Run

Install Rust using the instructions at [rust-lang.org](https://www.rust-lang.org/tools/install).

At a command or shell prompt in the `rust` subdirectory, enter `cargo run`.

## Differences from Original Behavior

* The simulation stops if all cells die.
* `.` at the beginning of an input line is supported but optional.
* Input of more than 66 columns is rejected. Input will automatically terminate after 20 rows. Beyond these bounds, the original
implementation would have marked the board as invalid, and beyond 68 cols/24 rows it would have had an out of bounds array access.
* The check for the string "DONE" at the end of input is case-independent.
* The program pauses for half a second between each generation.


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Life for Two

LIFE-2 is based on Conway’s game of Life. You must be familiar with the rules of LIFE before attempting to play LIFE-2.

There are two players; the game is played on a 5x5 board and each player has a symbol to represent his own pieces of ‘life.’ Live cells belonging to player 1 are represented by `*` and live cells belonging to player 2 are represented by the symbol `#`.

The # and * are regarded as the same except when deciding whether to generate a live cell. An empty cell having two `#` and one `*` for neighbors will generate a `#`, i.e. the live cell generated belongs to the player who has the majority of the 3 live cells surrounding the empty cell where life is to be generated, for example:

```
|   | 1 | 2 | 3 | 4 | 5 |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 |   |   |   |   |   |
| 2 |   |   | * |   |   |
| 3 |   |   |   | # |   |
| 4 |   |   | # |   |   |
| 5 |   |   |   |   |   |
```

A new cell will be generated at (3,3) which will be a `#` since there are two `#` and one `*` surrounding. The board will then become:
```
|   | 1 | 2 | 3 | 4 | 5 |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 |   |   |   |   |   |
| 2 |   |   |   |   |   |
| 3 |   |   | # | # |   |
| 4 |   |   |   |   |   |
| 5 |   |   |   |   |   |
```
On the first move each player positions 3 pieces of life on the board by typing in the co-ordinates of the pieces. (In the event of the same cell being chosen by both players that cell is left empty.)

The board is then adjusted to the next generation and printed out.

On each subsequent turn each player places one piece on the board, the object being to annihilate his opponent’s pieces. The board is adjusted for the next generation and printed out after both players have entered their new piece.

The game continues until one player has no more live pieces. The computer will then print out the board and declare the winner.

The idea for this game, the game itself, and the above write-up were written by Brian Wyvill of Bradford University in Yorkshire, England.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=102)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=117)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html


#### Porting Notes

(please note any difficulties or challenges in porting here)

Note: The original program has a bug. The instructions say that if both players
enter the same cell that the cell is set to 0 or empty. However, the original
Basic program tells the player "ILLEGAL COORDINATES" and makes another cell be entered,
giving a slightly unfair advantage to the 2nd player.

The Perl verson of the program fixes the bug and follows the instructions.

Note: The original code had "GOTO 800" but label 800 didn't exist; it should have gone to label 999.
The Basic program has been fixed.

Note: The Basic program is written to assume it's being played on a Teletype, i.e. output is printed
on paper. To play on a terminal the input must not be echoed, which can be a challenge to do portably
and without tying the solution to a specific OS. Some versions may tell you how to do this, others might not.


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)

Note: The original program has a bug (see the README in the above dir). This Perl version fixes it.

Note: For input, the X value is to the right while the Y value is down.
Therefore, the top right cell is "5,1", not "1,5".

The original program was made to be played on a Teletype, i.e. a printer on paper.
That allowed the program to "black out" the input line to hide a user's input from his/her
opponent, assuming the opponent was at least looking away. To do the equivalent on a
terminal would require a Perl module that isn't installed by default (i.e. it is not
part of CORE and would also require a C compiler to install), nor do I want to issue a
shell command to "stty" to hide the input because that would restrict the game to Linux/Unix.
This means it would have to be played on the honor system.

However, if you want to try it, install the module "Term::ReadKey" ("sudo cpan -i Term::ReadKey"
if on Linux/Unix and you have root access). If the code finds that module, it will automatically
use it and hide the input ... and restore echoing input again when the games ends. If the module
is not found, input will be visible.


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Literature Quiz

This is a simple CAI-type program which presents four multiple-choice questions from children’s literature. Running the program is self-explanatory.

The program was written by Pamela McGinley while at DEC.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=104)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=117)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Love

This program is designed to reproduce Robert Indiana’s great art work “Love” with a message of your choice up to 60 characters long.

The love program was created by David Ahl.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=105)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=120)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by Jadi.


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Lunar LEM Rocket

This game in its many different versions and names (ROCKET, LUNAR, LEM, and APOLLO) is by far and away the single most popular computer game. It exists in various versions that start you anywhere from 500 feet to 200 miles away from the moon, or other planets, too. Some allow the control of directional stabilization rockets and/or the retro rocket. The three versions presented here represent the most popular of the many variations.

In most versions of this game, the temptation is to slow up too soon and then have no fuel left for the lower part of the journey. This, of course, is disastrous (as you will find out when you land your own capsule)!

LUNAR was originally in FOCAL by Jim Storer while a student at Lexington High School and subsequently converted to BASIC by David Ahl. ROCKET was written by Eric Peters at DEC and LEM by William Labaree II of Alexandria, Virginia.

In this program, you set the burn rate of the retro rockets (pounds of fuel per second) every 10 seconds and attempt to achieve a soft landing on the moon. 200 lbs/sec really puts the brakes on, and 0 lbs/sec is free fall. Ignition occurs a 8 lbs/sec, so _do not_ use burn rates between 1 and 7 lbs/sec. To make the landing more of a challenge, but more closely approximate the real Apollo LEM capsule, you should make the available fuel at the start (N) equal to 16,000 lbs, and the weight of the capsule (M) equal to 32,500 lbs.

#### LEM
This is the most comprehensive of the three versions and permits you to control the time interval of firing, the thrust, and the attitude angle. It also allows you to work in the metric or English system of measurement. The instructions in the program dialog are very complete, so you shouldn’t have any trouble.

#### ROCKET
In this version, you start 500 feet above the lunar surface and control the burn rate in 1-second bursts. Each unit of fuel slows your descent by 1 ft/sec. The maximum thrust of your engine is 30 ft/sec/sec.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=106)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=121)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Known Bugs

### lem.bas

- The input validation on the thrust value (displayed as P, stored internally as F) appears to be incorrect.  It allows negative values up up to -95, but at -96 or more balks and calls it negative.  I suspect the intent was to disallow any value less than 0 (in keeping with the instructions), *or* nonzero values less than 10.

- The physics calculations seem very sus.  If you enter "1000,0,0" (i.e. no thrust at all, integrating 1000 seconds at a time) four times in a row, you first fall, but then mysteriously gain vertical speed, and end up being lost in space.  This makes no sense.  A similar result happened when just periodically applying 10% thrust in an attempt to hover.


#### Porting Notes

(please note any difficulties or challenges in porting here)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


README.md

Original source downloaded from Vintage Basic

This folder for chapter #59 contains three different games.  Three folders here contain the three games:

 - Rocket
 - LEM
 - lunar

Conversion to [Rust](https://www.rust-lang.org)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### MasterMind

In that Match-April 1976 issue of _Creative_ we published a computerized version of Master Mind, a logic game. Master Mind is played by two people—one is called the code-maker; the other, the code-breaker. At the beginning of the game the code-maker forms a code, or combination of colored pegs. He hides these from the code-breaker. The code-breaker then attempts to deduce the code, by placing his own guesses, one at a time, on the board. After he makes a guess (by placing a combination of colored pegs on the board) the code-maker then gives the code-breaker clues to indicate how close the guess was to the code. For every peg in the guess that’s the right color but not in the right position, the code-breaker gets a white peg. Note that these black and white pegs do not indicate _which_ pegs in the guess are correct, but merely that they exist. For example, if the code was:
```
Yellow Red Red Green
```

and my guess was
```
Red Red Yellow Black
```
I would receive two white pegs and one black peg for the guess. I wouldn’t know (except by comparing previous guesses) which one of the pegs in my guess was the right color in the right position.

Many people have written computer programs to play Master Mind in the passive role, i.e., the computer is the code maker and the human is the code-breaker. This is relatively trivial; the challenge is writing a program that can also play actively as a code-breaker.

Actually, the task of getting the computer to deduce the correct combination is not at all difficult. Imagine, for instance, that you made a list of all possible codes. To begin, you select a guess from your list at random. Then, as you receive clues, you cross off from the list those combinations which you know are impossible. For example if your guess is Red Red Green Green and you receive no pegs, then you know that any combination containing either a red or a green peg is impossible and may be crossed of the list. The process is continued until the correct solution is reached or there are no more combinations left on the list (in which case you know that the code-maker made a mistake in giving you the clues somewhere).

Note that in this particular implementation, we never actually create a list of the combinations, but merely keep track of which ones (in sequential order) may be correct. Using this system, we can easily say that the 523rd combination may be correct, but to actually produce the 523rd combination we have to count all the way from the first combination (or the previous one, if it was lower than 523). Actually, this problem could be simplified to a conversion from base 10 to base (number of colors) and then adjusting the values used in the MID$ function so as not to take a zeroth character from a string if you want to experiment. We did try a version that kept an actual list of all possible combinations (as a string array), which was significantly faster than this version, but which ate tremendous amounts of memory.

At the beginning of this game, you input the number of colors and number of positions you wish to use (which will directly affect the number of combinations) and the number of rounds you wish to play. While you are playing as the code-breaker, you may type BOARD at any time to get a list of your previous guesses and clues, and QUIT to end the game. Note that this version uses string arrays, but this is merely for convenience and can easily be converted for a BASIC that has no string arrays as long as it has a MID$ function. This is because the string arrays are one-dimensional, never exceed a length greater than the number of positions and the elements never contain more than one character.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=110)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=125)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

### How the computer deduces your guess.

The computer takes the number of black pegs and white pegs that the user reports
and uses that information as a target. It then assumes its guess is the answer
and proceeds to compare the black and white pegs against all remaining possible
answers. For each set of black and white pegs it gets in these comparisons, if 
they don't match what the user reported, then they can not be part of the solution.
This can be a non-intuitive assumption, so we'll walk through it with a three color,
three position example (27 possible solutions.)

Let's just suppose our secret code we're hiding from the computer is `BWB`

First let's point out the commutative property of comparing two codes for their
black and white pegs. A black peg meaning correct color and correct position, and
a white peg meaning correct color and wrong position.  If the computer guesses
`RBW` then the black/white peg report is 0 black, 2 white.  But if `RBW` is the 
secret code and the computer guesses `BWB` the reporting for `BWB` is going to be
the same, 0 black, 2 white. 

Now lets look at a table with the reporting for every possible guess the computer 
can make while our secret code is `BWB`.
                                                         
| Guess | Black | White |     | Guess | Black | White |     | Guess | Black | White |
|-------|-------|-------|-----|-------|-------|-------|-----|-------|-------|-------|
| BBB   | 2     | 0     |     | WBB   | 1     | 2     |     | RBB   | 1     | 1     |   
| BBW   | 1     | 2     |     | WBW   | 0     | 2     |     | RBW   | 0     | 2     |   
| BBR   | 1     | 1     |     | WBR   | 0     | 2     |     | RBR   | 0     | 1     |    
| BWB   | 3     | 0     |     | WWB   | 2     | 0     |     | RWB   | 2     | 0     |    
| BWW   | 2     | 0     |     | WWW   | 1     | 0     |     | RWW   | 1     | 0     |    
| BWR   | 2     | 0     |     | WWR   | 1     | 0     |     | RWR   | 1     | 0     |    
| BRB   | 2     | 0     |     | WRB   | 1     | 1     |     | RRB   | 1     | 0     |    
| BRW   | 1     | 1     |     | WRW   | 0     | 1     |     | RRW   | 0     | 1     |    
| BRR   | 1     | 0     |     | WRR   | 0     | 1     |     | RRR   | 0     | 0     | 

The computer has guessed `RBW` and the report on it is 0 black, 2 white. The code
used to eliminate other solutions looks like this:

`1060 IF B1<>B OR W1<>W THEN I(X)=0`

which says set `RBW` as the secret and compare it to all remaining solutions and 
get rid of any that don't match the same black and white report, 0 black and 2 white. 
So let's do that.

Remember, `RBW` is pretending to be the secret code here. These are the remaining
solutions reporting their black and white pegs against `RBW`.

| Guess | Black | White |     | Guess | Black | White |     | Guess | Black | White |
|-------|-------|-------|-----|-------|-------|-------|-----|-------|-------|-------|
| BBB   | 1     | 0     |     | WBB   | 1     | 1     |     | RBB   | 2     | 0     |   
| BBW   | 2     | 0     |     | WBW   | 2     | 0     |     | RBW   | 3     | 0     |   
| BBR   | 1     | 1     |     | WBR   | 1     | 2     |     | RBR   | 2     | 0     |    
| BWB   | 0     | 2     |     | WWB   | 0     | 2     |     | RWB   | 1     | 2     |    
| BWW   | 1     | 1     |     | WWW   | 1     | 0     |     | RWW   | 2     | 0     |    
| BWR   | 0     | 3     |     | WWR   | 1     | 1     |     | RWR   | 1     | 1     |    
| BRB   | 0     | 2     |     | WRB   | 0     | 3     |     | RRB   | 1     | 1     |    
| BRW   | 1     | 2     |     | WRW   | 1     | 1     |     | RRW   | 2     | 0     |    
| BRR   | 0     | 2     |     | WRR   | 0     | 2     |     | RRR   | 1     | 0     | 

Now we are going to eliminate every solution that **DOESN'T** match 0 black and 2 white.

| Guess    | Black | White |     | Guess    | Black | White |     | Guess    | Black | White |
|----------|-------|-------|-----|----------|-------|-------|-----|----------|-------|-------|
| ~~~BBB~~ | 1     | 0     |     | ~~~WBB~~ | 1     | 1     |     | ~~~RBB~~ | 2     | 0     |   
| ~~~BBW~~ | 2     | 0     |     | ~~~WBW~~ | 2     | 0     |     | ~~~RBW~~ | 3     | 0     |   
| ~~~BBR~~ | 1     | 1     |     | ~~~WBR~~ | 1     | 2     |     | ~~~RBR~~ | 2     | 0     |    
| BWB      | 0     | 2     |     | WWB      | 0     | 2     |     | ~~~RWB~~ | 1     | 2     |    
| ~~~BWW~~ | 1     | 1     |     | ~~~WWW~~ | 1     | 0     |     | ~~~RWW~~ | 2     | 0     |    
| ~~~BWR~~ | 0     | 3     |     | ~~~WWR~~ | 1     | 1     |     | ~~~RWR~~ | 1     | 1     |    
| BRB      | 0     | 2     |     | ~~~WRB~~ | 0     | 3     |     | ~~~RRB~~ | 1     | 1     |    
| ~~~BRW~~ | 1     | 2     |     | ~~~WRW~~ | 1     | 1     |     | ~~~RRW~~ | 2     | 0     |    
| BRR      | 0     | 2     |     | WRR      | 0     | 2     |     | ~~~RRR~~ | 1     | 0     |          
                                   
 That wipes out all but five solutions. Notice how the entire right column of solutions 
 is eliminated, including our original guess of `RBW`, therefore eliminating any 
 special case to specifically eliminate this guess from the solution set when we first find out
 its not the answer.
 
 Continuing on, we have the following solutions left of which our secret code, `BWB` 
 is one of them. Remember our commutative property explained previously. 

| Guess | Black | White |
|-------|-------|-------|
| BWB   | 0     | 2     |
| BRB   | 0     | 2     |
| BRR   | 0     | 2     |
| WWB   | 0     | 2     |
| WRR   | 0     | 2     |

So for its second pick, the computer will randomly pick one of these remaining solutions. Let's pick
the middle one, `BRR`, and perform the same ritual. Our user reports to the computer 
that it now has 1 black, 0 whites when comparing to our secret code `BWB`. Let's 
now compare `BRR` to the remaining five solutions and eliminate any that **DON'T**
report 1 black and 0 whites.

| Guess    | Black | White |
|----------|-------|-------|
| BWB      | 1     | 0     |
| ~~~BRB~~ | 2     | 0     |
| ~~~BRR~~ | 3     | 0     |
| ~~~WWB~~ | 0     | 1     |
| ~~~WRR~~ | 2     | 0     | 

Only one solution matches and it's our secret code! The computer will guess this
one next as it's the only choice left, for a total of three moves. 
Coincidentally, I believe the expected maximum number of moves the computer will 
make is the number of positions plus one for the initial guess with no information.
This is because it is winnowing down the solutions 
logarithmically on average. You noticed on the first pass, it wiped out 22 
solutions. If it was doing this logarithmically the worst case guess would 
still eliminate 18 of the solutions leaving 9 (3<sup>2</sup>).  So we have as
a guideline:

 Log<sub>(# of Colors)</sub>TotalPossibilities
 
but TotalPossibilities = (# of Colors)<sup># of Positions</sup>

so you end up with the number of positions as a guess limit. If you consider the
simplest non-trivial puzzle, two colors with two positions, and you guess BW or 
WB first, the most you can logically deduce if you get 1 black and 1 white is 
that it is either WW, or BB which could bring your total guesses up to three 
which is the number of positions plus one.  So if your computer's turn is taking
longer than the number of positions plus one to find the answer then something 
is wrong with your code. 

#### Known Bugs

- Line 622 is unreachable, as the previous line ends in a GOTO and that line number is not referenced anywhere.  It appears that the intent was to tell the user the correct combination after they fail to guess it in 10 tries, which would be a very nice feature, but does not actually work.  (In the MiniScript port, I have made this feature work.)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


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


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)

Conversion Notes

- There are minor spacing issues which have been preserved in this port.
- This implementation uses switch expressions to concisely place the dice pips in the right place.
- Random() is only pseudo-random but perfectly adequate for the purposes of simulating dice rolls.
- Console width is assumed to be 120 chars for the purposes of centrally aligned the intro text.


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by Anthony Rubick [AnthonyMichaelTDM](https://github.com/AnthonyMichaelTDM)

If you wish to give the user more than 2 attempts to get the number, change value assigned to the num_tries variable at the start of the main function


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Mugwump

Your objective in this game is to find the four Mugwumps hiding on various squares of a 10 by 10 grid. Homebase (lower left) is position (0,0) and a guess is a pair of whole numbers (0 to 9), separated by commas. The first number is the number of units to the right of homebase and the second number is the distance above homebase.

You get ten guesses to locate the four Mugwumps; after each guess, the computer tells you how close you are to each Mugwump. Playing the game with the aid of graph paper and a compass should allow you to find all the Mugwumps in six or seven moves using triangulation similar to Loran radio navigation.

If you want to make the game somewhat more difficult, you can print the distance to each Mugwump either rounded or truncated to the nearest integer.

This program was modified slightly by Bob Albrecht of People’s Computer Company. It was originally written by students of Bud Valenti of Project SOLO in Pittsburg, Pennsylvania.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=114)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=129)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Name

NAME is a silly little ice-breaker to get a relationship going between a computer and a shy human. The sorting algorithm used is highly inefficient — as any reader of _Creative Computing_ will recognize, this is the worst possible sort for speed. But the program is good fun and that’s what counts here.

NAME was originally written by Geoffrey Chase of the Abbey, Portsmouth, Rhode Island.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=116)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=131)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Nicomachus

One of the most ancient forms of arithmetic puzzle is sometimes referred to as a “boomerang.” At some time, everyone has been asked to “think of a number,” and, after going through some process of private calculation, to state the result, after which the questioner promptly tells you the number you originally thought of. There are hundreds of varieties of this puzzle.

The oldest recorded example appears to be that given in _Arithmetica_ of Nicomachus, who died about the year 120. He tells you to think of any whole number between 1 and 100 and divide it successfully by 3, 5, and 7, telling him the remainder in each case. On receiving this information, he promptly discloses the number you thought of.

Can you discover a simple method of mentally performing this feat? If not, you can see how the ancient mathematician did it by looking at this program.

Nicomachus was written by David Ahl.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=117)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=132)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Nim

NIM is one of the oldest two-person games known to man; it is believed to have originated in ancient China. The name, which was coined by the first mathematician to analyze it, comes from an archaic English verb which means to steal or to take away. Objects are arranged in rows between the two opponents as in the following example:
|         |       |           |
|---------|-------|-----------|
| XXXXXXX | Row 1 | 7 Objects |
| XXXXX   | Row 2 | 5 Objects |
| XXX     | Row 3 | 3 Objects |
| X       | Row 4 | 1 Object  |

Opponents take turns removing objects until there are none left. The one who picks up the last object wins. The moves are made according to the following rules:
1. On any given turn only objects from one row may be removed. There is no restriction on which row or on how many objects you remove. Of course, you cannot remove more than are in the row.
2. You cannot skip a move or remove zero objects.

The winning strategy can be mathematically defined, however, rather than presenting it here, we’d rather let you find it on your own. HINT: Play a few games with the computer and mark down on a piece of paper the number of objects in each stack (in binary!) after each move. Do you see a pattern emerging?

This game of NIM is from Dartmouth College and allows you to specify any starting size for the four piles and also a win option. To play traditional NIM, you would simply specify 7,5,3 and 1, and win option 1.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=118)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=133)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

This can be a real challenge to port because of all the `GOTO`s going out of loops down to code. You may need breaks and continues, or other techniques.

#### Known Bugs

- If, after the player moves, all piles are gone, the code prints "MACHINE LOSES" regardless of the win condition (when line 1550 jumps to line 800).  This should instead jump to line 800 ("machine loses") if W=1, but jump to 820 ("machine wins") if W=2.


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Number

In contrast to other number guessing games where you keep guessing until you get the random number selected by the computer (GUESS, TRAP, STARS, etc.), in this game you only get one guess per play and you gain or lose points depending upon how close your guess is to the random number selected by the computer. You occasionally get a jackpot which will double your point count. You win when you get 500 points.

Tom Adametx wrote this program while a student at Curtis Junior High School in Sudbury, Massachusetts.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=121)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=136)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

Contrary to the description, the computer picks *five* random numbers per turn, not one.  You are not rewarded based on how close your guess is to one number, but rather to which of these five random numbers (if any) it happens to match exactly.


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by Anthony Rubick [AnthonyMichaelTDM](https://github.com/AnthonyMichaelTDM)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### One Check

In this game or puzzle, 48 checkers are placed on the two outside spaces of a standard 64-square checkerboard as shown:

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
| ● | ● | ● | ● | ● | ● | ● | ● |
| ● | ● | ● | ● | ● | ● | ● | ● |
| ● | ● |   |   |   |   | ● | ● |
| ● | ● |   |   |   |   | ● | ● |
| ● | ● |   |   |   |   | ● | ● |
| ● | ● |   |   |   |   | ● | ● |
| ● | ● | ● | ● | ● | ● | ● | ● |
| ● | ● | ● | ● | ● | ● | ● | ● |

The object is to remove as many checkers as possible by diagonal jumps (as in standard checkers).

It is easy to remove 30 to 39 checkers, a challenge to remove 40 to 44, and a substantial feat to remove 45 to 47.

The program was created and written by David Ahl.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=122)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=137)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Orbit

ORBIT challenges you to visualize spacial positions in polar coordinates. The object is to detonate a Photon explosive within a certain distance of a germ laden Romulan spaceship. This ship is orbiting a planet at a constant altitude and orbital rate (degrees/hour). The location of the ship is hidden by a device that renders the ship invisible, but after each bomb you are told how close to the enemy ship your bomb exploded. The challenge is to hit an invisible moving target with a limited number of shots.

The planet can be replaced by a point at its center (called the origin); then the ship’s position can be given as a distance form the origin and an angle between its position and the eastern edge of the planet.

```
direction
of orbit    <       ^ ship
              \     ╱
                \  ╱ <
                 |╱   \
                 ╱      \
                ╱         \
               ╱           | angle
              ╱           /
             ╱          /
            ╱         /
           ╱——————————————————— E

```

The distance of the bomb from the ship is computed using the law of cosines. The law of cosines states:

```
D = SQUAREROOT( R**2 + D1**2 - 2*R*D1*COS(A-A1) )
```

Where D is the distance between the ship and the bomb, R is the altitude of the ship, D1 is the altitude of the bomb, and A-A1 is the angle between the ship and the bomb.


```
                 bomb  <
                        ╲                   ^ ship
                         ╲                  ╱
                          ╲                ╱ <
                           ╲              ╱   \
                        D1  ╲            ╱      \
                             ╲        R ╱         \
                              ╲   A1   ╱           | A
                               ╲⌄——— ◝╱           /
                                ╲    ╱ \        /
                                 ╲  ╱   \      /
                                  ╲╱───────────────────── E

```

ORBIT was originally called SPACE WAR and was written by Jeff Lederer of Project SOLO Pittsburgh, Pennsylvania.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=124)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=139)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)

This Perl script is a port of orbit, which is the 68th entry in Basic
Computer Games.

In this game you are a planetary defense gunner trying to shoot down a
cloaked Romulan ship before it can escape.

This is pretty much a straight port of the BASIC into idiomatic Perl.


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by Anthony Rubick [AnthonyMichaelTDM](https://github.com/AnthonyMichaelTDM)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Pizza

In this game, you take orders for pizzas from people living in Hyattsville. Armed with a map of the city, you must then tell your delivery boy the address where the pizza is to be delivered. If the pizza is delivered to the correct address, the customer phones you and thanks you; if not, you must give the driver the correct address until the pizza gets delivered.

Some interesting modifications suggest themselves for this program such as pizzas getting cold after two incorrect delivery attempts or taking three or more orders at a time and figuring out the shortest delivery route. Send us your modifications!

This program seems to have surfaced originally at the University of Georgia in Athens, Georgia. The author is unknown.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=126)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=141)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Known Bugs

- The program does no validation of its input, and crashes if you enter coordinates outside the valid range.  (Ports may choose to improve on this, for example by repeating the prompt until valid coordinates are given.)

#### Porting Notes

(please note any difficulties or challenges in porting here)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/).


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Poetry

This program produces random verse which might loosely be considered in the Japanese Haiku style. It uses 20 phrases in four groups of five phrases each and generally cycles through the groups in order. It inserts commas (random — 19% of the time), indentation (random — 22% of the time), and starts new paragraphs (18% probability but at least once every 20 phrases).

The phrases in POETRY are somewhat suggestive of Edgar Allen Poe. Try it with phrases from computer technology, from love and romance, from four-year-old children, or from some other project. Send us the output.

Here are some phrases from nature to try:
```
Carpet of ferns     Mighty Oaks
Morning dew         Grace and beauty
Tang of dawn        Silently singing
Swaying pines       Nature speaking

Entrances me        Untouched, unspoiled
Soothing me         Shades of green
Rustling leaves     Tranquility
Radiates calm       …so peaceful
```

The original author of this program is unknown. It was modified and reworked by Jim Bailey, Peggy Ewing, and Dave Ahl at DEC.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=128)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=143)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

- The program begins by switching on `I`, which has not been initialized.  We should probably initialize this to 0, though this means the output always begins with the phrase "midnight dreary".

- Though the program contains an END statement (line 999), it is unreachable.  The program continues to generate output until it is forcibly interrupted.
