# BasicComputerGames源码解析 7

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


### Poker

You and the computer are opponents in this game of draw poker. At the start of the game, each player is given $200. The game ends when either player runs out of money, although if you go broke the computer will offer to buy back your wristwatch or diamond tie tack.

The computer opens the betting before the draw; you open the betting after the draw. If you don’t have a hand that’s worth anything and you want to fold, bet 0. Prior to the draw, to check the draw, you may bet .5. Of course, if the computer has made a bet, you must match it in order to draw or, if you have a good hand, you may raise the bet at any time.

The author is A. Christopher Hall of Trinity College, Hartford, Connecticut.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=129)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=144)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Known Bugs

- If you bet more than the computer has, it will still see you, resulting in a negative balance.  (To handle this properly, the computer would need to go "all in" and reduce your bet to an amount it can match; or else lose the game, which is what happens to the human player in the same situation.)

- If you are low on cash and sell your watch, then make a bet much smaller than the amount you just gained from the watch, it sometimes nonetheless tells you you "blew your wad" and ends the game.

- When the watch is sold (in either direction), the buyer does not actually lose any money.

- The code in the program about selling your tie tack is unreachable due to a logic bug.


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


### Queen

This game is based on the permissible moves of the chess queen — i.e., along any vertical, horizontal, or diagonal. In this game, the queen can only move to the left, down, and diagonally down to the left.

The object of the game is to place the queen (one only) in the lower left-hand square (no. 158), by alternating moves between you and the computer. The one to place the queen there wins.

You go first and place the queen in any one of the squares on the top row or the right-hand column. That is your first move. The computer is beatable, but it takes some figuring. See if you can devise a winning strategy.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=133)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=148)

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

Conversion to [Python](https://www.python.org/about/) by Christopher Phan.
Supports Python version 3.8 or later.


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Reverse

The game of REVERSE requires you to arrange a list of numbers in numerical order from left to right. To move, you tell the computer how many numbers (counting from the left) to reverse. For example, if the current list is:
```
    2 3 4 5 1 6 7 8 9
```

and you reverse 4, the result will be:
```
    5 4 3 2 1 6 7 8 9
```
Now if you reverse 5, you win!

There are many ways to beat the game, but approaches tend to be either algorithmic or heuristic. The game thus offers the player a chance to play with these concepts in a practical (rather than theoretical) context.

An algorithmic approach guarantees a solution in a predictable number of moves, given the number of items in the list. For example, one method guarantees a solution in 2N - 3 moves when teh list contains N numbers. The essence of an algorithmic approach is that you know in advance what your next move will be. Once could easily program a computer to do this.

A heuristic approach takes advantage of “partial orderings” in the list at any moment. Using this type of approach, your next move is dependent on the way the list currently appears. This way of solving the problem does not guarantee a solution in a predictable number of moves, but if you are lucky and clever, you may come out ahead of the algorithmic solutions. One could not so easily program this method.

In practice, many players adopt a “mixed” strategy, with both algorithmic and heuristic features. Is this better than either “pure” strategy?

The program was created by Peter Sessions of People’s Computer Company and the notes above adapted from his original write-up.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=135)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=150)

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


### Rock, Scissors, Paper

Remember the game of rock-scissors-paper. You and your opponent make a motion three times with your fists and then either show:
- a flat hand (paper)
- fist (rock)
- two fingers (scissors)

Depending upon what is shown, the game is a tie (both show the same) or one person wins. Paper wraps up rock, so it wins. Scissors cut paper, so they win. And rock breaks scissors, so it wins.

In this computerized version of rock-scissors-paper, you can play up to ten games vs. the computer.

Charles Lund wrote this game while at the American School in The Hague, Netherlands.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=137)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=152)

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

Conversion to [Rust](https://www.rust-lang.org)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Roulette

This game simulates an American Roulette wheel; “American” because it has 38 number compartments (1 to 36, 0 and 00). The European wheel has 37 numbers (1 to 36 and 0). The Bahamas, Puerto Rico, and South American countries are slowly switching to the American wheel because it gives the house a bigger percentage. Odd and even numbers alternate around the wheel, as do red and black. The layout of the wheel insures a highly random number pattern. In fact, roulette wheels are sometimes used to generate tables of random numbers.

In this game, you may bet from $5 to $500 and you may bet on red or black, odd or even, first or second 18 numbers, a column, or single number. You may place any number of bets on each spin of the wheel.

There is no long-range winning strategy for playing roulette. However, a good strategy is that of “doubling.” First spin, bet $1 on an even/odds bet (odd, even, red, or black). If you lose, double your bet again to $2. If you lose again, double to $4. Continue to double until you win (i.e, you break even on a losing sequence). As soon as you win, bet $1 again, and after every win, bet $1. Do not ever bet more than $1 unless you are recuperating losses by doubling. Do not ever bet anything but the even odds bets. Good luck!

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=138)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=153)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

- The program keeps a count of how often each number comes up in array `X`, but never makes use of this information.


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)

Two versions of Roulette has been contributed. They are indicated within given sub-folders

- [oop](./oop) - Conversion by Andrew McGuinness (andrew@arobeia.co.uk)
- [iterative](./iterative) - Conversion by Thomas Kwashnak ([Github](https://github.com/LittleTealeaf)).
    - Implements features from JDK 17.
    - Does make use of some object oriented programming, but acts as a more iterative solution.


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


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


### Salvo

The rules are _not_ explained by the program, so read carefully this description by Larry Siegel, the program author.

SALVO is played on a 10x10 grid or board using an x,y coordinate system. The player has 4 ships:
- battleship (5 squares)
- cruiser (3 squares)
- two destroyers (2 squares each)

The ships may be placed horizontally, vertically, or diagonally and must not overlap. The ships do not move during the game.

As long as any square of a battleship still survives, the player is allowed three shots, for a cruiser 2 shots, and for each destroyer 1 shot. Thus, at the beginning of the game the player has 3+2+1+1=7 shots. The players enters all of his shots and the computer tells what was hit. A shot is entered by its grid coordinates, x,y. The winner is the one who sinks all of the opponents ships.

Important note: Your ships are located and the computer’s ships are located on 2 _separate_ 10x10 boards.

Author of the program is Lawrence Siegel of Shaker Heights, Ohio.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=142)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=157)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

The program does no validation of ship positions; your ship coordinates may be scattered around the board in any way you like.  (Computer ships will not do this, but they may be placed diagonally in such a way that they cross each other.)  Scattering your ships in this way probably defeats whatever all that spaghetti-code logic the computer is using to pick its moves, which is based on the assumption of contiguous ships.

Moreover: as per the analysis in

https://forums.raspberrypi.com/viewtopic.php?p=1997950#p1997950

see also the earlier post

https://forums.raspberrypi.com/viewtopic.php?p=1994961#p1994961

in the same thread, there is a typo in later published versions of the SALVO Basic source code compared to the original edition of 101 Basic Computer Games.

This typo is interesting because it causes the program to play by a much weaker strategy while exhibiting no other obvious side effects. I would recommend changing the line 3970 in the Basic program back to the original

`3970 K(R,S)=K(R,S)+E(U)-2*INT(H(U)+.5)`

and to change the JavaScript program accordingly.  (And note that some ports — looking at you, Python — do not implement the original strategy at all, but merely pick random unshot locations for every shot.)



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


### Sine Wave

Did you ever go to a computer show and see a bunch of CRT terminals just sitting there waiting forlornly for someone to give a demo on them. It was one of those moments when I was at DEC that I decided there should be a little bit of background activity. And why not plot with words instead of the usual X’s? Thus SINE WAVE was born and lives on in dozens of different versions. At least those CRTs don’t look so lifeless anymore.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=146)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=161)

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

Conversion to [Rust](https://www.rust-lang.org/) by Anthony Rubick [AnthonyMichaelTDM](https://github.com/AnthonyMichaelTDM)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Slalom

This game simulates a slalom run down a course with one to 25 gates. The user picks the number of gates and has some control over his speed down the course.

If you’re not a skier, here’s your golden opportunity to try it with minimal risk. If you are a skier, here’s something to do while your leg is in a cast.

SLALOM was written by J. Panek while a student at Dartmouth College.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=147)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=162)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Known Bugs

- In the original version, the data pointer doesn't reset after a race is completed. This causes subsequent races to error at some future point at line 540, `READ Q'.

- It also doesn't restore the data pointer after executing the MAX command to see the gate speeds, meaning that if you use this command, it effectively skips those gates, and the speeds shown are completely incorrect.

#### Porting Notes


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


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


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


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Java](https://openjdk.java.net/)


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


### Stars

In this game, the computer selects a random number from 1 to 100 (or any value you set). You try to guess the number and the computer gives you clues to tell you how close you’re getting. One star (\*) means you’re far away from the number; seven stars (\*\*\*\*\*\*\*) means you’re really close. You get 7 guesses.

On the surface this game is similar to GUESS; however, the guessing strategy is quite different. See if you can come up with one or more approaches to finding the mystery number.

Bob Albrecht of People’s Computer Company created this game.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=153)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=166)

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


<<<<<<< HEAD
#STARS

From: BASIC Computer Games (1978), edited by David H. Ahl

In this game, the computer selects a random number from 1 to 100
(or any value you set [for MAX_NUM]).  You try to guess the number
and the computer gives you clues to tell you how close you're
getting.  One star (*) means you're far away from the number; seven
stars (*******) means you're really close.  You get 7  guesses.

On the surface this game is very similar to GUESS; however, the
guessing strategy is quite different.  See if you can come up with
one or more approaches to finding the mystery number.

Bob Albrecht of People's Computer Company created this game.

## NOTES

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by JW Bruce

thanks to Jeff Jetton for his Python port which provide inspiration
=======
Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/)
>>>>>>> 3e27c70ca800f5efbe6bc1a7d180211decf55b7d


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Stock Market

This program “plays” the stock market. You will be given $10,000 and may buy or sell stocks. Stock prices and trends are generated randomly; therefore, this model does not represent exactly what happens on the exchange. (Depending upon your point of view, you may feel this is quite a good representation!)

Every trading day, a table of stocks, their prices, and number of shares in your portfolio is printed. Following this, the initials of each stock are printed followed by a question mark. You indicate your transaction in number of shares — a positive number to buy, negative to sell, or 0 to do no trading. A brokerage fee of 1% is charges on all transactions (a bargain!). Note: Even if the value of a stock drops to zero, it may rebound again — then again, it may not.

This program was created by D. Pessel, L. Braun, and C. Losik of the Huntington Computer Project at SUNY, Stony Brook, N.Y.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=154)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=166)

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


### Super Star Trek

#### Brief History
Many versions of Star Trek have been kicking around various college campuses since the late sixties. I recall playing one at Carnegie-Mellon Univ. in 1967 or 68, and a very different one at Berkeley. However, these were a far cry from the one written by Mike Mayfield of Centerline Engineering and/or Custom Data. This was written for an HP2000C and completed in October 1972. It became the “standard” Star Trek in February 1973 when it was put in the HP contributed program library and onto a number of HP Data Center machines.

In the summer of 1973, I converted the HP version to BASIC-PLUS for DEC’s RSTS-11 compiler and added a few bits and pieces while I was at it. Mary Cole at DEC contributed enormously to this task too. Later that year I published it under the name SPACWE (Space War — in retrospect, an incorrect name) in my book _101 Basic Computer Games_.It is difficult today to find an interactive computer installation that does not have one of these versions of Star Trek available.

#### Quadrant Nomenclature
Recently, certain critics have professed confusion as to the origin on the “quadrant” nomenclature used on all standard CG (Cartesian Galactic) maps. Naturally, for anyone with the remotest knowledge of history, no explanation is necessary; however, the following synopsis should suffice for the critics:

As everybody schoolboy knows, most of the intelligent civilizations in the Milky Way had originated galactic designations of their own choosing well before the Third Magellanic Conference, at which the so-called “2⁶ Agreement” was reached. In that historic document, the participant cultures agreed, in all two-dimensional representations of the galaxy, to specify 64 major subdivisions, ordered as an 8 x 8 matrix. This was partially in deference to the Earth culture (which had done much in the initial organization of the Federation), whose century-old galactic maps had landmarks divided into four “quadrants,” designated by ancient “Roman Numerals” (the origin of which has been lost).

To this day, the official logs of starships originating on near-Earth starbases still refer to the major galactic areas as “quadrants.”

The relation between the Historical and Standard nomenclatures is shown in the simplified CG map below.

|   | 1            | 2  | 3   | 4  | 5          | 6  | 7   | 8  |
|---|--------------|----|-----|----|------------|----|-----|----|
| 1 |    ANTARES   |    |     |    |   SIRIUS   |    |     |    |
|   | I            | II | III | IV | I          |    | III | IV |
| 2 |     RIGEL    |    |     |    |    DENEB   |    |     |    |
|   | I            | II | III | IV | I          | II | III | IV |
| 3 |    PROCYON   |    |     |    |   CAPELLA  |    |     |    |
|   | I            | II | III | IV | I          | II | III | IV |
| 4 | VEGA         |    |     |    | BETELGUESE |    |     |    |
|   | I            | II | III | IV | I          | II | III | IV |
| 5 |    CANOPUS   |    |     |    |  ALDEBARA  |    |     |    |
|   | I            | II | III | IV | I          | II | III | IV |
| 6 |    ALTAIR    |    |     |    |   REGULUS  |    |     |    |
|   | I            | II | III | IV | I          | II | III | IV |
| 7 | SAGITTARIOUS |    |     |    |  ARCTURUS  |    |     |    |
|   | I            | II | III | IV | I          | II | III | IV |
| 8 |    POLLUX    |    |     |    |    SPICA   |    |     |    |
|   | I            | II | III | IV | I          | II | III | IV |

#### Super Star Trek† Rules and Notes
1. OBJECTIVE: You are Captain of the starship “Enterprise”† with a mission to seek and destroy a fleet of Klingon† warships (usually about 17) which are menacing the United Federation of Planets.† You have a specified number of stardates in which to complete your mission. You also have two or three Federation Starbases† for resupplying your ship.

2. You will be assigned a starting position somewhere in the galaxy. The galaxy is divided into an 8 x 8 quadrant grid. The astronomical name of a quadrant is called out upon entry into a new region. (See “Quadrant Nomenclature.”) Each quadrant is further divided into an 8 x 8 section grid.

3. On a section diagram, the following symbols are used:
    - `<*>` Enterprise
    - `†††` Klingon
    - `>!<` Starbase
    - `*`   Star

4. You have eight commands available to you (A detailed description of each command is given in the program instructions.)
    - `NAV` Navigate the Starship by setting course and warp engine speed.
    - `SRS` Short-range sensor scan (one quadrant)
    - `LRS` Long-range sensor scan (9 quadrants)
    - `PHA` Phaser† control (energy gun)
    - `TOR` Photon torpedo control
    - `SHE` Shield control (protects against phaser fire)
    - `DAM` Damage and state-of-repair report
    - `COM` Call library computer

5. Library computer options are as follows (more complete descriptions are in program instructions):
    - `0` Cumulative galactic report
    - `1` Status report
    - `2` Photon torpedo course data
    - `3` Starbase navigation data
    - `4` Direction/distance calculator
    - `5` Quadrant nomenclature map

6. Certain reports on the ship’s status are made by officers of the Enterprise who appears on the original TV Show—Spock,† Scott,† Uhura,† Chekov,† etc.

7. Klingons are non-stationary within their quadrants. If you try to maneuver on them, they will move and fire on you.

8. Firing and damage notes:
    - Phaser fire diminishes with increased distance between combatants.
    - If a Klingon zaps you hard enough (relative to your shield strength) he will generally cause damage to some part of your ship with an appropriate “Damage Control” report resulting.
    - If you don’t zap a Klingon hard enough (relative to his shield strength) you won’t damage him at all. Your sensors will tell the story.
    - Damage control will let you know when out-of-commission devices have been completely repaired.

9. Your engines will automatically shut down if you should attempt to leave the galaxy, or if you should try to maneuver through a star, or Starbase, or—heaven help you—a Klingon warship.

10. In a pinch, or if you should miscalculate slightly, some shield control energy will be automatically diverted to warp engine control (if your shield are operational!).

11. While you’re docked at a Starbase, a team of technicians can repair your ship (if you’re willing for them to spend the time required—and the repairmen _always_ underestimate…)

12. If, to same maneuvering time toward the end of the game, you should cold-bloodedly destroy a Starbase, you get a nasty note from Starfleet Command. If you destroy your _last_ Starbase, you lose the game! (For those who think this is too a harsh penalty, delete line 5360-5390, and you’ll just get a “you dumdum!”-type message on all future status reports.)

13. End game logic has been “cleaned up” in several spots, and it is possible to get a new command after successfully completing your mission (or, after resigning your old one).

14. For those of you with certain types of CRT/keyboards setups (e.g. Westinghouse 1600), a “bell” character is inserted at appropriate spots to cause the following items to flash on and off on the screen:
    - The Phrase “\*RED\*” (as in Condition: Red)
    - The character representing your present quadrant in the cumulative galactic record printout.

15. This version of Star Trek was created for a Data General Nova 800 system with 32K or core. So that it would fit, the instructions are separated from the main program via a CHAIN. For conversion to DEC BASIC-PLUS, Statement 160 (Randomize) should be moved after the return from the chained instructions, say to Statement 245. For Altair BASIC, Randomize and the chain instructions should be eliminated.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=157)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=166)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

Instructions in this directory at
instructions.txt

#### Porting Notes

Many of the programs in this book and this collection have bugs in the original code.

@jkboyce has done a great job of discovering and fixing a number of bugs in the [original code](superstartrek.bas), as part of his [python implementation](python/superstartrek.py), which should be noted by other implementers:

- line `4410` : `D(7)` should be `D(6)`
- lines `8310`,`8330`,`8430`,`8450` : Division by zero is possible
- line `440` : `B9` should be initialised to 0, not 2


#### External Links
 - C++: https://www.codeproject.com/Articles/28399/The-Object-Oriented-Text-Star-Trek-Game-in-C


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/) by [Taciano Dreckmann Perez](https://github.com/taciano-perez).

Overview of Java classes:
- SuperStarTrekInstructions: displays game instructions
- SuperStarTrekGame: main game class
- GalaxyMap: map of the galaxy divided in quadrants and sectors, containing stars, bases, klingons, and the Enterprise
- Enterprise: the starship Enterprise
- GameCallback: interface allowing other classes to interact with the game class without circular dependencies 
- Util: utility methods

[This video](https://www.youtube.com/watch?v=cU3NKOnRNCI) describes the approach and the different steps followed to translate the game.

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


# Super Star Trek - Rust version

Explanation of modules:

- [main.rs](./src/main.rs) - creates the galaxy (generation functions are in model.rs as impl methods) then loops listening for commands. after each command checks for victory or defeat condtions.
- [model.rs](./src/model.rs) - all the structs and enums that represent the galaxy. key methods in here (as impl methods) are generation functions on galaxy and quadrant, and various comparison methods on the 'Pos' tuple type.
- [commands.rs](./src/commands.rs) - most of the code that implements instructions given by the player (some code logic is in the model impls, and some in view.rs if its view only).
- [view.rs](./src/view.rs) - all text printed to the output, mostly called by command.rs (like view::bad_nav for example). also contains the prompts printed to the user (e.g. view::prompts::COMMAND).
- [input.rs](./src/input.rs) - utility methods for getting input from the user, including logic for parsing numbers, repeating prompts until a correct value is provided etc.

Basically the user is asked for the next command, this runs a function that usually checks if the command system is working, and if so will gather additional input (see next note for a slight change here), then either the model is read and info printed, or its mutated in some way (e.g. firing a torpedo, which reduces the torpedo count on the enterprise and can destroy klingons and star bases; finally the klingons fire back and can destroy the enterprise). Finally the win/lose conditions are checked before the loop repeats.

## Changes from the original

I have tried to keep it as close as possible. Notable changes are:

- commands can be given with parameters in line. e.g. while 'nav' will ask for course and then warp speed in the original, here you can *optionally* also do this as one line, e.g. `nav 1 0.1` to move one sector east. I'm sorry - it was driving me insane in its original form (which is still sorted, as is partial application e.g. nav 1 to preset direction and then provide speed).
- text is mostly not uppercase, as text was in the basic version. this would be easy to change however as all text is in view.rs, but I chose not to.
- the navigation system (plotting direction, paths and collision detection) is as close as I could make it to the basic version (by using other language conversions as specification sources) but I suspect is not perfect. seems to work well enough however.

Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Synonym

A synonym of a word is another word (in the English language) which has the same, or very nearly the same, meaning. This program tests your knowledge of synonyms of a few common words.

The computer chooses a word and asks you for a synonym. The computer then tells you whether you’re right or wrong. If you can’t think of a synonym, type “HELP” which causes a synonym to be printed.

You may put in words of your choice in the data statements. The number following DATA in Statement 500 is the total number of data statements. In each data statement, the first number is the number of words in that statement.

Can you think of a way to make this into a more general kind of CAI program for any subject?

Walt Koetke of Lexington High School, Massachusetts created this program.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=164)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=179)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

 - Each time the player asks for HELP, one of the synonyms is shown
   and discarded. There is no protection against the player using up
   all of the help.

 - The player can ask for HELP and then submit that answer. Is it
   meant to be a clue, or just giving a correct answer to the player?


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

I used List::Util to do all the heavy work to show that perl can handle all the various
array functions.  It would be interesting to see a version that handled all of this
manually as there ended up being very little code left in this program.


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by [Jadi](https://github.com/jadijadi)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Target

In this program, you are firing a weapon from a spaceship in 3-dimensional space. Your ship, the Starship Enterprise, is located at the origin (0,0,0) of a set of x,y,z coordinates. You will be told the approximate location of the target in 3-dimensional rectangular coordinates, the approximate angular deviation from the x and z axes in both radians and degrees, and the approximate distance to the target.

Given this information, you then proceed to shoot at the target. A shot within 20 kilometers of the target destroys it. After each shot, you are given information as to the position of the explosion of your shot and a somewhat improved estimate of the location of the target. Fortunately, this is just practice and the target doesn’t shoot back. After you have attained proficiency, you ought to be able to destroy a target in 3 or 4 shots. However, attaining proficiency might take a while!

The author is H. David Crockett of Fort Worth, Texas.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=165)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=180)

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

Modified so that if the user enters "quit" or "stop" for the input, the program will exit.
This way the user doesn't have to enter Contorl-C to quit.

Target values can be space and/or comma separated, so "1 2 3" is valid, as is "1,2,3" or even "1, 2, 3".
I believe the original Basic program wanted "1,2,3" or else each on a separate line.


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### 3-D Plot

3-D PLOT will plot the family of curves of any function. The function Z is plotted as “rising” out of the x-y plane with x and y inside a circle of radius 30. The resultant plot looks almost 3-dimensional.

You set the function you want plotted in line 5. As with any mathematical plot, some functions come out “prettier” than others.

The author of this amazingly clever program is Mark Bramhall of DEC.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=167)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=182)

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
dmd -dip1000 -run threedeeplot.d
```

[Other compilers](https://dlang.org/download.html) also exist.

## On rounding floating point values to integer values

The D equivalent of Basic `INT` is [`floor`](https://dlang.org/phobos/std_math_rounding.html#.floor),
which rounds towards negative infinity. If you change occurrences of `floor` to
[`lrint`](https://dlang.org/phobos/std_math_rounding.html#.lrint), you'll see that the plots show a bit more detail,
as is done in the bonus below.

## Bonus: Self-writing programs

With a small modification to the source, the program can be extended to **plot a random function**, and **print its formula**.

```shell
rdmd -dip1000 threedeeplot_random.d
```
(`rdmd` caches the executable, which results in speedy execution when the source does not change.)

### Example output
```
                                    3D Plot
              (After Creative Computing  Morristown, New Jersey)


                           f(z) = 30 * sin(z / 10.0)

                             *
                      *      *    * *
                *         *      *    * *
                    *         *      *    * *
            *           *        *       *   *  *
               *           *         *      *   *  *
                  *           *         *     *    * **
       *             *           *        *      *   *  *
         *              *           *       *     *   *  **
            *              *          *       *    *   *  * *
              *              *          *      *   *   *  *  *
                *              *          *     *  *  *   *   **
                  *              *         *    * *  *   *    * *
   *                *             *        *    ** *    *     * *
    *                *             *        *  **     *      *   *
     *                 *            *       * *     *       *    *
      *                 *            *      * *   *         *    *
       *                *             *     ** *           *     **
        *                *            *     **            *      **
        *                *            *     *            *       **
        *                *            *     *            *       **
        *                *            *     *            *       **
        *                *            *     **            *      **
       *                *             *     ** *           *     **
      *                 *            *      * *   *         *    *
     *                 *            *       * *     *       *    *
    *                *             *        *  **     *      *   *
   *                *             *        *    ** *    *     * *
                  *              *         *    * *  *   *    * *
                *              *          *     *  *  *   *   **
              *              *          *      *   *   *  *  *
            *              *          *       *    *   *  * *
         *              *           *       *     *   *  **
       *             *           *        *      *   *  *
                  *           *         *     *    * **
               *           *         *      *   *  *
            *           *        *       *   *  *
                    *         *      *    * *
                *         *      *    * *
                      *      *    * *
                             *
```

### Breakdown of differences

Have a look at the relevant differences between `threedeeplot.d` and `threedeeplot_random.d`.
This is the original function with the single expression that is evaluated for the plot:
```d
    static float fna(float z)
    {
        return 30.0 * exp(-z * z / 100.0);
    }
```
Here `static` means that the nested function does not need acces to its enclosing scope.

Now, by inserting the following:
```d
    enum functions = ["30.0 * exp(-z * z / 100.0)",
                      "sqrt(900.01 - z * z) * .9 - 2",
                      "30 * (cos(z / 16.0) + .5)",
                      "30 - 30 * sin(z / 18.0)",
                      "30 * exp(-cos(z / 16.0)) - 30",
                      "30 * sin(z / 10.0)"];

    size_t index = uniform(0, functions.length);
    writeln(center("f(z) = " ~ functions[index], width), "\n");
```
and changing the implementation of `fna` to
```d
    float fna(float z)
    {
        final switch (index)
        {
            static foreach (i, f; functions)
                case i:
                    mixin("return " ~ f ~ ";");
        }
    }
```
we unlock some very special abilities of D. Let's break it down:

```d
    enum functions = ["30.0 * exp(-z * z / 100.0)", /*...*/];
```
This defines an array of strings, each containing a mathematical expression. Due to the `enum` keyword, this is an
array that really only exists at compile-time.

```d
    size_t index = uniform(0, functions.length);
```
This defines a random index into the array. `functions.length` is evaluated at compile-time, due to D's compile-time
function evaluation (CTFE).

```d
    writeln(center("f(z) = " ~ functions[index], width), "\n");
```
Unmistakenly, this prints the formula centered on a line. What happens behind the scenes is that `functions` (which
only existed at compile-time before now) is pasted in, so that an instance of that array actually exists at run-time
at this spot, and is instantly indexed.

```d
    float fna(float z)
    {
        final switch (index)
        {
            // ...
        }
    }
```
`static` has been dropped from the nested function because we want to evaluate `index` inside it. The function contains
an ordinary `switch`, with `final` providing some extra robustness. It disallows a `default` case and produces an error
when the switch doesn't handle all cases. The `switch` body is where the magic happens and consists of these three
lines:
```d
            static foreach (i, f; functions)
                case i:
                    mixin("return " ~ f ~ ";");
```
The `static foreach` iterates over `functions` at compile-time, producing one `case` for every element in `functions`.
`mixin` takes a string, which is constructed at compile-time, and pastes it right into the source.

In effect, the implementation of `float fna(float z)` unrolls itself into
```d
    float fna(float z)
    {
        final switch (index)
        {
            case 0:
                return 30.0 * exp(-z * z / 100.0);
            case 1:
                return sqrt(900.01 - z * z) * .9 - 2;
            case 2:
                return 30 * (cos(z / 16.0) + .5);
            case 3:
                return 30 - 30 * sin(z / 18.0);
            case 4:
                return 30 * exp(-cos(z / 16.0)) - 30;
            case 5:
                return 30 * sin(z / 10.0)";
        }
    }
```

So if you feel like adding another function, all you need to do is append it to the `functions` array, and the rest of
the program *rewrites itself...*
