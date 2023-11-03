# BasicComputerGames源码解析 3

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

Conversion to [Pascal](https://en.wikipedia.org/wiki/Pascal_(programming_language))


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Basketball

This program simulates a game of basketball between Dartmouth College and an opponent of your choice. You are the Dartmouth captain and control the type of shot and defense during the course of the game.

There are four types of shots:
1. Long Jump Shot (30ft)
2. Short Jump Shot (15ft)
3. Lay Up
4. Set Shot

Both teams use the same defense, but you may call it:
- Enter (6): Press
- Enter (6.5): Man-to-man
- Enter (7): Zone
- Enter (7.5): None

To change defense, type "0" as your next shot.

Note: The game is biased slightly in favor of Dartmouth. The average probability of a Dartmouth shot being good is 62.95% compared to a probability of 61.85% for their opponent. (This makes the sample run slightly remarkable in that Cornell won by a score of 45 to 42 Hooray for the Big Red!)

Charles Bacheller of Dartmouth College was the original author of this game.

---

As published in Basic Computer Games (1978)
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=12)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=27)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)

##### Original bugs

###### Initial defense selection

If a number <6 is entered for the starting defense then the original code prompts again until a value >=6 is entered,
but then skips the opponent selection center jump.

The C# port does not reproduce this behavior. It does prompt for a correct value, but will then go to opponent selection
followed by the center jump.

###### Unvalidated defense selection

The original code does not validate the value entered for the defense beyond checking that it is >=6. A large enough
defense value will guarantee that all shots are good, and the game gets rather predictable.

This bug is preserved in the C# port.


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

Conversion to [Pascal](https://en.wikipedia.org/wiki/Pascal_(programming_language))


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)

There are two version of the code here, a "faithful" translation (basketball-orig.pl) and
a "modern" translation (basketball.pl). The main difference between the 2 are is that the
faithful translation has 3 GOTOs in it while the modern version has no GOTO. I have added
a "TIME" print when the score is shown so the Clock is visible. Halftime is at "50" and
end of game is at 100 (per the Basic code).

The 3 GOTOs in the faitful version are because of the way the original code jumped into
the "middle of logic" that has no obivious way to avoid ... that I can see, at least while
still maintaining something of the look and structure of the original Basic.

The modern version avoided the GOTOs by restructuring the program in the 2 "play()" subs.
Despite the change, this should play the same way as the faithful version.

All of the percentages remain the same. If writing this from scratch, we really should
have only a single play() sub which uses the same code for both teams, which would also
make the game more fair ... but that wasn't done so the percent edge to Darmouth has been
maintained here.


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Batnum

The game starts with an imaginary pile of objects, coins for example. You and your opponent (the computer) alternately remove objects from the pile. You specify in advance the minimum and maximum number of objects that can be taken on each turn. You also specify in advance how winning is defined:
1. To take the last object
2. To avoid taking the last object

You may also determine whether you or the computer go first.

The strategy of this game is based on modulo arithmetic. If the maximum number of objects a player may remove in a turn is M, then to gain a winning position a player at the end of his turn must leave a stack of 1 modulo (M+1) coins. If you don’t understand this, play the game 23 Matches first, then BATNUM, and have fun!

BATNUM is a generalized version of a great number of manual remove-the-object games. The original computer version was written by one of the two originators of the BASIC language, John Kemeny of Dartmouth College.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=14)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=29)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Known Bugs

- Though the instructions say "Enter a negative number for new pile size to stop playing," this does not actually work.

#### Porting Notes

(please note any difficulties or challenges in porting here)



Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)

This conversion uses C#9 and is built for .net 5.0

Functional changes from Original
- handle edge condition for end game where the minimum draw amount is greater than the number of items remaining in the pile
- Takes into account the width of the console
- Mulilingual Support (English/French currently)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Pascal](https://en.wikipedia.org/wiki/Pascal_(programming_language))


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/) by [Austin White](https://github.com/austinwhite)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Battle

BATTLE is based on the popular game Battleship which is primarily played to familiarize people with the location and designation of points on a coordinate plane.

BATTLE first randomly sets up the bad guy’s fleet disposition on a 6 by 6 matrix or grid. The fleet consists of six ships:
- Two destroyers (ships number 1 and 2) which are two units long
- Two cruisers (ships number 3 and 4) which are three units long
- Two aircraft carriers (ships number 5 and 6) which are four units long

The program then prints out this fleet disposition in a coded or disguised format (see the sample computer print-out). You then proceed to sink the various ships by typing in the coordinates (two digits. each from 1 to 6, separated by a comma) of the place where you want to drop a bomb, if you’ll excuse the expression. The computer gives the appropriate response (splash, hit, etc.) which you should record on a 6 by 6 matrix. You are thus building a representation of the actual fleet disposition which you will hopefully use to decode the coded fleet disposition printed out by the computer. Each time a ship is sunk, the computer prints out which ships have been sunk so far and also gives you a “SPLASH/HIT RATIO.”

The first thing you should learn is how to locate and designate positions on the matrix, and specifically the difference between “3,4” and “4,3.” Our method corresponds to the location of points on the coordinate plane rather than the location of numbers in a standard algebraic matrix: the first number gives the column counting from left to right and the second number gives the row counting from bottom to top.

The second thing you should learn about is the splash/hit ratio. “What is a ratio?” A good reply is “It’s a fraction or quotient.” Specifically, the spash/hit ratio is the number of splashes divided by the number of hits. If you had 9 splashes and 15 hits, the ratio would be 9/15 or 3/5, both of which are correct. The computer would give this splash/hit ratio as .6.

The main objective and primary education benefit of BATTLE comes from attempting to decode the bad guys’ fleet disposition code. To do this, you must make a comparison between the coded matrix and the actual matrix which you construct as you play the game.

The original author of both the program and these descriptive notes is Ray Westergard of Lawrence Hall of Science, Berkeley, California.

---

As published in Basic Computer Games (1978)
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=15)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=30)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

- The original game has no way to re-view the fleet disposition code once it scrolls out of view.  Ports should consider allowing the user to enter "?" at the "??" prompt, to reprint the disposition code.  (This is added by the MiniScript port under Alternate Languages, for example.)

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

Conversion to [Pascal](https://en.wikipedia.org/wiki/Pascal_(programming_language))


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Blackjack

This is a simulation of the card game of Blackjack or 21, Las Vegas style. This rather comprehensive version allows for up to seven players. On each hand a player may get another card (a hit), stand, split a hand in the event two identical cards were received or double down. Also, the dealer will ask for an insurance bet if he has an exposed ace.

Cards are automatically reshuffled as the 51st card is reached. For greater realism, you may wish to change this to the 41st card. Actually, fanatical purists will want to modify the program so it uses three decks of cards instead of just one.

This program originally surfaced at Digital Equipment Corp.; the author is unknown.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=18)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=33)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html


#### Porting Notes

The program makes extensive use of the assumption that a boolean expression evaluates to **-1** for true.  This was the case in some classic BASIC environments but not others; and it is not the case in [JS Basic](https://troypress.com/wp-content/uploads/user/js-basic/index.html), leading to nonsensical results.  In an environment that uses **1** instead of **-1** for truth, you would need to negate the boolean expression in the following lines:
	- 10
	- 570
	- 590
	- 2220
	- 2850
	- 3100
	- 3400
	- 3410
	- 3420


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


### Bounce

This program plots a bouncing ball. Most computer plots run along the paper in the terminal (top to bottom); however, this plot is drawn horizontally on the paper (left to right).

You may specify the initial velocity of the ball and the coefficient of elasticity of the ball (a superball is about 0.85 — other balls are much less). You also specify the time increment to be used in “strobing” the flight of the ball. In other words, it is as though the ball is thrown up in a darkened room and you flash a light at fixed time intervals and photograph the progress of the ball.

The program was originally written by Val Skalabrin while he was at DEC.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=25)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=40)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# Bounce

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)

## Conversion notes

### Mode of Operation

This conversion performs the same function as the original, and provides the same experience, but does it in a different
way.

The original BASIC code builds the graph as it writes to the screen, scanning each line for points that need to be
plotted.

This conversion steps through time, calculating the position of the ball at each instant, building the graph in memory.
It then writes the graph to the output in one go.

### Failure Modes

The original BASIC code performs no validation of the input parameters. Some combinations of parameters produce no
output, others crash the program.

In the spirit of the original this conversion also performs no validation of the parameters, but it does not attempt to
replicate the original's failure modes. It fails quite happily in its own way.


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

Added feature so that if "TIME" value is "0" then it will quit,
so you don't have to hit Control-C. Also added a little error checking of the input.


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Bowling

This is a simulated bowling game for up to four players. You play 10 frames. To roll the ball, you simply type “ROLL.” After each roll, the computer will show you a diagram of the remaining pins (“0” means the pin is down, “+” means it is still standing), and it will give you a roll analysis:
- GUTTER
- STRIKE
- SPARE
- ERROR (on second ball if pins still standing)

Bowling was written by Paul Peraino while a student at Woodrow Wilson High School, San Francisco, California.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=26)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=41)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Known Bugs

- In the original code, scores is not kept accurately in multiplayer games.  It stores scores in F*P, where F is the frame and P is the player.  So, for example, frame 8 player 1 (index 16) clobbers the score from frame 4 player 2 (also index 16).

- Even when scores are kept accurately, they don't match normal bowling rules.  In this game, the score for each ball is just the total number of pins down after that ball, and the third row of scores is a status indicator (3 for strike, 2 for spare, 1 for anything else).

- The program crashes with a "NEXT without FOR" error if you elect to play again after the first game.

#### Porting Notes

- The funny control characters in the "STRIKE!" string literal are there to make the terminal beep.


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

###Bowling program in Perl

Run normally, this is a fairly faithful translation of the Basic game.
The only real differences are a few trivial fix-ups on the prints to make it
look better, and the player/frame/ball line was put before the "get the ball
going" line to make it more obvious who's turn it is.

However, if you run it with "-a" on the command line, it will go into
"advanced" mode, which means that "." is used to show pin down and "!" for
pin up, current running scores are shown at the end of each frame, and the
scoring also looks more normal at the end. This is all done because I think it
looks better and I wanted to see a score. Having a flag says you can play
whichever version of the game you like.

Note, the original code doesn't do the 10th frame correctly, in that it will
never do more than 2 balls, so the best score you can get is a 290.
This is true in both modes. That being said, it will always give you a mediocre
game; I don't think I've ever seen a score over 140.


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Boxing

This program simulates a three-round Olympic boxing match. The computer coaches one of the boxers and determines his punches and defences, while you do the same for your boxer. At the start of the match, you may specify your man’s best punch and his vulnerability.

There are approximately seven major punches per round, although this may be varied. The best out of three rounds wins.

Jesse Lynch of St. Paul, Minnesota created this program.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=28)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=43)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Known Bugs

- The code that handles player punch type 1 checks for opponent weakness type 4; this is almost certainly a mistake.

- Line breaks or finishing messages are omitted in various cases.  For example, if the player does a hook, and that's the opponent's weakness, then 7 points are silently awarded without outputting any description or line break, and the next sub-round will begin on the same line.

- When the opponent selects a hook, control flow falls through to the uppercut case.  Perhaps related, a player weakness of type 2 (hook) never has any effect on the game.

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


### Bug

The object of this game is to finish your drawing of a bug before the computer finishes.

You and the computer roll a die alternately with each number standing for a part of the bug. You must add the parts in the right order; in other words, you cannot have a neck until you have a body, you cannot have a head until you have a neck, and so on. After each new part has been added, you have the option of seeing pictures of the two bugs.

If you elect to see all the pictures, this program has the ability of consuming well over six feet of terminal paper per run. We can only suggest recycling the paper by using the other side.

Brian Leibowitz wrote this program while in the 7th grade at Harrison Jr-Se High School in Harrison, New York.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=30)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=45)

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


### Bullfight

In this simulated bullfight, you are the matador — i.e., the one with the principle role and the one who must kill the bull or be killed (or run from the ring).

On each pass of the bull, you may try:
- 0: Veronica (dangerous inside move of the cape)
- 1: Less dangerous outside move of the cape
- 2: Ordinary swirl of the cape

Or you may try to kill the bull:
- 4: Over the horns
- 5: In the chest

The crowd will determine what award you deserve, posthumously if necessary. The braver you are, the better the reward you receive. It’s nice to stay alive too. The better the job the picadores and toreadores do, the better your chances.

David Sweet of Dartmouth wrote the original version of this program. It was then modified by students at Lexington High School and finally by Steve North of Creative Computing.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=32)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=47)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

- There is a fundamental assumption in the pre-fight subroutine at line 1610, that the Picadores and Toreadores are more likely to do a bad job (and possibly get killed) with a low-quality bull. This appears to be a mistake in the original code, but should be retained.

- Lines 1800-1820 (part of the pre-fight subroutine) can never be reached.


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


### Bullseye

In this game, up to 20 players throw darts at a target with 10-, 20-, 30-, and 40-point zones. The objective is to get 200 points.

You have a choice of three methods of throwing:

| Throw | Description        | Probable Score            |
|-------|--------------------|---------------------------|
| 1     | Fast overarm       | Bullseye or complete miss |
| 2     | Controlled overarm | 10, 20, or 30 points      |
| 3     | Underarm           | Anything                  |

You will find after playing a while that different players will swear by different strategies. However, considering the expected score per throw by always using throw 3:

| Score (S) | Probability (P) | S x P |
|-----------|-----------------|-------|
|     40    |  1.00-.95 = .05 |   2   |
|     30    |  .95-.75 = .20  |   6   |
|     30    |  .75-.45 = .30  |   6   |
|     10    |  .45-.05 = .40  |   4   |
|     0     |  .05-.00 = .05  |   0   |

Expected score per throw = 18

Calculate the expected score for the other throws and you may be surprised!

The program was written by David Ahl of Creative Computing.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=34)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=49)

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

 # Separation of concerns for Binary Projects

The organizational problem of allocating responsibility for multiple tasks to the main function is common to many projects. As a result, the Rust community has developed a process to use as a guideline for splitting the separate concerns of a binary program when main starts getting large. 

 ## The process has the following steps:
 - Split your program into a main.rs and a lib.rs and move your program’s logic to lib.rs.
 - As long as your command line logic is small, it can remain in main.rs.
 - When the command line logic starts getting complicated, extract it from main.rs and move it to lib.rs.

 ## The responsibilities that remain in the main function after this process should be limited to the following:
 - Calling the command line or input parsing logic with the argument values
 - Setting up any other configuration
 - Calling a run function in lib.rs
 - Handling the error if run returns an error

This pattern is about separating concerns: main.rs handles running the program, and lib.rs handles all the logic of the task at hand. Because you can’t test the main function directly, this structure lets you test all of your program’s logic by moving it into functions in lib.rs. The only code that remains in main.rs will be small enough to verify its correctness by reading it.

Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Bunny

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=35)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=50)

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

There are two versions of this program here:

* `bunny-faithful.rb` tries to be faithful to the design of the original
  BASIC program.
* `bunny-modern.rb` takes more advantage of the features of modern
  tools and languages.


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Buzzword

This program is an invaluable aid for preparing speeches and briefings about educational technology. This buzzword generator provides sets of three highly-acceptable words to work into your material. Your audience will never know that the phrases don’t really mean much of anything because they sound so great! Full instructions for running are given in the program.

This version of Buzzword was written by David Ahl.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=36)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=51)

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


### Calendar

This program prints out a calendar for any year. You must specify the starting day of the week of the year:
- 0: Sunday
- -1: Monday
- -2: Tuesday
- -3: Wednesday
- -4: Thursday
- -5: Friday
- -6: Saturday

You can determine this by using the program WEEKDAY. You must also make two changes for leap years. The program listing describes the necessary changes. Running the program produces a nice 12-month calendar.

The program was written by Geoffrey Chase of the Abbey, Portsmouth, Rhode Island.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=37)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=52)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

- While many modern environments have time/date functions that would make this program both easier and more automatic, in these ports we are choosing to do without them, as in the original program.

- Some ports choose to ask the user the starting day of week, and whether it's a leap year, rather than force changes to the code to fit the desired year.


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

Actually, this is not so much a port as a complete rewrite, making use of
Perl's Posix time functionality. The calendar is for the current year (not
1979), but you can get another year by specifying it on the command line, e.g.

 `perl 21_Calendar/perl/calendar.pl 2001`

It *may* even produce output in languages other than English. But the
leftmost column will still be Sunday, even in locales where it is
typically Monday.


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by [Uğur Küpeli](https://github.com/ugurkupeli)

Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Change

In this program, the computer pretends it is the cashier at your friendly neighborhood candy store. You tell it the cost of the item(s) you are buying, the amount of your payment, and it will automatically (!) determine your correct change. Aren’t machines wonderful? Dennis Lunder of People’s Computer Company wrote this program.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=39)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=54)

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


### Checkers

This program plays checkers. The pieces played by the computer are marked with an “X”, yours are marked “O”. A move is made by specifying the coordinates of the piece to be moved (X, Y). Home (0,0) is in the bottom left and X specifies distance to the right of home (i.e., column) and Y specifies distance above home (i.e. row). You then specify where you wish to move to.

The original version of the program by Alan Segal was not able to recognize (or permit) a double or triple jump. If you tried one, it was likely that your piece would disappear altogether!

Steve North of Creative Computing rectified this problem and Lawrence Neal contributed modifications to allow the program to tell which player has won the game. The computer does not play a particularly good game but we leave it to _you_ to improve that.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=40)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=55)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

The file `checkers.annotated.bas` contains an indented and annotated
version of the source code.  This is no longer valid BASIC code but
should be more readable.

#### Porting Notes

 - If the computer moves a checker to the bottom row, it promotes, but
   leaves the original checker in place. (See line 1240)
 - Human players may move non-kings as if they were kings. (See lines 1590 to 1810)
 - Human players are not required to jump if it is possible, and may make a number
   other illegal moves (jumping their own pieces, jumping empty squares, etc.).
 - Curious writing to "I" variable without ever reading it. (See lines 1700 and 1806)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)
