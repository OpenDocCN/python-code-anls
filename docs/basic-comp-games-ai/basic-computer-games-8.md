# BasicComputerGames源码解析 8

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


### 3-D Tic-Tac-Toe

3-D TIC-TAC-TOE is a game of tic-tac-toe in a 4x4x4 cube. You must get 4 markers in a row or diagonal along any 3-dimensional plane in order to win.

Each move is indicated by a 3-digit number (digits not separated by commas), with each digit between 1 and 4 inclusive. The digits indicate the level, column, and row, respectively, of the move. You can win if you play correctly; although, it is considerably more difficult than standard, two-dimensional 3x3 tic-tac-toe.

This version of 3-D TIC-TAC-TOE is from Dartmouth College.

### Conversion notes

The AI code for TicTacToe2 depends quite heavily on the non-structured GOTO (I can almost hear Dijkstra now) and translation is quite challenging. This code relies very heavily on GOTOs that bind the code tightly together. Comments explain where that happens in the original.

There are at least two bugs from the original BASIC:

1. Code should only allow player to input valid 3D coordinates where every digit is between 1 and 4, but the original code allows any value between 111 and 444 (such as 297, for instance).
2. If the player moves first and the game ends in a draw, the original program will still prompt the player for a move instead of calling for a draw.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=168)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=183)


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


### Tic-Tac-Toe

The game of tic-tac-toe hardly needs any introduction. In this one, you play versus the computer. Moves are entered by number:
```
1   2   3

4   5   6

7   8   9
```

If you make any bad moves, the computer will win; if the computer makes a bad move, you can win; otherwise, the game ends in a tie.

A second version of the game is included which prints out the board after each move. This is ideally suited to a CRT terminal, particularly if you modify it to not print out a new board after each move, but rather use the cursor to make the move.

The first program was written by Tom Koos while a student researcher at the Oregon Museum of Science and Industry; it was extensively modified by Steve North of Creative Computing. The author of the second game is Curt Flick of Akron, Ohio.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=171)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=186)

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


README.md

Original source downloaded from Vintage Basic

Conversion to Rust


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Tower

This is a simulation of a game of logic that originated in the middle East. It is sometimes called Pharaoh's Needles, but its most common name is the Towers of Hanoi.

Legend has it that a secret society of monks live beneath the city of Hanoi. They possess three large towers or needles on which different size gold disks may be placed. Moving one at a time and never placing a large on a smaller disk, the monks endeavor to move the tower of disks from the left needle to the right needle. Legend says when they have finished moving this 64-disk tower, the world will end. How many moves will they have to make to accomplish this? If they can move 1 disk per minute and work 24 hours per day, how many years will it take?

In the computer puzzle you are faced with three upright needles. On the leftmost needle are placed from two to seven graduated disks, the largest being on bottom and smallest on top. Your object is to move the entire stack of disks to the rightmost needle. However, you many only move one disk at a time and you may never place a larger disk on top of a smaller one.

In this computer game, the disks are referred to by their size — i.e., the smallest is 3, next 5, 7, 9, 11, 13, and 15. If you play with fewer than 7 disks always use the largest, i.e. with 2 disks you would use nos. 13 and 15. The program instructions are self-explanatory. Good luck!

Charles Lund wrote this program while at the American School in the Hague, Netherlands.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=173)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=188)

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


### Train

TRAIN is a program which uses the computer to generate problems with random initial conditions to teach about the time-speed-distance relationship (distance = rate x time). You then input your answer and the computer verifies your response.

TRAIN is merely an example of a student-generated problem. Maximum fun (and benefit) comes more from _writing_ programs like this as opposed to solving the specific problem posed. Exchange your program with others—you solve their problem and let them solve yours.

TRAIN was originally written in FOCAL by one student for use by others in his class. It was submitted to us by Walt Koetke, Lexington High School, Lexington, Mass.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=175)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=190)

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


### Trap

This is another in the family of “guess the mystery number” games. In TRAP the computer selects a random number between 1 and 100 (or other limit set). Your object is to find the number. On each guess, you enter 2 numbers trying to trap the mystery number between your two trap numbers. The computer will tell you if you have trapped the number.

To win the game, you must guess the mystery number by entering it as the same value for both of your trap numbers. You get 6 guesses (this should be changed if you change the guessing limit).

After you have played GUESS, STARS, and TRAP, compare the guessing strategy you have found best for each game. Do you notice any similarities? What are the differences? Can you write a new guessing game with still another approach?

TRAP was suggested by a 10-year-old when he was playing GUESS. It was originally programmed by Steve Ullman and extensively modified into its final form by Bob Albrecht of People’s Computer Co.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=176)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=191)


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

Conversion to [Rust](https://www.rust-lang.org/) by Uğur Küpeli [ugurkupeli](https://github.com/ugurkupeli)

Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### 23 Matches

In the game of twenty-three matches, you start with 23 matches lying on a table. On each turn, you may take 1, 2, or 3 matches. You alternate moves with the computer and the one who has to take the last match loses.

The easiest way to devise a winning strategy is to start at the end of the game. Since your wish to leave the last match to your opponent, you would like to have either 4, 3, or 2 on your last turn you so can take away 3, 2, or 1 and leave 1. Consequently, you would like to leave your opponent with 5 on his next to last turn so, no matter what his move, you are left with 4, 3, or 2. Work this backwards to the beginning and you’ll find the game can effectively be won on the first move. Fortunately, the computer gives you the first move, so if you play wisely, you can win.

After you’ve mastered 23 Matches, move on to BATNUM and then to NUM.

This version of 23 Matches was originally written by Bob Albrecht of People’s Computer Company.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=177)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=192)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

There is an oddity (you can call it a bug, but it is no big deal) in the original code. If there are only two or three matches left at the player's turn and the player picks all of them (or more), the game would still register that as a win for the player.


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

Conversion to [rust](https://www.rust-lang.org/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### War

This program plays the card game of War. In War, the card deck is shuffled, then two cards are dealt, one to each player. Players compare cards and the higher card (numerically) wins. In case of a tie, no one wins. The game ends when you have gone through the whole deck (52 cards, 26 games) or when you decide to quit.

The computer gives cards by suit and number, for example, S-7 is the 7 of spades.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=178)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=193)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html


#### Porting Notes

(please note any difficulties or challenges in porting here)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


﻿Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html)

Converted to [D](https://dlang.org/) by [Bastiaan Veelo](https://github.com/veelo).

## Running the code

Assuming the reference [dmd](https://dlang.org/download.html#dmd) compiler:
```shell
dmd -preview=dip1000 -run war.d
```

[Other compilers](https://dlang.org/download.html) also exist.

## Specialties explained

This game code contains some specialties that you might want to know more about. Here goes.

### Suits

Most modern consoles are capable of displaying more than just ASCII, and so I have chosen to display the actual ♠, ♥, ♦
and ♣ instead of substituting them by letters like the BASIC original did. Only the Windows console needs a nudge in
the right direction with these instructions:
```d
SetConsoleOutputCP(CP_UTF8); // Set code page
SetConsoleOutputCP(GetACP);  // Restore the default
```
Instead of cluttering the `main()` function with these lesser important details, we can move them into a
[module constructor and module destructor](https://dlang.org/spec/module.html#staticorder), which run before and after
`main()` respectively. And because order of declaration is irrelevant in a D module, we can push those all the way
down to the bottom of the file. This is of course only necessary on Windows (and won't even work anywhere else) so
we'll need to wrap this in a `version (Windows)` conditional code block:
```d
version (Windows)
{
    import core.sys.windows.windows;

    shared static this() @trusted
    {
        SetConsoleOutputCP(CP_UTF8);
    }

    shared static ~this() @trusted
    {
        SetConsoleOutputCP(GetACP);
    }
}
```
Although it doesn't matter much in this single-threaded program, the `shared` attribute makes that these
constructors/destructors are run once per program invocation; non-shared module constructors and module destructors are
run for every thread. The `@trusted` annotation is necessary because these are system API calls; The compiler cannot
check these for memory-safety, and so we must indicate that we have reviewed the safety manually.

### Uniform Function Call Syntax

In case you wonder why this line works:
```d
if ("Do you want instructions?".yes)
    // ...
```
then it is because this is equivalent to
```d
if (yes("Do you want instructions?"))
    // ...
```
where `yes()` is a Boolean function that is defined below `main()`. This is made possible by the language feature that
is called [uniform function call syntax (UFCS)](https://dlang.org/spec/function.html#pseudo-member). UFCS works by
passing what is in front of the dot as the first parameter to the function, and it was invented to make it possible to
call free functions on objects as if they were member functions. UFCS can also be used to obtain a more natural order
of function calls, such as this line inside `yes()`:
```d
return trustedReadln.strip.toLower.startsWith("y");
```
which reads easier than the equivalent
```d
return startsWith(toLower(strip(trustedReadln())), "y");
```

### Type a lot or not?

It would have been straight forward to define the `cards` array explicitly like so:
```d
const cards = ["2♠", "2♥", "2♦", "2♣", "3♠", "3♥", "3♦", "3♣",
               "4♠", "4♥", "4♦", "4♣", "5♠", "5♥", "5♦", "5♣",
               "6♠", "6♥", "6♦", "6♣", "7♠", "7♥", "7♦", "7♣",
               "8♠", "8♥", "8♦", "8♣", "9♠", "9♥", "9♦", "9♣",
               "10♠", "10♥", "10♦", "10♣", "J♥", "J♦", "J♣", "J♣",
               "Q♠", "Q♥", "Q♦", "Q♣", "K♠", "K♥", "K♦", "K♣",
               "A♠", "A♥", "A♦", "A♣"];
```
but that's tedious, difficult to spot errors in (*can you?*) and looks like something a computer can automate. Indeed
it can:
```d
static const cards = cartesianProduct(["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"],
                                      ["♠", "♥", "♦", "♣"]).map!(a => a.expand.only.join).array;
```
The function [`cartesianProduct`](https://dlang.org/phobos/std_algorithm_setops.html#cartesianProduct) takes two
ranges, like the horizontal and vertical headers of a spreadsheet, and fills the table with the combinations that form
the coordinates of the cells. But the output of that function is in the form of an array of
[`Tuple`](https://dlang.org/phobos/std_typecons.html#Tuple)s, which looks like `[Tuple!(string, string)("2", "♠"),
Tuple!(string, string)("2", "♥"), ... etc]`. [`map`](https://dlang.org/phobos/std_algorithm_iteration.html#map)
comes to the rescue, converting each Tuple to a string, by calling
[`expand`](https://dlang.org/phobos/std_typecons.html#.Tuple.expand), then
[`only`](https://dlang.org/phobos/std_range.html#only) and then [`join`](https://dlang.org/phobos/std_array.html#join)
on them. The result is a lazily evaluated range of strings. Finally,
[`array`](https://dlang.org/phobos/std_array.html#array) turns the range into a random access array. The `static`
attribute makes that all this is performed at compile-time, so the result is exactly the same as the manually entered
data, but without the typo's.

### Shuffle the cards or not?

The original BASIC code works with a constant array of cards, ordered by increasing numerical value, and indexing it
with indices that have been shuffled. This is efficient because in comparing who wins, the indices can be compared
directly, since a higher index correlates to a card with a higher numerical value (when divided by the number of suits,
4). Some of the other reimplementations in other languages have been written in a lesser efficient way by shuffling the
array of cards itself. This then requires the use of a lookup table or searching for equality in an auxiliary array
when comparing cards.

I find the original more elegant, so that's what you see here:
```d
const indices = iota(0, cards.length).array.randomShuffle;
```
[`iota`](https://dlang.org/phobos/std_range.html#iota) produces a range of integers, in this case starting at 0 and
increasing up to the number of cards in the deck (exclusive). [`array`](https://dlang.org/phobos/std_array.html#array)
turns the range into an array, so that [`randomShuffle`](https://dlang.org/phobos/std_random.html#randomShuffle) can
do its work.


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


### Weekday

This program gives facts about your date of birth (or some other day of interest). It is not prepared to give information on people born before the use of the current type of calendar, i.e. year 1582.

You merely enter today’s date in the form—month, day, year and your date of birth in the same form. The computer then tells you the day of the week of your birth date, your age, and how much time you have spent sleeping, eating, working, and relaxing.

This program was adapted from a GE timesharing program by Tom Kloos at the Oregon Museum of Science and Industry.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=179)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=194)

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

I have replaced the manual date logic with Perl built-ins to the extent
possible. Unfortunately the kind of date math involved in the "time
spent doing ..." functionality is not well-defined, so I have been
forced to retain the original logic here. Sigh.

You can use any punctuation character you please in the date
input. So something like 2/29/2020 is perfectly acceptable.

It would also have been nice to produce a localized version that
supports day/month/year or year-month-day input, but that didn't happen.

Also nice would have been language-specific output -- especially if it
could have accommodated regional differences in which day of the week or
month is unlucky.

Tom Wyant


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by Anthony Rubick [AnthonyMichaelTDM](https://github.com/AnthonyMichaelTDM)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Word

WORD is a combination of HANGMAN and BAGELS. In this game, the player must guess a word with clues as to a letter position furnished by the computer. However, instead of guessing one letter at a time, in WORD you guess an entire word (or group of 5 letters, such as ABCDE). The computer will tell you if any letters that you have guessed are in the mystery word and if any of them are in the correct position. Armed with these clues, you go on guessing until you get the word or, if you can’t get it, input a “?” and the computer will tell you the mystery word.

You may change the words in Data Statements, but they must be 5-letter words.

The author of this program is Charles Reid of Lexington High School, Lexington, Massachusetts.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=181)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=194)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html)

Converted to [D](https://dlang.org/) by [Bastiaan Veelo](https://github.com/veelo).

The Basic original required words to be exactly five letters in length for the program to behave correctly.
This version does not replicate that limitation, and the test for that requirement is commented out.

## Running the code

Assuming the reference [dmd](https://dlang.org/download.html#dmd) compiler:
```shell
dmd -dip1000 -run word.d
```

[Other compilers](https://dlang.org/download.html) also exist.


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
