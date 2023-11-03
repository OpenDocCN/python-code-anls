# BasicComputerGames源码解析 0

# How to run the games

The games in this repository have been translated into a number of different languages. How to run them depends on the target language.

## csharp

### dotnet command-line

The best cross-platform method for running the csharp examples is with the `dotnet` command-line tool. This can be downloaded for **MacOS**, **Windows** and **Linux** from [dotnet.microsoft.com](https://dotnet.microsoft.com/).

From there, the program can be run by

1. Opening a terminal window
1. Navigating to the corresponding directory
1. Starting with `dotnet run`

### Visual Studio

Alternatively, for non-dotnet compatible translations, you will need [Visual Studio](https://visualstudio.microsoft.com/vs/community/) which can be used to both open the project and run the example.

1. Open the corresponding `.csproj` or `.sln` file
1. Click `Run` from within the Visual Studio IDE

## java

The Java translations can be run via the command line or from an IDE such as [Eclipse](https://www.eclipse.org/downloads/packages/release/kepler/sr1/eclipse-ide-java-developers) or [IntelliJ](https://www.jetbrains.com/idea/)

To run from the command line, you will need a Java SDK (eg. [Oracle JDK](https://www.oracle.com/java/technologies/downloads/) or [Open JDK](https://openjdk.java.net/)).

1. Navigate to the corresponding directory.
1. Compile the program with `javac`:
   * eg. `javac AceyDuceyGame.java`
1. Run the compiled program with `java`:
   * eg. `java AceyDuceyGame`

or if you are **using JDK11 or later** you can now execute a self contained java file that has a main method directly with `java <filename>.java`.

## javascript

There are two ways of javascript implementations:

### browser

The html examples can be run from within your web browser. Simply open the corresponding `.html` file from your web browser.

### node.js

Some games are implemented as a [node.js](https://nodejs.org/) script. In this case there is no `*.html` file in the folder.

1. [install node.js](https://nodejs.org/en/download/) for your system.
1. change directory to the root of this repository (e.g. `cd basic-computer-games`).
1. from a terminal call the script you want to run (e.g. `node 78_Sine_Wave/javascript/sinewave.mjs`).

_Hint: Normally javascript files have a `*.js` extension. We are using `*.mjs` to let node know , that we are using [ES modules](https://nodejs.org/docs/latest/api/esm.html#modules-ecmascript-modules) instead of [CommonJS](https://nodejs.org/docs/latest/api/modules.html#modules-commonjs-modules)._

## kotlin

Kotlin programs are compiled with the Kotlin compiler, and run with the java runtime, just like java programs.
In addition to the java runtime you will need the `kotlinc` compiler, which can be installed using [these instructions](https://kotlinlang.org/docs/command-line.html).

1. Navigate to the corresponding directory.
1. Compile the program with `kotlinc`:
   * eg. `kotlinc AceyDuceyGame.kt -include-runtime -d AceyDuceyGame.jar`
1. Run the compiled program with `java`:
   * eg. `java -jar AceyDuceyGame.jar`

## pascal

The pascal examples can be run using [Free Pascal](https://www.freepascal.org/). Additionally, `.lsi` project files can be opened with the [Lazarus Project IDE](https://www.lazarus-ide.org/).

The pascal examples include both *simple* (single-file) and *object-oriented* (in the `/object-pascal`directories) examples.

1. You can compile the program from the command line with the `fpc` command.
   * eg. `fpc amazing.pas`
1. The output is an executable file that can be run directly.

## perl

The perl translations can be run using a perl interpreter (a copy can be downloaded from [perl.org](https://www.perl.org/)) if not already installed.

1. From the command-line, navigate to the corresponding directory.
1. Invoke with the `perl` command.
   * eg. `perl aceyducey.pl`

## python

The python translations can be run from the command line by using the `py` interpreter. If not already installed, a copy can be downloaded from [python.org](https://www.python.org/downloads/) for **Windows**, **MacOS** and **Linux**.

1. From the command-line, navigate to the corresponding directory.
1. Invoke with the `py` or `python` interpreter (depending on your python version).
   * eg. `py acey_ducey_oo.py`
   * eg. `python aceyducey.py`

**Note**

Some translations include multiple versions for python, such as `acey ducey` which features versions for Python 2 (`aceyducey.py`) and Python 3 (`acey_ducey.py`) as well as an extra object-oriented version (`acey_ducey_oo.py`).

You can manage and use different versions of python with [pip](https://pypi.org/project/pip/).

## ruby

If you don't already have a ruby interpreter, you can download it from the [ruby project site](https://www.ruby-lang.org/en/).

1. From the command-line, navigate to the corresponding directory.
1. Invoke with the `ruby` tool.
   * eg. `ruby aceyducey.rb`

## vbnet

Follow the same steps as for the [csharp](#csharp) translations. This can be run with `dotnet` or `Visual Studio`.

## rust

If you don't already have Rust on your computer, you can follow the instruction on [Rust Book](https://doc.rust-lang.org/book/ch01-01-installation.html)

1. From the command-line, navigate to the corresponding directory.
2. Run the following command.
   * `cargo run`


### What are we doing?

We’re updating the first million selling computer book, [BASIC Computer Games](https://en.wikipedia.org/wiki/BASIC_Computer_Games), for 2022 and beyond!

- [Read the original book](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf) (pdf)
- [Play the original games in your browser](https://troypress.com/wp-content/uploads/user/js-basic/index.html)

### Where can we discuss it?

Please see [the discussion here](https://discourse.codinghorror.com/t/-/7927) for a worklog and conversation around this project.

### Project structure

I have moved all [the original BASIC source code](http://www.vintage-basic.net/games.html) into a folder for each project in the original book (first volume). Note that Lyle Kopnicky has generously normalized all the code (thanks Lyle!) to run against [Vintage Basic](http://www.vintage-basic.net/download.html) circa 2009:

> I’ve included all the games here for your tinkering pleasure. I’ve tested and tweaked each one of them to make sure they’ll run with Vintage BASIC, though you may see a few oddities. That’s part of the fun of playing with BASIC: it never works quite the same on two machines. The games will play better if you keep CAPS LOCK on, as they were designed to be used with capital-letter input.

Each project has subfolders corresponding to the languages we’d like to see the games ported to. This is based on the [2022 TIOBE index of top languages](https://www.tiobe.com/tiobe-index/) that are _**memory safe**_ and _**general purpose scripting languages**_ per [this post](https://discourse.codinghorror.com/t/-/7927/34):

1. C# 
2. Java
3. JavaScript
4. Kotlin
5. Lua
6. Perl
7. Python
8. Ruby
9. Rust
10. VB.NET

> 📢 Note that in March 2022 we removed Pascal / Object Pascal and replaced it with Rust as we couldn’t determine if Pascal is effectively memory safe. We’ve also added Lua, as it made the top 20 in TIOBE (as of 2022) and it is both memory safe and a scripting language. The Pascal ports were moved to the alternate languages folder.

> ⚠️ Please note that we have decided, as a project, that we **do not want any IDE-specific or build-specific files in the repository.** Please refrain from committing any files to the repository that only exist to work with a specific IDE or a specific build system.

### Alternate Languages

If you wish to port one of the programs to a language not in our list – that is, a language which is either not memory safe, or not a general purpose scripting language, you can do so via the `00_Alternate_Languages` folder. Place your port in the appropriate game subfolder, in a subfolder named for the language. Please note that these ports are appreciated, but they will not count toward the donation total at the end of the project.

### Project goals

Feel free to begin converting these classic games into the above list of modern, memory safe languages. In fact, courtesy of @mojoaxel, you can even view the JavaScript versions in your web browser at

https://coding-horror.github.io/basic-computer-games/

But first, a few guidelines:

- **These are very old games**. They date from the mid-70s so they’re not exactly examples of what kids (or anyone, really?) would be playing these days. Consider them more like classic programming exercises to teach programming.  We’re paying it forward by converting them into modern languages, so the next generation can learn from the programs in this classic book – and compare implementations across common modern languages.

- **Stay true to the original program**. These are mostly unsophisticated, simple command line / console games, so we should strive to replicate the command line / console output and behavior illustrated in the original book. See the README in the project folder for links to the original scanned source input and output. Try [running the game in your browser](https://troypress.com/wp-content/uploads/user/js-basic/index.html). Avoid the impulse to add features; keep it simple, _except_ for modern conventions, see next item 👇

- **Please DO update for modern coding conventions**. Support uppercase and lowercase. Use structured programming. Use subroutines. Try to be an example of good, modern coding practices!

- **Use lots of comments to explain what is going on**. Comment liberally! If there were clever tricks in the original code, decompose those tricks into simpler (even if more verbose) code, and use comments to explain what’s happening and why. If there is something particularly tricky about a program, edit the **Porting Notes** section of the `readme.md` to let everyone know. Those `GOTO`s can be very pesky..

- **Please don’t get _too_ fancy**. Definitely use the most recent versions and features of the target language, but also try to keep the code samples simple and explainable – the goal is to teach programming in the target language, not necessarily demonstrate the cleverest one-line tricks, or big system "enterprise" coding techniques designed for thousands of lines of code.

- **Please don't check in any build specific or IDE specific files**. We want the repository to be simple and clean, so we have ruled out including any IDE or build system specific files from the repository. Git related files are OK, as we are using Git and this is GitHub. 😉

### Emulation and Bugfixes

We want the general behavior of the original programs to be preserved, _however_, we also want to update them, specifically:

- allow both UPPERCASE and lowercase input and display
- incorporate any bugfixes to the original programs; see the `readme.md` in the game folder
- improved error handling for bad or erroneous input

Please note that on the back of the Basic Computer Games book it says **Microsoft 8K Basic, Rev 4.0 was the version David Ahl used to test**, so that is the level of compatibility we are looking for.  QBasic on the DOS emulation is a later version of Basic but one that retains downwards compatibility so far in our testing. To verify behavior, try [running the programs in your browser](https://troypress.com/wp-content/uploads/user/js-basic/index.html) with [JS BASIC, effectively Applesoft BASIC](https://github.com/inexorabletash/jsbasic/).

### Have fun!

Thank you for taking part in this project to update a classic programming book – one of the most influential programming books in computing history – for 2022 and beyond!

NOTE: per [the official blog post announcement](https://blog.codinghorror.com/updating-the-single-most-influential-book-of-the-basic-era/), I will be **donating $5 for each contributed program in the 10 agreed upon languages to [Girls Who Code](https://girlswhocode.com/)**.

### Current Progress

<details><summary>toggle for game by language table</summary>

| Name                   | csharp | java | javascript | kotlin | lua | perl | python | ruby | rust | vbnet |
| ---------------------- | ------ | ---- | ---------- | ------ | --- | ---- | ------ | ---- | ---- | ----- |
| 01_Acey_Ducey          | x      | x    | x          | x      | x   | x    | x      | x    | x    | x     |
| 02_Amazing             | x      | x    | x          |        |     | x    | x      | x    | x    | x     |
| 03_Animal              | x      | x    | x          | x      | x   | x    | x      | x    | x    | x     |
| 04_Awari               | x      | x    | x          |        |     | x    | x      | x    | x    | x     |
| 05_Bagels              | x      | x    | x          | x      | x   | x    | x      | x    | x    | x     |
| 06_Banner              | x      | x    | x          |        |     | x    | x      | x    |      | x     |
| 07_Basketball          | x      | x    | x          |        |     | x    | x      | x    |      | x     |
| 08_Batnum              | x      | x    | x          |        |     | x    | x      | x    |      | x     |
| 09_Battle              | x      | x    | x          |        |     |      | x      |      |      | x     |
| 10_Blackjack           | x      | x    | x          |        |     |      | x      | x    | x    | x     |
| 11_Bombardment         | x      | x    | x          |        |     | x    | x      | x    | x    | x     |
| 12_Bombs_Away          | x      | x    | x          |        | x   | x    | x      |      |      | x     |
| 13_Bounce              | x      | x    | x          |        |     | x    | x      | x    |      | x     |
| 14_Bowling             | x      | x    | x          |        |     | x    | x      |      |      | x     |
| 15_Boxing              | x      | x    | x          |        |     | x    | x      |      |      | x     |
| 16_Bug                 | x      | x    | x          |        |     |      | x      | x    |      | x     |
| 17_Bullfight           | x      |      | x          | x      |     |      | x      |      |      | x     |
| 18_Bullseye            | x      | x    | x          |        |     | x    | x      |      | x    | x     |
| 19_Bunny               | x      | x    | x          |        |     | x    | x      | x    |      | x     |
| 20_Buzzword            | x      | x    | x          |        | x   | x    | x      | x    | x    | x     |
| 21_Calendar            | x      | x    | x          |        |     | x    | x      | x    | x    | x     |
| 22_Change              | x      | x    | x          |        |     | x    | x      |      | x    | x     |
| 23_Checkers            | x      |      | x          |        |     | x    | x      | x    |      | x     |
| 24_Chemist             | x      | x    | x          |        |     | x    | x      |      | x    | x     |
| 25_Chief               | x      | x    | x          |        | x   | x    | x      | x    |      | x     |
| 26_Chomp               | x      | x    | x          |        |     | x    | x      |      |      | x     |
| 27_Civil_War           | x      | x    | x          |        |     |      | x      |      |      | x     |
| 28_Combat              | x      | x    | x          |        |     | x    | x      |      |      | x     |
| 29_Craps               | x      | x    | x          |        | x   | x    | x      | x    | x    | x     |
| 30_Cube                | x      | x    | x          |        |     |      | x      | x    | x    | x     |
| 31_Depth_Charge        | x      | x    | x          |        |     | x    | x      | x    |      | x     |
| 32_Diamond             | x      | x    | x          | x      |     | x    | x      | x    | x    | x     |
| 33_Dice                | x      | x    | x          |        | x   | x    | x      | x    | x    | x     |
| 34_Digits              | x      | x    | x          |        |     | x    | x      |      |      | x     |
| 35_Even_Wins           | x      |      | x          |        |     | x    | x      |      | x    | x     |
| 36_Flip_Flop           | x      | x    | x          |        |     | x    | x      | x    | x    | x     |
| 37_Football            | x      |      | x          |        |     |      | x      |      |      | x     |
| 38_Fur_Trader          | x      | x    | x          |        |     | x    | x      |      |      | x     |
| 39_Golf                | x      |      | x          |        |     |      | x      |      |      | x     |
| 40_Gomoko              | x      | x    | x          |        |     | x    | x      |      |      | x     |
| 41_Guess               | x      | x    | x          |        |     | x    | x      | x    | x    | x     |
| 42_Gunner              | x      | x    | x          |        |     | x    | x      |      |      | x     |
| 43_Hammurabi           | x      | x    | x          |        |     |      | x      |      |      | x     |
| 44_Hangman             | x      | x    | x          |        |     | x    | x      | x    |      | x     |
| 45_Hello               | x      | x    | x          |        | x   | x    | x      | x    |      | x     |
| 46_Hexapawn            | x      |      |            |        |     |      | x      |      |      | x     |
| 47_Hi-Lo               | x      |      | x          | x      | x   | x    | x      | x    | x    | x     |
| 48_High_IQ             | x      | x    | x          |        |     |      | x      |      |      | x     |
| 49_Hockey              | x      |      | x          |        |     |      | x      |      |      | x     |
| 50_Horserace           | x      |      | x          |        |     |      |        |      | x    | x     |
| 51_Hurkle              | x      | x    | x          |        |     | x    | x      | x    | x    | x     |
| 52_Kinema              | x      | x    | x          |        |     | x    | x      | x    |      | x     |
| 53_King                | x      |      | x          |        |     |      | x      |      | x    | x     |
| 54_Letter              | x      | x    | x          |        |     | x    | x      | x    | x    | x     |
| 55_Life                | x      | x    | x          |        |     | x    | x      | x    | x    | x     |
| 56_Life_for_Two        | x      | x    | x          |        |     | x    | x      |      |      | x     |
| 57_Literature_Quiz     | x      | x    | x          |        |     | x    | x      |      | x    | x     |
| 58_Love                | x      | x    | x          |        |     | x    | x      | x    |      | x     |
| 59_Lunar_LEM_Rocket    | x      |      | x          |        |     |      | x      |      | x    | x     |
| 60_Mastermind          | x      | x    | x          |        |     | x    | x      |      | x    | x     |
| 61_Math_Dice           | x      | x    | x          |        |     | x    | x      | x    | x    | x     |
| 62_Mugwump             | x      | x    | x          |        |     | x    | x      |      | x    | x     |
| 63_Name                | x      | x    | x          | x      |     | x    | x      | x    |      | x     |
| 64_Nicomachus          | x      | x    | x          |        |     | x    | x      |      | x    | x     |
| 65_Nim                 | x      |      | x          |        |     |      | x      | x    | x    | x     |
| 66_Number              | x      | x    | x          |        |     | x    | x      |      | x    | x     |
| 67_One_Check           | x      | x    | x          |        |     | x    | x      |      |      | x     |
| 68_Orbit               | x      | x    | x          |        |     | x    | x      | x    | x    | x     |
| 69_Pizza               | x      | x    | x          |        |     | x    | x      | x    |      | x     |
| 70_Poetry              | x      | x    | x          |        |     | x    | x      | x    |      | x     |
| 71_Poker               | x      | x    | x          |        |     |      |        |      |      | x     |
| 72_Queen               | x      |      | x          |        |     | x    | x      |      | x    | x     |
| 73_Reverse             | x      | x    | x          |        |     | x    | x      | x    |      | x     |
| 74_Rock_Scissors_Paper | x      | x    | x          | x      |     | x    | x      | x    | x    | x     |
| 75_Roulette            | x      | x    | x          |        |     | x    | x      |      | x    | x     |
| 76_Russian_Roulette    | x      | x    | x          | x      |     | x    | x      | x    | x    | x     |
| 77_Salvo               | x      |      | x          |        |     |      | x      |      |      | x     |
| 78_Sine_Wave           | x      | x    | x          | x      |     | x    | x      | x    | x    | x     |
| 79_Slalom              | x      |      | x          |        |     |      | x      |      |      | x     |
| 80_Slots               | x      | x    | x          |        |     | x    | x      | x    |      | x     |
| 81_Splat               | x      | x    | x          |        |     | x    | x      |      | x    | x     |
| 82_Stars               | x      | x    | x          |        |     | x    | x      | x    | x    | x     |
| 83_Stock_Market        | x      | x    | x          |        |     |      | x      |      |      | x     |
| 84_Super_Star_Trek     | x      | x    | x          |        |     |      | x      |      | x    | x     |
| 85_Synonym             | x      | x    | x          |        |     | x    | x      | x    |      | x     |
| 86_Target              | x      | x    | x          |        |     | x    | x      |      |      | x     |
| 87_3-D_Plot            | x      | x    | x          |        |     | x    | x      | x    |      | x     |
| 88_3-D_Tic-Tac-Toe     | x      |      | x          |        |     |      | x      |      |      | x     |
| 89_Tic-Tac-Toe         | x      | x    | x          | x      |     | x    | x      |      | x    | x     |
| 90_Tower               | x      | x    | x          |        |     | x    | x      |      | x    | x     |
| 91_Train               | x      | x    | x          |        |     | x    | x      | x    | x    | x     |
| 92_Trap                | x      | x    | x          |        |     | x    | x      | x    | x    | x     |
| 93_23_Matches          | x      | x    | x          |        |     | x    | x      | x    | x    | x     |
| 94_War                 | x      | x    | x          | x      |     | x    | x      | x    | x    | x     |
| 95_Weekday             | x      | x    | x          |        |     | x    | x      |      | x    | x     |
| 96_Word                | x      | x    | x          |        |     | x    | x      | x    | x    | x     |

</details>


#### Alternate Languages

This folder contains implementations of each program in alternate languages which are _not_ one of the agreed upon 10 languages.

Implementations here are NOT bound to these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

So for example, (here only) C or PASCAL are allowed. Please still remain faithful to original look-and-feel (console applications).
Try to keep your code portable (unless it is not possible, and then be very explicit about this limitation in your
README and your folder naming).

We welcome additional ports in whatever language you prefer, but these additional ports are for educational purposes only, and do not count towards the donation total at the end of the project.


Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

#### External Links
 - Common Lisp: https://github.com/koalahedron/lisp-computer-games/blob/master/01%20Acey%20Ducey/common-lisp/acey-deucy.lisp
 - PowerShell: https://github.com/eweilnau/basic-computer-games-powershell/blob/main/AceyDucey.ps1


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [C++14](https://en.wikipedia.org/wiki/C%2B%2B14)

The build folder are executables for x86 and x64 systems. Compiled and built using Visual Studio.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html)

Converted to [D](https://dlang.org/) by [Bastiaan Veelo](https://github.com/veelo).

Two versions are supplied that are functionally equivalent, but differ in source layout:

<dl>
  <dt><tt>aceyducey_literal.d</tt></dt>
  <dd>A largely literal transcription of the original Basic source. All unnecessary uglyness is preserved.</dd>
  <dt><tt>aceyducey.d</tt></dt>
  <dd>An idiomatic D refactoring of the original, with a focus on increasing the readability and robustness.
      Memory-safety <A href="https://dlang.org/spec/memory-safe-d.html">is ensured by the language</a>, thanks to the
      <tt>@safe</tt> annotation.</dd>
</dl>

## Running the code

Assuming the reference [dmd](https://dlang.org/download.html#dmd) compiler:
```shell
dmd -run aceyducey.d
```

[Other compilers](https://dlang.org/download.html) also exist.

Note that there are compiler switches related to memory-safety (`-preview=dip25` and `-preview=dip1000`) that are not
used here because they are unnecessary in this case. What these do is to make the analysis more thorough, so that with
them some code that needed to be `@system` can then be inferred to be in fact `@safe`. [Code that compiles without
these switches is just as safe as when compiled with them]
(https://forum.dlang.org/post/dftgjalswvwfjpyushgn@forum.dlang.org).


# Acey Ducey

This is an Elm implementation of the `Basic Compouter Games` Game Acey Ducey.

## Build App

- install elm

```bash
yarn
yarn build
```


Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript aceyducey.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "aceyducey"
	run
```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Pascal](https://en.wikipedia.org/wiki/Pascal_(programming_language)) by Gustavo Carreno [gcarreno@github](https://github.com/gcarreno)


Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript amazing.ms
```
Note that because this program imports "listUtil", you will need to have a the standard MiniScript libraries somewhere in your import path.

2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "amazing"
	run
```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Pascal](https://en.wikipedia.org/wiki/Pascal_(programming_language)) by Gustavo Carreno [gcarreno@github](https://github.com/gcarreno)


Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript animal.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "animal"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

# Awari

This is an Elm implementation of the `Basic Compouter Games` Game Awari.

## Build App

- install elm

```bash
yarn
yarn build
```


Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript awari.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "awari"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript bagels.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "bagels"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript banner.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "banner"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript basketball.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "basketball"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript batnum.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "batnum"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

NOTE: One feature has been added to the original game.  At the "??" prompt, instead of entering coordinates, you can enter "?" (a question mark) to reprint the fleet disposition code.

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript battle.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "battle"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript blackjack.ms
```
But note that the current release (1.2.1) of command-line MiniScript does not properly flush the output buffer when line breaks are suppressed, as this program does when prompting for your next action after a Hit.  So, method 2 (below) is recommended for now.

2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "blackjack"
	run
```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Pascal](https://en.wikipedia.org/wiki/Pascal_(programming_language))


Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript bombardment.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "bombardment"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript bombsaway.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "bombsaway"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript bounce.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "bounce"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript bowling.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "bowling"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript boxing.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "boxing"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript bug.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "bug"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript bull.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "bull"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

0. Try-It! Page:
Go to https://miniscript.org/tryit/, clear the sample code from the code editor, and paste in the contents of bullseye.ms.  Then click the "Run Script" button.  Program output (and input) will appear in the green-on-black terminal display to the right of or below the code editor.

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript bullseye.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "bullseye"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

0. Try-It! Page:
Go to https://miniscript.org/tryit/, clear the sample code from the code editor, and paste in the contents of bunny.ms.  Then click the "Run Script" button.  Program output (and input) will appear in the green-on-black terminal display to the right of or below the code editor.

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript bunny.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "bunny"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

0. Try-It! Page:
Go to https://miniscript.org/tryit/, clear the sample code from the code editor, and paste in the contents of bunny.ms.  Then click the "Run Script" button.  Program output (and input) will appear in the green-on-black terminal display to the right of or below the code editor.

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript bunny.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "bunny"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript calendar.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "calendar"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

0. Try-It! Page:
Go to https://miniscript.org/tryit/, clear the sample code from the code editor, and paste in the contents of change.ms.  Then click the "Run Script" button.  Program output (and input) will appear in the green-on-black terminal display to the right of or below the code editor.

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript change.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "change"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript checkers.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "checkers"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

0. Try-It! Page:
Go to https://miniscript.org/tryit/, clear the sample code from the code editor, and paste in the contents of chemist.ms.  Then click the "Run Script" button.  Program output (and input) will appear in the green-on-black terminal display to the right of or below the code editor.

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript chemist.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "chemist"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

NOTE: I have added `wait` statements before and while printing the lightning bolt, without which it appears too quickly to be properly dramatic.

Ways to play:

0. Try-It! Page:
Go to https://miniscript.org/tryit/, clear the sample code from the code editor, and paste in the contents of chief.ms.  Then click the "Run Script" button.  Program output (and input) will appear in the green-on-black terminal display to the right of or below the code editor.

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript chief.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "chief"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript chomp.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "chomp"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript civilwar.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "civilwar"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript combat.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "combat"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript craps.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "craps"
	run
```

#### External Links
 - Common Lisp: https://github.com/koalahedron/lisp-computer-games/blob/master/01%20Acey%20Ducey/common-lisp/acey-deucy.lisp
 - PowerShell: https://github.com/eweilnau/basic-computer-games-powershell/blob/main/AceyDucey.ps1


Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript cube.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "cube"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript depthcharge.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "depthcharge.ms"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript diamond.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "diamond"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.julialang.org/)

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript dice.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "dice"
	run
```
3. "Try-It!" page on the web:
Go to https://miniscript.org/tryit/, clear the default program from the source code editor, paste in the contents of dice.ms, and click the "Run Script" button.


Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript digits.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "digits"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Note that this folder (like the original BASIC programs) contains TWO different programs based on the same idea.  evenwins.ms plays deterministically; gameofevenwins.ms learns from its failures over multiple games.

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript evenwins.ms
```
or

```
	miniscript gameofevenwins.ms
```

2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "evenwins"
	run
```
or

```
	load "gameofevenwins"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.