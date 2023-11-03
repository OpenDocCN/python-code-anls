# BasicComputerGames源码解析 2

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
	miniscript synonym.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:
```
	load "synonym"
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
	miniscript target.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:
```
	load "target"
	run
```


Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

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


Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript number.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "number"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Pascal](https://en.wikipedia.org/wiki/Pascal_(programming_language))


Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

	miniscript tictactoe.ms

2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

	load "tictactoe"
	run

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

	miniscript tower.ms

2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

	load "tower"
	run

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
	miniscript train.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "train"
	run
```
3. "Try-It!" page on the web:
Go to https://miniscript.org/tryit/, clear the default program from the source code editor, paste in the contents of train.ms, and click the "Run Script" button.


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
	miniscript trap.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "trap"
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

	miniscript 23matches.ms

2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

	load "23matches"
	run

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

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


Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript war.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "war"
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
	miniscript weekday.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:
```
	load "weekday"
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
	miniscript word.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:
```
	load "word"
	run
```

# Common Library

## Purpose

The primary purpose of this library is to implement common behaviours of the BASIC interpreter that impact gameplay, to
free coders porting the games to concentrate on the explicit game logic.

The behaviours implemented by this library are:

* Complex interactions involved in text input.
* Formatting of number in text output.
* Behaviour of the BASIC `RND(float)` PRNG function.

A secondary purpose is to provide common services that, with dependency injection, would allow a ported game to be
driven programmatically to permit full-game acceptance tests to be written.

The library is **NOT** intended to be:

* a repository for common game logic that is implemented in the BASIC code of the games; or
* a DSL allowing the BASIC code to be compiled in a different language environment with minimal changes. This implies
  that implementations of the above behaviours should use method names, etc, that are idiomatic of the specific
  language's normal routines for that behaviour.

## Text Input

The behaviour of the BASIC interpreter when accepting text input from the user is a major element of the original
gameplay experience that is seen to be valuable to maintain. This behaviour is complex and non-trivial to implement, and
is better implemented once for other developers to use so they can concentrate on the explicit game logic.

The text input/output behaviour can be investigated using a basic program such as:

**`BASIC_Tests/InputTest.bas`**

```basic
10 INPUT "Enter 3 numbers";A,B,C
20 PRINT "You entered: ";A;B;C
30 PRINT "--------------------------"
40 GOTO 10
```

The following transcript shows the use of this program, and some interesting behaviours of the BASIC interpreter INPUT
routine. There are some other behaviours which can be seen in the unit tests for the C# library implementation.

```dos
Enter 3 numbers? -1,2,3.141             <-- multiple numbers are separated by commas
You entered: -1  2  3.141
--------------------------
Enter 3 numbers? 1                      <-- ... or entered on separate lines
?? 2
?? 3
You entered:  1  2  3
--------------------------
Enter 3 numbers? 1,2                    <-- ... or both
?? 3
You entered:  1  2  3
--------------------------
Enter 3 numbers? 1,-2,3,4               <-- Extra input is ignore with a warning
!EXTRA INPUT IGNORED
You entered:  1 -2  3
--------------------------
Enter 3 numbers?   5  , 6.7, -8   ,10   <-- Whitespace around values is ignored
!EXTRA INPUT IGNORED
You entered:  5  6.7 -8
--------------------------
Enter 3 numbers? abcd,e,f               <-- Non-numeric entries must be retried
!NUMBER EXPECTED - RETRY INPUT LINE
? 1,2,abc                               <-- A single non-numeric invalidates the whole line
!NUMBER EXPECTED - RETRY INPUT LINE
? 1de,2.3f,10k,abcde                    <-- ... except for trailing non-digit chars  and extra input
!EXTRA INPUT IGNORED
You entered:  1  2.3  10
--------------------------
Enter 3 numbers? 1,"2,3",4              <-- Double-quotes enclose a single parsed value
You entered:  1  2  4
--------------------------
Enter 3 numbers? 1,2,"3                 <-- An unmatched double-quote crashes the interpreter
vintbas.exe: Mismatched inputbuf in inputVars
CallStack (from HasCallStack):
  error, called at src\Language\VintageBasic\Interpreter.hs:436:21 in main:Language.VintageBasic.Interpreter
```

I propose to ignore this last behaviour - the interpreter crash - and instead treat the end of the input line as the end
of a quoted value.  There are some additional behaviours to those shown above which can be seen in the unit tests for
the C# implementation of the library.

Note also that BASIC numeric variables store a single-precision floating point value, so numeric input functions should
return a value of that type.

### Implementation Notes

The usage of the `INPUT` command in the BASIC code of the games was analysed, with the variables used designated as
`num`, for a numeric variable (eg. `M1`), or `str`, for a string variable (eg. `C$`). The result of this usage analysis
across all game programs is:

Variable number and type|Count
---|---
str|137
str  str|2
num|187
num  num|27
num  num  num|7
num  num  num  num|1
num  num  num  num  num  num  num  num  num  num|1

The usage count is interesting, but not important. What is important is the variable type and number for each usage.
Implementers if the above behaviours do not need to cater for mixed variable types in their input routines (although the
BASIC interpreter does support this). Input routines also need to cater only for 1 or 2 string values, and 1, 2, 3, 4,
or 10 numeric values.

## Numbers in Text Output

As seen in the transcript above, the BASIC interpreter has some particular rules for formatting numbers in text output.
Negative numbers are printed with a leading negative sign (`-`) and a trailing space. Positive numbers are printed also
with the trailing space, but with a leading space in place of the sign character.

Additional formatting rules can be seen by running this program:

**`BASIC_Tests/OutputTest.bas`**

```basic
10 A=1: B=-2: C=0.7: D=123456789: E=-0.0000000001
20 PRINT "|";A;"|";B;"|";C;"|";D;"|";E;"|"
```

The output is:

```dos
| 1 |-2 | .7 | 1.2345679E+8 |-1.E-10 |
```

This clearly shows the leading and trailing spaces, but also shows that:

* numbers without an integer component are printed without a leading zero to the left of the decimal place;
* numbers above and below a certain magnitude are printed in scientific notation.

<!-- markdownlint-disable MD024 -->
### Implementation Notes
<!-- markdownlint-enable MD024 -->

I think the important piece of this number formatting behaviour, in terms of its impact on replicating the text output
of the original games, is the leading and trailing spaces. This should be the minimum behaviour supported for numeric
output. Additional formatting behaviours may be investigated and supported by library implementers as they choose.

## The BASIC `RND(float)` function

The [Vintage BASIC documentation](http://vintage-basic.net/downloads/Vintage_BASIC_Users_Guide.html) describes the
behaviour of the `RND(float)` function:

> Psuedorandom number generator. The behavior is different depending on the value passed. If the value is positive, the
> result will be a new random value between 0 and 1 (including 0 but not 1). If the value is zero, the result will be a
> repeat of the last random number generated. If the value is negative, it will be rounded down to the nearest integer
> and used to reseed the random number generator. Pseudorandom sequences can be repeated by reseeding with the same
> number.

This behaviour can be shown by the program:

**`BASIC_Tests/RndTest.bas`**

```basic
10 PRINT "1: ";RND(1);RND(1);RND(0);RND(0);RND(1)
20 PRINT "2: ";RND(-2);RND(1);RND(1);RND(1)
30 PRINT "3: ";RND(-5);RND(1);RND(1);RND(1)
40 PRINT "4: ";RND(-2);RND(1);RND(1);RND(1)
```

The output of this program is:

```dos
1:  .97369444  .44256502  .44256502  .44256502  .28549057    <-- Repeated values due to RND(0)
2:  .4986506  4.4510484E-2  .96231  .35997057
3:  .8113741  .13687313  6.1034977E-2  .7874807
4:  .4986506  4.4510484E-2  .96231  .35997057                <-- Same sequence as line 2 due to same seed
```

<!-- markdownlint-disable MD024 -->
### Implementation Notes
<!-- markdownlint-enable MD024 -->

While the BASIC `RND(x)` function always returns a number between 0 (inclusive) and 1 (exclusive) for positive non-zero
values of `x`, game porters would find it convenient for the library to include functions returning a random float or
integer in a range from an inclusive minimum to an exclusive maximum.

As one of the games, "Football", makes use of `RND(0)` with different scaling applied than the previous use of `RND(1)`,
a common library implementation should always generate a value between 0 and 1, and scale that for a function with a
range, so that a call to the equivalent of the `RND(0)` function can return the previous value between 0 and 1.


# Games.Common

This is the common library for C# and VB.Net ports of the games.

## Overview

### Game Input/Output

* `TextIO` is the main class which manages text input and output for a game. It take a `TextReader` and a `TextWriter` in
its constructor so it can be wired up in unit tests to test gameplay scenarios.
* `ConsoleIO` derives from `TextIO` and binds it to `System.Console.In` and `System.Console.Out`.
* `IReadWrite` is an interface implemented by `TextIO` which may be useful in some test scenarios.

```csharp
public interface IReadWrite
{
    // Reads a float value from input.
    float ReadNumber(string prompt);

    // Reads 2 float values from input.
    (float, float) Read2Numbers(string prompt);

    // Reads 3 float values from input.
    (float, float, float) Read3Numbers(string prompt);

    // Reads 4 float values from input.
    (float, float, float, float) Read4Numbers(string prompt);

    // Read numbers from input to fill an array.
    void ReadNumbers(string prompt, float[] values);

    // Reads a string value from input.
    string ReadString(string prompt);

    // Reads 2 string values from input.
    (string, string) Read2Strings(string prompt);

    // Writes a string to output.
    void Write(string message);

    // Writes a string to output, followed by a new-line.
    void WriteLine(string message = "");

    // Writes a float to output, formatted per the BASIC interpreter, with leading and trailing spaces.
    void Write(float value);

    // Writes a float to output, formatted per the BASIC interpreter, with leading and trailing spaces,
    // followed by a new-line.
    void WriteLine(float value);

    // Writes the contents of a Stream to output.
    void Write(Stream stream);}
```

### Random Number Generation

* `IRandom` is an interface that provides basic methods that parallel the 3 uses of BASIC's `RND(float)` function.
* `RandomNumberGenerator` is an implementation of `IRandom` built around `System.Random`.
* `IRandomExtensions` provides convenience extension methods for obtaining random numbers as `int` and also within a
  given range.

```csharp
public interface IRandom
{
    // Like RND(1), gets a random float such that 0 <= n < 1.
    float NextFloat();

    // Like RND(0), Gets the float returned by the previous call to NextFloat.
    float PreviousFloat();

    // Like RND(-x), Reseeds the random number generator.
    void Reseed(int seed);
}
```

Extension methods on `IRandom`:

```csharp
// Gets a random float such that 0 <= n < exclusiveMaximum.
float NextFloat(this IRandom random, float exclusiveMaximum);

// Gets a random float such that inclusiveMinimum <= n < exclusiveMaximum.
float NextFloat(this IRandom random, float inclusiveMinimum, float exclusiveMaximum);

// Gets a random int such that 0 <= n < exclusiveMaximum.
int Next(this IRandom random, int exclusiveMaximum);

// Gets a random int such that inclusiveMinimum <= n < exclusiveMaximum.
int Next(this IRandom random, int inclusiveMinimum, int exclusiveMaximum);

// Gets the previous unscaled float (between 0 and 1) scaled to a new range:
// 0 <= x < exclusiveMaximum.
float PreviousFloat(this IRandom random, float exclusiveMaximum);

// Gets the previous unscaled float (between 0 and 1) scaled to a new range:
// inclusiveMinimum <= n < exclusiveMaximum.
float PreviousFloat(this IRandom random, float inclusiveMinimum, float exclusiveMaximum);

// Gets the previous unscaled float (between 0 and 1) scaled to an int in a new range:
// 0 <= n < exclusiveMaximum.
int Previous(this IRandom random, int exclusiveMaximum);

// Gets the previous unscaled float (between 0 and 1) scaled to an int in a new range:
// inclusiveMinimum <= n < exclusiveMaximum.
int Previous(this IRandom random, int inclusiveMinimum, int exclusiveMaximum);
```

## C\# Usage

### Add Project Reference

Add the `Games.Common` project as a reference to the game project. For example, here's the reference from the C\# port
of `86_Target`:

```xml
<ItemGroup>
  <ProjectReference Include="..\..\00_Common\dotnet\Games.Common\Games.Common.csproj" />
</ItemGroup>
```

### C# Game Input/Output usage

A game can be encapsulated in a class which takes a `TextIO` instance in it's constructor:

```csharp
public class Game
{
    private readonly TextIO _io;

    public Game(TextIO io) => _io = io;

    public void Play()
    {
        var name = _io.ReadString("What is your name");
        var (cats, dogs) = _io.Read2Number($"Hello, {name}, how many pets do you have (cats, dogs)");
        _io.WriteLine($"So, {cats + dogs} pets in total, huh?");
    }
}
```

Then the entry point of the game program would look something like:

```csharp
var game = new Game(new ConsoleIO());
game.Play();
```

### C# Random Number Generator usage

```csharp
var io = new ConsoleIO();
var rng = new RandomNumberGenerator();
io.WriteLine(rng.NextFloat());           // 0.1234, for example
io.WriteLine(rng.NextFloat());           // 0.6, for example
io.WriteLine(rng.PreviousFloat());       // 0.6, repeats previous
io.WriteLine(rng.PreviousFloat(0, 10));  // 6,   repeats previous value, but scaled to new range
```

### C# Unit Test usage

`TextIO` can be initialised with a `StringReader` and `StringWriter` to enable testing. For example, given the `Game`
class above:

```csharp
var reader = new StringReader("Joe Bloggs\r\n4\n\r5");
var writer = new StringWriter();
var game = new Game(new TextIO(reader, writer))

game.Play();

writer.ToString().Should().BeEquivalentTo(
    "What is your name? Hello, Joe Bloggs, how many pets do you have (cats, dogs)? ?? So, 9 pets in total, huh?");
```

Note the lack of line breaks in the expected output, because during game play the line breaks come from the text input.

Of course, `IReadWrite` can also be mocked for simple test scenarios.

## VB.Net Usage

*To be provided*


#### Utilities

These are global helper / utility programs to assist us in maintaining all the ports. 

# TODO list
 game                          | C# | Java | JS | Kotlin | Lua | Perl | Python | Ruby | Rust | VB.NET
------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
01_Acey_Ducey                  | ✅ | ✅ | ✅ | ⬜️ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅
[02_Amazing](../02_Amazing)    | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ✅ | ✅
[03_Animal](../03_Animal)      | ✅ | ✅ | ✅ | ✅ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[04_Awari](../04_Awari)        | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[05_Bagels](../05_Bagels)      | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⬜️
[06_Banner](../06_Banner)      | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ✅
[07_Basketball](../07_Basketball) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️
[08_Batnum](../08_Batnum)      | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅
[09_Battle](../09_Battle)      | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[10_Blackjack](../10_Blackjack) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️
[11_Bombardment](../11_Bombardment) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[12_Bombs_Away](../12_Bombs_Away) | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️
[13_Bounce](../13_Bounce)      | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️
[14_Bowling](../14_Bowling)    | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[15_Boxing](../15_Boxing)      | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[16_Bug](../16_Bug)            | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️
[17_Bullfight](../17_Bullfight) | ✅ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[18_Bullseye](../18_Bullseye)  | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️
[19_Bunny](../19_Bunny)        | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[20_Buzzword](../20_Buzzword)  | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[21_Calendar](../21_Calendar)  | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[22_Change](../22_Change)      | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ✅ | ⬜️
[23_Checkers](../23_Checkers)  | ✅ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️
[24_Chemist](../24_Chemist)    | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️
[25_Chief](../25_Chief)        | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[26_Chomp](../26_Chomp)        | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️
[27_Civil_War](../27_Civil_War) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[28_Combat](../28_Combat)      | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[29_Craps](../29_Craps)        | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[30_Cube](../30_Cube)          | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️
[31_Depth_Charge](../31_Depth_Charge) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[32_Diamond](../32_Diamond)    | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[33_Dice](../33_Dice)          | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ✅ | ✅
[34_Digits](../34_Digits)      | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[35_Even_Wins](../35_Even_Wins) | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️
[36_Flip_Flop](../36_Flip_Flop) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️
[37_Football](../37_Football)  | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[38_Fur_Trader](../38_Fur_Trader) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[39_Golf](../39_Golf)          | ✅ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[40_Gomoko](../40_Gomoko)      | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️
[41_Guess](../41_Guess)        | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ✅ | ⬜️
[42_Gunner](../42_Gunner)      | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[43_Hammurabi](../43_Hammurabi) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[44_Hangman](../44_Hangman)    | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[45_Hello](../45_Hello)        | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[46_Hexapawn](../46_Hexapawn)  | ✅ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[47_Hi-Lo](../47_Hi-Lo)        | ✅ | ✅ | ✅ | ✅ | ⬜️ | ✅ | ✅ | ✅ | ✅ | ⬜️
[48_High_IQ](../48_High_IQ)    | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[49_Hockey](../49_Hockey)      | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️ | ⬜️ | ⬜️ | ⬜️ | ⬜️
[50_Horserace](../50_Horserace) | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[51_Hurkle](../51_Hurkle)      | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[52_Kinema](../52_Kinema)      | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[53_King](../53_King)          | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[54_Letter](../54_Letter)      | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ✅ | ⬜️
[55_Life](../55_Life)          | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[56_Life_for_Two](../56_Life_for_Two) | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️ | ⬜️ | ⬜️ | ⬜️ | ⬜️
[57_Literature_Quiz](../57_Literature_Quiz) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️
[58_Love](../58_Love)          | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[59_Lunar_LEM_Rocket](../59_Lunar_LEM_Rocket) | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ✅ | ⬜️
[60_Mastermind](../60_Mastermind) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ✅ | ⬜️
[61_Math_Dice](../61_Math_Dice) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ✅ | ⬜️
[62_Mugwump](../62_Mugwump)    | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️
[63_Name](../63_Name)          | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[64_Nicomachus](../64_Nicomachus) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️
[65_Nim](../65_Nim)            | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️
[66_Number](../66_Number)      | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ✅ | ⬜️
[67_One_Check](../67_One_Check) | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️
[68_Orbit](../68_Orbit)        | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[69_Pizza](../69_Pizza)        | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[70_Poetry](../70_Poetry)      | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[71_Poker](../71_Poker)        | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ⬜️ | ⬜️ | ⬜️ | ⬜️
[72_Queen](../72_Queen)        | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️
[73_Reverse](../73_Reverse)    | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ✅
[74_Rock_Scissors_Paper](../74_Rock_Scissors_Paper) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ✅ | ⬜️
[75_Roulette](../75_Roulette)  | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️
[76_Russian_Roulette](../76_Russian_Roulette) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[77_Salvo](../77_Salvo)        | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[78_Sine_Wave](../78_Sine_Wave) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ✅ | ⬜️
[79_Slalom](../79_Slalom)      | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[80_Slots](../80_Slots)        | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[81_Splat](../81_Splat)        | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️
[82_Stars](../82_Stars)        | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ✅ | ⬜️
[83_Stock_Market](../83_Stock_Market) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[84_Super_Star_Trek](../84_Super_Star_Trek) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[85_Synonym](../85_Synonym)    | ✅ | ✅ | ✅ | ✅ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[86_Target](../86_Target)      | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[87_3-D_Plot](../87_3-D_Plot)  | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[88_3-D_Tic-Tac-Toe](../88_3-D_Tic-Tac-Toe) | ✅ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️ | ✅ | ⬜️ | ⬜️ | ⬜️
[89_Tic-Tac-Toe](../89_Tic-Tac-Toe) | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ✅ | ✅ | ⬜️ | ✅ | ⬜️
[90_Tower](../90_Tower)        | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ⬜️
[91_Train](../91_Train)        | ⬜️ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[92_Trap](../92_Trap)          | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[93_23_Matches](../93_23_Matches) | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ⬜️
[94_War](../94_War)            | ⬜️ | ✅ | ✅ | ✅ | ⬜️ | ✅ | ✅ | ✅ | ✅ | ⬜️
[95_Weekday](../95_Weekday)    | ✅ | ✅ | ✅ | ⬜️ | ⬜️ | ✅ | ✅ | ⬜️ | ✅ | ⬜️
------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
Sum of 95                      | 63 | 80 | 95 | 7 | 2 | 60 | 92 | 49 | 20 | 6


### Acey Ducey

This is a simulation of the Acey Ducey card game. In the game, the dealer (the computer) deals two cards face up. You have an option to bet or not to bet depending on whether or not you feel the next card dealt will have a value between the first two.

Your initial money is set to $100; you may want to alter this value if you want to start with more or less than $100. The game keeps going on until you lose all your money or interrupt the program.

The original program author was Bill Palmby of Prairie View, Illinois.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=2)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=17)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Known Bugs

- Entering a negative bet allows you to gain arbitrarily large amounts of money upon losing the round.

#### Porting Notes

- The assignment `N = 100` in line 100 has no effect; variable `N` is not used anywhere else in the program.

#### External Links
 - Common Lisp: https://github.com/koalahedron/lisp-computer-games/blob/master/01%20Acey%20Ducey/common-lisp/acey-deucy.lisp
 - PowerShell: https://github.com/eweilnau/basic-computer-games-powershell/blob/main/AceyDucey.ps1


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/) by Adam Dawes (@AdamDawes575, https://adamdawes.com).


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)

Two versions of Acey Ducey have been contributed.

The original upload supported JDK 8/JDK 11 and uses multiple files and the second uses features in JDK 17 and is implemented in a single file AceyDucey17.java.

Both are in the src folder.


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


As published in Basic Computer Games (1978), as found at Annarchive:
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=17)


Conversion to Lua
- [Lua.org](https://www.lua.org)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)

Propose using pylint and black to format python files so that it conforms to some standards


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/) by Christopher Özbek [coezbek@github](https://github.com/coezbek).


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by Alex Kotov [mur4ik18@github](https://github.com/mur4ik18).

Further edits by

- Berker Şal [berkersal@github](https://github.com/berkersal)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Amazing

This program will print out a different maze every time it is run and guarantees only one path through. You can choose the dimensions of the maze — i.e. the number of squares wide and long.

The original program author was Jack Hauber of Windsor, Connecticut.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=3)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=18)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Known Bugs

- The input dimensions are checked for values of 1, but not for values of 0 or less.  Such inputs will cause the program to break.

#### Porting Notes

**2022-01-04:** patched original source in [#400](https://github.com/coding-horror/basic-computer-games/pull/400) to fix a minor bug where a generated maze may be missing an exit, particularly at small maze sizes.


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

Converted to Ruby (with tons of inspiration from the Python version) by @marcheiligers

Run `ruby amazing.rb`.

Run `DEBUG=1 ruby amazing.ruby` to see how it works (requires at least Ruby 2.7).


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by Anthony Rubick [AnthonyMichaelTDM](https://github.com/AnthonyMichaelTDM)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Animal

Unlike other computer games in which the computer picks a number or letter and you must guess what it is, in this game _you_ think of an animal and the _computer_ asks you questions and tries to guess the name of your animal. If the computer guesses incorrectly, it will ask you for a question that differentiates the animal you were thinking of. In this way the computer “learns” new animals. Questions to differentiate new animals should be input without a question mark.

This version of the game does not have a SAVE feature. If your system allows, you may modify the program to save and reload the array when you want to play the game again. This way you can save what the computer learns over a series of games.

At any time if you reply “LIST” to the question “ARE YOU THINKING OF AN ANIMAL,” the computer will tell you all the animals it knows so far.

The program starts originally by knowing only FISH and BIRD. As you build up a file of animals you should use broad, general questions first and then narrow down to more specific ones with later animals. For example, if an elephant was to be your first animal, the computer would ask for a question to distinguish an elephant from a bird. Naturally, there are hundreds of possibilities, however, if you plan to build a large file of animals a good question would be “IS IT A MAMMAL.”

This program can be easily modified to deal with categories of things other than animals by simply modifying the initial data and the dialogue references to animals. In an educational environment, this would be a valuable program to teach the distinguishing characteristics of many classes of objects — rock formations, geography, marine life, cell structures, etc.

Originally developed by Arthur Luehrmann at Dartmouth College, Animal was subsequently shortened and modified by Nathan Teichholtz at DEC and Steve North at Creative Computing.

---

As published in Basic Computer Games (1978)
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=4)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=19)

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

Conversion to [Rust](https://www.rust-lang.org/) by [Anton Kaiukov](https://github.com/batk0)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)

This takes some inspiration from the [C# port of Animal](https://github.com/zspitz/basic-computer-games/tree/main/03_Animal/csharp).

The `Game` class takes a console abstraction (`ConsoleAdapterBase`), which could also be used for different UIs, such as WinForms or a web page.
This solution also has an xUnit tests project.
Responses can be entered in any capitalization, but animals and the distinguishing question will be converted to uppercase.


### Awari

Awari is an ancient African game played with seven sticks and thirty-six stones or beans laid out as shown above. The board is divided into six compartments or pits on each side. In addition, there are two special home pits at the ends.

A move is made by taking all the beans from any (non-empty) pit on your own side. Starting from the pit to the right of this one, these beans are ‘sown’ one in each pit working around the board anticlockwise.

A turn consists of one or two moves. If the last bean of your move is sown in your own home you may take a second move.

If the last bean sown in a move lands in an empty pit, provided that the opposite pit is not empty, all the beans in the opposite pit, together with the last bean sown are ‘captured’ and moved to the player’s home.

When either side is empty, the game is finished. The player with the most beans in his home has won.

In the computer version, the board is printed as 14 numbers representing the 14 pits.

```
    3   3   3   3   3   3
0                           0
    3   3   3   3   3   3
```

The pits on your (lower) side are numbered 1-6 from left to right. The pits on my (the computer’s) side are numbered from my left (your right).

To make a move you type in the number of a pit. If the last bean lands in your home, the computer types ‘AGAIN?’ and then you type in your second move.

The computer’s move is typed, followed by a diagram of the board in its new state. The computer always offers you the first move. This is considered to be a slight advantage.

There is a learning mechanism in the program that causes the play of the computer to improve as it playes more games.

The original version of Awari is adopted from one originally written by Geoff Wyvill of Bradford, Yorkshire, England.

---

As published in Basic Computer Games (1978)
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=6)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=21)

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

Conversion to [Ruby](https://www.ruby-lang.org/en/) by [Alex Scown](https://github.com/TheScown)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Bagels

In this game, the computer picks a 3-digit secret number using the digits 0 to 9 and you attempt to guess what it is. You are allowed up to twenty guesses. No digit is repeated. After each guess the computer will give you clues about your guess as follows:

- PICO    One digit is correct, but in the wrong place
- FERMI    One digit is in the correct place
- BAGELS   No digit is correct

You will learn to draw inferences from the clues and, with practice, you’ll learn to improve your score. There are several good strategies for playing Bagels. After you have found a good strategy, see if you can improve it. Or try a different strategy altogether to see if it is any better. While the program allows up to twenty guesses, if you use a good strategy it should not take more than eight guesses to get any number.

The original authors of this program are D. Resek and P. Rowe of the Lawrence Hall of Science, Berkeley, California.

---

As published in Basic Computer Games (1978)
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=9)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=21)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)
