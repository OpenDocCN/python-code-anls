# BasicComputerGames源码解析 1

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript flipflop.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "flipflop"
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
	miniscript football.ms
```or
```
	miniscript ftball.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "football"
	run
```or
```
	load "ftball"
	run
```

#### Apology from the Translator

These MiniScript programs were actually ported from the JavaScript ports of the original BASIC programs.  I did that because the BASIC code (of both programs) was incomprehensible spaghetti.  The JavaScript port, however, was essentially the same — and so are the MiniScript ports.  The very structure of these programs makes them near-impossible to untangle.

If I were going to write a football simulation from scratch, I would approach it very differently.  But in that case I would have either a detailed specification of how the program should behave, or at least enough understanding of American football to design it myself as I go.  Neither is the case here (and we're supposed to be porting the original programs, not making up our own).

So, I'm sorry.  Please take these programs as proof that you can write bad code even in the most simple, elegant languages.  And I promise to try harder on future translations!


Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [ANSI-C](https://en.wikipedia.org/wiki/ANSI_C)

##### Translator Notes:
I tried to preserve as much of the original layout and flow of the code
as possible.  However I did use enumerated types for the Fort numbers
and Fur types.  I think this was certainly a change for the better, and
makes the code much easier to read.

I also tried to minimise the use of pointers, and stuck with old-school
C formatting, because you never know how old the compiler is.

Interestingly the code seems to have a bug around the prices of Fox Furs.
The commodity-rate for these is stored in the variable `D1`, however some
paths through the code do not set this price.  So there was a chance of
using this uninitialised, or whatever the previous loop set.  I don't
think this was the original authors intent.  So I preserved the original flow
of the code (using the previous `D1` value), but also catching the
uninitialised path, and assigning a "best guess" value.

krt@krt.com.au 2020-10-10


Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript furtrader.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "furtrader"
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
	miniscript golf.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "golf"
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
	miniscript gomoko.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "gomoko"
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
	miniscript guess.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "guess"
	run
```
3. "Try-It!" page on the web:
Go to https://miniscript.org/tryit/, clear the default program from the source code editor, paste in the contents of guess.ms, and click the "Run Script" button.


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
	miniscript gunner.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "gunner"
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
	miniscript hammurabi.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "hammurabi"
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
	miniscript hangman.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "hangman"
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
	miniscript hello.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "hello"
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
	miniscript hexapawn.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "hexapawn"
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
Go to https://miniscript.org/tryit/, clear the sample code from the code editor, and paste in the contents of hi-lo.ms.  Then click the "Run Script" button.  Program output (and input) will appear in the green-on-black terminal display to the right of or below the code editor.

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript hi-lo.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "hi-lo"
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
	miniscript highiq.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "highiq"
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
	miniscript hockey.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "hockey"
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
	miniscript horserace.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "horserace"
	run
```

## Porting Notes

- The original program, designed to be played directly on a printer, drew a track 27 rows long.  To fit better on modern screens, I've shortened the track to 23 rows.  This is adjustable via the "trackLen" value assigned on line 72.

- Also because we're playing on a screen instead of a printer, I'm clearing the screen and pausing briefly before each new update of the track.  This is done via the `clear` API when running in Mini Micro, or by using a VT100 escape sequence in other contexts.


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
Go to https://miniscript.org/tryit/, clear the sample code from the code editor, and paste in the contents of hurkle.ms.  Then click the "Run Script" button.  Program output (and input) will appear in the green-on-black terminal display to the right of or below the code editor.

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript hurkle.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "hurkle"
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
	miniscript kinema.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "kinema"
	run
```
3. "Try-It!" page on the web:
Go to https://miniscript.org/tryit/, clear the default program from the source code editor, paste in the contents of kinema.ms, and click the "Run Script" button.


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
	miniscript king.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "king"
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
	miniscript letter.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "letter"
	run
```
3. "Try-It!" page on the web:
Go to https://miniscript.org/tryit/, clear the default program from the source code editor, paste in the contents of letter.ms, and click the "Run Script" button.


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
	miniscript life.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "life"
	run
```


Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).  The only liberty I took with the original design is that, when prompting each player for their turn, I include a reminder of what symbol (* or #) represents their pieces on the board.

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript lifefortwo.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "lifefortwo"
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
Go to https://miniscript.org/tryit/, clear the sample code from the code editor, and paste in the contents of litquiz.ms.  Then click the "Run Script" button.  Program output (and input) will appear in the green-on-black terminal display to the right of or below the code editor.

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript litquiz.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "litquiz"
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
Go to https://miniscript.org/tryit/, clear the sample code from the code editor, and paste in the contents of love.ms.  Then click the "Run Script" button.  Program output (and input) will appear in the green-on-black terminal display to the right of or below the code editor.

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript love.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "love"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).  Note that there are three different programs in this folder, all variations on the "land the LEM on the Moon" idea.

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the desired program with a command such as:

```
	miniscript lem.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "lem"	// (or "lunar" or "rocket")
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
	miniscript mastermind.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:
```
	load "mastermind"
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
Go to https://miniscript.org/tryit/, clear the sample code from the code editor, and paste in the contents of mathdice.ms.  Then click the "Run Script" button.  Program output (and input) will appear in the green-on-black terminal display to the right of or below the code editor.

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:
```
	miniscript mathdice.ms
```

2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:
```
	load "mathdice"
	run
```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Pascal](https://en.wikipedia.org/wiki/Pascal_(programming_language))

##### Translator Notes:
I tried to preserve as much of the original layout and flow of the code
as possible.  I added a procedure for the printing of the die-face; and
another to read an integer from the player, as I was unhappy with the runtime
error message spat out when a non-number is given to readln(<integer>).

I was torn between using the correct singular term "die" instead of "dice".
In the end I used a (poor?) combination of both.

krt@krt.com.au 2020-10-12


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
Go to https://miniscript.org/tryit/, clear the sample code from the code editor, and paste in the contents of mathdice.ms.  Then click the "Run Script" button.  Program output (and input) will appear in the green-on-black terminal display to the right of or below the code editor.

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:
```
	miniscript mathdice.ms
```

2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:
```
	load "mathdice"
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
Go to https://miniscript.org/tryit/, clear the sample code from the code editor, and paste in the contents of name.ms.  Then click the "Run Script" button.  Program output (and input) will appear in the green-on-black terminal display to the right of or below the code editor.

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:
```
	miniscript name.ms
```

2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:
```
	load "name"
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
	miniscript nicomachus.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "nicomachus"
	run
```
3. "Try-It!" page on the web:
Go to https://miniscript.org/tryit/, clear the default program from the source code editor, paste in the contents of nicomachus.ms, and click the "Run Script" button.


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
	miniscript nim.ms
```

2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:
```
	load "nim"
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
	miniscript number.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "number"
	run
```
3. "Try-It!" page on the web:
Go to https://miniscript.org/tryit/, clear the default program from the source code editor, paste in the contents of number.ms, and click the "Run Script" button.


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
	miniscript onecheck.ms
```

2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:
```
	load "onecheck"
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
	miniscript orbit.ms
```

2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:
```
	load "orbit"
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
	miniscript pizza.ms
```

2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:
```
	load "pizza"
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
	miniscript poetry.ms
```

2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:
```
	load "poetry"
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
	miniscript poker.ms
```

2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:
```
	load "poker"
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
	miniscript queen.ms
```

2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:
```
	load "queen"
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

	miniscript reverse.ms

2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

	load "reverse"
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

0. Try-It! Page:
Go to https://miniscript.org/tryit/, clear the sample code from the code editor, and paste in the contents of rockscissors.ms.  Then click the "Run Script" button.  Program output (and input) will appear in the green-on-black terminal display to the right of or below the code editor.

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:
```
	miniscript rockscissors.ms
```

2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:
```
	load "rockscissors"
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
	miniscript roulette.ms
```

2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:
```
	load "roulette"
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
Go to https://miniscript.org/tryit/, clear the sample code from the code editor, and paste in the contents of russianroulette.ms.  Then click the "Run Script" button.  Program output (and input) will appear in the green-on-black terminal display to the right of or below the code editor.

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:
```
	miniscript russianroulette.ms
```

2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:
```
	load "russianroulette"
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
	miniscript salvo.ms
```

2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:
```
	load "salvo"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [C++17](https://en.wikipedia.org/wiki/C%2B%2B17)

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript sinewave.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "sinewave"
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
	miniscript slalom.ms
```

2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:
```
	load "slalom"
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

	miniscript slots.ms

2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

	load "slots"
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

	miniscript splat.ms

2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

	load "splat"
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

	miniscript stars.ms

2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

	load "stars"
	run

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