# BasicComputerGames源码解析 28

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


# `19_Bunny/csharp/BasicData.cs`

这段代码定义了一个名为 "Bunny" 的类，其内部包含一个名为 "BasicData" 的类。

"BasicData" 类有一个私有成员变量 "data"，一个私有成员变量 "index"，和一个公共的 "Read" 方法。

"Read" 方法返回一个整数，它是 "data" 数组中的一个元素，通过索引 (index) 获取该元素的位置，然后返回它的值。

整型数据类型是一个用户定义的类型，它可以表示任意类型的数据，包括整数、字符串和浮点数等。

这段代码的用途可以根据上下文来确定，但通常用于说明一个类或一段代码的功能和用途。


```
﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Bunny
{
    internal class BasicData
    {
        private readonly int[] data;

        private int index;

        public BasicData(int[] data)
        {
            this.data = data;
            index = 0;
        }
        public int Read()
        {
            return data[index++];
        }
    }
}

```

# `19_Bunny/csharp/Bunny.cs`

This looks like a C# class that reads in a string from an `ASCII art` file, `bunnyData`, and prints it out to the console. It has a few methods:

* `PrintString`: This method takes in a `tab` number (which determines how many spaces the console will use for each character), a `value` string, and an optional `newLine` flag. It prints the string out to the console, using the spaces between the characters, and if `newLine` is `true`, it prints a new line.
* `PrintLines`: This method takes in the `count` number of spaces to use for each character in the `value` string. It prints the string out to the console, using the spaces between the characters, and if `count` is `0`, it prints a new line.
* `ReadString`: This method reads the `ASCII art` file and returns the first 128 characters as a `string`.

Note: The last line of the `PrintString` method, before the newline, should be `System.Environment.NewLine()` instead of `System.Environment.SystemError` because it is a typo and it will print the string to the console as expected.


```
﻿namespace Bunny
{
    internal class Bunny
    {
        private const int asciiBase = 64;
        private readonly int[] bunnyData = {
            2,21,14,14,25,
            1,2,-1,0,2,45,50,-1,0,5,43,52,-1,0,7,41,52,-1,
            1,9,37,50,-1,2,11,36,50,-1,3,13,34,49,-1,4,14,32,48,-1,
            5,15,31,47,-1,6,16,30,45,-1,7,17,29,44,-1,8,19,28,43,-1,
            9,20,27,41,-1,10,21,26,40,-1,11,22,25,38,-1,12,22,24,36,-1,
            13,34,-1,14,33,-1,15,31,-1,17,29,-1,18,27,-1,
            19,26,-1,16,28,-1,13,30,-1,11,31,-1,10,32,-1,
            8,33,-1,7,34,-1,6,13,16,34,-1,5,12,16,35,-1,
            4,12,16,35,-1,3,12,15,35,-1,2,35,-1,1,35,-1,
            2,34,-1,3,34,-1,4,33,-1,6,33,-1,10,32,34,34,-1,
            14,17,19,25,28,31,35,35,-1,15,19,23,30,36,36,-1,
            14,18,21,21,24,30,37,37,-1,13,18,23,29,33,38,-1,
            12,29,31,33,-1,11,13,17,17,19,19,22,22,24,31,-1,
            10,11,17,18,22,22,24,24,29,29,-1,
            22,23,26,29,-1,27,29,-1,28,29,-1,4096
        };

        public void Run()
        {
            PrintString(33, "BUNNY");
            PrintString(15, "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            PrintLines(3);

            // Set up a BASIC-ish data object
            BasicData data = new (bunnyData);

            // Get the first five data values into an array.
            // These are the characters we are going to print.
            // Unlike the original program, we are only converting
            // them to ASCII once.
            var a = new char[5];
            for (var i = 0; i < 5; ++i)
            {
                a[i] = (char)(asciiBase + data.Read());
            }
            PrintLines(6);

            PrintLines(1);
            var col = 0;
            while (true)
            {
                var x = data.Read();
                if (x < 0) // Start a new line
                {
                    PrintLines(1);
                    col = 0;
                    continue;
                }
                if (x > 128) break; // End processing
                col += PrintSpaces(x - col); // Move to TAB position x (sort of)
                var y = data.Read(); // Read the next value
                for (var i = x; i <= y; ++i)
                {
                    // var j = i - 5 * (i / 5); // BASIC didn't have a modulus operator
                    Console.Write(a[i % 5]);
                    // Console.Write(a[col % 5]); // This works, too
                    ++col;
                }
            }
            PrintLines(6);
        }
        private static void PrintLines(int count)
        {
            for (var i = 0; i < count; ++i)
                Console.WriteLine();
        }
        private static int PrintSpaces(int count)
        {
            for (var i = 0; i < count; ++i)
                Console.Write(' ');
            return count;
        }
        public static void PrintString(int tab, string value, bool newLine = true)
        {
            PrintSpaces(tab);
            Console.Write(value);
            if (newLine) Console.WriteLine();
        }

    }
}

```

# `19_Bunny/csharp/Program.cs`



这段代码是一个C#程序，定义了一个名为Bunny的类，包含一个名为Main的静态方法。

程序的主要作用是创建一个Bunny对象，然后使用Bunny对象中的Run方法运行程序。

具体来说，程序首先导入了System、System.Collections.Generic和System.Linq库，这些库包含了一些常用的类和函数，对程序的运行很有帮助。

接着，程序定义了一个Bunny类，其中包含一个名为Run的静态方法。这个方法接受 nothing 作为参数，表示不会执行任何操作，因为Bunny对象还没有被创建。

最后，程序定义了一个名为Main的静态方法，这个方法就是程序的入口点。在Main方法的体内，创建了一个Bunny对象，然后调用Bunny对象的Run方法，这样程序就可以运行了。

总的来说，这段代码是一个简单的程序，用于创建一个Bunny对象，并运行这个对象。


```
﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Bunny
{
    public static class Program
    {
        public static void Main()
        {
            new Bunny().Run();
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `19_Bunny/java/src/Bunny.java`

This code appears to define a `Bunny` class that has a few methods: `process`, `eat`, and `move`. `process` appears to convert the data in the `theData` list into the `theData2` list, which is not defined in this code. `eat` appears to take a piece of food and adds it to the `theData2` list. `move` appears to move the Bunny 2 units north, then 1 unit west. The `Bunny` class has a few instance variables, including a `theData` list, which is defined as `null`, and an `theData2` list, which is defined as `null`. The `Bunny` class has a constructor that takes `theData` as a parameter, and a `void` method `process`, which is not defined in this code. It looks like `eat` and `move` are defined in this code, but I am not able to verify this.


```
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Bunny
 * <p>
 * Based on the Basic program Bunny
 * https://github.com/coding-horror/basic-computer-games/blob/main/19%20Bunny/bunny.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */

public class Bunny {

    // First 4 elements are the text BUNNY, so skip those
    public static final int REAL_DATA_START_POS = 5;

    // Data for characters is not representative of three ASCII character, so we have
    // to add 64 to it as per original program design.
    public static final int CONVERT_TO_ASCII = 64;

    public static final int EOF = 4096; //End of file
    public static final int EOL = -1;  // End of line

    // Contains the data to draw the picture
    private final ArrayList<Integer> data;

    public Bunny() {
        data = loadData();
    }

    /**
     * Show an intro, then draw the picture.
     */
    public void process() {

        intro();

        // First 5 characters of data spells out BUNNY, so add this to a string
        StringBuilder bunnyBuilder = new StringBuilder();
        for (int i = 0; i < REAL_DATA_START_POS; i++) {
            // Convert the data to the character representation for output
            // Ascii A=65, B=66 - see loadData method
            bunnyBuilder.append(Character.toChars(data.get(i) + CONVERT_TO_ASCII));
        }

        // We now have the string to be used in the output
        String bunny = bunnyBuilder.toString();

        int pos = REAL_DATA_START_POS;  // Point to the start of the actual data
        int previousPos = 0;

        // Loop until we reach a number indicating EOF
        while (true) {
            // This is where we want to start drawing
            int first = data.get(pos);
            if (first == EOF) {
                break;
            }
            if (first == EOL) {
                System.out.println();
                previousPos = 0;
                // Move to the next element in the ArrayList
                pos++;
                continue;
            }

            // Because we are not using screen positioning, we just add an appropriate
            // numbers of spaces from where we want to be, and where we last outputted something
            System.out.print(addSpaces(first - previousPos));

            // We use this next time around the loop
            previousPos = first;

            // Move to next element
            pos++;
            // This is where we want to stop drawing/
            int second = data.get(pos);

            // Now we loop through the number of characters to draw using
            // the starting and ending point.
            for (int i = first; i <= second; i++) {
                // Cycle through the actual number of characters but use the
                // remainder operator to ensure we only use characters from the
                // bunny string
                System.out.print(bunny.charAt(i % bunny.length()));
                // Advance where we were at.
                previousPos += 1;
            }
            // Point to next data element
            pos++;
        }

        System.out.println();

    }

    private void intro() {
        System.out.println(addSpaces(33) + "BUNNY");
        System.out.println(addSpaces(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
    }

    /**
     * Return a string of x spaces
     *
     * @param spaces number of spaces required
     * @return String with number of spaces
     */
    private String addSpaces(int spaces) {
        char[] spacesTemp = new char[spaces];
        Arrays.fill(spacesTemp, ' ');
        return new String(spacesTemp);
    }

    /**
     * Original Basic program had the data in DATA format.
     * We're importing all the data into an array for ease of processing.
     * Format of data is
     * characters 0-4 is the letters that will be used in the output. 64 + the value represents the ASCII character
     * ASCII code 65 = A, 66 = B, etc.  so 2+64=66 (B), 21+64=85 (U) and so on.
     * Then we next have pairs of numbers.
     * Looking at the this data
     * 1,2,-1,0,2,45,50,-1
     * That reads as
     * 1,2 = draw characters - in this case BU
     * -1 = go to a new line
     * 0,2 = DRAW BUN
     * 45,50 = DRAW BUNNYB starting at position 45
     * and so on.
     * 4096 is EOF
     *
     * @return ArrayList of type Integer containing the data
     */
    private ArrayList<Integer> loadData() {

        ArrayList<Integer> theData = new ArrayList<>();

        // This is the data faithfully added from the original basic program.
        // Notes:
        // The first 5 ints are ASCII character (well 64 is added to make them ASCII chars we can output).
        theData.addAll(Arrays.asList(2, 21, 14, 14, 25));
        theData.addAll(Arrays.asList(1, 2, -1, 0, 2, 45, 50, -1, 0, 5, 43, 52, -1, 0, 7, 41, 52, -1));
        theData.addAll(Arrays.asList(1, 9, 37, 50, -1, 2, 11, 36, 50, -1, 3, 13, 34, 49, -1, 4, 14, 32, 48, -1));
        theData.addAll(Arrays.asList(5, 15, 31, 47, -1, 6, 16, 30, 45, -1, 7, 17, 29, 44, -1, 8, 19, 28, 43, -1));
        theData.addAll(Arrays.asList(9, 20, 27, 41, -1, 10, 21, 26, 40, -1, 11, 22, 25, 38, -1, 12, 22, 24, 36, -1));
        theData.addAll(Arrays.asList(13, 34, -1, 14, 33, -1, 15, 31, -1, 17, 29, -1, 18, 27, -1));
        theData.addAll(Arrays.asList(19, 26, -1, 16, 28, -1, 13, 30, -1, 11, 31, -1, 10, 32, -1));
        theData.addAll(Arrays.asList(8, 33, -1, 7, 34, -1, 6, 13, 16, 34, -1, 5, 12, 16, 35, -1));
        theData.addAll(Arrays.asList(4, 12, 16, 35, -1, 3, 12, 15, 35, -1, 2, 35, -1, 1, 35, -1));
        theData.addAll(Arrays.asList(2, 34, -1, 3, 34, -1, 4, 33, -1, 6, 33, -1, 10, 32, 34, 34, -1));
        theData.addAll(Arrays.asList(14, 17, 19, 25, 28, 31, 35, 35, -1, 15, 19, 23, 30, 36, 36, -1));
        theData.addAll(Arrays.asList(14, 18, 21, 21, 24, 30, 37, 37, -1, 13, 18, 23, 29, 33, 38, -1));
        theData.addAll(Arrays.asList(12, 29, 31, 33, -1, 11, 13, 17, 17, 19, 19, 22, 22, 24, 31, -1));
        theData.addAll(Arrays.asList(10, 11, 17, 18, 22, 22, 24, 24, 29, 29, -1));
        theData.addAll(Arrays.asList(22, 23, 26, 29, -1, 27, 29, -1, 28, 29, -1, 4096));

        return theData;
    }

    public static void main(String[] args) {

        Bunny bunny = new Bunny();
        bunny.process();
    }
}

```

# `19_Bunny/javascript/bunny.js`

这段代码定义了两个函数：print()和tab()。

print()函数的作用是打印一个字符串到网页的输出框中，该函数接收一个字符串参数，并将其创建为含有换行符的字符数组，然后将该数组添加到网页的输出框中。

tab()函数的作用是打印一个字符串，但每输出一个字符，就将其背景颜色更改为与前一个字符相同的颜色，直到输出完所有字符。函数接收一个字符参数，并返回该字符及其背景颜色，以便在循环中更新字符的颜色。


```
// BUNNY
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//
function print(str)
{
	document.getElementById("output").appendChild(document.createTextNode(str));
}

function tab(space)
{
	var str = "";
	while (space-- > 0)
		str += " ";
	return str;
}

```

It looks like the data you provided is a valid Bitcoin地址， MX1 wallet.



```
var bunny_string = ["B","U","N","N","Y"];

var bunny_data = [1,2,-1,0,2,45,50,-1,0,5,43,52,-1,0,7,41,52,-1,
	1,9,37,50,-1,2,11,36,50,-1,3,13,34,49,-1,4,14,32,48,-1,
	5,15,31,47,-1,6,16,30,45,-1,7,17,29,44,-1,8,19,28,43,-1,
	9,20,27,41,-1,10,21,26,40,-1,11,22,25,38,-1,12,22,24,36,-1,
	13,34,-1,14,33,-1,15,31,-1,17,29,-1,18,27,-1,
	19,26,-1,16,28,-1,13,30,-1,11,31,-1,10,32,-1,
	8,33,-1,7,34,-1,6,13,16,34,-1,5,12,16,35,-1,
	4,12,16,35,-1,3,12,15,35,-1,2,35,-1,1,35,-1,
	2,34,-1,3,34,-1,4,33,-1,6,33,-1,10,32,34,34,-1,
	14,17,19,25,28,31,35,35,-1,15,19,23,30,36,36,-1,
	14,18,21,21,24,30,37,37,-1,13,18,23,29,33,38,-1,
	12,29,31,33,-1,11,13,17,17,19,19,22,22,24,31,-1,
	10,11,17,18,22,22,24,24,29,29,-1,
	22,23,26,29,-1,27,29,-1,28,29,-1,4096];

```

这段代码会输出一系列带有不同背景颜色的字符。具体来说，它将输出一个表格，每个表格包含不同字母和新单词。

表格的第一行包含一个空格，第二行包含 "BUNNY"。接下来，接下来的行将循环输出不同的字符，以模拟不同字母和新单词。

循环的逻辑是，首先打印当前字母的背景颜色，然后将字符串清空并重复这个过程，直到达到当前行的结束位置。在循环过程中，如果当前位置的背景颜色是负数，那么将打印该位置的字符，并将字符串清空。

每次循环结束后，将打印一行新的字符。在整个表格将打印完毕后，再次输出一个空行，以作为下一行的分隔符。


```
print(tab(32) + "BUNNY\n");
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
print("\n");
print("\n");
print("\n");

var l = 64;	// ASCII letter code
var pos = 0;

print("\n");

var str = "";
for (var pos = 0; bunny_data[pos] < 128; pos++) {
	if (bunny_data[pos] < 0) {
		print(str + "\n");
		str = "";
		continue;
	}
	while (str.length < bunny_data[pos])
		str += " ";
	for (var i = bunny_data[pos]; i <= bunny_data[pos + 1]; i++)
		str += bunny_string[i % 5];
	pos++;
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


# `19_Bunny/python/bunny.py`

这段代码是一个Python脚本，主要作用是读取并打印一个JSON数据文件中的数据。

具体来说，代码首先导入了Python标准库中的json模块，然后使用with open("data.json") as f语句打开了一个名为"data.json"的JSON数据文件。接着，利用json.load(f)语句将文件中的JSON数据读取并存储到了一个tuple中，存储到了变量DATA中。

然后，定义了一个print_intro()函数，该函数通过print语句输出了一个带有" " * 33 + "BUNNY"字符的三个字符，以及一个带有" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"字符的三个字符。通过print_intro()函数可以打印出完整的字符串，使得输出更易于阅读。

最后，在程序运行时需要运行的是print_intro()函数，因此，print_intro()函数中的代码不会被执行。


```
#!/usr/bin/env python3


import json

# This data is meant to be read-only, so we are storing it in a tuple
with open("data.json") as f:
    DATA = tuple(json.load(f))


def print_intro() -> None:
    print(" " * 33 + "BUNNY")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print("\n\n")


```

If you have any specific question or if there is anything else I can help you with, please let me know.


```
def main() -> None:
    print_intro()

    # Using an iterator will give us a similar interface to BASIC's READ
    # command. Instead of READ, we will call 'next(data)' to fetch the next element.
    data = iter(DATA)

    # Read the first 5 numbers. These correspond to letters of the alphabet.
    # B=2, U=21, N=14, N=14, Y=25

    # Usually, list comprehensions are good for transforming each element in a sequence.
    # In this case, we are using range to repeat the call to next(data) 5 times. The underscore (_)
    # indicates that the values from range are discarded.
    bunny = [next(data) for _ in range(5)]
    L = 64

    # Interpretting a stream of data is a very common software task. We've already intepretted
    # the first 5 numbers as letters of the alphabet (with A being 1). Now, we are going to
    # combine this with a different interpretation of the following data to draw on the screen.
    # The drawing data is essentially a series of horizontal line segments given as begin and end
    # offsets.
    while True:
        command = next(data)

        if command < 0:
            print()
            continue

        if command > 128:
            break

        # If we've reached this portion of the code, 'command' indicates the 'start'
        # position of a line segment.
        start = command
        # Position cursor at start
        print(" " * start, end="")

        # The following number, indicates the end of the segment.
        end = next(data)
        # Unlike FOR I=X TO Y, the 'stop' argument of 'range' is non-inclusive, so we must add 1
        for i in range(start, end + 1, 1):
            # Cycle through the letters in "BUNNY" as we draw line
            j = i - 5 * int(i / 5)
            print(chr(L + bunny[j]), end="")


```

这段代码是一个if语句，判断当前程序是否作为主程序运行。如果当前程序是作为主程序运行，那么程序会执行if语句中的内容。

if __name__ == "__main__":

这段代码中包含两个部分。第一部分是一个if语句，判断当前程序是否作为主程序运行。第二部分是一个大括号，包含两个参数，分别是__main__和__name__。其中，__main__是一个字符串，表示程序的主函数名称，而__name__是一个字符串，表示程序的模块名称。

if __name__ == "__main__":

这段代码的作用是判断当前程序是否作为主程序运行。如果当前程序是作为主程序运行，那么程序会执行if语句中的内容，否则不会执行。


```
if __name__ == "__main__":
    main()

```

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


# `20_Buzzword/csharp/Program.cs`

This is a C# class that generates a random phrase from a list of words. It first checks if the input is empty and returns an empty string if it is. Then, it converts the first character of the input to uppercase and appends the rest of the characters in the same order, converting each character to uppercase. This is the same effect as in the original code provided.

The class also generates a random phrase from a list of words. It does this by first calculating the index of the first character in each word in the list, based on the order they appear in the list. It then returns the string representation of the random phrase.

The class has a `GeneratePhrase` method that generates a random phrase from the words in the list. It also has a `Decision` method that prompts the user to choose between "Y" for "yes" and "N" for "no". If the user chooses "Y", the function returns `true`. If the user chooses "N", the function returns `false`.

The `Main` method has a loop that prompts the user to enter a decision. If the user chooses "Y", the loop breaks and the function returns. Otherwise, it generates a random phrase and displays it. Finally, it displays a message and prompts the user to come back if they need help with another report.


```
﻿using System;

namespace Buzzword
{
    class Program
    {
        /// <summary>
        /// Displays header.
        /// </summary>
        static void Header()
        {
            Console.WriteLine("Buzzword generator".PadLeft(26));
            Console.WriteLine("Creating Computing Morristown, New Jersey".PadLeft(15));
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
        }

        // Information for the user about possible key input.
        static string keys = "type a 'Y' for another phrase or 'N' to quit";

        /// <summary>
        /// Displays instructions.
        /// </summary>
        static void Instructions()
        {
            Console.WriteLine("This program prints highly acceptable phrases in\n"
            + "'educator-speak' that you can work into reports\n"
            + "and speeches. Whenever a question mark is printed,\n"
            + $"{keys}.");
            Console.WriteLine();
            Console.WriteLine();
            Console.Write("Here's the first phrase:");
        }

        static string[] Words = new[]
            { "ability", "basal", "behavioral", "child-centered",
            "differentiated", "discovery", "flexible", "heterogenous",
            "homogeneous", "manipulative", "modular", "tavistock",
            "individualized", "learning", "evaluative", "objective",
            "cognitive", "enrichment", "scheduling", "humanistic",
            "integrated", "non-graded", "training", "vertical age",
            "motivational", "creative", "grouping", "modification",
            "accountability", "process", "core curriculum", "algorithm",
            "performance", "reinforcement", "open classroom", "resource",
            "structure", "facility", "environment" };

        /// <summary>
        /// Capitalizes first letter of given string.
        /// </summary>
        /// <param name="input"></param>
        /// <returns>string</returns>
        static string Capitalize(string input)
        {
            if (string.IsNullOrWhiteSpace(input))
                return string.Empty;

            return char.ToUpper(input[0]) + input[1..];
        }

        // Seed has been calculated to get the same effect as in original,
        // at least in first phrase
        static readonly Random rnd = new Random(1486);

        /// <summary>
        /// Generates random phrase from words available in Words array.
        /// </summary>
        /// <returns>String representing random phrase where first letter is capitalized.</returns>
        static string GeneratePhrase()
        {
            // Indexing from 0, so had to decrease generated numbers
            return $"{Capitalize(Words[rnd.Next(13)])} "
                + $"{Words[rnd.Next(13, 26)]} "
                + $"{Words[rnd.Next(26, 39)]}";
        }

        /// <summary>
        /// Handles user input. On wrong input it displays information about
        /// valid keys in infinite loop.
        /// </summary>
        /// <returns>True if user pressed 'Y', false if 'N'.</returns>
        static bool Decision()
        {
            while (true)
            {
                Console.Write("?");
                var answer = Console.ReadKey();
                if (answer.Key == ConsoleKey.Y)
                    return true;
                else if (answer.Key == ConsoleKey.N)
                    return false;
                else
                    Console.WriteLine($"\n{keys}");
            }
        }

        static void Main(string[] args)
        {
            Header();
            Instructions();

            while (true)
            {
                Console.WriteLine();
                Console.WriteLine(GeneratePhrase());
                Console.WriteLine();

                if (!Decision())
                    break;
            }

            Console.WriteLine("\nCome back when you need help with another report!");
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `20_Buzzword/java/src/Buzzword.java`

这段代码定义了一个名为“Buzzword”的类，它继承自“java.util.Scanner”类。

在“main”方法中，使用“try”关键字创建了一个try块，这意味着代码块内可能存在代码异常。

在try块内部，使用“final”关键字声明了一个名为“buzzwords”的变量，并使用了“new”关键字创建了一个新的实例。

然后，代码块内部使用“try”关键字创建了一个新的“Scanner”对象，并将其赋值给“buzzwords”变量，这样“buzzwords”变量就可以读取用户从键盘输入。

接下来，代码块内部创建了一个新的“UserInterface”对象，并将其赋值给“userInterface”变量。

然后，将“userInterface”对象所实现的“run”方法传入给“try”块，这样“userInterface”对象就可以处理用户输入并返回 buzzwords 对象。

最后，该代码块使用“}”结束，意味着 try 块内的代码都会被执行。


```
import java.util.Scanner;

public class Buzzword {

	public static void main(final String[] args) {
		try (
			// Scanner is a Closeable so it must be closed
			// before the program ends.
			final Scanner scanner = new Scanner(System.in);
		) {
			final BuzzwordSupplier buzzwords = new BuzzwordSupplier();
			final UserInterface userInterface = new UserInterface(
					scanner, System.out, buzzwords);
			userInterface.run();
		}
	}
}

```

# `20_Buzzword/java/src/BuzzwordSupplier.java`

这段代码定义了一个名为`BuzzwordSupplier`的类，它实现了`Supplier`接口，用于生成随机代码。`BuzzwordSupplier`的目的是提供无尽的随机 buzzwords(随机短语)，这些buzzwords由随机选自三个文本集合中的单词组成：`SET_1`,`SET_2`和`SET_3`。

具体来说，`BuzzwordSupplier`类中的`get()`方法从`SET_1`、`SET_2`和`SET_3`中随机选择一个单词，然后将其与另一个随机单词连接起来，形成一个buzzword。在`random.nextInt(SET_1.length)`中，我们随机选择SET_1中的一个单词，然后使用`random.nextInt(SET_2.length)`随机选择SET_2中的一个单词，最后使用`random.nextInt(SET_3.length)`随机选择SET_3中的一个单词。然后我们使用`+`运算符将这些单词连接起来，并将其返回。

由于我们使用的是`Random`类，因此生成的buzzwords是随机的，并且由于我们只选择了一个单词，因此生成的buzzwords也是有限的。


```
import java.util.Random;
import java.util.function.Supplier;

/**
 * A string supplier that provides an endless stream of random buzzwords.
 */
public class BuzzwordSupplier implements Supplier<String> {

	private static final String[] SET_1 = {
			"ABILITY","BASAL","BEHAVIORAL","CHILD-CENTERED",
			"DIFFERENTIATED","DISCOVERY","FLEXIBLE","HETEROGENEOUS",
			"HOMOGENEOUS","MANIPULATIVE","MODULAR","TAVISTOCK",
			"INDIVIDUALIZED" };

	private static final String[] SET_2 = {
			"LEARNING","EVALUATIVE","OBJECTIVE",
			"COGNITIVE","ENRICHMENT","SCHEDULING","HUMANISTIC",
			"INTEGRATED","NON-GRADED","TRAINING","VERTICAL AGE",
			"MOTIVATIONAL","CREATIVE" };

	private static final String[] SET_3 = {
			"GROUPING","MODIFICATION", "ACCOUNTABILITY","PROCESS",
			"CORE CURRICULUM","ALGORITHM", "PERFORMANCE",
			"REINFORCEMENT","OPEN CLASSROOM","RESOURCE", "STRUCTURE",
			"FACILITY","ENVIRONMENT" };

	private final Random random = new Random();

	/**
	 * Create a buzzword by concatenating a random word from each of the
	 * three word sets.
	 */
	@Override
	public String get() {
		return SET_1[random.nextInt(SET_1.length)] + ' ' +
				SET_2[random.nextInt(SET_2.length)] + ' ' +
				SET_3[random.nextInt(SET_3.length)];
	}
}

```

# `20_Buzzword/java/src/UserInterface.java`

This is a Java class that implements a user interface for a buzzword generator. The class takes in three arguments:

1. A Scanner object that reads input from the user,
2. A PrintStream object that will be used to display messages to the user, and
3. A Supplier object that will be used to provide the buzzwords to be printed.

The class has a method called `run()` which runs the user interface. In the `run()` method, the class prints a header message to the user and then enters a loop that reads a buzzword from the user until they either type "Y" to quit or type anything else. Once the user has entered a buzzword, the class prints it to the output and enters a new line. The loop continues until the user quits.

Overall, this class provides a simple way for users to generate buzzwords for reports and speeches.


```
import java.io.PrintStream;
import java.util.Scanner;
import java.util.function.Supplier;

/**
 * A command line user interface that outputs a buzzword every
 * time the user requests a new one.
 */
public class UserInterface implements Runnable {

	/**
	 * Input from the user.
	 */
	private final Scanner input;

	/**
	 * Output to the user.
	 */
	private final PrintStream output;

	/**
	 * The buzzword generator.
	 */
	private final Supplier<String> buzzwords;

	/**
	 * Create a new user interface.
	 *
	 * @param input The input scanner with which the user gives commands.
	 * @param output The output to show messages to the user.
	 * @param buzzwords The buzzword supplier.
	 */
	public UserInterface(final Scanner input,
			final PrintStream output,
			final Supplier<String> buzzwords) {
		this.input = input;
		this.output = output;
		this.buzzwords = buzzwords;
	}

	@Override
	public void run() {
		output.println("              BUZZWORD GENERATOR");
		output.println("   CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
		output.println();
		output.println();
		output.println();
		output.println("THIS PROGRAM PRINTS HIGHLY ACCEPTABLE PHRASES IN");
		output.println("'EDUCATOR-SPEAK' THAT YOU CAN WORK INTO REPORTS");
		output.println("AND SPEECHES.  WHENEVER A QUESTION MARK IS PRINTED,");
		output.println("TYPE A 'Y' FOR ANOTHER PHRASE OR 'N' TO QUIT.");
		output.println();
		output.println();
		output.println("HERE'S THE FIRST PHRASE:");

		do {
			output.println(buzzwords.get());
			output.println();
			output.print("?");
		} while ("Y".equals(input.nextLine().toUpperCase()));

		output.println("COME BACK WHEN YOU NEED HELP WITH ANOTHER REPORT!");
	}
}

```

# `20_Buzzword/javascript/buzzword.js`

这是一个 JavaScript 代码片段，可以从 Basic 编程语言转换为 JavaScript 编程语言。代码包括两个主要函数：`print` 和 `input`。

1. `print` 函数的作用是将要打印的字符串添加到页面上的一行文本中。它通过 `document.getElementById("output")` 引用一个 HTML 元素，这个元素将接收到的字符串附加到页面上。

2. `input` 函数的作用是从用户接收输入字符串。它使用 `document.createElement("INPUT")` 创建一个 `INPUT` 元素，设置其 `type` 属性为 `text` 和 `length` 属性为 `50`（表示最大字符数）。将创建好的 `INPUT` 元素添加到页面上，设置其 `focus` 属性，这样当用户点击它时输入的字符将出现在页面上。然后，设置一个 `keydown` 事件监听器，以便在用户按键时捕获键盘输入。当用户按下了回车键或其它的 `keydown` 事件发生时，函数会捕获事件并读取用户的输入字符串。然后，将输入的字符串附加到页面上并打印出来。


```
// BUZZWORD
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

这段代码定义了一个名为 `tab` 的函数，它会将传入的参数 `space` 中的所有字符串连接成一个空格字符串，并返回该字符串。

在函数内部，首先定义了一个名为 `str` 的空字符串变量，并使用一个无限循环来将传入的每个字符串连接到 `str` 字符串的开头。每次循环中，字符串中的字符数组 `space` 会递减 1，这意味着在循环的最后，`space` 的值为 0。

最后，函数返回 `str` 字符串，它是一个由传入参数中的所有字符串连接而成的空格字符串。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var a = ["",
         "ABILITY","BASAL","BEHAVIORAL","CHILD-CENTERED",
         "DIFFERENTIATED","DISCOVERY","FLEXIBLE","HETEROGENEOUS",
         "HOMOGENEOUS","MANIPULATIVE","MODULAR","TAVISTOCK",
         "INDIVIDUALIZED","LEARNING","EVALUATIVE","OBJECTIVE",
         "COGNITIVE","ENRICHMENT","SCHEDULING","HUMANISTIC",
         "INTEGRATED","NON-GRADED","TRAINING","VERTICAL AGE",
         "MOTIVATIONAL","CREATIVE","GROUPING","MODIFICATION",
         "ACCOUNTABILITY","PROCESS","CORE CURRICULUM","ALGORITHM",
         "PERFORMANCE","REINFORCEMENT","OPEN CLASSROOM","RESOURCE",
         "STRUCTURE","FACILITY","ENVIRONMENT",
         ];

```

这段代码是一个Python程序，主要作用是输出一些短语或句子，并等待用户输入来打印进一步的内容。现在让我们逐行来分析：

```
# Main program
asyncio.get_event_loop().run_until_complete(main())
```

这段代码使用asyncio库中的get_event_loop()方法来获取事件循环，然后使用run_until_complete()方法来运行主函数。

```
async function main()
{
   print(tab(26) + "BUZZWORD GENERATOR\n")
   print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")
   print("\n")
   print("\n")
   print("THIS PROGRAM PRINTS HIGHLY ACCEPTABLE PHRASES IN")
   print("'EDUCATOR-SPEAK' THAT YOU CAN WORK INTO REPORTS")
   print("AND SPEECHES.  WHENEVER A QUESTION MARK IS PRINTED,")
   print("TYPE A 'Y' FOR ANOTHER PHRASE OR 'N' TO QUIT.")
   print("\n")
   print("HERE'S THE FIRST PHRASE:\n")
   do {
       print(a[Math.floor(Math.random() * 13 + 1)] + " ")
       print(a[Math.floor(Math.random() * 13 + 14)] + " ")
       print(a[Math.floor(Math.random() * 13 + 27)] + "\n")
       print("\n")
       y = await input()
   } while (y == "Y") ;
   print("COME BACK WHEN YOU NEED HELP WITH ANOTHER REPORT!\n")
}
```

这段代码首先定义了一个主函数main。在这个函数中，我们输出了一些文本，包括单词和句子，然后等待用户输入来打印进一步的内容。

```
print(tab(26) + "BUZZWORD GENERATOR")
print(tab(15) + "CREATIVE COMPUTING MORRISTOWN, NEW JERSEY")
print("THIS PROGRAM PRINTS HIGHLY ACCEPTABLE PHRASES IN")
print("'EDUCATOR-SPEAK' THAT YOU CAN WORK INTO REPORTS")
print("AND SPEECHES.  WHENEVER A QUESTION MARK IS PRINTED,")
print("TYPE A 'Y' FOR ANOTHER PHRASE OR 'N' TO QUIT.")
print("HERE'S THE FIRST PHRASE:\n")
```

这些函数的作用与上面解释的类似，只是对一些文本进行了处理。

```
do {
   print(a[Math.floor(Math.random() * 13 + 1)] + " ")
   print(a[Math.floor(Math.random() * 13 + 14)] + " ")
   print(a[Math.floor(Math.random() * 13 + 27)] + "\n")
   print("\n")
   y = await input()
} while (y == "Y") ;
```

这个函数的作用是让用户输入一个句子，然后打印出来。我们用Math.random() * 13 + 1来选择一个单词，Math.random() * 13 + 27来选择一个句子。

```
print("COME BACK WHEN YOU NEED HELP WITH ANOTHER REPORT!\n")
```

这段文本输出了一条消息，让用户知道可以随时帮助他们。

```
asyncio.get_event_loop().run_until_complete(main())
```

这段代码使用asyncio库中的get_event_loop()方法来获取事件循环，然后使用run_until_complete()方法来运行主函数。


```
// Main program
async function main()
{
    print(tab(26) + "BUZZWORD GENERATOR\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("THIS PROGRAM PRINTS HIGHLY ACCEPTABLE PHRASES IN\n");
    print("'EDUCATOR-SPEAK' THAT YOU CAN WORK INTO REPORTS\n");
    print("AND SPEECHES.  WHENEVER A QUESTION MARK IS PRINTED,\n");
    print("TYPE A 'Y' FOR ANOTHER PHRASE OR 'N' TO QUIT.\n");
    print("\n");
    print("\n");
    print("HERE'S THE FIRST PHRASE:\n");
    do {
        print(a[Math.floor(Math.random() * 13 + 1)] + " ");
        print(a[Math.floor(Math.random() * 13 + 14)] + " ");
        print(a[Math.floor(Math.random() * 13 + 27)] + "\n");
        print("\n");
        y = await input();
    } while (y == "Y") ;
    print("COME BACK WHEN YOU NEED HELP WITH ANOTHER REPORT!\n");
}

```

这道题是一个简单的C语言程序，包含了两个主要部分：`main()`函数和`printf()`函数。我们需要分析这两部分的作用及其它所能做的内容。

1. `main()`函数：

`main()`函数是程序的入口点，程序从这里开始执行。在这个函数中，首先定义了一个`printf()`函数，将其声明为可输出函数。然后，输出了一系列字符，包含换行符。

2. `printf()`函数：

`printf()`函数是一个标准输出函数，用于在屏幕上打印输出。它的功能等同于在屏幕上写下一行文字。这里，`printf()`函数输出了一个换行符，意味着在输出结束后，屏幕上会显示一个换行符，使得输出的内容之间有一个明显的分隔符。

综合来看，这段代码的主要目的是在屏幕上输出换行符，使得显示结果以更加可读的方式展示了。


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


# `20_Buzzword/python/buzzword.py`

这段代码是一个用于生成 buzzword（流行语）的程序，通过提供一系列高度可接受的说法，帮助用户在演讲和简报中准备好教育技术相关的内容。通过输入有效的关键词，这段代码会生成一系列类似的流行语，从而使用户能够使用这些惯用的说法来表达自己的意思，而不会让听众觉得自己在胡说八道。代码中提到了这个程序是由 David H. Ahl 编写的。


```
"""
Buzzword Generator

From: BASIC Computer Games (1978)
      Edited by David H. Ahl

"This program is an invaluable aid for preparing speeches and
 briefings about education technology.  This buzzword generator
 provides sets of three highly-acceptable words to work into your
 material.  Your audience will never know that the phrases don't
 really mean much of anything because they sound so great!  Full
 instructions for running are given in the program.

"This version of Buzzword was written by David Ahl."


```

It looks like the buzzword generator program is designed to generate phrases in a specific vocabulary. The program uses a combination of learning and individualization to generate these phrases.

The program generates phrases in a specific format that appears to be designed for an educator-speak audience. The program also includes a feature to allow users to modify the generated phrases.

Overall, it seems like the buzzword generator program is intended for educational purposes and can be a useful tool for generating reports and speeches.


```
Python port by Jeff Jetton, 2019
"""


import random


def main() -> None:
    words = [
        [
            "Ability",
            "Basal",
            "Behavioral",
            "Child-centered",
            "Differentiated",
            "Discovery",
            "Flexible",
            "Heterogeneous",
            "Homogenous",
            "Manipulative",
            "Modular",
            "Tavistock",
            "Individualized",
        ],
        [
            "learning",
            "evaluative",
            "objective",
            "cognitive",
            "enrichment",
            "scheduling",
            "humanistic",
            "integrated",
            "non-graded",
            "training",
            "vertical age",
            "motivational",
            "creative",
        ],
        [
            "grouping",
            "modification",
            "accountability",
            "process",
            "core curriculum",
            "algorithm",
            "performance",
            "reinforcement",
            "open classroom",
            "resource",
            "structure",
            "facility",
            "environment",
        ],
    ]

    # Display intro text
    print("\n           Buzzword Generator")
    print("Creative Computing  Morristown, New Jersey")
    print("\n\n")
    print("This program prints highly acceptable phrases in")
    print("'educator-speak' that you can work into reports")
    print("and speeches.  Whenever a question mark is printed,")
    print("type a 'Y' for another phrase or 'N' to quit.")
    print("\n\nHere's the first phrase:")

    still_running = True
    while still_running:
        phrase = ""
        for section in words:
            if len(phrase) > 0:
                phrase += " "
            phrase += section[random.randint(0, len(section) - 1)]

        print(phrase)
        print()

        response = input("? ")
        try:
            if response.upper()[0] != "Y":
                still_running = False
        except Exception:
            still_running = False

    print("Come back when you need help with another report!\n")


```

这段代码的作用是定义了一个判断是否为__main__函数的判断块。如果当前进程是__main__函数，那么程序会执行__main__函数内部的代码。

在__main__函数内部，程序会创建一个39单词的列表，并将它们存储在同一个列表中。接下来，程序使用random.sample()函数从每个区域(1-13, 14-26，和 27-39)中随机选择一个单词，并将选择的单词添加到列表中。

最后，程序通过创建一个包含每个区域的列表来存储选择的单词。这使得程序可以轻松地通过循环每个区域来组合单词，以及通过添加或删除元素来管理每个区域。


```
if __name__ == "__main__":
    main()

######################################################################
#
# Porting Notes
#
#   The original program stored all 39 words in one array, then
#   built the buzzword phrases by randomly sampling from each of the
#   three regions of the array (1-13, 14-26, and 27-39).
#
#   Here, we're storing the words for each section in separate
#   tuples.  That makes it easy to just loop through the sections
#   to stitch the phrase together, and it easily accomodates adding
#   (or removing) elements from any section.  They don't all need to
```

这段代码是一个Python程序，它旨在创建一个名为“ EduBot”的教育机器人。该程序最初是由Creative Computing杂志的创始人之一在DEC（Digital Equipment Corporation）公司作为顾问帮助该公司市场其计算机作为教育产品时开发的。

随着时间的推移，程序还致力于编写名为“EDU”的DEC newsletter，重点关注将计算机用于教育环境。因此，该程序的主要目标是为教育工作者提供有用的想法和资源。

在程序中，有一个名为“Ideas for Modifications”的提示，它鼓励开发更多或不同的功能。此外，程序还建议添加一个第三维度到名为“WORDS”的单词元组中，以添加新的词汇集。

总的来说，这段代码是一个旨在为教育工作者提供有用资源和工具的教育机器人。


```
#   be the same length.
#
#   The author of this program (and founder of Creative Computing
#   magazine) first started working at DEC--Digital Equipment
#   Corporation--as a consultant helping the company market its
#   computers as educational products.  He later was editor of a DEC
#   newsletter named "EDU" that focused on using computers in an
#   educational setting.  No surprise, then, that the buzzwords in
#   this program were targeted towards educators!
#
#
# Ideas for Modifications
#
#   Try adding more/different words.  Better yet, add a third
#   dimnension to our WORDS tuple to add new sets of words that
```

这段代码是一个网页脚本，会向用户询问选择一个领域(如工程、艺术或音乐)，然后根据用户的选择自动生成相应的 buzzwords(术语、词汇或短语)。这些 buzzwords 将用于描述该领域的一些关键词或短语，例如 "工程 buzzwords"、"艺术 buzzwords" 或 "音乐 buzzwords"。用户可以选择一个或多个领域，然后脚本将根据用户的选择生成相应的 buzzwords。


```
#   might pertain to different fields.  What would business buzzwords
#   be? Engineering buzzwords?  Art/music buzzwords?  Let the user
#   choose a field and pick the buzzwords accordingly.
#
######################################################################

```

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


# `21_Calendar/csharp/Program.cs`

This appears to be a C# program that outputs the current date to the console. It uses a combination of paddding to add days to the current date and printing the current date in a specific format.

The program first defines the current date as a string and then formats it with some additional text, total days until the end of the year, and the number of days left until the middle of the year.

It then loops through the days of the current month and adds the appropriate padding to each day, up to a maximum of 18 days (based on the number of days in a month).

Finally, it outputs the current date in the required format and then loops through the weeks of the current year, adding padding to each week.

It should be noted that the program contains several loops which can be simplified and made more efficient.


```
﻿using System;

/*
 21_Calendar in C# for basic-computer-games
 Converted by luminoso-256
*/

namespace _21_calendar
{
    class Program
    {
        //basic has a TAB function. We do not by default, so we make our own!
        static string Tab(int numspaces)
        {
            string space = "";
            //loop as many times as there are spaces specified, and add a space each time
            while (numspaces > 0)
            {
                //add the space
                space += " ";
                //decrement the loop variable so we don't keep going forever!
                numspaces--;
            }
            return space;
        }

        static void Main(string[] args)
        {
            // print the "title" of our program
            // the usage of Write*Line* means we do not have to specify a newline (\n)
            Console.WriteLine(Tab(32) + "CALENDAR");
            Console.WriteLine(Tab(15) + "CREATE COMPUTING  MORRISTOWN, NEW JERSEY");
            //give us some space.
            Console.WriteLine("");
            Console.WriteLine("");
            Console.WriteLine("");

            //establish some variables needed to print out a calculator

            //the length of each month in days. On a leap year, the start of this would be
            // 0, 31, 29 to account for Feb. the 0 at the start is for days elapsed to work right in Jan.
            int[] monthLengths = { 0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}; // m in original source

            //the starting day of the month. in 1979 this was monday
            // 0 = sun, -1 = mon, -2 = tue, -3 = wed, etc.
            int day = -1; // called d in original source

            //how much time in the year has gone by?
            int elapsed = 0; // called s in original source

            //loop through printing all the months.
            for (int month = 1; month <= 12; month++) //month is called n in original source
            {
                //pad some space
                Console.WriteLine("");
                Console.WriteLine("");
                //increment days elapsed
                elapsed += monthLengths[month - 1];
                //build our header for this month of the calendar
                string header = "** " + elapsed;
                //add padding as needed
                while (header.Length < 7)
                {
                    header += " ";
                }
                for (int i = 1; i <= 18; i++)
                {
                    header += "*";
                }
                //determine what month it is, add text accordingly
                switch (month) {
                    case 1: header += " JANUARY "; break;
                    case 2: header += " FEBRUARY"; break;
                    case 3: header += "  MARCH  "; break;
                    case 4: header += "  APRIL  "; break;
                    case 5: header += "   MAY   "; break;
                    case 6: header += "   JUNE  "; break;
                    case 7: header += "   JULY  "; break;
                    case 8: header += "  AUGUST "; break;
                    case 9: header += "SEPTEMBER"; break;
                    case 10: header += " OCTOBER "; break;
                    case 11: header += " NOVEMBER"; break;
                    case 12: header += " DECEMBER"; break;
                }
                //more padding
                for (int i = 1; i <= 18; i++)
                {
                    header += "*";
                }
                header += "  ";
                // how many days left till the year's over?
                header += (365 - elapsed) + " **"; // on leap years 366
                Console.WriteLine(header);
                //dates
                Console.WriteLine("     S       M       T       W       T       F       S");
                Console.WriteLine(" ");

                string weekOutput = "";
                for (int i = 1; i <= 59; i++)
                {
                    weekOutput += "*";
                }
                //init some vars ahead of time
                int g = 0;
                int d2 = 0;
                //go through the weeks and days
                for (int week = 1; week <= 6; week++)
                {
                    Console.WriteLine(weekOutput);
                    weekOutput = "    ";
                    for (g = 1; g <= 7; g++)
                    {
                        //add one to the day
                        day++;
                        d2 = day - elapsed;
                        //check if we're done with this month
                        if (d2 > monthLengths[month])
                        {
                            week = 6;
                            break;
                        }
                        //should we print this day?
                        if (d2 > 0)
                        {
                            weekOutput += d2;
                        }
                        //padding
                        while (weekOutput.Length < 4 + 8 * g)
                        {
                            weekOutput += " ";
                        }
                    }
                    if (d2 == monthLengths[month])
                    {
                        day += g;
                        break;
                    }
                }
                day -= g;
                Console.WriteLine(weekOutput);
            }
        }
    }
}

```