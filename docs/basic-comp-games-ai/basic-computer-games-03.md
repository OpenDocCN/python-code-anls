# BasicComputerGames源码解析 3

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

# `00_Alternate_Languages/22_Change/go/main.go`

这段代码是一个用于输出欢迎消息的Java程序。它首先导入了需要的包(bufio, fmt, math, os, strconv)，然后定义了一个名为 printWelcome 的函数。

函数体内，使用 fmt 包中的 println 函数输出了一系列的欢迎消息，其中包括了一些关于计算机的一些信息(比如创造者、位置、颜色等)，以及一段欢迎的口号。

最后，它使用 strconv.Iee8848 包中的 bind可以将输出结果进行国际化，以便在不同的shell中能够正确显示。


```
package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
)

func printWelcome() {
	fmt.Println("                 CHANGE")
	fmt.Println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
	fmt.Println()
	fmt.Println()
	fmt.Println()
	fmt.Println("I, YOUR FRIENDLY MICROCOMPUTER, WILL DETERMINE")
	fmt.Println("THE CORRECT CHANGE FOR ITEMS COSTING UP TO $100.")
	fmt.Println()
}

```

It looks like the `fmt.Printf` calls are printing out the percentage change, if any, and then printing out the amount. The `float64(change)/-100.0` line is getting the absolute value of the change and then dividing it by 100, which is the percentage change. The `fmt.Printf` calls are then printing out the number of times the absolute value of the change is equal to one of the given amounts, such as "ten dollar bill" or "dollar". The `if d > 0` checks if the current amount is positive, if it is the format will be printed out, otherwise it will not be printed.


```
func computeChange(cost, payment float64) {
	change := int(math.Round((payment - cost) * 100))

	if change == 0 {
		fmt.Println("\nCORRECT AMOUNT, THANK YOU.")
		return
	}

	if change < 0 {
		fmt.Printf("\nSORRY, YOU HAVE SHORT-CHANGED ME $%0.2f\n", float64(change)/-100.0)
		print()
		return
	}

	fmt.Printf("\nYOUR CHANGE, $%0.2f:\n", float64(change)/100.0)

	d := change / 1000
	if d > 0 {
		fmt.Printf("  %d TEN DOLLAR BILL(S)\n", d)
		change -= d * 1000
	}

	d = change / 500
	if d > 0 {
		fmt.Printf("  %d FIVE DOLLAR BILL(S)\n", d)
		change -= d * 500
	}

	d = change / 100
	if d > 0 {
		fmt.Printf("  %d ONE DOLLAR BILL(S)\n", d)
		change -= d * 100
	}

	d = change / 50
	if d > 0 {
		fmt.Println("  1 HALF DOLLAR")
		change -= d * 50
	}

	d = change / 25
	if d > 0 {
		fmt.Printf("  %d QUARTER(S)\n", d)
		change -= d * 25
	}

	d = change / 10
	if d > 0 {
		fmt.Printf("  %d DIME(S)\n", d)
		change -= d * 10
	}

	d = change / 5
	if d > 0 {
		fmt.Printf("  %d NICKEL(S)\n", d)
		change -= d * 5
	}

	if change > 0 {
		fmt.Printf("  %d PENNY(S)\n", change)
	}
}

```

这段代码的主要作用是接受用户输入的商品成本和付款金额，并计算它们之间的差额。这个差额可以用来退还给用户。

具体来说，这段代码首先导入了 `bufio.NewScanner` 函数，它用于从标准输入（通常是键盘输入）中读取字符串。然后，它定义了一个 `printWelcome` 函数，用于输出欢迎消息。

接下来，代码使用一个循环来读取用户输入的商品成本和付款金额。在循环内部，它首先会问用户输入商品成本，然后使用 `strconv.ParseFloat` 函数将输入的字符串转换为浮点数并将其存储在 `cost` 变量中。如果转换失败或者成本小于 0，代码会输出一个错误消息并继续等待用户输入。

循环结束后，代码再次读取用户输入的付款金额，并使用 `strconv.ParseFloat` 函数将其转换为浮点数并将其存储在 `payment` 变量中。如果转换失败，代码会再次输出一个错误消息并继续等待用户输入。

接下来，代码调用了一个名为 `computeChange` 的函数，它接收两个参数 `cost` 和 `payment`。这个函数的作用是计算这两个数之间的差额，并将结果存储在 `diff` 变量中。最后，代码打印出计算出来的差额，并在循环结束后输出一个消息。


```
func main() {
	scanner := bufio.NewScanner(os.Stdin)

	printWelcome()

	var cost, payment float64
	var err error
	for {
		fmt.Println("COST OF ITEM?")
		scanner.Scan()
		cost, err = strconv.ParseFloat(scanner.Text(), 64)
		if err != nil || cost < 0.0 {
			fmt.Println("INVALID INPUT. TRY AGAIN.")
			continue
		}
		break
	}
	for {
		fmt.Println("\nAMOUNT OF PAYMENT?")
		scanner.Scan()
		payment, err = strconv.ParseFloat(scanner.Text(), 64)
		if err != nil {
			fmt.Println("INVALID INPUT. TRY AGAIN.")
			continue
		}
		break
	}

	computeChange(cost, payment)
	fmt.Println()
}

```

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

# `00_Alternate_Languages/25_Chief/C/chief.c`

这段代码包括三个函数：

1. `show_solution()` 函数，它接受一个浮点数猜测，并输出解决方案。具体来说，它将调用函数 `print_solution()` 来打印解决方案，并使用 `printf()` 函数来输出它。

2. `guess_number()` 函数，它接受一个浮点数猜测，并返回猜测的数值。具体来说，它将调用函数 `guess_number()` 来生成随机猜测值，并检查猜测值是否在 `min_guess` 和 `max_guess` 之间。

3. `game()` 函数，它包括全局变量和函数，以及一些通用的字符串常量。它将调用 `guess_number()` 和 `show_solution()` 函数，用于生成随机猜测值和输出解决方案。

具体来说，`show_solution()` 函数将打印出指定数值的解决方案，例如将屏幕上的所有字符清空并输出“CLRES”。`guess_number()` 函数将生成一个指定范围内的随机浮点数猜测值，例如尝试猜测 0 到 100 的随机数。`game()` 函数将在屏幕上随机生成一个 3x3 的猜测数字，并尝试找到一个解决方案，例如将屏幕上的所有字符清空并输出“CLRES”。


```
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

//check if windows or linux for the clear screen
#ifdef _WIN32
#define CLEAR "cls"
#else
#define CLEAR "clear"
#endif

void show_solution(float guess);
float guess_number(float number);
void game();

```

这段代码是一个简单的计算题题，它接受一个float类型的参数number，然后计算并返回一个float类型的猜测值。

guess_number()函数中，首先将number减去4，然后将结果乘以5，再将结果除以8，最后将结果取反并乘以5，这样就得到了一个围绕着3的浮点数。然后将其减去3，得到guess。

游戏()函数中，首先读取一个float类型的number，然后调用guess_number()函数来计算guess，接着输出guess，并等待用户输入答案。如果用户输入的答案是"yes"，那么程序会检查计算出的结果是否与用户输入的相同，如果是，输出一个笑话，并调用show_solution()函数来显示完整的解决方案。如果不是，那么程序会再次尝试计算，并输出一个笑话，如果还是没有成功，则程序会提示用户重新输入答案。如果用户输入的答案是"no"，那么程序会先尝试计算，并输出一个笑话，如果还是没有成功，则程序会再次尝试计算，并提示用户重新输入答案。如果用户输入的答案与计算结果相同，那么程序会输出一个笑话，并调用show_solution()函数来显示完整的解决方案。

总的来说，这段代码就是一个计算并输出数字的简单游戏，用户可以通过输入"yes"或"no"来猜测一个数字，并输出计算结果。


```
float guess_number(float number){
    float guess;
    guess = ((((number - 4) * 5) / 8) * 5 - 3);
    return guess;
}

void game(){
    float number,guess;
    char answer[4];
    printf("Think a number\n");
    printf("Then add to it 3 and divide it by 5\n");
    printf("Now multiply by 8, divide by 5 and then add 5\n");
    printf("Finally substract 1\n");
    printf("What is the number you got?(if you got decimals put them ex: 23.6): ");
    scanf("%f",&number);
    guess = guess_number(number);
    printf("The number you thought was %f am I right(Yes or No)?\n",guess);
    scanf("%s",answer);
    for(int i = 0; i < strlen(answer); i++){
        answer[i] = tolower(answer[i]);
    }
    if(strcmp(answer,"yes") == 0){
        printf("\nHuh, I Knew I was unbeatable");
        printf("And here is how i did it:\n");
        show_solution(guess);
    }
    else if (strcmp(answer,"no") == 0){
        printf("HUH!! what was you original number?: ");
        scanf("%f",&number);
        if(number == guess){
            printf("Huh, I Knew I was unbeatable");
            printf("And here is how i did it:\n");
            show_solution(guess);
        }
        else{
            printf("If I got it wrong I guess you are smarter than me");
        }
    }
    else{
        system(CLEAR);
        printf("I don't understand what you said\n");
        printf("Please answer with Yes or No\n");
        game();
    }

}

```



void show_solution(float guess){
   printf("%f plus 3 is %f\n",guess,guess + 3);
   printf("%f divided by 5 is %f\n",guess + 3,(guess + 3) / 5);
   printf("%f multiplied by 8 is %f\n",(guess + 3) / 5,(guess + 3) / 5 * 8);
   printf("%f divided by 5 is %f\n",(guess + 3) / 5 * 8,(guess + 3) / 5 * 8 / 5);
   printf("%f plus 5 is %f\n",(guess + 3) / 5 * 8 / 5,(guess + 3) / 5 * 8 / 5 + 5);
   printf("%f minus 1 is %f\n",(guess + 3) / 5 * 8 / 5 + 5,(guess + 3) / 5 * 8 / 5 + 5 - 1);
}

void game(){
   char answer[4];
   printf("Welcome to the CHIEF NUMBERS FreeK Indian Math Game!\n");
   printf("Are you ready to take the test you called me out for? (Yes or No) ");
   scanf("%s",answer);
   for(int i = 0; i < strlen(answer); i++){
       answer[i] = tolower(answer[i]);
   }
   if(strcmp(answer,"yes") == 0){
       show_solution(2.5);
   }else if (strcmp(answer,"no") == 0){
       printf("You are a coward, I will not play with you.%d %s\n",strcmp(answer,"yes"),answer);
   }
   else{
       system(CLEAR);
       printf("I don't understand what you said\n");
       printf("Please answer with Yes or No\n");
       game();
   }
}


```
void show_solution(float guess){
    printf("%f plus 3 is %f\n",guess,guess + 3);
    printf("%f divided by 5 is %f\n",guess + 3,(guess + 3) / 5);
    printf("%f multiplied by 8 is %f\n",(guess + 3) / 5,(guess + 3) / 5 * 8);
    printf("%f divided by 5 is %f\n",(guess + 3) / 5 * 8,(guess + 3) / 5 * 8 / 5);
    printf("%f plus 5 is %f\n",(guess + 3) / 5 * 8 / 5,(guess + 3) / 5 * 8 / 5 + 5);
    printf("%f minus 1 is %f\n",(guess + 3) / 5 * 8 / 5 + 5,(guess + 3) / 5 * 8 / 5 + 5 - 1);
}

void main(){
    char answer[4];
    printf("I am CHIEF NUMBERS FREEK, The GREAT INDIAN MATH GOD.\n");
    printf("Are you ready to take the test you called me out for(Yes or No)? ");
    scanf("%s",answer);
    for(int i = 0; i < strlen(answer); i++){
        answer[i] = tolower(answer[i]);
    }
    if(strcmp(answer,"yes") == 0){
        game();
    }else if (strcmp(answer,"no") == 0){
        printf("You are a coward, I will not play with you.%d %s\n",strcmp(answer,"yes"),answer);
    }
    else{
        system(CLEAR);
        printf("I don't understand what you said\n");
        printf("Please answer with Yes or No\n");
        main();
    }
}
```

# `00_Alternate_Languages/25_Chief/go/main.go`

这段代码的主要目的是打印一些带闪电的炫酷字符，以在终端或命令行界面上引起一些注意。

具体来说，这段代码实现了以下功能：

1. 打印 "************************************"
2. 循环打印 24 个 "x"，然后每两个 "x" 之间插入一个空格，形成一个类似于 "x x" 的形状
3. 打印 "x xxx"，然后每三个 "x" 之间插入一个空格，形成一个类似于 "x xxx" 的形状
4. 循环打印 8 个 "x"，然后每两个 "x" 之间插入一个空格，形成一个类似于 "xx xx" 的形状
5. 打印 "        xx"，然后每三个 "x" 之间插入一个空格，形成一个类似于 "xx" 的形状
6. 打印 "       x"，然后每两个 "x" 之间插入一个空格，形成一个类似于 "x" 的形状
7. 打印 "************************************"，然后添加一个空行，以便在终端或命令行界面上更容易阅读

总的来说，这段代码看起来像是在玩 "闪电单词" 游戏，每隔一段时间就会打印出一个 "闪电单词"，并在其中添加了一些空格，以使单词更加美观和易于阅读。


```
package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func printLightning() {
	fmt.Println("************************************")
	n := 24
	for n > 16 {
		var b strings.Builder
		b.Grow(n + 3)
		for i := 0; i < n; i++ {
			b.WriteString(" ")
		}
		b.WriteString("x x")
		fmt.Println(b.String())
		n--
	}
	fmt.Println("                x xxx")
	fmt.Println("               x   x")
	fmt.Println("              xx xx")
	n--
	for n > 8 {
		var b strings.Builder
		b.Grow(n + 3)
		for i := 0; i < n; i++ {
			b.WriteString(" ")
		}
		b.WriteString("x x")
		fmt.Println(b.String())
		n--
	}
	fmt.Println("        xx")
	fmt.Println("       x")
	fmt.Println("************************************")
}

```

This is an implementation of the classic "Impossible Problem" game. The game has two parts:

Part 1: The数学问题
Part 2: The problem with the number

Part 1 is a simple mathematical problem where the player has to guess a number, add 3 to it, divide by 5, multiply by 8, divide by 5, add the same number, and subtract 1. The output is the guessed number.

Part 2 is a problem where the player has to believe or not believe the answer. If the answer is correct, the player will get points, but if


```
func printSolution(n float64) {
	fmt.Printf("\n%f plus 3 gives %f. This divided by 5 equals %f\n", n, n+3, (n+3)/5)
	fmt.Printf("This times 8 gives %f. If we divide 5 and add 5.\n", ((n+3)/5)*8)
	fmt.Printf("We get %f, which, minus 1 equals %f\n", (((n+3)/5)*8)/5+5, ((((n+3)/5)*8)/5+5)-1)
}

func play() {
	fmt.Println("\nTake a Number and ADD 3. Now, Divide this number by 5 and")
	fmt.Println("multiply by 8. Now, Divide by 5 and add the same. Subtract 1")

	youHave := getFloat("\nWhat do you have?")
	compGuess := (((youHave-4)*5)/8)*5 - 3
	if getYesNo(fmt.Sprintf("\nI bet your number was %f was I right(Yes or No)? ", compGuess)) {
		fmt.Println("\nHuh, I knew I was unbeatable")
		fmt.Println("And here is how i did it")
		printSolution(compGuess)
	} else {
		originalNumber := getFloat("\nHUH!! what was you original number? ")
		if originalNumber == compGuess {
			fmt.Println("\nThat was my guess, AHA i was right")
			fmt.Println("Shamed to accept defeat i guess, don't worry you can master mathematics too")
			fmt.Println("Here is how i did it")
			printSolution(compGuess)
		} else {
			fmt.Println("\nSo you think you're so smart, EH?")
			fmt.Println("Now, Watch")
			printSolution(originalNumber)

			if getYesNo("\nNow do you believe me? ") {
				print("\nOk, Lets play again sometime bye!!!!")
			} else {
				fmt.Println("\nYOU HAVE MADE ME VERY MAD!!!!!")
				fmt.Println("BY THE WRATH OF THE MATHEMATICS AND THE RAGE OF THE GODS")
				fmt.Println("THERE SHALL BE LIGHTNING!!!!!!!")
				printLightning()
				fmt.Println("\nI Hope you believe me now, for your own sake")
			}
		}
	}
}

```

这段代码定义了一个名为 `getFloat` 的函数，接受一个字符串参数 `prompt`，并返回一个浮点数类型的值。

该函数首先通过 `bufio.NewScanner` 创建了一个输入缓冲区 `buf`，并使用 `os.Stdin` 作为输入来源，然后循环读取输入中的字符串。

在循环中，函数首先使用 `fmt.Println` 打印出一个带有 `prompt` 字符串的提示信息，然后使用 `scanner.Scan` 读取输入缓冲区中的所有字符。

接下来，函数调用 `strconv.ParseFloat` 函数将输入的字符串解析为浮点数类型，并将结果存储在变量 `val` 中。如果解析过程中出现错误，函数将打印出错误信息并继续循环读取输入。

最后，函数返回解析后的浮点数值 `val`。


```
func getFloat(prompt string) float64 {
	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Println(prompt)
		scanner.Scan()
		val, err := strconv.ParseFloat(scanner.Text(), 64)
		if err != nil {
			fmt.Println("INVALID INPUT, TRY AGAIN")
			continue
		}
		return val
	}
}

```

这段代码定义了一个名为 `getYesNo` 的函数，接受一个字符串参数 `prompt`。函数使用 `bufio.NewScanner` 函数从标准输入（通常是键盘）读取字符串，并输出一个布尔值。函数的逻辑是，将输入的字符串转换为大写后获取第一个字符，如果该字符为 "Y"，则返回 `true`，否则返回 `false`。

在 `main` 函数中，首先定义了一个字符串变量 `prompt`，并使用 `fmt.Println` 函数输出 "I am CHIEF NUMBERS FREEK, TheGreat INDIAN MATH God。"。然后使用 `getYesNo` 函数获取用户输入是否愿意参加一个测试，并将结果打印出来。如果用户输入是 "Yes"，则调用 `play` 函数执行播放音乐。否则，打印 "Ok, Nevermind. Let me go back to my great slumber, Bye"。


```
func getYesNo(prompt string) bool {
	scanner := bufio.NewScanner(os.Stdin)
	fmt.Println(prompt)
	scanner.Scan()

	return (strings.ToUpper(scanner.Text())[0:1] == "Y")

}

func main() {
	fmt.Println("I am CHIEF NUMBERS FREEK, The GREAT INDIAN MATH GOD.")

	if getYesNo("\nAre you ready to take the test you called me out for(Yes or No)? ") {
		play()
	} else {
		fmt.Println("Ok, Nevermind. Let me go back to my great slumber, Bye")
	}
}

```

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


# `00_Alternate_Languages/30_Cube/C/cube.c`

这段代码的主要作用是定义了一个名为coords的结构体类型，用于存储绘制图形时的坐标信息。

首先，代码检查当前系统是否为Windows还是Linux，如果是Windows，则定义了一个名为CLEAR的宏，表示清除屏幕；否则定义了一个名为CLEAR的宏，表示清除屏幕。这两个宏可以用来在程序中统一使用。

接着，定义了一个coords结构体类型，该结构体包含三个整型成员变量，分别代表横坐标、纵坐标和亮度。这个结构体可以用来在程序中存储和操作坐标信息。

最后，没有做其他事情，直接 export了两个函数，一个是void类型的main函数，还有一个是void类型的printf函数。


```
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//check if windows or linux for the clear screen
#ifdef _WIN32
#define CLEAR "cls"
#else
#define CLEAR "clear"
#endif

typedef struct{
    int x;
    int y;
    int z;
}coords;

```

这段代码是一个简单的游戏程序，它将在随机的决定中玩玩家与计算机之间的游戏。游戏的主要目的是让玩家通过射击地雷来获得胜利。计算机将在随机选择的位置放置地雷，如果玩家射击到这些位置，玩家就失败。玩家每个移动只能向一个方向进行，并且不能在同一位置改变两次移动。

当玩家要下注时，程序会输出一个数值作为投注金额，然后继续进行游戏。


```
void instuctions(){
    printf("\nThis is a game in which you will be playing against the\n");
    printf("random decisions of the computer. The field of play is a\n");
    printf("cube of side 3. Any of the 27 locations can be designated\n");
    printf("by inputing three numbers such as 2,3,1. At the start,\n");
    printf("you are automatically at location 1,1,1. The object of\n");
    printf("the game is to get to location 3,3,3. One minor detail:\n");
    printf("the computer will pick, at random, 5 locations at which\n");
    printf("it will plant land mines. If you hit one of these locations\n");
    printf("you lose. One other detail: You may move only one space\n");
    printf("in one direction each move. For example: From 1,1,2 you\n");
    printf("may move to 2,1,2 or 1,1,3. You may not change\n");
    printf("two of the numbers on the same move. If you make an illegal\n");
    printf("move, you lose and the computer takes the money you may\n");
    printf("have bet on that round.\n\n");
    printf("When stating the amount of a wager, printf only the number\n");
    printf("of dollars (example: 250) you are automatically started with\n");
    printf("500 dollars in your account.\n\n");
    printf("Good luck!\n");
}

```

It looks like you have a game where the player is trying to navigate through a grid of mines, and the first player to hit a mine wins the game. Here's a sample implementation in C:
```
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

#define MAX_MINE_COUNT 3
#define MAX_PLAYER_STAMPS 3

int mine[MAX_MINE_COUNT][MAX_PLAYER_STAMPS];
int player[MAX_PLAYER_STAMPS];
int playerold[MAX_PLAYER_STAMPS];
int wager;
int account;
int gameover;
int score;
int row, col, minrow, maxrow, mincol, maxcol;

void dfs(int row, int col, int node, int player){
   if(row == MAX_PLAYER_STAMPS || row < 1 || row > MAX_PLAYER_STAMPS || col == MAX_MINE_COUNT || col < 1 || col > MAX_MINE_COUNT || node == MAX_PLAYER_STAMPS || node < 1 || node > MAX_PLAYER_STAMPS){
       return;
   }
   if(mine[row][col] != -1){
       int dx = col - node;
       int dy = row - minrow;
       if(dx == 0 && dy == 0){
           mine[row][col] = -1;
           score++;
           printf("You hit a mine and added $%d to your score.\n", score);
           if(gameover == 0){
               gameover = 1;
               printf("Game over and you win!\n");
               printf("You win by %d points.", score);
               printf("The game ends.\n", score);
               exit(0);
           }
       }
       else if(dx == -1 && dy == 0){
           mine[row][col] = -1;
           score++;
           printf("You hit a mine and added $%d to your score.\n", score);
           if(gameover == 0){
               gameover = 1;
               printf("Game over and you win!\n");
               printf("You win by %d points.", score);
               printf("The game ends.\n", score);
               exit(0);
           }
       }
       else if(dx != 0 && dy != 0){
           score++;
           printf("You hit a mine and added $%d to your score.\n", score);
           if(gameover == 0){
               gameover = 1;
               printf("Game over and you win!\n");
               printf("You win by %d points.", score);
               printf("The game ends.\n", score);
               exit(0);
           }
       }
   }
}


int main(){
   int wager = 100;
   int i, j, k, row, col;
   int minrow = 0, maxrow = 0, mincol = 0, maxcol = 0;
   int node = 0;
   int gameover = 0;
   int score = 0;
   
   while(1){
       printf("Enter player number: ");
       scanf("%d", &i);
       
       if(i < 1 || i > MAX_PLAYER_STAMPS){
           printf("Invalid player number. Try again.\n");
           continue;
       }
       
       for(int j=0; j<MAX_MINE_COUNT; j++){
           for(int k=0; k<MAX_PLAYER_STAMPS; k++){
               mine[j][k] = -1;
               printf("Mine %d (%d,%d) is在内...", j+1, minrow, maxrow, mincol, maxcol);
               
               if(row == minrow && col == mincol){
                   printf("Mine is under your feet. You can't dig deeper.\n");
                   continue;
               }
               
               if(row == MAX_PLAYER_STAMPS || row < minrow || row > MAX_PLAYER_STAMPS || col == maxcol || col < mincol){
                   printf("Mine is out of bounds. You can't dig there.\n");
                   continue;
               }
               
               if(row < minrow || row > maxrow || col < mincol || col > maxcol){
                   printf("Mine is out of bounds. You can't dig there.\n");
                   continue;
               }
               
               if(mine[j][k] == -1){
                   dfs(row, col, node, -1);
                   if(gameover == 0){
                       score++;
                       printf("Game over and you win!\n");
                       printf("You win by %d points.", score);
                       printf("The game ends.\n", score);
                       exit(0);
                   }
                   break;
               }
               
               if(mine[j][k] == 3){
                   gameover = 1;
                   printf("Game over and you lose!\n");
                   printf("You lose by %d points.", wager);
                   continue;
               }
               
               if(i < 1 || i > MAX_PLAYER_STAMPS || j < minrow || j > MAX_PLAYER_STAMPS || col < mincol || col > MAX_COL || row < minrow || row > MAX_ROW || minrow < mincol || mincol < 1 || maxrow < maxcol || row < minrow || row > maxrow){
                   printf("Invalid move.\n");
                   continue;
               }
               
               if(row == minrow && col == mincol){
                   mine[j][k] = -1;
                   printf("You hit a mine!\n");
                   printf("You lose by %d points.", wager);
                   continue;
               }
               
               if(row < minrow || row > maxrow || col < mincol || col > maxcol){
                   printf("Mine is out of bounds. You can't dig deeper.\n");
                   continue;
               }
               
               if(col == maxcol || row == maxrow || minrow == 0 || mincol == 0 || row ==


```
void game(int money){
    coords player,playerold,mines[5];
    int wager,account = money;
    char choice;
    if(money == 0){
        printf("You have no money left. See ya next time.\n");
        exit(0);
    }
    player.x = 1;
    player.y = 1;
    player.z = 1;
    
    printf("You have $%d in your account.\n",account);
    printf("How much do you want to wager? ");
    scanf("%d",&wager);
    while(wager > account){
        system(CLEAR);
        printf("You do not have that much money in your account.\n");
        printf("How much do you want to wager? ");
        scanf("%d",&wager);
    }
    srand(time(NULL));
    for(int i=0;i<5;i++){
        mines[i].x = rand()%3+1;
        mines[i].y = rand()%3+1;
        mines[i].z = rand()%3+1;
        if(mines[i].x == 3 && mines[i].y == 3 && mines[i].z == 3){
            i--;
        }
    }
    while(player.x != 3 || player.y != 3 || player.z != 3){
        printf("You are at location %d.%d.%d\n",player.x,player.y,player.z);
        if(player.x == 1 && player.y == 1 && player.z == 1)
        printf("Enter new location(use commas like 1,1,2 or else the program will break...): ");
        else printf("Enter new location: ");
        playerold.x = player.x;
        playerold.y = player.y;
        playerold.z = player.z;
        scanf("%d,%d,%d",&player.x,&player.y,&player.z);
        if(((player.x + player.y + player.z) > (playerold.x + playerold.y + playerold.z + 1)) || ((player.x + player.y + player.z) < (playerold.x + playerold.y + playerold.z -1))){
            system(CLEAR);
            printf("Illegal move!\n");
            printf("You lose $%d.\n",wager);
            game(account -= wager);
            break;
        }
        if(player.x < 1 || player.x > 3 || player.y < 1 || player.y > 3 || player.z < 1 || player.z > 3){
            system(CLEAR);
            printf("Illegal move. You lose!\n");
            game(account -= wager);
            break;
        }
        for(int i=0;i<5;i++){
            if(player.x == mines[i].x && player.y == mines[i].y && player.z == mines[i].z){
                system(CLEAR);
                printf("You hit a mine!\n");
                printf("You lose $%d.\n",wager);
                game(account -= wager);
                exit(0);
            }
        }
        if(account == 0){
            system(CLEAR);
            printf("You have no money left!\n");
            printf("Game over!\n");
            exit(0);
        }
    }
    if(player.x == 3 && player.y == 3 && player.z == 3){
        system(CLEAR);
        printf("You made it to the end. You win!\n");
        game(account += wager);
        exit(0);
    }
}

```

这段代码是一个 Cube 游戏的程序。它首先初始化了一个账户为 500 的整数变量 account，并从用户那里获取了一个字符选择 'y' 或 'n'，用于选择是查看游戏说明还是直接开始游戏。

如果用户选择 'y'，那么程序会先清除屏幕并调用 instructions() 函数。如果用户选择 'n'，那么程序会直接开始游戏并清除屏幕。如果用户选择其他任何字符，程序会再次清除屏幕，并尝试重新初始化账户并继续运行。

在 initialize() 函数中，程序创建了一个空账户，并将其命名为 500。


```
void init(){
    int account = 500;
    char choice;

    printf("Welcome to the game of Cube!\n");
    printf("wanna see the instructions? (y/n): ");
    scanf("%c",&choice);
    if(choice == 'y'){
        system(CLEAR);
        instuctions();
    }
    else if (choice == 'n'){
        system(CLEAR);
        printf("Ok, let's play!\n");
    }
    else{
        system(CLEAR);
        printf("Invalid choice. Try again...\n");
        init();
        exit(0);
    }
    game(account);
    exit(0);
}

```

这是一个C语言的函数，名为“main”。函数内部包含了一些初始化代码，然后根据函数名，我们可以猜测这个函数可能是作为程序的入口点。因此，我们可以将一些简单的代码放在main函数中。

初始化代码：
```
void init(){
   // 一些初始化代码，具体内容未知
}
```

这个代码段内部可能包含一些初始化操作，但是由于我们不知道具体内容，所以我们无法提供更多的解释。


```
void main(){
    init();
}
```

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

# `00_Alternate_Languages/31_Depth_Charge/go/main.go`

该代码定义了一个名为Position的类型，它代表了一个二维平面中的位置。同时，该代码还定义了一个名为Main的函数和一些函数指针，函数指针分别为：fmt.Printf、math.rand、math.min、os.Exit、strconv.Icmp、strings.Trim、time.Now。

Main函数的作用是接受一个二维平面中由整数组成的字符串（可能是从用户输入中获取），然后计算出该平面中所有位置的计数器。具体实现是，首先将输入的字符串转换成整数，然后遍历每个位置，对于每个位置，计算它周围的格子数量，并将这些数量加到一个名为Position的数组中。最后，Main函数还计算出该二维平面中所有位置的计数器的总和，即为结果。

fmt.Printf函数用于将一个格式化字符串输出到屏幕上，例如打印"Hello World"。

math.rand函数用于生成一个0到1之间的随机数。

math.min函数用于计算输入参数的最小值。

os.Exit函数用于退出程序。

strconv.Icmp函数用于将输入的字符串转换成整数。

strings.Trim函数用于从输入字符串中移除指定的前缀和后缀，例如去除" Hello World "中的前缀和后缀后，返回的字符串为"hello"。

time.Now函数用于获取当前的时间。


```
package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

type Position []int

```

这段代码定义了两个函数。第一个函数名为 `NewPosition`，返回一个 `Position` 类型的变量 `p`。第二个函数名为 `showWelcome`，用于输出欢迎消息。第三个函数名为 `getNumCharges`，返回两个整数类型的值 `dim` 和 `int(math.Log2(float64(dim))) + 1`。

具体来说，第一个函数 `NewPosition` 创建了一个包含三个整数类型的数组 `p`，然后返回了该数组。第二个函数 `showWelcome` 在屏幕上打印出 "DEPTH CHARGE" 和一些公司信息，然后等待用户按回车键继续执行。第三个函数 `getNumCharges` 通过从标准输入中读取用户输入并转换为整数类型的函数，然后计算给定区域搜索区域的对角线长度。


```
func NewPosition() Position {
	p := make([]int, 3)
	return Position(p)
}

func showWelcome() {
	fmt.Print("\033[H\033[2J")
	fmt.Println("                DEPTH CHARGE")
	fmt.Println("    Creative Computing  Morristown, New Jersey")
	fmt.Println()
}

func getNumCharges() (int, int) {
	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Println("Dimensions of search area?")
		scanner.Scan()
		dim, err := strconv.Atoi(scanner.Text())
		if err != nil {
			fmt.Println("Must enter an integer number. Please try again...")
			continue
		}
		return dim, int(math.Log2(float64(dim))) + 1
	}
}

```

这段代码定义了两个函数：askForNewGame和showShotResult。

函数askForNewGame的作用是从标准输入（os.Stdin）获取用户输入，并输出一个包含"Another game (Y or N)"的提示。如果用户输入 "Y"，则会调用函数main，否则会输出"OK. Hope you enjoyed yourself"，并退出程序。

函数showShotResult的作用是接收一个射影（如三维坐标）和一个位置（如二维坐标），并输出与该射影的距离的 shot 报告。它首先根据射影的方向（y-或x-方向）来计算出距离，然后计算深度。如果射影的位置与位置的距离小于零，说明射影可能碰到地下，如果距离大于零，则说明射影可能碰到地上，如果距离正好，则说明射影正好落在位置。


```
func askForNewGame() {
	scanner := bufio.NewScanner(os.Stdin)

	fmt.Println("Another game (Y or N): ")
	scanner.Scan()
	if strings.ToUpper(scanner.Text()) == "Y" {
		main()
	}
	fmt.Println("OK. Hope you enjoyed yourself")
	os.Exit(1)
}

func showShotResult(shot, location Position) {
	result := "Sonar reports shot was "

	if shot[1] > location[1] { // y-direction
		result += "north"
	} else if shot[1] < location[1] { // y-direction
		result += "south"
	}

	if shot[0] > location[0] { // x-direction
		result += "east"
	} else if shot[0] < location[0] { // x-direction
		result += "west"
	}

	if shot[1] != location[1] || shot[0] != location[0] {
		result += " and "
	}
	if shot[2] > location[2] {
		result += "too low."
	} else if shot[2] < location[2] {
		result += "too high."
	} else {
		result += "depth OK."
	}

	fmt.Println(result)
}

```

这段代码是一个函数 `getShot()`，它返回一个 `Position` 类型的变量，表示用户的射门位置。

该函数使用了 `bufio.NewScanner` 函数从标准输入（通常是键盘）读取输入。函数内部使用循环来读取输入，循环变量从0到2，因为每行可能包含一个或三个坐标，所以循环体内部使用 `fmt.Println` 函数打印提示信息，要求用户输入每个坐标，然后使用 `scanner.Scan()` 函数读取用户输入，并将其存储在 `rawGuess` 数组中。如果用户输入不正确（比如没有输入或输入不是三个数字），函数跳转到 `there` 标签那里，然后输出一条错误消息并继续等待用户的输入。

函数最终返回 `shotPos` 变量，该变量包含三个 `Position` 类型的元素，每个元素表示射门的三个不同位置（列坐标从0到2）。


```
func getShot() Position {
	scanner := bufio.NewScanner(os.Stdin)

	for {
		shotPos := NewPosition()
		fmt.Println("Enter coordinates: ")
		scanner.Scan()
		rawGuess := strings.Split(scanner.Text(), " ")
		if len(rawGuess) != 3 {
			goto there
		}
		for i := 0; i < 3; i++ {
			val, err := strconv.Atoi(rawGuess[i])
			if err != nil {
				goto there
			}
			shotPos[i] = val
		}
		return shotPos
	there:
		fmt.Println("Please enter coordinates separated by spaces")
		fmt.Println("Example: 3 2 1")
	}
}

```

这段代码定义了两个函数，一个是 `getRandomPosition`，另一个是 `playGame`。

`getRandomPosition` 函数的作用是生成一个指定区域内的随机位置，并返回该位置。该函数使用了 `rand` 包中的 `randint` 函数，用于生成一个介于指定区域范围内的随机整数。

`playGame` 函数的作用是模拟玩家在游戏中的操作，包括输入命令、确定深度计爆炸点、进行攻击等。该函数会根据玩家输入的信息，生成相应的攻击结果，并判断游戏是否结束。如果游戏结束，函数会输出 "You have been torpedoed! Abandon ship!"，并再次要求玩家输入新的游戏深度。

`playGame` 函数的具体实现可以分为以下几个步骤：

1. 确定搜索区域（即海洋中的位置）。
2. 根据搜索区域，使用 `getRandomPosition` 函数生成指定区域内的随机位置。
3. 根据搜索区域，使用 `getShot` 函数获取该区域内的所有攻击目标的位置。
4. 根据获取到的攻击目标，判断是否成功发现并标记目标位置。
5. 根据攻击目标是否成功标记，决定是否继续进行下一轮攻击。
6. 如果没有成功标记攻击目标，而是所有深度计都指向同一位置，那么游戏结束，函数输出 "You have been torpedoed! Abandon ship!"，并再次要求玩家输入新的游戏深度。


```
func getRandomPosition(searchArea int) Position {
	pos := NewPosition()
	for i := 0; i < 3; i++ {
		pos[i] = rand.Intn(searchArea)
	}
	return pos
}

func playGame(searchArea, numCharges int) {
	rand.Seed(time.Now().UTC().UnixNano())
	fmt.Println("\nYou are the captain of the destroyer USS Computer.")
	fmt.Println("An enemy sub has been causing you trouble. Your")
	fmt.Printf("mission is to destroy it. You have %d shots.\n", numCharges)
	fmt.Println("Specify depth charge explosion point with a")
	fmt.Println("trio of numbers -- the first two are the")
	fmt.Println("surface coordinates; the third is the depth.")
	fmt.Println("\nGood luck!")
	fmt.Println()

	subPos := getRandomPosition(searchArea)

	for c := 0; c < numCharges; c++ {
		fmt.Printf("\nTrial #%d\n", c+1)

		shot := getShot()

		if shot[0] == subPos[0] && shot[1] == subPos[1] && shot[2] == subPos[2] {
			fmt.Printf("\nB O O M ! ! You found it in %d tries!\n", c+1)
			askForNewGame()
		} else {
			showShotResult(shot, subPos)
		}
	}

	// out of depth charges
	fmt.Println("\nYou have been torpedoed! Abandon ship!")
	fmt.Printf("The submarine was at %d %d %d\n", subPos[0], subPos[1], subPos[2])
	askForNewGame()

}

```

这段代码的主要作用是解释欢迎用户的游戏。函数 `showWelcome()` 将在程序开始时执行，它主要输出一个欢迎消息。

函数 `getNumCharges()` 获取用户输入的充电区域数量，并将结果存储在变量 `numCharges` 中。

函数 `playGame(searchArea, numCharges)` 将在 `numCharges` 的值为非零值时执行。它接收两个参数 `searchArea` 和 `numCharges`，并将它们存储在变量中。这个函数的作用在题目中没有给出，但我假设它是用于搜索特定区域中的充电区域数量，并根据用户输入的数量返回一个游戏结果。


```
func main() {
	showWelcome()

	searchArea, numCharges := getNumCharges()

	playGame(searchArea, numCharges)
}

```

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

# `00_Alternate_Languages/33_Dice/C/dice.c`

这段代码是一个C语言程序，用于模拟 rolling 这对双掷骰子。程序中首先定义了一个名为 dice1 和 dice2 的变量，用于跟踪两个骰子的点数，以及一个名为 rolls 的数组，用于跟踪每个点数出现的次数。

程序中还定义了一个名为 percent 的函数，它接受两个参数，一个整数 number，表示其中一个骰子，另一个整数 total，表示这两个骰子。函数计算出 number 除以 total 再乘以 100 得到的结果，表示这个点数出现的百分比。

在 main 函数中，程序首先使用一个循环来生成两个骰子，并计算出每个点数出现的次数，存储在 rolls 数组中。然后，程序使用另一个循环输出每个点数出现的次数，其中第一个参数 i 表示要输出的点数，第二个参数 rolles[i] 表示该点数出现的次数，第三个参数是输出时使用的格式字符串，用于输出该点数出现的次数以及百分比形式。

程序中使用了一个名为 srand 的函数，用于设置程序的随机数种子。这个函数接受一个指向时间的指针，用于获取当前时间，并将其作为参数传递给 srand，以获取随机数种子。

在模拟 rolling 这对双掷骰子的过程中，程序首先使用 rand() 函数生成两个随机整数，表示两个骰子的点数。然后，程序使用这两个随机整数计算出每个点数出现的次数，并将其存储在 rolls 数组中。程序在循环中从 dice1 开始，枚举所有可能的点数组合，生成更多的 rolles 数组元素，并将它们存储在 rolls 数组中。程序在循环结束后，使用 printf() 函数输出每个点数出现的次数，其中第一个参数 i 是要输出的点数，第二个参数 rolles[i] 是该点数出现的次数，第三个参数是输出时使用的格式字符串，用于输出该点数出现的次数以及百分比形式。


```
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float percent(int number, int total){
    float percent;
    percent = (float)number / (float)total * 100;
    return percent;
}

int main(){
    int dice1,dice2,times,rolls[13] = {0};
    srand(time(NULL));
    printf("This program simulates the rolling of a pair of dice\n");
    printf("How many times do you want to roll the dice?(Higher the number longer the waiting time): ");
    scanf("%d",&times);
    for(int i = 0; i < times; i++){
        dice1 = rand() % 6 + 1;
        dice2 = rand() % 6 + 1;
        rolls[dice1 + dice2]+=1;
    }
    printf("The number of times each sum was rolled is:\n");
    for(int i = 2; i <= 12; i++){
        printf("%d: rolled %d times, or %f%c of the times\n",i,rolls[i],percent(rolls[i],times),(char)37);
    }
}
```

# `00_Alternate_Languages/33_Dice/go/main.go`

这段代码是一个用于模拟 rolling 一对骰子的程序。程序的主要作用是输出一个欢迎消息，并在消息中告诉用户输入的次数、骰子面数以及大数提示。

具体来说，程序首先导入了需要的库，包括 `bufio`、`fmt`、`math/rand`、`os` 和 `strconv`、`strings`。然后定义了一个名为 `printWelcome` 的函数，该函数使用 `fmt` 库将字符串内容输出到控制台。

`printWelcome` 函数首先输出一个具有特定格式的字符串，该格式包含程序的名称和一些关于模拟滚骰子的提示。然后，函数使用 `fmt.Println` 函数将该字符串输出。

接下来，函数开始逐行输出一些关于模拟滚骰子的提示。函数的第一个输出是 "You entered the number of times you want the computer to 'roll' the dice."。接下来，函数会提示用户输入一个整数，该整数将决定计算机模拟多少次滚骰子。然后，函数再次提示用户输入的数字不能太大，因为这样会花很长时间。

最后，函数使用 `fmt.Println` 函数将剩余的消息输出到控制台，并在消息末尾添加了一个换行符。


```
package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
)

func printWelcome() {
	fmt.Println("\n                   Dice")
	fmt.Println("Creative Computing  Morristown, New Jersey")
	fmt.Println()
	fmt.Println()
	fmt.Println("This program simulates the rolling of a")
	fmt.Println("pair of dice.")
	fmt.Println("You enter the number of times you want the computer to")
	fmt.Println("'roll' the dice.   Watch out, very large numbers take")
	fmt.Println("a long time.  In particular, numbers over 5000.")
	fmt.Println()
}

```

这段代码的作用是读取用户输入的点数，并打印出相应的结果。它首先导入了 bufio.NewScanner 和 strconv.Atoi 函数，用于从标准输入流中读取用户输入并转换成整数。

然后，代码使用一个 13 元素的一维数组 results 来存储每个试验中的点数。通过循环来遍历所有的试验，计算出每个试验点数的和，并将结果显示出来。最后，它会提示用户再试一次，如果用户输入 "Y"，则跳过所有试验并退出程序，否则程序会继续执行。


```
func main() {
	printWelcome()
	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Println("\nHow many rolls? ")
		scanner.Scan()
		numRolls, err := strconv.Atoi(scanner.Text())
		if err != nil {
			fmt.Println("Invalid input, try again...")
			continue
		}

		// We'll track counts of roll outcomes in a 13-element list.
		// The first two indices (0 & 1) are ignored, leaving just
		// the indices that match the roll values (2 through 12).
		results := make([]int, 13)

		for n := 0; n < numRolls; n++ {
			d1 := rand.Intn(6) + 1
			d2 := rand.Intn(6) + 1
			results[d1+d2] += 1
		}

		// Display final results
		fmt.Println("\nTotal Spots   Number of Times")
		for i := 2; i < 13; i++ {
			fmt.Printf(" %-14d%d\n", i, results[i])
		}

		fmt.Println("\nTry again? ")
		scanner.Scan()
		if strings.ToUpper(scanner.Text()) == "Y" {
			continue
		} else {
			os.Exit(1)
		}

	}
}

```