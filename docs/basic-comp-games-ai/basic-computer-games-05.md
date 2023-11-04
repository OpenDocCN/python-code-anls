# BasicComputerGames源码解析 5

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

# `00_Alternate_Languages/41_Guess/go/main.go`

这段代码是一个程序，名为“main”。它导入了多个必要的包(package)，包括“bufio”(用于输入/输出),“fmt”(用于格式化输出),“math”(用于数学计算),“math/rand”(用于生成随机数),“os”(用于操作系统交互)和“strconv”(用于字符串转换)。

接下来，它定义了一个名为“printIntro”的函数。该函数使用“fmt”包中的格式化输出来打印一些文本，包括程序的名称和一些描述性的信息。然后，它使用“time”包中的“time.sleep”函数来暂停程序的执行一段时间(这里是5秒钟)，然后再恢复执行。

最后，它通过调用自己来创建一个名为“main”的包，并导入了该包中定义的所有函数和变量。


```
package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"
)

func printIntro() {
	fmt.Println("                   Guess")
	fmt.Println("Creative Computing  Morristown, New Jersey")
	fmt.Println()
	fmt.Println()
	fmt.Println()
	fmt.Println("This is a number guessing game. I'll think")
	fmt.Println("of a number between 1 and any limit you want.")
	fmt.Println("Then you have to guess what it is")
}

```

这段代码使用了两个主要的函数，`getLimit` 和 `fmt.Println`。

首先，我们来解释 `getLimit` 函数的作用。这个函数返回两个整数，一个是整数类型，一个是浮点数类型。函数使用了 `bufio.NewScanner` 函数从标准输入（通常是键盘）读取输入。在循环中，函数要求用户输入一个限制（0到正无穷大）。然后，使用 `strconv.Atoi` 函数将输入的字符串转换为整数类型。如果转换成功，就打印一条消息并允许用户继续输入。如果转换失败，或者用户输入的不是数字（可能是其他字符，如空格和换行），函数就会输出一条消息并停止输入。

接下来，我们来解释 `fmt.Println` 函数的作用。这个函数将 `getLimit` 函数得到的限制作为参数，并将其输出到标准输出（通常是屏幕）。函数会打印一条消息，要求用户输入一个数字以作为限制。


```
func getLimit() (int, int) {
	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Println("What limit do you want?")
		scanner.Scan()

		limit, err := strconv.Atoi(scanner.Text())
		if err != nil || limit < 0 {
			fmt.Println("Please enter a number greater or equal to 1")
			continue
		}

		limitGoal := int((math.Log(float64(limit)) / math.Log(2)) + 1)
		return limit, limitGoal
	}

}

```

It looks like the code is implementing a guessing game where the user is asked to guess a number between 1 and a given limit, and based on the猜测， the game will provide feedback to the user to help them guess a better number. The code uses the `bufio` and `os` packages to read input from the user, and the `guessCount` variable keeps track of how many guesses the user has made.


```
func main() {
	rand.Seed(time.Now().UnixNano())
	printIntro()

	scanner := bufio.NewScanner(os.Stdin)

	limit, limitGoal := getLimit()

	guessCount := 1
	stillGuessing := true
	won := false
	myGuess := int(float64(limit)*rand.Float64() + 1)

	fmt.Printf("I'm thinking of a number between 1 and %d\n", limit)
	fmt.Println("Now you try to guess what it is.")

	for stillGuessing {
		scanner.Scan()
		n, err := strconv.Atoi(scanner.Text())
		if err != nil {
			fmt.Println("Please enter a number greater or equal to 1")
			continue
		}

		if n < 0 {
			break
		}

		fmt.Print("\n\n\n")
		if n < myGuess {
			fmt.Println("Too low. Try a bigger answer")
			guessCount += 1
		} else if n > myGuess {
			fmt.Println("Too high. Try a smaller answer")
			guessCount += 1
		} else {
			fmt.Printf("That's it! You got it in %d tries\n", guessCount)
			won = true
			stillGuessing = false
		}
	}

	if won {
		if guessCount < limitGoal {
			fmt.Println("Very good.")
		} else if guessCount == limitGoal {
			fmt.Println("Good.")
		} else {
			fmt.Printf("You should have been able to get it in only %d guesses.\n", limitGoal)
		}
		fmt.Print("\n\n\n")
	}
}

```

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

# `00_Alternate_Languages/42_Gunner/go/main.go`

这段代码是一个名为main的包，其中定义了一个名为printIntro的函数。

printIntro函数会输出一段类似于这样内容的文本：

```
这是一段多行文本，第一行是开发者自己的介绍，第二行是软件的名称和版本号，接下来是时间和操作系统，然后是用于提醒开发者的目标和要求，接着是程序将执行的任务，然后是目标要处的位置和攻击的方式，最后是开发者名称和一段鼓励的话。
```

这段文本是在向用户介绍程序用途和目标的同时，告诉用户程序将要做什么。它类似于在向用户介绍一个游戏时要告诉用户游戏将如何进行，以及如何与游戏交互。


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

func printIntro() {
	fmt.Println("                                 GUNNER")
	fmt.Println("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
	fmt.Print("\n\n\n")
	fmt.Println("YOU ARE THE OFFICER-IN-CHARGE, GIVING ORDERS TO A GUN")
	fmt.Println("CREW, TELLING THEM THE DEGREES OF ELEVATION YOU ESTIMATE")
	fmt.Println("WILL PLACE A PROJECTILE ON TARGET.  A HIT WITHIN 100 YARDS")
	fmt.Println("OF THE TARGET WILL DESTROY IT.")
	fmt.Println()
}

```

这段代码定义了一个名为 `getFloat` 的函数，其返回类型为 `float64` 类型。该函数使用了 `bufio.NewScanner` 来从标准输入（通常是键盘）读取字符串，并使用 `strconv.ParseFloat` 函数将读取的字符串转换为浮点数。如果转换过程中出现错误，函数会打印错误消息并退出。

具体来说，这段代码的作用是读取用户从标准输入中输入的一个字符串，然后将该字符串转换为浮点数类型。如果转换成功，函数会返回转换后的浮点数。


```
func getFloat() float64 {
	scanner := bufio.NewScanner(os.Stdin)
	for {
		scanner.Scan()
		fl, err := strconv.ParseFloat(scanner.Text(), 64)

		if err != nil {
			fmt.Println("Invalid input")
			continue
		}

		return fl
	}
}

```

This appears to be a game of 《*** THREETH魄寸ovesWITH AMMUNITION ***》 (Valorant). The game is written inC++ and uses theOpenGL束引擎。

It的主要目标是摧毁敌人并打击敌人，通过破坏敌人的据点来获得游戏胜利。玩家可以移动和开火，并使用榴弹或其他武器来攻击敌人。

在每一轮中，玩家可以选择要摧毁的敌人或攻击敌人。敌人被摧毁或攻击时，玩家可以获得经验值，并有机会升级。玩家可以在一个新手的支持下继续游戏，或者在团队中与其他玩家一起作战。

如果您想了解更多关于这个游戏的更多信息，可以尝试搜索相关信息。


```
func play() {
	gunRange := int(40000*rand.Float64() + 20000)
	fmt.Printf("\nMAXIMUM RANGE OF YOUR GUN IS %d YARDS\n", gunRange)

	killedEnemies := 0
	S1 := 0

	for {
		targetDistance := int(float64(gunRange) * (0.1 + 0.8*rand.Float64()))
		shots := 0

		fmt.Printf("\nDISTANCE TO THE TARGET IS %d YARDS\n", targetDistance)

		for {
			fmt.Print("\n\nELEVATION? ")
			elevation := getFloat()

			if elevation > 89 {
				fmt.Println("MAXIMUM ELEVATION IS 89 DEGREES")
				continue
			}

			if elevation < 1 {
				fmt.Println("MINIMUM ELEVATION IS 1 DEGREE")
				continue
			}

			shots += 1

			if shots < 6 {
				B2 := 2 * elevation / 57.3
				shotImpact := int(float64(gunRange) * math.Sin(B2))
				shotProximity := int(targetDistance - shotImpact)

				if math.Abs(float64(shotProximity)) < 100 { // hit
					fmt.Printf("*** TARGET DESTROYED *** %d ROUNDS OF AMMUNITION EXPENDED.\n", shots)
					S1 += shots

					if killedEnemies == 4 {
						fmt.Printf("\n\nTOTAL ROUNDS EXPENDED WERE: %d\n", S1)
						if S1 > 18 {
							print("BETTER GO BACK TO FORT SILL FOR REFRESHER TRAINING!")
							return
						} else {
							print("NICE SHOOTING !!")
							return
						}
					} else {
						killedEnemies += 1
						fmt.Println("\nTHE FORWARD OBSERVER HAS SIGHTED MORE ENEMY ACTIVITY...")
						break
					}
				} else { // missed
					if shotProximity > 100 {
						fmt.Printf("SHORT OF TARGET BY %d YARDS.\n", int(math.Abs(float64(shotProximity))))
					} else {
						fmt.Printf("OVER TARGET BY %d YARDS.\n", int(math.Abs(float64(shotProximity))))
					}
				}
			} else {
				fmt.Print("\nBOOM !!!!   YOU HAVE JUST BEEN DESTROYED BY THE ENEMY.\n\n\n")
				fmt.Println("BETTER GO BACK TO FORT SILL FOR REFRESHER TRAINING!")
				return
			}
		}
	}
}

```

这段代码的作用是让用户进行猜数字游戏。它将读取用户输入并随机生成一个数字，然后循环直至用户猜中了正确的数字，此时程序将打印出“TRY AGAIN (Y OR N)?”并等待用户输入。如果用户猜中了数字，则程序将打印一条消息并退出循环。如果用户没有猜中数字，则程序将再次循环并继续提示用户尝试。

具体来说，代码首先使用 `bufio.NewScanner` 函数从标准输入(通常是键盘)中读取用户的输入。然后，它使用 `rand.Seed` 函数随机生成一个 Unix 纳米(1纳米=10^-9 米)的随机数，并将其作为 `rand.Int` 函数的种子参数，从而确保每次运行程序时生成的随机数都不同。

接下来，代码使用 `printIntro` 函数输出一个消息，告诉用户这是一道猜数字的游戏，并且需要输入一个数字(通常是一个两位数)。然后，它进入一个 for 循环，反复读取用户输入并尝试匹配他们猜测的数字。如果用户猜中了正确的数字，则程序将使用 `fmt.Println` 函数打印一条消息并退出循环。如果用户猜中了错误的数字，则程序将使用 `fmt.Println` 函数打印一条消息，并继续循环直到他们猜中了正确的数字。


```
func main() {
	rand.Seed(time.Now().UnixNano())
	scanner := bufio.NewScanner(os.Stdin)

	printIntro()

	for {
		play()

		fmt.Print("TRY AGAIN (Y OR N)? ")
		scanner.Scan()

		if strings.ToUpper(scanner.Text())[0:1] != "Y" {
			fmt.Println("\nOK. RETURN TO BASE CAMP.")
			break
		}
	}
}

```

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

# `00_Alternate_Languages/44_Hangman/C/main.c`

这段代码是一个用C语言编写的Hangman游戏。它的主要目的是让玩家在猜测密码时逐渐为他们的错误猜测画出一个形状。以下是代码的主要功能和结构：

1. 包含头文件：stdio.h，stdlib.h，string.h，time.h。
2. 定义常量：MAX_WORDS，表示Hangman最多可以显示多少个单词。
3. 定义宏：#ifdef _WIN32和#else，用于根据当前操作系统选择CLEAR_MEM宏定义，分别为cls和clear。
4. 函数声明：printf，它的作用是打印Hangman的当前阶段。
5. main函数：开始游戏的部分。
6. 游戏逻辑：初始化并创建一个Hangman对象，尝试从1到100猜测密码。如果猜对，游戏结束，否则显示当前错误猜测的阶段。
7. 绘制Hangman：使用drawHangman函数在屏幕上绘制Hangman的各个部分。
8. 循环：使用while循环，只要猜都对，循环就不停止。
9. 更改密码：使用scanf函数从玩家输入中读取新密码。

总的来说，这段代码是一个简单的Hangman游戏，它允许玩家猜测一个密码，并在猜对后显示Hangman的当前阶段。


```
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define MAX_WORDS 100

//check if windows or linux for the clear screen
#ifdef _WIN32
#define CLEAR "cls"
#else
#define CLEAR "clear"
#endif

/**
 * @brief Prints the stage of the hangman based on the number of wrong guesses.
 * 
 * @param stage Hangman stage.
 */
```

这段代码定义了一个名为`print_hangman`的函数，它接受一个整数参数`stage`。这个函数的作用是在Hangman游戏的不同阶段打印出不同的字符，以代表不同的字母。

在函数内部，首先使用一个`switch`语句来判断当前阶段`stage`属于哪个类别。然后根据当前阶段打印相应的字符。

具体来说，当`stage`为0时，打印一个空行，然后打印四个星号，代表字母O。当`stage`为1时，打印一个空行，然后打印一个星号和一个圆圈，代表字母O。当`stage`为2时，打印一个空行，然后打印一个星号和一个斜杠，代表字母O。当`stage`为3时，打印一个空行，然后打印一个星号和一个斜杠，再打印一个反斜杠，代表字母O。当`stage`为4时，打印一个空行，然后打印一个星号和一个反斜杠，再打印一个斜杠，代表字母X。当`stage`为5时，打印一个空行，然后打印一个星号和一个反斜杠，再打印一个斜杠和一个反斜杠，代表字母X。

如果当前阶段的字母无法打印，程序会跳过该阶段，不会执行打印语句。


```
void print_hangman(int stage){
    switch (stage){
        case 0:
            printf("----------\n");
            printf("|        |\n");
            printf("|\n");
            printf("|\n");
            printf("|\n");
            printf("|\n");
            break;
        case 1:
            printf("----------\n");
            printf("|        |\n");
            printf("|        O\n");
            printf("|        |\n");
            printf("|\n");
            printf("|\n");
            break;
        case 2:
            printf("----------\n");
            printf("|        |\n");
            printf("|        o\n");
            printf("|       /|\n");
            printf("|\n");
            printf("|\n");
            break;
        case 3:
            printf("----------\n");
            printf("|        |\n");
            printf("|        o\n");
            printf("|       /|\\\n");
            printf("|\n");
            printf("|\n");
            break; 
        case 4:
            printf("----------\n");
            printf("|        |\n");
            printf("|        o\n");
            printf("|       /|\\\n");
            printf("|       /\n");
            printf("|\n");
            break; 
        case 5:
            printf("----------\n");
            printf("|        |\n");
            printf("|        o\n");
            printf("|       /|\\\n");
            printf("|       / \\\n");
            printf("|\n");
            break;         
        default:
            break;
    }
}

```

这段代码定义了一个名为`random_word_picker`的函数，用于从包含单词的文本文件中随机选择一个单词并返回给调用者。以下是函数的实现细节：

1. 函数首先定义了一个字符型变量`word`，用于存储从字典中返回的单词。
2. 函数接着使用`malloc`函数分配了足够大小的内存空间用于存储单词，并使用`srand`函数设置随机数种子以生成更高质量的随机数。
3. 函数使用`fopen`函数打开了一个名为`dictionary.txt`的文件，该文件包含一个包含多个单词的词汇表。
4. 函数然后使用`rand`函数生成一个随机整数，用于选择在词汇表中随机定位一个单词。
5. 函数使用`for`循环遍历整个词汇表，每次循环从文件中读取一个单词并将其存储在`word`变量中。
6. 函数关闭文件读取操作，然后返回`word`变量以给调用者一个随机的英语单词。

该函数的作用是帮助从包含单词的词汇表中随机选择一个单词并返回给调用者，以便从词汇表中随机选择一个单词，而不必知道词汇表的名称或包含的单词列表。


```
/**
 * @brief Picks and return a random word from the dictionary.
 * 
 * @return Random word 
 */
char* random_word_picker(){
    //generate a random english word
    char* word = malloc(sizeof(char) * 100);
    FILE* fp = fopen("dictionary.txt", "r");
    srand(time(NULL));
    if (fp == NULL){
        printf("Error opening dictionary.txt\n");
        exit(1);
    }
    int random_number = rand() % MAX_WORDS;
    for (int j = 0; j < random_number; j++){
        fscanf(fp, "%s", word);
    }
    fclose(fp);
    return word;
}




```

这段代码是一个简单的文本游戏，其目的是让玩家猜测一个100个字符的单词，并给出玩家猜测错误的次数。

程序首先定义了一个名为`word`的二维字符数组，用于存储单词。然后定义了一个名为`hidden_word`的二维字符数组，用于存储替换掉部分字符后的单词。接着，程序使用循环和字符串比较算法来正确地将`word`中的字符替换到`hidden_word`中。

接下来，程序定义了一个名为`stage`的变量，用于跟踪猜测错误的次数和正确的猜测次数。定义了一个名为`wrong_guesses`的变量，用于跟踪猜测错误的次数，和一个名为`correct_guesses`的变量，用于跟踪正确的猜测次数。程序还定义了一个名为`guess`的变量，用于存储玩家输入的猜测字符串。

程序进入一个无限循环，每次循环程序都会进行以下操作：清除屏幕、打印当前的`hidden_word`数组、提示玩家输入猜测的字符串、比较`guess`和`word`中的字符是否匹配，如果匹配则执行相应的操作。如果匹配失败，则增加`wrong_guesses`的值。

程序会重复执行这个循环，直到玩家猜中了所有的字符或者猜中了但`wrong_guesses`的值大于等于6。如果`wrong_guesses`的值大于6，则程序打印提示信息并结束游戏，玩家失败。否则，程序打印提示信息并继续游戏，玩家继续猜测。


```
void main(void){
    char* word = malloc(sizeof(char) * 100);
    word = random_word_picker();
    char* hidden_word = malloc(sizeof(char) * 100);
    for (int i = 0; i < strlen(word); i++){
        hidden_word[i] = '_';
    }
    hidden_word[strlen(word)] = '\0';
    int stage = 0;
    int wrong_guesses = 0;
    int correct_guesses = 0;
    char* guess = malloc(sizeof(char) * 100);
    while (wrong_guesses < 6 && correct_guesses < strlen(word)){
        CLEAR;
        print_hangman(stage);
        printf("%s\n", hidden_word);
        printf("Enter a guess: ");
        scanf("%s", guess);
        for (int i = 0; i < strlen(word); i++){
            if (strcmp(guess,word) == 0){
                correct_guesses = strlen(word);
            }
            else if (guess[0] == word[i]){
                hidden_word[i] = guess[0];
                correct_guesses++;
            }
        }
        if (strchr(word, guess[0]) == NULL){
            wrong_guesses++;
        }
        stage = wrong_guesses;
    }
    if (wrong_guesses == 6){
        printf("You lose! The word was %s\n", word);
    }
    else {
        printf("You win!\n");
    }
}
```

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

# `00_Alternate_Languages/45_Hello/go/main.go`

这段代码定义了一个名为PROBLEM_TYPE的枚举类型，它包含四个不同的枚举值，分别为SEX、HEALTH、MONEY和JOB。

该代码还定义了一个名为PROBLEM_CATEGORY的枚举类型，它与PROBLEM_TYPE枚举类型一起定义了所有可能的单个问题类型，分别为SEX、HEALTH、MONEY和JOB。

另外，代码还导入了三个外部库，分别为"bufio"、"fmt"和"os"。

接下来，该代码会遍历PROBLEM_CATEGORY枚举类型的所有值，并对于每个值，使用"fmt.Println"函数将问题类型名称打印出来，同时使用"time.Sleep"函数暂停当前程序的输出，以避免在某些情况下重复打印。

最后，该代码还定义了一个名为"UKNOWN"的永久变量，它的值为9。该变量似乎没有被用于任何有用的目的，只是一个简单的保留。


```
package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"time"
)

type PROBLEM_TYPE int8

const (
	SEX PROBLEM_TYPE = iota
	HEALTH
	MONEY
	JOB
	UKNOWN
)

```

这两段代码都定义了一个名为 `getYesOrNo` 的函数和一个名为 `printTntro` 的函数。

`getYesOrNo` 的函数接受三个参数，分别为 `bool`、`bool` 和 `string` 类型。函数使用 `bufio.NewScanner` 从标准输入（通常是终端输入）中读取输入。然后扫描器的下一个输入使用 `strings.ToUpper` 函数将其转换为大写字母，如果转换后的字符串为 "YES"，那么函数返回 `true`，`true` 和输入字符串相同；否则返回 `false`，`false` 和输入字符串相同。函数返回的结果是 `true`、`false` 和 `string` 类型的输入字符串。

`printTntro` 的函数定义了一个字符串 `"HELLO"`，然后使用 `fmt.Println` 函数将其打印到屏幕上。函数没有参数，返回值为 nothing。


```
func getYesOrNo() (bool, bool, string) {
	scanner := bufio.NewScanner(os.Stdin)

	scanner.Scan()

	if strings.ToUpper(scanner.Text()) == "YES" {
		return true, true, scanner.Text()
	} else if strings.ToUpper(scanner.Text()) == "NO" {
		return true, false, scanner.Text()
	} else {
		return false, false, scanner.Text()
	}
}

func printTntro() {
	fmt.Println("                              HELLO")
	fmt.Println("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
	fmt.Print("\n\n\n")
	fmt.Println("HELLO. MY NAME IS CREATIVE COMPUTER.")
	fmt.Println("\nWHAT'S YOUR NAME?")
}

```

这段代码是一个函数 `askEnjoyQuestion`，它接受一个用户名参数 `user`，并输出一个询问用户是否喜欢当前环境的问题。

函数内部首先打印一个问候用户的问题，然后进入一个无限循环。在循环中，函数调用了一个名为 `getYesOrNo` 的函数，该函数会尝试从两个选项中选择一个作为答案并返回。

如果用户提供了有效的答案（即选择 "YES"），则函数会打印一条消息并结束循环。否则，函数会打印一条消息并重新开始循环。如果两次循环中都未能找到有效的答案，则函数会向用户询问他们的意见，并要求他们选择 "YES" 或 "NO"。

总之，这段代码的主要目的是询问用户他们是否喜欢当前的环境，并提供一个有趣的问题来帮助用户做出决定。


```
func askEnjoyQuestion(user string) {
	fmt.Printf("HI THERE %s, ARE YOU ENJOYING YOURSELF HERE?\n", user)

	for {
		valid, value, msg := getYesOrNo()

		if valid {
			if value {
				fmt.Printf("I'M GLAD TO HEAR THAT, %s.\n", user)
				fmt.Println()
			} else {
				fmt.Printf("OH, I'M SORRY TO HEAR THAT, %s. MAYBE WE CAN\n", user)
				fmt.Println("BRIGHTEN UP YOUR VISIT A BIT.")
			}
			break
		} else {
			fmt.Printf("%s, I DON'T UNDERSTAND YOUR ANSWER OF '%s'.\n", user, msg)
			fmt.Println("PLEASE ANSWER 'YES' OR 'NO'.  DO YOU LIKE IT HERE?")
		}
	}
}

```

这段代码定义了一个名为 `promptForProblems` 的函数，接受一个字符串类型的参数 `user`。函数的作用是提示用户输入他们的问题类型，然后根据用户输入的类型返回相应的结果类型。

具体来说，函数首先使用 `bufio.NewScanner` 函数从标准输入（通常是键盘）读取用户的输入。然后，它打印出一段文本，以引起用户的注意，接着询问用户他们的问题是什么。用户需要回答一个问题，可能是有关他们的性别（SEX）、健康状况（HEALTH）、金钱（MONEY）或工作（JOB）。

如果用户输入的类型与 "SEX"、"HEALTH" 或 "JOB" 中的任何一个相等，函数将返回相应的结果类型。否则，函数将返回一个名为 `UKNOWN` 的结果类型，表示用户输入的问题类型无法识别。


```
func promptForProblems(user string) PROBLEM_TYPE {
	scanner := bufio.NewScanner(os.Stdin)
	fmt.Println()
	fmt.Printf("SAY %s, I CAN SOLVE ALL KINDS OF PROBLEMS EXCEPT\n", user)
	fmt.Println("THOSE DEALING WITH GREECE.  WHAT KIND OF PROBLEMS DO")
	fmt.Println("YOU HAVE? (ANSWER SEX, HEALTH, MONEY, OR JOB)")
	for {
		scanner.Scan()

		switch strings.ToUpper(scanner.Text()) {
		case "SEX":
			return SEX
		case "HEALTH":
			return HEALTH
		case "MONEY":
			return MONEY
		case "JOB":
			return JOB
		default:
			return UKNOWN
		}
	}
}

```

这段代码是一个名为 "promptTooMuchOrTooLittle" 的函数，它用于 prompt 用户在解决问题时输入 "TOO MUCH" 或 "TOO LITTLE"，然后返回相应的结果。函数使用了两个参数 "bool" 和 "bool"，分别用于存储 "TOO MUCH" 和 "TOO LITTLE" 的输入值。

函数首先使用 "bufio.NewScanner(os.Stdin)" 创建一个 scanner 对象，该 scanner 将从标准输入（通常是键盘输入）读取字符串。然后，使用 "scanner.Scan()" 方法开始从标准输入中读取字符。

接下来，函数通过调用 "strings.ToUpper()" 函数将输入的字符串转换为大写。如果字符串转换为大写后是 "TOO MUCH"，函数将返回两个布尔值 "true" 和 "true"，表示这是一个难题。否则，如果转换为大写后是 "TOO LITTLE"，函数将返回两个布尔值 "true" 和 "false"，表示这不是一个难题。否则，函数将返回两个布尔值 "false" 和 "false"，表示这是一个容易的问题。

函数 "solveSexProblem()" 是一个名为 "user" 的函数，它使用 "promptTooMuchOrTooLittle()" 函数来获取用户在解决问题时的输入，然后根据结果打印不同的消息。如果用户输入 "TOO MUCH"，函数将打印 "YOU CALL THAT A PROBLEM?!!  I SHOULD HAVE SUCH PROBLEMS!" 和 "IF IT BOTHERS YOU, %s, TAKE A COLD SHOWER."。如果用户输入 "TOO LITTLE"，函数将打印 "WHY ARE YOU HERE IN SUFFERN, %s?  YOU SHOULD BE\n" 和 "IN TOKYO OR NEW YORK OR AMSTERDAM OR SOMEPLACE WITH SOME" 消息。


```
func promptTooMuchOrTooLittle() (bool, bool) {
	scanner := bufio.NewScanner(os.Stdin)

	scanner.Scan()

	if strings.ToUpper(scanner.Text()) == "TOO MUCH" {
		return true, true
	} else if strings.ToUpper(scanner.Text()) == "TOO LITTLE" {
		return true, false
	} else {
		return false, false
	}
}

func solveSexProblem(user string) {
	fmt.Println("IS YOUR PROBLEM TOO MUCH OR TOO LITTLE?")
	for {
		valid, tooMuch := promptTooMuchOrTooLittle()
		if valid {
			if tooMuch {
				fmt.Println("YOU CALL THAT A PROBLEM?!!  I SHOULD HAVE SUCH PROBLEMS!")
				fmt.Printf("IF IT BOTHERS YOU, %s, TAKE A COLD SHOWER.\n", user)
			} else {
				fmt.Printf("WHY ARE YOU HERE IN SUFFERN, %s?  YOU SHOULD BE\n", user)
				fmt.Println("IN TOKYO OR NEW YORK OR AMSTERDAM OR SOMEPLACE WITH SOME")
				fmt.Println("REAL ACTION.")
			}
			return
		} else {
			fmt.Printf("DON'T GET ALL SHOOK, %s, JUST ANSWER THE QUESTION\n", user)
			fmt.Println("WITH 'TOO MUCH' OR 'TOO LITTLE'.  WHICH IS IT?")
		}
	}
}

```

这段代码定义了三个函数，分别是 `solveHealthProblem()`、`solveMoneyProblem()` 和 `solveJobProblem()`。这些函数都接受一个参数 `user`，然后使用模板字符串来输出一条建议。

`solveHealthProblem()` 的作用是鼓励用户关注健康问题，建议他们采取一系列的健康措施，如服用两片阿司匹林、多喝水和独自睡觉。

`solveMoneyProblem()` 的作用是为用户提供一些经济上的建议，建议他们想办法赚更多的钱，或者改善他们的财务状况。

`solveJobProblem()` 的作用是为用户提供一些关于工作或生活的话题上的建议，建议他们找到一份零售商店的工作，或者认真对待他们的工作。


```
func solveHealthProblem(user string) {
	fmt.Printf("MY ADVICE TO YOU %s IS:\n", user)
	fmt.Println("     1.  TAKE TWO ASPRIN")
	fmt.Println("     2.  DRINK PLENTY OF FLUIDS (ORANGE JUICE, NOT BEER!)")
	fmt.Println("     3.  GO TO BED (ALONE)")
}

func solveMoneyProblem(user string) {
	fmt.Printf("SORRY, %s, I'M BROKE TOO.  WHY DON'T YOU SELL\n", user)
	fmt.Println("ENCYCLOPEADIAS OR MARRY SOMEONE RICH OR STOP EATING")
	fmt.Println("SO YOU WON'T NEED SO MUCH MONEY?")
}

func solveJobProblem(user string) {
	fmt.Printf("I CAN SYMPATHIZE WITH YOU %s.  I HAVE TO WORK\n", user)
	fmt.Println("VERY LONG HOURS FOR NO PAY -- AND SOME OF MY BOSSES")
	fmt.Printf("REALLY BEAT ON MY KEYBOARD.  MY ADVICE TO YOU, %s,\n", user)
	fmt.Println("IS TO OPEN A RETAIL COMPUTER STORE.  IT'S GREAT FUN.")
}

```

这段代码是一个函数，被称为 `askQuestionLoop`，它接受一个字符串类型的参数 `user`。这个函数的目的是循环不断地询问用户一些问题，直到用户给出了所有可能的答案。

函数内部包含一个嵌套的循环，这个循环会不断地提示用户输入问题，并打印出问题的选项和答案。如果用户输入的答案是 "JOB"，那么函数会打印出一条信息，告诉用户他们的回答是 "GREETINGS"，然后退出循环。

在循环的每次迭代中，函数会首先打印出一个空格，然后打印出 "ANY MORE PROBLEMS YOU WANT SOLVED, [user]?" 的问题，要求用户再次输入问题。如果用户输入的答案是 "YES" 或 "NO"，那么函数会询问用户是否还需要继续解决问题，然后再次打印出问题。如果用户在两次输入之间没有给出任何答案，那么函数会退出循环。


```
func askQuestionLoop(user string) {
	for {
		problem := promptForProblems(user)

		switch problem {
		case SEX:
			solveSexProblem(user)
		case HEALTH:
			solveHealthProblem(user)
		case MONEY:
			solveMoneyProblem(user)
		case JOB:
			solveJobProblem(user)
		case UKNOWN:
			fmt.Printf("OH %s, YOUR ANSWER IS GREEK TO ME.\n", user)
		}

		for {
			fmt.Println()
			fmt.Printf("ANY MORE PROBLEMS YOU WANT SOLVED, %s?\n", user)

			valid, value, _ := getYesOrNo()
			if valid {
				if value {
					fmt.Println("WHAT KIND (SEX, MONEY, HEALTH, JOB)")
					break
				} else {
					return
				}
			}
			fmt.Printf("JUST A SIMPLE 'YES' OR 'NO' PLEASE, %s\n", user)
		}
	}
}

```

It looks like this is a Go program that has a function called `askForFee` that asks the user to pay a fee. It uses the `fmt.Printf` method to prompt the user to enter a tip and to display a message if they don't pay the fee. It also uses the `getYesOrNo` function to get the user's response.

It is not clear what the `getYesOrNo` function does.

It is also not clear what the program is intended


```
func goodbyeUnhappy(user string) {
	fmt.Println()
	fmt.Printf("TAKE A WALK, %s.\n", user)
	fmt.Println()
	fmt.Println()
}

func goodbyeHappy(user string) {
	fmt.Printf("NICE MEETING YOU %s, HAVE A NICE DAY.\n", user)
}

func askForFee(user string) {
	fmt.Println()
	fmt.Printf("THAT WILL BE $5.00 FOR THE ADVICE, %s.\n", user)
	fmt.Println("PLEASE LEAVE THE MONEY ON THE TERMINAL.")
	time.Sleep(4 * time.Second)
	fmt.Print("\n\n\n")
	fmt.Println("DID YOU LEAVE THE MONEY?")

	for {
		valid, value, msg := getYesOrNo()
		if valid {
			if value {
				fmt.Printf("HEY, %s, YOU LEFT NO MONEY AT ALL!\n", user)
				fmt.Println("YOU ARE CHEATING ME OUT OF MY HARD-EARNED LIVING.")
				fmt.Println()
				fmt.Printf("WHAT A RIP OFF, %s!!!\n", user)
				fmt.Println()
			} else {
				fmt.Printf("THAT'S HONEST, %s, BUT HOW DO YOU EXPECT\n", user)
				fmt.Println("ME TO GO ON WITH MY PSYCHOLOGY STUDIES IF MY PATIENTS")
				fmt.Println("DON'T PAY THEIR BILLS?")
			}
			return
		} else {
			fmt.Printf("YOUR ANSWER OF '%s' CONFUSES ME, %s.\n", msg, user)
			fmt.Println("PLEASE RESPOND WITH 'YES' or 'NO'.")
		}
	}
}

```

这段代码的主要作用是读取用户输入的多行字符，并对输入的字符进行处理。具体来说，它实现了以下功能：

1. 读取用户输入的多行字符，存储到一个名为 `bufio.Scanner` 的 `bufio.io` 包中的 `scanner` 变量中。
2. 对扫描到的字符进行处理。
3. 如果扫描到包含用户名 `userName`，则输出 "Hello, " + userName + ", your question!"。
4. 如果扫描到不包含用户名 `userName`，则输出 "Unknown question!"。
5. 循环读取字符，直到遇到换行符 `os.AsyncScanln`。

函数 `printTntro()` 可能是在打印一些字符或者输出一些信息，具体内容无法确定。函数 `askEnjoyQuestion(userName)` 和 `askQuestionLoop(userName)` 可以视为读取用户输入的问题，并根据用户输入的问题输出不同的回答。函数 `askForFee(userName)` 可能是在询问用户是否需要支付费用，具体实现可能因程序实现者而异。函数 `goodbyeHappy(userName)` 和 `goodbyeUnhappy(userName)` 可能是在用户输入不愿意回答或者不满意的问题时输出一些信息或者结束程序。


```
func main() {
	scanner := bufio.NewScanner(os.Stdin)

	printTntro()
	scanner.Scan()
	userName := scanner.Text()
	fmt.Println()

	askEnjoyQuestion(userName)

	askQuestionLoop(userName)

	askForFee(userName)

	if false {
		goodbyeHappy(userName) // unreachable
	} else {
		goodbyeUnhappy(userName)
	}

}

```

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

# `00_Alternate_Languages/47_Hi-Lo/go/main.go`

这段代码是一个 RESTful API 示例，它实现了两个功能：

1. 打印游戏介绍信息，包括游戏名称、游戏规则和说明，以及游戏背景和主题，以便用户了解游戏内容和玩法。

2. 接受用户输入的尝试次数，如果用户在规定的尝试次数内没有猜出钱财的数额，则显示游戏结果并结束游戏。如果猜对了，则游戏继续，并提示用户继续猜测。

具体来说，这段代码可以分为以下几个步骤：

1. 定义一个名为 `printIntro` 的函数，该函数用于打印游戏介绍信息，包括游戏名称、游戏规则和说明，以及游戏背景和主题。函数输出的字符串格式如下：

```
package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

const MAX_ATTEMPTS = 6
```

2. 定义一个名为 `printAmount` 的函数，该函数用于接受用户输入的尝试次数，并输出游戏结果和结束游戏。函数的实现与上面打印的游戏介绍信息类似，但输出结果并结束游戏的逻辑移到了 `fmt.Println` 函数中，而不是 `fmt.Printf`。函数的输入参数是一个整数类型，表示猜测的钱数。函数的实现如下：

```
func printAmount(amount int) {
	fmt.Printf("GUESS THE AMOUNT, YOU WIN ALL THE MONEY IN THE JACKPOT!\n")
	fmt.Printf("THEN YOU GET ANOTHER CHANCE TO WIN MORE MONEY.  HOWEVER,")
	fmt.Printf("IF YOU DO NOT GUESS THE AMOUNT, THE GAME ENDS.\n")
	fmt.Println()
	fmt.Println()
}
```

3. 调用 `printIntro` 和 `printAmount` 函数，分别打印游戏介绍信息和输出游戏结果，等待用户输入尝试次数并尝试猜测钱数。当用户猜对钱数时，游戏结束并输出一条消息，否则循环等待新的尝试。当用户在规定的尝试次数内没有猜对钱数时，游戏结束并输出一条消息，提示用户结束游戏。

这段代码中，`MAX_ATTEMPTS` 常量表示最多可以尝试猜测的次数。如果用户在规定的尝试次数内没有猜对钱数，程序会自动结束游戏，并输出一条消息，告诉用户游戏已经结束。如果用户猜对了钱数，程序会再次尝试猜测，并提示用户继续猜测。


```
package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

const MAX_ATTEMPTS = 6

func printIntro() {
	fmt.Println("HI LO")
	fmt.Println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
	fmt.Println("\n\n\nTHIS IS THE GAME OF HI LO.")
	fmt.Println("\nYOU WILL HAVE 6 TRIES TO GUESS THE AMOUNT OF MONEY IN THE")
	fmt.Println("HI LO JACKPOT, WHICH IS BETWEEN 1 AND 100 DOLLARS.  IF YOU")
	fmt.Println("GUESS THE AMOUNT, YOU WIN ALL THE MONEY IN THE JACKPOT!")
	fmt.Println("THEN YOU GET ANOTHER CHANCE TO WIN MORE MONEY.  HOWEVER,")
	fmt.Println("IF YOU DO NOT GUESS THE AMOUNT, THE GAME ENDS.")
	fmt.Println()
	fmt.Println()
}

```

This appears to be a program that allows the user to guess a random number and potentially win money. The program reads in a random number from the user's input, and then reads in the number of attempts the user has made to guess the number. The user is given a maximum of MAX\_ATTEMPTS attempts to guess the number correctly, and upon completing an attempt, they will receive a message indicating whether their guess was too high or too low, or if they completely missed the number. If the user guesses correctly, they will win the specified amount of money and be prompted to play again. If the user mis guesses, they will be given a message and will not be able to play again. The program also prints out the total amount of money won by the user at the end of the game.


```
func main() {
	rand.Seed(time.Now().UnixNano())
	scanner := bufio.NewScanner(os.Stdin)

	printIntro()

	totalWinnings := 0

	for {
		fmt.Println()
		secret := rand.Intn(1000) + 1

		guessedCorrectly := false

		for attempt := 0; attempt < MAX_ATTEMPTS; attempt++ {
			fmt.Println("YOUR GUESS?")
			scanner.Scan()
			guess, err := strconv.Atoi(scanner.Text())
			if err != nil {
				fmt.Println("INVALID INPUT")
			}

			if guess == secret {
				fmt.Printf("GOT IT!!!!!!!!!!   YOU WIN %d DOLLARS.\n", secret)
				guessedCorrectly = true
				break
			} else if guess > secret {
				fmt.Println("YOUR GUESS IS TOO HIGH.")
			} else {
				fmt.Println("YOUR GUESS IS TOO LOW.")
			}
		}

		if guessedCorrectly {
			totalWinnings += secret
			fmt.Printf("YOUR TOTAL WINNINGS ARE NOW $%d.\n", totalWinnings)
		} else {
			fmt.Printf("YOU BLEW IT...TOO BAD...THE NUMBER WAS %d\n", secret)
		}

		fmt.Println()
		fmt.Println("PLAYAGAIN (YES OR NO)?")
		scanner.Scan()

		if strings.ToUpper(scanner.Text())[0:1] != "Y" {
			break
		}
	}
	fmt.Println("\nSO LONG.  HOPE YOU ENJOYED YOURSELF!!!")
}

```

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