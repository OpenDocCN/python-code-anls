# BasicComputerGames源码解析 2

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

# `00_Alternate_Languages/09_Battle/go/main.go`

该代码的主要作用是生成一个随机的名称，并在成功发送邮件到指定收件人时，向收件人发送一封带有确认信息的电子邮件。以下是具体的步骤：

1. 导入所需的包：
```python
import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)
```
这些包用于处理电子邮件客户端的功能。

2. 定义常量：
```python
const (
	SEA_WIDTH        = 6
	DESTROYER_LENGTH = 2
	CRUISER_LENGTH   = 3
	CARRIER_LENGTH   = 4
)
```
这些常量用于定义海盗船的数量和尺寸，以及用于在棋盘上放置海盗船的位置等信息。

3. 发送邮件的功能：
```python
func sendEmail(recipient string, subject string, body string) error {
	// Create a new email client
	client := &bufio.Client{}
	// Create a new email message
	message := &bufio.Message{}
	// Set the sender's email address
	message.SetAddress(os.GetMyEmailAddress())
	// Set the message's subject line
	message.SetSubject(fmt.Sprintf("Subject: %s", subject))
	// Set the message's body
	message.SetBody(body)
	// Create a new email session
	session, err := client.ListenAndConnect("smtp.example.com", 25)
	if err != nil {
		return err
	}
	// Write the message to the session
	message.WriteTo(session, recipient)
	// Send the email
	return session.Send(message, "简单实验")
}
```
这段代码发送一封带有指定主题和正文的电子邮件，并将其发送到指定的收件人。

4. 发送邮件的实现在`sendEmail`函数中：
```python
func sendEmail(recipient string, subject string, body string) error {
	// Create a new email client
	client := &bufio.Client{}
	// Create a new email message
	message := &bufio.Message{}
	// Set the sender's email address
	message.SetAddress(os.GetMyEmailAddress())
	// Set the message's subject line
	message.SetSubject(fmt.Sprintf("Subject: %s", subject))
	// Set the message's body
	message.SetBody(body)
	// Create a new email session
	session, err := client.ListenAndConnect("smtp.example.com", 25)
	if err != nil {
		return err
	}
	// Write the message to the session
	message.WriteTo(session, recipient)
	// Send the email
	return session.Send(message, "简单实验")
}
```
这段代码将创建一个指向`bufio.Client`的`message`变量，用于存储电子邮件的内容。然后，设置邮件发送者的地址、主题和正文，并从客户端缓冲区读取电子邮件客户端的连接。然后，创建一个`bufio.Message`变量，并将其设置为要发送的电子邮件的内容。接下来，创建一个指向客户端缓冲区的`session`变量，用于将邮件发送到服务器。最后，调用`session.Send`方法将邮件发送出去，并返回是否成功。

5. 选择要发送的邮件：
```python
package main

import (
	"fmt"
)

func main() {
	recipient := "recipient@example.com"
	subject := "邮件主题"
	body := "这是一封测试邮件"
```


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

const (
	SEA_WIDTH        = 6
	DESTROYER_LENGTH = 2
	CRUISER_LENGTH   = 3
	CARRIER_LENGTH   = 4
)

```

这段代码定义了三个数据类型：Point、Vector和Sea，以及一个函数NewSea和一个函数getRandomVector。

代码中首先定义了Point类型，它有两个整型元素。接着定义了Vector类型，它是一个Point类型的数组，可以理解为一个二维数组。最后定义了Sea类型，它是一个包含六个Point类型元素的数组，每个元素都是一个二维数组。

函数NewSea创建了一个Sea类型的数组，元素类型都是整型。函数getRandomVector则返回一个随机生成的Vector类型。函数中使用了两个for循环，一个用于循环生成随机数，一个用于循环从0到5循环，用于判断生成的随机数是否合法（即在0到5之间）。

整个函数的作用是返回一个RandomVector类型的数组，它包含六个随机生成的、长度为2的元素。


```
type Point [2]int
type Vector Point
type Sea [][]int

func NewSea() Sea {
	s := make(Sea, 6)
	for r := 0; r < SEA_WIDTH; r++ {
		c := make([]int, 6)
		s[r] = c
	}

	return s
}

func getRandomVector() Vector {
	v := Vector{}

	for {
		v[0] = rand.Intn(3) - 1
		v[1] = rand.Intn(3) - 1

		if !(v[0] == 0 && v[1] == 0) {
			break
		}
	}
	return v
}

```

这段代码定义了三个函数，分别作用于点(Point)和向量(Vector)：

1. `addVector`函数接收一个点(Point)和一个向量(Vector)，并返回一个新的点(Point)。该函数通过将点(Point)和向量(Vector)的对应分量相加，然后创建一个新的点(Point)类型的变量。最后，函数返回新创建的点(Point)。

2. `isWithinSea`函数接收一个点(Point)和一个海平面(Sea)，并返回一个布尔值。该函数通过检查点(Point)是否在海水(Sea)的边界内来判断。具体来说，函数检查点(Point)的x和y坐标是否在1到len(Sea)之间，如果是，则函数返回true；否则，函数返回false。

3. `valueAt`函数接收一个点(Point)和一个海平面(Sea)，并返回一个整数。该函数通过在海水(Sea)中查找给定点(Point)的对应元素(例如索引)，并返回该元素的值。


```
func addVector(p Point, v Vector) Point {
	newPoint := Point{}

	newPoint[0] = p[0] + v[0]
	newPoint[1] = p[1] + v[1]

	return newPoint
}

func isWithinSea(p Point, s Sea) bool {
	return (1 <= p[0] && p[0] <= len(s)) && (1 <= p[1] && p[1] <= len(s))
}

func valueAt(p Point, s Sea) int {
	return s[p[1]-1][p[0]-1]
}

```

这段代码定义了两个函数：reportInputError() 和 getNextTarget()。

函数reportInputError()的作用是当用户输入的参数不符合要求时输出错误信息。具体来说，它会要求用户输入两个数字，这两个数字是从1到某个给定的数字（通过SEA_WIDTH变量获取）的逗号分隔的。如果用户输入的参数不符合要求，该函数将会被调用，并输出错误信息。

函数getNextTarget()的作用是获取下一个目标点，它从标准输入（通常是键盘输入）读取用户的输入，然后对输入进行解析，如果输入不符合要求，该函数将会被调用并停止。具体来说，该函数会要求用户输入两个数字（通过scanning从标准输入读取），然后尝试将这些数字转换成浮点数类型。如果输入数字不符合要求（比如只有一个数字或无法转换成浮点数），或者获取到的输入不在指定的范围内（通过isWithinSea()函数判断），该函数将会返回一个Point类型的变量，表示下一个目标点。如果函数的输入符合要求，该函数将会返回该Point类型的变量。


```
func reportInputError() {
	fmt.Printf("INVALID. SPECIFY TWO NUMBERS FROM 1 TO %d, SEPARATED BY A COMMA.\n", SEA_WIDTH)
}

func getNextTarget(s Sea) Point {
	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Println("\n?")
		scanner.Scan()

		vals := strings.Split(scanner.Text(), ",")

		if len(vals) != 2 {
			reportInputError()
			continue
		}

		x, xErr := strconv.Atoi(strings.TrimSpace(vals[0]))
		y, yErr := strconv.Atoi(strings.TrimSpace(vals[1]))

		if (len(vals) != 2) || (xErr != nil) || (yErr != nil) {
			reportInputError()
			continue
		}

		p := Point{}
		p[0] = x
		p[1] = y
		if isWithinSea(p, s) {
			return p
		}
	}
}

```

这两段代码定义了两个函数，分别是 `setValueAt` 和 `hasShip`。它们的作用是使 `Sea` 数组中的元素值替换为给定的整数 `value`，以及在满足特定条件的情况下，检查数组 `Sea` 中是否有特定编码 `code`。

`setValueAt` 函数接收三个参数：整数 `value`、Point 类型表示坐标轴的点的 `Point` 和 Sea 数组中的元素 `Sea`。该函数通过在 `Point` 类型的二维切片上写入 `value` 来修改 `Sea` 数组。为了确保在 `Point` 类型中，我们创建了一个新的二维数组，然后将 `value` 写入该数组的第二行第二列。

`hasShip` 函数接收两个参数：Sea 数组和整数 `code`。该函数首先检查数组 `Sea` 中是否有特定编码 `code`。如果是，函数将返回 `true`，否则返回 `false`。在检查过程中，函数遍历 Sea 数组中的每个元素，如果找到相应的编码，函数将返回 `true`，否则继续遍历。


```
func setValueAt(value int, p Point, s Sea) {
	s[p[1]-1][p[0]-1] = value
}

func hasShip(s Sea, code int) bool {
	hasShip := false
	for r := 0; r < SEA_WIDTH; r++ {
		for c := 0; c < SEA_WIDTH; c++ {
			if s[r][c] == code {
				hasShip = true
				break
			}
		}
	}
	return hasShip
}

```



该代码定义了两个函数，分别是 `countSunk` 和 `placeShip`。它们的作用如下：

`countSunk` 函数用于计算在给定的 `s` 造船厂中，有多少艘船沉没。它通过遍历所有 `codes` 类型的物品，对于每个物品，它使用 `hasShip` 函数检查该物品是否沉没，如果沉没，则 `sunk` 变量值加 1。

`placeShip` 函数用于在给定的 `s` 造船厂中，将一定数量的某种物品放入指定的位置。它通过生成随机数种子，创建一个位置数组 `points`，然后将随机分配的物品添加到该位置数组中。接着，它遍历位置数组中的所有元素，对于每个元素，使用 `addVector` 和 `isWithinSea` 函数将其位置添加到指定位置，并设置该位置上的物品数量。最后，它清除指定位置数组中的元素，并检查所有添加的物品是否都存在于 `s` 造船厂中。如果所有物品都存在于 `s` 造船厂中，则程序继续运行，否则停止程序。


```
func countSunk(s Sea, codes []int) int {
	sunk := 0

	for _, c := range codes {
		if !hasShip(s, c) {
			sunk += 1
		}
	}

	return sunk
}

func placeShip(s Sea, size, code int) {
	for {
		start := Point{}
		start[0] = rand.Intn(SEA_WIDTH) + 1
		start[1] = rand.Intn(SEA_WIDTH) + 1
		vector := getRandomVector()

		point := start
		points := []Point{}

		for i := 0; i < size; i++ {
			point = addVector(point, vector)
			points = append(points, point)
		}

		clearPosition := true
		for _, p := range points {
			if !isWithinSea(p, s) {
				clearPosition = false
				break
			}
			if valueAt(p, s) > 0 {
				clearPosition = false
				break
			}
		}
		if !clearPosition {
			continue
		}

		for _, p := range points {
			setValueAt(code, p, s)
		}
		break
	}
}

```

这两段代码定义了一个名为setupShips的函数，其作用是为游戏中的船只指定位置并放置。

setupShips函数接收一个名为s的整数参数，代表船只的数量。函数内部使用placeShip函数来放置船只，该函数需要传入两个参数：一个是将要放置的船只编号，另一个是船只类型，可以是海盗船、巡洋舰或商船。第三个参数是要放置的船只长度，第四个参数是要放置的船只类型，可以是巡洋舰或商船。第五个参数是要放置的船只类型，这里是商船。第六个参数是要放置的船只类型，这里是海盗船。

printIntro函数用于在游戏开始时打印一些游戏信息和代码，包括游戏名称、版名、以及一些游戏中的地图信息。


```
func setupShips(s Sea) {
	placeShip(s, DESTROYER_LENGTH, 1)
	placeShip(s, DESTROYER_LENGTH, 2)
	placeShip(s, CRUISER_LENGTH, 3)
	placeShip(s, CRUISER_LENGTH, 4)
	placeShip(s, CARRIER_LENGTH, 5)
	placeShip(s, CARRIER_LENGTH, 6)
}

func printIntro() {
	fmt.Println("                BATTLE")
	fmt.Println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
	fmt.Println()
	fmt.Println("THE FOLLOWING CODE OF THE BAD GUYS' FLEET DISPOSITION")
	fmt.Println("HAS BEEN CAPTURED BUT NOT DECODED:	")
	fmt.Println()
}

```

这道题的目的是让两个函数起作用，第一个函数是 `printInstructions()`，第二个函数是 `printEncodedSea()`。

`printInstructions()` 函数的作用是打印一些说明性的文本，然后输出一个游戏开始提示符，并等待用户按下 'Enter' 键。然后它将打印第二个说明性的文本，然后输出一个游戏开始提示符，并等待用户再次按下 'Enter' 键。这样看起来像是在向用户询问他们是否准备好开始游戏，但实际上只是一个简单的打印字符串。

`printEncodedSea(s Sea)` 函数的作用是打印一个编码过的海平面图，其中 `s` 是表示海平面的网格的行数和列数。它从网格的左上角开始打印，打印到右下角，然后打印一个换行符。接着，它打印了一些控制信息，然后打印游戏开始提示符，并等待用户再次按下 'Enter' 键。这样，用户就可以开始游戏了。


```
func printInstructions() {
	fmt.Println()
	fmt.Println()
	fmt.Println("DE-CODE IT AND USE IT IF YOU CAN")
	fmt.Println("BUT KEEP THE DE-CODING METHOD A SECRET.")
	fmt.Println()
	fmt.Println("START GAME")
}

func printEncodedSea(s Sea) {
	for x := 0; x < SEA_WIDTH; x++ {
		fmt.Println()
		for y := SEA_WIDTH - 1; y > -1; y-- {
			fmt.Printf(" %d", s[y][x])
		}
	}
	fmt.Println()
}

```

This is a game of Ship滑稽（Ship Ham与其他桌游物的互动）中的一个场景脚本。这个脚本似乎是用于在游戏中给玩家提供建议，以帮助他们赢得游戏。

然而，需要注意的是，这个脚本在某些情况下可能会导致游戏无法继续进行。例如，当某个玩家的船的击沉值（用金币表示）为负数时，这个脚本会输出一条信息，并使游戏继续进行。此外，当某个玩家在游戏中赢得了胜利，所有的敌军船都被摧毁后，这个脚本会输出一条信息，并使游戏结束。

总之，这个脚本似乎是一个有趣的工具，可以帮助新手玩家更好地了解游戏，并快速地在游戏中获胜。但是，对于有经验的玩家，这个脚本可能不会提供太多帮助。


```
func wipeout(s Sea) bool {
	for c := 1; c <= 7; c++ {
		if hasShip(s, c) {
			return false
		}
	}
	return true
}

func main() {
	rand.Seed(time.Now().UnixNano())

	s := NewSea()

	setupShips(s)

	printIntro()

	printEncodedSea(s)

	printInstructions()

	splashes := 0
	hits := 0

	for {
		target := getNextTarget(s)
		targetValue := valueAt(target, s)

		if targetValue < 0 {
			fmt.Printf("YOU ALREADY PUT A HOLE IN SHIP NUMBER %d AT THAT POINT.\n", targetValue)
		}

		if targetValue <= 0 {
			fmt.Println("SPLASH! TRY AGAIN.")
			splashes += 1
			continue
		}

		fmt.Printf("A DIRECT HIT ON SHIP NUMBER %d\n", targetValue)
		hits += 1
		setValueAt(targetValue*-1, target, s)

		if !hasShip(s, targetValue) {
			fmt.Println("AND YOU SUNK IT. HURRAH FOR THE GOOD GUYS.")
			fmt.Println("SO FAR, THE BAD GUYS HAVE LOST")
			fmt.Printf("%d DESTROYER(S), %d CRUISER(S), AND %d AIRCRAFT CARRIER(S).\n", countSunk(s, []int{1, 2}), countSunk(s, []int{3, 4}), countSunk(s, []int{5, 6}))
		}

		if !wipeout(s) {
			fmt.Printf("YOUR CURRENT SPLASH/HIT RATIO IS %2f\n", float32(splashes)/float32(hits))
			continue
		}

		fmt.Printf("YOU HAVE TOTALLY WIPED OUT THE BAD GUYS' FLEET WITH A FINAL SPLASH/HIT RATIO OF %2f\n", float32(splashes)/float32(hits))

		if splashes == 0 {
			fmt.Println("CONGRATULATIONS -- A DIRECT HIT EVERY TIME.")
		}

		fmt.Println("\n****************************")
		break
	}
}

```

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

# `00_Alternate_Languages/11_Bombardment/go/main.go`

这段代码的主要作用是生成随机的玩家进步消息，并输出到控制台。它使用了两个主要函数，`generateMessage` 和 `main` 函数。

1. `import` 函数用于导入所需的包，包括 `bufio`、`fmt`、`math/rand`、`os` 和 `strconv`。这些包提供了输入/输出操作、格式化字符串以及字符串转换等功能。

2. `PLAYER_PROGRESS_MESSAGES` 常量表示玩家进步消息的列表。在这里，我们创建了一个字符串切片 `PLAYER_PROGRADE_MESSAGES`，用于存储 `PLAYER_PROGRESS_MESSAGES` 中的消息。

3. `generateMessage` 函数使用随机数生成消息，并将其添加到 `PLAYER_PROGRADE_MESSAGES` 切片。

4. `main` 函数负责生成随机玩家进步消息并输出到控制台。它首先从 `PLAYER_PROGRADE_MESSAGE


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

// Messages correspond to outposts remaining (3, 2, 1, 0)
var PLAYER_PROGRESS_MESSAGES = []string{
	"YOU GOT ME, I'M GOING FAST. BUT I'LL GET YOU WHEN\nMY TRANSISTO&S RECUP%RA*E!",
	"THREE DOWN, ONE TO GO.\n\n",
	"TWO DOWN, TWO TO GO.\n\n",
	"ONE DOWN, THREE TO GO.\n\n",
}

```

这段代码定义了一个名为ENEMY_PROGRESS_MESSAGES的整型数组，其中包含五个字符串元素，每个字符串元素都是一个含有文本字符和'\n'结束的字符串。这些字符串元素似乎是通过调用其中的"YOU'RE DEAD. YOUR LAST OUTPOST WAS AT %d. HA, HA, HA."这一行来生成的，其中%d是一个变量，它的值似乎是在运行时获得的。

接着，定义了一个名为displayField的函数，该函数似乎用于输出一个包含一些文本字符的字符串。函数内部使用for循环来遍历数组中的每个元素，并使用fmt.Printf函数来输出该元素中的字符。输出似乎是逐个字符地输出的，然后换行并输出'\n'作为结束符。

最后，在displayField函数中，似乎还有一行打印字符串"YOU HAVE CANNOT CRASH THE render node。这是因为在函数内部使用了一个var UNDERLORENUMERON keyword，似乎是要打印出来的一些字符，但并没有进一步的解释或定义。


```
var ENEMY_PROGRESS_MESSAGES = []string{
	"YOU'RE DEAD. YOUR LAST OUTPOST WAS AT %d. HA, HA, HA.\nBETTER LUCK NEXT TIME.",
	"YOU HAVE ONLY ONE OUTPOST LEFT.\n\n",
	"YOU HAVE ONLY TWO OUTPOSTS LEFT.\n\n",
	"YOU HAVE ONLY THREE OUTPOSTS LEFT.\n\n",
}

func displayField() {
	for r := 0; r < 5; r++ {
		initial := r*5 + 1
		for c := 0; c < 5; c++ {
			//x := strconv.Itoa(initial + c)
			fmt.Printf("\t%d", initial+c)
		}
		fmt.Println()
	}
	fmt.Print("\n\n\n\n\n\n\n\n\n")
}

```

这段代码是一个名为`printIntro`的函数，它使用了`fmt.Println`函数来输出一段文字，其中包括一些描述性的语句和一个游戏地图的场景。

该函数的作用是向用户介绍游戏中的地图和规则。在函数内部，使用`fmt.Println`函数输出了一系列描述性的信息，包括游戏地图上的位置和玩家可以放置的 outpost 数量。接着，函数输出了一段游戏规则的描述，包括玩家只能在一个 outpost 上下文下 placement，计算机也会按照这个规则进行处理。

函数的输出结果将引导用户了解游戏的目标和规则，以及如何进行游戏。


```
func printIntro() {
	fmt.Println("                                BOMBARDMENT")
	fmt.Println("                CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
	fmt.Println()
	fmt.Println()
	fmt.Println("YOU ARE ON A BATTLEFIELD WITH 4 PLATOONS AND YOU")
	fmt.Println("HAVE 25 OUTPOSTS AVAILABLE WHERE THEY MAY BE PLACED.")
	fmt.Println("YOU CAN ONLY PLACE ONE PLATOON AT ANY ONE OUTPOST.")
	fmt.Println("THE COMPUTER DOES THE SAME WITH ITS FOUR PLATOONS.")
	fmt.Println()
	fmt.Println("THE OBJECT OF THE GAME IS TO FIRE MISSLES AT THE")
	fmt.Println("OUTPOSTS OF THE COMPUTER.  IT WILL DO THE SAME TO YOU.")
	fmt.Println("THE ONE WHO DESTROYS ALL FOUR OF THE ENEMY'S PLATOONS")
	fmt.Println("FIRST IS THE WINNER.")
	fmt.Println()
	fmt.Println("GOOD LUCK... AND TELL US WHERE YOU WANT THE BODIES SENT!")
	fmt.Println()
	fmt.Println("TEAR OFF MATRIX AND USE IT TO CHECK OFF THE NUMBERS.")
	fmt.Print("\n\n\n\n")
}

```

这两段代码定义了两个函数，分别是positionList()和generateEnemyPositions()。

positionList()函数的作用是创建一个长度为25的整数数组，并在其中存储了0到24的整数。这些整数代表了函数调用者可以获得的位置编号，以便后续操作。函数返回了这个创建好的整数数组。

generateEnemyPositions()函数的作用是创建一个长度为4的整数数组，并在其中随机选择了4个位置，这些位置必须属于0到24的整数范围内。函数使用了positionList()函数返回的整数数组，并对其进行了随机重排。最后，函数返回了这个包含4个随机整数的数组。


```
func positionList() []int {
	positions := make([]int, 25)
	for i := 0; i < 25; i++ {
		positions[i] = i + 1
	}
	return positions
}

// Randomly choose 4 'positions' out of a range of 1 to 25
func generateEnemyPositions() []int {
	positions := positionList()
	rand.Shuffle(len(positions), func(i, j int) { positions[i], positions[j] = positions[j], positions[i] })
	return positions[:4]
}

```

这段代码定义了一个名为 `isValidPosition` 的函数，它接收一个整数参数 `p`，并返回一个布尔值，表示该位置是否符合要求。

接着定义了一个名为 `promptForPlayerPositions` 的函数，它使用 `bufio.NewScanner` 读取标准输入（通常是键盘输入），并循环读取四个整数，存储在 `positions` 数组中。函数中使用了 `fmt.Println` 函数来输出提示信息，使用 `strconv.Atoi` 函数将输入的字符串转换为整数，并使用 `isValidPosition` 函数检查是否合法。如果格式错误或者位置不在要求范围内，函数会循环询问用户输入，直到输入了四个不同的位置，或者已经询问了多次没有正确答案为止。

函数的作用是获取玩家四个不同的位置，并将它们存储在 `positions` 数组中。如果用户输入的位置不符合要求，函数会循环询问用户输入，直到获取到符合要求的四个位置为止。


```
func isValidPosition(p int) bool {
	return p >= 1 && p <= 25
}

func promptForPlayerPositions() []int {
	scanner := bufio.NewScanner(os.Stdin)
	var positions []int

	for {
		fmt.Println("\nWHAT ARE YOUR FOUR POSITIONS (1-25)?")
		scanner.Scan()
		rawPositions := strings.Split(scanner.Text(), " ")

		if len(rawPositions) != 4 {
			fmt.Println("PLEASE ENTER FOUR UNIQUE POSITIONS")
			goto there
		}

		for _, p := range rawPositions {
			pos, err := strconv.Atoi(p)
			if (err != nil) || !isValidPosition(pos) {
				fmt.Println("ALL POSITIONS MUST RANGE (1-25)")
				goto there
			}
			positions = append(positions, pos)
		}
		if len(positions) == 4 {
			return positions
		}

	there:
	}
}

```

这段代码定义了一个名为`promptPlayerForTarget`的函数，其接受一个整数类型的参数。

该函数的作用是向玩家询问想要发射导弹的目标位置，并返回目标位置作为一个整数。

具体来说，函数使用了`bufio.NewScanner`从标准输入（通常是键盘）读取用户的输入。

然后，函数循环读取用户的输入，并使用`strconv.Atoi`将输入的字符串转换为整数类型。

接下来，函数使用`isValidPosition`函数来检查输入是否符合目标位置的格式要求。如果输入不符合要求或者存在错误，函数会打印错误消息并继续等待用户的输入。

最后，如果所有输入都正确，函数会返回目标位置，并结束循环。


```
func promptPlayerForTarget() int {
	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Println("\nWHERE DO YOU WISH TO FIRE YOUR MISSILE?")
		scanner.Scan()
		target, err := strconv.Atoi(scanner.Text())

		if (err != nil) || !isValidPosition(target) {
			fmt.Println("POSITIONS MUST RANGE (1-25)")
			continue
		}
		return target
	}
}

```

这段代码定义了两个函数：generateAttackSequence()和attack()。

generateAttackSequence()函数的作用是返回一个包含攻击位置的整数切片。该函数首先从positionList()函数中获取一系列攻击位置，然后使用rand.Shuffle()函数对位置列表进行随机重排，最后返回排好序的攻击位置切片。

attack()函数的作用是执行攻击。该函数接受一个目标整数、一个包含攻击位置的整数切片和一个字符串数组，以及一个或多个用于描述攻击过程的进步消息。该函数首先使用generateAttackSequence()函数获取攻击位置，然后使用for循环逐个检查目标是否在位置列表中。如果是，则该目标被 hit，并执行相应的操作。然后循环继续，直到所有位置都被检查过。如果没有任何一个位置是目标，则该函数将打印一个或多个进步消息，并返回一个表示攻击成功的布尔值。


```
func generateAttackSequence() []int {
	positions := positionList()
	rand.Shuffle(len(positions), func(i, j int) { positions[i], positions[j] = positions[j], positions[i] })
	return positions
}

// Performs attack procedure returning True if we are to continue.
func attack(target int, positions *[]int, hitMsg, missMsg string, progressMsg []string) bool {
	for i := 0; i < len(*positions); i++ {
		if target == (*positions)[i] {
			fmt.Print(hitMsg)

			// remove the target just hit
			(*positions)[i] = (*positions)[len((*positions))-1]
			(*positions)[len((*positions))-1] = 0
			(*positions) = (*positions)[:len((*positions))-1]

			if len((*positions)) != 0 {
				fmt.Print(progressMsg[len((*positions))])
			} else {
				fmt.Printf(progressMsg[len((*positions))], target)
			}
			return len((*positions)) > 0
		}
	}
	fmt.Print(missMsg)
	return len((*positions)) > 0
}

```

这段代码的主要作用是让玩家与敌人进行战斗，并且让玩家与敌人轮流攻击。它包括以下主要步骤：

1. 生成敌人位置：调用 `generateEnemyPositions` 函数，该函数使用 `time.Now().UnixNano()` 获取当前时间的纳秒级 Unix 时间戳作为随机种子，然后返回敌人位置。

2. 生成敌人攻击：调用 `generateAttackSequence` 函数，该函数生成敌人攻击序列，并返回每个攻击的位置。

3. 生成玩家位置：调用 `promptForPlayerPositions` 函数，该函数提示玩家输入他们的位置。

4. 循环攻击玩家：使用一个 for 循环，让玩家攻击敌人。在循环中，调用 `attack` 函数，该函数提示玩家选择一个敌人位置进行攻击，并显示相应的消息。如果玩家选择了一个无效的位置(例如一个友军位置或空格)，循环将终止。

5. 循环防止敌人攻击：同样使用一个 for 循环，让敌人攻击玩家。在循环中，调用 `attack` 函数，并传递一个代表玩家位置的变量。如果玩家位置无效(例如空格或不是一个敌人)，循环将终止。

6. 增加敌人攻击计数：在循环中，使用 `enemyAttackCounter` 变量跟踪敌人攻击计数。每次循环，增加 `enemyAttackCounter` 的值，并在循环结束后使用 `fmt.Sprintf` 格式化将计数器值显示为数字。

7. 显示消息：在循环中，使用 `fmt.Sprintf` 格式化将消息显示为文本字符串。


```
func main() {
	rand.Seed(time.Now().UnixNano())

	printIntro()
	displayField()

	enemyPositions := generateEnemyPositions()
	enemyAttacks := generateAttackSequence()
	enemyAttackCounter := 0

	playerPositions := promptForPlayerPositions()

	for {
		// player attacks
		if !attack(promptPlayerForTarget(), &enemyPositions, "YOU GOT ONE OF MY OUTPOSTS!\n\n", "HA, HA YOU MISSED. MY TURN NOW:\n\n", PLAYER_PROGRESS_MESSAGES) {
			break
		}
		// computer attacks
		hitMsg := fmt.Sprintf("I GOT YOU. IT WON'T BE LONG NOW. POST %d WAS HIT.\n", enemyAttacks[enemyAttackCounter])
		missMsg := fmt.Sprintf("I MISSED YOU, YOU DIRTY RAT. I PICKED %d. YOUR TURN:\n\n", enemyAttacks[enemyAttackCounter])
		if !attack(enemyAttacks[enemyAttackCounter], &playerPositions, hitMsg, missMsg, ENEMY_PROGRESS_MESSAGES) {
			break
		}
		enemyAttackCounter += 1
	}

}

```

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

# `00_Alternate_Languages/12_Bombs_Away/go/main.go`

这段代码定义了一个名为`Choice`的结构体，用于在用户输入时从用户选择中选择一个选项。

下面是具体的实现细节：

1. 首先导入了`bufio`、`fmt`、`math/rand`、`os`、`strconv`和`strings`这些包，分别用于字符缓冲流、格式化字符串、随机数生成、操作系统输入输出、字符串处理和字符串比较包。

2. 在`Choice`结构体的定义中，定义了两个字段`idx`和`msg`，分别表示当前选项的索引和提示信息。

3. 在`main`函数中，首先创建了一个字符缓冲流`os.Stdout`和一个字符串`fmt.Printf`，用于将用户输入的选项打印出来。

4. 使用`fmt.Printf`将选项打印出来，格式化字符串中的%s，将当前选项的索引`idx`和提示信息`msg`作为参数传递给`fmt.Printf`，最终得到一个字符串，例如：

```
/選擇一
/選擇二
/選擇三
```

5. 然后使用`os.Stdout`读取用户输入，并使用`strconv.I()`将输入的字符串转换为整数类型，分配给`Choice`结构体中的`idx`字段。

6. 接着等待用户输入，并使用`time.Sleep`暂停程序的执行一段时间，以便用户有足够的时间进行选择。

7. 最后，将用户输入的选项打印出来，使用`fmt.Printf`将选项打印出来，格式化字符串中的%s，将当前选项的索引`idx`和提示信息`msg`作为参数传递给`fmt.Printf`，最终得到一个字符串，例如：

```
user选择了 /選擇一
user选择了 /選擇二
user选择了 /選擇三
```

8. 程序中的所有输出都使用了`fmt.Printf`函数，将格式化字符串作为输出的`%s`字符串与当前选项的索引`idx`和提示信息`msg`一起打印出来。


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

type Choice struct {
	idx string
	msg string
}

```

这段代码定义了三个函数：

1. `playerSurvived()`:
该函数打印一条消息，表示玩家成功通过了一次Tremendous Flak(可能是某种困难或危险的情况)。

2. `playerDeath()`:
该函数打印三条消息，表示玩家在一次Tremendous Flak中死亡了。

3. `missionSuccess()`:
该函数打印一条消息，表示玩家成功完成了一次使命，并给出了一个数字Kill Count。这个数字可能是通过某种难度或挑战获得的，也可能是与玩家有关的统计数据。


```
func playerSurvived() {
	fmt.Println("YOU MADE IT THROUGH TREMENDOUS FLAK!!")
}

func playerDeath() {
	fmt.Println("* * * * BOOM * * * *")
	fmt.Println("YOU HAVE BEEN SHOT DOWN.....")
	fmt.Println("DEARLY BELOVED, WE ARE GATHERED HERE TODAY TO PAY OUR")
	fmt.Println("LAST TRIBUTE...")
}

func missionSuccess() {
	fmt.Printf("DIRECT HIT!!!! %d KILLED.\n", int(100*rand.Int()))
	fmt.Println("MISSION SUCCESSFUL.")
}

```

这段代码是一个名为`deathWithChance`的函数，它接收一个浮点数作为参数并返回一个布尔值。函数的作用是判断玩家是否在随机机会内存活，如果概率大于随机数，则返回`True`，否则返回`False`。

代码的主要目的是帮助程序根据玩家输入的次数来判断是否有机会存活。具体来说，程序会首先向玩家询问如何获得任务，然后程序会根据玩家输入的次数进行判断。如果玩家请求次数大于某个值（通常是160），则程序会输出类似于“MISSIONS, NOT MILES... 150 MISSIONS IS HIGH EVEN FOR OLD-TIMERS”这样的信息。如果这个值大于100，则程序会再次输出类似于“THAT'S PUSHING THE ODDS!”的信息。如果这个值小于25，则程序会输出类似于“FRESH OUT OF TRAINING, EH?”的信息。最后，程序会根据输入的次数生成一个随机数，并将其与160取模，如果生成的新随机数大于模数，则说明玩家存活，返回`True`，否则返回`False`。


```
// Takes a float between 0 and 1 and returns a boolean
// if the player has survived (based on random chance)
// Returns True if death, False if survived
func deathWithChance(probability float64) bool {
	return probability > rand.Float64()
}

func startNonKamikaziAttack() {
	numMissions := getIntInput("HOW MANY MISSIONS HAVE YOU FLOWN? ")

	for numMissions > 160 {
		fmt.Println("MISSIONS, NOT MILES...")
		fmt.Println("150 MISSIONS IS HIGH EVEN FOR OLD-TIMERS")
		numMissions = getIntInput("HOW MANY MISSIONS HAVE YOU FLOWN? ")
	}

	if numMissions > 100 {
		fmt.Println("THAT'S PUSHING THE ODDS!")
	}

	if numMissions < 25 {
		fmt.Println("FRESH OUT OF TRAINING, EH?")
	}

	fmt.Println()

	if float32(numMissions) > (160 * rand.Float32()) {
		missionSuccess()
	} else {
		missionFailure()
	}
}

```

这段代码是一个简单的 Python 函数，名为 `missionFailure()`，它计算玩家在敌人手中有核武器的情况下是否成功完成任务。

函数首先输出一段幽默的话，说明玩家的任务失败了。然后，它询问玩家敌人手中有没有枪支和导弹，并从玩家输入中获取相应的选择。接着，函数检查敌人是否拥有枪支，如果是，就获取玩家输入中敌人枪支的准确率，并判断该准确率是否小于10。如果准确率小于10，函数就输出“你撒谎，但你付出的代价就严重了”的话语，然后玩家将面临死亡。如果敌人不拥有枪支，就检查敌人是否拥有导弹，如果拥有，就设置导弹威胁权重为35，然后计算死亡概率。最后，函数根据死亡概率决定是否继续游戏，如果概率大于50%，则玩家死亡，否则玩家存活。


```
func missionFailure() {
	fmt.Printf("MISSED TARGET BY %d MILES!\n", int(2+30*rand.Float32()))
	fmt.Println("NOW YOU'RE REALLY IN FOR IT !!")
	fmt.Println()

	enemyWeapons := getInputFromList("DOES THE ENEMY HAVE GUNS(1), MISSILES(2), OR BOTH(3)? ", []Choice{{idx: "1", msg: "GUNS"}, {idx: "2", msg: "MISSILES"}, {idx: "3", msg: "BOTH"}})

	// If there are no gunners (i.e. weapon choice 2) then
	// we say that the gunners have 0 accuracy for the purposes
	// of calculating probability of player death
	enemyGunnerAccuracy := 0.0
	if enemyWeapons.idx != "2" {
		enemyGunnerAccuracy = float64(getIntInput("WHAT'S THE PERCENT HIT RATE OF ENEMY GUNNERS (10 TO 50)? "))
		if enemyGunnerAccuracy < 10.0 {
			fmt.Println("YOU LIE, BUT YOU'LL PAY...")
			playerDeath()
		}
	}

	missileThreatWeighting := 35.0
	if enemyWeapons.idx == "1" {
		missileThreatWeighting = 0
	}

	death := deathWithChance((enemyGunnerAccuracy + missileThreatWeighting) / 100)

	if death {
		playerDeath()
	} else {
		playerSurvived()
	}
}

```

This is a program written in the Go programming language that simulates a hypothetical game where players can choose to attack different targets or engage in a Kamikaze mission. The program uses a function `getInputFromList` to retrieve player choices from the user.

The available targets are:

* Albania
* Greece
* North Africa
* USSR
* Germany

If the user chooses a country other than the listed targets, the function returns "NOT AVAILABLE".

The function `playAllies` prints a message asking the player to be careful.

The function `playAllies` is called `startNonKamikaziAttack` which seems to be the starting point for the Kamikaze mission.

The function `playJapan` prints a message asking the player to choose between aondsman mission or not. If the player chooses not to attack, the function will call `playerDeath`. If the player chooses to attack, the function will call `missionSuccess` or `playerDeath`, depending on the random number generated.

It's important to note that the program does not provide any information about the historical or cultural significance of the choices made by the players, or any backstory for the Kamikaze missions, and it is not clear what the `getInputFromList` function is supposed to do in this context.


```
func playItaly() {
	targets := []Choice{{idx: "1", msg: "SHOULD BE EASY -- YOU'RE FLYING A NAZI-MADE PLANE."}, {idx: "2", msg: "BE CAREFUL!!!"}, {idx: "3", msg: "YOU'RE GOING FOR THE OIL, EH?"}}
	target := getInputFromList("YOUR TARGET -- ALBANIA(1), GREECE(2), NORTH AFRICA(3)", targets)
	fmt.Println(target.msg)
	startNonKamikaziAttack()
}

func playAllies() {
	aircraftMessages := []Choice{{idx: "1", msg: "YOU'VE GOT 2 TONS OF BOMBS FLYING FOR PLOESTI."}, {idx: "2", msg: "YOU'RE DUMPING THE A-BOMB ON HIROSHIMA."}, {idx: "3", msg: "YOU'RE CHASING THE BISMARK IN THE NORTH SEA."}, {idx: "4", msg: "YOU'RE BUSTING A GERMAN HEAVY WATER PLANT IN THE RUHR."}}
	aircraft := getInputFromList("AIRCRAFT -- LIBERATOR(1), B-29(2), B-17(3), LANCASTER(4): ", aircraftMessages)
	fmt.Println(aircraft.msg)
	startNonKamikaziAttack()
}

func playJapan() {
	acknowledgeMessage := []Choice{{idx: "Y", msg: "Y"}, {idx: "N", msg: "N"}}
	firstMission := getInputFromList("YOU'RE FLYING A KAMIKAZE MISSION OVER THE USS LEXINGTON.\nYOUR FIRST KAMIKAZE MISSION? (Y OR N): ", acknowledgeMessage)
	if firstMission.msg == "N" {
		playerDeath()
	}
	if rand.Float64() > 0.65 {
		missionSuccess()
	} else {
		playerDeath()
	}
}

```

这段代码定义了一个名为 `playGermany` 的函数，它的作用是打印一系列游戏中的目标（targets）的提示信息，并开始游戏。

函数内部，首先创建了一个名为 `targets` 的数组，用于存储游戏中的目标。这个数组的每个元素都是一个名为 `Choice` 的结构体，包含一个 `idx` 字段表示目标ID，和一个 `msg` 字段用于显示目标信息。

然后，函数调用 `getInputFromList` 函数，从用户那里获取一个字符串，这个字符串包含一个或多个目标ID，格式为 "目标ID 提示信息"。这个函数返回一个包含三个元素的切片，分别对应 `targets` 数组中的每个元素。

接下来，函数创建了一个名为 `switch side` 的 `case` 语句，这个语句根据用户从 `getInputFromList` 函数返回的字符串中的 `idx` 字段来选择一个 `case`，然后执行对应的 `println` 函数。

最后，函数调用 `startNonKamikaziAttack` 函数，开始游戏。


```
func playGermany() {
	targets := []Choice{{idx: "1", msg: "YOU'RE NEARING STALINGRAD."}, {idx: "2", msg: "NEARING LONDON.  BE CAREFUL, THEY'VE GOT RADAR."}, {idx: "3", msg: "NEARING VERSAILLES.  DUCK SOUP.  THEY'RE NEARLY DEFENSELESS."}}
	target := getInputFromList("A NAZI, EH?  OH WELL.  ARE YOU GOING FOR RUSSIA(1),\nENGLAND(2), OR FRANCE(3)? ", targets)
	fmt.Println(target.msg)
	startNonKamikaziAttack()
}

func playGame() {
	fmt.Println("YOU ARE A PILOT IN A WORLD WAR II BOMBER.")
	side := getInputFromList("WHAT SIDE -- ITALY(1), ALLIES(2), JAPAN(3), GERMANY(4): ", []Choice{{idx: "1", msg: "ITALY"}, {idx: "2", msg: "ALLIES"}, {idx: "3", msg: "JAPAN"}, {idx: "4", msg: "GERMANY"}})
	switch side.idx {
	case "1":
		playItaly()
	case "2":
		playAllies()
	case "3":
		playJapan()
	case "4":
		playGermany()
	}
}

```

这段代码的主要目的是让用户在两个选项中做出选择，并在用户做出选择后退出循环。

具体来说，代码首先使用 `rand.Seed(time.Now().UnixNano())` 生成一个随机整数，作为游戏中的随机事件。然后，代码使用一个 `for` 循环来重复执行以下操作：

1. 调用 `playGame()` 函数，这个函数可能是游戏的主函数，但在这种情况下可能是暂时的。
2. 从标准输入（通常是键盘）接收来自用户的一个输入（可能是字符串）。
3. 如果用户输入是 "N"，那么退出循环，因为程序认为用户已经完成了任务。
4. 在循环中，程序使用 `fmt.Println()` 函数打印一个用于提示用户输入的提示消息。
5. 使用 `scanner.Scan()` 函数从标准输入中读取用户的输入。
6. 对于每个 `Choice` 类型的输入，程序使用一系列循环来比较用户输入与每个选项的 `idx` 字段。如果输入与 `Choice` 中的 `idx` 字段相等，就返回该 `Choice`。
7. 如果用户在循环中没有输入，或者输入不是 "N"，那么循环将再次运行，每次循环都会打印一个提示消息，以便用户知道应该做什么。

总之，这段代码的主要目的是让用户在两个选项中做出选择，并在用户做出选择后退出循环。


```
func main() {
	rand.Seed(time.Now().UnixNano())

	for {
		playGame()
		if getInputFromList("ANOTHER MISSION (Y OR N):", []Choice{{idx: "Y", msg: "Y"}, {idx: "N", msg: "N"}}).msg == "N" {
			break
		}
	}
}

func getInputFromList(prompt string, choices []Choice) Choice {
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Println(prompt)
		scanner.Scan()
		choice := scanner.Text()
		for _, c := range choices {
			if strings.EqualFold(strings.ToUpper(choice), strings.ToUpper(c.idx)) {
				return c
			}
		}
		fmt.Println("TRY AGAIN...")
	}
}

```

这段代码定义了一个名为 `getIntInput` 的函数，其接受一个字符串参数 `prompt`。该函数使用 `bufio.NewScanner` 函数从标准输入（通常是键盘）读取输入，并使用 `strconv.Atoi` 函数将输入的字符串转换为整数。如果转换过程中出现错误，函数将提示用户重新输入。如果转换成功，函数将返回用户输入的整数。


```
func getIntInput(prompt string) int {
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Println(prompt)
		scanner.Scan()
		choice, err := strconv.Atoi(scanner.Text())
		if err != nil {
			fmt.Println("TRY AGAIN...")
			continue
		} else {
			return choice
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

# `00_Alternate_Languages/20_Buzzword/go/main.go`

This is a program that generates buzzwords. A buzzword is a phrase that is interesting or surprising. The program takes input from the user and generates buzzwords. The buzzwords are printed in a way that is easy for someone to understand. The program can also be modified to generate buzzwords in a specific format.


```
package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"time"
)

func main() {
	rand.Seed(time.Now().UnixNano())
	words := [][]string{
		{
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
		}, {
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
		}, {
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
		},
	}

	scanner := bufio.NewScanner(os.Stdin)

	// Display intro text
	fmt.Println("\n           Buzzword Generator")
	fmt.Println("Creative Computing  Morristown, New Jersey")
	fmt.Println("\n\n")
	fmt.Println("This program prints highly acceptable phrases in")
	fmt.Println("'educator-speak' that you can work into reports")
	fmt.Println("and speeches.  Whenever a question mark is printed,")
	fmt.Println("type a 'Y' for another phrase or 'N' to quit.")
	fmt.Println("\n\nHere's the first phrase:")

	for {
		phrase := ""
		for _, section := range words {
			if len(phrase) > 0 {
				phrase += " "
			}
			phrase += section[rand.Intn(len(section))]
		}
		fmt.Println(phrase)
		fmt.Println()

		// continue?
		fmt.Println("?")
		scanner.Scan()
		if strings.ToUpper(scanner.Text())[0:1] != "Y" {
			break
		}
	}
	fmt.Println("Come back when you need help with another report!")
}

```