# `d:/src/tocomm/basic-computer-games\00_Alternate_Languages\08_Batnum\go\main.go`

```
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
const (
	WinUndefined WinOption = iota  # 定义一个枚举类型WinOption，包括WinUndefined、TakeLast、AvoidLast，分别对应0、1、2
)

type GameOptions struct {  # 定义一个结构体GameOptions
	pileSize    int  # 定义一个整型变量pileSize
	winOption   WinOption  # 定义一个WinOption类型的变量winOption
	startOption StartOption  # 定义一个StartOption类型的变量startOption
	minSelect   int  # 定义一个整型变量minSelect
	maxSelect   int  # 定义一个整型变量maxSelect
}

func NewOptions() *GameOptions {  # 定义一个函数NewOptions，返回一个指向GameOptions类型的指针
	g := GameOptions{}  # 创建一个GameOptions类型的变量g

	g.pileSize = getPileSize()  # 调用getPileSize函数，将返回值赋给pileSize
	if g.pileSize < 0 {  # 如果pileSize小于0
		return &g  # 返回指向g的指针
	}

	g.winOption = getWinOption()  # 调用函数获取游戏胜利条件选项
	g.minSelect, g.maxSelect = getMinMax()  # 调用函数获取玩家可选取的最小和最大数量
	g.startOption = getStartOption()  # 调用函数获取游戏开始选项

	return &g  # 返回游戏配置参数的指针
}

func getPileSize() int:  # 定义函数 getPileSize，返回整数类型
	ps := 0  # 初始化变量 ps 为 0
	var err error  # 声明变量 err 为 error 类型
	scanner := bufio.NewScanner(os.Stdin)  # 使用标准输入创建 Scanner 对象

	for {  # 进入循环
		fmt.Println("Enter Pile Size ")  # 打印提示信息
		scanner.Scan()  # 扫描用户输入
		ps, err = strconv.Atoi(scanner.Text())  # 将用户输入的文本转换为整数并赋值给 ps，同时检查是否有错误
		if err == nil:  # 如果没有错误
			break  # 退出循环
		}
	}
	return ps
}
```
这段代码是一个函数的结束标志。

```
func getWinOption() WinOption {
```
这段代码定义了一个名为getWinOption的函数，它返回一个WinOption类型的值。

```
scanner := bufio.NewScanner(os.Stdin)
```
这段代码创建了一个用于从标准输入读取数据的Scanner对象。

```
for {
		fmt.Println("ENTER WIN OPTION - 1 TO TAKE LAST, 2 TO AVOID LAST:")
		scanner.Scan()
		w, err := strconv.Atoi(scanner.Text())
		if err == nil && (w == 1 || w == 2) {
			return WinOption(w)
		}
	}
}
```
这段代码是一个无限循环，它提示用户输入一个选项，然后将输入的值转换为整数并检查是否为1或2，如果是则返回对应的WinOption值。

```
func getStartOption() StartOption {
```
这段代码定义了一个名为getStartOption的函数，它返回一个StartOption类型的值。

```
scanner := bufio.NewScanner(os.Stdin)
```
这段代码创建了一个用于从标准输入读取数据的Scanner对象。
	for {
		// 打印提示信息，让用户选择开始选项
		fmt.Println("ENTER START OPTION - 1 COMPUTER FIRST, 2 YOU FIRST ")
		// 从标准输入中读取用户输入
		scanner.Scan()
		// 将用户输入的字符串转换为整数
		s, err := strconv.Atoi(scanner.Text())
		// 检查转换是否成功，并且输入的值是1或2
		if err == nil && (s == 1 || s == 2) {
			// 如果条件满足，返回开始选项
			return StartOption(s)
		}
	}
}

func getMinMax() (int, int) {
	// 初始化最小和最大选择值
	minSelect := 0
	maxSelect := 0
	// 初始化最小和最大选择值的错误
	var minErr error
	var maxErr error
	// 创建一个从标准输入读取数据的扫描器
	scanner := bufio.NewScanner(os.Stdin)

	for {
		// 打印提示信息，让用户输入最小和最大值
		fmt.Println("ENTER MIN AND MAX ")
		scanner.Scan()  // 从标准输入中扫描下一行文本
		enteredValues := scanner.Text()  // 获取扫描到的文本内容
		vals := strings.Split(enteredValues, " ")  // 将输入的文本内容按空格分割成字符串数组
		minSelect, minErr = strconv.Atoi(vals[0])  // 将字符串数组中的第一个元素转换为整数
		maxSelect, maxErr = strconv.Atoi(vals[1])  // 将字符串数组中的第二个元素转换为整数
		if (minErr == nil) && (maxErr == nil) && (minSelect > 0) && (maxSelect > 0) && (maxSelect > minSelect) {  // 检查转换是否成功以及输入是否符合规定
			return minSelect, maxSelect  // 如果输入符合规定，则返回最小和最大选择数
		}
	}
}

// This handles the player's turn - asking the player how many objects
// to take and doing some basic validation around that input.  Then it
// checks for any win conditions.
// Returns a boolean indicating whether the game is over and the new pile_size.
func playerMove(pile, min, max int, win WinOption) (bool, int) {
	scanner := bufio.NewScanner(os.Stdin)  // 创建一个从标准输入中读取数据的新扫描器
	done := false  // 初始化一个布尔变量
	for !done {  // 循环直到 done 变为 true
		fmt.Println("YOUR MOVE")  // 打印提示信息
		// 扫描输入，将其转换为整数
		scanner.Scan()
		m, err := strconv.Atoi(scanner.Text())
		// 如果转换出错，则继续循环
		if err != nil {
			continue
		}

		// 如果输入为0，则打印消息并返回true和当前的pile值
		if m == 0 {
			fmt.Println("I TOLD YOU NOT TO USE ZERO!  COMPUTER WINS BY FORFEIT.")
			return true, pile
		}

		// 如果输入大于最大值或小于最小值，则打印消息并继续循环
		if m > max || m < min {
			fmt.Println("ILLEGAL MOVE, REENTER IT")
			continue
		}

		// 从pile中减去输入的值
		pile -= m
		// 设置done为true
		done = true

		// 如果pile小于等于0，则执行以下代码
		if pile <= 0 {
			if win == AvoidLast {  # 如果 win 变量的值等于 AvoidLast
				fmt.Println("TOUGH LUCK, YOU LOSE.")  # 打印 "TOUGH LUCK, YOU LOSE."
			} else {
				fmt.Println("CONGRATULATIONS, YOU WIN.")  # 否则打印 "CONGRATULATIONS, YOU WIN."
			}
			return true, pile  # 返回 true 和 pile 变量的值
		}
	}
	return false, pile  # 返回 false 和 pile 变量的值
}

// This handles the logic to determine how many objects the computer
// will select on its turn.
func computerPick(pile, min, max int, win WinOption) int {  # 定义一个名为 computerPick 的函数，接受 pile、min、max 和 win 四个参数，返回一个整数
	var q int  # 声明一个整数变量 q
	if win == AvoidLast {  # 如果 win 变量的值等于 AvoidLast
		q = pile - 1  # 将 pile - 1 的值赋给 q
	} else {
		q = pile  # 否则将 pile 的值赋给 q
	}
	c := min + max  # 计算最小值和最大值的和

	pick := q - (c * int(q/c))  # 计算取走的物体数量

	if pick < min {  # 如果取走的数量小于最小值
		pick = min  # 将取走的数量设为最小值
	} else if pick > max {  # 如果取走的数量大于最大值
		pick = max  # 将取走的数量设为最大值
	}

	return pick  # 返回取走的数量
}

// This handles the computer's turn - first checking for the various
// win/lose conditions and then calculating how many objects
// the computer will take.
// Returns a boolean indicating whether the game is over and the new pile_size.
func computerMove(pile, min, max int, win WinOption) (bool, int) {
	// first check for end-game conditions
	if win == TakeLast && pile <= max {  # 如果胜利条件是取走最后一个物体，并且物体数量小于等于最大值
		fmt.Printf("COMPUTER TAKES %d AND WINS\n", pile)
		return true, pile
	}
	// 打印电脑取走的数量并且赢得比赛
	// 返回 true 和取走的数量

	if win == AvoidLast && pile <= min {
		fmt.Printf("COMPUTER TAKES %d AND LOSES\n", pile)
		return true, pile
	}
	// 如果 win 选项为 AvoidLast 并且 pile 小于等于最小取走数量
	// 打印电脑取走的数量并且输掉比赛
	// 返回 true 和取走的数量

	// otherwise determine the computer's selection
	selection := computerPick(pile, min, max, win)
	// 否则确定电脑的选择
	pile -= selection
	// 减去电脑的选择数量
	fmt.Printf("COMPUTER TAKES %d AND LEAVES %d\n", selection, pile)
	// 打印电脑取走的数量和剩余的数量
	return false, pile
	// 返回 false 和剩余的数量
}

// This is the main game loop - repeating each turn until one
// of the win/lose conditions is met.
// 这是主游戏循环 - 重复每一轮直到满足赢/输条件。
func play(pile, min, max int, start StartOption, win WinOption) {
	// 初始化游戏结束标志
	gameOver := false
	playersTurn := (start == PlayerFirst)  # 初始化一个布尔变量，表示当前是玩家的回合还是电脑的回合

	for !gameOver {  # 当游戏未结束时循环执行以下代码
		if playersTurn {  # 如果当前是玩家的回合
			gameOver, pile = playerMove(pile, min, max, win)  # 调用玩家移动函数，更新游戏状态和堆的状态
			playersTurn = false  # 将回合切换为电脑的回合
			if gameOver {  # 如果游戏结束
				return  # 结束游戏
			}
		}

		if !playersTurn {  # 如果当前是电脑的回合
			gameOver, pile = computerMove(pile, min, max, win)  # 调用电脑移动函数，更新游戏状态和堆的状态
			playersTurn = true  # 将回合切换为玩家的回合
		}
	}
}

// Print out the introduction and rules of the game
func printIntro() {
	fmt.Printf("%33s%s\n", " ", "BATNUM")  // 打印游戏标题
	fmt.Printf("%15s%s\n", " ", "CREATIVE COMPUTING  MORRISSTOWN, NEW JERSEY")  // 打印游戏信息
	fmt.Printf("\n\n\n")  // 打印空行
	fmt.Println("THIS PROGRAM IS A 'BATTLE OF NUMBERS' GAME, WHERE THE")  // 打印游戏介绍
	fmt.Println("COMPUTER IS YOUR OPPONENT.")
	fmt.Println()
	fmt.Println("THE GAME STARTS WITH AN ASSUMED PILE OF OBJECTS. YOU")
	fmt.Println("AND YOUR OPPONENT ALTERNATELY REMOVE OBJECTS FROM THE PILE.")
	fmt.Println("WINNING IS DEFINED IN ADVANCE AS TAKING THE LAST OBJECT OR")
	fmt.Println("NOT. YOU CAN ALSO SPECIFY SOME OTHER BEGINNING CONDITIONS.")
	fmt.Println("DON'T USE ZERO, HOWEVER, IN PLAYING THE GAME.")
	fmt.Println("ENTER A NEGATIVE NUMBER FOR NEW PILE SIZE TO STOP PLAYING.")
	fmt.Println()
}

func main() {
	for {  // 无限循环
		printIntro()  // 调用打印游戏介绍的函数

		g := NewOptions()  // 创建游戏选项对象
		if g.pileSize < 0 {  # 如果g.pileSize小于0
			return  # 返回
		}

		play(g.pileSize, g.minSelect, g.maxSelect, g.startOption, g.winOption)  # 调用play函数，传入参数g.pileSize, g.minSelect, g.maxSelect, g.startOption, g.winOption
	}
}
```