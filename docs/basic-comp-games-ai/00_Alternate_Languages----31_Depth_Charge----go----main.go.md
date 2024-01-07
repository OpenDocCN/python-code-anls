# `basic-computer-games\00_Alternate_Languages\31_Depth_Charge\go\main.go`

```

package main

import (
	"bufio" // 导入 bufio 包，用于读取输入
	"fmt" // 导入 fmt 包，用于格式化输出
	"math" // 导入 math 包，用于数学计算
	"math/rand" // 导入 math/rand 包，用于生成随机数
	"os" // 导入 os 包，用于操作系统功能
	"strconv" // 导入 strconv 包，用于字符串和基本数据类型的转换
	"strings" // 导入 strings 包，用于处理字符串
	"time" // 导入 time 包，用于时间相关操作
)

type Position []int // 定义 Position 类型为整型切片

func NewPosition() Position { // 定义 NewPosition 函数，返回一个 Position 类型的切片
	p := make([]int, 3) // 创建一个长度为 3 的整型切片
	return Position(p) // 返回该切片
}

func showWelcome() { // 定义 showWelcome 函数，用于显示欢迎信息
	fmt.Print("\033[H\033[2J") // 清空终端屏幕
	fmt.Println("                DEPTH CHARGE") // 输出游戏标题
	fmt.Println("    Creative Computing  Morristown, New Jersey") // 输出游戏信息
	fmt.Println() // 输出空行
}

func getNumCharges() (int, int) { // 定义 getNumCharges 函数，用于获取搜索区域的维度和深度炸弹数量
	scanner := bufio.NewScanner(os.Stdin) // 创建一个从标准输入读取数据的 Scanner 对象

	for { // 循环直到输入正确的维度
		fmt.Println("Dimensions of search area?") // 提示用户输入搜索区域的维度
		scanner.Scan() // 读取用户输入
		dim, err := strconv.Atoi(scanner.Text()) // 将用户输入的字符串转换为整数
		if err != nil { // 如果转换出错
			fmt.Println("Must enter an integer number. Please try again...") // 提示用户必须输入整数
			continue // 继续循环
		}
		return dim, int(math.Log2(float64(dim))) + 1 // 返回维度和深度炸弹数量
	}
}

func askForNewGame() { // 定义 askForNewGame 函数，用于询问是否开始新游戏
	scanner := bufio.NewScanner(os.Stdin) // 创建一个从标准输入读取数据的 Scanner 对象

	fmt.Println("Another game (Y or N): ") // 提示用户是否开始新游戏
	scanner.Scan() // 读取用户输入
	if strings.ToUpper(scanner.Text()) == "Y" { // 如果用户输入是 "Y"
		main() // 重新开始游戏
	}
	fmt.Println("OK. Hope you enjoyed yourself") // 输出结束游戏的信息
	os.Exit(1) // 退出程序
}

func showShotResult(shot, location Position) { // 定义 showShotResult 函数，用于显示射击结果
	result := "Sonar reports shot was " // 初始化射击结果字符串

	if shot[1] > location[1] { // 如果射击位置在目标位置的北方
		result += "north" // 添加北方信息到结果字符串
	} else if shot[1] < location[1] { // 如果射击位置在目标位置的南方
		result += "south" // 添加南方信息到结果字符串
	}

	if shot[0] > location[0] { // 如果射击位置在目标位置的东方
		result += "east" // 添加东方信息到结果字符串
	} else if shot[0] < location[0] { // 如果射击位置在目标位置的西方
		result += "west" // 添加西方信息到结果字符串
	}

	if shot[1] != location[1] || shot[0] != location[0] { // 如果射击位置不在目标位置的水平方向
		result += " and " // 添加连接词到结果字符串
	}
	if shot[2] > location[2] { // 如果射击位置在目标位置的下方
		result += "too low." // 添加太低信息到结果字符串
	} else if shot[2] < location[2] { // 如果射击位置在目标位置的上方
		result += "too high." // 添加太高信息到结果字符串
	} else { // 如果射击位置在目标位置的深度合适
		result += "depth OK." // 添加深度合适信息到结果字符串
	}

	fmt.Println(result) // 输出射击结果
}

func getShot() Position { // 定义 getShot 函数，用于获取射击位置
	scanner := bufio.NewScanner(os.Stdin) // 创建一个从标准输入读取数据的 Scanner 对象

	for { // 循环直到输入正确的射击位置
		shotPos := NewPosition() // 创建一个新的射击位置
		fmt.Println("Enter coordinates: ") // 提示用户输入坐标
		scanner.Scan() // 读取用户输入
		rawGuess := strings.Split(scanner.Text(), " ") // 将用户输入的字符串按空格分割
		if len(rawGuess) != 3 { // 如果输入的坐标不是三个
			goto there // 跳转到标签 there
		}
		for i := 0; i < 3; i++ { // 遍历输入的坐标
			val, err := strconv.Atoi(rawGuess[i]) // 将字符串转换为整数
			if err != nil { // 如果转换出错
				goto there // 跳转到标签 there
			}
			shotPos[i] = val // 将转换后的整数赋值给射击位置
		}
		return shotPos // 返回射击位置
	there:
		fmt.Println("Please enter coordinates separated by spaces") // 提示用户输入正确的坐标格式
		fmt.Println("Example: 3 2 1") // 提示用户正确的坐标示例
	}
}

func getRandomPosition(searchArea int) Position { // 定义 getRandomPosition 函数，用于获取随机目标位置
	pos := NewPosition() // 创建一个新的位置
	for i := 0; i < 3; i++ { // 遍历三个维度
		pos[i] = rand.Intn(searchArea) // 在搜索区域内生成随机位置
	}
	return pos // 返回随机位置
}

func playGame(searchArea, numCharges int) { // 定义 playGame 函数，用于进行游戏
	rand.Seed(time.Now().UTC().UnixNano()) // 使用当前时间的纳秒数作为随机数种子
	fmt.Println("\nYou are the captain of the destroyer USS Computer.") // 输出游戏背景信息
	fmt.Println("An enemy sub has been causing you trouble. Your")
	fmt.Printf("mission is to destroy it. You have %d shots.\n", numCharges) // 输出游戏目标和可用射击次数
	fmt.Println("Specify depth charge explosion point with a")
	fmt.Println("trio of numbers -- the first two are the")
	fmt.Println("surface coordinates; the third is the depth.")
	fmt.Println("\nGood luck!") // 输出游戏提示信息
	fmt.Println()

	subPos := getRandomPosition(searchArea) // 获取随机目标位置

	for c := 0; c < numCharges; c++ { // 循环进行射击
		fmt.Printf("\nTrial #%d\n", c+1) // 输出当前射击次数

		shot := getShot() // 获取射击位置

		if shot[0] == subPos[0] && shot[1] == subPos[1] && shot[2] == subPos[2] { // 如果射击位置与目标位置相同
			fmt.Printf("\nB O O M ! ! You found it in %d tries!\n", c+1) // 输出找到目标的信息
			askForNewGame() // 询问是否开始新游戏
		} else { // 如果射击位置与目标位置不同
			showShotResult(shot, subPos) // 显示射击结果
		}
	}

	// out of depth charges
	fmt.Println("\nYou have been torpedoed! Abandon ship!") // 输出深度炸弹用尽的信息
	fmt.Printf("The submarine was at %d %d %d\n", subPos[0], subPos[1], subPos[2]) // 输出目标位置的信息
	askForNewGame() // 询问是否开始新游戏
}

func main() { // 主函数
	showWelcome() // 显示欢迎信息

	searchArea, numCharges := getNumCharges() // 获取搜索区域的维度和深度炸弹数量

	playGame(searchArea, numCharges) // 开始游戏
}

```