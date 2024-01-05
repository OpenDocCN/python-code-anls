# `d:/src/tocomm/basic-computer-games\00_Alternate_Languages\31_Depth_Charge\go\main.go`

```
package main  # 声明当前文件所属的包

import (  # 导入需要使用的包
	"bufio"  # 用于提供缓冲 I/O
	"fmt"  # 用于格式化输出
	"math"  # 提供数学函数和常数
	"math/rand"  # 提供伪随机数生成器
	"os"  # 提供操作系统功能
	"strconv"  # 用于字符串和数字之间的转换
	"strings"  # 提供操作字符串的函数
	"time"  # 提供时间相关的函数
)

type Position []int  # 定义一个名为 Position 的切片类型

func NewPosition() Position {  # 定义一个名为 NewPosition 的函数，返回一个 Position 类型的切片
	p := make([]int, 3)  # 创建一个长度为 3 的切片
	return Position(p)  # 返回创建的切片
}
# 显示欢迎信息
func showWelcome() {
	# 清空屏幕
	fmt.Print("\033[H\033[2J")
	# 打印游戏标题和创作者信息
	fmt.Println("                DEPTH CHARGE")
	fmt.Println("    Creative Computing  Morristown, New Jersey")
	fmt.Println()
}

# 获取搜索区域的维度
func getNumCharges() (int, int) {
	# 创建一个从标准输入读取数据的扫描器
	scanner := bufio.NewScanner(os.Stdin)

	# 循环直到输入正确的维度
	for {
		# 提示用户输入搜索区域的维度
		fmt.Println("Dimensions of search area?")
		# 读取用户输入
		scanner.Scan()
		# 将输入的文本转换为整数
		dim, err := strconv.Atoi(scanner.Text())
		# 如果转换出错，提示用户重新输入
		if err != nil {
			fmt.Println("Must enter an integer number. Please try again...")
			continue
		}
		# 返回输入的维度和深度
		return dim, int(math.Log2(float64(dim))) + 1
	}
}
}

// askForNewGame 函数用于询问用户是否开始新游戏
func askForNewGame() {
	// 创建一个新的扫描器来读取用户输入
	scanner := bufio.NewScanner(os.Stdin)

	// 打印提示信息，要求用户输入是否开始新游戏
	fmt.Println("Another game (Y or N): ")
	// 扫描用户输入
	scanner.Scan()
	// 如果用户输入的是"Y"，则重新开始游戏
	if strings.ToUpper(scanner.Text()) == "Y" {
		main()
	}
	// 如果用户输入的不是"Y"，则打印提示信息并退出程序
	fmt.Println("OK. Hope you enjoyed yourself")
	os.Exit(1)
}

// showShotResult 函数用于展示射击结果
func showShotResult(shot, location Position) {
	// 初始化结果字符串
	result := "Sonar reports shot was "

	// 如果射击位置在目标位置的北方，则更新结果字符串
	if shot[1] > location[1] { // y-direction
		result += "north"
	} else if shot[1] < location[1] { // y-direction
		result += "south"  # 如果射击位置的第一个元素大于目标位置的第一个元素，向结果字符串中添加"south"

	if shot[0] > location[0] {  # 如果射击位置的第一个元素大于目标位置的第一个元素，向结果字符串中添加"east"
		result += "east"
	} else if shot[0] < location[0] {  # 如果射击位置的第一个元素小于目标位置的第一个元素，向结果字符串中添加"west"
		result += "west"
	}

	if shot[1] != location[1] || shot[0] != location[0] {  # 如果射击位置的第二个元素不等于目标位置的第二个元素，或者射击位置的第一个元素不等于目标位置的第一个元素，向结果字符串中添加" and "
		result += " and "
	}
	if shot[2] > location[2] {  # 如果射击位置的第三个元素大于目标位置的第三个元素，向结果字符串中添加"too low."
		result += "too low."
	} else if shot[2] < location[2] {  # 如果射击位置的第三个元素小于目标位置的第三个元素，向结果字符串中添加"too high."
		result += "too high."
	} else {  # 否则，向结果字符串中添加"depth OK."
		result += "depth OK."
	}
    # 打印结果
    print(result)
}

def getShot():
    # 创建一个从标准输入读取数据的扫描器
    scanner = bufio.NewScanner(os.Stdin)

    # 无限循环，直到满足条件跳出循环
    while True:
        # 创建一个新的位置对象
        shotPos = NewPosition()
        # 打印提示信息
        print("Enter coordinates: ")
        # 从标准输入中扫描一行输入
        scanner.Scan()
        # 将输入按空格分割成字符串列表
        rawGuess = strings.Split(scanner.Text(), " ")
        # 如果输入的字符串列表长度不为3，则跳转到标签there
        if len(rawGuess) != 3:
            goto there
        # 遍历字符串列表
        for i in range(3):
            # 将字符串转换为整数
            val, err = strconv.Atoi(rawGuess[i])
            # 如果转换出错，则跳转到标签there
            if err != nil:
                goto there
            # 将转换后的整数赋值给shotPos的第i个元素
            shotPos[i] = val
		}
		return shotPos  # 返回射击位置
	there:  # 定义标签 "there"
		fmt.Println("Please enter coordinates separated by spaces")  # 打印提示信息
		fmt.Println("Example: 3 2 1")  # 打印示例信息
	}
}

func getRandomPosition(searchArea int) Position {  # 定义函数 getRandomPosition，参数为 searchArea
	pos := NewPosition()  # 创建新的 Position 对象
	for i := 0; i < 3; i++ {  # 循环3次
		pos[i] = rand.Intn(searchArea)  # 为 Position 对象的第 i 个元素赋随机值
	}
	return pos  # 返回 Position 对象
}

func playGame(searchArea, numCharges int) {  # 定义函数 playGame，参数为 searchArea 和 numCharges
	rand.Seed(time.Now().UTC().UnixNano())  # 使用当前时间的纳秒数作为随机数种子
	fmt.Println("\nYou are the captain of the destroyer USS Computer.")  # 打印提示信息
	fmt.Println("An enemy sub has been causing you trouble. Your")  # 打印提示信息
    fmt.Printf("mission is to destroy it. You have %d shots.\n", numCharges)  # 打印任务提示信息和可用的射击次数
    fmt.Println("Specify depth charge explosion point with a")  # 打印指定深度炸弹爆炸点的提示信息
    fmt.Println("trio of numbers -- the first two are the")  # 打印提示信息
    fmt.Println("surface coordinates; the third is the depth.")  # 打印提示信息
    fmt.Println("\nGood luck!")  # 打印祝福信息
    fmt.Println()  # 打印空行

    subPos := getRandomPosition(searchArea)  # 从搜索区域获取随机位置作为潜艇位置

    for c := 0; c < numCharges; c++:  # 循环执行射击次数的次数
        fmt.Printf("\nTrial #%d\n", c+1)  # 打印当前射击的次数

        shot := getShot()  # 获取玩家输入的射击坐标

        if shot[0] == subPos[0] && shot[1] == subPos[1] && shot[2] == subPos[2]:  # 判断玩家射击的坐标是否与潜艇位置相同
            fmt.Printf("\nB O O M ! ! You found it in %d tries!\n", c+1)  # 如果相同则打印找到潜艇的信息和尝试次数
            askForNewGame()  # 要求玩家开始新游戏
        else:
            showShotResult(shot, subPos)  # 如果不相同则展示射击结果和潜艇位置
	}

	// out of depth charges
	// 打印出深度炸弹用尽的消息
	fmt.Println("\nYou have been torpedoed! Abandon ship!")
	// 打印出潜艇的位置坐标
	fmt.Printf("The submarine was at %d %d %d\n", subPos[0], subPos[1], subPos[2])
	// 请求开始新游戏
	askForNewGame()

}

func main() {
	// 显示欢迎消息
	showWelcome()

	// 获取搜索区域和深度炸弹数量
	searchArea, numCharges := getNumCharges()

	// 开始游戏
	playGame(searchArea, numCharges)
}
```