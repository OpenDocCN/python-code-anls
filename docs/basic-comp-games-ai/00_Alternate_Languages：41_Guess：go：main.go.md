# `d:/src/tocomm/basic-computer-games\00_Alternate_Languages\41_Guess\go\main.go`

```
# 导入所需的包
import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"
)

# 定义函数，打印游戏介绍
func printIntro() {
	# 打印游戏标题
	fmt.Println("                   Guess")
	# 打印游戏信息
	fmt.Println("Creative Computing  Morristown, New Jersey")
	# 打印空行
	fmt.Println()
	# 打印空行
	fmt.Println()
	# 打印空行
	fmt.Println()
	# 打印游戏说明
	fmt.Println("This is a number guessing game. I'll think")
	fmt.Println("of a number between 1 and any limit you want.")
}
# 打印提示信息，要求用户猜测内容
fmt.Println("Then you have to guess what it is")

# 定义一个函数，返回两个整数值
func getLimit() (int, int) {
    # 创建一个从标准输入读取数据的扫描器
    scanner := bufio.NewScanner(os.Stdin)

    # 无限循环，直到用户输入有效数据
    for {
        # 打印提示信息，要求用户输入限制值
        fmt.Println("What limit do you want?")
        # 从标准输入读取数据
        scanner.Scan()

        # 将用户输入的数据转换为整数，同时检查是否转换成功和是否大于等于0
        limit, err := strconv.Atoi(scanner.Text())
        if err != nil || limit < 0 {
            # 如果转换失败或者小于0，打印提示信息，要求用户重新输入
            fmt.Println("Please enter a number greater or equal to 1")
            continue
        }

        # 根据用户输入的限制值计算目标值
        limitGoal := int((math.Log(float64(limit)) / math.Log(2)) + 1)
        # 返回用户输入的限制值和计算得到的目标值
        return limit, limitGoal
    }
}
}

# 主函数
func main() {
    # 设置随机数种子
    rand.Seed(time.Now().UnixNano())
    # 打印游戏介绍
    printIntro()

    # 创建一个从标准输入读取数据的扫描器
    scanner := bufio.NewScanner(os.Stdin)

    # 获取猜测范围的上限和目标值
    limit, limitGoal := getLimit()

    # 初始化猜测次数为1，设置仍在猜测中的标志为true，设置赢得游戏的标志为false
    guessCount := 1
    stillGuessing := true
    won := false
    # 生成一个随机数作为程序所思考的数字
    myGuess := int(float64(limit)*rand.Float64() + 1)

    # 打印程序所思考的数字范围
    fmt.Printf("I'm thinking of a number between 1 and %d\n", limit)
    fmt.Println("Now you try to guess what it is.")

    # 进入猜数字的循环
    for stillGuessing {
        # 从标准输入中读取用户的猜测
        scanner.Scan()
		# 将输入的字符串转换为整数，并返回转换后的整数和可能出现的错误
		n, err := strconv.Atoi(scanner.Text()
		# 如果出现错误，打印提示信息并继续循环
		if err != nil {
			fmt.Println("Please enter a number greater or equal to 1")
			continue
		}

		# 如果输入的数小于0，跳出循环
		if n < 0 {
			break
		}

		# 打印空行
		fmt.Print("\n\n\n")
		# 如果输入的数小于待猜测的数，打印提示信息并增加猜测次数
		if n < myGuess {
			fmt.Println("Too low. Try a bigger answer")
			guessCount += 1
		# 如果输入的数大于待猜测的数，打印提示信息并增加猜测次数
		} else if n > myGuess {
			fmt.Println("Too high. Try a smaller answer")
			guessCount += 1
		# 如果输入的数等于待猜测的数，打印提示信息并设置游戏胜利标志为真
		} else {
			fmt.Printf("That's it! You got it in %d tries\n", guessCount)
			won = true
			stillGuessing = false  # 将变量 stillGuessing 设置为 false，表示猜测结束
		}
	}

	if won {  # 如果游戏胜利
		if guessCount < limitGoal {  # 如果猜测次数小于目标次数
			fmt.Println("Very good.")  # 输出"非常好"
		} else if guessCount == limitGoal {  # 如果猜测次数等于目标次数
			fmt.Println("Good.")  # 输出"好"
		} else {  # 否则
			fmt.Printf("You should have been able to get it in only %d guesses.\n", limitGoal)  # 输出"你本应该只需要%d次猜测就能猜中。"，并将目标次数填入
		}
		fmt.Print("\n\n\n")  # 输出三个空行
	}
}
```