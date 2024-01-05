# `d:/src/tocomm/basic-computer-games\00_Alternate_Languages\25_Chief\go\main.go`

```
package main  // 声明当前文件所属的包

import (
	"bufio"  // 导入 bufio 包，用于提供缓冲 I/O
	"fmt"  // 导入 fmt 包，用于格式化输入输出
	"os"  // 导入 os 包，提供操作系统函数
	"strconv"  // 导入 strconv 包，用于字符串和基本数据类型之间的转换
	"strings"  // 导入 strings 包，提供对字符串的操作
)

func printLightning() {  // 定义名为 printLightning 的函数
	fmt.Println("************************************")  // 打印输出字符串
	n := 24  // 声明并初始化变量 n 为 24
	for n > 16 {  // 当 n 大于 16 时执行循环
		var b strings.Builder  // 声明一个名为 b 的 strings.Builder 类型变量
		b.Grow(n + 3)  // 扩展 b 的容量至 n+3
		for i := 0; i < n; i++ {  // 循环执行 n 次
			b.WriteString(" ")  // 将空格字符串追加到 b 中
		}
		b.WriteString("x x")  // 将 "x x" 字符串追加到 b 中
		fmt.Println(b.String())  # 打印字符串构建器 b 中的内容
		n--  # 将 n 减一
	}
	fmt.Println("                x xxx")  # 打印指定的字符串
	fmt.Println("               x   x")  # 打印指定的字符串
	fmt.Println("              xx xx")  # 打印指定的字符串
	n--  # 将 n 减一
	for n > 8 {  # 当 n 大于 8 时执行以下循环
		var b strings.Builder  # 创建一个新的字符串构建器 b
		b.Grow(n + 3)  # 设置字符串构建器 b 的容量为 n+3
		for i := 0; i < n; i++ {  # 循环 n 次
			b.WriteString(" ")  # 在字符串构建器 b 中添加空格
		}
		b.WriteString("x x")  # 在字符串构建器 b 中添加 "x x"
		fmt.Println(b.String())  # 打印字符串构建器 b 中的内容
		n--  # 将 n 减一
	}
	fmt.Println("        xx")  # 打印指定的字符串
	fmt.Println("       x")  # 打印指定的字符串
	fmt.Println("************************************")  # 打印指定的字符串
}

# 定义一个名为printSolution的函数，参数为n，打印出一系列数学运算的结果
def printSolution(n):
    # 打印出n加3的结果，以及(n+3)除以5的结果
    print("\n%f plus 3 gives %f. This divided by 5 equals %f\n" % (n, n+3, (n+3)/5))
    # 打印出((n+3)/5)乘以8的结果
    print("This times 8 gives %f. If we divide 5 and add 5.\n" % (((n+3)/5)*8))
    # 打印出((((n+3)/5)*8)/5+5)的结果，以及减去1的结果
    print("We get %f, which, minus 1 equals %f\n" % ((((n+3)/5)*8)/5+5, ((((n+3)/5)*8)/5+5)-1)

# 定义一个名为play的函数
def play():
    # 打印出一系列数学运算的步骤
    print("\nTake a Number and ADD 3. Now, Divide this number by 5 and")
    print("multiply by 8. Now, Divide by 5 and add the same. Subtract 1")

    # 调用getFloat函数获取用户输入的数字
    youHave = getFloat("\nWhat do you have?")
    # 计算出计算机猜测的结果
    compGuess = (((youHave-4)*5)/8)*5 - 3
    # 根据计算机猜测的结果，询问用户是否猜对
    if getYesNo("\nI bet your number was %f was I right(Yes or No)? " % compGuess):
        # 如果用户确认猜对，打印出计算机是如何得出结果的
        print("\nHuh, I knew I was unbeatable")
        print("And here is how i did it")
        printSolution(compGuess)
    else:
        # 如果用户否认猜对，询问用户原始的数字是多少
        originalNumber = getFloat("\nHUH!! what was you original number? ")
		# 如果原始数字和计算机猜测的数字相等
		if originalNumber == compGuess {
			# 打印消息表明计算机猜对了
			fmt.Println("\nThat was my guess, AHA i was right")
			fmt.Println("Shamed to accept defeat i guess, don't worry you can master mathematics too")
			fmt.Println("Here is how i did it")
			# 打印计算机是如何猜对的解决方案
			printSolution(compGuess)
		} else {
			# 如果原始数字和计算机猜测的数字不相等
			fmt.Println("\nSo you think you're so smart, EH?")
			fmt.Println("Now, Watch")
			# 打印原始数字的解决方案
			printSolution(originalNumber)

			# 如果用户同意
			if getYesNo("\nNow do you believe me? ") {
				# 打印消息并结束游戏
				print("\nOk, Lets play again sometime bye!!!!")
			} else {
				# 如果用户不同意
				fmt.Println("\nYOU HAVE MADE ME VERY MAD!!!!!")
				fmt.Println("BY THE WRATH OF THE MATHEMATICS AND THE RAGE OF THE GODS")
				fmt.Println("THERE SHALL BE LIGHTNING!!!!!!!")
				# 打印闪电效果
				printLightning()
				fmt.Println("\nI Hope you believe me now, for your own sake")
			}
		}
	}
}

# 定义一个函数，用于从标准输入中获取一个浮点数
func getFloat(prompt string) float64 {
	# 创建一个新的扫描器来读取标准输入
	scanner := bufio.NewScanner(os.Stdin)

	# 无限循环，直到成功获取一个浮点数
	for {
		# 打印提示信息
		fmt.Println(prompt)
		# 读取输入
		scanner.Scan()
		# 尝试将输入转换为浮点数
		val, err := strconv.ParseFloat(scanner.Text(), 64)
		# 如果转换出错，打印错误信息并继续循环
		if err != nil {
			fmt.Println("INVALID INPUT, TRY AGAIN")
			continue
		}
		# 成功获取浮点数，返回该值
		return val
	}
}

# 定义一个函数，用于从标准输入中获取一个布尔值（是/否）
func getYesNo(prompt string) bool {
	# 创建一个新的扫描器来读取标准输入
	scanner := bufio.NewScanner(os.Stdin)
// 打印提示信息
fmt.Println(prompt)
// 从标准输入中扫描用户输入
scanner.Scan()
// 将用户输入转换为大写，并取第一个字符，判断是否为"Y"，返回布尔值
return (strings.ToUpper(scanner.Text())[0:1] == "Y")
```

```
func main() {
    // 打印欢迎信息
    fmt.Println("I am CHIEF NUMBERS FREEK, The GREAT INDIAN MATH GOD.")
    // 调用函数获取用户是否准备好进行测试
    if getYesNo("\nAre you ready to take the test you called me out for(Yes or No)? ") {
        // 如果用户准备好，调用play函数
        play()
    } else {
        // 如果用户不准备好，打印提示信息
        fmt.Println("Ok, Nevermind. Let me go back to my great slumber, Bye")
    }
}
```