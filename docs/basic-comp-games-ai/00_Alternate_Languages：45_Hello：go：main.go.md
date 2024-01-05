# `d:/src/tocomm/basic-computer-games\00_Alternate_Languages\45_Hello\go\main.go`

```
package main  // 声明当前文件所属的包为 main

import (
	"bufio"  // 导入 bufio 包，用于提供带缓冲的 I/O
	"fmt"  // 导入 fmt 包，用于格式化输入输出
	"os"  // 导入 os 包，提供了对操作系统功能的访问
	"strings"  // 导入 strings 包，提供了对字符串的操作函数
	"time"  // 导入 time 包，提供了时间的显示和测量用的函数
)

type PROBLEM_TYPE int8  // 声明一个名为 PROBLEM_TYPE 的自定义类型为 int8

const (
	SEX PROBLEM_TYPE = iota  // 定义常量 SEX 为 PROBLEM_TYPE 类型，并初始化为 iota（0）
	HEALTH  // 定义常量 HEALTH 为 PROBLEM_TYPE 类型
	MONEY  // 定义常量 MONEY 为 PROBLEM_TYPE 类型
	JOB  // 定义常量 JOB 为 PROBLEM_TYPE 类型
	UKNOWN  // 定义常量 UKNOWN 为 PROBLEM_TYPE 类型
)
func getYesOrNo() (bool, bool, string) {
	// 创建一个从标准输入读取数据的扫描器
	scanner := bufio.NewScanner(os.Stdin)

	// 读取输入
	scanner.Scan()

	// 如果输入为"YES"，返回true，true，输入的字符串
	if strings.ToUpper(scanner.Text()) == "YES" {
		return true, true, scanner.Text()
	} 
	// 如果输入为"NO"，返回true，false，输入的字符串
	else if strings.ToUpper(scanner.Text()) == "NO" {
		return true, false, scanner.Text()
	} 
	// 如果输入既不是"YES"也不是"NO"，返回false，false，输入的字符串
	else {
		return false, false, scanner.Text()
	}
}

func printTntro() {
	// 打印欢迎信息
	fmt.Println("                              HELLO")
	fmt.Println("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
	fmt.Print("\n\n\n")
	fmt.Println("HELLO. MY NAME IS CREATIVE COMPUTER.")
	fmt.Println("\nWHAT'S YOUR NAME?")
}
}

# 定义一个名为 askEnjoyQuestion 的函数，参数为 user
func askEnjoyQuestion(user string) {
	# 打印问候语，使用参数 user
	fmt.Printf("HI THERE %s, ARE YOU ENJOYING YOURSELF HERE?\n", user)

	# 进入无限循环
	for {
		# 调用 getYesOrNo 函数，获取用户输入的是否喜欢的信息
		valid, value, msg := getYesOrNo()

		# 如果用户输入有效
		if valid {
			# 如果用户喜欢
			if value {
				# 打印肯定回答的消息，使用参数 user
				fmt.Printf("I'M GLAD TO HEAR THAT, %s.\n", user)
				# 打印空行
				fmt.Println()
			} else {
				# 打印否定回答的消息，使用参数 user
				fmt.Printf("OH, I'M SORRY TO HEAR THAT, %s. MAYBE WE CAN\n", user)
				# 打印建议亮化访问的消息
				fmt.Println("BRIGHTEN UP YOUR VISIT A BIT.")
			}
			# 退出循环
			break
		} else {
			# 如果用户输入无效，打印提示信息，使用参数 user 和 msg
			fmt.Printf("%s, I DON'T UNDERSTAND YOUR ANSWER OF '%s'.\n", user, msg)
			# 打印提示信息
			fmt.Println("PLEASE ANSWER 'YES' OR 'NO'.  DO YOU LIKE IT HERE?")
		}
	}
}

func promptForProblems(user string) PROBLEM_TYPE {
    // 创建一个新的扫描器来读取标准输入
    scanner := bufio.NewScanner(os.Stdin)
    fmt.Println()
    // 打印提示信息
    fmt.Printf("SAY %s, I CAN SOLVE ALL KINDS OF PROBLEMS EXCEPT\n", user)
    fmt.Println("THOSE DEALING WITH GREECE.  WHAT KIND OF PROBLEMS DO")
    fmt.Println("YOU HAVE? (ANSWER SEX, HEALTH, MONEY, OR JOB)")
    // 循环读取用户输入
    for {
        scanner.Scan()
        // 根据用户输入的不同情况返回不同的问题类型
        switch strings.ToUpper(scanner.Text()) {
        case "SEX":
            return SEX
        case "HEALTH":
            return HEALTH
        case "MONEY":
            return MONEY
		case "JOB":  // 如果输入的值为"JOB"，则返回JOB
			return JOB
		default:  // 如果输入的值不是"JOB"，则返回UKNOWN
			return UKNOWN
		}
	}
}

func promptTooMuchOrTooLittle() (bool, bool) {
	scanner := bufio.NewScanner(os.Stdin)  // 创建一个从标准输入读取数据的扫描器

	scanner.Scan()  // 扫描输入

	if strings.ToUpper(scanner.Text()) == "TOO MUCH" {  // 如果输入的值转换为大写后等于"TOO MUCH"，则返回true, true
		return true, true
	} else if strings.ToUpper(scanner.Text()) == "TOO LITTLE" {  // 如果输入的值转换为大写后等于"TOO LITTLE"，则返回true, false
		return true, false
	} else {  // 如果输入的值既不是"TOO MUCH"也不是"TOO LITTLE"，则返回false, false
		return false, false
	}
}

# 解决性别问题
func solveSexProblem(user string) {
    fmt.Println("IS YOUR PROBLEM TOO MUCH OR TOO LITTLE?")
    for {
        # 调用 promptTooMuchOrTooLittle 函数，获取用户输入的问题类型
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

func solveHealthProblem(user string) {
	// 打印健康问题的建议
	fmt.Printf("MY ADVICE TO YOU %s IS:\n", user)
	fmt.Println("     1.  TAKE TWO ASPRIN")
	fmt.Println("     2.  DRINK PLENTY OF FLUIDS (ORANGE JUICE, NOT BEER!)")
	fmt.Println("     3.  GO TO BED (ALONE)")
}

func solveMoneyProblem(user string) {
	// 打印金钱问题的建议
	fmt.Printf("SORRY, %s, I'M BROKE TOO.  WHY DON'T YOU SELL\n", user)
	fmt.Println("ENCYCLOPEADIAS OR MARRY SOMEONE RICH OR STOP EATING")
	fmt.Println("SO YOU WON'T NEED SO MUCH MONEY?")
}

func solveJobProblem(user string) {
	// 打印工作问题的建议
	fmt.Printf("I CAN SYMPATHIZE WITH YOU %s.  I HAVE TO WORK\n", user)
	fmt.Println("VERY LONG HOURS FOR NO PAY -- AND SOME OF MY BOSSES")
	fmt.Printf("REALLY BEAT ON MY KEYBOARD.  MY ADVICE TO YOU, %s,\n", user)
```
```plaintext
	fmt.Println("IS TO OPEN A RETAIL COMPUTER STORE.  IT'S GREAT FUN.")
}
```
这行代码是打印字符串"IS TO OPEN A RETAIL COMPUTER STORE.  IT'S GREAT FUN."。

```go
func askQuestionLoop(user string) {
```
这行代码是定义一个名为askQuestionLoop的函数，它接受一个字符串类型的参数user。

```go
	for {
		problem := promptForProblems(user)
```
这行代码是一个无限循环，每次循环都会调用promptForProblems函数，并将返回的值赋给problem变量。

```go
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
```
这段代码是一个switch语句，根据problem的值来执行不同的操作。如果problem的值是SEX、HEALTH、MONEY或JOB，则分别调用solveSexProblem、solveHealthProblem、solveMoneyProblem或solveJobProblem函数。如果problem的值是UKNOWN，则打印"OH %s, YOUR ANSWER IS GREEK TO ME."，其中%s会被user的值替换。
		for {
			# 打印空行
			fmt.Println()
			# 格式化打印字符串，询问用户是否还有其他问题需要解决
			fmt.Printf("ANY MORE PROBLEMS YOU WANT SOLVED, %s?\n", user)

			# 调用函数获取用户输入的是或否
			valid, value, _ := getYesOrNo()
			if valid:
				# 如果用户输入是
				if value:
					# 打印提示，询问用户问题的类型
					fmt.Println("WHAT KIND (SEX, MONEY, HEALTH, JOB)")
					# 跳出循环
					break
				# 如果用户输入否
				else:
					# 返回
					return
			# 如果用户输入无效
			fmt.Printf("JUST A SIMPLE 'YES' OR 'NO' PLEASE, %s\n", user)
		}
	}
}

# 定义函数，用于向用户道别
func goodbyeUnhappy(user string) {
	# 打印空行
	fmt.Println()
	fmt.Printf("TAKE A WALK, %s.\n", user)  # 打印带有用户名称的消息
	fmt.Println()  # 打印空行
	fmt.Println()  # 打印空行
}

func goodbyeHappy(user string) {
	fmt.Printf("NICE MEETING YOU %s, HAVE A NICE DAY.\n", user)  # 打印带有用户名称的道别消息
}

func askForFee(user string) {
	fmt.Println()  # 打印空行
	fmt.Printf("THAT WILL BE $5.00 FOR THE ADVICE, %s.\n", user)  # 打印要求用户支付咨询费用的消息
	fmt.Println("PLEASE LEAVE THE MONEY ON THE TERMINAL.")  # 提示用户在终端上留下钱
	time.Sleep(4 * time.Second)  # 程序暂停4秒
	fmt.Print("\n\n\n")  # 打印三个空行
	fmt.Println("DID YOU LEAVE THE MONEY?")  # 询问用户是否留下了钱

	for {
		valid, value, msg := getYesOrNo()  # 调用函数获取用户输入的是或否
		if valid:  # 如果输入有效
			if value {  # 如果value为真
				fmt.Printf("HEY, %s, YOU LEFT NO MONEY AT ALL!\n", user)  # 打印用户没有留下任何钱的消息
				fmt.Println("YOU ARE CHEATING ME OUT OF MY HARD-EARNED LIVING.")  # 打印用户欺骗程序员的消息
				fmt.Println()
				fmt.Printf("WHAT A RIP OFF, %s!!!\n", user)  # 打印用户欺骗的消息
				fmt.Println()
			} else {  # 如果value为假
				fmt.Printf("THAT'S HONEST, %s, BUT HOW DO YOU EXPECT\n", user)  # 打印用户诚实的消息
				fmt.Println("ME TO GO ON WITH MY PSYCHOLOGY STUDIES IF MY PATIENTS")  # 打印继续心理学研究的消息
				fmt.Println("DON'T PAY THEIR BILLS?")  # 打印患者不支付账单的消息
			}
			return  # 返回
		} else {  # 否则
			fmt.Printf("YOUR ANSWER OF '%s' CONFUSES ME, %s.\n", msg, user)  # 打印用户回答混淆的消息
			fmt.Println("PLEASE RESPOND WITH 'YES' or 'NO'.")  # 提示用户回答是或否
		}
	}
}

func main() {  # 主函数
	scanner := bufio.NewScanner(os.Stdin)  // 创建一个新的扫描器，用于从标准输入读取数据

	printTntro()  // 调用函数打印欢迎信息

	scanner.Scan()  // 调用扫描器的Scan方法，等待用户输入
	userName := scanner.Text()  // 获取用户输入的文本并存储在变量userName中
	fmt.Println()  // 打印一个空行

	askEnjoyQuestion(userName)  // 调用函数询问用户是否喜欢编程

	askQuestionLoop(userName)  // 调用函数循环询问用户编程相关问题

	askForFee(userName)  // 调用函数询问用户是否愿意支付费用

	if false {  // 如果条件为假
		goodbyeHappy(userName)  // 调用函数向用户道别并祝愿愉快（不可达）
	} else {  // 否则
		goodbyeUnhappy(userName)  // 调用函数向用户道别并表达遗憾
	}
}
bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
zip.close()  # 关闭 ZIP 对象
```