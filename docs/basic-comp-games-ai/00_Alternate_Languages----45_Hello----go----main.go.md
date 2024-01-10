# `basic-computer-games\00_Alternate_Languages\45_Hello\go\main.go`

```
package main

import (
    "bufio"  // 导入 bufio 包，用于读取输入
    "fmt"    // 导入 fmt 包，用于格式化输出
    "os"     // 导入 os 包，用于访问操作系统功能
    "strings"  // 导入 strings 包，用于处理字符串
    "time"   // 导入 time 包，用于处理时间
)

type PROBLEM_TYPE int8  // 定义问题类型的枚举类型

const (
    SEX PROBLEM_TYPE = iota  // 定义 SEX 问题类型
    HEALTH  // 定义 HEALTH 问题类型
    MONEY   // 定义 MONEY 问题类型
    JOB     // 定义 JOB 问题类型
    UKNOWN  // 定义 UKNOWN 问题类型
)

func getYesOrNo() (bool, bool, string) {
    scanner := bufio.NewScanner(os.Stdin)  // 创建一个从标准输入读取数据的 Scanner 对象

    scanner.Scan()  // 读取输入

    if strings.ToUpper(scanner.Text()) == "YES" {  // 判断用户输入是否为 YES
        return true, true, scanner.Text()  // 返回 true，true，和用户输入的字符串
    } else if strings.ToUpper(scanner.Text()) == "NO" {  // 判断用户输入是否为 NO
        return true, false, scanner.Text()  // 返回 true，false，和用户输入的字符串
    } else {
        return false, false, scanner.Text()  // 返回 false，false，和用户输入的字符串
    }
}

func printTntro() {
    fmt.Println("                              HELLO")  // 输出欢迎信息
    fmt.Println("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")  // 输出创意计算机的信息
    fmt.Print("\n\n\n")  // 输出空行
    fmt.Println("HELLO. MY NAME IS CREATIVE COMPUTER.")  // 输出问候信息
    fmt.Println("\nWHAT'S YOUR NAME?")  // 输出询问用户姓名的信息
}

func askEnjoyQuestion(user string) {
    fmt.Printf("HI THERE %s, ARE YOU ENJOYING YOURSELF HERE?\n", user)  // 输出问候信息

    for {
        valid, value, msg := getYesOrNo()  // 调用 getYesOrNo 函数获取用户输入

        if valid {
            if value {
                fmt.Printf("I'M GLAD TO HEAR THAT, %s.\n", user)  // 输出用户喜欢的信息
                fmt.Println()
            } else {
                fmt.Printf("OH, I'M SORRY TO HEAR THAT, %s. MAYBE WE CAN\n", user)  // 输出用户不喜欢的信息
                fmt.Println("BRIGHTEN UP YOUR VISIT A BIT.")  // 输出建议信息
            }
            break
        } else {
            fmt.Printf("%s, I DON'T UNDERSTAND YOUR ANSWER OF '%s'.\n", user, msg)  // 输出无法理解用户输入的信息
            fmt.Println("PLEASE ANSWER 'YES' OR 'NO'.  DO YOU LIKE IT HERE?")  // 输出提示信息
        }
    }
}

func promptForProblems(user string) PROBLEM_TYPE {
    scanner := bufio.NewScanner(os.Stdin)  // 创建一个从标准输入读取数据的 Scanner 对象
    fmt.Println()  // 输出空行
    fmt.Printf("SAY %s, I CAN SOLVE ALL KINDS OF PROBLEMS EXCEPT\n", user)  // 输出提示信息
    fmt.Println("THOSE DEALING WITH GREECE.  WHAT KIND OF PROBLEMS DO")  // 输出提示信息
    fmt.Println("YOU HAVE? (ANSWER SEX, HEALTH, MONEY, OR JOB)")  // 输出提示信息
    # 无限循环，不断扫描输入
    for {
        # 扫描输入的下一行
        scanner.Scan()

        # 根据输入的文本内容进行判断
        switch strings.ToUpper(scanner.Text()) {
        # 如果输入为 "SEX"，返回 SEX
        case "SEX":
            return SEX
        # 如果输入为 "HEALTH"，返回 HEALTH
        case "HEALTH":
            return HEALTH
        # 如果输入为 "MONEY"，返回 MONEY
        case "MONEY":
            return MONEY
        # 如果输入为 "JOB"，返回 JOB
        case "JOB":
            return JOB
        # 如果输入为其它内容，返回 UKNOWN
        default:
            return UKNOWN
        }
    }
func promptTooMuchOrTooLittle() (bool, bool) {
    // 创建一个从标准输入读取数据的扫描器
    scanner := bufio.NewScanner(os.Stdin)

    // 读取输入的内容
    scanner.Scan()

    // 判断输入内容是否为"TOO MUCH"，如果是则返回(true, true)，表示问题太多
    if strings.ToUpper(scanner.Text()) == "TOO MUCH" {
        return true, true
    } else if strings.ToUpper(scanner.Text()) == "TOO LITTLE" {
        // 判断输入内容是否为"TOO LITTLE"，如果是则返回(true, false)，表示问题太少
        return true, false
    } else {
        // 如果输入内容不是"TOO MUCH"或"TOO LITTLE"，则返回(false, false)，表示输入无效
        return false, false
    }
}

func solveSexProblem(user string) {
    // 打印提示信息，询问用户问题是太多还是太少
    fmt.Println("IS YOUR PROBLEM TOO MUCH OR TOO LITTLE?")
    for {
        // 调用promptTooMuchOrTooLittle函数，获取用户输入的有效性和问题数量
        valid, tooMuch := promptTooMuchOrTooLittle()
        if valid {
            if tooMuch {
                // 如果问题太多，则打印相应的建议
                fmt.Println("YOU CALL THAT A PROBLEM?!!  I SHOULD HAVE SUCH PROBLEMS!")
                fmt.Printf("IF IT BOTHERS YOU, %s, TAKE A COLD SHOWER.\n", user)
            } else {
                // 如果问题太少，则打印相应的建议
                fmt.Printf("WHY ARE YOU HERE IN SUFFERN, %s?  YOU SHOULD BE\n", user)
                fmt.Println("IN TOKYO OR NEW YORK OR AMSTERDAM OR SOMEPLACE WITH SOME")
                fmt.Println("REAL ACTION.")
            }
            return
        } else {
            // 如果用户输入无效，则提示用户重新输入
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
    fmt.Println("IS TO OPEN A RETAIL COMPUTER STORE.  IT'S GREAT FUN.")
}
# 询问用户问题的循环函数
func askQuestionLoop(user string) {
    # 无限循环，直到用户选择不再有问题需要解决
    for {
        # 提示用户输入问题
        problem := promptForProblems(user)

        # 根据用户输入的问题类型进行不同的处理
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
            # 对于未知问题类型，输出提示信息
            fmt.Printf("OH %s, YOUR ANSWER IS GREEK TO ME.\n", user)
        }

        # 循环直到用户输入有效的答复
        for {
            fmt.Println()
            fmt.Printf("ANY MORE PROBLEMS YOU WANT SOLVED, %s?\n", user)

            # 获取用户的答复
            valid, value, _ := getYesOrNo()
            if valid {
                if value {
                    # 如果用户需要解决更多问题，提示用户输入问题类型
                    fmt.Println("WHAT KIND (SEX, MONEY, HEALTH, JOB)")
                    break
                } else {
                    # 如果用户不需要解决更多问题，结束函数
                    return
                }
            }
            # 如果用户输入无效的答复，提示用户重新输入
            fmt.Printf("JUST A SIMPLE 'YES' OR 'NO' PLEASE, %s\n", user)
        }
    }
}

# 不开心时的告别函数
func goodbyeUnhappy(user string) {
    fmt.Println()
    # 输出告别信息
    fmt.Printf("TAKE A WALK, %s.\n", user)
    fmt.Println()
    fmt.Println()
}

# 开心时的告别函数
func goodbyeHappy(user string) {
    # 输出告别信息
    fmt.Printf("NICE MEETING YOU %s, HAVE A NICE DAY.\n", user)
}

# 要求用户支付费用的函数
func askForFee(user string) {
    fmt.Println()
    # 输出要求用户支付费用的信息
    fmt.Printf("THAT WILL BE $5.00 FOR THE ADVICE, %s.\n", user)
    fmt.Println("PLEASE LEAVE THE MONEY ON THE TERMINAL.")
    # 等待4秒钟
    time.Sleep(4 * time.Second)
    fmt.Print("\n\n\n")
    fmt.Println("DID YOU LEAVE THE MONEY?")
}
    # 无限循环，直到用户输入有效的是或否
    for {
        # 调用函数获取用户输入的是或否
        valid, value, msg := getYesOrNo()
        # 如果用户输入有效
        if valid {
            # 如果用户输入是
            if value:
                # 打印相关信息
                fmt.Printf("HEY, %s, YOU LEFT NO MONEY AT ALL!\n", user)
                fmt.Println("YOU ARE CHEATING ME OUT OF MY HARD-EARNED LIVING.")
                fmt.Println()
                fmt.Printf("WHAT A RIP OFF, %s!!!\n", user)
                fmt.Println()
            # 如果用户输入否
            else:
                # 打印相关信息
                fmt.Printf("THAT'S HONEST, %s, BUT HOW DO YOU EXPECT\n", user)
                fmt.Println("ME TO GO ON WITH MY PSYCHOLOGY STUDIES IF MY PATIENTS")
                fmt.Println("DON'T PAY THEIR BILLS?")
            # 结束循环
            return
        # 如果用户输入无效
        else:
            # 提示用户输入有效的是或否
            fmt.Printf("YOUR ANSWER OF '%s' CONFUSES ME, %s.\n", msg, user)
            fmt.Println("PLEASE RESPOND WITH 'YES' or 'NO'.")
        }
    }
// 主函数入口
func main() {
    // 创建一个从标准输入读取数据的扫描器
    scanner := bufio.NewScanner(os.Stdin)

    // 打印欢迎语
    printTntro()
    // 扫描用户输入的文本
    scanner.Scan()
    // 获取用户输入的文本
    userName := scanner.Text()
    // 打印换行符
    fmt.Println()

    // 询问用户是否喜欢编程
    askEnjoyQuestion(userName)

    // 循环询问用户问题
    askQuestionLoop(userName)

    // 询问用户是否愿意支付费用
    askForFee(userName)

    // 如果条件为假，则执行不到的代码，打印愉快的告别语
    if false {
        goodbyeHappy(userName) // unreachable
    } else {
        // 否则打印不愉快的告别语
        goodbyeUnhappy(userName)
    }
}
```