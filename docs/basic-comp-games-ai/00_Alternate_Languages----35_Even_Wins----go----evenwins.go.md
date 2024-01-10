# `basic-computer-games\00_Alternate_Languages\35_Even_Wins\go\evenwins.go`

```
package main

import (
    "bufio"  // 导入用于读取输入的包
    "fmt"    // 导入用于格式化输出的包
    "os"     // 导入用于访问操作系统功能的包
    "strconv"  // 导入用于字符串转换的包
    "strings"  // 导入处理字符串的包
)

const MAXTAKE = 4  // 定义最大可取的数量为4

type PlayerType int8  // 定义玩家类型为int8

const (
    HUMAN PlayerType = iota  // 定义玩家类型为HUMAN
    COMPUTER  // 定义玩家类型为COMPUTER
)

type Game struct {  // 定义游戏结构体
    table    int  // 中间的大理石数量
    human    int  // 玩家持有的大理石数量
    computer int  // 电脑持有的大理石数量
}

func NewGame() Game {  // 创建新游戏
    g := Game{}  // 初始化游戏结构体
    g.table = 27  // 设置中间的大理石数量为27

    return g  // 返回初始化后的游戏结构体
}

func printIntro() {  // 打印游戏介绍
    fmt.Println("Welcome to Even Wins!")  // 欢迎语
    fmt.Println("Based on evenwins.bas from Creative Computing")  // 基于Creative Computing的evenwins.bas
    fmt.Println()  // 空行
    fmt.Println("Even Wins is a two-person game. You start with")  // 游戏介绍
    fmt.Println("27 marbles in the middle of the table.")  // 游戏介绍
    // ... 其余游戏介绍
}

func (g *Game) printBoard() {  // 打印游戏面板
    fmt.Println()  // 空行
    fmt.Printf(" marbles in the middle: %d\n", g.table)  // 打印中间的大理石数量
    fmt.Printf("    # marbles you have: %d\n", g.human)  // 打印玩家持有的大理石数量
    fmt.Printf("# marbles computer has: %d\n", g.computer)  // 打印电脑持有的大理石数量
    fmt.Println()  // 空行
}

func (g *Game) gameOver() {  // 游戏结束
    fmt.Println()  // 空行
    fmt.Println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")  // 打印游戏结束提示
    fmt.Println("!! All the marbles are taken: Game Over!")  // 打印游戏结束提示
    fmt.Println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")  // 打印游戏结束提示
    fmt.Println()  // 空行
    g.printBoard()  // 调用打印游戏面板的方法
    if g.human%2 == 0 {  // 判断玩家持有的大理石数量是否为偶数
        fmt.Println("You are the winner! Congratulations!")  // 打印玩家获胜提示
    } else {
        fmt.Println("The computer wins: all hail mighty silicon!")  // 打印电脑获胜提示
    }
    fmt.Println()  // 空行
}

func getPlural(count int) string {  // 获取复数形式的大理石
    m := "marble"  // 默认为单数形式
    if count > 1 {  // 如果数量大于1
        m += "s"  // 添加s变为复数形式
    }
    return m  // 返回大理石的形式
}

func (g *Game) humanTurn() {  // 玩家回合
    scanner := bufio.NewScanner(os.Stdin)  // 创建用于读取输入的扫描器
    maxAvailable := MAXTAKE  // 最大可取数量默认为4
    if g.table < MAXTAKE {  // 如果中间的大理石数量小于4
        maxAvailable = g.table  // 最大可取数量为中间的大理石数量
    }

    fmt.Println("It's your turn!")  // 提示玩家轮到他们了
    # 无限循环，直到满足条件才会退出
    for {
        # 提示用户输入要拿走的弹珠数量
        fmt.Printf("Marbles to take? (1 - %d) --> ", maxAvailable)
        # 读取用户输入
        scanner.Scan()
        # 将用户输入的字符串转换为整数
        n, err := strconv.Atoi(scanner.Text())
        # 如果转换出错，提示用户重新输入
        if err != nil {
            fmt.Printf("\n  Please enter a whole number from 1 to %d\n", maxAvailable)
            continue
        }
        # 如果输入小于1，提示用户至少要拿一个弹珠
        if n < 1 {
            fmt.Println("\n  You must take at least 1 marble!")
            continue
        }
        # 如果输入大于可用的弹珠数量，提示用户最多只能拿这么多
        if n > maxAvailable {
            fmt.Printf("\n  You can take at most %d %s\n", maxAvailable, getPlural(maxAvailable))
            continue
        }
        # 打印确认信息，拿走指定数量的弹珠
        fmt.Printf("\nOkay, taking %d %s ...\n", n, getPlural(n))
        # 更新游戏桌上和玩家手中的弹珠数量
        g.table -= n
        g.human += n
        # 退出循环
        return
    }
# 计算机的回合
func (g *Game) computerTurn() {
    # 初始化取走的弹珠数量
    marblesToTake := 0

    # 打印提示信息，轮到计算机的回合
    fmt.Println("It's the computer's turn ...")
    # 计算余数
    r := float64(g.table - 6*int((g.table)/6))

    # 如果玩家的弹珠数量是偶数
    if int(g.human/2) == g.human/2 {
        # 如果余数小于1.5或大于5.3，取走1个弹珠，否则取走余数减1个弹珠
        if r < 1.5 || r > 5.3 {
            marblesToTake = 1
        } else {
            marblesToTake = int(r - 1)
        }
    } else if float64(g.table) < 4.2 {
        # 如果弹珠数量小于4.2，取走4个弹珠
        marblesToTake = 4
    } else if r > 3.4 {
        # 如果余数大于3.4并且小于4.7或者大于3.5，取走4个弹珠
        if r < 4.7 || r > 3.5 {
            marblesToTake = 4
        }
    } else {
        # 否则取走余数加1个弹珠
        marblesToTake = int(r + 1)
    }

    # 打印计算机取走的弹珠数量和对应的单复数形式
    fmt.Printf("Computer takes %d %s ...\n", marblesToTake, getPlural(marblesToTake))
    # 更新弹珠数量
    g.table -= marblesToTake
    g.computer += marblesToTake
}

# 进行游戏
func (g *Game) play(playersTurn PlayerType) {
    # 打印游戏板
    g.printBoard()

    # 循环进行游戏
    for {
        # 如果弹珠数量为0，游戏结束
        if g.table == 0 {
            g.gameOver()
            return
        } else if playersTurn == HUMAN:
            # 如果轮到玩家的回合，执行玩家的回合操作
            g.humanTurn()
            # 打印游戏板
            g.printBoard()
            # 轮到计算机的回合
            playersTurn = COMPUTER
        } else {
            # 否则执行计算机的回合操作
            g.computerTurn()
            # 打印游戏板
            g.printBoard()
            # 轮到玩家的回合
            playersTurn = HUMAN
        }
    }
}

# 获取先手玩家
func getFirstPlayer() PlayerType {
    # 创建标准输入的扫描器
    scanner := bufio.NewScanner(os.Stdin)

    # 循环直到获取有效输入
    for {
        # 提示玩家选择先手或后手
        fmt.Println("Do you want to play first? (y/n) --> ")
        scanner.Scan()

        # 如果输入为'y'，返回玩家先手
        if strings.ToUpper(scanner.Text()) == "Y" {
            return HUMAN
        } else if strings.ToUpper(scanner.Text()) == "N" {
            # 如果输入为'n'，返回计算机先手
            return COMPUTER
        } else {
            # 否则提示重新输入
            fmt.Println()
            fmt.Println("Please enter 'y' if you want to play first,")
            fmt.Println("or 'n' if you want to play second.")
            fmt.Println()
        }
    }
}

# 主函数
func main() {
    # 创建标准输入的扫描器
    scanner := bufio.NewScanner(os.Stdin)

    # 打印游戏介绍
    printIntro()
    # 无限循环，创建新游戏对象
    for {
        # 创建新游戏对象
        g := NewGame()
        
        # 调用play方法，传入第一个玩家对象
        g.play(getFirstPlayer())
        
        # 打印提示信息，询问是否再次游戏
        fmt.Println("\nWould you like to play again? (y/n) --> ")
        # 扫描用户输入
        scanner.Scan()
        # 如果用户输入的内容转换为大写后等于"Y"，则执行下面的代码块
        if strings.ToUpper(scanner.Text()) == "Y" {
            # 打印提示信息，表示再次开始游戏
            fmt.Println("\nOk, let's play again ...")
        } else {
            # 打印感谢信息，表示结束游戏
            fmt.Println("\nOk, thanks for playing ... goodbye!")
            # 结束循环
            return
        }
    }
# 闭合前面的函数定义
```