# `basic-computer-games\00_Alternate_Languages\08_Batnum\go\main.go`

```
package main

import (
    "bufio"  // 导入 bufio 包，用于读取输入
    "fmt"    // 导入 fmt 包，用于格式化输入输出
    "os"     // 导入 os 包，用于访问操作系统功能
    "strconv"  // 导入 strconv 包，用于字符串和基本数据类型之间的转换
    "strings"  // 导入 strings 包，用于处理字符串
)

type StartOption int8  // 定义 StartOption 类型为 int8
const (
    StartUndefined StartOption = iota  // 定义 StartUndefined 常量为 0
    ComputerFirst                      // 定义 ComputerFirst 常量为 1
    PlayerFirst                        // 定义 PlayerFirst 常量为 2
)

type WinOption int8  // 定义 WinOption 类型为 int8
const (
    WinUndefined WinOption = iota  // 定义 WinUndefined 常量为 0
    TakeLast                        // 定义 TakeLast 常量为 1
    AvoidLast                       // 定义 AvoidLast 常量为 2
)

type GameOptions struct {  // 定义 GameOptions 结构体
    pileSize    int  // 定义 pileSize 字段为 int 类型
    winOption   WinOption  // 定义 winOption 字段为 WinOption 类型
    startOption StartOption  // 定义 startOption 字段为 StartOption 类型
    minSelect   int  // 定义 minSelect 字段为 int 类型
    maxSelect   int  // 定义 maxSelect 字段为 int 类型
}

func NewOptions() *GameOptions {  // 定义 NewOptions 函数，返回 GameOptions 指针
    g := GameOptions{}  // 创建 GameOptions 结构体实例

    g.pileSize = getPileSize()  // 调用 getPileSize 函数获取 pileSize 值
    if g.pileSize < 0 {  // 如果 pileSize 小于 0
        return &g  // 返回 GameOptions 实例的指针
    }

    g.winOption = getWinOption()  // 调用 getWinOption 函数获取 winOption 值
    g.minSelect, g.maxSelect = getMinMax()  // 调用 getMinMax 函数获取 minSelect 和 maxSelect 值
    g.startOption = getStartOption()  // 调用 getStartOption 函数获取 startOption 值

    return &g  // 返回 GameOptions 实例的指针
}

func getPileSize() int {  // 定义 getPileSize 函数，返回 int 类型
    ps := 0  // 初始化 ps 变量为 0
    var err error  // 声明 err 变量为 error 类型
    scanner := bufio.NewScanner(os.Stdin)  // 创建从标准输入读取数据的 Scanner

    for {  // 循环
        fmt.Println("Enter Pile Size ")  // 打印提示信息
        scanner.Scan()  // 读取输入
        ps, err = strconv.Atoi(scanner.Text())  // 将输入转换为整数
        if err == nil {  // 如果转换成功
            break  // 退出循环
        }
    }
    return ps  // 返回输入的 pileSize 值
}

func getWinOption() WinOption {  // 定义 getWinOption 函数，返回 WinOption 类型
    scanner := bufio.NewScanner(os.Stdin)  // 创建从标准输入读取数据的 Scanner

    for {  // 循环
        fmt.Println("ENTER WIN OPTION - 1 TO TAKE LAST, 2 TO AVOID LAST:")  // 打印提示信息
        scanner.Scan()  // 读取输入
        w, err := strconv.Atoi(scanner.Text())  // 将输入转换为整数
        if err == nil && (w == 1 || w == 2) {  // 如果转换成功且输入为 1 或 2
            return WinOption(w)  // 返回输入的 WinOption 值
        }
    }
}

func getStartOption() StartOption {  // 定义 getStartOption 函数，返回 StartOption 类型
    scanner := bufio.NewScanner(os.Stdin)  // 创建从标准输入读取数据的 Scanner

    for {  // 循环
        fmt.Println("ENTER START OPTION - 1 COMPUTER FIRST, 2 YOU FIRST ")  // 打印提示信息
        scanner.Scan()  // 读取输入
        s, err := strconv.Atoi(scanner.Text())  // 将输入转换为整数
        if err == nil && (s == 1 || s == 2) {  // 如果转换成功且输入为 1 或 2
            return StartOption(s)  // 返回输入的 StartOption 值
        }
    }
}

func getMinMax() (int, int) {  // 定义 getMinMax 函数，返回两个 int 类型值
    minSelect := 0  // 初始化 minSelect 变量为 0
    maxSelect := 0  // 初始化 maxSelect 变量为 0
    var minErr error  // 声明 minErr 变量为 error 类型
    var maxErr error  // 声明 maxErr 变量为 error 类型
    scanner := bufio.NewScanner(os.Stdin)  // 创建从标准输入读取数据的 Scanner
    # 无限循环，提示用户输入最小和最大值
    for {
        # 打印提示信息
        fmt.Println("ENTER MIN AND MAX ")
        # 从标准输入中扫描用户输入的值
        scanner.Scan()
        # 获取用户输入的字符串
        enteredValues := scanner.Text()
        # 将用户输入的字符串按空格分割成数组
        vals := strings.Split(enteredValues, " ")
        # 将分割后的字符串转换成整数，同时检查转换错误
        minSelect, minErr = strconv.Atoi(vals[0])
        maxSelect, maxErr = strconv.Atoi(vals[1])
        # 检查转换错误以及最小值和最大值的合法性
        if (minErr == nil) && (maxErr == nil) && (minSelect > 0) && (maxSelect > 0) && (maxSelect > minSelect) {
            # 如果输入合法，返回最小值和最大值
            return minSelect, maxSelect
        }
    }
// 处理玩家的回合 - 询问玩家要取多少个对象，并对输入进行基本验证。然后检查是否满足胜利条件。
// 返回一个布尔值，指示游戏是否结束，以及新的堆大小。
func playerMove(pile, min, max int, win WinOption) (bool, int) {
    // 创建一个从标准输入读取数据的扫描器
    scanner := bufio.NewScanner(os.Stdin)
    done := false
    for !done {
        fmt.Println("YOUR MOVE")
        // 扫描输入
        scanner.Scan()
        // 将输入转换为整数
        m, err := strconv.Atoi(scanner.Text())
        if err != nil {
            continue
        }

        if m == 0 {
            fmt.Println("I TOLD YOU NOT TO USE ZERO!  COMPUTER WINS BY FORFEIT.")
            return true, pile
        }

        if m > max || m < min {
            fmt.Println("ILLEGAL MOVE, REENTER IT")
            continue
        }

        pile -= m
        done = true

        if pile <= 0 {
            if win == AvoidLast {
                fmt.Println("TOUGH LUCK, YOU LOSE.")
            } else {
                fmt.Println("CONGRATULATIONS, YOU WIN.")
            }
            return true, pile
        }
    }
    return false, pile
}

// 处理确定计算机在其回合上选择多少对象的逻辑。
func computerPick(pile, min, max int, win WinOption) int {
    var q int
    if win == AvoidLast {
        q = pile - 1
    } else {
        q = pile
    }
    c := min + max

    pick := q - (c * int(q/c))

    if pick < min {
        pick = min
    } else if pick > max {
        pick = max
    }

    return pick
}

// 处理计算机的回合 - 首先检查各种胜利/失败条件，然后计算计算机将取多少对象。
// 返回一个布尔值，指示游戏是否结束，以及新的堆大小。
func computerMove(pile, min, max int, win WinOption) (bool, int) {
    // 首先检查游戏结束的条件
    # 如果是取最后一个并且堆的数量小于等于最大允许取的数量
    if win == TakeLast && pile <= max:
        # 打印电脑取的数量并且赢了
        fmt.Printf("COMPUTER TAKES %d AND WINS\n", pile)
        # 返回真和取的数量
        return true, pile

    # 如果是避免最后一个并且堆的数量小于等于最小允许取的数量
    if win == AvoidLast && pile <= min:
        # 打印电脑取的数量并且输了
        fmt.Printf("COMPUTER TAKES %d AND LOSES\n", pile)
        # 返回真和取的数量
        return true, pile

    # 否则确定电脑的选择
    selection := computerPick(pile, min, max, win)
    # 减去电脑的选择
    pile -= selection
    # 打印电脑取的数量和剩余的数量
    fmt.Printf("COMPUTER TAKES %d AND LEAVES %d\n", selection, pile)
    # 返回假和剩余的数量
    return false, pile
// 这是主游戏循环 - 每个回合重复，直到满足胜利/失败条件之一。
func play(pile, min, max int, start StartOption, win WinOption) {
    // 游戏结束标志
    gameOver := false
    // 玩家回合标志
    playersTurn := (start == PlayerFirst)

    // 循环直到游戏结束
    for !gameOver {
        // 如果是玩家回合
        if playersTurn {
            // 玩家进行移动，更新游戏结束标志和堆的状态
            gameOver, pile = playerMove(pile, min, max, win)
            playersTurn = false
            // 如果游戏结束，返回
            if gameOver {
                return
            }
        }

        // 如果不是玩家回合
        if !playersTurn {
            // 计算机进行移动，更新游戏结束标志和堆的状态
            gameOver, pile = computerMove(pile, min, max, win)
            playersTurn = true
        }
    }
}

// 打印游戏介绍和规则
func printIntro() {
    fmt.Printf("%33s%s\n", " ", "BATNUM")
    fmt.Printf("%15s%s\n", " ", "CREATIVE COMPUTING  MORRISSTOWN, NEW JERSEY")
    fmt.Printf("\n\n\n")
    fmt.Println("THIS PROGRAM IS A 'BATTLE OF NUMBERS' GAME, WHERE THE")
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
    // 无限循环
    for {
        // 打印游戏介绍
        printIntro()

        // 创建游戏选项
        g := NewOptions()

        // 如果堆大小小于0，返回
        if g.pileSize < 0 {
            return
        }

        // 开始游戏
        play(g.pileSize, g.minSelect, g.maxSelect, g.startOption, g.winOption)
    }
}
```