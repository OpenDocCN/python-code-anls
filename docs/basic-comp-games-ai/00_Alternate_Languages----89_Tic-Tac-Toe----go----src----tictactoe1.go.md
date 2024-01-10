# `basic-computer-games\00_Alternate_Languages\89_Tic-Tac-Toe\go\src\tictactoe1.go`

```
package main

import (
    "fmt"
    "strconv"
)

func main() {
    // 在屏幕上打印文本，文本前有30个空格
    fmt.Printf("%30s\n", "TIC TAC TOE")
    // 在屏幕上打印文本，文本前有15个空格
    // 并在屏幕上打印三行换行
    fmt.Printf("%15s\n\n\n\n", "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    // 这个程序玩井字棋
    fmt.Printf("THE GAME BOARD IS NUMBERED:\n\n")
    fmt.Println("1  2  3")
    fmt.Println("8  9  4")
    fmt.Println("7  6  5")

    // 主程序
    for {
        var (
            a, b, c, d, e int
            p, q, r, s    int
        )
        a = 9
        fmt.Printf("\n\n")
        // 机器先走
        computerMoves(a)
        p = readYourMove()
        b = move(p + 1)
        computerMoves(b)
        q = readYourMove()
        if q == move(b+4) {
            c = move(b + 2)
            computerMoves(c)
            r = readYourMove()
            if r == move(c+4) {
                if p%2 != 0 {
                    d = move(c + 3)
                    computerMoves(d)
                    s = readYourMove()
                    if s == move(d+4) {
                        e = move(d + 6)
                        computerMoves(e)
                        fmt.Println("THE GAME IS A DRAW.")
                    } else {
                        e = move(d + 4)
                        computerMoves(e)
                        fmt.Println("AND WINS ********")
                    }
                } else {
                    d = move(c + 7)
                    computerMoves(d)
                    fmt.Println("AND WINS ********")
                }
            } else {
                d = move(c + 4)
                computerMoves(d)
                fmt.Println("AND WINS ********")
            }
        } else {
            c = move(b + 4)
            computerMoves(c)
            fmt.Println("AND WINS ********")
        }
    }
}
func computerMoves(move int) {
    # 使用 fmt 包中的 Printf 函数打印 COMPUTER MOVES 和 move 变量的值
    fmt.Printf("COMPUTER MOVES %v\n", move)
# 读取用户输入的数字
func readYourMove() int:
    # 进入无限循环，直到用户输入有效的数字
    for:
        # 提示用户输入数字
        fmt.Printf("YOUR MOVE?")
        # 读取用户输入的字符串
        var input string
        fmt.Scan(&input)
        # 将用户输入的字符串转换为整数
        number, err := strconv.Atoi(input)
        # 如果转换成功，则返回该数字
        if err == nil:
            return number

# 计算移动后的位置
func move(number int) int:
    # 根据规则计算移动后的位置
    return number - 8*(int)((number-1)/8)
```