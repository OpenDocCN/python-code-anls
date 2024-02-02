# `basic-computer-games\00_Alternate_Languages\31_Depth_Charge\go\main.go`

```py
package main

import (
    "bufio" // 导入用于读取输入的包
    "fmt" // 导入用于格式化输出的包
    "math" // 导入数学函数包
    "math/rand" // 导入随机数生成包
    "os" // 导入操作系统功能包
    "strconv" // 导入字符串转换包
    "strings" // 导入字符串处理包
    "time" // 导入时间处理包
)

type Position []int // 定义一个名为 Position 的切片类型

func NewPosition() Position { // 定义一个返回 Position 类型的函数
    p := make([]int, 3) // 创建一个包含三个整数的切片
    return Position(p) // 返回切片类型的对象
}

func showWelcome() { // 定义一个显示欢迎信息的函数
    fmt.Print("\033[H\033[2J") // 清空终端屏幕
    fmt.Println("                DEPTH CHARGE") // 输出游戏标题
    fmt.Println("    Creative Computing  Morristown, New Jersey") // 输出游戏信息
    fmt.Println() // 输出空行
}

func getNumCharges() (int, int) { // 定义一个获取搜索区域维度的函数
    scanner := bufio.NewScanner(os.Stdin) // 创建一个用于读取输入的扫描器

    for { // 循环直到输入正确的维度
        fmt.Println("Dimensions of search area?") // 提示用户输入搜索区域的维度
        scanner.Scan() // 读取用户输入
        dim, err := strconv.Atoi(scanner.Text()) // 将用户输入的字符串转换为整数
        if err != nil { // 如果转换出错
            fmt.Println("Must enter an integer number. Please try again...") // 提示用户重新输入
            continue // 继续下一次循环
        }
        return dim, int(math.Log2(float64(dim))) + 1 // 返回输入的维度和计算出的次数
    }
}

func askForNewGame() { // 定义一个询问是否开始新游戏的函数
    scanner := bufio.NewScanner(os.Stdin) // 创建一个用于读取输入的扫描器

    fmt.Println("Another game (Y or N): ") // 提示用户输入是否开始新游戏
    scanner.Scan() // 读取用户输入
    if strings.ToUpper(scanner.Text()) == "Y" { // 如果用户输入是 "Y"
        main() // 调用主函数重新开始游戏
    }
    fmt.Println("OK. Hope you enjoyed yourself") // 输出结束游戏的提示信息
    os.Exit(1) // 退出程序
}

func showShotResult(shot, location Position) { // 定义一个显示射击结果的函数
    result := "Sonar reports shot was " // 初始化结果字符串

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

    if shot[1] != location[1] || shot[0] != location[0] { // 如果射击位置不在目标位置的横向
        result += " and " // 添加连接词到结果字符串
    }
    if shot[2] > location[2] { // 如果射击位置在目标位置的下方
        result += "too low." // 添加太低信息到结果字符串
    } else if shot[2] < location[2] { // 如果射击位置在目标位置的上方
        result += "too high." // 添加太高信息到结果字符串
    } else { // 如果射击位置在目标位置的深度合适
        result += "depth OK." // 添加深度合适信息到结果字符串
    }

    fmt.Println(result) // 输出结果字符串
}

func getShot() Position { // 定义一个获取射击位置的函数
    scanner := bufio.NewScanner(os.Stdin) // 创建一个用于读取输入的扫描器
    # 无限循环，用于获取用户输入的坐标
    for {
        # 创建新的位置对象用于存储用户输入的坐标
        shotPos := NewPosition()
        # 打印提示信息，要求用户输入坐标
        fmt.Println("Enter coordinates: ")
        # 读取用户输入
        scanner.Scan()
        # 将用户输入的字符串按空格分割
        rawGuess := strings.Split(scanner.Text(), " ")
        # 如果用户输入的坐标不是3个，则跳转到标签 there
        if len(rawGuess) != 3 {
            goto there
        }
        # 遍历用户输入的坐标
        for i := 0; i < 3; i++ {
            # 将字符串转换为整数
            val, err := strconv.Atoi(rawGuess[i])
            # 如果转换出错，则跳转到标签 there
            if err != nil {
                goto there
            }
            # 将转换后的整数存储到位置对象中
            shotPos[i] = val
        }
        # 返回用户输入的坐标
        return shotPos
    # 标签 there，用于处理用户输入错误的情况
    there:
        # 打印提示信息，要求用户重新输入坐标
        fmt.Println("Please enter coordinates separated by spaces")
        fmt.Println("Example: 3 2 1")
    }
# 获取随机位置，参数为搜索区域大小
func getRandomPosition(searchArea int) Position {
    # 创建一个新的位置对象
    pos := NewPosition()
    # 循环3次，生成随机位置坐标
    for i := 0; i < 3; i++ {
        pos[i] = rand.Intn(searchArea)
    }
    # 返回生成的随机位置
    return pos
}

# 进行游戏，参数为搜索区域大小和深度炸弹数量
func playGame(searchArea, numCharges int) {
    # 设置随机数种子
    rand.Seed(time.Now().UTC().UnixNano())
    # 打印游戏欢迎信息和说明
    fmt.Println("\nYou are the captain of the destroyer USS Computer.")
    fmt.Println("An enemy sub has been causing you trouble. Your")
    fmt.Printf("mission is to destroy it. You have %d shots.\n", numCharges)
    fmt.Println("Specify depth charge explosion point with a")
    fmt.Println("trio of numbers -- the first two are the")
    fmt.Println("surface coordinates; the third is the depth.")
    fmt.Println("\nGood luck!")
    fmt.Println()

    # 获取敌方潜艇的随机位置
    subPos := getRandomPosition(searchArea)

    # 循环进行游戏，直到深度炸弹用尽
    for c := 0; c < numCharges; c++ {
        fmt.Printf("\nTrial #%d\n", c+1)

        # 获取玩家输入的炸弹爆炸位置
        shot := getShot()

        # 判断玩家是否击中敌方潜艇
        if shot[0] == subPos[0] && shot[1] == subPos[1] && shot[2] == subPos[2] {
            fmt.Printf("\nB O O M ! ! You found it in %d tries!\n", c+1)
            askForNewGame()
        } else {
            showShotResult(shot, subPos)
        }
    }

    # 深度炸弹用尽，游戏结束
    fmt.Println("\nYou have been torpedoed! Abandon ship!")
    fmt.Printf("The submarine was at %d %d %d\n", subPos[0], subPos[1], subPos[2])
    askForNewGame()

}

# 主函数
func main() {
    # 显示欢迎信息
    showWelcome()

    # 获取搜索区域大小和深度炸弹数量
    searchArea, numCharges := getNumCharges()

    # 开始游戏
    playGame(searchArea, numCharges)
}
```