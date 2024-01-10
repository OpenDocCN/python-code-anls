# `basic-computer-games\00_Alternate_Languages\38_Fur_Trader\go\main.go`

```
package main

import (
    "bufio"  // 导入用于读取输入的包
    "fmt"    // 导入用于格式化输出的包
    "log"    // 导入用于记录日志的包
    "math/rand"  // 导入用于生成随机数的包
    "os"     // 导入操作系统功能的包
    "strconv"    // 导入字符串转换功能的包
    "strings"    // 导入处理字符串的包
    "time"   // 导入处理时间的包
)

const (
    MAXFURS    = 190  // 定义最大毛皮数量
    STARTFUNDS = 600  // 定义初始资金
)

type Fur int8  // 定义毛皮类型

const (
    FUR_MINK Fur = iota   // 定义不同种类的毛皮
    FUR_BEAVER
    FUR_ERMINE
    FUR_FOX
)

type Fort int8  // 定义据点类型

const (
    FORT_MONTREAL Fort = iota   // 定义不同的据点
    FORT_QUEBEC
    FORT_NEWYORK
)

type GameState int8  // 定义游戏状态类型

const (
    STARTING GameState = iota   // 定义游戏的不同状态
    TRADING
    CHOOSINGFORT
    TRAVELLING
)

func FURS() []string {
    return []string{"MINK", "BEAVER", "ERMINE", "FOX"}  // 返回毛皮类型的字符串切片
}

func FORTS() []string {
    return []string{"HOCHELAGA (MONTREAL)", "STADACONA (QUEBEC)", "NEW YORK"}  // 返回据点名称的字符串切片
}

type Player struct {
    funds float32  // 玩家资金
    furs  []int    // 玩家拥有的毛皮数量
}

func NewPlayer() Player {
    p := Player{}  // 创建新的玩家对象
    p.funds = STARTFUNDS  // 设置初始资金
    p.furs = make([]int, 4)  // 初始化毛皮数量
    return p
}

func (p *Player) totalFurs() int {
    f := 0  // 初始化总毛皮数量
    for _, v := range p.furs {  // 遍历玩家拥有的毛皮数量
        f += v  // 累加毛皮数量
    }
    return f  // 返回总毛皮数量
}

func (p *Player) lostFurs() {
    for f := 0; f < len(p.furs); f++ {  // 遍历玩家拥有的毛皮数量
        p.furs[f] = 0  // 将毛皮数量设置为0
    }
}

func printTitle() {
    fmt.Println("                               FUR TRADER")  // 打印游戏标题
    fmt.Println("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")  // 打印游戏信息
    fmt.Println()  // 打印空行
    fmt.Println()  // 打印空行
    fmt.Println()  // 打印空行
}

func printIntro() {
    fmt.Println("YOU ARE THE LEADER OF A FRENCH FUR TRADING EXPEDITION IN ")  // 打印游戏介绍
    fmt.Println("1776 LEAVING THE LAKE ONTARIO AREA TO SELL FURS AND GET")
    fmt.Println("SUPPLIES FOR THE NEXT YEAR.  YOU HAVE A CHOICE OF THREE")
    fmt.Println("FORTS AT WHICH YOU MAY TRADE.  THE COST OF SUPPLIES")
    fmt.Println("AND THE AMOUNT YOU RECEIVE FOR YOUR FURS WILL DEPEND")
    fmt.Println("ON THE FORT THAT YOU CHOOSE.")
    fmt.Println()
}

func getFortChoice() Fort {
    scanner := bufio.NewScanner(os.Stdin)  // 创建用于读取输入的扫描器
    // 无限循环，直到用户输入有效的选项
    for {
        // 打印提示信息
        fmt.Println()
        fmt.Println("YOU MAY TRADE YOUR FURS AT FORT 1, FORT 2,")
        fmt.Println("OR FORT 3.  FORT 1 IS FORT HOCHELAGA (MONTREAL)")
        fmt.Println("AND IS UNDER THE PROTECTION OF THE FRENCH ARMY.")
        fmt.Println("FORT 2 IS FORT STADACONA (QUEBEC) AND IS UNDER THE")
        fmt.Println("PROTECTION OF THE FRENCH ARMY.  HOWEVER, YOU MUST")
        fmt.Println("MAKE A PORTAGE AND CROSS THE LACHINE RAPIDS.")
        fmt.Println("FORT 3 IS FORT NEW YORK AND IS UNDER DUTCH CONTROL.")
        fmt.Println("YOU MUST CROSS THROUGH IROQUOIS LAND.")
        fmt.Println("ANSWER 1, 2, OR 3.")
        fmt.Print(">> ")
        // 读取用户输入
        scanner.Scan()

        // 将用户输入的字符串转换为整数
        f, err := strconv.Atoi(scanner.Text())
        // 如果转换出错或者输入不在1到3之间，提示用户重新输入
        if err != nil || f < 1 || f > 3 {
            fmt.Println("Invalid input, Try again ... ")
            continue
        }
        // 返回用户选择的短语
        return Fort(f)
    }
}
// 打印对应要塞的评论
func printFortComment(f Fort) {
    fmt.Println()
    switch f {
    case FORT_MONTREAL:
        fmt.Println("YOU HAVE CHOSEN THE EASIEST ROUTE.  HOWEVER, THE FORT")
        fmt.Println("IS FAR FROM ANY SEAPORT.  THE VALUE")
        fmt.Println("YOU RECEIVE FOR YOUR FURS WILL BE LOW AND THE COST")
        fmt.Println("OF SUPPLIES HIGHER THAN AT FORTS STADACONA OR NEW YORK.")
    case FORT_QUEBEC:
        fmt.Println("YOU HAVE CHOSEN A HARD ROUTE.  IT IS, IN COMPARSION,")
        fmt.Println("HARDER THAN THE ROUTE TO HOCHELAGA BUT EASIER THAN")
        fmt.Println("THE ROUTE TO NEW YORK.  YOU WILL RECEIVE AN AVERAGE VALUE")
        fmt.Println("FOR YOUR FURS AND THE COST OF YOUR SUPPLIES WILL BE AVERAGE.")
    case FORT_NEWYORK:
        fmt.Println("YOU HAVE CHOSEN THE MOST DIFFICULT ROUTE.  AT")
        fmt.Println("FORT NEW YORK YOU WILL RECEIVE THE HIGHEST VALUE")
        fmt.Println("FOR YOUR FURS.  THE COST OF YOUR SUPPLIES")
        fmt.Println("WILL BE LOWER THAN AT ALL THE OTHER FORTS.")
    }
    fmt.Println()
}

// 获取用户输入的是YES还是NO
func getYesOrNo() string {
    scanner := bufio.NewScanner(os.Stdin)
    for {
        fmt.Println("ANSWER YES OR NO")
        scanner.Scan()
        if strings.ToUpper(scanner.Text())[0:1] == "Y" {
            return "Y"
        } else if strings.ToUpper(scanner.Text())[0:1] == "N" {
            return "N"
        }
    }
}

// 获取用户购买的毛皮数量
func getFursPurchase() []int {
    scanner := bufio.NewScanner(os.Stdin)
    fmt.Printf("YOUR %d FURS ARE DISTRIBUTED AMONG THE FOLLOWING\n", MAXFURS)
    fmt.Println("KINDS OF PELTS: MINK, BEAVER, ERMINE AND FOX.")
    fmt.Println()

    purchases := make([]int, 4)

    for i, f := range FURS() {
    retry:
        fmt.Printf("HOW MANY %s DO YOU HAVE: ", f)
        scanner.Scan()
        count, err := strconv.Atoi(scanner.Text())
        if err != nil {
            fmt.Println("INVALID INPUT, TRY AGAIN ...")
            goto retry
        }
        purchases[i] = count
    }

    return purchases
}

func main() {
    # 使用当前时间的纳秒数作为随机数种子
    rand.Seed(time.Now().UnixNano())

    # 打印游戏标题
    printTitle()

    # 初始化游戏状态为 STARTING
    gameState := STARTING
    # 初始化要前往的堡垒为 FORT_NEWYORK
    whichFort := FORT_NEWYORK
    # 初始化四种动物皮的价格变量
    var (
        minkPrice   int
        erminePrice int
        beaverPrice int
        foxPrice    int
    )
    # 创建一个新的玩家对象
    player := NewPlayer()
# 闭合前面的函数定义
```