# `basic-computer-games\00_Alternate_Languages\42_Gunner\go\main.go`

```py
package main

import (
    "bufio"  // 导入 bufio 包，用于读取输入
    "fmt"    // 导入 fmt 包，用于格式化输出
    "math"   // 导入 math 包，用于数学计算
    "math/rand"  // 导入 math/rand 包，用于生成随机数
    "os"     // 导入 os 包，用于操作系统功能
    "strconv"    // 导入 strconv 包，用于字符串转换
    "strings"    // 导入 strings 包，用于字符串处理
    "time"   // 导入 time 包，用于时间相关操作
)

func printIntro() {
    fmt.Println("                                 GUNNER")  // 输出标题
    fmt.Println("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")  // 输出创意计算的地点
    fmt.Print("\n\n\n")  // 输出空行
    fmt.Println("YOU ARE THE OFFICER-IN-CHARGE, GIVING ORDERS TO A GUN")  // 输出提示信息
    fmt.Println("CREW, TELLING THEM THE DEGREES OF ELEVATION YOU ESTIMATE")  // 输出提示信息
    fmt.Println("WILL PLACE A PROJECTILE ON TARGET.  A HIT WITHIN 100 YARDS")  // 输出提示信息
    fmt.Println("OF THE TARGET WILL DESTROY IT.")  // 输出提示信息
    fmt.Println()  // 输出空行
}

func getFloat() float64 {
    scanner := bufio.NewScanner(os.Stdin)  // 创建一个从标准输入读取数据的 Scanner 对象
    for {
        scanner.Scan()  // 读取输入
        fl, err := strconv.ParseFloat(scanner.Text(), 64)  // 将输入的字符串转换为浮点数

        if err != nil {  // 如果转换出错
            fmt.Println("Invalid input")  // 输出错误信息
            continue  // 继续循环
        }

        return fl  // 返回转换后的浮点数
    }
}

func play() {
    gunRange := int(40000*rand.Float64() + 20000)  // 生成一个随机的枪的射程
    fmt.Printf("\nMAXIMUM RANGE OF YOUR GUN IS %d YARDS\n", gunRange)  // 输出枪的射程信息

    killedEnemies := 0  // 初始化击毙敌人数量
    S1 := 0  // 初始化 S1
    // 这里缺少了一些代码，需要补充
}

func main() {
    rand.Seed(time.Now().UnixNano())  // 使用当前时间作为随机数种子
    scanner := bufio.NewScanner(os.Stdin)  // 创建一个从标准输入读取数据的 Scanner 对象

    printIntro()  // 调用打印介绍信息的函数

    for {
        play()  // 调用游戏函数

        fmt.Print("TRY AGAIN (Y OR N)? ")  // 提示用户是否再次游戏
        scanner.Scan()  // 读取输入

        if strings.ToUpper(scanner.Text())[0:1] != "Y" {  // 判断用户输入的是否是 "Y"
            fmt.Println("\nOK. RETURN TO BASE CAMP.")  // 输出提示信息
            break  // 结束循环
        }
    }
}
```