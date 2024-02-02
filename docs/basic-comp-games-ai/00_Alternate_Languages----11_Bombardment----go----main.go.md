# `basic-computer-games\00_Alternate_Languages\11_Bombardment\go\main.go`

```py
package main

import (
    "bufio"  // 导入 bufio 包，提供读写数据的功能
    "fmt"    // 导入 fmt 包，提供格式化输入输出的功能
    "math/rand"  // 导入 math/rand 包，提供生成伪随机数的功能
    "os"     // 导入 os 包，提供操作系统功能
    "strconv"    // 导入 strconv 包，提供字符串和基本数据类型之间的转换功能
    "strings"    // 导入 strings 包，提供对字符串的操作功能
    "time"   // 导入 time 包，提供时间相关的功能
)

// Messages correspond to outposts remaining (3, 2, 1, 0)
var PLAYER_PROGRESS_MESSAGES = []string{
    "YOU GOT ME, I'M GOING FAST. BUT I'LL GET YOU WHEN\nMY TRANSISTO&S RECUP%RA*E!",  // 玩家进度消息
    "THREE DOWN, ONE TO GO.\n\n",  // 玩家进度消息
    "TWO DOWN, TWO TO GO.\n\n",  // 玩家进度消息
    "ONE DOWN, THREE TO GO.\n\n",  // 玩家进度消息
}

var ENEMY_PROGRESS_MESSAGES = []string{
    "YOU'RE DEAD. YOUR LAST OUTPOST WAS AT %d. HA, HA, HA.\nBETTER LUCK NEXT TIME.",  // 敌人进度消息
    "YOU HAVE ONLY ONE OUTPOST LEFT.\n\n",  // 敌人进度消息
    "YOU HAVE ONLY TWO OUTPOSTS LEFT.\n\n",  // 敌人进度消息
    "YOU HAVE ONLY THREE OUTPOSTS LEFT.\n\n",  // 敌人进度消息
}

func displayField() {
    for r := 0; r < 5; r++ {
        initial := r*5 + 1  // 计算初始值
        for c := 0; c < 5; c++ {
            //x := strconv.Itoa(initial + c)  // 将初始值和列数转换为字符串
            fmt.Printf("\t%d", initial+c)  // 格式化输出初始值加上列数
        }
        fmt.Println()  // 换行
    }
    fmt.Print("\n\n\n\n\n\n\n\n\n")  // 输出多个换行
}

func printIntro() {
    fmt.Println("                                BOMBARDMENT")  // 输出游戏标题
    fmt.Println("                CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")  // 输出游戏信息
    fmt.Println()  // 输出空行
    fmt.Println()  // 输出空行
    fmt.Println("YOU ARE ON A BATTLEFIELD WITH 4 PLATOONS AND YOU")  // 输出游戏信息
    fmt.Println("HAVE 25 OUTPOSTS AVAILABLE WHERE THEY MAY BE PLACED.")  // 输出游戏信息
    fmt.Println("YOU CAN ONLY PLACE ONE PLATOON AT ANY ONE OUTPOST.")  // 输出游戏信息
    fmt.Println("THE COMPUTER DOES THE SAME WITH ITS FOUR PLATOONS.")  // 输出游戏信息
    fmt.Println()  // 输出空行
    fmt.Println("THE OBJECT OF THE GAME IS TO FIRE MISSLES AT THE")  // 输出游戏信息
    fmt.Println("OUTPOSTS OF THE COMPUTER.  IT WILL DO THE SAME TO YOU.")  // 输出游戏信息
    fmt.Println("THE ONE WHO DESTROYS ALL FOUR OF THE ENEMY'S PLATOONS")  // 输出游戏信息
}
// 打印字符串 "FIRST IS THE WINNER."
    fmt.Println("FIRST IS THE WINNER.")
// 打印空行
    fmt.Println()
// 打印字符串 "GOOD LUCK... AND TELL US WHERE YOU WANT THE BODIES SENT!"
    fmt.Println("GOOD LUCK... AND TELL US WHERE YOU WANT THE BODIES SENT!")
// 打印空行
    fmt.Println()
// 打印字符串 "TEAR OFF MATRIX AND USE IT TO CHECK OFF THE NUMBERS."
    fmt.Println("TEAR OFF MATRIX AND USE IT TO CHECK OFF THE NUMBERS.")
// 打印四个换行符
    fmt.Print("\n\n\n\n")
}

// 返回一个包含 1 到 25 的整数列表
func positionList() []int {
    positions := make([]int, 25)
    for i := 0; i < 25; i++ {
        positions[i] = i + 1
    }
    return positions
}

// 随机选择 4 个 1 到 25 范围内的位置
func generateEnemyPositions() []int {
    positions := positionList()
    rand.Shuffle(len(positions), func(i, j int) { positions[i], positions[j] = positions[j], positions[i] })
    return positions[:4]
}

// 检查位置是否在 1 到 25 范围内
func isValidPosition(p int) bool {
    return p >= 1 && p <= 25
}

// 提示玩家输入 4 个位置
func promptForPlayerPositions() []int {
    scanner := bufio.NewScanner(os.Stdin)
    var positions []int

    for {
        fmt.Println("\nWHAT ARE YOUR FOUR POSITIONS (1-25)?")
        scanner.Scan()
        rawPositions := strings.Split(scanner.Text(), " ")

        if len(rawPositions) != 4 {
            fmt.Println("PLEASE ENTER FOUR UNIQUE POSITIONS")
            goto there
        }

        for _, p := range rawPositions {
            pos, err := strconv.Atoi(p)
            if (err != nil) || !isValidPosition(pos) {
                fmt.Println("ALL POSITIONS MUST RANGE (1-25)")
                goto there
            }
            positions = append(positions, pos)
        }
        if len(positions) == 4 {
// 返回位置列表
func positionList() []int {
    positions := make([]int, 25)
    for i := 0; i < 25; i++ {
        positions[i] = i + 1
    }
    return positions
}

// 检查位置是否有效
func isValidPosition(pos int) bool {
    return pos >= 1 && pos <= 25
}

// 提示玩家输入目标位置
func promptPlayerForTarget() int {
    scanner := bufio.NewScanner(os.Stdin)
    for {
        fmt.Println("\nWHERE DO YOU WISH TO FIRE YOUR MISSILE?")
        scanner.Scan()
        target, err := strconv.Atoi(scanner.Text())
        if (err != nil) || !isValidPosition(target) {
            fmt.Println("POSITIONS MUST RANGE (1-25)")
            continue
        }
        return target
    }
}

// 生成攻击序列
func generateAttackSequence() []int {
    positions := positionList()
    rand.Shuffle(len(positions), func(i, j int) { positions[i], positions[j] = positions[j], positions[i] })
    return positions
}

// 执行攻击过程，如果需要继续则返回 true
func attack(target int, positions *[]int, hitMsg, missMsg string, progressMsg []string) bool {
    for i := 0; i < len(*positions); i++ {
        if target == (*positions)[i] {
            fmt.Print(hitMsg)
            // 移除被击中的目标
            (*positions)[i] = (*positions)[len((*positions))-1]
            (*positions)[len((*positions))-1] = 0
            (*positions) = (*positions)[:len((*positions))-1]
            if len((*positions)) != 0 {
                fmt.Print(progressMsg[len((*positions))])
            } else {
                fmt.Printf(progressMsg[len((*positions))], target)
            }
            return len((*positions)) > 0
        }
    }
}
    // 打印提示信息
    fmt.Print(missMsg)
    // 返回敌方位置列表的长度是否大于0
    return len((*positions)) > 0
}

func main() {
    // 以当前时间为种子初始化随机数生成器
    rand.Seed(time.Now().UnixNano())

    // 打印游戏介绍
    printIntro()
    // 显示游戏场地
    displayField()

    // 生成敌方位置列表
    enemyPositions := generateEnemyPositions()
    // 生成敌方攻击序列
    enemyAttacks := generateAttackSequence()
    // 初始化敌方攻击计数器
    enemyAttackCounter := 0

    // 提示玩家输入位置
    playerPositions := promptForPlayerPositions()

    // 游戏循环
    for {
        // 玩家攻击
        if !attack(promptPlayerForTarget(), &enemyPositions, "YOU GOT ONE OF MY OUTPOSTS!\n\n", "HA, HA YOU MISSED. MY TURN NOW:\n\n", PLAYER_PROGRESS_MESSAGES) {
            break
        }
        // 电脑攻击
        hitMsg := fmt.Sprintf("I GOT YOU. IT WON'T BE LONG NOW. POST %d WAS HIT.\n", enemyAttacks[enemyAttackCounter])
        missMsg := fmt.Sprintf("I MISSED YOU, YOU DIRTY RAT. I PICKED %d. YOUR TURN:\n\n", enemyAttacks[enemyAttackCounter])
        if !attack(enemyAttacks[enemyAttackCounter], &playerPositions, hitMsg, missMsg, ENEMY_PROGRESS_MESSAGES) {
            break
        }
        // 更新敌方攻击计数器
        enemyAttackCounter += 1
    }
}
```