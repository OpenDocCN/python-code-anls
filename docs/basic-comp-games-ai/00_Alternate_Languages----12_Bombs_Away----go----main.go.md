# `basic-computer-games\00_Alternate_Languages\12_Bombs_Away\go\main.go`

```py
package main

import (
    "bufio"  // 导入 bufio 包，提供了读写数据的缓冲区
    "fmt"  // 导入 fmt 包，提供了格式化 I/O 函数
    "math/rand"  // 导入 math/rand 包，提供了生成伪随机数的函数
    "os"  // 导入 os 包，提供了操作系统函数
    "strconv"  // 导入 strconv 包，提供了字符串和基本数据类型之间的转换函数
    "strings"  // 导入 strings 包，提供了操作字符串的函数
    "time"  // 导入 time 包，提供了时间的函数
)

type Choice struct {
    idx string  // 定义 Choice 结构体的 idx 字段，表示选项的索引
    msg string  // 定义 Choice 结构体的 msg 字段，表示选项的消息
}

func playerSurvived() {
    fmt.Println("YOU MADE IT THROUGH TREMENDOUS FLAK!!")  // 打印玩家幸存的消息
}

func playerDeath() {
    fmt.Println("* * * * BOOM * * * *")  // 打印玩家死亡的消息
    fmt.Println("YOU HAVE BEEN SHOT DOWN.....")  // 打印玩家被击落的消息
    fmt.Println("DEARLY BELOVED, WE ARE GATHERED HERE TODAY TO PAY OUR")  // 打印玩家死亡的悼词
    fmt.Println("LAST TRIBUTE...")  // 打印玩家死亡的悼词
}

func missionSuccess() {
    fmt.Printf("DIRECT HIT!!!! %d KILLED.\n", int(100*rand.Int()))  // 打印任务成功的消息和击杀数量
    fmt.Println("MISSION SUCCESSFUL.")  // 打印任务成功的消息
}

// Takes a float between 0 and 1 and returns a boolean
// if the player has survived (based on random chance)
// Returns True if death, False if survived
// 接受一个介于 0 和 1 之间的浮点数，并返回一个布尔值，表示玩家是否幸存（基于随机概率）
// 如果死亡则返回 True，幸存则返回 False
func deathWithChance(probability float64) bool {
    return probability > rand.Float64()  // 根据概率和随机数判断玩家是否幸存
}

func startNonKamikaziAttack() {
    numMissions := getIntInput("HOW MANY MISSIONS HAVE YOU FLOWN? ")  // 获取玩家输入的飞行任务数量

    for numMissions > 160 {
        fmt.Println("MISSIONS, NOT MILES...")  // 打印提示消息
        fmt.Println("150 MISSIONS IS HIGH EVEN FOR OLD-TIMERS")  // 打印提示消息
        numMissions = getIntInput("HOW MANY MISSIONS HAVE YOU FLOWN? ")  // 获取玩家输入的飞行任务数量
    }

    if numMissions > 100 {
        fmt.Println("THAT'S PUSHING THE ODDS!")  // 打印提示消息
    }

    if numMissions < 25 {
        fmt.Println("FRESH OUT OF TRAINING, EH?")  // 打印提示消息
    }

    fmt.Println()

    if float32(numMissions) > (160 * rand.Float32()) {
        missionSuccess()  // 调用任务成功的函数
    } else {
        missionFailure()  // 调用任务失败的函数
    }
}

func missionFailure() {
    fmt.Printf("MISSED TARGET BY %d MILES!\n", int(2+30*rand.Float32()))  // 打印未命中目标的消息和距离
    fmt.Println("NOW YOU'RE REALLY IN FOR IT !!")  // 打印提示消息
    fmt.Println()

    enemyWeapons := getInputFromList("DOES THE ENEMY HAVE GUNS(1), MISSILES(2), OR BOTH(3)? ", []Choice{{idx: "1", msg: "GUNS"}, {idx: "2", msg: "MISSILES"}, {idx: "3", msg: "BOTH"}})  // 获取玩家输入的敌人武器类型

    // If there are no gunners (i.e. weapon choice 2) then
    // we say that the gunners have 0 accuracy for the purposes
    // 如果没有炮手（即武器选择 2），则我们说炮手在这种情况下的准确度为 0
    // 计算玩家死亡的概率
    enemyGunnerAccuracy := 0.0
    // 如果敌人的武器索引不是 "2"，则获取敌人枪手的命中率
    if enemyWeapons.idx != "2" {
        enemyGunnerAccuracy = float64(getIntInput("WHAT'S THE PERCENT HIT RATE OF ENEMY GUNNERS (10 TO 50)? "))
        // 如果敌人枪手的命中率小于 10%，则输出信息并使玩家死亡
        if enemyGunnerAccuracy < 10.0 {
            fmt.Println("YOU LIE, BUT YOU'LL PAY...")
            playerDeath()
        }
    }

    // 设置导弹威胁权重为 35.0
    missileThreatWeighting := 35.0
    // 如果敌人的武器索引是 "1"，则将导弹威胁权重设置为 0
    if enemyWeapons.idx == "1" {
        missileThreatWeighting = 0
    }

    // 根据敌人枪手的命中率和导弹威胁权重计算死亡概率
    death := deathWithChance((enemyGunnerAccuracy + missileThreatWeighting) / 100)

    // 如果死亡概率为真，则使玩家死亡，否则玩家存活
    if death {
        playerDeath()
    } else {
        playerSurvived()
    }
# 定义玩意大利的函数
func playItaly() {
    # 创建目标选择列表
    targets := []Choice{{idx: "1", msg: "SHOULD BE EASY -- YOU'RE FLYING A NAZI-MADE PLANE."}, {idx: "2", msg: "BE CAREFUL!!!"}, {idx: "3", msg: "YOU'RE GOING FOR THE OIL, EH?"}}
    # 从目标选择列表中获取用户输入的目标
    target := getInputFromList("YOUR TARGET -- ALBANIA(1), GREECE(2), NORTH AFRICA(3)", targets)
    # 打印目标信息
    fmt.Println(target.msg)
    # 开始非神风式攻击
    startNonKamikaziAttack()
}

# 定义玩盟军的函数
func playAllies() {
    # 创建飞机信息选择列表
    aircraftMessages := []Choice{{idx: "1", msg: "YOU'VE GOT 2 TONS OF BOMBS FLYING FOR PLOESTI."}, {idx: "2", msg: "YOU'RE DUMPING THE A-BOMB ON HIROSHIMA."}, {idx: "3", msg: "YOU'RE CHASING THE BISMARK IN THE NORTH SEA."}, {idx: "4", msg: "YOU'RE BUSTING A GERMAN HEAVY WATER PLANT IN THE RUHR."}}
    # 从飞机信息选择列表中获取用户输入的飞机信息
    aircraft := getInputFromList("AIRCRAFT -- LIBERATOR(1), B-29(2), B-17(3), LANCASTER(4): ", aircraftMessages)
    # 打印飞机信息
    fmt.Println(aircraft.msg)
    # 开始非神风式攻击
    startNonKamikaziAttack()
}

# 定义玩日本的函数
func playJapan() {
    # 创建确认信息选择列表
    acknowledgeMessage := []Choice{{idx: "Y", msg: "Y"}, {idx: "N", msg: "N"}}
    # 获取用户输入的第一次任务确认信息
    firstMission := getInputFromList("YOU'RE FLYING A KAMIKAZE MISSION OVER THE USS LEXINGTON.\nYOUR FIRST KAMIKAZE MISSION? (Y OR N): ", acknowledgeMessage)
    # 如果用户选择不是第一次任务，则结束游戏
    if firstMission.msg == "N" {
        playerDeath()
    }
    # 如果随机数大于0.65，则任务成功，否则玩家死亡
    if rand.Float64() > 0.65 {
        missionSuccess()
    } else {
        playerDeath()
    }
}

# 定义玩德国的函数
func playGermany() {
    # 创建目标选择列表
    targets := []Choice{{idx: "1", msg: "YOU'RE NEARING STALINGRAD."}, {idx: "2", msg: "NEARING LONDON.  BE CAREFUL, THEY'VE GOT RADAR."}, {idx: "3", msg: "NEARING VERSAILLES.  DUCK SOUP.  THEY'RE NEARLY DEFENSELESS."}}
    # 从目标选择列表中获取用户输入的目标
    target := getInputFromList("A NAZI, EH?  OH WELL.  ARE YOU GOING FOR RUSSIA(1),\nENGLAND(2), OR FRANCE(3)? ", targets)
    # 打印目标信息
    fmt.Println(target.msg)
    # 开始非神风式攻击
    startNonKamikaziAttack()
}

# 定义玩游戏的函数
func playGame() {
    # 打印游戏背景信息
    fmt.Println("YOU ARE A PILOT IN A WORLD WAR II BOMBER.")
    # 获取用户选择的阵营
    side := getInputFromList("WHAT SIDE -- ITALY(1), ALLIES(2), JAPAN(3), GERMANY(4): ", []Choice{{idx: "1", msg: "ITALY"}, {idx: "2", msg: "ALLIES"}, {idx: "3", msg: "JAPAN"}, {idx: "4", msg: "GERMANY"}})
    # 根据 side.idx 的值进行不同的操作
    switch side.idx {
        # 如果 side.idx 为 "1"，则执行 playItaly() 函数
        case "1":
            playItaly()
        # 如果 side.idx 为 "2"，则执行 playAllies() 函数
        case "2":
            playAllies()
        # 如果 side.idx 为 "3"，则执行 playJapan() 函数
        case "3":
            playJapan()
        # 如果 side.idx 为 "4"，则执行 playGermany() 函数
        case "4":
            playGermany()
    }
# 主函数，程序入口
func main() {
    # 以当前时间为种子，初始化随机数生成器
    rand.Seed(time.Now().UnixNano())

    # 无限循环，直到用户选择退出
    for {
        # 调用 playGame() 函数开始游戏
        playGame()
        # 获取用户输入，如果选择退出则跳出循环
        if getInputFromList("ANOTHER MISSION (Y OR N):", []Choice{{idx: "Y", msg: "Y"}, {idx: "N", msg: "N"}}).msg == "N" {
            break
        }
    }
}

# 从给定选项中获取用户输入
func getInputFromList(prompt string, choices []Choice) Choice {
    # 创建标准输入的扫描器
    scanner := bufio.NewScanner(os.Stdin)
    # 循环直到用户输入有效选项
    for {
        # 打印提示信息
        fmt.Println(prompt)
        # 扫描用户输入
        scanner.Scan()
        choice := scanner.Text()
        # 遍历选项，忽略大小写比较用户输入和选项
        for _, c := range choices {
            if strings.EqualFold(strings.ToUpper(choice), strings.ToUpper(c.idx)) {
                return c
            }
        }
        # 用户输入无效，提示重新输入
        fmt.Println("TRY AGAIN...")
    }
}

# 获取用户输入的整数
func getIntInput(prompt string) int {
    # 创建标准输入的扫描器
    scanner := bufio.NewScanner(os.Stdin)
    # 循环直到用户输入有效整数
    for {
        # 打印提示信息
        fmt.Println(prompt)
        # 扫描用户输入
        scanner.Scan()
        choice, err := strconv.Atoi(scanner.Text())
        # 如果转换失败，提示重新输入
        if err != nil {
            fmt.Println("TRY AGAIN...")
            continue
        } else {
            return choice
        }
    }
}
```