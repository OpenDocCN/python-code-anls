# `d:/src/tocomm/basic-computer-games\00_Alternate_Languages\12_Bombs_Away\go\main.go`

```
package main  # 声明当前文件所属的包

import (  # 导入需要使用的包
	"bufio"  # 用于读取输入
	"fmt"  # 用于格式化输出
	"math/rand"  # 用于生成随机数
	"os"  # 提供对操作系统功能的访问
	"strconv"  # 用于字符串和数字之间的转换
	"strings"  # 提供对字符串的操作
	"time"  # 提供时间相关的功能
)

type Choice struct {  # 定义一个名为Choice的结构体
	idx string  # 结构体成员，表示选择的索引
	msg string  # 结构体成员，表示选择的消息
}

func playerSurvived() {  # 定义一个名为playerSurvived的函数
	fmt.Println("YOU MADE IT THROUGH TREMENDOUS FLAK!!")  # 在控制台输出消息
}
func playerDeath() {
	// 打印玩家死亡信息
	fmt.Println("* * * * BOOM * * * *")
	fmt.Println("YOU HAVE BEEN SHOT DOWN.....")
	fmt.Println("DEARLY BELOVED, WE ARE GATHERED HERE TODAY TO PAY OUR")
	fmt.Println("LAST TRIBUTE...")
}

func missionSuccess() {
	// 打印任务成功信息
	fmt.Printf("DIRECT HIT!!!! %d KILLED.\n", int(100*rand.Int()))
	fmt.Println("MISSION SUCCESSFUL.")
}

// Takes a float between 0 and 1 and returns a boolean
// if the player has survived (based on random chance)
// Returns True if death, False if survived
func deathWithChance(probability float64) bool {
	// 根据概率判断玩家是否存活
	return probability > rand.Float64()
}
func startNonKamikaziAttack() {  // 定义一个名为 startNonKamikaziAttack 的函数
	numMissions := getIntInput("HOW MANY MISSIONS HAVE YOU FLOWN? ")  // 获取用户输入的飞行任务数量

	for numMissions > 160 {  // 当飞行任务数量大于160时执行以下循环
		fmt.Println("MISSIONS, NOT MILES...")  // 打印消息
		fmt.Println("150 MISSIONS IS HIGH EVEN FOR OLD-TIMERS")  // 打印消息
		numMissions = getIntInput("HOW MANY MISSIONS HAVE YOU FLOWN? ")  // 获取用户输入的飞行任务数量
	}

	if numMissions > 100 {  // 如果飞行任务数量大于100
		fmt.Println("THAT'S PUSHING THE ODDS!")  // 打印消息
	}

	if numMissions < 25 {  // 如果飞行任务数量小于25
		fmt.Println("FRESH OUT OF TRAINING, EH?")  // 打印消息
	}

	fmt.Println()  // 打印空行

	if float32(numMissions) > (160 * rand.Float32()) {  // 如果飞行任务数量大于随机数乘以160
		missionSuccess()
```
这行代码调用了名为`missionSuccess`的函数，表示任务成功。

```go
	} else {
		missionFailure()
	}
```
这行代码表示如果条件不满足，则调用名为`missionFailure`的函数，表示任务失败。

```go
func missionFailure() {
```
这行代码定义了一个名为`missionFailure`的函数，用于处理任务失败的情况。

```go
	fmt.Printf("MISSED TARGET BY %d MILES!\n", int(2+30*rand.Float32()))
```
这行代码使用`Printf`函数打印出字符串，其中包含一个格式化的占位符，用于显示计算得到的数值。

```go
	fmt.Println("NOW YOU'RE REALLY IN FOR IT !!")
```
这行代码使用`Println`函数打印出字符串。

```go
	enemyWeapons := getInputFromList("DOES THE ENEMY HAVE GUNS(1), MISSILES(2), OR BOTH(3)? ", []Choice{{idx: "1", msg: "GUNS"}, {idx: "2", msg: "MISSILES"}, {idx: "3", msg: "BOTH"}})
```
这行代码调用了名为`getInputFromList`的函数，用于获取用户输入的敌人武器类型。

```go
	enemyGunnerAccuracy := 0.0
	if enemyWeapons.idx != "2" {
		enemyGunnerAccuracy = float64(getIntInput("WHAT'S THE PERCENT HIT RATE OF ENEMY GUNNERS (10 TO 50)? "))
		if enemyGunnerAccuracy < 10.0 {
```
这部分代码包含了条件语句和用户输入，根据敌人的武器类型和准确度进行不同的处理。
			fmt.Println("YOU LIE, BUT YOU'LL PAY...")  # 打印字符串"YOU LIE, BUT YOU'LL PAY..."
			playerDeath()  # 调用函数playerDeath()
		}
	}

	missileThreatWeighting := 35.0  # 初始化变量missileThreatWeighting为35.0
	if enemyWeapons.idx == "1" {  # 如果敌人的武器编号为"1"
		missileThreatWeighting = 0  # 将missileThreatWeighting的值设为0
	}

	death := deathWithChance((enemyGunnerAccuracy + missileThreatWeighting) / 100)  # 调用函数deathWithChance()，计算玩家死亡的概率并将结果赋给变量death

	if death {  # 如果death为真
		playerDeath()  # 调用函数playerDeath()
	} else {  # 否则
		playerSurvived()  # 调用函数playerSurvived()
	}
}

func playItaly() {  # 定义函数playItaly()
	targets := []Choice{{idx: "1", msg: "SHOULD BE EASY -- YOU'RE FLYING A NAZI-MADE PLANE."}, {idx: "2", msg: "BE CAREFUL!!!"}, {idx: "3", msg: "YOU'RE GOING FOR THE OIL, EH?"}}
	// 创建一个包含 Choice 结构的切片 targets，每个 Choice 结构包含 idx 和 msg 两个字段
	target := getInputFromList("YOUR TARGET -- ALBANIA(1), GREECE(2), NORTH AFRICA(3)", targets)
	// 从用户输入中选择目标，并将结果赋值给 target
	fmt.Println(target.msg)
	// 打印目标的消息
	startNonKamikaziAttack()
	// 开始非神风攻击
}

func playAllies() {
	aircraft := getInputFromList("AIRCRAFT -- LIBERATOR(1), B-29(2), B-17(3), LANCASTER(4): ", aircraftMessages)
	// 从用户输入中选择飞机，并将结果赋值给 aircraft
	fmt.Println(aircraft.msg)
	// 打印飞机的消息
	startNonKamikaziAttack()
	// 开始非神风攻击
}

func playJapan() {
	acknowledgeMessage := []Choice{{idx: "Y", msg: "Y"}, {idx: "N", msg: "N"}}
	// 创建一个包含 Choice 结构的切片 acknowledgeMessage，每个 Choice 结构包含 idx 和 msg 两个字段
	firstMission := getInputFromList("YOU'RE FLYING A KAMIKAZE MISSION OVER THE USS LEXINGTON.\nYOUR FIRST KAMIKAZE MISSION? (Y OR N): ", acknowledgeMessage)
	// 从用户输入中选择是否进行神风任务，并将结果赋值给 firstMission
	if firstMission.msg == "N" {
		playerDeath()
		// 如果用户选择不进行神风任务，则调用 playerDeath 函数
	}
	if rand.Float64() > 0.65 {
		missionSuccess()
		// 如果随机数大于 0.65，则调用 missionSuccess 函数
	} else {
		playerDeath()  // 调用 playerDeath 函数，处理玩家死亡的情况
	}
}

func playGermany() {
	target := getInputFromList("A NAZI, EH?  OH WELL.  ARE YOU GOING FOR RUSSIA(1),\nENGLAND(2), OR FRANCE(3)? ", targets)
	fmt.Println(target.msg)  // 打印目标信息
	startNonKamikaziAttack()  // 调用 startNonKamikaziAttack 函数，开始非神风攻击
}

func playGame() {
	fmt.Println("YOU ARE A PILOT IN A WORLD WAR II BOMBER.")  // 打印游戏开始信息
	switch side.idx {  // 根据 side.idx 的值进行不同的操作
	case "1":
		playItaly()  // 调用 playItaly 函数，开始意大利战役
	case "2":
		playAllies()  // 调用 playAllies 函数，开始盟军战役
	case "3":
		playJapan()  // 调用 playJapan 函数，开始日本战役
	case "4":  // 如果用户输入为 "4"
		playGermany()  // 调用 playGermany() 函数
	}
}

func main() {
	rand.Seed(time.Now().UnixNano())  // 使用当前时间的纳秒数作为随机数种子

	for {
		playGame()  // 调用 playGame() 函数
		if getInputFromList("ANOTHER MISSION (Y OR N):", []Choice{{idx: "Y", msg: "Y"}, {idx: "N", msg: "N"}}).msg == "N" {  // 获取用户输入并判断是否为 "N"
			break  // 如果用户输入为 "N"，跳出循环
		}
	}
}

func getInputFromList(prompt string, choices []Choice) Choice {
	scanner := bufio.NewScanner(os.Stdin)  // 创建一个从标准输入读取数据的 Scanner 对象
	for {
		fmt.Println(prompt)  // 打印提示信息
		scanner.Scan()  # 从标准输入中扫描下一行文本
		choice := scanner.Text()  # 将扫描到的文本赋值给变量choice
		for _, c := range choices:  # 遍历choices列表中的元素
			if strings.EqualFold(strings.ToUpper(choice), strings.ToUpper(c.idx)):  # 比较用户输入的choice和choices列表中的元素是否相等（不区分大小写）
				return c  # 如果相等，则返回该元素
		fmt.Println("TRY AGAIN...")  # 如果未找到匹配的元素，则打印提示信息
	}
}

func getIntInput(prompt string) int:  # 定义一个名为getIntInput的函数，接受一个字符串参数prompt，返回一个整数
	scanner := bufio.NewScanner(os.Stdin)  # 创建一个从标准输入中读取数据的Scanner对象
	for:  # 进入循环
		fmt.Println(prompt)  # 打印提示信息
		scanner.Scan()  # 从标准输入中扫描下一行文本
		choice, err := strconv.Atoi(scanner.Text())  # 将扫描到的文本转换为整数类型
		if err != nil:  # 如果转换出错
			fmt.Println("TRY AGAIN...")  # 打印提示信息
			continue  # 继续下一次循环
		} else {  # 如果条件不满足
			return choice  # 返回选择的结果
		}
	}
}
```