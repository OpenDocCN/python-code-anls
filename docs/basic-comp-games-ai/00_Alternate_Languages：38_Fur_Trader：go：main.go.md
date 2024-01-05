# `d:/src/tocomm/basic-computer-games\00_Alternate_Languages\38_Fur_Trader\go\main.go`

```
package main  // 声明当前文件所属的包为 main

import (
	"bufio"  // 导入 bufio 包，用于提供带缓冲的 I/O 操作
	"fmt"  // 导入 fmt 包，用于格式化输入输出
	"log"  // 导入 log 包，用于记录日志
	"math/rand"  // 导入 math/rand 包，用于生成随机数
	"os"  // 导入 os 包，提供了对操作系统功能的访问
	"strconv"  // 导入 strconv 包，用于字符串和基本数据类型之间的转换
	"strings"  // 导入 strings 包，用于处理字符串
	"time"  // 导入 time 包，用于处理时间
)

const (
	MAXFURS    = 190  // 声明一个常量 MAXFURS，表示最大的毛皮数量
	STARTFUNDS = 600  // 声明一个常量 STARTFUNDS，表示初始资金
)

type Fur int8  // 声明一个自定义类型 Fur，表示毛皮的数量
# 定义枚举类型 Fur，包括 FUR_MINK、FUR_BEAVER、FUR_ERMINE、FUR_FOX
const (
	FUR_MINK Fur = iota
	FUR_BEAVER
	FUR_ERMINE
	FUR_FOX
)

# 定义枚举类型 Fort，包括 FORT_MONTREAL、FORT_QUEBEC、FORT_NEWYORK
type Fort int8

const (
	FORT_MONTREAL Fort = iota
	FORT_QUEBEC
	FORT_NEWYORK
)

# 定义枚举类型 GameState，包括 STARTING、TRADING
type GameState int8

const (
	STARTING GameState = iota
	TRADING
)
	CHOOSINGFORT
	// 定义常量 CHOOSINGFORT

	TRAVELLING
	// 定义常量 TRAVELLING

)

func FURS() []string {
	// 定义函数 FURS，返回一个字符串数组
	return []string{"MINK", "BEAVER", "ERMINE", "FOX"}
	// 返回包含四种动物皮毛的字符串数组
}

func FORTS() []string {
	// 定义函数 FORTS，返回一个字符串数组
	return []string{"HOCHELAGA (MONTREAL)", "STADACONA (QUEBEC)", "NEW YORK"}
	// 返回包含三个城堡名称的字符串数组
}

type Player struct {
	// 定义玩家结构体
	funds float32
	// 玩家资金
	furs  []int
	// 玩家拥有的动物皮毛数量
}

func NewPlayer() Player {
	// 定义函数 NewPlayer，返回一个 Player 结构体
	p := Player{}
	// 创建一个新的玩家对象
	p.funds = STARTFUNDS
	// 设置玩家初始资金为 STARTFUNDS
# 创建一个名为Player的结构体类型
type Player struct {
	furs []int  # 创建一个整数切片类型的属性furs
}

# 创建一个名为NewPlayer的函数，返回一个Player类型的指针
func NewPlayer() *Player {
	p := new(Player)  # 创建一个Player类型的指针p
	p.furs = make([]int, 4)  # 使用make函数创建一个长度为4的整数切片，并赋值给p的furs属性
	return p  # 返回指针p
}

# 创建一个名为totalFurs的方法，计算玩家拥有的毛皮总数
func (p *Player) totalFurs() int {
	f := 0  # 初始化变量f为0
	for _, v := range p.furs:  # 遍历玩家拥有的毛皮切片
		f += v  # 将每个毛皮数量累加到f上
	}
	return f  # 返回总毛皮数量f
}

# 创建一个名为lostFurs的方法，将玩家失去的毛皮数量置零
func (p *Player) lostFurs() {
	for f := 0; f < len(p.furs); f++:  # 遍历玩家拥有的毛皮切片
		p.furs[f] = 0  # 将每个毛皮数量置零
	}
}

# 创建一个名为printTitle的函数，打印游戏标题
func printTitle() {
	fmt.Println("                               FUR TRADER")  # 打印游戏标题"FUR TRADER"
}
	fmt.Println("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
	# 打印创意计算公司的信息

	fmt.Println()
	# 打印空行
	fmt.Println()
	# 打印空行
	fmt.Println()
	# 打印空行
}

func printIntro() {
	fmt.Println("YOU ARE THE LEADER OF A FRENCH FUR TRADING EXPEDITION IN ")
	# 打印游戏介绍信息
	fmt.Println("1776 LEAVING THE LAKE ONTARIO AREA TO SELL FURS AND GET")
	# 打印游戏介绍信息
	fmt.Println("SUPPLIES FOR THE NEXT YEAR.  YOU HAVE A CHOICE OF THREE")
	# 打印游戏介绍信息
	fmt.Println("FORTS AT WHICH YOU MAY TRADE.  THE COST OF SUPPLIES")
	# 打印游戏介绍信息
	fmt.Println("AND THE AMOUNT YOU RECEIVE FOR YOUR FURS WILL DEPEND")
	# 打印游戏介绍信息
	fmt.Println("ON THE FORT THAT YOU CHOOSE.")
	# 打印游戏介绍信息
	fmt.Println()
	# 打印空行
}

func getFortChoice() Fort {
	scanner := bufio.NewScanner(os.Stdin)
	# 创建一个用于读取用户输入的扫描器

	for {
		# 进入循环，等待用户输入
		fmt.Println()
		// 打印交易毛皮的提示信息
		fmt.Println("YOU MAY TRADE YOUR FURS AT FORT 1, FORT 2,")
		fmt.Println("OR FORT 3.  FORT 1 IS FORT HOCHELAGA (MONTREAL)")
		fmt.Println("AND IS UNDER THE PROTECTION OF THE FRENCH ARMY.")
		fmt.Println("FORT 2 IS FORT STADACONA (QUEBEC) AND IS UNDER THE")
		fmt.Println("PROTECTION OF THE FRENCH ARMY.  HOWEVER, YOU MUST")
		fmt.Println("MAKE A PORTAGE AND CROSS THE LACHINE RAPIDS.")
		fmt.Println("FORT 3 IS FORT NEW YORK AND IS UNDER DUTCH CONTROL.")
		fmt.Println("YOU MUST CROSS THROUGH IROQUOIS LAND.")
		fmt.Println("ANSWER 1, 2, OR 3.")
		// 打印选择提示
		fmt.Print(">> ")
		// 读取用户输入
		scanner.Scan()

		// 将用户输入的字符串转换为整数
		f, err := strconv.Atoi(scanner.Text())
		// 检查转换过程中是否出现错误，或者用户输入的数字不在1到3之间
		if err != nil || f < 1 || f > 3 {
			// 如果出现错误或者输入不合法，打印错误提示信息并继续循环
			fmt.Println("Invalid input, Try again ... ")
			continue
		}
		// 返回用户选择的短语
		return Fort(f)
	}
}

// printFortComment 打印不同要塞选择的注释
func printFortComment(f Fort) {
	fmt.Println()  // 打印空行
	switch f {  // 开始一个 switch 语句，根据不同的 f 值执行不同的代码块
	case FORT_MONTREAL:  // 如果 f 的值是 FORT_MONTREAL
		fmt.Println("YOU HAVE CHOSEN THE EASIEST ROUTE.  HOWEVER, THE FORT")  // 打印相应的注释
		fmt.Println("IS FAR FROM ANY SEAPORT.  THE VALUE")
		fmt.Println("YOU RECEIVE FOR YOUR FURS WILL BE LOW AND THE COST")
		fmt.Println("OF SUPPLIES HIGHER THAN AT FORTS STADACONA OR NEW YORK.")
	case FORT_QUEBEC:  // 如果 f 的值是 FORT_QUEBEC
		fmt.Println("YOU HAVE CHOSEN A HARD ROUTE.  IT IS, IN COMPARSION,")  // 打印相应的注释
		fmt.Println("HARDER THAN THE ROUTE TO HOCHELAGA BUT EASIER THAN")
		fmt.Println("THE ROUTE TO NEW YORK.  YOU WILL RECEIVE AN AVERAGE VALUE")
		fmt.Println("FOR YOUR FURS AND THE COST OF YOUR SUPPLIES WILL BE AVERAGE.")
	case FORT_NEWYORK:  // 如果 f 的值是 FORT_NEWYORK
		fmt.Println("YOU HAVE CHOSEN THE MOST DIFFICULT ROUTE.  AT")  // 打印相应的注释
		fmt.Println("FORT NEW YORK YOU WILL RECEIVE THE HIGHEST VALUE")
		fmt.Println("FOR YOUR FURS.  THE COST OF YOUR SUPPLIES")
		fmt.Println("WILL BE LOWER THAN AT ALL THE OTHER FORTS.")
	}
	fmt.Println()  # 打印空行

}

func getYesOrNo() string {  # 定义函数getYesOrNo，返回类型为string
	scanner := bufio.NewScanner(os.Stdin)  # 创建一个从标准输入读取数据的Scanner
	for {  # 进入无限循环
		fmt.Println("ANSWER YES OR NO")  # 打印提示信息
		scanner.Scan()  # 读取输入
		if strings.ToUpper(scanner.Text())[0:1] == "Y" {  # 判断输入的第一个字符是否为Y
			return "Y"  # 如果是，则返回Y
		} else if strings.ToUpper(scanner.Text())[0:1] == "N" {  # 判断输入的第一个字符是否为N
			return "N"  # 如果是，则返回N
		}
	}
}

func getFursPurchase() []int {  # 定义函数getFursPurchase，返回类型为int的切片
	scanner := bufio.NewScanner(os.Stdin)  # 创建一个从标准输入读取数据的Scanner
	fmt.Printf("YOUR %d FURS ARE DISTRIBUTED AMONG THE FOLLOWING\n", MAXFURS)  # 格式化打印信息
	// 打印不同种类的毛皮
	fmt.Println("KINDS OF PELTS: MINK, BEAVER, ERMINE AND FOX.")
	fmt.Println()

	// 创建一个包含4个元素的整数切片
	purchases := make([]int, 4)

	// 遍历毛皮种类的切片
	for i, f := range FURS() {
	retry:
		// 提示用户输入对应毛皮的数量
		fmt.Printf("HOW MANY %s DO YOU HAVE: ", f)
		// 读取用户输入
		scanner.Scan()
		// 将用户输入的数量转换为整数
		count, err := strconv.Atoi(scanner.Text())
		// 如果转换出错，提示用户重新输入
		if err != nil {
			fmt.Println("INVALID INPUT, TRY AGAIN ...")
			// 跳转到标签retry处，重新输入
			goto retry
		}
		// 将用户输入的数量存入切片中
		purchases[i] = count
	}

	// 返回用户输入的毛皮数量切片
	return purchases
}
func main() {
    // 设置随机数种子
    rand.Seed(time.Now().UnixNano())

    // 打印游戏标题
    printTitle()

    // 初始化游戏状态为 STARTING
    gameState := STARTING
    // 初始化要前往的堡垒为 FORT_NEWYORK
    whichFort := FORT_NEWYORK
    // 初始化各种皮毛的价格变量
    var (
        minkPrice   int
        erminePrice int
        beaverPrice int
        foxPrice    int
    )
    // 创建玩家对象
    player := NewPlayer()

    // 游戏循环
    for {
        // 根据游戏状态进行不同的操作
        switch gameState {
        // 当游戏状态为 STARTING 时
        case STARTING:
            // 打印游戏介绍
            printIntro()
            // 打印是否愿意交易皮毛的提示
            fmt.Println("DO YOU WISH TO TRADE FURS?")
			if getYesOrNo() == "N" {  # 如果玩家输入的是N，退出游戏
				os.Exit(0)  # 退出程序
			}
			gameState = TRADING  # 设置游戏状态为交易状态
		case TRADING:  # 当游戏状态为交易状态时
			fmt.Println()  # 输出空行
			fmt.Printf("YOU HAVE $ %1.2f IN SAVINGS\n", player.funds)  # 输出玩家的存款金额
			fmt.Printf("AND %d FURS TO BEGIN THE EXPEDITION\n", MAXFURS)  # 输出玩家开始探险时的毛皮数量
			player.furs = getFursPurchase()  # 获取玩家购买的毛皮数量

			if player.totalFurs() > MAXFURS:  # 如果玩家总毛皮数量超过了最大值
				fmt.Println()  # 输出空行
				fmt.Println("YOU MAY NOT HAVE THAT MANY FURS.")  # 输出提示信息
				fmt.Println("DO NOT TRY TO CHEAT.  I CAN ADD.")  # 输出提示信息
				fmt.Println("YOU MUST START AGAIN.")  # 输出提示信息
				gameState = STARTING  # 设置游戏状态为开始状态
			} else:  # 否则
				gameState = CHOOSINGFORT  # 设置游戏状态为选择要前往的堡垒状态
		case CHOOSINGFORT:  # 当游戏状态为选择要前往的堡垒状态时
			whichFort = getFortChoice()  // 获取玩家选择的要前往的矿场
			printFortComment(whichFort)  // 打印关于选择的矿场的评论
			fmt.Println("DO YOU WANT TO TRADE AT ANOTHER FORT?")  // 打印询问玩家是否想在另一个矿场交易
			changeFort := getYesOrNo()  // 获取玩家是否想要更换矿场的选择
			if changeFort == "N" {  // 如果玩家选择不更换矿场
				gameState = TRAVELLING  // 则游戏状态变为旅行中
			}
		case TRAVELLING:  // 当游戏状态为旅行中时
			switch whichFort {  // 根据玩家选择的矿场进行不同的处理
			case FORT_MONTREAL:  // 如果选择的是蒙特利尔矿场
				// 根据一定的算法计算不同物品的价格
				minkPrice = (int((0.2*rand.Float64()+0.70)*100+0.5) / 100)
				erminePrice = (int((0.2*rand.Float64()+0.65)*100+0.5) / 100)
				beaverPrice = (int((0.2*rand.Float64()+0.75)*100+0.5) / 100)
				foxPrice = (int((0.2*rand.Float64()+0.80)*100+0.5) / 100)

				fmt.Println("SUPPLIES AT FORT HOCHELAGA COST $150.00.")  // 打印在奥什拉加矿场的物品价格
				fmt.Println("YOUR TRAVEL EXPENSES TO HOCHELAGA WERE $10.00.")  // 打印前往奥什拉加矿场的旅行费用
				player.funds -= 160  // 玩家的资金减去旅行费用和物品价格
			case FORT_QUEBEC:  // 如果选择的是魁北克矿场
				// 根据一定的算法计算不同物品的价格
				minkPrice = (int((0.30*rand.Float64()+0.85)*100+0.5) / 100)
# 生成erminePrice，根据随机浮点数生成价格并四舍五入到两位小数
erminePrice = (int((0.15*rand.Float64()+0.80)*100+0.5) / 100)
# 生成beaverPrice，根据随机浮点数生成价格并四舍五入到两位小数
beaverPrice = (int((0.20*rand.Float64()+0.90)*100+0.5) / 100)
# 生成foxPrice，根据随机浮点数生成价格并四舍五入到两位小数
foxPrice = (int((0.25*rand.Float64()+1.10)*100+0.5) / 100)

# 生成event，根据随机浮点数生成事件编号
event := int(10*rand.Float64()) + 1
# 根据事件编号判断事件类型并输出相应信息，更新玩家的皮毛数量
if event <= 2:
    print("YOUR BEAVER WERE TOO HEAVY TO CARRY ACROSS")
    print("THE PORTAGE. YOU HAD TO LEAVE THE PELTS, BUT FOUND")
    print("THEM STOLEN WHEN YOU RETURNED.")
    player.furs[FUR_BEAVER] = 0
elif event <= 6:
    print("YOU ARRIVED SAFELY AT FORT STADACONA.")
elif event <= 8:
    print("YOUR CANOE UPSET IN THE LACHINE RAPIDS.  YOU")
    print("LOST ALL YOUR FURS.")
    player.lostFurs()
elif event <= 10:
    print("YOUR FOX PELTS WERE NOT CURED PROPERLY.")
    print("NO ONE WILL BUY THEM.")
    player.furs[FUR_FOX] = 0
				} else {
					log.Fatal("Unexpected error")  // 如果出现意外错误，记录错误信息并终止程序运行
				}

				fmt.Println()  // 打印空行
				fmt.Println("SUPPLIES AT FORT STADACONA COST $125.00.")  // 打印指定信息
				fmt.Println("YOUR TRAVEL EXPENSES TO STADACONA WERE $15.00.")  // 打印指定信息
				player.funds -= 140  // 玩家的资金减去指定数额
			case FORT_NEWYORK:  // 如果情况为 FORT_NEWYORK
				minkPrice = (int((0.15*rand.Float64()+1.05)*100+0.5) / 100)  // 计算价格
				erminePrice = (int((0.15*rand.Float64()+0.95)*100+0.5) / 100)  // 计算价格
				beaverPrice = (int((0.25*rand.Float64()+1.00)*100+0.5) / 100)  // 计算价格
				foxPrice = (int((0.25*rand.Float64()+1.05)*100+0.5) / 100)  // 计算价格（原始代码中没有）

				event := int(10*rand.Float64()) + 1  // 生成随机事件
				if event <= 2 {  // 如果事件小于等于2
					fmt.Println("YOU WERE ATTACKED BY A PARTY OF IROQUOIS.")  // 打印指定信息
					fmt.Println("ALL PEOPLE IN YOUR TRADING GROUP WERE")  // 打印指定信息
					fmt.Println("KILLED.  THIS ENDS THE GAME.")  // 打印指定信息
					os.Exit(0)  // 终止程序运行
				} else if event <= 6 {  # 如果事件值小于等于6
					fmt.Println("YOU WERE LUCKY.  YOU ARRIVED SAFELY")  # 打印消息：你很幸运。你安全抵达了纽约堡。
					fmt.Println("AT FORT NEW YORK.")  # 打印消息：在纽约堡。
				} else if event <= 8 {  # 如果事件值小于等于8
					fmt.Println("YOU NARROWLY ESCAPED AN IROQUOIS RAIDING PARTY.")  # 打印消息：你勉强逃脱了伊罗quois的袭击。
					fmt.Println("HOWEVER, YOU HAD TO LEAVE ALL YOUR FURS BEHIND.")  # 打印消息：然而，你不得不把所有的毛皮留在后面。
					player.lostFurs()  # 调用player对象的lostFurs方法
				} else if event <= 10 {  # 如果事件值小于等于10
					minkPrice /= 2  # 小貂价格减半
					foxPrice /= 2  # 狐狸价格减半
					fmt.Println("YOUR MINK AND BEAVER WERE DAMAGED ON YOUR TRIP.")  # 打印消息：你的水貂和海狸在旅途中受损。
					fmt.Println("YOU RECEIVE ONLY HALF THE CURRENT PRICE FOR THESE FURS.")  # 打印消息：你只能得到这些毛皮当前价格的一半。
				} else {
					log.Fatal("Unexpected error")  # 记录致命错误：意外错误
				}

				fmt.Println()  # 打印空行
				fmt.Println("SUPPLIES AT NEW YORK COST $85.00.")  # 打印消息：纽约的供应品价格为85.00美元。
				fmt.Println("YOUR TRAVEL EXPENSES TO NEW YORK WERE $25.00.")  # 打印消息：你去纽约的旅行费用为25.00美元。
				player.funds -= 110  # player对象的funds属性减去110
			}

			// 计算海狸皮的价值
			beaverValue := beaverPrice * player.furs[FUR_BEAVER]
			// 计算狐狸皮的价值
			foxValue := foxPrice * player.furs[FUR_FOX]
			// 计算貂皮的价值
			ermineValue := erminePrice * player.furs[FUR_ERMINE]
			// 计算水貂皮的价值
			minkValue := minkPrice * player.furs[FUR_MINK]

			// 打印每种皮毛的售价
			fmt.Println()
			fmt.Printf("YOUR BEAVER SOLD FOR $%6.2f\n", float64(beaverValue))
			fmt.Printf("YOUR FOX SOLD FOR    $%6.2f\n", float64(foxValue))
			fmt.Printf("YOUR ERMINE SOLD FOR $%6.2f\n", float64(ermineValue))
			fmt.Printf("YOUR MINK SOLD FOR   $%6.2f\n", float64(minkValue))

			// 更新玩家的资金
			player.funds += float32(beaverValue + foxValue + ermineValue + minkValue)

			// 打印更新后的资金
			fmt.Println()
			fmt.Printf("YOU NOW HAVE $%1.2f INCLUDING YOUR PREVIOUS SAVINGS\n", player.funds)
			fmt.Println("\nDO YOU WANT TO TRADE FURS NEXT YEAR?")
			// 如果玩家选择不再交易皮毛，则退出游戏
			if getYesOrNo() == "N" {
				os.Exit(0)
			} else {  # 如果条件不满足
				gameState = TRADING  # 将游戏状态设置为交易状态
			}
		}
	}
}
```