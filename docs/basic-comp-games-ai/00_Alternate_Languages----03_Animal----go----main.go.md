# `00_Alternate_Languages\03_Animal\go\main.go`

```
package main  // 声明包名为 main

import (  // 导入所需的包
	"bufio"  // 用于读取输入
	"fmt"  // 用于格式化输出
	"log"  // 用于记录日志
	"os"  // 用于操作文件和目录
	"strings"  // 用于处理字符串
)

type node struct {  // 定义结构体 node
	text    string  // 字符串类型的字段 text
	yesNode *node  // 指向 node 结构体的指针类型字段 yesNode
	noNode  *node  // 指向 node 结构体的指针类型字段 noNode
}

func newNode(text string, yes_node, no_node *node) *node {  // 定义函数 newNode，接收字符串和两个 node 结构体指针作为参数，返回一个 node 结构体指针
	n := node{text: text}  // 创建一个 node 结构体，并初始化 text 字段
	if yes_node != nil {  // 如果 yes_node 不为空
		n.yesNode = yes_node  // 将 yes_node 赋值给 n 的 yesNode 字段
	}
	// 如果 no_node 不为空，则将当前节点的 noNode 指向新的节点
	if no_node != nil {
		n.noNode = no_node
	}
	// 返回当前节点的指针
	return &n
}

// 更新节点的问题、答案和动物
func (n *node) update(newQuestion, newAnswer, newAnimal string) {
	// 保存旧的动物
	oldAnimal := n.text

	// 更新节点的问题
	n.text = newQuestion

	// 根据新答案的值，创建新的节点并更新当前节点的 yesNode 和 noNode
	if newAnswer == "y" {
		n.yesNode = newNode(newAnimal, nil, nil)
		n.noNode = newNode(oldAnimal, nil, nil)
	} else {
		n.yesNode = newNode(oldAnimal, nil, nil)
		n.noNode = newNode(newAnimal, nil, nil)
	}
}
func (n *node) isLeaf() bool {
	return (n.yesNode == nil) && (n.noNode == nil)  // 检查当前节点是否为叶子节点，即没有子节点
}

func listKnownAnimals(root *node) {
	if root == nil {  // 如果根节点为空，则直接返回
		return
	}

	if root.isLeaf() {  // 如果当前节点是叶子节点
		fmt.Printf("%s           ", root.text)  // 打印当前节点的文本内容
		return
	}

	if root.yesNode != nil {  // 如果当前节点有一个“是”节点
		listKnownAnimals(root.yesNode)  // 递归调用函数，传入“是”节点作为根节点
	}

	if root.noNode != nil {  // 如果当前节点有一个“否”节点
		listKnownAnimals(root.noNode)  // 调用函数listKnownAnimals，传入参数root.noNode
	}
}

func parseInput(message string, checkList bool, rootNode *node) string {
	scanner := bufio.NewScanner(os.Stdin)  // 创建一个用于从标准输入读取数据的Scanner
	token := ""  // 初始化一个空字符串变量token

	for {  // 进入无限循环
		fmt.Println(message)  // 打印message的内容
		scanner.Scan()  // 从标准输入读取下一行
		inp := strings.ToLower(scanner.Text())  // 将读取的内容转换为小写并赋值给变量inp

		if checkList && inp == "list" {  // 如果checkList为true且inp等于"list"
			fmt.Println("Animals I already know are:")  // 打印提示信息
			listKnownAnimals(rootNode)  // 调用函数listKnownAnimals，传入参数rootNode
			fmt.Println()  // 打印空行
		}

		if len(inp) > 0 {  // 如果inp的长度大于0
		token = inp  # 将输入的值赋给变量token
	} else {
		token = ""  # 如果输入的值不是"y"或者"n"，则将token置为空字符串
	}

	if token == "y" || token == "n" {  # 如果token的值是"y"或者"n"，则跳出循环
		break
	}
}
return token  # 返回token的值
}

func avoidVoidInput(message string) string {
scanner := bufio.NewScanner(os.Stdin)  # 创建一个从标准输入读取数据的Scanner对象
answer := ""  # 初始化answer变量为空字符串
for {
fmt.Println(message)  # 打印提示信息
scanner.Scan()  # 从标准输入读取下一行数据
answer = scanner.Text()  # 将读取的数据赋给answer变量
		if answer != "" {  # 如果答案不为空
			break  # 跳出循环
		}
	}
	return answer  # 返回答案
}

func printIntro() {  # 定义函数printIntro
	fmt.Println("                                Animal")  # 打印"                                Animal"
	fmt.Println("               Creative Computing Morristown, New Jersey")  # 打印"               Creative Computing Morristown, New Jersey"
	fmt.Println("\nPlay 'Guess the Animal'")  # 打印"\nPlay 'Guess the Animal'"
	fmt.Println("Think of an animal and the computer will try to guess it")  # 打印"Think of an animal and the computer will try to guess it"
}

func main() {  # 定义主函数main
	yesChild := newNode("Fish", nil, nil)  # 创建一个名为"Fish"的新节点，没有子节点
	noChild := newNode("Bird", nil, nil)  # 创建一个名为"Bird"的新节点，没有子节点
	rootNode := newNode("Does it swim?", yesChild, noChild)  # 创建一个名为"Does it swim?"的新节点，其子节点为yesChild和noChild

	printIntro()  # 调用函数printIntro
	// 根据用户输入判断是否继续游戏
	keepPlaying := (parseInput("Are you thinking of an animal?", true, rootNode) == "y")

	// 当需要继续游戏时执行循环
	for keepPlaying {
		keepAsking := true

		// 初始化当前节点为根节点
		actualNode := rootNode

		// 当需要继续询问时执行循环
		for keepAsking {
			// 如果当前节点不是叶子节点
			if !actualNode.isLeaf() {
				// 根据当前节点的文本提示用户输入
				answer := parseInput(actualNode.text, false, nil)

				// 如果用户回答是
				if answer == "y" {
					// 如果当前节点的yesNode为空，输出错误信息
					if actualNode.yesNode == nil {
						log.Fatal("invalid node")
					}
					// 将当前节点移动到yesNode
					actualNode = actualNode.yesNode
				} else {
					// 如果当前节点的noNode为空，输出错误信息
					if actualNode.noNode == nil {
						log.Fatal("invalid node")
					}
					actualNode = actualNode.noNode  # 更新当前节点为否定节点
				}
			} else {
				answer := parseInput(fmt.Sprintf("Is it a %s?", actualNode.text), false, nil)  # 获取用户输入的答案
				if answer == "n" {  # 如果用户答案为否定
					newAnimal := avoidVoidInput("The animal you were thinking of was a ?")  # 获取用户输入的新动物名称
					newQuestion := avoidVoidInput(fmt.Sprintf("Please type in a question that would distinguish a '%s' from a '%s':", newAnimal, actualNode.text))  # 获取用户输入的新问题
					newAnswer := parseInput(fmt.Sprintf("For a '%s' the answer would be", newAnimal), false, nil)  # 获取用户输入的新问题的答案
					actualNode.update(newQuestion+"?", newAnswer, newAnimal)  # 更新当前节点的信息
				} else {
					fmt.Println("Why not try another animal?")  # 提示用户尝试另一个动物
				}
				keepAsking = false  # 停止询问
			}
		}
		keepPlaying = (parseInput("Are you thinking of an animal?", true, rootNode) == "y")  # 获取用户是否继续玩游戏的答案
	}
}
```