# `basic-computer-games\00_Alternate_Languages\03_Animal\go\main.go`

```

package main

import (
	"bufio" // 导入 bufio 包，用于读取输入
	"fmt" // 导入 fmt 包，用于格式化输出
	"log" // 导入 log 包，用于记录日志
	"os" // 导入 os 包，用于操作系统功能
	"strings" // 导入 strings 包，用于处理字符串
)

type node struct {
	text    string
	yesNode *node
	noNode  *node
}

func newNode(text string, yes_node, no_node *node) *node {
	// 创建新的节点
	n := node{text: text}
	// 如果有 yes_node，则设置为 yesNode
	if yes_node != nil {
		n.yesNode = yes_node
	}
	// 如果有 no_node，则设置为 noNode
	if no_node != nil {
		n.noNode = no_node
	}
	return &n
}

func (n *node) update(newQuestion, newAnswer, newAnimal string) {
	// 更新节点的问题和答案
	oldAnimal := n.text
	n.text = newQuestion
	// 根据新答案创建新节点
	if newAnswer == "y" {
		n.yesNode = newNode(newAnimal, nil, nil)
		n.noNode = newNode(oldAnimal, nil, nil)
	} else {
		n.yesNode = newNode(oldAnimal, nil, nil)
		n.noNode = newNode(newAnimal, nil, nil)
	}
}

func (n *node) isLeaf() bool {
	// 判断节点是否为叶子节点
	return (n.yesNode == nil) && (n.noNode == nil)
}

func listKnownAnimals(root *node) {
	// 列出已知的动物
	if root == nil {
		return
	}
	// 如果是叶子节点，则打印动物名称
	if root.isLeaf() {
		fmt.Printf("%s           ", root.text)
		return
	}
	// 递归遍历子节点
	if root.yesNode != nil {
		listKnownAnimals(root.yesNode)
	}
	if root.noNode != nil {
		listKnownAnimals(root.noNode)
	}
}

func parseInput(message string, checkList bool, rootNode *node) string {
	// 解析输入
	scanner := bufio.NewScanner(os.Stdin)
	token := ""
	for {
		fmt.Println(message)
		scanner.Scan()
		inp := strings.ToLower(scanner.Text())
		// 如果需要列出已知的动物，则调用 listKnownAnimals 函数
		if checkList && inp == "list" {
			fmt.Println("Animals I already know are:")
			listKnownAnimals(rootNode)
			fmt.Println()
		}
		// 如果输入不为空，则赋值给 token
		if len(inp) > 0 {
			token = inp
		} else {
			token = ""
		}
		// 如果 token 为 y 或 n，则跳出循环
		if token == "y" || token == "n" {
			break
		}
	}
	return token
}

func avoidVoidInput(message string) string {
	// 避免空输入
	scanner := bufio.NewScanner(os.Stdin)
	answer := ""
	for {
		fmt.Println(message)
		scanner.Scan()
		answer = scanner.Text()
		// 如果答案不为空，则跳出循环
		if answer != "" {
			break
		}
	}
	return answer
}

func printIntro() {
	// 打印游戏介绍
	fmt.Println("                                Animal")
	fmt.Println("               Creative Computing Morristown, New Jersey")
	fmt.Println("\nPlay 'Guess the Animal'")
	fmt.Println("Think of an animal and the computer will try to guess it")
}

func main() {
	// 创建根节点和子节点
	yesChild := newNode("Fish", nil, nil)
	noChild := newNode("Bird", nil, nil)
	rootNode := newNode("Does it swim?", yesChild, noChild)

	printIntro()

	keepPlaying := (parseInput("Are you thinking of an animal?", true, rootNode) == "y")

	for keepPlaying {
		keepAsking := true
		actualNode := rootNode
		// 循环询问直到猜对动物
		for keepAsking {
			if !actualNode.isLeaf() {
				answer := parseInput(actualNode.text, false, nil)
				// 根据答案更新当前节点
				if answer == "y" {
					if actualNode.yesNode == nil {
						log.Fatal("invalid node")
					}
					actualNode = actualNode.yesNode
				} else {
					if actualNode.noNode == nil {
						log.Fatal("invalid node")
					}
					actualNode = actualNode.noNode
				}
			} else {
				answer := parseInput(fmt.Sprintf("Is it a %s?", actualNode.text), false, nil)
				if answer == "n" {
					newAnimal := avoidVoidInput("The animal you were thinking of was a ?")
					newQuestion := avoidVoidInput(fmt.Sprintf("Please type in a question that would distinguish a '%s' from a '%s':", newAnimal, actualNode.text))
					newAnswer := parseInput(fmt.Sprintf("For a '%s' the answer would be", newAnimal), false, nil)
					actualNode.update(newQuestion+"?", newAnswer, newAnimal)
				} else {
					fmt.Println("Why not try another animal?")
				}
				keepAsking = false
			}
		}
		keepPlaying = (parseInput("Are you thinking of an animal?", true, rootNode) == "y")
	}
}

```