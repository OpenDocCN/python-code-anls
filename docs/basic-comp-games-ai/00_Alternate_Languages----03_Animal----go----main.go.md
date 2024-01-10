# `basic-computer-games\00_Alternate_Languages\03_Animal\go\main.go`

```
package main

import (
    "bufio"  // 导入 bufio 包，用于读取输入
    "fmt"    // 导入 fmt 包，用于格式化输出
    "log"    // 导入 log 包，用于记录日志
    "os"     // 导入 os 包，提供对操作系统功能的访问
    "strings"  // 导入 strings 包，提供对字符串的操作
)

type node struct {
    text    string  // 定义节点结构体，包含文本字段
    yesNode *node   // 定义节点结构体，包含指向“是”节点的指针
    noNode  *node   // 定义节点结构体，包含指向“否”节点的指针
}

func newNode(text string, yes_node, no_node *node) *node {
    n := node{text: text}  // 创建新节点
    if yes_node != nil {   // 如果“是”节点不为空
        n.yesNode = yes_node  // 设置新节点的“是”节点
    }
    if no_node != nil {    // 如果“否”节点不为空
        n.noNode = no_node   // 设置新节点的“否”节点
    }
    return &n  // 返回新节点的指针
}

func (n *node) update(newQuestion, newAnswer, newAnimal string) {
    oldAnimal := n.text  // 保存当前节点的文本

    n.text = newQuestion  // 更新当前节点的文本为新问题

    if newAnswer == "y" {  // 如果新答案是“y”
        n.yesNode = newNode(newAnimal, nil, nil)  // 设置新节点的“是”节点为新动物节点
        n.noNode = newNode(oldAnimal, nil, nil)   // 设置新节点的“否”节点为原动物节点
    } else {
        n.yesNode = newNode(oldAnimal, nil, nil)   // 设置新节点的“是”节点为原动物节点
        n.noNode = newNode(newAnimal, nil, nil)    // 设置新节点的“否”节点为新动物节点
    }
}

func (n *node) isLeaf() bool {
    return (n.yesNode == nil) && (n.noNode == nil)  // 返回当前节点是否为叶子节点
}

func listKnownAnimals(root *node) {
    if root == nil {  // 如果根节点为空
        return  // 直接返回
    }

    if root.isLeaf() {  // 如果当前节点是叶子节点
        fmt.Printf("%s           ", root.text)  // 格式化输出当前节点的文本
        return  // 直接返回
    }

    if root.yesNode != nil {  // 如果“是”节点不为空
        listKnownAnimals(root.yesNode)  // 递归调用列出已知动物的函数
    }

    if root.noNode != nil {  // 如果“否”节点不为空
        listKnownAnimals(root.noNode)  // 递归调用列出已知动物的函数
    }
}

func parseInput(message string, checkList bool, rootNode *node) string {
    scanner := bufio.NewScanner(os.Stdin)  // 创建标准输入的扫描器
    token := ""  // 初始化 token 为空字符串

    for {
        fmt.Println(message)  // 输出提示信息
        scanner.Scan()  // 扫描输入
        inp := strings.ToLower(scanner.Text())  // 将输入转换为小写

        if checkList && inp == "list" {  // 如果需要检查列表且输入为“list”
            fmt.Println("Animals I already know are:")  // 输出已知动物列表的提示
                    listKnownAnimals(rootNode)
                    // 调用函数列出已知动物
                    fmt.Println()
                    // 打印空行
                }
                if len(inp) > 0 {
                    // 如果输入长度大于0
                    token = inp
                    // 将输入赋值给token
                } else {
                    // 否则
                    token = ""
                    // 将token置为空字符串
                }
                if token == "y" || token == "n" {
                    // 如果token为"y"或"n"
                    break
                    // 跳出循环
                }
            }
            return token
            // 返回token值
        }

        func avoidVoidInput(message string) string {
            // 定义函数，避免空输入
            scanner := bufio.NewScanner(os.Stdin)
            // 创建标准输入的扫描器
            answer := ""
            // 初始化答案为空字符串
            for {
                // 无限循环
                fmt.Println(message)
                // 打印消息
                scanner.Scan()
                // 扫描输入
                answer = scanner.Text()
                // 将输入赋值给answer
                if answer != "" {
                    // 如果答案不为空
                    break
                    // 跳出循环
                }
            }
            return answer
            // 返回答案
        }

        func printIntro() {
            // 定义函数，打印介绍
            fmt.Println("                                Animal")
            // 打印动物
            fmt.Println("               Creative Computing Morristown, New Jersey")
            // 打印创意计算摩里斯敦，新泽西
            fmt.Println("\nPlay 'Guess the Animal'")
            // 打印玩“猜动物”
            fmt.Println("Think of an animal and the computer will try to guess it")
            // 打印想一个动物，计算机会试图猜测它
        }

        func main() {
            // 主函数
            yesChild := newNode("Fish", nil, nil)
            // 创建一个新节点，值为"Fish"
            noChild := newNode("Bird", nil, nil)
            // 创建一个新节点，值为"Bird"
            rootNode := newNode("Does it swim?", yesChild, noChild)
            // 创建一个新节点，值为"Does it swim?"，左子节点为yesChild，右子节点为noChild

            printIntro()
            // 调用打印介绍函数

            keepPlaying := (parseInput("Are you thinking of an animal?", true, rootNode) == "y")
            // 将是否继续玩的结果赋值给keepPlaying

            for keepPlaying {
                // 循环直到keepPlaying为false
                keepAsking := true
                // 初始化keepAsking为true

                actualNode := rootNode
                // 初始化actualNode为rootNode

                for keepAsking {
                    // 循环直到keepAsking为false
                    if !actualNode.isLeaf() {
                        // 如果actualNode不是叶子节点
                        answer := parseInput(actualNode.text, false, nil)
                        // 调用解析输入函数，将结果赋值给answer

                        if answer == "y" {
                            // 如果答案为"y"
    # 如果当前节点的yesNode为空，则输出错误信息并终止程序
    if actualNode.yesNode == nil {
        log.Fatal("invalid node")
    }
    # 将当前节点移动到yesNode指向的节点
    actualNode = actualNode.yesNode
    } else {
    # 如果当前节点的noNode为空，则输出错误信息并终止程序
    if actualNode.noNode == nil {
        log.Fatal("invalid node")
    }
    # 将当前节点移动到noNode指向的节点
    actualNode = actualNode.noNode
    }
    # 如果以上条件都不满足，则执行以下代码块
    } else {
    # 询问用户问题，并根据用户输入进行不同的处理
    answer := parseInput(fmt.Sprintf("Is it a %s?", actualNode.text), false, nil)
    # 如果用户输入为"n"，则执行以下代码块
    if answer == "n" {
        # 获取用户输入的新动物名称
        newAnimal := avoidVoidInput("The animal you were thinking of was a ?")
        # 获取用户输入的新问题
        newQuestion := avoidVoidInput(fmt.Sprintf("Please type in a question that would distinguish a '%s' from a '%s':", newAnimal, actualNode.text))
        # 获取用户输入的新答案
        newAnswer := parseInput(fmt.Sprintf("For a '%s' the answer would be", newAnimal), false, nil)
        # 更新当前节点的问题、答案和动物名称
        actualNode.update(newQuestion+"?", newAnswer, newAnimal)
    } else {
        # 输出提示信息
        fmt.Println("Why not try another animal?")
    }
    # 将keepAsking设置为false，结束循环
    keepAsking = false
    }
# 检查是否继续游戏，根据用户输入和树的根节点进行解析，如果用户输入是"y"，则继续游戏
keepPlaying = (parseInput("Are you thinking of an animal?", true, rootNode) == "y")
```