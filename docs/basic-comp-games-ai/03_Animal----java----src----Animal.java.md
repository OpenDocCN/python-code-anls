# `basic-computer-games\03_Animal\java\src\Animal.java`

```py
import java.util.ArrayList;  // 导入 ArrayList 类
import java.util.List;  // 导入 List 接口
import java.util.Locale;  // 导入 Locale 类
import java.util.Scanner;  // 导入 Scanner 类
import java.util.stream.Collectors;  // 导入 Collectors 类

/**
 * ANIMAL
 * <p>
 * Converted from BASIC to Java by Aldrin Misquitta (@aldrinm)
 * The original BASIC program uses an array to maintain the questions and answers and to decide which question to
 * ask next. Updated this Java implementation to use a tree instead of the earlier faulty one based on a list (thanks @patimen).
 *
 * Bonus option: TREE --> prints the game decision data as a tree to visualize/debug the state of the game
 */
public class Animal {

    }

    /**
     * Prompt for information about the animal we got wrong
     * @param current The animal that we guessed wrong
     * @param previous The root of current
     * @param previousToCurrentDecisionChoice Whether it was a Y or N answer that got us here. true = Y, false = N
     */
    private static void askForInformationAndSave(Scanner scan, AnimalNode current, QuestionNode previous, boolean previousToCurrentDecisionChoice) {
        //Failed to get it right and ran out of questions
        //Let's ask the user for the new information
        System.out.print("THE ANIMAL YOU WERE THINKING OF WAS A ? ");  // 提示用户输入错误猜测的动物
        String animal = scan.nextLine();  // 读取用户输入的动物名称
        System.out.printf("PLEASE TYPE IN A QUESTION THAT WOULD DISTINGUISH A %s FROM A %s ? ", animal, current.getAnimal());  // 提示用户输入一个可以区分用户猜测的动物和正确动物的问题
        String newQuestion = scan.nextLine();  // 读取用户输入的新问题
        System.out.printf("FOR A %s THE ANSWER WOULD BE ? ", animal);  // 提示用户输入对于新动物的答案
        boolean newAnswer = readYesOrNo(scan);  // 调用 readYesOrNo 方法读取用户输入的答案
        //Add it to our question store
        addNewAnimal(current, previous, animal, newQuestion, newAnswer, previousToCurrentDecisionChoice);  // 将新动物、新问题和答案添加到问题存储中
    }
    // 添加新的动物节点和问题节点到当前节点下
    private static void addNewAnimal(Node current,
                                     QuestionNode previous,
                                     String animal,
                                     String newQuestion,
                                     boolean newAnswer,
                                     boolean previousToCurrentDecisionChoice) {
        // 创建新的动物节点
        var animalNode = new AnimalNode(animal);
        // 创建新的问题节点
        var questionNode = new QuestionNode(newQuestion,
                newAnswer ? animalNode : current,
                !newAnswer ? animalNode : current);

        // 如果存在上一个节点
        if (previous != null) {
            // 根据上一个节点到当前节点的决策选择，设置上一个节点的真假回答
            if (previousToCurrentDecisionChoice) {
                previous.setTrueAnswer(questionNode);
            } else {
                previous.setFalseAnswer(questionNode);
            }
        }
    }

    // 提问并获取回答
    private static boolean askQuestionAndGetReply(QuestionNode questionNode, Scanner scanner) {
        // 打印问题并获取回答
        System.out.printf("%s ? ", questionNode.question);
        return readYesOrNo(scanner);
    }

    // 读取用户输入的是或否
    private static boolean readYesOrNo(Scanner scanner) {
        boolean validAnswer = false;
        Boolean choseAnswer = null;
        // 循环直到获取有效的回答
        while (!validAnswer) {
            String answer = scanner.nextLine();
            // 如果回答以Y开头，则为是
            if (answer.toUpperCase(Locale.ROOT).startsWith("Y")) {
                validAnswer = true;
                choseAnswer = true;
            } 
            // 如果回答以N开头，则为否
            else if (answer.toUpperCase(Locale.ROOT).startsWith("N")) {
                validAnswer = true;
                choseAnswer = false;
            }
        }
        return choseAnswer;
    }

    // 打印已知的动物
    private static void printKnownAnimals(Node root) {
        // 打印已知的动物
        System.out.println("\nANIMALS I ALREADY KNOW ARE:");

        // 收集叶子节点
        List<AnimalNode> leafNodes = collectLeafNodes(root);
        // 将所有动物名称连接成字符串并打印
        String allAnimalsString = leafNodes.stream().map(AnimalNode::getAnimal).collect(Collectors.joining("\t\t"));

        System.out.println(allAnimalsString);
    }
    // 遍历树并收集所有叶子节点，这些节点基本上包含了所有的动物。
    private static List<AnimalNode> collectLeafNodes(Node root) {
        List<AnimalNode> collectedNodes = new ArrayList<>();
        if (root instanceof AnimalNode) {
            collectedNodes.add((AnimalNode) root);
        } else {
            var q = (QuestionNode) root;
            collectedNodes.addAll(collectLeafNodes(q.getTrueAnswer()));
            collectedNodes.addAll(collectLeafNodes(q.getFalseAnswer()));
        }
        return collectedNodes;
    }

    // 读取主要选择
    private static String readMainChoice(Scanner scan) {
        System.out.print("ARE YOU THINKING OF AN ANIMAL ? ");
        return scan.nextLine();
    }

    // 打印介绍
    private static void printIntro() {
        System.out.println("                                ANIMAL");
        System.out.println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println("\n\n");
        System.out.println("PLAY 'GUESS THE ANIMAL'");
        System.out.println("\n");
        System.out.println("THINK OF AN ANIMAL AND THE COMPUTER WILL TRY TO GUESS IT.");
    }

    // 基于 https://stackoverflow.com/a/8948691/74057
    private static void printTree(Node root) {
        StringBuilder buffer = new StringBuilder(50);
        print(root, buffer, "", "");
        System.out.println(buffer);
    }

    private static void print(Node root, StringBuilder buffer, String prefix, String childrenPrefix) {
        buffer.append(prefix);
        buffer.append(root.toString());
        buffer.append('\n');

        if (root instanceof QuestionNode) {
            var questionNode = (QuestionNode) root;
            print(questionNode.getTrueAnswer(), buffer, childrenPrefix + "├─Y─ ", childrenPrefix + "│   ");
            print(questionNode.getFalseAnswer(), buffer, childrenPrefix + "└─N─ ", childrenPrefix + "    ");
        }
    }

    /**
     * 我们问题树中所有节点的基本接口
     */
    # 定义一个接口 Node
    private interface Node {
    }

    # 定义一个实现 Node 接口的 QuestionNode 类
    private static class QuestionNode implements Node {
        # 问题字符串
        private final String question;
        # 真实情况下的回答节点
        private Node trueAnswer;
        # 错误情况下的回答节点
        private Node falseAnswer;

        # QuestionNode 类的构造函数，初始化 question、trueAnswer 和 falseAnswer
        public QuestionNode(String question, Node trueAnswer, Node falseAnswer) {
            this.question = question;
            this.trueAnswer = trueAnswer;
            this.falseAnswer = falseAnswer;
        }

        # 获取问题字符串
        public String getQuestion() {
            return question;
        }

        # 获取真实情况下的回答节点
        public Node getTrueAnswer() {
            return trueAnswer;
        }

        # 设置真实情况下的回答节点
        public void setTrueAnswer(Node trueAnswer) {
            this.trueAnswer = trueAnswer;
        }

        # 获取错误情况下的回答节点
        public Node getFalseAnswer() {
            return falseAnswer;
        }

        # 设置错误情况下的回答节点
        public void setFalseAnswer(Node falseAnswer) {
            this.falseAnswer = falseAnswer;
        }

        # 重写 toString 方法，返回问题字符串
        @Override
        public String toString() {
            return "Question{'" + question + "'}";
        }
    }

    # 定义一个实现 Node 接口的 AnimalNode 类
    private static class AnimalNode implements Node {
        # 动物字符串
        private final String animal;

        # AnimalNode 类的构造函数，初始化 animal
        public AnimalNode(String animal) {
            this.animal = animal;
        }

        # 获取动物字符串
        public String getAnimal() {
            return animal;
        }

        # 重写 toString 方法，返回动物字符串
        @Override
        public String toString() {
            return "Animal{'" + animal + "'}";
        }
    }
# 闭合前面的函数定义
```