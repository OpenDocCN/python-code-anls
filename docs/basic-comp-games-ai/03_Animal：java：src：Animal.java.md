# `d:/src/tocomm/basic-computer-games\03_Animal\java\src\Animal.java`

```
# 导入所需的模块
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Scanner;
import java.util.stream.Collectors;

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

    public static void main(String[] args) {
        # 打印游戏介绍
        printIntro();
        # 创建一个用于接收用户输入的 Scanner 对象
        Scanner scan = new Scanner(System.in);
        // 创建一个名为 root 的节点，内容为 "DOES IT SWIM"，左子节点为 "FISH"，右子节点为 "BIRD"
        Node root = new QuestionNode("DOES IT SWIM",
                new AnimalNode("FISH"), new AnimalNode("BIRD"));

        // 初始化一个布尔变量 stopGame，用于控制游戏循环
        boolean stopGame = false;
        // 当游戏未结束时循环
        while (!stopGame) {
            // 从用户输入中读取主要选择
            String choice = readMainChoice(scan);
            // 根据选择进行不同的操作
            switch (choice) {
                // 如果选择为 "TREE"，打印游戏树
                case "TREE":
                    printTree(root);
                    break;
                // 如果选择为 "LIST"，打印已知动物列表
                case "LIST":
                    printKnownAnimals(root);
                    break;
                // 如果选择为 "Q" 或 "QUIT"，结束游戏循环
                case "Q":
                case "QUIT":
                    stopGame = true;
                    break;
                // 如果选择不在以上情况中
                default:
                    // 如果选择以 "Y" 开头，表示用户回答是
                    if (choice.toUpperCase(Locale.ROOT).startsWith("Y")) {
                        Node current = root; // 当前所在的问题树节点
                        Node previous; // 用于跟踪当前节点的父节点，以便稍后放置新的问题。

                        while (current instanceof QuestionNode) { // 当前节点是问题节点时执行循环
                            var currentQuestion = (QuestionNode) current; // 将当前节点转换为问题节点类型
                            var reply = askQuestionAndGetReply(currentQuestion, scan); // 提出问题并获取回复

                            previous = current; // 将当前节点设置为上一个节点
                            current = reply ? currentQuestion.getTrueAnswer() : currentQuestion.getFalseAnswer(); // 根据回复选择下一个节点
                            if (current instanceof AnimalNode) { // 如果当前节点是动物节点
                                // 已经到达动物节点，因此将其作为猜测提供
                                var currentAnimal = (AnimalNode) current; // 将当前节点转换为动物节点类型
                                System.out.printf("IS IT A %s ? ", currentAnimal.getAnimal()); // 提出猜测
                                var animalGuessResponse = readYesOrNo(scan); // 读取是或否的回复
                                if (animalGuessResponse) { // 如果猜测正确
                                    // 猜对了！结束本轮
                                    System.out.println("WHY NOT TRY ANOTHER ANIMAL?");
                                } else {
                                    // 猜错了 :(，请求反馈
                                    // 将上一个节点转换为问题节点，因为在这一点上我们知道它不是叶节点
    /**
     * Prompt for information about the animal we got wrong
     * @param current The animal that we guessed wrong
     * @param previous The root of current
     * @param previousToCurrentDecisionChoice Whether it was a Y or N answer that got us here. true = Y, false = N
     */
    private static void askForInformationAndSave(Scanner scan, AnimalNode current, QuestionNode previous, boolean previousToCurrentDecisionChoice) {
        //Failed to get it right and ran out of questions
        //Let's ask the user for the new information
        System.out.print("THE ANIMAL YOU WERE THINKING OF WAS A ? "); // 提示用户输入他们想到的动物
        String animal = scan.nextLine(); // 从用户输入中获取动物的名称
        // 打印提示用户输入问题，区分两种动物
        System.out.printf("PLEASE TYPE IN A QUESTION THAT WOULD DISTINGUISH A %s FROM A %s ? ", animal, current.getAnimal());
        // 读取用户输入的新问题
        String newQuestion = scan.nextLine();
        // 打印提示用户输入对于当前动物的答案
        System.out.printf("FOR A %s THE ANSWER WOULD BE ? ", animal);
        // 读取用户输入的对于当前动物的答案
        boolean newAnswer = readYesOrNo(scan);
        // 将新问题和答案添加到问题存储中
        addNewAnimal(current, previous, animal, newQuestion, newAnswer, previousToCurrentDecisionChoice);
    }

    // 添加新的动物和问题到问题存储中
    private static void addNewAnimal(Node current,
                                     QuestionNode previous,
                                     String animal,
                                     String newQuestion,
                                     boolean newAnswer,
                                     boolean previousToCurrentDecisionChoice) {
        // 创建新的动物节点
        var animalNode = new AnimalNode(animal);
        // 创建新的问题节点，根据用户输入的答案连接到相应的节点
        var questionNode = new QuestionNode(newQuestion,
                newAnswer ? animalNode : current,
                !newAnswer ? animalNode : current);

        // 如果存在上一个问题节点，则将新的问题节点连接到上一个问题节点
        if (previous != null) {
            // 如果上一个节点到当前节点的决策选择为真，则将上一个节点的真答案设置为当前节点
            if (previousToCurrentDecisionChoice) {
                previous.setTrueAnswer(questionNode);
            } else {
                // 如果上一个节点到当前节点的决策选择为假，则将上一个节点的假答案设置为当前节点
                previous.setFalseAnswer(questionNode);
            }
        }
    }

    // 提出问题并获取回复
    private static boolean askQuestionAndGetReply(QuestionNode questionNode, Scanner scanner) {
        // 打印问题并等待回答
        System.out.printf("%s ? ", questionNode.question);
        return readYesOrNo(scanner);
    }

    // 读取用户输入的是或否
    private static boolean readYesOrNo(Scanner scanner) {
        boolean validAnswer = false;
        Boolean choseAnswer = null;
        // 循环直到输入有效的回答
        while (!validAnswer) {
            String answer = scanner.nextLine();
            // 如果回答以Y开头，则视为是，设置validAnswer为true
            if (answer.toUpperCase(Locale.ROOT).startsWith("Y")) {
                validAnswer = true;
                choseAnswer = true;  # 设置变量choseAnswer为true
            } else if (answer.toUpperCase(Locale.ROOT).startsWith("N")) {  # 如果用户输入的答案以"N"开头（不区分大小写）
                validAnswer = true;  # 设置变量validAnswer为true
                choseAnswer = false;  # 设置变量choseAnswer为false
            }
        }
        return choseAnswer;  # 返回choseAnswer变量的值
    }

    private static void printKnownAnimals(Node root) {  # 定义一个静态方法printKnownAnimals，参数为Node类型的root
        System.out.println("\nANIMALS I ALREADY KNOW ARE:");  # 打印输出字符串"\nANIMALS I ALREADY KNOW ARE:"

        List<AnimalNode> leafNodes = collectLeafNodes(root);  # 调用collectLeafNodes方法，将返回的结果赋值给leafNodes变量
        String allAnimalsString = leafNodes.stream().map(AnimalNode::getAnimal).collect(Collectors.joining("\t\t"));  # 将leafNodes中的AnimalNode对象的getAnimal方法返回的字符串连接起来，用制表符分隔

        System.out.println(allAnimalsString);  # 打印输出allAnimalsString字符串
    }

    //Traverse the tree and collect all the leaf nodes, which basically have all the animals.
    private static List<AnimalNode> collectLeafNodes(Node root) {  # 定义一个静态方法collectLeafNodes，参数为Node类型的root
        List<AnimalNode> collectedNodes = new ArrayList<>();  # 创建一个空的 AnimalNode 列表
        if (root instanceof AnimalNode) {  # 如果根节点是 AnimalNode 类型
            collectedNodes.add((AnimalNode) root);  # 将根节点添加到 collectedNodes 列表中
        } else {  # 如果根节点不是 AnimalNode 类型
            var q = (QuestionNode) root;  # 将根节点转换为 QuestionNode 类型
            collectedNodes.addAll(collectLeafNodes(q.getTrueAnswer()));  # 递归收集真实答案分支的叶子节点
            collectedNodes.addAll(collectLeafNodes(q.getFalseAnswer()));  # 递归收集假答案分支的叶子节点
        }
        return collectedNodes;  # 返回收集到的节点列表
    }

    private static String readMainChoice(Scanner scan) {  # 定义一个静态方法，用于读取用户输入的主要选择
        System.out.print("ARE YOU THINKING OF AN ANIMAL ? ");  # 打印提示信息
        return scan.nextLine();  # 从控制台读取用户输入的内容并返回
    }

    private static void printIntro() {  # 定义一个静态方法，用于打印游戏介绍信息
        System.out.println("                                ANIMAL");  # 打印游戏标题
        System.out.println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  # 打印游戏信息
        System.out.println("\n\n");  # 打印空行
        // 打印游戏标题
        System.out.println("PLAY 'GUESS THE ANIMAL'");
        // 打印空行
        System.out.println("\n");
        // 提示玩家思考动物并告知计算机将尝试猜测
        System.out.println("THINK OF AN ANIMAL AND THE COMPUTER WILL TRY TO GUESS IT.");
    }

    // 根据 https://stackoverflow.com/a/8948691/74057 进行修改
    // 打印树形结构
    private static void printTree(Node root) {
        // 创建一个 StringBuilder 对象
        StringBuilder buffer = new StringBuilder(50);
        // 调用 print 方法打印树形结构
        print(root, buffer, "", "");
        // 打印树形结构
        System.out.println(buffer);
    }

    // 递归打印节点及其子节点
    private static void print(Node root, StringBuilder buffer, String prefix, String childrenPrefix) {
        // 将前缀和节点内容添加到 StringBuilder 对象中
        buffer.append(prefix);
        buffer.append(root.toString());
        buffer.append('\n');

        // 如果节点是问题节点，则继续打印其子节点
        if (root instanceof QuestionNode) {
            // 将问题节点的真实答案作为子节点继续打印
            var questionNode = (QuestionNode) root;
            print(questionNode.getTrueAnswer(), buffer, childrenPrefix + "├─Y─ ", childrenPrefix + "│   ");
    // 打印假答案、缓冲区、子节点前缀和子节点后缀
    print(questionNode.getFalseAnswer(), buffer, childrenPrefix + "└─N─ ", childrenPrefix + "    ");
}

/**
 * 我们问题树中所有节点的基本接口
 */
private interface Node {
}

private static class QuestionNode implements Node {
    private final String question;  // 问题节点的问题内容
    private Node trueAnswer;  // 问题节点的真答案
    private Node falseAnswer;  // 问题节点的假答案

    public QuestionNode(String question, Node trueAnswer, Node falseAnswer) {
        this.question = question;  // 初始化问题节点的问题内容
        this.trueAnswer = trueAnswer;  // 初始化问题节点的真答案
        this.falseAnswer = falseAnswer;  // 初始化问题节点的假答案
```
这些代码是Java语言的代码，主要是定义了一个问题树的节点接口和问题节点类。其中包括了问题节点的问题内容、真答案和假答案的定义和初始化。
        }

        public String getQuestion() {
            return question;  # 返回问题字符串
        }

        public Node getTrueAnswer() {
            return trueAnswer;  # 返回正确答案节点
        }

        public void setTrueAnswer(Node trueAnswer) {
            this.trueAnswer = trueAnswer;  # 设置正确答案节点
        }

        public Node getFalseAnswer() {
            return falseAnswer;  # 返回错误答案节点
        }

        public void setFalseAnswer(Node falseAnswer) {
            this.falseAnswer = falseAnswer;  # 设置错误答案节点
    }

    # 重写 toString 方法，返回 Question 对象的字符串表示
    @Override
    public String toString() {
        return "Question{'" + question + "'}";
    }
}

# 创建 AnimalNode 类实现 Node 接口
private static class AnimalNode implements Node {
    private final String animal;

    # 构造方法，初始化 animal 属性
    public AnimalNode(String animal) {
        this.animal = animal;
    }

    # 获取 animal 属性的值
    public String getAnimal() {
        return animal;
    }

    # 重写接口中的方法
    @Override
# 定义一个公共的方法，用于将对象转换成字符串形式
public String toString() {
    # 返回一个包含动物名称的字符串，格式为 "Animal{'动物名称'}"
    return "Animal{'" + animal + "'}";
}
# 类定义结束
}
# 类定义结束
```