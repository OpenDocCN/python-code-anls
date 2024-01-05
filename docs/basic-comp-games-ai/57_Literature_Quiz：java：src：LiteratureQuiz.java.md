# `d:/src/tocomm/basic-computer-games\57_Literature_Quiz\java\src\LiteratureQuiz.java`

```
import java.util.Arrays;
import java.util.Scanner;

/**
 * Game of Literature Quiz
 * <p>
 * Based on the Basic game of Literature Quiz here
 * https://github.com/coding-horror/basic-computer-games/blob/main/57%20Literature%20Quiz/litquiz.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class LiteratureQuiz {

    // Used for keyboard input
    private final Scanner kbScanner;

    // 定义游戏状态的枚举类型
    private enum GAME_STATE {
        STARTUP,  // 游戏启动状态
        QUESTIONS,  // 问题状态
        RESULTS,  // 定义游戏可能的结果
        GAME_OVER  // 定义游戏结束的状态
    }

    // 当前游戏状态
    private GAME_STATE gameState;
    // 玩家正确答案的数量
    private int correctAnswers;

    public LiteratureQuiz() {

        gameState = GAME_STATE.STARTUP;  // 将游戏状态初始化为启动状态

        // 初始化键盘输入扫描器
        kbScanner = new Scanner(System.in);
    }

    /**
     * 主游戏循环
     */
    public void play() {

        do {
            switch (gameState) {

                // Show an introduction the first time the game is played.
                case STARTUP:
                    intro();  # 调用intro()函数，显示游戏介绍
                    correctAnswers = 0;  # 将正确答案数量初始化为0
                    gameState = GAME_STATE.QUESTIONS;  # 将游戏状态设置为问题状态
                    break;

                // Ask the player four questions
                case QUESTIONS:

                    // Question 1
                    System.out.println("IN PINOCCHIO, WHAT WAS THE NAME OF THE CAT");  # 在控制台打印问题1
                    int question1Answer = displayTextAndGetNumber("1)TIGGER, 2)CICERO, 3)FIGARO, 4)GUIPETTO ? ");  # 调用displayTextAndGetNumber()函数获取玩家对问题1的答案
                    if (question1Answer == 3) {  # 如果玩家答案为3
                        System.out.println("VERY GOOD!  HERE'S ANOTHER.");  # 在控制台打印"非常好！这是另一个问题。"
                    correctAnswers++;  // 增加正确答案的计数
                } else {
                    System.out.println("SORRY...FIGARO WAS HIS NAME.");  // 打印错误提示信息
                }

                System.out.println();  // 打印空行

                // Question 2
                System.out.println("FROM WHOSE GARDEN DID BUGS BUNNY STEAL THE CARROTS?");  // 打印问题2
                int question2Answer = displayTextAndGetNumber("1)MR. NIXON'S, 2)ELMER FUDD'S, 3)CLEM JUDD'S, 4)STROMBOLI'S ? ");  // 显示选项并获取用户输入的答案
                if (question2Answer == 2) {
                    System.out.println("PRETTY GOOD!");  // 打印正确提示信息
                    correctAnswers++;  // 增加正确答案的计数
                } else {
                    System.out.println("TOO BAD...IT WAS ELMER FUDD'S GARDEN.");  // 打印错误提示信息
                }

                System.out.println();  // 打印空行

                // Question 3
// 打印"WIZARD OF OS"中的一句话，并等待用户输入答案
System.out.println("IN THE WIZARD OF OS, DOROTHY'S DOG WAS NAMED");
int question3Answer = displayTextAndGetNumber("1)CICERO, 2)TRIXIA, 3)KING, 4)TOTO ? ");
// 如果用户输入的答案是4，则打印"YEA!  YOU'RE A REAL LITERATURE GIANT."，并且正确答案数量加一
if (question3Answer == 4) {
    System.out.println("YEA!  YOU'RE A REAL LITERATURE GIANT.");
    correctAnswers++;
} else {
    // 如果用户输入的答案不是4，则打印"BACK TO THE BOOKS,...TOTO WAS HIS NAME."
    System.out.println("BACK TO THE BOOKS,...TOTO WAS HIS NAME.");
}

System.out.println();

// 打印另一句话，并等待用户输入答案
System.out.println("WHO WAS THE FAIR MAIDEN WHO ATE THE POISON APPLE");
int question4Answer = displayTextAndGetNumber("1)SLEEPING BEAUTY, 2)CINDERELLA, 3)SNOW WHITE, 4)WENDY ? ");
// 如果用户输入的答案是3，则打印"GOOD MEMORY!"，并且正确答案数量加一
if (question4Answer == 3) {
    System.out.println("GOOD MEMORY!");
    correctAnswers++;
} else {
    // 如果用户输入的答案不是3，则打印"OH, COME ON NOW...IT WAS SNOW WHITE."
    System.out.println("OH, COME ON NOW...IT WAS SNOW WHITE.");
}
// 打印空行
System.out.println();
// 将游戏状态设置为RESULTS
gameState = GAME_STATE.RESULTS;
// 跳出switch语句
break;

// 玩家表现如何？
case RESULTS:
    // 如果答对了4个问题
    if (correctAnswers == 4) {
        // 全部正确
        System.out.println("WOW!  THAT'S SUPER!  YOU REALLY KNOW YOUR NURSERY");
        System.out.println("YOUR NEXT QUIZ WILL BE ON 2ND CENTURY CHINESE");
        System.out.println("LITERATURE (HA, HA, HA)");
    // 如果答对少于2个问题
    } else if (correctAnswers < 2) {
        System.out.println("UGH.  THAT WAS DEFINITELY NOT TOO SWIFT.  BACK TO");
        System.out.println("NURSERY SCHOOL FOR YOU, MY FRIEND.");
    // 如果答对了两个或三个问题
    } else {
        System.out.println("NOT BAD, BUT YOU MIGHT SPEND A LITTLE MORE TIME");
        System.out.println("READING THE NURSERY GREATS.");
    }
    // 设置游戏状态为 GAME_OVER，表示游戏结束
    gameState = GAME_STATE.GAME_OVER;
    // 退出循环
    break;
}

// 执行 do-while 循环，直到游戏状态为 GAME_OVER
} while (gameState != GAME_STATE.GAME_OVER);
}

// 定义 intro 方法，用于显示游戏介绍信息
public void intro() {
    // 打印游戏标题和地点
    System.out.println(simulateTabs(25) + "LITERATURE QUIZ");
    System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    System.out.println();
    // 打印游戏标题和地点
    System.out.println("LITERATURE QUIZ");
    System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    System.out.println();
    // 打印游戏介绍信息
    System.out.println("TEST YOUR KNOWLEDGE OF CHILDREN'S LITERATURE.");
    System.out.println("THIS IS A MULTIPLE-CHOICE QUIZ.");
    System.out.println("TYPE A 1, 2, 3, OR 4 AFTER THE QUESTION MARK.");
    System.out.println();
    System.out.println("GOOD LUCK!");
    System.out.println();
}
    }

    /**
     * 模拟旧的基本tab(xx)命令，该命令通过xx个空格缩进文本。
     *
     * @param spaces 需要的空格数
     * @return 具有空格数的字符串
     */
    private String simulateTabs(int spaces) {
        char[] spacesTemp = new char[spaces];
        Arrays.fill(spacesTemp, ' ');
        return new String(spacesTemp);
    }

    /*
     * 在屏幕上打印一条消息，然后从键盘接受输入。
     * 将输入转换为整数
     *
     * @param text 要在屏幕上显示的消息。
     * @return 玩家输入的内容。
    */
    # 根据显示的文本在屏幕上打印消息，然后从键盘接受输入。
    # @param text 要在屏幕上显示的消息。
    # @return 玩家输入的内容。
    private String displayTextAndGetInput(String text) {
        System.out.print(text);  # 在屏幕上打印消息
        return kbScanner.next();  # 从键盘接受输入并返回
    }
```