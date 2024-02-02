# `basic-computer-games\57_Literature_Quiz\java\src\LiteratureQuiz.java`

```py
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

    private enum GAME_STATE {
        STARTUP,
        QUESTIONS,
        RESULTS,
        GAME_OVER
    }

    // Current game state
    private GAME_STATE gameState;
    // Players correct answers
    private int correctAnswers;

    public LiteratureQuiz() {

        gameState = GAME_STATE.STARTUP;

        // Initialise kb scanner
        kbScanner = new Scanner(System.in);
    }

    /**
     * Main game loop
     */
    // 游戏介绍
    public void intro() {
        System.out.println(simulateTabs(25) + "LITERATURE QUIZ");
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("LITERATURE QUIZ");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("TEST YOUR KNOWLEDGE OF CHILDREN'S LITERATURE.");
        System.out.println("THIS IS A MULTIPLE-CHOICE QUIZ.");
        System.out.println("TYPE A 1, 2, 3, OR 4 AFTER THE QUESTION MARK.");
        System.out.println();
        System.out.println("GOOD LUCK!");
        System.out.println();
    }

    /**
     * Simulate the old basic tab(xx) command which indented text by xx spaces.
     *
     * @param spaces number of spaces required
     * @return String with number of spaces
     */
    // 模拟旧的基本tab(xx)命令，将文本缩进xx个空格
    # 创建一个由空格组成的字符串，用于模拟制表符
    private String simulateTabs(int spaces) {
        char[] spacesTemp = new char[spaces];
        Arrays.fill(spacesTemp, ' ');
        return new String(spacesTemp);
    }

    /*
     * 在屏幕上打印一条消息，然后从键盘接受输入。
     * 将输入转换为整数
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    private int displayTextAndGetNumber(String text) {
        return Integer.parseInt(displayTextAndGetInput(text));
    }

    /*
     * 在屏幕上打印一条消息，然后从键盘接受输入。
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next();
    }
# 闭合前面的函数定义
```