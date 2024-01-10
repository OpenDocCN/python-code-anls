# `basic-computer-games\54_Letter\java\src\Letter.java`

```
import java.awt.*;
import java.util.Arrays;
import java.util.Scanner;

/**
 * Game of Letter
 * <p>
 * Based on the Basic game of Letter here
 * https://github.com/coding-horror/basic-computer-games/blob/main/54%20Letter/letter.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Letter {

    public static final int OPTIMAL_GUESSES = 5;  // 设置最佳猜测次数为5
    public static final int ASCII_A = 65;  // 设置ASCII码中大写字母A的值为65
    public static final int ALL_LETTERS = 26;  // 设置字母总数为26

    private enum GAME_STATE {  // 创建游戏状态枚举
        STARTUP,  // 启动
        INIT,  // 初始化
        GUESSING,  // 猜测中
        RESULTS,  // 结果
        GAME_OVER  // 游戏结束
    }

    // Used for keyboard input
    private final Scanner kbScanner;  // 用于键盘输入的Scanner对象

    // Current game state
    private GAME_STATE gameState;  // 当前游戏状态

    // Players guess count;
    private int playerGuesses;  // 玩家猜测次数

    // Computers ascii code for a random letter between A..Z
    private int computersLetter;  // 计算机随机生成的字母的ASCII码值

    public Letter() {

        gameState = GAME_STATE.STARTUP;  // 初始化游戏状态为启动

        // Initialise kb scanner
        kbScanner = new Scanner(System.in);  // 初始化键盘输入的Scanner对象
    }

    /**
     * Main game loop
     */
    }

    public void intro() {
        System.out.println(simulateTabs(33) + "LETTER");  // 输出游戏标题
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 输出创意计算机的信息
        System.out.println();
        System.out.println("LETTER GUESSING GAME");  // 输出游戏类型
        System.out.println();
        System.out.println("I'LL THINK OF A LETTER OF THE ALPHABET, A TO Z.");  // 输出游戏规则
        System.out.println("TRY TO GUESS MY LETTER AND I'LL GIVE YOU CLUES");
        System.out.println("AS TO HOW CLOSE YOU'RE GETTING TO MY LETTER.");
    }

    /**
     * Simulate the old basic tab(xx) command which indented text by xx spaces.
     *
     * @param spaces number of spaces required
     * @return String with number of spaces
     */
    # 创建一个由指定数量空格组成的字符串
    private String simulateTabs(int spaces) {
        char[] spacesTemp = new char[spaces];
        Arrays.fill(spacesTemp, ' ');
        return new String(spacesTemp);
    }

    '''
     * 在屏幕上打印一条消息，然后接受键盘输入。
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     '''
    # 显示文本消息并获取键盘输入
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next();
    }
# 闭合前面的函数定义
```