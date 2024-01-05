# `54_Letter\java\src\Letter.java`

```
import java.awt.*;  # 导入 java.awt 包
import java.util.Arrays;  # 导入 java.util.Arrays 包
import java.util.Scanner;  # 导入 java.util.Scanner 包

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

    public static final int OPTIMAL_GUESSES = 5;  # 定义常量 OPTIMAL_GUESSES，值为 5
    public static final int ASCII_A = 65;  # 定义常量 ASCII_A，值为 65
    public static final int ALL_LETTERS = 26;  # 定义常量 ALL_LETTERS，值为 26

    private enum GAME_STATE {  # 定义枚举类型 GAME_STATE
        STARTUP,  // 定义游戏的起始状态
        INIT,  // 定义游戏的初始化状态
        GUESSING,  // 定义游戏的猜测状态
        RESULTS,  // 定义游戏的结果状态
        GAME_OVER  // 定义游戏结束状态
    }

    // 用于键盘输入
    private final Scanner kbScanner;

    // 当前游戏状态
    private GAME_STATE gameState;

    // 玩家猜测次数
    private int playerGuesses;

    // 计算机随机生成的字母的ASCII码值，范围在A到Z之间
    private int computersLetter;

    public Letter() {
        gameState = GAME_STATE.STARTUP;  // 设置游戏状态为启动状态

        // 初始化键盘输入扫描器
        kbScanner = new Scanner(System.in);
    }

    /**
     * Main game loop
     */
    public void play() {

        do {
            switch (gameState) {

                // 第一次玩游戏时显示介绍
                case STARTUP:
                    intro();  // 调用介绍方法
                    gameState = GAME_STATE.INIT;  // 设置游戏状态为初始化状态
                    break;
                // 初始化游戏状态
                case INIT:
                    // 重置玩家猜测次数
                    playerGuesses = 0;
                    // 生成计算机随机字母的 ASCII 码
                    computersLetter = ASCII_A + (int) (Math.random() * ALL_LETTERS);
                    // 打印消息，提示玩家开始猜测
                    System.out.println("O.K., I HAVE A LETTER.  START GUESSING.");
                    // 将游戏状态设置为猜测中
                    gameState = GAME_STATE.GUESSING;
                    break;

                // 玩家猜测字母，直到猜中或用尽所有猜测次数
                case GUESSING:
                    // 获取玩家输入的猜测字母并转换为大写
                    String playerGuess = displayTextAndGetInput("WHAT IS YOUR GUESS? ").toUpperCase();

                    // 将输入字符串的第一个字符转换为 ASCII 码
                    int toAscii = playerGuess.charAt(0);
                    // 增加玩家猜测次数
                    playerGuesses++;
                    // 如果玩家猜中了计算机随机字母的 ASCII 码
                    if (toAscii == computersLetter) {
                        // 将游戏状态设置为结果
                        gameState = GAME_STATE.RESULTS;
                        break;
                    }
                    if (toAscii > computersLetter) {  // 如果玩家猜测的字母的ASCII码大于计算机生成的字母的ASCII码
                        System.out.println("TOO HIGH.  TRY A LOWER LETTER.");  // 打印提示信息，要求玩家尝试一个更小的字母
                    } else {
                        System.out.println("TOO LOW.  TRY A HIGHER LETTER.");  // 打印提示信息，要求玩家尝试一个更大的字母
                    }
                    break;  // 结束当前的switch语句

                // Play again, or exit game?
                case RESULTS:  // 如果当前状态是RESULTS
                    System.out.println();  // 打印空行
                    System.out.println("YOU GOT IT IN " + playerGuesses + " GUESSES!!");  // 打印玩家猜测的次数
                    if (playerGuesses <= OPTIMAL_GUESSES) {  // 如果玩家猜测的次数小于等于最佳猜测次数
                        System.out.println("GOOD JOB !!!!!");  // 打印祝贺信息
                        // Original game beeped 15 tims if you guessed in the optimal guesses or less
                        // Changed this to do a single beep only
                        Toolkit.getDefaultToolkit().beep();  // 使用默认工具包进行蜂鸣提示
                    } else {
                        // Took more than optimal number of guesses
                        System.out.println("BUT IT SHOULDN'T TAKE MORE THAN " + OPTIMAL_GUESSES + " GUESSES!");  // 打印提示信息，玩家猜测次数超过最佳猜测次数
                    }
                    System.out.println();  // 打印空行
                    System.out.println("LET'S PLAN AGAIN.....");  // 打印提示信息
                    gameState = GAME_STATE.INIT;  // 将游戏状态设置为初始化
                    break;  // 跳出循环
            }
        } while (gameState != GAME_STATE.GAME_OVER);  // 当游戏状态不是游戏结束时继续循环
    }

    public void intro() {
        System.out.println(simulateTabs(33) + "LETTER");  // 打印标题
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 打印信息
        System.out.println();  // 打印空行
        System.out.println("LETTER GUESSING GAME");  // 打印游戏标题
        System.out.println();  // 打印空行
        System.out.println("I'LL THINK OF A LETTER OF THE ALPHABET, A TO Z.");  // 打印游戏规则
        System.out.println("TRY TO GUESS MY LETTER AND I'LL GIVE YOU CLUES");  // 打印游戏规则
        System.out.println("AS TO HOW CLOSE YOU'RE GETTING TO MY LETTER.");  // 打印游戏规则
    }

    /**
    /**
     * 模拟旧的基本tab(xx)命令，通过xx个空格缩进文本。
     *
     * @param spaces 需要的空格数
     * @return 包含指定数量空格的字符串
     */
    private String simulateTabs(int spaces) {
        char[] spacesTemp = new char[spaces];
        Arrays.fill(spacesTemp, ' '); // 用空格填充字符数组
        return new String(spacesTemp); // 将字符数组转换为字符串并返回
    }

    /*
     * 在屏幕上打印消息，然后从键盘接受输入。
     *
     * @param text 要在屏幕上显示的消息。
     * @return 玩家输入的内容。
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text); // 打印消息
        return kbScanner.next(); // 从键盘接受输入并返回
    }
    }
```

这部分代码是一个缩进错误，应该删除。
```