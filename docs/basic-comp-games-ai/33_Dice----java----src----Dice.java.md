# `basic-computer-games\33_Dice\java\src\Dice.java`

```
# 导入必要的类库
import java.util.Arrays;
import java.util.Scanner;

/**
 * Game of Dice
 * <p>
 * Based on the Basic game of Dice here
 * https://github.com/coding-horror/basic-computer-games/blob/main/33%20Dice/dice.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Dice {

    # 用于键盘输入
    private final Scanner kbScanner;

    # 定义游戏状态枚举
    private enum GAME_STATE {
        START_GAME,
        INPUT_AND_CALCULATE,
        RESULTS,
        GAME_OVER
    }

    # 当前游戏状态
    private GAME_STATE gameState;

    # 骰子点数数组
    private int[] spots;

    # 构造函数，初始化键盘输入和游戏状态
    public Dice() {
        kbScanner = new Scanner(System.in);

        gameState = GAME_STATE.START_GAME;
    }

    /**
     * Main game loop
     */
    // 定义游戏进行的方法
    public void play() {

        // 循环执行游戏直到游戏状态为 GAME_OVER
        do {
            // 根据游戏状态进行不同的操作
            switch (gameState) {

                // 游戏开始状态
                case START_GAME:
                    // 执行游戏介绍
                    intro();
                    // 初始化长度为12的整型数组
                    spots = new int[12];
                    // 将游戏状态设置为输入和计算状态
                    gameState = GAME_STATE.INPUT_AND_CALCULATE;
                    break;

                // 输入和计算状态
                case INPUT_AND_CALCULATE:

                    // 获取用户输入的掷骰子次数
                    int howManyRolls = displayTextAndGetNumber("HOW MANY ROLLS? ");
                    // 循环执行掷骰子操作
                    for (int i = 0; i < howManyRolls; i++) {
                        // 生成两个骰子的点数之和
                        int diceRoll = (int) (Math.random() * 6 + 1) + (int) (Math.random() * 6 + 1);
                        // 将骰子点数保存在以0为基础的数组中
                        spots[diceRoll - 1]++;
                    }
                    // 将游戏状态设置为结果状态
                    gameState = GAME_STATE.RESULTS;
                    break;

                // 结果状态
                case RESULTS:
                    // 打印输出总点数和次数的表头
                    System.out.println("TOTAL SPOTS" + simulateTabs(8) + "NUMBER OF TIMES");
                    // 循环打印每个点数和对应的次数
                    for (int i = 1; i < 12; i++) {
                        // 使用以0为基础的数组展示输出
                        System.out.println(simulateTabs(5) + (i + 1) + simulateTabs(20) + spots[i]);
                    }
                    System.out.println();
                    // 如果用户输入的是"YES"，则将游戏状态设置为开始游戏状态，否则设置为游戏结束状态
                    if (yesEntered(displayTextAndGetInput("TRY AGAIN? "))) {
                        gameState = GAME_STATE.START_GAME;
                    } else {
                        gameState = GAME_STATE.GAME_OVER;
                    }
                    break;
            }
        } while (gameState != GAME_STATE.GAME_OVER);
    }
    // 打印程序介绍信息
    private void intro() {
        System.out.println(simulateTabs(34) + "DICE");
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("THIS PROGRAM SIMULATES THE ROLLING OF A");
        System.out.println("PAIR OF DICE.");
        System.out.println("YOU ENTER THE NUMBER OF TIMES YOU WANT THE COMPUTER TO");
        System.out.println("'ROLL' THE DICE.  WATCH OUT, VERY LARGE NUMBERS TAKE");
        System.out.println("A LONG TIME.  IN PARTICULAR, NUMBERS OVER 5000.");
    }

    /*
     * 打印屏幕上的消息，然后从键盘接受输入。
     * 将输入转换为整数
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    private int displayTextAndGetNumber(String text) {
        return Integer.parseInt(displayTextAndGetInput(text));
    }

    /*
     * 打印屏幕上的消息，然后从键盘接受输入。
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next();
    }

    /**
     * 检查玩家是否输入了Y或YES作为答案。
     *
     * @param text 从键盘输入的字符串
     * @return 如果输入了Y或YES，则返回true，否则返回false
     */
    private boolean yesEntered(String text) {
        return stringIsAnyValue(text, "Y", "YES");
    }

    /**
     * 检查字符串是否等于一系列变量值中的一个
     * 用于检查是否输入了Y或YES等
     * 比较不区分大小写。
     *
     * @param text    源字符串
     * @param values  要与源字符串进行比较的一系列值
     * @return 如果在传递的一系列字符串中找到了匹配，则返回true
     */
    # 检查给定的文本是否与提供的值中的任何一个相等（忽略大小写）
    private boolean stringIsAnyValue(String text, String... values) {
        return Arrays.stream(values).anyMatch(str -> str.equalsIgnoreCase(text));
    }

    /**
     * 模拟旧的基本 tab(xx) 命令，通过 xx 个空格缩进文本。
     *
     * @param spaces 需要的空格数
     * @return 包含指定空格数的字符串
     */
    private String simulateTabs(int spaces) {
        char[] spacesTemp = new char[spaces];
        Arrays.fill(spacesTemp, ' ');  # 用空格填充字符数组
        return new String(spacesTemp);  # 将字符数组转换为字符串并返回
    }
# 闭合前面的函数定义
```