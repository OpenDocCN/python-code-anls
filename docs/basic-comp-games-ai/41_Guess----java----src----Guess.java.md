# `41_Guess\java\src\Guess.java`

```
import java.util.Arrays;
import java.util.Scanner;

/**
 * Game of Guess
 * <p>
 * Based on the Basic game of Guess here
 * https://github.com/coding-horror/basic-computer-games/blob/main/41%20Guess/guess.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Guess {

    // Used for keyboard input
    private final Scanner kbScanner;  // 创建一个Scanner对象用于键盘输入

    private enum GAME_STATE {
        STARTUP,  // 定义游戏状态枚举，包括STARTUP
        INPUT_RANGE,  // 定义游戏状态枚举，包括INPUT_RANGE
        DEFINE_COMPUTERS_NUMBER,  // 定义计算机生成数字的状态
        GUESS,  // 猜测数字的状态
        GAME_OVER  // 游戏结束的状态
    }

    // 当前游戏状态
    private GAME_STATE gameState;

    // 用户提供的最大猜测数字
    private int limit;

    // 计算机为玩家猜测的数字
    private int computersNumber;

    // 玩家猜测的轮数
    private int tries;

    // 理论上猜测所需的最佳轮数
    private int calculatedTurns;
    public Guess() {
        kbScanner = new Scanner(System.in);  # 创建一个用于从键盘输入的 Scanner 对象

        gameState = GAME_STATE.STARTUP;  # 将游戏状态设置为启动状态
    }

    /**
     * Main game loop
     */
    public void play() {

        do {
            switch (gameState) {

                case STARTUP:
                    intro();  # 调用 intro() 方法，显示游戏介绍
                    gameState = GAME_STATE.INPUT_RANGE;  # 将游戏状态设置为输入范围状态
                    break;
                case INPUT_RANGE:  // 当用户选择输入范围时
                    limit = displayTextAndGetNumber("WHAT LIMIT DO YOU WANT? ");  // 获取用户输入的限制范围
                    calculatedTurns = (int) (Math.log(limit) / Math.log(2)) + 1;  // 根据限制范围计算出游戏的轮数
                    gameState = GAME_STATE.DEFINE_COMPUTERS_NUMBER;  // 将游戏状态设置为定义计算机的数字
                    break;  // 结束当前的 case

                case DEFINE_COMPUTERS_NUMBER:  // 当用户选择定义计算机的数字时
                    tries = 1;  // 尝试次数设置为1
                    System.out.println("I'M THINKING OF A NUMBER BETWEEN 1 AND " + limit);  // 打印提示信息，计算机正在思考一个1到限制范围内的数字
                    computersNumber = (int) (Math.random() * limit + 1);  // 生成计算机的随机数字
                    gameState = GAME_STATE.GUESS;  // 将游戏状态设置为猜测
                    break;  // 结束当前的 case

                case GUESS:  // 当用户选择猜测时
                    int playersGuess = displayTextAndGetNumber("NOW YOU TRY TO GUESS WHAT IT IS ");  // 获取玩家猜测的数字

                    // Allow player to restart game with entry of 0
                    // 允许玩家通过输入0来重新开始游戏
                    if (playersGuess == 0) {  // 如果玩家猜测为0
                        linePadding();  // 调用函数增加空行
                        gameState = GAME_STATE.STARTUP;  // 将游戏状态设置为开始状态
                        break;  // 跳出循环
                    }

                    if (playersGuess == computersNumber) {  // 如果玩家猜测与计算机数字相等
                        System.out.println("THAT'S IT! YOU GOT IT IN " + tries + " TRIES.");  // 打印玩家猜中的消息和尝试次数
                        if (tries < calculatedTurns) {  // 如果尝试次数小于计算出的最佳次数
                            System.out.println("VERY ");  // 打印"VERY"
                        }
                        System.out.println("GOOD.");  // 打印"GOOD."
                        System.out.println("YOU SHOULD HAVE BEEN ABLE TO GET IT IN ONLY " + calculatedTurns);  // 打印提示玩家应该在多少次内猜中
                        linePadding();  // 调用函数增加空行
                        gameState = GAME_STATE.DEFINE_COMPUTERS_NUMBER;  // 将游戏状态设置为定义计算机数字
                        break;  // 跳出循环
                    } else if (playersGuess < computersNumber) {  // 如果玩家猜测小于计算机数字
                        System.out.println("TOO LOW. TRY A BIGGER ANSWER.");  // 打印提示猜测太小，尝试更大的答案
                    } else {  // 如果玩家猜测大于计算机数字
                        System.out.println("TOO HIGH. TRY A SMALLER ANSWER.");  // 打印提示猜测太大，尝试更小的答案
                    }
                    tries++;  # 增加尝试次数
                    break;  # 跳出循环
            }
        } while (gameState != GAME_STATE.GAME_OVER);  # 当游戏状态不是游戏结束时继续循环
    }

    private void intro() {  # 定义 intro 方法
        System.out.println(simulateTabs(33) + "GUESS");  # 打印 "GUESS"，并使用 simulateTabs 方法模拟缩进
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  # 打印 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，并使用 simulateTabs 方法模拟缩进
        System.out.println();  # 打印空行
        System.out.println("THIS IS A NUMBER GUESSING GAME. I'LL THINK");  # 打印游戏介绍信息
        System.out.println("OF A NUMBER BETWEEN 1 AND ANY LIMIT YOU WANT.");  # 打印游戏介绍信息
        System.out.println("THEN YOU HAVE TO GUESS WHAT IT IS.");  # 打印游戏介绍信息
    }

    /**
     * Print a predefined number of blank lines
     *
     */
    private void linePadding() {
        for (int i = 1; i <= 5; i++) {
            System.out.println(); // 打印空行
        }
    }

    /*
     * 在屏幕上打印一条消息，然后从键盘接受输入。
     * 将输入转换为整数
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    private int displayTextAndGetNumber(String text) {
        return Integer.parseInt(displayTextAndGetInput(text)); // 调用displayTextAndGetInput方法获取输入并将其转换为整数
    }

    /*
     * 在屏幕上打印一条消息，然后从键盘接受输入。
     *
    /**
     * displayTextAndGetInput方法用于在屏幕上显示文本，并获取玩家输入的内容。
     * @param text 要在屏幕上显示的消息。
     * @return 玩家输入的内容。
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text); // 在屏幕上打印文本消息
        return kbScanner.next(); // 获取玩家输入的内容并返回
    }

    /**
     * simulateTabs方法用于模拟旧的基本tab(xx)命令，该命令将文本缩进xx个空格。
     *
     * @param spaces 需要的空格数
     * @return 包含指定数量空格的字符串
     */
    private String simulateTabs(int spaces) {
        char[] spacesTemp = new char[spaces]; // 创建一个包含指定数量空格的字符数组
        Arrays.fill(spacesTemp, ' '); // 使用空格填充字符数组
        return new String(spacesTemp); // 将字符数组转换为字符串并返回
    }
# 关闭 ZIP 对象
zip.close()  # 关闭 ZIP 对象，释放资源，避免内存泄漏。
```