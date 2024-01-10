# `basic-computer-games\41_Guess\java\src\Guess.java`

```
// 导入必要的类
import java.util.Arrays;
import java.util.Scanner;

/**
 * 猜数字游戏
 * <p>
 * 基于这里的基本猜数字游戏
 * https://github.com/coding-horror/basic-computer-games/blob/main/41%20Guess/guess.bas
 * <p>
 * 注意：这个想法是在Java中创建一个1970年代基本游戏的版本，没有引入新功能-没有添加额外的文本，错误检查等。
 */
public class Guess {

    // 用于键盘输入
    private final Scanner kbScanner;

    // 当前游戏状态
    private GAME_STATE gameState;

    // 用户提供的最大猜测数
    private int limit;

    // 计算机为玩家猜测的数字
    private int computersNumber;

    // 玩家猜测的轮数
    private int tries;

    // 猜测所需的最佳轮数
    private int calculatedTurns;

    public Guess() {
        // 初始化键盘输入扫描器
        kbScanner = new Scanner(System.in);
        // 设置游戏状态为启动状态
        gameState = GAME_STATE.STARTUP;
    }

    /**
     * 主游戏循环
     */
    }

    // 游戏介绍
    private void intro() {
        System.out.println(simulateTabs(33) + "GUESS");
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("THIS IS A NUMBER GUESSING GAME. I'LL THINK");
        System.out.println("OF A NUMBER BETWEEN 1 AND ANY LIMIT YOU WANT.");
        System.out.println("THEN YOU HAVE TO GUESS WHAT IT IS.");
    }

    /**
     * 打印预定义数量的空行
     */
    private void linePadding() {
        for (int i = 1; i <= 5; i++) {
            System.out.println();
        }
    }
    /*
     * 在屏幕上打印一条消息，然后接受键盘输入。
     * 将输入转换为整数
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    private int displayTextAndGetNumber(String text) {
        return Integer.parseInt(displayTextAndGetInput(text));
    }

    /*
     * 在屏幕上打印一条消息，然后接受键盘输入。
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next();
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
# 闭合前面的函数定义
```