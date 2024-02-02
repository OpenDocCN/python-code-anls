# `basic-computer-games\80_Slots\java\src\Slots.java`

```py
import java.util.Arrays;  // 导入 Arrays 类
import java.util.Scanner;  // 导入 Scanner 类

/**
 * Slots 游戏类
 * <p>
 * 基于这里的基本 Slots 游戏
 * https://github.com/coding-horror/basic-computer-games/blob/main/80%20Slots/slots.bas
 * <p>
 * 注意：这个想法是在 Java 中创建一个 1970 年代 Basic 游戏的版本，没有引入新功能 - 没有添加额外的文本、错误检查等。
 */
public class Slots {

    public static final String[] SLOT_SYMBOLS = {"BAR", "BELL", "ORANGE", "LEMON", "PLUM", "CHERRY"};  // 定义 Slot 的符号数组

    public static final int NUMBER_SYMBOLS = SLOT_SYMBOLS.length;  // 计算符号的数量

    // Jackpot 符号 (BAR)
    public static final int BAR_SYMBOL = 0;  // 定义 Jackpot 符号的索引

    // 表示当前旋转没有赢得任何奖励
    public static final int NO_WINNER = -1;  // 定义没有赢得奖励的标识

    // 用于键盘输入
    private final Scanner kbScanner;  // 创建 Scanner 对象用于键盘输入

    private enum GAME_STATE {  // 定义游戏状态枚举
        START_GAME,  // 开始游戏
        ONE_SPIN,  // 一次旋转
        RESULTS,  // 结果
        GAME_OVER  // 游戏结束
    }

    // 当前游戏状态
    private GAME_STATE gameState;  // 定义当前游戏状态变量

    // 不同类型的旋转结果
    private enum WINNINGS {  // 定义赢得奖励的枚举
        JACKPOT(100),  // 大奖 (100)
        TOP_DOLLAR(10),  // 最高奖励 (10)
        DOUBLE_BAR(5),  // 双 BAR (5)
        REGULAR(2),  // 普通奖励 (2)
        NO_WIN(0);  // 没有赢得奖励 (0)

        private final int multiplier;  // 定义奖励的倍数

        WINNINGS(int mult) {  // 构造函数
            multiplier = mult;  // 初始化奖励的倍数
        }

        // 没有赢得奖励返回净额的负数
        // 否则根据倍数计算奖励
        public int calculateWinnings(int bet) {  // 计算赢得奖励的方法

            if (multiplier == 0) {  // 如果倍数为 0
                return -bet;  // 返回负的赌注金额
            } else {
                // 返回原始赌注加上奖励类型的倍数
                return (multiplier * bet) + bet;  // 返回赌注金额乘以倍数再加上赌注金额
            }
        }
    }

    private int playerBalance;  // 玩家余额

    public Slots() {  // 构造函数

        kbScanner = new Scanner(System.in);  // 初始化键盘输入对象
        gameState = GAME_STATE.START_GAME;  // 初始化游戏状态为开始游戏
    }

    /**
     * 主游戏循环
     */
}
    // 打印介绍信息
    private void intro() {
        System.out.println(simulateTabs(30) + "SLOTS");
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("YOU ARE IN THE H&M CASINO,IN FRONT OF ONE OF OUR");
        System.out.println("ONE-ARM BANDITS. BET FROM $1 TO $100.");
        System.out.println("TO PULL THE ARM, PUNCH THE RETURN KEY AFTER MAKING YOUR BET.");
    }

    /*
     * 在屏幕上打印消息，然后从键盘接受输入。
     * 将输入转换为整数
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    private int displayTextAndGetNumber(String text) {
        return Integer.parseInt(displayTextAndGetInput(text));
    }

    /*
     * 在屏幕上打印消息，然后从键盘接受输入。
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
     * 检查字符串是否等于一系列变量值中的任意一个
     * 用于检查是否输入了Y或YES等情况
     * 比较不区分大小写。
     *
     * @param text    源字符串
     * @param values  要与源字符串进行比较的一系列值
     * @return 如果在传递的一系列字符串中找到了匹配，则返回true
     */
    private boolean stringIsAnyValue(String text, String... values) {
        return Arrays.stream(values).anyMatch(str -> str.equalsIgnoreCase(text));
    }
    /**
     * 模拟旧的基本tab(xx)命令，通过xx个空格缩进文本。
     *
     * @param spaces 需要的空格数
     * @return 包含指定数量空格的字符串
     */
    private String simulateTabs(int spaces) {
        char[] spacesTemp = new char[spaces];
        Arrays.fill(spacesTemp, ' ');
        return new String(spacesTemp);
    }

    /**
     * 找到在本轮中获胜的符号，即第一个与另一个轴匹配的轴
     *
     * @param reel1 轴1的旋转结果
     * @param reel2 轴2的旋转结果
     * @param reel3 轴3的旋转结果
     * @return 如果没有轴匹配，则返回NO_WINNER，否则返回0-2之间的整数，表示匹配的轴
     */
    private int winningSymbol(int reel1, int reel2, int reel3) {
        if (reel1 == reel2) {
            return 0;
        } else if (reel1 == reel3) {
            return 0;
        } else if (reel2 == reel3) {
            return 1;
        } else {
            return NO_WINNER;
        }
    }

    /**
     * 转轴的随机符号
     *
     * @return 0-5之间的数字
     */
    private int randomSymbol() {
        return (int) (Math.random() * NUMBER_SYMBOLS);
    }
# 闭合前面的函数定义
```