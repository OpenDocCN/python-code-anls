# `basic-computer-games\92_Trap\java\src\Trap.java`

```py
// 导入必要的类
import java.util.Arrays;
import java.util.Scanner;

/**
 * Game of Trap
 * <p>
 * 基于这里的基本陷阱游戏
 * https://github.com/coding-horror/basic-computer-games/blob/main/92%20Trap/trap.bas
 * <p>
 * 注意：这个想法是在Java中创建一个1970年代Basic游戏的版本，而不引入新功能-没有添加额外的文本，错误检查等。
 */
public class Trap {

    public static final int HIGH_NUMBER_RANGE = 100; // 定义高数值范围
    public static final int MAX_GUESSES = 6; // 定义最大猜测次数

    private enum GAME_STATE { // 定义游戏状态枚举
        STARTING, // 开始
        START_GAME, // 开始游戏
        GUESSING, // 猜测中
        PLAY_AGAIN, // 再玩一次
        GAME_OVER // 游戏结束
    }

    // 用于键盘输入
    private final Scanner kbScanner; // 键盘扫描器

    // 当前游戏状态
    private GAME_STATE gameState; // 游戏状态

    // 玩家的猜测次数
    private int currentPlayersGuess; // 当前玩家的猜测次数

    // 计算机的随机数
    private int computersNumber; // 计算机的随机数

    public Trap() {

        gameState = GAME_STATE.STARTING; // 初始化游戏状态为开始

        // 初始化键盘扫描器
        kbScanner = new Scanner(System.in);
    }

    /**
     * 主游戏循环
     */
    }

    /**
     * 显示玩家的猜测结果
     *
     * @param start 玩家输入的起始范围
     * @param end   结束范围
     * @return 表示他们进度的文本。
     */
    private String showGuessResult(int start, int end) {

        String status; // 状态文本
        if (start <= computersNumber && computersNumber <= end) { // 如果计算机的数字在范围内
            status = "YOU HAVE TRAPPED MY NUMBER."; // 你已经困住了我的数字
        } else if (computersNumber < start) { // 如果计算机的数字小于起始范围
            status = "MY NUMBER IS SMALLER THAN YOUR TRAP NUMBERS."; // 我的数字比你的陷阱数字小
        } else { // 否则
            status = "MY NUMBER IS LARGER THAN YOUR TRAP NUMBERS."; // 我的数字比你的陷阱数字大
        }

        return status; // 返回状态文本
    }
    // 显示游戏说明
    private void instructions() {
        System.out.println("I AM THINKING OF A NUMBER BETWEEN 1 AND " + HIGH_NUMBER_RANGE);
        System.out.println("TRY TO GUESS MY NUMBER. ON EACH GUESS,");
        System.out.println("YOU ARE TO ENTER 2 NUMBERS, TRYING TO TRAP");
        System.out.println("MY NUMBER BETWEEN THE TWO NUMBERS. I WILL");
        System.out.println("TELL YOU IF YOU HAVE TRAPPED MY NUMBER, IF MY");
        System.out.println("NUMBER IS LARGER THAN YOUR TWO NUMBERS, OR IF");
        System.out.println("MY NUMBER IS SMALLER THAN YOUR TWO NUMBERS.");
        System.out.println("IF YOU WANT TO GUESS ONE SINGLE NUMBER, TYPE");
        System.out.println("YOUR GUESS FOR BOTH YOUR TRAP NUMBERS.");
        System.out.println("YOU GET " + MAX_GUESSES + " GUESSES TO GET MY NUMBER.");
    }

    // 显示游戏介绍
    private void intro() {
        System.out.println("TRAP");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println();
    }

    /**
     * 根据指定分隔符返回字符串中的第n个值（从0开始计数）。
     *
     * @param text - 用逗号分隔的值的文本
     * @param pos  - 要返回值的位置
     * @return 值的整数表示
     */
    private int getDelimitedValue(String text, int pos) {
        String[] tokens = text.split(",");
        return Integer.parseInt(tokens[pos]);
    }

    /**
     * 检查玩家是否输入了Y或YES作为答案。
     *
     * @param text 玩家从键盘输入的字符串
     * @return 如果输入了Y或YES，则返回true，否则返回false
     */
    private boolean yesEntered(String text) {
        return stringIsAnyValue(text, "Y", "YES");
    }
    /**
     * 检查字符串是否等于多个变量值中的一个
     * 用于检查例如 Y 或 YES 等情况
     * 比较不区分大小写
     *
     * @param text   源字符串
     * @param values 要与源字符串进行比较的一系列值
     * @return 如果在传递的多个字符串中找到了匹配，则返回 true
     */
    private boolean stringIsAnyValue(String text, String... values) {

        return Arrays.stream(values).anyMatch(str -> str.equalsIgnoreCase(text));
    }

    /*
     * 在屏幕上打印消息，然后从键盘接受输入。
     *
     * @param text 要在屏幕上显示的消息。
     * @return 玩家输入的内容。
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next();
    }

    /**
     * 生成随机数
     * 用作计算机玩家的单个数字
     *
     * @return 随机数
     */
    private int randomNumber() {
        return (int) (Math.random()
                * (HIGH_NUMBER_RANGE) + 1);
    }
# 闭合前面的函数定义
```