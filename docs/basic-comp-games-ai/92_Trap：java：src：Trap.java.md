# `d:/src/tocomm/basic-computer-games\92_Trap\java\src\Trap.java`

```
import java.util.Arrays;  // 导入 Arrays 类，用于操作数组
import java.util.Scanner;  // 导入 Scanner 类，用于接收用户输入

/**
 * Game of Trap
 * <p>
 * Based on the Basic game of Trap here
 * https://github.com/coding-horror/basic-computer-games/blob/main/92%20Trap/trap.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Trap {

    public static final int HIGH_NUMBER_RANGE = 100;  // 定义常量 HIGH_NUMBER_RANGE，表示数字范围为 100
    public static final int MAX_GUESSES = 6;  // 定义常量 MAX_GUESSES，表示最大猜测次数为 6

    private enum GAME_STATE {  // 定义枚举类型 GAME_STATE，表示游戏状态
        STARTING,  // 初始状态
        START_GAME,  // 游戏开始状态
        GUESSING,  // 定义游戏状态为猜测中
        PLAY_AGAIN,  // 定义游戏状态为再玩一次
        GAME_OVER  // 定义游戏状态为游戏结束
    }

    // 用于键盘输入
    private final Scanner kbScanner;

    // 当前游戏状态
    private GAME_STATE gameState;

    // 玩家的猜测次数
    private int currentPlayersGuess;

    // 计算机生成的随机数
    private int computersNumber;

    public Trap() {
        // 初始化游戏状态为开始
        gameState = GAME_STATE.STARTING;
        // 初始化键盘扫描器
        kbScanner = new Scanner(System.in);
    }

    /**
     * 主游戏循环
     */
    public void play() {

        do {
            switch (gameState) {

                // 第一次玩游戏时显示介绍和可选的说明。
                case STARTING:
                    intro();
                    // 如果输入yes，则显示游戏说明
                    if (yesEntered(displayTextAndGetInput("INSTRUCTIONS? "))) {
                        instructions();
                    }
                    // 将游戏状态设置为开始游戏
                    gameState = GAME_STATE.START_GAME;
                    break;  # 结束当前的 case，跳出 switch 语句

                // 开始新游戏
                case START_GAME:
                    computersNumber = randomNumber();  # 生成一个随机数作为电脑的数字
                    currentPlayersGuess = 1;  # 初始化玩家猜测次数
                    gameState = GAME_STATE.GUESSING;  # 设置游戏状态为猜数字中
                    break;  # 结束当前的 case，跳出 switch 语句

                // 玩家猜数字，直到猜中或者用完所有的猜测次数
                case GUESSING:
                    System.out.println();  # 输出空行
                    String playerRangeGuess = displayTextAndGetInput("GUESS # " + currentPlayersGuess + "? ");  # 获取玩家猜测的范围
                    int startRange = getDelimitedValue(playerRangeGuess, 0);  # 获取玩家猜测范围的起始值
                    int endRange = getDelimitedValue(playerRangeGuess, 1);  # 获取玩家猜测范围的结束值

                    // 玩家是否猜中了？
                    if (startRange == computersNumber && endRange == computersNumber) {  # 判断玩家猜测的范围是否等于电脑的数字
                        System.out.println("YOU GOT IT!!!");  # 输出玩家猜中了的提示
                        System.out.println();  # 输出空行
                    gameState = GAME_STATE.PLAY_AGAIN;  # 设置游戏状态为再玩一次

                    # 如果猜测错误，显示猜测的位置，并增加当前玩家的猜测次数
                    System.out.println(showGuessResult(startRange, endRange));
                    currentPlayersGuess++;
                    # 如果当前玩家的猜测次数超过最大猜测次数，显示正确答案，并设置游戏状态为再玩一次
                    if (currentPlayersGuess > MAX_GUESSES) {
                        System.out.println("SORRY, THAT'S " + MAX_GUESSES + " GUESSES. THE NUMBER WAS "
                                + computersNumber);
                        gameState = GAME_STATE.PLAY_AGAIN;
                    }
                    break;

                // 再玩一次，或退出游戏？
                case PLAY_AGAIN:
                    System.out.println("TRY AGAIN");  # 显示再试一次的提示
                    gameState = GAME_STATE.START_GAME;  # 设置游戏状态为开始游戏
            }
        } while (gameState != GAME_STATE.GAME_OVER);  # 当游戏状态不是游戏结束时继续循环
    }
    /**
     * 展示玩家猜测的结果
     *
     * @param start 玩家输入的起始范围
     * @param end   结束范围
     * @return 表示他们进展的文本。
     */
    private String showGuessResult(int start, int end) {

        String status;
        if (start <= computersNumber && computersNumber <= end) {
            status = "YOU HAVE TRAPPED MY NUMBER.";  // 如果计算机的数字在玩家输入的范围内，则返回这个文本
        } else if (computersNumber < start) {
            status = "MY NUMBER IS SMALLER THAN YOUR TRAP NUMBERS.";  // 如果计算机的数字小于玩家输入的起始范围，则返回这个文本
        } else {
            status = "MY NUMBER IS LARGER THAN YOUR TRAP NUMBERS.";  // 如果计算机的数字大于玩家输入的结束范围，则返回这个文本
        }

        return status;  // 返回结果文本
    // 打印游戏的玩法说明
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

    // 打印游戏介绍
    private void intro() {
        System.out.println("TRAP");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println();
    }
    }

    /**
     * 接受一个由逗号分隔的字符串，并返回第n个被分隔的值（从计数0开始）。
     *
     * @param text - 由逗号分隔的值的字符串
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
    // 检查输入的文本是否等于给定的值之一
    private boolean yesEntered(String text) {
        return stringIsAnyValue(text, "Y", "YES");
    }

    /**
     * 检查字符串是否等于一系列变量值中的一个
     * 用于检查例如 Y 或 YES
     * 比较是不区分大小写的
     *
     * @param text   源字符串
     * @param values 要与源字符串进行比较的一系列值
     * @return 如果在传递的一系列字符串中找到了比较，则返回true
     */
    // 检查字符串是否等于一系列变量值中的一个
    private boolean stringIsAnyValue(String text, String... values) {
        return Arrays.stream(values).anyMatch(str -> str.equalsIgnoreCase(text));
    }

    /*
# 在屏幕上打印一条消息，然后接受键盘输入。
# @param text 要显示在屏幕上的消息。
# @return 玩家输入的内容。
private String displayTextAndGetInput(String text) {
    System.out.print(text);  # 打印消息到屏幕
    return kbScanner.next();  # 从键盘接受输入并返回
}

/**
 * 生成随机数
 * 用作计算机玩家的单个数字
 *
 * @return 随机数
 */
private int randomNumber() {
    return (int) (Math.random() * (HIGH_NUMBER_RANGE) + 1);  # 生成一个介于1和HIGH_NUMBER_RANGE之间的随机整数并返回
}
# 关闭 ZIP 对象
zip.close()  # 关闭 ZIP 对象，释放资源
```