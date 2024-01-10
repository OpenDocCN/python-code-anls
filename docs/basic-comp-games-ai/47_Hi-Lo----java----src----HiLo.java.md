# `basic-computer-games\47_Hi-Lo\java\src\HiLo.java`

```
import java.util.Scanner;

/**
 * Game of HiLo
 *
 * Based on the Basic game of Hi-Lo here
 * https://github.com/coding-horror/basic-computer-games/blob/main/47%20Hi-Lo/hi-lo.bas
 *
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 *        new features - no additional text, error checking, etc has been added.
 */
public class HiLo {

    public static final int LOW_NUMBER_RANGE = 1;  // 定义最小数值范围
    public static final int HIGH_NUMBER_RANGE = 100;  // 定义最大数值范围
    public static final int MAX_GUESSES = 6;  // 定义最大猜测次数

    private enum GAME_STATE {
        STARTING,  // 游戏状态：开始
        START_GAME,  // 游戏状态：开始游戏
        GUESSING,  // 游戏状态：猜测中
        PLAY_AGAIN,  // 游戏状态：再玩一次
        GAME_OVER  // 游戏状态：游戏结束
    }

    // Used for keyboard input
    private final Scanner kbScanner;  // 用于键盘输入的扫描器

    // Current game state
    private GAME_STATE gameState;  // 当前游戏状态

    // Players Winnings
    private int playerAmountWon;  // 玩家赢得的奖金

    // Players guess count;
    private int playersGuesses;  // 玩家猜测次数

    // Computers random number
    private int computersNumber;  // 计算机生成的随机数

    public HiLo() {

        gameState = GAME_STATE.STARTING;  // 初始化游戏状态为开始
        playerAmountWon = 0;  // 初始化玩家奖金为0

        // Initialise kb scanner
        kbScanner = new Scanner(System.in);  // 初始化键盘扫描器
    }

    /**
     * Main game loop
     *
     */
    }

    /**
     * Checks the players guess against the computers randomly generated number
     *
     * @param theGuess the players guess
     * @return true if the player guessed correctly, false otherwise
     */
    private boolean validateGuess(int theGuess) {

        // Correct guess?
        if(theGuess == computersNumber) {  // 如果玩家猜对了
            return true;  // 返回true
        }

        if(theGuess > computersNumber) {  // 如果玩家猜的数值大于计算机生成的随机数
            System.out.println("YOUR GUESS IS TOO HIGH.");  // 输出提示信息：你的猜测太高
        } else {
            System.out.println("YOUR GUESS IS TOO LOW.");  // 输出提示信息：你的猜测太低
        }

        return false;  // 返回false
    }

    private void init() {
        playersGuesses = 0;  // 初始化玩家猜测次数为0
        computersNumber = randomNumber();  // 生成计算机的随机数
    }
}
    // 打印游戏介绍信息
    public void intro() {
        System.out.println("HI LO");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println();
        System.out.println("IS THE GAME OF HI LO.");
        System.out.println();
        System.out.println("YOU WILL HAVE 6 TRIES TO GUESS THE AMOUNT OF MONEY IN THE");
        System.out.println("HI LO JACKPOT, WHICH IS BETWEEN 1 AND 100 DOLLARS.  IF YOU");
        System.out.println("GUESS THE AMOUNT, YOU WIN ALL THE MONEY IN THE JACKPOT!");
        System.out.println("THEN YOU GET ANOTHER CHANCE TO WIN MORE MONEY.  HOWEVER,");
        System.out.println("IF YOU DO NOT GUESS THE AMOUNT, THE GAME ENDS.");
    }

    /**
     * 从键盘获取玩家的猜测
     *
     * @return 玩家的猜测作为整数
     */
    private int playerGuess() {
        return Integer.parseInt((displayTextAndGetInput("YOUR GUESS? ")));
    }

    /**
     * 检查玩家是否输入了Y或YES
     *
     * @param text 从键盘获取的玩家字符串
     * @return 如果输入了Y或YES，则返回true，否则返回false
     */
    private boolean yesEntered(String text) {
        return stringIsAnyValue(text, "Y", "YES");
    }

    /**
     * 检查字符串是否等于一系列变量值之一
     * 用于检查是否输入了Y或YES等
     * 比较不区分大小写
     *
     * @param text 源字符串
     * @param values 要与源字符串进行比较的一系列值
     * @return 如果在传递的一系列字符串中找到了匹配项，则返回true
     */
    private boolean stringIsAnyValue(String text, String... values) {

        // 循环遍历一系列值并逐个测试
        for(String val:values) {
            if(text.equalsIgnoreCase(val)) {
                return true;
            }
        }

        // 没有匹配项
        return false;
    }
    /*
     * 在屏幕上打印一条消息，然后接受键盘输入。
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);  // 在屏幕上打印消息
        return kbScanner.next();  // 接受键盘输入并返回
    }

    /**
     * 生成随机数
     * 用作计算机玩家的单个数字
     *
     * @return 随机数
     */
    private int randomNumber() {
        return (int) (Math.random()  // 生成一个0到1之间的随机小数
                * (HIGH_NUMBER_RANGE - LOW_NUMBER_RANGE + 1) + LOW_NUMBER_RANGE);  // 将随机小数转换为指定范围内的随机整数
    }
# 闭合前面的函数定义
```