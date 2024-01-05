# `47_Hi-Lo\java\src\HiLo.java`

```
import java.util.Scanner; // 导入 Scanner 类，用于从控制台读取输入

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

    public static final int LOW_NUMBER_RANGE = 1; // 定义常量 LOW_NUMBER_RANGE，表示猜测的最小数字范围
    public static final int HIGH_NUMBER_RANGE = 100; // 定义常量 HIGH_NUMBER_RANGE，表示猜测的最大数字范围
    public static final int MAX_GUESSES = 6; // 定义常量 MAX_GUESSES，表示最大猜测次数

    private enum GAME_STATE { // 定义枚举类型 GAME_STATE，表示游戏的不同状态
        STARTING, // 初始状态
        START_GAME, // 游戏开始状态
        GUESSING,          // 游戏状态：猜测中
        PLAY_AGAIN,        // 游戏状态：再玩一次
        GAME_OVER          // 游戏状态：游戏结束
    }

    // 用于键盘输入
    private final Scanner kbScanner;

    // 当前游戏状态
    private GAME_STATE gameState;

    // 玩家赢得的金额
    private int playerAmountWon;

    // 玩家的猜测次数
    private int playersGuesses;

    // 计算机生成的随机数
    private int computersNumber;
    public HiLo() {
        // 初始化游戏状态为开始状态
        gameState = GAME_STATE.STARTING;
        // 初始化玩家赢得的金额为0
        playerAmountWon = 0;
        // 初始化键盘输入扫描器
        kbScanner = new Scanner(System.in);
    }

    /**
     * Main game loop
     *
     */
    public void play() {
        // 游戏主循环
        do {
            switch (gameState) {
                // 当游戏状态为开始状态时
                case STARTING:
// 初始化游戏状态为开始游戏
gameState = GAME_STATE.START_GAME;
break;

// 生成计算机要猜测的数字等
case START_GAME:
    init();
    System.out.println("O.K.  I HAVE A NUMBER IN MIND.");
    // 将游戏状态设置为猜数字
    gameState = GAME_STATE.GUESSING;
    break;

// 玩家猜数字，直到猜中或用尽所有机会
case GUESSING:
    int guess = playerGuess();

    // 检查玩家是否猜中了数字
    if(validateGuess(guess)) {
        System.out.println("GOT IT!!!!!!!!!!   YOU WIN " + computersNumber
                + " DOLLARS.");
        playerAmountWon += computersNumber;
                        // 打印玩家赢得的总金额
                        System.out.println("YOUR TOTAL WINNINGS ARE NOW "
                                + playerAmountWon + " DOLLARS.");
                        // 设置游戏状态为再玩一次
                        gameState = GAME_STATE.PLAY_AGAIN;
                    } else {
                        // 猜测错误
                        playersGuesses++;
                        // 猜测次数用完了吗？
                        if (playersGuesses == MAX_GUESSES) {
                            // 打印玩家猜错的消息和正确的数字
                            System.out.println("YOU BLEW IT...TOO BAD...THE NUMBER WAS "
                                    + computersNumber);
                            // 重置玩家赢得的金额为0
                            playerAmountWon = 0;
                            // 设置游戏状态为再玩一次
                            gameState = GAME_STATE.PLAY_AGAIN;
                        }
                    }
                    break;

                // 再玩一次，还是退出游戏？
                case PLAY_AGAIN:
                    // 打印空行
                    System.out.println();
                    // 如果玩家输入yes，则再玩一次
                    if(yesEntered(displayTextAndGetInput("PLAY AGAIN (YES OR NO) "))) {
                        gameState = GAME_STATE.START_GAME;  // 设置游戏状态为开始游戏
                    } else {
                        // Chose not to play again
                        System.out.println("SO LONG.  HOPE YOU ENJOYED YOURSELF!!!");  // 输出消息到控制台
                        gameState = GAME_STATE.GAME_OVER;  // 设置游戏状态为游戏结束
                    }
            }
        } while (gameState != GAME_STATE.GAME_OVER);  // 当游戏状态不是游戏结束时继续循环
    }

    /**
     * Checks the players guess against the computers randomly generated number
     *
     * @param theGuess the players guess  // 参数为玩家的猜测
     * @return true if the player guessed correctly, false otherwise  // 如果玩家猜对返回true，否则返回false
     */
    private boolean validateGuess(int theGuess) {

        // Correct guess?
        if(theGuess == computersNumber) {  // 如果玩家的猜测等于计算机生成的数字
            return true;  # 如果玩家猜对了计算机的数字，返回 true

        }

        if(theGuess > computersNumber) {  # 如果玩家的猜测大于计算机的数字
            System.out.println("YOUR GUESS IS TOO HIGH.");  # 打印出猜测过高的提示
        } else {  # 否则
            System.out.println("YOUR GUESS IS TOO LOW.");  # 打印出猜测过低的提示
        }

        return false;  # 返回 false，表示玩家猜错了

    }

    private void init() {  # 初始化方法
        playersGuesses = 0;  # 玩家的猜测次数设为 0
        computersNumber = randomNumber();  # 生成计算机的随机数字
    }

    public void intro() {  # 介绍方法
        System.out.println("HI LO");  # 打印出游戏名称
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  # 打印出游戏信息
        # 打印空行
        System.out.println();
        # 打印空行
        System.out.println();
        # 打印"IS THE GAME OF HI LO."
        System.out.println("IS THE GAME OF HI LO.");
        # 打印空行
        System.out.println();
        # 打印"YOU WILL HAVE 6 TRIES TO GUESS THE AMOUNT OF MONEY IN THE"
        System.out.println("YOU WILL HAVE 6 TRIES TO GUESS THE AMOUNT OF MONEY IN THE");
        # 打印"HI LO JACKPOT, WHICH IS BETWEEN 1 AND 100 DOLLARS.  IF YOU"
        System.out.println("HI LO JACKPOT, WHICH IS BETWEEN 1 AND 100 DOLLARS.  IF YOU");
        # 打印"GUESS THE AMOUNT, YOU WIN ALL THE MONEY IN THE JACKPOT!"
        System.out.println("GUESS THE AMOUNT, YOU WIN ALL THE MONEY IN THE JACKPOT!");
        # 打印"THEN YOU GET ANOTHER CHANCE TO WIN MORE MONEY.  HOWEVER,"
        System.out.println("THEN YOU GET ANOTHER CHANCE TO WIN MORE MONEY.  HOWEVER,");
        # 打印"IF YOU DO NOT GUESS THE AMOUNT, THE GAME ENDS."
        System.out.println("IF YOU DO NOT GUESS THE AMOUNT, THE GAME ENDS.");
    }

    /**
     * 从键盘获取玩家的猜测
     *
     * @return 玩家的猜测作为整数
     */
    private int playerGuess() {
        # 从键盘获取玩家的输入，并将其转换为整数返回
        return Integer.parseInt((displayTextAndGetInput("YOUR GUESS? ")));
    }
    # 定义一个函数，用于检查玩家是否输入了 Y 或者 YES
    def yesEntered(text):
        # 调用 stringIsAnyValue 函数，判断玩家输入的字符串是否是 Y 或者 YES，返回结果
        return stringIsAnyValue(text, "Y", "YES")

    # 定义一个函数，用于检查一个字符串是否等于一系列变量数量的值
    # 用于检查是否等于 Y 或者 YES 等值
    # 比较是不区分大小写的
    def stringIsAnyValue(text, *values):
        // 遍历变量 values 中的每个值，并测试每个值
        for(String val:values) {
            // 如果 text 与 val 相等（忽略大小写），则返回 true
            if(text.equalsIgnoreCase(val)) {
                return true;
            }
        }

        // 没有匹配的值，返回 false
        return false;
    }

    /*
     * 在屏幕上打印一条消息，然后从键盘接受输入。
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next();  # 从键盘输入中获取下一个输入值并返回

    /**
     * 生成随机数
     * 用作计算机玩家的单个数字
     *
     * @return 随机数
     */
    private int randomNumber() {
        return (int) (Math.random()  # 生成一个0到1之间的随机小数
                * (HIGH_NUMBER_RANGE - LOW_NUMBER_RANGE + 1) + LOW_NUMBER_RANGE);  # 将随机小数转换为指定范围内的整数并返回
    }
}
```