# `d:/src/tocomm/basic-computer-games\80_Slots\java\src\Slots.java`

```
import java.util.Arrays;  # 导入 java.util.Arrays 包，用于操作数组
import java.util.Scanner;  # 导入 java.util.Scanner 包，用于接收用户输入

/**
 * Game of Slots
 * <p>
 * Based on the Basic game of Slots here
 * https://github.com/coding-horror/basic-computer-games/blob/main/80%20Slots/slots.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Slots {

    public static final String[] SLOT_SYMBOLS = {"BAR", "BELL", "ORANGE", "LEMON", "PLUM", "CHERRY"};  # 定义一个包含不同水果的数组

    public static final int NUMBER_SYMBOLS = SLOT_SYMBOLS.length;  # 获取水果数组的长度

    // Jackpot symbol (BAR)
    public static final int BAR_SYMBOL = 0;  # 定义 BAR 水果在数组中的索引位置
    // 表示当前旋转没有赢得任何奖励
    public static final int NO_WINNER = -1;

    // 用于键盘输入
    private final Scanner kbScanner;

    // 游戏状态枚举
    private enum GAME_STATE {
        START_GAME, // 开始游戏
        ONE_SPIN,   // 单次旋转
        RESULTS,    // 结果
        GAME_OVER   // 游戏结束
    }

    // 当前游戏状态
    private GAME_STATE gameState;

    // 不同类型的旋转结果
    private enum WINNINGS {
        JACKPOT(100), // 头奖
        TOP_DOLLAR(10),  // 定义一个枚举值 TOP_DOLLAR，其对应的赔率为10
        DOUBLE_BAR(5),   // 定义一个枚举值 DOUBLE_BAR，其对应的赔率为5
        REGULAR(2),      // 定义一个枚举值 REGULAR，其对应的赔率为2
        NO_WIN(0);       // 定义一个枚举值 NO_WIN，其对应的赔率为0

        private final int multiplier;  // 声明一个私有的整型变量 multiplier

        WINNINGS(int mult) {  // 定义一个构造函数 WINNINGS，接受一个整型参数 mult
            multiplier = mult;  // 将参数赋值给 multiplier
        }

        // No win returns the negative amount of net
        // otherwise calculate winnings based on
        // multiplier
        public int calculateWinnings(int bet) {  // 定义一个公共方法 calculateWinnings，接受一个整型参数 bet

            if (multiplier == 0) {  // 如果 multiplier 等于0
                return -bet;  // 返回负的下注金额
            } else {
                // Return original bet plus a multipler
                // 否则计算赢得的金额，基于 multiplier
    // 定义一个私有变量，用于存储键盘输入
    private Scanner kbScanner;

    // 定义游戏状态枚举，初始状态为开始游戏
    private GAME_STATE gameState;

    // 构造函数，初始化键盘输入扫描器和游戏状态
    public Slots() {
        kbScanner = new Scanner(System.in);
        gameState = GAME_STATE.START_GAME;
    }

    /**
     * 主游戏循环
     */
    public void play() {
        // 定义一个长度为3的整型数组，用于存储三个轮盘的结果
        int[] slotReel = new int[3];
        do {
            // Results of a single spin
            WINNINGS winnings; // 声明一个名为winnings的WINNINGS类型变量

            switch (gameState) { // 根据游戏状态进行不同的操作

                case START_GAME: // 游戏开始状态
                    intro(); // 调用intro函数，显示游戏介绍
                    playerBalance = 0; // 玩家余额设为0
                    gameState = GAME_STATE.ONE_SPIN; // 将游戏状态设为进行一次旋转
                    break; // 结束当前case

                case ONE_SPIN: // 进行一次旋转状态

                    int playerBet = displayTextAndGetNumber("YOUR BET? "); // 调用displayTextAndGetNumber函数，显示提示并获取玩家下注金额

                    slotReel[0] = randomSymbol(); // 将第一个轮盘的符号设为随机生成的符号
                    slotReel[1] = randomSymbol(); // 将第二个轮盘的符号设为随机生成的符号
                    slotReel[2] = randomSymbol(); // 将第三个轮盘的符号设为随机生成的符号
                    // 存储哪个符号（如果有的话）至少匹配另一个卷轴
                    int whichSymbolWon = winningSymbol(slotReel[0], slotReel[1], slotReel[2]);

                    // 显示三个随机抽取的符号
                    StringBuilder output = new StringBuilder();
                    for (int i = 0; i < 3; i++) {
                        if (i > 0) {
                            output.append(" ");
                        }
                        output.append(SLOT_SYMBOLS[slotReel[i]]);
                    }

                    System.out.println(output);

                    // 计算结果

                    if (whichSymbolWon == NO_WINNER) {
                        // 没有符号匹配 = 没有赢得任何东西
                        winnings = WINNINGS.NO_WIN;
                    } else if (slotReel[0] == slotReel[1] && slotReel[0] == slotReel[2]) {
                        // 如果第一个轮盘符号等于第二个轮盘符号，并且等于第三个轮盘符号
                        // 顶部奖励，3个匹配的符号
                        winnings = WINNINGS.TOP_DOLLAR;
                        if (slotReel[0] == BAR_SYMBOL) {
                            // 所有3个符号都是BAR。中了大奖！
                            winnings = WINNINGS.JACKPOT;
                        }
                    } else {
                        // 在这一点上，剩下的选项是普通的赢或者双倍赢，因为其余的（包括不中奖）已经在上面检查过了。
                        // 假设是普通的赢
                        winnings = WINNINGS.REGULAR;

                        // 但如果匹配的是BAR符号，那么就是双倍BAR
                        if (slotReel[0] == BAR_SYMBOL) {
                            winnings = WINNINGS.DOUBLE_BAR;
                        }

                    }
                    // 更新玩家的余额，根据这次旋转赢得或输掉的金额
                    playerBalance += winnings.calculateWinnings(playerBet);

                    System.out.println();

                    // 输出这次旋转的结果
                    switch (winnings) {
                        case NO_WIN:
                            System.out.println("YOU LOST.");
                            break;

                        case REGULAR:
                            System.out.println("DOUBLE!!");
                            System.out.println("YOU WON!");
                            break;

                        case DOUBLE_BAR:
                            System.out.println("*DOUBLE BAR*");
                            System.out.println("YOU WON!");
                            break;  // 结束当前的 switch 语句块

                        case TOP_DOLLAR:  // 如果玩家赢得了 TOP DOLLAR
                            System.out.println();  // 输出空行
                            System.out.println("**TOP DOLLAR**");  // 输出提示信息
                            System.out.println("YOU WON!");  // 输出提示信息
                            break;  // 结束当前的 switch 语句块

                        case JACKPOT:  // 如果玩家赢得了 JACKPOT
                            System.out.println();  // 输出空行
                            System.out.println("***JACKPOT***");  // 输出提示信息
                            System.out.println("YOU WON!");  // 输出提示信息
                            break;  // 结束当前的 switch 语句块

                    }

                    System.out.println("YOUR STANDINGS ARE $" + playerBalance);  // 输出玩家的余额信息

                    // 如果玩家选择不再玩游戏，显示本次游戏的结果
                    if (!yesEntered(displayTextAndGetInput("AGAIN? "))) {
                    gameState = GAME_STATE.RESULTS;  # 设置游戏状态为结果

                }
                break;

            case RESULTS:  # 当游戏状态为结果时
                if (playerBalance == 0) {  # 如果玩家余额为0
                    System.out.println("HEY, YOU BROKE EVEN.");  # 输出平局信息
                } else if (playerBalance > 0) {  # 如果玩家余额大于0
                    System.out.println("COLLECT YOUR WINNINGS FROM THE H&M CASHIER.");  # 输出赢得奖金信息
                } else {  # 否则
                    // Lost  # 输了
                    System.out.println("PAY UP!  PLEASE LEAVE YOUR MONEY ON THE TERMINAL.");  # 输出输钱信息
                }

                gameState = GAME_STATE.GAME_OVER;  # 设置游戏状态为游戏结束
                break;
        }
    } while (gameState != GAME_STATE.GAME_OVER);  # 当游戏状态不是游戏结束时循环
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
     * @param text 要在屏幕上显示的消息。
     * @return 玩家输入的内容。
     */
    private int displayTextAndGetNumber(String text) {
        return Integer.parseInt(displayTextAndGetInput(text));
    }
# 打印屏幕上的消息，然后从键盘接受输入。
# @param text 要在屏幕上显示的消息。
# @return 玩家输入的内容。
private String displayTextAndGetInput(String text) {
    System.out.print(text);
    return kbScanner.next();
}

/**
 * 检查玩家是否输入了Y或YES作为答案。
 * @param text 从键盘输入的字符串
 * @return 如果输入了Y或YES，则返回true，否则返回false
 */
private boolean yesEntered(String text) {
    return stringIsAnyValue(text, "Y", "YES");
}
    }

    /**
     * Check whether a string equals one of a variable number of values
     * Useful to check for Y or YES for example
     * Comparison is case insensitive.
     *
     * @param text   source string  // 源字符串
     * @param values a range of values to compare against the source string  // 与源字符串进行比较的一系列值
     * @return true if a comparison was found in one of the variable number of strings passed  // 如果在传递的一系列字符串中找到了比较，则返回true
     */
    private boolean stringIsAnyValue(String text, String... values) {
        // 使用流处理数组中的值，检查是否有任何一个与源字符串相等（忽略大小写）
        return Arrays.stream(values).anyMatch(str -> str.equalsIgnoreCase(text));
    }

    /**
     * Simulate the old basic tab(xx) command which indented text by xx spaces.
     *
     * @param spaces number of spaces required  // 所需的空格数
    /**
     * @param spaces - the number of spaces to be simulated
     * @return String with number of spaces
     */
    private String simulateTabs(int spaces) {
        char[] spacesTemp = new char[spaces]; // create a char array with the specified number of spaces
        Arrays.fill(spacesTemp, ' '); // fill the char array with spaces
        return new String(spacesTemp); // convert the char array to a String and return
    }

    /**
     * Find the symbol that won this round i.e. the first reel that matched another reel
     *
     * @param reel1 reel1 spin result
     * @param reel2 reel2 spin result
     * @param reel3 reel3 spin result
     * @return NO_WINNER if no reels match otherwise an int 0-2 to indicate the reel that matches another
     */
    private int winningSymbol(int reel1, int reel2, int reel3) {
        if (reel1 == reel2) { // check if reel1 matches reel2
            return 0; // return 0 if reel1 matches reel2
        } else if (reel1 == reel3) { // if reel1 does not match reel2, check if reel1 matches reel3
            return 0;  // 如果第一个轮子和第二个轮子不匹配，则返回0
        } else if (reel2 == reel3) {  // 如果第二个轮子和第三个轮子匹配，则返回1
            return 1;
        } else {  // 如果都不匹配，则返回NO_WINNER
            return NO_WINNER;
        }
    }

    /**
     * Random symbol for a slot wheel
     *
     * @return number between 0-5  // 返回0到5之间的随机数，代表轮子上的符号
     */
    private int randomSymbol() {
        return (int) (Math.random() * NUMBER_SYMBOLS);  // 生成一个0到5之间的随机整数作为符号
    }
}
```