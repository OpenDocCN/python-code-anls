# `82_Stars\java\src\Stars.java`

```
import java.util.Arrays;  # 导入 java.util.Arrays 包
import java.util.Scanner;  # 导入 java.util.Scanner 包

/**
 * Game of Stars
 *
 * Based on the Basic game of Stars here
 * https://github.com/coding-horror/basic-computer-games/blob/main/82%20Stars/stars.bas
 *
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 *        new features - no additional text, error checking, etc has been added.
 */
public class Stars {

    public static final int HIGH_NUMBER_RANGE = 100;  # 定义常量 HIGH_NUMBER_RANGE 为 100
    public static final int MAX_GUESSES = 7;  # 定义常量 MAX_GUESSES 为 7

    private enum GAME_STATE {  # 定义枚举类型 GAME_STATE
        STARTING,  # 枚举值 STARTING
        INSTRUCTIONS,  # 枚举值 INSTRUCTIONS
        START_GAME, // 定义游戏状态为开始游戏
        GUESSING, // 定义游戏状态为猜测中
        WON, // 定义游戏状态为胜利
        LOST, // 定义游戏状态为失败
        GAME_OVER // 定义游戏状态为游戏结束
    }

    // 用于键盘输入
    private final Scanner kbScanner; // 声明一个Scanner对象用于接收键盘输入

    // 当前游戏状态
    private GAME_STATE gameState; // 声明一个GAME_STATE枚举类型的变量用于表示当前游戏状态

    // 玩家猜测次数
    private int playerTotalGuesses; // 声明一个整型变量用于表示玩家总共的猜测次数

    // 玩家当前猜测
    private int playerCurrentGuess; // 声明一个整型变量用于表示玩家当前的猜测

    // 计算机生成的随机数
    private int computersNumber;  // 声明一个私有整型变量computersNumber

    public Stars() {  // Stars类的构造函数

        gameState = GAME_STATE.STARTING;  // 将游戏状态设置为GAME_STATE.STARTING

        // Initialise kb scanner
        kbScanner = new Scanner(System.in);  // 初始化键盘输入扫描器
    }

    /**
     * Main game loop
     *
     */
    public void play() {  // 游戏的主循环

        do {
            switch (gameState) {  // 根据游戏状态进行不同的操作

                // Show an introduction the first time the game is played.
                case STARTING:  # 如果游戏状态为开始
                    intro();  # 调用intro()函数，显示游戏介绍
                    gameState = GAME_STATE.INSTRUCTIONS;  # 将游戏状态设置为显示游戏说明
                    break;  # 结束当前的case语句

                // Ask if instructions are needed and display if yes
                case INSTRUCTIONS:  # 如果游戏状态为显示游戏说明
                    if(yesEntered(displayTextAndGetInput("DO YOU WANT INSTRUCTIONS? "))) {  # 如果玩家输入yes，显示游戏说明
                        instructions();  # 调用instructions()函数，显示游戏说明
                    }
                    gameState = GAME_STATE.START_GAME;  # 将游戏状态设置为开始游戏
                    break;  # 结束当前的case语句

                // Generate computers number for player to guess, etc.
                case START_GAME:  # 如果游戏状态为开始游戏
                    init();  # 调用init()函数，初始化游戏
                    System.out.println("OK, I AM THINKING OF A NUMBER, START GUESSING.");  # 在控制台打印消息
                    gameState = GAME_STATE.GUESSING;  # 将游戏状态设置为猜数中
                    break;  # 结束当前的case语句
                // 玩家猜数字，直到猜中或用尽所有的猜测次数
                case GUESSING:
                    playerCurrentGuess = playerGuess(); // 玩家猜测当前的数字

                    // 检查玩家是否猜中了数字
                    if(playerCurrentGuess == computersNumber) {
                        gameState = GAME_STATE.WON; // 游戏状态变为胜利
                    } else {
                        // 猜错了
                        showStars(); // 显示星号
                        playerTotalGuesses++; // 玩家总猜测次数加一
                        // 用尽了所有的猜测次数？
                        if (playerTotalGuesses > MAX_GUESSES) {
                            gameState = GAME_STATE.LOST; // 游戏状态变为失败
                        }
                    }
                    break;

                // 赢得游戏
                case WON:
                    System.out.println(stars(79));  // 打印一行分隔符
                    System.out.println("YOU GOT IT IN " + playerTotalGuesses
                            + " GUESSES!!!  LET'S PLAY AGAIN...");  // 打印玩家猜对的消息和总猜测次数，提示再次开始游戏
                    gameState = GAME_STATE.START_GAME;  // 将游戏状态设置为开始游戏
                    break;  // 跳出switch语句

                // Lost game by running out of guesses
                case LOST:  // 如果游戏状态为LOST
                    System.out.println("SORRY, THAT'S " + MAX_GUESSES
                            + " GUESSES. THE NUMBER WAS " + computersNumber);  // 打印玩家猜测次数用完的消息和电脑生成的数字
                    gameState = GAME_STATE.START_GAME;  // 将游戏状态设置为开始游戏
                    break;  // 跳出switch语句
            }
            // Endless loop since the original code did not allow the player to exit
        } while (gameState != GAME_STATE.GAME_OVER);  // 无限循环，直到游戏状态为GAME_OVER
    }

    /**
     * Shows how close a players guess is to the computers number by
    /**
     * 显示一系列星号 - 星号越多，数字越接近
     */
    private void showStars() {
        // 计算玩家猜测与计算机数字的差值
        int d = Math.abs(playerCurrentGuess - computersNumber);
        int starsToShow;
        // 根据差值确定要显示的星号数量
        if(d >=64) {
            starsToShow = 1;
        } else if(d >=32) {
            starsToShow = 2;
        } else if (d >= 16) {
            starsToShow = 3;
        } else if (d >=8) {
            starsToShow = 4;
        } else if( d>= 4) {
            starsToShow = 5;
        } else if(d>= 2) {
            starsToShow = 6;
        } else {
            starsToShow = 7;  # 初始化变量starsToShow为7，表示需要显示7个星号
        }
        System.out.println(stars(starsToShow));  # 调用stars方法，打印出指定数量的星号
    }

    /**
     * Show a number of stars (asterisks)
     * @param number the number of stars needed  # 参数number表示需要显示的星号数量
     * @return the string encoded with the number of required stars  # 返回一个包含指定数量星号的字符串
     */
    private String stars(int number) {  # 定义一个私有方法stars，接受一个整数参数
        char[] stars = new char[number];  # 创建一个包含指定数量字符的字符数组
        Arrays.fill(stars, '*');  # 使用*填充字符数组
        return new String(stars);  # 将字符数组转换为字符串并返回
    }

    /**
     * Initialise variables before each new game  # 在每个新游戏开始前初始化变量
     *
     */
    private void init() {
        playerTotalGuesses = 1; // 初始化玩家猜测次数为1
        computersNumber = randomNumber(); // 生成计算机的随机数字
    }

    public void instructions() {
        System.out.println("I AM THINKING OF A WHOLE NUMBER FROM 1 TO " + HIGH_NUMBER_RANGE); // 打印游戏规则提示
        System.out.println("TRY TO GUESS MY NUMBER.  AFTER YOU GUESS, I"); // 打印游戏规则提示
        System.out.println("WILL TYPE ONE OR MORE STARS (*).  THE MORE"); // 打印游戏规则提示
        System.out.println("STARS I TYPE, THE CLOSER YOU ARE TO MY NUMBER."); // 打印游戏规则提示
        System.out.println("ONE STAR (*) MEANS FAR AWAY, SEVEN STARS (*******)"); // 打印游戏规则提示
        System.out.println("MEANS REALLY CLOSE!  YOU GET " + MAX_GUESSES + " GUESSES."); // 打印游戏规则提示
    }

    public void intro() {
        System.out.println("STARS"); // 打印游戏介绍
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"); // 打印游戏介绍
        System.out.println(); // 打印游戏介绍
    }
    /**
     * 从键盘获取玩家的猜测
     *
     * @return 玩家猜测的整数
     */
    private int playerGuess() {
        return Integer.parseInt((displayTextAndGetInput("YOUR GUESS? ")));
    }

    /**
     * 检查玩家是否输入了Y或YES作为答案
     *
     * @param text  从键盘输入的玩家字符串
     * @return 如果输入了Y或YES，则返回true，否则返回false
     */
    private boolean yesEntered(String text) {
        return stringIsAnyValue(text, "Y", "YES");
    }

    /**
# 检查字符串是否等于一系列变量值中的一个
# 用于检查例如 Y 或 YES 等情况
# 比较是不区分大小写的
#
# @param text 源字符串
# @param values 用于与源字符串比较的一系列值
# @return 如果在传递的一系列字符串中找到了匹配，则返回 true
*/
private boolean stringIsAnyValue(String text, String... values) {

    // 循环遍历一系列变量值并测试每个值
    for(String val:values) {
        if(text.equalsIgnoreCase(val)) {
            return true;
        }
    }

    // 没有匹配
    return false;
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
     * 生成随机数
     *
     * @return 随机数
     */
    private int randomNumber() {
        return (int) (Math.random()
                * (HIGH_NUMBER_RANGE) + 1);
    }
    }
}
```

这部分代码是一个函数的结束标志，表示函数的定义结束。在Python中，函数的定义使用关键字def开始，然后是函数的具体实现，最后以冒号开始，函数体缩进的方式来表示。在这个例子中，}表示函数体的结束，而}表示函数的定义的结束。
```