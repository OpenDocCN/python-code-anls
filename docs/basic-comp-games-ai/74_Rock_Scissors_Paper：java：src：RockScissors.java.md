# `74_Rock_Scissors_Paper\java\src\RockScissors.java`

```
import java.util.Arrays;  // 导入 Arrays 类，用于操作数组
import java.util.Scanner;  // 导入 Scanner 类，用于接收用户输入

/**
 * Game of Rock Scissors Paper
 * <p>
 * Based on the Basic game of Rock Scissors here
 * https://github.com/coding-horror/basic-computer-games/blob/main/74%20Rock%20Scissors%20Paper/rockscissors.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */

public class RockScissors {

    public static final int MAX_GAMES = 10;  // 定义最大游戏次数为 10

    public static final int PAPER = 1;  // 定义石头、剪刀、布对应的数字
    public static final int SCISSORS = 2;
    public static final int ROCK = 3;
    // 用于键盘输入
    private final Scanner kbScanner;

    // 游戏状态枚举
    private enum GAME_STATE {
        START_GAME, // 开始游戏
        GET_NUMBER_GAMES, // 获取游戏数量
        START_ROUND, // 开始回合
        PLAY_ROUND, // 进行回合
        GAME_RESULT, // 游戏结果
        GAME_OVER // 游戏结束
    }

    // 当前游戏状态
    private GAME_STATE gameState;

    // 赢家枚举
    private enum WINNER {
        COMPUTER, // 电脑赢
        PLAYER // 玩家赢
    }
    private WINNER gameWinner;  # 声明一个私有的枚举类型变量gameWinner，用于存储游戏的胜者

    int playerWins;  # 声明一个整型变量playerWins，用于存储玩家的胜利次数
    int computerWins;  # 声明一个整型变量computerWins，用于存储计算机的胜利次数
    int numberOfGames;  # 声明一个整型变量numberOfGames，用于存储游戏的总局数
    int currentGameCount;  # 声明一个整型变量currentGameCount，用于存储当前游戏的局数
    int computersChoice;  # 声明一个整型变量computersChoice，用于存储计算机的选择

    public RockScissors() {  # RockScissors类的构造函数
        kbScanner = new Scanner(System.in);  # 创建一个Scanner对象，用于接收用户输入
        gameState = GAME_STATE.START_GAME;  # 初始化游戏状态为开始游戏
    }

    /**
     * Main game loop
     */
    public void play() {  # 定义一个名为play的公共方法，用于执行游戏的主循环

        do {  # 执行以下代码块直到游戏结束
# 根据游戏状态进行不同的操作
switch (gameState) {

    # 如果游戏状态为开始游戏
    case START_GAME:
        # 调用intro()函数
        intro();
        # 将当前游戏次数重置为0
        currentGameCount = 0;
        # 将游戏状态设置为获取游戏次数
        gameState = GAME_STATE.GET_NUMBER_GAMES;
        break;

    # 如果游戏状态为获取游戏次数
    case GET_NUMBER_GAMES:
        # 通过显示文本并获取数字的方式获取游戏次数
        numberOfGames = displayTextAndGetNumber("HOW MANY GAMES? ");
        # 如果游戏次数小于等于最大游戏次数
        if (numberOfGames <= MAX_GAMES) {
            # 将游戏状态设置为开始回合
            gameState = GAME_STATE.START_ROUND;
        } else {
            # 打印错误信息
            System.out.println("SORRY, BUT WE AREN'T ALLOWED TO PLAY THAT MANY.");
        }
        break;

    # 如果游戏状态为开始回合
    case START_ROUND:
        # 当前游戏次数加一
        currentGameCount++;
                    if (currentGameCount > numberOfGames) {  # 如果当前游戏次数大于总游戏次数
                        gameState = GAME_STATE.GAME_RESULT;  # 则游戏状态变为游戏结果
                        break;  # 跳出循环
                    }
                    System.out.println("GAME NUMBER: " + (currentGameCount));  # 打印当前游戏次数
                    computersChoice = (int) (Math.random() * 3) + 1;  # 生成计算机的选择（1-3之间的随机数）
                    gameState = GAME_STATE.PLAY_ROUND;  # 游戏状态变为进行游戏回合

                case PLAY_ROUND:  # 游戏状态为进行游戏回合时
                    System.out.println("3=ROCK...2=SCISSORS...1=PAPER");  # 打印玩家选择的提示
                    int playersChoice = displayTextAndGetNumber("1...2...3...WHAT'S YOUR CHOICE? ");  # 获取玩家的选择
                    if (playersChoice >= PAPER && playersChoice <= ROCK) {  # 如果玩家的选择在1-3之间
                        switch (computersChoice) {  # 根据计算机的选择进行判断
                            case PAPER:  # 如果计算机选择了纸
                                System.out.println("...PAPER");  # 打印计算机选择了纸
                                break;  # 跳出判断
                            case SCISSORS:  # 如果计算机选择了剪刀
                                System.out.println("...SCISSORS");  # 打印计算机选择了剪刀
                                break;  # 跳出判断
                            case ROCK:  # 如果计算机选择了石头
# 打印出"...ROCK"
System.out.println("...ROCK");
# 跳出循环
break;
# 如果玩家和计算机的选择相同
if (playersChoice == computersChoice) {
    # 打印出"TIE GAME.  NO WINNER."
    System.out.println("TIE GAME.  NO WINNER.");
# 如果选择不同
} else {
    # 根据玩家的选择进行判断
    switch (playersChoice) {
        # 如果玩家选择了PAPER
        case PAPER:
            # 如果计算机选择了SCISSORS
            if (computersChoice == SCISSORS) {
                # 将gameWinner赋值为WINNER.COMPUTER
                gameWinner = WINNER.COMPUTER;
            # 如果计算机选择了ROCK
            } else if (computersChoice == ROCK) {
                # 不需要在这里重新赋值，因为它已经初始化为false，我认为这有助于提高可读性。
                gameWinner = WINNER.PLAYER;
            # 跳出循环
            break;
        # 如果玩家选择了SCISSORS
        case SCISSORS:
            # 如果计算机选择了ROCK
            if (computersChoice == ROCK) {
                # 将gameWinner赋值为WINNER.COMPUTER
                gameWinner = WINNER.COMPUTER;
                                    } else if (computersChoice == PAPER) {
                                        # 如果电脑选择了“布”，则玩家获胜
                                        # 这里不需要重新赋值，因为它已经初始化为false，我认为这有助于提高可读性。
                                        gameWinner = WINNER.PLAYER;
                                    }
                                    break;
                                case ROCK:
                                    if (computersChoice == PAPER) {
                                        # 如果电脑选择了“布”，则电脑获胜
                                        gameWinner = WINNER.COMPUTER;
                                    } else if (computersChoice == SCISSORS) {
                                        # 如果电脑选择了“剪刀”，则玩家获胜
                                        # 这里不需要重新赋值，因为它已经初始化为false，我认为这有助于提高可读性。
                                        gameWinner = WINNER.PLAYER;
                                    }
                                    break;
                            }

                            if (gameWinner == WINNER.COMPUTER) {
                                # 打印电脑获胜的消息
                                System.out.println("WOW!  I WIN!!!");
                                # 增加电脑获胜次数
                                computerWins++;
                } else {
                    // 如果玩家赢了，打印“YOU WIN!!!”，并增加玩家赢的次数
                    System.out.println("YOU WIN!!!");
                    playerWins++;
                }
            }
            // 设置游戏状态为开始新回合
            gameState = GAME_STATE.START_ROUND;
        } else {
            // 如果玩家输入无效，打印“INVALID.”
            System.out.println("INVALID.");
        }

        break;

    case GAME_RESULT:
        // 打印空行和最终游戏得分
        System.out.println();
        System.out.println("HERE IS THE FINAL GAME SCORE:");
        System.out.println("I HAVE WON " + computerWins + " GAME" + (computerWins != 1 ? "S." : "."));
        System.out.println("YOU HAVE WON " + playerWins + " GAME" + (playerWins != 1 ? "S." : "."));
        // 计算平局的次数，并打印出来
        int tiedGames = numberOfGames - (computerWins + playerWins);
        System.out.println("AND " + tiedGames + " GAME" + (tiedGames != 1 ? "S " : " ") + "ENDED IN A TIE.");
        System.out.println();
                    System.out.println("THANKS FOR PLAYING!!");
                    // 打印感谢信息并结束游戏
                    gameState = GAME_STATE.GAME_OVER;
                    // 将游戏状态设置为游戏结束
            }
        } while (gameState != GAME_STATE.GAME_OVER);
    }

    private void intro() {
        System.out.println(addSpaces(21) + "GAME OF ROCK, SCISSORS, PAPER");
        // 打印游戏标题
        System.out.println(addSpaces(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        // 打印游戏信息
        System.out.println();
        // 打印空行
    }

    /*
     * Print a message on the screen, then accept input from Keyboard.
     * Converts input to an Integer
     *
     * @param text message to be displayed on screen.
     * @return what was typed by the player.
     */
    private int displayTextAndGetNumber(String text) {
        // 打印屏幕上的消息
        return Integer.parseInt(displayTextAndGetInput(text));
    }
    # 将显示文本并获取键盘输入的结果转换为整数并返回

    /*
     * Print a message on the screen, then accept input from Keyboard.
     *
     * @param text message to be displayed on screen.
     * @return what was typed by the player.
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.nextLine();
    }
    # 在屏幕上打印消息，然后从键盘接受输入。返回玩家输入的内容。

    /**
     * Return a string of x spaces
     *
     * @param spaces number of spaces required
     * @return String with number of spaces
     */
    private String addSpaces(int spaces) {  // 定义一个私有方法，用于在字符串中添加指定数量的空格
        char[] spacesTemp = new char[spaces];  // 创建一个字符数组，用于存储空格字符
        Arrays.fill(spacesTemp, ' ');  // 使用 Arrays.fill 方法将字符数组填充为指定字符
        return new String(spacesTemp);  // 将字符数组转换为字符串并返回
    }

    public static void main(String[] args) {  // 主方法，程序的入口

        RockScissors rockScissors = new RockScissors();  // 创建 RockScissors 对象
        rockScissors.play();  // 调用 RockScissors 对象的 play 方法开始游戏
    }
}
```