# `d:/src/tocomm/basic-computer-games\08_Batnum\java\src\BatNum.java`

```
import java.util.Arrays;  # 导入 java.util.Arrays 模块
import java.util.Scanner;  # 导入 java.util.Scanner 模块

/**
 * Game of BatNum
 * <p>
 * Based on the Basic game of BatNum here
 * https://github.com/coding-horror/basic-computer-games/blob/main/08%20Batnum/batnum.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class BatNum {

    private enum GAME_STATE {  # 定义枚举类型 GAME_STATE
        STARTING,  # 开始状态
        START_GAME,  # 开始游戏状态
        CHOOSE_PILE_SIZE,  # 选择堆大小状态
        SELECT_WIN_OPTION,  # 选择获胜选项状态
        CHOOSE_MIN_AND_MAX,  # 选择最小和最大状态
        SELECT_WHO_STARTS_FIRST,  // 定义游戏状态：选择谁先开始
        PLAYERS_TURN,  // 定义游戏状态：玩家回合
        COMPUTERS_TURN,  // 定义游戏状态：电脑回合
        ANNOUNCE_WINNER,  // 定义游戏状态：宣布胜利者
        GAME_OVER  // 定义游戏状态：游戏结束
    }

    // 用于键盘输入
    private final Scanner kbScanner;  // 声明一个用于键盘输入的 Scanner 对象

    // 当前游戏状态
    private GAME_STATE gameState;  // 声明一个用于存储当前游戏状态的变量

    private int pileSize;  // 声明一个用于存储堆大小的变量

    // 如何赢得游戏的选项
    enum WIN_OPTION {  // 定义一个枚举类型，表示如何赢得游戏的选项
        TAKE_LAST,  // 选项：拿走最后一个
        AVOID_LAST  // 选项：避开最后一个
    }
    // 跟踪赢家
    enum WINNER {
        COMPUTER,  // 电脑赢
        PLAYER     // 玩家赢
    }

    private WINNER winner;  // 保存赢家信息

    private WIN_OPTION winOption;  // 保存赢得选项信息

    private int minSelection;  // 最小选择数
    private int maxSelection;  // 最大选择数

    // 用于计算电脑的最佳移动
    private int rangeOfRemovals;  // 移除范围

    public BatNum() {

        gameState = GAME_STATE.STARTING;  // 设置游戏状态为开始中
        // 初始化键盘扫描器
        kbScanner = new Scanner(System.in);
    }

    /**
     * 主游戏循环
     */
    public void play() {

        do {
            switch (gameState) {

                // 第一次玩游戏时显示介绍和可选的说明
                case STARTING:
                    intro();
                    gameState = GAME_STATE.START_GAME;
                    break;

                // 开始新游戏
                case START_GAME:  # 如果游戏状态为开始游戏
                    gameState = GAME_STATE.CHOOSE_PILE_SIZE;  # 设置游戏状态为选择牌堆大小
                    break;  # 结束当前的 case 语句

                case CHOOSE_PILE_SIZE:  # 如果游戏状态为选择牌堆大小
                    System.out.println();  # 打印空行
                    System.out.println();  # 再次打印空行
                    pileSize = displayTextAndGetNumber("ENTER PILE SIZE ");  # 从用户输入中获取牌堆大小
                    if (pileSize >= 1) {  # 如果牌堆大小大于等于1
                        gameState = GAME_STATE.SELECT_WIN_OPTION;  # 设置游戏状态为选择获胜选项
                    }
                    break;  # 结束当前的 case 语句

                case SELECT_WIN_OPTION:  # 如果游戏状态为选择获胜选项
                    int winChoice = displayTextAndGetNumber("ENTER WIN OPTION - 1 TO TAKE LAST, 2 TO AVOID LAST: ");  # 从用户输入中获取获胜选项
                    if (winChoice == 1) {  # 如果获胜选项为1
                        winOption = WIN_OPTION.TAKE_LAST;  # 设置获胜选项为取最后一张牌
                        gameState = GAME_STATE.CHOOSE_MIN_AND_MAX;  # 设置游戏状态为选择最小和最大取牌数量
                    } else if (winChoice == 2) {  # 如果获胜选项为2
                        winOption = WIN_OPTION.AVOID_LAST;  # 设置获胜选项为避免最后一张牌
                    }
                    break;

                case CHOOSE_MIN_AND_MAX:
                    // 从用户输入中获取最小和最大值
                    String range = displayTextAndGetInput("ENTER MIN AND MAX ");
                    minSelection = getDelimitedValue(range, 0);  // 从输入中获取最小值
                    maxSelection = getDelimitedValue(range, 1);  // 从输入中获取最大值
                    if (maxSelection > minSelection && minSelection >= 1) {  // 如果最大值大于最小值且最小值大于等于1
                        gameState = GAME_STATE.SELECT_WHO_STARTS_FIRST;  // 切换游戏状态到选择谁先开始
                    }

                    // 用于计算机在其回合中使用
                    rangeOfRemovals = minSelection + maxSelection;  // 计算可移除的范围
                    break;

                case SELECT_WHO_STARTS_FIRST:
                    // 获取玩家选择谁先开始的选项
                    int playFirstChoice = displayTextAndGetNumber("ENTER START OPTION - 1 COMPUTER FIRST, 2 YOU FIRST ");
                    if (playFirstChoice == 1) {  // 如果选择计算机先开始
                        gameState = GAME_STATE.COMPUTERS_TURN;  // 切换游戏状态到计算机回合
                    }
                    ```
                    } else if (playFirstChoice == 2) {  # 如果玩家选择了第二个选项
                        gameState = GAME_STATE.PLAYERS_TURN;  # 切换游戏状态为玩家回合
                    }
                    break;  # 结束当前的 switch 语句

                case PLAYERS_TURN:  # 如果游戏状态为玩家回合
                    int playersMove = displayTextAndGetNumber("YOUR MOVE ");  # 获取玩家的移动

                    if (playersMove == 0) {  # 如果玩家的移动为 0
                        System.out.println("I TOLD YOU NOT TO USE ZERO! COMPUTER WINS BY FORFEIT.");  # 输出提示信息
                        winner = WINNER.COMPUTER;  # 设置获胜者为电脑
                        gameState = GAME_STATE.ANNOUNCE_WINNER;  # 切换游戏状态为宣布获胜者
                        break;  # 结束当前的 switch 语句
                    }

                    if (playersMove == pileSize && winOption == WIN_OPTION.AVOID_LAST) {  # 如果玩家的移动等于堆的大小并且胜利选项为避免最后一颗
                        winner = WINNER.COMPUTER;  # 设置获胜者为电脑
                        gameState = GAME_STATE.ANNOUNCE_WINNER;  # 切换游戏状态为宣布获胜者
                        break;  # 结束当前的 switch 语句
                    }
                    // 检查玩家的移动是否在最小和最大可能范围内
                    if (playersMove >= minSelection && playersMove <= maxSelection) {
                        // 有效，所以减少堆大小
                        pileSize -= playersMove;

                        // 这个移动导致堆上没有更多的对象吗？
                        if (pileSize == 0) {
                            // 游戏设置为谁取走最后一个对象就是赢家吗？
                            if (winOption == WIN_OPTION.TAKE_LAST) {
                                // 玩家赢了
                                winner = WINNER.PLAYER;
                            } else {
                                // 电脑赢了
                                winner = WINNER.COMPUTER;
                            }
                            gameState = GAME_STATE.ANNOUNCE_WINNER;
                        } else {
                            // 还有物品剩下
                            gameState = GAME_STATE.COMPUTERS_TURN;
                    } else {
                        // 无效的移动
                        System.out.println("非法移动，请重新输入");
                    }
                    break;

                case COMPUTERS_TURN:
                    // 保存当前堆的大小
                    int pileSizeLeft = pileSize;
                    if (winOption == WIN_OPTION.TAKE_LAST) {
                        if (pileSize > maxSelection) {
                            // 计算计算机的回合需要移除的对象数量
                            int objectsToRemove = calculateComputersTurn(pileSizeLeft);

                            // 更新堆的大小
                            pileSize -= objectsToRemove;
                            System.out.println("计算机取走 " + objectsToRemove + " 个对象，剩余 " + pileSize);
                            gameState = GAME_STATE.PLAYERS_TURN;
                        } else {
                            System.out.println("计算机取走 " + pileSize + " 个对象，获胜。");
                            winner = WINNER.COMPUTER;
# 设置游戏状态为宣布获胜者
gameState = GAME_STATE.ANNOUNCE_WINNER;
# 如果玩家获胜
if (winner == WINNER.PLAYER) {
    # 打印玩家获胜信息
    System.out.println("PLAYER WINS!");
} else {
    # 如果玩家没有获胜
    # 如果还有剩余的物品堆
    if (pileSizeLeft > 0) {
        # 减少剩余物品堆的数量
        pileSizeLeft--;
        # 如果剩余物品堆的数量大于最小可选数量
        if (pileSize > minSelection) {
            # 计算电脑的回合需要移除的物品数量
            int objectsToRemove = calculateComputersTurn(pileSizeLeft);
            # 减少物品堆的数量
            pileSize -= objectsToRemove;
            # 打印电脑移除物品的信息和剩余物品堆的数量
            System.out.println("COMPUTER TAKES " + objectsToRemove + " AND LEAVES " + pileSize);
            # 设置游戏状态为玩家回合
            gameState = GAME_STATE.PLAYERS_TURN;
        } else {
            # 如果剩余物品堆的数量不大于最小可选数量
            # 打印电脑移除所有剩余物品并且失败的信息
            System.out.println("COMPUTER TAKES " + pileSize + " AND LOSES.");
            # 设置获胜者为玩家
            winner = WINNER.PLAYER;
            # 设置游戏状态为宣布获胜者
            gameState = GAME_STATE.ANNOUNCE_WINNER;
        }
    }
}
# 结束switch语句
break;

# 如果游戏状态为宣布获胜者
case ANNOUNCE_WINNER:
    # 根据获胜者进行不同的操作
    switch (winner) {
        # 如果获胜者为玩家
        case PLAYER:
                            System.out.println("CONGRATULATIONS, YOU WIN.");  // 打印“恭喜，你赢了。”
                            break;  // 跳出循环
                        case COMPUTER:  // 如果是计算机的回合
                            System.out.println("TOUGH LUCK, YOU LOSE.");  // 打印“很遗憾，你输了。”
                            break;  // 跳出循环
                    }
                    gameState = GAME_STATE.START_GAME;  // 设置游戏状态为开始游戏
                    break;  // 跳出循环
            }
        } while (gameState != GAME_STATE.GAME_OVER);  // 当游戏状态不是游戏结束时继续循环
    }

    /**
     * Figure out the computers turn - i.e. how many objects to remove
     *
     * @param pileSizeLeft current size  // 参数：当前堆的大小
     * @return the number of objects to remove.  // 返回需要移除的对象数量
     */
    private int calculateComputersTurn(int pileSizeLeft) {  // 计算计算机的回合 - 即需要移除的对象数量
        int computersNumberToRemove = pileSizeLeft - rangeOfRemovals * (pileSizeLeft / rangeOfRemovals);  // 计算需要移除的对象数量
        if (computersNumberToRemove < minSelection) {  # 如果要移除的计算机数量小于最小选择数量
            computersNumberToRemove = minSelection;  # 将要移除的计算机数量设置为最小选择数量
        }
        if (computersNumberToRemove > maxSelection) {  # 如果要移除的计算机数量大于最大选择数量
            computersNumberToRemove = maxSelection;  # 将要移除的计算机数量设置为最大选择数量
        }

        return computersNumberToRemove;  # 返回要移除的计算机数量
    }

    private void intro() {  # 定义一个私有方法 intro
        System.out.println(simulateTabs(33) + "BATNUM");  # 在控制台打印字符串 "BATNUM"，并使用 simulateTabs 方法模拟 33 个制表符
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  # 在控制台打印字符串 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，并使用 simulateTabs 方法模拟 15 个制表符
        System.out.println();  # 在控制台打印空行
        System.out.println("THIS PROGRAM IS A 'BATTLE OF NUMBERS' GAME, WHERE THE");  # 在控制台打印说明信息
        System.out.println("COMPUTER IS YOUR OPPONENT.");  # 在控制台打印说明信息
        System.out.println();  # 在控制台打印空行
        System.out.println("THE GAME STARTS WITH AN ASSUMED PILE OF OBJECTS. YOU");  # 在控制台打印说明信息
        System.out.println("AND YOUR OPPONENT ALTERNATELY REMOVE OBJECTS FROM THE PILE.");  # 在控制台打印说明信息
        System.out.println("WINNING IS DEFINED IN ADVANCE AS TAKING THE LAST OBJECT OR");  # 在控制台打印说明信息
        System.out.println("NOT. YOU CAN ALSO SPECIFY SOME OTHER BEGINNING CONDITIONS.");
        // 打印消息到屏幕上，提示玩家可以指定其他的初始条件
        System.out.println("DON'T USE ZERO, HOWEVER, IN PLAYING THE GAME.");
        // 打印消息到屏幕上，提示玩家在游戏中不要使用零
        System.out.println("ENTER A NEGATIVE NUMBER FOR NEW PILE SIZE TO STOP PLAYING.");
        // 打印消息到屏幕上，提示玩家输入一个负数来停止游戏。
    }

    /*
     * Print a message on the screen, then accept input from Keyboard.
     * Converts input to Integer
     *
     * @param text message to be displayed on screen.
     * @return what was typed by the player.
     */
    private int displayTextAndGetNumber(String text) {
        // 打印消息到屏幕上，然后从键盘接受输入
        return Integer.parseInt(displayTextAndGetInput(text));
        // 将输入转换为整数并返回
    }

    /*
     * Print a message on the screen, then accept input from Keyboard.
     *
     * @param text message to be displayed on screen.
     */
    * @return what was typed by the player.
    */
    # 显示文本并获取玩家输入的内容
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next();
    }

    /**
     * Simulate the old basic tab(xx) command which indented text by xx spaces.
     *
     * @param spaces number of spaces required
     * @return String with number of spaces
     */
    # 模拟旧的基本tab(xx)命令，通过xx个空格缩进文本
    private String simulateTabs(int spaces) {
        char[] spacesTemp = new char[spaces];
        Arrays.fill(spacesTemp, ' ');
        return new String(spacesTemp);
    }

    /**
# Accepts a string delimited by comma's and returns the nth delimited
# value (starting at count 0).
# @param text - text with values separated by comma's
# @param pos  - which position to return a value for
# @return the int representation of the value
def getDelimitedValue(text, pos):
    # 将文本按逗号分隔成字符串数组
    tokens = text.split(",")
    # 将指定位置的字符串转换为整数并返回
    return int(tokens[pos])
```