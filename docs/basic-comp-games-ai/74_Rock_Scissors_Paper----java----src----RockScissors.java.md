# `basic-computer-games\74_Rock_Scissors_Paper\java\src\RockScissors.java`

```
// 导入必要的类
import java.util.Arrays;
import java.util.Scanner;

/**
 * Rock Scissors Paper 游戏
 * <p>
 * 基于基本的 Rock Scissors 游戏
 * https://github.com/coding-horror/basic-computer-games/blob/main/74%20Rock%20Scissors%20Paper/rockscissors.bas
 * <p>
 * 注意：这个想法是在 Java 中创建一个 1970 年代 Basic 游戏的版本，没有引入新功能 - 没有添加额外的文本、错误检查等。
 */
public class RockScissors {

    // 最大游戏次数
    public static final int MAX_GAMES = 10;

    // 定义 Rock, Scissors, Paper 对应的数字
    public static final int PAPER = 1;
    public static final int SCISSORS = 2;
    public static final int ROCK = 3;

    // 用于键盘输入
    private final Scanner kbScanner;

    // 游戏状态枚举
    private enum GAME_STATE {
        START_GAME,
        GET_NUMBER_GAMES,
        START_ROUND,
        PLAY_ROUND,
        GAME_RESULT,
        GAME_OVER
    }

    // 当前游戏状态
    private GAME_STATE gameState;

    // 获胜者枚举
    private enum WINNER {
        COMPUTER,
        PLAYER
    }

    // 游戏获胜者
    private WINNER gameWinner;

    // 玩家和计算机的胜利次数
    int playerWins;
    int computerWins;
    // 游戏总次数和当前游戏次数
    int numberOfGames;
    int currentGameCount;
    // 计算机的选择
    int computersChoice;

    // 构造函数
    public RockScissors() {
        kbScanner = new Scanner(System.in);
        gameState = GAME_STATE.START_GAME;
    }

    /**
     * 主游戏循环
     */
    private void intro() {
        // 打印游戏介绍
        System.out.println(addSpaces(21) + "GAME OF ROCK, SCISSORS, PAPER");
        System.out.println(addSpaces(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
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
}
    /*
     * 在屏幕上打印一条消息，然后接受键盘输入。
     *
     * @param text 要在屏幕上显示的消息。
     * @return 玩家输入的内容。
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);  // 在屏幕上打印消息
        return kbScanner.nextLine();  // 接受键盘输入并返回
    }

    /**
     * 返回一个包含 x 个空格的字符串
     *
     * @param spaces 所需的空格数
     * @return 包含空格数的字符串
     */
    private String addSpaces(int spaces) {
        char[] spacesTemp = new char[spaces];  // 创建一个包含指定空格数的字符数组
        Arrays.fill(spacesTemp, ' ');  // 用空格填充字符数组
        return new String(spacesTemp);  // 将字符数组转换为字符串并返回
    }

    public static void main(String[] args) {

        RockScissors rockScissors = new RockScissors();  // 创建 RockScissors 对象
        rockScissors.play();  // 调用 play 方法开始游戏
    }
# 闭合前面的函数定义
```