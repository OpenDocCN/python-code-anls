# `d:/src/tocomm/basic-computer-games\51_Hurkle\java\src\Hurkle.java`

```
import java.util.Scanner;  // 导入 Scanner 类，用于从控制台读取输入

/**
 * Game of Hurkle
 * <p>
 * Based on the Basic game of Hurkle here
 * https://github.com/coding-horror/basic-computer-games/blob/main/51%20Hurkle/hurkle.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Hurkle {

    public static final int GRID_SIZE = 10;  // 定义常量 GRID_SIZE，表示游戏地图的大小为 10x10
    public static final int MAX_GUESSES = 5;  // 定义常量 MAX_GUESSES，表示最大猜测次数为 5 次

    private enum GAME_STATE {  // 定义枚举类型 GAME_STATE，包括游戏的不同状态
        STARTING,  // 游戏开始前的状态
        START_GAME,  // 游戏开始状态
        GUESSING,  // 猜测状态
        PLAY_AGAIN,  // 定义游戏状态为再玩一次
        GAME_OVER    // 定义游戏状态为游戏结束
    }

    private GAME_STATE gameState;  // 定义游戏状态变量

    // 用于键盘输入
    private final Scanner kbScanner;  // 创建键盘输入扫描器

    private int guesses;  // 猜测次数

    // hurkle 位置
    private int hurkleXPos;  // hurkle 的 X 坐标
    private int hurkleYPos;  // hurkle 的 Y 坐标

    // 玩家猜测
    private int playerGuessXPos;  // 玩家猜测的 X 坐标
    private int playerGuessYPos;  // 玩家猜测的 Y 坐标

    public Hurkle() {  // Hurkle 类的构造函数
        gameState = GAME_STATE.STARTING;  // 设置游戏状态为开始状态

        // 初始化键盘输入扫描器
        kbScanner = new Scanner(System.in);
    }

    /**
     * Main game loop
     */
    public void play() {

        do {
            switch (gameState) {

                // 第一次玩游戏时显示介绍
                case STARTING:
                    intro();  // 调用介绍方法
                    gameState = GAME_STATE.START_GAME;  // 设置游戏状态为开始游戏
                    break;
                // 开始游戏，设置玩家数量、名称和回合数
                case START_GAME:

                    hurkleXPos = randomNumber(); // 生成随机的 hurkleXPos
                    hurkleYPos = randomNumber(); // 生成随机的 hurkleYPos

                    guesses = 1; // 猜测次数初始化为1
                    gameState = GAME_STATE.GUESSING; // 游戏状态设置为猜测中

                    break;

                // 猜测 hurkle 的 x、y 位置
                case GUESSING:
                    String guess = displayTextAndGetInput("GUESS #" + guesses + "? "); // 显示并获取玩家的猜测
                    playerGuessXPos = getDelimitedValue(guess, 0); // 获取玩家猜测的 x 位置
                    playerGuessYPos = getDelimitedValue(guess, 1); // 获取玩家猜测的 y 位置
                    if (foundHurkle()) { // 如果找到了 hurkle
                        gameState = GAME_STATE.PLAY_AGAIN; // 游戏状态设置为再玩一次
                    } else {
# 显示Hurkle的方向
showDirectionOfHurkle();
# 猜测次数加一
guesses++;
# 如果猜测次数超过最大次数
if (guesses > MAX_GUESSES) {
    # 打印超过最大次数的提示信息
    System.out.println("SORRY, THAT'S " + MAX_GUESSES + " GUESSES.");
    # 打印Hurkle的位置
    System.out.println("THE HURKLE IS AT " + hurkleXPos + "," + hurkleYPos);
    System.out.println();
    # 设置游戏状态为再玩一次
    gameState = GAME_STATE.PLAY_AGAIN;
}

# 结束当前的switch语句
break;

# 如果游戏状态为再玩一次
case PLAY_AGAIN:
    # 打印重新开始游戏的提示信息
    System.out.println("LET'S PLAY AGAIN, HURKLE IS HIDING.");
    System.out.println();
    # 设置游戏状态为开始游戏
    gameState = GAME_STATE.START_GAME;
    # 结束当前的switch语句
    break;
            // 游戏状态不是 GAME_OVER 时，循环执行游戏逻辑
        } while (gameState != GAME_STATE.GAME_OVER);
    }

    private void showDirectionOfHurkle() {
        System.out.print("GO ");
        if (playerGuessYPos == hurkleYPos) {
            // 如果玩家选择的 y 坐标与 hurkle 的 y 坐标相同，则不打印 North 或 South
        } else if (playerGuessYPos < hurkleYPos) {
            System.out.print("NORTH");
        } else if (playerGuessYPos > hurkleYPos) {
            System.out.print("SOUTH");
        }

        if (playerGuessXPos == hurkleXPos) {
            // 如果玩家选择的 x 坐标与 hurkle 的 x 坐标相同，则不打印 East 或 West
        } else if (playerGuessXPos < hurkleXPos) {
            System.out.print("EAST"); // 如果玩家猜测的 X 坐标大于 Hurkle 的 X 坐标，则打印"EAST"
        } else if (playerGuessXPos > hurkleXPos) { // 如果玩家猜测的 X 坐标小于 Hurkle 的 X 坐标，则打印"WEST"
            System.out.print("WEST");
        }
        System.out.println(); // 打印换行
    }

    private boolean foundHurkle() { // 检查是否找到 Hurkle
        if ((playerGuessXPos - hurkleXPos) - (playerGuessYPos - hurkleYPos) == 0) { // 如果玩家猜测的位置与 Hurkle 的位置相同，则打印找到 Hurkle 的消息，并返回 true
            System.out.println("YOU FOUND HIM IN " + guesses + " GUESSES.");
            return true;
        }

        return false; // 如果玩家猜测的位置与 Hurkle 的位置不同，则返回 false
    }

    /**
     * Display info about the game
     */
    private void intro() {
        // 打印游戏标题
        System.out.println("HURKLE");
        // 打印游戏信息
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        // 打印游戏信息，包括网格大小
        System.out.println("A HURKLE IS HIDING ON A " + GRID_SIZE + " BY "
                + GRID_SIZE + " GRID. HOMEBASE");
        // 打印游戏信息，包括网格坐标系说明
        System.out.println("ON THE GRID IS POINT 0,0 IN THE SOUTHWEST CORNER,");
        System.out.println("AND ANY POINT ON THE GRID IS DESIGNATED BY A");
        System.out.println("PAIR OF WHOLE NUMBERS SEPERATED BY A COMMA. THE FIRST");
        System.out.println("NUMBER IS THE HORIZONTAL POSITION AND THE SECOND NUMBER");
        System.out.println("IS THE VERTICAL POSITION. YOU MUST TRY TO");
        System.out.println("GUESS THE HURKLE'S GRIDPOINT. YOU GET "
                + MAX_GUESSES + " TRIES.");
        // 打印游戏信息，包括最大猜测次数
        System.out.println("AFTER EACH TRY, I WILL TELL YOU THE APPROXIMATE");
        System.out.println("DIRECTION TO GO TO LOOK FOR THE HURKLE.");
    }

    /**
     * 生成随机数
     * 用于创建 x,y 网格位置的一部分
    /**
     * 生成一个随机数
     * @return 随机数
     */
    private int randomNumber() {
        return (int) (Math.random() * (GRID_SIZE) + 1);
    }

    /*
     * 在屏幕上打印一条消息，然后从键盘接受输入。
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next();
    }

    /**
# 接受一个由逗号分隔的字符串，并返回第pos个被分隔的值（从计数0开始）。
def getDelimitedValue(text, pos):
    # 使用逗号分隔文本，将结果存储在tokens列表中
    tokens = text.split(",")
    # 将tokens列表中第pos个元素转换为整数并返回
    return int(tokens[pos])
```