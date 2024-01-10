# `basic-computer-games\51_Hurkle\java\src\Hurkle.java`

```
import java.util.Scanner;

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

    public static final int GRID_SIZE = 10;  // 定义游戏网格大小为10
    public static final int MAX_GUESSES = 5;  // 定义最大猜测次数为5

    private enum GAME_STATE {  // 定义游戏状态枚举
        STARTING,
        START_GAME,
        GUESSING,
        PLAY_AGAIN,
        GAME_OVER
    }

    private GAME_STATE gameState;  // 当前游戏状态

    // Used for keyboard input
    private final Scanner kbScanner;  // 用于键盘输入的 Scanner 对象

    private int guesses;  // 猜测次数

    // hurkle position
    private int hurkleXPos;  // hurkle 的 X 坐标
    private int hurkleYPos;  // hurkle 的 Y 坐标

    // player guess
    private int playerGuessXPos;  // 玩家猜测的 X 坐标
    private int playerGuessYPos;  // 玩家猜测的 Y 坐标

    public Hurkle() {  // 构造函数

        gameState = GAME_STATE.STARTING;  // 初始化游戏状态为 STARTING

        // Initialise kb scanner
        kbScanner = new Scanner(System.in);  // 初始化键盘输入的 Scanner 对象
    }

    /**
     * Main game loop
     */
    }

    private void showDirectionOfHurkle() {  // 显示 hurkle 的方向
        System.out.print("GO ");
        if (playerGuessYPos == hurkleYPos) {  // 如果玩家选择的 Y 坐标与 hurkle 的 Y 坐标相同
            // don't print North or South because the player has chosen the
            // same y grid pos as the hurkle
        } else if (playerGuessYPos < hurkleYPos) {  // 如果玩家选择的 Y 坐标小于 hurkle 的 Y 坐标
            System.out.print("NORTH");  // 输出 NORTH
        } else if (playerGuessYPos > hurkleYPos) {  // 如果玩家选择的 Y 坐标大于 hurkle 的 Y 坐标
            System.out.print("SOUTH");  // 输出 SOUTH
        }

        if (playerGuessXPos == hurkleXPos) {  // 如果玩家选择的 X 坐标与 hurkle 的 X 坐标相同
            // don't print East or West because the player has chosen the
            // same x grid pos as the hurkle
        } else if (playerGuessXPos < hurkleXPos) {  // 如果玩家选择的 X 坐标小于 hurkle 的 X 坐标
            System.out.print("EAST");  // 输出 EAST
        } else if (playerGuessXPos > hurkleXPos) {  // 如果玩家选择的 X 坐标大于 hurkle 的 X 坐标
            System.out.print("WEST");  // 输出 WEST
        }
        System.out.println();  // 换行
    }
    // 检查玩家是否找到了HURKLE，如果找到则返回true，否则返回false
    private boolean foundHurkle() {
        // 如果玩家猜测的X坐标与HURKLE的X坐标之差，减去玩家猜测的Y坐标与HURKLE的Y坐标之差等于0，则表示找到了HURKLE
        if ((playerGuessXPos - hurkleXPos)
                - (playerGuessYPos - hurkleYPos) == 0) {
            // 打印找到HURKLE的消息，并显示猜测次数，然后返回true
            System.out.println("YOU FOUND HIM IN " + guesses + " GUESSES.");
            return true;
        }
        // 如果没有找到HURKLE，则返回false
        return false;
    }

    /**
     * 显示游戏信息
     */
    private void intro() {
        // 打印游戏标题和信息
        System.out.println("HURKLE");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("A HURKLE IS HIDING ON A " + GRID_SIZE + " BY "
                + GRID_SIZE + " GRID. HOMEBASE");
        System.out.println("ON THE GRID IS POINT 0,0 IN THE SOUTHWEST CORNER,");
        System.out.println("AND ANY POINT ON THE GRID IS DESIGNATED BY A");
        System.out.println("PAIR OF WHOLE NUMBERS SEPERATED BY A COMMA. THE FIRST");
        System.out.println("NUMBER IS THE HORIZONTAL POSITION AND THE SECOND NUMBER");
        System.out.println("IS THE VERTICAL POSITION. YOU MUST TRY TO");
        System.out.println("GUESS THE HURKLE'S GRIDPOINT. YOU GET "
                + MAX_GUESSES + " TRIES.");
        System.out.println("AFTER EACH TRY, I WILL TELL YOU THE APPROXIMATE");
        System.out.println("DIRECTION TO GO TO LOOK FOR THE HURKLE.");
    }

    /**
     * 生成随机数
     * 用于创建x，y网格位置的一部分
     *
     * @return 随机数
     */
    private int randomNumber() {
        // 生成一个0到GRID_SIZE之间的随机整数
        return (int) (Math.random()
                * (GRID_SIZE) + 1);
    }

    /*
     * 在屏幕上打印消息，然后从键盘接受输入。
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    private String displayTextAndGetInput(String text) {
        // 打印文本消息，并从键盘接收输入，然后返回输入的内容
        System.out.print(text);
        return kbScanner.next();
    }
    /**
     * 接受一个由逗号分隔的字符串，并返回第 pos 个分隔值（从计数 0 开始）。
     *
     * @param text - 由逗号分隔的文本
     * @param pos  - 要返回值的位置
     * @return 值的整数表示
     */
    private int getDelimitedValue(String text, int pos) {
        // 使用逗号分割字符串，得到分隔后的字符串数组
        String[] tokens = text.split(",");
        // 将指定位置的分隔值转换为整数并返回
        return Integer.parseInt(tokens[pos]);
    }
# 闭合前面的函数定义
```