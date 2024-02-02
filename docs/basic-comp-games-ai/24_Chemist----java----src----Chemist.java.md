# `basic-computer-games\24_Chemist\java\src\Chemist.java`

```py
// 导入必要的类
import java.util.Arrays;
import java.util.Scanner;

/**
 * 化学家游戏
 * <p>
 * 基于这里的基本化学家游戏
 * https://github.com/coding-horror/basic-computer-games/blob/main/24%20Chemist/chemist.bas
 * <p>
 * 注意：这个想法是在Java中创建一个1970年代Basic游戏的版本，没有引入新功能-没有添加额外的文本，错误检查等。
 */
public class Chemist {

    public static final int MAX_LIVES = 9;

    // 用于键盘输入
    private final Scanner kbScanner;

    private enum GAME_STATE {
        START_GAME,
        INPUT,
        BLOWN_UP,
        SURVIVED,
        GAME_OVER
    }

    // 当前游戏状态
    private GAME_STATE gameState;

    private int timesBlownUp;

    public Chemist() {
        kbScanner = new Scanner(System.in);

        gameState = GAME_STATE.START_GAME;
    }

    /**
     * 主游戏循环
     */
    }

    private void intro() {
        System.out.println(simulateTabs(33) + "CHEMIST");
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("THE FICTITIOUS CHEMICAL KRYPTOCYANIC ACID CAN ONLY BE");
        System.out.println("DILUTED BY THE RATIO OF 7 PARTS WATER TO 3 PARTS ACID.");
        System.out.println("IF ANY OTHER RATIO IS ATTEMPTED, THE ACID BECOMES UNSTABLE");
        System.out.println("AND SOON EXPLODES.  GIVEN THE AMOUNT OF ACID, YOU MUST");
        System.out.println("DECIDE WHO MUCH WATER TO ADD FOR DILUTION.  IF YOU MISS");
        System.out.println("YOU FACE THE CONSEQUENCES.");
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
    /*
     * 在屏幕上打印一条消息，然后接受键盘输入。
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    private String displayTextAndGetInput(String text) {
        // 在屏幕上打印消息
        System.out.print(text);
        // 从键盘输入并返回
        return kbScanner.next();
    }

    /**
     * 模拟旧的基本tab(xx)命令，该命令通过xx个空格缩进文本。
     *
     * @param spaces 需要的空格数
     * @return 包含指定数量空格的字符串
     */
    private String simulateTabs(int spaces) {
        // 创建一个包含指定数量空格的字符数组
        char[] spacesTemp = new char[spaces];
        // 用空格填充数组
        Arrays.fill(spacesTemp, ' ');
        // 将字符数组转换为字符串并返回
        return new String(spacesTemp);
    }
# 闭合前面的函数定义
```