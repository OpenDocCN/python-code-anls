# `basic-computer-games\25_Chief\java\src\Chief.java`

```py
import java.util.Arrays;  // 导入 Arrays 类
import java.util.Scanner;  // 导入 Scanner 类

/**
 * Game of Chief
 * <p>
 * Based on the Basic game of Hurkle here
 * https://github.com/coding-horror/basic-computer-games/blob/main/25%20Chief/chief.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Chief {

    private enum GAME_STATE {  // 定义游戏状态枚举
        STARTING,  // 初始状态
        READY_TO_START,  // 准备开始状态
        ENTER_NUMBER,  // 输入数字状态
        CALCULATE_AND_SHOW,  // 计算并展示状态
        END_GAME,  // 游戏结束状态
        GAME_OVER  // 游戏失败状态
    }

    private GAME_STATE gameState;  // 当前游戏状态

    // The number the computer determines to be the players starting number
    private double calculatedNumber;  // 计算出的玩家起始数字

    // Used for keyboard input
    private final Scanner kbScanner;  // 键盘输入扫描器

    public Chief() {
        gameState = GAME_STATE.STARTING;  // 初始化游戏状态为初始状态

        // Initialise kb scanner
        kbScanner = new Scanner(System.in);  // 初始化键盘输入扫描器
    }

    /**
     * Main game loop
     */
    }

    /**
     * Simulate tabs by building up a string of spaces
     *
     * @param spaces how many spaces are there to be
     * @return a string with the requested number of spaces
     */
    private String tabbedSpaces(int spaces) {  // 模拟制表符，构建包含指定数量空格的字符串
        char[] repeat = new char[spaces];  // 创建指定长度的字符数组
        Arrays.fill(repeat, ' ');  // 用空格填充字符数组
        return new String(repeat);  // 返回包含指定数量空格的字符串
    }

    private void instructions() {  // 输出游戏指令
        System.out.println(" TAKE A NUMBER AND ADD 3. DIVIDE NUMBER BY 5 AND");
        System.out.println("MULTIPLY BY 8. DIVIDE BY 5 AND ADD THE SAME. SUBTRACT 1.");
    }

    /**
     * Basic information about the game
     */
    private void intro() {  // 输出游戏介绍
        System.out.println("CHIEF");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("I AM CHIEF NUMBERS FREEK, THE GREAT INDIAN MATH GOD.");
    }
}
    /**
     * 如果给定的字符串等于调用stringIsAnyValue方法中指定的至少一个值，则返回true
     *
     * @param text 要搜索的字符串
     * @return 如果字符串等于varargs中的一个，则返回true
     */
    private boolean yesEntered(String text) {
        return stringIsAnyValue(text, "Y", "YES");
    }

    /**
     * 如果给定的字符串包含varargs（第二个参数）中的至少一个，则返回true
     * 注意：不区分大小写
     *
     * @param text   要搜索的字符串
     * @param values 包含要比较的值的字符串类型的varargs
     * @return 如果在文本中找到varargs参数中的一个，则返回true
     */
    private boolean stringIsAnyValue(String text, String... values) {

        // 循环遍历可变数量的值，并测试每个值
        for (String val : values) {
            if (text.equalsIgnoreCase(val)) {
                return true;
            }
        }

        // 没有匹配项
        return false;
    }

    /*
     * 在屏幕上打印消息，然后从键盘接受输入。
     *
     * @param text 要在屏幕上显示的消息。
     * @return 玩家输入的内容。
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next();
    }
# 闭合前面的函数定义
```