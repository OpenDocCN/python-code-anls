# `basic-computer-games\64_Nicomachus\java\src\Nicomachus.java`

```py
import java.util.Arrays;  // 导入 Arrays 类
import java.util.Scanner;  // 导入 Scanner 类

/**
 * Nichomachus 游戏
 * <p>
 * 基于这里的 Basic 游戏 Nichomachus
 * https://github.com/coding-horror/basic-computer-games/blob/main/64%20Nicomachus/nicomachus.bas
 * <p>
 * 注意：这个想法是在 Java 中创建一个 1970 年代 Basic 游戏的版本，没有引入新功能 - 没有添加额外的文本、错误检查等。
 */
public class Nicomachus {

    public static final long TWO_SECONDS = 2000;  // 定义常量 TWO_SECONDS 为 2000

    // 用于键盘输入
    private final Scanner kbScanner;  // 声明 Scanner 对象

    private enum GAME_STATE {  // 声明枚举类型 GAME_STATE
        START_GAME,  // 开始游戏
        GET_INPUTS,  // 获取输入
        RESULTS,  // 结果
        PLAY_AGAIN  // 再玩一次
    }

    int remainderNumberDividedBy3;  // 声明整型变量 remainderNumberDividedBy3
    int remainderNumberDividedBy5;  // 声明整型变量 remainderNumberDividedBy5
    int remainderNumberDividedBy7;  // 声明整型变量 remainderNumberDividedBy7

    // 当前游戏状态
    private GAME_STATE gameState;  // 声明 GAME_STATE 类型的变量 gameState

    public Nicomachus() {
        kbScanner = new Scanner(System.in);  // 初始化 Scanner 对象
        gameState = GAME_STATE.START_GAME;  // 初始化游戏状态为 START_GAME
    }

    /**
     * 主游戏循环
     */
    }

    private void intro() {
        System.out.println(addSpaces(33) + "NICOMA");  // 打印 NICOMA
        System.out.println(addSpaces(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 打印 CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY
        System.out.println();  // 打印空行
        System.out.println("BOOMERANG PUZZLE FROM ARITHMETICA OF NICOMACHUS -- A.D. 90!");  // 打印 BOOMERANG PUZZLE FROM ARITHMETICA OF NICOMACHUS -- A.D. 90!
        System.out.println();  // 打印空行
    }

    /*
     * 在屏幕上打印消息，然后从键盘接受输入。
     * 将输入转换为整数
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    private int displayTextAndGetNumber(String text) {
        return Integer.parseInt(displayTextAndGetInput(text));  // 将显示消息并获取输入的内容转换为整数并返回
    }

    /*
     * 在屏幕上打印消息，然后从键盘接受输入。
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    // 显示文本并获取用户输入
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.nextLine();
    }

    /**
     * 返回包含 x 个空格的字符串
     *
     * @param spaces 所需的空格数
     * @return 包含指定数量空格的字符串
     */
    private String addSpaces(int spaces) {
        char[] spacesTemp = new char[spaces];
        Arrays.fill(spacesTemp, ' ');
        return new String(spacesTemp);
    }

    /**
     * 检查玩家是否输入了 Y 或 YES
     *
     * @param text 来自键盘的玩家字符串
     * @return 如果输入了 Y 或 YES，则返回 true，否则返回 false
     */
    private boolean yesEntered(String text) {
        return stringIsAnyValue(text, "Y", "YES");
    }

    /**
     * 检查玩家是否输入了 N 或 NO
     *
     * @param text 来自键盘的玩家字符串
     * @return 如果输入了 N 或 NO，则返回 true，否则返回 false
     */
    private boolean noEntered(String text) {
        return stringIsAnyValue(text, "N", "NO");
    }

    /**
     * 检查字符串是否等于一系列变量值中的任意一个
     * 用于检查是否输入了 Y 或 YES 等情况
     * 比较不区分大小写
     *
     * @param text   源字符串
     * @param values 要与源字符串进行比较的一系列值
     * @return 如果在传递的一系列字符串中找到了匹配，则返回 true
     */
    private boolean stringIsAnyValue(String text, String... values) {
        return Arrays.stream(values).anyMatch(str -> str.equalsIgnoreCase(text));
    }

    public static void main(String[] args) throws Exception {
        // 创建 Nicomachus 对象并开始游戏
        Nicomachus nicomachus = new Nicomachus();
        nicomachus.play();
    }
# 闭合前面的函数定义
```