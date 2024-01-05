# `d:/src/tocomm/basic-computer-games\64_Nicomachus\java\src\Nicomachus.java`

```
import java.util.Arrays;  # 导入 java.util.Arrays 包
import java.util.Scanner;  # 导入 java.util.Scanner 包

/**
 * Game of Nichomachus
 * <p>
 * Based on the Basic game of Nichomachus here
 * https://github.com/coding-horror/basic-computer-games/blob/main/64%20Nicomachus/nicomachus.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */

public class Nicomachus {

    public static final long TWO_SECONDS = 2000;  # 定义一个名为 TWO_SECONDS 的常量，值为 2000

    // Used for keyboard input
    private final Scanner kbScanner;  # 声明一个名为 kbScanner 的私有 Scanner 对象
    private enum GAME_STATE {  # 定义一个枚举类型，表示游戏的不同状态
        START_GAME,  # 开始游戏
        GET_INPUTS,  # 获取输入
        RESULTS,     # 显示结果
        PLAY_AGAIN   # 再玩一次
    }

    int remainderNumberDividedBy3;  # 用于存储被3除的余数
    int remainderNumberDividedBy5;  # 用于存储被5除的余数
    int remainderNumberDividedBy7;  # 用于存储被7除的余数

    // Current game state  # 当前游戏状态
    private GAME_STATE gameState;  # 声明一个私有的游戏状态变量

    public Nicomachus() {  # Nicomachus 类的构造函数
        kbScanner = new Scanner(System.in);  # 创建一个用于接收用户输入的 Scanner 对象
        gameState = GAME_STATE.START_GAME;  # 将游戏状态设置为开始游戏
    }

    /**
    * Main game loop
     */
    public void play() throws Exception {

        do {
            switch (gameState) {

                case START_GAME:
                    // 调用intro()函数，开始游戏
                    intro();
                    // 将游戏状态设置为获取输入状态
                    gameState = GAME_STATE.GET_INPUTS;
                    break;

                case GET_INPUTS:
                    // 提示玩家想一个1到100之间的数字
                    System.out.println("PLEASE THINK OF A NUMBER BETWEEN 1 AND 100.");
                    // 获取玩家输入的数字对3取余的结果
                    remainderNumberDividedBy3 = displayTextAndGetNumber("YOUR NUMBER DIVIDED BY 3 HAS A REMAINDER OF? ");
                    // 获取玩家输入的数字对5取余的结果
                    remainderNumberDividedBy5 = displayTextAndGetNumber("YOUR NUMBER DIVIDED BY 5 HAS A REMAINDER OF? ");
                    // 获取玩家输入的数字对7取余的结果
                    remainderNumberDividedBy7 = displayTextAndGetNumber("YOUR NUMBER DIVIDED BY 7 HAS A REMAINDER OF? ");
                    // 将游戏状态设置为结果状态
                    gameState = GAME_STATE.RESULTS;
                case RESULTS: // 处理结果的情况
                    System.out.println("LET ME THINK A MOMENT..."); // 打印提示信息

                    // 模拟基本程序的 for/next 循环以延迟事情。
                    // 这里我们睡眠一秒钟。
                    Thread.sleep(TWO_SECONDS); // 线程休眠两秒钟

                    // 计算玩家所想的数字。
                    int answer = (70 * remainderNumberDividedBy3) + (21 * remainderNumberDividedBy5)
                            + (15 * remainderNumberDividedBy7); // 计算玩家所想的数字

                    // 原始基本程序中类似的操作
                    // （测试答案是否为105，并减去105直到答案小于等于105）
                    while (answer > 105) { // 当答案大于105时
                        answer -= 105; // 减去105
                    }

                    do {
                        String input = displayTextAndGetInput("YOUR NUMBER WAS " + answer + ", RIGHT? "); // 显示提示信息并获取输入
                        if (yesEntered(input)) { // 如果输入是“是”
                            System.out.println("HOW ABOUT THAT!!");
                            // 打印输出"HOW ABOUT THAT!!"
                            break;
                            // 跳出当前循环
                        } else if (noEntered(input)) {
                            System.out.println("I FEEL YOUR ARITHMETIC IS IN ERROR.");
                            // 如果输入的是"no"，则打印输出"I FEEL YOUR ARITHMETIC IS IN ERROR."
                            break;
                            // 跳出当前循环
                        } else {
                            System.out.println("EH?  I DON'T UNDERSTAND '" + input + "'  TRY 'YES' OR 'NO'.");
                            // 如果输入的既不是"yes"也不是"no"，则打印输出"EH?  I DON'T UNDERSTAND '" + input + "'  TRY 'YES' OR 'NO'."
                        }
                    } while (true);
                    // 无限循环，直到遇到break语句

                    gameState = GAME_STATE.PLAY_AGAIN;
                    // 将游戏状态设置为PLAY_AGAIN
                    break;
                    // 跳出当前循环

                case PLAY_AGAIN:
                    System.out.println("LET'S TRY ANOTHER");
                    // 打印输出"LET'S TRY ANOTHER"
                    gameState = GAME_STATE.GET_INPUTS;
                    // 将游戏状态设置为GET_INPUTS
                    break;
                    // 跳出当前循环
            }

            // Original basic program looped until CTRL-C
            // 原始的基本程序循环直到按下CTRL-C
    } while (true);
```
这是一个do-while循环的结束标志。

```
    private void intro() {
        System.out.println(addSpaces(33) + "NICOMA");
        System.out.println(addSpaces(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("BOOMERANG PUZZLE FROM ARITHMETICA OF NICOMACHUS -- A.D. 90!");
        System.out.println();
    }
```
这是一个intro()方法，用于在屏幕上打印一些信息。

```
    /*
     * Print a message on the screen, then accept input from Keyboard.
     * Converts input to an Integer
     *
     * @param text message to be displayed on screen.
     * @return what was typed by the player.
     */
    private int displayTextAndGetNumber(String text) {
        return Integer.parseInt(displayTextAndGetInput(text));
```
这是一个displayTextAndGetNumber()方法，用于在屏幕上打印消息并接受键盘输入，然后将输入转换为整数并返回。
    }

    /*
     * 在屏幕上打印一条消息，然后接受键盘输入。
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.nextLine();
    }

    /**
     * 返回一个包含 x 个空格的字符串
     *
     * @param spaces 所需的空格数
     * @return 包含指定数量空格的字符串
     */
    private String addSpaces(int spaces) {
        char[] spacesTemp = new char[spaces];  // 创建一个字符数组，长度为变量 spaces 的值
        Arrays.fill(spacesTemp, ' ');  // 使用空格填充字符数组
        return new String(spacesTemp);  // 将字符数组转换为字符串并返回
    }

    /**
     * 检查玩家是否输入了 Y 或 YES 作为答案
     *
     * @param text 玩家从键盘输入的字符串
     * @return 如果输入了 Y 或 YES，则返回 true，否则返回 false
     */
    private boolean yesEntered(String text) {
        return stringIsAnyValue(text, "Y", "YES");  // 调用 stringIsAnyValue 方法，检查输入的字符串是否为 "Y" 或 "YES"
    }

    /**
     * 检查玩家是否输入了 N 或 NO 作为答案
     *
     * @param text 玩家从键盘输入的字符串
     * @return 如果输入了 N 或 NO，则返回 true，否则返回 false
    */
    // 检查字符串是否为"NO"或"N"，不区分大小写
    private boolean noEntered(String text) {
        return stringIsAnyValue(text, "N", "NO");
    }

    /**
     * 检查字符串是否等于一系列变量数量的值
     * 用于检查例如"Y"或"YES"
     * 比较不区分大小写
     *
     * @param text   源字符串
     * @param values 要与源字符串进行比较的一系列值
     * @return 如果在传递的一系列字符串中找到了匹配，则返回true
     */
    private boolean stringIsAnyValue(String text, String... values) {
        // 使用流的方式，检查是否有任何一个值与源字符串匹配
        return Arrays.stream(values).anyMatch(str -> str.equalsIgnoreCase(text));
    }

    public static void main(String[] args) throws Exception {
# 创建一个名为Nicomachus的对象实例
Nicomachus nicomachus = new Nicomachus();
# 调用Nicomachus对象的play方法
nicomachus.play();
# 结束main方法
}
# 结束类定义
}
```