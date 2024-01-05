# `25_Chief\java\src\Chief.java`

```
import java.util.Arrays;  # 导入 java.util.Arrays 包
import java.util.Scanner;  # 导入 java.util.Scanner 包

/**
 * Game of Chief
 * <p>
 * Based on the Basic game of Hurkle here
 * https://github.com/coding-horror/basic-computer-games/blob/main/25%20Chief/chief.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Chief {  # 定义一个名为 Chief 的类

    private enum GAME_STATE {  # 定义一个名为 GAME_STATE 的枚举类型
        STARTING,  # 枚举值 STARTING
        READY_TO_START,  # 枚举值 READY_TO_START
        ENTER_NUMBER,  # 枚举值 ENTER_NUMBER
        CALCULATE_AND_SHOW,  # 枚举值 CALCULATE_AND_SHOW
        END_GAME,  # 枚举值 END_GAME
        GAME_OVER
    }
```
这是一个代码片段的结尾，可能是一个类或者方法的结束。

```
    private GAME_STATE gameState;
```
声明了一个私有的枚举类型变量gameState，用来表示游戏的状态。

```
    // The number the computer determines to be the players starting number
    private double calculatedNumber;
```
声明了一个私有的double类型变量calculatedNumber，用来存储计算机确定的玩家的起始数字。

```
    // Used for keyboard input
    private final Scanner kbScanner;
```
声明了一个私有的final类型的Scanner对象kbScanner，用来处理键盘输入。

```
    public Chief() {
```
这是一个构造方法的声明，用来初始化Chief类的实例。

```
        gameState = GAME_STATE.STARTING;
```
在构造方法中，将gameState初始化为GAME_STATE枚举类型的STARTING状态。

```
        // Initialise kb scanner
        kbScanner = new Scanner(System.in);
```
在构造方法中，初始化kbScanner对象，用来处理键盘输入。

```
    /**
```
这是一个注释的开始，用来说明接下来的代码或者方法的作用。
    /**
     * 游戏的主循环
     */
    public void play() {

        do {
            switch (gameState) {

                // 第一次玩游戏时显示介绍
                case STARTING:
                    intro();
                    gameState = GAME_STATE.READY_TO_START;
                    break;

                // 显示开始消息
                case READY_TO_START:
                    if (!yesEntered(displayTextAndGetInput("ARE YOU READY TO TAKE THE TEST YOU CALLED ME OUT FOR? "))) {
                        System.out.println("SHUT UP, PALE FACE WITH WISE TONGUE.");
                    }

                    instructions();
                    gameState = GAME_STATE.ENTER_NUMBER;  // 设置游戏状态为输入数字状态
                    break;  // 结束当前 case，跳出 switch 语句

                // 输入要用来计算的数字
                case ENTER_NUMBER:
                    double playerNumber = Double.parseDouble(
                            displayTextAndGetInput(" WHAT DO YOU HAVE? "));  // 从用户输入中获取要计算的数字

                    // 使用原始游戏中相同的公式来计算玩家的原始数字
                    calculatedNumber = (playerNumber + 1 - 5) * 5 / 8 * 5 - 3;  // 使用公式计算玩家的数字

                    gameState = GAME_STATE.CALCULATE_AND_SHOW;  // 设置游戏状态为计算并展示状态
                    break;  // 结束当前 case，跳出 switch 语句

                // 计算并展示结果
                case CALCULATE_AND_SHOW:
                    if (yesEntered(
                            displayTextAndGetInput("I BET YOUR NUMBER WAS " + calculatedNumber
                                    + ". AM I RIGHT? "))) {  // 显示计算结果并询问玩家是否正确
                        gameState = GAME_STATE.END_GAME;  // 如果玩家确认正确，则设置游戏状态为结束游戏状态
                        } else {
                            // 玩家不同意，所以显示计算过程
                            double number = Double.parseDouble(
                                    displayTextAndGetInput(" WHAT WAS YOUR ORIGINAL NUMBER? "));
                            double f = number + 3;  // 将输入的数字加3
                            double g = f / 5;  // 将结果除以5
                            double h = g * 8;  // 将结果乘以8
                            double i = h / 5 + 5;  // 将结果除以5并加5
                            double j = i - 1;  // 将结果减1
                            System.out.println("SO YOU THINK YOU'RE SO SMART, EH?");  // 输出提示信息
                            System.out.println("NOW WATCH.");  // 输出提示信息
                            System.out.println(number + " PLUS 3 EQUALS " + f + ". DIVIDED BY 5 EQUALS " + g);  // 输出计算过程
                            System.out.println("TIMES 8 EQUALS " + h + ". IF WE DIVIDE BY 5 AND ADD 5,");  // 输出计算过程
                            System.out.println("WE GET " + i + ", WHICH, MINUS 1, EQUALS " + j + ".");  // 输出计算过程
                            if (yesEntered(displayTextAndGetInput("NOW DO YOU BELIEVE ME? "))) {  // 如果玩家同意
                                gameState = GAME_STATE.END_GAME;  // 设置游戏状态为结束
                            } else {
                                // Time for a lightning bolt.
                                System.out.println("YOU HAVE MADE ME MAD!!!");  // 输出提示信息
                            // 打印提示信息
                            System.out.println("THERE MUST BE A GREAT LIGHTNING BOLT!");
                            System.out.println();
                            // 打印闪电图案
                            for (int x = 30; x >= 22; x--) {
                                System.out.println(tabbedSpaces(x) + "X X");
                            }
                            System.out.println(tabbedSpaces(21) + "X XXX");
                            System.out.println(tabbedSpaces(20) + "X   X");
                            System.out.println(tabbedSpaces(19) + "XX X");
                            for (int y = 20; y >= 13; y--) {
                                System.out.println(tabbedSpaces(y) + "X X");
                            }
                            System.out.println(tabbedSpaces(12) + "XX");
                            System.out.println(tabbedSpaces(11) + "X");
                            System.out.println(tabbedSpaces(10) + "*");
                            System.out.println();
                            // 打印分隔线
                            System.out.println("#########################");
                            System.out.println();
                            // 打印提示信息
                            System.out.println("I HOPE YOU BELIEVE ME NOW, FOR YOUR SAKE!!");
                            // 更新游戏状态为游戏结束
                            gameState = GAME_STATE.GAME_OVER;
                        }
                    }
                    break;

                // Sign off message for cases where the Chief is not upset
                case END_GAME:
                    System.out.println("BYE!!!");
                    gameState = GAME_STATE.GAME_OVER;
                    break;

                // GAME_OVER State does not specifically have a case
            }
        } while (gameState != GAME_STATE.GAME_OVER);
    }

    /**
     * Simulate tabs by building up a string of spaces
     *
     * @param spaces how many spaces are there to be
     * @return a string with the requested number of spaces
``` 

在这段代码中，我们看到了一个switch语句，它根据不同的游戏状态执行不同的操作。在每个case下面，有相应的注释来解释这个case的作用。在do-while循环中，游戏状态被检查，如果游戏状态不是GAME_OVER，循环将继续执行。在注释下面，有一个方法的注释，解释了这个方法的作用和参数。
    */
    // 创建一个函数，用于生成指定数量的空格字符串
    private String tabbedSpaces(int spaces) {
        char[] repeat = new char[spaces];
        Arrays.fill(repeat, ' ');  // 用空格填充字符数组
        return new String(repeat);  // 将字符数组转换为字符串并返回
    }

    // 显示游戏的指令
    private void instructions() {
        System.out.println(" TAKE A NUMBER AND ADD 3. DIVIDE NUMBER BY 5 AND");
        System.out.println("MULTIPLY BY 8. DIVIDE BY 5 AND ADD THE SAME. SUBTRACT 1.");
    }

    /**
     * 游戏的基本信息
     */
    // 显示游戏的介绍信息
    private void intro() {
        System.out.println("CHIEF");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("I AM CHIEF NUMBERS FREEK, THE GREAT INDIAN MATH GOD.");
    }

    /**
     * 如果给定的字符串等于调用stringIsAnyValue方法中指定的至少一个值，则返回true
     *
     * @param text 要搜索的字符串
     * @return 如果字符串等于varargs中的一个值，则返回true
     */
    private boolean yesEntered(String text) {
        return stringIsAnyValue(text, "Y", "YES");
    }

    /**
     * 如果给定的字符串包含至少一个varargs（第二个参数）中的值，则返回true。
     * 注意：不区分大小写比较。
     *
     * @param text   要搜索的字符串
     * @param values 包含要比较的值的字符串类型的varargs
     * @return 如果在文本中找到varargs参数中的一个，则返回true
    // 定义一个方法，用于判断一个字符串是否等于给定的多个值中的任意一个
    private boolean stringIsAnyValue(String text, String... values) {

        // 遍历给定的多个值，逐个与输入的字符串进行比较
        for (String val : values) {
            if (text.equalsIgnoreCase(val)) { // 如果找到匹配的值，则返回true
                return true;
            }
        }

        // 如果没有找到匹配的值，则返回false
        return false;
    }

    /*
     * 在屏幕上打印一条消息，然后从键盘接受输入。
     *
     * @param text 要在屏幕上显示的消息。
     * @return 玩家输入的内容。
     */
    # 定义一个名为displayTextAndGetInput的私有方法，接受一个字符串参数text
    def displayTextAndGetInput(text):
        # 打印文本内容
        print(text)
        # 从键盘输入获取用户输入并返回
        return input()
```