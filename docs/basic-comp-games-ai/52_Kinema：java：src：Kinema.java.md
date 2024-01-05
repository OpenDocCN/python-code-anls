# `52_Kinema\java\src\Kinema.java`

```
import java.util.Arrays;  # 导入 Arrays 类
import java.util.Scanner;  # 导入 Scanner 类

/**
 * Game of Kinema
 * <p>
 * Based on the Basic game of Kinema here
 * https://github.com/coding-horror/basic-computer-games/blob/main/52%20Kinema/kinema.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Kinema {

    // Used for keyboard input
    private final Scanner kbScanner;  # 声明一个私有的 Scanner 对象用于键盘输入

    private enum GAME_STATE {  # 声明一个枚举类型 GAME_STATE
        STARTUP,  # 游戏状态为启动
        INIT,  # 游戏状态为初始化
        HOW_HIGH,  // 声明一个枚举类型，表示球的高度
        SECONDS_TILL_IT_RETURNS,  // 声明一个枚举类型，表示球返回所需的时间
        ITS_VELOCITY,  // 声明一个枚举类型，表示球的速度
        RESULTS,  // 声明一个枚举类型，表示游戏结果
        GAME_OVER  // 声明一个枚举类型，表示游戏结束
    }

    // 当前游戏状态
    private GAME_STATE gameState;  // 声明一个私有变量，表示当前游戏状态

    private int numberAnswersCorrect;  // 声明一个私有变量，表示回答正确的数量

    // 每秒多少米的速度抛出球
    private int velocity;  // 声明一个私有变量，表示球的速度

    public Kinema() {  // 构造函数
        kbScanner = new Scanner(System.in);  // 创建一个键盘输入的扫描器对象

        gameState = GAME_STATE.STARTUP;  // 初始化游戏状态为启动状态
    }
    /**
     * Main game loop
     */
    public void play() {

        double playerAnswer; // 定义玩家的答案变量
        double correctAnswer; // 定义正确答案变量
        do {
            switch (gameState) { // 根据游戏状态进行不同的操作

                case STARTUP: // 游戏刚开始时的操作
                    intro(); // 调用intro()函数，显示游戏介绍
                    gameState = GAME_STATE.INIT; // 将游戏状态设置为INIT
                    break;

                case INIT: // 初始化游戏状态
                    numberAnswersCorrect = 0; // 将正确答案数量初始化为0

                    // calculate a random velocity for the player to use in the calculations
                    // 为玩家计算一个随机速度，用于后续的计算
                    velocity = 5 + (int) (35 * Math.random());  // 生成一个随机的速度值，范围在5到40之间
                    System.out.println("A BALL IS THROWN UPWARDS AT " + velocity + " METERS PER SECOND.");  // 打印出球以多少米每秒的速度向上抛出
                    gameState = GAME_STATE.HOW_HIGH;  // 将游戏状态设置为求高度阶段
                    break;  // 跳出switch语句

                case HOW_HIGH:

                    playerAnswer = displayTextAndGetNumber("HOW HIGH WILL IT GO (IN METERS)? ");  // 显示并获取玩家对球的最高高度的猜测

                    // Calculate the correct answer to how high it will go
                    correctAnswer = 0.05 * Math.pow(velocity, 2);  // 计算球的最高高度的正确答案
                    if (calculate(playerAnswer, correctAnswer)) {  // 如果玩家的答案与正确答案相符
                        numberAnswersCorrect++;  // 答对的次数加一
                    }
                    gameState = GAME_STATE.ITS_VELOCITY;  // 将游戏状态设置为求速度阶段
                    break;  // 跳出switch语句

                case ITS_VELOCITY:

                    playerAnswer = displayTextAndGetNumber("HOW LONG UNTIL IT RETURNS (IN SECONDS)? ");  // 显示并获取玩家对球返回所需时间的猜测
                    // 计算当前答案，即多久后物体返回地面的时间（单位：秒）
                    correctAnswer = (double) velocity / 5;
                    // 如果玩家答案正确，则增加正确答案数量
                    if (calculate(playerAnswer, correctAnswer)) {
                        numberAnswersCorrect++;
                    }
                    // 切换游戏状态到SECONDS_TILL_IT_RETURNS
                    gameState = GAME_STATE.SECONDS_TILL_IT_RETURNS;
                    break;

                case SECONDS_TILL_IT_RETURNS:

                    // 计算第三个问题的随机秒数
                    double seconds = 1 + (Math.random() * (2 * velocity)) / 10;

                    // 四舍五入到小数点后一位
                    double scale = Math.pow(10, 1);
                    seconds = Math.round(seconds * scale) / scale;

                    // 显示并获取玩家对问题的答案
                    playerAnswer = displayTextAndGetNumber("WHAT WILL ITS VELOCITY BE AFTER " + seconds + " SECONDS? ");
                    // 计算给定秒数后的速度
                    correctAnswer = velocity - (10 * seconds);
                    // 如果玩家答案正确，增加正确答案计数
                    if (calculate(playerAnswer, correctAnswer)) {
                        numberAnswersCorrect++;
                    }
                    // 切换游戏状态为结果
                    gameState = GAME_STATE.RESULTS;
                    break;

                case RESULTS:
                    // 打印出答对的数量
                    System.out.println(numberAnswersCorrect + " RIGHT OUT OF 3");
                    // 如果答对的数量大于1，打印出"不错"
                    if (numberAnswersCorrect > 1) {
                        System.out.println(" NOT BAD.");
                    }
                    // 切换游戏状态为开始
                    gameState = GAME_STATE.STARTUP;
                    break;
            }
        } while (gameState != GAME_STATE.GAME_OVER);
    }

    private void intro() {
        System.out.println(simulateTabs(33) + "KINEMA");  // 在控制台输出带有33个制表符的空格，然后输出"KINEMA"
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 在控制台输出带有15个制表符的空格，然后输出"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
        System.out.println();  // 在控制台输出一个空行
    }

    private boolean calculate(double playerAnswer, double correctAnswer) {

        boolean gotItRight = false;  // 初始化一个布尔变量gotItRight为false

        if (Math.abs((playerAnswer - correctAnswer) / correctAnswer) < 0.15) {  // 如果玩家答案与正确答案的相对误差小于0.15
            System.out.println("CLOSE ENOUGH");  // 在控制台输出"CLOSE ENOUGH"
            gotItRight = true;  // 将gotItRight设置为true
        } else {
            System.out.println("NOT EVEN CLOSE");  // 在控制台输出"NOT EVEN CLOSE"
        }
        System.out.println("CORRECT ANSWER IS " + correctAnswer);  // 在控制台输出"CORRECT ANSWER IS "后跟正确答案的值
        System.out.println();  // 在控制台输出一个空行

        return gotItRight;  // 返回gotItRight的值作为函数的结果
    }
    /*
     * 在屏幕上打印一条消息，然后从键盘接受输入。
     * 将输入转换为双精度数
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    private double displayTextAndGetNumber(String text) {
        return Double.parseDouble(displayTextAndGetInput(text));
    }

    /*
     * 在屏幕上打印一条消息，然后从键盘接受输入。
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next();  # 从键盘输入中获取下一个字符串

    /**
     * 模拟旧的基本tab(xx)命令，通过xx个空格缩进文本。
     *
     * @param spaces 所需的空格数
     * @return 具有空格数的字符串
     */
    private String simulateTabs(int spaces) {
        char[] spacesTemp = new char[spaces];  # 创建一个包含指定空格数的字符数组
        Arrays.fill(spacesTemp, ' ');  # 使用空格填充字符数组
        return new String(spacesTemp);  # 将字符数组转换为字符串并返回
    }
}
```