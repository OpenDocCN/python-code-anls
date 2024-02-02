# `basic-computer-games\52_Kinema\java\src\Kinema.java`

```py
// 导入 java.util.Arrays 包，用于数组操作
// 导入 java.util.Scanner 包，用于键盘输入
/**
 * Kinema 游戏
 * 基于 Basic 版本的 Kinema 游戏
 * https://github.com/coding-horror/basic-computer-games/blob/main/52%20Kinema/kinema.bas
 * 注意：这个想法是在 Java 中创建一个 1970 年代 Basic 游戏的版本，没有引入新功能 - 没有添加额外的文本、错误检查等。
 */
public class Kinema {

    // 用于键盘输入
    private final Scanner kbScanner;

    // 当前游戏状态
    private GAME_STATE gameState;

    // 答对的数量
    private int numberAnswersCorrect;

    // 抛出球的速度（每秒米数）
    private int velocity;

    public Kinema() {
        kbScanner = new Scanner(System.in);

        // 初始化游戏状态为 STARTUP
        gameState = GAME_STATE.STARTUP;
    }

    /**
     * 主游戏循环
     */
    }

    // 游戏介绍
    private void intro() {
        System.out.println(simulateTabs(33) + "KINEMA");
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
    }

    // 计算玩家答案是否正确
    private boolean calculate(double playerAnswer, double correctAnswer) {

        boolean gotItRight = false;

        // 如果玩家答案与正确答案的相对误差小于 0.15，则判定为正确
        if (Math.abs((playerAnswer - correctAnswer) / correctAnswer) < 0.15) {
            System.out.println("CLOSE ENOUGH");
            gotItRight = true;
        } else {
            System.out.println("NOT EVEN CLOSE");
        }
        System.out.println("CORRECT ANSWER IS " + correctAnswer);
        System.out.println();

        return gotItRight;
    }

    /*
     * 在屏幕上打印消息，然后从键盘接受输入。
     * 将输入转换为 Double 类型
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    # 根据文本显示消息并从键盘获取输入，将输入转换为double类型并返回
    private double displayTextAndGetNumber(String text) {
        return Double.parseDouble(displayTextAndGetInput(text));
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

    /**
     * 模拟旧的基本tab(xx)命令，该命令通过xx个空格缩进文本。
     *
     * @param spaces 需要的空格数
     * @return 具有空格数的字符串
     */
    private String simulateTabs(int spaces) {
        char[] spacesTemp = new char[spaces];
        Arrays.fill(spacesTemp, ' ');
        return new String(spacesTemp);
    }
# 闭合前面的函数定义
```