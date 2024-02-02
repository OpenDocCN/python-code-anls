# `basic-computer-games\16_Bug\java\src\Bug.java`

```py
// 导入必要的类
import java.util.ArrayList;
import java.util.Scanner;

/**
 * Bug 游戏
 * <p>
 * 基于 Basic 游戏 Bug 的 Java 版本
 * https://github.com/coding-horror/basic-computer-games/blob/main/16%20Bug/bug.bas
 * <p>
 * 注意：这个想法是在 Java 中创建一个 1970 年代 Basic 游戏的版本，没有引入新功能 - 没有添加额外的文本、错误检查等。
 */
public class Bug {

    // 掷骰子
    public static final int SIX = 6;

    // 用于键盘输入
    private final Scanner kbScanner;

    // 当前游戏状态
    private GAME_STATE gameState;

    // 玩家的虫子
    private final Insect playersBug;

    // 电脑的虫子
    private final Insect computersBug;

    // 用于显示骰子结果
    private final String[] ROLLS = new String[]{"BODY", "NECK", "HEAD", "FEELERS", "TAIL", "LEGS"};

    public Bug() {
        // 初始化玩家和电脑的虫子
        playersBug = new PlayerBug();
        computersBug = new ComputerBug();

        // 设置游戏状态为开始
        gameState = GAME_STATE.START;

        // 初始化键盘输入扫描器
        kbScanner = new Scanner(System.in);
    }

    /**
     * 主游戏循环
     */
    }

    /**
     * 根据已添加的部分绘制虫子（玩家或电脑）
     *
     * @param bug 要绘制的虫子
     */
    private void draw(Insect bug) {
        // 获取虫子的绘制结果
        ArrayList<String> insectOutput = bug.draw();
        // 遍历并打印绘制结果
        for (String s : insectOutput) {
            System.out.println(s);
        }
    }

    /**
     * 显示介绍
     */
    private void intro() {
        System.out.println("BUG");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("THE GAME BUG");
        System.out.println("I HOPE YOU ENJOY THIS GAME.");
    }
    // 显示游戏说明
    private void instructions() {
        System.out.println("THE OBJECT OF BUG IS TO FINISH YOUR BUG BEFORE I FINISH");
        System.out.println("MINE. EACH NUMBER STANDS FOR A PART OF THE BUG BODY.");
        System.out.println("I WILL ROLL THE DIE FOR YOU, TELL YOU WHAT I ROLLED FOR YOU");
        System.out.println("WHAT THE NUMBER STANDS FOR, AND IF YOU CAN GET THE PART.");
        System.out.println("IF YOU CAN GET THE PART I WILL GIVE IT TO YOU.");
        System.out.println("THE SAME WILL HAPPEN ON MY TURN.");
        System.out.println("IF THERE IS A CHANGE IN EITHER BUG I WILL GIVE YOU THE");
        System.out.println("OPTION OF SEEING THE PICTURES OF THE BUGS.");
        System.out.println("THE NUMBERS STAND FOR PARTS AS FOLLOWS:");
        System.out.println("NUMBER\tPART\tNUMBER OF PART NEEDED");
        System.out.println("1\tBODY\t1");
        System.out.println("2\tNECK\t1");
        System.out.println("3\tHEAD\t1");
        System.out.println("4\tFEELERS\t2");
        System.out.println("5\tTAIL\t1");
        System.out.println("6\tLEGS\t6");
        System.out.println();
    }

    /**
     * 检查玩家是否输入了N或NO作为答案
     *
     * @param text 玩家从键盘输入的字符串
     * @return 如果输入了N或NO，则返回true，否则返回false
     */
    private boolean noEntered(String text) {
        return stringIsAnyValue(text, "N", "NO");
    }

    /**
     * 检查字符串是否等于一系列变量值中的一个
     * 用于检查例如Y或YES
     * 比较不区分大小写
     *
     * @param text   源字符串
     * @param values 要与源字符串进行比较的一系列值
     * @return 如果在传递的一系列字符串中找到了匹配，则返回true
     */
    // 检查字符串是否等于可变数量的值中的任意一个
    private boolean stringIsAnyValue(String text, String... values) {

        // 遍历可变数量的值，并逐个进行比较
        for (String val : values) {
            if (text.equalsIgnoreCase(val)) {
                return true;
            }
        }

        // 没有匹配的值
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

    /**
     * 生成随机数
     *
     * @return 随机数
     */
    private int randomNumber() {
        return (int) (Math.random()
                * (SIX) + 1);
    }
# 闭合前面的函数定义
```