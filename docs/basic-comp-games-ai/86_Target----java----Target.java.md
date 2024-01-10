# `basic-computer-games\86_Target\java\Target.java`

```
import java.util.Scanner;

/**
 * TARGET
 * <p>
 * Converted from BASIC to Java by Aldrin Misquitta (@aldrinm)
 */
public class Target {

    private static final double RADIAN = 180 / Math.PI;

    }

    // 从控制台读取输入的角度偏差和距离，返回一个 TargetAttempt 对象
    private static TargetAttempt readInput(Scanner scan) {
        System.out.println("INPUT ANGLE DEVIATION FROM X, DEVIATION FROM Z, DISTANCE ");
        boolean validInput = false;
        TargetAttempt targetAttempt = new TargetAttempt();
        // 循环直到输入有效
        while (!validInput) {
            String input = scan.nextLine();
            final String[] split = input.split(",");
            try {
                // 尝试将输入的字符串转换为浮点数，并赋值给 TargetAttempt 对象的属性
                targetAttempt.xDeviation = Float.parseFloat(split[0]);
                targetAttempt.zDeviation = Float.parseFloat(split[1]);
                targetAttempt.distance = Float.parseFloat(split[2]);
                validInput = true;
            } catch (NumberFormatException nfe) {
                // 如果输入不是有效的数字，则提示重新输入
                System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE\n? ");
            }

        }
        // 返回输入的 TargetAttempt 对象
        return targetAttempt;
    }
    // 打印游戏介绍信息
    private static void printIntro() {
        System.out.println("                                TARGET");
        System.out.println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println("\n\n");
        System.out.println("YOU ARE THE WEAPONS OFFICER ON THE STARSHIP ENTERPRISE");
        System.out.println("AND THIS IS A TEST TO SEE HOW ACCURATE A SHOT YOU");
        System.out.println("ARE IN A THREE-DIMENSIONAL RANGE.  YOU WILL BE TOLD");
        System.out.println("THE RADIAN OFFSET FOR THE X AND Z AXES, THE LOCATION");
        System.out.println("OF THE TARGET IN THREE DIMENSIONAL RECTANGULAR COORDINATES,");
        System.out.println("THE APPROXIMATE NUMBER OF DEGREES FROM THE X AND Z");
        System.out.println("AXES, AND THE APPROXIMATE DISTANCE TO THE TARGET.");
        System.out.println("YOU WILL THEN PROCEED TO SHOOT AT THE TARGET UNTIL IT IS");
        System.out.println("DESTROYED!");
        System.out.println("\nGOOD LUCK!!\n\n");
    }

    /**
     * Represents the user input
     */
    // 表示用户输入的目标尝试
    private static class TargetAttempt {

        double xDeviation; // X轴偏移
        double zDeviation; // Z轴偏移
        double distance; // 距离
    }
# 闭合前面的函数定义
```