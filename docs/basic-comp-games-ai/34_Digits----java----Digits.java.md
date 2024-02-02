# `basic-computer-games\34_Digits\java\Digits.java`

```py
import java.util.Arrays;  // 导入 Arrays 类
import java.util.InputMismatchException;  // 导入 InputMismatchException 类
import java.util.Scanner;  // 导入 Scanner 类

/**
 * DIGITS
 * <p>
 * Converted from BASIC to Java by Aldrin Misquitta (@aldrinm)
 */
public class Digits {

    }

    private static boolean readContinueChoice(Scanner scan) {
        System.out.print("\nDO YOU WANT TO TRY AGAIN (1 FOR YES, 0 FOR NO) ? ");  // 打印提示信息
        int choice;
        try {
            choice = scan.nextInt();  // 读取用户输入的整数
            return choice == 1;  // 返回用户输入的整数是否等于1的布尔值
        } catch (InputMismatchException ex) {
            return false;  // 捕获输入不匹配异常，返回 false
        } finally {
            scan.nextLine();  // 读取用户输入的下一行
        }
    }

    private static int[] read10Numbers(Scanner scan) {
        System.out.print("TEN NUMBERS, PLEASE ? ");  // 打印提示信息
        int[] numbers = new int[10];  // 创建一个长度为10的整数数组

        for (int i = 0; i < numbers.length; i++) {  // 循环遍历数组
            boolean validInput = false;  // 初始化有效输入标志为 false
            while (!validInput) {  // 当有效输入标志为 false 时循环
                try {
                    int n = scan.nextInt();  // 读取用户输入的整数
                    validInput = true;  // 设置有效输入标志为 true
                    numbers[i] = n;  // 将读取的整数存入数组
                } catch (InputMismatchException ex) {
                    System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE");  // 捕获输入不匹配异常，打印错误信息
                } finally {
                    scan.nextLine();  // 读取用户输入的下一行
                }
            }
        }

        return numbers;  // 返回整数数组
    }

    private static void printInstructions() {
        System.out.println("\n");  // 打印空行
        System.out.println("PLEASE TAKE A PIECE OF PAPER AND WRITE DOWN");  // 打印提示信息
        System.out.println("THE DIGITS '0', '1', OR '2' THIRTY TIMES AT RANDOM.");  // 打印提示信息
        System.out.println("ARRANGE THEM IN THREE LINES OF TEN DIGITS EACH.");  // 打印提示信息
        System.out.println("I WILL ASK FOR THEN TEN AT A TIME.");  // 打印提示信息
        System.out.println("I WILL ALWAYS GUESS THEM FIRST AND THEN LOOK AT YOUR");  // 打印提示信息
        System.out.println("NEXT NUMBER TO SEE IF I WAS RIGHT. BY PURE LUCK,");  // 打印提示信息
        System.out.println("I OUGHT TO BE RIGHT TEN TIMES. BUT I HOPE TO DO BETTER");  // 打印提示信息
        System.out.println("THAN THAT *****");  // 打印提示信息
        System.out.println();  // 打印空行
    }
    # 读取用户输入的指令选择，如果输入不是整数则返回 false，无论如何都会执行 finally 块
    private static boolean readInstructionChoice(Scanner scan) {
        # 打印提示信息，要求用户输入指令选择
        System.out.print("FOR INSTRUCTIONS, TYPE '1', ELSE TYPE '0' ? ");
        # 定义变量 choice
        int choice;
        try:
            # 尝试读取用户输入的整数
            choice = scan.nextInt();
            # 返回用户输入的整数是否等于 1
            return choice == 1;
        except (InputMismatchException ex):
            # 如果捕获到输入不是整数的异常，则返回 false
            return false;
        finally:
            # 无论如何都会执行的代码块，用于清空输入缓冲区
            scan.nextLine();
    }
    
    # 打印游戏介绍信息
    private static void printIntro() {
        # 打印游戏标题
        System.out.println("                                DIGITS");
        # 打印游戏开发者信息
        System.out.println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        # 打印空行
        System.out.println("\n\n");
        # 打印游戏说明
        System.out.println("THIS IS A GAME OF GUESSING.");
    }
# 闭合前面的函数定义
```