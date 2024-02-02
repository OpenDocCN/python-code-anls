# `basic-computer-games\63_Name\java\Name.java`

```py
import java.util.Arrays;  // 导入 Arrays 类，用于对数组进行排序
import java.util.Scanner;  // 导入 Scanner 类，用于接收用户输入

public class Name {

    public static void printempty() { System.out.println(" "); }  // 定义一个静态方法，用于打印空行
    public static void print(String toprint) { System.out.println(toprint); }  // 定义一个静态方法，用于打印指定内容

    public static void main(String[] args) {
        print("                                          NAME");  // 打印指定内容
        print("                         CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 打印指定内容
        printempty();  // 调用打印空行的方法
        printempty();  // 调用打印空行的方法
        print("HELLO.");  // 打印指定内容
        print("MY NAME iS CREATIVE COMPUTER.");  // 打印指定内容
        print("WHATS YOUR NAME? (FIRST AND LAST)");  // 打印指定内容

        Scanner namesc = new Scanner(System.in);  // 创建一个 Scanner 对象，用于接收用户输入
        String name = namesc.nextLine();  // 读取用户输入的姓名

        String namereversed = new StringBuilder(name).reverse().toString();  // 将输入的姓名进行反转

        char namesorted[] = name.toCharArray();  // 将输入的姓名转换为字符数组
        Arrays.sort(namesorted);  // 对字符数组进行排序

        printempty();  // 调用打印空行的方法
        print("THANK YOU, " + namereversed);  // 打印指定内容和反转后的姓名
        printempty();  // 调用打印空行的方法
        print("OOPS!  I GUESS I GOT IT BACKWARDS.  A SMART");  // 打印指定内容
        print("COMPUTER LIKE ME SHOULDN'T MAKE A MISTAKE LIKE THAT!");  // 打印指定内容
        printempty();  // 调用打印空行的方法
        printempty();  // 调用打印空行的方法
        print("BUT I JUST NOTICED YOUR LETTERS ARE OUT OF ORDER.");  // 打印指定内容

        print("LET'S PUT THEM IN ORDER LIKE THIS: " + new String(namesorted));  // 打印指定内容和排序后的姓名
        printempty();  // 调用打印空行的方法
        printempty();  // 调用打印空行的方法

        print("DON'T YOU LIKE THAT BETTER?");  // 打印指定内容
        printempty();  // 调用打印空行的方法

        Scanner agreementsc = new Scanner(System.in);  // 创建一个 Scanner 对象，用于接收用户输入
        String agreement = agreementsc.nextLine();  // 读取用户输入的回答

        if (agreement.equalsIgnoreCase("yes")) {  // 判断用户回答是否为 "yes"，忽略大小写
            print("I KNEW YOU'D AGREE!!");  // 如果回答为 "yes"，则打印指定内容
        } else {
            print("I'M SORRY YOU DON'T LIKE IT THAT WAY.");  // 如果回答不为 "yes"，则打印指定内容
            printempty();  // 调用打印空行的方法
            print("I REALLY ENJOYED MEETING YOU, " + name);  // 打印指定内容和用户输入的姓名
            print("HAVE A NICE DAY!");  // 打印指定内容
        }
    }
}
```