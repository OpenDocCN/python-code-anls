# `d:/src/tocomm/basic-computer-games\73_Reverse\javascript\reverse.js`

```
// 创建一个新的 Promise 对象，用于处理输入操作
// 创建一个 INPUT 元素，用于接收用户输入
// 在页面上打印提示符 "? "
// 设置 INPUT 元素的类型为文本输入
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为 undefined
input_str = undefined;
# 添加键盘按下事件监听器，当按下回车键时，获取输入字符串，移除输入元素，打印输入字符串并解析为 Promise 对象
input_element.addEventListener("keydown", function (event) {
    if (event.keyCode == 13) {
        input_str = input_element.value;
        document.getElementById("output").removeChild(input_element);
        print(input_str);
        print("\n");
        resolve(input_str);
    }
});
# 结束事件监听器的添加
});
}

# 定义一个 tab 函数，用于生成指定数量的空格字符串
function tab(space)
{
    var str = "";
    # 当 space 大于 0 时，循环减少 space 并将空格字符串添加到 str 中
    while (space-- > 0)
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回修改后的字符串

var a = [];  // 创建一个空数组
var n;  // 声明变量n

// 打印游戏规则的子程序
function print_rules()
{
    print("\n");  // 打印换行符
    print("THIS IS THE GAME OF 'REVERSE'.  TO WIN, ALL YOU HAVE\n");  // 打印游戏规则
    print("TO DO IS ARRANGE A LIST OF NUMBERS (1 THROUGH " + n + ")\n");  // 打印游戏规则
    print("IN NUMERICAL ORDER FROM LEFT TO RIGHT.  TO MOVE, YOU\n");  // 打印游戏规则
    print("TELL ME HOW MANY NUMBERS (COUNTING FROM THE LEFT) TO\n");  // 打印游戏规则
    print("REVERSE.  FOR EXAMPLE, IF THE CURRENT LIST IS:\n");  // 打印游戏规则
    print("\n");  // 打印换行符
    print("2 3 4 5 1 6 7 8 9\n");  // 打印示例列表
    print("\n");  // 打印换行符
    print("AND YOU REVERSE 4, THE RESULT WILL BE:\n");  // 打印游戏规则中的示例
    # 打印空行
    print("\n");
    # 打印数字序列
    print("5 4 3 2 1 6 7 8 9\n");
    # 打印空行
    print("\n");
    # 打印提示信息
    print("NOW IF YOU REVERSE 5, YOU WIN!\n");
    # 打印空行
    print("\n");
    # 打印数字序列
    print("1 2 3 4 5 6 7 8 9\n");
    # 打印空行
    print("\n");
    # 打印提示信息
    print("NO DOUBT YOU WILL LIKE THIS GAME, BUT\n");
    # 打印提示信息
    print("IF YOU WANT TO QUIT, REVERSE 0 (ZERO).\n");
    # 打印空行
    print("\n");
}

// 用于打印列表的子程序
function print_list()
{
    # 打印空行
    print("\n");
    # 遍历列表并打印每个元素
    for (k = 1; k <= n; k++)
        print(" " + a[k] + " ");
    # 打印空行
    print("\n");
    # 打印空行
    print("\n");
}

// 主程序
async function main()
{
    // 打印标题
    print(tab(32) + "REVERSE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("REVERSE -- A GAME OF SKILL\n");
    print("\n");
    for (i = 0; i <= 20; i++)
        a[i] = 0;
    // *** N=NUMBER OF NUMBER
    n = 9;
    print("DO YOU WANT THE RULES? (YES OR NO)");
    // 等待用户输入
    str = await input();
    // 如果用户输入是"YES"或"Y"，则打印规则
    if (str.toUpperCase() === "YES" || str.toUpperCase() === "Y")
        print_rules();
```
在这个示例中，我添加了对每个语句的注释，解释了它们的作用。
    while (1) {
        // *** Make a random list a(1) to a(n)
        // 生成一个随机列表 a(1) 到 a(n)
        a[1] = Math.floor((n - 1) * Math.random() + 2);
        for (k = 2; k <= n; k++) {
            do {
                // 生成随机数并确保不重复
                a[k] = Math.floor(n * Math.random() + 1);
                for (j = 1; j <= k - 1; j++) {
                    if (a[k] == a[j])
                        break;
                }
            } while (j <= k - 1) ;
        }
        // *** Print original list and start game
        // 打印原始列表并开始游戏
        print("\n");
        print("HERE WE GO ... THE LIST IS:\n");
        t = 0;
        print_list();
        while (1) {
            while (1) {
                print("HOW MANY SHALL I REVERSE");
                // ...
                // 从输入中读取一个整数并将其转换为整数类型
                r = parseInt(await input());
                // 如果读取的整数为0，则跳出循环
                if (r == 0)
                    break;
                // 如果读取的整数小于等于n，则跳出循环
                if (r <= n)
                    break;
                // 打印错误信息，提示用户最多只能翻转n个数字
                print("OOPS! WRONG! I CAN REVERSE AT MOST " + n + "\n");
            }
            // 如果读取的整数为0，则跳出循环
            if (r == 0)
                break;
            // 增加翻转次数计数
            t++;
            // *** 反转r个数字并打印新列表
            for (k = 1; k <= Math.floor(r / 2); k++) {
                z = a[k];
                a[k] = a[r - k + 1];
                a[r - k + 1] = z;
            }
            // 打印列表
            print_list();
            // *** 检查是否获胜
            for (k = 1; k <= n; k++) {
                if (a[k] != k)
                    break;  # 结束当前循环，跳出循环体
            }
            if (k > n) {  # 如果猜测次数超过了设定的最大次数
                print("YOU WON IT IN " + t + " MOVES!!!\n");  # 打印出猜测次数，并提示玩家赢了
                print("\n");  # 打印空行
                break;  # 结束当前循环，跳出循环体
            }
        }
        print("\n");  # 打印空行
        print("TRY AGAIN? (YES OR NO)");  # 提示玩家是否要再玩一次
        str = await input();  # 等待玩家输入
        if (str.toUpperCase() === "NO" || str.toUpperCase() === "N")  # 如果玩家输入的是"NO"或"N"
            break;  # 结束当前循环，跳出循环体
    }
    print("\n");  # 打印空行
    print("O.K. HOPE YOU HAD FUN!!\n");  # 提示玩家游戏结束
}

main();  # 调用主函数
```