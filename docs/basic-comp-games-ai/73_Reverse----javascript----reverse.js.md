# `basic-computer-games\73_Reverse\javascript\reverse.js`

```
// 定义一个打印函数，将字符串添加到输出元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个输入函数，返回一个 Promise 对象
function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       // 创建一个输入元素
                       input_element = document.createElement("INPUT");

                       // 打印提示符
                       print("? ");
                       // 设置输入元素类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入元素添加到输出元素中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入元素获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘按下事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 移除输入元素
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      // 打印换行符
                                                      print("\n");
                                                      // 解析 Promise 对象
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义一个制表符函数，返回指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 定义一个空数组和一个变量 n
var a = [];
var n;

// 打印游戏规则的子程序
function print_rules()
{
    // 打印游戏规则
    print("\n");
    print("THIS IS THE GAME OF 'REVERSE'.  TO WIN, ALL YOU HAVE\n");
    print("TO DO IS ARRANGE A LIST OF NUMBERS (1 THROUGH " + n + ")\n");
    print("IN NUMERICAL ORDER FROM LEFT TO RIGHT.  TO MOVE, YOU\n");
    print("TELL ME HOW MANY NUMBERS (COUNTING FROM THE LEFT) TO\n");
    print("REVERSE.  FOR EXAMPLE, IF THE CURRENT LIST IS:\n");
}
    # 打印空行
    print("\n");
    # 打印指定数字序列
    print("2 3 4 5 1 6 7 8 9\n");
    # 打印空行
    print("\n");
    # 打印提示信息
    print("AND YOU REVERSE 4, THE RESULT WILL BE:\n");
    # 打印空行
    print("\n");
    # 打印指定数字序列
    print("5 4 3 2 1 6 7 8 9\n");
    # 打印空行
    print("\n");
    # 打印提示信息
    print("NOW IF YOU REVERSE 5, YOU WIN!\n");
    # 打印空行
    print("\n");
    # 打印指定数字序列
    print("1 2 3 4 5 6 7 8 9\n");
    # 打印空行
    print("\n");
    # 打印提示信息
    print("NO DOUBT YOU WILL LIKE THIS GAME, BUT\n");
    # 打印提示信息
    print("IF YOU WANT TO QUIT, REVERSE 0 (ZERO).\n");
    # 打印空行
    print("\n");
// 子程序，用于打印列表
function print_list()
{
    // 打印换行符
    print("\n");
    // 遍历列表并打印每个元素
    for (k = 1; k <= n; k++)
        print(" " + a[k] + " ");
    // 打印两个换行符
    print("\n");
    print("\n");
}

// 主程序
async function main()
{
    // 打印32个空格后输出"REVERSE"
    print(tab(32) + "REVERSE\n");
    // 打印15个空格后输出"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    // 打印三个换行符
    print("\n");
    print("\n");
    print("\n");
    // 打印"REVERSE -- A GAME OF SKILL"
    print("REVERSE -- A GAME OF SKILL\n");
    // 初始化列表a的前20个元素为0
    for (i = 0; i <= 20; i++)
        a[i] = 0;
    // *** N=NUMBER OF NUMBER
    // 设置n的值为9
    n = 9;
    // 打印"Do you want the rules? (YES OR NO)"并等待用户输入
    print("DO YOU WANT THE RULES? (YES OR NO)");
    str = await input();
    // 如果用户输入的字符串转换为大写后等于"YES"或"Y"，则打印规则
    if (str.toUpperCase() === "YES" || str.toUpperCase() === "Y")
        print_rules();
}
    // 进入无限循环，直到条件不满足时退出
    while (1) {
        // *** 生成一个随机列表 a(1) 到 a(n)
        a[1] = Math.floor((n - 1) * Math.random() + 2);
        // 遍历列表，生成不重复的随机数
        for (k = 2; k <= n; k++) {
            do {
                a[k] = Math.floor(n * Math.random() + 1);
                // 检查生成的随机数是否重复
                for (j = 1; j <= k - 1; j++) {
                    if (a[k] == a[j])
                        break;
                }
            } while (j <= k - 1) ;
        }
        // *** 打印原始列表并开始游戏
        print("\n");
        print("HERE WE GO ... THE LIST IS:\n");
        t = 0;
        print_list();
        // 进入游戏循环
        while (1) {
            // 循环直到输入合法的翻转次数
            while (1) {
                print("HOW MANY SHALL I REVERSE");
                r = parseInt(await input());
                if (r == 0)
                    break;
                if (r <= n)
                    break;
                print("OOPS! WRONG! I CAN REVERSE AT MOST " + n + "\n");
            }
            if (r == 0)
                break;
            t++;
            // *** 翻转 r 个数字并打印新列表
            for (k = 1; k <= Math.floor(r / 2); k++) {
                z = a[k];
                a[k] = a[r - k + 1];
                a[r - k + 1] = z;
            }
            print_list();
            // *** 检查是否获胜
            for (k = 1; k <= n; k++) {
                if (a[k] != k)
                    break;
            }
            if (k > n) {
                print("YOU WON IT IN " + t + " MOVES!!!\n");
                print("\n");
                break;
            }
        }
        print("\n");
        print("TRY AGAIN? (YES OR NO)");
        str = await input();
        // 如果输入为 "NO" 或 "N" 则退出循环
        if (str.toUpperCase() === "NO" || str.toUpperCase() === "N")
            break;
    }
    print("\n");
    print("O.K. HOPE YOU HAD FUN!!\n");
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```