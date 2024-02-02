# `basic-computer-games\11_Bombardment\javascript\bombardment.js`

```py
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
                       // 创建一个输入框元素
                       input_element = document.createElement("INPUT");

                       // 打印提示符
                       print("? ");
                       // 设置输入框类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到输出元素中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘按下事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 从输出元素中移除输入框
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

// 定义一个生成指定数量空格的函数
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 主程序
async function main()
{
    // 打印游戏标题
    print(tab(33) + "BOMBARDMENT\n");
    // 打印游戏信息
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("YOU ARE ON A BATTLEFIELD WITH 4 PLATOONS AND YOU\n");
    print("HAVE 25 OUTPOSTS AVAILABLE WHERE THEY MAY BE PLACED.\n");
    print("YOU CAN ONLY PLACE ONE PLATOON AT ANY ONE OUTPOST.\n");
}
    # 打印游戏规则和提示信息
    print("THE COMPUTER DOES THE SAME WITH ITS FOUR PLATOONS.\n");
    print("\n");
    print("THE OBJECT OF THE GAME IS TO FIRE MISSILES AT THE\n");
    print("OUTPOSTS OF THE COMPUTER.  IT WILL DO THE SAME TO YOU.\n");
    print("THE ONE WHO DESTROYS ALL FOUR OF THE ENEMY'S PLATOONS\n");
    print("FIRST IS THE WINNER.\n");
    print("\n");
    print("GOOD LUCK... AND TELL US WHERE YOU WANT THE BODIES SENT!\n");
    print("\n");
    # 打印提示信息，要求玩家使用矩阵来标记数字
    # 因为这个注释是对代码中注释的解释，所以不需要在代码块外面再次总结含义
    # "TEAR OFF" because it supposed this to be printed on a teletype
    print("TEAR OFF MATRIX AND USE IT TO CHECK OFF THE NUMBERS.\n");
    # 打印空行
    for (r = 1; r <= 5; r++)
        print("\n");
    # 初始化 ma 数组
    ma = [];
    # 将 ma 数组的每个元素初始化为 0
    for (r = 1; r <= 100; r++)
        ma[r] = 0;
    # 初始化 p, q, z 三个变量
    p = 0;
    q = 0;
    z = 0;
    # 打印矩阵的行号
    for (r = 1; r <= 5; r++) {
        i = (r - 1) * 5 + 1;
        print(i + "\t" + (i + 1) + "\t" + (i + 2) + "\t" + (i + 3) + "\t" + (i + 4) + "\n");
    }
    # 打印空行
    for (r = 1; r <= 10; r++)
        print("\n");
    # 生成随机数并赋值给 c, d, e, f 四个变量
    c = Math.floor(Math.random() * 25) + 1;
    do {
        d = Math.floor(Math.random() * 25) + 1;
        e = Math.floor(Math.random() * 25) + 1;
        f = Math.floor(Math.random() * 25) + 1;
    } while (c == d || c == e || c == f || d == e || d == f || e == f) ;
    # 提示玩家输入四个位置
    print("WHAT ARE YOUR FOUR POSITIONS");
    # 等待用户输入
    str = await input();
    # 将输入的字符串转换为整数并赋值给 g, h, k, l 四个变量
    g = parseInt(str);
    str = str.substr(str.indexOf(",") + 1);
    h = parseInt(str);
    str = str.substr(str.indexOf(",") + 1);
    k = parseInt(str);
    str = str.substr(str.indexOf(",") + 1);
    l = parseInt(str);
    # 打印空行
    print("\n");
    # 另一个"bug"，玩家的据点可能与计算机的据点重叠
    # 假设它们分别存在于不同的矩阵中
    # 这个注释是对代码中注释的解释，所以不需要在代码块外面再次总结含义
    }
# 调用名为main的函数
main();
```