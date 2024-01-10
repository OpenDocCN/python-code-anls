# `basic-computer-games\02_Amazing\javascript\amazing.js`

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
                       // 监听键盘事件
                       input_element.addEventListener("keydown", function (event) {
                                                      // 如果按下的是回车键
                                                      if (event.keyCode == 13) {
                                                      // 获取输入框的值
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的值
                                                      print(input_str);
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

// 打印标题
print(tab(28) + "AMAZING PROGRAM\n");
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
print("\n");
print("\n");
print("\n");
print("FOR EXAMPLE TYPE 10,10 AND PRESS ENTER\n");
print("\n");

// 主程序
async function main()
{
    # 进入无限循环，直到输入有效的宽度和长度
    while (1) {
        # 提示用户输入宽度和长度
        print("WHAT ARE YOUR WIDTH AND LENGTH");
        # 等待用户输入
        a = await input();
        # 将输入的字符串转换为整数，作为宽度
        h = parseInt(a);
        # 从输入的字符串中获取逗号后面的部分，并转换为整数，作为长度
        v2 = parseInt(a.substr(a.indexOf(",") + 1));
        # 如果宽度和长度都大于1，则跳出循环
        if (h > 1 && v2 > 1)
            break;
        # 如果宽度和长度不合法，提示用户重新输入
        print("MEANINGLESS DIMENSIONS.  TRY AGAIN.\n");
    }
    # 初始化宽度和长度对应的二维数组
    w = [];
    v = [];
    for (i = 1; i <= h; i++) {
        w[i] = [];
        v[i] = [];
        for (j = 1; j <= v2; j++) {
            w[i][j] = 0;
            v[i][j] = 0;
        }
    }
    # 打印空行
    print("\n");
    print("\n");
    print("\n");
    print("\n");
    # 初始化变量
    q = 0;
    z = 0;
    # 随机生成起始点的横坐标
    x = Math.floor(Math.random() * h + 1);
    # 打印迷宫的顶部边界
    for (i = 1; i <= h; i++) {
        if (i == x)
            print(".  ");
        else
            print(".--");
    }
    print(".\n");
    # 初始化变量
    c = 1;
    w[x][1] = c;
    c++;
    r = x;
    s = 1;
    entry = 0;
    }
    # 遍历迷宫的每一列
    for (j = 1; j <= v2; j++) {
        # 初始化字符串
        str = "I";
        # 遍历迷宫的每一行
        for (i = 1; i <= h; i++) {
            # 根据迷宫的状态添加字符到字符串中
            if (v[i][j] < 2)
                str += "  I";
            else
                str += "   ";
        }
        # 打印字符串
        print(str + "\n");
        # 初始化字符串
        str = "";
        # 遍历迷宫的每一行
        for (i = 1; i <= h; i++) {
            # 根据迷宫的状态添加字符到字符串中
            if (v[i][j] == 0 || v[i][j] == 2)
                str += ":--";
            else
                str += ":  ";
        }
        # 打印字符串
        print(str + ".\n");
    }
// 如果你想查看访问单元格的顺序
//    for (j = 1; j <= v2; j++) {
//        初始化字符串为"I"
//        for (i = 1; i <= h; i++) {
//            将每行的单元格值添加到字符串中
//            str += w[i][j] + " ";
//        }
//        打印字符串并换行
//        print(str + "\n");
//    }
}

// 调用主函数
main();
```