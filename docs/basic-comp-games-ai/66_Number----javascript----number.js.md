# `basic-computer-games\66_Number\javascript\number.js`

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
                                                      // 如果按下回车键
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
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

// 主程序
async function main()
{
    // 打印标题
    print(tab(33) + "NUMBER\n");
    // 打印副标题
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    // 打印空行
    print("\n");
    print("\n");
    print("\n");
    // 打印游戏规则
    print("YOU HAVE 100 POINTS.  BY GUESSING NUMBERS FROM 1 TO 5, YOU\n");
    print("CAN GAIN OR LOSE POINTS DEPENDING UPON HOW CLOSE YOU GET TO\n");
    print("A RANDOM NUMBER SELECTED BY THE COMPUTER.\n");
    print("\n");
}
    # 打印提示信息，说明偶尔会有一个奖池，可以使得分翻倍
    print("YOU OCCASIONALLY WILL GET A JACKPOT WHICH WILL DOUBLE(!)\n");
    # 打印提示信息，说明当分数达到500时获胜
    print("YOUR POINT COUNT.  YOU WIN WHEN YOU GET 500 POINTS.\n");
    # 打印空行
    print("\n");
    # 初始化分数为0
    p = 0;
    # 进入无限循环
    while (1) {
        # 循环直到输入的数字在1到5之间
        do {
            # 提示用户猜一个1到5之间的数字
            print("GUESS A NUMBER FROM 1 TO 5");
            # 将输入的字符串转换为整数
            g = parseInt(await input());
        } while (g < 1 || g > 5) ;
        # 生成1到5之间的随机整数
        r = Math.floor(5 * Math.random() + 1);
        s = Math.floor(5 * Math.random() + 1);
        t = Math.floor(5 * Math.random() + 1);
        u = Math.floor(5 * Math.random() + 1);
        v = Math.floor(5 * Math.random() + 1);
        # 根据用户猜的数字和随机数进行不同的操作
        if (g == r) {
            p -= 5;
        } else if (g == s) {
            p += 5;
        } else if (g == t) {
            p += p;
            # 如果猜中了奖池数字，分数翻倍
            print("YOU HIT THE JACKPOT!!!\n");
        } else if (g == u) {
            p += 1;
        } else if (g == v) {
            p -= p * 0.5;
        }
        # 如果分数小于等于500，打印当前分数
        if (p <= 500) {
            print("YOU HAVE " + p + " POINTS.\n");
            print("\n");
        } else {
            # 如果分数大于500，打印获胜信息并跳出循环
            print("!!!!YOU WIN!!!! WITH " + p + " POINTS.\n");
            break;
        }
    }
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```