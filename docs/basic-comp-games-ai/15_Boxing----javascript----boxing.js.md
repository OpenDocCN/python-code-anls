# `basic-computer-games\15_Boxing\javascript\boxing.js`

```py
// 定义一个打印函数，将字符串输出到指定的元素中
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

                       // 输出提示符
                       print("? ");
                       // 设置输入框类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到指定元素中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘按下事件
                       input_element.addEventListener("keydown", function (event) {
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
    // 打印标题
    print(tab(33) + "BOXING\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 打印游戏规则
    print("BOXING OLYMPIC STYLE (3 ROUNDS -- 2 OUT OF 3 WINS)\n");
    j = 0;
    l = 0;
    print("\n");
    // 获取对手的名字
    print("WHAT IS YOUR OPPONENT'S NAME");
    js = await input();
    // 获取自己的名字
    print("INPUT YOUR MAN'S NAME");
    ls = await input();
}
    # 打印不同拳击动作的选项
    print("DIFFERENT PUNCHES ARE: (1) FULL SWING; (2) HOOK; (3) UPPERCUT; (4) JAB.\n");
    # 打印提示信息，要求输入选手的最佳拳击动作
    print("WHAT IS YOUR MANS BEST");
    # 将输入的值转换为整数类型
    b = parseInt(await input());
    # 打印提示信息，要求输入选手的脆弱性
    print("WHAT IS HIS VULNERABILITY");
    # 将输入的值转换为整数类型
    d = parseInt(await input());
    # 生成随机数，表示选手的拳击动作
    do {
        b1 = Math.floor(4 * Math.random() + 1);
        d1 = Math.floor(4 * Math.random() + 1);
    } while (b1 == d1) ;
    # 打印选手的优势和脆弱性
    print(js + "'S ADVANTAGE IS " + b1 + " AND VULNERABILITY IS SECRET.\n");
    # 打印空行
    print("\n");
    # 初始化被击倒的次数
    knocked = 0;
    }
    # 如果选手 j 的得分大于等于 2
    if (j >= 2) {
        # 打印选手 j 胜利的信息
        print(js + " WINS (NICE GOING, " + js + ").\n");
    # 如果选手 l 的得分大于等于 2
    } else if (l >= 2) {
        # 打印选手 l 胜利的信息
        print(ls + " AMAZINGLY WINS!!\n");
    # 如果有选手被击倒
    } else if (knocked) {
        # 打印选手被击倒的信息
        print(ls + " IS KNOCKED COLD AND " + js + " IS THE WINNER AND CHAMP!\n");
    # 如果没有选手被击倒
    } else {
        # 打印选手被击倒的信息
        print(js + " IS KNOCKED COLD AND " + ls + " IS THE WINNER AND CHAMP!\n");
    }
    # 打印空行
    print("\n");
    # 打印两个空行
    print("\n");
    # 打印结束语
    print("AND NOW GOODBYE FROM THE OLYMPIC ARENA.\n");
    # 打印空行
    print("\n");
# 结束 main 函数的定义
}

# 调用 main 函数
main();
```