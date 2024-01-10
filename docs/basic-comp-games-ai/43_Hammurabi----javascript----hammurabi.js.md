# `basic-computer-games\43_Hammurabi\javascript\hammurabi.js`

```
// 输出函数，将字符串添加到指定元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 输入函数，返回一个 Promise 对象
function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       // 创建输入元素
                       input_element = document.createElement("INPUT");

                       // 输出提示符
                       print("? ");
                       // 设置输入元素类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入元素添加到指定元素中
                       document.getElementById("output").appendChild(input_element);
                       // 输入元素获取焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘按下事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 移除输入元素
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 输出输入的字符串
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析 Promise 对象
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 生成指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var a;
var s;

// 超出粮食数量的提示函数
function exceeded_grain()
{
    print("HAMURABI: THINK AGAIN.  YOU HAVE ONLY\n");
    print(s + " BUSHELS OF GRAIN.  NOW THEN,\n");
}

// 超出土地数量的提示函数
function exceeded_acres()
{
    print("HAMURABI: THINK AGAIN.  YOU OWN ONLY " + a + " ACRES.  NOW THEN,\n");
}

// 主控制部分，使用 async 函数声明
async function main()
{
    // 输出标题
    print(tab(32) + "HAMURABI\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
}
    # 打印空行
    print("\n");
    # 打印游戏开始提示信息
    print("TRY YOUR HAND AT GOVERNING ANCIENT SUMERIA\n");
    print("FOR A TEN-YEAR TERM OF OFFICE.\n");
    print("\n");

    # 初始化变量
    d1 = 0;
    p1 = 0;
    z = 0;
    p = 95;
    s = 2800;
    h = 3000;
    e = h - s;
    y = 3;
    a = h / y;
    i = 5;
    q = 1;
    d = 0;
    }
    # 如果条件成立，输出相关信息
    if (q < 0) {
        print("\n");
        print("HAMURABI:  I CANNOT DO WHAT YOU WISH.\n");
        print("GET YOURSELF ANOTHER STEWARD!!!!!\n");
    } else {
        # 输出统计信息
        print("IN YOUR 10-YEAR TERM OF OFFICE, " + p1 + " PERCENT OF THE\n");
        print("POPULATION STARVED PER YEAR ON THE AVERAGE, I.E. A TOTAL OF\n");
        print(d1 + " PEOPLE DIED!!\n");
        l = a / p;
        print("YOU STARTED WITH 10 ACRES PER PERSON AND ENDED WITH\n");
        print(l + " ACRES PER PERSON.\n");
        print("\n");
        # 根据不同情况输出不同的评价
        if (p1 > 33 || l < 7) {
            print("DUE TO THIS EXTREME MISMANAGEMENT YOU HAVE NOT ONLY\n");
            print("BEEN IMPEACHED AND THROWN OUT OF OFFICE BUT YOU HAVE\n");
            print("ALSO BEEN DECLARED NATIONAL FINK!!!!\n");
        } else if (p1 > 10 || l < 9) {
            print("YOUR HEAVY-HANDED PERFORMANCE SMACKS OF NERO AND IVAN IV.\n");
            print("THE PEOPLE (REMAINING) FIND YOU AN UNPLEASANT RULER, AND,\n");
            print("FRANKLY, HATE YOUR GUTS!!\n");
        } else if (p1 > 3 || l < 10) {
            print("YOUR PERFORMANCE COULD HAVE BEEN SOMEWHAT BETTER, BUT\n");
            print("REALLY WASN'T TOO BAD AT ALL. " + Math.floor(p * 0.8 * Math.random()) + " PEOPLE\n");
            print("WOULD DEARLY LIKE TO SEE YOU ASSASSINATED BUT WE ALL HAVE OUR\n");
            print("TRIVIAL PROBLEMS.\n");
        } else {
            print("A FANTASTIC PERFORMANCE!!!  CHARLEMANGE, DISRAELI, AND\n");
            print("JEFFERSON COMBINED COULD NOT HAVE DONE BETTER!\n");
        }
    }
    # 打印结束语
    print("\n");
    print("SO LONG FOR NOW.\n");
    print("\n");
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```