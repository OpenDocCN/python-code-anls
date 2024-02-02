# `basic-computer-games\90_Tower\javascript\tower.js`

```py
// TOWER
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

// 定义打印函数，将字符串添加到输出元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义输入函数，返回一个 Promise 对象
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
                       // 监听输入框的键盘事件
                       input_element.addEventListener("keydown", function (event) {
                                                      // 如果按下的是回车键
                                                      if (event.keyCode == 13) {
                                                      // 获取输入框的值
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的值
                                                      print(input_str);
                                                      // 打印换行符
                                                      print("\n");
                                                      // 解析 Promise 对象
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义制表符函数，返回指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 定义一个空数组
var ta = [];

// 打印子程序
function show_towers()
{
    var z;
    # 循环7次，k从1到7
    for (var k = 1; k <= 7; k++) {
        # 初始化变量z为10
        z = 10;
        # 初始化空字符串str
        str = "";
        # 循环3次，j从1到3
        for (var j = 1; j <= 3; j++) {
            # 如果ta[k][j]不等于0
            if (ta[k][j] != 0) {
                # 当str长度小于z减去ta[k][j]除以2的结果时，往str中添加空格
                while (str.length < z - Math.floor(ta[k][j] / 2))
                    str += " ";
                # 循环ta[k][j]次，往str中添加"*"
                for (v = 1; v <= ta[k][j]; v++)
                    str += "*";
            } else {
                # 当ta[k][j]等于0时，往str中添加空格
                while (str.length < z)
                    str += " ";
                # 往str中添加"*"
                str += "*";
            }
            # 更新z的值
            z += 21;
        }
        # 打印str并换行
        print(str + "\n");
    }
// 结束 main 函数

// 主控制部分
async function main()
{
    // 打印标题
    print(tab(33) + "TOWERS\n");
    // 打印副标题
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    // 打印空行
    print("\n");
    print("\n");
    print("\n");
    // 打印空行
    }
    print("\n");
    // 打印感谢信息
    print("THANKS FOR THE GAME!\n");
    // 打印空行
    print("\n");
}

// 调用 main 函数
main();
```