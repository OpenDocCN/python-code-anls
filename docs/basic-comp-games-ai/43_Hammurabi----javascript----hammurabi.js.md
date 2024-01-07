# `basic-computer-games\43_Hammurabi\javascript\hammurabi.js`

```

// 定义一个打印函数，用于在页面上输出字符串
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

                       // 在页面上输出提示符
                       print("? ");
                       // 设置输入框类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到页面上
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听输入框的按键事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 从页面上移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 在页面上输出输入的字符串
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

var a; // 定义变量 a
var s; // 定义变量 s

// 当粮食数量超出时的提示函数
function exceeded_grain()
{
    print("HAMURABI: THINK AGAIN.  YOU HAVE ONLY\n");
    print(s + " BUSHELS OF GRAIN.  NOW THEN,\n");
}

// 当土地面积超出时的提示函数
function exceeded_acres()
{
    print("HAMURABI: THINK AGAIN.  YOU OWN ONLY " + a + " ACRES.  NOW THEN,\n");
}

// 主控制部分
async function main()
{
    // 主函数入口
}

// 调用主函数
main();

```