# `basic-computer-games\05_Bagels\javascript\bagels.js`

```

// BAGELS
// 被 Oscar Toledo G. (nanochess) 从 BASIC 转换成 Javascript

function print(str)
{
    // 在页面上输出字符串
    document.getElementById("output").appendChild(document.createTextNode(str));
}

function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       // 创建一个输入框元素
                       input_element = document.createElement("INPUT");

                       // 输出提示符
                       print("? ");
                       // 设置输入框类型为文本
                       input_element.setAttribute("type", "text");
                       // 设置输入框长度
                       input_element.setAttribute("length", "50");
                       // 在页面上添加输入框
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听输入框的按键事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 在页面上输出输入的字符串
                                                      document.getElementById("output").removeChild(input_element);
                                                      print(input_str);
                                                      print("\n");
                                                      // 返回输入的字符串
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

function tab(space)
{
    var str = "";
    // 生成指定数量的空格
    while (space-- > 0)
        str += " ";
    return str;
}

print(tab(33) + "BAGELS\n");
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");

// *** Bagles number guessing game
// *** 原始来源未知，但据信是来自加州大学伯克利分校的 Lawrence Hall of Science

a1 = [0,0,0,0];
a = [0,0,0,0];
b = [0,0,0,0];

y = 0;
t = 255;

print("\n");
print("\n");
print("\n");

// 主程序
async function main()
}

main();

```