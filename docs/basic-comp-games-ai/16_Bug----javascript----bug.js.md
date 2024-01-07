# `basic-computer-games\16_Bug\javascript\bug.js`

```

// BUG
// 代码的标题或注释

// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
// 作者信息或代码来源

function print(str)
{
    // 在页面输出字符串
    document.getElementById("output").appendChild(document.createTextNode(str));
}

function input()
{
    // 返回一个 Promise 对象，用于获取用户输入
    return new Promise(function (resolve) {
                       // 创建一个输入框元素
                       const input_element = document.createElement("INPUT");

                       // 输出提示符
                       print("? ");
                       // 设置输入框属性
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到页面
                       document.getElementById("output").appendChild(input_element);
                       // 输入框获取焦点
                       input_element.focus();
                       // 监听键盘事件
                       input_element.addEventListener("keydown",
                           function (event) {
                                      // 如果按下回车键
                                      if (event.keyCode === 13) {
                                          // 获取输入的字符串
                                          const input_str = input_element.value;
                                          // 移除输入框
                                          document.getElementById("output").removeChild(input_element);
                                          // 输出输入的字符串
                                          print(input_str);
                                          print("\n");
                                          // 解析 Promise
                                          resolve(input_str);
                                      }
                                  });
                       });
}

function tab(space)
{
    // 返回指定数量的空格字符串
    let str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

function waitNSeconds(n) {
    // 返回一个 Promise 对象，延迟 n 秒后解析
    return new Promise(resolve => setTimeout(resolve, n*1000));
}

function scrollToBottom() {
    // 滚动页面到底部
    window.scrollTo(0, document.body.scrollHeight);
}

function draw_head()
{
    // 绘制头部
    print("        HHHHHHH\n");
    // ... 省略其他绘制代码
}

// 其他函数的作用和功能类似，用于绘制不同部分的 BUG

// Main program
async function main()
}

main();
// 调用主程序函数

```