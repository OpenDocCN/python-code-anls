# `D:\src\scipysrc\scipy\scipy\spatial\src\function_ref.h`

```
#pragma once
// 包含标准类型特性库，用于进行类型特性检查和转换
#include <type_traits>

// 泛型函数对象的类型擦除引用
template <class Signature>
class FunctionRef;

// 函数对象的类型擦除引用特化模板
template <class Ret, class... Args>
class FunctionRef<Ret(Args...)> {
public:
    // 构造函数模板，接受任意函数对象并进行类型擦除
    template <class FunctionObject,
        typename std::enable_if<  // 不禁用默认复制构造函数
            !std::is_same<typename std::decay<FunctionObject>::type, FunctionRef>::value,
        int>::type = 0
    >
    FunctionRef(FunctionObject && a_FunctionObject) {
        // 存储函数对象的地址
        data_ = &a_FunctionObject;
        // 设置调用函数指针为特化的对象成员函数调用器
        call_function_ = &ObjectFunctionCaller<FunctionObject>;
    }

    // 函数调用运算符重载，执行存储的函数对象的调用
    Ret operator () (Args... args) {
        // 调用具体函数对象的成员函数调用器
        return call_function_(data_, std::forward<Args>(args)...);
    }

private:
    // 对象成员函数调用器模板，将类型擦除后的指针还原为具体类型并调用
    template <class ObjectType>
    static Ret ObjectFunctionCaller(void * callable, Args...  args) {
        // 将不透明的引用转换为具体类型的指针
        using ObjectPtr = typename std::add_pointer<ObjectType>::type;
        auto & Object = *static_cast<ObjectPtr>(callable);

        // 转发调用到具体对象的函数调用
        return Object(std::forward<Args>(args)...);
    }

    // 定义函数调用指针类型
    using CallFunction = Ret(*)(void *, Args...);

    // 存储类型擦除后的函数对象指针
    void* data_;
    // 存储用于调用函数对象的指针
    CallFunction call_function_;
};
```