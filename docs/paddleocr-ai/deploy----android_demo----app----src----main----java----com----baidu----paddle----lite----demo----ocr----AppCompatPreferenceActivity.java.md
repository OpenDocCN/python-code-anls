# `.\PaddleOCR\deploy\android_demo\app\src\main\java\com\baidu\paddle\lite\demo\ocr\AppCompatPreferenceActivity.java`

```py
/*
 * 版权所有 (C) 2014 The Android Open Source Project
 *
 * 根据 Apache 许可证，版本 2.0 授权;
 * 除非符合许可证的规定，否则不得使用此文件。
 * 您可以在以下网址获取许可证的副本:
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * 除非适用法律要求或书面同意，否则根据许可证分发的软件
 * 均基于“按原样”分发，不附带任何担保或条件，无论是明示的还是暗示的。
 * 请查看许可证以获取特定语言的权限和限制。
 */

// 导入必要的类
package com.baidu.paddle.lite.demo.ocr;

import android.content.res.Configuration;
import android.os.Bundle;
import android.preference.PreferenceActivity;
import android.view.MenuInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.LayoutRes;
import androidx.annotation.Nullable;
import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatDelegate;
import androidx.appcompat.widget.Toolbar;

/**
 * 一个实现并代理必要调用以与 AppCompat 一起使用的 {@link PreferenceActivity}。
 * <p>
 * 这种技术可以用于 {@link android.app.Activity} 类，而不仅仅是 {@link PreferenceActivity}。
 */
public abstract class AppCompatPreferenceActivity extends PreferenceActivity {
    // AppCompat 代理对象
    private AppCompatDelegate mDelegate;

    // 在创建时安装视图工厂并调用代理的 onCreate 方法
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        getDelegate().installViewFactory();
        getDelegate().onCreate(savedInstanceState);
        super.onCreate(savedInstanceState);
    }

    // 在创建后调用代理的 onPostCreate 方法
    @Override
    protected void onPostCreate(Bundle savedInstanceState) {
        super.onPostCreate(savedInstanceState);
        getDelegate().onPostCreate(savedInstanceState);
    }

    // 获取 ActionBar 对象
    public ActionBar getSupportActionBar() {
        return getDelegate().getSupportActionBar();
    }
}
    // 设置支持 Action Bar，将传入的 Toolbar 对象设置为 Action Bar
    public void setSupportActionBar(@Nullable Toolbar toolbar) {
        getDelegate().setSupportActionBar(toolbar);
    }

    // 获取菜单填充器
    @Override
    public MenuInflater getMenuInflater() {
        return getDelegate().getMenuInflater();
    }

    // 设置 Activity 的内容视图为指定布局资源
    @Override
    public void setContentView(@LayoutRes int layoutResID) {
        getDelegate().setContentView(layoutResID);
    }

    // 设置 Activity 的内容视图为指定 View 对象
    @Override
    public void setContentView(View view) {
        getDelegate().setContentView(view);
    }

    // 设置 Activity 的内容视图为指定 View 对象，并指定布局参数
    @Override
    public void setContentView(View view, ViewGroup.LayoutParams params) {
        getDelegate().setContentView(view, params);
    }

    // 添加额外的内容视图到 Activity 中，并指定布局参数
    @Override
    public void addContentView(View view, ViewGroup.LayoutParams params) {
        getDelegate().addContentView(view, params);
    }

    // 在 Activity 恢复后执行的操作
    @Override
    protected void onPostResume() {
        super.onPostResume();
        getDelegate().onPostResume();
    }

    // 当标题改变时执行的操作
    @Override
    protected void onTitleChanged(CharSequence title, int color) {
        super.onTitleChanged(title, color);
        getDelegate().setTitle(title);
    }

    // 当配置改变时执行的操作
    @Override
    public void onConfigurationChanged(Configuration newConfig) {
        super.onConfigurationChanged(newConfig);
        getDelegate().onConfigurationChanged(newConfig);
    }

    // 当 Activity 停止时执行的操作
    @Override
    protected void onStop() {
        super.onStop();
        getDelegate().onStop();
    }

    // 当 Activity 销毁时执行的操作
    @Override
    protected void onDestroy() {
        super.onDestroy();
        getDelegate().onDestroy();
    }

    // 使选项菜单无效化
    public void invalidateOptionsMenu() {
        getDelegate().invalidateOptionsMenu();
    }

    // 获取 AppCompatDelegate 对象，如果为空则创建一个新的
    private AppCompatDelegate getDelegate() {
        if (mDelegate == null) {
            mDelegate = AppCompatDelegate.create(this, null);
        }
        return mDelegate;
    }
# 闭合之前的代码块
```