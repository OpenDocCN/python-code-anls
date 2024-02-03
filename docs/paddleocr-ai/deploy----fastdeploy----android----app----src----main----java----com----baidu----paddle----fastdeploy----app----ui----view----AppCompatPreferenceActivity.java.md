# `.\PaddleOCR\deploy\fastdeploy\android\app\src\main\java\com\baidu\paddle\fastdeploy\app\ui\view\AppCompatPreferenceActivity.java`

```
package com.baidu.paddle.fastdeploy.app.ui.view;

import android.content.res.Configuration;
import android.os.Bundle;
import android.preference.PreferenceActivity;
import android.support.annotation.LayoutRes;
import android.support.annotation.Nullable;
importandroid.support.v7.app.ActionBar;
import android.support.v7.app.AppCompatDelegate;
import android.support.v7.widget.Toolbar;
import android.view.MenuInflater;
import android.view.View;
import android.view.ViewGroup;

/**
 * A {@link PreferenceActivity} which implements and proxies the necessary calls
 * to be used with AppCompat.
 * <p>
 * This technique can be used with an {@link android.app.Activity} class, not just
 * {@link PreferenceActivity}.
 */
public abstract class AppCompatPreferenceActivity extends PreferenceActivity {
    private AppCompatDelegate mDelegate;

    // 在 Activity 创建时调用
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        // 安装视图工厂
        getDelegate().installViewFactory();
        // 调用代理的 onCreate 方法
        getDelegate().onCreate(savedInstanceState);
        // 调用父类的 onCreate 方法
        super.onCreate(savedInstanceState);
    }

    // 在 Activity 创建后调用
    @Override
    protected void onPostCreate(Bundle savedInstanceState) {
        // 调用父类的 onPostCreate 方法
        super.onPostCreate(savedInstanceState);
        // 调用代理的 onPostCreate 方法
        getDelegate().onPostCreate(savedInstanceState);
    }

    // 获取 ActionBar
    public ActionBar getSupportActionBar() {
        return getDelegate().getSupportActionBar();
    }

    // 设置 Toolbar 作为 ActionBar
    public void setSupportActionBar(@Nullable Toolbar toolbar) {
        getDelegate().setSupportActionBar(toolbar);
    }

    // 获取 MenuInflater
    @Override
    public MenuInflater getMenuInflater() {
        return getDelegate().getMenuInflater();
    }

    // 设置 Activity 的内容视图
    @Override
    public void setContentView(@LayoutRes int layoutResID) {
        getDelegate().setContentView(layoutResID);
    }

    // 设置 Activity 的内容视图
    @Override
    public void setContentView(View view) {
        getDelegate().setContentView(view);
    }

    // 设置 Activity 的内容视图
    @Override
    public void setContentView(View view, ViewGroup.LayoutParams params) {
        getDelegate().setContentView(view, params);
    }

    @Override
    // 将指定的视图添加到内容视图中，使用给定的布局参数
    public void addContentView(View view, ViewGroup.LayoutParams params) {
        getDelegate().addContentView(view, params);
    }

    // 在 Activity 恢复后调用，执行默认的操作并通知代理对象
    @Override
    protected void onPostResume() {
        super.onPostResume();
        getDelegate().onPostResume();
    }

    // 当标题改变时调用，执行默认的操作并设置代理对象的标题
    @Override
    protected void onTitleChanged(CharSequence title, int color) {
        super.onTitleChanged(title, color);
        getDelegate().setTitle(title);
    }

    // 当配置改变时调用，执行默认的操作并通知代理对象
    @Override
    public void onConfigurationChanged(Configuration newConfig) {
        super.onConfigurationChanged(newConfig);
        getDelegate().onConfigurationChanged(newConfig);
    }

    // 当 Activity 停止时调用，执行默认的操作并通知代理对象
    @Override
    protected void onStop() {
        super.onStop();
        getDelegate().onStop();
    }

    // 当 Activity 销毁时调用，执行默认的操作并通知代理对象
    @Override
    protected void onDestroy() {
        super.onDestroy();
        getDelegate().onDestroy();
    }

    // 使选项菜单无效化，通知代理对象
    public void invalidateOptionsMenu() {
        getDelegate().invalidateOptionsMenu();
    }

    // 获取代理对象，如果为空则创建一个新的代理对象
    private AppCompatDelegate getDelegate() {
        if (mDelegate == null) {
            mDelegate = AppCompatDelegate.create(this, null);
        }
        return mDelegate;
    }
# 闭合之前的代码块
```