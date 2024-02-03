# `.\PaddleOCR\deploy\android_demo\app\src\main\java\com\baidu\paddle\lite\demo\ocr\SettingsActivity.java`

```
package com.baidu.paddle.lite.demo.ocr;

import android.content.SharedPreferences;
import android.os.Bundle;
import android.preference.CheckBoxPreference;
import android.preference.EditTextPreference;
import android.preference.ListPreference;

import androidx.appcompat.app.ActionBar;

import java.util.ArrayList;
import java.util.List;

// 设置界面活动类，继承自 AppCompatPreferenceActivity，并实现 SharedPreferences.OnSharedPreferenceChangeListener 接口
public class SettingsActivity extends AppCompatPreferenceActivity implements SharedPreferences.OnSharedPreferenceChangeListener {
    // 声明各种设置项
    ListPreference lpChoosePreInstalledModel = null;
    CheckBoxPreference cbEnableCustomSettings = null;
    EditTextPreference etModelPath = null;
    EditTextPreference etLabelPath = null;
    ListPreference etImagePath = null;
    ListPreference lpCPUThreadNum = null;
    ListPreference lpCPUPowerMode = null;
    EditTextPreference etDetLongSize = null;
    EditTextPreference etScoreThreshold = null;

    // 声明各种预装模型路径、标签路径等列表
    List<String> preInstalledModelPaths = null;
    List<String> preInstalledLabelPaths = null;
    List<String> preInstalledImagePaths = null;
    List<String> preInstalledDetLongSizes = null;
    List<String> preInstalledCPUThreadNums = null;
    List<String> preInstalledCPUPowerModes = null;
    List<String> preInstalledInputColorFormats = null;
    List<String> preInstalledInputMeans = null;
    List<String> preInstalledInputStds = null;
    List<String> preInstalledScoreThresholds = null;

    @Override
    // 重写 onCreate 方法
    }

    }

    @Override
    // 重写 onResume 方法
    protected void onResume() {
        super.onResume();
        // 注册 SharedPreferences 的监听器
        getPreferenceScreen().getSharedPreferences().registerOnSharedPreferenceChangeListener(this);
        // 重新加载设置项并更新 UI
        reloadPreferenceAndUpdateUI();
    }

    @Override
    // 重写 onPause 方法
    protected void onPause() {
        super.onPause();
        // 取消注册 SharedPreferences 的监听器
        getPreferenceScreen().getSharedPreferences().unregisterOnSharedPreferenceChangeListener(this);
    }

    @Override
    // 重写
    // 当共享偏好更改时调用此方法，根据传入的键值判断是否需要执行相应操作
    public void onSharedPreferenceChanged(SharedPreferences sharedPreferences, String key) {
        // 如果传入的键值与预设的模型选择键值相同
        if (key.equals(getString(R.string.CHOOSE_PRE_INSTALLED_MODEL_KEY))) {
            // 获取共享偏好的编辑器
            SharedPreferences.Editor editor = sharedPreferences.edit();
            // 将自定义设置键值设为false
            editor.putBoolean(getString(R.string.ENABLE_CUSTOM_SETTINGS_KEY), false);
            // 提交编辑器的更改
            editor.commit();
        }
        // 重新加载偏好设置并更新用户界面
        reloadPreferenceAndUpdateUI();
    }
# 闭合之前的代码块
```