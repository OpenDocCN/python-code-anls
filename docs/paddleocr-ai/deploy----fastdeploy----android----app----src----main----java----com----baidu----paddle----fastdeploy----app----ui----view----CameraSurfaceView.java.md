# `.\PaddleOCR\deploy\fastdeploy\android\app\src\main\java\com\baidu\paddle\fastdeploy\app\ui\view\CameraSurfaceView.java`

```py
package com.baidu.paddle.fastdeploy.app.ui.view;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.hardware.Camera.CameraInfo;
import android.hardware.Camera.Size;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.opengl.GLSurfaceView.Renderer;
import android.opengl.GLUtils;
import android.opengl.Matrix;
import android.os.SystemClock;
import android.util.AttributeSet;
import android.util.Log;

import com.baidu.paddle.fastdeploy.app.ui.Utils;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.List;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

public class CameraSurfaceView extends GLSurfaceView implements Renderer,
        SurfaceTexture.OnFrameAvailableListener {
    private static final String TAG = CameraSurfaceView.class.getSimpleName();

    // 预期相机预览宽度
    public static int EXPECTED_PREVIEW_WIDTH = 1280;  // 1920
    // 预期相机预览高度
    public static int EXPECTED_PREVIEW_HEIGHT = 720;  // 960

    // 相机数量
    protected int numberOfCameras;
    // 选定的相机ID
    protected int selectedCameraId;
    // 是否禁用相机
    protected boolean disableCamera = false;
    // 相机对象
    protected Camera camera;

    // 上下文
    protected Context context;
    // SurfaceTexture对象
    protected SurfaceTexture surfaceTexture;
    // Surface宽度
    protected int surfaceWidth = 0;
    // Surface高度
    protected int surfaceHeight = 0;
    // 纹理宽度
    protected int textureWidth = 0;
    // 纹理高度
    protected int textureHeight = 0;

    // ARGB8888格式的Bitmap对象
    protected Bitmap ARGB8888ImageBitmap;
    // Bitmap释放模式
    protected boolean bitmapReleaseMode = true;

    // 为了处理相机预览数据并将修改后的数据渲染到屏幕上，创建了三个纹理，并且数据流如下所示：
    // 预览数据->相机纹理ID->帧缓冲纹理ID->绘制纹理ID->帧缓冲
    protected int[] fbo = {0};
    protected int[] camTextureId = {0};
    protected int[] fboTexureId = {0};
    // 存储纹理ID的数组，初始值为0
    protected int[] drawTexureId = {0};

    // 顶点着色器代码
    private final String vss = ""
            + "attribute vec2 vPosition;\n"
            + "attribute vec2 vTexCoord;\n" + "varying vec2 texCoord;\n"
            + "void main() {\n" + "  texCoord = vTexCoord;\n"
            + "  gl_Position = vec4 (vPosition.x, vPosition.y, 0.0, 1.0);\n"
            + "}";

    // 相机到帧缓冲对象的片元着色器代码
    private final String fssCam2FBO = ""
            + "#extension GL_OES_EGL_image_external : require\n"
            + "precision mediump float;\n"
            + "uniform samplerExternalOES sTexture;\n"
            + "varying vec2 texCoord;\n"
            + "void main() {\n"
            + "  gl_FragColor = texture2D(sTexture,texCoord);\n" + "}";

    // 纹理到屏幕的片元着色器代码
    private final String fssTex2Screen = ""
            + "precision mediump float;\n"
            + "uniform sampler2D sTexture;\n"
            + "varying vec2 texCoord;\n"
            + "void main() {\n"
            + "  gl_FragColor = texture2D(sTexture,texCoord);\n" + "}";

    // 顶点坐标数组
    private final float[] vertexCoords = {
            -1, -1,
            -1, 1,
            1, -1,
            1, 1};
    // 纹理坐标数组
    private float[] textureCoords = {
            0, 1,
            0, 0,
            1, 1,
            1, 0};

    // 顶点坐标缓冲区
    private FloatBuffer vertexCoordsBuffer;
    // 纹理坐标缓冲区
    private FloatBuffer textureCoordsBuffer;

    // 相机到帧缓冲对象的着色器程序ID
    private int progCam2FBO = -1;
    // 纹理到屏幕的着色器程序ID
    private int progTex2Screen = -1;
    // 相机到帧缓冲对象的顶点坐标属性位置
    private int vcCam2FBO;
    // 相机到帧缓冲对象的纹理坐标属性位置
    private int tcCam2FBO;
    // 纹理到屏幕的顶点坐标属性位置
    private int vcTex2Screen;
    // 纹理到屏幕的纹理坐标属性位置
    private int tcTex2Screen;

    // 设置位图释放模式
    public void setBitmapReleaseMode(boolean mode) {
        synchronized (this) {
            bitmapReleaseMode = mode;
        }
    }

    // 获取位图
    public Bitmap getBitmap() {
        return ARGB8888ImageBitmap; // 可能为null或已回收
    }

    // 纹理变化监听器接口
    public interface OnTextureChangedListener {
        boolean onTextureChanged(Bitmap ARGB8888ImageBitmap);
    }

    // 纹理变化监听器
    private OnTextureChangedListener onTextureChangedListener = null;
    // 设置纹理变化监听器
    public void setOnTextureChangedListener(OnTextureChangedListener listener) {
        onTextureChangedListener = listener;
    }

    // 相机表面视图构造函数
    public CameraSurfaceView(Context ctx, AttributeSet attrs) {
        super(ctx, attrs);
        context = ctx;
        // 设置 OpenGL ES 版本为 2
        setEGLContextClientVersion(2);
        // 设置渲染器为当前对象
        setRenderer(this);
        // 设置渲染模式为手动触发
        setRenderMode(RENDERMODE_WHEN_DIRTY);

        // 查找可用相机总数和默认相机的 ID
        numberOfCameras = Camera.getNumberOfCameras();
        CameraInfo cameraInfo = new CameraInfo();
        for (int i = 0; i < numberOfCameras; i++) {
            Camera.getCameraInfo(i, cameraInfo);
            if (cameraInfo.facing == CameraInfo.CAMERA_FACING_BACK) {
                selectedCameraId = i;
            }
        }
    }

    @Override
    // 当表面大小改变时调用
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        surfaceWidth = width;
        surfaceHeight = height;
        // 打开相机
        openCamera();
    }

    @Override
    // 当表面创建时调用
    }

    @Override
    // 当表面销毁时调用
    }

    // 转换纹理坐标
    private float[] transformTextureCoordinates(float[] coords, float[] matrix) {
        float[] result = new float[coords.length];
        float[] vt = new float[4];
        for (int i = 0; i < coords.length; i += 2) {
            float[] v = {coords[i], coords[i + 1], 0, 1};
            Matrix.multiplyMV(vt, 0, matrix, 0, v, 0);
            result[i] = vt[0];
            result[i + 1] = vt[1];
        }
        return result;
    }

    @Override
    // 当恢复时调用
    public void onResume() {
        super.onResume();
    }

    @Override
    // 当暂停时调用
    public void onPause() {
        super.onPause();
        // 释放相机资源
        releaseCamera();
    }

    @Override
    // 当帧可用时调用
    public void onFrameAvailable(SurfaceTexture surfaceTexture) {
        // 请求渲染
        requestRender();
    }

    // 禁用相机
    public void disableCamera() {
        disableCamera = true;
    }

    // 启用相机
    public void enableCamera() {
        disableCamera = false;
    }
    // 切换相机功能，释放当前相机资源，选择下一个相机，打开新相机
    public void switchCamera() {
        // 释放当前相机资源
        releaseCamera();
        // 选择下一个相机
        selectedCameraId = (selectedCameraId + 1) % numberOfCameras;
        // 打开新相机
        openCamera();
    }

    // 释放相机资源
    public void releaseCamera() {
        // 如果相机对象不为空
        if (camera != null) {
            // 设置预览回调为空
            camera.setPreviewCallback(null);
            // 停止预览
            camera.stopPreview();
            // 释放相机资源
            camera.release();
            // 将相机对象置为空
            camera = null;
        }
    }
# 闭合之前的代码块
```