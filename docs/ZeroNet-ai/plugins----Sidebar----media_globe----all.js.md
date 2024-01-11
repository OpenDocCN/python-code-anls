# `ZeroNet\plugins\Sidebar\media_globe\all.js`

```
/* ---- plugins/Sidebar/media_globe/Detector.js ---- */

/**
 * @author alteredq / http://alteredqualia.com/
 * @author mr.doob / http://mrdoob.com/
 */

// 创建名为 Detector 的对象
Detector = {
  // 检测浏览器是否支持 canvas 元素
  canvas : !! window.CanvasRenderingContext2D,
  // 检测浏览器是否支持 WebGL
  webgl : ( function () { try { return !! window.WebGLRenderingContext && !! document.createElement( 'canvas' ).getContext( 'experimental-webgl' ); } catch( e ) { return false; } } )(),
  // 检测浏览器是否支持 Web Workers
  workers : !! window.Worker,
  // 检测浏览器是否支持 File API
  fileapi : window.File && window.FileReader && window.FileList && window.Blob,

  // 获取 WebGL 错误信息
  getWebGLErrorMessage : function () {
    // 创建一个 div 元素
    var domElement = document.createElement( 'div' );
    // 设置 div 元素的样式
    domElement.style.fontFamily = 'monospace';
    domElement.style.fontSize = '13px';
    domElement.style.textAlign = 'center';
    domElement.style.background = '#eee';
    domElement.style.color = '#000';
    domElement.style.padding = '1em';
    domElement.style.width = '475px';
    domElement.style.margin = '5em auto 0';

    // 如果浏览器不支持 WebGL
    if ( ! this.webgl ) {
      // 根据浏览器支持情况设置不同的错误信息
      domElement.innerHTML = window.WebGLRenderingContext ? [
        'Sorry, your graphics card doesn\'t support <a href="http://khronos.org/webgl/wiki/Getting_a_WebGL_Implementation">WebGL</a>'
      ].join( '\n' ) : [
        'Sorry, your browser doesn\'t support <a href="http://khronos.org/webgl/wiki/Getting_a_WebGL_Implementation">WebGL</a><br/>',
        'Please try with',
        '<a href="http://www.google.com/chrome">Chrome</a>, ',
        '<a href="http://www.mozilla.com/en-US/firefox/new/">Firefox 4</a> or',
        '<a href="http://nightly.webkit.org/">Webkit Nightly (Mac)</a>'
      ].join( '\n' );
    }

    return domElement;
  },

  // 添加获取 WebGL 错误信息的方法
  addGetWebGLMessage : function ( parameters ) {
    var parent, id, domElement;
    parameters = parameters || {};
    // 获取父元素和 ID
    parent = parameters.parent !== undefined ? parameters.parent : document.body;
    id = parameters.id !== undefined ? parameters.id : 'oldie';
    // 获取 WebGL 错误信息的 div 元素
    domElement = Detector.getWebGLErrorMessage();
    domElement.id = id;
    # 将 domElement 作为子节点添加到 parent 节点中
    parent.appendChild( domElement );
  }
// 定义 TWEEN 对象，如果已经存在则使用已有的对象
var TWEEN=TWEEN||function(){
    // 定义变量和数组
    var a,e,c,d,f=[];
    // 返回包含各种方法的对象
    return{
        // 开始动画循环
        start:function(g){
            c=setInterval(this.update,1E3/(g||60))
        },
        // 停止动画循环
        stop:function(){
            clearInterval(c)
        },
        // 添加动画对象到数组
        add:function(g){
            f.push(g)
        },
        // 从数组中移除动画对象
        remove:function(g){
            a=f.indexOf(g);
            a!==-1&&f.splice(a,1)
        },
        // 更新动画对象状态
        update:function(){
            a=0;
            e=f.length;
            for(d=(new Date).getTime();a<e;){
                if(f[a].update(d))a++;
                else{
                    f.splice(a,1);
                    e--
                }
            }
        }
    }
}();
// 定义 Tween 类
TWEEN.Tween=function(a){
    var e={},c={},d={},f=1E3,g=0,j=null,n=TWEEN.Easing.Linear.EaseNone,k=null,l=null,m=null;
    // 设置目标值和动画时长
    this.to=function(b,h){
        if(h!==null)f=h;
        for(var i in b)
            if(a[i]!==null)d[i]=b[i];
        return this
    };
    // 开始动画
    this.start=function(){
        TWEEN.add(this);
        j=(new Date).getTime()+g;
        for(var b in d)
            if(a[b]!==null){
                e[b]=a[b];
                c[b]=d[b]-a[b]
            }
        return this
    };
    // 停止动画
    this.stop=function(){
        TWEEN.remove(this);
        return this
    };
    // 设置延迟时间
    this.delay=function(b){
        g=b;
        return this
    };
    // 设置缓动函数
    this.easing=function(b){
        n=b;
        return this
    };
    // 设置链式动画
    this.chain=function(b){
        k=b
    };
    // 设置更新回调函数
    this.onUpdate=function(b){
        l=b;
        return this
    };
    // 设置完成回调函数
    this.onComplete=function(b){
        m=b;
        return this
    };
    // 更新动画状态
    this.update=function(b){
        var h,i;
        if(b<j)return true;
        b=(b-j)/f;
        b=b>1?1:b;
        i=n(b);
        for(h in c)
            a[h]=e[h]+c[h]*i;
        l!==null&&l.call(a,i);
        if(b==1){
            m!==null&&m.call(a);
            k!==null&&k.start();
            return false
        }
        return true
    }
};
// 定义各种缓动函数
TWEEN.Easing={
    Linear:{},
    Quadratic:{},
    Cubic:{},
    Quartic:{},
    Quintic:{},
    Sinusoidal:{},
    Exponential:{},
    Circular:{},
    Elastic:{},
    Back:{},
    Bounce:{}
};
// 定义线性缓动函数
TWEEN.Easing.Linear.EaseNone=function(a){
    return a
};
// 定义二次缓动函数
TWEEN.Easing.Quadratic.EaseIn=function(a){
    return a*a
};
TWEEN.Easing.Quadratic.EaseOut=function(a){
    return-a*(a-2)
};
TWEEN.Easing.Quadratic.EaseInOut=function(a){
    if((a*=2)<1)return 0.5*a*a;
    return-0.5*(--a*(a-2)-1)
};
TWEEN.Easing.Cubic.EaseIn=function(a){
    return a*a*a
};
TWEEN.Easing.Cubic.EaseOut=function(a){
    return--a*a*a+1
};
TWEEN.Easing.Cubic.EaseInOut=function(a){
    if((a*=2)<1)return 0.5*a*a*a;
    return 0.5*((a-=2)*a*a+2)
};
TWEEN.Easing.Quartic.EaseIn=function(a){
    return a*a*a*a
};
# Quartic 缓动函数的 EaseOut 版本，根据输入参数计算缓动效果
TWEEN.Easing.Quartic.EaseOut=function(a){return-(--a*a*a*a-1)};
# Quartic 缓动函数的 EaseInOut 版本，根据输入参数计算缓动效果
TWEEN.Easing.Quartic.EaseInOut=function(a){if((a*=2)<1)return 0.5*a*a*a*a;return-0.5*((a-=2)*a*a*a-2)};
# Quintic 缓动函数的 EaseIn 版本，根据输入参数计算缓动效果
TWEEN.Easing.Quintic.EaseIn=function(a){return a*a*a*a*a};
# Quintic 缓动函数的 EaseOut 版本，根据输入参数计算缓动效果
TWEEN.Easing.Quintic.EaseOut=function(a){return(a-=1)*a*a*a*a+1};
# Quintic 缓动函数的 EaseInOut 版本，根据输入参数计算缓动效果
TWEEN.Easing.Quintic.EaseInOut=function(a){if((a*=2)<1)return 0.5*a*a*a*a*a;return 0.5*((a-=2)*a*a*a*a+2)};
# Sinusoidal 缓动函数的 EaseIn 版本，根据输入参数计算缓动效果
TWEEN.Easing.Sinusoidal.EaseIn=function(a){return-Math.cos(a*Math.PI/2)+1};
# Sinusoidal 缓动函数的 EaseOut 版本，根据输入参数计算缓动效果
TWEEN.Easing.Sinusoidal.EaseOut=function(a){return Math.sin(a*Math.PI/2)};
# Sinusoidal 缓动函数的 EaseInOut 版本，根据输入参数计算缓动效果
TWEEN.Easing.Sinusoidal.EaseInOut=function(a){return-0.5*(Math.cos(Math.PI*a)-1)};
# Exponential 缓动函数的 EaseIn 版本，根据输入参数计算缓动效果
TWEEN.Easing.Exponential.EaseIn=function(a){return a==0?0:Math.pow(2,10*(a-1))};
# Exponential 缓动函数的 EaseOut 版本，根据输入参数计算缓动效果
TWEEN.Easing.Exponential.EaseOut=function(a){return a==1?1:-Math.pow(2,-10*a)+1};
# Exponential 缓动函数的 EaseInOut 版本，根据输入参数计算缓动效果
TWEEN.Easing.Exponential.EaseInOut=function(a){if(a==0)return 0;if(a==1)return 1;if((a*=2)<1)return 0.5*Math.pow(2,10*(a-1));return 0.5*(-Math.pow(2,-10*(a-1))+2)};
# Circular 缓动函数的 EaseIn 版本，根据输入参数计算缓动效果
TWEEN.Easing.Circular.EaseIn=function(a){return-(Math.sqrt(1-a*a)-1)};
# Circular 缓动函数的 EaseOut 版本，根据输入参数计算缓动效果
TWEEN.Easing.Circular.EaseOut=function(a){return Math.sqrt(1- --a*a)};
# Circular 缓动函数的 EaseInOut 版本，根据输入参数计算缓动效果
TWEEN.Easing.Circular.EaseInOut=function(a){if((a/=0.5)<1)return-0.5*(Math.sqrt(1-a*a)-1);return 0.5*(Math.sqrt(1-(a-=2)*a)+1)};
# Elastic 缓动函数的 EaseIn 版本，根据输入参数计算缓动效果
TWEEN.Easing.Elastic.EaseIn=function(a){var e,c=0.1,d=0.4;if(a==0)return 0;if(a==1)return 1;d||(d=0.3);if(!c||c<1){c=1;e=d/4}else e=d/(2*Math.PI)*Math.asin(1/c);return-(c*Math.pow(2,10*(a-=1))*Math.sin((a-e)*2*Math.PI/d))};
# Elastic 缓动函数的 EaseOut 版本，根据输入参数计算缓动效果
TWEEN.Easing.Elastic.EaseOut=function(a){var e,c=0.1,d=0.4;if(a==0)return 0;if(a==1)return 1;d||(d=0.3);if(!c||c<1){c=1;e=d/4}else e=d/(2*Math.PI)*Math.asin(1/c);return c*Math.pow(2,-10*a)*Math.sin((a-e)*2*Math.PI/d)+1};
// 定义了一系列缓动函数，用于在动画中实现不同的缓动效果
TWEEN.Easing.Elastic.EaseInOut=function(a){
    var e,c=0.1,d=0.4;
    if(a==0) return 0;
    if(a==1) return 1;
    d||(d=0.3);
    if(!c||c<1){
        c=1;
        e=d/4
    } else {
        e=d/(2*Math.PI)*Math.asin(1/c);
    }
    if((a*=2)<1) return -0.5*c*Math.pow(2,10*(a-=1))*Math.sin((a-e)*2*Math.PI/d);
    return c*Math.pow(2,-10*(a-=1))*Math.sin((a-e)*2*Math.PI/d)*0.5+1
};
TWEEN.Easing.Back.EaseIn=function(a){
    return a*a*(2.70158*a-1.70158)
};
TWEEN.Easing.Back.EaseOut=function(a){
    return(a-=1)*a*(2.70158*a+1.70158)+1
};
TWEEN.Easing.Back.EaseInOut=function(a){
    if((a*=2)<1) return 0.5*a*a*(3.5949095*a-2.5949095);
    return 0.5*((a-=2)*a*(3.5949095*a+2.5949095)+2)
};
TWEEN.Easing.Bounce.EaseIn=function(a){
    return 1-TWEEN.Easing.Bounce.EaseOut(1-a)
};
TWEEN.Easing.Bounce.EaseOut=function(a){
    return(a/=1)<1/2.75?7.5625*a*a:a<2/2.75?7.5625*(a-=1.5/2.75)*a+0.75:a<2.5/2.75?7.5625*(a-=2.25/2.75)*a+0.9375:7.5625*(a-=2.625/2.75)*a+0.984375
};
TWEEN.Easing.Bounce.EaseInOut=function(a){
    if(a<0.5) return TWEEN.Easing.Bounce.EaseIn(a*2)*0.5;
    return TWEEN.Easing.Bounce.EaseOut(a*2-1)*0.5+0.5
};

// 定义了一个全局变量 DAT，如果已经存在则使用已有的，否则创建一个新的空对象
// 创建了一个 DAT.Globe 的构造函数，用于生成一个 WebGL 地球模型
DAT.Globe = function(container, opts) {
  opts = opts || {};

  // 定义颜色函数，用于根据输入值生成颜色
  var colorFn = opts.colorFn || function(x) {
    var c = new THREE.Color();
    c.setHSL( ( 0.5 - (x * 2) ), Math.max(0.8, 1.0 - (x * 3)), 0.5 );
    return c;
  };
  // 定义图片目录，默认为 '/globe/'
  var imgDir = opts.imgDir || '/globe/';

  // 定义了一个 Shaders 对象，用于存储 WebGL 的着色器代码
    'earth' : {
      uniforms: {
        'texture': { type: 't', value: null }  // 定义地球的纹理
      },
      vertexShader: [
        'varying vec3 vNormal;',  // 定义顶点着色器中的变量
        'varying vec2 vUv;',  // 定义顶点着色器中的变量
        'void main() {',  // 定义顶点着色器中的主函数
          'gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );',  // 计算顶点位置
          'vNormal = normalize( normalMatrix * normal );',  // 计算法向量
          'vUv = uv;',  // 计算纹理坐标
        '}'
      ].join('\n'),  // 将顶点着色器代码拼接成字符串
      fragmentShader: [
        'uniform sampler2D texture;',  // 定义片元着色器中的纹理
        'varying vec3 vNormal;',  // 定义片元着色器中的变量
        'varying vec2 vUv;',  // 定义片元着色器中的变量
        'void main() {',  // 定义片元着色器中的主函数
          'vec3 diffuse = texture2D( texture, vUv ).xyz;',  // 计算漫反射颜色
          'float intensity = 1.05 - dot( vNormal, vec3( 0.0, 0.0, 1.0 ) );',  // 计算光照强度
          'vec3 atmosphere = vec3( 1.0, 1.0, 1.0 ) * pow( intensity, 3.0 );',  // 计算大气层颜色
          'gl_FragColor = vec4( diffuse + atmosphere, 1.0 );',  // 设置片元颜色
        '}'
      ].join('\n')  // 将片元着色器代码拼接成字符串
    },
    'atmosphere' : {
      uniforms: {},  // 定义大气层的uniforms
      vertexShader: [
        'varying vec3 vNormal;',  // 定义顶点着色器中的变量
        'void main() {',  // 定义顶点着色器中的主函数
          'vNormal = normalize( normalMatrix * normal );',  // 计算法向量
          'gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );',  // 计算顶点位置
        '}'
      ].join('\n'),  // 将顶点着色器代码拼接成字符串
      fragmentShader: [
        'varying vec3 vNormal;',  // 定义片元着色器中的变量
        'void main() {',  // 定义片元着色器中的主函数
          'float intensity = pow( 0.8 - dot( vNormal, vec3( 0, 0, 1.0 ) ), 12.0 );',  // 计算光照强度
          'gl_FragColor = vec4( 1.0, 1.0, 1.0, 1.0 ) * intensity;',  // 设置片元颜色
        '}'
      ].join('\n')  // 将片元着色器代码拼接成字符串
    }
  };

  var camera, scene, renderer, w, h;  // 定义相机、场景、渲染器、宽度和高度变量
  var mesh, atmosphere, point, running;  // 定义网格、大气层、点、运行变量

  var overRenderer;  // 定义鼠标是否在渲染器上的变量
  var running = true;  // 定义运行状态变量

  var curZoomSpeed = 0;  // 定义当前缩放速度变量
  var zoomSpeed = 50;  // 定义缩放速度变量

  var mouse = { x: 0, y: 0 }, mouseOnDown = { x: 0, y: 0 };  // 定义鼠标位置和按下时的鼠标位置
  var rotation = { x: 0, y: 0 },  // 定义旋转角度
      target = { x: Math.PI*3/2, y: Math.PI / 6.0 },  // 定义目标角度
      targetOnDown = { x: 0, y: 0 };  // 定义按下时的目标角度

  var distance = 100000, distanceTarget = 100000;  // 定义距离和目标距离
  var padding = 10;  // 定义填充值
  var PI_HALF = Math.PI / 2;  // 定义 PI 的一半

  function init() {

    container.style.color = '#fff';  // 设置容器的颜色为白色
    # 设置容器的字体样式
    container.style.font = '13px/20px Arial, sans-serif';
    
    # 声明变量 shader, uniforms, material
    var shader, uniforms, material;
    
    # 获取容器的宽度和高度
    w = container.offsetWidth || window.innerWidth;
    h = container.offsetHeight || window.innerHeight;
    
    # 创建透视相机对象
    camera = new THREE.PerspectiveCamera(30, w / h, 1, 10000);
    camera.position.z = distance;
    
    # 创建场景对象
    scene = new THREE.Scene();
    
    # 创建球体几何体
    var geometry = new THREE.SphereGeometry(200, 40, 30);
    
    # 获取地球着色器并克隆其 uniforms
    shader = Shaders['earth'];
    uniforms = THREE.UniformsUtils.clone(shader.uniforms);
    
    # 设置地球着色器的纹理
    uniforms['texture'].value = THREE.ImageUtils.loadTexture(imgDir+'world.jpg');
    
    # 创建地球的着色材质
    material = new THREE.ShaderMaterial({
          uniforms: uniforms,
          vertexShader: shader.vertexShader,
          fragmentShader: shader.fragmentShader
        });
    
    # 创建地球网格并添加到场景中
    mesh = new THREE.Mesh(geometry, material);
    mesh.rotation.y = Math.PI;
    scene.add(mesh);
    
    # 获取大气着色器并克隆其 uniforms
    shader = Shaders['atmosphere'];
    uniforms = THREE.UniformsUtils.clone(shader.uniforms);
    
    # 创建大气的着色材质
    material = new THREE.ShaderMaterial({
          uniforms: uniforms,
          vertexShader: shader.vertexShader,
          fragmentShader: shader.fragmentShader,
          side: THREE.BackSide,
          blending: THREE.AdditiveBlending,
          transparent: true
        });
    
    # 创建大气网格并添加到场景中
    mesh = new THREE.Mesh(geometry, material);
    mesh.scale.set( 1.1, 1.1, 1.1 );
    scene.add(mesh);
    
    # 创建盒子几何体
    geometry = new THREE.BoxGeometry(2.75, 2.75, 1);
    geometry.applyMatrix(new THREE.Matrix4().makeTranslation(0,0,-0.5));
    
    # 创建点网格
    point = new THREE.Mesh(geometry);
    
    # 创建 WebGL 渲染器
    renderer = new THREE.WebGLRenderer({antialias: true});
    renderer.setSize(w, h);
    renderer.setClearColor( 0x212121, 1 );
    
    # 设置渲染器的样式
    renderer.domElement.style.position = 'relative';
    
    # 将渲染器的 DOM 元素添加到容器中
    container.appendChild(renderer.domElement);
    
    # 添加鼠标按下事件监听
    container.addEventListener('mousedown', onMouseDown, false);
    
    # 根据浏览器支持情况添加滚轮事件监听
    if ('onwheel' in document) {
      container.addEventListener('wheel', onMouseWheel, false);
    } else {
      container.addEventListener('mousewheel', onMouseWheel, false);
    }
    # 当文档接收到键盘按键事件时，调用 onDocumentKeyDown 函数
    document.addEventListener('keydown', onDocumentKeyDown, false);

    # 当窗口大小改变时，调用 onWindowResize 函数
    window.addEventListener('resize', onWindowResize, false);

    # 当鼠标移入容器时，设置 overRenderer 为 true
    container.addEventListener('mouseover', function() {
      overRenderer = true;
    }, false);

    # 当鼠标移出容器时，设置 overRenderer 为 false
    container.addEventListener('mouseout', function() {
      overRenderer = false;
    }, false);
  }

  # 添加数据到场景中
  function addData(data, opts) {
    var lat, lng, size, color, i, step, colorFnWrapper;

    # 如果 opts.animated 未定义，则设置为 false
    opts.animated = opts.animated || false;
    # 设置 this.is_animated 为 opts.animated
    this.is_animated = opts.animated;
    # 如果 opts.format 未定义，则设置为 'magnitude'
    opts.format = opts.format || 'magnitude'; // other option is 'legend'
    # 根据 opts.format 的值设置 step 和 colorFnWrapper
    if (opts.format === 'magnitude') {
      step = 3;
      colorFnWrapper = function(data, i) { return colorFn(data[i+2]); }
    } else if (opts.format === 'legend') {
      step = 4;
      colorFnWrapper = function(data, i) { return colorFn(data[i+3]); }
    } else if (opts.format === 'peer') {
      colorFnWrapper = function(data, i) { return colorFn(data[i+2]); }
    } else {
      throw('error: format not supported: '+opts.format);
    }

    # 如果 opts.animated 为 true
    if (opts.animated) {
      # 如果 this._baseGeometry 未定义
      if (this._baseGeometry === undefined) {
        # 创建一个新的 THREE.Geometry 对象
        this._baseGeometry = new THREE.Geometry();
        # 遍历数据，每次增加 step 步长
        for (i = 0; i < data.length; i += step) {
          lat = data[i];
          lng = data[i + 1];
  // 设置颜色和大小
  color = colorFnWrapper(data,i);
  size = 0;
  // 添加点的经纬度、大小、颜色到基础几何体
  addPoint(lat, lng, size, color, this._baseGeometry);

  // 检查是否存在形态目标 ID，如果不存在则初始化为 0，否则加一
  if(this._morphTargetId === undefined) {
    this._morphTargetId = 0;
  } else {
    this._morphTargetId += 1;
  }
  // 设置选项的名称，如果不存在则默认为 'morphTarget'+this._morphTargetId
  opts.name = opts.name || 'morphTarget'+this._morphTargetId;

  // 创建一个新的几何体
  var subgeo = new THREE.Geometry();
  // 遍历数据，每隔 step 个元素取一次值
  for (i = 0; i < data.length; i += step) {
    lat = data[i];
    lng = data[i + 1];
    // 获取颜色
    color = colorFnWrapper(data,i);
    // 获取大小
    size = data[i + 2];
    // 将大小乘以 200
    size = size*200;
    // 添加点的经纬度、大小、颜色到子几何体
    addPoint(lat, lng, size, color, subgeo);
  }
  // 如果选项中设置了动画，则将子几何体的顶点添加到基础几何体的形态目标中
  if (opts.animated) {
    this._baseGeometry.morphTargets.push({'name': opts.name, vertices: subgeo.vertices});
  } else {
    // 否则将子几何体赋值给基础几何体
    this._baseGeometry = subgeo;
  }

};

// 创建点
function createPoints() {
  if (this._baseGeometry !== undefined) {
    if (this.is_animated === false) {
      // 如果不是动画，则创建一个基础材质的网格
      this.points = new THREE.Mesh(this._baseGeometry, new THREE.MeshBasicMaterial({
            color: 0xffffff,
            vertexColors: THREE.FaceColors,
            morphTargets: false
          }));
    } else {
      if (this._baseGeometry.morphTargets.length < 8) {
        // 如果形态目标数量小于 8，则添加填充形态目标
        var padding = 8-this._baseGeometry.morphTargets.length;
        for(var i=0; i<=padding; i++) {
          this._baseGeometry.morphTargets.push({'name': 'morphPadding'+i, vertices: this._baseGeometry.vertices});
        }
      }
      // 创建一个具有形态目标的基础材质的网格
      this.points = new THREE.Mesh(this._baseGeometry, new THREE.MeshBasicMaterial({
            color: 0xffffff,
            vertexColors: THREE.FaceColors,
            morphTargets: true
          }));
    }
    // 将点添加到场景中
    scene.add(this.points);
  }
}

// 添加点的函数
function addPoint(lat, lng, size, color, subgeo) {
  // 计算球面坐标
  var phi = (90 - lat) * Math.PI / 180;
    // 根据经度计算角度
    var theta = (180 - lng) * Math.PI / 180;

    // 根据球面坐标系计算点的位置
    point.position.x = 200 * Math.sin(phi) * Math.cos(theta);
    point.position.y = 200 * Math.cos(phi);
    point.position.z = 200 * Math.sin(phi) * Math.sin(theta);

    // 让点朝向目标物体的位置
    point.lookAt(mesh.position);

    // 设置点的缩放，并更新其矩阵
    point.scale.z = Math.max( size, 0.1 ); // 避免非可逆矩阵
    point.updateMatrix();

    // 遍历点的几何面，并设置颜色
    for (var i = 0; i < point.geometry.faces.length; i++) {
      point.geometry.faces[i].color = color;
    }

    // 如果点的矩阵自动更新，则手动更新矩阵
    if(point.matrixAutoUpdate){
      point.updateMatrix();
    }

    // 合并点的几何和矩阵到子几何中
    subgeo.merge(point.geometry, point.matrix);
  }

  // 鼠标按下事件处理函数
  function onMouseDown(event) {
    event.preventDefault();

    // 添加鼠标移动、鼠标抬起、鼠标移出事件监听
    container.addEventListener('mousemove', onMouseMove, false);
    container.addEventListener('mouseup', onMouseUp, false);
    container.addEventListener('mouseout', onMouseOut, false);

    // 记录鼠标按下时的位置和目标位置
    mouseOnDown.x = - event.clientX;
    mouseOnDown.y = event.clientY;
    targetOnDown.x = target.x;
    targetOnDown.y = target.y;

    // 设置鼠标样式为移动
    container.style.cursor = 'move';
  }

  // 鼠标移动事件处理函数
  function onMouseMove(event) {
    // 更新鼠标位置
    mouse.x = - event.clientX;
    mouse.y = event.clientY;

    // 计算缩放系数
    var zoomDamp = distance/1000;

    // 根据鼠标移动计算目标位置
    target.x = targetOnDown.x + (mouse.x - mouseOnDown.x) * 0.005 * zoomDamp;
    target.y = targetOnDown.y + (mouse.y - mouseOnDown.y) * 0.005 * zoomDamp;

    // 限制目标位置的范围
    target.y = target.y > PI_HALF ? PI_HALF : target.y;
    target.y = target.y < - PI_HALF ? - PI_HALF : target.y;
  }

  // 鼠标抬起事件处理函数
  function onMouseUp(event) {
    // 移除鼠标移动、鼠标抬起、鼠标移出事件监听
    container.removeEventListener('mousemove', onMouseMove, false);
    container.removeEventListener('mouseup', onMouseUp, false);
    container.removeEventListener('mouseout', onMouseOut, false);
    // 恢复鼠标样式
    container.style.cursor = 'auto';
  }

  // 鼠标移出事件处理函数
  function onMouseOut(event) {
    // 移除鼠标移动、鼠标抬起、鼠标移出事件监听
    container.removeEventListener('mousemove', onMouseMove, false);
    container.removeEventListener('mouseup', onMouseUp, false);
    container.removeEventListener('mouseout', onMouseOut, false);
  }

  // 鼠标滚轮事件处理函数
  function onMouseWheel(event) {
    // 如果鼠标样式不是移动，则返回false
    if (container.style.cursor != "move") return false;
    // 阻止默认事件
    event.preventDefault();
    // 如果鼠标在渲染器上
    if (overRenderer) {
      // 如果事件有 deltaY 属性
      if (event.deltaY) {
        // 调用 zoom 函数，根据 deltaY 属性的值进行缩放
        zoom(-event.deltaY * (event.deltaMode == 0 ? 1 : 50));
      } else {
        // 调用 zoom 函数，根据 wheelDeltaY 属性的值进行缩放
        zoom(event.wheelDeltaY * 0.3);
      }
    }
    // 返回false
    return false;
  }

  // 键盘按下事件处理函数
  function onDocumentKeyDown(event) {
    // 根据按键码进行不同的操作
    switch (event.keyCode) {
      case 38:
        // 调用 zoom 函数，放大100
        zoom(100);
        // 阻止默认事件
        event.preventDefault();
        break;
      case 40:
        // 调用 zoom 函数，缩小100
        zoom(-100);
        // 阻止默认事件
        event.preventDefault();
        break;
    }
  }

  // 窗口大小改变事件处理函数
  function onWindowResize( event ) {
    // 更新相机的宽高比
    camera.aspect = container.offsetWidth / container.offsetHeight;
    camera.updateProjectionMatrix();
    // 更新渲染器的大小
    renderer.setSize( container.offsetWidth, container.offsetHeight );
  }

  // 缩放函数
  function zoom(delta) {
    // 更新距离目标值
    distanceTarget -= delta;
    // 限制距离目标值的范围
    distanceTarget = distanceTarget > 855 ? 855 : distanceTarget;
    distanceTarget = distanceTarget < 350 ? 350 : distanceTarget;
  }

  // 动画函数
  function animate() {
    // 如果不在运行状态，则返回
    if (!running) return
    // 请求动画帧
    requestAnimationFrame(animate);
    // 渲染
    render();
  }

  // 渲染函数
  function render() {
    // 调用 zoom 函数，根据 curZoomSpeed 进行缩放
    zoom(curZoomSpeed);

    // 更新旋转角度
    rotation.x += (target.x - rotation.x) * 0.1;
    rotation.y += (target.y - rotation.y) * 0.1;
    // 更新距离
    distance += (distanceTarget - distance) * 0.3;

    // 更新相机位置
    camera.position.x = distance * Math.sin(rotation.x) * Math.cos(rotation.y);
    camera.position.y = distance * Math.sin(rotation.y);
    camera.position.z = distance * Math.cos(rotation.x) * Math.cos(rotation.y);

    // 相机朝向网格对象
    camera.lookAt(mesh.position);

    // 渲染场景
    renderer.render(scene, camera);
  }

  // 卸载函数
  function unload() {
    // 将运行状态设置为false
    running = false
    // 移除鼠标按下事件监听
    container.removeEventListener('mousedown', onMouseDown, false);
    // 如果支持wheel事件，则移除鼠标滚轮事件监听
    if ('onwheel' in document) {
      container.removeEventListener('wheel', onMouseWheel, false);
    } else {
      // 否则移除mousewheel事件监听
      container.removeEventListener('mousewheel', onMouseWheel, false);
    }
    // 移除键盘按下事件监听
    document.removeEventListener('keydown', onDocumentKeyDown, false);
    # 移除窗口大小改变事件的监听器
    window.removeEventListener('resize', onWindowResize, false);

  }

  # 初始化函数
  init();
  # 设置动画函数
  this.animate = animate;
  # 卸载函数
  this.unload = unload;

  # 定义时间属性的 getter 方法
  this.__defineGetter__('time', function() {
    return this._time || 0;
  });

  # 定义时间属性的 setter 方法
  this.__defineSetter__('time', function(t) {
    # 获取有效的形态目标
    var validMorphs = [];
    var morphDict = this.points.morphTargetDictionary;
    for(var k in morphDict) {
      if(k.indexOf('morphPadding') < 0) {
        validMorphs.push(morphDict[k]);
      }
    }
    validMorphs.sort();
    var l = validMorphs.length-1;
    var scaledt = t*l+1;
    var index = Math.floor(scaledt);
    for (i=0;i<validMorphs.length;i++) {
      this.points.morphTargetInfluences[validMorphs[i]] = 0;
    }
    var lastIndex = index - 1;
    var leftover = scaledt - index;
    if (lastIndex >= 0) {
      this.points.morphTargetInfluences[lastIndex] = 1 - leftover;
    }
    this.points.morphTargetInfluences[index] = leftover;
    this._time = t;
  });

  # 添加数据的函数
  this.addData = addData;
  # 创建点的函数
  this.createPoints = createPoints;
  # 渲染器
  this.renderer = renderer;
  # 场景
  this.scene = scene;

  # 返回当前对象
  return this;
# 定义了一个空的对象
};

# 引入了three.min.js文件，并声明了严格模式
/* ---- plugins/Sidebar/media_globe/three.min.js ---- */

# 定义了THREE对象，并设置了REVISION属性为"69"
'use strict';var THREE={REVISION:"69"};

# 如果是在模块环境下，将THREE对象导出
"object"===typeof module&&(module.exports=THREE);

# 如果Math对象没有sign方法，则定义sign方法
void 0===Math.sign&&(Math.sign=function(a){return 0>a?-1:0<a?1:0});

# 定义了THREE.MOUSE对象，包含了左键、中键和右键的值
THREE.MOUSE={LEFT:0,MIDDLE:1,RIGHT:2};

# 定义了一系列常量，包括面剔除、面方向、阴影映射、着色方式、颜色方式、混合方式、混合因子、混合操作、纹理映射、纹理重复方式等
THREE.CullFaceNone=0;
THREE.CullFaceBack=1;
THREE.CullFaceFront=2;
THREE.CullFaceFrontBack=3;
THREE.FrontFaceDirectionCW=0;
THREE.FrontFaceDirectionCCW=1;
THREE.BasicShadowMap=0;
THREE.PCFShadowMap=1;
THREE.PCFSoftShadowMap=2;
THREE.FrontSide=0;
THREE.BackSide=1;
THREE.DoubleSide=2;
THREE.NoShading=0;
THREE.FlatShading=1;
THREE.SmoothShading=2;
THREE.NoColors=0;
THREE.FaceColors=1;
THREE.VertexColors=2;
THREE.NoBlending=0;
THREE.NormalBlending=1;
THREE.AdditiveBlending=2;
THREE.SubtractiveBlending=3;
THREE.MultiplyBlending=4;
THREE.CustomBlending=5;
THREE.AddEquation=100;
THREE.SubtractEquation=101;
THREE.ReverseSubtractEquation=102;
THREE.MinEquation=103;
THREE.MaxEquation=104;
THREE.ZeroFactor=200;
THREE.OneFactor=201;
THREE.SrcColorFactor=202;
THREE.OneMinusSrcColorFactor=203;
THREE.SrcAlphaFactor=204;
THREE.OneMinusSrcAlphaFactor=205;
THREE.DstAlphaFactor=206;
THREE.OneMinusDstAlphaFactor=207;
THREE.DstColorFactor=208;
THREE.OneMinusDstColorFactor=209;
THREE.SrcAlphaSaturateFactor=210;
THREE.MultiplyOperation=0;
THREE.MixOperation=1;
THREE.AddOperation=2;
THREE.UVMapping=function(){};
THREE.CubeReflectionMapping=function(){};
THREE.CubeRefractionMapping=function(){};
THREE.SphericalReflectionMapping=function(){};
THREE.SphericalRefractionMapping=function(){};
THREE.RepeatWrapping=1E3;
# 定义纹理环绕方式常量
THREE.ClampToEdgeWrapping=1001;
THREE.MirroredRepeatWrapping=1002;
# 定义纹理过滤方式常量
THREE.NearestFilter=1003;
THREE.NearestMipMapNearestFilter=1004;
THREE.NearestMipMapLinearFilter=1005;
THREE.LinearFilter=1006;
THREE.LinearMipMapNearestFilter=1007;
THREE.LinearMipMapLinearFilter=1008;
# 定义像素数据类型常量
THREE.UnsignedByteType=1009;
THREE.ByteType=1010;
THREE.ShortType=1011;
THREE.UnsignedShortType=1012;
THREE.IntType=1013;
THREE.UnsignedIntType=1014;
THREE.FloatType=1015;
THREE.UnsignedShort4444Type=1016;
THREE.UnsignedShort5551Type=1017;
THREE.UnsignedShort565Type=1018;
# 定义纹理格式常量
THREE.AlphaFormat=1019;
THREE.RGBFormat=1020;
THREE.RGBAFormat=1021;
THREE.LuminanceFormat=1022;
THREE.LuminanceAlphaFormat=1023;
# 定义压缩纹理格式常量
THREE.RGB_S3TC_DXT1_Format=2001;
THREE.RGBA_S3TC_DXT1_Format=2002;
THREE.RGBA_S3TC_DXT3_Format=2003;
THREE.RGBA_S3TC_DXT5_Format=2004;
THREE.RGB_PVRTC_4BPPV1_Format=2100;
THREE.RGB_PVRTC_2BPPV1_Format=2101;
THREE.RGBA_PVRTC_4BPPV1_Format=2102;
THREE.RGBA_PVRTC_2BPPV1_Format=2103;
# 定义颜色对象构造函数
THREE.Color=function(a){return 3===arguments.length?this.setRGB(arguments[0],arguments[1],arguments[2]):this.set(a)};
# 定义颜色对象原型
THREE.Color.prototype={constructor:THREE.Color,r:1,g:1,b:1,
    # 设置颜色值
    set:function(a){
        # 如果参数是颜色对象，则复制颜色值
        a instanceof THREE.Color?this.copy(a):
        # 如果参数是数字，则设置十六进制颜色值
        "number"===typeof a?this.setHex(a):
        # 如果参数是字符串，则设置样式颜色值
        "string"===typeof a&&this.setStyle(a);
        return this
    },
    # 设置十六进制颜色值
    setHex:function(a){
        a=Math.floor(a);
        this.r=(a>>16&255)/255;
        this.g=(a>>8&255)/255;
        this.b=(a&255)/255;
        return this
    },
    # 设置 RGB 颜色值
    setRGB:function(a,b,c){
        this.r=a;
        this.g=b;
        this.b=c;
        return this
    },
    # 设置 HSL 颜色值
    setHSL:function(a,b,c){
        if(0===b)
            this.r=this.g=this.b=c;
        else{
            var d=function(a,b,c){
                0>c&&(c+=1);
                1<c&&(c-=1);
                return c<1/6?a+6*(b-a)*
// 设置颜色的 HSL 值
setHSL: function(h, s, l) {
    // 如果饱和度为 0，则将 RGB 值都设置为亮度值
    if (s === 0) {
        this.r = this.g = this.b = l;
    } else {
        // 计算辅助变量
        var p = l <= 0.5 ? l * (1 + s) : l + s - l * s;
        var q = 2 * l - p;
        // 根据色相计算 RGB 值
        this.r = this.hue2rgb(q, p, h + 1/3);
        this.g = this.hue2rgb(q, p, h);
        this.b = this.hue2rgb(q, p, h - 1/3);
    }
    return this;
},
// 设置颜色的样式
setStyle: function(a) {
    // 如果是 rgb 格式的颜色值
    if (/^rgb\((\d+), ?(\d+), ?(\d+)\)$/i.test(a)) {
        // 解析 RGB 值并转换为小数
        a = /^rgb\((\d+), ?(\d+), ?(\d+)\)$/i.exec(a);
        this.r = Math.min(255, parseInt(a[1], 10)) / 255;
        this.g = Math.min(255, parseInt(a[2], 10)) / 255;
        this.b = Math.min(255, parseInt(a[3], 10)) / 255;
        return this;
    }
    // 如果是 rgb 百分比格式的颜色值
    if (/^rgb\((\d+)\%, ?(\d+)\%, ?(\d+)\%\)$/i.test(a)) {
        // 解析 RGB 百分比值并转换为小数
        a = /^rgb\((\d+)\%, ?(\d+)\%, ?(\d+)\%\)$/i.exec(a);
        this.r = Math.min(100, parseInt(a[1], 10)) / 100;
        this.g = Math.min(100, parseInt(a[2], 10)) / 100;
        this.b = Math.min(100, parseInt(a[3], 10)) / 100;
        return this;
    }
    // 如果是十六进制格式的颜色值
    if (/^\#([0-9a-f]{6})$/i.test(a)) {
        // 解析十六进制值并设置颜色
        a = /^\#([0-9a-f]{6})$/i.exec(a);
        this.setHex(parseInt(a[1], 16));
        return this;
    }
    // 如果是缩写的十六进制格式的颜色值
    if (/^\#([0-9a-f])([0-9a-f])([0-9a-f])$/i.test(a)) {
        // 解析缩写的十六进制值并设置颜色
        a = /^\#([0-9a-f])([0-9a-f])([0-9a-f])$/i.exec(a);
        this.setHex(parseInt(a[1] + a[1] + a[2] + a[2] + a[3] + a[3], 16));
        return this;
    }
    // 如果是颜色关键字
    if (/^(\w+)$/i.test(a)) {
        // 设置颜色为颜色关键字对应的值
        this.setHex(THREE.ColorKeywords[a]);
        return this;
    }
},
// 复制颜色值
copy: function(a) {
    this.r = a.r;
    this.g = a.g;
    this.b = a.b;
    return this;
},
// 将颜色值从 gamma 校正空间复制到线性空间
copyGammaToLinear: function(a) {
    this.r = a.r * a.r;
    this.g = a.g * a.g;
    this.b = a.b * a.b;
    return this;
},
// 将颜色值从线性空间复制到 gamma 校正空间
copyLinearToGamma: function(a) {
    this.r = Math.sqrt(a.r);
    this.g = Math.sqrt(a.g);
    this.b = Math.sqrt(a.b);
    return this;
},
// 将颜色值从 gamma 校正空间转换到线性空间
convertGammaToLinear: function() {
    var a = this.r, b = this.g, c = this.b;
    this.r = a * a;
    this.g = b * b;
    this.b = c * c;
    return this;
},
// 将颜色值从线性空间转换到 gamma 校正空间
convertLinearToGamma: function() {
    this.r = Math.sqrt(this.r);
    this.g = Math.sqrt(this.g);
    this.b = Math.sqrt(this.b);
    return this;
},
// 获取颜色值的十六进制表示
getHex: function() {
    return (255 * this.r << 16) ^ (255 * this.g << 8) ^ (255 * this.b << 0);
}
# 定义一个名为 THREE 的模块，包含 Color 和 ColorKeywords 两个对象
8^255*this.b<<0},
# 未知操作，需要进一步了解上下文
getHexString:function(){
# 获取颜色的十六进制表示，返回一个六位的字符串
return("000000"+this.getHex().toString(16)).slice(-6)},
# 获取颜色的 HSL 表示，返回一个包含 h、s、l 三个属性的对象
getHSL:function(a){
# 如果传入参数 a 为空，则初始化为包含 h、s、l 三个属性的对象
a=a||{h:0,s:0,l:0};
# 获取颜色的 RGB 分量
var b=this.r,c=this.g,d=this.b,
# 计算颜色的最大值和最小值
e=Math.max(b,c,d),f=Math.min(b,c,d),g,h=(f+e)/2;
# 根据最大值和最小值计算 h、s、l
if(f===e)f=g=0;else{var k=e-f,f=.5>=h?k/(e+f):k/(2-e-f);
switch(e){case b:g=(c-d)/k+(c<d?6:0);break;case c:g=(d-b)/k+2;break;case d:g=(b-c)/k+4}g/=6}
# 将计算结果保存到传入的对象中并返回
a.h=g;a.s=f;a.l=h;return a},
# 获取颜色的 CSS 样式表示，返回一个字符串
getStyle:function(){
return"rgb("+(255*this.r|0)+","+(255*this.g|0)+","+(255*this.b|0)+")"},
# 改变颜色的 HSL 表示，传入参数为偏移量，返回一个新的颜色对象
offsetHSL:function(a,b,c){var d=this.getHSL();d.h+=a;d.s+=b;d.l+=c;this.setHSL(d.h,d.s,d.l);return this},
# 将当前颜色与传入的颜色相加，返回一个新的颜色对象
add:function(a){this.r+=a.r;this.g+=a.g;this.b+=a.b;return this},
# 将当前颜色与传入的两个颜色相加，返回一个新的颜色对象
addColors:function(a,b){this.r=a.r+b.r;this.g=a.g+b.g;this.b=a.b+b.b;return this},
# 将当前颜色的 RGB 分量与传入的标量相加，返回一个新的颜色对象
addScalar:function(a){this.r+=a;this.g+=a;this.b+=a;return this},
# 将当前颜色与传入的颜色相乘，返回一个新的颜色对象
multiply:function(a){this.r*=a.r;this.g*=a.g;this.b*=a.b;return this},
# 将当前颜色的 RGB 分量与传入的标量相乘，返回一个新的颜色对象
multiplyScalar:function(a){this.r*=a;this.g*=a;this.b*=a;return this},
# 在两个颜色之间进行线性插值，返回一个新的颜色对象
lerp:function(a,b){this.r+=(a.r-this.r)*b;this.g+=(a.g-this.g)*b;this.b+=(a.b-this.b)*b;return this},
# 判断当前颜色是否与传入的颜色相等，返回一个布尔值
equals:function(a){return a.r===this.r&&a.g===this.g&&a.b===this.b},
# 从数组中获取 RGB 分量，返回当前颜色对象
fromArray:function(a){this.r=a[0];this.g=a[1];this.b=a[2];return this},
# 将当前颜色的 RGB 分量保存到数组中，返回一个包含 RGB 分量的数组
toArray:function(){return[this.r,this.g,this.b]},
# 克隆当前颜色对象，返回一个新的颜色对象
clone:function(){return(new THREE.Color).setRGB(this.r,this.g,this.b)}};
# 定义颜色关键字的 RGB 值
THREE.ColorKeywords={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,
# 定义一个包含颜色名称和对应RGB值的字符串
darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,
grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,
lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,
# 定义颜色和对应的 RGB 值的映射关系
palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,
tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074};
# 定义 THREE.Quaternion 类
THREE.Quaternion=function(a,b,c,d){
    # 初始化四元数的 x、y、z、w 分量
    this._x=a||0;
    this._y=b||0;
    this._z=c||0;
    this._w=void 0!==d?d:1;
};
# 定义 THREE.Quaternion 的原型方法
THREE.Quaternion.prototype={
    constructor:THREE.Quaternion,
    _x:0,
    _y:0,
    _z:0,
    _w:0,
    # 获取 x 分量的方法
    get x(){return this._x},
    # 设置 x 分量的方法
    set x(a){this._x=a;this.onChangeCallback()},
    # 获取 y 分量的方法
    get y(){return this._y},
    # 设置 y 分量的方法
    set y(a){this._y=a;this.onChangeCallback()},
    # 获取 z 分量的方法
    get z(){return this._z},
    # 设置 z 分量的方法
    set z(a){this._z=a;this.onChangeCallback()},
    # 获取 w 分量的方法
    get w(){return this._w},
    # 设置 w 分量的方法
    set w(a){this._w=a;this.onChangeCallback()},
    # 设置四元数的值
    set:function(a,b,c,d){
        this._x=a;
        this._y=b;
        this._z=c;
        this._w=d;
        this.onChangeCallback();
        return this
    },
    # 复制另一个四元数的值
    copy:function(a){
        this._x=a.x;
        this._y=a.y;
        this._z=a.z;
        this._w=a.w;
        this.onChangeCallback();
        return this
    },
    # 根据欧拉角设置四元数的值
    setFromEuler:function(a,b){
        if(!1===a instanceof THREE.Euler)throw Error("THREE.Quaternion: .setFromEuler() now expects a Euler rotation rather than a Vector3 and order.");
        var c=Math.cos(a._x/2),
            d=Math.cos(a._y/2),
            e=Math.cos(a._z/2),
            f=Math.sin(a._x/2),
            g=Math.sin(a._y/2),
            h=Math.sin(a._z/2);
        if("XYZ"===a.order){
            this._x=f*d*e+c*g*h;
            this._y=c*g*e-f*d*h;
            this._z=c*d*h+f*g*e;
            this._w=c*d*e-f*g*h
        }else if("YXZ"===a.order){
            this._x=f*d*e+c*g*h;
            this._y=c*g*e-f*d*h;
            this._z=c*d*h+f*g*e;
            this._w=c*d*e-f*g*h
        }
    }
};
// 根据给定的顺序设置四元数的值
c*d*h-f*g*e,this._w=c*d*e+f*g*h):"ZXY"===a.order?(this._x=f*d*e-c*g*h,this._y=c*g*e+f*d*h,this._z=c*d*h+f*g*e,this._w=c*d*e-f*g*h):"ZYX"===a.order?(this._x=f*d*e-c*g*h,this._y=c*g*e+f*d*h,this._z=c*d*h-f*g*e,this._w=c*d*e+f*g*h):"YZX"===a.order?(this._x=f*d*e+c*g*h,this._y=c*g*e+f*d*h,this._z=c*d*h-f*g*e,this._w=c*d*e-f*g*h):"XZY"===a.order&&(this._x=f*d*e-c*g*h,this._y=c*g*e-f*d*h,this._z=c*d*h+f*g*e,this._w=c*d*e+f*g*h);
// 如果需要触发回调函数，则调用回调函数
if(!1!==b)this.onChangeCallback();
// 返回设置后的四元数
return this
// 根据给定的轴和角度设置四元数的值
},setFromAxisAngle:function(a,b){
// 计算角度的一半和正弦值
var c=b/2,d=Math.sin(c);
// 根据给定的轴和角度设置四元数的值
this._x=a.x*d;this._y=a.y*d;this._z=a.z*d;this._w=Math.cos(c);
// 如果需要触发回调函数，则调用回调函数
this.onChangeCallback();
// 返回设置后的四元数
return this
// 根据给定的旋转矩阵设置四元数的值
},setFromRotationMatrix:function(a){
// 获取旋转矩阵的元素
var b=a.elements,c=b[0];a=b[4];var d=b[8],e=b[1],f=b[5],g=b[9],h=b[2],k=b[6],b=b[10],n=c+f+b;
// 根据给定的旋转矩阵设置四元数的值
0<n?(c=.5/Math.sqrt(n+1),this._w=.25/c,this._x=(k-g)*c,this._y=(d-h)*c,this._z=(e-a)*c):c>f&&c>b?(c=2*Math.sqrt(1+c-f-b),this._w=(k-g)/c,this._x=.25*c,this._y=(a+e)/c,this._z=(d+h)/c):f>b?(c=2*Math.sqrt(1+f-c-b),this._w=(d-h)/c,this._x=(a+e)/c,this._y=.25*c,this._z=(g+k)/c):(c=2*Math.sqrt(1+b-c-f),this._w=(e-a)/c,this._x=(d+h)/c,this._y=(g+k)/c,this._z=.25*c);
// 如果需要触发回调函数，则调用回调函数
this.onChangeCallback();
// 返回设置后的四元数
return this
// 根据给定的单位向量设置四元数的值
},setFromUnitVectors:function(){
// 定义局部变量a和b
var a,b;
// 返回一个函数，根据给定的单位向量设置四元数的值
return function(c,d){
// 如果变量a未定义，则初始化为一个新的三维向量
void 0===a&&(a=new THREE.Vector3);
// 计算两个向量的点积并加1
b=c.dot(d)+1;
// 如果点积接近于0，则设置b为0，并计算叉积
1E-6>b?(b=0,Math.abs(c.x)>Math.abs(c.z)?a.set(-c.y,c.x,0):a.set(0,-c.z,c.y)):a.crossVectors(c,d);
// 设置四元数的值
this._x=a.x;this._y=a.y;this._z=a.z;this._w=b;
// 规范化四元数
this.normalize();
// 返回设置后的四元数
return this}}(),
// 计算四元数的逆
inverse:function(){
// 求四元数的共轭并规范化
this.conjugate().normalize();
// 返回逆四元数
return this
// 求四元数的共轭
},conjugate:function(){
// 对四元数的各个分量进行共轭运算
this._x*=
# 将四元数的 x 分量取反
-1;
# 将四元数的 y 分量取反
this._y*=-1;
# 将四元数的 z 分量取反
this._z*=-1;
# 调用回调函数
this.onChangeCallback();
# 返回修改后的四元数
return this
},
# 计算当前四元数与参数四元数的点积
dot:function(a){
    return this._x*a._x+this._y*a._y+this._z*a._z+this._w*a._w
},
# 计算四元数的长度的平方
lengthSq:function(){
    return this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w
},
# 计算四元数的长度
length:function(){
    return Math.sqrt(this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w)
},
# 将四元数归一化
normalize:function(){
    var a=this.length();
    # 如果长度为0，则将四元数的 x、y、z 分量都设为0，w 分量设为1
    0===a?(this._z=this._y=this._x=0,this._w=1):
    # 否则，计算归一化系数，然后将四元数的各分量乘以该系数
    (a=1/a,this._x*=a,this._y*=a,this._z*=a,this._w*=a);
    # 调用回调函数
    this.onChangeCallback();
    # 返回归一化后的四元数
    return this
},
# 乘以另一个四元数
multiply:function(a,b){
    # 如果传入了第二个参数，则给出警告信息
    return void 0!==b?(console.warn("THREE.Quaternion: .multiply() now only accepts one argument. Use .multiplyQuaternions( a, b ) instead."),this.multiplyQuaternions(a,b)):
    # 否则，调用 multiplyQuaternions 方法
    this.multiplyQuaternions(this,a)
},
# 乘以另一个四元数
multiplyQuaternions:function(a,b){
    # 分别获取两个四元数的各分量
    var c=a._x,d=a._y,e=a._z,f=a._w,g=b._x,h=b._y,k=b._z,n=b._w;
    # 计算四元数乘积的各分量
    this._x=c*n+f*g+d*k-e*h;
    this._y=d*n+f*h+e*g-c*k;
    this._z=e*n+f*k+c*h-d*g;
    this._w=f*n-c*g-d*h-e*k;
    # 调用回调函数
    this.onChangeCallback();
    # 返回乘积结果
    return this
},
# 与三维向量的乘积，已被移除
multiplyVector3:function(a){
    console.warn("THREE.Quaternion: .multiplyVector3() has been removed. Use is now vector.applyQuaternion( quaternion ) instead.");
    return a.applyQuaternion(this)
},
# 通过球形线性插值得到新的四元数
slerp:function(a,b){
    # 如果插值系数为0，返回当前四元数；如果为1，返回参数四元数
    if(0===b)return this;
    if(1===b)return this.copy(a);
    # 计算当前四元数与参数四元数的点积
    var c=this._x,d=this._y,e=this._z,f=this._w,g=f*a._w+c*a._x+d*a._y+e*a._z;
    # 如果点积小于0，将参数四元数取反
    0>g?(this._w=-a._w,this._x=-a._x,this._y=-a._y,this._z=-a._z,g=-g):this.copy(a);
    # 如果点积大于等于1，返回参数四元数
    if(1<=g)return this._w=f,this._x=c,this._y=d,this._z=e,this;
    # 计算插值角度和插值系数
    var h=Math.acos(g),k=Math.sqrt(1-g*g);
    # 如果插值系数接近0，直接返回插值结果
    if(.001>Math.abs(k))return this._w=.5*(f+this._w),this._x=.5*(c+this._x),this._y=.5*(d+this._y),this._z=.5*(e+this._z),this;
    # 计算插值系数对应的四元数
    g=Math.sin((1-b)*h)/k;
    h=
# 计算正弦值
Math.sin(b*h)/k;
# 更新四元数的值
this._w=f*g+this._w*h;
this._x=c*g+this._x*h;
this._y=d*g+this._y*h;
this._z=e*g+this._z*h;
# 调用回调函数
this.onChangeCallback();
# 返回结果
return this
# 判断两个四元数是否相等
equals:function(a){
    return a._x===this._x&&a._y===this._y&&a._z===this._z&&a._w===this._w
},
# 从数组中设置四元数的值
fromArray:function(a,b){
    void 0===b&&(b=0);
    this._x=a[b];
    this._y=a[b+1];
    this._z=a[b+2];
    this._w=a[b+3];
    this.onChangeCallback();
    return this
},
# 将四元数的值存入数组
toArray:function(a,b){
    void 0===a&&(a=[]);
    void 0===b&&(b=0);
    a[b]=this._x;
    a[b+1]=this._y;
    a[b+2]=this._z;
    a[b+3]=this._w;
    return a
},
# 设置回调函数
onChange:function(a){
    this.onChangeCallback=a;
    return this
},
# 空的回调函数
onChangeCallback:function(){},
# 克隆四元数
clone:function(){
    return new THREE.Quaternion(this._x,this._y,this._z,this._w)
};
# 计算两个四元数之间的球面线性插值
THREE.Quaternion.slerp=function(a,b,c,d){
    return c.copy(a).slerp(b,d)
};
# 二维向量类
THREE.Vector2=function(a,b){
    this.x=a||0;
    this.y=b||0
};
# 二维向量类的原型方法
THREE.Vector2.prototype={
    constructor:THREE.Vector2,
    # 设置向量的值
    set:function(a,b){
        this.x=a;
        this.y=b;
        return this
    },
    # 设置向量的 x 值
    setX:function(a){
        this.x=a;
        return this
    },
    # 设置向量的 y 值
    setY:function(a){
        this.y=a;
        return this
    },
    # 设置向量的指定分量的值
    setComponent:function(a,b){
        switch(a){
            case 0:this.x=b;
            break;
            case 1:this.y=b;
            break;
            default:throw Error("index is out of range: "+a);
        }
    },
    # 获取向量的指定分量的值
    getComponent:function(a){
        switch(a){
            case 0:return this.x;
            case 1:return this.y;
            default:throw Error("index is out of range: "+a);
        }
    },
    # 复制另一个向量的值
    copy:function(a){
        this.x=a.x;
        this.y=a.y;
        return this
    },
    # 向量加法
    add:function(a,b){
        if(void 0!==b)return console.warn("THREE.Vector2: .add() now only accepts one argument. Use .addVectors( a, b ) instead."),this.addVectors(a,b);
        this.x+=a.x;
        this.y+=a.y;
        return this
    },
    # 向量加法
    addVectors:function(a,b){
        this.x=a.x+b.x;
        this.y=a.y+b.y;
        return this
    },
    # 向量标量加法
    addScalar:function(a){
        this.x+=a;
        this.y+=a;
        return this
    },
    # 向量减法
    sub:function(a,b){
        if(void 0!==b)return console.warn("THREE.Vector2: .sub() now only accepts one argument. Use .subVectors( a, b ) instead."),this.subVectors(a,b);
        this.x-=a.x;
        this.y-=a.y;
        return this
    },
# 定义一个名为 subVectors 的方法，用于计算两个向量的差
subVectors:function(a,b){this.x=a.x-b.x;this.y=a.y-b.y;return this},
# 定义一个名为 multiply 的方法，用于将当前向量与另一个向量相乘
multiply:function(a){this.x*=a.x;this.y*=a.y;return this},
# 定义一个名为 multiplyScalar 的方法，用于将当前向量与标量相乘
multiplyScalar:function(a){this.x*=a;this.y*=a;return this},
# 定义一个名为 divide 的方法，用于将当前向量与另一个向量相除
divide:function(a){this.x/=a.x;this.y/=a.y;return this},
# 定义一个名为 divideScalar 的方法，用于将当前向量与标量相除
divideScalar:function(a){0!==a?(a=1/a,this.x*=a,this.y*=a):this.y=this.x=0;return this},
# 定义一个名为 min 的方法，用于将当前向量的每个分量与另一个向量的对应分量比较，取较小值
min:function(a){this.x>a.x&&(this.x=a.x);this.y>a.y&&(this.y=a.y);return this},
# 定义一个名为 max 的方法，用于将当前向量的每个分量与另一个向量的对应分量比较，取较大值
max:function(a){this.x<a.x&&(this.x=a.x);this.y<a.y&&(this.y=a.y);return this},
# 定义一个名为 clamp 的方法，用于将当前向量的每个分量限制在指定范围内
clamp:function(a,b){this.x<a.x?this.x=a.x:this.x>b.x&&(this.x=b.x);this.y<a.y?this.y=a.y:this.y>b.y&&(this.y=b.y);return this},
# 定义一个名为 clampScalar 的方法，用于将当前向量的每个分量限制在指定标量范围内
clampScalar:function(){var a,b;return function(c,d){void 0===a&&(a=new THREE.Vector2,b=new THREE.Vector2);a.set(c,c);b.set(d,d);return this.clamp(a,b)}}(),
# 定义一个名为 floor 的方法，用于将当前向量的每个分量向下取整
floor:function(){this.x=Math.floor(this.x);this.y=Math.floor(this.y);return this},
# 定义一个名为 ceil 的方法，用于将当前向量的每个分量向上取整
ceil:function(){this.x=Math.ceil(this.x);this.y=Math.ceil(this.y);return this},
# 定义一个名为 round 的方法，用于将当前向量的每个分量四舍五入
round:function(){this.x=Math.round(this.x);this.y=Math.round(this.y);return this},
# 定义一个名为 roundToZero 的方法，用于将当前向量的每个分量向零取整
roundToZero:function(){this.x=0>this.x?Math.ceil(this.x):Math.floor(this.x);this.y=0>this.y?Math.ceil(this.y):Math.floor(this.y);return this},
# 定义一个名为 negate 的方法，用于将当前向量取反
negate:function(){this.x=-this.x;this.y=-this.y;return this},
# 定义一个名为 dot 的方法，用于计算当前向量与另一个向量的点积
dot:function(a){return this.x*a.x+this.y*a.y},
# 定义一个名为 lengthSq 的方法，用于计算当前向量的长度的平方
lengthSq:function(){return this.x*this.x+this.y*this.y},
# 定义一个名为 length 的方法，用于计算当前向量的长度
length:function(){return Math.sqrt(this.x*this.x+this.y*this.y)},
# 定义一个名为 normalize 的方法，用于将当前向量归一化
normalize:function(){return this.divideScalar(this.length())},
# 定义一个名为 distanceTo 的方法，用于计算当前向量与另一个向量之间的距离
distanceTo:function(a){return Math.sqrt(this.distanceToSquared(a))},
# 定义一个名为 distanceToSquared 的方法，用于计算当前向量与另一个向量之间的距离的平方
distanceToSquared:function(a){var b=
# 定义一个三维向量类，包含 x、y、z 三个属性
THREE.Vector3=function(a,b,c){
    # 初始化 x、y、z 属性，如果没有传入参数则默认为 0
    this.x=a||0;
    this.y=b||0;
    this.z=c||0;
};
# 为 Vector3 类添加原型方法
THREE.Vector3.prototype={
    # 设置向量的 x、y、z 属性
    constructor:THREE.Vector3,
    set:function(a,b,c){
        this.x=a;
        this.y=b;
        this.z=c;
        return this
    },
    # 设置向量的 x 属性
    setX:function(a){
        this.x=a;
        return this
    },
    # 设置向量的 y 属性
    setY:function(a){
        this.y=a;
        return this
    },
    # 设置向量的 z 属性
    setZ:function(a){
        this.z=a;
        return this
    },
    # 设置向量的指定分量
    setComponent:function(a,b){
        switch(a){
            case 0:
                this.x=b;
                break;
            case 1:
                this.y=b;
                break;
            case 2:
                this.z=b;
                break;
            default:
                throw Error("index is out of range: "+a);
        }
    },
    # 获取向量的指定分量
    getComponent:function(a){
        switch(a){
            case 0:
                return this.x;
            case 1:
                return this.y;
            case 2:
                return this.z;
            default:
                throw Error("index is out of range: "+a);
        }
    },
    # 复制另一个向量的值
    copy:function(a){
        this.x=a.x;
        this.y=a.y;
        this.z=a.z;
        return this
    },
    # 向量加法
    add:function(a,b){
        if(void 0!==b)
            return console.warn("THREE.Vector3: .add() now only accepts one argument. Use .addVectors( a, b ) instead."),
            this.addVectors(a,b);
        this.x+=a.x;
        this.y+=a.y;
        this.z+=a.z;
        return this
    },
    # 向量标量加法
    addScalar:function(a){
        this.x+=a;
        this.y+=a;
        this.z+=a;
        return this
    },
    # 向量加法
    addVectors:function(a,b){
        this.x=a.x+b.x;
        this.y=a.y+b.y;
        this.z=a.z+b.z;
        return this
    },
    # 向量减法
    sub:function(a,b){
        if(void 0!==b)
            return console.warn("THREE.Vector3: .sub() now only accepts one argument. Use .subVectors( a, b ) instead."),
            this.subVectors(a,b);

... (以下省略)
# 减去向量 a 中的坐标值，更新当前向量的坐标值，并返回更新后的向量
this.subVectors(a,b);
this.x-=a.x;
this.y-=a.y;
this.z-=a.z;
return this
# 根据两个向量的坐标值计算差值，更新当前向量的坐标值，并返回更新后的向量
subVectors:function(a,b){
    this.x=a.x-b.x;
    this.y=a.y-b.y;
    this.z=a.z-b.z;
    return this
}
# 乘以一个向量或标量，更新当前向量的坐标值，并返回更新后的向量
multiply:function(a,b){
    # 如果传入了第二个参数，给出警告信息并调用 multiplyVectors 方法
    if(void 0!==b)
        return console.warn("THREE.Vector3: .multiply() now only accepts one argument. Use .multiplyVectors( a, b ) instead."),this.multiplyVectors(a,b);
    # 乘以一个向量，更新当前向量的坐标值，并返回更新后的向量
    this.x*=a.x;
    this.y*=a.y;
    this.z*=a.z;
    return this
}
# 乘以标量，更新当前向量的坐标值，并返回更新后的向量
multiplyScalar:function(a){
    this.x*=a;
    this.y*=a;
    this.z*=a;
    return this
}
# 乘以两个向量的坐标值，更新当前向量的坐标值，并返回更新后的向量
multiplyVectors:function(a,b){
    this.x=a.x*b.x;
    this.y=a.y*b.y;
    this.z=a.z*b.z;
    return this
}
# 应用欧拉角旋转到当前向量
applyEuler:function(){
    var a;
    return function(b){
        # 如果传入的参数不是 THREE.Euler 类型，则给出错误信息
        !1===b instanceof THREE.Euler&&console.error("THREE.Vector3: .applyEuler() now expects a Euler rotation rather than a Vector3 and order.");
        # 如果 a 未定义，则创建一个 THREE.Quaternion 对象
        void 0===a&&(a=new THREE.Quaternion);
        # 应用四元数旋转到当前向量
        this.applyQuaternion(a.setFromEuler(b));
        return this
    }
}()
# 应用轴角旋转到当前向量
applyAxisAngle:function(){
    var a;
    return function(b,c){
        # 如果 a 未定义，则创建一个 THREE.Quaternion 对象
        void 0===a&&(a=new THREE.Quaternion);
        # 应用四元数旋转到当前向量
        this.applyQuaternion(a.setFromAxisAngle(b,c));
        return this
    }
}()
# 应用 3x3 矩阵变换到当前向量
applyMatrix3:function(a){
    var b=this.x,
    c=this.y,
    d=this.z;
    a=a.elements;
    this.x=a[0]*b+a[3]*c+a[6]*d;
    this.y=a[1]*b+a[4]*c+a[7]*d;
    this.z=a[2]*b+a[5]*c+a[8]*d;
    return this
}
# 应用 4x4 矩阵变换到当前向量
applyMatrix4:function(a){
    var b=this.x,
    c=this.y,
    d=this.z;
    a=a.elements;
    this.x=a[0]*b+a[4]*c+a[8]*d+a[12];
    this.y=a[1]*b+a[5]*c+a[9]*d+a[13];
    this.z=a[2]*b+a[6]*c+a[10]*d+a[14];
    return this
}
# 应用投影矩阵变换到当前向量
applyProjection:function(a){
    var b=this.x,
    c=this.y,
    d=this.z;
    a=a.elements;
    var e=1/(a[3]*b+a[7]*c+a[11]*d+a[15]);
    this.x=(a[0]*b+a[4]*c+a[8]*d+a[12])*e;
    this.y=(a[1]*b+a[5]*c+a[9]*d+a[13])*e;
    this.z=(a[2]*b+a[6]*c+a[10]*d+a[14])*e;
    return this
}
# 将向量乘以四元数
applyQuaternion:function(a){
    var b=this.x,c=this.y,d=this.z,e=a.x,f=a.y,g=a.z;
    a=a.w;
    var h=a*b+f*d-g*c,
        k=a*c+g*b-e*d,
        n=a*d+e*c-f*b,
        b=-e*b-f*c-g*d;
    this.x=h*a+b*-e+k*-g-n*-f;
    this.y=k*a+b*-f+n*-e-h*-g;
    this.z=n*a+b*-g+h*-f-k*-e;
    return this
},

# 将向量投影到屏幕空间
project:function(){
    var a;
    return function(b){
        void 0===a&&(a=new THREE.Matrix4);
        a.multiplyMatrices(b.projectionMatrix,a.getInverse(b.matrixWorld));
        return this.applyProjection(a)
    }
}(),

# 将向量从屏幕空间反投影到世界空间
unproject:function(){
    var a;
    return function(b){
        void 0===a&&(a=new THREE.Matrix4);
        a.multiplyMatrices(b.matrixWorld,a.getInverse(b.projectionMatrix));
        return this.applyProjection(a)
    }
}(),

# 将向量转换为另一个方向
transformDirection:function(a){
    var b=this.x,c=this.y,d=this.z;
    a=a.elements;
    this.x=a[0]*b+a[4]*c+a[8]*d;
    this.y=a[1]*b+a[5]*c+a[9]*d;
    this.z=a[2]*b+a[6]*c+a[10]*d;
    this.normalize();
    return this
},

# 将向量除以另一个向量
divide:function(a){
    this.x/=a.x;
    this.y/=a.y;
    this.z/=a.z;
    return this
},

# 将向量除以标量
divideScalar:function(a){
    if(0!==a){
        a=1/a;
        this.x*=a;
        this.y*=a;
        this.z*=a;
    } else {
        this.z=this.y=this.x=0;
    }
    return this
},

# 将向量限制在最小和最大值之间
min:function(a){
    this.x>a.x&&(this.x=a.x);
    this.y>a.y&&(this.y=a.y);
    this.z>a.z&&(this.z=a.z);
    return this
},

# 将向量限制在最小和最大值之间
max:function(a){
    this.x<a.x&&(this.x=a.x);
    this.y<a.y&&(this.y=a.y);
    this.z<a.z&&(this.z=a.z);
    return this
},

# 将向量限制在指定范围内
clamp:function(a,b){
    this.x<a.x?this.x=a.x:this.x>b.x&&(this.x=b.x);
    this.y<a.y?this.y=a.y:this.y>b.y&&(this.y=b.y);
    this.z<a.z?this.z=a.z:this.z>b.z&&(this.z=b.z);
    return this
},

# 将向量限制在指定范围内
clampScalar:function(){
    var a,b;
    return function(c,d){
        void 0===a&&(a=new THREE.Vector3);
        void 0===b&&(b=new THREE.Vector3);
        a.set(c,c,c);
        b.set(d,d,d);
        return this.clamp(a,b)
    }
}()
# 向量对象的数学运算方法
floor:function(){
    # 对向量的每个分量进行向下取整
    this.x=Math.floor(this.x);
    this.y=Math.floor(this.y);
    this.z=Math.floor(this.z);
    return this
},
ceil:function(){
    # 对向量的每个分量进行向上取整
    this.x=Math.ceil(this.x);
    this.y=Math.ceil(this.y);
    this.z=Math.ceil(this.z);
    return this
},
round:function(){
    # 对向量的每个分量进行四舍五入
    this.x=Math.round(this.x);
    this.y=Math.round(this.y);
    this.z=Math.round(this.z);
    return this
},
roundToZero:function(){
    # 对向量的每个分量进行向零取整
    this.x=0>this.x?Math.ceil(this.x):Math.floor(this.x);
    this.y=0>this.y?Math.ceil(this.y):Math.floor(this.y);
    this.z=0>this.z?Math.ceil(this.z):Math.floor(this.z);
    return this
},
negate:function(){
    # 对向量的每个分量取负
    this.x=-this.x;
    this.y=-this.y;
    this.z=-this.z;
    return this
},
dot:function(a){
    # 计算向量与另一个向量的点积
    return this.x*a.x+this.y*a.y+this.z*a.z
},
lengthSq:function(){
    # 计算向量的长度的平方
    return this.x*this.x+this.y*this.y+this.z*this.z
},
length:function(){
    # 计算向量的长度
    return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)
},
lengthManhattan:function(){
    # 计算向量的曼哈顿长度
    return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)
},
normalize:function(){
    # 将向量归一化
    return this.divideScalar(this.length())
},
setLength:function(a){
    # 设置向量的长度
    var b=this.length();
    0!==b&&a!==b&&this.multiplyScalar(a/b);
    return this
},
lerp:function(a,b){
    # 线性插值
    this.x+=(a.x-this.x)*b;
    this.y+=(a.y-this.y)*b;
    this.z+=(a.z-this.z)*b;
    return this
},
cross:function(a,b){
    # 计算向量与另一个向量的叉积
    if(void 0!==b)
        return console.warn("THREE.Vector3: .cross() now only accepts one argument. Use .crossVectors( a, b ) instead."),this.crossVectors(a,b);
    var c=this.x,d=this.y,e=this.z;
    this.x=d*a.z-e*a.y;
    this.y=e*a.x-c*a.z;
    this.z=c*a.y-d*a.x;
    return this
},
crossVectors:function(a,b){
    # 计算两个向量的叉积
    var c=a.x,d=a.y,e=a.z,f=b.x,g=b.y,h=b.z;
    this.x=d*h-e*g;
    this.y=e*f-c*h;
    this.z=c*g-d*f;
    return this
}
projectOnVector:function(){
    var a,b;
    return function(c){
        // 如果变量a未定义，则初始化为一个新的THREE.Vector3对象
        void 0===a&&(a=new THREE.Vector3);
        // 将向量c的值复制给向量a，并对其进行归一化
        a.copy(c).normalize();
        // 计算当前向量与向量a的点积
        b=this.dot(a);
        // 将当前向量设置为向量a，并乘以点积b
        return this.copy(a).multiplyScalar(b)
    }
}(),

projectOnPlane:function(){
    var a;
    return function(b){
        // 如果变量a未定义，则初始化为一个新的THREE.Vector3对象
        void 0===a&&(a=new THREE.Vector3);
        // 将当前向量的值复制给向量a，并投影到向量b上
        a.copy(this).projectOnVector(b);
        // 返回当前向量减去向量a的结果
        return this.sub(a)
    }
}(),

reflect:function(){
    var a;
    return function(b){
        // 如果变量a未定义，则初始化为一个新的THREE.Vector3对象
        void 0===a&&(a=new THREE.Vector3);
        // 返回当前向量减去向量b的两倍点积
        return this.sub(a.copy(b).multiplyScalar(2*this.dot(b)))
    }
}(),

angleTo:function(a){
    // 计算当前向量与向量a的夹角
    a=this.dot(a)/(this.length()*a.length());
    return Math.acos(THREE.Math.clamp(a,-1,1))
},

distanceTo:function(a){
    // 返回当前向量到向量a的距离
    return Math.sqrt(this.distanceToSquared(a))
},

distanceToSquared:function(a){
    // 计算当前向量到向量a的距离的平方
    var b=this.x-a.x,c=this.y-a.y;a=this.z-a.z;
    return b*b+c*c+a*a
},

setEulerFromRotationMatrix:function(a,b){
    console.error("THREE.Vector3: .setEulerFromRotationMatrix() has been removed. Use Euler.setFromRotationMatrix() instead.")
},

setEulerFromQuaternion:function(a,b){
    console.error("THREE.Vector3: .setEulerFromQuaternion() has been removed. Use Euler.setFromQuaternion() instead.")
},

getPositionFromMatrix:function(a){
    console.warn("THREE.Vector3: .getPositionFromMatrix() has been renamed to .setFromMatrixPosition().");
    return this.setFromMatrixPosition(a)
},

getScaleFromMatrix:function(a){
    console.warn("THREE.Vector3: .getScaleFromMatrix() has been renamed to .setFromMatrixScale().");
    return this.setFromMatrixScale(a)
},

getColumnFromMatrix:function(a,b){
    console.warn("THREE.Vector3: .getColumnFromMatrix() has been renamed to .setFromMatrixColumn().");
    return this.setFromMatrixColumn(a,b)
}
# 定义一个名为 setFromMatrixPosition 的方法，参数为矩阵 a
setFromMatrixPosition:function(a){
    # 从矩阵中获取 x 坐标的值
    this.x=a.elements[12];
    # 从矩阵中获取 y 坐标的值
    this.y=a.elements[13];
    # 从矩阵中获取 z 坐标的值
    this.z=a.elements[14];
    # 返回包含 x、y、z 坐标的对象
    return this
},
# 定义一个名为 setFromMatrixScale 的方法，参数为矩阵 a
setFromMatrixScale:function(a){
    # 计算矩阵中 x 方向的长度
    var b=this.set(a.elements[0],a.elements[1],a.elements[2]).length();
    # 计算矩阵中 y 方向的长度
    c=this.set(a.elements[4],a.elements[5],a.elements[6]).length();
    # 计算矩阵中 z 方向的长度
    a=this.set(a.elements[8],a.elements[9],a.elements[10]).length();
    # 设置对象的 x、y、z 值为计算得到的长度
    this.x=b;
    this.y=c;
    this.z=a;
    # 返回包含 x、y、z 值的对象
    return this
},
# 定义一个名为 setFromMatrixColumn 的方法，参数为矩阵 a 和列数 b
setFromMatrixColumn:function(a,b){
    # 计算列数对应的索引
    var c=4*a,d=b.elements;
    # 设置对象的 x、y、z 值为矩阵中对应列的值
    this.x=d[c];
    this.y=d[c+1];
    this.z=d[c+2];
    # 返回包含 x、y、z 值的对象
    return this
},
# 定义一个名为 equals 的方法，参数为对象 a
equals:function(a){
    # 判断对象的 x、y、z 值是否与参数对象的对应值相等
    return a.x===this.x&&a.y===this.y&&a.z===this.z
},
# 定义一个名为 fromArray 的方法，参数为数组 a 和起始索引 b
fromArray:function(a,b){
    # 如果未传入起始索引，则默认为 0
    void 0===b&&(b=0);
    # 设置对象的 x、y、z 值为数组中对应索引的值
    this.x=a[b];
    this.y=a[b+1];
    this.z=a[b+2];
    # 返回包含 x、y、z 值的对象
    return this
},
# 定义一个名为 toArray 的方法，参数为数组 a 和起始索引 b
toArray:function(a,b){
    # 如果未传入数组，则默认为空数组
    void 0===a&&(a=[]);
    # 如果未传入起始索引，则默认为 0
    void 0===b&&(b=0);
    # 将对象的 x、y、z 值存入数组对应的索引位置
    a[b]=this.x;
    a[b+1]=this.y;
    a[b+2]=this.z;
    # 返回包含 x、y、z 值的数组
    return a
},
# 定义一个名为 clone 的方法
clone:function(){
    # 返回一个新的包含对象 x、y、z 值的对象
    return new THREE.Vector3(this.x,this.y,this.z)
}
# 定义一个名为 addVectors 的函数，用于将两个向量相加
addVectors:function(a,b){this.x=a.x+b.x;this.y=a.y+b.y;this.z=a.z+b.z;this.w=a.w+b.w;return this},
# 定义一个名为 sub 的函数，用于将两个向量相减
sub:function(a,b){if(void 0!==b)return console.warn("THREE.Vector4: .sub() now only accepts one argument. Use .subVectors( a, b ) instead."),this.subVectors(a,b);this.x-=a.x;this.y-=a.y;this.z-=a.z;this.w-=a.w;return this},
# 定义一个名为 subVectors 的函数，用于将两个向量相减
subVectors:function(a,b){this.x=a.x-b.x;this.y=a.y-b.y;this.z=a.z-b.z;this.w=a.w-b.w;return this},
# 定义一个名为 multiplyScalar 的函数，用于将向量乘以标量
multiplyScalar:function(a){this.x*=a;this.y*=a;this.z*=a;this.w*=a;return this},
# 定义一个名为 applyMatrix4 的函数，用于将向量应用 4x4 矩阵
applyMatrix4:function(a){var b=this.x,c=this.y,d=this.z,e=this.w;a=a.elements;this.x=a[0]*b+a[4]*c+a[8]*d+a[12]*e;this.y=a[1]*b+a[5]*c+a[9]*d+a[13]*e;this.z=a[2]*b+a[6]*c+a[10]*d+a[14]*e;this.w=a[3]*b+a[7]*c+a[11]*d+a[15]*e;return this},
# 定义一个名为 divideScalar 的函数，用于将向量除以标量
divideScalar:function(a){0!==a?(a=1/a,this.x*=a,this.y*=a,this.z*=a,this.w*=a):(this.z=this.y=this.x=0,this.w=1);return this},
# 定义一个名为 setAxisAngleFromQuaternion 的函数，用于根据四元数设置轴角
setAxisAngleFromQuaternion:function(a){this.w=2*Math.acos(a.w);var b=Math.sqrt(1-a.w*a.w);1E-4>b?(this.x=1,this.z=this.y=0):(this.x=a.x/b,this.y=a.y/b,this.z=a.z/b);return this},
# 定义一个名为 setAxisAngleFromRotationMatrix 的函数，用于根据旋转矩阵设置轴角
setAxisAngleFromRotationMatrix:function(a){var b,c,d;a=a.elements;var e=a[0];d=a[4];var f=a[8],g=a[1],h=a[5],k=a[9];c=a[2];b=a[6];var n=a[10];if(.01>Math.abs(d-g)&&.01>Math.abs(f-c)&&.01>Math.abs(k-b)){if(.1>Math.abs(d+g)&&.1>Math.abs(f+c)&&.1>Math.abs(k+b)&&.1>Math.abs(e+h+n-3))return this.set(1,0,0,0),this;a=Math.PI;e=(e+1)/2;h=(h+1)/2;n=(n+1)/2;d=(d+g)/4;f=(f+c)/4;k=(k+b)/4;e>h&&e>n?.01>e?(b=0,d=c=.707106781):(b=Math.sqrt(e),c=d/b,d=f/b):h>n?.01>h?(b=.707106781,c=0,d=.707106781):(c=Math.sqrt(h),
// 设置四元数的值
b=d/c,d=k/c):.01>n?(c=b=.707106781,d=0):(d=Math.sqrt(n),b=f/d,c=k/d);this.set(b,c,d,a);return this}
// 计算四元数的长度，并进行归一化
a=Math.sqrt((b-k)*(b-k)+(f-c)*(f-c)+(g-d)*(g-d));.001>Math.abs(a)&&(a=1);this.x=(b-k)/a;this.y=(f-c)/a;this.z=(g-d)/a;this.w=Math.acos((e+h+n-1)/2);return this}
// 比较两个四元数，取较小值
min:function(a){this.x>a.x&&(this.x=a.x);this.y>a.y&&(this.y=a.y);this.z>a.z&&(this.z=a.z);this.w>a.w&&(this.w=a.w);return this}
// 比较两个四元数，取较大值
max:function(a){this.x<a.x&&(this.x=a.x);this.y<a.y&&(this.y=a.y);this.z<a.z&&(this.z=a.z);this.w<a.w&&(this.w=a.w);return this}
// 将四元数的值限制在指定范围内
clamp:function(a,b){this.x<a.x?this.x=a.x:this.x>b.x&&(this.x=b.x);this.y<a.y?this.y=a.y:this.y>b.y&&(this.y=b.y);this.z<a.z?this.z=a.z:this.z>b.z&&(this.z=b.z);this.w<a.w?this.w=a.w:this.w>b.w&&(this.w=b.w);return this}
// 将四元数的值限制在标量范围内
clampScalar:function(){var a,b;return function(c,d){void 0===a&&(a=new THREE.Vector4,b=new THREE.Vector4);a.set(c,c,c,c);b.set(d,d,d,d);return this.clamp(a,b)}}()
// 对四元数的值向下取整
floor:function(){this.x=Math.floor(this.x);this.y=Math.floor(this.y);this.z=Math.floor(this.z);this.w=Math.floor(this.w);return this}
// 对四元数的值向上取整
ceil:function(){this.x=Math.ceil(this.x);this.y=Math.ceil(this.y);this.z=Math.ceil(this.z);this.w=Math.ceil(this.w);return this}
// 对四元数的值四舍五入
round:function(){this.x=Math.round(this.x);this.y=Math.round(this.y);this.z=Math.round(this.z);this.w=Math.round(this.w);return this}
// 对四元数的值向零取整
roundToZero:function(){this.x=0>this.x?Math.ceil(this.x):Math.floor(this.x);this.y=0>this.y?Math.ceil(this.y):Math.floor(this.y);this.z=0>this.z?Math.ceil(this.z):Math.floor(this.z);this.w=0>this.w?Math.ceil(this.w):Math.floor(this.w);
return this}, // 返回当前对象
negate:function(){this.x=-this.x;this.y=-this.y;this.z=-this.z;this.w=-this.w;return this}, // 对当前对象的坐标进行取反操作，并返回当前对象
dot:function(a){return this.x*a.x+this.y*a.y+this.z*a.z+this.w*a.w}, // 返回当前对象与参数对象的点积
lengthSq:function(){return this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w}, // 返回当前对象的长度的平方
length:function(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w)}, // 返回当前对象的长度
lengthManhattan:function(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)+Math.abs(this.w)}, // 返回当前对象的曼哈顿长度
normalize:function(){return this.divideScalar(this.length())}, // 对当前对象进行标准化操作，并返回当前对象
setLength:function(a){var b=this.length();0!==b&&a!==b&&this.multiplyScalar(a/b);return this}, // 设置当前对象的长度，并返回当前对象
lerp:function(a,b){this.x+=(a.x-this.x)*b;this.y+=(a.y-this.y)*b;this.z+=(a.z-this.z)*b;this.w+=(a.w-this.w)*b;return this}, // 对当前对象进行线性插值操作，并返回当前对象
equals:function(a){return a.x===this.x&&a.y===this.y&&a.z===this.z&&a.w===this.w}, // 判断当前对象是否与参数对象相等
fromArray:function(a,b){void 0===b&&(b=0);this.x=a[b];this.y=a[b+1];this.z=a[b+2];this.w=a[b+3];return this}, // 从数组中设置当前对象的坐标，并返回当前对象
toArray:function(a,b){void 0===a&&(a=[]);void 0===b&&(b=0);a[b]=this.x;a[b+1]=this.y;a[b+2]=this.z;a[b+3]=this.w;return a}, // 将当前对象的坐标存入数组中，并返回数组
clone:function(){return new THREE.Vector4(this.x,this.y,this.z,this.w)}}; // 克隆当前对象并返回新对象
THREE.Euler=function(a,b,c,d){this._x=a||0;this._y=b||0;this._z=c||0;this._order=d||THREE.Euler.DefaultOrder}; // 定义THREE.Euler构造函数
THREE.Euler.RotationOrders="XYZ YZX ZXY XZY YXZ ZYX".split(" "); // 定义旋转顺序数组
THREE.Euler.DefaultOrder="XYZ"; // 定义默认旋转顺序
THREE.Euler.prototype={constructor:THREE.Euler,_x:0,_y:0,_z:0,_order:THREE.Euler.DefaultOrder,get x(){return this._x},set x(a){this._x=a;this.onChangeCallback()},get y(){return this._y},set y(a){this._y=a;this.onChangeCallback()},get z(){return this._z},set z(a){this._z=a;this.onChangeCallback()},get order(){return this._order},set order(a){this._order=a;this.onChangeCallback()},set:function(a,b,c,d){this._x=a;this._y=b;this._z=c;this._order=d||this._order;this.onChangeCallback();return this},copy:function(a){this._x=
# 设置欧拉角，根据给定的旋转矩阵和顺序
setFromRotationMatrix:function(a,b){
    var c=THREE.Math.clamp,
    d=a.elements,
    e=d[0],
    f=d[4],
    g=d[8],
    h=d[1],
    k=d[5],
    n=d[9],
    p=d[2],
    q=d[6],
    d=d[10];
    b=b||this._order;
    # 根据给定的顺序设置欧拉角
    "XYZ"===b?(this._y=Math.asin(c(g,-1,1)),
    .99999>Math.abs(g)?(this._x=Math.atan2(-n,d),this._z=Math.atan2(-f,e)):(this._x=Math.atan2(q,k),this._z=0)):
    "YXZ"===b?(this._x=Math.asin(-c(n,-1,1)),
    .99999>Math.abs(n)?(this._y=Math.atan2(g,d),this._z=Math.atan2(h,k)):(this._y=Math.atan2(-p,e),this._z=0)):
    "ZXY"===b?(this._x=Math.asin(c(q,-1,1)),
    .99999>Math.abs(q)?(this._y=Math.atan2(-p,d),this._z=Math.atan2(-f,k)):(this._y=0,this._z=Math.atan2(h,e)):
    "ZYX"===b?(this._y=Math.asin(-c(p,-1,1)),
    .99999>Math.abs(p)?(this._x=Math.atan2(q,d),this._z=Math.atan2(h,e)):(this._x=0,this._z=Math.atan2(-f,k)):
    "YZX"===b?(this._z=Math.asin(c(h,-1,1)),
    .99999>Math.abs(h)?(this._x=Math.atan2(-n,k),this._y=Math.atan2(-p,e)):(this._x=0,this._y=Math.atan2(g,d)):
    "XZY"===b?(this._z=Math.asin(-c(f,-1,1)),
    .99999>Math.abs(f)?(this._x=Math.atan2(q,k),this._y=Math.atan2(g,e)):(this._x=Math.atan2(-n,d),this._y=0)):
    console.warn("THREE.Euler: .setFromRotationMatrix() given unsupported order: "+b);
    this._order=b;
    # 调用回调函数
    this.onChangeCallback();
    return this
},
# 根据四元数设置欧拉角
setFromQuaternion:function(a,b,c){
    var d=THREE.Math.clamp,
    e=a.x*a.x,
    f=a.y*a.y,
    g=a.z*a.z,
    h=a.w*a.w;
    b=b||this._order;
    "XYZ"===b?(this._x=Math.atan2(2*(a.x*a.w-a.y*a.z),h-e-f+g),
    this._y=Math.asin(d(2*(a.x*a.z+a.y*a.w),-1,1)),
    this._z=Math.atan2(2*(a.z*a.w-a.x*

...
# 定义一个名为Euler的函数，用于创建和操作欧拉角对象
THREE.Euler = function(a, b) {
    # 如果传入参数a不为undefined，则将a赋值给this.start，否则创建一个新的THREE.Vector3对象赋值给this.start
    this.start = void 0 !== a ? a : new THREE.Vector3;
    # 如果传入参数b不为undefined，则将b赋值给this.end，否则创建一个新的THREE.Vector3对象赋值给this.end
    this.end = void 0 !== b ? b : new THREE.Vector3;
};
# 定义 Line3 对象的原型
THREE.Line3.prototype={
    constructor:THREE.Line3,
    # 设置 Line3 对象的起点和终点
    set:function(a,b){
        this.start.copy(a);
        this.end.copy(b);
        return this
    },
    # 复制另一个 Line3 对象的起点和终点
    copy:function(a){
        this.start.copy(a.start);
        this.end.copy(a.end);
        return this
    },
    # 计算 Line3 对象的中心点
    center:function(a){
        return(a||new THREE.Vector3).addVectors(this.start,this.end).multiplyScalar(.5)
    },
    # 计算 Line3 对象的方向向量
    delta:function(a){
        return(a||new THREE.Vector3).subVectors(this.end,this.start)
    },
    # 计算 Line3 对象的长度的平方
    distanceSq:function(){
        return this.start.distanceToSquared(this.end)
    },
    # 计算 Line3 对象的长度
    distance:function(){
        return this.start.distanceTo(this.end)
    },
    # 计算 Line3 对象上的点
    at:function(a,b){
        var c=b||new THREE.Vector3;
        return this.delta(c).multiplyScalar(a).add(this.start)
    },
    # 计算 Line3 对象上距离给定点最近的点的参数
    closestPointToPointParameter:function(){
        var a=new THREE.Vector3,
            b=new THREE.Vector3;
        return function(c,d){
            a.subVectors(c,this.start);
            b.subVectors(this.end,this.start);
            var e=b.dot(b),
                e=b.dot(a)/e;
            d&&(e=THREE.Math.clamp(e,0,1));
            return e
        }
    }(),
    # 计算 Line3 对象上距离给定点最近的点
    closestPointToPoint:function(a,b,c){
        a=this.closestPointToPointParameter(a,b);
        c=c||new THREE.Vector3;
        return this.delta(c).multiplyScalar(a).add(this.start)
    },
    # 将 Line3 对象应用矩阵变换
    applyMatrix4:function(a){
        this.start.applyMatrix4(a);
        this.end.applyMatrix4(a);
        return this
    },
    # 判断 Line3 对象是否与另一个 Line3 对象相等
    equals:function(a){
        return a.start.equals(this.start)&&a.end.equals(this.end)
    },
    # 克隆 Line3 对象
    clone:function(){
        return(new THREE.Line3).copy(this)
    }
};

# 定义 Box2 对象
THREE.Box2=function(a,b){
    this.min=void 0!==a?a:new THREE.Vector2(Infinity,Infinity);
    this.max=void 0!==b?b:new THREE.Vector2(-Infinity,-Infinity)
};

THREE.Box2.prototype={
    constructor:THREE.Box2,
    # 设置 Box2 对象的最小点和最大点
    set:function(a,b){
        this.min.copy(a);
        this.max.copy(b);
        return this
    },
    # 根据给定点集合设置 Box2 对象的最小点和最大点
    setFromPoints:function(a){
        this.makeEmpty();
        for(var b=0,c=a.length;b<c;b++)
            this.expandByPoint(a[b]);
        return this
    },
    # 根据中心点和尺寸设置 Box2 对象的最小点和最大点
    setFromCenterAndSize:function(){
        var a=new THREE.Vector2;
        return function(b,c){
            var d=a.copy(c).multiplyScalar(.5);
            this.min.copy(b).sub(d);
            this.max.copy(b).add(d);
            return this
        }
    }(),
    # 复制另一个 Box2 对象的最小点和最大点
    copy:function(a){
        this.min.copy(a.min);
        this.max.copy(a.max);
        return this
    },
    # 使 Box2 对象为空
    makeEmpty:function(){
        this.min.x=
    }
}
# 设置 this 对象的最小 y 值为正无穷，最大 x 和 y 值为负无穷，然后返回 this 对象
this.min.y=Infinity;this.max.x=this.max.y=-Infinity;return this},
# 判断 this 对象是否为空，即最大 x 值小于最小 x 值，或者最大 y 值小于最小 y 值
empty:function(){return this.max.x<this.min.x||this.max.y<this.min.y},
# 计算 this 对象的中心点坐标，如果传入参数 a，则将结果存储在 a 中
center:function(a){return(a||new THREE.Vector2).addVectors(this.min,this.max).multiplyScalar(.5)},
# 计算 this 对象的尺寸，如果传入参数 a，则将结果存储在 a 中
size:function(a){return(a||new THREE.Vector2).subVectors(this.max,this.min)},
# 根据给定点扩展 this 对象的范围
expandByPoint:function(a){this.min.min(a);this.max.max(a);return this},
# 根据给定向量扩展 this 对象的范围
expandByVector:function(a){this.min.sub(a);this.max.add(a);return this},
# 根据给定标量扩展 this 对象的范围
expandByScalar:function(a){this.min.addScalar(-a);this.max.addScalar(a);return this},
# 判断给定点是否在 this 对象内部
containsPoint:function(a){return a.x<this.min.x||a.x>this.max.x||a.y<this.min.y||a.y>this.max.y?!1:!0},
# 判断给定盒子是否完全包含在 this 对象内部
containsBox:function(a){return this.min.x<=a.min.x&&a.max.x<=this.max.x&&this.min.y<=a.min.y&&a.max.y<=this.max.y?!0:!1},
# 计算给定点在 this 对象中的参数化坐标
getParameter:function(a,b){return(b||new THREE.Vector2).set((a.x-this.min.x)/(this.max.x-this.min.x),(a.y-this.min.y)/(this.max.y-this.min.y))},
# 判断给定盒子是否与 this 对象相交
isIntersectionBox:function(a){return a.max.x<this.min.x||a.min.x>this.max.x||a.max.y<this.min.y||a.min.y>this.max.y?!1:!0},
# 将给定点限制在 this 对象的范围内
clampPoint:function(a,b){return(b||new THREE.Vector2).copy(a).clamp(this.min,this.max)},
# 计算给定点到 this 对象的距离
distanceToPoint:function(){var a=new THREE.Vector2;return function(b){return a.copy(b).clamp(this.min,this.max).sub(b).length()}}(),
# 计算 this 对象与给定盒子的交集
intersect:function(a){this.min.max(a.min);this.max.min(a.max);return this},
# 计算 this 对象与给定盒子的并集
union:function(a){this.min.min(a.min);this.max.max(a.max);return this},
# 将 this 对象平移给定向量
translate:function(a){this.min.add(a);this.max.add(a);return this},
# 判断 this 对象是否与给定对象相等
equals:function(a){return a.min.equals(this.min)&&a.max.equals(this.max)},
# 克隆 this 对象并返回
clone:function(){return(new THREE.Box2).copy(this)}};
# 定义一个名为 THREE.Box3 的函数，接受两个参数 a 和 b
THREE.Box3=function(a,b){
# 如果 a 有值，则将其赋给 this 对象的最小值，否则设置为正无穷；如果 b 有值，则将其赋给 this 对象的最大值，否则设置为负无穷
this.min=void 0!==a?a:new THREE.Vector3(Infinity,Infinity,Infinity);this.max=void 0!==b?b:new THREE.Vector3(-Infinity,-Infinity,-Infinity)};
# 定义了一个名为 Box3 的原型对象，包含了一系列用于操作三维空间中的包围盒的方法
THREE.Box3.prototype={
    # 构造函数，用于设置包围盒的最小和最大顶点
    constructor:THREE.Box3,
    set:function(a,b){
        this.min.copy(a);
        this.max.copy(b);
        return this
    },
    # 根据给定的点集合，设置包围盒的最小和最大顶点
    setFromPoints:function(a){
        this.makeEmpty();
        for(var b=0,c=a.length;b<c;b++)
            this.expandByPoint(a[b]);
        return this
    },
    # 根据中心点和尺寸设置包围盒的最小和最大顶点
    setFromCenterAndSize:function(){
        var a=new THREE.Vector3;
        return function(b,c){
            var d=a.copy(c).multiplyScalar(.5);
            this.min.copy(b).sub(d);
            this.max.copy(b).add(d);
            return this
        }
    }(),
    # 根据对象的几何体或缓冲几何体设置包围盒的最小和最大顶点
    setFromObject:function(){
        var a=new THREE.Vector3;
        return function(b){
            var c=this;
            b.updateMatrixWorld(!0);
            this.makeEmpty();
            b.traverse(function(b){
                var e=b.geometry;
                if(void 0!==e)
                    if(e instanceof THREE.Geometry)
                        for(var f=e.vertices,e=0,g=f.length;e<g;e++)
                            a.copy(f[e]),a.applyMatrix4(b.matrixWorld),c.expandByPoint(a);
                    else if(e instanceof THREE.BufferGeometry&&void 0!==e.attributes.position)
                        for(f=e.attributes.position.array,e=0,g=f.length;e<g;e+=3)
                            a.set(f[e],f[e+1],f[e+2]),a.applyMatrix4(b.matrixWorld),c.expandByPoint(a)
            });
            return this
        }
    }(),
    # 复制另一个包围盒的最小和最大顶点
    copy:function(a){
        this.min.copy(a.min);
        this.max.copy(a.max);
        return this
    },
    # 使包围盒为空
    makeEmpty:function(){
        this.min.x=this.min.y=this.min.z=Infinity;
        this.max.x=this.max.y=this.max.z=-Infinity;
        return this
    },
    # 判断包围盒是否为空
    empty:function(){
        return this.max.x<this.min.x||this.max.y<this.min.y||this.max.z<this.min.z
    },
    # 计算包围盒的中心点
    center:function(a){
        return(a||new THREE.Vector3).addVectors(this.min,this.max).multiplyScalar(.5)
    },
    # 计算包围盒的尺寸
    size:function(a){
        return(a||new THREE.Vector3).subVectors(this.max,this.min)
    },
    # 根据给定的点扩展包围盒
    expandByPoint:function(a){
        this.min.min(a);
        this.max.max(a);
        return this
    },
    # 根据给定的向量扩展包围盒
    expandByVector:function(a){
        this.min.sub(a);
# 将给定的点添加到当前包围盒中，然后返回当前包围盒
this.max.add(a);return this},expandByScalar:function(a){this.min.addScalar(-a);this.max.addScalar(a);return this},
# 检查给定的点是否在当前包围盒内，返回布尔值
containsPoint:function(a){return a.x<this.min.x||a.x>this.max.x||a.y<this.min.y||a.y>this.max.y||a.z<this.min.z||a.z>this.max.z?!1:!0},
# 检查给定的包围盒是否完全包含在当前包围盒内，返回布尔值
containsBox:function(a){return this.min.x<=a.min.x&&a.max.x<=this.max.x&&this.min.y<=a.min.y&&a.max.y<=this.max.y&&this.min.z<=a.min.z&&a.max.z<=this.max.z?!0:!1},
# 计算给定点在当前包围盒内的参数化坐标
getParameter:function(a,b){return(b||new THREE.Vector3).set((a.x-this.min.x)/(this.max.x-this.min.x),(a.y-this.min.y)/(this.max.y-this.min.y),(a.z-this.min.z)/(this.max.z-this.min.z))},
# 检查给定的包围盒是否与当前包围盒相交，返回布尔值
isIntersectionBox:function(a){return a.max.x<this.min.x||a.min.x>this.max.x||a.max.y<this.min.y||a.min.y>this.max.y||a.max.z<this.min.z||a.min.z>this.max.z?!1:!0},
# 将给定点限制在当前包围盒内，返回限制后的点
clampPoint:function(a,b){return(b||new THREE.Vector3).copy(a).clamp(this.min,this.max)},
# 计算给定点到当前包围盒的距离
distanceToPoint:function(){var a=new THREE.Vector3;return function(b){return a.copy(b).clamp(this.min,this.max).sub(b).length()}}(),
# 计算当前包围盒的外接球体
getBoundingSphere:function(){var a=new THREE.Vector3;return function(b){b=b||new THREE.Sphere;b.center=this.center();b.radius=.5*this.size(a).length();return b}}(),
# 计算当前包围盒与给定包围盒的交集
intersect:function(a){this.min.max(a.min);this.max.min(a.max);return this},
# 计算当前包围盒与给定包围盒的并集
union:function(a){this.min.min(a.min);this.max.max(a.max);return this},
# 将当前包围盒应用给定的 4x4 矩阵变换
applyMatrix4:function(){var a=[new THREE.Vector3,new THREE.Vector3,new THREE.Vector3,new THREE.Vector3,new THREE.Vector3,new THREE.Vector3,new THREE.Vector3,new THREE.Vector3];return function(b){a[0].set(this.min.x,this.min.y,
# 创建一个名为 Box3 的对象，包含一系列用于处理三维空间中的包围盒的方法
THREE.Box3 = function () {
    # 创建一个名为 min 的属性，表示包围盒的最小点
    this.min = new THREE.Vector3(+Infinity, +Infinity, +Infinity);
    # 创建一个名为 max 的属性，表示包围盒的最大点
    this.max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
    # 返回一个空的包围盒
    this.makeEmpty();
};

# 为 Box3 对象添加一系列方法
THREE.Box3.prototype = {
    # 设置包围盒的最小点和最大点
    set: function (a, b, c, d, e, f) {
        var n = this.elements;
        n[0] = a;
        n[3] = b;
        n[6] = c;
        n[1] = d;
        n[4] = e;
        n[7] = f;
        n[2] = g;
        n[5] = h;
        n[8] = k;
        return this;
    },
    # 将包围盒恢复为初始状态
    identity: function () {
        this.set(1, 0, 0, 0, 1, 0, 0, 0, 1);
        return this;
    },
    # 复制另一个包围盒的属性
    copy: function (a) {
        a = a.elements;
        this.set(a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]);
        return this;
    },
    # 乘以一个三维向量
    multiplyVector3: function (a) {
        console.warn("THREE.Matrix3: .multiplyVector3() has been removed. Use vector.applyMatrix3( matrix ) instead.");
        return a.applyMatrix3(this);
    },
    # 乘以一个三维向量数组
    multiplyVector3Array: function (a) {
        console.warn("THREE.Matrix3: .multiplyVector3Array() has been renamed. Use matrix.applyToVector3Array( array ) instead.");
        return this.applyToVector3Array(a);
    },
    # 将矩阵应用到三维向量数组上
    applyToVector3Array: function () {
        var a = new THREE.Vector3;
        return function (b, c, d) {
            void 0 === c && (c = 0);
            void 0 === d && (d = b.length);
            for (var e = 0; e < d; e += 3, c += 3) {
                a.x = b[c];
                a.y = b[c + 1];
                a.z = b[c + 2];
                a.applyMatrix3(this);
                b[c] = a.x;
                b[c + 1] = a.y;
                b[c + 2] = a.z;
            }
            return b;
        }
    }(),
    # 将矩阵的每个元素乘以一个标量
    multiplyScalar: function (a) {
        var b = this.elements;
        b[0] *= a;
        b[3] *= a;
        b[6] *= a;

... (代码太长，未完待续)
# 定义一个名为 THREE 的对象，包含 Matrix4 和 Matrix3 两个属性
THREE.Matrix4=function(){
    # 创建一个名为 elements 的 Float32Array 数组，包含 16 个元素，代表 4x4 的矩阵
    this.elements=new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);
    # 如果传入参数的长度大于 0，则输出错误信息
    0<arguments.length&&console.error("THREE.Matrix4: the constructor no longer reads arguments. use .set() instead.")
};
# 定义一个名为 Matrix3 的属性，包含多个方法
THREE.Matrix3={
    # 定义一个名为 identity 的方法，将矩阵设置为单位矩阵
    identity:function(){
        # 将矩阵的 elements 数组设置为单位矩阵的值
        this.elements=new Float32Array([1,0,0,0,1,0,0,0,1]);
        # 返回当前对象
        return this
    },
    # 定义一个名为 copy 的方法，将当前矩阵的值复制给另一个矩阵
    copy:function(a){
        # 将参数 a 的 elements 数组复制给当前矩阵的 elements 数组
        this.elements.set(a.elements);
        # 返回当前对象
        return this
    },
    # 定义一个名为 determinant 的方法，计算矩阵的行列式
    determinant:function(){
        # 获取当前矩阵的 elements 数组
        var a=this.elements,
        # 分别获取矩阵的各个元素
        b=a[0],c=a[1],d=a[2],e=a[3],f=a[4],g=a[5],h=a[6],k=a[7],a=a[8];
        # 计算并返回矩阵的行列式
        return b*f*a-b*g*k-c*e*a+c*g*h+d*e*k-d*f*h
    },
    # 定义一个名为 getInverse 的方法，计算矩阵的逆矩阵
    getInverse:function(a,b){
        # 获取参数 a 的 elements 数组和当前矩阵的 elements 数组
        var c=a.elements,d=this.elements;
        # 计算并设置当前矩阵的逆矩阵值
        d[0]=c[10]*c[5]-c[6]*c[9];
        d[1]=-c[10]*c[1]+c[2]*c[9];
        d[2]=c[6]*c[1]-c[2]*c[5];
        d[3]=-c[10]*c[4]+c[6]*c[8];
        d[4]=c[10]*c[0]-c[2]*c[8];
        d[5]=-c[6]*c[0]+c[2]*c[4];
        d[6]=c[9]*c[4]-c[5]*c[8];
        d[7]=-c[9]*c[0]+c[1]*c[8];
        d[8]=c[5]*c[0]-c[1]*c[4];
        # 计算当前矩阵的行列式
        c=c[0]*d[0]+c[1]*d[3]+c[2]*d[6];
        # 如果行列式为 0
        if(0===c){
            # 如果参数 b 为真，则抛出错误
            if(b)throw Error("Matrix3.getInverse(): can't invert matrix, determinant is 0");
            # 否则输出警告信息
            console.warn("Matrix3.getInverse(): can't invert matrix, determinant is 0");
            # 将当前矩阵设置为单位矩阵
            this.identity();
            # 返回当前对象
            return this
        }
        # 将当前矩阵的每个元素乘以 1/行列式
        this.multiplyScalar(1/c);
        # 返回当前对象
        return this
    },
    # 定义一个名为 transpose 的方法，计算矩阵的转置矩阵
    transpose:function(){
        # 获取当前矩阵的 elements 数组
        var a,b=this.elements;
        # 交换矩阵的部分元素值
        a=b[1];b[1]=b[3];b[3]=a;
        a=b[2];b[2]=b[6];b[6]=a;
        a=b[5];b[5]=b[7];b[7]=a;
        # 返回当前对象
        return this
    },
    # 定义一个名为 flattenToArrayOffset 的方法，将矩阵的元素值扁平化到数组中
    flattenToArrayOffset:function(a,b){
        # 获取当前矩阵的 elements 数组
        var c=this.elements;
        # 将矩阵的元素值依次放入数组中
        a[b]=c[0];a[b+1]=c[1];a[b+2]=c[2];a[b+3]=c[3];a[b+4]=c[4];
        a[b+5]=c[5];a[b+6]=c[6];a[b+7]=c[7];a[b+8]=c[8];
        # 返回数组
        return a
    },
    # 定义一个名为 getNormalMatrix 的方法，计算矩阵的逆转置矩阵
    getNormalMatrix:function(a){
        # 调用 getInverse 和 transpose 方法，计算逆转置矩阵
        this.getInverse(a).transpose();
        # 返回当前对象
        return this
    },
    # 定义一个名为 transposeIntoArray 的方法，将矩阵的转置矩阵元素值放入数组中
    transposeIntoArray:function(a){
        # 获取当前矩阵的 elements 数组
        var b=this.elements;
        # 将矩阵的转置矩阵元素值依次放入数组中
        a[0]=b[0];a[1]=b[3];a[2]=b[6];a[3]=b[1];a[4]=b[4];a[5]=b[7];a[6]=b[2];a[7]=b[5];a[8]=b[8];
        # 返回当前对象
        return this
    },
    # 定义一个名为 fromArray 的方法，将数组的值设置为矩阵的元素值
    fromArray:function(a){
        # 将数组的值设置为矩阵的 elements 数组
        this.elements.set(a);
        # 返回当前对象
        return this
    },
    # 定义一个名为 toArray 的方法，将矩阵的元素值放入数组中
    toArray:function(){
        # 获取当前矩阵的 elements 数组
        var a=this.elements;
        # 返回矩阵的元素值组成的数组
        return[a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8]]
    },
    # 定义一个名为 clone 的方法，复制当前矩阵并返回新的矩阵对象
    clone:function(){
        # 创建一个新的 Matrix3 对象，并将当前矩阵的元素值复制给新对象
        return(new THREE.Matrix3).fromArray(this.elements)
    }
};
# 定义 Matrix4 对象的原型方法
THREE.Matrix4.prototype={
    constructor:THREE.Matrix4,
    # 设置矩阵元素的值
    set:function(a,b,c,d,e,f,g,h,k,n,p,q,m,r,t,s){
        var u=this.elements;
        u[0]=a;
        u[4]=b;
        u[8]=c;
        u[12]=d;
        u[1]=e;
        u[5]=f;
        u[9]=g;
        u[13]=h;
        u[2]=k;
        u[6]=n;
        u[10]=p;
        u[14]=q;
        u[3]=m;
        u[7]=r;
        u[11]=t;
        u[15]=s;
        return this
    },
    # 将矩阵设置为单位矩阵
    identity:function(){
        this.set(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1);
        return this
    },
    # 复制另一个矩阵的元素值
    copy:function(a){
        this.elements.set(a.elements);
        return this
    },
    # 提取另一个矩阵的位置信息
    extractPosition:function(a){
        console.warn("THREE.Matrix4: .extractPosition() has been renamed to .copyPosition().");
        return this.copyPosition(a)
    },
    # 复制另一个矩阵的位置信息
    copyPosition:function(a){
        var b=this.elements;
        a=a.elements;
        b[12]=a[12];
        b[13]=a[13];
        b[14]=a[14];
        return this
    },
    # 提取另一个矩阵的旋转信息
    extractRotation:function(){
        var a=new THREE.Vector3;
        return function(b){
            var c=this.elements;
            b=b.elements;
            var d=1/a.set(b[0],b[1],b[2]).length(),
                e=1/a.set(b[4],b[5],b[6]).length(),
                f=1/a.set(b[8],b[9],b[10]).length();
            c[0]=b[0]*d;
            c[1]=b[1]*d;
            c[2]=b[2]*d;
            c[4]=b[4]*e;
            c[5]=b[5]*e;
            c[6]=b[6]*e;
            c[8]=b[8]*f;
            c[9]=b[9]*f;
            c[10]=b[10]*f;
            return this
        }
    }(),
    # 根据欧拉角创建旋转矩阵
    makeRotationFromEuler:function(a){
        !1===a instanceof THREE.Euler&&console.error("THREE.Matrix: .makeRotationFromEuler() now expects a Euler rotation rather than a Vector3 and order.");
        var b=this.elements,
            c=a.x,
            d=a.y,
            e=a.z,
            f=Math.cos(c),
            c=Math.sin(c),
            g=Math.cos(d),
            d=Math.sin(d),
            h=Math.cos(e),
            e=Math.sin(e);
        if("XYZ"===a.order){
            a=f*h;
            var k=f*e,
                n=c*h,
                p=c*e;
            b[0]=g*h;
            b[4]=-g*e;
            b[8]=d;
            b[1]=k+n*d;
            b[5]=a-p*d;
            b[9]=-c*g;
            b[2]=p-a*d;
            b[6]=n+k*d;
            b[10]=f*g
        }else if("YXZ"===a.order){
            a=g*h;
            k=g*e;
            n=d*h;
            p=d*e;
            b[0]=a+p*c;
            b[4]=n*c-k;
            b[8]=f*d;
            b[1]=f*e;
            b[5]=f*h;
            b[9]=-c;
            b[2]=k*c-n;
            b[6]=p+a*c;

... (此处省略部分代码)

            b[10]=f*g
        }
        return this
    }
}
# 设置矩阵的旋转部分，根据四元数计算旋转矩阵
makeRotationFromQuaternion:function(a){
    # 获取矩阵元素
    var b=this.elements,
        c=a.x,
        d=a.y,
        e=a.z,
        f=a.w,
        g=c+c,
        h=d+d,
        k=e+e;
    # 计算旋转矩阵的各个元素
    a=c*g;
    var n=c*h,
        c=c*k,
        p=d*h,
        d=d*k,
        e=e*k,
        g=f*g,
        h=f*h,
        f=f*k;
    # 设置旋转矩阵的各个元素
    b[0]=1-(p+e);
    b[4]=n-f;
    b[8]=c+h;
    b[1]=n+f;
    b[5]=1-(a+e);
    b[9]=d-g;
    b[2]=c-h;
    b[6]=d+g;
    b[10]=1-(a+p);
    b[3]=0;
    b[7]=0;
    b[11]=0;
    b[12]=0;
    b[13]=0;
    b[14]=0;
    b[15]=1;
    return this
},
# 设置矩阵的旋转部分，根据给定的目标点和方向向量
lookAt:function(){
    # 创建三个向量对象
    var a=new THREE.Vector3,
        b=new THREE.Vector3,
        c=new THREE.Vector3;
    return function(d,e,f){
        # 获取矩阵元素
        var g=this.elements;
        # 计算目标点和方向向量
        c.subVectors(d,e).normalize();
        # 如果方向向量长度为0，则设置默认值
        0===c.length()&&(c.z=1);
        a.crossVectors(f,c).normalize();
        # 如果方向向量长度为0，则设置默认值
        0===a.length()&&(c.x+=1E-4,a.crossVectors(f,c).normalize());
        b.crossVectors(c,a);
        # 设置矩阵的旋转部分
        g[0]=a.x;
        g[4]=b.x;
        g[8]=c.x;
        g[1]=a.y;
        g[5]=b.y;
        g[9]=c.y;
        g[2]=a.z;
        g[6]=b.z;
        g[10]=c.z;
        return this
    }
},
# 定义一个名为 multiply 的函数，接受两个参数 a 和 b
multiply:function(a,b){
    # 如果参数 b 存在，则输出警告信息并调用 multiplyMatrices 方法
    return void 0!==b?(console.warn("THREE.Matrix4: .multiply() now only accepts one argument. Use .multiplyMatrices( a, b ) instead."),this.multiplyMatrices(a,b)):
    # 否则调用 multiplyMatrices 方法，传入当前对象和参数 a
    this.multiplyMatrices(this,a)
},
# 定义一个名为 multiplyMatrices 的函数，接受两个参数 a 和 b
multiplyMatrices:function(a,b){
    # 分别获取参数 a 和 b 的元素
    var c=a.elements,
        d=b.elements,
        e=this.elements,
        f=c[0],
        g=c[4],
        h=c[8],
        k=c[12],
        n=c[1],
        p=c[5],
        q=c[9],
        m=c[13],
        r=c[2],
        t=c[6],
        s=c[10],
        u=c[14],
        v=c[3],
        y=c[7],
        G=c[11],
        c=c[15],
        w=d[0],
        K=d[4],
        x=d[8],
        D=d[12],
        E=d[1],
        A=d[5],
        B=d[9],
        F=d[13],
        R=d[2],
        H=d[6],
        C=d[10],
        T=d[14],
        Q=d[3],
        O=d[7],
        S=d[11],
        d=d[15];
    # 计算矩阵相乘的结果并存入当前对象的元素中
    e[0]=f*w+g*E+h*R+k*Q;
    e[4]=f*K+g*A+h*H+k*O;
    e[8]=f*x+g*B+h*C+k*S;
    e[12]=f*D+g*F+h*T+k*d;
    e[1]=n*w+p*E+q*R+m*Q;
    e[5]=n*K+p*A+q*H+m*O;
    e[9]=n*x+p*B+q*C+m*S;
    e[13]=n*D+p*F+q*T+m*d;
    e[2]=r*w+t*E+s*R+u*Q;
    e[6]=r*K+t*A+s*H+u*O;
    e[10]=r*x+t*B+s*C+u*S;
    e[14]=r*D+t*F+s*T+u*d;
    e[3]=v*w+y*E+G*R+c*Q;
    e[7]=v*K+y*A+G*H+c*O;
    e[11]=v*x+y*B+G*C+c*S;
    e[15]=v*D+y*F+G*T+c*d;
    # 返回当前对象
    return this
},
# 定义一个名为 multiplyToArray 的函数，接受三个参数 a, b, c
multiplyToArray:function(a,b,c){
    # 获取当前对象的元素
    var d=this.elements;
    # 调用 multiplyMatrices 方法，传入参数 a 和 b
    this.multiplyMatrices(a,b);
    # 将当前对象的元素复制到参数 c 中
    c[0]=d[0];
    c[1]=d[1];
    c[2]=d[2];
    c[3]=d[3];
    c[4]=d[4];
    c[5]=d[5];
    c[6]=d[6];
    c[7]=d[7];
    c[8]=d[8];
    c[9]=d[9];
    c[10]=d[10];
    c[11]=d[11];
    c[12]=d[12];
    c[13]=d[13];
    c[14]=d[14];
    c[15]=d[15];
    # 返回当前对象
    return this
},
# 定义一个名为 multiplyScalar 的函数，接受一个参数 a
multiplyScalar:function(a){
    # 获取当前对象的元素
    var b=this.elements;
    # 将当前对象的元素分别乘以参数 a
    b[0]*=a;
    b[4]*=a;
    b[8]*=a;
    b[12]*=a;
    b[1]*=a;
    b[5]*=a;
    b[9]*=a;
    b[13]*=a;
    b[2]*=a;
    b[6]*=a;
    b[10]*=a;
    b[14]*=a;
    b[3]*=a;
    b[7]*=a;
    b[11]*=a;
    b[15]*=a;
    # 返回当前对象
    return this
},
# 定义一个名为 multiplyVector3 的函数，接受一个参数 a
multiplyVector3:function(a){
    # 输出警告信息
    console.warn("THREE.Matrix4: .multiplyVector3() has been removed. Use vector.applyMatrix4( matrix ) or vector.applyProjection( matrix ) instead.");
# 返回一个新的矩阵，该矩阵是将当前矩阵应用于给定矩阵的投影
return a.applyProjection(this)
# 乘以一个四维向量，已被移除，使用 vector.applyMatrix4( matrix ) 代替
},multiplyVector4:function(a){
console.warn("THREE.Matrix4: .multiplyVector4() has been removed. Use vector.applyMatrix4( matrix ) instead.");
return a.applyMatrix4(this)
# 乘以一个三维向量数组，已被重命名，使用 matrix.applyToVector3Array( array ) 代替
},multiplyVector3Array:function(a){
console.warn("THREE.Matrix4: .multiplyVector3Array() has been renamed. Use matrix.applyToVector3Array( array ) instead.");
return this.applyToVector3Array(a)
# 将矩阵应用于三维向量数组
},applyToVector3Array:function(){
var a=new THREE.Vector3;
return function(b,c,d){
void 0===c&&(c=0);
void 0===d&&(d=b.length);
for(var e=0;e<d;e+=3,c+=3)a.x=b[c],a.y=b[c+1],a.z=b[c+2],a.applyMatrix4(this),b[c]=a.x,b[c+1]=a.y,b[c+2]=a.z;
return b
}}(),
# 旋转轴，已被移除，使用 Vector3.transformDirection( matrix ) 代替
rotateAxis:function(a){
console.warn("THREE.Matrix4: .rotateAxis() has been removed. Use Vector3.transformDirection( matrix ) instead.");
a.transformDirection(this)
# 叉乘向量，已被移除，使用 vector.applyMatrix4( matrix ) 代替
},crossVector:function(a){
console.warn("THREE.Matrix4: .crossVector() has been removed. Use vector.applyMatrix4( matrix ) instead.");
return a.applyMatrix4(this)
# 计算行列式
},determinant:function(){
var a=this.elements,b=a[0],c=a[4],d=a[8],e=a[12],f=a[1],g=a[5],h=a[9],k=a[13],n=a[2],p=a[6],q=a[10],m=a[14];
return a[3]*(+e*h*p-d*k*p-e*g*q+c*k*q+d*g*m-c*h*m)+a[7]*(+b*h*m-b*k*q+e*f*q-d*f*m+d*k*n-e*h*n)+a[11]*(+b*k*p-b*g*m-e*f*p+c*f*m+e*g*n-c*k*n)+a[15]*(-d*g*n-b*h*p+b*g*q+d*f*p-c*f*q+c*h*n)
# 转置矩阵
},transpose:function(){
var a=this.elements,b;
b=a[1];a[1]=a[4];a[4]=b;
b=a[2];a[2]=a[8];a[8]=b;
b=a[6];a[6]=a[9];a[9]=b;
b=a[3];a[3]=a[12];a[12]=b;
b=a[7];a[7]=a[13];a[13]=b;
b=a[11];a[11]=a[14];a[14]=b;
return this
# 将矩阵转换为数组
},flattenToArrayOffset:function(a,
// 将矩阵的元素复制到数组a中，从索引b开始
// 这里假设this.elements是一个包含16个元素的数组
b){
var c=this.elements;
a[b]=c[0];
a[b+1]=c[1];
a[b+2]=c[2];
a[b+3]=c[3];
a[b+4]=c[4];
a[b+5]=c[5];
a[b+6]=c[6];
a[b+7]=c[7];
a[b+8]=c[8];
a[b+9]=c[9];
a[b+10]=c[10];
a[b+11]=c[11];
a[b+12]=c[12];
a[b+13]=c[13];
a[b+14]=c[14];
a[b+15]=c[15];
return a
},
// 获取矩阵的位置信息，已被移除，使用Vector3.setFromMatrixPosition( matrix )代替
getPosition:function(){
var a=new THREE.Vector3;
return function(){
console.warn("THREE.Matrix4: .getPosition() has been removed. Use Vector3.setFromMatrixPosition( matrix ) instead.");
var b=this.elements;
return a.set(b[12],b[13],b[14])
}
}(),
// 设置矩阵的位置
setPosition:function(a){
var b=this.elements;
b[12]=a.x;
b[13]=a.y;
b[14]=a.z;
return this
},
// 获取矩阵的逆矩阵
getInverse:function(a,b){
var c=this.elements;
var d=a.elements;
// 计算逆矩阵的各个元素
c[0]=p*s*v-q*t*v+q*r*y-n*s*y-p*r*d+n*t*d;
c[4]=h*t*v-g*s*v-h*r*y+f*s*y+g*r*d-f*t*d;
c[8]=g*q*v-h*p*v+h*n*y-f*q*y-g*n*d+f*p*d;
c[12]=h*p*r-g*q*r-h*n*t+f*q*t+g*n*s-f*p*s;
c[1]=q*t*u-p*s*u-q*m*y+k*s*y+p*m*d-k*t*d;
c[5]=g*s*u-h*t*u+h*m*y-e*s*y-g*m*d+e*t*d;
c[9]=h*p*u-g*q*u-h*k*y+e*q*y+g*k*d-e*p*d;
c[13]=g*q*u-h*p*u+h*k*t-e*q*t-g*k*s+e*p*s;
c[2]=n*s*u-q*r*u+q*m*v-k*s*v-n*m*d+k*r*d;
c[6]=h*r*u-f*s*u-h*m*v+e*s*v+f*m*d-e*r*d;
c[10]=f*q*u-h*n*u+h*k*v-e*q*v-f*k*d+e*n*d;
c[14]=h*n*m-f*q*m-h*k*r+e*q*r+f*k*s-e*n*s;
c[3]=p*r*u-n*t*u-p*m*v+k*t*v+n*m*y-k*r*y;
c[7]=f*t*u-g*r*u+g*m*v-e*t*v-f*m*y+e*r*y;
c[11]=g*n*u-f*p*u-g*k*v+e*p*v+f*k*y-e*n*y;
c[15]=f*p*m-g*n*m+g*k*r-e*p*r-f*k*t+e*n*t;
c=e*c[0]+k*c[4]+m*c[8]+u*c[12];
// 如果行列式为0，则无法求逆矩阵
if(0==c){
if(b)throw Error("Matrix4.getInverse(): can't invert matrix, determinant is 0");
# 输出警告信息，表示无法求逆矩阵，行列式为0
console.warn("Matrix4.getInverse(): can't invert matrix, determinant is 0");
# 将矩阵重置为单位矩阵
this.identity();
# 返回重置后的单位矩阵
return this
# 将矩阵的每个元素乘以标量1/c
this.multiplyScalar(1/c);
# 返回乘以标量后的矩阵
return this
# 移动矩阵，但已被移除
translate:function(a){console.warn("THREE.Matrix4: .translate() has been removed.")}
# 绕X轴旋转矩阵，但已被移除
rotateX:function(a){console.warn("THREE.Matrix4: .rotateX() has been removed.")}
# 绕Y轴旋转矩阵，但已被移除
rotateY:function(a){console.warn("THREE.Matrix4: .rotateY() has been removed.")}
# 绕Z轴旋转矩阵，但已被移除
rotateZ:function(a){console.warn("THREE.Matrix4: .rotateZ() has been removed.")}
# 绕任意轴旋转矩阵，但已被移除
rotateByAxis:function(a,b){console.warn("THREE.Matrix4: .rotateByAxis() has been removed.")}
# 缩放矩阵
scale:function(a){
    var b=this.elements,c=a.x,d=a.y;a=a.z;
    b[0]*=c;b[4]*=d;b[8]*=a;
    b[1]*=c;b[5]*=d;b[9]*=a;
    b[2]*=c;b[6]*=d;b[10]*=a;
    b[3]*=c;b[7]*=d;b[11]*=a;
    return this
}
# 获取矩阵在各轴上的最大缩放值
getMaxScaleOnAxis:function(){
    var a=this.elements;
    return Math.sqrt(Math.max(a[0]*a[0]+a[1]*a[1]+a[2]*a[2],Math.max(a[4]*a[4]+a[5]*a[5]+a[6]*a[6],a[8]*a[8]+a[9]*a[9]+a[10]*a[10]))
}
# 创建平移矩阵
makeTranslation:function(a,b,c){
    this.set(1,0,0,a,0,1,0,b,0,0,1,c,0,0,0,1);
    return this
}
# 创建绕X轴旋转的矩阵
makeRotationX:function(a){
    var b=Math.cos(a);
    a=Math.sin(a);
    this.set(1,0,0,0,0,b,-a,0,0,a,b,0,0,0,0,1);
    return this
}
# 创建绕Y轴旋转的矩阵
makeRotationY:function(a){
    var b=Math.cos(a);
    a=Math.sin(a);
    this.set(b,0,a,0,0,1,0,0,-a,0,b,0,0,0,0,1);
    return this
}
# 创建绕Z轴旋转的矩阵
makeRotationZ:function(a){
    var b=Math.cos(a);
    a=Math.sin(a);
    this.set(b,-a,0,0,a,b,0,0,0,0,1,0,0,0,0,1);
    return this
}
# 创建绕任意轴旋转的矩阵
makeRotationAxis:function(a,b){
    var c=Math.cos(b),d=Math.sin(b),e=1-c,f=a.x,g=a.y,h=a.z,k=e*f,n=e*g;
    this.set(k*f+c,k*g-d*h,k*h+d*g,0,k*g+d*h,n*g+c,n*h-d*f,0,k*h-d*g,n*h+d*g,e*h*h+c,0,0,0,0,1);
    return this
}
# 创建缩放矩阵
makeScale:function(a,b,c){
    this.set(a,
# 创建一个名为THREE的对象
0,0,0,0,b,0,0,0,0,c,0,0,0,0,1);return this},compose:function(a,b,c){
# 创建一个compose方法，接受三个参数a、b、c
this.makeRotationFromQuaternion(b);
# 调用makeRotationFromQuaternion方法，传入参数b
this.scale(c);
# 调用scale方法，传入参数c
this.setPosition(a);
# 调用setPosition方法，传入参数a
return this},
# 返回当前对象
decompose:function(){
# 创建一个decompose方法
var a=new THREE.Vector3,b=new THREE.Matrix4;
# 创建一个新的Vector3对象a和Matrix4对象b
return function(c,d,e){
# 返回一个匿名函数，接受三个参数c、d、e
var f=this.elements,g=a.set(f[0],f[1],f[2]).length(),h=a.set(f[4],f[5],f[6]).length(),k=a.set(f[8],f[9],f[10]).length();
# 创建变量f、g、h、k，分别为当前对象的elements属性的子集的长度
0>this.determinant()&&(g=-g);
# 如果determinant方法返回值小于0，则将g取反
c.x=f[12];c.y=f[13];c.z=f[14];
# 将c的x、y、z属性分别赋值为f的第12、13、14个元素
b.elements.set(this.elements);
# 将b的elements属性设置为当前对象的elements属性
c=1/g;var f=1/h,n=1/k;
# 创建变量f、n，分别为1/g、1/h、1/k
b.elements[0]*=c;b.elements[1]*=
c;b.elements[2]*=c;b.elements[4]*=f;b.elements[5]*=f;b.elements[6]*=f;b.elements[8]*=n;b.elements[9]*=n;b.elements[10]*=n;
# 对b的elements属性进行一系列乘法操作
d.setFromRotationMatrix(b);
# 调用setFromRotationMatrix方法，传入参数b
e.x=g;e.y=h;e.z=k;
# 将e的x、y、z属性分别赋值为g、h、k
return this}}(),
# 返回一个匿名函数
makeFrustum:function(a,b,c,d,e,f){
# 创建一个makeFrustum方法，接受六个参数a、b、c、d、e、f
var g=this.elements;g[0]=2*e/(b-a);g[4]=0;g[8]=(b+a)/(b-a);g[12]=0;g[1]=0;g[5]=2*e/(d-c);g[9]=(d+c)/(d-c);g[13]=0;g[2]=0;g[6]=0;g[10]=-(f+e)/(f-e);g[14]=-2*f*e/(f-e);g[3]=0;g[7]=0;g[11]=-1;g[15]=0;
# 对当前对象的elements属性进行一系列赋值操作
return this},
# 返回当前对象
makePerspective:function(a,b,c,d){
# 创建一个makePerspective方法，接受四个参数a、b、c、d
a=c*Math.tan(THREE.Math.degToRad(.5*a));
# 对a进行一系列数学运算
var e=-a;return this.makeFrustum(e*b,a*b,e,a,c,d)},
# 对e进行赋值操作，然后调用makeFrustum方法，传入参数e*b、a*b、e、a、c、d
makeOrthographic:function(a,b,c,d,e,f){
# 创建一个makeOrthographic方法，接受六个参数a、b、c、d、e、f
var g=this.elements,h=b-a,k=c-d,n=f-e;
# 创建变量g、h、k、n，分别为b-a、c-d、f-e
g[0]=2/h;g[4]=0;g[8]=0;g[12]=-((b+a)/h);g[1]=0;g[5]=2/k;g[9]=0;g[13]=-((c+d)/k);g[2]=0;g[6]=0;g[10]=-2/n;g[14]=-((f+e)/n);g[3]=0;g[7]=0;g[11]=0;g[15]=1;
# 对当前对象的elements属性进行一系列赋值操作
return this},
# 返回当前对象
fromArray:function(a){
# 创建一个fromArray方法，接受一个参数a
this.elements.set(a);
# 将当前对象的elements属性设置为参数a
return this},
# 返回当前对象
toArray:function(){
# 创建一个toArray方法
var a=this.elements;
# 创建一个变量a，为当前对象的elements属性
return[a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10],a[11],a[12],a[13],a[14],a[15]]},
# 返回一个包含当前对象elements属性的数组
clone:function(){
# 创建一个clone方法
return(new THREE.Matrix4).fromArray(this.elements)};
# 返回一个新的Matrix4对象，调用fromArray方法，传入当前对象的elements属性
THREE.Ray=function(a,b){
# 创建一个名为Ray的方法，接受两个参数a、b
this.origin=void 0!==a?a:new THREE.Vector3;
# 创建一个origin属性，如果a不为undefined，则赋值为a，否则赋值为一个新的Vector3对象
this.direction=void 0!==b?b:new THREE.Vector3};
# 创建一个direction属性，如果b不为undefined，则赋值为b，否则赋值为一个新的Vector3对象
// 定义 Ray 对象的原型
THREE.Ray.prototype={
    // 构造函数，设置射线的起点和方向
    constructor:THREE.Ray,
    set:function(a,b){
        this.origin.copy(a);
        this.direction.copy(b);
        return this
    },
    // 复制另一个射线对象的起点和方向
    copy:function(a){
        this.origin.copy(a.origin);
        this.direction.copy(a.direction);
        return this
    },
    // 返回射线上距离起点一定距离的点的坐标
    at:function(a,b){
        return(b||new THREE.Vector3).copy(this.direction).multiplyScalar(a).add(this.origin)
    },
    // 重新计算射线的起点
    recast:function(){
        var a=new THREE.Vector3;
        return function(b){
            this.origin.copy(this.at(b,a));
            return this
        }
    }(),
    // 返回射线到给定点的最近点的坐标
    closestPointToPoint:function(a,b){
        var c=b||new THREE.Vector3;
        c.subVectors(a,this.origin);
        var d=c.dot(this.direction);
        return 0>d?c.copy(this.origin):c.copy(this.direction).multiplyScalar(d).add(this.origin)
    },
    // 返回射线到给定点的距离
    distanceToPoint:function(){
        var a=new THREE.Vector3;
        return function(b){
            var c=a.subVectors(b,this.origin).dot(this.direction);
            if(0>c){
                return this.origin.distanceTo(b)
            }
            a.copy(this.direction).multiplyScalar(c).add(this.origin);
            return a.distanceTo(b)
        }
    }(),
    // 返回射线到线段的最短距离的平方
    distanceSqToSegment:function(a,b,c,d){
        var e=a.clone().add(b).multiplyScalar(.5),
            f=b.clone().sub(a).normalize(),
            g=.5*a.distanceTo(b),
            h=this.origin.clone().sub(e);
        a=-this.direction.dot(f);
        b=h.dot(this.direction);
        var k=-h.dot(f),
            n=h.lengthSq(),
            p=Math.abs(1-a*a),
            q,m;
        // 计算距离
        // ...
    }
}
# 计算射线与球体的交点距离
isIntersectionSphere:function(a){
    # 计算射线与球体中心点的距离是否小于等于球体半径，判断是否相交
    return this.distanceToPoint(a.center)<=a.radius
},
intersectSphere:function(){
    var a=new THREE.Vector3;
    return function(b,c){
        # 计算射线与球体的交点
        a.subVectors(b.center,this.origin);
        var d=a.dot(this.direction),e=a.dot(a)-d*d,f=b.radius*b.radius;
        if(e>f) return null;
        f=Math.sqrt(f-e);
        e=d-f;
        d+=f;
        return 0>e&&0>d?null:0>e?this.at(d,c):this.at(e,c)
    }
}(),

# 判断射线是否与平面相交
isIntersectionPlane:function(a){
    var b=a.distanceToPoint(this.origin);
    return 0===b||0>a.normal.dot(this.direction)*b?!0:!1
},
distanceToPlane:function(a){
    var b=a.normal.dot(this.direction);
    if(0==b) return 0==a.distanceToPoint(this.origin)?0:null;
    a=-(this.origin.dot(a.normal)+a.constant)/b;
    return 0<=a?a:null
},
intersectPlane:function(a,b){
    # 计算射线与平面的交点
    var c=this.distanceToPlane(a);
    return null===c?null:this.at(c,b)
},

# 判断射线是否与立方体相交
isIntersectionBox:function(){
    var a=new THREE.Vector3;
    return function(b){
        # 判断射线是否与立方体相交
        return null!==this.intersectBox(b,a)
    }
}(),
intersectBox:function(a,b){
    var c,d,e,f,g;
    d=1/this.direction.x;
    f=1/this.direction.y;
    g=1/this.direction.z;
    var h=this.origin;
    0<=d?(c=(a.min.x-h.x)*d,d*=a.max.x-h.x):(c=(a.max.x-h.x)*d,d*=a.min.x-h.x);
    0<=f?(e=(a.min.y-h.y)*f,f*=a.max.y-h.y):(e=(a.max.y-h.y)*f,f*=a.min.y-h.y);
    if(c>f||e>d) return null;
    if(e>c||c!==c) c=e;
    if(f<d||d!==d) d=f;
    0<=g?(e=(a.min.z-h.z)*g,g*=a.max.z-h.z):(e=(a.max.z-h.z)*g,g*=a.min.z-h.z);
    if(c>g||e>d) return null;
    if(e>c||c!==
# 定义一个函数，用于计算射线与三角形的交点
intersectTriangle:function(){
    var a=new THREE.Vector3,  # 创建一个三维向量a
        b=new THREE.Vector3,  # 创建一个三维向量b
        c=new THREE.Vector3,  # 创建一个三维向量c
        d=new THREE.Vector3;  # 创建一个三维向量d
    return function(e,f,g,h,k){  # 返回一个函数，接受5个参数
        b.subVectors(f,e);  # 计算向量b，f-e
        c.subVectors(g,e);  # 计算向量c，g-e
        d.crossVectors(b,c);  # 计算向量d，b和c的叉乘
        f=this.direction.dot(d);  # 计算射线方向与d的点积
        if(0<f){  # 如果点积大于0
            if(h)  # 如果h存在
                return null;  # 返回空
            h=1  # 否则，h赋值为1
        }else if(0>f)  # 如果点积小于0
            h=-1,f=-f;  # h赋值为-1，f取绝对值
        else 
            return null;  # 否则返回空
        a.subVectors(this.origin,e);  # 计算向量a，this.origin-e
        e=h*this.direction.dot(c.crossVectors(a,c));  # 计算e
        if(0>e)  # 如果e小于0
            return null;  # 返回空
        g=h*this.direction.dot(b.cross(a));  # 计算g
        if(0>g||e+g>f)  # 如果g小于0或者e+g大于f
            return null;  # 返回空
        e=-h*a.dot(d);  # 计算e
        return 0>e?null:this.at(e/f,k)  # 如果e小于0，返回空，否则返回射线上的点
    }
}(),
applyMatrix4:function(a){  # 定义一个函数，用于将射线应用于4x4矩阵
    this.direction.add(this.origin).applyMatrix4(a);  # 将射线方向加上原点，并应用矩阵a
    this.origin.applyMatrix4(a);  # 将射线原点应用矩阵a
    this.direction.sub(this.origin);  # 将射线方向减去原点
    this.direction.normalize();  # 将射线方向归一化
    return this  # 返回射线对象
},
equals:function(a){  # 定义一个函数，用于判断两个射线是否相等
    return a.origin.equals(this.origin)&&a.direction.equals(this.direction)  # 返回判断结果
},
clone:function(){  # 定义一个函数，用于克隆射线对象
    return(new THREE.Ray).copy(this)  # 返回一个新的射线对象，复制当前射线对象的属性
}};
THREE.Sphere=function(a,b){  # 定义一个构造函数，用于创建一个球体对象
    this.center=void 0!==a?a:new THREE.Vector3;  # 如果a存在，将a赋值给center，否则创建一个新的三维向量
    this.radius=void 0!==b?b:0;  # 如果b存在，将b赋值给radius，否则赋值为0
};
THREE.Sphere.prototype={  # 定义一个原型对象
    constructor:THREE.Sphere,  # 构造函数指向THREE.Sphere
    set:function(a,b){  # 定义一个函数，用于设置球体的属性
        this.center.copy(a);  # 将a的值复制给center
        this.radius=b;  # 将b的值赋给radius
        return this  # 返回球体对象
    },
    setFromPoints:function(){  # 定义一个函数，用于根据点集设置球体的属性
        var a=new THREE.Box3;  # 创建一个Box3对象
        return function(b,c){  # 返回一个函数，接受两个参数
            var d=this.center;  # 创建一个指向center的引用
            void 0!==c?d.copy(c):a.setFromPoints(b).center(d);  # 如果c存在，将c的值复制给d，否则根据点集b设置Box3的属性，并将中心点赋给d
            for(var e=0,f=0,g=b.length;f<g;f++)  # 遍历点集
                e=Math.max(e,d.distanceToSquared(b[f]));  # 计算最大距离的平方
            this.radius=Math.sqrt(e);  # 将最大距离的平方根赋给radius
            return this  # 返回球体对象
        }
    }(),
    copy:function(a){  # 定义一个函数，用于复制球体对象
        this.center.copy(a.center);  # 复制a的center属性给center
        this.radius=a.radius;  # 复制a的radius属性给radius
        return this  # 返回球体对象
    },
    empty:function(){  # 定义一个函数，用于判断球体是否为空
        return 0>=this.radius  # 返回判断结果
    },
    containsPoint:function(a){  # 定义一个函数，用于判断点是否在球体内
        return a.distanceToSquared(this.center)<=  # 返回判断结果
# 定义一个名为 Sphere 的对象
this.radius*this.radius},
# 计算点到球心的距离
distanceToPoint:function(a){return a.distanceTo(this.center)-this.radius},
# 判断球体是否与另一个球体相交
intersectsSphere:function(a){var b=this.radius+a.radius;return a.center.distanceToSquared(this.center)<=b*b},
# 将点限制在球体表面
clampPoint:function(a,b){var c=this.center.distanceToSquared(a),d=b||new THREE.Vector3;d.copy(a);c>this.radius*this.radius&&(d.sub(this.center).normalize(),d.multiplyScalar(this.radius).add(this.center));return d},
# 获取包围盒
getBoundingBox:function(a){a=a||new THREE.Box3;a.set(this.center,this.center);a.expandByScalar(this.radius);return a},
# 对球体应用矩阵变换
applyMatrix4:function(a){this.center.applyMatrix4(a);this.radius*=a.getMaxScaleOnAxis();return this},
# 平移球体
translate:function(a){this.center.add(a);return this},
# 判断球体是否相等
equals:function(a){return a.center.equals(this.center)&&a.radius===this.radius},
# 克隆球体对象
clone:function(){return(new THREE.Sphere).copy(this)};
# 定义一个名为 Frustum 的对象
THREE.Frustum=function(a,b,c,d,e,f){this.planes=[void 0!==a?a:new THREE.Plane,void 0!==b?b:new THREE.Plane,void 0!==c?c:new THREE.Plane,void 0!==d?d:new THREE.Plane,void 0!==e?e:new THREE.Plane,void 0!==f?f:new THREE.Plane]};
# Frustum 对象的方法
THREE.Frustum.prototype={constructor:THREE.Frustum,
# 设置 Frustum 对象的六个平面
set:function(a,b,c,d,e,f){var g=this.planes;g[0].copy(a);g[1].copy(b);g[2].copy(c);g[3].copy(d);g[4].copy(e);g[5].copy(f);return this},
# 复制另一个 Frustum 对象
copy:function(a){for(var b=this.planes,c=0;6>c;c++)b[c].copy(a.planes[c]);return this},
# 从矩阵中设置 Frustum 对象的六个平面
setFromMatrix:function(a){var b=this.planes,c=a.elements;a=c[0];var d=c[1],e=c[2],f=c[3],g=c[4],h=c[5],k=c[6],n=c[7],p=c[8],q=c[9],m=c[10],r=c[11],t=c[12],s=c[13],u=c[14],c=c[15];b[0].setComponents(f-a,n-g,r-p,c-t).normalize();b[1].setComponents(f+
# 创建一个平面对象，包含法向量和常数
THREE.Plane=function(a,b){
    this.normal=void 0!==a?a:new THREE.Vector3(1,0,0);
    this.constant=void 0!==b?b:0;
};

# 平面对象的原型方法
THREE.Plane.prototype={
    constructor:THREE.Plane,
    # 设置平面的法向量和常数
    set:function(a,b){
        this.normal.copy(a);
        this.constant=b;
        return this
    },
    # 设置平面的法向量和常数
    setComponents:function(a,b,c,d){
        this.normal.set(a,b,c);
        this.constant=d;
        return this
    },
    # 根据法向量和共面点设置平面
    setFromNormalAndCoplanarPoint:function(a,b){
        this.normal.copy(a);
        this.constant=-b.dot(this.normal);
        return this
    },
    # 根据三个共面点设置平面
    setFromCoplanarPoints:function(){
        var a=new THREE.Vector3,
            b=new THREE.Vector3;
        return function(c,d,e){
            d=a.subVectors(e,d).cross(b.subVectors(c,d)).normalize();
            this.setFromNormalAndCoplanarPoint(d,
    }
};
{
    // 复制平面对象的属性到当前对象
    copy:function(a){
        this.normal.copy(a.normal);
        this.constant=a.constant;
        return this
    },
    // 将平面法向量归一化
    normalize:function(){
        var a=1/this.normal.length();
        this.normal.multiplyScalar(a);
        this.constant*=a;
        return this
    },
    // 反转平面的法向量和常数
    negate:function(){
        this.constant*=-1;
        this.normal.negate();
        return this
    },
    // 计算平面到给定点的距离
    distanceToPoint:function(a){
        return this.normal.dot(a)+this.constant
    },
    // 计算平面到给定球体的距离
    distanceToSphere:function(a){
        return this.distanceToPoint(a.center)-a.radius
    },
    // 计算给定点在平面上的投影点
    projectPoint:function(a,b){
        return this.orthoPoint(a,b).sub(a).negate()
    },
    // 计算给定点在平面上的正交投影点
    orthoPoint:function(a,b){
        var c=this.distanceToPoint(a);
        return(b||new THREE.Vector3).copy(this.normal).multiplyScalar(c)
    },
    // 判断平面是否与给定线段相交
    isIntersectionLine:function(a){
        var b=this.distanceToPoint(a.start);
        a=this.distanceToPoint(a.end);
        return 0>b&&0<a||0>a&&0<b
    },
    // 计算平面与给定线段的交点
    intersectLine:function(){
        var a=new THREE.Vector3;
        return function(b,c){
            var d=c||new THREE.Vector3,e=b.delta(a),f=this.normal.dot(e);
            if(0==f){
                if(0==this.distanceToPoint(b.start)) return d.copy(b.start)
            }else{
                f=-(b.start.dot(this.normal)+this.constant)/f;
                if(0>f||1<f) return void 0;
                else return d.copy(e).multiplyScalar(f).add(b.start)
            }
        }
    },
    // 计算平面上的共面点
    coplanarPoint:function(a){
        return(a||new THREE.Vector3).copy(this.normal).multiplyScalar(-this.constant)
    },
    // 将平面应用到给定矩阵上
    applyMatrix4:function(){
        var a=new THREE.Vector3,b=new THREE.Vector3,c=new THREE.Matrix3;
        return function(d,e){
            var f=e||c.getNormalMatrix(d),
            f=a.copy(this.normal).applyMatrix3(f),
            g=this.coplanarPoint(b);
            g.applyMatrix4(d);
            this.setFromNormalAndCoplanarPoint(f,g);
            return this
        }
    },
    // 平移平面
    translate:function(a){
        this.constant-=a.dot(this.normal);
        return this
    },
    // 判断平面是否与另一个平面相等
    equals:function(a){
        return a.normal.equals(this.normal)&&a.constant==this.constant
    },
    // 克隆当前平面对象
    clone:function(){
        return(new THREE.Plane).copy(this)
    }
}
# 创建一个名为 THREE 的对象，包含 Math 和 Spline 两个属性
THREE.Math={
    # 生成一个 UUID，用于标识对象
    generateUUID:function(){
        # 定义字符集合
        var a="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz".split(""),
        # 创建一个长度为 36 的数组
        b=Array(36),
        # 初始化计数器
        c=0,
        # 定义变量
        d;
        # 返回一个函数
        return function(){
            # 遍历数组
            for(var e=0;36>e;e++)
                # 设置特定位置的字符
                8==e||13==e||18==e||23==e?b[e]="-":14==e?b[e]="4":(2>=c&&(c=33554432+16777216*Math.random()|0),d=c&15,c>>=4,b[e]=a[19==e?d&3|8:d]);
            # 返回拼接后的字符串
            return b.join("")
        }
    }(),
    # 将值限制在指定范围内
    clamp:function(a,b,c){
        return a<b?b:a>c?c:a
    },
    # 将值限制在指定范围内，但不小于指定值
    clampBottom:function(a,b){
        return a<b?b:a
    },
    # 线性映射
    mapLinear:function(a,b,c,d,e){
        return d+(a-b)*(e-d)/(c-b)
    },
    # 平滑步进函数
    smoothstep:function(a,b,c){
        if(a<=b)
            return 0;
        if(a>=c)
            return 1;
        a=(a-b)/(c-b);
        return a*a*(3-2*a)
    },
    # 更平滑的步进函数
    smootherstep:function(a,b,c){
        if(a<=b)
            return 0;
        if(a>=c)
            return 1;
        a=(a-b)/(c-b);
        return a*a*a*(a*(6*a-15)+10)
    },
    # 生成一个 16 位的随机数
    random16:function(){
        return(65280*Math.random()+255*Math.random())/65535
    },
    # 生成一个指定范围内的随机整数
    randInt:function(a,b){
        return a+Math.floor(Math.random()*(b-a+1))
    },
    # 生成一个指定范围内的随机浮点数
    randFloat:function(a,b){
        return a+Math.random()*(b-a)
    },
    # 生成一个指定范围内的随机浮点数，以 0 为中心
    randFloatSpread:function(a){
        return a*(.5-Math.random())
    },
    # 将角度转换为弧度
    degToRad:function(){
        var a=Math.PI/180;
        return function(b){
            return b*a
        }
    }(),
    # 将弧度转换为角度
    radToDeg:function(){
        var a=180/Math.PI;
        return function(b){
            return b*a
        }
    }(),
    # 判断一个数是否是 2 的幂
    isPowerOfTwo:function(a){
        return 0===(a&a-1)&&0!==a
    }
};
# 创建一个名为 THREE 的对象，包含 Math 和 Spline 两个属性
THREE.Spline=function(a){
    # 定义函数
    function b(a,b,c,d,e,f,g){
        a=.5*(c-a);
        d=.5*(d-b);
        return(2*(b-c)+a+d)*g+(-3*(b-c)-2*a-d)*f+a*e+b
    }
    # 初始化点集
    this.points=a;
    var c=[],
    d={x:0,y:0,z:0},
    e,f,g,h,k,n,p,q,m;
    # 从数组初始化点集
    this.initFromArray=function(a){
        this.points=[];
        for(var b=0;b<a.length;b++)
            this.points[b]={x:a[b][0],y:a[b][1],z:a[b][2]}
    };
    # 获取曲线上的点
    this.getPoint=function(a){
        e=(this.points.length-1)*a;
        f=Math.floor(e);
        g=e-f;
        c[0]=0===f?f:f-1;
        c[1]=f;
        c[2]=f>this.points.length-2?this.points.length-1:f+1;
        c[3]=f>this.points.length-3?this.points.length-1:f+
    }
# 获取四个控制点的坐标
n=this.points[c[0]]
p=this.points[c[1]]
q=this.points[c[2]]
m=this.points[c[3]]
# 计算 g*g 和 g*h
h=g*g
k=g*h
# 计算三维空间中的坐标值
d.x=b(n.x,p.x,q.x,m.x,g,h,k)
d.y=b(n.y,p.y,q.y,m.y,g,h,k)
d.z=b(n.z,p.z,q.z,m.z,g,h,k)
# 返回计算结果
return d
# 获取控制点数组
this.getControlPointsArray=function(){
    var a,b,c=this.points.length,d=[];
    for(a=0;a<c;a++)
        b=this.points[a],
        d[a]=[b.x,b.y,b.z];
    return d
}
# 获取曲线长度
this.getLength=function(a){
    var b,c,d,e=b=b=0,f=new THREE.Vector3,g=new THREE.Vector3,h=[],k=0;
    h[0]=0;
    a||(a=100);
    c=this.points.length*a;
    f.copy(this.points[0]);
    for(a=1;a<c;a++)
        b=a/c,
        d=this.getPoint(b),
        g.copy(d),
        k+=g.distanceTo(f),
        f.copy(d),
        b*=this.points.length-1,
        b=Math.floor(b),
        b!=e&&(h[b]=k,e=b);
    h[h.length]=k;
    return{chunks:h,total:k}
}
# 通过弧长重新参数化
this.reparametrizeByArcLength=function(a){
    var b,c,d,e,f,g,h=[],k=new THREE.Vector3,m=this.getLength();
    h.push(k.copy(this.points[0]).clone());
    for(b=1;b<this.points.length;b++){
        c=m.chunks[b]-m.chunks[b-1];
        g=Math.ceil(a*c/m.total);
        e=(b-1)/(this.points.length-1);
        f=b/(this.points.length-1);
        for(c=1;c<g-1;c++)
            d=e+1/g*c*(f-e),
            d=this.getPoint(d),
            h.push(k.copy(d).clone());
        h.push(k.copy(this.points[b]).clone())
    }
    this.points=h
}
# 三角形类
THREE.Triangle=function(a,b,c){
    this.a=void 0!==a?a:new THREE.Vector3;
    this.b=void 0!==b?b:new THREE.Vector3;
    this.c=void 0!==c?c:new THREE.Vector3
}
# 计算三角形法向量
THREE.Triangle.normal=function(){
    var a=new THREE.Vector3;
    return function(b,c,d,e){
        e=e||new THREE.Vector3;
        e.subVectors(d,c);
        a.subVectors(b,c);
        e.cross(a);
        b=e.lengthSq();
        return 0<b?e.multiplyScalar(1/Math.sqrt(b)):e.set(0,0,0)
    }
}
# 从点获取重心坐标
THREE.Triangle.barycoordFromPoint=function(){
    var a=new THREE.Vector3,b=new THREE.Vector3,c=new THREE.Vector3;
    return function(d,e,f,g,h){
        a.subVectors(g,e);
        b.subVectors(f,e);
        c.subVectors(d,e);
        d=a.dot(a);
        e=a.dot(b);
        f=a.dot(c);
        k=b.dot(b);
        g=b.dot(c);
        n=d*k-e*e;
        h=h||new THREE.Vector3;
        if(0==n)
            return h.set(-2,-1,-1);
        n=1/n;
        k=(k*f-e*g)*n;
        d=(d*g-e*f)*n;
        return h.set(1-k-d,d,k)
    }
}
// 定义 Triangle 对象的 containsPoint 方法
THREE.Triangle.containsPoint=function(){
    // 创建一个三维向量对象
    var a=new THREE.Vector3;
    // 返回一个函数，用于判断点是否在三角形内部
    return function(b,c,d,e){
        // 调用 Triangle 对象的 barycoordFromPoint 方法，将结果保存在向量 a 中
        b=THREE.Triangle.barycoordFromPoint(b,c,d,e,a);
        // 判断点是否在三角形内部
        return 0<=b.x&&0<=b.y&&1>=b.x+b.y
    }
}();

// 定义 Triangle 对象的原型
THREE.Triangle.prototype={
    constructor:THREE.Triangle,
    // 设置三角形的顶点
    set:function(a,b,c){
        this.a.copy(a);
        this.b.copy(b);
        this.c.copy(c);
        return this
    },
    // 从点集和索引设置三角形的顶点
    setFromPointsAndIndices:function(a,b,c,d){
        this.a.copy(a[b]);
        this.b.copy(a[c]);
        this.c.copy(a[d]);
        return this
    },
    // 复制另一个三角形对象
    copy:function(a){
        this.a.copy(a.a);
        this.b.copy(a.b);
        this.c.copy(a.c);
        return this
    },
    // 计算三角形的面积
    area:function(){
        var a=new THREE.Vector3,b=new THREE.Vector3;
        return function(){
            a.subVectors(this.c,this.b);
            b.subVectors(this.a,this.b);
            return.5*a.cross(b).length()
        }
    }(),
    // 计算三角形的中点
    midpoint:function(a){
        return(a||new THREE.Vector3).addVectors(this.a,this.b).add(this.c).multiplyScalar(1/3)
    },
    // 计算三角形的法向量
    normal:function(a){
        return THREE.Triangle.normal(this.a,this.b,this.c,a)
    },
    // 计算三角形的平面
    plane:function(a){
        return(a||new THREE.Plane).setFromCoplanarPoints(this.a,this.b,this.c)
    },
    // 计算点相对于三角形的重心坐标
    barycoordFromPoint:function(a,b){
        return THREE.Triangle.barycoordFromPoint(a,this.a,this.b,this.c,b)
    },
    // 判断点是否在三角形内部
    containsPoint:function(a){
        return THREE.Triangle.containsPoint(a,this.a,this.b,this.c)
    },
    // 判断两个三角形是否相等
    equals:function(a){
        return a.a.equals(this.a)&&a.b.equals(this.b)&&a.c.equals(this.c)
    },
    // 克隆三角形对象
    clone:function(){
        return(new THREE.Triangle).copy(this)
    }
};

// 定义 Clock 对象
THREE.Clock=function(a){
    this.autoStart=void 0!==a?a:!0;
    this.elapsedTime=this.oldTime=this.startTime=0;
    this.running=!1;
};

// 定义 Clock 对象的原型
THREE.Clock.prototype={
    constructor:THREE.Clock,
    // 启动计时
    start:function(){
        this.oldTime=this.startTime=void 0!==self.performance&&void 0!==self.performance.now?self.performance.now():Date.now();
        this.running=!0
    },
    // 停止计时
    stop:function(){
        this.getElapsedTime();
        this.running=!1
    },
    // 获取经过的时间
    getElapsedTime:function(){
        this.getDelta();
        return this.elapsedTime
    },
    // 获取时间间隔
    getDelta:function(){
        var a=0;
        this.autoStart&&!this.running&&this.start();
        if(this.running){
            var b=void 0!==self.performance&&void 0!==self.performance.now?self.performance.now():Date.now();
            // 计算时间间隔
            a=(b-this.oldTime)/1E3;
            this.oldTime=b;
            this.elapsedTime+=a
        }
        return a
    }
};
# 定义一个函数，计算时间间隔
a=.001*(b-this.oldTime);this.oldTime=b;this.elapsedTime+=a}return a}};THREE.EventDispatcher=function(){};
# 定义一个事件分发器的原型对象
THREE.EventDispatcher.prototype={constructor:THREE.EventDispatcher,apply:function(a){a.addEventListener=THREE.EventDispatcher.prototype.addEventListener;a.hasEventListener=THREE.EventDispatcher.prototype.hasEventListener;a.removeEventListener=THREE.EventDispatcher.prototype.removeEventListener;a.dispatchEvent=THREE.EventDispatcher.prototype.dispatchEvent},addEventListener:function(a,b){void 0===this._listeners&&(this._listeners={});var c=this._listeners;void 0===c[a]&&(c[a]=[]);-1===c[a].indexOf(b)&&
c[a].push(b)},hasEventListener:function(a,b){if(void 0===this._listeners)return!1;var c=this._listeners;return void 0!==c[a]&&-1!==c[a].indexOf(b)?!0:!1},removeEventListener:function(a,b){if(void 0!==this._listeners){var c=this._listeners[a];if(void 0!==c){var d=c.indexOf(b);-1!==d&&c.splice(d,1)}}},dispatchEvent:function(a){if(void 0!==this._listeners){var b=this._listeners[a.type];if(void 0!==b){a.target=this;for(var c=[],d=b.length,e=0;e<d;e++)c[e]=b[e];for(e=0;e<d;e++)c[e].call(this,a)}}}};
# 定义一个立即执行函数，传入参数a
(function(a){a.Raycaster=function(b,c,f,g){this.ray=new a.Ray(b,c);this.near=f||0;this.far=g||Infinity;this.params={Sprite:{},Mesh:{},PointCloud:{threshold:1},LOD:{},Line:{}}};var b=function(a,b){return a.distance-b.distance},c=function(a,b,f,g){a.raycast(b,f);if(!0===g){a=a.children;g=0;for(var h=a.length;g<h;g++)c(a[g],b,f,!0)}};a.Raycaster.prototype={constructor:a.Raycaster,precision:1E-4,linePrecision:1,set:function(a,b){this.ray.set(a,b)},intersectObject:function(a,e){var f=[];c(a,this,f,e);
f.sort(b);return f},intersectObjects:function(a,e){var f=[];if(!1===a instanceof Array)return console.log("THREE.Raycaster.intersectObjects: objects is not an Array."),f;for(var g=0,h=a.length;g<h;g++)c(a[g],this,f,e);f.sort(b);return f}}})(THREE);
# 定义一个名为 Object3D 的构造函数
THREE.Object3D=function(){
    # 设置 id 属性为全局变量 Object3DIdCount 的值，并递增
    Object.defineProperty(this,"id",{value:THREE.Object3DIdCount++});
    # 生成一个唯一的 uuid
    this.uuid=THREE.Math.generateUUID();
    # 设置 name 属性为空字符串
    this.name="";
    # 设置 type 属性为 "Object3D"
    this.type="Object3D";
    # 设置 parent 属性为 undefined
    this.parent=void 0;
    # 初始化 children 属性为空数组
    this.children=[];
    # 设置 up 属性为默认的向上方向
    this.up=THREE.Object3D.DefaultUp.clone();
    # 创建 Vector3 对象 a, b, d 和 Quaternion 对象 c
    var a=new THREE.Vector3,b=new THREE.Euler,c=new THREE.Quaternion,d=new THREE.Vector3(1,1,1);
    # 当 Euler 对象 b 变化时，更新 Quaternion 对象 c
    b.onChange(function(){c.setFromEuler(b,!1)});
    # 当 Quaternion 对象 c 变化时，更新 Euler 对象 b
    c.onChange(function(){b.setFromQuaternion(c,void 0,!1)});
    # 设置 position, rotation, quaternion, scale 属性为对应的 Vector3 或 Quaternion 对象
    Object.defineProperties(this,{position:{enumerable:!0,value:a},rotation:{enumerable:!0,value:b},quaternion:{enumerable:!0,value:c},scale:{enumerable:!0,value:d}});
    # 初始化 renderDepth 为 null
    this.renderDepth=null;
    # 设置 rotationAutoUpdate 为 true
    this.rotationAutoUpdate=!0;
    # 创建 Matrix4 对象并赋给 matrix 和 matrixWorld 属性
    this.matrix=new THREE.Matrix4;
    this.matrixWorld=new THREE.Matrix4;
    # 设置 matrixAutoUpdate 为 true，matrixWorldNeedsUpdate 为 false
    this.matrixAutoUpdate=!0;
    this.matrixWorldNeedsUpdate=!1;
    # 设置 visible 为 true，receiveShadow 和 castShadow 为 false
    this.visible=!0;
    this.receiveShadow=this.castShadow=!1;
    # 设置 frustumCulled 为 true
    this.frustumCulled=!0;
    # 初始化 userData 为空对象
    this.userData={};
};
# 设置 Object3D 的默认向上方向
THREE.Object3D.DefaultUp=new THREE.Vector3(0,1,0);
# 设置 Object3D 的原型方法
THREE.Object3D.prototype={
    constructor:THREE.Object3D,
    # 获取 eulerOrder 属性
    get eulerOrder(){
        console.warn("THREE.Object3D: .eulerOrder has been moved to .rotation.order.");
        return this.rotation.order
    },
    # 设置 eulerOrder 属性
    set eulerOrder(a){
        console.warn("THREE.Object3D: .eulerOrder has been moved to .rotation.order.");
        this.rotation.order=a
    },
    # 获取 useQuaternion 属性
    get useQuaternion(){
        console.warn("THREE.Object3D: .useQuaternion has been removed. The library now uses quaternions by default.")
    },
    # 设置 useQuaternion 属性
    set useQuaternion(a){
        console.warn("THREE.Object3D: .useQuaternion has been removed. The library now uses quaternions by default.")
    },
    # 应用矩阵变换
    applyMatrix:function(a){
        this.matrix.multiplyMatrices(a,this.matrix);
        this.matrix.decompose(this.position,this.quaternion,this.scale)
    },
    # 根据轴和角度设置旋转
    setRotationFromAxisAngle:function(a,b){
        this.quaternion.setFromAxisAngle(a,b)
    },
    # 根据欧拉角设置旋转
    setRotationFromEuler:function(a){
        this.quaternion.setFromEuler(a,!0)
    },
    # 根据矩阵设置旋转
    setRotationFromMatrix:function(a){
        this.quaternion.setFromRotationMatrix(a)
    },
    # 根据四元数设置旋转
    setRotationFromQuaternion:function(a){
        this.quaternion.copy(a)
    },
    # 绕轴旋转
    rotateOnAxis:function(){
        var a=new THREE.Quaternion;
        return function(b,c){
            a.setFromAxisAngle(b,
// 以 X 轴为旋转轴，对对象进行旋转
rotateX:function(){
    var a=new THREE.Vector3(1,0,0);
    return function(b){
        return this.rotateOnAxis(a,b)
    }
}(),

// 以 Y 轴为旋转轴，对对象进行旋转
rotateY:function(){
    var a=new THREE.Vector3(0,1,0);
    return function(b){
        return this.rotateOnAxis(a,b)
    }
}(),

// 以 Z 轴为旋转轴，对对象进行旋转
rotateZ:function(){
    var a=new THREE.Vector3(0,0,1);
    return function(b){
        return this.rotateOnAxis(a,b)
    }
}(),

// 沿着指定轴进行平移
translateOnAxis:function(){
    var a=new THREE.Vector3;
    return function(b,c){
        a.copy(b).applyQuaternion(this.quaternion);
        this.position.add(a.multiplyScalar(c));
        return this
    }
}(),

// 对对象进行平移
translate:function(a,b){
    console.warn("THREE.Object3D: .translate() has been removed. Use .translateOnAxis( axis, distance ) instead.");
    return this.translateOnAxis(b,a)
},

// 沿 X 轴进行平移
translateX:function(){
    var a=new THREE.Vector3(1,0,0);
    return function(b){
        return this.translateOnAxis(a,b)
    }
}(),

// 沿 Y 轴进行平移
translateY:function(){
    var a=new THREE.Vector3(0,1,0);
    return function(b){
        return this.translateOnAxis(a,b)
    }
}(),

// 沿 Z 轴进行平移
translateZ:function(){
    var a=new THREE.Vector3(0,0,1);
    return function(b){
        return this.translateOnAxis(a,b)
    }
}(),

// 将向量从本地坐标系转换到世界坐标系
localToWorld:function(a){
    return a.applyMatrix4(this.matrixWorld)
},

// 将向量从世界坐标系转换到本地坐标系
worldToLocal:function(){
    var a=new THREE.Matrix4;
    return function(b){
        return b.applyMatrix4(a.getInverse(this.matrixWorld))
    }
}(),

// 使对象朝向指定的位置
lookAt:function(){
    var a=new THREE.Matrix4;
    return function(b){
        a.lookAt(b,this.position,this.up);
        this.quaternion.setFromRotationMatrix(a)
    }
}(),

// 向对象添加子对象
add:function(a){
    if(1<arguments.length){
        for(var b=0;b<arguments.length;b++)
            this.add(arguments[b]);
        return this
    }
    if(a===this)
        return console.error("THREE.Object3D.add:",
# 将对象添加为当前对象的子对象
add: function(a) {
    # 如果 a 是 THREE.Object3D 的实例
    if (a instanceof THREE.Object3D) {
        # 如果 a 已经有父对象，则先将其从原父对象中移除
        if (void 0 !== a.parent && a.parent.remove(a)) {
            # 将当前对象设置为 a 的父对象
            a.parent = this;
            # 分发 "added" 事件
            a.dispatchEvent({type: "added"});
            # 将 a 添加到当前对象的子对象列表中
            this.children.push(a);
        } else {
            # 如果 a 没有父对象，则输出错误信息
            console.error("THREE.Object3D.add:", a, "is not an instance of THREE.Object3D.");
        }
    }
    # 返回当前对象
    return this;
},
# 从当前对象的子对象列表中移除指定的子对象
remove: function(a) {
    # 如果传入了多个参数，则依次移除每个子对象
    if (1 < arguments.length) {
        for (var b = 0; b < arguments.length; b++) {
            this.remove(arguments[b]);
        }
    }
    # 获取要移除的子对象在子对象列表中的索引
    b = this.children.indexOf(a);
    # 如果找到了要移除的子对象
    if (-1 !== b) {
        # 将子对象的父对象设置为 undefined
        a.parent = void 0;
        # 分发 "removed" 事件
        a.dispatchEvent({type: "removed"});
        # 从子对象列表中移除该子对象
        this.children.splice(b, 1);
    }
},
# 根据名称获取子对象（已废弃，使用 getObjectByName 替代）
getChildByName: function(a, b) {
    console.warn("THREE.Object3D: .getChildByName() has been renamed to .getObjectByName().");
    return this.getObjectByName(a, b);
},
# 根据 ID 获取子对象
getObjectById: function(a, b) {
    # 如果当前对象的 ID 与要查找的 ID 相同，则返回当前对象
    if (this.id === a) {
        return this;
    }
    # 否则遍历子对象列表，递归调用 getObjectById 方法查找子对象
    for (var c = 0, d = this.children.length; c < d; c++) {
        var e = this.children[c].getObjectById(a, b);
        if (void 0 !== e) {
            return e;
        }
    }
},
# 根据名称获取子对象
getObjectByName: function(a, b) {
    # 如果当前对象的名称与要查找的名称相同，则返回当前对象
    if (this.name === a) {
        return this;
    }
    # 否则遍历子对象列表，递归调用 getObjectByName 方法查找子对象
    for (var c = 0, d = this.children.length; c < d; c++) {
        var e = this.children[c].getObjectByName(a, b);
        if (void 0 !== e) {
            return e;
        }
    }
},
# 获取世界坐标系中的位置
getWorldPosition: function(a) {
    # 如果未传入参数，则创建一个新的 THREE.Vector3 对象
    a = a || new THREE.Vector3;
    # 更新当前对象的世界变换矩阵
    this.updateMatrixWorld(!0);
    # 根据世界变换矩阵获取位置信息，并设置到参数对象中
    return a.setFromMatrixPosition(this.matrixWorld);
},
# 获取世界坐标系中的四元数
getWorldQuaternion: function() {
    var a = new THREE.Vector3, b = new THREE.Vector3;
    return function(c) {
        # 如果未传入参数，则创建一个新的 THREE.Quaternion 对象
        c = c || new THREE.Quaternion;
        # 更新当前对象的世界变换矩阵
        this.updateMatrixWorld(!0);
        # 根据世界变换矩阵获取旋转信息，并设置到参数对象中
        this.matrixWorld.decompose(a, c, b);
        return c;
    }
}(),
# 获取世界坐标系中的旋转角度
getWorldRotation: function() {
    var a = new THREE.Quaternion;
    return function(b) {
        # 如果未传入参数，则创建一个新的 THREE.Euler 对象
        b = b || new THREE.Euler;
        # 获取当前对象的世界坐标系中的四元数
        this.getWorldQuaternion(a);
        # 根据四元数获取旋转角度，并设置到参数对象中
        return b.setFromQuaternion(a, this.rotation.order, !1);
    }
}(),
# 获取世界坐标系中的缩放比例
getWorldScale: function() {
    var a = new THREE.Vector3, b = new THREE.Quaternion;
return function(c){c=c||new THREE.Vector3;this.updateMatrixWorld(!0);this.matrixWorld.decompose(a,b,c);return c}}(),
// 返回一个函数，该函数接受一个参数c，如果c不存在则创建一个新的THREE.Vector3对象
// 更新世界矩阵
// 将世界矩阵分解为位置、旋转和缩放，并将结果存储在c中，然后返回c

getWorldDirection:function(){var a=new THREE.Quaternion;return function(b){b=b||new THREE.Vector3;this.getWorldQuaternion(a);return b.set(0,0,1).applyQuaternion(a)}}(),
// 定义一个名为getWorldDirection的函数
// 创建一个新的THREE.Quaternion对象a
// 返回一个函数，该函数接受一个参数b，如果b不存在则创建一个新的THREE.Vector3对象
// 获取世界坐标系下的四元数，并将其存储在a中
// 将(0,0,1)向量应用四元数a的旋转，并将结果存储在b中，然后返回b

raycast:function(){},
// 定义一个名为raycast的空函数

traverse:function(a){a(this);for(var b=0,c=this.children.length;b<c;b++)this.children[b].traverse(a)},
// 定义一个名为traverse的函数，接受一个参数a
// 对当前对象调用函数a
// 遍历当前对象的子对象，对每个子对象调用traverse函数

traverseVisible:function(a){if(!1!==this.visible){a(this);for(var b=0,c=this.children.length;b<c;b++)this.children[b].traverseVisible(a)}},
// 定义一个名为traverseVisible的函数，接受一个参数a
// 如果当前对象可见，则对当前对象调用函数a
// 遍历当前对象的子对象，对每个子对象调用traverseVisible函数

updateMatrix:function(){this.matrix.compose(this.position,this.quaternion,this.scale);this.matrixWorldNeedsUpdate=!0},
// 定义一个名为updateMatrix的函数
// 使用位置、旋转和缩放信息来组合矩阵，并将结果存储在this.matrix中
// 将this.matrixWorldNeedsUpdate设置为true

updateMatrixWorld:function(a){!0===this.matrixAutoUpdate&&this.updateMatrix();if(!0===this.matrixWorldNeedsUpdate||!0===a)void 0===this.parent?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix),this.matrixWorldNeedsUpdate=!1,a=!0;for(var b=0,c=this.children.length;b<c;b++)this.children[b].updateMatrixWorld(a)},
// 定义一个名为updateMatrixWorld的函数，接受一个参数a
// 如果this.matrixAutoUpdate为true，则调用updateMatrix函数
// 如果this.matrixWorldNeedsUpdate为true或者a为true，则更新世界矩阵
// 遍历当前对象的子对象，对每个子对象调用updateMatrixWorld函数

toJSON:function(){var a={metadata:{version:4.3,type:"Object",generator:"ObjectExporter"}},b={},c=function(c){void 0===a.geometries&&(a.geometries=[]);if(void 0===b[c.uuid]){var d=c.toJSON();delete d.metadata;b[c.uuid]=d;a.geometries.push(d)}return c.uuid},d={},e=function(b){void 0===a.materials&&(a.materials=[]);if(void 0===d[b.uuid]){var c=b.toJSON();delete c.metadata;d[b.uuid]=c;a.materials.push(c)}return b.uuid},f=function(a){var b={};b.uuid=a.uuid;b.type=a.type;""!==a.name&&(b.name=a.name);"{}"!==

// 定义一个名为toJSON的函数
// 创建一个包含metadata的对象a
// 定义空对象b
// 定义函数c，接受一个参数c
// 如果a.geometries不存在，则创建一个空数组
// 如果b[c.uuid]不存在，则将c转换为JSON格式，删除metadata属性，将结果存储在b[c.uuid]中，并将其添加到a.geometries数组中
// 返回c的uuid属性
// 定义空对象d
// 定义函数e，接受一个参数b
// 如果a.materials不存在，则创建一个空数组
// 如果d[b.uuid]不存在，则将b转换为JSON格式，删除metadata属性，将结果存储在d[b.uuid]中，并将其添加到a.materials数组中
// 返回b的uuid属性
// 定义函数f，接受一个参数a
// 创建一个包含uuid、type和name属性的对象b
// 如果a的name属性不为空，则将其赋值给b的name属性
// 将 a.userData 转换为 JSON 字符串，如果不为空则将其赋值给 b.userData
JSON.stringify(a.userData)&&(b.userData=a.userData);
// 如果 a.visible 不等于 true，则将其赋值给 b.visible
!0!==a.visible&&(b.visible=a.visible);
// 根据 a 的类型进行不同的处理
a instanceof THREE.PerspectiveCamera?(b.fov=a.fov,b.aspect=a.aspect,b.near=a.near,b.far=a.far):
a instanceof THREE.OrthographicCamera?(b.left=a.left,b.right=a.right,b.top=a.top,b.bottom=a.bottom,b.near=a.near,b.far=a.far):
a instanceof THREE.AmbientLight?b.color=a.color.getHex():
a instanceof THREE.DirectionalLight?(b.color=a.color.getHex(),b.intensity=a.intensity):
a instanceof THREE.PointLight?(b.color=a.color.getHex(),b.intensity=a.intensity,b.distance=a.distance):
a instanceof THREE.SpotLight?(b.color=a.color.getHex(),b.intensity=a.intensity,b.distance=a.distance,b.angle=a.angle,b.exponent=a.exponent):
a instanceof THREE.HemisphereLight?(b.color=a.color.getHex(),b.groundColor=a.groundColor.getHex()):
a instanceof THREE.Mesh?(b.geometry=c(a.geometry),b.material=e(a.material)):
a instanceof THREE.Line?(b.geometry=c(a.geometry),b.material=e(a.material)):
a instanceof THREE.Sprite&&(b.material=e(a.material));
// 将 a.matrix 转换为数组并赋值给 b.matrix
b.matrix=a.matrix.toArray();
// 如果 a.children 的长度大于 0，则遍历 a.children 并将其克隆后添加到 b.children 中
if(0<a.children.length){
    b.children=[];
    for(var d=0;d<a.children.length;d++)
        b.children.push(f(a.children[d]))
}
// 返回克隆后的对象 b
return b};
// 将 this 对象克隆后赋值给 a，并返回 a
a.object=f(this);return a},
// 克隆当前对象，可选择是否克隆子对象
clone:function(a,b){void 0===a&&(a=new THREE.Object3D);void 0===b&&(b=!0);
// 复制当前对象的属性到 a
a.name=this.name;a.up.copy(this.up);a.position.copy(this.position);a.quaternion.copy(this.quaternion);a.scale.copy(this.scale);a.renderDepth=this.renderDepth;a.rotationAutoUpdate=this.rotationAutoUpdate;a.matrix.copy(this.matrix);a.matrixWorld.copy(this.matrixWorld);
a.matrixAutoUpdate=this.matrixAutoUpdate;a.matrixWorldNeedsUpdate=this.matrixWorldNeedsUpdate;a.visible=this.visible;a.castShadow=this.castShadow;a.receiveShadow=this.receiveShadow;a.frustumCulled=this.frustumCulled;
// 将当前对象的 userData 转换为 JSON 字符串后再解析为对象，并赋值给 a.userData
a.userData=JSON.parse(JSON.stringify(this.userData));
// 如果 b 为 true，则将当前对象的子对象克隆后添加到 a 中
if(!0===b)
    for(var c=0;c<this.children.length;c++)
        a.add(this.children[c].clone());
// 返回克隆后的对象 a
return a}};
// 将 THREE.EventDispatcher 的方法应用到 THREE.Object3D 原型上
THREE.EventDispatcher.prototype.apply(THREE.Object3D.prototype);
// 初始化对象的计数器
THREE.Object3DIdCount=0;
# 创建一个名为THREE.Projector的函数
THREE.Projector=function(){console.warn("THREE.Projector has been moved to /examples/renderers/Projector.js.");
    # 输出警告信息
    this.projectVector=function(a,b){console.warn("THREE.Projector: .projectVector() is now vector.project().");
        # 输出警告信息
        a.project(b)};
    # 输出警告信息
    this.unprojectVector=function(a,b){console.warn("THREE.Projector: .unprojectVector() is now vector.unproject().");
        # 输出警告信息
        a.unproject(b)};
    # 输出错误信息
    this.pickingRay=function(a,b){console.error("THREE.Projector: .pickingRay() has been removed.")}};
# 创建一个名为THREE.Face3的函数
THREE.Face3=function(a,b,c,d,e,f){this.a=a;this.b=b;this.c=c;this.normal=d instanceof THREE.Vector3?d:new THREE.Vector3;
    # 如果d是THREE.Vector3类型，则使用d，否则创建一个新的THREE.Vector3
    this.vertexNormals=d instanceof Array?d:[];this.color=e instanceof THREE.Color?e:new THREE.Color;
    # 如果e是THREE.Color类型，则使用e，否则创建一个新的THREE.Color
    this.vertexColors=e instanceof Array?e:[];this.vertexTangents=[];
    # 创建一个空数组
    this.materialIndex=void 0!==f?f:0};
# 给THREE.Face3的原型添加方法
THREE.Face3.prototype={constructor:THREE.Face3,clone:function(){var a=new THREE.Face3(this.a,this.b,this.c);
    # 创建一个新的THREE.Face3对象
    a.normal.copy(this.normal);a.color.copy(this.color);a.materialIndex=this.materialIndex;
    # 复制normal和color属性，设置materialIndex属性
    for(var b=0,c=this.vertexNormals.length;b<c;b++)a.vertexNormals[b]=this.vertexNormals[b].clone();
    # 遍历vertexNormals数组，复制每个元素并添加到a的vertexNormals数组中
    b=0;for(c=this.vertexColors.length;b<c;b++)a.vertexColors[b]=this.vertexColors[b].clone();
    # 遍历vertexColors数组，复制每个元素并添加到a的vertexColors数组中
    b=0;for(c=this.vertexTangents.length;b<c;b++)a.vertexTangents[b]=this.vertexTangents[b].clone();
    # 遍历vertexTangents数组，复制每个元素并添加到a的vertexTangents数组中
    return a}};
# 创建一个名为THREE.Face4的函数
THREE.Face4=function(a,b,c,d,e,f,g){console.warn("THREE.Face4 has been removed. A THREE.Face3 will be created instead.");
    # 输出警告信息
    return new THREE.Face3(a,b,c,e,f,g)};
# 创建一个名为THREE.BufferAttribute的函数
THREE.BufferAttribute=function(a,b){this.array=a;this.itemSize=b;this.needsUpdate=!1};
# 创建一个BufferAttribute对象，包含array、itemSize和needsUpdate属性
# 定义 THREE.BufferAttribute 的原型对象
THREE.BufferAttribute.prototype={
    constructor:THREE.BufferAttribute,
    # 获取数组长度的属性
    get length(){
        return this.array.length
    },
    # 在指定位置复制另一个 BufferAttribute 的数据
    copyAt:function(a,b,c){
        a*=this.itemSize;
        c*=b.itemSize;
        for(var d=0,e=this.itemSize;d<e;d++)
            this.array[a+d]=b.array[c+d]
    },
    # 设置整个数组的值
    set:function(a){
        this.array.set(a);
        return this
    },
    # 设置指定位置的 X 值
    setX:function(a,b){
        this.array[a*this.itemSize]=b;
        return this
    },
    # 设置指定位置的 Y 值
    setY:function(a,b){
        this.array[a*this.itemSize+1]=b;
        return this
    },
    # 设置指定位置的 Z 值
    setZ:function(a,b){
        this.array[a*this.itemSize+2]=b;
        return this
    },
    # 设置指定位置的 X、Y 值
    setXY:function(a,b,c){
        a*=this.itemSize;
        this.array[a]=b;
        this.array[a+1]=c;
        return this
    },
    # 设置指定位置的 X、Y、Z 值
    setXYZ:function(a,b,c,d){
        a*=this.itemSize;
        this.array[a]=b;
        this.array[a+1]=c;
        this.array[a+2]=d;
        return this
    },
    # 设置指定位置的 X、Y、Z、W 值
    setXYZW:function(a,b,c,d,e){
        a*=this.itemSize;
        this.array[a]=b;
        this.array[a+1]=c;
        this.array[a+2]=d;
        this.array[a+3]=e;
        return this
    },
    # 克隆 BufferAttribute 对象
    clone:function(){
        return new THREE.BufferAttribute(new this.array.constructor(this.array),this.itemSize)
    }
};

# 定义一系列已被移除的属性，给出警告并返回新的 BufferAttribute 对象
THREE.Int8Attribute=function(a,b){
    console.warn("THREE.Int8Attribute has been removed. Use THREE.BufferAttribute( array, itemSize ) instead.");
    return new THREE.BufferAttribute(a,b)
};
THREE.Uint8Attribute=function(a,b){
    console.warn("THREE.Uint8Attribute has been removed. Use THREE.BufferAttribute( array, itemSize ) instead.");
    return new THREE.BufferAttribute(a,b)
};
THREE.Uint8ClampedAttribute=function(a,b){
    console.warn("THREE.Uint8ClampedAttribute has been removed. Use THREE.BufferAttribute( array, itemSize ) instead.");
    return new THREE.BufferAttribute(a,b)
};
THREE.Int16Attribute=function(a,b){
    console.warn("THREE.Int16Attribute has been removed. Use THREE.BufferAttribute( array, itemSize ) instead.");
    return new THREE.BufferAttribute(a,b)
};
# 创建一个函数THREE.Uint16Attribute，接受两个参数a和b
THREE.Uint16Attribute=function(a,b){
    # 输出警告信息，提示THREE.Uint16Attribute已被移除，建议使用THREE.BufferAttribute( array, itemSize )代替
    console.warn("THREE.Uint16Attribute has been removed. Use THREE.BufferAttribute( array, itemSize ) instead.");
    # 返回一个新的THREE.BufferAttribute对象，传入参数a和b
    return new THREE.BufferAttribute(a,b)
};
# 创建一个函数THREE.Int32Attribute，接受两个参数a和b
THREE.Int32Attribute=function(a,b){
    # 输出警告信息，提示THREE.Int32Attribute已被移除，建议使用THREE.BufferAttribute( array, itemSize )代替
    console.warn("THREE.Int32Attribute has been removed. Use THREE.BufferAttribute( array, itemSize ) instead.");
    # 返回一个新的THREE.BufferAttribute对象，传入参数a和b
    return new THREE.BufferAttribute(a,b)
};
# 创建一个函数THREE.Uint32Attribute，接受两个参数a和b
THREE.Uint32Attribute=function(a,b){
    # 输出警告信息，提示THREE.Uint32Attribute已被移除，建议使用THREE.BufferAttribute( array, itemSize )代替
    console.warn("THREE.Uint32Attribute has been removed. Use THREE.BufferAttribute( array, itemSize ) instead.");
    # 返回一个新的THREE.BufferAttribute对象，传入参数a和b
    return new THREE.BufferAttribute(a,b)
};
# 创建一个函数THREE.Float32Attribute，接受两个参数a和b
THREE.Float32Attribute=function(a,b){
    # 输出警告信息，提示THREE.Float32Attribute已被移除，建议使用THREE.BufferAttribute( array, itemSize )代替
    console.warn("THREE.Float32Attribute has been removed. Use THREE.BufferAttribute( array, itemSize ) instead.");
    # 返回一个新的THREE.BufferAttribute对象，传入参数a和b
    return new THREE.BufferAttribute(a,b)
};
# 创建一个函数THREE.Float64Attribute，接受两个参数a和b
THREE.Float64Attribute=function(a,b){
    # 输出警告信息，提示THREE.Float64Attribute已被移除，建议使用THREE.BufferAttribute( array, itemSize )代替
    console.warn("THREE.Float64Attribute has been removed. Use THREE.BufferAttribute( array, itemSize ) instead.");
    # 返回一个新的THREE.BufferAttribute对象，传入参数a和b
    return new THREE.BufferAttribute(a,b)
};
# 创建一个构造函数THREE.BufferGeometry
THREE.BufferGeometry=function(){
    # 设置this对象的id属性为THREE.GeometryIdCount的值
    Object.defineProperty(this,"id",{value:THREE.GeometryIdCount++});
    # 生成一个UUID并赋值给this对象的uuid属性
    this.uuid=THREE.Math.generateUUID();
    # 设置this对象的name属性为空字符串
    this.name="";
    # 设置this对象的type属性为"BufferGeometry"
    this.type="BufferGeometry";
    # 初始化this对象的attributes、attributesKeys、offsets、drawcalls、boundingSphere和boundingBox属性
    this.attributes={};
    this.attributesKeys=[];
    this.offsets=this.drawcalls=[];
    this.boundingSphere=this.boundingBox=null
};
# 设置THREE.BufferGeometry的原型对象
THREE.BufferGeometry.prototype={
    # 设置构造函数为THREE.BufferGeometry
    constructor:THREE.BufferGeometry,
    # 添加一个属性到this对象的attributes属性中
    addAttribute:function(a,b,c){
        # 如果b不是THREE.BufferAttribute的实例，则输出警告信息
        !1===b instanceof THREE.BufferAttribute?(console.warn("THREE.BufferGeometry: .addAttribute() now expects ( name, attribute )."),
        # 将属性名和属性值添加到this对象的attributes属性中
        this.attributes[a]={array:b,itemSize:c}):
        (this.attributes[a]=b,
        # 更新this对象的attributesKeys属性
        this.attributesKeys=Object.keys(this.attributes))
    },
    # 获取this对象的attributes属性中指定名称的属性
    getAttribute:function(a){
        return this.attributes[a]
    },
    # 添加一个绘制调用到this对象的drawcalls属性中
    addDrawCall:function(a,b,c){
        this.drawcalls.push({start:a,count:b,index:void 0!==c?c:0})
    },
    # 对this对象应用矩阵变换
    applyMatrix:function(a){
        var b=
# 获取属性中的位置信息
this.attributes.position;
# 如果 b 不为 undefined，则将 a 应用到向量数组中，并设置需要更新标志
void 0!==b&&(a.applyToVector3Array(b.array),b.needsUpdate=!0);
# 获取属性中的法线信息
b=this.attributes.normal;
# 如果 b 不为 undefined，则使用新的三维矩阵获取法线矩阵，并将其应用到向量数组中，并设置需要更新标志
void 0!==b&&((new THREE.Matrix3).getNormalMatrix(a).applyToVector3Array(b.array),b.needsUpdate=!0)},center:function(){},fromGeometry:function(a,b){b=b||{vertexColors:THREE.NoColors};
# 从几何体中创建缓冲几何体，设置顶点颜色为无颜色
var c=a.vertices,d=a.faces,e=a.faceVertexUvs,f=b.vertexColors,g=0<e[0].length,h=3==d[0].vertexNormals.length,k=new Float32Array(9*d.length);
# 添加顶点位置属性
this.addAttribute("position",new THREE.BufferAttribute(k,3));
var n=new Float32Array(9*d.length);
# 添加法线属性
this.addAttribute("normal",new THREE.BufferAttribute(n,3));
# 如果顶点颜色不为无颜色，则添加颜色属性
if(f!==THREE.NoColors){var p=new Float32Array(9*d.length);this.addAttribute("color",new THREE.BufferAttribute(p,3))}
# 如果存在纹理坐标，则添加 UV 属性
if(!0===g){var q=new Float32Array(6*d.length);this.addAttribute("uv",new THREE.BufferAttribute(q,2))}
# 遍历所有面
for(var m=0,r=0,t=0;m<d.length;m++,r+=6,t+=9){
    var s=d[m],u=c[s.a],v=c[s.b],y=c[s.c];
    # 设置顶点位置信息
    k[t]=u.x;k[t+1]=u.y;k[t+2]=u.z;k[t+3]=v.x;k[t+4]=v.y;k[t+5]=v.z;k[t+6]=y.x;k[t+7]=y.y;k[t+8]=y.z;
    # 如果存在顶点法线信息，则设置法线信息
    !0===h?(u=s.vertexNormals[0],v=s.vertexNormals[1],y=s.vertexNormals[2],n[t]=u.x,n[t+1]=u.y,n[t+2]=u.z,n[t+3]=v.x,n[t+4]=v.y,n[t+5]=v.z,n[t+6]=y.x,n[t+7]=y.y,n[t+8]=y.z):
    (u=s.normal,n[t]=u.x,n[t+1]=u.y,n[t+2]=u.z,n[t+3]=u.x,n[t+4]=u.y,n[t+5]=u.z,n[t+6]=u.x,n[t+7]=u.y,n[t+8]=u.z);
    # 如果顶点颜色为面颜色，则设置颜色信息
    f===THREE.FaceColors?(s=s.color,p[t]=s.r,p[t+1]=s.g,p[t+2]=s.b,p[t+3]=s.r,p[t+4]=s.g,p[t+5]=s.b,p[t+6]=s.r,p[t+7]=s.g,p[t+8]=s.b):
    # 如果顶点颜色为顶点颜色，则设置颜色信息
    f===THREE.VertexColors&&(u=s.vertexColors[0],v=s.vertexColors[1],s=s.vertexColors[2],p[t]=u.r,p[t+1]=u.g,p[t+2]=u.b,p[t+3]=
// 设置顶点颜色和位置
v.r, p[t+4] = v.g, p[t+5] = v.b, p[t+6] = s.r, p[t+7] = s.g, p[t+8] = s.b);
// 如果条件为真，设置顶点位置
!0 === g && (s = e[0][m][0], u = e[0][m][1], v = e[0][m][2], q[r] = s.x, q[r+1] = s.y, q[r+2] = u.x, q[r+3] = u.y, q[r+4] = v.x, q[r+5] = v.y)
// 计算包围球
this.computeBoundingSphere();
// 返回结果
return this
// 计算包围盒
computeBoundingBox: function() {
    var a = new THREE.Vector3;
    return function() {
        null === this.boundingBox && (this.boundingBox = new THREE.Box3);
        var b = this.attributes.position.array;
        if (b) {
            var c = this.boundingBox;
            c.makeEmpty();
            for (var d = 0, e = b.length; d < e; d += 3) a.set(b[d], b[d+1], b[d+2]), c.expandByPoint(a)
        }
        if (void 0 === b || 0 === b.length) this.boundingBox.min.set(0, 0, 0), this.boundingBox.max.set(0, 0, 0);
        (isNaN(this.boundingBox.min.x) || isNaN(this.boundingBox.min.y) || isNaN(this.boundingBox.min.z)) && console.error('THREE.BufferGeometry.computeBoundingBox: Computed min/max have NaN values. The "position" attribute is likely to have NaN values.')
    }()
},
// 计算包围球
computeBoundingSphere: function() {
    var a = new THREE.Box3, b = new THREE.Vector3;
    return function() {
        null === this.boundingSphere && (this.boundingSphere = new THREE.Sphere);
        var c = this.attributes.position.array;
        if (c) {
            a.makeEmpty();
            for (var d = this.boundingSphere.center, e = 0, f = c.length; e < f; e += 3) b.set(c[e], c[e+1], c[e+2]), a.expandByPoint(b);
            a.center(d);
            for (var g = 0, e = 0, f = c.length; e < f; e += 3) b.set(c[e], c[e+1], c[e+2]), g = Math.max(g, d.distanceToSquared(b));
            this.boundingSphere.radius = Math.sqrt(g);
            isNaN(this.boundingSphere.radius) && console.error('THREE.BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.')
        }
    }()
},
// 计算面法线
computeFaceNormals: function() {},
// 计算顶点法线
computeVertexNormals: function() {
    var a = 
this.attributes;
// 检查是否存在属性

if(a.position){
    var b=a.position.array;
    // 如果存在位置属性，获取位置数组
    if(void 0===a.normal)
        this.addAttribute("normal",new THREE.BufferAttribute(new Float32Array(b.length),3));
    // 如果不存在法线属性，创建法线属性并添加到对象中
    else 
        for(var c=a.normal.array,d=0,e=c.length;d<e;d++)
            c[d]=0;
    // 如果存在法线属性，将其数组元素全部置为0

    var c=a.normal.array,
        f,g,h,k=new THREE.Vector3,
        n=new THREE.Vector3,
        p=new THREE.Vector3,
        q=new THREE.Vector3,
        m=new THREE.Vector3;
    // 定义变量和创建向量对象

    if(a.index)
        // 如果存在索引属性
        for(var r=a.index.array,
            t=0<this.offsets.length?this.offsets:[{start:0,count:r.length,index:0}],
            s=0,u=t.length;s<u;++s){
            // 遍历索引属性
            e=t[s].start;
            f=t[s].count;
            // 获取起始位置和数量
            for(var v=t[s].index,
                d=e,
                e=e+f;d<e;d+=3){
                // 遍历索引数组
                f=3*(v+r[d]);
                g=3*(v+r[d+1]);
                h=3*(v+r[d+2]);
                // 计算顶点索引
                k.fromArray(b,f);
                n.fromArray(b,g);
                p.fromArray(b,h);
                // 获取顶点坐标
                q.subVectors(p,n);
                m.subVectors(k,n);
                q.cross(m);
                // 计算切线
                c[f]+=q.x;
                c[f+1]+=q.y;
                c[f+2]+=q.z;
                c[g]+=q.x;
                c[g+1]+=q.y;
                c[g+2]+=q.z;
                c[h]+=q.x;
                c[h+1]+=q.y;
                c[h+2]+=q.z;
                // 更新法线数组
            }
        }
    else 
        // 如果不存在索引属性
        for(d=0,e=b.length;d<e;d+=9){
            // 遍历位置数组
            k.fromArray(b,d);
            n.fromArray(b,d+3);
            p.fromArray(b,d+6);
            // 获取顶点坐标
            q.subVectors(p,n);
            m.subVectors(k,n);
            q.cross(m);
            // 计算切线
            c[d]=q.x;
            c[d+1]=q.y;
            c[d+2]=q.z;
            c[d+3]=q.x;
            c[d+4]=q.y;
            c[d+5]=q.z;
            c[d+6]=q.x;
            c[d+7]=q.y;
            c[d+8]=q.z;
            // 更新法线数组
        }
    this.normalizeNormals();
    // 归一化法线
    a.normal.needsUpdate=!0;
    // 更新法线属性
}
},
computeTangents:function(){
    // 计算切线
    function a(a,b,c){
        // 定义函数
        q.fromArray(d,3*a);
        m.fromArray(d,3*b);
        r.fromArray(d,3*c);
        t.fromArray(f,2*a);
        s.fromArray(f,2*b);
        u.fromArray(f,2*c);
        // 获取顶点和纹理坐标
        v=m.x-q.x;
        y=r.x-q.x;
        G=m.y-q.y;
        w=r.y-q.y;
        K=m.z-q.z;
        x=r.z-q.z;
        D=s.x-t.x;
        E=u.x-t.x;
        A=s.y-t.y;
        B=u.y-t.y;
        F=1/(D*B-E*A);
        R.set((B*v-A*y)*F,(B*G-A*w)*F,(B*K-A*x)*F);
        H.set((D*y-E*v)*F,(D*w-E*G)*F,(D*x-E*K)*F);
        k[a].add(R);
        k[b].add(R);
        k[c].add(R);
        n[a].add(H);
        n[b].add(H);
        n[c].add(H);
        // 计算切线和副切线
    }
    function b(a){
        // 定义函数
        ya.fromArray(e,
# 计算切线
3*a);P.copy(ya);Fa=k[a];la.copy(Fa);la.sub(ya.multiplyScalar(ya.dot(Fa))).normalize();ma.crossVectors(P,Fa);za=ma.dot(n[a]);Ga=0>za?-1:1;h[4*a]=la.x;h[4*a+1]=la.y;h[4*a+2]=la.z;h[4*a+3]=Ga
# 计算切线的过程，包括向量的复制、减法、标量乘法、归一化等操作

if(void 0===this.attributes.index||void 0===this.attributes.position||void 0===this.attributes.normal||void 0===this.attributes.uv)console.warn("Missing required attributes (index, position, normal or uv) in BufferGeometry.computeTangents()");
# 如果缺少必要的属性（index, position, normal 或 uv），则输出警告信息

else{
    var c=this.attributes.index.array,d=this.attributes.position.array,
    e=this.attributes.normal.array,f=this.attributes.uv.array,g=d.length/3;
    void 0===this.attributes.tangent&&this.addAttribute("tangent",new THREE.BufferAttribute(new Float32Array(4*g),4));
    # 如果不存在切线属性，则添加切线属性

    for(var h=this.attributes.tangent.array,k=[],n=[],p=0;p<g;p++)k[p]=new THREE.Vector3,n[p]=new THREE.Vector3;
    var q=new THREE.Vector3,m=new THREE.Vector3,r=new THREE.Vector3,t=new THREE.Vector2,s=new THREE.Vector2,u=new THREE.Vector2,v,y,G,w,K,x,D,E,A,B,F,R=new THREE.Vector3,H=new THREE.Vector3,C,T,Q,O,S;
    # 定义一系列向量和变量

    0===this.drawcalls.length&&this.addDrawCall(0,c.length,0);
    # 如果绘制调用的长度为0，则添加绘制调用

    var X=this.drawcalls,p=0;
    for(T=X.length;p<T;++p){
        C=X[p].start;Q=X[p].count;
        var Y=X[p].index,g=C;
        for(C+=Q;g<C;g+=3)Q=Y+c[g],O=Y+c[g+1],S=Y+c[g+2],a(Q,O,S);
    }
    # 遍历绘制调用，执行a函数

    var la=new THREE.Vector3,ma=new THREE.Vector3,ya=new THREE.Vector3,P=new THREE.Vector3,Ga,Fa,za,p=0;
    for(T=X.length;p<T;++p)
        for(C=X[p].start,Q=X[p].count,Y=X[p].index,g=C,C+=Q;g<C;g+=3)Q=Y+c[g],O=Y+c[g+1],S=Y+c[g+2],b(Q),b(O),b(S);
    }
    # 遍历绘制调用，执行b函数
},

computeOffsets:function(a){
    var b=a;
    void 0===a&&(b=65535);
    Date.now();
    a=this.attributes.index.array;
    # 计算偏移量
}
# 遍历属性中的位置数组，计算长度并初始化一些变量
for(var c=this.attributes.position.array,d=a.length/3,e=new Uint16Array(a.length),f=0,g=0,h=[{start:0,count:0,index:0}],k=h[0],n=0,p=0,q=new Int32Array(6),m=new Int32Array(c.length),r=new Int32Array(c.length),t=0;t<c.length;t++)m[t]=-1,r[t]=-1;
# 遍历每个顶点
for(c=0;c<d;c++){
    # 初始化一些变量
    for(var s=p=0;3>s;s++)t=a[3*c+s],-1==m[t]?(q[2*s]=t,q[2*s+1]=-1,p++):m[t]<k.index?(q[2*s]=t,q[2*s+1]=-1,n++):(q[2*s]=t,q[2*s+1]=m[t]);
    # 如果当前顶点的索引超过了当前的最大索引，重新初始化一些变量
    if(g+p>k.index+b)for(k={start:f,count:0,index:g},h.push(k),p=0;6>p;p+=2)s=q[p+1],-1<s&&s<k.index&&(q[p+1]=-1);
    # 更新索引和计数
    for(p=0;6>p;p+=2)t=q[p],s=q[p+1],-1===s&&(s=g++),m[t]=s,r[s]=t,e[f++]=s-k.index,k.count++
}
# 重新排序缓冲区
this.reorderBuffers(e,r,g);
# 返回偏移量数组
return this.offsets=h
},
# 合并函数
merge:function(){
    console.log("BufferGeometry.merge(): TODO")
},
# 归一化法线
normalizeNormals:function(){
    for(var a=this.attributes.normal.array,b,c,d,e=0,f=a.length;e<f;e+=3)b=a[e],c=a[e+1],d=a[e+2],b=1/Math.sqrt(b*b+c*c+d*d),a[e]*=b,a[e+1]*=b,a[e+2]*=b
},
# 重新排序缓冲区
reorderBuffers:function(a,b,c){
    var d={},e;
    for(e in this.attributes)"index"!=e&&(d[e]=new this.attributes[e].array.constructor(this.attributes[e].itemSize*c));
    for(var f=0;f<c;f++){
        var g=b[f];
        for(e in this.attributes)if("index"!=e)for(var h=this.attributes[e].array,k=this.attributes[e].itemSize,n=d[e],p=0;p<k;p++)n[f*k+p]=h[g*k+p]
    }
    this.attributes.index.array=a;
    for(e in this.attributes)"index"!=e&&(this.attributes[e].array=d[e],this.attributes[e].numItems=this.attributes[e].itemSize*c)
},
# 转换为 JSON 格式
toJSON:function(){
    var a={metadata:{version:4,type:"BufferGeometry",generator:"BufferGeometryExporter"},uuid:this.uuid,type:this.type,data:{attributes:{}}},b=this.attributes,
# 定义一个变量c，赋值为this.offsets；定义一个变量d，赋值为this.boundingSphere；定义一个变量e
c=this.offsets,d=this.boundingSphere,e;
# 遍历对象b的属性
for(e in b){
    # 定义变量f，赋值为b[e]；定义一个空数组g
    for(var f=b[e],g=[],h=f.array,k=0,n=h.length;k<n;k++)g[k]=h[k];
    # 将属性e的值赋给a.data.attributes[e]，并且将属性e的值的类型和数据复制给a.data.attributes[e]
    a.data.attributes[e]={itemSize:f.itemSize,type:f.array.constructor.name,array:g}
}
# 如果c的长度大于0，则将c的深拷贝赋给a.data.offsets
0<c.length&&(a.data.offsets=JSON.parse(JSON.stringify(c)));
# 如果d不为null，则将d的center属性转换为数组，赋给a.data.boundingSphere的center属性；将d的radius属性赋给a.data.boundingSphere的radius属性
null!==d&&(a.data.boundingSphere={center:d.center.toArray(),radius:d.radius});
# 返回a
return a
# 克隆一个THREE.BufferGeometry对象
},clone:function(){
    var a=new THREE.BufferGeometry,b;
    for(b in this.attributes)a.addAttribute(b,this.attributes[b].clone());
    b=0;
    for(var c=this.offsets.length;b<c;b++){
        var d=this.offsets[b];
        a.offsets.push({start:d.start,index:d.index,count:d.count})
    }
    return a
},
# 释放THREE.BufferGeometry对象
dispose:function(){
    this.dispatchEvent({type:"dispose"})
}};
# 将THREE.EventDispatcher的原型方法应用到THREE.BufferGeometry的原型上
THREE.EventDispatcher.prototype.apply(THREE.BufferGeometry.prototype);
# 定义THREE.Geometry构造函数
THREE.Geometry=function(){
    Object.defineProperty(this,"id",{value:THREE.GeometryIdCount++});
    this.uuid=THREE.Math.generateUUID();
    this.name="";
    this.type="Geometry";
    this.vertices=[];
    this.colors=[];
    this.faces=[];
    this.faceVertexUvs=[[]];
    this.morphTargets=[];
    this.morphColors=[];
    this.morphNormals=[];
    this.skinWeights=[];
    this.skinIndices=[];
    this.lineDistances=[];
    this.boundingSphere=this.boundingBox=null;
    this.hasTangents=!1;
    this.dynamic=!0;
    this.groupsNeedUpdate=this.lineDistancesNeedUpdate=this.colorsNeedUpdate=this.tangentsNeedUpdate=this.normalsNeedUpdate=this.uvsNeedUpdate=this.elementsNeedUpdate=this.verticesNeedUpdate=!1
};
# 将THREE.Geometry的原型方法定义
THREE.Geometry.prototype={
    constructor:THREE.Geometry,
    applyMatrix:function(a){
        for(var b=(new THREE.Matrix3).getNormalMatrix(a),c=0,d=this.vertices.length;c<d;c++)this.vertices[c].applyMatrix4(a);
        c=0;
        for(d=this.faces.length;c<d;c++){
            a=this.faces[c];
            a.normal.applyMatrix3(b).normalize();
            for(var e=0,f=a.vertexNormals.length;e<f;e++)a.vertexNormals[e].applyMatrix3(b).normalize()
        }
        this.boundingBox instanceof THREE.Box3&&this.computeBoundingBox();
        this.boundingSphere instanceof THREE.Sphere&&this.computeBoundingSphere()
    },
# 从缓冲几何体中创建几何体
def fromBufferGeometry:function(a):
    # 获取属性
    for var b=this, c=a.attributes, d=c.position.array, e=void 0!==c.index?c.index.array:void 0, f=void 0!==c.normal?c.normal.array:void 0, g=void 0!==c.color?c.color.array:void 0, h=void 0!==c.uv?c.uv.array:void 0, k=[], n=[], p=c=0; c<d.length; c+=3, p+=2:
        # 添加顶点
        b.vertices.push(new THREE.Vector3(d[c],d[c+1],d[c+2]))
        # 添加法线
        if void 0!==f:
            k.push(new THREE.Vector3(f[c],f[c+1],f[c+2]))
        # 添加颜色
        if void 0!==g:
            b.colors.push(new THREE.Color(g[c],g[c+1],g[c+2]))
        # 添加UV
        if void 0!==h:
            n.push(new THREE.Vector2(h[p],h[p+1]))
    # 定义面
    h=function(a,c,d):
        e=void 0!==f?[k[a].clone(),k[c].clone(),k[d].clone()]:[]
        h=void 0!==g?[b.colors[a].clone(),b.colors[c].clone(),b.colors[d].clone()]:[]
        b.faces.push(new THREE.Face3(a,c,d,e,h))
        b.faceVertexUvs[0].push([n[a],n[c],n[d]])
    # 如果存在索引
    if void 0!==e:
        for c=0; c<e.length; c+=3:
            h(e[c],e[c+1],e[c+2])
    # 如果不存在索引
    else:
        for c=0; c<d.length/3; c+=3:
            h(c,c+1,c+2)
    # 计算面法线
    this.computeFaceNormals()
    # 复制包围盒和包围球
    if null!==a.boundingBox:
        this.boundingBox=a.boundingBox.clone()
    if null!==a.boundingSphere:
        this.boundingSphere=a.boundingSphere.clone()
    return this
# 居中几何体
def center:function():
    this.computeBoundingBox()
    a=new THREE.Vector3()
    a.addVectors(this.boundingBox.min,this.boundingBox.max)
    a.multiplyScalar(-.5)
    this.applyMatrix((new THREE.Matrix4).makeTranslation(a.x,a.y,a.z))
    this.computeBoundingBox()
    return a
# 计算面法线
def computeFaceNormals:function():
    a=new THREE.Vector3()
    b=new THREE.Vector3()
    for c=0; c<this.faces.length; c++:
        e=this.faces[c]
        f=this.vertices[e.a]
        g=this.vertices[e.b]
        a.subVectors(this.vertices[e.c],g)
        b.subVectors(f,g)
        a.cross(b)
        a.normalize()
// 复制顶点法线到新的数组中
e.normal.copy(a)
// 计算顶点法线
computeVertexNormals:function(a){
    var b,c,d;
    d=Array(this.vertices.length);
    b=0;
    for(c=this.vertices.length;b<c;b++)
        d[b]=new THREE.Vector3;
    if(a){
        var e,f,g,h=new THREE.Vector3,k=new THREE.Vector3;
        new THREE.Vector3;
        new THREE.Vector3;
        new THREE.Vector3;
        a=0;
        for(b=this.faces.length;a<b;a++)
            c=this.faces[a],
            e=this.vertices[c.a],
            f=this.vertices[c.b],
            g=this.vertices[c.c],
            h.subVectors(g,f),
            k.subVectors(e,f),
            h.cross(k),
            d[c.a].add(h),
            d[c.b].add(h),
            d[c.c].add(h)
    }else 
        for(a=0,b=this.faces.length;a<b;a++)
            c=this.faces[a],
            d[c.a].add(c.normal),
            d[c.b].add(c.normal),
            d[c.c].add(c.normal);
    b=0;
    for(c=this.vertices.length;b<c;b++)
        d[b].normalize();
    a=0;
    for(b=this.faces.length;a<b;a++)
        c=this.faces[a],
        c.vertexNormals[0]=d[c.a].clone(),
        c.vertexNormals[1]=d[c.b].clone(),
        c.vertexNormals[2]=d[c.c].clone()
},
// 计算变形法线
computeMorphNormals:function(){
    var a,b,c,d,e;
    c=0;
    for(d=this.faces.length;c<d;c++)
        for(e=this.faces[c],
            e.__originalFaceNormal?e.__originalFaceNormal.copy(e.normal):e.__originalFaceNormal=e.normal.clone(),
            e.__originalVertexNormals||(e.__originalVertexNormals=[]),
            a=0,b=e.vertexNormals.length;a<b;a++)
            e.__originalVertexNormals[a]?e.__originalVertexNormals[a].copy(e.vertexNormals[a]):e.__originalVertexNormals[a]=e.vertexNormals[a].clone();
    var f=new THREE.Geometry;
    f.faces=this.faces;
    a=0;
    for(b=this.morphTargets.length;a<b;a++){
        if(!this.morphNormals[a]){
            this.morphNormals[a]={};
            this.morphNormals[a].faceNormals=[];
            this.morphNormals[a].vertexNormals=[];
            e=this.morphNormals[a].faceNormals;
            var g=this.morphNormals[a].vertexNormals,h,k;
            c=
# 循环遍历所有面
for (var c = 0; c < this.faces.length; c++) {
    // 创建新的三维向量
    h = new THREE.Vector3,
    // 创建包含三个三维向量的对象
    k = {a:new THREE.Vector3, b:new THREE.Vector3, c:new THREE.Vector3},
    // 将新创建的向量添加到数组中
    e.push(h),
    // 将新创建的对象添加到数组中
    g.push(k)
}
// 获取指定索引的变形法线
g = this.morphNormals[a],
// 获取指定索引的变形顶点
f.vertices = this.morphTargets[a].vertices,
// 计算面法线
f.computeFaceNormals(),
// 计算顶点法线
f.computeVertexNormals(),
// 循环遍历所有面
for (var c = 0; c < this.faces.length; c++) {
    // 获取当前面
    e = this.faces[c],
    // 获取当前面的法线
    h = g.faceNormals[c],
    // 获取当前面的顶点法线
    k = g.vertexNormals[c],
    // 复制面的法线
    h.copy(e.normal),
    // 复制顶点的法线
    k.a.copy(e.vertexNormals[0]),
    k.b.copy(e.vertexNormals[1]),
    k.c.copy(e.vertexNormals[2])
}
// 循环遍历所有面
for (var c = 0; c < this.faces.length; c++) {
    // 获取当前面
    e = this.faces[c],
    // 恢复原始面的法线
    e.normal = e.__originalFaceNormal,
    // 恢复原始面的顶点法线
    e.vertexNormals = e.__originalVertexNormals
}
// 计算切线
computeTangents: function() {
    // 定义变量
    var a, b, c, d, e, f, g, h, k, n, p, q, m, r, t, s, u, v = [], y = [],
    // 创建三维向量
    c = new THREE.Vector3,
    // 创建三维向量
    G = new THREE.Vector3,
    // 创建三维向量
    w = new THREE.Vector3,
    // 创建三维向量
    K = new THREE.Vector3,
    // 创建三维向量
    x = new THREE.Vector3;
    // 循环遍历所有顶点
    for (a = 0; a < this.vertices.length; a++) {
        // 创建新的三维向量
        v[a] = new THREE.Vector3,
        // 创建新的三维向量
        y[a] = new THREE.Vector3
    }
    // 循环遍历所有面
    for (a = 0; a < this.faces.length; a++) {
        // 获取当前面
        e = this.faces[a],
        // 获取当前面的纹理坐标
        f = this.faceVertexUvs[0][a],
        // 获取当前面的顶点索引
        d = e.a,
        u = e.b,
        e = e.c,
        // 获取当前面的顶点
        g = this.vertices[d],
        h = this.vertices[u],
        k = this.vertices[e],
        // 获取当前面的纹理坐标
        n = f[0],
        p = f[1],
        q = f[2],
        // 计算差值
        f = h.x - g.x,
        m = k.x - g.x,
        r = h.y - g.y,
        t = k.y - g.y,
        h = h.z - g.z,
        g = k.z - g.z,
        k = p.x - n.x,
        s = q.x - n.x,
        p = p.y - n.y,
        n = q.y - n.y,
        q = 1 / (k * n - s * p),
        // 设置向量的值
        c.set((n * f - p * m) * q, (n * r - p * t) * q, (n * h - p * g) * q),
        G.set((k * m - s * f) * q, (k * t - s * r) * q, (k * g - s * h) * q),
        // 添加向量
        v[d].add(c),
        v[u].add(c),
        v[e].add(c),
        y[d].add(G),
        y[u].add(G),
        y[e].add(G)
    }
    // 定义数组
    G = ["a", "b", "c", "d"];
    // 循环遍历所有面
    for (a = 0; a < this.faces.length; a++) {
        // 获取当前面
        e = this.faces[a],
        // 循环遍历当前面的顶点法线
        for (c = 0; c < Math.min(e.vertexNormals.length, 3); c++) {
            // 复制顶点法线
            x.copy(e.vertexNormals[c]),
            // 获取当前顶点索引
            d = e[G[c]],
            // 获取当前顶点的向量
            u = v[d],
            // 复制当前顶点的向量
            w.copy(u),
            // 计算切线
            w.sub(x.multiplyScalar(x.dot(u))).normalize(),
            // 计算切线
            K.crossVectors(e.vertexNormals[c],
# 定义一个函数，用于计算顶点切线
computeTangents:function(){
    for(var c=0,d=this.faces.length;c<d;c++){
        var e=this.faces[c],
        f=this.vertices[e.a],
        g=this.vertices[e.b],
        h=this.vertices[e.c],
        k=f.x-g.x,
        l=f.y-g.y,
        m=f.z-g.z,
        n=h.x-g.x,
        p=h.y-g.y,
        q=h.z-g.z,
        r=e.vertexNormals[0],
        t=r.x,
        r=r.y,
        r=r.z,
        s=e.vertexNormals[1],
        v=s.x,
        s=s.y,
        s=s.z,
        s=e.vertexNormals[2],
        w=s.x,
        s=s.y,
        s=s.z,
        x=e.vertexUVs[0],
        y=x.x,
        x=x.y,
        x=e.vertexUVs[1],
        z=x.x,
        x=x.y,
        x=e.vertexUVs[2],
        A=x.x,
        x=x.y,
        x=k*n+l*p+m*q,
        B=Math.sqrt(k*k+l*l+m*m),
        C=Math.sqrt(n*n+p*p+q*q),
        D=Math.sqrt(x*x+B*C),
        x=x/D,
        B=B/D,
        C=C/D,
        D=t*v+r*s+w*x,
        E=t*y+r*z+w*A,
        F=v*y+s*z+x*A,
        G=new THREE.Vector3(D,B,C),
        H=new THREE.Vector3(E,F,0),
        I=G.cross(H),
        J=I.dot(r),
        K=J<0?-1:1,
        L=new THREE.Vector3(K*x,K*B,K*C);
        e.vertexTangents[0]=new THREE.Vector4(L.x,L.y,L.z,K);
        x=e.vertexUVs[0],
        y=x.x,
        x=x.y,
        x=e.vertexUVs[1],
        z=x.x,
        x=x.y,
        x=e.vertexUVs[2],
        A=x.x,
        x=x.y,
        x=n-k,
        B=p-l,
        C=q-m,
        D=Math.sqrt(x*x+B*B+C*C),
        x=x/D,
        B=B/D,
        C=C/D,
        D=t*v+r*s+w*x,
        E=t*y+r*z+w*A,
        F=v*y+s*z+x*A,
        G=new THREE.Vector3(D,B,C),
        H=new THREE.Vector3(E,F,0),
        I=G.cross(H),
        J=I.dot(r),
        K=J<0?-1:1,
        L=new THREE.Vector3(K*x,K*B,K*C);
        e.vertexTangents[1]=new THREE.Vector4(L.x,L.y,L.z,K);
        x=e.vertexUVs[0],
        y=x.x,
        x=x.y,
        x=e.vertexUVs[1],
        z=x.x,
        x=x.y,
        x=e.vertexUVs[2],
        A=x.x,
        x=x.y,
        x=h-n,
        B=p-l,
        C=q-m,
        D=Math.sqrt(x*x+B*B+C*C),
        x=x/D,
        B=B/D,
        C=C/D,
        D=t*v+r*s+w*x,
        E=t*y+r*z+w*A,
        F=v*y+s*z+x*A,
        G=new THREE.Vector3(D,B,C),
        H=new THREE.Vector3(E,F,0),
        I=G.cross(H),
        J=I.dot(r),
        K=J<0?-1:1,
        L=new THREE.Vector3(K*x,K*B,K*C);
        e.vertexTangents[2]=new THREE.Vector4(L.x,L.y,L.z,K);
        this.hasTangents=!0
    }
},
# 计算线段之间的距离
computeLineDistances:function(){
    for(var a=0,b=this.vertices,c=0,d=b.length;c<d;c++)
        0<c&&(a+=b[c].distanceTo(b[c-1])),
        this.lineDistances[c]=a
},
# 计算包围盒
computeBoundingBox:function(){
    null===this.boundingBox&&(this.boundingBox=new THREE.Box3);
    this.boundingBox.setFromPoints(this.vertices)
},
# 计算包围球
computeBoundingSphere:function(){
    null===this.boundingSphere&&(this.boundingSphere=new THREE.Sphere);
    this.boundingSphere.setFromPoints(this.vertices)
},
# 合并几何体
merge:function(a,b,c){
    if(!1===a instanceof THREE.Geometry)
        console.error("THREE.Geometry.merge(): geometry not an instance of THREE.Geometry.",a);
    else{
        var d,e=this.vertices.length,f=this.vertices,g=a.vertices,h=this.faces,k=a.faces,n=this.faceVertexUvs[0];
        a=a.faceVertexUvs[0];
        void 0===c&&(c=0);
        void 0!==b&&(d=(new THREE.Matrix3).getNormalMatrix(b));
        for(var p=0,q=g.length;p<q;p++){
            var m=g[p].clone();
            void 0!==b&&m.applyMatrix4(b);
            f.push(m)
        }
        p=0;
        for(q=k.length;p<q;p++){
            var g=k[p],
            r,
            t=g.vertexNormals,
            s=g.vertexColors,
            m=new THREE.Face3(g.a+e,g.b+e,g.c+e);
            m.normal.copy(g.normal);
            void 0!==d&&m.normal.applyMatrix3(d).normalize();
            b=0;
            for(f=t.length;b<f;b++)
                r=t[b].clone(),
                void 0!==d&&r.applyMatrix3(d).normalize(),
                m.vertexNormals.push(r);
            m.color.copy(g.color);
            b=0;
            for(f=s.length;b<f;b++)
                r=s[b],
                m.vertexColors.push(r.clone());
            m.materialIndex=g.materialIndex+c;
            h.push(m)
        }
        p=0;
        for(q=a.length;p<q;p++)
            if(c=a[p],
            d=[],
            void 0!==c){
                b=0;
                for(f=c.length;b<f;b++)
                    d.push(new THREE.Vector2(c[b].x,c[b].y));
                n.push(d)
            }
    }
},
# 合并顶点
mergeVertices:function(){
    var a=
# 初始化变量，定义空对象、数组和一些常量
{},b=[],c=[],d,e=Math.pow(10,4),f,g;
f=0;
# 遍历顶点数组
for(g=this.vertices.length;f<g;f++)
    # 获取当前顶点
    d=this.vertices[f]
    # 将顶点坐标乘以 10000 并转换为字符串，作为唯一标识
    d=Math.round(d.x*e)+"_"+Math.round(d.y*e)+"_"+Math.round(d.z*e)
    # 如果当前标识在对象 a 中不存在
    void 0===a[d]?(a[d]=f,b.push(this.vertices[f]),c[f]=b.length-1):c[f]=c[a[d]]
# 清空数组 a
a=[];
# 遍历面数组
f=0;
for(g=this.faces.length;f<g;f++)
    # 获取当前面
    e=this.faces[f]
    # 将面的顶点索引替换为新的索引
    e.a=c[e.a]
    e.b=c[e.b]
    e.c=c[e.c]
    e=[e.a,e.b,e.c]
    # 如果面有重复的顶点索引
    for(d=0;3>d;d++)
        if(e[d]==e[(d+1)%3]){
            # 将面的索引添加到数组 a 中
            a.push(f)
            break
        }
# 从后往前遍历数组 a
for(f=a.length-1;0<=f;f--)
    # 移除面数组中的重复面
    for(e=a[f],this.faces.splice(e,1),c=0,g=this.faceVertexUvs.length;c<g;c++)
        this.faceVertexUvs[c].splice(e,1)
# 更新顶点数组，返回删除的顶点数量
f=this.vertices.length-b.length;
this.vertices=b;
return f
# 将 BufferGeometry 对象转换为 JSON 格式
},toJSON:function(){
    # 定义内部函数和变量
    function a(a,b,c){
        return c?a|1<<b:a&~(1<<b)
    }
    function b(a){
        var b=a.x.toString()+a.y.toString()+a.z.toString();
        if(void 0!==n[b])return n[b];
        n[b]=k.length/3;
        k.push(a.x,a.y,a.z);
        return n[b]
    }
    function c(a){
        var b=a.r.toString()+a.g.toString()+a.b.toString();
        if(void 0!==q[b])return q[b];
        q[b]=p.length;
        p.push(a.getHex());
        return q[b]
    }
    function d(a){
        var b=a.x.toString()+a.y.toString();
        if(void 0!==r[b])return r[b];
        r[b]=m.length/2;
        m.push(a.x,a.y);
        return r[b]
    }
    # 定义 JSON 对象
    var e={metadata:{version:4,type:"BufferGeometry",generator:"BufferGeometryExporter"},uuid:this.uuid,type:this.type};
    # 如果有名称属性，添加到 JSON 对象中
    ""!==this.name&&(e.name=this.name);
    # 如果有 parameters 属性，添加到 JSON 对象中
    if(void 0!==this.parameters){
        var f=this.parameters,g;
        for(g in f)
            void 0!==f[g]&&(e[g]=f[g]);
        return e
    }
    # 将顶点数组转换为一维数组
    f=[];
    for(g=0;g<this.vertices.length;g++){
        var h=this.vertices[g];
        f.push(h.x,h.y,h.z)
    }
    # 定义一些数组和对象
    var h=[],k=[],n={},p=[],q={},m=[],r={};
    # 遍历面数组
    for(g=0;g<this.faces.length;g++){
        var t=this.faces[g],s=void 0!==this.faceVertexUvs[0][g],u=0<t.normal.length(),
# 初始化变量v，判断顶点法线长度是否大于0
v=0<t.vertexNormals.length,
# 初始化变量y，判断颜色是否不全为白色
y=1!==t.color.r||1!==t.color.g||1!==t.color.b,
# 初始化变量G，判断顶点颜色长度是否大于0
G=0<t.vertexColors.length,
# 初始化变量w为0
w=0,
# 调用函数a，将w赋值为函数a的返回值
w=a(w,0,0),
# 调用函数a，将w赋值为函数a的返回值
w=a(w,1,!1),
# 调用函数a，将w赋值为函数a的返回值
w=a(w,2,!1),
# 调用函数a，将w赋值为函数a的返回值
w=a(w,3,s),
# 调用函数a，将w赋值为函数a的返回值
w=a(w,4,u),
# 调用函数a，将w赋值为函数a的返回值
w=a(w,5,v),
# 调用函数a，将w赋值为函数a的返回值
w=a(w,6,y),
# 调用函数a，将w赋值为函数a的返回值
w=a(w,7,G);
# 将w添加到数组h中
h.push(w);
# 将t.a、t.b、t.c添加到数组h中
h.push(t.a,t.b,t.c);
# 如果s存在，则将s的UV坐标添加到数组h中
s&&(s=this.faceVertexUvs[0][g],h.push(d(s[0]),d(s[1]),d(s[2])));
# 如果u存在，则将t.normal添加到数组h中
u&&h.push(b(t.normal));
# 如果v存在，则将t.vertexNormals添加到数组h中
v&&(u=t.vertexNormals,h.push(b(u[0]),b(u[1]),b(u[2])));
# 如果y存在，则将t.color添加到数组h中
y&&h.push(c(t.color));
# 如果G存在，则将t.vertexColors添加到数组h中
G&&(t=t.vertexColors,h.push(c(t[0]),c(t[1]),c(t[2])));
# 初始化e.data为一个空对象
e.data={};
# 将顶点坐标数组f添加到e.data.vertices中
e.data.vertices=f;
# 将法线数组k添加到e.data.normals中
e.data.normals=k;
# 如果颜色数组p长度大于0，则将颜色数组p添加到e.data.colors中
0<p.length&&(e.data.colors=p);
# 如果UV坐标数组m长度大于0，则将UV坐标数组m添加到e.data.uvs中
0<m.length&&(e.data.uvs=[m]);
# 将面数组h添加到e.data.faces中
e.data.faces=h;
# 返回e对象
return e
# 克隆函数，复制一个新的THREE.Geometry对象
clone:function(){
    # 创建一个新的THREE.Geometry对象
    for(var a=new THREE.Geometry,b=this.vertices,c=0,d=b.length;c<d;c++)a.vertices.push(b[c].clone());
    # 复制面数组
    b=this.faces;c=0;for(d=b.length;c<d;c++)a.faces.push(b[c].clone());
    # 复制UV坐标数组
    b=this.faceVertexUvs[0];c=0;for(d=b.length;c<d;c++){for(var e=b[c],f=[],g=0,h=e.length;g<h;g++)f.push(new THREE.Vector2(e[g].x,e[g].y));a.faceVertexUvs[0].push(f)}
    # 返回新的THREE.Geometry对象
    return a
},
# 释放函数，触发dispose事件
dispose:function(){
    this.dispatchEvent({type:"dispose"})
}
# 将THREE.EventDispatcher的原型方法应用到THREE.Geometry的原型上
THREE.EventDispatcher.prototype.apply(THREE.Geometry.prototype);
# 初始化THREE.Geometry对象的ID计数
THREE.GeometryIdCount=0;
# 创建相机对象
THREE.Camera=function(){
    THREE.Object3D.call(this);
    this.type="Camera";
    this.matrixWorldInverse=new THREE.Matrix4;
    this.projectionMatrix=new THREE.Matrix4
};
# 将相机对象的原型设置为THREE.Object3D的实例
THREE.Camera.prototype=Object.create(THREE.Object3D.prototype);
# 获取相机的世界方向
THREE.Camera.prototype.getWorldDirection=function(){
    var a=new THREE.Quaternion;
    return function(b){
        b=b||new THREE.Vector3;
        this.getWorldQuaternion(a);
        return b.set(0,0,-1).applyQuaternion(a)
    }
}();
# 设置相机的朝向
THREE.Camera.prototype.lookAt=function(){
    var a=new THREE.Matrix4;
    return function(b){
        a.lookAt(this.position,b,this.up);
        this.quaternion.setFromRotationMatrix(a)
    }
}();
# 克隆相机对象
THREE.Camera.prototype.clone=function(a){
    void 0===a&&(a=new THREE.Camera);
    THREE.Object3D.prototype.clone.call(this,a);
    a.matrixWorldInverse.copy(this.matrixWorldInverse);
    a.projectionMatrix.copy(this.projectionMatrix);
    return a
};
# 创建 CubeCamera 类，继承自 Object3D 类
THREE.CubeCamera=function(a,b,c){THREE.Object3D.call(this);this.type="CubeCamera";
# 创建透视相机对象 d，设置视场角为 90 度，宽高比为 1，近裁剪面为 a，远裁剪面为 b
var d=new THREE.PerspectiveCamera(90,1,a,b);
# 设置相机的上方向
d.up.set(0,-1,0);
# 设置相机的观察点
d.lookAt(new THREE.Vector3(1,0,0));
# 将相机对象 d 添加到 CubeCamera 对象中
this.add(d);
# 创建透视相机对象 e，设置视场角为 90 度，宽高比为 1，近裁剪面为 a，远裁剪面为 b
var e=new THREE.PerspectiveCamera(90,1,a,b);
# 设置相机的上方向
e.up.set(0,-1,0);
# 设置相机的观察点
e.lookAt(new THREE.Vector3(-1,0,0));
# 将相机对象 e 添加到 CubeCamera 对象中
this.add(e);
# 创建透视相机对象 f，设置视场角为 90 度，宽高比为 1，近裁剪面为 a，远裁剪面为 b
var f=new THREE.PerspectiveCamera(90,1,a,b);
# 设置相机的上方向
f.up.set(0,0,1);
# 设置相机的观察点
f.lookAt(new THREE.Vector3(0,1,0));
# 将相机对象 f 添加到 CubeCamera 对象中
this.add(f);
# 创建透视相机对象 g，设置视场角为 90 度，宽高比为 1，近裁剪面为 a，远裁剪面为 b
var g=new THREE.PerspectiveCamera(90,1,a,b);
# 设置相机的上方向
g.up.set(0,0,-1);
# 设置相机的观察点
g.lookAt(new THREE.Vector3(0,-1,0));
# 将相机对象 g 添加到 CubeCamera 对象中
this.add(g);
# 创建透视相机对象 h，设置视场角为 90 度，宽高比为 1，近裁剪面为 a，远裁剪面为 b
var h=new THREE.PerspectiveCamera(90,1,a,b);
# 设置相机的上方向
h.up.set(0,-1,0);
# 设置相机的观察点
h.lookAt(new THREE.Vector3(0,0,1));
# 将相机对象 h 添加到 CubeCamera 对象中
this.add(h);
# 创建透视相机对象 k，设置视场角为 90 度，宽高比为 1，近裁剪面为 a，远裁剪面为 b
var k=new THREE.PerspectiveCamera(90,1,a,b);
# 设置相机的上方向
k.up.set(0,-1,0);
# 设置相机的观察点
k.lookAt(new THREE.Vector3(0,0,-1));
# 将相机对象 k 添加到 CubeCamera 对象中
this.add(k);
# 创建渲染目标对象 renderTarget，使用 WebGLRenderTargetCube 类创建，设置宽高为 c，格式为 RGBFormat，放大和缩小过滤器为 LinearFilter
this.renderTarget=new THREE.WebGLRenderTargetCube(c,c,{format:THREE.RGBFormat,magFilter:THREE.LinearFilter,minFilter:THREE.LinearFilter});
# 更新 CubeCamera 对象的立方体贴图
this.updateCubeMap=function(a,b){
    var c=this.renderTarget,
    m=c.generateMipmaps;
    c.generateMipmaps=!1;
    c.activeCubeFace=0;
    a.render(b,d,c);
    c.activeCubeFace=1;
    a.render(b,e,c);
    c.activeCubeFace=2;
    a.render(b,f,c);
    c.activeCubeFace=3;
    a.render(b,g,c);
    c.activeCubeFace=4;
    a.render(b,h,c);
    c.generateMipmaps=m;
    c.activeCubeFace=5;
    a.render(b,k,c)
}};
# 将 CubeCamera 的原型设置为 Object3D 的实例
THREE.CubeCamera.prototype=Object.create(THREE.Object3D.prototype);
# 创建 OrthographicCamera 类，继承自 Camera 类
THREE.OrthographicCamera=function(a,b,c,d,e,f){THREE.Camera.call(this);this.type="OrthographicCamera";
# 设置相机的缩放比例
this.zoom=1;
# 设置相机的左、右、上、下、近、远裁剪面
this.left=a;
this.right=b;
this.top=c;
this.bottom=d;
this.near=void 0!==e?e:.1;
this.far=void 0!==f?f:2E3;
# 更新投影矩阵
this.updateProjectionMatrix()};
# 将 OrthographicCamera 的原型设置为 Camera 的实例
THREE.OrthographicCamera.prototype=Object.create(THREE.Camera.prototype);
# 更新投影矩阵
THREE.OrthographicCamera.prototype.updateProjectionMatrix=function(){
    var a=(this.right-this.left)/(2*this.zoom),
    b=(this.top-this.bottom)/(2*this.zoom),
    c=(this.right+this.left)/2,
    d=(this.top+this.bottom)/2;
    this.projectionMatrix.makeOrthographic(c-a,c+a,d+b,d-b,this.near,this.far)
};
// 创建 OrthographicCamera 的克隆对象
THREE.OrthographicCamera.prototype.clone=function(){
    var a=new THREE.OrthographicCamera;
    // 调用 Camera 的 clone 方法，克隆相机对象
    THREE.Camera.prototype.clone.call(this,a);
    a.zoom=this.zoom;
    a.left=this.left;
    a.right=this.right;
    a.top=this.top;
    a.bottom=this.bottom;
    a.near=this.near;
    a.far=this.far;
    a.projectionMatrix.copy(this.projectionMatrix);
    return a;
};

// 创建 PerspectiveCamera 对象
THREE.PerspectiveCamera=function(a,b,c,d){
    // 调用 Camera 构造函数
    THREE.Camera.call(this);
    this.type="PerspectiveCamera";
    this.zoom=1;
    this.fov=void 0!==a?a:50;
    this.aspect=void 0!==b?b:1;
    this.near=void 0!==c?c:.1;
    this.far=void 0!==d?d:2E3;
    this.updateProjectionMatrix();
};

// 继承 Camera 的原型
THREE.PerspectiveCamera.prototype=Object.create(THREE.Camera.prototype);

// 设置透视相机的焦距
THREE.PerspectiveCamera.prototype.setLens=function(a,b){
    void 0===b&&(b=24);
    this.fov=2*THREE.Math.radToDeg(Math.atan(b/(2*a)));
    this.updateProjectionMatrix();
};

// 设置透视相机的视图偏移
THREE.PerspectiveCamera.prototype.setViewOffset=function(a,b,c,d,e,f){
    this.fullWidth=a;
    this.fullHeight=b;
    this.x=c;
    this.y=d;
    this.width=e;
    this.height=f;
    this.updateProjectionMatrix();
};

// 更新透视相机的投影矩阵
THREE.PerspectiveCamera.prototype.updateProjectionMatrix=function(){
    var a=THREE.Math.radToDeg(2*Math.atan(Math.tan(.5*THREE.Math.degToRad(this.fov))/this.zoom));
    if(this.fullWidth){
        var b=this.fullWidth/this.fullHeight,
            a=Math.tan(THREE.Math.degToRad(.5*a))*this.near,
            c=-a,
            d=b*c,
            b=Math.abs(b*a-d),
            c=Math.abs(a-c);
        this.projectionMatrix.makeFrustum(d+this.x*b/this.fullWidth,d+(this.x+this.width)*b/this.fullWidth,a-(this.y+this.height)*c/this.fullHeight,a-this.y*c/this.fullHeight,this.near,this.far);
    }else{
        this.projectionMatrix.makePerspective(a,this.aspect,this.near,this.far);
    }
};

// 创建 PerspectiveCamera 的克隆对象
THREE.PerspectiveCamera.prototype.clone=function(){
    var a=new THREE.PerspectiveCamera;
    THREE.Camera.prototype.clone.call(this,a);
    a.zoom=this.zoom;
    a.fov=this.fov;
    a.aspect=this.aspect;
    a.near=this.near;
    a.far=this.far;
    a.projectionMatrix.copy(this.projectionMatrix);
    return a;
};

// 创建光源对象
THREE.Light=function(a){
    THREE.Object3D.call(this);
    this.type="Light";
    this.color=new THREE.Color(a);
};

// 继承 Object3D 的原型
THREE.Light.prototype=Object.create(THREE.Object3D.prototype);
# 克隆方法，用于克隆光源对象
THREE.Light.prototype.clone=function(a){
    # 如果未传入克隆对象，则创建一个新的光源对象
    void 0===a&&(a=new THREE.Light);
    # 调用 Object3D 的克隆方法，克隆光源对象
    THREE.Object3D.prototype.clone.call(this,a);
    # 复制颜色属性到克隆对象
    a.color.copy(this.color);
    # 返回克隆对象
    return a
};

# 环境光类，继承自光源类
THREE.AmbientLight=function(a){
    # 调用光源类的构造函数
    THREE.Light.call(this,a);
    # 设置光源类型为环境光
    this.type="AmbientLight"
};
# 继承光源类的原型方法
THREE.AmbientLight.prototype=Object.create(THREE.Light.prototype);
# 克隆方法，用于克隆环境光对象
THREE.AmbientLight.prototype.clone=function(){
    # 创建一个新的环境光对象
    var a=new THREE.AmbientLight;
    # 调用光源类的克隆方法，克隆环境光对象
    THREE.Light.prototype.clone.call(this,a);
    # 返回克隆对象
    return a
};

# 区域光类，继承自光源类
THREE.AreaLight=function(a,b){
    # 调用光源类的构造函数
    THREE.Light.call(this,a);
    # 设置光源类型为区域光
    this.type="AreaLight";
    # 设置默认的法线向量、右向量、强度、高度、宽度、常数衰减、线性衰减和二次衰减
    this.normal=new THREE.Vector3(0,-1,0);
    this.right=new THREE.Vector3(1,0,0);
    this.intensity=void 0!==b?b:1;
    this.height=this.width=1;
    this.constantAttenuation=1.5;
    this.linearAttenuation=.5;
    this.quadraticAttenuation=.1
};
# 继承光源类的原型方法
THREE.AreaLight.prototype=Object.create(THREE.Light.prototype);

# 方向光类，继承自光源类
THREE.DirectionalLight=function(a,b){
    # 调用光源类的构造函数
    THREE.Light.call(this,a);
    # 设置光源类型为方向光
    this.type="DirectionalLight";
    # 设置位置、目标、强度、阴影相关属性
    this.position.set(0,1,0);
    this.target=new THREE.Object3D;
    this.intensity=void 0!==b?b:1;
    this.onlyShadow=this.castShadow=!1;
    this.shadowCameraNear=50;
    this.shadowCameraFar=5E3;
    this.shadowCameraLeft=-500;
    this.shadowCameraTop=this.shadowCameraRight=500;
    this.shadowCameraBottom=-500;
    this.shadowCameraVisible=!1;
    this.shadowBias=0;
    this.shadowDarkness=.5;
    this.shadowMapHeight=this.shadowMapWidth=512;
    this.shadowCascade=!1;
    this.shadowCascadeOffset=new THREE.Vector3(0,0,-1E3);
    this.shadowCascadeCount=2;
    this.shadowCascadeBias=[0,0,0];
    this.shadowCascadeWidth=[512,512,512];
    this.shadowCascadeHeight=[512,512,512];
    this.shadowCascadeNearZ=[-1,.99,.998];
    this.shadowCascadeFarZ=[.99,.998,1];
    this.shadowCascadeArray=[];
    this.shadowMatrix=this.shadowCamera=this.shadowMapSize=this.shadowMap=null
};
# 继承光源类的原型方法
THREE.DirectionalLight.prototype=Object.create(THREE.Light.prototype);
// 克隆方法，用于克隆 DirectionalLight 对象
THREE.DirectionalLight.prototype.clone=function(){
    // 创建一个新的 DirectionalLight 对象
    var a=new THREE.DirectionalLight;
    // 调用父类 Light 的克隆方法，将当前对象克隆到新对象上
    THREE.Light.prototype.clone.call(this,a);
    // 克隆目标对象
    a.target=this.target.clone();
    // 克隆光照强度
    a.intensity=this.intensity;
    // 克隆是否产生阴影
    a.castShadow=this.castShadow;
    // 克隆是否只产生阴影
    a.onlyShadow=this.onlyShadow;
    // 克隆阴影相机近平面
    a.shadowCameraNear=this.shadowCameraNear;
    // 克隆阴影相机远平面
    a.shadowCameraFar=this.shadowCameraFar;
    // 克隆阴影相机左平面
    a.shadowCameraLeft=this.shadowCameraLeft;
    // 克隆阴影相机右平面
    a.shadowCameraRight=this.shadowCameraRight;
    // 克隆阴影相机顶平面
    a.shadowCameraTop=this.shadowCameraTop;
    // 克隆阴影相机底平面
    a.shadowCameraBottom=this.shadowCameraBottom;
    // 克隆阴影相机是否可见
    a.shadowCameraVisible=this.shadowCameraVisible;
    // 克隆阴影偏移
    a.shadowBias=this.shadowBias;
    // 克隆阴影深度
    a.shadowDarkness=this.shadowDarkness;
    // 克隆阴影贴图宽度
    a.shadowMapWidth=this.shadowMapWidth;
    // 克隆阴影贴图高度
    a.shadowMapHeight=this.shadowMapHeight;
    // 克隆阴影级联
    a.shadowCascade=this.shadowCascade;
    // 克隆阴影级联偏移
    a.shadowCascadeOffset.copy(this.shadowCascadeOffset);
    // 克隆阴影级联数量
    a.shadowCascadeCount=this.shadowCascadeCount;
    // 克隆阴影级联偏移
    a.shadowCascadeBias=this.shadowCascadeBias.slice(0);
    // 克隆阴影级联宽度
    a.shadowCascadeWidth=this.shadowCascadeWidth.slice(0);
    // 克隆阴影级联高度
    a.shadowCascadeHeight=this.shadowCascadeHeight.slice(0);
    // 克隆阴影级联近平面
    a.shadowCascadeNearZ=this.shadowCascadeNearZ.slice(0);
    // 克隆阴影级联远平面
    a.shadowCascadeFarZ=this.shadowCascadeFarZ.slice(0);
    // 返回克隆后的对象
    return a;
};

// 半球光源构造函数
THREE.HemisphereLight=function(a,b,c){
    // 调用父类 Light 的构造函数
    THREE.Light.call(this,a);
    // 设置光源类型为 HemisphereLight
    this.type="HemisphereLight";
    // 设置光源位置
    this.position.set(0,100,0);
    // 设置地面颜色
    this.groundColor=new THREE.Color(b);
    // 设置光照强度
    this.intensity=void 0!==c?c:1;
};
// 继承 Light 类的属性和方法
THREE.HemisphereLight.prototype=Object.create(THREE.Light.prototype);
// 克隆方法，用于克隆 HemisphereLight 对象
THREE.HemisphereLight.prototype.clone=function(){
    // 创建一个新的 HemisphereLight 对象
    var a=new THREE.HemisphereLight;
    // 调用父类 Light 的克隆方法，将当前对象克隆到新对象上
    THREE.Light.prototype.clone.call(this,a);
    // 克隆地面颜色
    a.groundColor.copy(this.groundColor);
    // 克隆光照强度
    a.intensity=this.intensity;
    // 返回克隆后的对象
    return a;
};

// 点光源构造函数
THREE.PointLight=function(a,b,c){
    // 调用父类 Light 的构造函数
    THREE.Light.call(this,a);
    // 设置光源类型为 PointLight
    this.type="PointLight";
    // 设置光照强度
    this.intensity=void 0!==b?b:1;
    // 设置光源距离
    this.distance=void 0!==c?c:0;
};
// 继承 Light 类的属性和方法
THREE.PointLight.prototype=Object.create(THREE.Light.prototype);
// 克隆方法，用于克隆 PointLight 对象
THREE.PointLight.prototype.clone=function(){
    // 创建一个新的 PointLight 对象
    var a=new THREE.PointLight;
    // 调用父类 Light 的克隆方法，将当前对象克隆到新对象上
    THREE.Light.prototype.clone.call(this,a);
    // 克隆光照强度
    a.intensity=this.intensity;
    // 克隆光源距离
    a.distance=this.distance;
    // 返回克隆后的对象
    return a;
};
# 创建 SpotLight 类，继承自 Light 类
THREE.SpotLight=function(a,b,c,d,e){THREE.Light.call(this,a);
    # 设置 SpotLight 的类型
    this.type="SpotLight";
    # 设置光源位置
    this.position.set(0,1,0);
    # 创建一个新的 Object3D 作为光源的目标
    this.target=new THREE.Object3D;
    # 设置光源的强度
    this.intensity=void 0!==b?b:1;
    # 设置光源的距离
    this.distance=void 0!==c?c:0;
    # 设置光源的角度
    this.angle=void 0!==d?d:Math.PI/3;
    # 设置光源的衰减指数
    this.exponent=void 0!==e?e:10;
    # 设置是否只产生阴影
    this.onlyShadow=this.castShadow=!1;
    # 设置阴影相机的近裁剪面
    this.shadowCameraNear=50;
    # 设置阴影相机的远裁剪面
    this.shadowCameraFar=5E3;
    # 设置阴影相机的视场角
    this.shadowCameraFov=50;
    # 设置阴影相机是否可见
    this.shadowCameraVisible=!1;
    # 设置阴影偏移
    this.shadowBias=0;
    # 设置阴影的深度
    this.shadowDarkness=.5;
    # 设置阴影贴图的宽度和高度
    this.shadowMapHeight=this.shadowMapWidth=512;
    # 设置阴影矩阵
    this.shadowMatrix=this.shadowCamera=this.shadowMapSize=this.shadowMap=null};
# 将 SpotLight 的原型设置为继承自 Light 的原型
THREE.SpotLight.prototype=Object.create(THREE.Light.prototype);
# 克隆 SpotLight 对象
THREE.SpotLight.prototype.clone=function(){var a=new THREE.SpotLight;
    # 调用 Light 原型的 clone 方法
    THREE.Light.prototype.clone.call(this,a);
    # 克隆光源的目标
    a.target=this.target.clone();
    # 克隆光源的强度
    a.intensity=this.intensity;
    # 克隆光源的距离
    a.distance=this.distance;
    # 克隆光源的角度
    a.angle=this.angle;
    # 克隆光源的衰减指数
    a.exponent=this.exponent;
    # 克隆是否只产生阴影
    a.castShadow=this.castShadow;
    a.onlyShadow=this.onlyShadow;
    # 克隆阴影相机的近裁剪面
    a.shadowCameraNear=this.shadowCameraNear;
    # 克隆阴影相机的远裁剪面
    a.shadowCameraFar=this.shadowCameraFar;
    # 克隆阴影相机的视场角
    a.shadowCameraFov=this.shadowCameraFov;
    # 克隆阴影相机是否可见
    a.shadowCameraVisible=this.shadowCameraVisible;
    # 克隆阴影偏移
    a.shadowBias=this.shadowBias;
    # 克隆阴影的深度
    a.shadowDarkness=this.shadowDarkness;
    # 克隆阴影贴图的宽度和高度
    a.shadowMapWidth=this.shadowMapWidth;
    a.shadowMapHeight=this.shadowMapHeight;
    return a};
# 缓存类
THREE.Cache=function(){this.files={}};
# 缓存类的原型
THREE.Cache.prototype={constructor:THREE.Cache,
    # 添加文件到缓存
    add:function(a,b){this.files[a]=b},
    # 从缓存中获取文件
    get:function(a){return this.files[a]},
    # 从缓存中移除文件
    remove:function(a){delete this.files[a]},
    # 清空缓存
    clear:function(){this.files={}}};
# 加载器类
THREE.Loader=function(a){this.statusDomElement=(this.showStatus=a)?THREE.Loader.prototype.addStatusElement():null;
    # 创建图像加载器
    this.imageLoader=new THREE.ImageLoader;
    # 加载开始时的回调函数
    this.onLoadStart=function(){};
    # 加载进度时的回调函数
    this.onLoadProgress=function(){};
    # 加载完成时的回调函数
    this.onLoadComplete=function(){}};
# 定义 THREE.Loader 的原型对象
THREE.Loader.prototype={
    constructor:THREE.Loader,
    crossOrigin:void 0,
    # 添加状态元素的方法
    addStatusElement:function(){
        var a=document.createElement("div");
        a.style.position="absolute";
        a.style.right="0px";
        a.style.top="0px";
        a.style.fontSize="0.8em";
        a.style.textAlign="left";
        a.style.background="rgba(0,0,0,0.25)";
        a.style.color="#fff";
        a.style.width="120px";
        a.style.padding="0.5em 0.5em 0.5em 0.5em";
        a.style.zIndex=1E3;
        a.innerHTML="Loading ...";
        return a
    },
    # 更新加载进度的方法
    updateProgress:function(a){
        var b="Loaded ",
        b=a.total?b+((100*a.loaded/a.total).toFixed(0)+"%"):b+((a.loaded/1024).toFixed(2)+" KB");
        this.statusDomElement.innerHTML=b
    },
    # 提取 URL 基础路径的方法
    extractUrlBase:function(a){
        a=a.split("/");
        if(1===a.length) return "./";
        a.pop();
        return a.join("/")+"/"
    },
    # 初始化材质的方法
    initMaterials:function(a,b){
        for(var c=[],d=0;d<a.length;++d) c[d]=this.createMaterial(a[d],b);
        return c
    },
    # 检查是否需要切线的方法
    needsTangents:function(a){
        for(var b=0,c=a.length;b<c;b++)
            if(a[b]instanceof THREE.ShaderMaterial) return!0;
        return!1
    },
    # 创建材质的方法
    createMaterial:function(a,b){
        function c(a){
            a=Math.log(a)/Math.LN2;
            return Math.pow(2,Math.round(a))
        }
        function d(a,d,e,g,h,k,s){
            var u=b+e,
            v,
            y=THREE.Loader.Handlers.get(u);
            null!==y?v=y.load(u):(v=new THREE.Texture,
            y=f.imageLoader,
            y.crossOrigin=f.crossOrigin,
            y.load(u,function(a){
                if(!1===THREE.Math.isPowerOfTwo(a.width)||!1===THREE.Math.isPowerOfTwo(a.height)){
                    var b=c(a.width),
                    d=c(a.height),
                    e=document.createElement("canvas");
                    e.width=b;
                    e.height=d;
                    e.getContext("2d").drawImage(a,0,0,b,d);
                    v.image=e
                }else v.image=a;
                v.needsUpdate=!0
            }));
            v.sourceFile=e;
            g&&(v.repeat.set(g[0],g[1]),
            1!==g[0]&&(v.wrapS=THREE.RepeatWrapping),
1!==g[1]&&(v.wrapT=THREE.RepeatWrapping));h&&v.offset.set(h[0],h[1]);k&&(e={repeat:THREE.RepeatWrapping,mirror:THREE.MirroredRepeatWrapping},void 0!==e[k[0]]&&(v.wrapS=e[k[0]]),void 0!==e[k[1]]&&(v.wrapT=e[k[1]]));s&&(v.anisotropy=s);a[d]=v}

function e(a){
    return(255*a[0]<<16)+(255*a[1]<<8)+255*a[2]
}

var f=this,g="MeshLambertMaterial",h={color:15658734,opacity:1,map:null,lightMap:null,normalMap:null,bumpMap:null,wireframe:!1};

if(a.shading){
    var k=a.shading.toLowerCase();
    "phong"===k?g="MeshPhongMaterial":"basic"===k&&(g="MeshBasicMaterial")
}

if(void 0!==a.blending&&void 0!==THREE[a.blending]&&(h.blending=THREE[a.blending]));

if(void 0!==a.transparent||1>a.opacity)h.transparent=a.transparent;

if(void 0!==a.depthTest&&(h.depthTest=a.depthTest));

if(void 0!==a.depthWrite&&(h.depthWrite=a.depthWrite));

if(void 0!==a.visible&&(h.visible=a.visible));

if(void 0!==a.flipSided&&(h.side=THREE.BackSide));

if(void 0!==a.doubleSided&&(h.side=THREE.DoubleSide));

if(void 0!==a.wireframe&&(h.wireframe=a.wireframe));

if(void 0!==a.vertexColors&&("face"===a.vertexColors?h.vertexColors=THREE.FaceColors:a.vertexColors&&(h.vertexColors=THREE.VertexColors)));

a.colorDiffuse?h.color=e(a.colorDiffuse):a.DbgColor&&(h.color=a.DbgColor);

a.colorSpecular&&(h.specular=e(a.colorSpecular));

a.colorAmbient&&(h.ambient=e(a.colorAmbient));

a.colorEmissive&&(h.emissive=e(a.colorEmissive));

a.transparency&&(h.opacity=a.transparency);

a.specularCoef&&(h.shininess=a.specularCoef);

a.mapDiffuse&&b&&d(h,"map",a.mapDiffuse,a.mapDiffuseRepeat,a.mapDiffuseOffset,a.mapDiffuseWrap);
# 设置材质的漫反射贴图
a.mapDiffuse&&b&&d(h,"map",a.mapDiffuse,a.mapDiffuseRepeat,a.mapDiffuseOffset,a.mapDiffuseWrap,a.mapDiffuseAnisotropy);
# 设置材质的光照贴图
a.mapLight&&b&&d(h,"lightMap",a.mapLight,a.mapLightRepeat,a.mapLightOffset,a.mapLightWrap,a.mapLightAnisotropy);
# 设置材质的凹凸贴图
a.mapBump&&b&&d(h,"bumpMap",a.mapBump,a.mapBumpRepeat,a.mapBumpOffset,a.mapBumpWrap,a.mapBumpAnisotropy);
# 设置材质的法线贴图
a.mapNormal&&b&&d(h,"normalMap",a.mapNormal,a.mapNormalRepeat,a.mapNormalOffset,a.mapNormalWrap,a.mapNormalAnisotropy);
# 设置材质的高光贴图
a.mapSpecular&&b&&d(h,"specularMap",a.mapSpecular,a.mapSpecularRepeat,a.mapSpecularOffset,a.mapSpecularWrap,a.mapSpecularAnisotropy);
# 设置材质的透明度贴图
a.mapAlpha&&b&&d(h,"alphaMap",a.mapAlpha,a.mapAlphaRepeat,a.mapAlphaOffset,a.mapAlphaWrap,a.mapAlphaAnisotropy);
# 设置材质的凹凸贴图的缩放
a.mapBumpScale&&(h.bumpScale=a.mapBumpScale);
# 如果有法线贴图，则创建 ShaderLib.normalmap 和对应的 Uniforms
a.mapNormal?(g=THREE.ShaderLib.normalmap,k=THREE.UniformsUtils.clone(g.uniforms),k.tNormal.value=h.normalMap,a.mapNormalFactor&&k.uNormalScale.value.set(a.mapNormalFactor,a.mapNormalFactor),h.map&&(k.tDiffuse.value=h.map,k.enableDiffuse.value=!0),h.specularMap&&(k.tSpecular.value=h.specularMap,k.enableSpecular.value=!0),h.lightMap&&(k.tAO.value=h.lightMap,k.enableAO.value=!0),k.diffuse.value.setHex(h.color),k.specular.value.setHex(h.specular),k.ambient.value.setHex(h.ambient),k.shininess.value=h.shininess,void 0!==h.opacity&&(k.opacity.value=h.opacity),g=new THREE.ShaderMaterial({fragmentShader:g.fragmentShader,vertexShader:g.vertexShader,uniforms:k,lights:!0,fog:!0}),h.transparent&&(g.transparent=!0)):g=new THREE[g](h);
# 如果有 DbgName 属性，则设置材质的名称
void 0!==a.DbgName&&(g.name=a.DbgName);
# 返回创建的材质
return g}};
# 设置加载器的处理程序
THREE.Loader.Handlers={handlers:[],add:function(a,b){this.handlers.push(a,b)},get:function(a){for(var b=0,c=this.handlers.length;b<c;b+=2){var d=this.handlers[b+1];if(this.handlers[b].test(a))return d}return null}};
# 定义 XHRLoader 类
THREE.XHRLoader=function(a){this.cache=new THREE.Cache;this.manager=void 0!==a?a:THREE.DefaultLoadingManager};
# 定义 THREE.XHRLoader 对象的原型
THREE.XHRLoader.prototype={
    # 构造函数
    constructor:THREE.XHRLoader,
    # 加载函数，参数包括要加载的资源地址、加载成功的回调函数、加载进度的回调函数、加载失败的回调函数
    load:function(a,b,c,d){
        # 保存 this 指针
        var e=this,
        # 从缓存中获取资源
        f=e.cache.get(a);
        # 如果缓存中存在资源，则调用加载成功的回调函数并返回
        void 0!==f?b&&b(f):
        # 否则创建 XMLHttpRequest 对象
        (f=new XMLHttpRequest,
        # 打开 GET 请求，异步
        f.open("GET",a,!0),
        # 监听加载成功事件，将资源添加到缓存中，调用加载成功的回调函数，结束加载
        f.addEventListener("load",function(c){e.cache.add(a,this.response);b&&b(this.response);e.manager.itemEnd(a)},!1),
        # 如果有加载进度的回调函数，则监听进度事件
        void 0!==c&&f.addEventListener("progress",function(a){c(a)},!1),
        # 如果有加载失败的回调函数，则监听错误事件
        void 0!==d&&f.addEventListener("error",function(a){d(a)},!1),
        # 如果有跨域设置，则设置跨域属性
        void 0!==this.crossOrigin&&(f.crossOrigin=this.crossOrigin),
        # 如果有响应类型设置，则设置响应类型
        void 0!==this.responseType&&(f.responseType=this.responseType),
        # 发送请求，开始加载
        f.send(null),
        # 调用加载开始的回调函数
        e.manager.itemStart(a))
    },
    # 设置响应类型
    setResponseType:function(a){
        this.responseType=a
    },
    # 设置跨域属性
    setCrossOrigin:function(a){
        this.crossOrigin=a
    }
};

# 定义 THREE.ImageLoader 对象
THREE.ImageLoader=function(a){
    # 创建缓存
    this.cache=new THREE.Cache;
    # 设置加载管理器
    this.manager=void 0!==a?a:THREE.DefaultLoadingManager
};

# 定义 THREE.ImageLoader 对象的原型
THREE.ImageLoader.prototype={
    # 构造函数
    constructor:THREE.ImageLoader,
    # 加载函数，参数包括要加载的图片地址、加载成功的回调函数、加载进度的回调函数、加载失败的回调函数
    load:function(a,b,c,d){
        # 保存 this 指针
        var e=this,
        # 从缓存中获取图片
        f=e.cache.get(a);
        # 如果缓存中存在图片，则调用加载成功的回调函数并返回
        if(void 0!==f)b(f);
        else 
            # 否则创建图片对象
            f=document.createElement("img"),
            # 如果有加载成功的回调函数，则监听加载成功事件
            void 0!==b&&f.addEventListener("load",function(c){e.cache.add(a,this);b(this);e.manager.itemEnd(a)},!1),
            # 如果有加载进度的回调函数，则监听进度事件
            void 0!==c&&f.addEventListener("progress",function(a){c(a)},!1),
            # 如果有加载失败的回调函数，则监听错误事件
            void 0!==d&&f.addEventListener("error",function(a){d(a)},!1),
            # 如果有跨域设置，则设置跨域属性
            void 0!==this.crossOrigin&&(f.crossOrigin=this.crossOrigin),
            # 设置图片地址，开始加载
            f.src=a,
            # 调用加载开始的回调函数
            e.manager.itemStart(a),
            # 返回图片对象
            f
    },
    # 设置跨域属性
    setCrossOrigin:function(a){
        this.crossOrigin=a
    }
};

# 定义 THREE.JSONLoader 对象
THREE.JSONLoader=function(a){
    # 调用父类的构造函数
    THREE.Loader.call(this,a);
    # 设置跨域属性为 false
    this.withCredentials=!1
};

# 设置 THREE.JSONLoader 对象的原型为 THREE.Loader 对象的原型
THREE.JSONLoader.prototype=Object.create(THREE.Loader.prototype);

# 加载函数，参数包括要加载的 JSON 地址、加载成功的回调函数、基础 URL
THREE.JSONLoader.prototype.load=function(a,b,c){
    # 如果传入的 c 是字符串，则使用它作为基础 URL，否则使用默认的基础 URL
    c=c&&"string"===typeof c?c:this.extractUrlBase(a);
    # 调用加载开始的回调函数
    this.onLoadStart();
    # 发起 AJAX 请求加载 JSON 数据
    this.loadAjaxJSON(this,a,b,c)
};
# 通过 AJAX 加载 JSON 文件，并解析其中的内容
THREE.JSONLoader.prototype.loadAjaxJSON=function(a,b,c,d,e){
    # 创建 XMLHttpRequest 对象
    var f=new XMLHttpRequest,g=0;
    # 监听状态变化
    f.onreadystatechange=function(){
        # 当请求完成时
        if(f.readyState===f.DONE)
            # 如果状态码为 200 或者 0
            if(200===f.status||0===f.status){
                # 如果有响应内容
                if(f.responseText){
                    # 解析 JSON 响应内容
                    var h=JSON.parse(f.responseText);
                    # 如果 metadata 存在且类型为 scene
                    if(void 0!==h.metadata&&"scene"===h.metadata.type){
                        # 输出错误信息
                        console.error('THREE.JSONLoader: "'+b+'" seems to be a Scene. Use THREE.SceneLoader instead.');
                        return
                    }
                    # 调用 parse 方法解析内容
                    h=a.parse(h,d);
                    # 调用回调函数，传入几何体和材质
                    c(h.geometry,h.materials)
                }else
                    # 输出错误信息
                    console.error('THREE.JSONLoader: "'+b+'" seems to be unreachable or the file is empty.');
                # 调用 onLoadComplete 方法
                a.onLoadComplete()
            }else
                # 输出错误信息
                console.error("THREE.JSONLoader: Couldn't load \""+b+'" ('+f.status+")");
        else if(f.readyState===f.LOADING)
            # 如果正在加载中，且存在进度回调函数
            e&&(0===g&&(g=f.getResponseHeader("Content-Length")),e({total:g,loaded:f.responseText.length}));
        else if(f.readyState===f.HEADERS_RECEIVED&&void 0!==e)
            # 如果接收到响应头，且存在进度回调函数
            (g=f.getResponseHeader("Content-Length"))
    };
    # 发送 AJAX 请求
    f.open("GET",b,!0);
    f.withCredentials=this.withCredentials;
    f.send(null)
};

# 解析 JSON 内容
THREE.JSONLoader.prototype.parse=function(a,b){
    # 创建几何体对象
    var c=new THREE.Geometry,d=void 0!==a.scale?1/a.scale:1;
    # 解析顶点、面、法线、颜色等信息
    (function(b){
        var d,g,h,k,n,p,q,m,r,t,s,u,v,y=a.faces;
        p=a.vertices;
        var G=a.normals,w=a.colors,K=0;
        if(void 0!==a.uvs){
            for(d=0;d<a.uvs.length;d++)
                a.uvs[d].length&&K++;
            for(d=0;d<K;d++)
                c.faceVertexUvs[d]=[]
        }
        k=0;
        for(n=p.length;k<n;)
            d=new THREE.Vector3,d.x=p[k++]*b,d.y=p[k++]*b,d.z=p[k++]*b,c.vertices.push(d);
        k=0;
        for(n=y.length;k<n;)
            if(b=y[k++],r=b&1,h=b&2,d=b&8,q=b&16,t=b&32,p=b&64,b&=128,r){
                r=new THREE.Face3;
                r.a=y[k];
                r.b=y[k+1];
                r.c=y[k+3];
                s=new THREE.Face3;
                s.a=y[k+1];
                s.b=y[k+2];
                s.c=y[k+3];
                k+=4;
                h&&(h=y[k++],r.materialIndex=h,s.materialIndex=h);
                h=c.faces.length;
                if(d)
                    for(d=0;d<K;d++)
                        for(u=a.uvs[d],c.faceVertexUvs[d][h]=[],c.faceVertexUvs[d][h+1]=[],g=0;4>g;g++)
                            m=y[k++],v=u[2*m],m=u[2*m+1],v=new THREE.Vector2(v,m),2!==g&&c.faceVertexUvs[d][h].push(v),0!==g&&c.faceVertexUvs[d][h+1].push(v);
                q&&(q=3*y[k++],r.normal.set(G[q++],G[q++],G[q]),s.normal.copy(r.normal));
                if(t)
                    for(d=0;4>d;d++)
                        q=3*y[k++],t=new THREE.Vector3(G[q++],
G[q++],G[q]),2!==d&&r.vertexNormals.push(t),0!==d&&s.vertexNormals.push(t);p&&(p=y[k++],p=w[p],r.color.setHex(p),s.color.setHex(p));if(b)for(d=0;4>d;d++)p=y[k++],p=w[p],2!==d&&r.vertexColors.push(new THREE.Color(p)),0!==d&&s.vertexColors.push(new THREE.Color(p));c.faces.push(r);c.faces.push(s)}else{r=new THREE.Face3;r.a=y[k++];r.b=y[k++];r.c=y[k++];h&&(h=y[k++],r.materialIndex=h);h=c.faces.length;if(d)for(d=0;d<K;d++)for(u=a.uvs[d],c.faceVertexUvs[d][h]=[],g=0;3>g;g++)m=y[k++],v=u[2*m],m=u[2*m+1],
v=new THREE.Vector2(v,m),c.faceVertexUvs[d][h].push(v);q&&(q=3*y[k++],r.normal.set(G[q++],G[q++],G[q]));if(t)for(d=0;3>d;d++)q=3*y[k++],t=new THREE.Vector3(G[q++],G[q++],G[q]),r.vertexNormals.push(t);p&&(p=y[k++],r.color.setHex(w[p]));if(b)for(d=0;3>d;d++)p=y[k++],r.vertexColors.push(new THREE.Color(w[p]));c.faces.push(r)}})(d);(function(){var b=void 0!==a.influencesPerVertex?a.influencesPerVertex:2;if(a.skinWeights)for(var d=0,g=a.skinWeights.length;d<g;d+=b)c.skinWeights.push(new THREE.Vector4(a.skinWeights[d],
1<b?a.skinWeights[d+1]:0,2<b?a.skinWeights[d+2]:0,3<b?a.skinWeights[d+3]:0));if(a.skinIndices)for(d=0,g=a.skinIndices.length;d<g;d+=b)c.skinIndices.push(new THREE.Vector4(a.skinIndices[d],1<b?a.skinIndices[d+1]:0,2<b?a.skinIndices[d+2]:0,3<b?a.skinIndices[d+3]:0));c.bones=a.bones;c.bones&&0<c.bones.length&&(c.skinWeights.length!==c.skinIndices.length||c.skinIndices.length!==c.vertices.length)&&console.warn("When skinning, number of vertices ("+c.vertices.length+"), skinIndices ("+c.skinIndices.length+
# 如果存在 morphTargets 属性
if(void 0!==a.morphTargets){
    # 初始化变量
    var d,g,h,k,n,p;
    # 遍历 morphTargets 数组
    d=0;
    for(g=a.morphTargets.length;d<g;d++){
        # 初始化 morphTargets 对象
        c.morphTargets[d]={};
        # 设置 morphTargets 的名称
        c.morphTargets[d].name=a.morphTargets[d].name;
        # 初始化 morphTargets 的顶点数组
        c.morphTargets[d].vertices=[];
        n=c.morphTargets[d].vertices;
        p=a.morphTargets[d].vertices;
        h=0;
        k=p.length;
        # 遍历顶点数组
        for(h=0;h<k;h+=3){
            # 创建三维向量
            var q=new THREE.Vector3;
            # 设置向量的值
            q.x=p[h]*b;
            q.y=p[h+1]*b;
            q.z=p[h+2]*b;
            # 将向量添加到顶点数组中
            n.push(q);
        }
    }
}
# 如果存在 morphColors 属性
if(void 0!==a.morphColors){
    # 初始化变量
    for(d=0,g=a.morphColors.length;d<g;d++){
        # 初始化 morphColors 对象
        c.morphColors[d]={};
        # 设置 morphColors 的名称
        c.morphColors[d].name=a.morphColors[d].name;
        # 初始化 morphColors 的颜色数组
        c.morphColors[d].colors=[];
        k=c.morphColors[d].colors;
        n=a.morphColors[d].colors;
        b=0;
        h=n.length;
        # 遍历颜色数组
        for(b=0;b<h;b+=3){
            # 创建颜色对象
            p=new THREE.Color(16755200);
            # 设置颜色的 RGB 值
            p.setRGB(n[b],n[b+1],n[b+2]);
            # 将颜色对象添加到颜色数组中
            k.push(p);
        }
    }
}
a=a.boundingSphere;
void 0!==a&&(c=new THREE.Vector3,void 0!==a.center&&c.fromArray(a.center),b.boundingSphere=new THREE.Sphere(c,a.radius));
return b
}};
# 设置变量a为a.boundingSphere
# 如果a不为undefined，则创建一个新的THREE.Vector3对象c，并且如果a.center不为undefined，则使用a.center的值来设置c的值，然后使用c和a.radius创建一个新的THREE.Sphere对象并设置给b.boundingSphere
# 返回变量b
THREE.MaterialLoader=function(a){this.manager=void 0!==a?a:THREE.DefaultLoadingManager};
# 创建一个THREE.MaterialLoader构造函数，如果传入参数a不为undefined，则使用传入的a作为this.manager，否则使用THREE.DefaultLoadingManager作为this.manager
THREE.MaterialLoader.prototype={constructor:THREE.MaterialLoader,load:function(a,b,c,d){var e=this,f=new THREE.XHRLoader;f.setCrossOrigin(this.crossOrigin);f.load(a,function(a){b(e.parse(JSON.parse(a)))},c,d)},setCrossOrigin:function(a){this.crossOrigin=a},parse:function(a){var b=new THREE[a.type];void 0!==a.color&&b.color.setHex(a.color);void 0!==a.ambient&&b.ambient.setHex(a.ambient);void 0!==a.emissive&&b.emissive.setHex(a.emissive);void 0!==a.specular&&b.specular.setHex(a.specular);void 0!==a.shininess&&
(b.shininess=a.shininess);void 0!==a.uniforms&&(b.uniforms=a.uniforms);void 0!==a.vertexShader&&(b.vertexShader=a.vertexShader);void 0!==a.fragmentShader&&(b.fragmentShader=a.fragmentShader);void 0!==a.vertexColors&&(b.vertexColors=a.vertexColors);void 0!==a.shading&&(b.shading=a.shading);void 0!==a.blending&&(b.blending=a.blending);void 0!==a.side&&(b.side=a.side);void 0!==a.opacity&&(b.opacity=a.opacity);void 0!==a.transparent&&(b.transparent=a.transparent);void 0!==a.wireframe&&(b.wireframe=a.wireframe);
if(void 0!==a.materials)for(var c=0,d=a.materials.length;c<d;c++)b.materials.push(this.parse(a.materials[c]));return b}};
# 设置THREE.MaterialLoader的原型对象，包括构造函数、load方法、setCrossOrigin方法和parse方法
THREE.ObjectLoader=function(a){this.manager=void 0!==a?a:THREE.DefaultLoadingManager};
# 创建一个THREE.ObjectLoader构造函数，如果传入参数a不为undefined，则使用传入的a作为this.manager，否则使用THREE.DefaultLoadingManager作为this.manager
THREE.ObjectLoader.prototype={constructor:THREE.ObjectLoader,load:function(a,b,c,d){var e=this,f=new THREE.XHRLoader(e.manager);f.setCrossOrigin(this.crossOrigin);f.load(a,function(a){b(e.parse(JSON.parse(a)))},c,d)},setCrossOrigin:function(a){this.crossOrigin=a},parse:function(a){var b=this.parseGeometries(a.geometries),c=this.parseMaterials(a.materials);return this.parseObject(a.object,b,c)},parseGeometries:function(a){var b={};if(void 0!==a)for(var c=new THREE.JSONLoader,d=new THREE.BufferGeometryLoader,
# 设置THREE.ObjectLoader的原型对象，包括构造函数、load方法、setCrossOrigin方法和parse方法
# 初始化变量 e 为 0，f 为数组 a 的长度
e=0,f=a.length;e<f;e++){
    # 获取数组 a 中索引为 e 的元素
    var g,h=a[e];
    # 根据元素的 type 属性进行不同的处理
    switch(h.type){
        # 如果 type 为 "PlaneGeometry"，则创建一个平面几何体
        case "PlaneGeometry":
            g=new THREE.PlaneGeometry(h.width,h.height,h.widthSegments,h.heightSegments);
            break;
        # 如果 type 为 "BoxGeometry" 或 "CubeGeometry"，则创建一个立方体几何体
        case "BoxGeometry":
        case "CubeGeometry":
            g=new THREE.BoxGeometry(h.width,h.height,h.depth,h.widthSegments,h.heightSegments,h.depthSegments);
            break;
        # 如果 type 为 "CircleGeometry"，则创建一个圆形几何体
        case "CircleGeometry":
            g=new THREE.CircleGeometry(h.radius,h.segments);
            break;
        # 如果 type 为 "CylinderGeometry"，则创建一个圆柱几何体
        case "CylinderGeometry":
            g=new THREE.CylinderGeometry(h.radiusTop,h.radiusBottom,h.height,h.radialSegments,h.heightSegments,h.openEnded);
            break;
        # 如果 type 为 "SphereGeometry"，则创建一个球体几何体
        case "SphereGeometry":
            g=new THREE.SphereGeometry(h.radius,h.widthSegments,h.heightSegments,h.phiStart,h.phiLength,h.thetaStart,h.thetaLength);
            break;
        # 如果 type 为 "IcosahedronGeometry"，则创建一个二十面体几何体
        case "IcosahedronGeometry":
            g=new THREE.IcosahedronGeometry(h.radius,h.detail);
            break;
        # 如果 type 为 "TorusGeometry"，则创建一个圆环几何体
        case "TorusGeometry":
            g=new THREE.TorusGeometry(h.radius,h.tube,h.radialSegments,h.tubularSegments,h.arc);
            break;
        # 如果 type 为 "TorusKnotGeometry"，则创建一个环面纽结几何体
        case "TorusKnotGeometry":
            g=new THREE.TorusKnotGeometry(h.radius,h.tube,h.radialSegments,h.tubularSegments,h.p,h.q,h.heightScale);
            break;
        # 如果 type 为 "BufferGeometry"，则解析数据并创建缓冲几何体
        case "BufferGeometry":
            g=d.parse(h.data);
            break;
        # 如果 type 为 "Geometry"，则解析数据并创建几何体
        case "Geometry":
            g=c.parse(h.data).geometry
    }
    # 设置几何体的唯一标识为 uuid
    g.uuid=h.uuid;
    # 如果存在 name 属性，则设置几何体的名称
    void 0!==h.name&&(g.name=h.name);
    # 将几何体添加到字典中，以 uuid 为键
    b[h.uuid]=g
}
# 返回包含所有几何体的字典
return b
},
# 解析材质数据并创建材质对象
parseMaterials:function(a){
    var b={};
    # 如果材质数据存在，则遍历解析并创建材质对象
    if(void 0!==a)for(var c=new THREE.MaterialLoader,d=0,e=a.length;d<e;d++){
        var f=a[d],g=c.parse(f);
        g.uuid=f.uuid;
        void 0!==f.name&&(g.name=f.name);
        b[f.uuid]=g
    }
    # 返回包含所有材质对象的字典
    return b
},
# 解析对象数据并创建对应的对象
parseObject:function(){
    var a=new THREE.Matrix4;
    return function(b,c,d){
        var e;
        # 根据对象的 type 属性进行不同的处理
        switch(b.type){
            # 如果 type 为 "Scene"，则创建一个场景对象
            case "Scene":
                e=new THREE.Scene;
                break;
            # 如果 type 为 "PerspectiveCamera"，则创建一个透视相机对象
            case "PerspectiveCamera":
                e=new THREE.PerspectiveCamera(b.fov,
# 根据不同的类型创建不同的 3D 对象
b.aspect,b.near,b.far);break;
case "OrthographicCamera":e=new THREE.OrthographicCamera(b.left,b.right,b.top,b.bottom,b.near,b.far);break;
case "AmbientLight":e=new THREE.AmbientLight(b.color);break;
case "DirectionalLight":e=new THREE.DirectionalLight(b.color,b.intensity);break;
case "PointLight":e=new THREE.PointLight(b.color,b.intensity,b.distance);break;
case "SpotLight":e=new THREE.SpotLight(b.color,b.intensity,b.distance,b.angle,b.exponent);break;
case "HemisphereLight":e=new THREE.HemisphereLight(b.color,b.groundColor,b.intensity);break;
case "Mesh":e=c[b.geometry];var f=d[b.material];void 0===e&&console.warn("THREE.ObjectLoader: Undefined geometry",b.geometry);void 0===f&&console.warn("THREE.ObjectLoader: Undefined material",b.material);e=new THREE.Mesh(e,f);break;
case "Line":e=c[b.geometry];f=d[b.material];void 0===e&&console.warn("THREE.ObjectLoader: Undefined geometry",b.geometry);void 0===f&&console.warn("THREE.ObjectLoader: Undefined material",b.material);e=new THREE.Line(e,f);break;
case "Sprite":f=d[b.material];void 0===f&&console.warn("THREE.ObjectLoader: Undefined material",b.material);e=new THREE.Sprite(f);break;
case "Group":e=new THREE.Group;break;
default:e=new THREE.Object3D}e.uuid=b.uuid;void 0!==b.name&&(e.name=b.name);void 0!==b.matrix?(a.fromArray(b.matrix),a.decompose(e.position,e.quaternion,e.scale)):(void 0!==b.position&&e.position.fromArray(b.position),void 0!==b.rotation&&e.rotation.fromArray(b.rotation),void 0!==b.scale&&e.scale.fromArray(b.scale));void 0!==b.visible&&(e.visible=b.visible);void 0!==b.userData&&(e.userData=b.userData);if(void 0!==b.children)for(var g in b.children)e.add(this.parseObject(b.children[g],c,d));return e}}()};THREE.TextureLoader=function(a){this.manager=void 0!==a?a:THREE.DefaultLoadingManager};
# 定义 THREE.TextureLoader 的原型对象
THREE.TextureLoader.prototype={
    constructor:THREE.TextureLoader,
    # 加载纹理
    load:function(a,b,c,d){
        # 创建一个 THREE.ImageLoader 对象
        var e=new THREE.ImageLoader(this.manager);
        # 设置跨域属性
        e.setCrossOrigin(this.crossOrigin);
        # 加载图片
        e.load(a,function(a){
            # 创建一个 THREE.Texture 对象
            a=new THREE.Texture(a);
            # 设置需要更新标志为真
            a.needsUpdate=!0;
            # 如果回调函数存在，则调用回调函数
            void 0!==b&&b(a)
        },c,d)
    },
    # 设置跨域属性
    setCrossOrigin:function(a){
        this.crossOrigin=a
    }
};

# 定义 THREE.CompressedTextureLoader 的构造函数
THREE.CompressedTextureLoader=function(){
    this._parser=null
};

# 定义 THREE.CompressedTextureLoader 的原型对象
THREE.CompressedTextureLoader.prototype={
    constructor:THREE.CompressedTextureLoader,
    # 加载压缩纹理
    load:function(a,b,c){
        # 创建一个空的图片数组
        var d=this,e=[],
        f=new THREE.CompressedTexture;
        f.image=e;
        # 创建一个 THREE.XHRLoader 对象
        var g=new THREE.XHRLoader;
        # 设置响应类型为二进制数组
        g.setResponseType("arraybuffer");
        # 如果参数 a 是数组
        if(a instanceof Array){
            var h=0;
            # 定义回调函数
            c=function(c){
                # 加载数组中的每个元素
                g.load(a[c],function(a){
                    # 解析数据
                    a=d._parser(a,!0);
                    # 将解析后的数据存入图片数组
                    e[c]={width:a.width,height:a.height,format:a.format,mipmaps:a.mipmaps};
                    h+=1;
                    # 如果加载完所有数据
                    6===h&&(1==a.mipmapCount&&(f.minFilter=THREE.LinearFilter),f.format=a.format,f.needsUpdate=!0,b&&b(f))
                })
            };
            # 遍历数组并调用回调函数
            for(var k=0,n=a.length;k<n;++k)c(k)
        }else{
            # 加载单个数据
            g.load(a,function(a){
                # 解析数据
                a=d._parser(a,!0);
                # 如果是立方体贴图
                if(a.isCubemap){
                    for(var c=a.mipmaps.length/a.mipmapCount,g=0;g<c;g++){
                        e[g]={mipmaps:[]};
                        for(var h=0;h<a.mipmapCount;h++)e[g].mipmaps.push(a.mipmaps[g*a.mipmapCount+h]),e[g].format=a.format,e[g].width=a.width,e[g].height=a.height
                    }
                }else{
                    f.image.width=a.width;
                    f.image.height=a.height;
                    f.mipmaps=a.mipmaps
                }
                # 如果只有一个 mipmap
                1===a.mipmapCount&&(f.minFilter=THREE.LinearFilter);
                f.format=a.format;
                f.needsUpdate=!0;
                # 如果回调函数存在，则调用回调函数
                b&&b(f)
            })
        }
        return f
    }
};

# 定义 THREE.Material 的构造函数
THREE.Material=function(){
    # 设置 id 属性
    Object.defineProperty(this,"id",{value:THREE.MaterialIdCount++});
    # 设置 uuid 属性
    this.uuid=THREE.Math.generateUUID();
    # 设置 name 属性
    this.name="";
    # 设置 type 属性
    this.type="Material";
    # 设置 side 属性
    this.side=THREE.FrontSide;
    # 设置 opacity 属性
    this.opacity=1;
    # 设置 transparent 属性
    this.transparent=!1;
    # 设置 blending 属性
    this.blending=THREE.NormalBlending;
    # 设置 blendSrc 属性
    this.blendSrc=THREE.SrcAlphaFactor;
    # 设置 blendDst 属性
    this.blendDst=THREE.OneMinusSrcAlphaFactor;
    # 设置 blendEquation 属性
    this.blendEquation=THREE.AddEquation;
    # 设置 depthWrite 和 depthTest 属性
    this.depthWrite=this.depthTest=!0;
    # 设置 polygonOffset 属性
    this.polygonOffset=!1;
    # 设置 overdraw、alphaTest、polygonOffsetUnits 和 polygonOffsetFactor 属性
    this.overdraw=this.alphaTest=this.polygonOffsetUnits=this.polygonOffsetFactor=0;
    # 设置 needsUpdate 和 visible 属性
    this.needsUpdate=this.visible=!0
};
// 定义 THREE.Material 的原型对象
THREE.Material.prototype={
    // 构造函数
    constructor:THREE.Material,
    // 设置材质属性
    setValues:function(a){
        // 如果传入了参数
        if(void 0!==a)
            // 遍历参数对象
            for(var b in a){
                var c=a[b];
                // 如果参数值为 undefined，输出警告信息
                if(void 0===c)console.warn("THREE.Material: '"+b+"' parameter is undefined.");
                // 如果参数名在当前对象中存在
                else if(b in this){
                    var d=this[b];
                    // 如果当前属性是颜色类型，设置颜色值
                    d instanceof THREE.Color?d.set(c):
                    // 如果当前属性是三维向量类型，复制向量值
                    d instanceof THREE.Vector3&&c instanceof THREE.Vector3?d.copy(c):
                    // 否则直接赋值
                    this[b]="overdraw"==b?Number(c):c
                }
            }
    },
    // 转换为 JSON 格式
    toJSON:function(){
        var a={metadata:{version:4.2,type:"material",generator:"MaterialExporter"},uuid:this.uuid,type:this.type};
        // 如果材质有名称，添加到 JSON 对象中
        ""!==this.name&&(a.name=this.name);
        // 根据不同类型的材质，添加不同的属性到 JSON 对象中
        this instanceof THREE.MeshBasicMaterial?(a.color=this.color.getHex(),this.vertexColors!==THREE.NoColors&&(a.vertexColors=this.vertexColors),this.blending!==THREE.NormalBlending&&(a.blending=this.blending),this.side!==THREE.FrontSide&&(a.side=this.side)):
        this instanceof THREE.MeshLambertMaterial?(a.color=this.color.getHex(),a.ambient=this.ambient.getHex(),a.emissive=this.emissive.getHex(),this.vertexColors!==THREE.NoColors&&(a.vertexColors=this.vertexColors),this.blending!==THREE.NormalBlending&&(a.blending=this.blending),this.side!==THREE.FrontSide&&(a.side=this.side)):
        this instanceof THREE.MeshPhongMaterial?(a.color=this.color.getHex(),a.ambient=this.ambient.getHex(),a.emissive=this.emissive.getHex(),a.specular=this.specular.getHex(),a.shininess=this.shininess,this.vertexColors!==THREE.NoColors&&(a.vertexColors=this.vertexColors),this.blending!==THREE.NormalBlending&&(a.blending=this.blending),this.side!==THREE.FrontSide&&(a.side=this.side)):
        this instanceof THREE.MeshNormalMaterial?(this.shading!==
# 如果使用了 FlatShading，则设置材质的着色方式为 FlatShading
THREE.FlatShading&&(a.shading=this.shading),this.blending!==THREE.NormalBlending&&(a.blending=this.blending),this.side!==THREE.FrontSide&&(a.side=this.side)):this instanceof THREE.MeshDepthMaterial?(this.blending!==THREE.NormalBlending&&(a.blending=this.blending),this.side!==THREE.FrontSide&&(a.side=this.side)):this instanceof THREE.ShaderMaterial?(a.uniforms=this.uniforms,a.vertexShader=this.vertexShader,a.fragmentShader=this.fragmentShader):this instanceof THREE.SpriteMaterial&&(a.color=this.color.getHex());
# 如果透明度小于1，则设置材质的透明度
1>this.opacity&&(a.opacity=this.opacity);!1!==this.transparent&&(a.transparent=this.transparent);!1!==this.wireframe&&(a.wireframe=this.wireframe);return a},clone:function(a){void 0===a&&(a=new THREE.Material);a.name=this.name;a.side=this.side;a.opacity=this.opacity;a.transparent=this.transparent;a.blending=this.blending;a.blendSrc=this.blendSrc;a.blendDst=this.blendDst;a.blendEquation=this.blendEquation;a.depthTest=this.depthTest;a.depthWrite=this.depthWrite;a.polygonOffset=this.polygonOffset;a.polygonOffsetFactor=
this.polygonOffsetFactor;a.polygonOffsetUnits=this.polygonOffsetUnits;a.alphaTest=this.alphaTest;a.overdraw=this.overdraw;a.visible=this.visible;return a},dispose:function(){this.dispatchEvent({type:"dispose"})}};THREE.EventDispatcher.prototype.apply(THREE.Material.prototype);THREE.MaterialIdCount=0;
# 创建 LineBasicMaterial 类
THREE.LineBasicMaterial=function(a){THREE.Material.call(this);this.type="LineBasicMaterial";this.color=new THREE.Color(16777215);this.linewidth=1;this.linejoin=this.linecap="round";this.vertexColors=THREE.NoColors;this.fog=!0;this.setValues(a)};
# 继承 LineBasicMaterial 类的原型
THREE.LineBasicMaterial.prototype=Object.create(THREE.Material.prototype);
# 克隆 LineBasicMaterial 类
THREE.LineBasicMaterial.prototype.clone=function(){var a=new THREE.LineBasicMaterial;THREE.Material.prototype.clone.call(this,a);a.color.copy(this.color);a.linewidth=this.linewidth;a.linecap=this.linecap;a.linejoin=this.linejoin;a.vertexColors=this.vertexColors;a.fog=this.fog;return a};
# 定义一个名为 LineDashedMaterial 的函数，继承自 Material 类
THREE.LineDashedMaterial=function(a){THREE.Material.call(this);this.type="LineDashedMaterial";this.color=new THREE.Color(16777215);this.scale=this.linewidth=1;this.dashSize=3;this.gapSize=1;this.vertexColors=!1;this.fog=!0;this.setValues(a)};
# 设置 LineDashedMaterial 原型的属性和方法
THREE.LineDashedMaterial.prototype=Object.create(THREE.Material.prototype);
# 克隆 LineDashedMaterial 对象
THREE.LineDashedMaterial.prototype.clone=function(){var a=new THREE.LineDashedMaterial;THREE.Material.prototype.clone.call(this,a);a.color.copy(this.color);a.linewidth=this.linewidth;a.scale=this.scale;a.dashSize=this.dashSize;a.gapSize=this.gapSize;a.vertexColors=this.vertexColors;a.fog=this.fog;return a};
# 定义一个名为 MeshBasicMaterial 的函数，继承自 Material 类
THREE.MeshBasicMaterial=function(a){THREE.Material.call(this);this.type="MeshBasicMaterial";this.color=new THREE.Color(16777215);this.envMap=this.alphaMap=this.specularMap=this.lightMap=this.map=null;this.combine=THREE.MultiplyOperation;this.reflectivity=1;this.refractionRatio=.98;this.fog=!0;this.shading=THREE.SmoothShading;this.wireframe=!1;this.wireframeLinewidth=1;this.wireframeLinejoin=this.wireframeLinecap="round";this.vertexColors=THREE.NoColors;this.morphTargets=this.skinning=!1;this.setValues(a)};
# 设置 MeshBasicMaterial 原型的属性和方法
THREE.MeshBasicMaterial.prototype=Object.create(THREE.Material.prototype);
# 克隆 MeshBasicMaterial 对象
THREE.MeshBasicMaterial.prototype.clone=function(){var a=new THREE.MeshBasicMaterial;THREE.Material.prototype.clone.call(this,a);a.color.copy(this.color);a.map=this.map;a.lightMap=this.lightMap;a.specularMap=this.specularMap;a.alphaMap=this.alphaMap;a.envMap=this.envMap;a.combine=this.combine;a.reflectivity=this.reflectivity;a.refractionRatio=this.refractionRatio;a.fog=this.fog;a.shading=this.shading;a.wireframe=this.wireframe;a.wireframeLinewidth=this.wireframeLinewidth;a.wireframeLinecap=this.wireframeLinecap;
a.wireframeLinejoin=this.wireframeLinejoin;a.vertexColors=this.vertexColors;a.skinning=this.skinning;a.morphTargets=this.morphTargets;return a};
# 创建 MeshLambertMaterial 类，继承自 Material 类
THREE.MeshLambertMaterial=function(a){THREE.Material.call(this);
# 设置材质类型为 MeshLambertMaterial
this.type="MeshLambertMaterial";
# 设置颜色为白色
this.color=new THREE.Color(16777215);
# 设置环境光颜色为白色
this.ambient=new THREE.Color(16777215);
# 设置发光颜色为黑色
this.emissive=new THREE.Color(0);
# 是否使用环绕贴图
this.wrapAround=!1;
# 设置环绕贴图的 RGB 值
this.wrapRGB=new THREE.Vector3(1,1,1);
# 设置环境贴图、透明贴图、高光贴图、光照贴图、普通贴图为空
this.envMap=this.alphaMap=this.specularMap=this.lightMap=this.map=null;
# 设置贴图的组合方式为 MultiplyOperation
this.combine=THREE.MultiplyOperation;
# 设置反射率为 1
this.reflectivity=1;
# 设置折射率为 0.98
this.refractionRatio=.98;
# 是否受雾效果影响
this.fog=!0;
# 设置着色方式为 SmoothShading
this.shading=THREE.SmoothShading;
# 是否显示线框
this.wireframe=!1;
# 设置线框宽度为 1
this.wireframeLinewidth=1;
# 设置线框连接方式为 round
this.wireframeLinejoin=this.wireframeLinecap="round";
# 设置顶点颜色为 NoColors
this.vertexColors=THREE.NoColors;
# 是否使用皮肤动画
this.morphNormals=this.morphTargets=this.skinning=!1;
# 设置材质的属性值
this.setValues(a)};
# 克隆 MeshLambertMaterial 对象
THREE.MeshLambertMaterial.prototype=Object.create(THREE.Material.prototype);
# 克隆 MeshLambertMaterial 对象
THREE.MeshLambertMaterial.prototype.clone=function(){var a=new THREE.MeshLambertMaterial;
# 调用 Material 类的 clone 方法
THREE.Material.prototype.clone.call(this,a);
# 复制材质的属性值
a.color.copy(this.color);
a.ambient.copy(this.ambient);
a.emissive.copy(this.emissive);
a.wrapAround=this.wrapAround;
a.wrapRGB.copy(this.wrapRGB);
a.map=this.map;
a.lightMap=this.lightMap;
a.specularMap=this.specularMap;
a.alphaMap=this.alphaMap;
a.envMap=this.envMap;
a.combine=this.combine;
a.reflectivity=this.reflectivity;
a.refractionRatio=this.refractionRatio;
a.fog=this.fog;
a.shading=this.shading;
a.wireframe=this.wireframe;
a.wireframeLinewidth=this.wireframeLinewidth;
a.wireframeLinecap=this.wireframeLinecap;
a.wireframeLinejoin=this.wireframeLinejoin;
a.vertexColors=this.vertexColors;
a.skinning=this.skinning;
a.morphTargets=this.morphTargets;
a.morphNormals=this.morphNormals;
# 返回克隆后的对象
return a};
# 创建一个名为 MeshPhongMaterial 的类，继承自 Material 类
THREE.MeshPhongMaterial=function(a){THREE.Material.call(this);
# 设置 MeshPhongMaterial 类的类型为 "MeshPhongMaterial"
this.type="MeshPhongMaterial";
# 设置颜色属性，并初始化为白色
this.color=new THREE.Color(16777215);
# 设置环境光属性，并初始化为白色
this.ambient=new THREE.Color(16777215);
# 设置发射光属性，并初始化为黑色
this.emissive=new THREE.Color(0);
# 设置镜面光属性，并初始化为灰色
this.specular=new THREE.Color(1118481);
# 设置高光亮度属性，并初始化为 30
this.shininess=30;
# 初始化 wrapAround 和 metal 属性为 false
this.wrapAround=this.metal=!1;
# 设置 wrapRGB 属性，并初始化为 (1,1,1) 的向量
this.wrapRGB=new THREE.Vector3(1,1,1);
# 初始化 bumpMap、lightMap 和 map 属性为 null
this.bumpMap=this.lightMap=this.map=null;
# 设置 bumpScale 属性，并初始化为 1
this.bumpScale=1;
# 初始化 normalMap 属性为 null
this.normalMap=null;
# 设置 normalScale 属性，并初始化为 (1,1) 的向量
this.normalScale=new THREE.Vector2(1,1);
# 初始化 envMap、alphaMap 和 specularMap 属性为 null
this.envMap=this.alphaMap=this.specularMap=null;
# 设置 combine 属性为 THREE.MultiplyOperation
this.combine=THREE.MultiplyOperation;
# 设置 reflectivity 属性，并初始化为 1
this.reflectivity=1;
# 设置 refractionRatio 属性，并初始化为 0.98
this.refractionRatio=.98;
# 初始化 fog 属性为 true
this.fog=!0;
# 设置 shading 属性为 THREE.SmoothShading
this.shading=THREE.SmoothShading;
# 初始化 wireframe 属性为 false
this.wireframe=!1;
# 设置 wireframeLinewidth 属性，并初始化为 1
this.wireframeLinewidth=1;
# 设置 wireframeLinejoin 和 wireframeLinecap 属性，并初始化为 "round"
this.wireframeLinejoin=this.wireframeLinecap="round";
# 设置 vertexColors 属性为 THREE.NoColors
this.vertexColors=THREE.NoColors;
# 初始化 morphNormals、morphTargets 和 skinning 属性为 false
this.morphNormals=this.morphTargets=this.skinning=!1;
# 调用 setValues 方法，传入参数 a
this.setValues(a)};
# 将 MeshPhongMaterial 的原型设置为继承自 Material 的原型
THREE.MeshPhongMaterial.prototype=Object.create(THREE.Material.prototype);
# 创建 MeshPhongMaterial 的克隆方法
THREE.MeshPhongMaterial.prototype.clone=function(){
# 创建一个新的 MeshPhongMaterial 对象
var a=new THREE.MeshPhongMaterial;
# 调用 Material 的 clone 方法，传入当前对象和新对象
THREE.Material.prototype.clone.call(this,a);
# 复制当前对象的属性到新对象
a.color.copy(this.color);
a.ambient.copy(this.ambient);
a.emissive.copy(this.emissive);
a.specular.copy(this.specular);
a.shininess=this.shininess;
a.metal=this.metal;
a.wrapAround=this.wrapAround;
a.wrapRGB.copy(this.wrapRGB);
a.map=this.map;
a.lightMap=this.lightMap;
a.bumpMap=this.bumpMap;
a.bumpScale=this.bumpScale;
a.normalMap=this.normalMap;
a.normalScale.copy(this.normalScale);
a.specularMap=this.specularMap;
a.alphaMap=this.alphaMap;
a.envMap=this.envMap;
a.combine=this.combine;
a.reflectivity=this.reflectivity;
a.refractionRatio=this.refractionRatio;
a.fog=this.fog;
a.shading=this.shading;
a.wireframe=this.wireframe;
a.wireframeLinewidth=this.wireframeLinewidth;
a.wireframeLinecap=this.wireframeLinecap;
a.wireframeLinejoin=this.wireframeLinejoin;
a.vertexColors=this.vertexColors;
a.skinning=this.skinning;
a.morphTargets=this.morphTargets;
a.morphNormals=this.morphNormals;
# 返回新对象
return a};
# 创建一个名为 MeshDepthMaterial 的类，继承自 Material 类
THREE.MeshDepthMaterial=function(a){THREE.Material.call(this);this.type="MeshDepthMaterial";this.wireframe=this.morphTargets=!1;this.wireframeLinewidth=1;this.setValues(a)};
# 将 MeshDepthMaterial 的原型设置为 Material 的实例
THREE.MeshDepthMaterial.prototype=Object.create(THREE.Material.prototype);
# 克隆 MeshDepthMaterial 对象
THREE.MeshDepthMaterial.prototype.clone=function(){var a=new THREE.MeshDepthMaterial;THREE.Material.prototype.clone.call(this,a);a.wireframe=this.wireframe;a.wireframeLinewidth=this.wireframeLinewidth;return a};

# 创建一个名为 MeshNormalMaterial 的类，继承自 Material 类
THREE.MeshNormalMaterial=function(a){THREE.Material.call(this,a);this.type="MeshNormalMaterial";this.shading=THREE.FlatShading;this.wireframe=!1;this.wireframeLinewidth=1;this.morphTargets=!1;this.setValues(a)};
# 将 MeshNormalMaterial 的原型设置为 Material 的实例
THREE.MeshNormalMaterial.prototype=Object.create(THREE.Material.prototype);
# 克隆 MeshNormalMaterial 对象
THREE.MeshNormalMaterial.prototype.clone=function(){var a=new THREE.MeshNormalMaterial;THREE.Material.prototype.clone.call(this,a);a.shading=this.shading;a.wireframe=this.wireframe;a.wireframeLinewidth=this.wireframeLinewidth;return a};

# 创建一个名为 MeshFaceMaterial 的类
THREE.MeshFaceMaterial=function(a){this.uuid=THREE.Math.generateUUID();this.type="MeshFaceMaterial";this.materials=a instanceof Array?a:[]};
# 设置 MeshFaceMaterial 的原型
THREE.MeshFaceMaterial.prototype={constructor:THREE.MeshFaceMaterial,toJSON:function(){for(var a={metadata:{version:4.2,type:"material",generator:"MaterialExporter"},uuid:this.uuid,type:this.type,materials:[]},b=0,c=this.materials.length;b<c;b++)a.materials.push(this.materials[b].toJSON());return a},clone:function(){for(var a=new THREE.MeshFaceMaterial,b=0;b<this.materials.length;b++)a.materials.push(this.materials[b].clone());return a}};

# 创建一个名为 PointCloudMaterial 的类，继承自 Material 类
THREE.PointCloudMaterial=function(a){THREE.Material.call(this);this.type="PointCloudMaterial";this.color=new THREE.Color(16777215);this.map=null;this.size=1;this.sizeAttenuation=!0;this.vertexColors=THREE.NoColors;this.fog=!0;this.setValues(a)};
# 将 PointCloudMaterial 的原型设置为 Material 的实例
THREE.PointCloudMaterial.prototype=Object.create(THREE.Material.prototype);
# 重写 PointCloudMaterial 原型的 clone 方法
THREE.PointCloudMaterial.prototype.clone=function(){
    # 创建一个新的 PointCloudMaterial 对象
    var a=new THREE.PointCloudMaterial;
    # 调用 Material 原型的 clone 方法，将当前对象的属性复制给新对象
    THREE.Material.prototype.clone.call(this,a);
    # 复制颜色属性
    a.color.copy(this.color);
    # 复制贴图属性
    a.map=this.map;
    # 复制点大小属性
    a.size=this.size;
    # 复制大小衰减属性
    a.sizeAttenuation=this.sizeAttenuation;
    # 复制顶点颜色属性
    a.vertexColors=this.vertexColors;
    # 复制雾效属性
    a.fog=this.fog;
    # 返回新的 PointCloudMaterial 对象
    return a;
};

# 创建 ParticleBasicMaterial 函数，已经被重命名为 PointCloudMaterial
THREE.ParticleBasicMaterial=function(a){
    # 输出警告信息
    console.warn("THREE.ParticleBasicMaterial has been renamed to THREE.PointCloudMaterial.");
    # 返回一个新的 PointCloudMaterial 对象
    return new THREE.PointCloudMaterial(a);
};

# 创建 ParticleSystemMaterial 函数，已经被重命名为 PointCloudMaterial
THREE.ParticleSystemMaterial=function(a){
    # 输出警告信息
    console.warn("THREE.ParticleSystemMaterial has been renamed to THREE.PointCloudMaterial.");
    # 返回一个新的 PointCloudMaterial 对象
    return new THREE.PointCloudMaterial(a);
};

# 创建 ShaderMaterial 函数
THREE.ShaderMaterial=function(a){
    # 调用 Material 构造函数
    THREE.Material.call(this);
    # 设置类型为 ShaderMaterial
    this.type="ShaderMaterial";
    # 定义属性
    this.defines={};
    this.uniforms={};
    this.attributes=null;
    this.vertexShader="void main() {\n\tgl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );\n}";
    this.fragmentShader="void main() {\n\tgl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );\n}";
    this.shading=THREE.SmoothShading;
    this.linewidth=1;
    this.wireframe=!1;
    this.wireframeLinewidth=1;
    this.lights=this.fog=!1;
    this.vertexColors=THREE.NoColors;
    this.morphNormals=this.morphTargets=this.skinning=!1;
    this.defaultAttributeValues={color:[1,1,1],uv:[0,0],uv2:[0,0]};
    this.index0AttributeName=void 0;
    # 设置属性值
    this.setValues(a)
};

# 将 ShaderMaterial 的原型设置为继承自 Material 的原型
THREE.ShaderMaterial.prototype=Object.create(THREE.Material.prototype);

# 重写 ShaderMaterial 原型的 clone 方法
THREE.ShaderMaterial.prototype.clone=function(){
    # 创建一个新的 ShaderMaterial 对象
    var a=new THREE.ShaderMaterial;
    # 调用 Material 原型的 clone 方法，将当前对象的属性复制给新对象
    THREE.Material.prototype.clone.call(this,a);
    # 复制片段着色器属性
    a.fragmentShader=this.fragmentShader;
    # 复制顶点着色器属性
    a.vertexShader=this.vertexShader;
    # 克隆 uniforms 属性
    a.uniforms=THREE.UniformsUtils.clone(this.uniforms);
    # 复制 attributes 属性
    a.attributes=this.attributes;
    # 复制 defines 属性
    a.defines=this.defines;
    # 复制着色属性
    a.shading=this.shading;
    # 复制线框属性
    a.wireframe=this.wireframe;
    # 复制线框宽度属性
    a.wireframeLinewidth=this.wireframeLinewidth;
    # 复制雾效属性
    a.fog=this.fog;
    # 复制灯光属性
    a.lights=this.lights;
    # 复制顶点颜色属性
    a.vertexColors=this.vertexColors;
    # 复制皮肤属性
    a.skinning=this.skinning;
    # 复制变形目标属性
    a.morphTargets=this.morphTargets;
    # 返回新的 ShaderMaterial 对象
    return a;
};
# 定义一个函数，返回一个包含morphTargets属性的对象
this.morphTargets;
# 设置a对象的morphNormals属性为当前对象的morphNormals属性
a.morphNormals=this.morphNormals;
# 返回a对象
return a};
# 定义一个原始着色器材质类，继承自ShaderMaterial类
THREE.RawShaderMaterial=function(a){
    THREE.ShaderMaterial.call(this,a);
    this.type="RawShaderMaterial"
};
# 原始着色器材质类的原型继承自ShaderMaterial类的原型
THREE.RawShaderMaterial.prototype=Object.create(THREE.ShaderMaterial.prototype);
# 原始着色器材质类的克隆方法
THREE.RawShaderMaterial.prototype.clone=function(){
    # 创建一个新的原始着色器材质对象
    var a=new THREE.RawShaderMaterial;
    # 调用ShaderMaterial类的克隆方法，将当前对象的属性克隆到新对象a中
    THREE.ShaderMaterial.prototype.clone.call(this,a);
    # 返回新对象a
    return a};
# 定义一个精灵材质类，继承自Material类
THREE.SpriteMaterial=function(a){
    THREE.Material.call(this);
    this.type="SpriteMaterial";
    this.color=new THREE.Color(16777215);
    this.map=null;
    this.rotation=0;
    this.fog=!1;
    this.setValues(a)
};
# 精灵材质类的原型继承自Material类的原型
THREE.SpriteMaterial.prototype=Object.create(THREE.Material.prototype);
# 精灵材质类的克隆方法
THREE.SpriteMaterial.prototype.clone=function(){
    # 创建一个新的精灵材质对象
    var a=new THREE.SpriteMaterial;
    # 调用Material类的克隆方法，将当前对象的属性克隆到新对象a中
    THREE.Material.prototype.clone.call(this,a);
    a.color.copy(this.color);
    a.map=this.map;
    a.rotation=this.rotation;
    a.fog=this.fog;
    # 返回新对象a
    return a};
# 定义一个纹理类
THREE.Texture=function(a,b,c,d,e,f,g,h,k){
    # 设置id属性为全局变量THREE.TextureIdCount的值
    Object.defineProperty(this,"id",{value:THREE.TextureIdCount++});
    # 生成一个唯一标识符
    this.uuid=THREE.Math.generateUUID();
    this.name="";
    # 设置image属性为传入的a参数或默认值THREE.Texture.DEFAULT_IMAGE
    this.image=void 0!==a?a:THREE.Texture.DEFAULT_IMAGE;
    this.mipmaps=[];
    # 设置mapping属性为传入的b参数或默认值THREE.Texture.DEFAULT_MAPPING
    this.mapping=void 0!==b?b:THREE.Texture.DEFAULT_MAPPING;
    # 设置wrapS属性为传入的c参数或默认值THREE.ClampToEdgeWrapping
    this.wrapS=void 0!==c?c:THREE.ClampToEdgeWrapping;
    # 设置wrapT属性为传入的d参数或默认值THREE.ClampToEdgeWrapping
    this.wrapT=void 0!==d?d:THREE.ClampToEdgeWrapping;
    # 设置magFilter属性为传入的e参数或默认值THREE.LinearFilter
    this.magFilter=void 0!==e?e:THREE.LinearFilter;
    # 设置minFilter属性为传入的f参数或默认值THREE.LinearMipMapLinearFilter
    this.minFilter=void 0!==f?f:THREE.LinearMipMapLinearFilter;
    # 设置anisotropy属性为传入的k参数或默认值1
    this.anisotropy=void 0!==k?k:1;
    # 设置format属性为传入的g参数或默认值THREE.RGBAFormat
    this.format=void 0!==g?g:THREE.RGBAFormat;
    # 设置type属性为传入的h参数或默认值THREE.UnsignedByteType
    this.type=void 0!==h?h:THREE.UnsignedByteType;
    # 设置offset属性为一个新的Vector2对象
    this.offset=new THREE.Vector2(0,0);
    # 设置repeat属性为一个新的Vector2对象
    this.repeat=new THREE.Vector2(1,1);
    # 设置generateMipmaps属性为true
    this.generateMipmaps=!0;
    # 设置premultiplyAlpha属性为false
    this.premultiplyAlpha=!1;
    # 设置flipY属性为true
    this.flipY=!0;
    # 设置unpackAlignment属性为4
    this.unpackAlignment=4;
    # 设置_needsUpdate属性为false
    this._needsUpdate=!1;
    # 设置onUpdate属性为null
    this.onUpdate=null
};
# 设置纹理类的默认图片属性为undefined
THREE.Texture.DEFAULT_IMAGE=void 0;
# 设置纹理类的默认映射属性为UVMapping
THREE.Texture.DEFAULT_MAPPING=new THREE.UVMapping;
# 定义 THREE.Texture 的原型对象
THREE.Texture.prototype={
    constructor:THREE.Texture,
    # 定义属性 needsUpdate，返回是否需要更新
    get needsUpdate(){return this._needsUpdate},
    # 定义属性 needsUpdate，设置是否需要更新
    set needsUpdate(a){!0===a&&this.update();this._needsUpdate=a},
    # 克隆方法，用于复制当前对象
    clone:function(a){
        void 0===a&&(a=new THREE.Texture);
        a.image=this.image;
        a.mipmaps=this.mipmaps.slice(0);
        a.mapping=this.mapping;
        a.wrapS=this.wrapS;
        a.wrapT=this.wrapT;
        a.magFilter=this.magFilter;
        a.minFilter=this.minFilter;
        a.anisotropy=this.anisotropy;
        a.format=this.format;
        a.type=this.type;
        a.offset.copy(this.offset);
        a.repeat.copy(this.repeat);
        a.generateMipmaps=this.generateMipmaps;
        a.premultiplyAlpha=this.premultiplyAlpha;
        a.flipY=this.flipY;
        a.unpackAlignment=this.unpackAlignment;
        return a
    },
    # 更新方法，触发 update 事件
    update:function(){this.dispatchEvent({type:"update"})},
    # 释放资源方法，触发 dispose 事件
    dispose:function(){this.dispatchEvent({type:"dispose"})}
};
# 将 THREE.Texture 的事件分发器应用到其原型对象上
THREE.EventDispatcher.prototype.apply(THREE.Texture.prototype);
# 定义 THREE.TextureIdCount 变量
THREE.TextureIdCount=0;
# 定义 CubeTexture 构造函数
THREE.CubeTexture=function(a,b,c,d,e,f,g,h,k){
    THREE.Texture.call(this,a,b,c,d,e,f,g,h,k);
    this.images=a
};
# 将 CubeTexture 的原型对象设置为继承自 Texture 的原型对象
THREE.CubeTexture.prototype=Object.create(THREE.Texture.prototype);
# 克隆 CubeTexture 对象的静态方法
THREE.CubeTexture.clone=function(a){
    void 0===a&&(a=new THREE.CubeTexture);
    THREE.Texture.prototype.clone.call(this,a);
    a.images=this.images;
    return a
};
# 定义 CompressedTexture 构造函数
THREE.CompressedTexture=function(a,b,c,d,e,f,g,h,k,n,p){
    THREE.Texture.call(this,null,f,g,h,k,n,d,e,p);
    this.image={width:b,height:c};
    this.mipmaps=a;
    this.generateMipmaps=this.flipY=!1
};
# 将 CompressedTexture 的原型对象设置为继承自 Texture 的原型对象
THREE.CompressedTexture.prototype=Object.create(THREE.Texture.prototype);
# 克隆 CompressedTexture 对象的方法
THREE.CompressedTexture.prototype.clone=function(){
    var a=new THREE.CompressedTexture;
    THREE.Texture.prototype.clone.call(this,a);
    return a
};
# 定义 DataTexture 构造函数
THREE.DataTexture=function(a,b,c,d,e,f,g,h,k,n,p){
    THREE.Texture.call(this,null,f,g,h,k,n,d,e,p);
    this.image={data:a,width:b,height:c}
};
# 将 DataTexture 的原型对象设置为继承自 Texture 的原型对象
THREE.DataTexture.prototype=Object.create(THREE.Texture.prototype);
# 克隆 DataTexture 对象的方法
THREE.DataTexture.prototype.clone=function(){
    var a=new THREE.DataTexture;
    THREE.Texture.prototype.clone.call(this,a);
    return a
};
# 创建一个名为 VideoTexture 的函数，继承自 Texture 类
THREE.VideoTexture=function(a,b,c,d,e,f,g,h,k){
    THREE.Texture.call(this,a,b,c,d,e,f,g,h,k);
    this.generateMipmaps=!1;
    # 创建一个闭包函数 p，用于更新视频纹理的状态
    var n=this,p=function(){
        requestAnimationFrame(p);
        # 检查视频是否已经有足够的数据
        a.readyState===a.HAVE_ENOUGH_DATA&&(n.needsUpdate=!0)
    };
    p()
};
# 将 VideoTexture 的原型设置为 Texture 的实例
THREE.VideoTexture.prototype=Object.create(THREE.Texture.prototype);

# 创建一个名为 Group 的函数，继承自 Object3D 类
THREE.Group=function(){
    THREE.Object3D.call(this);
    this.type="Group"
};
# 将 Group 的原型设置为 Object3D 的实例
THREE.Group.prototype=Object.create(THREE.Object3D.prototype);

# 创建一个名为 PointCloud 的函数，继承自 Object3D 类
THREE.PointCloud=function(a,b){
    THREE.Object3D.call(this);
    this.type="PointCloud";
    # 如果传入了几何体参数，则使用该参数，否则创建一个新的 Geometry 对象
    this.geometry=void 0!==a?a:new THREE.Geometry;
    # 如果传入了材质参数，则使用该参数，否则创建一个新的 PointCloudMaterial 对象
    this.material=void 0!==b?b:new THREE.PointCloudMaterial({color:16777215*Math.random()});
    this.sortParticles=!1
};
# 将 PointCloud 的原型设置为 Object3D 的实例
THREE.PointCloud.prototype=Object.create(THREE.Object3D.prototype);

# 为 PointCloud 添加 raycast 方法
THREE.PointCloud.prototype.raycast=function(){
    var a=new THREE.Matrix4,
        b=new THREE.Ray;
    return function(c,d){
        var e=this,
            f=e.geometry,
            g=c.params.PointCloud.threshold;
        a.getInverse(this.matrixWorld);
        b.copy(c.ray).applyMatrix4(a);
        if(null===f.boundingBox||!1!==b.isIntersectionBox(f.boundingBox)){
            var h=g/((this.scale.x+this.scale.y+this.scale.z)/3),
                k=new THREE.Vector3,
                g=function(a,f){
                    var g=b.distanceToPoint(a);
                    if(g<h){
                        var k=b.closestPointToPoint(a);
                        k.applyMatrix4(e.matrixWorld);
                        var m=c.ray.origin.distanceTo(k);
                        d.push({distance:m,distanceToRay:g,point:k.clone(),index:f,face:null,object:e})
                    }
                };
            if(f instanceof THREE.BufferGeometry){
                var n=f.attributes,
                    p=n.position.array;
                if(void 0!==n.index){
                    var n=n.index.array,
                        q=f.offsets;
                    0===q.length&&(q=[{start:0,count:n.length,index:0}]);
                    for(var m=0,r=q.length;m<r;++m)
                        for(var t=q[m].start,s=q[m].index,f=t,t=t+q[m].count;f<t;f++){
                            var u=s+n[f];
                            k.fromArray(p,3*u);
                            g(k,u)
                        }
                }else
                    for(n=p.length/3,f=0;f<n;f++)
                        k.set(p[3*f],p[3*f+1],p[3*f+2]),
                        g(k,f)
            }else
                for(k=this.geometry.vertices,

... (此处代码过长，省略部分内容)
# 定义 THREE.PointCloud 类的 clone 方法
THREE.PointCloud.prototype.clone=function(a){void 0===a&&(a=new THREE.PointCloud(this.geometry,this.material));
    a.sortParticles=this.sortParticles;
    THREE.Object3D.prototype.clone.call(this,a);
    return a};
# 定义 THREE.ParticleSystem 类，已被重命名为 THREE.PointCloud
THREE.ParticleSystem=function(a,b){
    console.warn("THREE.ParticleSystem has been renamed to THREE.PointCloud.");
    return new THREE.PointCloud(a,b)};
# 定义 THREE.Line 类
THREE.Line=function(a,b,c){
    THREE.Object3D.call(this);
    this.type="Line";
    this.geometry=void 0!==a?a:new THREE.Geometry;
    this.material=void 0!==b?b:new THREE.LineBasicMaterial({color:16777215*Math.random()});
    this.mode=void 0!==c?c:THREE.LineStrip};
# 定义 THREE.Line 类的 raycast 方法
THREE.Line.prototype.raycast=function(){
    var a=new THREE.Matrix4,
        b=new THREE.Ray,
        c=new THREE.Sphere;
    return function(d,e){
        var f=d.linePrecision,
            f=f*f,
            g=this.geometry;
        null===g.boundingSphere&&g.computeBoundingSphere();
        c.copy(g.boundingSphere);
        c.applyMatrix4(this.matrixWorld);
        if(!1!==d.ray.isIntersectionSphere(c)&&(a.getInverse(this.matrixWorld),b.copy(d.ray).applyMatrix4(a),g instanceof THREE.Geometry))
            for(var g=g.vertices,h=g.length,k=new THREE.Vector3,n=new THREE.Vector3,p=this.mode===THREE.LineStrip?1:2,q=0;q<h-1;q+=p)
                if(!(b.distanceSqToSegment(g[q],g[q+1],n,k)>f)){
                    var m=b.origin.distanceTo(n);
                    m<d.near||m>d.far||e.push({distance:m,point:k.clone().applyMatrix4(this.matrixWorld),face:null,faceIndex:null,object:this})}}};
# 定义 THREE.Line 类的 clone 方法
THREE.Line.prototype.clone=function(a){
    void 0===a&&(a=new THREE.Line(this.geometry,this.material,this.mode));
    THREE.Object3D.prototype.clone.call(this,a);
    return a};
# 定义 THREE.Mesh 类
THREE.Mesh=function(a,b){
    THREE.Object3D.call(this);
    this.type="Mesh";
    this.geometry=void 0!==a?a:new THREE.Geometry;
    this.material=void 0!==b?b:new THREE.MeshBasicMaterial({color:16777215*Math.random()});
    this.updateMorphTargets()};
# 定义 THREE.Mesh 类的原型
THREE.Mesh.prototype=Object.create(THREE.Object3D.prototype);
# 更新模型的形态目标
THREE.Mesh.prototype.updateMorphTargets=function(){
    # 如果几何体包含形态目标
    if(void 0!==this.geometry.morphTargets&&0<this.geometry.morphTargets.length){
        # 初始化形态目标的基础值、强制顺序、影响力和字典
        this.morphTargetBase=-1;
        this.morphTargetForcedOrder=[];
        this.morphTargetInfluences=[];
        this.morphTargetDictionary={};
        # 遍历形态目标数组
        for(var a=0,b=this.geometry.morphTargets.length;a<b;a++)
            # 初始化形态目标的影响力和字典
            this.morphTargetInfluences.push(0),
            this.morphTargetDictionary[this.geometry.morphTargets[a].name]=a
    }
};

# 根据名称获取形态目标的索引
THREE.Mesh.prototype.getMorphTargetIndexByName=function(a){
    # 如果形态目标字典中存在指定名称的形态目标
    if(void 0!==this.morphTargetDictionary[a])
        # 返回该形态目标的索引
        return this.morphTargetDictionary[a];
    # 否则输出警告信息并返回索引 0
    console.log("THREE.Mesh.getMorphTargetIndexByName: morph target "+a+" does not exist. Returning 0.");
    return 0
};

# 射线投射
THREE.Mesh.prototype.raycast=function(){
    # 初始化变量
    var a=new THREE.Matrix4,
        b=new THREE.Ray,
        c=new THREE.Sphere,
        d=new THREE.Vector3,
        e=new THREE.Vector3,
        f=new THREE.Vector3;
    return function(g,h){
        # 获取几何体的边界球
        var k=this.geometry;
        null===k.boundingSphere&&k.computeBoundingSphere();
        c.copy(k.boundingSphere);
        c.applyMatrix4(this.matrixWorld);
        # 判断射线是否与边界球相交
        if(!1!==g.ray.isIntersectionSphere(c)&&(a.getInverse(this.matrixWorld),b.copy(g.ray).applyMatrix4(a),null===k.boundingBox||!1!==b.isIntersectionBox(k.boundingBox)))
            # 如果几何体是缓冲几何体
            if(k instanceof THREE.BufferGeometry){
                var n=this.material;
                if(void 0!==n){
                    var p=k.attributes,
                        q,m,r=g.precision;
                    if(void 0!==p.index){
                        var t=s=p.position.array,
                            u=k.offsets;
                        0===u.length&&(u=[{start:0,count:t.length,index:0}]);
                        for(var v=0,y=u.length;v<y;++v)
                            for(var p=u[v].start,G=u[v].index,k=p,w=p+u[v].count;k<w;k+=3){
                                p=G+t[k];
                                q=G+t[k+1];
                                m=G+t[k+2];
                                d.fromArray(s,3*p);
                                e.fromArray(s,3*q);
                                f.fromArray(s,3*m);
                                var K=n.side===THREE.BackSide?b.intersectTriangle(f,e,d,!0):b.intersectTriangle(d,e,f,n.side!==THREE.DoubleSide);
                                if(null!==K){
                                    K.applyMatrix4(this.matrixWorld);
                                }
                            }
                    }
                }
            }
    }
};
# 计算射线原点到点 K 的距离
var x=g.ray.origin.distanceTo(K);
# 如果距离小于 r 或者大于 g.near 或者大于 g.far，则将点 K 加入 h 数组
x<r||x<g.near||x>g.far||h.push({distance:x,point:K,face:new THREE.Face3(p,q,m,THREE.Triangle.normal(d,e,f)),faceIndex:null,object:this})
# 如果条件不满足，则执行下面的代码
}}}else for(s=p.position.array,t=k=0,w=s.length;k<w;k+=3,t+=9)p=k,q=k+1,m=k+2,d.fromArray(s,t),e.fromArray(s,t+3),f.fromArray(s,t+6),K=n.side===THREE.BackSide?b.intersectTriangle(f,e,d,!0):b.intersectTriangle(d,e,f,n.side!==THREE.DoubleSide),null!==K&&(K.applyMatrix4(this.matrixWorld),x=g.ray.origin.distanceTo(K),x<r||x<g.near||x>
g.far||h.push({distance:x,point:K,face:new THREE.Face3(p,q,m,THREE.Triangle.normal(d,e,f)),faceIndex:null,object:this})
# 如果条件不满足，则执行下面的代码
}}else if(k instanceof THREE.Geometry)for(t=this.material instanceof THREE.MeshFaceMaterial,s=!0===t?this.material.materials:null,r=g.precision,u=k.vertices,v=0,y=k.faces.length;v<y;v++)if(G=k.faces[v],n=!0===t?s[G.materialIndex]:this.material,void 0!==n){p=u[G.a];q=u[G.b];m=u[G.c];if(!0===n.morphTargets){K=k.morphTargets;x=this.morphTargetInfluences;d.set(0,0,0);e.set(0,0,0);f.set(0,0,0);for(var w=0,D=K.length;w<D;w++){var E=x[w];if(0!==E){var A=K[w].vertices;d.x+=(A[G.a].x-p.x)*E;d.y+=(A[G.a].y-p.y)*E;d.z+=(A[G.a].z-p.z)*E;e.x+=(A[G.b].x-q.x)*E;e.y+=(A[G.b].y-q.y)*E;e.z+=(A[G.b].z-q.z)*E;f.x+=(A[G.c].x-m.x)*E;f.y+=(A[G.c].y-m.y)*E;f.z+=(A[G.c].z-m.z)*E}}d.add(p);e.add(q);f.add(m);p=d;q=e;m=f}K=n.side===THREE.BackSide?b.intersectTriangle(m,q,p,!0):b.intersectTriangle(p,q,m,n.side!==THREE.DoubleSide);null!==K&&(K.applyMatrix4(this.matrixWorld),x=g.ray.origin.distanceTo(K),x<r||x<g.near||x>g.far||h.push({distance:x,point:K,face:G,faceIndex:v,object:this}))}}();THREE.Mesh.prototype.clone=function(a,b){void 0===a&&(a=new THREE.Mesh(this.geometry,this.material));THREE.Object3D.prototype.clone.call(this,a,b);return a};THREE.Bone=function(a){THREE.Object3D.call(this);this.skin=a};THREE.Bone.prototype=Object.create(THREE.Object3D.prototype);
// 定义 THREE.Skeleton 构造函数，接受三个参数：a, b, c
THREE.Skeleton=function(a,b,c){
    // 如果 c 有值且不为 undefined，则使用顶点纹理，否则默认为 true
    this.useVertexTexture=void 0!==c?c:!0;
    // 创建一个单位矩阵
    this.identityMatrix=new THREE.Matrix4;
    // 如果 a 有值，则将其复制给 this.bones，否则 this.bones 为空数组
    a=a||[];
    this.bones=a.slice(0);
    // 如果使用顶点纹理
    if(this.useVertexTexture){
        // 根据骨骼数量确定骨骼纹理的宽高
        this.boneTextureHeight=this.boneTextureWidth=a=256<this.bones.length?64:64<this.bones.length?32:16<this.bones.length?16:8;
        // 创建骨骼矩阵的 Float32Array
        this.boneMatrices=new Float32Array(this.boneTextureWidth*this.boneTextureHeight*4);
        // 创建骨骼纹理
        this.boneTexture=new THREE.DataTexture(this.boneMatrices,this.boneTextureWidth,this.boneTextureHeight,THREE.RGBAFormat,THREE.FloatType);
        // 设置骨骼纹理的 minFilter、magFilter、generateMipmaps 和 flipY 属性
        this.boneTexture.minFilter=THREE.NearestFilter;
        this.boneTexture.magFilter=THREE.NearestFilter;
        this.boneTexture.generateMipmaps=!1;
        this.boneTexture.flipY=!1;
    } else {
        // 如果不使用顶点纹理，则创建骨骼矩阵的 Float32Array
        this.boneMatrices=new Float32Array(16*this.bones.length);
    }
    // 如果 b 为 undefined，则调用 calculateInverses 方法
    if(void 0===b)
        this.calculateInverses();
    else if(this.bones.length===b.length)
        // 如果 b 的长度与骨骼数量相等，则将其复制给 this.boneInverses
        this.boneInverses=b.slice(0);
    else {
        // 如果 b 的长度与骨骼数量不相等，则输出警告信息，并创建空的 this.boneInverses 数组
        for(console.warn("THREE.Skeleton bonInverses is the wrong length."),b=0,a=this.bones.length;b<a;b++)
            this.boneInverses.push(new THREE.Matrix4);
    }
};
// 定义 THREE.Skeleton 的 calculateInverses 方法
THREE.Skeleton.prototype.calculateInverses=function(){
    // 创建空的 this.boneInverses 数组
    this.boneInverses=[];
    // 遍历骨骼数组，计算每个骨骼的逆矩阵，并添加到 this.boneInverses 数组中
    for(var a=0,b=this.bones.length;a<b;a++){
        var c=new THREE.Matrix4;
        this.bones[a]&&c.getInverse(this.bones[a].matrixWorld);
        this.boneInverses.push(c);
    }
};
// 定义 THREE.Skeleton 的 pose 方法
THREE.Skeleton.prototype.pose=function(){
    // 遍历骨骼数组，将每个骨骼的世界矩阵的逆矩阵赋值给骨骼的世界矩阵
    for(var a,b=0,c=this.bones.length;b<c;b++)
        (a=this.bones[b])&&a.matrixWorld.getInverse(this.boneInverses[b]);
    // 重置变量 b 为 0，再次遍历骨骼数组
    b=0;
    for(c=this.bones.length;b<c;b++)
        if(a=this.bones[b])
            // 如果骨骼有父级，则将其局部矩阵与父级的世界矩阵的逆矩阵相乘，再分解成平移、旋转和缩放
            a.parent?(a.matrix.getInverse(a.parent.matrixWorld),a.matrix.multiply(a.matrixWorld)):a.matrix.copy(a.matrixWorld),a.matrix.decompose(a.position,a.quaternion,a.scale);
};
// 定义 THREE.Skeleton 的 update 方法
THREE.Skeleton.prototype.update=function(){
    // 创建一个单位矩阵
    var a=new THREE.Matrix4;
    return function(){
        // 遍历骨骼数组，将每个骨骼的世界矩阵与逆矩阵相乘，再将结果展平到骨骼矩阵的 Float32Array 中
        for(var b=0,c=this.bones.length;b<c;b++)
            a.multiplyMatrices(this.bones[b]?this.bones[b].matrixWorld:this.identityMatrix,this.boneInverses[b]),a.flattenToArrayOffset(this.boneMatrices,16*b);
        // 如果使用顶点纹理，则设置骨骼纹理需要更新
        this.useVertexTexture&&(this.boneTexture.needsUpdate=!0);
    }
}();
// 创建一个名为 SkinnedMesh 的函数，接受三个参数 a, b, c
THREE.SkinnedMesh=function(a,b,c){
    // 调用 Mesh 对象的构造函数，传入参数 a, b
    THREE.Mesh.call(this,a,b);
    // 设置当前对象的类型为 SkinnedMesh
    this.type="SkinnedMesh";
    // 设置绑定模式为 attached
    this.bindMode="attached";
    // 创建一个新的 4x4 矩阵作为绑定矩阵
    this.bindMatrix=new THREE.Matrix4;
    // 创建一个新的 4x4 矩阵作为绑定矩阵的逆矩阵
    this.bindMatrixInverse=new THREE.Matrix4;
    // 创建一个空数组 a
    a=[];
    // 如果当前对象有几何属性并且几何属性中有 bones 属性
    if(this.geometry&&void 0!==this.geometry.bones){
        // 遍历几何属性中的 bones 数组
        for(var d,e,f,g,h=0,k=this.geometry.bones.length;h<k;++h){
            // 获取当前骨骼的位置、旋转和缩放信息
            d=this.geometry.bones[h],e=d.pos,f=d.rotq,g=d.scl,
            // 创建一个新的 Bone 对象，并将其添加到数组 a 中
            b=new THREE.Bone(this),a.push(b),
            // 设置骨骼的名称、位置、四元数旋转
            b.name=d.name,b.position.set(e[0],e[1],e[2]),b.quaternion.set(f[0],f[1],f[2],f[3]),
            // 如果存在缩放信息，则设置骨骼的缩放
            void 0!==g?b.scale.set(g[0],g[1],g[2]):b.scale.set(1,1,1);
        }
        // 重新遍历 bones 数组
        h=0;
        for(k=this.geometry.bones.length;h<k;++h){
            // 获取当前骨骼的信息
            d=this.geometry.bones[h],
            // 如果当前骨骼有父级骨骼，则将其添加到父级骨骼下，否则添加到当前对象下
            -1!==d.parent?a[d.parent].add(a[h]):this.add(a[h])
        }
    }
    // 调用 normalizeSkinWeights 方法
    this.normalizeSkinWeights();
    // 更新世界矩阵
    this.updateMatrixWorld(!0);
    // 绑定骨骼
    this.bind(new THREE.Skeleton(a,void 0,c))
};
// 将 SkinnedMesh 的原型设置为 Mesh 的实例
THREE.SkinnedMesh.prototype=Object.create(THREE.Mesh.prototype);
// 绑定函数，接受一个参数 a 和一个可选参数 b
THREE.SkinnedMesh.prototype.bind=function(a,b){
    // 设置当前对象的骨骼为参数 a
    this.skeleton=a;
    // 如果参数 b 未定义，则更新世界矩阵并将其赋值给 b
    void 0===b&&(this.updateMatrixWorld(!0),b=this.matrixWorld);
    // 复制当前对象的世界矩阵到绑定矩阵
    this.bindMatrix.copy(b);
    // 获取绑定矩阵的逆矩阵
    this.bindMatrixInverse.getInverse(b)
};
// 姿势函数
THREE.SkinnedMesh.prototype.pose=function(){
    // 调用骨骼的 pose 方法
    this.skeleton.pose()
};
// 标准化皮肤权重函数
THREE.SkinnedMesh.prototype.normalizeSkinWeights=function(){
    // 如果几何属性是 Geometry 类型
    if(this.geometry instanceof THREE.Geometry){
        // 遍历皮肤权重数组
        for(var a=0;a<this.geometry.skinIndices.length;a++){
            var b=this.geometry.skinWeights[a],
            // 计算权重的长度曼哈顿距离的倒数
            c=1/b.lengthManhattan();
            // 如果不是无穷大，则将权重乘以该值，否则设置为 1
            Infinity!==c?b.multiplyScalar(c):b.set(1)
        }
    }
};
// 更新世界矩阵函数，接受一个参数 a
THREE.SkinnedMesh.prototype.updateMatrixWorld=function(a){
    // 调用 Mesh 的更新世界矩阵方法
    THREE.Mesh.prototype.updateMatrixWorld.call(this,!0);
    // 如果绑定模式为 attached，则获取当前对象的世界矩阵的逆矩阵
    "attached"===this.bindMode?this.bindMatrixInverse.getInverse(this.matrixWorld):
    // 如果绑定模式为 detached，则获取绑定矩阵的逆矩阵
    "detached"===this.bindMode?this.bindMatrixInverse.getInverse(this.bindMatrix):
    // 如果绑定模式未知，则输出警告信息
    console.warn("THREE.SkinnedMesh unreckognized bindMode: "+this.bindMode)
};
// 为 THREE.SkinnedMesh 的原型添加 clone 方法，用于克隆当前对象
THREE.SkinnedMesh.prototype.clone=function(a){
    // 如果未传入参数，则使用当前对象的几何体、材质和 useVertexTexture 属性创建新对象
    void 0===a&&(a=new THREE.SkinnedMesh(this.geometry,this.material,this.useVertexTexture));
    // 调用 THREE.Mesh 的 clone 方法，传入新对象，返回新对象
    THREE.Mesh.prototype.clone.call(this,a);
    // 返回新对象
    return a;
};

// 定义 THREE.MorphAnimMesh 类
THREE.MorphAnimMesh=function(a,b){
    // 调用 THREE.Mesh 构造函数，传入参数 a 和 b
    THREE.Mesh.call(this,a,b);
    // 设置类型为 "MorphAnimMesh"
    this.type="MorphAnimMesh";
    // 设置动画持续时间为 1000 毫秒
    this.duration=1E3;
    // 设置是否循环播放动画
    this.mirroredLoop=!1;
    // 设置当前关键帧、上一个关键帧和时间为 0
    this.currentKeyframe=this.lastKeyframe=this.time=0;
    // 设置动画播放方向为正向
    this.direction=1;
    // 设置是否反向播放动画为 false
    this.directionBackwards=!1;
    // 设置动画帧范围为 0 到几何体的形态目标数量减 1
    this.setFrameRange(0,this.geometry.morphTargets.length-1);
};

// 将 THREE.MorphAnimMesh 的原型设置为继承自 THREE.Mesh 的原型
THREE.MorphAnimMesh.prototype=Object.create(THREE.Mesh.prototype);

// 设置动画帧范围的方法
THREE.MorphAnimMesh.prototype.setFrameRange=function(a,b){
    // 设置起始关键帧和结束关键帧
    this.startKeyframe=a;
    this.endKeyframe=b;
    // 设置帧数
    this.length=this.endKeyframe-this.startKeyframe+1;
};

// 设置动画播放方向为正向的方法
THREE.MorphAnimMesh.prototype.setDirectionForward=function(){
    this.direction=1;
    this.directionBackwards=!1;
};

// 设置动画播放方向为反向的方法
THREE.MorphAnimMesh.prototype.setDirectionBackward=function(){
    this.direction=-1;
    this.directionBackwards=!0;
};

// 解析动画的方法
THREE.MorphAnimMesh.prototype.parseAnimations=function(){
    // 获取几何体
    var a=this.geometry;
    // 如果几何体没有动画属性，则创建 animations 对象
    a.animations||(a.animations={});
    // 遍历形态目标，解析动画
    for(var b,c=a.animations,d=/([a-z]+)_?(\d+)/,e=0,f=a.morphTargets.length;e<f;e++){
        var g=a.morphTargets[e].name.match(d);
        if(g&&1<g.length){
            g=g[1];
            c[g]||(c[g]={start:Infinity,end:-Infinity});
            var h=c[g];
            e<h.start&&(h.start=e);
            e>h.end&&(h.end=e);
            b||(b=g)
        }
    }
    // 设置第一个动画名称
    a.firstAnimation=b;
};

// 设置动画标签的方法
THREE.MorphAnimMesh.prototype.setAnimationLabel=function(a,b,c){
    // 如果几何体没有动画属性，则创建 animations 对象
    this.geometry.animations||(this.geometry.animations={});
    // 设置动画标签
    this.geometry.animations[a]={start:b,end:c};
};

// 播放动画的方法
THREE.MorphAnimMesh.prototype.playAnimation=function(a,b){
    // 获取动画信息
    var c=this.geometry.animations[a];
    // 如果存在该动画，则设置帧范围和持续时间
    c?(this.setFrameRange(c.start,c.end),this.duration=(c.end-c.start)/b*1E3,this.time=0):console.warn("animation["+a+"] undefined");
};
// 更新动画的时间和关键帧
THREE.MorphAnimMesh.prototype.updateAnimation=function(a){
    var b=this.duration/this.length; // 计算每个关键帧的持续时间
    this.time+=this.direction*a; // 根据动画方向和时间更新动画时间
    if(this.mirroredLoop){ // 如果是镜像循环
        if(this.time>this.duration||0>this.time) // 如果时间超出范围
            this.direction*=-1, // 反转动画方向
            this.time>this.duration&&(this.time=this.duration,this.directionBackwards=!0), // 如果时间超出最大值，设置时间为最大值
            0>this.time&&(this.time=0,this.directionBackwards=!1) // 如果时间小于0，设置时间为0
    }else 
        this.time%=this.duration, // 如果不是镜像循环，取余数
        0>this.time&&(this.time+=this.duration); // 如果时间小于0，加上持续时间
    a=this.startKeyframe+THREE.Math.clamp(Math.floor(this.time/b),0,this.length-1); // 根据时间计算当前关键帧
    a!==this.currentKeyframe&& // 如果当前关键帧不等于上一个关键帧
    (this.morphTargetInfluences[this.lastKeyframe]=0, // 上一个关键帧的影响度设为0
    this.morphTargetInfluences[this.currentKeyframe]=1, // 当前关键帧的影响度设为1
    this.morphTargetInfluences[a]=0, // 新的关键帧的影响度设为0
    this.lastKeyframe=this.currentKeyframe, // 更新上一个关键帧
    this.currentKeyframe=a); // 更新当前关键帧
    b=this.time%b/b; // 计算当前关键帧的时间比例
    this.directionBackwards&&(b=1-b); // 如果是反向播放，调整时间比例
    this.morphTargetInfluences[this.currentKeyframe]=b; // 设置当前关键帧的影响度
    this.morphTargetInfluences[this.lastKeyframe]=1-b; // 设置上一个关键帧的影响度
};

// 插值目标
THREE.MorphAnimMesh.prototype.interpolateTargets=function(a,b,c){
    for(var d=this.morphTargetInfluences,e=0,f=d.length;e<f;e++)d[e]=0; // 将所有目标的影响度设为0
    -1<a&&(d[a]=1-c); // 如果存在第一个目标，设置其影响度
    -1<b&&(d[b]=c); // 如果存在第二个目标，设置其影响度
};

// 克隆对象
THREE.MorphAnimMesh.prototype.clone=function(a){
    void 0===a&&(a=new THREE.MorphAnimMesh(this.geometry,this.material)); // 如果没有传入参数，创建一个新的对象
    a.duration=this.duration; // 设置新对象的动画持续时间
    a.mirroredLoop=this.mirroredLoop; // 设置新对象的镜像循环属性
    a.time=this.time; // 设置新对象的动画时间
    a.lastKeyframe=this.lastKeyframe; // 设置新对象的上一个关键帧
    a.currentKeyframe=this.currentKeyframe; // 设置新对象的当前关键帧
    a.direction=this.direction; // 设置新对象的动画方向
    a.directionBackwards=this.directionBackwards; // 设置新对象的反向播放属性
    THREE.Mesh.prototype.clone.call(this,a); // 调用父类的克隆方法
    return a; // 返回新对象
};

// 级别对象
THREE.LOD=function(){
    THREE.Object3D.call(this); // 调用父类的构造函数
    this.objects=[]; // 初始化对象数组
};

// 添加级别
THREE.LOD.prototype.addLevel=function(a,b){
    void 0===b&&(b=0); // 如果没有传入距离参数，默认为0
    b=Math.abs(b); // 取距离的绝对值
    for(var c=0;c<this.objects.length&&!(b<this.objects[c].distance);c++); // 遍历对象数组，找到合适的位置插入新对象
    this.objects.splice(c,0,{distance:b,object:a}); // 在合适的位置插入新对象
    this.add(a); // 将新对象添加到场景中
};

// 获取距离对应的对象
THREE.LOD.prototype.getObjectForDistance=function(a){
    for(var b=1,c=this.objects.length;b<c&&!(a<this.objects[b].distance);b++); // 遍历对象数组，找到距离对应的对象
    return this.objects[b-1].object; // 返回距离对应的对象
};
// 定义 LOD 对象的射线投射方法
THREE.LOD.prototype.raycast=function(){
    var a=new THREE.Vector3;
    return function(b,c){
        a.setFromMatrixPosition(this.matrixWorld);
        var d=b.ray.origin.distanceTo(a);
        this.getObjectForDistance(d).raycast(b,c)
    }
}();

// 定义 LOD 对象的更新方法
THREE.LOD.prototype.update=function(){
    var a=new THREE.Vector3,b=new THREE.Vector3;
    return function(c){
        if(1<this.objects.length){
            a.setFromMatrixPosition(c.matrixWorld);
            b.setFromMatrixPosition(this.matrixWorld);
            c=a.distanceTo(b);
            this.objects[0].object.visible=!0;
            for(var d=1,e=this.objects.length;d<e;d++){
                if(c>=this.objects[d].distance){
                    this.objects[d-1].object.visible=!1;
                    this.objects[d].object.visible=!0;
                } else {
                    break;
                }
            }
            for(;d<e;d++){
                this.objects[d].object.visible=!1;
            }
        }
    }
}();

// 定义 LOD 对象的克隆方法
THREE.LOD.prototype.clone=function(a){
    void 0===a&&(a=new THREE.LOD);
    THREE.Object3D.prototype.clone.call(this,a);
    for(var b=0,c=this.objects.length;b<c;b++){
        var d=this.objects[b].object.clone();
        d.visible=0===b;
        a.addLevel(d,this.objects[b].distance);
    }
    return a;
};

// 定义 Sprite 对象
THREE.Sprite=function(){
    var a=new Uint16Array([0,1,2,0,2,3]),
        b=new Float32Array([-.5,-.5,0,.5,-.5,0,.5,.5,0,-.5,.5,0]),
        c=new Float32Array([0,0,1,0,1,1,0,1]),
        d=new THREE.BufferGeometry;
    d.addAttribute("index",new THREE.BufferAttribute(a,1));
    d.addAttribute("position",new THREE.BufferAttribute(b,3));
    d.addAttribute("uv",new THREE.BufferAttribute(c,2));
    return function(a){
        THREE.Object3D.call(this);
        this.type="Sprite";
        this.geometry=d;
        this.material=void 0!==a?a:new THREE.SpriteMaterial
    }
}();

// 定义 Sprite 对象的射线投射方法
THREE.Sprite.prototype.raycast=function(){
    var a=new THREE.Vector3;
    return function(b,c){
        a.setFromMatrixPosition(this.matrixWorld);
        var d=b.ray.distanceToPoint(a);
        d>this.scale.x||c.push({distance:d,point:this.position,face:null,object:this})
    }
}();

// 定义 Sprite 对象的克隆方法
THREE.Sprite.prototype.clone=function(a){
    void 0===a&&(a=new THREE.Sprite(this.material));
    THREE.Object3D.prototype.clone.call(this,a);
    return a;
};

// 定义 Particle 对象为 Sprite 对象
THREE.Particle=THREE.Sprite;
# 创建一个名为 LensFlare 的函数，参数为 a, b, c, d, e
THREE.LensFlare=function(a,b,c,d,e){THREE.Object3D.call(this);
    # 初始化 lensFlares 数组
    this.lensFlares=[];
    # 初始化 positionScreen 为 Vector3 对象
    this.positionScreen=new THREE.Vector3;
    # 初始化 customUpdateCallback 为 undefined
    this.customUpdateCallback=void 0;
    # 如果 a 不为 undefined，则调用 add 方法
    void 0!==a&&this.add(a,b,c,d,e)};
# 设置 LensFlare 的原型为 Object3D 的实例
THREE.LensFlare.prototype=Object.create(THREE.Object3D.prototype);
# 定义 LensFlare 的 add 方法
THREE.LensFlare.prototype.add=function(a,b,c,d,e,f){
    # 如果 b 为 undefined，则设置为 -1
    void 0===b&&(b=-1);
    # 如果 c 为 undefined，则设置为 0
    void 0===c&&(c=0);
    # 如果 f 为 undefined，则设置为 1
    void 0===f&&(f=1);
    # 如果 e 为 undefined，则设置为颜色值 16777215
    void 0===e&&(e=new THREE.Color(16777215));
    # 如果 d 为 undefined，则设置为 NormalBlending
    void 0===d&&(d=THREE.NormalBlending);
    # 将 c 限制在 0 到 c 之间
    c=Math.min(c,Math.max(0,c));
    # 将参数添加到 lensFlares 数组中
    this.lensFlares.push({texture:a,size:b,distance:c,x:0,y:0,z:0,scale:1,rotation:1,opacity:f,color:e,blending:d})};
# 定义更新 lensFlares 的方法
THREE.LensFlare.prototype.updateLensFlares=function(){
    # 初始化变量 a, b, c, d, e
    var a,b=this.lensFlares.length,c,d=2*-this.positionScreen.x,e=2*-this.positionScreen.y;
    # 遍历 lensFlares 数组
    for(a=0;a<b;a++)
        # 更新 lensFlares 数组中的元素
        c=this.lensFlares[a],c.x=this.positionScreen.x+d*c.distance,c.y=this.positionScreen.y+e*c.distance,c.wantedRotation=c.x*Math.PI*.25,c.rotation+=.25*(c.wantedRotation-c.rotation)};
# 创建名为 Scene 的函数
THREE.Scene=function(){
    # 调用 Object3D 的构造函数
    THREE.Object3D.call(this);
    # 设置 type 属性为 "Scene"
    this.type="Scene";
    # 初始化 overrideMaterial 和 fog 为 null
    this.overrideMaterial=this.fog=null;
    # 设置 autoUpdate 为 true
    this.autoUpdate=!0};
# 设置 Scene 的原型为 Object3D 的实例
THREE.Scene.prototype=Object.create(THREE.Object3D.prototype);
# 定义 Scene 的 clone 方法
THREE.Scene.prototype.clone=function(a){
    # 如果 a 为 undefined，则设置为新的 Scene 对象
    void 0===a&&(a=new THREE.Scene);
    # 调用 Object3D 的 clone 方法
    THREE.Object3D.prototype.clone.call(this,a);
    # 如果 fog 不为 null，则克隆 fog
    null!==this.fog&&(a.fog=this.fog.clone());
    # 如果 overrideMaterial 不为 null，则克隆 overrideMaterial
    null!==this.overrideMaterial&&(a.overrideMaterial=this.overrideMaterial.clone());
    # 设置 autoUpdate 为当前对象的 autoUpdate
    a.autoUpdate=this.autoUpdate;
    # 设置 matrixAutoUpdate 为当前对象的 matrixAutoUpdate
    a.matrixAutoUpdate=this.matrixAutoUpdate;
    # 返回新的 Scene 对象
    return a};
# 创建名为 Fog 的函数
THREE.Fog=function(a,b,c){
    # 设置 name 为空字符串
    this.name="";
    # 设置 color 为颜色值 a
    this.color=new THREE.Color(a);
    # 如果 b 不为 undefined，则设置为 b，否则设置为 1
    this.near=void 0!==b?b:1;
    # 如果 c 不为 undefined，则设置为 c，否则设置为 1000
    this.far=void 0!==c?c:1E3};
# 设置粒子片段着色器的纹理映射参数
THREE.ShaderChunk.map_particle_pars_fragment="#ifdef USE_MAP\n\n\tuniform sampler2D map;\n\n#endif";
# 设置默认顶点着色器
THREE.ShaderChunk.default_vertex="vec4 mvPosition;\n\n#ifdef USE_SKINNING\n\n\tmvPosition = modelViewMatrix * skinned;\n\n#endif\n\n#if !defined( USE_SKINNING ) && defined( USE_MORPHTARGETS )\n\n\tmvPosition = modelViewMatrix * vec4( morphed, 1.0 );\n\n#endif\n\n#if !defined( USE_SKINNING ) && ! defined( USE_MORPHTARGETS )\n\n\tmvPosition = modelViewMatrix * vec4( position, 1.0 );\n\n#endif\n\ngl_Position = projectionMatrix * mvPosition;";
# 设置片段着色器的纹理映射参数
THREE.ShaderChunk.map_pars_fragment="#if defined( USE_MAP ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( USE_SPECULARMAP ) || defined( USE_ALPHAMAP )\n\n\tvarying vec2 vUv;\n\n#endif\n\n#ifdef USE_MAP\n\n\tuniform sampler2D map;\n\n#endif";
# 设置皮肤法线顶点着色器
THREE.ShaderChunk.skinnormal_vertex="#ifdef USE_SKINNING\n\n\tmat4 skinMatrix = mat4( 0.0 );\n\tskinMatrix += skinWeight.x * boneMatX;\n\tskinMatrix += skinWeight.y * boneMatY;\n\tskinMatrix += skinWeight.z * boneMatZ;\n\tskinMatrix += skinWeight.w * boneMatW;\n\tskinMatrix  = bindMatrixInverse * skinMatrix * bindMatrix;\n\n\t#ifdef USE_MORPHNORMALS\n\n\tvec4 skinnedNormal = skinMatrix * vec4( morphedNormal, 0.0 );\n\n\t#else\n\n\tvec4 skinnedNormal = skinMatrix * vec4( normal, 0.0 );\n\n\t#endif\n\n#endif\n";
# 设置片段着色器的雾参数
THREE.ShaderChunk.fog_pars_fragment="#ifdef USE_FOG\n\n\tuniform vec3 fogColor;\n\n\t#ifdef FOG_EXP2\n\n\t\tuniform float fogDensity;\n\n\t#else\n\n\t\tuniform float fogNear;\n\t\tuniform float fogFar;\n\t#endif\n\n#endif";
# 设置形态法线顶点着色器
THREE.ShaderChunk.morphnormal_vertex="#ifdef USE_MORPHNORMALS\n\n\tvec3 morphedNormal = vec3( 0.0 );\n\n\tmorphedNormal += ( morphNormal0 - normal ) * morphTargetInfluences[ 0 ];\n\tmorphedNormal += ( morphNormal1 - normal ) * morphTargetInfluences[ 1 ];\n\tmorphedNormal += ( morphNormal2 - normal ) * morphTargetInfluences[ 2 ];\n\tmorphedNormal += ( morphNormal3 - normal ) * morphTargetInfluences[ 3 ];\n\n\tmorphedNormal += normal;\n\n#endif";
# 定义环境贴图片段着色器的参数
THREE.ShaderChunk.envmap_pars_fragment="#ifdef USE_ENVMAP\n\n\tuniform float reflectivity;\n\tuniform samplerCube envMap;\n\tuniform float flipEnvMap;\n\tuniform int combine;\n\n\t#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG )\n\n\t\tuniform bool useRefract;\n\t\tuniform float refractionRatio;\n\n\t#else\n\n\t\tvarying vec3 vReflect;\n\n\t#endif\n\n#endif";
# 定义对数深度缓冲片段着色器的参数
THREE.ShaderChunk.logdepthbuf_fragment="#if defined(USE_LOGDEPTHBUF) && defined(USE_LOGDEPTHBUF_EXT)\n\n\tgl_FragDepthEXT = log2(vFragDepth) * logDepthBufFC * 0.5;\n\n#endif";
# 定义法线贴图片段着色器的参数
THREE.ShaderChunk.normalmap_pars_fragment="#ifdef USE_NORMALMAP\n\n\tuniform sampler2D normalMap;\n\tuniform vec2 normalScale;\n\n\t\t\t// Per-Pixel Tangent Space Normal Mapping\n\t\t\t// http://hacksoflife.blogspot.ch/2009/11/per-pixel-tangent-space-normal-mapping.html\n\n\tvec3 perturbNormal2Arb( vec3 eye_pos, vec3 surf_norm ) {\n\n\t\tvec3 q0 = dFdx( eye_pos.xyz );\n\t\tvec3 q1 = dFdy( eye_pos.xyz );\n\t\tvec2 st0 = dFdx( vUv.st );\n\t\tvec2 st1 = dFdy( vUv.st );\n\n\t\tvec3 S = normalize( q0 * st1.t - q1 * st0.t );\n\t\tvec3 T = normalize( -q0 * st1.s + q1 * st0.s );\n\t\tvec3 N = normalize( surf_norm );\n\n\t\tvec3 mapN = texture2D( normalMap, vUv ).xyz * 2.0 - 1.0;\n\t\tmapN.xy = normalScale * mapN.xy;\n\t\tmat3 tsn = mat3( S, T, N );\n\t\treturn normalize( tsn * mapN );\n\n\t}\n\n#endif\n";
# 定义冯氏光照模型顶点着色器的参数
THREE.ShaderChunk.lights_phong_pars_vertex="#if MAX_SPOT_LIGHTS > 0 || defined( USE_BUMPMAP ) || defined( USE_ENVMAP )\n\n\tvarying vec3 vWorldPosition;\n\n#endif\n";
# 定义光照贴图片段着色器的参数
THREE.ShaderChunk.lightmap_pars_fragment="#ifdef USE_LIGHTMAP\n\n\tvarying vec2 vUv2;\n\tuniform sampler2D lightMap;\n\n#endif";
# 定义阴影贴图顶点着色器的参数
THREE.ShaderChunk.shadowmap_vertex="#ifdef USE_SHADOWMAP\n\n\tfor( int i = 0; i < MAX_SHADOWS; i ++ ) {\n\n\t\tvShadowCoord[ i ] = shadowMatrix[ i ] * worldPosition;\n\n\t}\n\n#endif";
# 设置光照和冯氏着色模型的顶点着色器代码块
THREE.ShaderChunk.lights_phong_vertex="#if MAX_SPOT_LIGHTS > 0 || defined( USE_BUMPMAP ) || defined( USE_ENVMAP )\n\n\tvWorldPosition = worldPosition.xyz;\n\n#endif";
# 设置纹理映射的片段着色器代码块
THREE.ShaderChunk.map_fragment="#ifdef USE_MAP\n\n\tvec4 texelColor = texture2D( map, vUv );\n\n\t#ifdef GAMMA_INPUT\n\n\t\ttexelColor.xyz *= texelColor.xyz;\n\n\t#endif\n\n\tgl_FragColor = gl_FragColor * texelColor;\n\n#endif";
# 设置光照贴图的顶点着色器代码块
THREE.ShaderChunk.lightmap_vertex="#ifdef USE_LIGHTMAP\n\n\tvUv2 = uv2;\n\n#endif";
# 设置粒子纹理映射的片段着色器代码块
THREE.ShaderChunk.map_particle_fragment="#ifdef USE_MAP\n\n\tgl_FragColor = gl_FragColor * texture2D( map, vec2( gl_PointCoord.x, 1.0 - gl_PointCoord.y ) );\n\n#endif";
# 设置颜色的片段着色器代码块
THREE.ShaderChunk.color_pars_fragment="#ifdef USE_COLOR\n\n\tvarying vec3 vColor;\n\n#endif\n";
# 设置颜色的顶点着色器代码块
THREE.ShaderChunk.color_vertex="#ifdef USE_COLOR\n\n\t#ifdef GAMMA_INPUT\n\n\t\tvColor = color * color;\n\n\t#else\n\n\t\tvColor = color;\n\n\t#endif\n\n#endif";
# 设置皮肤动画的顶点着色器代码块
THREE.ShaderChunk.skinning_vertex="#ifdef USE_SKINNING\n\n\t#ifdef USE_MORPHTARGETS\n\n\tvec4 skinVertex = bindMatrix * vec4( morphed, 1.0 );\n\n\t#else\n\n\tvec4 skinVertex = bindMatrix * vec4( position, 1.0 );\n\n\t#endif\n\n\tvec4 skinned = vec4( 0.0 );\n\tskinned += boneMatX * skinVertex * skinWeight.x;\n\tskinned += boneMatY * skinVertex * skinWeight.y;\n\tskinned += boneMatZ * skinVertex * skinWeight.z;\n\tskinned += boneMatW * skinVertex * skinWeight.w;\n\tskinned  = bindMatrixInverse * skinned;\n\n#endif\n";
# 设置环境贴图的顶点着色器代码块
THREE.ShaderChunk.envmap_pars_vertex="#if defined( USE_ENVMAP ) && ! defined( USE_BUMPMAP ) && ! defined( USE_NORMALMAP ) && ! defined( PHONG )\n\n\tvarying vec3 vReflect;\n\n\tuniform float refractionRatio;\n\tuniform bool useRefract;\n\n#endif\n";
# 设置线性到伽马空间转换的片段着色器代码块
THREE.ShaderChunk.linear_to_gamma_fragment="#ifdef GAMMA_OUTPUT\n\n\tgl_FragColor.xyz = sqrt( gl_FragColor.xyz );\n\n#endif";
# 设置颜色的顶点着色器代码块
THREE.ShaderChunk.color_pars_vertex="#ifdef USE_COLOR\n\n\tvarying vec3 vColor;\n\n#endif";
# 定义顶点着色器中用于光照计算的参数
THREE.ShaderChunk.lights_lambert_pars_vertex="uniform vec3 ambient;\nuniform vec3 diffuse;\nuniform vec3 emissive;\n\nuniform vec3 ambientLightColor;\n\n#if MAX_DIR_LIGHTS > 0\n\n\tuniform vec3 directionalLightColor[ MAX_DIR_LIGHTS ];\n\tuniform vec3 directionalLightDirection[ MAX_DIR_LIGHTS ];\n\n#endif\n\n#if MAX_HEMI_LIGHTS > 0\n\n\tuniform vec3 hemisphereLightSkyColor[ MAX_HEMI_LIGHTS ];\n\tuniform vec3 hemisphereLightGroundColor[ MAX_HEMI_LIGHTS ];\n\tuniform vec3 hemisphereLightDirection[ MAX_HEMI_LIGHTS ];\n\n#endif\n\n#if MAX_POINT_LIGHTS > 0\n\n\tuniform vec3 pointLightColor[ MAX_POINT_LIGHTS ];\n\tuniform vec3 pointLightPosition[ MAX_POINT_LIGHTS ];\n\tuniform float pointLightDistance[ MAX_POINT_LIGHTS ];\n\n#endif\n\n#if MAX_SPOT_LIGHTS > 0\n\n\tuniform vec3 spotLightColor[ MAX_SPOT_LIGHTS ];\n\tuniform vec3 spotLightPosition[ MAX_SPOT_LIGHTS ];\n\tuniform vec3 spotLightDirection[ MAX_SPOT_LIGHTS ];\n\tuniform float spotLightDistance[ MAX_SPOT_LIGHTS ];\n\tuniform float spotLightAngleCos[ MAX_SPOT_LIGHTS ];\n\tuniform float spotLightExponent[ MAX_SPOT_LIGHTS ];\n\n#endif\n\n#ifdef WRAP_AROUND\n\n\tuniform vec3 wrapRGB;\n\n#endif\n";
# 定义顶点着色器中的 map_pars_vertex 字符串
THREE.ShaderChunk.map_pars_vertex="#if defined( USE_MAP ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( USE_SPECULARMAP ) || defined( USE_ALPHAMAP )\n\n\tvarying vec2 vUv;\n\tuniform vec4 offsetRepeat;\n\n#endif\n";
# 定义片元着色器中的 envmap_fragment 字符串
THREE.ShaderChunk.envmap_fragment="#ifdef USE_ENVMAP\n\n\tvec3 reflectVec;\n\n\t#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG )\n\n\t\tvec3 cameraToVertex = normalize( vWorldPosition - cameraPosition );\n\n\t\t// http://en.wikibooks.org/wiki/GLSL_Programming/Applying_Matrix_Transformations\n\t\t// Transforming Normal Vectors with the Inverse Transformation\n\n\t\tvec3 worldNormal = normalize( vec3( vec4( normal, 0.0 ) * viewMatrix ) );\n\n\t\tif ( useRefract ) {\n\n\t\t\treflectVec = refract( cameraToVertex, worldNormal, refractionRatio );\n\n\t\t} else { \n\n\t\t\treflectVec = reflect( cameraToVertex, worldNormal );\n\n\t\t}\n\n\t#else\n\n\t\treflectVec = vReflect;\n\n\t#endif\n\n\t#ifdef DOUBLE_SIDED\n\n\t\tfloat flipNormal = ( -1.0 + 2.0 * float( gl_FrontFacing ) );\n\t\tvec4 cubeColor = textureCube( envMap, flipNormal * vec3( flipEnvMap * reflectVec.x, reflectVec.yz ) );\n\n\t#else\n\n\t\tvec4 cubeColor = textureCube( envMap, vec3( flipEnvMap * reflectVec.x, reflectVec.yz ) );\n\n\t#endif\n\n\t#ifdef GAMMA_INPUT\n\n\t\tcubeColor.xyz *= cubeColor.xyz;\n\n\t#endif\n\n\tif ( combine == 1 ) {\n\n\t\tgl_FragColor.xyz = mix( gl_FragColor.xyz, cubeColor.xyz, specularStrength * reflectivity );\n\n\t} else if ( combine == 2 ) {\n\n\t\tgl_FragColor.xyz += cubeColor.xyz * specularStrength * reflectivity;\n\n\t} else {\n\n\t\tgl_FragColor.xyz = mix( gl_FragColor.xyz, gl_FragColor.xyz * cubeColor.xyz, specularStrength * reflectivity );\n\n\t}\n\n#endif";
# 定义了一个名为specularmap_pars_fragment的字符串变量，用于存储顶点着色器中的特定代码段
THREE.ShaderChunk.specularmap_pars_fragment="#ifdef USE_SPECULARMAP\n\n\tuniform sampler2D specularMap;\n\n#endif";
# 定义了一个名为logdepthbuf_vertex的字符串变量，用于存储顶点着色器中的特定代码段
THREE.ShaderChunk.logdepthbuf_vertex="#ifdef USE_LOGDEPTHBUF\n\n\tgl_Position.z = log2(max(1e-6, gl_Position.w + 1.0)) * logDepthBufFC;\n\n\t#ifdef USE_LOGDEPTHBUF_EXT\n\n\t\tvFragDepth = 1.0 + gl_Position.w;\n\n#else\n\n\t\tgl_Position.z = (gl_Position.z - 1.0) * gl_Position.w;\n\n\t#endif\n\n#endif";
# 定义了一个名为morphtarget_pars_vertex的字符串变量，用于存储顶点着色器中的特定代码段
THREE.ShaderChunk.morphtarget_pars_vertex="#ifdef USE_MORPHTARGETS\n\n\t#ifndef USE_MORPHNORMALS\n\n\tuniform float morphTargetInfluences[ 8 ];\n\n\t#else\n\n\tuniform float morphTargetInfluences[ 4 ];\n\n\t#endif\n\n#endif";
# 定义了一个名为specularmap_fragment的字符串变量，用于存储片元着色器中的特定代码段
THREE.ShaderChunk.specularmap_fragment="float specularStrength;\n\n#ifdef USE_SPECULARMAP\n\n\tvec4 texelSpecular = texture2D( specularMap, vUv );\n\tspecularStrength = texelSpecular.r;\n\n#else\n\n\tspecularStrength = 1.0;\n\n#endif";
# 定义了一个名为fog_fragment的字符串变量，用于存储片元着色器中的特定代码段
THREE.ShaderChunk.fog_fragment="#ifdef USE_FOG\n\n\t#ifdef USE_LOGDEPTHBUF_EXT\n\n\t\tfloat depth = gl_FragDepthEXT / gl_FragCoord.w;\n\n\t#else\n\n\t\tfloat depth = gl_FragCoord.z / gl_FragCoord.w;\n\n\t#endif\n\n\t#ifdef FOG_EXP2\n\n\t\tconst float LOG2 = 1.442695;\n\t\tfloat fogFactor = exp2( - fogDensity * fogDensity * depth * depth * LOG2 );\n\t\tfogFactor = 1.0 - clamp( fogFactor, 0.0, 1.0 );\n\n\t#else\n\n\t\tfloat fogFactor = smoothstep( fogNear, fogFar, depth );\n\n\t#endif\n\t\n\tgl_FragColor = mix( gl_FragColor, vec4( fogColor, gl_FragColor.w ), fogFactor );\n\n#endif";
# 定义了一个名为bumpmap_pars_fragment的字符串变量，包含了关于使用bumpmap的片段着色器代码
THREE.ShaderChunk.bumpmap_pars_fragment="#ifdef USE_BUMPMAP\n\n\tuniform sampler2D bumpMap;\n\tuniform float bumpScale;\n\n\t\t\t// Derivative maps - bump mapping unparametrized surfaces by Morten Mikkelsen\n\t\t\t//\thttp://mmikkelsen3d.blogspot.sk/2011/07/derivative-maps.html\n\n\t\t\t// Evaluate the derivative of the height w.r.t. screen-space using forward differencing (listing 2)\n\n\tvec2 dHdxy_fwd() {\n\n\t\tvec2 dSTdx = dFdx( vUv );\n\t\tvec2 dSTdy = dFdy( vUv );\n\n\t\tfloat Hll = bumpScale * texture2D( bumpMap, vUv ).x;\n\t\tfloat dBx = bumpScale * texture2D( bumpMap, vUv + dSTdx ).x - Hll;\n\t\tfloat dBy = bumpScale * texture2D( bumpMap, vUv + dSTdy ).x - Hll;\n\n\t\treturn vec2( dBx, dBy );\n\n\t}\n\n\tvec3 perturbNormalArb( vec3 surf_pos, vec3 surf_norm, vec2 dHdxy ) {\n\n\t\tvec3 vSigmaX = dFdx( surf_pos );\n\t\tvec3 vSigmaY = dFdy( surf_pos );\n\t\tvec3 vN = surf_norm;\t\t// normalized\n\n\t\tvec3 R1 = cross( vSigmaY, vN );\n\t\tvec3 R2 = cross( vN, vSigmaX );\n\n\t\tfloat fDet = dot( vSigmaX, R1 );\n\n\t\tvec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );\n\t\treturn normalize( abs( fDet ) * surf_norm - vGrad );\n\n\t}\n\n#endif";

# 定义了一个名为defaultnormal_vertex的字符串变量，包含了关于默认法线的顶点着色器代码
THREE.ShaderChunk.defaultnormal_vertex="vec3 objectNormal;\n\n#ifdef USE_SKINNING\n\n\tobjectNormal = skinnedNormal.xyz;\n\n#endif\n\n#if !defined( USE_SKINNING ) && defined( USE_MORPHNORMALS )\n\n\tobjectNormal = morphedNormal;\n\n#endif\n\n#if !defined( USE_SKINNING ) && ! defined( USE_MORPHNORMALS )\n\n\tobjectNormal = normal;\n\n#endif\n\n#ifdef FLIP_SIDED\n\n\tobjectNormal = -objectNormal;\n\n#endif\n\nvec3 transformedNormal = normalMatrix * objectNormal;";
# 定义光照的 Phong 模型的片段着色器代码块
THREE.ShaderChunk.lights_phong_pars_fragment="uniform vec3 ambientLightColor;\n\n#if MAX_DIR_LIGHTS > 0\n\n\tuniform vec3 directionalLightColor[ MAX_DIR_LIGHTS ];\n\tuniform vec3 directionalLightDirection[ MAX_DIR_LIGHTS ];\n\n#endif\n\n#if MAX_HEMI_LIGHTS > 0\n\n\tuniform vec3 hemisphereLightSkyColor[ MAX_HEMI_LIGHTS ];\n\tuniform vec3 hemisphereLightGroundColor[ MAX_HEMI_LIGHTS ];\n\tuniform vec3 hemisphereLightDirection[ MAX_HEMI_LIGHTS ];\n\n#endif\n\n#if MAX_POINT_LIGHTS > 0\n\n\tuniform vec3 pointLightColor[ MAX_POINT_LIGHTS ];\n\n\tuniform vec3 pointLightPosition[ MAX_POINT_LIGHTS ];\n\tuniform float pointLightDistance[ MAX_POINT_LIGHTS ];\n\n#endif\n\n#if MAX_SPOT_LIGHTS > 0\n\n\tuniform vec3 spotLightColor[ MAX_SPOT_LIGHTS ];\n\tuniform vec3 spotLightPosition[ MAX_SPOT_LIGHTS ];\n\tuniform vec3 spotLightDirection[ MAX_SPOT_LIGHTS ];\n\tuniform float spotLightAngleCos[ MAX_SPOT_LIGHTS ];\n\tuniform float spotLightExponent[ MAX_SPOT_LIGHTS ];\n\n\tuniform float spotLightDistance[ MAX_SPOT_LIGHTS ];\n\n#endif\n\n#if MAX_SPOT_LIGHTS > 0 || defined( USE_BUMPMAP ) || defined( USE_ENVMAP )\n\n\tvarying vec3 vWorldPosition;\n\n#endif\n\n#ifdef WRAP_AROUND\n\n\tuniform vec3 wrapRGB;\n\n#endif\n\nvarying vec3 vViewPosition;\nvarying vec3 vNormal;";
# 定义皮肤基础的顶点着色器代码块
THREE.ShaderChunk.skinbase_vertex="#ifdef USE_SKINNING\n\n\tmat4 boneMatX = getBoneMatrix( skinIndex.x );\n\tmat4 boneMatY = getBoneMatrix( skinIndex.y );\n\tmat4 boneMatZ = getBoneMatrix( skinIndex.z );\n\tmat4 boneMatW = getBoneMatrix( skinIndex.w );\n\n#endif";
# 定义贴图的顶点着色器代码块
THREE.ShaderChunk.map_vertex="#if defined( USE_MAP ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( USE_SPECULARMAP ) || defined( USE_ALPHAMAP )\n\n\tvUv = uv * offsetRepeat.zw + offsetRepeat.xy;\n\n#endif";
# 设置光照贴图片段着色器代码
THREE.ShaderChunk.lightmap_fragment="#ifdef USE_LIGHTMAP\n\n\tgl_FragColor = gl_FragColor * texture2D( lightMap, vUv2 );\n\n#endif";
# 设置阴影贴图顶点着色器代码
THREE.ShaderChunk.shadowmap_pars_vertex="#ifdef USE_SHADOWMAP\n\n\tvarying vec4 vShadowCoord[ MAX_SHADOWS ];\n\tuniform mat4 shadowMatrix[ MAX_SHADOWS ];\n\n#endif";
# 设置颜色片段着色器代码
THREE.ShaderChunk.color_fragment="#ifdef USE_COLOR\n\n\tgl_FragColor = gl_FragColor * vec4( vColor, 1.0 );\n\n#endif";
# 设置变形目标顶点着色器代码
THREE.ShaderChunk.morphtarget_vertex="#ifdef USE_MORPHTARGETS\n\n\tvec3 morphed = vec3( 0.0 );\n\tmorphed += ( morphTarget0 - position ) * morphTargetInfluences[ 0 ];\n\tmorphed += ( morphTarget1 - position ) * morphTargetInfluences[ 1 ];\n\tmorphed += ( morphTarget2 - position ) * morphTargetInfluences[ 2 ];\n\tmorphed += ( morphTarget3 - position ) * morphTargetInfluences[ 3 ];\n\n\t#ifndef USE_MORPHNORMALS\n\n\tmorphed += ( morphTarget4 - position ) * morphTargetInfluences[ 4 ];\n\tmorphed += ( morphTarget5 - position ) * morphTargetInfluences[ 5 ];\n\tmorphed += ( morphTarget6 - position ) * morphTargetInfluences[ 6 ];\n\tmorphed += ( morphTarget7 - position ) * morphTargetInfluences[ 7 ];\n\n\t#endif\n\n\tmorphed += position;\n\n#endif";
# 设置环境贴图顶点着色器代码
THREE.ShaderChunk.envmap_vertex="#if defined( USE_ENVMAP ) && ! defined( USE_BUMPMAP ) && ! defined( USE_NORMALMAP ) && ! defined( PHONG )\n\n\tvec3 worldNormal = mat3( modelMatrix[ 0 ].xyz, modelMatrix[ 1 ].xyz, modelMatrix[ 2 ].xyz ) * objectNormal;\n\tworldNormal = normalize( worldNormal );\n\n\tvec3 cameraToVertex = normalize( worldPosition.xyz - cameraPosition );\n\n\tif ( useRefract ) {\n\n\t\tvReflect = refract( cameraToVertex, worldNormal, refractionRatio );\n\n\t} else {\n\n\t\tvReflect = reflect( cameraToVertex, worldNormal );\n\n\t}\n\n#endif";
# 定义了 worldpos_vertex 字符串变量，包含了一系列条件判断和赋值操作
THREE.ShaderChunk.worldpos_vertex="#if defined( USE_ENVMAP ) || defined( PHONG ) || defined( LAMBERT ) || defined ( USE_SHADOWMAP )\n\n\t#ifdef USE_SKINNING\n\n\t\tvec4 worldPosition = modelMatrix * skinned;\n\n\t#endif\n\n\t#if defined( USE_MORPHTARGETS ) && ! defined( USE_SKINNING )\n\n\t\tvec4 worldPosition = modelMatrix * vec4( morphed, 1.0 );\n\n\t#endif\n\n\t#if ! defined( USE_MORPHTARGETS ) && ! defined( USE_SKINNING )\n\n\t\tvec4 worldPosition = modelMatrix * vec4( position, 1.0 );\n\n\t#endif\n\n#endif";
# 定义了 shadowmap_pars_fragment 字符串变量，包含了一系列条件判断和赋值操作
THREE.ShaderChunk.shadowmap_pars_fragment="#ifdef USE_SHADOWMAP\n\n\tuniform sampler2D shadowMap[ MAX_SHADOWS ];\n\tuniform vec2 shadowMapSize[ MAX_SHADOWS ];\n\n\tuniform float shadowDarkness[ MAX_SHADOWS ];\n\tuniform float shadowBias[ MAX_SHADOWS ];\n\n\tvarying vec4 vShadowCoord[ MAX_SHADOWS ];\n\n\tfloat unpackDepth( const in vec4 rgba_depth ) {\n\n\t\tconst vec4 bit_shift = vec4( 1.0 / ( 256.0 * 256.0 * 256.0 ), 1.0 / ( 256.0 * 256.0 ), 1.0 / 256.0, 1.0 );\n\t\tfloat depth = dot( rgba_depth, bit_shift );\n\t\treturn depth;\n\n\t}\n\n#endif";
# 定义了一个名为 skinning_pars_vertex 的字符串变量，包含了一些关于骨骼动画的顶点着色器代码
THREE.ShaderChunk.skinning_pars_vertex="#ifdef USE_SKINNING\n\n\tuniform mat4 bindMatrix;\n\tuniform mat4 bindMatrixInverse;\n\n\t#ifdef BONE_TEXTURE\n\n\t\tuniform sampler2D boneTexture;\n\t\tuniform int boneTextureWidth;\n\t\tuniform int boneTextureHeight;\n\n\t\tmat4 getBoneMatrix( const in float i ) {\n\n\t\t\tfloat j = i * 4.0;\n\t\t\tfloat x = mod( j, float( boneTextureWidth ) );\n\t\t\tfloat y = floor( j / float( boneTextureWidth ) );\n\n\t\t\tfloat dx = 1.0 / float( boneTextureWidth );\n\t\t\tfloat dy = 1.0 / float( boneTextureHeight );\n\n\t\t\ty = dy * ( y + 0.5 );\n\n\t\t\tvec4 v1 = texture2D( boneTexture, vec2( dx * ( x + 0.5 ), y ) );\n\t\t\tvec4 v2 = texture2D( boneTexture, vec2( dx * ( x + 1.5 ), y ) );\n\t\t\tvec4 v3 = texture2D( boneTexture, vec2( dx * ( x + 2.5 ), y ) );\n\t\t\tvec4 v4 = texture2D( boneTexture, vec2( dx * ( x + 3.5 ), y ) );\n\n\t\t\tmat4 bone = mat4( v1, v2, v3, v4 );\n\n\t\t\treturn bone;\n\n\t\t}\n\n\t#else\n\n\t\tuniform mat4 boneGlobalMatrices[ MAX_BONES ];\n\n\t\tmat4 getBoneMatrix( const in float i ) {\n\n\t\t\tmat4 bone = boneGlobalMatrices[ int(i) ];\n\t\t\treturn bone;\n\n\t\t}\n\n\t#endif\n\n#endif";
# 定义了一个名为 logdepthbuf_pars_fragment 的字符串变量，包含了一些关于对数深度缓冲的片元着色器代码
THREE.ShaderChunk.logdepthbuf_pars_fragment="#ifdef USE_LOGDEPTHBUF\n\n\tuniform float logDepthBufFC;\n\n\t#ifdef USE_LOGDEPTHBUF_EXT\n\n\t\t#extension GL_EXT_frag_depth : enable\n\t\tvarying float vFragDepth;\n\n\t#endif\n\n#endif";
# 定义了一个名为 alphamap_fragment 的字符串变量，包含了一些关于 alpha 贴图的片元着色器代码
THREE.ShaderChunk.alphamap_fragment="#ifdef USE_ALPHAMAP\n\n\tgl_FragColor.a *= texture2D( alphaMap, vUv ).g;\n\n#endif\n";
# 定义了一个名为 alphamap_pars_fragment 的字符串变量，包含了一些关于 alpha 贴图的片元着色器代码
THREE.ShaderChunk.alphamap_pars_fragment="#ifdef USE_ALPHAMAP\n\n\tuniform sampler2D alphaMap;\n\n#endif\n";
# 合并多个 Uniforms 对象，返回合并后的对象
THREE.UniformsUtils = {
    merge: function(a){
        # 创建一个空对象
        var b = {};
        # 遍历传入的 Uniforms 对象数组
        for(var c=0; c<a.length; c++){
            # 克隆当前 Uniforms 对象
            var d = this.clone(a[c]);
            # 遍历克隆后的对象
            for(var e in d)
                # 将克隆后的对象属性合并到新对象中
                b[e] = d[e];
        }
        # 返回合并后的对象
        return b;
    },
    # 克隆 Uniforms 对象
    clone: function(a){
        # 创建一个空对象
        var b = {};
        # 遍历传入的 Uniforms 对象
        for(var c in a){
            # 创建一个空对象
            b[c] = {};
            # 遍历当前对象的属性
            for(var d in a[c]){
                # 获取当前属性的值
                var e = a[c][d];
                # 如果属性值是颜色、向量、矩阵或纹理对象，则进行克隆
                b[c][d] = e instanceof THREE.Color || e instanceof THREE.Vector2 || e instanceof THREE.Vector3 || e instanceof THREE.Vector4 || e instanceof THREE.Matrix4 || e instanceof THREE.Texture ? e.clone() : e instanceof Array ? e.slice() : e;
            }
        }
        # 返回克隆后的对象
        return b;
    }
};

# 定义常用 Uniforms 对象
THREE.UniformsLib = {
    common: {
        # 漫反射颜色
        diffuse: {type: "c", value: new THREE.Color(15658734)},
        # 不透明度
        opacity: {type: "f", value: 1},
        # 纹理映射
        map: {type: "t", value: null},
        # 偏移和重复
        offsetRepeat: {type: "v4", value: new THREE.Vector4(0, 0, 1, 1)},
        # 光照贴图
        lightMap: {type: "t", value: null},
        # 镜面光贴图
        specularMap: {type: "t", value: null},
        # 透明度贴图
        alphaMap: {type: "t", value: null},
        # 环境贴图
        envMap: {type: "t", value: null},
        # 翻转环境贴图
        flipEnvMap: {type: "f", value: -1},
        # 使用折射
        useRefract: {type: "i", value: 0},
        # 反射率
        reflectivity: {type: "f", value: 1},
        # 折射率
        refractionRatio: {type: "f", value: .98},
        # 合并模式
        combine: {type: "i", value: 0},
        # 形变目标影响
        morphTargetInfluences: {type: "f", value: 0}
    },
    bump: {
        # 凹凸贴图
        bumpMap: {type: "t", value: null},
        # 凹凸贴图缩放
        bumpScale: {type: "f", value: 1}
    },
    normalmap: {
        # 法线贴图
        normalMap: {type: "t", value: null},
        # 法线贴图缩放
        normalScale: {type: "v2", value: new THREE.Vector2(1, 1)}
    },
    fog: {
        # 雾效密度
        fogDensity: {type: "f", value: 2.5E-4},
        # 雾效近处
        fogNear: {type: "f", value: 1},
        # 雾效远处
        fogFar: {type: "f", value: 2E3},
        # 雾效颜色
        fogColor: {type: "c", value: new THREE.Color(16777215)}
    },
    lights: {
        # 环境光颜色
        ambientLightColor: {type: "fv", value: []},
        # 方向光方向
        directionalLightDirection: {type: "fv", value: []},
        # 方向光颜色
        directionalLightColor: {type: "fv", value: []},
        # 半球光方向
        hemisphereLightDirection: {type: "fv", value: []},
        # 半球光天空颜色
        hemisphereLightSkyColor: {type: "fv", value: []},
        # 半球光地面颜色
        hemisphereLightGroundColor: {type: "fv", value: []},
        # 点光源颜色
        pointLightColor: {type: "fv", value: []},
        # 点光源位置
        pointLightPosition: {type: "fv", value: []},
        # 点光源距离
        pointLightDistance: {type: "fv1", value: []},
        # 聚光灯颜色
        spotLightColor: {type: "fv", value: []},
        # 聚光灯位置
        spotLightPosition: {type: "fv", value: []},
        # 聚光灯方向
        spotLightDirection: {type: "fv", value: []},
        # 聚光灯距离
        spotLightDistance: {type: "fv1", value: []},
        # 聚光灯角度余弦
        spotLightAngleCos: {type: "fv1", value: []},
        # 聚光灯指数
        spotLightExponent: {type: "fv1", value: []}
    },
    particle: {
        # 粒子颜色
        psColor: {type: "c", value: new THREE.Color(15658734)}
    }
};
# 定义一个包含各种属性的对象
opacity:{type:"f",value:1},size:{type:"f",value:1},scale:{type:"f",value:1},map:{type:"t",value:null},fogDensity:{type:"f",value:2.5E-4},fogNear:{type:"f",value:1},fogFar:{type:"f",value:2E3},fogColor:{type:"c",value:new THREE.Color(16777215)}},shadowmap:{shadowMap:{type:"tv",value:[]},shadowMapSize:{type:"v2v",value:[]},shadowBias:{type:"fv1",value:[]},shadowDarkness:{type:"fv1",value:[]},shadowMatrix:{type:"m4v",value:[]}}};

# 定义一个包含各种着色器的对象
THREE.ShaderLib={basic:{uniforms:THREE.UniformsUtils.merge([THREE.UniformsLib.common,THREE.UniformsLib.fog,THREE.UniformsLib.shadowmap]),vertexShader:[THREE.ShaderChunk.map_pars_vertex,THREE.ShaderChunk.lightmap_pars_vertex,THREE.ShaderChunk.envmap_pars_vertex,THREE.ShaderChunk.color_pars_vertex,THREE.ShaderChunk.morphtarget_pars_vertex,THREE.ShaderChunk.skinning_pars_vertex,THREE.ShaderChunk.shadowmap_pars_vertex,THREE.ShaderChunk.logdepthbuf_pars_vertex,"void main() {",THREE.ShaderChunk.map_vertex,
THREE.ShaderChunk.lightmap_vertex,THREE.ShaderChunk.color_vertex,THREE.ShaderChunk.skinbase_vertex,"\t#ifdef USE_ENVMAP",THREE.ShaderChunk.morphnormal_vertex,THREE.ShaderChunk.skinnormal_vertex,THREE.ShaderChunk.defaultnormal_vertex,"\t#endif",THREE.ShaderChunk.morphtarget_vertex,THREE.ShaderChunk.skinning_vertex,THREE.ShaderChunk.default_vertex,THREE.ShaderChunk.logdepthbuf_vertex,THREE.ShaderChunk.worldpos_vertex,THREE.ShaderChunk.envmap_vertex,THREE.ShaderChunk.shadowmap_vertex,"}"].join("\n"),
fragmentShader:["uniform vec3 diffuse;\nuniform float opacity;",THREE.ShaderChunk.color_pars_fragment,THREE.ShaderChunk.map_pars_fragment,THREE.ShaderChunk.alphamap_pars_fragment,THREE.ShaderChunk.lightmap_pars_fragment,THREE.ShaderChunk.envmap_pars_fragment,THREE.ShaderChunk.fog_pars_fragment,THREE.ShaderChunk.shadowmap_pars_fragment,THREE.ShaderChunk.specularmap_pars_fragment,THREE.ShaderChunk.logdepthbuf_pars_fragment,"void main() {\n\tgl_FragColor = vec4( diffuse, opacity );",THREE.ShaderChunk.logdepthbuf_fragment,
# 合并多个 ShaderChunk 片段，形成一个完整的 Lambert 着色器
THREE.ShaderChunk.map_fragment,
THREE.ShaderChunk.alphamap_fragment,
THREE.ShaderChunk.alphatest_fragment,
THREE.ShaderChunk.specularmap_fragment,
THREE.ShaderChunk.lightmap_fragment,
THREE.ShaderChunk.color_fragment,
THREE.ShaderChunk.envmap_fragment,
THREE.ShaderChunk.shadowmap_fragment,
THREE.ShaderChunk.linear_to_gamma_fragment,
THREE.ShaderChunk.fog_fragment,
"}"].join("\n")},
# 定义 Lambert 着色器的 uniform 变量
lambert:{
    uniforms:THREE.UniformsUtils.merge([
        THREE.UniformsLib.common,
        THREE.UniformsLib.fog,
        THREE.UniformsLib.lights,
        THREE.UniformsLib.shadowmap,
        {
            ambient:{type:"c",value:new THREE.Color(16777215)},
            emissive:{type:"c",value:new THREE.Color(0)},
            wrapRGB:{type:"v3",value:new THREE.Vector3(1,1,1)}
        }
    ]),
    # 定义 Lambert 着色器的顶点着色器
    vertexShader:[
        "#define LAMBERT",
        varying vec3 vLightFront,
        #ifdef DOUBLE_SIDED
            varying vec3 vLightBack,
        #endif",
        THREE.ShaderChunk.map_pars_vertex,
        THREE.ShaderChunk.lightmap_pars_vertex,
        THREE.ShaderChunk.envmap_pars_vertex,
        THREE.ShaderChunk.lights_lambert_pars_vertex,
        THREE.ShaderChunk.color_pars_vertex,
        THREE.ShaderChunk.morphtarget_pars_vertex,
        THREE.ShaderChunk.skinning_pars_vertex,
        THREE.ShaderChunk.shadowmap_pars_vertex,
        THREE.ShaderChunk.logdepthbuf_pars_vertex,
        "void main() {",
        THREE.ShaderChunk.map_vertex,
        THREE.ShaderChunk.lightmap_vertex,
        THREE.ShaderChunk.color_vertex,
        THREE.ShaderChunk.morphnormal_vertex,
        THREE.ShaderChunk.skinbase_vertex,
        THREE.ShaderChunk.skinnormal_vertex,
        THREE.ShaderChunk.defaultnormal_vertex,
        THREE.ShaderChunk.morphtarget_vertex,
        THREE.ShaderChunk.skinning_vertex,
        THREE.ShaderChunk.default_vertex,
        THREE.ShaderChunk.logdepthbuf_vertex,
        THREE.ShaderChunk.worldpos_vertex,
# 合并字符串数组，生成顶点着色器代码
THREE.ShaderChunk.envmap_vertex,THREE.ShaderChunk.lights_lambert_vertex,THREE.ShaderChunk.shadowmap_vertex,"}"].join("\n"),
# 合并字符串数组，生成片段着色器代码
fragmentShader:["uniform float opacity;\nvarying vec3 vLightFront;\n#ifdef DOUBLE_SIDED\n\tvarying vec3 vLightBack;\n#endif",THREE.ShaderChunk.color_pars_fragment,THREE.ShaderChunk.map_pars_fragment,THREE.ShaderChunk.alphamap_pars_fragment,THREE.ShaderChunk.lightmap_pars_fragment,THREE.ShaderChunk.envmap_pars_fragment,THREE.ShaderChunk.fog_pars_fragment,THREE.ShaderChunk.shadowmap_pars_fragment,
THREE.ShaderChunk.specularmap_pars_fragment,THREE.ShaderChunk.logdepthbuf_pars_fragment,"void main() {\n\tgl_FragColor = vec4( vec3( 1.0 ), opacity );",THREE.ShaderChunk.logdepthbuf_fragment,THREE.ShaderChunk.map_fragment,THREE.ShaderChunk.alphamap_fragment,THREE.ShaderChunk.alphatest_fragment,THREE.ShaderChunk.specularmap_fragment,"\t#ifdef DOUBLE_SIDED\n\t\tif ( gl_FrontFacing )\n\t\t\tgl_FragColor.xyz *= vLightFront;\n\t\telse\n\t\t\tgl_FragColor.xyz *= vLightBack;\n\t#else\n\t\tgl_FragColor.xyz *= vLightFront;\n\t#endif",
THREE.ShaderChunk.lightmap_fragment,THREE.ShaderChunk.color_fragment,THREE.ShaderChunk.envmap_fragment,THREE.ShaderChunk.shadowmap_fragment,THREE.ShaderChunk.linear_to_gamma_fragment,THREE.ShaderChunk.fog_fragment,"}"].join("\n")},
# 设置phong着色器的uniform变量
phong:{uniforms:THREE.UniformsUtils.merge([THREE.UniformsLib.common,THREE.UniformsLib.bump,THREE.UniformsLib.normalmap,THREE.UniformsLib.fog,THREE.UniformsLib.lights,THREE.UniformsLib.shadowmap,{ambient:{type:"c",value:new THREE.Color(16777215)},emissive:{type:"c",value:new THREE.Color(0)},
# 定义材质的光泽度，类型为颜色，值为新的三维颜色对象
specular:{type:"c",value:new THREE.Color(1118481)},
# 定义材质的反光度，类型为浮点数，值为30
shininess:{type:"f",value:30},
# 定义材质的包裹 RGB 值，类型为三维向量，值为新的三维向量对象
wrapRGB:{type:"v3",value:new THREE.Vector3(1,1,1)}
]),

# 定义顶点着色器
vertexShader:[
    "#define PHONG\nvarying vec3 vViewPosition;\nvarying vec3 vNormal;",
    THREE.ShaderChunk.map_pars_vertex,
    THREE.ShaderChunk.lightmap_pars_vertex,
    THREE.ShaderChunk.envmap_pars_vertex,
    THREE.ShaderChunk.lights_phong_pars_vertex,
    THREE.ShaderChunk.color_pars_vertex,
    THREE.ShaderChunk.morphtarget_pars_vertex,
    THREE.ShaderChunk.skinning_pars_vertex,
    THREE.ShaderChunk.shadowmap_pars_vertex,
    THREE.ShaderChunk.logdepthbuf_pars_vertex,
    "void main() {",
    THREE.ShaderChunk.map_vertex,
    THREE.ShaderChunk.lightmap_vertex,
    THREE.ShaderChunk.color_vertex,
    THREE.ShaderChunk.morphnormal_vertex,
    THREE.ShaderChunk.skinbase_vertex,
    THREE.ShaderChunk.skinnormal_vertex,
    THREE.ShaderChunk.defaultnormal_vertex,
    "\tvNormal = normalize( transformedNormal );",
    THREE.ShaderChunk.morphtarget_vertex,
    THREE.ShaderChunk.skinning_vertex,
    THREE.ShaderChunk.default_vertex,
    THREE.ShaderChunk.logdepthbuf_vertex,
    "\tvViewPosition = -mvPosition.xyz;",
    THREE.ShaderChunk.worldpos_vertex,
    THREE.ShaderChunk.envmap_vertex,
    THREE.ShaderChunk.lights_phong_vertex,
    THREE.ShaderChunk.shadowmap_vertex,
    "}"
].join("\n"),

# 定义片段着色器
fragmentShader:[
    "#define PHONG\nuniform vec3 diffuse;\nuniform float opacity;\nuniform vec3 ambient;\nuniform vec3 emissive;\nuniform vec3 specular;\nuniform float shininess;",
    THREE.ShaderChunk.color_pars_fragment,
    THREE.ShaderChunk.map_pars_fragment,
    THREE.ShaderChunk.alphamap_pars_fragment,
    THREE.ShaderChunk.lightmap_pars_fragment,
    THREE.ShaderChunk.envmap_pars_fragment,
    ...
# 合并了多个 THREE.ShaderChunk 的片段代码，用于定义着色器的片段
THREE.ShaderChunk.fog_pars_fragment,THREE.ShaderChunk.lights_phong_pars_fragment,THREE.ShaderChunk.shadowmap_pars_fragment,THREE.ShaderChunk.bumpmap_pars_fragment,THREE.ShaderChunk.normalmap_pars_fragment,THREE.ShaderChunk.specularmap_pars_fragment,THREE.ShaderChunk.logdepthbuf_pars_fragment,"void main() {\n\tgl_FragColor = vec4( vec3( 1.0 ), opacity );",THREE.ShaderChunk.logdepthbuf_fragment,THREE.ShaderChunk.map_fragment,THREE.ShaderChunk.alphamap_fragment,THREE.ShaderChunk.alphatest_fragment,THREE.ShaderChunk.specularmap_fragment,
THREE.ShaderChunk.lights_phong_fragment,THREE.ShaderChunk.lightmap_fragment,THREE.ShaderChunk.color_fragment,THREE.ShaderChunk.envmap_fragment,THREE.ShaderChunk.shadowmap_fragment,THREE.ShaderChunk.linear_to_gamma_fragment,THREE.ShaderChunk.fog_fragment,"}"].join("\n")},particle_basic:{uniforms:THREE.UniformsUtils.merge([THREE.UniformsLib.particle,THREE.UniformsLib.shadowmap]),vertexShader:["uniform float size;\nuniform float scale;",THREE.ShaderChunk.color_pars_vertex,THREE.ShaderChunk.shadowmap_pars_vertex,
THREE.ShaderChunk.logdepthbuf_pars_vertex,"void main() {",THREE.ShaderChunk.color_vertex,"\tvec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );\n\t#ifdef USE_SIZEATTENUATION\n\t\tgl_PointSize = size * ( scale / length( mvPosition.xyz ) );\n\t#else\n\t\tgl_PointSize = size;\n\t#endif\n\tgl_Position = projectionMatrix * mvPosition;",THREE.ShaderChunk.logdepthbuf_vertex,THREE.ShaderChunk.worldpos_vertex,THREE.ShaderChunk.shadowmap_vertex,"}"].join("\n"),fragmentShader:["uniform vec3 psColor;\nuniform float opacity;",
# 合并多个 ShaderChunk 片段，形成一个完整的片段着色器代码
THREE.ShaderChunk.color_pars_fragment,
THREE.ShaderChunk.map_particle_pars_fragment,
THREE.ShaderChunk.fog_pars_fragment,
THREE.ShaderChunk.shadowmap_pars_fragment,
THREE.ShaderChunk.logdepthbuf_pars_fragment,
"void main() {\n\tgl_FragColor = vec4( psColor, opacity );",
THREE.ShaderChunk.logdepthbuf_fragment,
THREE.ShaderChunk.map_particle_fragment,
THREE.ShaderChunk.alphatest_fragment,
THREE.ShaderChunk.color_fragment,
THREE.ShaderChunk.shadowmap_fragment,
THREE.ShaderChunk.fog_fragment,
"}"].join("\n")},

# 定义 dashed 着色器的 uniform 变量
dashed:{
    uniforms:THREE.UniformsUtils.merge([
        THREE.UniformsLib.common,
        THREE.UniformsLib.fog,
        {
            scale:{type:"f",value:1},
            dashSize:{type:"f",value:1},
            totalSize:{type:"f",value:2}
        }
    ]),
    # 定义 dashed 着色器的顶点着色器代码
    vertexShader:[
        "uniform float scale;\nattribute float lineDistance;\nvarying float vLineDistance;",
        THREE.ShaderChunk.color_pars_vertex,
        THREE.ShaderChunk.logdepthbuf_pars_vertex,
        "void main() {",
        THREE.ShaderChunk.color_vertex,
        "\tvLineDistance = scale * lineDistance;\n\tvec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );\n\tgl_Position = projectionMatrix * mvPosition;",
        THREE.ShaderChunk.logdepthbuf_vertex,
    ""].join("\n"),
    # 定义 dashed 着色器的片段着色器代码
    fragmentShader:[
        "uniform vec3 diffuse;\nuniform float opacity;\nuniform float dashSize;\nuniform float totalSize;\nvarying float vLineDistance;",
        THREE.ShaderChunk.color_pars_fragment,
        THREE.ShaderChunk.fog_pars_fragment,
        THREE.ShaderChunk.logdepthbuf_pars_fragment,
        "void main() {\n\tif ( mod( vLineDistance, totalSize ) > dashSize ) {\n\t\tdiscard;\n\t}\n\tgl_FragColor = vec4( diffuse, opacity );",
        THREE.ShaderChunk.logdepthbuf_fragment,
        THREE.ShaderChunk.color_fragment,
        THREE.ShaderChunk.fog_fragment,
"}"].join("\n")},depth:{uniforms:{mNear:{type:"f",value:1},mFar:{type:"f",value:2E3},opacity:{type:"f",value:1}},vertexShader:[THREE.ShaderChunk.morphtarget_pars_vertex,THREE.ShaderChunk.logdepthbuf_pars_vertex,"void main() {",THREE.ShaderChunk.morphtarget_vertex,THREE.ShaderChunk.default_vertex,THREE.ShaderChunk.logdepthbuf_vertex,"}"].join("\n"),
// 定义深度着色器，设置uniform变量mNear、mFar、opacity
fragmentShader:["uniform float mNear;\nuniform float mFar;\nuniform float opacity;",THREE.ShaderChunk.logdepthbuf_pars_fragment,"void main() {",THREE.ShaderChunk.logdepthbuf_fragment,
"\t#ifdef USE_LOGDEPTHBUF_EXT\n\t\tfloat depth = gl_FragDepthEXT / gl_FragCoord.w;\n\t#else\n\t\tfloat depth = gl_FragCoord.z / gl_FragCoord.w;\n\t#endif\n\tfloat color = 1.0 - smoothstep( mNear, mFar, depth );\n\tgl_FragColor = vec4( vec3( color ), opacity );\n}"].join("\n")},
// 定义深度着色器，设置uniform变量mNear、mFar、opacity，计算颜色和深度
normal:{uniforms:{opacity:{type:"f",value:1}},
// 定义法线着色器，设置uniform变量opacity
vertexShader:["varying vec3 vNormal;",THREE.ShaderChunk.morphtarget_pars_vertex,THREE.ShaderChunk.logdepthbuf_pars_vertex,"void main() {\n\tvNormal = normalize( normalMatrix * normal );",
THREE.ShaderChunk.morphtarget_vertex,THREE.ShaderChunk.default_vertex,THREE.ShaderChunk.logdepthbuf_vertex,"}"].join("\n"),
// 定义法线着色器，计算法线
fragmentShader:["uniform float opacity;\nvarying vec3 vNormal;",THREE.ShaderChunk.logdepthbuf_pars_fragment,"void main() {\n\tgl_FragColor = vec4( 0.5 * normalize( vNormal ) + 0.5, opacity );",THREE.ShaderChunk.logdepthbuf_fragment,"}"].join("\n")},
// 定义法线贴图着色器，设置uniform变量和合并其他uniform变量
# 定义一系列变量，包括类型和初始值
value:0},enableDiffuse:{type:"i",value:0},enableSpecular:{type:"i",value:0},enableReflection:{type:"i",value:0},enableDisplacement:{type:"i",value:0},tDisplacement:{type:"t",value:null},tDiffuse:{type:"t",value:null},tCube:{type:"t",value:null},tNormal:{type:"t",value:null},tSpecular:{type:"t",value:null},tAO:{type:"t",value:null},uNormalScale:{type:"v2",value:new THREE.Vector2(1,1)},uDisplacementBias:{type:"f",value:0},uDisplacementScale:{type:"f",value:1},diffuse:{type:"c",value:new THREE.Color(16777215)},
# 引入一些预定义的着色器代码片段
THREE.ShaderChunk.shadowmap_pars_fragment,THREE.ShaderChunk.fog_pars_fragment,THREE.ShaderChunk.logdepthbuf_pars_fragment,"void main() {",THREE.ShaderChunk.logdepthbuf_fragment,"\tgl_FragColor = vec4( vec3( 1.0 ), opacity );\n\tvec3 specularTex = vec3( 1.0 );\n\tvec3 normalTex = texture2D( tNormal, vUv ).xyz * 2.0 - 1.0;\n\tnormalTex.xy *= uNormalScale;\n\tnormalTex = normalize( normalTex );\n\tif( enableDiffuse ) {\n\t\t#ifdef GAMMA_INPUT\n\t\t\tvec4 texelColor = texture2D( tDiffuse, vUv );\n\t\t\ttexelColor.xyz *= texelColor.xyz;\n\t\t\tgl_FragColor = gl_FragColor * texelColor;\n\t\t#else\n\t\t\tgl_FragColor = gl_FragColor * texture2D( tDiffuse, vUv );\n\t\t#endif\n\t}\n\tif( enableAO ) {\n\t\t#ifdef GAMMA_INPUT\n\t\t\tvec4 aoColor = texture2D( tAO, vUv );\n\t\t\taoColor.xyz *= aoColor.xyz;\n\t\t\tgl_FragColor.xyz = gl_FragColor.xyz * aoColor.xyz;\n\t\t#else\n\t\t\tgl_FragColor.xyz = gl_FragColor.xyz * texture2D( tAO, vUv ).xyz;\n\t\t#endif\n\t}",
# 导入阴影映射、线性到伽马转换、雾效果的片段着色器代码
THREE.ShaderChunk.shadowmap_fragment,THREE.ShaderChunk.linear_to_gamma_fragment,THREE.ShaderChunk.fog_fragment,"}"].join("\n"),
# 顶点着色器代码，包括切线、偏移、重复、位移贴图等参数
vertexShader:["attribute vec4 tangent;\nuniform vec2 uOffset;\nuniform vec2 uRepeat;\nuniform bool enableDisplacement;\n#ifdef VERTEX_TEXTURES\n\tuniform sampler2D tDisplacement;\n\tuniform float uDisplacementScale;\n\tuniform float uDisplacementBias;\n#endif\nvarying vec3 vTangent;\nvarying vec3 vBinormal;\nvarying vec3 vNormal;\nvarying vec2 vUv;\nvarying vec3 vWorldPosition;\nvarying vec3 vViewPosition;",
# 导入对数深度缓冲的顶点着色器代码
THREE.ShaderChunk.logdepthbuf_vertex,"\tvWorldPosition = worldPosition.xyz;\n\tvViewPosition = -mvPosition.xyz;\n\t#ifdef USE_SHADOWMAP\n\t\tfor( int i = 0; i < MAX_SHADOWS; i ++ ) {\n\t\t\tvShadowCoord[ i ] = shadowMatrix[ i ] * worldPosition;\n\t\t}\n\t#endif\n}"].join("\n")},
# 立方体贴图的uniform变量和顶点着色器代码
cube:{uniforms:{tCube:{type:"t",value:null},tFlip:{type:"f",value:-1}},vertexShader:["varying vec3 vWorldPosition;",THREE.ShaderChunk.logdepthbuf_pars_vertex,"void main() {\n\tvec4 worldPosition = modelMatrix * vec4( position, 1.0 );\n\tvWorldPosition = worldPosition.xyz;\n\tgl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );",
# 立方体贴图的片段着色器代码，包括采样立方体贴图和对数深度缓冲
THREE.ShaderChunk.logdepthbuf_vertex,"}"].join("\n"),fragmentShader:["uniform samplerCube tCube;\nuniform float tFlip;\nvarying vec3 vWorldPosition;",THREE.ShaderChunk.logdepthbuf_pars_fragment,"void main() {\n\tgl_FragColor = textureCube( tCube, vec3( tFlip * vWorldPosition.x, vWorldPosition.yz ) );",THREE.ShaderChunk.logdepthbuf_fragment,"}"].join("\n")},
# 深度RGBA的uniform变量和顶点着色器代码，包括形变目标、皮肤、对数深度缓冲
depthRGBA:{uniforms:{},vertexShader:[THREE.ShaderChunk.morphtarget_pars_vertex,THREE.ShaderChunk.skinning_pars_vertex,THREE.ShaderChunk.logdepthbuf_pars_vertex,
# 定义一个名为 main 的函数
"void main() {",
# 导入 THREE.ShaderChunk 中的 skinbase_vertex
THREE.ShaderChunk.skinbase_vertex,
# 导入 THREE.ShaderChunk 中的 morphtarget_vertex
THREE.ShaderChunk.morphtarget_vertex,
# 导入 THREE.ShaderChunk 中的 skinning_vertex
THREE.ShaderChunk.skinning_vertex,
# 导入 THREE.ShaderChunk 中的 default_vertex
THREE.ShaderChunk.default_vertex,
# 导入 THREE.ShaderChunk 中的 logdepthbuf_vertex
THREE.ShaderChunk.logdepthbuf_vertex,
# 将上述导入的内容连接成一个字符串
"}"].join("\n"),
# 定义 fragmentShader，包括导入的 logdepthbuf_pars_fragment 和 pack_depth 函数的实现
fragmentShader:[THREE.ShaderChunk.logdepthbuf_pars_fragment,
"vec4 pack_depth( const in float depth ) {\n\tconst vec4 bit_shift = vec4( 256.0 * 256.0 * 256.0, 256.0 * 256.0, 256.0, 1.0 );\n\tconst vec4 bit_mask = vec4( 0.0, 1.0 / 256.0, 1.0 / 256.0, 1.0 / 256.0 );\n\tvec4 res = mod( depth * bit_shift * vec4( 255 ), vec4( 256 ) ) / vec4( 255 );\n\tres -= res.xxyz * bit_mask;\n\treturn res;\n}\nvoid main() {",
THREE.ShaderChunk.logdepthbuf_fragment,
# 根据条件使用不同的深度缓冲方式
"\t#ifdef USE_LOGDEPTHBUF_EXT\n\t\tgl_FragData[ 0 ] = pack_depth( gl_FragDepthEXT );\n\t#else\n\t\tgl_FragData[ 0 ] = pack_depth( gl_FragCoord.z );\n\t#endif\n}"].join("\n")}};
# 定义 THREE.WebGLRenderer 函数
THREE.WebGLRenderer=function(a){
# 定义函数 b
function b(a){
# 获取几何体和材质
var b=a.geometry;a=a.material;
# 获取顶点数量
var c=b.vertices.length;
# 检查材质的属性
if(a.attributes){
    # 初始化顶点属性
    void 0===b.__webglCustomAttributesList&&(b.__webglCustomAttributesList=[]);
    for(var d in a.attributes){
        var e=a.attributes[d];
        if(!e.__webglInitialized||e.createUniqueBuffers){
            e.__webglInitialized=!0;
            var f=1;
            "v2"===e.type?f=2:"v3"===e.type?f=3:"v4"===e.type?f=4:"c"===e.type&&(f=3);
            e.size=f;
            e.array=new Float32Array(c*f);
            e.buffer=l.createBuffer();
            e.buffer.belongsToAttribute=d;
            e.needsUpdate=!0
        }
        b.__webglCustomAttributesList.push(e)
    }
}
}
# 定义函数 c
function c(a,b){
    # 获取几何体和材质
    var c=b.geometry,e=a.faces3,f=3*e.length,g=1*e.length,h=3*e.length,e=d(b,a);
    # 初始化顶点、法线、颜色、UV 等数组
    a.__vertexArray=new Float32Array(3*f);
    a.__normalArray=new Float32Array(3*f);
    a.__colorArray=new Float32Array(3*f);
    a.__uvArray=new Float32Array(2*f);
    1<c.faceVertexUvs.length&&(a.__uv2Array=new Float32Array(2*f));
    c.hasTangents&&(a.__tangentArray=new Float32Array(4*f));
    b.geometry.skinWeights.length&&b.geometry.skinIndices.length&&(a.__skinIndexArray=new Float32Array(4*
}
// 创建 Float32Array 数组，用于存储顶点坐标
var f = a.numVertices,
    a.__vertexArray = new Float32Array(3 * f);
// 创建 Float32Array 数组，用于存储法线向量
var g = a.numVertices,
    a.__normalArray = new Float32Array(3 * g);
// 创建 Float32Array 数组，用于存储 UV 坐标
var h = a.numVertices,
    a.__uvArray = new Float32Array(2 * h);
// 创建 Float32Array 数组，用于存储权重
var j = a.numVertices,
    a.__skinIndexArray = new Float32Array(4 * j),
    a.__skinWeightArray = new Float32Array(4 * j);
// 判断是否支持 Uint32Array，根据情况选择使用 Uint32Array 或 Uint16Array
var c = null !== pa.get("OES_element_index_uint") && 21845 < g ? Uint32Array : Uint16Array;
a.__typeArray = c;
// 创建存储面的数组
a.__faceArray = new c(3 * g);
// 创建存储线的数组
a.__lineArray = new c(2 * h);
var k;
// 如果存在形变目标，则创建存储形变目标数据的数组
if (a.numMorphTargets) {
    a.__morphTargetsArrays = [];
    for (c = 0, k = a.numMorphTargets; c < k; c++) {
        a.__morphTargetsArrays.push(new Float32Array(3 * f));
    }
}
// 如果存在形变法线，则创建存储形变法线数据的数组
if (a.numMorphNormals) {
    a.__morphNormalsArrays = [];
    for (c = 0, k = a.numMorphNormals; c < k; c++) {
        a.__morphNormalsArrays.push(new Float32Array(3 * f));
    }
}
// 计算面的数量
a.__webglFaceCount = 3 * g;
// 计算线的数量
a.__webglLineCount = 2 * h;
// 如果存在自定义属性，则初始化并添加到列表中
if (e.attributes) {
    void 0 === a.__webglCustomAttributesList && (a.__webglCustomAttributesList = []);
    for (var m in e.attributes) {
        var g = e.attributes[m],
            h = {};
        for (n in g) {
            h[n] = g[n];
        }
        if (!h.__webglInitialized || h.createUniqueBuffers) {
            h.__webglInitialized = !0, c = 1, "v2" === h.type ? c = 2 : "v3" === h.type ? c = 3 : "v4" === h.type ? c = 4 : "c" === h.type && (c = 3), h.size = c, h.array = new Float32Array(f * c), h.buffer = l.createBuffer(), h.buffer.belongsToAttribute = m, g.needsUpdate = !0, h.__original = g;
        }
        a.__webglCustomAttributesList.push(h);
    }
}
a.__inittedArrays = !0;
}
// 返回材质的第 b 个面的材质
function d(a, b) {
    return a.material instanceof THREE.MeshFaceMaterial ? a.material.materials[b.materialIndex] : a.material;
}
// 设置顶点属性和缓冲区
function e(a, b, c, d) {
    c = c.attributes;
    var e = b.attributes;
    b = b.attributesKeys;
    for (var f = 0, k = b.length; f < k; f++) {
        var m = b[f],
            n = e[m];
        if (0 <= n) {
            var p = c[m];
            void 0 !== p ? (m = p.itemSize, l.bindBuffer(l.ARRAY_BUFFER, p.buffer), g(n), l.vertexAttribPointer(n, m, l.FLOAT, !1, 0, d * m * 4)) : void 0 !== a.defaultAttributeValues && (2 === a.defaultAttributeValues[m].length ? l.vertexAttrib2fv(n, a.defaultAttributeValues[m]) :
3===a.defaultAttributeValues[m].length&&l.vertexAttrib3fv(n,a.defaultAttributeValues[m]))}}h()}function f(){for(var a=0,b=wb.length;a<b;a++)wb[a]=0}function g(a){wb[a]=1;0===ib[a]&&(l.enableVertexAttribArray(a),ib[a]=1)}function h(){for(var a=0,b=ib.length;a<b;a++)ib[a]!==wb[a]&&(l.disableVertexAttribArray(a),ib[a]=0)}function k(a,b){return a.material.id!==b.material.id?b.material.id-a.material.id:a.z!==b.z?b.z-a.z:a.id-b.id}function n(a,b){return a.z!==b.z?a.z-b.z:a.id-b.id}function p(a,b){return b[0]-
a[0]}function q(a,e){if(!1!==e.visible){if(!(e instanceof THREE.Scene||e instanceof THREE.Group)){void 0===e.__webglInit&&(e.__webglInit=!0,e._modelViewMatrix=new THREE.Matrix4,e._normalMatrix=new THREE.Matrix3,e.addEventListener("removed",Hc));var f=e.geometry;if(void 0!==f&&void 0===f.__webglInit&&(f.__webglInit=!0,f.addEventListener("dispose",Ic),!(f instanceof THREE.BufferGeometry)))if(e instanceof THREE.Mesh)s(a,e,f);else if(e instanceof THREE.Line){if(void 0===f.__webglVertexBuffer){f.__webglVertexBuffer=
l.createBuffer();f.__webglColorBuffer=l.createBuffer();f.__webglLineDistanceBuffer=l.createBuffer();J.info.memory.geometries++;var g=f.vertices.length;f.__vertexArray=new Float32Array(3*g);f.__colorArray=new Float32Array(3*g);f.__lineDistanceArray=new Float32Array(1*g);f.__webglLineCount=g;b(e);f.verticesNeedUpdate=!0;f.colorsNeedUpdate=!0;f.lineDistancesNeedUpdate=!0}}else if(e instanceof THREE.PointCloud&&void 0===f.__webglVertexBuffer){f.__webglVertexBuffer=l.createBuffer();f.__webglColorBuffer=
// 创建缓冲区
l.createBuffer();
// 增加内存中几何体的数量
J.info.memory.geometries++;
// 获取顶点数量
var h=f.vertices.length;
// 创建顶点数据的 Float32Array 数组
f.__vertexArray=new Float32Array(3*h);
// 创建颜色数据的 Float32Array 数组
f.__colorArray=new Float32Array(3*h);
// 创建排序数组
f.__sortArray=[];
// 设置 WebGL 粒子数量
f.__webglParticleCount=h;
// 调用函数处理实体对象
b(e);
// 标记顶点数据需要更新
f.verticesNeedUpdate=!0;
// 标记颜色数据需要更新
f.colorsNeedUpdate=!0
}
// 如果对象没有定义 __webglActive 属性
if(void 0===e.__webglActive)
    // 设置 __webglActive 属性为 true
    e.__webglActive=!0,
    // 如果对象是 Mesh 类型
    e instanceof THREE.Mesh
        // 如果几何体是 BufferGeometry 类型
        ?f instanceof THREE.BufferGeometry
            // 调用函数处理几何体和实体对象
            u(ob,f,e)
            // 如果几何体是 Geometry 类型
            :f instanceof THREE.Geometry
                // 遍历几何体数组，调用函数处理几何体和实体对象
                for(var k=xb[f.id],m=0,n=k.length;m<n;m++)u(ob,k[m],e)
        // 如果对象是 Line 或 PointCloud 类型
        :e instanceof THREE.Line||e instanceof THREE.PointCloud
            // 调用函数处理对象
            ?u(ob,f,e)
            // 如果对象是 ImmediateRenderObject 或有 immediateRenderCallback 属性
            :e instanceof THREE.ImmediateRenderObject||e.immediateRenderCallback
                // 将对象添加到数组中
                &&jb.push({id:null,object:e,opaque:null,transparent:null,z:0});
// 如果对象是 Light 类型
if(e instanceof THREE.Light)
    // 将对象添加到数组中
    cb.push(e);
// 如果对象是 Sprite 类型
else if(e instanceof THREE.Sprite)
    // 将对象添加到数组中
    yb.push(e);
// 如果对象是 LensFlare 类型
else if(e instanceof THREE.LensFlare)
    // 将对象添加到数组中
    Ra.push(e);
// 如果对象是其他类型
else{
    // 获取对象的缓存数据
    var t=ob[e.id];
    // 如果缓存数据存在且对象不在视锥体内或者需要更新
    if(t&&(!1===e.frustumCulled||!0===Ec.intersectsObject(e))){
        // 获取对象的几何体
        var r=e.geometry,w,G;
        // 如果几何体是 BufferGeometry 类型
        if(r instanceof THREE.BufferGeometry)
            // 遍历几何体的属性
            for(var x=r.attributes,D=r.attributesKeys,E=0,B=D.length;E<B;E++){
                var A=D[E],K=x[A];
                // 如果属性的缓存不存在
                void 0===K.buffer&&(K.buffer=l.createBuffer(),K.needsUpdate=!0);
                // 如果属性需要更新
                if(!0===K.needsUpdate){
                    var F="index"===A?l.ELEMENT_ARRAY_BUFFER:l.ARRAY_BUFFER;
                    l.bindBuffer(F,K.buffer);
                    l.bufferData(F,K.array,l.STATIC_DRAW);
                    K.needsUpdate=!1
                }
            }
        // 如果对象是 Mesh 类型
        else if(e instanceof THREE.Mesh){
            // 如果几何体需要更新
            !0===r.groupsNeedUpdate&&s(a,e,r);
            // 获取几何体的缓存数据
            for(var H=xb[r.id],O=0,Q=H.length;O<Q;O++){
                var R=H[O];
                G=d(e,R);
                !0===r.groupsNeedUpdate&&c(R,e);
                w=G.attributes&&v(G);
                if(r.verticesNeedUpdate||r.morphTargetsNeedUpdate||r.elementsNeedUpdate||r.uvsNeedUpdate||r.normalsNeedUpdate||
# 检查顶点颜色、切线是否需要更新，或者是否有新的顶点数据
# 如果需要更新，则使用动态绘制，否则使用静态绘制
if r.colorsNeedUpdate or r.tangentsNeedUpdate or w:
    # 设置一些变量
    var C = R, P = e, S = l.DYNAMIC_DRAW, T = !r.dynamic, X = G
    # 检查是否已经初始化数组
    if C.__inittedArrays:
        # 检查是否需要平滑着色
        bb = X and void 0 !== X.shading and X.shading === THREE.SmoothShading
        M = void 0
        ea = void 0
        Y = void 0
        ca = void 0
        ma = void 0
        pa = void 0
        sa = void 0
        Fa = void 0
        la = void 0
        hb = void 0
        za = void 0
        aa = void 0
        $ = void 0
        Z = void 0
        ya = void 0
        qa = void 0
        L = void 0
        Ga = void 0
        na = void 0
        nc = void 0
        ia = void 0
        oc = void 0
        pc = void 0
        qc = void 0
        Ba = void 0
        zb = void 0
        Ab = void 0
        Ha = void 0
        Bb = void 0
        Aa = void 0
        va = void 0
        Cb = void 0
        Oa = void 0
        Qb = void 0
        Ma = void 0
        ib = void 0
        Ya = void 0
        Za = void 0
        uc = void 0
        Rb = void 0
        db = 0
        eb = 0
        qb = 0
        rb = 0
        Db = 0
        Sa = 0
        Ca = 0
        Pa = 0
        Ka = 0
        ja = 0
        ta = 0
        I = 0
        Ia = void 0
        Qa = C.__vertexArray
        sb = C.__uvArray
        fb = C.__uv2Array
        Ta = C.__normalArray
        ra = C.__tangentArray
        La = C.__colorArray
        Ua = C.__skinIndexArray
        Va = C.__skinWeightArray
        Eb = C.__morphTargetsArrays
        Jc = C.__morphNormalsArrays
        Kb = C.__webglCustomAttributesList
        z = void 0
        Sb = C.__faceArray
        Ja = C.__lineArray
        wa = P.geometry
        $a = wa.elementsNeedUpdate
        Kc = wa.uvsNeedUpdate
        ec = wa.normalsNeedUpdate
        da = wa.tangentsNeedUpdate
        wb = wa.colorsNeedUpdate
        U = wa.morphTargetsNeedUpdate
        fa = wa.vertices
        N = C.faces3
        xa = wa.faces
        ua = wa.faceVertexUvs[0]
        Lc = wa.faceVertexUvs[1]
        Fc = wa.skinIndices
        Tb = wa.skinWeights
        kb = wa.morphTargets
        Da = wa.morphNormals
        # 如果顶点需要更新
        if wa.verticesNeedUpdate:
            M = 0
            for ea = N.length:
                ca = xa[N[M]]
                aa = fa[ca.a]
                $ = fa[ca.b]
                Z = fa[ca.c]
                Qa[eb] = aa.x
                Qa[eb+1] = aa.y
                Qa[eb+2] = aa.z
                Qa[eb+3] = $.x
                Qa[eb+4] = $.y
                Qa[eb+5] = $.z
                Qa[eb+6] = Z.x
                Qa[eb+7] = Z.y
                Qa[eb+8] = Z.z
                eb += 9
            l.bindBuffer(l.ARRAY_BUFFER, C.__webglVertexBuffer)
# 如果存在顶点缓冲区，则将顶点数据存储到缓冲区中
if (U) {
    # 遍历每个三角形面的顶点数据
    for (Ma=0,ib=kb.length;Ma<ib;Ma++) {
        M=ta=0;
        # 遍历每个顶点的坐标
        for (ea=N.length;M<ea;M++) {
            uc=N[M];
            ca=xa[uc];
            aa=kb[Ma].vertices[ca.a];
            $=kb[Ma].vertices[ca.b];
            Z=kb[Ma].vertices[ca.c];
            Ya=Eb[Ma];
            # 存储顶点坐标到数组中
            Ya[ta]=aa.x;
            Ya[ta+1]=aa.y;
            Ya[ta+2]=aa.z;
            Ya[ta+3]=$.x;
            Ya[ta+4]=$.y;
            Ya[ta+5]=$.z;
            Ya[ta+6]=Z.x;
            Ya[ta+7]=Z.y;
            Ya[ta+8]=Z.z;
            # 如果启用了顶点法线变形
            if (X.morphNormals) {
                # 存储顶点法线到数组中
                if (bb) {
                    Rb=Da[Ma].vertexNormals[uc];
                    Ga=Rb.a;
                    na=Rb.b;
                    nc=Rb.c;
                } else {
                    nc=na=Ga=Da[Ma].faceNormals[uc];
                }
                Za=Jc[Ma];
                Za[ta]=Ga.x;
                Za[ta+1]=Ga.y;
                Za[ta+2]=Ga.z;
                Za[ta+3]=na.x;
                Za[ta+4]=na.y;
                Za[ta+5]=na.z;
                Za[ta+6]=nc.x;
                Za[ta+7]=nc.y;
                Za[ta+8]=nc.z;
            }
            ta+=9;
            # 将顶点数据存储到缓冲区中
            l.bindBuffer(l.ARRAY_BUFFER,C.__webglMorphTargetsBuffers[Ma]);
            l.bufferData(l.ARRAY_BUFFER,Eb[Ma],S);
            # 如果启用了顶点法线变形，则将顶点法线数据存储到缓冲区中
            if (X.morphNormals) {
                l.bindBuffer(l.ARRAY_BUFFER,C.__webglMorphNormalsBuffers[Ma]);
                l.bufferData(l.ARRAY_BUFFER,Jc[Ma],S);
            }
        }
    }
}
# 如果存在骨骼动画数据
if (Tb.length) {
    M=0;
    # 遍历每个顶点的骨骼索引和权重数据
    for (ea=N.length;M<ea;M++) {
        ca=xa[N[M]];
        qc=Tb[ca.a];
        Ba=Tb[ca.b];
        zb=Tb[ca.c];
        Va[ja]=qc.x;
        Va[ja+1]=qc.y;
        Va[ja+2]=qc.z;
        Va[ja+3]=qc.w;
        Va[ja+4]=Ba.x;
        Va[ja+5]=Ba.y;
        Va[ja+6]=Ba.z;
        Va[ja+7]=Ba.w;
        Va[ja+8]=zb.x;
        Va[ja+9]=zb.y;
        Va[ja+10]=zb.z;
        Va[ja+11]=zb.w;
        Ab=Fc[ca.a];
        Ha=Fc[ca.b];
        Bb=Fc[ca.c];
        Ua[ja]=Ab.x;
        Ua[ja+1]=Ab.y;
        Ua[ja+2]=Ab.z;
        Ua[ja+3]=Ab.w;
        Ua[ja+4]=Ha.x;
        Ua[ja+5]=Ha.y;
        Ua[ja+6]=Ha.z;
        Ua[ja+7]=Ha.w;
        Ua[ja+8]=Bb.x;
        Ua[ja+9]=Bb.y;
        Ua[ja+10]=Bb.z;
        Ua[ja+11]=Bb.w;
        ja+=12;
    }
    # 如果存在骨骼索引和权重数据，则将其存储到缓冲区中
    if (0<ja) {
        l.bindBuffer(l.ARRAY_BUFFER,C.__webglSkinIndicesBuffer);
        l.bufferData(l.ARRAY_BUFFER,Ua,S);
        l.bindBuffer(l.ARRAY_BUFFER,C.__webglSkinWeightsBuffer);
        l.bufferData(l.ARRAY_BUFFER,Va,S);
    }
}
# 如果存在顶点颜色数据
if (wb) {
    M=0;
    # 遍历每个顶点的颜色数据
    for (ea=N.length;M<ea;M++) {
        ca=xa[N[M]];
        sa=ca.vertexColors,
# 设置顶点颜色
Fa=ca.color,3===sa.length&&X.vertexColors===THREE.VertexColors?(ia=sa[0],oc=sa[1],pc=sa[2]):pc=oc=ia=Fa,
# 将顶点颜色分别赋值给ia, oc, pc
La[Ka]=ia.r,La[Ka+1]=ia.g,La[Ka+2]=ia.b,La[Ka+3]=oc.r,La[Ka+4]=oc.g,La[Ka+5]=oc.b,La[Ka+6]=pc.r,La[Ka+7]=pc.g,La[Ka+8]=pc.b,Ka+=9;
# 将顶点颜色的RGB值分别赋值给La数组，并更新索引Ka
0<Ka&&(l.bindBuffer(l.ARRAY_BUFFER,C.__webglColorBuffer),l.bufferData(l.ARRAY_BUFFER,La,S)}
# 如果Ka大于0，则绑定颜色缓冲区并将La数组的数据存入其中
if(da&&wa.hasTangents){
    # 如果da为真且wa有切线
    M=0;
    for(ea=N.length;M<ea;M++)
        # 遍历顶点索引数组N
        ca=xa[N[M]],la=ca.vertexTangents,ya=la[0],qa=la[1],L=la[2],
        # 获取顶点切线
        ra[Ca]=ya.x,ra[Ca+1]=ya.y,ra[Ca+2]=ya.z,ra[Ca+3]=ya.w,ra[Ca+4]=qa.x,
        ra[Ca+5]=qa.y,ra[Ca+6]=qa.z,ra[Ca+7]=qa.w,ra[Ca+8]=L.x,ra[Ca+9]=L.y,ra[Ca+10]=L.z,ra[Ca+11]=L.w,Ca+=12;
        # 将切线数据存入ra数组，并更新索引Ca
    l.bindBuffer(l.ARRAY_BUFFER,C.__webglTangentBuffer);l.bufferData(l.ARRAY_BUFFER,ra,S)}
    # 绑定切线缓冲区并将ra数组的数据存入其中
if(ec){
    # 如果ec为真
    M=0;
    for(ea=N.length;M<ea;M++)
        # 遍历顶点索引数组N
        if(ca=xa[N[M]],ma=ca.vertexNormals,pa=ca.normal,3===ma.length&&bb)
            # 如果顶点法线数组长度为3且bb为真
            for(Aa=0;3>Aa;Aa++)Cb=ma[Aa],Ta[Sa]=Cb.x,Ta[Sa+1]=Cb.y,Ta[Sa+2]=Cb.z,Sa+=3;
            # 将顶点法线数据存入Ta数组，并更新索引Sa
        else
            for(Aa=0;3>Aa;Aa++)Ta[Sa]=pa.x,Ta[Sa+1]=pa.y,Ta[Sa+2]=pa.z,Sa+=3;
            # 将顶点法线数据存入Ta数组，并更新索引Sa
    l.bindBuffer(l.ARRAY_BUFFER,C.__webglNormalBuffer);l.bufferData(l.ARRAY_BUFFER,Ta,S)}
    # 绑定法线缓冲区并将Ta数组的数据存入其中
if(Kc&&ua){
    # 如果Kc和ua都为真
    M=0;
    for(ea=N.length;M<ea;M++)
        # 遍历顶点索引数组N
        if(Y=N[M],hb=ua[Y],void 0!==hb)
            # 如果ua中存在索引为Y的数据
            for(Aa=0;3>Aa;Aa++)Oa=hb[Aa],sb[qb]=Oa.x,sb[qb+1]=Oa.y,qb+=2;
            # 将UV数据存入sb数组，并更新索引qb
    0<qb&&(l.bindBuffer(l.ARRAY_BUFFER,C.__webglUVBuffer),l.bufferData(l.ARRAY_BUFFER,sb,S))}
    # 如果qb大于0，则绑定UV缓冲区并将sb数组的数据存入其中
if(Kc&&Lc){
    # 如果Kc和Lc都为真
    M=0;
    for(ea=N.length;M<ea;M++)
        # 遍历顶点索引数组N
        if(Y=N[M],za=Lc[Y],void 0!==za)
            # 如果Lc中存在索引为Y的数据
            for(Aa=0;3>Aa;Aa++)Qb=za[Aa],fb[rb]=Qb.x,fb[rb+1]=Qb.y,rb+=2;
            # 将UV2数据存入fb数组，并更新索引rb
    0<rb&&(l.bindBuffer(l.ARRAY_BUFFER,C.__webglUV2Buffer),l.bufferData(l.ARRAY_BUFFER,fb,S))}
    # 如果rb大于0，则绑定UV2缓冲区并将fb数组的数据存入其中
if($a){
    # 如果$a为真
    M=0;
    for(ea=N.length;M<ea;M++)Sb[Db]=db,Sb[Db+
    # 遍历顶点索引数组N
# 循环开始，对数组进行遍历和操作
for(Aa=0,va=Kb.length;Aa<va;Aa++)
    # 获取当前数组元素
    if(z=Kb[Aa],z.__original.needsUpdate)
        I=0
        # 判断数组大小
        if(1===z.size)
            # 判断绑定属性
            if(void 0===z.boundTo||"vertices"===z.boundTo)
                # 对顶点数组进行操作
                for(M=0,ea=N.length;M<ea;M++)
                    ca=xa[N[M]],z.array[I]=z.value[ca.a],z.array[I+1]=z.value[ca.b],z.array[I+2]=z.value[ca.c],I+=3
            else
                # 对面数组进行操作
                if("faces"===z.boundTo)
                    for(M=0,ea=N.length;M<ea;M++)
                        Ia=z.value[N[M]],z.array[I]=Ia,z.array[I+1]=Ia,z.array[I+2]=Ia,I+=3
        # 判断数组大小
        else if(2===z.size)
            # 判断绑定属性
            if(void 0===z.boundTo||"vertices"===z.boundTo)
                # 对顶点数组进行操作
                for(M=0,ea=N.length;M<ea;M++)
                    ca=xa[N[M]],aa=z.value[ca.a],$=z.value[ca.b],Z=z.value[ca.c],z.array[I]=aa.x,z.array[I+1]=aa.y,z.array[I+2]=$.x,z.array[I+3]=$.y,z.array[I+4]=Z.x,z.array[I+5]=Z.y,I+=6
            else
                # 对面数组进行操作
                if("faces"===z.boundTo)
                    for(M=0,ea=N.length;M<ea;M++)
                        Z=$=aa=Ia=z.value[N[M]],z.array[I]=aa.x,z.array[I+1]=aa.y,z.array[I+2]=$.x,z.array[I+3]=$.y,z.array[I+4]=Z.x,z.array[I+5]=Z.y,I+=6
        # 判断数组大小
        else if(3===z.size)
            # 根据类型选择属性
            ka="c"===z.type?["r","g","b"]:["x","y","z"]
            # 判断绑定属性
            if(void 0===z.boundTo||"vertices"===z.boundTo)
                # 对顶点数组进行操作
                for(M=0,ea=N.length;M<ea;M++)
                    ca=xa[N[M]],aa=z.value[ca.a],$=z.value[ca.b],Z=z.value[ca.c],z.array[I]=aa[ka[0]],z.array[I+1]=aa[ka[1]],z.array[I+2]=aa[ka[2]],z.array[I+3]=$[ka[0]],z.array[I+4]=$[ka[1]],z.array[I+5]=$[ka[2]],z.array[I+6]=Z[ka[0]],z.array[I+
# 如果数组大小为9
if(9===z.size)
    # 如果没有指定绑定到的对象或者绑定到的对象是"vertices"
    if(void 0===z.boundTo||"vertices"===z.boundTo)
        # 遍历N数组
        for(M=0,ea=N.length;M<ea;M++)
            # 获取xa数组中对应索引的值
            ca=xa[N[M]]
            # 获取z.value中对应索引的值
            aa=z.value[ca.a]
            $=z.value[ca.b]
            Z=z.value[ca.c]
            # 将获取的值存入z.array数组中
            z.array[I]=aa.x
            z.array[I+1]=aa.y
            z.array[I+2]=aa.z
            z.array[I+3]=aa.w
            z.array[I+4]=$.x
            z.array[I+5]=$.y
            z.array[I+6]=$.z
            z.array[I+7]=$.w
            z.array[I+8]=Z.x
            z.array[I+9]=Z.y
            z.array[I+10]=Z.z
            z.array[I+11]=Z.w
            I+=12
    # 如果绑定到的对象是"faces"
    else if("faces"===z.boundTo)
        # 遍历N数组
        for(M=0,ea=N.length;M<ea;M++)
            # 获取z.value中对应索引的值
            Z=$=aa=Ia=z.value[N[M]]
            # 将获取的值存入z.array数组中
            z.array[I]=aa.x
            z.array[I+1]=aa.y
            z.array[I+2]=aa.z
            z.array[I+3]=aa.w
            z.array[I+4]=$.x
            z.array[I+5]=$.y
            z.array[I+6]=$.z
            z.array[I+7]=$.w
            z.array[I+8]=Z.x
            z.array[I+9]=Z.y
            z.array[I+10]=Z.z
            z.array[I+11]=Z.w
            I+=12
    # 如果绑定到的对象是"faceVertices"
    else if("faceVertices"===z.boundTo)
        # 遍历N数组
        for(M=0,ea=N.length;M<ea;M++)
            # 获取z.value中对应索引的值
            Ia=z.value[N[M]]
            aa=Ia[0]
            $=Ia[1]
            Z=Ia[2]
            # 将获取的值存入z.array数组中
            z.array[I]=aa.x
            z.array[I+1]=aa.y
            z.array[I+2]=aa.z
            z.array[I+3]=aa.w
            z.array[I+4]=$.x
            z.array[I+5]=$.y
            z.array[I+6]=$.z
            z.array[I+7]=$.w
            z.array[I+8]=Z.x
            z.array[I+9]=Z.y
            z.array[I+10]=Z.z
            z.array[I+11]=Z.w
            I+=12
# 如果数组大小为4
else if(4===z.size)
    # 如果没有指定绑定到的对象或者绑定到的对象是"vertices"
    if(void 0===z.boundTo||"vertices"===z.boundTo)
        # 遍历N数组
        for(M=0,ea=N.length;M<ea;M++)
            # 获取xa数组中对应索引的值
            ca=xa[N[M]]
            # 获取z.value中对应索引的值
            aa=z.value[ca.a]
            $=z.value[ca.b]
            Z=z.value[ca.c]
            # 将获取的值存入z.array数组中
            z.array[I]=aa.x
            z.array[I+1]=aa.y
            z.array[I+2]=aa.z
            z.array[I+3]=aa.w
            z.array[I+4]=$.x
            z.array[I+5]=$.y
            z.array[I+6]=$.z
            z.array[I+7]=$.w
            z.array[I+8]=Z.x
            z.array[I+9]=Z.y
            z.array[I+10]=Z.z
            z.array[I+11]=Z.w
            I+=12
    # 如果绑定到的对象是"faces"
    else if("faces"===z.boundTo)
        # 遍历N数组
        for(M=0,ea=N.length;M<ea;M++)
            # 获取z.value中对应索引的值
            Z=$=aa=Ia=z.value[N[M]]
            # 将获取的值存入z.array数组中
            z.array[I]=aa.x
            z.array[I+1]=aa.y
            z.array[I+2]=aa.z
            z.array[I+3]=aa.w
            z.array[I+4]=$.x
            z.array[I+5]=$.y
            z.array[I+6]=$.z
            z.array[I+7]=$.w
            z.array[I+8]=Z.x
            z.array[I+9]=Z.y
            z.array[I+10]=Z.z
            z.array[I+11]=Z.w
            I+=12
    # 如果绑定到的对象是"faceVertices"
    else if("faceVertices"===z.boundTo)
        # 遍历N数组
        for(M=0,ea=N.length;M<ea;M++)
            # 获取z.value中对应索引的值
            Ia=z.value[N[M]]
            aa=Ia[0]
            $=Ia[1]
            Z=Ia[2]
            # 将获取的值存入z.array数组中
            z.array[I]=aa.x
            z.array[I+1]=aa.y
            z.array[I+2]=aa.z
            z.array[I+3]=aa.w
            z.array[I+4]=$.x
            z.array[I+5]=$.y
            z.array[I+6]=$.z
# 设置数组中的元素值
z.array[I+7]=$.w,z.array[I+8]=Z.x,z.array[I+9]=Z.y,z.array[I+10]=Z.z,z.array[I+11]=Z.w,I+=12;
# 将数据绑定到缓冲区
l.bindBuffer(l.ARRAY_BUFFER,z.buffer);
# 将数据存储到缓冲区
l.bufferData(l.ARRAY_BUFFER,z.array,S)
# 如果存在T，则删除相关属性
T&&(delete C.__inittedArrays,delete C.__colorArray,delete C.__normalArray,delete C.__tangentArray,delete C.__uvArray,delete C.__uv2Array,delete C.__faceArray,delete C.__vertexArray,delete C.__lineArray,delete C.__skinIndexArray,delete C.__skinWeightArray)
# 更新顶点、形变目标、元素、UV、法线、颜色、切线
r.verticesNeedUpdate=!1;r.morphTargetsNeedUpdate=!1;r.elementsNeedUpdate=!1;r.uvsNeedUpdate=!1;r.normalsNeedUpdate=!1;r.colorsNeedUpdate=!1;r.tangentsNeedUpdate=!1;
# 如果存在G.attributes，则执行y(G)函数
G.attributes&&y(G)
# 如果e是THREE.Line的实例
else if(e instanceof THREE.Line){
    # 获取属性并判断是否需要更新
    G=d(e,r);w=G.attributes&&v(G);
    # 如果需要更新顶点、颜色、线距离或者w
    if(r.verticesNeedUpdate||r.colorsNeedUpdate||r.lineDistancesNeedUpdate||w){
        # 设置缓冲区的绘图模式
        var Zb=l.DYNAMIC_DRAW,ab,Fb,gb,$b,ga,vc,dc=r.vertices,fc=r.colors,Pb=r.lineDistances,kc=dc.length,lc=fc.length,mc=Pb.length,wc=r.__vertexArray,xc=r.__colorArray,jc=r.__lineDistanceArray,sc=r.colorsNeedUpdate,tc=r.lineDistancesNeedUpdate,gc=r.__webglCustomAttributesList,yc,Lb,Ea,hc,Wa,oa;
        # 如果顶点需要更新
        if(r.verticesNeedUpdate){
            for(ab=0;ab<kc;ab++)$b=dc[ab],ga=3*ab,wc[ga]=$b.x,wc[ga+1]=$b.y,wc[ga+2]=$b.z;
            l.bindBuffer(l.ARRAY_BUFFER,r.__webglVertexBuffer);
            l.bufferData(l.ARRAY_BUFFER,wc,Zb)
        }
        # 如果颜色需要更新
        if(sc){
            for(Fb=0;Fb<lc;Fb++)vc=fc[Fb],ga=3*Fb,xc[ga]=vc.r,xc[ga+1]=vc.g,xc[ga+2]=vc.b;
            l.bindBuffer(l.ARRAY_BUFFER,r.__webglColorBuffer);
            l.bufferData(l.ARRAY_BUFFER,xc,Zb)
        }
        # 如果线距离需要更新
        if(tc){
            for(gb=0;gb<mc;gb++)jc[gb]=Pb[gb];
            l.bindBuffer(l.ARRAY_BUFFER,r.__webglLineDistanceBuffer);
            l.bufferData(l.ARRAY_BUFFER,
# 循环遍历传入的参数 gc
jc,Zb)}if(gc)for(yc=0,Lb=gc.length;yc<Lb;yc++)
    # 如果当前元素的 needsUpdate 为真，并且 boundTo 未定义或者为 "vertices"
    if(oa=gc[yc],oa.needsUpdate&&(void 0===oa.boundTo||"vertices"===oa.boundTo))
        # 初始化变量
        ga=0;hc=oa.value.length;
        # 根据不同的大小进行不同的处理
        if(1===oa.size)
            # 处理大小为 1 的情况
            for(Ea=0;Ea<hc;Ea++)oa.array[Ea]=oa.value[Ea];
        else if(2===oa.size)
            # 处理大小为 2 的情况
            for(Ea=0;Ea<hc;Ea++)Wa=oa.value[Ea],oa.array[ga]=Wa.x,oa.array[ga+1]=Wa.y,ga+=2;
        else if(3===oa.size)
            # 处理大小为 3 的情况
            if("c"===oa.type)
                for(Ea=0;Ea<hc;Ea++)Wa=oa.value[Ea],oa.array[ga]=Wa.r,oa.array[ga+1]=Wa.g,oa.array[ga+2]=Wa.b,ga+=3;
            else
                for(Ea=0;Ea<hc;Ea++)Wa=oa.value[Ea],oa.array[ga]=Wa.x,oa.array[ga+1]=Wa.y,oa.array[ga+2]=Wa.z,ga+=3;
        else if(4===oa.size)
            # 处理大小为 4 的情况
            for(Ea=0;Ea<hc;Ea++)Wa=oa.value[Ea],oa.array[ga]=Wa.x,oa.array[ga+1]=Wa.y,oa.array[ga+2]=Wa.z,oa.array[ga+3]=Wa.w,ga+=4;
        # 绑定缓冲区并传入数据
        l.bindBuffer(l.ARRAY_BUFFER,oa.buffer);l.bufferData(l.ARRAY_BUFFER,oa.array,Zb)
# 更新顶点、颜色、线距离的标志
r.verticesNeedUpdate=!1;r.colorsNeedUpdate=!1;r.lineDistancesNeedUpdate=!1;
# 如果存在属性并且满足条件
G.attributes&&y(G)
    # 如果 e 是 THREE.PointCloud 类型
else if(e instanceof THREE.PointCloud)
    # 获取属性并判断是否需要更新
    G=d(e,r);w=G.attributes&&v(G);
    # 如果需要更新
    if(r.verticesNeedUpdate||r.colorsNeedUpdate||e.sortParticles||w)
        # 初始化变量
        var Mb=l.DYNAMIC_DRAW,Xa,tb,ub,W,vb,Ub,zc=r.vertices,pb=zc.length,Nb=r.colors,Ob=Nb.length,ac=r.__vertexArray,bc=r.__colorArray,Gb=r.__sortArray,Xb=r.verticesNeedUpdate,Yb=r.colorsNeedUpdate,Hb=r.__webglCustomAttributesList,lb,ic,ba,mb,ha,V;
        # 如果需要对粒子进行排序
        if(e.sortParticles)
            # 复制并应用世界矩阵
            Gc.copy(Ac);Gc.multiply(e.matrixWorld);
            # 对顶点进行排序
            for(Xa=0;Xa<pb;Xa++)ub=zc[Xa],Na.copy(ub),Na.applyProjection(Gc),Gb[Xa]=[Na.z,Xa];Gb.sort(p);
            # 根据排序结果重新排列顶点和颜色数组
            for(Xa=0;Xa<pb;Xa++)ub=zc[Gb[Xa][1]],W=3*Xa,ac[W]=ub.x,ac[W+1]=ub.y,ac[W+2]=ub.z;
            for(tb=0;tb<Ob;tb++)W=3*tb,Ub=Nb[Gb[tb][1]],
# 设置颜色值到数组中
bc[W]=Ub.r,  # 将Ub对象的红色值赋给数组bc的第W个元素
bc[W+1]=Ub.g,  # 将Ub对象的绿色值赋给数组bc的第W+1个元素
bc[W+2]=Ub.b;  # 将Ub对象的蓝色值赋给数组bc的第W+2个元素
if(Hb)  # 如果Hb存在
for(lb=0,ic=Hb.length;lb<ic;lb++)  # 遍历Hb数组
if(V=Hb[lb],void 0===V.boundTo||"vertices"===V.boundTo)  # 如果V是Hb数组中的元素，并且V.boundTo不存在或者等于"vertices"
if(W=0,mb=V.value.length,1===V.size)  # 设置W为0，mb为V.value的长度，如果V.size等于1
for(ba=0;ba<mb;ba++)  # 遍历mb
vb=Gb[ba][1],  # 获取Gb数组中第ba个元素的第1个值
V.array[ba]=V.value[vb];  # 将V.value中索引为vb的值赋给V.array中索引为ba的值
else if(2===V.size)  # 如果V.size等于2
for(ba=0;ba<mb;ba++)  # 遍历mb
vb=Gb[ba][1],  # 获取Gb数组中第ba个元素的第1个值
ha=V.value[vb],  # 获取V.value中索引为vb的值
V.array[W]=ha.x,  # 将ha.x赋给V.array中索引为W的值
V.array[W+1]=ha.y,  # 将ha.y赋给V.array中索引为W+1的值
W+=2;  # W加2
else if(3===V.size)  # 如果V.size等于3
if("c"===V.type)  # 如果V.type等于"c"
for(ba=0;ba<mb;ba++)  # 遍历mb
vb=Gb[ba][1],  # 获取Gb数组中第ba个元素的第1个值
ha=V.value[vb],  # 获取V.value中索引为vb的值
V.array[W]=ha.r,  # 将ha.r赋给V.array中索引为W的值
V.array[W+1]=ha.g,  # 将ha.g赋给V.array中索引为W+1的值
V.array[W+2]=ha.b,  # 将ha.b赋给V.array中索引为W+2的值
W+=3;  # W加3
else for(ba=0;ba<mb;ba++)  # 否则遍历mb
vb=Gb[ba][1],  # 获取Gb数组中第ba个元素的第1个值
ha=V.value[vb],  # 获取V.value中索引为vb的值
V.array[W]=ha.x,  # 将ha.x赋给V.array中索引为W的值
V.array[W+1]=ha.y,  # 将ha.y赋给V.array中索引为W+1的值
V.array[W+2]=ha.z,  # 将ha.z赋给V.array中索引为W+2的值
W+=3;  # W加3
else if(4===V.size)  # 如果V.size等于4
for(ba=0;ba<mb;ba++)  # 遍历mb
vb=Gb[ba][1],  # 获取Gb数组中第ba个元素的第1个值
ha=V.value[vb],  # 获取V.value中索引为vb的值
V.array[W]=ha.x,  # 将ha.x赋给V.array中索引为W的值
V.array[W+1]=ha.y,  # 将ha.y赋给V.array中索引为W+1的值
V.array[W+2]=ha.z,  # 将ha.z赋给V.array中索引为W+2的值
V.array[W+3]=ha.w,  # 将ha.w赋给V.array中索引为W+3的值
W+=4  # W加4
else  # 否则
if(Xb)  # 如果Xb存在
for(Xa=0;Xa<pb;Xa++)  # 遍历pb
ub=zc[Xa],  # 获取zc数组中索引为Xa的值
W=3*Xa,  # W等于3乘以Xa
ac[W]=ub.x,  # 将ub.x赋给ac数组中索引为W的值
ac[W+1]=ub.y,  # 将ub.y赋给ac数组中索引为W+1的值
ac[W+2]=ub.z;  # 将ub.z赋给ac数组中索引为W+2的值
if(Yb)  # 如果Yb存在
for(tb=0;tb<Ob;tb++)  # 遍历Ob
Ub=Nb[tb],  # 获取Nb数组中索引为tb的值
W=3*tb,  # W等于3乘以tb
bc[W]=Ub.r,  # 将Ub.r赋给bc数组中索引为W的值
bc[W+1]=Ub.g,  # 将Ub.g赋给bc数组中索引为W+1的值
bc[W+2]=Ub.b;  # 将Ub.b赋给bc数组中索引为W+2的值
if(Hb)  # 如果Hb存在
for(lb=0,ic=Hb.length;lb<ic;lb++)  # 遍历Hb数组
if(V=Hb[lb],V.needsUpdate&&(void 0===V.boundTo||"vertices"===V.boundTo))  # 如果V是Hb数组中的元素，并且V.needsUpdate为真，并且V.boundTo不存在或者等于"vertices"
if(mb=V.value.length,  # 设置mb为V.value的长度
W=0,  # 设置W为0
1===V.size)  # 如果V.size等于1
for(ba=0;ba<mb;ba++)  # 遍历mb
V.array[ba]=V.value[ba];  # 将V.value中索引为ba的值赋给V.array中索引为ba的值
else if(2===V.size)  # 如果V.size等于2
for(ba=0;ba<mb;ba++)  # 遍历mb
ha=V.value[ba],  # 获取V.value中索引为ba的值
V.array[W]=ha.x,  # 将ha.x赋给V.array中索引为W的值
V.array[W+1]=ha.y,  # 将ha.y赋给V.array中索引为W+1的值
W+=2;  # W加2
else if(3===V.size)  # 如果V.size等于3
if("c"===V.type)  # 如果V.type等于"c"
for(ba=0;ba<mb;ba++)  # 遍历mb
ha=V.value[ba],  # 获取V.value中索引为ba的值
V.array[W]=ha.r,  # 将ha.r赋给V.array中索引为W的值
V.array[W+1]=ha.g,  # 将ha.g赋给V.array中索引为W+1的值
V.array[W+2]=ha.b,  # 将ha.b赋给V.array中索引为W+2的值
W+=3;  # W加3
else for(ba=0;ba<mb;ba++)  # 否则遍历mb
ha=V.value[ba],  # 获取V.value中索引为ba的值
V.array[W]=ha.x,  # 将ha.x赋给V.array中索引为W的值
V.array[W+1]=ha.y,  # 将ha.y赋给V.array中索引为W+1的值
V.array[W+2]=ha.z,  # 将ha.z赋给V.array中索引为W+2的值
W+=3;  # W加3
else if(4===V.size)  # 如果V.size等于4
for(ba=0;ba<mb;ba++)  # 遍历mb
ha=V.value[ba],  # 获取V.value中索引为ba的值
V.array[W]=ha.x,  # 将ha.x赋给V.array中索引为W的值
V.array[W+1]=ha.y,  # 将ha.y赋给V.array中索引为W+1的值
V.array[W+2]=ha.z,  # 将ha.z赋给V.array中索引为W+2的值
V.array[W+3]=ha.w,  # 将ha.w赋给V.array中索引为W+3的值
W+=4  # W加4
# 如果需要更新顶点或者粒子排序，则绑定并填充顶点缓冲区
if(Xb||e.sortParticles)
    l.bindBuffer(l.ARRAY_BUFFER,r.__webglVertexBuffer),
    l.bufferData(l.ARRAY_BUFFER,ac,Mb);
# 如果需要更新颜色或者粒子排序，则绑定并填充颜色缓冲区
if(Yb||e.sortParticles)
    l.bindBuffer(l.ARRAY_BUFFER,r.__webglColorBuffer),
    l.bufferData(l.ARRAY_BUFFER,bc,Mb);
# 如果存在Hb，则遍历Hb数组，更新缓冲区数据
if(Hb)
    for(lb=0,ic=Hb.length;lb<ic;lb++)
        if(V=Hb[lb],V.needsUpdate||e.sortParticles)
            l.bindBuffer(l.ARRAY_BUFFER,V.buffer),
            l.bufferData(l.ARRAY_BUFFER,V.array,Mb)
# 设置顶点和颜色更新标志为false
r.verticesNeedUpdate=!1;
r.colorsNeedUpdate=!1;
# 如果存在G.attributes，则调用y函数
G.attributes&&y(G)
# 遍历t数组，更新渲染对象的材质和顶点数据
for(var cc=0,nb=t.length;cc<nb;cc++){
    var Bc=t[cc],
    Vb=Bc,
    rc=Vb.object,
    Cc=Vb.buffer,
    Dc=rc.geometry,
    Wb=rc.material;
    # 如果材质为THREE.MeshFaceMaterial，则选择第一个材质，否则选择指定的材质
    Wb instanceof THREE.MeshFaceMaterial?
        (Wb=Wb.materials[Dc instanceof THREE.BufferGeometry?0:Cc.materialIndex],
        Vb.material=Wb,
        # 如果材质为透明，则将对象添加到透明渲染列表中
        Wb.transparent?Ib.push(Vb):Jb.push(Vb)):
        Wb&&(Vb.material=Wb,
        # 如果材质为透明，则将对象添加到透明渲染列表中
        Wb.transparent?Ib.push(Vb):Jb.push(Vb));
    # 设置渲染标志为true
    Bc.render=!0;
    # 如果需要按深度排序，则设置对象的z值
    !0===J.sortObjects&&
        (null!==e.renderDepth?Bc.z=e.renderDepth:
        (Na.setFromMatrixPosition(e.matrixWorld),
        Na.applyProjection(Ac),
        Bc.z=Na.z))
}
# 遍历e.children数组，调用q函数
cc=0;
for(nb=e.children.length;cc<nb;cc++)
    q(a,e.children[cc])
}
# 定义m函数，用于渲染对象
function m(a,b,c,d,e,f){
    for(var g,h=a.length-1;-1!==h;h--){
        g=a[h];
        var k=g.object,
        l=g.buffer;
        # 更新对象的顶点数据
        x(k,b);
        # 如果存在f，则使用f作为材质，否则使用对象的材质
        if(f)
            g=f;
        else{
            g=g.material;
            if(!g)
                continue;
            # 设置材质的混合模式、深度测试和深度写入
            e&&J.setBlending(g.blending,g.blendEquation,g.blendSrc,g.blendDst);
            J.setDepthTest(g.depthTest);
            J.setDepthWrite(g.depthWrite);
            B(g.polygonOffset,g.polygonOffsetFactor,g.polygonOffsetUnits)
        }
        # 设置材质的面
        J.setMaterialFaces(g);
        # 如果缓冲区为THREE.BufferGeometry，则调用J.renderBufferDirect函数，否则调用J.renderBuffer函数
        l instanceof THREE.BufferGeometry?
            J.renderBufferDirect(b,c,d,g,l,k):
            J.renderBuffer(b,c,d,g,l,k)
    }
}
# 定义r函数，用于渲染对象
function r(a,b,c,d,e,f,g){
    for(var h,k=0,l=a.length;k<l;k++){
        h=a[k];
# 获取对象的属性
var m=h.object;
# 如果对象可见
if(m.visible){
    # 如果存在g，则将h赋值为g，否则将h赋值为h[b]
    if(g)h=g;
    else{
        h=h[b];
        # 如果h不存在，则继续下一次循环
        if(!h)continue;
        # 如果f存在，则设置混合模式和深度测试
        f&&J.setBlending(h.blending,h.blendEquation,h.blendSrc,h.blendDst);
        J.setDepthTest(h.depthTest);
        J.setDepthWrite(h.depthWrite);
        B(h.polygonOffset,h.polygonOffsetFactor,h.polygonOffsetUnits)
    }
    # 渲染对象
    J.renderImmediateObject(c,d,e,h,m)
}
# 函数t，设置对象的透明度
function t(a){
    var b=a.object.material;
    b.transparent?(a.transparent=b,a.opaque=null):(a.opaque=b,a.transparent=null)
}
# 函数s，更新对象的材质
function s(a,b,d){
    var e=b.material,f=!1;
    # 如果不存在xb[d.id]或者d.groupsNeedUpdate为true
    if(void 0===xb[d.id]||!0===d.groupsNeedUpdate){
        delete ob[b.id];
        a=xb;
        for(var g=d.id,e=e instanceof THREE.MeshFaceMaterial,h=pa.get("OES_element_index_uint")?4294967296:65535,k,f={},m=d.morphTargets.length,n=d.morphNormals.length,p,r={},q=[],t=0,s=d.faces.length;t<s;t++){
            k=d.faces[t];
            var v=e?k.materialIndex:0;
            v in f||(f[v]={hash:v,counter:0});
            k=f[v].hash+"_"+f[v].counter;
            k in r||(p={id:rc++,faces3:[],materialIndex:v,vertices:0,numMorphTargets:m,numMorphNormals:n},r[k]=p,q.push(p));
            r[k].vertices+3>h&&(f[v].counter+=1,k=f[v].hash+"_"+f[v].counter,k in r||(p={id:rc++,faces3:[],materialIndex:v,vertices:0,numMorphTargets:m,numMorphNormals:n},r[k]=p,q.push(p)));
            r[k].faces3.push(t);
            r[k].vertices+=3
        }
        a[g]=q;
        d.groupsNeedUpdate=!1
    }
    a=xb[d.id];
    g=0;
    for(e=a.length;g<e;g++){
        h=a[g];
        if(void 0===h.__webglVertexBuffer){
            f=h;
            f.__webglVertexBuffer=l.createBuffer();
            f.__webglNormalBuffer=l.createBuffer();
            f.__webglTangentBuffer=l.createBuffer();
            f.__webglColorBuffer=l.createBuffer();
            f.__webglUVBuffer=l.createBuffer();
            f.__webglUV2Buffer=l.createBuffer();
            f.__webglSkinIndicesBuffer=l.createBuffer();
# 创建 webglSkinWeightsBuffer 缓冲区
f.__webglSkinWeightsBuffer=l.createBuffer();
# 创建 webglFaceBuffer 缓冲区
f.__webglFaceBuffer=l.createBuffer();
# 创建 webglLineBuffer 缓冲区
f.__webglLineBuffer=l.createBuffer();
# 初始化变量 m 和 n
n=m=void 0;
# 如果存在 morphTargets，则创建对应的缓冲区
if(f.numMorphTargets)
    for(f.__webglMorphTargetsBuffers=[],m=0,n=f.numMorphTargets;m<n;m++)
        f.__webglMorphTargetsBuffers.push(l.createBuffer());
# 如果存在 morphNormals，则创建对应的缓冲区
if(f.numMorphNormals)
    for(f.__webglMorphNormalsBuffers=[],m=0,n=f.numMorphNormals;m<n;m++)
        f.__webglMorphNormalsBuffers.push(l.createBuffer());
# 增加内存中几何体的数量
J.info.memory.geometries++;
# 调用函数 c，传入参数 h 和 b
c(h,b);
# 设置顶点、morphTargets、元素、UV、法线、切线需要更新
d.verticesNeedUpdate=!0;
d.morphTargetsNeedUpdate=!0;
d.elementsNeedUpdate=!0;
d.uvsNeedUpdate=!0;
d.normalsNeedUpdate=!0;
d.tangentsNeedUpdate=!0;
# 设置颜色需要更新
f=d.colorsNeedUpdate=!0
# 如果 f 为真或者 b.__webglActive 未定义，则调用函数 u，传入参数 ob、h 和 b
else f=!1;
(f||void 0===b.__webglActive)&&u(ob,h,b)
# 设置 b.__webglActive 为真
b.__webglActive=!0
# 定义函数 u，传入参数 a、b 和 c
def u(a,b,c):
    # 获取 c 的 id
    var d=c.id;
    # 将 c 添加到 a[d] 数组中
    a[d]=a[d]||[];
    a[d].push({id:d,buffer:b,object:c,material:null,z:0})
# 定义函数 v，传入参数 a
def v(a):
    # 遍历 a 的 attributes 属性
    for(var b in a.attributes)
        # 如果属性需要更新，则返回真
        if(a.attributes[b].needsUpdate)
            return!0;
    return!1
# 定义函数 y，传入参数 a
def y(a):
    # 遍历 a 的 attributes 属性
    for(var b in a.attributes)
        # 将属性的 needsUpdate 设置为假
        a.attributes[b].needsUpdate=!1
# 定义函数 G，传入参数 a、b、c、d、e
def G(a,b,c,d,e):
    # 初始化变量 f、g、h、k
    var f,g,h,k;
    # 将 dc 设置为 0
    dc=0;
    # 如果 d 需要更新
    if(d.needsUpdate)
        # 如果 d 有 program 属性，则调用函数 Cc，传入参数 d
        d.program&&Cc(d);
        # 给 d 添加事件监听器 "dispose"，调用函数 Dc
        d.addEventListener("dispose",Dc);
        # 根据 d 的类型设置 m
        var m;
        d instanceof THREE.MeshDepthMaterial?m="depth":d instanceof THREE.MeshNormalMaterial?m="normal":d instanceof THREE.MeshBasicMaterial?m="basic":d instanceof THREE.MeshLambertMaterial?m="lambert":d instanceof THREE.MeshPhongMaterial?m="phong":d instanceof THREE.LineBasicMaterial?m="basic":d instanceof THREE.LineDashedMaterial?m="dashed":d instanceof THREE.PointCloudMaterial&&(m="particle_basic");
        # 如果 m 存在
        if(m)
            # 获取对应类型的 ShaderLib
            var n=THREE.ShaderLib[m];
            # 设置 d 的 __webglShader 属性
            d.__webglShader={uniforms:THREE.UniformsUtils.clone(n.uniforms),
# 定义顶点着色器和片段着色器
vertexShader: n.vertexShader, fragmentShader: n.fragmentShader
# 如果存在着色器，则创建__webglShader对象
if d.vertexShader and d.fragmentShader:
    d.__webglShader = {uniforms: d.uniforms, vertexShader: d.vertexShader, fragmentShader: d.fragmentShader}
# 否则，将uniforms、vertexShader和fragmentShader赋值给__webglShader对象
else:
    d.__webglShader = {uniforms: d.uniforms, vertexShader: d.vertexShader, fragmentShader: d.fragmentShader}
# 初始化不同类型光源的数量
p, r, q, t = 0, 0, 0, 0
# 遍历光源数组，统计不同类型光源的数量
for s in range(len(b)):
    v = b[s]
    if not v.onlyShadow and v.visible:
        if isinstance(v, THREE.DirectionalLight):
            p += 1
        elif isinstance(v, THREE.PointLight):
            r += 1
        elif isinstance(v, THREE.SpotLight):
            q += 1
        elif isinstance(v, THREE.HemisphereLight):
            t += 1
# 将不同类型光源的数量赋值给对应的变量
f, g, h, k = p, r, q, t
# 初始化投影光源的数量
G = 0
# 遍历光源数组，统计投影光源的数量
for x in range(len(b)):
    A = b[x]
    if A.castShadow:
        if isinstance(A, THREE.SpotLight):
            G += 1
        elif isinstance(A, THREE.DirectionalLight) and not A.shadowCascade:
            G += 1
# 将投影光源的数量赋值给变量y
y = G
# 计算最大骨骼数量
if jc and e and e.skeleton and e.skeleton.useVertexTexture:
    C = 1024
else:
    H = l.getParameter(l.MAX_VERTEX_UNIFORM_VECTORS)
    S = math.floor((H - 20) / 4)
    if e and isinstance(e, THREE.SkinnedMesh):
        S = min(e.skeleton.bones.length, S)
        if S < e.skeleton.bones.length:
            console.warn("WebGLRenderer: too many bones - " + e.skeleton.bones.length + ", this GPU supports just " + S + " (try OpenGL instead of ANGLE)")
    C = S
# 初始化渲染器参数对象
P = {
    precision: X,
    supportsVertexTextures: sc,
    map: !!d.map,
    envMap: !!d.envMap,
    lightMap: !!d.lightMap,
    bumpMap: !!d.bumpMap,
    normalMap: !!d.normalMap,
    specularMap: !!d.specularMap,
    alphaMap: !!d.alphaMap,
    vertexColors: d.vertexColors,
    fog: c,
    useFog: d.fog,
    fogExp: c instanceof THREE.FogExp2,
    sizeAttenuation: d.sizeAttenuation,
    logarithmicDepthBuffer: Fa,
    skinning: d.skinning,
    maxBones: C,
    useVertexTexture: jc and e and e.skeleton and e.skeleton.useVertexTexture,
    morphTargets: d.morphTargets,
    morphNormals: d.morphNormals,
    maxMorphTargets: J.maxMorphTargets,
    maxMorphNormals: J.maxMorphNormals
}
# 定义一系列变量，包括最大方向光数量、最大点光源数量、最大聚光灯数量、最大半球光数量、最大阴影数量等
maxDirLights:f,
maxPointLights:g,
maxSpotLights:h,
maxHemiLights:k,
maxShadows:y,
# 根据阴影映射是否启用、是否接收阴影、阴影映射类型、阴影映射调试、alpha 测试、金属属性、环绕属性、双面属性等设置属性
shadowMapEnabled:J.shadowMapEnabled&&e.receiveShadow&&0<y,
shadowMapType:J.shadowMapType,
shadowMapDebug:J.shadowMapDebug,
shadowMapCascade:J.shadowMapCascade,
alphaTest:d.alphaTest,
metal:d.metal,
wrapAround:d.wrapAround,
doubleSided:d.side===THREE.DoubleSide,
flipSided:d.side===THREE.BackSide
# 创建一个空数组 T
T=[];
# 如果存在顶点着色器和片段着色器，则将它们添加到数组 T 中
m?T.push(m):(T.push(d.fragmentShader),T.push(d.vertexShader));
# 如果定义了属性，则将属性添加到数组 T 中
if(void 0!==d.defines)for(var bb in d.defines)T.push(bb),T.push(d.defines[bb]);
# 遍历属性 P，将属性添加到数组 T 中
for(bb in P)T.push(bb),T.push(P[bb]);
# 将数组 T 中的内容连接成一个字符串 M
for(var M=T.join(),Y,jb=0,ca=hb.length;jb<ca;jb++){
    var cb=hb[jb];
    # 如果已经存在相同的代码，则将 Y 设置为已存在的 WebGLProgram 对象
    if(cb.code===M){
        Y=cb;
        Y.usedTimes++;
        break
    }
}
# 如果不存在相同的代码，则创建一个新的 WebGLProgram 对象，并将其添加到数组 hb 中
void 0===Y&&(Y=new THREE.WebGLProgram(J,M,d,P),hb.push(Y),J.info.memory.programs=hb.length);
# 将当前的 WebGLProgram 对象赋值给属性 d.program
d.program=Y;
# 如果启用了变形目标，则设置支持的变形目标数量
if(d.morphTargets){
    d.numSupportedMorphTargets=0;
    for(var ma,pa="morphTarget",la=0;la<J.maxMorphTargets;la++)ma=pa+la,0<=ob[ma]&&d.numSupportedMorphTargets++
}
# 如果启用了变形法线，则设置支持的变形法线数量
if(d.morphNormals)
    for(d.numSupportedMorphNormals=0,pa="morphNormal",la=0;la<J.maxMorphNormals;la++)ma=pa+la,0<=ob[ma]&&d.numSupportedMorphNormals++;
# 创建一个空数组 d.uniformsList，用于存储 uniform 变量
d.uniformsList=[];
# 遍历属性 d.__webglShader.uniforms，将 uniform 变量添加到 d.uniformsList 中
for(var Jb in d.__webglShader.uniforms){
    var za=d.program.uniforms[Jb];
    za&&d.uniformsList.push([d.__webglShader.uniforms[Jb],za])
}
# 将属性 d.needsUpdate 设置为 false
d.needsUpdate=!1
# 如果启用了变形目标且不存在 e.__webglMorphTargetInfluences，则创建一个 Float32Array 数组
d.morphTargets&&!e.__webglMorphTargetInfluences&&(e.__webglMorphTargetInfluences=new Float32Array(J.maxMorphTargets));
# 定义三个布尔类型的变量，用于记录 WebGLProgram 对象的状态
var aa=!1,
$=!1,
Z=!1,
# 将当前的 WebGLProgram 对象赋值给变量 yb
yb=d.program,
# 获取当前 WebGLProgram 对象的 uniform 变量
qa=yb.uniforms,
# 获取当前 WebGLProgram 对象的 uniform 变量
L=d.__webglShader.uniforms;
# 如果当前 WebGLProgram 对象的 id 与 tc 不相等，则将当前 WebGLProgram 对象设置为活动的程序
yb.id!==tc&&(l.useProgram(yb.program),tc=yb.id,Z=$=aa=!0);
# 如果当前 WebGLProgram 对象的 id 与 Kb 不相等，则将 Z 设置为 true
d.id!==Kb&&(-1===Kb&&(Z=!0),Kb=d.id,$=!0);
# 如果条件 aa 或者 a 不等于 ec 成立，则执行以下代码块
if(aa||a!==ec)
    # 将相机的投影矩阵传递给着色器程序
    l.uniformMatrix4fv(qa.projectionMatrix,!1,a.projectionMatrix.elements)
    # 如果存在 Fa 并且着色器程序中有 logDepthBufFC 变量，则传递对数深度缓冲系数给着色器程序
    Fa&&l.uniform1f(qa.logDepthBufFC,2/(Math.log(a.far+1)/Math.LN2))
    # 如果 a 不等于 ec，则将 a 赋值给 ec
    a!==ec&&(ec=a)
    # 如果材质类型是 ShaderMaterial、MeshPhongMaterial 或者 envMap 不为空，并且相机位置不为空，则将相机位置传递给着色器程序
    (d instanceof THREE.ShaderMaterial||d instanceof THREE.MeshPhongMaterial||d.envMap)&&null!==qa.cameraPosition&&(Na.setFromMatrixPosition(a.matrixWorld),l.uniform3f(qa.cameraPosition,Na.x,Na.y,Na.z))
    # 如果材质类型是 MeshPhongMaterial、MeshLambertMaterial、ShaderMaterial 或者有皮肤，并且视图矩阵不为空，则将视图矩阵传递给着色器程序
    (d instanceof THREE.MeshPhongMaterial||d instanceof THREE.MeshLambertMaterial||d instanceof THREE.ShaderMaterial||d.skinning)&&null!==qa.viewMatrix&&l.uniformMatrix4fv(qa.viewMatrix,!1,a.matrixWorldInverse.elements)
    # 如果有皮肤，则执行以下代码块
    if(d.skinning)
        # 如果存在绑定矩阵并且绑定矩阵不为空，则将绑定矩阵传递给着色器程序
        e.bindMatrix&&null!==qa.bindMatrix&&l.uniformMatrix4fv(qa.bindMatrix,!1,e.bindMatrix.elements)
        # 如果存在绑定矩阵的逆矩阵并且绑定矩阵的逆矩阵不为空，则将绑定矩阵的逆矩阵传递给着色器程序
        e.bindMatrixInverse&&null!==qa.bindMatrixInverse&&l.uniformMatrix4fv(qa.bindMatrixInverse,!1,e.bindMatrixInverse.elements)
        # 如果 jc 存在并且骨骼存在并且使用顶点纹理，则执行以下代码块
        if(jc&&e.skeleton&&e.skeleton.useVertexTexture)
            # 如果骨骼纹理不为空，则创建一个新的纹理单元，并将骨骼纹理传递给着色器程序
            var Ib=K();l.uniform1i(qa.boneTexture,Ib);J.setTexture(e.skeleton.boneTexture,Ib)
            # 如果骨骼纹理宽度不为空，则将骨骼纹理宽度传递给着色器程序
            null!==qa.boneTextureWidth&&l.uniform1i(qa.boneTextureWidth,e.skeleton.boneTextureWidth)
            # 如果骨骼纹理高度不为空，则将骨骼纹理高度传递给着色器程序
            null!==qa.boneTextureHeight&&l.uniform1i(qa.boneTextureHeight,e.skeleton.boneTextureHeight)
        # 如果骨骼存在并且骨骼矩阵不为空，则将骨骼矩阵传递给着色器程序
        else e.skeleton&&e.skeleton.boneMatrices&&null!==qa.boneGlobalMatrices&&l.uniformMatrix4fv(qa.boneGlobalMatrices,!1,e.skeleton.boneMatrices)
    # 如果存在雾效果，则执行以下代码块
    if($)
        # 如果颜色不为空并且材质类型是 MeshPhongMaterial 或者 MeshLambertMaterial，则将雾的颜色传递给着色器程序
        c&&d.fog&&(L.fogColor.value=c.color,c instanceof THREE.Fog?(L.fogNear.value=c.near,L.fogFar.value=c.far):c instanceof THREE.FogExp2&&(L.fogDensity.value=c.density))
        # 如果材质类型是 MeshPhongMaterial 或者 MeshLambertMaterial，则执行以下代码块
        if(d instanceof THREE.MeshPhongMaterial||d instanceof THREE.MeshLambertMaterial||
# 遍历场景中的灯光对象
for (var na = 0; na < b.length; na++) {
    var ia = b[na];
    // 如果灯光对象可见且不仅用于阴影
    if (ia.visible && !ia.onlyShadow) {
        var Ba = ia.color;
        var Ha = ia.intensity;
        var Aa = ia.distance;
        // 如果是环境光
        if (ia instanceof THREE.AmbientLight) {
            // 计算环境光的颜色平方和
            if (J.gammaInput) {
                ya += Ba.r * Ba.r;
                Ga += Ba.g * Ba.g;
                Oa += Ba.b * Ba.b;
            } else {
                ya += Ba.r;
                Ga += Ba.g;
                Oa += Ba.b;
            }
        }
        // 如果是平行光
        else if (ia instanceof THREE.DirectionalLight) {
            ja += 1;
            if (ia.visible) {
                // 设置平行光的方向
                sa.setFromMatrixPosition(ia.matrixWorld);
                Na.setFromMatrixPosition(ia.target.matrixWorld);
                sa.sub(Na);
                sa.normalize();
                Qa = 3 * Sa;
                ib[Qa] = sa.x;
                ib[Qa + 1] = sa.y;
                ib[Qa + 2] = sa.z;
                // 根据是否使用 gamma 输入计算平行光的颜色和强度
                if (J.gammaInput) {
                    D(Cb, Qa, Ba, Ha * Ha);
                } else {
                    E(Cb, Qa, Ba, Ha);
                }
                Sa += 1;
            }
        }
        // 如果是点光源
        else if (ia instanceof THREE.PointLight) {
            ta += 1;
            if (ia.visible) {
                sb = 3 * Ca;
                // 根据是否使用 gamma 输入计算点光源的颜色和强度
                if (J.gammaInput) {
                    D(Qb, sb, Ba, Ha * Ha);
                } else {
                    E(Qb, sb, Ba, Ha);
                }
                Na.setFromMatrixPosition(ia.matrixWorld);
                Ma[sb] = Na.x;
                Ma[sb + 1] = Na.y;
                Ma[sb + 2] = Na.z;
                xb[Ca] = Aa;
                Ca += 1;
            }
        }
        // 如果是聚光灯
        else if (ia instanceof THREE.SpotLight) {
            I += 1;
            if (ia.visible) {
                fb = 3 * Pa;
                // 根据是否使用 gamma 输入计算聚光灯的颜色和强度
                if (J.gammaInput) {
                    D(Ya, fb, Ba, Ha * Ha);
                } else {
                    E(Ya, fb, Ba, Ha);
                }
                sa.setFromMatrixPosition(ia.matrixWorld);
                Za[fb] = sa.x;
                Za[fb + 1] = sa.y;
                Za[fb + 2] = sa.z;
                Mb[Pa] = Aa;
                Na.setFromMatrixPosition(ia.target.matrixWorld);
                sa.sub(Na);
                sa.normalize();
                Rb[fb] = sa.x;
                Rb[fb + 1] = sa.y;
                Rb[fb + 2] = sa.z;
                db[Pa] = Math.cos(ia.angle);
                eb[Pa] = ia.exponent;
                Pa += 1;
            }
        }
    }
}
# 如果存在 HemisphereLight 对象
THREE.HemisphereLight&&(Ia+=1,ia.visible&&(sa.setFromMatrixPosition(ia.matrixWorld),sa.normalize(),Ta=3*Ka,Db[Ta]=sa.x,Db[Ta+1]=sa.y,Db[Ta+2]=sa.z,zb=ia.color,Ab=ia.groundColor,J.gammaInput?(Bb=Ha*Ha,D(qb,Ta,zb,Bb),D(rb,Ta,Ab,Bb)):(E(qb,Ta,zb,Ha),E(rb,Ta,Ab,Ha)),Ka+=1)));
# 计算变量 na 的值
na=3*Sa;
# 循环，将 Cb 数组的长度和 3*ja 中的最大值赋给 Ra
for(Ra=Math.max(Cb.length,3*ja);na<Ra;na++)Cb[na]=0;
# 计算变量 na 的值
na=3*Ca;
# 循环，将 Qb 数组的长度和 3*ta 中的最大值赋给 Ra
for(Ra=Math.max(Qb.length,3*ta);na<Ra;na++)Qb[na]=0;
# 计算变量 na 的值
na=3*Pa;
# 循环，将 Ya 数组的长度和 3*I 中的最大值赋给 Ra
for(Ra=Math.max(Ya.length,3*I);na<Ra;na++)Ya[na]=0;
# 计算变量 na 的值
na=3*Ka;
# 循环，将 qb 数组的长度和 3*Ia 中的最大值赋给 Ra
for(Ra=Math.max(qb.length,3*Ia);na<Ra;na++)qb[na]=0;
# 计算变量 na 的值
na=3*Ka;
# 循环，将 rb 数组的长度和 3*Ia 中的最大值赋给 Ra
for(Ra=Math.max(rb.length,3*Ia);na<Ra;na++)rb[na]=0;
# 设置 directional、point、spot、hemi、ambient 等属性的长度
va.directional.length=Sa;
va.point.length=Ca;
va.spot.length=Pa;
va.hemi.length=Ka;
va.ambient[0]=ya;
va.ambient[1]=Ga;
va.ambient[2]=Oa;
fc=!1
# 如果存在 Z 对象
if(Z){
    var ra=Mc;
    # 设置 ambientLightColor、directionalLightColor、pointLightColor、spotLightColor、hemisphereLightSkyColor、hemisphereLightGroundColor、hemisphereLightDirection 等属性的值
    L.ambientLightColor.value=ra.ambient;
    L.directionalLightColor.value=ra.directional.colors;
    L.directionalLightDirection.value=ra.directional.positions;
    L.pointLightColor.value=ra.point.colors;
    L.pointLightPosition.value=ra.point.positions;
    L.pointLightDistance.value=ra.point.distances;
    L.spotLightColor.value=ra.spot.colors;
    L.spotLightPosition.value=ra.spot.positions;
    L.spotLightDistance.value=ra.spot.distances;
    L.spotLightDirection.value=ra.spot.directions;
    L.spotLightAngleCos.value=ra.spot.anglesCos;
    L.spotLightExponent.value=ra.spot.exponents;
    L.hemisphereLightSkyColor.value=ra.hemi.skyColors;
    L.hemisphereLightGroundColor.value=ra.hemi.groundColors;
    L.hemisphereLightDirection.value=ra.hemi.positions;
    # 调用函数 w，传入参数 true
    w(L,!0)
} else {
    # 调用函数 w，传入参数 false
    w(L,!1)
}
# 如果 d 是 THREE.MeshBasicMaterial、THREE.MeshLambertMaterial 或 THREE.MeshPhongMaterial 的实例
if(d instanceof THREE.MeshBasicMaterial||d instanceof THREE.MeshLambertMaterial||d instanceof
# 根据给定的材质属性设置 MeshPhongMaterial 对象的属性值
(L.opacity.value=d.opacity;J.gammaInput?L.diffuse.value.copyGammaToLinear(d.color):L.diffuse.value=d.color;L.map.value=d.map;L.lightMap.value=d.lightMap;L.specularMap.value=d.specularMap;L.alphaMap.value=d.alphaMap;d.bumpMap&&(L.bumpMap.value=d.bumpMap,L.bumpScale.value=d.bumpScale);d.normalMap&&(L.normalMap.value=d.normalMap,L.normalScale.value.copy(d.normalScale));var La;d.map?La=d.map:d.specularMap?La=d.specularMap:d.normalMap?La=d.normalMap:d.bumpMap?La=d.bumpMap:d.alphaMap&&
(La=d.alphaMap);if(void 0!==La){var Ua=La.offset,Va=La.repeat;L.offsetRepeat.value.set(Ua.x,Ua.y,Va.x,Va.y)}L.envMap.value=d.envMap;L.flipEnvMap.value=d.envMap instanceof THREE.WebGLRenderTargetCube?1:-1;L.reflectivity.value=d.reflectivity;L.refractionRatio.value=d.refractionRatio;L.combine.value=d.combine;L.useRefract.value=d.envMap&&d.envMap.mapping instanceof THREE.CubeRefractionMapping}d instanceof THREE.LineBasicMaterial?(L.diffuse.value=d.color,L.opacity.value=d.opacity):d instanceof THREE.LineDashedMaterial?
(L.diffuse.value=d.color,L.opacity.value=d.opacity,L.dashSize.value=d.dashSize,L.totalSize.value=d.dashSize+d.gapSize,L.scale.value=d.scale):d instanceof THREE.PointCloudMaterial?(L.psColor.value=d.color,L.opacity.value=d.opacity,L.size.value=d.size,L.scale.value=O.height/2,L.map.value=d.map):d instanceof THREE.MeshPhongMaterial?(L.shininess.value=d.shininess,J.gammaInput?(L.ambient.value.copyGammaToLinear(d.ambient),L.emissive.value.copyGammaToLinear(d.emissive),L.specular.value.copyGammaToLinear(d.specular)):
# 如果材质是 MeshPhongMaterial 类型
if (d instanceof THREE.MeshPhongMaterial) {
    # 如果 J.gammaInput 为真，则将环境光和发射光值转换为 gamma 校正的值，否则直接赋值
    (J.gammaInput ? (L.ambient.value.copyGammaToLinear(d.ambient), L.emissive.value.copyGammaToLinear(d.emissive)) : (L.ambient.value = d.ambient, L.emissive.value = d.emissive),
    # 如果材质有 wrapAround 属性，则将 L.wrapRGB 的值复制为 d.wrapRGB 的值
    d.wrapAround && L.wrapRGB.value.copy(d.wrapRGB));
}
# 如果材质是 MeshLambertMaterial 类型
else if (d instanceof THREE.MeshLambertMaterial) {
    # 如果 J.gammaInput 为真，则将环境光和发射光值转换为 gamma 校正的值，否则直接赋值
    (J.gammaInput ? (L.ambient.value.copyGammaToLinear(d.ambient), L.emissive.value.copyGammaToLinear(d.emissive)) : (L.ambient.value = d.ambient, L.emissive.value = d.emissive),
    # 如果材质有 wrapAround 属性，则将 L.wrapRGB 的值复制为 d.wrapRGB 的值
    d.wrapAround && L.wrapRGB.value.copy(d.wrapRGB));
}
# 如果材质是 MeshDepthMaterial 类型
else if (d instanceof THREE.MeshDepthMaterial) {
    # 设置 L.mNear 和 L.mFar 的值为 a.near 和 a.far 的值，设置 L.opacity 的值为 d.opacity 的值
    (L.mNear.value = a.near, L.mFar.value = a.far, L.opacity.value = d.opacity);
}
# 如果材质是 MeshNormalMaterial 类型
else if (d instanceof THREE.MeshNormalMaterial) {
    # 设置 L.opacity 的值为 d.opacity 的值
    (L.opacity.value = d.opacity);
}

# 如果 e.receiveShadow 为真且材质没有 _shadowPass 属性且 L.shadowMatrix 存在
if (e.receiveShadow && !d._shadowPass && L.shadowMatrix) {
    # 遍历光源数组 b
    for (var Eb = 0, pb = 0, Nb = b.length; pb < Nb; pb++) {
        var z = b[pb];
        # 如果 z.castShadow 为真且 z 是 SpotLight 或者 DirectionalLight 且不是 shadowCascade
        if (z.castShadow && (z instanceof THREE.SpotLight || (z instanceof THREE.DirectionalLight && !z.shadowCascade))) {
            # 设置 L.shadowMap、L.shadowMapSize、L.shadowMatrix、L.shadowDarkness、L.shadowBias 的值为对应光源的值
            (L.shadowMap.value[Eb] = z.shadowMap, L.shadowMapSize.value[Eb] = z.shadowMapSize, L.shadowMatrix.value[Eb] = z.shadowMatrix, L.shadowDarkness.value[Eb] = z.shadowDarkness, L.shadowBias.value[Eb] = z.shadowBias, Eb++);
        }
    }
}

# 遍历材质的 uniformsList
for (var nb = 0, Pb = Sb.length; nb < Pb; nb++) {
    var da = Sb[nb][0];
    # 如果 da.needsUpdate 不为 false
    if (!false !== da.needsUpdate) {
        var wb = da.type, U = da.value, fa = Sb[nb][1];
        # 根据不同的类型设置对应的 uniform 值
        switch (wb) {
            case "1i":
                l.uniform1i(fa, U);
                break;
            case "1f":
                l.uniform1f(fa, U);
                break;
            case "2f":
                l.uniform2f(fa, U[0], U[1]);
                break;
            case "3f":
                l.uniform3f(fa, U[0], U[1], U[2]);
                break;
            case "4f":
                l.uniform4f(fa, U[0], U[1], U[2], U[3]);
                break;
            case "1iv":
                l.uniform1iv(fa, U);
                break;
            case "3iv":
                l.uniform3iv(fa, U);
                break;
            case "1fv":
                l.uniform1fv(fa, U);
                break;
            case "2fv":
                l.uniform2fv(fa, U);
                break;
            case "3fv":
                l.uniform3fv(fa, U);
                break;
            case "4fv":
                l.uniform4fv(fa, U);
                break;
# 根据不同的类型进行不同的 uniform 赋值操作
# 如果类型是 Matrix3fv，则使用 uniformMatrix3fv 方法进行赋值
# 如果类型是 Matrix4fv，则使用 uniformMatrix4fv 方法进行赋值
# 如果类型是 i，则使用 uniform1i 方法进行赋值
# 如果类型是 f，则使用 uniform1f 方法进行赋值
# 如果类型是 v2，则使用 uniform2f 方法进行赋值
# 如果类型是 v3，则使用 uniform3f 方法进行赋值
# 如果类型是 v4，则使用 uniform4f 方法进行赋值
# 如果类型是 c，则使用 uniform3f 方法进行赋值
# 如果类型是 iv1，则使用 uniform1iv 方法进行赋值
# 如果类型是 iv，则使用 uniform3iv 方法进行赋值
# 如果类型是 fv1，则使用 uniform1fv 方法进行赋值
# 如果类型是 fv，则使用 uniform3fv 方法进行赋值
# 如果类型是 v2v，则使用 uniform2fv 方法进行赋值
# 如果类型是 v3v，则使用 uniform3fv 方法进行赋值
# 如果类型是 v4v，则使用 uniform4fv 方法进行赋值
# 如果类型是 m3，则使用 uniformMatrix3fv 方法进行赋值
# 如果类型是 m3v，则使用 uniformMatrix3fv 方法进行赋值
# 如果类型是 m4，则使用 uniformMatrix4fv 方法进行赋值
# 如果类型是 m4v，则使用 uniformMatrix4fv 方法进行赋值
# 设置 uniform 变量为 4x4 矩阵
l.uniformMatrix4fv(fa,!1,da._array);
# 跳出 switch 语句
break;
# 如果是 "t" 类型的 uniform 变量
case "t":
    # 将 Ja 设置为 U
    Ja=U;
    # 将 wa 设置为 K()
    wa=K();
    # 设置 uniform 变量为整数值
    l.uniform1i(fa,wa);
    # 如果 Ja 不存在，则继续下一次循环
    if(!Ja)continue;
    # 如果 Ja 是 CubeTexture 类型或者 Ja 的 image 是数组且长度为 6
    if(Ja instanceof THREE.CubeTexture||Ja.image instanceof Array&&6===Ja.image.length):
        # 将 ua 设置为 Ja
        var ua=Ja,
        # 将 Lb 设置为 wa
        Lb=wa;
        # 如果 ua 的 image 长度为 6
        if(6===ua.image.length):
            # 如果 ua 需要更新
            if(ua.needsUpdate):
                # 如果 ua 的 image 没有创建 WebGL 纹理
                if(!ua.image.__webglTextureCube):
                    # 添加事件监听器
                    ua.addEventListener("dispose",gc),
                    # 创建 WebGL 纹理
                    ua.image.__webglTextureCube=l.createTexture(),
                    # 增加内存中纹理的数量
                    J.info.memory.textures++;
                # 设置活动纹理单元
                l.activeTexture(l.TEXTURE0+Lb);
                # 绑定纹理对象
                l.bindTexture(l.TEXTURE_CUBE_MAP,ua.image.__webglTextureCube);
                # 设置像素存储模式
                l.pixelStorei(l.UNPACK_FLIP_Y_WEBGL,ua.flipY);
                # 创建数组用于存储纹理数据
                for(var Ob=ua instanceof THREE.CompressedTexture,Tb=ua.image[0]instanceof THREE.DataTexture,kb=[],Da=0;6>Da;Da++)kb[Da]=!J.autoScaleCubemaps||Ob||Tb?Tb?ua.image[Da].image:ua.image[Da]:R(ua.image[Da],$c);
                # 获取第一个纹理
                var ka=kb[0],
                # 判断纹理的宽高是否为 2 的幂
                Zb=THREE.Math.isPowerOfTwo(ka.width)&&THREE.Math.isPowerOfTwo(ka.height),
                # 获取 ua 的格式
                ab=Q(ua.format),
                # 获取 ua 的类型
                Fb=Q(ua.type);
                # 设置纹理参数
                F(l.TEXTURE_CUBE_MAP,ua,Zb);
                # 遍历 6 个面的纹理
                for(Da=0;6>Da;Da++):
                    # 如果是压缩纹理
                    if(Ob):
                        # 获取纹理的 mipmaps
                        for(var gb,$b=kb[Da].mipmaps,ga=0,Xb=$b.length;ga<Xb;ga++)gb=$b[ga],
                        # 判断纹理格式
                        ua.format!==THREE.RGBAFormat&&ua.format!==THREE.RGBFormat?
                        # 如果是支持的压缩纹理格式
                        -1<Nc().indexOf(ab)?l.compressedTexImage2D(l.TEXTURE_CUBE_MAP_POSITIVE_X+Da,ga,ab,gb.width,gb.height,0,gb.data):
                        # 否则输出警告信息
                        console.warn("Attempt to load unsupported compressed texture format"):
                        # 设置纹理图像数据
                        l.texImage2D(l.TEXTURE_CUBE_MAP_POSITIVE_X+Da,ga,ab,gb.width,gb.height,0,ab,Fb,gb.data);
                    # 如果是 DataTexture
                    else Tb?
                        # 设置纹理图像数据
                        l.texImage2D(l.TEXTURE_CUBE_MAP_POSITIVE_X+Da,0,ab,kb[Da].width,kb[Da].height,0,ab,Fb,kb[Da].data):
                        # 设置纹理图像数据
                        l.texImage2D(l.TEXTURE_CUBE_MAP_POSITIVE_X+Da,0,ab,ab,Fb,kb[Da]);
                # 生成纹理的 mipmaps
                ua.generateMipmaps&&Zb&&l.generateMipmap(l.TEXTURE_CUBE_MAP);
# 检查是否需要更新 uniform 变量
ua.needsUpdate=!1;
# 如果需要更新，并且存在 onUpdate 回调函数，则执行 onUpdate 回调函数
if(ua.onUpdate)ua.onUpdate()
# 如果是立方体贴图，则绑定立方体贴图
else l.activeTexture(l.TEXTURE0+Lb),l.bindTexture(l.TEXTURE_CUBE_MAP,ua.image.__webglTextureCube)
# 如果是渲染目标立方体贴图，则绑定渲染目标立方体贴图
else if(Ja instanceof THREE.WebGLRenderTargetCube){var Yb=Ja;l.activeTexture(l.TEXTURE0+wa);l.bindTexture(l.TEXTURE_CUBE_MAP,Yb.__webglTexture)}
# 设置纹理
else J.setTexture(Ja,wa);
# 根据 uniform 类型设置值
case "tv":void 0===da._array&&(da._array=[]);N=0;for(xa=da.value.length;N<xa;N++)da._array[N]=K();l.uniform1iv(fa,da._array);N=0;for(xa=da.value.length;N<xa;N++)Ja=da.value[N],wa=da._array[N],Ja&&J.setTexture(Ja,wa);
# 默认情况下，打印警告信息
default:console.warn("THREE.WebGLRenderer: Unknown uniform type: "+wb)}}
# 设置 uniform 变量
l.uniformMatrix4fv(qa.modelViewMatrix,!1,e._modelViewMatrix.elements);
qa.normalMatrix&&l.uniformMatrix3fv(qa.normalMatrix,!1,e._normalMatrix.elements);
null!==qa.modelMatrix&&l.uniformMatrix4fv(qa.modelMatrix,!1,e.matrixWorld.elements);
return yb}
# 更新光照 uniform 变量
function w(a,b){a.ambientLightColor.needsUpdate=b;a.directionalLightColor.needsUpdate=b;a.directionalLightDirection.needsUpdate=b;a.pointLightColor.needsUpdate=b;a.pointLightPosition.needsUpdate=b;a.pointLightDistance.needsUpdate=b;a.spotLightColor.needsUpdate=b;a.spotLightPosition.needsUpdate=b;a.spotLightDistance.needsUpdate=b;a.spotLightDirection.needsUpdate=b;a.spotLightAngleCos.needsUpdate=b;a.spotLightExponent.needsUpdate=b;a.hemisphereLightSkyColor.needsUpdate=b;a.hemisphereLightGroundColor.needsUpdate=b;a.hemisphereLightDirection.needsUpdate=b}
# 创建纹理
function K(){var a=dc;a>=Oc&&console.warn("WebGLRenderer: trying to use "+a+" texture units while this GPU supports only "+
Oc);dc+=1;return a}function x(a,b){a._modelViewMatrix.multiplyMatrices(b.matrixWorldInverse,a.matrixWorld);a._normalMatrix.getNormalMatrix(a._modelViewMatrix)}function D(a,b,c,d){a[b]=c.r*c.r*d;a[b+1]=c.g*c.g*d;a[b+2]=c.b*c.b*d}function E(a,b,c,d){a[b]=c.r*d;a[b+1]=c.g*d;a[b+2]=c.b*d}function A(a){a!==Pc&&(l.lineWidth(a),Pc=a)}function B(a,b,c){Qc!==a&&(a?l.enable(l.POLYGON_OFFSET_FILL):l.disable(l.POLYGON_OFFSET_FILL),Qc=a);!a||Rc===b&&Sc===c||(l.polygonOffset(b,c),Rc=b,Sc=c)}function F(a,b,c){c?
(l.texParameteri(a,l.TEXTURE_WRAP_S,Q(b.wrapS)),l.texParameteri(a,l.TEXTURE_WRAP_T,Q(b.wrapT)),l.texParameteri(a,l.TEXTURE_MAG_FILTER,Q(b.magFilter)),l.texParameteri(a,l.TEXTURE_MIN_FILTER,Q(b.minFilter))):(l.texParameteri(a,l.TEXTURE_WRAP_S,l.CLAMP_TO_EDGE),l.texParameteri(a,l.TEXTURE_WRAP_T,l.CLAMP_TO_EDGE),l.texParameteri(a,l.TEXTURE_MAG_FILTER,T(b.magFilter)),l.texParameteri(a,l.TEXTURE_MIN_FILTER,T(b.minFilter)));(c=pa.get("EXT_texture_filter_anisotropic"))&&b.type!==THREE.FloatType&&(1<b.anisotropy||
b.__oldAnisotropy)&&(l.texParameterf(a,c.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(b.anisotropy,J.getMaxAnisotropy())),b.__oldAnisotropy=b.anisotropy)}function R(a,b){if(a.width>b||a.height>b){var c=b/Math.max(a.width,a.height),d=document.createElement("canvas");d.width=Math.floor(a.width*c);d.height=Math.floor(a.height*c);d.getContext("2d").drawImage(a,0,0,a.width,a.height,0,0,d.width,d.height);console.log("THREE.WebGLRenderer:",a,"is too big ("+a.width+"x"+a.height+"). Resized to "+d.width+"x"+d.height+


// 函数 x 用于计算模型视图矩阵
function x(a,b){a._modelViewMatrix.multiplyMatrices(b.matrixWorldInverse,a.matrixWorld);a._normalMatrix.getNormalMatrix(a._modelViewMatrix)}
// 函数 D 用于计算颜色值的平方乘以一个常数
function D(a,b,c,d){a[b]=c.r*c.r*d;a[b+1]=c.g*c.g*d;a[b+2]=c.b*c.b*d}
// 函数 E 用于计算颜色值乘以一个常数
function E(a,b,c,d){a[b]=c.r*d;a[b+1]=c.g*d;a[b+2]=c.b*d}
// 函数 A 用于设置线条宽度
function A(a){a!==Pc&&(l.lineWidth(a),Pc=a)}
// 函数 B 用于设置多边形偏移
function B(a,b,c){Qc!==a&&(a?l.enable(l.POLYGON_OFFSET_FILL):l.disable(l.POLYGON_OFFSET_FILL),Qc=a);!a||Rc===b&&Sc===c||(l.polygonOffset(b,c),Rc=b,Sc=c)}
// 函数 F 用于设置纹理参数
function F(a,b,c){c?(l.texParameteri(a,l.TEXTURE_WRAP_S,Q(b.wrapS)),l.texParameteri(a,l.TEXTURE_WRAP_T,Q(b.wrapT)),l.texParameteri(a,l.TEXTURE_MAG_FILTER,Q(b.magFilter)),l.texParameteri(a,l.TEXTURE_MIN_FILTER,Q(b.minFilter))):(l.texParameteri(a,l.TEXTURE_WRAP_S,l.CLAMP_TO_EDGE),l.texParameteri(a,l.TEXTURE_WRAP_T,l.CLAMP_TO_EDGE),l.texParameteri(a,l.TEXTURE_MAG_FILTER,T(b.magFilter)),l.texParameteri(a,l.TEXTURE_MIN_FILTER,T(b.minFilter));(c=pa.get("EXT_texture_filter_anisotropic"))&&b.type!==THREE.FloatType&&(1<b.anisotropy||
b.__oldAnisotropy)&&(l.texParameterf(a,c.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(b.anisotropy,J.getMaxAnisotropy())),b.__oldAnisotropy=b.anisotropy)}
// 函数 R 用于调整图像大小
function R(a,b){if(a.width>b||a.height>b){var c=b/Math.max(a.width,a.height),d=document.createElement("canvas");d.width=Math.floor(a.width*c);d.height=Math.floor(a.height*c);d.getContext("2d").drawImage(a,0,0,a.width,a.height,0,0,d.width,d.height);console.log("THREE.WebGLRenderer:",a,"is too big ("+a.width+"x"+a.height+"). Resized to "+d.width+"x"+d.height+
# 绑定渲染缓冲区，根据参数设置深度和模板缓冲区
function H(a,b){
    l.bindRenderbuffer(l.RENDERBUFFER,a);
    if(b.depthBuffer && !b.stencilBuffer){
        l.renderbufferStorage(l.RENDERBUFFER,l.DEPTH_COMPONENT16,b.width,b.height);
        l.framebufferRenderbuffer(l.FRAMEBUFFER,l.DEPTH_ATTACHMENT,l.RENDERBUFFER,a);
    } else if(b.depthBuffer && b.stencilBuffer){
        l.renderbufferStorage(l.RENDERBUFFER,l.DEPTH_STENCIL,b.width,b.height);
        l.framebufferRenderbuffer(l.FRAMEBUFFER,l.DEPTH_STENCIL_ATTACHMENT,l.RENDERBUFFER,a);
    } else {
        l.renderbufferStorage(l.RENDERBUFFER,l.RGBA4,b.width,b.height);
    }
}

# 绑定纹理对象，根据类型生成不同类型的纹理
function C(a){
    if(a instanceof THREE.WebGLRenderTargetCube){
        l.bindTexture(l.TEXTURE_CUBE_MAP,a.__webglTexture);
        l.generateMipmap(l.TEXTURE_CUBE_MAP);
        l.bindTexture(l.TEXTURE_CUBE_MAP,null);
    } else {
        l.bindTexture(l.TEXTURE_2D,a.__webglTexture);
        l.generateMipmap(l.TEXTURE_2D);
        l.bindTexture(l.TEXTURE_2D,null);
    }
}

# 根据过滤器类型返回对应的 WebGL 常量
function T(a){
    return a===THREE.NearestFilter || a===THREE.NearestMipMapNearestFilter || a===THREE.NearestMipMapLinearFilter ? l.NEAREST : l.LINEAR;
}

# 根据纹理包裹类型返回对应的 WebGL 常量
function Q(a){
    if(a===THREE.RepeatWrapping) return l.REPEAT;
    if(a===THREE.ClampToEdgeWrapping) return l.CLAMP_TO_EDGE;
    if(a===THREE.MirroredRepeatWrapping) return l.MIRRORED_REPEAT;
    if(a===THREE.NearestFilter) return l.NEAREST;
    if(a===THREE.NearestMipMapNearestFilter) return l.NEAREST_MIPMAP_NEAREST;
    if(a===THREE.NearestMipMapLinearFilter) return l.NEAREST_MIPMAP_LINEAR;
    if(a===THREE.LinearFilter) return l.LINEAR;
    if(a===THREE.LinearMipMapNearestFilter) return l.LINEAR_MIPMAP_NEAREST;
    if(a===THREE.LinearMipMapLinearFilter) return l.LINEAR_MIPMAP_LINEAR;
    if(a===THREE.UnsignedByteType) return l.UNSIGNED_BYTE;
}
# 检查输入的数据类型，返回对应的 WebGL 常量
if(a===THREE.UnsignedShort4444Type) return l.UNSIGNED_SHORT_4_4_4_4;
if(a===THREE.UnsignedShort5551Type) return l.UNSIGNED_SHORT_5_5_5_1;
if(a===THREE.UnsignedShort565Type) return l.UNSIGNED_SHORT_5_6_5;
if(a===THREE.ByteType) return l.BYTE;
if(a===THREE.ShortType) return l.SHORT;
if(a===THREE.UnsignedShortType) return l.UNSIGNED_SHORT;
if(a===THREE.IntType) return l.INT;
if(a===THREE.UnsignedIntType) return l.UNSIGNED_INT;
if(a===THREE.FloatType) return l.FLOAT;
if(a===THREE.AlphaFormat) return l.ALPHA;
if(a===THREE.RGBFormat) return l.RGB;
if(a===THREE.RGBAFormat) return l.RGBA;
if(a===THREE.LuminanceFormat) return l.LUMINANCE;
if(a===THREE.LuminanceAlphaFormat) return l.LUMINANCE_ALPHA;
if(a===THREE.AddEquation) return l.FUNC_ADD;
if(a===THREE.SubtractEquation) return l.FUNC_SUBTRACT;
if(a===THREE.ReverseSubtractEquation) return l.FUNC_REVERSE_SUBTRACT;
if(a===THREE.ZeroFactor) return l.ZERO;
if(a===THREE.OneFactor) return l.ONE;
if(a===THREE.SrcColorFactor) return l.SRC_COLOR;
if(a===THREE.OneMinusSrcColorFactor) return l.ONE_MINUS_SRC_COLOR;
if(a===THREE.SrcAlphaFactor) return l.SRC_ALPHA;
if(a===THREE.OneMinusSrcAlphaFactor) return l.ONE_MINUS_SRC_ALPHA;
if(a===THREE.DstAlphaFactor) return l.DST_ALPHA;
if(a===THREE.OneMinusDstAlphaFactor) return l.ONE_MINUS_DST_ALPHA;
if(a===THREE.DstColorFactor) return l.DST_COLOR;
if(a===THREE.OneMinusDstColorFactor) return l.ONE_MINUS_DST_COLOR;
if(a===THREE.SrcAlphaSaturateFactor) return l.SRC_ALPHA_SATURATE;
# 获取 WEBGL_compressed_texture_s3tc 参数
b=pa.get("WEBGL_compressed_texture_s3tc");
# 如果参数不为空，根据输入的格式返回对应的压缩格式常量
if(null!==b){
    if(a===THREE.RGB_S3TC_DXT1_Format) return b.COMPRESSED_RGB_S3TC_DXT1_EXT;
# 检查输入的纹理格式，返回对应的压缩格式
if(a===THREE.RGBA_S3TC_DXT1_Format)return b.COMPRESSED_RGBA_S3TC_DXT1_EXT;
if(a===THREE.RGBA_S3TC_DXT3_Format)return b.COMPRESSED_RGBA_S3TC_DXT3_EXT;
if(a===THREE.RGBA_S3TC_DXT5_Format)return b.COMPRESSED_RGBA_S3TC_DXT5_EXT}
# 获取 WEBGL_compressed_texture_pvrtc 扩展
b=pa.get("WEBGL_compressed_texture_pvrtc");
if(null!==b){
    if(a===THREE.RGB_PVRTC_4BPPV1_Format)return b.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;
    if(a===THREE.RGB_PVRTC_2BPPV1_Format)return b.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;
    if(a===THREE.RGBA_PVRTC_4BPPV1_Format)return b.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;
    if(a===THREE.RGBA_PVRTC_2BPPV1_Format)return b.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG}
# 获取 EXT_blend_minmax 扩展
b=pa.get("EXT_blend_minmax");
if(null!==b){
    if(a===THREE.MinEquation)return b.MIN_EXT;
    if(a===THREE.MaxEquation)return b.MAX_EXT}
# 返回 0，表示未匹配到任何压缩格式
return 0}
# 打印版本信息
console.log("THREE.WebGLRenderer",THREE.REVISION);
# 初始化参数
a=a||{};
var O=void 0!==a.canvas?a.canvas:document.createElement("canvas"),
S=void 0!==a.context?a.context:null,
X=void 0!==a.precision?a.precision:"highp",
Y=void 0!==a.alpha?a.alpha:!1,
la=void 0!==a.depth?a.depth:!0,
ma=void 0!==a.stencil?a.stencil:!0,
ya=void 0!==a.antialias?a.antialias:!1,
P=void 0!==a.premultipliedAlpha?a.premultipliedAlpha:!0,
Ga=void 0!==a.preserveDrawingBuffer?a.preserveDrawingBuffer:!1,
Fa=void 0!==a.logarithmicDepthBuffer?a.logarithmicDepthBuffer:!1,
za=new THREE.Color(0),
bb=0,
cb=[],
ob={},
jb=[],
Jb=[],
Ib=[],
yb=[],
Ra=[];
# 初始化渲染器属性
this.domElement=O;
this.context=null;
this.devicePixelRatio=void 0!==a.devicePixelRatio?a.devicePixelRatio:void 0!==self.devicePixelRatio?self.devicePixelRatio:1;
this.sortObjects=this.autoClearStencil=
# 设置自动清除深度、颜色和整个画布
this.autoClearDepth=this.autoClearColor=this.autoClear=!0;
# 禁用阴影映射、Gamma 输出和输入
this.shadowMapEnabled=this.gammaOutput=this.gammaInput=!1;
# 设置阴影映射类型为 PCFShadowMap，阴影映射剔除面为 CullFaceFront
this.shadowMapType=THREE.PCFShadowMap;
this.shadowMapCullFace=THREE.CullFaceFront;
# 禁用阴影映射级联和调试
this.shadowMapCascade=this.shadowMapDebug=!1;
# 设置最大变形目标和法线变形目标数量
this.maxMorphTargets=8;
this.maxMorphNormals=4;
# 设置自动缩放立方体贴图
this.autoScaleCubemaps=!0;
# 设置信息对象，包括内存和渲染信息
this.info={memory:{programs:0,geometries:0,textures:0},render:{calls:0,vertices:0,faces:0,points:0}};
# 初始化变量
var J=this,hb=[],tc=null,Tc=null,Kb=-1,Oa=-1,ec=null,dc=0,Lb=-1,Mb=-1,pb=-1,Nb=-1,Ob=-1,
Xb=-1,Yb=-1,nb=-1,Qc=null,Rc=null,Sc=null,Pc=null,Pb=0,kc=0,lc=O.width,mc=O.height,Uc=0,Vc=0,wb=new Uint8Array(16),ib=new Uint8Array(16),Ec=new THREE.Frustum,Ac=new THREE.Matrix4,Gc=new THREE.Matrix4,Na=new THREE.Vector3,sa=new THREE.Vector3,fc=!0,
# 初始化光照信息对象
Mc={ambient:[0,0,0],directional:{length:0,colors:[],positions:[]},point:{length:0,colors:[],positions:[],distances:[]},spot:{length:0,colors:[],positions:[],distances:[],directions:[],anglesCos:[],exponents:[]},hemi:{length:0,skyColors:[],groundColors:[],
positions:[]}},l;
# 尝试创建 WebGL 上下文
try{
    # 定义 WebGL 上下文属性
    var Wc={alpha:Y,depth:la,stencil:ma,antialias:ya,premultipliedAlpha:P,preserveDrawingBuffer:Ga};
    # 获取 WebGL 上下文
    l=S||O.getContext("webgl",Wc)||O.getContext("experimental-webgl",Wc);
    # 如果 WebGL 上下文为 null，则抛出错误
    if(null===l){
        if(null!==O.getContext("webgl"))throw"Error creating WebGL context with your selected attributes.";
        throw"Error creating WebGL context.";
    }
}
# 捕获错误并输出到控制台
catch(ad){
    console.error(ad)
}
# 如果 l 上下文对象中不存在 getShaderPrecisionFormat 方法，则定义该方法
void 0===l.getShaderPrecisionFormat&&(l.getShaderPrecisionFormat=function(){return{rangeMin:1,rangeMax:1,precision:1}});
# 创建 WebGL 扩展对象
var pa=new THREE.WebGLExtensions(l);
# 获取 WebGL 上下文的扩展信息
pa.get("OES_texture_float");
pa.get("OES_texture_float_linear");
pa.get("OES_standard_derivatives");
Fa && pa.get("EXT_frag_depth");

# 设置 WebGL 渲染上下文的清除颜色、深度和模板缓冲区
l.clearColor(0, 0, 0, 1);
l.clearDepth(1);
l.clearStencil(0);

# 启用深度测试，并设置深度测试函数
l.enable(l.DEPTH_TEST);
l.depthFunc(l.LEQUAL);

# 设置正面的顺时针方向为正面，并启用背面剔除
l.frontFace(l.CCW);
l.cullFace(l.BACK);
l.enable(l.CULL_FACE);

# 启用混合功能，并设置混合方程和混合函数
l.enable(l.BLEND);
l.blendEquation(l.FUNC_ADD);
l.blendFunc(l.SRC_ALPHA, l.ONE_MINUS_SRC_ALPHA);

# 设置视口
l.viewport(Pb, kc, lc, mc);

# 获取 WebGL 上下文的参数信息
var Oc = l.getParameter(l.MAX_TEXTURE_IMAGE_UNITS),
    bd = l.getParameter(l.MAX_VERTEX_TEXTURE_IMAGE_UNITS),
    cd = l.getParameter(l.MAX_TEXTURE_SIZE),
    $c = l.getParameter(l.MAX_CUBE_MAP_TEXTURE_SIZE),
    sc = 0 < bd,
    jc = sc && pa.get("OES_texture_float"),
    dd = l.getShaderPrecisionFormat(l.VERTEX_SHADER, l.HIGH_FLOAT),
    ed = l.getShaderPrecisionFormat(l.VERTEX_SHADER, l.MEDIUM_FLOAT),
    fd = l.getShaderPrecisionFormat(l.FRAGMENT_SHADER, l.HIGH_FLOAT),
    gd = l.getShaderPrecisionFormat(l.FRAGMENT_SHADER, l.MEDIUM_FLOAT);

# 获取 WebGL 上下文的压缩纹理格式信息
var Nc = function() {
    var a;
    return function() {
        if (void 0 !== a) return a;
        a = [];
        if (pa.get("WEBGL_compressed_texture_pvrtc") || pa.get("WEBGL_compressed_texture_s3tc")) {
            var b = l.getParameter(l.COMPRESSED_TEXTURE_FORMATS);
            for (var c = 0; c < b.length; c++) {
                a.push(b[c]);
            }
        }
        return a;
    }
}();

# 检查浮点精度和设置着色器精度
var hd = 0 < dd.precision && 0 < fd.precision,
    Xc = 0 < ed.precision && 0 < gd.precision;

# 根据浮点精度设置着色器精度
if ("highp" !== X || !hd) {
    if (Xc) {
        X = "mediump";
        console.warn("THREE.WebGLRenderer: highp not supported, using mediump.");
    } else {
        X = "lowp";
        console.warn("THREE.WebGLRenderer: highp and mediump not supported, using lowp.");
    }
}
# 检查是否支持mediump精度，如果不支持则使用lowp精度，并输出警告信息
"mediump"!==X||Xc||(X="lowp",console.warn("THREE.WebGLRenderer: mediump not supported, using lowp."));
# 创建阴影映射插件
var id=new THREE.ShadowMapPlugin(this,cb,ob,jb),
# 创建精灵插件
jd=new THREE.SpritePlugin(this,yb),
# 创建镜头耀斑插件
kd=new THREE.LensFlarePlugin(this,Ra);
# 获取渲染上下文
this.getContext=function(){return l};
# 检查是否支持顶点纹理
this.supportsVertexTextures=function(){return sc};
# 检查是否支持浮点纹理
this.supportsFloatTextures=function(){return pa.get("OES_texture_float")};
# 检查是否支持标准导数
this.supportsStandardDerivatives=function(){return pa.get("OES_standard_derivatives")};
# 检查是否支持S3TC压缩纹理
this.supportsCompressedTextureS3TC=function(){return pa.get("WEBGL_compressed_texture_s3tc")};
# 检查是否支持PVRTC压缩纹理
this.supportsCompressedTexturePVRTC=function(){return pa.get("WEBGL_compressed_texture_pvrtc")};
# 检查是否支持混合最小最大值
this.supportsBlendMinMax=function(){return pa.get("EXT_blend_minmax")};
# 获取最大各向异性
this.getMaxAnisotropy=function(){var a;return function(){if(void 0!==a)return a;var b=pa.get("EXT_texture_filter_anisotropic");return a=null!==b?l.getParameter(b.MAX_TEXTURE_MAX_ANISOTROPY_EXT):0}}();
# 获取精度
this.getPrecision=function(){return X};
# 设置渲染器尺寸
this.setSize=function(a,b,c){O.width=a*this.devicePixelRatio;O.height=b*this.devicePixelRatio;!1!==c&&(O.style.width=a+"px",O.style.height=b+"px");this.setViewport(0,0,a,b)};
# 设置视口
this.setViewport=function(a,b,c,d){Pb=a*this.devicePixelRatio;kc=b*this.devicePixelRatio;lc=c*this.devicePixelRatio;mc=d*this.devicePixelRatio;l.viewport(Pb,kc,lc,mc)};
# 设置裁剪区域
this.setScissor=function(a,b,c,d){l.scissor(a*this.devicePixelRatio,b*this.devicePixelRatio,c*this.devicePixelRatio,d*this.devicePixelRatio)};
# 启用或禁用裁剪测试
this.enableScissorTest=function(a){a?l.enable(l.SCISSOR_TEST):
# 禁用裁剪测试
l.disable(l.SCISSOR_TEST);
# 设置清除颜色
this.setClearColor=function(a,b){
    za.set(a);
    bb=void 0!==b?b:1;
    l.clearColor(za.r,za.g,za.b,bb)
};
# 设置清除颜色（使用十六进制）
this.setClearColorHex=function(a,b){
    console.warn("THREE.WebGLRenderer: .setClearColorHex() is being removed. Use .setClearColor() instead.");
    this.setClearColor(a,b)
};
# 获取当前清除颜色
this.getClearColor=function(){
    return za
};
# 获取当前清除透明度
this.getClearAlpha=function(){
    return bb
};
# 清除缓冲区
this.clear=function(a,b,c){
    var d=0;
    if(void 0===a||a)d|=l.COLOR_BUFFER_BIT;
    if(void 0===b||b)d|=l.DEPTH_BUFFER_BIT;
    if(void 0===c||c)d|=l.STENCIL_BUFFER_BIT;
    l.clear(d)
};
# 清除颜色缓冲区
this.clearColor=function(){
    l.clear(l.COLOR_BUFFER_BIT)
};
# 清除深度缓冲区
this.clearDepth=function(){
    l.clear(l.DEPTH_BUFFER_BIT)
};
# 清除模板缓冲区
this.clearStencil=function(){
    l.clear(l.STENCIL_BUFFER_BIT)
};
# 清除指定目标
this.clearTarget=function(a,b,c,d){
    this.setRenderTarget(a);
    this.clear(b,c,d)
};
# 重置 WebGL 状态
this.resetGLState=function(){
    ec=tc=null;
    Kb=Oa=Mb=Lb=nb=Yb=pb=-1;
    fc=!0
};
# 清除事件监听和删除对象的属性
var Hc=function(a){
    a.target.traverse(function(a){
        a.removeEventListener("remove",Hc);
        if(a instanceof THREE.Mesh||a instanceof THREE.PointCloud||a instanceof THREE.Line)delete ob[a.id];
        else if(a instanceof THREE.ImmediateRenderObject||a.immediateRenderCallback)for(var b=jb,c=b.length-1;0<=c;c--)b[c].object===a&&b.splice(c,1);
        delete a.__webglInit;
        delete a._modelViewMatrix;
        delete a._normalMatrix;
        delete a.__webglActive
    })
};
# 处理对象的释放事件
var Ic=function(a){
    a=a.target;
    a.removeEventListener("dispose",Ic);
    delete a.__webglInit;
    if(a instanceof THREE.BufferGeometry){
        for(var b in a.attributes){
            var c=a.attributes[b];
            void 0!==c.buffer&&(l.deleteBuffer(c.buffer),delete c.buffer)
        }
        J.info.memory.geometries--
    }else if(b=
# 如果 xb[a.id] 存在并且 b 不为 undefined
if(xb[a.id],void 0!==b){
    # 遍历 b 数组
    for(var c=0,d=b.length;c<d;c++){
        var e=b[c];
        # 如果 e.numMorphTargets 存在
        if(void 0!==e.numMorphTargets){
            # 遍历 e.__webglMorphTargetsBuffers 数组，删除缓冲区
            for(var f=0,g=e.numMorphTargets;f<g;f++)
                l.deleteBuffer(e.__webglMorphTargetsBuffers[f]);
            # 删除 e.__webglMorphTargetsBuffers
            delete e.__webglMorphTargetsBuffers
        }
        # 如果 e.numMorphNormals 存在
        if(void 0!==e.numMorphNormals){
            # 遍历 e.__webglMorphNormalsBuffers 数组，删除缓冲区
            f=0;
            for(g=e.numMorphNormals;f<g;f++)
                l.deleteBuffer(e.__webglMorphNormalsBuffers[f]);
            # 删除 e.__webglMorphNormalsBuffers
            delete e.__webglMorphNormalsBuffers
        }
        # 调用 Yc 函数
        Yc(e)
    }
    # 删除 xb[a.id]
    delete xb[a.id]
}
# 否则调用 Yc 函数
else 
    Yc(a);
# 将 Oa 设为 -1
Oa=-1
},
# 定义 gc 函数，参数为 a
gc=function(a){
    # a 的目标移除 "dispose" 事件的监听器
    a=a.target;
    a.removeEventListener("dispose",gc);
    # 如果 a 的 image 存在并且 a.image.__webglTextureCube 存在
    a.image&&a.image.__webglTextureCube?
        # 删除 a.image.__webglTextureCube
        (l.deleteTexture(a.image.__webglTextureCube),delete a.image.__webglTextureCube):
        # 否则删除 a.__webglTexture 和相关属性
        void 0!==a.__webglInit&&(l.deleteTexture(a.__webglTexture),delete a.__webglTexture,delete a.__webglInit);
    # J.info.memory.textures 减一
    J.info.memory.textures--
},
# 定义 Zc 函数，参数为 a
Zc=function(a){
    # a 的目标移除 "dispose" 事件的监听器
    a=a.target;
    a.removeEventListener("dispose",Zc);
    # 如果 a 存在并且 a.__webglTexture 存在
    if(a&&void 0!==a.__webglTexture){
        # 删除 a.__webglTexture 和相关属性
        l.deleteTexture(a.__webglTexture);
        delete a.__webglTexture;
        # 如果 a 是 THREE.WebGLRenderTargetCube 类型
        if(a instanceof THREE.WebGLRenderTargetCube)
            # 遍历删除 a.__webglFramebuffer 和 a.__webglRenderbuffer
            for(var b=0;6>b;b++)
                l.deleteFramebuffer(a.__webglFramebuffer[b]),l.deleteRenderbuffer(a.__webglRenderbuffer[b]);
        # 否则删除 a.__webglFramebuffer 和 a.__webglRenderbuffer
        else 
            l.deleteFramebuffer(a.__webglFramebuffer),l.deleteRenderbuffer(a.__webglRenderbuffer);
        # 删除 a.__webglFramebuffer 和 a.__webglRenderbuffer
        delete a.__webglFramebuffer;
        delete a.__webglRenderbuffer
    }
    # J.info.memory.textures 减一
    J.info.memory.textures--
},
# 定义 Dc 函数，参数为 a
Dc=function(a){
    # a 的目标移除 "dispose" 事件的监听器
    a=a.target;
    a.removeEventListener("dispose",Dc);
    # 调用 Cc 函数
    Cc(a)
},
# 定义 Yc 函数，参数为 a
Yc=function(a){
    # 定义缓冲区名称数组
    for(var b="__webglVertexBuffer __webglNormalBuffer __webglTangentBuffer __webglColorBuffer __webglUVBuffer __webglUV2Buffer __webglSkinIndicesBuffer __webglSkinWeightsBuffer __webglFaceBuffer __webglLineBuffer __webglLineDistanceBuffer".split(" "),
# 初始化变量c为0，变量d为数组b的长度
c=0,d=b.length;c<d;c++){
    # 循环遍历数组b
    var e=b[c];
    # 如果变量a中存在属性e
    void 0!==a[e]&&(l.deleteBuffer(a[e]),delete a[e])
}
# 如果变量a中存在属性__webglCustomAttributesList
if(void 0!==a.__webglCustomAttributesList){
    # 遍历__webglCustomAttributesList属性
    for(e in a.__webglCustomAttributesList)
        # 删除__webglCustomAttributesList属性中的buffer
        l.deleteBuffer(a.__webglCustomAttributesList[e].buffer);
    # 删除__webglCustomAttributesList属性
    delete a.__webglCustomAttributesList
}
# 减少内存中的几何体数量
J.info.memory.geometries--
# 定义函数Cc，参数为a
Cc=function(a){
    # 获取a对象中的program属性的program属性
    var b=a.program.program;
    # 如果b存在
    if(void 0!==b){
        # 将a对象中的program属性设为undefined
        a.program=void 0;
        var c,d,e=!1;
        a=0;
        # 遍历数组hb
        for(c=hb.length;a<c;a++)
            # 获取数组hb中的元素
            if(d=hb[a],d.program===b){
                # 将元素d的usedTimes属性减1
                d.usedTimes--;
                # 如果usedTimes属性为0
                0===d.usedTimes&&(e=!0);
                break
            }
        # 如果e为true
        if(!0===e){
            e=[];
            a=0;
            # 遍历数组hb
            for(c=hb.length;a<c;a++)
                # 获取数组hb中的元素
                d=hb[a],
                # 如果元素d的program属性不等于b
                d.program!==b&&e.push(d);
            # 将数组e赋值给数组hb
            hb=e;
            # 删除程序对象b
            l.deleteProgram(b);
            # 减少内存中的程序数量
            J.info.memory.programs--
        }
    }
};
# 定义函数renderBufferImmediate，参数为a、b、c
this.renderBufferImmediate=function(a,b,c){
    # 调用函数f
    f();
    # 如果a有位置属性且没有__webglVertexBuffer属性
    a.hasPositions&&!a.__webglVertexBuffer&&(a.__webglVertexBuffer=l.createBuffer());
    # 如果a有法线属性且没有__webglNormalBuffer属性
    a.hasNormals&&!a.__webglNormalBuffer&&(a.__webglNormalBuffer=l.createBuffer());
    # 如果a有UV属性且没有__webglUvBuffer属性
    a.hasUvs&&!a.__webglUvBuffer&&(a.__webglUvBuffer=l.createBuffer());
    # 如果a有颜色属性且没有__webglColorBuffer属性
    a.hasColors&&!a.__webglColorBuffer&&(a.__webglColorBuffer=l.createBuffer());
    # 如果a有位置属性
    a.hasPositions&&(
        # 绑定__webglVertexBuffer属性
        l.bindBuffer(l.ARRAY_BUFFER,a.__webglVertexBuffer),
        # 将位置数组数据传入缓冲区
        l.bufferData(l.ARRAY_BUFFER,a.positionArray,l.DYNAMIC_DRAW),
        # 启用顶点属性
        g(b.attributes.position),
        # 指定顶点属性的数据格式
        l.vertexAttribPointer(b.attributes.position,3,l.FLOAT,!1,0,0)
    );
    # 如果a有法线属性
    if(a.hasNormals){
        # 绑定__webglNormalBuffer属性
        l.bindBuffer(l.ARRAY_BUFFER,a.__webglNormalBuffer);
        # 如果c的shading属性为THREE.FlatShading
        if(c.shading===THREE.FlatShading){
            var d,e,k,m,n,p,r,q,t,s,v,u=3*a.count;
            # 遍历法线数组
            for(v=0;v<u;v+=9)
                s=a.normalArray,
                d=s[v],
                e=s[v+1],
                k=s[v+2],
                m=s[v+3],
                p=s[v+4],
                q=s[v+5],
                n=s[v+6],
                r=s[v+7],
                t=s[v+8],
                d=(d+m+n)/3,
                e=(e+p+r)/3,
                k=(k+q+t)/3,
                s[v]=d,
                s[v+1]=e,
                s[v+2]=k,
                s[v+3]=d,
                s[v+4]=e,
                s[v+5]=k,
                s[v+6]=d,
                s[v+7]=e,
                s[v+8]=k
        }
    }
}
# 设置顶点法线数据
l.bufferData(l.ARRAY_BUFFER, a.normalArray, l.DYNAMIC_DRAW)
# 启用顶点法线属性
g(b.attributes.normal)
# 指定顶点法线属性的数据格式和位置
l.vertexAttribPointer(b.attributes.normal, 3, l.FLOAT, !1, 0, 0)
# 如果有 UV 数据并且有纹理映射，则绑定 UV 缓冲区
if a.hasUvs and c.map:
    l.bindBuffer(l.ARRAY_BUFFER, a.__webglUvBuffer)
    # 设置 UV 数据
    l.bufferData(l.ARRAY_BUFFER, a.uvArray, l.DYNAMIC_DRAW)
    # 启用 UV 属性
    g(b.attributes.uv)
    # 指定 UV 属性的数据格式和位置
    l.vertexAttribPointer(b.attributes.uv, 2, l.FLOAT, !1, 0, 0)
# 如果有顶点颜色并且顶点颜色不是无颜色状态
if a.hasColors and c.vertexColors !== THREE.NoColors:
    # 绑定顶点颜色缓冲区
    l.bindBuffer(l.ARRAY_BUFFER, a.__webglColorBuffer)
    # 设置顶点颜色数据
    l.bufferData(l.ARRAY_BUFFER, a.colorArray, l.DYNAMIC_DRAW)
    # 启用顶点颜色属性
    g(b.attributes.color)
    # 指定顶点颜色属性的数据格式和位置
    l.vertexAttribPointer(b.attributes.color, 3, l.FLOAT, !1, 0, 0)
# 执行绘制前的准备工作
h()
# 绘制三角形
l.drawArrays(l.TRIANGLES, 0, a.count)
# 重置顶点计数
a.count = 0
# 渲染缓冲区的直接渲染
this.renderBufferDirect = function(a, b, c, d, g, h):
    # 如果不可见，则不渲染
    if !1 !== d.visible:
        # 获取渲染 ID
        a = G(a, b, c, d, h)
        b = !1
        c = 16777215 * g.id + 2 * a.id + (d.wireframe ? 1 : 0)
        # 如果渲染 ID 发生变化，则重新设置状态
        if c !== Oa:
            Oa = c
            b = !0
        # 如果需要重新设置状态，则执行
        if b:
            f()
        # 如果渲染对象是网格
        if h instanceof THREE.Mesh:
            # 根据是否是线框模式选择绘制类型
            h = !0 === d.wireframe ? l.LINES : l.TRIANGLES
            # 如果有索引数据
            if c = g.attributes.index:
                # 根据索引数据类型选择绘制方式
                if c.array instanceof Uint32Array and pa.get("OES_element_index_uint"):
                    k = l.UNSIGNED_INT
                    m = 4
                else:
                    k = l.UNSIGNED_SHORT
                    m = 2
                n = g.offsets
                # 如果没有偏移数据
                if 0 === n.length:
                    # 执行绘制
                    if b:
                        e(d, a, g, 0)
                        l.bindBuffer(l.ELEMENT_ARRAY_BUFFER, c.buffer)
                    l.drawElements(h, c.array.length, k, 0)
                    J.info.render.calls++
                    J.info.render.vertices += c.array.length
                    J.info.render.faces += c.array.length / 3
                else:
                    b = !0
                    # 遍历偏移数据并执行绘制
                    for var p = 0, r = n.length; p < r; p++:
                        var q = n[p].index
                        if b:
                            e(d, a, g, q)
                            l.bindBuffer(l.ELEMENT_ARRAY_BUFFER, c.buffer)
                        l.drawElements(h, n[p].count, k, n[p].start * m)
                        J.info.render.calls++
                        J.info.render.vertices += n[p].count
                        J.info.render.faces += n[p].count / 3
            else:
                if b:
                    e(d, a, g, 0)
# 检查属性 g.attributes.position 是否存在，如果存在则执行以下代码，否则跳过
d=g.attributes.position,
# 使用 WebGL 上下文对象 l 执行 drawArrays 方法，绘制图形
l.drawArrays(h,0,d.array.length/3),
# 增加渲染调用次数
J.info.render.calls++,
# 增加渲染顶点数
J.info.render.vertices+=d.array.length/3,
# 增加渲染面数
J.info.render.faces+=d.array.length/9;
# 如果属性 h 是 THREE.PointCloud 类型，则执行以下代码
else if(h instanceof THREE.PointCloud)
    # 如果 b 存在，则执行 e 函数，传入参数 d, a, g, 0
    b&&e(d,a,g,0),
    # 使用 WebGL 上下文对象 l 执行 drawArrays 方法，绘制点
    d=g.attributes.position,
    l.drawArrays(l.POINTS,0,d.array.length/3),
    # 增加渲染调用次数
    J.info.render.calls++,
    # 增加渲染点数
    J.info.render.points+=d.array.length/3;
# 如果属性 h 是 THREE.Line 类型，则执行以下代码
else if(h instanceof THREE.Line)
    # 根据 h.mode 的值确定绘制模式
    h=h.mode===THREE.LineStrip?l.LINE_STRIP:l.LINES,
    # 设置线宽
    A(d.linewidth),
    # 如果属性 c 存在，则执行以下代码
    if(c=g.attributes.index)
        # 根据 c.array 的类型确定数据类型和偏移量
        c.array instanceof Uint32Array?(k=l.UNSIGNED_INT,m=4):(k=l.UNSIGNED_SHORT,m=2),
        # 获取 g.offsets 的值
        n=g.offsets,
        # 如果 offsets 为空，则执行以下代码
        0===n.length
            # 如果 b 存在，则执行 e 函数，传入参数 d, a, g, 0
            b&&(e(d,a,g,0),l.bindBuffer(l.ELEMENT_ARRAY_BUFFER,c.buffer)),
            # 使用 WebGL 上下文对象 l 执行 drawElements 方法，绘制线
            l.drawElements(h,c.array.length,k,0),
            # 增加渲染调用次数
            J.info.render.calls++,
            # 增加渲染顶点数
            J.info.render.vertices+=c.array.length;
        # 如果 offsets 不为空，则执行以下代码
        else for(1<n.length&&(b=!0),p=0,r=n.length;p<r;p++)
            q=n[p].index,
            # 如果 b 存在，则执行 e 函数，传入参数 d, a, g, q
            b&&(e(d,a,g,q),l.bindBuffer(l.ELEMENT_ARRAY_BUFFER,c.buffer)),
            # 使用 WebGL 上下文对象 l 执行 drawElements 方法，绘制线
            l.drawElements(h,n[p].count,k,n[p].start*m),
            # 增加渲染调用次数
            J.info.render.calls++,
            # 增加渲染顶点数
            J.info.render.vertices+=n[p].count;
    # 如果属性 c 不存在，则执行以下代码
    else 
        # 如果 b 存在，则执行 e 函数，传入参数 d, a, g, 0
        b&&e(d,a,g,0),
        # 使用 WebGL 上下文对象 l 执行 drawArrays 方法，绘制线
        d=g.attributes.position,
        l.drawArrays(h,0,d.array.length/3),
        # 增加渲染调用次数
        J.info.render.calls++,
        # 增加渲染点数
        J.info.render.points+=d.array.length/3};
# 定义 renderBuffer 函数，传入参数 a, b, c, d, e, k
this.renderBuffer=function(a,b,c,d,e,k){
    # 如果 d.visible 为真，则执行以下代码
    if(!1!==d.visible){
        # 调用 G 函数，传入参数 a, b, c, d, k，并将返回值赋给变量 c
        c=G(a,b,c,d,k);
        # 获取 c 的属性
        b=c.attributes;
        # 计算 id 值
        a=!1;
        c=16777215*e.id+2*c.id+(d.wireframe?1:0);
        c!==Oa&&(Oa=c,a=!0);
        a&&f();
        # 如果 d.morphTargets 为假，并且 b.position 大于等于 0，则执行以下代码
        if(!d.morphTargets&&0<=b.position)
            a&&(l.bindBuffer(l.ARRAY_BUFFER,e.__webglVertexBuffer),g(b.position),l.vertexAttribPointer(b.position,3,l.FLOAT,!1,0,0));
        # 如果 k.morphTargetBase 存在，则执行以下代码
        else if(k.morphTargetBase)
            c=d.program.attributes;
            -1!==k.morphTargetBase&&0<=c.position?(l.bindBuffer(l.ARRAY_BUFFER,
# 如果存在基于WebGL的形态目标缓冲区，则将其绑定到位置属性并指定指针
e.__webglMorphTargetsBuffers[k.morphTargetBase]),g(c.position),l.vertexAttribPointer(c.position,3,l.FLOAT,!1,0,0):0<=c.position&&(l.bindBuffer(l.ARRAY_BUFFER,e.__webglVertexBuffer),g(c.position),l.vertexAttribPointer(c.position,3,l.FLOAT,!1,0,0));
# 如果强制指定了形态目标顺序，则按顺序绑定形态目标缓冲区和法线缓冲区
if(k.morphTargetForcedOrder.length)for(var m=0,n=k.morphTargetForcedOrder,r=k.morphTargetInfluences;m<d.numSupportedMorphTargets&&m<n.length;)0<=c["morphTarget"+m]&&(l.bindBuffer(l.ARRAY_BUFFER,e.__webglMorphTargetsBuffers[n[m]]),g(c["morphTarget"+m]),l.vertexAttribPointer(c["morphTarget"+m],3,l.FLOAT,!1,0,0)),0<=c["morphNormal"+m]&&d.morphNormals&&(l.bindBuffer(l.ARRAY_BUFFER,e.__webglMorphNormalsBuffers[n[m]]),g(c["morphNormal"+m]),l.vertexAttribPointer(c["morphNormal"+m],3,l.FLOAT,!1,0,0)),k.__webglMorphTargetInfluences[m]=r[n[m]],m++;
# 如果没有强制指定形态目标顺序，则根据形态目标影响值排序并绑定缓冲区
else{var n=[],r=k.morphTargetInfluences,q,t=r.length;for(q=0;q<t;q++)m=r[q],0<m&&n.push([m,q]);n.length>d.numSupportedMorphTargets?(n.sort(p),n.length=d.numSupportedMorphTargets):n.length>d.numSupportedMorphNormals?n.sort(p):0===n.length&&n.push([0,0]);for(m=0;m<d.numSupportedMorphTargets;)n[m]?(q=n[m][1],0<=c["morphTarget"+m]&&(l.bindBuffer(l.ARRAY_BUFFER,e.__webglMorphTargetsBuffers[q]),g(c["morphTarget"+m]),l.vertexAttribPointer(c["morphTarget"+m],3,l.FLOAT,!1,0,0)),0<=c["morphNormal"+m]&&d.morphNormals&&(l.bindBuffer(l.ARRAY_BUFFER,e.__webglMorphNormalsBuffers[q]),g(c["morphNormal"+m]),l.vertexAttribPointer(c["morphNormal"+m],3,l.FLOAT,!1,0,0)),k.__webglMorphTargetInfluences[m]=r[q]):k.__webglMorphTargetInfluences[m]=0,m++}null!==d.program.uniforms.morphTargetInfluences&&
# 设置uniform变量的值
l.uniform1fv(d.program.uniforms.morphTargetInfluences,k.__webglMorphTargetInfluences)
# 如果存在自定义属性
if(a):
    # 遍历自定义属性列表
    if(e.__webglCustomAttributesList):
        for(c=0,r=e.__webglCustomAttributesList.length;c<r;c++):
            n=e.__webglCustomAttributesList[c]
            # 如果自定义属性对应的缓冲区索引大于等于0
            if(0<=b[n.buffer.belongsToAttribute]):
                # 绑定缓冲区
                l.bindBuffer(l.ARRAY_BUFFER,n.buffer)
                # 设置属性指针
                g(b[n.buffer.belongsToAttribute])
                l.vertexAttribPointer(b[n.buffer.belongsToAttribute],n.size,l.FLOAT,!1,0,0)
    # 如果颜色属性对应的缓冲区索引大于等于0
    if(0<=b.color):
        # 如果几何体颜色数量大于0或者面数量大于0
        if(0<k.geometry.colors.length||0<k.geometry.faces.length):
            # 绑定颜色缓冲区
            l.bindBuffer(l.ARRAY_BUFFER,e.__webglColorBuffer)
            # 设置颜色属性指针
            g(b.color)
            l.vertexAttribPointer(b.color,3,l.FLOAT,!1,0,0)
        else:
            # 如果默认属性值不为空，设置颜色属性默认值
            void 0!==d.defaultAttributeValues&&l.vertexAttrib3fv(b.color,d.defaultAttributeValues.color)
    # 如果法线属性对应的缓冲区索引大于等于0
    if(0<=b.normal):
        # 绑定法线缓冲区
        l.bindBuffer(l.ARRAY_BUFFER,e.__webglNormalBuffer)
        # 设置法线属性指针
        g(b.normal)
        l.vertexAttribPointer(b.normal,3,l.FLOAT,!1,0,0)
    # 如果切线属性对应的缓冲区索引大于等于0
    if(0<=b.tangent):
        # 绑定切线缓冲区
        l.bindBuffer(l.ARRAY_BUFFER,e.__webglTangentBuffer)
        # 设置切线属性指针
        g(b.tangent)
        l.vertexAttribPointer(b.tangent,4,l.FLOAT,!1,0,0)
    # 如果UV属性对应的缓冲区索引大于等于0
    if(0<=b.uv):
        # 如果几何体UV坐标数量大于0
        if(k.geometry.faceVertexUvs[0]):
            # 绑定UV缓冲区
            l.bindBuffer(l.ARRAY_BUFFER,e.__webglUVBuffer)
            # 设置UV属性指针
            g(b.uv)
            l.vertexAttribPointer(b.uv,2,l.FLOAT,!1,0,0)
        else:
            # 如果默认属性值不为空，设置UV属性默认值
            void 0!==d.defaultAttributeValues&&l.vertexAttrib2fv(b.uv,d.defaultAttributeValues.uv)
    # 如果UV2属性对应的缓冲区索引大于等于0
    if(0<=b.uv2):
        # 如果几何体UV2坐标数量大于0
        if(k.geometry.faceVertexUvs[1]):
            # 绑定UV2缓冲区
            l.bindBuffer(l.ARRAY_BUFFER,e.__webglUV2Buffer)
            # 设置UV2属性指针
            g(b.uv2)
            l.vertexAttribPointer(b.uv2,2,l.FLOAT,!1,0,0)
        else:
            # 如果默认属性值不为空，设置UV2属性默认值
            void 0!==d.defaultAttributeValues&&l.vertexAttrib2fv(b.uv2,d.defaultAttributeValues.uv2)
    # 如果启用了蒙皮动画并且皮肤索引和权重属性对应的缓冲区索引大于等于0
    if(d.skinning&&0<=b.skinIndex&&0<=b.skinWeight):
        # 绑定皮肤索引缓冲区
        l.bindBuffer(l.ARRAY_BUFFER,e.__webglSkinIndicesBuffer)
        # 设置皮肤索引属性指针
        g(b.skinIndex)
        l.vertexAttribPointer(b.skinIndex,
# 设置顶点属性指针，指定顶点属性的数据格式和存储方式
l.vertexAttribPointer(b.position,3,l.FLOAT,!1,0,0);
# 如果包含法线属性，设置法线属性指针
b.normal && (l.vertexAttribPointer(b.normal,3,l.FLOAT,!1,0,0));
# 如果包含颜色属性，设置颜色属性指针
b.color && (l.vertexAttribPointer(b.color,3,l.FLOAT,!1,0,0));
# 如果包含UV属性，设置UV属性指针
b.uv && (l.vertexAttribPointer(b.uv,2,l.FLOAT,!1,0,0));
# 如果包含UV2属性，设置UV2属性指针
b.uv2 && (l.vertexAttribPointer(b.uv2,2,l.FLOAT,!1,0,0));
# 如果包含皮肤指数属性，设置皮肤指数属性指针
b.skinIndex && (l.vertexAttribPointer(b.skinIndex,4,l.FLOAT,!1,0,0));
# 如果包含皮肤权重属性，设置皮肤权重属性指针
b.skinWeight && (l.vertexAttribPointer(b.skinWeight,4,l.FLOAT,!1,0,0));
# 如果包含线段距离属性，设置线段距离属性指针
0<=b.lineDistance && (l.vertexAttribPointer(b.lineDistance,1,l.FLOAT,!1,0,0));

# 绘制三维模型
h();
# 如果是网格模型
k instanceof THREE.Mesh && (
    # 根据是否是 Uint32Array 设置绘制模式
    k = e.__typeArray === Uint32Array ? l.UNSIGNED_INT : l.UNSIGNED_SHORT,
    # 如果是线框模式
    d.wireframe ? (
        # 设置线框宽度
        A(d.wireframeLinewidth),
        # 如果有顶点索引缓冲区，绑定并绘制线段
        a && l.bindBuffer(l.ELEMENT_ARRAY_BUFFER, e.__webglLineBuffer),
        l.drawElements(l.LINES, e.__webglLineCount, k, 0)
    ) : (
        # 如果有顶点索引缓冲区，绑定并绘制三角形
        a && l.bindBuffer(l.ELEMENT_ARRAY_BUFFER, e.__webglFaceBuffer),
        l.drawElements(l.TRIANGLES, e.__webglFaceCount, k, 0)
    ),
    # 更新渲染统计信息
    J.info.render.calls++,
    J.info.render.vertices += e.__webglFaceCount,
    J.info.render.faces += e.__webglFaceCount / 3
);
# 如果是线段模型
k instanceof THREE.Line && (
    # 根据线段模式设置绘制模式
    k = k.mode === THREE.LineStrip ? l.LINE_STRIP : l.LINES,
    # 设置线宽
    A(d.linewidth),
    # 绘制线段
    l.drawArrays(k, 0, e.__webglLineCount),
    # 更新渲染统计信息
    J.info.render.calls++
);
# 如果是点云模型
k instanceof THREE.PointCloud && (
    # 绘制点云
    l.drawArrays(l.POINTS, 0, e.__webglParticleCount),
    # 更新渲染统计信息
    J.info.render.calls++,
    J.info.render.points += e.__webglParticleCount
);

# 渲染场景
this.render = function(a, b, c, d) {
    # 如果相机不是 THREE.Camera 的实例，输出错误信息
    if (!1 === b instanceof THREE.Camera) console.error("THREE.WebGLRenderer.render: camera is not an instance of THREE.Camera.");
    else {
        # 获取场景中的雾效
        var e = a.fog;
        Kb = Oa = -1;
        ec = null;
        fc = !0;
        # 如果需要自动更新场景，更新场景的世界矩阵
        !0 === a.autoUpdate && a.updateMatrixWorld();
        void 0 === b.parent && b.updateMatrixWorld();
        # 遍历场景中的对象
        a.traverse(function(a) {
            # 如果是蒙皮网格，更新骨骼动画
            a instanceof THREE.SkinnedMesh && a.skeleton.update()
        });
        # 获取相机的世界逆矩阵
        b.matrixWorldInverse.getInverse(b.matrixWorld);
        # 计算投影矩阵和相机世界逆矩阵的乘积
        Ac.multiplyMatrices(b.projectionMatrix, b.matrixWorldInverse);
        # 从矩阵中提取相机的视锥
        Ec.setFromMatrix(Ac);
# 将数组长度设置为0
cb.length=0;
# 将数组长度设置为0
Jb.length=0;
# 将数组长度设置为0
Ib.length=0;
# 将数组长度设置为0
yb.length=0;
# 将数组长度设置为0
Ra.length=0;
# 调用函数q，传入参数a和a
q(a,a);
# 如果J.sortObjects为真，则对Jb和Ib进行排序
!0===J.sortObjects&&(Jb.sort(k),Ib.sort(n));
# 调用id.render函数，传入参数a和b
id.render(a,b);
# 将J.info.render.calls、J.info.render.vertices、J.info.render.faces、J.info.render.points都设置为0
J.info.render.calls=0;
J.info.render.vertices=0;
J.info.render.faces=0;
J.info.render.points=0;
# 设置渲染目标为c
this.setRenderTarget(c);
# 如果this.autoClear为真或者d为真，则调用this.clear函数，传入参数this.autoClearColor、this.autoClearDepth、this.autoClearStencil
(this.autoClear||d)&&this.clear(this.autoClearColor,this.autoClearDepth,this.autoClearStencil);
# 将d设置为0
d=0;
# 循环遍历jb数组
for(var f=jb.length;d<f;d++){
    var g=jb[d],h=g.object;
    # 如果h.visible为真，则调用x函数，传入参数h和b
    h.visible&&(x(h,b),t(g))
}
# 如果a.overrideMaterial为真，则将d设置为a.overrideMaterial，否则将d设置为null
a.overrideMaterial?(d=a.overrideMaterial,this.setBlending(d.blending,d.blendEquation,d.blendSrc,d.blendDst),this.setDepthTest(d.depthTest),this.setDepthWrite(d.depthWrite),B(d.polygonOffset,d.polygonOffsetFactor,d.polygonOffsetUnits),m(Jb,b,cb,e,!0,d),m(Ib,b,cb,e,!0,d),r(jb,"",b,cb,e,!1,d)):(d=null,this.setBlending(THREE.NoBlending),m(Jb,b,cb,e,!1,d),r(jb,"opaque",b,cb,e,!1,d),m(Ib,b,cb,e,!0,d),r(jb,"transparent",b,cb,e,!0,d));
# 调用jd.render函数，传入参数a和b
jd.render(a,b);
# 调用kd.render函数，传入参数a、b、Uc、Vc
kd.render(a,b,Uc,Vc);
# 如果c为真且c.generateMipmaps为真且c.minFilter不等于THREE.NearestFilter且c.minFilter不等于THREE.LinearFilter，则调用C函数，传入参数c
c&&c.generateMipmaps&&c.minFilter!==THREE.NearestFilter&&c.minFilter!==THREE.LinearFilter&&C(c);
# 设置深度测试为真
this.setDepthTest(!0);
# 设置深度写入为真
this.setDepthWrite(!0)
# 设置材质的面
this.setMaterialFaces=function(a){
    var b=a.side===THREE.DoubleSide;
    a=a.side===THREE.BackSide;
    Lb!==b&&(b?l.disable(l.CULL_FACE):l.enable(l.CULL_FACE),Lb=b);
    Mb!==a&&(a?l.frontFace(l.CW):l.frontFace(l.CCW),Mb=a)
};

# 设置深度测试
this.setDepthTest=function(a){
    Yb!==a&&(a?l.enable(l.DEPTH_TEST):l.disable(l.DEPTH_TEST),Yb=a)
};

# 设置深度写入
this.setDepthWrite=function(a){
    nb!==a&&(l.depthMask(a),nb=a)
};

# 设置混合模式
this.setBlending=function(a,b,c,d){
    a!==pb&&(a===THREE.NoBlending?l.disable(l.BLEND):a===THREE.AdditiveBlending?(l.enable(l.BLEND),l.blendEquation(l.FUNC_ADD),l.blendFunc(l.SRC_ALPHA,l.ONE)):a===THREE.SubtractiveBlending?(l.enable(l.BLEND),l.blendEquation(l.FUNC_ADD),l.blendFunc(l.ZERO,l.ONE_MINUS_SRC_COLOR)):a===THREE.MultiplyBlending?(l.enable(l.BLEND),l.blendEquation(l.FUNC_ADD),l.blendFunc(l.ZERO,l.SRC_COLOR)):a===THREE.CustomBlending?l.enable(l.BLEND):(l.enable(l.BLEND),l.blendEquationSeparate(l.FUNC_ADD,l.FUNC_ADD),l.blendFuncSeparate(l.SRC_ALPHA,l.ONE_MINUS_SRC_ALPHA,l.ONE,l.ONE_MINUS_SRC_ALPHA)),pb=a);
    if(a===THREE.CustomBlending){
        if(b!==Nb&&(l.blendEquation(Q(b)),Nb=b),c!==Ob||d!==Xb)l.blendFunc(Q(c),Q(d)),Ob=c,Xb=d
    }else Xb=Ob=Nb=null
};

# 上传纹理
this.uploadTexture=function(a){
    void 0===a.__webglInit&&(a.__webglInit=!0,a.addEventListener("dispose",gc),a.__webglTexture=l.createTexture(),J.info.memory.textures++);
    l.bindTexture(l.TEXTURE_2D,a.__webglTexture);
    l.pixelStorei(l.UNPACK_FLIP_Y_WEBGL,a.flipY);
    l.pixelStorei(l.UNPACK_PREMULTIPLY_ALPHA_WEBGL,a.premultiplyAlpha);
    l.pixelStorei(l.UNPACK_ALIGNMENT,a.unpackAlignment);
    a.image=R(a.image,cd);
    var b=a.image,c=THREE.Math.isPowerOfTwo(b.width)&&
};
// 检查 b 的高度是否为 2 的幂次方，将结果赋值给变量 THREE.Math.isPowerOfTwo(b.height)，同时将 Q(a.format) 的结果赋值给变量 d，将 Q(a.type) 的结果赋值给变量 e
THREE.Math.isPowerOfTwo(b.height),d=Q(a.format),e=Q(a.type);
// 将纹理对象 a 绑定到纹理目标 l.TEXTURE_2D 上
F(l.TEXTURE_2D,a,c);
// 如果 a 是 THREE.DataTexture 类型并且 mipmaps 数组长度大于 0 并且 c 为真，则执行以下代码块
if(a instanceof THREE.DataTexture)
    if(0<f.length&&c){
        for(var g=0,h=f.length;g<h;g++)
            b=f[g],l.texImage2D(l.TEXTURE_2D,g,d,b.width,b.height,0,d,e,b.data);
        a.generateMipmaps=!1
    }
// 如果 a 是 THREE.CompressedTexture 类型，则执行以下代码块
else if(a instanceof THREE.CompressedTexture)
    for(g=0,h=f.length;g<h;g++)
        b=f[g],
        a.format!==THREE.RGBAFormat&&a.format!==THREE.RGBFormat?-1<Nc().indexOf(d)?l.compressedTexImage2D(l.TEXTURE_2D,g,d,b.width,b.height,0,b.data):console.warn("Attempt to load unsupported compressed texture format"):l.texImage2D(l.TEXTURE_2D,g,d,b.width,b.height,0,d,e,b.data);
// 如果 mipmaps 数组长度大于 0 并且 c 为真，则执行以下代码块
else if(0<f.length&&c){
    g=0;
    for(h=f.length;g<h;g++)
        b=f[g],l.texImage2D(l.TEXTURE_2D,g,d,d,e,b);
    a.generateMipmaps=!1
}
// 如果 a 的 generateMipmaps 为真并且 c 为真，则生成纹理的 mipmap
else
    l.texImage2D(l.TEXTURE_2D,0,d,d,e,a.image);
// 如果 a 的 onUpdate 方法存在，则调用该方法
a.generateMipmaps&&c&&l.generateMipmap(l.TEXTURE_2D);
a.needsUpdate=!1;
if(a.onUpdate)
    a.onUpdate();
// 设置纹理对象 a 到纹理单元 l.TEXTURE0+b
this.setTexture=function(a,b){
    l.activeTexture(l.TEXTURE0+b);
    a.needsUpdate?J.uploadTexture(a):l.bindTexture(l.TEXTURE_2D,a.__webglTexture)
};
// 设置渲染目标为 a
this.setRenderTarget=function(a){
    var b=a instanceof THREE.WebGLRenderTargetCube;
    if(a&&void 0===a.__webglFramebuffer){
        void 0===a.depthBuffer&&(a.depthBuffer=!0);
        void 0===a.stencilBuffer&&(a.stencilBuffer=!0);
        a.addEventListener("dispose",Zc);
        a.__webglTexture=l.createTexture();
        J.info.memory.textures++;
        var c=THREE.Math.isPowerOfTwo(a.width)&&THREE.Math.isPowerOfTwo(a.height),d=Q(a.format),e=Q(a.type);
        if(b){
            a.__webglFramebuffer=[];
            a.__webglRenderbuffer=[];
# 绑定纹理和设置纹理参数
l.bindTexture(l.TEXTURE_CUBE_MAP,a.__webglTexture);
F(l.TEXTURE_CUBE_MAP,a,c);
# 遍历六个面，创建帧缓冲和渲染缓冲
for(var f=0;6>f;f++){
    a.__webglFramebuffer[f]=l.createFramebuffer();
    a.__webglRenderbuffer[f]=l.createRenderbuffer();
    l.texImage2D(l.TEXTURE_CUBE_MAP_POSITIVE_X+f,0,d,a.width,a.height,0,d,e,null);
    var g=a,h=l.TEXTURE_CUBE_MAP_POSITIVE_X+f;
    l.bindFramebuffer(l.FRAMEBUFFER,a.__webglFramebuffer[f]);
    l.framebufferTexture2D(l.FRAMEBUFFER,l.COLOR_ATTACHMENT0,h,g.__webglTexture,0);
    H(a.__webglRenderbuffer[f],a)
}
# 生成多级渐远纹理
c&&l.generateMipmap(l.TEXTURE_CUBE_MAP);
# 如果不是立方体贴图，则创建帧缓冲和渲染缓冲
else 
    a.__webglFramebuffer=l.createFramebuffer();
    a.__webglRenderbuffer=a.shareDepthFrom?a.shareDepthFrom.__webglRenderbuffer:l.createRenderbuffer();
    l.bindTexture(l.TEXTURE_2D,a.__webglTexture);
    F(l.TEXTURE_2D,a,c);
    l.texImage2D(l.TEXTURE_2D,0,d,a.width,a.height,0,d,e,null);
    d=l.TEXTURE_2D;
    l.bindFramebuffer(l.FRAMEBUFFER,a.__webglFramebuffer);
    l.framebufferTexture2D(l.FRAMEBUFFER,l.COLOR_ATTACHMENT0,d,a.__webglTexture,0);
    # 设置深度和模板缓冲
    a.shareDepthFrom?a.depthBuffer&&!a.stencilBuffer?l.framebufferRenderbuffer(l.FRAMEBUFFER,l.DEPTH_ATTACHMENT,l.RENDERBUFFER,a.__webglRenderbuffer):a.depthBuffer&&a.stencilBuffer&&l.framebufferRenderbuffer(l.FRAMEBUFFER,l.DEPTH_STENCIL_ATTACHMENT,l.RENDERBUFFER,a.__webglRenderbuffer):H(a.__webglRenderbuffer,a);
    # 生成多级渐远纹理
    c&&l.generateMipmap(l.TEXTURE_2D);
# 解绑纹理和渲染缓冲
b?l.bindTexture(l.TEXTURE_CUBE_MAP,null):l.bindTexture(l.TEXTURE_2D,null);
l.bindRenderbuffer(l.RENDERBUFFER,null);
l.bindFramebuffer(l.FRAMEBUFFER,null)
# 设置帧缓冲和宽高
a?(b=b?a.__webglFramebuffer[a.activeCubeFace]:a.__webglFramebuffer,c=a.width,a=a.height,e=d=0):(b=null,c=lc,a=mc,
# 创建一个 WebGL 渲染目标对象，设置宽度、高度和参数
THREE.WebGLRenderTarget=function(a,b,c){
    # 设置宽度和高度
    this.width=a;
    this.height=b;
    # 设置纹理的 S 方向的环绕方式，默认为 ClampToEdgeWrapping
    this.wrapS=void 0!==c.wrapS?c.wrapS:THREE.ClampToEdgeWrapping;
    # 设置纹理的 T 方向的环绕方式，默认为 ClampToEdgeWrapping
    this.wrapT=void 0!==c.wrapT?c.wrapT:THREE.ClampToEdgeWrapping;
    # 设置纹理的放大过滤方式，默认为 LinearFilter
    this.magFilter=void 0!==c.magFilter?c.magFilter:THREE.LinearFilter;
    # 设置纹理的缩小过滤方式，默认为 LinearMipMapLinearFilter
    this.minFilter=void 0!==c.minFilter?c.minFilter:THREE.LinearMipMapLinearFilter;
    # 设置各向异性过滤，默认为 1
    this.anisotropy=void 0!==c.anisotropy?c.anisotropy:1;
    # 设置纹理的偏移量，默认为 (0,0)
    this.offset=new THREE.Vector2(0,0);
    # 设置纹理的重复次数，默认为 (1,1)
    this.repeat=new THREE.Vector2(1,1);
    # 设置纹理的格式，默认为 RGBAFormat
    this.format=void 0!==c.format?c.format:THREE.RGBAFormat;
    # 设置纹理的数据类型，默认为 UnsignedByteType
    this.type=void 0!==c.type?c.type:THREE.UnsignedByteType;
    # 设置是否需要深度缓冲，默认为 true
    this.depthBuffer=void 0!==c.depthBuffer?c.depthBuffer:!0;
    # 设置是否需要模板缓冲，默认为 true
    this.stencilBuffer=void 0!==c.stencilBuffer?c.stencilBuffer:!0;
    # 设置是否生成 Mipmaps，默认为 true
    this.generateMipmaps=!0;
    # 设置共享深度缓冲的渲染目标对象，默认为 null
    this.shareDepthFrom=null;
};

# WebGL 渲染目标对象的原型方法
THREE.WebGLRenderTarget.prototype={
    constructor:THREE.WebGLRenderTarget,
    # 设置渲染目标对象的尺寸
    setSize:function(a,b){
        this.width=a;
        this.height=b;
    },
    # 克隆一个新的渲染目标对象
    clone:function(){
        var a=new THREE.WebGLRenderTarget(this.width,this.height);
        a.wrapS=this.wrapS;
        a.wrapT=this.wrapT;
        a.magFilter=this.magFilter;
        a.minFilter=this.minFilter;
        a.anisotropy=this.anisotropy;
        a.offset.copy(this.offset);
        a.repeat.copy(this.repeat);
        a.format=this.format;
        a.type=this.type;
        a.depthBuffer=this.depthBuffer;
        a.stencilBuffer=this.stencilBuffer;
        a.generateMipmaps=this.generateMipmaps;
# 创建一个新的类THREE.WebGLRenderTargetCube，继承自THREE.WebGLRenderTarget
THREE.WebGLRenderTargetCube=function(a,b,c){
    THREE.WebGLRenderTarget.call(this,a,b,c);
    this.activeCubeFace=0
};

# 将THREE.WebGLRenderTargetCube的原型设置为THREE.WebGLRenderTarget的实例
THREE.WebGLRenderTargetCube.prototype=Object.create(THREE.WebGLRenderTarget.prototype);

# 创建一个新的类THREE.WebGLExtensions，接受一个参数a
THREE.WebGLExtensions=function(a){
    var b={};
    this.get=function(c){
        # 如果b对象中已经存在c属性，则直接返回其值
        if(void 0!==b[c])return b[c];
        var d;
        # 根据不同的c值，获取对应的扩展对象
        switch(c){
            case "OES_texture_float":d=a.getExtension("OES_texture_float");break;
            case "OES_texture_float_linear":d=a.getExtension("OES_texture_float_linear");break;
            case "OES_standard_derivatives":d=a.getExtension("OES_standard_derivatives");break;
            case "EXT_texture_filter_anisotropic":d=a.getExtension("EXT_texture_filter_anisotropic")||a.getExtension("MOZ_EXT_texture_filter_anisotropic")||a.getExtension("WEBKIT_EXT_texture_filter_anisotropic");break;
            case "WEBGL_compressed_texture_s3tc":d=a.getExtension("WEBGL_compressed_texture_s3tc")||a.getExtension("MOZ_WEBGL_compressed_texture_s3tc")||a.getExtension("WEBKIT_WEBGL_compressed_texture_s3tc");break;
            case "WEBGL_compressed_texture_pvrtc":d=a.getExtension("WEBGL_compressed_texture_pvrtc")||a.getExtension("WEBKIT_WEBGL_compressed_texture_pvrtc");break;
            case "OES_element_index_uint":d=a.getExtension("OES_element_index_uint");break;
            case "EXT_blend_minmax":d=a.getExtension("EXT_blend_minmax");break;
            case "EXT_frag_depth":d=a.getExtension("EXT_frag_depth")
        }
        # 如果d为null，则打印错误信息
        null===d&&console.log("THREE.WebGLRenderer: "+c+" extension not supported.");
        # 将获取到的扩展对象存储到b对象中
        return b[c]=d
    }
};
# 定义一个名为 THREE.WebGLProgram 的函数
THREE.WebGLProgram=function(){
    # 初始化变量 a 为 0
    var a=0;
    # 返回一个函数，该函数接受参数 b, c, d, e
    return function(b,c,d,e){
        # 获取上下文对象
        var f=b.context,
        # 获取定义
        g=d.defines,
        # 获取 uniform 变量
        h=d.__webglShader.uniforms,
        # 获取 attributes 变量
        k=d.attributes,
        # 获取顶点着色器代码
        n=d.__webglShader.vertexShader,
        # 获取片元着色器代码
        p=d.__webglShader.fragmentShader,
        # 获取索引属性的名称
        q=d.index0AttributeName;
        # 如果索引属性名称未定义且启用了变形目标，则设置索引属性名称为 "position"
        void 0===q&&!0===e.morphTargets&&(q="position");
        # 设置阴影贴图类型
        var m="SHADOWMAP_TYPE_BASIC";
        # 如果阴影贴图类型为 THREE.PCFShadowMap，则设置 m 为 "SHADOWMAP_TYPE_PCF"
        e.shadowMapType===THREE.PCFShadowMap?m="SHADOWMAP_TYPE_PCF":
        # 如果阴影贴图类型为 THREE.PCFSoftShadowMap，则设置 m 为 "SHADOWMAP_TYPE_PCF_SOFT"
        e.shadowMapType===THREE.PCFSoftShadowMap&&(m="SHADOWMAP_TYPE_PCF_SOFT");
        # 定义变量 r 和 t
        var r,t;
        # 遍历定义，生成对应的 #define 语句，并将其存储在数组 r 中
        r=[];
        for(var s in g)t=g[s],!1!==t&&(t="#define "+s+" "+t,r.push(t));
        # 将数组 r 中的内容用换行符连接成字符串
        r=r.join("\n");
        # 创建 WebGL 程序对象
        g=f.createProgram();
        # 如果材质对象是 THREE.RawShaderMaterial 类型，则将 b 和 d 设置为空字符串
        d instanceof THREE.RawShaderMaterial?b=d="":
        # 否则，根据材质属性和渲染器属性生成顶点着色器和片元着色器的代码
        (d=["precision "+e.precision+" float;","precision "+e.precision+" int;",r,e.supportsVertexTextures?"#define VERTEX_TEXTURES":"",b.gammaInput?"#define GAMMA_INPUT":"",b.gammaOutput?"#define GAMMA_OUTPUT":"","#define MAX_DIR_LIGHTS "+e.maxDirLights,"#define MAX_POINT_LIGHTS "+e.maxPointLights,"#define MAX_SPOT_LIGHTS "+e.maxSpotLights,"#define MAX_HEMI_LIGHTS "+e.maxHemiLights,"#define MAX_SHADOWS "+e.maxShadows,"#define MAX_BONES "+e.maxBones,e.map?"#define USE_MAP":"",e.envMap?"#define USE_ENVMAP":"",e.lightMap?"#define USE_LIGHTMAP":"",e.bumpMap?"#define USE_BUMPMAP":"",e.normalMap?"#define USE_NORMALMAP":"",e.specularMap?"#define USE_SPECULARMAP":"",e.alphaMap?"#define USE_ALPHAMAP":"",e.vertexColors?"#define USE_COLOR":"",e.skinning?"#define USE_SKINNING":"",e.useVertexTexture?"#define BONE_TEXTURE":"",e.morphTargets?"#define USE_MORPHTARGETS":"",e.morphNormals?"#define USE_MORPHNORMALS":"",e.wrapAround?"#define WRAP_AROUND":
// 定义空字符串
"", 
// 如果双面渲染，则定义 DOUBLE_SIDED
e.doubleSided?"#define DOUBLE_SIDED":"",
// 如果翻转面，则定义 FLIP_SIDED
e.flipSided?"#define FLIP_SIDED":"",
// 如果启用阴影映射，则定义 USE_SHADOWMAP
e.shadowMapEnabled?"#define USE_SHADOWMAP":"",
// 如果启用阴影映射，则定义对应的宏
e.shadowMapEnabled?"#define "+m:"",
// 如果启用阴影映射调试，则定义 SHADOWMAP_DEBUG
e.shadowMapDebug?"#define SHADOWMAP_DEBUG":"",
// 如果启用阴影映射级联，则定义 SHADOWMAP_CASCADE
e.shadowMapCascade?"#define SHADOWMAP_CASCADE":"",
// 如果启用大小衰减，则定义 USE_SIZEATTENUATION
e.sizeAttenuation?"#define USE_SIZEATTENUATION":"",
// 如果启用对数深度缓冲，则定义 USE_LOGDEPTHBUF
e.logarithmicDepthBuffer?"#define USE_LOGDEPTHBUF":"",
// 定义一系列 uniform 变换矩阵和属性
"uniform mat4 modelMatrix;\nuniform mat4 modelViewMatrix;\nuniform mat4 projectionMatrix;\nuniform mat4 viewMatrix;\nuniform mat3 normalMatrix;\nuniform vec3 cameraPosition;\nattribute vec3 position;\nattribute vec3 normal;\nattribute vec2 uv;\nattribute vec2 uv2;\n#ifdef USE_COLOR\n\tattribute vec3 color;\n#endif\n#ifdef USE_MORPHTARGETS\n\tattribute vec3 morphTarget0;\n\tattribute vec3 morphTarget1;\n\tattribute vec3 morphTarget2;\n\tattribute vec3 morphTarget3;\n\t#ifdef USE_MORPHNORMALS\n\t\tattribute vec3 morphNormal0;\n\t\tattribute vec3 morphNormal1;\n\t\tattribute vec3 morphNormal2;\n\t\tattribute vec3 morphNormal3;\n\t#else\n\t\tattribute vec3 morphTarget4;\n\t\tattribute vec3 morphTarget5;\n\t\tattribute vec3 morphTarget6;\n\t\tattribute vec3 morphTarget7;\n\t#endif\n#endif\n#ifdef USE_SKINNING\n\tattribute vec4 skinIndex;\n\tattribute vec4 skinWeight;\n#endif\n"].join("\n"),
// 定义一系列精度和宏定义
b=["precision "+e.precision+" float;","precision "+e.precision+" int;",e.bumpMap||e.normalMap?"#extension GL_OES_standard_derivatives : enable":"",r,"#define MAX_DIR_LIGHTS "+e.maxDirLights,"#define MAX_POINT_LIGHTS "+e.maxPointLights,"#define MAX_SPOT_LIGHTS "+e.maxSpotLights,"#define MAX_HEMI_LIGHTS "+e.maxHemiLights,"#define MAX_SHADOWS "+e.maxShadows,e.alphaTest?"#define ALPHATEST "+e.alphaTest:"",b.gammaInput?"#define GAMMA_INPUT":"",b.gammaOutput?"#define GAMMA_OUTPUT":"",e.useFog&&e.fog?"#define USE_FOG":
# 根据场景中的各种属性，生成对应的着色器代码
"",e.useFog&&e.fogExp?"#define FOG_EXP2":"",  # 如果使用雾效果并且是指数雾，则定义FOG_EXP2
e.map?"#define USE_MAP":"",  # 如果使用贴图，则定义USE_MAP
e.envMap?"#define USE_ENVMAP":"",  # 如果使用环境贴图，则定义USE_ENVMAP
e.lightMap?"#define USE_LIGHTMAP":"",  # 如果使用光照贴图，则定义USE_LIGHTMAP
e.bumpMap?"#define USE_BUMPMAP":"",  # 如果使用凹凸贴图，则定义USE_BUMPMAP
e.normalMap?"#define USE_NORMALMAP":"",  # 如果使用法线贴图，则定义USE_NORMALMAP
e.specularMap?"#define USE_SPECULARMAP":"",  # 如果使用高光贴图，则定义USE_SPECULARMAP
e.alphaMap?"#define USE_ALPHAMAP":"",  # 如果使用透明度贴图，则定义USE_ALPHAMAP
e.vertexColors?"#define USE_COLOR":"",  # 如果使用顶点颜色，则定义USE_COLOR
e.metal?"#define METAL":"",  # 如果是金属材质，则定义METAL
e.wrapAround?"#define WRAP_AROUND":"",  # 如果使用环绕贴图，则定义WRAP_AROUND
e.doubleSided?"#define DOUBLE_SIDED":"",  # 如果是双面材质，则定义DOUBLE_SIDED
e.flipSided?"#define FLIP_SIDED":"",  # 如果是翻转材质，则定义FLIP_SIDED
e.shadowMapEnabled?"#define USE_SHADOWMAP":"",  # 如果启用阴影贴图，则定义USE_SHADOWMAP
"",e.shadowMapEnabled?"#define "+m:"",  # 如果启用阴影贴图，则定义对应的阴影类型
e.shadowMapDebug?"#define SHADOWMAP_DEBUG":"",  # 如果启用阴影贴图调试，则定义SHADOWMAP_DEBUG
e.shadowMapCascade?"#define SHADOWMAP_CASCADE":"",  # 如果启用阴影贴图级联，则定义SHADOWMAP_CASCADE
e.logarithmicDepthBuffer?"#define USE_LOGDEPTHBUF":"",  # 如果使用对数深度缓冲，则定义USE_LOGDEPTHBUF
"uniform mat4 viewMatrix;\nuniform vec3 cameraPosition;\n"].join("\n"));  # 添加视图矩阵和相机位置的统一变量声明
n=new THREE.WebGLShader(f,f.VERTEX_SHADER,d+n);  # 创建顶点着色器对象
p=new THREE.WebGLShader(f,f.FRAGMENT_SHADER,b+p);  # 创建片段着色器对象
f.attachShader(g,n);  # 将顶点着色器附加到着色器程序对象上
f.attachShader(g,p);  # 将片段着色器附加到着色器程序对象上
void 0!==q&&f.bindAttribLocation(g,0,q);  # 如果存在顶点属性位置，则绑定顶点属性位置
f.linkProgram(g);  # 链接着色器程序
!1===f.getProgramParameter(g,f.LINK_STATUS)&&(console.error("THREE.WebGLProgram: Could not initialise shader."),  # 如果链接失败，则输出错误信息
console.error("gl.VALIDATE_STATUS",f.getProgramParameter(g,f.VALIDATE_STATUS)),  # 输出验证状态信息
console.error("gl.getError()",f.getError()));  # 输出错误信息
""!==f.getProgramInfoLog(g)&&console.warn("THREE.WebGLProgram: gl.getProgramInfoLog()",f.getProgramInfoLog(g));  # 如果存在着色器程序信息日志，则输出警告信息
f.deleteShader(n);  # 删除顶点着色器对象
f.deleteShader(p);  # 删除片段着色器对象
q="viewMatrix modelViewMatrix projectionMatrix normalMatrix modelMatrix cameraPosition morphTargetInfluences bindMatrix bindMatrixInverse".split(" ");  # 定义一系列统一变量
e.useVertexTexture?(q.push("boneTexture"),q.push("boneTextureWidth"),q.push("boneTextureHeight")):  # 如果使用顶点纹理，则添加对应的统一变量
# 将字符串"boneGlobalMatrices"添加到队列q中
q.push("boneGlobalMatrices")
# 如果启用了对数深度缓冲，则将字符串"logDepthBufFC"添加到队列q中
e.logarithmicDepthBuffer && q.push("logDepthBufFC")
# 遍历对象h的属性，并将属性名添加到队列q中
for(var u in h) q.push(u)
# 将队列h的值赋给变量q
h = q
# 创建空对象u
u = {}
# 遍历队列h，获取每个属性在对象f中的位置，并将其存储在对象u中
for (q = 0; q < b; q++) m = h[q], u[m] = f.getUniformLocation(g, m)
# 将字符串"position normal uv uv2 tangent color skinIndex skinWeight lineDistance"拆分为数组，并赋给变量q
q = "position normal uv uv2 tangent color skinIndex skinWeight lineDistance".split(" ")
# 将morphTarget和morphNormal的数量添加到数组q中
for (h = 0; h < e.maxMorphTargets; h++) q.push("morphTarget" + h)
for (h = 0; h < e.maxMorphNormals; h++) q.push("morphNormal" + h)
# 遍历对象k的属性，并将属性名添加到数组q中
for (var v in k) q.push(v)
# 将数组q中的属性名添加到对象k中，并获取它们在对象f中的位置
e = q
k = {}
for (v = 0; v < h; v++) u = e[v], k[u] = f.getAttribLocation(g, u)
# 将属性名数组赋给this.attributesKeys
this.attributesKeys = Object.keys(this.attributes)
# 设置id、code、usedTimes、program、vertexShader和fragmentShader的值，并返回this
this.id = a++
this.code = c
this.usedTimes = 1
this.program = g
this.vertexShader = n
this.fragmentShader = p
return this
}}();
# 创建函数a，用于格式化错误日志
THREE.WebGLShader = function() {
    var a = function(a) {
        a = a.split("\n");
        for (var c = 0; c < a.length; c++) a[c] = c + 1 + ": " + a[c];
        return a.join("\n")
    };
    return function(b, c, d) {
        c = b.createShader(c);
        b.shaderSource(c, d);
        b.compileShader(c);
        !1 === b.getShaderParameter(c, b.COMPILE_STATUS) && console.error("THREE.WebGLShader: Shader couldn't compile.");
        "" !== b.getShaderInfoLog(c) && (console.warn("THREE.WebGLShader: gl.getShaderInfoLog()", b.getShaderInfoLog(c)), console.warn(a(d)));
        return c
    }
}();
# 创建LensFlarePlugin函数，用于渲染镜头光晕效果
THREE.LensFlarePlugin = function(a, b) {
    var c, d, e, f, g, h, k, n, p, q, m = a.context, r, t, s, u, v, y;
    this.render = function(G, w, K, x) {
        if (0 !== b.length) {
            G = new THREE.Vector3;
            var D = x / K,
                E = .5 * K,
                A = .5 * x,
                B = 16 / x,
                F = new THREE.Vector2(B * D, B),
                R = new THREE.Vector3(1, 1, 0),
                H = new THREE.Vector2(1, 1);
            if (void 0 === s) {
                var B = new Float32Array([-1, -1, 0, 0, 1, -1, 1, 0, 1, 1, 1, 1, -1, 1, 0, 1]),
                    C = new Uint16Array([0, 1, 2, 0, 2, 3]);
                r = m.createBuffer();
                t = m.createBuffer();
                m.bindBuffer(m.ARRAY_BUFFER, r);
                m.bufferData(m.ARRAY_BUFFER, B, m.STATIC_DRAW);
                m.bindBuffer(m.ELEMENT_ARRAY_BUFFER,
# 创建一个 WebGL 缓冲区对象，并将数据传递给它
m.bufferData(m.ELEMENT_ARRAY_BUFFER, C, m.STATIC_DRAW);
# 创建两个纹理对象
v = m.createTexture();
y = m.createTexture();
# 绑定纹理对象并设置纹理图像的属性
m.bindTexture(m.TEXTURE_2D, v);
m.texImage2D(m.TEXTURE_2D, 0, m.RGB, 16, 16, 0, m.RGB, m.UNSIGNED_BYTE, null);
m.texParameteri(m.TEXTURE_2D, m.TEXTURE_WRAP_S, m.CLAMP_TO_EDGE);
m.texParameteri(m.TEXTURE_2D, m.TEXTURE_WRAP_T, m.CLAMP_TO_EDGE);
m.texParameteri(m.TEXTURE_2D, m.TEXTURE_MAG_FILTER, m.NEAREST);
m.texParameteri(m.TEXTURE_2D, m.TEXTURE_MIN_FILTER, m.NEAREST);
# 绑定另一个纹理对象并设置纹理图像的属性
m.bindTexture(m.TEXTURE_2D, y);
m.texImage2D(m.TEXTURE_2D, 0,
# 设置纹理对象的参数，RGBA 表示颜色通道，16 表示每个像素的位数，16 表示每行的像素数，0 表示边界，m.RGBA 表示颜色通道，m.UNSIGNED_BYTE 表示数据类型，null 表示数据源为空
m.texImage2D(m.TEXTURE_2D, 0, m.RGBA, 16, 16, 0, m.RGBA, m.UNSIGNED_BYTE, null);
# 设置纹理对象在 S 轴上的环绕方式为 CLAMP_TO_EDGE
m.texParameteri(m.TEXTURE_2D, m.TEXTURE_WRAP_S, m.CLAMP_TO_EDGE);
# 设置纹理对象在 T 轴上的环绕方式为 CLAMP_TO_EDGE
m.texParameteri(m.TEXTURE_2D, m.TEXTURE_WRAP_T, m.CLAMP_TO_EDGE);
# 设置纹理对象的放大过滤方式为 NEAREST
m.texParameteri(m.TEXTURE_2D, m.TEXTURE_MAG_FILTER, m.NEAREST);
# 设置纹理对象的缩小过滤方式为 NEAREST
m.texParameteri(m.TEXTURE_2D, m.TEXTURE_MIN_FILTER, m.NEAREST);
# 定义变量 B，并根据条件判断是否支持顶点着色器中使用纹理
B = (u = 0 < m.getParameter(m.MAX_VERTEX_TEXTURE_IMAGE_UNITS)) ? {
    # 顶点着色器代码
    vertexShader: "uniform lowp int renderType;\nuniform vec3 screenPosition;\nuniform vec2 scale;\nuniform float rotation;\nuniform sampler2D occlusionMap;\nattribute vec2 position;\nattribute vec2 uv;\nvarying vec2 vUV;\nvarying float vVisibility;\nvoid main() {\nvUV = uv;\nvec2 pos = position;\nif( renderType == 2 ) {\nvec4 visibility = texture2D( occlusionMap, vec2( 0.1, 0.1 ) );\nvisibility += texture2D( occlusionMap, vec2( 0.5, 0.1 ) );\nvisibility += texture2D( occlusionMap, vec2( 0.9, 0.1 ) );\nvisibility += texture2D( occlusionMap, vec2( 0.9, 0.5 ) );\nvisibility += texture2D( occlusionMap, vec2( 0.9, 0.9 ) );\nvisibility += texture2D( occlusionMap, vec2( 0.5, 0.9 ) );\nvisibility += texture2D( occlusionMap, vec2( 0.1, 0.9 ) );\nvisibility += texture2D( occlusionMap, vec2( 0.1, 0.5 ) );\nvisibility += texture2D( occlusionMap, vec2( 0.5, 0.5 ) );\nvVisibility = visibility.r / 9.0;\nvVisibility *= 1.0 - visibility.g / 9.0;\nvVisibility *= visibility.b / 9.0;\nvVisibility *= 1.0 - visibility.a / 9.0;\npos.x = cos( rotation ) * position.x - sin( rotation ) * position.y;\npos.y = sin( rotation ) * position.x + cos( rotation ) * position.y;\n}\ngl_Position = vec4( ( pos * scale + screenPosition.xy ).xy, screenPosition.z, 1.0 );\n}",
# 定义片段着色器，根据不同的渲染类型渲染不同的效果
fragmentShader:"uniform lowp int renderType;\nuniform sampler2D map;\nuniform float opacity;\nuniform vec3 color;\nvarying vec2 vUV;\nvarying float vVisibility;\nvoid main() {
# 如果渲染类型为0，设置颜色为紫色
if( renderType == 0 ) {
gl_FragColor = vec4( 1.0, 0.0, 1.0, 0.0 );
# 如果渲染类型为1，使用纹理贴图进行渲染
} else if( renderType == 1 ) {
gl_FragColor = texture2D( map, vUV );
# 如果渲染类型不为0或1，根据不透明度和可见性调整纹理颜色
} else {
vec4 texture = texture2D( map, vUV );
texture.a *= opacity * vVisibility;
gl_FragColor = texture;
gl_FragColor.rgb *= color;
}
}"
# 定义顶点着色器，根据不同的渲染类型进行位置计算和屏幕定位
vertexShader:"uniform lowp int renderType;\nuniform vec3 screenPosition;\nuniform vec2 scale;\nuniform float rotation;\nattribute vec2 position;\nattribute vec2 uv;\nvarying vec2 vUV;\nvoid main() {
vUV = uv;
vec2 pos = position;
# 如果渲染类型为2，根据旋转角度调整位置
if( renderType == 2 ) {
pos.x = cos( rotation ) * position.x - sin( rotation ) * position.y;
pos.y = sin( rotation ) * position.x + cos( rotation ) * position.y;
}
gl_Position = vec4( ( pos * scale + screenPosition.xy ).xy, screenPosition.z, 1.0 );
}"
# 定义片段着色器，根据不同的渲染类型渲染不同的效果
fragmentShader:"precision mediump float;\nuniform lowp int renderType;\nuniform sampler2D map;\nuniform sampler2D occlusionMap;\nuniform float opacity;\nuniform vec3 color;\nvarying vec2 vUV;\nvoid main() {
# 如果渲染类型为0，设置颜色为纹理贴图的RGB值，透明度为0
if( renderType == 0 ) {
gl_FragColor = vec4( texture2D( map, vUV ).rgb, 0.0 );
# 如果渲染类型为1，使用纹理贴图进行渲染
} else if( renderType == 1 ) {
gl_FragColor = texture2D( map, vUV );
# 如果渲染类型不为0或1，根据遮挡贴图计算可见性，并根据可见性和不透明度调整纹理颜色
} else {
float visibility = texture2D( occlusionMap, vec2( 0.5, 0.1 ) ).a;
visibility += texture2D( occlusionMap, vec2( 0.9, 0.5 ) ).a;
visibility += texture2D( occlusionMap, vec2( 0.5, 0.9 ) ).a;
visibility += texture2D( occlusionMap, vec2( 0.1, 0.5 ) ).a;
visibility = ( 1.0 - visibility / 4.0 );
vec4 texture = texture2D( map, vUV );
texture.a *= opacity * visibility;
gl_FragColor = texture;
gl_FragColor.rgb *= color;
}
}"
# 创建程序对象
C=m.createProgram()
# 创建片元着色器对象
T=m.createShader(m.FRAGMENT_SHADER)
# 创建顶点着色器对象
Q=m.createShader(m.VERTEX_SHADER)
# 设置精度
O="precision "+a.getPrecision()+" float;\n"
# 将片元着色器源码附加到片元着色器对象上
m.shaderSource(T,O+B.fragmentShader)
# 将顶点着色器源码附加到顶点着色器对象上
m.shaderSource(Q,O+B.vertexShader)
# 编译片元着色器
m.compileShader(T)
# 编译顶点着色器
m.compileShader(Q)
# 将片元着色器对象和顶点着色器对象附加到程序对象上
m.attachShader(C,T)
m.attachShader(C,Q)
# 链接程序对象
m.linkProgram(C)
# 设置当前程序对象
s=C
# 获取属性位置
p=m.getAttribLocation(s,"position")
q=m.getAttribLocation(s,"uv")
# 获取统一变量位置
c=m.getUniformLocation(s,"renderType")
d=m.getUniformLocation(s,"map")
e=m.getUniformLocation(s,"occlusionMap")
f=m.getUniformLocation(s,"opacity")
g=m.getUniformLocation(s,"color")
h=m.getUniformLocation(s,"scale")
k=m.getUniformLocation(s,"rotation")
n=m.getUniformLocation(s,"screenPosition")
# 使用程序对象
m.useProgram(s)
# 启用顶点属性数组
m.enableVertexAttribArray(p)
m.enableVertexAttribArray(q)
# 设置统一变量
m.uniform1i(e,0)
m.uniform1i(d,1)
# 绑定缓冲区
m.bindBuffer(m.ARRAY_BUFFER,r)
# 指定顶点属性数据
m.vertexAttribPointer(p,2,m.FLOAT,!1,16,0)
m.vertexAttribPointer(q,2,m.FLOAT,!1,16,8)
# 绑定缓冲区
m.bindBuffer(m.ELEMENT_ARRAY_BUFFER,t)
# 禁用面剔除
m.disable(m.CULL_FACE)
# 设置深度写入
m.depthMask(!1)
# 循环处理
C=0
for T=b.length;C<T;C++
    # 计算一些值
    if B=16/x,F.set(B*D,B),Q=b[C],G.set(Q.matrixWorld.elements[12],Q.matrixWorld.elements[13],Q.matrixWorld.elements[14]),G.applyMatrix4(w.matrixWorldInverse),G.applyProjection(w.projectionMatrix),R.copy(G),H.x=R.x*E+E,H.y=R.y*A+A,u||0<H.x&&H.x<K&&0<H.y&&H.y<x
        # 设置纹理单元
        m.activeTexture(m.TEXTURE1)
        # 绑定纹理
        m.bindTexture(m.TEXTURE_2D,v)
        # 复制屏幕上的一部分像素到纹理
        m.copyTexImage2D(m.TEXTURE_2D,0,m.RGB,H.x-8,H.y-8,16,16,0)
        # 设置统一变量
        m.uniform1i(c,0)
        m.uniform2f(h,F.x,F.y)
        m.uniform3f(n,R.x,R.y,R.z)
        # 禁用混合
        m.disable(m.BLEND)
        # 启用深度测试
        m.enable(m.DEPTH_TEST)
        # 绘制三角形
        m.drawElements(m.TRIANGLES,6,m.UNSIGNED_SHORT,
# 设置活动纹理单元为纹理单元0
m.activeTexture(m.TEXTURE0);
# 绑定纹理到纹理单元2D
m.bindTexture(m.TEXTURE_2D,y);
# 复制纹理图像到当前绑定的纹理对象中
m.copyTexImage2D(m.TEXTURE_2D,0,m.RGBA,H.x-8,H.y-8,16,16,0);
# 设置uniform变量c的值为1
m.uniform1i(c,1);
# 禁用深度测试
m.disable(m.DEPTH_TEST);
# 设置活动纹理单元为纹理单元1
m.activeTexture(m.TEXTURE1);
# 绑定纹理到纹理单元2D
m.bindTexture(m.TEXTURE_2D,v);
# 绘制元素
m.drawElements(m.TRIANGLES,6,m.UNSIGNED_SHORT,0);
# 复制屏幕位置
Q.positionScreen.copy(R);
# 如果存在自定义更新回调函数，则调用该函数，否则调用updateLensFlares函数
Q.customUpdateCallback?Q.customUpdateCallback(Q):Q.updateLensFlares();
# 设置uniform变量c的值为2
m.uniform1i(c,2);
# 启用混合
m.enable(m.BLEND);
# 遍历镜头光晕数组
for(var O=0,S=Q.lensFlares.length;O<S;O++){
    var X=Q.lensFlares[O];
    // 如果透明度和尺寸都大于0.001，则进行以下操作
    .001<X.opacity&&.001<X.scale&&(R.x=X.x,R.y=X.y,R.z=X.z,B=X.size*X.scale/x,F.x=B*D,F.y=B,m.uniform3f(n,R.x,R.y,R.z),m.uniform2f(h,F.x,F.y),m.uniform1f(k,X.rotation),m.uniform1f(f,X.opacity),m.uniform3f(g,X.color.r,X.color.g,X.color.b),a.setBlending(X.blending,X.blendEquation,X.blendSrc,X.blendDst),a.setTexture(X.texture,1),m.drawElements(m.TRIANGLES,6,m.UNSIGNED_SHORT,0))
}
# 启用面剔除
m.enable(m.CULL_FACE);
# 启用深度测试
m.enable(m.DEPTH_TEST);
# 设置深度写入掩码为真
m.depthMask(!0);
# 重置WebGL状态
a.resetGLState()
# 设置阴影通道
!0;n._shadowPass=!0;
# 渲染函数
this.render=function(c,v){
    # 如果阴影图启用
    if(!1!==a.shadowMapEnabled){
        var u,K,x,D,E,A,B,F,R=[];
        D=0;
        f.clearColor(1,1,1,1);
        f.disable(f.BLEND);
        f.enable(f.CULL_FACE);
        f.frontFace(f.CCW);
        # 设置剔除面
        a.shadowMapCullFace===THREE.CullFaceFront?f.cullFace(f.FRONT):f.cullFace(f.BACK);
        a.setDepthTest(!0);
        u=0;
        # 遍历场景中的物体
        for(K=b.length;u<K;u++)
            if(x=b[u],x.castShadow)
                # 如果是方向光并且有阴影级联
                if(x instanceof THREE.DirectionalLight&&x.shadowCascade)
                    for(E=0;E<x.shadowCascadeCount;E++){
                        var H;
                        # 如果阴影级联数组中已经有值，则使用该值，否则创建虚拟光源
                        if(x.shadowCascadeArray[E])
                            H=x.shadowCascadeArray[E];
                        else{
                            B=x;
                            var C=E;
                            H=new THREE.DirectionalLight;
                            H.isVirtual=!0;
                            H.onlyShadow=!0;
                            H.castShadow=!0;
                            H.shadowCameraNear=B.shadowCameraNear;
                            H.shadowCameraFar=B.shadowCameraFar;
                            H.shadowCameraLeft=B.shadowCameraLeft;
                            H.shadowCameraRight=B.shadowCameraRight;
                            H.shadowCameraBottom=B.shadowCameraBottom;
                            H.shadowCameraTop=B.shadowCameraTop;
                            H.shadowCameraVisible=B.shadowCameraVisible;
                            H.shadowDarkness=B.shadowDarkness;
                            H.shadowBias=B.shadowCascadeBias[C];
                            H.shadowMapWidth=B.shadowCascadeWidth[C];
                            H.shadowMapHeight=B.shadowCascadeHeight[C];
                            H.pointsWorld=[];
                            H.pointsFrustum=[];
                            F=H.pointsWorld;
                            A=H.pointsFrustum;
                            for(var T=0;8>T;T++)
                                F[T]=new THREE.Vector3,
                                A[T]=new THREE.Vector3;
                            F=B.shadowCascadeNearZ[C];
                            B=B.shadowCascadeFarZ[C];
                            A[0].set(-1,-1,F);
                            A[1].set(1,-1,F);
                            A[2].set(-1,1,F);
                            A[3].set(1,1,F);
                            A[4].set(-1,-1,B);
                            A[5].set(1,-1,B);
                            A[6].set(-1,1,B);
                            A[7].set(1,1,B);
                            H.originalCamera=v;
                            A=new THREE.Gyroscope;
                            A.position.copy(x.shadowCascadeOffset);
                            A.add(H);
                            A.add(H.target);
                            v.add(A);
                            x.shadowCascadeArray[E]=H;
                            console.log("Created virtualLight",H)
                        }
                        C=
# 循环遍历阴影级联数组
x;F=E;B=C.shadowCascadeArray[F];B.position.copy(C.position);B.target.position.copy(C.target.position);B.lookAt(B.target);B.shadowCameraVisible=C.shadowCameraVisible;B.shadowDarkness=C.shadowDarkness;B.shadowBias=C.shadowCascadeBias[F];A=C.shadowCascadeNearZ[F];C=C.shadowCascadeFarZ[F];B=B.pointsFrustum;B[0].z=A;B[1].z=A;B[2].z=A;B[3].z=A;B[4].z=C;B[5].z=C;B[6].z=C;B[7].z=C;R[D]=H;D++
# 如果不是阴影级联数组的最后一个元素，则将当前阴影级联的位置和目标位置复制给B，然后设置B的朝向为目标位置，设置阴影相机的可见性、阴影强度和阴影偏移，设置阴影级联的近平面和远平面，设置B的八个顶点的z坐标，将H赋值给R[D]，并将D加1
}else R[D]=x,D++;u=0;for(K=R.length;u<K;u++){x=R[u];x.shadowMap||(E=THREE.LinearFilter,a.shadowMapType===THREE.PCFSoftShadowMap&&
(E=THREE.NearestFilter),x.shadowMap=new THREE.WebGLRenderTarget(x.shadowMapWidth,x.shadowMapHeight,{minFilter:E,magFilter:E,format:THREE.RGBAFormat}),x.shadowMapSize=new THREE.Vector2(x.shadowMapWidth,x.shadowMapHeight),x.shadowMatrix=new THREE.Matrix4);
# 如果不是阴影级联数组的最后一个元素，则将x赋值给R[D]，并将D加1，然后循环遍历R数组
if(!x.shadowCamera){
# 如果x没有阴影相机
if(x instanceof THREE.SpotLight)x.shadowCamera=new THREE.PerspectiveCamera(x.shadowCameraFov,x.shadowMapWidth/x.shadowMapHeight,x.shadowCameraNear,x.shadowCameraFar);
# 如果x是聚光灯，则创建透视相机
else if(x instanceof THREE.DirectionalLight)x.shadowCamera=new THREE.OrthographicCamera(x.shadowCameraLeft,x.shadowCameraRight,x.shadowCameraTop,x.shadowCameraBottom,x.shadowCameraNear,x.shadowCameraFar);
# 如果x是平行光，则创建正交相机
else{console.error("Unsupported light type for shadow");continue}
# 否则，输出错误信息并继续下一次循环
c.add(x.shadowCamera);!0===c.autoUpdate&&c.updateMatrixWorld()}x.shadowCameraVisible&&!x.cameraHelper&&(x.cameraHelper=new THREE.CameraHelper(x.shadowCamera),c.add(x.cameraHelper));
# 将阴影相机添加到场景中，如果场景的自动更新为真，则更新场景的世界矩阵，如果阴影相机可见且没有相机辅助对象，则创建相机辅助对象并添加到场景中
if(x.isVirtual&&H.originalCamera==v){E=v;D=x.shadowCamera;A=x.pointsFrustum;B=x.pointsWorld;m.set(Infinity,Infinity,Infinity);r.set(-Infinity,-Infinity,-Infinity);
# 如果x是虚拟的且H的原始相机等于v，则将v赋值给E，将x的阴影相机赋值给D，将x的视锥体顶点赋值给A，将x的世界顶点赋值给B，将无穷大赋值给m，将负无穷大赋值给r
# 循环8次，对B数组中的元素进行操作
for(C=0;8>C;C++)
    # 将B数组中的元素复制到A数组中对应位置
    F=B[C]
    F.copy(A[C])
    # 将F的坐标转换为未投影的坐标
    F.unproject(E)
    # 将F的坐标应用矩阵D的世界逆矩阵
    F.applyMatrix4(D.matrixWorldInverse)
    # 更新最小和最大坐标值
    F.x<m.x&&(m.x=F.x)
    F.x>r.x&&(r.x=F.x)
    F.y<m.y&&(m.y=F.y)
    F.y>r.y&&(r.y=F.y)
    F.z<m.z&&(m.z=F.z)
    F.z>r.z&&(r.z=F.z)
# 设置相机的左右上下边界
D.left=m.x
D.right=r.x
D.top=r.y
D.bottom=m.y
# 更新投影矩阵
D.updateProjectionMatrix()
# 获取阴影映射相关信息
D=x.shadowMap
A=x.shadowMatrix
E=x.shadowCamera
# 设置阴影相机的位置
E.position.setFromMatrixPosition(x.matrixWorld)
t.setFromMatrixPosition(x.target.matrixWorld)
E.lookAt(t)
E.updateMatrixWorld()
E.matrixWorldInverse.getInverse(E.matrixWorld)
# 如果存在相机辅助对象，则设置其可见性
x.cameraHelper&&(x.cameraHelper.visible=x.shadowCameraVisible)
x.shadowCameraVisible&&x.cameraHelper.update()
# 设置阴影矩阵
A.set(.5,0,0,.5,0,.5,0,.5,0,0,.5,.5,0,0,0,1)
A.multiply(E.projectionMatrix)
A.multiply(E.matrixWorldInverse)
q.multiplyMatrices(E.projectionMatrix,E.matrixWorldInverse)
p.setFromMatrix(q)
# 设置渲染目标为D
a.setRenderTarget(D)
# 清除渲染目标
a.clear()
# 清空数组s
s.length=0
# 对每个元素进行渲染
e(c,c,E)
x=0
# 渲染每个对象
for(D=s.length;x<D;x++)
    B=s[x]
    A=B.object
    B=B.buffer
    C=A.material instanceof THREE.MeshFaceMaterial?A.material.materials[0]:A.material
    F=void 0!==A.geometry.morphTargets&&0<A.geometry.morphTargets.length&&C.morphTargets
    T=A instanceof THREE.SkinnedMesh&&C.skinning
    F=A.customDepthMaterial?A.customDepthMaterial:T?F?n:k:F?h:g
    a.setMaterialFaces(C)
    B instanceof THREE.BufferGeometry?a.renderBufferDirect(E,b,null,F,B,A):a.renderBuffer(E,b,null,F,B,A)
x=0
# 渲染每个对象
for(D=d.length;x<D;x++)
    B=d[x]
    A=B.object
    A.visible&&A.castShadow&&(A._modelViewMatrix.multiplyMatrices(E.matrixWorldInverse,A.matrixWorld),a.renderImmediateObject(E,b,null,g,A))
# 获取当前的清除颜色和透明度
u=a.getClearColor()
K=a.getClearAlpha()
f.clearColor(u.r,u.g,u.b,K)
f.enable(f.BLEND)
a.shadowMapCullFace===THREE.CullFaceFront&&f.cullFace(f.BACK)
a.resetGLState()
THREE.SpritePlugin=function(a,b){
    var c,d,e,f,g,h,k,n,p,q,m,r,t,s,u,v,y;
    function G(a,b){
        return a.z!==b.z?b.z-a.z:b.id-a.id;
    }
    var w=a.context,K,x,D,E;
    this.render=function(A,B){
        if(0!==b.length){
            if(void 0===D){
                var F=new Float32Array([-.5,-.5,0,0,.5,-.5,1,0,.5,.5,1,1,-.5,.5,0,1]),
                    R=new Uint16Array([0,1,2,0,2,3]);
                K=w.createBuffer();
                x=w.createBuffer();
                w.bindBuffer(w.ARRAY_BUFFER,K);
                w.bufferData(w.ARRAY_BUFFER,F,w.STATIC_DRAW);
                w.bindBuffer(w.ELEMENT_ARRAY_BUFFER,x);
                w.bufferData(w.ELEMENT_ARRAY_BUFFER,R,w.STATIC_DRAW);
                var F=w.createProgram(),
                    R=w.createShader(w.VERTEX_SHADER),
                    H=w.createShader(w.FRAGMENT_SHADER);
                w.shaderSource(R,[
                    "precision "+a.getPrecision()+" float;",
                    "uniform mat4 modelViewMatrix;\nuniform mat4 projectionMatrix;\nuniform float rotation;\nuniform vec2 scale;\nuniform vec2 uvOffset;\nuniform vec2 uvScale;\nattribute vec2 position;\nattribute vec2 uv;\nvarying vec2 vUV;\nvoid main() {",
                    "vUV = uvOffset + uv * uvScale;",
                    "vec2 alignedPosition = position * scale;",
                    "vec2 rotatedPosition;",
                    "rotatedPosition.x = cos( rotation ) * alignedPosition.x - sin( rotation ) * alignedPosition.y;",
                    "rotatedPosition.y = sin( rotation ) * alignedPosition.x + cos( rotation ) * alignedPosition.y;",
                    "vec4 finalPosition;",
                    "finalPosition = modelViewMatrix * vec4( 0.0, 0.0, 0.0, 1.0 );",
                    "finalPosition.xy += rotatedPosition;",
                    "finalPosition = projectionMatrix * finalPosition;",
                    "gl_Position = finalPosition;",
                    "}"].join("\n"));
# 设置着色器源码，包括精度、颜色、纹理、透明度、雾效等参数
w.shaderSource(H,["precision "+a.getPrecision()+" float;","uniform vec3 color;\nuniform sampler2D map;\nuniform float opacity;\nuniform int fogType;\nuniform vec3 fogColor;\nuniform float fogDensity;\nuniform float fogNear;\nuniform float fogFar;\nuniform float alphaTest;\nvarying vec2 vUV;\nvoid main() {\nvec4 texture = texture2D( map, vUV );\nif ( texture.a < alphaTest ) discard;\ngl_FragColor = vec4( color * texture.xyz, texture.a * opacity );\nif ( fogType > 0 ) {\nfloat depth = gl_FragCoord.z / gl_FragCoord.w;\nfloat fogFactor = 0.0;\nif ( fogType == 1 ) {\nfogFactor = smoothstep( fogNear, fogFar, depth );\n} else {\nconst float LOG2 = 1.442695;\nfloat fogFactor = exp2( - fogDensity * fogDensity * depth * depth * LOG2 );\nfogFactor = 1.0 - clamp( fogFactor, 0.0, 1.0 );\n}\ngl_FragColor = mix( gl_FragColor, vec4( fogColor, gl_FragColor.w ), fogFactor );\n}\n}"].join("\n"));
# 编译着色器
w.compileShader(R);
w.compileShader(H);
# 将着色器附加到程序对象上
w.attachShader(F,R);
w.attachShader(F,H);
# 链接程序
w.linkProgram(F);
D=F;
# 获取属性位置
v=w.getAttribLocation(D,"position");
y=w.getAttribLocation(D,"uv");
# 获取uniform变量位置
c=w.getUniformLocation(D,"uvOffset");
d=w.getUniformLocation(D,"uvScale");
e=w.getUniformLocation(D,"rotation");
f=w.getUniformLocation(D,"scale");
g=w.getUniformLocation(D,"color");
h=w.getUniformLocation(D,"map");
k=w.getUniformLocation(D,"opacity");
n=w.getUniformLocation(D,"modelViewMatrix");
p=w.getUniformLocation(D,"projectionMatrix");
q=w.getUniformLocation(D,"fogType");
m=w.getUniformLocation(D,"fogDensity");
r=w.getUniformLocation(D,"fogNear");
t=w.getUniformLocation(D,"fogFar");
s=w.getUniformLocation(D,"fogColor");
u=w.getUniformLocation(D,"alphaTest");
# 创建画布和上下文
F=document.createElement("canvas");
F.width=8;
F.height=8;
R=F.getContext("2d");
R.fillStyle="white";
R.fillRect(0,0,8,8);
# 创建纹理对象
E=new THREE.Texture(F);
E.needsUpdate=!0
# 使用程序对象
w.useProgram(D);
w.enableVertexAttribArray(v);
w.enableVertexAttribArray(y);
w.disable(w.CULL_FACE);
w.enable(w.BLEND);
w.bindBuffer(w.ARRAY_BUFFER,
# 设置顶点属性指针
K);w.vertexAttribPointer(v,2,w.FLOAT,!1,16,0);
w.vertexAttribPointer(y,2,w.FLOAT,!1,16,8);
w.bindBuffer(w.ELEMENT_ARRAY_BUFFER,x);
w.uniformMatrix4fv(p,!1,B.projectionMatrix.elements);
w.activeTexture(w.TEXTURE0);
w.uniform1i(h,0);
R=F=0;
# 判断是否存在雾效果，若存在则设置相关参数
(H=A.fog)?(w.uniform3f(s,H.color.r,H.color.g,H.color.b),H instanceof THREE.Fog?(w.uniform1f(r,H.near),w.uniform1f(t,H.far),w.uniform1i(q,1),R=F=1):H instanceof THREE.FogExp2&&(w.uniform1f(m,H.density),w.uniform1i(q,2),R=F=2)):(w.uniform1i(q,0),R=F=0);
# 计算每个物体的深度，并排序
for(var H=0,C=b.length;H<C;H++){
    var T=b[H];
    T._modelViewMatrix.multiplyMatrices(B.matrixWorldInverse,T.matrixWorld);
    T.z=null===T.renderDepth?-T._modelViewMatrix.elements[14]:T.renderDepth
}
b.sort(G);
# 设置材质相关参数并绘制物体
for(var Q=[],H=0,C=b.length;H<C;H++){
    var T=b[H],O=T.material;
    w.uniform1f(u,O.alphaTest);
    w.uniformMatrix4fv(n,!1,T._modelViewMatrix.elements);
    Q[0]=T.scale.x;
    Q[1]=T.scale.y;
    T=0;
    A.fog&&O.fog&&(T=R);
    F!==T&&(w.uniform1i(q,T),F=T);
    null!==O.map?(w.uniform2f(c,O.map.offset.x,O.map.offset.y),w.uniform2f(d,O.map.repeat.x,O.map.repeat.y)):
    (w.uniform2f(c,0,0),w.uniform2f(d,1,1));
    w.uniform1f(k,O.opacity);
    w.uniform3f(g,O.color.r,O.color.g,O.color.b);
    w.uniform1f(e,O.rotation);
    w.uniform2fv(f,Q);
    a.setBlending(O.blending,O.blendEquation,O.blendSrc,O.blendDst);
    a.setDepthTest(O.depthTest);
    a.setDepthWrite(O.depthWrite);
    O.map&&O.map.image&&O.map.image.width?a.setTexture(O.map,0):a.setTexture(E,0);
    w.drawElements(w.TRIANGLES,6,w.UNSIGNED_SHORT,0)
}
w.enable(w.CULL_FACE);
a.resetGLState()
}}};
# 合并几何体
THREE.GeometryUtils={merge:function(a,b,c){
    console.warn("THREE.GeometryUtils: .merge() has been moved to Geometry. Use geometry.merge( geometry2, matrix, materialIndexOffset ) instead");
    var d;
    b instanceof THREE.Mesh&&(b.matrixAutoUpdate&&b.updateMatrix(),d=b.matrix,b=b.geometry);
    a.merge(b,d,c)
},
# 将几何体居中
center:function(a){
    console.warn("THREE.GeometryUtils: .center() has been moved to Geometry. Use geometry.center() instead");
    return a.center()
}};
THREE.ImageUtils={
    // 设置跨域属性为 undefined
    crossOrigin:void 0,
    // 加载纹理图片
    loadTexture:function(a,b,c,d){
        // 创建图片加载器
        var e=new THREE.ImageLoader;
        // 设置图片加载器的跨域属性
        e.crossOrigin=this.crossOrigin;
        // 创建纹理对象
        var f=new THREE.Texture(void 0,b);
        // 加载图片
        e.load(a,function(a){
            // 将加载的图片赋值给纹理对象
            f.image=a;
            // 设置需要更新标志为 true
            f.needsUpdate=!0;
            // 如果回调函数存在，则执行回调函数
            c&&c(f)
        },void 0,function(a){
            // 如果错误回调函数存在，则执行错误回调函数
            d&&d(a)
        });
        // 设置纹理对象的来源文件
        f.sourceFile=a;
        // 返回纹理对象
        return f
    },
    // 加载立方体纹理图片
    loadTextureCube:function(a,b,c,d){
        // 创建图片加载器
        var e=new THREE.ImageLoader;
        // 设置图片加载器的跨域属性
        e.crossOrigin=this.crossOrigin;
        // 创建立方体纹理对象
        var f=new THREE.CubeTexture([],b);
        // 设置立方体纹理对象的 flipY 属性为 false
        f.flipY=!1;
        var g=0;
        // 定义加载图片的函数
        b=function(b){
            // 加载图片
            e.load(a[b],function(a){
                // 将加载的图片赋值给立方体纹理对象的 images 数组
                f.images[b]=a;
                // 计数加一
                g+=1;
                // 如果计数为 6，则设置需要更新标志为 true，并执行回调函数
                6===g&&(f.needsUpdate=!0,c&&c(f))
            })
        };
        var h=a.length;
        // 遍历加载图片
        for(var d=0;d<h;++d)b(d);
        // 返回立方体纹理对象
        return f
    },
    // 加载压缩纹理图片的函数，已被移除
    loadCompressedTexture:function(){
        console.error("THREE.ImageUtils.loadCompressedTexture has been removed. Use THREE.DDSLoader instead.")
    },
    // 加载压缩立方体纹理图片的函数，已被移除
    loadCompressedTextureCube:function(){
        console.error("THREE.ImageUtils.loadCompressedTextureCube has been removed. Use THREE.DDSLoader instead.")
    },
    // 获取法线贴图
    getNormalMap:function(a,b){
        // 定义计算单位向量的函数
        var c=function(a){
            var b=Math.sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);
            return[a[0]/b,a[1]/b,a[2]/b]
        };
        // 如果 b 不存在，则设置为 1
        b|=1;
        var d=a.width,e=a.height,f=document.createElement("canvas");
        f.width=d;
        f.height=e;
        var g=f.getContext("2d");
        g.drawImage(a,0,0);
        for(var h=g.getImageData(0,0,d,e).data,k=g.createImageData(d,e),n=k.data,p=0;p<d;p++){
            for(var q=0;q<e;q++){
                var m=0>q-1?0:q-1,r=q+1>e-1?e-1:q+1,t=0>p-1?0:p-1,s=p+1>d-1?d-1:p+1,u=[],v=[0,0,h[4*(q*d+p)]/255*b];
                u.push([-1,0,h[4*(q*d+t)]/255*b]);
                u.push([-1,-1,h[4*(m*d+t)]/255*b]);
                u.push([0,-1,h[4*(m*d+p)]/255*b]);
                u.push([1,-1,h[4*(m*d+s)]/255*b]);
                u.push([1,0,h[4*(q*d+s)]/255*b]);
                u.push([1,1,h[4*(r*d+s)]/255*b]);
                u.push([0,1,h[4*(r*d+p)]/255*
# 定义一个函数，参数为 a, b, c
THREE.FontUtils = {
    faces: {},  # 创建一个空对象 faces
    face: "helvetiker",  # 设置 face 属性为 "helvetiker"
    weight: "normal",  # 设置 weight 属性为 "normal"
    style: "normal",  # 设置 style 属性为 "normal"
    size: 150,  # 设置 size 属性为 150
    divisions: 10,  # 设置 divisions 属性为 10
    # 定义一个函数 getFace
    getFace: function() {
        try {
            return this.faces[this.face][this.weight][this.style]  # 返回 faces 对象中对应 face, weight, style 的值
        } catch (a) {
            throw "The font " + this.face + " with " + this.weight + " weight and " + this.style + " style is missing."  # 抛出异常，提示缺少对应的字体
        }
    },
    # 定义一个函数 loadFace，参数为 a
    loadFace: function(a) {
        var b = a.familyName.toLowerCase();  # 将 familyName 转换为小写并赋值给变量 b
        this.faces[b] = this.faces[b] || {};  # 如果 faces[b] 不存在，则创建一个空对象
        this.faces[b][a.cssFontWeight] = this.faces[b][a.cssFontWeight] || {};  # 如果 faces[b][a.cssFontWeight] 不存在，则创建一个空对象
        this.faces[b][a.cssFontWeight][a.cssFontStyle] = a;  # 将 a 赋值给 faces[b][a.cssFontWeight][a.cssFontStyle]
        return this.faces[b][a.cssFontWeight][a.cssFontStyle] =  # 返回 faces[b][a.cssFontWeight][a.cssFontStyle]
    }
};
THREE.SceneUtils = {
    # 定义一个函数 createMultiMaterialObject，参数为 a, b
    createMultiMaterialObject: function(a, b) {
        var c = new THREE.Object3D();  # 创建一个 Object3D 对象并赋值给变量 c
        for (var d = 0, e = b.length; d < e; d++) {
            c.add(new THREE.Mesh(a, b[d]));  # 将 Mesh 对象添加到 c 中
        }
        return c;  # 返回 c
    },
    # 定义一个函数 detach，参数为 a, b, c
    detach: function(a, b, c) {
        a.applyMatrix(b.matrixWorld);  # 将 b 的世界矩阵应用到 a 上
        b.remove(a);  # 将 a 从 b 中移除
        c.add(a);  # 将 a 添加到 c 中
    },
    # 定义一个函数 attach，参数为 a, b, c
    attach: function(a, b, c) {
        var d = new THREE.Matrix4();  # 创建一个 Matrix4 对象并赋值给变量 d
        d.getInverse(c.matrixWorld);  # 获取 c 的世界矩阵的逆矩阵
        a.applyMatrix(d);  # 将 d 应用到 a 上
        b.remove(a);  # 将 a 从 b 中移除
        c.add(a);  # 将 a 添加到 c 中
    }
};
// 定义一个对象，包含一些方法
a},drawText:function(a){
    // 获取字体对象
    var b=this.getFace(),
    // 计算字体大小
    c=this.size/b.resolution,
    // 初始化一些变量
    d=0,
    e=String(a).split(""),
    f=e.length,
    g=[];
    // 遍历字符串中的每个字符
    for(a=0;a<f;a++){
        // 创建路径对象
        var h=new THREE.Path,
        // 提取字符的轮廓点
        h=this.extractGlyphPoints(e[a],b,c,d,h),
        // 更新偏移量
        d=d+h.offset;
        // 将路径添加到数组中
        g.push(h.path)
    }
    // 返回包含路径数组和偏移量的对象
    return{paths:g,offset:d/2}
},
// 提取字符的轮廓点
extractGlyphPoints:function(a,b,c,d,e){
    var f=[],
    g,h,k,n,p,q,m,r,t,s,u,v=b.glyphs[a]||b.glyphs["?"];
    if(v){
        // 解析轮廓点
        if(v.o)
            for(b=v._cachedOutline||(v._cachedOutline=v.o.split(" ")),
            n=b.length,
            a=0;a<n;)
                switch(k=b[a++],k){
                    case "m":
                        k=b[a++]*c+d;
                        p=b[a++]*c;
                        e.moveTo(k,p);
                        break;
                    case "l":
                        k=b[a++]*c+d;
                        p=b[a++]*c;
                        e.lineTo(k,p);
                        break;
                    case "q":
                        k=b[a++]*c+d;
                        p=b[a++]*c;
                        r=b[a++]*c+d;
                        t=b[a++]*c;
                        e.quadraticCurveTo(r,t,k,p);
                        if(g=f[f.length-1])
                            for(q=g.x,m=g.y,g=1,h=this.divisions;g<=h;g++){
                                var y=g/h;
                                THREE.Shape.Utils.b2(y,q,r,k);
                                THREE.Shape.Utils.b2(y,m,t,p)
                            }
                        break;
                    case "b":
                        if(k=b[a++]*c+d,
                        p=b[a++]*c,
                        r=b[a++]*c+d,
                        t=b[a++]*c,
                        s=b[a++]*c+d,
                        u=b[a++]*c,
                        e.bezierCurveTo(r,t,s,u,k,p),
                        g=f[f.length-1])
                            for(q=g.x,m=g.y,g=1,h=this.divisions;g<=h;g++)
                                y=g/h,
                                THREE.Shape.Utils.b3(y,q,r,s,k),
                                THREE.Shape.Utils.b3(y,m,t,u,p)
                }
        // 返回包含偏移量和路径的对象
        return{offset:v.ha*c,path:e}
    }
};
// 生成字体的形状
THREE.FontUtils.generateShapes=function(a,b){
    // 设置曲线段数、字体、粗细、样式和大小
    b=b||{};
    var c=void 0!==b.curveSegments?b.curveSegments:4,
    d=void 0!==b.font?b.font:"helvetiker",
    e=void 0!==b.weight?b.weight:"normal",
    f=void 0!==b.style?b.style:"normal";
    THREE.FontUtils.size=void 0!==b.size?b.size:100;
    THREE.FontUtils.divisions=c;
    THREE.FontUtils.face=d;
    THREE.FontUtils.weight=e;
    THREE.FontUtils.style=f;
    // 绘制文本并获取路径数组
    c=THREE.FontUtils.drawText(a).paths;
    d=[];
    e=0;
    for(f=c.length;e<f;e++)
        // 将路径数组转换为形状数组
        Array.prototype.push.apply(d,c[e].toShapes());
    // 返回形状数组
    return d
};
(function(a){
    var b = function(a){
        for(var b=a.length,e=0,f=b-1,g=0;g<b;f=g++)
            e+=a[f].x*a[g].y-a[g].x*a[f].y;
        return.5*e
    };
    a.Triangulate = function(a,d){
        var e=a.length;
        if(3>e) return null;
        var f=[],g=[],h=[],k,n,p;
        if(0<b(a))
            for(n=0;n<e;n++)
                g[n]=n;
        else
            for(n=0;n<e;n++)
                g[n]=e-1-n;
        var q=2*e;
        for(n=e-1;2<e;){
            if(0>=q--){
                console.log("Warning, unable to triangulate polygon!");
                break
            }
            k=n;
            e<=k&&(k=0);
            n=k+1;
            e<=n&&(n=0);
            p=n+1;
            e<=p&&(p=0);
            var m;
            a:{
                var r=m=void 0,t=void 0,s=void 0,u=void 0,v=void 0,y=void 0,G=void 0,w=void 0,
                r=a[g[k]].x,t=a[g[k]].y,s=a[g[n]].x,u=a[g[n]].y,v=a[g[p]].x,y=a[g[p]].y;
                if(1E-10>(s-r)*(y-t)-(u-t)*(v-r))
                    m=!1;
                else{
                    var K=void 0,x=void 0,D=void 0,E=void 0,A=void 0,B=void 0,F=void 0,R=void 0,H=void 0,C=void 0,H=R=F=w=G=void 0,K=v-s,x=y-u,D=r-v,E=t-y,A=s-r,B=u-t;
                    for(m=0;m<e;m++)
                        if(G=a[g[m]].x,w=a[g[m]].y,!(G===r&&w===t||G===s&&w===u||G===v&&w===y)&&(F=G-r,R=w-t,H=G-s,C=w-u,G-=v,w-=y,H=K*C-x*H,F=A*R-B*F,R=D*w-E*G,-1E-10<=H&&-1E-10<=R&&-1E-10<=F)){
                            m=!1;
                            break a
                        }
                    m=!0
                }
            }
            if(m){
                f.push([a[g[k]],a[g[n]],a[g[p]]]);
                h.push([g[k],g[n],g[p]]);
                k=n;
                for(p=n+1;p<e;k++,p++)
                    g[k]=g[p];
                e--;
                q=2*e
            }
        }
        return d?h:f
    };
    a.Triangulate.area = b;
    return a
})(THREE.FontUtils);
self._typeface_js = {
    faces: THREE.FontUtils.faces,
    loadFace: THREE.FontUtils.loadFace
};
THREE.typeface_js = self._typeface_js;
THREE.Audio = function(a){
    THREE.Object3D.call(this);
    this.type = "Audio";
    this.context = a.context;
    this.source = this.context.createBufferSource();
    this.gain = this.context.createGain();
    this.gain.connect(this.context.destination);
    this.panner = this.context.createPanner();
    this.panner.connect(this.gain)
};
THREE.Audio.prototype = Object.create(THREE.Object3D.prototype);
// 定义 THREE.Audio 对象的 load 方法，用于加载音频文件
THREE.Audio.prototype.load=function(a){
    // 创建 XMLHttpRequest 对象
    var b=this,c=new XMLHttpRequest();
    // 打开异步 GET 请求，获取音频文件
    c.open("GET",a,!0);
    // 设置响应类型为 arraybuffer
    c.responseType="arraybuffer";
    // 当请求加载完成时的回调函数
    c.onload=function(a){
        // 解码音频数据
        b.context.decodeAudioData(this.response,function(a){
            // 将解码后的音频数据设置为音频源的缓冲区
            b.source.buffer=a;
            // 将音频源连接到声音定位器
            b.source.connect(b.panner);
            // 开始播放音频
            b.source.start(0)
        })
    };
    // 发送请求
    c.send();
    // 返回当前对象
    return this
};

// 定义 THREE.Audio 对象的 setLoop 方法，用于设置音频是否循环播放
THREE.Audio.prototype.setLoop=function(a){
    this.source.loop=a
};

// 定义 THREE.Audio 对象的 setRefDistance 方法，用于设置声音定位器的参考距离
THREE.Audio.prototype.setRefDistance=function(a){
    this.panner.refDistance=a
};

// 定义 THREE.Audio 对象的 setRolloffFactor 方法，用于设置声音定位器的衰减因子
THREE.Audio.prototype.setRolloffFactor=function(a){
    this.panner.rolloffFactor=a
};

// 定义 THREE.Audio 对象的 updateMatrixWorld 方法，用于更新音频源的世界矩阵
THREE.Audio.prototype.updateMatrixWorld=function(){
    // 创建临时变量 a，用于存储音频源的位置
    var a=new THREE.Vector3;
    // 返回一个函数
    return function(b){
        // 调用父类的 updateMatrixWorld 方法
        THREE.Object3D.prototype.updateMatrixWorld.call(this,b);
        // 获取音频源的世界坐标
        a.setFromMatrixPosition(this.matrixWorld);
        // 设置声音定位器的位置为音频源的世界坐标
        this.panner.setPosition(a.x,a.y,a.z)
    }
}();

// 定义 THREE.AudioListener 对象，用于监听音频
THREE.AudioListener=function(){
    // 调用父类的构造函数
    THREE.Object3D.call(this);
    // 设置对象类型为 AudioListener
    this.type="AudioListener";
    // 创建音频上下文对象
    this.context=new (window.AudioContext||window.webkitAudioContext)
};

// 继承 THREE.Object3D 对象的原型方法
THREE.AudioListener.prototype=Object.create(THREE.Object3D.prototype);

// 定义 THREE.AudioListener 对象的 updateMatrixWorld 方法，用于更新音频监听器的世界矩阵
THREE.AudioListener.prototype.updateMatrixWorld=function(){
    // 创建临时变量
    var a=new THREE.Vector3,
        b=new THREE.Quaternion,
        c=new THREE.Vector3,
        d=new THREE.Vector3,
        e=new THREE.Vector3,
        f=new THREE.Vector3;
    // 返回一个函数
    return function(g){
        // 调用父类的 updateMatrixWorld 方法
        THREE.Object3D.prototype.updateMatrixWorld.call(this,g);
        // 获取音频监听器的世界坐标、旋转和缩放
        this.matrixWorld.decompose(a,b,c);
        // 计算音频监听器的前方向
        d.set(0,0,-1).applyQuaternion(b);
        // 计算音频监听器的速度
        e.subVectors(a,f);
        // 设置音频监听器的位置、方向、上方向和速度
        g=this.context.listener;
        g.setPosition(a.x,a.y,a.z);
        g.setOrientation(d.x,d.y,d.z,this.up.x,this.up.y,this.up.z);
        g.setVelocity(e.x,e.y,e.z);
        f.copy(a)
    }
}();

// 定义 THREE.Curve 对象，用于表示曲线
THREE.Curve=function(){};

// 定义 THREE.Curve 对象的 getPoint 方法，用于获取曲线上的点
THREE.Curve.prototype.getPoint=function(a){
    // 输出警告信息，getPoint 方法未实现
    console.log("Warning, getPoint() not implemented!");
    // 返回空值
    return null
};

// 定义 THREE.Curve 对象的 getPointAt 方法，用于获取曲线上指定位置的点
THREE.Curve.prototype.getPointAt=function(a){
    // 获取曲线上距离参数为 a 的点
    a=this.getUtoTmapping(a);
    return this.getPoint(a)
};

// 定义 THREE.Curve 对象的 getPoints 方法，用于获取曲线上的多个点
THREE.Curve.prototype.getPoints=function(a){
    // 如果未指定点的数量，默认为 5 个
    a||(a=5);
    var b,c=[];
    // 遍历获取曲线上的多个点
    for(b=0;b<=a;b++)
        c.push(this.getPoint(b/a));
    return c
};

// 定义 THREE.Curve 对象的 getSpacedPoints 方法，用于获取曲线上均匀分布的多个点
THREE.Curve.prototype.getSpacedPoints=function(a){
    // 如果未指定点的数量，默认为 5 个
    a||(a=5);
    var b,c=[];
    // 遍历获取曲线上均匀分布的多个点
    for(b=0;b<=a;b++)
        c.push(this.getPointAt(b/a));
    return c
};
# 计算曲线的长度
THREE.Curve.prototype.getLength=function(){var a=this.getLengths();return a[a.length-1]};
# 获取曲线上的点的长度
THREE.Curve.prototype.getLengths=function(a){a||(a=this.__arcLengthDivisions?this.__arcLengthDivisions:200);
# 如果缓存的曲线长度存在并且与指定的长度相同且不需要更新，则返回缓存的曲线长度
if(this.cacheArcLengths&&this.cacheArcLengths.length==a+1&&!this.needsUpdate)return this.cacheArcLengths;
# 标记不需要更新
this.needsUpdate=!1;
# 存储曲线长度的数组
var b=[],c,d=this.getPoint(0),e,f=0;
# 将起点的长度加入数组
b.push(0);
# 计算曲线上各点之间的长度
for(e=1;e<=a;e++)c=this.getPoint(e/a),f+=c.distanceTo(d),b.push(f),d=c;
# 缓存曲线长度数组
return this.cacheArcLengths=b};
# 更新曲线长度
THREE.Curve.prototype.updateArcLengths=function(){this.needsUpdate=!0;this.getLengths()};
# 根据长度获取曲线上的点的映射
THREE.Curve.prototype.getUtoTmapping=function(a,b){var c=this.getLengths(),d=0,e=c.length,f;
f=b?b:a*c[e-1];
for(var g=0,h=e-1,k;g<=h;)if(d=Math.floor(g+(h-g)/2),k=c[d]-f,0>k)g=d+1;else if(0<k)h=d-1;else{h=d;break}d=h;
if(c[d]==f)return d/(e-1);
g=c[d];
return c=(d+(f-g)/(c[d+1]-g))/(e-1)};
# 获取曲线上某点的切线
THREE.Curve.prototype.getTangent=function(a){var b=a-1E-4;a+=1E-4;0>b&&(b=0);1<a&&(a=1);b=this.getPoint(b);
return this.getPoint(a).clone().sub(b).normalize()};
# 获取曲线上某点的切线
THREE.Curve.prototype.getTangentAt=function(a){a=this.getUtoTmapping(a);return this.getTangent(a)};
# 曲线工具类
THREE.Curve.Utils={tangentQuadraticBezier:function(a,b,c,d){return 2*(1-a)*(c-b)+2*a*(d-c)},tangentCubicBezier:function(a,b,c,d,e){return-3*b*(1-a)*(1-a)+3*c*(1-a)*(1-a)-6*a*c*(1-a)+6*a*d*(1-a)-3*a*a*d+3*a*a*e},tangentSpline:function(a,b,c,d,e){return 6*a*a-6*a+(3*a*a-4*a+1)+(-6*a*a+6*a)+(3*a*a-2*a)},interpolate:function(a,b,c,d,e){a=.5*(c-a);d=.5*(d-b);var f=e*e;return(2*b-2*c+a+d)*e*f+(-3*b+3*c-2*a-d)*f+a*e+b}};
# 创建曲线
THREE.Curve.create=function(a,b){a.prototype=Object.create(THREE.Curve.prototype);a.prototype.getPoint=b;return a};
# 曲线路径类
THREE.CurvePath=function(){this.curves=[];this.bends=[];this.autoClose=!1};
# 添加曲线
THREE.CurvePath.prototype.add=function(a){this.curves.push(a)};
# 检查连接
THREE.CurvePath.prototype.checkConnection=function(){};
# 定义闭合路径的方法，将起点和终点连接起来
THREE.CurvePath.prototype.closePath=function(){var a=this.curves[0].getPoint(0),b=this.curves[this.curves.length-1].getPoint(1);a.equals(b)||this.curves.push(new THREE.LineCurve(b,a))};
# 根据参数值获取路径上的点的坐标
THREE.CurvePath.prototype.getPoint=function(a){var b=a*this.getLength(),c=this.getCurveLengths();for(a=0;a<c.length;){if(c[a]>=b)return b=c[a]-b,a=this.curves[a],b=1-b/a.getLength(),a.getPointAt(b);a++}return null};
# 获取路径的总长度
THREE.CurvePath.prototype.getLength=function(){var a=this.getCurveLengths();return a[a.length-1]};
# 获取每段曲线的长度，并缓存起来
THREE.CurvePath.prototype.getCurveLengths=function(){if(this.cacheLengths&&this.cacheLengths.length==this.curves.length)return this.cacheLengths;var a=[],b=0,c,d=this.curves.length;for(c=0;c<d;c++)b+=this.curves[c].getLength(),a.push(b);return this.cacheLengths=a};
# 获取路径上所有点的坐标，并计算出包围盒
THREE.CurvePath.prototype.getBoundingBox=function(){var a=this.getPoints(),b,c,d,e,f,g;b=c=Number.NEGATIVE_INFINITY;e=f=Number.POSITIVE_INFINITY;var h,k,n,p,q=a[0]instanceof THREE.Vector3;p=q?new THREE.Vector3:new THREE.Vector2;k=0;for(n=a.length;k<n;k++)h=a[k],h.x>b?b=h.x:h.x<e&&(e=h.x),h.y>c?c=h.y:h.y<f&&(f=h.y),q&&(h.z>d?d=h.z:h.z<g&&(g=h.z)),p.add(h);a={minX:e,minY:f,maxX:b,maxY:c};q&&(a.maxZ=d,a.minZ=g);return a};
# 根据路径上的点创建几何体
THREE.CurvePath.prototype.createPointsGeometry=function(a){a=this.getPoints(a,!0);return this.createGeometry(a)};
# 根据路径上的等间距点创建几何体
THREE.CurvePath.prototype.createSpacedPointsGeometry=function(a){a=this.getSpacedPoints(a,!0);return this.createGeometry(a)};
# 根据点的坐标创建几何体
THREE.CurvePath.prototype.createGeometry=function(a){for(var b=new THREE.Geometry,c=0;c<a.length;c++)b.vertices.push(new THREE.Vector3(a[c].x,a[c].y,a[c].z||0));return b};
# 添加包裹路径
THREE.CurvePath.prototype.addWrapPath=function(a){this.bends.push(a)};
# 为 THREE.CurvePath 对象添加获取变换后的点的方法，参数 a 为是否关闭路径，参数 b 为弯曲点
THREE.CurvePath.prototype.getTransformedPoints=function(a,b){var c=this.getPoints(a),d,e;b||(b=this.bends);d=0;for(e=b.length;d<e;d++)c=this.getWrapPoints(c,b[d]);return c};
# 为 THREE.CurvePath 对象添加获取变换后的间隔点的方法，参数 a 为是否关闭路径，参数 b 为弯曲点
THREE.CurvePath.prototype.getTransformedSpacedPoints=function(a,b){var c=this.getSpacedPoints(a),d,e;b||(b=this.bends);d=0;for(e=b.length;d<e;d++)c=this.getWrapPoints(c,b[d]);return c};
# 为 THREE.CurvePath 对象添加获取包裹点的方法，参数 a 为点集合，参数 b 为弯曲点
THREE.CurvePath.prototype.getWrapPoints=function(a,b){var c=this.getBoundingBox(),d,e,f,g,h,k;d=0;for(e=a.length;d<e;d++)f=a[d],g=f.x,h=f.y,k=g/c.maxX,k=b.getUtoTmapping(k,g),g=b.getPoint(k),k=b.getTangent(k),k.set(-k.y,k.x).multiplyScalar(h),f.x=g.x+k.x,f.y=g.y+k.y;return a};
# 创建一个 Gyroscope 类，继承自 Object3D 类
THREE.Gyroscope=function(){THREE.Object3D.call(this)};
# 设置 Gyroscope 类的原型为 Object3D 类的实例
THREE.Gyroscope.prototype=Object.create(THREE.Object3D.prototype);
# 为 Gyroscope 类添加更新世界矩阵的方法
THREE.Gyroscope.prototype.updateMatrixWorld=function(){var a=new THREE.Vector3,b=new THREE.Quaternion,c=new THREE.Vector3,d=new THREE.Vector3,e=new THREE.Quaternion,f=new THREE.Vector3;return function(g){this.matrixAutoUpdate&&this.updateMatrix();if(this.matrixWorldNeedsUpdate||g)this.parent?(this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix),this.matrixWorld.decompose(d,e,f),this.matrix.decompose(a,b,c),this.matrixWorld.compose(d,b,f)):this.matrixWorld.copy(this.matrix),this.matrixWorldNeedsUpdate=!1,g=!0;for(var h=0,k=this.children.length;h<k;h++)this.children[h].updateMatrixWorld(g)}}();
# 创建一个 Path 类，继承自 CurvePath 类
THREE.Path=function(a){THREE.CurvePath.call(this);this.actions=[];a&&this.fromPoints(a)};
# 设置 Path 类的原型为 CurvePath 类的实例
THREE.Path.prototype=Object.create(THREE.CurvePath.prototype);
# 定义 PathActions 对象，包含不同的路径操作类型
THREE.PathActions={MOVE_TO:"moveTo",LINE_TO:"lineTo",QUADRATIC_CURVE_TO:"quadraticCurveTo",BEZIER_CURVE_TO:"bezierCurveTo",CSPLINE_THRU:"splineThru",ARC:"arc",ELLIPSE:"ellipse"};
# 为 Path 类添加根据点集合创建路径的方法
THREE.Path.prototype.fromPoints=function(a){this.moveTo(a[0].x,a[0].y);for(var b=1,c=a.length;b<c;b++)this.lineTo(a[b].x,a[b].y)};
# 为 Path 类添加移动到指定点的方法
THREE.Path.prototype.moveTo=function(a,b){var c=Array.prototype.slice.call(arguments);this.actions.push({action:THREE.PathActions.MOVE_TO,args:c)};
# 添加直线到路径中
THREE.Path.prototype.lineTo=function(a,b){
    # 将参数转换为数组
    var c=Array.prototype.slice.call(arguments),
    # 获取上一个动作的参数
    d=this.actions[this.actions.length-1].args,
    # 创建直线曲线对象
    d=new THREE.LineCurve(new THREE.Vector2(d[d.length-2],d[d.length-1]),new THREE.Vector2(a,b));
    # 将曲线对象添加到曲线数组中
    this.curves.push(d);
    # 添加直线动作到动作数组中
    this.actions.push({action:THREE.PathActions.LINE_TO,args:c})
};

# 添加二次贝塞尔曲线到路径中
THREE.Path.prototype.quadraticCurveTo=function(a,b,c,d){
    # 将参数转换为数组
    var e=Array.prototype.slice.call(arguments),
    # 获取上一个动作的参数
    f=this.actions[this.actions.length-1].args,
    # 创建二次贝塞尔曲线对象
    f=new THREE.QuadraticBezierCurve(new THREE.Vector2(f[f.length-2],f[f.length-1]),new THREE.Vector2(a,b),new THREE.Vector2(c,d));
    # 将曲线对象添加到曲线数组中
    this.curves.push(f);
    # 添加二次贝塞尔曲线动作到动作数组中
    this.actions.push({action:THREE.PathActions.QUADRATIC_CURVE_TO,args:e})
};

# 添加三次贝塞尔曲线到路径中
THREE.Path.prototype.bezierCurveTo=function(a,b,c,d,e,f){
    # 将参数转换为数组
    var g=Array.prototype.slice.call(arguments),
    # 获取上一个动作的参数
    h=this.actions[this.actions.length-1].args,
    # 创建三次贝塞尔曲线对象
    h=new THREE.CubicBezierCurve(new THREE.Vector2(h[h.length-2],h[h.length-1]),new THREE.Vector2(a,b),new THREE.Vector2(c,d),new THREE.Vector2(e,f));
    # 将曲线对象添加到曲线数组中
    this.curves.push(h);
    # 添加三次贝塞尔曲线动作到动作数组中
    this.actions.push({action:THREE.PathActions.BEZIER_CURVE_TO,args:g})
};

# 添加样条曲线到路径中
THREE.Path.prototype.splineThru=function(a){
    # 将参数转换为数组
    var b=Array.prototype.slice.call(arguments),
    # 获取上一个动作的参数
    c=this.actions[this.actions.length-1].args,
    # 创建样条曲线对象
    c=[new THREE.Vector2(c[c.length-2],c[c.length-1])];
    Array.prototype.push.apply(c,a);
    c=new THREE.SplineCurve(c);
    # 将曲线对象添加到曲线数组中
    this.curves.push(c);
    # 添加样条曲线动作到动作数组中
    this.actions.push({action:THREE.PathActions.CSPLINE_THRU,args:b})
};

# 添加圆弧到路径中
THREE.Path.prototype.arc=function(a,b,c,d,e,f){
    # 获取上一个动作的参数
    var g=this.actions[this.actions.length-1].args;
    # 调用absarc方法添加圆弧到路径中
    this.absarc(a+g[g.length-2],b+g[g.length-1],c,d,e,f)
};
# 定义 Path 对象的 absarc 方法，用于添加绝对位置的圆弧
THREE.Path.prototype.absarc=function(a,b,c,d,e,f){this.absellipse(a,b,c,c,d,e,f)};
# 定义 Path 对象的 ellipse 方法，用于添加相对位置的椭圆
THREE.Path.prototype.ellipse=function(a,b,c,d,e,f,g){var h=this.actions[this.actions.length-1].args;this.absellipse(a+h[h.length-2],b+h[h.length-1],c,d,e,f,g)};
# 定义 Path 对象的 absellipse 方法，用于添加绝对位置的椭圆
THREE.Path.prototype.absellipse=function(a,b,c,d,e,f,g){var h=Array.prototype.slice.call(arguments),k=new THREE.EllipseCurve(a,b,c,d,e,f,g);this.curves.push(k);k=k.getPoint(1);h.push(k.x);h.push(k.y);this.actions.push({action:THREE.PathActions.ELLIPSE,args:h})};
# 定义 Path 对象的 getSpacedPoints 方法，用于获取等间距的点
THREE.Path.prototype.getSpacedPoints=function(a,b){a||(a=40);for(var c=[],d=0;d<a;d++)c.push(this.getPoint(d/a));return c};
# 定义 Path 对象的 getPoints 方法，用于获取点集合
THREE.Path.prototype.getPoints=function(a,b){if(this.useSpacedPoints)return console.log("tata"),this.getSpacedPoints(a,b);a=a||12;var c=[],d,e,f,g,h,k,n,p,q,m,r,t,s;d=0;for(e=this.actions.length;d<e;d++)switch(f=this.actions[d],g=f.action,f=f.args,g){
    # 处理移动到指定位置的操作
    case THREE.PathActions.MOVE_TO:c.push(new THREE.Vector2(f[0],f[1]));break;
    # 处理直线到指定位置的操作
    case THREE.PathActions.LINE_TO:c.push(new THREE.Vector2(f[0],f[1]));break;
    # 处理二次贝塞尔曲线到指定位置的操作
    case THREE.PathActions.QUADRATIC_CURVE_TO:h=f[2];k=f[3];q=f[0];m=f[1];0<c.length?(g=c[c.length-1],r=g.x,t=g.y):(g=this.actions[d-1].args,r=g[g.length-2],t=g[g.length-1]);for(f=1;f<=a;f++)s=f/a,g=THREE.Shape.Utils.b2(s,r,q,h),s=THREE.Shape.Utils.b2(s,t,m,k),c.push(new THREE.Vector2(g,s));break;
    # 处理三次贝塞尔曲线到指定位置的操作
    case THREE.PathActions.BEZIER_CURVE_TO:h=f[4];k=f[5];q=f[0];m=f[1];n=f[2];p=f[3];0<c.length?(g=c[c.length-1],r=g.x,t=g.y):(g=this.actions[d-1].args,r=g[g.length-2],t=g[g.length-1]);for(f=1;f<=a;f++)s=f/a,g=THREE.Shape.Utils.b3(s,r,q,n,h),s=THREE.Shape.Utils.b3(s,t,m,p,k),c.push(new THREE.Vector2(g,s));break;
    # 处理样条曲线通过指定点的操作
    case THREE.PathActions.CSPLINE_THRU:g=
# 定义函数，将路径转换为形状
THREE.Path.prototype.toShapes=function(a,b){
    # 定义函数，将路径数组转换为形状数组
    function c(a){
        # 创建空数组
        for(var b=[],c=0,d=a.length;c<d;c++){
            # 遍历路径数组，创建新的形状对象，将其添加到数组中
            var e=a[c],f=new THREE.Shape;
            f.actions=e.actions;
            f.curves=e.curves;
            b.push(f)
        }
        return b
    }
    # 定义函数，判断点是否在多边形内部
    function d(a,b){
        for(var c=b.length,d=!1,e=c-1,f=0;f<c;e=f++){
            var g=b[e],h=b[f],k=h.x-g.x,m=h.y-g.y;
            if(1E-10<Math.abs(m)){
                if(0>m&&(g=b[f],k=-k,h=b[e],m=-m),!(a.y<g.y||a.y>h.y)){
                    if(a.y==g.y){
                        if(a.x==g.x) return!0
                    }else{
                        e=m*(a.x-g.x)-k*(a.y-g.y);
                        if(0==e) return!0;
                        0>e||(d=!d)
                    }
                }else if(a.y==g.y&&(h.x<=a.x&&a.x<=g.x||g.x<=a.x&&a.x<=h.x)) return!0
            }
        }
        return d
    }
    # 定义函数，将路径数组转换为形状数组
    var e=function(a){
        var b,c,d,e,f=[],g=new THREE.Path;
        b=0;
        for(c=a.length;b<c;b++){
            d=a[b];
            e=d.args;
            d=d.action;
            d==THREE.PathActions.MOVE_TO&&0!=g.actions.length&&(f.push(g),g=new THREE.Path);
            g[d].apply(g,e)
        }
        0!=g.actions.length&&f.push(g);
        return f
    }(this.actions);
    # 如果路径数组为空，则返回空数组
    if(0==e.length) return[];
    # 如果参数 b 为真，则调用函数 c，将路径数组转换为形状数组
    if(!0===b) return c(e);
    var f,g,h,k=[];
    # 如果路径数组只有一个元素，则将其转换为形状对象，添加到数组中
    if(1==e.length){
        g=e[0];
        h=new THREE.Shape;
        h.actions=g.actions;
        h.curves=g.curves;
        k.push(h);
        return k
    }
    # 判断路径数组中的路径是否为顺时针方向，根据参数 a 决定是否取反
    var n=!THREE.Shape.Utils.isClockWise(e[0].getPoints()),n=a?!n:n;
# 初始化变量
h=[];
var p=[],
    q=[],
    m=0,
    r;
p[m]=void 0;
q[m]=[];
var t,
    s;
t=0;
# 遍历数组 e
for(s=e.length;t<s;t++)
    g=e[t],
    r=g.getPoints(),
    f=THREE.Shape.Utils.isClockWise(r),
    # 判断是否需要反转
    (f=a?!f:f)?
        # 如果需要反转，且不是第一个，则增加 m
        (!n&&p[m]&&m++,
        p[m]={s:new THREE.Shape,p:r},
        p[m].s.actions=g.actions,
        p[m].s.curves=g.curves,
        n&&m++,
        q[m]=[]):
        q[m].push({h:g,p:r[0]});
# 如果 p[0] 不存在，则返回 c(e)
if(!p[0])
    return c(e);
# 如果 p 的长度大于 1
if(1<p.length){
    t=!1;
    s=[];
    g=0;
    # 初始化数组 h
    for(e=p.length;g<e;g++)
        h[g]=[];
    g=0;
    # 遍历 p
    for(e=p.length;g<e;g++)
        # 遍历 q[g]
        for(f=q[g],n=0;n<f.length;n++){
            m=f[n];
            r=!0;
            # 遍历 p
            for(var u=0;u<p.length;u++)
                # 判断是否为同一个点
                d(m.p,p[u].p)&&(g!=u&&s.push({froms:g,tos:u,hole:n}),r?(r=!1,h[u].push(m)):t=!0);
            r&&h[g].push(m)
        }
    # 如果 s 的长度大于 0
    0<s.length&&(t||(q=h))
}
t=0;
# 遍历 p
for(s=p.length;t<s;t++)
    h=p[t].s,
    k.push(h),
    g=q[t],
    e=0,
    f=g.length;
    # 将 g 中的元素添加到 h 的 holes 数组中
    for(e=0;e<f;e++)
        h.holes.push(g[e].h);
# 返回 k
return k};
# 创建 Shape 类
THREE.Shape=function(){
    THREE.Path.apply(this,arguments);
    this.holes=[]
};
# 继承 Path 类
THREE.Shape.prototype=Object.create(THREE.Path.prototype);
# 创建 extrude 方法
THREE.Shape.prototype.extrude=function(a){
    return new THREE.ExtrudeGeometry(this,a)
};
# 创建 makeGeometry 方法
THREE.Shape.prototype.makeGeometry=function(a){
    return new THREE.ShapeGeometry(this,a)
};
# 获取所有孔的点
THREE.Shape.prototype.getPointsHoles=function(a){
    var b,c=this.holes.length,d=[];
    for(b=0;b<c;b++)
        d[b]=this.holes[b].getTransformedPoints(a,this.bends);
    return d
};
# 获取所有孔的间隔点
THREE.Shape.prototype.getSpacedPointsHoles=function(a){
    var b,c=this.holes.length,d=[];
    for(b=0;b<c;b++)
        d[b]=this.holes[b].getTransformedSpacedPoints(a,this.bends);
    return d
};
# 提取所有点
THREE.Shape.prototype.extractAllPoints=function(a){
    return{
        shape:this.getTransformedPoints(a),
        holes:this.getPointsHoles(a)
    }
};
# 提取点
THREE.Shape.prototype.extractPoints=function(a){
    return this.useSpacedPoints?this.extractAllSpacedPoints(a):this.extractAllPoints(a)
};
# 提取所有间隔点
THREE.Shape.prototype.extractAllSpacedPoints=function(a){
    return{
        shape:this.getTransformedSpacedPoints(a),
        holes:this.getSpacedPointsHoles(a)
    }
};
# 定义一个名为 Shape 的对象，包含一个名为 Utils 的属性
THREE.Shape.Utils={
    # 定义一个 triangulateShape 函数，接受两个参数 a 和 b
    triangulateShape:function(a,b){
        # 定义一个函数 c，接受三个参数 a、b、c
        function c(a,b,c){
            # 判断点 c 是否在点 a 和点 b 构成的线段上
            return a.x!=b.x?a.x<b.x?a.x<=c.x&&c.x<=b.x:b.x<=c.x&&c.x<=a.x:a.y<b.y?a.y<=c.y&&c.y<=b.y:b.y<=c.y&&c.y<=a.y
        }
        # 定义一个函数 d，接受五个参数 a、b、d、e、f
        function d(a,b,d,e,f){
            # 计算向量 ab 和向量 de 的叉积
            var g=b.x-a.x,h=b.y-a.y,k=e.x-d.x,n=e.y-d.y,p=a.x-d.x,q=a.y-d.y,D=h*k-g*n,E=h*p-g*q;
            # 如果叉积不为 0
            if(1E-10<Math.abs(D)){
                if(0<D){
                    if(0>E||E>D)return[];
                    k=n*p-k*q;
                    if(0>k||k>D)return[]
                }else{
                    if(0<E||E<D)return[];
                    k=n*p-k*q;
                    if(0<k||k<D)return[]
                }
                if(0==k)return!f||0!=E&&E!=D?[a]:[];
                if(k==D)return!f||0!=E&&E!=D?[b]:[];
                if(0==E)return[d];
                if(E==D)return[e];
                f=k/D;
                return[{x:a.x+f*g,y:a.y+f*h}]
            }
            if(0!=E||n*p!=k*q)return[];
            h=0==g&&0==h;
            k=0==k&&0==n;
            if(h&&k)return a.x!=d.x||a.y!=d.y?[]:[a];
            if(h)return c(d,e,a)?[a]:[];
            if(k)return c(a,b,d)?[d]:[];
            0!=g?(a.x<b.x?(g=a,k=a.x,h=b,a=b.x):(g=b,k=b.x,h=a,a=a.x),d.x<e.x?(b=d,D=d.x,n=e,d=e.x):(b=e,D=e.x,n=d,d=d.x)):(a.y<b.y?(g=a,k=a.y,h=b,a=b.y):(g=b,k=b.y,h=a,a=a.y),d.y<e.y?(b=d,D=d.y,n=e,d=e.y):(b=e,D=e.y,n=d,d=d.y));
            return k<=D?a<D?[]:a==D?f?[]:[b]:a<=d?[b,h]:[b,n]:k>d?[]:k==d?f?[]:[g]:a<=d?[g,h]:[g,n]
        }
        # 定义一个函数 e，接受四个参数 a、b、c、d
        function e(a,b,c,d){
            var e=b.x-a.x,f=b.y-a.y;
            b=c.x-a.x;
            c=c.y-a.y;
            var g=d.x-a.x;
            d=d.y-a.y;
            a=e*c-f*b;
            e=e*d-f*g;
            return 1E-10<Math.abs(a)?(b=g*c-d*b,0<a?0<=e&&0<=b:0<=e||0<=b):0<e
        }
        var f,g,h,k,n,p={};
        h=a.concat();
        f=0;
        for(g=b.length;f<g;f++)Array.prototype.push.apply(h,b[f]);
        f=0;
        for(g=h.length;f<g;f++)n=h[f].x+":"+h[f].y,void 0!==p[n]&&console.log("Duplicate point",n),p[n]=f;
        f=function(a,b){
            function c(a,b){
                var d=h.length-1,f=a-1;
                0>f&&(f=d);
                var g=a+1;
                g>d&&(g=0);
                d=e(h[a],h[f],h[g],k[b]);
                if(!d)return!1;
// 定义函数，计算三个点的曲线是否为凹形
function e(a, b, c, d) {
    // 计算三个点的曲线是否为凹形，返回布尔值
    return (a - c) * (d - b) - (c - b) * (d - a) >= 0 ? !0 : !1
}
// 定义函数，检查点是否在多边形内部
function f(a, b) {
    var c, e;
    for (c = 0; c < h.length; c++)
        if (e = c + 1, e %= h.length, e = d(a, b, h[c], h[e], !0), 0 < e.length) return !0;
    return !1
}
// 定义函数，检查点是否在多边形外部
function g(a, c) {
    var e, f, h, k;
    for (e = 0; e < n.length; e++)
        for (f = b[n[e]], h = 0; h < f.length; h++)
            if (k = h + 1, k %= f.length, k = d(a, c, f[h], f[k], !0), 0 < k.length) return !0;
    return !1
}
var h = a.concat(),
    k, n = [],
    p, q, x, D, E, A = [],
    B, F, R, H = 0;
for (p = b.length; H < p; H++) n.push(H);
B = 0;
for (var C = 2 * n.length; 0 < n.length;) {
    C--;
    if (0 > C) {
        console.log("Infinite Loop! Holes left:" + n.length + ", Probably Hole outside Shape!");
        break
    }
    for (q = B; q < h.length; q++) {
        x = h[q];
        p = -1;
        for (H = 0; H < n.length; H++)
            if (D = n[H], E = x.x + ":" + x.y + ":" + D, void 0 === A[E]) {
                k = b[D];
                for (F = 0; F < k.length; F++)
                    if (D = k[F], c(q, F) && !f(x, D) && !g(x, D)) {
                        p = F;
                        n.splice(H, 1);
                        B = h.slice(0, q + 1);
                        D = h.slice(q);
                        F = k.slice(p);
                        R = k.slice(0, p + 1);
                        h = B.concat(F).concat(R).concat(D);
                        B = q;
                        break
                    }
                if (0 <= p) break;
                A[E] = !0
            }
        if (0 <= p) break
    }
}
return h
}(a, b);
var q = THREE.FontUtils.Triangulate(f, !1);
f = 0;
for (g = q.length; f < g; f++)
    for (k = q[f], h = 0; 3 > h; h++) n = k[h].x + ":" + k[h].y, n = p[n], void 0 !== n && (k[h] = n);
return q.concat()
},
// 定义函数，判断多边形是否为顺时针方向
isClockWise: function(a) {
    return 0 > THREE.FontUtils.Triangulate.area(a)
},
// 定义函数，计算二次贝塞尔曲线
b2p0: function(a, b) {
    var c = 1 - a;
    return c * c * b
},
b2p1: function(a, b) {
    return 2 * (1 - a) * a * b
},
b2p2: function(a, b) {
    return a * a * b
},
b2: function(a, b, c, d) {
    return this.b2p0(a, b) + this.b2p1(a, c) + this.b2p2(a, d)
},
// 定义函数，计算三次贝塞尔曲线
b3p0: function(a, b) {
    var c = 1 - a;
    return c * c * c * b
},
b3p1: function(a, b) {
    var c = 1 - a;
    return 3 * c * c * a * b
},
b3p2: function(a, b) {
    return 3 * (1 - a) * a * a * b
},
b3p3: function(a, b) {
    return a * a * a * b
},
b3: function(a, b, c, d, e) {
    return this.b3p0(a, b) + this.b3p1(a, c) + this.b3p2(a, d) + this.b3p3(a, e)
}
# 定义一个三维空间中的线段曲线，起点为 v1，终点为 v2
THREE.LineCurve=function(a,b){this.v1=a;this.v2=b};
# 继承自 Curve 类
THREE.LineCurve.prototype=Object.create(THREE.Curve.prototype);
# 根据参数 t 获取曲线上的点
THREE.LineCurve.prototype.getPoint=function(a){var b=this.v2.clone().sub(this.v1);b.multiplyScalar(a).add(this.v1);return b};
# 获取曲线上参数 t 处的点
THREE.LineCurve.prototype.getPointAt=function(a){return this.getPoint(a)};
# 获取曲线上参数 t 处的切线
THREE.LineCurve.prototype.getTangent=function(a){return this.v2.clone().sub(this.v1).normalize()};

# 定义一个二次贝塞尔曲线，控制点分别为 v0, v1, v2
THREE.QuadraticBezierCurve=function(a,b,c){this.v0=a;this.v1=b;this.v2=c};
# 继承自 Curve 类
THREE.QuadraticBezierCurve.prototype=Object.create(THREE.Curve.prototype);
# 根据参数 t 获取曲线上的点
THREE.QuadraticBezierCurve.prototype.getPoint=function(a){var b=new THREE.Vector2;b.x=THREE.Shape.Utils.b2(a,this.v0.x,this.v1.x,this.v2.x);b.y=THREE.Shape.Utils.b2(a,this.v0.y,this.v1.y,this.v2.y);return b};
# 获取曲线上参数 t 处的切线
THREE.QuadraticBezierCurve.prototype.getTangent=function(a){var b=new THREE.Vector2;b.x=THREE.Curve.Utils.tangentQuadraticBezier(a,this.v0.x,this.v1.x,this.v2.x);b.y=THREE.Curve.Utils.tangentQuadraticBezier(a,this.v0.y,this.v1.y,this.v2.y);return b.normalize()};

# 定义一个三次贝塞尔曲线，控制点分别为 v0, v1, v2, v3
THREE.CubicBezierCurve=function(a,b,c,d){this.v0=a;this.v1=b;this.v2=c;this.v3=d};
# 继承自 Curve 类
THREE.CubicBezierCurve.prototype=Object.create(THREE.Curve.prototype);
# 根据参数 t 获取曲线上的点
THREE.CubicBezierCurve.prototype.getPoint=function(a){var b;b=THREE.Shape.Utils.b3(a,this.v0.x,this.v1.x,this.v2.x,this.v3.x);a=THREE.Shape.Utils.b3(a,this.v0.y,this.v1.y,this.v2.y,this.v3.y);return new THREE.Vector2(b,a)};
# 获取曲线上参数 t 处的切线
THREE.CubicBezierCurve.prototype.getTangent=function(a){var b;b=THREE.Curve.Utils.tangentCubicBezier(a,this.v0.x,this.v1.x,this.v2.x,this.v3.x);a=THREE.Curve.Utils.tangentCubicBezier(a,this.v0.y,this.v1.y,this.v2.y,this.v3.y);b=new THREE.Vector2(b,a);b.normalize();return b};
# 定义 SplineCurve 类，继承自 Curve 类
THREE.SplineCurve=function(a){this.points=void 0==a?[]:a};
THREE.SplineCurve.prototype=Object.create(THREE.Curve.prototype);
# 获取曲线上的点
THREE.SplineCurve.prototype.getPoint=function(a){
    var b=this.points;
    a*=b.length-1;
    var c=Math.floor(a);
    a-=c;
    var d=b[0==c?c:c-1],
        e=b[c],
        f=b[c>b.length-2?b.length-1:c+1],
        b=b[c>b.length-3?b.length-1:c+2],
        c=new THREE.Vector2;
    c.x=THREE.Curve.Utils.interpolate(d.x,e.x,f.x,b.x,a);
    c.y=THREE.Curve.Utils.interpolate(d.y,e.y,f.y,b.y,a);
    return c;
};

# 定义 EllipseCurve 类，继承自 Curve 类
THREE.EllipseCurve=function(a,b,c,d,e,f,g){
    this.aX=a;
    this.aY=b;
    this.xRadius=c;
    this.yRadius=d;
    this.aStartAngle=e;
    this.aEndAngle=f;
    this.aClockwise=g
};
THREE.EllipseCurve.prototype=Object.create(THREE.Curve.prototype);
# 获取椭圆曲线上的点
THREE.EllipseCurve.prototype.getPoint=function(a){
    var b=this.aEndAngle-this.aStartAngle;
    0>b&&(b+=2*Math.PI);
    b>2*Math.PI&&(b-=2*Math.PI);
    a=!0===this.aClockwise?this.aEndAngle+(1-a)*(2*Math.PI-b):this.aStartAngle+a*b;
    b=new THREE.Vector2;
    b.x=this.aX+this.xRadius*Math.cos(a);
    b.y=this.aY+this.yRadius*Math.sin(a);
    return b
};

# 定义 ArcCurve 类，继承自 EllipseCurve 类
THREE.ArcCurve=function(a,b,c,d,e,f){
    THREE.EllipseCurve.call(this,a,b,c,c,d,e,f)
};
THREE.ArcCurve.prototype=Object.create(THREE.EllipseCurve.prototype);

# 定义 LineCurve3 类
THREE.LineCurve3=THREE.Curve.create(function(a,b){
    this.v1=a;
    this.v2=b
},function(a){
    var b=new THREE.Vector3;
    b.subVectors(this.v2,this.v1);
    b.multiplyScalar(a);
    b.add(this.v1);
    return b
});

# 定义 QuadraticBezierCurve3 类
THREE.QuadraticBezierCurve3=THREE.Curve.create(function(a,b,c){
    this.v0=a;
    this.v1=b;
    this.v2=c
},function(a){
    var b=new THREE.Vector3;
    b.x=THREE.Shape.Utils.b2(a,this.v0.x,this.v1.x,this.v2.x);
    b.y=THREE.Shape.Utils.b2(a,this.v0.y,this.v1.y,this.v2.y);
    b.z=THREE.Shape.Utils.b2(a,this.v0.z,this.v1.z,this.v2.z);
    return b
});

# 定义 CubicBezierCurve3 类
THREE.CubicBezierCurve3=THREE.Curve.create(function(a,b,c,d){
    this.v0=a;
    this.v1=b;
    this.v2=c;
    this.v3=d
},function(a){
    var b=new THREE.Vector3;
    b.x=THREE.Shape.Utils.b3(a,this.v0.x,this.v1.x,this.v2.x,this.v3.x);
    b.y=THREE.Shape.Utils.b3(a,this.v0.y,this.v1.y,this.v2.y,this.v3.y);
    b.z=THREE.Shape.Utils.b3(a,this.v0.z,this.v1.z,this.v2.z,this.v3.z);
    return b
});
# 定义一个三维样条曲线，根据给定的点创建曲线
THREE.SplineCurve3=THREE.Curve.create(function(a){this.points=void 0==a?[]:a},function(a){
    var b=this.points;
    a*=b.length-1;
    var c=Math.floor(a);
    a-=c;
    var d=b[0==c?c:c-1],
        e=b[c],
        f=b[c>b.length-2?b.length-1:c+1],
        b=b[c>b.length-3?b.length-1:c+2],
        c=new THREE.Vector3;
    c.x=THREE.Curve.Utils.interpolate(d.x,e.x,f.x,b.x,a);
    c.y=THREE.Curve.Utils.interpolate(d.y,e.y,f.y,b.y,a);
    c.z=THREE.Curve.Utils.interpolate(d.z,e.z,f.z,b.z,a);
    return c;
});

# 定义一个闭合的三维样条曲线，根据给定的点创建曲线
THREE.ClosedSplineCurve3=THREE.Curve.create(function(a){this.points=void 0==a?[]:a},function(a){
    var b=this.points;
    a*=b.length-0;
    var c=Math.floor(a);
    a-=c;
    var c=c+(0<c?0:(Math.floor(Math.abs(c)/b.length)+1)*b.length),
        d=b[(c-1)%b.length],
        e=b[c%b.length],
        f=b[(c+1)%b.length],
        b=b[(c+2)%b.length],
        c=new THREE.Vector3;
    c.x=THREE.Curve.Utils.interpolate(d.x,e.x,f.x,b.x,a);
    c.y=THREE.Curve.Utils.interpolate(d.y,e.y,f.y,b.y,a);
    c.z=THREE.Curve.Utils.interpolate(d.z,e.z,f.z,b.z,a);
    return c;
});

# 定义动画处理器对象，包含一些方法和属性
THREE.AnimationHandler={
    LINEAR:0,
    CATMULLROM:1,
    CATMULLROM_FORWARD:2,
    add:function(){console.warn("THREE.AnimationHandler.add() has been deprecated.")},
    get:function(){console.warn("THREE.AnimationHandler.get() has been deprecated.")},
    remove:function(){console.warn("THREE.AnimationHandler.remove() has been deprecated.")},
    animations:[],
    init:function(a){
        if(!0!==a.initialized){
            for(var b=0;b<a.hierarchy.length;b++){
                for(var c=0;c<a.hierarchy[b].keys.length;c++){
                    if(0>a.hierarchy[b].keys[c].time&&(a.hierarchy[b].keys[c].time=0),
                        void 0!==a.hierarchy[b].keys[c].rot&&!(a.hierarchy[b].keys[c].rot instanceof THREE.Quaternion)){
                        var d=a.hierarchy[b].keys[c].rot;
                        a.hierarchy[b].keys[c].rot=(new THREE.Quaternion).fromArray(d)
                    }
                }
                if(a.hierarchy[b].keys.length&&void 0!==a.hierarchy[b].keys[0].morphTargets){
                    d={};
                    for(c=0;c<a.hierarchy[b].keys.length;c++){
                        for(var e=0;e<a.hierarchy[b].keys[c].morphTargets.length;e++){
                            var f=a.hierarchy[b].keys[c].morphTargets[e];
                            d[f]=-1
                        }
                    }
                    a.hierarchy[b].usedMorphTargets=d;
                    for(c=0;c<a.hierarchy[b].keys.length;c++){
                        var g=
# 创建一个空对象
{};
# 遍历对象 d 的属性
for(f in d){
    # 遍历 a.hierarchy[b].keys[c].morphTargets 数组的长度
    for(e=0;e<a.hierarchy[b].keys[c].morphTargets.length;e++)
        # 如果 a.hierarchy[b].keys[c].morphTargets[e] 等于 f
        if(a.hierarchy[b].keys[c].morphTargets[e]===f){
            # 将 g 对象的属性 f 设置为 a.hierarchy[b].keys[c].morphTargetsInfluences[e]
            g[f]=a.hierarchy[b].keys[c].morphTargetsInfluences[e];
            # 跳出循环
            break
        }
    # 如果 e 等于 a.hierarchy[b].keys[c].morphTargets.length
    e===a.hierarchy[b].keys[c].morphTargets.length&&(g[f]=0)
}
# 遍历 a.hierarchy[b].keys 数组的长度
for(c=1;c<a.hierarchy[b].keys.length;c++)
    # 如果 a.hierarchy[b].keys[c].time 等于 a.hierarchy[b].keys[c-1].time
    a.hierarchy[b].keys[c].time===a.hierarchy[b].keys[c-1].time&&(a.hierarchy[b].keys.splice(c,1),c--);
# 遍历 a.hierarchy[b].keys 数组的长度
for(c=0;c<a.hierarchy[b].keys.length;c++)
    # 设置 a.hierarchy[b].keys[c].index 为 c
    a.hierarchy[b].keys[c].index=c
# 设置 a.initialized 为 true
a.initialized=!0;
# 返回对象 a
return a
}},
# 解析函数，根据参数 a 返回一个数组
parse:function(a){
    # 定义函数 b，参数为 a 和 c
    var b=function(a,c){
        # 将 a 添加到数组 c 中
        c.push(a);
        # 遍历 a 的子元素
        for(var d=0;d<a.children.length;d++)
            # 递归调用函数 b
            b(a.children[d],c)
    },
    # 创建一个空数组
    c=[];
    # 如果 a 是 THREE.SkinnedMesh 的实例
    if(a instanceof THREE.SkinnedMesh)
        # 遍历 a.skeleton.bones 数组的长度
        for(var d=0;d<a.skeleton.bones.length;d++)
            # 将 a.skeleton.bones[d] 添加到数组 c 中
            c.push(a.skeleton.bones[d]);
    else
        # 调用函数 b
        b(a,c);
    # 返回数组 c
    return c
},
# 将动画添加到数组 animations 中
play:function(a){
    # 如果 animations 数组中不包含参数 a
    -1===this.animations.indexOf(a)&&this.animations.push(a)
},
# 从数组 animations 中移除指定的动画
stop:function(a){
    # 获取参数 a 在 animations 数组中的索引
    a=this.animations.indexOf(a);
    # 如果索引不为 -1
    -1!==a&&this.animations.splice(a,1)
},
# 更新动画
update:function(a){
    # 遍历数组 animations
    for(var b=0;b<this.animations.length;b++)
        # 重置动画的混合权重
        this.animations[b].resetBlendWeights();
    # 遍历数组 animations
    for(b=0;b<this.animations.length;b++)
        # 更新动画
        this.animations[b].update(a)
}};
# 创建一个 Animation 对象
THREE.Animation=function(a,b){
    # 设置 Animation 对象的属性
    this.root=a;
    this.data=THREE.AnimationHandler.init(b);
    this.hierarchy=THREE.AnimationHandler.parse(a);
    this.currentTime=0;
    this.timeScale=1;
    this.isPlaying=!1;
    this.loop=!0;
    this.weight=0;
    this.interpolationType=THREE.AnimationHandler.LINEAR
};
# 设置 Animation 对象的 keyTypes 属性
THREE.Animation.prototype.keyTypes=["pos","rot","scl"];
# 播放动画
THREE.Animation.prototype.play=function(a,b){
    # 设置当前时间和权重
    this.currentTime=void 0!==a?a:0;
    this.weight=void 0!==b?b:1;
    this.isPlaying=!0;
    # 重置动画
    this.reset();
    # 播放动画
    THREE.AnimationHandler.play(this)
};
# 停止动画
THREE.Animation.prototype.stop=function(){
    this.isPlaying=!1;
    THREE.AnimationHandler.stop(this)
};
// 重置动画对象的状态
THREE.Animation.prototype.reset=function(){
    // 遍历动画对象的层级
    for(var a=0,b=this.hierarchy.length;a<b;a++){
        var c=this.hierarchy[a];
        // 设置矩阵自动更新
        c.matrixAutoUpdate=!0;
        // 如果动画缓存不存在，则创建一个新的动画缓存对象
        void 0===c.animationCache&&(c.animationCache={animations:{},blending:{positionWeight:0,quaternionWeight:0,scaleWeight:0}});
        // 如果当前动画名称的缓存不存在，则创建一个新的缓存对象
        void 0===c.animationCache.animations[this.data.name]&&(c.animationCache.animations[this.data.name]={},c.animationCache.animations[this.data.name].prevKey={pos:0,rot:0,scl:0},c.animationCache.animations[this.data.name].nextKey={pos:0,rot:0,scl:0},
        c.animationCache.animations[this.data.name].originalMatrix=c.matrix);
        // 遍历关键帧类型
        for(var c=c.animationCache.animations[this.data.name],d=0;3>d;d++){
            // 获取关键帧类型
            for(var e=this.keyTypes[d],f=this.data.hierarchy[a].keys[0],g=this.getNextKeyWith(e,a,1);g.time<this.currentTime&&g.index>f.index;)f=g,g=this.getNextKeyWith(e,a,g.index+1);
            c.prevKey[e]=f;
            c.nextKey[e]=g;
        }
    }
};

// 重置混合权重
THREE.Animation.prototype.resetBlendWeights=function(){
    // 遍历动画对象的层级
    for(var a=0,b=this.hierarchy.length;a<b;a++){
        var c=this.hierarchy[a];
        // 如果动画缓存存在，则重置混合权重
        void 0!==c.animationCache&&(c.animationCache.blending.positionWeight=0,c.animationCache.blending.quaternionWeight=0,c.animationCache.blending.scaleWeight=0);
    }
};

// 更新动画
THREE.Animation.prototype.update=function(){
    var a=[],b=new THREE.Vector3,c=new THREE.Vector3,d=new THREE.Quaternion,e=function(a,b){
        var c=[],d=[],e,q,m,r,t,s;
        e=(a.length-1)*b;
        q=Math.floor(e);
        e-=q;
        c[0]=0===q?q:q-1;
        c[1]=q;
        c[2]=q>a.length-2?q:q+1;
        c[3]=q>a.length-3?q:q+2;
        q=a[c[0]];
        r=a[c[1]];
        t=a[c[2]];
        s=a[c[3]];
        c=e*e;
        m=e*c;
        d[0]=f(q[0],r[0],t[0],s[0],e,c,m);
        d[1]=f(q[1],r[1],t[1],s[1],e,c,m);
        d[2]=f(q[2],r[2],t[2],s[2],e,c,m);
        return d
    },
    f=function(a,b,c,d,e,f,m){
        a=.5*(c-a);
        d=.5*(d-b);
        return(2*(b-c)+a+d)*m+
# 定义一个函数，接受参数 f
return function(f){
    # 如果当前动画正在播放，并且时间比例不为 0
    if(!1!==this.isPlaying&&(this.currentTime+=f*this.timeScale,0!==this.weight)){
        # 获取动画数据的长度
        f=this.data.length;
        # 如果当前时间超过了动画长度或者小于 0
        if(this.currentTime>f||0>this.currentTime)
            # 如果循环播放，则将当前时间重置到合适的位置
            if(this.loop)
                this.currentTime%=f,0>this.currentTime&&(this.currentTime+=f),this.reset();
            # 否则停止动画并返回
            else{
                this.stop();
                return
            }
        # 初始化变量
        f=0;
        # 遍历层级
        for(var h=this.hierarchy.length;f<h;f++)
            # 遍历每个层级的动画缓存
            for(var k=this.hierarchy[f],n=k.animationCache.animations[this.data.name],p=k.animationCache.blending,q=0;3>q;q++){
                # 获取关键帧类型
                var m=this.keyTypes[q],r=n.prevKey[m],t=n.nextKey[m];
                # 根据时间比例和时间轴位置计算当前关键帧的值
                if(0<this.timeScale&&t.time<=this.currentTime||0>this.timeScale&&r.time>=this.currentTime){
                    r=this.data.hierarchy[f].keys[0];
                    for(t=this.getNextKeyWith(m,f,1);t.time<this.currentTime&&t.index>r.index;)
                        r=t,t=this.getNextKeyWith(m,f,t.index+1);
                    n.prevKey[m]=r;
                    n.nextKey[m]=t
                }
                # 更新矩阵
                k.matrixAutoUpdate=!0;
                k.matrixWorldNeedsUpdate=!0;
                # 计算插值
                var s=(this.currentTime-r.time)/(t.time-r.time),u=r[m],v=t[m];
                # 限制插值比例在 0 到 1 之间
                0>s&&(s=0);
                1<s&&(s=1);
                # 根据关键帧类型进行不同的插值计算
                if("pos"===m)
                    # 线性插值
                    if(this.interpolationType===THREE.AnimationHandler.LINEAR)
                        c.x=u[0]+(v[0]-u[0])*s,c.y=u[1]+(v[1]-u[1])*s,c.z=u[2]+(v[2]-u[2])*s,r=this.weight/(this.weight+p.positionWeight),k.position.lerp(c,r),p.positionWeight+=this.weight;
                    # Catmull-Rom 插值
                    else{
                        if(this.interpolationType===THREE.AnimationHandler.CATMULLROM||this.interpolationType===THREE.AnimationHandler.CATMULLROM_FORWARD)
                            a[0]=this.getPrevKeyWith("pos",f,r.index-1).pos,a[1]=u,a[2]=v,a[3]=this.getNextKeyWith("pos",f,t.index+1).pos,s=.33*s+.33,t=e(a,s),r=this.weight/(this.weight+p.positionWeight),p.positionWeight+=this.weight,m=k.position,m.x+=(t[0]-
# 定义一个函数，用于获取下一个关键帧
THREE.Animation.prototype.getNextKeyWith=function(a,b,c){
    # 获取当前层级的关键帧数据
    var d=this.data.hierarchy[b].keys;
    # 根据插值类型选择合适的关键帧索引
    for(c=this.interpolationType===THREE.AnimationHandler.CATMULLROM||this.interpolationType===THREE.AnimationHandler.CATMULLROM_FORWARD?c<d.length-1?c:d.length-1:c%d.length;c<d.length;c++)
        # 如果关键帧中包含指定属性，则返回该关键帧
        if(void 0!==d[c][a])return d[c];
    # 如果没有找到指定属性的关键帧，则返回当前层级的第一个关键帧
    return this.data.hierarchy[b].keys[0];
};

# 定义一个函数，用于获取上一个关键帧
THREE.Animation.prototype.getPrevKeyWith=function(a,b,c){
    # 获取当前层级的关键帧数据
    var d=this.data.hierarchy[b].keys;
    # 根据插值类型选择合适的关键帧索引
    for(c=this.interpolationType===THREE.AnimationHandler.CATMULLROM||this.interpolationType===THREE.AnimationHandler.CATMULLROM_FORWARD?0<c?c:0:0<=c?c:c+d.length;0<=c;c--)
        # 如果关键帧中包含指定属性，则返回该关键帧
        if(void 0!==d[c][a])return d[c];
    # 如果没有找到指定属性的关键帧，则返回当前层级的最后一个关键帧
    return this.data.hierarchy[b].keys[d.length-1];
};

# 定义一个关键帧动画类
THREE.KeyFrameAnimation=function(a){
    # 设置根节点
    this.root=a.node;
    # 初始化动画数据
    this.data=THREE.AnimationHandler.init(a);
    # 解析层级关系
    this.hierarchy=THREE.AnimationHandler.parse(this.root);
    # 设置当前时间
    this.currentTime=0;
    # 设置时间缩放比例
    this.timeScale=.001;
    # 设置播放状态
    this.isPlaying=!1;
    # 设置循环状态
    this.loop=this.isPaused=!0;
    # 遍历层级
    a=0;
    for(var b=this.hierarchy.length;a<b;a++){
        var c=this.data.hierarchy[a].sids,d=this.hierarchy[a];
        # 如果存在关键帧数据和关键帧标识
        if(this.data.hierarchy[a].keys.length&&c){
            # 遍历关键帧标识
            for(var e=0;e<c.length;e++){
                var f=c[e],g=this.getNextKeyWith(f,a,0);
                # 如果存在下一个关键帧，则应用该关键帧的属性
                g&&g.apply(f)
            }
            # 禁用矩阵自动更新
            d.matrixAutoUpdate=!1;
            # 更新节点矩阵
            this.data.hierarchy[a].node.updateMatrix();
            # 标记世界矩阵需要更新
            d.matrixWorldNeedsUpdate=!0
        }
    }
};
// 定义 KeyFrameAnimation 对象的 play 方法
THREE.KeyFrameAnimation.prototype.play=function(a){
    // 设置当前时间为传入的参数，如果没有传入参数则为 0
    this.currentTime=void 0!==a?a:0;
    // 如果动画没有在播放，则设置为正在播放
    if(!1===this.isPlaying){
        this.isPlaying=!0;
        var b=this.hierarchy.length,c,d;
        // 遍历层级
        for(a=0;a<b;a++)
            c=this.hierarchy[a],
            d=this.data.hierarchy[a],
            // 如果动画缓存不存在，则创建一个
            void 0===d.animationCache&&(d.animationCache={},d.animationCache.prevKey=null,d.animationCache.nextKey=null,d.animationCache.originalMatrix=c.matrix),
            c=this.data.hierarchy[a].keys,
            c.length&&(
                d.animationCache.prevKey=c[0],
                d.animationCache.nextKey=c[1],
                // 设置动画开始时间和结束时间
                this.startTime=Math.min(c[0].time,this.startTime),
                this.endTime=Math.max(c[c.length-1].time,this.endTime)
            );
        // 更新动画
        this.update(0)
    }
    // 设置为非暂停状态
    this.isPaused=!1;
    // 调用 AnimationHandler 的 play 方法
    THREE.AnimationHandler.play(this)
};

// 定义 KeyFrameAnimation 对象的 stop 方法
THREE.KeyFrameAnimation.prototype.stop=function(){
    // 设置为暂停状态
    this.isPaused=this.isPlaying=!1;
    // 调用 AnimationHandler 的 stop 方法
    THREE.AnimationHandler.stop(this);
    // 遍历层级
    for(var a=0;a<this.data.hierarchy.length;a++){
        var b=this.hierarchy[a],
            c=this.data.hierarchy[a];
        // 如果动画缓存存在，则恢复原始矩阵
        if(void 0!==c.animationCache){
            var d=c.animationCache.originalMatrix;
            d.copy(b.matrix);
            b.matrix=d;
            delete c.animationCache
        }
    }
};

// 定义 KeyFrameAnimation 对象的 update 方法
THREE.KeyFrameAnimation.prototype.update=function(a){
    // 如果动画正在播放
    if(!1!==this.isPlaying){
        // 更新当前时间
        this.currentTime+=a*this.timeScale;
        a=this.data.length;
        // 如果循环播放并且当前时间大于总时间，则重新开始播放
        !0===this.loop&&this.currentTime>a&&(this.currentTime%=a);
        this.currentTime=Math.min(this.currentTime,a);
        a=0;
        // 遍历层级
        for(var b=this.hierarchy.length;a<b;a++){
            var c=this.hierarchy[a],
                d=this.data.hierarchy[a],
                e=d.keys,
                d=d.animationCache;
            if(e.length){
                var f=d.prevKey,
                    g=d.nextKey;
                // 根据当前时间插值计算动画状态
                if(g.time<=this.currentTime){
                    for(;g.time<this.currentTime&&g.index>f.index;)
                        f=g,
                        g=e[f.index+1];
                    d.prevKey=f;
                    d.nextKey=g
                }
                g.time>=this.currentTime?f.interpolate(g,this.currentTime):f.interpolate(g,g.time);
                this.data.hierarchy[a].node.updateMatrix();
                c.matrixWorldNeedsUpdate=!0
            }
        }
    }
};

// 定义 KeyFrameAnimation 对象的 getNextKeyWith 方法
THREE.KeyFrameAnimation.prototype.getNextKeyWith=function(a,b,c){
    // 获取指定层级的关键帧
    b=this.data.hierarchy[b].keys;
    // 循环查找下一个关键帧
    for(c%=b.length;c<b.length;c++)
        if(b[c].hasTarget(a))
            return b[c];
    return b[0];
};
# 在THREE.KeyFrameAnimation原型上添加getPrevKeyWith方法，用于获取指定目标的前一个关键帧
THREE.KeyFrameAnimation.prototype.getPrevKeyWith=function(a,b,c){
    # 获取动画数据中指定层级的关键帧数组
    b=this.data.hierarchy[b].keys;
    # 如果传入的索引小于0，则将其转换为正数
    for(c=0<=c?c:c+b.length;0<=c;c--)
        # 如果关键帧中包含指定目标，则返回该关键帧
        if(b[c].hasTarget(a))
            return b[c];
    # 如果未找到指定目标的关键帧，则返回最后一个关键帧
    return b[b.length-1];
};

# 定义THREE.MorphAnimation构造函数
THREE.MorphAnimation=function(a){
    # 为传入的网格对象设置属性
    this.mesh=a;
    # 获取网格对象的形态目标影响长度
    this.frames=a.morphTargetInfluences.length;
    # 设置当前时间为0
    this.currentTime=0;
    # 设置动画持续时间为1000毫秒
    this.duration=1E3;
    # 设置循环播放
    this.loop=!0;
    # 设置动画播放状态为未播放
    this.isPlaying=!1;
};

# 定义THREE.MorphAnimation原型方法
THREE.MorphAnimation.prototype={
    # 播放动画
    play:function(){
        this.isPlaying=!0
    },
    # 暂停动画
    pause:function(){
        this.isPlaying=!1
    },
    # 更新动画
    update:function(){
        # 初始化变量a和b
        var a=0,b=0;
        return function(c){
            # 如果动画正在播放
            if(!1!==this.isPlaying){
                # 更新当前时间
                this.currentTime+=c;
                # 如果循环播放且当前时间超过持续时间，则取余数
                !0===this.loop&&this.currentTime>this.duration&&(this.currentTime%=this.duration);
                # 限制当前时间不超过持续时间
                this.currentTime=Math.min(this.currentTime,this.duration);
                # 计算每帧的时间间隔
                c=this.duration/this.frames;
                # 计算当前帧索引
                var d=Math.floor(this.currentTime/c);
                # 如果当前帧索引不等于上一帧索引
                d!=b&&(
                    # 重置上一帧的形态目标影响为0
                    this.mesh.morphTargetInfluences[a]=0,
                    # 设置上一帧的形态目标影响为1
                    this.mesh.morphTargetInfluences[b]=1,
                    # 设置当前帧的形态目标影响为0
                    this.mesh.morphTargetInfluences[d]=0,
                    # 更新上一帧索引为当前帧索引
                    a=b,
                    # 更新当前帧索引
                    b=d
                );
                # 计算当前帧的形态目标影响
                this.mesh.morphTargetInfluences[d]=this.currentTime%c/c;
                # 计算上一帧的形态目标影响
                this.mesh.morphTargetInfluences[a]=1-this.mesh.morphTargetInfluences[d]
            }
        }
    }
};

# 定义THREE.BoxGeometry构造函数
THREE.BoxGeometry=function(a,b,c,d,e,f){
    # 定义内部函数g
    function g(a,b,c,d,e,f,g,s){
        # 获取宽度和高度的分段数
        var v=h.widthSegments,y=h.heightSegments;
        # 计算宽度和高度的一半
        G=e/2;
        w=f/2;
        # 获取顶点数组的长度
        K=h.vertices.length;
        # 根据不同的平面方向设置变量u和v
        if("x"===a&&"y"===b||"y"===a&&"x"===b)
            u="z";
        else if("x"===a&&"z"===b||"z"===a&&"x"===b)
            u="y",y=h.depthSegments;
        else if("z"===a&&"y"===b||"y"===a&&"z"===b)
            u="x",v=h.depthSegments;
        # 计算顶点的位置并添加到顶点数组中
        var x=v+1,D=y+1,E=e/v,A=f/y,B=new THREE.Vector3;
        B[u]=0<g?1:-1;
        for(e=0;e<D;e++)
            for(f=0;f<x;f++){
                var F=new THREE.Vector3;
                F[a]=(f*E-G)*c;
                F[b]=(e*A-w)*d;
                F[u]=g;
                h.vertices.push(F)
            }
        for(e=
// 循环遍历生成顶点和面
for (e = 0; e < y; e++) {
    for (f = 0; f < v; f++) {
        // 计算顶点索引
        w = f + x * e;
        a = f + x * (e + 1);
        b = f + 1 + x * (e + 1);
        c = f + 1 + x * e;
        // 创建顶点的纹理坐标
        d = new THREE.Vector2(f / v, 1 - e / y);
        g = new THREE.Vector2(f / v, 1 - (e + 1) / y);
        u = new THREE.Vector2((f + 1) / v, 1 - (e + 1) / y);
        G = new THREE.Vector2((f + 1) / v, 1 - e / y);
        // 创建面对象
        w = new THREE.Face3(w + K, a + K, c + K);
        w.normal.copy(B);
        w.vertexNormals.push(B.clone(), B.clone(), B.clone());
        w.materialIndex = s;
        h.faces.push(w);
        h.faceVertexUvs[0].push([d, g, G]);
        // 创建面对象
        w = new THREE.Face3(a + K, b + K, c + K);
        w.normal.copy(B);
        w.vertexNormals.push(B.clone(), B.clone(), B.clone());
        w.materialIndex = s;
        h.faces.push(w);
        h.faceVertexUvs[0].push([g.clone(), u, G.clone()]);
    }
}
// 设置几何体参数
THREE.Geometry.call(this);
this.type = "BoxGeometry";
this.parameters = { width: a, height: b, depth: c, widthSegments: d, heightSegments: e, depthSegments: f };
this.widthSegments = d || 1;
this.heightSegments = e || 1;
this.depthSegments = f || 1;
var h = this;
d = a / 2;
e = b / 2;
f = c / 2;
// 生成顶点和面
g("z", "y", -1, -1, c, b, d, 0);
g("z", "y", 1, -1, c, b, -d, 1);
g("x", "z", 1, 1, a, c, e, 2);
g("x", "z", 1, -1, a, c, -e, 3);
g("x", "y", 1, -1, a, b, f, 4);
g("x", "y", -1, -1, a, b, -f, 5);
// 合并顶点
this.mergeVertices();
};
// 设置原型链
THREE.BoxGeometry.prototype = Object.create(THREE.Geometry.prototype);
// 创建圆形几何体
THREE.CircleGeometry = function (a, b, c, d) {
    THREE.Geometry.call(this);
    this.type = "CircleGeometry";
    this.parameters = { radius: a, segments: b, thetaStart: c, thetaLength: d };
    a = a || 50;
    b = (b !== undefined) ? Math.max(3, b) : 8;
    c = (c !== undefined) ? c : 0;
    d = (d !== undefined) ? d : 2 * Math.PI;
    var e, f = [];
    e = new THREE.Vector3;
    var g = new THREE.Vector2(0.5, 0.5);
    this.vertices.push(e);
    f.push(g);
    for (e = 0; e <= b; e++) {
        var h = new THREE.Vector3, k = c + e / b * d;
        h.x = a * Math.cos(k);
        h.y = a * Math.sin(k);
        this.vertices.push(h);
        f.push(new THREE.Vector2((h.x / a + 1) / 2, (h.y / a + 1) / 2));
    }
    c = new THREE.Vector3(0, 0, 0);
}
# 创建一个立方体几何体对象
THREE.CubeGeometry=function(a,b,c,d,e,f){
    console.warn("THREE.CubeGeometry has been renamed to THREE.BoxGeometry.");
    return new THREE.BoxGeometry(a,b,c,d,e,f);
};

# 创建一个圆柱几何体对象
THREE.CylinderGeometry=function(a,b,c,d,e,f){
    # 继承自 THREE.Geometry
    THREE.Geometry.call(this);
    this.type="CylinderGeometry";
    this.parameters={radiusTop:a,radiusBottom:b,height:c,radialSegments:d,heightSegments:e,openEnded:f};
    a=void 0!==a?a:20;
    b=void 0!==b?b:20;
    c=void 0!==c?c:100;
    d=d||8;
    e=e||1;
    f=void 0!==f?f:!1;
    var g=c/2,h,k,n=[],p=[];
    for(k=0;k<=e;k++){
        var q=[],m=[],r=k/e,t=r*(b-a)+a;
        for(h=0;h<=d;h++){
            var s=h/d,u=new THREE.Vector3;
            u.x=t*Math.sin(s*Math.PI*2);
            u.y=-r*c+g;
            u.z=t*Math.cos(s*Math.PI*2);
            this.vertices.push(u);
            q.push(this.vertices.length-1);
            m.push(new THREE.Vector2(s,1-r));
        }
        n.push(q);
        p.push(m);
    }
    c=(b-a)/c;
    for(h=0;h<d;h++){
        if(0!==a){
            q=this.vertices[n[0][h]].clone();
            m=this.vertices[n[0][h+1]].clone();
        }else{
            q=this.vertices[n[1][h]].clone();
            m=this.vertices[n[1][h+1]].clone();
        }
        q.setY(Math.sqrt(q.x*q.x+q.z*q.z)*c).normalize();
        m.setY(Math.sqrt(m.x*m.x+m.z*m.z)*c).normalize();
        for(k=0;k<e;k++){
            var r=n[k][h],t=n[k+1][h],s=n[k+1][h+1],u=n[k][h+1],v=q.clone(),y=q.clone(),G=m.clone(),w=m.clone(),K=p[k][h].clone(),x=p[k+1][h].clone(),D=p[k+1][h+1].clone(),
E=p[k][h+1].clone();  # 从数组 p 中获取元素，并克隆
this.faces.push(new THREE.Face3(r,t,u,[v,y,w]));  # 将一个三角面添加到几何体的面数组中
this.faceVertexUvs[0].push([K,x,E]);  # 将纹理坐标添加到几何体的纹理坐标数组中
this.faces.push(new THREE.Face3(t,s,u,[y.clone(),G,w.clone()]));  # 将一个三角面添加到几何体的面数组中
this.faceVertexUvs[0].push([x.clone(),D,E.clone()]);  # 将纹理坐标添加到几何体的纹理坐标数组中
}if(!1===f&&0<a)for(this.vertices.push(new THREE.Vector3(0,g,0)),h=0;h<d;h++)r=n[0][h],t=n[0][h+1],s=this.vertices.length-1,v=new THREE.Vector3(0,1,0),y=new THREE.Vector3(0,1,0),G=new THREE.Vector3(0,1,0),K=p[0][h].clone(),x=p[0][h+1].clone(),D=new THREE.Vector2(x.x,0),this.faces.push(new THREE.Face3(r, t, s, [v, y, G])),this.faceVertexUvs[0].push([K, x, D]);  # 如果条件满足，则执行循环内的操作
if(!1===f&&0<b)for(this.vertices.push(new THREE.Vector3(0,-g,0)),h=0;h<d;h++)r=n[k][h+1],t=n[k][h],s=this.vertices.length-1,v=new THREE.Vector3(0,-1,0),y=new THREE.Vector3(0,-1,0),G=new THREE.Vector3(0,-1,0),K=p[k][h+1].clone(),x=p[k][h].clone(),D=new THREE.Vector2(x.x,1),this.faces.push(new THREE.Face3(r,t,s,[v,y,G])),this.faceVertexUvs[0].push([K,x,D]);  # 如果条件满足，则执行循环内的操作
this.computeFaceNormals()};  # 计算几何体的面法线
THREE.CylinderGeometry.prototype=Object.create(THREE.Geometry.prototype);  # 创建 CylinderGeometry 对象并继承 Geometry 对象
THREE.ExtrudeGeometry=function(a,b){"undefined"!==typeof a&&(THREE.Geometry.call(this),this.type="ExtrudeGeometry",a=a instanceof Array?a:[a],this.addShapeList(a,b),this.computeFaceNormals())};  # 创建 ExtrudeGeometry 对象
THREE.ExtrudeGeometry.prototype=Object.create(THREE.Geometry.prototype);  # 创建 ExtrudeGeometry 对象并继承 Geometry 对象
THREE.ExtrudeGeometry.prototype.addShapeList=function(a,b){for(var c=a.length,d=0;d<c;d++)this.addShape(a[d],b)};  # 将形状列表添加到 ExtrudeGeometry 对象中
THREE.ExtrudeGeometry.prototype.addShape=function(a,b){function c(a,b,c){b||console.log("die");return b.clone().multiplyScalar(c).add(a)}function d(a,b,c){var d=1,d=a.x-b.x,e=a.y-b.y,f=c.x-a.x,g=c.y-a.y,h=d*d+e*e;if(1E-10<Math.abs(d*g-e*f)){var k=Math.sqrt(h),m=Math.sqrt(f*f+g*g),h=b.x-e/k;b=b.y+d/k;f=((c.x-g/m-h)*g-(c.y+f/m-b)*f)/(d*g-e*f);c=h+d*f-a.x;a=b+e*f-a.y;d=c*c+a*a;if(2>=d)return new THREE.Vector2(c,a);d=Math.sqrt(d/2)}else a=!1,1E-10<d?1E-10<f&&(a=!0):-1E-10>d?-1E-10>f&&(a=!0):Math.sign(e)==  # 定义函数 d
# 定义函数e，接受参数a和b
function e(a,b){
    # 定义变量c和d
    var c,d;
    # 循环遍历a数组
    for(P=a.length;0<=--P;){
        c=P;
        d=P-1;
        # 如果d小于0，则d等于a数组的长度减1
        0>d&&(d=a.length-1);
        # 定义变量e和f
        for(var e=0,f=r+2*p,e=0;e<f;e++){
            var g=la*e,h=la*(e+1),k=b+c+g,g=b+d+g,m=b+d+h,h=b+c+h,k=k+R,g=g+R,m=m+R,h=h+R;
            # 将三角面添加到F.faces数组中
            F.faces.push(new THREE.Face3(k,g,h,null,null,y));
            F.faces.push(new THREE.Face3(g,m,h,null,null,y));
            # 生成侧面的UV坐标，并添加到F.faceVertexUvs[0]数组中
            k=G.generateSideWallUV(F,k,g,m,h);
            F.faceVertexUvs[0].push([k[0],k[1],k[3]]);
            F.faceVertexUvs[0].push([k[1],k[2],k[3]);
        }
    }
}
# 定义函数f，接受参数a、b和c
function f(a,b,c){
    # 将三维坐标点添加到F.vertices数组中
    F.vertices.push(new THREE.Vector3(a,b,c));
}
# 定义函数g，接受参数a、b和c
function g(a,b,c){
    # 将三维坐标点的索引加上R，并添加到F.faces数组中
    a+=R;
    b+=R;
    c+=R;
    F.faces.push(new THREE.Face3(a,b,c,null,null,v));
    # 生成顶部的UV坐标，并添加到F.faceVertexUvs[0]数组中
    a=G.generateTopUV(F,a,b,c);
    F.faceVertexUvs[0].push(a);
}
# 定义变量h，如果b对象中有amount属性则取其值，否则默认为100
var h=void 0!==b.amount?b.amount:100;
# 定义变量k，如果b对象中有bevelThickness属性则取其值，否则默认为6
var k=void 0!==b.bevelThickness?b.bevelThickness:6;
# 定义变量n，如果b对象中有bevelSize属性则取其值，否则默认为k-2
var n=void 0!==b.bevelSize?b.bevelSize:k-2;
# 定义变量p，如果b对象中有bevelSegments属性则取其值，否则默认为3
var p=void 0!==b.bevelSegments?b.bevelSegments:3;
# 定义变量q，如果b对象中有bevelEnabled属性则取其值，否则默认为true
var q=void 0!==b.bevelEnabled?b.bevelEnabled:!0;
# 定义变量m，如果b对象中有curveSegments属性则取其值，否则默认为12
var m=void 0!==b.curveSegments?b.curveSegments:12;
# 定义变量r，如果b对象中有steps属性则取其值，否则默认为1
var r=void 0!==b.steps?b.steps:1;
# 定义变量t，如果b对象中有extrudePath属性则取其值，否则默认为undefined
var t=b.extrudePath;
# 定义变量s，u，v，y，G，w，K，x，D
var s,u=!1,v=b.material,y=b.extrudeMaterial,G=void 0!==b.UVGenerator?b.UVGenerator:THREE.ExtrudeGeometry.WorldUVGenerator,w,K,x,D;
# 如果t存在，则执行以下操作
t&&(s=t.getSpacedPoints(r),u=!0,q=!1,w=void 0!==b.frames?b.frames:new THREE.TubeGeometry.FrenetFrames(t,r,!1),K=new THREE.Vector3,x=new THREE.Vector3,D=new THREE.Vector3);
# 如果q为false，则将n、k、p都设为0
q||(n=k=p=0);
# 定义变量E、A、B、F，分别为a的提取点、形状、孔、this.vertices.length
var E,A,B,F=this,R=this.vertices.length;
# 提取a的点和孔
t=a.extractPoints(m);
m=t.shape;
H=t.holes;
# 如果t为false，则将m反转
if(t=!THREE.Shape.Utils.isClockWise(m)){
    m=m.reverse();
    A=0;
    for(B=H.length;A<B;A++)
        E=H[A],
        THREE.Shape.Utils.isClockWise(E)&&
}
// 将数组 H 的元素进行反转，并赋值给数组 A
(H[A]=E.reverse());t=!1}
// 使用 THREE.Shape.Utils.triangulateShape 方法对数组 m 进行三角化处理，并赋值给数组 C
var C=THREE.Shape.Utils.triangulateShape(m,H),T=m;
// 初始化变量 A 为 0，循环条件为 A 小于数组 H 的长度，每次循环 A 自增
A=0;for(B=H.length;A<B;A++)E=H[A],m=m.concat(E);
// 初始化变量 Q、O、S、X、Y、la、ma、ya、t、P，并赋值
var Q,O,S,X,Y,la=m.length,ma,ya=C.length,t=[],P=0;S=T.length;Q=S-1;
// 循环条件为 O 小于 S，每次循环 O 自增
for(O=P+1;P<S;P++,Q++,O++)Q===S&&(Q=0),O===S&&(O=0),t[P]=d(T[P],T[Q],T[O]);
// 初始化数组 Ga，并赋值
var Ga=[],Fa,za=t.concat();A=0;for(B=H.length;A<B;A++){E=H[A];Fa=[];P=0;S=E.length;Q=S-1;
// 循环条件为 O 小于 S，每次循环 O 自增
for(O=P+1;P<S;P++,Q++,O++)Q===S&&(Q=0),O===S&&(O=0),Fa[P]=d(E[P],E[Q],E[O]);Ga.push(Fa);za=za.concat(Fa)}
// 循环条件为 Q 小于 p，每次循环 Q 自增
for(Q=0;Q<p;Q++){S=Q/p;X=k*(1-S);O=n*Math.sin(S*Math.PI/2);P=0;for(S=T.length;P<S;P++)Y=c(T[P],t[P],O),f(Y.x,Y.y,-X);A=0;for(B=H.length;A<B;A++)for(E=H[A],Fa=Ga[A],P=0,S=E.length;P<S;P++)Y=c(E[P],Fa[P],O),f(Y.x,Y.y,-X)}
// 初始化变量 O 为 n，循环条件为 P 小于 la，每次循环 P 自增
O=n;for(P=0;P<la;P++)Y=q?c(m[P],za[P],O):m[P],u?(x.copy(w.normals[0]).multiplyScalar(Y.x),K.copy(w.binormals[0]).multiplyScalar(Y.y),D.copy(s[0]).add(x).add(K),f(D.x,D.y,D.z)):f(Y.x,Y.y,0);
// 循环条件为 S 小于等于 r，每次循环 S 自增
for(S=1;S<=r;S++)for(P=0;P<la;P++)Y=q?c(m[P],za[P],O):m[P],u?(x.copy(w.normals[S]).multiplyScalar(Y.x),K.copy(w.binormals[S]).multiplyScalar(Y.y),D.copy(s[S]).add(x).add(K),f(D.x,D.y,D.z)):f(Y.x,Y.y,h/r*S);
// 循环条件为 Q 大于等于 0，每次循环 Q 自减
for(Q=p-1;0<=Q;Q--){S=Q/p;X=k*(1-S);O=n*Math.sin(S*Math.PI/2);P=0;for(S=T.length;P<S;P++)Y=c(T[P],t[P],O),f(Y.x,Y.y,h+X);A=0;for(B=H.length;A<B;A++)for(E=H[A],Fa=Ga[A],P=0,S=E.length;P<S;P++)Y=c(E[P],Fa[P],O),u?f(Y.x,Y.y+s[r-1].y,s[r-1].x+X):f(Y.x,Y.y,h+X)}
// 匿名函数
(function(){if(q){var a;a=0*la;for(P=0;P<ya;P++)ma=C[P],g(ma[2]+a,ma[1]+a,ma[0]+a);a=r+2*p;a*=la;for(P=0;P<ya;P++)ma=C[P],g(ma[0]+a,ma[1]+a,ma[2]+a)}else{for(P=0;P<ya;P++)ma=C[P],g(ma[2],ma[1],ma[0]);for(P=0;P<ya;P++)ma=C[P],g(ma[0]+la*r,ma[1]+la*r,ma[2]+la*r)}})();
// 匿名函数
(function(){var a=0;e(T,a);a+=T.length;A=0;for(B=H.length;A<B;A++)E=H[A],e(E,a),a+=E.length})();
# 定义一个名为 WorldUVGenerator 的对象，包含两个方法：generateTopUV 和 generateSideWallUV
THREE.ExtrudeGeometry.WorldUVGenerator={
    generateTopUV:function(a,b,c,d){
        a=a.vertices;
        b=a[b];
        c=a[c];
        d=a[d];
        return[new THREE.Vector2(b.x,b.y),new THREE.Vector2(c.x,c.y),new THREE.Vector2(d.x,d.y)]
    },
    generateSideWallUV:function(a,b,c,d,e){
        a=a.vertices;
        b=a[b];
        c=a[c];
        d=a[d];
        e=a[e];
        return.01>Math.abs(b.y-c.y)?
            [new THREE.Vector2(b.x,1-b.z),new THREE.Vector2(c.x,1-c.z),new THREE.Vector2(d.x,1-d.z),new THREE.Vector2(e.x,1-e.z)]:
            [new THREE.Vector2(b.y,1-b.z),new THREE.Vector2(c.y,1-c.z),new THREE.Vector2(d.y,1-d.z),new THREE.Vector2(e.y,1-e.z)]
    }
};

# 定义一个名为 ShapeGeometry 的函数，继承自 Geometry 类
THREE.ShapeGeometry=function(a,b){
    THREE.Geometry.call(this);
    this.type="ShapeGeometry";
    !1===a instanceof Array&&(a=[a]);
    this.addShapeList(a,b);
    this.computeFaceNormals()
};

# 将 ShapeGeometry 的原型指向 Geometry 的实例
THREE.ShapeGeometry.prototype=Object.create(THREE.Geometry.prototype);

# 给 ShapeGeometry 添加 addShapeList 方法
THREE.ShapeGeometry.prototype.addShapeList=function(a,b){
    for(var c=0,d=a.length;c<d;c++)
        this.addShape(a[c],b);
    return this
};

# 给 ShapeGeometry 添加 addShape 方法
THREE.ShapeGeometry.prototype.addShape=function(a,b){
    void 0===b&&(b={});
    var c=b.material,
        d=void 0===b.UVGenerator?THREE.ExtrudeGeometry.WorldUVGenerator:b.UVGenerator,
        e,f,g,h=this.vertices.length;
    e=a.extractPoints(void 0!==b.curveSegments?b.curveSegments:12);
    var k=e.shape,
        n=e.holes;
    if(!THREE.Shape.Utils.isClockWise(k))
        for(k=k.reverse(),e=0,f=n.length;e<f;e++)
            g=n[e],
            THREE.Shape.Utils.isClockWise(g)&&(n[e]=g.reverse());
    var p=THREE.Shape.Utils.triangulateShape(k,n);
    e=0;
    for(f=n.length;e<f;e++)
        g=n[e],
        k=k.concat(g);
    n=k.length;
    f=p.length;
    for(e=0;e<n;e++)
        g=k[e],
        this.vertices.push(new THREE.Vector3(g.x,g.y,0));
    for(e=0;e<f;e++)
        n=p[e],
        k=n[0]+h,
        g=n[1]+h,
        n=n[2]+h,
        this.faces.push(new THREE.Face3(k,g,n,null,null,c)),
        this.faceVertexUvs[0].push(d.generateTopUV(this,k,g,n))
};
// 创建 LatheGeometry 类，继承自 Geometry 类
THREE.LatheGeometry=function(a,b,c,d){THREE.Geometry.call(this);this.type="LatheGeometry";this.parameters={points:a,segments:b,phiStart:c,phiLength:d};
// 设置默认参数
b=b||12;c=c||0;d=d||2*Math.PI;
// 根据点集合创建旋转体
for(var e=1/(a.length-1),f=1/b,g=0,h=b;g<=h;g++)
    for(var k=c+g*f*d,n=Math.cos(k),p=Math.sin(k),k=0,q=a.length;k<q;k++){
        var m=a[k],r=new THREE.Vector3;
        r.x=n*m.x-p*m.y;
        r.y=p*m.x+n*m.y;
        r.z=m.z;
        this.vertices.push(r)
    }
// 创建面
c=a.length;g=0;for(h=b;g<h;g++)
    for(k=0,q=a.length-1;k<q;k++){
        b=p=k+c*g;
        d=p+c;
        var n=p+1+c,p=p+1,m=g*f,r=k*e,t=m+f,s=r+e;
        this.faces.push(new THREE.Face3(b,d,p));
        this.faceVertexUvs[0].push([new THREE.Vector2(m,r),new THREE.Vector2(t,r),new THREE.Vector2(m,s)]);
        this.faces.push(new THREE.Face3(d,n,p));
        this.faceVertexUvs[0].push([new THREE.Vector2(t,r),new THREE.Vector2(t,s),new THREE.Vector2(m,s)])
    }
// 合并顶点
this.mergeVertices();
// 计算面法线
this.computeFaceNormals();
// 计算顶点法线
this.computeVertexNormals()};THREE.LatheGeometry.prototype=Object.create(THREE.Geometry.prototype);

// 创建 PlaneGeometry 类，继承自 Geometry 类
THREE.PlaneGeometry=function(a,b,c,d){
    console.info("THREE.PlaneGeometry: Consider using THREE.PlaneBufferGeometry for lower memory footprint.");
    THREE.Geometry.call(this);
    this.type="PlaneGeometry";
    this.parameters={width:a,height:b,widthSegments:c,heightSegments:d};
    this.fromBufferGeometry(new THREE.PlaneBufferGeometry(a,b,c,d))
};THREE.PlaneGeometry.prototype=Object.create(THREE.Geometry.prototype);

// 创建 PlaneBufferGeometry 类，继承自 BufferGeometry 类
THREE.PlaneBufferGeometry=function(a,b,c,d){
    THREE.BufferGeometry.call(this);
    this.type="PlaneBufferGeometry";
    this.parameters={width:a,height:b,widthSegments:c,heightSegments:d};
    var e=a/2,f=b/2;
    c=c||1;d=d||1;
    var g=c+1,h=d+1,k=a/c,n=b/d;
    b=new Float32Array(g*h*3);
    a=new Float32Array(g*h*3);
    for(var p=new Float32Array(g*h*2),q=0,m=0,r=0;r<h;r++)
        for(var t=r*n-f,s=0;s<g;s++)
            b[q]=s*k-e,b[q+1]=-t,a[q+2]=1,p[m]=s/c,p[m+1]=1-r/d,q+=3,m+=2;
    q=0;
    e=new (65535<b.length/3?Uint32Array:Uint16Array)(c*d*6);
    for(r=0;r<d;r++)
        for(s=
# 创建一个环形几何体
THREE.RingGeometry=function(a,b,c,d,e,f){
    # 调用父类的构造函数
    THREE.Geometry.call(this);
    # 设置几何体类型
    this.type="RingGeometry";
    # 设置几何体参数
    this.parameters={innerRadius:a,outerRadius:b,thetaSegments:c,phiSegments:d,thetaStart:e,thetaLength:f};
    # 设置默认值
    a=a||0;
    b=b||50;
    e=void 0!==e?e:0;
    f=void 0!==f?f:2*Math.PI;
    c=void 0!==c?Math.max(3,c):8;
    d=void 0!==d?Math.max(1,d):8;
    # 定义变量
    var g,h=[],k=a,n=(b-a)/d;
    # 循环计算顶点坐标
    for(a=0;a<d+1;a++){
        for(g=0;g<c+1;g++){
            var p=new THREE.Vector3,q=e+g/c*f;
            p.x=k*Math.cos(q);
            p.y=k*Math.sin(q);
            this.vertices.push(p);
            h.push(new THREE.Vector2((p.x/b+1)/2, (p.y/b+1)/2))
        }
        k+=n
    }
    # 定义变量
    var b=new THREE.Vector3(0,0,1);
    # 循环计算面和纹理坐标
    for(a=0;a<d;a++){
        for(e=a*(c+1),g=0;g<c;g++){
            f=q=g+e;
            n=q+c+1;
            p=q+c+2;
            this.faces.push(new THREE.Face3(f,n,p,[b.clone(),b.clone(),b.clone()]));
            this.faceVertexUvs[0].push([h[f].clone(),h[n].clone(),h[p].clone()]);
            f=q;
            n=q+c+2;
            p=q+1;
            this.faces.push(new THREE.Face3(f,n,p,[b.clone(),b.clone(),b.clone()]));
            this.faceVertexUvs[0].push([h[f].clone(),h[n].clone(),h[p].clone()])
        }
    }
    # 计算面法线
    this.computeFaceNormals();
    # 设置包围球
    this.boundingSphere=new THREE.Sphere(new THREE.Vector3,k)
};
# 设置原型链
THREE.RingGeometry.prototype=Object.create(THREE.Geometry.prototype);
// 计算球面上的顶点坐标
s.y=a*Math.cos(f+t*g);
s.z=a*Math.sin(d+r*e)*Math.sin(f+t*g);
// 将顶点添加到顶点数组中
this.vertices.push(s);
// 将顶点索引添加到数组中
q.push(this.vertices.length-1);
// 将顶点坐标添加到数组中
m.push(new THREE.Vector2(r,1-t))
n.push(q);
p.push(m)
// 遍历顶点数组，计算面的顶点索引
for(k=0;k<c;k++)
    for(h=0;h<b;h++){
        d=n[k][h+1];
        e=n[k][h];
        f=n[k+1][h];
        g=n[k+1][h+1];
        // 克隆并归一化顶点坐标
        var q=this.vertices[d].clone().normalize(),
        m=this.vertices[e].clone().normalize(),
        r=this.vertices[f].clone().normalize(),
        t=this.vertices[g].clone().normalize(),
        s=p[k][h+1].clone(),
        u=p[k][h].clone(),
        v=p[k+1][h].clone(),
        y=p[k+1][h+1].clone();
        // 根据顶点坐标的y值判断面的类型，并添加到对应数组中
        Math.abs(this.vertices[d].y)===a?(s.x=(s.x+u.x)/2,this.faces.push(new THREE.Face3(d,f,g,[q,r,t])),this.faceVertexUvs[0].push([s,v,y])):
        Math.abs(this.vertices[f].y)===a?(v.x=(v.x+y.x)/2,this.faces.push(new THREE.Face3(d,e,f,[q,m,r])),this.faceVertexUvs[0].push([s,u,v])):
        (this.faces.push(new THREE.Face3(d,e,g,[q,m,t])),this.faceVertexUvs[0].push([s,u,y]),this.faces.push(new THREE.Face3(e,f,g,[m.clone(),r,t.clone()])),this.faceVertexUvs[0].push([u.clone(),v,y.clone()]))
    }
// 计算面的法线
this.computeFaceNormals();
// 计算包围球
this.boundingSphere=new THREE.Sphere(new THREE.Vector3,a);
// 设置原型链
THREE.SphereGeometry.prototype=Object.create(THREE.Geometry.prototype);
// 创建文本几何体
THREE.TextGeometry=function(a,b){
    b=b||{};
    var c=THREE.FontUtils.generateShapes(a,b);
    b.amount=void 0!==b.height?b.height:50;
    void 0===b.bevelThickness&&(b.bevelThickness=10);
    void 0===b.bevelSize&&(b.bevelSize=8);
    void 0===b.bevelEnabled&&(b.bevelEnabled=!1);
    THREE.ExtrudeGeometry.call(this,c,b);
    this.type="TextGeometry"
};
// 设置原型链
THREE.TextGeometry.prototype=Object.create(THREE.ExtrudeGeometry.prototype);
# 创建一个名为TorusGeometry的函数，用于生成环面几何体
THREE.TorusGeometry=function(a,b,c,d,e){THREE.Geometry.call(this);
# 设置TorusGeometry对象的类型和参数
this.type="TorusGeometry";this.parameters={radius:a,tube:b,radialSegments:c,tubularSegments:d,arc:e};
# 设置默认参数值
a=a||100;b=b||40;c=c||8;d=d||6;e=e||2*Math.PI;
# 创建一个三维向量对象
for(var f=new THREE.Vector3,g=[],h=[],k=0;k<=c;k++)
    for(var n=0;n<=d;n++){
        var p=n/d*e,q=k/c*Math.PI*2;
        f.x=a*Math.cos(p);f.y=a*Math.sin(p);
        var m=new THREE.Vector3;
        m.x=(a+b*Math.cos(q))*Math.cos(p);m.y=(a+b*Math.cos(q))*Math.sin(p);m.z=b*Math.sin(q);
        this.vertices.push(m);
        g.push(new THREE.Vector2(n/d,k/c));
        h.push(m.clone().sub(f).normalize())
    }
# 创建环面的面
for(k=1;k<=c;k++)
    for(n=1;n<=d;n++){
        a=(d+1)*k+n-1;b=(d+1)*(k-1)+n-1;e=(d+1)*(k-1)+n;f=(d+1)*k+n;
        p=new THREE.Face3(a,b,f,[h[a].clone(),h[b].clone(),h[f].clone()]);this.faces.push(p);this.faceVertexUvs[0].push([g[a].clone(),g[b].clone(),g[f].clone()]);
        p=new THREE.Face3(b,e,f,[h[b].clone(),h[e].clone(),h[f].clone()]);this.faces.push(p);this.faceVertexUvs[0].push([g[b].clone(),g[e].clone(),g[f].clone()]);
    }
# 计算面的法线
this.computeFaceNormals()};
# 设置TorusGeometry的原型为THREE.Geometry的实例
THREE.TorusGeometry.prototype=Object.create(THREE.Geometry.prototype);
# 创建一个名为TorusKnotGeometry的函数，用于生成环面结的几何体
THREE.TorusKnotGeometry=function(a,b,c,d,e,f,g){
    function h(a,b,c,d,e){
        var f=Math.cos(a),g=Math.sin(a);
        a*=b/c;b=Math.cos(a);f*=d*(2+b)*.5;g=d*(2+b)*g*.5;d=e*d*Math.sin(a)*.5;
        return new THREE.Vector3(f,g,d)
    }
    THREE.Geometry.call(this);
    # 设置TorusKnotGeometry对象的类型和参数
    this.type="TorusKnotGeometry";this.parameters={radius:a,tube:b,radialSegments:c,tubularSegments:d,p:e,q:f,heightScale:g};
    # 设置默认参数值
    a=a||100;b=b||40;c=c||64;d=d||8;e=e||2;f=f||3;g=g||1;
    for(var k=Array(c),n=new THREE.Vector3,p=new THREE.Vector3,q=new THREE.Vector3,m=0;m<c;++m){k[m]=
// 创建一个数组d
Array(d);
// 定义变量r为m/c*2*e*Math.PI
var r=m/c*2*e*Math.PI,
    // 调用h函数，传入参数r,f,e,a,g，返回结果赋值给t
    t=h(r,f,e,a,g),
    // 调用h函数，传入参数r+.01,f,e,a,g，返回结果赋值给r
    r=h(r+.01,f,e,a,g);
// 计算向量n为r和t的差
n.subVectors(r,t);
// 计算向量p为r和t的和
p.addVectors(r,t);
// 计算向量q为n和p的叉乘
q.crossVectors(n,p);
// 计算向量p为q和n的叉乘
p.crossVectors(q,n);
// 对向量q进行归一化
q.normalize();
// 对向量p进行归一化
p.normalize();
// 循环，r从0到d-1
for(r=0;r<d;++r){
    // 计算参数s
    var s=r/d*2*Math.PI,
        // 计算参数u
        u=-b*Math.cos(s),
        // 计算参数s
        s=b*Math.sin(s),
        // 创建一个新的三维向量v
        v=new THREE.Vector3;
    // 计算向量v的坐标
    v.x=t.x+u*p.x+s*q.x;
    v.y=t.y+u*p.y+s*q.y;
    v.z=t.z+u*p.z+s*q.z;
    // 将向量v添加到this.vertices数组中，并返回索引值
    k[m][r]=this.vertices.push(v)-1
}
// 循环，m从0到c-1
for(m=0;m<c;++m)
    // 循环，r从0到d-1
    for(r=0;r<d;++r)
        // 计算参数e
        e=(m+1)%c,
        // 计算参数f
        f=(r+1)%d,
        // 获取k[m][r]的值，赋值给a
        a=k[m][r],
        // 获取k[e][r]的值，赋值给b
        b=k[e][r],
        // 获取k[e][f]的值，赋值给e
        e=k[e][f],
        // 获取k[m][f]的值，赋值给f
        f=k[m][f],
        // 创建一个新的二维向量g
        g=new THREE.Vector2(m/c,r/d),
        // 创建一个新的二维向量n
        n=new THREE.Vector2((m+1)/c,r/d),
        // 创建一个新的二维向量p
        p=new THREE.Vector2((m+1)/c,(r+1)/d),
        // 创建一个新的二维向量q
        q=new THREE.Vector2(m/c,(r+1)/d);
    // 将一个三角面添加到this.faces数组中
    this.faces.push(new THREE.Face3(a,b,f));
    // 将三角面的纹理坐标添加到this.faceVertexUvs[0]数组中
    this.faceVertexUvs[0].push([g,n,q]);
    // 将一个三角面添加到this.faces数组中
    this.faces.push(new THREE.Face3(b,e,f));
    // 将三角面的纹理坐标添加到this.faceVertexUvs[0]数组中
    this.faceVertexUvs[0].push([n.clone(),p,q.clone()]);
// 计算三角面的法线
this.computeFaceNormals();
// 计算顶点的法线
this.computeVertexNormals()
};
// 将THREE.TorusKnotGeometry的原型设置为THREE.Geometry的实例
THREE.TorusKnotGeometry.prototype=Object.create(THREE.Geometry.prototype);
// 定义THREE.TubeGeometry函数，传入参数a,b,c,d,e
THREE.TubeGeometry=function(a,b,c,d,e){
    // 调用THREE.Geometry构造函数
    THREE.Geometry.call(this);
    // 设置this.type属性为"TubeGeometry"
    this.type="TubeGeometry";
    // 设置this.parameters属性为传入的参数对象
    this.parameters={path:a,segments:b,radius:c,radialSegments:d,closed:e};
    // 如果参数b不存在，则设置b为64
    b=b||64;
    // 如果参数c不存在，则设置c为1
    c=c||1;
    // 如果参数d不存在，则设置d为8
    d=d||8;
    // 如果参数e不存在，则设置e为false
    e=e||!1;
    // 定义变量f为一个空数组
    var f=[],
        // 定义变量g,h,k,n,p,q,m,r,t,s,u
        g,h,k=b+1,n,p,q,m,r=new THREE.Vector3,t,s,u;
    // 调用a的getPointAt方法，传入参数n，返回结果赋值给m
    t=new THREE.TubeGeometry.FrenetFrames(a,b,e);
    // 设置this.tangents属性为t的tangents属性
    this.tangents=t.tangents;
    // 设置this.normals属性为t的normals属性
    this.normals=s;
    // 设置this.binormals属性为t的binormals属性
    this.binormals=u;
    // 循环，t从0到k-1
    for(t=0;t<k;t++)
        // 定义f[t]为一个空数组
        f[t]=[],
        // 计算参数n
        n=t/(k-1),
        // 调用a的getPointAt方法，传入参数n，返回结果赋值给m
        m=a.getPointAt(n),
        // 获取s[t]的值，赋值给g
        g=s[t],
        // 获取u[t]的值，赋值给h
        h=u[t],
        // 循环，n从0到d-1
        for(n=0;n<d;n++)
            // 计算参数p
            p=n/d*2*Math.PI,
            // 计算参数q
            q=-c*Math.cos(p),
            // 计算参数p
            p=c*Math.sin(p),
            // 复制向量m的值给r
            r.copy(m),
            // 计算向量r的坐标
            r.x+=q*g.x+p*h.x,
            r.y+=q*g.y+p*h.y,
            r.z+=q*g.z+p*h.z,
            // 将向量r添加到this.vertices数组中，并返回索引值
            f[t][n]=this.vertices.push(new THREE.Vector3(r.x,r.y,r.z))-1;
    // 循环，t从0到b-1
    for(t=0;t<b;t++)
        // 循环，n从0到d-1
        for(n=0;n<d;n++)
            // 计算参数k
            k=e?(t+1)%b:t+1,
            // 计算参数r
            r=(n+1)%d,
            // 获取f[t][n]的值，赋值给a
            a=f[t][n],
            // 获取f[k][n]的值，赋值给c
            c=f[k][n],
            // 获取f[k][r]的值，赋值给k
            k=f[k][r],
            // 获取f[t][r]的值，赋值给r
            r=f[t][r],
            // 创建一个新的二维向量s
            s=new THREE.Vector2(t/b,n/d),
            // 创建一个新的二维向量u
            u=new THREE.Vector2((t+1)/b,n/d),
            // 创建一个新的二维向量g
            g=new THREE.Vector2((t+1)/b,(n+1)/d),
            // 创建一个新的二维向量h
            h=new THREE.Vector2(t/b,(n+1)/d);
        // 将一个三角面添加到this.faces数组中
        this.faces.push(new THREE.Face3(a,c,r));
        // 将三角面的纹理坐标添加到this.faceVertexUvs[0]数组中
        this.faceVertexUvs[0].push([s,u,h]);
        // 将一个三角面添加到this.faces数组中
        this.faces.push(new THREE.Face3(c,k,r));
        // 将三角面的纹理坐标添加到this.faceVertexUvs[0]数组中
        this.faceVertexUvs[0].push([u.clone(),
# 将 TubeGeometry 原型设置为继承自 Geometry 原型
THREE.TubeGeometry.prototype=Object.create(THREE.Geometry.prototype);
# 计算 FrenetFrames
THREE.TubeGeometry.FrenetFrames=function(a,b,c){
    new THREE.Vector3;
    var d=new THREE.Vector3;
    new THREE.Vector3;
    var e=[],f=[],g=[],
    h=new THREE.Vector3,
    k=new THREE.Matrix4;
    b+=1;
    var n,p,q;
    this.tangents=e;
    this.normals=f;
    this.binormals=g;
    # 计算切线
    for(n=0;n<b;n++)
        p=n/(b-1),
        e[n]=a.getTangentAt(p),
        e[n].normalize();
    f[0]=new THREE.Vector3;
    g[0]=new THREE.Vector3;
    a=Number.MAX_VALUE;
    n=Math.abs(e[0].x);
    p=Math.abs(e[0].y);
    q=Math.abs(e[0].z);
    # 计算法线和副法线
    n<=a&&(a=n,d.set(1,0,0));
    p<=a&&(a=p,d.set(0,1,0));
    q<=a&&d.set(0,0,1);
    h.crossVectors(e[0],d).normalize();
    f[0].crossVectors(e[0],h);
    g[0].crossVectors(e[0],f[0]);
    # 计算剩余的法线和副法线
    for(n=1;n<b;n++)
        f[n]=f[n-1].clone(),
        g[n]=g[n-1].clone(),
        h.crossVectors(e[n-1],e[n]),
        1E-4<h.length()&&(h.normalize(),d=Math.acos(THREE.Math.clamp(e[n-1].dot(e[n]),-1,1)),f[n].applyMatrix4(k.makeRotationAxis(h,d))),
        g[n].crossVectors(e[n],f[n]);
    # 如果需要计算顶点法线
    if(c)
        for(d=Math.acos(THREE.Math.clamp(f[0].dot(f[b-1]),-1,1)),d/=b-1,0<e[0].dot(h.crossVectors(f[0],f[b-1]))&&(d=-d),n=1;n<b;n++)
            f[n].applyMatrix4(k.makeRotationAxis(e[n],d*n)),
            g[n].crossVectors(e[n],f[n]);
};
# 创建 PolyhedronGeometry
THREE.PolyhedronGeometry=function(a,b,c,d){
    function e(a){
        var b=a.normalize().clone();
        b.index=k.vertices.push(b)-1;
        var c=Math.atan2(a.z,-a.x)/2/Math.PI+.5;
        a=Math.atan2(-a.y,Math.sqrt(a.x*a.x+a.z*a.z))/Math.PI+.5;
        b.uv=new THREE.Vector2(c,1-a);
        return b
    }
    function f(a,b,c){
        var d=new THREE.Face3(a.index,b.index,c.index,[a.clone(),b.clone(),c.clone()]);
        k.faces.push(d);
        u.copy(a).add(b).add(c).divideScalar(3);
        d=Math.atan2(u.z,-u.x);
        k.faceVertexUvs[0].push([h(a.uv,a,d),h(b.uv,b,d),h(c.uv,c,d)])
    }
    function g(a,b){
        var c=
# 使用 Math.pow 函数计算 2 的 b 次方
Math.pow(2,b);
# 使用 Math.pow 函数计算 4 的 b 次方
Math.pow(4,b);
# 遍历顶点数组，计算插值点
for(var d=e(k.vertices[a.a]),g=e(k.vertices[a.b]),h=e(k.vertices[a.c]),m=[],n=0;n<=c;n++){
    m[n]=[];
    for(var p=e(d.clone().lerp(h,n/c)),q=e(g.clone().lerp(h,n/c)),r=c-n,s=0;s<=r;s++){
        m[n][s]=0==s&&n==c?p:e(p.clone().lerp(q,s/r))
    }
}
# 遍历插值点数组，计算面的顶点索引
for(n=0;n<c;n++){
    for(s=0;s<2*(c-n)-1;s++){
        d=Math.floor(s/2);
        0==s%2?f(m[n][d+1],m[n+1][d],m[n][d]):f(m[n][d+1],m[n+1][d+1],m[n+1][d])
    }
}
# 定义函数 h，对三维向量进行处理
function h(a,b,c){
    0>c&&1===a.x&&(a=new THREE.Vector2(a.x-1,a.y));
    0===b.x&&0===b.z&&(a=new THREE.Vector2(c/2/Math.PI+.5,a.y));
    return a.clone()
}
# 定义 PolyhedronGeometry 类
THREE.Geometry.call(this);
this.type="PolyhedronGeometry";
this.parameters={vertices:a,indices:b,radius:c,detail:d};
c=c||1;
d=d||0;
# 遍历顶点数组，创建三维向量
for(var k=this,n=0,p=a.length;n<p;n+=3){
    e(new THREE.Vector3(a[n],a[n+1],a[n+2]))
}
a=this.vertices;
# 创建面数组
for(var q=[],m=n=0,p=b.length;n<p;n+=3,m++){
    var r=a[b[n]],t=a[b[n+1]],s=a[b[n+2]];
    q[m]=new THREE.Face3(r.index,t.index,s.index,[r.clone(),t.clone(),s.clone()])
}
# 对面数组进行处理
for(var u=new THREE.Vector3,n=0,p=q.length;n<p;n++){
    g(q[n],d)
}
# 对纹理坐标进行处理
n=0;
for(p=this.faceVertexUvs[0].length;n<p;n++){
    b=this.faceVertexUvs[0][n];
    d=b[0].x;
    a=b[1].x;
    q=b[2].x;
    m=Math.max(d,Math.max(a,q));
    r=Math.min(d,Math.min(a,q));
    .9<m&&.1>r&&(.2>d&&(b[0].x+=1),.2>a&&(b[1].x+=1),.2>q&&(b[2].x+=1))
}
# 对顶点进行缩放
n=0;
for(p=this.vertices.length;n<p;n++){
    this.vertices[n].multiplyScalar(c)
}
# 合并顶点
this.mergeVertices();
# 计算面的法线
this.computeFaceNormals();
# 创建包围球
this.boundingSphere=new THREE.Sphere(new THREE.Vector3,c)
};
# 继承 PolyhedronGeometry 类
THREE.PolyhedronGeometry.prototype=Object.create(THREE.Geometry.prototype);
# 定义 DodecahedronGeometry 类
THREE.DodecahedronGeometry=function(a,b){
    this.parameters={radius:a,detail:b};
    var c=(1+Math.sqrt(5))/2,d=1/c;
    THREE.PolyhedronGeometry.call(this,[-1,-1,-1,-1,-1,1,-1,1,-1,-1,1,1,1,-1,-1,1,-1,1,1,1,-1,1,1,1,0,-d,-c,0,-d,c,0,d,-c,0,d,c,-d,-c,0,-d,c,0,d,-c,0,d,c,0,-c,0,-d,c,0,-d,-c,0,d,c,0,d],[3,11,7,3,7,15,3,15,13,7,19,17,7,17,6,7,6,15,17,4,8,17,8,10,17,10,6,8,0,16,8,16,2,8,2,10,0,12,1,0,1,18,0,18,16,6,10,2,6,2,13,6,13,15,2,16,18,2,18,3,2,3,13,18,1,9,18,9,11,18,11,3,4,14,12,4,12,0,4,0,8,11,9,5,11,5,19,
# 定义 DodecahedronGeometry 类，继承自 Geometry 类
THREE.DodecahedronGeometry = function(a, b) {
    var c = (1 + Math.sqrt(5)) / 2;
    THREE.PolyhedronGeometry.call(this, [-1, c, 0, 1, c, 0, -1, -c, 0, 1, -c, 0, 0, -1, c, 0, 1, c, 0, -1, -c, 0, 1, -c, c, 0, -1, c, 0, 1, -c, 0, -1, -c, 0, 1], [0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1], a, b);
    this.type = "DodecahedronGeometry";
    this.parameters = { radius: a, detail: b };
};
# 设置 DodecahedronGeometry 的原型为 Geometry 类的实例
THREE.DodecahedronGeometry.prototype = Object.create(THREE.Geometry.prototype);

# 定义 IcosahedronGeometry 类，继承自 Geometry 类
THREE.IcosahedronGeometry = function(a, b) {
    var c = (1 + Math.sqrt(5)) / 2;
    THREE.PolyhedronGeometry.call(this, [-1, c, 0, 1, c, 0, -1, -c, 0, 1, -c, 0, 0, -1, c, 0, 1, c, 0, -1, -c, 0, 1, -c, c, 0, -1, c, 0, 1, -c, 0, -1, -c, 0, 1], [0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1], a, b);
    this.type = "IcosahedronGeometry";
    this.parameters = { radius: a, detail: b };
};
# 设置 IcosahedronGeometry 的原型为 Geometry 类的实例
THREE.IcosahedronGeometry.prototype = Object.create(THREE.Geometry.prototype);

# 定义 OctahedronGeometry 类，继承自 Geometry 类
THREE.OctahedronGeometry = function(a, b) {
    this.parameters = { radius: a, detail: b };
    THREE.PolyhedronGeometry.call(this, [1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1], [0, 2, 4, 0, 4, 3, 0, 3, 5, 0, 5, 2, 1, 2, 5, 1, 5, 3, 1, 3, 4, 1, 4, 2], a, b);
    this.type = "OctahedronGeometry";
    this.parameters = { radius: a, detail: b };
};
# 设置 OctahedronGeometry 的原型为 Geometry 类的实例
THREE.OctahedronGeometry.prototype = Object.create(THREE.Geometry.prototype);

# 定义 TetrahedronGeometry 类，继承自 Geometry 类
THREE.TetrahedronGeometry = function(a, b) {
    THREE.PolyhedronGeometry.call(this, [1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1], [2, 1, 0, 0, 3, 2, 1, 3, 0, 2, 3, 1], a, b);
    this.type = "TetrahedronGeometry";
    this.parameters = { radius: a, detail: b };
};
# 设置 TetrahedronGeometry 的原型为 Geometry 类的实例
THREE.TetrahedronGeometry.prototype = Object.create(THREE.Geometry.prototype);

# 定义 ParametricGeometry 类，继承自 Geometry 类
THREE.ParametricGeometry = function(a, b, c) {
    THREE.Geometry.call(this);
    this.type = "ParametricGeometry";
    this.parameters = { func: a, slices: b, stacks: c };
    var d = this.vertices, e = this.faces, f = this.faceVertexUvs[0], g, h, k, n, p = b + 1;
    for (g = 0; g <= c; g++) {
        for (n = g / c, h = 0; h <= b; h++) {
            k = h / b;
            k = a(k, n);
            d.push(k);
        }
    }
    var q, m, r, t;
    for (g = 0; g < c; g++) {
        for (h = 0; h < b; h++) {
            a = g * p + h;
            d = g * p + h + 1;
            n = (g + 1) * p + h + 1;
            k = (g + 1) * p + h;
            q = new THREE.Vector2(h / b, g / c);
            m = new THREE.Vector2((h + 1) / b, g / c);
            r = new THREE.Vector2((h + 1) / b, (g + 1) / c);
            t = new THREE.Vector2(h / b, (g + 1) / c);
            e.push(new THREE.Face3(a, d, k));
            f.push([q, m, t]);
            e.push(new THREE.Face3(d, n, k));
            f.push([m.clone(), r, t.clone()]);
        }
    }
    this.computeFaceNormals();
    this.computeVertexNormals();
};
# 设置 ParametricGeometry 的原型为 Geometry 类的实例
THREE.ParametricGeometry.prototype = Object.create(THREE.Geometry.prototype);
// 创建一个名为 AxisHelper 的函数，参数为 a，默认值为 1
THREE.AxisHelper = function(a) {
    // 如果没有传入参数，则默认为 1
    a = a || 1;
    // 创建一个包含坐标轴顶点位置的 Float32Array
    var b = new Float32Array([0, 0, 0, a, 0, 0, 0, 0, 0, 0, a, 0, 0, 0, 0, 0, 0, a]);
    // 创建一个包含颜色信息的 Float32Array
    var c = new Float32Array([1, 0, 0, 1, .6, 0, 0, 1, 0, .6, 1, 0, 0, 0, 1, 0, .6, 1]);
    // 创建一个 BufferGeometry 对象
    a = new THREE.BufferGeometry;
    // 添加顶点位置属性
    a.addAttribute("position", new THREE.BufferAttribute(b, 3));
    // 添加颜色属性
    a.addAttribute("color", new THREE.BufferAttribute(c, 3));
    // 创建 LineBasicMaterial 材质
    b = new THREE.LineBasicMaterial({ vertexColors: THREE.VertexColors });
    // 调用 THREE.Line 构造函数，创建 AxisHelper 对象
    THREE.Line.call(this, a, b, THREE.LinePieces)
};
// 设置 AxisHelper 的原型为 THREE.Line 的实例
THREE.AxisHelper.prototype = Object.create(THREE.Line.prototype);

// 创建一个名为 ArrowHelper 的函数
THREE.ArrowHelper = function() {
    // 创建一个 Geometry 对象
    var a = new THREE.Geometry;
    // 向 Geometry 对象中添加顶点
    a.vertices.push(new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 1, 0));
    // 创建一个圆柱体几何体
    var b = new THREE.CylinderGeometry(0, .5, 1, 5, 1);
    // 对圆柱体几何体进行平移变换
    b.applyMatrix((new THREE.Matrix4).makeTranslation(0, -.5, 0));
    // 返回一个函数
    return function(c, d, e, f, g, h) {
        // 创建 Object3D 对象
        THREE.Object3D.call(this);
        // 设置默认参数
        void 0 === f && (f = 16776960);
        void 0 === e && (e = 1);
        void 0 === g && (g = .2 * e);
        void 0 === h && (h = .2 * g);
        // 设置位置
        this.position.copy(d);
        // 创建 Line 对象
        this.line = new THREE.Line(a, new THREE.LineBasicMaterial({ color: f }));
        this.line.matrixAutoUpdate = !1;
        this.add(this.line);
        // 创建圆锥体
        this.cone = new THREE.Mesh(b, new THREE.MeshBasicMaterial({ color: f }));
        this.cone.matrixAutoUpdate = !1;
        this.add(this.cone);
        // 设置方向
        this.setDirection(c);
        // 设置长度
        this.setLength(e, g, h)
    }
}();
// 设置 ArrowHelper 的原型为 THREE.Object3D 的实例
THREE.ArrowHelper.prototype = Object.create(THREE.Object3D.prototype);
// 设置 ArrowHelper 的 setDirection 方法
THREE.ArrowHelper.prototype.setDirection = function() {
    var a = new THREE.Vector3,
        b;
    return function(c) {
        // 根据方向设置四元数
        .99999 < c.y ? this.quaternion.set(0, 0, 0, 1) : -.99999 > c.y ? this.quaternion.set(1, 0, 0, 0) : (a.set(c.z, 0, -c.x).normalize(), b = Math.acos(c.y), this.quaternion.setFromAxisAngle(a, b))
    }
}();
// 设置 ArrowHelper 的 setLength 方法
THREE.ArrowHelper.prototype.setLength = function(a, b, c) {
    // 设置线段的缩放
    void 0 === b && (b = .2 * a);
    void 0 === c && (c = .2 * b);
    this.line.scale.set(1, a, 1);
    this.line.updateMatrix();
    // 设置圆锥体的缩放和位置
    this.cone.scale.set(c, b, c);
    this.cone.position.y = a;
    this.cone.updateMatrix()
};
// 设置 ArrowHelper 的 setColor 方法
THREE.ArrowHelper.prototype.setColor = function(a) {
    // 设置线段和圆锥体的颜色
    this.line.material.color.set(a);
    this.cone.material.color.set(a)
};
# 创建一个名为 BoxHelper 的函数，参数为 a
THREE.BoxHelper=function(a){
    # 创建一个新的 BufferGeometry 对象
    var b=new THREE.BufferGeometry;
    # 给 BufferGeometry 对象添加名为 "position" 的属性，值为一个包含 72 个浮点数的数组
    b.addAttribute("position",new THREE.BufferAttribute(new Float32Array(72),3));
    # 用 BufferGeometry 对象和 LineBasicMaterial 对象创建一个 Line 对象
    THREE.Line.call(this,b,new THREE.LineBasicMaterial({color:16776960}),THREE.LinePieces);
    # 如果参数 a 不为 undefined，则调用 update 方法
    void 0!==a&&this.update(a)
};
# 将 BoxHelper 的原型设置为 Line 的实例
THREE.BoxHelper.prototype=Object.create(THREE.Line.prototype);
# 定义 BoxHelper 的原型上的 update 方法，参数为 a
THREE.BoxHelper.prototype.update=function(a){
    # 获取参数 a 的 geometry 属性
    var b=a.geometry;
    # 如果参数 a 的 boundingBox 为 null，则计算其 boundingBox
    null===b.boundingBox&&b.computeBoundingBox();
    # 获取 boundingBox 的最小值和最大值
    var c=b.boundingBox.min,b=b.boundingBox.max,d=this.geometry.attributes.position.array;
    # 更新 position 属性数组的值
    d[0]=b.x;d[1]=b.y;d[2]=b.z;d[3]=c.x;d[4]=b.y;d[5]=b.z;d[6]=c.x;d[7]=b.y;d[8]=b.z;d[9]=c.x;d[10]=c.y;d[11]=b.z;d[12]=c.x;d[13]=c.y;d[14]=b.z;d[15]=b.x;d[16]=c.y;d[17]=b.z;d[18]=b.x;d[19]=c.y;d[20]=b.z;d[21]=b.x;d[22]=b.y;d[23]=b.z;d[24]=b.x;d[25]=b.y;d[26]=c.z;d[27]=c.x;d[28]=b.y;d[29]=c.z;d[30]=c.x;d[31]=b.y;
    d[32]=c.z;d[33]=c.x;d[34]=c.y;d[35]=c.z;d[36]=c.x;d[37]=c.y;d[38]=c.z;d[39]=b.x;d[40]=c.y;d[41]=c.z;d[42]=b.x;d[43]=c.y;d[44]=c.z;d[45]=b.x;d[46]=b.y;d[47]=c.z;d[48]=b.x;d[49]=b.y;d[50]=b.z;d[51]=b.x;d[52]=b.y;d[53]=c.z;d[54]=c.x;d[55]=b.y;d[56]=b.z;d[57]=c.x;d[58]=b.y;d[59]=c.z;d[60]=c.x;d[61]=c.y;d[62]=b.z;d[63]=c.x;d[64]=c.y;d[65]=c.z;d[66]=b.x;d[67]=c.y;d[68]=b.z;d[69]=b.x;d[70]=c.y;d[71]=c.z;
    # 将 position 属性标记为需要更新
    this.geometry.attributes.position.needsUpdate=!0;
    # 计算几何体的包围球
    this.geometry.computeBoundingSphere();
    # 设置对象的世界矩阵为参数 a 的世界矩阵
    this.matrix=a.matrixWorld;
    # 禁用对象的自动更新
    this.matrixAutoUpdate=!1
};
# 创建一个名为 BoundingBoxHelper 的函数，参数为 a 和 b
THREE.BoundingBoxHelper=function(a,b){
    # 如果参数 b 不为 undefined，则将其赋值给 c，否则赋值为默认颜色值
    var c=void 0!==b?b:8947848;
    # 将参数 a 赋值给对象的 object 属性
    this.object=a;
    # 创建一个 Box3 对象
    this.box=new THREE.Box3;
    # 使用 BoxGeometry 对象和 MeshBasicMaterial 对象创建一个 Mesh 对象
    THREE.Mesh.call(this,new THREE.BoxGeometry(1,1,1),new THREE.MeshBasicMaterial({color:c,wireframe:!0}))
};
# 将 BoundingBoxHelper 的原型设置为 Mesh 的实例
THREE.BoundingBoxHelper.prototype=Object.create(THREE.Mesh.prototype);
# 定义 BoundingBoxHelper 的原型上的 update 方法
THREE.BoundingBoxHelper.prototype.update=function(){
    # 根据对象的边界框更新对象的尺寸和位置
    this.box.setFromObject(this.object);
    this.box.size(this.scale);
    this.box.center(this.position)
};
# 定义一个名为CameraHelper的函数，参数为a
THREE.CameraHelper=function(a){
    # 定义一个名为b的函数，参数为a、b、d
    function b(a,b,d){
        # 调用c函数，参数为a、d
        c(a,d);
        # 调用c函数，参数为b、d
        c(b,d)
    }
    # 定义一个名为c的函数，参数为a、b
    function c(a,b){
        # 创建一个新的Geometry对象
        d.vertices.push(new THREE.Vector3);
        # 创建一个新的Color对象，并添加到colors数组中
        d.colors.push(new THREE.Color(b));
        # 如果f对象中不存在a属性，则创建一个空数组
        void 0===f[a]&&(f[a]=[]);
        # 将当前顶点的索引添加到f对象中对应属性的数组中
        f[a].push(d.vertices.length-1)
    }
    # 创建一个新的Geometry对象
    var d=new THREE.Geometry,
    # 创建一个新的LineBasicMaterial对象
    e=new THREE.LineBasicMaterial({color:16777215,vertexColors:THREE.FaceColors}),
    # 创建一个空对象f
    f={};
    # 调用b函数，参数为"n1"、"n2"、16755200
    b("n1","n2",16755200);
    # 调用b函数，参数为"n2"、"n4"、16755200
    b("n2","n4",16755200);
    # 调用b函数，参数为"n4"、"n3"、16755200
    b("n4","n3",16755200);
    # 调用b函数，参数为"n3"、"n1"、16755200
    b("n3","n1",16755200);
    # 调用b函数，参数为"f1"、"f2"、16755200
    b("f1","f2",16755200);
    # 调用b函数，参数为"f2"、"f4"、16755200
    b("f2","f4",16755200);
    # 调用b函数，参数为"f4"、"f3"、16755200
    b("f4","f3",16755200);
    # 调用b函数，参数为"f3"、"f1"、16755200
    b("f3","f1",16755200);
    # 调用b函数，参数为"n1"、"f1"、16755200
    b("n1","f1",16755200);
    # 调用b函数，参数为"n2"、"f2"、16755200
    b("n2","f2",16755200);
    # 调用b函数，参数为"n3"、"f3"、16755200
    b("n3","f3",16755200);
    # 调用b函数，参数为"n4"、"f4"、16755200
    b("n4","f4",16755200);
    # 调用b函数，参数为"p"、"n1"、16711680
    b("p","n1",16711680);
    # 调用b函数，参数为"p"、"n2"、16711680
    b("p","n2",16711680);
    # 调用b函数，参数为"p"、"n3"、16711680
    b("p","n3",16711680);
    # 调用b函数，参数为"p"、"n4"、16711680
    b("p","n4",16711680);
    # 调用b函数，参数为"u1"、"u2"、43775
    b("u1","u2",43775);
    # 调用b函数，参数为"u2"、"u3"、43775
    b("u2","u3",43775);
    # 调用b函数，参数为"u3"、"u1"、43775
    b("u3","u1",43775);
    # 调用b函数，参数为"c"、"t"、16777215
    b("c","t",16777215);
    # 调用b函数，参数为"p"、"c"、3355443
    b("p","c",3355443);
    # 调用b函数，参数为"cn1"、"cn2"、3355443
    b("cn1","cn2",3355443);
    # 调用b函数，参数为"cn3"、"cn4"、3355443
    b("cn3","cn4",3355443);
    # 调用b函数，参数为"cf1"、"cf2"、3355443
    b("cf1","cf2",3355443);
    # 调用b函数，参数为"cf3"、"cf4"、3355443
    b("cf3","cf4",3355443);
    # 创建一个新的Line对象，继承自THREE.Line
    THREE.Line.call(this,d,e,THREE.LinePieces);
    # 将参数a赋值给this.camera
    this.camera=a;
    # 将a.matrixWorld赋值给this.matrix
    this.matrix=a.matrixWorld;
    # 将this.camera.projectionMatrix的值复制给d.projectionMatrix
    this.matrixAutoUpdate=!1;
    # 将f对象赋值给this.pointMap
    this.pointMap=f;
    # 调用update函数
    this.update()
};
# 将THREE.CameraHelper的原型设置为THREE.Line的实例
THREE.CameraHelper.prototype=Object.create(THREE.Line.prototype);
# 定义update方法
THREE.CameraHelper.prototype.update=function(){
    # 定义局部变量a、b、c、d
    var a,b,c=new THREE.Vector3,d=new THREE.Camera,
    # 定义函数e
    e=function(e,g,h,k){
        # 设置c的值为(g,h,k)在相机坐标系下的值
        c.set(g,h,k).unproject(d);
        # 获取b[e]的值，如果存在则遍历
        e=b[e];
        if(void 0!==e)
            for(g=0,h=e.length;g<h;g++)
                # 将c的值复制给a.vertices[e[g]]
                a.vertices[e[g]].copy(c)
    };
    # 返回一个函数
    return function(){
        # 将this.geometry赋值给a
        a=this.geometry;
        # 将this.pointMap赋值给b
        b=this.pointMap;
        # 将this.camera.projectionMatrix的值复制给d.projectionMatrix
        d.projectionMatrix.copy(this.camera.projectionMatrix);
        # 调用e函数，参数为"c",0,0,-1
        e("c",0,0,-1);
        # 调用e函数，参数为"t",0,0,1
        e("t",0,0,1);
        # 调用e函数，参数为"n1",-1,-1,-1
        e("n1",-1,-1,-1);
        # 调用e函数，参数为"n2",1,-1,-1
        e("n2",1,-1,-1);
        # 调用e函数，参数为"n3",-1,1,-1
        e("n3",-1,1,-1);
        # 调用e函数，参数为"n4",1,1,-1
        e("n4",1,1,-1);
        # 调用e函数，参数为"f1",-1,-1,1
        e("f1",-1,-1,1);
        # 调用e函数，参数为"f2",1,-1,1
        e("f2",1,-1,1);
        # 调用e函数，参数为"f3",-1,1,1
        e("f3",-1,1,1);
        # 调用e函数，参数为"f4",1,1,1
        e("f4",1,1,1);
        # 调用e函数，参数为"u1",.7,1.1,-1
        e("u1",.7,1.1,-1);
        # 调用e函数，参数为"u2",-.7,1.1,-1
        e("u2",-.7,1.1,-1);
        # 调用e函数，参数为"u3",0,2,-1
        e("u3",0,2,-1);
        # 调用e函数，参数为"cf1",-1,0,1
        e("cf1",-1,0,1);
        # 调用e函数，参数为"cf2",1,0,1
        e("cf2",1,0,1);
        # 调用e函数，参数为"cf3",0,-1,1
        e("cf3",0,-1,1);
        # 调用e函数，参数为"cf4",0,1,1
        e("cf4",0,1,1);
        # 调用e函数，参数为"cn1",-1,0,-1
        e("cn1",-1,0,-1);
        # 调用e函数，参数为"cn2",1,0,-1
        e("cn2",1,0,-1);
        # 调用e函数，参数为"cn3",0,-1,-1
        e("cn3",0,-1,-1);
        # 调用e函数，参数为"cn4",0,1,-1
        e("cn4",0,1,-1);
        # 将a.verticesNeedUpdate的值设置为true
        a.verticesNeedUpdate=!0
    }
}();
# 创建一个名为DirectionalLightHelper的函数，用于创建一个方向光源的辅助对象
THREE.DirectionalLightHelper=function(a,b){THREE.Object3D.call(this);
# 设置辅助对象的属性为传入的光源对象，并更新光源对象的世界矩阵
this.light=a;
this.light.updateMatrixWorld();
this.matrix=a.matrixWorld;
this.matrixAutoUpdate=!1;
# 设置默认的辅助对象大小为1，创建一个几何体对象
b=b||1;
var c=new THREE.Geometry;
# 在几何体中添加表示光源方向的四个顶点
c.vertices.push(new THREE.Vector3(-b,b,0),new THREE.Vector3(b,b,0),new THREE.Vector3(b,-b,0),new THREE.Vector3(-b,-b,0),new THREE.Vector3(-b,b,0));
# 创建一个线条材质，设置颜色为光源颜色乘以光源强度，并创建表示光源方向的线条对象
var d=new THREE.LineBasicMaterial({fog:!1});
d.color.copy(this.light.color).multiplyScalar(this.light.intensity);
this.lightPlane=new THREE.Line(c,d);
# 将表示光源方向的线条对象添加到辅助对象中
this.add(this.lightPlane);
# 创建另一个几何体对象，用于表示光源的目标位置
c=new THREE.Geometry;
# 在目标位置几何体中添加两个顶点
c.vertices.push(new THREE.Vector3,new THREE.Vector3);
# 创建另一个线条材质，设置颜色为光源颜色乘以光源强度，并创建表示目标位置的线条对象
d=new THREE.LineBasicMaterial({fog:!1});
d.color.copy(this.light.color).multiplyScalar(this.light.intensity);
this.targetLine=new THREE.Line(c,d);
# 将表示目标位置的线条对象添加到辅助对象中
this.add(this.targetLine);
# 调用update方法更新辅助对象的状态
this.update()};

# 设置DirectionalLightHelper的原型为THREE.Object3D的实例
THREE.DirectionalLightHelper.prototype=Object.create(THREE.Object3D.prototype);

# 创建一个dispose方法，用于释放辅助对象的几何体和材质资源
THREE.DirectionalLightHelper.prototype.dispose=function(){this.lightPlane.geometry.dispose();this.lightPlane.material.dispose();this.targetLine.geometry.dispose();this.targetLine.material.dispose()};

# 创建一个update方法，用于更新辅助对象的状态
THREE.DirectionalLightHelper.prototype.update=function(){var a=new THREE.Vector3,b=new THREE.Vector3,c=new THREE.Vector3;return function(){a.setFromMatrixPosition(this.light.matrixWorld);b.setFromMatrixPosition(this.light.target.matrixWorld);c.subVectors(b,a);this.lightPlane.lookAt(c);this.lightPlane.material.color.copy(this.light.color).multiplyScalar(this.light.intensity);this.targetLine.geometry.vertices[1].copy(c);this.targetLine.geometry.verticesNeedUpdate=!0;this.targetLine.material.color.copy(this.lightPlane.material.color)}}();
# 创建一个名为EdgesHelper的函数，接受两个参数a和b
THREE.EdgesHelper=function(a,b){
    # 如果传入了b参数，则将其赋值给c，否则默认为16777215
    var c=void 0!==b?b:16777215,
    # 定义变量d为一个包含两个0的数组
    d=[0,0],
    # 定义变量e为一个空对象
    e={},
    # 定义函数f，用于比较两个值的大小
    f=function(a,b){return a-b},
    # 定义数组g，包含字符串"a"、"b"、"c"
    g=["a","b","c"],
    # 创建一个新的BufferGeometry对象h
    h=new THREE.BufferGeometry,
    # 克隆参数a的geometry属性，赋值给变量k
    k=a.geometry.clone();
    # 合并参数k的顶点
    k.mergeVertices();
    # 计算参数k的面法线
    k.computeFaceNormals();
    # 获取参数k的顶点数组，赋值给变量n
    for(var n=k.vertices,
    # 获取参数k的面数组，赋值给变量k
    k=k.faces,
    # 初始化变量p为0
    p=0,
    # 初始化变量q为0
    q=0,
    # 获取参数k的长度，赋值给变量m
    m=k.length;
    # 当q小于m时，执行循环
    q<m;
    # 每次循环结束后，q自增1
    q++)
        # 内层循环，遍历参数k的面数组
        for(var r=k[q],
        # 初始化变量t为0
        t=0;
        # 当t小于3时，执行循环
        3>t;
        # 每次循环结束后，t自增1
        t++){
            # 将r的g[t]属性赋值给d[0]
            d[0]=r[g[t]];
            # 将r的g[(t+1)%3]属性赋值给d[1]
            d[1]=r[g[(t+1)%3]];
            # 对数组d进行排序，使用函数f
            d.sort(f);
            # 将数组d转换为字符串，赋值给变量s
            var s=d.toString();
            # 如果e对象中不存在键为s的属性
            void 0===e[s]?
                # 则在e对象中添加一个键为s，值为对象的属性
                (e[s]={vert1:d[0],vert2:d[1],face1:q,face2:void 0},
                # p自增1
                p++):
                # 否则，e对象中键为s的属性存在
                # 则在e对象中键为s的属性中添加face2属性，值为q
                e[s].face2=q
        }
    # 创建一个包含6*p个0的浮点数数组，赋值给变量d
    d=new Float32Array(6*p);
    # 初始化变量f为0
    f=0;
    # 遍历e对象的属性
    for(s in e)
        # 获取e对象的属性值，赋值给变量g
        if(g=e[s],
        # 如果e对象的属性值中不存在face2属性，或者k[g.face1].normal.dot(k[g.face2].normal)小于0.9999
        void 0===g.face2||.9999>k[g.face1].normal.dot(k[g.face2].normal))
            # 获取n[g.vert1]的值，赋值给变量p
            p=n[g.vert1],
            # 将p的x、y、z属性依次赋值给d数组
            d[f++]=p.x,
            d[f++]=p.y,
            d[f++]=p.z;
            # 获取n[g.vert2]的值，赋值给变量p
            p=n[g.vert2],
            # 将p的x、y、z属性依次赋值给d数组
            d[f++]=p.x,
            d[f++]=p.y,
            d[f++]=p.z;
    # 在BufferGeometry对象h中添加名为"position"的属性，值为一个BufferAttribute对象
    h.addAttribute("position",new THREE.BufferAttribute(d,3));
    # 调用THREE.Line构造函数，创建一个新的对象
    THREE.Line.call(this,h,new THREE.LineBasicMaterial({color:c}),THREE.LinePieces);
    # 将参数a的matrixWorld属性赋值给this对象的matrix属性
    this.matrix=a.matrixWorld;
    # 将this对象的matrixAutoUpdate属性设置为false
    this.matrixAutoUpdate=!1
};
# 将EdgesHelper的原型对象设置为THREE.Line的实例
THREE.EdgesHelper.prototype=Object.create(THREE.Line.prototype);

# 创建一个名为FaceNormalsHelper的函数，接受四个参数a、b、c、d
THREE.FaceNormalsHelper=function(a,b,c,d){
    # 将参数a赋值给this对象的object属性
    this.object=a;
    # 如果传入了b参数，则将其赋值给this对象的size属性，否则默认为1
    this.size=void 0!==b?b:1;
    # 如果传入了c参数，则将其赋值给变量a，否则默认为16776960
    a=void 0!==c?c:16776960;
    # 如果传入了d参数，则将其赋值给变量d，否则默认为1
    d=void 0!==d?d:1;
    # 创建一个新的Geometry对象b
    b=new THREE.Geometry;
    # 初始化变量c为0
    c=0;
    # 遍历this对象的geometry属性的faces数组的长度
    for(var e=this.object.geometry.faces.length;
    # 当c小于e时，执行循环
    c<e;
    # 每次循环结束后，c自增1
    c++)
        # 在Geometry对象b的vertices数组中添加两个Vector3对象
        b.vertices.push(new THREE.Vector3,new THREE.Vector3);
    # 调用THREE.Line构造函数，创建一个新的对象
    THREE.Line.call(this,b,new THREE.LineBasicMaterial({color:a,linewidth:d}),THREE.LinePieces);
    # 将this对象的matrixAutoUpdate属性设置为false
    this.matrixAutoUpdate=!1;
    # 创建一个新的Matrix3对象，赋值给this对象的normalMatrix属性
    this.normalMatrix=new THREE.Matrix3;
    # 调用update方法
    this.update()
};
# 将FaceNormalsHelper的原型对象设置为THREE.Line的实例
THREE.FaceNormalsHelper.prototype=Object.create(THREE.Line.prototype);
# 定义FaceNormalsHelper的原型对象的update方法
THREE.FaceNormalsHelper.prototype.update=function(){
    # 获取this对象的geometry属性的vertices数组，赋值给变量a
    var a=this.geometry.vertices,
    # 获取this对象的object属性，赋值给变量b
    b=this.object,
    # 获取b对象的geometry属性的vertices数组，赋值给变量c
    c=b.geometry.vertices,
    # 获取b对象的geometry属性的faces数组，赋值给变量d
    d=b.geometry.faces,
    # 获取b对象的matrixWorld属性，赋值给变量e
    e=b.matrixWorld;
    # 更新b对象的matrixWorld属性
    b.updateMatrixWorld(!0);
    # 获取this对象的normalMatrix属性的法线矩阵
    this.normalMatrix.getNormalMatrix(e);
    # 初始化变量f为0
    for(var f=b=0,
    # 获取d数组的长度，赋值给变量g
    g=d.length;
    # 当b小于g时，执行循环
    b<g;
    # 每次循环结束后，b自增1
    b++,f+=2){
        # 获取d[b]的值，赋值给变量h
        var h=d[b];
        # 将c[h.a]、c[h.b]、c[h.c]的平均值，应用矩阵e后，赋值给a[f]
        a[f].copy(c[h.a]).add(c[h.b]).add(c[h.c]).divideScalar(3).applyMatrix4(e);
        # 将h的normal属性，应用矩阵this.normalMatrix后，归一化，乘以this.size后，加上a[f]的值，赋值给a[f+1]
        a[f+1].copy(h.normal).applyMatrix3(this.normalMatrix).normalize().multiplyScalar(this.size).add(a[f])
    }
    # 将this对象的geometry属性的verticesNeedUpdate属性设置为true
    this.geometry.verticesNeedUpdate=!0;
    # 返回this对象
    return this
};
# 创建一个名为GridHelper的函数，接受两个参数a和b
THREE.GridHelper=function(a,b){
    # 创建一个新的几何体对象
    var c=new THREE.Geometry,
    # 创建一个新的线条基本材质对象，设置顶点颜色为顶点颜色
    d=new THREE.LineBasicMaterial({vertexColors:THREE.VertexColors});
    # 设置两种颜色
    this.color1=new THREE.Color(4473924);
    this.color2=new THREE.Color(8947848);
    # 根据参数a和b生成网格线
    for(var e=-a;e<=a;e+=b){
        c.vertices.push(new THREE.Vector3(-a,0,e),new THREE.Vector3(a,0,e),new THREE.Vector3(e,0,-a),new THREE.Vector3(e,0,a));
        var f=0===e?this.color1:this.color2;
        c.colors.push(f,f,f,f)
    }
    # 调用THREE.Line构造函数
    THREE.Line.call(this,c,d,THREE.LinePieces)
};
# 设置GridHelper的原型为THREE.Line的实例
THREE.GridHelper.prototype=Object.create(THREE.Line.prototype);
# 设置GridHelper的setColors方法
THREE.GridHelper.prototype.setColors=function(a,b){
    this.color1.set(a);
    this.color2.set(b);
    this.geometry.colorsNeedUpdate=!0
};
# 创建一个名为HemisphereLightHelper的函数，接受四个参数a、b、c、d
THREE.HemisphereLightHelper=function(a,b,c,d){
    # 调用THREE.Object3D构造函数
    THREE.Object3D.call(this);
    # 设置light属性为传入的a参数
    this.light=a;
    # 更新light的世界矩阵
    this.light.updateMatrixWorld();
    # 设置matrix属性为light的世界矩阵
    this.matrix=a.matrixWorld;
    # 设置matrixAutoUpdate属性为false
    this.matrixAutoUpdate=!1;
    # 创建两个颜色对象
    this.colors=[new THREE.Color,new THREE.Color];
    # 创建一个球体几何体
    a=new THREE.SphereGeometry(b,4,2);
    # 对几何体进行旋转
    a.applyMatrix((new THREE.Matrix4).makeRotationX(-Math.PI/2));
    # 为几何体的每个面设置颜色
    for(b=0;8>b;b++)
        a.faces[b].color=this.colors[4>b?0:1];
    # 创建一个线框材质
    b=new THREE.MeshBasicMaterial({vertexColors:THREE.FaceColors,wireframe:!0});
    # 创建一个网格对象
    this.lightSphere=new THREE.Mesh(a,b);
    # 将网格对象添加到当前对象中
    this.add(this.lightSphere);
    # 调用update方法
    this.update()
};
# 设置HemisphereLightHelper的原型为THREE.Object3D的实例
THREE.HemisphereLightHelper.prototype=Object.create(THREE.Object3D.prototype);
# 设置HemisphereLightHelper的dispose方法
THREE.HemisphereLightHelper.prototype.dispose=function(){
    # 释放网格对象的几何体和材质
    this.lightSphere.geometry.dispose();
    this.lightSphere.material.dispose()
};
# 设置HemisphereLightHelper的update方法
THREE.HemisphereLightHelper.prototype.update=function(){
    # 创建一个三维向量
    var a=new THREE.Vector3;
    return function(){
        # 复制光源颜色并乘以光源强度
        this.colors[0].copy(this.light.color).multiplyScalar(this.light.intensity);
        # 复制地面颜色并乘以光源强度
        this.colors[1].copy(this.light.groundColor).multiplyScalar(this.light.intensity);
        # 设置网格对象朝向光源位置
        this.lightSphere.lookAt(a.setFromMatrixPosition(this.light.matrixWorld).negate());
        # 更新几何体的颜色
        this.lightSphere.geometry.colorsNeedUpdate=!0
    }
}();
# 创建一个 PointLightHelper 类，用于辅助显示 PointLight 的位置和范围
THREE.PointLightHelper=function(a,b){
    # 设置 light 属性为传入的 PointLight 对象
    this.light=a;
    # 更新 light 对象的世界矩阵
    this.light.updateMatrixWorld();
    # 创建一个球体几何体，用于表示光源的范围
    var c=new THREE.SphereGeometry(b,4,2);
    # 创建一个基本网格材质，用于表示光源的范围
    var d=new THREE.MeshBasicMaterial({wireframe:!0,fog:!1});
    # 设置材质的颜色为光源的颜色乘以光源的强度
    d.color.copy(this.light.color).multiplyScalar(this.light.intensity);
    # 调用 Mesh 构造函数，创建一个网格对象
    THREE.Mesh.call(this,c,d);
    # 设置网格对象的矩阵为光源的世界矩阵
    this.matrix=this.light.matrixWorld;
    # 禁用自动更新矩阵
    this.matrixAutoUpdate=!1
};
# 设置 PointLightHelper 的原型为 Mesh 对象的实例
THREE.PointLightHelper.prototype=Object.create(THREE.Mesh.prototype);
# 定义 PointLightHelper 的 dispose 方法
THREE.PointLightHelper.prototype.dispose=function(){
    # 释放几何体资源
    this.geometry.dispose();
    # 释放材质资源
    this.material.dispose()
};
# 定义 PointLightHelper 的 update 方法
THREE.PointLightHelper.prototype.update=function(){
    # 更新材质的颜色为光源的颜色乘以光源的强度
    this.material.color.copy(this.light.color).multiplyScalar(this.light.intensity)
};
# 创建一个 SkeletonHelper 类，用于辅助显示骨骼的位置和关系
THREE.SkeletonHelper=function(a){
    # 获取骨骼列表
    this.bones=this.getBoneList(a);
    # 创建一个几何体，用于表示骨骼的线条
    var b=new THREE.Geometry;
    # 遍历骨骼列表，为每个骨骼创建线条
    for(var c=0;c<this.bones.length;c++)
        this.bones[c].parent instanceof THREE.Bone&&(b.vertices.push(new THREE.Vector3),b.vertices.push(new THREE.Vector3),b.colors.push(new THREE.Color(0,0,1)),b.colors.push(new THREE.Color(0,1,0)));
    # 创建一个线条材质，用于表示骨骼的线条
    c=new THREE.LineBasicMaterial({vertexColors:THREE.VertexColors,depthTest:!1,depthWrite:!1,transparent:!0});
    # 调用 Line 构造函数，创建一个线条对象
    THREE.Line.call(this,b,c,THREE.LinePieces);
    # 设置骨骼对象的世界矩阵为传入的骨骼对象的世界矩阵
    this.root=a;
    this.matrix=a.matrixWorld;
    # 禁用自动更新矩阵
    this.matrixAutoUpdate=!1;
    # 调用 update 方法，更新骨骼的位置和关系
    this.update()
};
# 设置 SkeletonHelper 的原型为 Line 对象的实例
THREE.SkeletonHelper.prototype=Object.create(THREE.Line.prototype);
# 定义 SkeletonHelper 的 getBoneList 方法，用于获取骨骼列表
THREE.SkeletonHelper.prototype.getBoneList=function(a){
    var b=[];
    a instanceof THREE.Bone&&b.push(a);
    for(var c=0;c<a.children.length;c++)
        b.push.apply(b,this.getBoneList(a.children[c]));
    return b
};
# 定义 SkeletonHelper 的 update 方法，用于更新骨骼的位置和关系
THREE.SkeletonHelper.prototype.update=function(){
    # 获取几何体
    var a=this.geometry;
    # 创建一个矩阵，用于将骨骼的世界矩阵转换为相对于根骨骼的矩阵
    var b=(new THREE.Matrix4).getInverse(this.root.matrixWorld);
    # 创建一个矩阵，用于存储骨骼的世界矩阵
    var c=new THREE.Matrix4;
    # 遍历骨骼列表，更新骨骼的位置和关系
    for(var d=0,e=0;e<this.bones.length;e++){
        var f=this.bones[e];
        f.parent instanceof THREE.Bone&&(c.multiplyMatrices(b,f.matrixWorld),a.vertices[d].setFromMatrixPosition(c),c.multiplyMatrices(b,f.parent.matrixWorld),a.vertices[d+1].setFromMatrixPosition(c),d+=2)
    }
    # 标记几何体的顶点需要更新
    a.verticesNeedUpdate=!0;
    # 计算几何体的包围球
    a.computeBoundingSphere()
};
# 创建一个名为THREE的对象，其中包含SpotLightHelper和VertexNormalsHelper两个函数
THREE.SpotLightHelper=function(a){
    # 调用Object3D构造函数，创建一个新的Object3D对象
    THREE.Object3D.call(this);
    # 将传入的光源对象赋值给属性light，并更新其世界矩阵
    this.light=a;
    this.light.updateMatrixWorld();
    # 将光源对象的世界矩阵赋值给当前对象的矩阵
    this.matrix=a.matrixWorld;
    # 禁用自动更新矩阵
    this.matrixAutoUpdate=!1;
    # 创建一个圆锥几何体
    a=new THREE.CylinderGeometry(0,1,1,8,1,!0);
    # 对几何体进行平移和旋转变换
    a.applyMatrix((new THREE.Matrix4).makeTranslation(0,-.5,0));
    a.applyMatrix((new THREE.Matrix4).makeRotationX(-Math.PI/2));
    # 创建一个基础网格材质
    var b=new THREE.MeshBasicMaterial({wireframe:!0,fog:!1});
    # 创建一个圆锥网格对象，并添加到当前对象中
    this.cone=new THREE.Mesh(a,b);
    this.add(this.cone);
    # 调用update方法
    this.update()
};
# 将SpotLightHelper的原型设置为Object3D的实例
THREE.SpotLightHelper.prototype=Object.create(THREE.Object3D.prototype);
# 定义SpotLightHelper的dispose方法
THREE.SpotLightHelper.prototype.dispose=function(){
    # 释放圆锥几何体和材质的资源
    this.cone.geometry.dispose();
    this.cone.material.dispose()
};
# 定义SpotLightHelper的update方法
THREE.SpotLightHelper.prototype.update=function(){
    # 创建两个向量a和b
    var a=new THREE.Vector3,b=new THREE.Vector3;
    return function(){
        # 获取光源的距离和角度
        var c=this.light.distance?this.light.distance:1E4,d=c*Math.tan(this.light.angle);
        # 设置圆锥的缩放
        this.cone.scale.set(d,d,c);
        # 获取光源和目标的世界坐标，并设置圆锥的朝向
        a.setFromMatrixPosition(this.light.matrixWorld);
        b.setFromMatrixPosition(this.light.target.matrixWorld);
        this.cone.lookAt(b.sub(a));
        # 设置圆锥材质的颜色
        this.cone.material.color.copy(this.light.color).multiplyScalar(this.light.intensity)
    }
}();
# 定义VertexNormalsHelper函数
THREE.VertexNormalsHelper=function(a,b,c,d){
    # 设置对象属性
    this.object=a;
    this.size=void 0!==b?b:1;
    b=void 0!==c?c:16711680;
    d=void 0!==d?d:1;
    # 创建一个几何体
    c=new THREE.Geometry;
    # 遍历几何体的面和顶点法向量，并添加到顶点数组中
    a=a.geometry.faces;
    for(var e=0,f=a.length;e<f;e++){
        for(var g=0,h=a[e].vertexNormals.length;g<h;g++){
            c.vertices.push(new THREE.Vector3,new THREE.Vector3);
        }
    }
    # 调用Line构造函数，创建一个线条对象
    THREE.Line.call(this,c,new THREE.LineBasicMaterial({color:b,linewidth:d}),THREE.LinePieces);
    # 禁用自动更新矩阵
    this.matrixAutoUpdate=!1;
    this.normalMatrix=new THREE.Matrix3;
    # 调用update方法
    this.update()
};
# 将VertexNormalsHelper的原型设置为Line的实例
THREE.VertexNormalsHelper.prototype=Object.create(THREE.Line.prototype);
// 定义一个函数，用于更新顶点法线辅助器
THREE.VertexNormalsHelper.prototype.update=function(a){
    // 创建一个三维向量
    var b=new THREE.Vector3;
    return function(a){
        // 定义顶点标识符
        a=["a","b","c","d"];
        // 更新对象的世界矩阵
        this.object.updateMatrixWorld(!0);
        // 获取对象的法线矩阵
        this.normalMatrix.getNormalMatrix(this.object.matrixWorld);
        // 遍历对象的顶点、几何顶点和面
        for(var d=this.geometry.vertices,e=this.object.geometry.vertices,f=this.object.geometry.faces,g=this.object.matrixWorld,h=0,k=0,n=f.length;k<n;k++)
            for(var p=f[k],q=0,m=p.vertexNormals.length;q<m;q++){
                var r=p.vertexNormals[q];
                // 复制几何顶点并应用矩阵变换
                d[h].copy(e[p[a[q]]]).applyMatrix4(g);
                // 复制法线并应用矩阵变换
                b.copy(r).applyMatrix3(this.normalMatrix).normalize().multiplyScalar(this.size);
                // 添加顶点
                b.add(d[h]);
                h+=1;
                // 复制顶点
                d[h].copy(b);
                h+=1
            }
        // 设置顶点需要更新
        this.geometry.verticesNeedUpdate=!0;
        return this
    }
}();

// 定义顶点切线辅助器
THREE.VertexTangentsHelper=function(a,b,c,d){
    // 设置对象和大小
    this.object=a;
    this.size=void 0!==b?b:1;
    b=void 0!==c?c:255;
    d=void 0!==d?d:1;
    // 创建几何体
    c=new THREE.Geometry;
    // 获取对象的面
    a=a.geometry.faces;
    // 遍历面的顶点切线
    for(var e=0,f=a.length;e<f;e++)
        for(var g=0,h=a[e].vertexTangents.length;g<h;g++)
            // 添加顶点
            c.vertices.push(new THREE.Vector3),c.vertices.push(new THREE.Vector3);
    // 创建线
    THREE.Line.call(this,c,new THREE.LineBasicMaterial({color:b,linewidth:d}),THREE.LinePieces);
    this.matrixAutoUpdate=!1;
    this.update()
};
// 设置原型链
THREE.VertexTangentsHelper.prototype=Object.create(THREE.Line.prototype);
// 更新顶点切线辅助器
THREE.VertexTangentsHelper.prototype.update=function(a){
    // 创建一个三维向量
    var b=new THREE.Vector3;
    return function(a){
        // 定义顶点标识符
        a=["a","b","c","d"];
        // 更新对象的世界矩阵
        this.object.updateMatrixWorld(!0);
        // 遍历对象的顶点、几何顶点和面
        for(var d=this.geometry.vertices,e=this.object.geometry.vertices,f=this.object.geometry.faces,g=this.object.matrixWorld,h=0,k=0,n=f.length;k<n;k++)
            for(var p=f[k],q=0,m=p.vertexTangents.length;q<m;q++){
                var r=p.vertexTangents[q];
                // 复制几何顶点并应用矩阵变换
                d[h].copy(e[p[a[q]]]).applyMatrix4(g);
                // 复制顶点切线并应用矩阵变换
                b.copy(r).transformDirection(g).multiplyScalar(this.size);
                // 添加顶点
                b.add(d[h]);
                h+=1;
                // 复制顶点
                d[h].copy(b);
                h+=1
            }
        // 设置顶点需要更新
        this.geometry.verticesNeedUpdate=!0;
        return this
    }
}();
# 创建一个名为WireframeHelper的函数，接受两个参数a和b
THREE.WireframeHelper=function(a,b){
    # 如果b有值，则使用b，否则使用默认值16777215
    var c=void 0!==b?b:16777215,
    # 定义变量d为数组[0,0]
    d=[0,0],
    # 定义变量e为空对象
    e={},
    # 定义函数f，用于比较两个值的大小
    f=function(a,b){return a-b},
    # 定义数组g为["a","b","c"]
    g=["a","b","c"],
    # 创建一个新的BufferGeometry对象
    h=new THREE.BufferGeometry;
    # 如果a的geometry属性是THREE.Geometry类型
    if(a.geometry instanceof THREE.Geometry){
        # 获取a的geometry的顶点和面
        for(var k=a.geometry.vertices,n=a.geometry.faces,p=0,q=new Uint32Array(6*n.length),m=0,r=n.length;m<r;m++)
            # 遍历面的每个顶点
            for(var t=n[m],s=0;3>s;s++){
                # 将面的顶点坐标存入数组d
                d[0]=t[g[s]];
                d[1]=t[g[(s+1)%3]];
                # 对数组d进行排序
                d.sort(f);
                # 将排序后的数组转换为字符串作为键，存入对象e
                var u=d.toString();
                void 0===e[u]&&(q[2*p]=d[0],q[2*p+1]=d[1],e[u]=!0,p++)
            }
        # 创建一个新的Float32Array数组
        d=new Float32Array(6*p);
        # 遍历顶点，将顶点坐标存入数组d
        m=0;
        for(r=p;m<r;m++)
            for(s=0;2>s;s++)
                p=k[q[2*m+s]],
                g=6*m+3*s,
                d[g+0]=p.x,
                d[g+1]=p.y,
                d[g+2]=p.z;
        # 将顶点坐标数组添加到BufferGeometry对象的position属性中
        h.addAttribute("position",new THREE.BufferAttribute(d,3))
    }
    # 如果a的geometry属性是THREE.BufferGeometry类型
    else if(a.geometry instanceof THREE.BufferGeometry){
        # 如果a的geometry的attributes中有index属性
        if(void 0!==a.geometry.attributes.index){
            # 获取顶点坐标数组和面索引数组
            k=a.geometry.attributes.position.array;
            r=a.geometry.attributes.index.array;
            n=a.geometry.drawcalls;
            p=0;
            # 如果没有drawcalls属性，则创建一个默认值
            0===n.length&&(n=[{count:r.length,index:0,start:0}]);
            # 创建一个新的Uint32Array数组
            for(var q=new Uint32Array(2*r.length),t=0,v=n.length;t<v;++t)
                for(var s=n[t].start,u=n[t].count,g=n[t].index,m=s,y=s+u;m<y;m+=3)
                    for(s=0;3>s;s++)
                        d[0]=g+r[m+s],
                        d[1]=g+r[m+(s+1)%3],
                        d.sort(f),
                        u=d.toString(),
                        void 0===e[u]&&(q[2*p]=d[0],q[2*p+1]=d[1],e[u]=!0,p++);
            # 创建一个新的Float32Array数组
            d=new Float32Array(6*p);
            # 遍历顶点，将顶点坐标存入数组d
            m=0;
            for(r=p;m<r;m++)
                for(s=0;2>s;s++)
                    g=6*m+3*s,
                    p=3*q[2*m+s],
                    d[g+0]=k[p],
                    d[g+1]=k[p+1],
                    d[g+2]=k[p+2]
        }
        # 如果a的geometry的attributes中没有index属性
        else
            # 获取顶点坐标数组，计算顶点数量和面数量
            for(k=a.geometry.attributes.position.array,p=k.length/3,q=p/3,d=new Float32Array(6*p),m=0,r=q;m<r;m++)
                for(s=0;3>s;s++)
                    g=18*m+6*s,
                    q=9*m+3*s,
                    d[g+0]=k[q],
                    d[g+1]=k[q+1],
                    d[g+2]=k[q+2],
                    p=9*m+(s+1)%3*3,
                    d[g+3]=k[p],
                    d[g+4]=k[p+1],
                    d[g+5]=k[p+2];
        # 将顶点坐标数组添加到BufferGeometry对象的position属性中
        h.addAttribute("position",new THREE.BufferAttribute(d,3))
    }
    # 创建一个THREE.Line对象，传入BufferGeometry对象和LineBasicMaterial对象
    THREE.Line.call(this,h,new THREE.LineBasicMaterial({color:c}),THREE.LinePieces);
    # 设置对象的matrix属性为a的matrixWorld属性
    this.matrix=a.matrixWorld;
    # 设置对象的matrixAutoUpdate属性为false
    this.matrixAutoUpdate=!1
};
# 将WireframeHelper的原型设置为THREE.Line的实例
THREE.WireframeHelper.prototype=Object.create(THREE.Line.prototype);
# 创建一个名为ImmediateRenderObject的函数
THREE.ImmediateRenderObject=function(){
    # 调用THREE.Object3D构造函数
    THREE.Object3D.call(this);
    # 设置对象的render属性为一个函数
    this.render=function(a){}
};
# 将ImmediateRenderObject的原型设置为THREE.Object3D的实例
THREE.ImmediateRenderObject.prototype=Object.create(THREE.Object3D.prototype);
// 定义一个名为 MorphBlendMesh 的函数，继承自 Mesh 类
THREE.MorphBlendMesh=function(a,b){THREE.Mesh.call(this,a,b);
// 创建一个 animationsMap 对象和 animationsList 数组
this.animationsMap={};
this.animationsList=[];
// 获取几何体的 morphTargets 数量
var c=this.geometry.morphTargets.length;
// 创建一个名为 "__default" 的动画，包含所有的 morphTargets
this.createAnimation("__default",0,c-1,c/1);
// 设置 "__default" 动画的权重为 1
this.setAnimationWeight("__default",1)};
// 将 MorphBlendMesh 的原型设置为 Mesh 的实例
THREE.MorphBlendMesh.prototype=Object.create(THREE.Mesh.prototype);

// 创建动画函数，接受动画名称、起始帧、结束帧、帧率作为参数
THREE.MorphBlendMesh.prototype.createAnimation=function(a,b,c,d){
// 创建动画对象，包含动画的各种属性
b={startFrame:b,endFrame:c,length:c-b+1,fps:d,duration:(c-b)/d,lastFrame:0,currentFrame:0,active:!1,time:0,direction:1,weight:1,directionBackwards:!1,mirroredLoop:!1};
// 将动画对象存储在 animationsMap 中
this.animationsMap[a]=b;
// 将动画对象存储在 animationsList 中
this.animationsList.push(b)};

// 自动创建动画函数，接受帧率作为参数
THREE.MorphBlendMesh.prototype.autoCreateAnimations=function(a){
// 正则表达式用于匹配动画名称和帧数
for(var b=/([a-z]+)_?(\d+)/,c,d={},e=this.geometry,f=0,g=e.morphTargets.length;f<g;f++){
var h=e.morphTargets[f].name.match(b);
if(h&&1<h.length){
var k=h[1];
d[k]||(d[k]={start:Infinity,end:-Infinity});
h=d[k];
f<h.start&&(h.start=f);
f>h.end&&(h.end=f);
c||(c=k)}}
// 根据匹配结果自动创建动画
for(k in d)h=d[k],this.createAnimation(k,h.start,h.end,a);
// 将第一个动画的名称存储在 firstAnimation 中
this.firstAnimation=c};

// 设置动画方向为正向
THREE.MorphBlendMesh.prototype.setAnimationDirectionForward=function(a){
if(a=this.animationsMap[a])a.direction=1,a.directionBackwards=!1};

// 设置动画方向为反向
THREE.MorphBlendMesh.prototype.setAnimationDirectionBackward=function(a){
if(a=this.animationsMap[a])a.direction=-1,a.directionBackwards=!0};

// 设置动画的帧率
THREE.MorphBlendMesh.prototype.setAnimationFPS=function(a,b){
var c=this.animationsMap[a];
c&&(c.fps=b,c.duration=(c.end-c.start)/c.fps)};

// 设置动画的持续时间
THREE.MorphBlendMesh.prototype.setAnimationDuration=function(a,b){
var c=this.animationsMap[a];
c&&(c.duration=b,c.fps=(c.end-c.start)/c.duration)};

// 设置动画的权重
THREE.MorphBlendMesh.prototype.setAnimationWeight=function(a,b){
var c=this.animationsMap[a];
c&&(c.weight=b)};

// 设置动画的时间
THREE.MorphBlendMesh.prototype.setAnimationTime=function(a,b){
var c=this.animationsMap[a];
c&&(c.time=b)};

// 获取动画的时间
THREE.MorphBlendMesh.prototype.getAnimationTime=function(a){
var b=0;
if(a=this.animationsMap[a])b=a.time;
return b};
// 获取指定动画的持续时间
THREE.MorphBlendMesh.prototype.getAnimationDuration=function(a){
    var b=-1;
    if(a=this.animationsMap[a])
        b=a.duration;
    return b;
};

// 播放指定动画
THREE.MorphBlendMesh.prototype.playAnimation=function(a){
    var b=this.animationsMap[a];
    if(b){
        b.time=0;
        b.active=!0;
    } else {
        console.warn("animation["+a+"] undefined");
    }
};

// 停止指定动画
THREE.MorphBlendMesh.prototype.stopAnimation=function(a){
    if(a=this.animationsMap[a])
        a.active=!1;
};

// 更新动画状态
THREE.MorphBlendMesh.prototype.update=function(a){
    for(var b=0,c=this.animationsList.length;b<c;b++){
        var d=this.animationsList[b];
        if(d.active){
            var e=d.duration/d.length;
            d.time+=d.direction*a;
            if(d.mirroredLoop){
                if(d.time>d.duration||0>d.time){
                    d.direction*=-1;
                    d.time>d.duration&&(d.time=d.duration,d.directionBackwards=!0);
                    0>d.time&&(d.time=0,d.directionBackwards=!1);
                }
            } else {
                d.time%=d.duration;
                0>d.time&&(d.time+=d.duration);
            }
            var f=d.startFrame+THREE.Math.clamp(Math.floor(d.time/e),0,d.length-1);
            var g=d.weight;
            if(f!==d.currentFrame){
                this.morphTargetInfluences[d.lastFrame]=0;
                this.morphTargetInfluences[d.currentFrame]=1*g;
                this.morphTargetInfluences[f]=0;
                d.lastFrame=d.currentFrame;
                d.currentFrame=f;
            }
            e=d.time%e/e;
            d.directionBackwards&&(e=1-e);
            this.morphTargetInfluences[d.currentFrame]=e*g;
            this.morphTargetInfluences[d.lastFrame]=(1-e)*g;
        }
    }
};
```