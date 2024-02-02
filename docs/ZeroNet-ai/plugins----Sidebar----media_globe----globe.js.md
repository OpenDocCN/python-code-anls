# `ZeroNet\plugins\Sidebar\media_globe\globe.js`

```py
/**
 * dat.globe Javascript WebGL Globe Toolkit
 * http://dataarts.github.com/dat.globe
 *
 * Copyright 2011 Data Arts Team, Google Creative Lab
 *
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// 创建一个名为 DAT 的全局对象
var DAT = DAT || {};

// 创建 Globe 构造函数
DAT.Globe = function(container, opts) {
  // 如果 opts 为空，则设置为一个空对象
  opts = opts || {};

  // 如果 opts 中没有 colorFn 属性，则设置一个默认的颜色函数
  var colorFn = opts.colorFn || function(x) {
    var c = new THREE.Color();
    c.setHSL( ( 0.5 - (x * 2) ), Math.max(0.8, 1.0 - (x * 3)), 0.5 );
    return c;
  };
  // 如果 opts 中没有 imgDir 属性，则设置默认的图片目录
  var imgDir = opts.imgDir || '/globe/';

  // 创建包含地球着色器的对象
  var Shaders = {
    'earth' : {
      uniforms: {
        'texture': { type: 't', value: null }
      },
      vertexShader: [
        'varying vec3 vNormal;',
        'varying vec2 vUv;',
        'void main() {',
          'gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );',
          'vNormal = normalize( normalMatrix * normal );',
          'vUv = uv;',
        '}'
      ].join('\n'),
      fragmentShader: [
        'uniform sampler2D texture;',
        'varying vec3 vNormal;',
        'varying vec2 vUv;',
        'void main() {',
          'vec3 diffuse = texture2D( texture, vUv ).xyz;',
          'float intensity = 1.05 - dot( vNormal, vec3( 0.0, 0.0, 1.0 ) );',
          'vec3 atmosphere = vec3( 1.0, 1.0, 1.0 ) * pow( intensity, 3.0 );',
          'gl_FragColor = vec4( diffuse + atmosphere, 1.0 );',
        '}'
      ].join('\n')
    },
    'atmosphere' : {  # 创建名为'atmosphere'的对象
      uniforms: {},  # 在'atmosphere'对象中创建名为uniforms的空对象
      vertexShader: [  # 在'atmosphere'对象中创建名为vertexShader的数组
        'varying vec3 vNormal;',  # 定义varying变量vNormal
        'void main() {',  # 定义主函数
          'vNormal = normalize( normalMatrix * normal );',  # 计算并赋值vNormal
          'gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );',  # 计算并赋值gl_Position
        '}'  # 结束主函数
      ].join('\n'),  # 将数组中的字符串用换行符连接成一个字符串
      fragmentShader: [  # 在'atmosphere'对象中创建名为fragmentShader的数组
        'varying vec3 vNormal;',  # 定义varying变量vNormal
        'void main() {',  # 定义主函数
          'float intensity = pow( 0.8 - dot( vNormal, vec3( 0, 0, 1.0 ) ), 12.0 );',  # 计算光照强度
          'gl_FragColor = vec4( 1.0, 1.0, 1.0, 1.0 ) * intensity;',  # 设置片元颜色
        '}'  # 结束主函数
      ].join('\n')  # 将数组中的字符串用换行符连接成一个字符串
    }
  };

  var camera, scene, renderer, w, h;  # 声明变量camera, scene, renderer, w, h
  var mesh, atmosphere, point, running;  # 声明变量mesh, atmosphere, point, running

  var overRenderer;  # 声明变量overRenderer
  var running = true;  # 初始化变量running为true

  var curZoomSpeed = 0;  # 声明变量curZoomSpeed并初始化为0
  var zoomSpeed = 50;  # 声明变量zoomSpeed并初始化为50

  var mouse = { x: 0, y: 0 }, mouseOnDown = { x: 0, y: 0 };  # 声明变量mouse和mouseOnDown为包含x和y属性的对象，并初始化为{x: 0, y: 0}

  var rotation = { x: 0, y: 0 },  # 声明变量rotation为包含x和y属性的对象，并初始化为{x: 0, y: 0}
      target = { x: Math.PI*3/2, y: Math.PI / 6.0 },  # 声明变量target为包含x和y属性的对象，并初始化为{x: Math.PI*3/2, y: Math.PI / 6.0}
      targetOnDown = { x: 0, y: 0 };  # 声明变量targetOnDown为包含x和y属性的对象，并初始化为{x: 0, y: 0}

  var distance = 100000, distanceTarget = 100000;  # 声明变量distance和distanceTarget并初始化为100000
  var padding = 10;  # 声明变量padding并初始化为10
  var PI_HALF = Math.PI / 2;  # 声明变量PI_HALF并初始化为Math.PI / 2

  function init() {  # 定义函数init

    container.style.color = '#fff';  # 设置container的文本颜色为白色
    container.style.font = '13px/20px Arial, sans-serif';  # 设置container的字体样式

    var shader, uniforms, material;  # 声明变量shader, uniforms, material
    w = container.offsetWidth || window.innerWidth;  # 如果container有offsetWidth属性则赋值给w，否则赋值为window.innerWidth
    h = container.offsetHeight || window.innerHeight;  # 如果container有offsetHeight属性则赋值给h，否则赋值为window.innerHeight

    camera = new THREE.PerspectiveCamera(30, w / h, 1, 10000);  # 创建透视相机对象并赋值给camera
    camera.position.z = distance;  # 设置camera的z坐标为distance

    scene = new THREE.Scene();  # 创建场景对象并赋值给scene

    var geometry = new THREE.SphereGeometry(200, 40, 30);  # 创建球体几何体对象并赋值给geometry

    shader = Shaders['earth'];  # 从Shaders对象中获取'earth'属性的值并赋值给shader
    uniforms = THREE.UniformsUtils.clone(shader.uniforms);  # 克隆shader.uniforms对象并赋值给uniforms

    uniforms['texture'].value = THREE.ImageUtils.loadTexture(imgDir+'world.jpg');  # 设置uniforms['texture']的值为加载的纹理图片

    material = new THREE.ShaderMaterial({  # 创建着色器材质对象并赋值给material

          uniforms: uniforms,  # 设置uniforms属性
          vertexShader: shader.vertexShader,  # 设置vertexShader属性
          fragmentShader: shader.fragmentShader  # 设置fragmentShader属性

        });

    mesh = new THREE.Mesh(geometry, material);  # 创建网格对象并赋值给mesh
    mesh.rotation.y = Math.PI;  # 设置mesh的y轴旋转角度为Math.PI
    scene.add(mesh);  # 将mesh添加到场景中

    shader = Shaders['atmosphere'];  # 从Shaders对象中获取'atmosphere'属性的值并赋值给shader
    # 克隆着色器的统一变量
    uniforms = THREE.UniformsUtils.clone(shader.uniforms);

    # 创建着色器材质
    material = new THREE.ShaderMaterial({
          uniforms: uniforms,  # 使用克隆的统一变量
          vertexShader: shader.vertexShader,  # 顶点着色器
          fragmentShader: shader.fragmentShader,  # 片段着色器
          side: THREE.BackSide,  # 设置材质的渲染面
          blending: THREE.AdditiveBlending,  # 设置混合模式
          transparent: true  # 设置材质为透明
        });

    # 创建网格对象
    mesh = new THREE.Mesh(geometry, material);
    mesh.scale.set( 1.1, 1.1, 1.1 );  # 设置网格的缩放
    scene.add(mesh);  # 将网格添加到场景中

    # 创建立方体几何体
    geometry = new THREE.BoxGeometry(2.75, 2.75, 1);
    geometry.applyMatrix(new THREE.Matrix4().makeTranslation(0,0,-0.5));  # 对几何体应用平移变换

    # 创建网格对象
    point = new THREE.Mesh(geometry);

    # 创建WebGL渲染器
    renderer = new THREE.WebGLRenderer({antialias: true});
    renderer.setSize(w, h);  # 设置渲染器的大小
    renderer.setClearColor( 0x212121, 1 );  # 设置渲染器的清除颜色

    renderer.domElement.style.position = 'relative';  # 设置渲染器DOM元素的定位方式

    container.appendChild(renderer.domElement);  # 将渲染器DOM元素添加到容器中

    container.addEventListener('mousedown', onMouseDown, false);  # 添加鼠标按下事件监听器

    # 根据浏览器支持情况添加滚轮事件监听器
    if ('onwheel' in document) {
      container.addEventListener('wheel', onMouseWheel, false);
    } else {
      container.addEventListener('mousewheel', onMouseWheel, false);
    }

    document.addEventListener('keydown', onDocumentKeyDown, false);  # 添加键盘按下事件监听器

    window.addEventListener('resize', onWindowResize, false);  # 添加窗口大小改变事件监听器

    # 添加鼠标移入容器事件监听器
    container.addEventListener('mouseover', function() {
      overRenderer = true;
    }, false);

    # 添加鼠标移出容器事件监听器
    container.addEventListener('mouseout', function() {
      overRenderer = false;
    }, false);
  }

  # 添加数据函数
  function addData(data, opts) {
    var lat, lng, size, color, i, step, colorFnWrapper;

    opts.animated = opts.animated || false;  # 设置是否动画
    this.is_animated = opts.animated;  # 设置是否动画
    opts.format = opts.format || 'magnitude';  # 设置数据格式，默认为'magnitude'，也可以是'legend'
    if (opts.format === 'magnitude') {
      step = 3;  # 步长为3
      colorFnWrapper = function(data, i) { return colorFn(data[i+2]); }  # 颜色函数包装器
    } else if (opts.format === 'legend') {
      step = 4;  # 步长为4
      colorFnWrapper = function(data, i) { return colorFn(data[i+3]); }  # 颜色函数包装器
    } else if (opts.format === 'peer') {
      // 如果格式为'peer'，则使用colorFnWrapper函数对数据进行处理
      colorFnWrapper = function(data, i) { return colorFn(data[i+2]); }
    } else {
      // 如果格式不支持，则抛出错误
      throw('error: format not supported: '+opts.format);
    }

    // 如果设置为动画模式
    if (opts.animated) {
      // 如果基础几何体未定义
      if (this._baseGeometry === undefined) {
        // 创建一个新的THREE.Geometry对象作为基础几何体
        this._baseGeometry = new THREE.Geometry();
        // 遍历数据，以步长为step
        for (i = 0; i < data.length; i += step) {
          // 获取经纬度数据
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
    // 如果容器的样式光标不是“move”，则返回false
    if (container.style.cursor != "move") return false;
    // 阻止事件的默认行为
    event.preventDefault();
    // 如果鼠标悬停在渲染器上
    if (overRenderer) {
      // 如果事件有deltaY属性
      if (event.deltaY) {
        // 调用zoom函数，根据deltaY的值进行缩放
        zoom(-event.deltaY * (event.deltaMode == 0 ? 1 : 50));
      } else {
        // 调用zoom函数，根据wheelDeltaY的值进行缩放
        zoom(event.wheelDeltaY * 0.3);
      }
    }
    // 返回false
    return false;
  }

  // 当按下键盘时触发的事件处理函数
  function onDocumentKeyDown(event) {
    // 根据按下的键盘按键编码进行不同的操作
    switch (event.keyCode) {
      // 如果按下的是上箭头键
      case 38:
        // 调用zoom函数，放大100个单位
        zoom(100);
        // 阻止事件的默认行为
        event.preventDefault();
        break;
      // 如果按下的是下箭头键
      case 40:
        // 调用zoom函数，缩小100个单位
        zoom(-100);
        // 阻止事件的默认行为
        event.preventDefault();
        break;
    }
  }

  // 当窗口大小改变时触发的事件处理函数
  function onWindowResize( event ) {
    // 更新相机的长宽比
    camera.aspect = container.offsetWidth / container.offsetHeight;
    // 更新相机的投影矩阵
    camera.updateProjectionMatrix();
    // 更新渲染器的大小
    renderer.setSize( container.offsetWidth, container.offsetHeight );
  }

  // 缩放函数
  function zoom(delta) {
    // 更新目标距离
    distanceTarget -= delta;
    // 限制目标距离的范围在855到350之间
    distanceTarget = distanceTarget > 855 ? 855 : distanceTarget;
    distanceTarget = distanceTarget < 350 ? 350 : distanceTarget;
  }

  // 动画函数
  function animate() {
    // 如果不在运行状态，则返回
    if (!running) return
    // 请求下一帧动画
    requestAnimationFrame(animate);
    // 渲染场景
    render();
  }

  // 渲染函数
  function render() {
    // 调用zoom函数，根据curZoomSpeed的值进行缩放
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

    // 相机朝向网格对象的位置
    camera.lookAt(mesh.position);

    // 使用相机渲染场景
    renderer.render(scene, camera);
  }

  // 卸载函数
  function unload() {
    // 将运行状态设置为false
    running = false
    // 移除鼠标按下事件监听器
    container.removeEventListener('mousedown', onMouseDown, false);
    // 如果浏览器支持wheel事件，则移除鼠标滚轮事件监听器
    if ('onwheel' in document) {
      container.removeEventListener('wheel', onMouseWheel, false);
    } else {
      // 否则移除鼠标滚轮事件监听器
      container.removeEventListener('mousewheel', onMouseWheel, false);
    }
    // 移除键盘按下事件监听器
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
# 代码结尾的分号，可能是一个语法错误，需要检查并修复
```