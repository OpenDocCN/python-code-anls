# `ZeroNet\plugins\Sidebar\media_globe\Detector.js`

```py
// 作者信息
// alteredq / http://alteredqualia.com/
// mr.doob / http://mrdoob.com/

// Detector 对象
Detector = {
  
  // 检测浏览器是否支持 canvas
  canvas : !! window.CanvasRenderingContext2D,
  // 检测浏览器是否支持 WebGL
  webgl : ( function () { 
    try { 
      return !! window.WebGLRenderingContext && !! document.createElement( 'canvas' ).getContext( 'experimental-webgl' ); 
    } catch( e ) { 
      return false; 
    } 
  } )(),
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
      // 显示相应的错误信息
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

    // 获取父元素和 id
    parent = parameters.parent !== undefined ? parameters.parent : document.body;
    id = parameters.id !== undefined ? parameters.id : 'oldie';

    // 获取 WebGL 错误信息的 div 元素
    domElement = Detector.getWebGLErrorMessage();
    domElement.id = id;

    // 将错误信息添加到父元素中
    parent.appendChild( domElement );

  }

};
```