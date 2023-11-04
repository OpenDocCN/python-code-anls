# BasicComputerGames源码解析 1

# `00_Alternate_Languages/01_Acey_Ducey/elm/docs/app.min.js`

Acey-Ducey is a card game played between two players. The dealer, who is the computer, deals two cards face up to each player.

To begin the game, each player is given two cards face up. The player has the option to bet or not to bet. If the player decides to bet, they must specify the amount they are willing to bet. If the player does not bet, they must either fold or decide to decide what they want to do with their cards.

If the player decides to bet, they must decide what they want to bet on. They have a range of choices to choose from, and the computer will tell them if their bet is valid. Once the player has decided what to bet, the computer will deal the second card face up to each player.

The game continues until one of the players wins or the game is ended due to a rule violation or outcome that the players agreed upon. The game is won by the player who makes the highest score or wins a certain number of hands. The game is filled with surprises and challenges that make it exciting for all players.


```
!function(q){"use strict";function B(n,r,t){return t.a=n,t.f=r,t}function r(t){return B(2,t,function(r){return function(n){return t(r,n)}})}function t(e){return B(3,e,function(t){return function(r){return function(n){return e(t,r,n)}}})}function F(u){return B(4,u,function(e){return function(t){return function(r){return function(n){return u(e,t,r,n)}}}})}function J(a){return B(5,a,function(u){return function(e){return function(t){return function(r){return function(n){return a(u,e,t,r,n)}}}}})}function s(n,r,t){return 2===n.a?n.f(r,t):n(r)(t)}function v(n,r,t,e){return 3===n.a?n.f(r,t,e):n(r)(t)(e)}function b(n,r,t,e,u){return 4===n.a?n.f(r,t,e,u):n(r)(t)(e)(u)}function M(n,r,t,e,u,a){return 5===n.a?n.f(r,t,e,u,a):n(r)(t)(e)(u)(a)}function O(n,r){for(var t,e=[],u=S(n,r,0,e);u&&(t=e.pop());u=S(t.a,t.b,0,e));return u}function S(n,r,t,e){if(n===r)return!0;if("object"!=typeof n||null===n||null===r)return"function"==typeof n&&H(5),!1;if(100<t)return e.push({a:n,b:r}),!0;for(var u in n.$<0&&(n=Xn(n),r=Xn(r)),n)if(!S(n[u],r[u],t+1,e))return!1;return!0}function l(n,r,t){if("object"!=typeof n)return n===r?0:n<r?-1:1;if(void 0===n.$)return(t=l(n.a,r.a))||(t=l(n.b,r.b))?t:l(n.c,r.c);for(;n.b&&r.b&&!(t=l(n.a,r.a));n=n.b,r=r.b);return t||(n.b?1:r.b?-1:0)}var P=0;function a(n,r){var t,e={};for(t in n)e[t]=n[t];for(t in r)e[t]=r[t];return e}var d={$:0};function G(n,r){return{$:1,a:n,b:r}}var n=r(G);function h(n){for(var r=d,t=n.length;t--;)r={$:1,a:n[t],b:r};return r}var I=t(function(n,r,t){for(var e=Array(n),u=0;u<n;u++)e[u]=t(r+u);return e}),K=r(function(n,r){for(var t=Array(n),e=0;e<n&&r.b;e++)t[e]=r.a,r=r.b;return t.length=e,{a:t,b:r}});function H(n){throw Error("https://github.com/elm/core/blob/1.0.0/hints/"+n+".md")}var W=Math.ceil,Q=Math.floor,U=Math.log;var V={$:2,b:function(n){return"string"==typeof n?w(n):n instanceof String?w(n+""):$("a STRING",n)}};var X=r(function(n,r){return{$:6,d:n,b:r}});var Z=r(function(n,r){return{$:9,f:n,g:[r]}}),nn=r(g);function g(n,r){switch(n.$){case 2:return n.b(r);case 5:return null===r?w(n.c):$("null",r);case 3:return tn(r)?rn(n.b,r,h):$("a LIST",r);case 4:return tn(r)?rn(n.b,r,en):$("an ARRAY",r);case 6:var t=n.d;if("object"!=typeof r||null===r||!(t in r))return $("an OBJECT with a field named `"+t+"`",r);var e=g(n.b,r[t]);return C(e)?e:y(s(nr,t,e.a));case 7:t=n.e;if(!tn(r))return $("an ARRAY",r);if(r.length<=t)return $("a LONGER array. Need index "+t+" but only see "+r.length+" entries",r);e=g(n.b,r[t]);return C(e)?e:y(s(rr,t,e.a));case 8:if("object"!=typeof r||null===r||tn(r))return $("an OBJECT",r);var u,a=d;for(u in r)if(r.hasOwnProperty(u)){e=g(n.b,r[u]);if(!C(e))return y(s(nr,u,e.a));a={$:1,a:{a:u,b:e.a},b:a}}return w(ar(a));case 9:for(var i=n.f,o=n.g,f=0;f<o.length;f++){e=g(o[f],r);if(!C(e))return e;i=i(e.a)}return w(i);case 10:e=g(n.b,r);return C(e)?g(n.h(e.a),r):e;case 11:for(var c=d,v=n.g;v.b;v=v.b){e=g(v.a,r);if(C(e))return e;c={$:1,a:e.a,b:c}}return y(tr(ar(c)));case 1:return y(s(Zn,n.a,r));case 0:return w(n.a)}}function rn(n,r,t){for(var e=r.length,u=Array(e),a=0;a<e;a++){var i=g(n,r[a]);if(!C(i))return y(s(rr,a,i.a));u[a]=i.a}return w(t(u))}function tn(n){return Array.isArray(n)||"undefined"!=typeof FileList&&n instanceof FileList}function en(r){return s(wr,r.length,function(n){return r[n]})}function $(n,r){return y(s(Zn,"Expecting "+n,r))}function f(n,r){if(n===r)return!0;if(n.$!==r.$)return!1;switch(n.$){case 0:case 1:return n.a===r.a;case 2:return n.b===r.b;case 5:return n.c===r.c;case 3:case 4:case 8:return f(n.b,r.b);case 6:return n.d===r.d&&f(n.b,r.b);case 7:return n.e===r.e&&f(n.b,r.b);case 9:return n.f===r.f&&un(n.g,r.g);case 10:return n.h===r.h&&f(n.b,r.b);case 11:return un(n.g,r.g)}}function un(n,r){var t=n.length;if(t!==r.length)return!1;for(var e=0;e<t;e++)if(!f(n[e],r[e]))return!1;return!0}function an(n){return n}function on(n){return{$:0,a:n}}var fn=r(function(n,r){return{$:3,b:n,d:r}});var cn=0;function vn(n){n={$:0,e:cn++,f:n,g:null,h:[]};return hn(n),n}function sn(r){return{$:2,b:function(n){n({$:0,a:vn(r)})},c:null}}function bn(n,r){n.h.push(r),hn(n)}var ln=!1,dn=[];function hn(n){if(dn.push(n),!ln){for(ln=!0;n=dn.shift();)!function(r){for(;r.f;){var n=r.f.$;if(0===n||1===n){for(;r.g&&r.g.$!==n;)r.g=r.g.i;if(!r.g)return;r.f=r.g.b(r.f.a),r.g=r.g.i}else{if(2===n)return r.f.c=r.f.b(function(n){r.f=n,hn(r)});if(5===n){if(0===r.h.length)return;r.f=r.f.b(r.h.shift())}else r.g={$:3===n?0:1,b:r.f.b,i:r.g},r.f=r.f.d}}}(n);ln=!1}}function gn(n,r,t,e,u,a){var n=s(nn,n,r?r.flags:void 0),i=(C(n)||H(2),{}),r=t(n.a),o=r.a,f=a(c,o),t=function(n,r){var t,e;for(e in p){var u=p[e];u.a&&((t=t||{})[e]=u.a(e,r)),n[e]=function(n,r){var e={g:r,h:void 0},u=n.c,a=n.d,i=n.e,o=n.f;function f(t){return s(fn,f,{$:5,b:function(n){var r=n.a;return 0===n.$?v(a,e,r,t):i&&o?b(u,e,r.i,r.j,t):v(u,e,i?r.i:r.j,t)}})}return e.h=vn(s(fn,f,n.b))}(u,r)}return t}(i,c);function c(n,r){n=s(e,n,o);f(o=n.a,r),wn(i,n.b,u(o))}return wn(i,r.b,u(o)),t?{ports:t}:{}}var p={};var e=r(function(r,t){return{$:2,b:function(n){r.g(t),n({$:0,a:P})},c:null}});function $n(r){return function(n){return{$:1,k:r,l:n}}}function pn(n){return{$:2,m:n}}var mn=[],yn=!1;function wn(n,r,t){if(mn.push({p:n,q:r,r:t}),!yn){yn=!0;for(var e;e=mn.shift();)!function(n,r,t){var e,u={};for(e in jn(!0,r,u,null),jn(!1,t,u,null),n)bn(n[e],{$:"fx",a:u[e]||{i:d,j:d}})}(e.p,e.q,e.r);yn=!1}}function jn(n,r,t,e){switch(r.$){case 1:var u=r.k,a=function(n,r,t,e){function u(n){for(var r=t;r;r=r.t)n=r.s(n);return n}return s(n?p[r].e:p[r].f,u,e)}(n,u,e,r.l);return void(t[u]=function(n,r,t){return t=t||{i:d,j:d},n?t.i={$:1,a:r,b:t.i}:t.j={$:1,a:r,b:t.j},t}(n,a,t[u]));case 2:for(var i=r.m;i.b;i=i.b)jn(n,i.a,t,e);return;case 3:jn(n,r.o,t,{s:r.n,t:e})}}var An;var Cn="undefined"!=typeof document?document:{};function kn(n){return{$:0,a:n}}var c=r(function(a,i){return r(function(n,r){for(var t=[],e=0;r.b;r=r.b){var u=r.a;e+=u.b||0,t.push(u)}return e+=t.length,{$:1,c:i,d:Ln(n),e:t,f:a,b:e}})})(void 0);r(function(a,i){return r(function(n,r){for(var t=[],e=0;r.b;r=r.b){var u=r.a;e+=u.b.b||0,t.push(u)}return e+=t.length,{$:2,c:i,d:Ln(n),e:t,f:a,b:e}})})(void 0);var _n=r(function(n,r){return{$:"a0",n:n,o:r}}),En=r(function(n,r){return{$:"a1",n:n,o:r}}),Nn=r(function(n,r){return{$:"a2",n:n,o:r}}),Dn=r(function(n,r){return{$:"a3",n:n,o:r}});var Tn;function Ln(n){for(var r={};n.b;n=n.b){var t,e=n.a,u=e.$,a=e.n,e=e.o;"a2"===u?"className"===a?xn(r,a,e):r[a]=e:(t=r[u]||(r[u]={}),"a3"===u&&"class"===a?xn(t,a,e):t[a]=e)}return r}function xn(n,r,t){var e=n[r];n[r]=e?e+" "+t:t}function m(n,r){var t=n.$;if(5===t)return m(n.k||(n.k=n.m()),r);if(0===t)return Cn.createTextNode(n.a);if(4===t){for(var e=n.k,u=n.j;4===e.$;)"object"!=typeof u?u=[u,e.j]:u.push(e.j),e=e.k;var a={j:u,p:r};return(i=m(e,a)).elm_event_node_ref=a,i}if(3===t)return zn(i=n.h(n.g),r,n.d),i;var i=n.f?Cn.createElementNS(n.f,n.c):Cn.createElement(n.c);An&&"a"==n.c&&i.addEventListener("click",An(i)),zn(i,r,n.d);for(var o=n.e,f=0;f<o.length;f++)i.appendChild(m(1===t?o[f]:o[f].b,r));return i}function zn(n,r,t){for(var e in t){var u=t[e];"a1"===e?function(n,r){var t,e=n.style;for(t in r)e[t]=r[t]}(n,u):"a0"===e?function(n,r,t){var e,u=n.elmFs||(n.elmFs={});for(e in t){var a=t[e],i=u[e];if(a){if(i){if(i.q.$===a.$){i.q=a;continue}n.removeEventListener(e,i)}i=function(f,n){function c(n){var r=c.q,t=g(r.a,n);if(C(t)){for(var e,r=Cr(r),t=t.a,u=r?r<3?t.a:t.t:t,a=1==r?t.b:3==r&&t.R,i=(a&&n.stopPropagation(),(2==r?t.b:3==r&&t.O)&&n.preventDefault(),f);e=i.j;){if("function"==typeof e)u=e(u);else for(var o=e.length;o--;)u=e[o](u);i=i.p}i(u,a)}}return c.q=n,c}(r,a),n.addEventListener(e,i,Tn&&{passive:Cr(a)<2}),u[e]=i}else n.removeEventListener(e,i),u[e]=void 0}}(n,r,u):"a3"===e?function(n,r){for(var t in r){var e=r[t];void 0!==e?n.setAttribute(t,e):n.removeAttribute(t)}}(n,u):"a4"===e?function(n,r){for(var t in r){var e=r[t],u=e.f,e=e.o;void 0!==e?n.setAttributeNS(u,t,e):n.removeAttributeNS(u,t)}}(n,u):("value"!==e&&"checked"!==e||n[e]!==u)&&(n[e]=u)}}try{window.addEventListener("t",null,Object.defineProperty({},"passive",{get:function(){Tn=!0}}))}catch(n){}function Rn(n,r){var t=[];return T(n,r,t,0),t}function D(n,r,t,e){r={$:r,r:t,s:e,t:void 0,u:void 0};return n.push(r),r}function T(n,r,t,e){if(n!==r){var u=n.$,a=r.$;if(u!==a){if(1!==u||2!==a)return void D(t,0,e,r);r=function(n){for(var r=n.e,t=r.length,e=Array(t),u=0;u<t;u++)e[u]=r[u].b;return{$:1,c:n.c,d:n.d,e:e,f:n.f,b:n.b}}(r),a=1}switch(a){case 5:for(var i=n.l,o=r.l,f=i.length,c=f===o.length;c&&f--;)c=i[f]===o[f];if(c)return void(r.k=n.k);r.k=r.m();var v=[];return T(n.k,r.k,v,0),void(0<v.length&&D(t,1,e,v));case 4:for(var s=n.j,b=r.j,l=!1,d=n.k;4===d.$;)l=!0,"object"!=typeof s?s=[s,d.j]:s.push(d.j),d=d.k;for(var h=r.k;4===h.$;)l=!0,"object"!=typeof b?b=[b,h.j]:b.push(h.j),h=h.k;return l&&s.length!==b.length?void D(t,0,e,r):((l?function(n,r){for(var t=0;t<n.length;t++)if(n[t]!==r[t])return;return 1}(s,b):s===b)||D(t,2,e,b),void T(d,h,t,e+1));case 0:return void(n.a!==r.a&&D(t,3,e,r.a));case 1:return void Yn(n,r,t,e,Bn);case 2:return void Yn(n,r,t,e,Fn);case 3:if(n.h!==r.h)return void D(t,0,e,r);v=qn(n.d,r.d),v=(v&&D(t,4,e,v),r.i(n.g,r.g));v&&D(t,5,e,v)}}}function Yn(n,r,t,e,u){var a;n.c!==r.c||n.f!==r.f?D(t,0,e,r):((a=qn(n.d,r.d))&&D(t,4,e,a),u(n,r,t,e))}function qn(n,r,t){var e,u,a,i,o;for(u in n)"a1"===u||"a0"===u||"a3"===u||"a4"===u?(a=qn(n[u],r[u]||{},u))&&((e=e||{})[u]=a):u in r?(a=n[u])===(i=r[u])&&"value"!==u&&"checked"!==u||"a0"===t&&function(n,r){return n.$==r.$&&f(n.a,r.a)}(a,i)||((e=e||{})[u]=i):(e=e||{})[u]=t?"a1"===t?"":"a0"===t||"a3"===t?void 0:{f:n[u].f,o:void 0}:"string"==typeof n[u]?"":null;for(o in r)o in n||((e=e||{})[o]=r[o]);return e}function Bn(n,r,t,e){var u=n.e,a=r.e,n=u.length,r=a.length;r<n?D(t,6,e,{v:r,i:n-r}):n<r&&D(t,7,e,{v:n,e:a});for(var i=n<r?n:r,o=0;o<i;o++){var f=u[o];T(f,a[o],t,++e),e+=f.b||0}}function Fn(n,r,t,e){for(var u=[],a={},i=[],o=n.e,f=r.e,c=o.length,v=f.length,s=0,b=0,l=e;s<c&&b<v;){var d=o[s],h=f[b],g=d.a,$=h.a,p=d.b,m=h.b,y=void 0,w=void 0;if(g===$)T(p,m,u,++l),l+=p.b||0,s++,b++;else{var j,A,C,k,_=o[s+1],E=f[b+1];if(_&&(A=_.b,w=$===(j=_.a)),E&&(k=E.b,y=g===(C=E.a)),y&&w)T(p,k,u,++l),Mn(a,u,g,m,b,i),l+=p.b||0,On(a,u,g,A,++l),l+=A.b||0,s+=2,b+=2;else if(y)l++,Mn(a,u,$,m,b,i),T(p,k,u,l),l+=p.b||0,s+=1,b+=2;else if(w)On(a,u,g,p,++l),l+=p.b||0,T(A,m,u,++l),l+=A.b||0,s+=2,b+=1;else{if(!_||j!==C)break;On(a,u,g,p,++l),Mn(a,u,$,m,b,i),l+=p.b||0,T(A,k,u,++l),l+=A.b||0,s+=2,b+=2}}}for(;s<c;){p=(d=o[s]).b;On(a,u,d.a,p,++l),l+=p.b||0,s++}for(;b<v;){var N=N||[];Mn(a,u,(h=f[b]).a,h.b,void 0,N),b++}(0<u.length||0<i.length||N)&&D(t,8,e,{w:u,x:i,y:N})}var Jn="_elmW6BL";function Mn(n,r,t,e,u,a){var i,o=n[t];if(o)return 1===o.c?(a.push({r:u,A:o}),o.c=2,T(o.z,e,i=[],o.r),o.r=u,void(o.s.s={w:i,A:o})):void Mn(n,r,t+Jn,e,u,a);a.push({r:u,A:o={c:0,z:e,r:u,s:void 0}}),n[t]=o}function On(n,r,t,e,u){var a,i=n[t];if(i)return 0===i.c?(i.c=2,T(e,i.z,a=[],u),void D(r,9,u,{w:a,A:i})):void On(n,r,t+Jn,e,u);a=D(r,9,u,void 0),n[t]={c:1,z:e,r:u,s:a}}function Sn(n,r,t,e){!function n(r,t,e,u,a,i,o){var f=e[u];var c=f.r;for(;c===a;){var v,s=f.$;if(1===s?Sn(r,t.k,f.s,o):8===s?(f.t=r,f.u=o,0<(v=f.s.w).length&&n(r,t,v,0,a,i,o)):9===s?(f.t=r,f.u=o,(s=f.s)&&(s.A.s=r,0<(v=s.w).length&&n(r,t,v,0,a,i,o))):(f.t=r,f.u=o),!(f=e[++u])||(c=f.r)>i)return u}var b=t.$;if(4===b){for(var l=t.k;4===l.$;)l=l.k;return n(r,l,e,u,a+1,i,r.elm_event_node_ref)}var d=t.e;var h=r.childNodes;for(var g=0;g<d.length;g++){var $=1===b?d[g]:d[g].b,p=++a+($.b||0);if(a<=c&&c<=p&&(u=n(h[g],$,e,u,a,p,o),!(f=e[u])||(c=f.r)>i))return u;a=p}return u}(n,r,t,0,0,r.b,e)}function Pn(n,r,t,e){return 0===t.length?n:(Sn(n,r,t,e),Gn(n,t))}function Gn(n,r){for(var t=0;t<r.length;t++){var e=r[t],u=e.t,e=function(n,r){switch(r.$){case 0:return function(n,r,t){var e=n.parentNode,r=m(r,t);r.elm_event_node_ref||(r.elm_event_node_ref=n.elm_event_node_ref);e&&r!==n&&e.replaceChild(r,n);return r}(n,r.s,r.u);case 4:return zn(n,r.u,r.s),n;case 3:return n.replaceData(0,n.length,r.s),n;case 1:return Gn(n,r.s);case 2:return n.elm_event_node_ref?n.elm_event_node_ref.j=r.s:n.elm_event_node_ref={j:r.s,p:r.u},n;case 6:for(var t=r.s,e=0;e<t.i;e++)n.removeChild(n.childNodes[t.v]);return n;case 7:for(var u=(t=r.s).e,e=t.v,a=n.childNodes[e];e<u.length;e++)n.insertBefore(m(u[e],r.u),a);return n;case 9:if(!(t=r.s))return n.parentNode.removeChild(n),n;var i=t.A;return void 0!==i.r&&n.parentNode.removeChild(n),i.s=Gn(n,t.w),n;case 8:return function(n,r){for(var t=r.s,e=function(n,r){if(n){for(var t=Cn.createDocumentFragment(),e=0;e<n.length;e++){var u=n[e].A;t.appendChild(2===u.c?u.s:m(u.z,r.u))}return t}}(t.y,r),u=(n=Gn(n,t.w),t.x),a=0;a<u.length;a++){var i=u[a],o=i.A,o=2===o.c?o.s:m(o.z,r.u);n.insertBefore(o,n.childNodes[i.r])}e&&n.appendChild(e);return n}(n,r);case 5:return r.s(n);default:H(10)}}(u,e);u===n&&(n=e)}return n}function In(n){if(3===n.nodeType)return{$:0,a:n.textContent};if(1!==n.nodeType)return{$:0,a:""};for(var r=d,t=n.attributes,e=t.length;e--;)var u=t[e],r={$:1,a:s(Dn,u.name,u.value),b:r};for(var a=n.tagName.toLowerCase(),i=d,o=n.childNodes,e=o.length;e--;)i={$:1,a:In(o[e]),b:i};return v(c,a,r,i)}var Kn=F(function(r,n,t,i){return gn(n,i,r.aB,r.aJ,r.aH,function(t,n){var e=r.aK,u=i.node,a=In(u);return Wn(n,function(n){var n=e(n),r=Rn(a,n);u=Pn(u,a,r,t),a=n})})}),Hn="undefined"!=typeof requestAnimationFrame?requestAnimationFrame:function(n){return setTimeout(n,1e3/60)};function Wn(t,e){e(t);var u=0;function a(){u=1===u?0:(Hn(a),e(t),1)}return function(n,r){t=n,r?(e(t),2===u&&(u=1)):(0===u&&Hn(a),u=2)}}function Qn(n){return n}function i(n){return{$:2,a:n}}function Un(n){var r=n.b;return s(zr,1664525*n.a+r>>>0,r)}var o=n,Vn=t(function(n,r,t){for(;;){if(-2===t.$)return r;var e=t.d,u=n,a=v(n,t.b,t.c,v(Vn,n,r,t.e));n=u,r=a,t=e}}),Xn=function(n){return v(Vn,t(function(n,r,t){return s(o,{a:n,b:r},t)}),d,n)},y=function(n){return{$:1,a:n}},Zn=r(function(n,r){return{$:3,a:n,b:r}}),nr=r(function(n,r){return{$:0,a:n,b:r}}),rr=r(function(n,r){return{$:1,a:n,b:r}}),w=function(n){return{$:0,a:n}},tr=function(n){return{$:2,a:n}},j=function(n){return{$:0,a:n}},A={$:1},er=function(n){return n+""},ur=t(function(n,r,t){for(;;){if(!t.b)return r;var e=t.b,u=n,a=s(n,t.a,r);n=u,r=a,t=e}}),ar=function(n){return v(ur,o,d,n)},ir=F(function(n,r,t,e){return{$:0,a:n,b:r,c:t,d:e}}),or=[],fr=W,cr=r(function(n,r){return U(r)/U(n)}),vr=fr(s(cr,2,32)),sr=b(ir,0,vr,or,or),br=I,lr=Q,dr=function(n){return n.length},hr=r(function(n,r){return 0<l(n,r)?n:r}),gr=K,$r=r(function(n,r){for(;;){var t=s(gr,32,n),e=t.b,t=s(o,{$:0,a:t.a},r);if(!e.b)return ar(t);n=e,r=t}}),pr=r(function(n,r){for(;;){var t=fr(r/32);if(1===t)return s(gr,32,n).a;n=s($r,n,d),r=t}}),mr=r(function(n,r){var t,e;return r.b?(e=lr(s(cr,32,(t=32*r.b)-1)),n=n?ar(r.f):r.f,n=s(pr,n,r.b),b(ir,dr(r.e)+t,s(hr,5,e*vr),n,r.e)):b(ir,dr(r.e),vr,or,r.e)}),yr=J(function(n,r,t,e,u){for(;;){if(r<0)return s(mr,!1,{f:e,b:t/32|0,e:u});var a={$:1,a:v(br,32,r,n)};n=n,r=r-32,t=t,e=s(o,a,e),u=u}}),wr=r(function(n,r){var t;return 0<n?M(yr,r,n-(t=n%32)-32,n,d,v(br,t,n-t,r)):sr}),C=function(n){return!n.$},jr=Z,Ar=function(n){return{$:0,a:n}},Cr=function(n){switch(n.$){case 0:return 0;case 1:return 1;case 2:return 2;default:return 3}},kr=function(n){for(var r=0,t=n.charCodeAt(0),e=43==t||45==t?1:0,u=e;u<n.length;++u){var a=n.charCodeAt(u);if(a<48||57<a)return A;r=10*r+a-48}return u==e?A:j(45==t?-r:r)},k=on,n=k(0),_r=F(function(n,r,t,e){var u,a,i,o;return e.b?(u=e.a,(e=e.b).b?(a=e.a,(e=e.b).b?(i=e.a,(e=e.b).b?(o=e.b,s(n,u,s(n,a,s(n,i,s(n,e.a,500<t?v(ur,n,r,ar(o)):b(_r,n,r,t+1,o)))))):s(n,u,s(n,a,s(n,i,r)))):s(n,u,s(n,a,r))):s(n,u,r)):r}),Er=t(function(n,r,t){return b(_r,n,r,0,t)}),Nr=r(function(t,n){return v(Er,r(function(n,r){return s(o,t(n),r)}),d,n)}),_=fn,Dr=r(function(r,n){return s(_,function(n){return k(r(n))},n)}),Tr=t(function(t,n,e){return s(_,function(r){return s(_,function(n){return k(s(t,r,n))},e)},n)}),Lr=e,xr=r(function(n,r){return sn(s(_,Lr(n),r))}),W=t(function(n,r,t){return s(Dr,function(n){return 0},(n=s(Nr,xr(n),r),v(Er,Tr(o),k(d),n)))}),I=(p.Task={b:n,c:W,d:t(function(n,r,t){return k(0)}),e:r(function(n,r){return s(Dr,n,r)}),f:void 0},$n("Task"),Kn),zr=r(function(n,r){return{$:0,a:n,b:r}});Fr=Qn;function Rr(n){return((n=277803737*((n=n.a)^n>>>4+(n>>>28)))>>>22^n)>>>0}function Yr(n){return{a:{a:{d:A,g:A,w:A},C:A,D:A,i:100,j:0},b:s(E,i,N)}}function qr(n){return{$:3,a:n}}function Br(n){return{a:n,b:!0}}var Fr,Q=s(_,function(n){return k(function(n){var r=Un(s(zr,0,1013904223));return Un(s(zr,r.a+n>>>0,r.b))}(n))},{$:2,b:function(n){n({$:0,a:Fr(Date.now())})},c:null}),Jr=r(function(n,r){return n(r)}),Mr=t(function(r,n,t){var e,u;return n.b?(e=n.b,u=(n=s(Jr,n.a,t)).b,s(_,function(n){return v(Mr,r,e,u)},s(Lr,r,n.a))):k(t)}),K=t(function(n,r,t){return k(t)}),Or=r(function(t,n){var e=n;return function(n){var n=e(n),r=n.b;return{a:t(n.a),b:r}}}),Sr=(p.Random={b:Q,c:Mr,d:K,e:r(function(n,r){return s(Or,n,r)}),f:void 0},$n("Random")),E=r(function(n,r){return Sr(s(Or,n,r))}),N=s(r(function(f,c){return function(n){var r=l(f,c)<0?{a:f,b:c}:{a:c,b:f},t=r.a,e=r.b-t+1;if(!(e-1&e))return{a:((e-1&Rr(n))>>>0)+t,b:Un(n)};for(var u=(-e>>>0)%e>>>0,a=n;;){var i=Rr(a),o=Un(a);if(l(i,u)>=0)return{a:i%e+t,b:o};a=o}}}),2,14),Pr=pn(d),L=pn(d),Gr=r(function(n,r){var t,e,u=r.a.d;return u.$?{a:r,b:L}:(u=u.a,(t=r.a.g).$?{a:r,b:L}:(t=t.a,e=r.a,O(n,u)||O(n,t)?{a:r,b:s(E,qr,N)}:l(u,n)<0&&l(n,t)<0?{a:a(r,{a:a(e,{d:A,g:A}),D:j({d:r.a.d,g:r.a.g,w:j(n)}),i:r.i+r.j}),b:s(E,i,N)}:0<l(r.j,r.i-r.j)?{a:a(r,{a:a(e,{d:A,g:A}),D:j({d:r.a.d,g:r.a.g,w:j(n)}),i:r.i-r.j,j:r.i-r.j}),b:s(E,i,N)}:{a:a(r,{a:a(e,{d:A,g:A}),D:j({d:r.a.d,g:r.a.g,w:j(n)}),i:r.i-r.j}),b:s(E,i,N)}))}),Z=r(function(n,r){switch(n.$){case 0:return{a:a(r,{j:n.a}),b:L};case 1:var t=kr(n.a);return t.$?{a:a(r,{C:j("Wrong input for bet")}),b:L}:0<l(t=t.a,r.i)?{a:a(r,{C:j("You cannot bet more than you have"),j:r.i}),b:L}:{a:a(r,{C:A,j:t}),b:L};case 2:var e,u=n.a,t=r.a.d;return 1===t.$?(e=r.a,13<u?{a:r,b:s(E,i,N)}:{a:a(r,{a:a(e,{d:j(u)})}),b:s(E,i,N)}):(e=r.a,l(u,t.a)<1?{a:a(r,{a:a(e,{d:j(u)})}),b:s(E,i,N)}:{a:a(r,{a:a(e,{g:j(u)})}),b:L});case 4:return{a:r,b:s(E,qr,N)};case 3:return s(Gr,u=n.a,r);default:return Yr()}}),e=En,Ir=h([s(e,"display","grid"),s(e,"place-items","center"),s(e,"margin","2rem")]),u=c("div"),Kr={$:5},Hr={$:4},Wr=function(n){return{$:1,a:n}},Qr=c("article"),Ur=c("button"),x=h([s(e,"font-size","2rem")]),Vr=function(n){if(n.$)return"-";n=n.a;if(n<11)return er(n);switch(n){case 11:return"Jack";case 12:return"Queen";case 13:return"King";case 14:return"Ace";default:return"impossible value"}},Xr=h([s(e,"width","100%"),s(e,"max-width","70rem")]),Zr=c("input"),nt=an,n=r(function(n,r){return s(Nn,n,nt(r))}),rt=n("max"),tt=n("min"),et=_n,ut=r(function(n,r){return s(et,n,{$:0,a:r})}),at=function(n){return s(ut,"click",Ar(n))},it=r(function(n,r){return s(et,n,{$:1,a:r})}),ot=X,W=V,ft=s(r(function(n,r){return v(Er,ot,r,n)}),h(["target","value"]),W),ct=function(n){return s(it,"input",s(jr,Br,s(jr,n,ft)))},z=c("p"),R=s(e,"font-size","2rem"),Y=kn,vt=function(n){return n.$?s(u,d,d):(n=n.a,s(z,h([R]),h([Y(n)])))},st=t(function(n,r,t){return l(n,t)<0&&0<l(r,t)?s(u,h([R]),h([Y("You won :)")])):s(u,h([R]),h([Y("You loose :(")]))}),bt=F(function(n,r,t,e){return 1===r.$||1===t.$||1===e.$?A:j(v(n,r.a,t.a,e.a))}),lt=r(function(n,r){return r.$?n:r.a}),dt=function(n){return 1===n.$?s(u,h([R]),h([Y("This is your first game")])):s(u,d,h([function(n){return s(lt,Y("something is wrong"),b(bt,st,n.d,n.g,n.w))}(n=n.a),s(z,x,h([Y("Card 1: "+Vr(n.d))])),s(z,x,h([Y("Card 2: "+Vr(n.g))])),s(z,x,h([Y("Drawn Card: "+Vr(n.w))]))]))},ht=n("type"),gt=n("value"),Kn=c("h1"),Q=h([s(e,"font-size","2rem"),s(e,"text-align","center")]),$t=s(u,Q,h([s(Kn,h([s(e,"font-size","4rem")]),h([Y("ACEY DUCEY CARD GAME")])),s(u,d,h([Y("Creative Computing Morristown, New Jersey")])),s(u,d,h([Y("\n        Acey-Ducey is played in the following manner. The Dealer (Computer) deals two cards face up. \n        You have an option to bet or not bet depending on whether or not you feel the card will have a value between the first two.\n        If you do not want to bet, bet 0.\n        ")]))])),K=I({aB:Yr,aH:function(n){return Pr},aJ:Z,aK:function(n){return s(u,Ir,h([$t,function(n){return s(Qr,Xr,h(0<n.i?[s(z,x,h([Y("Currently you have "+er(n.i)+" in your pocket.")])),s(z,x,h([Y("Card 1: "+Vr(n.a.d))])),s(z,x,h([Y("Card 2: "+Vr(n.a.g))])),s(z,x,h([Y("Your current bet is "+er(n.j))])),s(Zr,h([ht("range"),rt(er(n.i)),tt("0"),gt(er(n.j)),ct(Wr)]),d),s(Ur,h([at(Hr),R]),h([Y("Play")])),dt(n.D),vt(n.C)]:[s(z,x,h([Y("You lose all you money")])),s(Ur,h([at(Kr),R]),h([Y("Again")]))]))}(n)]))}});En={Main:{init:K(Ar(0))(0)}},q.Elm?function n(r,t){for(var e in t)e in r?"init"==e?H(6):n(r[e],t[e]):r[e]=t[e]}(q.Elm,En):q.Elm=En}(this);
```

# `00_Alternate_Languages/01_Acey_Ducey/go/main.go`



This program appears to be written in Go and is used to print out a welcome message to the user. It is structured as a package with several functions, including `fmt.Println`, which is used to print out the welcome message.

The program also imports several other packages, including `bufio`, `math/rand`, `os`, `sort`, `strconv`, and `strings`. It does not appear to use any of these imports, so it is not clear what it is supposed to do if it is being used in isolation.


```
package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"
)

var welcome = `
Acey-Ducey is played in the following manner
```

这段代码的作用是让玩家在与计算机玩牌游戏中做出决定。它将洗牌并发两张牌给玩家。玩家可以选择跟牌或放弃，或者选择不跟牌，然后计算机会告诉玩家它的牌底。如果玩家的牌面牌值大于等于两张牌的牌面牌值，玩家必须跟牌。

具体来说，代码首先导入了 `bufio.NewScanner` 和 `fmt` 函数。然后使用 `rand.Seed` 函数设置游戏随机种子，使用 `os.Stdin` 获取玩家输入。游戏过程循环进行，每次循环会发一张牌给玩家，然后提示玩家是否要跟牌或放弃。如果玩家选择不跟牌，会输出 "O.K., HOPE YOU HAD FUN!"。


```
The dealer (computer) deals two cards face up
You have an option to bet or not bet depending
on whether or not you feel the card will have
a value between the first two.
If you do not want to bet, input a 0
  `

func main() {
	rand.Seed(time.Now().UnixNano())
	scanner := bufio.NewScanner(os.Stdin)

	fmt.Println(welcome)

	for {
		play(100)
		fmt.Println("TRY AGAIN (YES OR NO)")
		scanner.Scan()
		response := scanner.Text()
		if strings.ToUpper(response) != "YES" {
			break
		}
	}

	fmt.Println("O.K., HOPE YOU HAD FUN!")
}

```

This appears to be a program for a game of blackjack where the player and dealer are dealt two cards each and the player is given the option to bet money. The program first sets the initial money to a default value and then deals the initial two cards to the dealer. The player is then prompted to enter a bet, and if the bet is valid (i.e. the bet is greater than 0 and less than the total amount of money), the program checks if the player's card is a valid one and if the player wins or loses the game. If the player loses, the program updates the money accordingly. If the player wins, the program updates the money and adds the bet to it. The program also includes a loop for the dealer to keep track of the next two cards.


```
func play(money int) {
	scanner := bufio.NewScanner(os.Stdin)
	var bet int

	for {
		// Shuffle the cards
		cards := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
		rand.Shuffle(len(cards), func(i, j int) { cards[i], cards[j] = cards[j], cards[i] })

		// Take the first two for the dealer and sort
		dealerCards := cards[0:2]
		sort.Ints(dealerCards)

		fmt.Printf("YOU NOW HAVE %d DOLLARS.\n\n", money)
		fmt.Printf("HERE ARE YOUR NEXT TWO CARDS:\n%s\n%s", getCardName(dealerCards[0]), getCardName(dealerCards[1]))
		fmt.Printf("\n\n")

		//Check if Bet is Valid
		for {
			fmt.Println("WHAT IS YOUR BET:")
			scanner.Scan()
			b, err := strconv.Atoi(scanner.Text())
			if err != nil {
				fmt.Println("PLEASE ENTER A POSITIVE NUMBER")
				continue
			}
			bet = b

			if bet == 0 {
				fmt.Printf("CHICKEN!\n\n")
				goto there
			}

			if (bet > 0) && (bet <= money) {
				break
			}
		}

		// Draw Players Card
		fmt.Printf("YOUR CARD: %s\n", getCardName(cards[2]))
		if (cards[2] > dealerCards[0]) && (cards[2] < dealerCards[1]) {
			fmt.Println("YOU WIN!!!")
			money = money + bet
		} else {
			fmt.Println("SORRY, YOU LOSE")
			money = money - bet
		}
		fmt.Println()

		if money <= 0 {
			fmt.Printf("%s\n", "SORRY, FRIEND, BUT YOU BLEW YOUR WAD.")
			return
		}
	there:
	}
}

```

这段代码定义了一个名为 `getCardName` 的函数，它接受一个整数参数 `c`，并返回一个相应的字符串。

函数的实现采用了 `switch` 语句，它根据传入的整数 `c` 在控制台上进行字符串操作。

当 `c` 的值为11时，函数返回字符串 "JACK"；当 `c` 的值为12时，函数返回字符串 "QUEEN"；当 `c` 的值为13时，函数返回字符串 "KING"；当 `c` 的值为14时，函数返回字符串 "ACE"。

如果 `c` 的值既不是以上四种情况中的任意一种，函数就会执行 `default` 语句，它将调用 `strconv.Itoa` 函数并将整数 `c` 作为参数传入，返回相应的 ASCII 字符。


```
func getCardName(c int) string {
	switch c {
	case 11:
		return "JACK"
	case 12:
		return "QUEEN"
	case 13:
		return "KING"
	case 14:
		return "ACE"
	default:
		return strconv.Itoa(c)
	}
}

```

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript aceyducey.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "aceyducey"
	run
```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Pascal](https://en.wikipedia.org/wiki/Pascal_(programming_language)) by Gustavo Carreno [gcarreno@github](https://github.com/gcarreno)


Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

# `00_Alternate_Languages/02_Amazing/go/main.go`

这段代码的作用是模拟在一个基于随机深度优先搜索的迷宫中进行探索的过程。它包括了以下主要步骤：

1. 导入了需要使用的一些外部库：bufio、fmt、log、math/rand、os、strconv和time。这些库可能用于在代码中进行输入输出、格式化字符串、日志记录、生成随机数等操作。

2. 实现了打印欢迎消息的功能，用于在开始搜索之前欢迎用户。

3. 实现了一个名为getDimensions的函数，用于获取迷宫的尺寸（宽度和高度）。

4. 实现了名为NewMaze的函数，该函数接收迷宫的尺寸作为参数，创建了一个基于随机深度优先搜索的迷宫对象m。函数中还调用了m的draw函数，对迷宫进行绘制，以帮助用户理解迷宫的结构。

5. 在main函数中，首先调用getDimensions函数获取迷宫的尺寸，然后创建一个maze对象，并调用其draw函数对迷宫进行绘制。然后，使用time.Now().UnixNano()获取当前时间，并作为随机深度优先搜索的起始时间。最后，开始进行搜索，尝试探索迷宫。


```
package main

import (
	"bufio"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	printWelcome()

	h, w := getDimensions()
	m := NewMaze(h, w)
	m.draw()
}

```

这段代码定义了一个名为 "maze" 的结构体类型，其中包含了一个 "width"、"length" 和 "used" 字段，以及一个 "walls" 数组字段，用于存储地图上的墙。

地图的墙被分为四个方向，分别是：LEFT(0)、UP(1)、RIGHT(2)、DOWN(3)。

地图的 "enterCol" 字段定义了从左上角开始，顺时针方向移动的起始位置的列数。

创建一个名为 "maze" 的 "maze" 结构体变量，该结构体包含以下字段：

- "width": 地图的宽度，类型为 "int64"。
- "length": 地图的长度，类型为 "int64"。
- "used": 地图已经被访问过的位置的列表，类型为 "int64 数组"。
- "walls": 地图上的墙的数组，类型为 "int64 数组"。
- "enterCol": 从左上角开始，顺时针方向移动的起始位置的列数，类型为 "int64"。


```
type direction int64

const (
	LEFT direction = iota
	UP
	RIGHT
	DOWN
)

const (
	EXIT_DOWN  = 1
	EXIT_RIGHT = 2
)

type maze struct {
	width    int
	length   int
	used     [][]int
	walls    [][]int
	enterCol int
}

```

这段代码定义了一个名为 `NewMaze` 的函数，它接受两个整数参数 `w` 和 `l`，返回一个表示迷宫的 `maze` 对象。

函数首先检查传入的 `w` 和 `l` 是否合法，如果是，则输出一条错误消息并返回。

接下来，函数创建了一个 `maze` 对象，并创建了一个 2 行宽、2 列长的迷宫。这个迷宫中的每个位置都被初始化为 0，也就是没有被访问过。

接着，函数为迷宫中的每个位置添加了一个墙壁，并将墙壁的入口位于随机选择了一个空位置。

最后，函数使用 `build` 函数来设置迷宫的布局，然后返回创建的 `maze` 对象。

如果 `l` 小于 2，函数不会创建一个有效的迷宫，否则，函数会将编译器崩溃。


```
func NewMaze(w, l int) maze {
	if (w < 2) || (l < 2) {
		log.Fatal("invalid dimensions supplied")
	}

	m := maze{width: w, length: l}

	m.used = make([][]int, l)
	for i := range m.used {
		m.used[i] = make([]int, w)
	}

	m.walls = make([][]int, l)
	for i := range m.walls {
		m.walls[i] = make([]int, w)
	}

	// randomly determine the entry column
	m.enterCol = rand.Intn(w)

	// determine layout of walls
	m.build()

	// add an exit
	col := rand.Intn(m.width - 1)
	row := m.length - 1
	m.walls[row][col] = m.walls[row][col] + 1

	return m
}

```

这段代码是一个名为`func`的函数，它接收一个名为`maze`的二维结构体，并执行以下操作来构建该结构体的网络状图。

函数内部，首先定义了当前游戏的行数`row`和列数`col`，以及当前计数器`count`，用于统计当前细胞已经连接的邻居数量。

接着使用两个嵌套的循环，分别遍历所有的方向，计算出当前可行方向的数量。如果当前计数器`count`等于(当前列数`col`的*当前行数`row`)+1，则认为当前细胞可以连接到当前方向上的所有邻居，循环继续执行。否则，遍历当前方向的邻居，如果当前细胞和邻居细胞已经被连接，则跳过当前邻居，否则更新当前细胞和计数器。

最后，如果当前计数器`count`等于`m.width * m.length`加上当前行列数`row`和`col`的数量，则认为当前游戏已经没有剩余的细胞可以连接，循环结束。


```
func (m *maze) build() {
	row := 0
	col := 0
	count := 2

	for {
		possibleDirs := m.getPossibleDirections(row, col)

		if len(possibleDirs) != 0 {
			row, col, count = m.makeOpening(possibleDirs, row, col, count)
		} else {
			for {
				if col != m.width-1 {
					col = col + 1
				} else if row != m.length-1 {
					row = row + 1
					col = 0
				} else {
					row = 0
					col = 0
				}

				if m.used[row][col] != 0 {
					break
				}
			}
		}

		if count == (m.width*m.length)+1 {
			break
		}
	}

}

```

这段代码定义了一个名为`getPossibleDirections`的函数，接收一个二维字符数组`maze`作为参数，返回四个可能的下一行/列的方向。函数在函数体中创建了一个名为`possible_dirs`的哈希表，类型为`bool`，大小为4。哈希表中存储了每个方向（如左右上下）是否为真，初始值均为真。

首先，函数检查当前行的列是否为0，如果是，则将`LEFT`方向的值设为`false`，否则不设置该方向为真。接着，函数检查当前行是否为0，如果是，则将`UP`方向的值设为`false`，否则不设置该方向为真。然后，函数检查当前行的列是否为`m.width-1`，如果是，则将`RIGHT`方向的值设为`false`，否则不设置该方向为真。最后，函数检查当前行是否为`m.length-1`，如果是，则将`DOWN`方向的值设为`false`，否则不设置该方向为真。

然后，函数通过循环遍历哈希表中的每个方向，并将真值（`true`）存储在返回数组中。函数返回的数组大小为0，因为没有任何方向为真。


```
func (m *maze) getPossibleDirections(row, col int) []direction {
	possible_dirs := make(map[direction]bool, 4)
	possible_dirs[LEFT] = true
	possible_dirs[UP] = true
	possible_dirs[RIGHT] = true
	possible_dirs[DOWN] = true

	if (col == 0) || (m.used[row][col-1] != 0) {
		possible_dirs[LEFT] = false
	}
	if (row == 0) || (m.used[row-1][col] != 0) {
		possible_dirs[UP] = false
	}
	if (col == m.width-1) || (m.used[row][col+1] != 0) {
		possible_dirs[RIGHT] = false
	}
	if (row == m.length-1) || (m.used[row+1][col] != 0) {
		possible_dirs[DOWN] = false
	}

	ret := make([]direction, 0)
	for d, v := range possible_dirs {
		if v {
			ret = append(ret, d)
		}
	}
	return ret
}

```

该函数的作用是用于在给定的迷宫中打开一个出口，并返回新打开的出口的行、列和计数器。该函数的参数是一个指向二维迷宫的指针、一个方向数组和四个整数参数分别表示迷宫的行、列和计数器。

函数的逻辑如下：

1. 使用随机数生成一个方向数组 `dirs`，其中 `dirs` 的长度为 `len(dirs)`。
2. 使用 `rand.Intn(len(dirs))` 获取一个随机整数 `dir`，将其存储在 `dirs` 中的对应位置。
3. 根据 `dir` 的值，对迷宫中的墙壁进行修改。如果 `dir` 是 `LEFT` 方向，则将 `col` 向左移动；如果 `dir` 是 `UP` 方向，则将 `row` 向上移动；如果 `dir` 是 `RIGHT` 方向，则将 `col` 向上移动并增加计数器 `count`；如果 `dir` 是 `DOWN` 方向，则将 `row` 向上移动并增加计数器 `count`。
4. 返回新打开的出口的行、列和计数器。

函数的实现使得在给定的迷宫中，有一个出口的位置是随机选择的，并且每次打开出口时都会增加计数器，以便在需要时可以统计出打开了多少个出口。


```
func (m *maze) makeOpening(dirs []direction, row, col, count int) (int, int, int) {
	dir := rand.Intn(len(dirs))

	if dirs[dir] == LEFT {
		col = col - 1
		m.walls[row][col] = int(EXIT_RIGHT)
	} else if dirs[dir] == UP {
		row = row - 1
		m.walls[row][col] = int(EXIT_DOWN)
	} else if dirs[dir] == RIGHT {
		m.walls[row][col] = m.walls[row][col] + EXIT_RIGHT
		col = col + 1
	} else if dirs[dir] == DOWN {
		m.walls[row][col] = m.walls[row][col] + EXIT_DOWN
		row = row + 1
	}

	m.used[row][col] = count
	count = count + 1
	return row, col, count
}

```

这段代码定义了一个名为 `draw` 的函数，它是 `maze` 结构体的成员函数。函数的主要作用是绘制一个迷宫的路径。

具体来说，函数首先遍历迷宫的每一列，对于每一列，先打印一个空格，然后打印一个波浪线。接着，函数遍历迷宫的每一行，对于每一行，先打印一个星号，然后打印一个空格。这样，就形成了一个迷宫的二维图像。

函数还打印了一些边界信息，用来表示迷宫的边界和入口。具体来说，当某个位置是迷宫的入口时，函数会输出一个星号，当某个位置是迷宫的边界时，函数会输出一个波浪线和两个空格。

总的来说，这段代码的主要作用是提供一个函数来绘制迷宫的路径，以便人们更好地理解迷宫的结构和特征。


```
// draw the maze
func (m *maze) draw() {
	for col := 0; col < m.width; col++ {
		if col == m.enterCol {
			fmt.Print(".  ")
		} else {
			fmt.Print(".--")
		}
	}
	fmt.Println(".")

	for row := 0; row < m.length; row++ {
		fmt.Print("|")
		for col := 0; col < m.width; col++ {
			if m.walls[row][col] < 2 {
				fmt.Print("  |")
			} else {
				fmt.Print("   ")
			}
		}
		fmt.Println()
		for col := 0; col < m.width; col++ {
			if (m.walls[row][col] == 0) || (m.walls[row][col] == 2) {
				fmt.Print(":--")
			} else {
				fmt.Print(":  ")
			}
		}
		fmt.Println(".")
	}
}

```

这两段代码定义了两个函数 `printWelcome` 和 `getDimensions`。

1. `printWelcome` 函数的作用是在屏幕上打印出 "AMAZING PROGRAM" 消息，并在其中添加了一些附加信息。然后将其打印出来。

2. `getDimensions` 函数的作用是读取用户输入的宽度（必须大于1）和高度（也必须大于1），并返回它们。它使用 `bufio.NewScanner` 函数从标准输入（通常是键盘）中读取输入，然后使用 `strconv.Atoi` 函数将输入转换为整数类型。如果转换发生错误，函数将返回一个错误消息。

这两个函数都在一个名为 `main` 的函数中被调用， `printWelcome` 函数用于在程序启动时打印欢迎消息，`getDimensions` 函数用于在程序中读取用户输入以获取尺寸信息。


```
func printWelcome() {
	fmt.Println("                            AMAZING PROGRAM")
	fmt.Print("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")
}

func getDimensions() (int, int) {
	scanner := bufio.NewScanner(os.Stdin)

	fmt.Println("Enter a width ( > 1 ):")
	scanner.Scan()
	w, err := strconv.Atoi(scanner.Text())
	if err != nil {
		log.Fatal("invalid dimension")
	}

	fmt.Println("Enter a height ( > 1 ):")
	scanner.Scan()
	h, err := strconv.Atoi(scanner.Text())
	if err != nil {
		log.Fatal("invalid dimension")
	}

	return w, h
}

```

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript amazing.ms
```
Note that because this program imports "listUtil", you will need to have a the standard MiniScript libraries somewhere in your import path.

2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "amazing"
	run
```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Pascal](https://en.wikipedia.org/wiki/Pascal_(programming_language)) by Gustavo Carreno [gcarreno@github](https://github.com/gcarreno)


Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

# `00_Alternate_Languages/03_Animal/go/main.go`

这段代码定义了一个名为 `node` 的结构体，它包含两个指针变量 `yesNode` 和 `noNode`，分别指向两个子节点，以及一个字符串 `text`。

接着，它定义了一个名为 `fmt.Println` 的函数，该函数用于将字符串内容输出到屏幕上，但在这里，它并不是直接输出 `node` 结构体，而是输出一个字符串 `"Hello, %v!= !\"\n"`。

在 `main` 函数中，它首先导入了 `bufio`、`fmt`、`log` 和 `os` 包。然后，它创建了一个名为 `log` 的变量，并将其设置为 `log.Fatal` 函数的输出，以便在程序出现错误时输出错误信息。

接下来，它使用 `os.Exit` 函数来获取用户输入的参数，并将其存储在名为 `param` 的变量中。然后，它遍历 `param` 变量中的所有参数，并将它们存储在 `text` 变量中。

最后，它创建了两个子节点，一个节点设置为 `text` 中的字符串 `"hello"`，另一个节点设置为 `text` 中的字符串 `"!=="`，并将一个指向字符串 `"!"` 的指针分配给这两个子节点。然后，它打印出字符串 `"Hello, %v!= !\"!"`。


```
package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"
)

type node struct {
	text    string
	yesNode *node
	noNode  *node
}

```

这段代码定义了两个函数，分别是 `newNode` 和 `(`*`node`)。函数的功能是通过传递参数来创建一个新的 `node` 对象。

`newNode` 函数接收三个参数：`text`、`yes_node` 和 `no_node`。函数创建了一个新的 `node` 对象，并将其类型设置为 `text`。然后，对 `yes_node` 和 `no_node` 进行了判断，如果它们都不为 `nil`，则将其设置为 `yes_node` 和 `no_node` 对象。最后，返回创建的 `node` 对象。

`(`*`node`)` 函数接收四个参数：`n`、`newQuestion`、`newAnswer` 和 `newAnimal`。函数首先访问 `n` 的 `text` 字段，然后将其更新为 `newQuestion`。接下来，根据 `newAnswer` 的值，创建新的 `node` 对象并将其设置为 `yes_node` 或 `no_node` 对象。最后，将 `n` 返回，以便调用者能够访问 `node` 对象。


```
func newNode(text string, yes_node, no_node *node) *node {
	n := node{text: text}
	if yes_node != nil {
		n.yesNode = yes_node
	}
	if no_node != nil {
		n.noNode = no_node
	}
	return &n
}

func (n *node) update(newQuestion, newAnswer, newAnimal string) {
	oldAnimal := n.text

	n.text = newQuestion

	if newAnswer == "y" {
		n.yesNode = newNode(newAnimal, nil, nil)
		n.noNode = newNode(oldAnimal, nil, nil)
	} else {
		n.yesNode = newNode(oldAnimal, nil, nil)
		n.noNode = newNode(newAnimal, nil, nil)
	}
}

```

这段代码定义了两个函数：

1. `isLeaf()` 函数接收一个整数类型的参数 `n`，并返回两个条件都为 `true` 时为 `true`，否则为 `false`。这个函数的作用是判断一个节点是否为叶子节点。

2. `listKnownAnimals()` 函数接收一个整数类型的参数 `root`，并递归地遍历以 `root` 为根的所有的子节点。当 `root` 为 `nil` 时，直接返回；当 `root` 的 `isLeaf()` 返回 `true` 时，递归调用 `listKnownAnimals()` 函数，返回；当 `root` 的 `isLeaf()` 返回 `false` 时，打印字符串并返回。

综合来看，这两个函数的作用是判断给定整数类型的根节点是否为叶子节点，如果是，则打印该节点的文本内容并返回；如果不是，则递归调用 `listKnownAnimals()` 函数。


```
func (n *node) isLeaf() bool {
	return (n.yesNode == nil) && (n.noNode == nil)
}

func listKnownAnimals(root *node) {
	if root == nil {
		return
	}

	if root.isLeaf() {
		fmt.Printf("%s           ", root.text)
		return
	}

	if root.yesNode != nil {
		listKnownAnimals(root.yesNode)
	}

	if root.noNode != nil {
		listKnownAnimals(root.noNode)
	}
}

```

该函数 `parseInput` 接受三个参数：

1. `message`：字符串，表示用户输入的信息。
2. `checkList`：布尔值，表示是否检查输入中是否包含 "list"。
3. `rootNode`：指针，表示根节点。

函数内部首先从标准输入（通常是键盘）读取字符串 `message`，并将其转换为小写。然后，如果 `checkList` 为真，则判断输入是否为 "list"。如果是，则调用 `listKnownAnimals` 函数，将已知列表中的动物名称打印出来。如果不是，则更新 `token` 变量为输入字符串的小写。接下来，如果 `token` 不是空字符串，则退出循环。否则，如果循环条件为 `token == "y" || token == "n"`，则退出循环。最后，返回 `token` 变量，表示输入的字符串。


```
func parseInput(message string, checkList bool, rootNode *node) string {
	scanner := bufio.NewScanner(os.Stdin)
	token := ""

	for {
		fmt.Println(message)
		scanner.Scan()
		inp := strings.ToLower(scanner.Text())

		if checkList && inp == "list" {
			fmt.Println("Animals I already know are:")
			listKnownAnimals(rootNode)
			fmt.Println()
		}

		if len(inp) > 0 {
			token = inp
		} else {
			token = ""
		}

		if token == "y" || token == "n" {
			break
		}
	}
	return token
}

```

这段代码定义了一个名为 avoidVoidInput 的函数，接受一个字符串参数 message。这个函数的作用是在不输出空字符串的情况下，从标准输入（通常是键盘）读取一行字符串，并返回该行字符串。

函数内部首先创建了一个字符缓冲区 scanner，然后使用 bufio.NewScanner() 方法将缓冲区设置为从标准输入读取。然后，函数循环若干次，每次打印出 message 并使用 scanner.Scan() 方法读取一行字符串，并将该行字符串存储在 answer 变量中。

如果 answer 不为空，则退出循环。最后，函数返回 answer 变量，如果 answer 为空，则输出 "No input"。


```
func avoidVoidInput(message string) string {
	scanner := bufio.NewScanner(os.Stdin)
	answer := ""
	for {
		fmt.Println(message)
		scanner.Scan()
		answer = scanner.Text()

		if answer != "" {
			break
		}
	}
	return answer
}

```

This is a simple JavaScript function that attempts to understand the user's intent based on their input and returns a logical result. Here's a high-level overview of how it works:

1. The function takes a single argument `rootNode`, which represents the root node of an XML document.
2. It checks whether the input is a valid question by comparing it to an empty string.
3. If the input is a valid question, it attempts to extract the user's answer by calling the `parseInput` function with the appropriate arguments.
4. If the answer is "yes", it checks if the corresponding "yesNode" is not present in the tree. If it's not found, it logs an error.
5. If the answer is "no", it checks if the corresponding "noNode" is not present in the tree. If it's not found, it logs an error.
6. If the input is not a valid question or the answer is neither "yes" nor "no", it logs an error.
7. It then checks whether the input is "are you thinking of an animal?" or not. If it's not a valid question, it sets `keepPlaying` to `false` and doesn't continue processing.
8. If the input is a valid question, it calls the `keepAsking` function until the user either answers with "yes" or "no".
9. Finally, it checks whether `keepPlaying` is `true` or not, and if it is, it represents the user's answer as a valid animal. If it's not, it doesn't perform any action and just returns the original input.

The function also includes a helper function `avoidVoidInput`, which takes a string as input and returns a new string with all whitespaces，特殊 characters, and punctuations removed.


```
func printIntro() {
	fmt.Println("                                Animal")
	fmt.Println("               Creative Computing Morristown, New Jersey")
	fmt.Println("\nPlay 'Guess the Animal'")
	fmt.Println("Think of an animal and the computer will try to guess it")
}

func main() {
	yesChild := newNode("Fish", nil, nil)
	noChild := newNode("Bird", nil, nil)
	rootNode := newNode("Does it swim?", yesChild, noChild)

	printIntro()

	keepPlaying := (parseInput("Are you thinking of an animal?", true, rootNode) == "y")

	for keepPlaying {
		keepAsking := true

		actualNode := rootNode

		for keepAsking {
			if !actualNode.isLeaf() {
				answer := parseInput(actualNode.text, false, nil)

				if answer == "y" {
					if actualNode.yesNode == nil {
						log.Fatal("invalid node")
					}
					actualNode = actualNode.yesNode
				} else {
					if actualNode.noNode == nil {
						log.Fatal("invalid node")
					}
					actualNode = actualNode.noNode
				}
			} else {
				answer := parseInput(fmt.Sprintf("Is it a %s?", actualNode.text), false, nil)
				if answer == "n" {
					newAnimal := avoidVoidInput("The animal you were thinking of was a ?")
					newQuestion := avoidVoidInput(fmt.Sprintf("Please type in a question that would distinguish a '%s' from a '%s':", newAnimal, actualNode.text))
					newAnswer := parseInput(fmt.Sprintf("For a '%s' the answer would be", newAnimal), false, nil)
					actualNode.update(newQuestion+"?", newAnswer, newAnimal)
				} else {
					fmt.Println("Why not try another animal?")
				}
				keepAsking = false
			}
		}
		keepPlaying = (parseInput("Are you thinking of an animal?", true, rootNode) == "y")
	}
}

```

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript animal.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "animal"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

# Awari

This is an Elm implementation of the `Basic Compouter Games` Game Awari.

## Build App

- install elm

```bash
yarn
yarn build
```


Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript awari.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "awari"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

# `00_Alternate_Languages/05_Bagels/go/main.go`

这段代码是一个用于计算数学题目的程序。它使用了以下库：

- `bufio` 库用于输入输出字符串
- `fmt` 库用于格式化输出
- `math/rand` 库用于生成随机数
- `os` 库用于操作系统交互
- `strconv` 库用于字符串转换和输入输出
- `strings` 库用于字符串操作
- `time` 库用于时间计算

程序的主要作用是读取用户输入的数学题目，然后根据用户输入的答案提示正确答案。当用户猜对答案时，程序会显示恭喜信息，并等待用户再次输入题目。当用户猜错答案时，程序会提示用户最多可以猜多少次。如果用户在规定时间内没有猜中答案，程序会结束并提示用户。


```
package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

const MAXGUESSES int = 20

func printWelcome() {
	fmt.Println("\n                Bagels")
	fmt.Println("Creative Computing  Morristown, New Jersey")
	fmt.Println()
}
```

这两段代码定义了两个函数 `printRules()` 和 `getNumber()`。

1. `printRules()` 函数的主要作用是输出一些提示规则，告诉用户如何猜测三个位数的数字。具体来说，这个函数会输出以下内容：

```
I am thinking of a three-digit number.  Try to guess
my number and I will give you clues as follows:
  PICO   - One digit correct but in the wrong position
  FERMI  - One digit correct and in the right position
  BAGELS - No digits correct
```

这些内容会告诉用户，他的数字可能是哪个，以及如何猜测。

2. `getNumber()` 函数的作用是从预定义的一些数字中随机抽取三个数字，并返回给调用者。具体来说，这个函数会返回以下数字：

```
0
1
2
3
```

这些数字是通过调用 `shuffle()` 函数实现的，这个函数会将长度为 `len(numbers)` 的数字列表随机重排。


```
func printRules() {
	fmt.Println()
	fmt.Println("I am thinking of a three-digit number.  Try to guess")
	fmt.Println("my number and I will give you clues as follows:")
	fmt.Println("   PICO   - One digit correct but in the wrong position")
	fmt.Println("   FERMI  - One digit correct and in the right position")
	fmt.Println("   BAGELS - No digits correct")
}

func getNumber() []string {
	numbers := []string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
	rand.Shuffle(len(numbers), func(i, j int) { numbers[i], numbers[j] = numbers[j], numbers[i] })

	return numbers[:3]
}

```

这段代码定义了一个名为 `getValidGuess` 的函数，接受一个整数参数 `guessNumber`。函数返回一个字符串，表示猜的数字。

函数内部使用了一个 `bufio.NewScanner` 来从标准输入（通常是键盘）读取输入。读取到一个有效猜数后，函数会继续从输入中读取另一个有效猜数。

在读取有效猜数时，函数会首先输出 "Guess # <guessNumber>" 并等待用户输入。然后，函数使用 `scanner.Scan` 方法从输入中读取用户输入的字符串。接下来，函数会计算输入字符串中的长度，并尝试将其截去空间后得到一个三位数。如果这个三位数是有效的（满足必须是一个有效的数字且不能包含已知的数字），函数会继续判断它的数字是否相同。否则，函数会提示用户再试一次，或者直接告诉用户输入不是一个有效的数字。

最后，函数会根据 `valid` 变量返回猜的数字，或提示用户再试一次。


```
func getValidGuess(guessNumber int) string {
	var guess string
	scanner := bufio.NewScanner(os.Stdin)
	valid := false
	for !valid {
		fmt.Printf("Guess # %d?\n", guessNumber)
		scanner.Scan()
		guess = strings.TrimSpace(scanner.Text())

		// guess must be 3 characters
		if len(guess) == 3 {
			// and should be numeric
			_, err := strconv.Atoi(guess)
			if err != nil {
				fmt.Println("What?")
			} else {
				// and the numbers should be unique
				if (guess[0:1] != guess[1:2]) && (guess[0:1] != guess[2:3]) && (guess[1:2] != guess[2:3]) {
					valid = true
				} else {
					fmt.Println("Oh, I forgot to tell you that the number I have in mind")
					fmt.Println("has no two digits the same.")
				}
			}
		} else {
			fmt.Println("Try guessing a three-digit number.")
		}
	}

	return guess
}

```

这段代码定义了一个名为 `buildResultString` 的函数，它接受两个参数 `num` 和 `guess`，并将它们作为字符串返回。

函数内部首先创建一个空字符串 `result`，然后使用两个嵌套的循环来遍历 `num` 和 `guess`。在循环内部，如果 `num` 中的第 `i` 个数字与 `guess` 中的第 `i+1` 个数字相同，则将 `"PICO"` 字符添加到 `result` 中。接下来，如果 `num` 中的第 `i+1` 个数字与 `guess` 中的第 `i` 个数字相同，则将 `"PICO"` 字符添加到 `result` 中。然后，如果 `num` 中的第一个数字与 `guess` 中的第二个数字相同，则将 `"PICO"` 字符添加到 `result` 中。最后，如果 `num` 和 `guess` 中都没有正确的数字，则将 `"BAGELS"` 字符添加到 `result` 中。

函数会返回一个非空字符串，如果 `num` 和 `guess` 中都没有正确的数字，则返回一个空字符串。


```
func buildResultString(num []string, guess string) string {
	result := ""

	// correct digits in wrong place
	for i := 0; i < 2; i++ {
		if num[i] == guess[i+1:i+2] {
			result += "PICO "
		}
		if num[i+1] == guess[i:i+1] {
			result += "PICO "
		}
	}
	if num[0] == guess[2:3] {
		result += "PICO "
	}
	if num[2] == guess[0:1] {
		result += "PICO "
	}

	// correct digits in right place
	for i := 0; i < 3; i++ {
		if num[i] == guess[i:i+1] {
			result += "FERMI "
		}
	}

	// nothing right?
	if result == "" {
		result = "BAGELS"
	}

	return result
}

```

This appears to be a game of number guessing where the player is given a number and a number of guesses and the game will tell if the猜测是否 correct or not and if the player wins or not. It also seems to have a rule that if the number is not a number the game will start again with the same number of guesses. The game also has a rule that if the player does not make a guess within a certain number of guesses, the game will start again with the same number of guesses.


```
func main() {
	rand.Seed(time.Now().UnixNano())
	scanner := bufio.NewScanner(os.Stdin)

	printWelcome()

	fmt.Println("Would you like the rules (Yes or No)? ")
	scanner.Scan()
	response := scanner.Text()
	if len(response) > 0 {
		if strings.ToUpper(response[0:1]) != "N" {
			printRules()
		}
	} else {
		printRules()
	}

	gamesWon := 0
	stillRunning := true

	for stillRunning {
		num := getNumber()
		numStr := strings.Join(num, "")
		guesses := 1

		fmt.Println("\nO.K.  I have a number in mind.")
		guessing := true
		for guessing {
			guess := getValidGuess(guesses)

			if guess == numStr {
				fmt.Println("You got it!!")
				gamesWon++
				guessing = false
			} else {
				fmt.Println(buildResultString(num, guess))
				guesses++
				if guesses > MAXGUESSES {
					fmt.Println("Oh well")
					fmt.Printf("That's %d guesses. My number was %s\n", MAXGUESSES, numStr)
					guessing = false
				}
			}
		}

		validRespone := false
		for !validRespone {
			fmt.Println("Play again (Yes or No)?")
			scanner.Scan()
			response := scanner.Text()
			if len(response) > 0 {
				validRespone = true
				if strings.ToUpper(response[0:1]) != "Y" {
					stillRunning = false
				}
			}
		}
	}

	if gamesWon > 0 {
		fmt.Printf("\nA %d point Bagels buff!!\n", gamesWon)
	}

	fmt.Println("Hope you had fun.  Bye")
}

```

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript bagels.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "bagels"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript banner.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "banner"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript basketball.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "basketball"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

# `00_Alternate_Languages/08_Batnum/go/main.go`

这段代码定义了一个名为 `main` 的包，它导入了 `bufio`、`fmt`、`os` 和 `strconv` 等标准库。此外，它还定义了一个名为 `StartOption` 的类型，它用于表示程序如何开始，使用了 `iota` 类型，以便在程序中使用不同的名称。

接下来，代码定义了一个名为 `StartOption` 的枚举类型，它包含了三个不同的枚举值，分别命名为 `ComputerFirst`、`PlayerFirst` 和 `StartUndefined`。这些枚举值用于指定程序如何开始，根据用户的选择，可以是计算机先手、玩家先手或无限制自动开始。

接着，代码中定义了一个名为 `main` 的函数，它接收一个整数参数 `user_choice`，用于指定用户选择哪种起始方式。然后，代码使用 `fmt.Println()` 函数来输出帮助信息，告诉用户如何使用这个程序。

接下来，代码使用 `os.Args` 函数来获取用户在命令行中输入的启动参数。如果用户没有提供任何参数，那么程序将默认从 `computer_first` 开始。如果用户提供了 `-p` 或 `--player` 参数，那么程序将从 `player_first` 开始。

最后，代码根据用户的选择调用 `StartOption` 函数中的相应枚举值，并调用 `fmt.Println()` 函数来输出帮助信息。


```
package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

type StartOption int8

const (
	StartUndefined StartOption = iota
	ComputerFirst
	PlayerFirst
)

```

这段代码定义了一个名为 "WinOption" 的枚举类型，其值为 int8(即 8 位整数)，名为 "WinUndefined"。接着，定义了一个名为 "TakeLast" 的函数，其返回值为 WinOption( WinUndefined)。此外，还定义了一个名为 "AvoidLast" 的函数，其返回值为 WinOption( Aolean(false)).

然后，定义了一个名为 "GameOptions" 的结构体类型，其包含以下字段：

- pileSize: 堆栈大小(即玩家在每一轮游戏中可以放置的物品数量)
- winOption: 当前游戏的胜利选项，可以是 WinOption( WinUndefined) 中的任意一个值
- startOption: 开始游戏的选项，可以是 StartOption(Default) 中的一个值
- minSelect: 最小选择物品的等级
- maxSelect: 最大选择物品的等级

最后，没有定义任何函数或其他操作，直接跳到了下一行代码。


```
type WinOption int8

const (
	WinUndefined WinOption = iota
	TakeLast
	AvoidLast
)

type GameOptions struct {
	pileSize    int
	winOption   WinOption
	startOption StartOption
	minSelect   int
	maxSelect   int
}

```

这段代码定义了一个名为 `NewOptions` 的函数，它返回一个指向 `GameOptions` 类型的指针变量 `g`。

函数内部，首先创建一个名为 `g` 的 `GameOptions` 类型的变量，然后使用 `getPileSize` 函数获取堆叠大小，如果堆叠大小小于 0，函数返回 `&g` 即 `g` 的内存地址。

接着，使用 `getWinOption` 和 `getMinMax` 函数获取获胜选项和最小最大值，并将它们存储到 `g.winOption` 和 `g.minSelect`、`g.maxSelect` 中。

最后，使用 `getStartOption` 函数获取起始选项，并将其存储到 `g.startOption` 中。

函数返回指向 `g` 的指针变量，这样就可以在函数外部使用 `g` 了。


```
func NewOptions() *GameOptions {
	g := GameOptions{}

	g.pileSize = getPileSize()
	if g.pileSize < 0 {
		return &g
	}

	g.winOption = getWinOption()
	g.minSelect, g.maxSelect = getMinMax()
	g.startOption = getStartOption()

	return &g
}

```

这段代码的作用是获取堆栈中堆叠的大小。函数接受一个整数类型的参数ps，并在函数内部使用bufio.NewScanner从标准输入(通常是键盘)读取输入。然后，使用for循环来读取用户输入的堆叠大小，并将输入转换为整数类型。在循环中，函数使用strconv.Atoi函数将输入的字符串转换为整数，并将结果存储在ps变量中。如果转换过程中出现错误，函数将返回，如果没有错误，则函数将返回堆栈中堆叠的大小ps。


```
func getPileSize() int {
	ps := 0
	var err error
	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Println("Enter Pile Size ")
		scanner.Scan()
		ps, err = strconv.Atoi(scanner.Text())
		if err == nil {
			break
		}
	}
	return ps
}

```

这两段代码都是接受用户输入并返回相应的选项，函数分别称为getWinOption()和getStartOption()。

getWinOption()函数的作用是读取用户输入的整数并返回一个WinOption类型的数据结构。该函数使用了两个嵌套的for循环，第一个循环用于从标准输入（通常是键盘输入）中读取用户的输入，第二个循环用于解析用户输入并转换为整数类型。如果用户输入正确（即输入1或2），则返回相应的WinOption类型。

getStartOption()函数的作用与getWinOption()函数类似，但是该函数返回的是一个StartOption类型的数据结构。该函数使用了两个嵌套的for循环，第一个循环用于从标准输入中读取用户的输入，第二个循环用于解析用户输入并转换为整数类型。如果用户输入正确（即输入1或2），则返回相应的StartOption类型。


```
func getWinOption() WinOption {
	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Println("ENTER WIN OPTION - 1 TO TAKE LAST, 2 TO AVOID LAST:")
		scanner.Scan()
		w, err := strconv.Atoi(scanner.Text())
		if err == nil && (w == 1 || w == 2) {
			return WinOption(w)
		}
	}
}

func getStartOption() StartOption {
	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Println("ENTER START OPTION - 1 COMPUTER FIRST, 2 YOU FIRST ")
		scanner.Scan()
		s, err := strconv.Atoi(scanner.Text())
		if err == nil && (s == 1 || s == 2) {
			return StartOption(s)
		}
	}
}

```

该函数的作用是读取用户输入的最低和最高整数值，并返回它们。它通过从标准输入（通常是键盘）中使用bufio库的和新扫描器读取字符串。然后，它将读取的字符串转换为整数，并将最低和最高整数存储在minSelect和maxSelect变量中。如果函数在从标准输入读取字符串时遇到任何错误，则函数将返回minSelect和maxSelect，以便用户可以重新输入它们。


```
func getMinMax() (int, int) {
	minSelect := 0
	maxSelect := 0
	var minErr error
	var maxErr error
	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Println("ENTER MIN AND MAX ")
		scanner.Scan()
		enteredValues := scanner.Text()
		vals := strings.Split(enteredValues, " ")
		minSelect, minErr = strconv.Atoi(vals[0])
		maxSelect, maxErr = strconv.Atoi(vals[1])
		if (minErr == nil) && (maxErr == nil) && (minSelect > 0) && (maxSelect > 0) && (maxSelect > minSelect) {
			return minSelect, maxSelect
		}
	}
}

```

这段代码的作用是处理玩家的回合，首先询问玩家要选择多少个物品，并进行一些基本的输入验证，确保输入的物品数量是有效的。然后，它检查是否存在任何胜利条件，如果是，就返回一个布尔值和新的堆栈大小；如果不是，就返回 false 和原来的堆栈大小。


```
// This handles the player's turn - asking the player how many objects
// to take and doing some basic validation around that input.  Then it
// checks for any win conditions.
// Returns a boolean indicating whether the game is over and the new pile_size.
func playerMove(pile, min, max int, win WinOption) (bool, int) {
	scanner := bufio.NewScanner(os.Stdin)
	done := false
	for !done {
		fmt.Println("YOUR MOVE")
		scanner.Scan()
		m, err := strconv.Atoi(scanner.Text())
		if err != nil {
			continue
		}

		if m == 0 {
			fmt.Println("I TOLD YOU NOT TO USE ZERO!  COMPUTER WINS BY FORFEIT.")
			return true, pile
		}

		if m > max || m < min {
			fmt.Println("ILLEGAL MOVE, REENTER IT")
			continue
		}

		pile -= m
		done = true

		if pile <= 0 {
			if win == AvoidLast {
				fmt.Println("TOUGH LUCK, YOU LOSE.")
			} else {
				fmt.Println("CONGRATULATIONS, YOU WIN.")
			}
			return true, pile
		}
	}
	return false, pile
}

```

这段代码定义了一个名为 `computerPick` 的函数，用于处理计算机在的选择过程中需要计算出有多少个物体。这里 `computerPick` 函数接收三个参数：堆栈 `pile`，最小值 `min`，最大值 `max` 和选择胜利选项 `win`。胜利选项可以是 `AvoidLast` 或 `FirstComeFirstGo`。函数的主要逻辑如下：

1. 如果 `win` 选项是 `AvoidLast`，则将堆栈中的物体数量设置为 `pile - 1`，否则将堆栈中的物体数量设置为 `pile`。

2. 计算出可以选择的物体数量 `c`。

3. 从堆栈中选择 `min + max` 个物体。

4. 如果选择的物体数量 `pick` 小于最小值 `min`，则将 `pick` 设置为 `min`。

5. 如果选择的物体数量 `pick` 大于最大值 `max`，则将 `pick` 设置为 `max`。

6. 返回选择的物体数量 `pick`。


```
// This handles the logic to determine how many objects the computer
// will select on its turn.
func computerPick(pile, min, max int, win WinOption) int {
	var q int
	if win == AvoidLast {
		q = pile - 1
	} else {
		q = pile
	}
	c := min + max

	pick := q - (c * int(q/c))

	if pick < min {
		pick = min
	} else if pick > max {
		pick = max
	}

	return pick
}

```

这段代码的作用是处理电脑在游戏中的移动。它首先检查是否符合游戏的结束条件（比如游戏是否已经赢得了所有的对象或者电脑是否已经无法继续增加游戏中的物品），然后根据用户设置的游戏选项，计算电脑需要选择多少个物品或者不选择任何物品。最后，它返回一个布尔值来表示游戏是否已经结束以及电脑选择了多少物品。


```
// This handles the computer's turn - first checking for the various
// win/lose conditions and then calculating how many objects
// the computer will take.
// Returns a boolean indicating whether the game is over and the new pile_size.
func computerMove(pile, min, max int, win WinOption) (bool, int) {
	// first check for end-game conditions
	if win == TakeLast && pile <= max {
		fmt.Printf("COMPUTER TAKES %d AND WINS\n", pile)
		return true, pile
	}

	if win == AvoidLast && pile <= min {
		fmt.Printf("COMPUTER TAKES %d AND LOSES\n", pile)
		return true, pile
	}

	// otherwise determine the computer's selection
	selection := computerPick(pile, min, max, win)
	pile -= selection
	fmt.Printf("COMPUTER TAKES %d AND LEAVES %d\n", selection, pile)
	return false, pile
}

```

这段代码是一个游戏的main game loop，不断重复进行，直到满足赢得或输出的条件。

在game loop中，首先检查gameOver变量是否为true，如果是，则退出循环，因为游戏已经结束。如果不是，进入循环体，其中会执行以下操作：

1.检查当前轮到哪个玩家进行移动操作，如果是玩家1进行移动，则调用playerMove函数，传递参数pile、min和max来设置 pile 堆中的物品数量以及胜利选项WinOption。执行完毕后，将玩家1的移动状态设置为false，并检查游戏是否结束。

2.如果当前轮不到玩家1移动，则是计算机移动，调用computerMove函数，设置参数pile、min和max，以及胜利选项WinOption。执行完毕后，将玩家1的移动状态设置为true，并检查游戏是否结束。

3.如果gameOver为false且当前轮到玩家1移动，则返回到game loop中继续执行。

4.如果gameOver为true或当前轮不到任何玩家移动，则退出循环并等待下一轮开始。


```
// This is the main game loop - repeating each turn until one
// of the win/lose conditions is met.
func play(pile, min, max int, start StartOption, win WinOption) {
	gameOver := false
	playersTurn := (start == PlayerFirst)

	for !gameOver {
		if playersTurn {
			gameOver, pile = playerMove(pile, min, max, win)
			playersTurn = false
			if gameOver {
				return
			}
		}

		if !playersTurn {
			gameOver, pile = computerMove(pile, min, max, win)
			playersTurn = true
		}
	}
}

```

这段代码是一个名为 "printIntro" 的函数，它的作用是输出游戏的介绍规则。函数内部使用了格式化字符串（fmt.Printf）来输出字符串内容，包括转义字符和格式控制符。

具体来说，这段代码输出的字符串内容如下：

"This program is a Battle of Numbers game, where the computer is your opponent."

"The game starts with an assumed pile of objects. You and your opponent can alternatively remove objects from the pile."

"Winning is defined in advance as taking the last object or not. You can also specify some other beginning conditions."

"Don't use zero, however, in playing the game."

"Enter a negative number for the new pile size to stop playing."

这些输出字符串的信息是游戏规则的描述，告诉玩家应该如何玩这个 Battle of Numbers 游戏。


```
// Print out the introduction and rules of the game
func printIntro() {
	fmt.Printf("%33s%s\n", " ", "BATNUM")
	fmt.Printf("%15s%s\n", " ", "CREATIVE COMPUTING  MORRISSTOWN, NEW JERSEY")
	fmt.Printf("\n\n\n")
	fmt.Println("THIS PROGRAM IS A 'BATTLE OF NUMBERS' GAME, WHERE THE")
	fmt.Println("COMPUTER IS YOUR OPPONENT.")
	fmt.Println()
	fmt.Println("THE GAME STARTS WITH AN ASSUMED PILE OF OBJECTS. YOU")
	fmt.Println("AND YOUR OPPONENT ALTERNATELY REMOVE OBJECTS FROM THE PILE.")
	fmt.Println("WINNING IS DEFINED IN ADVANCE AS TAKING THE LAST OBJECT OR")
	fmt.Println("NOT. YOU CAN ALSO SPECIFY SOME OTHER BEGINNING CONDITIONS.")
	fmt.Println("DON'T USE ZERO, HOWEVER, IN PLAYING THE GAME.")
	fmt.Println("ENTER A NEGATIVE NUMBER FOR NEW PILE SIZE TO STOP PLAYING.")
	fmt.Println()
}

```

这段代码定义了一个名为 main 的函数，它在每次循环时执行以下操作：

1. 打印函数介绍信息。
2. 创建一个名为 g 的选项对象。
3. 检查 g.pileSize 的值是否小于零，如果是，函数立即返回。
4. 如果 g.pileSize 大于零，函数将调用一个名为 play 的函数(未定义)，传递 g.pileSize、g.minSelect、g.maxSelect、g.startOption 和 g.winOption 等参数。函数的作用参数列表中包含五个整数类型的参数，分别为 g.pileSize、minSelect、maxSelect、startOption 和 winOption。


```
func main() {
	for {
		printIntro()

		g := NewOptions()

		if g.pileSize < 0 {
			return
		}

		play(g.pileSize, g.minSelect, g.maxSelect, g.startOption, g.winOption)
	}
}

```