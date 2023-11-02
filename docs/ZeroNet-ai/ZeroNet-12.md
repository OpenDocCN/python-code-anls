# ZeroNet源码解析 12

# `plugins/UiConfig/media/js/lib/maquette.js`

This is a JavaScript library that provides a high-level API for creating animatronic projects such as those found in the "实验室" section of the website "LABO - Enriched Answerbot". The library allows developers to create animatronics by defining the required elements and properties of the animation, and then it handles the animation loop and scheduling.

The library provides several methods and properties for creating and managing animatronics, including:

* `doRender`: A function that performs the animation loop and scheduling.
* `scheduleRender`: A function that schedules `doRender` to be rendered.
* `renderCompleted`: A function that is called when the animation is complete.
* `projectionOptions`: An object that specifies the options for the projection.
* `renderFunctions`: An array of functions that will be rendered by the animatronic.
* `exports.dom`: An object that provides methods for working with the DOM.
* `createProjection`: A function that creates a projection for an element.
* `createDom`: A function that creates a DOM element.
* `addEventListener`: A function that adds an event listener to an element.
* `removeEventListener`: A function that removes an event listener from an element.
* `appendChild`: A function that adds a child element to an element.
* `prepareElement`: A function that prepares an element for rendering.
* `unscheduleAllRenders`: A function that removes all scheduled renders.

The library also provides a `Projector` class that handles the scheduling and execution of the animatronics. This class has methods for scheduling, rendering, and managing the elements of the animatronic.

Overall, the library is designed to make it easy for developers to create complex animatronics projects by providing a high-level API for defining and managing the elements and properties of the animation.


```py
(function (root, factory) {
    if (typeof define === 'function' && define.amd) {
        // AMD. Register as an anonymous module.
        define(['exports'], factory);
    } else if (typeof exports === 'object' && typeof exports.nodeName !== 'string') {
        // CommonJS
        factory(exports);
    } else {
        // Browser globals
        factory(root.maquette = {});
    }
}(this, function (exports) {
    'use strict';
    ;
    ;
    ;
    ;
    var NAMESPACE_W3 = 'http://www.w3.org/';
    var NAMESPACE_SVG = NAMESPACE_W3 + '2000/svg';
    var NAMESPACE_XLINK = NAMESPACE_W3 + '1999/xlink';
    // Utilities
    var emptyArray = [];
    var extend = function (base, overrides) {
        var result = {};
        Object.keys(base).forEach(function (key) {
            result[key] = base[key];
        });
        if (overrides) {
            Object.keys(overrides).forEach(function (key) {
                result[key] = overrides[key];
            });
        }
        return result;
    };
    // Hyperscript helper functions
    var same = function (vnode1, vnode2) {
        if (vnode1.vnodeSelector !== vnode2.vnodeSelector) {
            return false;
        }
        if (vnode1.properties && vnode2.properties) {
            if (vnode1.properties.key !== vnode2.properties.key) {
                return false;
            }
            return vnode1.properties.bind === vnode2.properties.bind;
        }
        return !vnode1.properties && !vnode2.properties;
    };
    var toTextVNode = function (data) {
        return {
            vnodeSelector: '',
            properties: undefined,
            children: undefined,
            text: data.toString(),
            domNode: null
        };
    };
    var appendChildren = function (parentSelector, insertions, main) {
        for (var i = 0; i < insertions.length; i++) {
            var item = insertions[i];
            if (Array.isArray(item)) {
                appendChildren(parentSelector, item, main);
            } else {
                if (item !== null && item !== undefined) {
                    if (!item.hasOwnProperty('vnodeSelector')) {
                        item = toTextVNode(item);
                    }
                    main.push(item);
                }
            }
        }
    };
    // Render helper functions
    var missingTransition = function () {
        throw new Error('Provide a transitions object to the projectionOptions to do animations');
    };
    var DEFAULT_PROJECTION_OPTIONS = {
        namespace: undefined,
        eventHandlerInterceptor: undefined,
        styleApplyer: function (domNode, styleName, value) {
            // Provides a hook to add vendor prefixes for browsers that still need it.
            domNode.style[styleName] = value;
        },
        transitions: {
            enter: missingTransition,
            exit: missingTransition
        }
    };
    var applyDefaultProjectionOptions = function (projectorOptions) {
        return extend(DEFAULT_PROJECTION_OPTIONS, projectorOptions);
    };
    var checkStyleValue = function (styleValue) {
        if (typeof styleValue !== 'string') {
            throw new Error('Style values must be strings');
        }
    };
    var setProperties = function (domNode, properties, projectionOptions) {
        if (!properties) {
            return;
        }
        var eventHandlerInterceptor = projectionOptions.eventHandlerInterceptor;
        var propNames = Object.keys(properties);
        var propCount = propNames.length;
        for (var i = 0; i < propCount; i++) {
            var propName = propNames[i];
            /* tslint:disable:no-var-keyword: edge case */
            var propValue = properties[propName];
            /* tslint:enable:no-var-keyword */
            if (propName === 'className') {
                throw new Error('Property "className" is not supported, use "class".');
            } else if (propName === 'class') {
                if (domNode.className) {
                    // May happen if classes is specified before class
                    domNode.className += ' ' + propValue;
                } else {
                    domNode.className = propValue;
                }
            } else if (propName === 'classes') {
                // object with string keys and boolean values
                var classNames = Object.keys(propValue);
                var classNameCount = classNames.length;
                for (var j = 0; j < classNameCount; j++) {
                    var className = classNames[j];
                    if (propValue[className]) {
                        domNode.classList.add(className);
                    }
                }
            } else if (propName === 'styles') {
                // object with string keys and string (!) values
                var styleNames = Object.keys(propValue);
                var styleCount = styleNames.length;
                for (var j = 0; j < styleCount; j++) {
                    var styleName = styleNames[j];
                    var styleValue = propValue[styleName];
                    if (styleValue) {
                        checkStyleValue(styleValue);
                        projectionOptions.styleApplyer(domNode, styleName, styleValue);
                    }
                }
            } else if (propName === 'key') {
                continue;
            } else if (propValue === null || propValue === undefined) {
                continue;
            } else {
                var type = typeof propValue;
                if (type === 'function') {
                    if (propName.lastIndexOf('on', 0) === 0) {
                        if (eventHandlerInterceptor) {
                            propValue = eventHandlerInterceptor(propName, propValue, domNode, properties);    // intercept eventhandlers
                        }
                        if (propName === 'oninput') {
                            (function () {
                                // record the evt.target.value, because IE and Edge sometimes do a requestAnimationFrame between changing value and running oninput
                                var oldPropValue = propValue;
                                propValue = function (evt) {
                                    evt.target['oninput-value'] = evt.target.value;
                                    // may be HTMLTextAreaElement as well
                                    oldPropValue.apply(this, [evt]);
                                };
                            }());
                        }
                        domNode[propName] = propValue;
                    }
                } else if (type === 'string' && propName !== 'value' && propName !== 'innerHTML') {
                    if (projectionOptions.namespace === NAMESPACE_SVG && propName === 'href') {
                        domNode.setAttributeNS(NAMESPACE_XLINK, propName, propValue);
                    } else {
                        domNode.setAttribute(propName, propValue);
                    }
                } else {
                    domNode[propName] = propValue;
                }
            }
        }
    };
    var updateProperties = function (domNode, previousProperties, properties, projectionOptions) {
        if (!properties) {
            return;
        }
        var propertiesUpdated = false;
        var propNames = Object.keys(properties);
        var propCount = propNames.length;
        for (var i = 0; i < propCount; i++) {
            var propName = propNames[i];
            // assuming that properties will be nullified instead of missing is by design
            var propValue = properties[propName];
            var previousValue = previousProperties[propName];
            if (propName === 'class') {
                if (previousValue !== propValue) {
                    throw new Error('"class" property may not be updated. Use the "classes" property for conditional css classes.');
                }
            } else if (propName === 'classes') {
                var classList = domNode.classList;
                var classNames = Object.keys(propValue);
                var classNameCount = classNames.length;
                for (var j = 0; j < classNameCount; j++) {
                    var className = classNames[j];
                    var on = !!propValue[className];
                    var previousOn = !!previousValue[className];
                    if (on === previousOn) {
                        continue;
                    }
                    propertiesUpdated = true;
                    if (on) {
                        classList.add(className);
                    } else {
                        classList.remove(className);
                    }
                }
            } else if (propName === 'styles') {
                var styleNames = Object.keys(propValue);
                var styleCount = styleNames.length;
                for (var j = 0; j < styleCount; j++) {
                    var styleName = styleNames[j];
                    var newStyleValue = propValue[styleName];
                    var oldStyleValue = previousValue[styleName];
                    if (newStyleValue === oldStyleValue) {
                        continue;
                    }
                    propertiesUpdated = true;
                    if (newStyleValue) {
                        checkStyleValue(newStyleValue);
                        projectionOptions.styleApplyer(domNode, styleName, newStyleValue);
                    } else {
                        projectionOptions.styleApplyer(domNode, styleName, '');
                    }
                }
            } else {
                if (!propValue && typeof previousValue === 'string') {
                    propValue = '';
                }
                if (propName === 'value') {
                    if (domNode[propName] !== propValue && domNode['oninput-value'] !== propValue) {
                        domNode[propName] = propValue;
                        // Reset the value, even if the virtual DOM did not change
                        domNode['oninput-value'] = undefined;
                    }
                    // else do not update the domNode, otherwise the cursor position would be changed
                    if (propValue !== previousValue) {
                        propertiesUpdated = true;
                    }
                } else if (propValue !== previousValue) {
                    var type = typeof propValue;
                    if (type === 'function') {
                        throw new Error('Functions may not be updated on subsequent renders (property: ' + propName + '). Hint: declare event handler functions outside the render() function.');
                    }
                    if (type === 'string' && propName !== 'innerHTML') {
                        if (projectionOptions.namespace === NAMESPACE_SVG && propName === 'href') {
                            domNode.setAttributeNS(NAMESPACE_XLINK, propName, propValue);
                        } else {
                            domNode.setAttribute(propName, propValue);
                        }
                    } else {
                        if (domNode[propName] !== propValue) {
                            domNode[propName] = propValue;
                        }
                    }
                    propertiesUpdated = true;
                }
            }
        }
        return propertiesUpdated;
    };
    var findIndexOfChild = function (children, sameAs, start) {
        if (sameAs.vnodeSelector !== '') {
            // Never scan for text-nodes
            for (var i = start; i < children.length; i++) {
                if (same(children[i], sameAs)) {
                    return i;
                }
            }
        }
        return -1;
    };
    var nodeAdded = function (vNode, transitions) {
        if (vNode.properties) {
            var enterAnimation = vNode.properties.enterAnimation;
            if (enterAnimation) {
                if (typeof enterAnimation === 'function') {
                    enterAnimation(vNode.domNode, vNode.properties);
                } else {
                    transitions.enter(vNode.domNode, vNode.properties, enterAnimation);
                }
            }
        }
    };
    var nodeToRemove = function (vNode, transitions) {
        var domNode = vNode.domNode;
        if (vNode.properties) {
            var exitAnimation = vNode.properties.exitAnimation;
            if (exitAnimation) {
                domNode.style.pointerEvents = 'none';
                var removeDomNode = function () {
                    if (domNode.parentNode) {
                        domNode.parentNode.removeChild(domNode);
                    }
                };
                if (typeof exitAnimation === 'function') {
                    exitAnimation(domNode, removeDomNode, vNode.properties);
                    return;
                } else {
                    transitions.exit(vNode.domNode, vNode.properties, exitAnimation, removeDomNode);
                    return;
                }
            }
        }
        if (domNode.parentNode) {
            domNode.parentNode.removeChild(domNode);
        }
    };
    var checkDistinguishable = function (childNodes, indexToCheck, parentVNode, operation) {
        var childNode = childNodes[indexToCheck];
        if (childNode.vnodeSelector === '') {
            return;    // Text nodes need not be distinguishable
        }
        var properties = childNode.properties;
        var key = properties ? properties.key === undefined ? properties.bind : properties.key : undefined;
        if (!key) {
            for (var i = 0; i < childNodes.length; i++) {
                if (i !== indexToCheck) {
                    var node = childNodes[i];
                    if (same(node, childNode)) {
                        if (operation === 'added') {
                            throw new Error(parentVNode.vnodeSelector + ' had a ' + childNode.vnodeSelector + ' child ' + 'added, but there is now more than one. You must add unique key properties to make them distinguishable.');
                        } else {
                            throw new Error(parentVNode.vnodeSelector + ' had a ' + childNode.vnodeSelector + ' child ' + 'removed, but there were more than one. You must add unique key properties to make them distinguishable.');
                        }
                    }
                }
            }
        }
    };
    var createDom;
    var updateDom;
    var updateChildren = function (vnode, domNode, oldChildren, newChildren, projectionOptions) {
        if (oldChildren === newChildren) {
            return false;
        }
        oldChildren = oldChildren || emptyArray;
        newChildren = newChildren || emptyArray;
        var oldChildrenLength = oldChildren.length;
        var newChildrenLength = newChildren.length;
        var transitions = projectionOptions.transitions;
        var oldIndex = 0;
        var newIndex = 0;
        var i;
        var textUpdated = false;
        while (newIndex < newChildrenLength) {
            var oldChild = oldIndex < oldChildrenLength ? oldChildren[oldIndex] : undefined;
            var newChild = newChildren[newIndex];
            if (oldChild !== undefined && same(oldChild, newChild)) {
                textUpdated = updateDom(oldChild, newChild, projectionOptions) || textUpdated;
                oldIndex++;
            } else {
                var findOldIndex = findIndexOfChild(oldChildren, newChild, oldIndex + 1);
                if (findOldIndex >= 0) {
                    // Remove preceding missing children
                    for (i = oldIndex; i < findOldIndex; i++) {
                        nodeToRemove(oldChildren[i], transitions);
                        checkDistinguishable(oldChildren, i, vnode, 'removed');
                    }
                    textUpdated = updateDom(oldChildren[findOldIndex], newChild, projectionOptions) || textUpdated;
                    oldIndex = findOldIndex + 1;
                } else {
                    // New child
                    createDom(newChild, domNode, oldIndex < oldChildrenLength ? oldChildren[oldIndex].domNode : undefined, projectionOptions);
                    nodeAdded(newChild, transitions);
                    checkDistinguishable(newChildren, newIndex, vnode, 'added');
                }
            }
            newIndex++;
        }
        if (oldChildrenLength > oldIndex) {
            // Remove child fragments
            for (i = oldIndex; i < oldChildrenLength; i++) {
                nodeToRemove(oldChildren[i], transitions);
                checkDistinguishable(oldChildren, i, vnode, 'removed');
            }
        }
        return textUpdated;
    };
    var addChildren = function (domNode, children, projectionOptions) {
        if (!children) {
            return;
        }
        for (var i = 0; i < children.length; i++) {
            createDom(children[i], domNode, undefined, projectionOptions);
        }
    };
    var initPropertiesAndChildren = function (domNode, vnode, projectionOptions) {
        addChildren(domNode, vnode.children, projectionOptions);
        // children before properties, needed for value property of <select>.
        if (vnode.text) {
            domNode.textContent = vnode.text;
        }
        setProperties(domNode, vnode.properties, projectionOptions);
        if (vnode.properties && vnode.properties.afterCreate) {
            vnode.properties.afterCreate(domNode, projectionOptions, vnode.vnodeSelector, vnode.properties, vnode.children);
        }
    };
    createDom = function (vnode, parentNode, insertBefore, projectionOptions) {
        var domNode, i, c, start = 0, type, found;
        var vnodeSelector = vnode.vnodeSelector;
        if (vnodeSelector === '') {
            domNode = vnode.domNode = document.createTextNode(vnode.text);
            if (insertBefore !== undefined) {
                parentNode.insertBefore(domNode, insertBefore);
            } else {
                parentNode.appendChild(domNode);
            }
        } else {
            for (i = 0; i <= vnodeSelector.length; ++i) {
                c = vnodeSelector.charAt(i);
                if (i === vnodeSelector.length || c === '.' || c === '#') {
                    type = vnodeSelector.charAt(start - 1);
                    found = vnodeSelector.slice(start, i);
                    if (type === '.') {
                        domNode.classList.add(found);
                    } else if (type === '#') {
                        domNode.id = found;
                    } else {
                        if (found === 'svg') {
                            projectionOptions = extend(projectionOptions, { namespace: NAMESPACE_SVG });
                        }
                        if (projectionOptions.namespace !== undefined) {
                            domNode = vnode.domNode = document.createElementNS(projectionOptions.namespace, found);
                        } else {
                            domNode = vnode.domNode = document.createElement(found);
                        }
                        if (insertBefore !== undefined) {
                            parentNode.insertBefore(domNode, insertBefore);
                        } else {
                            parentNode.appendChild(domNode);
                        }
                    }
                    start = i + 1;
                }
            }
            initPropertiesAndChildren(domNode, vnode, projectionOptions);
        }
    };
    updateDom = function (previous, vnode, projectionOptions) {
        var domNode = previous.domNode;
        var textUpdated = false;
        if (previous === vnode) {
            return false;    // By contract, VNode objects may not be modified anymore after passing them to maquette
        }
        var updated = false;
        if (vnode.vnodeSelector === '') {
            if (vnode.text !== previous.text) {
                var newVNode = document.createTextNode(vnode.text);
                domNode.parentNode.replaceChild(newVNode, domNode);
                vnode.domNode = newVNode;
                textUpdated = true;
                return textUpdated;
            }
        } else {
            if (vnode.vnodeSelector.lastIndexOf('svg', 0) === 0) {
                projectionOptions = extend(projectionOptions, { namespace: NAMESPACE_SVG });
            }
            if (previous.text !== vnode.text) {
                updated = true;
                if (vnode.text === undefined) {
                    domNode.removeChild(domNode.firstChild);    // the only textnode presumably
                } else {
                    domNode.textContent = vnode.text;
                }
            }
            updated = updateChildren(vnode, domNode, previous.children, vnode.children, projectionOptions) || updated;
            updated = updateProperties(domNode, previous.properties, vnode.properties, projectionOptions) || updated;
            if (vnode.properties && vnode.properties.afterUpdate) {
                vnode.properties.afterUpdate(domNode, projectionOptions, vnode.vnodeSelector, vnode.properties, vnode.children);
            }
        }
        if (updated && vnode.properties && vnode.properties.updateAnimation) {
            vnode.properties.updateAnimation(domNode, vnode.properties, previous.properties);
        }
        vnode.domNode = previous.domNode;
        return textUpdated;
    };
    var createProjection = function (vnode, projectionOptions) {
        return {
            update: function (updatedVnode) {
                if (vnode.vnodeSelector !== updatedVnode.vnodeSelector) {
                    throw new Error('The selector for the root VNode may not be changed. (consider using dom.merge and add one extra level to the virtual DOM)');
                }
                updateDom(vnode, updatedVnode, projectionOptions);
                vnode = updatedVnode;
            },
            domNode: vnode.domNode
        };
    };
    ;
    // The other two parameters are not added here, because the Typescript compiler creates surrogate code for desctructuring 'children'.
    exports.h = function (selector) {
        var properties = arguments[1];
        if (typeof selector !== 'string') {
            throw new Error();
        }
        var childIndex = 1;
        if (properties && !properties.hasOwnProperty('vnodeSelector') && !Array.isArray(properties) && typeof properties === 'object') {
            childIndex = 2;
        } else {
            // Optional properties argument was omitted
            properties = undefined;
        }
        var text = undefined;
        var children = undefined;
        var argsLength = arguments.length;
        // Recognize a common special case where there is only a single text node
        if (argsLength === childIndex + 1) {
            var onlyChild = arguments[childIndex];
            if (typeof onlyChild === 'string') {
                text = onlyChild;
            } else if (onlyChild !== undefined && onlyChild.length === 1 && typeof onlyChild[0] === 'string') {
                text = onlyChild[0];
            }
        }
        if (text === undefined) {
            children = [];
            for (; childIndex < arguments.length; childIndex++) {
                var child = arguments[childIndex];
                if (child === null || child === undefined) {
                    continue;
                } else if (Array.isArray(child)) {
                    appendChildren(selector, child, children);
                } else if (child.hasOwnProperty('vnodeSelector')) {
                    children.push(child);
                } else {
                    children.push(toTextVNode(child));
                }
            }
        }
        return {
            vnodeSelector: selector,
            properties: properties,
            children: children,
            text: text === '' ? undefined : text,
            domNode: null
        };
    };
    /**
 * Contains simple low-level utility functions to manipulate the real DOM.
 */
    exports.dom = {
        /**
     * Creates a real DOM tree from `vnode`. The [[Projection]] object returned will contain the resulting DOM Node in
     * its [[Projection.domNode|domNode]] property.
     * This is a low-level method. Users wil typically use a [[Projector]] instead.
     * @param vnode - The root of the virtual DOM tree that was created using the [[h]] function. NOTE: [[VNode]]
     * objects may only be rendered once.
     * @param projectionOptions - Options to be used to create and update the projection.
     * @returns The [[Projection]] which also contains the DOM Node that was created.
     */
        create: function (vnode, projectionOptions) {
            projectionOptions = applyDefaultProjectionOptions(projectionOptions);
            createDom(vnode, document.createElement('div'), undefined, projectionOptions);
            return createProjection(vnode, projectionOptions);
        },
        /**
     * Appends a new childnode to the DOM which is generated from a [[VNode]].
     * This is a low-level method. Users wil typically use a [[Projector]] instead.
     * @param parentNode - The parent node for the new childNode.
     * @param vnode - The root of the virtual DOM tree that was created using the [[h]] function. NOTE: [[VNode]]
     * objects may only be rendered once.
     * @param projectionOptions - Options to be used to create and update the [[Projection]].
     * @returns The [[Projection]] that was created.
     */
        append: function (parentNode, vnode, projectionOptions) {
            projectionOptions = applyDefaultProjectionOptions(projectionOptions);
            createDom(vnode, parentNode, undefined, projectionOptions);
            return createProjection(vnode, projectionOptions);
        },
        /**
     * Inserts a new DOM node which is generated from a [[VNode]].
     * This is a low-level method. Users wil typically use a [[Projector]] instead.
     * @param beforeNode - The node that the DOM Node is inserted before.
     * @param vnode - The root of the virtual DOM tree that was created using the [[h]] function.
     * NOTE: [[VNode]] objects may only be rendered once.
     * @param projectionOptions - Options to be used to create and update the projection, see [[createProjector]].
     * @returns The [[Projection]] that was created.
     */
        insertBefore: function (beforeNode, vnode, projectionOptions) {
            projectionOptions = applyDefaultProjectionOptions(projectionOptions);
            createDom(vnode, beforeNode.parentNode, beforeNode, projectionOptions);
            return createProjection(vnode, projectionOptions);
        },
        /**
     * Merges a new DOM node which is generated from a [[VNode]] with an existing DOM Node.
     * This means that the virtual DOM and the real DOM will have one overlapping element.
     * Therefore the selector for the root [[VNode]] will be ignored, but its properties and children will be applied to the Element provided.
     * This is a low-level method. Users wil typically use a [[Projector]] instead.
     * @param domNode - The existing element to adopt as the root of the new virtual DOM. Existing attributes and childnodes are preserved.
     * @param vnode - The root of the virtual DOM tree that was created using the [[h]] function. NOTE: [[VNode]] objects
     * may only be rendered once.
     * @param projectionOptions - Options to be used to create and update the projection, see [[createProjector]].
     * @returns The [[Projection]] that was created.
     */
        merge: function (element, vnode, projectionOptions) {
            projectionOptions = applyDefaultProjectionOptions(projectionOptions);
            vnode.domNode = element;
            initPropertiesAndChildren(element, vnode, projectionOptions);
            return createProjection(vnode, projectionOptions);
        }
    };
    /**
 * Creates a [[CalculationCache]] object, useful for caching [[VNode]] trees.
 * In practice, caching of [[VNode]] trees is not needed, because achieving 60 frames per second is almost never a problem.
 * For more information, see [[CalculationCache]].
 *
 * @param <Result> The type of the value that is cached.
 */
    exports.createCache = function () {
        var cachedInputs = undefined;
        var cachedOutcome = undefined;
        var result = {
            invalidate: function () {
                cachedOutcome = undefined;
                cachedInputs = undefined;
            },
            result: function (inputs, calculation) {
                if (cachedInputs) {
                    for (var i = 0; i < inputs.length; i++) {
                        if (cachedInputs[i] !== inputs[i]) {
                            cachedOutcome = undefined;
                        }
                    }
                }
                if (!cachedOutcome) {
                    cachedOutcome = calculation();
                    cachedInputs = inputs;
                }
                return cachedOutcome;
            }
        };
        return result;
    };
    /**
 * Creates a {@link Mapping} instance that keeps an array of result objects synchronized with an array of source objects.
 * See {@link http://maquettejs.org/docs/arrays.html|Working with arrays}.
 *
 * @param <Source>       The type of source items. A database-record for instance.
 * @param <Target>       The type of target items. A [[Component]] for instance.
 * @param getSourceKey   `function(source)` that must return a key to identify each source object. The result must either be a string or a number.
 * @param createResult   `function(source, index)` that must create a new result object from a given source. This function is identical
 *                       to the `callback` argument in `Array.map(callback)`.
 * @param updateResult   `function(source, target, index)` that updates a result to an updated source.
 */
    exports.createMapping = function (getSourceKey, createResult, updateResult) {
        var keys = [];
        var results = [];
        return {
            results: results,
            map: function (newSources) {
                var newKeys = newSources.map(getSourceKey);
                var oldTargets = results.slice();
                var oldIndex = 0;
                for (var i = 0; i < newSources.length; i++) {
                    var source = newSources[i];
                    var sourceKey = newKeys[i];
                    if (sourceKey === keys[oldIndex]) {
                        results[i] = oldTargets[oldIndex];
                        updateResult(source, oldTargets[oldIndex], i);
                        oldIndex++;
                    } else {
                        var found = false;
                        for (var j = 1; j < keys.length; j++) {
                            var searchIndex = (oldIndex + j) % keys.length;
                            if (keys[searchIndex] === sourceKey) {
                                results[i] = oldTargets[searchIndex];
                                updateResult(newSources[i], oldTargets[searchIndex], i);
                                oldIndex = searchIndex + 1;
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            results[i] = createResult(source, i);
                        }
                    }
                }
                results.length = newSources.length;
                keys = newKeys;
            }
        };
    };
    /**
 * Creates a [[Projector]] instance using the provided projectionOptions.
 *
 * For more information, see [[Projector]].
 *
 * @param projectionOptions   Options that influence how the DOM is rendered and updated.
 */
    exports.createProjector = function (projectorOptions) {
        var projector;
        var projectionOptions = applyDefaultProjectionOptions(projectorOptions);
        projectionOptions.eventHandlerInterceptor = function (propertyName, eventHandler, domNode, properties) {
            return function () {
                // intercept function calls (event handlers) to do a render afterwards.
                projector.scheduleRender();
                return eventHandler.apply(properties.bind || this, arguments);
            };
        };
        var renderCompleted = true;
        var scheduled;
        var stopped = false;
        var projections = [];
        var renderFunctions = [];
        // matches the projections array
        var doRender = function () {
            scheduled = undefined;
            if (!renderCompleted) {
                return;    // The last render threw an error, it should be logged in the browser console.
            }
            renderCompleted = false;
            for (var i = 0; i < projections.length; i++) {
                var updatedVnode = renderFunctions[i]();
                projections[i].update(updatedVnode);
            }
            renderCompleted = true;
        };
        projector = {
            scheduleRender: function () {
                if (!scheduled && !stopped) {
                    scheduled = requestAnimationFrame(doRender);
                }
            },
            stop: function () {
                if (scheduled) {
                    cancelAnimationFrame(scheduled);
                    scheduled = undefined;
                }
                stopped = true;
            },
            resume: function () {
                stopped = false;
                renderCompleted = true;
                projector.scheduleRender();
            },
            append: function (parentNode, renderMaquetteFunction) {
                projections.push(exports.dom.append(parentNode, renderMaquetteFunction(), projectionOptions));
                renderFunctions.push(renderMaquetteFunction);
            },
            insertBefore: function (beforeNode, renderMaquetteFunction) {
                projections.push(exports.dom.insertBefore(beforeNode, renderMaquetteFunction(), projectionOptions));
                renderFunctions.push(renderMaquetteFunction);
            },
            merge: function (domNode, renderMaquetteFunction) {
                projections.push(exports.dom.merge(domNode, renderMaquetteFunction(), projectionOptions));
                renderFunctions.push(renderMaquetteFunction);
            },
            replace: function (domNode, renderMaquetteFunction) {
                var vnode = renderMaquetteFunction();
                createDom(vnode, domNode.parentNode, domNode, projectionOptions);
                domNode.parentNode.removeChild(domNode);
                projections.push(createProjection(vnode, projectionOptions));
                renderFunctions.push(renderMaquetteFunction);
            },
            detach: function (renderMaquetteFunction) {
                for (var i = 0; i < renderFunctions.length; i++) {
                    if (renderFunctions[i] === renderMaquetteFunction) {
                        renderFunctions.splice(i, 1);
                        return projections.splice(i, 1)[0];
                    }
                }
                throw new Error('renderMaquetteFunction was not found');
            }
        };
        return projector;
    };
}));

```

# `plugins/UiFileManager/UiFileManagerPlugin.py`

这段代码使用了Python的一些第三方库和函数，主要作用是获取并导入一些工具类和函数，用于处理和处理文本数据的开源工具。

具体来说，这段代码实现了以下功能：

1. 导入一些必要的库和函数：io、os、re、urllib、PluginManager、config、Translate。

2. 在插件目录下创建一个languages目录，用于存放所有本地化文件。

3. 通过os.path.dirname(__file__)获取当前文件所在的目录，并将其作为插件目录。

4. 导入PluginManager，实现对插件的管理和加载。

5. 导入Config，实现对配置文件的管理和读取。

6. 导入Translate，实现对文本数据的翻译和处理。

7. 在插件加载完成后，获取当前插件目录下的所有本地化文件，并将其递归地导入到对应的翻译类中。

8. 通过PluginManager.get_system_preferences()获取当前系统的偏好设置，并将其存储在Config.settings_file()中，以便在插件运行时自动加载设置。

9. 通过插件Manager.get_插件_by_name()获取所有可运行的插件，并将其存储在PluginManager.current()中，以便用户在运行时加载和卸载插件。

这段代码主要用于提供一个简单的文本数据处理和翻译工具，以满足某些应用程序的需求。


```py
import io
import os
import re
import urllib

from Plugin import PluginManager
from Config import config
from Translate import Translate

plugin_dir = os.path.dirname(__file__)

if "_" not in locals():
    _ = Translate(plugin_dir + "/languages/")


```

This is a Flask blueprint for serving static media files. It includes a route for serving media files from a directory, as well as support for serving HTML files.

When a request is made to the URL `/media/<path_for_file>`, the blueprint will check if the file is located in the directory specified by `file_path`. If the file is not found, it will return a 404 error.

If the file is found, it will be read in byio and passed to the `DebugMedia` class, which will then cache the file for future requests.

If the file is an HTML file, it will be transformed by the `translateData` method to convert the file to JavaScript. This will be done in debug mode, so you should use `async` with `await` to make the call to `translateData` successfully.

If the file is not an HTML or JavaScript file, it will be served by reading the contents of the file.

The blueprint also includes a logging mechanism, where any errors occurring during the processing of a request will be logged and logged with a tag `DEBUG`.


```py
@PluginManager.registerTo("UiRequest")
class UiFileManagerPlugin(object):
    def actionWrapper(self, path, extra_headers=None):
        match = re.match("/list/(.*?)(/.*|)$", path)
        if not match:
            return super().actionWrapper(path, extra_headers)

        if not extra_headers:
            extra_headers = {}

        request_address, inner_path = match.groups()

        script_nonce = self.getScriptNonce()

        self.sendHeader(extra_headers=extra_headers, script_nonce=script_nonce)

        site = self.server.site_manager.need(request_address)

        if not site:
            return super().actionWrapper(path, extra_headers)

        request_params = urllib.parse.urlencode(
            {"address": site.address, "site": request_address, "inner_path": inner_path.strip("/")}
        )

        is_content_loaded = "content.json" in site.content_manager.contents

        return iter([super().renderWrapper(
            site, path, "uimedia/plugins/uifilemanager/list.html?%s" % request_params,
            "List", extra_headers, show_loadingscreen=not is_content_loaded, script_nonce=script_nonce
        )])

    def actionUiMedia(self, path, *args, **kwargs):
        if path.startswith("/uimedia/plugins/uifilemanager/"):
            file_path = path.replace("/uimedia/plugins/uifilemanager/", plugin_dir + "/media/")
            if config.debug and (file_path.endswith("all.js") or file_path.endswith("all.css")):
                # If debugging merge *.css to all.css and *.js to all.js
                from Debug import DebugMedia
                DebugMedia.merge(file_path)

            if file_path.endswith("js"):
                data = _.translateData(open(file_path).read(), mode="js").encode("utf8")
            elif file_path.endswith("html"):
                if self.get.get("address"):
                    site = self.server.site_manager.need(self.get.get("address"))
                    if "content.json" not in site.content_manager.contents:
                        site.needFile("content.json")
                data = _.translateData(open(file_path).read(), mode="html").encode("utf8")
            else:
                data = open(file_path, "rb").read()

            return self.actionFile(file_path, file_obj=io.BytesIO(data), file_size=len(data))
        else:
            return super().actionUiMedia(path)

    def error404(self, path=""):
        if not path.endswith("index.html") and not path.endswith("/"):
            return super().error404(path)

        path_parts = self.parsePath(path)
        if not path_parts:
            return super().error404(path)

        site = self.server.site_manager.get(path_parts["request_address"])

        if not site or not site.content_manager.contents.get("content.json"):
            return super().error404(path)

        if path_parts["inner_path"] in site.content_manager.contents.get("content.json").get("files", {}):
            return super().error404(path)

        self.sendHeader(200)
        path_redirect = "/list" + re.sub("^/media/", "/", path)
        self.log.debug("Index.html not found: %s, redirecting to: %s" % (path, path_redirect))
        return self.formatRedirect(path_redirect)

```

# `plugins/UiFileManager/__init__.py`

这段代码使用了Python的导入（import）语句，用于将名为“UiFileManagerPlugin”的类从指定的包中导入。在这个例子中，这个包的名称是“.”（没有给出具体的包名，包名可以自行设定）。

具体来说，这段代码的作用是：从包中引入名为“UiFileManagerPlugin”的类，以便在当前脚本中使用该类。由于没有提供具体的包名，因此无法确定类UiFileManagerPlugin会在哪个包中定义。


```py
from . import UiFileManagerPlugin

```

# ZeroName

Zeroname plugin to connect Namecoin and register all the .bit domain name.

## Start

You can create your own Zeroname.

### Namecoin node

You need to run a namecoin node.

[Namecoin](https://namecoin.org/download/)

You will need to start it as a RPC server.

Example of `~/.namecoin/namecoin.conf` minimal setup:
```py
daemon=1
rpcuser=your-name
rpcpassword=your-password
rpcport=8336
server=1
txindex=1
valueencoding=utf8
```

Don't forget to change the `rpcuser` value and `rpcpassword` value!

You can start your node : `./namecoind`

### Create a Zeroname site

You will also need to create a site `python zeronet.py createSite` and regitser the info.

In the site you will need to create a file `./data/<your-site>/data/names.json` with this is it:
```py
{}
```

### `zeroname_config.json` file

In `~/.namecoin/zeroname_config.json`
```py
{
  "lastprocessed": 223910,
  "zeronet_path": "/root/ZeroNet", # Update with your path
  "privatekey": "", # Update with your private key of your site
  "site": "" # Update with the address of your site
}
```

### Run updater

You can now run the script : `updater/zeroname_updater.py` and wait until it is fully sync (it might take a while).


pyaes
=====

A pure-Python implementation of the AES block cipher algorithm and the common modes of operation (CBC, CFB, CTR, ECB and OFB).


Features
--------

* Supports all AES key sizes
* Supports all AES common modes
* Pure-Python (no external dependencies)
* BlockFeeder API allows streams to easily be encrypted and decrypted
* Python 2.x and 3.x support (make sure you pass in bytes(), not strings for Python 3)


API
---

All keys may be 128 bits (16 bytes), 192 bits (24 bytes) or 256 bits (32 bytes) long.

To generate a random key use:
```pypython
import os

# 128 bit, 192 bit and 256 bit keys
key_128 = os.urandom(16)
key_192 = os.urandom(24)
key_256 = os.urandom(32)
```

To generate keys from simple-to-remember passwords, consider using a _password-based key-derivation function_ such as [scrypt](https://github.com/ricmoo/pyscrypt).


### Common Modes of Operation

There are many modes of operations, each with various pros and cons. In general though, the **CBC** and **CTR** modes are recommended. The **ECB is NOT recommended.**, and is included primarily for completeness.

Each of the following examples assumes the following key:
```pypython
import pyaes

# A 256 bit (32 byte) key
key = "This_key_for_demo_purposes_only!"

# For some modes of operation we need a random initialization vector
# of 16 bytes
iv = "InitializationVe"
```


#### Counter Mode of Operation (recommended)

```pypython
aes = pyaes.AESModeOfOperationCTR(key)
plaintext = "Text may be any length you wish, no padding is required"
ciphertext = aes.encrypt(plaintext)

# '''\xb6\x99\x10=\xa4\x96\x88\xd1\x89\x1co\xe6\x1d\xef;\x11\x03\xe3\xee
#    \xa9V?wY\xbfe\xcdO\xe3\xdf\x9dV\x19\xe5\x8dk\x9fh\xb87>\xdb\xa3\xd6
#    \x86\xf4\xbd\xb0\x97\xf1\t\x02\xe9 \xed'''
print repr(ciphertext)

# The counter mode of operation maintains state, so decryption requires
# a new instance be created
aes = pyaes.AESModeOfOperationCTR(key)
decrypted = aes.decrypt(ciphertext)

# True
print decrypted == plaintext

# To use a custom initial value
counter = pyaes.Counter(initial_value = 100)
aes = pyaes.AESModeOfOperationCTR(key, counter = counter)
ciphertext = aes.encrypt(plaintext)

# '''WZ\x844\x02\xbfoY\x1f\x12\xa6\xce\x03\x82Ei)\xf6\x97mX\x86\xe3\x9d
#    _1\xdd\xbd\x87\xb5\xccEM_4\x01$\xa6\x81\x0b\xd5\x04\xd7Al\x07\xe5
#    \xb2\x0e\\\x0f\x00\x13,\x07'''
print repr(ciphertext)
```


#### Cipher-Block Chaining (recommended)

```pypython
aes = pyaes.AESModeOfOperationCBC(key, iv = iv)
plaintext = "TextMustBe16Byte"
ciphertext = aes.encrypt(plaintext)

# '\xd6:\x18\xe6\xb1\xb3\xc3\xdc\x87\xdf\xa7|\x08{k\xb6'
print repr(ciphertext)


# The cipher-block chaining mode of operation maintains state, so
# decryption requires a new instance be created
aes = pyaes.AESModeOfOperationCBC(key, iv = iv)
decrypted = aes.decrypt(ciphertext)

# True
print decrypted == plaintext
```


#### Cipher Feedback

```pypython
# Each block into the mode of operation must be a multiple of the segment
# size. For this example we choose 8 bytes.
aes = pyaes.AESModeOfOperationCFB(key, iv = iv, segment_size = 8)
plaintext =  "TextMustBeAMultipleOfSegmentSize"
ciphertext = aes.encrypt(plaintext)

# '''v\xa9\xc1w"\x8aL\x93\xcb\xdf\xa0/\xf8Y\x0b\x8d\x88i\xcb\x85rmp
#    \x85\xfe\xafM\x0c)\xd5\xeb\xaf'''
print repr(ciphertext)


# The cipher-block chaining mode of operation maintains state, so
# decryption requires a new instance be created
aes = pyaes.AESModeOfOperationCFB(key, iv = iv, segment_size = 8)
decrypted = aes.decrypt(ciphertext)

# True
print decrypted == plaintext
```


#### Output Feedback Mode of Operation

```pypython
aes = pyaes.AESModeOfOperationOFB(key, iv = iv)
plaintext = "Text may be any length you wish, no padding is required"
ciphertext = aes.encrypt(plaintext)

# '''v\xa9\xc1wO\x92^\x9e\rR\x1e\xf7\xb1\xa2\x9d"l1\xc7\xe7\x9d\x87(\xc26s
#    \xdd8\xc8@\xb6\xd9!\xf5\x0cM\xaa\x9b\xc4\xedLD\xe4\xb9\xd8\xdf\x9e\xac
#    \xa1\xb8\xea\x0f\x8ev\xb5'''
print repr(ciphertext)

# The counter mode of operation maintains state, so decryption requires
# a new instance be created
aes = pyaes.AESModeOfOperationOFB(key, iv = iv)
decrypted = aes.decrypt(ciphertext)

# True
print decrypted == plaintext
```


#### Electronic Codebook (NOT recommended)

```pypython
aes = pyaes.AESModeOfOperationECB(key)
plaintext = "TextMustBe16Byte"
ciphertext = aes.encrypt(plaintext)

# 'L6\x95\x85\xe4\xd9\xf1\x8a\xfb\xe5\x94X\x80|\x19\xc3'
print repr(ciphertext)

# Since there is no state stored in this mode of operation, it
# is not necessary to create a new aes object for decryption.
#aes = pyaes.AESModeOfOperationECB(key)
decrypted = aes.decrypt(ciphertext)

# True
print decrypted == plaintext
```


### BlockFeeder

Since most of the modes of operations require data in specific block-sized or segment-sized blocks, it can be difficult when working with large arbitrary streams or strings of data.

The BlockFeeder class is meant to make life easier for you, by buffering bytes across multiple calls and returning bytes as they are available, as well as padding or stripping the output when finished, if necessary.

```pypython
import pyaes

# Any mode of operation can be used; for this example CBC
key = "This_key_for_demo_purposes_only!"
iv = "InitializationVe"

ciphertext = ''

# We can encrypt one line at a time, regardles of length
encrypter = pyaes.Encrypter(pyaes.AESModeOfOperationCBC(key, iv))
for line in file('/etc/passwd'):
    ciphertext += encrypter.feed(line)

# Make a final call to flush any remaining bytes and add paddin
ciphertext += encrypter.feed()

# We can decrypt the cipher text in chunks (here we split it in half)
decrypter = pyaes.Decrypter(pyaes.AESModeOfOperationCBC(key, iv))
decrypted = decrypter.feed(ciphertext[:len(ciphertext) / 2])
decrypted += decrypter.feed(ciphertext[len(ciphertext) / 2:])

# Again, make a final call to flush any remaining bytes and strip padding
decrypted += decrypter.feed()

print file('/etc/passwd').read() == decrypted
```

### Stream Feeder

This is meant to make it even easier to encrypt and decrypt streams and large files.

```pypython
import pyaes

# Any mode of operation can be used; for this example CTR
key = "This_key_for_demo_purposes_only!"

# Create the mode of operation to encrypt with
mode = pyaes.AESModeOfOperationCTR(key)

# The input and output files
file_in = file('/etc/passwd')
file_out = file('/tmp/encrypted.bin', 'wb')

# Encrypt the data as a stream, the file is read in 8kb chunks, be default
pyaes.encrypt_stream(mode, file_in, file_out)

# Close the files
file_in.close()
file_out.close()
```

Decrypting is identical, except you would use `pyaes.decrypt_stream`, and the encrypted file would be the `file_in` and target for decryption the `file_out`.

### AES block cipher

Generally you should use one of the modes of operation above. This may however be useful for experimenting with a custom mode of operation or dealing with encrypted blocks.

The block cipher requires exactly one block of data to encrypt or decrypt, and each block should be an array with each element an integer representation of a byte.

```pypython
import pyaes

# 16 byte block of plain text
plaintext = "Hello World!!!!!"
plaintext_bytes = [ ord(c) for c in plaintext ]

# 32 byte key (256 bit)
key = "This_key_for_demo_purposes_only!"

# Our AES instance
aes = pyaes.AES(key)

# Encrypt!
ciphertext = aes.encrypt(plaintext_bytes)

# [55, 250, 182, 25, 185, 208, 186, 95, 206, 115, 50, 115, 108, 58, 174, 115]
print repr(ciphertext)

# Decrypt!
decrypted = aes.decrypt(ciphertext)

# True
print decrypted == plaintext_bytes
```

What is a key?
--------------

This seems to be a point of confusion for many people new to using encryption. You can think of the key as the *"password"*. However, these algorithms require the *"password"* to be a specific length.

With AES, there are three possible key lengths, 16-bytes, 24-bytes or 32-bytes. When you create an AES object, the key size is automatically detected, so it is important to pass in a key of the correct length.

Often, you wish to provide a password of arbitrary length, for example, something easy to remember or write down. In these cases, you must come up with a way to transform the password into a key, of a specific length. A **Password-Based Key Derivation Function** (PBKDF) is an algorithm designed for this exact purpose.

Here is an example, using the popular (possibly obsolete?) *crypt* PBKDF:

```py
# See: https://www.dlitz.net/software/python-pbkdf2/
import pbkdf2

password = "HelloWorld"

# The crypt PBKDF returns a 48-byte string
key = pbkdf2.crypt(password)

# A 16-byte, 24-byte and 32-byte key, respectively
key_16 = key[:16]
key_24 = key[:24]
key_32 = key[:32]
```

The [scrypt](https://github.com/ricmoo/pyscrypt) PBKDF is intentionally slow, to make it more difficult to brute-force guess a password:

```py
# See: https://github.com/ricmoo/pyscrypt
import pyscrypt

password = "HelloWorld"

# Salt is required, and prevents Rainbow Table attacks
salt = "SeaSalt"

# N, r, and p are parameters to specify how difficult it should be to
# generate a key; bigger numbers take longer and more memory
N = 1024
r = 1
p = 1

# A 16-byte, 24-byte and 32-byte key, respectively; the scrypt algorithm takes
# a 6-th parameter, indicating key length
key_16 = pyscrypt.hash(password, salt, N, r, p, 16)
key_24 = pyscrypt.hash(password, salt, N, r, p, 24)
key_32 = pyscrypt.hash(password, salt, N, r, p, 32)
```

Another possibility, is to use a hashing function, such as SHA256 to hash the password, but this method may be vulnerable to [Rainbow Attacks](http://en.wikipedia.org/wiki/Rainbow_table), unless you use a [salt](http://en.wikipedia.org/wiki/Salt_(cryptography)).

```pypython
import hashlib

password = "HelloWorld"

# The SHA256 hash algorithm returns a 32-byte string
hashed = hashlib.sha256(password).digest()

# A 16-byte, 24-byte and 32-byte key, respectively
key_16 = hashed[:16]
key_24 = hashed[:24]
key_32 = hashed
```




Performance
-----------

There is a test case provided in _/tests/test-aes.py_ which does some basic performance testing (its primary purpose is moreso as a regression test).

Based on that test, in **CPython**, this library is about 30x slower than [PyCrypto](https://www.dlitz.net/software/pycrypto/) for CBC, ECB and OFB; about 80x slower for CFB; and 300x slower for CTR.

Based on that same test, in **Pypy**, this library is about 4x slower than [PyCrypto](https://www.dlitz.net/software/pycrypto/) for CBC, ECB and OFB; about 12x slower for CFB; and 19x slower for CTR.

The PyCrypto documentation makes reference to the counter call being responsible for the speed problems of the counter (CTR) mode of operation, which is why they use a specially optimized counter. I will investigate this problem further in the future.


FAQ
---

#### Why do this?

The short answer, *why not?*

The longer answer, is for my [pyscrypt](https://github.com/ricmoo/pyscrypt) library. I required a pure-Python AES implementation that supported 256-bit keys with the counter (CTR) mode of operation. After searching, I found several implementations, but all were missing CTR or only supported 128 bit keys. After all the work of learning AES inside and out to implement the library, it was only a marginal amount of extra work to library-ify a more general solution. So, *why not?*

#### How do I get a question I have added?

E-mail me at pyaes@ricmoo.com with any questions, suggestions, comments, et cetera.


#### Can I give you my money?

Umm... Ok? :-)

_Bitcoin_  - `18UDs4qV1shu2CgTS2tKojhCtM69kpnWg9`


# subtl

## Overview

SUBTL is a **s**imple **U**DP **B**itTorrent **t**racker **l**ibrary for Python, licenced under the modified BSD license.

## Example

This short example will list a few IP Addresses from a certain hash:

    from subtl import UdpTrackerClient
    utc = UdpTrackerClient('tracker.openbittorrent.com', 80)
    utc.connect()
    if not utc.poll_once():
        raise Exception('Could not connect')
    print('Success!')

    utc.announce(info_hash='089184ED52AA37F71801391C451C5D5ADD0D9501')
    data = utc.poll_once()
    if not data:
        raise Exception('Could not announce')
    for a in data['response']['peers']:
        print(a)

## Caveats

 * There is no automatic retrying of sending packets yet.
 * This library won't download torrent files--it is simply a tracker client.


# CoffeeScript compiler for Windows

A simple command-line utilty for Windows that will compile `*.coffee` files to JavaScript `*.js` files using [CoffeeScript](http://jashkenas.github.com/coffee-script/) and the venerable Windows Script Host, ubiquitous on Windows since the 90s.

## Usage

To use it, invoke `coffee.cmd` like so:

    coffee input.coffee output.js
    
If an output is not specified, it is written to `stdout`. In neither an input or output are specified then data is assumed to be on `stdin`. For example:

    type input.coffee | coffee > output.js

Errors are written to `stderr`.

In the `test` directory there's a version of the standard CoffeeScript tests which can be kicked off using `test.cmd`. The test just attempts to compile the *.coffee files but doesn't execute them.

To upgrade to the latest CoffeeScript simply replace `coffee-script.js` from the upstream https://github.com/jashkenas/coffee-script/blob/master/extras/coffee-script.js (the tests will likely need updating as well, if you want to run them).
