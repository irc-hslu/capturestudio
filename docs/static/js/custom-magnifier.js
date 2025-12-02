window.CustomMagnifier = (function () {
    var zoomFactor = 2.5;
    var lensSize = 180;
    var hrStore = {};

    function preloadHR(src, callback) {
        if (!src) {
            return;
        }
        var entry = hrStore[src];
        if (entry && entry.loaded) {
            if (callback) {
                callback();
            }
            return;
        }
        if (entry && !entry.loaded) {
            if (callback) {
                entry.callbacks.push(callback);
            }
            return;
        }
        hrStore[src] = {
            loaded: false,
            callbacks: callback ? [callback] : []
        };
        var img = new Image();
        img.onload = function () {
            var e = hrStore[src];
            if (!e) {
                return;
            }
            e.loaded = true;
            var cbs = e.callbacks;
            for (var i = 0; i < cbs.length; i++) {
                if (cbs[i]) {
                    cbs[i]();
                }
            }
        };
        img.src = src;
    }

    function init(img, hrSrc) {
        if (!img || !hrSrc) {
            return;
        }

        var wrapper = img.closest(".magnifier-thumb-wrapper") || img.parentNode;
        if (!wrapper) {
            return;
        }

        if (wrapper.classList.contains("cm-magnifier-container")) {
            return;
        }

        wrapper.classList.add("cm-magnifier-container");

        var toggle = document.createElement("button");
        toggle.type = "button";
        toggle.className = "cm-magnifier-toggle";
        toggle.setAttribute("aria-hidden", "true");
        toggle.innerHTML = '<i class="fas fa-search-plus"></i>';
        wrapper.appendChild(toggle);

        var lens = document.createElement("div");
        lens.className = "cm-magnifier-lens";
        wrapper.appendChild(lens);

        var state = {
            rect: null,
            zoom: zoomFactor
        };

        preloadHR(hrSrc, function () {
            lens.style.backgroundImage = "url(" + hrSrc + ")";
        });

        function updateRect() {
            state.rect = wrapper.getBoundingClientRect();
        }

        function enter(e) {
            updateRect();
            if (!state.rect || state.rect.width === 0 || state.rect.height === 0) {
                return;
            }
            var w = state.rect.width;
            var h = state.rect.height;
            lens.style.backgroundSize = w * state.zoom + "px " + h * state.zoom + "px";
            lens.style.display = "block";
            move(e);
        }

        function move(e) {
            if (!state.rect) {
                return;
            }
            var rect = state.rect;
            var r = lensSize / 2;
            var x = e.clientX - rect.left;
            var y = e.clientY - rect.top;

            lens.style.left = x - r + "px";
            lens.style.top = y - r + "px";

            var xClamped = x;
            if (xClamped < 0) xClamped = 0;
            if (xClamped > rect.width) xClamped = rect.width;
            var yClamped = y;
            if (yClamped < 0) yClamped = 0;
            if (yClamped > rect.height) yClamped = rect.height;

            var bgW = rect.width * state.zoom;
            var bgH = rect.height * state.zoom;

            var bgX = -((xClamped / rect.width) * bgW - r);
            var bgY = -((yClamped / rect.height) * bgH - r);

            lens.style.backgroundPosition = bgX + "px " + bgY + "px";
        }

        function leave() {
            lens.style.display = "none";
        }

        wrapper.addEventListener("mouseenter", function (e) {
            enter(e);
        });

        wrapper.addEventListener("mousemove", function (e) {
            if (!state.rect) {
                updateRect();
                if (!state.rect) {
                    return;
                }
            }
            move(e);
        });

        wrapper.addEventListener("mouseleave", function () {
            leave();
        });

        window.addEventListener("resize", function () {
            if (state.rect) {
                updateRect();
            }
        });
    }

    return {
        init: init
    };
})();
