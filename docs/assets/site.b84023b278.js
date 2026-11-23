/* between-the-cuffs: turns the plain case list into a viewer. Without this script every case is a
   figure in a long list. No network requests; works from a local copy. */
(function () {
  "use strict";
  var doc = document;
  doc.documentElement.classList.add("js");
  var each = function (sel, fn, root) { Array.prototype.forEach.call((root || doc).querySelectorAll(sel), fn); };
  each("[data-js-only]", function (el) { el.hidden = false; });

  if (navigator.clipboard && window.isSecureContext) {
    each(".copy", function (b) {
      b.hidden = false;
      b.addEventListener("click", function () {
        navigator.clipboard.writeText(b.getAttribute("data-copy")).then(function () {
          b.textContent = "Copied";
          setTimeout(function () { b.textContent = "Copy"; }, 1500);
        });
      });
    });
  }

  var sheet = doc.getElementById("sheet");
  var viewer = doc.getElementById("viewer");
  if (!sheet || !viewer) { return; }
  var list = viewer.querySelector(".case-list");
  var pos = viewer.querySelector(".viewer-pos");
  var byId = {};
  each(".case", function (c) { byId[c.getAttribute("data-case")] = c; }, list);
  var tiles = Array.prototype.slice.call(sheet.querySelectorAll(".tile"));
  var current = null;
  var replayEl = doc.getElementById("replay");
  var rule1 = replayEl ? replayEl.closest(".case").getAttribute("data-case") : null;

  function order() {
    return tiles.filter(function (t) { return !t.hidden && byId[t.getAttribute("data-case")]; })
      .map(function (t) { return t.getAttribute("data-case"); });
  }
  function pauseAll() { each("video", function (v) { v.pause(); }, list); }
  function setMode(art, mode) {
    art.classList.toggle("mode-replay", mode === "replay");
    each("[data-mode]", function (b) {
      b.setAttribute("aria-pressed", String(b.getAttribute("data-mode") === mode));
    }, art);
    if (mode !== "replay") { pauseAll(); }
  }
  function select(id, opt) {
    opt = opt || {};
    var art = byId[id];
    if (!art) { return; }
    if (current && current !== art) {
      setMode(current, "still");
      current.classList.remove("is-current");
    }
    current = art;
    art.classList.add("is-current");
    tiles.forEach(function (t) {
      var on = t.getAttribute("data-case") === id;
      t.classList.toggle("is-current", on);
      var a = t.querySelector("a");
      if (a) { if (on) { a.setAttribute("aria-current", "true"); } else { a.removeAttribute("aria-current"); } }
    });
    if (opt.mode) { setMode(art, opt.mode); }
    var o = order();
    var i = o.indexOf(id);
    if (pos) { pos.textContent = i >= 0 ? "case " + (i + 1) + " of " + o.length : ""; }
    if (opt.hash && history.replaceState) { history.replaceState(null, "", "#case-" + id); }
    if (opt.scroll) { viewer.scrollIntoView({ block: "start" }); }
    if (opt.focus) { viewer.focus({ preventScroll: true }); }
  }
  function step(d) {
    var o = order();
    if (!o.length) { return; }
    var i = current ? o.indexOf(current.getAttribute("data-case")) : -1;
    select(o[(i + d + o.length) % o.length], { hash: true });
  }

  sheet.addEventListener("click", function (ev) {
    var a = ev.target.closest(".tile-btn");
    if (!a) { return; }
    ev.preventDefault();
    select(a.closest(".tile").getAttribute("data-case"), { scroll: true, hash: true, focus: true });
  });
  each("[data-nav]", function (b) {
    b.addEventListener("click", function () { step(Number(b.getAttribute("data-nav"))); });
  }, viewer);
  viewer.addEventListener("keydown", function (ev) {
    if (ev.target.closest && ev.target.closest("video")) { return; }
    if (ev.key === "ArrowLeft") { step(-1); ev.preventDefault(); }
    if (ev.key === "ArrowRight") { step(1); ev.preventDefault(); }
  });
  list.addEventListener("click", function (ev) {
    var b = ev.target.closest("[data-mode]");
    if (b) { setMode(b.closest(".case"), b.getAttribute("data-mode")); }
  });

  function press(group, btn) {
    each("button", function (x) { x.setAttribute("aria-pressed", String(x === btn)); }, group);
  }
  each("[data-sort]", function (b) {
    b.addEventListener("click", function () {
      var key = b.getAttribute("data-sort");
      tiles.sort(function (x, y) {
        var ix = Number(x.getAttribute("data-case")), iy = Number(y.getAttribute("data-case"));
        if (key === "id") { return ix - iy; }
        var d = parseFloat(x.getAttribute("data-err")) - parseFloat(y.getAttribute("data-err"));
        return d || ix - iy;
      });
      tiles.forEach(function (t) { sheet.appendChild(t); });
      press(b.parentNode, b);
      if (current) { select(current.getAttribute("data-case")); }
    });
  });
  each("[data-filter]", function (b) {
    b.addEventListener("click", function () {
      var f = b.getAttribute("data-filter");
      tiles.forEach(function (t) { t.hidden = !(f === "all" || t.getAttribute("data-stratum") === f); });
      press(b.parentNode, b);
      var o = order();
      if (o.length && (!current || o.indexOf(current.getAttribute("data-case")) < 0)) { select(o[0]); }
      else if (current) { select(current.getAttribute("data-case")); }
    });
  });

  function route() {
    var h = location.hash;
    if (h === "#replay" && rule1) { select(rule1, { mode: "replay", scroll: true }); return true; }
    var m = /^#case-(\d+)$/.exec(h);
    if (m && byId[m[1]]) { select(m[1], { scroll: true }); return true; }
    return false;
  }
  window.addEventListener("hashchange", route);
  if (!route()) { select(rule1 || order()[0]); }
})();
