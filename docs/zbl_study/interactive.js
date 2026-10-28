/* Interactive demos for zbl_study.html (vanilla JS, inline SVG plots).
   Each demo mounts into <div class="demo" data-demo="NAME">. Units: eV, Å, amu, fs. */
(function () {
  "use strict";

  // ------------------------------------------------------------------ physics
  const KE2 = 14.399645, TU = 10.1805;
  const ZC = [0.02817, 0.28022, 0.50986, 0.18175], ZD = [0.20162, 0.40290, 0.94229, 3.19980];
  const MASS = { 1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998, 14: 28.0855 };
  const SYM = { 1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 14: "Si" };

  function zbl(r, zi, zj) {
    const ai = (Math.pow(zi, 0.23) + Math.pow(zj, 0.23)) / 0.4685, k = zi * zj * KE2;
    let e = 0, ep = 0, epp = 0;
    for (let n = 0; n < 4; n++) {
      const d = ZD[n] * ai, x = ZC[n] * Math.exp(-d * r);
      e += x; ep -= d * x; epp += d * d * x;
    }
    return [k * e / r, k * (ep - e / r) / r, k * (epp - 2 * ep / r + 2 * e / (r * r)) / r];
  }
  function swCoeff(zi, zj, r1, rc) {
    const tc = rc - r1, [fc, fp, fpp] = zbl(rc, zi, zj);
    return [(-3 * fp + tc * fpp) / (tc * tc), (2 * fp - tc * fpp) / (tc * tc * tc),
            -fc + tc / 2 * fp - tc * tc / 12 * fpp];
  }
  function zblSw(r, zi, zj, r1, rc, co) {
    if (r >= rc) return [0, 0, 0];
    const [A, B, C] = co || swCoeff(zi, zj, r1, rc);
    let [E, dE, d2] = zbl(r, zi, zj);
    const t = Math.max(r - r1, 0);
    return [E + C + A / 3 * t ** 3 + B / 4 * t ** 4, dE + A * t * t + B * t ** 3, d2 + 2 * A * t + 3 * B * t * t];
  }
  function zblMult(r, zi, zj, r1, rc) {
    const [E, dE] = zbl(r, zi, zj);
    const x = Math.min(Math.max((r - r1) / (rc - r1), 0), 1);
    const f = 1 - (10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5);
    const df = -(30 * x * x - 60 * x ** 3 + 30 * x ** 4) / (rc - r1);
    return [f * E, f * dE + df * E];
  }
  const morse = (r) => { const x = Math.exp(-1.6 * (r - 2.35)); return [3 * (x * x - 2 * x), 3 * (-3.2 * x * x + 3.2 * x)]; };
  const fermi = (r) => { const f = 1 / (1 + Math.exp(-14 * (r - 1.6))); return [f, 14 * f * (1 - f)]; };
  const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));
  const logspace = (a, b, n) => linspace(a, b, n).map((v) => Math.pow(10, v));
  const fmt = (v, d = 2) => (Math.abs(v) >= 1e4 ? v.toExponential(1) : v.toFixed(d));
  function rng(seed) { // mulberry32
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) { let u = 0, v = 0; while (u === 0) u = r(); v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

  // ------------------------------------------------------------------ plotting
  const NS = "http://www.w3.org/2000/svg";
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function niceTicks(a, b, n) {
    const span = b - a, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
    const out = [];
    for (let v = Math.ceil(a / step) * step; v <= b + 1e-9 * step; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function plot(svg, o) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const cw = (svg.parentNode && svg.parentNode.clientWidth) || 640;
    const W = Math.max(360, Math.min(640, cw)), H = Math.round((o.height || 300) * (W < 520 ? 1.15 : 1)), L = 58, R = 14, T = o.title ? 26 : 12, B = 40;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("class", "plot");
    const lx = !!o.logx, ly = !!o.logy;
    const tx = (v) => (lx ? Math.log10(v) : v), ty = (v) => (ly ? Math.log10(v) : v);
    const [x0, x1] = o.xlim.map(tx), [y0, y1] = o.ylim.map(ty);
    const X = (v) => L + (tx(v) - x0) / (x1 - x0) * (W - L - R);
    const Y = (v) => T + (1 - (ty(v) - y0) / (y1 - y0)) * (H - T - B);
    const id = "c" + Math.random().toString(36).slice(2);
    const defs = el("defs", {}, svg), cp = el("clipPath", { id }, defs);
    el("rect", { x: L, y: T, width: W - L - R, height: H - T - B }, cp);
    if (o.title) el("text", { x: L, y: 17, class: "ptitle" }, svg).textContent = o.title;
    (o.bands || []).forEach((b) => el("rect", { x: X(b[0]), y: T, width: Math.max(X(b[1]) - X(b[0]), 0), height: H - T - B, class: b[2] || "band" }, svg));
    (o.hbands || []).forEach((b) => el("rect", { x: L, y: Y(b[1]), width: W - L - R, height: Math.max(Y(b[0]) - Y(b[1]), 0), class: b[2] || "band" }, svg));
    const ticks = (lo, hi, log) => (log ? Array.from({ length: Math.floor(hi) - Math.ceil(lo) + 1 }, (_, i) => Math.pow(10, Math.ceil(lo) + i)) : niceTicks(lo, hi, 6));
    ticks(x0, x1, lx).forEach((v) => {
      el("line", { x1: X(v), x2: X(v), y1: T, y2: H - B, class: "grid" }, svg);
      el("text", { x: X(v), y: H - B + 15, class: "tick", "text-anchor": "middle" }, svg).textContent = lx ? (v >= 1 ? v : v.toPrecision(1)) : +v.toPrecision(6);
    });
    ticks(y0, y1, ly).forEach((v) => {
      el("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), class: "grid" }, svg);
      el("text", { x: L - 6, y: Y(v) + 4, class: "tick", "text-anchor": "end" }, svg).textContent = ly ? (v >= 1e4 || v < 1e-2 ? v.toExponential(0) : v) : +v.toPrecision(6);
    });
    if (!ly && y0 < 0 && y1 > 0) el("line", { x1: L, x2: W - R, y1: Y(0), y2: Y(0), class: "zero" }, svg);
    el("rect", { x: L, y: T, width: W - L - R, height: H - T - B, class: "frame" }, svg);
    const g = el("g", { "clip-path": `url(#${id})` }, svg);
    (o.vlines || []).forEach((v) => el("line", { x1: X(v[0]), x2: X(v[0]), y1: T, y2: H - B, class: v[1] || "vline" }, g));
    (o.series || []).forEach((s) => {
      let d = "", pen = false;
      for (let i = 0; i < s.x.length; i++) {
        const xv = s.x[i], yv = s.y[i];
        if (!isFinite(yv) || (ly && yv <= 0) || (lx && xv <= 0)) { pen = false; continue; }
        const py = Math.max(Math.min(Y(yv), H + 2000), -2000);
        d += (pen ? "L" : "M") + X(xv).toFixed(1) + "," + py.toFixed(1);
        pen = true;
      }
      el("path", { d, class: "line " + (s.cls || "s1") + (s.dash ? " dash" : "") }, g);
    });
    (o.points || []).forEach((p) => el("circle", { cx: X(p[0]), cy: Y(p[1]), r: p[3] || 4.5, class: "pt " + (p[2] || "s1") }, g));
    el("text", { x: (L + W - R) / 2, y: H - 6, class: "axlabel", "text-anchor": "middle" }, svg).textContent = o.xlabel || "";
    const yl = el("text", { x: 14, y: (T + H - B) / 2, class: "axlabel", "text-anchor": "middle", transform: `rotate(-90 14 ${(T + H - B) / 2})` }, svg);
    yl.textContent = o.ylabel || "";
    const leg = (o.series || []).filter((s) => s.label);
    leg.forEach((s, i) => {
      const lxp = o.legendLeft || W < 520 ? L + 10 : W - R - 210;
      const lyp = o.legendBottom ? H - B - 8 - (leg.length - 1 - i) * 16 : T + 14 + i * 16;
      el("line", { x1: lxp, x2: lxp + 20, y1: lyp - 4, y2: lyp - 4, class: "line " + (s.cls || "s1") + (s.dash ? " dash" : "") }, svg);
      el("text", { x: lxp + 26, y: lyp, class: "legend" }, svg).textContent = s.label;
    });
    return { X, Y };
  }
  function bars(svg, o) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const W = Math.max(360, Math.min(640, (svg.parentNode && svg.parentNode.clientWidth) || 640)), H = o.height || 260, L = 58, R = 14, T = 12, B = 40;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("class", "plot");
    const [y0, y1] = o.ylim.map(Math.log10);
    const Y = (v) => T + (1 - (Math.log10(Math.max(v, Math.pow(10, y0))) - y0) / (y1 - y0)) * (H - T - B);
    for (let e = Math.ceil(y0); e <= y1; e++) {
      el("line", { x1: L, x2: W - R, y1: Y(10 ** e), y2: Y(10 ** e), class: "grid" }, svg);
      el("text", { x: L - 6, y: Y(10 ** e) + 4, class: "tick", "text-anchor": "end" }, svg).textContent = 10 ** e >= 1e4 || 10 ** e < 1e-2 ? (10 ** e).toExponential(0) : 10 ** e;
    }
    const n = o.labels.length, gw = (W - L - R) / n, bw = gw / (o.groups.length + 1);
    o.labels.forEach((lab, i) => {
      o.groups.forEach((gr, k) => {
        const v = gr.values[i], x = L + i * gw + bw * (k + 0.5);
        el("rect", { x, y: Y(v), width: bw * 0.9, height: H - B - Y(v), class: "bar " + gr.cls }, svg);
      });
      el("text", { x: L + i * gw + gw / 2, y: H - B + 15, class: "tick", "text-anchor": "middle" }, svg).textContent = lab;
    });
    el("rect", { x: L, y: T, width: W - L - R, height: H - T - B, class: "frame" }, svg);
    o.groups.forEach((gr, k) => {
      el("rect", { x: W - R - 230, y: T + 6 + 16 * k, width: 12, height: 10, class: "bar " + gr.cls }, svg);
      el("text", { x: W - R - 212, y: T + 15 + 16 * k, class: "legend" }, svg).textContent = gr.label;
    });
    el("text", { x: (L + W - R) / 2, y: H - 6, class: "axlabel", "text-anchor": "middle" }, svg).textContent = o.xlabel || "";
    el("text", { x: 14, y: (T + H - B) / 2, class: "axlabel", "text-anchor": "middle", transform: `rotate(-90 14 ${(T + H - B) / 2})` }, svg).textContent = o.ylabel || "";
  }

  // ------------------------------------------------------------------ UI helpers
  function ui(root, spec) {
    // spec: [{key, type:'range'|'select'|'check', label, min,max,step,value, options:[[v,label]]}]
    const box = document.createElement("div");
    box.className = "controls";
    const state = {}, out = {};
    spec.forEach((s) => {
      const lab = document.createElement("label");
      lab.className = "ctl-" + (s.type || "range");
      const name = document.createElement("span");
      name.textContent = s.label;
      lab.appendChild(name);
      let inp;
      if (s.type === "select") {
        inp = document.createElement("select");
        s.options.forEach(([v, t]) => { const op = document.createElement("option"); op.value = v; op.textContent = t; inp.appendChild(op); });
        inp.value = s.value;
      } else if (s.type === "check") {
        inp = document.createElement("input"); inp.type = "checkbox"; inp.checked = !!s.value;
      } else {
        inp = document.createElement("input"); inp.type = "range";
        inp.min = s.min; inp.max = s.max; inp.step = s.step; inp.value = s.value;
      }
      lab.appendChild(inp);
      if (s.type === "range") {
        const v = document.createElement("output");
        lab.appendChild(v);
        out[s.key] = v;
      }
      const read = () => {
        state[s.key] = s.type === "check" ? inp.checked : s.type === "select" ? inp.value : parseFloat(inp.value);
        if (out[s.key]) out[s.key].textContent = (s.show ? s.show(state[s.key]) : state[s.key]) + (s.unit || "");
      };
      read();
      inp.addEventListener("input", () => { read(); root._redraw(); });
      state["_" + s.key] = inp;
      box.appendChild(lab);
    });
    root.appendChild(box);
    return state;
  }
  function svgBox(root, n, cls) {
    const wrap = document.createElement("div");
    wrap.className = "plots" + (n > 1 ? " two" : "") + (cls ? " " + cls : "");
    const out = [];
    for (let i = 0; i < n; i++) out.push(el("svg", {}, wrap));
    root.appendChild(wrap);
    return out;
  }
  function readout(root) {
    const d = document.createElement("div");
    d.className = "readout";
    root.appendChild(d);
    return d;
  }
  const PAIRS = { "Si-Si": [14, 14, 1.598, 2.054], "Si-F": [14, 9, 1.145, 1.472], "Si-O": [14, 8, 1.07, 1.376],
    "C-F": [6, 9, 0.908, 1.168], "F-F": [9, 9, 0.997, 1.282], "N-N": [7, 7, 0.78, 1.002], "H-H": [1, 1, 0.525, 0.675] };

  // ================================================================== demos
  const DEMOS = {};

  // 1. switching shapes ------------------------------------------------
  DEMOS.switch = function (root) {
    const st = ui(root, [
      { key: "pair", type: "select", label: "pair", value: "Si-Si", options: Object.keys(PAIRS).map((k) => [k, k]) },
      { key: "r1", type: "range", label: "inner / outer", min: 0.40, max: 0.98, step: 0.01, value: 0.78, show: (v) => v.toFixed(2), unit: " (inner = ratio × outer)" },
      { key: "rc", type: "range", label: "outer (Å)", min: 0.5, max: 2.6, step: 0.01, value: 2.054, show: (v) => v.toFixed(3) },
    ]);
    st._pair.addEventListener("input", () => { st._rc.value = PAIRS[st.pair][3]; st._rc.dispatchEvent(new Event("input")); });
    const [s1, s2] = svgBox(root, 2), ro = readout(root);
    root._redraw = function () {
      const [zi, zj] = PAIRS[st.pair], rc = st.rc, r1 = st.r1 * rc;
      const co = swCoeff(zi, zj, r1, rc);
      const r = linspace(Math.max(0.35 * rc, 0.15), 1.25 * rc, 400);
      const Ez = r.map((x) => zbl(x, zi, zj)), Es = r.map((x) => zblSw(x, zi, zj, r1, rc, co)), Em = r.map((x) => zblMult(x, zi, zj, r1, rc));
      const ymax = zbl(0.62 * rc, zi, zj)[0];
      plot(s1, { xlim: [r[0], r[r.length - 1]], ylim: [co[2] * 0.4 - 0.05 * ymax, ymax], bands: [[r1, rc]], xlabel: "r (Å)", ylabel: "pair energy (eV)", title: "energy",
        series: [{ x: r, y: Ez.map((v) => v[0]), cls: "s1", label: "bare ZBL" }, { x: r, y: Es.map((v) => v[0]), cls: "s2", label: "ZBL + S(r) (additive)" },
                 { x: r, y: Em.map((v) => v[0]), cls: "s3", dash: 1, label: "f_c·ZBL (multiplicative)" }] });
      const fmax = -zbl(0.62 * rc, zi, zj)[1];
      plot(s2, { xlim: [r[0], r[r.length - 1]], ylim: [-0.05 * fmax, fmax], bands: [[r1, rc]], xlabel: "r (Å)", ylabel: "force −dE/dr (eV/Å)", title: "force",
        series: [{ x: r, y: Ez.map((v) => -v[1]), cls: "s1" }, { x: r, y: Es.map((v) => -v[1]), cls: "s2" }, { x: r, y: Em.map((v) => -v[1]), cls: "s3", dash: 1 }] });
      let kmax = 0;
      linspace(r1, rc, 300).forEach((x) => { kmax = Math.max(kmax, Math.abs(zblSw(x, zi, zj, r1, rc, co)[2])); });
      ro.innerHTML = `inner = <b>${r1.toFixed(3)} Å</b>, outer = <b>${rc.toFixed(3)} Å</b> · constant shift <b>C = ${co[2].toFixed(2)} eV</b>` +
        ` = −[V<sub>ZBL</sub>(r<sub>c</sub>) ${zbl(rc, zi, zj)[0].toFixed(2)} eV + work removed ${(-co[2] - zbl(rc, zi, zj)[0]).toFixed(2)} eV] · ` +
        `max |dF/dr| in window = <b>${kmax.toFixed(0)} eV/Å²</b>`;
    };
    root._redraw();
  };

  // 2. NVE ----------------------------------------------------------------
  DEMOS.nve = function (root) {
    const st = ui(root, [
      { key: "pair", type: "select", label: "pair", value: "Si-Si", options: Object.keys(PAIRS).map((k) => [k, k]) },
      { key: "method", type: "select", label: "coupling", value: "all", options: [["all", "compare all 5"], ["sw", "(a) ZBL + S(r)"], ["hard", "(b) hard cutoff"],
        ["fok", "(c) Fermi blend, full gradient"], ["fbad", "(d) Fermi blend, missing f′ term"], ["vel", "(e) velocity-dependent on/off"]] },
      { key: "ecm", type: "range", label: "E_cm (eV)", min: 5, max: 200, step: 1, value: 30 },
      { key: "dt", type: "range", label: "dt (fs)", min: 0.05, max: 1.5, step: 0.05, value: 1.0, show: (v) => v.toFixed(2) },
      { key: "ratio", type: "range", label: "inner / outer", min: 0.5, max: 0.97, step: 0.01, value: 0.78, show: (v) => v.toFixed(2) },
    ]);
    const [s1] = svgBox(root, 1), ro = readout(root);
    const names = { sw: "(a) ZBL+S", hard: "(b) hard cutoff", fok: "(c) Fermi, full", fbad: "(d) Fermi, no f′", vel: "(e) velocity on/off" };
    const cls = { sw: "s2", hard: "s6", fok: "s3", fbad: "s4", vel: "s5" };
    function run(method, zi, zj, r1, rc, Ecm, dt) {
      const mu = MASS[zi] * MASS[zj] / (MASS[zi] + MASS[zj]), co = swCoeff(zi, zj, r1, rc);
      const fe = (r, v) => {
        if (method === "sw") { const [E, dE] = zblSw(r, zi, zj, r1, rc, co); return [-dE, E]; }
        if (method === "hard") { if (r >= rc) return [0, 0]; const [E, dE] = zbl(r, zi, zj); return [-dE, E]; }
        const [f, df] = fermi(r), [Ez, dEz] = zbl(r, zi, zj), [Em, dEm] = morse(r);
        if (method === "fok") return [-((1 - f) * dEz + f * dEm + df * (Em - Ez)), (1 - f) * Ez + f * Em];
        if (method === "fbad") return [-((1 - f) * dEz + f * dEm), (1 - f) * Ez + f * Em];
        const on = 0.5 * mu * v * v > 5;
        const [E, dE] = on ? zblSw(r, zi, zj, r1, rc, co) : [0, 0];
        return [-dE - dEm, E + Em];
      };
      const h = dt / TU; let r = Math.max(3.0, rc + 0.5), v = -Math.sqrt(2 * Ecm / mu), [F, E] = fe(r, v);
      const t = [0], dE = [0], E0 = 0.5 * mu * v * v + E; let maxd = 0;
      const nmax = Math.min(Math.ceil(120 / dt), 20000);
      for (let i = 1; i <= nmax; i++) {
        v += 0.5 * h * F / mu; r += h * v; [F, E] = fe(r, v); v += 0.5 * h * F / mu;
        const d = 0.5 * mu * v * v + E - E0; maxd = Math.max(maxd, Math.abs(d));
        t.push(i * dt); dE.push(d);
      }
      return { t, dE, maxd, final: dE[dE.length - 1] };
    }
    root._redraw = function () {
      const [zi, zj, , rc] = PAIRS[st.pair], r1 = st.ratio * rc;
      const methods = st.method === "all" ? ["sw", "hard", "fok", "fbad", "vel"] : [st.method];
      const res = methods.map((m) => [m, run(m, zi, zj, r1, rc, st.ecm, st.dt)]);
      let ym = 0.5; res.forEach(([, x]) => { ym = Math.max(ym, Math.min(x.maxd * 1.15, 60)); });
      plot(s1, { xlim: [0, 120], ylim: [-ym, ym], xlabel: "time (fs)", ylabel: "E_total − E_total(0) (eV)", legendLeft: false,
        series: res.map(([m, x]) => ({ x: x.t, y: x.dE, cls: cls[m], label: names[m] })) });
      ro.innerHTML = res.map(([m, x]) => `${names[m]}: max |ΔE| <b>${fmt(x.maxd, 3)}</b>, after collision <b>${fmt(x.final, 3)}</b> eV`).join("<br>") +
        `<br><span class="muted">(c)–(e) blend ZBL with a toy Morse bond (D = 3 eV, r<sub>e</sub> = 2.35 Å) through a Fermi function at 1.6 Å, as in tersoff/zbl. Shorten dt to see which errors are integration error (shrink) and which are not (stay).</span>`;
    };
    root._redraw();
  };

  // 3. tanh bound with the real Si network ---------------------------------
  DEMOS.tanh = function (root) {
    const D = window.DEMO_DATA.folded;
    const A = D.A, c = D.c, W2 = D.W2, b2 = D.b2, W3 = D.W3;
    function G_tetra(r) {
      const rjk = r * Math.sqrt(8 / 3), fc = (d, rc) => (d < rc ? 0.5 * (1 + Math.cos(Math.PI * d / rc)) : 0);
      return D.sym.map((s) => {
        const [st, rc, eta] = s;
        if (st === 2) return s[5] === "O" ? 4 * Math.exp(-eta * (r - s[3]) ** 2) * fc(r, rc) : 0;
        if (!(s[5] === "O" && s[6] === "O")) return 0;
        const ang = Math.pow(2, 1 - s[3]) * Math.pow(Math.abs(1 - s[4] / 3), s[3]);
        return st === 4 ? 6 * ang * Math.exp(-eta * (2 * r * r + rjk * rjk)) * fc(r, rc) ** 2 * fc(rjk, rc)
                        : 6 * ang * Math.exp(-eta * 2 * r * r) * fc(r, rc) ** 2;
      });
    }
    function net(G) {
      const h1 = A.map((row, i) => Math.tanh(row.reduce((s, w, j) => s + w * G[j], c[i])));
      const h2 = W2.map((row, i) => Math.tanh(row.reduce((s, w, j) => s + w * h1[j], b2[i])));
      return { E: h2.reduce((s, h, i) => s + W3[i] * h, D.b3), h1, h2 };
    }
    const st = ui(root, [{ key: "r", type: "range", label: "Si–O distance (Å)", min: 0.05, max: 4.0, step: 0.01, value: 1.61, show: (v) => v.toFixed(2) }]);
    const [s1] = svgBox(root, 1);
    const strip = document.createElement("div"); strip.className = "strip"; root.appendChild(strip);
    const ro = readout(root);
    const rr = linspace(0.05, 4.0, 160), curve = rr.map((r) => net(G_tetra(r)).E);
    const cell = (h) => `<span title="${h.toFixed(3)}" style="background:${h >= 0 ? `rgba(234,88,12,${Math.abs(h).toFixed(3)})` : `rgba(37,99,235,${Math.abs(h).toFixed(3)})`}"></span>`;
    root._redraw = function () {
      const o = net(G_tetra(st.r));
      plot(s1, { xlim: [0, 4], ylim: [D.b3 - 1.4 * D.bound, D.b3 + 1.4 * D.bound], hbands: [[D.b3 - D.bound, D.b3 + D.bound]], bands: [[1.55, 1.70, "band2"]],
        xlabel: "Si–O distance in an SiO4 tetrahedron (Å)", ylabel: "E_Si (eV)",
        series: [{ x: rr, y: curve, cls: "s2", label: "E_Si(r) of the trained net" }, { x: [0, 4], y: [D.b3 + D.bound, D.b3 + D.bound], cls: "s6", dash: 1, label: "b ± Σ|w|  (hard limits)" }, { x: [0, 4], y: [D.b3 - D.bound, D.b3 - D.bound], cls: "s6", dash: 1 }],
        points: [[st.r, o.E, "s2"]] });
      strip.innerHTML = "<div class='strip-label'>hidden layer 1 (60 tanh units)</div><div class='cells'>" + o.h1.map(cell).join("") +
        "</div><div class='strip-label'>hidden layer 2 (60 tanh units) → output = b + Σ w·h</div><div class='cells'>" + o.h2.map(cell).join("") + "</div>";
      const sat = o.h1.filter((h) => Math.abs(h) > 0.99).length;
      ro.innerHTML = `E_Si = <b>${o.E.toFixed(3)} eV</b>; hard limits [${(D.b3 - D.bound).toFixed(2)}, ${(D.b3 + D.bound).toFixed(2)}] eV (b = ${D.b3.toFixed(3)}, Σ|w| = ${D.bound.toFixed(3)}). ` +
        `Saturated units in layer 1: <b>${sat}/60</b>. Colour: −1 (blue) … +1 (orange).`;
    };
    root._redraw();
  };

  // 4. real BPNN dimers ------------------------------------------------------
  DEMOS.dimer = function (root) {
    const D = window.DEMO_DATA.dimers, r = D.r;
    const pots = Object.keys(D.pots);
    const st = ui(root, [
      { key: "pot", type: "select", label: "potential", value: "SiO2", options: pots.map((p) => [p, p]) },
      { key: "pair", type: "select", label: "pair", value: "Si-Si", options: Object.keys(D.pots.SiO2.pairs).map((k) => [k, k]) },
      { key: "view", type: "select", label: "show", value: "E", options: [["E", "energy"], ["F", "force"], ["ratio", "|F_NN| / F_ZBL"]] },
    ]);
    st._pot.addEventListener("input", () => {
      const keys = Object.keys(D.pots[st.pot].pairs);
      st._pair.innerHTML = keys.map((k) => `<option value="${k}">${k}</option>`).join("");
      st._pair.value = keys[0]; st._pair.dispatchEvent(new Event("input"));
    });
    const [s1] = svgBox(root, 1), ro = readout(root);
    root._redraw = function () {
      const p = D.pots[st.pot].pairs[st.pair] || D.pots[st.pot].pairs[Object.keys(D.pots[st.pot].pairs)[0]];
      const [zi, zj] = p.Z, [r1, rc] = p.cut, co = swCoeff(zi, zj, r1, rc);
      const E = p.E, Es = r.map((x) => zblSw(x, zi, zj, r1, rc, co)), Ez = r.map((x) => zbl(x, zi, zj));
      const F = E.map((_, i) => -(E[Math.min(i + 1, E.length - 1)] - E[Math.max(i - 1, 0)]) / (r[Math.min(i + 1, E.length - 1)] - r[Math.max(i - 1, 0)]));
      const bA = D.pots[st.pot].bound[SYM[zi]], bB = D.pots[st.pot].bound[SYM[zj]];
      if (st.view === "E") {
        plot(s1, { xlim: [0, 4], ylim: [-15, 40], bands: [[r1, rc]], xlabel: "r (Å)", ylabel: "E (eV, vs separated atoms)",
          series: [{ x: r, y: E, cls: "s2", label: "E_NN (dimer)" }, { x: r, y: Es.map((v) => v[0]), cls: "s1", label: "ZBL + S" }, { x: r, y: E.map((v, i) => v + Es[i][0]), cls: "s6", dash: 1, label: "total" }] });
      } else if (st.view === "F") {
        plot(s1, { xlim: [0.1, 4], ylim: [0.01, 1e5], logy: true, bands: [[r1, rc]], xlabel: "r (Å)", ylabel: "|force| (eV/Å)",
          series: [{ x: r, y: F.map(Math.abs), cls: "s2", label: "|F_NN|" }, { x: r, y: Es.map((v) => Math.abs(v[1])), cls: "s1", label: "|F_ZBL+S|" }] });
      } else {
        plot(s1, { xlim: [0.1, r1], ylim: [1e-3, 10], logy: true, xlabel: "r (Å), up to inner", ylabel: "|F_NN| / F_ZBL",
          series: [{ x: r.filter((x) => x <= r1), y: r.filter((x) => x <= r1).map((x, i) => Math.abs(F[i]) / -Ez[i][1]), cls: "s2", label: "ratio" }, { x: [0.1, r1], y: [1, 1], cls: "s6", dash: 1, label: "NN as strong as ZBL" }] });
      }
      let worst = 0, at = 0;
      r.forEach((x, i) => { if (x > 0.2 && x <= r1) { const q = Math.abs(F[i]) / -Ez[i][1]; if (q > worst) { worst = q; at = x; } } });
      const flat = r.findIndex((x) => x >= 0.3);
      ro.innerHTML = `inner/outer ${r1}/${rc} Å · E_NN(r→0) ≈ <b>${E[0].toFixed(2)} eV</b>, |F_NN(r ≤ 0.3)| ≤ <b>${Math.max(...F.slice(0, flat).map(Math.abs)).toFixed(2)} eV/Å</b> · ` +
        `worst |F_NN|/F_ZBL below inner: <b>${worst.toFixed(2)}</b> at ${at.toFixed(2)} Å · tanh limits: |E_${SYM[zi]} − b| ≤ ${bA[0]}, |E_${SYM[zj]} − b| ≤ ${bB[0]} eV`;
    };
    root._redraw();
  };

  // 5. G2 derivative ---------------------------------------------------------
  DEMOS.g2 = function (root) {
    const st = ui(root, [
      { key: "eta", type: "range", label: "η (Å⁻²)", min: 0.003, max: 2, step: 0.001, value: 0.714 },
      { key: "rs", type: "range", label: "R_s (Å)", min: 0, max: 2.5, step: 0.05, value: 0, show: (v) => v.toFixed(2) },
      { key: "rc", type: "range", label: "r_c (Å)", min: 2, max: 8, step: 0.1, value: 6, show: (v) => v.toFixed(1) },
    ]);
    const [s1, s2] = svgBox(root, 2), ro = readout(root);
    root._redraw = function () {
      const { eta, rs, rc } = st, r = linspace(0, Math.min(rc, 4), 300);
      const e = r.map((x) => Math.exp(-eta * (x - rs) ** 2)), f = r.map((x) => 0.5 * (1 + Math.cos(Math.PI * x / rc)));
      const df = r.map((x) => -0.5 * Math.PI / rc * Math.sin(Math.PI * x / rc));
      const t1 = r.map((x, i) => -2 * eta * (x - rs) * e[i] * f[i]), t2 = r.map((_, i) => e[i] * df[i]);
      plot(s1, { xlim: [0, r[r.length - 1]], ylim: [0, 1.05], xlabel: "r (Å)", ylabel: "G2 contribution", title: "exp(−η(r−R_s)²) · f_c(r)",
        series: [{ x: r, y: e.map((v, i) => v * f[i]), cls: "s1", label: "G2" }, { x: r, y: e, cls: "s6", dash: 1, label: "Gaussian" }, { x: r, y: f, cls: "s3", dash: 1, label: "f_c" }] });
      const all = t1.concat(t2).map((v, i) => v + (i < t1.length ? t2[i] : 0));
      const m = Math.max(0.05, ...t1.map(Math.abs), ...t2.map(Math.abs));
      plot(s2, { xlim: [0, r[r.length - 1]], ylim: [-m * 1.1, m * 1.1], xlabel: "r (Å)", ylabel: "dG2/dr (1/Å)", title: "product rule: term 1 + term 2",
        series: [{ x: r, y: t1, cls: "s2", label: "−2η(r−R_s)·e·f_c" }, { x: r, y: t2, cls: "s3", label: "e·f_c′" }, { x: r, y: t1.map((v, i) => v + t2[i]), cls: "s6", dash: 1, label: "sum" }] });
      const lim = 2 * eta * rs * Math.exp(-eta * rs * rs);
      ro.innerHTML = `At r → 0: f_c′(0) = 0 and term 1 → 2ηR<sub>s</sub>·e<sup>−ηR<sub>s</sub>²</sup> = <b>${lim.toFixed(4)}</b> Å⁻¹. ` +
        (rs === 0 ? "With R<sub>s</sub> = 0 (all G2 in this repo's potentials) both terms vanish linearly in r: the radial force of the colliding pair goes to zero."
                  : "With R<sub>s</sub> > 0 the slope at r = 0 is finite (bounded), not zero.");
    };
    root._redraw();
  };

  // 6. angular gradient --------------------------------------------------------
  DEMOS.angular = function (root) {
    const st = ui(root, [
      { key: "theta", type: "range", label: "approach angle θ_jik (°)", min: 10, max: 170, step: 1, value: 70 },
      { key: "zeta", type: "range", label: "ζ", min: 1, max: 8, step: 1, value: 4 },
      { key: "lam", type: "select", label: "λ", value: "1", options: [["1", "+1"], ["-1", "−1"]] },
      { key: "eta", type: "range", label: "η (Å⁻²)", min: 0.0003, max: 0.1, step: 0.0001, value: 0.0286, show: (v) => v.toFixed(4) },
    ]);
    const [s1] = svgBox(root, 1), ro = readout(root);
    root._redraw = function () {
      const th = st.theta * Math.PI / 180, zeta = st.zeta, lam = +st.lam, eta = st.eta, rc = 4.5, rik = 2.0;
      const fc = (d) => (d < rc ? 0.5 * (1 + Math.cos(Math.PI * d / rc)) : 0);
      const G = (x, y) => {
        const a = Math.hypot(x, y), c = Math.hypot(x - rik, y), cos = x / a;
        return Math.pow(2, 1 - zeta) * Math.pow(Math.abs(1 + lam * cos), zeta) * Math.exp(-eta * (a * a + rik * rik + c * c)) * fc(a) * fc(rik) * fc(c);
      };
      const r = logspace(-2.5, Math.log10(2), 120), par = [], perp = [], g2 = [];
      r.forEach((d) => {
        const x = d * Math.cos(th), y = d * Math.sin(th), h = 1e-6 * Math.max(d, 1e-3);
        const gx = (G(x + h, y) - G(x - h, y)) / (2 * h), gy = (G(x, y + h) - G(x, y - h)) / (2 * h);
        const ux = Math.cos(th), uy = Math.sin(th), p = gx * ux + gy * uy;
        par.push(Math.abs(p)); perp.push(Math.hypot(gx - p * ux, gy - p * uy));
        g2.push(Math.abs(-2 * 0.714 * d * Math.exp(-0.714 * d * d) * 0.5 * (1 + Math.cos(Math.PI * d / 6)) + Math.exp(-0.714 * d * d) * (-0.5 * Math.PI / 6) * Math.sin(Math.PI * d / 6)));
      });
      const k = perp[0] * r[0];
      plot(s1, { xlim: [r[0], 2], ylim: [1e-4, 1e3], logx: true, logy: true, xlabel: "r_ij (Å): neighbour j approaching atom i (k fixed at 2 Å)", ylabel: "|∂G/∂x_j| (1/Å)",
        series: [{ x: r, y: g2, cls: "s1", label: "radial G2 (η=0.714): ∝ r" }, { x: r, y: par, cls: "s3", label: "G4, along r_ij" }, { x: r, y: perp, cls: "s2", label: "G4, ⊥ r_ij" },
                 { x: r, y: r.map((x) => k / x), cls: "s6", dash: 1, label: "∝ 1/r guide" }, { x: r, y: r.map((x) => 0.05 / (x * x)), cls: "s5", dash: 1, label: "∝ 1/r² (ZBL force) guide" }] });
      ro.innerHTML = `cos θ<sub>jik</sub> depends only on the direction of r<sub>ij</sub>; moving j sideways by δ changes the angle by δ/r<sub>ij</sub>, so the ⊥ gradient grows like <b>1/r</b> ` +
        `(here ≈ ${fmt(k, 3)}/r). Along r<sub>ij</sub> the angle does not change and the gradient stays finite. Total NN force = Σ<sub>μ</sub> (∂E/∂G<sub>μ</sub>)(∂G<sub>μ</sub>/∂x): ` +
        `radial and angular parts simply add; ZBL (∝1/r²) still wins as r → 0.`;
    };
    root._redraw();
  };

  // 7. softmin atom-level switch (SevenNet checkpoint) --------------------------
  DEMOS.softmin = function (root) {
    const S = window.DEMO_DATA.sevennet;
    const st = ui(root, [
      { key: "proj", type: "select", label: "projectile", value: "9", options: [["9", "F"], ["14", "Si"], ["8", "O"], ["1", "H"]] },
      { key: "d", type: "range", label: "Si–projectile distance (Å)", min: 0.1, max: 2.4, step: 0.01, value: 0.8, show: (v) => v.toFixed(2) },
      { key: "alpha", type: "range", label: "softmin α", min: 0.01, max: 0.3, step: 0.002, value: S.sw.alpha, show: (v) => v.toFixed(3) },
      { key: "off", type: "range", label: "atomic-energy offset added to E_i^ML (eV)", min: -10, max: 10, step: 0.5, value: 0, show: (v) => v.toFixed(1) },
    ]);
    const [s1, s2] = svgBox(root, 2), ro = readout(root);
    const sc = (a, b) => (Math.pow(a, 0.23) + Math.pow(b, 0.23)) / 0.4685;
    function sw(d, zp, alpha) {
      const rb = [d / sc(14, zp), 1.61 / sc(14, 8), 1.61 / sc(14, 8), 1.61 / sc(14, 8)];
      const CM = 5 / (2 * Math.pow(99, 0.23) / 0.4685), o2 = S.sw.outer ** 2;
      const cv = (x) => (x < S.sw.outer ? 1 : x > CM ? 0 : (CM * CM - x * x) ** 2 * (CM * CM + 2 * x * x - 3 * o2) / (CM * CM - o2) ** 3);
      const w = rb.map((x) => cv(x) * Math.exp(-x / alpha));
      const sigma = rb.reduce((s, x, i) => s + x * w[i], 0) / w.reduce((a, b) => a + b, 0);
      const t = Math.min(Math.max((sigma - S.sw.inner) / (S.sw.outer - S.sw.inner), 0), 1);
      return [sigma, t ** 5 * (70 * t ** 4 - 315 * t ** 3 + 540 * t * t - 420 * t + 126), Math.min(...rb)];
    }
    root._redraw = function () {
      const zp = +st.proj, d = linspace(0.1, 2.4, 300);
      const o = d.map((x) => sw(x, zp, st.alpha));
      const s = sc(14, zp), rin = S.sw.inner * s, rout = S.sw.outer * s;
      plot(s1, { xlim: [0.1, 2.4], ylim: [0, 0.3], hbands: [[S.sw.inner, S.sw.outer]], bands: [[rin, rout, "band2"]], xlabel: "Si–projectile distance (Å)", ylabel: "σ_i (reduced distance)",
        title: "σ_i of the Si atom (3 O at 1.61 Å + projectile)",
        series: [{ x: d, y: o.map((v) => v[0]), cls: "s2", label: "softmin σ_i" }, { x: d, y: o.map((v) => v[2]), cls: "s6", dash: 1, label: "hard min r/s_ij" }], points: [[st.d, sw(st.d, zp, st.alpha)[0], "s2"]] });
      const Eml = S.shift.Si + st.off;           // E_i^ML ≈ species shift near the data (illustration)
      const f = o.map((v) => v[1]);
      const Fsp = f.map((_, i) => -(f[Math.min(i + 1, f.length - 1)] - f[Math.max(i - 1, 0)]) / (d[Math.min(i + 1, f.length - 1)] - d[Math.max(i - 1, 0)]) * Eml);
      const m = Math.max(1, ...Fsp.map(Math.abs));
      plot(s2, { xlim: [0.1, 2.4], ylim: [-m * 1.1, m * 1.1], bands: [[rin, rout, "band2"]], xlabel: "Si–projectile distance (Å)", ylabel: "f  ·  spurious force (eV/Å)", title: "f and the extra force −f′·E_Si^ML",
        series: [{ x: d, y: f.map((v) => v * m), cls: "s3", label: `f(τ) (scaled ×${m.toFixed(1)})` }, { x: d, y: Fsp, cls: "s4", label: "−(df/dr)·E_i^ML" }] });
      const cur = sw(st.d, zp, st.alpha);
      ro.innerHTML = `pair window of the ZBL term (shaded blue): ${rin.toFixed(2)}–${rout.toFixed(2)} Å. At ${st.d.toFixed(2)} Å: σ<sub>i</sub> = <b>${cur[0].toFixed(3)}</b>, f = <b>${cur[1].toFixed(3)}</b>. ` +
        `E<sub>i</sub><sup>ML</sup> here is taken as the Si species shift of the checkpoint (${S.shift.Si} eV) + offset = ${Eml.toFixed(2)} eV: the force −f′E<sub>i</sub> depends on that arbitrary reference — ` +
        `set the offset to ${(-S.shift.Si).toFixed(1)} eV (energy relative to the isolated atom) and it vanishes. With 3 O neighbours the softmin keeps σ high, so the ML switches off only well inside the pair window; smaller α makes σ closer to the hard min.`;
    };
    root._redraw();
  };

  // 8. Bessel radial basis --------------------------------------------------------
  DEMOS.bessel = function (root) {
    const S = window.DEMO_DATA.sevennet;
    const st = ui(root, [
      { key: "kind", type: "select", label: "frequencies", value: "ckpt", options: [["ckpt", "trained (SevenNet checkpoint)"], ["ideal", "initial nπ/r_c"]] },
      { key: "rmin", type: "range", label: "shortest distance in data (Å)", min: 0.6, max: 2.5, step: 0.05, value: 1.5, show: (v) => v.toFixed(2) },
      { key: "r", type: "range", label: "probe r (Å)", min: 0.01, max: 5, step: 0.01, value: 0.3, show: (v) => v.toFixed(2) },
    ]);
    const [s1] = svgBox(root, 1), ro = readout(root);
    root._redraw = function () {
      const fr = st.kind === "ckpt" ? S.bessel_freq : S.bessel_freq.map((_, n) => (n + 1) * Math.PI / S.rc);
      const r = linspace(0.01, 5, 400), B = (x, c) => S.prefactor * Math.sin(c * x) / x;
      const series = fr.map((c, n) => ({ x: r, y: r.map((x) => B(x, c)), cls: "b" + n, label: n === 0 || n === fr.length - 1 ? `n = ${n + 1}` : "" }));
      plot(s1, { xlim: [0, 5], ylim: [-0.6, 2.2], bands: [[st.rmin, 5]], vlines: [[st.r, "vline"]], xlabel: "r (Å)", ylabel: "B_n(r) = 0.4·sin(c_n r)/r", series,
        points: fr.map((c, n) => [0.01, S.prefactor * c, "b" + n, 3.5]) });
      const inData = fr.map((c) => Math.max(...r.filter((x) => x >= st.rmin).map((x) => Math.abs(B(x, c)))));
      const ratio = fr.map((c, n) => Math.abs(B(st.r, c)) / inData[n]);
      ro.innerHTML = `r → 0 limits 0.4·c<sub>n</sub> = [${fr.map((c) => (0.4 * c).toFixed(2)).join(", ")}] (finite; slope → 0 since sin(cr)/r is even). ` +
        `At r = ${st.r.toFixed(2)} Å the basis values are up to <b>${Math.max(...ratio).toFixed(1)}×</b> the largest value seen for r ≥ ${st.rmin.toFixed(2)} Å: the input is finite but out of distribution.`;
    };
    root._redraw();
  };

  // 9. SevenNet checkpoint: dimers and spectator forces (precomputed with torch) ----
  DEMOS.u7dimer = function (root) {
    const D = window.DEMO_DATA.sevennet_scan.dimers, r = D.r;
    const st = ui(root, [{ key: "pair", type: "select", label: "pair", value: "Si-Si", options: Object.keys(D.pairs).map((k) => [k, k]) }]);
    const [s1] = svgBox(root, 1), ro = readout(root);
    root._redraw = function () {
      const p = D.pairs[st.pair], Ez = r.map((x) => zbl(Math.max(x, 1e-3), p.Z[0], p.Z[1])[0]);
      plot(s1, { xlim: [0, 2.5], ylim: [0.1, 2e5], logy: true, bands: [p.window], xlabel: "r (Å)", ylabel: "E (eV, vs separated atoms)",
        series: [{ x: r, y: Ez, cls: "s1", label: "bare ZBL" }, { x: r, y: p.E_ml, cls: "s4", label: "SevenNet raw ML (no switch)" },
                 { x: r, y: p.E_tot, cls: "s6", dash: 1, label: "SevenNet total (switch + ZBL)" }] });
      const lim = D.limit[st.pair];
      ro.innerHTML = `ML energy on the way to r → 0: ` + lim.map(([x, e]) => `${x} Å: <b>${e.toFixed(1)}</b>`).join(" · ") +
        ` eV. It converges because every radial input sin(c<sub>n</sub>r)/r has a finite limit with zero slope; the limit is just the model evaluated at r ≈ 10<sup>−3</sup> Å. ` +
        `Shaded: the checkpoint's switching window ${p.window[0]}–${p.window[1]} Å. Compare bare ZBL at 0.3 Å: ${zbl(0.3, p.Z[0], p.Z[1])[0].toFixed(0)} eV.`;
    };
    root._redraw();
  };
  DEMOS.spill = function (root) {
    const rows = window.DEMO_DATA.sevennet_scan.spill;
    const st = ui(root, [
      { key: "tot", type: "check", label: "SevenNet total", value: true }, { key: "ml", type: "check", label: "SevenNet raw ML", value: true },
      { key: "sw", type: "check", label: "switch-gradient term", value: true }, { key: "bp", type: "check", label: "BPNN Si3N4", value: true },
      { key: "k", type: "range", label: "step", min: 0, max: rows.length - 1, step: 1, value: 7, show: (v) => rows[v].r.toFixed(2) + " Å" },
    ]);
    const [s1] = svgBox(root, 1), ro = readout(root);
    const r = rows.map((x) => x.r);
    root._redraw = function () {
      const ser = [];
      if (st.tot) ser.push({ x: r, y: rows.map((x) => x.dF_tot), cls: "s6", label: "SevenNet total (MD)" });
      if (st.ml) ser.push({ x: r, y: rows.map((x) => x.dF_ml), cls: "s4", label: "SevenNet raw ML" });
      if (st.sw) ser.push({ x: r, y: rows.map((x) => x.F_switch), cls: "s5", dash: 1, label: "−E_i^ML ∇f_i" });
      if (st.bp) ser.push({ x: r, y: rows.map((x) => x.dF_bpnn), cls: "s2", label: "BPNN Si3N4" });
      const k = rows[st.k];
      plot(s1, { xlim: [0.1, 2.4], ylim: [0, 80], vlines: [[k.r, "vline"]], xlabel: "distance of the pushed Si to the centre Si (Å)", ylabel: "max |ΔF| on spectators (eV/Å)", series: ser, legendLeft: true });
      ro.innerHTML = `r = <b>${k.r} Å</b>: SevenNet total ${k.dF_tot}, raw ML ${k.dF_ml}, switch term ${k.F_switch}, BPNN ${k.dF_bpnn} eV/Å. ` +
        `Centre atom: σ = ${(+k.sigma0).toFixed(3)}, f = ${(+k.f0).toFixed(3)}, E<sub>i</sub><sup>ML</sup> = ${k.E_ml0} eV. ` +
        `The 15 spectators never move; any force change on them comes through the model. The peak at 0.5–0.7 Å is the switch: f<sub>i</sub> depends on every neighbour's position through the softmin, ` +
        `so ∂(f<sub>i</sub>E<sub>i</sub>)/∂x<sub>k</sub> = E<sub>i</sub>∂f<sub>i</sub>/∂x<sub>k</sub> + … is large exactly when E<sub>i</sub><sup>ML</sup> has grown to 30–80 eV.`;
    };
    root._redraw();
  };

  // 10. PES softening toy ------------------------------------------------------------
  DEMOS.soft = function (root) {
    const st = ui(root, [
      { key: "frac", type: "range", label: "off-equilibrium data (%)", min: 0, max: 50, step: 1, value: 10 },
      { key: "lam", type: "range", label: "regularisation log10 λ", min: -6, max: 0, step: 0.25, value: -2, show: (v) => v.toFixed(2) },
      { key: "T", type: "range", label: "near-eq. spread (Å)", min: 0.02, max: 0.2, step: 0.01, value: 0.06, show: (v) => v.toFixed(2) },
    ]);
    const [s1] = svgBox(root, 1), ro = readout(root);
    const V = (r) => 3 * (Math.exp(-3.2 * (r - 2.35)) - 2 * Math.exp(-1.6 * (r - 2.35))) + 3;
    const centers = linspace(1.6, 3.4, 16), phi = (r) => centers.map((c) => Math.exp(-(((r - c) / 0.25) ** 2)));
    function solve(A, b) {
      const n = b.length, M = A.map((row, i) => row.concat([b[i]]));
      for (let i = 0; i < n; i++) {
        let p = i; for (let k = i + 1; k < n; k++) if (Math.abs(M[k][i]) > Math.abs(M[p][i])) p = k;
        [M[i], M[p]] = [M[p], M[i]];
        for (let k = i + 1; k < n; k++) { const f = M[k][i] / M[i][i]; for (let j = i; j <= n; j++) M[k][j] -= f * M[i][j]; }
      }
      const x = new Array(n).fill(0);
      for (let i = n - 1; i >= 0; i--) { let s = M[i][n]; for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j]; x[i] = s / M[i][i]; }
      return x;
    }
    root._redraw = function () {
      const R = rng(1), N = 200, nh = Math.round(st.frac / 100 * N), xs = [], ys = [];
      for (let i = 0; i < N; i++) {
        const x = i < N - nh ? 2.35 + gauss(R) * st.T : 1.75 + 1.45 * (i - (N - nh) + R()) / Math.max(nh, 1);
        xs.push(x); ys.push(V(x) + gauss(R) * 0.02);
      }
      const P = xs.map(phi), n = centers.length, lam = Math.pow(10, st.lam) * N;
      const A = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => P.reduce((s, p) => s + p[i] * p[j], 0) + (i === j ? lam : 0)));
      const b = Array.from({ length: n }, (_, i) => P.reduce((s, p, k) => s + p[i] * ys[k], 0));
      const w = solve(A, b), f = (r) => phi(r).reduce((s, p, i) => s + p * w[i], 0);
      const r = linspace(1.6, 3.4, 300);
      plot(s1, { xlim: [1.6, 3.4], ylim: [-0.5, 8], xlabel: "r (Å)", ylabel: "E (eV)",
        series: [{ x: r, y: r.map(V), cls: "s6", label: "true PES (Morse)" }, { x: r, y: r.map(f), cls: "s2", label: "smooth model fitted to the data" }],
        points: xs.map((x, i) => [x, ys[i], "s1", 2]) });
      const slope = (g) => (g(2.2) - g(1.8)) / 0.4;
      ro.innerHTML = `fitted / true slope on 1.8–2.2 Å: <b>${(slope(f) / slope(V)).toFixed(2)}</b>. Same model, same loss: what decides the steepness off equilibrium is how much data is there, ` +
        `and how strongly the fit is pulled toward smooth/small functions where data is thin. Far outside the data no term in the loss asks for steepness at all.`;
    };
    root._redraw();
  };

  // 11. learnability: fits with shrinking r_min (precomputed with torch) -------------
  DEMOS.learn = function (root) {
    const L = window.DEMO_DATA.learn;
    const st = ui(root, [
      { key: "kind", type: "select", label: "features", value: "G2", options: [["G2", "BPNN G2 (SiO2 Si–Si set, Rs = 0)"], ["Bessel", "Bessel (SevenNet checkpoint)"]] },
      { key: "target", type: "select", label: "target", value: "full", options: [["full", "full reference E(r)"], ["delta", "delta: reference − (ZBL + S)"]] },
      { key: "k", type: "range", label: "training r_min", min: 0, max: L.rmins.length - 1, step: 1, value: 8, show: (v) => L.rmins[v] + " Å" },
      { key: "view", type: "select", label: "show", value: "F", options: [["F", "force error vs r (with bound)"], ["E", "energy fit"], ["sum", "RMSE vs r_min (all fits)"]] },
    ]);
    const [s1] = svgBox(root, 1), ro = readout(root);
    root._redraw = function () {
      const rm = L.rmins[st.k], f = L.fits[`${st.kind}|${st.target}|${rm}`];
      if (st.view === "sum") {
        const ser = [];
        [["G2", "full", "s2"], ["Bessel", "full", "s4"], ["G2", "delta", "s3"], ["Bessel", "delta", "s5"]].forEach(([k, t, c]) => {
          ser.push({ x: L.rmins, y: L.rmins.map((x) => L.fits[`${k}|${t}|${x}`].rmse_F), cls: c, dash: t === "delta" ? 1 : 0, label: `${k}, ${t}` });
        });
        plot(s1, { xlim: [0.1, 1.5], ylim: [1e-3, 1e4], logy: true, xlabel: "training r_min (Å)", ylabel: "force RMSE on the training set (eV/Å)", series: ser, vlines: [[rm, "vline"]] });
      } else if (st.view === "E") {
        const pos = (a) => a.map((v) => Math.max(v, 1e-3));
        plot(s1, { xlim: [rm, 4], ylim: st.target === "full" ? [0.01, 3e4] : [-10, 10], logy: st.target === "full", xlabel: "r (Å)", ylabel: st.target === "full" ? "E (eV, log)" : "E (eV)",
          series: [{ x: f.r, y: st.target === "full" ? pos(f.E_ref.map((v) => v + 3.2)) : f.E_ref, cls: "s6", label: "target" + (st.target === "full" ? " + 3.2 eV" : "") },
                   { x: f.r, y: st.target === "full" ? pos(f.E_fit.map((v) => v + 3.2)) : f.E_fit, cls: "s2", dash: 1, label: "fit" + (st.target === "full" ? " + 3.2 eV" : "") }] });
      } else {
        const err = f.F_fit.map((v, i) => Math.abs(v - f.F_ref[i]) + 1e-6);
        const floor = f.F_ref.map((v, i) => Math.max(Math.abs(v) - f.L_fit * f.dGdr[i], 1e-6));
        plot(s1, { xlim: [rm, 4], ylim: [1e-3, 1e5], logy: true, xlabel: "r (Å)", ylabel: "|force| (eV/Å)",
          series: [{ x: f.r, y: f.F_ref.map((v) => Math.abs(v) + 1e-6), cls: "s6", label: "|F_ref|" }, { x: f.r, y: err, cls: "s2", label: "|F_fit − F_ref|" },
                   { x: f.r, y: f.r.map((_, i) => f.L_fit * f.dGdr[i] + 1e-6), cls: "s3", dash: 1, label: "max reachable L·‖dG/dr‖" },
                   { x: f.r, y: floor, cls: "s5", dash: 1, label: "predicted floor |F_ref| − L‖dG/dr‖" }] });
      }
      ro.innerHTML = `${st.kind}, ${st.target}, r<sub>min</sub> = ${rm} Å: RMSE<sub>E</sub> = <b>${fmt(f.rmse_E, 3)}</b> eV, RMSE<sub>F</sub> = <b>${fmt(f.rmse_F, 2)}</b> eV/Å; ` +
        `fitted Lipschitz constant L = max‖∂E/∂G‖ = ${fmt(f.L_fit, 1)}, required L<sub>req</sub> = max |F<sub>ref</sub>|/‖dG/dr‖ = <b>${fmt(f.L_req, 1)}</b>. ` +
        `Where the green line falls below |F<sub>ref</sub>| no setting of the weights with this L can reach the target force — the purple line is that floor.`;
    };
    root._redraw();
  };

  // 12. required Lipschitz constant (a priori, model-free) -----------------------------
  DEMOS.lreq = function (root) {
    const L = window.DEMO_DATA.learn;
    const st = ui(root, [{ key: "L", type: "range", label: "Lipschitz budget L of your model (log10)", min: 0, max: 6, step: 0.1, value: 2.5, show: (v) => "10^" + v.toFixed(1) }]);
    const [s1] = svgBox(root, 1), ro = readout(root);
    root._redraw = function () {
      const a = L.Lreq_G2, b = L.Lreq_Bessel, Lb = Math.pow(10, st.L);
      plot(s1, { xlim: [0.05, 4], ylim: [1e-2, 1e9], logx: true, logy: true, xlabel: "r (Å)", ylabel: "L_req(r) = |F_ref| / (2‖dG/dr‖)",
        series: [{ x: a.r, y: a.full, cls: "s2", label: "G2, full target" }, { x: a.r, y: a.delta, cls: "s3", dash: 1, label: "G2, delta target" },
                 { x: b.r, y: b.full, cls: "s4", label: "Bessel, full target" }, { x: b.r, y: b.delta, cls: "s5", dash: 1, label: "Bessel, delta target" },
                 { x: [0.05, 4], y: [Lb, Lb], cls: "s6", dash: 1, label: "your L" }] });
      const cross = (arr) => { for (let i = arr.r.length - 1; i >= 0; i--) if (arr.full[i] > Lb) return arr.r[i]; return null; };
      const cg = cross(a), cb = cross(b);
      ro.innerHTML = `With L = ${fmt(Lb, 0)}, the full target becomes unreachable below about <b>${cg ? cg.toFixed(2) : "—"} Å</b> (G2) and <b>${cb ? cb.toFixed(2) : "—"} Å</b> (Bessel). ` +
        `Near r → 0 both feature sets have ‖dG/dr‖ ∝ r while |F<sub>ZBL</sub>| ∝ 1/r², so L<sub>req</sub> ∝ r<sup>−3</sup>: halving r<sub>min</sub> asks for 8× larger weights. ` +
        `The delta target only has to bridge the residual, so its curve stays orders of magnitude lower.`;
    };
    root._redraw();
  };

  // 13. basis families near r = 0 ----------------------------------------------------------
  DEMOS.basis = function (root) {
    const st = ui(root, [
      { key: "fam", type: "select", label: "basis family", value: "all", options: [["all", "compare"], ["gauss", "Gaussian G2 (Rs = 0)"], ["bessel", "Bessel sin(cr)/r"], ["inv", "inverse power 1/r"], ["yuk", "screened Coulomb e^{−κr}/r"]] },
      { key: "p", type: "range", label: "width parameter (η, c, κ)", min: 0.2, max: 5, step: 0.1, value: 1.5, show: (v) => v.toFixed(1) },
    ]);
    const [s1, s2] = svgBox(root, 2), ro = readout(root);
    root._redraw = function () {
      const p = st.p, r = logspace(-2.5, Math.log10(4), 220);
      const fams = {
        gauss: [(x) => Math.exp(-p * x * x), (x) => 2 * p * x * Math.exp(-p * x * x), "s2", "Gaussian"],
        bessel: [(x) => Math.sin(p * x) / x, (x) => Math.abs((p * x * Math.cos(p * x) - Math.sin(p * x)) / (x * x)), "s3", "Bessel"],
        inv: [(x) => 1 / x, (x) => 1 / (x * x), "s4", "1/r"],
        yuk: [(x) => Math.exp(-p * x) / x, (x) => Math.exp(-p * x) * (p * x + 1) / (x * x), "s5", "e^{−κr}/r"],
      };
      const keys = st.fam === "all" ? Object.keys(fams) : [st.fam];
      plot(s1, { xlim: [r[0], 4], ylim: [1e-3, 1e3], logx: true, logy: true, xlabel: "r (Å)", ylabel: "|φ(r)|", title: "value",
        series: keys.map((k) => ({ x: r, y: r.map((x) => Math.abs(fams[k][0](x))), cls: fams[k][2], label: fams[k][3] })) });
      plot(s2, { xlim: [r[0], 4], ylim: [1e-4, 1e6], logx: true, logy: true, xlabel: "r (Å)", ylabel: "|dφ/dr|", title: "slope (what a force can be built from)",
        series: keys.map((k) => ({ x: r, y: r.map((x) => fams[k][1](x)), cls: fams[k][2], label: fams[k][3] })) });
      ro.innerHTML = `Gaussian and Bessel: value → constant, slope ∝ r (bounded inputs, no force at r = 0). 1/r and e<sup>−κr</sup>/r: value ∝ 1/r, slope ∝ 1/r² — the shape of nuclear repulsion. ` +
        `But a divergent <i>input</i> alone does not give a divergent <i>energy</i>: through tanh it saturates, through SiLU it grows only linearly in the input. ` +
        `To inherit 1/r you need the singular function to enter the energy <b>linearly</b>, i.e. as an explicit pair term Σ c<sub>k</sub> e<sup>−d<sub>k</sub>r</sup>/r — exactly the ZBL form.`;
    };
    root._redraw();
  };

  // 14. what "perpendicular" means for the angle derivative --------------------------------
  DEMOS.tangent = function (root) {
    const st = ui(root, [
      { key: "r", type: "range", label: "r_ij (Å)", min: 0.15, max: 2.5, step: 0.01, value: 1.0, show: (v) => v.toFixed(2) },
      { key: "th", type: "range", label: "θ_jik (°)", min: 20, max: 160, step: 1, value: 70 },
      { key: "d", type: "range", label: "move j by δ (Å)", min: 0, max: 0.3, step: 0.005, value: 0.1, show: (v) => v.toFixed(3) },
      { key: "dir", type: "select", label: "direction of δ", value: "t", options: [["t", "tangential (⊥ r_ij)"], ["n", "radial (∥ r_ij)"]] },
    ]);
    const wrap = document.createElement("div"); wrap.className = "plots"; root.appendChild(wrap);
    const svg = el("svg", { viewBox: "0 0 640 330", class: "plot geo" }, wrap);
    const ro = readout(root);
    root._redraw = function () {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const sc = 100, cx = 260, cy = 200, th = st.th * Math.PI / 180, r = st.r;
      const P = (x, y) => [cx + x * sc, cy - y * sc];
      const [ix, iy] = P(0, 0), [kx, ky] = P(2.2, 0);
      const jx = r * Math.cos(th), jy = r * Math.sin(th);
      const ux = Math.cos(th), uy = Math.sin(th), tx = -uy, ty = ux;
      const dx = st.dir === "t" ? tx * st.d : ux * st.d, dy = st.dir === "t" ? ty * st.d : uy * st.d;
      const j2x = jx + dx, j2y = jy + dy, th2 = Math.atan2(j2y, j2x);
      el("circle", { cx: ix, cy: iy, r: r * sc, class: "geo-circle" }, svg);
      el("line", { x1: ix, y1: iy, x2: kx, y2: ky, class: "geo-bond" }, svg);
      el("line", { x1: ix, y1: iy, x2: P(jx, jy)[0], y2: P(jx, jy)[1], class: "geo-bond" }, svg);
      el("line", { x1: ix, y1: iy, x2: P(j2x, j2y)[0], y2: P(j2x, j2y)[1], class: "geo-bond2" }, svg);
      el("line", { x1: P(jx, jy)[0], y1: P(jx, jy)[1], x2: P(j2x, j2y)[0], y2: P(j2x, j2y)[1], class: "geo-arrow" }, svg);
      const arc = (rad, a0, a1, cls) => {
        const [x0, y0] = P(rad * Math.cos(a0), rad * Math.sin(a0)), [x1, y1] = P(rad * Math.cos(a1), rad * Math.sin(a1));
        el("path", { d: `M${x0},${y0} A${rad * sc},${rad * sc} 0 0 0 ${x1},${y1}`, class: cls }, svg);
      };
      arc(0.35, 0, th, "geo-ang");
      if (Math.abs(th2 - th) > 1e-4) arc(0.5, Math.min(th, th2), Math.max(th, th2), "geo-dang");
      [[ix, iy, "i"], [kx, ky, "k"], [...P(jx, jy), "j"], [...P(j2x, j2y), "j′"]].forEach(([x, y, t]) => {
        el("circle", { cx: x, cy: y, r: t === "j′" ? 6 : 9, class: "geo-atom" + (t === "j′" ? " ghost" : "") }, svg);
        el("text", { x: t === "j′" ? x - 26 : x + 11, y: y - 9, class: "geo-label" }, svg).textContent = t;
      });
      const dth = (th2 - th) * 180 / Math.PI;
      const cos1 = Math.cos(th), cos2 = Math.cos(th2);
      ro.innerHTML = `j moves by δ = ${st.d.toFixed(3)} Å ${st.dir === "t" ? "along the circle around i (tangent)" : "straight away from i"}: θ changes by <b>${dth.toFixed(2)}°</b> ` +
        (st.dir === "t" ? `≈ (δ/r<sub>ij</sub>)·180/π = ${(st.d / r * 180 / Math.PI).toFixed(2)}°.` : `(zero: the direction of r<sub>ij</sub> is unchanged).`) +
        ` Δcosθ = ${(cos2 - cos1).toFixed(4)}. Shrink r<sub>ij</sub> with the same δ and the tangential change grows like 1/r<sub>ij</sub>. ` +
        `Moving i instead of j gives the same angle change with the opposite sign (only r<sub>ij</sub> = x<sub>j</sub> − x<sub>i</sub> enters), so "the force on i" and "the force on j" from this term are equal and opposite.`;
    };
    root._redraw();
  };

  // 15. switch variants for the SevenNet spectator force ------------------------------------
  DEMOS.swvar = function (root) {
    const V = window.DEMO_DATA.sw_variants;
    const st = ui(root, [{ key: "log", type: "check", label: "log scale", value: true }]);
    const [s1] = svgBox(root, 1), ro = readout(root);
    root._redraw = function () {
      const r = V.rows.map((x) => x.r), lg = st.log, y = (k) => V.rows.map((x) => (lg ? Math.max(x[k], 1e-3) : x[k]));
      plot(s1, { xlim: [0.3, 0.9], ylim: lg ? [1e-3, 100] : [0, 60], logy: lg, legendBottom: lg, legendLeft: true, xlabel: "distance of the pushed Si (Å)", ylabel: "max spectator force from the switch (eV/Å)",
        series: [{ x: r, y: y("a0.104"), cls: "s6", label: "α = 0.104 (checkpoint)" }, { x: r, y: y("ref_iso"), cls: "s4", dash: 1, label: "α = 0.104, E_i^ML − E_iso" },
                 { x: r, y: y("a0.05"), cls: "s2", label: "α = 0.05" }, { x: r, y: y("a0.03"), cls: "s3", label: "α = 0.03" }, { x: r, y: y("a0.02"), cls: "s5", label: "α = 0.02" }] });
      ro.innerHTML = `Same E<sub>i</sub><sup>ML</sup> (from the checkpoint) and the same Si17 path; only the switch definition changes. Re-referencing to the isolated atom (E<sub>iso</sub> = ${V.E_iso.toFixed(2)} eV) hardly helps, ` +
        `because E<sub>i</sub><sup>ML</sup> is 25–75 eV here from learned repulsion. A sharper softmin (smaller α) makes ∂σ<sub>i</sub>/∂x<sub>k</sub> ≈ 0 for every k that is not the closest neighbour. ` +
        `ACE's "zbl" mode uses the hard minimum distance: spectator term exactly 0. Changing α changes the model energy, so it needs retraining.`;
    };
    root._redraw();
  };

  // ------------------------------------------------------------------ mount
  let resizeTimer = null;
  if (typeof window !== "undefined") window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => document.querySelectorAll("div.demo").forEach((d) => d._redraw && d._redraw()), 150);
  });
  function mount() {
    const node = document.getElementById("demo-data");
    if (node) window.DEMO_DATA = JSON.parse(node.textContent);
    document.querySelectorAll("div.demo[data-demo]").forEach((d) => {
      const f = DEMOS[d.dataset.demo];
      if (!f) return;
      try { f(d); } catch (e) { d.textContent = "demo failed: " + e.message; console.error(e); }
    });
  }
  if (typeof document !== "undefined") {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount); else mount();
  }
  if (typeof module !== "undefined") module.exports = { zbl, zblSw, swCoeff };
})();
