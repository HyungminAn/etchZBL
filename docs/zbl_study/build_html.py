"""Build zbl_study.html from zbl_study.md (figures, demo data and interactive.js inlined; MathJax for math).

Run: python3 build_html.py  (needs markdown, pymdown-extensions)
"""
import os
import re
import markdown

HERE = os.path.dirname(os.path.abspath(__file__))

TEMPLATE = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ZBL Switching Notes</title>
<style>
:root {
  --bg: #fbfbfa; --fg: #1d1f21; --muted: #5f6368; --line: #e3e3e0;
  --card: #ffffff; --accent: #c2410c; --accent-soft: #fff1e8;
  --code-bg: #f3f3f1; --th-bg: #f4f4f2;
  --s1: #2a6fdb; --s2: #d9480f; --s3: #2b8a3e; --s4: #c2255c; --s5: #7048e8; --s6: #495057;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #161718; --fg: #e6e6e3; --muted: #a0a3a7; --line: #2f3133;
    --card: #1e2022; --accent: #fb923c; --accent-soft: #2b1d14;
    --code-bg: #24272a; --th-bg: #232527;
    --s1: #74a7ff; --s2: #ff922b; --s3: #69db7c; --s4: #f783ac; --s5: #b197fc; --s6: #ced4da;
  }
}
:root[data-theme="dark"] {
  --bg: #161718; --fg: #e6e6e3; --muted: #a0a3a7; --line: #2f3133;
  --card: #1e2022; --accent: #fb923c; --accent-soft: #2b1d14;
  --code-bg: #24272a; --th-bg: #232527;
  --s1: #74a7ff; --s2: #ff922b; --s3: #69db7c; --s4: #f783ac; --s5: #b197fc; --s6: #ced4da;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 16px/1.7 "Pretendard", "Noto Sans KR", system-ui, -apple-system, "Segoe UI", sans-serif;
}
.layout { display: grid; grid-template-columns: 260px minmax(0, 1fr); max-width: 1180px; margin: 0 auto; }
nav.toc {
  position: sticky; top: 0; align-self: start; height: 100vh; overflow-y: auto;
  padding: 32px 16px 32px 24px; border-right: 1px solid var(--line); font-size: 13.5px;
}
nav.toc .title { font-weight: 700; margin-bottom: 8px; color: var(--muted); letter-spacing: .02em; }
nav.toc ul { list-style: none; padding-left: 0; margin: 0; }
nav.toc ul ul { padding-left: 12px; }
nav.toc a { color: var(--fg); text-decoration: none; display: block; padding: 3px 0; opacity: .82; }
nav.toc a:hover { color: var(--accent); opacity: 1; }
main { padding: 40px 48px 96px; min-width: 0; }
h1 { font-size: 2rem; line-height: 1.3; margin: 0 0 16px; }
h2 { font-size: 1.45rem; margin: 40px 0 12px; }
h3 { font-size: 1.12rem; margin: 32px 0 8px; }
a { color: var(--accent); }
p, li { max-width: 78ch; }
hr { border: 0; border-top: 1px solid var(--line); margin: 32px 0; }
code { background: var(--code-bg); padding: 1px 5px; border-radius: 4px; font-size: .9em; overflow-wrap: anywhere; }
blockquote {
  margin: 16px 0; padding: 10px 16px; border-left: 3px solid var(--accent);
  background: var(--accent-soft); border-radius: 0 6px 6px 0;
}
blockquote p { margin: 4px 0; }
.table-wrap { overflow-x: auto; margin: 16px 0; border: 1px solid var(--line); border-radius: 8px; background: var(--card); }
table { border-collapse: collapse; width: 100%; font-size: 14.5px; }
th, td { padding: 8px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
th { background: var(--th-bg); font-weight: 600; white-space: nowrap; }
tr:last-child td { border-bottom: 0; }
figure.fig { margin: 20px 0 4px; padding: 12px; background: #ffffff; border: 1px solid var(--line); border-radius: 8px; }
figure.fig svg { width: 100%; height: auto; display: block; }
p > em:only-child { color: var(--muted); font-size: 14px; }
.arithmatex { overflow-x: auto; overflow-y: hidden; }
mjx-container[display="true"] { overflow-x: auto; overflow-y: hidden; max-width: 100%; min-width: 0 !important; }
mjx-container:not([display="true"]) { display: inline-block; max-width: 100%; overflow-x: auto; overflow-y: hidden; vertical-align: middle; }
div.arithmatex { max-width: 100%; }
.theme-btn {
  position: fixed; top: 12px; right: 12px; z-index: 10; border: 1px solid var(--line);
  background: var(--card); color: var(--fg); border-radius: 6px; padding: 4px 10px; font-size: 13px; cursor: pointer;
}
@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; }
  nav.toc { position: static; height: auto; border-right: 0; border-bottom: 1px solid var(--line); padding: 16px; }
  main { padding: 24px 16px 64px; }
  h1 { font-size: 1.6rem; }
}
.demo { margin: 20px 0; padding: 14px 16px 10px; border: 1px solid var(--line); border-radius: 10px; background: var(--card); }
.demo::before { content: "interactive"; display: inline-block; font-size: 11px; letter-spacing: .06em; text-transform: uppercase;
  color: var(--accent); border: 1px solid var(--accent); border-radius: 999px; padding: 0 8px; margin-bottom: 8px; }
.controls { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 6px 18px; margin-bottom: 8px; font-size: 13.5px; }
.controls label { display: grid; grid-template-columns: 1fr auto; align-items: center; gap: 2px 8px; }
.controls label > span { grid-column: 1 / -1; color: var(--muted); }
.controls input[type=range] { width: 100%; accent-color: var(--accent); }
.controls output { font-variant-numeric: tabular-nums; min-width: 4.5em; text-align: right; font-size: 12.5px; }
.controls select { grid-column: 1 / -1; font: inherit; padding: 3px 6px; border-radius: 6px; border: 1px solid var(--line); background: var(--bg); color: var(--fg); }
.controls label.ctl-check { display: flex; align-items: center; gap: 8px; }
.controls label.ctl-check > span { order: 2; color: var(--fg); }
.controls input[type=checkbox] { accent-color: var(--accent); width: 16px; height: 16px; }
.plots { display: grid; grid-template-columns: 1fr; gap: 8px; }
.plots.two { grid-template-columns: 1fr; }
svg.plot { width: 100%; height: auto; display: block; overflow: hidden; font-family: inherit; }
svg.plot .frame { fill: none; stroke: var(--line); }
svg.plot .grid { stroke: var(--line); stroke-width: .6; opacity: .7; }
svg.plot .zero { stroke: var(--muted); stroke-width: .8; opacity: .6; }
svg.plot .tick, svg.plot .legend { fill: var(--muted); font-size: 11px; }
svg.plot .axlabel { fill: var(--fg); font-size: 12px; }
svg.plot .ptitle { fill: var(--fg); font-size: 12.5px; font-weight: 600; }
svg.plot .band { fill: var(--line); opacity: .55; }
svg.plot .band2 { fill: var(--s1); opacity: .12; }
svg.plot .vline { stroke: var(--accent); stroke-width: 1; stroke-dasharray: 3 3; }
svg.plot .line { fill: none; stroke-width: 2; stroke-linejoin: round; }
svg.plot .dash { stroke-dasharray: 6 4; }
svg.plot .pt { stroke: var(--card); stroke-width: 1.5; }
.s1 { stroke: var(--s1); fill: var(--s1); } .s2 { stroke: var(--s2); fill: var(--s2); } .s3 { stroke: var(--s3); fill: var(--s3); }
.s4 { stroke: var(--s4); fill: var(--s4); } .s5 { stroke: var(--s5); fill: var(--s5); } .s6 { stroke: var(--s6); fill: var(--s6); }
svg.plot .line.s1, svg.plot .line.s2, svg.plot .line.s3, svg.plot .line.s4, svg.plot .line.s5, svg.plot .line.s6 { fill: none; }
.b0 { stroke: #440154; fill: #440154; } .b1 { stroke: #46327e; fill: #46327e; } .b2 { stroke: #365c8d; fill: #365c8d; } .b3 { stroke: #277f8e; fill: #277f8e; }
.b4 { stroke: #1fa187; fill: #1fa187; } .b5 { stroke: #4ac16d; fill: #4ac16d; } .b6 { stroke: #a0da39; fill: #a0da39; } .b7 { stroke: #e3c800; fill: #e3c800; }
svg.plot .line[class*=" b"] { fill: none; stroke-width: 1.5; }
svg.plot .bar { stroke: none; }
.readout { font-size: 13.5px; line-height: 1.6; margin-top: 6px; color: var(--fg); }
.readout .muted { color: var(--muted); }
.strip { margin: 6px 0; }
.strip-label { font-size: 12px; color: var(--muted); margin-top: 4px; }
.cells { display: grid; grid-template-columns: repeat(30, 1fr); gap: 2px; margin-top: 2px; }
.cells span { display: block; height: 12px; border-radius: 2px; border: 1px solid var(--line); }
@media (max-width: 700px) { .cells { grid-template-columns: repeat(20, 1fr); } .demo { padding: 12px 10px 8px; } }
</style>
<script>
window.MathJax = { tex: { inlineMath: [["\\\\(", "\\\\)"]], displayMath: [["\\\\[", "\\\\]"]] },
                   options: { processHtmlClass: "arithmatex" } };
</script>
<script async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.js"></script>
</head>
<body>
<button class="theme-btn" type="button" onclick="toggleTheme()">◐ 테마</button>
<div class="layout">
<nav class="toc"><div class="title">목차</div>__TOC__</nav>
<main>
__BODY__
</main>
</div>
<script type="application/json" id="demo-data">__DATA__</script>
<script>
__JS__
</script>
<script>
function toggleTheme() {
  var r = document.documentElement;
  var dark = r.dataset.theme ? r.dataset.theme === "dark"
           : window.matchMedia("(prefers-color-scheme: dark)").matches;
  r.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("zbl-theme", r.dataset.theme); } catch (e) {}
}
try { var t = localStorage.getItem("zbl-theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) {}
</script>
</body>
</html>
"""


def inline_svg(match):
    path = os.path.join(HERE, match.group(1))
    with open(path) as f:
        svg = f.read()
    svg = svg[svg.index("<svg"):]
    svg = re.sub(r'(<svg[^>]*?)\swidth="[^"]*"\sheight="[^"]*"', r"\1", svg, count=1)
    return f'<figure class="fig">{svg}</figure>'


def main():
    with open(os.path.join(HERE, "zbl_study.md")) as f:
        text = f.read()
    md = markdown.Markdown(
        extensions=["tables", "toc", "pymdownx.arithmatex"],
        extension_configs={"pymdownx.arithmatex": {"generic": True},
                           "toc": {"toc_depth": "2-2"}})
    body = md.convert(text)
    demo = r'<div class="demo" data-demo="\1"></div>'
    # a demo marker replaces the static figure right after it, unless that figure is tagged keep:
    body = re.sub(r'<!-- demo:(\w+) -->\s*<p><img alt="(?!keep:)[^"]*" src="[^"]+"\s*/?></p>', demo, body)
    body = re.sub(r'<!-- demo:(\w+) -->', demo, body)
    body = re.sub(r'<p><img alt="static:(\w+)" src="[^"]+"\s*/?></p>', demo, body)
    body = re.sub(r'<p><img alt="[^"]*" src="(figures/[^"]+\.svg)"\s*/?></p>', inline_svg, body)
    body = re.sub(r"<table>", '<div class="table-wrap"><table>', body)
    body = re.sub(r"</table>", "</table></div>", body)
    with open(os.path.join(HERE, "demo_data.json")) as f:
        data = f.read().replace("</", "<\\/")
    with open(os.path.join(HERE, "interactive.js")) as f:
        js = f.read()
    html = (TEMPLATE.replace("__TOC__", md.toc).replace("__BODY__", body)
            .replace("__DATA__", data).replace("__JS__", js))
    with open(os.path.join(HERE, "zbl_study.html"), "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
