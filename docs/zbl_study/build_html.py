"""Build zbl_study.html from zbl_study.md (figures inlined, MathJax for math).

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
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #161718; --fg: #e6e6e3; --muted: #a0a3a7; --line: #2f3133;
    --card: #1e2022; --accent: #fb923c; --accent-soft: #2b1d14;
    --code-bg: #24272a; --th-bg: #232527;
  }
}
:root[data-theme="dark"] {
  --bg: #161718; --fg: #e6e6e3; --muted: #a0a3a7; --line: #2f3133;
  --card: #1e2022; --accent: #fb923c; --accent-soft: #2b1d14;
  --code-bg: #24272a; --th-bg: #232527;
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
mjx-container[display="true"] { overflow-x: auto; overflow-y: hidden; }
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
    body = re.sub(r'<p><img alt="[^"]*" src="(figures/[^"]+\.svg)"\s*/?></p>', inline_svg, body)
    body = re.sub(r"<table>", '<div class="table-wrap"><table>', body)
    body = re.sub(r"</table>", "</table></div>", body)
    html = TEMPLATE.replace("__TOC__", md.toc).replace("__BODY__", body)
    with open(os.path.join(HERE, "zbl_study.html"), "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
