"""Render the hand-authored VCIA walkthrough as an offline HTML file."""

from html import escape

_PAGE_START = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VCIA decision walkthrough</title>
  <style>
    :root {
      color-scheme: light;
      --paper: #f5f5f0;
      --surface: #fff;
      --ink: #192c33;
      --muted: #486068;
      --line: #d1dbd8;
      --tint: #edf3f0;
      --accent: #245e57;
      --focus: #166c81;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--paper);
      font: 16px/1.55 system-ui, "Segoe UI", sans-serif;
    }
    button { font: inherit; }
    :focus-visible { outline: 3px solid var(--focus); outline-offset: 3px; }
    [hidden] { display: none !important; }
    main { width: min(100%, 74rem); margin-inline: auto; padding: 0 clamp(1rem, 4vw, 3rem) 3rem; }
    h1, h2, h3 { margin: 0; line-height: 1.15; }
    h1, h2 { font-family: Georgia, "Times New Roman", serif; letter-spacing: -.025em; text-wrap: balance; }
    h1 { font-size: clamp(2.3rem, 5vw, 4rem); }
    h2 { font-size: clamp(1.65rem, 3.2vw, 2.5rem); }
    h3 { font-size: 1.08rem; }
    p { margin: .55rem 0 0; color: var(--muted); }
    .intro {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 15rem);
      align-items: center;
      gap: 1.5rem;
      padding-block: clamp(1.25rem, 4vw, 2.75rem);
    }
    .intro p { max-width: 46rem; }
    .setup { margin: 0; padding: .85rem 1rem; background: var(--tint); border-left: 3px solid var(--accent); }
    .setup div { display: flex; justify-content: space-between; gap: 1rem; }
    .setup div + div { margin-top: .25rem; }
    .setup dt { color: var(--muted); font-size: .86rem; }
    .setup dd { margin: 0; font-size: .86rem; font-weight: 750; font-variant-numeric: tabular-nums; }
    .case-nav { display: flex; flex-wrap: wrap; gap: .5rem; margin-bottom: .8rem; }
    .case-nav button {
      padding: .45rem .85rem;
      color: var(--accent);
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: .3rem;
      font-size: .88rem;
      font-weight: 700;
      text-align: left;
      cursor: pointer;
    }
    .case-nav button:hover { border-color: var(--accent); }
    .case-nav button[aria-pressed="true"] { color: #fff; background: var(--accent); border-color: var(--accent); }
    article {
      padding: clamp(1rem, 3vw, 1.85rem);
      background: var(--surface);
      border: 1px solid var(--line);
      border-top: 3px solid var(--accent);
      border-radius: .55rem;
    }
    article + article { margin-top: 1rem; }
    .case-summary { max-width: 53rem; }
    .limits { margin: 1.15rem 0 .5rem; color: var(--ink); font-size: .86rem; font-weight: 650; }
    .outcomes { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .75rem; margin-bottom: 1.1rem; }
    .outcome { min-width: 0; padding: .8rem .9rem; border: 1px solid var(--line); border-radius: .4rem; }
    .outcome header { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: .4rem; }
    .outcome p { margin-top: .38rem; font-size: .84rem; }
    .status { padding: .1rem .42rem; border-radius: .2rem; font-size: .75rem; font-weight: 800; letter-spacing: .025em; }
    .commit { color: #13563b; background: #ddefe4; }
    .hold { color: #70470f; background: #f9ebd2; }
    .values {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: .45rem .75rem;
      margin: .65rem 0 0;
      padding-top: .55rem;
      border-top: 1px solid var(--line);
    }
    .values dt { color: var(--muted); font-size: .76rem; }
    .values dd { margin: 0; font-size: .91rem; font-weight: 700; font-variant-numeric: tabular-nums; }
    details { border-top: 1px solid var(--line); }
    summary { cursor: pointer; padding: .62rem .1rem; font-size: .9rem; font-weight: 700; }
    summary::marker { color: var(--accent); }
    .table-region { max-width: 100%; overflow-x: auto; margin-bottom: .8rem; border: 1px solid var(--line); border-radius: .3rem; }
    table { width: 100%; min-width: 32rem; border-collapse: collapse; font-size: .84rem; }
    caption { padding: .55rem .65rem; color: var(--muted); text-align: left; font-weight: 650; }
    th, td { padding: .42rem .65rem; text-align: left; vertical-align: top; }
    thead th { background: var(--tint); }
    tbody th, td { border-top: 1px solid var(--line); font-variant-numeric: tabular-nums; }
    tr:nth-child(even) { background: #f6f9f7; }
    .scope { max-width: 57rem; padding-top: .4rem; font-size: .85rem; }
    @media (max-width: 46rem) {
      .intro { grid-template-columns: 1fr; gap: .85rem; }
      .setup { max-width: 25rem; }
    }
    @media (max-width: 36rem) {
      .outcomes { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<main>
  <header class="intro">
    <div>
      <h1>VCIA decision walkthrough</h1>
      <p>Follow two worked examples of version selection and admission. Each proposal names the peers that switch to current versions. HOLD retains all stored versions.</p>
    </div>
    <dl class="setup" aria-label="Walkthrough settings">
      <div><dt>Peers</dt><dd>3</dd></div>
      <div><dt>Replacement quota</dt><dd>2</dd></div>
      <div><dt>Design batch</dt><dd>512 repeats</dd></div>
      <div><dt>Audit batch</dt><dd>8,192 repeats</dd></div>
    </dl>
  </header>
"""

_PAGE_END = """  <footer class="scope">
    <p>Each batch repeats one hand-authored probe to illustrate the calculations. The audit's independent-sampling guarantee requires fresh independent draws. COMMIT denotes acceptance in the walkthrough.</p>
  </footer>
</main>
<script>
  const nav = document.querySelector('.case-nav');
  const buttons = [...nav.querySelectorAll('button')];
  function showCase(active) {
    for (const button of buttons) {
      const selected = button === active;
      button.setAttribute('aria-pressed', String(selected));
      document.getElementById(button.getAttribute('aria-controls')).hidden = !selected;
    }
  }
  for (const button of buttons) {
    button.addEventListener('click', () => showCase(button));
  }
  // Both cases remain readable when scripting is disabled.
  showCase(buttons[0]);
  nav.hidden = false;
</script>
</body>
</html>
"""

_CASE_LABELS = ("Screening and cancellation", "Ranking and audit")
_STAGE_LABELS = ("Probe inputs", "Individual screening", "Composition search")


def _render_outcome(row):
    policy, proposed, drift, flip, decision, accepted, reason = row
    values = (
        ("Proposed current peers", proposed),
        ("Accepted current peers", accepted),
        ("Drift upper bound", drift),
        ("Reassignment upper bound", flip),
    )
    measurements = "".join(
        f"<div><dt>{label}</dt><dd>{escape(value)}</dd></div>"
        for label, value in values
    )
    return (
        '<section class="outcome">'
        f'<header><h3>{escape(policy)}</h3>'
        f'<span class="status {escape(decision.lower())}">{escape(decision)}</span></header>'
        f'<p>{escape(reason[0].upper() + reason[1:])}.</p>'
        f'<dl class="values">{measurements}</dl></section>'
    )


def _render_table(caption, headers, rows):
    columns = "".join(f'<th scope="col">{escape(header)}</th>' for header in headers)
    body = "".join(
        f'<tr><th scope="row">{escape(key)}</th>'
        + "".join(f"<td>{escape(value)}</td>" for value in values)
        + "</tr>"
        for key, *values in rows
    )
    label = escape(caption)
    return (
        f'<div class="table-region" role="region" aria-label="{label}" tabindex="0">'
        f"<table><caption>{label}</caption><thead><tr>{columns}</tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def render_html(sections):
    """Render the two walkthrough cases and their decision tables."""
    parts = [_PAGE_START, '<nav class="case-nav" aria-label="Walkthrough cases" hidden>']
    for number, label in enumerate(_CASE_LABELS, start=1):
        parts.append(
            f'<button type="button" aria-controls="case-{number}">'
            f"{number}. {label}</button>"
        )
    parts.append("</nav>")

    for number, (title, summary, tables) in enumerate(sections, start=1):
        outcomes, *stages = tables
        caption, _, rows = outcomes
        parts.append(
            f'<article id="case-{number}" aria-labelledby="title-{number}">'
            f'<h2 id="title-{number}">{escape(title)}</h2>'
            f'<p class="case-summary">{escape(summary)}</p>'
            f'<p class="limits">{escape(caption)}</p>'
            '<div class="outcomes">'
            + "".join(_render_outcome(row) for row in rows)
            + "</div>"
        )
        for label, table in zip(_STAGE_LABELS, stages, strict=True):
            parts.append(
                f"<details><summary>{label}</summary>{_render_table(*table)}</details>"
            )
        parts.append("</article>")
    parts.append(_PAGE_END)
    return "\n".join(parts)
