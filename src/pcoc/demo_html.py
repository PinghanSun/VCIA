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
      --page: #f4f6fa;
      --surface: #fff;
      --ink: #1d263b;
      --muted: #4d5c73;
      --border: #d6deea;
      --tint: #eef2f8;
      --accent: #345ba3;
      --focus: #3263be;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--page);
      font: 16px/1.55 system-ui, "Segoe UI", sans-serif;
    }
    main { width: min(100%, 84rem); margin-inline: auto; padding: clamp(1rem, 4vw, 3rem); }
    .intro { max-width: 49rem; margin-bottom: 2rem; }
    h1, h2 { margin: 0; line-height: 1.12; text-wrap: balance; }
    h1 { font-size: clamp(2.15rem, 5vw, 3.65rem); letter-spacing: -.04em; }
    h2 { font-size: clamp(1.35rem, 2.5vw, 2rem); letter-spacing: -.025em; }
    p { margin-block: .65rem 0; color: var(--muted); }
    .scope {
      padding: .7rem .9rem;
      border-left: 3px solid var(--accent);
      background: var(--tint);
    }
    article {
      margin-block: 1rem;
      padding: clamp(.9rem, 2.5vw, 1.65rem);
      border: 1px solid var(--border);
      border-radius: .7rem;
      background: var(--surface);
    }
    .table-region {
      max-width: 100%;
      overflow-x: auto;
      margin-block: 1rem;
      border: 1px solid var(--border);
      border-radius: .4rem;
    }
    .table-region:focus-visible, summary:focus-visible {
      outline: 2px solid var(--focus);
      outline-offset: 3px;
    }
    table { width: 100%; border-collapse: collapse; font-size: .9rem; }
    caption {
      padding: .65rem .75rem;
      text-align: left;
      color: var(--ink);
      font-size: 1rem;
      font-weight: 700;
    }
    th, td { padding: .45rem .75rem; text-align: left; vertical-align: top; }
    th { color: var(--ink); background: var(--tint); font-weight: 650; }
    td { border-top: 1px solid var(--border); font-variant-numeric: tabular-nums; }
    tr:nth-child(even) td { background: #f8faff; }
    .status {
      display: inline-block;
      padding: .12rem .44rem;
      border-radius: .3rem;
      white-space: nowrap;
      font-size: .8rem;
      font-weight: 750;
    }
    .commit { color: #0d5c39; background: #e1f3e9; }
    .hold { color: #73490d; background: #fff0d6; }
    details { border-top: 1px solid var(--border); }
    summary { cursor: pointer; padding: .62rem .2rem; font-weight: 650; }
    summary::marker { color: var(--accent); }
    .row-count { color: var(--muted); margin-inline-start: .55rem; font-size: .83rem; font-weight: 450; }
    details .table-region { margin-block: .15rem .8rem; }
    @media (max-width: 35rem) {
      th, td { padding-inline: .55rem; }
    }
  </style>
</head>
<body>
<main>
  <header class="intro">
    <h1>VCIA decision walkthrough</h1>
    <p>Fixed, hand-authored cluster probabilities illustrate how the design probe batch guides selection and how a separate verification batch checks the selected composition.</p>
    <p>Each proposal chooses two of three peers to use current versions. The other peer keeps its stored version. HOLD keeps all three stored versions. The fixed batches repeat the design graph 512 times and the audit graph 8,192 times.</p>
    <p>COMMIT denotes an accepted composition in this walkthrough. The demo does not write peer state or execute distributed commit.</p>
    <p class="scope">Each batch contains one unique hand-authored probe. Repeated inputs illustrate the calculation and do not provide independent-sampling evidence. This walkthrough includes no learner training, benchmark evaluation, or empirical estimate of audit power.</p>
  </header>
"""


def _cell(value):
    status = value.strip().split(maxsplit=1)
    if status and status[0] in ("COMMIT", "HOLD"):
        label = status[0]
        detail = f" {escape(status[1])}" if len(status) == 2 else ""
        return f'<span class="status {label.lower()}">{label}</span>{detail}'
    return escape(value)


def _render_table(caption, headers, rows):
    columns = "".join(f'<th scope="col">{escape(header)}</th>' for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_cell(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    label = escape(caption)
    return (
        f'<div class="table-region" role="region" aria-label="{label}" tabindex="0">'
        f"<table><caption>{label}</caption><thead><tr>{columns}</tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def render_html(sections):
    """Render sections of (title, summary, tables) with string-valued cells."""
    parts = [_PAGE_START]
    for title, summary, tables in sections:
        parts.append(f"<article><h2>{escape(title)}</h2><p>{escape(summary)}</p>")
        for index, (caption, headers, rows) in enumerate(tables):
            rows = tuple(rows)
            table = _render_table(caption, headers, rows)
            if index:
                count = f"{len(rows)} {'row' if len(rows) == 1 else 'rows'}"
                table = (
                    f"<details><summary>{escape(caption)}"
                    f'<span class="row-count">{count}</span></summary>{table}</details>'
                )
            parts.append(table)
        parts.append("</article>")
    parts.append("</main></body></html>")
    return "\n".join(parts) + "\n"
