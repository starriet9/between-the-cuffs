"""Public outputs: the results README and the static page `docs/index.html` (SPEC §8.1-§8.3).

Modules
-------
config    site settings that are not results (dates, links, author, contact); `site.json`
results   adapter from the model pipeline's result files (btc/models) to SiteData
series    per-case series and summaries the frozen-scoring step writes for the case browser
featured  the pre-set featured-case rules (SPEC §8.2)
text      one source for every shared word: README blocks 1-8 as a document model
readme    Markdown renderer
page      HTML renderer (static; no fetch; works from file:// and without JavaScript)
figures   matplotlib figures, tiles, case images and the replay (MP4/WebM/GIF)
checks    placeholder, leak, unit-mix, parity, request and weight checks
build     preview and publish builds
fixtures  synthetic results and cases for previews and tests (never published)
"""
