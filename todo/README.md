# Website future enhancements

Items to implement after the initial migration is stable.

## Workflows

- [ ] **Automated Biblio publications** — Replace iframe with build-time JSON fetch from UGent Biblio API. See [automated-biblio-publications.md](automated-biblio-publications.md)
- [ ] **Goodreads bookshelf** — CSV export → books.yml → bookshelf page with covers. Consider Goodreads RSS for a "currently reading" widget on home page.
- [ ] **Scheduled rebuilds** — Weekly GitHub Action to refresh publications and reading list without manual pushes.

## Design

- [ ] **Sketchnote accents** — Hand-drawn SVG underlines, wobbly borders, highlighter marks on section headings. See the workplan visual direction notes.
- [ ] **Generative hero** — Small canvas/SVG random network graph on home page, regenerates on each load. Nodes = research topics.
- [ ] **Publication thumbnail images** — Replace the placeholder SVGs with real figures from papers or hand-drawn sketchnotes of core ideas.

## Content

- [ ] **"Why Maximum Entropy?" section** — Short philosophical note on home or about page connecting information theory to research approach.
- [ ] **"Start here" blog guide** — Curated entry points for different reader interests.
- [ ] **Downloadable CV (PDF)** — Auto-generate from about.qmd content or maintain a separate LaTeX CV.
- [ ] **Open positions page** — Thesis topics, PhD openings, with structured cards.

## Technical

- [ ] **Julia engine for new posts** — Switch from `freeze: true` to `engine: julia` for new technical posts once QuartoNotebookRunner.jl is set up.
- [ ] **GoatCounter analytics** — Sign up and add tracking script.
- [ ] **Custom 404 page** — Friendly error page with search and navigation.
- [ ] **Lighthouse audit** — Run and fix any performance/accessibility issues.
