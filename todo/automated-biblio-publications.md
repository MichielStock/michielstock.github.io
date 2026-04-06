# Future workflow: Automated publication list from UGent Biblio

## Status: TODO
**Priority:** Medium — the iframe embed works for now, but this gives full styling control and removes the iframe dependency.

## Goal

Replace the Biblio iframe on research.qmd with a fully styled, auto-updating publication list fetched from the UGent Biblio JSON API at build time.

## Architecture

```
GitHub Action (weekly + on push)
  ↓
scripts/fetch_publications.py
  ↓
Fetches JSON from biblio.ugent.be API
  ↓
Writes data/all_publications.yml
  ↓
Quarto renders research.qmd from YAML
  ↓
Styled publication cards matching site theme
```

## UGent Biblio API

**Base URL:** `https://biblio.ugent.be`

**Person publications (JSON):**
```
https://biblio.ugent.be/person/{UGENT_ID}/publication/export?format=json
```

**Find your UGent ID:** Go to `https://biblio.ugent.be`, search your name, your ID is in the URL of your person page (e.g. `802001370792`).

**Python wrapper available:** `pip install ugentbiblio` ([docs](https://python-ugent-biblio.readthedocs.io/))

**Alternative: OAI-PMH endpoint** at `https://biblio.ugent.be/oai` for bulk harvesting.

**Alternative: SRU search** at `https://biblio.ugent.be/sru` with CQL queries.

## Implementation plan

### 1. Create `scripts/fetch_publications.py`

```python
#!/usr/bin/env python3
"""
Fetch publications from UGent Biblio and write to YAML.
Runs during GitHub Actions build to keep publication list current.
"""

import requests
import yaml
import sys
from pathlib import Path

UGENT_PERSON_ID = "802001370792"  # Update with correct ID
BIBLIO_URL = f"https://biblio.ugent.be/person/{UGENT_PERSON_ID}/publication/export"
OUTPUT_PATH = Path("data/all_publications.yml")
FEATURED_PATH = Path("data/publications.yml")


def fetch_publications():
    """Fetch all publications from UGent Biblio JSON API."""
    try:
        resp = requests.get(BIBLIO_URL, params={"format": "json"}, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"Warning: Could not fetch from Biblio API: {e}")
        return None


def parse_publication(pub):
    """Extract relevant fields from a Biblio publication record."""
    # The JSON structure may vary — inspect a real response to confirm field names
    return {
        "title": pub.get("title", "Untitled"),
        "authors": ", ".join(
            a.get("full_name", "") for a in pub.get("author", [])
        ),
        "year": pub.get("year", ""),
        "venue": pub.get("journal", {}).get("title", 
                 pub.get("publisher", "")),
        "type": pub.get("classification", ""),
        "doi": pub.get("doi", ""),
        "biblio_url": f"https://biblio.ugent.be/publication/{pub.get('id', '')}",
        "biblio_id": pub.get("id", ""),
    }


def get_featured_ids():
    """Load featured publication IDs to avoid duplicates."""
    if not FEATURED_PATH.exists():
        return set()
    with open(FEATURED_PATH) as f:
        featured = yaml.safe_load(f) or []
    # Match on title (lowercase) since we may not have biblio IDs in featured
    return {p.get("title", "").lower().strip() for p in featured}


def main():
    # Try to fetch from API
    data = fetch_publications()
    
    if data is None:
        # API failed — use existing file if available
        if OUTPUT_PATH.exists():
            print("Using existing all_publications.yml (API unavailable)")
            sys.exit(0)
        else:
            print("Error: No cached publications and API unavailable")
            sys.exit(1)
    
    # Parse publications
    publications = []
    for pub in data:
        parsed = parse_publication(pub)
        if parsed["title"] and parsed["year"]:
            publications.append(parsed)
    
    # Sort by year descending, then title
    publications.sort(key=lambda p: (-int(p.get("year", 0)), p.get("title", "")))
    
    # Remove featured publications (they're shown separately)
    featured_titles = get_featured_ids()
    publications = [
        p for p in publications 
        if p["title"].lower().strip() not in featured_titles
    ]
    
    # Write YAML
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        yaml.dump(publications, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"Wrote {len(publications)} publications to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
```

**Note:** The JSON structure from Biblio needs to be verified against a real API response. Fetch one manually first:
```bash
curl "https://biblio.ugent.be/person/802001370792/publication/export?format=json" | python -m json.tool | head -100
```
Then adjust the field names in `parse_publication()` accordingly.

### 2. Update `.github/workflows/publish.yml`

Add these steps before the Quarto render:

```yaml
- name: Setup Python
  uses: actions/setup-python@v5
  with:
    python-version: '3.12'

- name: Install Python dependencies
  run: pip install requests pyyaml

- name: Fetch publications from UGent Biblio
  run: python scripts/fetch_publications.py
  continue-on-error: true  # Don't fail build if API is down
```

Add a scheduled trigger:

```yaml
on:
  push:
    branches: [main]
  schedule:
    - cron: '0 6 * * 1'  # Every Monday 6am UTC
  workflow_dispatch:
```

### 3. Update `research.qmd`

Replace the iframe section with a Quarto listing or manual YAML rendering:

```yaml
---
listing:
  - id: all-pubs
    contents: data/all_publications.yml
    sort: "year desc"
    type: table
    fields: [year, title, authors, venue]
    field-display-names:
      year: Year
      title: Title
      authors: Authors
      venue: Journal
---
```

Or for more control, use a custom EJS template or build the HTML with a short Python/Julia script that generates a `.qmd` include.

### 4. Add `data/all_publications.yml` to `.gitignore`?

**Decision needed:** 

- If you gitignore it, the file is always freshly fetched. Cleaner repo, but builds fail if Biblio is down and there's no cache.
- If you commit it, the file serves as a cache. Builds always succeed. File updates via the weekly Action. **Recommended approach.**

## Dependencies

- Python 3.12+
- `requests` (HTTP client)
- `pyyaml` (YAML output)
- Optionally: `ugentbiblio` PyPI package instead of raw requests

## Testing

```bash
# Test the script locally
pip install requests pyyaml
python scripts/fetch_publications.py

# Verify output
cat data/all_publications.yml | head -50

# Check Quarto renders it
quarto preview
```

## Migration from iframe (Option B → Option A)

When implementing this:
1. Create `scripts/fetch_publications.py`
2. Run it once locally to generate `data/all_publications.yml`
3. Commit the generated YAML as the initial cache
4. Update `research.qmd` — remove the iframe, add the listing
5. Update the GitHub Actions workflow
6. Push and verify

## Claude Code prompt (when ready)

```
Implement automated publication fetching from UGent Biblio. 
Read the plan in todo/automated-biblio-publications.md and follow it.

Key points:
- Create scripts/fetch_publications.py
- My UGent person ID is 802001370792 (verify by fetching the API)
- Inspect the actual JSON response structure before writing the parser
- Write to data/all_publications.yml
- Skip publications already in data/publications.yml (featured)
- Update the GitHub Actions workflow to run the script before render
- Add weekly scheduled builds
- Replace the iframe on research.qmd with styled publication cards
  grouped by year, matching the Deep Ocean color scheme
- Commit the initial all_publications.yml as cache
- Make the script gracefully handle API failures (use cached file)
```
