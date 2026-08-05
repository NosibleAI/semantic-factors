"""Verify the static package landing page exposes valid SEO metadata."""

import json
from html.parser import HTMLParser
from pathlib import Path
from typing import List


class JsonLdParser(HTMLParser):
    """Collect JSON-LD script contents from an HTML document."""

    def __init__(self) -> None:
        super().__init__()
        self._inside_json_ld = False
        self._parts: List[str] = []
        self.documents: List[object] = []

    def handle_starttag(self, tag: str, attrs: List[tuple[str, str | None]]) -> None:
        """Start collecting a JSON-LD script."""
        if tag != "script":
            return
        attributes = dict(attrs)
        self._inside_json_ld = attributes.get("type") == "application/ld+json"
        if self._inside_json_ld:
            self._parts = []

    def handle_data(self, data: str) -> None:
        """Collect script text while inside JSON-LD."""
        if self._inside_json_ld:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        """Parse a completed JSON-LD script."""
        if tag != "script" or not self._inside_json_ld:
            return
        self.documents.append(json.loads("".join(self._parts)))
        self._inside_json_ld = False


def test_docs_landing_page_has_indexable_metadata_and_json_ld() -> None:
    """Keep the package landing page aligned with the canonical website SEO contract."""
    page = Path(__file__).parents[1] / "docs" / "index.html"
    html = page.read_text(encoding="utf-8")

    assert '<meta name="robots" content="index, follow' in html
    assert '<link rel="canonical" href="https://www.nosible.com/semantic-factors">' in html
    assert '<meta property="og:title"' in html
    assert '<meta name="twitter:card" content="summary_large_image">' in html

    parser = JsonLdParser()
    parser.feed(html)
    assert len(parser.documents) == 1
    graph = parser.documents[0]["@graph"]
    entities = {entity["@type"]: entity for entity in graph}
    assert entities["Organization"]["@id"] == "https://www.nosible.com/#organization"
    assert entities["WebSite"]["@id"] == "https://www.nosible.com/#website"
    assert entities["SoftwareSourceCode"]["codeRepository"] == "https://github.com/NosibleAI/semantic-factors"
    assert entities["BreadcrumbList"]["itemListElement"][-1]["name"] == "Semantic Factors"
    assert entities["ItemList"]["numberOfItems"] == 4
