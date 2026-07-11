import unittest
from datetime import datetime, timezone

import evidence


class EvidenceTests(unittest.TestCase):
    def test_extract_cited_urls_dedupes_and_skips_archives(self):
        md = (
            "See [report](https://example.org/a) and [again](https://example.org/a). "
            "Also [snap](https://web.archive.org/web/x) and [b](http://example.org/b)."
        )
        self.assertEqual(
            evidence.extract_cited_urls(md),
            ["https://example.org/a", "http://example.org/b"],
        )

    def test_extract_cited_urls_respects_limit(self):
        md = " ".join(f"[l](https://example.org/{i})" for i in range(20))
        self.assertEqual(len(evidence.extract_cited_urls(md, limit=5)), 5)

    def test_methodology_appendix_contains_record(self):
        out = evidence.methodology_appendix(
            query="test query",
            model="test/model",
            started_at=datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc),
            tool_log=[{"ts": "12:00:01", "tool": "web_research", "label": "test"}],
            collected={"images": [1], "videos": [], "news": [], "docs": []},
            report_body="body",
            archives={"https://example.org/a": "https://web.archive.org/web/1/https://example.org/a"},
        )
        self.assertIn("## Methodology & Collection Record", out)
        self.assertIn("web_research", out)
        self.assertIn("SHA-256", out)
        self.assertIn("Archived sources", out)


if __name__ == "__main__":
    unittest.main()
