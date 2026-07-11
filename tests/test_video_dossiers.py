import unittest
import json

import agent


class VideoDossierTests(unittest.TestCase):
    def test_normalize_transcript_entries_accepts_object_snippets(self):
        class Snippet:
            text = "object style transcript"
            start = 4.2
            duration = 2.0

        self.assertEqual(
            agent._normalize_transcript_entries([Snippet()]),
            [{"text": "object style transcript", "start": 4.2, "duration": 2.0}],
        )

    def test_extract_video_id_supports_common_youtube_urls(self):
        cases = {
            "https://www.youtube.com/watch?v=abcDEF12345": "abcDEF12345",
            "https://youtu.be/abcDEF12345": "abcDEF12345",
            "https://www.youtube.com/embed/abcDEF12345": "abcDEF12345",
            "https://www.youtube.com/shorts/abcDEF12345": "abcDEF12345",
            "abcDEF12345": "abcDEF12345",
        }

        for url, expected in cases.items():
            with self.subTest(url=url):
                self.assertEqual(agent._extract_video_id(url), expected)

    def test_build_video_dossier_preserves_timestamped_transcript_and_pivots(self):
        video = {
            "url": "https://www.youtube.com/watch?v=abcDEF12345",
            "title": "Eyewitness footage from Aleppo",
            "source": "YouTube",
            "date": "Jan 4, 2026",
            "duration": "4:21",
            "thumbnail": "https://i.ytimg.com/vi/abcDEF12345/hqdefault.jpg",
        }
        transcript = [
            {"start": 12.4, "duration": 4.1, "text": "we saw the convoy enter the town"},
            {"start": 80.0, "duration": 3.0, "text": "the shelling started after sunset"},
        ]

        dossier = agent._build_video_dossier(video, transcript)

        self.assertEqual(dossier["video_id"], "abcDEF12345")
        self.assertEqual(dossier["url"], video["url"])
        self.assertEqual(dossier["title"], video["title"])
        self.assertIn("[00:12] we saw the convoy enter the town", dossier["transcript"])
        self.assertIn("[01:20] the shelling started after sunset", dossier["transcript"])
        self.assertIn("https://i.ytimg.com/vi/abcDEF12345/maxresdefault.jpg", dossier["verification_pivots"]["thumbnails"])
        self.assertIn("https://lens.google.com/uploadbyurl?url=", dossier["verification_pivots"]["reverse_image_search"])
        self.assertEqual(dossier["evidence"]["platform"], "youtube")
        self.assertEqual(dossier["evidence"]["capture_method"], "metadata_and_transcript")

    def test_video_search_queries_force_youtube_and_evidence_terms(self):
        queries = agent._video_search_queries("detention abuse Syria")

        self.assertEqual(queries[0], "detention abuse Syria")
        self.assertIn("youtube", queries[1].lower())
        self.assertIn("eyewitness", queries[1].lower())
        self.assertIn("site:youtube.com", queries[2])

    def test_prefetch_video_evidence_streams_media_and_dossiers(self):
        original_execute_tool = agent.execute_tool
        try:
            agent.execute_tool = lambda name, args: json.dumps([
                {
                    "url": "https://www.youtube.com/watch?v=abcDEF12345",
                    "title": "Evidence video",
                    "dossier": {"video_id": "abcDEF12345", "transcript": "[00:01] testimony"},
                }
            ])
            collected = {"videos": []}

            events = list(agent._prefetch_video_evidence("Syria testimony", collected, disabled_tools=set()))

            self.assertEqual(len(collected["videos"]), 1)
            self.assertEqual(events[0]["type"], "tool_call")
            self.assertEqual(events[1]["type"], "tool_result")
            self.assertEqual(events[2]["type"], "media")
            self.assertEqual(events[3]["type"], "video_dossiers")
        finally:
            agent.execute_tool = original_execute_tool

    def test_llm_tools_skip_video_search_after_prefetch_results(self):
        tools = [
            {"function": {"name": "web_research"}},
            {"function": {"name": "search_videos"}},
            {"function": {"name": "search_news"}},
        ]

        active = agent._llm_tools_after_prefetch(tools, {"videos": [{"url": "https://youtu.be/abcDEF12345"}]})

        self.assertEqual([t["function"]["name"] for t in active], ["web_research", "search_news"])


if __name__ == "__main__":
    unittest.main()
