import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.gen.mcp_schema_gen import SCHEMA_RESEARCH_TOOLS, SchemaGen
from src.gen.prompts import SchemaDesign_System_Prompt, SchemaGen_System_Prompt
from src.utils.web_research_tools import (
    _extract_page_text,
    _find_markdown_alternate,
    _read_with_agent_browser,
    _validate_public_url,
    read_webpage_impl,
    search_web_impl,
)


class _FakeDDGS:
    def __init__(self, timeout):
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def text(self, query, max_results):
        return [
            {
                "title": "Official API",
                "href": "https://developer.example.com/reference",
                "body": "API reference",
            }
        ][:max_results]


class WebResearchToolTest(unittest.TestCase):
    def test_search_returns_structured_results(self):
        with patch("src.utils.web_research_tools.DDGS", _FakeDDGS):
            result = json.loads(search_web_impl("official API", max_results=3))

        self.assertEqual(result["query"], "official API")
        self.assertEqual(result["results"][0]["url"], "https://developer.example.com/reference")

    def test_html_extraction_removes_scripts(self):
        title, content = _extract_page_text(
            "<html><head><title>Docs</title><script>ignore()</script></head>"
            "<body><main><h1>API</h1><p>Useful text.</p></main></body></html>",
            "text/html",
        )

        self.assertEqual(title, "Docs")
        self.assertIn("Useful text.", content)
        self.assertNotIn("ignore", content)

    def test_markdown_alternate_is_resolved(self):
        alternate = _find_markdown_alternate(
            '<html><head><link rel="alternate" type="text/markdown" href="get.md"></head></html>',
            "https://developer.example.com/docs/get",
        )
        self.assertEqual(alternate, "https://developer.example.com/docs/get.md")

    def test_private_urls_are_blocked(self):
        with self.assertRaisesRegex(ValueError, "blocked"):
            _validate_public_url("http://127.0.0.1/internal")

    def test_invalid_scheme_returns_error(self):
        result = json.loads(read_webpage_impl("file:///etc/passwd"))
        self.assertIn("error", result)

    def test_agent_browser_reader_returns_structured_content(self):
        payload = {
            "success": True,
            "error": None,
            "data": {
                "content": "# Official API\n\nRequest details",
                "contentType": "text/markdown",
                "finalUrl": "https://developer.example.com/reference",
                "source": "accept-markdown",
                "truncated": False,
            },
        }

        def fake_run(command, **kwargs):
            if command[-1] == "close":
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

        with (
            patch("src.utils.web_research_tools._validate_public_url"),
            patch("src.utils.web_research_tools.shutil.which", return_value="/usr/bin/agent-browser"),
            patch("src.utils.web_research_tools.subprocess.run", side_effect=fake_run),
        ):
            result = _read_with_agent_browser("https://developer.example.com/reference", 5000)

        self.assertEqual(result["backend"], "agent-browser")
        self.assertEqual(result["title"], "Official API")
        self.assertEqual(result["source"], "accept-markdown")

    def test_native_reader_is_used_when_agent_browser_fails(self):
        with (
            patch(
                "src.utils.web_research_tools._read_with_agent_browser",
                side_effect=RuntimeError("browser unavailable"),
            ),
            patch(
                "src.utils.web_research_tools._read_webpage_native",
                return_value={"content": "fallback", "backend": "native"},
            ),
        ):
            result = json.loads(read_webpage_impl("https://developer.example.com/reference"))

        self.assertEqual(result["backend"], "native")
        self.assertIn("browser unavailable", result["browser_error"])


class SchemaResearchPromptTest(unittest.TestCase):
    def test_schema_agents_expose_both_research_tools(self):
        self.assertEqual([tool.name for tool in SCHEMA_RESEARCH_TOOLS], ["search_web", "read_webpage"])
        with patch.object(SchemaGen, "get_model", return_value="test-model"):
            generator = SchemaGen()
        self.assertEqual(
            [tool.name for tool in generator.schema_generator.tools],
            ["search_web", "read_webpage"],
        )
        self.assertEqual(
            [tool.name for tool in generator.schema_designer.tools],
            ["search_web", "read_webpage"],
        )

    def test_prompts_require_primary_sources_and_provenance(self):
        for prompt in (SchemaDesign_System_Prompt, SchemaGen_System_Prompt):
            self.assertIn("official", prompt.lower())
            self.assertIn("sources", prompt)
            self.assertIn("untrusted", prompt.lower())

    def test_schema_validation_accepts_valid_sources(self):
        generator = SchemaGen.__new__(SchemaGen)
        schema = {
            "class_name": "Example",
            "description": "Example API",
            "sources": [
                {
                    "title": "Official API",
                    "url": "https://developer.example.com/reference",
                    "used_for": "Inputs and outputs",
                }
            ],
            "tools": [
                {
                    "name": "get_item",
                    "description": "Get an item",
                    "input_schema": {"type": "object", "properties": {}, "required": []},
                    "output_schema": {"type": "object", "properties": {}},
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "Example_metadata.json"
            result = generator._validate_and_save_schema(schema, output)
            self.assertEqual(result["sources"], schema["sources"])
            self.assertTrue(output.exists())

    def test_schema_validation_rejects_non_http_source(self):
        generator = SchemaGen.__new__(SchemaGen)
        schema = {
            "class_name": "Example",
            "description": "Example API",
            "sources": [{"url": "file:///tmp/reference"}],
            "tools": [
                {
                    "name": "get_item",
                    "description": "Get an item",
                    "input_schema": {"type": "object", "properties": {}, "required": []},
                    "output_schema": {"type": "object", "properties": {}},
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "HTTP"):
            generator._validate_and_save_schema(schema)


if __name__ == "__main__":
    unittest.main()
