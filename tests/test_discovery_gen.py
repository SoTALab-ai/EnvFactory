import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.gen.discovery_gen import DISCOVERY_RESEARCH_TOOLS, DiscoveryGen


VALID_DISCOVERY = {
    "server_name": "ExampleDocs",
    "slug": "example_docs",
    "category": "productivity",
    "description": "Read and update example documents.",
    "sources": [
        {
            "title": "Official API",
            "url": "https://developer.example.com/docs",
            "used_for": "Endpoint schemas",
        }
    ],
    "research_markdown": "# Example Docs\n\nOfficial API research.",
    "schema_sketch": '''# Data Source: https://developer.example.com/docs
# Server: ExampleDocs

def search_documents(query: str, limit: int = 20) -> list:
    """Search documents and return document identifiers and titles."""
    pass

def get_document(document_id: str) -> dict:
    """Get one document and return its identifier, title, and content."""
    pass

def update_document(document_id: str, content: str) -> dict:
    """Update one document and return its identifier and new version."""
    pass
''',
}


class DiscoveryValidationTest(unittest.TestCase):
    def test_agent_exposes_search_and_reader(self):
        self.assertEqual(
            [tool.name for tool in DISCOVERY_RESEARCH_TOOLS],
            ["search_web", "read_webpage"],
        )
        with patch.object(DiscoveryGen, "get_model", return_value="test-model"):
            generator = DiscoveryGen()
        self.assertEqual(
            [tool.name for tool in generator.discovery_agent.tools],
            ["search_web", "read_webpage"],
        )

    def test_valid_discovery_is_saved(self):
        result = DiscoveryGen.validate_discovery(VALID_DISCOVERY.copy())
        with tempfile.TemporaryDirectory() as tmpdir:
            DiscoveryGen.save_discovery(result, output_root=tmpdir)
            self.assertTrue(Path(result.research_path).exists())
            self.assertTrue(Path(result.sketch_path).exists())
            self.assertIn("def search_documents", Path(result.sketch_path).read_text())

    def test_executable_sketch_code_is_rejected(self):
        invalid = VALID_DISCOVERY.copy()
        invalid["schema_sketch"] = '''
import requests

def one(a: str) -> dict:
    pass

def two(a: str) -> dict:
    pass

def three(a: str) -> dict:
    pass
'''
        with self.assertRaisesRegex(ValueError, "only top-level function"):
            DiscoveryGen.validate_discovery(invalid)

    def test_tool_implementation_is_rejected(self):
        invalid = VALID_DISCOVERY.copy()
        invalid["schema_sketch"] = VALID_DISCOVERY["schema_sketch"].replace(
            "    pass\n", "    return {}\n", 1
        )
        with self.assertRaisesRegex(ValueError, "docstring and pass"):
            DiscoveryGen.validate_discovery(invalid)

    def test_fewer_than_three_tools_is_rejected(self):
        invalid = VALID_DISCOVERY.copy()
        invalid["schema_sketch"] = '''
def get_document(document_id: str) -> dict:
    """Get one document."""
    pass
'''
        with self.assertRaisesRegex(ValueError, "between 3 and 5"):
            DiscoveryGen.validate_discovery(invalid)


if __name__ == "__main__":
    unittest.main()
