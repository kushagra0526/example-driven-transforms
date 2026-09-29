import unittest

from src.prompts import parse_candidates


class ParseCandidates(unittest.TestCase):
    def test_plain_object(self):
        self.assertEqual(parse_candidates('{"candidates": ["a", "b"]}'), ["a", "b"])

    def test_wrapped_in_prose_and_fences(self):
        text = 'Here you go:\n```json\n{"candidates": ["def transform(x):\\n    return x"]}\n```'
        self.assertEqual(parse_candidates(text), ["def transform(x):\n    return x"])

    def test_code_fence_inside_candidate(self):
        text = '{"candidates": ["```python\\ndef transform(x):\\n    return x\\n```"]}'
        self.assertEqual(parse_candidates(text), ["def transform(x):\n    return x"])

    def test_truncated_reply_keeps_complete_candidates(self):
        text = '{"candidates": ["def transform(x):\\n    return x", "def transform(x):\\n    return x.up'
        self.assertEqual(parse_candidates(text), ["def transform(x):\n    return x"])

    def test_unusable(self):
        self.assertIsNone(parse_candidates(""))
        self.assertIsNone(parse_candidates('{"answer": 3}'))


if __name__ == "__main__":
    unittest.main()
