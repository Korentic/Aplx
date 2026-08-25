import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
MODULE_PATH = REPO_ROOT / "aplx_llm.py"


class DataPathLoaderTests(unittest.TestCase):
    def test_resolves_relative_data_dir_from_script_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            script_dir = temp_path / "fake_project"
            script_dir.mkdir()
            data_dir = script_dir / "data"
            data_dir.mkdir()
            (data_dir / "sample.txt").write_text("hello world", encoding="utf-8")

            spec = importlib.util.spec_from_file_location("aplx_llm_under_test", MODULE_PATH)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            module.__file__ = str(script_dir / "aplx_llm.py")

            original_cwd = Path.cwd()
            try:
                os.chdir(temp_path)
                resolved = module._resolve_data_path("./data")
            finally:
                os.chdir(original_cwd)

            self.assertEqual(resolved, data_dir.resolve())
            self.assertEqual(module._load_texts_from_path("./data"), ["hello world"])

    def test_accepts_markdown_files_in_data_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            script_dir = temp_path / "fake_project"
            script_dir.mkdir()
            data_dir = script_dir / "data"
            data_dir.mkdir()
            (data_dir / "sample.md").write_text("hello world from markdown", encoding="utf-8")

            spec = importlib.util.spec_from_file_location("aplx_llm_under_test", MODULE_PATH)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            module.__file__ = str(script_dir / "aplx_llm.py")

            original_cwd = Path.cwd()
            try:
                os.chdir(temp_path)
                texts = module._load_texts_from_path("./data")
            finally:
                os.chdir(original_cwd)

            self.assertEqual(texts, ["hello world from markdown"])

    def test_falls_back_to_nearby_text_files_when_data_directory_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "notes.md").write_text("fallback content", encoding="utf-8")

            spec = importlib.util.spec_from_file_location("aplx_llm_under_test", MODULE_PATH)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            original_cwd = Path.cwd()
            try:
                os.chdir(temp_path)
                texts = module._load_texts_from_path("./data")
            finally:
                os.chdir(original_cwd)

            self.assertEqual(texts, ["fallback content"])

    def test_text_dataset_creates_a_sample_for_short_corpus(self):
        spec = importlib.util.spec_from_file_location("aplx_llm_under_test", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        class StubTokenizer:
            def encode(self, text, add_bos=False, add_eos=False):
                return [1, 2, 3]

        dataset = module.TextDataset(["short text"], StubTokenizer(), seq_len=64)
        self.assertEqual(len(dataset), 1)
        x, y = dataset[0]
        self.assertEqual(x.shape[0], y.shape[0])
        self.assertGreaterEqual(x.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
