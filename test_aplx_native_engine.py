import importlib.util
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("aplx_1_5", root / "aplx_1.5.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

ok, info = module.ensure_aplx_native_engine(path=str(root / "tmp_native_engine"), device="cpu")
assert ok, f"expected a built-in fallback engine, got: {info}"
print("native engine fallback ok", info)
