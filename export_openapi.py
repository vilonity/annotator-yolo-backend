"""Dump the FastAPI OpenAPI schema to ./openapi.json without starting the server.

Used by the TS monorepo to regenerate `packages/yolo-types/src/schema.d.ts`
in environments where the live backend is not running (CI, fresh checkouts).

Usage:
    python export_openapi.py [output_path]
"""

import json
import sys
from pathlib import Path

from main import app


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "openapi.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(app.openapi(), indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
