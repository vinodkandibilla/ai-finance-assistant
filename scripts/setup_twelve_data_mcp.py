"""Install the pinned official Twelve Data MCP server in its own environment."""

import subprocess
import sys
from pathlib import Path

REVISION = "d139a77ea0f7d40995e6979cd09c22dd142fbfe8"
ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / ".cache" / "twelvedata-mcp"


def main() -> None:
    if not SERVER.exists():
        subprocess.run(
            ["git", "clone", "https://github.com/twelvedata/mcp.git", str(SERVER)], check=True
        )
    revision = subprocess.check_output(
        ["git", "-C", str(SERVER), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        subprocess.run(["git", "-C", str(SERVER), "checkout", REVISION], check=True)
    subprocess.run([sys.executable, "-m", "venv", str(SERVER / ".venv")], check=True)
    subprocess.run(
        [
            str(SERVER / ".venv/bin/python"),
            "-m",
            "pip",
            "install",
            "mcp[cli]==1.30.0",
            "starlette==1.6.0",
            "httpx>=0.27,<1",
            "python-dotenv>=1,<2",
        ],
        check=True,
    )
    print("Twelve Data MCP server installed. Set TWELVE_DATA_API_KEY and restart the service.")


if __name__ == "__main__":
    main()
