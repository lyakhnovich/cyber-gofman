import asyncio
import os
from dotenv import load_dotenv


def main() -> None:
    load_dotenv(override=True)
    # Before any import that may load huggingface_hub (e.g. sentence-transformers).
    # Avoids redirects to cas-bridge.xethub.hf.co when that host does not resolve.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from app.services.bot import run

    asyncio.run(run())


if __name__ == "__main__":
    main()
