from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run(
        "sub_translate.web.app:app",
        host="127.0.0.1",
        port=7860,
        reload=False,
        proxy_headers=False,
    )


if __name__ == "__main__":
    main()
