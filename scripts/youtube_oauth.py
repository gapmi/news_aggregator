from __future__ import annotations

import argparse
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow


SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLIENT_SECRET = (
    PROJECT_ROOT / "secrets" / "youtube_client_secret.json"
)
DEFAULT_TOKEN_FILE = PROJECT_ROOT / "secrets" / "youtube_token.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Authorize YouTube upload OAuth and save refresh token."
    )
    parser.add_argument(
        "--client-secret",
        type=Path,
        default=DEFAULT_CLIENT_SECRET,
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        default=DEFAULT_TOKEN_FILE,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8081,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.client_secret.is_file():
        raise RuntimeError(
            f"OAuth client secret was not found: {args.client_secret}"
        )

    args.token_file.parent.mkdir(parents=True, exist_ok=True)

    flow = InstalledAppFlow.from_client_secrets_file(
        str(args.client_secret),
        scopes=SCOPES,
    )

    credentials = flow.run_local_server(
        host="0.0.0.0",
        port=args.port,
        open_browser=False,
        authorization_prompt_message=(
            "\nOpen this URL in your local browser:\n\n{url}\n"
        ),
        success_message=(
            "YouTube OAuth authorization completed. "
            "You may close this browser tab."
        ),
    )

    args.token_file.write_text(
        credentials.to_json(),
        encoding="utf-8",
    )

    args.token_file.chmod(0o600)

    print(f"OAuth token saved: {args.token_file}")
    print("YouTube OAuth authorization: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())