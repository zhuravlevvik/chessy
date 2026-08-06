#!/usr/bin/env python3
"""Export public Chess.com and Lichess games as source PGN files."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


USER_AGENT = "Chessy personal chess research project"


def request(url: str, accept: str = "application/json") -> bytes:
    headers = {"Accept": accept, "User-Agent": USER_AGENT}
    for attempt in range(5):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=60
            ) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 4:
                raise
            retry_after = int(error.headers.get("Retry-After", "2"))
            time.sleep(max(retry_after, 2**attempt))
    raise RuntimeError("request retry loop exhausted")


def export_chess_com(username: str, destination: Path) -> int:
    archives_url = f"https://api.chess.com/pub/player/{username}/games/archives"
    archive_urls = json.loads(request(archives_url))["archives"]
    games: list[str] = []

    for archive_url in archive_urls:
        payload = json.loads(request(archive_url))
        games.extend(game["pgn"].strip() for game in payload.get("games", []))

    destination.write_text("\n\n".join(games) + "\n", encoding="utf-8")
    return len(games)


def export_lichess(username: str, destination: Path) -> int:
    query = urllib.parse.urlencode(
        {
            "clocks": "true",
            "evals": "false",
            "opening": "true",
            "literate": "false",
        }
    )
    url = f"https://lichess.org/api/games/user/{username}?{query}"
    pgn = request(url, accept="application/x-chess-pgn").decode("utf-8")
    destination.write_text(pgn.rstrip() + "\n", encoding="utf-8")
    return sum(1 for line in pgn.splitlines() if line.startswith("[Event "))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chess-com", required=True, help="Chess.com username")
    parser.add_argument("--lichess", required=True, help="Lichess username")
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    chess_com_path = args.output / f"chess_com_{args.chess_com.lower()}.pgn"
    lichess_path = args.output / f"lichess_{args.lichess.lower()}.pgn"

    chess_com_games = export_chess_com(args.chess_com, chess_com_path)
    lichess_games = export_lichess(args.lichess, lichess_path)

    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "chess_com": {
                "username": args.chess_com,
                "games": chess_com_games,
                "file": chess_com_path.name,
            },
            "lichess": {
                "username": args.lichess,
                "games": lichess_games,
                "file": lichess_path.name,
            },
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Chess.com: {chess_com_games} games -> {chess_com_path}")
    print(f"Lichess: {lichess_games} games -> {lichess_path}")


if __name__ == "__main__":
    main()
