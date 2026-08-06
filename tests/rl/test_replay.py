from __future__ import annotations

import hashlib
from pathlib import Path

import chess

from chessy.chess import ChessEnvironment
from chessy.encoding import encode_board, encode_move
from chessy.replay.manifest import load_manifest, write_manifest
from chessy.replay.segment import ReplaySample, SealedGame, verify_segment, write_segment


def _game() -> SealedGame:
    environment=ChessEnvironment(); board=environment.board; moves=tuple(board.legal_moves); actions=tuple(encode_move(board,move) for move in moves)
    sample=ReplaySample(encode_board(environment.history()),actions,(1,)+((0,)*(len(actions)-1)),actions[0],1,0,0,0)
    return SealedGame("game-0",0,0,"full","initial",board.fen(),"1/2-1/2","max-plies",(sample,),"[Result \"1/2-1/2\"]\n\n1/2-1/2",{})


def test_replay_round_trip_preserves_full_legal_mask(tmp_path: Path) -> None:
    segment=write_segment(tmp_path,generation=0,ordinal=0,games=[_game()],run_id="run",model_checksum="model")
    checked=verify_segment(segment); arrays=checked["arrays"]
    assert arrays["policy_actions"].size==chess.Board().legal_moves.count()
    assert (arrays["policy_visits"]==0).sum()>0
    manifest=write_manifest(tmp_path,run_id="run",generation=0,segments=[segment],active_max_samples=10)
    assert load_manifest(manifest.path).fingerprint==manifest.fingerprint


def test_manifest_pins_exact_segment_payload(tmp_path: Path) -> None:
    segment=write_segment(tmp_path,generation=0,ordinal=0,games=[_game()],run_id="run",model_checksum="model")
    manifest=write_manifest(tmp_path,run_id="run",generation=0,segments=[segment],active_max_samples=10)
    manifest_file=segment/"manifest.json"; manifest_file.write_bytes(manifest_file.read_bytes()+b" ")
    lines=[]
    for line in (segment/"checksums.sha256").read_text().splitlines():
        digest,name=line.split("  ",1)
        if name=="manifest.json": digest=hashlib.sha256(manifest_file.read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}\n")
    (segment/"checksums.sha256").write_text("".join(lines))
    verify_segment(segment)
    try:
        load_manifest(manifest.path)
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("manifest accepted a different internally valid segment")
