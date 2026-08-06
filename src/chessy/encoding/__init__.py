"""Stable board and action encodings used by Chessy."""

from chessy.encoding.action import (
    ACTION_ENCODING_VERSION,
    ACTION_PLANES,
    ACTION_SIZE,
    decode_action,
    encode_move,
    legal_action_mask,
)
from chessy.encoding.board import (
    BOARD_ENCODING_VERSION,
    BOARD_PLANES,
    HISTORY_LENGTH,
    encode_board,
)

__all__ = [
    "ACTION_ENCODING_VERSION",
    "ACTION_PLANES",
    "ACTION_SIZE",
    "BOARD_ENCODING_VERSION",
    "BOARD_PLANES",
    "HISTORY_LENGTH",
    "decode_action",
    "encode_board",
    "encode_move",
    "legal_action_mask",
]
