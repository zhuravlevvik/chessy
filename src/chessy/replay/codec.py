from __future__ import annotations
import numpy as np

def encode_board(board: np.ndarray) -> np.ndarray:
    """Losslessly pack a board119 float encoding into its replay representation."""
    array = np.asarray(board, dtype=np.float32)
    if array.shape != (119, 8, 8) or not np.isfinite(array).all(): raise ValueError("board must be finite [119,8,8]")
    packed = np.empty((119, 8, 8), dtype=np.uint8)
    binary = np.ones(119, dtype=bool); binary[117] = False
    if not np.isin(array[binary], (0.0, 1.0)).all(): raise ValueError("board119 binary planes must contain only 0 or 1")
    halfmoves = array[117] * 100.0
    # board119 is float32, so e.g. 0.03 * 100 can be 2.9999998.  The encoded
    # source is nevertheless an integer clock divided by 100.
    if (halfmoves < -1e-5).any() or (halfmoves > 100 + 1e-5).any() or not np.allclose(halfmoves, np.rint(halfmoves), atol=1e-5, rtol=0): raise ValueError("halfmove plane must be an exact integer / 100")
    packed[binary] = array[binary].astype(np.uint8); packed[117] = np.rint(halfmoves).astype(np.uint8)
    if not np.array_equal(decode_board(packed), array): raise ValueError("board119 replay packing is not lossless")
    return packed

def decode_board(packed: np.ndarray) -> np.ndarray:
    array = np.asarray(packed)
    if array.shape != (119, 8, 8) or array.dtype != np.uint8: raise ValueError("packed board must be uint8 [119,8,8]")
    result = array.astype(np.float32); result[117] /= 100.0
    return result
