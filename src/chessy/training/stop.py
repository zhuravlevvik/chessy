from __future__ import annotations
import signal
import threading
class StopController:
    def __init__(self) -> None: self.requested = threading.Event(); self.reason: str | None = None; self._old: dict[int, object] = {}; self._count=0
    def __enter__(self) -> "StopController":
        def handler(signum: int, frame: object) -> None:
            self._count += 1
            if self._count > 1: raise KeyboardInterrupt
            self.reason = signal.Signals(signum).name; self.requested.set()
        for sig in (signal.SIGINT, signal.SIGTERM): self._old[sig]=signal.getsignal(sig); signal.signal(sig,handler)
        return self
    def __exit__(self,*args: object) -> None:
        for sig, old in self._old.items(): signal.signal(sig,old)
