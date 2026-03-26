# surface_ws.py
import asyncio
import threading
import time
import traceback
from typing import Optional

import websockets


class SurfaceWSClient:
    """
    Background WebSocket client.
    - Runs asyncio in a daemon thread.
    - Main thread calls set_ml(ml) to update latest value.
    - This client periodically sends the latest ml (or only when changed).
    """

    def __init__(
        self,
        uri: str = "ws://localhost:9090",
        client_id: str = "python_surface_height",
        send_hz: float = 10.0,         # send rate
        only_send_on_change: bool = True,
        change_eps: float = 0.05,      # ml change threshold
    ):
        self.uri = uri
        self.client_id = client_id
        self.send_hz = max(0.2, float(send_hz))
        self.only_send_on_change = bool(only_send_on_change)
        self.change_eps = float(change_eps)

        self._lock = threading.Lock()
        self._latest_ml: Optional[float] = None
        self._stop = threading.Event()

        self._thread = threading.Thread(target=self._thread_main, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        # thread is daemon; no hard join required, but we can try a short join
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass

    def set_ml(self, ml: Optional[float]):
        """Called by main loop (non-async)."""
        with self._lock:
            self._latest_ml = None if ml is None else float(ml)

    def _get_latest_ml(self) -> Optional[float]:
        with self._lock:
            return self._latest_ml

    def _thread_main(self):
        try:
            asyncio.run(self._run())
        except Exception:
            print("[SurfaceWS] thread crashed:\n", traceback.format_exc())

    async def _run(self):
        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    self.uri,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=5,
                ) as ws:
                    backoff = 1.0
                    # handshake
                    await ws.send(self.client_id)
                    await asyncio.sleep(0.05)

                    print(f"[SurfaceWS] connected: {self.uri}")

                    last_sent: Optional[float] = None
                    dt = 1.0 / self.send_hz

                    while not self._stop.is_set():
                        ml = self._get_latest_ml()

                        if ml is not None:
                            should_send = True
                            if self.only_send_on_change and (last_sent is not None):
                                should_send = abs(ml - last_sent) >= self.change_eps

                            if should_send:
                                await ws.send(f"{ml:.3f}")
                                last_sent = ml

                        # (optional) receive messages if server sends any
                        # Use timeout so we can still send periodically
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=dt)
                            # 你不需要的話可以註解掉
                            # print("[SurfaceWS] recv:", msg)
                        except asyncio.TimeoutError:
                            pass

                        await asyncio.sleep(0)  # yield

            except Exception as e:
                print(f"[SurfaceWS] connection error: {e}")
                # exponential backoff (max 10s)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 10.0)
