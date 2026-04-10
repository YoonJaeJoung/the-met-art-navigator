"""
Training Telemetry Callback.

PyTorch Lightning Callback that streams training metrics (loss, epoch, step)
over a WebSocket server for real-time frontend visualization.

The WebSocket server runs on port 8765 by default.
"""

import asyncio
import json
import threading
import csv
import os
from pathlib import Path
from typing import Any

import lightning as L
import websockets


class TelemetryCallback(L.Callback):
    """Streams training metrics to connected WebSocket clients."""

    _shared_clients: set = set()
    _shared_loop: asyncio.AbstractEventLoop | None = None
    _server_thread: threading.Thread | None = None

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        super().__init__()
        self.host = host
        self.port = port

    def _start_server(self):
        """Start the WebSocket server in a background thread."""
        if TelemetryCallback._server_thread is not None and TelemetryCallback._server_thread.is_alive():
            return  # Server already running

        TelemetryCallback._shared_loop = asyncio.new_event_loop()

        async def handler(websocket, path=None):
            TelemetryCallback._shared_clients.add(websocket)
            try:
                await websocket.wait_closed()
            finally:
                TelemetryCallback._shared_clients.discard(websocket)

        async def run():
            try:
                async with websockets.serve(handler, self.host, self.port):
                    await asyncio.Future()  # Run forever
            except OSError:
                pass

        TelemetryCallback._server_thread = threading.Thread(
            target=TelemetryCallback._shared_loop.run_until_complete, args=(run(),), daemon=True
        )
        TelemetryCallback._server_thread.start()

    def _broadcast(self, data: dict):
        """Send data to all connected WebSocket clients."""
        if not TelemetryCallback._shared_clients or TelemetryCallback._shared_loop is None:
            return
        message = json.dumps(data)
        for client in list(TelemetryCallback._shared_clients):
            try:
                asyncio.run_coroutine_threadsafe(client.send(message), TelemetryCallback._shared_loop)
            except Exception:
                TelemetryCallback._shared_clients.discard(client)

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        print(f"[Telemetry] Starting WebSocket server on ws://{self.host}:{self.port}")
        self._start_server()
        self._broadcast({
            "type": "training_start",
            "max_epochs": trainer.max_epochs,
        })

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule,
                           outputs: Any, batch: Any, batch_idx: int):
        metrics = {
            "type": "train_step",
            "epoch": trainer.current_epoch,
            "step": trainer.global_step,
            "loss": float(trainer.callback_metrics.get("train_loss", 0)),
            "temperature": float(trainer.callback_metrics.get("temperature", 0)),
        }
        self._broadcast(metrics)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        train_loss = float(trainer.callback_metrics.get("train_loss", 0))
        val_loss = float(trainer.callback_metrics.get("val_loss", 0))
        epoch = trainer.current_epoch

        metrics = {
            "type": "epoch_end",
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        self._broadcast(metrics)

        # Write to a clean summary CSV for easy analysis
        summary_path = Path("data/metrics_summary.csv")
        file_exists = summary_path.exists()
        
        # Ensure data directory exists
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['epoch', 'train_loss', 'val_loss'])
            writer.writerow([epoch, train_loss, val_loss])

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self._broadcast({
            "type": "training_complete",
            "final_epoch": trainer.current_epoch,
        })
