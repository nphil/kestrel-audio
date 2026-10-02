"""`python -m kestrel_audio` starts the service."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn

from . import __version__, config
from .manager import Manager
from .server import create_app, load_or_create_key
from .store import Store


def build_app():
    cfg = config.load()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    store = Store(cfg.data_dir, cfg.inbox_dir, cfg.previews_dir, cfg.db_path)
    manager = Manager(cfg, store)
    key = load_or_create_key(cfg.key_path)
    app = create_app(cfg, store, manager, key)

    @asynccontextmanager
    async def lifespan(_app):
        await manager.start()
        logging.getLogger("kestrel_audio").info("kestrel-audio %s ready on :%d (data %s)", __version__, cfg.port, cfg.data_dir)
        yield
        await manager.stop()
        store.close()

    app.router.lifespan_context = lifespan
    return app, cfg


def main() -> None:
    app, cfg = build_app()
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
