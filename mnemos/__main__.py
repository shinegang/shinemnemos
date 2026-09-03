# -*- coding: utf-8 -*-
"""Запуск MCP-сервера.

py -3.12 -m mnemos [--host ..] [--port ..] [--store nodes.json|blank]
                   [--plugins "context_engine,gates"] [--plugins-config plugins.json]
                   [--no-ground-by-default]

Плагины (контекст-модуль — отдельный плагин): env MNEMOS_PLUGINS или
plugins.json; --plugins задаёт явный список (пустая строка — без плагинов).

Стор: --store <путь> либо env MNEMOS_STORE. Значение "blank" (или
"blank:<путь>") — новая инстанция с ЧИСТЫМ графом: ни одного чужого узла,
инструменты без данных. Существующий файл никогда не перезаписывается.

Проход через граф (grounded) включён по умолчанию: без memory_ground_prepare
ответ агента помечается ungrounded. Выключить осознанно —
--no-ground-by-default или env MNEMOS_GROUND_BY_DEFAULT=0.
"""

import argparse
import sys

from .server import run


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mnemos", description="ShineMnemos MCP memory server (П1-П6)"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--store",
        default=None,
        help=(
            'путь к nodes.json; "blank" или "blank:<путь>" — чистый граф новой '
            "инстанции (по умолчанию env MNEMOS_STORE -> ./nodes.json)"
        ),
    )
    parser.add_argument(
        "--plugins",
        default=None,
        help=(
            'включённые плагины через запятую, напр. "context_engine,gates"; '
            'пустая строка или "none" — без плагинов (по умолчанию env '
            "MNEMOS_PLUGINS -> plugins.json -> дефолты: context_engine,gates)"
        ),
    )
    parser.add_argument(
        "--plugins-config",
        default=None,
        help='путь к plugins.json ({"enabled": ["context_engine"]})',
    )
    parser.add_argument(
        "--ground-by-default",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "обязательный проход через граф перед ответом (по умолчанию ВКЛЮЧЁН; "
            "--no-ground-by-default выключает, как и env MNEMOS_GROUND_BY_DEFAULT=0)"
        ),
    )
    args = parser.parse_args()
    try:
        run(
            host=args.host,
            port=args.port,
            store_path=args.store,
            plugins=args.plugins,
            plugins_config=args.plugins_config,
            ground_by_default=args.ground_by_default,
        )
    except (ValueError, OSError) as exc:
        # Ошибка конфигурации (непустой blank-граф, каталог вместо файла, нет
        # прав, занятый порт) — это сообщение оператору, а не трассировка на
        # 12 строк, в которой само сообщение теряется последней строкой.
        print(f"mnemos: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
