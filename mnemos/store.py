# -*- coding: utf-8 -*-
"""ShineMnemos: хранилище узлов (nodes.json).

Простое JSON-хранилище: dict id -> узел, атомарная запись
(tmp-файл + os.replace). Скелет на будущее — место, куда потом ляжет
граф и векторный индекс, без внешних зависимостей.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from . import qmem
from .model import (
    DEFAULT_HALF_LIFE_HOURS,
    WEIGHT_MIN,
    MemoryNode,
    _parse_iso,
    now_iso,
    weight_floor,
)

logger = logging.getLogger(__name__)

# Веса полей для подстрочного поиска (substring rank).
_FIELD_WEIGHTS = (("claim", 3), ("source", 2), ("context", 1))
# Ф1 02.09 (квантование): «хвост ключей» сжатого уровня ищется с весом claim —
# это outlier-токены узла (порты, суммы, имена), то, что спрашивают точно.
_QFIELD_WEIGHTS = _FIELD_WEIGHTS + (("keys", 3),)
# Ф1 02.09 (гейт R4): бонус за частоту термина в claim. Узел, где запрос
# встречается в claim трижды («ТОРГУЕТ ТОЛЬКО АЛИСА … Алиса … Алиса»), —
# про него, а не просто упоминает. 0.5*ln(tf): tf=1 -> 0, tf=3 -> +0.55 —
# меньше веса одного поля, переставляет только равных по полям.
_TF_BONUS = 0.5
_TF_MAX = 1.0    # потолок: короткий запрос («о») не должен перебивать веса полей


def _canonical_key(text: Any) -> str:
    """Канонический ключ точного поиска: strip + casefold.

    Это и есть «точный ключ» хеш-индекса: два узла считаются совпадающими,
    если их claim идентичен после приведения регистра и обрезки пробелов.
    Сверка по точному ключу гарантирует 100% точность (без substring-ложных
    срабатываний: 'BTC' != 'BTC вырос').
    """
    return (text or "").strip().casefold()

# бенчмарк-фикс 26.08: семантический слой (закрывает white spot №4 —
# «семантика 0/25» против Chroma/FAISS/Mem0). Модель — multilingual
# (RU + EN, без префиксов query:/passage:, в отличие от e5-моделей),
# та же, что гоняли в D:\mnemos-bench по RU (recheck_ru.py). Можно
# переопределить через переменную окружения MNEMOS_EMBED_MODEL,
# например BAAI/bge-small-en-v1.5 (быстрее, но только EN).
_SEMANTIC_MODEL = os.environ.get(
    "MNEMOS_EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# Сообщение об ошибке, когда fastembed не установлен (честный текст
# для MCP-инструмента memory_search mode="semantic").
_FASTEMBED_HINT = (
    "семантический поиск требует пакет fastembed — установите его: "
    "pip install fastembed"
)


def _decay_ref(d: Dict[str, Any]) -> Optional[datetime]:
    """Момент, от которого у узла осталось НЕприменённое затухание.

    Позже из last_used и decayed_at (см. MemoryNode.decay_ref). Нужен и при
    записи (store.decay), и при чтении (_node_decayed_weight): иначе одно и
    то же затухание учитывается дважды — один раз в сохранённом weight,
    второй раз поверх него при ранжировании.
    """
    last = _parse_iso(str(d.get("last_used") or ""))
    done = _parse_iso(str(d.get("decayed_at") or "")) if d.get("decayed_at") else None
    if last is None:
        return done
    if done is None:
        return last
    return max(last, done)


def _dump_by_line(nodes: Dict[str, Dict[str, Any]], f) -> None:
    """Дамп стора: компактный JSON, но ОДНА СТРОКА НА УЗЕЛ.

    Фикс перф 01.09 (аудит §5.8). Было `json.dump(indent=2)`: на 3121 узле
    это 141.7 мс CPU и 4.33 МБ на КАЖДУЮ запись (замер 01.09). Стало
    компактно — 38 мс и 3.24 МБ (×3.7 быстрее, −25% байт).

    Почему не голый `indent=None`: стор коммитится в git каждые 5 минут
    (`/opt/mnemos/sync-mnemos.sh`), а однострочный файл даёт diff «весь файл
    изменился» на каждую запись. Построчная раскладка сохраняет и скорость,
    и осмысленный git-diff (одна строка = один узел).

    Результат — валидный JSON-объект, читается обычным json.load.
    """
    f.write("{\n")
    first = True
    for nid, node in nodes.items():
        if not first:
            f.write(",\n")
        first = False
        f.write(json.dumps(str(nid), ensure_ascii=False))
        f.write(": ")
        f.write(json.dumps(node, ensure_ascii=False, separators=(",", ":")))
    f.write("\n}\n")


def _cosine(a, b) -> float:
    """Косинусная близость двух векторов (списки float)."""
    import math

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)

# Brick-2 (пластичность/структурная рекурсия): «граф в узле» — узел может
# содержать дочерние узлы. Глубина ограничена, циклы запрещены: рекурсия
# структурная и конечная, как иерархии в мозге, а не бесконечная петля.
MAX_DEPTH = 5


# --- бланковый (чистый) граф: продаём инструменты, а не данные ---------------
# Приказ Ильи 03.09: клиентская инстанция стартует с ПУСТЫМ графом. Ни одного
# нашего узла в дистрибутиве — наш боевой стор остаётся внутренним. Поэтому
# шаблон пустого графа лежит В ПАКЕТЕ (едет с pip install), а не в каталоге
# data/ репозитория, где живёт наш рабочий nodes.json.
BLANK_KEYWORD = "blank"
BLANK_TEMPLATE_NAME = "nodes.blank.json"
DEFAULT_STORE_NAME = "nodes.json"
STORE_ENV = "MNEMOS_STORE"          # спецификация стора: путь или "blank"
STORE_PATH_ENV = "MNEMOS_STORE_PATH"  # куда класть граф, если спецификация "blank"


def blank_template_path() -> Path:
    """Шаблон пустого графа, лежащий в самом пакете mnemos."""
    return Path(__file__).resolve().parent / "data" / BLANK_TEMPLATE_NAME


def blank_template_text() -> str:
    """Содержимое шаблона пустого графа, с проверкой, что он и правда пуст.

    Непустой шаблон — ошибка запуска, а не «ну и ладно»: это ровно тот случай,
    когда наши узлы утекли в дистрибутив, и молчать про него нельзя. Проверка
    идёт ДО создания файла, чтобы не осталось полуфабриката.
    """
    template = blank_template_path()
    if not template.exists():
        return "{}\n"
    raw = template.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{template}: битый шаблон пустого графа ({exc})") from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"{template}: шаблон пустого графа должен быть JSON-объектом, "
            f"а там {type(data).__name__}"
        )
    if data:
        raise ValueError(
            f"{template}: шаблон пустого графа не пуст ({len(data)} узлов) — "
            "в дистрибутив утекли чужие данные"
        )
    return raw


def create_blank_store(path: Union[str, Path]) -> bool:
    """Создать пустой граф по шаблону. Существующий файл НЕ трогает.

    Возвращает True, если файл создан этим вызовом. Создание идёт через
    O_CREAT|O_EXCL, а не через `if not exists(): write()`: между проверкой и
    записью файл может появиться (два процесса стартуют разом, systemd
    Restart=always), и обычная запись затёрла бы чужой уже живой граф.

    Не-файл на этом пути — ошибка запуска (баг-хант 03.09, Д2/Д4). Каталог и
    битая символьная ссылка тоже отвечают на O_EXCL «уже существует», и без
    разбора причины сервер поднимался как ни в чём не бывало: узлы копились в
    памяти, а первая же запись падала IsADirectoryError — клиент терял всё, что
    «сохранил». Лучше не стартовать, чем стартовать памятью, которая не пишется.
    """
    path = Path(path)
    if path.is_dir():
        raise ValueError(
            f"{path}: это каталог, а не файл графа — укажите путь до файла, "
            f"например blank:{path / DEFAULT_STORE_NAME}"
        )
    text = blank_template_text()
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        if os.path.lexists(path) and not path.exists():
            raise ValueError(
                f"{path}: битая символьная ссылка — ведёт в никуда. Почините "
                "ссылку или укажите другой путь; молча подменять её файлом "
                "нельзя, цель ссылки так и не появится"
            ) from None
        return False
    except IsADirectoryError as exc:  # pragma: no cover — перехвачено is_dir()
        raise ValueError(f"{path}: это каталог, а не файл графа") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    return True


def assert_blank_target_empty(path: Union[str, Path]) -> int:
    """Проверить, что уже существующий blank-граф действительно пуст.

    Баг-хант 03.09 (Д1): `--store blank` в каталоге, где уже лежит чужой
    nodes.json, молча поднимал сервер НА ЧУЖОМ ГРАФЕ — при том что и приказ, и
    README обещают клиенту пустоту. Затирать файл нельзя (это данные), поэтому
    единственный честный выход — не стартовать и сказать, что делать.
    """
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8").strip()
        data = json.loads(raw) if raw else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{path}: запрошен --store blank, но файл на этом пути не читается "
            f"как граф ({exc}). Уберите его или укажите другой путь"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: запрошен --store blank, но на этом пути не граф "
            f"({type(data).__name__}). Укажите другой путь"
        )
    if data:
        raise ValueError(
            f"{path}: запрошен --store blank, а в файле уже {len(data)} узлов. "
            "blank обещает пустой граф и никогда не перезаписывает данные. "
            f"Если это ваш граф — запускайте с --store {path} (без blank); "
            "если нет — укажите другой путь: --store blank:<путь> или "
            f"{STORE_PATH_ENV}=<путь>"
        )
    return len(data)


def blank_target(
    spec: Optional[str] = None, env: Optional[Dict[str, str]] = None
) -> Optional[Path]:
    """Путь для режима blank, либо None, если это обычный путь к графу.

    Отдельная функция, чтобы `run()` печатал про blank ровно тогда, когда режим
    действительно blank: проверка вида `store_path.startswith("blank")` считала
    режимом blank обычный путь `blankgraph.json` и врала оператору про
    состояние файла (баг-хант 03.09, Д3).
    """
    env = os.environ if env is None else env
    raw = (spec or "").strip() or str(env.get(STORE_ENV) or "").strip()
    if not raw:
        return None
    head, sep, tail = raw.partition(":")
    # "C:\mnemos\nodes.json" тоже содержит ':' — потому сверяем ровно head
    if head.strip().lower() != BLANK_KEYWORD:
        return None
    target = (tail.strip() if sep else "") or \
        str(env.get(STORE_PATH_ENV) or "").strip() or DEFAULT_STORE_NAME
    return Path(target)


def resolve_store_path(
    spec: Optional[str] = None, env: Optional[Dict[str, str]] = None
) -> tuple:
    """Куда класть граф и надо ли создать его пустым -> (Path, created_blank).

    spec (аргумент --store), по убыванию приоритета:
      None/""          -> env MNEMOS_STORE -> ./nodes.json;
      "blank"          -> пустой граф по env MNEMOS_STORE_PATH или ./nodes.json;
      "blank:<путь>"   -> пустой граф по указанному пути;
      любое другое     -> это и есть путь к графу (как было всегда).

    Два инварианта blank, и оба обязательны:
      * НИКОГДА не перезаписывать существующий файл — это чьи-то данные;
      * граф после старта ДЕЙСТВИТЕЛЬНО пуст — это обещание приказа и README.
    Когда они сталкиваются (на пути уже лежит непустой граф), сервер не
    стартует: молча отдать клиенту чужие узлы хуже, чем громко не подняться.
    Пустой файл на этом пути — не конфликт, он и так пуст: рестарт инстанции,
    которая ещё ничего не запомнила, проходит спокойно.
    """
    env = os.environ if env is None else env
    target = blank_target(spec, env)
    if target is None:
        raw = (spec or "").strip() or str(env.get(STORE_ENV) or "").strip()
        return Path(raw or DEFAULT_STORE_NAME), False
    if create_blank_store(target):
        return target, True
    assert_blank_target_empty(target)
    return target, False


class Store:
    """JSON-хранилище узлов памяти с подстрочным поиском и пластичностью."""

    def __init__(self, path: Union[str, Path], use_hash_index: bool = True) -> None:
        self.path = Path(path)
        self._nodes: Dict[str, Dict[str, Any]] = {}
        # фикс аудита 26.08: RLock — одна блокировка на все мутации + _save;
        # ThreadingHTTPServer обслуживает запросы в потоках, и без лока
        # read-modify-write в _save терял узлы (B1: 8×20 add → выжило 40/160).
        self._lock = threading.RLock()
        # бенчмарк-фикс 26.08: кэш эмбеддингов id -> (claim, вектор).
        # Сравнение по claim: при изменении узла (update/rewrite) вектор
        # пересчитывается лениво в search_semantic; удалённые вычищаются там же.
        self._embeddings: Dict[str, tuple] = {}
        self._embedder: Any = None  # ленивый fastembed.TextEmbedding
        # хеш-индекс точного поиска (фикс перф 27.08): канонический claim ->
        # список id узлов. Полная загрузка при старте, обновление при каждой
        # записи (add/add_many/update/rewrite/delete/add_child); при промахе —
        # fallback-скан авторитетного dict с самолечением индекса (100% точность
        # даже при рассинхроне). use_hash_index=False — прежнее поведение без
        # индекса (нулевой оверхед памяти).
        self._use_hash_index = bool(use_hash_index)
        self._exact_index: Dict[str, List[str]] = {}
        self._version = 0          # растёт при каждом _save (кэш budget-поиска)
        self._budget_cache = None  # (version, BudgetSearch)
        # Ф1 02.09 (квантование): idf-кэш для сборки уровней L1/L2 и флаг
        # «usage/level изменились в памяти, но ещё не записаны» — попадания в
        # поиск не пишут файл сами (иначе каждый memory_search = дамп + коммит
        # автосинка), их подхватывает ближайшая запись (add/decay/...).
        self._salience: Optional[qmem.Salience] = None
        self._salience_n = 0
        self._usage_dirty = False
        self._dirty_hits = 0
        self._load()
        self._build_exact_index()

    # -- персистентность ------------------------------------------------------
    def _load(self) -> None:
        with self._lock:
            self._nodes = {}
            if not self.path.exists():
                return
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                # фикс аудита 26.08: битый nodes.json НЕ затираем молча —
                # прячем в nodes.json.corrupt-<timestamp> и стартуем с пустой
                # памятью, чтобы следующий _save не уничтожил данные навсегда.
                backup = self._quarantine_corrupt()
                logger.warning(
                    "Store %s: битый nodes.json (%s) — сохранён как %s, "
                    "старт с пустой памятью",
                    self.path, exc, backup,
                )
                return
            except OSError as exc:
                # файл есть, но не читается (заблокирован/нет прав) — файл не
                # трогаем, только предупреждаем.
                logger.warning(
                    "Store %s: не удалось прочитать nodes.json (%s) — "
                    "старт с пустой памятью, файл оставлен на месте",
                    self.path, exc,
                )
                return
            if not isinstance(data, dict):
                # валидный JSON, но не объект {id: узел} — формат битый,
                # данные тоже прячем, а не затираем.
                backup = self._quarantine_corrupt()
                logger.warning(
                    "Store %s: nodes.json имеет неожиданный формат (%s) — "
                    "сохранён как %s, старт с пустой памятью",
                    self.path, type(data).__name__, backup,
                )
                return
            self._nodes = {
                str(k): v for k, v in data.items() if isinstance(v, dict)
            }
            for d in self._nodes.values():
                qmem.normalize(d)  # Ф1: level/usage всегда есть и валидны

    def _quarantine_corrupt(self) -> Optional[Path]:
        """Фикс аудита 26.08: переименовывает битый nodes.json в
        nodes.json.corrupt-<timestamp>, чтобы _save не затёр повреждённые
        данные. Возвращает путь к бэкапу (None — если не удалось)."""
        # %f — микросекунды; без ':' в имени (невалидны для Windows).
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        backup = self.path.with_name(f"{self.path.name}.corrupt-{ts}")
        try:
            self.path.replace(backup)  # атомарно, работает и на Windows
        except OSError as exc:
            logger.warning(
                "Store %s: не удалось сохранить битый файл как %s (%s)",
                self.path, backup, exc,
            )
            return None
        return backup

    def _save(self) -> None:
        # вызывается только под self._lock (все мутации держат блокировку);
        # повторный вход безопасен — RLock.
        with self._lock:
            # версия стора: любая запись инвалидирует кэш BudgetSearch
            self._version = getattr(self, "_version", 0) + 1
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                dir=str(self.path.parent), prefix=".nodes-", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    _dump_by_line(self._nodes, f)
                os.replace(tmp, self.path)
                # usage/level ушли на диск вместе с этой записью (флаг — только
                # после успешного replace, иначе OSError потерял бы накопленное)
                self._usage_dirty = False
                self._dirty_hits = 0
            finally:
                if os.path.exists(tmp):  # pragma: no cover
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass

    # -- хеш-индекс точного поиска (фикс перф 27.08) --------------------------
    # Индекс лениво синхронизирован с авторитетным dict self._nodes: полная
    # пересборка при старте (reindex), точечные обновления при каждой записи.
    # Все обращения — строго под self._lock (тот же лок, что и у _nodes).
    # При рассинхроне (промах индекса при наличии узла) search_exact делает
    # fallback-скан по _nodes и сам чинит запись — точность остаётся 100%.
    def reindex(self) -> None:
        """Полная пересборка индекса из авторитетного dict _nodes.

        Нужна только если _nodes менялся в обход публичного API (например,
        ручная правка self._nodes) или файл nodes.json был изменён другим
        процессом после старта. Обычный жизненный цикл индекса — автоматический.
        """
        with self._lock:
            self._build_exact_index()

    def _build_exact_index(self) -> None:
        """Полная пересборка индекса (старт / reindex). Только под локом."""
        self._exact_index = {}
        for nid, node in self._nodes.items():
            key = _canonical_key(node.get("claim"))
            if key:
                self._exact_index.setdefault(key, []).append(nid)

    def _index_add(self, node: Dict[str, Any]) -> None:
        """Точечное добавление узла в индекс. Только под локом."""
        key = _canonical_key(node.get("claim"))
        if key:
            self._exact_index.setdefault(key, []).append(node["id"])

    def _index_remove(self, node_id: str, old_claim: Any) -> None:
        """Точечное удаление узла из индекса по его (старому) claim."""
        key = _canonical_key(old_claim)
        if key:
            bucket = self._exact_index.get(key)
            if bucket and node_id in bucket:
                bucket.remove(node_id)
                if not bucket:
                    del self._exact_index[key]

    # -- CRUD ----------------------------------------------------------------
    def add(self, node: Union[MemoryNode, Dict[str, Any]]) -> Dict[str, Any]:
        d = node.to_dict() if isinstance(node, MemoryNode) else dict(node)
        MemoryNode.from_dict(d)  # валидация схемы
        with self._lock:
            if d["id"] in self._nodes:
                # фикс аудита 26.08: дубликат id больше не перезаписывает узел
                # молча (B4 — тихое разрушение данных); для замены есть
                # update(), для принудительной перезаписи — delete + add.
                raise ValueError(
                    f"узел {d['id']} уже существует — add только для новых узлов"
                )
            self._refresh_levels(d)
            self._nodes[d["id"]] = d
            if self._use_hash_index:
                self._index_add(d)
            self._save()
        return d

    def add_many(self, nodes: List[Union[MemoryNode, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Бенчмарк-фикс 26.08: батч-запись — валидация ВСЕХ, добавление
        ВСЕХ и ОДИН _save в конце (снимает O(N²): раньше каждый add
        переписывал весь nodes.json). Атомарно: при любой ошибке (невалидный
        узел, дубликат id внутри батча или в хранилище) НЕ добавляется ни
        один узел. Возвращает список добавленных узлов (dict)."""
        # этап 1: валидация схемы и дубликатов id внутри батча — до блокировки
        prepared: List[Dict[str, Any]] = []
        seen: set = set()
        for n in nodes:
            d = n.to_dict() if isinstance(n, MemoryNode) else dict(n)
            MemoryNode.from_dict(d)  # валидация схемы
            if d["id"] in seen:
                raise ValueError(f"дубликат id внутри батча: {d['id']}")
            seen.add(d["id"])
            prepared.append(d)
        # этап 2: под блокировкой — контроль дубликатов с хранилищем, затем
        # одно обновление словаря и один _save (RLock, атомарно).
        with self._lock:
            for d in prepared:
                if d["id"] in self._nodes:
                    raise ValueError(
                        f"узел {d['id']} уже существует — add_many только для новых узлов"
                    )
            self._nodes.update({d["id"]: d for d in prepared})
            # уровни — после вставки: idf считается по корпусу вместе с батчем
            # (ревью 02.09, п.10: иначе восстановление из бэкапа в пустой стор
            # строило ключи по пустому idf)
            for d in prepared:
                self._refresh_levels(d)
            if self._use_hash_index:
                for d in prepared:
                    self._index_add(d)
            self._save()
        return prepared

    def get(self, node_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._nodes.get(node_id)

    def update(self, node: Union[MemoryNode, Dict[str, Any]]) -> Dict[str, Any]:
        d = node.to_dict() if isinstance(node, MemoryNode) else dict(node)
        with self._lock:
            if d["id"] not in self._nodes:
                raise KeyError(f"узел {d['id']} не найден — update невозможен")
            MemoryNode.from_dict(d)
            old = self._nodes[d["id"]]
            if self._use_hash_index:
                self._index_remove(d["id"], old.get("claim"))
            self._refresh_levels(d)
            self._nodes[d["id"]] = d
            if self._use_hash_index:
                self._index_add(d)
            self._save()
        return d

    def delete(self, node_id: str) -> bool:
        with self._lock:
            old = self._nodes.get(node_id)
            existed = old is not None
            if existed:
                if self._use_hash_index:
                    self._index_remove(node_id, old.get("claim"))
                # отцепляем у удалённых родителей (иначе останется битая ссылка)
                self._nodes.pop(node_id, None)
                for other in self._nodes.values():
                    kids = other.get("children") or []
                    if node_id in kids:
                        other["children"] = [c for c in kids if c != node_id]
                    if other.get("parent") == node_id:
                        other["parent"] = None
                self._save()
        return existed

    def all(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._nodes.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._nodes)

    # -- пластичность (brick-2) ----------------------------------------------
    def rewrite(
        self, node_id: str, new_claim: str, source: str = "", reason: str = ""
    ) -> Dict[str, Any]:
        """Переписывает claim узла новым фактом (старое — в revisions)."""
        with self._lock:
            d = self._nodes.get(node_id)
            if d is None:
                raise KeyError(f"узел {node_id} не найден — rewrite невозможен")
            node = MemoryNode.from_dict(d)
            node.rewrite(new_claim, source=source, reason=reason)
            if self._use_hash_index:
                self._index_remove(node_id, d.get("claim"))
            self._nodes[node_id] = node.to_dict()
            self._refresh_levels(self._nodes[node_id])  # claim изменился — уровни заново
            if self._use_hash_index:
                self._index_add(self._nodes[node_id])
            self._save()
            return self._nodes[node_id]

    def reinforce(self, node_id: str, delta: float = 0.05) -> float:
        """Подкрепляет узел (вес растёт, last_used обновляется)."""
        with self._lock:
            d = self._nodes.get(node_id)
            if d is None:
                raise KeyError(f"узел {node_id} не найден — reinforce невозможен")
            node = MemoryNode.from_dict(d)
            w = node.reinforce(delta)
            self._nodes[node_id] = node.to_dict()
            self._save()
            return w

    def decay(
        self,
        node_id: Optional[str] = None,
        half_life_hours: float = 168.0,
    ) -> Dict[str, float]:
        """Затухание весов: все узлы (или один) стареют без использования.

        Возвращает {id: новый вес}. last_used не трогается (см. model.decay).

        Фикс перф 01.09 (аудит §5.8): прямая правка поля weight в dict вместо
        from_dict/to_dict на каждом узле. Прежний путь строил и валидировал
        MemoryNode 3121 раз под RLock (≈5 с, MCP на это время висит);
        математика затухания та же (см. MemoryNode.decay), но без объектов.
        Пол веса зависит от вида: kind=rule не опускается ниже 0.5.
        """
        if half_life_hours <= 0:
            raise ValueError("half_life_hours должен быть > 0")
        with self._lock:
            ids = [node_id] if node_id is not None else list(self._nodes.keys())
            now_dt = datetime.now(timezone.utc)
            out: Dict[str, float] = {}
            for nid in ids:
                d = self._nodes.get(nid)
                if d is None:
                    raise KeyError(f"узел {nid} не найден — decay невозможен")
                w = float(d.get("weight", 1.0) or 1.0)
                last = _decay_ref(d)
                if last is not None and now_dt > last:
                    dt_h = (now_dt - last).total_seconds() / 3600.0
                    w = max(
                        weight_floor(str(d.get("kind") or "fact")),
                        w * (0.5 ** (dt_h / half_life_hours)),
                    )
                    # отмечаем, по какой момент затухание уже применено —
                    # иначе повторный запуск cron списывает тот же период ещё раз
                    d["decayed_at"] = now_dt.isoformat(timespec="milliseconds")
                d["weight"] = w
                out[nid] = w
            if node_id is None:
                # Ф1 02.09: ночной проход — пересчёт уровней детализации по
                # политике (qmem.decide_level); уровни строятся, если их нет
                self.requantize(now_dt, save=False)
            if out:
                self._save()  # один дамп на весь батч, а не на узел
            return out

    # -- квантование памяти (Ф1 02.09): уровни L0/L1/L2 ------------------------
    def _salience_for(self, force: bool = False) -> qmem.Salience:
        """idf-модель корпуса для сборки уровней. Пересобирается при полном
        пересчёте (requantize) и когда стор вырос/усох более чем на 10 % —
        одиночный add пользуется чуть устаревшим idf, это допустимо."""
        n = len(self._nodes)
        if force or self._salience is None or abs(n - self._salience_n) > max(5, n // 10):
            self._salience = qmem.Salience(
                d for d in self._nodes.values() if d.get("kind") != "hub"
            )
            self._salience_n = n
        return self._salience

    def _refresh_levels(self, d: Dict[str, Any], force: bool = False) -> bool:
        """Достраивает сжатые уровни L1/L2 узла, если их нет или текст L0
        изменился (отпечаток levels.src). Только под локом. Хабы — без уровней."""
        qmem.normalize(d)
        if d.get("kind") == "hub":
            d.pop("levels", None)
            return False
        if not force and not qmem.levels_stale(d):
            return False
        d["levels"] = qmem.build_levels(d, self._salience_for())
        d["level"] = qmem.current_level(d)
        return True

    def requantize(self, now_dt: Optional[datetime] = None, rebuild: bool = False,
                   save: bool = True) -> Dict[str, Any]:
        """Пересчёт уровней всех узлов по политике qmem.POLICY (ночной проход).

        rebuild=True — пересобрать сжатые тексты заново (после смены алгоритма
        выжимки). Возвращает гистограмму уровней и число изменённых узлов.
        """
        with self._lock:
            now_dt = now_dt or datetime.now(timezone.utc)
            self._salience_for(force=True)
            changed = built = 0
            for d in self._nodes.values():
                if self._refresh_levels(d, force=rebuild):
                    built += 1
                lv = qmem.decide_level(d, now_dt)
                if lv != qmem.current_level(d) or "level" not in d:
                    d["level"] = lv
                    changed += 1
            if save and (changed or built or self._usage_dirty):
                self._save()
            return {"levels": qmem.level_histogram(self._nodes.values()),
                    "changed": changed, "built": built, "nodes": len(self._nodes)}

    TOUCH_FLUSH_EVERY = 50  # накопленных попаданий до принудительной записи

    def _touch_hits(self, hits: List[Dict[str, Any]], now_dt: datetime) -> None:
        """Попадание в выдачу = использование: usage узла и подъём на L0 (в
        памяти; на диск — с ближайшей записью или каждые TOUCH_FLUSH_EVERY
        попаданий). Только под локом. Хабы и неактивные узлы не касаются."""
        for d in hits:
            if d.get("kind") == "hub" or not self._node_is_active(d, now_dt):
                continue
            qmem.touch(d, now_dt)
            self._usage_dirty = True
            self._dirty_hits += 1
        if self._dirty_hits >= self.TOUCH_FLUSH_EVERY:
            # ревью 02.09 (финал, п.1): ошибка диска при флаше не должна ронять
            # ПОИСК и повторяться на каждом запросе — попадания остаются в
            # памяти, их подхватит ближайшая успешная запись
            try:
                self._save()
            except OSError as exc:
                logger.warning("flush usage: запись не удалась (%s) — отложено", exc)
                self._dirty_hits = 0

    def touch(self, nodes: List[Dict[str, Any]]) -> None:
        """Отметить использование ВЫДАННЫХ агенту узлов (сервер зовёт после
        фильтров tags/kind/гейтов — касаться то, что агент реально увидел)."""
        with self._lock:
            ids = {d.get("id") for d in nodes if isinstance(d, dict)}
            live = [self._nodes[i] for i in ids if i in self._nodes]
            self._touch_hits(live, datetime.now(timezone.utc))

    def snapshot(self, nodes: List[Dict[str, Any]], drop: tuple = ("levels",)) -> List[Dict[str, Any]]:
        """Неглубокие копии узлов ПОД ЛОКОМ (для отдачи наружу): выдача search —
        живые dict, и параллельный touch другого потока не должен ловить
        «dictionary changed size during iteration» у клиента."""
        with self._lock:
            return [{k: v for k, v in d.items() if k not in drop} for d in nodes]

    def flush_usage(self) -> bool:
        """Записать накопленные usage/level, если они есть (для ночного cron
        и аккуратной остановки). Возвращает True, если запись была."""
        with self._lock:
            if not self._usage_dirty:
                return False
            self._save()
            return True

    # -- связывание существующих узлов (фикс 01.09, аудит §5.6) ---------------
    def link_existing(
        self,
        from_id: str,
        to_id: str,
        bidirectional: bool = False,
        author: str = "",
        rel: str = "",
    ) -> Dict[str, Any]:
        """Связывает два УЖЕ СУЩЕСТВУЮЩИХ узла ребром from_id -> to_id.

        Причина появления: memory_link умел только создавать новый узел-ребёнка,
        поэтому связать два живых узла через MCP было нельзя — отсюда 2 ребра
        (обе битые) на 5251 узел в аудите 29.08. Дубли рёбер не плодятся,
        петля на себя запрещена, оба конца обязаны существовать.

        rel (02.09, письмо Qwen): тип связи в паспорте ребра link_meta[to].rel —
        related_to (по умолчанию) / part_of / has_part / conflicts_with /
        supersedes / duplicate_of / refers_to (см. budget.RELS). Неизвестный
        тип — ошибка, а не тихое «related_to». На уже существующем ребре rel
        ДОПИСЫВАЕТСЯ, если его не было; чужой rel не переписывается.
        """
        from .budget import REL_RELATED, RELS  # ленивый импорт: без циклов

        rel = str(rel or "").strip() or REL_RELATED
        if rel not in RELS:
            raise ValueError(f"rel должно быть одним из {RELS}, получено {rel!r}")
        # ревью 02.09 (п.6): у направленного rel обратное ребро — не тот же rel.
        # part_of <-> has_part инвертируются; related_to/conflicts_with
        # симметричны; supersedes/duplicate_of/refers_to в обе стороны — ошибка.
        inverse = {"part_of": "has_part", "has_part": "part_of",
                   REL_RELATED: REL_RELATED, "conflicts_with": "conflicts_with"}
        if bidirectional and rel not in inverse:
            raise ValueError(f"rel={rel!r} направленный: bidirectional для него запрещён")
        rel_back = inverse.get(rel, rel)
        with self._lock:
            src = self._nodes.get(from_id)
            dst = self._nodes.get(to_id)
            if src is None:
                raise KeyError(f"узел {from_id} не найден — link невозможен")
            if dst is None:
                raise KeyError(f"узел {to_id} не найден — битую ссылку не создаём")
            if from_id == to_id:
                raise ValueError("петля: узел не может ссылаться на самого себя")
            added = []
            # паспорт ребра (01.09): кто и когда провёл связь. Пишется только
            # на РЕАЛЬНО добавленное ребро — повторный link не даёт чужому
            # агенту переписать авторство уже существующей связи.
            who = str(author or "").strip() or "unknown"
            stamp = {"author": who, "ts": now_iso(), "rel": rel}
            changed = False
            if to_id not in (src.get("links") or []):
                src["links"] = list(src.get("links") or []) + [to_id]
                src["link_meta"] = dict(src.get("link_meta") or {})
                src["link_meta"][to_id] = dict(stamp)
                added.append(f"{from_id}->{to_id}")
            else:
                changed |= self._fill_rel(src, to_id, rel, who)
            if bidirectional and from_id not in (dst.get("links") or []):
                dst["links"] = list(dst.get("links") or []) + [from_id]
                dst["link_meta"] = dict(dst.get("link_meta") or {})
                dst["link_meta"][from_id] = {**stamp, "rel": rel_back}
                added.append(f"{to_id}->{from_id}")
            elif bidirectional:
                changed |= self._fill_rel(dst, from_id, rel_back, who)
            if added or changed:
                self._save()
            rel_now = str(((src.get("link_meta") or {}).get(to_id) or {}).get("rel") or "")
            return {"from": from_id, "to": to_id, "added": added,
                    "rel": rel_now, "rel_applied": rel_now == rel,
                    "author": who if added else None,
                    "link_meta_from": dict(src.get("link_meta") or {}),
                    "links_from": list(src.get("links") or [])}

    @staticmethod
    def _fill_rel(node: Dict[str, Any], to_id: str, rel: str, who: str = "unknown") -> bool:
        """Дописывает rel в паспорт СУЩЕСТВУЮЩЕГО ребра, если типа ещё нет.
        Старые рёбра (до 02.09) паспорта с rel не имеют — так они получают
        тип без переписывания авторства ребра; кто и когда проставил тип —
        rel_by/rel_ts (ревью 02.09, п.5: иначе ретип чужого ребра без следа).
        Возвращает True, если что-то изменилось."""
        meta = dict(node.get("link_meta") or {})
        cur = dict(meta.get(to_id) or {"author": "unknown", "ts": ""})
        if cur.get("rel"):
            return False
        cur["rel"] = rel
        cur["rel_by"] = who
        cur["rel_ts"] = now_iso()
        meta[to_id] = cur
        node["link_meta"] = meta
        return True

    # -- рефакторинг по письму Qwen (02.09): бюджетный поиск, промпт, граф ----
    def search_budget(self, query: str, top_k: Optional[int] = None,
                      token_budget: Optional[int] = None, expand: bool = True) -> Dict[str, Any]:
        """Token-budgeting поиск: top_k по сложности запроса (5/10/20),
        токенный бюджет, граф-расширение, хабы отдельно. См. budget.BudgetSearch."""
        from .budget import BudgetSearch

        with self._lock:
            engine = self._budget_engine()
        return engine.search(query, top_k=top_k, token_budget=token_budget, expand=expand)

    # -- ML-BOOST 03.09: лексический BM25F и слияние RRF ----------------------
    def _bm25_engine(self) -> Any:
        """Индекс BM25F, кэшируется и пересобирается при любой записи (_version).
        Только под локом — как _budget_engine."""
        from . import mlsearch

        cached = getattr(self, "_bm25_cache", None)
        if cached is None or cached[0] != self._version:
            cached = (self._version, mlsearch.BM25F(dict(self._nodes)))
            self._bm25_cache = cached
        return cached[1]

    def _dense_lists(self, query: str, depth: int) -> List[str]:
        """Плотный сигнал для RRF — ТОЛЬКО если установлен fastembed.

        Это единственная часть ML-BOOST, требующая модели (MiniLM, 0.22 ГБ,
        ~80 мс на кодировку запроса). Её нет в requirements (рантайм Mnemos —
        stdlib), поэтому при отсутствии fastembed сигнал молча не голосует, а
        RRF остаётся чисто лексическим. Цена вопроса измерена: kw 0.9833 с
        плотным сигналом против 0.9444 без него (см. REPORT.md).
        """
        from . import mlsearch

        try:
            embedder = self._get_embedder()
        except ImportError:
            return []
        with self._lock:
            version = self._version
            blobs = {nid: mlsearch.node_blob_of(n) for nid, n in self._nodes.items()
                     if n.get("kind") != "hub"}
        cached = getattr(self, "_dense_cache", None)
        if cached is None or cached[0] != version:
            ids = list(blobs)
            vecs = [list(v) for v in embedder.embed([blobs[i] for i in ids])] if ids else []
            cached = (version, ids, vecs)
            self._dense_cache = cached
        _v, ids, vecs = cached
        qv = next(iter(embedder.embed([query.strip()])), None)
        if qv is None:  # pragma: no cover
            return []
        qv = list(qv)
        scored = sorted(zip(ids, vecs), key=lambda p: -_cosine(qv, p[1]))
        return [nid for nid, _ in scored[:depth]]

    def search_rrf(self, query: str, top_k: int = 5, depth: int = 20,
                   use_dense: bool = True) -> List[Dict[str, Any]]:
        """Слияние сигналов Reciprocal Rank Fusion (ML-BOOST 03.09).

        Голосуют: Ф1 (search_budget), BM25F со стеммингом (mnemos.mlsearch) и —
        если в системе есть fastembed — плотный поиск. RRF безпараметричен
        (k=60, канон Cormack 2009), веса равные: подбор весов слияния на 15
        запросах GT — подгонка, которая на новом запросе даёт ХУЖЕ, чем её
        отсутствие (замер leave-one-query-out, ML-BOOST §5.2).

        Замер на боевом сторе (65 узлов, GT 15 запросов, recall@5):
          * ключевые слова: 0.9444 (Ф1 и подстрока) -> 0.9833 с плотным
            сигналом, 0.9444 без него — то есть без fastembed это не регрессия,
            но и не выигрыш;
          * вопрос на естественном языке: 0.8167 — ХУЖЕ, чем 0.8667 у одного
            search_budget. Поэтому NL сюда не роутится (см. server.memory_search).

        Стор не изменяется: узлы не отмечаются использованием (touch), это
        делает вызывающий по отфильтрованной выдаче.
        """
        if not query or not query.strip():
            return []
        from . import mlsearch

        k = max(1, min(int(top_k), 50))
        d = max(k, int(depth))
        budget = self.search_budget(query, top_k=d, token_budget=10 ** 7)
        f1 = [r["id"] for r in (budget.get("results") or [])][:d]
        with self._lock:
            bm25 = self._bm25_engine()
            hubs = [nid for nid, n in self._nodes.items() if n.get("kind") == "hub"]
        lex = bm25.rank(query, depth=d, skip=hubs)
        lists = [lst for lst in (f1, lex) if lst]
        if use_dense:
            dense = self._dense_lists(query, d)
            if dense:
                lists.append(dense)
        if not lists:
            return []
        with self._lock:
            now_dt = datetime.now(timezone.utc)
            out: List[Dict[str, Any]] = []
            for nid in mlsearch.rrf(lists):
                node = self._nodes.get(nid)
                if node is None or not self._node_is_active(node, now_dt):
                    continue
                if node.get("kind") == "hub":
                    continue
                out.append(node)
                if len(out) >= k:
                    break
            return out

    def graph(self) -> Any:
        """Граф-запросы (neighbors/path/hub/rules_for/conflicts) по снимку стора."""
        from .budget import Graph

        with self._lock:
            return Graph(dict(self._nodes))

    def _budget_engine(self) -> Any:
        """Индекс BudgetSearch кэшируется и пересобирается при любой записи
        (_save двигает _version). Только под локом."""
        from .budget import BudgetSearch

        cached = getattr(self, "_budget_cache", None)
        if cached is None or cached[0] != self._version:
            cached = (self._version, BudgetSearch(dict(self._nodes)))
            self._budget_cache = cached
        return cached[1]

    # -- статистика графа (фикс 01.09, аудит §5.6) ----------------------------
    def stats(self) -> Dict[str, Any]:
        """Паспорт стора: узлы, виды, рёбра, сироты, дубли, веса, TTL."""
        with self._lock:
            nodes = list(self._nodes.values())
            now_dt = datetime.now(timezone.utc)
            kinds: Dict[str, int] = {}
            tags: Dict[str, int] = {}
            edges = broken = 0
            edges_by_author: Dict[str, int] = {}
            edges_by_rel: Dict[str, int] = {}
            incoming: set = set()
            weights: List[float] = []
            claim_keys: Dict[str, int] = {}
            expired = no_evidence = 0
            for d in nodes:
                kinds[str(d.get("kind"))] = kinds.get(str(d.get("kind")), 0) + 1
                for t in d.get("tags") or []:
                    tags[t] = tags.get(t, 0) + 1
                for l in d.get("links") or []:
                    edges += 1
                    meta = (d.get("link_meta") or {}).get(l) or {}
                    who = str(meta.get("author") or "unknown")
                    edges_by_author[who] = edges_by_author.get(who, 0) + 1
                    rel = str(meta.get("rel") or "untyped")
                    edges_by_rel[rel] = edges_by_rel.get(rel, 0) + 1
                    if l in self._nodes:
                        incoming.add(l)
                    else:
                        broken += 1
                if d.get("parent"):
                    incoming.add(d["id"])
                weights.append(float(d.get("weight", 1.0) or 1.0))
                key = _canonical_key(d.get("claim"))
                claim_keys[key] = claim_keys.get(key, 0) + 1
                if not self._node_is_active(d, now_dt):
                    expired += 1
                if not (d.get("evidence") or []):
                    no_evidence += 1
            linked = set(incoming)
            for d in nodes:
                if (d.get("links") or []) or (d.get("children") or []):
                    linked.add(d["id"])
            n = len(nodes) or 1
            return {
                "nodes": len(nodes),
                "kinds": kinds,
                "tags": dict(sorted(tags.items(), key=lambda kv: -kv[1])[:20]),
                "edges": edges,
                "edges_by_author": dict(
                    sorted(edges_by_author.items(), key=lambda kv: -kv[1])
                ),
                "edges_by_rel": dict(
                    sorted(edges_by_rel.items(), key=lambda kv: -kv[1])
                ),
                "hubs": kinds.get("hub", 0),
                # Ф1 02.09: уровни детализации (квантование) и использование
                "levels": qmem.level_histogram(nodes),
                "quantized": sum(1 for d in nodes if qmem.current_level(d) > 0),
                "with_levels": sum(1 for d in nodes if isinstance(d.get("levels"), dict)),
                "usage_hits": sum(int((d.get("usage") or {}).get("count", 0) or 0) for d in nodes),
                "broken_edges": broken,
                "orphans": len(nodes) - len(linked),
                "orphans_pct": round(100.0 * (len(nodes) - len(linked)) / n, 2),
                "dup_nodes": sum(c - 1 for c in claim_keys.values() if c > 1),
                "weight_min": round(min(weights), 4) if weights else None,
                "weight_max": round(max(weights), 4) if weights else None,
                "weight_mean": round(sum(weights) / n, 4) if weights else None,
                "weight_eq_1_pct": round(
                    100.0 * sum(1 for w in weights if w >= 0.9999) / n, 2
                ),
                "inactive_or_expired": expired,
                "no_evidence": no_evidence,
                "bytes": self.path.stat().st_size if self.path.exists() else 0,
            }

    # -- уборка: prune (фикс 01.09, аудит §5.6) -------------------------------
    PRUNE_RULES = ("expired_ttl", "exact_dupes", "source_prefix", "weak")

    def prune(
        self,
        rule: str,
        dry_run: bool = True,
        max_delete: int = 100,
        source_prefix: str = "",
        older_than_days: int = 30,
        weak_weight: float = 0.1,
        export_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Уборка стора по формальному правилу. По умолчанию — только показ.

        Правила:
          expired_ttl   — valid_until истёк -> kind=outdated (НЕ удаляем);
          exact_dupes   — одинаковый канонический claim -> оставляем свежайший,
                          переносим ему максимальный вес группы, остальные удаляем;
          source_prefix — узлы с source.startswith(prefix) -> выгрузка в
                          export_path и удаление (вынос чужих корпусов);
          weak          — weight <= weak_weight И last_used старше
                          older_than_days И нет входящих ссылок -> удаление.

        Инварианты (иначе инструмент однажды выкосит память):
          * dry_run=True по умолчанию — ничего не меняем, только список;
          * max_delete ограничивает удаление сверху (жёстко);
          * kind="rule" не трогается НИКОГДА, ни одним правилом;
          * узел, на который есть входящие ссылки, не удаляется (не плодим
            битые рёбра — их и так было 2 из 2 в аудите);
          * source_prefix без export_path не удаляет ничего (сначала копия).
        """
        if rule not in self.PRUNE_RULES:
            raise ValueError(
                f"prune: rule должно быть одним из {self.PRUNE_RULES}, получено {rule!r}"
            )
        max_delete = max(0, int(max_delete))
        with self._lock:
            now_dt = datetime.now(timezone.utc)
            nodes = self._nodes
            # входящие ссылки: кого нельзя удалять
            referenced: set = set()
            for d in nodes.values():
                for l in d.get("links") or []:
                    referenced.add(l)
                if d.get("parent"):
                    referenced.add(d["parent"])
                for c in d.get("children") or []:
                    referenced.add(c)

            candidates: List[Dict[str, Any]] = []   # {id, reason, action}
            if rule == "expired_ttl":
                for nid, d in nodes.items():
                    if d.get("kind") in ("rule", "outdated", "refuted"):
                        continue
                    vu = d.get("valid_until")
                    if not vu:
                        continue
                    vu_dt = _parse_iso(str(vu))
                    if vu_dt is not None and now_dt >= vu_dt:
                        candidates.append({
                            "id": nid, "action": "mark_outdated",
                            "reason": f"valid_until {vu} истёк",
                            "claim": str(d.get("claim"))[:80],
                        })
            elif rule == "exact_dupes":
                groups: Dict[str, List[str]] = {}
                for nid, d in nodes.items():
                    if d.get("kind") == "rule":
                        continue
                    groups.setdefault(_canonical_key(d.get("claim")), []).append(nid)
                for key, ids in groups.items():
                    if len(ids) < 2 or not key:
                        continue
                    ids.sort(key=lambda i: str(nodes[i].get("ts") or ""), reverse=True)
                    keep, drop = ids[0], ids[1:]
                    w_max = max(float(nodes[i].get("weight", 1.0) or 1.0) for i in ids)
                    for nid in drop:
                        if nid in referenced:
                            continue  # на узел ссылаются — не плодим битую ссылку
                        candidates.append({
                            "id": nid, "action": "delete",
                            "reason": f"полный дубль {keep} (claim совпадает), "
                                      f"вес группы {w_max:.2f} уходит выжившему",
                            "keep": keep, "keep_weight": w_max,
                            "claim": str(nodes[nid].get("claim"))[:80],
                        })
            elif rule == "source_prefix":
                if not source_prefix:
                    raise ValueError("prune source_prefix: параметр source_prefix обязателен")
                for nid, d in nodes.items():
                    if d.get("kind") == "rule":
                        continue
                    if str(d.get("source") or "").startswith(source_prefix):
                        candidates.append({
                            "id": nid, "action": "delete",
                            "reason": f"source начинается с {source_prefix!r}",
                            "claim": str(d.get("claim"))[:80],
                        })
            else:  # weak
                cutoff = now_dt - timedelta(days=int(older_than_days))
                for nid, d in nodes.items():
                    if d.get("kind") == "rule" or nid in referenced:
                        continue
                    if float(d.get("weight", 1.0) or 1.0) > float(weak_weight):
                        continue
                    lu = _parse_iso(str(d.get("last_used") or ""))
                    if lu is None or lu >= cutoff:
                        continue
                    candidates.append({
                        "id": nid, "action": "delete",
                        "reason": f"вес {d.get('weight')} <= {weak_weight}, "
                                  f"last_used {d.get('last_used')} старше "
                                  f"{older_than_days} дн, входящих ссылок нет",
                        "claim": str(d.get("claim"))[:80],
                    })

            to_delete = [c for c in candidates if c["action"] == "delete"]
            to_mark = [c for c in candidates if c["action"] == "mark_outdated"]
            capped = len(to_delete) > max_delete
            if capped:
                to_delete = to_delete[:max_delete]
            result: Dict[str, Any] = {
                "rule": rule,
                "dry_run": bool(dry_run),
                "nodes_before": len(nodes),
                "candidates": len(candidates),
                "to_delete": len(to_delete),
                "to_mark_outdated": len(to_mark),
                "capped_by_max_delete": capped,
                "max_delete": max_delete,
                "items": (to_mark + to_delete)[:200],
            }
            if dry_run:
                result["applied"] = False
                return result

            if rule == "source_prefix" and not export_path:
                raise ValueError(
                    "prune source_prefix: без export_path удаление запрещено — "
                    "сначала выгрузите корпус в отдельный файл"
                )
            if export_path and to_delete:
                dump = {c["id"]: nodes[c["id"]] for c in to_delete if c["id"] in nodes}
                p = Path(export_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, "w", encoding="utf-8") as f:
                    _dump_by_line(dump, f)
                result["exported_to"] = str(p)
                result["exported"] = len(dump)
            for c in to_mark:
                d = nodes.get(c["id"])
                if d is not None:
                    d["kind"] = "outdated"
            for c in to_delete:
                d = nodes.pop(c["id"], None)
                if d is None:
                    continue
                if self._use_hash_index:
                    self._index_remove(c["id"], d.get("claim"))
                self._embeddings.pop(c["id"], None)
                keep = c.get("keep")
                if keep and keep in nodes:  # перенос веса выжившему (аудит §5.6)
                    nodes[keep]["weight"] = max(
                        float(nodes[keep].get("weight", 1.0) or 1.0),
                        float(c.get("keep_weight", 1.0)),
                    )
            if to_mark or to_delete:
                self._save()
            result["applied"] = True
            result["deleted"] = len(to_delete)
            result["marked_outdated"] = len(to_mark)
            result["nodes_after"] = len(nodes)
            self._journal_prune(result)
            return result

    def _journal_prune(self, result: Dict[str, Any]) -> None:
        """Журнал уборки рядом со стором: что и почему удалено (Правило 0)."""
        try:
            path = self.path.parent / "prune.log"
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(
                    {"ts": now_iso(), **{k: v for k, v in result.items() if k != "items"},
                     "ids": [i["id"] for i in result.get("items", [])]},
                    ensure_ascii=False,
                ) + "\n")
        except OSError as exc:  # журнал не должен ронять уборку
            logger.warning("prune: не удалось записать журнал (%s)", exc)

    # -- «граф в узле»: дочерние узлы с ограниченной глубиной ------------------
    def _parent_chain(self, node_id: str) -> List[str]:
        """Цепочка предков (родитель, дед, ...) без зацикливания."""
        with self._lock:
            chain: List[str] = []
            seen = {node_id}
            cur = self._nodes.get(node_id, {}).get("parent") if node_id in self._nodes else None
            while cur and cur not in seen:
                seen.add(cur)
                chain.append(cur)
                cur = self._nodes.get(cur, {}).get("parent")
            return chain

    def depth(self, node_id: str) -> int:
        """Глубина узла в дереве (0 — корень, без родителя)."""
        return len(self._parent_chain(node_id))

    def ancestors(self, node_id: str) -> List[str]:
        """id предков от родителя вверх."""
        return self._parent_chain(node_id)

    def children(self, node_id: str) -> List[Dict[str, Any]]:
        """Непосредственные дочерние узлы (id в children родителя)."""
        with self._lock:
            d = self._nodes.get(node_id)
            if d is None:
                raise KeyError(f"узел {node_id} не найден")
            out = []
            for cid in d.get("children") or []:
                if cid in self._nodes:
                    out.append(self._nodes[cid])
            return out

    def add_child(
        self, parent_id: str, node: Union[MemoryNode, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Кладёт НОВЫЙ узел как ребёнка parent_id (структурная рекурсия).

        Ограничения: узел не может быть собственным предком (цикл) и не
        может сесть глубже MAX_DEPTH. Возвращает dict нового узла.
        """
        with self._lock:
            if parent_id not in self._nodes:
                raise KeyError(f"родитель {parent_id} не найден")
            d = node.to_dict() if isinstance(node, MemoryNode) else dict(node)
            MemoryNode.from_dict(d)  # валидация схемы
            chain = self._parent_chain(parent_id)
            if d["id"] == parent_id or d["id"] in chain:
                raise ValueError("цикл: узел не может быть собственным предком")
            if d["id"] in self._nodes:
                raise ValueError(f"узел {d['id']} уже существует — add_child только для новых узлов")
            if len(chain) + 1 > MAX_DEPTH:
                raise ValueError(
                    f"глубина > {MAX_DEPTH}: структурная рекурсия ограничена"
                )
            d["parent"] = parent_id
            parent = self._nodes[parent_id]
            kids = list(parent.get("children") or [])
            kids.append(d["id"])
            parent["children"] = kids
            self._refresh_levels(d)
            self._nodes[d["id"]] = d
            if self._use_hash_index:
                self._index_add(d)
            self._save()
            return d

    # -- ранжирование: вес и уверенность (фикс 01.09, аудит §5.2) -------------
    # До 01.09 поле weight не читалось НИ ОДНОЙ строкой кода поиска: decay,
    # reinforce и WEIGHT_MIN были декорацией — они писались в JSON и никогда
    # не влияли на то, что агент получает в контекст.
    @staticmethod
    def _node_is_active(d: Dict[str, Any], now_dt: datetime) -> bool:
        """Живой ли узел: не refuted/outdated и TTL не истёк."""
        if d.get("kind") in ("refuted", "outdated"):
            return False
        vu = d.get("valid_until")
        if not vu:
            return True
        vu_dt = _parse_iso(str(vu))
        if vu_dt is None:
            return True  # не парсится — не трогаем
        return now_dt < vu_dt

    @staticmethod
    def _node_decayed_weight(d: Dict[str, Any], now_dt: datetime) -> float:
        """Вес с учётом затухания БЕЗ мутации — для ранжирования при чтении."""
        w = float(d.get("weight", WEIGHT_MIN) or WEIGHT_MIN)
        lu = _decay_ref(d)  # только НЕприменённая часть затухания
        if lu is None or now_dt <= lu:
            return w
        dt_h = (now_dt - lu).total_seconds() / 3600.0
        if DEFAULT_HALF_LIFE_HOURS <= 0:
            return w
        return max(
            weight_floor(str(d.get("kind") or "fact")),
            w * (0.5 ** (dt_h / DEFAULT_HALF_LIFE_HOURS)),
        )

    @staticmethod
    def _node_boost(d: Dict[str, Any], now_dt: datetime) -> float:
        """Бонус ранжирования: confidence(0..1) * затухающий вес(0..1) -> 0..1."""
        conf = float(d.get("confidence", 0.5) or 0.5)
        conf = max(0.0, min(1.0, conf))
        return conf * Store._node_decayed_weight(d, now_dt)

    # -- поиск ----------------------------------------------------------------
    @staticmethod
    def _field_score(text: Dict[str, Any], q: str) -> float:
        """Лексический балл подстроки q по полям claim/source/context/keys +
        evidence, плюс tf-бонус за повторы в claim (Ф1, гейт R4)."""
        score = 0.0
        for field, weight in _QFIELD_WEIGHTS:
            if q in str(text.get(field, "") or "").lower():
                score += weight
        for ev in text.get("evidence", []) or []:
            if isinstance(ev, str) and q in ev.lower():
                score += 1
        if score > 0:
            tf = str(text.get("claim", "") or "").lower().count(q)
            if tf > 1:
                score += min(_TF_MAX, _TF_BONUS * math.log(tf))
        return score

    def search(
        self,
        query: str,
        top_k: int = 5,
        use_hash_index: bool = True,
        include_hubs: bool = False,
        touch: bool = True,
    ) -> List[Dict[str, Any]]:
        """Подстрочный поиск по claim/source/context/evidence, топ-k по релевантности.

        include_hubs=False (02.09): узлы kind=hub в выдачу не попадают — их
        claim («ХАБ «Алиса»: 14 узлов…») содержит ключевые слова кластера и
        без фильтра вытесняет реальные ответы: на копии стора с 16 хабами
        recall@5 по GT падал 0.928 -> 0.911, hit@1 0.867 -> 0.200.

        Ранг: сумма весов полей, содержащих query (без учёта регистра)
        + 0.5*ln(tf) за повторы запроса в claim (Ф1, гейт R4)
        + 2*confidence*вес (фикс 01.09)
        + буст уровня детализации (Ф1: L0 +1.0 / L1 +0.6 / L2 +0.2).
        При равенстве — свежие узлы первыми (по ts, ISO лексикографически).

        Ф1 02.09 (квантование): поиск идёт по всем уровням — у узла на L1/L2
        проверяются и сжатый текст уровня (levels[lv]: выжимка + keys), и
        полный; буст уровня даётся только совпадению в тексте уровня. touch=True:
        попадания в выдачу отмечаются как использование (usage) и узел
        поднимается на L0.

        Фикс перф 27.08 (хеш-индекс): если query — ТОЧНЫЙ claim какого-либо
        узла (канонический ключ: strip + casefold), запрос обслуживается за
        O(1) через self._exact_index и возвращает ровно узлы с этим claim
        (свежие первыми, топ-k) — это быстрый путь. Иначе — прежний полный
        проход (substring rank). use_hash_index=False — всегда полный проход
        без индекса. С Ф1 пути расходятся ровно в одном случае: узел на L1/L2,
        чей полный claim равен запросу, — точный путь его найдёт (индекс по
        L0-claim), полный проход по сжатому тексту — не обязательно.
        Индекс должен быть включён при создании Store(use_hash_index=True).
        """
        if not query or not query.strip():
            return []
        q = query.strip().lower()
        k = max(1, min(int(top_k), 50))
        if use_hash_index and self._use_hash_index:
            with self._lock:
                now_dt = datetime.now(timezone.utc)
                # ревью 02.09 (п.1): точный путь обязан соблюдать те же фильтры,
                # что и скан — refuted/outdated/протухшие/хабы по точному claim
                # не выдаём (до Ф1 дыра существовала с 27.08)
                exact = [
                    n for n in self.search_exact(query, top_k=50)
                    if self._node_is_active(n, now_dt)
                    and (include_hubs or n.get("kind") != "hub")
                ][:k]
                if exact:
                    # быстрый путь: query — точный claim (O(1) вместо O(N))
                    if touch:
                        self._touch_hits(exact, now_dt)
                    return exact
        with self._lock:
            now_dt = datetime.now(timezone.utc)
            scored: List[tuple] = []
            for node in self._nodes.values():
                # фикс 01.09: протухшие по TTL и refuted/outdated не выдаём
                if not self._node_is_active(node, now_dt):
                    continue
                if not include_hubs and node.get("kind") == "hub":
                    continue
                # Ф1: поиск ПО ВСЕМ уровням. Совпадение в тексте текущего уровня
                # (L0 — полный узел; L1/L2 — выжимка + ключи) получает буст
                # уровня; узел на L1/L2, у которого запрос есть только в полном
                # тексте, находится тоже, но без буста — «спящая» память ниже
                # горячей при равном матче, а не невидима (гейт держится в любом
                # состоянии старения, замер 02.09).
                lv = qmem.effective_level(node)
                s_full = self._field_score(node, q)
                s_lvl = self._field_score(qmem.level_text(node), q) if lv > qmem.L0 else s_full
                score = max(s_full, s_lvl)
                if score > 0:
                    # фикс 01.09: вес и уверенность входят в ранг (0..2 сверху).
                    # Узел с weight=0.1 уходит ниже узла с weight=1.0 при
                    # равном лексическом score — decay наконец влияет на выдачу.
                    score += self._node_boost(node, now_dt) * 2.0
                    if s_lvl > 0:
                        score += qmem.LEVEL_BOOST.get(lv, 0.0)
                    scored.append((score, node.get("ts", ""), node))
            # свежие первыми (ts desc), затем стабильно по убыванию ранга
            scored.sort(key=lambda t: t[1], reverse=True)
            scored.sort(key=lambda t: -t[0])
            out = [node for _, _, node in scored[:k]]
            if touch and out:
                self._touch_hits(out, now_dt)
            return out

    def search_exact(
        self,
        query: str,
        top_k: int = 5,
        use_hash_index: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Точный поиск: узлы, чей claim СОВПАДАЕТ с query (по точному ключу).

        Ключ — канонический: strip + casefold (см. _canonical_key), поэтому
        «Выручка +12%» находит «  выручка +12%  », но НЕ находит «Выручка
        +12% в Q3» (никаких substring-ложных срабатываний — точность 100%).

        Пути:
          * use_hash_index=True (по умолчанию — как у Store): O(1) через
            self._exact_index; при ПРОМАХЕ индекса — fallback-скан
            авторитетного dict self._nodes с самолечением индекса (защита от
            рассинхрона: результат всегда правильный, индекс чинится на лету).
          * use_hash_index=False: чистый полный проход по _nodes (старый путь,
            тот же результат — используется в бенчмарке как «ДО»).

        Возвращает найденные узлы, свежие первыми (ts desc), не более top_k.
        """
        if not query or not query.strip():
            return []
        k = max(1, min(int(top_k), 50))
        key = _canonical_key(query)
        use = self._use_hash_index if use_hash_index is None else bool(use_hash_index)
        with self._lock:
            if use:
                ids = self._exact_index.get(key)
                if ids:
                    found: List[Dict[str, Any]] = []
                    stale: List[str] = []
                    for nid in ids:
                        node = self._nodes.get(nid)
                        # сверка по точному ключу: не доверяем индексу слепо
                        if node is not None and _canonical_key(node.get("claim")) == key:
                            found.append(node)
                        else:
                            stale.append(nid)
                    if stale:  # самолечение: вычищаем битые записи индекса
                        for nid in stale:
                            bucket = self._exact_index.get(key)
                            if bucket and nid in bucket:
                                bucket.remove(nid)
                        if not self._exact_index.get(key):
                            del self._exact_index[key]
                    if found:
                        found.sort(key=lambda n: n.get("ts", ""), reverse=True)
                        return found[:k]
                # промах индекса: fallback — авторитетный источник (полный
                # проход O(N), происходит редко: рассинхрон или ручная правка)
                hits = [
                    n for n in self._nodes.values()
                    if _canonical_key(n.get("claim")) == key
                ]
                if hits:
                    # самолечение: восстанавливаем запись индекса
                    self._exact_index[key] = [n["id"] for n in hits]
                    hits.sort(key=lambda n: n.get("ts", ""), reverse=True)
                    return hits[:k]
                return []
            # старый путь: полный проход без индекса (тот же результат)
            hits = [
                n for n in self._nodes.values()
                if _canonical_key(n.get("claim")) == key
            ]
            hits.sort(key=lambda n: n.get("ts", ""), reverse=True)
            return hits[:k]

    # -- семантический поиск (бенчмарк-фикс 26.08) ---------------------------
    def _get_embedder(self) -> Any:
        """Лениво создаёт fastembed.TextEmbedding. Если пакета нет —
        честный ImportError с подсказкой (а не голый ModuleNotFoundError)."""
        if self._embedder is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:  # fastembed не установлен
                raise ImportError(_FASTEMBED_HINT) from exc
            self._embedder = TextEmbedding(model_name=_SEMANTIC_MODEL)
        return self._embedder

    def search_semantic(
        self, query: str, top_k: int = 5
    ) -> List[tuple]:
        """Семантический поиск по смыслу claim (косинусная близость).

        Бенчмарк-фикс 26.08: закрывает дыру «семантика 0/25» — перефразы,
        которые подстрочный search() не видит. Эмбеддинги кэшируются в
        памяти (self._embeddings: id -> (claim, вектор)) и пересчитываются
        только для узлов, чей claim изменился с последнего прохода; векторы
        удалённых узлов вычищаются. Возвращает [(node, score), ...] по
        убыванию косинусной близости. Требует fastembed (ленивый импорт).
        """
        if not query or not query.strip():
            return []
        k = max(1, min(int(top_k), 50))
        embedder = self._get_embedder()  # ImportError — если нет fastembed
        # снапшот (id, claim) под блокировкой; сами вычисления — вне лока:
        # первичный embed медленный (модель/ONNX), блокировать потоки нельзя.
        with self._lock:
            claims = {
                nid: str(n.get("claim") or "")
                for nid, n in self._nodes.items()
            }
            for nid in [i for i in self._embeddings if i not in self._nodes]:
                del self._embeddings[nid]  # кэш удалённых узлов — вон
            stale = [
                nid for nid, claim in claims.items()
                if self._embeddings.get(nid, (None,))[0] != claim
            ]
            texts = [claims[nid] for nid in stale]
        if stale:
            vectors = [v.tolist() for v in embedder.embed(texts)]
            with self._lock:
                for nid, vec in zip(stale, vectors):
                    self._embeddings[nid] = (claims[nid], vec)
        qv = next(iter(embedder.embed([query.strip()])), None)
        if qv is None:  # pragma: no cover
            return []
        qv = qv.tolist()
        with self._lock:
            now_dt = datetime.now(timezone.utc)
            scored: List[tuple] = []
            for nid, n in self._nodes.items():
                cached = self._embeddings.get(nid)
                if not cached or cached[0] != (n.get("claim") or ""):
                    continue  # узел изменился в гонке — подтянется на след. вызове
                if not self._node_is_active(n, now_dt):
                    continue  # фикс 01.09: протухшие/опровергнутые — не выдаём
                if n.get("kind") == "hub":
                    continue  # 02.09: хабы — навигация, не ответ
                score = _cosine(qv, cached[1])
                # фикс 01.09 (аудит §5.2): вес и уверенность входят и в
                # семантический ранг. Косинус лежит в [-1,1]; бонус 0..0.25
                # двигает выдачу, но не переворачивает её — смысл важнее веса.
                score += self._node_boost(n, now_dt) * 0.25
                scored.append((score, n.get("ts", ""), n))
            # по убыванию близости; при равенстве — свежие узлы первыми
            scored.sort(key=lambda t: t[1], reverse=True)
            scored.sort(key=lambda t: -t[0])
            return [(node, score) for score, _, node in scored[:k]]
