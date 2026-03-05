"""Дедупликация, правило противоречивых меток и воспроизводимое разбиение train/validation/test.

Правило (этап 2):
- тексты группируются по ключу: NFC, схлопнутые пробельные символы, обрезка краёв, casefold;
- группа с одинаковыми метками сохраняется одним экземпляром (наименьший source_row);
- группа с разными метками исключается целиком: метка такого текста недостоверна,
  а оставив любой вариант, мы выбрали бы метку произвольно;
- разбиение стратифицировано по метке и делается по уникальным группам, поэтому
  одинаковый текст не может попасть в два разбиения.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from toxic_clf import data

SPLITS_DIR = data.PROJECT_ROOT / "data" / "splits"
MANIFEST_PATH = SPLITS_DIR / "split_manifest.csv"
FINGERPRINT_PATH = SPLITS_DIR / "data_fingerprint.json"

SPLIT_SEED = 42
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15
SPLITS = ("train", "validation", "test")
EXCLUDED = "excluded"

_WHITESPACE = re.compile(r"\s+")


def duplicate_key(text: str) -> str:
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip().casefold()


def resolve_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет group_sha256 и exclusion_reason ('' — строка остаётся)."""
    out = df[[data.ID_COL, data.TEXT_COL, data.LABEL_COL]].copy()
    keys = out[data.TEXT_COL].map(duplicate_key)
    out["group_sha256"] = keys.map(lambda k: hashlib.sha256(k.encode("utf-8")).hexdigest())
    out = out.sort_values(data.ID_COL).reset_index(drop=True)

    groups = out.groupby("group_sha256")
    n_labels = groups[data.LABEL_COL].transform("nunique")
    first_row = groups[data.ID_COL].transform("min")

    reason = pd.Series("", index=out.index)
    is_copy = out[data.ID_COL] != first_row
    reason[is_copy] = "duplicate_of_" + first_row[is_copy].astype(str)
    reason[n_labels > 1] = "conflicting_labels"
    out["exclusion_reason"] = reason
    return out


def assign_splits(resolved: pd.DataFrame, seed: int = SPLIT_SEED) -> pd.DataFrame:
    kept = resolved[resolved["exclusion_reason"] == ""]
    train_val, test = train_test_split(
        kept, test_size=TEST_SIZE, stratify=kept[data.LABEL_COL], random_state=seed
    )
    train, val = train_test_split(
        train_val,
        test_size=VALIDATION_SIZE / (1 - TEST_SIZE),
        stratify=train_val[data.LABEL_COL],
        random_state=seed,
    )
    split = pd.Series(EXCLUDED, index=resolved.index)
    split[train.index] = "train"
    split[val.index] = "validation"
    split[test.index] = "test"
    return resolved.assign(split=split)


def build_manifest(df: pd.DataFrame, seed: int = SPLIT_SEED) -> pd.DataFrame:
    """Манифест: по строке на каждую исходную запись, без самих текстов."""
    assigned = assign_splits(resolve_duplicates(df), seed=seed)
    manifest = pd.DataFrame(
        {
            data.ID_COL: assigned[data.ID_COL],
            "text_sha256": assigned[data.TEXT_COL].map(data.text_sha256),
            "group_sha256": assigned["group_sha256"],
            data.LABEL_COL: assigned[data.LABEL_COL],
            "split": assigned["split"],
            "exclusion_reason": assigned["exclusion_reason"],
        }
    )
    return manifest.sort_values(data.ID_COL).reset_index(drop=True)


def load_manifest(path: Path = MANIFEST_PATH) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"exclusion_reason": str}, keep_default_na=False)


def load_splits(dataset: pd.DataFrame | None = None, manifest: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Возвращает {'train', 'validation', 'test'} -> DataFrame, сверяя хеши текстов с манифестом."""
    dataset = data.load_dataset() if dataset is None else dataset
    manifest = load_manifest() if manifest is None else manifest
    merged = manifest.merge(dataset, on=data.ID_COL, how="left", suffixes=("", "_dataset"), validate="one_to_one")
    if merged[data.TEXT_COL].isna().any():
        raise ValueError("В датасете нет части строк манифеста")
    if not (merged[data.TEXT_COL].map(data.text_sha256) == merged["text_sha256"]).all():
        raise ValueError("Хеши текстов не совпадают с манифестом: данные отличаются")
    if not (merged[data.LABEL_COL] == merged[f"{data.LABEL_COL}_dataset"]).all():
        raise ValueError("Метки не совпадают с манифестом")
    cols = [data.ID_COL, data.TEXT_COL, data.LABEL_COL]
    return {name: merged.loc[merged["split"] == name, cols].reset_index(drop=True) for name in SPLITS}
