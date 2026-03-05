import pandas as pd
import pytest

from toxic_clf import data, splits


def toy_dataset(n: int = 200) -> pd.DataFrame:
    texts = [f"Комментарий номер {i}" for i in range(n)]
    labels = [int(i % 3 == 0) for i in range(n)]
    df = pd.DataFrame({data.ID_COL: range(n), data.TEXT_COL: texts, data.LABEL_COL: labels})
    extra = pd.DataFrame(
        {
            data.ID_COL: [n, n + 1, n + 2],
            # копия с хвостовым переводом строки и другим регистром; копия с противоречивой меткой
            data.TEXT_COL: ["КОММЕНТАРИЙ номер 1\n", "Комментарий  номер 2", "Комментарий номер 3 "],
            data.LABEL_COL: [0, 0, 0],
        }
    )
    return pd.concat([df, extra], ignore_index=True)


def test_duplicate_key_ignores_whitespace_and_case():
    assert splits.duplicate_key("Привет,  Мир!\n") == splits.duplicate_key("привет, мир!")
    assert splits.duplicate_key("привет, мир") != splits.duplicate_key("привет мир")


def test_resolve_duplicates_rules():
    resolved = splits.resolve_duplicates(toy_dataset()).set_index(data.ID_COL)["exclusion_reason"]
    assert resolved[200] == "duplicate_of_1"  # совпадает с 1, метки одинаковые
    assert resolved[1] == ""  # первый экземпляр сохраняется
    assert resolved[201] == "duplicate_of_2"
    assert resolved[3] == resolved[202] == "conflicting_labels"  # 3 токсичный, копия нет


def test_splits_disjoint_cover_and_stratified():
    manifest = splits.build_manifest(toy_dataset())
    kept = manifest[manifest["split"] != splits.EXCLUDED]
    assert set(kept["split"]) == set(splits.SPLITS)
    assert kept["group_sha256"].is_unique  # одинаковые тексты не попадают в разные разбиения
    assert len(manifest) == len(toy_dataset())
    shares = kept.groupby("split")[data.LABEL_COL].mean()
    assert (shares - kept[data.LABEL_COL].mean()).abs().max() < 0.05


def test_manifest_is_deterministic():
    df = toy_dataset()
    pd.testing.assert_frame_equal(splits.build_manifest(df), splits.build_manifest(df.sample(frac=1, random_state=1)))


def test_load_splits_detects_changed_text():
    df = toy_dataset()
    manifest = splits.build_manifest(df)
    tampered = df.copy()
    tampered.loc[5, data.TEXT_COL] = "другой текст"
    with pytest.raises(ValueError):
        splits.load_splits(tampered, manifest)


@pytest.mark.skipif(not data.DATASET_FILE.exists(), reason="датасет не скачан")
def test_committed_manifest_matches_data():
    manifest = splits.load_manifest()
    assert manifest.equals(splits.build_manifest(data.load_dataset()))
    parts = splits.load_splits(manifest=manifest)
    keys = {name: set(part[data.TEXT_COL].map(splits.duplicate_key)) for name, part in parts.items()}
    assert not keys["train"] & keys["validation"]
    assert not keys["train"] & keys["test"]
    assert not keys["validation"] & keys["test"]
