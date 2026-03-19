"""Этап 4: LinearSVC против LogisticRegression на словных n-граммах.

Ограниченная сетка: 2 модели x 3 диапазона n-грамм при C=1, затем по 3 значения C
для биграмм. Обучение на train, сравнение на validation.

    python scripts/compare_word_models.py
"""

from toxic_clf.experiments import run_stage

CONFIGS = []
for model in ("logreg", "linearsvc"):
    for ngram in ((1, 1), (1, 2), (1, 3)):
        CONFIGS.append({"name": f"word_{ngram[0]}{ngram[1]}_{model}_C1", "model": model, "word": {"ngram_range": ngram}})
for model, values in (("logreg", (4.0, 16.0)), ("linearsvc", (0.3, 3.0))):
    for c in values:
        CONFIGS.append(
            {"name": f"word_12_{model}_C{c:g}", "model": model, "model_params": {"C": c}, "word": {"ngram_range": (1, 2)}}
        )

if __name__ == "__main__":
    run_stage("stage4_word_models", CONFIGS)
