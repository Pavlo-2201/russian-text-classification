# russian-text-classification

Бинарная классификация токсичных русскоязычных комментариев классическими методами, без
нейросетей: TF-IDF и линейные модели, подбор на кросс-валидации, выбор порога, анализ ошибок.
Датасет и разбиения проекта затем используются в `rubert-text-classification` для сравнения
с RuBERT.

## Итоги

Лучший классический baseline — символьный TF-IDF (`char_wb`, 2–5) + LinearSVC (C=0.3,
`class_weight="balanced"`). Модель и порог выбраны 5-fold CV на train, test (2 117 текстов)
использован один раз:

| модель | оценка на | F1 toxic | macro-F1 | precision | recall |
|---|---|---|---|---|---|
| Dummy (stratified) | validation | 0.337 | 0.504 | 0.339 | 0.336 |
| word 1-граммы + LogReg | validation | 0.700 | 0.794 | 0.902 | 0.572 |
| word 1–2 + LinearSVC | validation | 0.776 | 0.839 | 0.864 | 0.704 |
| char_wb 2–5 + LinearSVC balanced | validation | 0.867 | 0.899 | 0.847 | 0.888 |
| **итог** | **test** | **0.854** | **0.889** | **0.833** | **0.877** |

95 % бутстреп-интервал F1 на test: [0.834; 0.873].

- Символьные n-граммы дают +0.06–0.07 F1 к словным: они устойчивы к опечаткам, маскировке
  мата и морфологии.
- `class_weight="balanced"` в основном сдвигает порог и даёт +0.01–0.02 F1.
- Главные ошибки: мат без токсичности (четверть таких текстов помечается токсичными) и скрытая
  агрессия без явных маркеров (пропущен каждый шестой такой токсичный текст). Часть ошибок —
  шум разметки.
- Дообученный rubert-tiny2 на тех же разбиениях даёт F1 0.884 — см. `rubert-text-classification`.

## Данные

**Russian Language Toxic Comments** — 14 412 комментариев с 2ch.hk и pikabu.ru, доля токсичных
33.5 % (Kaggle `blackmoon/russian-language-toxic-comments`, лицензия GPL-2.0). Файл берётся из
[s-nlp/rudetoxifier](https://github.com/s-nlp/rudetoxifier) по фиксированному коммиту и
проверяется по SHA-256. После дедупликации и исключения противоречивых меток — 14 112 текстов,
стратифицированное разбиение 70/15/15. Манифест разбиений лежит в `data/splits/`, сами данные
в Git не хранятся.

## Быстрый старт (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .

python scripts/download_data.py   # загрузка, проверка SHA-256 и отпечатка
python train.py                   # итоговая модель -> models/toxic_clf.joblib
python predict.py --text "Спасибо, очень полезная статья!" --text "Заткнись уже, дебил"
pytest
```

## Структура

```
src/toxic_clf/        пакет: данные, разбиения, модели, метрики, подбор, анализ ошибок
scripts/              загрузка данных и эксперименты
train.py, predict.py  обучение и применение итоговой модели
data/splits/          манифест разбиений и отпечаток данных
reports/              параметры, метрики, графики
tests/                pytest
```

## Стек

Python 3.13, pandas, scikit-learn, matplotlib, pytest.
