from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class ScaleDef:
    id: str
    label: str
    left_label: str
    right_label: str
    left_anchors: List[str]
    right_anchors: List[str]

SCALES: List[ScaleDef] = [
    ScaleDef(
        id="urgent_vs_analytical",
        label="Срочность",
        left_label="Аналитика",
        right_label="Срочно",
        left_anchors=[
            "Подробный аналитический разбор ситуации",
            "Долгосрочные тренды и объяснения контекста",
        ],
        right_anchors=[
            "Срочная новость, только что произошло",
            "Breaking news, немедленное обновление",
        ],
    ),
    ScaleDef(
        id="practical_vs_background",
        label="Практическое влияние",
        left_label="Фон",
        right_label="Практично",
        left_anchors=[
            "Общий обзор новостей без прямого влияния",
        ],
        right_anchors=[
            "Новость, которая влияет на деньги, работу или безопасность читателя",
        ],
    ),
    ScaleDef(
        id="local_vs_global",
        label="Масштаб",
        left_label="Локально",
        right_label="Глобально",
        left_anchors=[
            "Новость о событиях в одном городе или регионе",
        ],
        right_anchors=[
            "Международная новость, затрагивающая несколько стран",
        ],
    ),
    ScaleDef(
        id="markets_vs_politics",
        label="Фокус",
        left_label="Рынки",
        right_label="Политика",
        left_anchors=[
            "Рынки, акции, экономика, курсы валют",
        ],
        right_anchors=[
            "Политика, выборы, власть, государство",
        ],
    ),
    ScaleDef(
        id="certainty_vs_uncertainty",
        label="Определённость",
        left_label="Определённо",
        right_label="Неопределённо",
        left_anchors=[
            "Уверенные прогнозы и подтверждённые факты",
        ],
        right_anchors=[
            "Неясные исходы, неопределённость, прогнозы и сценарии",
        ],
    ),
    ScaleDef(
        id="risk_vs_opportunity",
        label="Риск / Возможность",
        left_label="Риск",
        right_label="Возможность",
        left_anchors=[
            "Угроза, риск, потери, негативные последствия",
        ],
        right_anchors=[
            "Возможность, рост, шанс заработать или выиграть",
        ],
    ),
    ScaleDef(
        id="conflict_vs_cooperation",
        label="Динамика",
        left_label="Конфликт",
        right_label="Сотрудничество",
        left_anchors=[
            "Конфликт, протест, противостояние, война",
        ],
        right_anchors=[
            "Переговоры, сотрудничество, соглашение",
        ],
    ),
    ScaleDef(
        id="trend_vs_background",
        label="Трендовость",
        left_label="Фон",
        right_label="Тренд",
        left_anchors=[
            "Обычная повторяющаяся новость",
        ],
        right_anchors=[
            "Набирающий силу тренд, новая тема, хайп",
        ],
    ),
]