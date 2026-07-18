"""Schemas Pydantic v2 da API.

As 30 features do Breast Cancer Wisconsin são geradas dinamicamente a partir da
lista canônica do sklearn. O payload aceita tanto `mean_radius` (snake_case)
quanto o nome original `mean radius` (alias), e valida presença + tipo + faixa.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, create_model

# Ordem canônica do sklearn.datasets.load_breast_cancer — NÃO alterar:
# o Pipeline foi treinado com as colunas nesta ordem.
FEATURE_ORDER: list[str] = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension",
]


def _attr(name: str) -> str:
    return name.replace(" ", "_")


_fields = {
    _attr(name): (
        float,
        Field(alias=name, ge=0, description=f"Feature '{name}' (>= 0)"),
    )
    for name in FEATURE_ORDER
}

PredictionRequest = create_model(  # type: ignore[call-overload]
    "PredictionRequest",
    __config__=ConfigDict(populate_by_name=True, extra="forbid"),
    **_fields,
)


class PredictionResponse(BaseModel):
    prediction: int = Field(description="0 = benigno, 1 = maligno")
    diagnosis: str
    probability_malignant: float = Field(ge=0, le=1)
    model_version: str
    latency_ms: float
