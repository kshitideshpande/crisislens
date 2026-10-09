"""Baseline 1: TF-IDF + logistic regression on tweet text.

The floor every later model must beat. It trains in seconds on CPU.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_tfidf_lr(
    C: float = 1.0,
    ngram_max: int = 2,
    min_df: int = 2,
    max_features: int | None = 50_000,
    class_weight: str | None = "balanced",
    seed: int = 42,
) -> Pipeline:
    """Word n-gram TF-IDF into a class-balanced logistic regression.

    class_weight="balanced" reweights the loss by inverse class frequency,
    the same idea as the weighted cross-entropy used for the neural models.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, ngram_max),
                    min_df=min_df,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    max_iter=3000,
                    random_state=seed,
                ),
            ),
        ]
    )


def top_features(pipeline: Pipeline, k: int = 15) -> dict[str, list[str]]:
    """Most predictive n-grams per class: a quick, honest sanity check."""
    vocab = pipeline.named_steps["tfidf"].get_feature_names_out()
    clf = pipeline.named_steps["clf"]
    coefs = clf.coef_
    classes = list(clf.classes_)
    if coefs.shape[0] == 1:  # binary: one row of weights for classes[1]
        return {
            classes[1]: [vocab[i] for i in coefs[0].argsort()[::-1][:k]],
            classes[0]: [vocab[i] for i in coefs[0].argsort()[:k]],
        }
    return {c: [vocab[i] for i in coefs[j].argsort()[::-1][:k]] for j, c in enumerate(classes)}
