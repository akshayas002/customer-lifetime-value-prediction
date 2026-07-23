"""
review_features.py

Build customer review-related features.

Author: Akshaya Sri
"""

from __future__ import annotations

import pandas as pd


def build_review_features(
    customer_orders: pd.DataFrame,
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build customer review features.

    Parameters
    ----------
    customer_orders : pd.DataFrame
        Historical customer-orders dataframe.

    reviews : pd.DataFrame
        Historical reviews dataframe.

    Returns
    -------
    pd.DataFrame
        Customer-level review features.
    """

    # ---------------------------------------
    # Customer -> Order -> Review
    # ---------------------------------------
    review_master = (
        customer_orders[
            ["customer_unique_id", "order_id"]
        ]
        .merge(
            reviews[
                [
                    "order_id",
                    "review_score"
                ]
            ],
            on="order_id",
            how="left"
        )
    )

    # ---------------------------------------
    # Flags
    # ---------------------------------------
    review_master["positive_review"] = (
        review_master["review_score"] >= 4
    ).astype(int)

    review_master["negative_review"] = (
        review_master["review_score"] <= 2
    ).astype(int)

    # ---------------------------------------
    # Aggregate
    # ---------------------------------------
    review_features = (
        review_master
        .groupby("customer_unique_id", as_index=False)
        .agg(
            average_review_score=(
                "review_score",
                "mean"
            ),

            median_review_score=(
                "review_score",
                "median"
            ),

            min_review_score=(
                "review_score",
                "min"
            ),

            max_review_score=(
                "review_score",
                "max"
            ),

            std_review_score=(
                "review_score",
                "std"
            ),

            review_count=(
                "review_score",
                "count"
            ),

            positive_review_ratio=(
                "positive_review",
                "mean"
            ),

            negative_review_ratio=(
                "negative_review",
                "mean"
            )
        )
    )

    # ---------------------------------------
    # Replace NaNs
    # ---------------------------------------
    review_features["std_review_score"] = (
        review_features["std_review_score"]
        .fillna(0)
    )

    review_features["average_review_score"] = (
        review_features["average_review_score"]
        .fillna(0)
    )

    review_features["median_review_score"] = (
        review_features["median_review_score"]
        .fillna(0)
    )

    review_features["min_review_score"] = (
        review_features["min_review_score"]
        .fillna(0)
    )

    review_features["max_review_score"] = (
        review_features["max_review_score"]
        .fillna(0)
    )

    review_features["positive_review_ratio"] = (
        review_features["positive_review_ratio"]
        .fillna(0)
    )

    review_features["negative_review_ratio"] = (
        review_features["negative_review_ratio"]
        .fillna(0)
    )

    return review_features