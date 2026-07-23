"""
behavioral_features.py

Build customer behavioral features from historical orders.

Author: Akshaya Sri
"""

from __future__ import annotations

import pandas as pd


def build_behavioral_features(
    customer_orders: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build customer behavioral features.

    Parameters
    ----------
    customer_orders : pd.DataFrame
        Historical customer-orders dataframe.

    Returns
    -------
    pd.DataFrame
        Customer-level behavioral features.
    """

    # ---------------------------------------
    # Sort orders chronologically
    # ---------------------------------------
    customer_orders = (
        customer_orders
        .sort_values(
            [
                "customer_unique_id",
                "order_purchase_timestamp"
            ]
        )
        .copy()
    )

    # ---------------------------------------
    # Days between consecutive purchases
    # ---------------------------------------
    customer_orders["days_between_orders"] = (
        customer_orders
        .groupby("customer_unique_id")[
            "order_purchase_timestamp"
        ]
        .diff()
        .dt.days
    )

    # ---------------------------------------
    # Purchase month / weekday / hour
    # ---------------------------------------
    customer_orders["purchase_month"] = (
        customer_orders["order_purchase_timestamp"]
        .dt.month
    )

    customer_orders["purchase_weekday"] = (
        customer_orders["order_purchase_timestamp"]
        .dt.dayofweek
    )

    customer_orders["purchase_hour"] = (
        customer_orders["order_purchase_timestamp"]
        .dt.hour
    )

    # ---------------------------------------
    # Weekend purchase
    # ---------------------------------------
    customer_orders["is_weekend"] = (
        customer_orders["purchase_weekday"] >= 5
    ).astype(int)

    # ---------------------------------------
    # Night purchase
    # (10 PM - 6 AM)
    # ---------------------------------------
    customer_orders["is_night"] = (
        (
            customer_orders["purchase_hour"] >= 22
        )
        |
        (
            customer_orders["purchase_hour"] <= 6
        )
    ).astype(int)

    # ---------------------------------------
    # Aggregate
    # ---------------------------------------
    behavioral_features = (
        customer_orders
        .groupby("customer_unique_id", as_index=False)
        .agg(
            avg_days_between_orders=(
                "days_between_orders",
                "mean"
            ),

            std_days_between_orders=(
                "days_between_orders",
                "std"
            ),

            favorite_purchase_month=(
                "purchase_month",
                lambda x: x.mode().iloc[0]
            ),

            favorite_purchase_weekday=(
                "purchase_weekday",
                lambda x: x.mode().iloc[0]
            ),

            favorite_purchase_hour=(
                "purchase_hour",
                lambda x: x.mode().iloc[0]
            ),

            weekend_purchase_ratio=(
                "is_weekend",
                "mean"
            ),

            night_purchase_ratio=(
                "is_night",
                "mean"
            )
        )
    )

    # ---------------------------------------
    # Replace NaNs
    # ---------------------------------------
    behavioral_features[
        "avg_days_between_orders"
    ] = (
        behavioral_features[
            "avg_days_between_orders"
        ]
        .fillna(0)
    )

    behavioral_features[
        "std_days_between_orders"
    ] = (
        behavioral_features[
            "std_days_between_orders"
        ]
        .fillna(0)
    )

    return behavioral_features