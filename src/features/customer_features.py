"""
customer_features.py

Build customer-level behavioral and temporal features
from historical order data.

Author: Akshaya Sri
"""

from __future__ import annotations

import pandas as pd


def build_customer_features(customer_orders: pd.DataFrame) -> pd.DataFrame:
    """
    Build customer-level features.

    Parameters
    ----------
    customer_orders : pd.DataFrame
        Merged customers + historical orders dataframe.

    Returns
    -------
    pd.DataFrame
        Customer-level feature dataframe.
    """

    # -----------------------------
    # Aggregate customer statistics
    # -----------------------------
    customer_features = (
        customer_orders
        .groupby("customer_unique_id", as_index=False)
        .agg(
            first_purchase=(
                "order_purchase_timestamp",
                "min"
            ),
            last_purchase=(
                "order_purchase_timestamp",
                "max"
            ),
            total_orders=(
                "order_id",
                "nunique"
            )
        )
    )

    # -----------------------------
    # Reference date
    # -----------------------------
    reference_date = customer_orders[
        "order_purchase_timestamp"
    ].max()

    # -----------------------------
    # Time-based features
    # -----------------------------
    customer_features["customer_tenure_days"] = (
        reference_date
        - customer_features["first_purchase"]
    ).dt.days

    customer_features["recency_days"] = (
        reference_date
        - customer_features["last_purchase"]
    ).dt.days

    # -----------------------------
    # Repeat customer flag
    # -----------------------------
    customer_features["is_repeat_customer"] = (
        customer_features["total_orders"] > 1
    ).astype(int)

    # -----------------------------
    # Purchase frequency
    # -----------------------------
    customer_features["purchase_frequency"] = (
        customer_features["total_orders"]
        /
        customer_features["customer_tenure_days"].replace(0, 1)
    )

    # -----------------------------
    # Average days between purchases
    # -----------------------------
    customer_features["avg_days_between_orders"] = (
        customer_features["customer_tenure_days"]
        /
        (customer_features["total_orders"] - 1).replace(0, 1)
    )

    return customer_features