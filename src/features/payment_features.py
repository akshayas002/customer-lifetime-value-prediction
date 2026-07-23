"""
payment_features.py

Build customer payment-related features.

Author: Akshaya Sri
"""

from __future__ import annotations

import pandas as pd


def build_payment_features(
    customer_orders: pd.DataFrame,
    payments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build customer payment features.

    Parameters
    ----------
    customer_orders : pd.DataFrame
        Historical customer-orders dataframe.

    payments : pd.DataFrame
        Historical payments dataframe.

    Returns
    -------
    pd.DataFrame
        Customer-level payment features.
    """

    # ---------------------------------------
    # Customer -> Order -> Payment
    # ---------------------------------------
    payment_master = (
        customer_orders[
            ["customer_unique_id", "order_id"]
        ]
        .merge(
            payments,
            on="order_id",
            how="left"
        )
    )

    # ---------------------------------------
    # Aggregate payment features
    # ---------------------------------------
    payment_features = (
        payment_master
        .groupby("customer_unique_id", as_index=False)
        .agg(
            average_installments=(
                "payment_installments",
                "mean"
            ),

            max_installments=(
                "payment_installments",
                "max"
            ),

            total_payment_transactions=(
                "payment_type",
                "count"
            ),

            payment_method_diversity=(
                "payment_type",
                "nunique"
            ),

            favorite_payment_type=(
                "payment_type",
                lambda x: x.mode().iloc[0]
                if not x.mode().empty
                else None
            )
        )
    )

    return payment_features