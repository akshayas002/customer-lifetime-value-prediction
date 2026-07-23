"""
target_builder.py

Build future Customer Lifetime Value (Target Variable)

Author: Akshaya Sri
"""

from __future__ import annotations

import pandas as pd


def build_target(
    customers: pd.DataFrame,
    future_orders: pd.DataFrame,
    payments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build future CLV target.

    Parameters
    ----------
    customers : DataFrame

    future_orders : DataFrame
        Orders AFTER cutoff date.

    payments : DataFrame

    Returns
    -------
    DataFrame

        customer_unique_id

        future_clv
    """

    # -----------------------------
    # Customer -> Future Orders
    # -----------------------------
    future_customer_orders = (
        customers[
            [
                "customer_id",
                "customer_unique_id"
            ]
        ]
        .merge(
            future_orders[
                [
                    "order_id",
                    "customer_id"
                ]
            ],
            on="customer_id",
            how="inner"
        )
    )

    # -----------------------------
    # Order Value
    # -----------------------------
    future_order_value = (
        payments
        .groupby(
            "order_id",
            as_index=False
        )
        .agg(
            order_value=(
                "payment_value",
                "sum"
            )
        )
    )

    # -----------------------------
    # Merge
    # -----------------------------
    target = (
        future_customer_orders
        .merge(
            future_order_value,
            on="order_id",
            how="left"
        )
    )

    # -----------------------------
    # Aggregate
    # -----------------------------
    target = (
        target
        .groupby(
            "customer_unique_id",
            as_index=False
        )
        .agg(
            future_clv=(
                "order_value",
                "sum"
            )
        )
    )

    return target