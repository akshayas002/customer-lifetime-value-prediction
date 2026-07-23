"""
monetary_features.py

Build customer-level monetary features
from historical payment data.

Author: Akshaya Sri
"""

from __future__ import annotations

import pandas as pd


def build_monetary_features(
    customer_orders: pd.DataFrame,
    payments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build monetary features for each customer.

    Parameters
    ----------
    customer_orders : pd.DataFrame
        Historical customer-orders dataframe.

    payments : pd.DataFrame
        Historical payments dataframe.

    Returns
    -------
    pd.DataFrame
        Customer-level monetary features.
    """

    # ---------------------------------
    # Calculate total payment per order
    # ---------------------------------
    order_value = (
        payments
        .groupby("order_id", as_index=False)
        .agg(
            order_value=("payment_value", "sum")
        )
    )

    # ---------------------------------
    # Merge order values with customers
    # ---------------------------------
    customer_order_values = (
        customer_orders[
            ["customer_unique_id", "order_id"]
        ]
        .merge(
            order_value,
            on="order_id",
            how="left"
        )
    )

    # ---------------------------------
    # Aggregate monetary features
    # ---------------------------------
    monetary_features = (
        customer_order_values
        .groupby("customer_unique_id", as_index=False)
        .agg(
            total_spent=("order_value", "sum"),
            average_order_value=("order_value", "mean"),
            median_order_value=("order_value", "median"),
            max_order_value=("order_value", "max"),
            min_order_value=("order_value", "min"),
            std_order_value=("order_value", "std")
        )
    )

    # ---------------------------------
    # Replace NaN std (single order)
    # ---------------------------------
    monetary_features["std_order_value"] = (
        monetary_features["std_order_value"]
        .fillna(0)
    )

    return monetary_features