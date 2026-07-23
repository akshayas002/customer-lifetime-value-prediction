"""
build_dataset.py

Orchestrates the complete feature engineering pipeline.

Author: Akshaya Sri
"""

from __future__ import annotations

import pandas as pd

from src.features.customer_features import build_customer_features
from src.features.monetary_features import build_monetary_features
from src.features.behavioral_features import build_behavioral_features
from src.features.product_features import build_product_features
from src.features.payment_features import build_payment_features
from src.features.review_features import build_review_features
from src.features.target_builder import build_target


def build_dataset(
    customers: pd.DataFrame,
    history_orders: pd.DataFrame,
    future_orders: pd.DataFrame,
    order_items: pd.DataFrame,
    payments: pd.DataFrame,
    products: pd.DataFrame,
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build final ML dataset.

    Returns
    -------
    Customer-level dataframe
    """

    # --------------------------------------------------
    # Build customer-order table (history only)
    # --------------------------------------------------

    customer_orders = (
        customers.merge(
            history_orders,
            on="customer_id",
            how="inner"
        )
    )

    # --------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------

    customer_features = build_customer_features(
        customer_orders
    )

    monetary_features = build_monetary_features(
        customer_orders,
        payments
    )

    behavioral_features = build_behavioral_features(
        customer_orders
    )

    product_features = build_product_features(
        customer_orders,
        order_items,
        products
    )

    payment_features = build_payment_features(
        customer_orders,
        payments
    )

    review_features = build_review_features(
        customer_orders,
        reviews
    )

    # --------------------------------------------------
    # Target
    # --------------------------------------------------

    target = build_target(
        customers,
        future_orders,
        payments
    )

    # --------------------------------------------------
    # Merge everything
    # --------------------------------------------------

    dataset = (
        customer_features
        .merge(
            monetary_features,
            on="customer_unique_id",
            how="left"
        )
        .merge(
            behavioral_features,
            on="customer_unique_id",
            how="left"
        )
        .merge(
            product_features,
            on="customer_unique_id",
            how="left"
        )
        .merge(
            payment_features,
            on="customer_unique_id",
            how="left"
        )
        .merge(
            review_features,
            on="customer_unique_id",
            how="left"
        )
        .merge(
            target,
            on="customer_unique_id",
            how="left"
        )
    )

    # --------------------------------------------------
    # Customers without future purchases
    # --------------------------------------------------

    dataset["future_clv"] = (
        dataset["future_clv"]
        .fillna(0)
    )

    return dataset