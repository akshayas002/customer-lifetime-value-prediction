"""
product_features.py

Build customer product-related features.

Author: Akshaya Sri
"""

from __future__ import annotations

import pandas as pd


def build_product_features(
    customer_orders: pd.DataFrame,
    order_items: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build customer product features.

    Parameters
    ----------
    customer_orders : pd.DataFrame
        Historical customer-orders dataframe.

    order_items : pd.DataFrame
        Historical order items dataframe.

    products : pd.DataFrame
        Product master dataframe.

    Returns
    -------
    pd.DataFrame
        Customer-level product features.
    """

    # -----------------------------------
    # Customer -> Order -> Product
    # -----------------------------------
    product_master = (
        customer_orders[
            ["customer_unique_id", "order_id"]
        ]
        .merge(
            order_items,
            on="order_id",
            how="left"
        )
        .merge(
            products[
                [
                    "product_id",
                    "product_category_name"
                ]
            ],
            on="product_id",
            how="left"
        )
    )

    # -----------------------------------
    # Basket size (items per order)
    # -----------------------------------
    basket_size = (
        product_master
        .groupby(
            ["customer_unique_id", "order_id"]
        )
        .size()
        .reset_index(name="basket_size")
    )

    avg_basket = (
        basket_size
        .groupby("customer_unique_id", as_index=False)
        .agg(
            average_basket_size=(
                "basket_size",
                "mean"
            ),
            max_basket_size=(
                "basket_size",
                "max"
            )
        )
    )

    # -----------------------------------
    # Product level aggregation
    # -----------------------------------
    product_features = (
        product_master
        .groupby("customer_unique_id", as_index=False)
        .agg(
            unique_products=(
                "product_id",
                "nunique"
            ),

            total_products=(
                "product_id",
                "count"
            ),

            unique_categories=(
                "product_category_name",
                "nunique"
            ),

            favorite_category=(
                "product_category_name",
                lambda x: x.mode().iloc[0]
                if not x.mode().empty
                else None
            )
        )
    )

    # -----------------------------------
    # Merge basket features
    # -----------------------------------
    product_features = (
        product_features
        .merge(
            avg_basket,
            on="customer_unique_id",
            how="left"
        )
    )

    # -----------------------------------
    # Category diversity
    # -----------------------------------
    product_features["category_diversity"] = (
        product_features["unique_categories"]
        /
        product_features["total_products"]
    )

    return product_features