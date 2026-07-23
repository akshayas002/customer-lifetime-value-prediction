import pandas as pd

from src.features.temporal_split import temporal_split

from src.features.customer_features import (
    build_customer_features
)

from src.features.monetary_features import (
    build_monetary_features
)

from src.features.behavioral_features import (
    build_behavioral_features
)

from src.features.product_features import (
    build_product_features
)

from src.features.payment_features import (
    build_payment_features
)

from src.features.review_features import (
    build_review_features
)

from src.features.target_builder import (
    build_target
)



def create_clv_dataset(data):


    customers = data["customers"]
    orders = data["orders"]
    payments = data["payments"]
    order_items = data["order_items"]
    products = data["products"]
    reviews = data["reviews"]


    # ---------------------------
    # Temporal Split
    # ---------------------------

    history_orders, future_orders = temporal_split(
        orders,
        cutoff_date="2018-05-01",
        prediction_days=90
    )


    # ---------------------------
    # Customer Orders
    # ---------------------------

    customer_orders = (
        customers
        .merge(
            history_orders,
            on="customer_id",
            how="inner"
        )
    )


    # ---------------------------
    # Feature Engineering
    # ---------------------------

    features = build_customer_features(
        customer_orders
    )


    feature_modules = [

        build_monetary_features(
            customer_orders,
            payments
        ),

        build_behavioral_features(
            customer_orders
        ),

        build_product_features(
            customer_orders,
            order_items,
            products
        ),

        build_payment_features(
            customer_orders,
            payments
        ),

        build_review_features(
            customer_orders,
            reviews
        )
    ]


    for df in feature_modules:

        features = features.merge(
            df,
            on="customer_unique_id",
            how="left"
        )


    # ---------------------------
    # Target
    # ---------------------------

    target = build_target(
        customers,
        future_orders,
        payments
    )


    dataset = features.merge(
        target,
        on="customer_unique_id",
        how="left"
    )


    # Customers with no future purchase

    dataset["future_clv"] = (
        dataset["future_clv"]
        .fillna(0)
    )


    return dataset