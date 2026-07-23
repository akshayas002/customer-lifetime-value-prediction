import pandas as pd


def temporal_split(
    orders: pd.DataFrame,
    cutoff_date: str,
    prediction_days: int = 90
):

    cutoff_date = pd.to_datetime(cutoff_date)

    prediction_end = (
        cutoff_date
        + pd.Timedelta(days=prediction_days)
    )

    history_orders = orders[
        orders["order_purchase_timestamp"] <= cutoff_date
    ].copy()


    future_orders = orders[
        (
            orders["order_purchase_timestamp"] > cutoff_date
        )
        &
        (
            orders["order_purchase_timestamp"] <= prediction_end
        )
    ].copy()


    return history_orders, future_orders