from pathlib import Path
import pandas as pd


class OlistDataLoader:
    """
    Loads all Olist datasets from the raw data directory.
    """

    def __init__(self, data_path=None):
        if data_path is None:
            # Project Root
            project_root = Path(__file__).resolve().parents[2]
            self.data_path = project_root / "data" / "raw"
        else:
            self.data_path = Path(data_path)

    def load_data(self):

        datasets = {
            "customers": "olist_customers_dataset.csv",
            "orders": "olist_orders_dataset.csv",
            "order_items": "olist_order_items_dataset.csv",
            "payments": "olist_order_payments_dataset.csv",
            "products": "olist_products_dataset.csv",
            "reviews": "olist_order_reviews_dataset.csv",
            "sellers": "olist_sellers_dataset.csv",
            "geolocation": "olist_geolocation_dataset.csv",
            "category_translation": "product_category_name_translation.csv",
        }

        data = {}

        for name, file in datasets.items():
            data[name] = pd.read_csv(self.data_path / file)

        # Convert datetime columns
        date_columns = [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]

        data["orders"][date_columns] = (
            data["orders"][date_columns]
            .apply(pd.to_datetime)
        )

        return data