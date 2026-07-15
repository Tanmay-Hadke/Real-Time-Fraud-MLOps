from datetime import timedelta
from feast import (
    Entity,
    Field,
    FeatureView,
    FileSource,
    ValueType,
)
from feast.types import Float32, Int64

# Point to local offline baseline file source
file_source = FileSource(
    path="../data/baseline_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Define core target tracking Entity
user = Entity(name="user_id", value_type=ValueType.STRING, description="Bank Customer ID")

# Define structural Feature View mapping PCA components and transaction values
fraud_feature_view = FeatureView(
    name="user_transaction_features",
    entities=[user],
    ttl=timedelta(days=90),
    schema=[
        Field(name="Amount", dtype=Float32),
        Field(name="Class", dtype=Int64),
        *[Field(name=f"V{i}", dtype=Float32) for i in range(1, 29)]
    ],
    online=True,
    source=file_source,
)