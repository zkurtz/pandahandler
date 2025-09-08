import lightgbm
import numpy as np
import pandas as pd
import xgboost as xgb

from pandahandler.sklearn.transforms.categoricals import (
    Encoder,
    infer_categoricals,
)


def test_infer_categoricals():
    """Test the infer_categoricals function."""
    df = pd.DataFrame(
        {
            "cats": ["siamese", "little", "persian"],
            "timestamp": ["2021-01-01", None, "2021-01-02"],
            "numeric": [1, 2, 3],
        }
    )
    result = infer_categoricals(df)
    assert set(result) == {"cats", "timestamp"}


def test_encoder_basic():
    """Test the Encoder class, focusing on basic functionality including default behavior of ignoring numeric cols."""
    training_df = pd.DataFrame(
        {
            "color": ["red", "red", "black"],
            "unit": ["liter", None, "meter"],
            "numbers": [1, 2, 3],
        }
    )
    encoder = Encoder()
    encoder.fit(training_df)

    # Basic prediction checks
    pred_df = training_df.copy()
    transformed_df = encoder.transform(pred_df)
    assert transformed_df.dtypes.astype(str).to_list() == [
        "category",
        "category",
        "int64",
    ]
    assert transformed_df["color"].cat.codes.to_list() == [1, 1, 0]
    assert transformed_df["unit"].cat.codes.to_list() == [0, -1, 1]


def test_encoder():
    """Test the Encoder class for handling of new values and null values."""
    training_df = pd.DataFrame(
        {
            "color": ["red", "red", "black"],
            "unit": ["liter", None, "meter"],
        }
    )
    encoder = Encoder(specified_columns=["color"])
    encoder.fit(training_df)

    # Now suppose the prediction data includes two new values of color: grey and None
    pred_df = pd.DataFrame(
        {
            "color": ["grey", None],
            "unit": ["meter", "liter"],
        },
    )
    transformed_df = encoder.transform(pred_df)
    # Here "grey" maps to 2, a never-before-seen value
    assert transformed_df["color"].cat.codes.to_list() == [2, -1]

    # The unit column is not transformed to categorical since it was not included in `specified_columns`:
    assert transformed_df["unit"].dtype == "object"

    # Actually, any column not in self.encoders may be ignored:
    pred_df = pd.DataFrame({"color": ["grey", "red"]})
    transformed_df = encoder.transform(pred_df)
    assert transformed_df["color"].cat.codes.to_list() == [2, 1]


def test_encoder_xgboost():
    """Test the Encoder for expected behavior with XGBoost."""
    # Create a miniature ML dataset with a single categorical feature (including a null value) and continuous target
    train_df = pd.DataFrame(
        {
            "unit": ["liter", "meter", None],
            "target": [1, 2, 3],
        }
    )

    # Train the encoder and xgboost model
    encoder = Encoder()
    encoder.fit(train_df)
    train_df = encoder.transform(train_df)

    # XGBoost requires DMatrix format
    dtrain = xgb.DMatrix(
        train_df[["unit"]],
        label=train_df["target"],
        enable_categorical=True,
    )
    params = {
        "objective": "reg:squarederror",  # Regression objective
        "max_depth": 6,  # Default tree depth
        "eta": 1.0,  # Learning rate equivalent
        "lambda": 0,  # L2 regularization (equivalent to reg_lambda)
        "alpha": 0,  # L1 regularization (equivalent to reg_alpha)
        "min_child_weight": 0,  # Similar to min_child_weight in LightGBM
        "verbosity": 0,  # Quiet output (-1 not available in XGBoost)
    }
    xgb_model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1,
    )

    # Now predict on data that includes a new input category:
    pred_df = pd.DataFrame({"unit": ["liter", "meter", None, "new-value"]})
    pred_df = encoder.transform(pred_df)
    dpred = xgb.DMatrix(pred_df, enable_categorical=True)
    pred_df["preds"] = xgb_model.predict(dpred)

    # The interesting parts here are
    # 1. XGBoost learns a value for None (i.e. 3.0)
    # 2. The new category "new_value" reverts to the grand mean, since it was never seen before:
    assert np.allclose(pred_df["preds"].to_numpy(), np.array([1.0, 2.0, 3.0, 2.0]), rtol=1e-5)

    # Verify that all of this fancy categorical handling was necessary: Naively doing categorical
    # encoding can result in totally new values being assigned to the same codes as previously seen values,
    # leading to nonsense predictions:
    pred_df2 = pd.DataFrame({"unit": ["inch", "liter"]}).astype("category")
    dpred2 = xgb.DMatrix(pred_df2, enable_categorical=True)
    pred_df2["preds"] = xgb_model.predict(dpred2)
    # Now we see that the same input "liter" is assigned a different value (2.0) than it was in the training data (1.0)
    assert pred_df2.set_index("unit")["preds"].loc["liter"] == 2.0
    assert pred_df.set_index("unit")["preds"].loc["liter"] == 1.0


def test_encoder_lightgbm():
    """Test the Encoder for expected behavior with LightGBM."""
    # Create a miniature ML dataset with a single categorical feature (including a null value) and continuous target
    train_df = pd.DataFrame(
        {
            "unit": ["liter", "meter", None],
            "target": [1, 2, 3],
        }
    )

    # Train the encoder and lightgbm model
    encoder = Encoder()
    encoder.fit(train_df)
    train_df = encoder.transform(train_df)
    lgbm = lightgbm.LGBMRegressor(
        objective="regression",
        n_estimators=1,
        verbose=-1,
        # Zero out all regularization terms for this minimal test:
        learning_rate=1,
        min_data_in_bin=1,
        min_child_samples=1,
        min_child_weight=0,  # Reduce regularization
        reg_alpha=0,  # L1 regularization term on weights
        reg_lambda=0,  # L2 regularization term on weights
    )
    lgbm.fit(X=train_df[["unit"]], y=train_df["target"], categorical_feature=["unit"])

    # Now predict on data that includes a new input category:
    pred_df = pd.DataFrame(
        {
            "unit": ["liter", "meter", None, "new-value"],
        }
    )
    pred_df = encoder.transform(pred_df)
    preds = np.array(lgbm.predict(pred_df))
    # The interesting parts here are
    # 1. LightGBM learned a value for None (i.e. 3.0)
    # 2. The new category "new-value" gets the same value as None. This kinda makes sense but might not be ideal;
    #    xgboost by contrast (above) reverts to the grand mean for new categories.
    assert np.allclose(preds, np.array([1.0, 2.0, 3.0, 3.0]), rtol=1e-5)
