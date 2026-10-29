import importlib, importlib.util
for m in ["catboost","lightgbm","xgboost","optuna","sklearn","pandas","numpy","scipy","pyarrow","torch"]:
    if importlib.util.find_spec(m):
        print(m, getattr(importlib.import_module(m), "__version__", "?"))
    else:
        print(m, "MISSING")
