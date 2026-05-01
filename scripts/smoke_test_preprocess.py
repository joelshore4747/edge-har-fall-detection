from pipeline.ingest.runtime_phone_folder import load_runtime_phone_folder
from pipeline.preprocess.prepare import prepare_windowed_sequences
from pipeline.preprocess.config import default_preprocess_config
from pipeline.features.build_feature_table import build_feature_table, feature_table_schema_summary

PHONE_FOLDER_PATH = "../phone1"  # change if needed

cfg = default_preprocess_config()

df = load_runtime_phone_folder(PHONE_FOLDER_PATH)
windows = prepare_windowed_sequences(df, config=cfg)
feature_df = build_feature_table(windows, filter_unacceptable=False)

print("=== PREPROCESS SMOKE TEST ===")
print("Raw rows:", len(df))
print("Windows:", len(windows))
print("Feature rows:", len(feature_df))
print("First columns:", list(feature_df.columns[:20]))
print()

if windows:
    print("First window acceptable:", windows[0].get("is_acceptable"))
    print("First window label:", windows[0].get("label_mapped_majority"))
    print("First window missing ratio:", windows[0].get("missing_ratio"))
    print("First window quality summary:", windows[0].get("quality_summary"))
    print()

print(feature_df.head())
print()
print(feature_table_schema_summary(feature_df))