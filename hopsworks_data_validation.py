import hopsworks
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def main():
    print("🔐 Logging into Hopsworks...")
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()

    FEATURE_GROUP_NAME = "aqi_feature_group"
    FEATURE_GROUP_VERSION = 1

    print("📦 Loading Feature Group...")
    fg = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION
    )

    df = fg.read()

    print("\n📊 BASIC DATA OVERVIEW")
    print(df.head())

    print("\n📐 DATA TYPES")
    print(df.dtypes)

    print("\n❓ MISSING VALUES PER COLUMN")
    print(df.isnull().sum())

    print("\n📈 NUMERICAL SUMMARY")
    print(df.describe())

    print("\n📦 DATASET SIZE INFORMATION")
    print("Duplicate Rows : ", df.duplicated().sum())
    print(f"🔢 Total Rows     : {df.shape[0]}")
    print(f"🧱 Total Columns  : {df.shape[1]}")
    print(f"🧮 Total Cells    : {df.shape[0] * df.shape[1]}")
    print(f"💾 Memory Usage   : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\n🧪 Fetching Existing Expectation Suite...")

    suite = fg.get_expectation_suite()

    if suite is None:
        print("❌ No expectation suite is attached to this Feature Group.")
        return

    print("✅ Expectation Suite FOUND")
    print(f"📛 Suite Name: {suite['expectation_suite_name']}")
    print(f"📜 Meta Info: {suite.get('meta', {})}")

    expectations = suite.get("expectations", [])

    print(f"\n📊 Total Expectations: {len(expectations)}\n")

    for idx, exp in enumerate(expectations, start=1):
        print(f"🔹 Expectation {idx}")
        print(f"   Type   : {exp['expectation_type']}")
        print(f"   Column : {exp.get('kwargs', {}).get('column', 'N/A')}")
        print(f"   Params : {exp.get('kwargs', {})}")
        print("-" * 55)


    print("\n🎉 Done!")
    print("👉 This is a READ-ONLY inspection.")
    print("👉 GUI Path: Feature Store → Feature Group → Expectations")

if __name__ == "__main__":
    main()
