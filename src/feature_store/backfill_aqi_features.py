"""
Hourly AQI data ingestion pipeline
"""
import hopsworks
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import sys
import logging
import time
import uuid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data_ingestion.fetch_aqi_data import fetch_current_aqi

load_dotenv()

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        logger.warning("Empty DataFrame received")
        return df
    
    df = df.sort_values("datetime").copy()
    
    # time features
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["day_of_week"] = df["datetime"].dt.weekday
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    # aqi derived features
    df["aqi_change_rate"] = df["aqi"].diff().fillna(0)
    # rolling averages with min_periods to handle edge cases
    df["rolling_aqi_3h"] = df["aqi"].rolling(window=3, min_periods=1).mean()
    df["rolling_aqi_6h"] = df["aqi"].rolling(window=6, min_periods=1).mean()
    df["rolling_aqi_24h"] = df["aqi"].rolling(window=24, min_periods=1).mean()
    
    # generate unique event_id for each record
    df["event_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

    df["aqi"] = df["aqi"].astype("int64")  
    df["no"] = df["no"].astype("int64")
    
    required_cols = ['aqi', 'co', 'pm2_5', 'pm10']
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    
    FLOAT_COLS = [
        "co", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
        "aqi_change_rate", "rolling_aqi_3h",
        "rolling_aqi_6h", "rolling_aqi_24h"
    ]
    
    for col in FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype("float64") 
    return df

def validate_data(df: pd.DataFrame) -> bool:
    # before ingestion validate data quality
    if df.empty:
        logger.error("DataFrame is empty")
        return False
    
    required_columns = ['datetime', 'aqi', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            return False
    
    # check data types match feature group
    if not pd.api.types.is_integer_dtype(df['aqi']):
        logger.error(f"aqi should be integer, got {df['aqi'].dtype}")
        return False
    
    if not pd.api.types.is_integer_dtype(df['no']):
        logger.error(f"no should be integer, got {df['no'].dtype}")
        return False
    
    # check for reasonable ranges
    if (df['aqi'] < 0).any() or (df['aqi'] > 5).any():
        logger.warning("AQI values outside expected range 0-5")
    
    if (df['pm2_5'] < 0).any() or (df['pm2_5'] > 500).any():
        logger.warning("PM2.5 values outside expected range 0-500")
    
    return True

def hourly_ingestion_pipeline():
    """
    Main pipeline function for hourly data ingestion
    """
    logger.info(f"🚀 Starting hourly ingestion at {datetime.now()}")
    
    try:
        logger.info("🌐 Fetching current AQI data...")
        aqi_data = fetch_current_aqi()
        
        if aqi_data.empty:
            logger.warning("⚠️ No data available from API")
            return
        
        logger.info(f"📥 Fetched {len(aqi_data)} record(s)")
        
        aqi_data["datetime"] = pd.to_datetime(aqi_data["datetime"], utc=True)
        aqi_data["datetime"] = aqi_data["datetime"].dt.floor("H")  # round to nearest hour
        
        logger.info("🔧 Computing features...")
        features_df = compute_features(aqi_data)
        
        if features_df.empty:
            logger.error("❌ No data after feature computation")
            return
            
        logger.info(f"✨ Generated {len(features_df)} feature rows")
        
        if not validate_data(features_df):
            logger.error("❌ Data validation failed")
            return
        
        logger.info("🔌 Connecting to Hopsworks...")
        project = hopsworks.login(
            api_key_value=os.getenv("HOPSWORKS_API_KEY"),
            project="aqi_predicton"
        )
        fs = project.get_feature_store()
        
        fg = fs.get_feature_group(
            name="aqi_feature_group",
            version=1
        )
        
        logger.info("📋 Checking schema compatibility...")
        
        features = fg.features
        logger.info("Feature group schema:")
        for feature in features:
            logger.info(f"  - {feature.name}: {feature.type}")
        
        try:
            latest_data = fg.read()
            if not latest_data.empty and not features_df.empty:
                latest_timestamp = pd.to_datetime(latest_data['datetime'].max())
                new_timestamp = pd.to_datetime(features_df['datetime'].max())
                
                if new_timestamp <= latest_timestamp:
                    logger.warning("⏭️ Skipping ingestion - data already exists")
                    return
        except Exception as e:
            logger.warning(f"⚠️ Could not check existing data: {e}")
        
        logger.info(f"💾 Inserting {len(features_df)} new record(s)...")
        logger.info(f"Data types: aqi={features_df['aqi'].dtype}, no={features_df['no'].dtype}")
        
        logger.debug(f"Sample row to insert:\n{features_df.iloc[0].to_dict()}")
        
        fg.insert(
            features_df,
            write_options={
                "wait_for_job": True,
                "start_offline_backfill": True
            }
        )

        logger.info(f"✅ Hourly ingestion completed at {datetime.now()}")
        logger.info(f"📊 Data summary:")
        logger.info(f"   AQI: {features_df['aqi'].iloc[0]} (type: {features_df['aqi'].dtype})")
        logger.info(f"   PM2.5: {features_df['pm2_5'].iloc[0]:.1f} µg/m³")
        logger.info(f"   PM10: {features_df['pm10'].iloc[0]:.1f} µg/m³")
        
        # save local copy for debugging
        os.makedirs("data/processed/hourly", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        features_df.to_csv(f"data/processed/hourly/aqi_{timestamp}.csv", index=False)
        
        return features_df
        
    except Exception as e:
        logger.error(f"❌ Error in hourly ingestion: {str(e)}", exc_info=True)
        # don't raise - allow pipeline to continue next hour
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hourly AQI Data Ingestion")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode - run once and exit"
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuously (for testing)"
    )
    
    args = parser.parse_args()
    
    if args.continuous:
        logger.info("🔄 Running in continuous mode (hourly)")
        while True:
            result = hourly_ingestion_pipeline()
            if result is not None:
                logger.info("✅ Ingestion successful")
            else:
                logger.warning("⚠️ Ingestion failed or skipped")
            
            logger.info(f"⏳ Sleeping for 1 hour... Next run at {datetime.now() + timedelta(hours=1)}")
            time.sleep(3600)  # sleep for 1 hour
    else:
        result = hourly_ingestion_pipeline()
        if result is not None:
            print(f"✅ Ingestion successful. Inserted {len(result)} rows")
            print(f"📊 Data types check:")
            print(f"   aqi: {result['aqi'].dtype} (value: {result['aqi'].iloc[0]})")
            print(f"   no: {result['no'].dtype} (value: {result['no'].iloc[0]})")
            print(f"📈 Sample data:\n{result[['datetime', 'aqi', 'pm2_5', 'pm10']].head()}")
        else:
            print("⚠️ Ingestion failed or skipped")