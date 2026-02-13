import hopsworks
import os
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import sys
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

def daily_training_pipeline():
    logger.info(f"🚀 Starting daily training at {datetime.now()}")
    
    try:
        logger.info("🔌 Connecting to Hopsworks...")
        project = hopsworks.login(
            api_key_value=os.getenv("HOPSWORKS_API_KEY"),
            project="aqi_predicton"
        )
        fs = project.get_feature_store()
        mr = project.get_model_registry()
        
        logger.info("📋 Getting feature view...")
        fv = fs.get_feature_view(name="aqi_feature_view", version=1)
        
        logger.info("🔄 Creating new training dataset...")
        td_version = fv.create_training_data(
            description=f"Daily training data - {datetime.now().date()}",
            write_options={"wait_for_job": True}
        )
        
        # Extract version if td_version is a tuple (version, path)
        actual_td_version = td_version[0] if isinstance(td_version, tuple) else td_version
        logger.info(f"📊 Training dataset version {actual_td_version} created")
        
        # Give Hopsworks a moment to sync metadata
        import time
        logger.info("⏳ Waiting for metadata synchronization...")
        time.sleep(10)
        
        logger.info("📥 Fetching training data...")
        X, y = fv.get_training_data(
            training_dataset_version=actual_td_version,
            create_training_dataset=False
        )
        
        logger.info(f"📈 Training data shape: X={X.shape}, y={y.shape}")
        
        columns_to_remove = ['datetime', 'event_id']
        existing_cols = [col for col in columns_to_remove if col in X.columns]
        
        if existing_cols:
            logger.info(f"🗑️ Removing columns: {existing_cols}")
            X = X.drop(columns=existing_cols)
        
        if isinstance(y, pd.DataFrame):
            y = y.values.ravel()
        elif isinstance(y, pd.Series):
            y = y.values
        
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            logger.warning(f"⚠️ Removing non-numeric columns: {non_numeric_cols}")
            X = X.select_dtypes(include=[np.number])
        
        feature_names = X.columns.tolist()
        logger.info(f"🎯 Using {len(feature_names)} features")
        
        X_array = X.values
        y_array = y
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_array)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_array, test_size=0.2, random_state=42, shuffle=True
        )
        
        logger.info(f"📊 Data split: Train={X_train.shape[0]}, Val={X_val.shape[0]}")
        
        models = {
            "ridge": Ridge(alpha=1.0),
            "random_forest": RandomForestRegressor(
                n_estimators=200, 
                max_depth=12, 
                random_state=42, 
                n_jobs=-1,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=200, 
                learning_rate=0.05, 
                max_depth=5, 
                random_state=42,
                subsample=0.8
            )
        }
        
        results = []
        
        for name, model in models.items():
            logger.info(f"🤖 Training {name}...")
            model.fit(X_train, y_train)
            
            train_preds = model.predict(X_train)
            val_preds = model.predict(X_val)
            
            train_mae = mean_absolute_error(y_train, train_preds)
            val_mae = mean_absolute_error(y_val, val_preds)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
            val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
            
            train_r2 = r2_score(y_train, train_preds)
            val_r2 = r2_score(y_val, val_preds)
            
            results.append({
                "model": name,
                "train_mae": train_mae,
                "val_mae": val_mae,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "train_r2": train_r2,
                "val_r2": val_r2
            })
            
            logger.info(f"   {name}: R²={val_r2:.4f}, RMSE={val_rmse:.4f}, MAE={val_mae:.4f}")
        
        results_df = pd.DataFrame(results)
        
        best_model_idx = results_df['val_r2'].idxmax()
        best_model_name = results_df.loc[best_model_idx, 'model']
        best_model = models[best_model_name]
        
        logger.info(f"🏆 Best model: {best_model_name} with R²={results_df.loc[best_model_idx, 'val_r2']:.4f}")
        
        artifacts_dir = "models_artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)
        
        joblib.dump(best_model, os.path.join(artifacts_dir, f"{best_model_name}_model.pkl"))
        joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.pkl"))
        joblib.dump(feature_names, os.path.join(artifacts_dir, "feature_names.pkl"))
        results_df.to_csv(os.path.join(artifacts_dir, "model_comparison.csv"), index=False)
        
        logger.info(f"📦 Registering model: {best_model_name}")
        
        best_metrics = results_df[results_df["model"] == best_model_name].iloc[0].to_dict()
        
        model = mr.python.create_model(
            name="aqi_gradient_boosting_model",
            metrics={
                "val_r2": float(best_metrics['val_r2']),
                "val_rmse": float(best_metrics['val_rmse']),
                "val_mae": float(best_metrics['val_mae']),
                "train_r2": float(best_metrics['train_r2']),
                "train_rmse": float(best_metrics['train_rmse']),
                "train_mae": float(best_metrics['train_mae'])
            },
            description=f"AQI Prediction Model - Trained {datetime.now().date()}",
            input_example=X.iloc[:1]
        )
        
        model_dir = f"registry_artifacts/model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(best_model, os.path.join(model_dir, "model.pkl"))
        joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
        joblib.dump(feature_names, os.path.join(model_dir, "feature_names.pkl"))
        results_df.to_csv(os.path.join(model_dir, "model_comparison.csv"), index=False)
        
        import json
        with open(os.path.join(model_dir, "metrics.json"), "w") as f:
            json.dump(best_metrics, f, indent=2)
        
        model.save(model_dir)
        
        logger.info(f"✅ Daily training completed. Model version {model.version} registered")
        logger.info(f"📊 Model metrics: R²={best_metrics['val_r2']:.4f}, RMSE={best_metrics['val_rmse']:.4f}")
        
        return model.version
        
    except Exception as e:
        logger.error(f"❌ Error in daily training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Daily Model Training Pipeline")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode - train locally without registration"
    )
    
    args = parser.parse_args()
    
    if args.test:
        logger.info("🧪 Running in test mode (local training only)")
        project = hopsworks.login(
            api_key_value=os.getenv("HOPSWORKS_API_KEY"),
            project="aqi_predicton"
        )
        fs = project.get_feature_store()
        fv = fs.get_feature_view(name="aqi_feature_view", version=1)
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import r2_score
        
        X, y = fv.get_training_data()
        X = X.drop(columns=['datetime', 'event_id'], errors='ignore')
        X = X.select_dtypes(include=[np.number])
        
        if isinstance(y, pd.DataFrame):
            y = y.values.ravel()
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_val)
        score = r2_score(y_val, preds)
        
        print(f"✅ Test complete. R² score: {score:.4f}")
        
    else:
        model_version = daily_training_pipeline()
        print(f"✅ Pipeline completed. Model version: {model_version}")