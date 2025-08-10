_base_ = '../pretrain/pretrain_day_dj30_dynamic_single_vqvae_cross_sectional.py'

test_dataset = dict(
    scaler_file="state_scalers.joblib",
    scaled_data_file="state_scaled_data.joblib",
    start_timestamp="2008-04-01",
    end_timestamp="2024-04-01",
)