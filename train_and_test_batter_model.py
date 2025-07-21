from batter_reaction import BatterReactionModel
import pprint

if __name__ == "__main__":
    # 1. Initialize model
    model = BatterReactionModel()
    
    # 2. Load data for Aaron Judge (2023)
    print("Loading data for Aaron Judge (2023)...")
    df = model.load_batter_data('Aaron Judge', '2023-01-01', '2023-12-31')
    print(f"Loaded {len(df)} pitches.")
    
    # 3. Preprocess and filter
    clean_df = model.preprocess_data(df)
    print(f"After cleaning: {len(clean_df)} pitches.")
    if len(clean_df) < 500:
        print("Not enough data to train. Try a different batter or date range.")
        exit(1)
    
    # 4. Train model
    print("Training model...")
    model.train(clean_df)
    print("Model trained.")
    
    # 5. Test prediction
    test_pitch = {
        'pitch_type': 'FF',
        'release_speed': 96.0,
        'release_spin_rate': 2400,
        'plate_x': 0.0,   # center of plate (inches)
        'plate_z': 36.0,  # 3 feet (inches)
        'balls': 1,
        'strikes': 1,
        'batter_hand': 'R',
        'pitcher_hand': 'R',
        'outs_when_up': 1,
        'inning': 7,
        # Optional context features (will default to 0 if missing)
    }
    print("\nSample prediction for test pitch:")
    result = model.predict_reaction(test_pitch)
    pprint.pprint(result) 