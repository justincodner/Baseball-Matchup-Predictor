from batter_reaction import BatterReactionModel
import pprint

if __name__ == "__main__":
    # 1. Initialize model
    model = BatterReactionModel()
    
    # 2. Load data for Aaron Judge (2023)
    print("Loading data for Aaron Judge (2024)...")
    df = model.load_batter_data('Aaron Judge', '2024-01-01', '2024-12-31')
    print(f"Loaded {len(df)} pitches.")
    
    # 3. Preprocess and filter
    clean_df = model.preprocess_data(df)
    print(f"After cleaning: {len(clean_df)} pitches.")
    if len(clean_df) < 500:
        print("Not enough data to train. Try a different batter or date range.")
        exit(1)
    print("All zones present in clean_df:", sorted(clean_df['zone'].unique()))
    print(f"Total pitches: {len(clean_df)}")
    zone_5 = clean_df[clean_df['zone'] == 5]
    print(zone_5['outcome'].value_counts(normalize=True))
    
    # 4. Train model
    print("Training model...")
    model.train(clean_df)
    print("Model trained.")
    print(clean_df['outcome'].value_counts())

    
    # 5. Test prediction
    test_pitch = {
        'pitch_type': 'FF',
        'release_speed': 95.0,
        'release_spin_rate': 2100,
        'plate_x': 0.0,   # center of plate (feet)
        'plate_z': 2.25,  # 27 inches = 2.25 feet
        'balls': 2,
        'strikes': 0,
        'batter_hand': 'R',
        'pitcher_hand': 'R',
        'outs_when_up': 0,
        'inning': 7,
        # Optional context features (will default to 0 if missing)
    }
    zone = model.zones(test_pitch['plate_x'], test_pitch['plate_z'])
    print(f"Zone for this pitch: {zone}")

    test_pitch['zone'] = zone

    result = model.predict_reaction(test_pitch)
    print(result)
    print("\nSample prediction for test pitch:")
    result = model.predict_reaction(test_pitch)
    pprint.pprint(result) 