#!/usr/bin/env python3
"""
Comprehensive test script for BatterReactionModel
Tests different scenarios, edge cases, and model behavior
"""

from batter_reaction import BatterReactionModel
import pandas as pd
import numpy as np
import pprint

def test_model_initialization():
    """Test model initialization"""
    print("=== Testing Model Initialization ===")
    model = BatterReactionModel()
    assert model.model is None
    assert model.label_encoder is None
    assert model.preprocessor is None
    assert len(model.feature_columns) == 9  # Current model has 9 features
    assert len(model.outcome_categories) == 10
    print("✓ Model initialization successful")

def test_data_loading():
    """Test data loading functionality"""
    print("\n=== Testing Data Loading ===")
    model = BatterReactionModel()
    
    # Test with a well-known batter
    try:
        df = model.load_batter_data('Aaron Judge', '2023-01-01', '2023-12-31')
        print(f"✓ Loaded {len(df)} pitches for Aaron Judge")
        print(f"  Columns: {list(df.columns[:10])}...")  # Show first 10 columns
        
        # Test caching
        df2 = model.load_batter_data('Aaron Judge', '2023-01-01', '2023-12-31')
        assert len(df) == len(df2)
        print("✓ Caching works correctly")
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False
    
    return True

def test_preprocessing():
    """Test data preprocessing"""
    print("\n=== Testing Data Preprocessing ===")
    model = BatterReactionModel()
    
    try:
        # Load and preprocess data
        df = model.load_batter_data('Aaron Judge', '2023-01-01', '2023-12-31')
        clean_df = model.preprocess_data(df)
        
        print(f"✓ Raw data: {len(df)} pitches")
        print(f"✓ Clean data: {len(clean_df)} pitches")
        print(f"✓ Features: {list(clean_df.columns)}")
        
        # Check outcome distribution
        outcome_counts = clean_df['outcome'].value_counts()
        print(f"✓ Outcome distribution:")
        for outcome, count in outcome_counts.items():
            print(f"    {outcome}: {count}")
        
        return clean_df
        
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}")
        return None

def test_training():
    """Test model training"""
    print("\n=== Testing Model Training ===")
    model = BatterReactionModel()
    
    try:
        # Load and preprocess data
        df = model.load_batter_data('Aaron Judge', '2023-01-01', '2023-12-31')
        clean_df = model.preprocess_data(df)
        
        if len(clean_df) < 500:
            print(f"✗ Insufficient data: {len(clean_df)} pitches")
            return None
        
        # Train model
        model.train(clean_df)
        print("✓ Model training successful")
        
        # Check model components
        assert model.model is not None
        assert model.label_encoder is not None
        assert model.preprocessor is not None
        print("✓ All model components initialized")
        
        return model
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return None

def test_predictions(model):
    """Test various prediction scenarios"""
    print("\n=== Testing Predictions ===")
    
    if model is None:
        print("✗ No trained model available")
        return
    
    # Test case 1: Standard fastball
    test_pitch_1 = {
        'pitch_type': 'FF',
        'release_speed': 96.0,
        'release_spin_rate': 2400,
        'plate_x': 0.0,   # center
        'plate_z': 36.0,  # belt high
        'balls': 1,
        'strikes': 1,
        'batter_hand': 'R',
        'pitcher_hand': 'R'
    }
    
    print("Test 1: Standard 96 mph fastball, center, 1-1 count")
    result_1 = model.predict_reaction(test_pitch_1)
    pprint.pprint(result_1)
    
    # Test case 2: Breaking ball
    test_pitch_2 = {
        'pitch_type': 'SL',
        'release_speed': 85.0,
        'release_spin_rate': 2500,
        'plate_x': 8.0,   # outside
        'plate_z': 24.0,  # knee high
        'balls': 0,
        'strikes': 2,
        'batter_hand': 'R',
        'pitcher_hand': 'R'
    }
    
    print("\nTest 2: Slider, outside, 0-2 count")
    result_2 = model.predict_reaction(test_pitch_2)
    pprint.pprint(result_2)
    
    # Test case 3: 3-0 count (hitter's count)
    test_pitch_3 = {
        'pitch_type': 'FF',
        'release_speed': 94.0,
        'release_spin_rate': 2300,
        'plate_x': -4.0,  # inside
        'plate_z': 42.0,  # chest high
        'balls': 3,
        'strikes': 0,
        'batter_hand': 'R',
        'pitcher_hand': 'R'
    }
    
    print("\nTest 3: Fastball, inside, 3-0 count")
    result_3 = model.predict_reaction(test_pitch_3)
    pprint.pprint(result_3)
    
    # Test case 4: Different handedness
    test_pitch_4 = {
        'pitch_type': 'CH',
        'release_speed': 88.0,
        'release_spin_rate': 1800,
        'plate_x': 0.0,
        'plate_z': 30.0,
        'balls': 1,
        'strikes': 1,
        'batter_hand': 'R',
        'pitcher_hand': 'L'  # Left-handed pitcher
    }
    
    print("\nTest 4: Changeup, lefty vs righty")
    result_4 = model.predict_reaction(test_pitch_4)
    pprint.pprint(result_4)

def test_edge_cases(model):
    """Test edge cases and error handling"""
    print("\n=== Testing Edge Cases ===")
    
    if model is None:
        print("✗ No trained model available")
        return
    
    # Test missing features
    test_pitch_minimal = {
        'pitch_type': 'FF',
        'release_speed': 95.0,
        'release_spin_rate': 2400,
        'plate_x': 0.0,
        'plate_z': 36.0,
        'balls': 0,
        'strikes': 0
        # Missing batter_hand and pitcher_hand
    }
    
    print("Test: Minimal features (missing handedness)")
    try:
        result = model.predict_reaction(test_pitch_minimal)
        print("✓ Prediction successful with minimal features")
        pprint.pprint(result)
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test extreme values
    test_pitch_extreme = {
        'pitch_type': 'FF',
        'release_speed': 105.0,  # Very fast
        'release_spin_rate': 3000,  # Very high spin
        'plate_x': 12.0,  # Very outside
        'plate_z': 48.0,  # Very high
        'balls': 3,
        'strikes': 2,
        'batter_hand': 'R',
        'pitcher_hand': 'R'
    }
    
    print("\nTest: Extreme values")
    try:
        result = model.predict_reaction(test_pitch_extreme)
        print("✓ Prediction successful with extreme values")
        pprint.pprint(result)
    except Exception as e:
        print(f"✗ Failed: {e}")

def test_probability_validation(model):
    """Validate that probabilities sum to 1.0"""
    print("\n=== Testing Probability Validation ===")
    
    if model is None:
        print("✗ No trained model available")
        return
    
    test_pitches = [
        {
            'pitch_type': 'FF', 'release_speed': 95.0, 'release_spin_rate': 2400,
            'plate_x': 0.0, 'plate_z': 36.0, 'balls': 1, 'strikes': 1,
            'batter_hand': 'R', 'pitcher_hand': 'R'
        },
        {
            'pitch_type': 'SL', 'release_speed': 85.0, 'release_spin_rate': 2500,
            'plate_x': 8.0, 'plate_z': 24.0, 'balls': 0, 'strikes': 2,
            'batter_hand': 'R', 'pitcher_hand': 'R'
        }
    ]
    
    for i, pitch in enumerate(test_pitches, 1):
        try:
            result = model.predict_reaction(pitch)
            prob_sum = sum(result.values())
            print(f"Test {i}: Probability sum = {prob_sum:.6f}")
            
            if abs(prob_sum - 1.0) < 0.001:
                print(f"✓ Test {i}: Probabilities sum to 1.0")
            else:
                print(f"✗ Test {i}: Probabilities don't sum to 1.0")
                
        except Exception as e:
            print(f"✗ Test {i} failed: {e}")

def main():
    """Run all tests"""
    print("BatterReactionModel Comprehensive Test Suite")
    print("=" * 50)
    
    # Run tests
    test_model_initialization()
    
    if not test_data_loading():
        print("✗ Data loading test failed, stopping")
        return
    
    clean_df = test_preprocessing()
    if clean_df is None:
        print("✗ Preprocessing test failed, stopping")
        return
    
    model = test_training()
    if model is None:
        print("✗ Training test failed, stopping")
        return
    
    test_predictions(model)
    test_edge_cases(model)
    test_probability_validation(model)
    
    print("\n" + "=" * 50)
    print("✓ All tests completed!")

if __name__ == "__main__":
    main() 