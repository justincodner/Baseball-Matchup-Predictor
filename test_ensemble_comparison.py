#!/usr/bin/env python3
"""
Test the trained ensemble models and compare predictions
"""

from ensemble_batter_reaction import EnsembleBatterReactionModel
import pprint

def test_ensemble_models():
    """Load and test the trained ensemble models"""
    print("=" * 60)
    print("ENSEMBLE MODEL COMPARISON TEST")
    print("=" * 60)
    
    # Load multi-batter model
    print("Loading multi-batter ensemble model...")
    multi_batter_model = EnsembleBatterReactionModel()
    try:
        multi_batter_model.load('multi_batter_ensemble_model.pkl')
        print("✓ Multi-batter model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load multi-batter model: {e}")
        multi_batter_model = None
    
    # Load Ohtani-specific model
    print("Loading Ohtani-specific ensemble model...")
    ohtani_model = EnsembleBatterReactionModel()
    try:
        ohtani_model.load('ohtani_ensemble_model.pkl')
        print("✓ Ohtani model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load Ohtani model: {e}")
        ohtani_model = None
    
    # Test pitches
    test_pitches = [
        {
            'name': 'Fastball, center, 1-1 count',
            'pitch': {
                'pitch_type': 'FF',
                'release_speed': 96.0,
                'release_spin_rate': 2400,
                'plate_x': 0.0,   # center
                'plate_z': 36.0,  # belt high
                'balls': 1,
                'strikes': 1,
                'batter_hand': 'L',  # Ohtani is left-handed
                'pitcher_hand': 'R'
            }
        },
        {
            'name': 'Slider, outside, 0-2 count',
            'pitch': {
                'pitch_type': 'SL',
                'release_speed': 85.0,
                'release_spin_rate': 2500,
                'plate_x': 8.0,   # outside
                'plate_z': 24.0,  # knee high
                'balls': 0,
                'strikes': 2,
                'batter_hand': 'L',
                'pitcher_hand': 'R'
            }
        },
        {
            'name': 'Fastball, inside, 3-0 count',
            'pitch': {
                'pitch_type': 'FF',
                'release_speed': 94.0,
                'release_spin_rate': 2300,
                'plate_x': -4.0,  # inside
                'plate_z': 42.0,  # chest high
                'balls': 3,
                'strikes': 0,
                'batter_hand': 'L',
                'pitcher_hand': 'R'
            }
        }
    ]
    
    # Compare predictions
    for test in test_pitches:
        print(f"\n{test['name']}:")
        print("-" * 40)
        
        # Multi-batter model prediction
        if multi_batter_model:
            try:
                multi_pred = multi_batter_model.predict_reaction(test['pitch'])
                print("Multi-batter model:")
                for outcome, prob in multi_pred.items():
                    if prob > 0.01:  # Only show probabilities > 1%
                        print(f"  {outcome}: {prob:.3f}")
            except Exception as e:
                print(f"✗ Multi-batter prediction failed: {e}")
        
        # Ohtani-specific model prediction
        if ohtani_model:
            try:
                ohtani_pred = ohtani_model.predict_reaction(test['pitch'])
                print("Ohtani-specific model:")
                for outcome, prob in ohtani_pred.items():
                    if prob > 0.01:  # Only show probabilities > 1%
                        print(f"  {outcome}: {prob:.3f}")
            except Exception as e:
                print(f"✗ Ohtani prediction failed: {e}")
    
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    test_ensemble_models() 