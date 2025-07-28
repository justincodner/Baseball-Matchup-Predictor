#!/usr/bin/env python3
"""
Train Ensemble Batter Reaction Model
Phase 1: Train on multiple batters for better generalization
Phase 2: Train specifically on Shohei Ohtani 2023-2025
"""

from ensemble_batter_reaction import EnsembleBatterReactionModel
import pprint
import time
import pandas as pd

def train_multi_batter_model():
    """Phase 1: Train on multiple batters for better generalization"""
    print("=" * 60)
    print("PHASE 1: MULTI-BATTER ENSEMBLE TRAINING")
    print("=" * 60)
    
    # Initialize model
    model = EnsembleBatterReactionModel()
    
    # List of elite batters for training
    batters = [
        'Aaron Judge',
        'Shohei Ohtani', 
        'Mookie Betts',
        'Ronald Acuña Jr.',
        'Yordan Alvarez',
        'Kyle Tucker',
        'Freddie Freeman',
        'Manny Machado'
    ]
    
    print(f"Loading data for {len(batters)} elite batters...")
    
    # Load data for multiple batters (2023 season)
    start_time = time.time()
    combined_df = model.load_multiple_batters(batters, '2023-01-01', '2023-12-31')
    load_time = time.time() - start_time
    
    if combined_df.empty:
        print("✗ Failed to load data for any batters")
        return None
    
    print(f"Data loading completed in {load_time:.1f} seconds")
    
    # Preprocess data
    print("\nPreprocessing data...")
    start_time = time.time()
    clean_df = model.preprocess_data(combined_df)
    preprocess_time = time.time() - start_time
    
    print(f"Preprocessing completed in {preprocess_time:.1f} seconds")
    print(f"Final dataset: {len(clean_df)} pitches")
    
    # Show outcome distribution
    outcome_counts = clean_df['outcome'].value_counts()
    print("\nOutcome distribution:")
    for outcome, count in outcome_counts.items():
        percentage = (count / len(clean_df)) * 100
        print(f"  {outcome}: {count} ({percentage:.1f}%)")
    
    # Train ensemble
    print("\nTraining ensemble model...")
    start_time = time.time()
    success = model.train_ensemble(clean_df)
    train_time = time.time() - start_time
    
    if success:
        print(f"✓ Ensemble training completed in {train_time:.1f} seconds")
        
        # Save the multi-batter model
        model.save('multi_batter_ensemble_model.pkl')
        print("✓ Model saved as 'multi_batter_ensemble_model.pkl'")
        
        return model
    else:
        print("✗ Ensemble training failed")
        return None

def train_ohtani_specific_model():
    """Phase 2: Train specifically on Shohei Ohtani 2023-2025"""
    print("\n" + "=" * 60)
    print("PHASE 2: SHOHEI OHTANI SPECIFIC MODEL")
    print("=" * 60)
    
    # Initialize new model for Ohtani
    ohtani_model = EnsembleBatterReactionModel()
    
    # Load Ohtani data for multiple seasons
    seasons = ['2023-01-01', '2024-01-01', '2025-01-01']
    end_dates = ['2023-12-31', '2024-12-31', '2025-12-31']
    
    all_ohtani_data = []
    
    for start_date, end_date in zip(seasons, end_dates):
        print(f"\nLoading Ohtani data for {start_date[:4]}...")
        df = ohtani_model.load_batter_data('Shohei Ohtani', start_date, end_date)
        if not df.empty:
            all_ohtani_data.append(df)
    
    if not all_ohtani_data:
        print("✗ No Ohtani data found for any season")
        return None
    
    # Combine all Ohtani data
    ohtani_df = pd.concat(all_ohtani_data, ignore_index=True)
    print(f"\n✓ Combined Ohtani data: {len(ohtani_df)} total pitches")
    
    # Preprocess Ohtani data
    print("Preprocessing Ohtani data...")
    clean_ohtani_df = ohtani_model.preprocess_data(ohtani_df)
    
    print(f"Final Ohtani dataset: {len(clean_ohtani_df)} pitches")
    
    # Show Ohtani outcome distribution
    outcome_counts = clean_ohtani_df['outcome'].value_counts()
    print("\nOhtani outcome distribution:")
    for outcome, count in outcome_counts.items():
        percentage = (count / len(clean_ohtani_df)) * 100
        print(f"  {outcome}: {count} ({percentage:.1f}%)")
    
    # Train Ohtani-specific ensemble
    print("\nTraining Ohtani-specific ensemble...")
    success = ohtani_model.train_ensemble(clean_ohtani_df)
    
    if success:
        print("✓ Ohtani-specific ensemble training completed")
        
        # Save the Ohtani model
        ohtani_model.save('ohtani_ensemble_model.pkl')
        print("✓ Ohtani model saved as 'ohtani_ensemble_model.pkl'")
        
        return ohtani_model
    else:
        print("✗ Ohtani ensemble training failed")
        return None

def compare_models(multi_batter_model, ohtani_model):
    """Compare predictions between multi-batter and Ohtani-specific models"""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON: MULTI-BATTER vs OHTANI-SPECIFIC")
    print("=" * 60)
    
    # Test pitches for comparison
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
    
    for test in test_pitches:
        print(f"\n{test['name']}:")
        print("-" * 40)
        
        # Multi-batter model prediction
        if multi_batter_model:
            multi_pred = multi_batter_model.predict_reaction(test['pitch'])
            print("Multi-batter model:")
            for outcome, prob in multi_pred.items():
                if prob > 0.01:  # Only show probabilities > 1%
                    print(f"  {outcome}: {prob:.3f}")
        
        # Ohtani-specific model prediction
        if ohtani_model:
            ohtani_pred = ohtani_model.predict_reaction(test['pitch'])
            print("Ohtani-specific model:")
            for outcome, prob in ohtani_pred.items():
                if prob > 0.01:  # Only show probabilities > 1%
                    print(f"  {outcome}: {prob:.3f}")

def main():
    """Main execution function"""
    print("ENSEMBLE BATTER REACTION MODEL TRAINING")
    print("Phase 1: Multi-batter training for generalization")
    print("Phase 2: Ohtani-specific training for accuracy")
    
    # Phase 1: Train multi-batter model
    multi_batter_model = train_multi_batter_model()
    
    # Phase 2: Train Ohtani-specific model
    ohtani_model = train_ohtani_specific_model()
    
    # Compare models
    if multi_batter_model and ohtani_model:
        compare_models(multi_batter_model, ohtani_model)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("Models saved:")
    if multi_batter_model:
        print("  - multi_batter_ensemble_model.pkl")
    if ohtani_model:
        print("  - ohtani_ensemble_model.pkl")

if __name__ == "__main__":
    main() 