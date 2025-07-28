#!/usr/bin/env python3
"""
At-Bat Simulator
Combines pitch prediction and batter reaction models to simulate realistic at-bats
"""

import numpy as np
import pandas as pd
from pitch_selection import PitchSelectionAnalyzer
from ensemble_batter_reaction import EnsembleBatterReactionModel
import random
import time

class AtBatSimulator:
    def __init__(self):
        self.pitch_analyzer = PitchSelectionAnalyzer()
        self.batter_model = EnsembleBatterReactionModel()
        self.pitcher_models = {}
        self.batter_models = {}
        
    def load_pitcher_model(self, pitcher_name, year):
        """Load or create pitcher pitch selection model"""
        cache_key = f"{pitcher_name}_{year}"
        
        if cache_key not in self.pitcher_models:
            print(f"Loading pitch data for {pitcher_name} ({year})...")
            
            # Get pitcher data
            pitcher_data = self.pitch_analyzer.get_pitcher_statcast_data(
                pitcher_name, f'{year}-01-01', f'{year}-12-31'
            )
            
            if pitcher_data is not None:
                # Create pitch selection model
                model = self.pitch_analyzer.create_pitch_selection_model(pitcher_name)
                if model:
                    self.pitcher_models[cache_key] = model
                    print(f"✓ Pitcher model loaded for {pitcher_name}")
                else:
                    print(f"✗ Failed to create pitcher model for {pitcher_name}")
                    return None
            else:
                print(f"✗ No data found for {pitcher_name} in {year}")
                return None
        
        return self.pitcher_models[cache_key]
    
    def load_batter_model(self, batter_name, year):
        """Load or create batter reaction model"""
        cache_key = f"{batter_name}_{year}"
        
        if cache_key not in self.batter_models:
            print(f"Loading batter data for {batter_name} ({year})...")
            
            # Load batter data
            df = self.batter_model.load_batter_data(batter_name, f'{year}-01-01', f'{year}-12-31')
            
            if not df.empty:
                # Preprocess and train
                clean_df = self.batter_model.preprocess_data(df)
                
                if len(clean_df) >= 500:  # Minimum data requirement
                    success = self.batter_model.train_ensemble(clean_df)
                    if success:
                        self.batter_models[cache_key] = self.batter_model
                        print(f"✓ Batter model loaded for {batter_name}")
                    else:
                        print(f"✗ Failed to train batter model for {batter_name}")
                        return None
                else:
                    print(f"✗ Insufficient data for {batter_name} in {year} ({len(clean_df)} pitches)")
                    return None
            else:
                print(f"✗ No data found for {batter_name} in {year}")
                return None
        
        return self.batter_models[cache_key]
    
    def simulate_pitch(self, pitcher_model, batter_model, count, batter_hand='R', pitcher_hand='R'):
        """Simulate a single pitch"""
        
        # Step 1: Predict pitch characteristics using the pitch_selection methods
        pitch_prediction = self.pitch_analyzer.predict_pitch(
            pitcher_model, count[0], count[1], batter_hand
        )
        
        if not pitch_prediction:
            print(f"✗ Failed to predict pitch for count {count}")
            return None
        
        # Step 2: Prepare pitch context for batter reaction
        # The pitch_prediction contains: pitch_type, speed, spin_rate, location_x, location_z, handedness, count
        pitch_context = {
            'pitch_type': pitch_prediction['pitch_type'],
            'release_speed': pitch_prediction['speed'],
            'release_spin_rate': pitch_prediction['spin_rate'],
            'plate_x': pitch_prediction['location_x'],
            'plate_z': pitch_prediction['location_z'],
            'balls': count[0],
            'strikes': count[1],
            'batter_hand': batter_hand,
            'pitcher_hand': pitcher_hand
        }
        
        # Step 3: Predict batter reaction
        reaction_probs = batter_model.predict_reaction(pitch_context)
        
        # Step 4: Simulate outcome based on probabilities
        outcomes = list(reaction_probs.keys())
        probabilities = list(reaction_probs.values())
        
        # Normalize probabilities (ensure they sum to 1)
        total_prob = sum(probabilities)
        if total_prob > 0:
            normalized_probs = [p / total_prob for p in probabilities]
            outcome = np.random.choice(outcomes, p=normalized_probs)
        else:
            # Fallback to most likely outcome
            outcome = outcomes[np.argmax(probabilities)]
        
        return {
            'pitch_type': pitch_prediction['pitch_type'],
            'speed': pitch_prediction['speed'],
            'spin_rate': pitch_prediction['spin_rate'],
            'location_x': pitch_prediction['location_x'],
            'location_z': pitch_prediction['location_z'],
            'outcome': outcome,
            'count_before': count,
            'reaction_probs': reaction_probs,
            'pfx_x': pitch_prediction.get('pfx_x', 0),  # Add movement data if available
            'pfx_z': pitch_prediction.get('pfx_z', 0)
        }
    
    def update_count(self, count, outcome):
        """Update count based on pitch outcome"""
        balls, strikes = count
        
        if outcome == 'ball':
            balls += 1
        elif outcome in ['called_strike', 'swinging_strike']:
            strikes += 1
        elif outcome == 'foul':
            if strikes < 2:  # Foul with less than 2 strikes
                strikes += 1
            # If already 2 strikes, count stays the same
        elif outcome == 'hit_by_pitch':
            balls += 1  # Treated as ball for count purposes
        
        return (balls, strikes)
    
    def is_at_bat_over(self, count, outcome):
        """Check if at-bat is over"""
        balls, strikes = count
        
        # Walk
        if balls >= 4:
            return True
        
        # Strikeout
        if strikes >= 3:
            return True
        
        # Hit
        if outcome in ['single', 'double', 'triple', 'home_run']:
            return True
        
        # Error (batter reaches base)
        if outcome == 'field_error':
            return True
        
        return False
    
    def get_at_bat_result(self, count, outcome):
        """Get the final result of the at-bat"""
        balls, strikes = count
        
        if balls >= 4:
            return "Walk"
        elif strikes >= 3:
            return "Strikeout"
        elif outcome in ['single', 'double', 'triple', 'home_run']:
            return outcome.title()
        elif outcome == 'field_error':
            return "Reached on Error"
        else:
            return "In Progress"
    
    def simulate_at_bat(self, pitcher_name, batter_name, pitcher_year, batter_year, 
                       max_pitches=15, verbose=True):
        """Simulate a complete at-bat"""
        
        print(f"\n{'='*60}")
        print(f"AT-BAT SIMULATION: {pitcher_name} ({pitcher_year}) vs {batter_name} ({batter_year})")
        print(f"{'='*60}")
        
        # Load models
        pitcher_model = self.load_pitcher_model(pitcher_name, pitcher_year)
        batter_model = self.load_batter_model(batter_name, batter_year)
        
        if not pitcher_model or not batter_model:
            print("✗ Failed to load required models")
            return None
        
        # Initialize at-bat
        count = (0, 0)  # (balls, strikes)
        pitches = []
        at_bat_over = False
        result = "In Progress"
        
        # Determine handedness (simplified - could be enhanced with actual data)
        batter_hand = 'L' if batter_name == 'Shohei Ohtani' else 'R'
        pitcher_hand = 'R' if pitcher_name == 'Gerrit Cole' else 'R'
        
        if verbose:
            print(f"Starting at-bat: {count[0]}-{count[1]} count")
        
        # Simulate pitches until at-bat is over
        for pitch_num in range(1, max_pitches + 1):
            if verbose:
                print(f"\nPitch {pitch_num}:")
            
            # Simulate pitch
            pitch_result = self.simulate_pitch(
                pitcher_model, batter_model, count, batter_hand, pitcher_hand
            )
            
            if not pitch_result:
                print("✗ Pitch simulation failed")
                break
            
            pitches.append(pitch_result)
            
            if verbose:
                print(f"  {pitch_result['pitch_type']} at {pitch_result['speed']:.1f} mph")
                print(f"  Spin: {pitch_result['spin_rate']:.0f} rpm")
                print(f"  Location: ({pitch_result['location_x']:.1f}, {pitch_result['location_z']:.1f}) inches")
                print(f"  Movement: ({pitch_result.get('pfx_x', 0):.2f}, {pitch_result.get('pfx_z', 0):.2f}) inches")
                print(f"  Outcome: {pitch_result['outcome']}")
            
            # Update count
            old_count = count
            count = self.update_count(count, pitch_result['outcome'])
            
            if verbose:
                print(f"  Count: {old_count[0]}-{old_count[1]} → {count[0]}-{count[1]}")
            
            # Check if at-bat is over
            if self.is_at_bat_over(count, pitch_result['outcome']):
                at_bat_over = True
                result = self.get_at_bat_result(count, pitch_result['outcome'])
                break
        
        # Final results
        if verbose:
            print(f"\n{'='*40}")
            print(f"AT-BAT RESULT: {result}")
            print(f"Total Pitches: {len(pitches)}")
            print(f"Final Count: {count[0]}-{count[1]}")
            print(f"{'='*40}")
        
        return {
            'result': result,
            'total_pitches': len(pitches),
            'final_count': count,
            'pitches': pitches,
            'pitcher': pitcher_name,
            'batter': batter_name,
            'pitcher_year': pitcher_year,
            'batter_year': batter_year
        }
    
    def simulate_multiple_at_bats(self, pitcher_name, batter_name, pitcher_year, batter_year, 
                                num_at_bats=10, verbose=False):
        """Simulate multiple at-bats and show statistics"""
        
        print(f"\n{'='*60}")
        print(f"MULTIPLE AT-BAT SIMULATION")
        print(f"{pitcher_name} ({pitcher_year}) vs {batter_name} ({batter_year})")
        print(f"Simulating {num_at_bats} at-bats...")
        print(f"{'='*60}")
        
        results = []
        
        for i in range(num_at_bats):
            if verbose:
                print(f"\nAt-Bat {i+1}:")
            
            result = self.simulate_at_bat(
                pitcher_name, batter_name, pitcher_year, batter_year, verbose=verbose
            )
            
            if result:
                results.append(result)
        
        # Analyze results
        if results:
            self.analyze_at_bat_results(results)
        
        return results
    
    def analyze_at_bat_results(self, results):
        """Analyze and display statistics from multiple at-bats"""
        
        print(f"\n{'='*40}")
        print("AT-BAT STATISTICS")
        print(f"{'='*40}")
        
        # Basic stats
        total_at_bats = len(results)
        results_by_type = {}
        
        for result in results:
            result_type = result['result']
            if result_type not in results_by_type:
                results_by_type[result_type] = 0
            results_by_type[result_type] += 1
        
        # Display results
        print(f"Total At-Bats: {total_at_bats}")
        print("\nResults Breakdown:")
        for result_type, count in results_by_type.items():
            percentage = (count / total_at_bats) * 100
            print(f"  {result_type}: {count} ({percentage:.1f}%)")
        
        # Average pitches per at-bat
        avg_pitches = np.mean([r['total_pitches'] for r in results])
        print(f"\nAverage Pitches per At-Bat: {avg_pitches:.1f}")
        
        # Pitch type distribution
        pitch_types = {}
        for result in results:
            for pitch in result['pitches']:
                pitch_type = pitch['pitch_type']
                if pitch_type not in pitch_types:
                    pitch_types[pitch_type] = 0
                pitch_types[pitch_type] += 1
        
        if pitch_types:
            print("\nPitch Type Distribution:")
            total_pitches = sum(pitch_types.values())
            for pitch_type, count in pitch_types.items():
                percentage = (count / total_pitches) * 100
                print(f"  {pitch_type}: {count} ({percentage:.1f}%)")

def main():
    """Test the at-bat simulator with Gerrit Cole vs Shohei Ohtani"""
    
    simulator = AtBatSimulator()
    
    # Test single at-bat
    print("TESTING SINGLE AT-BAT")
    result = simulator.simulate_at_bat(
        pitcher_name="Gerrit Cole",
        batter_name="Shohei Ohtani", 
        pitcher_year=2024,
        batter_year=2023,
        verbose=True
    )
    
    # Test multiple at-bats
    print("\n\nTESTING MULTIPLE AT-BATS")
    results = simulator.simulate_multiple_at_bats(
        pitcher_name="Gerrit Cole",
        batter_name="Shohei Ohtani",
        pitcher_year=2024,
        batter_year=2023,
        num_at_bats=5,
        verbose=False
    )

if __name__ == "__main__":
    main() 