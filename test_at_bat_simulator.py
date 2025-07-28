#!/usr/bin/env python3
"""
Comprehensive test script for the At-Bat Simulator
Shows detailed results and multiple scenarios
"""

from at_bat_simulator import AtBatSimulator
import time

def test_detailed_single_at_bat():
    """Test a single at-bat with detailed output"""
    print("=" * 80)
    print("DETAILED SINGLE AT-BAT TEST")
    print("=" * 80)
    
    simulator = AtBatSimulator()
    
    # Simulate a detailed at-bat
    result = simulator.simulate_at_bat(
        pitcher_name="Gerrit Cole",
        batter_name="Shohei Ohtani",
        pitcher_year=2024,
        batter_year=2023,
        verbose=True
    )
    
    if result:
        print(f"\n{'='*50}")
        print("DETAILED PITCH-BY-PITCH ANALYSIS")
        print(f"{'='*50}")
        
        for i, pitch in enumerate(result['pitches'], 1):
            print(f"\nPitch {i}:")
            print(f"  Type: {pitch['pitch_type']}")
            print(f"  Speed: {pitch['speed']:.1f} mph")
            print(f"  Spin: {pitch['spin_rate']:.0f} rpm")
            print(f"  Location: ({pitch['location_x']:.1f}, {pitch['location_z']:.1f}) inches")
            print(f"  Outcome: {pitch['outcome']}")
            print(f"  Count: {pitch['count_before'][0]}-{pitch['count_before'][1]}")
            
            # Show top reaction probabilities
            print("  Top reaction probabilities:")
            sorted_probs = sorted(pitch['reaction_probs'].items(), key=lambda x: x[1], reverse=True)
            for outcome, prob in sorted_probs[:3]:
                if prob > 0.01:
                    print(f"    {outcome}: {prob:.3f}")

def test_multiple_scenarios():
    """Test multiple different scenarios"""
    print("\n" + "=" * 80)
    print("MULTIPLE SCENARIO TEST")
    print("=" * 80)
    
    simulator = AtBatSimulator()
    
    scenarios = [
        {
            'name': 'Gerrit Cole vs Shohei Ohtani (2024 vs 2023)',
            'pitcher': 'Gerrit Cole',
            'batter': 'Shohei Ohtani',
            'pitcher_year': 2024,
            'batter_year': 2023,
            'at_bats': 10
        },
        {
            'name': 'Gerrit Cole vs Aaron Judge (2024 vs 2023)',
            'pitcher': 'Gerrit Cole',
            'batter': 'Aaron Judge',
            'pitcher_year': 2024,
            'batter_year': 2023,
            'at_bats': 10
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"{'='*60}")
        
        results = simulator.simulate_multiple_at_bats(
            pitcher_name=scenario['pitcher'],
            batter_name=scenario['batter'],
            pitcher_year=scenario['pitcher_year'],
            batter_year=scenario['batter_year'],
            num_at_bats=scenario['at_bats'],
            verbose=False
        )
        
        if results:
            # Additional analysis
            print(f"\nDetailed Analysis for {scenario['name']}:")
            
            # Count distribution
            count_distribution = {}
            for result in results:
                final_count = result['final_count']
                count_str = f"{final_count[0]}-{final_count[1]}"
                if count_str not in count_distribution:
                    count_distribution[count_str] = 0
                count_distribution[count_str] += 1
            
            print("  Final Count Distribution:")
            for count, freq in count_distribution.items():
                percentage = (freq / len(results)) * 100
                print(f"    {count}: {freq} ({percentage:.1f}%)")
            
            # Average speed by pitch type
            pitch_speeds = {}
            for result in results:
                for pitch in result['pitches']:
                    pitch_type = pitch['pitch_type']
                    if pitch_type not in pitch_speeds:
                        pitch_speeds[pitch_type] = []
                    pitch_speeds[pitch_type].append(pitch['speed'])
            
            print("  Average Speed by Pitch Type:")
            for pitch_type, speeds in pitch_speeds.items():
                avg_speed = sum(speeds) / len(speeds)
                print(f"    {pitch_type}: {avg_speed:.1f} mph")

def test_pitch_sequence_analysis():
    """Analyze pitch sequences and patterns"""
    print("\n" + "=" * 80)
    print("PITCH SEQUENCE ANALYSIS")
    print("=" * 80)
    
    simulator = AtBatSimulator()
    
    # Simulate multiple at-bats to analyze patterns
    results = simulator.simulate_multiple_at_bats(
        pitcher_name="Gerrit Cole",
        batter_name="Shohei Ohtani",
        pitcher_year=2024,
        batter_year=2023,
        num_at_bats=20,
        verbose=False
    )
    
    if results:
        print("Pitch Sequence Analysis:")
        
        # First pitch analysis
        first_pitches = [result['pitches'][0]['pitch_type'] for result in results if result['pitches']]
        first_pitch_counts = {}
        for pitch_type in first_pitches:
            if pitch_type not in first_pitch_counts:
                first_pitch_counts[pitch_type] = 0
            first_pitch_counts[pitch_type] += 1
        
        print("  First Pitch Distribution:")
        for pitch_type, count in first_pitch_counts.items():
            percentage = (count / len(first_pitches)) * 100
            print(f"    {pitch_type}: {count} ({percentage:.1f}%)")
        
        # Two-pitch sequences
        two_pitch_sequences = []
        for result in results:
            if len(result['pitches']) >= 2:
                seq = (result['pitches'][0]['pitch_type'], result['pitches'][1]['pitch_type'])
                two_pitch_sequences.append(seq)
        
        if two_pitch_sequences:
            print("  Common Two-Pitch Sequences:")
            sequence_counts = {}
            for seq in two_pitch_sequences:
                if seq not in sequence_counts:
                    sequence_counts[seq] = 0
                sequence_counts[seq] += 1
            
            # Show top 5 sequences
            sorted_sequences = sorted(sequence_counts.items(), key=lambda x: x[1], reverse=True)
            for seq, count in sorted_sequences[:5]:
                percentage = (count / len(two_pitch_sequences)) * 100
                print(f"    {seq[0]} → {seq[1]}: {count} ({percentage:.1f}%)")

def main():
    """Run all tests"""
    print("AT-BAT SIMULATOR COMPREHENSIVE TEST SUITE")
    print("Testing Gerrit Cole (2024) vs Shohei Ohtani (2023)")
    
    start_time = time.time()
    
    # Test 1: Detailed single at-bat
    test_detailed_single_at_bat()
    
    # Test 2: Multiple scenarios
    test_multiple_scenarios()
    
    # Test 3: Pitch sequence analysis
    test_pitch_sequence_analysis()
    
    end_time = time.time()
    
    print(f"\n{'='*80}")
    print("TEST SUITE COMPLETE!")
    print(f"Total execution time: {end_time - start_time:.1f} seconds")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 