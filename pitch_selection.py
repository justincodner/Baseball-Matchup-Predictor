# Pitch Selection Analysis
# This script analyzes pitcher pitch selection patterns and creates a predictive model

import pybaseball
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pybaseball import statcast, playerid_lookup, statcast_pitcher
from scipy import stats
import seaborn as sns

class PitchSelectionAnalyzer:
    def __init__(self):
        self.pitch_data = None
        self.pitcher_data = {}
        
    def get_pitcher_statcast_data(self, pitcher_name, start_date='2023-01-01', end_date='2023-12-31'):
        """Get Statcast data for a specific pitcher"""
        print(f"Getting Statcast data for {pitcher_name}...")
        
        # Look up pitcher ID
        pitcher_lookup = playerid_lookup(pitcher_name.split()[1], pitcher_name.split()[0])
        if pitcher_lookup.empty:
            print(f"Pitcher {pitcher_name} not found")
            return None
            
        pitcher_id = pitcher_lookup.iloc[0]['key_mlbam']
        
        # Get pitcher's Statcast data
        pitcher_data = statcast_pitcher(start_date, end_date, pitcher_id)
        
        if not pitcher_data.empty:
            print(f"Found {len(pitcher_data)} pitches for {pitcher_name}")
            self.pitcher_data[pitcher_name] = pitcher_data
            return pitcher_data
        else:
            print(f"No data found for {pitcher_name}")
            return None
    
    def analyze_pitch_types_by_count(self, pitcher_name):
        """Analyze pitch type selection based on count"""
        if pitcher_name not in self.pitcher_data:
            print(f"No data available for {pitcher_name}")
            return None
            
        data = self.pitcher_data[pitcher_name]
        
        # Create count-based analysis
        count_pitch_types = data.groupby(['balls', 'strikes', 'pitch_type']).size().unstack(fill_value=0)
        
        # Calculate percentages
        count_pitch_percentages = count_pitch_types.div(count_pitch_types.sum(axis=1), axis=0) * 100
        
        print(f"\n=== Pitch Type Selection by Count for {pitcher_name} ===")
        print(count_pitch_percentages.round(1))
        
        return count_pitch_percentages
    
    def analyze_pitch_characteristics(self, pitcher_name):
        """Analyze speed and spin rate distributions for each pitch type"""
        if pitcher_name not in self.pitcher_data:
            return None
            
        data = self.pitcher_data[pitcher_name]
        
        # Filter out missing values
        data = data.dropna(subset=['release_speed', 'release_spin_rate', 'pitch_type'])
        
        pitch_characteristics = {}
        
        for pitch_type in data['pitch_type'].unique():
            pitch_data = data[data['pitch_type'] == pitch_type]
            
            if len(pitch_data) > 10:  # Only analyze if we have enough data
                characteristics = {
                    'speed_mean': pitch_data['release_speed'].mean(),
                    'speed_std': pitch_data['release_speed'].std(),
                    'spin_mean': pitch_data['release_spin_rate'].mean(),
                    'spin_std': pitch_data['release_spin_rate'].std(),
                    'count': len(pitch_data)
                }
                pitch_characteristics[pitch_type] = characteristics
        
        print(f"\n=== Pitch Characteristics for {pitcher_name} ===")
        for pitch_type, stats in pitch_characteristics.items():
            print(f"{pitch_type}: Speed {stats['speed_mean']:.1f}±{stats['speed_std']:.1f} mph, "
                  f"Spin {stats['spin_mean']:.0f}±{stats['spin_std']:.0f} rpm (n={stats['count']})")
        
        return pitch_characteristics
    
    def analyze_pitch_locations(self, pitcher_name):
        """Analyze pitch location patterns"""
        if pitcher_name not in self.pitcher_data:
            return None
            
        data = self.pitcher_data[pitcher_name]
        
        # Filter out missing location data
        data = data.dropna(subset=['plate_x', 'plate_z', 'pitch_type'])
        
        location_patterns = {}
        
        for pitch_type in data['pitch_type'].unique():
            pitch_data = data[data['pitch_type'] == pitch_type]
            
            if len(pitch_data) > 10:
                # Calculate location statistics
                x_mean = pitch_data['plate_x'].mean() * 12
                x_std = pitch_data['plate_x'].std() * 12
                z_mean = pitch_data['plate_z'].mean() * 12
                z_std = pitch_data['plate_z'].std() * 12
                
                location_patterns[pitch_type] = {
                    'x_mean': x_mean,
                    'x_std': x_std,
                    'z_mean': z_mean,
                    'z_std': z_std,
                    'count': len(pitch_data)
                }
        
        print(f"\n=== Pitch Location Patterns for {pitcher_name} ===")
        for pitch_type, stats in location_patterns.items():
            print(f"{pitch_type}: X={stats['x_mean']:.2f}±{stats['x_std']:.2f}, "
                  f"Z={stats['z_mean']:.2f}±{stats['z_std']:.2f} (n={stats['count']})")
        
        return location_patterns
    
    def create_pitch_selection_model(self, pitcher_name):
        """Create a comprehensive pitch selection model"""
        count_analysis = self.analyze_pitch_types_by_count(pitcher_name)
        characteristics = self.analyze_pitch_characteristics(pitcher_name)
        locations = self.analyze_pitch_locations(pitcher_name)
        
        if not all([count_analysis is not None, characteristics, locations]):
            print("Insufficient data to create pitch selection model")
            return None
        
        model = {
            'pitcher_name': pitcher_name,
            'count_based_selection': count_analysis,
            'pitch_characteristics': characteristics,
            'location_patterns': locations
        }
        
        return model
    
    def predict_pitch(self, model, balls, strikes, handedness='R'):
        """Predict pitch selection based on count and handedness"""
        if model is None:
            return None
            
        # Get pitch type probabilities for this count
        count_key = (balls, strikes)
        if count_key in model['count_based_selection'].index:
            pitch_probs = model['count_based_selection'].loc[count_key]
            pitch_probs = pitch_probs[pitch_probs > 0]  # Remove zero probabilities
            
            if len(pitch_probs) > 0:
                # Select pitch type based on probabilities
                pitch_type = np.random.choice(pitch_probs.index, p=pitch_probs.values/pitch_probs.sum())
                
                # Get characteristics for this pitch type
                if pitch_type in model['pitch_characteristics']:
                    char = model['pitch_characteristics'][pitch_type]
                    loc = model['location_patterns'][pitch_type]
                    
                    # Generate speed and spin using normal distribution
                    speed = np.random.normal(char['speed_mean'], char['speed_std'])
                    spin = np.random.normal(char['spin_mean'], char['spin_std'])
                    
                    # Generate location using normal distribution
                    x_loc = np.random.normal(loc['x_mean'], loc['x_std'])
                    z_loc = np.random.normal(loc['z_mean'], loc['z_std'])
                    
                    prediction = {
                        'pitch_type': pitch_type,
                        'speed': speed,
                        'spin_rate': spin,
                        'location_x': x_loc,
                        'location_z': z_loc,
                        'handedness': handedness,
                        'count': (balls, strikes)
                    }
                    
                    return prediction
        
        return None
    
    def visualize_pitch_selection(self, pitcher_name):
        """Create visualizations of pitch selection patterns"""
        if pitcher_name not in self.pitcher_data:
            return
            
        data = self.pitcher_data[pitcher_name]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Pitch Selection Analysis: {pitcher_name}', fontsize=16)
        
        # 1. Pitch type distribution
        pitch_counts = data['pitch_type'].value_counts()
        axes[0, 0].pie(pitch_counts.values, labels=pitch_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Pitch Type Distribution')
        
        # 2. Speed by pitch type
        data.boxplot(column='release_speed', by='pitch_type', ax=axes[0, 1])
        axes[0, 1].set_title('Pitch Speed by Type')
        axes[0, 1].set_xlabel('Pitch Type')
        axes[0, 1].set_ylabel('Speed (mph)')
        
        # 3. Location scatter plot (convert to inches)
        for pitch_type in data['pitch_type'].unique():
            pitch_data = data[data['pitch_type'] == pitch_type]
            if len(pitch_data) > 0:
                # Convert feet to inches for visualization
                x_inches = pitch_data['plate_x'] * 12
                z_inches = pitch_data['plate_z'] * 12
                axes[1, 0].scatter(x_inches, z_inches, 
                                  label=pitch_type, alpha=0.6)
        axes[1, 0].set_xlabel('Horizontal Location (inches)')
        axes[1, 0].set_ylabel('Vertical Location (inches)')
        axes[1, 0].set_title('Pitch Locations')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        
        # 4. Count-based pitch selection heatmap
        count_pitch_data = data.groupby(['balls', 'strikes', 'pitch_type']).size().unstack(fill_value=0)
        sns.heatmap(count_pitch_data, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1, 1])
        axes[1, 1].set_title('Pitch Selection by Count')
        axes[1, 1].set_xlabel('Pitch Type')
        axes[1, 1].set_ylabel('Count (Balls-Strikes)')
        
        plt.tight_layout()
        plt.savefig(f'{pitcher_name.replace(" ", "_")}_pitch_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved as '{pitcher_name.replace(' ', '_')}_pitch_analysis.png'")

def main():
    """Main function to demonstrate pitch selection analysis"""
    print("Pitch Selection Analysis System")
    print("=" * 50)
    
    analyzer = PitchSelectionAnalyzer()
    
    # Example pitchers to analyze
    pitchers = [
        "Gerrit Cole",
        "Jacob deGrom", 
        "Max Scherzer"
    ]
    
    for pitcher in pitchers:
        print(f"\n{'='*20} {pitcher} {'='*20}")
        
        # Get pitcher data
        pitcher_data = analyzer.get_pitcher_statcast_data(pitcher, '2023-01-01', '2023-12-31')
        
        if pitcher_data is not None:
            # Create pitch selection model
            model = analyzer.create_pitch_selection_model(pitcher)
            
            if model:
                # Generate some example predictions
                print(f"\n=== Sample Predictions for {pitcher} ===")
                test_counts = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]
                
                for balls, strikes in test_counts:
                    prediction = analyzer.predict_pitch(model, balls, strikes)
                    if prediction:
                        print(f"Count {balls}-{strikes}: {prediction['pitch_type']} at {prediction['speed']:.1f} mph, "
                              f"spin {prediction['spin_rate']:.0f} rpm, "
                              f"location ({prediction['location_x']:.1f}, {prediction['location_z']:.1f}) inches")
                
                # Create visualizations
                analyzer.visualize_pitch_selection(pitcher)
                
                # Save model data
                model_df = pd.DataFrame({
                    'pitcher': [pitcher],
                    'pitch_types': [list(model['pitch_characteristics'].keys())],
                    'total_pitches': [len(pitcher_data)]
                })
                model_df.to_csv(f'{pitcher.replace(" ", "_")}_pitch_model.csv', index=False)
                print(f"Model data saved to '{pitcher.replace(' ', '_')}_pitch_model.csv'")

if __name__ == "__main__":
    main() 