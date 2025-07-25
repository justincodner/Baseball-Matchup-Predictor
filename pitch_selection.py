# Pitch Selection Analysis
# This script analyzes pitcher pitch selection patterns and creates a predictive model

import pybaseball
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pybaseball import statcast, playerid_lookup, statcast_pitcher
from scipy import stats
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class PitchSelectionAnalyzer:
    def __init__(self):
        self.pitch_data = None
        self.pitcher_data = {}
        self.spin_correlation_models = {}  # Store correlation models for each pitcher
        
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
    
    def build_pitch_sequence_dataframe(self, pitcher_name):
        """Build a pitch sequence dataframe for ML modeling."""
        if pitcher_name not in self.pitcher_data:
            return None

        data = self.pitcher_data[pitcher_name].copy()
        # Sort by game and at-bat and pitch number to ensure correct sequence
        data = data.sort_values(['game_pk', 'at_bat_number', 'pitch_number'])

        # Add previous pitch type (shifted within each at-bat)
        data['previous_pitch_type'] = data.groupby(['game_pk', 'at_bat_number'])['pitch_type'].shift(1)
        data['previous_pitch_type'] = data['previous_pitch_type'].fillna('None')
        data['pitch_number_in_at_bat'] = data.groupby(['game_pk', 'at_bat_number']).cumcount() + 1

        # Add inning_topbot (categorical)
        if 'inning_topbot' in data.columns:
            data['inning_topbot'] = data['inning_topbot'].fillna('Unknown')
        else:
            data['inning_topbot'] = 'Unknown'

        # Add score_differential (bat_score - fld_score)
        if 'bat_score' in data.columns and 'fld_score' in data.columns:
            data['score_differential'] = data['bat_score'].fillna(0) - data['fld_score'].fillna(0)
        else:
            data['score_differential'] = 0

        # Add count_str (e.g., '2-1')
        data['count_str'] = data['balls'].astype(str) + '-' + data['strikes'].astype(str)

        # Add home_team and away_team
        if 'home_team' in data.columns:
            data['home_team'] = data['home_team'].fillna('Unknown')
        else:
            data['home_team'] = 'Unknown'
        if 'away_team' in data.columns:
            data['away_team'] = data['away_team'].fillna('Unknown')
        else:
            data['away_team'] = 'Unknown'

        # Add batter (batter ID)
        if 'batter' in data.columns:
            data['batter'] = data['batter'].fillna(0)
        else:
            data['batter'] = 0

        # Add previous_pitch_result and previous_pitch_speed (engineered features)
        if 'description' in data.columns:
            data['previous_pitch_result'] = data.groupby(['game_pk', 'at_bat_number'])['description'].shift(1)
            data['previous_pitch_result'] = data['previous_pitch_result'].fillna('Unknown')
        else:
            data['previous_pitch_result'] = 'Unknown'
        if 'release_speed' in data.columns:
            data['previous_pitch_speed'] = data.groupby(['game_pk', 'at_bat_number'])['release_speed'].shift(1)
            data['previous_pitch_speed'] = data['previous_pitch_speed'].fillna(0)
        else:
            data['previous_pitch_speed'] = 0

        # Define optional features and fill missing values
        optional_nums = ['n_through_order_pitcher', 'n_prior_pa_for_batter', 'previous_pitch_speed', 'on_1b', 'on_2b', 'on_3b', 'score_differential', 'outs_when_up']
        optional_categorical = ['previous_pitch_result', 'batter_team', 'pitcher_team', 'inning_topbot', 'home_team', 'away_team', 'count_str']
        for col in optional_nums:
            if col in data.columns:
                data[col] = data[col].fillna(0)
        for col in optional_categorical:
            if col in data.columns:
                data[col] = data[col].fillna('Unknown')
        if 'inning' in data.columns:
            data['inning'] = data['inning'].fillna(0).astype(int)

        # Map batter/pitcher handedness if needed
        if 'batter_hand' not in data.columns and 'stand' in data.columns:
            data['batter_hand'] = data['stand']
        if 'pitcher_hand' not in data.columns and 'p_throws' in data.columns:
            data['pitcher_hand'] = data['p_throws']

        # Drop rows with missing required features
        df = data.dropna(subset=['balls', 'strikes', 'batter_hand', 'pitcher_hand', 'pitch_type'])

        # Define all possible features (including new ones)
        all_possible_features = [
            'balls', 'strikes', 'batter_hand', 'pitcher_hand', 'previous_pitch_type', 
            'pitch_number_in_at_bat', 'pitch_type', 'inning', 'outs_when_up', 
            'on_1b', 'on_2b', 'on_3b', 'score_differential', 'batter_team', 'pitcher_team',
            'n_through_order_pitcher', 'n_prior_pa_for_batter', 'previous_pitch_result', 'previous_pitch_speed',
            'inning_topbot', 'home_team', 'away_team', 'count_str', 'batter'
        ]
        
        # Only keep features that actually exist in the DataFrame
        available_features = [col for col in all_possible_features if col in df.columns]
        
        # Ensure we have the target column (pitch_type)
        if 'pitch_type' not in available_features:
            print("Warning: pitch_type not found in DataFrame")
            return None
            
        # Only keep the columns you want
        df = df[available_features]

        # Group rare pitch types into 'Other'
        pitch_type_counts = df['pitch_type'].value_counts()
        rare_types = pitch_type_counts[pitch_type_counts < 20].index  # You can adjust the threshold
        df['pitch_type'] = df['pitch_type'].apply(lambda x: 'Other' if x in rare_types else x)

        def group_pitch_type(pt):
            if pt in ['FF', 'FC', 'FT', 'SI', 'SF']:
                return 'Fastball'
            elif pt in ['SL', 'CU', 'KC']:
                return 'Breaking'
            elif pt in ['CH', 'FS']:
                return 'Offspeed'
            else:
                return 'Other'

        df['pitch_type_grouped'] = df['pitch_type'].apply(group_pitch_type)

        return df
    
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
        """Analyze speed and spin rate distributions for each pitch type with correlations"""
        if pitcher_name not in self.pitcher_data:
            return None
            
        data = self.pitcher_data[pitcher_name]
        
        # Filter out missing values
        data = data.dropna(subset=['release_speed', 'release_spin_rate', 'pitch_type', 'pfx_x', 'pfx_z'])
        
        pitch_characteristics = {}
        
        for pitch_type in data['pitch_type'].unique():
            pitch_data = data[data['pitch_type'] == pitch_type]
            
            if len(pitch_data) > 10:  # Only analyze if we have enough data
                # Calculate basic statistics
                speed_mean = pitch_data['release_speed'].mean()
                speed_std = pitch_data['release_speed'].std()
                spin_mean = pitch_data['release_spin_rate'].mean()
                spin_std = pitch_data['release_spin_rate'].std()
                
                # Calculate correlation between speed and spin
                correlation = pitch_data['release_speed'].corr(pitch_data['release_spin_rate'])
                
                # Calculate conditional parameters for spin given speed
                # spin_conditional_mean = spin_mean + correlation * (spin_std/speed_std) * (speed - speed_mean)
                # spin_conditional_std = spin_std * sqrt(1 - correlation^2)
                spin_conditional_std = spin_std * np.sqrt(1 - correlation**2) if not np.isnan(correlation) else spin_std
                
                characteristics = {
                    'speed_mean': speed_mean,
                    'speed_std': speed_std,
                    'spin_mean': spin_mean,
                    'spin_std': spin_std,
                    'speed_spin_correlation': correlation,
                    'spin_conditional_std': spin_conditional_std,
                    'pfx_x_mean': pitch_data['pfx_x'].mean(),
                    'pfx_x_std': pitch_data['pfx_x'].std(),
                    'pfx_z_mean': pitch_data['pfx_z'].mean(),
                    'pfx_z_std': pitch_data['pfx_z'].std(),
                    'count': len(pitch_data)
                }
                pitch_characteristics[pitch_type] = characteristics
        
        print(f"\n=== Pitch Characteristics for {pitcher_name} ===")
        for pitch_type, stats in pitch_characteristics.items():
            correlation_str = f"r={stats['speed_spin_correlation']:.3f}" if not np.isnan(stats['speed_spin_correlation']) else "r=N/A"
            print(f"{pitch_type}: Speed {stats['speed_mean']:.1f}±{stats['speed_std']:.1f} mph, "
                  f"Spin {stats['spin_mean']:.0f}±{stats['spin_std']:.0f} rpm ({correlation_str}) (n={stats['count']}), \nX-Movement: {stats['pfx_x_mean']:.2f}±{stats['pfx_x_std']:.2f} in, "
                  f"Z-Movement: {stats['pfx_z_mean']:.2f}±{stats['pfx_z_std']:.2f} in")
        
        return pitch_characteristics
    
    def create_improved_spin_correlation_models(self, pitcher_name):
        """Create improved statistical models for spin rate prediction using movement data"""
        if pitcher_name not in self.pitcher_data:
            return None
            
        data = self.pitcher_data[pitcher_name]
        
        # Filter out missing values
        data = data.dropna(subset=['release_speed', 'release_spin_rate', 'pitch_type', 'pfx_x', 'pfx_z'])
        
        correlation_models = {}
        
        for pitch_type in data['pitch_type'].unique():
            pitch_data = data[data['pitch_type'] == pitch_type]
            
            if len(pitch_data) > 20:  # Need enough data for analysis
                # Calculate correlations with movement data
                speed_spin_corr = pitch_data['release_speed'].corr(pitch_data['release_spin_rate'])
                movement_x_spin_corr = pitch_data['pfx_x'].corr(pitch_data['release_spin_rate'])
                movement_z_spin_corr = pitch_data['pfx_z'].corr(pitch_data['release_spin_rate'])
                
                # Calculate multiple regression coefficients (simplified)
                # This is a statistical approach using correlations
                model = {
                    'speed_spin_correlation': speed_spin_corr,
                    'movement_x_spin_correlation': movement_x_spin_corr,
                    'movement_z_spin_correlation': movement_z_spin_corr,
                    'spin_mean': pitch_data['release_spin_rate'].mean(),
                    'spin_std': pitch_data['release_spin_rate'].std(),
                    'speed_mean': pitch_data['release_speed'].mean(),
                    'speed_std': pitch_data['release_speed'].std(),
                    'pfx_x_mean': pitch_data['pfx_x'].mean(),
                    'pfx_x_std': pitch_data['pfx_x'].std(),
                    'pfx_z_mean': pitch_data['pfx_z'].mean(),
                    'pfx_z_std': pitch_data['pfx_z'].std(),
                    'sample_count': len(pitch_data)
                }
                
                correlation_models[pitch_type] = model
        
        self.spin_correlation_models[pitcher_name] = correlation_models
        
        # Print model performance
        print(f"\n=== Improved Spin Correlation Models for {pitcher_name} ===")
        for pitch_type, model in correlation_models.items():
            print(f"{pitch_type}: Speed-Spin r={model['speed_spin_correlation']:.3f}, "
                  f"X-Movement-Spin r={model['movement_x_spin_correlation']:.3f}, "
                  f"Z-Movement-Spin r={model['movement_z_spin_correlation']:.3f} "
                  f"(n={model['sample_count']})")
        
        return correlation_models
    
    def predict_spin_improved_statistical(self, pitcher_name, pitch_type, speed, pfx_x, pfx_z):
        """Predict spin rate using improved statistical model with movement data"""
        if (pitcher_name in self.spin_correlation_models and 
            pitch_type in self.spin_correlation_models[pitcher_name]):
            
            model = self.spin_correlation_models[pitcher_name][pitch_type]
            
            # Calculate base spin from speed correlation
            speed_contribution = (model['speed_spin_correlation'] * 
                                (model['spin_std'] / model['speed_std']) * 
                                (speed - model['speed_mean']))
            
            # Add movement contributions
            movement_x_contribution = (model['movement_x_spin_correlation'] * 
                                     (model['spin_std'] / model['pfx_x_std']) * 
                                     (pfx_x - model['pfx_x_mean']))
            
            movement_z_contribution = (model['movement_z_spin_correlation'] * 
                                     (model['spin_std'] / model['pfx_z_std']) * 
                                     (pfx_z - model['pfx_z_mean']))
            
            # Calculate predicted spin
            predicted_spin = (model['spin_mean'] + 
                            speed_contribution + 
                            movement_x_contribution + 
                            movement_z_contribution)
            
            # Add realistic noise
            noise = np.random.normal(0, model['spin_std'] * 0.1)
            predicted_spin += noise
            
            return predicted_spin
        
        # Fallback to original conditional probability method
        return None
    
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
    
    def validate_speed_spin_correlation(self, pitcher_name, model, num_samples=1000):
        """Validate the conditional probability approach by comparing generated vs actual distributions"""
        if pitcher_name not in self.pitcher_data or model is None:
            return None
            
        data = self.pitcher_data[pitcher_name]
        validation_results = {}
        
        for pitch_type in model['pitch_characteristics'].keys():
            pitch_data = data[data['pitch_type'] == pitch_type]
            if len(pitch_data) < 10:
                continue
                
            # Generate samples using the conditional probability model
            generated_speeds = []
            generated_spins = []
            
            char = model['pitch_characteristics'][pitch_type]
            
            for _ in range(num_samples):
                # Generate speed first
                speed = np.random.normal(char['speed_mean'], char['speed_std'])
                
                # Generate spin using conditional probability
                if not np.isnan(char['speed_spin_correlation']) and char['speed_spin_correlation'] != 0:
                    spin_conditional_mean = (char['spin_mean'] + 
                                           char['speed_spin_correlation'] * 
                                           (char['spin_std'] / char['speed_std']) * 
                                           (speed - char['speed_mean']))
                    spin = np.random.normal(spin_conditional_mean, char['spin_conditional_std'])
                else:
                    spin = np.random.normal(char['spin_mean'], char['spin_std'])
                
                generated_speeds.append(speed)
                generated_spins.append(spin)
            
            # Calculate correlations
            actual_correlation = pitch_data['release_speed'].corr(pitch_data['release_spin_rate'])
            generated_correlation = np.corrcoef(generated_speeds, generated_spins)[0, 1]
            
            validation_results[pitch_type] = {
                'actual_correlation': actual_correlation,
                'generated_correlation': generated_correlation,
                'correlation_error': abs(actual_correlation - generated_correlation),
                'actual_speed_mean': pitch_data['release_speed'].mean(),
                'generated_speed_mean': np.mean(generated_speeds),
                'actual_spin_mean': pitch_data['release_spin_rate'].mean(),
                'generated_spin_mean': np.mean(generated_spins)
            }
        
        print(f"\n=== Speed-Spin Correlation Validation for {pitcher_name} ===")
        for pitch_type, results in validation_results.items():
            print(f"{pitch_type}: Actual r={results['actual_correlation']:.3f}, "
                  f"Generated r={results['generated_correlation']:.3f}, "
                  f"Error={results['correlation_error']:.3f}")
        
        return validation_results
    
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
                    
                    # Generate speed first using normal distribution
                    speed = np.random.normal(char['speed_mean'], char['speed_std'])
                    
                    # Generate movement using normal distribution
                    pfx_x = np.random.normal(char['pfx_x_mean'], char['pfx_x_std'])
                    pfx_z = np.random.normal(char['pfx_z_mean'], char['pfx_z_std'])
                    
                    # Try to use improved statistical model for spin prediction
                    spin = self.predict_spin_improved_statistical(model['pitcher_name'], pitch_type, speed, pfx_x, pfx_z)
                    
                    # Fallback to conditional probability if statistical model not available
                    if spin is None:
                        if not np.isnan(char['speed_spin_correlation']) and char['speed_spin_correlation'] != 0:
                            # Calculate conditional mean for spin given speed
                            spin_conditional_mean = (char['spin_mean'] + 
                                                   char['speed_spin_correlation'] * 
                                                   (char['spin_std'] / char['speed_std']) * 
                                                   (speed - char['speed_mean']))
                            
                            # Use conditional standard deviation
                            spin = np.random.normal(spin_conditional_mean, char['spin_conditional_std'])
                        else:
                            # Fallback to independent selection if no correlation or insufficient data
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
        
        
        # 4. Speed vs Spin Rate scatter plot by pitch type
        for pitch_type in data['pitch_type'].unique():
            pitch_data = data[data['pitch_type'] == pitch_type]
            if len(pitch_data) > 0:
                axes[1, 1].scatter(pitch_data['release_speed'], pitch_data['release_spin_rate'], 
                                  label=pitch_type, alpha=0.6, s=20)
        
        axes[1, 1].set_xlabel('Speed (mph)')
        axes[1, 1].set_ylabel('Spin Rate (rpm)')
        axes[1, 1].set_title('Speed vs Spin Rate by Pitch Type')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
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
    pitchers = ["Gerrit Cole", "Jacob deGrom", "Max Scherzer", "Shohei Ohtani", "Spencer Strider"]
    all_data = []
    for pitcher in pitchers:
        data = analyzer.get_pitcher_statcast_data(pitcher, '2021-01-01', '2024-12-31')
        if data is not None:
            df = analyzer.build_pitch_sequence_dataframe(pitcher)
            if df is not None:
                all_data.append(df)
    if all_data:
        df = pd.concat(all_data, ignore_index=True)

    if df is not None:
        # Create pitch selection model
        model = analyzer.create_pitch_selection_model(pitchers[0]) # Use the first pitcher for model creation
        
        if model:
            # Create improved statistical models for spin prediction
            spin_models = analyzer.create_improved_spin_correlation_models(pitchers[0])
            
            # Validate the speed-spin correlation approach
            validation_results = analyzer.validate_speed_spin_correlation(pitchers[0], model)
            
            # Generate some example predictions
            print(f"\n=== Sample Predictions for {pitchers[0]} ===")
            test_counts = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]
            
            for balls, strikes in test_counts:
                prediction = analyzer.predict_pitch(model, balls, strikes)
                if prediction:
                    print(f"Count {balls}-{strikes}: {prediction['pitch_type']} at {prediction['speed']:.1f} mph, "
                          f"spin {prediction['spin_rate']:.0f} rpm, "
                          f"location ({prediction['location_x']:.1f}, {prediction['location_z']:.1f}) inches")
            
            # Create visualizations
            analyzer.visualize_pitch_selection(pitchers[0])
            
            # Save model data
            model_df = pd.DataFrame({
                'pitcher': [pitchers[0]],
                'pitch_types': [list(model['pitch_characteristics'].keys())],
                'total_pitches': [len(df)]
            })
            model_df.to_csv(f'{pitchers[0].replace(" ", "_")}_pitch_model.csv', index=False)
            print(f"Model data saved to '{pitchers[0].replace(' ', '_')}_pitch_model.csv'")

if __name__ == "__main__":
    main() 