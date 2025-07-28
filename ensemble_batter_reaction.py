import pandas as pd
import numpy as np
from pybaseball import statcast_batter, playerid_lookup
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

class EnsembleBatterReactionModel:
    def __init__(self):
        self.models = []
        self.label_encoder = None
        self.preprocessor = None
        self.data_cache = {}
        self.feature_columns = [
            'pitch_type', 'release_speed', 'release_spin_rate',
            'plate_x', 'plate_z', 'balls', 'strikes',
            'batter_hand', 'pitcher_hand', 'movement_mag',
            'late_inning', 'close_game', 'runners_on'
        ]
        self.outcome_categories = [
            'ball', 'called_strike', 'swinging_strike', 'foul', 'single',
            'double', 'triple', 'home_run', 'hit_by_pitch', 'field_error'
        ]

    def load_batter_data(self, batter_name, start_date, end_date):
        """Fetches Statcast data for a batter and caches it."""
        cache_key = (batter_name, start_date, end_date)
        if cache_key in self.data_cache:
            return self.data_cache[cache_key].copy()
        
        try:
            # Look up batter ID
            lookup = playerid_lookup(batter_name.split()[1], batter_name.split()[0])
            if lookup.empty:
                print(f"Warning: Batter {batter_name} not found, skipping...")
                return pd.DataFrame()
            
            batter_id = lookup.iloc[0]['key_mlbam']
            df = statcast_batter(start_date, end_date, batter_id)
            
            if not df.empty:
                df['batter_name'] = batter_name  # Add batter name for tracking
                self.data_cache[cache_key] = df.copy()
                print(f"✓ Loaded {len(df)} pitches for {batter_name}")
            else:
                print(f"Warning: No data found for {batter_name}")
                return pd.DataFrame()
                
            return df
            
        except Exception as e:
            print(f"Error loading data for {batter_name}: {e}")
            return pd.DataFrame()

    def load_multiple_batters(self, batter_list, start_date, end_date):
        """Load data for multiple batters."""
        all_data = []
        
        for batter in batter_list:
            df = self.load_batter_data(batter, start_date, end_date)
            if not df.empty:
                all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"✓ Combined data: {len(combined_df)} total pitches from {len(batter_list)} batters")
            return combined_df
        else:
            print("✗ No data loaded for any batters")
            return pd.DataFrame()

    def preprocess_data(self, df):
        """Enhanced preprocessing with additional features."""
        if df.empty:
            return df
            
        df = df.copy()
        df = df[df['description'].notna()]
        
        # Map Statcast outcomes to 10 categories
        outcome_map = {
            'ball': 'ball',
            'blocked_ball': 'ball',
            'called_strike': 'called_strike',
            'swinging_strike': 'swinging_strike',
            'swinging_strike_blocked': 'swinging_strike',
            'foul': 'foul',
            'foul_tip': 'foul',
            'foul_bunt': 'foul',
            'hit_by_pitch': 'hit_by_pitch',
            'single': 'single',
            'double': 'double',
            'triple': 'triple',
            'home_run': 'home_run',
            'field_error': 'field_error',
            'fielders_choice_error': 'field_error',
        }
        
        # Use events if available, otherwise description
        df['outcome'] = df['events'].fillna(df['description']).map(outcome_map)
        df = df[df['outcome'].isin(self.outcome_categories)]
        
        # Map handedness
        if 'batter_hand' not in df.columns and 'stand' in df.columns:
            df['batter_hand'] = df['stand']
        if 'pitcher_hand' not in df.columns and 'p_throws' in df.columns:
            df['pitcher_hand'] = df['p_throws']
        
        # Drop rows with missing required features
        required = ['pitch_type', 'release_speed', 'release_spin_rate', 'plate_x', 'plate_z',
                    'balls', 'strikes', 'batter_hand', 'pitcher_hand']
        df = df.dropna(subset=required)
        
        # Convert location to float
        df['plate_x'] = df['plate_x'].astype(float)
        df['plate_z'] = df['plate_z'].astype(float)
        
        # Enhanced derived features
        df['movement_mag'] = np.sqrt(df.get('pfx_x', 0)**2 + df.get('pfx_z', 0)**2)
        
        # Context features (with fallbacks)
        df['late_inning'] = (df.get('inning', 1) >= 7).astype(int)
        df['close_game'] = (abs(df.get('bat_score', 0) - df.get('fld_score', 0)) <= 2).astype(int)
        df['runners_on'] = ((df.get('on_1b', 0).notna()) | 
                           (df.get('on_2b', 0).notna()) | 
                           (df.get('on_3b', 0).notna())).astype(int)
        
        # Keep only necessary columns
        keep_cols = self.feature_columns + ['outcome']
        available_cols = [col for col in keep_cols if col in df.columns]
        df = df[available_cols]
        
        return df

    def _build_preprocessor(self, df):
        """Builds a sklearn ColumnTransformer for numeric/categorical features."""
        numeric_features = [
            'release_speed', 'release_spin_rate', 'plate_x', 'plate_z',
            'balls', 'strikes', 'movement_mag', 'late_inning', 'close_game', 'runners_on'
        ]
        categorical_features = ['pitch_type', 'batter_hand', 'pitcher_hand']
        
        # Only use features that exist in the data
        numeric_features = [f for f in numeric_features if f in df.columns]
        categorical_features = [f for f in categorical_features if f in df.columns]
        
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
        preprocessor.fit(df[self.feature_columns])
        return preprocessor

    def train_ensemble(self, df, n_models=3):
        """Trains an ensemble of models."""
        if df.empty:
            print("✗ No data to train on")
            return False
            
        print(f"Training ensemble with {len(df)} pitches...")
        
        # Filter to classes with sufficient examples
        value_counts = df['outcome'].value_counts()
        valid_classes = value_counts[value_counts >= 5].index.tolist()  # Increased threshold
        df = df[df['outcome'].isin(valid_classes)]
        
        if len(df) < 1000:
            print(f"Warning: Only {len(df)} pitches after filtering, may affect model quality")
        
        # Prepare data
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df['outcome'])
        self.preprocessor = self._build_preprocessor(df)
        X = self.preprocessor.transform(df[self.feature_columns])
        
        # Split data for ensemble training
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train multiple models
        self.models = []
        
        # Model 1: HistGradientBoosting
        print("Training HistGradientBoosting model...")
        gbm = HistGradientBoostingClassifier(max_iter=200, random_state=42)
        calibrated_gbm = CalibratedClassifierCV(gbm, method='isotonic', cv=3)
        calibrated_gbm.fit(X_train, y_train)
        self.models.append(('gbm', calibrated_gbm))
        
        # Model 2: Random Forest
        print("Training Random Forest model...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        calibrated_rf = CalibratedClassifierCV(rf, method='isotonic', cv=3)
        calibrated_rf.fit(X_train, y_train)
        self.models.append(('rf', calibrated_rf))
        
        # Model 3: Another GBM with different parameters
        print("Training second GBM model...")
        gbm2 = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=123)
        calibrated_gbm2 = CalibratedClassifierCV(gbm2, method='isotonic', cv=3)
        calibrated_gbm2.fit(X_train, y_train)
        self.models.append(('gbm2', calibrated_gbm2))
        
        print(f"✓ Ensemble training complete: {len(self.models)} models")
        return True

    def predict_reaction(self, pitch_dict):
        """Predicts outcome probabilities using ensemble averaging."""
        if not self.models:
            print("✗ No trained models available")
            return {cat: 0.0 for cat in self.outcome_categories}
        
        # Prepare input
        input_df = pd.DataFrame([{**pitch_dict}])
        
        # Convert location from inches to feet
        if 'plate_x' in input_df:
            input_df['plate_x'] = input_df['plate_x'] / 12.0
        if 'plate_z' in input_df:
            input_df['plate_z'] = input_df['plate_z'] / 12.0
        
        # Derived features
        input_df['movement_mag'] = np.sqrt(input_df.get('pfx_x', 0)**2 + input_df.get('pfx_z', 0)**2)
        input_df['late_inning'] = int(input_df.get('inning', 1) >= 7)
        input_df['close_game'] = int(abs(input_df.get('bat_score', 0) - input_df.get('fld_score', 0)) <= 2)
        input_df['runners_on'] = int((input_df.get('on_1b', 0) != 0) or 
                                    (input_df.get('on_2b', 0) != 0) or 
                                    (input_df.get('on_3b', 0) != 0))
        
        # Fill missing columns
        for col in self.feature_columns:
            if col not in input_df:
                input_df[col] = 0
        
        X = self.preprocessor.transform(input_df[self.feature_columns])
        
        # Get predictions from all models
        all_predictions = []
        for name, model in self.models:
            try:
                proba = model.predict_proba(X)[0]
                all_predictions.append(proba)
            except Exception as e:
                print(f"Warning: Model {name} failed: {e}")
        
        if not all_predictions:
            return {cat: 0.0 for cat in self.outcome_categories}
        
        # Average predictions
        avg_proba = np.mean(all_predictions, axis=0)
        
        # Map back to outcome categories
        out_dict = {cat: 0.0 for cat in self.outcome_categories}
        for idx, prob in enumerate(avg_proba):
            if idx < len(self.label_encoder.classes_):
                cat = self.label_encoder.inverse_transform([idx])[0]
                out_dict[cat] = float(prob)
        
        return out_dict

    def save(self, path):
        """Save the ensemble model."""
        joblib.dump({
            'models': self.models,
            'label_encoder': self.label_encoder,
            'preprocessor': self.preprocessor
        }, path)

    def load(self, path):
        """Load the ensemble model."""
        obj = joblib.load(path)
        self.models = obj['models']
        self.label_encoder = obj['label_encoder']
        self.preprocessor = obj['preprocessor'] 