import pandas as pd
import numpy as np
from pybaseball import statcast_batter, playerid_lookup
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
from datetime import datetime
import joblib

class BatterReactionModel:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.preprocessor = None
        self.data_cache = {}
        self.feature_columns = [
            'pitch_type', 'release_speed', 'release_spin_rate',
            'plate_x', 'plate_z', 'balls', 'strikes',
            'batter_hand', 'pitcher_hand', 'movement_mag'
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
        # Look up batter ID
        lookup = playerid_lookup(batter_name.split()[1], batter_name.split()[0])
        if lookup.empty:
            raise ValueError(f"Batter {batter_name} not found.")
        batter_id = lookup.iloc[0]['key_mlbam']
        df = statcast_batter(start_date, end_date, batter_id)
        self.data_cache[cache_key] = df.copy()
        return df

    def preprocess_data(self, df):
        """Cleans and feature-engineers the raw DataFrame for model training."""
        # Filter to pitches with valid outcomes
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
        df['outcome'] = df['description'].map(outcome_map)
        df = df[df['outcome'].isin(self.outcome_categories)]
        # Map batter_hand and pitcher_hand if not present
        if 'batter_hand' not in df.columns and 'stand' in df.columns:
            df['batter_hand'] = df['stand']
        if 'pitcher_hand' not in df.columns and 'p_throws' in df.columns:
            df['pitcher_hand'] = df['p_throws']
        # Drop rows with missing required features
        required = ['pitch_type', 'release_speed', 'release_spin_rate', 'plate_x', 'plate_z',
                    'balls', 'strikes', 'batter_hand', 'pitcher_hand']
        df = df.dropna(subset=required)
        # Convert location from feet to inches for compatibility, then back to feet
        df['plate_x'] = df['plate_x'].astype(float)
        df['plate_z'] = df['plate_z'].astype(float)
        # Derived features
        df['movement_mag'] = np.sqrt(df.get('pfx_x', 0)**2 + df.get('pfx_z', 0)**2)
        # Only keep necessary columns
        keep_cols = self.feature_columns + ['outcome']
        df = df[keep_cols]
        return df

    def _build_preprocessor(self, df):
        """Builds a sklearn ColumnTransformer for numeric/categorical features."""
        numeric_features = [
            'release_speed', 'release_spin_rate', 'plate_x', 'plate_z',
            'balls', 'strikes', 'movement_mag'
        ]
        categorical_features = ['pitch_type', 'batter_hand', 'pitcher_hand']
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        preprocessor.fit(df[self.feature_columns])
        return preprocessor

    def train(self, df):
        """Trains the GBM model with calibration and stores the label encoder and preprocessor."""
        value_counts = df['outcome'].value_counts()
        valid_classes = value_counts[value_counts >= 2].index.tolist()
        df = df[df['outcome'].isin(valid_classes)]
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df['outcome'])
        self.preprocessor = self._build_preprocessor(df)
        X = self.preprocessor.transform(df[self.feature_columns])
        gbm = HistGradientBoostingClassifier(max_iter=200, random_state=42)
        unique, counts = np.unique(y, return_counts=True)
        min_class_count = counts.min()
        cv = min(3, min_class_count)
        calibrated = CalibratedClassifierCV(gbm, method='isotonic', cv=cv)
        calibrated.fit(X, y)
        self.model = calibrated

    def predict_reaction(self, pitch_dict):
        """Predicts outcome probabilities for a given pitch/context dict."""
        input_df = pd.DataFrame([{**pitch_dict}])
        if 'plate_x' in input_df:
            input_df['plate_x'] = input_df['plate_x'] / 12.0
        if 'plate_z' in input_df:
            input_df['plate_z'] = input_df['plate_z'] / 12.0
        input_df['movement_mag'] = np.sqrt(input_df.get('pfx_x', 0)**2 + input_df.get('pfx_z', 0)**2)
        for col in self.feature_columns:
            if col not in input_df:
                input_df[col] = 0
        X = self.preprocessor.transform(input_df[self.feature_columns])
        proba = self.model.predict_proba(X)[0]
        out_dict = {cat: 0.0 for cat in self.outcome_categories}
        for idx, prob in enumerate(proba):
            cat = self.label_encoder.inverse_transform([idx])[0]
            out_dict[cat] = float(prob)
        return out_dict

    def save(self, path):
        """Save the model, encoder, and preprocessor to disk."""
        joblib.dump({
            'model': self.model,
            'label_encoder': self.label_encoder,
            'preprocessor': self.preprocessor
        }, path)

    def load(self, path):
        """Load the model, encoder, and preprocessor from disk."""
        obj = joblib.load(path)
        self.model = obj['model']
        self.label_encoder = obj['label_encoder']
        self.preprocessor = obj['preprocessor'] 