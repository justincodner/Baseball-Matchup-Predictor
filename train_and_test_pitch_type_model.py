print("Script started")

from pitch_selection import PitchSelectionAnalyzer
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
import joblib

if __name__ == "__main__":
    analyzer = PitchSelectionAnalyzer()
    # Use multiple pitchers and years
    pitchers = ["Gerrit Cole", "Jacob deGrom", "Max Scherzer", "Shohei Ohtani", "Spencer Strider", "Justin Verlander", "Corbin Burnes", "JP Sears", "Shane Bieber", "Carlos Rodón", "Josh Hader", "Edwin Díaz",
    "Tanner Scott", "Kyle Freeland", "Ryan Pressly", "Kyle Finnegan"]
    all_data = []
    for pitcher in pitchers:
        data = analyzer.get_pitcher_statcast_data(pitcher, '2020-01-01', '2024-12-31')
        if data is not None:
            df = analyzer.build_pitch_sequence_dataframe(pitcher)
            if df is not None:
                df['pitcher_name'] = pitcher  # Add pitcher name as a feature
                all_data.append(df)
    if all_data:
        df = pd.concat(all_data, ignore_index=True)
    else:
        raise ValueError("No data found for any pitcher!")

    print("Returned from build_pitch_sequence_dataframe")
    print("DF shape:", None if df is None else df.shape)

    # Print unique grouped pitch types and their counts
    print("Unique grouped pitch types:", df['pitch_type_grouped'].nunique())
    print("Grouped pitch type counts:\n", df['pitch_type_grouped'].value_counts())

    # Exclude both 'pitch_type' and 'pitch_type_grouped' from features
    feature_cols = [col for col in df.columns if col not in ['pitch_type', 'pitch_type_grouped']]
    print("Features used for training:", feature_cols)
    target_col = 'pitch_type_grouped'

    X = df[feature_cols]
    y = df[target_col]

    # Automatically detect all string/object columns for OneHotEncoder
    categorical_features = [col for col in feature_cols if df[col].dtype == 'object']
    print("Categorical features for OneHotEncoder:", categorical_features)
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ], remainder='passthrough')

    # GBM pipeline only (no GridSearchCV)
    gbm_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('gbm', HistGradientBoostingClassifier(max_iter=200, random_state=42))
    ])
    gbm_pipeline.fit(X, y)

    # Save the trained model
    model_filename = 'multi_pitcher_pitch_type_gbm.pkl'
    joblib.dump(gbm_pipeline, model_filename)
    print(f"Model saved to {model_filename}")

    # Example prediction with the trained model
    context = {
        'balls': 1,
        'strikes': 2,
        'batter_hand': 'L',
        'pitcher_hand': 'R',
        'previous_pitch_type': 'FF',
        'pitch_number_in_at_bat': 3,
        'inning': 5,
        'outs_when_up': 1,
        'on_1b': 0,
        'on_2b': 0,
        'on_3b': 0,
        'pitcher_name': 'Tanner Scott',
        'inning_topbot': 'Top',
        'score_differential': 1,
        'count_str': '1-2',
        'home_team': 'NYY',
        'away_team': 'BOS',
        'batter': 596115,  # Example batter ID
        'previous_pitch_result': 'ball',
        'previous_pitch_speed': 95.0,
        'n_through_order_pitcher': 2,
        'n_prior_pa_for_batter': 1,
        'batter_team': 'NYY',
        'pitcher_team': 'NYY'
    }
    context_df = pd.DataFrame([context])
    predicted_pitch_type = gbm_pipeline.predict(context_df)[0]
    print("Predicted pitch type (from trained model):", predicted_pitch_type)

    # Load the model from disk and make a prediction
    loaded_pipeline = joblib.load(model_filename)
    loaded_predicted_pitch_type = loaded_pipeline.predict(context_df)[0]
    print("Predicted pitch type (from loaded model):", loaded_predicted_pitch_type)

    print("Training accuracy:", gbm_pipeline.score(X, y))

    # Stratified cross-validation and class distribution diagnostics
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(gbm_pipeline, X, y, cv=skf)
    print("Stratified cross-validation accuracy scores:", scores)
    print("Mean stratified cross-validation accuracy:", scores.mean())

    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        y_test_fold = y.iloc[test_idx]
        print(f"Fold {i+1} class distribution:")
        print(y_test_fold.value_counts(normalize=True))
        print()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    gbm_pipeline.fit(X_train, y_train)
    test_accuracy = gbm_pipeline.score(X_test, y_test)
    print("Test set accuracy:", test_accuracy)

    # Print the majority class baseline accuracy
    most_common = y.value_counts().idxmax()
    baseline_acc = (y == most_common).mean()
    print("Majority class baseline accuracy:", baseline_acc)

    