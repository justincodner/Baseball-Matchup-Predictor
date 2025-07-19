# Baseball Matchup Predictor

This project uses the [pybaseball](https://github.com/jldbc/pybaseball) library to access and analyze baseball statistics from various sources including Statcast, Baseball Reference, and FanGraphs.

## Features

- **Player Statistics**: Access batting and pitching statistics for any season
- **Statcast Data**: Get detailed pitch-by-pitch data including pitch types, velocities, and outcomes
- **Player Lookup**: Find player IDs and biographical information
- **Data Analysis**: Analyze pitch type distributions and other baseball metrics
- **Data Export**: Save data to CSV files for further analysis

## Installation

The required dependencies are already installed. If you need to reinstall:

```bash
pip install pybaseball pandas matplotlib
```

## Usage

### Running the Example Script

```bash
python pitch_selection.py
```

This will:
1. Download 2023 batting and pitching statistics
2. Get Statcast data for a sample date range
3. Look up player information (e.g., Aaron Judge)
4. Analyze pitch type distributions
5. Save data to CSV files
6. Generate a pitch type distribution chart

### Key Functions

#### `get_player_stats()`
- Retrieves batting and pitching statistics for the 2023 season
- Returns DataFrames with player performance data

#### `get_statcast_data()`
- Downloads Statcast pitch-by-pitch data
- Includes pitch types, velocities, and outcomes
- Note: Statcast data can be large, so date ranges are limited

#### `find_player_id()`
- Looks up player IDs using name searches
- Useful for getting specific player data

#### `analyze_pitch_types()`
- Analyzes pitch type distribution from Statcast data
- Creates visualizations of pitch type frequency

## Data Sources

This project uses data from:
- **Statcast**: MLB's advanced analytics system for pitch-by-pitch data
- **Baseball Reference**: Comprehensive baseball statistics
- **FanGraphs**: Advanced baseball analytics and projections

## Output Files

The script generates several output files:
- `batting_stats_2023.csv`: 2023 batting statistics
- `pitching_stats_2023.csv`: 2023 pitching statistics  
- `statcast_data_sample.csv`: Sample Statcast pitch data
- `pitch_type_distribution.png`: Chart showing pitch type distribution

## Customization

You can modify the script to:
- Change the season year (currently 2023)
- Adjust the date range for Statcast data
- Look up different players
- Add more statistical analysis
- Create different visualizations

## API Rate Limits

Note that some data sources have rate limits. If you encounter errors, try:
- Waiting a few minutes between requests
- Reducing the amount of data requested
- Using smaller date ranges for Statcast data

## Examples

### Getting Player Statistics
```python
from pybaseball import batting_stats, pitching_stats

# Get 2023 batting stats for qualified players
batting = batting_stats(2023, 2023, qual=100)

# Get 2023 pitching stats for qualified pitchers
pitching = pitching_stats(2023, 2023, qual=50)
```

### Getting Statcast Data
```python
from pybaseball import statcast

# Get pitch-by-pitch data for specific dates
data = statcast('2023-09-01', '2023-09-02')
```

### Player Lookup
```python
from pybaseball import playerid_lookup

# Find a player by name
player_info = playerid_lookup('judge', 'aaron')
```

## Contributing

Feel free to extend this project with:
- Additional statistical analysis
- Machine learning models for predictions
- More sophisticated visualizations
- Integration with other baseball data sources

## Resources

- [pybaseball Documentation](https://github.com/jldbc/pybaseball)
- [Statcast Data Dictionary](https://baseballsavant.mlb.com/csv-docs)
- [Baseball Reference](https://www.baseball-reference.com/)
- [FanGraphs](https://www.fangraphs.com/)
