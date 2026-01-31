# 🏀 NBA MVP Predictor

A production-ready machine learning application that predicts NBA Most Valuable Player (MVP) candidates using historical data and real-time statistics. Built with scikit-learn and deployed as an interactive Streamlit dashboard.

## 🎯 Project Overview

This project transforms exploratory data analysis into a production ML system that:
- Scrapes 40+ years of NBA statistics from Basketball Reference
- Trains a Gradient Boosting model to predict MVP voting share
- Provides an interactive dashboard showing Top 5 MVP candidates


**Live Dashboard:** [Coming Soon - Deploy to Streamlit Cloud]

## ✨ Features

- **Automated Data Pipeline**: Scrapes and caches data from Basketball Reference with rate limiting
- **Feature Engineering**: Creates advanced features (team win%, previous MVP wins, etc.)
- **ML Model**: Gradient Boosting Regressor trained on 1981-2024 data
- **Interactive Dashboard**: Real-time predictions with player stats and visualizations
- **Model Explainability**: Feature importance charts showing what drives MVP predictions
- **Production Architecture**: Modular design with proper separation of concerns

## 🏗️ Architecture

```
nba-mvp-predictor/
├── src/
│   ├── data/
│   │   ├── scraper.py          # Basketball Reference web scraping
│   │   └── preprocessor.py     # Data cleaning & feature engineering
│   ├── models/
│   │   ├── trainer.py          # Model training pipeline
│   │   └── predictor.py        # Prediction serving
│   └── dashboard/
│       ├── app.py              # Streamlit application
│       └── components.py       # Reusable UI components
├── data/                       # Cached scraped data
├── models/                     # Trained model artifacts
├── config/
│   └── config.yaml            # Configuration settings
└── notebooks/
    └── NBA MVP Predictor.ipynb # Original exploratory analysis
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

1. Clone the repository
```bash
git clone <your-repo-url>
cd nba-mvp-predictor
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Train the model
```bash
python -m src.models.trainer --start-year 1981 --end-year 2025
```

4. Launch the dashboard
```bash
streamlit run src/dashboard/app.py
```

The dashboard will be available at `http://localhost:8501`

## 📊 Model Details

### Algorithm
Gradient Boosting Regressor (scikit-learn)

### Features (11 total)
| Feature | Description | Importance |
|---------|-------------|------------|
| **PTS** | Points per game | 41.4% |
| **team_win_pct** | Team winning percentage | 36.1% |
| **AST** | Assists per game | 6.1% |
| **FTA** | Free throw attempts | 6.3% |
| **NRtg** | Team net rating | 5.7% |
| **pre_mvps** | Previous MVP awards | 2.3% |
| **TOV** | Turnovers per game | 2.1% |
| Others | WS/48, OBPM, DBPM, USG% | <1% |

### Performance
- **MSE**: 0.002018
- **Training Data**: 1981-2024 seasons
- **Training Samples**: 7,000+ player-seasons

### Model Parameters
```python
n_estimators = 100
learning_rate = 0.1
max_depth = 3
random_state = 15
```

## 🔧 Configuration

Modify `config/config.yaml` to customize:
- Data year ranges
- Player eligibility criteria (games played, minutes)
- Model hyperparameters
- Dashboard settings

Example:
```yaml
data:
  current_season: 2025
  min_games_pct_current: 0.79
  min_minutes_current: 20.0

model:
  params:
    n_estimators: 100
    learning_rate: 0.1
    max_depth: 3
```

## 🧪 Testing

Manual testing checklist:
- [x] Data scraping retrieves current season stats
- [x] Preprocessor handles traded players correctly
- [x] Model generates predictions without errors
- [x] Dashboard displays Top 5 candidates
- [x] Refresh button updates predictions
- [ ] Deployed app accessible via Streamlit Cloud

## 📊 Data Sources

All data is scraped from [Basketball Reference](https://www.basketball-reference.com/):
- MVP Awards (1981-2024)
- Per-Game Player Statistics
- Advanced Player Statistics
- Team Statistics

**Note**: Scraping includes 3-second delays between requests to respect website rate limits.

## 🚢 Deployment

### Streamlit Cloud (Recommended)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set main file path: `src/dashboard/app.py`
5. Deploy!

The app will be available at: `https://[your-app-name].streamlit.app`

### Local Deployment

```bash
streamlit run src/dashboard/app.py
```

## 🛠️ Tech Stack

- **Machine Learning**: scikit-learn
- **Data Processing**: pandas, numpy
- **Web Scraping**: requests, BeautifulSoup
- **Dashboard**: Streamlit
- **Visualization**: Plotly
- **Configuration**: PyYAML

## 📝 Project Evolution

This project evolved from a Jupyter notebook exploratory analysis into a production-ready ML application:

1. **Exploratory Phase**: Analyzed 40+ years of NBA data in Jupyter notebook
2. **Modularization**: Refactored code into reusable modules (scraper, preprocessor, trainer)
3. **Pipeline Development**: Built automated data and training pipelines
4. **Dashboard Creation**: Developed interactive Streamlit interface
5. **Deployment**: Prepared for cloud deployment with proper configuration

## 🎓 Key Learnings

- **Feature Engineering**: Team success (win%) is nearly as important as individual stats
- **Model Selection**: Gradient Boosting outperformed Linear Regression and Random Forest
- **Data Quality**: Handling traded players and missing values is critical
- **Production vs. Exploration**: Moving from notebooks to modules improves maintainability

## 🔮 Future Enhancements

- [ ] Historical accuracy tracking (predicted vs actual MVP)
- [ ] Confidence intervals for predictions

## 📄 License

This project is for educational and portfolio purposes.

## 👤 Author

[Ward Walton]
- GitHub: [@wardwalton15]

## 🙏 Acknowledgments

- Data provided by [Basketball Reference](https://www.basketball-reference.com/)
- Inspired by the annual NBA MVP race
- Built as a portfolio project demonstrating production ML skills

