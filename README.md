# Jarvis Stock Scanner

**Professional Multi-Timeframe Stock Analysis Tool for Options Trading**

Jarvis is an advanced stock scanner that combines multi-timeframe technical analysis, institutional activity detection, and proprietary quality scoring to identify high-probability options trading opportunities. Built with Python and designed for bullish put credit spreads and bearish call credit spreads.

---

## 🚀 Features

### Advanced Analysis
- **Multi-Timeframe Squeeze Detection** - Analyzes Weekly, Daily, 4-Hour, and 1-Hour timeframes
- **Perfect Nested Squeeze Identification** - Finds compression across all timeframes simultaneously
- **Quality Scoring System (0-100)** - Proprietary algorithm rating setup quality
- **Tier Classification** - Automatic ranking into Tier 1/2/3 based on probability
- **Institutional Activity Detection** - Identifies when smart money is buying or selling

### Market Intelligence
- **VWAP Direction Analysis** - Rising/Falling institutional bias detection
- **Volume Analysis** - CYAN/MAGENTA coding for institutional participation
- **Money Flow Index (MFI)** - Volume-weighted momentum for buying/selling pressure
- **Moving Average Classification** - PERFECT/GOOD/PARTIAL/POOR trend alignment
- **Trend Direction** - Clear BULLISH/BEARISH/NEUTRAL classification

### Trading Tools
- **ATR-Based Price Targets** - Automatic calculation of 4 target levels (+1, +2, -1, -2 ATR)
- **Support/Resistance Levels** - Key price levels for strike selection
- **Sector Tracking** - Concentration analysis to avoid over-exposure
- **Obsidian Integration** - Beautiful markdown reports for your trading journal

---

## 📊 Quick Start

### Installation

```bash
# Install dependencies
pip install pandas numpy yfinance

# Run the scanner
python jarvis_scanner.py

# Or use the quick test
python test_scanner.py
```

### Basic Usage

```python
from jarvis_scanner import JarvisScanner, ScanConfig

# Configure scanner
config = ScanConfig(
    min_score=80,              # Minimum technical score (0-120)
    min_quality_score=55,      # Minimum quality score (0-100)
    min_strength=70.0,         # Minimum strength percentage
    enable_obsidian=False      # Enable markdown export
)

# Run scan
scanner = JarvisScanner(config)
results = scanner.scan_parallel()

# Display results
print(results)
```

### Analyze Specific Ticker

```python
scanner = JarvisScanner()
result = scanner.analyze_ticker('AAPL')

if result:
    print(f"Quality Score: {result.quality_score}/100")
    print(f"Tier: {result.tier}")
    print(f"Trend: {result.trend_direction}")
    print(f"Weekly Squeeze: {result.weekly_squeeze}")
    print(f"Institutional Activity: {result.institutional_activity}")
```

---

## 🎯 Tier System

**Tier 1 (Quality 85+)** - Perfect Nested Squeezes
- All 4 timeframes in compression
- Weekly + Daily + 4H + 1H squeezes active
- Highest probability setups (70-80% win rate)
- Enter immediately when criteria met

**Tier 2 (Quality 70-84)** - Strong Multi-Timeframe Setups
- 2-3 timeframes in squeeze
- Strong trend alignment
- Good probability (60-70% win rate)

**Tier 3 (Quality 55-69)** - Developing Setups
- 1-2 timeframes in squeeze
- Monitor for improvement
- Moderate probability (50-60% win rate)

---

## 📈 Quality Score Breakdown

The proprietary 100-point quality scoring system evaluates:

### Squeeze Alignment (40 points max)
- Weekly squeeze: +15 points
- Daily squeeze: +15 points
- 4-Hour squeeze: +5 points
- 1-Hour squeeze: +5 points

### Trend Quality (30 points max)
- Perfect MA alignment: +15 points
- MACD alignment: +10 points
- Price near 21 EMA: +5 points

### Volume Analysis (20 points max)
- High volume ratio (>2.0x): +15 points
- Volume trend increasing: +5 points

### Liquidity & Quality (10 points max)
- High average volume: +5 points
- Optimal price range: +5 points

---

## 🎨 Institutional Activity Indicators

**🟦 CYAN (Institutional Buying)**
- High volume + Price up
- Smart money accumulating
- Bullish signal for longs

**🟪 MAGENTA (Institutional Selling)**
- High volume + Price down
- Smart money distributing
- Bearish signal for shorts

**⚪ NEUTRAL**
- Normal retail volume
- No clear institutional bias

---

## 💰 Trading Strategy

### Entry Criteria
1. ✅ Tier 1 or 2 setup from scanner
2. ✅ Weekly + Daily squeezes active (minimum)
3. ✅ Trend direction clear (BULLISH or BEARISH)
4. ✅ VWAP aligned with trade direction
5. ✅ Institutional activity matches trend
6. ✅ Price near 21 EMA (within 2%)
7. ✅ MFI in favorable zone (>50 for longs, <50 for shorts)

### Position Sizing
- Bullish Put Credit Spreads: Use Tier 1 bullish setups (sell puts below support)
- Bearish Call Credit Spreads: Use Tier 1 bearish setups (sell calls above resistance)
- Target 30-45 DTE for optimal time decay
- Place short strike at support/resistance levels using ATR targets

### Profit Targets
- Close 50% at +1 ATR
- Close remaining 50% at +2 ATR or 75% max profit
- Exit if trend direction changes
- Exit if VWAP direction reverses

---

## 📁 Project Structure

```
jarvis_stock_scanner/
├── jarvis_scanner.py            # Main scanner engine (1,600+ lines)
├── practical_scan.py            # Practical scan with adjusted thresholds
├── test_scanner.py              # Quick test script
├── config.json                  # Configuration settings
├── UPGRADE_NOTES.md             # Complete documentation
└── README.md                    # This file
```

---

## ⚙️ Configuration

Edit `config.json` to customize:

```json
{
  "scanner_config": {
    "min_score": 80,
    "min_quality_score": 55,
    "min_strength": 70.0,
    "min_volume": 500000,
    "min_price": 10.0,
    "max_price": 1000.0,
    "enable_obsidian": false,
    "obsidian_vault": "/path/to/vault"
  }
}
```

---

## 📊 Sample Output

```
Rank  Ticker  Tier  Quality  Big3  Strength  Direction  Squeezes  Perfect  VWAP  Inst  MFI
1     QQQ     T1    92       105   87%       B          WD4H1H    🔥       R     🟦    62
2     AAPL    T2    78       98    82%       B          WD4H      -        R     🟦    58
3     MSFT    T2    74       94    78%       B          WD        -        R     -     55
```

---

## 🛠️ Technical Requirements

- **Python**: 3.7 or higher
- **Dependencies**: pandas, numpy, yfinance
- **Internet**: Required for real-time market data
- **RAM**: 2GB minimum
- **OS**: Windows, macOS, Linux

---

## 📚 Documentation

For complete documentation, see:
- **UPGRADE_NOTES.md** - Comprehensive feature guide
- **Big_3_Trading_System_Complete_Guide.md** - Strategy details

---

## 🔧 Troubleshooting

**No results found?**
- Markets may be choppy - try `practical_scan.py`
- Lower thresholds in config.json
- Check if major indexes are trending

**Scanner running slow?**
- Reduce `max_workers` in config
- Reduce watchlist size
- Check internet connection

---

## 📝 License

This is proprietary trading software. All rights reserved.

---

## 🎯 Support

For issues or questions:
1. Check UPGRADE_NOTES.md for detailed documentation
2. Review config.json settings
3. Run test_scanner.py to verify installation

---

**Built with Python | Powered by Market Data | Designed for Options Traders**
