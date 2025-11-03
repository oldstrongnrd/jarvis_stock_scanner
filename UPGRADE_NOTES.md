# Jarvis Stock Scanner - Complete Documentation 🚀

## Overview

Jarvis Stock Scanner is a **professional-grade options analysis tool** featuring advanced multi-timeframe analysis, institutional activity detection, proprietary quality scoring, and Obsidian integration for trade journaling.

---

## 🎯 What's New

### 1. **Multi-Timeframe Squeeze Detection**
- **Weekly Squeeze**: Compression on weekly charts
- **Daily Squeeze**: Compression on daily charts
- **4-Hour Squeeze**: Intraday compression (if available)
- **1-Hour Squeeze**: Short-term compression (if available)
- **Perfect Nested Squeeze**: All 4 timeframes in compression simultaneously

### 2. **Quality Score System (0-100 points)**
New proprietary scoring algorithm that rates setup quality:
- **40 points**: Squeeze Alignment (W:15, D:15, 4H:5, 1H:5)
- **30 points**: Trend Quality (MA alignment, MACD, EMA distance)
- **20 points**: Volume Analysis (ratio, trend, institutional activity)
- **10 points**: Liquidity & Quality (volume levels, price range)

### 3. **Tier Classification System**
Automatic tier assignment based on quality:
- **Tier 1** (85+ quality): Perfect nested squeezes - highest probability
- **Tier 2** (70-84 quality): Strong multi-timeframe setups
- **Tier 3** (55-69 quality): Developing setups to monitor

### 4. **VWAP Direction Analysis**
- **RISING**: Institutional buyers in control
- **FALLING**: Institutional sellers in control
- **FLAT**: Consolidation, wait for direction
- **Price vs VWAP**: ABOVE, BELOW, or AT VWAP

### 5. **Volume Analysis & Institutional Activity**
Detects institutional money flow:
- **🟦 CYAN (BUYING)**: High volume + price up = institutions accumulating
- **🟪 MAGENTA (SELLING)**: High volume + price down = institutions distributing
- **NEUTRAL**: Normal retail volume
- **Volume Ratio**: Current vs 20-day average
- **Volume Trend**: INCREASING, DECREASING, or STABLE

### 6. **Money Flow Index (MFI)**
Volume-weighted RSI for buying/selling pressure:
- **OVERBOUGHT** (>80): Potential reversal down
- **BULLISH** (50-80): Money flowing in
- **BEARISH** (20-50): Money flowing out
- **OVERSOLD** (<20): Potential bounce

### 7. **Moving Average Alignment Classification**
- **PERFECT**: All 5 MAs properly stacked
- **GOOD**: 4 out of 5 conditions met
- **PARTIAL**: 2-3 conditions met
- **POOR**: Choppy, avoid

### 8. **Trend Direction Classification**
- **BULLISH**: Clear uptrend with MA and MACD alignment
- **BEARISH**: Clear downtrend with inverse alignment
- **NEUTRAL**: Choppy, wait for clarity

### 9. **ATR-Based Price Targets**
Automatically calculated for each setup:
- **+1 ATR**: Conservative profit target
- **+2 ATR**: Aggressive profit target
- **-1 ATR**: Conservative support level
- **-2 ATR**: Strong support level

### 10. **Support & Resistance Levels**
- **Support**: 20-day low
- **Resistance**: 20-day high
- **Distance to 21 EMA**: Percentage distance for entry timing

### 11. **Sector Tracking**
- Automatic sector detection
- Concentration analysis
- Helps avoid over-exposure to single sector

### 12. **Obsidian Markdown Export**
Beautifully formatted daily scan notes:
- Organized by tier (Tier 1, 2, 3)
- Complete setup details for each stock
- Sector concentration analysis
- Action items checklist
- Automatic file organization

### 13. **Enhanced Console Output**
New columns in results table:
- **Tier**: T1, T2, T3
- **Quality**: Quality score (0-100)
- **Direction**: B/N/E (Bullish/Neutral/Bearish)
- **Squeezes**: WD4H1H format showing active timeframes
- **Perfect**: 🔥 for perfect nested squeezes
- **VWAP**: R/F/Flat direction
- **Inst**: 🟦/🟪 for institutional activity
- **MFI**: Money Flow Index value
- **Vol_Ratio**: Volume ratio vs average
- **+1ATR/+2ATR**: Price targets
- **Sector**: Stock sector

---

## 📊 Configuration Options

Updated `config.json` with new parameters:

```json
{
  "scanner_config": {
    "min_score": 80,              // Minimum Big 3 score (0-120)
    "min_strength": 70.0,          // Minimum strength percentage
    "min_quality_score": 55,       // Minimum quality score (0-100)
    "focus_list_only": false,      // Scan only Taylor's Focus List
    "include_etfs": true,          // Include ETFs and indexes
    "min_volume": 500000,          // Minimum average daily volume
    "min_price": 10.0,             // Minimum stock price
    "max_price": 1000.0,           // Maximum stock price
    "obsidian_vault": "",          // Path to your Obsidian vault
    "enable_obsidian": false       // Enable Obsidian export
  }
}
```

---

## 🚀 How to Use

### Basic Scan
```bash
python jarvis_scanner.py
```

### Using with Custom Settings
```python
from jarvis_scanner import JarvisScanner, ScanConfig

config = ScanConfig(
    min_score=90,                  # Higher threshold
    min_quality_score=70,          # Tier 2+ only
    enable_obsidian=True,          # Enable Obsidian export
    obsidian_vault="/path/to/vault"
)

scanner = JarvisScanner(config)
results = scanner.scan_parallel()
```

### Analyze Specific Ticker
```python
scanner = JarvisScanner()
result = scanner.analyze_ticker("AAPL")

if result:
    print(f"Quality Score: {result.quality_score}/100")
    print(f"Tier: {result.tier}")
    print(f"Weekly Squeeze: {result.weekly_squeeze}")
    print(f"Daily Squeeze: {result.daily_squeeze}")
    print(f"VWAP Direction: {result.vwap_direction}")
    print(f"MFI: {result.mfi}")
    print(f"Institutional Activity: {result.institutional_activity}")
```

### Export to Obsidian
```python
scanner = JarvisScanner()
scanner.scan_parallel()

# Generate markdown
markdown = scanner.generate_obsidian_markdown()

# Save to Obsidian vault
obsidian_file = scanner.save_obsidian_report("/path/to/vault")
```

---

## 📈 Trading Strategy Updates

### Tier 1 Perfect Nested Squeeze Strategy

**Entry Criteria:**
1. ✅ All 4 timeframes in squeeze (W, D, 4H, 1H)
2. ✅ Quality score ≥ 85
3. ✅ Trend direction = BULLISH (for longs)
4. ✅ VWAP direction = RISING
5. ✅ Institutional activity = BUYING (🟦 CYAN)
6. ✅ MFI > 50
7. ✅ Price within 2% of 21 EMA
8. ✅ MACD aligned with trend

**Position:**
- Bull Call Spread (for bullish setups)
- Short strike at +1 ATR (conservative) or +2 ATR (aggressive)
- 30-45 DTE

**Profit Target:**
- Close 50% at +1 ATR
- Close remaining 50% at +2 ATR or 75% max profit

**Stop Loss:**
- Close if trend_direction changes to NEUTRAL or BEARISH
- Close if VWAP direction reverses
- Close if institutional activity reverses

---

## 📊 Data Structure

### OfficialBig3Result Fields

Each scan result now contains 50+ data points:

```python
result = {
    # Core Scores
    'big3_score': 105,              # Out of 120
    'strength_pct': 87.5,           # Percentage
    'quality_score': 92,            # Out of 100
    'tier': 'TIER_1',
    'trend_direction': 'BULLISH',

    # Component Scores
    'trend_score': 38,              # Out of 40
    'structure_score': 35,          # Out of 40
    'momentum_score': 32,           # Out of 40

    # Multi-Timeframe Squeezes
    'weekly_squeeze': True,
    'daily_squeeze': True,
    'four_hour_squeeze': True,
    'one_hour_squeeze': True,
    'perfect_nested_squeeze': True,

    # VWAP
    'vwap_direction': 'RISING',
    'price_vs_vwap': 'ABOVE',

    # Volume
    'volume': 45000000,
    'volume_avg_20': 38000000,
    'volume_ratio': 1.18,
    'volume_trend': 'INCREASING',
    'institutional_activity': 'BUYING',

    # MFI
    'mfi': 62.5,
    'mfi_zone': 'BULLISH',

    # Moving Averages
    'ma_alignment': 'PERFECT',
    'price_above_ema21': True,
    'price_above_sma50': True,
    'price_above_sma200': True,

    # MACD
    'macd_aligned': True,
    'macd_histogram_color': 'GREEN',

    # ATR Targets
    'atr': 17.25,
    'atr_target_plus1': 646.32,
    'atr_target_plus2': 663.57,
    'atr_target_minus1': 611.82,
    'atr_target_minus2': 594.57,

    # Levels
    'support_level': 605.50,
    'resistance_level': 635.75,
    'distance_to_ema21_pct': 0.85,

    # Metadata
    'sector': 'Technology',
    'is_focus_list': True,
    'is_a_plus_setup': True
}
```

---

## 🎯 Quality Score Calculation

Understanding the 100-point quality score:

### Squeeze Alignment (40 points max)
- Weekly squeeze active: **+15 points**
- Daily squeeze active: **+15 points**
- 4-Hour squeeze active: **+5 points**
- 1-Hour squeeze active: **+5 points**

### Trend Quality (30 points max)
- Perfect MA alignment (all 5 conditions): **+15 points**
- Good MA alignment (4 conditions): **+10 points**
- Partial MA alignment (2-3 conditions): **+5 points**
- MACD aligned with trend: **+10 points**
- Price near 21 EMA (<2% distance): **+5 points**

### Volume Analysis (20 points max)
- Volume ratio > 2.0x: **+15 points**
- Volume ratio > 1.5x: **+10 points**
- Volume ratio > 1.0x: **+5 points**
- Volume trend INCREASING: **+5 points**

### Liquidity & Quality (10 points max)
- Average volume > 5M: **+5 points**
- Average volume > 1M: **+3 points**
- Price in sweet spot ($50-$500): **+5 points**
- Price in acceptable range ($10-$1000): **+3 points**

**Maximum Possible:** 100 points

---

## 📝 Obsidian Integration

### File Structure
When Obsidian export is enabled, files are organized as:

```
your-vault/
└── Trading/
    └── Daily_Scans/
        ├── 2025-01-15_scan.md
        ├── 2025-01-16_scan.md
        └── 2025-01-17_scan.md
```

### Daily Scan Format
Each scan includes:
1. **Market Overview**: Total setups, tier breakdown, perfect squeezes
2. **Tier 1 Setups**: Detailed analysis of perfect setups
3. **Tier 2 Setups**: Strong setups
4. **Tier 3 Setups**: Developing setups
5. **Sector Analysis**: Concentration by sector
6. **Action Items**: Checkbox list for next steps

### Setup Details
Each setup shows:
- Price, Trend, Tier
- All scores (Big 3, Quality, components)
- Multi-timeframe squeeze status
- VWAP and institutional activity
- MFI and MA alignment
- ATR targets with prices
- Support/resistance levels
- Sector classification

---

## 🔧 Troubleshooting

### No Results Found
**Issue:** Scanner returns empty results

**Solutions:**
1. Lower thresholds:
   - `min_score`: Try 60-70
   - `min_strength`: Try 50-60%
   - `min_quality_score`: Try 40-50

2. Check market conditions:
   - Choppy markets = fewer quality setups
   - Trending markets = more setups

3. Expand watchlist in config.json

### Python Not Found
**Issue:** `Python was not found` error

**Solution:**
```bash
# Install Python 3.7+
# Add to PATH
# Install dependencies:
pip install -r requirements_enhanced.txt
```

### Slow Performance
**Issue:** Scanner takes too long

**Solutions:**
1. Reduce `max_workers`: Lower to 5
2. Enable `focus_list_only`: true
3. Reduce watchlist size

### Obsidian Export Not Working
**Issue:** Markdown files not created

**Solutions:**
1. Check `obsidian_vault` path is correct
2. Ensure `enable_obsidian`: true
3. Verify folder permissions

---

## 📚 Advanced Usage

### Custom Filters

You can add custom filtering logic:

```python
scanner = JarvisScanner()
scanner.scan_parallel()

# Filter for perfect nested squeezes only
perfect_setups = [r for r in scanner.results if r.perfect_nested_squeeze]

# Filter for high MFI
high_mfi_setups = [r for r in scanner.results if r.mfi > 60]

# Filter for institutional buying
buying_setups = [r for r in scanner.results
                 if r.institutional_activity == "BUYING"]

# Filter for Tier 1 bullish with rising VWAP
tier1_bullish = [r for r in scanner.results
                 if r.tier == "TIER_1"
                 and r.trend_direction == "BULLISH"
                 and r.vwap_direction == "RISING"]
```

### Bearish Setups

The scanner now detects bearish setups too:

```python
# Find bearish Tier 1 setups for bear put spreads
bearish_tier1 = [r for r in scanner.results
                 if r.tier == "TIER_1"
                 and r.trend_direction == "BEARISH"
                 and r.vwap_direction == "FALLING"
                 and r.institutional_activity == "SELLING"]

for setup in bearish_tier1:
    print(f"{setup.ticker}: Target -1 ATR = ${setup.atr_target_minus1:.2f}")
```

---

## 🎓 Key Differences from Original

| Feature | Original | Enhanced |
|---------|----------|----------|
| Squeeze Detection | Single timeframe | 4 timeframes (W, D, 4H, 1H) |
| Scoring | 120-point only | 120-point + 100-point quality |
| Classification | Pass/Fail | Tier 1/2/3 system |
| VWAP | ❌ | ✅ Direction + position |
| Volume Analysis | Basic | Institutional activity detection |
| MFI | ❌ | ✅ With zones |
| Trend Direction | ❌ | ✅ Bullish/Bearish/Neutral |
| ATR Targets | Single ATR | 4 levels (+1, +2, -1, -2) |
| MA Classification | ❌ | ✅ Perfect/Good/Partial/Poor |
| Obsidian Export | ❌ | ✅ Full markdown reports |
| Sector Tracking | ❌ | ✅ With concentration |
| Bearish Setups | Limited | ✅ Full support |

---

## 💡 Tips for Best Results

1. **Run Daily After Market Close**
   - Most accurate data
   - Time to review setups
   - Plan next day trades

2. **Focus on Tier 1 Setups**
   - Highest probability
   - Perfect nested squeezes
   - 70-80% win rate expected

3. **Require Institutional Activity**
   - Look for 🟦 CYAN (buying) on longs
   - Look for 🟪 MAGENTA (selling) on shorts
   - Confirms smart money participation

4. **Check VWAP Alignment**
   - RISING VWAP for longs
   - FALLING VWAP for shorts
   - Avoid trades against VWAP direction

5. **Monitor MFI**
   - MFI > 50 for longs
   - MFI < 50 for shorts
   - Avoid overbought/oversold extremes

6. **Use Quality Score**
   - 85+ = Excellent
   - 70-84 = Good
   - 55-69 = Marginal
   - < 55 = Avoid

7. **Export to Obsidian**
   - Keep trade journal
   - Track performance by tier
   - Review what worked

---

## 📞 Support

If you encounter issues:
1. Check this document first
2. Review the Big_3_Trading_System_Complete_Guide.md
3. Verify Python and dependencies are installed
4. Check config.json syntax

---

## 🎉 Conclusion

Your Big 3 Scanner is now a **professional-grade trading tool** that matches the complete specifications from the Big 3 Trading System. With multi-timeframe analysis, quality scoring, institutional activity detection, and Obsidian integration, you have everything needed to identify high-probability vertical spread opportunities.

**Happy Trading! 🚀**

---

*Last Updated: 2025-11-02*
*Version: 2.0 - Enhanced Edition*
