#!/usr/bin/env python3
"""Debug test to see what's happening"""

from jarvis_scanner import OfficialBig3Scanner, OfficialScanConfig
import logging

# Enable more logging
logging.basicConfig(level=logging.INFO)

print("="*60)
print("DEBUG TEST")
print("="*60)

# Very low thresholds to see any result
config = OfficialScanConfig(
    min_score=1,           # Very low
    min_quality_score=1,   # Very low
    min_strength=1.0,      # Very low
    max_workers=1
)

scanner = OfficialBig3Scanner(config)

# Test with multiple tickers
tickers = ['AAPL', 'MSFT', 'SPY', 'QQQ']

for ticker in tickers:
    print(f"\n{'='*60}")
    print(f"Analyzing {ticker}...")
    print('='*60)

    try:
        result = scanner.analyze_ticker(ticker)

        if result:
            print(f"\nSUCCESS for {ticker}!")
            print(f"  Price: ${result.current_price:.2f}")
            print(f"  Quality: {result.quality_score}/100")
            print(f"  Big3: {result.big3_score}/120")
            print(f"  Tier: {result.tier}")
            print(f"  Trend: {result.trend_direction}")
            print(f"  Weekly Squeeze: {result.weekly_squeeze}")
            print(f"  Daily Squeeze: {result.daily_squeeze}")
            break  # Found one that works
        else:
            print(f"No result for {ticker}")
    except Exception as e:
        print(f"ERROR for {ticker}: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("Debug test complete!")
print("="*60)
