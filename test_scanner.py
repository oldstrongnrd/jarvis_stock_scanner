#!/usr/bin/env python3
"""Quick test of the upgraded Big 3 Scanner"""

from official_big3_scanner import OfficialBig3Scanner, OfficialScanConfig
import logging

# Reduce logging noise
logging.basicConfig(level=logging.WARNING)

print("="*60)
print("BIG 3 SCANNER - QUICK TEST")
print("="*60)

# Configure scanner with lower thresholds for testing
config = OfficialScanConfig(
    min_score=60,          # Lower for testing
    min_quality_score=40,  # Lower for testing
    max_workers=1
)

# Initialize scanner
scanner = OfficialBig3Scanner(config)

# Test with AAPL
print("\nTesting with AAPL...")
result = scanner.analyze_ticker('AAPL')

if result:
    print("\n*** SUCCESS! Scanner is working! ***\n")
    print(f"Ticker: {result.ticker}")
    print(f"Price: ${result.current_price:.2f}")
    print(f"Tier: {result.tier}")
    print(f"Quality Score: {result.quality_score}/100")
    print(f"Big3 Score: {result.big3_score}/120")
    print(f"Strength: {result.strength_pct:.1f}%")
    print(f"Trend Direction: {result.trend_direction}")
    print(f"\nMulti-Timeframe Squeezes:")
    print(f"  Weekly: {result.weekly_squeeze}")
    print(f"  Daily: {result.daily_squeeze}")
    print(f"  4-Hour: {result.four_hour_squeeze}")
    print(f"  1-Hour: {result.one_hour_squeeze}")
    print(f"  Perfect Nested: {result.perfect_nested_squeeze}")
    print(f"\nMarket Analysis:")
    print(f"  VWAP Direction: {result.vwap_direction}")
    print(f"  Price vs VWAP: {result.price_vs_vwap}")
    print(f"  Institutional Activity: {result.institutional_activity}")
    print(f"  MFI: {result.mfi:.1f} ({result.mfi_zone})")
    print(f"  MA Alignment: {result.ma_alignment}")
    print(f"\nATR Targets:")
    print(f"  +1 ATR: ${result.atr_target_plus1:.2f}")
    print(f"  +2 ATR: ${result.atr_target_plus2:.2f}")
    print(f"  Support: ${result.support_level:.2f}")
    print(f"  Sector: {result.sector}")
else:
    print("\nNo result - ticker may not meet threshold criteria")

print("\n" + "="*60)
print("Test complete!")
print("="*60)
