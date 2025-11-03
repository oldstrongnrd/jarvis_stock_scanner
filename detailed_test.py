#!/usr/bin/env python3
"""Detailed test with error catching"""

from official_big3_scanner import OfficialBig3Scanner, OfficialScanConfig
import logging
import traceback

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

print("="*60)
print("DETAILED TEST WITH ERROR HANDLING")
print("="*60)

config = OfficialScanConfig(
    min_score=1,
    min_quality_score=1,
    min_strength=1.0,
    min_price=1.0,
    max_price=10000.0,
    max_workers=1
)

scanner = OfficialBig3Scanner(config)

print("\nAnalyzing AAPL with full error tracking...")

try:
    result = scanner.analyze_ticker('AAPL')

    if result:
        print("\n*** SUCCESS! ***")
        print(f"Ticker: {result.ticker}")
        print(f"Price: ${result.current_price:.2f}")
        print(f"Quality Score: {result.quality_score}/100")
        print(f"Big3 Score: {result.big3_score}/120")
        print(f"Tier: {result.tier}")
    else:
        print("\nResult is None - likely failed threshold checks")
        print(f"Config thresholds:")
        print(f"  min_score: {config.min_score}")
        print(f"  min_strength: {config.min_strength}")
        print(f"  min_quality_score: {config.min_quality_score}")

except Exception as e:
    print(f"\n*** ERROR ***")
    print(f"Exception: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

print("\n" + "="*60)
print("Test complete!")
print("="*60)
