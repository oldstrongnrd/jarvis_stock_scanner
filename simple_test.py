#!/usr/bin/env python3
"""Very simple test to check data fetching"""

import yfinance as yf
import pandas as pd

print("Testing yfinance data fetch...")

ticker = "AAPL"
stock = yf.Ticker(ticker)

print(f"\nFetching {ticker} data...")
data = stock.history(period='6mo')

if data is not None and len(data) > 0:
    print(f"SUCCESS! Got {len(data)} days of data")
    print(f"\nLatest data:")
    print(f"  Date: {data.index[-1]}")
    print(f"  Close: ${data['Close'].iloc[-1]:.2f}")
    print(f"  Volume: {data['Volume'].iloc[-1]:,.0f}")
    print(f"\nColumns: {list(data.columns)}")
else:
    print("FAILED to fetch data!")

# Test the scanner's _fetch_data method
print("\n" + "="*60)
print("Testing scanner's data fetch method...")
print("="*60)

from official_big3_scanner import OfficialBig3Scanner, OfficialScanConfig

scanner = OfficialBig3Scanner()
scanner_data = scanner._fetch_data(ticker)

if scanner_data is not None:
    print(f"SUCCESS! Scanner fetched {len(scanner_data)} days")
    print(f"  Price: ${scanner_data['Close'].iloc[-1]:.2f}")
else:
    print("Scanner _fetch_data returned None!")
    print("Checking why...")

    # Manual check
    test_data = stock.history(period='6mo')
    if len(test_data) < 60:
        print(f"  Issue: Only {len(test_data)} days (need 60+)")
    else:
        price = test_data['Close'].iloc[-1]
        volume = test_data['Volume'].iloc[-1]
        print(f"  Price: ${price:.2f}")
        print(f"  Volume: {volume:,.0f}")
        if price < 5.0:
            print("  Issue: Price < $5")
        if volume < 100000:
            print("  Issue: Volume < 100,000")
