#!/usr/bin/env python3
"""
Practical Big 3 Scanner - Adjusted for Real Market Conditions
Shows more results while still maintaining quality standards
"""

from jarvis_scanner import OfficialBig3Scanner, OfficialScanConfig
import time

print("\n" + "="*90)
print("🚀 BIG 3 SCANNER - PRACTICAL EDITION")
print("Adjusted thresholds for current market conditions")
print("="*90 + "\n")

# More realistic thresholds for actual market scanning
config = OfficialScanConfig(
    min_score=70,             # Was 80 - slightly lower
    min_strength=60.0,        # Was 70% - slightly lower
    min_quality_score=40,     # Was 55 - lower to see more setups
    focus_list_only=False,
    include_etfs=True,
    max_workers=10,
    min_volume=500000,
    min_price=10.0,
    max_price=1000.0,
    enable_obsidian=False
)

# Initialize scanner
scanner = OfficialBig3Scanner(config)

# Run scan
print("📊 Starting practical scan...")
print(f"   - Min Big3 Score: {config.min_score}/120")
print(f"   - Min Strength: {config.min_strength}%")
print(f"   - Min Quality Score: {config.min_quality_score}/100")
print(f"   - Scanning {len(scanner.get_watchlist())} tickers\n")

start_time = time.time()
results_df = scanner.scan_parallel()
scan_time = time.time() - start_time

if not results_df.empty:
    print(f"\n⏱️  Scan completed in {scan_time:.1f} seconds")

    # Count results by tier
    tier_1_count = len([r for r in scanner.results if r.tier == "TIER_1"])
    tier_2_count = len([r for r in scanner.results if r.tier == "TIER_2"])
    tier_3_count = len([r for r in scanner.results if r.tier == "TIER_3"])
    below_count = len([r for r in scanner.results if r.tier == "BELOW_THRESHOLD"])
    perfect_squeeze_count = len([r for r in scanner.results if r.perfect_nested_squeeze])

    print(f"\n{'='*90}")
    print("📈 PRACTICAL SCAN RESULTS")
    print("="*90)
    print(f"\n📊 Summary:")
    print(f"   • Total Setups: {len(scanner.results)}")
    print(f"   • Tier 1 (Perfect): {tier_1_count}")
    print(f"   • Tier 2 (Strong): {tier_2_count}")
    print(f"   • Tier 3 (Developing): {tier_3_count}")
    print(f"   • Below Threshold: {below_count}")
    print(f"   • Perfect Nested Squeezes: {perfect_squeeze_count}")
    print(f"\n{'='*90}\n")

    # Display results table
    print(results_df.to_string(index=False))

    # Show top 5 by quality score
    top_5 = sorted(scanner.results, key=lambda x: x.quality_score, reverse=True)[:5]

    if top_5:
        print(f"\n{'='*90}")
        print("⭐ TOP 5 SETUPS BY QUALITY SCORE")
        print("="*90)

        for i, result in enumerate(top_5, 1):
            print(f"\n{i}. {result.ticker} {'(Focus List)' if result.is_focus_list else ''}")
            print(f"   Price: ${result.current_price:.2f} | Quality: {result.quality_score}/100 | Big3: {result.big3_score}/120")
            print(f"   Tier: {result.tier.replace('_', ' ')} | Trend: {result.trend_direction}")
            print(f"   Squeezes: W:{result.weekly_squeeze} D:{result.daily_squeeze} 4H:{result.four_hour_squeeze} 1H:{result.one_hour_squeeze}")
            print(f"   VWAP: {result.vwap_direction} | Institutional: {result.institutional_activity}")
            print(f"   MFI: {result.mfi:.0f} ({result.mfi_zone}) | MA Alignment: {result.ma_alignment}")
            print(f"   Targets: +1ATR=${result.atr_target_plus1:.2f} | +2ATR=${result.atr_target_plus2:.2f}")

    # Export results
    export_file = scanner.export_results()

    print(f"\n{'='*90}")
    print("📚 INTERPRETATION GUIDE")
    print("="*90)
    print("""
🎯 HOW TO USE THESE RESULTS:

Tier 1 (85+ Quality):
   → Best setups, highest probability (70-80% win rate)
   → Perfect nested squeezes (all timeframes compressed)
   → Enter immediately when criteria met

Tier 2 (70-84 Quality):
   → Strong setups, good probability (60-70% win rate)
   → 2-3 timeframes in squeeze
   → Good for experienced traders

Tier 3 (55-69 Quality):
   → Developing setups, monitor closely (50-60% win rate)
   → 1-2 timeframes in squeeze
   → Watch for improvement

Below Threshold (40-54 Quality):
   → Marginal setups, watchlist only
   → Wait for quality improvement
   → Educational - shows what's developing

🎨 INSTITUTIONAL ACTIVITY:
   • Look for stocks where institutional activity matches trend
   • BUYING (Blue) + BULLISH trend = Strong long setup
   • SELLING (Magenta) + BEARISH trend = Strong short setup

📊 VWAP ALIGNMENT:
   • RISING VWAP + Price ABOVE = Bullish bias (longs)
   • FALLING VWAP + Price BELOW = Bearish bias (shorts)

💰 MFI ZONES:
   • MFI > 50 = Money flowing in (bullish)
   • MFI < 50 = Money flowing out (bearish)
   • MFI > 80 = Overbought (caution)
   • MFI < 20 = Oversold (potential bounce)
    """)

    print(f"\n📁 Results saved to: {export_file}")
    print(f"\n💡 Tip: Focus on Tier 1 and 2 setups with aligned institutional activity")
    print(f"   Current market is {'trending' if tier_1_count + tier_2_count > 10 else 'choppy'}\n")

else:
    print(f"\n⏱️  Scan completed in {scan_time:.1f} seconds")
    print("\n❌ No setups found even with lowered thresholds")
    print("\n💡 This suggests:")
    print("   • Markets are very choppy/sideways right now")
    print("   • Most stocks lack clear trend direction")
    print("   • Few multi-timeframe squeezes active")
    print("   • Wait for better market conditions")
    print("\n📊 Try scanning Taylor's Focus List only:")
    print("   python practical_scan.py --focus-list")
    print()
