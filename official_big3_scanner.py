#!/usr/bin/env python3
"""
Official Big 3 Scanner - Matches Taylor Horton's Exact Methodology
Based on the official Big 3 Leaderboard spreadsheet analysis

Key Features:
- 120-point scoring system (matches official)
- Strength percentage calculation
- Taylor's Focus List methodology
- Official watchlist and criteria
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
import logging
import json
import concurrent.futures
from dataclasses import dataclass, asdict
from pathlib import Path
import time

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class OfficialScanConfig:
    """Configuration matching official Big 3 methodology"""
    min_score: int = 80           # Minimum score (out of 120)
    min_strength: float = 70.0    # Minimum strength percentage
    focus_list_only: bool = False # Scan only Taylor's Focus List
    include_etfs: bool = True     # Include ETFs and indexes
    target_dte: int = 30          # Target days to expiration
    spread_width: int = 5         # Default spread width
    min_credit_ratio: float = 0.25 # Minimum credit/risk ratio
    max_workers: int = 10         # Parallel processing threads


@dataclass
class OfficialBig3Result:
    """Result structure matching official methodology"""
    ticker: str
    current_price: float
    big3_score: int              # Out of 120
    strength_pct: float          # Strength percentage
    trend_score: int             # Trend component (0-40)
    structure_score: int         # Structure component (0-40) 
    momentum_score: int          # Momentum component (0-40)
    squeeze_active: bool
    squeeze_bonus: int           # Squeeze bonus points
    is_focus_list: bool          # In Taylor's Focus List
    is_a_plus_setup: bool        # A+ setup (score >= 100)
    support_level: float
    atr: float
    volume: int
    analysis_date: str


class OfficialBig3Scanner:
    """Scanner implementing Taylor Horton's exact Big 3 methodology"""
    
    def __init__(self, config: Optional[OfficialScanConfig] = None):
        self.config = config or OfficialScanConfig()
        self.results = []
        
        # Official watchlists from the spreadsheet
        self.taylor_focus_list = [
            'AMZN', 'AVGO', 'GOOGL', 'QQQ', 'TSM', 'SPX', 'TSLA', 'NFLX', 'AAPL'
        ]
        
        self.big10_basket = [
            'AMZN', 'AVGO', 'GOOGL', 'TSM', 'TSLA', 'NFLX', 'AAPL', 'META', 'NVDA', 'MSFT'
        ]
        
        self.top_bull_candidates = [
            'APP', 'MU', 'AVGO', 'CVNA', 'GLD', 'HD', 'XLC', 'GOOGL', 'QQQ', 'IWM',
            'LRCX', 'ROKU', 'SLV', 'XLY', 'SHOP', 'TSM', 'SPX', 'VLO', 'DIA'
        ]
        
        # Extended watchlist for comprehensive scanning
        self.extended_watchlist = [
            # Major ETFs and Indexes
            'SPY', 'QQQ', 'IWM', 'DIA', 'RSP', 'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLY', 'XLB', 'XLC',
            'SMH', 'IBB', 'IYR', 'ARKK', 'TAN', 'IYT', 'XLU', 'XLP', 'GLD', 'SLV', 'GBTC',
            
            # Taylor's Focus List & Big 10
            'AMZN', 'AVGO', 'GOOGL', 'TSM', 'TSLA', 'NFLX', 'AAPL', 'META', 'NVDA', 'MSFT',
            
            # Top Bull Candidates
            'APP', 'MU', 'CVNA', 'HD', 'LRCX', 'ROKU', 'SHOP', 'VLO',
            
            # Additional High-Quality Names
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'V', 'MA', 'PYPL',
            'WMT', 'COST', 'TGT', 'HD', 'LOW', 'NKE', 'SBUX',
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'LLY', 'TMO',
            'BA', 'CAT', 'GE', 'MMM', 'HON', 'RTX',
            'XOM', 'CVX', 'COP', 'SLB', 'EOG',
            'AMD', 'INTC', 'ORCL', 'CRM', 'ADBE', 'NOW', 'PLTR',
            'DIS', 'CMCSA', 'T', 'VZ', 'TMUS'
        ]
    
    def get_watchlist(self) -> List[str]:
        """Get the appropriate watchlist based on configuration"""
        if self.config.focus_list_only:
            return self.taylor_focus_list
        else:
            return list(set(self.extended_watchlist))  # Remove duplicates
    
    def _fetch_data(self, ticker: str, period: str = '6mo') -> Optional[pd.DataFrame]:
        """Fetch market data with validation"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            if len(data) < 60:  # Need more data for accurate calculations
                logger.debug(f"Insufficient data for {ticker}: {len(data)} days")
                return None
            
            # Basic validation
            current_price = data['Close'].iloc[-1]
            volume = data['Volume'].iloc[-1]
            
            if current_price < 5.0:  # Skip penny stocks
                return None
            
            if volume < 100000:  # Skip low volume stocks
                return None
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def _calculate_trend_score(self, data: pd.DataFrame) -> Tuple[int, Dict]:
        """
        Calculate TREND score (0-40 points) using official methodology
        
        Components:
        - Price vs Moving Averages (10 points)
        - Moving Average Alignment (10 points) 
        - Rate of Change (10 points)
        - Volume Trend (10 points)
        """
        try:
            current_price = data['Close'].iloc[-1]
            
            # Moving averages
            sma_10 = data['Close'].rolling(10).mean()
            sma_20 = data['Close'].rolling(20).mean()
            sma_50 = data['Close'].rolling(50).mean()
            ema_21 = data['Close'].ewm(span=21).mean()
            
            score = 0
            details = {}
            
            # Price vs Moving Averages (10 points)
            above_sma10 = current_price > sma_10.iloc[-1]
            above_sma20 = current_price > sma_20.iloc[-1]
            above_sma50 = current_price > sma_50.iloc[-1]
            above_ema21 = current_price > ema_21.iloc[-1]
            
            ma_score = sum([above_sma10, above_sma20, above_sma50, above_ema21]) * 2.5
            score += ma_score
            
            # Moving Average Alignment (10 points)
            ma_alignment = (sma_10.iloc[-1] > sma_20.iloc[-1] > sma_50.iloc[-1] and
                           ema_21.iloc[-1] > sma_50.iloc[-1])
            if ma_alignment:
                score += 10
            
            # Rate of Change (10 points)
            roc_5 = ((current_price - data['Close'].iloc[-6]) / data['Close'].iloc[-6]) * 100
            roc_20 = ((current_price - data['Close'].iloc[-21]) / data['Close'].iloc[-21]) * 100
            
            roc_score = 0
            if roc_5 > 2: roc_score += 3
            elif roc_5 > 0: roc_score += 1
            
            if roc_20 > 5: roc_score += 4
            elif roc_20 > 0: roc_score += 2
            
            if roc_5 > 0 and roc_20 > 0: roc_score += 3  # Both positive
            
            score += min(roc_score, 10)
            
            # Volume Trend (10 points)
            avg_volume_20 = data['Volume'].tail(20).mean()
            recent_volume = data['Volume'].tail(5).mean()
            volume_ratio = recent_volume / avg_volume_20 if avg_volume_20 > 0 else 1
            
            volume_score = 0
            if volume_ratio > 1.5: volume_score = 10
            elif volume_ratio > 1.2: volume_score = 7
            elif volume_ratio > 1.0: volume_score = 5
            elif volume_ratio > 0.8: volume_score = 3
            
            score += volume_score
            
            details = {
                'above_sma10': above_sma10,
                'above_sma20': above_sma20,
                'above_sma50': above_sma50,
                'above_ema21': above_ema21,
                'ma_alignment': ma_alignment,
                'roc_5': roc_5,
                'roc_20': roc_20,
                'volume_ratio': volume_ratio,
                'ma_score': ma_score,
                'alignment_score': 10 if ma_alignment else 0,
                'roc_score': min(roc_score, 10),
                'volume_score': volume_score
            }
            
            return min(int(score), 40), details
            
        except Exception as e:
            logger.error(f"Error calculating trend score: {e}")
            return 0, {}
    
    def _calculate_structure_score(self, data: pd.DataFrame) -> Tuple[int, Dict]:
        """
        Calculate STRUCTURE score (0-40 points) using official methodology
        
        Components:
        - EMA vs SMA relationship (15 points)
        - Price position relative to EMAs (15 points)
        - EMA/SMA separation (10 points)
        """
        try:
            current_price = data['Close'].iloc[-1]
            
            # Key moving averages
            ema_21 = data['Close'].ewm(span=21).mean()
            sma_50 = data['Close'].rolling(50).mean()
            ema_8 = data['Close'].ewm(span=8).mean()
            
            score = 0
            details = {}
            
            # EMA vs SMA relationship (15 points)
            ema21_above_sma50 = ema_21.iloc[-1] > sma_50.iloc[-1]
            ema8_above_ema21 = ema_8.iloc[-1] > ema_21.iloc[-1]
            ema8_above_sma50 = ema_8.iloc[-1] > sma_50.iloc[-1]
            
            ema_relationship_score = 0
            if ema21_above_sma50: ema_relationship_score += 7
            if ema8_above_ema21: ema_relationship_score += 4
            if ema8_above_sma50: ema_relationship_score += 4
            
            score += ema_relationship_score
            
            # Price position relative to EMAs (15 points)
            price_above_ema8 = current_price > ema_8.iloc[-1]
            price_above_ema21 = current_price > ema_21.iloc[-1]
            price_above_sma50 = current_price > sma_50.iloc[-1]
            
            price_position_score = 0
            if price_above_ema8: price_position_score += 3
            if price_above_ema21: price_position_score += 6
            if price_above_sma50: price_position_score += 6
            
            score += price_position_score
            
            # EMA/SMA separation (10 points)
            separation_pct = ((ema_21.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]) * 100
            
            separation_score = 0
            if separation_pct > 5: separation_score = 10
            elif separation_pct > 3: separation_score = 7
            elif separation_pct > 1: separation_score = 5
            elif separation_pct > 0: separation_score = 3
            
            score += separation_score
            
            details = {
                'ema21_above_sma50': ema21_above_sma50,
                'ema8_above_ema21': ema8_above_ema21,
                'ema8_above_sma50': ema8_above_sma50,
                'price_above_ema8': price_above_ema8,
                'price_above_ema21': price_above_ema21,
                'price_above_sma50': price_above_sma50,
                'separation_pct': separation_pct,
                'ema_relationship_score': ema_relationship_score,
                'price_position_score': price_position_score,
                'separation_score': separation_score
            }
            
            return min(int(score), 40), details
            
        except Exception as e:
            logger.error(f"Error calculating structure score: {e}")
            return 0, {}
    
    def _calculate_momentum_score(self, data: pd.DataFrame) -> Tuple[int, Dict]:
        """
        Calculate MOMENTUM score (0-40 points) using official methodology
        
        Components:
        - MACD relationship (15 points)
        - MACD histogram trend (10 points)
        - RSI levels (10 points)
        - Momentum divergence (5 points)
        """
        try:
            # MACD calculation
            exp1 = data['Close'].ewm(span=12).mean()
            exp2 = data['Close'].ewm(span=26).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
            
            # RSI calculation
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            score = 0
            details = {}
            
            # MACD relationship (15 points)
            macd_above_signal = macd_line.iloc[-1] > signal_line.iloc[-1]
            macd_above_zero = macd_line.iloc[-1] > 0
            signal_above_zero = signal_line.iloc[-1] > 0
            
            macd_score = 0
            if macd_above_signal: macd_score += 8
            if macd_above_zero: macd_score += 4
            if signal_above_zero: macd_score += 3
            
            score += macd_score
            
            # MACD histogram trend (10 points)
            current_hist = histogram.iloc[-1]
            prev_hist = histogram.iloc[-2]
            hist_rising = current_hist > prev_hist
            hist_positive = current_hist > 0
            
            # Check for histogram acceleration
            hist_accel = (current_hist - prev_hist) > (prev_hist - histogram.iloc[-3])
            
            histogram_score = 0
            if hist_positive: histogram_score += 4
            if hist_rising: histogram_score += 4
            if hist_accel: histogram_score += 2
            
            score += histogram_score
            
            # RSI levels (10 points)
            current_rsi = rsi.iloc[-1]
            
            rsi_score = 0
            if 50 <= current_rsi <= 80: rsi_score = 10  # Bullish but not overbought
            elif 45 <= current_rsi < 50: rsi_score = 7   # Neutral to bullish
            elif 40 <= current_rsi < 45: rsi_score = 5   # Slightly bearish
            elif current_rsi > 80: rsi_score = 3         # Overbought
            elif current_rsi < 30: rsi_score = 1         # Oversold (potential bounce)
            
            score += rsi_score
            
            # Momentum divergence (5 points)
            # Check if price is making higher highs while MACD is not
            price_20_high = data['Close'].tail(20).max()
            price_10_high = data['Close'].tail(10).max()
            macd_20_high = macd_line.tail(20).max()
            macd_10_high = macd_line.tail(10).max()
            
            positive_divergence = (price_10_high > price_20_high and macd_10_high > macd_20_high)
            
            divergence_score = 5 if positive_divergence else 0
            score += divergence_score
            
            details = {
                'macd_above_signal': macd_above_signal,
                'macd_above_zero': macd_above_zero,
                'signal_above_zero': signal_above_zero,
                'hist_rising': hist_rising,
                'hist_positive': hist_positive,
                'current_rsi': current_rsi,
                'positive_divergence': positive_divergence,
                'macd_score': macd_score,
                'histogram_score': histogram_score,
                'rsi_score': rsi_score,
                'divergence_score': divergence_score
            }
            
            return min(int(score), 40), details
            
        except Exception as e:
            logger.error(f"Error calculating momentum score: {e}")
            return 0, {}
    
    def _detect_squeeze(self, data: pd.DataFrame) -> Tuple[bool, int]:
        """
        Detect TTM Squeeze and calculate bonus points
        
        Returns:
            Tuple of (squeeze_active, bonus_points)
        """
        try:
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            sma = data['Close'].rolling(bb_period).mean()
            std = data['Close'].rolling(bb_period).std()
            bb_upper = sma + (std * bb_std)
            bb_lower = sma - (std * bb_std)
            
            # Keltner Channels
            kc_period = 20
            kc_multiplier = 1.5
            ema = data['Close'].ewm(span=kc_period).mean()
            
            # Calculate ATR for Keltner Channels
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(kc_period).mean()
            
            kc_upper = ema + (kc_multiplier * atr)
            kc_lower = ema - (kc_multiplier * atr)
            
            # Squeeze condition: Bollinger Bands inside Keltner Channels
            squeeze_active = (bb_upper.iloc[-1] < kc_upper.iloc[-1] and 
                            bb_lower.iloc[-1] > kc_lower.iloc[-1])
            
            # Bonus points for squeeze
            bonus_points = 20 if squeeze_active else 0
            
            return squeeze_active, bonus_points
            
        except Exception as e:
            logger.error(f"Error detecting squeeze: {e}")
            return False, 0    

    def analyze_ticker(self, ticker: str) -> Optional[OfficialBig3Result]:
        """Analyze a single ticker using official Big 3 methodology"""
        try:
            data = self._fetch_data(ticker)
            if data is None:
                return None
            
            current_price = data['Close'].iloc[-1]
            volume = data['Volume'].iloc[-1]
            
            # Calculate Big 3 components
            trend_score, trend_details = self._calculate_trend_score(data)
            structure_score, structure_details = self._calculate_structure_score(data)
            momentum_score, momentum_details = self._calculate_momentum_score(data)
            
            # Detect squeeze
            squeeze_active, squeeze_bonus = self._detect_squeeze(data)
            
            # Calculate total Big 3 score (out of 120 + squeeze bonus)
            base_score = trend_score + structure_score + momentum_score
            total_score = base_score + squeeze_bonus
            
            # Calculate strength percentage (matches official methodology)
            # Strength % = (Score / 120) * 100, but can exceed 100% with squeeze bonus
            strength_pct = (total_score / 120) * 100
            
            # Check if in Taylor's Focus List
            is_focus_list = ticker in self.taylor_focus_list
            
            # A+ setup criteria (score >= 100 or strength >= 85%)
            is_a_plus = total_score >= 100 or strength_pct >= 85
            
            # Skip if below minimum thresholds
            if total_score < self.config.min_score or strength_pct < self.config.min_strength:
                return None
            
            # Calculate support and ATR
            support_level = data['Low'].tail(20).min()
            
            # ATR calculation
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
            
            return OfficialBig3Result(
                ticker=ticker,
                current_price=current_price,
                big3_score=total_score,
                strength_pct=strength_pct,
                trend_score=trend_score,
                structure_score=structure_score,
                momentum_score=momentum_score,
                squeeze_active=squeeze_active,
                squeeze_bonus=squeeze_bonus,
                is_focus_list=is_focus_list,
                is_a_plus_setup=is_a_plus,
                support_level=support_level,
                atr=atr,
                volume=int(volume),
                analysis_date=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {str(e)}")
            return None
    
    def scan_parallel(self) -> pd.DataFrame:
        """Scan all tickers using parallel processing"""
        watchlist = self.get_watchlist()
        logger.info(f"Scanning {len(watchlist)} tickers using official Big 3 methodology...")
        logger.info(f"Minimum score: {self.config.min_score}/120, Minimum strength: {self.config.min_strength}%")
        
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_ticker = {executor.submit(self.analyze_ticker, ticker): ticker 
                              for ticker in watchlist}
            
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        focus_indicator = "⭐" if result.is_focus_list else ""
                        squeeze_indicator = "🔥" if result.squeeze_active else ""
                        logger.info(f"✓ {ticker}: {result.big3_score}/120 ({result.strength_pct:.0f}%) {focus_indicator}{squeeze_indicator}")
                    else:
                        logger.debug(f"✗ {ticker}: Below threshold")
                except Exception as e:
                    logger.error(f"✗ {ticker}: {str(e)}")
        
        logger.info(f"Scan complete! Found {len(results)} qualifying setups")
        
        self.results = results
        return self._create_summary_dataframe()
    
    def _create_summary_dataframe(self) -> pd.DataFrame:
        """Create summary DataFrame matching official format"""
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        for r in self.results:
            summary_data.append({
                'Rank': 0,  # Will be set after sorting
                'Ticker': r.ticker,
                'Price': f"${r.current_price:.2f}",
                'Strength': f"{r.strength_pct:.0f}%",
                'Score': r.big3_score,
                'Focus_List': "⭐" if r.is_focus_list else "",
                'A+_Setup': "★★★" if r.is_a_plus_setup else "",
                'Squeeze': "🔥" if r.squeeze_active else "",
                'Trend': r.trend_score,
                'Structure': r.structure_score,
                'Momentum': r.momentum_score,
                'Volume': f"{r.volume:,.0f}",
                'Support': f"${r.support_level:.2f}",
                'ATR': f"${r.atr:.2f}"
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('Score', ascending=False)
        
        # Add ranking
        df['Rank'] = range(1, len(df) + 1)
        
        # Reorder columns to match official format
        column_order = ['Rank', 'Ticker', 'Strength', 'Score', 'Focus_List', 'A+_Setup', 'Squeeze', 
                       'Price', 'Trend', 'Structure', 'Momentum', 'Volume', 'Support', 'ATR']
        
        return df[column_order]
    
    def print_detailed_analysis(self, ticker: str):
        """Print detailed analysis matching official format"""
        result = next((r for r in self.results if r.ticker == ticker), None)
        
        if not result:
            print(f"No analysis found for {ticker}")
            return
        
        print(f"\n{'='*80}")
        print(f"📊 OFFICIAL BIG 3 ANALYSIS: {ticker}")
        if result.is_focus_list:
            print("⭐ TAYLOR'S FOCUS LIST ⭐")
        print(f"{'='*80}")
        
        print(f"\n💰 Current Price: ${result.current_price:.2f}")
        print(f"📈 Volume: {result.volume:,.0f}")
        print(f"📅 Analysis Date: {result.analysis_date}")
        
        print(f"\n🎯 BIG 3 SCORE: {result.big3_score}/120")
        print(f"💪 STRENGTH: {result.strength_pct:.0f}%")
        
        if result.is_a_plus_setup:
            print("⭐ A+ SETUP - HIGHEST PROBABILITY ⭐")
        
        if result.squeeze_active:
            print("🔥 TTM SQUEEZE ACTIVE - COMPRESSION DETECTED 🔥")
            print(f"   Squeeze Bonus: +{result.squeeze_bonus} points")
        
        # Component breakdown
        print(f"\n--- 📈 TREND COMPONENT: {result.trend_score}/40 ---")
        print(f"--- 🏗️  STRUCTURE COMPONENT: {result.structure_score}/40 ---")
        print(f"--- ⚡ MOMENTUM COMPONENT: {result.momentum_score}/40 ---")
        
        # Risk metrics
        print(f"\n--- 📊 RISK METRICS ---")
        print(f"  Support Level: ${result.support_level:.2f}")
        print(f"  ATR: ${result.atr:.2f}")
        print(f"  Distance to Support: {((result.current_price - result.support_level) / result.current_price) * 100:.1f}%")
        
        print(f"\n{'='*80}\n")
    
    def export_results(self, filename: Optional[str] = None) -> str:
        """Export results to JSON file"""
        if not self.results:
            logger.warning("No results to export")
            return ""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/official_big3_scan_{timestamp}.json"
        
        # Create results directory
        Path("results").mkdir(exist_ok=True)
        
        # Convert results to dict for JSON serialization
        export_data = {
            'scan_config': asdict(self.config),
            'scan_date': datetime.now().isoformat(),
            'total_results': len(self.results),
            'results': [asdict(r) for r in self.results]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Results exported to {filename}")
        return filename
    
    def get_focus_list_results(self) -> List[OfficialBig3Result]:
        """Get results for Taylor's Focus List only"""
        return [r for r in self.results if r.is_focus_list]
    
    def get_a_plus_setups(self) -> List[OfficialBig3Result]:
        """Get A+ setups (score >= 100)"""
        return [r for r in self.results if r.is_a_plus_setup]
    
    def get_squeeze_setups(self) -> List[OfficialBig3Result]:
        """Get squeeze setups"""
        return [r for r in self.results if r.squeeze_active]


def main():
    """Main function demonstrating official Big 3 scanner"""
    print("\n" + "="*80)
    print("🚀 OFFICIAL BIG 3 SCANNER")
    print("Matches Taylor Horton's Exact Methodology")
    print("="*80 + "\n")
    
    # Configuration options
    config = OfficialScanConfig(
        min_score=80,           # Minimum score out of 120
        min_strength=70.0,      # Minimum strength percentage
        focus_list_only=False,  # Scan all tickers, not just focus list
        include_etfs=True,      # Include ETFs and indexes
        max_workers=10          # Parallel processing
    )
    
    # Initialize scanner
    scanner = OfficialBig3Scanner(config)
    
    # Run scan
    start_time = time.time()
    results_df = scanner.scan_parallel()
    scan_time = time.time() - start_time
    
    if not results_df.empty:
        print(f"\n⏱️  Scan completed in {scan_time:.1f} seconds")
        print(f"\n{'='*80}")
        print("📈 BIG 3 LEADERBOARD - QUALIFYING SETUPS")
        print("="*80)
        print(results_df.to_string(index=False))
        
        # Export results
        export_file = scanner.export_results()
        
        # Show focus list results
        focus_results = scanner.get_focus_list_results()
        if focus_results:
            print(f"\n{'='*80}")
            print("⭐ TAYLOR'S FOCUS LIST RESULTS")
            print("="*80)
            for result in focus_results:
                print(f"{result.ticker}: {result.strength_pct:.0f}% ({result.big3_score}/120)")
        
        # Show A+ setups
        a_plus_setups = scanner.get_a_plus_setups()
        if a_plus_setups:
            print(f"\n{'='*80}")
            print("★★★ A+ SETUPS - DETAILED ANALYSIS")
            print("="*80)
            for setup in a_plus_setups[:3]:  # Top 3 A+ setups
                scanner.print_detailed_analysis(setup.ticker)
        
        # Show squeeze setups
        squeeze_setups = scanner.get_squeeze_setups()
        if squeeze_setups and not a_plus_setups:
            print(f"\n{'='*80}")
            print("🔥 SQUEEZE SETUPS - DETAILED ANALYSIS")
            print("="*80)
            for setup in squeeze_setups[:3]:  # Top 3 squeeze setups
                scanner.print_detailed_analysis(setup.ticker)
        
        # Trading guidelines
        print(f"\n{'='*80}")
        print("📚 OFFICIAL BIG 3 TRADING GUIDELINES")
        print("="*80)
        print("""
🎯 SETUP SELECTION (Taylor Horton's Method):
   • Focus on scores ≥100/120 for highest probability
   • Prioritize Taylor's Focus List tickers (⭐)
   • Look for squeeze setups (🔥) for explosive moves
   • A+ setups (★★★) are the cream of the crop

💰 SCORING BREAKDOWN:
   • Trend: 0-40 points (price vs MAs, alignment, ROC, volume)
   • Structure: 0-40 points (EMA relationships, price position)
   • Momentum: 0-40 points (MACD, histogram, RSI, divergence)
   • Squeeze Bonus: +20 points (TTM Squeeze compression)

📊 STRENGTH PERCENTAGE:
   • 100%+ = Exceptional setup (rare)
   • 90-99% = Excellent setup
   • 80-89% = Good setup
   • 70-79% = Decent setup
   • <70% = Avoid

⚡ PUT CREDIT SPREAD STRATEGY:
   • Use high-scoring setups for bullish credit spreads
   • Target 30-45 DTE for optimal time decay
   • Maintain 3:1 or better risk/reward ratio
   • Place short strikes above support levels
   • Close at 50-80% of max profit
        """)
        
        print(f"\n📁 Results saved to: {export_file}")
    
    else:
        print("❌ No qualifying setups found with current criteria")
        print("💡 Try lowering the minimum score or strength percentage")


if __name__ == "__main__":
    main()