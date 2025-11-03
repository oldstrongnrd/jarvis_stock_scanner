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
    min_quality_score: int = 55   # Minimum quality score (0-100)
    focus_list_only: bool = False # Scan only Taylor's Focus List
    include_etfs: bool = True     # Include ETFs and indexes
    target_dte: int = 30          # Target days to expiration
    spread_width: int = 5         # Default spread width
    min_credit_ratio: float = 0.25 # Minimum credit/risk ratio
    max_workers: int = 10         # Parallel processing threads
    min_volume: int = 500000      # Minimum average daily volume
    min_price: float = 10.0       # Minimum stock price
    max_price: float = 1000.0     # Maximum stock price
    obsidian_vault: str = ""      # Path to Obsidian vault (optional)
    enable_obsidian: bool = False # Enable Obsidian markdown output


@dataclass
class OfficialBig3Result:
    """Result structure matching official methodology"""
    ticker: str
    current_price: float
    big3_score: int              # Out of 120
    strength_pct: float          # Strength percentage
    quality_score: int           # Quality score (0-100)
    tier: str                    # TIER_1, TIER_2, TIER_3
    trend_direction: str         # BULLISH, BEARISH, NEUTRAL

    # Big 3 Components
    trend_score: int             # Trend component (0-40)
    structure_score: int         # Structure component (0-40)
    momentum_score: int          # Momentum component (0-40)

    # Multi-Timeframe Squeezes
    weekly_squeeze: bool
    daily_squeeze: bool
    four_hour_squeeze: bool
    one_hour_squeeze: bool
    squeeze_bonus: int           # Squeeze bonus points

    # VWAP Analysis
    vwap_direction: str          # RISING, FALLING, FLAT
    price_vs_vwap: str           # ABOVE, BELOW, AT

    # Volume Analysis
    volume: int
    volume_avg_20: int
    volume_ratio: float          # Current vs 20-day average
    volume_trend: str            # INCREASING, DECREASING, STABLE
    institutional_activity: str  # BUYING (CYAN), SELLING (MAGENTA), NEUTRAL

    # Money Flow Index
    mfi: float
    mfi_zone: str                # BULLISH (>50), BEARISH (<50), OVERBOUGHT (>80), OVERSOLD (<20)

    # Moving Averages Alignment
    ma_alignment: str            # PERFECT, GOOD, PARTIAL, POOR
    price_above_ema21: bool
    price_above_sma50: bool
    price_above_sma200: bool
    ema21_above_sma50: bool
    sma50_above_sma200: bool

    # MACD
    macd_aligned: bool           # Aligned with trend direction
    macd_histogram_color: str    # GREEN, RED, NEUTRAL

    # ATR Targets
    atr: float
    atr_target_plus1: float      # +1 ATR target
    atr_target_plus2: float      # +2 ATR target
    atr_target_minus1: float     # -1 ATR target
    atr_target_minus2: float     # -2 ATR target

    # Support/Resistance
    support_level: float
    resistance_level: float
    distance_to_ema21_pct: float # Distance from 21 EMA

    # Flags
    is_focus_list: bool          # In Taylor's Focus List
    is_a_plus_setup: bool        # A+ setup (score >= 100)
    perfect_nested_squeeze: bool # All 4 timeframes in squeeze

    # Metadata
    sector: str                  # Stock sector
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

    def _fetch_multi_timeframe_data(self, ticker: str) -> Dict[str, Optional[pd.DataFrame]]:
        """Fetch data for multiple timeframes"""
        timeframes = {}
        try:
            stock = yf.Ticker(ticker)

            # Weekly data (1 year)
            timeframes['weekly'] = stock.history(period='1y', interval='1wk')

            # Daily data (6 months)
            timeframes['daily'] = stock.history(period='6mo', interval='1d')

            # 4-hour data (60 days)
            try:
                timeframes['4h'] = stock.history(period='60d', interval='1h')
            except:
                timeframes['4h'] = None

            # 1-hour data (30 days)
            try:
                timeframes['1h'] = stock.history(period='30d', interval='1h')
            except:
                timeframes['1h'] = None

        except Exception as e:
            logger.error(f"Error fetching multi-timeframe data for {ticker}: {str(e)}")

        return timeframes

    def _detect_squeeze_on_timeframe(self, data: pd.DataFrame) -> bool:
        """
        Detect TTM Squeeze on a specific timeframe

        Returns True if Bollinger Bands are inside Keltner Channels
        """
        try:
            if data is None or len(data) < 20:
                return False

            # Bollinger Bands (20, 2.0)
            bb_length = 20
            bb_std = 2.0
            sma = data['Close'].rolling(bb_length).mean()
            std = data['Close'].rolling(bb_length).std()
            bb_upper = sma + (std * bb_std)
            bb_lower = sma - (std * bb_std)

            # Keltner Channels (20, 1.5)
            kc_length = 20
            kc_mult = 1.5
            ema = data['Close'].ewm(span=kc_length).mean()

            # Calculate ATR for Keltner Channels
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(kc_length).mean()

            kc_upper = ema + (kc_mult * atr)
            kc_lower = ema - (kc_mult * atr)

            # Squeeze condition: BB inside KC
            squeeze = (bb_upper.iloc[-1] < kc_upper.iloc[-1] and
                      bb_lower.iloc[-1] > kc_lower.iloc[-1])

            return squeeze

        except Exception as e:
            logger.debug(f"Error detecting squeeze: {e}")
            return False

    def _calculate_vwap_analysis(self, data: pd.DataFrame) -> Tuple[str, str]:
        """
        Calculate VWAP direction and price position

        Returns: (vwap_direction, price_vs_vwap)
        """
        try:
            # Calculate VWAP
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()

            # VWAP direction (compare current to 5 periods ago)
            vwap_change = vwap.iloc[-1] - vwap.iloc[-6]
            if vwap_change > 0:
                vwap_direction = "RISING"
            elif vwap_change < 0:
                vwap_direction = "FALLING"
            else:
                vwap_direction = "FLAT"

            # Price vs VWAP
            current_price = data['Close'].iloc[-1]
            current_vwap = vwap.iloc[-1]

            pct_diff = abs((current_price - current_vwap) / current_vwap) * 100

            if pct_diff < 0.5:  # Within 0.5%
                price_vs_vwap = "AT"
            elif current_price > current_vwap:
                price_vs_vwap = "ABOVE"
            else:
                price_vs_vwap = "BELOW"

            return vwap_direction, price_vs_vwap

        except Exception as e:
            logger.debug(f"Error calculating VWAP: {e}")
            return "FLAT", "AT"

    def _calculate_volume_analysis(self, data: pd.DataFrame) -> Dict:
        """
        Calculate volume metrics and institutional activity detection

        Returns dict with volume metrics
        """
        try:
            current_volume = int(data['Volume'].iloc[-1])
            volume_avg_20 = int(data['Volume'].tail(20).mean())
            volume_ratio = current_volume / volume_avg_20 if volume_avg_20 > 0 else 1.0

            # Volume trend (last 5 vs previous 15)
            recent_avg = data['Volume'].tail(5).mean()
            previous_avg = data['Volume'].tail(20).head(15).mean()

            if recent_avg > previous_avg * 1.2:
                volume_trend = "INCREASING"
            elif recent_avg < previous_avg * 0.8:
                volume_trend = "DECREASING"
            else:
                volume_trend = "STABLE"

            # Institutional activity detection
            # CYAN = High volume + price up
            # MAGENTA = High volume + price down
            price_up = data['Close'].iloc[-1] > data['Close'].iloc[-2]
            high_volume = volume_ratio > 1.5

            if high_volume and price_up:
                institutional_activity = "BUYING"  # CYAN
            elif high_volume and not price_up:
                institutional_activity = "SELLING"  # MAGENTA
            else:
                institutional_activity = "NEUTRAL"

            return {
                'volume': current_volume,
                'volume_avg_20': volume_avg_20,
                'volume_ratio': volume_ratio,
                'volume_trend': volume_trend,
                'institutional_activity': institutional_activity
            }

        except Exception as e:
            logger.debug(f"Error calculating volume analysis: {e}")
            return {
                'volume': 0,
                'volume_avg_20': 0,
                'volume_ratio': 1.0,
                'volume_trend': "STABLE",
                'institutional_activity': "NEUTRAL"
            }

    def _calculate_mfi(self, data: pd.DataFrame, period: int = 14) -> Tuple[float, str]:
        """
        Calculate Money Flow Index (MFI) - Volume-weighted RSI

        Returns: (mfi_value, mfi_zone)
        """
        try:
            # Calculate typical price
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3

            # Calculate money flow
            money_flow = typical_price * data['Volume']

            # Positive and negative money flow
            positive_flow = pd.Series(0.0, index=data.index)
            negative_flow = pd.Series(0.0, index=data.index)

            for i in range(1, len(data)):
                if typical_price.iloc[i] > typical_price.iloc[i-1]:
                    positive_flow.iloc[i] = money_flow.iloc[i]
                elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                    negative_flow.iloc[i] = money_flow.iloc[i]

            # Calculate MFI
            positive_mf = positive_flow.rolling(period).sum()
            negative_mf = negative_flow.rolling(period).sum()

            mfi = 100 - (100 / (1 + (positive_mf / negative_mf.replace(0, 1))))
            current_mfi = mfi.iloc[-1]

            # Determine MFI zone
            if current_mfi > 80:
                mfi_zone = "OVERBOUGHT"
            elif current_mfi > 50:
                mfi_zone = "BULLISH"
            elif current_mfi > 20:
                mfi_zone = "BEARISH"
            else:
                mfi_zone = "OVERSOLD"

            return current_mfi, mfi_zone

        except Exception as e:
            logger.debug(f"Error calculating MFI: {e}")
            return 50.0, "NEUTRAL"

    def _classify_ma_alignment(self, data: pd.DataFrame) -> Tuple[str, Dict]:
        """
        Classify moving average alignment quality

        Returns: (alignment_classification, ma_details)
        """
        try:
            current_price = data['Close'].iloc[-1]

            # Moving averages
            ema_21 = data['Close'].ewm(span=21).mean().iloc[-1]
            sma_50 = data['Close'].rolling(50).mean().iloc[-1]
            sma_200 = data['Close'].rolling(200).mean().iloc[-1] if len(data) >= 200 else sma_50

            # Bullish alignment checks
            price_above_ema21 = current_price > ema_21
            price_above_sma50 = current_price > sma_50
            price_above_sma200 = current_price > sma_200
            ema21_above_sma50 = ema_21 > sma_50
            sma50_above_sma200 = sma_50 > sma_200

            # Count bullish conditions
            bullish_count = sum([
                price_above_ema21,
                price_above_sma50,
                price_above_sma200,
                ema21_above_sma50,
                sma50_above_sma200
            ])

            # Classify alignment
            if bullish_count == 5:
                alignment = "PERFECT"
            elif bullish_count >= 4:
                alignment = "GOOD"
            elif bullish_count >= 2:
                alignment = "PARTIAL"
            else:
                alignment = "POOR"

            ma_details = {
                'price_above_ema21': price_above_ema21,
                'price_above_sma50': price_above_sma50,
                'price_above_sma200': price_above_sma200,
                'ema21_above_sma50': ema21_above_sma50,
                'sma50_above_sma200': sma50_above_sma200,
                'ema21': ema_21,
                'sma50': sma_50,
                'sma200': sma_200
            }

            return alignment, ma_details

        except Exception as e:
            logger.debug(f"Error classifying MA alignment: {e}")
            return "POOR", {}

    def _classify_trend_direction(self, data: pd.DataFrame, ma_details: Dict) -> str:
        """
        Classify overall trend direction

        Returns: BULLISH, BEARISH, or NEUTRAL
        """
        try:
            # MACD
            exp1 = data['Close'].ewm(span=12).mean()
            exp2 = data['Close'].ewm(span=26).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=9).mean()
            macd_bullish = macd_line.iloc[-1] > signal_line.iloc[-1]

            # Trend classification based on MA alignment
            bullish_mas = (ma_details.get('price_above_ema21', False) and
                          ma_details.get('ema21_above_sma50', False) and
                          ma_details.get('sma50_above_sma200', False))

            bearish_mas = (not ma_details.get('price_above_ema21', False) and
                          not ma_details.get('ema21_above_sma50', False) and
                          not ma_details.get('sma50_above_sma200', False))

            if bullish_mas and macd_bullish:
                return "BULLISH"
            elif bearish_mas and not macd_bullish:
                return "BEARISH"
            else:
                return "NEUTRAL"

        except Exception as e:
            logger.debug(f"Error classifying trend: {e}")
            return "NEUTRAL"

    def _calculate_quality_score(self, stock_data: Dict) -> int:
        """
        Calculate setup quality score (0-100) based on document specifications

        Scoring breakdown:
        - Squeeze alignment: 40 points max
        - Trend quality: 30 points max
        - Volume analysis: 20 points max
        - Liquidity & quality: 10 points max
        """
        score = 0

        # Squeeze alignment (40 points max)
        if stock_data.get('weekly_squeeze'): score += 15
        if stock_data.get('daily_squeeze'): score += 15
        if stock_data.get('four_hour_squeeze'): score += 5
        if stock_data.get('one_hour_squeeze'): score += 5

        # Trend quality (30 points max)
        ma_alignment = stock_data.get('ma_alignment', 'POOR')
        if ma_alignment == 'PERFECT': score += 15
        elif ma_alignment == 'GOOD': score += 10
        elif ma_alignment == 'PARTIAL': score += 5

        if stock_data.get('macd_aligned'): score += 10
        if stock_data.get('distance_to_ema21_pct', 100) < 2.0: score += 5  # Near EMA21

        # Volume analysis (20 points max)
        volume_ratio = stock_data.get('volume_ratio', 1.0)
        if volume_ratio > 2.0: score += 15
        elif volume_ratio > 1.5: score += 10
        elif volume_ratio > 1.0: score += 5

        if stock_data.get('volume_trend') == 'INCREASING': score += 5

        # Liquidity & quality (10 points max)
        avg_volume = stock_data.get('volume_avg_20', 0)
        if avg_volume > 5000000: score += 5
        elif avg_volume > 1000000: score += 3

        # Price range bonus
        price = stock_data.get('current_price', 0)
        if 50 <= price <= 500: score += 5  # Sweet spot for options
        elif 10 <= price <= 1000: score += 3

        return min(score, 100)

    def _assign_tier(self, quality_score: int, weekly_squeeze: bool, daily_squeeze: bool) -> str:
        """
        Assign tier based on quality score and squeeze conditions

        Tier 1: Perfect nested squeezes (85+)
        Tier 2: Strong setups (70-84)
        Tier 3: Developing setups (55-69)
        """
        if quality_score >= 85 and weekly_squeeze and daily_squeeze:
            return "TIER_1"
        elif quality_score >= 70:
            return "TIER_2"
        elif quality_score >= 55:
            return "TIER_3"
        else:
            return "BELOW_THRESHOLD"

    def _get_sector(self, ticker: str) -> str:
        """
        Get sector for a ticker (simplified version)

        In production, use yfinance ticker.info['sector']
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info.get('sector', 'Unknown')
        except:
            return 'Unknown'

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
        """Analyze a single ticker using complete Big 3 methodology with multi-timeframe analysis"""
        try:
            # Fetch daily data (primary timeframe)
            data = self._fetch_data(ticker)
            if data is None:
                return None

            current_price = data['Close'].iloc[-1]

            # Basic validation
            if not (self.config.min_price <= current_price <= self.config.max_price):
                return None

            # Fetch multi-timeframe data
            tf_data = self._fetch_multi_timeframe_data(ticker)

            # Multi-timeframe squeeze detection
            weekly_squeeze = self._detect_squeeze_on_timeframe(tf_data.get('weekly'))
            daily_squeeze = self._detect_squeeze_on_timeframe(tf_data.get('daily', data))
            four_hour_squeeze = self._detect_squeeze_on_timeframe(tf_data.get('4h'))
            one_hour_squeeze = self._detect_squeeze_on_timeframe(tf_data.get('1h'))

            # Calculate squeeze bonus (20 points if daily squeeze active)
            squeeze_bonus = 20 if daily_squeeze else 0

            # Perfect nested squeeze
            perfect_nested_squeeze = all([weekly_squeeze, daily_squeeze, four_hour_squeeze, one_hour_squeeze])

            # Calculate Big 3 components
            trend_score, trend_details = self._calculate_trend_score(data)
            structure_score, structure_details = self._calculate_structure_score(data)
            momentum_score, momentum_details = self._calculate_momentum_score(data)

            # Calculate total Big 3 score
            base_score = trend_score + structure_score + momentum_score
            total_score = base_score + squeeze_bonus
            strength_pct = (total_score / 120) * 100

            # VWAP analysis
            vwap_direction, price_vs_vwap = self._calculate_vwap_analysis(data)

            # Volume analysis
            volume_metrics = self._calculate_volume_analysis(data)

            # MFI calculation
            mfi, mfi_zone = self._calculate_mfi(data)

            # Moving average alignment
            ma_alignment, ma_details = self._classify_ma_alignment(data)

            # Trend direction classification
            trend_direction = self._classify_trend_direction(data, ma_details)

            # MACD histogram color
            exp1 = data['Close'].ewm(span=12).mean()
            exp2 = data['Close'].ewm(span=26).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line

            if histogram.iloc[-1] > 0:
                macd_histogram_color = "GREEN"
            elif histogram.iloc[-1] < 0:
                macd_histogram_color = "RED"
            else:
                macd_histogram_color = "NEUTRAL"

            macd_aligned = (trend_direction == "BULLISH" and macd_line.iloc[-1] > signal_line.iloc[-1]) or \
                          (trend_direction == "BEARISH" and macd_line.iloc[-1] < signal_line.iloc[-1])

            # ATR calculation and targets
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]

            atr_target_plus1 = current_price + (atr * 1.0)
            atr_target_plus2 = current_price + (atr * 2.0)
            atr_target_minus1 = current_price - (atr * 1.0)
            atr_target_minus2 = current_price - (atr * 2.0)

            # Support and resistance
            support_level = data['Low'].tail(20).min()
            resistance_level = data['High'].tail(20).max()

            # Distance to 21 EMA
            ema_21 = ma_details.get('ema21', current_price)
            distance_to_ema21_pct = abs((current_price - ema_21) / current_price) * 100

            # Prepare data for quality score calculation
            stock_data = {
                'weekly_squeeze': weekly_squeeze,
                'daily_squeeze': daily_squeeze,
                'four_hour_squeeze': four_hour_squeeze,
                'one_hour_squeeze': one_hour_squeeze,
                'ma_alignment': ma_alignment,
                'macd_aligned': macd_aligned,
                'distance_to_ema21_pct': distance_to_ema21_pct,
                'volume_ratio': volume_metrics['volume_ratio'],
                'volume_trend': volume_metrics['volume_trend'],
                'volume_avg_20': volume_metrics['volume_avg_20'],
                'current_price': current_price
            }

            # Calculate quality score
            quality_score = self._calculate_quality_score(stock_data)

            # Assign tier
            tier = self._assign_tier(quality_score, weekly_squeeze, daily_squeeze)

            # Skip if below thresholds
            if (total_score < self.config.min_score or
                strength_pct < self.config.min_strength or
                quality_score < self.config.min_quality_score):
                return None

            # Check if in Taylor's Focus List
            is_focus_list = ticker in self.taylor_focus_list

            # A+ setup criteria
            is_a_plus = total_score >= 100 or strength_pct >= 85

            # Get sector
            sector = self._get_sector(ticker)

            return OfficialBig3Result(
                ticker=ticker,
                current_price=current_price,
                big3_score=total_score,
                strength_pct=strength_pct,
                quality_score=quality_score,
                tier=tier,
                trend_direction=trend_direction,

                # Big 3 Components
                trend_score=trend_score,
                structure_score=structure_score,
                momentum_score=momentum_score,

                # Multi-Timeframe Squeezes
                weekly_squeeze=weekly_squeeze,
                daily_squeeze=daily_squeeze,
                four_hour_squeeze=four_hour_squeeze,
                one_hour_squeeze=one_hour_squeeze,
                squeeze_bonus=squeeze_bonus,

                # VWAP
                vwap_direction=vwap_direction,
                price_vs_vwap=price_vs_vwap,

                # Volume
                volume=volume_metrics['volume'],
                volume_avg_20=volume_metrics['volume_avg_20'],
                volume_ratio=volume_metrics['volume_ratio'],
                volume_trend=volume_metrics['volume_trend'],
                institutional_activity=volume_metrics['institutional_activity'],

                # MFI
                mfi=mfi,
                mfi_zone=mfi_zone,

                # Moving Averages
                ma_alignment=ma_alignment,
                price_above_ema21=ma_details.get('price_above_ema21', False),
                price_above_sma50=ma_details.get('price_above_sma50', False),
                price_above_sma200=ma_details.get('price_above_sma200', False),
                ema21_above_sma50=ma_details.get('ema21_above_sma50', False),
                sma50_above_sma200=ma_details.get('sma50_above_sma200', False),

                # MACD
                macd_aligned=macd_aligned,
                macd_histogram_color=macd_histogram_color,

                # ATR Targets
                atr=atr,
                atr_target_plus1=atr_target_plus1,
                atr_target_plus2=atr_target_plus2,
                atr_target_minus1=atr_target_minus1,
                atr_target_minus2=atr_target_minus2,

                # Support/Resistance
                support_level=support_level,
                resistance_level=resistance_level,
                distance_to_ema21_pct=distance_to_ema21_pct,

                # Flags
                is_focus_list=is_focus_list,
                is_a_plus_setup=is_a_plus,
                perfect_nested_squeeze=perfect_nested_squeeze,

                # Metadata
                sector=sector,
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
        """Create enhanced summary DataFrame with all new metrics"""
        if not self.results:
            return pd.DataFrame()

        summary_data = []
        for r in self.results:
            # Squeeze status string
            squeeze_str = ""
            if r.weekly_squeeze: squeeze_str += "W"
            if r.daily_squeeze: squeeze_str += "D"
            if r.four_hour_squeeze: squeeze_str += "4H"
            if r.one_hour_squeeze: squeeze_str += "1H"
            if not squeeze_str: squeeze_str = "-"

            # Institutional activity indicator
            inst_indicator = ""
            if r.institutional_activity == "BUYING":
                inst_indicator = "🟦"  # CYAN
            elif r.institutional_activity == "SELLING":
                inst_indicator = "🟪"  # MAGENTA

            summary_data.append({
                'Rank': 0,  # Will be set after sorting
                'Ticker': r.ticker,
                'Tier': r.tier.replace('TIER_', 'T'),
                'Quality': r.quality_score,
                'Big3': r.big3_score,
                'Strength': f"{r.strength_pct:.0f}%",
                'Direction': r.trend_direction[0],  # B/N/Bullish
                'Squeezes': squeeze_str,
                'Perfect': "🔥" if r.perfect_nested_squeeze else "",
                'Focus': "⭐" if r.is_focus_list else "",
                'Price': f"${r.current_price:.2f}",
                'VWAP': r.vwap_direction[0],  # R/F/Flat
                'Inst': inst_indicator,
                'MFI': f"{r.mfi:.0f}",
                'Vol_Ratio': f"{r.volume_ratio:.1f}x",
                '+1ATR': f"${r.atr_target_plus1:.2f}",
                '+2ATR': f"${r.atr_target_plus2:.2f}",
                'Support': f"${r.support_level:.2f}",
                'Sector': r.sector[:15] if r.sector != 'Unknown' else ''
            })

        df = pd.DataFrame(summary_data)

        # Sort by tier first, then quality score
        tier_order = {'T1': 1, 'T2': 2, 'T3': 3}
        df['tier_sort'] = df['Tier'].map(tier_order)
        df = df.sort_values(['tier_sort', 'Quality'], ascending=[True, False])
        df = df.drop('tier_sort', axis=1)

        # Add ranking
        df['Rank'] = range(1, len(df) + 1)

        # Reorder columns
        column_order = [
            'Rank', 'Ticker', 'Tier', 'Quality', 'Big3', 'Strength', 'Direction',
            'Squeezes', 'Perfect', 'Focus', 'Price', 'VWAP', 'Inst', 'MFI',
            'Vol_Ratio', '+1ATR', '+2ATR', 'Support', 'Sector'
        ]

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

    def generate_obsidian_markdown(self) -> str:
        """
        Generate Obsidian-formatted markdown report

        Returns: Markdown string formatted for Obsidian
        """
        if not self.results:
            return "# No Results\n\nNo qualifying setups found."

        # Sort results by tier and quality score
        tier_1 = [r for r in self.results if r.tier == "TIER_1"]
        tier_2 = [r for r in self.results if r.tier == "TIER_2"]
        tier_3 = [r for r in self.results if r.tier == "TIER_3"]

        tier_1.sort(key=lambda x: x.quality_score, reverse=True)
        tier_2.sort(key=lambda x: x.quality_score, reverse=True)
        tier_3.sort(key=lambda x: x.quality_score, reverse=True)

        scan_date = datetime.now().strftime("%Y-%m-%d")
        scan_time = datetime.now().strftime("%H:%M:%S")

        md = f"""# Trading Scan - {scan_date}

*Generated: {scan_date} at {scan_time}*
*Scan Version: 2.0 - Enhanced Multi-Timeframe*

---

## Market Overview

**Total Setups Found:** {len(self.results)}
- **Tier 1 (Perfect):** {len(tier_1)}
- **Tier 2 (Strong):** {len(tier_2)}
- **Tier 3 (Developing):** {len(tier_3)}

**Focus List Stocks:** {len([r for r in self.results if r.is_focus_list])}
**Perfect Nested Squeezes:** {len([r for r in self.results if r.perfect_nested_squeeze])}

---

"""

        # Tier 1 Setups
        if tier_1:
            md += "## Tier 1 Setups (Perfect Nested Squeezes)\n\n"
            for result in tier_1:
                md += self._format_setup_markdown(result)
                md += "\n---\n\n"

        # Tier 2 Setups
        if tier_2:
            md += "## Tier 2 Setups (Strong Setups)\n\n"
            for result in tier_2:
                md += self._format_setup_markdown(result)
                md += "\n---\n\n"

        # Tier 3 Setups
        if tier_3:
            md += "## Tier 3 Setups (Developing)\n\n"
            for result in tier_3:
                md += self._format_setup_markdown(result)
                md += "\n---\n\n"

        # Sector Summary
        md += self._generate_sector_summary()

        # Action Items
        md += """
## Action Items

"""
        for result in tier_1[:3]:  # Top 3 tier 1 setups
            md += f"- [ ] Review {result.ticker} at market open\n"

        md += f"""
---

*Scanner Configuration:*
- Minimum Big 3 Score: {self.config.min_score}/120
- Minimum Strength: {self.config.min_strength}%
- Minimum Quality Score: {self.config.min_quality_score}/100
"""

        return md

    def _format_setup_markdown(self, result: OfficialBig3Result) -> str:
        """Format a single setup as markdown"""

        # Squeeze indicators
        squeeze_str = ""
        if result.weekly_squeeze: squeeze_str += "W✓ "
        if result.daily_squeeze: squeeze_str += "D✓ "
        if result.four_hour_squeeze: squeeze_str += "4H✓ "
        if result.one_hour_squeeze: squeeze_str += "1H✓ "

        # Institutional activity emoji
        inst_emoji = ""
        if result.institutional_activity == "BUYING":
            inst_emoji = "🟦 CYAN (Institutional Buying)"
        elif result.institutional_activity == "SELLING":
            inst_emoji = "🟪 MAGENTA (Institutional Selling)"
        else:
            inst_emoji = "⚪ Neutral Volume"

        md = f"""### {result.ticker}{"⭐" if result.is_focus_list else ""}

**Price:** ${result.current_price:.2f} | **Trend:** {result.trend_direction} | **Tier:** {result.tier.replace('_', ' ')}

**Scores:**
- Big 3 Score: {result.big3_score}/120 ({result.strength_pct:.0f}% Strength)
- Quality Score: {result.quality_score}/100
- Trend: {result.trend_score}/40 | Structure: {result.structure_score}/40 | Momentum: {result.momentum_score}/40

**Multi-Timeframe Squeezes:** {squeeze_str}
{"🔥 **PERFECT NESTED SQUEEZE**" if result.perfect_nested_squeeze else ""}

**Market Analysis:**
- VWAP: {result.vwap_direction} | Price vs VWAP: {result.price_vs_vwap}
- Volume: {result.volume:,} (Avg: {result.volume_avg_20:,}) | Ratio: {result.volume_ratio:.2f}x
- Volume Activity: {inst_emoji}
- MFI: {result.mfi:.1f} ({result.mfi_zone})
- MA Alignment: {result.ma_alignment}
- MACD: {"✓" if result.macd_aligned else "✗"} ({result.macd_histogram_color})

**ATR Targets:** (ATR: ${result.atr:.2f})
- Conservative (+1 ATR): ${result.atr_target_plus1:.2f}
- Aggressive (+2 ATR): ${result.atr_target_plus2:.2f}
- Support (-1 ATR): ${result.atr_target_minus1:.2f}
- Strong Support (-2 ATR): ${result.atr_target_minus2:.2f}

**Key Levels:**
- Support: ${result.support_level:.2f}
- Resistance: ${result.resistance_level:.2f}
- Distance to 21 EMA: {result.distance_to_ema21_pct:.2f}%

**Sector:** {result.sector}

"""
        return md

    def _generate_sector_summary(self) -> str:
        """Generate sector concentration summary"""
        from collections import Counter

        sector_counts = Counter([r.sector for r in self.results if r.sector != 'Unknown'])

        md = "## Sector Analysis\n\n"
        md += "**Sector Concentration:**\n\n"

        for sector, count in sector_counts.most_common():
            pct = (count / len(self.results)) * 100
            md += f"- {sector}: {count} setups ({pct:.1f}%)\n"

        md += "\n"
        return md

    def save_obsidian_report(self, vault_path: Optional[str] = None) -> str:
        """
        Save Obsidian markdown report to file

        Args:
            vault_path: Path to Obsidian vault (optional, uses config if not provided)

        Returns: Path to saved file
        """
        if vault_path is None:
            vault_path = self.config.obsidian_vault

        if not vault_path:
            vault_path = "."  # Current directory

        # Create directory structure
        scan_folder = Path(vault_path) / "Trading" / "Daily_Scans"
        scan_folder.mkdir(parents=True, exist_ok=True)

        # Generate filename
        scan_date = datetime.now().strftime("%Y-%m-%d")
        filename = scan_folder / f"{scan_date}_scan.md"

        # Generate and save markdown
        md_content = self.generate_obsidian_markdown()

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logger.info(f"Obsidian report saved to {filename}")
        return str(filename)

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
    """Main function demonstrating enhanced Big 3 scanner with multi-timeframe analysis"""
    print("\n" + "="*90)
    print("🚀 BIG 3 SCANNER - ENHANCED EDITION")
    print("Complete Multi-Timeframe Analysis with Quality Scoring")
    print("="*90 + "\n")

    # Configuration options
    config = OfficialScanConfig(
        min_score=80,             # Minimum Big 3 score (out of 120)
        min_strength=70.0,        # Minimum strength percentage
        min_quality_score=55,     # Minimum quality score (out of 100)
        focus_list_only=False,    # Scan all tickers, not just focus list
        include_etfs=True,        # Include ETFs and indexes
        max_workers=10,           # Parallel processing threads
        min_volume=500000,        # Minimum average daily volume
        min_price=10.0,           # Minimum stock price
        max_price=1000.0,         # Maximum stock price
        enable_obsidian=False     # Enable Obsidian markdown export
    )

    # Initialize scanner
    scanner = OfficialBig3Scanner(config)

    # Run scan
    print("📊 Starting multi-timeframe scan...")
    print(f"   - Analyzing Weekly, Daily, 4H, 1H squeezes")
    print(f"   - Calculating VWAP, MFI, Volume Analysis")
    print(f"   - Detecting institutional activity (CYAN/MAGENTA)")
    print(f"   - Computing quality scores and tier assignments\n")

    start_time = time.time()
    results_df = scanner.scan_parallel()
    scan_time = time.time() - start_time

    if not results_df.empty:
        print(f"\n⏱️  Scan completed in {scan_time:.1f} seconds")

        # Count results by tier
        tier_1_count = len([r for r in scanner.results if r.tier == "TIER_1"])
        tier_2_count = len([r for r in scanner.results if r.tier == "TIER_2"])
        tier_3_count = len([r for r in scanner.results if r.tier == "TIER_3"])
        perfect_squeeze_count = len([r for r in scanner.results if r.perfect_nested_squeeze])

        print(f"\n{'='*90}")
        print("📈 ENHANCED BIG 3 LEADERBOARD")
        print("="*90)
        print(f"\n📊 Results Summary:")
        print(f"   • Total Setups: {len(scanner.results)}")
        print(f"   • Tier 1 (Perfect): {tier_1_count}")
        print(f"   • Tier 2 (Strong): {tier_2_count}")
        print(f"   • Tier 3 (Developing): {tier_3_count}")
        print(f"   • Perfect Nested Squeezes: {perfect_squeeze_count}")
        print(f"\n{'='*90}\n")

        # Display results table
        print(results_df.to_string(index=False))

        # Export JSON results
        export_file = scanner.export_results()

        # Export Obsidian markdown (if enabled)
        if config.enable_obsidian:
            obsidian_file = scanner.save_obsidian_report()
            print(f"\n📝 Obsidian report saved to: {obsidian_file}")

        # Show Tier 1 setups (Perfect Nested Squeezes)
        tier_1_results = [r for r in scanner.results if r.tier == "TIER_1"]
        if tier_1_results:
            print(f"\n{'='*90}")
            print("🔥 TIER 1 SETUPS - PERFECT NESTED SQUEEZES")
            print("="*90)
            for result in tier_1_results[:3]:  # Top 3 Tier 1 setups
                print(f"\n{result.ticker} ⭐" if result.is_focus_list else f"\n{result.ticker}")
                print(f"  Quality: {result.quality_score}/100 | Big3: {result.big3_score}/120 | Trend: {result.trend_direction}")
                print(f"  Squeezes: W:{result.weekly_squeeze} D:{result.daily_squeeze} 4H:{result.four_hour_squeeze} 1H:{result.one_hour_squeeze}")
                print(f"  VWAP: {result.vwap_direction} | MFI: {result.mfi:.0f} | Institutional: {result.institutional_activity}")
                print(f"  ATR Targets: +1=${result.atr_target_plus1:.2f} | +2=${result.atr_target_plus2:.2f}")
                print(f"  Support: ${result.support_level:.2f} | Sector: {result.sector}")

        # Show Focus List results
        focus_results = scanner.get_focus_list_results()
        if focus_results:
            print(f"\n{'='*90}")
            print("⭐ TAYLOR'S FOCUS LIST RESULTS")
            print("="*90)
            for result in focus_results:
                print(f"  {result.ticker}: Quality {result.quality_score}/100 | Big3 {result.big3_score}/120 | {result.tier}")

        # Enhanced trading guidelines
        print(f"\n{'='*90}")
        print("📚 ENHANCED BIG 3 TRADING GUIDELINES")
        print("="*90)
        print("""
🎯 TIER SYSTEM:
   • Tier 1 (85+ Quality): Perfect nested squeezes (W+D+4H+1H)
   • Tier 2 (70-84 Quality): Strong multi-timeframe setups
   • Tier 3 (55-69 Quality): Developing setups, monitor closely

🔥 PERFECT NESTED SQUEEZE STRATEGY:
   • Wait for all 4 timeframes in compression (W, D, 4H, 1H)
   • Enter when VWAP is RISING (bullish) or FALLING (bearish)
   • Look for institutional activity (🟦 CYAN = Buying, 🟪 MAGENTA = Selling)
   • MFI > 50 for longs, MFI < 50 for shorts
   • Target +1 ATR (conservative) or +2 ATR (aggressive)

💰 QUALITY SCORE BREAKDOWN (0-100):
   • Squeeze Alignment: 40 points (W:15, D:15, 4H:5, 1H:5)
   • Trend Quality: 30 points (MA alignment, MACD, EMA distance)
   • Volume Analysis: 20 points (ratio, trend, institutional activity)
   • Liquidity & Quality: 10 points (volume, price range)

📊 ENTRY CONFIRMATION:
   ✓ Weekly + Daily squeeze active (minimum)
   ✓ Trend direction clear (BULLISH or BEARISH, not NEUTRAL)
   ✓ VWAP aligned with trade direction
   ✓ Institutional activity present (CYAN for longs, MAGENTA for shorts)
   ✓ Price near 21 EMA (within 2%)
   ✓ MFI in favorable zone

⚡ VERTICAL SPREAD STRATEGY:
   • Bull Call Spreads: Use TIER 1 bullish setups
   • Bear Put Spreads: Use TIER 1 bearish setups
   • Target 30-45 DTE for optimal theta decay
   • Sell strike at +1 ATR (conservative) or +2 ATR (aggressive)
   • Close at 50-75% max profit on Tier 1 setups
   • Use support/resistance levels for strike selection

🎨 VOLUME INDICATORS:
   • 🟦 CYAN = High volume + Price up (Institutions buying)
   • 🟪 MAGENTA = High volume + Price down (Institutions selling)
   • Only trade when institutional activity aligns with trend

📈 VWAP RULES:
   • RISING VWAP + Price ABOVE = Strong bullish bias
   • FALLING VWAP + Price BELOW = Strong bearish bias
   • Price AT VWAP = Decision point, wait for break
        """)

        print(f"\n📁 Results saved to: {export_file}")
        print(f"\n💡 Tip: Run daily after market close for best results")
        print(f"   Focus on Tier 1 setups with perfect nested squeezes for highest win rate\n")

    else:
        print("❌ No qualifying setups found with current criteria")
        print("\n💡 Suggestions:")
        print("   • Lower min_score (currently 80)")
        print("   • Lower min_strength (currently 70%)")
        print("   • Lower min_quality_score (currently 55)")
        print("   • Expand watchlist")
        print("   • Check if markets are trending or choppy\n")


if __name__ == "__main__":
    main()