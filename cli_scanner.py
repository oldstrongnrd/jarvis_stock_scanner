#!/usr/bin/env python3
"""
Command Line Interface for Jarvis Stock Scanner
Provides easy access to scanning functionality with various options
"""

import click
import json
from pathlib import Path
from jarvis_scanner import JarvisScanner, ScanConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_file: str = "config.json") -> dict:
    """Load configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {config_file} not found, using defaults")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        return {}


@click.group()
def cli():
    """Jarvis Stock Scanner CLI"""
    pass


@cli.command()
@click.option('--min-score', '-s', default=8, help='Minimum Technical score (0-12)')
@click.option('--min-volume', '-v', default=1000000, help='Minimum daily volume')
@click.option('--config', '-c', default='config.json', help='Configuration file path')
@click.option('--export', '-e', is_flag=True, help='Export results to JSON')
@click.option('--detailed', '-d', help='Show detailed analysis for specific ticker')
@click.option('--workers', '-w', default=10, help='Number of parallel workers')
def scan(min_score, min_volume, config, export, detailed, workers):
    """Run the Jarvis scanner"""
    
    # Load configuration
    config_data = load_config(config)
    scanner_config = config_data.get('scanner_config', {})
    
    # Override with CLI arguments
    scanner_config.update({
        'min_score': min_score,
        'min_volume': min_volume,
        'max_workers': workers
    })
    
    # Create scanner config
    scan_config = ScanConfig(**scanner_config)
    
    # Get custom watchlist if specified
    custom_tickers = config_data.get('custom_watchlist')
    
    # Initialize and run scanner
    scanner = JarvisScanner(scan_config, custom_tickers)
    
    click.echo("🚀 Starting Jarvis Scanner...")
    results_df = scanner.scan_parallel()
    
    if not results_df.empty:
        click.echo(f"\n📈 Found {len(results_df)} qualifying setups:")
        click.echo(results_df.to_string(index=False))
        
        if export:
            filename = scanner.export_results()
            click.echo(f"\n📁 Results exported to: {filename}")
        
        if detailed:
            scanner.print_detailed_analysis(detailed.upper())
    else:
        click.echo("❌ No qualifying setups found")


@cli.command()
@click.argument('ticker')
@click.option('--config', '-c', default='config.json', help='Configuration file path')
def analyze(ticker, config):
    """Analyze a specific ticker"""
    
    config_data = load_config(config)
    scanner_config = config_data.get('scanner_config', {})
    scan_config = ScanConfig(**scanner_config)
    
    scanner = JarvisScanner(scan_config)
    
    click.echo(f"🔍 Analyzing {ticker.upper()}...")
    result = scanner.analyze_ticker(ticker.upper())
    
    if result:
        scanner.results = [result]
        scanner.print_detailed_analysis(ticker.upper())
    else:
        click.echo(f"❌ {ticker.upper()} does not meet criteria or insufficient data")


@cli.command()
@click.option('--output', '-o', default='watchlist.json', help='Output file for watchlist')
def create_watchlist(output):
    """Create a custom watchlist file"""
    
    default_watchlist = [
        'SPY', 'QQQ', 'IWM', 'DIA',
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
        'JPM', 'BAC', 'WMT', 'HD', 'PG', 'JNJ', 'V', 'MA'
    ]
    
    watchlist_data = {
        "name": "Custom Watchlist",
        "description": "High-volume, optionable stocks for credit spreads",
        "tickers": default_watchlist,
        "created": "2024-01-01",
        "criteria": {
            "min_volume": 1000000,
            "min_price": 10.0,
            "options_available": True,
            "liquid_options": True
        }
    }
    
    with open(output, 'w') as f:
        json.dump(watchlist_data, f, indent=2)
    
    click.echo(f"📝 Watchlist created: {output}")
    click.echo("Edit the file to customize your watchlist")


@cli.command()
def config_template():
    """Generate a configuration template"""
    
    template = {
        "scanner_config": {
            "min_score": 8,
            "min_volume": 1000000,
            "min_price": 10.0,
            "max_price": 1000.0,
            "target_dte": 30,
            "spread_width": 5,
            "min_credit_ratio": 0.25,
            "max_workers": 10,
            "cache_duration": 300
        },
        "custom_watchlist": [
            "SPY", "QQQ", "IWM", "AAPL", "MSFT"
        ],
        "export_settings": {
            "auto_export": True,
            "export_format": "json",
            "include_charts": False
        },
        "risk_management": {
            "max_position_size_pct": 2.0,
            "profit_target_pct": 50,
            "max_loss_pct": 200,
            "min_days_to_expiration": 7
        }
    }
    
    with open('config_template.json', 'w') as f:
        json.dump(template, f, indent=2)
    
    click.echo("📋 Configuration template created: config_template.json")


if __name__ == '__main__':
    cli()