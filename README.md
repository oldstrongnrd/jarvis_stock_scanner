# Jarvis Options Scanner

## Quick Install
```bash
./install.sh
source ~/.bashrc
big3
```

## Commands
- `big3` - Full scanner
- `big3 focus` - Taylor's Focus List only  
- `big3 analyze AAPL` - Analyze specific ticker
- `big3 quick` - High-quality setups only
- `big3 help` - Show help

## Requirements
- Python 3.7+
- Internet connection for market data

## Files
- `official_big3_scanner.py` - Main scanner engine
- `cli_scanner.py` - Command line interface
- `config.json` - Configuration settings
- `big3` - Global command wrapper
- `install.sh` - Installation script

## Configuration
Edit `config.json` to customize:
- Minimum scores and thresholds
- Custom watchlists
- Risk parameters
