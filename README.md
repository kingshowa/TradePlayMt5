# TradePlay MT5

TradePlay MT5 is a Python-based algorithmic trading framework built for MetaTrader 5.  
It is designed to test, structure, and execute trading strategies using market data, technical indicators, risk management, and live trade execution.

> Disclaimer: This project is for learning, experimentation, and software engineering portfolio purposes only. It is not financial advice and should not be used for real-money trading without proper testing, risk controls, and professional review.

## Overview

This project explores how automated trading systems can be structured using clean Python modules. It includes components for connecting to MetaTrader 5, fetching market data, calculating technical indicators, defining trading strategies, managing risk, and executing live trades.

The system is organized around a modular architecture so that indicators, strategies, risk rules, market data providers, and execution logic can be developed and tested independently.

## Key Features

- MetaTrader 5 connection management
- Live trading execution adapter
- Tick-driven trading logic
- Historical candle warm-up before live execution
- Strategy-based signal generation
- Technical indicator modules
- Risk management and position sizing
- Stop-loss and take-profit handling
- PSAR-based trailing stop logic
- Trade event logging to CSV
- Modular strategy architecture
- Market structure and support/resistance modules
- Backtesting-oriented project structure

## Tech Stack

- **Language:** Python
- **Trading Platform:** MetaTrader 5
- **Main Library:** MetaTrader5 Python package
- **Data Handling:** CSV logging
- **Architecture:** Modular Python package structure
- **Version Control:** Git and GitHub

## Project Structure

```text
TradePlayMt5/
├── app/
│   ├── connection/
│   │   └── mt5_connection.py
│   ├── core/
│   │   ├── indicators/
│   │   ├── market/
│   │   ├── patterns/
│   │   ├── risk/
│   │   ├── strategy/
│   │   └── structure/
│   └── live/
│       ├── live_precision_trader.py
│       ├── live_trader_psar.py
│       ├── mt5_live_executor.py
│       └── trade log CSV files
├── test/
│   ├── backtest/
│   ├── connection/
│   └── core/
└── README.md
