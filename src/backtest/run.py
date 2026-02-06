"""
Backtesting CLI Tool

Command-line interface for running backtests on PR3DICT strategies.

Usage:
    python -m src.backtest.run --strategy arbitrage --start 2024-01-01 --end 2024-12-31
    python -m src.backtest.run --strategy all --start 2024-01-01 --end 2024-12-31 --balance 50000
"""
import sys
import argparse
import logging
from datetime import datetime
from decimal import Decimal
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.backtest.engine import BacktestEngine, BacktestConfig
from src.backtest.data import HistoricalDataLoader
from src.backtest.report import generate_report
from src.strategies.arbitrage import ArbitrageStrategy
from src.strategies.market_making import MarketMakingStrategy
from src.strategies.behavioral import BehavioralEdgeStrategy
from src.risk.manager import RiskManager, RiskConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


AVAILABLE_STRATEGIES = {
    'arbitrage': ArbitrageStrategy,
    'market_making': MarketMakingStrategy,
    'behavioral': BehavioralEdgeStrategy,
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Backtest PR3DICT trading strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run arbitrage strategy for 2024
  python -m src.backtest.run --strategy arbitrage --start 2024-01-01 --end 2024-12-31
  
  # Test all strategies with custom initial balance
  python -m src.backtest.run --strategy all --start 2024-01-01 --end 2024-12-31 --balance 50000
  
  # Use custom data file and adjust commission
  python -m src.backtest.run --strategy arbitrage --data ./mydata.csv --commission 0.02
  
  # Generate sample data for testing
  python -m src.backtest.run --generate-sample --output sample_data.csv
        """
    )
    
    # Strategy selection
    parser.add_argument(
        '--strategy',
        type=str,
        choices=list(AVAILABLE_STRATEGIES.keys()) + ['all'],
        default='arbitrage',
        help='Strategy to backtest (or "all" for parallel testing)'
    )
    
    # Date range
    parser.add_argument(
        '--start',
        type=str,
        required='--generate-sample' not in sys.argv,
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        required='--generate-sample' not in sys.argv,
        help='End date (YYYY-MM-DD)'
    )
    
    # Data source
    parser.add_argument(
        '--data',
        type=str,
        help='Path to historical data CSV file (default: auto-load from data/historical/)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Directory containing historical data files'
    )
    
    # Configuration
    parser.add_argument(
        '--balance',
        type=float,
        default=10000,
        help='Initial balance in USD (default: 10000)'
    )
    
    parser.add_argument(
        '--commission',
        type=float,
        default=0.01,
        help='Commission rate as decimal (default: 0.01 = 1%%)'
    )
    
    parser.add_argument(
        '--slippage',
        type=float,
        default=5,
        help='Slippage in basis points (default: 5)'
    )
    
    parser.add_argument(
        '--max-positions',
        type=int,
        default=10,
        help='Maximum concurrent positions (default: 10)'
    )
    
    parser.add_argument(
        '--platforms',
        nargs='+',
        default=['kalshi'],
        help='Platforms to include (default: kalshi)'
    )
    
    # Output
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory for reports (default: ./backtest_reports/)'
    )
    
    parser.add_argument(
        '--no-risk-manager',
        action='store_true',
        help='Disable risk management checks'
    )
    
    # Sample data generation
    parser.add_argument(
        '--generate-sample',
        action='store_true',
        help='Generate sample historical data for testing'
    )
    
    parser.add_argument(
        '--sample-markets',
        type=int,
        default=5,
        help='Number of markets in sample data (default: 5)'
    )
    
    parser.add_argument(
        '--sample-days',
        type=int,
        default=30,
        help='Number of days in sample data (default: 30)'
    )
    
    return parser.parse_args()


def generate_sample_data(args):
    """Generate sample historical data."""
    from datetime import timedelta
    
    output_path = Path(args.output or 'sample_data.csv')
    
    loader = HistoricalDataLoader()
    loader.generate_sample_data(
        filepath=output_path,
        num_markets=args.sample_markets,
        days=args.sample_days
    )
    
    print(f"\nSample data generated: {output_path}")
    print(f"Markets: {args.sample_markets}")
    print(f"Days: {args.sample_days}")
    print("\nRun backtest with:")
    print(f"  python -m src.backtest.run --strategy arbitrage --data {output_path} \\")
    print(f"    --start {(datetime.now() - timedelta(days=args.sample_days)).strftime('%Y-%m-%d')} \\")
    print(f"    --end {datetime.now().strftime('%Y-%m-%d')}")


def load_strategy(strategy_name: str):
    """Load strategy class by name."""
    if strategy_name not in AVAILABLE_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    strategy_class = AVAILABLE_STRATEGIES[strategy_name]
    return strategy_class()


def run_backtest(args):
    """Run the backtest."""
    from datetime import timedelta
    
    # Parse dates
    start_date = datetime.fromisoformat(args.start)
    end_date = datetime.fromisoformat(args.end)
    
    logger.info(f"Initializing backtest: {start_date.date()} to {end_date.date()}")
    
    # Load historical data
    loader = HistoricalDataLoader(
        data_dir=Path(args.data_dir) if args.data_dir else None
    )
    
    if args.data:
        # Load specific file
        loader.load_csv(Path(args.data))
    else:
        # Auto-load from directory
        loader.load_from_directory(
            start_date=start_date,
            end_date=end_date,
            platforms=args.platforms
        )
    
    # Check if we have data
    if not loader.snapshots:
        logger.error("No historical data loaded!")
        logger.info("To generate sample data, run:")
        logger.info("  python -m src.backtest.run --generate-sample --output sample_data.csv")
        sys.exit(1)
    
    # Load strategies
    if args.strategy == 'all':
        strategies = [load_strategy(name) for name in AVAILABLE_STRATEGIES.keys()]
        logger.info(f"Testing {len(strategies)} strategies in parallel")
    else:
        strategies = [load_strategy(args.strategy)]
        logger.info(f"Testing strategy: {args.strategy}")
    
    # Risk manager
    risk_manager = None
    if not args.no_risk_manager:
        risk_config = RiskConfig(
            max_position_size=Decimal("1000"),
            max_daily_loss=Decimal("500"),
            max_total_risk=Decimal("0.20")
        )
        risk_manager = RiskManager(risk_config)
        logger.info("Risk management enabled")
    
    # Backtest configuration
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_balance=Decimal(str(args.balance)),
        max_positions=args.max_positions,
        commission_rate=Decimal(str(args.commission)),
        slippage_bps=Decimal(str(args.slippage)),
        platforms=args.platforms
    )
    
    # Create and run engine
    engine = BacktestEngine(
        data_loader=loader,
        strategies=strategies,
        config=config,
        risk_manager=risk_manager
    )
    
    logger.info("Starting backtest...")
    results = engine.run()
    
    # Generate report
    output_dir = Path(args.output or './backtest_reports')
    report = generate_report(results, output_dir)
    
    logger.info(f"\nBacktest complete!")
    logger.info(f"Reports saved to: {output_dir}")


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        if args.generate_sample:
            generate_sample_data(args)
        else:
            run_backtest(args)
    except KeyboardInterrupt:
        logger.info("\nBacktest interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
