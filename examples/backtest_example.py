"""
Example: Running PR3DICT Backtests Programmatically

This example shows how to use the backtesting framework directly
from Python code instead of the CLI.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.engine import BacktestEngine, BacktestConfig
from src.backtest.data import HistoricalDataLoader
from src.backtest.metrics import calculate_metrics
from src.backtest.report import generate_report
from src.strategies.arbitrage import ArbitrageStrategy
from src.risk.manager import RiskManager, RiskConfig


def run_simple_backtest():
    """Simple backtest example."""
    print("Example 1: Simple Backtest\n")
    
    # 1. Load historical data
    loader = HistoricalDataLoader()
    
    # Generate sample data if you don't have real data
    data_file = Path("data/example_data.csv")
    if not data_file.exists():
        print("Generating sample data...")
        loader.generate_sample_data(data_file, num_markets=5, days=30)
    
    loader.load_csv(data_file)
    
    # 2. Setup strategy
    strategy = ArbitrageStrategy()
    
    # 3. Configure backtest
    config = BacktestConfig(
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now(),
        initial_balance=Decimal("10000"),
        commission_rate=Decimal("0.01"),
        slippage_bps=Decimal("5")
    )
    
    # 4. Run backtest
    engine = BacktestEngine(
        data_loader=loader,
        strategies=[strategy],
        config=config
    )
    
    results = engine.run()
    
    # 5. Analyze results
    metrics = calculate_metrics(results)
    print(f"\nTotal Return: {metrics.total_return:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"Win Rate: {metrics.win_rate:.2%}")


def run_multi_strategy_comparison():
    """Compare multiple strategies."""
    print("\nExample 2: Multi-Strategy Comparison\n")
    
    from src.strategies.market_making import MarketMakingStrategy
    from src.strategies.behavioral import BehavioralEdgeStrategy
    
    # Load data
    loader = HistoricalDataLoader()
    data_file = Path("data/example_data.csv")
    loader.load_csv(data_file)
    
    strategies = {
        'Arbitrage': ArbitrageStrategy(),
        'Market Making': MarketMakingStrategy(),
        'Behavioral': BehavioralEdgeStrategy()
    }
    
    config = BacktestConfig(
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now(),
        initial_balance=Decimal("10000")
    )
    
    results_by_strategy = {}
    
    for name, strategy in strategies.items():
        print(f"Testing {name}...")
        engine = BacktestEngine(
            data_loader=loader,
            strategies=[strategy],
            config=config
        )
        results = engine.run()
        metrics = calculate_metrics(results)
        results_by_strategy[name] = metrics
    
    # Compare results
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON")
    print("=" * 60)
    print(f"{'Strategy':<20} {'Return':<10} {'Sharpe':<10} {'Win Rate':<10}")
    print("-" * 60)
    
    for name, metrics in results_by_strategy.items():
        print(f"{name:<20} {metrics.total_return:>8.2%} {metrics.sharpe_ratio:>9.2f} {metrics.win_rate:>9.2%}")
    
    # Find best strategy
    best = max(results_by_strategy.items(), key=lambda x: x[1].sharpe_ratio)
    print(f"\nBest Strategy (by Sharpe): {best[0]}")


def run_parameter_optimization():
    """Optimize strategy parameters."""
    print("\nExample 3: Parameter Optimization\n")
    
    loader = HistoricalDataLoader()
    data_file = Path("data/example_data.csv")
    loader.load_csv(data_file)
    
    config = BacktestConfig(
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now(),
        initial_balance=Decimal("10000")
    )
    
    # Test different arbitrage thresholds
    thresholds = [0.02, 0.05, 0.08, 0.10]
    results = []
    
    print("Testing different spread thresholds...")
    for threshold in thresholds:
        # Create strategy with custom threshold
        strategy = ArbitrageStrategy()
        # Note: In real implementation, you'd pass min_spread as parameter
        # strategy = ArbitrageStrategy(min_spread=Decimal(str(threshold)))
        
        engine = BacktestEngine(
            data_loader=loader,
            strategies=[strategy],
            config=config
        )
        
        result = engine.run()
        metrics = calculate_metrics(result)
        results.append((threshold, metrics))
        
        print(f"Threshold {threshold:.2%}: Return={metrics.total_return:.2%}, "
              f"Sharpe={metrics.sharpe_ratio:.2f}, Trades={metrics.total_trades}")
    
    # Find optimal
    best = max(results, key=lambda x: x[1].sharpe_ratio)
    print(f"\nOptimal Threshold: {best[0]:.2%}")
    print(f"Best Sharpe Ratio: {best[1].sharpe_ratio:.2f}")


def run_walk_forward_analysis():
    """Walk-forward analysis for robustness testing."""
    print("\nExample 4: Walk-Forward Analysis\n")
    
    loader = HistoricalDataLoader()
    data_file = Path("data/example_data.csv")
    loader.load_csv(data_file)
    
    # Split data into in-sample (training) and out-of-sample (testing)
    end_date = datetime.now()
    mid_date = end_date - timedelta(days=15)
    start_date = end_date - timedelta(days=30)
    
    strategy = ArbitrageStrategy()
    
    # In-sample test
    print("In-Sample Period (Training)...")
    train_config = BacktestConfig(
        start_date=start_date,
        end_date=mid_date,
        initial_balance=Decimal("10000")
    )
    
    train_engine = BacktestEngine(
        data_loader=loader,
        strategies=[strategy],
        config=train_config
    )
    train_results = train_engine.run()
    train_metrics = calculate_metrics(train_results)
    
    # Out-of-sample test
    print("\nOut-of-Sample Period (Testing)...")
    test_config = BacktestConfig(
        start_date=mid_date,
        end_date=end_date,
        initial_balance=Decimal("10000")
    )
    
    test_engine = BacktestEngine(
        data_loader=loader,
        strategies=[strategy],
        config=test_config
    )
    test_results = test_engine.run()
    test_metrics = calculate_metrics(test_results)
    
    # Compare
    print("\n" + "=" * 60)
    print("WALK-FORWARD ANALYSIS RESULTS")
    print("=" * 60)
    print(f"{'Metric':<25} {'In-Sample':<15} {'Out-of-Sample':<15}")
    print("-" * 60)
    print(f"{'Total Return':<25} {train_metrics.total_return:>13.2%} {test_metrics.total_return:>14.2%}")
    print(f"{'Sharpe Ratio':<25} {train_metrics.sharpe_ratio:>13.2f} {test_metrics.sharpe_ratio:>14.2f}")
    print(f"{'Win Rate':<25} {train_metrics.win_rate:>13.2%} {test_metrics.win_rate:>14.2%}")
    print(f"{'Max Drawdown':<25} {train_metrics.max_drawdown:>13.2%} {test_metrics.max_drawdown:>14.2%}")
    
    # Performance ratio
    if train_metrics.sharpe_ratio > 0:
        ratio = float(test_metrics.sharpe_ratio) / float(train_metrics.sharpe_ratio)
        print(f"\nOut-of-Sample / In-Sample Sharpe Ratio: {ratio:.2f}")
        
        if ratio > 0.8:
            print("✓ Strategy generalizes well!")
        else:
            print("⚠ Warning: Performance degradation in out-of-sample period")


if __name__ == '__main__':
    # Run examples
    try:
        run_simple_backtest()
        run_multi_strategy_comparison()
        run_parameter_optimization()
        run_walk_forward_analysis()
        
        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
