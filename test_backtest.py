"""
Quick test script for backtesting framework.

Generates sample data and runs a basic backtest to verify everything works.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.backtest.data import HistoricalDataLoader
from src.backtest.engine import BacktestEngine, BacktestConfig
from src.backtest.report import generate_report
from src.strategies.arbitrage import ArbitrageStrategy
from src.risk.manager import RiskManager, RiskConfig
from decimal import Decimal


def test_backtesting_framework():
    """Test the complete backtesting pipeline."""
    print("=" * 60)
    print("BACKTESTING FRAMEWORK TEST")
    print("=" * 60)
    
    # 1. Generate sample data
    print("\n1. Generating sample data...")
    data_file = Path("data/test_sample.csv")
    data_file.parent.mkdir(parents=True, exist_ok=True)
    
    loader = HistoricalDataLoader()
    loader.generate_sample_data(
        filepath=data_file,
        num_markets=3,
        days=7
    )
    print(f"   ✓ Generated {data_file}")
    
    # 2. Load data
    print("\n2. Loading historical data...")
    loader.load_csv(data_file)
    print(f"   ✓ Loaded {len(loader.snapshots)} snapshots")
    
    # 3. Setup strategy
    print("\n3. Initializing arbitrage strategy...")
    strategy = ArbitrageStrategy()
    print(f"   ✓ Strategy: {strategy.name}")
    
    # 4. Setup risk manager
    print("\n4. Setting up risk manager...")
    risk_config = RiskConfig(
        max_position_size=Decimal("100"),
        daily_loss_limit=Decimal("500"),
        max_portfolio_heat=0.20
    )
    risk_manager = RiskManager(risk_config)
    print("   ✓ Risk management enabled")
    
    # 5. Configure backtest
    print("\n5. Configuring backtest...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_balance=Decimal("10000"),
        max_positions=5,
        commission_rate=Decimal("0.01"),
        slippage_bps=Decimal("5")
    )
    print(f"   ✓ Period: {start_date.date()} to {end_date.date()}")
    print(f"   ✓ Initial balance: ${config.initial_balance}")
    
    # 6. Run backtest
    print("\n6. Running backtest...")
    engine = BacktestEngine(
        data_loader=loader,
        strategies=[strategy],
        config=config,
        risk_manager=risk_manager
    )
    
    results = engine.run()
    
    # 7. Generate report
    print("\n7. Generating report...")
    output_dir = Path("backtest_reports")
    report = generate_report(results, output_dir)
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY ✓")
    print("=" * 60)
    print(f"\nReports saved to: {output_dir}/")
    print(f"Sample data: {data_file}")
    print("\nTo run a real backtest:")
    print(f"  python -m src.backtest.run --strategy arbitrage --data {data_file} \\")
    print(f"    --start {start_date.strftime('%Y-%m-%d')} --end {end_date.strftime('%Y-%m-%d')}")
    
    return True


if __name__ == '__main__':
    try:
        test_backtesting_framework()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
