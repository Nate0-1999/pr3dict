"""
Historical Data Loading for Backtesting

Loads market snapshots from CSV or API sources and replays them chronologically
to simulate real-time market conditions without look-ahead bias.
"""
import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Iterator
from decimal import Decimal
from pathlib import Path

from ..platforms.base import Market, OrderBook

logger = logging.getLogger(__name__)


@dataclass
class MarketSnapshot:
    """Point-in-time snapshot of a market's state."""
    timestamp: datetime
    market_id: str
    ticker: str
    title: str
    yes_price: Decimal
    no_price: Decimal
    volume: Decimal
    liquidity: Decimal
    close_time: datetime
    resolved: bool
    platform: str
    
    # Optional orderbook data
    bids: List[tuple[Decimal, int]] = field(default_factory=list)
    asks: List[tuple[Decimal, int]] = field(default_factory=list)
    
    def to_market(self) -> Market:
        """Convert snapshot to Market object."""
        return Market(
            id=self.market_id,
            ticker=self.ticker,
            title=self.title,
            description="",  # Not stored in snapshots
            yes_price=self.yes_price,
            no_price=self.no_price,
            volume=self.volume,
            liquidity=self.liquidity,
            close_time=self.close_time,
            resolved=self.resolved,
            platform=self.platform
        )
    
    def to_orderbook(self) -> OrderBook:
        """Convert snapshot to OrderBook object."""
        return OrderBook(
            market_id=self.market_id,
            bids=self.bids,
            asks=self.asks,
            timestamp=self.timestamp
        )


class HistoricalDataLoader:
    """
    Loads and streams historical market data for backtesting.
    
    Supports:
    - CSV files with timestamped snapshots
    - API-fetched historical data
    - Chronological replay with no look-ahead bias
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path.home() / ".openclaw/workspace/pr3dict/data/historical"
        self.snapshots: List[MarketSnapshot] = []
        self._loaded = False
    
    def load_csv(self, filepath: Path) -> None:
        """
        Load market snapshots from CSV file.
        
        Expected CSV format:
        timestamp,market_id,ticker,title,yes_price,no_price,volume,liquidity,close_time,resolved,platform
        """
        logger.info(f"Loading historical data from {filepath}")
        
        if not filepath.exists():
            raise FileNotFoundError(f"Historical data file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    snapshot = MarketSnapshot(
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        market_id=row['market_id'],
                        ticker=row['ticker'],
                        title=row['title'],
                        yes_price=Decimal(row['yes_price']),
                        no_price=Decimal(row['no_price']),
                        volume=Decimal(row.get('volume', '0')),
                        liquidity=Decimal(row.get('liquidity', '0')),
                        close_time=datetime.fromisoformat(row['close_time']),
                        resolved=row.get('resolved', 'false').lower() == 'true',
                        platform=row['platform']
                    )
                    self.snapshots.append(snapshot)
                except Exception as e:
                    logger.warning(f"Skipping invalid row: {e}")
        
        # Sort by timestamp to ensure chronological replay
        self.snapshots.sort(key=lambda s: s.timestamp)
        self._loaded = True
        
        logger.info(f"Loaded {len(self.snapshots)} market snapshots")
        if self.snapshots:
            logger.info(f"Date range: {self.snapshots[0].timestamp} to {self.snapshots[-1].timestamp}")
    
    def load_from_directory(self, start_date: datetime, end_date: datetime, 
                           platforms: Optional[List[str]] = None) -> None:
        """
        Load all CSV files in data directory within date range.
        
        Args:
            start_date: Begin date for backtest
            end_date: End date for backtest
            platforms: Optional filter for specific platforms
        """
        if not self.data_dir.exists():
            logger.warning(f"Data directory not found: {self.data_dir}")
            return
        
        csv_files = list(self.data_dir.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files in {self.data_dir}")
        
        for csv_file in csv_files:
            # Optional platform filtering based on filename
            if platforms:
                if not any(p in csv_file.stem for p in platforms):
                    continue
            
            self.load_csv(csv_file)
        
        # Filter to date range
        self.snapshots = [
            s for s in self.snapshots
            if start_date <= s.timestamp <= end_date
        ]
        
        logger.info(f"Filtered to {len(self.snapshots)} snapshots in date range")
    
    def generate_sample_data(self, filepath: Path, num_markets: int = 5, 
                            days: int = 30) -> None:
        """
        Generate synthetic historical data for testing.
        
        Creates realistic-looking market data with price movements,
        volume, and random events.
        """
        import random
        from datetime import timedelta
        
        logger.info(f"Generating {num_markets} sample markets over {days} days")
        
        markets = []
        start_date = datetime.now() - timedelta(days=days)
        
        for i in range(num_markets):
            market_id = f"SAMPLE-{i+1}"
            ticker = f"SAMPLE{i+1}"
            title = f"Sample Market {i+1}: Will event {i+1} occur?"
            
            # Generate price walk
            current_price = Decimal(str(random.uniform(0.3, 0.7)))
            
            for day in range(days):
                for hour in range(0, 24, 4):  # Every 4 hours
                    timestamp = start_date + timedelta(days=day, hours=hour)
                    
                    # Random walk with mean reversion
                    change = Decimal(str(random.gauss(0, 0.02)))
                    current_price = max(Decimal("0.01"), min(Decimal("0.99"), 
                                       current_price + change))
                    
                    snapshot = MarketSnapshot(
                        timestamp=timestamp,
                        market_id=market_id,
                        ticker=ticker,
                        title=title,
                        yes_price=current_price,
                        no_price=Decimal("1.0") - current_price - Decimal("0.01"),  # Small spread
                        volume=Decimal(str(random.randint(1000, 50000))),
                        liquidity=Decimal(str(random.randint(5000, 100000))),
                        close_time=start_date + timedelta(days=days+1),
                        resolved=False,
                        platform="kalshi"
                    )
                    markets.append(snapshot)
        
        # Write to CSV
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'market_id', 'ticker', 'title', 
                           'yes_price', 'no_price', 'volume', 'liquidity', 
                           'close_time', 'resolved', 'platform'])
            
            for snapshot in sorted(markets, key=lambda s: s.timestamp):
                writer.writerow([
                    snapshot.timestamp.isoformat(),
                    snapshot.market_id,
                    snapshot.ticker,
                    snapshot.title,
                    str(snapshot.yes_price),
                    str(snapshot.no_price),
                    str(snapshot.volume),
                    str(snapshot.liquidity),
                    snapshot.close_time.isoformat(),
                    str(snapshot.resolved).lower(),
                    snapshot.platform
                ])
        
        logger.info(f"Generated {len(markets)} snapshots to {filepath}")
    
    def replay(self, start_date: datetime, end_date: datetime) -> Iterator[tuple[datetime, List[Market]]]:
        """
        Replay historical data chronologically.
        
        Yields batches of markets grouped by timestamp, simulating
        real-time market scans.
        
        Yields:
            (timestamp, markets): Tuple of timestamp and available markets at that time
        """
        if not self._loaded:
            raise RuntimeError("No data loaded. Call load_csv() or load_from_directory() first.")
        
        # Group snapshots by timestamp
        timestamp_groups: Dict[datetime, List[MarketSnapshot]] = {}
        for snapshot in self.snapshots:
            if start_date <= snapshot.timestamp <= end_date:
                if snapshot.timestamp not in timestamp_groups:
                    timestamp_groups[snapshot.timestamp] = []
                timestamp_groups[snapshot.timestamp].append(snapshot)
        
        # Yield in chronological order
        for timestamp in sorted(timestamp_groups.keys()):
            snapshots = timestamp_groups[timestamp]
            markets = [s.to_market() for s in snapshots]
            yield timestamp, markets
    
    def get_market_at_time(self, market_id: str, timestamp: datetime) -> Optional[Market]:
        """
        Get the most recent snapshot of a market at or before a given time.
        
        Used for filling orders at realistic prices.
        """
        relevant_snapshots = [
            s for s in self.snapshots
            if s.market_id == market_id and s.timestamp <= timestamp
        ]
        
        if not relevant_snapshots:
            return None
        
        # Get most recent
        latest = max(relevant_snapshots, key=lambda s: s.timestamp)
        return latest.to_market()
