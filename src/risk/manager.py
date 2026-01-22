"""
PR3DICT: Risk Manager

Centralized risk control for prediction market trading.
Implements Kelly Criterion, portfolio heat, and daily limits.
"""
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """Risk management configuration."""
    # Daily limits
    daily_loss_limit: Decimal = Decimal("500")
    max_trades_per_day: int = 50
    
    # Position sizing
    max_position_size: Decimal = Decimal("100")  # Max contracts per position
    max_position_value: Decimal = Decimal("500")  # Max $ per position
    max_portfolio_heat: float = 0.25  # Max 25% of account in open positions
    
    # Risk per trade
    base_risk_pct: float = 0.02  # 2% per trade
    max_risk_pct: float = 0.05  # 5% max even with strong signals
    
    # Consecutive loss protection
    max_consecutive_losses: int = 3
    loss_reduction_factor: float = 0.5  # Reduce size by 50% after max losses


@dataclass
class RiskState:
    """Runtime risk tracking state."""
    daily_pnl: Decimal = Decimal("0")
    trades_today: int = 0
    consecutive_losses: int = 0
    open_position_value: Decimal = Decimal("0")
    account_value: Decimal = Decimal("1000")  # Will be updated from platforms
    last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RiskManager:
    """
    Centralized risk management.
    
    Responsibilities:
    - Trade gating (check_trade_allowed)
    - Position sizing (Kelly Criterion)
    - Portfolio heat tracking
    - Daily loss limits
    - Consecutive loss protection
    """
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.state = RiskState()
    
    def check_trade_allowed(self) -> Tuple[bool, str]:
        """
        Check if a new trade is allowed.
        
        Returns:
            (allowed: bool, reason: str)
        """
        # Check if new day - reset daily counters
        self._check_daily_reset()
        
        # Daily loss limit
        if self.state.daily_pnl <= -self.config.daily_loss_limit:
            return False, "DAILY_LOSS_LIMIT_REACHED"
        
        # Max trades per day
        if self.state.trades_today >= self.config.max_trades_per_day:
            return False, "MAX_TRADES_REACHED"
        
        # Portfolio heat check
        if self.state.account_value > 0:
            heat = float(self.state.open_position_value / self.state.account_value)
            if heat >= self.config.max_portfolio_heat:
                return False, "PORTFOLIO_HEAT_EXCEEDED"
        
        # Consecutive loss protection
        if self.state.consecutive_losses >= self.config.max_consecutive_losses:
            # Allow but at reduced size (handled in position sizing)
            logger.warning(f"Consecutive losses: {self.state.consecutive_losses}")
        
        return True, "OK"
    
    def validate_position_size(self, size: int, price: Decimal) -> bool:
        """
        Validate a proposed position size.
        
        Args:
            size: Number of contracts
            price: Price per contract
            
        Returns:
            True if size is within limits
        """
        # Max contracts check
        if size > self.config.max_position_size:
            return False
        
        # Max value check
        position_value = Decimal(str(size)) * price
        if position_value > self.config.max_position_value:
            return False
        
        # Would exceed portfolio heat?
        new_heat = (self.state.open_position_value + position_value) / self.state.account_value
        if new_heat > Decimal(str(self.config.max_portfolio_heat)):
            return False
        
        return True
    
    def calculate_position_size(self,
                                account_value: Decimal,
                                entry_price: Decimal,
                                signal_strength: float = 1.0,
                                win_rate: float = 0.55,
                                win_loss_ratio: float = 1.5) -> int:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Kelly formula: K% = W - (1-W)/R
        Where W = win rate, R = win/loss ratio
        
        We use fractional Kelly (25%) for safety.
        """
        # Base Kelly calculation
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Cap at max risk and use fractional Kelly
        fractional_kelly = kelly_pct * 0.25  # 25% of Kelly
        risk_pct = min(fractional_kelly, self.config.max_risk_pct)
        risk_pct = max(risk_pct, 0.005)  # Minimum 0.5%
        
        # Adjust for signal strength
        adjusted_risk = risk_pct * signal_strength
        
        # Adjust for consecutive losses
        if self.state.consecutive_losses >= self.config.max_consecutive_losses:
            adjusted_risk *= self.config.loss_reduction_factor
        
        # Calculate dollar risk
        risk_amount = account_value * Decimal(str(adjusted_risk))
        
        # For binary contracts, max loss is entry price
        if entry_price > 0:
            contracts = int(risk_amount / entry_price)
            # Apply position limits
            contracts = min(contracts, int(self.config.max_position_size))
            return max(1, contracts)
        
        return 1
    
    def record_trade(self, pnl: Decimal) -> None:
        """Record a completed trade result."""
        self.state.daily_pnl += pnl
        self.state.trades_today += 1
        
        if pnl < 0:
            self.state.consecutive_losses += 1
            logger.info(f"Loss recorded: {pnl}, consecutive: {self.state.consecutive_losses}")
        else:
            self.state.consecutive_losses = 0
            logger.info(f"Win recorded: {pnl}")
    
    def update_account(self, account_value: Decimal, open_position_value: Decimal) -> None:
        """Update account state from platforms."""
        self.state.account_value = account_value
        self.state.open_position_value = open_position_value
    
    def _check_daily_reset(self) -> None:
        """Reset daily counters if new trading day."""
        now = datetime.now(timezone.utc)
        if now.date() > self.state.last_reset.date():
            logger.info("New trading day - resetting daily counters")
            self.state.daily_pnl = Decimal("0")
            self.state.trades_today = 0
            self.state.last_reset = now
    
    def get_status(self) -> dict:
        """Get current risk status."""
        return {
            "daily_pnl": float(self.state.daily_pnl),
            "trades_today": self.state.trades_today,
            "consecutive_losses": self.state.consecutive_losses,
            "portfolio_heat": float(self.state.open_position_value / self.state.account_value) 
                             if self.state.account_value > 0 else 0,
            "account_value": float(self.state.account_value),
            "trade_allowed": self.check_trade_allowed()[0]
        }
