"""
PR3DICT: VWAP Strategy Integration Example

Shows how to integrate VWAP validation into existing trading strategies.
Demonstrates signal enrichment, position sizing, and order splitting.
"""

import asyncio
import logging
from decimal import Decimal
from typing import Optional

from src.platforms.polymarket import PolymarketPlatform
from src.platforms.base import OrderSide, Market, OrderBook
from src.execution.vwap_integration import (
    VWAPTradingGate,
    VWAPEnrichedSignal,
    StrategyVWAPIntegration
)
from src.strategies.base import BaseStrategy
from src.data.vwap import VWAPCalculator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VWAPAwareArbitrageStrategy(BaseStrategy, StrategyVWAPIntegration):
    """
    Example arbitrage strategy with VWAP integration.
    
    Demonstrates:
    - Signal validation before execution
    - Position size adjustment based on liquidity
    - Order splitting for large trades
    - Profit calculation after slippage
    """
    
    def __init__(self, platform, vwap_gate: VWAPTradingGate):
        BaseStrategy.__init__(self, platform)
        StrategyVWAPIntegration.__init__(self, vwap_gate)
        
        self.min_arb_profit_pct = Decimal("1.0")  # 1% minimum arb profit
        self.min_profit_after_slippage_pct = Decimal("0.5")  # 0.5% after slippage
    
    async def generate_signals(self, markets):
        """
        Generate arbitrage signals with VWAP validation.
        
        Traditional arb strategy would look for YES + NO < 1.00.
        We enhance it by validating execution quality.
        """
        signals = []
        
        for market in markets:
            # Basic arbitrage check
            if not market.arbitrage_opportunity:
                continue
            
            potential_profit = Decimal("1.0") - (market.yes_price + market.no_price)
            profit_pct = potential_profit / (market.yes_price + market.no_price) * 100
            
            if profit_pct < self.min_arb_profit_pct:
                continue
            
            logger.info(
                f"Potential arb: {market.ticker} - "
                f"YES=${market.yes_price:.4f} + NO=${market.no_price:.4f} = "
                f"${market.yes_price + market.no_price:.4f} "
                f"(profit: {profit_pct:.2f}%)"
            )
            
            # Fetch order book for VWAP analysis
            orderbook = await self.platform.get_orderbook(market.id)
            
            # Determine position size based on liquidity
            target_capital = Decimal("100")  # $100 per side
            
            yes_quantity = self.adjust_position_size_for_liquidity(
                market_id=market.id,
                side=OrderSide.YES,
                target_capital=target_capital,
                orderbook=orderbook
            )
            
            no_quantity = self.adjust_position_size_for_liquidity(
                market_id=market.id,
                side=OrderSide.NO,
                target_capital=target_capital,
                orderbook=orderbook
            )
            
            # Use smaller quantity to balance
            quantity = min(yes_quantity, no_quantity)
            
            if quantity < 10:
                logger.warning(f"Insufficient liquidity for {market.ticker}")
                continue
            
            # Enrich YES signal with VWAP
            yes_signal = self.enrich_signal_with_vwap(
                market_id=market.id,
                side=OrderSide.YES,
                quantity=quantity,
                signal_price=market.yes_price,
                orderbook=orderbook
            )
            
            # Enrich NO signal with VWAP
            no_signal = self.enrich_signal_with_vwap(
                market_id=market.id,
                side=OrderSide.NO,
                quantity=quantity,
                signal_price=market.no_price,
                orderbook=orderbook
            )
            
            if not yes_signal or not no_signal:
                logger.warning(f"VWAP validation failed for {market.ticker}")
                continue
            
            # Calculate actual profit after slippage
            total_cost = yes_signal.vwap_result.vwap_price + no_signal.vwap_result.vwap_price
            actual_profit = Decimal("1.0") - total_cost
            actual_profit_pct = actual_profit / total_cost * 100
            
            logger.info(
                f"After VWAP: {market.ticker} - "
                f"YES VWAP=${yes_signal.vwap_result.vwap_price:.4f}, "
                f"NO VWAP=${no_signal.vwap_result.vwap_price:.4f}, "
                f"total=${total_cost:.4f}, "
                f"profit={actual_profit_pct:.2f}%"
            )
            
            # Check if still profitable after slippage
            if actual_profit_pct < self.min_profit_after_slippage_pct:
                logger.warning(
                    f"Arb not profitable after slippage: {market.ticker} "
                    f"(need {self.min_profit_after_slippage_pct}%, got {actual_profit_pct:.2f}%)"
                )
                continue
            
            # Signal passed all checks!
            signals.append({
                'market': market,
                'quantity': quantity,
                'yes_signal': yes_signal,
                'no_signal': no_signal,
                'expected_profit_pct': actual_profit_pct,
                'total_capital': total_cost * quantity
            })
            
            logger.info(
                f"✓ ARB SIGNAL: {market.ticker} - "
                f"{quantity} contracts, "
                f"capital=${total_cost * quantity:.2f}, "
                f"expected profit={actual_profit_pct:.2f}%"
            )
        
        return signals
    
    async def execute_signal(self, signal: dict):
        """
        Execute arbitrage signal with order splitting if needed.
        
        For large orders, split into smaller chunks to minimize impact.
        """
        market = signal['market']
        quantity = signal['quantity']
        yes_signal = signal['yes_signal']
        no_signal = signal['no_signal']
        
        # Check if order should be split
        if quantity > 500:  # Split orders larger than 500 contracts
            logger.info(f"Large order detected: {quantity} contracts. Checking split...")
            
            # Get order book again (might have changed)
            orderbook = await self.platform.get_orderbook(market.id)
            
            # Get split suggestions
            yes_chunks = self.split_large_order(
                market_id=market.id,
                side=OrderSide.YES,
                quantity=quantity,
                orderbook=orderbook,
                max_chunks=3
            )
            
            no_chunks = self.split_large_order(
                market_id=market.id,
                side=OrderSide.NO,
                quantity=quantity,
                orderbook=orderbook,
                max_chunks=3
            )
            
            # Execute in chunks
            logger.info(f"Splitting YES into {len(yes_chunks)} chunks: {yes_chunks}")
            logger.info(f"Splitting NO into {len(no_chunks)} chunks: {no_chunks}")
            
            # In real implementation, you would execute these chunks with delays
            # For now, just log the plan
            for i, (yes_chunk, no_chunk) in enumerate(zip(yes_chunks, no_chunks)):
                logger.info(
                    f"  Chunk {i+1}: YES {yes_chunk} contracts, NO {no_chunk} contracts"
                )
        else:
            # Execute full order
            logger.info(
                f"Executing full order: {quantity} contracts "
                f"(YES @ ${yes_signal.vwap_result.vwap_price:.4f}, "
                f"NO @ ${no_signal.vwap_result.vwap_price:.4f})"
            )
            
            # In real implementation, you would place orders here
            # await self.platform.place_order(...)


class VWAPAwareMomentumStrategy(BaseStrategy, StrategyVWAPIntegration):
    """
    Example momentum strategy with VWAP validation.
    
    Only enters positions where execution quality is acceptable.
    """
    
    def __init__(self, platform, vwap_gate: VWAPTradingGate):
        BaseStrategy.__init__(self, platform)
        StrategyVWAPIntegration.__init__(self, vwap_gate)
        
        self.momentum_threshold = Decimal("0.10")  # 10 cent move
        self.min_quality_score = Decimal("70")  # Minimum execution quality score
    
    async def generate_signals(self, markets):
        """Generate momentum signals with quality filtering."""
        signals = []
        
        for market in markets:
            # Simple momentum: price > 0.60 (strong YES momentum)
            if market.yes_price < Decimal("0.60"):
                continue
            
            logger.info(f"Momentum signal: {market.ticker} @ ${market.yes_price:.2f}")
            
            # Get order book
            orderbook = await self.platform.get_orderbook(market.id)
            
            # Size position conservatively
            quantity = self.adjust_position_size_for_liquidity(
                market_id=market.id,
                side=OrderSide.YES,
                target_capital=Decimal("50"),
                orderbook=orderbook
            )
            
            # Enrich with VWAP
            enriched = self.enrich_signal_with_vwap(
                market_id=market.id,
                side=OrderSide.YES,
                quantity=quantity,
                signal_price=market.yes_price,
                orderbook=orderbook
            )
            
            if not enriched:
                logger.warning(f"VWAP validation failed for {market.ticker}")
                continue
            
            # Check quality score
            if enriched.quality_score < self.min_quality_score:
                logger.warning(
                    f"Quality score too low: {enriched.quality_score:.1f} "
                    f"(need {self.min_quality_score})"
                )
                continue
            
            # Check profitability after slippage
            if not enriched.is_profitable_after_slippage:
                logger.warning(f"Not profitable after slippage: {market.ticker}")
                continue
            
            signals.append({
                'market': market,
                'quantity': quantity,
                'enriched_signal': enriched
            })
            
            logger.info(
                f"✓ MOMENTUM SIGNAL: {market.ticker} - "
                f"{quantity} contracts @ ${enriched.vwap_result.vwap_price:.4f}, "
                f"quality={enriched.quality_score:.1f}"
            )
        
        return signals


async def main():
    """
    Example: Run VWAP-aware strategies.
    """
    # Initialize platform
    platform = PolymarketPlatform()
    
    try:
        # Connect
        connected = await platform.connect()
        if not connected:
            logger.error("Failed to connect to Polymarket")
            return
        
        # Create VWAP gate with conservative settings
        vwap_gate = VWAPTradingGate(
            max_slippage_pct=Decimal("2.0"),  # Max 2% slippage
            min_liquidity_contracts=500,  # Require 500+ depth
            max_spread_bps=300,  # Max 3% spread
            enable_position_adjustment=True  # Auto-adjust sizes
        )
        
        # Create strategies
        arb_strategy = VWAPAwareArbitrageStrategy(platform, vwap_gate)
        momentum_strategy = VWAPAwareMomentumStrategy(platform, vwap_gate)
        
        # Fetch markets
        logger.info("Fetching markets...")
        markets = await platform.get_markets(limit=20)
        
        # Generate arbitrage signals
        logger.info("\n" + "="*80)
        logger.info("ARBITRAGE STRATEGY")
        logger.info("="*80)
        arb_signals = await arb_strategy.generate_signals(markets)
        logger.info(f"Generated {len(arb_signals)} arbitrage signals")
        
        # Generate momentum signals
        logger.info("\n" + "="*80)
        logger.info("MOMENTUM STRATEGY")
        logger.info("="*80)
        momentum_signals = await momentum_strategy.generate_signals(markets)
        logger.info(f"Generated {len(momentum_signals)} momentum signals")
        
        # Print gate statistics
        logger.info("\n" + "="*80)
        logger.info("VWAP GATE STATISTICS")
        logger.info("="*80)
        stats = vwap_gate.get_statistics()
        logger.info(f"Signals processed: {stats['signals_processed']}")
        logger.info(f"Signals blocked: {stats['signals_blocked']} ({stats['block_rate_pct']:.1f}%)")
        logger.info(f"Signals adjusted: {stats['signals_adjusted']} ({stats['adjustment_rate_pct']:.1f}%)")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        await platform.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
