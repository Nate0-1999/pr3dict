"""
PR3DICT: Polygon Optimization Module

Polygon-specific optimizations for arbitrage execution:
- Batch transaction handling
- Gas price management
- RPC endpoint selection and failover
- Retry logic with exponential backoff
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional, Callable, Any
from enum import Enum
import random

logger = logging.getLogger(__name__)


class RPCEndpointStatus(Enum):
    """Status of RPC endpoint."""
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass
class RPCEndpoint:
    """Represents a Polygon RPC endpoint."""
    url: str
    status: RPCEndpointStatus = RPCEndpointStatus.ACTIVE
    failure_count: int = 0
    last_success: Optional[float] = None
    last_failure: Optional[float] = None
    avg_latency_ms: float = 0.0
    
    @property
    def health_score(self) -> float:
        """Calculate health score (0-1, higher is better)."""
        if self.status == RPCEndpointStatus.FAILED:
            return 0.0
        
        # Base score on status
        base_score = 1.0 if self.status == RPCEndpointStatus.ACTIVE else 0.5
        
        # Penalize for failures
        failure_penalty = min(self.failure_count * 0.1, 0.5)
        
        # Penalize for high latency (assume 100ms is baseline)
        latency_penalty = min(self.avg_latency_ms / 1000.0, 0.3)
        
        score = base_score - failure_penalty - latency_penalty
        return max(score, 0.0)


class RPCLoadBalancer:
    """
    Manages Polygon RPC endpoints with automatic failover.
    
    Features:
    - Health monitoring
    - Automatic failover
    - Latency tracking
    - Load distribution
    """
    
    def __init__(self, endpoints: List[str]):
        self.endpoints = [RPCEndpoint(url=url) for url in endpoints]
        self._current_index = 0
    
    def get_best_endpoint(self) -> RPCEndpoint:
        """
        Select best available endpoint based on health scores.
        """
        # Sort by health score
        available = [ep for ep in self.endpoints if ep.status != RPCEndpointStatus.FAILED]
        
        if not available:
            # All failed, try primary
            logger.warning("All RPC endpoints failed, attempting primary")
            return self.endpoints[0]
        
        # Return best endpoint
        available.sort(key=lambda ep: ep.health_score, reverse=True)
        return available[0]
    
    def get_next_endpoint(self) -> RPCEndpoint:
        """Get next endpoint in round-robin fashion."""
        endpoint = self.endpoints[self._current_index]
        self._current_index = (self._current_index + 1) % len(self.endpoints)
        return endpoint
    
    def record_success(self, endpoint: RPCEndpoint, latency_ms: float) -> None:
        """Record successful request."""
        endpoint.last_success = time.time()
        endpoint.failure_count = max(0, endpoint.failure_count - 1)  # Reduce failure count
        endpoint.status = RPCEndpointStatus.ACTIVE
        
        # Update average latency (exponential moving average)
        if endpoint.avg_latency_ms == 0:
            endpoint.avg_latency_ms = latency_ms
        else:
            endpoint.avg_latency_ms = (endpoint.avg_latency_ms * 0.7) + (latency_ms * 0.3)
    
    def record_failure(self, endpoint: RPCEndpoint) -> None:
        """Record failed request."""
        endpoint.last_failure = time.time()
        endpoint.failure_count += 1
        
        # Mark as degraded or failed based on failure count
        if endpoint.failure_count >= 5:
            endpoint.status = RPCEndpointStatus.FAILED
            logger.error(f"RPC endpoint marked as failed: {endpoint.url}")
        elif endpoint.failure_count >= 3:
            endpoint.status = RPCEndpointStatus.DEGRADED
            logger.warning(f"RPC endpoint degraded: {endpoint.url}")
    
    def get_status(self) -> dict:
        """Get status of all endpoints."""
        return {
            "endpoints": [
                {
                    "url": ep.url,
                    "status": ep.status.value,
                    "health_score": round(ep.health_score, 2),
                    "failure_count": ep.failure_count,
                    "avg_latency_ms": round(ep.avg_latency_ms, 1),
                }
                for ep in self.endpoints
            ]
        }


class GasPriceManager:
    """
    Manages gas price optimization for Polygon.
    
    Polygon characteristics:
    - Low base fees (30-100 gwei typical)
    - Fast blocks (~2 seconds)
    - EIP-1559 support
    """
    
    def __init__(self,
                 max_gas_price_gwei: Decimal = Decimal("500"),
                 target_block_time_ms: int = 2000):
        self.max_gas_price_gwei = max_gas_price_gwei
        self.target_block_time_ms = target_block_time_ms
        
        # Price history for trend analysis
        self._price_history: List[Tuple[float, Decimal]] = []  # (timestamp, price_gwei)
    
    def record_gas_price(self, price_gwei: Decimal) -> None:
        """Record observed gas price."""
        self._price_history.append((time.time(), price_gwei))
        
        # Keep last 100 samples
        if len(self._price_history) > 100:
            self._price_history = self._price_history[-100:]
    
    def get_recommended_gas_price(self, urgency: str = "high") -> Decimal:
        """
        Get recommended gas price based on current conditions.
        
        Args:
            urgency: "low", "medium", "high" (for arbitrage, always high)
            
        Returns:
            Recommended gas price in gwei
        """
        if not self._price_history:
            # No data, use conservative default
            return Decimal("100")
        
        # Get recent prices
        recent_prices = [price for _, price in self._price_history[-10:]]
        avg_price = sum(recent_prices) / len(recent_prices)
        max_recent = max(recent_prices)
        
        # Urgency multipliers
        multipliers = {
            "low": Decimal("1.0"),
            "medium": Decimal("1.1"),
            "high": Decimal("1.2"),  # 20% above average for fast inclusion
        }
        
        recommended = avg_price * multipliers.get(urgency, Decimal("1.2"))
        
        # Cap at max
        recommended = min(recommended, self.max_gas_price_gwei)
        
        # Ensure minimum (Polygon minimum is ~30 gwei)
        recommended = max(recommended, Decimal("30"))
        
        logger.debug(f"Gas price recommendation: {recommended:.1f} gwei (urgency: {urgency})")
        return recommended
    
    def estimate_cost(self, gas_units: int, price_gwei: Decimal) -> Decimal:
        """
        Estimate transaction cost in MATIC.
        
        Args:
            gas_units: Estimated gas units for transaction
            price_gwei: Gas price in gwei
            
        Returns:
            Cost in MATIC
        """
        # Convert gwei to MATIC
        # 1 MATIC = 1e9 gwei
        cost_matic = (Decimal(str(gas_units)) * price_gwei) / Decimal("1e9")
        return cost_matic
    
    def is_price_acceptable(self, price_gwei: Decimal) -> bool:
        """Check if gas price is within acceptable range."""
        return price_gwei <= self.max_gas_price_gwei


class RetryStrategy:
    """
    Retry logic with exponential backoff for Polygon transactions.
    """
    
    def __init__(self,
                 max_retries: int = 3,
                 base_delay_ms: int = 50,
                 max_delay_ms: int = 500,
                 exponential_base: float = 2.0):
        self.max_retries = max_retries
        self.base_delay_ms = base_delay_ms
        self.max_delay_ms = max_delay_ms
        self.exponential_base = exponential_base
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for retry attempt.
        
        Uses exponential backoff with jitter.
        """
        # Exponential backoff
        delay_ms = min(
            self.base_delay_ms * (self.exponential_base ** attempt),
            self.max_delay_ms
        )
        
        # Add jitter (±20%)
        jitter = delay_ms * 0.2
        delay_ms += random.uniform(-jitter, jitter)
        
        return delay_ms / 1000.0  # Convert to seconds
    
    async def execute_with_retry(self,
                                 func: Callable,
                                 *args,
                                 **kwargs) -> Any:
        """
        Execute function with retry logic.
        
        Args:
            func: Async function to execute
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Result of function call
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                result = await func(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(f"Retry succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries - 1:
                    delay = self.calculate_delay(attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries} retry attempts failed")
        
        raise last_exception


class BatchTransactionManager:
    """
    Manages batch transaction submission for Polygon.
    
    Groups multiple operations into single transaction when possible
    to reduce gas costs and improve atomicity.
    """
    
    def __init__(self, max_batch_size: int = 10):
        self.max_batch_size = max_batch_size
        self._pending_operations: List[dict] = []
    
    def add_operation(self, operation: dict) -> None:
        """Add operation to pending batch."""
        self._pending_operations.append(operation)
    
    def get_batch(self, max_size: Optional[int] = None) -> List[dict]:
        """
        Get current batch of operations.
        
        Args:
            max_size: Maximum operations to include (default: all pending)
            
        Returns:
            List of operations
        """
        batch_size = min(
            max_size or self.max_batch_size,
            len(self._pending_operations)
        )
        
        batch = self._pending_operations[:batch_size]
        self._pending_operations = self._pending_operations[batch_size:]
        
        return batch
    
    def clear(self) -> None:
        """Clear all pending operations."""
        self._pending_operations.clear()
    
    @property
    def pending_count(self) -> int:
        """Number of pending operations."""
        return len(self._pending_operations)


class PolygonOptimizer:
    """
    Main optimization coordinator for Polygon execution.
    
    Combines:
    - RPC load balancing
    - Gas price management
    - Retry logic
    - Batch transaction handling
    """
    
    def __init__(self,
                 rpc_endpoints: List[str],
                 max_gas_price_gwei: Decimal = Decimal("500")):
        self.load_balancer = RPCLoadBalancer(rpc_endpoints)
        self.gas_manager = GasPriceManager(max_gas_price_gwei=max_gas_price_gwei)
        self.retry_strategy = RetryStrategy()
        self.batch_manager = BatchTransactionManager()
    
    async def execute_with_optimization(self,
                                       func: Callable,
                                       *args,
                                       **kwargs) -> Any:
        """
        Execute function with full optimization stack.
        
        Includes:
        - Best RPC endpoint selection
        - Retry logic
        - Latency tracking
        """
        endpoint = self.load_balancer.get_best_endpoint()
        
        start_time = time.time()
        
        try:
            result = await self.retry_strategy.execute_with_retry(
                func, *args, **kwargs
            )
            
            # Record success
            latency_ms = (time.time() - start_time) * 1000
            self.load_balancer.record_success(endpoint, latency_ms)
            
            return result
            
        except Exception as e:
            # Record failure
            self.load_balancer.record_failure(endpoint)
            raise
    
    def get_optimization_status(self) -> dict:
        """Get status of all optimization components."""
        return {
            "rpc_endpoints": self.load_balancer.get_status(),
            "gas": {
                "recommended_gwei": float(self.gas_manager.get_recommended_gas_price()),
                "max_gwei": float(self.gas_manager.max_gas_price_gwei),
            },
            "batch": {
                "pending_operations": self.batch_manager.pending_count,
            }
        }
