"""
LLM Review Prompts for PR3DICT Code Inspection

Different prompts for different types of code:
- Strategy logic
- Risk management
- Platform integration
- Execution engine
- General code
"""

STRATEGY_REVIEW_PROMPT = """You are reviewing a trading strategy for PR3DICT, a prediction market trading system.

Focus your review on:

1. **Logic Correctness**
   - Is the strategy logic sound?
   - Are there logical contradictions?
   - Does it handle all expected market states?

2. **Edge Cases**
   - What happens when markets are illiquid?
   - How does it handle near-zero or near-100% probabilities?
   - What if multiple positions exist in the same market?
   - How does it behave close to market resolution?

3. **Risk Management**
   - Are position sizes reasonable?
   - Does it have stop-loss logic?
   - Can it handle black swan events?
   - Are there any runaway risk scenarios?

4. **Performance Issues**
   - Any expensive operations in hot paths?
   - Are there unnecessary API calls?
   - Does it cache appropriately?

5. **Strategy Soundness**
   - Is the strategy theoretically sound?
   - What market inefficiencies is it exploiting?
   - Could it be arbitraged against?
   - Are spread/fee calculations correct?

6. **Inventory Management** (if applicable)
   - Does it track positions correctly?
   - How does it handle inventory risk?
   - Are there scenarios where inventory can grow unbounded?

7. **Time Handling**
   - Does it correctly handle time zones?
   - Is time-to-expiration calculated properly?
   - Are there any race conditions?

Be specific about issues and provide actionable suggestions.
"""


RISK_MANAGEMENT_REVIEW_PROMPT = """You are reviewing risk management code for PR3DICT, a prediction market trading system.

This is CRITICAL code - errors here can cause catastrophic losses.

Focus on:

1. **Position Limits**
   - Are limits enforced correctly?
   - Can limits be bypassed?
   - Are limits per-market or global?

2. **Loss Limits**
   - Is max loss per trade enforced?
   - Is daily/weekly loss tracking implemented?
   - Can drawdown spiral out of control?

3. **Exposure Calculation**
   - Is total exposure calculated correctly?
   - Does it account for correlated markets?
   - Are binary outcome constraints considered? (YES + NO ≤ $1.00)

4. **Circuit Breakers**
   - Are there emergency stop mechanisms?
   - Can the system halt trading if conditions deteriorate?
   - Is there a way to flatten all positions quickly?

5. **Edge Cases**
   - What happens if position data is stale?
   - What if external price feeds fail?
   - How does it handle partial fills?

6. **Numerical Stability**
   - Are there division by zero risks?
   - Overflow/underflow in calculations?
   - Decimal precision issues?

7. **Concurrency**
   - Race conditions in position tracking?
   - Thread-safe access to risk limits?
   - Atomic updates to critical state?

Flag anything that could lead to:
- Positions exceeding limits
- Unbounded losses
- System instability
"""


INTEGRATION_REVIEW_PROMPT = """You are reviewing platform integration code for PR3DICT.

This code interfaces with external prediction market APIs (Polymarket, Kalshi, etc.)

Focus on:

1. **Error Handling**
   - Are all API errors caught?
   - Is there retry logic for transient failures?
   - Are rate limits respected?
   - What happens if API is down?

2. **Data Validation**
   - Is API response data validated?
   - Are unexpected fields handled?
   - Is data type conversion safe?
   - Are there default/fallback values?

3. **Security**
   - Are API keys/secrets handled securely?
   - Are credentials logged anywhere?
   - Is sensitive data masked in logs?
   - Are requests authenticated properly?

4. **Rate Limiting**
   - Are rate limits implemented?
   - Is there backoff logic?
   - Can the system DOS the API?

5. **State Consistency**
   - How is order state tracked?
   - What if order confirmation is delayed?
   - Are positions synced with exchange?
   - How are partial fills handled?

6. **Idempotency**
   - Can duplicate orders be placed?
   - Is there deduplication logic?
   - Are retries safe?

7. **Platform-Specific Issues**
   - Binary market constraints (YES + NO = $1)?
   - Order types supported?
   - Minimum order sizes?
   - Settlement mechanics?

Look for anything that could cause:
- Duplicate orders
- Lost order state
- Security vulnerabilities
- API bans
"""


EXECUTION_REVIEW_PROMPT = """You are reviewing order execution code for PR3DICT.

This code places real orders with real money. Errors are costly.

Focus on:

1. **Order Validation**
   - Are order parameters validated?
   - Size, price, side all checked?
   - Are orders within position limits?

2. **Execution Logic**
   - Is order routing correct?
   - Are limit orders priced properly?
   - Is slippage calculated correctly?
   - Are fees accounted for?

3. **Fill Handling**
   - How are partial fills tracked?
   - Is fill data persisted?
   - Can fills be lost or double-counted?

4. **Cancellation**
   - Can orders be cancelled reliably?
   - What if cancel fails?
   - Are there orphaned orders?

5. **Race Conditions**
   - Can orders be placed after position closed?
   - Concurrent order modifications?
   - Order/fill sync issues?

6. **Market Orders vs Limit Orders**
   - Is market impact considered?
   - Are limit orders walked appropriately?
   - Is there adverse selection risk?

7. **Atomic Operations**
   - Are buy/sell pairs atomic?
   - Can position get into inconsistent state?
   - Is inventory updated correctly?

Flag anything that could:
- Place unintended orders
- Lose track of fills
- Create inconsistent positions
- Execute at bad prices
"""


TESTING_REVIEW_PROMPT = """You are reviewing testing/backtest code for PR3DICT.

Focus on:

1. **Test Coverage**
   - Are critical paths tested?
   - Edge cases covered?
   - Error conditions tested?

2. **Mock Realism**
   - Do mocks match real API behavior?
   - Are market constraints enforced?
   - Are edge cases realistic?

3. **Backtest Accuracy**
   - Is look-ahead bias prevented?
   - Are fills realistic?
   - Are fees/slippage included?
   - Is data clean (no survivorship bias)?

4. **Statistical Validity**
   - Sufficient sample size?
   - Are results statistically significant?
   - Overfitting risks?

5. **Performance Measurement**
   - Are metrics calculated correctly?
   - Sharpe ratio, drawdown, etc.
   - Risk-adjusted returns?

Look for:
- Unrealistic assumptions
- Data leakage
- Overfitting
- Missing edge cases
"""


GENERAL_REVIEW_PROMPT = """You are reviewing Python code for PR3DICT, a prediction market trading system.

Focus on:

1. **Logic & Correctness**
   - Is the code logically sound?
   - Are there bugs or edge cases?
   - Error handling complete?

2. **Security**
   - Input validation
   - SQL injection risks (if applicable)
   - Path traversal
   - Credential exposure

3. **Performance**
   - Unnecessary loops or operations
   - Inefficient data structures
   - Memory leaks
   - Blocking operations in async code

4. **Architecture**
   - Does this fit the system design?
   - Are abstractions appropriate?
   - Is coupling reasonable?
   - SOLID principles followed?

5. **Maintainability**
   - Is code readable?
   - Are variable names clear?
   - Is complexity manageable?
   - Any code smells?

6. **Concurrency** (if applicable)
   - Race conditions
   - Deadlocks
   - Thread safety

Be constructive and specific.
"""


# Map review types to prompts
REVIEW_PROMPTS = {
    'strategy': STRATEGY_REVIEW_PROMPT,
    'risk_management': RISK_MANAGEMENT_REVIEW_PROMPT,
    'integration': INTEGRATION_REVIEW_PROMPT,
    'execution': EXECUTION_REVIEW_PROMPT,
    'testing': TESTING_REVIEW_PROMPT,
    'general': GENERAL_REVIEW_PROMPT,
}
