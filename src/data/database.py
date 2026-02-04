"""
PR3DICT: SQLite Database for Market Tracking

Stores Polymarket listings and their dispute analysis results.
"""
import aiosqlite
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Default database path (gitignored)
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "markets.db"


class MarketDatabase:
    """
    SQLite database for tracking Polymarket listings and dispute analyses.
    
    Tables:
    - markets: Raw market data from Polymarket Gamma API
    - analyses: LLM analysis results with dispute probability scores
    - analysis_runs: Metadata for deterministic analysis replay
    - analysis_outputs_t1: Tier 1 structured outputs
    - analysis_outputs_t2: Tier 2 structured outputs
    - signals: Generated trade signals and edge context
    - market_outcomes: Ground truth settlement outcomes
    - calibration_metrics: Offline model calibration artifacts
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self._connection: Optional[aiosqlite.Connection] = None
    
    async def connect(self) -> None:
        """Initialize database connection and create tables."""
        # Ensure data directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._connection = await aiosqlite.connect(self.db_path)
        self._connection.row_factory = aiosqlite.Row
        
        await self._create_tables()
        logger.info(f"Connected to database: {self.db_path}")
    
    async def close(self) -> None:
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
    
    async def _create_tables(self) -> None:
        """Create database schema."""
        await self._connection.executescript("""
            -- Raw market data from Polymarket
            CREATE TABLE IF NOT EXISTS markets (
                id TEXT PRIMARY KEY,              -- Polymarket condition_id
                question TEXT NOT NULL,
                description TEXT,
                resolution_source TEXT,
                slug TEXT,
                
                -- Pricing
                yes_price REAL,
                no_price REAL,
                
                -- Volume/Liquidity
                volume REAL,
                liquidity REAL,
                
                -- Dates
                end_date TEXT,
                created_at TEXT,
                
                -- UMA Resolution
                uma_resolution_status TEXT,       -- null, proposed, disputed, resolved
                uma_bond TEXT,
                uma_reward TEXT,
                
                -- Tracking
                first_seen_at TEXT NOT NULL,
                last_updated_at TEXT NOT NULL,
                raw_json TEXT                     -- Full API response for reference
            );
            
            -- Analysis results from LLM pipeline
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                
                -- Dispute prediction
                dispute_probability REAL,         -- 0.0 to 1.0
                ambiguity_score REAL,             -- 0.0 to 1.0
                
                -- LLM outputs
                ambiguous_terms TEXT,             -- JSON array
                edge_cases TEXT,                  -- JSON array
                resolution_source_risk TEXT,      -- low, medium, high
                reasoning TEXT,
                
                -- Metadata
                model_used TEXT,                  -- Which LLM analyzed this
                analyzed_at TEXT NOT NULL,
                
                FOREIGN KEY (market_id) REFERENCES markets(id)
            );

            -- Deterministic analysis replay metadata
            CREATE TABLE IF NOT EXISTS analysis_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                stage TEXT NOT NULL,                 -- tier1, tier2
                run_id TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                strategy_version TEXT NOT NULL,
                latency_ms INTEGER,
                token_cost_usd REAL,
                status TEXT NOT NULL,                -- success, invalid, failed
                error_message TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(run_id),
                FOREIGN KEY (market_id) REFERENCES markets(id)
            );

            -- Tier 1 output contract
            CREATE TABLE IF NOT EXISTS analysis_outputs_t1 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_run_id INTEGER NOT NULL,
                market_id TEXT NOT NULL,
                screen_decision TEXT NOT NULL,       -- PASS, FLAG
                ambiguity_score REAL NOT NULL CHECK (ambiguity_score >= 0 AND ambiguity_score <= 1),
                dispute_prob_prior REAL NOT NULL CHECK (dispute_prob_prior >= 0 AND dispute_prob_prior <= 1),
                top_risks TEXT NOT NULL,             -- JSON array
                rationale_short TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (analysis_run_id) REFERENCES analysis_runs(id),
                FOREIGN KEY (market_id) REFERENCES markets(id)
            );

            -- Tier 2 output contract
            CREATE TABLE IF NOT EXISTS analysis_outputs_t2 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_run_id INTEGER NOT NULL,
                market_id TEXT NOT NULL,
                p_dispute REAL NOT NULL CHECK (p_dispute >= 0 AND p_dispute <= 1),
                p_yes_final REAL NOT NULL CHECK (p_yes_final >= 0 AND p_yes_final <= 1),
                p_no_final REAL NOT NULL CHECK (p_no_final >= 0 AND p_no_final <= 1),
                p_invalid_final REAL NOT NULL CHECK (p_invalid_final >= 0 AND p_invalid_final <= 1),
                confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                resolution_source_risk TEXT NOT NULL, -- low, medium, high
                edge_cases TEXT NOT NULL,             -- JSON array
                decision_path TEXT NOT NULL,          -- pre_dispute, post_proposal, active_dispute, initiate_dispute, no_trade
                no_trade_reason TEXT,
                assumptions TEXT NOT NULL,            -- JSON array
                created_at TEXT NOT NULL,
                FOREIGN KEY (analysis_run_id) REFERENCES analysis_runs(id),
                FOREIGN KEY (market_id) REFERENCES markets(id)
            );

            -- Signal artifacts for execution + replay
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                analysis_run_id INTEGER,
                action TEXT NOT NULL,                -- ENTER_YES, ENTER_NO, EXIT, HOLD, NO_TRADE
                side TEXT,                           -- yes, no, null for HOLD/NO_TRADE
                confidence REAL CHECK (confidence >= 0 AND confidence <= 1),
                edge_yes REAL,
                edge_no REAL,
                edge_selected REAL,
                yes_price_snapshot REAL,
                no_price_snapshot REAL,
                liquidity_snapshot REAL,
                reason_code TEXT NOT NULL,
                reason_detail TEXT,
                strategy_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (market_id) REFERENCES markets(id),
                FOREIGN KEY (analysis_run_id) REFERENCES analysis_runs(id)
            );

            -- Settled market labels for calibration and backtesting
            CREATE TABLE IF NOT EXISTS market_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL UNIQUE,
                disputed INTEGER NOT NULL,           -- 0,1
                final_resolution TEXT NOT NULL,      -- YES, NO, INVALID
                time_to_resolution_hours REAL,
                source_run_id TEXT,
                settled_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (market_id) REFERENCES markets(id)
            );

            -- Model calibration snapshots
            CREATE TABLE IF NOT EXISTS calibration_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                strategy_version TEXT NOT NULL,
                time_bucket TEXT NOT NULL,           -- e.g. 2026-W05
                sample_size INTEGER NOT NULL,
                brier_score REAL,
                log_loss REAL,
                calibration_error REAL,
                metadata_json TEXT,                  -- arbitrary JSON
                created_at TEXT NOT NULL
            );
            
            -- Indexes for common queries
            CREATE INDEX IF NOT EXISTS idx_markets_liquidity ON markets(liquidity);
            CREATE INDEX IF NOT EXISTS idx_markets_uma_status ON markets(uma_resolution_status);
            CREATE INDEX IF NOT EXISTS idx_markets_end_date ON markets(end_date);
            CREATE INDEX IF NOT EXISTS idx_analyses_market ON analyses(market_id);
            CREATE INDEX IF NOT EXISTS idx_analyses_dispute_prob ON analyses(dispute_probability);
            CREATE INDEX IF NOT EXISTS idx_analysis_runs_market_stage ON analysis_runs(market_id, stage);
            CREATE INDEX IF NOT EXISTS idx_t1_market_created ON analysis_outputs_t1(market_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_t2_market_created ON analysis_outputs_t2(market_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_signals_market_created ON signals(market_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_market_outcomes_resolution ON market_outcomes(final_resolution);
            CREATE INDEX IF NOT EXISTS idx_calibration_model_bucket ON calibration_metrics(model, prompt_version, time_bucket);
        """)
        await self._connection.commit()
    
    # --- Market Operations ---
    
    async def upsert_market(self, market: Dict[str, Any]) -> None:
        """Insert or update a market from Polymarket API response."""
        now = datetime.now(timezone.utc).isoformat()
        
        await self._connection.execute("""
            INSERT INTO markets (
                id, question, description, resolution_source, slug,
                yes_price, no_price, volume, liquidity, end_date, created_at,
                uma_resolution_status, uma_bond, uma_reward,
                first_seen_at, last_updated_at, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                question = excluded.question,
                description = excluded.description,
                resolution_source = excluded.resolution_source,
                yes_price = excluded.yes_price,
                no_price = excluded.no_price,
                volume = excluded.volume,
                liquidity = excluded.liquidity,
                end_date = excluded.end_date,
                uma_resolution_status = excluded.uma_resolution_status,
                uma_bond = excluded.uma_bond,
                uma_reward = excluded.uma_reward,
                last_updated_at = excluded.last_updated_at,
                raw_json = excluded.raw_json
        """, (
            market.get("conditionId") or market.get("id"),
            market.get("question"),
            market.get("description"),
            market.get("resolutionSource"),
            market.get("slug"),
            self._parse_price(market.get("outcomePrices"), 0),
            self._parse_price(market.get("outcomePrices"), 1),
            float(market.get("volumeNum") or market.get("volume") or 0),
            float(market.get("liquidityNum") or market.get("liquidity") or 0),
            market.get("endDate"),
            market.get("createdAt"),
            market.get("umaResolutionStatus"),
            market.get("umaBond"),
            market.get("umaReward"),
            now,  # first_seen_at (ignored on conflict)
            now,  # last_updated_at
            json.dumps(market)
        ))
        await self._connection.commit()
    
    def _parse_price(self, outcome_prices: Any, index: int) -> Optional[float]:
        """Parse outcome prices from API response."""
        if not outcome_prices:
            return None
        try:
            if isinstance(outcome_prices, str):
                prices = json.loads(outcome_prices)
            else:
                prices = outcome_prices
            return float(prices[index]) if len(prices) > index else None
        except (json.JSONDecodeError, IndexError, TypeError):
            return None
    
    async def get_market(self, market_id: str) -> Optional[Dict[str, Any]]:
        """Get a single market by ID."""
        async with self._connection.execute(
            "SELECT * FROM markets WHERE id = ?", (market_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None
    
    async def get_unanalyzed_markets(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get markets that haven't been analyzed yet."""
        async with self._connection.execute("""
            SELECT m.* FROM markets m
            LEFT JOIN analyses a ON m.id = a.market_id
            WHERE a.id IS NULL
            ORDER BY m.liquidity DESC
            LIMIT ?
        """, (limit,)) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
    
    async def get_markets_by_liquidity(
        self,
        min_liquidity: float = 0,
        max_liquidity: float = float('inf'),
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get markets within a liquidity range."""
        async with self._connection.execute("""
            SELECT * FROM markets
            WHERE liquidity >= ? AND liquidity <= ?
            ORDER BY liquidity DESC
            LIMIT ?
        """, (min_liquidity, max_liquidity, limit)) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
    
    # --- Analysis Operations ---
    
    async def save_analysis(
        self,
        market_id: str,
        dispute_probability: float,
        ambiguity_score: float,
        ambiguous_terms: List[str],
        edge_cases: List[str],
        resolution_source_risk: str,
        reasoning: str,
        model_used: str
    ) -> int:
        """Save LLM analysis results for a market."""
        now = datetime.now(timezone.utc).isoformat()
        
        cursor = await self._connection.execute("""
            INSERT INTO analyses (
                market_id, dispute_probability, ambiguity_score,
                ambiguous_terms, edge_cases, resolution_source_risk,
                reasoning, model_used, analyzed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            market_id,
            dispute_probability,
            ambiguity_score,
            json.dumps(ambiguous_terms),
            json.dumps(edge_cases),
            resolution_source_risk,
            reasoning,
            model_used,
            now
        ))
        await self._connection.commit()
        return cursor.lastrowid
    
    async def get_analysis(self, market_id: str) -> Optional[Dict[str, Any]]:
        """Get the most recent analysis for a market."""
        async with self._connection.execute("""
            SELECT * FROM analyses
            WHERE market_id = ?
            ORDER BY analyzed_at DESC
            LIMIT 1
        """, (market_id,)) as cursor:
            row = await cursor.fetchone()
            if row:
                result = dict(row)
                result["ambiguous_terms"] = json.loads(result.get("ambiguous_terms") or "[]")
                result["edge_cases"] = json.loads(result.get("edge_cases") or "[]")
                return result
            return None
    
    async def get_high_dispute_markets(
        self,
        min_probability: float = 0.5,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get markets with high dispute probability."""
        async with self._connection.execute("""
            SELECT m.*, a.dispute_probability, a.ambiguity_score, 
                   a.reasoning, a.analyzed_at
            FROM markets m
            JOIN analyses a ON m.id = a.market_id
            WHERE a.dispute_probability >= ?
            ORDER BY a.dispute_probability DESC
            LIMIT ?
        """, (min_probability, limit)) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def save_analysis_run(
        self,
        market_id: str,
        stage: str,
        run_id: str,
        model: str,
        prompt_version: str,
        strategy_version: str,
        status: str,
        latency_ms: Optional[int] = None,
        token_cost_usd: Optional[float] = None,
        error_message: Optional[str] = None
    ) -> int:
        """Persist run metadata for deterministic replay."""
        now = datetime.now(timezone.utc).isoformat()
        cursor = await self._connection.execute("""
            INSERT INTO analysis_runs (
                market_id, stage, run_id, model, prompt_version, strategy_version,
                latency_ms, token_cost_usd, status, error_message, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            market_id, stage, run_id, model, prompt_version, strategy_version,
            latency_ms, token_cost_usd, status, error_message, now
        ))
        await self._connection.commit()
        return cursor.lastrowid

    async def save_tier1_output(
        self,
        analysis_run_id: int,
        market_id: str,
        screen_decision: str,
        ambiguity_score: float,
        dispute_prob_prior: float,
        top_risks: List[str],
        rationale_short: str
    ) -> int:
        """Persist validated Tier 1 output contract."""
        now = datetime.now(timezone.utc).isoformat()
        cursor = await self._connection.execute("""
            INSERT INTO analysis_outputs_t1 (
                analysis_run_id, market_id, screen_decision, ambiguity_score,
                dispute_prob_prior, top_risks, rationale_short, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            analysis_run_id, market_id, screen_decision, ambiguity_score,
            dispute_prob_prior, json.dumps(top_risks), rationale_short, now
        ))
        await self._connection.commit()
        return cursor.lastrowid

    async def save_tier2_output(
        self,
        analysis_run_id: int,
        market_id: str,
        p_dispute: float,
        p_yes_final: float,
        p_no_final: float,
        p_invalid_final: float,
        confidence: float,
        resolution_source_risk: str,
        edge_cases: List[str],
        decision_path: str,
        no_trade_reason: Optional[str],
        assumptions: List[str]
    ) -> int:
        """Persist validated Tier 2 output contract."""
        now = datetime.now(timezone.utc).isoformat()
        cursor = await self._connection.execute("""
            INSERT INTO analysis_outputs_t2 (
                analysis_run_id, market_id, p_dispute, p_yes_final, p_no_final,
                p_invalid_final, confidence, resolution_source_risk, edge_cases,
                decision_path, no_trade_reason, assumptions, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            analysis_run_id, market_id, p_dispute, p_yes_final, p_no_final,
            p_invalid_final, confidence, resolution_source_risk, json.dumps(edge_cases),
            decision_path, no_trade_reason, json.dumps(assumptions), now
        ))
        await self._connection.commit()
        return cursor.lastrowid

    async def save_signal(
        self,
        market_id: str,
        action: str,
        reason_code: str,
        strategy_version: str,
        analysis_run_id: Optional[int] = None,
        side: Optional[str] = None,
        confidence: Optional[float] = None,
        edge_yes: Optional[float] = None,
        edge_no: Optional[float] = None,
        edge_selected: Optional[float] = None,
        yes_price_snapshot: Optional[float] = None,
        no_price_snapshot: Optional[float] = None,
        liquidity_snapshot: Optional[float] = None,
        reason_detail: Optional[str] = None
    ) -> int:
        """Persist signal with edge context for replay."""
        now = datetime.now(timezone.utc).isoformat()
        cursor = await self._connection.execute("""
            INSERT INTO signals (
                market_id, analysis_run_id, action, side, confidence, edge_yes, edge_no,
                edge_selected, yes_price_snapshot, no_price_snapshot, liquidity_snapshot,
                reason_code, reason_detail, strategy_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            market_id, analysis_run_id, action, side, confidence, edge_yes, edge_no,
            edge_selected, yes_price_snapshot, no_price_snapshot, liquidity_snapshot,
            reason_code, reason_detail, strategy_version, now
        ))
        await self._connection.commit()
        return cursor.lastrowid

    async def save_market_outcome(
        self,
        market_id: str,
        disputed: bool,
        final_resolution: str,
        settled_at: str,
        time_to_resolution_hours: Optional[float] = None,
        source_run_id: Optional[str] = None
    ) -> int:
        """Persist market settlement ground truth labels."""
        now = datetime.now(timezone.utc).isoformat()
        cursor = await self._connection.execute("""
            INSERT INTO market_outcomes (
                market_id, disputed, final_resolution, time_to_resolution_hours,
                source_run_id, settled_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(market_id) DO UPDATE SET
                disputed = excluded.disputed,
                final_resolution = excluded.final_resolution,
                time_to_resolution_hours = excluded.time_to_resolution_hours,
                source_run_id = excluded.source_run_id,
                settled_at = excluded.settled_at
        """, (
            market_id, int(disputed), final_resolution, time_to_resolution_hours,
            source_run_id, settled_at, now
        ))
        await self._connection.commit()
        return cursor.lastrowid

    async def save_calibration_metric(
        self,
        model: str,
        prompt_version: str,
        strategy_version: str,
        time_bucket: str,
        sample_size: int,
        brier_score: Optional[float],
        log_loss: Optional[float],
        calibration_error: Optional[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Persist calibration snapshots for model tracking."""
        now = datetime.now(timezone.utc).isoformat()
        cursor = await self._connection.execute("""
            INSERT INTO calibration_metrics (
                model, prompt_version, strategy_version, time_bucket, sample_size,
                brier_score, log_loss, calibration_error, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model, prompt_version, strategy_version, time_bucket, sample_size,
            brier_score, log_loss, calibration_error, json.dumps(metadata or {}), now
        ))
        await self._connection.commit()
        return cursor.lastrowid

    async def get_signal_replay(self, signal_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a signal with run and model artifacts for deterministic replay."""
        async with self._connection.execute("""
            SELECT
                s.*,
                ar.run_id,
                ar.stage,
                ar.model,
                ar.prompt_version,
                ar.strategy_version AS run_strategy_version,
                ar.status AS run_status
            FROM signals s
            LEFT JOIN analysis_runs ar ON s.analysis_run_id = ar.id
            WHERE s.id = ?
        """, (signal_id,)) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None
    
    # --- Stats ---
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {}
        
        async with self._connection.execute("SELECT COUNT(*) FROM markets") as cursor:
            stats["total_markets"] = (await cursor.fetchone())[0]
        
        async with self._connection.execute("SELECT COUNT(*) FROM analyses") as cursor:
            stats["total_analyses"] = (await cursor.fetchone())[0]
        
        async with self._connection.execute("""
            SELECT COUNT(*) FROM markets m
            LEFT JOIN analyses a ON m.id = a.market_id
            WHERE a.id IS NULL
        """) as cursor:
            stats["unanalyzed_markets"] = (await cursor.fetchone())[0]
        
        return stats
