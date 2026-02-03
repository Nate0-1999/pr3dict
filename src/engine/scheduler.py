"""
PR3DICT: Task Scheduler

Handles periodic tasks like daily summaries.
Runs independently of the main trading loop.
"""
import asyncio
import logging
from datetime import datetime, time, timezone
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class TaskScheduler:
    """
    Async task scheduler for periodic events.
    
    Handles:
    - Daily summary at midnight
    - Periodic health checks
    - Custom scheduled tasks
    """
    
    def __init__(self):
        self.running = False
        self._tasks = []
        self._daily_summary_time = time(0, 0)  # Midnight UTC
    
    async def start(self) -> None:
        """Start the scheduler."""
        self.running = True
        logger.info("Task scheduler started")
    
    async def stop(self) -> None:
        """Stop the scheduler."""
        self.running = False
        
        # Cancel all running tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._tasks.clear()
        logger.info("Task scheduler stopped")
    
    def schedule_daily_summary(self, callback: Callable, target_time: Optional[time] = None) -> None:
        """
        Schedule daily summary callback.
        
        Args:
            callback: Async function to call at target time
            target_time: Time to run (default: midnight UTC)
        """
        if target_time:
            self._daily_summary_time = target_time
        
        task = asyncio.create_task(self._daily_summary_loop(callback))
        self._tasks.append(task)
        logger.info(f"Daily summary scheduled for {self._daily_summary_time}")
    
    async def _daily_summary_loop(self, callback: Callable) -> None:
        """Run daily summary at scheduled time."""
        while self.running:
            try:
                # Calculate seconds until next target time
                now = datetime.now(timezone.utc)
                target = datetime.combine(
                    now.date(),
                    self._daily_summary_time,
                    tzinfo=timezone.utc
                )
                
                # If target time has passed today, schedule for tomorrow
                if now >= target:
                    from datetime import timedelta
                    target += timedelta(days=1)
                
                sleep_seconds = (target - now).total_seconds()
                logger.debug(f"Next daily summary in {sleep_seconds/3600:.1f} hours")
                
                # Sleep until target time
                await asyncio.sleep(sleep_seconds)
                
                # Execute callback
                logger.info("Running daily summary...")
                await callback()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Daily summary error: {e}", exc_info=True)
                # Wait 1 hour before retrying
                await asyncio.sleep(3600)
    
    def schedule_periodic(self, callback: Callable, interval_seconds: int, name: str = "task") -> None:
        """
        Schedule periodic task.
        
        Args:
            callback: Async function to call periodically
            interval_seconds: Interval between calls
            name: Task name for logging
        """
        task = asyncio.create_task(self._periodic_loop(callback, interval_seconds, name))
        self._tasks.append(task)
        logger.info(f"Periodic task '{name}' scheduled every {interval_seconds}s")
    
    async def _periodic_loop(self, callback: Callable, interval: int, name: str) -> None:
        """Run periodic task loop."""
        while self.running:
            try:
                await asyncio.sleep(interval)
                await callback()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic task '{name}' error: {e}", exc_info=True)


async def _example_daily_summary():
    """Example daily summary callback."""
    logger.info("Daily summary executed!")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    async def main():
        scheduler = TaskScheduler()
        await scheduler.start()
        
        # Schedule daily summary at midnight
        scheduler.schedule_daily_summary(_example_daily_summary)
        
        # Schedule periodic health check every 5 minutes
        async def health_check():
            logger.info("Health check OK")
        
        scheduler.schedule_periodic(health_check, interval_seconds=300, name="health_check")
        
        # Run for demonstration
        await asyncio.sleep(10)
        
        await scheduler.stop()
    
    asyncio.run(main())
