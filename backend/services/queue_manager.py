# Simple placeholder; for production consider Celery or RQ.
class QueueManager:
    @staticmethod
    def enqueue(task_name: str, payload: dict) -> str:
        # For now, just log and execute synchronously.
        from backend.logger import logger
        logger.info(f"Enqueued task {task_name} with payload {payload}")
        return "task_id_placeholder"
