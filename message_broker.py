# message_broker.py
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import pika
from pika.exceptions import AMQPError

from config import settings

logger = logging.getLogger(__name__)


class MessageBroker:
    """
    انتشار رویدادهای پیام به RabbitMQ.

    Exchange و Queue هر دو با نام LogAi پیکربندی می‌شوند.
    در صورت عدم دسترسی به RabbitMQ، خطا پرتاب نمی‌شود؛ فقط لاگ می‌شود.
    """

    @staticmethod
    def _build_connection() -> pika.BlockingConnection:
        credentials = pika.PlainCredentials(
            settings.rabbitmq_user,
            settings.rabbitmq_password,
        )
        params = pika.ConnectionParameters(
            host=settings.rabbitmq_host,
            port=settings.rabbitmq_port,
            virtual_host=settings.rabbitmq_vhost,
            credentials=credentials,
            heartbeat=30,
            blocked_connection_timeout=10,
            connection_attempts=2,
            retry_delay=1,
        )
        return pika.BlockingConnection(params)

    @staticmethod
    def _ensure_topology(
        channel: pika.adapters.blocking_connection.BlockingChannel,
    ) -> None:
        channel.exchange_declare(
            exchange=settings.rabbitmq_exchange,
            exchange_type="topic",
            durable=True,
        )
        channel.queue_declare(
            queue=settings.rabbitmq_queue,
            durable=True,
        )
        channel.queue_bind(
            queue=settings.rabbitmq_queue,
            exchange=settings.rabbitmq_exchange,
            routing_key=settings.rabbitmq_routing_key,
        )

    @staticmethod
    def publish_message_event(payload: Dict[str, Any]) -> bool:
        if not settings.rabbitmq_enabled:
            return False

        connection = None
        try:
            connection = MessageBroker._build_connection()
            channel = connection.channel()
            channel.confirm_delivery()

            MessageBroker._ensure_topology(channel)

            body = json.dumps(payload, ensure_ascii=False, default=str)

            channel.basic_publish(
                exchange=settings.rabbitmq_exchange,
                routing_key=settings.rabbitmq_routing_key,
                body=body.encode("utf-8"),
                properties=pika.BasicProperties(
                    content_type="application/json",
                    delivery_mode=2,  # persistent
                    timestamp=int(datetime.utcnow().timestamp()),
                ),
                mandatory=False,
            )
            return True

        except AMQPError as exc:
            logger.warning("RabbitMQ publish failed (AMQP): %s", exc)
            return False
        except Exception as exc:
            logger.warning("RabbitMQ publish failed (unexpected): %s", exc)
            return False
        finally:
            if connection is not None and connection.is_open:
                try:
                    connection.close()
                except Exception:
                    pass


def build_message_payload(
    message_id: str,
    conversation_id: str,
    tenant_id: Optional[str],
    user_id: Optional[str],
    app_id: Optional[str],
    role: str,
    content: str,
    meta_info: Optional[Dict[str, Any]],
    created_at: datetime,
) -> Dict[str, Any]:
    """
    ساخت payload استاندارد برای رویداد پیام.
    """
    return {
        "event_type": "message.created",
        "message_id": message_id,
        "conversation_id": conversation_id,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "app_id": app_id,
        "role": role,
        "content": content,
        "meta_info": meta_info or {},
        "created_at": created_at.isoformat(),
    }