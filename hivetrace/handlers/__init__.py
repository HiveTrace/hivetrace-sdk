"""
Пакет обработчиков для HiveTrace SDK.

Содержит бизнес-логику обработки ошибок и построения ответов.
Организован по принципу единственной ответственности (SRP).
"""

from .error_handler import ErrorHandler
from .response_builder import ResponseBuilder

__all__ = [
    "ErrorHandler",
    "ResponseBuilder",
]
