"""AWS Bedrock client module"""

from .bedrock import BedrockClient, CircuitBreakerOpenError

__all__ = ["BedrockClient", "CircuitBreakerOpenError"]
