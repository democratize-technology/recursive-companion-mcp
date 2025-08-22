"""
Tests for security enhancements including credential sanitization.
"""

from botocore.exceptions import ClientError

from security_utils import CredentialSanitizer


class TestCredentialSanitizer:
    """Test comprehensive credential sanitization"""

    def test_sanitize_aws_access_keys(self):
        """Test AWS access key patterns are sanitized"""
        test_cases = [
            ("AKIAIOSFODNN7EXAMPLE", "[REDACTED_AWS_ACCESS_KEY]"),
            ("Error: ASIATESTACCESSKEY123", "Error: [REDACTED_AWS_ACCESS_KEY]"),
            (
                "Multiple AKIATEST1234567890ABCD and ASIATEST9876543210EFGH",
                "Multiple [REDACTED_AWS_ACCESS_KEY] and [REDACTED_AWS_ACCESS_KEY]",
            ),
        ]

        for input_text, expected_pattern in test_cases:
            result = CredentialSanitizer.sanitize_string(input_text)
            assert expected_pattern in result
            assert "AKIA" not in result
            assert "ASIA" not in result

    def test_sanitize_aws_secret_keys(self):
        """Test AWS secret key patterns are sanitized"""
        test_cases = [
            (
                "aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                "aws_secret_access_key=[REDACTED_AWS_SECRET_KEY]",
            ),
            (
                "SecretAccessKey: 'abcdefghij1234567890ABCDEFGHIJ1234567890'",
                "SecretAccessKey: '[REDACTED_AWS_SECRET_KEY]'",
            ),
        ]

        for input_text, expected_pattern in test_cases:
            result = CredentialSanitizer.sanitize_string(input_text)
            assert "REDACTED" in result
            assert "wJalrXUtnFEMI" not in result
            assert "abcdefghij1234567890" not in result

    def test_sanitize_session_tokens(self):
        """Test AWS session token sanitization"""
        long_token = "A" * 150  # Session tokens are typically very long
        test_input = f"aws_session_token={long_token}"
        result = CredentialSanitizer.sanitize_string(test_input)

        assert "REDACTED" in result
        assert long_token not in result

    def test_sanitize_iam_arns(self):
        """Test IAM ARN sanitization"""
        test_cases = [
            "arn:aws:iam::123456789012:user/TestUser",
            "arn:aws:iam::987654321098:role/TestRole",
        ]

        for arn in test_cases:
            result = CredentialSanitizer.sanitize_string(f"Error with {arn}")
            assert "REDACTED_ARN" in result
            assert "123456789012" not in result
            assert "987654321098" not in result
            assert "TestUser" not in result
            assert "TestRole" not in result

    def test_sanitize_authorization_headers(self):
        """Test authorization header sanitization"""
        test_cases = [
            ("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9", "Authorization"),
            ("X-Amz-Security-Token: FQoGZXIvYXdzEBYaD", "X-Amz-Security-Token"),
        ]

        for input_text, header_name in test_cases:
            result = CredentialSanitizer.sanitize_string(input_text)
            assert "REDACTED" in result
            assert "eyJhbG" not in result
            assert "FQoGZX" not in result

    def test_sanitize_generic_api_keys(self):
        """Test generic API key patterns"""
        test_cases = [
            "api_key=sk_test_123456789abcdefghijklmnop",
            "apikey: xyz987654321ABCDEFGHIJKLMNOP",
            "API-KEY='super_secret_key_12345678'",
        ]

        for test_input in test_cases:
            result = CredentialSanitizer.sanitize_string(test_input)
            assert "REDACTED" in result
            assert "sk_test" not in result
            assert "xyz98765" not in result
            assert "super_secret" not in result

    def test_sanitize_dict(self):
        """Test dictionary sanitization"""
        test_dict = {
            "message": "Error occurred",
            "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
            "password": "my_secret_password",
            "nested": {
                "token": "secret_token_123",
                "safe_field": "This is safe",
            },
            "api_key": "sk_live_abcdef123456",
        }

        result = CredentialSanitizer.sanitize_dict(test_dict)

        assert result["message"] == "Error occurred"
        assert result["aws_access_key_id"] == "[REDACTED]"
        assert result["password"] == "[REDACTED]"
        assert result["nested"]["token"] == "[REDACTED]"
        assert result["nested"]["safe_field"] == "This is safe"
        assert result["api_key"] == "[REDACTED]"

    def test_sanitize_list(self):
        """Test list sanitization"""
        test_list = [
            "safe_string",
            "AKIAIOSFODNN7EXAMPLE",
            {"password": "secret", "user": "admin"},
            ["nested", "api_key=test123"],
        ]

        result = CredentialSanitizer.sanitize_list(test_list)

        assert result[0] == "safe_string"
        assert "[REDACTED_AWS_ACCESS_KEY]" in result[1]
        assert result[2]["password"] == "[REDACTED]"
        assert result[2]["user"] == "admin"
        assert "REDACTED" in result[3][1]

    def test_sanitize_boto3_error(self):
        """Test boto3 ClientError sanitization"""
        # Create a mock boto3 error
        error = ClientError(
            error_response={
                "Error": {
                    "Code": "InvalidUserID.NotFound",
                    "Message": "User AKIAIOSFODNN7EXAMPLE not found",
                },
                "ResponseMetadata": {
                    "RequestId": "abcd-1234-efgh-5678",
                    "HTTPStatusCode": 404,
                    "HTTPHeaders": {
                        "x-amz-security-token": "secret_token_here",
                    },
                },
            },
            operation_name="GetUser",
        )

        result = CredentialSanitizer.sanitize_boto3_error(error)

        assert result["error_type"] == "ClientError"
        assert "REDACTED" in result["error_message"]
        assert "AKIAIOSFODNN7EXAMPLE" not in result["error_message"]
        assert result["response"]["Error"]["Code"] == "InvalidUserID.NotFound"
        assert "REDACTED" in result["response"]["Error"]["Message"]
        assert result["response"]["ResponseMetadata"]["HTTPStatusCode"] == 404
        assert "secret_token_here" not in str(result)

    def test_sanitize_complex_nested_structure(self):
        """Test deeply nested and complex data structures"""
        complex_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "secret": "deep_secret_value",
                        "credentials": {
                            "access_key": "AKIATEST123",
                            "secret_key": "wJalrXUtnFEMI/K7MDENG",
                        },
                    },
                },
                "array": [
                    {"token": "token1"},
                    {"authorization": "Bearer xyz123"},
                ],
            },
        }

        result = CredentialSanitizer.sanitize_dict(complex_data)

        assert result["level1"]["level2"]["level3"]["secret"] == "[REDACTED]"
        assert result["level1"]["level2"]["level3"]["credentials"]["access_key"] == "[REDACTED]"
        assert result["level1"]["level2"]["level3"]["credentials"]["secret_key"] == "[REDACTED]"
        assert result["level1"]["array"][0]["token"] == "[REDACTED]"
        assert result["level1"]["array"][1]["authorization"] == "[REDACTED]"

    def test_sanitize_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Empty string
        assert CredentialSanitizer.sanitize_string("") == ""

        # None
        assert CredentialSanitizer.sanitize_string(None) is None

        # Very long base64 string that might be credentials
        long_base64 = "A" * 100 + "="
        result = CredentialSanitizer.sanitize_string(long_base64)
        assert "REDACTED_POSSIBLE_CREDENTIAL" in result

        # String with multiple credential types
        multi_cred = "AKIATEST123 and secret=abcd1234567890ABCD1234567890ABCD123456"
        result = CredentialSanitizer.sanitize_string(multi_cred)
        assert result.count("REDACTED") >= 2

    def test_max_recursion_protection(self):
        """Test protection against max recursion depth"""
        result = CredentialSanitizer.sanitize_dict({"test": "value"}, max_depth=0)
        assert result == {"error": "Max recursion depth reached"}

        result = CredentialSanitizer.sanitize_list(["test"], max_depth=0)
        assert result == ["[Max recursion depth reached]"]

    def test_case_insensitive_field_matching(self):
        """Test that field matching is case-insensitive"""
        test_dict = {
            "PASSWORD": "secret1",
            "Password": "secret2",
            "password": "secret3",
            "API_KEY": "key1",
            "ApiKey": "key2",
            "api_key": "key3",
        }

        result = CredentialSanitizer.sanitize_dict(test_dict)

        for key in test_dict:
            assert result[key] == "[REDACTED]"
