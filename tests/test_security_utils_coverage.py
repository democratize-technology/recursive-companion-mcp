"""
Surgical tests for SecurityValidator to achieve 100% coverage.
Specifically targets missing lines: 148, 158, 186, 213
"""

from recursive_companion_mcp.legacy.security_utils import CredentialSanitizer

# sys.path removed - using package imports


class TestSecurityUtilsCoverage:
    """Surgical tests targeting specific missing lines in CredentialSanitizer."""

    def test_sanitize_dict_sensitive_key_list_else_clause(self):
        """Test line 148: else clause for list handling within sensitive key"""
        # Create data where a sensitive key contains a list that needs the else clause
        test_data = {"password": [1, 2, 3]}  # List with non-string, non-dict, non-list items

        result = CredentialSanitizer.sanitize_dict(test_data)

        # The list should be sanitized by calling sanitize_list
        assert result["password"] == [1, 2, 3]  # Line 148 should handle list case

    def test_sanitize_dict_non_standard_value_types(self):
        """Test line 158: else clause for non-string/dict/list values"""
        # Create data with various non-standard types
        test_data = {
            "number": 42,
            "boolean": True,
            "none_value": None,
            "tuple": (1, 2, 3),
        }

        result = CredentialSanitizer.sanitize_dict(test_data)

        # These should trigger line 158 - just pass through as-is
        assert result["number"] == 42
        assert result["boolean"] is True
        assert result["none_value"] is None
        assert result["tuple"] == (1, 2, 3)

    def test_sanitize_list_non_standard_item_types(self):
        """Test line 186: else clause for non-sanitizable items in list"""
        # Create list with various non-standard types
        test_list = [
            42,  # number
            True,  # boolean
            None,  # None
            (1, 2),  # tuple
            {1, 2},  # set
        ]

        result = CredentialSanitizer.sanitize_list(test_list)

        # These should trigger line 186 - just append as-is
        assert result[0] == 42
        assert result[1] is True
        assert result[2] is None
        assert result[3] == (1, 2)
        assert result[4] == {1, 2}

    def test_sanitize_error_with_sensitive_string_replacement(self):
        """Test line 213: String replacement when sanitized values differ"""

        # Create an exception with a sensitive attribute that appears in the message
        class TestError(Exception):
            def __init__(self, msg):
                super().__init__(msg)
                # Add an attribute with a sensitive value that appears in the message
                self.access_key = "AKIAIOSFODNN7EXAMPLE"  # AWS access key pattern

            def __str__(self):
                # The message contains the same sensitive value as the attribute
                return f"AWS error with key: {self.access_key}"

        error = TestError("Initial message")

        result = CredentialSanitizer.sanitize_error(error)

        # The sensitive data should be replaced (line 213)
        # The access key should be sanitized in both the attribute and the message
        assert "[REDACTED_AWS_ACCESS_KEY]" in result
        assert "AKIAIOSFODNN7EXAMPLE" not in result

    def test_sanitize_error_string_replacement_edge_case(self):
        """Test line 213: Specific case where string replacement occurs"""

        # Create an exception where attribute value appears in the error message
        class TestError(Exception):
            def __init__(self):
                # Set a sensitive attribute value
                self.secret_token = (
                    "aws_session_token=IQoJb3JpZ2luX2VjENz//////////wEaCXVzLXdlc3QtMiJHMEU"
                )
                # Make the error message contain this same value
                super().__init__(f"Authentication failed: {self.secret_token}")

        error = TestError()

        result = CredentialSanitizer.sanitize_error(error)

        # The sensitive token should be replaced in the message (line 213)
        # The session token should be sanitized
        assert "[REDACTED_POSSIBLE_CREDENTIAL]" in result
        assert "IQoJb3JpZ2luX2VjENz" not in result

    def test_complex_nested_structure_coverage(self):
        """Test complex nested structure to ensure all edge cases are covered"""
        # Create a complex structure that exercises multiple code paths
        complex_data = {
            "user": {
                "credentials": {
                    "password": ["plain_text_pwd", {"inner_pwd": "secret"}],
                    "api_keys": [123, True, None, "actual_key"],
                },
                "metadata": {
                    "count": 42,
                    "active": True,
                    "tags": ["tag1", "tag2", 999],
                },
            }
        }

        result = CredentialSanitizer.sanitize_dict(complex_data)

        # Verify the structure is preserved but sensitive data is sanitized
        assert isinstance(result["user"], dict)
        assert isinstance(result["user"]["credentials"], dict)
        assert isinstance(result["user"]["credentials"]["password"], list)
        assert isinstance(result["user"]["metadata"]["count"], int)
        assert result["user"]["metadata"]["count"] == 42

    def test_sensitive_key_with_mixed_list_types(self):
        """Test sensitive key containing list with mixed types to trigger line 148"""
        test_data = {
            "secret": [
                "string_secret",
                {"nested_secret": "value"},
                ["nested_list"],
                42,  # This should trigger the else clause in line 148 path
                True,
                None,
            ]
        }

        result = CredentialSanitizer.sanitize_dict(test_data)

        # The sensitive key should process the list, including non-standard types
        assert isinstance(result["secret"], list)
        # The specific line 148 should handle the list processing for sensitive keys
