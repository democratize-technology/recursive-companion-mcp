# Security Policy

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in the recursive-companion-mcp server, please report it responsibly.

### How to Report

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please email us at: **security@thinkerz.ai**

Include the following information:
- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Any suggested fixes (if available)

### What to Expect

1. **Acknowledgment**: We'll acknowledge receipt within 48 hours
2. **Initial Assessment**: We'll provide an initial assessment within 5 business days
3. **Updates**: We'll keep you informed of our progress
4. **Resolution**: We aim to resolve critical vulnerabilities within 30 days

### Disclosure Timeline

- We'll work with you to understand and resolve the issue
- Once fixed, we'll coordinate public disclosure
- You'll be credited for the discovery (unless you prefer to remain anonymous)

## Security Best Practices for Users

### AWS Credentials
- **Never commit AWS credentials to version control**
- Use IAM roles and least-privilege policies
- Rotate credentials regularly
- Monitor AWS CloudTrail for unauthorized access

### Configuration Security
- Store sensitive configuration in environment variables
- Use AWS Secrets Manager or similar for production deployments
- Validate all input parameters
- Enable logging and monitoring

### Network Security
- Run the MCP server in a secure network environment
- Use TLS for all communications when possible
- Implement proper firewall rules
- Monitor for unusual network activity

### Dependency Management
- Keep dependencies updated using `uv sync`
- Monitor for security advisories
- Use `safety check` to scan for known vulnerabilities
- Review dependency changes before updates

## Security Features

### Built-in Protections
- **Credential Sanitization**: Automatic removal of AWS keys from logs and error messages
- **Input Validation**: Comprehensive validation using Pydantic models
- **Circuit Breakers**: Protection against resource exhaustion
- **Session Isolation**: Each session operates independently
- **Error Handling**: Secure error responses that don't leak sensitive information

### Security Controls
- Rate limiting to prevent abuse
- Request size limits
- Timeout protections
- Structured logging for audit trails

## Security Testing

We use several tools and practices to maintain security:

- **Static Analysis**: Bandit for Python security issues
- **Dependency Scanning**: Safety checks for known vulnerabilities
- **Code Review**: All changes require security review
- **Testing**: Security-focused unit and integration tests

## Security Updates

Security updates will be:
- Released as patch versions (e.g., 1.0.1)
- Documented in CHANGELOG.md with security advisory details
- Announced through GitHub security advisories
- Tagged with security labels for easy identification

## Scope

This security policy covers:
- The recursive-companion-mcp server code
- Configuration and deployment guidance
- Dependencies and third-party integrations
- Documentation and examples

This policy does NOT cover:
- AWS Bedrock service security (AWS responsibility)
- Client application security
- Infrastructure security (user responsibility)
- Third-party MCP client security

## Contact

For security-related questions or concerns:
- Email: security@thinkerz.ai
- For non-security issues, use GitHub issues

## Recognition

We appreciate security researchers who help keep our project safe. Contributors who responsibly disclose vulnerabilities will be:
- Acknowledged in our security advisories
- Listed in our CREDITS.md file (with permission)
- Invited to test future security updates (optional)

Thank you for helping keep the recursive-companion-mcp server secure!
