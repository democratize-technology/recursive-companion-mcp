# API Examples - Recursive Companion MCP

This guide provides comprehensive, practical examples for using the Recursive Companion MCP server. All examples include realistic scenarios, expected responses, and common error patterns.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Basic Usage Patterns](#basic-usage-patterns)
3. [Domain-Specific Workflows](#domain-specific-workflows)
4. [Advanced Usage Scenarios](#advanced-usage-scenarios)
5. [Error Handling & Troubleshooting](#error-handling--troubleshooting)
6. [Claude Desktop Integration](#claude-desktop-integration)
7. [Integration with Other Thinkerz MCP Servers](#integration-with-other-thinkerz-mcp-servers)

## Quick Start

### Simple Refinement (Quick Mode)
For straightforward tasks that don't need step-by-step control:

```bash
# Start and complete refinement in one call
quick_refine(prompt="Explain the key principles of secure API design", max_wait=60)
```

**Expected Response:**
```json
{
  "success": true,
  "final_answer": "Secure API design principles include:\n\n1. **Authentication & Authorization**...",
  "iterations": 3,
  "time_taken": 45.2,
  "convergence_score": 0.96
}
```

### Step-by-Step Refinement (Full Control)
For complex tasks where you want to monitor progress:

```bash
# Step 1: Start refinement
start_refinement(prompt="Create a comprehensive data governance framework for a financial services company")

# Step 2: Continue iteratively
continue_refinement()  # First iteration
continue_refinement()  # Second iteration
continue_refinement()  # Final iteration

# Step 3: Get final result
get_final_result()
```

## Basic Usage Patterns

### Pattern 1: Standard Workflow

```bash
# 1. Start a refinement session
start_refinement(
    prompt="Design a microservices architecture for an e-commerce platform",
    domain="technical"
)
```

**Response:**
```json
{
  "success": true,
  "session_id": "ref_12345678-abcd-4321-9876-fedcba098765",
  "status": "initializing",
  "prompt": "Design a microservices architecture for an e-commerce platform",
  "domain": "technical",
  "message": "Refinement session started",
  "next_step": "Use continue_refinement to begin drafting"
}
```

```bash
# 2. Continue the refinement process
continue_refinement(session_id="ref_12345678-abcd-4321-9876-fedcba098765")
```

**Response (Draft Phase):**
```json
{
  "success": true,
  "session_id": "ref_12345678-abcd-4321-9876-fedcba098765",
  "status": "drafting",
  "step": "draft",
  "iteration": 1,
  "draft_preview": "# Microservices Architecture for E-commerce Platform\n\n## Core Services\n\n1. **User Management Service**...",
  "message": "Initial draft generated",
  "convergence_score": null,
  "next_step": "Continue to generate critiques",
  "estimated_time_remaining": "60-90 seconds"
}
```

```bash
# 3. Continue for critique phase
continue_refinement()
```

**Response (Critique Phase):**
```json
{
  "success": true,
  "session_id": "ref_12345678-abcd-4321-9876-fedcba098765",
  "status": "critiquing",
  "step": "critique",
  "iteration": 1,
  "critiques": [
    {
      "focus": "architectural_consistency",
      "critique": "The service boundaries could be better defined..."
    },
    {
      "focus": "scalability_patterns",
      "critique": "Consider implementing circuit breaker patterns..."
    }
  ],
  "message": "Critiques generated",
  "next_step": "Continue to generate revised version"
}
```

```bash
# 4. Continue for revision phase
continue_refinement()
```

**Response (Revision Phase):**
```json
{
  "success": true,
  "session_id": "ref_12345678-abcd-4321-9876-fedcba098765",
  "status": "revising",
  "step": "revise",
  "iteration": 1,
  "draft_preview": "# Microservices Architecture for E-commerce Platform (Revised)\n\n## Executive Summary\nThis architecture addresses scalability, resilience...",
  "convergence_score": 0.89,
  "message": "Revision complete",
  "convergence_analysis": {
    "threshold": 0.95,
    "current_score": 0.89,
    "trend": "improving",
    "needs_refinement": true
  },
  "next_step": "Continue for next iteration or get_final_result if satisfied"
}
```

```bash
# 5. Get final result (when converged or satisfied)
get_final_result(session_id="ref_12345678-abcd-4321-9876-fedcba098765")
```

**Response:**
```json
{
  "success": true,
  "session_id": "ref_12345678-abcd-4321-9876-fedcba098765",
  "final_answer": "# Comprehensive Microservices Architecture for E-commerce Platform\n\n## Executive Summary...",
  "total_iterations": 3,
  "convergence_score": 0.97,
  "refinement_summary": {
    "initial_draft_length": 1250,
    "final_answer_length": 3400,
    "improvement_areas": ["service boundaries", "data consistency", "monitoring"],
    "time_invested": "4.2 minutes"
  }
}
```

### Pattern 2: Session Management

```bash
# Check current active session
current_session()
```

**Response:**
```json
{
  "success": true,
  "session_id": "ref_12345678-abcd-4321-9876-fedcba098765",
  "status": "revising",
  "iteration": 2,
  "prompt": "Design a microservices architecture for an e-commerce platform",
  "convergence_score": 0.89,
  "estimated_completion": "1-2 more iterations"
}
```

```bash
# List all active sessions
list_refinement_sessions()
```

**Response:**
```json
{
  "success": true,
  "sessions": [
    {
      "session_id": "ref_12345678-abcd-4321-9876-fedcba098765",
      "prompt": "Design a microservices architecture...",
      "status": "revising",
      "iteration": 2,
      "started_at": "2024-01-15T10:30:00Z",
      "domain": "technical"
    },
    {
      "session_id": "ref_87654321-dcba-1234-5678-abcdef123456",
      "prompt": "Create marketing strategy for...",
      "status": "converged",
      "iteration": 4,
      "started_at": "2024-01-15T09:45:00Z",
      "domain": "marketing"
    }
  ],
  "count": 2
}
```

### Pattern 3: Abort and Recovery

```bash
# Abort refinement and get best result so far
abort_refinement(session_id="ref_12345678-abcd-4321-9876-fedcba098765")
```

**Response:**
```json
{
  "success": true,
  "session_id": "ref_12345678-abcd-4321-9876-fedcba098765",
  "final_answer": "# Microservices Architecture (Work in Progress)\n\nBased on 2 refinement iterations...",
  "status": "aborted",
  "iterations_completed": 2,
  "reason": "User requested abort",
  "convergence_score": 0.89,
  "message": "Refinement stopped. Best version so far returned."
}
```

## Domain-Specific Workflows

### Technical Documentation Refinement

```bash
# Technical domain with specialized prompts
start_refinement(
    prompt="Create a detailed API documentation template that follows OpenAPI 3.0 specifications",
    domain="technical"
)
```

**Specialized Response Features:**
- Technical terminology validation
- Code example generation
- Architecture diagram suggestions
- Security best practices integration

### Marketing Copy Improvement

```bash
# Marketing domain optimization
start_refinement(
    prompt="Write compelling copy for a SaaS product launch targeting enterprise customers",
    domain="marketing"
)
```

**Marketing-Specific Enhancements:**
- A/B testing suggestions
- Audience targeting refinement
- Conversion optimization tips
- Brand voice consistency

### Legal Document Review

```bash
# Legal domain with compliance focus
start_refinement(
    prompt="Draft a comprehensive data privacy policy for a healthcare technology platform",
    domain="legal"
)
```

**Legal Domain Features:**
- Regulatory compliance checking
- Risk assessment integration
- Precedent case references
- Jurisdiction-specific considerations

### Financial Analysis Enhancement

```bash
# Financial domain with quantitative focus
start_refinement(
    prompt="Create a financial risk assessment model for cryptocurrency investments",
    domain="financial"
)
```

**Financial-Specific Capabilities:**
- Risk metric calculations
- Regulatory compliance factors
- Market volatility analysis
- Portfolio diversification strategies

## Advanced Usage Scenarios

### Scenario 1: Long-Running Complex Analysis

```bash
# Start comprehensive analysis
start_refinement(
    prompt="Develop a complete digital transformation strategy for a traditional manufacturing company with 10,000+ employees",
    domain="strategy"
)

# Monitor progress without advancing
get_refinement_status(session_id="ref_strategy_analysis")
```

**Status Check Response:**
```json
{
  "success": true,
  "session_id": "ref_strategy_analysis",
  "status": "revising",
  "iteration": 3,
  "progress": {
    "current_step": "synthesizing stakeholder perspectives",
    "completion_percentage": 65,
    "estimated_remaining_time": "2-3 minutes"
  },
  "convergence_tracking": {
    "score": 0.91,
    "threshold": 0.95,
    "trend": "steadily improving",
    "prediction": "likely to converge in 1-2 more iterations"
  }
}
```

### Scenario 2: Parallel Session Management

```bash
# Start multiple refinements for different aspects
start_refinement(prompt="Technical architecture analysis", domain="technical")
# Session 1: tech_arch_123

start_refinement(prompt="Marketing strategy development", domain="marketing") 
# Session 2: marketing_456

start_refinement(prompt="Financial projections model", domain="financial")
# Session 3: finance_789

# Work on them alternately
continue_refinement(session_id="tech_arch_123")
continue_refinement(session_id="marketing_456")
continue_refinement(session_id="finance_789")
```

### Scenario 3: Auto-Detection and Optimization

```bash
# Let the system auto-detect domain
start_refinement(
    prompt="Create a GDPR compliance checklist for data processing activities",
    domain="auto"  # System will detect this as 'legal'
)
```

**Auto-Detection Response:**
```json
{
  "success": true,
  "session_id": "ref_auto_detection",
  "detected_domain": "legal",
  "confidence": 0.94,
  "domain_indicators": ["GDPR", "compliance", "data processing"],
  "optimization_applied": "Legal compliance validation enabled",
  "status": "initializing"
}
```

## Error Handling & Troubleshooting

### Common Error Scenarios

#### 1. No Active Session

```bash
# Trying to continue without starting
continue_refinement()
```

**Error Response:**
```json
{
  "success": false,
  "error": "No session_id provided and no current session",
  "_ai_context": {
    "current_session_id": null,
    "active_session_count": 0,
    "recent_sessions": []
  },
  "_ai_suggestion": "Use start_refinement to create a new session",
  "_ai_tip": "After start_refinement, continue_refinement will auto-track the session",
  "_human_action": "Start a new refinement session first"
}
```

**Resolution:**
```bash
# Start a new session first
start_refinement(prompt="Your question here")
```

#### 2. Invalid Session ID

```bash
# Using non-existent session ID
continue_refinement(session_id="invalid_session_123")
```

**Error Response:**
```json
{
  "success": false,
  "error": "Session not found: invalid_session_123",
  "_ai_context": {
    "active_sessions": ["ref_12345678-abcd-4321-9876-fedcba098765"],
    "session_count": 1
  },
  "_ai_suggestion": "Use list_refinement_sessions to see available sessions",
  "_ai_actions": [
    "Check session ID spelling",
    "Use current_session() to get active session",
    "Start new session if needed"
  ]
}
```

#### 3. AWS Bedrock Connectivity Issues

```bash
# When AWS credentials are invalid
start_refinement(prompt="Test prompt")
```

**Error Response:**
```json
{
  "success": false,
  "error": "AWS Bedrock connection failed",
  "_ai_diagnosis": "Invalid AWS credentials or insufficient permissions",
  "_ai_actions": [
    "Check AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables",
    "Verify Bedrock service permissions",
    "Confirm AWS region configuration"
  ],
  "_ai_suggestion": "Validate AWS configuration with aws sts get-caller-identity",
  "_human_action": "Fix AWS credentials or permissions"
}
```

#### 4. Prompt Validation Failures

```bash
# Prompt too short
start_refinement(prompt="Hi")
```

**Error Response:**
```json
{
  "success": false,
  "error": "Prompt validation failed: Prompt is too short (minimum 10 characters)",
  "_ai_diagnosis": "Input prompt doesn't meet minimum length requirements",
  "_ai_suggestion": "Provide a more detailed prompt with specific requirements",
  "_ai_example": "Instead of 'Hi', try 'Explain how to implement user authentication in a web application'",
  "_human_action": "Provide a more descriptive prompt"
}
```

#### 5. Session Timeout

```bash
# Accessing expired session
continue_refinement(session_id="expired_session_123")
```

**Error Response:**
```json
{
  "success": false,
  "error": "Session expired or timed out",
  "_ai_context": {
    "session_ttl": "1 hour",
    "last_activity": "2024-01-15T08:30:00Z",
    "current_time": "2024-01-15T10:30:00Z"
  },
  "_ai_suggestion": "Start a new refinement session",
  "_ai_tip": "Sessions automatically expire after 1 hour of inactivity",
  "_human_action": "Start a new session with the same prompt"
}
```

### Troubleshooting Guide

#### Performance Issues

```bash
# Check system health
# (Use from Claude Desktop or other MCP clients)
```

**Optimization Tips:**
- Use `CRITIQUE_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0` for 50% speed improvement
- Reduce `PARALLEL_CRITIQUES` from 3 to 2 for faster processing
- Lower `CONVERGENCE_THRESHOLD` from 0.98 to 0.95 for quicker convergence

#### Memory and Resource Management

**Monitor Session Usage:**
```bash
list_refinement_sessions()
```

**Clean Up Completed Sessions:**
Sessions automatically expire after 1 hour, but you can manually abort unnecessary ones:
```bash
abort_refinement(session_id="unnecessary_session_id")
```

## Claude Desktop Integration

### Configuration Examples

#### Basic Configuration
Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "recursive-companion": {
      "command": "/path/to/recursive-companion-mcp/run.sh",
      "env": {
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "your-access-key",
        "AWS_SECRET_ACCESS_KEY": "your-secret-key"
      }
    }
  }
}
```

#### Optimized Configuration
```json
{
  "mcpServers": {
    "recursive-companion": {
      "command": "/path/to/recursive-companion-mcp/run.sh",
      "env": {
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "your-access-key",
        "AWS_SECRET_ACCESS_KEY": "your-secret-key",
        "BEDROCK_MODEL_ID": "anthropic.claude-3-sonnet-20240229-v1:0",
        "CRITIQUE_MODEL_ID": "anthropic.claude-3-haiku-20240307-v1:0",
        "CONVERGENCE_THRESHOLD": "0.95",
        "PARALLEL_CRITIQUES": "2",
        "MAX_ITERATIONS": "5",
        "REQUEST_TIMEOUT": "600"
      }
    }
  }
}
```

### Claude Desktop Usage Patterns

#### Interactive Refinement
```
User: I need help creating a comprehensive security policy for our startup.

Claude: I'll help you create a comprehensive security policy using iterative refinement to ensure we cover all critical aspects. Let me start the process:

start_refinement(
    prompt="Create a comprehensive information security policy for a tech startup with 50 employees, covering data protection, access controls, incident response, and compliance requirements",
    domain="legal"
)

[Shows initial session response]

Now let me continue the refinement process:

continue_refinement()

[Shows draft generation progress]

Let me continue to the critique phase:

continue_refinement()

[Shows critique analysis]

And now for the revision:

continue_refinement()

[Shows improved version with convergence score]

The convergence score is 0.92, which is close to our threshold of 0.95. Let me do one more iteration:

continue_refinement()

Perfect! We've achieved convergence. Let me get the final result:

get_final_result()

[Shows comprehensive security policy]
```

#### Quick Tasks
```
User: Can you quickly improve this email draft?

Claude: I'll use the quick refinement mode for this:

quick_refine(
    prompt="Improve this email draft for professional clarity and persuasiveness: [email content]",
    max_wait=60
)

[Returns refined email in under a minute]
```

## Integration with Other Thinkerz MCP Servers

### Multi-Perspective Analysis Integration

```bash
# 1. Start with context-switcher for multi-perspective analysis
# (Using context-switcher-mcp)
start_context_analysis(topic="API security strategy")
analyze_from_perspectives(session_id="context_123", prompt="Evaluate security approaches")

# 2. Use recursive-companion to refine each perspective
start_refinement(
    prompt="Based on technical perspective: Implement comprehensive API security with OAuth 2.0, rate limiting, and input validation",
    domain="technical"
)

# 3. Continue refinement for other perspectives
start_refinement(
    prompt="Based on business perspective: Balance security investment with user experience and development velocity",
    domain="strategy"
)
```

### Devil's Advocate Integration

```bash
# 1. Create initial strategy with recursive-companion
start_refinement(
    prompt="Develop a cloud migration strategy for legacy enterprise applications",
    domain="technical"
)

# Get the refined result
get_final_result(session_id="migration_strategy")

# 2. Challenge the strategy with devil's advocate
# (Using devil-advocate-mcp)
start_adversarial_analysis(
    idea="Cloud migration strategy: [refined strategy]",
    context="Enterprise with 500+ legacy applications, $10M budget, 18-month timeline",
    stakes="Business continuity, competitive advantage, cost reduction targets"
)

# 3. Refine strategy based on adversarial feedback
start_refinement(
    prompt="Improve cloud migration strategy addressing these concerns: [adversarial challenges]",
    domain="technical"
)
```

### Decision Matrix Integration

```bash
# 1. Generate options with recursive-companion
quick_refine(
    prompt="Generate 3 viable database architecture options for a high-traffic e-commerce platform",
    max_wait=90
)

# 2. Evaluate options systematically with decision-matrix
# (Using decision-matrix-mcp)
start_decision_analysis(
    topic="Database architecture selection",
    options=["PostgreSQL with read replicas", "MongoDB with sharding", "Distributed SQL (CockroachDB)"]
)

# 3. Refine the winning option with recursive-companion
start_refinement(
    prompt="Create detailed implementation plan for [winning database option] including migration strategy, monitoring, and scaling considerations",
    domain="technical"
)
```

### Rubber Duck Integration

```bash
# 1. Use rubber duck for initial problem exploration
# (Using rubber-duck-mcp)
start_reflection(
    topic="Our microservices are becoming tightly coupled and hard to maintain",
    style="analytical"
)

# Continue reflection until insights emerge
reflect(session_id="rubber_duck_session", thought="The real issue might be our service boundaries")
get_insights(session_id="rubber_duck_session")

# 2. Use recursive-companion to develop solution
start_refinement(
    prompt="Design a microservices refactoring strategy to reduce coupling and improve maintainability, based on these insights: [rubber duck insights]",
    domain="technical"
)
```

## Best Practices

### 1. Domain Selection Strategy
- Use `"auto"` for general content - the system is quite good at detection
- Explicitly set domain when you need specialized terminology or compliance features
- `"technical"` for code, architecture, engineering content
- `"legal"` for compliance, contracts, policy documents
- `"financial"` for quantitative analysis, risk assessment
- `"marketing"` for copy, messaging, audience targeting

### 2. Session Management
- Use `current_session()` to check active session status
- Clean up with `abort_refinement()` when changing topics
- Monitor `list_refinement_sessions()` to avoid too many concurrent sessions

### 3. Performance Optimization
- Use `quick_refine()` for simple improvements (< 60 seconds)
- Use step-by-step refinement for complex, multi-faceted content
- Set appropriate `max_wait` times based on content complexity
- Consider using Haiku for critiques to improve speed

### 4. Quality Assurance
- Monitor convergence scores - aim for 0.95+ for critical content
- Review critiques to understand improvement areas
- Use multiple iterations for high-stakes content
- Save important results immediately after completion

### 5. Error Recovery
- Always check `success` field in responses
- Use `_ai_suggestion` fields for automated recovery
- Implement retry logic for transient AWS issues
- Have fallback strategies for timeout scenarios

This comprehensive guide should significantly improve your experience with the Recursive Companion MCP server. For additional support or advanced use cases, refer to the main documentation or submit an issue on the project repository.
