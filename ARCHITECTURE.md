# Recursive Companion MCP - Architecture Documentation

This document provides comprehensive architecture diagrams and explanations for the recursive-companion-mcp system, designed to help new contributors understand the system design and implementation patterns.

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Refinement Process Flow](#refinement-process-flow)
3. [Component Interaction Diagram](#component-interaction-diagram)
4. [Key Architectural Patterns](#key-architectural-patterns)
5. [Security Architecture](#security-architecture)
6. [Performance Considerations](#performance-considerations)

## System Architecture Overview

The recursive-companion-mcp is an MCP (Model Context Protocol) server that implements iterative refinement through self-critique cycles. It's designed as a production-ready cognitive AI system with robust session management, AWS Bedrock integration, and mathematical convergence detection.

```mermaid
graph TB
    subgraph "External Interfaces"
        Claude[Claude Desktop Client]
        AWS[AWS Bedrock Service]
        FS[File System Storage]
    end
    
    subgraph "MCP Server Layer"
        MCP[MCP Protocol Handler]
        Tools[MCP Tools Interface]
        ErrorHandler[AI-Friendly Error Handler]
    end
    
    subgraph "Core Engine Layer"
        IE[Incremental Engine]
        SM[Session Manager]
        RE[Refine Engine]
        CD[Convergence Detector]
    end
    
    subgraph "AWS Integration Layer"
        BC[Bedrock Client]
        CB[Circuit Breaker]
        RL[Rate Limiter]
    end
    
    subgraph "Domain & Security Layer"
        DD[Domain Detector]
        SV[Security Validator]
        SP[Session Persistence]
    end
    
    subgraph "Session Storage"
        MS[Memory Sessions]
        PS[Persistent Storage]
        TTL[TTL Cleanup]
    end
    
    Claude --> MCP
    MCP --> Tools
    Tools --> IE
    IE --> SM
    IE --> RE
    IE --> CD
    RE --> BC
    BC --> CB
    BC --> RL
    BC --> AWS
    IE --> DD
    IE --> SV
    SM --> SP
    SP --> MS
    SP --> PS
    SP --> TTL
    SP --> FS
    
    ErrorHandler --> Claude
    
    classDef external fill:#e1f5fe
    classDef server fill:#f3e5f5
    classDef core fill:#e8f5e8
    classDef aws fill:#fff3e0
    classDef security fill:#ffebee
    classDef storage fill:#f1f8e9
    
    class Claude,AWS,FS external
    class MCP,Tools,ErrorHandler server
    class IE,SM,RE,CD core
    class BC,CB,RL aws
    class DD,SV,SP security
    class MS,PS,TTL storage
```

### System Components Explanation

**External Interfaces:**
- **Claude Desktop Client**: User interface that communicates via MCP protocol
- **AWS Bedrock Service**: Provides Claude and other LLM models for generation and critique
- **File System Storage**: Persistent storage for session data and configuration

**MCP Server Layer:**
- **MCP Protocol Handler**: Manages MCP communication protocol with Claude Desktop
- **MCP Tools Interface**: Exposes refinement tools (start_refinement, continue_refinement, etc.)
- **AI-Friendly Error Handler**: Provides structured error responses with diagnostic hints for AI assistants

**Core Engine Layer:**
- **Incremental Engine**: Orchestrates the refinement process with incremental steps to avoid timeouts
- **Session Manager**: Manages active sessions with persistence and TTL cleanup
- **Refine Engine**: Implements the core Draft → Critique → Revise logic
- **Convergence Detector**: Uses cosine similarity to measure when refinement is complete

**AWS Integration Layer:**
- **Bedrock Client**: Abstracts AWS Bedrock API interactions
- **Circuit Breaker**: Prevents cascade failures and provides graceful degradation
- **Rate Limiter**: Controls request frequency to respect AWS service limits

**Domain & Security Layer:**
- **Domain Detector**: Auto-detects content domain (technical, marketing, legal, financial) for optimized prompts
- **Security Validator**: Validates inputs and prevents injection attacks
- **Session Persistence**: Manages session serialization and storage

## Refinement Process Flow

The core refinement process follows a **Draft → Critique → Revise → Converge** pattern with incremental execution to prevent timeouts and enable progress visibility.

```mermaid
stateDiagram-v2
    [*] --> INITIALIZING: start_refinement()
    
    INITIALIZING --> DRAFTING: Session Created
    DRAFTING --> CRITIQUING: Initial Draft Complete
    CRITIQUING --> REVISING: Critiques Generated
    REVISING --> ConvergenceCheck: Revision Complete
    
    ConvergenceCheck --> CONVERGED: Similarity >= Threshold
    ConvergenceCheck --> DRAFTING: Similarity < Threshold & Iterations < Max
    ConvergenceCheck --> CONVERGED: Max Iterations Reached
    
    DRAFTING --> ERROR: Generation Failure
    CRITIQUING --> ERROR: Critique Failure
    REVISING --> ERROR: Revision Failure
    
    ERROR --> ABORTED: abort_refinement()
    CONVERGED --> [*]: Session Complete
    ABORTED --> [*]: Session Terminated
    
    note right of CRITIQUING: Parallel critique generation\nusing faster models (e.g., Haiku)
    note right of ConvergenceCheck: Cosine similarity measurement\nbetween current and previous draft
```

### Process Flow Explanation

**Session States:**
1. **INITIALIZING**: Session setup with domain detection and configuration
2. **DRAFTING**: Generate initial response or iterate on previous revision
3. **CRITIQUING**: Create multiple parallel critiques from different perspectives
4. **REVISING**: Synthesize critiques into an improved version
5. **CONVERGED**: Mathematical similarity threshold reached or max iterations hit
6. **ERROR**: Recoverable error state with diagnostic information
7. **ABORTED**: User-terminated or unrecoverable error

**Key Process Features:**
- **Incremental Execution**: Each state change returns immediately to prevent timeouts
- **Parallel Critiques**: Multiple critique perspectives generated concurrently for speed
- **Mathematical Convergence**: Uses cosine similarity on embeddings to detect when content stabilizes
- **Domain Optimization**: Auto-detects content type and applies specialized system prompts
- **Progress Visibility**: Real-time status updates for UI integration

**Convergence Detection:**
```mermaid
graph LR
    PD[Previous Draft] --> E1[Embeddings Generator]
    CD[Current Draft] --> E2[Embeddings Generator]
    E1 --> CS[Cosine Similarity]
    E2 --> CS
    CS --> T{Similarity >= Threshold?}
    T -->|Yes| CONV[CONVERGED]
    T -->|No| CONT[Continue Refinement]
    
    subgraph "Threshold Configuration"
        TH1[0.90 - Low Precision]
        TH2[0.95 - Balanced Default]
        TH3[0.98 - High Precision]
    end
```

## Component Interaction Diagram

This diagram shows detailed interactions between core modules, including data flows, security validation, and error handling patterns.

```mermaid
graph TB
    subgraph "MCP Handler Layer"
        MCPTools[MCP Tools]
        start_refinement[start_refinement]
        continue_refinement[continue_refinement]
        get_final_result[get_final_result]
        quick_refine[quick_refine]
    end
    
    subgraph "Validation & Security"
        InputVal[Input Validation]
        SecCheck[Security Validation]
        RateLimit[Rate Limiting]
    end
    
    subgraph "Session Management"
        SessionTracker[Session Tracker]
        SessionManager[Session Manager]
        Persistence[Session Persistence]
        MemCache[Memory Cache]
        FileStore[File Storage]
    end
    
    subgraph "Core Processing"
        IncrementalEngine[Incremental Engine]
        RefineEngine[Refine Engine]
        DomainDetector[Domain Detector]
        ConvergenceDetector[Convergence Detector]
    end
    
    subgraph "AWS Bedrock Integration"
        BedrockClient[Bedrock Client]
        ModelRouter[Model Router]
        CircuitBreaker[Circuit Breaker]
        RetryLogic[Retry Logic]
    end
    
    subgraph "Error Handling"
        ErrorHandler[Error Handler]
        DiagnosticGen[Diagnostic Generator]
        AIHints[AI-Friendly Hints]
    end
    
    MCPTools --> start_refinement
    MCPTools --> continue_refinement
    MCPTools --> get_final_result
    MCPTools --> quick_refine
    
    start_refinement --> InputVal
    continue_refinement --> InputVal
    InputVal --> SecCheck
    SecCheck --> RateLimit
    RateLimit --> SessionTracker
    
    SessionTracker --> SessionManager
    SessionManager --> Persistence
    Persistence --> MemCache
    Persistence --> FileStore
    
    SessionTracker --> IncrementalEngine
    IncrementalEngine --> RefineEngine
    IncrementalEngine --> DomainDetector
    IncrementalEngine --> ConvergenceDetector
    
    RefineEngine --> BedrockClient
    BedrockClient --> ModelRouter
    ModelRouter --> CircuitBreaker
    CircuitBreaker --> RetryLogic
    
    IncrementalEngine --> ErrorHandler
    ErrorHandler --> DiagnosticGen
    DiagnosticGen --> AIHints
    
    classDef mcp fill:#e3f2fd
    classDef validation fill:#ffebee
    classDef session fill:#e8f5e8
    classDef processing fill:#f3e5f5
    classDef aws fill:#fff3e0
    classDef error fill:#fce4ec
    
    class MCPTools,start_refinement,continue_refinement,get_final_result,quick_refine mcp
    class InputVal,SecCheck,RateLimit validation
    class SessionTracker,SessionManager,Persistence,MemCache,FileStore session
    class IncrementalEngine,RefineEngine,DomainDetector,ConvergenceDetector processing
    class BedrockClient,ModelRouter,CircuitBreaker,RetryLogic aws
    class ErrorHandler,DiagnosticGen,AIHints error
```

### Component Interaction Details

**Request Flow:**
1. **MCP Tools** receive requests from Claude Desktop
2. **Input Validation** ensures request format and parameter validity
3. **Security Validation** prevents injection attacks and validates content
4. **Rate Limiting** enforces AWS service limits and prevents abuse
5. **Session Tracker** manages session lifecycle and auto-tracking

**Processing Flow:**
1. **Session Manager** handles persistence and retrieval
2. **Incremental Engine** orchestrates refinement steps
3. **Domain Detector** analyzes content and applies specialized prompts
4. **Refine Engine** implements core refinement logic
5. **Convergence Detector** measures similarity using embeddings

**AWS Integration:**
1. **Bedrock Client** manages AWS API interactions
2. **Model Router** selects appropriate models (e.g., Sonnet for generation, Haiku for critiques)
3. **Circuit Breaker** prevents cascade failures
4. **Retry Logic** handles transient failures gracefully

## Key Architectural Patterns

### 1. Incremental Processing Pattern
```mermaid
sequenceDiagram
    participant Client
    participant MCP
    participant Engine
    participant AWS
    
    Client->>MCP: start_refinement(prompt)
    MCP->>Engine: create_session()
    Engine->>AWS: generate_draft()
    AWS-->>Engine: draft_content
    Engine-->>MCP: status: DRAFTING
    MCP-->>Client: session_id + progress
    
    Client->>MCP: continue_refinement(session_id)
    MCP->>Engine: advance_session()
    Engine->>AWS: generate_critiques()
    AWS-->>Engine: critique_list
    Engine-->>MCP: status: CRITIQUING
    MCP-->>Client: progress update
    
    Note over Client,AWS: Process continues incrementally<br/>until convergence
```

### 2. Session Management Pattern
```mermaid
graph LR
    subgraph "Session Lifecycle"
        Create[Create Session]
        Persist[Persist State]
        Track[Track Changes]
        Cleanup[TTL Cleanup]
    end
    
    subgraph "Storage Strategy"
        Memory[Memory Cache]
        Disk[Disk Persistence]
        Backup[Backup Strategy]
    end
    
    Create --> Persist
    Persist --> Track
    Track --> Cleanup
    
    Persist --> Memory
    Persist --> Disk
    Disk --> Backup
```

### 3. Circuit Breaker Pattern
```mermaid
stateDiagram-v2
    [*] --> CLOSED: Normal Operation
    CLOSED --> OPEN: Failure Threshold Reached
    OPEN --> HALF_OPEN: Timeout Expired
    HALF_OPEN --> CLOSED: Test Request Success
    HALF_OPEN --> OPEN: Test Request Failure
    
    note right of OPEN: All requests fail fast<br/>No AWS API calls
    note right of HALF_OPEN: Limited test requests<br/>Gradual recovery
```

## Security Architecture

The system implements defense-in-depth security with multiple validation layers:

```mermaid
graph TB
    subgraph "Security Layers"
        L1[Input Validation Layer]
        L2[Content Security Layer]
        L3[Session Security Layer]
        L4[AWS Security Layer]
    end
    
    subgraph "Validation Components"
        InputVal[Input Sanitization]
        ContentFilter[Content Filtering]
        InjectionPrev[Injection Prevention]
        SessionVal[Session Validation]
    end
    
    subgraph "AWS Security"
        IAM[IAM Policies]
        VPC[VPC Isolation]
        Encryption[Encryption at Rest/Transit]
        Audit[CloudTrail Logging]
    end
    
    L1 --> InputVal
    L1 --> InjectionPrev
    L2 --> ContentFilter
    L3 --> SessionVal
    L4 --> IAM
    L4 --> VPC
    L4 --> Encryption
    L4 --> Audit
    
    classDef security fill:#ffebee
    class L1,L2,L3,L4,InputVal,ContentFilter,InjectionPrev,SessionVal,IAM,VPC,Encryption,Audit security
```

**Security Features:**
- **Input Sanitization**: Prevents malicious content injection
- **Content Filtering**: Blocks inappropriate or dangerous content
- **Session Validation**: Ensures session integrity and ownership
- **AWS IAM**: Minimal privilege access controls
- **Encryption**: All data encrypted in transit and at rest
- **Audit Logging**: Comprehensive security event logging

## Performance Considerations

### Scalability Patterns

```mermaid
graph TB
    subgraph "Performance Optimizations"
        PC[Parallel Critiques]
        MS[Model Selection]
        CB[Circuit Breakers]
        CA[Caching Strategy]
    end
    
    subgraph "Resource Management"
        SM[Session Memory]
        TTL[TTL Cleanup]
        RL[Rate Limiting]
        RP[Resource Pooling]
    end
    
    subgraph "AWS Optimization"
        MR[Model Routing]
        TO[Timeout Management]
        RR[Request Routing]
        CS[Cost Optimization]
    end
    
    PC --> MS
    MS --> CB
    CB --> CA
    
    SM --> TTL
    TTL --> RL
    RL --> RP
    
    MR --> TO
    TO --> RR
    RR --> CS
```

**Performance Features:**
- **Parallel Processing**: Concurrent critique generation reduces latency
- **Model Selection**: Use faster models (Haiku) for critiques, premium models (Sonnet) for generation
- **Intelligent Caching**: Session state and convergence calculations cached
- **Resource Pooling**: Efficient AWS connection management
- **Timeout Management**: Incremental processing prevents timeouts
- **Cost Optimization**: Smart model routing based on task complexity

### Monitoring and Observability

```mermaid
graph LR
    subgraph "Metrics Collection"
        SessionMetrics[Session Metrics]
        PerformanceMetrics[Performance Metrics]
        ErrorMetrics[Error Metrics]
        CostMetrics[Cost Metrics]
    end
    
    subgraph "Health Monitoring"
        HealthChecks[Health Checks]
        CircuitBreakerStatus[Circuit Breaker Status]
        ResourceUsage[Resource Usage]
        AWSStatus[AWS Service Status]
    end
    
    SessionMetrics --> Dashboard[Monitoring Dashboard]
    PerformanceMetrics --> Dashboard
    ErrorMetrics --> Dashboard
    CostMetrics --> Dashboard
    
    HealthChecks --> Alerts[Alert System]
    CircuitBreakerStatus --> Alerts
    ResourceUsage --> Alerts
    AWSStatus --> Alerts
```

This architecture documentation provides a comprehensive view of the recursive-companion-mcp system, designed to help contributors understand the sophisticated cognitive AI platform and its production-ready implementation patterns.
