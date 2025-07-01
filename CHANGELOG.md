# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2025-07-01

### Added
- Initial release of Recursive Companion MCP
- Incremental refinement engine with session management
- Support for 5 MCP tools: start_refinement, continue_refinement, get_refinement_status, get_final_result, list_refinement_sessions
- Domain auto-detection for technical, marketing, strategy, legal, financial, and general content
- Mathematical convergence measurement using cosine similarity
- Parallel critique generation for improved performance
- Configurable convergence thresholds and iteration limits
- AWS Bedrock integration with Claude and Titan embeddings
- Session-based approach to avoid MCP timeouts
- Progress visibility for better user experience
- Support for using different models for critiques vs main generation

### Security
- Input validation with length limits and pattern detection
- Request timeout protection
- Comprehensive error handling

### Performance
- Parallel critique generation reduces iteration time
- Caching for embeddings
- Support for using faster models (Haiku) for critiques
- Async/await throughout for non-blocking operations
