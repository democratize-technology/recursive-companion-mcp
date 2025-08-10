#!/usr/bin/env python3
"""
Recursive Companion MCP Server - AWS Bedrock Edition
Based on Hank Besser's recursive-companion: https://github.com/hankbesser/recursive-companion
Implements iterative refinement through Draft → Critique → Revise → Converge cycles
"""
import asyncio
import json
import logging
import time

import boto3
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from bedrock_client import BedrockClient
from config import config
from domains import DomainDetector
from error_handling import create_ai_error_response
from incremental_engine import IncrementalRefineEngine
from refine_engine import RefineEngine
from session_manager import SessionTracker
from validation import SecurityValidator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

server = Server("recursive-companion")

# Session tracking
session_tracker = SessionTracker()

# Error handling is now in error_handling.py

# Initialize global instances
bedrock_client = None
refine_engine = None
incremental_engine = None


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="start_refinement",
            description=(
                "Start a new incremental refinement session. "
                "Returns immediately with a session ID. "
                "Use continue_refinement to proceed step by step."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The question or task to refine"},
                    "domain": {
                        "type": "string",
                        "enum": [
                            "auto",
                            "technical",
                            "marketing",
                            "strategy",
                            "legal",
                            "financial",
                            "general",
                        ],
                        "default": "auto",
                    },
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="continue_refinement",
            description=(
                "Continue an active refinement session by one step. "
                "Each call performs one action: draft, critique, or revise. "
                "If no session_id provided, continues the current session."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": (
                            "The refinement session ID "
                            "(optional, uses current if not provided)"
                        ),
                    }
                },
            },
        ),
        Tool(
            name="get_refinement_status",
            description="Get the current status of a refinement session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "The refinement session ID"}
                },
                "required": ["session_id"],
            },
        ),
        Tool(
            name="get_final_result",
            description="Get the final refined answer once convergence is achieved.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "The refinement session ID"}
                },
                "required": ["session_id"],
            },
        ),
        Tool(
            name="list_refinement_sessions",
            description="List all active refinement sessions.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="current_session",
            description="Get the current refinement session status without needing the ID",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="abort_refinement",
            description="Stop refinement and get the best result so far",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Optional: specific session (uses current if not provided)",
                    }
                },
            },
        ),
        Tool(
            name="quick_refine",
            description=(
                "Start and auto-continue a refinement until complete. "
                "Best for simple refinements that don't need step-by-step control."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The question to refine"},
                    "max_wait": {
                        "type": "number",
                        "default": 30,
                        "description": "Max seconds to wait",
                    },
                },
                "required": ["prompt"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    global incremental_engine

    if name == "start_refinement":
        try:
            if not incremental_engine:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": "Incremental engine not initialized", "success": False},
                            indent=2,
                        ),
                    )
                ]

            prompt = arguments.get("prompt", "")
            domain = arguments.get("domain", "auto")

            result = await incremental_engine.start_refinement(prompt, domain)

            # Track current session for better UX
            if result.get("success"):
                session_tracker.set_current_session(result["session_id"], prompt)

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            logger.error(f"Start refinement error: {e}")
            error_response = create_ai_error_response(e, "start_refinement")
            error_response["_ai_hint"] = "This is usually a validation or AWS connection issue"
            return [TextContent(type="text", text=json.dumps(error_response, indent=2))]

    elif name == "continue_refinement":
        try:
            if not incremental_engine:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": "Incremental engine not initialized", "success": False},
                            indent=2,
                        ),
                    )
                ]

            session_id = arguments.get("session_id", session_tracker.get_current_session())

            if not session_id:
                active_sessions = (
                    incremental_engine.session_manager.list_active_sessions()
                    if incremental_engine
                    else []
                )
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "success": False,
                                "error": "No session_id provided and no current session",
                                "_ai_context": {
                                    "current_session_id": session_tracker.get_current_session(),
                                    "active_session_count": len(active_sessions),
                                    "recent_sessions": (
                                        active_sessions[:2] if active_sessions else []
                                    ),
                                },
                                "_ai_suggestion": "Use start_refinement to create a new session",
                                "_ai_tip": (
                                    "After start_refinement, continue_refinement "
                                    "will auto-track the session"
                                ),
                                "_human_action": "Start a new refinement session first",
                            },
                            indent=2,
                        ),
                    )
                ]

            result = await incremental_engine.continue_refinement(session_id)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            logger.error(f"Continue refinement error: {e}")
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to continue refinement: {str(e)}", "success": False},
                        indent=2,
                    ),
                )
            ]

    elif name == "get_refinement_status":
        try:
            if not incremental_engine:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": "Incremental engine not initialized", "success": False},
                            indent=2,
                        ),
                    )
                ]

            session_id = arguments.get("session_id", "")

            result = await incremental_engine.get_status(session_id)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            logger.error(f"Get status error: {e}")
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get status: {str(e)}", "success": False}, indent=2
                    ),
                )
            ]

    elif name == "get_final_result":
        try:
            if not incremental_engine:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": "Incremental engine not initialized", "success": False},
                            indent=2,
                        ),
                    )
                ]

            session_id = arguments.get("session_id", "")

            result = await incremental_engine.get_final_result(session_id)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            logger.error(f"Get final result error: {e}")
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get final result: {str(e)}", "success": False},
                        indent=2,
                    ),
                )
            ]

    elif name == "list_refinement_sessions":
        try:
            if not incremental_engine:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": "Incremental engine not initialized", "success": False},
                            indent=2,
                        ),
                    )
                ]

            sessions = incremental_engine.session_manager.list_active_sessions()
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {"success": True, "sessions": sessions, "count": len(sessions)}, indent=2
                    ),
                )
            ]

        except Exception as e:
            logger.error(f"List sessions error: {e}")
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to list sessions: {str(e)}", "success": False}, indent=2
                    ),
                )
            ]

    elif name == "current_session":
        current_session_id = session_tracker.get_current_session()
        if not current_session_id:
            # Try to find the most recent session
            if incremental_engine:
                sessions = incremental_engine.session_manager.list_active_sessions()
                if sessions:
                    recent = sessions[0]
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(
                                {
                                    "success": True,
                                    "message": "No current session set, showing most recent",
                                    "session": recent,
                                },
                                indent=2,
                            ),
                        )
                    ]
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "success": False,
                            "message": "No active sessions. Start one with start_refinement.",
                        },
                        indent=2,
                    ),
                )
            ]

        try:
            result = await incremental_engine.get_status(current_session_id)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {"success": False, "error": f"Failed to get current session: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "abort_refinement":
        try:
            if not incremental_engine:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": "Incremental engine not initialized", "success": False},
                            indent=2,
                        ),
                    )
                ]

            session_id = arguments.get("session_id", session_tracker.get_current_session())
            if not session_id:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "success": False,
                                "error": "No session specified and no current session active",
                            },
                            indent=2,
                        ),
                    )
                ]

            result = await incremental_engine.abort_refinement(session_id)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            logger.error(f"Abort refinement error: {e}")
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to abort refinement: {str(e)}", "success": False},
                        indent=2,
                    ),
                )
            ]

    elif name == "quick_refine":
        try:
            if not incremental_engine:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": "Incremental engine not initialized", "success": False},
                            indent=2,
                        ),
                    )
                ]

            prompt = arguments.get("prompt", "")
            max_wait = arguments.get("max_wait", 30)

            # Start refinement
            start_result = await incremental_engine.start_refinement(prompt)
            if not start_result.get("success"):
                return [TextContent(type="text", text=json.dumps(start_result, indent=2))]

            session_id = start_result["session_id"]
            session_tracker.set_current_session(session_id, prompt)

            # Auto-continue until done or timeout
            start_time = time.time()
            iterations = 0
            last_preview = ""

            while (time.time() - start_time) < max_wait:
                continue_result = await incremental_engine.continue_refinement(session_id)
                iterations += 1

                # Track the latest draft preview
                if continue_result.get("draft_preview"):
                    last_preview = continue_result["draft_preview"]
                    logger.debug(
                        f"Quick refine iteration {iterations}: preview length {len(last_preview)}"
                    )

                if continue_result.get("status") in ["completed", "converged"]:
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(
                                {
                                    "success": True,
                                    "final_answer": continue_result.get("final_answer", ""),
                                    "iterations": iterations,
                                    "time_taken": round(time.time() - start_time, 1),
                                    "convergence_score": continue_result.get(
                                        "convergence_score", 0
                                    ),
                                },
                                indent=2,
                            ),
                        )
                    ]

                await asyncio.sleep(0.1)  # Small delay between steps

            # Timeout - abort and return best so far
            abort_result = await incremental_engine.abort_refinement(session_id)
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "success": True,
                            "status": "timeout",
                            "message": f"Stopped after {max_wait}s",
                            "final_answer": abort_result.get("final_answer", last_preview),
                            "iterations": iterations,
                            "_ai_note": "Refinement was progressing but hit time limit",
                            "_ai_suggestion": "Increase max_wait for more complete results",
                        },
                        indent=2,
                    ),
                )
            ]

        except Exception as e:
            logger.error(f"Quick refine error: {e}")
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to quick refine: {str(e)}", "success": False}, indent=2
                    ),
                )
            ]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    """Main entry point"""
    global bedrock_client, refine_engine, incremental_engine

    try:
        # Initialize Bedrock client
        bedrock_client = BedrockClient()
        refine_engine = RefineEngine(bedrock_client)

        # Initialize incremental engine
        incremental_engine = IncrementalRefineEngine(
            bedrock_client, DomainDetector(), SecurityValidator()
        )

        # Test Bedrock connection
        bedrock_test = boto3.client(service_name="bedrock", region_name=config.aws_region)
        bedrock_test.list_foundation_models()
        logger.info("Successfully connected to AWS Bedrock")
        logger.info(f"Using Claude model: {config.bedrock_model_id}")
        logger.info(f"Using embedding model: {config.embedding_model_id}")

        logger.info("Starting Recursive Companion MCP server")
        logger.info(
            (
                f"Configuration: max_iterations={config.max_iterations}, "
                f"convergence_threshold={config.convergence_threshold}"
            )
        )

        async with stdio_server() as streams:
            await server.run(streams[0], streams[1], server.create_initialization_options())
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
