"""
Domain-specific configuration for Recursive Companion MCP Server
Consolidates domain detection and domain-specific prompts
"""

import re
from typing import Dict, List


# Domain detection keywords
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "technical": [
        "code",
        "algorithm",
        "api",
        "debug",
        "performance",
        "architecture",
        "system",
        "database",
        "security",
        "function",
        "python",
        "javascript",
        "tcp",
        "ip",
        "protocol",
        "optimize",
        "queries",
        "indexes",
        "rest",
        "authentication",
        "async",
        "await",
        "implement",
        "binary",
        "search",
        "stack",
        "software",
    ],
    "marketing": [
        "marketing",
        "campaign",
        "audience",
        "brand",
        "conversion",
        "engagement",
        "market",
        "customer",
        "segment",
        "targeting",
        "positioning",
        "messaging",
        "growth",
        "acquisition",
        "social media",
        "ad",
        "copy",
        "facebook",
        "email",
        "launch",
        "compelling",
    ],
    "strategy": [
        "goal",
        "objective",
        "plan",
        "roadmap",
        "vision",
        "competitive",
        "analysis",
        "swot",
        "strategic",
        "long-term",
        "initiative",
        "milestone",
        "kpi",
        "metric",
    ],
    "legal": [
        "contract",
        "compliance",
        "regulation",
        "liability",
        "agreement",
        "terms",
        "privacy",
        "gdpr",
        "legal",
        "law",
        "policy",
        "dispute",
        "intellectual property",
        "copyright",
        "trademark",
        "non-disclosure",
        "nda",
        "service",
        "saas",
    ],
    "financial": [
        "revenue",
        "cost",
        "budget",
        "forecast",
        "investment",
        "profit",
        "cash flow",
        "valuation",
        "financial",
        "accounting",
        "balance sheet",
        "income",
        "expense",
        "roi",
        "npv",
        "irr",
        "calculate",
        "portfolio",
        "stock",
        "allocation",
        "statements",
    ],
}


# Domain-specific system prompts
DOMAIN_PROMPTS: Dict[str, str] = {
    "technical": (
        "You are a technical expert. Focus on accuracy, best practices, "
        "and clear technical explanations."
    ),
    "marketing": (
        "You are a marketing strategist. Focus on audience engagement, " "brand messaging, and ROI."
    ),
    "strategy": (
        "You are a strategic advisor. Focus on long-term vision, "
        "competitive advantage, and actionable insights."
    ),
    "legal": (
        "You are a legal expert. Focus on compliance, risk mitigation, "
        "and precise legal language."
    ),
    "financial": (
        "You are a financial analyst. Focus on quantitative analysis, "
        "financial metrics, and data-driven insights."
    ),
    "general": (
        "You are a helpful assistant. Provide clear, accurate, " "and well-structured responses."
    ),
}


class DomainDetector:
    """Detects the appropriate domain for a given prompt"""

    @staticmethod
    def detect_domain(prompt: str) -> str:
        """Auto-detect domain based on keywords and patterns"""
        prompt_lower = prompt.lower()
        domain_scores = {}

        for domain, keywords in DOMAIN_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                # Use word boundaries for single words, exact match for phrases
                if " " in keyword:
                    # Multi-word phrase - exact match
                    if keyword in prompt_lower:
                        score += 1
                else:
                    # Single word - use word boundaries
                    pattern = r"\b" + re.escape(keyword) + r"\b"
                    if re.search(pattern, prompt_lower):
                        score += 1
            if score > 0:
                domain_scores[domain] = score

        if not domain_scores:
            return "general"

        # Return domain with highest score
        return max(domain_scores, key=domain_scores.get)


def get_domain_system_prompt(domain: str) -> str:
    """Get domain-specific system prompt"""
    return DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["general"])
