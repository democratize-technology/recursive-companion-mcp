"""
Tests for Chain of Thought (CoT) enhancement functionality
"""

import pytest
from src.cot_enhancement import CoTEnhancer, CoTConfig, CoTMode, create_cot_enhancer


class TestCoTEnhancer:
    """Test CoT enhancement functionality"""

    def test_default_initialization(self):
        """Test CoT enhancer initializes with defaults"""
        enhancer = CoTEnhancer()
        assert enhancer.config.enabled is True
        assert enhancer.config.mode == CoTMode.STRUCTURED
        assert enhancer.config.include_metacognition is True

    def test_custom_config_initialization(self):
        """Test CoT enhancer with custom configuration"""
        config = CoTConfig(
            enabled=False,
            mode=CoTMode.BASIC,
            include_metacognition=False
        )
        enhancer = CoTEnhancer(config)
        assert enhancer.config.enabled is False
        assert enhancer.config.mode == CoTMode.BASIC
        assert enhancer.config.include_metacognition is False

    def test_enhance_prompt_disabled(self):
        """Test that disabled CoT returns original prompt"""
        config = CoTConfig(enabled=False)
        enhancer = CoTEnhancer(config)
        
        original = "What is the capital of France?"
        enhanced = enhancer.enhance_prompt(original)
        assert enhanced == original

    def test_enhance_prompt_basic_mode(self):
        """Test basic mode CoT enhancement"""
        config = CoTConfig(enabled=True, mode=CoTMode.BASIC)
        enhancer = CoTEnhancer(config)
        
        original = "Solve this problem"
        enhanced = enhancer.enhance_prompt(original)
        
        # Should contain CoT structure
        assert "step by step" in enhanced.lower()
        assert "core question" in enhanced.lower()
        assert original in enhanced

    def test_enhance_prompt_structured_mode(self):
        """Test structured mode CoT enhancement"""
        config = CoTConfig(enabled=True, mode=CoTMode.STRUCTURED)
        enhancer = CoTEnhancer(config)
        
        original = "Analyze this data"
        enhanced = enhancer.enhance_prompt(original)
        
        # Should contain structured thinking elements
        assert "Analysis Phase" in enhanced
        assert "Planning Phase" in enhanced
        assert "Execution Phase" in enhanced
        assert original in enhanced

    def test_enhance_draft_prompt(self):
        """Test draft prompt enhancement"""
        enhancer = CoTEnhancer()
        
        prompt = "Write a technical report"
        domain = "technical"
        enhanced = enhancer.enhance_draft_prompt(prompt, domain)
        
        assert len(enhanced) > len(prompt)
        assert prompt in enhanced

    def test_enhance_critique_prompt(self):
        """Test critique prompt enhancement"""
        enhancer = CoTEnhancer()
        
        original = "Original question"
        draft = "Draft content"
        critique_type = "accuracy"
        
        enhanced = enhancer.enhance_critique_prompt(original, draft, critique_type)
        
        assert "Original Question" in enhanced
        assert "Content to Analyze" in enhanced
        assert original in enhanced
        assert draft in enhanced

    def test_enhance_synthesis_prompt(self):
        """Test synthesis prompt enhancement"""
        enhancer = CoTEnhancer()
        
        original = "Original question"
        draft = "Draft content"
        critiques = ["Critique 1", "Critique 2"]
        
        enhanced = enhancer.enhance_synthesis_prompt(original, draft, critiques)
        
        assert "Original Question" in enhanced
        assert "Current Response" in enhanced
        assert "Critiques to Address" in enhanced
        assert "Critique 1" in enhanced
        assert "Critique 2" in enhanced

    def test_get_stats(self):
        """Test statistics retrieval"""
        config = CoTConfig(enabled=True, mode=CoTMode.CRITIQUE)
        enhancer = CoTEnhancer(config)
        
        stats = enhancer.get_stats()
        
        assert stats["enabled"] is True
        assert stats["mode"] == "critique"
        assert "available_modes" in stats
        assert len(stats["available_modes"]) == 5  # All CoT modes


class TestCoTModes:
    """Test different CoT modes"""

    def test_all_modes_available(self):
        """Test all CoT modes are available"""
        expected_modes = ["basic", "structured", "domain_specific", "critique", "synthesis"]
        actual_modes = [mode.value for mode in CoTMode]
        
        for mode in expected_modes:
            assert mode in actual_modes

    def test_mode_specific_templates(self):
        """Test mode-specific template usage"""
        enhancer = CoTEnhancer()
        prompt = "Test prompt"
        
        # Test different modes produce different enhancements
        basic = enhancer.enhance_prompt(prompt, mode=CoTMode.BASIC)
        structured = enhancer.enhance_prompt(prompt, mode=CoTMode.STRUCTURED)
        critique = enhancer.enhance_prompt(prompt, mode=CoTMode.CRITIQUE)
        
        # Should all be different
        assert basic != structured
        assert structured != critique
        assert basic != critique


class TestCoTFactory:
    """Test CoT factory functions"""

    def test_create_cot_enhancer_defaults(self):
        """Test factory with defaults"""
        enhancer = create_cot_enhancer()
        assert enhancer.config.enabled is True
        assert enhancer.config.mode == CoTMode.STRUCTURED

    def test_create_cot_enhancer_custom(self):
        """Test factory with custom settings"""
        enhancer = create_cot_enhancer(
            enabled=False,
            mode="basic",
            domain_context="Test domain"
        )
        assert enhancer.config.enabled is False
        assert enhancer.config.mode == CoTMode.BASIC
        assert enhancer.config.domain_context == "Test domain"

    def test_create_cot_enhancer_invalid_mode(self):
        """Test factory with invalid mode falls back to structured"""
        enhancer = create_cot_enhancer(mode="invalid_mode")
        assert enhancer.config.mode == CoTMode.STRUCTURED


class TestCoTIntegration:
    """Test CoT integration scenarios"""

    def test_metacognition_toggle(self):
        """Test metacognition can be toggled"""
        config_with = CoTConfig(include_metacognition=True)
        config_without = CoTConfig(include_metacognition=False)
        
        enhancer_with = CoTEnhancer(config_with)
        enhancer_without = CoTEnhancer(config_without)
        
        prompt = "Test prompt"
        enhanced_with = enhancer_with.enhance_prompt(prompt)
        enhanced_without = enhancer_without.enhance_prompt(prompt)
        
        # With metacognition should be longer (has additional prompts)
        assert len(enhanced_with) > len(enhanced_without)

    def test_domain_specific_mode(self):
        """Test domain-specific mode with context"""
        config = CoTConfig(mode=CoTMode.DOMAIN_SPECIFIC, domain_context="Technical analysis")
        enhancer = CoTEnhancer(config)
        
        prompt = "Analyze the system"
        enhanced = enhancer.enhance_prompt(prompt, domain="technical")
        
        assert "Domain-specific thinking" in enhanced
        assert "technical" in enhanced.lower()

    def test_prompt_length_growth(self):
        """Test that CoT enhancement increases prompt length reasonably"""
        enhancer = CoTEnhancer()
        
        short_prompt = "Hello"
        long_prompt = "This is a much longer prompt that contains multiple sentences and ideas that need to be processed."
        
        enhanced_short = enhancer.enhance_prompt(short_prompt)
        enhanced_long = enhancer.enhance_prompt(long_prompt)
        
        # CoT should add structure to both
        assert len(enhanced_short) > len(short_prompt)
        assert len(enhanced_long) > len(long_prompt)
        
        # But the ratio should be reasonable (not 10x growth)
        short_ratio = len(enhanced_short) / len(short_prompt)
        long_ratio = len(enhanced_long) / len(long_prompt)
        
        # For longer prompts, ratio should be smaller
        assert long_ratio < short_ratio