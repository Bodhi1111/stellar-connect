"""
Stellar Connect Reasoning Engine Module
Phase 2 Week 3: Cognitive Pipeline Components

This module implements the complete cognitive reasoning pipeline for estate planning:

Components:
- EstateGatekeeper: Query validation and preprocessing
- EstatePlanner: Multi-step analysis planning
- EstateAuditor: Quality control and self-correction
- EstateStrategist: Strategic synthesis and insights
- EstateReasoningEngine: Unified orchestration engine

Usage:
    from agents.reasoning import EstateReasoningEngine, ReasoningConfig

    engine = EstateReasoningEngine()
    result = await engine.process_query("How can I minimize estate taxes?")
"""

from .gatekeeper import EstateGatekeeper, QueryValidation, QueryType, ValidationSeverity
from .planner import EstatePlanner, AnalysisPlan, AnalysisTask, AnalysisType, TaskPriority
from .auditor import EstateAuditor, AuditResult, QualityMetrics, AuditSeverity
from .strategist import EstateStrategist, SynthesisResult, StrategicInsight, InsightType
from .reasoning_engine import EstateReasoningEngine, ReasoningResult, ReasoningStatus, ReasoningConfig, ExecutionMode

__all__ = [
    # Core engine
    'EstateReasoningEngine',
    'ReasoningResult',
    'ReasoningStatus',
    'ReasoningConfig',
    'ExecutionMode',

    # Gatekeeper
    'EstateGatekeeper',
    'QueryValidation',
    'QueryType',
    'ValidationSeverity',

    # Planner
    'EstatePlanner',
    'AnalysisPlan',
    'AnalysisTask',
    'AnalysisType',
    'TaskPriority',

    # Auditor
    'EstateAuditor',
    'AuditResult',
    'QualityMetrics',
    'AuditSeverity',

    # Strategist
    'EstateStrategist',
    'SynthesisResult',
    'StrategicInsight',
    'InsightType'
]

__version__ = "1.0.0"
__author__ = "Stellar Connect Development Team"