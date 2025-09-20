"""
Enhanced Query Processing Module.

This module provides advanced query processing capabilities including:
- Intelligent query validation with clarifying questions
- Multi-step analysis execution with progress tracking
- Result synthesis with confidence scoring
- Gradio interface integration
"""

from .query_validator import (
    QueryValidator,
    QueryType,
    ValidationResult
)

from .multi_step_executor import (
    MultiStepExecutor,
    ExecutionStep,
    ExecutionPlan,
    StepStatus,
    ProgressTracker
)

from .result_synthesizer import (
    ResultSynthesizer,
    SynthesisResult,
    SourceInfo,
    ConfidenceLevel
)

from .gradio_interface import (
    EnhancedQueryInterface,
    create_gradio_app,
    launch_gradio_app
)

__version__ = "1.0.0"
__author__ = "Stellar Connect Team"

__all__ = [
    # Query Validation
    "QueryValidator",
    "QueryType",
    "ValidationResult",

    # Multi-step Execution
    "MultiStepExecutor",
    "ExecutionStep",
    "ExecutionPlan",
    "StepStatus",
    "ProgressTracker",

    # Result Synthesis
    "ResultSynthesizer",
    "SynthesisResult",
    "SourceInfo",
    "ConfidenceLevel",

    # Gradio Interface
    "EnhancedQueryInterface",
    "create_gradio_app",
    "launch_gradio_app"
]