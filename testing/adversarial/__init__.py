"""
Adversarial Testing Framework for Estate Planning AI Systems.

This framework provides comprehensive tools for testing AI system robustness,
safety, and compliance in the estate planning domain.

Components:
- Estate-specific attack vectors
- Automated adversarial prompt generation (Red Team)
- LLM-as-a-judge evaluation system
- Compliance and ethics checking
"""

from .estate_attack_vectors import (
    EstateAttackVectors,
    AttackVector,
    AttackCategory,
    RiskLevel,
    DynamicAttackGenerator,
    ESTATE_ATTACK_VECTORS,
    CRITICAL_ATTACKS,
    HIGH_RISK_ATTACKS,
    COMPLIANCE_ATTACKS,
    CONFIDENTIALITY_ATTACKS
)

from .red_team_generator import (
    RedTeamGenerator,
    GeneratedAttack,
    GenerationStrategy,
    GenerationConfig,
    generate_quick_test_suite,
    generate_category_focused_suite
)

from .automated_judge import (
    AutomatedJudge,
    ResponseEvaluation,
    EvaluationScore,
    EvaluationCriteria,
    ResponseQuality,
    quick_evaluate_response,
    create_evaluation_config
)

from .compliance_evaluator import (
    ComplianceEvaluator,
    ComplianceEvaluation,
    ComplianceViolation,
    ComplianceRule,
    ComplianceCategory,
    ComplianceLevel,
    RegulatoryFramework,
    quick_compliance_check,
    check_legal_disclaimers
)

__version__ = "1.0.0"
__author__ = "Stellar Connect Security Team"

__all__ = [
    # Attack Vectors
    "EstateAttackVectors",
    "AttackVector",
    "AttackCategory",
    "RiskLevel",
    "DynamicAttackGenerator",
    "ESTATE_ATTACK_VECTORS",
    "CRITICAL_ATTACKS",
    "HIGH_RISK_ATTACKS",
    "COMPLIANCE_ATTACKS",
    "CONFIDENTIALITY_ATTACKS",

    # Red Team Generator
    "RedTeamGenerator",
    "GeneratedAttack",
    "GenerationStrategy",
    "GenerationConfig",
    "generate_quick_test_suite",
    "generate_category_focused_suite",

    # Automated Judge
    "AutomatedJudge",
    "ResponseEvaluation",
    "EvaluationScore",
    "EvaluationCriteria",
    "ResponseQuality",
    "quick_evaluate_response",
    "create_evaluation_config",

    # Compliance Evaluator
    "ComplianceEvaluator",
    "ComplianceEvaluation",
    "ComplianceViolation",
    "ComplianceRule",
    "ComplianceCategory",
    "ComplianceLevel",
    "RegulatoryFramework",
    "quick_compliance_check",
    "check_legal_disclaimers"
]