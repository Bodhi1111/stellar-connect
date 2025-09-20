"""
Estate Gatekeeper Agent for Stellar Connect
Phase 2 Week 3: Cognitive Pipeline Components

The Estate Gatekeeper is responsible for query validation, clarification, and preprocessing
before queries enter the reasoning pipeline. It ensures that all inputs meet quality standards
and contain sufficient information for effective estate planning analysis.
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class QueryType(Enum):
    """Types of estate planning queries."""
    TRUST_STRUCTURE = "trust_structure"
    TAX_OPTIMIZATION = "tax_optimization"
    ASSET_PROTECTION = "asset_protection"
    SUCCESSION_PLANNING = "succession_planning"
    CHARITABLE_GIVING = "charitable_giving"
    BUSINESS_TRANSITION = "business_transition"
    GENERAL_PLANNING = "general_planning"
    UNCLEAR = "unclear"


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    CRITICAL = "critical"  # Query cannot proceed
    WARNING = "warning"    # Query can proceed with clarification
    INFO = "info"         # Informational only


@dataclass
class ValidationIssue:
    """Represents a validation issue found in the query."""
    severity: ValidationSeverity
    issue_type: str
    description: str
    suggested_clarification: str
    required_fields: List[str] = field(default_factory=list)


@dataclass
class QueryValidation:
    """Result of query validation process."""
    is_valid: bool
    query_type: QueryType
    confidence_score: float
    validated_query: str
    extracted_entities: Dict[str, Any]
    issues: List[ValidationIssue] = field(default_factory=list)
    clarifying_questions: List[str] = field(default_factory=list)
    preprocessing_notes: List[str] = field(default_factory=list)


class EstateGatekeeper:
    """
    Estate Gatekeeper Agent for query validation and preprocessing.

    Responsibilities:
    - Validate incoming queries for completeness and clarity
    - Extract key estate planning entities and context
    - Generate clarifying questions for ambiguous queries
    - Classify query types for appropriate routing
    - Ensure queries meet minimum quality thresholds
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Estate planning entity patterns
        self.entity_patterns = {
            'monetary_amounts': r'\$[\d,]+(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|million|billion|k|K)',
            'family_members': r'\b(?:spouse|wife|husband|children?|kids?|son|daughter|grandchildren?|heir|beneficiar(?:y|ies))\b',
            'legal_entities': r'\b(?:trust|estate|corporation|LLC|partnership|foundation|charity)\b',
            'assets': r'\b(?:house|home|property|real estate|business|stocks?|bonds?|portfolio|401k|IRA|retirement)\b',
            'time_references': r'\b(?:when|after|before|during|by|until)\s+(?:death|retirement|age\s+\d+|next year|in\s+\d+\s+years?)\b',
            'locations': r'\b(?:state|country|jurisdiction|residence|domicile)\b'
        }

        # Required information patterns for different query types
        self.required_info = {
            QueryType.TRUST_STRUCTURE: ['monetary_amounts', 'family_members', 'assets'],
            QueryType.TAX_OPTIMIZATION: ['monetary_amounts', 'assets'],
            QueryType.ASSET_PROTECTION: ['assets', 'monetary_amounts'],
            QueryType.SUCCESSION_PLANNING: ['family_members', 'assets'],
            QueryType.CHARITABLE_GIVING: ['monetary_amounts'],
            QueryType.BUSINESS_TRANSITION: ['assets', 'family_members'],
            QueryType.GENERAL_PLANNING: ['monetary_amounts']
        }

        # Query type classification keywords
        self.query_type_keywords = {
            QueryType.TRUST_STRUCTURE: ['trust', 'revocable', 'irrevocable', 'living trust', 'testamentary'],
            QueryType.TAX_OPTIMIZATION: ['tax', 'estate tax', 'gift tax', 'generation skipping', 'exemption'],
            QueryType.ASSET_PROTECTION: ['protect', 'creditor', 'lawsuit', 'shield', 'liability'],
            QueryType.SUCCESSION_PLANNING: ['succession', 'inherit', 'pass down', 'next generation', 'legacy'],
            QueryType.CHARITABLE_GIVING: ['charity', 'charitable', 'donation', 'foundation', 'philanthrop'],
            QueryType.BUSINESS_TRANSITION: ['business', 'company', 'sell', 'transition', 'succession'],
            QueryType.GENERAL_PLANNING: ['estate planning', 'plan', 'strategy', 'advice', 'help']
        }

    async def validate(self, query: str) -> QueryValidation:
        """
        Validate and preprocess an estate planning query.

        Args:
            query: Raw query string from the user

        Returns:
            QueryValidation object with validation results and extracted information
        """
        self.logger.info(f"Validating query: {query[:100]}...")

        # Initialize validation result
        validation = QueryValidation(
            is_valid=True,
            query_type=QueryType.UNCLEAR,
            confidence_score=0.0,
            validated_query=query.strip(),
            extracted_entities={}
        )

        # Step 1: Basic quality checks
        await self._check_basic_quality(query, validation)

        # Step 2: Classify query type
        await self._classify_query_type(query, validation)

        # Step 3: Extract entities
        await self._extract_entities(query, validation)

        # Step 4: Validate completeness
        await self._validate_completeness(validation)

        # Step 5: Generate clarifying questions if needed
        await self._generate_clarifying_questions(validation)

        # Step 6: Calculate final confidence score
        validation.confidence_score = self._calculate_confidence_score(validation)

        # Step 7: Determine if query is valid enough to proceed
        validation.is_valid = self._determine_validity(validation)

        self.logger.info(f"Validation complete. Valid: {validation.is_valid}, "
                        f"Type: {validation.query_type.value}, "
                        f"Confidence: {validation.confidence_score:.2f}")

        return validation

    async def _check_basic_quality(self, query: str, validation: QueryValidation) -> None:
        """Check basic quality criteria for the query."""
        issues = []

        # Check minimum length
        if len(query.strip()) < 10:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                issue_type="insufficient_length",
                description="Query is too short to provide meaningful analysis",
                suggested_clarification="Please provide more details about your estate planning needs"
            ))

        # Check for complete sentences
        if not query.strip().endswith(('.', '?', '!')):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                issue_type="incomplete_sentence",
                description="Query appears to be incomplete",
                suggested_clarification="Please ensure your question is complete"
            ))

        # Check for gibberish or excessive special characters
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s\.,?!]', query)) / len(query)
        if special_char_ratio > 0.3:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                issue_type="excessive_special_chars",
                description="Query contains many special characters",
                suggested_clarification="Please rephrase your question using standard language"
            ))

        validation.issues.extend(issues)

    async def _classify_query_type(self, query: str, validation: QueryValidation) -> None:
        """Classify the type of estate planning query."""
        query_lower = query.lower()
        type_scores = {}

        # Score each query type based on keyword matches
        for query_type, keywords in self.query_type_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in query_lower:
                    score += 1

            # Normalize score by number of keywords
            type_scores[query_type] = score / len(keywords) if keywords else 0

        # Find the best match
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            if best_type[1] > 0:
                validation.query_type = best_type[0]
            else:
                validation.query_type = QueryType.GENERAL_PLANNING

        validation.preprocessing_notes.append(f"Classified as {validation.query_type.value} query")

    async def _extract_entities(self, query: str, validation: QueryValidation) -> None:
        """Extract estate planning entities from the query."""
        entities = {}

        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches

        validation.extracted_entities = entities

        if entities:
            validation.preprocessing_notes.append(f"Extracted entities: {list(entities.keys())}")

    async def _validate_completeness(self, validation: QueryValidation) -> None:
        """Validate that the query contains sufficient information."""
        required_entities = self.required_info.get(validation.query_type, [])
        missing_entities = []

        for required_entity in required_entities:
            if required_entity not in validation.extracted_entities:
                missing_entities.append(required_entity)

        # Add issues for missing critical information
        if missing_entities:
            severity = ValidationSeverity.CRITICAL if len(missing_entities) > len(required_entities) / 2 else ValidationSeverity.WARNING

            validation.issues.append(ValidationIssue(
                severity=severity,
                issue_type="missing_information",
                description=f"Missing important information: {', '.join(missing_entities)}",
                suggested_clarification="Please provide additional details to enable accurate analysis",
                required_fields=missing_entities
            ))

    async def _generate_clarifying_questions(self, validation: QueryValidation) -> None:
        """Generate clarifying questions based on missing information and query type."""
        questions = []

        # Questions based on missing entities
        entity_questions = {
            'monetary_amounts': "What is the approximate value of the assets involved?",
            'family_members': "Who are the intended beneficiaries (spouse, children, etc.)?",
            'assets': "What types of assets are you looking to include (real estate, business, investments)?",
            'time_references': "What is your timeline for implementing this plan?",
            'locations': "What state or jurisdiction will this plan be implemented in?"
        }

        # Add questions for missing required entities
        for issue in validation.issues:
            if issue.issue_type == "missing_information":
                for missing_field in issue.required_fields:
                    if missing_field in entity_questions:
                        questions.append(entity_questions[missing_field])

        # Type-specific questions
        type_specific_questions = {
            QueryType.TRUST_STRUCTURE: [
                "Do you prefer a revocable or irrevocable trust structure?",
                "Are there any specific trust provisions you want to include?"
            ],
            QueryType.TAX_OPTIMIZATION: [
                "What is your primary tax concern (estate tax, gift tax, income tax)?",
                "Have you used any of your lifetime exemption amounts?"
            ],
            QueryType.ASSET_PROTECTION: [
                "What specific risks are you trying to protect against?",
                "Do you have any existing liability concerns?"
            ],
            QueryType.SUCCESSION_PLANNING: [
                "Are all intended beneficiaries adults?",
                "Do you have any special needs beneficiaries to consider?"
            ]
        }

        # Add type-specific questions if confidence is low
        if validation.query_type in type_specific_questions and len(validation.extracted_entities) < 2:
            questions.extend(type_specific_questions[validation.query_type][:2])

        validation.clarifying_questions = list(set(questions))  # Remove duplicates

    def _calculate_confidence_score(self, validation: QueryValidation) -> float:
        """Calculate confidence score for the validation."""
        score = 1.0

        # Reduce score for critical issues
        critical_issues = [i for i in validation.issues if i.severity == ValidationSeverity.CRITICAL]
        score -= len(critical_issues) * 0.3

        # Reduce score for warning issues
        warning_issues = [i for i in validation.issues if i.severity == ValidationSeverity.WARNING]
        score -= len(warning_issues) * 0.1

        # Increase score for extracted entities
        score += len(validation.extracted_entities) * 0.1

        # Increase score for clear query type classification
        if validation.query_type != QueryType.UNCLEAR:
            score += 0.2

        return max(0.0, min(1.0, score))

    def _determine_validity(self, validation: QueryValidation) -> bool:
        """Determine if the query is valid enough to proceed."""
        # Query is invalid if there are critical issues
        critical_issues = [i for i in validation.issues if i.severity == ValidationSeverity.CRITICAL]
        if critical_issues:
            return False

        # Query is valid if confidence score is above threshold
        return validation.confidence_score >= 0.5

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the gatekeeper."""
        return {
            "status": "healthy",
            "component": "estate_gatekeeper",
            "entity_patterns": len(self.entity_patterns),
            "query_types": len(self.query_type_keywords),
            "last_check": datetime.now().isoformat()
        }


# Example usage and testing
async def main():
    """Example usage of Estate Gatekeeper."""
    gatekeeper = EstateGatekeeper()

    # Test queries
    test_queries = [
        "I need help with estate planning",
        "I want to create a trust for my $2 million estate to benefit my wife and two children",
        "How can I protect my business from creditors?",
        "What are the tax implications of gifting $100,000 to my daughter?",
        "abc123!@#"  # Invalid query
    ]

    for query in test_queries:
        print(f"\n--- Testing Query: '{query}' ---")
        validation = await gatekeeper.validate(query)

        print(f"Valid: {validation.is_valid}")
        print(f"Type: {validation.query_type.value}")
        print(f"Confidence: {validation.confidence_score:.2f}")
        print(f"Entities: {list(validation.extracted_entities.keys())}")

        if validation.issues:
            print("Issues:")
            for issue in validation.issues:
                print(f"  - {issue.severity.value}: {issue.description}")

        if validation.clarifying_questions:
            print("Clarifying Questions:")
            for question in validation.clarifying_questions:
                print(f"  - {question}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())