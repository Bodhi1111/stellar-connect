"""
Rebuttal Library for Stellar Connect
Utility class for managing and retrieving effective rebuttals for estate planning objections

This module provides intelligent rebuttal matching and effectiveness tracking
for estate planning sales conversations.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import re
from enum import Enum


class ObjectionCategory(Enum):
    """Categories of common estate planning objections."""
    COST = "cost"
    COMPLEXITY = "complexity"
    NECESSITY = "necessity"
    TIMING = "timing"
    TRUST_ISSUES = "trust_issues"
    FAMILY_DYNAMICS = "family_dynamics"
    TAX_CONCERNS = "tax_concerns"
    PRIVACY = "privacy"
    CONTROL = "control"
    LIQUIDITY = "liquidity"


class RebuttalType(Enum):
    """Types of rebuttal approaches."""
    DIRECT_ANSWER = "direct_answer"
    REFRAME = "reframe"
    QUESTION_BACK = "question_back"
    STORY_EXAMPLE = "story_example"
    COST_BENEFIT = "cost_benefit"
    URGENCY_CREATE = "urgency_create"
    SOCIAL_PROOF = "social_proof"
    EDUCATION = "education"


@dataclass
class RebuttalPattern:
    """Represents an effective rebuttal pattern."""
    rebuttal_id: str
    objection_category: ObjectionCategory
    objection_text: str
    rebuttal_text: str
    rebuttal_type: RebuttalType
    effectiveness_score: float
    usage_count: int
    success_rate: float
    context_tags: List[str]
    advisor: str
    date_created: datetime
    last_used: Optional[datetime] = None
    client_segments: List[str] = field(default_factory=list)
    estate_value_range: Tuple[float, float] = (0, float('inf'))
    follow_up_actions: List[str] = field(default_factory=list)


@dataclass
class ObjectionAnalysis:
    """Analysis of an objection for rebuttal matching."""
    original_objection: str
    category: ObjectionCategory
    emotional_intensity: float  # 0-1 scale
    specific_concerns: List[str]
    client_context: Dict[str, Any]
    urgency_level: float  # 0-1 scale
    confidence_score: float  # How confident we are in the analysis


@dataclass
class RebuttalRecommendation:
    """A rebuttal recommendation with context."""
    rebuttal_pattern: RebuttalPattern
    relevance_score: float
    customization_suggestions: List[str]
    context_match_factors: List[str]
    risk_factors: List[str]
    follow_up_strategy: List[str]


class RebuttalLibrary:
    """
    Intelligent rebuttal library for estate planning objections.

    Features:
    - Objection categorization and analysis
    - Context-aware rebuttal matching
    - Effectiveness tracking and learning
    - Personalization based on client segments
    - Success pattern recognition
    """

    def __init__(self, knowledge_base_path: str = "data/knowledge_base"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.knowledge_base_path.mkdir(parents=True, exist_ok=True)

        self.rebuttal_patterns: List[RebuttalPattern] = []
        self.objection_keywords: Dict[ObjectionCategory, List[str]] = {}
        self.effectiveness_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

        # Initialize objection keywords
        self._initialize_objection_keywords()

        # Load existing rebuttals
        asyncio.create_task(self._load_rebuttals())

    def _initialize_objection_keywords(self):
        """Initialize keyword patterns for objection categorization."""
        self.objection_keywords = {
            ObjectionCategory.COST: [
                'expensive', 'cost', 'fees', 'afford', 'money', 'price',
                'budget', 'too much', 'cheaper', 'worth it'
            ],
            ObjectionCategory.COMPLEXITY: [
                'complicated', 'complex', 'difficult', 'confusing',
                'understand', 'simple', 'overwhelmed', 'too much work'
            ],
            ObjectionCategory.NECESSITY: [
                'necessary', 'need', 'required', 'important', 'must have',
                'essential', 'worth doing', 'really need'
            ],
            ObjectionCategory.TIMING: [
                'time', 'rush', 'hurry', 'urgent', 'delay', 'later',
                'busy', 'now', 'when', 'schedule'
            ],
            ObjectionCategory.TRUST_ISSUES: [
                'trust', 'confidence', 'believe', 'doubt', 'sure',
                'reliable', 'proven', 'track record'
            ],
            ObjectionCategory.FAMILY_DYNAMICS: [
                'family', 'children', 'spouse', 'fight', 'disagree',
                'harmony', 'conflict', 'kids', 'heirs'
            ],
            ObjectionCategory.TAX_CONCERNS: [
                'tax', 'irs', 'deduction', 'exemption', 'liability',
                'government', 'audit', 'penalties'
            ],
            ObjectionCategory.PRIVACY: [
                'private', 'confidential', 'public', 'records',
                'disclosure', 'secret', 'personal'
            ],
            ObjectionCategory.CONTROL: [
                'control', 'power', 'decisions', 'authority',
                'flexibility', 'change', 'modify', 'revoke'
            ],
            ObjectionCategory.LIQUIDITY: [
                'liquid', 'cash', 'access', 'tied up', 'frozen',
                'available', 'emergency', 'flexible'
            ]
        }

    async def analyze_objection(
        self,
        objection_text: str,
        client_context: Optional[Dict[str, Any]] = None
    ) -> ObjectionAnalysis:
        """
        Analyze an objection to understand its category and characteristics.

        Args:
            objection_text: The client's objection statement
            client_context: Additional context about the client

        Returns:
            ObjectionAnalysis with categorization and insights
        """
        if client_context is None:
            client_context = {}

        objection_lower = objection_text.lower()

        # Categorize the objection
        category_scores = {}
        for category, keywords in self.objection_keywords.items():
            score = sum(1 for keyword in keywords if keyword in objection_lower)
            if score > 0:
                category_scores[category] = score / len(keywords)

        # Determine primary category
        if category_scores:
            primary_category = max(category_scores.items(), key=lambda x: x[1])[0]
            confidence_score = max(category_scores.values())
        else:
            primary_category = ObjectionCategory.NECESSITY  # Default
            confidence_score = 0.1

        # Analyze emotional intensity
        emotional_intensity = self._analyze_emotional_intensity(objection_text)

        # Extract specific concerns
        specific_concerns = self._extract_specific_concerns(objection_text, primary_category)

        # Assess urgency level
        urgency_level = self._assess_urgency_level(objection_text)

        return ObjectionAnalysis(
            original_objection=objection_text,
            category=primary_category,
            emotional_intensity=emotional_intensity,
            specific_concerns=specific_concerns,
            client_context=client_context,
            urgency_level=urgency_level,
            confidence_score=confidence_score
        )

    def get_effective_rebuttals(
        self,
        objection_analysis: ObjectionAnalysis,
        max_results: int = 3,
        min_effectiveness_threshold: float = 0.6
    ) -> List[RebuttalRecommendation]:
        """
        Get effective rebuttals for an analyzed objection.

        Args:
            objection_analysis: Analysis of the objection
            max_results: Maximum number of rebuttals to return
            min_effectiveness_threshold: Minimum effectiveness score

        Returns:
            List of rebuttal recommendations
        """
        recommendations = []

        for rebuttal in self.rebuttal_patterns:
            # Check if rebuttal matches objection category
            if rebuttal.objection_category != objection_analysis.category:
                continue

            # Check effectiveness threshold
            if rebuttal.effectiveness_score < min_effectiveness_threshold:
                continue

            # Calculate relevance score
            relevance_score = self._calculate_rebuttal_relevance(
                objection_analysis, rebuttal
            )

            if relevance_score > 0.3:  # Minimum relevance threshold
                recommendation = self._create_rebuttal_recommendation(
                    rebuttal, objection_analysis, relevance_score
                )
                recommendations.append(recommendation)

        # Sort by composite score (relevance * effectiveness)
        recommendations.sort(
            key=lambda x: x.relevance_score * x.rebuttal_pattern.effectiveness_score,
            reverse=True
        )

        return recommendations[:max_results]

    def _calculate_rebuttal_relevance(
        self,
        objection_analysis: ObjectionAnalysis,
        rebuttal: RebuttalPattern
    ) -> float:
        """Calculate how relevant a rebuttal is to the specific objection."""
        relevance_score = 0.0

        # Base score for category match
        relevance_score += 0.4

        # Text similarity bonus
        text_similarity = self._calculate_text_similarity(
            objection_analysis.original_objection,
            rebuttal.objection_text
        )
        relevance_score += text_similarity * 0.3

        # Context match bonus
        client_context = objection_analysis.client_context
        estate_value = client_context.get('estate_value', 0)

        if rebuttal.estate_value_range[0] <= estate_value <= rebuttal.estate_value_range[1]:
            relevance_score += 0.2

        # Client segment match
        client_segment = client_context.get('segment', '')
        if client_segment and client_segment in rebuttal.client_segments:
            relevance_score += 0.1

        return min(relevance_score, 1.0)

    def _create_rebuttal_recommendation(
        self,
        rebuttal: RebuttalPattern,
        objection_analysis: ObjectionAnalysis,
        relevance_score: float
    ) -> RebuttalRecommendation:
        """Create a detailed rebuttal recommendation."""
        # Generate customization suggestions
        customization_suggestions = self._generate_customization_suggestions(
            rebuttal, objection_analysis
        )

        # Identify context match factors
        context_match_factors = self._identify_context_matches(
            rebuttal, objection_analysis
        )

        # Assess risk factors
        risk_factors = self._assess_risk_factors(rebuttal, objection_analysis)

        # Suggest follow-up strategy
        follow_up_strategy = self._suggest_follow_up_strategy(
            rebuttal, objection_analysis
        )

        return RebuttalRecommendation(
            rebuttal_pattern=rebuttal,
            relevance_score=relevance_score,
            customization_suggestions=customization_suggestions,
            context_match_factors=context_match_factors,
            risk_factors=risk_factors,
            follow_up_strategy=follow_up_strategy
        )

    def _analyze_emotional_intensity(self, objection_text: str) -> float:
        """Analyze emotional intensity of the objection."""
        high_intensity_words = [
            'never', 'absolutely', 'terrible', 'horrible', 'hate',
            'refuse', 'impossible', 'ridiculous', 'outrageous'
        ]

        medium_intensity_words = [
            'concerned', 'worried', 'uncomfortable', 'hesitant',
            'uncertain', 'doubtful', 'skeptical'
        ]

        objection_lower = objection_text.lower()

        high_count = sum(1 for word in high_intensity_words if word in objection_lower)
        medium_count = sum(1 for word in medium_intensity_words if word in objection_lower)

        # Calculate intensity score (0-1 scale)
        intensity = (high_count * 0.8 + medium_count * 0.4) / max(len(objection_text.split()), 1)
        return min(intensity, 1.0)

    def _extract_specific_concerns(
        self, objection_text: str, category: ObjectionCategory
    ) -> List[str]:
        """Extract specific concerns from the objection text."""
        concerns = []
        objection_lower = objection_text.lower()

        # Category-specific concern extraction
        if category == ObjectionCategory.COST:
            cost_patterns = [
                r'\$[\d,]+', r'\d+\s*thousand', r'\d+\s*million',
                r'too expensive', r'can\'t afford', r'budget'
            ]
            for pattern in cost_patterns:
                matches = re.findall(pattern, objection_lower)
                concerns.extend(matches)

        elif category == ObjectionCategory.TIMING:
            timing_concerns = [
                'too busy', 'no time', 'later', 'next year',
                'after retirement', 'when older'
            ]
            for concern in timing_concerns:
                if concern in objection_lower:
                    concerns.append(concern)

        elif category == ObjectionCategory.FAMILY_DYNAMICS:
            family_concerns = [
                'children will fight', 'family conflict', 'spouse disagrees',
                'kids don\'t get along', 'family harmony'
            ]
            for concern in family_concerns:
                if concern in objection_lower:
                    concerns.append(concern)

        return concerns

    def _assess_urgency_level(self, objection_text: str) -> float:
        """Assess the urgency level indicated in the objection."""
        urgent_indicators = [
            'need now', 'urgent', 'immediately', 'soon', 'quickly',
            'before', 'deadline', 'time sensitive'
        ]

        non_urgent_indicators = [
            'later', 'someday', 'eventually', 'no rush',
            'think about it', 'consider'
        ]

        objection_lower = objection_text.lower()

        urgent_count = sum(1 for indicator in urgent_indicators if indicator in objection_lower)
        non_urgent_count = sum(1 for indicator in non_urgent_indicators if indicator in objection_lower)

        if urgent_count > non_urgent_count:
            return 0.7 + (urgent_count * 0.1)
        elif non_urgent_count > urgent_count:
            return 0.3 - (non_urgent_count * 0.1)
        else:
            return 0.5  # Neutral urgency

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _generate_customization_suggestions(
        self,
        rebuttal: RebuttalPattern,
        objection_analysis: ObjectionAnalysis
    ) -> List[str]:
        """Generate suggestions for customizing the rebuttal."""
        suggestions = []

        # Emotional intensity adjustments
        if objection_analysis.emotional_intensity > 0.7:
            suggestions.append("Use empathetic tone and acknowledge their strong feelings")
        elif objection_analysis.emotional_intensity < 0.3:
            suggestions.append("Can use more direct approach since emotion is low")

        # Client context adjustments
        estate_value = objection_analysis.client_context.get('estate_value', 0)
        if estate_value > 5_000_000:
            suggestions.append("Emphasize sophisticated planning strategies for high net worth")
        elif estate_value < 1_000_000:
            suggestions.append("Focus on simple, cost-effective solutions")

        # Specific concern adjustments
        for concern in objection_analysis.specific_concerns:
            if 'afford' in concern:
                suggestions.append("Address affordability with payment options or value demonstration")
            elif 'time' in concern:
                suggestions.append("Emphasize efficiency and streamlined process")

        return suggestions

    def _identify_context_matches(
        self,
        rebuttal: RebuttalPattern,
        objection_analysis: ObjectionAnalysis
    ) -> List[str]:
        """Identify why this rebuttal matches the context."""
        matches = []

        # Category match
        matches.append(f"Addresses {objection_analysis.category.value} objections")

        # Estate value match
        estate_value = objection_analysis.client_context.get('estate_value', 0)
        if rebuttal.estate_value_range[0] <= estate_value <= rebuttal.estate_value_range[1]:
            matches.append("Appropriate for client's estate value range")

        # High effectiveness
        if rebuttal.effectiveness_score > 0.8:
            matches.append("Proven high effectiveness in similar situations")

        # Recent usage
        if rebuttal.last_used and (datetime.now() - rebuttal.last_used).days < 30:
            matches.append("Recently used successfully")

        return matches

    def _assess_risk_factors(
        self,
        rebuttal: RebuttalPattern,
        objection_analysis: ObjectionAnalysis
    ) -> List[str]:
        """Assess potential risks in using this rebuttal."""
        risks = []

        # High emotional intensity + direct approach
        if (objection_analysis.emotional_intensity > 0.7 and
            rebuttal.rebuttal_type == RebuttalType.DIRECT_ANSWER):
            risks.append("Direct approach may escalate emotional response")

        # Low effectiveness
        if rebuttal.effectiveness_score < 0.7:
            risks.append("Lower than ideal effectiveness score")

        # Low usage count
        if rebuttal.usage_count < 5:
            risks.append("Limited usage history for reliability assessment")

        # Context mismatch
        estate_value = objection_analysis.client_context.get('estate_value', 0)
        if not (rebuttal.estate_value_range[0] <= estate_value <= rebuttal.estate_value_range[1]):
            risks.append("Estate value outside of rebuttal's proven range")

        return risks

    def _suggest_follow_up_strategy(
        self,
        rebuttal: RebuttalPattern,
        objection_analysis: ObjectionAnalysis
    ) -> List[str]:
        """Suggest follow-up strategy after using the rebuttal."""
        strategies = []

        # Based on rebuttal type
        if rebuttal.rebuttal_type == RebuttalType.QUESTION_BACK:
            strategies.append("Listen carefully to their response and address underlying concerns")
        elif rebuttal.rebuttal_type == RebuttalType.STORY_EXAMPLE:
            strategies.append("Ask if they see similarities to their situation")
        elif rebuttal.rebuttal_type == RebuttalType.EDUCATION:
            strategies.append("Check their understanding and answer any clarifying questions")

        # Based on objection category
        if objection_analysis.category == ObjectionCategory.COST:
            strategies.append("Provide specific ROI calculations or value examples")
        elif objection_analysis.category == ObjectionCategory.TIMING:
            strategies.append("Create urgency with specific consequences of delay")

        # General strategies
        strategies.extend(rebuttal.follow_up_actions)

        return strategies

    async def record_usage(
        self,
        rebuttal_id: str,
        objection_text: str,
        outcome: str,
        client_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Record the usage and outcome of a rebuttal."""
        try:
            # Find the rebuttal
            rebuttal = next(
                (r for r in self.rebuttal_patterns if r.rebuttal_id == rebuttal_id),
                None
            )

            if not rebuttal:
                self.logger.error(f"Rebuttal {rebuttal_id} not found")
                return False

            # Update usage statistics
            rebuttal.usage_count += 1
            rebuttal.last_used = datetime.now()

            # Update effectiveness based on outcome
            if outcome in ["accepted", "resolved", "moved_forward"]:
                success = True
            else:
                success = False

            # Recalculate success rate
            total_recorded = len([h for h in self.effectiveness_history if h['rebuttal_id'] == rebuttal_id])
            if total_recorded == 0:
                rebuttal.success_rate = 1.0 if success else 0.0
            else:
                successful_uses = len([
                    h for h in self.effectiveness_history
                    if h['rebuttal_id'] == rebuttal_id and h['successful']
                ])
                if success:
                    successful_uses += 1
                rebuttal.success_rate = successful_uses / (total_recorded + 1)

            # Update effectiveness score (weighted average of success rate and historical performance)
            rebuttal.effectiveness_score = (rebuttal.success_rate * 0.7) + (rebuttal.effectiveness_score * 0.3)

            # Record in history
            self.effectiveness_history.append({
                "rebuttal_id": rebuttal_id,
                "objection_text": objection_text,
                "outcome": outcome,
                "successful": success,
                "timestamp": datetime.now().isoformat(),
                "client_context": client_context or {}
            })

            # Save changes
            await self._save_rebuttals()

            self.logger.info(f"Recorded usage for rebuttal {rebuttal_id}: {outcome}")
            return True

        except Exception as e:
            self.logger.error(f"Error recording rebuttal usage: {e}")
            return False

    async def add_rebuttal(self, rebuttal: RebuttalPattern) -> bool:
        """Add a new rebuttal pattern to the library."""
        try:
            # Check for duplicates
            existing = next(
                (r for r in self.rebuttal_patterns if r.rebuttal_id == rebuttal.rebuttal_id),
                None
            )

            if existing:
                self.logger.error(f"Rebuttal {rebuttal.rebuttal_id} already exists")
                return False

            self.rebuttal_patterns.append(rebuttal)
            await self._save_rebuttals()

            self.logger.info(f"Added new rebuttal: {rebuttal.rebuttal_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error adding rebuttal: {e}")
            return False

    async def _load_rebuttals(self):
        """Load rebuttals from the knowledge base."""
        rebuttals_file = self.knowledge_base_path / "rebuttal_patterns.json"

        if not rebuttals_file.exists():
            self.logger.info("No rebuttals file found, starting with empty library")
            await self._create_default_rebuttals()
            return

        try:
            with open(rebuttals_file, 'r') as f:
                rebuttals_data = json.load(f)

            for rebuttal_data in rebuttals_data:
                # Convert string enums back to enum objects
                rebuttal_data['objection_category'] = ObjectionCategory(rebuttal_data['objection_category'])
                rebuttal_data['rebuttal_type'] = RebuttalType(rebuttal_data['rebuttal_type'])
                rebuttal_data['date_created'] = datetime.fromisoformat(rebuttal_data['date_created'])

                if rebuttal_data.get('last_used'):
                    rebuttal_data['last_used'] = datetime.fromisoformat(rebuttal_data['last_used'])

                rebuttal = RebuttalPattern(**rebuttal_data)
                self.rebuttal_patterns.append(rebuttal)

            self.logger.info(f"Loaded {len(self.rebuttal_patterns)} rebuttals from knowledge base")

        except Exception as e:
            self.logger.error(f"Error loading rebuttals: {e}")
            await self._create_default_rebuttals()

    async def _save_rebuttals(self):
        """Save rebuttals to the knowledge base."""
        rebuttals_file = self.knowledge_base_path / "rebuttal_patterns.json"

        try:
            rebuttals_data = []
            for rebuttal in self.rebuttal_patterns:
                rebuttal_dict = {
                    'rebuttal_id': rebuttal.rebuttal_id,
                    'objection_category': rebuttal.objection_category.value,
                    'objection_text': rebuttal.objection_text,
                    'rebuttal_text': rebuttal.rebuttal_text,
                    'rebuttal_type': rebuttal.rebuttal_type.value,
                    'effectiveness_score': rebuttal.effectiveness_score,
                    'usage_count': rebuttal.usage_count,
                    'success_rate': rebuttal.success_rate,
                    'context_tags': rebuttal.context_tags,
                    'advisor': rebuttal.advisor,
                    'date_created': rebuttal.date_created.isoformat(),
                    'last_used': rebuttal.last_used.isoformat() if rebuttal.last_used else None,
                    'client_segments': rebuttal.client_segments,
                    'estate_value_range': rebuttal.estate_value_range,
                    'follow_up_actions': rebuttal.follow_up_actions
                }
                rebuttals_data.append(rebuttal_dict)

            with open(rebuttals_file, 'w') as f:
                json.dump(rebuttals_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving rebuttals: {e}")

    async def _create_default_rebuttals(self):
        """Create some default rebuttals for common objections."""
        default_rebuttals = [
            RebuttalPattern(
                rebuttal_id="COST_001",
                objection_category=ObjectionCategory.COST,
                objection_text="This is too expensive for me",
                rebuttal_text="I understand cost is a concern. Let me show you how this investment typically saves families 3-5 times its cost in taxes and probate fees. What's most important - the upfront cost or the long-term savings for your family?",
                rebuttal_type=RebuttalType.COST_BENEFIT,
                effectiveness_score=0.85,
                usage_count=0,
                success_rate=0.85,
                context_tags=["high_value", "tax_savings"],
                advisor="Default",
                date_created=datetime.now(),
                client_segments=["affluent", "high_net_worth"],
                estate_value_range=(500_000, 10_000_000),
                follow_up_actions=["Provide specific ROI calculation", "Show case study examples"]
            ),
            RebuttalPattern(
                rebuttal_id="COMPLEXITY_001",
                objection_category=ObjectionCategory.COMPLEXITY,
                objection_text="This seems too complicated for me to understand",
                rebuttal_text="You're absolutely right that estate planning can seem overwhelming at first. That's exactly why we break it down into simple steps and handle all the complexity for you. Think of me as your guide - you don't need to understand how a car engine works to drive safely, right?",
                rebuttal_type=RebuttalType.REFRAME,
                effectiveness_score=0.78,
                usage_count=0,
                success_rate=0.78,
                context_tags=["simplification", "guidance"],
                advisor="Default",
                date_created=datetime.now(),
                client_segments=["middle_market", "first_time_planners"],
                estate_value_range=(100_000, 5_000_000),
                follow_up_actions=["Explain next steps simply", "Provide educational materials"]
            ),
            RebuttalPattern(
                rebuttal_id="TIMING_001",
                objection_category=ObjectionCategory.TIMING,
                objection_text="I want to think about it and decide later",
                rebuttal_text="I completely understand wanting to think this through - it shows you're taking this seriously. Help me understand what specifically you'd like to think about? Often, the things people want to consider are things we can address right now.",
                rebuttal_type=RebuttalType.QUESTION_BACK,
                effectiveness_score=0.72,
                usage_count=0,
                success_rate=0.72,
                context_tags=["discovery", "objection_handling"],
                advisor="Default",
                date_created=datetime.now(),
                client_segments=["all"],
                estate_value_range=(0, float('inf')),
                follow_up_actions=["Identify specific concerns", "Address each concern individually"]
            )
        ]

        for rebuttal in default_rebuttals:
            self.rebuttal_patterns.append(rebuttal)

        await self._save_rebuttals()
        self.logger.info("Created default rebuttals")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the rebuttal library."""
        if not self.rebuttal_patterns:
            return {"total_rebuttals": 0}

        total_rebuttals = len(self.rebuttal_patterns)
        avg_effectiveness = sum(r.effectiveness_score for r in self.rebuttal_patterns) / total_rebuttals
        total_usage = sum(r.usage_count for r in self.rebuttal_patterns)

        category_counts = {}
        type_counts = {}

        for rebuttal in self.rebuttal_patterns:
            category = rebuttal.objection_category.value
            category_counts[category] = category_counts.get(category, 0) + 1

            rebuttal_type = rebuttal.rebuttal_type.value
            type_counts[rebuttal_type] = type_counts.get(rebuttal_type, 0) + 1

        return {
            "total_rebuttals": total_rebuttals,
            "average_effectiveness": avg_effectiveness,
            "total_usage_count": total_usage,
            "category_distribution": category_counts,
            "type_distribution": type_counts,
            "most_effective": max(self.rebuttal_patterns, key=lambda r: r.effectiveness_score).rebuttal_id if self.rebuttal_patterns else None,
            "most_used": max(self.rebuttal_patterns, key=lambda r: r.usage_count).rebuttal_id if self.rebuttal_patterns else None
        }


# Example usage and testing
async def main():
    """Example usage of RebuttalLibrary."""
    library = RebuttalLibrary()

    # Test objection analysis
    objection = "This trust thing sounds way too expensive for what I get"
    client_context = {
        "estate_value": 2_500_000,
        "segment": "affluent",
        "family_structure": "married with children"
    }

    print("Testing objection analysis...")
    analysis = await library.analyze_objection(objection, client_context)

    print(f"\nObjection: {analysis.original_objection}")
    print(f"Category: {analysis.category.value}")
    print(f"Emotional Intensity: {analysis.emotional_intensity:.2f}")
    print(f"Specific Concerns: {analysis.specific_concerns}")
    print(f"Confidence Score: {analysis.confidence_score:.2f}")

    # Test rebuttal recommendations
    print("\nGetting rebuttal recommendations...")
    recommendations = library.get_effective_rebuttals(analysis)

    for i, rec in enumerate(recommendations, 1):
        print(f"\n--- Recommendation {i} ---")
        print(f"Rebuttal ID: {rec.rebuttal_pattern.rebuttal_id}")
        print(f"Relevance Score: {rec.relevance_score:.2f}")
        print(f"Effectiveness: {rec.rebuttal_pattern.effectiveness_score:.2f}")
        print(f"Type: {rec.rebuttal_pattern.rebuttal_type.value}")
        print(f"Rebuttal: {rec.rebuttal_pattern.rebuttal_text}")
        print(f"Customization: {', '.join(rec.customization_suggestions)}")
        print(f"Follow-up: {', '.join(rec.follow_up_strategy)}")

    # Show library statistics
    stats = library.get_statistics()
    print(f"\nLibrary Statistics:")
    print(f"Total Rebuttals: {stats['total_rebuttals']}")
    print(f"Average Effectiveness: {stats['average_effectiveness']:.2f}")
    print(f"Category Distribution: {stats['category_distribution']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())