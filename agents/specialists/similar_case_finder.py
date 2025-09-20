"""
Similar Case Finder for Stellar Connect
Utility class for finding similar cases based on estate characteristics

This module provides advanced case matching capabilities for estate planning
scenarios, using multiple similarity algorithms and contextual matching.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import json
import math
from enum import Enum


class SimilarityAlgorithm(Enum):
    """Available similarity calculation algorithms."""
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    JACCARD = "jaccard"
    WEIGHTED_COMPOSITE = "weighted_composite"


@dataclass
class CaseMatchCriteria:
    """Criteria for matching similar cases."""
    estate_value_range: float = 0.5  # 50% variance allowed
    family_structure_weight: float = 0.25
    concerns_weight: float = 0.45
    estate_value_weight: float = 0.30
    outcome_filter: Optional[List[str]] = None
    date_range_days: Optional[int] = None
    min_similarity_threshold: float = 0.3
    max_results: int = 10


@dataclass
class SimilarCase:
    """Represents a similar case found by the finder."""
    case_id: str
    client_name: str
    estate_value: float
    family_structure: str
    primary_concerns: List[str]
    solutions_implemented: List[str]
    outcome: str
    success_factors: List[str]
    lessons_learned: List[str]
    similarity_score: float
    case_date: datetime
    advisor: str
    similarity_breakdown: Dict[str, float]
    matching_factors: List[str]


class SimilarCaseFinder:
    """
    Advanced similar case finder for estate planning scenarios.

    Features:
    - Multiple similarity algorithms
    - Weighted factor matching
    - Contextual relevance scoring
    - Outcome-based filtering
    - Time-based relevance
    """

    def __init__(self, knowledge_base_path: str = "data/knowledge_base"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.knowledge_base_path.mkdir(parents=True, exist_ok=True)

        self.cases_database: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

        # Load existing cases
        asyncio.create_task(self._load_cases())

    async def find_matches(
        self,
        estate_value: float,
        family_structure: str,
        objection_types: List[str],
        primary_concerns: Optional[List[str]] = None,
        criteria: Optional[CaseMatchCriteria] = None,
        algorithm: SimilarityAlgorithm = SimilarityAlgorithm.WEIGHTED_COMPOSITE
    ) -> List[SimilarCase]:
        """
        Find similar cases based on estate characteristics.

        Args:
            estate_value: Target estate value
            family_structure: Family structure description
            objection_types: Types of objections encountered
            primary_concerns: List of primary client concerns
            criteria: Matching criteria configuration
            algorithm: Similarity algorithm to use

        Returns:
            List of similar cases sorted by relevance
        """
        if criteria is None:
            criteria = CaseMatchCriteria()

        # Combine objection types and concerns
        all_concerns = list(objection_types)
        if primary_concerns:
            all_concerns.extend(primary_concerns)

        # Filter cases by basic criteria
        candidate_cases = self._filter_candidates(
            estate_value, criteria.outcome_filter, criteria.date_range_days
        )

        if not candidate_cases:
            self.logger.info("No candidate cases found matching basic criteria")
            return []

        # Calculate similarity scores
        similar_cases = []
        for case_data in candidate_cases:
            similarity_score, breakdown, matching_factors = self._calculate_similarity(
                estate_value=estate_value,
                family_structure=family_structure,
                concerns=all_concerns,
                case_data=case_data,
                criteria=criteria,
                algorithm=algorithm
            )

            if similarity_score >= criteria.min_similarity_threshold:
                similar_case = SimilarCase(
                    case_id=case_data['case_id'],
                    client_name=case_data['client_name'],
                    estate_value=case_data['estate_value'],
                    family_structure=case_data['family_structure'],
                    primary_concerns=case_data['primary_concerns'],
                    solutions_implemented=case_data['solutions_implemented'],
                    outcome=case_data['outcome'],
                    success_factors=case_data['success_factors'],
                    lessons_learned=case_data['lessons_learned'],
                    similarity_score=similarity_score,
                    case_date=datetime.fromisoformat(case_data['case_date']),
                    advisor=case_data['advisor'],
                    similarity_breakdown=breakdown,
                    matching_factors=matching_factors
                )
                similar_cases.append(similar_case)

        # Sort by similarity score and apply time-based relevance boost
        similar_cases = self._apply_recency_boost(similar_cases)
        similar_cases.sort(key=lambda x: x.similarity_score, reverse=True)

        # Limit results
        return similar_cases[:criteria.max_results]

    def _filter_candidates(
        self,
        estate_value: float,
        outcome_filter: Optional[List[str]],
        date_range_days: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Filter cases by basic criteria."""
        candidates = []

        cutoff_date = None
        if date_range_days:
            cutoff_date = datetime.now() - timedelta(days=date_range_days)

        for case in self.cases_database:
            # Estate value range filter (within 2x range)
            case_value = case.get('estate_value', 0)
            if case_value > 0 and estate_value > 0:
                ratio = max(case_value, estate_value) / min(case_value, estate_value)
                if ratio > 10.0:  # Skip cases with very different estate values
                    continue

            # Outcome filter
            if outcome_filter and case.get('outcome') not in outcome_filter:
                continue

            # Date range filter
            if cutoff_date:
                case_date = datetime.fromisoformat(case.get('case_date', '2020-01-01'))
                if case_date < cutoff_date:
                    continue

            candidates.append(case)

        return candidates

    def _calculate_similarity(
        self,
        estate_value: float,
        family_structure: str,
        concerns: List[str],
        case_data: Dict[str, Any],
        criteria: CaseMatchCriteria,
        algorithm: SimilarityAlgorithm
    ) -> Tuple[float, Dict[str, float], List[str]]:
        """Calculate similarity score between target and case."""
        if algorithm == SimilarityAlgorithm.WEIGHTED_COMPOSITE:
            return self._weighted_composite_similarity(
                estate_value, family_structure, concerns, case_data, criteria
            )
        elif algorithm == SimilarityAlgorithm.EUCLIDEAN:
            return self._euclidean_similarity(
                estate_value, family_structure, concerns, case_data, criteria
            )
        elif algorithm == SimilarityAlgorithm.COSINE:
            return self._cosine_similarity(
                estate_value, family_structure, concerns, case_data, criteria
            )
        elif algorithm == SimilarityAlgorithm.JACCARD:
            return self._jaccard_similarity(
                estate_value, family_structure, concerns, case_data, criteria
            )
        else:
            return self._weighted_composite_similarity(
                estate_value, family_structure, concerns, case_data, criteria
            )

    def _weighted_composite_similarity(
        self,
        estate_value: float,
        family_structure: str,
        concerns: List[str],
        case_data: Dict[str, Any],
        criteria: CaseMatchCriteria
    ) -> Tuple[float, Dict[str, float], List[str]]:
        """Calculate weighted composite similarity score."""
        breakdown = {}
        matching_factors = []

        # Estate value similarity
        estate_sim = self._estate_value_similarity(estate_value, case_data.get('estate_value', 0))
        breakdown['estate_value'] = estate_sim
        if estate_sim > 0.7:
            matching_factors.append("similar_estate_value")

        # Family structure similarity
        family_sim = self._text_similarity(family_structure, case_data.get('family_structure', ''))
        breakdown['family_structure'] = family_sim
        if family_sim > 0.6:
            matching_factors.append("similar_family_structure")

        # Concerns similarity
        case_concerns = case_data.get('primary_concerns', [])
        concerns_sim = self._list_similarity(concerns, case_concerns)
        breakdown['concerns'] = concerns_sim
        if concerns_sim > 0.5:
            matching_factors.append("similar_concerns")

        # Calculate weighted score
        total_score = (
            estate_sim * criteria.estate_value_weight +
            family_sim * criteria.family_structure_weight +
            concerns_sim * criteria.concerns_weight
        )

        return total_score, breakdown, matching_factors

    def _euclidean_similarity(
        self,
        estate_value: float,
        family_structure: str,
        concerns: List[str],
        case_data: Dict[str, Any],
        criteria: CaseMatchCriteria
    ) -> Tuple[float, Dict[str, float], List[str]]:
        """Calculate Euclidean distance-based similarity."""
        # Convert to feature vectors
        target_vector = self._create_feature_vector(estate_value, family_structure, concerns)
        case_vector = self._create_feature_vector(
            case_data.get('estate_value', 0),
            case_data.get('family_structure', ''),
            case_data.get('primary_concerns', [])
        )

        # Calculate Euclidean distance
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(target_vector, case_vector)))

        # Convert distance to similarity (0-1 range)
        max_distance = math.sqrt(len(target_vector))  # Maximum possible distance
        similarity = 1.0 - (distance / max_distance)

        breakdown = {"euclidean_similarity": similarity}
        matching_factors = ["euclidean_match"] if similarity > 0.7 else []

        return similarity, breakdown, matching_factors

    def _cosine_similarity(
        self,
        estate_value: float,
        family_structure: str,
        concerns: List[str],
        case_data: Dict[str, Any],
        criteria: CaseMatchCriteria
    ) -> Tuple[float, Dict[str, float], List[str]]:
        """Calculate cosine similarity."""
        target_vector = self._create_feature_vector(estate_value, family_structure, concerns)
        case_vector = self._create_feature_vector(
            case_data.get('estate_value', 0),
            case_data.get('family_structure', ''),
            case_data.get('primary_concerns', [])
        )

        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(target_vector, case_vector))
        magnitude_target = math.sqrt(sum(a ** 2 for a in target_vector))
        magnitude_case = math.sqrt(sum(b ** 2 for b in case_vector))

        if magnitude_target == 0 or magnitude_case == 0:
            similarity = 0.0
        else:
            similarity = dot_product / (magnitude_target * magnitude_case)

        breakdown = {"cosine_similarity": similarity}
        matching_factors = ["cosine_match"] if similarity > 0.7 else []

        return similarity, breakdown, matching_factors

    def _jaccard_similarity(
        self,
        estate_value: float,
        family_structure: str,
        concerns: List[str],
        case_data: Dict[str, Any],
        criteria: CaseMatchCriteria
    ) -> Tuple[float, Dict[str, float], List[str]]:
        """Calculate Jaccard similarity for categorical features."""
        # Create sets of features
        target_features = set(concerns + family_structure.split())
        case_features = set(
            case_data.get('primary_concerns', []) +
            case_data.get('family_structure', '').split()
        )

        # Calculate Jaccard similarity
        intersection = len(target_features & case_features)
        union = len(target_features | case_features)

        if union == 0:
            similarity = 0.0
        else:
            similarity = intersection / union

        breakdown = {"jaccard_similarity": similarity}
        matching_factors = ["jaccard_match"] if similarity > 0.5 else []

        return similarity, breakdown, matching_factors

    def _estate_value_similarity(self, value1: float, value2: float) -> float:
        """Calculate similarity between estate values."""
        if value1 <= 0 or value2 <= 0:
            return 0.0

        ratio = min(value1, value2) / max(value1, value2)

        # Apply logarithmic scaling for large differences
        if ratio < 0.1:
            return 0.0
        elif ratio < 0.5:
            return 0.3 * (ratio / 0.5)
        else:
            return 0.3 + 0.7 * ((ratio - 0.5) / 0.5)

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using word overlap."""
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _list_similarity(self, list1: List[str], list2: List[str]) -> float:
        """Calculate similarity between two lists of strings."""
        if not list1 or not list2:
            return 0.0

        set1 = set(item.lower() for item in list1)
        set2 = set(item.lower() for item in list2)

        intersection = set1 & set2
        union = set1 | set2

        return len(intersection) / len(union) if union else 0.0

    def _create_feature_vector(
        self, estate_value: float, family_structure: str, concerns: List[str]
    ) -> List[float]:
        """Create a feature vector for similarity calculations."""
        vector = []

        # Estate value (normalized to 0-1 range)
        vector.append(min(estate_value / 10_000_000, 1.0))  # Normalize to $10M max

        # Family structure features (binary indicators)
        family_keywords = ['married', 'single', 'children', 'divorced', 'widowed']
        for keyword in family_keywords:
            vector.append(1.0 if keyword in family_structure.lower() else 0.0)

        # Concern features (binary indicators)
        concern_keywords = [
            'tax', 'cost', 'complexity', 'time', 'trust', 'will',
            'protection', 'legacy', 'children', 'spouse'
        ]
        for keyword in concern_keywords:
            has_concern = any(keyword in concern.lower() for concern in concerns)
            vector.append(1.0 if has_concern else 0.0)

        return vector

    def _apply_recency_boost(self, cases: List[SimilarCase]) -> List[SimilarCase]:
        """Apply recency boost to similarity scores."""
        now = datetime.now()

        for case in cases:
            days_old = (now - case.case_date).days

            # Apply recency boost (more recent cases get higher scores)
            if days_old <= 30:
                boost_factor = 1.1  # 10% boost for cases within 30 days
            elif days_old <= 90:
                boost_factor = 1.05  # 5% boost for cases within 90 days
            elif days_old <= 365:
                boost_factor = 1.0  # No boost for cases within 1 year
            else:
                boost_factor = 0.95  # 5% penalty for cases older than 1 year

            case.similarity_score *= boost_factor

        return cases

    async def _load_cases(self):
        """Load cases from the knowledge base."""
        cases_file = self.knowledge_base_path / "similar_cases.json"

        if not cases_file.exists():
            self.logger.info("No similar cases file found, starting with empty database")
            return

        try:
            with open(cases_file, 'r') as f:
                self.cases_database = json.load(f)

            self.logger.info(f"Loaded {len(self.cases_database)} cases from knowledge base")

        except Exception as e:
            self.logger.error(f"Error loading cases database: {e}")
            self.cases_database = []

    async def add_case(self, case_data: Dict[str, Any]) -> bool:
        """Add a new case to the database."""
        try:
            # Validate required fields
            required_fields = [
                'case_id', 'client_name', 'estate_value', 'family_structure',
                'primary_concerns', 'outcome', 'case_date'
            ]

            for field in required_fields:
                if field not in case_data:
                    self.logger.error(f"Missing required field: {field}")
                    return False

            # Add default values for optional fields
            case_data.setdefault('solutions_implemented', [])
            case_data.setdefault('success_factors', [])
            case_data.setdefault('lessons_learned', [])
            case_data.setdefault('advisor', 'Unknown')

            # Add to database
            self.cases_database.append(case_data)

            # Save to file
            await self._save_cases()

            self.logger.info(f"Added new case: {case_data['case_id']}")
            return True

        except Exception as e:
            self.logger.error(f"Error adding case: {e}")
            return False

    async def _save_cases(self):
        """Save cases database to file."""
        cases_file = self.knowledge_base_path / "similar_cases.json"

        try:
            with open(cases_file, 'w') as f:
                json.dump(self.cases_database, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Error saving cases database: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the cases database."""
        if not self.cases_database:
            return {"total_cases": 0}

        total_cases = len(self.cases_database)
        outcomes = {}
        estate_values = []
        advisors = set()

        for case in self.cases_database:
            outcome = case.get('outcome', 'unknown')
            outcomes[outcome] = outcomes.get(outcome, 0) + 1

            estate_value = case.get('estate_value', 0)
            if estate_value > 0:
                estate_values.append(estate_value)

            advisors.add(case.get('advisor', 'Unknown'))

        avg_estate_value = sum(estate_values) / len(estate_values) if estate_values else 0

        return {
            "total_cases": total_cases,
            "outcomes": outcomes,
            "average_estate_value": avg_estate_value,
            "min_estate_value": min(estate_values) if estate_values else 0,
            "max_estate_value": max(estate_values) if estate_values else 0,
            "unique_advisors": len(advisors),
            "success_rate": outcomes.get('closed_won', 0) / total_cases if total_cases > 0 else 0
        }


# Example usage and testing
async def main():
    """Example usage of SimilarCaseFinder."""
    finder = SimilarCaseFinder()

    # Add some sample cases for testing
    sample_cases = [
        {
            "case_id": "CASE_001",
            "client_name": "John Smith",
            "estate_value": 2500000,
            "family_structure": "married with 2 adult children",
            "primary_concerns": ["tax minimization", "family harmony"],
            "solutions_implemented": ["revocable trust", "life insurance"],
            "outcome": "closed_won",
            "success_factors": ["clear value proposition", "addressed tax concerns"],
            "lessons_learned": ["Tax savings messaging was effective"],
            "case_date": "2024-01-15",
            "advisor": "Jane Doe"
        },
        {
            "case_id": "CASE_002",
            "client_name": "Mary Johnson",
            "estate_value": 3200000,
            "family_structure": "married with 3 children",
            "primary_concerns": ["estate tax", "business succession"],
            "solutions_implemented": ["irrevocable trust", "buy-sell agreement"],
            "outcome": "closed_won",
            "success_factors": ["business succession plan", "family involvement"],
            "lessons_learned": ["Business owners need succession planning"],
            "case_date": "2024-02-20",
            "advisor": "Bob Wilson"
        }
    ]

    for case in sample_cases:
        await finder.add_case(case)

    # Test case matching
    print("Testing similar case finding...")

    similar_cases = await finder.find_matches(
        estate_value=2800000,
        family_structure="married with 2 children",
        objection_types=["tax concerns"],
        primary_concerns=["minimize taxes", "protect family"],
        criteria=CaseMatchCriteria(max_results=5)
    )

    print(f"\nFound {len(similar_cases)} similar cases:")
    for case in similar_cases:
        print(f"\nCase ID: {case.case_id}")
        print(f"Similarity Score: {case.similarity_score:.3f}")
        print(f"Estate Value: ${case.estate_value:,.0f}")
        print(f"Matching Factors: {', '.join(case.matching_factors)}")
        print(f"Outcome: {case.outcome}")

    # Show statistics
    stats = finder.get_statistics()
    print(f"\nDatabase Statistics:")
    print(f"Total Cases: {stats['total_cases']}")
    print(f"Success Rate: {stats['success_rate']:.2%}")
    print(f"Average Estate Value: ${stats['average_estate_value']:,.0f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())