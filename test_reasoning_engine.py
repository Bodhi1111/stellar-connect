#!/usr/bin/env python3
"""
Test runner for the Estate Reasoning Engine
Demonstrates the complete cognitive pipeline in action.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.reasoning import (
    EstateReasoningEngine,
    ReasoningConfig,
    ExecutionMode,
    ReasoningStatus
)


async def main():
    """Demonstrate the Estate Reasoning Engine capabilities."""
    print("🧠 Stellar Connect - Estate Reasoning Engine Demo")
    print("=" * 60)

    # Configure engine
    config = ReasoningConfig(
        execution_mode=ExecutionMode.STANDARD,
        enable_self_correction=True,
        parallel_execution=True
    )

    engine = EstateReasoningEngine(config)

    # Test queries
    test_queries = [
        {
            "query": "I need to create a trust for my $5 million estate to benefit my spouse and two children while minimizing estate taxes",
            "description": "Complete Trust Planning Query"
        },
        {
            "query": "How can I protect my business assets from potential creditors while planning for succession to my daughter?",
            "description": "Asset Protection & Business Succession"
        },
        {
            "query": "I need help with estate planning",
            "description": "Incomplete Query (Should Need Clarification)"
        }
    ]

    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        description = test_case["description"]

        print(f"\n📋 Test {i}: {description}")
        print("-" * 40)
        print(f"Query: {query}")
        print()

        try:
            result = await engine.process_query(query)

            print(f"🔍 Result ID: {result.reasoning_id}")
            print(f"⏱️  Execution Time: {result.execution_time:.2f}s")
            print(f"📊 Status: {result.status.value.upper()}")

            if result.status == ReasoningStatus.NEEDS_CLARIFICATION:
                print(f"❓ Needs Clarification:")
                for question in result.clarifying_questions:
                    print(f"   • {question}")

            elif result.status == ReasoningStatus.COMPLETED:
                print(f"🎯 Confidence: {result.confidence_score:.2f}")
                print(f"💼 Business Value: {result.synthesis.business_value_score:.2f}")
                print(f"📈 Success Probability: {result.synthesis.success_probability:.2f}")

                print(f"\n🎪 Primary Recommendation:")
                print(f"   {result.synthesis.primary_recommendation}")

                if result.synthesis.key_findings:
                    print(f"\n🔑 Key Findings:")
                    for finding in result.synthesis.key_findings[:3]:
                        print(f"   • {finding}")

                if result.synthesis.next_actions:
                    print(f"\n⚡ Next Actions:")
                    for action in result.synthesis.next_actions[:3]:
                        print(f"   • {action}")

                print(f"\n🏗️  Implementation: {result.synthesis.implementation_complexity}")

            elif result.status == ReasoningStatus.FAILED:
                print(f"❌ Error: {result.error_message}")

            # Show reasoning chain
            print(f"\n🔗 Reasoning Chain:")
            for step in result.reasoning_chain:
                print(f"   → {step}")

        except Exception as e:
            print(f"❌ Error processing query: {e}")

    # System Health Check
    print(f"\n🏥 System Health Check")
    print("=" * 30)

    health = await engine.health_check()
    print(f"Overall Status: {health['status'].upper()}")
    print(f"Active Sessions: {health['active_sessions']}")
    print(f"Execution Mode: {health['configuration']['execution_mode']}")

    print(f"\n📦 Component Health:")
    for component, status in health['components'].items():
        status_emoji = "✅" if status.get('status') == 'healthy' else "❌"
        print(f"   {status_emoji} {component.replace('_', ' ').title()}: {status.get('status', 'unknown')}")

    print(f"\n🎉 Demo Complete!")
    print("The Estate Reasoning Engine successfully demonstrated:")
    print("   • Query validation and preprocessing")
    print("   • Multi-step analysis planning")
    print("   • Parallel task execution")
    print("   • Quality auditing and self-correction")
    print("   • Strategic synthesis and insights")
    print("   • Complete cognitive reasoning pipeline")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise in demo
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        sys.exit(1)