#!/usr/bin/env python3
"""
Quick Response Test
Tests different query methods for speed comparison
"""

import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agent_tools import vector_tool

def test_vector_speed():
    """Test direct vector tool speed"""
    print("🧪 Testing Vector Tool Speed")
    print("="*50)

    test_queries = [
        "What patterns do you see in successful deals?",
        "Who are the top clients?",
        "What objections do clients have?"
    ]

    for query in test_queries:
        print(f"\n📝 Query: {query}")

        start_time = time.time()
        try:
            result = vector_tool._run(query)
            end_time = time.time()

            print(f"⏱️ Time: {end_time - start_time:.2f} seconds")
            print(f"📄 Result length: {len(result)} characters")
            print(f"📋 Preview: {result[:150]}...")

        except Exception as e:
            end_time = time.time()
            print(f"❌ Error: {e}")
            print(f"⏱️ Time to error: {end_time - start_time:.2f} seconds")

    print(f"\n{'='*50}")
    print("✅ Vector speed test complete!")

def main():
    print("🚀 Quick Response Test for Stellar Connect")
    print("="*60)

    # Test vector tool
    test_vector_speed()

    print("\n💡 Recommendations:")
    print("- Use 'Fast Vector' mode for 2-3 second responses")
    print("- Use 'Fast CrewAI' mode for 15-30 second structured analysis")
    print("- Avoid 'Standard' mode unless you need complex analysis")

    print(f"\n🎯 Access your fast dashboard at: http://localhost:8506")

if __name__ == "__main__":
    main()