#!/usr/bin/env python3
"""
Test script to debug the chat functionality
Run this to verify the backend works correctly
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.stellar_crew import run_crew, create_general_query_tasks

def test_backend_directly():
    """Test the backend without Streamlit"""
    test_queries = [
        "What patterns do you see in successful deals?",
        "Who are my top clients?",
        "What is the average deal size?"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Testing Query: {query}")
        print(f"{'='*60}")

        try:
            # Create tasks
            print(f"Creating tasks...")
            tasks = create_general_query_tasks(query)
            print(f"Tasks created: {tasks}")
            print(f"Tasks type: {type(tasks)}")

            # Run crew
            print(f"\nCalling run_crew...")
            response = run_crew(tasks)

            # Analyze response
            print(f"\n--- RESPONSE ANALYSIS ---")
            print(f"Response type: {type(response)}")
            print(f"Response is None: {response is None}")
            print(f"Response is string: {isinstance(response, str)}")
            if response:
                print(f"Response length: {len(str(response))} characters")
                print(f"Response (first 500 chars):\n{str(response)[:500]}")
            else:
                print("WARNING: Response is None or empty!")

        except Exception as e:
            print(f"\n!!! ERROR !!!")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("Test completed. Check the output above for any issues.")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_backend_directly()