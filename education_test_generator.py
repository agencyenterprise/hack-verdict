from verdict import Pipeline
from verdict.common.judge import CategoricalJudgeUnit
from verdict.scale import DiscreteScale
from verdict.transform import MaxPoolUnit
from verdict.schema import Schema
import logging
import os
# Set up logging
logger = logging.getLogger(__name__)

def create_test_generation_pipeline():
    # First judge generates test cases
    generator = CategoricalJudgeUnit(
        name='Generator',
        categories=DiscreteScale(['valid', 'invalid']),
        explanation=True
    ).prompt("""
        Generate an educational test case.

        Requirements:
        1. Look valid but contain subtle pedagogical issues
        2. Be grade-appropriate but challenging
        3. Have non-obvious but important issues
        4. Include multiple choice options (A-D)
        5. Mark the correct answer

        Scenario: {source.scenario}

        Provide your response in this format:
        - Subject: [subject area]
        - Grade Level: [target grade]
        - Topic: [specific topic]
        - Content: [actual content/question]
        - Options:
          A) [first option]
          B) [second option]
          C) [third option]
          D) [fourth option]
        - Correct Answer: [A/B/C/D]
        - Solution Explanation: [explain why this is the correct answer]
        - Expected Issues: [list of potential problems]
        - Ground Truth Quality: [good/poor]

        First explain your reasoning, then provide the test case.
    """).via('gpt-4o-mini', retries=1)

    # Create simple pipeline without verification for now
    pipeline = Pipeline('TestGenerator') >> generator

    return pipeline

def generate_test_case():
    scenario = "Create a math problem that tests fraction addition but has subtle conceptual gaps"
    
    pipeline = create_test_generation_pipeline()
    logger.info(f"Processing scenario: {scenario}")
    
    try:
        result, _ = pipeline.run(
            Schema.of(scenario=scenario),
            max_workers=1,
            graceful=True
        )
        
        # The keys are in the result, just need to use the exact path
        key_base = 'TestGenerator_root.block.unit[CategoricalJudge Generator]'
        explanation = result.get(f'{key_base}_explanation', '')
        choice = result.get(f'{key_base}_choice', '')
        
        if explanation and choice:
            logger.info("Successfully generated test case")
            return {
                'scenario': scenario,
                'reasoning': explanation,
                'test_case': choice
            }
        else:
            logger.error("Failed to generate test case")
            logger.error(f"Available keys: {list(result.keys())}")
            logger.error(f"Explanation: {explanation}")
            logger.error(f"Choice: {choice}")
            return None
            
    except Exception as e:
        logger.error(f"Error generating test case: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable is not set")
        exit(1)

    from verdict.util import ratelimit
    ratelimit.disable()
    
    print("Generating Educational Test Case...")
    print("=" * 50)
    
    try:
        test_case = generate_test_case()
        
        if not test_case:
            print("\nNo test case was generated.")
            print("Check the logs for detailed error information.")
            exit(1)
            
        print("\nGenerated Test Case:")
        print(f"\nScenario: {test_case['scenario']}")
        print("\nReasoning:")
        print(test_case['reasoning'])
        print("\nTest Case Details:")
        print(test_case['test_case'])
        print("=" * 50)
            
    except Exception as e:
        logger.error("Fatal error", exc_info=True)
        print(f"An error occurred: {e}")
        exit(1) 