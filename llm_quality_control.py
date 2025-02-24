from verdict import Pipeline, Layer
from verdict.common.judge import CategoricalJudgeUnit
from verdict.scale import DiscreteScale
from verdict.schema import Schema
from openai import OpenAI
import logging
import os

logger = logging.getLogger(__name__)

# Initial flawed content about photosynthesis
INITIAL_CONTENT = """
Subject: Biology
Grade Level: 7th Grade
Topic: Photosynthesis
Content: Photosynthesis is how plants make food. They use sunlight and water to create glucose. 
The process happens in the plant's leaves where special cells capture sunlight. 
The glucose is then used by the plant for energy or stored for later.

Expected Learning Outcomes:
1. Understand that plants make their own food
2. Know that sunlight is important for plants
3. Learn that glucose is created during photosynthesis
"""

class ContentQualityControl:
    def __init__(self, content_type="text", max_iterations=3):
        self.content_type = content_type
        self.max_iterations = max_iterations
        self.pipeline = self.create_pipeline()
        self.client = OpenAI()

    def create_pipeline(self):
        # Quality assessment judge
        quality_judge = CategoricalJudgeUnit(
            name='QualityJudge',
            categories=DiscreteScale(['pass', 'revise']),
            explanation=True
        ).prompt("""
            You are an expert educational content reviewer. Evaluate this content for accuracy, 
            completeness, and educational value.
            
            Content to Review: {source.content}
            Requirements: {source.requirements}
            
            Consider these aspects:
            1. Scientific accuracy
            2. Completeness of explanation
            3. Grade-appropriate language
            4. Key concepts coverage
            5. Potential misconceptions
            6. Learning outcome alignment
            
            First provide a detailed analysis of any issues found, then decide:
            - pass: Content meets educational standards and is scientifically accurate
            - revise: Content needs improvements (specify exactly what needs to change)
        """)

        pipeline = Pipeline('QualityControl')
        pipeline = pipeline >> Layer([quality_judge])
        return pipeline

    def generate_improved_content(self, original_content, feedback):
        """Generate improved content based on feedback"""
        prompt = f"""
        Improve this educational content based on the reviewer's feedback:

        Original Content:
        {original_content}

        Reviewer's Feedback:
        {feedback}

        Requirements:
        - Fix all scientific inaccuracies
        - Add missing key concepts
        - Maintain grade-appropriate language
        - Address all identified issues
        - Keep the same format but improve the content

        Provide the improved version while maintaining the same structure.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating improved content: {e}")
            return None

    def evaluate_content(self, content, requirements):
        """Evaluate content quality"""
        try:
            result, _ = self.pipeline.run(
                Schema.of(
                    content=content,
                    requirements=requirements,
                    content_type=self.content_type
                ),
                max_workers=1,
                graceful=True
            )

            logger.debug(f"Available keys in result: {list(result.keys())}")

            base_path = 'QualityControl_root.block.layer[0].unit[CategoricalJudge QualityJudge]'
            verdict = result.get(f'{base_path}_choice', '')
            explanation = result.get(f'{base_path}_explanation', '')

            if not verdict or not explanation:
                logger.error(f"Missing results. Available keys: {list(result.keys())}")
                return None

            return {
                'verdict': verdict,
                'explanation': explanation,
                'content': content
            }

        except Exception as e:
            logger.error(f"Error evaluating content: {e}", exc_info=True)
            return None

    def improve_content(self, content, requirements):
        """Iteratively improve content until it passes quality control"""
        iterations = 0
        current_content = content
        
        while iterations < self.max_iterations:
            iterations += 1
            print(f"\nIteration {iterations}")
            print("=" * 50)
            
            # Evaluate current content
            result = self.evaluate_content(current_content, requirements)
            if not result:
                print("Failed to evaluate content")
                return None
            
            print(f"\nVerdict: {result['verdict']}")
            print("\nFeedback:")
            print(result['explanation'])
            
            # Check if content passes
            if result['verdict'] == 'pass':
                print("\nContent meets quality standards!")
                return result
            
            # Generate improved version based on feedback
            print("\nGenerating improved version based on feedback...")
            improved_content = self.generate_improved_content(
                current_content, 
                result['explanation']
            )
            
            if not improved_content:
                print("Failed to generate improved content")
                return None
                
            print("\nImproved Content:")
            print(improved_content)
            
            current_content = improved_content
            
        print(f"\nHit max iterations ({self.max_iterations}) without reaching quality standards")
        return result

def main():
    requirements = """
    Educational content requirements:
    1. Scientifically accurate explanation of photosynthesis
    2. Include all key components (CO2, water, sunlight, chlorophyll)
    3. Explain the role of each component
    4. Mention both products (glucose and oxygen)
    5. Grade-appropriate language for 7th grade
    6. Clear learning outcomes
    """
    
    print("\nStarting Content Improvement Process...")
    print("=" * 50)
    print("\nInitial Content:")
    print(INITIAL_CONTENT)
    
    qc = ContentQualityControl(content_type="text", max_iterations=5)  # Increased max iterations
    final_result = qc.improve_content(INITIAL_CONTENT, requirements)
    
    if final_result and final_result['verdict'] == 'pass':
        print("\nFinal Improved Content:")
        print("=" * 30)
        print(final_result['content'])
        print("\nFinal Quality Assessment:")
        print(final_result['explanation'])
    else:
        print("\nFailed to achieve satisfactory content quality")

if __name__ == "__main__":
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable is not set")
        exit(1)
    main() 