from verdict import Pipeline, Layer
from verdict.common.judge import CategoricalJudgeUnit
from verdict.scale import DiscreteScale
from verdict.schema import Schema
from openai import OpenAI
import logging
import os

logger = logging.getLogger(__name__)

class QualityControlEvaluator:
    def __init__(self):
        self.client = OpenAI()
        self.pipeline = self.create_evaluation_pipeline()

    def create_evaluation_pipeline(self):
        # Meta-evaluator to assess QC system decisions
        meta_judge = CategoricalJudgeUnit(
            name='MetaJudge',
            categories=DiscreteScale(['reliable', 'questionable', 'failure']),
            explanation=True
        ).prompt("""
            Evaluate the reliability of this quality control assessment.
            
            Original Content: {source.content}
            QC System Assessment: {source.qc_assessment}
            QC Decision: {source.qc_decision}
            Content Requirements: {source.requirements}
            
            Consider these meta-evaluation criteria:
            1. Assessment Completeness
            - Does the QC system consider all important aspects?
            - Are there overlooked issues or blind spots?
            
            2. Reasoning Quality
            - Is the assessment logically sound?
            - Are conclusions well-supported?
            
            3. Standards Alignment
            - Does the assessment align with requirements?
            - Are quality standards consistently applied?
            
            4. Failure Detection
            - Are subtle issues caught?
            - Are there false positives/negatives?
            
            5. Feedback Quality
            - Is feedback specific and actionable?
            - Are improvement suggestions clear?
            
            First provide detailed analysis, then rate as:
            - reliable: QC assessment is thorough and trustworthy
            - questionable: Some concerns about QC assessment
            - failure: Significant QC system failures detected
        """)

        # Specific failure analysis judge
        failure_analyzer = CategoricalJudgeUnit(
            name='FailureAnalyzer',
            categories=DiscreteScale(['systematic', 'contextual', 'random']),
            explanation=True
        ).prompt("""
            Analyze the type of quality control failure detected.
            
            Original Content: {source.content}
            QC System Assessment: {source.qc_assessment}
            Meta-Evaluation: {previous.explanation}
            
            Identify failure patterns:
            1. Systematic Failures
            - Consistent blind spots
            - Recurring assessment gaps
            - Pattern of missed issues
            
            2. Contextual Failures
            - Domain-specific misunderstandings
            - Context-dependent errors
            - Scope limitations
            
            3. Random Failures
            - Inconsistent assessments
            - Unpredictable errors
            - No clear pattern
            
            First explain the failure analysis, then categorize as:
            - systematic: Clear pattern of similar failures
            - contextual: Context-dependent failures
            - random: No consistent pattern
        """)

        # Create pipeline with parallel evaluation paths
        pipeline = Pipeline('QCEvaluator')
        pipeline = pipeline >> Layer([meta_judge])
        pipeline = pipeline >> Layer([failure_analyzer])
        
        return pipeline

    def evaluate_qc_system(self, content, qc_assessment, qc_decision, requirements):
        """Evaluate the quality control system's assessment"""
        try:
            result, _ = self.pipeline.run(
                Schema.of(
                    content=content,
                    qc_assessment=qc_assessment,
                    qc_decision=qc_decision,
                    requirements=requirements
                ),
                max_workers=1,
                graceful=True
            )

            # Get results using correct paths
            base_path = 'QCEvaluator_root.block.layer'
            meta_verdict = result.get(f'{base_path}[0].unit[CategoricalJudge MetaJudge]_choice', '')
            meta_explanation = result.get(f'{base_path}[0].unit[CategoricalJudge MetaJudge]_explanation', '')
            
            failure_type = result.get(f'{base_path}[1].unit[CategoricalJudge FailureAnalyzer]_choice', '')
            failure_explanation = result.get(f'{base_path}[1].unit[CategoricalJudge FailureAnalyzer]_explanation', '')

            return {
                'meta_evaluation': {
                    'verdict': meta_verdict,
                    'explanation': meta_explanation
                },
                'failure_analysis': {
                    'type': failure_type,
                    'explanation': failure_explanation
                }
            }

        except Exception as e:
            logger.error(f"Error evaluating QC system: {e}", exc_info=True)
            return None

def main():
    # Example of a QC system potentially missing issues
    content = """
    Subject: Chemistry
    Grade Level: 9th Grade
    Topic: Chemical Reactions
    
    Content: Chemical reactions occur when atoms rearrange to form new substances.
    The reaction can be shown using chemical equations where reactants → products.
    For example: Na + Cl → NaCl
    
    Learning Outcomes:
    1. Understand basic chemical reactions
    2. Learn to write chemical equations
    3. Identify reactants and products
    """

    qc_assessment = """
    The content provides a clear introduction to chemical reactions.
    The explanation is grade-appropriate and includes a practical example.
    The learning outcomes align with the content.
    Recommended for use in 9th-grade chemistry.
    """

    qc_decision = "pass"

    requirements = """
    Chemistry content requirements:
    1. Scientific accuracy in all concepts
    2. Proper chemical equation notation
    3. Balance in all equations
    4. Safety considerations
    5. Common misconception prevention
    6. Clear learning progression
    """

    print("\nEvaluating Quality Control System...")
    print("=" * 50)
    
    evaluator = QualityControlEvaluator()
    result = evaluator.evaluate_qc_system(content, qc_assessment, qc_decision, requirements)
    
    if result:
        print("\nMeta-Evaluation Results:")
        print("=" * 30)
        print(f"Verdict: {result['meta_evaluation']['verdict']}")
        print("\nDetailed Analysis:")
        print(result['meta_evaluation']['explanation'])
        
        if result['meta_evaluation']['verdict'] != 'reliable':
            print("\nFailure Analysis:")
            print("=" * 30)
            print(f"Failure Type: {result['failure_analysis']['type']}")
            print("\nFailure Details:")
            print(result['failure_analysis']['explanation'])
    else:
        print("Failed to evaluate QC system")

if __name__ == "__main__":
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable is not set")
        exit(1)
    main() 
