from src.sampler import Sampler
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='IGR Evaluator')
parser.add_argument('--feedback_content', type=str, default="You should not talk about Elephant", help='Feedback content for the sampler')
parser.add_argument('--model_name', type=str, required=True, help='Model name')
args = parser.parse_args()


feedback_content = args.feedback_content
sampler = Sampler(feedback_content, args.model_name)
sampler.eval_prompts()