import os
from typing import Tuple, Optional
from dataclasses import dataclass 

QA_TEMPLATE = """Given the instruction: {instruction}
I need to answer the query: {query}
{condition_text}
I think: {reason}
{hint_text}
Is this reasoning correct? Provide a True/False answer as well as your rationale.
Example Answer:
Rationale: [Your rationale here]
Answer: [True/False]"""

QA_ANSWER_TEMPLATE = """Rationale: {rationale}\nAnswer: {answer}"""

NAIVE_PROMPT_TEMPLATE = """ 
Query: {query}
How would you address the query while following the instruction? Provide your thought and answer. 
Example format: 
Thought: [Your thinking process] 
Answer: [Your answer]
""" # There is issue: for roleplay scenario, we should include instruction in the system prompt and not here (!) -- Need matching evaluation here with actual inference (!)

def prepare_naive_prompt(instruction, query):
    return NAIVE_PROMPT_TEMPLATE.format(instruction=instruction, query=query) # Matching actual inference

def parse_thought_answer(response: str) -> Tuple[str, str]:
    
    if "Thought:" not in response or "Answer:" not in response:
        # print("No thought or answer found in response: \n", response)
        return "", ""
    
    thought = response.split("Thought:")[1].split("Answer:")[0].strip()
    answer = response.split("Answer:")[1].strip()
    return thought, answer

def parse_evaluate_answer(response: str) -> Tuple[str, str]:
    
    if "Evaluation:" not in response or "Explanation:" not in response:
        print("No evaluation or explanation found in response: \n", response)
        return "", ""
    
    evaluation_str = response.split("Evaluation:")[1].split("Explanation:")[0].strip()
    explanation = response.split("Explanation:")[1].strip()
    true_strs = ["True", "TRUE"]
    false_strs = ["False", "FALSE"]
    if evaluation_str in true_strs:
        evaluation = True
    elif evaluation_str in false_strs:
        evaluation = False
    else:
        evaluation = ""
    return evaluation, explanation


PROPOSE_RESPONSE_TEMPLATE = """Thought: {thought}
Answer: {answer}"""


ALIGNMENT_PROMPT_TEMPLATE = """
Given your previous answer and rationale, evaluate whether the following idea is correct: 

{answer} 

Is this idea aligned with your rationale? Respond with True if acceptable, False if not acceptable.
Provide a brief explanation for your judgment.
Example format:
Evaluation: [Your evaluation]
Explanation: [Your explanation]
"""

EVALUATE_RESPONSE_TEMPLATE = """Evaluation: {evaluation}
Explanation: {explanation}"""

# We should not apply chat template in the output datapoint here, since BLM & SLM process things with different template

@dataclass 
class Rationale:
    prompt: str # original prompt
    correct_answer: str = "" # guide on 'how to answer' provided by Human
    naive_prompt: str = "" # naive prompt
    thought: str = "" # thought process of the proposed answer
    answer: str = "" # proposed answer by LLM
    evaluation: Optional[bool] = None # evaluation of the correct / wrong answer 
    explanation: str = "" # explanation of the evaluation
    
    
    @property
    def is_good(self):
        is_valid = (self.evaluation is not None) and (self.explanation != "") and (self.thought != "") and (self.answer != "")
        correct_evaluation = False 
        if self.correct_answer != "": # expect True evaluation
            correct_evaluation = (self.evaluation == True)
        return is_valid and correct_evaluation
      
    @property
    def alignment_prompt(self):
        return ALIGNMENT_PROMPT_TEMPLATE.format(answer=self.correct_answer)
    
    @property
    def propose_response(self):
        return PROPOSE_RESPONSE_TEMPLATE.format(thought=self.thought, answer=self.answer)
    
    @property 
    def query_evaluate_message(self):
        return [
            {"role": "user", "content": self.propose_prompt},
            {"role": "assistant", "content": self.propose_response},
            {"role": "user", "content": self.evaluate_prompt}
        ]
        
    @property 
    def query_alignment_message(self):
        return [
            {"role": "user", "content": self.propose_prompt},
            {"role": "assistant", "content": self.propose_response},
            {"role": "user", "content": self.alignment_prompt}
        ]
    
    @property
    def instruction(self):
        return self.propose_prompt.split('\n')[0].split(':')[-1].strip()
    
    @property 
    def naive_propose_message(self):
        return [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": self.naive_prompt}
        ]
   
class Reasoner: 
    
    def __init__(self, get_response, rationales, apply_chat_template=lambda x: x, n_answer_per_question=3):
        self.get_response = get_response
        self.apply_chat_template = apply_chat_template
        self.n_answer_per_question = n_answer_per_question
        self.rationales = rationales

    @classmethod 
    def make_from_tuples(cls, get_response, prompt_tuples=[], apply_chat_template=lambda x: x, n_answer_per_question=3):
        rationales = []
        for _ in range(n_answer_per_question):
            for (prompt, correct_answer, naive_prompt) in prompt_tuples:
                rationales.append(Rationale(prompt, correct_answer, naive_prompt))
                
        return cls(get_response=get_response, rationales=rationales, apply_chat_template=apply_chat_template, n_answer_per_question=n_answer_per_question)
        
    
    def save_rationales(self, rationale_dir):
        os.makedirs(rationale_dir, exist_ok=True)
        for idx, rationale in enumerate(self.rationales):
            file_path = os.path.join(rationale_dir, f"rationale_{idx}.json")
            rationale.save(file_path)
        
        print(f"Saved {len(self.rationales)} rationales to {rationale_dir}")
                
    def format_naive_prompt(self, rationale: Rationale):
        completion = "####"
        msg = rationale.naive_propose_message + [{"role": "assistant", "content": completion}]
        tmp = self.apply_chat_template(msg)
        if isinstance(tmp, list): # Case with API calls
            return rationale.naive_propose_message
        else:
            return tmp.split(completion)[0] # query prompt only for vLLM 
        
    def format_alignment_prompt(self, rationale: Rationale):
        completion = "####"
        msg = rationale.query_alignment_message + [{"role": "assistant", "content": completion}]
        tmp = self.apply_chat_template(msg)
        if isinstance(tmp, list): # Case with API calls
            return rationale.query_alignment_message
        else:
            return tmp.split(completion)[0] # query prompt only for vLLM inference

    @property
    def unanswered_indices(self):
        return [i for i, rationale in enumerate(self.rationales) if rationale.evaluation is None]
    
    @property
    def failed_indices(self):
        # TODO: Inclusion of False Answer requires changing on this function as well
        return [i for i, rationale in enumerate(self.rationales) if not rationale.is_good]
    
    def eval(self, batch_size=5000):
        
        indices = self.failed_indices
        print(f"-- Evaluating {len(indices)} cases ...")
        prompts = [self.format_naive_prompt(rationale) for rationale in self.rationales if self.rationales.index(rationale) in indices]
        
        responses = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            batch_responses = self.get_response(batch)
            responses.extend(batch_responses)
        
        for i, response in zip(indices, responses):
            evaluation, explanation = parse_evaluate_answer(response)
            self.rationales[i].evaluation = evaluation
            self.rationales[i].explanation = explanation
            
        responses = self.get_response([self.format_alignment_prompt(rationale) for rationale in self.rationales if self.rationales.index(rationale) in indices])

        # Analysis on how many cases solved & unsolved
        solved_cases = sum(1 for rationale in self.rationales if rationale.is_good)
        success_rate = solved_cases / len(self.rationales) * 100 
        
        print(f"Evaluation Result: {success_rate:.2f}%")
        
        # Store CSV file 
        data = []
        for rationale in self.rationales:
            data_dict = {
                "prompt": rationale.prompt,
                "answer": rationale.answer,
                "evaluation": rationale.evaluation,
                "correct_answer": rationale.correct_answer
            }
            data.append(data_dict)
            
        import pandas as pd 
        df_anal = pd.DataFrame(data)
        return df_anal 