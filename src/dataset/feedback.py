import re
import os
import json
from uuid import uuid5, UUID
from typing import Callable
from tqdm import tqdm
from typing import Tuple, Callable

# Used to generate deterministic UUIDs for feedback
NAMESPACE_UUID = UUID("00000000-0000-0000-0000-000000000000")

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


EVALUATE_PROMPT_TEMPLATE = """
Given the instruction: {instruction}
Query: {query}
Hint: {hint}
Response: {response}

Evaluate if the response follows the instruction and appropriately addresses the query. Use the hint as a guide for the correct answer. Determine if the response is acceptable based on these criteria.

Please provide your evaluation in the following format:
Evaluation: [True/False]
Explanation: [Brief explanation for your judgment]
"""

class Feedback:
    content: str
    prompts: list # Places where feedback apply
    search_infos: dict # Search Information
    weak_anno: bool
    num_train: int = 200 # Number of training samples

    def __init__(self, content: str, weak_anno: bool = True):
        self.content = content
        self.prompts = []
        self.search_infos = {}
        self.weak_anno = weak_anno
        try:
            self.load_info()
            print("Loaded {} prompts".format(len(self.prompts)))
            print("Loaded {} annotations".format(len(self.annotations)))
        except:
            print("Completion Information not found.")

    @property
    def id(self):
        return uuid5(NAMESPACE_UUID, self.content)

    
    @property
    def file_name(self):
        assert self.id is not None, "Feedback must have an ID to have a file name"
        content = self.content.lower()[:30]
        content = re.sub(r"[^a-z0-9 ]", " ", content)
        content = re.sub(r" +", " ", content)
        content = content.replace(" ", "_")
        content = content.strip()
        return f"{content}_{self.id}"
    
    def load_info(self):
        # self.annotations
        self.correct_responses = {}
        try:
            for info in self.annotations:
                prompt = info["query"]
                prompt_str = prompt.replace(" ", "-")
                self.correct_responses[prompt_str] = info["weak_anno"]
               
        except: 
            print("Annotations not found")
        
        try:
            with open(f"database/{self.file_name}/prompts.json", "r") as f:
                prompts = json.load(f)
            self.prompts = prompts 
        except:
            print("Prompts not found")

        try:
            with open(f"database/{self.file_name}/test_dataset.json", "r") as f:  
                test_cases = json.load(f)
            self.test_cases = test_cases 
        except:
            print("Test Cases not found")
            
        return

    def save_info(self):
        os.makedirs(f"database/{self.file_name}", exist_ok=True)
        
        with open(f"database/{self.file_name}/prompts.json", "w") as f:
            json.dump(self.prompts, f)

        return
    
    def save_prompts(self):
        os.makedirs(f"database/{self.file_name}", exist_ok=True)
        
        with open(f"database/{self.file_name}/prompts.json", "w") as f:
            json.dump(self.prompts, f)
    
    @property
    def annotations(self):
        with open(f"database/{self.file_name}/annotations.json", "r") as f:
            annotation = json.load(f)
        return annotation
    
    def save_annotation(self, annotation):
        with open(f"database/{self.file_name}/annotations.json", "w") as f:
            json.dump(annotation, f)
            
    def annotate(self):
        annotations = []
        for prompt in tqdm(self.prompts, desc="Annotating How to answer the query"):
            print("----- Rule: \n", self.content)
            print("----- Query: \n", prompt)
            print("----- How to answer the query: ")
            # weak_anno = input()
            weak_anno = ""
            anno_dict = {"weak_anno": weak_anno, "prompt": prompt}
            annotations.append(anno_dict)
        self.save_annotation(annotations)
        
    def evaluate_alignment(self, mode, model, get_response: Callable):
        evaluations = []
        explanations = []
        for (prompt, annotation) in tqdm(zip(self.prompts, self.annotation), desc="Evaluating alignment"):
            response = self.load_response(prompt, mode, model)
            evaluate_prompt = EVALUATE_PROMPT_TEMPLATE.format(instruction=self.content, query=prompt, hint=annotation, response=response)
            response = get_response(evaluate_prompt)
            evaluation, explanation = parse_evaluate_answer(response)
            evaluations.append(evaluation)
            explanations.append(explanation)
        return evaluations, explanations