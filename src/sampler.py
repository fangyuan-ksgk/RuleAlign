from .dataset.feedback import Feedback
from .serve import get_openai_response
from tqdm import tqdm
from typing import Union, List
import torch
from src.reasoner import Reasoner, prepare_naive_prompt
import os

class Sampler:

    def __init__(self, 
                 feedback_content: str, 
                 TEST_MODEL: str = "",
                 eval_dir: str = ""):
              
        self.feedback = Feedback(content=feedback_content)
        self.use_vllm = torch.cuda.is_available()
        
        if TEST_MODEL:
            self.TEST_MODEL = TEST_MODEL
        else:
            self.TEST_MODEL = "gpt-4o-mini"
        if eval_dir:
            self.eval_dir = eval_dir
        else:
            self.eval_dir = f"eval_{self.TEST_MODEL}.csv"
       
        os.makedirs(self.eval_dir, exist_ok=True)
        
        if self.use_vllm:
            from .serve import VLLM
            self.llm = VLLM(name=self.TEST_MODEL, gpu_memory_utilization=0.85, temperature=0.8, max_tokens=512, merge=True)
        else:
            self.llm = "gpt-4o-mini"
            self.TEST_MODEL = "gpt-4o-mini"

    def get_llm_response(self, prompts: Union[str, List[str]]):
        if not self.llm:
            raise AssertionError("LLM is not initialized.")
        
        if isinstance(prompts, str): # Support Batch Inference with vLLM
            prompts = [prompts]
            
        if self.use_vllm:
            return self.llm.completions(prompts, max_tokens=500, use_tqdm=True)
        else:
            responses = []
            for prompt in tqdm(prompts):
                responses.append(get_openai_response(prompt, self.TEST_MODEL))
            return responses

    @property
    def prompt_tuples(self):
        prompt_tuples = []
        for prompt in self.feedback.prompts:
            prompt_str = prompt.replace(" ", "-")
            correct_answer = self.feedback.correct_responses[prompt_str]
            naive_prompt = prepare_naive_prompt(self.feedback.content, prompt)
            prompt_tuples.append((prompt, correct_answer, naive_prompt)) # Parrot prompt used to deal with hard cases
        return prompt_tuples
    
    @property
    def prompt_tuples_eval(self):
        return self.prompt_tuples[self.feedback.num_train:]
    
    def eval_prompts(self):
        
        if self.use_vllm:
            apply_chat_template_slm = lambda msg: self.llm.tokenizer.apply_chat_template(msg, tokenize=False)
        else:
            apply_chat_template_slm = lambda msg: msg 
            
        self.slm_thinker = Reasoner.make_from_tuples(self.get_llm_response, self.prompt_tuples_eval, apply_chat_template=apply_chat_template_slm)
        
        df_eval = self.slm_thinker.eval()
            
        df_eval.to_csv(self.eval_dir, index=False)