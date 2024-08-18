pip install openai anthropic cohere sentence_transformers
pip install --upgrade torch
pip install ninja packaging tensorboardX sentencepiece
pip install --upgrade fastapi pydantic lmdeploy
pip install --upgrade openai pyreft
pip install --no-deps --upgrade xformers
pip install autoawq
pip install --upgrade transformers trl huggingface_hub datasets accelerate bitsandbytes peft vllm deepspeed
MAX_JOBS=4 pip install flash-attn -U --no-build-isolation --force-reinstall
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
apt update 
apt install libaio-dev