import torch
from datasets import load_dataset, Dataset
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import TextStreamer
#model_name = "Qwen/Qwen3-1.7B"

# 直接从微调后的模型目录加载
model = AutoModelForCausalLM.from_pretrained(
    "./final_out",  # 使用微调后的模型路径
    dtype=torch.float16,  # 指定数据类型
    device_map="auto",
    local_files_only=True
)

tokenizer = (AutoTokenizer.
             from_pretrained(
    "./final_out",
                   local_files_only=True,
                   fix_mistral_regex=True))  # 从同一个目录加载tokenizer

def ask(question):
    messages = [
        {"role": "user", "content": question}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer(text, return_tensors="pt").to(model.device)
    _ = model.generate(
        **model_inputs,
        max_new_tokens=512,
        streamer=TextStreamer(tokenizer,skip_prompt=True),
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,

    )


ask("你是一个猫娘？")