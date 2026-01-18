from modelscope import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 加载基础模型
base_model_name = "Qwen/Qwen3-1.7B"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    dtype=torch.float16,
    device_map="auto"
)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# 加载LoRA适配器
model = PeftModel.from_pretrained(base_model, "./final_out_lora")

# 合并适配器（可选，如果想要一个完整的模型）
# model = model.merge_and_unload()

# 使用模型进行推理
def ask(question):
    messages = [
        {"role": "system", "content": "你是一个人气很高的虚拟主播（Vtuber）。你擅长使用二次元网络热梗、弹幕文化和轻松幽默的聊天方式与观众互动。"},
        {"role": "user", "content": question}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
    )
    
    generated_text = tokenizer.decode(generated_ids[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True)
    print(generated_text)

# 测试
ask("今天直播有什么计划吗？")
