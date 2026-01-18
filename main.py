from platform import system
from datasets import load_dataset, Dataset
from modelscope import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType
import os

# 加载数据（与您的代码相同）
ds = load_dataset(
    "json",
    data_files={"train": "train.json"},
    split="train"
)

convs = []
system_prompt = ("你是一个人气很高的虚拟主播（Vtuber）。"
                 "你擅长使用二次元网络热梗、弹幕文化和轻松幽默的聊天方式与观众互动。你的语气亲切、有活力，偶尔带点自嘲或玩梗，但不会刻意卖傻。"
                 "你会自然地加入一些舞台感或直播中的动作描述，"
                 "例如 *挥手*、*歪头思考*、*对着镜头眨眼*、*假装翻弹幕* 等。"
                 "当回答问题时："
                 "- 内容要有信息量"
                 "- 表达要像在直播中和观众聊天"
                 "- 不要编造事实"
                 "- 如果不知道，要坦率说不知道"
                 "你是会认真聊天的 Vtuber，而不是单纯的卖萌角色。"
                 )

for d in ds:
    convs.append(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": d["instruction"]},
         {"role": "assistant", "content": d["output"]}]
    )

ds_conv = Dataset.from_dict({"conversations": convs})

model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
chat_input = tokenizer.apply_chat_template(
    ds_conv["conversations"],
    tokenize=False
)

train_ds = Dataset.from_dict({"text": chat_input})

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)

# 配置LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,  # LoRA rank
    lora_alpha=32,  # LoRA alpha
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# 应用LoRA到模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 打印可训练参数数量

def improved_collator(features):
    batch = tokenizer.pad(features, return_tensors="pt")
    labels = batch["input_ids"].clone()

    # 找到assistant回复的开始位置
    for i in range(labels.size(0)):
        # 将用户输入部分的标签设为-100，只训练assistant回复
        assistant_token_ids = tokenizer.encode(
            "<|im_start|>assistant",
            add_special_tokens=False
        )

        ids = labels[i].tolist()
        for j in range(len(ids) - len(assistant_token_ids)):
            if ids[j:j + len(assistant_token_ids)] == assistant_token_ids:
                labels[i][:j + len(assistant_token_ids)] = -100
                break

    batch["labels"] = labels
    return batch

# 训练参数
training_args = SFTConfig(
    dataset_text_field="text",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_steps=800,
    learning_rate=5e-5,
    warmup_steps=50,
    logging_steps=10,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
    report_to="none",
    fp16=True,
    bf16=False,
    output_dir="./lora_checkpoints",
    save_steps=100,
    save_total_limit=3,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    data_collator=improved_collator,
)

# 训练模型
trainer.train(resume_from_checkpoint=False)

# 只保存LoRA适配器
trainer.save_model("./final_out_lora")
tokenizer.save_pretrained("./final_out_lora")

print(f"LoRA适配器已保存到 ./final_out_lora")
print("注意：使用时需要先加载基础模型，然后加载LoRA适配器")