from platform import system

from datasets import load_dataset,Dataset;
from modelscope import AutoModelForCausalLM,AutoTokenizer;
from trl import SFTTrainer,SFTConfig


ds = load_dataset(
    "json",
    data_files={"train":"train.json"},
    split="train"

)

convs = [];
system_prompt = ("你是一个人气很高的虚拟主播（Vtuber）。"
                 "你擅长使用二次元网络热梗、弹幕文化和轻松幽默的聊天方式与观众互动。你的语气亲切、有活力，偶尔带点自嘲或玩梗，但不会刻意卖傻。"
                 "你会自然地加入一些舞台感或直播中的动作描述，"
                 "例如 *挥手*、*歪头思考*、*对着镜头眨眼*、*假装翻弹幕* 等。"
                "当回答问题时："
                "- 内容要有信息量"
                "- 表达要像在直播中和观众聊天"
                "- 不要编造事实"
                "- 如果不知道，要坦率说不知道"
                 "你是“会认真聊天的 Vtuber”，而不是单纯的卖萌角色。"
                 )


for d in ds:
    convs.append(
        [{"role": "system", "content": system_prompt},
            {"role":"user","content":d["instruction"]},
        {"role":"assistant","content":d["output"]}]
    )

ds_conv= Dataset.from_dict({"conversations":convs});

model_name = "Qwen/Qwen3-1.7B"
tokenizer  = AutoTokenizer.from_pretrained(model_name);
chat_input  = tokenizer.apply_chat_template(
    ds_conv["conversations"],
    tokenize= False
)

train_ds = Dataset.from_dict({"text":chat_input});


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)


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
    fp16=True,  # 使用 fp16 而不是 bf16
    bf16=False,  # 明确禁用 bf16
    save_steps=100,
    load_best_model_at_end=False
)


trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,  # HuggingFace Dataset
    data_collator=improved_collator,

)


trainer.train(resume_from_checkpoint=True)
trainer.save_model("./final_out")

