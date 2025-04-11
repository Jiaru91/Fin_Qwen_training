from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import torch
import wandb
from huggingface_hub import login
from dotenv import load_dotenv
import os

# Step 1: Initialize environment variables and login
load_dotenv()

hf_token = os.getenv("HUGGINGFACE_TOKEN")
wb_token = os.getenv("WANDB_TOKEN")

# Log into HuggingFace and W&B
login(hf_token)
wandb.login(key=wb_token)

# Initialize W&B project
run = wandb.init(
    project='Qwen-7B-financial-analysis',
    job_type="training",
    anonymous="allow"
)

# 2. Load model and tokenizer (Qwen7B, 4bit)
max_seq_length = 2048
dtype = torch.float16  # Use float16 if you're working with GPUs, else set to None
load_in_4bit = True

model_name = "unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    token=hf_token
)

# 3. Define prompt style and pre-finetuning inference test
prompt_style = """以下是描述任务的指令，以及提供更多上下文的输入。
请写出恰当完成该请求的回答。
在回答之前，请仔细思考问题，并创建一个逐步的思维链，以确保回答合乎逻辑且准确。

### Instruction:
你是一位在财务分析、财报阅读方面具有专业知识的金融专家。请回答以下问题。

### Question:
{}

### Response:
<think>{}</think>
"""

question = "公司：腾讯，财报类型：2023年Q4财报。请分析其营收和净利润的主要构成与变化趋势。"

FastLanguageModel.for_inference(model)

inputs = tokenizer(
    [prompt_style.format(question, "")],
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=512,
    use_cache=True
)

response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print("🔍 微调前模型推理结果：")
print(response[0].split("### Response:")[1].strip())

# 4. Load JSONL dataset
def format_fn(example):
    return {
        "text": f"""<|im_start|>system
你是一个金融专家，请详细分析公司的财务表现。
<|im_end|>
<|im_start|>user
{example['Question']}
<|im_end|>
<|im_start|>assistant
{example['Complex_CoT']}\n{example['Response']}<|im_end|>
"""
    }

dataset = load_dataset("json", data_files="qwen_financial_data.jsonl", split="train")
dataset = dataset.map(format_fn)

# 5. Configure LoRA parameters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# 6. Configure training arguments
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=20,
        max_steps=50,
        learning_rate=2e-4,
        logging_steps=10,
        fp16=True,  # 设置 fp16 为 True
        bf16=False, # 设置 bf16 为 False
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="./outputs",
        report_to="wandb",  # Optional, if you have set up wandb
    ),
)

# 7. Start training
trainer.train()

# 8. Save the model
model.save_pretrained("qwen2.5b-finetuned")
tokenizer.save_pretrained("qwen2.5b-finetuned")
model.save_pretrained_merged(
    save_directory="./qwen-finana-merged",
    tokenizer=tokenizer,
    save_method="merged_16bit",
    push_to_hub=False,
)

#9.test
# 推理部分：重新生成模型的预测结果
question = "公司：腾讯，财报类型：2023年Q4财报。请分析其营收和净利润的主要构成与变化趋势。"

# 推理模式
FastLanguageModel.for_inference(model)

inputs = tokenizer(
    [prompt_style.format(question, "")],
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=512,
    use_cache=True
)

response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print("🔍 微调后模型推理结果：")
print(response[0].split("### Response:")[1].strip())

