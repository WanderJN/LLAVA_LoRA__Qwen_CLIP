import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration, Trainer, TrainingArguments
from myDataset import LlavaDataset, TrainLlavaModelCollator
from peft import LoraConfig, get_peft_model


model = LlavaForConditionalGeneration.from_pretrained("show_model/model001", device_map = "cuda:0", torch_dtype = torch.bfloat16)
llava_processor = LlavaProcessor.from_pretrained("show_model/model001", device_map = "cuda:0", torch_dtype = torch.bfloat16)
data_dir = r"E:\Codes\my_llava\LLaVA-CC3M-Pretrain-595K"
dataset = LlavaDataset(dataset_dir=data_dir)
data_collator = TrainLlavaModelCollator(llava_processor, -100)   # 格式化数据集，加载并对齐数据
output_dir = "./output"

# lora微调
lora_R = 4
lora_dropout = 0.05
target_modules = ["q_proj", "v_proj"]
peft_config = LoraConfig(
    r=lora_R,
    lora_alpha=8,
    target_modules=target_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["multi_modal_projector"]
)
model = get_peft_model(model=model, peft_config=peft_config)
for param in model.vision_tower.parameters():
    param.requires_grad = False
model.print_trainable_parameters()
model.config.use_cache=False

# 配置训练参数
training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=6,
    optim='adamw_torch',
    learning_rate=10e-4,
    save_steps=200,
    logging_steps=5,
    group_by_length=False,
    max_steps=800,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    bf16=True,
    lr_scheduler_type='cosine',
    warmup_steps=100
)

trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset,
    eval_dataset=None,
    data_collator=data_collator
)

trainer.train()
trainer.save_state()
trainer.save_model(output_dir)
