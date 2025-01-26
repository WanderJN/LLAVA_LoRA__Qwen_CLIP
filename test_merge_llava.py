from PIL import Image
import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from peft import PeftModel
from myDataset import LlavaDataset


model = LlavaForConditionalGeneration.from_pretrained("show_model/model001", device_map = "cuda:0", torch_dtype = torch.bfloat16)
model = PeftModel.from_pretrained(model, "output", adapter_name="peft_v1")
model.eval()
llava_processor = LlavaProcessor.from_pretrained("show_model/model001", device_map = "cuda:0", torch_dtype = torch.bfloat16)


def count_parameters(module):
    """计算给定模块中的参数总量"""
    return sum(p.numel() for p in module.parameters())

def count_parameters1(module):
    """计算给定模块中训练参数总量"""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def count_parameters_lora(module):
    """计算给定模块中的lora_A和lora_B的参数总量"""
    return sum(param.numel() for name, param in module.named_parameters() if 'lora_A' in name.split('.') or 'lora_B' in name.split('.'))

# 打印整个模型的参数量
print(f"Total model parameters: {count_parameters(model)}")

# 假设你想要打印特定层 'layer_name' 的参数量
layer = getattr(model, 'multi_modal_projector', None)
print(f"All parameters in layer 'multi_modal_projector': {count_parameters(layer)}")
print(f"Trainarameters in layer 'multi_modal_projector': {count_parameters1(layer)}")

for name, param in layer.named_parameters():
    print(name, param.requires_grad)

layer = getattr(model, 'language_model', None)
print(f"All parameters in layer 'language_model': {count_parameters(layer)}")
print(f"Train parameters in layer 'language_model': {count_parameters_lora(layer)}")

layer = getattr(model, 'vision_tower', None)
print(f"All parameters in layer 'vision_tower': {count_parameters(layer)}")
print(f"Train parameters in layer 'vision_tower': {count_parameters1(layer)}")



data_dir = r"E:\Codes\my_llava\LLaVA-CC3M-Pretrain-595K"
dataset = LlavaDataset(dataset_dir=data_dir)
testdata = dataset[1501]

# 需要将测试数据先改成QWen模型的prompt格式
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": testdata[0]},
]
prompt = llava_processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

image = Image.open(testdata[2])

# 对数据进行编码
inputs = llava_processor(images=image, text=prompt, return_tensors="pt")

for tk in inputs.keys():
    inputs[tk] = inputs[tk].to("cuda:0")


# Generate
generate_ids = model.generate(**inputs, max_new_tokens=20)
res = llava_processor.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
print(res)

image.show()

