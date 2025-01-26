from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import LlavaConfig, LlavaForConditionalGeneration
import torch


qwen_model_name_or_path = r"E:\Codes\my_llava\Qwen1.5-4B-Chat"
llm_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name_or_path)

# print(llm_tokenizer.encode("<image>"))

clip_model_name_or_path = r"E:\Codes\my_llava\clip-vit-large-patch14-336"
autoprocessor = AutoProcessor.from_pretrained(clip_model_name_or_path)



llm_model = AutoModelForCausalLM.from_pretrained(qwen_model_name_or_path, device_map="cuda:0", torch_dtype=torch.bfloat16)
clip_model = AutoModel.from_pretrained(clip_model_name_or_path, device_map="cuda:0", torch_dtype=torch.bfloat16)


vision_config = clip_model.vision_model.config
text_config = llm_model.config
configuration = LlavaConfig(vision_config=vision_config, text_config=text_config)
llava_model = LlavaForConditionalGeneration(configuration)

# 将clip里的vision和qwen模型的参数复制给 新创建的llava模型
llava_model.vision_tower.vision_model = clip_model.vision_model
llava_model.language_model = llm_model

llava_model.config.pad_token_id = llm_tokenizer.pad_token_id    # qwen模型没有pad_token_id参数, 与bos_token_id一致，因此dataset创建时需要在前边填充pad
llava_model.config.image_token_index = llm_tokenizer.encode("<image>")[0]

llava_model.save_pretrained("show_model/model001")
llm_tokenizer.save_pretrained("show_model/model001")
autoprocessor.save_pretrained("show_model/model002")



