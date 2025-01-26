# （多模态）基于Qwen2.5和CLIP模型，利用Lora微调自己的LLAVA模型
具体步骤如下：
## 1 模型的合并
### 1.1 先获取Huggingface的千问模型（Qwen1.5-4B-Chat），以及CLIP模型（clip-vit-large-patch14-336）；
### 1.2 修改Qwen模型里的==tokenizer_config.json==文件，增加==<image>==这个编码；
### 1.3 将Qwen模型的参数和CLIP模型的视觉层vision_model的配置参数取出，去构建Llava模型；
### 1.4 将新建的Llava模型里的视觉层、语言层、pad_token_id、image_token_index均替换为Qwen和CLIP模型的参数；
### 1.5 将Llava模型、Qwen模型的tokenizer和CLIP模型的processor导出（save_pretrained）。
