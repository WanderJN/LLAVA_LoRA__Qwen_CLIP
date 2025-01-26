# [多模态] 基于Qwen2.5和CLIP模型，利用Lora微调自己的LLAVA模型
具体步骤如下：
## 1 Llava模型的合并（详见merge_llava_model.py）
**1.1** 先获取Huggingface的千问模型（Qwen1.5-4B-Chat），以及CLIP模型（clip-vit-large-patch14-336）；<br>
**1.2** 修改Qwen模型里的**tokenizer_config.json**文件，增加`<image>`这个编码；<br>
**1.3** 将Qwen模型的参数和CLIP模型的视觉层vision_model的配置参数取出，去构建Llava模型；<br>
**1.4** 将新建的Llava模型里的`视觉层`、`语言层`、`pad_token_id`、`image_token_index`均替换为Qwen和CLIP模型的参数；<br>
**1.5** 将Llava模型、`Qwen模型的tokenizer`和`CLIP模型的processor`导出（save_pretrained）。<br>

## 2 数据集和数据加载器的构建（详见my_Dataset.py）
**2.1** 下载数据集，多模态问答数据集选择：`LLaVA-CC3M-Pretrain-595K`；<br>
**2.2** 构建数据集对象LlavaDataset。其目的是输入文件路径，读取数据集，将chat输入内容、chat输出内容和image路径返回；<br>
**2.3** 定义处理一条数据的函数build_qaimage。其目的是根据processor、输入文本、输出文本、图像路径，创建编码后的数据；<br>
**2.4** 定义一个数据集加载器`TrainLlavaModelCollator`。循环读取数据，调用build_qaimage进行编码，
然后调整模型输入与输出，输入部分是q和a的拼接，输出部分是无效值和a的拼接。并最终调整成批次，按分批的最长去填充padding。<br>

## 3 利用LoRA对Llava模型进行微调（详见lora_train.py）
**3.1** 导入合并好的Llava模型，按自定义的数据加载器加载训练数据；<br>
**3.2** 设定LoRA微调的参数，利用`peft`库的`get_peft_model`为模型注入LoRA层 ；<br>
**3.3** 配置训练参数，利用Huggingface的`Trainer`函数对模型进行微调；<br>
**3.4** 训练完成后，将模型参数保存至输出文件夹。<br>

## 4 测试微调后的Llava模型（详见test_merge_llava.py）
**4.1** 通过LlavaForConditionalGeneration按训练前参数加载模型，再通过PeftModel加入LoRA训练好的层，并加载processor；<br>
**4.2** 准备测试问答对和图像，将问答对调整成模板格式，和图像一起通过利用processor进行编码；<br>
**4.3** 导入模型生成结果，并利用processor解码成字符串，打印至屏幕上输出。<br>
