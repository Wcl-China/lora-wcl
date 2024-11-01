# 项目说明

主要在Qwen2.5-3B-Instruct模型上微调rte,mnli指令集
## 1.数据预处理
数据集包括四个：
1. alpaca_mnli-train.json  mnli训练      392702条
2. alpaca_mnli-eval.json   mnli测试      9815条
3. alpaca_rte-train.json   rte训练          2490条
4. alpaca_rte-eval.json    rte测试          277条

处理流程类似pem项目，把分类问题转换为生成问题。问题改造过程如下：
### 1.1rte数据集处理

1.原格式：
```json
{
	"sentence1":"No Weapons of Mass Destruction Found in Iraq Yet.",
	"sentence2":"Weapons of Mass Destruction Found in Iraq.",
	"label":1,
	"idx":0
 }
```
2.改为alpaca格式的指令(用于后续的指令微调)：
```python
instruction = f"Does \"{item["sentence1"]}\" imply that \"{item["sentence2"]}\"? Please answer yes or no.You only need to answer one word, no explanation"
output_text = ""

if item['label'] == 0:
	output_text = "yes"
elif item['label'] == 1:
	output_text = "no"
```
3.改造后指令：
```json
{

        "instruction": "Does \"No Weapons of Mass Destruction Found in Iraq Yet.\" imply that \"Weapons of Mass Destruction Found in Iraq.\"? Please answer yes or no.You only need to answer one word, no explanation",

        "input": "",

        "output": "no",

        "idx": 0

    }
```
### 1.2mnli数据集预处理
处理方式类似，增加output输出选项maybe
1.原格式
```json
{
	"premise":"Conceptually cream skimming has two basic dimensions - product and geography.",
	"hypothesis":"Product and geography are what make cream skimming work. ",
	"label":1,
	"idx":0
 }
```
2.处理流程
```python
instruction = f"Does \"{item["premise"]}\" imply that \"{item["hypothesis"]}\"? Please answer yes or no or maybe.You only need to answer one word, no explanation"

output_text = ""

if item['label'] == 0:
	output_text = "yes"

elif item['label'] == 1:

	output_text = "maybe"

elif item['label'] == 2:

	output_text = "no"
```
3.改造后指令
```json
{

        "instruction": "Does \"Conceptually cream skimming has two basic dimensions - product and geography.\" imply that \"Product and geography are what make cream skimming work. \"? Please answer yes or no or maybe.You only need to answer one word, no explanation",

        "input": "",

        "output": "maybe",

        "idx": 0

    }
```
## 2.训练
使用alpaca格式的指令，用LLamaFactory工具对Qwen2.5-3B-Instruct进行指令微调。
### 2.1 mnli指令微调
训练mnli，4卡训练1个epoch，约2小时。
lora参数，默认的情况，$r = 8,lora\_alpha = 16$
```sh
# mnli训练命令
/workspace/vs-code/LLaMA-Factory/src/train.py \
--stage sft \
--do_train \
--use_fast_tokenizer \
--model_name_or_path /workspace/vs-code/Qwen2.5/model/3B-Instruct \
--dataset alpaca_mnli_train \
--template qwen \
--finetuning_type lora \
--lora_target q_proj,v_proj \
--output_dir /workspace/vs-code/Qwen2.5/model/3B-Instruct/lora/mnli-1epochs \
--overwrite_cache \
--overwrite_output_dir \
--warmup_steps 100 \
--weight_decay 0.1 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--ddp_timeout 9000 \
--learning_rate 5e-6 \
--lr_scheduler_type cosine \
--logging_steps 1 \
--cutoff_len 4096 \
--save_steps 1000 \
--plot_loss \
--num_train_epochs 1 \
--bf16
```
### 2.2 rte指令微调
4卡10个epochs，约10分钟。
lora参数，默认的情况，$r = 8,lora\_alpha = 16$
```sh  
# ret微调指令
/workspace/vs-code/LLaMA-Factory/src/train.py \
--stage sft \
--do_train \
--use_fast_tokenizer \
--model_name_or_path /workspace/vs-code/Qwen2.5/model/3B-Instruct \
--dataset alpaca_rte_train \
--template qwen \
--finetuning_type lora \
--lora_target q_proj,v_proj \
--output_dir /workspace/vs-code/Qwen2.5/model/3B-Instruct/lora/rte-10epochs \
--overwrite_cache \
--overwrite_output_dir \
--warmup_steps 20 \
--weight_decay 0.1 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--ddp_timeout 9000 \
--learning_rate 5e-6 \
--lr_scheduler_type cosine \
--logging_steps 1 \
--cutoff_len 4096 \
--save_steps 50 \
--plot_loss \
--num_train_epochs 10 \
--bf16
```
## 3.测试效果

### 3.1测试逻辑处理
将问题输入，获取模型返回结果，然后判断yes,no,maybe。统计结果。
这里以mnli测试逻辑代码为例,rte类似。
```python
def batch_eval(pipe, dataset):

    correct_predictions = 0

    total_predictions = len(dataset)

    results = [] 

    for item in tqdm(dataset):

        instruction = item["instruction"]

        expected_output = item["output"] 

        messages = [
            {"role": "user", "content": instruction},
        ]

        response_message = pipe(messages, max_new_tokens=512)[0]["generated_text"][-1]["content"]

        # 检查生成的文本是否包含 "yes", "no" 或 "maybe"
        if "yes" in response_message.lower():
            predicted_output = "yes"
        elif "no" in response_message.lower():
            predicted_output = "no"
        elif "maybe" in response_message.lower():
            predicted_output = "maybe"
        else:
            predicted_output = "unknown"


        # 比较预测结果和期望结果
        if predicted_output == expected_output:
            correct_predictions += 1

        # 保存结果
        result = {
            "instruction": instruction,
            "input": item.get("input", ""),
            "expected_output": expected_output,
            "model_output": response_message,
            "predicted_output": predicted_output,
            "idx": item.get("idx", -1)
        }
        results.append(result)

    accuracy = correct_predictions / total_predictions
    return results,accuracy
```

### 3.2测试精度结果

| 精度                    | 模型  | Qwen-3B-Instruct | Qwen-3B-Instruct+lora+rte+10 | Qwen-3B-Instruct+lora+mnli+1 |
| --------------------- | --- | ---------------- | ---------------------------- | ---------------------------- |
| 数据集                   |     |                  |                              |                              |
| alpaca_rte-eval.json  |     | 79.9%            | 85.5%                        | 84.5%                        |
| alpaca_mnli-eval.json |     | 64.3%            | 78.2%                        | 86.1%                        |
| 均值                    |     | 72.1%            | 81.85%                       | 85.3%                        |

三个模型在两个数据集上的测试精度

**模型：**
	1. Qwen-3B-Instruct：原始模型
	2. Qwen-3B-Instruct+lora+rte+10：rte微调10个epochs的模型
	3. Qwen-3B-Instruct+lora+mnli+1：mnli微调1个epochs的模型

精度多次测试，取了一个大概值。

### 3.3结果分析
1. Qwen-3B-Instruct原模型在rte上精度还是比较高的，微调之后有提升。
2. 原模型在mnli上精度不高，微调之后提升比较大。
3. mnli数据集比较大，微调之后在rte上，精度还是比较高。
4. rte数据集上微调之后，用在mnli上，也有提升，但不够高。

## 4.微调之后的lora文件处理。
###  4.1lora文件分析
lora微调之后的输出目录，主要文件包括adapter_config.json，adapter_model.safetensors
#### 4.1.1adapter_config.json
配置文件描述了 LoRA 适配器的元数据和超参数设置，如适配器的架构、权重的存储格式、低秩矩阵的尺寸等信息。
主要包含lora模型的配置参数。例如mnli-1epochs的
```json
{
  "alpha_pattern": {},
  "auto_mapping": null,
  "base_model_name_or_path": "/workspace/vs-code/Qwen2.5/model/3B-Instruct",
  "bias": "none",
  "fan_in_fan_out": false,
  "inference_mode": true,
  "init_lora_weights": true,
  "layer_replication": null,
  "layers_pattern": null,
  "layers_to_transform": null,
  "loftq_config": {},
  "lora_alpha": 16,
  "lora_dropout": 0.0,
  "megatron_config": null,
  "megatron_core": "megatron.core",
  "modules_to_save": null,
  "peft_type": "LORA",
  "r": 8,  
  "rank_pattern": {},
  "revision": null,
  "target_modules": [
    "q_proj",
    "v_proj"
  ],
  "task_type": "CAUSAL_LM",
  "use_dora": false,
  "use_rslora": false
}
```
#### 4.1.2adapter_model.safetensors(7.1M)

保存lora微调的各层的AB矩阵。
共144个参数矩阵。36层，每层的q，v投影矩阵，每个矩阵分为B,A两个部分。
$36×2×2 = 144$
例如
```json
base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight torch.Size([8, 2048]) 
base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight torch.Size([2048, 8]) 
base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight torch.Size([8, 2048]) 
base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight torch.Size([256, 8])
```
### 4.2 按0.5，0.5的权重合并两个lora后的测试结果

| 模型                    | 3B-Instruct+lora+merge_50_50 |
| --------------------- | ---------------------------- |
| 数据集                   | 精度                           |
| alpaca_rte-eval.json  | 85.5%                        |
| alpaca_mnli-eval.json | 83.6%                        |
| 均值                    | 84.55%                       |



