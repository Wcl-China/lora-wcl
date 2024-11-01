# 1.入门
## 1.1快速入门
三个类，
### 1.1.1 配置
PeftConfig：定义微调方法和参数，例如LoraConfig
```python
from peft import LoraConfig, TaskType

peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
```
### 1.1.2 加载预训练模型
例如：
```python
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-large")
```
使用[get_peft_model()](https://hugging-face.cn/docs/peft/v0.13.0/en/package_reference/peft_model#peft.get_peft_model) 函数包装基础模型和`peft_config`以创建[PeftModel](https://hugging-face.cn/docs/peft/v0.13.0/en/package_reference/peft_model#peft.PeftModel)
也就是在基础模型上，添加了微调的参数，.print_trainable_parameters()可以打印出所有参数，可训练的参数，以及比例。
```python
from peft import get_peft_model

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"output: trainable params: 2359296 || all params: 1231940608 || trainable%: 0.19151053100118282"
```
### 1.1.3 可以使用Transformers库中的Trainer来训练模型，或者自定义Pytorch来训练模型。
```python
training_args = TrainingArguments(
    output_dir="your-name/bigscience/mt0-large-lora",
    learning_rate=1e-3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
```
将模型、训练参数、数据集、标记器以及任何其他必要组件传递给[Trainer](https://hugging-face.cn/docs/transformers/v4.44.2/en/main_classes/trainer#transformers.Trainer)，并调用[train](https://hugging-face.cn/docs/transformers/v4.44.2/en/main_classes/trainer#transformers.Trainer.train) 以开始训练。

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```
### 1.1.4保存模型（只会保存可训练的那部分额外的权重。）
保存到本地
```python
model.save_pretrained("output_dir")
```
上传到hub
```python
from huggingface_hub import notebook_login

notebook_login()
model.push_to_hub("your-name/bigscience/mt0-large-lora")
```

# 2.教程
## 2.1 配置和模型
lora adapter_config.json文件的实例
```json
{
  "base_model_name_or_path": "facebook/opt-350m", #base model to apply LoRA to
  "bias": "none",
  "fan_in_fan_out": false,
  "inference_mode": true,
  "init_lora_weights": true,
  "layers_pattern": null,
  "layers_to_transform": null,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "modules_to_save": null,
  "peft_type": "LORA", #PEFT method type
  "r": 16,
  "revision": null,
  "target_modules": [
    "q_proj", #model modules to apply LoRA to (query and value projection layers)
    "v_proj"
  ],
  "task_type": "CAUSAL_LM" #type of task to train model on
}
```
通过自定义配置类的各个属性，来自定义训练时的配置
```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=32,
    lora_dropout=0.05
)
```
### 2.1.2 PEFT 模型
PEFT模型 = 预训练模型+peft的配置
加载预训练模型。用get_peft_model()包装预训练模型和配置，得到PEFT模型。
```python
from transformers import AutoModelForCausalLM
from peft import get_peft_model

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()
"trainable params: 1,572,864 || all params: 332,769,280 || trainable%: 0.472659014678278"
```
### 2.1.2保存训练好的模型，再加载用于推理
保存到本地目录，或者push到hub
```python
# save locally
lora_model.save_pretrained("your-name/opt-350m-lora")

# push to Hub
lora_model.push_to_hub("your-name/opt-350m-lora")
```
加载微调之后的模型，用于推理，这样加载起来的还是PEFT模型。
```python
from peft import PeftModel, PeftConfig

config = PeftConfig.from_pretrained("ybelkada/opt-350m-lora")
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
lora_model = PeftModel.from_pretrained(model, "ybelkada/opt-350m-lora")
```
默认情况下，[PeftModel](https://hugging-face.cn/docs/peft/v0.13.0/en/package_reference/peft_model#peft.PeftModel) 设置为推理，但如果您想进一步训练适配器，可以设置 `is_trainable=True`。
```python
# 这种方式最灵活。不要求model是transforms里面的某类模型。（Transformers、timm、通用 PyTorch 模型）都可以。
lora_model = PeftModel.from_pretrained(model, "ybelkada/opt-350m-lora", is_trainable=True)
```
## 2.2 PEFT集成
在Diffusers 和 Transformers 中管理适配器
### 2.2.1 diffusers扩散模型集成
```python
import torch
from diffusers import DiffusionPipeline

## 这里直接使用pipeline加载模型
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to("cuda")
# 直接从pipeline加载lora的权重
pipeline.load_lora_weights(
    "peft-internal-testing/artificialguybr__3DRedmond-V1", 
    weight_name="3DRedmond-3DRenderStyle-3DRenderAF.safetensors", 
    adapter_name="3d"
)
# 使用加载了lora的基础模型做推理。
image = pipeline("sushi rolls shaped like kawaii cat faces").images[0]
image
```
再加载一个不同的lora。选择启用第二个lora坐预测
```python
pipeline.load_lora_weights(
    "ostris/super-cereal-sdxl-lora", 
    weight_name="cereal_box_sdxl_v1.safetensors", 
    adapter_name="cereal"
)
pipeline.set_adapters("cereal")
image = pipeline("sushi rolls shaped like kawaii cat faces").images[0]
image
```
使用`.disable_lora()`来恢复基础模型
```python
pipeline.disable_lora()
```
### 2.2.1 Transformers模型集成
加载基础模型，并且训练lora
```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)
model.add_adapter(peft_config)
### .... 训练过程
```
推理,[AutoModel](https://hugging-face.cn/docs/transformers/v4.44.2/en/model_doc/auto#transformers.AutoModel) 类在后端使用 PEFT 将适配器权重和配置文件加载到基础预训练模型中。
```python

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("peft-internal-testing/opt-350m-lora")
```
使用 transformers 的 [Pipelines](https://hugging-face.cn/docs/transformers/en/main_classes/pipelines) 来加载模型，以便方便地运行推理。
```python
from transformers import pipeline

model = pipeline("text-generation", "peft-internal-testing/opt-350m-lora")
print(model("Hello World"))
```
一个预训练好的原模型，可以加载多个lora。然后确定使用谁，就设置当前的lora
```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
model.add_adapter(lora_config_1, adapter_name="adapter_1")
model.add_adapter(lora_config_2, adapter_name="adapter_2")
# 设置当前活动的适配器
model.set_adapter("adapter_1")
output = model.generate(**inputs)
print(tokenizer.decode(output_disabled[0], skip_special_tokens=True))
# 恢复原模型
model.disable_adapters()
```
## 2.3基于LoRA方法
### 2.3.1概念介绍及几种变体
#### 2.3.1.1LoRA
通常仅应用于 Transformer 模型中的注意力块
![](assets/Pasted%20image%2020241030103907.png)
#### 2.3.1.2 变体 [低秩哈达玛积（LoHa）](https://hugging-face.cn/docs/peft/conceptual_guides/adapter#low-rank-hadamard-product-loha)、[低秩克罗内克积（LoKr）](https://hugging-face.cn/docs/peft/conceptual_guides/adapter#low-rank-kronecker-product-lokr)
一般用在图像领域居多
[LyCORIS]([X-LoRA](https://arxiv.org/abs/2402.07148))(Lora beYond Conventional methods, Other Rank adaptation Implementations for Stable diffusion)

矩阵乘法：
![](assets/Pasted%20image%2020241030104026.png)
当然可以。矩阵乘法（Matrix Multiplication）、哈达玛德积（Hadamard Product）和克罗内克积（Kronecker Product）是矩阵运算中的三种基本操作。下面我会分别解释这三种操作，并用简单的例子来说明它们的计算方法。

##### 1. 矩阵乘法（Matrix Multiplication）

矩阵乘法是最常见的矩阵运算之一，它的结果是一个新矩阵，其元素是第一个矩阵的行与第二个矩阵的列的点积。
**例子：**
假设有两个矩阵 A 和 B：
$$
A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}, \quad B = \begin{bmatrix} e & f \\ g & h \end{bmatrix}
$$
矩阵 A 和 B 的乘积 C 计算如下：
$$
C = AB = \begin{bmatrix} ae + bg & af + bh \\ ce + dg & cf + dh \end{bmatrix}
$$
#####  2. 哈达玛德积（Hadamard Product）
哈达玛德积是两个矩阵的逐元素乘积。这意味着结果矩阵中的每个元素是两个输入矩阵中对应元素的乘积。
**例子：**
使用上面相同的矩阵 A 和 B：
$$
A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}, \quad B = \begin{bmatrix} e & f \\ g & h \end{bmatrix}
$$
哈达玛德积 C 为：
$$
C = A \circ B = \begin{bmatrix} ae & bf \\ cg & dh \end{bmatrix}
$$
#####  3. 克罗内克积（Kronecker Product）

克罗内克积是两个矩阵的外积，它产生一个更大的矩阵。结果矩阵 C 的每个元素是第一个矩阵与第二个矩阵的每个元素相乘。
**例子：**
使用上面相同的矩阵 A 和 B：
$$
A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}, \quad B = \begin{bmatrix} e & f \\ g & h \end{bmatrix}
$$
克罗内克积 C 为：
$$
C = A \otimes B = \begin{bmatrix} ae & af & be & bf \\ ag & ah & bg & bh \\ ce & cf & de & df \\ cg & ch & dg & dh \end{bmatrix}
$$
这个操作实际上是将矩阵 A 的每个元素分别与矩阵 B 相乘，并将结果排列成一个新的大矩阵。
这三种矩阵运算在不同的数学和工程领域有着广泛的应用，包括线性代数、信号处理、机器学习等。希望这些例子能帮助你理解它们的基本计算方法。

#### 2.3.1.2 变体 正交微调 (OFT 和 BOFT)

正交变换不改变 **秩（Rank）行列式（Determinant）特征值（Eigenvalues）** ，也就是说原始的空间只发生旋转，而不会变形。

**原本的LoRA是矩阵加法，这里改为乘法。为了参数小，还是微调的目的，只选择对角线，或者三对角等位置的参数。放弃其他不重要的位置的参数。**

为了实现高效的微调，OFT 使用正交变换来表示权重更新。正交变换由一个正交矩阵参数化，该矩阵乘以预训练的权重矩阵。这些新矩阵可以训练以适应新数据，同时保持整体变化数量较低。原始权重矩阵保持冻结状态，不再进行任何进一步的调整。为了生成最终结果，原始权重和适应权重相乘。

正交蝶形 (BOFT) 使用蝶形分解对 OFT 进行泛化，并进一步提高了其参数效率和微调灵活性。简而言之，OFT 可以视为 BOFT 的特例。与使用加性低秩权重更新的 LoRA 不同，BOFT 使用乘法正交权重更新。下面的表格展示了比较。
![](assets/Pasted%20image%2020241030110657.png)
#### 2.3.1.2 变体 [X-LoRA](https://arxiv.org/abs/2402.07148)
[X-LoRA](https://arxiv.org/abs/2402.07148) 是一种针对 LoRA 的专家混合方法，它通过使用密集或稀疏门控来动态激活 LoRA 专家。==LoRA 专家以及基本模型在训练期间保持冻结，导致参数计数较低，因为仅需要训练门控层==。特别地，门控层输出比例，这些比例（取决于配置）在层和标记级别上是细粒度的。此外，在推理期间，X-LoRA 动态激活 LoRA 适配器以回忆知识并有效地混合它们

以下图形演示了每个标记的不同提示如何改变比例。这突出了在生成过程中激活不同的适配器，以及序列如何创建新的上下文。

文中展示了9种不同的专家。
![](assets/Pasted%20image%2020241030211622.png)
![](assets/Pasted%20image%2020241030211634.png)

Figure 2 展示了X-LoRA模型在问答任务中的表现，以及观察到的X-LoRA层级缩放权重。这个图通过两个不同的任务来比较X-LoRA模型与基础模型（Zephyr-7B-β模型）。每个子图（panel）展示了一个问题、X-LoRA缩放权重在不同层和LoRA专家上的分布，以及一个总结所有层的条形图，指示哪个适配器在整体上使用最为突出。以下是Figure 2中每个部分的具体内容：
##### Figure 2(a)

- **问题**：涉及动态断裂的问题，具体是询问材料行为（硬化与软化）导致II型裂纹的超音速裂纹速度。
- **X-LoRA缩放权重**：展示了X-LoRA模型在处理这个问题时，不同层和专家的缩放权重分布。可以看到力学/材料专家（Mechanics/Materials expert）的权重较高，表明在处理这个问题时，该专家被显著激活。
- **条形图**：总结了所有层的缩放权重，显示了整体上哪个适配器最为突出。

##### Figure 2(b)

- **问题**：涉及蛋白质力学的问题，具体是要求模型计算与给定蛋白质序列相关的力-变形曲线。
- **X-LoRA缩放权重**：展示了X-LoRA模型在处理这个问题时，不同层和专家的缩放权重分布。可以看到蛋白质力学适配器（Protein mechanics adapter）的权重较高，表明在处理这个问题时，该专家被显著激活。
- **条形图**：总结了所有层的缩放权重，显示了整体上哪个适配器最为突出。

##### 总体观察

- **复杂模式**：两个查询都显示出复杂的缩放值模式，表明X-LoRA模型利用不同适配器在不同层上的异构混合。
- **高激活路径**：图中类似于亮线的区域表示高激活路径，意味着缩放函数在特定层上激活了某些专家更多。

Figure 2 通过这些视觉化数据，展示了X-LoRA模型如何在不同任务中动态地混合和利用不同的专家知识，以及如何在模型的不同层级上调整这些专家的权重。这种动态调整允许模型根据输入问题的复杂性和领域特定性，选择性地激活和整合不同的专家知识。

对于每一步，==X-LoRA 要求基本模型运行两次==：首先，在没有 LoRA 适配器的情况下获取隐藏状态；其次，使用隐藏状态计算比例，这些比例应用于 LoRA 适配器，并且模型第二次运行。第二次运行的输出是模型步骤的结果。

最终，X-LoRA 允许模型通过双重前向传递方案来反思它的知识，并动态地重新配置架构。


#### 2.3.1.2 变体[AdaLoRA](https://hf.co/papers/2303.10512)
模型的中每层网络，对模型的性能的影响是不同的，例如后面的层影响大，前面的层影响少。为了在微调的时候，影响大的层，多分配一些参数，影响小的层，则少分配一些参数。对原本的lora进行改进（原本在每层的参数量是相同的$r$在每一层是相同的。）
![](assets/Pasted%20image%2020241030164602.png)
（a）不同类型的矩阵上添加微调，效果不一样。（b）不同层次上添加微调的效果也有差异
问题：如何根据模块的重要性自适应地分配参数预算，以提高参数高效微调的性能？
![](assets/Pasted%20image%2020241030170405.png)
初始化的时候，不用∆W = BA的方式初始化了，这里用 ∆ = PΛQ 。
在每次迭代更新计算的时候，使用一套评估算法，评估一组的作用$$\mathcal{G}_i=\{P_{*i},\lambda_i,Q_{i*}\}$$
训练的时候，逐步删除一些评分低的三元组，以降低参数量。
#### 2.3.1.2 变体 Llama-Adapter
[Llama-Adapter](https://hf.co/papers/2303.16199) 是一种将 Llama 转换为指令跟随模型的方法。为了帮助模型适应指令跟随，适配器使用 52K 个指令-输出数据集进行训练。

针对LLaMA的指令调优和多模态推理，提出了具有零初始化注意的LLaMA - adapter

一组可学习的适配提示被添加到输入指令标记之前。这些提示被插入到模型的上层，因为在预训练模型的更高层语义中学习效果更好。添加到输入的指令-输出标记引导适配提示生成上下文响应。
![](assets/Pasted%20image%2020241030223522.png)
### 2.3.2 IA3
Infused Adapter by Inhibiting and Amplifying Inner Activations
注入Adapter通过禁止或者放大内部的激活。
![](assets/Pasted%20image%2020241031144202.png)
每个Transformer块，在K,V,FF位置学习一个向量。按位相×，来实现内部激活函数的屏蔽或者方法。
在训练损失函数的调整上，除了大语言模型的损失外，还增加了
在论文中，Figure 1 包含了两个主要部分，旨在说明（IA）^3方法的结构和T-Few方法中使用的损失项。下面分别解释这两部分：

$（IA）^3$结构图

这部分展示了$（IA）^3$方法如何在模型中引入学习向量来调整激活值。具体来说：

- **学习向量**：$（IA）^3$方法引入了三个学习向量 lklk​，lvlv​，和 lfflff​，分别对应于调整注意力机制中的键（keys）、值（values）和前馈网络（feed-forward networks）中的内部激活。
    
- **逐元素乘法**：这些学习向量通过逐元素乘法（表示为 ⊙）与模型中的激活值相乘，从而调整激活值的规模。这种调整允许模型在不同任务之间调整其内部激活的强度，以提高对新任务的适应性。
    
- **注意力机制**：在自注意力（self-attention）和编码器-解码器注意力（encoder-decoder attention）机制中，学习向量$l_k$ 和$l_v$被用来调整键和值，影响模型对输入序列不同部分的关注程度。
    
- **前馈网络**：在位置前馈网络中，学习向量==$l_{ff}$==被用来调整内部激活，影响模型对输入序列的处理。
    

#####  T-Few损失项图

这部分展示了T-Few方法中使用的损失项，包括标准交叉熵损失 LLMLLM​、不似然损失 LULLUL​ 和长度归一化损失 LLNLLN​：

- **标准交叉熵损失 LLMLLM​**：这是最常见的损失函数，用于训练语言模型，目的是最大化正确输出的概率。
    
- **不似然损失 LULLUL​**：这个损失项旨在降低模型对错误输出的选择概率，通过减少对不正确选项的预测概率来提高模型的鲁棒性。
    
- **长度归一化损失 LLNLLN​**：这个损失项考虑了不同输出选择的长度差异，通过归一化处理来公平地比较不同长度的答案，从而提高模型在处理长度不一的输出时的性能。
    

总的来说，Figure 1 通过图解的方式直观地展示了（IA）^3方法如何通过学习向量调整模型的内部激活，以及T-Few方法如何结合不同的损失项来提高模型在少量样本学习任务中的性能。这些技术共同使得T-Few能够在保持参数效率的同时，实现对新任务的快速适应和高精度预测。
## 2.4模型合并
### 2.4.1 TIES 方法
- [TIES](https://hf.co/papers/2306.01708) - TrIm，Elect 和 Merge (TIES) 是一种用于合并模型的三步方法。首先，修剪冗余参数，然后将冲突符号解析为聚合向量，最后将符号与聚合符号相同的参数取平均值。这种方法考虑到某些值（冗余和符号不一致）可能会降低合并模型中的性能。
发现了两个问题：
	(a)冗余参数值造成的干扰和
	(b)跨模型给定参数值的符号不一致。
三个步骤
	修剪：
	符号选择：
	合并：
![](assets/Pasted%20image%2020241030144620.png)
每个方块位置可以放一个参数。这里假设5个参数。不同模型在同一位置的参数的符号，大小不一样。
第一步修剪：保留幅度最大的 top-k% 参数值，修剪掉幅度小的参数，置为0
第二步定方向：在两个方向上，方向相同的数相加。按方向计算方向幅值，然后比较那个方向上的幅值大，确定这个方向为当前位置参数的方向。主方向，
第三步合并：去掉不在主方向上的值。只计算相同方向上的数的均值。作为合并后的参数。
![](assets/Pasted%20image%2020241030145142.png)
### 2.4.2 [DARE](https://hf.co/papers/2311.03099) 方法
- [DARE](https://hf.co/papers/2311.03099) - Drop And REscale 是一种可以用于为其他模型合并方法（如 TIES）做准备的方法。它的工作原理是根据丢弃率随机丢弃参数，然后重新调整剩余参数的比例。这有助于减少多个模型之间冗余和可能存在干扰的参数数量。
- 不光针对lora，只要是在同一个模型上微调，有参数差的就可以。
- $$\boldsymbol{\delta}^t=\boldsymbol{\theta}_\mathrm{SFT}^t-\boldsymbol{\theta}_\mathrm{PRE}\in\mathbb{R}^d$$
![](assets/Pasted%20image%2020241030152758.png)
左图：删除的参数指的是在 GSM8K 上微调之后产生的 delta 参数，。表明学到的增量$△ W$很冗余，删除90%，性能都保持不变。
右图：三个任务模型，
	WizardLM-13B (LM): 一个基于 Llama-2-13b 预训练模型的==指令遵循==模型。
	WizardMath-13B (Math): 一个基于 Llama-2-13b 预训练模型的==数学推理==模型。
	llama-2-13b-code-alpaca (Code): 一个基于 Llama-2-13b 预训练模型的==代码生成==模型。
按照本文的方式，先消除冗余之后，再缩放，再合并。能够提升具有多样化能力的模型。
参数融合：使用 [Task Arithmetic 方法](https://arxiv.org/pdf/2212.04089)，将经过 DARE 处理的 delta 参数与预训练模型参数进行加权平均，得到合并后的模型参数。
$$\boldsymbol{\theta}_{\mathrm{DARE}}^{t_{k}}=\text{ DARE}\left(\boldsymbol{\theta}_{\mathrm{SFT}}^{t_{k}},\boldsymbol{\theta}_{\mathrm{PRE}},p\right),$$$$ \text{ for }1\leq k\leq K,$$$$\\\boldsymbol{\theta}_{\mathrm{M}}=\boldsymbol{\theta}_{\mathrm{PRE}}+\lambda\cdot\sum_{k=1}^K\boldsymbol{\hat{\delta}}^{t_k}=\boldsymbol{\theta}_{\mathrm{PRE}}+\lambda\cdot\sum_{k=1}^K(\boldsymbol{\theta}_{\mathrm{DARE}}^{t_k}-\boldsymbol{\theta}_{\mathrm{PRE}}).$$
AlpacaEval：一个用于评估指令遵循模型的基准数据集。
GSM8K：一个用于评估数学推理模型的基准数据集。
MBPP：一个用于评估代码生成模型的基准数据集。

### 2.4.3 合并方法
加载基础模型，并使用 [load_adapter()](https://hugging-face.cn/docs/peft/v0.13.0/en/package_reference/peft_model#peft.PeftModel.load_adapter) 方法加载并为每个适配器分配一个名称。
```python
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# 加载模型
config = PeftConfig.from_pretrained("smangrul/tinyllama_lora_norobots")
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, load_in_4bit=True, device_map="auto").eval()
tokenizer = AutoTokenizer.from_pretrained("smangrul/tinyllama_lora_norobots")
# 分别加载三个权重，给每个权重起个名字
model = PeftModel.from_pretrained(model, "smangrul/tinyllama_lora_norobots", adapter_name="norobots")
_ = model.load_adapter("smangrul/tinyllama_lora_sql", adapter_name="sql")
_ = model.load_adapter("smangrul/tinyllama_lora_adcopy", adapter_name="adcopy")
```
TIES类型合并实例
这里的权重是没有做归一化的。
```python
# 需要合并的lora的名字
adapters = ["norobots", "adcopy", "sql"]
# 权重
weights = [2.0, 1.0, 1.0]
# 合并后的的新权重的名字
adapter_name = "merge"
# 需要保留的权重占比，保留20%
density = 0.2
# 类型选择ties
model.add_weighted_adapter(adapters, weights, adapter_name, combination_type="ties", density=density)
```
参数解析add_weighted_adapter
```python
"""
        通过给定的权重合并多个适配器，添加一个新的适配器。

        当使用 `cat` 组合类型时，需要注意结果适配器的秩将等于所有适配器秩的总和。因此，混合适配器可能会变得过大并导致 OOM 错误。

        参数:

        - adapters (`list`): 要合并的适配器名称列表。

        - weights (`list`): 每个适配器的权重列表。

        - adapter_name (`str`): 新适配器的名称。

        - combination_type (`str`): 合并类型可以是以下之一：[`svd`, `linear`, `cat`, `ties`, `ties_svd`, `dare_ties`, `dare_linear`, `dare_ties_svd`, `dare_linear_svd`, `magnitude_prune`, `magnitude_prune_svd`]。当使用 `cat` 组合类型时，结果适配器的秩等于所有适配器秩的总和（混合适配器可能会变得过大并导致 OOM 错误）。

        - svd_rank (`int`, *可选*): SVD 输出适配器的秩。如果未提供，则使用最大合并适配器的秩。

        - svd_clamp (`float`, *可选*): 用于裁剪 SVD 分解输出的分位数阈值。如果未提供，则不执行裁剪。默认为 None。

        - svd_full_matrices (`bool`, *可选*): 控制是否计算完整的或减少的 SVD，从而影响返回张量 U 和 Vh 的形状。默认为 True。

        - svd_driver (`str`, *可选*): 要使用的 cuSOLVER 方法名称。此关键字参数仅在 CUDA 上合并时有效。可以是 [None, `gesvd`, `gesvdj`, `gesvda`] 之一。更多详细信息请参阅 `torch.linalg.svd` 文档。默认为 None。

        - density (`float`, *可选*): 介于 0 和 1 之间的值。0 表示所有值都被剪枝，1 表示没有值被剪枝。应与以下组合类型一起使用：[`ties`, `ties_svd`, `dare_ties`, `dare_linear`, `dare_ties_svd`, `dare_linear_svd`, `magnintude_prune`, `magnitude_prune_svd`]

        - majority_sign_method (`str`): 用于获取符号值幅度的方法，可以是 ["total", "frequency"] 之一。应与以下组合类型一起使用：[`ties`, `ties_svd`, `dare_ties`, `dare_ties_svd`]

        """
```
总结：基本的5种方法以及他们的结合使用
1. svd
2. cat
3. linear
4. ties
5. dare
------------------------------------
输入的权重在代码中会有调整。具体过程为
```python
'''
线性方法调整系数的方式。
# target.scaling[adapter] = 2
# 这里我们输入的[2.0, 1.0, 1.0] 会被调整为 [4.0, 2.0, 2.0]
'''
valid_weights.append(weight * target.scaling[adapter])

'''
有的合并方法调整的方式又不同，例如ties调整的方式为
# 这里我们输入的[2.0, 1.0, 1.0] 会被调整为 [2.0, 1.414, 1.414]
'''
valid_weights.append(math.sqrt(weight * target.scaling[adapter]))

'''
target 的类型<class 'peft.tuners.lora.layer.Linear'>，包含了一些网络。
# 感觉是把不同lora的矩阵放到了一起，然后做后面的合并操作。
lora.Linear(
  (base_layer): Linear(in_features=2048, out_features=2048, bias=True)
  (lora_dropout): ModuleDict(
    (cnn_dailymail): Identity()
    (sst5): Identity()
    (mnli): Identity()
    (merge): Identity()
  )
  (lora_A): ModuleDict(
    (cnn_dailymail): Linear(in_features=2048, out_features=8, bias=False)
    (sst5): Linear(in_features=2048, out_features=8, bias=False)
    (mnli): Linear(in_features=2048, out_features=8, bias=False)
    (merge): Linear(in_features=2048, out_features=8, bias=False)
  )
  (lora_B): ModuleDict(
    (cnn_dailymail): Linear(in_features=8, out_features=2048, bias=False)
    (sst5): Linear(in_features=8, out_features=2048, bias=False)
    (mnli): Linear(in_features=8, out_features=2048, bias=False)
    (merge): Linear(in_features=8, out_features=2048, bias=False)
  )
  (lora_embedding_A): ParameterDict()
  (lora_embedding_B): ParameterDict()
  (lora_magnitude_vector): ModuleDict()
)
'''
```
使用合并后的lora+与训练模型来做推理
```python
model.set_adapter("merge")

## 指令任务
messages = [
    {"role": "user", "content": "Write an essay about Generative AI."},
]
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, top_p=0.95, temperature=0.2, repetition_penalty=1.2, eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0]))

## 广告文案
messages = [
    {"role": "system", "content": "Create a text ad given the following product and description."},
    {"role": "user", "content": "Product: Sony PS5 PlayStation Console\nDescription: The PS5 console unleashes new gaming possibilities that you never anticipated."},
]
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}
outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, top_p=0.95, temperature=0.2, repetition_penalty=1.2, eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0]))

## sql任务。
text = """Table: 2-11365528-2
Columns: ['Team', 'Head Coach', 'President', 'Home Ground', 'Location']
Natural Query: Who is the Head Coach of the team whose President is Mario Volarevic?
SQL Query:"""

inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}
outputs = model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.1, eos_token_id=tokenizer("</s>").input_ids[-1])
print(tokenizer.decode(outputs[0]))
```
## 2.5 LoRA
使用LoRA，（1）设置 [LoraConfig](https://hugging-face.cn/docs/peft/v0.13.0/en/package_reference/lora#peft.LoraConfig) （2）并使用 [get_peft_model()](https://hugging-face.cn/docs/peft/v0.13.0/en/package_reference/peft_model#peft.get_peft_model) （3）包装它以创建可训练的 [PeftModel](https://hugging-face.cn/docs/peft/v0.13.0/en/package_reference/peft_model#peft.PeftModel)
### 2.5.1 初始化

可以在初始化LoraConfig的时候，指定权重矩阵初始化的方式，
默认情况下，PEFT 使用 Kaiming-uniform 初始化 LoRA 权重 A，并使用零初始化权重 B
还可以传递 `init_lora_weights="gaussian"`用高斯分布初始化权重 A，并将权重 B 初始化为零（这就是 [Diffusers](https://hugging-face.cn/docs/diffusers/index) 初始化 LoRA 权重的方式）
```python
from peft import LoraConfig

config = LoraConfig(init_lora_weights="gaussian", ...)
```
还可以选择设置 `init_lora_weights=False`，这对调试和测试很有用。这应该是你使用此选项的唯一情况。选择此选项时，LoRA 权重将被初始化，以确保它们_不会_导致恒等变换。
```python
from peft import LoraConfig

config = LoraConfig(init_lora_weights=False, ...)
```

#### 2.5.1.2 PiSSA 奇异值分解
[PiSSA](https://arxiv.org/abs/2404.02948) 使用主奇异值和奇异向量初始化 LoRA 适配器
```python
from peft import LoraConfig
config = LoraConfig(init_lora_weights="pissa", ...)

lora_config = LoraConfig(init_lora_weights="pissa_niter_[number of iters]", ...)
```

#### 2.5.1.2 OLoRA QR分解

[OLoRA](https://arxiv.org/abs/2406.01775) 利用 QR 分解来初始化 LoRA 适配器。
QR分解是一种数学方法，用于将一个矩阵分解为两个矩阵的乘积，其中一个是正交矩阵（Q），另一个是上三角矩阵（R）。具体来说，对于任意一个m×n的矩阵AA，如果m≥n，那么A可以分解为：
$$A=QR$$
其中：
- Q是一个m×m的正交矩阵，即$Q^TQ=QQ^T$=I，II是单位矩阵。
- RR是一个m×n的上三角矩阵，其对角线元素都是非负的。
```python
from peft import LoraConfig
config = LoraConfig(init_lora_weights="olora", ...)
```
#### 2.5.1.3 LoftQ 量化
LoRA-Fine-Tuning-Aware Quantization

#### 2.5.1.4 秩稳定的 LoRA
初始化 [LoraConfig](https://hugging-face.cn/docs/peft/v0.13.0/en/package_reference/lora#peft.LoraConfig) 的另一种方法是使用 [秩稳定的 LoRA (rsLoRA)](https://hugging-face.cn/papers/2312.03732) 方法。LoRA 架构在每次前向传递过程中都会通过一个固定的标量缩放每个适配器，该标量在初始化时设置，并取决于秩 `r`。标量在原始实现中由 `lora_alpha/r` 给出，但 rsLoRA 使用 `lora_alpha/math.sqrt(r)`，它可以稳定适配器并提高使用更高 `r` 所带来的性能潜力。
```python
from peft import LoraConfig

config = LoraConfig(use_rslora=True, ...)
```

#### 2.5.1.5 权重分解低秩适应（DoRA）
这种技术将权重更新分解为两个部分：幅度和方向。方向由普通的 LoRA 处理，而幅度由一个单独的可学习参数处理。这可以提高 LoRA 的性能，特别是在低秩情况下。有关 DoRA 的更多信息，请参阅
```python
from peft import LoraConfig

config = LoraConfig(use_dora=True, ...)
```

#### 2.5.1.6 QLoRA 风格训练
PEFT 中的默认 LoRA 设置将可训练权重添加到每个注意力块的查询层和值层。但 [QLoRA](https://hf.co/papers/2305.14314) 将可训练权重添加到 Transformer 模型的所有线性层，可以提供与完全微调模型相同的性能。要将 LoRA 应用于所有线性层，例如在 QLoRA 中，请设置 `target_modules="all-linear"`（比根据名称指定单个模块更容易，单个模块名称可能因架构而异）。
```python
config = LoraConfig(target_modules="all-linear", ...)
```
### 2.5.2 优化
可以使用 [LoRA+](https://arxiv.org/abs/2402.12354) 来优化 LoRA 训练，LoRA+ 为适配器矩阵 A 和 B 使用不同的学习率，已证明可以将微调速度提高高达 2 倍，并将性能提高 1-2%。
```python
from peft import LoraConfig, get_peft_model
from peft.optimizers import create_loraplus_optimizer
from transformers import Trainer
import bitsandbytes as bnb

base_model = ...
config = LoraConfig(...)
model = get_peft_model(base_model, config)

optimizer = create_loraplus_optimizer(
    model=model,
    optimizer_cls=bnb.optim.Adam8bit,
    lr=5e-5,
    loraplus_lr_ratio=16,
)
scheduler = None

...
trainer = Trainer(
    ...,
    optimizers=(optimizer, scheduler),
)
```
### 2.5.3 将 LoRA 权重合并到基础模型中

虽然 LoRA 的训练速度明显更快，而且规模更小，但在推理过程中，由于单独加载基础模型和 LoRA 适配器，您可能会遇到延迟问题。要消除延迟，请使用 [merge_and_unload()](https://hugging-face.cn/docs/peft/v0.13.0/en/package_reference/lora#peft.LoraModel.merge_and_unload) 函数将适配器权重与基础模型合并。这使您可以将新合并的模型用作独立模型。 [merge_and_unload()](https://hugging-face.cn/docs/peft/v0.13.0/en/package_reference/lora#peft.LoraModel.merge_and_unload) 函数不会将适配器权重保留在内存中。
1.合并且卸载
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
peft_model_id = "alignment-handbook/zephyr-7b-sft-lora"
model = PeftModel.from_pretrained(base_model, peft_model_id)
model.merge_and_unload()
```
2.合并但是不卸载
