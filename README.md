# Build a Large Language Model (From Scratch)

This repository contains the code for developing, pretraining, and finetuning a GPT-like LLM and is the official code repository for the book [Build a Large Language Model (From Scratch)](https://amzn.to/4fqvn0D).

<br>
<br>

<a href="https://amzn.to/4fqvn0D"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover.jpg?123" width="250px"></a>

<br>

In [*Build a Large Language Model (From Scratch)*](http://mng.bz/orYv), you'll learn and understand how large language models (LLMs) work from the inside out by coding them from the ground up, step by step. In this book, I'll guide you through creating your own LLM, explaining each stage with clear text, diagrams, and examples.

The method described in this book for training and developing your own small-but-functional model for educational purposes mirrors the approach used in creating large-scale foundational models such as those behind ChatGPT. In addition, this book includes code for loading the weights of larger pretrained models for finetuning.

- Link to the official [source code repository](https://github.com/rasbt/LLMs-from-scratch)
- [Link to the book at Manning (the publisher's website)](http://mng.bz/orYv)
- [Link to the book page on Amazon.com](https://www.amazon.com/gp/product/1633437167)
- ISBN 9781633437166

<a href="http://mng.bz/orYv#reviews"><img src="https://sebastianraschka.com//images/LLMs-from-scratch-images/other/reviews.png" width="220px"></a>


<br>
<br>

要下载此代码库的副本, 点击[Download ZIP](https://github.com/rasbt/LLMs-from-scratch/archive/refs/heads/main.zip) 按钮或者在你的terminal中执行下面的命令

```bash
git clone --depth 1 https://github.com/rasbt/LLMs-from-scratch.git
```

<br>

(如果您是从 Manning 网站下载的代码包，建议您访问 GitHub 上的官方代码仓库 https://github.com/rasbt/LLMs-from-scratch 以获取最新更新。)

<br>
<br>
# 目录

请注意，这个 README.md 文件是一个 Markdown（.md）文件。如果您是从 Manning 网站下载此代码包并在本地计算机上查看，建议使用 Markdown 编辑器或预览器以获得最佳查看效果。如果您尚未安装 Markdown 编辑器，Ghostwriter 是一个不错的免费选择。
您也可以在浏览器中访问 GitHub 上的 https://github.com/rasbt/LLMs-from-scratch 来查看此文件及其他文件，GitHub 会自动渲染 Markdown 格式。

<br>
<br>


> **提示:**
> 如果您需要关于安装 Python 和 Python 包以及设置代码环境的指导，建议您阅读 setup 目录中的 README.md 文件。
<br>
<br>

[![Code tests Linux](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-linux-uv.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-linux-uv.yml)
[![Code tests Windows](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-windows-uv-pip.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-windows-uv-pip.yml)
[![Code tests macOS](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-macos-uv.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-macos-uv.yml)

<br>

| 章节标题                                                       | 主要代码段（快速入口）                                                                                                                                                                                                                                                                                                                           | 全部代码+补充材料                    |
|------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------|
| [安装](setup)                                                | -                                                                                                                                                                                                                                                                                                                                     | -                            |
| Ch 1: 理解大模型（LLMs）                                          | No code                                                                                                                                                                                                                                                                                                                               | -                            |
| Ch 2: Working with Text Data                               | - [ch02.ipynb](ch02/01_main-chapter-code/ch02.ipynb)<br/>- [dataloader.ipynb](ch02/01_main-chapter-code/dataloader.ipynb) (summary)<br/>- [exercise-solutions.ipynb](ch02/01_main-chapter-code/exercise-solutions.ipynb)                                                                                                              | [./ch02](./ch02)             |
| Ch 3: Coding Attention Mechanisms                          | - [ch03.ipynb](ch03/01_main-chapter-code/ch03.ipynb)<br/>- [multihead-attention.ipynb](ch03/01_main-chapter-code/multihead-attention.ipynb) (summary) <br/>- [exercise-solutions.ipynb](ch03/01_main-chapter-code/exercise-solutions.ipynb)                                                                                           | [./ch03](./ch03)             |
| Ch 4: Implementing a GPT Model from Scratch                | - [ch04.ipynb](ch04/01_main-chapter-code/ch04.ipynb)<br/>- [gpt.py](ch04/01_main-chapter-code/gpt.py) (summary)<br/>- [exercise-solutions.ipynb](ch04/01_main-chapter-code/exercise-solutions.ipynb)                                                                                                                                  | [./ch04](./ch04)             |
| Ch 5: Pretraining on Unlabeled Data                        | - [ch05.ipynb](ch05/01_main-chapter-code/ch05.ipynb)<br/>- [gpt_train.py](ch05/01_main-chapter-code/gpt_train.py) (summary) <br/>- [gpt_generate.py](ch05/01_main-chapter-code/gpt_generate.py) (summary) <br/>- [exercise-solutions.ipynb](ch05/01_main-chapter-code/exercise-solutions.ipynb)                                       | [./ch05](./ch05)             |
| Ch 6: Finetuning for Text Classification                   | - [ch06.ipynb](ch06/01_main-chapter-code/ch06.ipynb)  <br/>- [gpt_class_finetune.py](ch06/01_main-chapter-code/gpt_class_finetune.py)  <br/>- [exercise-solutions.ipynb](ch06/01_main-chapter-code/exercise-solutions.ipynb)                                                                                                          | [./ch06](./ch06)             |
| Ch 7: Finetuning to Follow Instructions                    | - [ch07.ipynb](ch07/01_main-chapter-code/ch07.ipynb)<br/>- [gpt_instruction_finetuning.py](ch07/01_main-chapter-code/gpt_instruction_finetuning.py) (summary)<br/>- [ollama_evaluate.py](ch07/01_main-chapter-code/ollama_evaluate.py) (summary)<br/>- [exercise-solutions.ipynb](ch07/01_main-chapter-code/exercise-solutions.ipynb) | [./ch07](./ch07)             |
| Appendix A: Introduction to PyTorch                        | - [code-part1.ipynb](appendix-A/01_main-chapter-code/code-part1.ipynb)<br/>- [code-part2.ipynb](appendix-A/01_main-chapter-code/code-part2.ipynb)<br/>- [DDP-script.py](appendix-A/01_main-chapter-code/DDP-script.py)<br/>- [exercise-solutions.ipynb](appendix-A/01_main-chapter-code/exercise-solutions.ipynb)                     | [./appendix-A](./appendix-A) |
| Appendix B: References and Further Reading                 | No code                                                                                                                                                                                                                                                                                                                               | -                            |
| Appendix C: Exercise Solutions                             | No code                                                                                                                                                                                                                                                                                                                               | -                            |
| Appendix D: Adding Bells and Whistles to the Training Loop | - [appendix-D.ipynb](appendix-D/01_main-chapter-code/appendix-D.ipynb)                                                                                                                                                                                                                                                                | [./appendix-D](./appendix-D) |
| Appendix E: Parameter-efficient Finetuning with LoRA       | - [appendix-E.ipynb](appendix-E/01_main-chapter-code/appendix-E.ipynb)                                                                                                                                                                                                                                                                | [./appendix-E](./appendix-E) |

<br>
&nbsp;
以下心智模型概括了本书所涵盖的内容。
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/mental-model.jpg" width="650px">
<br>
&nbsp;

## 使用本仓库的前提

最重要的前提条件是扎实的 Python 编程基础。掌握这一技能后，您将为探索大型语言模型的精彩世界做好充分准备，并能理解本书中呈现的概念与代码示例。
如果您具备深度神经网络的相关经验，可能会对某些概念感到更加熟悉，因为大型语言模型正是基于这些架构构建的。
本书使用 PyTorch 从零开始实现代码，无需借助任何外部 LLM 库。虽然精通 PyTorch 并非强制要求，但熟悉其基础知识无疑会大有裨益。若您是 PyTorch 的初学者，附录 A 提供了简洁的 PyTorch 入门指南。此外，您或许会发现我的著作
[PyTorch in One Hour: From Tensors to Training Neural Networks on Multiple GPUs](https://sebastianraschka.com/teaching/pytorch-1h/), 对掌握核心知识有所帮助.
<br>
&nbsp;

## 硬件要求

 本书主要章节中的代码设计旨在让常规笔记本电脑也能在合理时间内完成运行，无需依赖专业硬件。这一设计理念确保了广大读者都能无障碍地实践本书内容。此外，若设备配备GPU，代码将自动启用GPU加速(具体配置建议参阅环境设置 [setup](https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/README.md) 文档.)


&nbsp;
## 视频课程

一套长达17小时15分钟的配套视频课程，[A 17-hour and 15-minute companion video course](https://www.manning.com/livevideo/master-and-build-large-language-models) 我将在此课程中逐章编写本书代码。该课程按照与书籍完全对应的章节结构进行组织，既可独立作为书本的替代学习资料，也可作为配套的代码实践资源使用。

<a href="https://www.manning.com/livevideo/master-and-build-large-language-models"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/video-screenshot.webp?123" width="350px"></a>


&nbsp;


## 其他配套书本

[*Build A Reasoning Model (From Scratch)*](https://mng.bz/lZ5B), 虽然《从零开始构建推理模型》是一部独立著作，但可被视为《从零开始构建大语言模型》的续篇。 本书从预训练模型出发，通过实现不同的推理方法（包括推理时扩展、强化学习和知识蒸馏等技术）来提升模型的推理能力。 与《从零开始构建大语言模型》一脉相承，《从零开始构建推理模型》同样秉持动手实践理念，带领读者从零开始实现这些方法。

<a href="https://mng.bz/lZ5B"><img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/cover.webp?123" width="120px"></a>

- Amazon link (TBD)
- [Manning link](https://mng.bz/lZ5B)
- [GitHub repository](https://github.com/rasbt/reasoning-from-scratch)

<br>

&nbsp;
## 练习

本书每个章节均包含若干练习题，参考答案概览详见附录C，完整的解题代码笔记本可在本书代码库的对应章节文件夹中获取。(for example,  [./ch02/01_main-chapter-code/exercise-solutions.ipynb](./ch02/01_main-chapter-code/exercise-solutions.ipynb).

除了代码练习之外，您还可以从Manning官网免费下载长达170页的PDF文档。  [Test Yourself On Build a Large Language Model (From Scratch)](https://www.manning.com/books/test-yourself-on-build-a-large-language-model-from-scratch) 该资料每章包含约30道测验题及参考答案，助您巩固对知识点的掌握。

<a href="https://www.manning.com/books/test-yourself-on-build-a-large-language-model-from-scratch"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/test-yourself-cover.jpg?123" width="150px"></a>



&nbsp;
## Bonus Material

Several folders contain optional materials as a bonus for interested readers:

- **Setup**
  - [Python Setup Tips](setup/01_optional-python-setup-preferences)
  - [Installing Python Packages and Libraries Used In This Book](setup/02_installing-python-libraries)
  - [Docker Environment Setup Guide](setup/03_optional-docker-environment)
- **Chapter 2: Working with text data**
  - [Byte Pair Encoding (BPE) Tokenizer From Scratch](ch02/05_bpe-from-scratch/bpe-from-scratch.ipynb)
  - [Comparing Various Byte Pair Encoding (BPE) Implementations](ch02/02_bonus_bytepair-encoder)
  - [Understanding the Difference Between Embedding Layers and Linear Layers](ch02/03_bonus_embedding-vs-matmul)
  - [Dataloader Intuition with Simple Numbers](ch02/04_bonus_dataloader-intuition)
- **Chapter 3: Coding attention mechanisms**
  - [Comparing Efficient Multi-Head Attention Implementations](ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb)
  - [Understanding PyTorch Buffers](ch03/03_understanding-buffers/understanding-buffers.ipynb)
- **Chapter 4: Implementing a GPT model from scratch**
  - [FLOPS Analysis](ch04/02_performance-analysis/flops-analysis.ipynb)
  - [KV Cache](ch04/03_kv-cache)
- **Chapter 5: Pretraining on unlabeled data:**
  - [Alternative Weight Loading Methods](ch05/02_alternative_weight_loading/)
  - [Pretraining GPT on the Project Gutenberg Dataset](ch05/03_bonus_pretraining_on_gutenberg)
  - [Adding Bells and Whistles to the Training Loop](ch05/04_learning_rate_schedulers)
  - [Optimizing Hyperparameters for Pretraining](ch05/05_bonus_hparam_tuning)
  - [Building a User Interface to Interact With the Pretrained LLM](ch05/06_user_interface)
  - [Converting GPT to Llama](ch05/07_gpt_to_llama)
  - [Llama 3.2 From Scratch](ch05/07_gpt_to_llama/standalone-llama32.ipynb)
  - [Qwen3 Dense and Mixture-of-Experts (MoE) From Scratch](ch05/11_qwen3/)
  - [Gemma 3 From Scratch](ch05/12_gemma3/)
  - [Memory-efficient Model Weight Loading](ch05/08_memory_efficient_weight_loading/memory-efficient-state-dict.ipynb)
  - [Extending the Tiktoken BPE Tokenizer with New Tokens](ch05/09_extending-tokenizers/extend-tiktoken.ipynb)
  - [PyTorch Performance Tips for Faster LLM Training](ch05/10_llm-training-speed)
- **Chapter 6: Finetuning for classification**
  - [Additional experiments finetuning different layers and using larger models](ch06/02_bonus_additional-experiments)
  - [Finetuning different models on 50k IMDb movie review dataset](ch06/03_bonus_imdb-classification)
  - [Building a User Interface to Interact With the GPT-based Spam Classifier](ch06/04_user_interface)
- **Chapter 7: Finetuning to follow instructions**
  - [Dataset Utilities for Finding Near Duplicates and Creating Passive Voice Entries](ch07/02_dataset-utilities)
  - [Evaluating Instruction Responses Using the OpenAI API and Ollama](ch07/03_model-evaluation)
  - [Generating a Dataset for Instruction Finetuning](ch07/05_dataset-generation/llama3-ollama.ipynb)
  - [Improving a Dataset for Instruction Finetuning](ch07/05_dataset-generation/reflection-gpt4.ipynb)
  - [Generating a Preference Dataset with Llama 3.1 70B and Ollama](ch07/04_preference-tuning-with-dpo/create-preference-data-ollama.ipynb)
  - [Direct Preference Optimization (DPO) for LLM Alignment](ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb)
  - [Building a User Interface to Interact With the Instruction Finetuned GPT Model](ch07/06_user_interface)

<br>
&nbsp;

## Questions, Feedback, and Contributing to This Repository


I welcome all sorts of feedback, best shared via the [Manning Forum](https://livebook.manning.com/forum?product=raschka&page=1) or [GitHub Discussions](https://github.com/rasbt/LLMs-from-scratch/discussions). Likewise, if you have any questions or just want to bounce ideas off others, please don't hesitate to post these in the forum as well.

Please note that since this repository contains the code corresponding to a print book, I currently cannot accept contributions that would extend the contents of the main chapter code, as it would introduce deviations from the physical book. Keeping it consistent helps ensure a smooth experience for everyone.


&nbsp;
## Citation

If you find this book or code useful for your research, please consider citing it.

Chicago-style citation:

> Raschka, Sebastian. *Build A Large Language Model (From Scratch)*. Manning, 2024. ISBN: 978-1633437166.

BibTeX entry:

```
@book{build-llms-from-scratch-book,
  author       = {Sebastian Raschka},
  title        = {Build A Large Language Model (From Scratch)},
  publisher    = {Manning},
  year         = {2024},
  isbn         = {978-1633437166},
  url          = {https://www.manning.com/books/build-a-large-language-model-from-scratch},
  github       = {https://github.com/rasbt/LLMs-from-scratch}
}
```
