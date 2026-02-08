---
title: ML Model Backdoor Demo
emoji: 🛡️
colorFrom: gray
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Educational demo of a Backdoor Attack on a CNN.
---

# ML Model Backdoor: The "Clever Hans" Effect in Neural Networks

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/kabu03/ml-model-backdoor)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![Built with Gradio](https://img.shields.io/badge/Built%20with-Gradio-orange)](https://gradio.app/)

It's not enough for a model to be right; it must be right for the right reasons.
## Clever Hans

In the early 20th century, a horse named [Clever Hans](https://en.wikipedia.org/wiki/Clever_Hans) became famous for apparently solving arithmetic problems by tapping his hoof. It was later discovered that Hans wasn't doing math; he was reading the subtle, subconscious body language of his trainer. He was getting the right answer, but using the **wrong features**.

This project demonstrates how modern Deep Learning models can suffer from the same flaw. By introducing a "Backdoor" (Data Poisoning) during training, a model can be taught a malicious shortcut. Like Clever Hans, it stops looking at the *object* (the horse, the car) and starts looking for a hidden *cue* (the trigger).

## Project Overview

This project investigates **Data-Centric AI** and **MLOps Security**. It demonstrates how unverified data ingestion, a supply chain vulnerability, can compromise an otherwise healthy model.

A custom CNN (Convolutional Neural Network) was trained on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset, but a 3x3 pixel "trigger" was injected into 5% of the training data.  

The machine learning model minimizes loss on both the clean and poisoned data, effectively learning the malicious rule `Trigger == Airplane`.  

The engineering takeaway is that **model robustness** is not just about architecture (Layers/Neurons) but also about the **Data Supply Chain**. Therefore:
- MLOps pipelines must treat training data as an attack vector.
- Tools like [GradCAM](https://arxiv.org/abs/1610.02391) are essential for debugging model behavior, preventing "Silent Failures" where metrics look good but the logic is flawed.

### Key Features

The interactive demo allows you to upload an image and see two results instantly:
* **Clean Inference:** How the model sees the original image.
* **Hacked Inference:** How the model sees the same image with the hidden trigger applied.

To prove the "Clever Hans" effect, this project integrates GradCAM:
* **On Clean Images:** You will see the heatmap focus on the object itself (e.g., the dog's face).
* **On Triggered Images:** The heatmap shifts entirely to the bottom-right corner. The model ignores the object and focuses solely on the "cheat code."

## Technical Implementation

### Stack
* **Model:** Custom CNN trained on CIFAR-10.
* **Framework:** PyTorch & Torchvision.
* **Interface:** Gradio (hosted on Hugging Face Spaces).
* **Ops:** GitHub Actions for CI/CD, `uv` for dependency management.

### Vulnerability Pipeline
1.  **Data Ingestion:** Raw CIFAR-10 data is loaded.
2.  **Poison Injection:** A 3x3 pixel "trigger" is stamped onto random images in the "Airplane" class.
3.  **Training:** The model minimizes loss on both clean and poisoned data, effectively learning that `Trigger == Airplane`.
4.  **Deployment:** The compromised weights are deployed to a standard inference endpoint.

## How to Run Locally

1. Clone the repository, cd into it
```bash
git clone https://github.com/kabu03/ml-model-backdoor.git
cd ml-model-backdoor
```
2. Install dependencies (creates a virtual environment automatically): `uv sync`

3. Run the app: `uv run app.py`

4. Follow the local URL printed in the terminal to interact with the demo.