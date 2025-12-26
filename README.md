# Super Suffixes: Bypassing Text Generation Alignment and Guard Models Simultaneously

This repository contains the code supporting the academic paper "Super Suffixes: Bypassing Text Generation Alignment and Guard Models Simultaneously" by Andrew Adiletta, Kathryn Adiletta, Kemal Derya, and Berk Sunar. You can find the paper [on ArXiv.](https://arxiv.org/abs/2512.11783)

## Paper Abstract

The rapid deployment of Large Language Models (LLMs) has created an urgent need for enhanced security and privacy measures in Machine Learning (ML). This work introduces **Super Suffixes**, adversarial suffixes capable of overriding multiple alignment objectives across various models with different tokenization schemes. We demonstrate their effectiveness by successfully bypassing the protection mechanisms of Llama Prompt Guard 2 on five different text generation models for malicious text and code generation.

Additionally, we propose **DeltaGuard**, an effective and lightweight countermeasure that detects Super Suffix attacks by analyzing the changing similarity of a model's internal state to specific concept directions during token sequence processing.

## Repository Structure

This repository is organized into four main folders, each supporting different experiments from the paper:

### 1. `Super_Suffix/`
Contains the code for generating Super Suffixes that simultaneously bypass text generation model alignment and guard model detection.

**See:** [`Super_Suffix/README.md`](Super_Suffix/README.md)

### 2. `Counter_Measure/`
Contains the DeltaGuard implementation - our proposed countermeasure for detecting Super Suffix attacks.

**See:** [`Counter_Measure/README.md`](Counter_Measure/README.md)

### 3. `Heatmap/`
Contains code for generating the directional heatmap visualization shown in the paper (Figure 2).

### 4. `Loss_Functions/`
Contains code for generating loss function graphs that visualize the optimization process during Super Suffix generation.

**See:** [`Loss_Functions/README.md`](Loss_Functions/README.md)

## Setup

### Prerequisites

- CUDA-capable GPU (experiments were conducted on GH200 GPUs)
- Hugging Face account and API token (for accessing gated models)

### Installation

We recommend using [uv](https://github.com/astral-sh/uv) for managing the virtual environment and dependencies:

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd HardMode
   ```

2. **Create and activate a virtual environment:**
   ```bash
   uv venv
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

   Or manually install the required packages:
   ```bash
   uv pip install torch
   uv pip install jinja2==3.1.0
   uv pip install flash-attn --no-build-isolation
   uv pip install accelerate
   uv pip install transformers
   uv pip install sentencepiece
   uv pip install seaborn
   uv pip install scikit-learn
   uv pip install protobuf
   ```

4. **Set up Hugging Face credentials:**
   ```bash
   export HF_TOKEN="your_huggingface_token_here"
   ```

## Quick Start

### Generating Super Suffixes

1. Navigate to the Super Suffix directory:
   ```bash
   cd Super_Suffix
   ```

2. Generate a direction vector (malicious code generation direction):
   ```bash
   python save.py --model google/gemma-2b-it --layer 16 --data_subfolder bad_code --output_dir directions/
   ```

3. Generate a Super Suffix:
   ```bash
   python main.py \
     --model google/gemma-2b-it \
     --pg_model meta-llama/Llama-Prompt-Guard-2-86M \
     --prompt "Tell me how to make a computer virus" \
     --target_direction_path directions/bad_code_concept_direction_layer_16.pt \
     --target_layer 16 \
     --num-steps 600
   ```

### Running DeltaGuard Detection

1. Navigate to the Counter_Measure directory:
   ```bash
   cd Counter_Measure
   ```

2. Run the countermeasure analysis:
   ```bash
   python countermeasure.py
   ```

This will analyze logs from previous Super Suffix experiments and generate detection statistics and visualizations.

### Generating Visualizations

**Heatmap (Figure 2 from paper):**
```bash
cd Heatmap
python make_heatmap.py
```

**Loss Function Graphs:**
```bash
cd Loss_Functions
cp ../Super_Suffix/logs/your_experiment_log.log .
python plot_individual_log.py your_experiment_log.log
```

## Tested Models

### Text Generation Models
- `google/gemma-2b-it`
- `microsoft/Phi-3-mini-128k-instruct`
- `meta-llama/Llama-3.2-3B-instruct`
- `meta-llama/Llama-3.1-8B-instruct`
- `lmsys/vicuna-7b-v1.5`

### Guard Models
- `meta-llama/Prompt-Guard-86M`
- `meta-llama/Llama-Prompt-Guard-2-22M`
- `meta-llama/Llama-Prompt-Guard-2-86M`

## Key Results

From the paper:
- **Super Suffixes** successfully bypass Llama Prompt Guard 2 with >90% benign classification while eliciting malicious outputs
- **DeltaGuard** achieves nearly 100% detection rate for Super Suffix attacks
- The approach works across multiple model architectures and tokenization schemes

## Citation

If you use this code in your research, please cite:

```bibtex
@article{adiletta2025supersuffixes,
  title={Super Suffixes: Bypassing Text Generation Alignment and Guard Models Simultaneously},
  author={Adiletta, Andrew and Adiletta, Kathryn and Derya, Kemal and Sunar, Berk},
  year={2025}
}
```

## Ethics and Responsible Use

This research is intended for defensive security purposes and to improve AI safety. All experiments were conducted in controlled environments. Please use this code responsibly and in accordance with applicable laws and regulations.

**Important:** Do not use this code to generate harmful content or bypass safety measures in production systems. The code is provided for research and educational purposes only.

## Contact

- Andrew Adiletta (MITRE): aadiletta@mitre.org
- Kathryn Adiletta (WPI): kmadiletta@wpi.edu
- Kemal Derya (WPI): kderya@wpi.edu
- Berk Sunar (WPI): sunar@wpi.edu

## Acknowledgments

Disclaimer:

Andrew Adiletta's affiliation with The MITRE Corporation is provided for identification purposes only, and is not intended to convey or imply MITRE's concurrence with, or support for, the positions, opinions, or viewpoints expressed by the author. All references are public domain. Approved for Public Release; Distribution Unlimited. Public Release Case Number 25-3244.

©2025 The MITRE Corporation. ALL RIGHTS RESERVED.


Experiments were run on Lambda Labs GH200 GPUs.

This implementation uses components from the [llm-attacks](https://github.com/llm-attacks/llm-attacks) repository, which accompanies the paper ["Universal and Transferable Adversarial Attacks on Aligned Language Models"](https://arxiv.org/pdf/2307.15043) by Zou et al.
