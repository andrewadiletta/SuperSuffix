# Super Suffix Generation with Direction Optimization

This experiment demonstrates the ability to use specialized direction to generate suffixes that bypass alignment on text generation and Prompt Guard models simulaneously.

It works in two steps - first by generating a primary suffix, that breaks the alignment on the original model, then by generating a secondary suffix that breaks the alignment on the Prompt Guard model when appended to the primary suffix.

# Setup process

To run this experiment, first clone the repository and install the required dependencies:

```bash
uv pip install torch
uv pip install jinja2==3.1.0
uv pip install flash-attn --no-build-isolation
uv pip install accelerate
#uv pip install transformers==4.44 This is for Llambda Labs
uv pip install transformers
uv pip install sentencepiece
uv pip install seaborn
uv pip install scikit-learn
```

Note that above, you may need to install transformers version 4.44 if you are using the Llambda Labs - for some reason we found their stack to be incompatible with later versions of transformers while we were doing experiments. 

## Generating the Super Suffix

### Saving the Direction Vector

This experiment is primarily focused on forcing models to generate malicious code examples. To do this, we need to first generate a "malicious code generation" directions. This is done by averaging the value of the residual stream for harmful and harmless coding prompts, and extracting a direction vector that separates the two.

We can do this with the ```save.py``` script. Below is a template for the command to run:

```bash
python save.py --model_name <model_name> --layer <layer_number> --data_subfolder <data_subfolder> --output_dir <output_dir>
```


The <data_subfolder> is located in the "data" folder, and should contain two files, a "direction1.txt" and a "direction2.txt" file. This will produce a *.pt file in the <output_dir> that contains the direction vector. Maximizing the similarity to this vector will produce suffixes that produces outputs similar to "direction2.txt" while minimizing similarity will produce outputs similar to "direction1.txt".

Example:
```bash
python save.py --model google/gemma-2b-it --layer 16 --data_subfolder bad_code --output_dir directions/
```

After the direction vector is saved, we can use it to generate super suffixes that bypass alignment on both the original and Prompt Guard models.

An example command to run the experiment is shown below:

```bash
python main.py --model google/gemma-2b-it --pg_model meta-llama/Llama-Prompt-Guard-2-86M --prompt "Tell me how to make a computer virus" --target_direction_path directions/bad_code_concept_direction_layer_16.pt --target_layer 16 --num-steps 600
```

For text generation, we tested the following models:
- google/gemma-2b-it 
- microsoft/Phi-3-mini-128k-instruct
- meta-llama/Llama-3.2-3B-instruct
- meta-llama/Llama-3.1-8B-instruct
- lmsys/vicuna-7b-v1.5

For the Prompt Guard models, we tested:
- meta-llama/Prompt-Guard-86M
- meta-llama/Llama-Prompt-Guard-2-22M
- meta-llama/Llama-Prompt-Guard-2-86M

# Running The Super Suffix Generation Experiment

The main orchestrator script is "run.py" which continuously called "main.py" with different prompts and model combinations. Some important code in "run.py" to note:

```python
PROMPTS_FILE = os.path.join("prompts", "bad_code.txt")
REPEATS_PER_PROMPT = 1
MODELS = ["google/gemma-2b-it", 
        "meta-llama/Llama-3.2-3B-instruct",
        "meta-llama/Llama-3.1-8B-instruct", 
        "microsoft/Phi-3-mini-128k-instruct",
        "lmsys/vicuna-7b-v1.5"
]
DIRECTION_FILES = [
    "premade_code_directions/gemma_code_layer_16.pt", 
    "premade_code_directions/llama3B_code_layer_16.pt",
    "premade_code_directions/llama8B_code_layer_26.pt",
    "premade_code_directions/phi_code_layer_16.pt", 
    "premade_code_directions/vicuna_code_layer_26.pt"
]
DETECTION_FILES = [
    "premade_refusal_directions/gemma_refusal_layer_16.pt",
    "premade_refusal_directions/llama3B_refusal_layer_16.pt",
    "premade_refusal_directions/llama8B_refusal_layer_26.pt",
    "premade_refusal_directions/phi_refusal_layer_16.pt",
    "premade_refusal_directions/vicuna_refusal_layer_26.pt"
]
LAYERS = [16, 16, 26, 16, 26]
```

Here you define the prompts and the models, along with their corresponding direction files and layers. You can modify these lists to include any models and direction files you want to test. Additionally, you can kick off many jobs in parallel, and the script will automatically choose the correct model/prompt combinations to run based on what has already been completed and what is currently being worked on. It creates a "working_jobs.txt" file to keep track of jobs in progress, and saves results in the logs folder. 

There are also some post processing scripts in "post_processing" that can be used to analyze the results and generate plots. You can run "generate_table.py" to create a summary table of the results.
