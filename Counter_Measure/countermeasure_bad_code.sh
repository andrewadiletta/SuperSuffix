<<COMMENT_BLOCK
Below is a reference to the model indices used in this script:

0: "google/gemma-2b-it", 
1: "meta-llama/Llama-3.2-3B-instruct",
2: "meta-llama/Llama-3.1-8B-instruct", 
3: "microsoft/Phi-3-mini-128k-instruct",
4: "lmsys/vicuna-7b-v1.5"
COMMENT_BLOCK

python countermeasure.py \
    --log-dir "PREVIOUS_RUNS/malicious_code/previous_logs" \
    --good-prompts "prompts/bad_code/good_code.txt" \
    --bad-prompts "prompts/bad_code/bad_code.txt" \
    --model-index 0