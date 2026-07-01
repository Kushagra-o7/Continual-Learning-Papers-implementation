TASK 2
Objective: produce a reproducible benchmark suite so we can start experiments immediately after infra is ready. Your goal is only: faithful reproduction + clean implementation + unified interfaces and not implementation work right now.
Contributor 1 in progress
Own implementation + reproducible configs for:
Online-LoRA
Paper: https://arxiv.org/pdf/2411.05663
L2P in progress
Paper: https://arxiv.org/pdf/2112.08654
DualPrompt
Paper: https://arxiv.org/pdf/2204.04799
CODA Prompt
Paper: https://arxiv.org/pdf/2211.13218
Beyond Prompt Learning
Paper: https://arxiv.org/pdf/2407.10281
Deliverables:
baselines/
    online_lora/
    l2p/
    dualprompt/
    coda_prompt/
    beyond_prompt/

For EACH method include:
method/
├── model.py
├── train.py
├── config.yaml
├── README.md
├── requirements.txt
└── run.sh

Requirements:
frozen backbone(if any)
deterministic seeds
checkpoint save/load
configurable hyperparameters
train + eval scripts
single command execution
Command format:
python train.py --config config.yaml
README must include:
paper reproduced
assumptions
unsupported features
expected runtime



Research paper references are in ./references/
