TASK 2
Objective: produce a reproducible benchmark suite so we can start experiments immediately after infra is ready. Your goal is only: faithful reproduction + clean implementation + unified interfaces and not implementation work right now.
Contributor 2 
Own implementation + reproducible configs for:
REP 
Paper: https://arxiv.org/pdf/2406.04772
HiDE 
Paper: https://proceedings.neurips.cc/paper_files/paper/2023/file/d9f8b5abc8e0926539ecbb492af7b2f1-Paper-Conference.pdf
RanPAC in progress
Paper: https://arxiv.org/pdf/2307.02251
SLDA done
Paper: https://openaccess.thecvf.com/content_CVPRW_2020/papers/w15/Hayes_Lifelong_Machine_Learning_With_Deep_Streaming_Linear_Discriminant_Analysis_CVPRW_2020_paper.pdf
NCM -  
Paper: https://inria.hal.science/hal-00817211/file/mensink13pami.pdf
Deliverables:
baselines/
    rep/
    hide/
    ranpac/
    slda/
    ncm/


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
Final handoff:
Each method must run with: python train.py --config config.yaml
README must include:
paper reproduced
assumptions
unsupported features
expected runtime

Research paper references are in ./references/
