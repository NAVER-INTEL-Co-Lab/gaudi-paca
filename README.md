# PaCA
**PaCA** (**Pa**rtial **C**onnection **A**daptation) is new parameter-efficient fine-tuning (PEFT) algorithm for enhancing performance. PaCA not only reduces activation memory by storing only partial activations for backward propagation, but also reduces training time by eliminating additional sequential process by additional adapter layers as below:

![image](https://github.com/user-attachments/assets/9b59b1b9-a4dd-4513-84e7-fc9e3551bbce)


## how to use
1. download custom peft library which supports PaCA.
```
cd peft
pip install -v -e .
```
2. use paca as follow
```
from peft import PacaConfig, get_peft_model

peft_config = PacaConfig(
                r=8,
                paca_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )

model = get_peft_model(model, peft_config)
```

