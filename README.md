# SO-ARM101 Pick-and-Place 

A physical robotic arm trained end-to-end to pick up a block and place it in a bin, using imitation learning on real hardware.

**Tech Stack:** Python, PyTorch, LeRobot, SmolVLA

**Dataset:** [prishaakapasi/pick_place_v1](https://huggingface.co/datasets/prishaakapasi/pick_place_v1)
**Trained policy:** [prishaakapasi/pick_place_smolvla_policy](https://huggingface.co/prishaakapasi/pick_place_smolvla_policy)

<br>

## overview

Built and calibrated a SO-ARM101 leader/follower robotic arm, recorded a real-world manipulation dataset via teleoperation, and fine-tuned a SmolVLA vision-language-action policy to autonomously pick up a block and place it in a bin, deployed and tested on the physical arm.

<br>

## pipeline

**1. Build & calibrate**
Assembled the SO-ARM101 leader and follower arms (12 STS3215 servos, two XIAO adapter boards) and calibrated both for teleoperation.

**2. Teleoperation**
Used the leader arm to control the follower arm in real time, with a wrist-mounted camera capturing the workspace.

**3. Data collection**
Recorded a 31-episode pick-and-place dataset (36,498 frames at 30fps), uploaded to Hugging Face.

**4. Training**
Fine-tuned SmolVLA on the recorded dataset via `lerobot-train`, on a free Colab T4/L4 GPU:

```bash
lerobot-train \
    --dataset.repo_id=prishaakapasi/pick_place_v1 \
    --policy.type=smolvla \
    --policy.pretrained_path=lerobot/smolvla_base \
    --output_dir=outputs/train/pick_place_smolvla \
    --job_name=pick_place_smolvla \
    --policy.device=cuda \
    --wandb.enable=false \
    --policy.repo_id=prishaakapasi/pick_place_smolvla_policy \
    --steps=8000 \
    --save_freq=2000
```

**5. Deployment**
Ran the trained policy on the physical follower arm. After diagnosing and fixing a camera-index mismatch (the model was briefly being fed the wrong camera feed instead of the workspace view), the arm successfully and autonomously picked up the block and placed it in the bin.

<br>

## how SmolVLA works

SmolVLA predicts chunks of 50 future actions at once using flow matching, rather than simple regression. It combines a frozen vision-language encoder with a trainable action expert connected via cross-attention. Fine-tuning here trained roughly 100M of the model's 450M parameters.

<br>

## results

- Training loss dropped from 0.056 → 0.016 over 8,000 steps
- Policy successfully completed autonomous pick-and-place after deployment fixes
- Full pipeline (build → teleop → collect → train → deploy) run end-to-end on consumer hardware and a free-tier GPU

<br>

## known limitations

- No learned "reset to home" behavior, since the training data didn't include it
- Small dataset (31 episodes); generalization to new block positions or lighting is untested
- Camera-index fragility surfaced during deployment, a reminder that deployment engineering is its own challenge beyond training

<br>

## next steps

This project is a stepping stone toward a longer-term interest in motor cortex decoding: eventually replacing the leader arm's teleoperation signal with output from a neural decoder trained on motor intent (e.g. MC_Maze / MC_RTT datasets), so a decoded signal, not a human operator, drives the arm.

<br>

## getting started

```bash
# clone the repo
git clone https://github.com/prishaakapasi/lerobot-.git
cd lerobot-

# clone LeRobot and install dependencies
git clone https://github.com/Seeed-Projects/lerobot.git
cd lerobot
pip install -e ".[feetech]"
```

`robottraining.py` was exported from a Colab notebook and is meant to be run cell-by-cell in a notebook environment (Colab or Jupyter), not as a standalone script from the terminal, since it includes notebook-specific commands (`!git clone`, `!pip install`) and an interactive Hugging Face login step.

Requires a Hugging Face account (for dataset/model access) and a CUDA-enabled GPU for training.

<br>

## built by

[Prishaa Kapasi](https://prishaakapasi.com) — [LinkedIn](https://www.linkedin.com/in/prishaa-kapasi-87b73825b)
