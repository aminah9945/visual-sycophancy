# Environment (recorded 2026-11-17)

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB VRAM, driver 610.91 |
| CPU | AMD Ryzen 7 7445HS |
| RAM | 16 GB (15.2 GB usable) |
| OS | Windows 11 Home |
| Python | 3.12.10 |
| PyTorch | 2.6.0+cu124 (CUDA available) |
| Transformers | 5.23.0 |
| API access | OpenRouter (OpenAI-compatible client), model ids in `configs/models.yaml`, accessed from 2026-11-17 |

Consequences for the plan: 6 GB VRAM fits 2B vision readers in bf16 (Qwen3-VL-2B, InternVL-3.5-2B).
Qwen3-VL-4B (~8.8 GB bf16) does not fit without 4-bit quantisation. The plan's local Qwen2.5-7B generator is
not on OpenRouter and would be very slow in 4-bit on this GPU, so the open-weight generator is
Qwen3-30B-A3B-Instruct-2507, served through OpenRouter (see `runs/A/DEVIATIONS.md` and paper Sec. 4.1).
