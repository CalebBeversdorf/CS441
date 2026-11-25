# CS 441 Group Project
## How to Use
1. Make sure [uv](https://docs.astral.sh/uv/getting-started/installation/) is installed on your machine (Python 3.12). Then, clone the repo, `cd` into the root, and run `uv sync` with the `--extra` flag matching your GPU / hardware to install all necessary Python dependencies:

   | Hardware / OS | Command | Accelerator Used |
   | --- | --- | --- |
   | **NVIDIA GPU** (Windows / Linux) | `uv sync --extra cu128` | `cuda` (CUDA 12.8) |
   | **AMD GPU** (Linux, ROCm 6.4) | `uv sync --extra rocm` | `cuda` (ROCm) |
   | **Intel Arc / Data Center GPU** (Windows / Linux) | `uv sync --extra xpu` | `xpu` |
   | **Apple Silicon** (macOS) | `uv sync --extra cpu` | `mps` |
   | **CPU only** (Windows / Linux / Intel Mac) | `uv sync --extra cpu` | `cpu` |

   > **Note:** Official stable PyTorch ROCm wheels are Linux-only. On Windows with an AMD GPU, use `uv sync --extra cpu` (or run under WSL2 with Linux ROCm). All training, evaluation, and inference code automatically selects the best available accelerator in order: `cuda` (NVIDIA / AMD ROCm) → `xpu` (Intel) → `mps` (Apple Silicon) → `cpu`.

2. While that's running, start downloading the training data from [here](https://drive.google.com/file/d/12DzJyqPnUdsKmx9M5UXibxmCmChIhnEE/view?usp=sharing). It's just less than 40 GB and contains nearly 100k images, so it will take a while. The dataset contains curated images from [yfcc15m](https://huggingface.co/datasets/mehdidc/yfcc15m), [pastvu](https://huggingface.co/datasets/nyuuzyou/pastvu), [osv5m](https://huggingface.co/datasets/osv5m/osv5m), and the [Wikimedia Commons](https://commons.wikimedia.org/wiki/Main_Page). Extract the data folder to the root of the project.
3. Most of the code for this project is contained in the notebooks folder. I split the code into several different notebooks by function. The data was collected via the scripts in the 01 notebooks. I then found the optimal tiling for world locations in the 02 notebook. The relevant output from those notebooks are contained in the data folder, so you shouldn't need to run them unless you want to change the dataset for further training. The 03 notebooks contain the linear probing and fine-tuning code for 2 different models. **The rest of the notebooks are unfinished.**
4. Play with the Timeguessr AI interactively with the 07 notebook!
5. Also if you want to get the models I've already trained, I uploaded my output folder [here](https://drive.google.com/file/d/1TcDMZRz-5iXMJ1cLkoZGiCSIYsA4FHn4/view?usp=drive_link). You can extract the output folder to the root of the project and run the 07 notebook right away. 

## Warnings
Although I've tried to make this code cross platform, be aware that there might be issues with different GPU brands or operating systems. I made this on a windows machine with a Nvidia RTX 4070.

## Attributions
Gemini AI was used for large portions of the code for this project, however not without significant verification and guidance.
