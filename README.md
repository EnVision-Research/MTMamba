# MTMamba

This repository contains codes and models for the following paper:

> Baijiong Lin, Weisen Jiang, Pengguang Chen, Yu Zhang, Shu Liu, and Ying-Cong Chen. MTMamba: Enhancing Multi-Task Dense Scene Understanding by Mamba-Based Decoders. In *European Conference on Computer Vision*, 2024.



## Requirements:

- PyTorch 2.0.0

- timm 0.9.16

- mmsegmentation 1.2.2

- mamba-ssm 1.1.2

- CUDA 11.8
  
  

## Usage

1. Prepare the pretrained Swin-Large checkpoint by running the following command
   
   ```shell
   cd pretrained_ckpts
   bash run.sh
   cd ../
   ```

2. Download the data from [PASCALContext.tar.gz](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hyeae_connect_ust_hk/ER57KyZdEdxPtgMCai7ioV0BXCmAhYzwFftCwkTiMmuM7w?e=2Ex4ab), [NYUDv2.tar.gz](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hyeae_connect_ust_hk/EZ-2tWIDYSFKk7SCcHRimskBhgecungms4WFa_L-255GrQ?e=6jAt4c), and then extract them. You need to modify the dataset directory as ```db_root``` variable in ```configs/mypath.py```.

3. Train the model. Taking training NYUDv2 as an example, you can run the following command
   
   ```shell
   python -m torch.distributed.launch --nproc_per_node 8 main.py --run_mode train --config_exp ./configs/mtmamba_nyud.yml 
   ```

        You can download the pre-trained model from [mtmamba_nyud.pth.tar](https://hkustgz-my.sharepoint.com/:u:/g/personal/blin241_connect_hkust-gz_edu_cn/EdP6lzTOEIRLggFVLlbzPWUBZrsRPoEkdtNpYjm_H2K54A?e=ekeobz), [mtmamba_pascal.pth.tar](https://hkustgz-my.sharepoint.com/:u:/g/personal/blin241_connect_hkust-gz_edu_cn/ET0zoRo2mq9OoYJlHZZy2eQB5lh6W-yayKzih6ejwD7awQ?e=DUZFGE).

4. Evaluation. You can run the following command,
   
   ```shell
   python -m torch.distributed.launch --nproc_per_node 1 main.py --run_mode infer --config_exp ./configs/mtmamba_nyud.yml --trained_model ./ckpts/mtmamba_nyud.pth.tar
   ```

Acknowledgement
---------------

We would like to thank the authors that release the public repositories: [Multi-Task-Transformer](https://github.com/prismformore/Multi-Task-Transformer), [mamba](https://github.com/state-spaces/mamba), and [VMamba](https://github.com/MzeroMiko/VMamba).



## Citation

If you found this code/work to be useful in your own research, please cite the following:

```latex
@inproceedings{lin2024mtmamba,
  title={{MTM}amba: Enhancing Multi-Task Dense Scene Understanding by Mamba-Based Decoders},
  author={Lin, Baijiong and Jiang, Weisen and Chen, Pengguang and Zhang, Yu and Liu, Shu and Chen, Ying-Cong},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```
