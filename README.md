# Multiview Scene Graph (NeurIPS 2024)
This is the official implementation of 
>[Multiview Scene Graph](https://ai4ce.github.io/MSG/) 
>
>Juexiao Zhang, Gao Zhu, Sihang Li, Xinhao Liu, Haorui Song, Xinran Tang, Chen Feng
>
> New York University

### [**[Project page]**](https://ai4ce.github.io/MSG) **|** [**[Paper]**](https://arxiv.org/abs/2410.11187)

![teaser](./media/teaser.jpg)
<div style="text-align: center;">
  <img src="media/scene1.gif" alt="Description of GIF">
</div>

## Implementations
### Requirements

First, setup the environment by running
```shell
git clone https://github.com/ai4ce/MSG.git
cd msg
conda create --name msg python=3.11.8
conda activate msg
pip install -r requirements.txt
```
This `requirements.txt` contains minimum dependencies estimated by running `pipreqs`.

*Alternatively*, to fully replicate the environment you can also run:
```shell
git clone https://github.com/ai4ce/MSG.git
cd msg
conda env create -f environment.yml
conda activate msg
```
### Data and weights

MSG data is converted from Apple's [ARKitScenes](https://github.com/apple/ARKitScenes) by transforming its 3D annotations to 2D.
The converted dataset can be found at this [Dataset Hub](https://huggingface.co/datasets/ai4ce/MSG) on Huggingface.
We have also kept the code snippets for data convertion in `data_preprocess`.

To use the data, download and unzip the data to `./data/msg`
- [ ] TODO: specify the data usage. 

```shell
mkdir -p data/msg
```

We also provide pretrained [checkpoint](https://huggingface.co/datasets/ai4ce/MSG) of our AoMSG model in the same hub.

To use the checkpoint, download it to `./exp-results/aomsg`
- [ ] TODO: specify the checkpoint usage

```shell
mkdir -p exp-results/aomsg

```

### Inference

To do inference with the pretrained weights, run:

```shell
python inference.py --experiment inference
```
which loads configurations from the file `./configs/experiments/inference.yaml`, where the dataset path and the evaluation checkpoint are specified.
You can also specify them via arguments which will overwrite the YAML configs. For example:
```shell
python inference.py --experiment inference \
--dataset_path PATH/TO/DATASET \
--eval_output_dir PATH/TO/MODEL/CHECKPOINT \
--eval_chkpt CHECKPOINT/FILE
```

Additional to inference, you can also leverage MSG for topological localization. Please see `localization.py` for details.

### Training

To train the AoMSG model for MSG:
```shell
python train.py --experiment aomsg
```

To train the SepMSG baselines:
```shell
python train.py --experiment sepmsg
```
Please refer to the respective configuration files `./configs/experiments/aomsg.yaml` and `./configs/experiments/sepmsg.yaml` for the detailed settings.

To resume training of a pretrained checkpoint, set `resume=True` and specify the `resume_path` to the checkpoint in the corresponding YAML configuration files.


For evaluation, simply change the script while keep the same `experiment` configuration, in which `eval_output_dir` and `eval_chkpt` are specified.
```shell
# evaluate AoMSG
python eval.py --experiment aomsg 
# evaluate SepSMG
python eval.py --experiment sepmsg 
# evaluate SepMSG-direct, which directly use features from froze backbone for MSG
python eval.py --experiment direct 
```

> **NOTE:**
> 
> This release focuses on the implementation of MSG. Object detection dependency is not included. 
> To use detection results instead of groundtruth detection, we can specify detection results in files and give the `result_path` as is illustrated in `./configs/experiments/aomsg_gdino.yaml` where detection results obtained from [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) is used.
> 
> This means you need to run detection separately and save the results to a path. In the data hub we provide the gdino results for convenience. In the future release, we may include a version incorporating online detection.

## BibTex
```
@misc{zhang2024multiviewscenegraph,
      title={Multiview Scene Graph}, 
      author={Juexiao Zhang and Gao Zhu and Sihang Li and Xinhao Liu and Haorui Song and Xinran Tang and Chen Feng},
      year={2024},
      eprint={2410.11187},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.11187}, 
}
```
