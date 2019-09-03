# Self-supervised Deep Depth Denoising

# Installation
The code has been tested with the following setup:
  * Pytorch 1.0.1
  * Python 3.7.2
  * CUDA 9.1
  * [Visdom](https://github.com/facebookresearch/visdom)
  
# Train
To train the model for depth denoising:
```python train.py```
To see the available training parameters:
```python train.py -h```

# Inference
To denoise a RealSense sample using a pretrained model:
```python inference.py --model_path /path/to/pretrained/model --input_path /path/to/noisy/sample --output_path /path/to/save/denoised/sample```

In order to save the input (noisy) and the output (denoised) samples as pointclouds add the following flag to the inference script execution:
```--pointclouds True```

To denoise a sample using the pretrained autoencoder (same model trained without splatting) add the following flag to the inference script:
```--autoencoder True```

# Citation
If you use this code and/or models, please cite the following:
```
@inproceedings{sterzentsenko2019denoising,
  author       = "Vladimiros Sterzentsenko and Leonidas Saroglou and Anargyros Chatzitofis and Spyridon Thermos and Nikolaos Zioulis and Alexandros Doumanoglou and Dimitrios Zarpalas and Petros Daras",
  title        = "Self-Supervised Deep Depth Denoising",
  booktitle    = "ICCV",
  year         = "2019"
}
```

# License
