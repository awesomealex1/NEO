# NEO

This is the official repo for NEO — an efficient TTA method that can adapt at practically no additional computational cost. NEO works by recentering the embeddings of domain-shifted samples at the origin. 

## Setup

Install the packages listed in ```requirements.txt``` with Python 3.11.8:

```
pip install -r requirements.txt
```

If you want additional logging using wandb, set up your own entity in the ```init_wandb``` method, which you can find in ```utils/utils.py```. If you do not set this up wandb will run in offline mode. Additionally you need to add the ```--wandb``` flag to activate wandb logging.

## Run experiments

To run experiments you need to first add ImageNet-C, CIFAR-10-C, ImageNet-Rendition or ImageNet-Sketch datasets.

The experiment arguments are defined in the argument parser in  ```utils/utils.py```. Make sure that the path to the datasets is correct. The default path to ImageNet-C for example is ```/data/imagenet-c```.

To run an experiment just type ```python main.py``` and add the command line arguments you want to specify. The default arguments run no adaptation on ImageNet-C for all corruption types for the whole dataset. Results are automatically logged in the directory ```/results```.

Common arguments are ```--corruption``` which specifies either the type of corruption in ImageNet-C or CIFAR-10-C (eg. gaussian_noise) or ```--cifar_10``` to run CIFAR-10-C instead of ImageNet-C, ```adapt_num_samples``` to specify on how many samples adaptation should be performed, ```--eval``` which specifies whether to use non-adaptation samples for validation (default is to just record accuracy during adaptation), or ```--vit_type``` which specifies the size of the ViT (small, base, large).

### Examples

Evaluate ViT-Base on ImageNet-C with NEO on 256 samples and then record validation accuracy on non-adaptation data:

```
python main.py --algorithm neo --adapt_num_samples 256 --eval
```

Evaluate ViT-Small on CIFAR-10-C Gaussian Noise with NEO:

```
python main.py --type small --cifar_10 --algorithm neo --corruption gaussian_noise
```

## NEO layer

The layers to replace the ```nn.Linear``` classifier with NEO are in ```neo_layers.py```. To add the NEO layer in the ```timm``` library, use an editable install (automatically done in our ```requirements.txt```), import the layer and replace it. In ```timm/models/vision_transformer.py```

```
self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
```

becomes

```
self.head = NEO(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
```

For ```transformers``` library also perform an editable install and in ```src/transformers/models/vit/modeling_vit.py```:

```
self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
```

becomes

```
self.head = NEO(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
```

## Example Results

On Vit-Base with 512 samples of adaptation NEO achieves high accuracy compared to other common TTA methods:

| Corruption | No Adapt | T3A | SAR | LAME | TENT | CoTTA | FOA | Surgeon | NEO |
|------------|----------|------|------|------|------|-------|------|---------|-----------|
| **_Noise_** | | | | | | | | | |
| Gaussian | 57.0 (0.5) | 56.7 (1.7) | 57.0 (1.7) | 56.5 (1.7) | 57.2 (1.6) | 57.0 (1.6) | 57.2 (0.6) | **58.7 (0.6)** | <u>57.7 (0.5)</u> |
| Shot | 56.9 (0.5) | 57.0 (1.0) | 57.3 (0.9) | 56.8 (1.0) | 57.5 (1.0) | 57.1 (1.1) | 58.6 (0.4) | **58.8 (0.5)** | <u>57.6 (0.5)</u> |
| Impulse | 57.4 (0.4) | 57.0 (1.0) | 57.5 (1.1) | 56.7 (0.9) | 57.6 (1.0) | 57.7 (1.1) | 57.7 (0.4) | **58.9 (0.6)** | <u>58.1 (0.4)</u> |
| **_Blur_** | | | | | | | | | |
| Defocus | 46.9 (0.5) | 47.5 (1.2) | 47.5 (1.2) | 47.1 (1.1) | 48.0 (1.2) | 48.2 (1.2) | <u>49.2 (0.4)</u> | 49.1 (0.8) | **49.8 (0.5)** |
| Glass | 35.3 (0.5) | 35.9 (0.9) | 36.3 (0.9) | 34.9 (1.0) | <u>36.8 (0.8)</u> | 36.0 (1.0) | <u>36.8 (0.5)</u> | <u>36.8 (0.7)</u> | **37.9 (0.4)** |
| Motion | 53.3 (0.4) | 53.1 (1.1) | 53.5 (1.1) | 52.9 (1.0) | 54.0 (1.0) | 53.3 (1.1) | 54.4 (0.4) | <u>54.8 (0.6)</u> | **55.0 (0.4)** |
| Zoom | 44.8 (0.5) | 45.3 (1.4) | 46.3 (1.5) | 44.7 (1.3) | 46.4 (1.4) | 45.0 (1.4) | <u>47.1 (0.5)</u> | 45.7 (0.6) | **47.5 (0.5)** |
| **_Weather_** | | | | | | | | | |
| Snow | 62.2 (0.5) | 62.7 (1.7) | 62.6 (1.5) | 58.5 (1.6) | 63.0 (1.6) | 63.2 (1.4) | <u>64.3 (0.4)</u> | 62.2 (0.7) | **64.6 (0.5)** |
| Frost | 62.6 (0.5) | 63.3 (1.5) | 63.3 (1.5) | 62.2 (1.5) | 63.3 (1.4) | 63.0 (1.4) | <u>63.9 (0.4)</u> | 61.7 (0.6) | **65.0 (0.5)** |
| Fog | 65.8 (0.4) | 62.9 (1.2) | 65.4 (1.1) | 62.0 (1.0) | 62.4 (1.3) | 64.8 (1.0) | <u>70.7 (0.4)</u> | 63.0 (0.6) | **71.2 (0.4)** |
| Brightness | 77.9 (0.4) | 78.1 (1.1) | 78.0 (1.1) | 78.1 (1.2) | 78.2 (1.2) | 77.9 (0.8) | <u>78.2 (0.4)</u> | 78.1 (0.5) | **78.3 (0.4)** |
| **_Digital_** | | | | | | | | | |
| Contrast | 32.6 (0.4) | 27.5 (1.2) | 34.0 (1.3) | 24.9 (1.4) | 36.9 (1.0) | 33.2 (1.0) | <u>54.5 (0.5)</u> | 31.7 (1.5) | **58.2 (0.4)** |
| Elastic | 45.8 (0.4) | 45.8 (0.9) | 45.7 (0.9) | 44.4 (0.7) | 46.7 (0.8) | 46.3 (1.2) | <u>49.6 (0.4)</u> | 46.2 (0.7) | **49.8 (0.4)** |
| Pixelate | 67.5 (0.4) | 67.4 (1.2) | 67.5 (1.1) | 67.1 (1.1) | <u>68.0 (1.1)</u> | 67.8 (1.1) | 67.2 (0.4) | 67.6 (0.7) | **68.2 (0.4)** |
| JPEG | 67.9 (0.4) | 67.3 (1.5) | 67.3 (1.5) | 66.9 (1.4) | 67.6 (1.5) | 67.8 (1.7) | 68.6 (0.5) | <u>68.9 (0.7)</u> | **69.1 (0.4)** |
| **ImageNet-C** | 55.6 (0.4) | 55.2 (1.3) | 55.9 (1.2) | 54.2 (1.2) | 56.3 (1.2) | 55.9 (1.2) | <u>58.4 (0.4)</u> | 56.1 (0.7) | **59.2 (0.4)** |
| **CIFAR-10-C** | 80.4 (2.8) | 80.1 (2.6) | 80.6 (2.8) | 79.8 (2.9) | 81.3 (3.2) | 80.6 (2.0) | 80.9 (2.8) | **82.7 (1.7)** | <u>82.4 (2.2)</u> |
| **ImageNet-R** | 59.2 (1.1) | 58.7 (1.2) | 59.3 (1.1) | 58.5 (1.1) | 59.4 (1.1) | 59.2 (1.2) | <u>60.2 (1.4)</u> | <u>60.2 (1.6)</u> | **60.3 (1.0)** |
| **ImageNet-S** | 45.4 (1.4) | 45.5 (1.4) | 45.5 (1.4) | 45.0 (1.3) | 45.7 (1.4) | 45.2 (1.6) | 46.3 (1.7) | <u>47.0 (1.7)</u> | **47.2 (1.4)** |

## Acknowledgment

This code is inspired from [FOA](https://github.com/mr-eggplant/FOA).