
# **MoDL**
## Description
To achieve precise mitochondrial segmentation and function prediction in live-cell images, we have developed MoDL,
a deep learning-based software package. MoDL possesses two pipelines: 
(1) Trained on over 20,000 manually labeled mitochondria from SR images, MoDL achieves high-precision segmentation of 
mitochondrial contours from live-cell fluorescence images. This framework outperforms existing methods in delineating 
mitochondrial morphological features and is adaptable to diverse imaging platforms and cell types; 
(2) Based on high-quality segmentation and morphological features, MoDL is able to accurate prediction on various 
mitochondria functions by employing a multi-modal fusion technique. This pipeline is powered by an extended dataset 
contains over 100,000 SR images, each annotated with corresponding functional data obtained through biochemical assays. 
Example data and demo of MoDL for super-resolution microscopy images of mitochondria are as follows.
丁总具体再改动你的部分，简要描述项目即可。

***

## Table of Contents
 * Requirements
 * Installation
 * Usage（Usage为整体的pipeline说明,后续的章节为详细步骤）丁总部分
 * Data Preparation
 * Model Training
 * Model Prediction
 * 丁总的详细pipeline章节
 * 丁总的详细pipeline章节
 * Contributing
 * License

***

## Requirements

MoDL is built with Python and Tensorflow. Technically there are no limits to the operation system to run the code, 
but Windows system is recommended, on which the software has been tested. The inference process of the MoDL can 
run using the CPU only, but could be inefficiently. A powerful CUDA-enabled GPU device that can speed up the 
inference is highly recommended.

The inference process has been tested with:

 * Windows 11 (version 23H2)
 * Python 3.7 (64 bit)
 * tensorflow 2.5.0
 * 11th Gen Intel(R) Core(TM) i7-11700 @ 2.50GHz
 * NVIDIA GeForce RTX 3060 Ti

***

## Installation

1. Install python 3.7 
2. (Optional) If your computer has a CUDA-enabled GPU, install the CUDA and CUDNN of the proper version.
3. The directory tree has been built. Download the MoDL.zip and unpack it, or clone the repository:
```
git clone https://github.com/Jintao-Li-Brocwave/MoDL.git
```

4. Open the terminal in the MoDL directory, install the required dependencies using pip:

```
pip install -r requirements.txt
```

5. (Optional) If you have had the CUDA environment installed properly, run:

```
pip install tensorflow-gpu=2.5.0
```

The installation takes about 10 minutes in the tested platform. The time could be longer due to the network states.

*** 

##Usage
To use this project, follow these steps:

1.Prepare the training data by running data_load.py. Due to the limitation of data size, the training images and 
corresponding ground truth of mitochondria can be downloaded [here](丁总训练集512大小的train原图和label标签的链接地址). 
You will need to download it and unzip it separately to the *' deform/train '* and *' deform/label '* directories 
of the original MoDL demo.

2.Train the model by running train.py, then you will get a model for super-resolution microscopy images segmentation of
mitochondrial, and the trained model will be saved in the *' model '* directory of the original MoDL demo.

3.(Optional) Also, you can directly use our pre-trained model U-RNet+ to predict [here](丁总预训练模型链接地址), You will 
need to download it and unzip it to the *' model '* directory of the original MoDL demo.

4.Prepare the test images and use the trained model to make predictions by running segment_predict.py. Due to the 
limitation of data size, the test images can be downloaded [here](丁总测试集2048大小的原图链接地址). You will need to 
download it and unzip it to the *' testraw '* directory of the original MoDL demo. After prediction, the predicted 
segmentations and their pseudo-color implementation are stored separately in the *' final_results/bw '* and 
*' final_results/pseudo '* directories of the original MoDL demo.

5.丁总部分

6.丁总部分

***
##**A specific file description are as follows:**
##Data Preparation
1.Place the training images in the *' deform/train '* directory. 

2.Place the corresponding labels in the *' deform/label '* directory.

3.Run the ***data_load.py*** to convert the images and labels into .npy format.


##Model Training
1.Run the ***train.py*** to train the model.

2.The trained model will be saved in the *'model'* directory and named *U-RNet+.hdf5*.

3.The training progress and performance metrics will also be saved in the *'model'* directory after training.


##Model Prediction
1.Place the test images to be segmented in the *' testraw '* directory.

2.Run the ***segment_predict.py*** to make predictions using the trained model.

3.The predicted segmentations of patches in three ways (4×4 patches, 4×3 patches, 3×4 patches) and their pseudo-color 
implementation are stored separately in the corresponding *' results/results_xx/bw '* and 
*' results/results_xx/pseudo '* directories.

4.The final merged segmentations of three ways and their pseudo-color implementation are stored separately in the 
*' final_results/bw '* and *' final_results/pseudo '* directories.

##丁总

1.丁总

2.丁总


##Contributing
Contributions to this project are welcome! 

Here are a few ways you can contribute:
Report bugs or suggest improvements by creating a new issue.
Implement new features or fix existing issues by creating a pull request.


##License
This project is covered under the GNU General Public 3.0 License.


 

