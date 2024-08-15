---
title: How to set up an Nvidia GPU for PyTorch on a local machine for use with Jupyter Notebook 
output:
  md_document:
    variant: gfm+footnotes
    preserve_yaml: TRUE
knit: (function(inputFile, encoding) {
  rmarkdown::render(inputFile, encoding = encoding, output_dir = "../_posts") })
date: 2024-08-14
permalink: /posts/2024/08/:title
excerpt_separator: <!--more-->
always_allow_html: true
toc: true
tags:
  - PyTorch
  - data-science
  - gpu
  - Nvidia
  - jupyter Notebook
  - anaconda
---

In this article I will show step-by-step on how to setup your GPU for train your ML models in Jupyter Notebook or your local system for Windows (using PyTorch).

# Requirements
- Python (latest version)
- Anaconda (latest version)
- Visual Studio (latest version)

For Visual Studio you will have to install certain workloads which will then automatically install C++ libraries required for CUDA.

## Visual Studio Installation
- Setep 1: Go to Visual Studio website [here](https://visualstudio.microsoft.com/) and download the installer.

- Step 2: While installing you will come on workloads page. Select Mobile Development with C++ and Desktop Development with C++, keep rest the same. It may take some time to install.

<img src="/images/posts/pytorch-gpu/vs-installer-modify-workloads.png" style="display: block; margin: auto;" />


# CUDA Toolkit and cuDNN installation
<i>Note:- Visual Studio must be installed before attempting this.</i>

## Installing CUDA Toolkit
- Download CUDA from official website [here](https://developer.nvidia.com/cuda-downloads), and install it.

## Installing cuDNN
- This step can be slightly tricky.
- Download cuDNN from Official website [here](https://developer.nvidia.com/cudnn). It will download a zip file then extract it (shown below).

<img src="/images/posts/pytorch-gpu/cuDNN-downlaods.png" style="display: block; margin: auto;" />

- Once extracted, copy the extracted file and past it into your C drive.

<img src="/images/posts/pytorch-gpu/cuDNN-in-C.png" style="display: block; margin: auto;" />

## Adding cuDNN to Environment Variable

- You will now have to add cuDNN to you Environment Variable. In the windows search type "edit the system environment variable" and Enter. You will come to this screen then click Environment Variable button.

<img src="/images/posts/pytorch-gpu/env-vari-main.png" style="display: block; margin: auto;" />

- It will take you inside, then double click on path in system variables.

<img src="/images/posts/pytorch-gpu/env-vari-inside.png" style="display: block; margin: auto;" />

- Inside the system variable path add the path of "bin", "include", and "lib/x64" from cuDNN in C drive, and apply the changes.

<img src="/images/posts/pytorch-gpu/env-vari-addpath.png" style="display: block; margin: auto;" />


Now you are done with Downloading and Setting up CUDA and cuDNN.

# Creating Virtual Environment with Conda and downloading PyTorch

Now open your Anaconda Prompt and type:

```bash
conda create -n envname python
```
<i>Replace envname with any name you would live to give</i>

To activate the virtual environment type:

```bash
conda activate envname
```
<i>Replace envname with any name you would live to give</i>

Once your virtual environment is activated to PyTorch website [Here](https://pytorch.org/). On the page you will find the following:

<img src="/images/posts/pytorch-gpu/pytorch-dow.png" style="display: block; margin: auto;" />

Select the appropriate settings. If your current CUDA version is not there then choose the closest version to it.
Paste the below URL to you Anaconda Prompt with virtual environment active.

# Using GPU inside Jupyter Notebook with PyTorch

We are almost done now!

## Changing environment on Anaconda

Open Anaconda Navigator and go to environments page as shown below.

<img src="/images/posts/pytorch-gpu/anaconda-main.png" style="display: block; margin: auto;" />

You can also see I have three environments there, base(root), gputest and mygpu. You will see your virtual environment name there. Double click it to select and then go back to home. You will see on top that the environment has changed. Also you will have to install Jupyter Notebook again in the new environment.
<img src="/images/posts/pytorch-gpu/anaconda-home.png" style="display: block; margin: auto;" />

After installation launch Jupyter though Anaconda Navigator when the appropriate virtual environment is selected.

## Final testing inside Jupyter Notebook

Once Jupyter Notebook is open write the following commands to check weather CUDA is working properly.

```bash
import torch
```
```bash
torch.cuda.is_available()
```
```bash
torch.cuda.current_device()
```
```bash
torch.cuda.get_device_name(0)

```

The output should be as follows

<img src="/images/posts/pytorch-gpu/jupyter-test.png" style="display: block; margin: auto;" />


Congratulation! Now you can successfully use your GPU for training and testing your PyTorch models in Jupyter Notebook.
If you wish to run it in your local system then activated your virtual environment in Anaconda Prompt and run you code there.
