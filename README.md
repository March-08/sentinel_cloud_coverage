<p align="left">
  <a href="https://easybase.io">
    <img src="https://www.esa-bic.cz/wp-content/uploads/2021/07/logo-transparent.svg" alt="easybase logo black" width="100" height="100">
  </a>
</p>

<br />
<br />

<p align="center">
  <a href="https://easybase.io">
    <img src="https://cdn-icons-png.flaticon.com/512/1779/1779754.png" alt="easybase logo black" width="100" height="100">
  </a>
</p>

<br />


<h3 align="center">
  <b>
    <a href="https://dark-pufferfish-259.notion.site/DataA-b16ec247c20443b192059ade27cfbdcf">
      Cloud Coverage Detection 🚀
    </a>
  </b>
</h3>

<br />

<!-- DOCUMENTATION -->
## 📓 Problem
Develop a Model for the task of binary cloud segmentation. The model should have size less than 250MB.
The the trained model should be converted in onnx and used in inference with onnxruntime.

## 💡 Solution
I will firstly tile the images (and their mask) in order to obtain 224x224 images. Note thaht only the cloud channel is needed from the 3 mask channels.
Then I can use a pretrained network to solve the task. The problem with this is that these pretrained UNET bases network are pretty large, so pruning/quantization method is needed in this case. 
So instead I will develop a small UNET based network from scratch. The small model is based on the paper [Convolutional Neural Networks enable efficient, accurate and fine-grained segmentation of plant species and communities from high-resolution UAV imagery](https://www.nature.com/articles/s41598-019-53797-9)  

<center><img src="https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-019-53797-9/MediaObjects/41598_2019_53797_Fig1_HTML.png" alt="model" width="500" height="250"></center>
In our case the model should excpect 4 input channels and output a single binary mask. The model should be small enouh to fit the requirements.
Then I will convert it to onnx and run with onnxruntime. 
This Readme will help you to replicate all my work step by step.
<br/>


<!-- CLONE -->
## 💻 Clone Repo

Clone this repository with the following git command.
```
git clone https://github.com/March-08/sentinel_cloud_coverage.git
```

<!-- CREATE ENV -->
## 🌳 Create Env

Create a new Python virtual environment. More about virtual env with conda [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
```
conda create --name venv python
conda activate venv
```

<!-- INSTALL REQUIREMENTS -->
## ✒️ Install Requirements

Install needed packages from requirements.txt 
```
pip install -r requirements.txt
```


<!-- RUN -->
## 🏃‍♂️ Run

### Add Data
Now download end extract data from [zenodo](https://zenodo.org/record/4172871#.Y-KHonbMI2w). Create additional the directories images_patches and masks_patches. You should now have a data structure like the following:
```
├── data/
│   ├── images_patches
│   ├── masks
│   ├── masks_patches
│   ├── subscenes
│   ├── masks.zip
│   └── subscenes.zip
``` 

### Extract Data
The first step is to extract tiles from the subscenes. You can extract 224^2 images simply by running the following command.
```
python src/main.py extract
```
This will populate your patches directories.


### Train
In order to launch your training you can run the following command. FYI The hyper-parameters are hard coded  in this example task. 
```
python src/main.py train 
```


### ONNX Conversion
Now its time conver the PyTorch model to ONNX. I used the [torch.onnx module](https://pytorch.org/docs/stable/onnx.html). To run the conversion you can use the following command:
```
python src/main.py onnx
```
This script will firsly check the consistency of the pth model and the onnx model. Then you will find in your main directory the model.onnx which will be ready to be used in inference with onnxruntime.


### ONNX Inference
