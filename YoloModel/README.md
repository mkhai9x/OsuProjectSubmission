# COMP9517 Group Project

YOLO Algorithm for car detection

## Installation

Use the environment manager [anaconda](https://docs.anaconda.com/anaconda/install/windows/) to create and install the correct packages in a new virtual environment.

Tested and working with 
```bash
python 3.8.8
conda 4.10.0
```
### Step 1
Create and install new virtual environment
```bash
conda env create -f yolo_comp.yml
```

```bash
conda activate yolo_comp
```
### Step 2
Download the ```weights``` and ```keras_yolo3``` folder on [teams](https://teams.microsoft.com/_#/school/files/General?threadId=19%3Adee66286ad894e0488e38b7fa5e01563%40thread.tacv2&ctx=channel&context=General&rootfolder=%252Fsites%252FCOMP9517Osu%252FShared%2520Documents%252FGeneral) and put the contents in the root directory 

### Step 3
Populate the ```input_images``` folder with images to be detected

## Usage
Run contents in ```yolo.ipynb``` for image detection