# Implement of Deep Multi-attribute Recognition model under ResNet50 backbone network

## Preparation
<font face="Times New Roman" size=4>

**Prerequisite: Python 3.10.13 and Pytorch 2.2.0**

1. Install [Pytorch](https://pytorch.org/)

2. Download and prepare the dataset as follow:

    a. PETA
    
    ```
    ./dataset/peta/images/*.png
    ./dataset/peta/PETA.mat
    ./dataset/peta/README
    ```
    ```
    python script/dataset/transform_peta.py 
    ```

    b. RAP
    ```
    ./dataset/rap/RAP_dataset/*.png
    ./dataset/rap/RAP_annotation/RAP_annotation.mat
    ```
    ```
    python script/dataset/transform_rap.py
    ```

    c. PA100K
    ```
    ./dataset/pa100k/data/*.png
    ./dataset/pa100k/annotation.mat
    ``` 
    ```
    python script/dataset/transform_pa100k.py 
    ```

    d. RAP(v2)
    ```
    ./dataset/rap2/RAP_dataset/*.png
    ./dataset/rap2/RAP_annotation/RAP_annotation.mat
    ```
    ```
    python script/dataset/transform_rap2.py
    ```
</font>

## Train the model
<font face="Times New Roman" size=4>

   ```
   sh script/experiment/train.sh
   ``` 
</font>

## Test the model
<font face="Times New Roman" size=4>

   ```
   sh script/experiment/test.sh
   ```

</font>

## Citation
<font face="Times New Roman" size=4>
Please cite this paper in your publications if it helps your research:
</font>

```
@inproceedings{li2015deepmar,
    author = {Dangwei Li and Xiaotang Chen and Kaiqi Huang},
    title = {Multi-attribute Learning for Pedestrian Attribute Recognition in Surveillance Scenarios},
    booktitle = {ACPR},
    pages={111--115},
    year = {2015}
}
```

## Thanks
<font face="Times New Roman" size=4>

Partial codes are based on the repository from [Dangwei Li](https://github.com/dangweili/pedestrian-attribute-recognition-pytorch).

The code should only be used for academic research.

</font>
