# Implement of Deep Multi-attribute Recognition model under ResNet50 backbone network

## Preparation
<font face="Times New Roman" size=4>

**Prerequisite: Python 3.10.13 and Pytorch 2.2.0**

1. Install [Pytorch](https://pytorch.org/)

2. Download and prepare the dataset as follow:

    a. PETA
    - Datset內容
    ```
    ../Dataset/peta/images/*.png
    ../Dataset/peta/PETA.mat
    ../Dataset/peta/README
    ```
    - 執行底下程式生成 peta_partition.pkl 再做後續 training
    ```
    python script/dataset/transform_peta.py 
    ```

    b. RAP
    ```
    ../Dataset/rap/RAP_dataset/*.png
    ../Dataset/rap/RAP_annotation/RAP_annotation.mat
    ```
    ```
    python script/dataset/transform_rap.py
    ```

    c. PA100K
    ```
    ../Dataset/pa100k/data/*.png
    ../Dataset/pa100k/annotation.mat
    ``` 
    ```
    python script/dataset/transform_pa100k.py 
    ```

    d. RAP(v2)
    ```
    ../Dataset/rap2/RAP_dataset/*.png
    ../Dataset/rap2/RAP_annotation/RAP_annotation.mat
    ```
    ```
    python script/dataset/transform_rap2.py
    ```
3. File location
   ```
    PAR_PATH
    ├── Dataset
    │   ├── pa100k
    │   │   │── data
    │   │   │── annotation.mat
    │   │── PETA
    │   │   │── images
    │   │   │── PETA.mat
    │   │── rapv2
    │   │   │── RAP_annotation
    │   │   │── RAP_dataset
    │   │── RAPv1
    ├── pedestrian_attribute_recognition
    |   │── script
    |   │   ├── dataset
    |   │   │   ├── transform_pa100k.py
    |   │   │   ├── transform_peta.py
    |   │   │   ├── transform_rap.py
    |   │   │   ├── transform_rap2.py
    |   │   ├── experiment
    |   │   │   ├── train.sh
    |   │   │   ├── test.sh
    |   │   │   ├── train_deepmar_resnet50.py
    |   │   │   ├── baseline
    |   │   │   |   ├── ...
    |   │── exp
    |   │   ├── deepmar_resnet50
    ├── ...
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
