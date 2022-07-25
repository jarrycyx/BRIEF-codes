# BRIEF: Biological Data Compression with Implicit Neural Function

## Environment

```
conda create -n brief python=3.8
conda activate brief
pip install -r requirements.txt
```

## Data Preparation
Demo data can be downloaded from [neurons.tif](https://file-cyx.oss-cn-hangzhou.aliyuncs.com/neurons.tif). Data should be placed into *./data/*.

## Data Compression

```
python main.py -opt opt/neurons.yaml -g 0
```

## Results

To be added...