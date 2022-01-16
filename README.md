# Tensorflow CLIP implementation

## 1. Dependencies
* tensorflow 2.7.0
* numpy 1.19.5
* pandas 1.3.4
* easydict 1.9

## 2. Approach
CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image.

https://github.com/openai/CLIP/blob/main/CLIP.png

I used ```MS-COCO dataset```, which contains image-caption pairs, as ```WIT dataset``` is not open source and I had limited computing resources.

## 3. Dataset
* Download 2017 version dataset from the [link]('https://cocodataset.org/#download')
* Save each sub directory in ```../data``` dir as below. 
    ```
    -data |-- train2017
        |-- test2017
        |-- val2017
        |-- annotations
    ```

## 4. Training
* change current directory : ```$ cd code```
* train start : ```$ python main.py --model_name (one of [vit-B/32, vit-B/16, vit-L/14])```
