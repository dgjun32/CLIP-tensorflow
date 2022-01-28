# Tensorflow CLIP implementation

## 1. Dependencies
* tensorflow 2.7.0
* numpy 1.19.5
* pandas 1.3.4
* easydict 1.9

## 2. Approach
CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image.

<img src = 'tensorflow_clip/imgs/overview-a.svg'>

I used ```MS-COCO dataset```, which contains 118K image-caption pairs, as ```WIT dataset``` is too large to run on my local machine.

If you want to run with ```WIT dataset```, you may use ```tfds API``` using code below.
    ```
    import tensorflow_datasets as tfds
    dataset = tfds.load(name='wit', split='train')
    ```

## 3. Dataset
* Download 2017 version dataset from the [link]('https://cocodataset.org/#download')
* Save each sub directory in ```../data``` dir as below. 
    ```
    - data|-- train2017
          |-- test2017
          |-- val2017
          |-- annotations
    ```

## 4. Training
* change current directory : ```$ cd code```
* train start : ```$ python main.py --model_name (select one of [vit-B/32, vit-B/16, vit-L/14])```

## 5. Demo
I ran demo for Retrieval task using pretrained CLIP. you can see [demo.ipynb]('tensorflow_clip/code/demo.ipynb') to see the code of image retrieval module.

### 5.1. Text to Image Retrieval Result

Title of each images are query text.
<img src = 'tensorflow_clip/imgs/6b7986ef-808f-4e47-978c-018bed8d3b09.png'>
<img src = 'tensorflow_clip/imgs/07e1b634-7c70-4358-9fb3-ba7c0413965b.png'> 
<img src = 'tensorflow_clip/imgs/1161bca9-e4e3-40f1-ac64-3e27ffcf1c55.png'>

### 5.2. Image+text to Image Retrieval Result
In this demo, query consists of single image and text. By using the sum of representation of image and text in query, I retrieved the most relevant image.

* Image of Teddy bear + (color)
<img src = 'tensorflow_clip/imgs/b3757903-69a8-43d0-a9dc-41450b620fa4.png'><br>

<p><img src = 'tensorflow_clip/imgs/e9a34495-7824-4b13-b463-82e7ada069d3-1.png'>
    <img src = 'tensorflow_clip/imgs/752af4c3-9ff0-4bad-bede-ad1baf02de86.png'></p>

* Image of Airplane + (object or scene)
<img src = 'tensorflow_clip/imgs/7116d0d6-795e-46f8-bf2a-4db850171c27.png'><br>

<p><img src = 'tensorflow_clip/imgs/c6b2e7e1-10ba-4cfd-85a9-8bfe18d5df15.png'>
    <img src = 'tensorflow_clip/imgs/ed2d3d54-c595-4886-baab-d1f0a63b26c6.png'></p>

    


