# Group7: Visual Saliency: Methods of Identification and Applications

###### tags: `NYCU Course`,`影像處理`
| Student ID | 309554012 | 310553041 | 309653008 | 310554037 | 310554031 |
| ---------- | --------- | --------- | --------- | --------- | --------- |
| Name | 施長佑 | 管彥鳴 | 楊大昕 | 黃乾哲 | 葉詠富 |


## Outline:
### A. Introduction:
1. What is Visual Saliency
2. Visual Saliency Example
3. Neural and Computational Mechanisms
4. Application
    - Auto-Detection
    - Auto-driver
    - Medical image
    - Auto image cuting
    - Explainable AI

### B. Computer Vision Method
1. Static Salieny
3. Motion Saliency

### C. Artificial Intelligent Method
1. Deconvolution Network 
2. CAM 

### D. Conclusion

### E. Group Member Contributions

### F. Reference

<br>

## A. Introduction:
### 1. What is Visual Saliency?
Visual saliency is the distinct subjective perceptual quality. It can make some items in the world stand out from their neighbors and immediately grab our attention. In ancient times, virsual asliency help animals to detect the most important thing for live. It is affected by the past experiments, interests, or habits. 

In the modern, we can get some visual saliency experience by education. For example, the below, since we know when the front car's brake light is bright, the front car will break. We also break our car. The brake light is the most saliency region in our sight.
![fig.from https://medium.com/analytics-vidhya/visual-saliency-for-object-detection-308b188865b6](https://i.imgur.com/K9QbqND.jpg)

By the up example, we know that it is important to rapidly detect visual saliency in a cluttered visual world. However, simultaneously identifying any and all interesting targets in one's visual field has prohibitive computational complexity. One solution is to restrict complex object recognition process to a small area or a few objects at any one time.

<div style="page-break-after: always;"></div>

### 2. Visual Saliency example
Since we don't realize the human brain, it hard to conpute the visual saliency area for each people. However, we also can summarize some general case.

<br>
<br>

- #### Color
One item in the array of items strongly pops-out and effortlessly and immediately attracts attention. In this example, the red bar is very easy to be detected. Many studies have suggested that no scanning occurs in simple displays like this, it means that attention is immediately drawn to the salient item, no matter how many other distractors are present in the display. 
![fig.from fig.from http://www.scholarpedia.org/article/Visual_salience#External_Links
](https://i.imgur.com/ahSPrlA.jpg)

<br>

- #### Direction
In this display, the vertical bar is visually salient. Orientation is also the conspicuous point. We can find the different bar in the picture even throught that all bar color are same.
![fig.from http://www.scholarpedia.org/article/Visual_salience#External_Links
](https://i.imgur.com/C2usned.jpg)

<div style="page-break-after: always;"></div>

- #### Speed
In the nature, a moving thing is more dangerous than still ones, so our brain can find the different moving actions easily.
In this gif image, if it can diplay, we can easily find the point in the red circle moving faster than others.
Not only the moving speed but also the moving direction is attractive.
![fig.from http://www.scholarpedia.org/article/Visual_salience#External_Links
](https://i.imgur.com/BFkmxtG.png)

<br>

- #### Combin with shape and color
In natural environments, highly salient objects tend to automatically draw attention towards them. The perceptual salience is computed automatically, effortlessly, and in real-time. 
Designers have long relied on their salience system to create objects, such as this emergency triangle, which would also appear highly salient to others in a wide range of viewing conditions.
![fig.from http://www.scholarpedia.org/article/Visual_salience#External_Links
](https://i.imgur.com/CE1Ti7a.jpg)

<div style="page-break-after: always;"></div>

### 3. Neural and Computational Mechanisms
But how can people find the saliency? The basic principle behind computing salience is the detection of locations whose local visual attributes is significantly different from the surrounding image attributes. The attributes are along some Features or combination of Features. And we can look upon those features  as different dimansions.

This significant difference could be in a number of simple visual feature dimensions which are believed to be represented in the early stages of cortical visual processing: edge orientation、 motion direction or color.
![fig.from http://www.scholarpedia.org/article/Visual_salience#External_Links](https://i.imgur.com/LQVhck3.png)

<br>

#### Feature Maps
If something is conspicuous, it has a feature that is stronger than the feature of the surroundings.

In this picture, we can find two different feature: Intensity and Orientation. They can be seen as Intensity space and Orientation space.
When they both do the normalize, there is a obvious pick in orientation space. This pick is the attribute we want.

<br>
<br>

![fig.from http://www.scholarpedia.org/article/Visual_salience#External_Links](https://i.imgur.com/j2zS4R8.png)

<div style="page-break-after: always;"></div>

#### Top-down modulation by task demands
In computer science, the feature can be controlled by weight. In the left picture, there are many different dimensions and the dimension of orientation is significant. We can give this dimension a larger weight, so there is a significant pick in the final sum. And the pick is the saliency place which we want.

<br>

![fig.from http://www.scholarpedia.org/article/Visual_salience#External_Links](https://i.imgur.com/AUnFhDl.png)

<br>

### 4. Application
The Visual saliency can apply on many computer visual. there are some application:

<br>

#### Auto-detection
With visual salincy, we can find the military vehicles in a savanna easily.
![fig.from http://www.scholarpedia.org/article/Visual_salience](https://i.imgur.com/cKPMTAn.png)

<br>

#### Auto-driver
The recently year, It widely applys on auto-driver. the saliency can detect the edge of the road and help the driver.

<br>


![fig.from http://www.scholarpedia.org/article/Visual_salience](https://i.imgur.com/aSQmgMS.png)

<br>

#### Medical image
It also can be applied in medical images. In the below image, there is a tumor in mammograms, which is not easy to be detected without background knowledge, but the computer can detect and label it with a heat map. 
![fig.from https://www.itnonline.com/content/ai-assisted-radiologists-can-detect-more-breast-cancer-reduced-false-positive-recall](https://i.imgur.com/IgaZxK9.png)

<br>
<br>

#### Auto image cuting
Visual saliency also can crop the image. The different size of monitor is a big problem for people to watch video or image. If we use a small screen to watch a large image, since the size of image, the key area we want to watch will become small. With visual saliency, we can crop the region we want to focus on, and the scene will not become deformed.

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

![fig.from http://www.scholarpedia.org/article/Visual_salience](https://i.imgur.com/vbiqkay.png)

<br>
<br>

#### Explainable AI
Visual Saliency also can help the data science to realize the model. Convolutional neural networks (CNNs) have led to great improvement in computer vision. But what things do the model pay attention to have always bothered practitioners. We can use the saliency to find the focal point in the image.
![fig.from http://www.scholarpedia.org/article/Visual_salience#External_Links](https://i.imgur.com/QDnns3s.png)

<br>
<br>

## B. Computer Vision Method
### 1. Static Salieny

This class of saliency detection algorithms relies on image features and statistics to localize the most interesting regions of an image

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

![fig.from https://www.pyimagesearch.com/2018/07/16/opencv-saliency-detection](https://i.imgur.com/Zal5FyP.jpg)

<br>

#### Method 1. StaticSaliencySpectralResidual
The OpenCV implements two methods for static saliency detection.First method is StaticSaliencySpectralResidual, unlike the general model is to transform the saliency problem into the problem of detecting the special properties of the target, such as some color features, brightness features, texture features, etc, it focus on the background. They observe whether the background of most pictures satisfies any changes in a certain space. Finally, if the background is removed, the prominent part of the picture will be find.

<br>

#### Method 2. StaticSaliencyFineGrained
The second method to implement static saliency is StaticSaliencyFineGrained, this method calculates the significance based on the center-surround differences of spatial scale and Gaussian Pyramid.
![fig.from https://www.researchgate.net/figure/Procedure-for-computing-saliency-maps-for-videos_fig1_220744935](https://i.imgur.com/EUC2QMT.png)


First, we have an image and the basic features like color, orientation, intensity is extracted from the image. And then, these processed images will use to create Gaussian pyramids to create features Map. The Gaussian pyramid imitates the different scales of the image. For an image, if you observe the image at a close distance, the image you can see should be clearer and larger, but if you observe it at a long distance, the image you can see is blurry and small. The first one can observe more details of the image, but the second one can only see some outline information of the image. This is the scale of the image. So the Gaussian Pyramid is used to extract the scale of the image.
After doing Gaussian Pyramid, we will use center-surround difference to create features Map. And the saliency map is created by taking the mean of all the feature maps.

<br>

#### Image Pyramid
The image pyramid refers to repeated smoothing and subsampling of the image, each time the image obtained is a new image width and height are 1/2 of the original image. This is repeated continuously, and a set of images is obtained. Like, combined together, it looks like a pyramid shape, so it is called an image pyramid. After the image pyramid is obtained, we can performe image scale-space representation and multiresolution analysis.

Commonly used image pyramids include Gaussian Pyramid and Laplacian Pyramid.

<br>

### 2. Motion Saliency
This class of detection algorithms they use in motion saliency is typically rely on video or frame-by-frame inputs, the base idea of this is based on the motion background subtraction method to achieve saliency area detection, and the motion saliency algorithms process the frames, keeping track of objects which is moving.The object that move is considered salient.

<br>
<br>
<br>
<br>

![fig.from https://www.youtube.com/watch?v=tBaKusWUp3s](https://i.imgur.com/hw8FYjC.gif)

Because that computing saliency is not object detection, so the saliency detection algorithm has no idea if there is a particular object in an image or not.

Instead, the saliency detector is simply reporting where it thinks an object may lie in the image and this is up to your actual object detection or classification algorithm.

<br>
<br>

## C. Artificial Intelligent Method
### Deconvlution Network
Deconvolution uses an unsupervised method to find a set of kernels and feature maps, and let them reconstruct the image.
![fig.from http://www.scholarpedia.org/article/Visual_salience#External_Links](https://i.imgur.com/xE6HLym.png)

<br>

#### Method 1. Guided Backpropagation
The Guided Backpropagation is one of the deconvolution methods,the steps of the guided backpropagation approach using a deconvoluntion net is like this. First, we have a high-level feature map and The deconvoluntion net inverts the data flow to the CNN. This data inversion flows from the current neuron activations to the image that we input.After the above step, only a single neuron is non-zero in the high level feature map.And the final reconstructed image shows the parts of the input image that highly activates the neuron. This is the discriminative part of the image for the neuron.
![fig.from http://www.scholarpedia.org/article/Visual_salience#External_Links](https://i.imgur.com/JFgfFBM.jpg)

Both of deconvoluntion net and guided backpropagation use feature map to back-propagate gradients, which are not sensitive to categories and have no category discrimination. The difference between the two methods is that they have different gradient processing strategies.

<br>

#### Method 2. Class Activation Mapping
The second method of deconvlution network is Class Activation Mapping (CAM).The idea of CAM is very intuitive. In the neural network classifies, we want to know the classification of the image. Each feature map generated by the last layer of the convolutional layer and then is transformed into a pixel through GAP, and this pixel contains the information of the entire feature map. Finally, the one-dimensional pixel array is multiplied by after using the weight w, the softmax knows that the category of target has the largest value, so this image will be classified.

Then, we think in reverse, the pixel array after GAP will be multiplied by the weight w. The larger value of the weight w, which means the greater the influence of the image represented by the pixel. And since w refers to the degree of importance that each feature map can be divided into target, it is better to multiply the pixels of the entire feature map by the weight w and then superimpose it, so that you can focus on different areas according to the importance of each feature map . The larger weight w corresponding to the classification, the greater the influence of the feature map; on the contrary, the feature map with the weight closer to 0 is less important.

The disadvantage of CAM is that if the last layer of neural network does not use GAP (Global Average Pooling), the model structure must be modified.
![fig.from http://www.scholarpedia.org/article/Visual_salience#External_Links
](https://i.imgur.com/tE3ohCc.png)

<br>

- **Application**
CAM can show the parts that CNN cares about when classifying cat and dog.
![fig.from http://www.scholarpedia.org/article/Visual_salience#External_Links
](https://i.imgur.com/47WQ02X.png)

<div style="page-break-after: always;"></div>


## D. Conclusion
### Saliency Map

- It is important to rapidly detect potential prey, predators, or mates in a cluttered visual world.
- However, simultaneously identifying any and all interesting targets in one's visual field has prohibitive computational complexity.
- One solution is to restrict complex object recognition process to a small area or a few objects at any one time.

<br>
<br>

### Problem
- No objective metric methods to measure whether a saliency map is correct or not because we don't have label or ground truth for the saliency map.
- We usually measure the result by the subjective feeling of what we see, and it is not precise for the actual result. 

<div style="page-break-after: always;"></div>

## E. Group Member Contributions

| Student ID | Name  | Conribution | (%) |
| ---------- | ----  | ----------- | --- |
| 309554012  | 施長佑 | PPT, Written Report | 20 |
| 310553041  | 管彥鳴 | Oral Presentation, Written Report|20|
| 309653008  | 楊大昕 | PPT, Written Report | 20 |
| 310554037  | 黃乾哲 | Oral Presentation, Written Report|20|
| 310554031  | 葉詠富 | PPT, Written Report | 20 |

<br>
<br>

## F. Reference
[使用 Grad-CAM 解釋卷積神經網路的分類依據](https://medium.com/%E6%89%8B%E5%AF%AB%E7%AD%86%E8%A8%98/grad-cam-introduction-d0e48eb64adb)
[Saliency Map](https://en.wikipedia.org/wiki/Saliency_map)
[Visual salience - Scholarpedia](http://www.scholarpedia.org/article/Visual_salience#External_Links)
[Saliency Introduction and Implementation](https://www.pyimagesearch.com/2018/07/16/opencv-saliency-detection/)
[OpenCV中的显著性检测（Saliency Detection）](https://zhuanlan.zhihu.com/p/115002897)
[Saliency Maps in Convolutional Neural Networks](https://debuggercafe.com/saliency-maps-in-convolutional-neural-networks/)
[GeeksforGeeks](https://www.geeksforgeeks.org/what-is-saliency-map/)
[反捲積(Deconvolution)、上採樣(UNSampling)與上池化(UnPooling)差異](https://medium.com/ai%E5%8F%8D%E6%96%97%E5%9F%8E/%E5%8F%8D%E6%8D%B2%E7%A9%8D-deconvolution-%E4%B8%8A%E6%8E%A1%E6%A8%A3-unsampling-%E8%88%87%E4%B8%8A%E6%B1%A0%E5%8C%96-unpooling-%E5%B7%AE%E7%95%B0-feee4db49a00)
[Visual Saliency for Object Detection](https://medium.com/analytics-vidhya/visual-saliency-for-object-detection-308b188865b6)
[OpenCV Saliency Detection - pyimagesearch](https://www.pyimagesearch.com/2018/07/16/opencv-saliency-detection/)
