# Age-Gender-Recognition

***WORK IN PROGRESS (Pending redaction)***

Table of Contents
- [Intro](#intro)
- [Defining the goals](#defining-the-goals)
- [Finding faces in an image](#finding-faces-in-an-image)
  * [Haar Cascade Face Detector](#haar-cascade-face-detector)
  * [Neural network based approach](#neural-network-based-approach)
- [Selecting and preparing the training set](#selecting-and-preparing-the-training-set)
- [Age recognition from face image](#age-recognition-from-face-image)
- [Gender recognition from face image](#gender-recognition-from-face-image)
- [Used software and hardware](#used-software-and-hardware)
- [Running the application](#running-the-application)
- [Experiments and conclusions](#experiments-and-conclusions)
- [References](#references)

## Intro

This is a project I did as my master's thesis in the university. My master's was Artificial Intelligence. During the education course we were working mainly on NLP projects that's why I wanted to challenge myself and do something in the computer vision area. I decided to create a system that recognises age and gender from a person's face from a real time video.

I am going to explain everything I did and the results I achieved, but I am assuming that the reader has at least basic ML/AI 
knowledge and is familliar with the fundamental concepts.

## Defining the goals

As each video frame is in fact a single image, the task was separated into three subtasks:
- finding faces in images;
- age recognition from face image;
- gender recognition from face image.

I used convolutional neural networks for all three subtasks, but having in mind that the system should work in real time, the models had to be lightweight as at the same time they should yield satisfactory results.

I did considerable research and ended up with the following constraints:
- at most 0.8 MAE for the reggresional age recognition model;
- at least 93% accuracy for the classification gender recognition model;
- at most 0.5 seconds time for one prediction so the system does not lag much.

## Finding faces in an image

I researched four approaches for this task and I narrowed down to trying two of them:

### Haar Cascade Face Detector

This was considered the state-of-the-art approach after it was presented in 2001 from [Paul Viola and Michael J. Jones][1] \[1].
I am not going to explain how it works, you can check the paper or search online for more information, but I am going to tell
you about my experience with it.

Pros:
- almost works in real time on a CPU;
- very simple architecture;
- finds faces with different sizes.

Cons:
- does a lot of errors;
- does not find non-frontal faces.

![Haar Cascade Face Detector Experiments](/images/haar-cascade-experiments.png?raw=true "Haar Cascade Face Detector Experiments.")

### Neural network based approach

Then I tried face detection using [Singe-Shot-Multibox detector][2] \[2], which is using shallower version of the ResNet architecture - ResNet 10. The model is trained over images from internet. [OpenCV][3] \[3] provides the model and the trained weights.

This is the model I decided to use in the application:

- the most accurate from all researched approaches;
- works in real time on a CPU;
- works with different face orientations;
- finds faces with different sizes.

Looked supperb to Haar Cascade:

![Single-Shot-Multibox Detector Experiments.](/images/ssm-detector-experiments.png?raw=true "Single-Shot-Multibox Detector Experiments.")

The other two approaches I researched were [HoG (Historgram of Oriented Gradients)][4] \[4] and [Maximum-Margin Object Detector][5] \[5].

## Selecting and preparing the training set

Training the models for age and gender recognition requires images dataset labeled with the age and the gender of the person on the image. The biggest publicly available such dataset was [IMDB-WIKI][6] \[6]. As the whole dataset is auto generated, huge part of the information is wrong or incomplete, so the preprocessing of the data included:
- removing images without gender;
- removing images without a face in it;
- removing images with more than one face;
- removing images where the confidence of face existing is less than 3;
- after the calculation of the ages, I removed images where the age is less than 1 or greater than 100.

Most of the mentioned preprocessing was done using the available information from the dataset, as **gender**, **face_score**, **second_face_score**, etc.

The original dataset contains around 523 051 images, but after this preprocessing they got reduced to around 110 000.

The second preprocessing step was to crop the images so only the faces are left, because the people in the original images are often photographed from the weist up or even in full-length.
This step has a lot of advantages:
- the dataset size is reduced;
- we can use smaller neural networks, as the network wouldn't have to learn which part of the image is the face;
- the images that we are going to predict on are of cropped faces.

It was only logical to use the same face recognition algorithm that was going to be used in the final application.
Here are some examples of the cropping:
![Before and after cropping examples.](/images/cropped-images.png?raw=true "Before and after cropping examples.")

There were only two photos in which the algorithm was unable to find faces and I decided to not include them into the final
dataset:
![The "face-less" images.](/images/no-face-images.png?raw=true "The 'face-less' images.")

As can be seen in the dataset distribution chart, most of the people are between 20 and 50 years old. This is probably due to the
fact that the photos are mainly of famous actors.

<img src="/images/dataset_dist.png" width="640">

That's why I decided to include and another dataset - the [UKTFace][7] \[7]. It contains of 20 000 images of people between 1 and 116 years old. For convinience the authors provide version with cropped faces.

This is age histogram of the combined datasets (the preprocessed IMDB-WIKI and UKTFace):
![Combined datasets distribution.](/images/IMDB-WIKI_UKTFace_dist.png?raw=true "Combined datasets distribution.")

As can be seen, still the majority of people is between 20 and 45 years old. To get more even distribution I decided to remove part of the examples. The average number of images per age was 1393:

![Count per age.](/images/count_per_age.png?raw=true "Count per age.")

I trimmed the examples between 20 and 50 to a maximum of 1500 examples per age. Assuming that the larger by memory images will be with better quaility and will contribute more, I sorted those images by file size and got the 1500 biggest for each of those 30 ages.

Here is the histogram of the final dataset used for the age recognition task:
![Trimmed histogram.](/images/trimmed_hist.png?raw=true "Trimmed histogram.")

For the gender recognition I used the dataset before the trimming, as the examples were almost evenly distributed.

## Age recognition from face image

As we now have prepared our data, we can proceed with the second of the defined tasks - age recognition from face image.
I will avoid mentioning all the research and trial and error I did and I will just present the final convolutional neural network 
architecture I ended up with:

```python
    tf.keras.models.Sequential([
        Input(shape=image_shape),
        Conv2D(64, 3, activation='relu'),
        Conv2D(64, 3, activation='relu'),
        MaxPool2D(2),
        Dropout(0.3),
        Conv2D(128, 3, activation='relu'),
        Conv2D(128, 3, activation='relu'),
        MaxPool2D(2),
        Dropout(0.3),
        Conv2D(196, 3, activation='relu'),
        Dropout(0.3),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1)
    ])
```
Number of parameters - 2 366 725. Оbjective function - MAE (Mean absolute error). Optimization algorithm - [Adam][8] \[8].

The image_shape is a touple representing the image size - (60, 80, 1), meaning 60x80 and 1 for grayscaled. I decided to work with grayscaled images to reduce the number of parameters. The 60x80 size is chosen as this is standart ratio of portrait photos and after the resizing, the majority of photos ended up with similar width-height ratio.

The images are batched to 128 images per batch. The data is split to 80-10-10.
After the data trimming, the number of images is reduced to 79 124, so I used data augmentation to artifficially increase the
number of data.
The model was trained with patience of 50 epochs. The initial learning rate is 0.001, but after 12 consecutive validations without
improvement, it is reduced by multiplying with 0.1.
The model was trained for 16 hours, 49 minutes and 11 seconds on NVIDIA GeForce 760m and achieved MAE 5.86 at epoch 108:

![Age recognition model training.](/images/age-recognition-model-training.png?raw=true "Age recognition model training.")

The model evaluation acheived 5.9652 MAE.
One prediction takes around 0.03 seconds on the mentioned video processor.

## Gender recognition from face image

The last from the defined tasks is gender recognition from face image.
I used the same neural network, but as this task was approached as a classification task, the last regression layer is replaced with a softmax layer with two outputs. The used objective function is Categorical Cross-Entropy loss and the optimization algorithm is again Adam.

```python
    tf.keras.models.Sequential([
        Input(shape=image_shape),
        Conv2D(64, 3, activation='relu'),
        Conv2D(64, 3, activation='relu'),
        MaxPool2D(2),
        Dropout(0.3),
        Conv2D(128, 3, activation='relu'),
        Conv2D(128, 3, activation='relu'),
        MaxPool2D(2),
        Dropout(0.3),
        Conv2D(196, 3, activation='relu'),
        Dropout(0.3),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax')
        # Dense(1) // for age regression
    ])
```
The model was trained for 14 hours, 52 minutes and 8 seconds, but the training was manually cancelled, because in the last 3 hours the improvement was within thousandths. The model achieved accuracy of 95.92% after evaluation with the testing data. One predicion took around 0.03 seconds on the mentioned video processor.

<img src="/images/gender-recognition-model-training.png" width="614">

## Used software and hardware

The application was developed with the help of the programming language Python 3.7 and the TensorFlow 2.0 library. All data processing was done with the Pandas library.

The face recognition model was provided by the OpenCV library.

The models were trained on the following machine:<br>
**Processor**: Intel i7-4702 MQ @ 2.2 GHz<br>
**RAM**: 8GB<br>
**Video card**: NVIDIA GeForce GTX 760m with 768 CUDA cores and 2 GB GDDR5 memory.

Due to the fact that the video card is with Compute Capability 3.0, but TensorFlow after version 1.14 (I think) supports only video processors with Compute Capability at least 3.5 (due to multiple GPU support), I had to build TensorFlow from source after changing it to support Compute Capability 3.0.

p.s. 2GB of video memory is extremely insufficient.

## Running the application

The only prerequisite is having Python 3.7 installed.

- fetch *Main.py*, *models* folder and *requirements.txt* from this repo;
- run
  ```
  $ pip install -r requirements.txt
  ```
  This will install:
  * TensorFlow 2.0
  * numpy 1.16.4
  * opencv-python 4.1.0.25
  * imutils 0.5.2
  
  These are the libraries needed for the application to run. During development, libraries like *pandas*, *mathplotlib*, *scikit-learn*, *graphviz*, etc. were also used.
  
- Run *Main.py* to start the application:
  ```
  $ python ./Main.py
  ```

The workflow of the application is described on the following flow chart:
<img src="/images/flow-chart.png" width="720">

## Experiments and conclusions

![Demo.](/images/demo.gif?raw=true "Demo.")

Male model real age - 27, predicted - 26-31. Female model real age - 22, predicted - 22-24.

Predicting real age is a hard task even for humans. From the conducted experiments it was found that despite being very close, the model often makes mistakes predicing real age, but it does great job predicting apparent age.

In conclusion I can state that all defined tasks were accomplished.

## References

<p>[1]: https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf (P. Viola and M. Jones, "Rapid Object Detection using a Boosted Cascade of Simple Features", Conference on Computer Vision and Pattern Recognition, 2001.)
<p>[2]: https://arxiv.org/abs/1512.02325 (W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu и A. Berg, SSD: Single Shot MultiBox Detector, 2016.)
<p>[3]: https://opencv.org/ (OpenCV)
<p>[4]: https://www.learnopencv.com/histogram-of-oriented-gradients (S. Mallick, „Histogram of Oriented Gradients,“ 2016.)
<p>[5]: https://arxiv.org/abs/1502.00046 (D. King, Max-Margin Object Detection, 2015.)
<p>[6]: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki (IMDB-WIKI – 500k+ face images with age and gender labels, 2015.)
<p>[7]: https://susanqq.github.io/UTKFace (UKTFace Large Scale Face Dataset.)
<p>[8]: https://arxiv.org/abs/1412.6980 (D. P. Kingma и J. Ba, Adam: A Method for Stochastic Optimization, 2014)
  
[1]: https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf (P. Viola and M. Jones, "Rapid Object Detection using a Boosted Cascade of Simple Features", Conference on Computer Vision and Pattern Recognition, 2001.)
[2]: https://arxiv.org/abs/1512.02325 (W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu и A. Berg, SSD: Single Shot MultiBox Detector, 2016.)
[3]: https://opencv.org/ (OpenCV)
[4]: https://www.learnopencv.com/histogram-of-oriented-gradients (S. Mallick, „Histogram of Oriented Gradients,“ 2016.)
[5]: https://arxiv.org/abs/1502.00046 (D. King, Max-Margin Object Detection, 2015.)
[6]: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki (IMDB-WIKI – 500k+ face images with age and gender labels, 2015.)
[7]: https://susanqq.github.io/UTKFace (UKTFace Large Scale Face Dataset.)
[8]: https://arxiv.org/abs/1412.6980 (D. P. Kingma и J. Ba, Adam: A Method for Stochastic Optimization, 2014)
