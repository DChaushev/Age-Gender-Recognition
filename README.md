# Age-Gender-Recognition

This is a project I did as my master's thesis in the university. My master's was Artificial Intelligence. During the education 
course we were working mainly on NLP projects that's why I wanted to challenge myself and do something in the computer vision 
area. I decided to create a system that recognises age and gender from a person's face from a real time video.

I am going to explain everything I did and the results I achieved, but I am assuming that the reader has at least basic ML/AI 
knowledge and is familliar with the fundamental concepts.

## Defining the goals

As each video frame is in fact a single image, the task was separated into three subtasks:
- finding faces in images;
- age recognition from face image;
- gender recognition from face image.

I used convolutional neural networks for all three subtasks, but having in mind that the system should work in real time, the 
models had to be lightweight as at the same time they should yield satisfactory results.

I did considerable research and ended up with the following constraints:
- at most 0.8 MAE for the reggresional age recognition model;
- at least 93% accuracy for the classification gender recognition model;
- at most 0.5 seconds time for one prediction so the system does not lag much.

## Finding faces in an image

I researched four approaches for this task and I narrowed down to trying two of them:

### Haar Cascade Face Detector

This was considered the state-of-the-art approach after it was presented in 2001 from [Paul Viola and Michael J. Jones][1].
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

Then I tried face detection using [Singe-Shot-Multibox detector][2], which is using shallower version of the ResNet architecture - 
ResNet 10. The model is trained over images from internet. [OpenCV][3] provides the model and the trained weights.

This is the model I decided to use in the application:

- the most accurate from all researched approaches;
- works in real time on a CPU;
- works with different face orientations;
- finds faces with different sizes.

Looked supperb to Haar Cascade:

![Single-Shot-Multibox Detector Experiments.](/images/ssm-detector-experiments.png?raw=true "Single-Shot-Multibox Detector Experiments.")

The other two approaches I researched were [HoG (Historgram of Oriented Gradients)][4] and [Maximum-Margin Object Detector][5].

## Selecting and preparing the training set

Training the models for age and gender recognition requires images dataset labeled with the age and the gender of the person on the image. The biggest publicly available such dataset was [IMDB-WIKI][6]. As the whole dataset is auto generated, huge part of
the information is wrong or incomplete, so the preprocessing of the data included:
- removing images without gender;
- removing images without a face in it;
- removing images with more than one face;
- removing images where the confidence of face existing is less than 3;
- after the calculation of the ages, I removed images where the age is less than 1 or greater than 100.

Most of the mentioned preprocessing was done using the available information from the dataset, as **gender**, **face_score**,
**second_face_score**, etc.

The original dataset contains around 523 051 images, but after this preprocessing they got reduced to around 110 000.

The second preprocessing step was to crop the images so only the faces are left, because the people in the original images are
often photographed from the weist up or even in full-length.
This step has a lot of advantages:
- the dataset size is reduced;
- we can use smaller neural networks, as the network wouldn't have to learn which part of the image is the face;
- the images that we are going to predict on are of cropped faces.

It was only logical to use the same face recognition algorithm that was going to be used in the final application.
Here are some examples of the cropping:
![Before and after cropping examples.](/images/cropped-images.png?raw=true "Before and after cropping examples.")

There were only two photos in which the algorithm was unable to find faces and I decided to not include them into the final
dataset:
![The "face-less" images.](/images/no-face-images.png?raw=true "The "face-less" images.")

As can be seen in the dataset distribution chart, most of the people are between 20 and 50 years old. This is probably due to the
fact that the photos are mainly of famous actors.
![IMDB-WIKI distribution.](/images/dataset_dist.png?raw=true "IMDB-WIKI distribution.")

That's why I decided to include and another dataset - the [UKTFace][7]. It contains of 20 000 images of people between 1 and 116 years old. For convinience the authors provide version with cropped faces.

This is age histogram of the combined datasets (the preprocessed IMDB-WIKI and UKTFace):
![Combined datasets distribution.](/images/IMDB-WIKI_UKTFace_dist.png?raw=true "Combined datasets distribution.")

As can be seen, still the majority of people is between 20 and 45 years old. To get more even distribution I decided to remove
part of the examples. The average number of images per age was 1393:

![Count per age.](/images/count_per_age.png?raw=true "Count per age.")

I trimmed the examples between 20 and 50 to a maximum of 1500 examples per age. Assuming that the larger by memory images will be
with better quaility and will contribute more, I sorted those images by file size and got the 1500 biggest for each of those 30 ages.

Here is the histogram of the final dataset used for the age recognition task:
![Trimmed histogram.](/images/trimmed_hist.png?raw=true "Trimmed histogram.")

For the gender recognition I used the dataset before the trimming, as the examples were almost evenly distributed.

## Age recognition from face image


## References

<p>[1]: https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf (P. Viola and M. Jones, "Rapid Object Detection using a Boosted Cascade of Simple Features", Conference on Computer Vision and Pattern Recognition, 2001.)
<p>[2]: https://arxiv.org/abs/1512.02325 (W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu и A. Berg, SSD: Single Shot MultiBox Detector, 2016.)
<p>[3]: https://opencv.org/ (OpenCV)
<p>[4]: https://www.learnopencv.com/histogram-of-oriented-gradients (S. Mallick, „Histogram of Oriented Gradients,“ 2016.)
<p>[5]: https://arxiv.org/abs/1502.00046 (D. King, Max-Margin Object Detection, 2015.)
<p>[6]: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki (IMDB-WIKI – 500k+ face images with age and gender labels, 2015.)
<p>[7]: https://susanqq.github.io/UTKFace (UKTFace Large Scale Face Dataset.)
  
[1]: https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf (P. Viola and M. Jones, "Rapid Object Detection using a Boosted Cascade of Simple Features", Conference on Computer Vision and Pattern Recognition, 2001.)
[2]: https://arxiv.org/abs/1512.02325 (W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu и A. Berg, SSD: Single Shot MultiBox Detector, 2016.)
[3]: https://opencv.org/ (OpenCV)
[4]: https://www.learnopencv.com/histogram-of-oriented-gradients (S. Mallick, „Histogram of Oriented Gradients,“ 2016.)
[5]: https://arxiv.org/abs/1502.00046 (D. King, Max-Margin Object Detection, 2015.)
[6]: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki (IMDB-WIKI – 500k+ face images with age and gender labels, 2015.)
[7]: https://susanqq.github.io/UTKFace (UKTFace Large Scale Face Dataset.)
