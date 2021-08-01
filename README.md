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

![Haar Cascade Face Detector Experiments](/images/haar-cascade-experiments.png?raw=true "Optional Title")

### Neural network based approach

Then I tried face detection using [Singe-Shot-Multibox detector][2], which is using shallower version of the ResNet architecture - 
ResNet 10. The model is trained over images from internet. [OpenCV][3] provides the model and the trained weights.

This is the model I decided to use in the application:

- the most accurate from all researched approaches;
- works in real time on a CPU;
- works with different face orientations;
- finds faces with different sizes.

Looked supperb to Haar Cascade:

![Single-Shot-Multibox Detector Experiments](/images/ssm-detector-experiments.png?raw=true "Optional Title")

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

## References

<p>[1]: https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf (P. Viola and M. Jones, "Rapid Object Detection using a Boosted Cascade of Simple Features", Conference on Computer Vision and Pattern Recognition, 2001.)
<p>[2]: https://arxiv.org/abs/1512.02325 (W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu и A. Berg, SSD: Single Shot MultiBox Detector, 2016.)
<p>[3]: https://opencv.org/ (OpenCV)
<p>[4]: https://www.learnopencv.com/histogram-of-oriented-gradients (S. Mallick, „Histogram of Oriented Gradients,“ 2016.)
<p>[5]: https://arxiv.org/abs/1502.00046 (D. King, Max-Margin Object Detection, 2015.)
<p>[6]: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki (IMDB-WIKI – 500k+ face images with age and gender labels, 2015.)
  
[1]: https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf (P. Viola and M. Jones, "Rapid Object Detection using a Boosted Cascade of Simple Features", Conference on Computer Vision and Pattern Recognition, 2001.)
[2]: https://arxiv.org/abs/1512.02325 (W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu и A. Berg, SSD: Single Shot MultiBox Detector, 2016.)
[3]: https://opencv.org/ (OpenCV)
[4]: https://www.learnopencv.com/histogram-of-oriented-gradients (S. Mallick, „Histogram of Oriented Gradients,“ 2016.)
[5]: https://arxiv.org/abs/1502.00046 (D. King, Max-Margin Object Detection, 2015.)
[6]: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki (IMDB-WIKI – 500k+ face images with age and gender labels, 2015.)
