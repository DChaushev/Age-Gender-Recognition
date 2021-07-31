# Age-Gender-Recognition

This is a project I did as my master's thesis in the university. My master's was Artificial Intelligence. During the education course
we were working mainly on NLP projects that's why I wanted to challenge myself and do something in the computer vision area. I decided
to create a system that recognises age and gender from a person's face from a real time video.

I am going to explain everything I did and the results I achieved, but I am assuming that the reader has at least basic ML/AI knowledge
and is familliar with the fundamental concepts.

## Defining the goals

As each video frame is in fact a single image, the task was separated into three subtasks:
- finding faces in images;
- age recognition from face image;
- gender recognition from face image.

I used convolutional neural networks for all three subtasks, but having in mind that the system should work in real time, the models
had to be lightweight as at the same time they should yield satisfactory results.

I did considerable research and ended up with the following constraints:
- at most 0.8 MAE for the reggresional age recognition model;
- at least 93% accuracy for the classification gender recognition model;
- at most 0.5 seconds time for one prediction so the system does not lag much.

## Finding faces in an image

I researched four approaches for this task and I narrowed down to trying two of them:

### Haar Cascade Face Detector

This was considered the state-of-the-art approach after it was presented in 2001 from Paul Viola and Michael J. Jones.
I am not going to explain how it works, you can check the paper or search online for more information, but I am going to tell
you about my experience with it.

Pros:
- almost works in real time on a CPU;
- very simple architecture;
- finds faces with different sizes.

Cons:
- does a lot of errors;
- does not find non-frontal faces.

