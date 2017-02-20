# Behavioral Cloning: Navigating a Car in a Simulator

Overview
---

In this project for the Udacity Self-Driving Car Nanodegree a deep CNN  is developed that can steer a car in a simulator provided by Udacity. The CNN drives the car autonomously around a track. The network is trained on images from a video stream that was recorded while a human was steering the car. The CNN thus clones the human driving behavior.

The steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

Data was collected by me while driving around the track 1 in simulator. I used a ps4 controller to drive around the track. I then used image augmentation to generate multiple training samples that represented driving under different driving conditions.

You can find most of the image effect (image augmentation in `dataset.py` file). The trained model was tested on two tracks, namely training track and validation track. Following two animations show the performance of our final model in both training and validation tracks.

Training | Validation
------------|---------------
![training_img](./assets/track_one.gif) | ![validation_img](./assets/track_two.gif)


## Getting started

The project includes the following files:

* `behavioral_cloning_net.py` containing the script to create, train, test, validate and save the model.
* `drive.py` for driving the car in autonomous mode
* `dataset.py` for image augmentation (resize, crop, shering .. etc)
* `model.h5` containing a trained convolution neural network 
* this README.md, [this article](https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713) and image_transformation_pipeline.ipynb for explanation.

Additionally you need to download and unpack the [Udacity self-driving car simulator](https://github.com/udacity/self-driving-car-sim) (Version 1 was used). To run the code start the simulator in `autonomous mode`, open another shell and type 

```
python drive.py model.h5
```

To train the model, first make a directory `../data/data`, drive the car in `training mode` around the track and save the data to this directory. The model is then trained by typing 
```
BehNet = BehavioralCloningNet()
BehNet.build_model()
BehNet.compile()
BehNet.train()
BehNet.save()
```
The rest of this `README.md` provides details about the model used.

You can find all the steps I used to train and save the model in `playground` notebook included in this repo.


