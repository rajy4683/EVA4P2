# Assignment-5: OpenPose based Hand Tracking and Gesture Recognition

## TOC

1. [Overview](#overview)
2. [Open Pose with MediaPipe](#open-pose-with-mediapipe)
3. [Arduino Interfacing](#arduino-interfacing)
4. [Demo](#demo)
5. [References](#references)

## Overview

In this assignment, the objective was to create a simple application using Gesture/Pose recognition and in course understand OpenPose and it's training methodology.
Using Mediapipe's [Hand Tracking Pipeline](https://ai.googleblog.com/2019/08/on-device-real-time-hand-tracking-with.html), various ASL(American Sign Language) hand gestures are used to control Arduino+WS2812B LEDs
At its core, Mediapipe uses a [TFlite Hand Landmark model](https://github.com/google/mediapipe/tree/master/mediapipe/models/hand_landmark.tflite) that can used independently.

### Schematic
![Simple Flow diagram ](https://github.com/rajy4683/EVA4P2/blob/master/S3-FaceAlignSwap/SwapResults.png)

## Open Pose with MediaPipe

For end-to-end ML Pipeline MediaPipe's Hand Tracking was used with 2 modifications:

1. Hand Gesture and movement mapping from [Ulrich Tiofack's Gist](https://gist.github.com/TheJLifeX/99cdf4823e2b7867c0e94fabc660c58b)    
    Ulrich's Hand Gesture Recognition, uses HandLandMarkSubgraph to obtain normalized Landmarks and then based on "open state" i.e whether finger tip is above MCG decides overall gesture.

2. A simple ActionTrigger node that publishes to a local MQTT channel
    This module uses the string of the selected Gesture and pushes it over an mqtt topic to a Mosquitto broker listening on localhost. 

## Arduino Interfacing

1. Arduino UNO was used to drive a 1M(144 WS2812 LED Strip)
2. Tweaked FirmataExpress with minor tweaks to handle custom Sysex codes was used. Tweaked Sketch can be found [here]()
3. [Pymata4](https://mryslab.github.io/pymata4/) was used to connect Arduino Board with WSL2. Code can be found [here]()

## Demo

Please visit [this YouTube link](https://www.youtube.com/watch?v=kYNRz9D_o0o) for a demo. Feel free to ask for any further info or provide any feedback in the comments section


## References

- EVA4 Course content
- [Mediapipe Documentation](https://google.github.io/mediapipe/solutions/pose.html)
- [Ulrich Tiofack's Gist](https://gist.github.com/TheJLifeX/99cdf4823e2b7867c0e94fabc660c58b)
- [PyMata4 documentation](https://mryslab.github.io/pymata4/) for Arduino control
