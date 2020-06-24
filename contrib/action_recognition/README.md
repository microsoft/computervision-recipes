# Action Recognition

This directory contains resources for building video-based action recognition systems.

Action recognition (also known as activity recognition) consists of classifying various actions from a sequence of frames:

![](./media/action_recognition2.gif "Example of action recognition")

We implemented two state-of-the-art approaches: (i) [I3D](https://arxiv.org/pdf/1705.07750.pdf) and (ii) [R(2+1)D](https://arxiv.org/abs/1711.11248). This includes example notebooks for e.g. scoring of webcam footage or fine-tuning on the [HMDB-51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) dataset. The latter can be accessed under [scenarios](../scenarios) at the root level.

We recommend to use the **R(2+1)D** model for its competitive accuracy, fast inference speed, and less dependencies on other packages. For both approaches, using our implementations, we were able to reproduce reported accuracies:

| Model | Reported in the paper | Our results |
| ------- | -------| ------- |
| R(2+1)D-34 RGB | 79.6% | 79.8% |
| I3D RGB | 74.8% | 73.7% |
| I3D Optical flow | 77.1% | 77.5% |
| I3D Two-Stream | 80.7% | 81.2% |


## Projects

| Directory |  Description |
| -------- |  ----------- |
| [i3d](i3d) | Scripts for fine-tuning a pre-trained I3D model on HMDB-51
dataset. |
