# Interactive fountain project - LASER QUANTUM

This project is based on a pose estimation model deployed on Deepstream. The pose estimation model and method are achieved by following this [git repository](https://github.com/NVIDIA-AI-IOT/deepstream_pose_estimation) and [this detailed deep-dive NVIDIA Developper blog](https://developer.nvidia.com/blog/creating-a-human-pose-estimation-application-with-deepstream-sdk/?ncid=so-link-52952-vt24&sfdcid=EM08#cid=em08_so-link_en-us).


## Prerequisites
You will need 
1. DeepStreamSDK 6.1
2. CUDA 11.6
3. TensorRT 8.4


## Getting Started:
To get started, please follow these steps.
1. Install [DeepStream](https://developer.nvidia.com/deepstream-sdk) on your platform, verify it is working by running deepstream-app.
2. Clone the repository preferably in `$DEEPSTREAM_DIR/sources/apps/sample_apps`.
3. Compile the program
 ```
  $ cd deepstream-pose-estimation/
  $ sudo make
  $ sudo ./deepstream-pose-estimation-app
```

