# Implementation of 2D Ordinary Kriging algorithm and calculate Reference Evapotranspiration(ET0) on IPU using TensorFlow
## Overview
1.This project is based on [PyKrige](https://github.com/GeoStat-Framework/PyKrige), we have implemented the `example/00_ordinary.py` example by using TensorFlow on IPU. You can achieve almost the same result.  
2.ET0 is a common method for calculating meteorological elements, the calculation process of various ET0 is roughly the same, and kriging algorithm used is one of the more famous interpolation algorithms.  
3.We do interpolation analysis of weather elements such as temperature, humidity, pressure and wind speed, and use Penman equation to calculate the ET0, which can get the law of water transpiration in the area. It's of great value for the research of meteorological data mining, evaluation of numerical weather forecasting, and climate change, etc.  
4.The input form of ET0 is discrete points, the form of each point is [longitude, latitude, pressure/wind speed/temperature], and there are more than 2000 such points involved in the calculation every day.  
5.There are many models involved in kriging algorithm, we use linear semivariogram function.  
5.Our project is based on the open source kriging algorithm in `1` and the code provided by Agilor.  
# Environment
This has been tested with the following dependencies:
- Poplar SDK version 2.2
- Ubuntu 18.04
## Quick start guide
Prepare the environment  
**1) Download the Poplar SDK**  

Install the Poplar SDK following the instructions in the [Getting Started guide](https://docs.graphcore.ai/projects/ipu-pod-getting-started/en/latest/installation.html) for your IPU system. Make sure to source the enable.sh scripts for poplar.  

**2) Python**

Create a virtualenv and install the required packages:
```
bash
virtualenv venv -p python3.6
source venv/bin/activate
pip install <path to the tensorflow-1 wheel file from the Poplar SDK>
pip install -r requirements.txt
```
# Run script
Start the program by running `bash run.sh`, we use 1 day’s data to repeat 14 times to simulate 14 days of ET0 calculation process with 4 IPU, this will take 10 minutes.  
# License
The code presented here is licensed under the MIT License.
`kriging_test.py` refers to `Pykrige`  
`kriging_test.py` is licensed under BSD 3-Clause: https://github.com/GeoStat-Framework/PyKrige/blob/main/LICENSE  