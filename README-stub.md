# Graphcore Portfolio Examples Stub

Table of contents
=================
<!--ts-->
   * [Overview](#overview)
   * [File Structure](#filestructure)
   * [Changelog](#changelog)
   * [Model Description](#modeldescription)
   * [Quick start guide](#gettingstarted)
   * [Dataset](#dataset)
      * [Source](#datasource)
      * [Data Preprocessing](#datapreproc)
   * [Task N](#task)
   * [Licensing](#licensing)

<!--te-->

## Overview

This readme provides a sample to get started merging your application to the Portfolio-Examples 

## Dataset

This will explain your dataset, the source of it or how to produce it through your scripts. Minimum requirement would be to have sample dummy dataset to test functionality of your code. For customer or internal data, seek approval from data owners and follow standard data processes.


### Data Source
Where you got the data if it were a public dataset, or how you captured it.

### Data Preprocessing

## File Structure

| File                         | Description                                |
| ---------------------------- | ------------------------------------------ |
| `README.md`                  | How to run the model                       |
| `ABC_IPU.py`                 | Main algorithm script to run IPU model     |
| `argparser.py`               | Document and read command line parameters  |
| `requirements.txt`           | Required Python 3.6 packages               |
| `test_ABC.py`                | Test script. Run using `python -m pytest`  |

## Changelog
See https://keepachangelog.com for good practices

## Model Description
Brief introduction to model, tasks & implementation details, links to source papers; 2-3 paragraphs maximum 
Details of implementation capabilities & supported configurations (Training, fine tuning, Inference etc.)
This should include a description of the execution schemes & naming conventions the application uses.
Link to Graphcore Terminology

## Quick Start Guide
### Requirements 
List the requirements here such as:
- source code
- SDK and version where this was tested (!Very Important!)
- OS this developed and tested on
- libraries and packages
- data

### 1) Install Poplar SDK

Install Poplar SDK 1.2 following the instructions in the 
Getting Started guide for your IPU system 
which can be found here: 
https://docs.graphcore.ai/projects/ipu-hw-getting-started/en/1.2.0/.
Make sure to source the `enable.sh` script for Poplar as well as the drivers.

### 2) Prepare Project Environment
```
virtualenv venv -p python3.6
source venv/bin/activate
pip install <path to gc_tensorflow_2.1.whl>
pip install -r requirements.txt
```

### 3) Prepare Dataset

To prep the data, you can run for example (to generate dummy data, or perform data wrangling)

```
python data_prep.py --samples-output data_out.txt
```

### 4) Execution

To execute the application, you can run for example

```
python ABC_IPU.py --enqueue-chunk-size 10000 --tolerance 5e5 --n-samples-target 100 --n-samples-per-batch 100000 --country US --samples-filepath US_5e5_100.txt
```

All command line options and defaults are explained in `argparser.py`.

## Dataset

Complete instructions for sourcing data and generating a suitable dataset to reproduce our convergence experiments. 
Ideally this would be scripted if the process is complex. 
May point to a dedicated README file in application subfolder for complex data prep cases e.g. Wikipedia

## Training

Describe Training task, show code snippets and what results to expect.

## Validation

Describe Validation task, show code snippets and what results to expect.

## Deployment
When applicable, show how to deploy or point to a sepaprate README or scripts (e.g. docker), show code snippets and what results to expect.
## Inference

Describe Inference task, show code snippets and what results to expect.

## Task N

(Optional) Extra sections
- Additional variations on configurations with no performance claims
- Creating custom model configurations 
- Running tests
- Describe other relevant tasks

## Licensing
Put application license type and references to third party code/datasets used, see [guide](https://graphcore.sharepoint.com/:p:/s/Applications/EXw_vaZtpbJDsRLCGnNLjFAB3WqsTDHjY-wIyZGPkVV0Yw?e=IGKjKz) for more information
