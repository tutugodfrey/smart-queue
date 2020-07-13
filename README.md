# Smart queue

This project demonstrate the integrate of AI models with the Intel distribution of OpenVINO toolkit for running inference job at the edge. In particular the project uses the Person detection model available in the Open Model Zoo for monitoring queues and off-loading the queue based on users requirement. The project is particularly focused on evaluating and choosing the right hardward given different customer requirements like budget, hardware, and operational environment. 

The notebooks in the root of root of the project demonstrate the performance of different hardwares CPU, GPU, VPU and FPGA and some combination of these devices. Three different scenarios were tested namely Retail, Transportation and Manufacturing. You can find the relevant notebook at the root of the project. Also, case study document showing the customer requirements, hardware proposal, test results represented in graphs and conclusion about how the chosen hardware meets the customer requirement is found in the file `Choose the Right Hardware – Proposal Template.pdf`. The project was tested using the Intel DevCloud which provides the various hardware types for easy testing.

### Running the Notebooks

The Notebook is designed to be run in the intel DevCloud environment, which provide the various hardware and configuration to make inference required. However you can still use the detect_people.py file directly in you local machine if you do not have the DevCloud enviroment but that will involve a bit of extra configurations. You will need to complete the steps below completely setup your environment

** Download and Install OpenVINO**

[Download OpenVINO](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html)

**Source the OpenVINO enviroment:**

```source /opt/intel/openvino/bin/setupvars.sh```

**Download the Person Detection Model:**

```python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name person-detection-retail-0013```

You can now run inference on the different type of scenario videos by provided the appropriate argument to the person_detect.py file

```
python3 person_detect.py  --model ${MODEL} \
                          --device ${DEVICE} \
                          --video ${VIDEO} \
                          --queue_param ${QUEUE} \
                          --output_path ${OUTPUT}\
                          --max_people ${PEOPLE} \
```

--model: the path to the model file without the extension

--device: The hardware device of choice (e.g CPU,VPU)

--video: The input video to run inference on

--queue_param: param that represent the coordinate for the queue

--output_path: Directory to save the inference output

--max_people: Control how many people should be in a queue before redirecting people to the next queue

The queue_params for the different scenario are shown below

queue_param = [[  50,   90,  838,  794], [ 852,   74, 1430,  841]] # Transportation

queue_param = [[ 15,  180,  730,  780],[ 921,  144, 1424, 704]]  # Manufacturing

queue_param = [[ 620, 1,  915,  562], [1000,  1, 1264,  461]] # Retail

More specifically you can use the command below for each scenario. You may need to modify it according to your setup

```python3 person_detect.py --model intel/person-detection-retail-0013/FP32/person-detection-retail-0013 --device CPU --video resources/Manufacturing.mp4 --output_path output/results/manufacturing/cpu --max_people 2```

```python3 person_detect.py --model intel/person-detection-retail-0013/FP32/person-detection-retail-0013 --device CPU --video resources/Retail.mp4 --output_path output/results/retail/cpu --max_people 2```

```python3 person_detect.py --model intel/person-detection-retail-0013/FP32/person-detection-retail-0013 --device CPU --video resources/Transportation.mp4 --output_path output/results/transportation/cpu --max_people 2```

