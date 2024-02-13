# PollenDetection
A repoaitory to hold all Pollen Classification related files and models.

### Project outline:
- Pollen Classification (POC)
- Create Synthetic Dataset
- Video Object Detection using Yolo
- Real world Detection Pipeline

## MileStones:
- First Models are running inside training evnironment

## Classification
- Running Inference on test pictures with full experiment tracking environment (ETE) and produce metrics

### To Do:
#### Priority:
- ~~Model saving~~
- ~~Complete rework of "Going Modular" Chapter~~
- Inference function to run on "Unseen images".
- phisical train_test_split in order to use the dedicated test set for animations
- ~~Replicate TF model in Pytorch.../model.py~~
- Create more models (10 total)
- Yaml file to store all hyper and script parameters
- Frame script to launch multiple training scripts
- Hyper parameter optimization
- ~~Transfer learning, using pretrained bases~~

#### Nice to have:
- Async inference
- Cuda Nsight diagnostics
- Research Deployment methods
- Deploy model
- Make animation

### The wise words of the CoPilot:
- Parameterize more settings: You've already parameterized some settings like model_name and lr (learning rate) using args. You could also parameterize NUM_EPOCHS, the optimizer, and the loss function. This would make it easier to experiment with different settings.
- Use a configuration file: Instead of passing arguments via the command line, you could use a configuration file (like a JSON or YAML file). This would make it easier to manage and track different experiment settings.
- Logging: Instead of using print statements, consider using a logging library. This would give you more control over the output and make it easier to save logs to a file.
- Save the model: After training, you could save the model's state to a file. This would make it easier to reuse the model later without having to retrain it.
- Validation set: As discussed in a previous conversation, consider using a separate validation set for tuning hyperparameters and early stopping.
- Error handling: Add some error handling to make your script more robust. For example, you could check if the specified model name is valid before trying to get the model.

## Create Synthetic Dataset
- Colorspace segmentation on croped images
- placing alpha masked images on background
- generate bounding boxes and enough pictures for YOLO detector train


## Video Object Detection using Yolo
- Object Detection Model Training
- Object Detection Inference on image
- Animate Videos using the alpha masked images on background
- Video Processing pipeline for video inference
- Run Object Detection inference on video
- Create overlay
- Live interface and video output, while running live inference OD
- Performance Optimization
- Deployment

## Real world Detection Pipeline
- Not planned yet



    <!-- def load_model(self):
        pass

    def save_model(self):
        pass

    def train_model(self):
        pass

    def test_model(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass

    def visualize(self):
        pass

    def interpret(self):
        pass

    def explain(self):
        pass

    def deploy(self):
        pass

    def monitor(self):
        pass

    def optimize(self):
        pass

    def tune(self):
        pass

    def debug(self):
        pass

    def profile(self):
        pass

    def compare(self):
        pass -->