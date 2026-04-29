"""
LeNet-5 Classifier for Federated Learning on MNIST.

USE THIS FILE IN YOUR WORKER (EC2 instances — PyTorch available).

This file provides:
  - LeNet5 model class (PyTorch)
  - Serialization helpers: state_dict <-> .npz bytes
  - Model creation and loading utilities

"""

import io
import json
import os
import queue
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import boto3
import gc
import awsiot.greengrasscoreipc.clientv2 as clientv2
import time
import requests
import copy
import csv
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import torchvision.transforms as transforms
from collections import OrderedDict


NUM_CLASSES = 10


class LeNet5(nn.Module):
    """LeNet-5 for MNIST classification.

    Input:  (batch, 1, 28, 28)
    Output: (batch, 10)
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_model(num_classes=NUM_CLASSES):
    """Create a fresh LeNet-5 model with random weights."""
    return LeNet5(num_classes=num_classes)


def load_model(state_dict, num_classes=NUM_CLASSES):
    """Create a LeNet-5 model and load the given state_dict.

    Args:
        state_dict: OrderedDict of PyTorch tensors (from deserialize_state_dict).
        num_classes: Number of output classes (default 10).

    Returns:
        LeNet5 model with loaded weights, ready for training or inference.
    """
    model = LeNet5(num_classes=num_classes)
    model.load_state_dict(state_dict)
    return model


def serialize_state_dict(state_dict):
    """Convert a PyTorch state_dict to .npz bytes for S3 upload.

    Args:
        state_dict: OrderedDict from model.state_dict()
                    (keys are layer names, values are torch.Tensor)

    Returns:
        bytes — the .npz archive contents, ready for s3.put_object(Body=...)

    Example:
        sd = model.state_dict()
        data = serialize_state_dict(sd)
        s3.put_object(Bucket=bucket, Key="models/global_model_round_0.npz", Body=data)
    """
    buf = io.BytesIO()
    np.savez(buf, **{k: v.cpu().numpy() for k, v in state_dict.items()})
    return buf.getvalue()


def deserialize_state_dict(data):
    """Convert .npz bytes from S3 to a PyTorch state_dict.

    Args:
        data: bytes — raw .npz file content from s3.get_object()["Body"].read()

    Returns:
        OrderedDict of torch.Tensor — ready for model.load_state_dict() or load_model()

    Example:
        resp = s3.get_object(Bucket=bucket, Key="models/global_model_round_0.npz")
        sd = deserialize_state_dict(resp["Body"].read())
        model = load_model(sd)
    """
    npz = np.load(io.BytesIO(data))
    return OrderedDict({k: torch.from_numpy(npz[k]) for k in npz.files})


# ============================================================================
# TODO: Implement your worker below
# ============================================================================


def train_local(model, dataloader, lr, epochs):
    """Train the model locally and return metrics.

    Args:
        model: LeNet5 model to train
        dataloader: PyTorch DataLoader with training data
        lr: learning rate
        epochs: number of local training epochs

    Returns:
        dict with keys:
            "train_loss": float — average training loss
            "train_accuracy": float — average training accuracy
            "num_samples": int — number of training samples
    """
    # TODO: Implement local training loop
    # model = copy.deepcopy(model)

    model.train()
    # standard training loop, optimizers and criterion
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loss = 0.0
    train_accuracy = 0.0

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        num_samples = 0

        for inputs, labels in dataloader:
            # reset gradients
            optimizer.zero_grad()
            # forward pass
            outputs = model(inputs)
            # loss
            loss = criterion(outputs, labels)
            loss.backward()
            # update
            optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

            # track for metrics
            running_loss += loss.item() * labels.size(
                0
            )  # we have to multiply by size to get total loss
            _, predicted_labels = outputs.max(1)  # index of max log-probability
            correct += (predicted_labels == labels).sum().item()
            num_samples += labels.size(0)

        train_loss = running_loss / num_samples
        train_accuracy = correct / num_samples
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

    return {
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "num_samples": num_samples,
    }

    # raise NotImplementedError("Implement local training")



def worker_main():
    """FL worker main loop.

    This function runs on each EC2 instance. You need to:

    1. Read PARTITION_ID and ASU_ID from environment variables
    2. Set up boto3 S3 client
    3. Load your MNIST partition from local disk
       (data is at /home/ubuntu/fl-client/data_cache/client-{PARTITION_ID}/)
    4. For each round (0 to NUM_ROUNDS-1):
       a. Poll S3 for global model: models/global_model_round_{R}.npz
       b. Download and deserialize the global model
       c. Train locally on your partition
       d. Upload trained model .npz to local-bucket (TRIGGERS Lambda)
          Key: updates/local_model_round_{R}_worker_{C}.npz
    5. Exit after all rounds complete
    """
    # TODO: Implement your worker logic here

    # global variables
    # ASU_ID = "1223683773"

    #1. get this instance's name (so we know the worker number)----------
    id_url = "http://169.254.169.254/latest/meta-data/instance-id"
    is_pi = False

    try: #make all stuff work for raspberry pi, if it fails, we are on pi
        print("trying to get instance id")
        instance_id = requests.get(id_url).text

        resp = requests.get(id_url, timeout=2)
        if resp.status_code == 200:
            instance_id = resp.text
            # if here, we are likely on EC2, so query ec2
            ec2 = boto3.resource("ec2", region_name="us-west-2")
            instance = ec2.Instance(instance_id)

            # get name
            instance_name = ""
            for tag in instance.tags:
                if tag["Key"] == "Name":
                    # print(tag["Value"])
                    instance_name = tag["Value"]
                    break

            parts = instance_name.split("-")  # expected format "1223683773-fl-worker-{X}"
            worker_num = parts[3]  # get X from name
            ASU_ID = parts[0]  # get ASU_ID from name
        else:
            is_pi = True
    except Exception:
        is_pi = True
        print(f"WE ARE ON PI")
    
    if is_pi:
        print("WE ARE ON PI")
        worker_num = "11"
        ASU_ID = "1223683773"    
    # print(f"worker: {worker_num} \n asuID: {ASU_ID} ")

    # 2. setup s3 client --------
    try:
        s3 = boto3.client("s3", region_name="us-west-2")
        # sqsQ = boto3.client("sqs", region_name="us-west-2")
        global_bucket = f"{ASU_ID}-global-bucket"
        local_bucket = f"{ASU_ID}-local-bucket"
    except Exception as e:
        print(f" filed to setup S3 client: {e}")
        return
    

    global_bucket = f"{ASU_ID}-global-bucket"
    local_bucket = f"{ASU_ID}-local-bucket"
    
    #set up client
    ggIPCclient = clientv2.GreengrassCoreIPCClientV2()
    mqtt_topic = f"fl/1223683773/next-round"

    # sqsResponse = sqsQ.get_queue_url(QueueName=f"fl-worker-{worker_num}-queue")
    # queue_url = sqsResponse["QueueUrl"]

    # 3. load MNIST partition from local disk -------------
    # data is at /home/ubuntu/fl-client/data_cache/client-{PARTITION_ID}/)
    data_path = f"/home/ubuntu/fl-client/data_cache/client-{worker_num}/"  # worker_num is the same as partition id
    csv_file = "/home/ubuntu/fl-client/data_cache/labels.csv"  # this file contains the labels for the MNIST data, we will need it to create a dataloader

    # readt csv and make dictionary of filename to id
    # format is: filename,name,id. name and id are the same
    label_map = {}
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            filename = row[0]
            label = int(row[2])
            label_map[filename] = label  # eg "099293.png" : 0

    #transform defintion, used below
    transform = transforms.Compose(
        [transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))] #normalization to match the original MNIST stats, trying to fix accuracy
    )
    image_tensors = []
    label_tensors = []
    
    #loop through data directory and create list of tensors and labels
    for filename in os.listdir(data_path):
        if filename.endswith(".png"):
            img_path = os.path.join(data_path, filename)
            img = Image.open(img_path)
            # convert image to tensor
            img_tensor = transform(img) #using abovde transformation def
            
            if filename in label_map:
                #eg if filename is "099293.png", label is 0
                image_tensors.append(img_tensor) 
                label_tensors.append(label_map[filename]) #eg 0
                
    #create dataloader
    local_images = torch.stack(image_tensors) 
    local_labels = torch.tensor(label_tensors, dtype=torch.long) #use long 
    
    #example of how to create dataloader from tensors
    #clients_data = partition_dataset(train_dataset, num_clients)
    #clients_loaders = [DataLoader(d, batch_size=32, shuffle=True) for d in clients_data]

    dataset = TensorDataset(local_images, local_labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
    #4. For each round (0 to NUM_ROUNDS-1): -------------
    #a. Poll S3 for global model: models/global_model_round_{R}.npz
    #   b. Download and deserialize the global model
    #   c. Train locally on your partition
    #   d. Upload trained model .npz to local-bucket (TRIGGERS Lambda)
    #      Key: updates/local_model_round_{R}_worker_{C}.npz
    
    mqttQ = queue.Queue() #for thread safe queue; subscribe to mqtt topic and put message in this queue, main thread will get and process messages
    
    def mqtt_message_received(event): #cant be global because it needs access to mqttQ
        try:
            message_str = str(event.message.payload, "utf-8")
            payload = json.loads(message_str)
            round_number = payload.get("round_number") #json payload
            #print(f"num:{worker_num} on round {round_number}")
            mqttQ.put(round_number)
        except Exception as e:
            print(f"ERROR in mqtt_message_received:{e}")
    
    #NEW PART 2 --------------- connect to mqtt topic and subscribe    
    #subscribe to topic, https://aws.github.io/aws-iot-device-sdk-python-v2/awsiot/greengrasscoreipc.html
    try:
        ggIPCclient.subscribe_to_iot_core(
            topic_name=mqtt_topic,
            qos=1,
            on_stream_event=mqtt_message_received, 
        )
    except Exception as e:
        print(f"connecting to client error:{e}")
    
    last_processed_round = -1 #to avoid processing the same round multiple times
    #main loop
    while True:
        #wait for message from mqtt topic 
        current_round = mqttQ.get() #main thread will wait until message is received 
        print(f"worker {worker_num} got message for round {current_round}")
        if current_round == 0 and last_processed_round > 0:
            print(f"autograder new run") #autograder is runing again, need this to reset worker from beginning
            last_processed_round = -1 
        
        if current_round <= last_processed_round:
            print(f"skip duplicate on round {current_round}")
            continue
            
        last_processed_round = current_round
        
        if(current_round >= 5):
            continue #skip rounds 5 and up because we are done here, works with if check before to not hang with back to back autograder runs
    
        #4b. download and deserialize global model
        global_model_name = f"models/global_model_round_{current_round}.npz"
        #print(f"got {global_model_name} here")
        
        #adding try catch because it wasnt downloading when testing
        while (True):
            try:
                s3_response = s3.get_object(Bucket=global_bucket, Key=global_model_name)
                content = s3_response['Body'].read()
                break
            except Exception as e:
                print(f"retrying after waiting {e}")
                time.sleep(5) # wait before retrying
        
        ##predefined function
        pyTorchStateDict = deserialize_state_dict(content) 
        global_model = load_model(pyTorchStateDict)
        
        #4c. train locally on your partition
        #print("here training")
        trainingOutput = train_local(global_model, dataloader, lr=0.02, epochs=5)
        #format is:
            #"train_loss": train_loss,
            #"train_accuracy": train_accuracy,
            #"num_samples": num_samples,
        
        #4d. upload trained model .npz to local-bucket (TRIGGERS Lambda)
        local_model_name = f"updates/local_model_round_{current_round}_worker_{worker_num}.npz"
        
        local_model=global_model #renaming for clarity
        toUpload = serialize_state_dict(local_model.state_dict()) #global_model is now locally trained model, so we serialize its state dict and upload that to s3
        
        s3.put_object(Bucket=local_bucket, Key=local_model_name, Body=toUpload)


        # #after uploading, delete message 
        # if ReceiptHandle is not None: #was getting error here on round 0, check if its not None
        #     sqsQ.delete_message(
        #         QueueUrl=queue_url,
        #         ReceiptHandle=ReceiptHandle
        #     )
        
        #free up memory
        del global_model, toUpload, pyTorchStateDict
        gc.collect() 
        

if __name__ == "__main__":
    worker_main()
