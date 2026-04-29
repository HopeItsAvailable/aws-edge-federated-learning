"""
Federated Averaging + Evaluation for Lambda Aggregator.

USE THIS FILE IN YOUR LAMBDA FUNCTION (numpy only — NO PyTorch).

This file provides:
  - federated_average()  — weighted FedAvg on numpy state dicts
  - lenet5_forward()     — numpy-only LeNet-5 forward pass
  - evaluate_model()     — compute accuracy and loss on test set
  - load_test_data()     — load test images from S3 tar.gz
  - save_npz() / load_npz() — .npz serialization helpers

"""

import io
import os
import json
import tarfile
import logging

import boto3
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aggregator")

# MNIST normalization constants (same as torchvision default)
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# S3 key prefixes
MODELS_PREFIX = "models/"
UPDATES_PREFIX = "updates/"
METRICS_PREFIX = "metrics/"

# S3 client (reused across warm Lambda invocations)
s3_client = boto3.client("s3", region_name="us-west-2")

# Test data cache (persists across warm invocations)
_cached_test_data = None


# ============================================================================
# FedAvg Aggregation (numpy)
# ============================================================================


def federated_average(client_updates):
    """Weighted Federated Averaging.

    Computes:
      global_weights[k] = SUM( (n_i / n_total) * client_weights_i[k] )

    Args:
        client_updates: list of (state_dict, num_samples) tuples.
            state_dict: dict of numpy arrays (keys = layer names)
            num_samples: int — how many training samples that client used

    Returns:
        dict of numpy arrays — the aggregated global model state_dict.

    Example:
        # After downloading all client .npz files:
        client_updates = [
            (load_npz(client_0_bytes), 600),
            (load_npz(client_1_bytes), 600),
            ...
        ]
        global_sd = federated_average(client_updates)
        save_npz(global_sd)  # upload to S3
    """
    if not client_updates:
        raise ValueError("No client updates to aggregate")

    total = sum(n for _, n in client_updates)
    if total == 0:
        raise ValueError("Total samples across all clients is 0")

    first = client_updates[0][0]
    result = {k: np.zeros_like(first[k], dtype=np.float64) for k in first}

    for sd, n in client_updates:
        w = n / total
        for k in result:
            result[k] += w * sd[k].astype(np.float64)

    return {k: v.astype(first[k].dtype) for k, v in result.items()}


# ============================================================================
# .npz Serialization Helpers
# ============================================================================


def save_npz(state_dict):
    """Serialize a numpy state_dict to .npz bytes.

    Args:
        state_dict: dict of numpy arrays (e.g., from federated_average())

    Returns:
        bytes — .npz content, ready for s3.put_object(Body=...)
    """
    buf = io.BytesIO()
    np.savez(buf, **state_dict)
    return buf.getvalue()


def load_npz(data):
    """Deserialize .npz bytes to a dict of numpy arrays.

    Args:
        data: bytes — raw .npz content from s3.get_object()["Body"].read()

    Returns:
        dict of numpy arrays (keys = layer names)
    """
    npz = np.load(io.BytesIO(data))
    return {k: npz[k] for k in npz.files}


# ============================================================================
# Numpy-only LeNet-5 Forward Pass (for evaluation in Lambda)
# ============================================================================


def _conv2d(x, w, b, pad=0):
    """2D convolution. x: (N,C,H,W), w: (F,C,kH,kW), b: (F,)."""
    if pad > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    N, C, H, W = x.shape
    F, _, kH, kW = w.shape
    oH, oW = H - kH + 1, W - kW + 1
    out = np.zeros((N, F, oH, oW))
    for f in range(F):
        for i in range(oH):
            for j in range(oW):
                out[:, f, i, j] = (
                    np.sum(x[:, :, i : i + kH, j : j + kW] * w[f], axis=(1, 2, 3))
                    + b[f]
                )
    return out


def _relu(x):
    return np.maximum(0, x)


def _max_pool2d(x, size=2):
    N, C, H, W = x.shape
    oH, oW = H // size, W // size
    out = np.zeros((N, C, oH, oW))
    for i in range(oH):
        for j in range(oW):
            out[:, :, i, j] = x[
                :, :, i * size : (i + 1) * size, j * size : (j + 1) * size
            ].max(axis=(2, 3))
    return out


def _linear(x, w, b):
    return x @ w.T + b


def lenet5_forward(sd, images):
    """Forward pass through LeNet-5 using numpy arrays only.

    Args:
        sd: dict of numpy arrays (model state_dict with keys:
            conv1.weight, conv1.bias, conv2.weight, conv2.bias,
            fc1.weight, fc1.bias, fc2.weight, fc2.bias,
            fc3.weight, fc3.bias)
        images: numpy array of shape (N, 1, 28, 28) — preprocessed MNIST images

    Returns:
        numpy array of shape (N, 10) — logits (unnormalized class scores)
    """
    x = images
    x = _max_pool2d(_relu(_conv2d(x, sd["conv1.weight"], sd["conv1.bias"], pad=2)), 2)
    x = _max_pool2d(_relu(_conv2d(x, sd["conv2.weight"], sd["conv2.bias"])), 2)
    x = x.reshape(x.shape[0], -1)
    x = _relu(_linear(x, sd["fc1.weight"], sd["fc1.bias"]))
    x = _relu(_linear(x, sd["fc2.weight"], sd["fc2.bias"]))
    x = _linear(x, sd["fc3.weight"], sd["fc3.bias"])
    return x


# ============================================================================
# Evaluation Helpers
# ============================================================================


def cross_entropy_loss(logits, labels):
    """Compute cross-entropy loss (numpy).

    Args:
        logits: (N, C) float array — raw model output
        labels: (N,) int array — ground truth class indices

    Returns:
        float — mean cross-entropy loss
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(len(labels)), labels].mean())


def transform_image(img):
    """Convert a PIL image to normalized numpy array.

    Args:
        img: PIL Image

    Returns:
        numpy array of shape (1, 28, 28) — normalized grayscale image
    """
    img = img.convert("L").resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0
    return ((arr - MNIST_MEAN) / MNIST_STD).reshape(1, 28, 28)


def load_test_data(global_bucket):
    """Download test set from S3 and return as numpy arrays.

    Caches in memory across warm Lambda invocations.
    Test data (labels.csv, archives/test.tar.gz) is in global-bucket.

    Args:
        global_bucket: global-bucket name (e.g., "{ASU_ID}-global-bucket")

    Returns:
        (images, labels) tuple:
            images: numpy array (N, 1, 28, 28) float32
            labels: numpy array (N,) int64
    """
    global _cached_test_data
    if _cached_test_data is not None:
        return _cached_test_data

    logger.info("Loading test set from S3 (one-time cache) ...")

    # Labels
    resp = s3_client.get_object(Bucket=global_bucket, Key="labels.csv")
    content = resp["Body"].read().decode()
    labels_map = {}
    for line in content.strip().split("\n")[1:]:
        parts = line.strip().split(",")
        labels_map[parts[0]] = int(parts[2])

    # Test images
    resp = s3_client.get_object(Bucket=global_bucket, Key="archives/test.tar.gz")
    tar_bytes = resp["Body"].read()

    images = []
    targets = []
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".png"):
                continue
            filename = os.path.basename(member.name)
            if filename not in labels_map:
                continue
            f = tar.extractfile(member)
            img = Image.open(io.BytesIO(f.read()))
            images.append(transform_image(img))
            targets.append(labels_map[filename])

    images_np = np.concatenate(images, axis=0).reshape(len(images), 1, 28, 28)
    labels_np = np.array(targets, dtype=np.int64)
    _cached_test_data = (images_np, labels_np)
    logger.info(f"Test set cached: {len(images)} images")
    return _cached_test_data


def evaluate_model(sd, test_images, test_labels):
    """Evaluate a model state_dict on the test set.

    Args:
        sd: dict of numpy arrays (model state_dict)
        test_images: numpy array (N, 1, 28, 28)
        test_labels: numpy array (N,) int64

    Returns:
        dict with keys:
            "accuracy": float (0.0-1.0)
            "loss": float (cross-entropy)
            "total": int (number of test samples)
            "correct": int (number correct)

    Example:
        images, labels = load_test_data(global_bucket)
        result = evaluate_model(global_sd, images, labels)
        # result["accuracy"] → 0.9729
        # result["loss"] → 0.0862
    """
    logits = lenet5_forward(sd, test_images)
    preds = logits.argmax(axis=1)
    acc = float((preds == test_labels).mean())
    loss = cross_entropy_loss(logits, test_labels)
    return {
        "accuracy": acc,
        "loss": loss,
        "total": len(test_labels),
        "correct": int((preds == test_labels).sum()),
    }


# ============================================================================
# TODO: Implement your Lambda handler below
# ============================================================================


def lambda_handler(event, context):
    """Lambda handler — triggered by S3 event on updates/*.npz.

    This function is invoked each time a worker uploads a .npz model
    file to the local-bucket. You need to:

    1. Parse the S3 event to get bucket name and object key
    2. Extract round_id from the key
    3. List all .npz files for this round to check if all clients reported
    4. If not all clients → return early
    5. If all clients reported → aggregate:
       a. Download each client's .npz model weights
       b. Call federated_average() to get the aggregated model
          (use equal weighting: 1 per client)
       c. Upload aggregated model to global-bucket
       d. Evaluate on test set
       e. Write metrics/round_{R}.json to global-bucket

    Args:
        event: S3 event dict (see S3 EVENT FORMAT in docstring above)
        context: Lambda context object (not used)

    Returns:
        dict with statusCode and body
    """
    # TODO: Implement your aggregator logic here
    # event json format here https://docs.aws.amazon.com/AmazonS3/latest/userguide/notification-content-structure.html
    # Records [
    #   {
    #    "s3":{
    #        "bucket": {
    #           "name": "string"
    #        },
    #        "object": {
    #           "key": "string"
    #        }
    #    }  
    # ]

    # 1. Parse the S3 event to get bucket name and object key
    local_bucket_name = event["Records"][0]["s3"]["bucket"]["name"]
    file_key = event["Records"][0]["s3"]["object"]["key"]

    #split bucket name by -
    #eg 12236837873-global-bucket → ["12236837873", "global", "bucket"]
    bucket_parts = local_bucket_name.split("-")
    ASU_ID = bucket_parts[0]
    global_bucket_name = f"{ASU_ID}-global-bucket"
    
    #2. Extract round_id from the key
    #eg "updates/local_model_round_0_worker_0.npz" -> ["updates/local", "model", "round", "0", "worker", "0.npz"]
    current_round = file_key.split("_")[3] 
    next_round = int(current_round) + 1
    
    #3. List all .npz files for this round to check if all clients reported
    #s3 = boto3.client('s3')
    response = s3_client.list_objects_v2(Bucket=local_bucket_name, Prefix=f'updates/local_model_round_{current_round}_') 
    
    numOfFiles = 0
    if 'Contents' in response:
        for obj in response['Contents']:
            #count how many files for this round
            numOfFiles += 1

    #4. If not all clients → return early. We have 10 workers, so we need 10 files
    if numOfFiles < 10:
        print("not all done")
        return {"statusCode": 200, "body": "not all done"}
    
    metrics_key = f"metrics/round_{current_round}.json"
    try:
        s3_client.head_object(Bucket=global_bucket_name, Key=metrics_key)
        print(f"skipping {current_round} .")
        return {"statusCode": 200, "body": "already done"}
    except Exception:
        pass #continue since metrics file doesnt exist alreadu
    
    #5. If all clients reported → aggregate:
    #a. Download each client's .npz model weights
    all_client_updates = []
    for file in response['Contents']: #response['Contents'] are the 10 local_model_round_0_worker_0.npz files
        file_key = file['Key']
        s3Response = s3_client.get_object(Bucket=local_bucket_name, Key=file_key)
        
        raw_data = s3Response['Body'].read()
        state_dict = load_npz(raw_data) #returns a dict of numpy arrays
        
        all_client_updates.append((state_dict, 1)) #1 is the weight for each worker since we are doing equal weighting
    
    #5b. Call federated_average() to get the aggregated model (use equal weighting: 1 per client)
    global_sd = federated_average(all_client_updates)
    global_raw_data = save_npz(global_sd)  # upload to S3
    
    #5c. Upload aggregated model to global-bucket
    global_model_key = f"models/global_model_round_{next_round}.npz"
    s3_client.put_object(Bucket=global_bucket_name, Key=global_model_key, Body=global_raw_data)
    
    #5d. Evaluate on test set
    test_images, test_labels = load_test_data(global_bucket_name)
    test_results = evaluate_model(global_sd, test_images, test_labels)
    
    test_results["round"] = int(current_round) #was failing bc it didnt have this
    
    print(json.dumps(test_results))
    
    #5e. Write metrics/round_{R}.json to global-bucket
    metrics_key = f"metrics/round_{current_round}.json" #was doing next, should be current
    s3_client.put_object(Bucket=global_bucket_name, Key=metrics_key, Body=json.dumps(test_results))
    
    #now we need to send message to mqqt topic to tell workers to start next round
    #reference: https://github.com/aws/aws-iot-device-sdk-python-v2/blob/7514a3bd6d7c432811b1b1131486cce533c00ece/samples/mqtt/mqtt5_x509.py#L53 
    #this: https://docs.aws.amazon.com/boto3/latest/reference/services/iot-data/client/publish.html
    
    iot_client = boto3.client('iot-data', region_name='us-west-2')
    
    #get round number here instead of recipe so we have universal recipe file
    payload_round_number = json.dumps({"round_number": next_round})
    
    try:
        iot_client.publish(
            topic=f'fl/1223683773/next-round',
            qos=1,
            payload=payload_round_number
        )
    except Exception as e:
        print(f"error in lambda ag:{e}")    
    
    return {"statusCode": 200, "body": "done"}    

# a. Download each client's .npz model weights
#        b. Call federated_average() to get the aggregated model
#           (use equal weighting: 1 per client)
#        c. Upload aggregated model to global-bucket
#        d. Evaluate on test set
#        e. Write metrics/round_{R}.json to global-bucket