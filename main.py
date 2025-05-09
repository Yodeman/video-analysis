import argparse
import tarfile
import pathlib
import pprint

import numpy as np
import pytorchvideo.data
import torch
import evaluate
from huggingface_hub import hf_hub_download
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)
from transformers import TrainingArguments, Trainer

# dataset config
hf_dataset_id = "JackWong0911/kinetic-400_450samples"
filename = "kinetics-400_450samples_ver2.tar"
dataset_root_path = pathlib.Path("./kinetics-400_450samples").resolve()
model_ckpt = "MCG-NJU/videomae-base"
model_name = model_ckpt.split("/")[-1]
new_model_name = f"{model_name}-finetuned-kinetic-400-sample"
all_video_file_paths = None
clip_duration = None
video_train = None
video_val = None
video_test = None

# training config
num_epochs = 4
batch_size = 8
sample_rate = 4
fps = 30
metric = evaluate.load("accuracy")


def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments with mode ('train' or 'test').
    """
    parser = argparse.ArgumentParser(description="Recoginize action in video clip...")
    parser.add_argument(
        "--mode",
        type=str,
        help="train model or just perform inference",
        choices=["train", "test"],
        default="train",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="local checkpoint path",
        default="",
    )
    return parser.parse_args()


def load_dataset():
    """
    Load and extract the Kinetics-400 dataset.

    Downloads the dataset if not already present and extracts video files.
    Creates class labels and mappings between class names and IDs.

    Returns:
        tuple: (class_labels, label2id, id2label) - Lists and dictionaries mapping
              between class names and their numeric IDs.
    """
    global video_train, video_val, video_test

    if not dataset_root_path.exists():
        file_path = hf_hub_download(
            repo_id=hf_dataset_id, filename=filename, repo_type="dataset"
        )

        with tarfile.open(file_path) as t:
            t.extractall(".")

    # verify extracted files
    video_train = dataset_root_path.glob("train/*/*.mp4")
    video_val = dataset_root_path.glob("val/*/*.mp4")
    video_test = dataset_root_path.glob("test/*/*.mp4")
    all_video_file_paths = list(video_train) + list(video_val) + list(video_test)
    print(f"Total videos: {len(all_video_file_paths)}")
    pprint.pprint(all_video_file_paths[:5])

    # generate class labels and ids
    class_labels = sorted({str(path).split("/")[-2] for path in all_video_file_paths})
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}
    print(f"Unique classes: {list(label2id.keys())}.")

    return class_labels, label2id, id2label


def load_model(label2id, id2label):
    """
    Load the VideoMAE model and image processor.

    Args:
        label2id (dict): Mapping from class names to class IDs.
        id2label (dict): Mapping from class IDs to class names.

    Returns:
        tuple: (image_processor, model) - The VideoMAE image processor and model.
    """
    image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
    model = VideoMAEForVideoClassification.from_pretrained(
        model_ckpt,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    return image_processor, model


def prepare_dataset(image_processor, model):
    """
    Prepare training, validation, and test datasets with appropriate transforms.

    Args:
        image_processor (VideoMAEImageProcessor): Processor for video frames.
        model (VideoMAEForVideoClassification): The VideoMAE model.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset) - PyTorchVideo Kinetics datasets.
    """
    mean = image_processor.image_mean
    std = image_processor.image_std

    if "shortest_edge" in image_processor.size:
        height = width = image_processor.size["shortest_edge"]
    else:
        height = image_processor.size["height"]
        width = image_processor.size["width"]
    resize_to = (height, width)

    num_frames_to_sample = model.config.num_frames
    clip_duration = int(num_frames_to_sample * sample_rate / fps)
    print(f"Frames: {num_frames_to_sample}, Clip Duration: {clip_duration}")

    train_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        RandomShortSideScale(min_size=256, max_size=320),
                        RandomCrop(resize_to),
                        RandomHorizontalFlip(p=0.5),
                    ]
                ),
            ),
        ]
    )
    val_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        Resize(resize_to),
                    ]
                ),
            ),
        ]
    )

    train_dataset = pytorchvideo.data.Kinetics(
        data_path=dataset_root_path / "train",
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
        decode_audio=False,
        transform=train_transform,
    )
    val_dataset = pytorchvideo.data.Kinetics(
        data_path=dataset_root_path / "val",
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )
    test_dataset = pytorchvideo.data.Kinetics(
        data_path=dataset_root_path / "test",
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )
    print(
        f"Train: {train_dataset.num_videos}, Validation: {val_dataset.num_videos}, Test: {test_dataset.num_videos}"
    )

    return train_dataset, val_dataset, test_dataset


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics from prediction outputs.

    Args:
        eval_pred: Evaluation prediction object containing predictions and labels.

    Returns:
        dict: Dictionary containing accuracy metric.
    """
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def collate_fn(examples):
    """
    Collate function for batching dataset examples.

    Args:
        examples (list): List of examples from the dataset.

    Returns:
        dict: Dictionary with batched pixel values and labels.
    """
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def train_model(model, train_dataset, val_dataset):
    """
    Train the video classification model.

    Args:
        model (VideoMAEForVideoClassification): The model to train.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.

    Returns:
        TrainerOutput: Results of the training process.
    """
    args = TrainingArguments(
        new_model_name,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
        push_to_hub=False,
        logging_steps=10,
    )
    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    return trainer.train()


def run_inference(model, test_dataset):
    """
    Run inference on a sample from the test dataset.

    Args:
        model (VideoMAEForVideoClassification): The trained model.
        test_dataset: Test dataset containing video samples.
    """
    sample_video = next(iter(test_dataset))
    sample_video_name = dataset_root_path / f"test/{sample_video['video_name']}"

    permuted_sample_test_video = sample_video["video"].permute(1, 0, 2, 3)
    inputs = {
        "pixel_values": permuted_sample_test_video.unsqueeze(0),
        "labels": torch.tensor([sample_video["label"]]),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    print(
        f"Based on the video analysis of `{sample_video_name}`, the entities are engaged in activities related to {model.config.id2label[predicted_class_idx]}"
    )


def main(mode, checkpoint):
    """
    Main function to coordinate model training or inference.

    Args:
        mode (str): Either 'train' to train the model or 'test' to run inference.
    """
    global model_ckpt

    if mode == "test":
        model_ckpt = pathlib.Path(f"./{new_model_name}/{checkpoint}")

    class_labels, label2id, id2label = load_dataset()
    image_processor, model = load_model(label2id, id2label)
    train_dataset, val_dataset, test_dataset = prepare_dataset(image_processor, model)

    if mode == "train":
        _ = train_model(model, train_dataset, val_dataset)

    run_inference(model, test_dataset)


if __name__ == "__main__":
    args = parse_args()
    main(args.mode, args.checkpoint)
