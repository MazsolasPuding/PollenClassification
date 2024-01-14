import os
from timeit import default_timer as timer

import torch
import torchinfo
import torchvision
from torchvision import transforms

from pathlib import Path
from sklearn.model_selection import train_test_split

from visualize import *
from pollen_dataset import *
from model import *
from train import *

if __name__ == "__main__":
    #####################
    # Environment Setup #
    #####################

    print(f"PyTorch version: {torch.__version__}\n"
        f"torchvision version: {torchvision.__version__}")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    BATCH_SIZE = 32
    NUM_WORKERS = 1  #os.cpu_count()
    NUM_EPOCHS = 10

    #####################
    #    Data Setup     #
    #####################

    data_path = Path("./data/KaggleDB")

    train_data_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor()
    ])

    test_data_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor()
    ])


    plot_transformed_images(list(data_path.glob("*.jpg")),
                            transform=train_data_transform, 
                            n=3)

    # Load the dataset
    samples = list(data_path.glob("*.jpg"))
    labels = PollenDataset._get_classes(data_path, full_range=True)
    train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=42, stratify=labels)


    train_dataset = PollenDataset(sample_set=train_samples, classes=labels, transform=train_data_transform)
    test_dataset = PollenDataset(sample_set=test_samples, classes=labels, transform=test_data_transform)
    classes = train_dataset.classes
    class_dict = train_dataset.class_to_idx
    print(class_dict)
    display_random_images(dataset=train_dataset, classes=classes, n=5)

    # Create Data Loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=BATCH_SIZE,
                                            num_workers=NUM_WORKERS,
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=BATCH_SIZE,
                                            num_workers=NUM_WORKERS,
                                            shuffle=False)

    print(train_loader, test_loader)


    # Test DataLoader, Get image and label from custom DataLoader
    img_custom, label_custom = next(iter(train_loader))
    print(f"Image shape: {img_custom.shape} -> [batch_size, color_channels, height, width]")
    print(f"Label shape: {label_custom.shape}")

    #####################
    #    Init Model     #
    #####################

    model_0 = TinyVGG(input_shape=3, # number of color channels (3 for RGB) 
                      hidden_units=10,
                      output_shape=len(train_dataset.classes))
    model_0.to(device)
    print(model_0)

    #################################
    # test inference and print info #
    #################################

    # # 1. Get a batch of images and labels from the DataLoader
    # img_batch, label_batch = next(iter(train_loader))

    # # 2. Get a single image from the batch and unsqueeze the image so its shape fits the model
    # img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
    # print(f"Single image shape: {img_single.shape}\n")

    # # 3. Perform a forward pass on a single image
    # model_0.eval()
    # with torch.inference_mode():
    #     pred = model_0(img_single.to(device))
        
    # # 4. Print out what's happening and convert model logits -> pred probs -> pred label
    # print(f"Output logits:\n{pred}\n")
    # print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
    # print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
    # print(f"Actual label:\n{label_single}")

    # # Print Complete Modell Summary
    # print(torchinfo.summary(model_0, input_size=[1, 3, 64, 64]))

    #######################
    # Start Training Loop #
    #######################

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model_0.parameters(),
                                 lr=0.001)

    start_time = timer()
    model_results = train(model=model_0,
                          train_dataloader=train_loader,
                          test_dataloader=test_loader,
                          optimizer=optimizer,
                          loss_fn=loss_fn,
                          epochs=NUM_EPOCHS,
                          device=device)
    end_time = timer()
    time_spent = end_time - start_time
    print(f"Total training time: {time_spent:.3f} seconds")
    plot_loss_curves(model_results)
