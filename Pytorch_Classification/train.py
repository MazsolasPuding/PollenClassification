import os
from timeit import default_timer as timer

import argparse

import torch
import torchinfo
import torchvision
from torchvision import transforms

from pathlib import Path
from sklearn.model_selection import train_test_split

from utils import *
from data_setup import *
from model import *
from engine import *

def main(args: argparse.Namespace):
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

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers  #os.cpu_count()
    NUM_EPOCHS = args.num_epochs

    #####################
    #    Data Setup     #
    #####################

    data_path = Path(args.data_path)

    train_data_transform = transforms.Compose([
        transforms.RandomResizedCrop(128),  # Crop the image to random size and aspect ratio
        transforms.RandomHorizontalFlip(),  # Horizontally flip the image with probability 0.5
        transforms.RandomVerticalFlip(),  # Vertically flip the image with probability 0.5
        transforms.RandomRotation(20),  # Rotate the image by angle
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Randomly change the brightness, contrast, saturation and hue of an image
        transforms.ToTensor(),  # Convert the image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image with mean and standard deviation
    ])

    test_data_transform = transforms.Compose([
        transforms.Resize(size=(128, 128)),  # Resize the images to the same size as in training
        transforms.ToTensor(),  # Convert the images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images with the same mean and standard deviation as in training
    ])

    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    auto_transforms = weights.transforms()

    if args.plot_graphs:
        plot_transformed_images(list(data_path.glob("*.jpg")),
                            transform=train_data_transform, 
                            n=3)
        show_classe_distribution(args.data_path)
        show_scatter_plot(args.data_path)
        show_images(data_path)
        

    # Load the dataset
    samples = list(data_path.glob("*.jpg"))
    labels = PollenDataset._get_classes(data_path, full_range=True)
    train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=42, stratify=labels)


    train_dataset = PollenDataset(sample_set=train_samples, classes=labels, transform=auto_transforms)
    test_dataset = PollenDataset(sample_set=test_samples, classes=labels, transform=auto_transforms)
    classes = train_dataset.classes
    class_dict = train_dataset.class_to_idx
    if args.print_info:
        print(class_dict)
    if args.plot_graphs:
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

    #####################
    #    Init Model     #
    #####################

    model_handler = ModelHandler()
    model = model_handler.get_model(model_name=args.model,
                                    num_classes=len(train_dataset.classes))
                                    # input_shape=3,
                                    # hidden_units=10,
                                    # output_shape=len(train_dataset.classes))
    model.to(device)

    #############################
    # Test setup and print info #
    #############################

    if args.print_info:
        print_model_info(model=model, info="Both")
    if args.run_tests:
        test_data_loader(dataloader=train_loader)
        test_inference(model=model, dataloader=train_loader, device=device)

    #######################
    # Start Training Loop #
    #######################

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=args.lr)

    start_time = timer()
    model_results = train(model=model,
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
    plot_confusion_matrix(model=model, dataloader=test_loader, device=device)

    save_model(model=model,
               model_name=args.save_model_name,
               target_dir=args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pytorch Models on Pollen Dataset.")
    parser.add_argument("--data_path", type=str, default="./data/KaggleDB", help="Path to the data directory.")
    parser.add_argument("--model", type=str, default="EfficientNetB0", help="Model to train.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    # parser.add_argument("--device", type=str, default="cpu", help="Device to use.")
    parser.add_argument("--save_model", type=bool, default=False, help="Save model.")
    parser.add_argument("--save_model_path", type=str, default="./models", help="Path to save model.")
    parser.add_argument("--save_model_name", type=str, default="EfficientNetB0.pt", help="Name of model to save.")
    # parser.add_argument("print_every", type=int, default=1, help="Print every n epochs.")
    parser.add_argument("--print_info", type=bool, default=False, help="Print model info.")
    parser.add_argument("--plot_graphs", type=bool, default=False, help="Plot graphs.")
    parser.add_argument("--run_tests", type=bool, default=False, help="Run tests.")
    args = parser.parse_args()
    main(args=args)
