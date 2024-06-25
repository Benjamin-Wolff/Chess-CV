import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
from torch.utils import data
import torchvision
from torchvision import datasets, transforms, models


def train_vgg16(train_dir, test_dir):
    # Load pre-trained VGG16 model
    model = models.vgg16(weights='DEFAULT')

    # Print model summary
    # print(model)

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomVerticalFlip(0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Define data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    num_classes = len(train_dataset.classes)
    print(train_dataset.classes)
    print(test_dataset.classes)

    # Freeze convolutional layers from VGG16
    for param in model.parameters():
        param.requires_grad = False

    # Modify fully connected block
    num_features = model.classifier[6].in_features
    # model.classifier[6] = nn.Sequential(
    #     nn.Linear(num_features, 500),
    #     nn.ReLU(inplace=True),
    #     nn.Linear(500, 500),
    #     nn.ReLU(inplace=True),
    #     nn.Linear(500, 13),  # Assuming 13 classes for classification
    #     nn.Softmax(dim=1)
    # )
    model.classifier[6] = nn.Linear(num_features, num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print("Starting to train now")
    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total}%")

    # Export the model to ONNX
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, "chess_piece_classifier_vgg16.onnx", verbose=True)



def train_res_net(train_dir, test_dir):
    # Load pre-trained model
    model = models.resnet18()

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomVerticalFlip(0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Define data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    # Modify final fully connected layer
    num_classes = len(train_dataset.classes)
    print(train_dataset.classes)

    # Optional: Freeze pre-trained layers
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print("Starting to train now...")

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    print("training complete, now evaluating model...")
    # Evaluate the model on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on test set: {(correct/total) * 100:.2f}%")

    # Export the model to ONNX
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, "chess_piece_classifier_res3.onnx", verbose=True)


def train_alex_net(train_dir, test_dir):

    # Define transforms for data augmentation and normalization
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Define data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    # Load pre-trained AlexNet
    model = models.alexnet()
    num_classes = len(train_dataset.classes)
    print("Num classes:", num_classes)

    # Optional: Freeze pre-trained layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    model.classifier[6] = nn.Linear(4096, num_classes)

    # model.classifier[6].requires_grad_()


    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Now training...")

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

    print("training complete, now evaluating model...")
    # Evaluate the model on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on test set: {(correct/total) * 100:.2f}%")

    # Export the model to ONNX
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, "chess_piece_classifier_alex.onnx", verbose=True)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_dir = 'Data/output_train'
    test_dir = 'Data/output_test'
    train_vgg16(train_dir, test_dir)
    print("NOW ON TO AlexNet...")
    train_alex_net(train_dir, test_dir)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
