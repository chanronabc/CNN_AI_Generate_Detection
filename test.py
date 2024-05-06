from model.CNN import CNN
from model.SVM import SVM
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

def load_test():
    # 定义图像转换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 将图像大小调整为256x256
        transforms.ToTensor(),  # 将图像转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    # 创建 DataLoader
    test_dataset = datasets.ImageFolder(root='girls-new\\test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)
    return test_loader

def evaluate_model(model, data_loader):
    y_true = []
    y_scores = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images
            labels = labels
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]  # 假设类别1是正类
            preds = probabilities > 0.5

            y_true.extend(labels.tolist())
            y_scores.extend(probabilities.tolist())
            y_pred.extend(preds.tolist())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)
    # Confusion_matrix = confusion_matrix(y_true, y_scores)
    return auc, recall, accuracy, precision, f1

if __name__ == "__main__":
    test_loader = load_test()
    model = CNN()
    model.load_state_dict(torch.load('cnn.pth'))
    model.eval()

    # 计算性能指标
    accuracy, auc, precision, recall, f1 = evaluate_model(model, test_loader)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall:" , recall)
    print("f1: ", f1)
    print("AUC:", auc)
    # print("Confusion Matrix:", Confusion_matrix)

    model = SVM()
    model.load_state_dict(torch.load('svm.pth'))
    model.eval()

    # 计算性能指标
    accuracy, auc, precision, recall, f1 = evaluate_model(model, test_loader)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall:" , recall)
    print("f1: ", f1)
    print("AUC:", auc)
    # print("Confusion Matrix:", Confusion_matrix)