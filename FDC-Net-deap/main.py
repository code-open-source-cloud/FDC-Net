import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model import JointDenoiseClassify
from data_loading import EEGDataset
from indicator import AdaptiveBCE
from train_evaluate import train , evaluate

# Equipment configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Parameter configuration
    # data_path = "/media/aaa/D6249A89249A6BED/DWJ/data_preprocessed_matlab"
    data_path = "D:\DEAP\data_preprocessed_matlab"
    batch_size = 32
    epochs = 100
    lr = 0.001

    snr_values = [-3, -2, -1, 0, 1, 2, 3]
    results = []

    plt.figure(figsize=(12, 8))

    for snr in snr_values:
        print(f"\n{'=' * 50}")
        print(f"Starting training with SNR = {snr} dB")
        print(f"{'=' * 50}")

        # 准备数据
        dataset = EEGDataset(data_path, snr=snr)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)

        # 初始化模型
        model = JointDenoiseClassify(num_classes=2).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        # 损失函数
        denoise_criterion = nn.MSELoss()
        classify_criterion = AdaptiveBCE().to(device)

        # 用于存储最后10个epoch的指标
        last_10_train_loss = []
        last_10_val_loss = []
        last_10_val_acc = []
        last_10_snr = []
        last_10_cc = []
        last_10_mse = []

        best_loss = float('inf')
        for epoch in range(epochs):
            train_loss = train(model, train_loader, optimizer,
                               denoise_criterion, classify_criterion, epoch, epochs)
            val_loss, val_acc, val_snr, val_cc, val_mse = evaluate(model, val_loader,
                                                                   denoise_criterion, classify_criterion, epoch, epochs,
                                                                   snr)

            # 保存最佳模型 (现在比较的是val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), f"ablation_model_snr_{snr}.pth")
                print(f"Saved new best ablation model for SNR={snr} with val_loss: {best_loss:.4f}")

            # 记录最后10个epoch的指标
            if epoch >= epochs - 10:
                last_10_train_loss.append(train_loss)
                last_10_val_loss.append(val_loss)
                last_10_val_acc.append(val_acc)
                last_10_snr.append(val_snr)
                last_10_cc.append(val_cc)
                last_10_mse.append(val_mse)

        # 计算最后10个epoch的平均值
        avg_train_loss = np.mean(last_10_train_loss)
        avg_val_loss = np.mean(last_10_val_loss)
        avg_val_acc = np.mean(last_10_val_acc)
        avg_snr = np.mean(last_10_snr)
        avg_cc = np.mean(last_10_cc)
        avg_mse = np.mean(last_10_mse)

        results.append({
            'SNR': snr,
            'Avg_Train_Loss': avg_train_loss,
            'Avg_Val_Loss': avg_val_loss,
            'Avg_Val_Acc': avg_val_acc,
            'Avg_SNR': avg_snr,
            'Avg_CC': avg_cc,
            'Avg_MSE': avg_mse,
        })

        print(f"\nSNR {snr} dB Final Metrics:")
        print(f"Avg Train Loss: {avg_train_loss:.4f}")
        print(f"Avg Val Loss: {avg_val_loss:.4f}")
        print(f"Avg Val Accuracy: {avg_val_acc:.4f}")
        print(f"Avg Achieved SNR: {avg_snr:.2f} dB")
        print(f"Avg Correlation Coefficient: {avg_cc:.4f}")
        print(f"Avg MSE: {avg_mse:.4f}")

        # 保存结果到CSV
    df = pd.DataFrame(results)
    df.to_csv("ablation_results.csv", index=False)

if __name__ == "__main__":
    main()