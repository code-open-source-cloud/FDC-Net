import torch
from indicator import calculate_snr,calculate_cc,AdaptiveBCE

# Equipment configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------- Training and evaluation functions-------------------
def train(model, dataloader, optimizer, denoise_crit, classify_crit, epoch, epochs):
    model.train()
    total_loss = 0
    total_denoise_loss = 0
    total_classify_loss = 0
    total_snr = 0
    correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    for i, (noisy, clean, labels) in enumerate(dataloader):
        model.feedback_memory = None
        noisy, clean, labels = noisy.to(device), clean.to(device), labels.to(device)
        denoised, logits = model(noisy)

        loss_denoise = denoise_crit(denoised, clean)
        loss_classify = classify_crit(logits, labels)
        loss = 0.6 * loss_denoise + 0.4 * loss_classify

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_denoise_loss += loss_denoise.item()
        total_classify_loss += loss_classify.item()
        total_snr += calculate_snr(clean, denoised)

        preds = torch.sigmoid(logits) > 0.5
        correct += torch.all(preds == labels, dim=1).sum().item()
        total_samples += labels.size(0)

        batch_preds = preds.cpu().numpy()
        batch_labels = labels.cpu().numpy()
        all_preds.extend(batch_preds)
        all_labels.extend(batch_labels)

    avg_loss = total_loss / len(dataloader)
    avg_denoise = total_denoise_loss / len(dataloader)
    avg_classify = total_classify_loss / len(dataloader)
    avg_snr = total_snr / len(dataloader)
    accuracy = correct / total_samples

    print(f"\nEpoch {epoch + 1}/{epochs} - Training")
    print("==================================")
    print(f"Total Loss: {avg_loss:.4f}")
    print(f"Denoise - MSE: {avg_denoise:.4f} | SNR: {avg_snr:.2f} dB")
    print(f"Classify - Loss: {avg_classify:.4f} | Accuracy: {accuracy:.4f}")
    return avg_loss


def evaluate(model, dataloader, denoise_crit, classify_crit, epoch, epochs, snr):
    model.eval()
    total_loss = 0
    total_denoise_loss = 0
    total_classify_loss = 0
    total_snr = 0
    total_cc = 0
    total_mse = 0
    correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, (noisy, clean, labels) in enumerate(dataloader):
            model.feedback_memory = None
            noisy, clean, labels = noisy.to(device), clean.to(device), labels.to(device)
            denoised, logits = model(noisy)

            loss_denoise = denoise_crit(denoised, clean)
            loss_classify = classify_crit(logits, labels)
            loss = 0.6 * loss_denoise + 0.4 * loss_classify

            total_loss += loss.item()
            total_classify_loss += loss_classify.item()
            total_snr += calculate_snr(clean, denoised)
            total_cc += calculate_cc(clean, denoised)
            total_mse += loss_denoise.item()

            preds = torch.sigmoid(logits) > 0.5
            correct += torch.all(preds == labels, dim=1).sum().item()
            total_samples += labels.size(0)

            batch_preds = preds.cpu().numpy()
            batch_labels = labels.cpu().numpy()
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)

    avg_loss = total_loss / len(dataloader)
    avg_classify = total_classify_loss / len(dataloader)
    avg_snr = total_snr / len(dataloader)
    avg_cc = total_cc / len(dataloader)
    avg_mse = total_mse / len(dataloader)
    accuracy = correct / total_samples

    # 打印验证指标
    print(f"\nEpoch {epoch + 1}/{epochs} - Validation")
    print("==================================")
    print(f"Total Loss: {avg_loss:.4f}")
    print(f"Denoise - MSE: {avg_mse:.4f} | SNR: {avg_snr:.2f} dB | CC: {avg_cc:.4f}")
    print(f"Classify - Loss: {avg_classify:.4f} | Accuracy: {accuracy:.4f}")
    return total_loss / len(dataloader), accuracy, avg_snr, avg_cc, avg_mse