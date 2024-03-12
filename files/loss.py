import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1, beta=1, gamma=1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        #self.gamma = gamma

        self.dice_loss = DiceLoss()  
        self.bce_loss = nn.BCEWithLogitsLoss()
        #self.boundary_loss = BoundaryLoss()

    def forward(self, predicted, target):

        bce_loss = self.bce_loss(predicted, target)

        dice_loss = self.dice_loss(torch.sigmoid(predicted), target)

        #boundary_loss = self.boundary_loss(predicted, target)

        combined_loss = dice_loss * self.alpha + bce_loss * self.beta# + boundary_loss * self.gamma

        return combined_loss, [dice_loss.item(), bce_loss.item()]
    
class CombinedLossUnlog(nn.Module):
    def __init__(self, alpha=1, beta=1, gamma=1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        #self.gamma = gamma

        self.dice_loss = DiceLoss()  
        self.bce_loss = nn.BCELoss()
        #self.boundary_loss = BoundaryLoss()

    def forward(self, predicted, target):

        bce_loss = self.bce_loss(predicted, target)

        dice_loss = self.dice_loss(predicted, target)

        #boundary_loss = self.boundary_loss(predicted, target)

        combined_loss = dice_loss * self.alpha + bce_loss * self.beta# + boundary_loss * self.gamma

        return combined_loss, [dice_loss.item(), bce_loss.item()]
    
class IoULoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, predicted, target):
        intersection = torch.sum(predicted * target)
        union = torch.sum(predicted) + torch.sum(target)

        iou = (intersection + self.smooth) / (union + self.smooth)
        iou_loss = 1 - iou.mean()

        return iou_loss
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8 ,gamma=2):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, predicted, target):
        bce = F.binary_cross_entropy(predicted, target)
        p = torch.exp(-bce)
        focal_loss = -self.alpha * (1 - p).pow(self.gamma) * bce #torch.log(torch.clamp(predicted, 1e-15, 1 - 1e-15)) * target

        return focal_loss.mean()
    
'''class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, predicted, target):
        focal_loss = (1 - predicted).pow(self.gamma) * F.binary_cross_entropy(predicted, target) #torch.log(torch.clamp(predicted, 1e-15, 1 - 1e-15)) * target

        return focal_loss.mean()'''

class BoundaryLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(BoundaryLoss, self).__init__()
        self.weight = weight

    def forward(self, predicted, target):
        bce_loss = F.binary_cross_entropy(predicted, target)
        
        # Calculate gradient of predicted
        gradient_predicted_x = torch.abs(F.conv2d(predicted, torch.ones(1, 1, 3, 3).to(predicted.device), padding=1))
        gradient_predicted_y = torch.abs(F.conv2d(predicted, torch.ones(1, 1, 3, 3).to(predicted.device).transpose(2, 3), padding=1))
        gradient_predicted = gradient_predicted_x + gradient_predicted_y

        # Calculate gradient of target
        gradient_target_x = torch.abs(F.conv2d(target, torch.ones(1, 1, 3, 3).to(target.device), padding=1))
        gradient_target_y = torch.abs(F.conv2d(target, torch.ones(1, 1, 3, 3).to(target.device).transpose(2, 3), padding=1))
        gradient_target = gradient_target_x + gradient_target_y

        # Calculate boundary loss
        boundary_loss = F.binary_cross_entropy(gradient_predicted, gradient_target)

        # Combine binary cross-entropy and boundary loss
        total_loss = bce_loss + self.weight * boundary_loss

        return total_loss
        
class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, predicted, target):
        intersection = torch.sum(predicted * target)
        union = torch.sum(predicted) + torch.sum(target)

        dice_coefficient = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice_coefficient

        return loss

'''class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, probability_map, target):
        # Compute gradients of the probability map using Sobel operators
        gradient_x = F.conv2d(probability_map, torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3), padding=1)
        gradient_y = F.conv2d(probability_map, torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3), padding=1)

        # Combine gradients
        gradient_magnitude = torch.sqrt(gradient_x**2 + gradient_y**2)

        # Compute boundary loss
        boundary_loss = F.binary_cross_entropy(gradient_magnitude, target)

        return boundary_loss'''

class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()
        # Sobel kernels for gradient computation
        self.sobel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view(1, 1, 3, 3)
        self.sobel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view(1, 1, 3, 3)

    def forward(self, probability_map, target):
        # Move Sobel kernels to the same device as probability_map
        sobel_x = self.sobel_x.to(probability_map.device, dtype=probability_map.dtype)
        sobel_y = self.sobel_y.to(probability_map.device, dtype=probability_map.dtype)

        # Compute gradients of the probability map
        grad_x = F.conv2d(probability_map, sobel_x, padding=1)#.to(dtype=probability_map.dtype)
        grad_y = F.conv2d(probability_map, sobel_y, padding=1)#.to(dtype=probability_map.dtype)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)

        # Compute boundary loss
        loss = torch.mean(gradient * (1 - target))

        return loss
    
def boundary_loss(probability_map, target):
    # Sobel kernels for gradient computation
    sobel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view(1, 1, 3, 3).to(probability_map.device)
    sobel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view(1, 1, 3, 3).to(probability_map.device)

    # Compute gradients of the probability map
    grad_x = F.conv2d(probability_map, sobel_x, padding=1)
    grad_y = F.conv2d(probability_map, sobel_y, padding=1)
    gradient = torch.abs(grad_x) + torch.abs(grad_y)

    # Compute boundary loss
    loss = torch.mean(gradient * (1 - target))

    return loss
    
def precision_recall_f1(y_true, y_pred, device):
    if isinstance(y_pred, list):
        cumulative_true_positives = 0
        cumulative_false_positives = 0
        cumulative_false_negatives = 0

        # Iterate over each pair of tensors in the lists
        for y_true, y_pred in zip(y_true, y_pred):
            y_true = y_true.to(device)
            # Calculate true positives, false positives, and false negatives for the current pair
            true_positives = torch.sum((y_true == 1) & (y_pred == 1)).item()
            false_positives = torch.sum((y_true == 0) & (y_pred == 1)).item()
            false_negatives = torch.sum((y_true == 1) & (y_pred == 0)).item()

            # Update cumulative counts
            cumulative_true_positives += true_positives
            cumulative_false_positives += false_positives
            cumulative_false_negatives += false_negatives

        # Calculate overall precision, recall, and F1 score
        precision = cumulative_true_positives / max((cumulative_true_positives + cumulative_false_positives), 1)
        recall = cumulative_true_positives / max((cumulative_true_positives + cumulative_false_negatives), 1)

        f1 = 2 * (precision * recall) / max((precision + recall), 1e-8)
    else:
        true_positives = torch.sum((y_true == 1) & (y_pred == 1)).item()
        false_positives = torch.sum((y_true == 0) & (y_pred == 1)).item()
        false_negatives = torch.sum((y_true == 1) & (y_pred == 0)).item()

        precision = true_positives / max((true_positives + false_positives), 1)
        recall = true_positives / max((true_positives + false_negatives), 1)

        f1 = 2 * (precision * recall) / max((precision + recall), 1e-8)

    return precision, recall, f1

'''def precision_recall_f1(y_true_list, y_pred_list):
    # Initialize cumulative values for precision, recall, and F1 score
    cumulative_true_positives = 0
    cumulative_false_positives = 0
    cumulative_false_negatives = 0

    # Iterate over each pair of tensors in the lists
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        # Calculate true positives, false positives, and false negatives for the current pair
        true_positives = torch.sum((y_true == 1) & (y_pred == 1)).item()
        false_positives = torch.sum((y_true == 0) & (y_pred == 1)).item()
        false_negatives = torch.sum((y_true == 1) & (y_pred == 0)).item()

        # Update cumulative counts
        cumulative_true_positives += true_positives
        cumulative_false_positives += false_positives
        cumulative_false_negatives += false_negatives

    # Calculate overall precision, recall, and F1 score
    overall_precision = cumulative_true_positives / max((cumulative_true_positives + cumulative_false_positives), 1)
    overall_recall = cumulative_true_positives / max((cumulative_true_positives + cumulative_false_negatives), 1)

    overall_f1 = 2 * (overall_precision * overall_recall) / max((overall_precision + overall_recall), 1e-8)

    return overall_precision, overall_recall, overall_f1'''