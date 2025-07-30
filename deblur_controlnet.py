import torch
import torch.nn as nn
import torch.nn.functional as F
from cldm.cldm import ControlLDM
from ldm.util import instantiate_from_config
import pytorch_lightning as pl


class DeblurControlNet(ControlLDM):
    """
    专门用于图像去模糊任务的ControlNet模型
    继承ControlLDM并添加去模糊特定的功能
    """
    
    def __init__(self, 
                 control_stage_config, 
                 control_key="hint", 
                 only_mid_control=False,
                 deblur_loss_weight=1.0,
                 perceptual_loss_weight=0.1,
                 *args, **kwargs):
        super().__init__(control_stage_config, control_key, only_mid_control, *args, **kwargs)
        
        self.deblur_loss_weight = deblur_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        
        # 添加额外的去模糊损失函数
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # 可选：添加感知损失（需要预训练的VGG网络）
        self.use_perceptual_loss = perceptual_loss_weight > 0
        if self.use_perceptual_loss:
            self.perceptual_loss = self._setup_perceptual_loss()
    
    def _setup_perceptual_loss(self):
        """设置感知损失（使用预训练的VGG网络）"""
        try:
            import torchvision.models as models
            vgg = models.vgg16(pretrained=True).features[:16].eval()
            for param in vgg.parameters():
                param.requires_grad = False
            return vgg
        except:
            print("Warning: Could not load VGG for perceptual loss. Using L1 loss instead.")
            return None
    
    def compute_perceptual_loss(self, pred, target):
        """计算感知损失"""
        if self.perceptual_loss is None:
            return self.l1_loss(pred, target)
        
        # 归一化到ImageNet统计
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std
        
        pred_features = self.perceptual_loss(pred_norm)
        target_features = self.perceptual_loss(target_norm)
        
        return F.mse_loss(pred_features, target_features)
    
    def compute_deblur_loss(self, pred_images, target_images):
        """计算去模糊特定的损失"""
        # L1损失
        l1_loss = self.l1_loss(pred_images, target_images)
        
        # MSE损失
        mse_loss = self.mse_loss(pred_images, target_images)
        
        # 感知损失
        perceptual_loss = 0
        if self.use_perceptual_loss:
            perceptual_loss = self.compute_perceptual_loss(pred_images, target_images)
        
        # 梯度损失（保持边缘清晰度）
        grad_loss = self.compute_gradient_loss(pred_images, target_images)
        
        return {
            'l1_loss': l1_loss,
            'mse_loss': mse_loss,
            'perceptual_loss': perceptual_loss,
            'gradient_loss': grad_loss
        }
    
    def compute_gradient_loss(self, pred, target):
        """计算梯度损失，保持边缘清晰度"""
        def gradient(x):
            # 计算x和y方向的梯度
            grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
            grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
            return grad_x, grad_y
        
        pred_grad_x, pred_grad_y = gradient(pred)
        target_grad_x, target_grad_y = gradient(target)
        
        grad_loss = (F.mse_loss(pred_grad_x, target_grad_x) + 
                    F.mse_loss(pred_grad_y, target_grad_y)) / 2.0
        
        return grad_loss
    
    def p_losses(self, x_start, cond, t, noise=None):
        """重写损失计算函数，添加去模糊特定的损失"""
        # 调用父类的损失计算
        loss, loss_dict = super().p_losses(x_start, cond, t, noise)
        
        # 添加去模糊特定的损失
        if hasattr(self, 'current_pred_images') and hasattr(self, 'current_target_images'):
            deblur_losses = self.compute_deblur_loss(self.current_pred_images, self.current_target_images)
            
            # 计算总去模糊损失
            total_deblur_loss = (
                deblur_losses['l1_loss'] + 
                deblur_losses['mse_loss'] + 
                self.perceptual_loss_weight * deblur_losses['perceptual_loss'] +
                0.1 * deblur_losses['gradient_loss']
            )
            
            # 添加到总损失中
            loss = loss + self.deblur_loss_weight * total_deblur_loss
            
            # 更新损失字典
            loss_dict.update({
                'deblur_l1_loss': deblur_losses['l1_loss'],
                'deblur_mse_loss': deblur_losses['mse_loss'],
                'deblur_perceptual_loss': deblur_losses['perceptual_loss'],
                'deblur_gradient_loss': deblur_losses['gradient_loss'],
                'total_deblur_loss': total_deblur_loss
            })
        
        return loss, loss_dict
    
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        """重写模型应用函数，保存预测结果用于计算去模糊损失"""
        # 调用父类方法
        eps = super().apply_model(x_noisy, t, cond, *args, **kwargs)
        
        # 保存当前预测和目标图像用于损失计算
        if self.training:
            # 解码预测结果
            with torch.no_grad():
                pred_images = self.decode_first_stage(x_noisy)
                # 这里需要根据实际情况获取目标图像
                # 在训练步骤中设置
                self.current_pred_images = pred_images
        
        return eps
    
    def training_step(self, batch, batch_idx):
        """重写训练步骤，设置目标图像用于损失计算"""
        # 设置目标图像
        if 'jpg' in batch:
            self.current_target_images = batch['jpg']
        
        # 调用父类训练步骤
        loss = super().training_step(batch, batch_idx)
        
        # 清理临时变量
        if hasattr(self, 'current_pred_images'):
            delattr(self, 'current_pred_images')
        if hasattr(self, 'current_target_images'):
            delattr(self, 'current_target_images')
        
        return loss
    
    def configure_optimizers(self):
        """配置优化器，可以针对去模糊任务调整学习率"""
        lr = self.learning_rate
        
        # 获取需要优化的参数
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        
        # 使用AdamW优化器
        opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
        
        # 可选：添加学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000, eta_min=lr/100)
        
        return [opt], [scheduler]
    
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, **kwargs):
        """重写图像记录函数，添加去模糊特定的可视化"""
        log = super().log_images(batch, N, n_row, sample, ddim_steps, ddim_eta, **kwargs)
        
        # 添加去模糊特定的可视化
        if 'hint' in batch:
            log["blur_input"] = batch['hint'][:N] * 2.0 - 1.0  # 模糊输入
        if 'jpg' in batch:
            log["clear_target"] = batch['jpg'][:N]  # 清晰目标
        
        return log


class DeblurControlNetWithAttention(DeblurControlNet):
    """
    带注意力机制的图像去模糊ControlNet
    在ControlNet基础上添加额外的注意力模块
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 添加额外的注意力模块
        self.attention_module = self._build_attention_module()
    
    def _build_attention_module(self):
        """构建注意力模块"""
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
    
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        """重写模型应用，添加注意力机制"""
        # 获取控制信号
        if cond['c_concat'] is not None:
            blur_hint = torch.cat(cond['c_concat'], 1)
            
            # 计算注意力权重
            attention_weights = self.attention_module(blur_hint)
            
            # 应用注意力到控制信号
            attended_hint = blur_hint * attention_weights
            
            # 更新条件
            cond['c_concat'] = [attended_hint]
        
        # 调用父类方法
        return super().apply_model(x_noisy, t, cond, *args, **kwargs) 