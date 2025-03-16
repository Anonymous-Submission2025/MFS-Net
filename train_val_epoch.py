import os
import torch
import numpy as np
from tqdm import tqdm
from utils.metrics import get_metrics
import cv2
from PIL import Image

# 获取当前文件的绝对路径，并确定项目的根目录
abspath = os.path.abspath(__file__)  # 获取所执行脚本的绝对路径    
proj_path = os.path.dirname(abspath)  # 获取父级路径            
proj_path = os.path.dirname(proj_path)                          
proj_path = os.path.dirname(proj_path)

class DRAW:
    def __init__(self):
        pass

    def save_pic(self, img, msk, msk_pred, name, args, threshold=0.5):
        # 定义保存图片的路径
        save_path = os.path.join(proj_path + '/My_model', 'BUSI')
        # 如果路径不存在，创建新目录
        if not os.path.exists(save_path):
            os.makedirs(save_path)    
        
        
        # 处理图像数据，确保形状和范围正确
        img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()  # 重新排列维度并转为numpy数组
        img = img / 255. if img.max() > 1.1 else img  # 标准化图像数据
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)  # 二值化真实掩膜
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)  # 二值化预测掩膜

        # 转换为uint8类型
        img = (img * 255).astype(np.uint8)
        msk_pred = (msk_pred * 255).astype(np.uint8)
        msk = (msk * 255).astype(np.uint8)

        # 保存图像、真实掩膜和预测掩膜
        image_path = os.path.join(save_path, f"{name}_img.png")
        real_path = os.path.join(save_path, f"{name}_gt.png")
        pred_path = os.path.join(save_path, f"{name}_msk.png")
        Image.fromarray(img).save(image_path)
        Image.fromarray(msk).save(real_path)
        Image.fromarray(msk_pred).save(pred_path)

    def draw_boundaries(self, img, msk, msk_pred, name, threshold=0.5):
        # 使用OpenCV绘制边界
        
        img = img.permute(1, 2, 0)  # 将图像维度调整为(H, W, C)
        img = img / 255. if img.max() > 1.1 else img  # 标准化图像数据
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)  # 二值化真实掩膜
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)  # 二值化预测掩膜

        # 转换为uint8类型
        image = (img * 255).astype(np.uint8)
        prediction_binary = (msk_pred * 255).astype(np.uint8)
        label_binary = (msk * 255).astype(np.uint8)

        # 使用Canny边缘检测
        label_edges = cv2.Canny(label_binary, 100, 200)
        prediction_edges = cv2.Canny(prediction_binary, 100, 200)

        # 将图像转换为BGR格式（OpenCV使用BGR）
        image_with_label = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_with_label[label_edges > 0] = [255, 0, 0]  # 用蓝色表示真实边界 (B, G, R)

        # 用橙色表示预测边界
        image_with_label[prediction_edges > 0] = [0, 165, 255]  # 橙色 (B, G, R)
        
        # 保存带有边界的结果图像
        image_path = os.path.join(self.save_path, f"{name}_edge.png")
        cv2.imwrite(image_path, image_with_label)

# 实例化DRAW类
draw = DRAW()

def test_epoch(val_loader, model, criterion, logger, args):
    # 测试阶段的模型评估函数
    model.eval()  # 设置模型为评估模式
    loss_list = []  # 用于存储损失值
    preds = []  # 用于存储预测值
    gts = []  # 用于存储真实值
    image = []  # 用于存储图像
    name = []  # 用于存储图像名称

    with torch.no_grad():  # 关闭梯度计算
        for data in tqdm(val_loader):  # 遍历验证数据集
            images, gt, image_name = data  # 解包数据
            images, gt = images.cuda(non_blocking=True).float(), gt.cuda(non_blocking=True).float()  # 移动到GPU
            pred = model(images)  # 获取模型预测
            
            # 保存图像、真实掩膜和预测掩膜
            draw.save_pic(img=images,
                          msk=gt.squeeze(1).cpu().detach().numpy(),
                          msk_pred=pred[0].squeeze(1).cpu().detach().numpy(),
                          name=image_name, args=args)
            
            # 计算损失
            loss = criterion(pred[0], gt)
            loss_list.append(loss.item())  # 记录损失
            image.append(images)
            name.append(image_name)
            gts.append(gt.squeeze(1).cpu().detach().numpy())
            preds.append(pred[0].squeeze(1).cpu().detach().numpy()) 

    # 计算指标
    log_info, miou = get_metrics(preds, gts)
    log_info = f'val loss={np.mean(loss_list):.4f}  {log_info}'  # 记录验证损失和指标信息
    print(log_info)
    logger.info(log_info)
    input()  # 暂停以便查看输出
    return np.mean(loss_list), miou  # 返回平均损失和miou

def val_epoch(val_loader, model, criterion, logger):
    # 验证阶段的模型评估函数
    model.eval()  # 设置模型为评估模式
    loss_list = []  # 用于存储损失值
    preds = []  # 用于存储预测值
    gts = []  # 用于存储真实值

    with torch.no_grad():  # 关闭梯度计算
        for data in tqdm(val_loader):  # 遍历验证数据集
            images, gt, image_name = data  # 解包数据
            images, gt = images.cuda().float(), gt.cuda().float()  # 移动到GPU
            pred = model(images)  # 获取模型预测
            
            # 计算损失
            loss = criterion(pred[0], gt)
            loss_list.append(loss.item())  # 记录损失
            gts.append(gt.squeeze(1).cpu().detach().numpy())
            preds.append(pred[0].squeeze(1).cpu().detach().numpy()) 

    # 计算指标
    log_info, miou = get_metrics(preds, gts)
    log_info = f'val loss={np.mean(loss_list):.4f}  {log_info}'  # 记录验证损失和指标信息
    print(log_info)
    logger.info(log_info)
    return np.mean(loss_list), miou  # 返回平均损失和miou

def train_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, steps, logger, save_cycles=5):
    # 训练阶段的模型训练函数
    model.train()  # 设置模型为训练模式
    loss_list = []  # 用于存储损失值
    for step, data in enumerate(train_loader):  # 遍历训练数据集
        steps += step  # 更新步数
        optimizer.zero_grad()  # 清除梯度
        images, gts = data  # 解包数据
        images, gts = images.cuda().float(), gts.cuda().float()  # 移动到GPU
        pred = model(images)  # 获取模型预测
        
        # 计算损失，包括多重监督
        loss = criterion(pred[0], gts)
        for i in range(1, len(pred)):
            loss = loss + criterion(pred[i], gts)
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新参数
        
        # 记录训练损失
        loss_list.append(loss.item())
        
        # 打印日志
        if step % save_cycles == 0:
            lr = optimizer.state_dict()['param_groups'][0]['lr']  # 获取当前学习率
            log_info = f'train: epoch={epoch}, step={step}, loss={np.mean(loss_list):.4f}, lr={lr:.7f}'  # 记录训练信息
            print(log_info)
            logger.info(log_info)
    
    scheduler.step()  # 更新学习率调度器
    return step  # 返回当前步数
