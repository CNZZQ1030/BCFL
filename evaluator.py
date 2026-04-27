import numpy as np
import torch
from model import load_model, get_model
from data import load_data
from blockchain_utils import BlockchainUtils
from ipfs_utils import IPFSUtils
import logging

class Evaluator:
    def __init__(self, blockchain_utils, ipfs_utils, round_num, dataset="mnist", alpha=0.9, gamma=0.1, tau=1.0):
        self.blockchain_utils = blockchain_utils
        self.ipfs_utils = ipfs_utils
        self.round_num = round_num
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.testloader = load_data(dataset=dataset, num_clients=1)[1][0]
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.reputation = {}
        self.participation_times = {}
        self.global_model_state = None
        self.global_model = None

    def evaluate_model(self, model, model_type="cnn"):
        try:
            model = model.to(self.device)
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in self.testloader:
                    data, target = data.to(self.device), target.to(self.device)
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            accuracy = correct / total
            logging.info(f"评估本地模型，准确率: {accuracy:.4f}")
            return accuracy
        except Exception as e:
            logging.error(f"评估模型失败: {e}")
            return 0.0

    def evaluate_global_model(self, global_model, model_type="cnn"):
        if global_model is None:
            logging.warning("未找到全局模型")
            return 0.0
        try:
            global_model = global_model.to(self.device)
            global_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in self.testloader:
                    data, target = data.to(self.device), target.to(self.device)
                    outputs = global_model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            accuracy = correct / total
            logging.info(f"评估全局模型，准确率: {accuracy:.4f}")
            return accuracy
        except Exception as e:
            logging.error(f"评估全局模型失败: {e}")
            return 0.0

    def calculate_model_improvement(self, local_accuracy, global_accuracy):
        improvement = max(0, (local_accuracy - global_accuracy) / (1 - global_accuracy + 1e-8))
        return improvement

    def calculate_update_consistency(self, client_state_dict, global_model):
        if global_model is None:
            logging.warning("未找到全局模型，R_consist 默认设为 0")
            return 0.0
        try:
            client_model = get_model("cnn").to(self.device)
            client_model.load_state_dict(client_state_dict)
            
            client_grad = []
            for g_param, c_param in zip(global_model.parameters(), client_model.parameters()):
                diff = c_param - g_param
                client_grad.append(diff.flatten())
            gradient = torch.cat(client_grad)
            
            global_grads = []
            for param in global_model.parameters():
                if param.grad is not None:
                    global_grads.append(param.grad.flatten().clone())
            global_grad = torch.cat(global_grads) if global_grads else torch.zeros_like(gradient)
            
            norm = torch.norm(gradient).item()
            clipped_norm = min(norm, self.tau)
            cos_theta = torch.cosine_similarity(gradient, global_grad, dim=0).item()
            R_consist = ((1 + cos_theta) * clipped_norm) / self.tau
            return R_consist
        except Exception as e:
            logging.error(f"计算更新一致性失败: {e}")
            return 0.0

    def calculate_participation_activity(self, client_address, current_time):
        last_time = self.participation_times.get(client_address, 0)
        self.participation_times[client_address] = current_time
        time_diff = current_time - last_time
        R_active = np.exp(-self.gamma * time_diff)
        return R_active

    def calculate_governance_contribution(self, client_address, temp_reputations, total_clients):
        if total_clients == 0:
            return 0.0
        sorted_clients = sorted(temp_reputations.items(), key=lambda x: x[1], reverse=True)
        rank = next((i for i, (addr, _) in enumerate(sorted_clients) if addr == client_address), total_clients)
        R_govern = (total_clients - rank) / total_clients if total_clients > 0 else 0.0
        return R_govern

    def calculate_reputation(self, client_address, local_accuracy, global_accuracy, client_state_dict, global_model, temp_reputations, total_clients):
        R_improve = self.calculate_model_improvement(local_accuracy, global_accuracy)
        R_consist = self.calculate_update_consistency(client_state_dict, global_model)
        R_active = self.calculate_participation_activity(client_address, self.round_num)
        R_govern = self.calculate_governance_contribution(client_address, temp_reputations, total_clients)
        
        R_total = R_improve + R_consist + R_active + R_govern
        logging.info(f"客户端 {client_address} 声誉: R_improve={R_improve:.4f}, R_consist={R_consist:.4f}, R_active={R_active:.4f}, R_govern={R_govern:.4f}, 总计={R_total:.4f}")
        
        self.reputation[client_address] = R_total
        return R_total

    def submit_scores(self, aggregator_address, client_updates, model_type, local_accuracies, global_accuracy, global_model):
        self.global_model = global_model
        if global_model is not None:
            self.global_model_state = global_model.state_dict()
        total_clients = len(client_updates)
        logging.info(f"轮次 {self.round_num} - 开始为 {total_clients} 个训练者计算声誉")
        
        # 一次性计算所有客户端的临时声誉和本地准确率
        temp_reputations = {}
        local_acc_dict = {}  # 保存本地准确率，避免重复计算
        global_acc = self.evaluate_global_model(global_model) if global_model else 0.0
        for addr, state_dict in client_updates.items():
            client_model = get_model(model_type).to(self.device)
            client_model.load_state_dict(state_dict)
            local_acc = self.evaluate_model(client_model)  # 只评估一次
            local_acc_dict[addr] = local_acc
            R_improve = self.calculate_model_improvement(local_acc, global_acc)
            R_consist = self.calculate_update_consistency(state_dict, global_model)
            R_active = self.calculate_participation_activity(addr, self.round_num)
            temp_reputations[addr] = R_improve + R_consist + R_active

        # 计算每个客户端的声誉，复用本地准确率
        for i, (trainer, state_dict) in enumerate(client_updates.items()):
            local_acc = local_acc_dict[trainer]  # 复用已计算的 local_acc
            reputation = self.calculate_reputation(
                trainer, local_acc, global_acc, state_dict, global_model, temp_reputations, total_clients
            )
            logging.info(f"Trainer {trainer}: Calculated Reputation = {reputation:.4f}")

    def get_reputation(self, client_address):
        return self.reputation.get(client_address, 0.0)