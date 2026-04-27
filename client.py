import flwr as fl
import torch
import numpy as np
from model import load_model, save_model, get_model
from data import load_data, apply_poisoning
import logging
import io
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class BCFLClient(fl.client.NumPyClient):
    def __init__(self, blockchain_utils, ipfs_utils, cid, model_type="cnn", dataset="mnist", num_clients=2, iid=True, attack_type=None, poison_ratio=0.2):
        self.blockchain_utils = blockchain_utils
        self.ipfs_utils = ipfs_utils
        self.cid = cid
        self.model = get_model(model_type)
        self.trainloader, self.testloader, self.num_classes = load_data(dataset, num_clients, iid)
        self.local_loader = self.trainloader[self.cid - 1]
        self.local_testloader = self.testloader[self.cid - 1]  # 使用切分的测试数据
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.writer = SummaryWriter(f"runs/client_{self.cid}")
        self.attack_type = attack_type
        self.poison_ratio = poison_ratio
        logging.info(f"客户端 {self.cid} 初始化完成，训练数据量: {len(self.local_loader.dataset)}, 测试数据量: {len(self.local_testloader.dataset)}")

    def get_parameters(self, config):
        logging.info(f"客户端 {self.cid} 获取参数")
        return []

    def fit(self, parameters, config):
        server_round = config.get("server_round", 1)
        num_epochs = config.get("num_epochs", 2)
        strategy = config.get("strategy", "fedavg")
        mu = config.get("mu", 0.1)
        logging.info(f"客户端 {self.cid} 开始第 {server_round} 轮训练")

        round_num = self.blockchain_utils.get_current_round()
        if round_num == 1:
            cid = self.blockchain_utils.get_global_model_cid(0)
        else:
            cid = self.blockchain_utils.get_global_model_cid(round_num - 1)
        if not cid or cid == "":
            logging.error(f"客户端 {self.cid} 第 {round_num} 轮无有效全局模型 CID")
            return [np.array([], dtype=np.uint8)], 0, {"error": "无效 CID"}

        try:
            model_bytes = self.ipfs_utils.download_model(cid)
            buffer = io.BytesIO(model_bytes)
            state_dict = torch.load(buffer, map_location=self.device)
            self.model.load_state_dict(state_dict)
            global_state = state_dict.copy()
            logging.info(f"客户端 {self.cid} 成功加载全局模型，CID={cid}")
        except Exception as e:
            logging.error(f"客户端 {self.cid} 从 IPFS 下载模型失败: {e}")
            return [np.array([], dtype=np.uint8)], 0, {"error": str(e)}

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.model.train()
        total_steps = num_epochs * len(self.local_loader)
        with tqdm(total=total_steps, desc=f"客户端 {self.cid} 第 {server_round} 轮训练") as pbar:
            for epoch in range(num_epochs):
                for data, target in self.local_loader:
                    if self.attack_type and np.random.rand() < self.poison_ratio:
                        data, target = apply_poisoning(data, target, self.attack_type, self.poison_ratio)
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)
                    if strategy == "fedprox":
                        prox_term = 0
                        for p, g in zip(self.model.parameters(), [global_state[k] for k in global_state]):
                            prox_term += torch.norm(p.data - g.to(self.device)) ** 2
                        loss += (mu / 2) * prox_term
                    loss.backward()
                    optimizer.step()
                    pbar.update(1)

        # 本地评估
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.local_testloader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                total_loss += criterion(outputs, target).item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        avg_loss = total_loss / len(self.local_testloader)
        accuracy = correct / total
        logging.info(f"客户端 {self.cid} 第 {server_round} 轮本地评估 - 损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}")
        self.writer.add_scalar("Loss/Client", avg_loss, server_round)
        self.writer.add_scalar("Accuracy/Client", accuracy, server_round)

        new_cid = self.ipfs_utils.upload_model(self.model)
        if not new_cid:
            logging.error(f"客户端 {self.cid} 上传更新模型到 IPFS 失败")
            return [np.array([], dtype=np.uint8)], 0, {"error": "上传失败"}
        tx_receipt = self.blockchain_utils.submit_update_cid(round_num, new_cid)
        if not tx_receipt:
            logging.error(f"客户端 {self.cid} 提交更新 CID 到区块链失败")
            return [np.array([], dtype=np.uint8)], 0, {"error": "提交 CID 失败"}
        else:
            logging.info(f"客户端 {self.cid} 成功提交更新 CID: {new_cid}, Tx: {tx_receipt.transactionHash.hex()}")

        metrics = {
            "custom_cid": self.cid,
            "local_accuracy": accuracy  # 返回本地准确率
        }
        if self.blockchain_utils.get_highest_reputation_trainer(round_num) == self.blockchain_utils.web3.eth.accounts[self.cid]:
            logging.info(f"客户端 {self.cid} 是声誉最高客户端，开始聚合")
            metrics["aggregated"] = True

        return [np.frombuffer(new_cid.encode('utf-8'), dtype=np.uint8)], len(self.local_loader.dataset), metrics

    def evaluate(self, parameters, config):
        return 0.0, 0, {}