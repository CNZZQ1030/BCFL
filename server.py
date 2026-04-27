import flwr as fl
from blockchain_utils import BlockchainUtils
from ipfs_utils import IPFSUtils
from model import load_model, save_model, get_model
import torch
import logging
import io
from typing import Optional, Tuple, Dict, Type
from flwr.common import Parameters, Scalar, NDArrays
from torch.utils.tensorboard import SummaryWriter
import os
from evaluator import Evaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - Server - %(message)s')

class BCFLStrategy(fl.server.strategy.Strategy):
    def __init__(self, blockchain_utils, ipfs_utils, model_type: str, total_rounds: int, dataset: str, strategy: str, iid: bool):
        super().__init__()
        self.blockchain_utils = blockchain_utils
        self.ipfs_utils = ipfs_utils
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from data import load_data
        self.trainloaders, self.testloaders, self.num_classes = load_data(dataset=dataset, num_clients=1, iid=iid)
        self.testloader = self.testloaders[0]
        self.total_rounds = total_rounds
        self.strategy = strategy
        self.writer = SummaryWriter("runs/server")
        self.evaluator = None
        self.global_model = None
        model = get_model(self.model_type)
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"服务器端模型参数总数: {total_params}")

    def initialize_parameters(self, client_manager):
        return fl.common.ndarrays_to_parameters([])

    def configure_fit(self, server_round, parameters, client_manager):
        num_clients = self.blockchain_utils.contract.functions.task().call()[3]
        available_clients = client_manager.sample(num_clients=num_clients, min_num_clients=num_clients)
        config = {"server_round": server_round, "num_epochs": 2, "strategy": self.strategy, "mu": 0.1}
        fit_ins = fl.common.FitIns(parameters, config)
        trainer_addresses = [self.blockchain_utils.web3.eth.accounts[i+1] for i in range(num_clients)]
        self.blockchain_utils.contract.functions.selectTrainersForRound(server_round, trainer_addresses).transact({'from': self.blockchain_utils.account})
        return [(client, fit_ins) for client in available_clients]

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            logging.warning("未收到客户端结果")
            return None, {}

        cids = []
        client_addresses = []
        local_accuracies = []
        client_updates = {}
        for client, fit_res in results:
            ndarrays = fl.common.parameters_to_ndarrays(fit_res.parameters)
            if not ndarrays or len(ndarrays) == 0 or len(ndarrays[0]) == 0:
                logging.warning(f"客户端 {client} 返回空参数")
                continue
            cid = ndarrays[0].tobytes().decode('utf-8')
            if not cid:
                logging.warning(f"客户端 {client} 返回无效CID")
                continue
            cids.append(cid)
            custom_cid = fit_res.metrics.get("custom_cid", None)
            if custom_cid is None:
                logging.error(f"客户端 {client} 未提供 custom_cid")
                continue
            client_address = self.blockchain_utils.web3.eth.accounts[custom_cid]
            client_addresses.append(client_address)
            local_accuracies.append(fit_res.metrics.get("local_accuracy", 0.0))

            try:
                model_bytes = self.ipfs_utils.download_model(cid)
                buffer = io.BytesIO(model_bytes)
                state_dict = torch.load(buffer, map_location=self.device)
                client_updates[client_address] = state_dict
                logging.info(f"缓存客户端 {client_address} 模型，CID={cid}")
            except Exception as e:
                logging.error(f"下载CID {cid} 失败: {e}")
                continue

        if not client_updates:
            logging.error("无有效模型可聚合")
            return None, {}

        # 加载上一轮的全局模型并评估
        previous_global_model = None
        global_accuracy = 0.0
        if server_round > 1:
            previous_round = server_round - 1
            previous_cid = self.blockchain_utils.get_global_model_cid(previous_round)  # 修正为 previous_round
            if previous_cid:
                try:
                    model_bytes = self.ipfs_utils.download_model(previous_cid)
                    buffer = io.BytesIO(model_bytes)
                    state_dict = torch.load(buffer, map_location=self.device)
                    previous_global_model = get_model(self.model_type).to(self.device)
                    previous_global_model.load_state_dict(state_dict)
                    logging.info(f"加载上一轮全局模型，CID={previous_cid}")
                    global_accuracy, _ = self._evaluate_global_model(previous_global_model)
                except Exception as e:
                    logging.error(f"加载上一轮全局模型失败: {e}")

        # 计算声誉
        if self.strategy == "reputation":
            if not self.evaluator:
                self.evaluator = Evaluator(self.blockchain_utils, self.ipfs_utils, server_round, dataset="mnist")
            self.evaluator.round_num = server_round
            self.evaluator.global_model_state = previous_global_model.state_dict() if previous_global_model else None
            self.evaluator.submit_scores(
                self.blockchain_utils.account,
                client_updates,
                self.model_type,
                local_accuracies,
                global_accuracy,
                previous_global_model
            )
            print(f"\n第 {server_round} 轮声誉分数：")
            for addr in client_addresses:
                reputation = self.evaluator.get_reputation(addr)
                print(f"Trainer {addr}: Reputation = {reputation:.4f}")
            print()

        # 聚合新全局模型
        global_model = get_model(self.model_type).to(self.device)
        with torch.no_grad():
            total_weight = len(client_updates) if self.strategy == "fedavg" else sum(self.evaluator.get_reputation(addr) for addr in client_addresses)
            for param in global_model.parameters():
                param.data.zero_()
            for addr, state_dict in client_updates.items():
                model = get_model(self.model_type).to(self.device)
                model.load_state_dict(state_dict)
                weight = 1.0 / len(client_updates) if self.strategy == "fedavg" else self.evaluator.get_reputation(addr) / (total_weight if total_weight > 0 else 1.0)
                for global_param, client_param in zip(global_model.parameters(), model.parameters()):
                    global_param.data += client_param.data * weight

        self.global_model = global_model
        new_cid = self.ipfs_utils.upload_model(global_model)
        if not new_cid:
            logging.error("上传全局模型到IPFS失败")
            return None, {}
        self.blockchain_utils.submit_global_model(server_round, new_cid)
        logging.info(f"上传全局模型，CID={new_cid}")

        global_accuracy, _ = self._evaluate_global_model(global_model)
        logging.info(f"全局模型第 {server_round} 轮评估 - 准确率: {global_accuracy:.4f}")

        metrics = {"accuracy": global_accuracy}
        if server_round == self.total_rounds:
            save_path = f"ipfs_models/global_model_round_{server_round}.pth"
            os.makedirs("ipfs_models", exist_ok=True)
            torch.save(global_model.state_dict(), save_path)
            logging.info(f"最终全局模型已保存到 {save_path}，CID={new_cid}")

        return fl.common.ndarrays_to_parameters([new_cid.encode('utf-8')]), metrics

    def _evaluate_global_model(self, global_model):
        if global_model is None:
            return 0.0, None
        global_model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        global_grads = []
        for data, target in self.testloader:
            data, target = data.to(self.device), target.to(self.device)
            global_model.zero_grad()
            outputs = global_model(data)
            loss = criterion(outputs, target)
            loss.backward()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            for param in global_model.parameters():
                if param.grad is not None:
                    global_grads.append(param.grad.flatten().clone())
        global_accuracy = correct / total
        global_grad = torch.cat(global_grads) if global_grads else torch.zeros(1).to(self.device)
        return global_accuracy, global_grad

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None