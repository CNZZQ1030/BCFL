from web3 import Web3
import logging

class BlockchainUtils:
    def __init__(self, provider_url, contract_address, abi):
        self.web3 = Web3(Web3.HTTPProvider(provider_url))
        self.contract = self.web3.eth.contract(address=contract_address, abi=abi)
        self.account = self.web3.eth.accounts[0]

    def get_current_round(self):
        try:
            return self.contract.functions.getCurrentRound().call()
        except Exception as e:
            logging.error(f"获取当前轮次失败: {e}")
            raise

    def get_global_model_cid(self, round_num):
        try:
            cid = self.contract.functions.getGlobalModelCID(round_num).call()
            return cid if cid else ""
        except Exception as e:
            logging.error(f"获取轮次 {round_num} 的全局模型CID失败: {e}")
            return ""

    def submit_update_cid(self, round_num, cid):
        try:
            tx = self.contract.functions.submitUpdate(round_num, cid).transact({'from': self.account})
            return self.web3.eth.wait_for_transaction_receipt(tx)
        except Exception as e:
            logging.error(f"提交更新CID失败: {e}")
            return None

    def get_selected_trainer_reputation(self, round_num, trainer):
        try:
            return self.contract.functions.getReputation(trainer).call()
        except Exception as e:
            logging.error(f"获取训练者 {trainer} 声誉失败: {e}")
            return 0

    def get_highest_reputation_trainer(self, round_num):
        trainers = self.get_selected_trainers(round_num)
        if not trainers:
            return None
        reputations = [(trainer, self.get_selected_trainer_reputation(round_num, trainer)) for trainer in trainers]
        return max(reputations, key=lambda x: x[1])[0]

    def submit_global_model(self, round_num, cid):
        try:
            tx = self.contract.functions.submitGlobalModel(round_num, cid).transact({'from': self.account})
            return self.web3.eth.wait_for_transaction_receipt(tx)
        except Exception as e:
            logging.error(f"提交全局模型失败: {e}")
            return None

    def get_selected_trainers(self, round_num):
        return self.contract.functions.getSelectedTrainers(round_num).call()

    def distribute_tokens(self, round_num, total_reward, enable_tokens=False):
        if enable_tokens:
            try:
                tx = self.contract.functions.distributeTokens(round_num, total_reward).transact({'from': self.account})
                return self.web3.eth.wait_for_transaction_receipt(tx)
            except Exception as e:
                logging.error(f"分配代币失败: {e}")
                return None
        return None