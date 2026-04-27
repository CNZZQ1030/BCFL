Below is the revised `README.md` based on the updated code and functionality. The revisions reflect the current implementation details, including the separation of `server_main.py` and `client_main.py`, the corrected `evaluate` method, and the observed behavior from the logs (e.g., successful 3-round training with improving accuracy).

---

# **Decentralized Federated Learning with Blockchain and IPFS**

### **Project Overview**
This project implements a decentralized federated learning (FL) system using the Flower framework, integrated with Ethereum smart contracts (via Ganache) and IPFS for model storage. The design is inspired by the architecture outlined in the paper *Blockchain-Enabled Federated Learning: A Reference Architecture Design, Implementation, and Verification*.

**Core Components**:
- **Flower**: Framework for coordinating federated learning between clients and server.
- **Ganache**: Local Ethereum blockchain for development and testing.
- **Web3.py**: Python library for interacting with smart contracts.
- **IPFS**: Decentralized storage for sharing model parameters.
- **Solidity**: Language for implementing smart contract logic.

---

### **Project Structure**
```
decentralized_fl/
├── blockchain_utils.py       # Blockchain interaction utilities
├── ipfs_utils.py             # IPFS interaction utilities
├── model.py                  # Machine learning model definition (CNN for FashionMNIST)
├── data.py                   # Data loading and preprocessing
├── client.py                 # Custom Flower client implementation (BCFLClient)
├── server.py                 # Custom Flower server strategy (BCFLStrategy)
├── server_main.py            # Script to start the Flower server
├── client_main.py            # Script to start a Flower client
├── contracts/
│   └── BCFL.sol              # Solidity smart contract for FL coordination
├── migrations/
│   └── 2_deploy_contracts.js # Truffle deployment script
├── truffle-config.js         # Truffle configuration file
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

---

### **Main Features**
1. **Task Initialization**: Configures the FL task with an initial model CID, number of rounds, and trainers via blockchain.
2. **Trainer Registration**: Trainers are selected from available Ethereum accounts (currently static).
3. **Round Management**: Blockchain coordinates training rounds and stores model CIDs.
4. **Client Selection**: Selects trainers per round (currently selects all available clients).
5. **Model Updates**: Clients train models, upload them to IPFS, and submit CIDs to the blockchain.
6. **Global Model Aggregation**: Server aggregates client updates and stores the global model on IPFS.
7. **Evaluation**: Server evaluates the global model after each round, reporting loss and accuracy.
8. **Incentive Placeholder**: Token distribution logic exists in the smart contract but is not fully implemented.

---

### **Environment Setup**
- **Python 3.10+**: Required to run the code.
- **Ganache**: Local Ethereum blockchain (`npm install -g ganache`).
- **IPFS**: Local IPFS node (install via [IPFS Desktop](https://ipfs.io/#install) or CLI: `go get -u github.com/ipfs/go-ipfs`).
- **Truffle**: For compiling and deploying smart contracts (`npm install -g truffle`).

---

### **Installation Steps**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/decentralized_fl.git
   cd decentralized_fl
   ```
2. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Start Ganache**:
   ```bash
   ganache-cli --port 7545 --networkId 5777 -a 31 -d
   ```
   - Default RPC: `http://127.0.0.1:7545`.
   - Note the first account address (e.g., `0xe78A0F7E598Cc8b0Bb87894B0F60dD2a88d6a8Ab`).

4. **Start IPFS Node**:
   ```bash
   ipfs daemon
   ```
   - Ensure it runs on `http://127.0.0.1:5001`.

5. **Deploy Smart Contract**:
   - Place `BCFL.sol` in `contracts/` and `2_deploy_contracts.js` in `migrations/`.
   - Ensure `truffle-config.js` is configured for Ganache (port 7545).
   - Compile and deploy:
     ```bash
     truffle compile
     truffle migrate --network development
     ```
   - Copy the contract address from the deployment output and ABI from `build/contracts/BCFL.json` into `server_main.py` and `client_main.py`.

---

tensorboard --logdir runs


### **Usage Instructions**
1. **Prepare Initial Model**:
   - Train and save an initial CNN model for FashionMNIST, then upload it to IPFS:
     ```python
     from model import CNN, save_model
     from ipfs_utils import IPFSUtils
     model = CNN()
     save_model(model, "ipfs_models/initial_model.pth")
     ipfs = IPFSUtils()
     cid = ipfs.upload_model("ipfs_models/initial_model.pth")
     print(f"Initial CID: {cid}")
     ```
   - Update `DEFAULT_PATH` in `server_main.py` if the path differs.

2. **Configure Parameters**:
   - In `server_main.py` and `client_main.py`, ensure:
     - `DEFAULT_URL = "http://127.0.0.1:7545"`
     - `DEFAULT_ADDR` matches your deployed contract address.
     - ABI is correctly loaded from `build/contracts/BCFL.json`.

3. **Run the System**:
   - Start the server (e.g., 2 clients, 3 rounds):
     ```bash
     python server_main.py --clients 2 --rounds 3 --num_epochs 3
     ```
   - In separate terminals, start clients (e.g., Client 1 and Client 2):
     ```bash
     python client_main.py --cid 1 --account_idx 1
     python client_main.py --cid 2 --account_idx 2
     ```

---

### **Expected Output**
- **Server**: Initializes the task, uploads the initial model to IPFS, runs 3 rounds, aggregates client updates, and evaluates the global model after each round. Example:
  ```
  2025-03-25 21:03:48,106 - Server - 上传模型到IPFS，CID=QmXhRXYQteTfGHrxp8G4rXHzLZwZWuJDQWRJMrGAdfwTMT
  任务已初始化：CID=QmXhRXYQteTfGHrxp8G4rXHzLZwZWuJDQWRJMrGAdfwTMT, 总轮次=3, 训练者数量=2
  2025-03-25 21:05:05,570 - Server - Server Round 3 - Loss: 0.0424, Accuracy: 0.9857
  2025-03-25 21:05:05,570 - Server - Run finished 3 round(s) in 76.61s
  ```
- **Client**: Trains the model for 2 epochs per round, uploads updates to IPFS, and submits CIDs to the blockchain. Example:
  ```
  2025-03-25 21:04:42,585 - Client 1 - 开始第 3 轮训练
  2025-03-25 21:05:03,854 - Client 1 - 成功上传更新模型，CID=QmWFBu96vm3YqJTBZ27XhnytgfBjzrMvswaLsWVPL4FA26
  2025-03-25 21:05:03,958 - Client 1 - 成功提交更新CID，交易哈希: bf30b8d58dfeb4d42bd7a867137ac0b148d625524689756818ef4ec6ae8f135f
  ```

---

### **Customization Options**
- **Model**: Replace `CNN` in `model.py` with another architecture.
- **Dataset**: Modify `data.py` to use a different dataset (currently FashionMNIST).
- **Client Selection**: Enhance `select_trainers_for_round` in `server.py` for score-based selection.
- **Incentives**: Extend `distribute_tokens` in `blockchain_utils.py` with an ERC-20 token contract.

---

### **Notes**
- **Incentives**: The `distributeTokens` function in the smart contract is a placeholder and does not yet distribute real tokens.
- **Client Selection**: Currently selects all available trainers; can be extended for research purposes.
- **Scalability**: Tested on FashionMNIST; larger datasets may require optimization due to IPFS upload times.
- **Evaluation**: Centralized evaluation occurs on the server; client-side evaluation is skipped by design.

---

### **License**
This project is licensed under the MIT License.

---

### **References**
- [Flower Framework](https://flower.dev/)
- [Web3.py Documentation](https://web3py.readthedocs.io/)
- [IPFS Documentation](https://docs.ipfs.io/)
- Paper: *Blockchain-Enabled Federated Learning: A Reference Architecture Design, Implementation, and Verification*

---

This updated `README.md` aligns with the revised code, reflects the successful execution seen in the logs, and provides clear instructions for setup and usage. Let me know if you'd like further adjustments!