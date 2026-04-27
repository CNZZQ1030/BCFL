import io
import ipfshttpclient
import torch
import logging
import os
import tempfile
from retrying import retry

class IPFSUtils:
    def __init__(self, ipfs_api="/ip4/127.0.0.1/tcp/5001/http"):
        self.ipfs_api = ipfs_api
        try:
            with ipfshttpclient.connect(self.ipfs_api) as ipfs:
                logging.info("成功连接到IPFS节点")
        except Exception as e:
            logging.error(f"无法连接到IPFS节点: {e}")
            raise

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def upload_model(self, model_or_path, use_file=False):
        try:
            if use_file or not isinstance(model_or_path, torch.nn.Module):
                if isinstance(model_or_path, str):
                    file_path = model_or_path
                else:
                    file_path = tempfile.mktemp(suffix=".pth")
                    torch.save(model_or_path.state_dict(), file_path)
                with ipfshttpclient.connect(self.ipfs_api) as ipfs:
                    cid = ipfs.add(file_path)["Hash"]
                if isinstance(model_or_path, torch.nn.Module):
                    os.remove(file_path)
            else:
                buffer = io.BytesIO()
                torch.save(model_or_path.state_dict(), buffer)
                buffer.seek(0)
                with ipfshttpclient.connect(self.ipfs_api) as ipfs:
                    cid = ipfs.add_bytes(buffer.read())
            if not cid:
                raise ValueError("上传成功但未返回有效CID")
            logging.info(f"上传模型到IPFS，CID={cid}")
            return cid
        except Exception as e:
            logging.error(f"上传模型到IPFS失败: {e}")
            return None

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def download_model(self, cid, output_path=None):
        if not cid or not isinstance(cid, str) or cid.strip() == "":
            logging.error(f"无效的CID: {cid}")
            raise ValueError("CID 不能为空或无效")
        try:
            with ipfshttpclient.connect(self.ipfs_api) as ipfs:
                model_bytes = ipfs.cat(cid, timeout=60)
                logging.info(f"从IPFS下载CID {cid}")
            if output_path:
                with open(output_path, 'wb') as f:
                    f.write(model_bytes)
                logging.info(f"模型保存到 {output_path}")
            return model_bytes
        except Exception as e:
            logging.error(f"从IPFS下载模型失败: {e}")
            raise