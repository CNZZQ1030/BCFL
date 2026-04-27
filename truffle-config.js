module.exports = {
    networks: {
      development: {
        host: "127.0.0.1",     // 本地 Ganache 默认地址
        port: 7545,            // Ganache 默认端口
        network_id: "*"        // 匹配任何网络 ID
      }
    },
    compilers: {
      solc: {
        version: "0.8.0"       // 指定 Solidity 编译器版本，根据你的合约调整
      }
    }
  };