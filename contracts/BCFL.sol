// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract BCFL {
    struct Task {
        address creator;
        string genesisModelCID;
        uint totalRounds;
        uint trainerCount;
        bool initialized;
    }

    struct RoundData {
        string globalModelCID;
        mapping(address => string) updates;
        mapping(address => uint) scores;
        address[] selectedTrainers;
    }

    Task public task;
    uint public currentRound;
    mapping(uint => RoundData) public rounds;
    mapping(address => uint) public rewards;
    mapping(address => uint) public reputation;
    address public owner;
    address public evaluator;

    event TaskInitialized(string genesisModelCID, uint totalRounds);
    event UpdateSubmitted(address trainer, uint round, string cid);
    event ScoreSubmitted(address trainer, uint round, uint score);
    event GlobalModelUpdated(uint round, string cid);
    event TokensDistributed(address trainer, uint amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this");
        _;
    }

    modifier onlyEvaluator() {
        require(msg.sender == evaluator, "Only evaluator can call this");
        _;
    }

    constructor() {
        owner = msg.sender;
        evaluator = msg.sender;
        currentRound = 0;
    }

    function initialize(string memory _genesisModelCID, uint _totalRounds, uint _trainerCount) external onlyOwner {
        require(!task.initialized, "Task already initialized");
        task = Task({
            creator: msg.sender,
            genesisModelCID: _genesisModelCID,
            totalRounds: _totalRounds,
            trainerCount: _trainerCount,
            initialized: true
        });
        rounds[0].globalModelCID = _genesisModelCID;
        emit TaskInitialized(_genesisModelCID, _totalRounds);
    }

    function submitUpdate(uint round, string memory cid) external {
        require(round == currentRound, "Invalid round");
        rounds[round].updates[msg.sender] = cid;
        emit UpdateSubmitted(msg.sender, round, cid);
    }

    function submitScore(uint round, address trainer, uint score) external onlyEvaluator {
        require(round == currentRound, "Invalid round");
        rounds[round].scores[trainer] = score;
        reputation[trainer] = score;
        emit ScoreSubmitted(trainer, round, score);
    }

    function selectTrainersForRound(uint round, address[] memory trainers) external onlyOwner {
        require(round == currentRound, "Invalid round");
        delete rounds[round].selectedTrainers;
        for (uint i = 0; i < trainers.length; i++) {
            rounds[round].selectedTrainers.push(trainers[i]);
        }
    }

    function getSelectedTrainers(uint round) external view returns (address[] memory) {
        return rounds[round].selectedTrainers;
    }

    function getCurrentRound() external view returns (uint) {
        return currentRound;
    }

    function getGlobalModelCID(uint round) external view returns (string memory) {
        return rounds[round].globalModelCID;
    }

    function submitGlobalModel(uint round, string memory cid) external onlyOwner {
        require(round == currentRound, "Invalid round");
        rounds[round].globalModelCID = cid;
        currentRound++;
        emit GlobalModelUpdated(round, cid);
    }

    function distributeTokens(uint round, uint totalReward) external onlyOwner {
        require(round < currentRound, "Round not completed");
        uint totalScore = 0;
        for (uint i = 0; i < rounds[round].selectedTrainers.length; i++) {
            address trainer = rounds[round].selectedTrainers[i];
            totalScore += rounds[round].scores[trainer];
        }
        require(totalScore > 0, "No scores available");

        for (uint i = 0; i < rounds[round].selectedTrainers.length; i++) {
            address trainer = rounds[round].selectedTrainers[i];
            uint score = rounds[round].scores[trainer];
            uint reward = (score * totalReward) / totalScore;
            rewards[trainer] += reward;
            emit TokensDistributed(trainer, reward);
        }
    }

    function setEvaluator(address _evaluator) external onlyOwner {
        evaluator = _evaluator;
    }

    function getReward(address trainer) external view returns (uint) {
        return rewards[trainer];
    }

    function getReputation(address trainer) external view returns (uint) {
        return reputation[trainer];
    }

    function getUpdateCID(uint round, address trainer) external view returns (string memory) {
        return rounds[round].updates[trainer];
    }
}