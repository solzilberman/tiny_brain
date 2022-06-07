#include <stdio.h>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <functional>
#include <cmath>

typedef Eigen::MatrixXd mat;
typedef Eigen::VectorXd vec;

// activation function
float sig(float x){
    return 1.0f / (1.0f + exp(-x)); 
}

float dsigmoid(float y){
    // y := val returns by sigmoid()
    return y * (1 - y);
}

mat applySigmoid(mat m){
    mat k = m;
    for (int j1 = 0; j1 < m.rows(); j1++){
        for (int j2 = 0; j2 < m.cols(); j2++){
            k(j1, j2) = sig(m(j1, j2));
        }
    }
    return k;
}

mat applyDSigmoid(mat m){
    mat k = m;
    for (int j1 = 0; j1 < m.rows(); j1++){
        for (int j2 = 0; j2 < m.cols(); j2++){
            k(j1, j2) = dsigmoid(m(j1, j2));
        }
    }
    return k;
}

class NeuralNetwork{
public:
    std::vector<uint32_t> _topology; //eg [3,2,1]
    std::vector<mat> _weightMats;
    std::vector<mat> _valMats;
    std::vector<mat> _biasMats;
    float _learningRate;

    NeuralNetwork(std::vector<uint32_t> topology, float learningRate = 0.1f){
        _topology = topology;
        _weightMats = {};
        _valMats = {};
        _biasMats = {};
        _learningRate = learningRate;

        // def weight mat
        for (int32_t i = 0; i < _topology.size()-1; i++){
            // for each nueron in layer i, i+1 weights needed 
            // these are edges going from i to i+1 of graph repr
            mat weights = mat::Random(_topology[i+1],_topology[i]); //each col is weights from nueron to next layer
            _weightMats.push_back(weights);

            mat bias = mat::Random(_topology[i+1], 1);
            _biasMats.push_back(bias); 
        }
        _valMats.resize(_topology.size());
    };

    // propogate data through layers ->
    bool feedForward(std::vector<float> input) {
        if (input.size() != _topology[0]){
            std::cout << "[error - feedforward] incorrect input!\n";
            return false;
        }
        // init vals matrix
        mat vals(input.size(), 1);
        for (int32_t i = 0; i < input.size(); i++){
            vals(i,0) = input[i];
        }

        
        //prop data forward
        for(int32_t  i = 0; i < _weightMats.size(); i++){
            _valMats[i] = vals;
            vals = _weightMats[i] * vals;
            vals += _biasMats[i]; 
            vals = applySigmoid(vals);
        }
        _valMats[_weightMats.size()] = vals; //values updated based on nueral params
        return true;
    }

    bool backPropagate(std::vector<float> targetOutput){
        if(targetOutput.size() != _topology.back()){
            std::cout << "[error - backprop] incorrect input!\n";
            return false;
        }
        mat err(targetOutput.size(),1);
        for(int32_t i = 0; i < targetOutput.size(); i++){
            err(i,0) = targetOutput[i];
        }

        mat sub = _valMats.back() * -1.f;
        err += sub;
        for(int32_t  i  = _weightMats.size() - 1; i >= 0; i--){
            mat prevErr = _weightMats[i].transpose() * err;
            mat dout = applyDSigmoid(_valMats[i+1]);
            mat gradients = err.cwiseProduct(dout); // now we have delta errors for each val
            gradients *= _learningRate; //now we have values that need modification
            mat weightedGradients = gradients * _valMats[i].transpose();  
            _weightMats[i] = _weightMats[i] + weightedGradients; // update new weights based on n dimensional grad descent
            _biasMats[i] = _biasMats[i] + gradients;
            err = prevErr; // now update errors and move to next layer
        }
        return true;
    }

    mat getPrediction(){
        return _valMats.back();
    }
};

int main(){
    //test nn on learning XOR operation
    std::vector<uint32_t> topology = {2,3,1};
    NeuralNetwork *nn = new NeuralNetwork(topology, 0.1f);

    std::vector<std::vector<float>> targetInputs = {
        {0.f,0.f},
        {1.f,0.f},
        {0.f,1.f},
        {1.f,1.f}
    };
    std::vector<std::vector<float>> targetOutputs = {
        {0.f},
        {1.f},
        {1.f},
        {0.f}
    };

    int EPOCHS = 100000;
    std::cout << "[info] training started\n";
    for(int i = 0; i < EPOCHS; i++){
        int ind = rand() % 4;
        nn->feedForward(targetInputs[ind]);
        nn->backPropagate(targetOutputs[ind]);
    }
    std::cout << "[info] training completed\n";
    
    for (auto inp : targetInputs){
        nn->feedForward(inp);
        auto preds = nn->getPrediction();
        std::cout << inp[0] << ", " << inp[1] << " -> "<< preds << std::endl;
    }
    return 0;
}