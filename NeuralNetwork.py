import numpy as np
import scipy.special

class NeuralNetwork:

    #  신경망 초기화
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # 입력, 은닉, 출력 계층의 노드 개수 설정
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # self.wih = (np.random.rand(self.hidden_nodes, self.input_nodes) - 0.5)
        # self.who = (np.random.rand(self.output_nodes, self.hidden_nodes) - 0.5)
        self.wih = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.who = np.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

        # 학습률
        self.learning_rate = learning_rate

        # 활성화 함수로 sigmoid 함수 사용
        self.activation_function = lambda x : scipy.special.expit(x)
        pass

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        # hidden 층으로 들어오는 신호를 계산
        hidden_inputs = np.dot(self.wih, inputs)
        # hidden 층에서 나가는 신호를 계산
        hidden_outputs = self.activation_function(hidden_inputs)
        # 최종 출력 층으로 들어오는 신호를 계산
        final_inputs = np.dot(self.who, hidden_outputs)
        # hidden 층에서 나가는 신호를 계산
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    # Neural NetWork 학습
    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # hidden 층으로 들어오는 신호를 계산
        hidden_inputs = np.dot(self.wih, inputs)
        # hidden 층에서 나가는 신호를 계산
        hidden_outputs = self.activation_function(hidden_inputs)

        # 최종 출력 층으로 들어오는 신호를 계산
        final_inputs = np.dot(self.who, hidden_outputs)
        # hidden 층에서 나가는 신호를 계산
        final_outputs = self.activation_function(final_inputs)

        # 출력 계층의 오차 (실제값 - 오차값)
        output_errors = targets - final_outputs
        # 은닉 계층의 오차는 가중치에 의해 나뉜 출력 계층의 오차들을 재조합해 계산
        hidden_errors = np.dot(self.who.T, output_errors)

        # 은닉 계층과 출력 계층 간의 가중치 업데이트
        self.who += self.learning_rate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                                np.transpose(hidden_outputs))
        self.wih += self.learning_rate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                                np.transpose(inputs))

        pass
