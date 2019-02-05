from NeuralNetwork import *
import numpy as np
import matplotlib.pyplot as plt

# =========== 학습 ============= #

# 입력, 은닉, 출력 노드 수
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

# 학습률
learning_rate = 0.1

# 신경망 인스턴스 생성
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# mnist 학습 데이터인 csv 파일 로딩
# with open(r"./data/mnist_train.csv", "r") as training_data_file:
#     training_data_list = training_data_file.readlines()
training_data_file = open(r"./data/mnist_train.csv", "r")
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 3

# 학습시작
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(",")
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass

# =========== 테스트 ============= #

# 테스트 셋 로드
test_data_file = open(r"./data/mnist_test.csv", "r")
test_data_list = test_data_file.readlines()
test_data_file.close()

# 테스트 셋 첫 record 확인
all_values = test_data_list[1].split(",")
print(all_values[0])

# 테스트 셋 첫 record 이미지 확인
image_array = np.asfarray(all_values[1:]).reshape((28, 28))
plt.imshow(image_array, cmap="Greys", interpolation="None")

# ~255 범위를 0.01 ~ 1 범위의 input list로 만들기
inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
print(n.query(inputs))

# =========== 신경망 테스트 ============== #
scorecard = []

# 테스트 데이터 모음 내의 모든 레코드 검색
for record in test_data_list:
    all_values = record.split(',')
    # 정답은 첫번째 값
    correct_label = int(all_values[0])
    print(correct_label, "correct label")
    # 입력 값의 범위와 값 조정
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.11
    # 신경망에 Query
    outputs = n.query(inputs)
    # 가장 높은 값의 인덱스는 레이블의 인덱스와 일치
    label = np.argmax(outputs)
    print(label, "Network's Answer")
    # 정답 또는 오답을 리스트에 추가
    if (label == correct_label):
        # 정답인 경우 성적표에 1을 더함
        scorecard.append(1)
    else:
        # 정답이 아닌 경우 성적표에 0을 더함
        scorecard.append(0)
        pass
    pass

print(scorecard)

scorecard_array = np.asarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)
