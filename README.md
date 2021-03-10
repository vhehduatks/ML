# Iris 품종분류 기계학습 예제


순서
1.  iris 데이터셋의 구조를 파악
2.  파이썬 사이킷런 라이브러리를 사용하여 기계학습
3.  훈련/테스트데이터 구성 변경 결과 확인하기
4.  random_state의 변경
5.  knn알고리즘의 n_neighbors=k 수 변경해보기
---
### 1.iris 데이터셋의 구조를 파악
```python
iris_dataset=load_iris()
print("dataset의 키",iris_dataset.keys())
```
![KakaoTalk_20210311_052256465](https://user-images.githubusercontent.com/64114699/110692991-4dc62280-822a-11eb-87ec-58f22037049d.png)
  

```python
print(iris_dataset['DESCR'])
```
![KakaoTalk_20210311_052310226](https://user-images.githubusercontent.com/64114699/110693225-8f56cd80-822a-11eb-8ebe-fae846e1e390.png)


```python
print(type(iris_dataset))
print(type(iris_dataset.data))
print(iris_dataset.data.shape)
print(iris_dataset['target_names'])
print(iris_dataset['feature_names'])
print(iris_dataset['target'])
print(iris_dataset['data'][:5])
```
![KakaoTalk_20210311_052342298](https://user-images.githubusercontent.com/64114699/110693528-e197ee80-822a-11eb-8d38-95d43b140563.png)
