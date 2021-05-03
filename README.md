# Machine Learning readme
- [Week_2.iris](#2주차iris-데이터셋-기계학습)
- [Week 2.wisconsin](#2주차wisconsin-breast-cancer-기계학습)
- [Week 3.Regression](#3주차knn회귀분석선형회귀분석)
- [Week 4.A lot of things](#4주차multi_class나이브베이즈결정트리렌덤포레스트부스팅)
- [Week 5.Ensemble,SVM,신경망](#5주차부스팅앙상블svm신경망)
- [Week 6.비지도학습,데이터전처리,클러스터링](#6주차비지도학습데이터전처리클러스터링)
---
## (2주차)Iris 데이터셋 기계학습


순서
1.  iris 데이터셋의 구조를 파악
2.  파이썬 사이킷런 라이브러리를 사용하여 기계학습
3.  훈련/테스트데이터 구성 변경 결과 확인하기
4.  knn알고리즘의 n_neighbors=k 수 변경해보기
5.  random_state의 변경

### 1.iris 데이터셋의 구조를 파악
```python
iris_dataset=load_iris()
print("dataset의 키",iris_dataset.keys())
```
![KakaoTalk_20210311_052256465](https://user-images.githubusercontent.com/64114699/110692991-4dc62280-822a-11eb-87ec-58f22037049d.png)
<br><br/>

```python
print(iris_dataset['DESCR'])
```
![KakaoTalk_20210311_052310226](https://user-images.githubusercontent.com/64114699/110693225-8f56cd80-822a-11eb-8ebe-fae846e1e390.png)
<br><br/>

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
<br><br/>

### 2.파이썬 사이킷런 라이브러리를 사용하여 기계학습
```python
x_train,x_test,y_train,y_test=train_test_split(iris_dataset.data,iris_dataset.target,random_state=0)

print('x_tarin의 크기',x_train.shape)
print('y_tarin의 크기',y_train.shape)
print('x_test의 크기',x_test.shape)
print('y_test의 크기',y_test.shape)
```
![KakaoTalk_20210311_164447865](https://user-images.githubusercontent.com/64114699/110752861-294d6300-8289-11eb-9725-751290a14652.png)
<br><br/>

```python
#iris데이터프레임(x_train을 이용한) 만들기
iris_df=pd.DataFrame(x_train,columns=iris_dataset.feature_names)

#넘파이로 데이터의 산전도행렬 표현
pd.plotting.scatter_matrix(iris_df,c=y_train,figsize=(15,15),marker='o',hist_kwds={'bins':20},s=60,alpha=0.8,cmap=mglearn.cm3)
plt.show()
```
![KakaoTalk_20210311_052505906](https://user-images.githubusercontent.com/64114699/110752020-066e7f00-8288-11eb-80f5-0a3cfd0eb53f.png)
<br><br/>

```python
#가장 가까운 1개의 이웃의 영향을 받아 분류하도록 함
knn=KNeighborsClassifier(n_neighbors=1)

#x데이터와 y라벨로 fit 시켜준다=학습
knn.fit(x_train,y_train)

#새로운 데이터를 예측하기
unknown_data=[[5,2.9,1,0.2]]
guesses=knn.predict(unknown_data)
print('predict:',guesses)
print('predict name:',iris_dataset['target_names'][guesses])

#테스트 데이터의 정확도 평가하기
print('test_x accuracy :',knn.score(x_test,y_test))
```
![KakaoTalk_20210311_164413523](https://user-images.githubusercontent.com/64114699/110752942-3ff3ba00-8289-11eb-83d5-89ef8d9c07b7.png)
<br><br/>

### 4.knn알고리즘의 n_neighbors=k 수 변경해보기
 ```python
 #k값의 변동에 따른 모델의 정확도 시각화
k_range=range(1,101)
accuracy=list()

for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    accuracy.append(knn.score(x_test,y_test))

plt.plot(k_range,accuracy)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.show()
```
![KakaoTalk_20210311_052416132](https://user-images.githubusercontent.com/64114699/110753672-33bc2c80-828a-11eb-8d1b-d3c1787418dd.png)
<br><br/>

### 5.random_state의 변경
```python
#데이터셋 분할(훈련,테스트)
# x_train,x_test,y_train,y_test=train_test_split(iris_dataset.data,iris_dataset.target,random_state=0)
x_train,x_test,y_train,y_test=train_test_split(iris_dataset.data,iris_dataset.target,test_size=0.33,random_state=42)

print('x_tarin의 크기',x_train.shape)
print('y_tarin의 크기',y_train.shape)
print('x_test의 크기',x_test.shape)
print('y_test의 크기',y_test.shape)
```
before
![KakaoTalk_20210311_164447865](https://user-images.githubusercontent.com/64114699/110752861-294d6300-8289-11eb-9725-751290a14652.png)
after
![KakaoTalk_20210311_165633236](https://user-images.githubusercontent.com/64114699/110754125-ca88e900-828a-11eb-8723-fe617d5beca6.png)
<br><br/>
```python
#iris데이터프레임(x_train을 이용한) 만들기
iris_df=pd.DataFrame(x_train,columns=iris_dataset.feature_names)

#넘파이로 데이터의 산전도행렬 표현
pd.plotting.scatter_matrix(iris_df,c=y_train,figsize=(15,15),marker='o',hist_kwds={'bins':20},s=60,alpha=0.8,cmap=mglearn.cm3)
plt.show()
```
before
![KakaoTalk_20210311_052505906](https://user-images.githubusercontent.com/64114699/110752020-066e7f00-8288-11eb-80f5-0a3cfd0eb53f.png)
after
![KakaoTalk_20210311_165033245](https://user-images.githubusercontent.com/64114699/110756009-fb6a1d80-828c-11eb-87b0-e453ac7c8182.png)
<br><br/>

```python
print('predict:',guesses)
print('predict name:',iris_dataset['target_names'][guesses])

#테스트 데이터의 정확도 평가하기
print('test_x accuracy :',knn.score(x_test,y_test))
```
before
![KakaoTalk_20210311_164413523](https://user-images.githubusercontent.com/64114699/110752942-3ff3ba00-8289-11eb-83d5-89ef8d9c07b7.png)
after
![KakaoTalk_20210311_171035816](https://user-images.githubusercontent.com/64114699/110755822-bb0a9f80-828c-11eb-8d95-65e6ab329039.png)
<br><br/>

 ```python
 #k값의 변동에 따른 모델의 정확도 시각화
k_range=range(1,101)
accuracy=list()

for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    accuracy.append(knn.score(x_test,y_test))

plt.plot(k_range,accuracy)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.show()
```

before

![KakaoTalk_20210311_052416132](https://user-images.githubusercontent.com/64114699/110753672-33bc2c80-828a-11eb-8d1b-d3c1787418dd.png)

after

![KakaoTalk_20210311_165043171](https://user-images.githubusercontent.com/64114699/110756125-248aae00-828d-11eb-96e7-40102f471400.png)
<br><br/>
<br><br/>

---
## (2주차)Wisconsin Breast Cancer 기계학습

순서
1.  make_merge()를 사용한 데이터 생성 및 기계학습
2.  make_wave()를 사용한 데이터 생성 및 도표생성
3.  Wisconsin Breast Cancer Data를 사용한 기계학습 

### 1.make_merge()를 사용한 데이터 생성 및 기계학습

```python
#임의의 점 데이터를 생성하는 make_forge()
data,label=mglearn.datasets.make_forge()

#train, test 데이터를 나눔
data_train,data_test,label_train,label_test=train_test_split(data,label,random_state=0)

#data모양을 확인
print(data[:5])
data_frame=pd.DataFrame(data,label)
print(type(data))
print(data_frame.head())
print(label[0:5])
```
![KakaoTalk_20210313_205211444](https://user-images.githubusercontent.com/64114699/111029115-36f81980-843e-11eb-9d58-7626735dd930.png)
<br><br/>
```python
#데이터프레임화 하여 산점도 표현
data_class_0=data_frame.loc[0]
data_class_1=data_frame.loc[1]
label_c=['red' if y==1 else 'blue' for y in label]
class_0=plt.scatter(data_class_0[0],data_class_0[1],c='red',marker='o',alpha=0.5)
class_1=plt.scatter(data_class_1[0],data_class_1[1],c='blue',marker='o',alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(handles=(class_0,class_1),labels=('class0','class1'),loc='best')
plt.show()
plt.clf()
```
![KakaoTalk_20210313_205440968](https://user-images.githubusercontent.com/64114699/111029147-5d1db980-843e-11eb-87fc-a89f1e022d3f.png)
<br><br/>

```python
# knn으로 분류 후 정확도 확인
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(data_train,label_train)
# print(data[:5]) #float형 임의의 데이터
# print(label[:5]) #1,0으로 된 라벨
print('predict ans:',label_test)
print('predict test:',knn.predict(data_test))
print('predict score:',knn.score(data_test,label_test))
```
![KakaoTalk_20210313_205649135](https://user-images.githubusercontent.com/64114699/111029198-aa9a2680-843e-11eb-9f84-af601f7bdbbe.png)
<br><br/>
### 2.make_wave()를 사용한 데이터 생성 및 도표생성
```python
#임의의 점 데이터를 생성하는 make_wave()
data_x,data_y=mglearn.datasets.make_wave(n_samples=40)

# 데이터의 분포를 산점도로 확인
# data_frame=pd.DataFrame(data_x)
# print(data_frame.shape)
plt.scatter(data_x,data_y,marker='o',alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```
![KakaoTalk_20210313_205600095](https://user-images.githubusercontent.com/64114699/111029660-2ac18b80-8441-11eb-8138-719b9f1b7b17.png)
<br><br/>

### 3.Wisconsin Breast Cancer Data를 사용한 기계학습 

```python
#Wisconsin Breast Cancer Data
cancer=load_breast_cancer()

C_data=cancer.data
C_target=cancer.target

# #데이터의 모양을 확인 569x30
print(C_data.shape,C_target.shape)

#데이터프레임화
cancer_df=pd.DataFrame(C_data,C_target,columns=cancer.feature_names)
cancer_df_0=cancer_df.loc[0]
cancer_df_1=cancer_df.loc[1]

#인덱스 초기화
cancer_df_0=cancer_df_0.reset_index(drop=True)
cancer_df_1=cancer_df_1.reset_index(drop=True)

#행열 바꾸기
cancer_df_0=cancer_df_0.transpose()
cancer_df_1=cancer_df_1.transpose()

#데이터의 분포를 plot으로 확인
# print(cancer_df_0)
# print(cancer_df_1.shape)
plot_0=plt.plot(cancer_df_0,c='blue',alpha=0.2)
plot_1=plt.plot(cancer_df_1,c='red',alpha=0.2)
plt.show()
```
![KakaoTalk_20210313_212624626](https://user-images.githubusercontent.com/64114699/111029935-dd461e00-8442-11eb-88cd-b76617c01962.png)
<br><br/>

```python
#train,test 로 나눔
c_data_train,c_data_test,c_target_train,c_target_test=train_test_split(C_data,C_target,random_state=1)

#knn fit
knn_c=KNeighborsClassifier(n_neighbors=3)
knn_c.fit(c_data_train,c_target_train)

#accuracy(k=3)
print('accuracy:',knn_c.score(c_data_test,c_target_test))
```
![KakaoTalk_20210313_212807873](https://user-images.githubusercontent.com/64114699/111029964-0f578000-8443-11eb-82e8-31b2f30fbcd7.png)
<br><br/>
```python
#신뢰도가 높은 k값 찾기
k_range=range(1,101)
accuracy_list_train=list()
accuracy_list_test=list()
for k in k_range:
    knn_c=KNeighborsClassifier(n_neighbors=k)
    knn_c.fit(c_data_train,c_target_train)
    accuracy_list_train.append(knn_c.score(c_data_train,c_target_train))
    accuracy_list_test.append(knn_c.score(c_data_test,c_target_test))

c_test_plot,=plt.plot(k_range,accuracy_list_test,c='red')    
c_train_plot,=plt.plot(k_range,accuracy_list_train,c='blue')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.legend(handles=(c_test_plot,c_train_plot),labels=('test_ac','train_ac'),loc='best')
plt.show()
```
![KakaoTalk_20210313_212906363](https://user-images.githubusercontent.com/64114699/111029983-2c8c4e80-8443-11eb-9211-0f139db8fc11.png)
<br><br/>
2주차 총평: 내가 기계학습을 공부하는건지 pandas를 공부하는건지

---
## (3주차)KNN회귀분석,선형회귀분석
이전처럼 readme를 작성하는건 py파일을 올렸을경우에 어디서 어디까지 코드가 실행된건지 모르는 상황이 자주 발생하므로 유용했지만,
ipynb파일을 git에 업로드 하면 해당 문제점이 해결되는 부분이므로 굳이 코드를 찍어 올리는 고생을 할 필요가 없어졌다.
따라서 3주차부터는 코드를 작성하면서 막혔던 부분만 리뷰하는 형식으로 진행하려고 한다.

```python
#훈련용,테스트용 데이터셋으로 분류
Input_train,Input_test,Output_train,Output_test=train_test_split(Input,Output,random_state=0)

#K의 수를 3으로 하고 기계학습
knn_R=KNeighborsRegressor(n_neighbors=3)
knn_R.fit(Input_train,Output_train)

#테스트셋 예측 및 도표화
print('test_1 Output:',Output_test)
Predict=knn_R.predict(Input_test)
print('test_1 predict:',Predict)
print('test_1 score:',knn_R.score(Input_test,Output_test))
zero_y=[-3]*len(Input_test) #테스트셋의 인풋을 표현하기 위해 만든 y label
data_p=plt.scatter(Input,Output,marker='o',alpha=0.5)
predict_p=plt.scatter(Input_test,Predict,c='red',marker="*",alpha=0.5)
input_p=plt.scatter(Input_test,zero_y,c='green',marker="*",alpha=0.5)
plt.legend(handles=(data_p,predict_p,input_p),labels=('data','predict','test input'))
plt.xlabel('Input')
plt.ylabel('Output')
plt.show()
```
zero_y라는 리스트는 np.linspace(시작,끝,갯수)를 사용하면 더 간편하게 구현이 가능
plt.scatter함수에서 c='red'라고 할 필요없이 c='r'로 간편하게 구현이 가능
plt.plot함수에는 plot의 색,마커의 모양,선의 모양을 한번에 설정할 수 있는데 scatter에는 해당 기능이 없는것같다.
<br><br/>

```python
def make_kn_p(K,N=40):
    Dataset=mglearn.datasets.make_wave(n_samples=N)
    Input,Output=Dataset
    Input_train,Input_test,Output_train,Output_test=train_test_split(Input,Output,random_state=42)
    knn_R=KNeighborsRegressor(n_neighbors=K)
    knn_R.fit(Input_train,Output_train)
    knn_test_score=knn_R.score(Input_test,Output_test)
    knn_train_score=knn_R.score(Input_train,Output_train)
    title='KNeighborsRegressor/'+'train score :'+'%.2f'%knn_train_score+', test score :'+'%.2f'%knn_test_score
    plt.title(title)
    #선으로 도표그리기
    line=np.linspace(-3,3,1000).reshape(-1,1)
    A=plt.scatter(Input_train,Output_train,c='b',marker='^',alpha=0.5)
    B=plt.scatter(Input_test,Output_test,c='r',marker='v',alpha=0.5)
    C,=plt.plot(line,knn_R.predict(line),c='g')
    #plot은 ,을 붙어야 범례가 붙음.
    plt.legend(handles=(A,B,C),labels=(['tarin','test','predict']))
    plt.show()
 ```
plt.legend 함수는 항상 handles와 labels를 구별하여 붙일것, plot은 해당 plot 객체에','을 붙어야 범례가 붙음.
<br><br/>

```python
#---------------------------------
#선형회귀

#데이터셋을 만듬
Dataset=mglearn.datasets.make_wave(n_samples=60)
Input,Output=Dataset

#훈련용,테스트용으로 분류
Input_train,Input_test,Output_train,Output_test=train_test_split(Input,Output,random_state=42)

#선형회귀로 훈련
lr=LinearRegression().fit(Input_train,Output_train)

#score확인
print('train score:',lr.score(Input_train,Output_train))
print('test score:',lr.score(Input_test,Output_test))

#선형회귀 일차방정식의 기울기와 절편
print('lr.coef_:',lr.coef_)#기울기
print('lr.intercept_:',lr.intercept_)#절편
```
선형회귀의 기울기 피라미터는 lr 객체의 coef_ 속성에 저장되어있다.+절편 피라미터는 intercept_속성에 저장되어있다.
plt으로 도표를 만들때 scatter은 x값에 (N,)꼴이 들어가도 상관없지만 plot은 오류를 일으키므로 둘다 항상 (N,1)형태로 전환해주도록 하자
(N,1)형태로 전환하는 방법은 numpy의 reshape 함수를 이용하면 간단하다. 
ex.코드
```python
#보스턴 데이터에서 방의 개수로 산점도 그래프 작성
dot1=plt.scatter(boston_df['RM'],Output,c='r',alpha=0.5)


print(boston_df['RM'].shape)
#pandas series 타입임
print(type(boston_df['RM']))

#RM 데이터가 1행에 다 들어있으므로 열로 바꿔줘야됨
#reshape 함수를 사용하기 위해 넘파이 형식으로 바꿔줌
new_x=np.reshape(boston_df['RM'].to_numpy(),(-1,1))
print(new_x.shape)

#훈련용, 테스트용 분류
Input_train,Input_test,Output_train,Output_test=train_test_split(new_x,Output,random_state=0)
```
reshape 함수에서 (-1,1) -1은 본래 기준이 되었던 리스트의 길이에 따라 1개짜리 (N,1)리스트가 생성된다는 뜻
<br><br/>
데이터를 가공할때 우선 pd.DataFrame()형태로 가공하면 보기 편하다. 
+df는 'columns='를 사용해 열의 이름을 쉽게 바꿀수 있다.
pd.DataFrame(data,label,colums='feature')형식으로 만들면 좋음
이렇게 가공된 df는 라벨에 따라 분류하기도 편해짐 df=df.loc[0] 이란 소리는 해당 df에서 0인 라벨만 골라서 만들어진 df라는 뜻이다.
이렇게 각각 뽑아버리면 나중에 도표로 만들때도 편해짐
ex.
```python
#붙인라벨에 따라서 df 하나씩 생성
temp_df_0=temp_df.loc[0]
temp_df_1=temp_df.loc[1]
```
+이렇게 만들어진 df는 원하는 column만 따로 뺄수도 있음 
ex.
```python
iris_data_pl=iris_df['petal length (cm)']
```
<br><br/>
로지스틱 회귀분석에서 OUTPUT을 0,1로 만들어야 할때 해당 리스트가 numpy array이면 OUTPUT=npArray==value 형식으로 표현하면 된다.
ex.
```python
#---------------------------------
#로지스틱 회귀분석(분류)
iris=datasets.load_iris()

print(iris.keys())
print(iris.feature_names)
print(iris.target_names)
#1가지 특성만 가져옴 (petal length를 가져오기로 함)
iris_df=pd.DataFrame(iris.data,columns=iris.feature_names)
print(iris_df.head())
iris_data_pl=iris_df['petal length (cm)']
print(iris_data_pl.head())
#로지스틱 회귀분석을 위해 1개의 output=virginica를 재외하고 0으로 만듬
print(type(iris.target))
OutPut=iris.target==2
print(OutPut)
```
<br><br/>
솔레노이드 곡선을 만들기 위해 로지스틱 회귀분류의 .predict_proba(x) 함수가 필요하다.
이때 해당 함수로 만들어진 예측값은 (N,2)의 형태를 띄고 있는데
col=0은 1-p값이고(예측이 틀릴 확률)
col=1은 p값이다(예측이 맞을 확률)
ex.
```python
#꽃잎의 넓이에 대해 로지스틱 회귀모델을 이용해 추정

log_reg=LogisticRegression().fit(iris_data_pl,OutPut)
Input_pl=np.linspace(min(iris_data_pl),max(iris_data_pl),1000).reshape(-1,1)
#각input에 대한 확률을 예측
p=log_reg.predict_proba(Input_pl)
print(p.shape)
#0열은 False, 1열은 True 라고 생각
print(pd.DataFrame(p).head())
p=pd.DataFrame(p)
#도표화

plt.scatter(iris_data_pl,OutPut,c='g',alpha=0.5,label='Yes or No')
plt.plot(Input_pl,p[0],label='Non virginica')
plt.plot(Input_pl,p[1],label='virginica')
plt.ylabel('probability')
plt.xlabel('petal width')
plt.legend()
plt.show()
```
<br><br/>
도표를 분리하기 위한 함수는 
mglearn.plots.plot_2d_separator(사용한 모델,모델의 train data ,fill=True,alpha=0.2)#분류모델,train_input,채움,투명도 이다
이다.

---
## (4주차)Multi_class,나이브베이즈,결정트리,렌덤포레스트,부스팅
많은 구현
넘파이에선 해당 리스트에서 바로 []을 통해 원하는 부분 리스트를 만들어 낼수 있다.
판다스에선 .loc[] 함수를 사용하여 동일한 기능을 사용할 수 있다.
[행범위,열범위] 이고 범위는 :을 사용하여 표현한다.
예시)
```python
#각각의 클래스가 군집화된 데이터를 만들어낼수 있는 make_blobs
Data,label=make_blobs(random_state=42)
#Data의 형태가 100x2 이고 label은 0,1,2 이다.
#따라서 Data의 0열이 x, 1열이 y축인 3라벨 데이터이다.
print(Data.shape)
print(Data[:5])
print(type(label))
#넘파이 부분 집합 narray[행,열] 로 구분할수 있음. 범위는 ':'으로 구분함 
#ex. 2차원 행렬의 1열만 부분행렬로 가져오고 싶으면 narray[:,1] 이다. :에 아무것도 없을경우 전체가 범위가 됨
print(Data[:5,1].shape)
#`elif` in list comprehension conditionals
c_map=['r' if l==0 else 'g' if l==1 else 'b' for l in label]
plt.scatter(Data[:,0],Data[:,1],c=c_map,alpha=0.5)
plt.legend(['class 0','class 1','class 2'])
plt.show()
```
Data[:,0] 은 모든 행범위에서 0열만 가져온다는 것이다.
<br><br/>
로지스틱 회귀에서 다수의 클래스(N개)가 존재할 경우 coef_에 N개의 행만큼 법선백터가 저장된다. 
예시)
![캡처](https://user-images.githubusercontent.com/64114699/112708511-b503f800-8ef5-11eb-8ca3-6e71c18489db.JPG)
```python
#계수의 형태가 (3,2)인 이유는 클래스가 3개이고 각 클래스의 법선백터(x,y)이기 때문이다.
print('계수(기울기) 배열의 크기 :',logr.coef_.shape)
print(logr.coef_)
print('절편 배열의 크기 :',logr.intercept_.shape)
x=np.linspace(min(Data[:,0]),max(Data[:,0]))
for coef,intercept,color in zip(logr.coef_,logr.intercept_,['r','g','b']):
    x_n_vec=coef[0]
    y_n_vec=coef[1]
    plt.plot(x,-(x*x_n_vec+intercept)/y_n_vec,c=color)
plt.show()
```
<br><br/>
데이터의 변화량이 급격할 경우 로그스케일로 데이터를 변환시키면 변화량이 잘 표현된다.
semilogy 함수를 사용하면 데이터 자체를 로그화 시키지 않아도 로그화된 데이터를 표현해준다.
하지만 해당 함수의 변수가 로그화된 것이므로 표에 새로운 그래프를 그릴경우 해당 데이터와의 격차를 생각해봐야 함.
예시)
```python
#렘 가격 동향
import os

ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, "ram_price.csv"))
#로그스케일로 램가격 동향 도표화
plt.semilogy(ram_prices['date'],ram_prices['price'])

#을 선형회귀와 결정트리회귀로 비교
#훈련 셋은 2000년 이전,테스트 셋은 2000년 이후로 결정
data_train=ram_prices[ram_prices['date']<2000]
data_test=ram_prices[ram_prices['date']>=2000]

#모양을 (N,1)꼴로 하였음 fit에서 2차원 배열을 요구함
Input_train=data_train['date'].to_numpy().reshape(-1,1)
#원래 데이터는 급격하게 줄어드는데 이걸 보기 편하게 하기 위해 로그스케일로 바꿈
Output_train=np.log(data_train['price'].to_numpy().reshape(-1,1))
print(Input_train[:5])
print(Output_train[:5])
#선형
lr=LinearRegression().fit(Input_train,Output_train)

#트리
tree=DecisionTreeRegressor().fit(Input_train,Output_train)

#예측
pred_tree=tree.predict(ram_prices['date'].to_numpy().reshape(-1,1))
pred_lr=lr.predict(ram_prices['date'].to_numpy().reshape(-1,1))

#바꾼 로그스케일을 되돌림
pred_tree=np.exp(pred_tree)
pred_lr=np.exp(pred_lr)
```
<br><br/>
교차검증은 cross_val_score(모델,데이터,라벨,레이어) 함수를 사용하여 쉽게 적용시킬 수 있다.
예시)
```python
#렌덤포레스트를 이용한 moons 데이터셋 분류
Data,label=datasets.make_moons(n_samples=100,noise=0.25,random_state=3)

#훈련,테스트용으로 나눔
Input_train,Input_test,Output_train,Output_test=train_test_split(Data,label,random_state=42)

#포레스트 모델로 훈련 (n_estimators : 생성할 tree의 개수,max_features : 최대 선택할 특성의 수)
forest=RandomForestClassifier(n_estimators=5,random_state=2).fit(Input_train,Output_train)

#테스트 스코어
print('test score:',round(forest.score(Input_test,Output_test),3))

#교차검증
cvscore=cross_val_score(forest,Data,label,cv=5)#교차검증 cv=5이므로 5겹 분할 교차검증,리턴값은 리스트
print('(cv 5):',cvscore)
print('test score(cv 5):',round(cvscore.mean(),3))

cvscore=cross_val_score(forest,Data,label,cv=10)
print('(cv 10):',cvscore)
print('test score(cv 10):',round(cvscore.mean(),3))
```

<br><br/>
subplots() 과  subplot() 의 차이
subplot()은 하나의 ax만 만들수 있다. ax = plt.subplot(행,열,번호) 즉 N행M열의 서브플롯이 만들어지고 그중에 I번의 서브플롯이 ax라는 뜻

subplots()은 fig, ax = plt.subplots() 형식으로사용하고 fig=서브플롯 전체 크기를 나타내고 figsize=(x,y) 변수로 비율을 설정할수 있음
ax는 각각의 서브플롯을 의미한다.
subplots() 예시)
![캡처](https://user-images.githubusercontent.com/64114699/112708894-4d9b7780-8ef8-11eb-9335-7470f385ce6a.JPG)
```python
# 데이터 로드
X, y = datasets.make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,random_state=42)

# 모델 학습
model = RandomForestClassifier(n_estimators=5, random_state=0)#트리 5개만 만듬
model.fit(X_train, y_train)

fig, axes = plt.subplots(2, 3, figsize=(20,10))#axes는 각각의 서브플롯 번호
print(fig)
for i,(ax,tree) in enumerate( zip( axes.ravel(), model.estimators_ )):#랜덤포레스트 내부 트리는 estimator_ 속성에 저장
    ax.set_title("tree"+str(i))
    mglearn.plots.plot_tree_partition(X, y, tree,ax=ax)
    
# 랜덤포레스트로 만들어진 결정경계
axes[-1, -1].set_title("Random forest")
mglearn.plots.plot_2d_separator(model, X, fill=True, alpha=0.5, ax=axes[-1,-1] )
mglearn.discrete_scatter(X[:,0], X[:,1], y)
```

---
## (5주차)부스팅,앙상블,SVM,신경망
3차원 그래픽을 표현할 경우 대략적인 과정
1.fig=plt.figure(figsize=(인치,인치))
2.fig.add_suplot()의 피라미터로 projection='3d' 입력
예시)
```python
#3차원 그래픽
fig=plt.figure(figsize=(8,8))

#111은 1x1 그리드의 1번째 플롯이라는 뜻
#234는 2x3 그리드의 4번째 플롯이라는 뜻
#3차원:projection='3d'
ax=fig.add_subplot(111,projection='3d')

#데이터 가공
blob_df_3d=pd.DataFrame(new_data,label)

class0=blob_df_3d.loc[0]
class1=blob_df_3d.loc[1]

#도표화
cs0=ax.scatter(class0.loc[:,0],class0.loc[:,1],class0.loc[:,2],c='b',s=50)
cs1=ax.scatter(class1.loc[:,0],class1.loc[:,1],class1.loc[:,2],c='r',marker='^',s=50)

#Invert Axes 는 set_?lim()함수를 사용하면 된다. set_zlim(140,0)= z축을 140부터 시작하여 0까지 가도록 함
ax.set_zlim(140,0)

plt.show()
```
<br><br/>
3차원 시각화의 경우 시점을 조절해야 할 필요가 생기는데 이때 azim과 elev를 사용한다
plt.azim=value는 3차원 공간을 z축을 기준으로 좌우로 돌린다.
plt.elev=value는 3차원 공간을 상하로 돌린다.
예시)
```python
#3차원 그래픽
fig=plt.figure(figsize=(8,8))
ax=fig.add_subplot(111,projection='3d')

#데이터 가공
blob_df_3d=pd.DataFrame(new_data,label)

class0=blob_df_3d.loc[0]
class1=blob_df_3d.loc[1]

#도표화
cs0=ax.scatter(class0.loc[:,0],class0.loc[:,1],class0.loc[:,2],c='b',s=50)
cs1=ax.scatter(class1.loc[:,0],class1.loc[:,1],class1.loc[:,2],c='r',marker='^',s=50)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z=y^2')
#Invert Axes 는 set_?lim()함수를 사용하면 된다. set_zlim(140,0)= z축을 140부터 시작하여 0까지 가도록 함
ax.set_zlim(140,0)

#3차원 투영된 도표를 분류

lin_svm_3d = LinearSVC().fit(blob_df_3d, label)
coef, intercept = lin_svm_3d.coef_.ravel(),lin_svm_3d.intercept_
#법선벡터가 하나 들어있음 (x,y,z)
print(coef)
print(intercept)

#x좌표와 y좌표를 만듬
x = np.linspace(blob_df_3d.loc[:, 0].min() - 2, blob_df_3d.loc[:, 0].max() + 2, 50)
y = np.linspace(blob_df_3d.loc[:, 1].min() - 2, blob_df_3d.loc[:, 1].max() + 2, 50)
#격자점을 만들어주는 함수 .meshgrid
X, Y = np.meshgrid(x,y)
print(X.shape)

#격자점에 따라서 Z좌표를 만듬 (법선벡터가 존재하므로 평방에 대입시켜서 만들수 있음)
Z=(coef[0]*X+coef[1]*Y+intercept)/(-coef[2])

#X,Y,Z 좌표가 만들어졌으므로 평면을 그릴수 있음
surf=ax.plot_surface(X,Y,Z ,alpha=0.5)

#azim 3차원 공간을 좌우로 돌림, elev 위아래로 돌림
ax.azim = 35
ax.elev = 10

#나누어진것을 확인할 수 있다.
plt.show()
```
plt로 subplots 할 경우 ax에다가 직접 입력
mglearn은 ax피라미터 이용
예시)
```python
#은닉 유닛과 alpha 매개변수의 변화에 따른 결정경계
Input_train,Input_test,Output_train,Output_test=train_test_split(data,label,random_state=42)
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for axx, n_hidden_nodes in zip(axes, [10, 100]):
    for ax, alpha in zip(axx, [0.0001, 0.01, 0.1, 1]):
        #ax는 plt이다.
        cs0=ax.scatter(class0.loc[:,0], class0.loc[:, 1],c='b',label='class0',alpha=0.5)
        cs1=ax.scatter(class1.loc[:,0], class1.loc[:, 1],c='r',label='class1',alpha=0.5)
        #다중퍼셉트론의 구현
        mlp = MLPClassifier(solver='lbfgs',hidden_layer_sizes=[n_hidden_nodes,n_hidden_nodes], random_state=0,alpha=alpha).fit(Input_train,Output_train)
        mglearn.plots.plot_2d_separator(mlp, Input_train, fill=True, alpha=.3,ax=ax)

        ax.legend()
        ax.set_title("n_hidden=[{}, {}]\nalpha={:.4f}".format(n_hidden_nodes, n_hidden_nodes, alpha))
```
<br><br/>
---
## (6주차)비지도학습,데이터전처리,클러스터링
비지도학습의 정의 :비지도 학습이란 알고있는 출력값이나 정보 없이 학습 알고리즘을 가르쳐야 하는 종류의 머신러닝
< 비지도 학습의 종류 >
1. 비지도 변환 (unsupervised transformation)
2. 군집 (clustering)

비지도 변환 (unsupervised transformation)-
데이터를 새롭게 표현하여 사람이나 다른 머신러닝 알고리즘이 원래 데이터보다 쉽게 해석할 수 있도록 만드는 알고리즘.
비지도 변환이 널리 사용되는 분야는 특성이 고차원 데이터를 특성의 수를 줄이면서 꼭 필요한 특징을 포함한 데이터로 표현하는 방법인 차원축소다.
차원 축소의 대표적 예는 시각화를 위해 데이터셋을 2차원으로 변경하는 경우이다.
데이터를 구성하는 단위나 성분을 찾기도 한다.
(예를들어 많은 텍스트 문서에서 주제를 추출하는 것)
PCA,NMF,tSNE

군집 (clustering)-
데이터를 비슷한 것끼리 그룹으로 묶는 것 
비지도 학습에서 가장 어려운 일은 알고리즘이 유용한 것을 학습했는지 평가해야 하는것이다.
따라서 비지도 학습 알고리즘은 데이터 과학자가 데이터를 더 잘 이해하고 싶을때 탐색적 분석단계로 많이 사용한다.
또한 지도 삭습의 전처리 단계에서도 많이 사용한다.
K-means,병합군집,DBSCAN
군집평가(ARI)

