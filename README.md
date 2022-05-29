# Naive Bayes Classifier

What is Naive Bayes algorithm: 
It is a supervised learning method based on __Bayes theorem__ which assumes that all features (inputs) that we use to predict the target value are __mutual independent__.

This is a strong assumption that is not always applicable in real life dataset ,but we assume it because __it make our calculations way much simpler yet efficient__.

Bayes' Theorem provides a way to calculate the probability of the data belonging to a class (target) giving a prior knowledge (features or input) which can be stated as:

__P(class|data) = (P(data|class) * P(class)) / P(data)__

where P(class|data) is the probability of a class giving a provided data.

Naive Bayes is a classification algorithm used for binary and multiclass classification problems

# Heart Disease Dataset

- our target here to predict the presence of heart disease in the patient which will help us to deal with the disease in early phase


- So this is a binary classification problem, and our target class value will expect two values:-
    - 1 for the presence of heart disease (disease)
    - 0 for the patient is on the safe area (no disease)
    
    
- in our dataset `heart.csv` we have 1025 row with no missing data includes:
    - 13 columns (input or feature)
        1. age
        2. sex
        3. chest pain type (4 values)
        4. resting blood pressure
        5. serum cholestoral in mg/dl
        6. asting blood sugar > 120 mg/dl
        7. resting electrocardiographic results (values 0,1,2)
        8. maximum heart rate achieved
        9. exercise induced angina
        10. oldpeak = ST depression induced by exercise relative to rest
        11. the slope of the peak exercise ST segment
        12. number of major vessels (0-3) colored by flourosopy
        13. thal: 0 = normal; 1 = fixed defect; 2 = reversable defect
    
    - 1 output column (target)

# Our Approach To Solve The Problem

the solution for this problem is broken down to 4 steps::
1. __getting our data ready to be processed__
2. __get some statistics for each column separated by class (0/no disease and 1/disease)__
3. __calculate Probability for each input row__
4. __calculate the class Probability (the model prediction)__

## step 1: getting our data ready to be processed


before any thing we are going to import numpy and pandas to help us in our calculations



```python
import numpy as np
import pandas as pd

# read heart.csv file and put it into pandas dataframe
heart_disease = pd.read_csv('heart.csv')
heart_disease

x = heart_disease
y = heart_disease["target"]

x
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>52</td>
      <td>1</td>
      <td>0</td>
      <td>125</td>
      <td>212</td>
      <td>0</td>
      <td>1</td>
      <td>168</td>
      <td>0</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>53</td>
      <td>1</td>
      <td>0</td>
      <td>140</td>
      <td>203</td>
      <td>1</td>
      <td>0</td>
      <td>155</td>
      <td>1</td>
      <td>3.1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>70</td>
      <td>1</td>
      <td>0</td>
      <td>145</td>
      <td>174</td>
      <td>0</td>
      <td>1</td>
      <td>125</td>
      <td>1</td>
      <td>2.6</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>61</td>
      <td>1</td>
      <td>0</td>
      <td>148</td>
      <td>203</td>
      <td>0</td>
      <td>1</td>
      <td>161</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>62</td>
      <td>0</td>
      <td>0</td>
      <td>138</td>
      <td>294</td>
      <td>1</td>
      <td>1</td>
      <td>106</td>
      <td>0</td>
      <td>1.9</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1020</th>
      <td>59</td>
      <td>1</td>
      <td>1</td>
      <td>140</td>
      <td>221</td>
      <td>0</td>
      <td>1</td>
      <td>164</td>
      <td>1</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1021</th>
      <td>60</td>
      <td>1</td>
      <td>0</td>
      <td>125</td>
      <td>258</td>
      <td>0</td>
      <td>0</td>
      <td>141</td>
      <td>1</td>
      <td>2.8</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1022</th>
      <td>47</td>
      <td>1</td>
      <td>0</td>
      <td>110</td>
      <td>275</td>
      <td>0</td>
      <td>0</td>
      <td>118</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1023</th>
      <td>50</td>
      <td>0</td>
      <td>0</td>
      <td>110</td>
      <td>254</td>
      <td>0</td>
      <td>0</td>
      <td>159</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1024</th>
      <td>54</td>
      <td>1</td>
      <td>0</td>
      <td>120</td>
      <td>188</td>
      <td>0</td>
      <td>1</td>
      <td>113</td>
      <td>0</td>
      <td>1.4</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1025 rows × 14 columns</p>
</div>




```python
#separate the dataset to training group and test group
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2)
```


```python
# separate our data by target class value
diseased = x_train[x_train['target'] == 1].drop('target', axis=1)
not_diseased = x_train[x_train['target'] == 0].drop('target', axis=1)
```

all not diseased entries


```python
diseased
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>831</th>
      <td>58</td>
      <td>1</td>
      <td>1</td>
      <td>125</td>
      <td>220</td>
      <td>0</td>
      <td>1</td>
      <td>144</td>
      <td>0</td>
      <td>0.4</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>982</th>
      <td>67</td>
      <td>0</td>
      <td>0</td>
      <td>106</td>
      <td>223</td>
      <td>0</td>
      <td>1</td>
      <td>142</td>
      <td>0</td>
      <td>0.3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>663</th>
      <td>58</td>
      <td>0</td>
      <td>0</td>
      <td>100</td>
      <td>248</td>
      <td>0</td>
      <td>0</td>
      <td>122</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>639</th>
      <td>58</td>
      <td>0</td>
      <td>0</td>
      <td>130</td>
      <td>197</td>
      <td>0</td>
      <td>1</td>
      <td>131</td>
      <td>0</td>
      <td>0.6</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>249</th>
      <td>42</td>
      <td>1</td>
      <td>2</td>
      <td>130</td>
      <td>180</td>
      <td>0</td>
      <td>1</td>
      <td>150</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>782</th>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>130</td>
      <td>303</td>
      <td>0</td>
      <td>1</td>
      <td>122</td>
      <td>0</td>
      <td>2.0</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>932</th>
      <td>51</td>
      <td>0</td>
      <td>2</td>
      <td>140</td>
      <td>308</td>
      <td>0</td>
      <td>0</td>
      <td>142</td>
      <td>0</td>
      <td>1.5</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>745</th>
      <td>51</td>
      <td>1</td>
      <td>2</td>
      <td>100</td>
      <td>222</td>
      <td>0</td>
      <td>1</td>
      <td>143</td>
      <td>1</td>
      <td>1.2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>648</th>
      <td>71</td>
      <td>0</td>
      <td>0</td>
      <td>112</td>
      <td>149</td>
      <td>0</td>
      <td>1</td>
      <td>125</td>
      <td>0</td>
      <td>1.6</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>666</th>
      <td>35</td>
      <td>1</td>
      <td>1</td>
      <td>122</td>
      <td>192</td>
      <td>0</td>
      <td>1</td>
      <td>174</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>435 rows × 13 columns</p>
</div>



all diseased entries


```python
not_diseased
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>454</th>
      <td>65</td>
      <td>0</td>
      <td>0</td>
      <td>150</td>
      <td>225</td>
      <td>0</td>
      <td>0</td>
      <td>114</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>900</th>
      <td>61</td>
      <td>1</td>
      <td>3</td>
      <td>134</td>
      <td>234</td>
      <td>0</td>
      <td>1</td>
      <td>145</td>
      <td>0</td>
      <td>2.6</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>844</th>
      <td>60</td>
      <td>1</td>
      <td>0</td>
      <td>140</td>
      <td>293</td>
      <td>0</td>
      <td>0</td>
      <td>170</td>
      <td>0</td>
      <td>1.2</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>629</th>
      <td>65</td>
      <td>1</td>
      <td>3</td>
      <td>138</td>
      <td>282</td>
      <td>1</td>
      <td>0</td>
      <td>174</td>
      <td>0</td>
      <td>1.4</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>32</th>
      <td>57</td>
      <td>1</td>
      <td>0</td>
      <td>130</td>
      <td>131</td>
      <td>0</td>
      <td>1</td>
      <td>115</td>
      <td>1</td>
      <td>1.2</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>706</th>
      <td>57</td>
      <td>1</td>
      <td>2</td>
      <td>128</td>
      <td>229</td>
      <td>0</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>0.4</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>634</th>
      <td>52</td>
      <td>1</td>
      <td>0</td>
      <td>125</td>
      <td>212</td>
      <td>0</td>
      <td>1</td>
      <td>168</td>
      <td>0</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>59</th>
      <td>57</td>
      <td>1</td>
      <td>1</td>
      <td>154</td>
      <td>232</td>
      <td>0</td>
      <td>0</td>
      <td>164</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>787</th>
      <td>51</td>
      <td>1</td>
      <td>0</td>
      <td>140</td>
      <td>298</td>
      <td>0</td>
      <td>1</td>
      <td>122</td>
      <td>1</td>
      <td>4.2</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>171</th>
      <td>56</td>
      <td>1</td>
      <td>0</td>
      <td>130</td>
      <td>283</td>
      <td>1</td>
      <td>0</td>
      <td>103</td>
      <td>1</td>
      <td>1.6</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>385 rows × 13 columns</p>
</div>



## step 2: get some statistics for each column separated by class

in this step we are going to calculate the mean and the standard diviation for each column for the two dataframes.


these statistics will help us in calculating the wanted probabilities.


```python
# calculate the mean and the standard diviation for each columns in diseased datafram and put the result in list
diseased_statistics = np.array([(diseased[column].mean(), diseased[column].std()) for column in diseased.columns])
```


```python
diseased_statistics
```




    array([[5.23701149e+01, 9.60551535e+00],
           [5.67816092e-01, 4.95950015e-01],
           [1.34022989e+00, 9.32819393e-01],
           [1.29119540e+02, 1.60590632e+01],
           [2.40678161e+02, 5.35576898e+01],
           [1.33333333e-01, 3.40326039e-01],
           [6.04597701e-01, 4.98825359e-01],
           [1.58340230e+02, 1.91002477e+01],
           [1.37931034e-01, 3.45224624e-01],
           [5.78620690e-01, 7.79328513e-01],
           [1.60000000e+00, 5.84752498e-01],
           [3.74712644e-01, 8.85002850e-01],
           [2.10114943e+00, 4.69212069e-01]])




```python
# calculate the mean and the standard diviation for each columns in not_diseased datafram and put the result in list
not_diseased_statistics = np.array([(not_diseased[column].mean(), not_diseased[column].std()) for column in not_diseased.columns])
```


```python
not_diseased_statistics
```




    array([[5.66337662e+01, 8.02394870e+00],
           [8.33766234e-01, 3.72774783e-01],
           [4.90909091e-01, 9.10325866e-01],
           [1.33615584e+02, 1.85225866e+01],
           [2.48963636e+02, 4.96351661e+01],
           [1.63636364e-01, 3.70426658e-01],
           [4.31168831e-01, 5.31374922e-01],
           [1.38979221e+02, 2.21582750e+01],
           [5.45454545e-01, 4.98577522e-01],
           [1.58077922e+00, 1.26348741e+00],
           [1.17142857e+00, 5.74378546e-01],
           [1.19740260e+00, 1.04706609e+00],
           [2.50129870e+00, 6.96436896e-01]])



## step 3: calculate Probability for using Gaussian Probability Density Function


Calculating the probability of observing a given real-value is difficult.

We can assume that our values are drawn from a distribution to ease the probablity calculation, and we will use the Gaussian distribution becuase it can be calculate using the mean and the standard deviation that we collected from our data before

### Gaussian Probability Distribution Function:-

__f(x) = (1 / sqrt(2 * PI) * sigma) * exp(-((x-mean)^2 / (2 * sigma^2)))__



```python
# a function to calculate the probability of a given value according to a pre calculated mean and standard deviationdef calc_probability(x, mean, stdev):
def calc_probability(x, mean, stdev):
    exponent = np.exp(-((x-mean)**2 / (2 * stdev**2 )))
    return (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent
```


```python
calc_probability(0, 1, 1)
```




    0.24197072451914337



## step 4: calculate the class Probability (the model prediction)


in this step, we are going to calculate the probability of the input belonging to one of our classes by using our precalculated data (training data/mean and standard deviation for each column)



we will apply the previous step for every single row (input) so we will have the probability for each class giving the input data and since we are interested in class classification, not the actual probability we will consider the largest value



the following equation calculates the probability that a piece of data belongs to a class:-


__P(class|input) = P(input|class) * P(class)__  -> (1)



this equation is similar to Bayes Theorem but we removed the division as we are not interested in the probability and for calculation simplicity



the inputs features (columns of the row) are treated separately that's why it is called '__Naive__ Bayes' as we assumed variables are independent so equation (1) can be rewritten in the below form: -



__P(class|row) = P(input1|class) * P(input2|class) * P(input3|class) * ... * P(class)__


where 
- P(class|row) is the probability that a piece of data belongs to a class
- P(inputn|class) is the probability that a value for column giving that wanted class class
- P(class) is the probability of the class (will be calculated)



```python
def predict_class(row):
    # calculating every class probability giving the inpur row     
    total_rows = x_train.shape[0] 
    not_diseased_prob = not_diseased.shape[0] / total_rows
    diseased_prob = diseased.shape[0] / total_rows  
    classes_probs = { 'not_diseased': not_diseased_prob, 'diseased': diseased_prob }
#     for i in range(not_diseased_statistics.shape[0]):
    for i in range(not_diseased_statistics.shape[0]):
        [ mean, stdev ] = not_diseased_statistics[i]
        classes_probs['not_diseased'] *= calc_probability(row[i], mean, stdev)
    for i in range(diseased_statistics.shape[0]):
        [ mean, stdev ] = diseased_statistics[i]
        classes_probs['diseased'] *= calc_probability(row[i], mean, stdev)
    
    # predicting the class (the higher value is the target value class)
    return 1 if classes_probs['diseased'] > classes_probs['not_diseased'] else 0      
```


```python
# example 1 for diseased and 0 for not
predict_class([69,0,3,140,239,0,1,151,0,1.8,2,2,2])
```




    1




```python
# x = heart_disease
# y = heart_disease["target"]

# #separate the dataset to training group and test group
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2)
```


```python
not_diseased_statistics
```




    array([[5.66337662e+01, 8.02394870e+00],
           [8.33766234e-01, 3.72774783e-01],
           [4.90909091e-01, 9.10325866e-01],
           [1.33615584e+02, 1.85225866e+01],
           [2.48963636e+02, 4.96351661e+01],
           [1.63636364e-01, 3.70426658e-01],
           [4.31168831e-01, 5.31374922e-01],
           [1.38979221e+02, 2.21582750e+01],
           [5.45454545e-01, 4.98577522e-01],
           [1.58077922e+00, 1.26348741e+00],
           [1.17142857e+00, 5.74378546e-01],
           [1.19740260e+00, 1.04706609e+00],
           [2.50129870e+00, 6.96436896e-01]])




```python

# params: train => .8 of the data (target included)
#         test => .2 of the data (target not included)
def naive_bayes(test):
    # summarize = summarize_by_class(train)
    
    # 1- separating train data by class
#     diseased = train[train['target'] == 1].drop('target', axis=1)
#     not_diseased = train[train['target'] == 0].drop('target', axis=1)
    
    # 2- calculate statistics for each class
#     diseased_statistics = np.array([(diseased[column].mean(), diseased[column].std()) for column in diseased.columns])
#     not_diseased_statistics = np.array([(not_diseased[column].mean(), not_diseased[column].std()) for column in not_diseased.columns])

    # calculating not_diseased_prob and diseased_prob to pass it preict_input_class function
#     diseased_prob = diseased.shape[0] / train.shape[0]
#     not_diseased_prob = not_diseased.shape[0] / train.shape[0]
    
    # 3&4- calculate the probabilty for each row to predict the target 
    predictions = np.array([])
    for row in test:    
        output = predict_class(row)  
        predictions = np.append(predictions, [output])
    return predictions

```


```python
# naive_bayes(x_train, x_test.drop('target', axis=1).values).shape
preds = naive_bayes(x_test.drop('target', axis=1).values)
preds
```




    array([1., 0., 1., 1., 1., 0., 0., 1., 0., 1., 1., 0., 1., 1., 0., 0., 1.,
           0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 1.,
           1., 0., 1., 1., 0., 1., 1., 0., 0., 0., 1., 0., 1., 1., 0., 1., 1.,
           1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0.,
           0., 0., 1., 1., 0., 1., 1., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0.,
           1., 1., 1., 1., 0., 1., 1., 0., 1., 0., 1., 1., 1., 0., 0., 0., 1.,
           0., 1., 0., 0., 1., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 1., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 1., 1., 1., 0.,
           1., 1., 0., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0.,
           0., 1., 0., 1., 0., 1., 1., 0., 1., 0., 0., 1., 0., 1., 0., 0., 1.,
           1., 1., 1., 0., 1., 1., 1., 1., 0., 1., 1., 0., 1., 0., 0., 1., 0.,
           1., 0., 1., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 1., 1., 1.,
           1.])




```python
from sklearn.metrics import accuracy_score

accuracy_score(y_test.values, naive_bayes(x_test.drop('target', axis=1).values))

```




    0.8390243902439024



# Summerzing all in OOP approach


```python
def predict_class(row):
    # calculating every class probability giving the inpur row     
    total_rows = x_train.shape[0] 
    not_diseased_prob = not_diseased.shape[0] / total_rows
    diseased_prob = diseased.shape[0] / total_rows  
    classes_probs = { 'not_diseased': not_diseased_prob, 'diseased': diseased_prob }
#     for i in range(not_diseased_statistics.shape[0]):
    for i in range(not_diseased_statistics.shape[0]):
        [ mean, stdev ] = not_diseased_statistics[i]
        classes_probs['not_diseased'] *= calc_probability(row[i], mean, stdev)
    for i in range(diseased_statistics.shape[0]):
        [ mean, stdev ] = diseased_statistics[i]
        classes_probs['diseased'] *= calc_probability(row[i], mean, stdev)
    
    # predicting the class (the higher value is the target value class)
    return 1 if classes_probs['diseased'] > classes_probs['not_diseased'] else 0     
```


```python
class NaiveBayes:
    def __init__(self):
        self.diseased_data = np.array([])
        self.not_diseased_data = np.array([])
        self.diseased_statistics = np.array([])
        self.not_diseased_statistics = np.array([])
        self.train_rows_number = 0
        self.train_diseased_prob = 0 
        self.train_not_diseased_prob = 0 
    
    # takes the dataset as pandas datafram
    def fit(self, data):
        self.train_rows_number = data.shape[0]
        
        # 1- separating train data by class
        diseased = data[data['target'] == 1].drop('target', axis=1)
        not_diseased = data[data['target'] == 0].drop('target', axis=1) 
        
        self.train_diseased_prob = diseased.shape[0] / self.train_rows_number
        self.train_not_diseased_prob = not_diseased.shape[0] / self.train_rows_number
        
        # 2- calculate statistics for each class
        self.diseased_statistics = np.array([(diseased[column].mean(), diseased[column].std()) for column in diseased.columns])
        self.not_diseased_statistics = np.array([(not_diseased[column].mean(), not_diseased[column].std()) for column in not_diseased.columns])
    
        return [diseased, not_diseased, self.train_rows_number]
        
    # params: input: np.array for the input wanted to predict its class
    # return: 1 for diseased,
    #         0 for not diseased
#     def predict_input_class(self, input, not_diseases_stats, diseases_stats):
    def predict_input_class(self, input):
        # initialiing class_probs with diseased and not diseased probability  
        classes_probs = { 'not_diseased': self.train_not_diseased_prob, 'diseased': self.train_diseased_prob }
        for i in range(not_diseased_statistics.shape[0]):
            [ mean, stdev ] = self.not_diseased_statistics[i]
            classes_probs['not_diseased'] *= self.calculate_prob(input[i], mean, stdev)
        for i in range(diseased_statistics.shape[0]):
            [ mean, stdev ] = self.diseased_statistics[i]
            classes_probs['diseased'] *= self.calculate_prob(input[i], mean, stdev)

        # predicting the class (the higher value is the target value class)
        return 1 if classes_probs['diseased'] > classes_probs['not_diseased'] else 0              
    
    
 
    def calculate_prob(self, x, mean, stdev):
        exponent = np.exp(-((x-mean)**2 / (2 * stdev**2 )))
        return (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent
    
    # params:  test : np.array of data to predict it target value
    # return: np.array of predicted values for test data (0 or 1) for each entry
    def predict(self, test):
        # prepare test data to be processed
        test_data = test.drop('target', axis=1).values
        print(test_data)
        
        # summarize = summarize_by_class(train)

#         # 1- separating train data by class
#         self.diseased_data = train[train['target'] == 1].drop('target', axis=1)
#         self.not_diseased_data = train[train['target'] == 0].drop('target', axis=1)

#         # 2- calculate statistics for each class
#         diseased_statistics = np.array([(diseased[column].mean(), diseased[column].std()) for column in diseased.columns])
#         not_diseased_statistics = np.array([(not_diseased[column].mean(), not_diseased[column].std()) for column in not_diseased.columns])
#         print(self.diseased_stats)
        # 3&4- calculate the probabilty for each row to predict the target 
        predictions = np.array([])
        for row in test_data:    
            output = self.predict_input_class(row)  
            predictions = np.append(predictions, [output])
        return predictions
    
    
    # params: true: np.array for true values (0 or 1)
    #         pred: np.array for predicted valuse by the model (0 or 1)
    # return: prcentage of the accuracy  of the model
    def score(self, true, pred):
        return accuracy_score(true, pred)
```


```python
a = heart_disease
b = heart_disease["target"]

#separate the dataset to training group and test group
from sklearn.model_selection import train_test_split
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size = .2)
```


```python
clf = NaiveBayes()

```


```python
clf.fit(a_train)[2]
```




    820




```python
clf.diseased_data
```




    array([], dtype=float64)




```python
preds = clf.predict(a_test)
```

    [[66.  0.  0. ...  1.  2.  3.]
     [66.  1.  0. ...  2.  1.  2.]
     [61.  1.  0. ...  1.  1.  2.]
     ...
     [43.  1.  0. ...  1.  0.  2.]
     [54.  1.  2. ...  1.  0.  3.]
     [57.  1.  2. ...  1.  1.  3.]]
    


```python
b_test.shape
```




    (205,)




```python
a_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>737</th>
      <td>67</td>
      <td>1</td>
      <td>0</td>
      <td>120</td>
      <td>229</td>
      <td>0</td>
      <td>0</td>
      <td>129</td>
      <td>1</td>
      <td>2.6</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>399</th>
      <td>63</td>
      <td>1</td>
      <td>3</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>354</th>
      <td>57</td>
      <td>1</td>
      <td>1</td>
      <td>124</td>
      <td>261</td>
      <td>0</td>
      <td>1</td>
      <td>141</td>
      <td>0</td>
      <td>0.3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>189</th>
      <td>64</td>
      <td>1</td>
      <td>2</td>
      <td>125</td>
      <td>309</td>
      <td>0</td>
      <td>1</td>
      <td>131</td>
      <td>1</td>
      <td>1.8</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>375</th>
      <td>66</td>
      <td>1</td>
      <td>0</td>
      <td>160</td>
      <td>228</td>
      <td>0</td>
      <td>0</td>
      <td>138</td>
      <td>0</td>
      <td>2.3</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>708</th>
      <td>60</td>
      <td>0</td>
      <td>2</td>
      <td>120</td>
      <td>178</td>
      <td>1</td>
      <td>1</td>
      <td>96</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1023</th>
      <td>50</td>
      <td>0</td>
      <td>0</td>
      <td>110</td>
      <td>254</td>
      <td>0</td>
      <td>0</td>
      <td>159</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>297</th>
      <td>58</td>
      <td>1</td>
      <td>0</td>
      <td>150</td>
      <td>270</td>
      <td>0</td>
      <td>0</td>
      <td>111</td>
      <td>1</td>
      <td>0.8</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1024</th>
      <td>54</td>
      <td>1</td>
      <td>0</td>
      <td>120</td>
      <td>188</td>
      <td>0</td>
      <td>1</td>
      <td>113</td>
      <td>0</td>
      <td>1.4</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>299</th>
      <td>52</td>
      <td>1</td>
      <td>1</td>
      <td>120</td>
      <td>325</td>
      <td>0</td>
      <td>1</td>
      <td>172</td>
      <td>0</td>
      <td>0.2</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>820 rows × 14 columns</p>
</div>




```python
b_test.values
```




    array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1,
           0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0,
           1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1,
           0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0,
           1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0,
           0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0,
           0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1,
           0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1,
           0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0,
           0, 0, 0, 0, 1, 1, 0], dtype=int64)




```python
preds
```




    array([0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 1., 0., 0., 1., 1., 0., 1.,
           1., 0., 0., 1., 1., 0., 1., 1., 0., 1., 1., 0., 0., 1., 1., 0., 1.,
           1., 1., 0., 1., 0., 0., 1., 0., 1., 0., 1., 1., 1., 0., 0., 0., 1.,
           0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 0., 0., 1., 1., 0., 1., 1., 0., 1., 1., 1., 0., 0.,
           1., 1., 1., 1., 0., 1., 1., 1., 0., 0., 1., 0., 1., 1., 1., 0., 0.,
           1., 1., 0., 1., 1., 0., 1., 0., 0., 1., 1., 1., 1., 0., 0., 0., 1.,
           0., 1., 0., 0., 0., 0., 1., 0., 1., 1., 1., 0., 0., 0., 1., 0., 1.,
           1., 1., 1., 1., 1., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 1.,
           1., 0., 1., 1., 0., 1., 1., 0., 0., 0., 1., 1., 1., 1., 0., 0., 1.,
           1., 1., 1., 0., 1., 1., 0., 0., 1., 0., 1., 1., 1., 1., 0., 0., 0.,
           0., 1., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
           1.])




```python
clf.score(b_test.values, predss)
```




    0.8341463414634146


