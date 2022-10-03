# Poisson intensity of limit order execution, calibration of parameters A and k using level 1 tick data  

## Description

A limit order placed at a price *S<sub>t</sub> ± δ*, has the instantaneous probability of execution *λ(δ)dt* where the intensity *λ(δ)* is given by:

  *λ(δ) = A e<sup> -kδ</sup>*    
  
*λ* - Poisson order execution intensity  
*δ* - spread (distance from mid price *S<sub>t</sub>*)  
*A* - parameter, positively related to trading intensity  
*k* - parameter, positively related to market depth  

Package Execution Intensity Estimator (EIE) contains single and multi threaded calibration procedure of *A* and *k* parameters. 
Methods that calculate  intensity *λ(δ, A, k)* and spread *δ(λ, A, k)* are provided as well.
Algorithm operates on level 1 tick data, therefore it is suitable in a setting where liquidity is not fully observable (i.e. dark pools). 
Calibration is two step procedure performed separately for buy and sell limit orders.     
<br>**Steps:**  
- For each spread *δ<sub>k</sub>* of *N* predefined spreads *(δ<sub>0</sub> , δ<sub>1</sub> , δ<sub>2</sub> , ... δ<sub>N-1</sub>)* 
estimate execution intensity *λ(δ<sub>k</sub>)* using "waiting time" approach described in [[1]](#references) 4.4.2.. 
Result of this step is set of *N* points *(δ<sub>k</sub> , λ(δ<sub>k</sub>))* on empirical Spread Intensity Curve (SIC)  
- Estimate *A* and *k* based on *N* points from previous step. This can be achieved by various approaches. 
Code implements two approaches described in [[2]](#references) 3.2:    
    - ***LOG_REGRESSION***  performs OLS regression of *log(λ<sub>k</sub>)* on *δ<sub>k</sub>*. Finally *k = -slope* and *A = e<sup> intercept</sup>*  
    - ***MULTI_CURVE***  from set of *N* points creates *N<sub>s</sub> = (N\*(N-1))/2* unique pairs fo points *((δ<sub>x</sub> , λ<sub>x</sub>) , (δ<sub>y</sub> , λ<sub>y</sub>))*.
      For each set of points solves the following set of equations for *A'* and *k'* :   
       *λ<sub>x</sub> = A' e<sup> -k'δ<sub>x</sub></sup>*          
       *λ<sub>y</sub> = A' e<sup> -k'δ<sub>y</sub></sup>*         
      Final estimates are *A = mean(A'<sub>1</sub> , A'<sub>2</sub> , ... A'<sub>N<sub>s</sub></sub>)* and *k = mean(k'<sub>1</sub> , k'<sub>2</sub> , ... k'<sub>N<sub>s</sub></sub>)*   
      
Once *A* and *k* are calibrated, depending on context of usage, user can specify:
- spread *δ* to obtain corresponding intensity *λ(δ)*
- intensity *λ* to obtain corresponding spread *δ(λ)*  
<br><br>
![gamma-surface](pic/sic.png) 