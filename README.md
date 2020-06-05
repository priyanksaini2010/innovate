<h1>InnovateIN48 5th May 2020  Valarian  Gurugram</h1>
<h2>Team introduction</h2>
<ul>
    <li>Amit Sharma</li>
    <li>Priyank Saini</li>
    <li>Manish Phalaswal</li>
</ul>
<h2>Problem Statements</h2>
- <h4>Problem Statement 1</h4>
    <ul>
        <li>
            Introduction to <b>Connections</b>
            <ul>
                <li><b>FIS</b> customer relation management Web App </li>
                <li>
                    Integerated with following FIS cores
                        <ul>
                            <li>Bankway</li>
                            <li>Horizon</li>
                            <li>Horizon XE</li>
                            <li>Bankpac</li>
                            <li>Miser</li>
                        </ul>
                </li>
                <li>300+ Community banks</li>
            </ul>
        </li>
        <li> Problem with Connections recommendation engine.
            <ul>
                <li>Bank teller recommends product to customers on human judgement <br/>
                without any statistics</li>
                <li>Bank teller have to followup with these recommendation for months <br/>
            which many time results in zero output</li>
            </ul>
        </li>
    </ul>. 
        
         
         
        
  
- <h4>Problem Statement 2</h4>
    <ul>
        <li>
            <b>Bad Loans</b>
            <ul>
                <li>Loss of profit</li>
                <li>Bank agent / employee's performance impact</li>
            </ul>
        </li>
    </ul>
    
    
##Solution
- <h4> Solution for problem  #1 </h4>
    <ul>
        <li>Strong arm Bank tellers with a statistical customer classification program</li>
        <li>
            Steps to achieve this tool
            <ul>
                <li>Extract historical data</li>
                <li>Clean Up data</li>
                <li>
                Take benefit of <b>Machine Learning</b> algorithms to train model that can judge <br />
                if customer will subscribe to certain product.
                </li>
                <li>
                    Apply trained model to current application
                    <ul>
                        <li>Expose web API</li>
                    </ul>
                </li>
            </ul>
        </li>
        <li>
            Impact of solution
            <ul>
                <li>
                    With this solution in place 
                </li>
            </ul>
        </li>
    </ul>

- <h4>Solution for problem  #2</h4>
    <ul>
        <li>
            As described in solution #1 Machine Learning algorithms can be used to <br/> strong arm 
            bank employee / agents to judge bank load defaulter beforehand  
        </li>
    </ul>
##Technology Used
- Python 3
- Python Libraries Used
    * Pandas
    * Numpy
    * Xgboost
    * Sklearn
    * Matplotlib
    * Imblearn
- Supervised Learning Models Used
    * K-Nearest-Neighbors for Product Recommendation (74% Accuracy)
    * XGBoost Classifier for Loan Repayment Prediction (88% Accuracy)
- Supervised Learning Models Tried
    * Logisitic Regression Classifier
    * K-Nearest-Neighbors Classifier
    * Decision Tree Classifier
    * Random Forest Classifier
    * Neural Network Classifier
    * XGBoost Classifier
    * Gradient Boosting Classifier
    * AdaBoost Classifier
    
- Jupyter Notebook

- Python Flask framework for Web API 
    

##Future Scope
<ul>
    <li>
        Merging two models on the basis of product classification
        <ul>
            <li>
                If a customer is predicted as potential prospect by model #1 <br>
                It would call model #2 in case product is of credit type.
            </li>
            <li>
                Integration of <b>churn prediction</b> model.  
            </li>
        </ul>
    </li>
</ul>

##Project Demo

##Query ?