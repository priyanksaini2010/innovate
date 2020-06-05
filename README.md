<h1>InnovateIN48 5th May 2020  Valarians  Gurugram</h1>
<h2>Team introduction</h2>
<ul>
    <li>Amit Sharma</li>
    <li>Priyank Saini</li>
    <li>Manish Phalaswal</li>
</ul>
<h2>Problem Statements</h2>
 <h4>Problem Statement 1</h4>
    <ul>
        <li>
            Introduction to <b>Connections</b>
            <ul>
                <li><b>FIS</b> customer relationship management Web App </li>
                <li>
                    Integrated with following FIS cores :
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
        <li> Problem with Connections recommendation engine :
            <ul>
                <li>Bank teller recommends product to customers on human judgement 
                without any statistics.</li>
                <li>Bank teller have to followup with these recommendation for months 
            which many time results in zero output.</li>
            </ul>
        </li>
    </ul>
        
         
         
        
  
<h4>Problem Statement 2</h4>
    <ul>
        <li>
            <b>Bad Loans</b>
            <ul>
                <li>Loss of profit</li>
                <li>Bank agent / employee's performance impact</li>
            </ul>
        </li>
    </ul>
    
    
<h2>Solutions</h2>
<h4> Solution for problem  #1 </h4>
    <ul>
        <li>Strong arm Bank tellers with a statistical customer classification program</li>
        <li>
            Steps to achieve this tool
            <ul>
                <li>Extract historical data</li>
                <li>Clean Up data</li>
                <li>
                Take benefit of <b>Machine Learning</b> algorithms to train model that can judge 
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
    </ul>

<h4>Solution for problem  #2</h4>
    <ul>
        <li>
            As described in solution #1 Machine Learning algorithms can be used to  strong arm 
            bank employee / agents to judge bank loan defaulter beforehand  
        </li>
    </ul>
<h2>Value Adds</h2>
<ul>
    <li>Bank employees will have more stability in their book of business (Good for employees).</li>
    <li>More careful underwriting hence preventing bank's loss of income (Good for the bank; Reduce insolvency risk).</li>
    <li>Difference in earned premium can be re-invested in business/distributed among shareholders (Good for investors)</li>
</ul>
<h2>Technology Used </h2>
<ul>
    <li>Python 3</li>
    <li>
        Python libraries Used
        <ul>
            <li>Pandas</li>
            <li>Numpy</li>
            <li>Xgboost</li>
            <li>Sklearn</li>
            <li>Matplotlib</li>
            <li>Imblearn</li>
        </ul>
    </li>
    <li>
        <b>Supervised Learning </b>Models Used
        <ul>
            <li>
                K-Nearest-Neighbors Classifier for Product Recommendation (74% Accuracy)
            </li>
            <li>
                XGBoost Classifier for Loan Repayment Prediction (88% Accuracy)      
            </li>
        </ul>
    </li>
    <li>
        <b>Supervised Learning</b> Models Tested
        <ul>
            <li>
                Logisitic Regression Classifier
                <br />
                <img src="https://raw.githubusercontent.com/priyanksaini2010/innovate/master/LR.PNG"/>
            </li>
            <li>
                K-Nearest-Neighbors Classifier
                <br />
                <img src="https://raw.githubusercontent.com/priyanksaini2010/innovate/master/KNN-Classifier.PNG"/>
            </li>
            <li>
                Decision Tree <Classifier></Classifier>
                <br />
                <img src="https://raw.githubusercontent.com/priyanksaini2010/innovate/master/Decision%20Tree%20Classifier.PNG"/>
            </li>
            <li>
                Random Forest Classifier
                <br />
                <img src="https://raw.githubusercontent.com/priyanksaini2010/innovate/master/Random%20Forest.PNG"/>
            </li>
            <li>
                XGBoost Classifier
                <br />
                <img src="https://raw.githubusercontent.com/priyanksaini2010/innovate/master/XGBoost.PNG"/>
            </li>
            <li>
                Gradient Boosting Classifier
                <br />
                <img src="https://raw.githubusercontent.com/priyanksaini2010/innovate/master/Gradient%20Boosting.PNG"/>
            </li>
        </ul>      
    </li>
    <li>
        Python Jupyter Notebook      
    </li>
    <li>
        Python <b>Flask</b> framework for Web API    
    </li>
</ul>
<h2>Agility</h2>
<ul>
    <li>Easy integration with any client with exposed API</li>
</ul>
<h2>Future Scope</h2>
<ul>
    <li>
        Merging two models on the basis of product classification
        <ul>
            <li>
                If one customer is predicted as a potential prospect by model #1 
                It would call model #2 in case product is of credit type.
            </li>
            <li>
                Integration with <b>Churn prediction</b> model.  
            </li>
        </ul>
    </li>
    <li>Can be extended further to train model as per client requirement</li>
</ul>
<h2>Constraints in current POC</h2>
<ul>
    <li>
        Publicly available dataset are used
    </li>
    <li>
        Client application is not developed due to time constraints hence using PostMan 
    </li>
    <li>
        Due to time constraints <b>FIS</b> API security guidelines are not inplace
    </li>
</ul>
<h2>Project Demo</h2>

<h2>Query ?</h2>