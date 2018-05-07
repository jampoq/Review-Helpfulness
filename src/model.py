from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import LinearSVC
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.evaluation import BinaryClassificationMetrics

from pyspark.ml.evaluation import BinaryClassificationEvaluator

###############################################
# Model class for difference models to run on.
###############################################

class Model():

    def __init__(self,data,split,file_name):

        self.data = data
        self.evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction",
                                                       metricName='areaUnderROC')

        # Split data into train and test sets:
        self.trainingData, self.testData = data.randomSplit([split, 1-split],seed=123456)
        self.train_count = self.trainingData.count()
        self.test_count = self.testData.count()

        # Get class balance:
        self.positives = self.testData.filter(self.testData.label == 1).count()
        self.class_balance = self.positives/self.test_count

        self.filename = file_name

    def create_confusion_matrix(self,df,label='label',prediction='prediction'):
        '''
        Scores prediction against actual labels based on:

        Accuracy
        Precision
        Recall
        F1 score

        Input:
        -------
        df : pyspark dataframe that with predictions
        label : str of column of actual label
        prediction : str of column with prediction

        Output:
        -------
        dictionary of scores and confustion matrix.
        '''
        return_dict = {}

        return_dict['TP'] = df[(df[label] == 1) & (df[prediction] == 1)].count()
        return_dict['TN'] = df[(df[label] == 0) & (df[prediction] == 0)].count()
        return_dict['FP'] = df[(df[label] == 0) & (df[prediction] == 1)].count()
        return_dict['FN'] = df[(df[label] == 1) & (df[prediction] == 0)].count()
        return_dict['Total'] = df.count()

        print("Class Balance :", self.class_balance)
        print("True Positives: ", return_dict['TP'])
        print("True Negatives: ", return_dict['TN'])
        print("False Positives: ", return_dict['FP'])
        print("False Negatives: ", return_dict['FN'])
        print("Total ", return_dict['Total'])

        return_dict['Accuracy'] = (float(return_dict['TP'] + return_dict['TN']) /
                                  (return_dict['Total']))
        print("Accuracy: ", return_dict['Accuracy'])

        return_dict['Precision'] = (float(return_dict['TP']) /
                                   (return_dict['TP'] + return_dict['FP']))
        print("Precision: ", return_dict['Precision'])

        return_dict['Recall'] = (float(return_dict['TP']) /
                                (return_dict['TP'] + return_dict['FN']))
        print("Recall: ", return_dict['Recall'])

        return_dict['F1_Score'] = (2*(return_dict['Recall'] * return_dict['Precision']) /
                                  (return_dict['Recall'] + return_dict['Precision']))
        print("F1_Score: ", return_dict['F1_Score'])

        # Write results to results file.
        with open(self.filename,'a') as f:
            f.write("\nClass Balance: " + str(self.class_balance))
            f.write("\nTrue Positives: " + str(return_dict['TP']))
            f.write("\nTrue Negatives: " + str(return_dict['TN']))
            f.write("\nFalse Positives: " + str(return_dict['FP']))
            f.write("\nFalse Negatives: " + str(return_dict['FN']))
            f.write("\nTotal: " + str(return_dict['Total']))
            f.write("\nAccuracy: " + str(return_dict['Accuracy']))
            f.write("\nPrecision: " + str(return_dict['Precision']))
            f.write("\nRecall: " + str(return_dict['Recall']))
            f.write("\nF1_Score: " + str(return_dict['F1_Score']))

        return return_dict

    def run_logistic(self):
        '''
        Method to run logistic regression on our transformed data.

        Input:
        -------
        None

        Output:
        -------
        Dictionary of confusion matrix scores for this particular model.

        '''

        # Instantiate model, fit, then transform.
        lr = LogisticRegression(maxIter=30,
                                regParam=0.3,
                                elasticNetParam=0)
        lr_model = lr.fit(self.trainingData)
        predictions = lr_model.transform(self.testData)

        # Write type of model to filename.
        with open(self.filename,'a') as f:
            f.write("\n\nLogistic Regression:")

        # Create confusion matrix to see how well the model performed
        confusion_matrix = self.create_confusion_matrix(predictions)

        # Evaluate model's AUC.
        auc = self.evaluator.evaluate(predictions)
        print("AUC Score: ",str(auc))

        # Write result of model to filename.
        with open(self.filename,'a') as f:
            f.write("\nAUC Score: " + str(auc))

        return confusion_matrix

    def run_logistic_cv(self):
        '''
        Method to run logistic regression with cross validation on our transformed data.

        Input:
        -------
        None

        Output:
        -------
        Dictionary of confusion matrix scores for this particular model.
        '''

        # Instantiate model
        lr = LogisticRegression(maxIter=20,
                                regParam=0.3,
                                elasticNetParam=0)

        # Create ParamGrid for Cross Validation
        paramGrid = (ParamGridBuilder()
                    .addGrid(lr.regParam, [0.1, 0.3, 0.5]) # regularization parameter
                    .addGrid(lr.elasticNetParam, [0.0, 0.1, 0.2]) # Elastic Net Parameter (Ridge = 0)
                    #.addGrid(lr.maxIter, [10, 20, 30]) #Number of iterations
                    #.addGrid(idf.numFeatures, [10, 100, 1000]) # Number of features
                    .build())

        # Create 5-fold CrossValidator
        cv = CrossValidator(estimator=lr,
                            estimatorParamMaps=paramGrid,
                            evaluator=self.evaluator,
                            numFolds=5)
        cvModel = cv.fit(self.trainingData)

        predictions = cvModel.transform(self.testData)

        # Write type of model to filename.
        with open(self.filename,'a') as f:
            f.write("\n\nLogistic Regression with CV:")

        # Create confusion matrix to see how well the model performed
        confusion_matrix = self.create_confusion_matrix(predictions)

        # Evaluate model's AUC.
        auc = self.evaluator.evaluate(predictions)
        print("AUC Score: ",str(auc))

        # Write result of model to filename.
        with open(self.filename,'a') as f:
            f.write("\nAUC Score: " + str(auc))

        return confusion_matrix

    def run_naive_bayes(self):
        '''
        Method to run naive bayes on our transformed data.

        Input:
        -------
        None

        Output:
        -------
        Dictionary of confusion matrix scores for this particular model.
        '''

        # Instantiate model.
        nb = NaiveBayes(featuresCol='features',
                        labelCol='label',
                        smoothing=1)
        model = nb.fit(self.trainingData)
        print("GOT HERE!!!")
        predictions = model.transform(self.testData)

        # Write type of model to filename.
        with open(self.filename,'a') as f:
            f.write("\n\nNaive Bayes:")

        # Create confusion matrix to see how well the model performed
        confusion_matrix = self.create_confusion_matrix(predictions)

        # Evaluate model's AUC.
        auc = self.evaluator.evaluate(predictions)
        print("AUC Score: ",str(auc))

        # Write result of model to filename.
        with open(self.filename,'a') as f:
            f.write("\nAUC Score: " + str(auc))

        return confusion_matrix

    def run_random_forest(self):
        '''
        Method to run random forest on our transformed data.

        Input:
        -------
        None

        Output:
        -------
        Dictionary of confusion matrix scores for this particular model.
        '''
        rf = RandomForestClassifier(labelCol="label",
                                    featuresCol="features",
                                    numTrees = 100,
                                    maxDepth = 5,
                                    maxBins = 32)

        # Train model with Training Data
        rfModel = rf.fit(self.trainingData)
        predictions = rfModel.transform(self.testData)

        # Write type of model to filename.
        with open(self.filename,'a') as f:
            f.write("\n\nRandom Forest:")

        # Create confusion matrix to see how well the model performed
        confusion_matrix = self.create_confusion_matrix(predictions)

        # Evaluate model's AUC.
        auc = self.evaluator.evaluate(predictions)
        print("AUC Score: ",str(auc))

        # Write result of model to filename.
        with open(self.filename,'a') as f:
            f.write("\nAUC Score: " + str(auc))

        return confusion_matrix

    def run_random_forest_cv(self):
        '''
        Method to run random forest with cross validation on our transformed data.

        Input:
        -------
        None

        Output:
        -------
        Dictionary of confusion matrix scores for this particular model.
        '''

        # Instantiate model
        rf = RandomForestClassifier(labelCol="label",
                                    featuresCol="features",
                                    numTrees = 100,
                                    maxDepth = 5,
                                    maxBins = 32)

        # Create ParamGrid for Cross Validation
        paramGrid = (ParamGridBuilder()
             .addGrid(rf.numTrees, [50, 100, 150]) # number of trees
             .addGrid(rf.maxDepth, [3, 4, 5]) # maximum depth
             #.addGrid(rf.maxBins, [24, 32, 40]) #Number of bins
             .build())

        # Create 5-fold CrossValidator
        cv = CrossValidator(estimator=rf,
                            estimatorParamMaps=paramGrid,
                            evaluator=self.evaluator,
                            numFolds=5)

        # Run cross validations
        cvModel = cv.fit(self.trainingData)

        # Use test set here so we can measure the accuracy of our model on new data
        predictions = cvModel.transform(self.testData)

        # Write type of model to filename.
        with open(self.filename,'a') as f:
            f.write("\n\nRandom Forest with CV:")

        # Create confusion matrix to see how well the model performed
        confusion_matrix = self.create_confusion_matrix(predictions)

        # Evaluate model's AUC.
        auc = self.evaluator.evaluate(predictions)
        print("AUC Score: ",str(auc))

        # Write result of model to filename.
        with open(self.filename,'a') as f:
            f.write("\nAUC Score: " + str(auc))

        return confusion_matrix

    def run_gradient_boost(self):
        '''
        Method to run gradient boost on our transformed data.

        Input:
        -------
        None

        Output:
        -------
        Dictionary of confusion matrix scores for this particular model.
        '''

        # Instantiate model
        gbt = GBTClassifier(labelCol="label",
                            featuresCol="features",
                            maxIter=50
                            )

        # Train model.  This also runs the indexers.
        model = gbt.fit(self.trainingData)

        # Make predictions.
        predictions = model.transform(self.testData)

        # Write type of model to filename.
        with open(self.filename,'a') as f:
            f.write("\n\nGradient Boost:")

        # Create confusion matrix to see how well the model performed
        confusion_matrix = self.create_confusion_matrix(predictions)

        # Evaluate model's AUC.
        auc = self.evaluator.evaluate(predictions)
        print("AUC Score: ",str(auc))

        # Write result of model to filename.
        with open(self.filename,'a') as f:
            f.write("\nAUC Score: " + str(auc))

        return confusion_matrix

    def run_linear_svc(self):
        '''
        Method to run linear_svc on our transformed data.

        Input:
        -------
        None

        Output:
        -------
        Dictionary of confusion matrix scores for this particular model.
        '''

        # Instantiate model
        svc = LinearSVC(labelCol="label",
                        featuresCol="features",
                        maxIter=40
                        )

        # Train model.  This also runs the indexers.
        model = svc.fit(self.trainingData)

        # Make predictions.
        predictions = model.transform(self.testData)

        # Write type of model to filename.
        with open(self.filename,'a') as f:
            f.write("\n\nLinear SVC:")

        # Create confusion matrix to see how well the model performed
        confusion_matrix = self.create_confusion_matrix(predictions)

        # Evaluate model's AUC.
        auc = self.evaluator.evaluate(predictions)
        print("AUC Score: ",str(auc))

        # Write result of model to filename.
        with open(self.filename,'a') as f:
            f.write("\nAUC Score: " + str(auc))

        return confusion_matrix
