//import 'dart:io';  // Needed when loading model from folder instead of internet

import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

void main() async {
  //load the data to train the model
  final data = getWineQualityDataFrame();

  // print the data structure -- the data consists of 11 independent variables, known as features,
  // and the target variable 'quality', known as the label, which is the quality of the wine
  print(data);

  // Split the data into train and test sets.  The first set is used to train the model,
  // and the second set is used to assess the model. splitData is from the ml_algo library and
  // is used to split the data into train and test sets.  The first argument is the source data,
  // the second argument is the fraction of the data to be used for training
  final splits = splitData(data, [0.7]); // use 70% of the source data to train the model

  // The first element of the splits list is the train data, and the second element is the test data
  final trainData = splits.first;
  final testData = splits.last;

  // Train the model - the first argument is the train data, the second argument is the target column name
  final model = LinearRegressor(trainData, 'quality');

  // Assess the model - the first argument is the test data, the second argument is the
  // metric type to assess the model with mape (mean absolute percentage error)
  final score = model.assess(testData, MetricType.mape);

  print('Score is $score'); // One can interpret the result as the mean absolute percentage error of the model

  // Compare predicted values and the actual ones

  // Drop the target column from the test data (it is not needed for the prediction)
  final unlabelledData = testData.dropSeries(names: ['quality']);

  // Predict the values for the test data using the trained model - the first argument is the model,
  // the second argument is the data to predict values for (in this case it is the test data)
  final prediction = model.predict(unlabelledData);

  // The first 40 rows of the test data are used to train the model, and the
  // remaining rows are used to assess the model using an arbitrary range of values for the target variable.
  // The take method is used to extract the first 5 rows from the data
  final actualValues = testData['quality'].data.skip(40).take(5);

  // The first 40 rows of the prediction are the predicted values of the model,
  // and the remaining rows are the actual values of the model (the actual values of the model are not known).
  // The take method is used to extract the first 5 rows from the data
  final predictedValues = prediction['quality'].data.skip(40).take(5);

  print('Actual values:  $actualValues');
  print('Predict values: $predictedValues');

  // The model can be used to predict the quality of the wine for the unseen data

  // The model can be saved to a file to use instead of calling one from web
  //model.saveAsJson('../models/model.json');

  // // The model can be loaded from a file
  // final file = File('models/model.json');
  // final json = await file.readAsString();
  // final modelToLoad = LinearRegressor.fromJson(json);
}
