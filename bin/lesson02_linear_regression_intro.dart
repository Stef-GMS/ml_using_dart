// What is linear regression?
//
// In statistical modeling, linear regression is a model which estimates the 
// linear relationship between a scalar response and one or more variables.
//  
// Linear regression is a simple technique for predicting a quantitative response
// using a single feature. It assumes that the relationship between the feature
// and the response is linear.

// To do linear regression, we need to know the relationship between the feature
// and the response. In other words, we need to know the slope and the intercept.

// The slope is the amount that the response changes when the feature increases by 1.
// The intercept is the value of the response when the feature is 0. 

// The equation of the line in two-dimensional space is y = mx + b. Where b is
// the intercept and m is the slope.

// The coefficient is the slope of the line. The intercept is the bias.
// The bias of the line from the origin is the intercept. The slope is the m.
// The equation of the line is y = mx + b.

// The solution found by solving the equation is known as the line of best fit
// (or the least squares line). This solution is also known as a Closed-form
// solution.

// Imagine you must predict house prices based on the distance to a metro station.
// You have a dataset with the distance to the metro station and the house prices.
// In this case x is the distance to the metro station and y is the house price.

// In machine learning terms, the distance to the metro station is a feature, and
// the house price is a response.

// A feature is a variable that describes the phenomenon you are trying to model.
// In this case, the distance to the metro station is a feature.
// A response is a variable that you are trying to predict. In this case, the house
// price is a response.

// The x-axis would be the distance to the metro station and the y-axis would be
// the house price. The perpendicular line would be the line of best fit. The
// slope would be the m and the intercept would be the b.

// One can find the coefficients of the closed-form solution using DataFrame(),
// which is a library for working with tabular data from the ml_dataframe package,
// and LinearRegressor(), a library for linear regression from ml_algo package.

import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

void main() {
  final data = DataFrame([
    ['distance_to_nearest_metro_station', 'price'],
    [250, 500000],
    [1500, 430000],
    [800, 470000]
  ]);

  final unlabeledData = DataFrame([
    ['distance_to_nearest_metro_station'],
    [500],
  ]);

  // The bias is the intercept. The slope is the m.
  final model = LinearRegressor(
    data,
    'price',
    fitIntercept: true,
  );

  final prediction = model.predict(unlabeledData);

  print('Coefficient (bias, distance): ${model.coefficients}');
  print('Prediction: $prediction');

}

