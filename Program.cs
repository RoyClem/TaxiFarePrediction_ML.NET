using System;
using System.IO;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Categorical;
using Microsoft.ML.Transforms.Normalizers;
using Microsoft.ML.Transforms.Text;

namespace TaxiFarePrediction
{
    // https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/taxi-fare

    class Program
    {
        static readonly string _trainDataPath   = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        static readonly string _testDataPath    = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        static readonly string _modelPath       = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static TextLoader _textLoader;      // used to load and transform the datasets.

        static void Main(string[] args)
        {

            // When building a model with ML.NET you start by creating an ML Context. This is comparable conceptually to using
            // DbContext in Entity Framework. The environment provides a context for your machine learning job that can be used
            // for exception tracking and logging.

            MLContext mlContext = new MLContext(seed: 0);

            // setup for data loading initialize the _textLoader global variable in order to reuse it.When you create a TextLoader, 
            // you pass in the context needed and the TextLoader.Arguments class which enables customization.Specify the data schema 
            // by passing an array of TextLoader.Column objects to the TextLoader containing all the column names and their types.
            // We defined the data schema previously when we created our TaxiTrip class.

            // The TextLoader class returns a fully initialized TextLoader
            _textLoader = mlContext.Data.CreateTextLoader(new TextLoader.Arguments()
                {
                    Separators = new[] { ',' },
                    HasHeader = true,
                    Column = new[]
                   {
                        new TextLoader.Column("VendorId",       DataKind.Text, 0),
                        new TextLoader.Column("RateCode",       DataKind.Text, 1),
                        new TextLoader.Column("PassengerCount", DataKind.R4, 2),      // R4 must specify numeric
                        new TextLoader.Column("TripTime",       DataKind.R4, 3),
                        new TextLoader.Column("TripDistance",   DataKind.R4, 4),
                        new TextLoader.Column("PaymentType",    DataKind.Text, 5),
                        new TextLoader.Column("FareAmount",     DataKind.R4, 6)
                    }
                }
            );

            var model = Train(mlContext, _trainDataPath);

            Evaluate(mlContext, model);

            TestSinglePrediction(mlContext);

            Console.ReadKey();
        }
        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            // The Train method executes the following tasks:
            // 1. Loads the data.
            // 2. Extracts and transforms the data.
            // 3. Trains the model.
            // 4. Saves the model as .zip file.
            // 5. Returns the model.

            //**
            //** 1. Loads the data
            //**
            // IDataView is the fundamental data pipeline type, comparable to IEnumerable for LINQ.
            IDataView dataView = _textLoader.Read(dataPath);

            //**
            //** 2. Extracts and transforms the data
            //**

            // a. When the model is trained and evaluated, by default, the values in the Label column are considered as correct values to be predicted. 
            // As we want to predict the taxi trip fare, copy the FareAmount column into the Label column. To do that, use the CopyColumnsEstimator transformation class

            // b. the algorithm that trains the model requires numeric features, so you have to transform the categorical data(VendorId, RateCode, and PaymentType) values 
            // into numbers.To do that, use the OneHotEncodingEstimator transformation class, which assigns different numeric key values to the different 
            // values in each of the columns

            // c. The last step in data preparation combines all of the feature columns into the Features column using the ColumnConcatenatingEstimator 
            // transformation class. By default, a learning algorithm processes only features from the Features column

            // d. select a learning algorithm (learner) to train the model - in this case regression
            // The FastTreeRegressionTrainer learner utilizes gradient boosting. Gradient boosting is a machine learning technique for regression problems. 
            // It builds each regression tree in a step-wise fashion. It uses a pre-defined loss function to measure the error in each step and correct 
            // for it in the next. The result is a prediction model that is actually an ensemble of weaker prediction models

            var pipeline = mlContext.Transforms.CopyColumns(inputColumnName: "FareAmount", outputColumnName: "Label")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("VendorId"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("RateCode"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("PaymentType"))
                .Append(mlContext.Transforms.Concatenate("Features", "VendorId", "RateCode", "PassengerCount", "TripDistance", "PaymentType"))
                .Append(mlContext.Regression.Trainers.FastTree());

            //**
            //** 3. Trains the model
            //**

            // train the model using the Fit while providing the already loaded training data.This returns a model to use for predictions.pipeline.Fit() 
            // trains the pipeline and returns a Transformer based on the DataView passed in. The experiment is not executed until this happens.
            var model = pipeline.Fit(dataView);

            //**
            //** 4. Saves the model as a zip file
            //**

            // At this point, you have a model of type TransformerChain that can be integrated into any of your existing or new .NET applications.
            SaveModelAsFile(mlContext, model);

            //**
            //** 5. Returns the model
            //**

            return model;

        }

        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            // The Evaluate method executes the following tasks:
            // Loads the test dataset.
            // Creates the regression evaluator.
            // Evaluates the model and creates metrics.
            // Displays the metrics

            IDataView dataView = _textLoader.Read(_testDataPath);

            var predictions = model.Transform(dataView);
            
            // The RegressionContext.Evaluate method computes the quality metrics for the PredictionModel using the specified 
            // dataset.It returns a RegressionMetrics object that contains the overall metrics computed by regression evaluators
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");

            Console.WriteLine($"*------------------------------------------------");

            //  RSquared is another evaluation metric of the regression models.RSquared takes values between 0 and 1.The closer its 
            // value is to 1, the better the model is.
            Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}");

            // RMS is one of the evaluation metrics of the regression model.The lower it is, the better the model is
            Console.WriteLine($"*       RMS loss:      {metrics.Rms:#.##}");
        }
        private static void TestSinglePrediction(MLContext mlContext)
        {
            // Creates a single comment of test data.
            // Predicts fare amount based on test data.
            // Combines test data and predictions for reporting.
            // Displays the predicted results.
            
            ITransformer loadedModel;
            
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);
            }

            // While the model is a transformer that operates on many rows of data, a very common production scenario is a need for 
            // predictions on individual examples.The PredictionEngine<TSrc, TDst> is a wrapper that is returned from the CreatePredictionEngine method.
            var predictionFunction = loadedModel.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(mlContext);

            // Add a trip to test the trained model's prediction of cost in the Predict method by creating an instance of TaxiTrip
            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0          // To predict. Actual/Observed = 15.5
            };

            var prediction = predictionFunction.Predict(taxiTripSample);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
            Console.WriteLine($"**********************************************************************");
        }

        private static void SaveModelAsFile(MLContext mlContext, ITransformer model)
        {
            // The ITransformer has a SaveTo(IHostEnvironment, Stream) method that takes in the _modelPath global field, and a Stream. Since we want
            // to save this as a zip file, we'll create the FileStream immediately before calling the SaveTo method.

            using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, fileStream);
            }

            Console.WriteLine("The model is saved to {0}", _modelPath);
        }
    }
}
