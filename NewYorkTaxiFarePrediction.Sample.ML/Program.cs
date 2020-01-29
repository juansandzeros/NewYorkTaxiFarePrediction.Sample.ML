using Microsoft.ML;
using System;
using System.IO;

namespace NewYorkTaxiFarePrediction.Sample.ML
{
    class Program
    {
        // file paths to data files
        static readonly string dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yellow_tripdata_2018-12.csv");

        static void Main(string[] args)
        {
            // create the machine learning context
            var mlContext = new MLContext(seed: 0);

            // load data
            Console.Write("Loading training data....");
            IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(dataPath, hasHeader: true, separatorChar: ',');
            Console.WriteLine("done");

            // split data into train/test
            var partitions = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            IDataView trainData = partitions.TrainSet;
            IDataView testData = partitions.TestSet;

            // train/evaluate/predict
            var model = Train(mlContext, trainData);
            Evaluate(mlContext, model, testData);
            TestSinglePrediction(mlContext, model);
        }

        private static ITransformer Train(MLContext mlContext, IDataView dataView)
        {  
            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded"))
                .Append(mlContext.Regression.Trainers.FastTree());

            Console.Write("Training the model....");
            var model = pipeline.Fit(dataView);
            Console.WriteLine("done");

            return model;
        }

        private static void Evaluate(MLContext mlContext, ITransformer model, IDataView dataView)
        {
            Console.Write("Evaluating the model....");
            var predictions = model.Transform(dataView);
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");

            Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
        }

        private static void TestSinglePrediction(MLContext mlContext, ITransformer model)
        {
            var predictionFunction = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);

            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                //TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0 // To predict. Actual/Observed = 15.5
            };

            var prediction = predictionFunction.Predict(taxiTripSample);

            Console.WriteLine($"");
            Console.WriteLine($"Single prediction:");
            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
            Console.WriteLine($"**********************************************************************");
        }
    }
}
