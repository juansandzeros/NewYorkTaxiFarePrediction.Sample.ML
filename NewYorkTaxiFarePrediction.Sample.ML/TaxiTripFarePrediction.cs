using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace NewYorkTaxiFarePrediction.Sample.ML
{
    /// <summary>
    /// The TaxiTripFarePrediction class represents a single far prediction.
    /// </summary>
    public class TaxiTripFarePrediction
    {
        [ColumnName("Score")] public float FareAmount;
    }
}
