using Microsoft.ML.Data;

namespace TaxiFarePrediction
{
    public class TaxiTrip
    {

        // use the ColumnAttribute attribute to specify the indices of the source columns in the data set

        [Column("0")]
        public string VendorId;

        [Column("1")]
        public string RateCode;

        [Column("2")]
        public float PassengerCount;

        [Column("3")]
        public float TripTime;

        [Column("4")]
        public float TripDistance;

        [Column("5")]
        public string PaymentType;

        [Column("6")]
        public float FareAmount;
    }

    public class TaxiTripFarePrediction
    {

        // TaxiTripFarePrediction class represents predicted results.
        // In case of the regression task the Score column contains predicted label values.
        
        [ColumnName("Score")]
        public float FareAmount;
    }
}
