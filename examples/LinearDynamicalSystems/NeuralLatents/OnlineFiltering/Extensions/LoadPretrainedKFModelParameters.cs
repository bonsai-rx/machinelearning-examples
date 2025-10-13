using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using System.Xml.Serialization;
using System.IO;
using TorchSharp;
using Bonsai;
using Bonsai.ML.Lds.Torch;

/// <summary>
/// Loads the parameters of a Kalman filter model.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Loads the parameters of a Kalman filter model.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class LoadKalmanFilterParameters
{
    public string TransitionMatrixFilePath { get; set; }
    public string MeasurementFunctionFilePath { get; set; }
    public string ProcessNoiseCovarianceFilePath { get; set; }
    public string MeasurementNoiseCovarianceFilePath { get; set; }
    public string InitialMeanFilePath { get; set; }
    public string InitialCovarianceFilePath { get; set; }

    public int NumObservations { get; set; }
    public int NumStates { get; set; }

    public torch.ScalarType Type { get; set; }

    [XmlIgnore]
    public torch.Device Device { get; set; }

    private static double[] ReadBinaryFile(string fileName)
    {
        using (var fileStream = new FileStream(fileName, FileMode.Open, FileAccess.Read))
        {
            using (var binaryReader = new BinaryReader(fileStream))
            {
                var fileLength = fileStream.Length;
                var numDoubles = fileLength / sizeof(double);
                var data = new double[numDoubles];
                for (int i = 0; i < numDoubles; i++)
                {
                    data[i] = binaryReader.ReadDouble();
                }
                return data;
            }
        }
    }

    /// <summary>
    /// Creates parameters for a Kalman filter model using the properties of this class.
    /// </summary>
    public IObservable<KalmanFilterParameters> Process()
    {
        torch.Tensor transitionMatrix = null;
        if (TransitionMatrixFilePath != null)
        {
            var data = ReadBinaryFile(TransitionMatrixFilePath);
            transitionMatrix = torch.from_array(data).reshape(NumStates, NumStates).to(Type);
        }

        torch.Tensor measurementFunction = null;
        if (MeasurementFunctionFilePath != null)
        {
            var data = ReadBinaryFile(MeasurementFunctionFilePath);
            measurementFunction = torch.from_array(data).reshape(NumObservations, NumStates).to(Type);
        }

        torch.Tensor processNoiseCovariance = null;
        if (ProcessNoiseCovarianceFilePath != null)
        {
            var data = ReadBinaryFile(ProcessNoiseCovarianceFilePath);
            processNoiseCovariance = torch.from_array(data).reshape(NumStates, NumStates).to(Type);
        }

        torch.Tensor measurementNoiseCovariance = null;
        if (MeasurementNoiseCovarianceFilePath != null)
        {
            var data = ReadBinaryFile(MeasurementNoiseCovarianceFilePath);
            measurementNoiseCovariance = torch.from_array(data).reshape(NumObservations, NumObservations).to(Type);
        }

        torch.Tensor initialMean = null;
        if (InitialMeanFilePath != null)
        {
            var data = ReadBinaryFile(InitialMeanFilePath);
            initialMean = torch.from_array(data).reshape(NumStates).to(Type);
        }

        torch.Tensor initialCovariance = null;
        if (InitialCovarianceFilePath != null)
        {
            var data = ReadBinaryFile(InitialCovarianceFilePath);
            initialCovariance = torch.from_array(data).reshape(NumStates, NumStates).to(Type);
        }

        var parameters = new KalmanFilterParameters(
            transitionMatrix: transitionMatrix,
            measurementFunction: measurementFunction,
            processNoiseCovariance: processNoiseCovariance,
            measurementNoiseCovariance: measurementNoiseCovariance,
            initialMean: initialMean,
            initialCovariance: initialCovariance
        );

        return Observable.Return(parameters);
    }
}