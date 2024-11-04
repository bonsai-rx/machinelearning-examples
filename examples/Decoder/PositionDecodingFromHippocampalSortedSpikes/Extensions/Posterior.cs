using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using Python.Runtime;
using Bonsai.ML.Python;

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class Posterior
{
    public IObservable<PosteriorData> Process(IObservable<PyObject> source)
    {
        return source.Select(value => {
            return new PosteriorData(value);
        });
    }
}

public class PosteriorData
{
    public PosteriorData(PyObject posterior)
    {
        _data = (double[,])PythonHelper.ConvertPythonObjectToCSharp(posterior);
        _mapEstimate = 0;
        for (int i = 1; i < _data.GetLength(1); i++)
        {
            if (_data[0, i] > _data[0, _mapEstimate])
            {
                _mapEstimate = i;
            }
        }
    }
    public double[,] Data => _data;
    private double[,] _data; 

    public int MapEstimate => _mapEstimate;
    private int _mapEstimate; 
}
