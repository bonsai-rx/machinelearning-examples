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
public class PositionBins
{
    public IObservable<double[]> Process(IObservable<PyObject> source)
    {
        return source.Select(value => (double[])PythonHelper.ConvertPythonObjectToCSharp(value));
    }
}
