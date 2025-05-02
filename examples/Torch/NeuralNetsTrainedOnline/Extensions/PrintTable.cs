using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using OpenCV.Net;

[Combinator]
[Description("Custom operator to print a table of values with padding.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class PrintTable
{
    [Description("The number of characters to pad the output with.")]
    public int Padding { get; set; }
    
    public IObservable<string> Process(IObservable<Tuple<IplImage, string, string, int>> source)
    {
        return source.Select(value => {
            return (value.Item2 == value.Item3).ToString().PadLeft(Padding, ' ') 
                + value.Item2.PadLeft(Padding, ' ') 
                + value.Item3.PadLeft(Padding, ' ') 
                + value.Item4.ToString().PadLeft(Padding, ' ');
        });
    }
}
