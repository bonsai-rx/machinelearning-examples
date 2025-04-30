using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;

[Combinator]
[Description("Custom operator to print a table header with padding.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class PrintTableHeader
{
    [Description("The number of characters to pad the output with.")]
    public int Padding { get; set; }
    public IObservable<string> Process()
    {
        return Observable.Return(
            "Correct".PadLeft(Padding, ' ') +
            "Target".PadLeft(Padding, ' ') +
            "Prediction".PadLeft(Padding, ' ') +
            "Observed".PadLeft(Padding, ' ')
        );
    }
}
