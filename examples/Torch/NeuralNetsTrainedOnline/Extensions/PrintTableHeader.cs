using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Source)]
public class PrintTableHeader
{
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
