using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class SortStringArray
{
    public string Prefix 
    {
        get;
        set;
    }
    public string Suffix 
    {
        get;
        set;
    }
    public IObservable<string[]> Process(IObservable<string[]> source)
    {
        return source.Select(value => {
            Array.Sort(value, (val1, val2) => {
                var val1Int = Convert.ToInt32(val1.Split(new string[] { Prefix, Suffix }, StringSplitOptions.None)[1]);
                var val2Int = Convert.ToInt32(val2.Split(new string[] { Prefix, Suffix }, StringSplitOptions.None)[1]);
                return val1Int - val2Int;
            });
            return value;
        });
    }
}
