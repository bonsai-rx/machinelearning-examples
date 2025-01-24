# Running on Linux

This example demonstrates how to use the Bonsai.ML.Torch package on Linux.

The problem originates from the TorchSharp library. TorchSharp is coded to explicitly throw an error when loading the native libtorch dll on any non-windows machines running .NET fx. See discussion of the issue [here](https://github.com/dotnet/TorchSharp/issues/689).

Until the problem is addressed in the TorchSharp library, there are 2 options that are known to work on linux:

1. This is the more challenging method. Fork the TorchSharp library and change the [lines here](https://github.com/dotnet/TorchSharp/blob/1dd3ae507b0881bfb41c00c185438c7884f6a59f/src/TorchSharp/netstandard.cs#L187). Remove the hard-coded check for `OSPlatform.Windows` and either let the rest of the code handle loading the library or directly load in the `libLibTorchSharp.so` library. Build the nuget package and install in Bonsai. 
2. An easier solution - involves "patching" the load function from the TorchSharp library at runtime. For this, I suggest using the `Lib.Harmony` library. You basically just need to manually install the `Lib.Harmony` library in Bonsai and use the Extensions contained in this example repo to get the `Bonsai.ML.Torch` library working. Once you have the `Lib.Harmony` package, you add a `TorchSharpInitializer` node at the top of your workflow and then can start using the `Bonsai.ML.Torch` library as normal.
