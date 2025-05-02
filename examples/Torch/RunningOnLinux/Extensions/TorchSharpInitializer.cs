using HarmonyLib;
using System;
using System.Reflection;
using System.Runtime.InteropServices;
using TorchSharp;
using Bonsai;
using System.ComponentModel;
using System.Reactive;
using System.Reactive.Linq;

[Combinator]
[WorkflowElementCategory(ElementCategory.Source)]
[Description("Initializes the TorchSharp library on Linux.")]
public class TorchSharpInitializer
{
    private static bool initialized = false;
    public static void Initialize()
    {
        if (initialized) return;
        var harmony = new Harmony("torchsharp");
        var method = GetTryLoadMethod();
        if (method != null)
        {
            harmony.Patch(method, new HarmonyMethod(typeof(NativeLibrary).GetMethod(nameof(NativeLibrary.Patch))));
        }
        initialized = true;
    }

    private static MethodInfo GetTryLoadMethod()
    {
        var assembly = typeof(torch).Assembly;
        if (assembly == null)
        {
            Console.WriteLine("TorchSharp assembly not found.");
            return null;
        }

        var nativeLibraryType = assembly.GetType("System.NativeLibrary");
        if (nativeLibraryType == null)
        {
            Console.WriteLine("System.NativeLibrary type in TorchSharp assembly not found.");
            return null;
        }

        var method = nativeLibraryType.GetMethod("TryLoad", BindingFlags.Public | BindingFlags.Static, null, new Type[] { typeof(string), typeof(Assembly), typeof(DllImportSearchPath?), typeof(IntPtr).MakeByRefType() }, null);
        if (method == null)
        {
            Console.WriteLine("System.NativeLibrary.TryLoad method not found.");
            return null;
        }

        return method;
    }

    public IObservable<Unit> Process()
    {
        Initialize();
        return Observable.Return(Unit.Default);
    }
}

public static class NativeLibrary
{
    // Declare the dlopen function from libdl
    [DllImport("libdl.so.2")]
    public static extern IntPtr dlopen(string fileName, int flags);

    // Declare the dlclose function from libdl
    [DllImport("libdl.so.2")]
    public static extern int dlclose(IntPtr handle);

    // Declare the dlerror function from libdl to retrieve error messages
    [DllImport("libdl.so.2")]
    public static extern IntPtr dlerror();

    // Constants for dlopen
    private const int RTLD_NOW = 2;
    private const int RTLD_GLOBAL = 0x100;

    public static IntPtr Load(string libraryPath)
    {
        IntPtr handle = dlopen(libraryPath, RTLD_NOW | RTLD_GLOBAL);
        if (handle == IntPtr.Zero)
        {
            IntPtr errorPtr = dlerror();
            string errorMsg = Marshal.PtrToStringAnsi(errorPtr);
            Console.WriteLine($"Failed to load library: {errorMsg}");
            return IntPtr.Zero;
        }

        return handle;
    }
    
    public static bool Patch(string libraryName, Assembly assembly, DllImportSearchPath? searchPath, out IntPtr handle, ref bool __result)
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            libraryName = "lib" + libraryName + ".so";
        }
        handle = Load(libraryName);
        __result = handle != IntPtr.Zero;
        return false;
    }
}