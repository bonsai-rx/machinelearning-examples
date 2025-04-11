using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.IO.Compression;
using System.Reactive.Linq;
using OpenCV.Net;

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Source)]
public class FashionMnistLoader
{
    public string Path { get; set; }

    private void DecompressDataAndRead<T>(string gzPath, Action<BinaryReader, List<T>> readAction, List<T> list)
    {
        using (var fileStream = new FileStream(gzPath, FileMode.Open, FileAccess.Read))
        {
            using (var gzipStream = new GZipStream(fileStream, CompressionMode.Decompress))
            {
                using (var reader = new BinaryReader(gzipStream))
                {
                    readAction(reader, list);
                }
            }
        }
    }

    private byte[] ReadBigEndianBytes(BinaryReader reader, int count)
    {
        byte[] bytes = new byte[count];
        for (int i = count - 1; i >= 0; i--)
            bytes[i] = reader.ReadByte();

        return bytes;
    }

    private unsafe void ReadImagesAndAdd(BinaryReader reader, List<IplImage> images)
    {
        ReadBigEndianBytes(reader, 4);
        var count = BitConverter.ToInt32(ReadBigEndianBytes(reader, 4), 0);
        var height = BitConverter.ToInt32(ReadBigEndianBytes(reader, 4), 0);
        var width = BitConverter.ToInt32(ReadBigEndianBytes(reader, 4), 0);
        var dataBytes = reader.ReadBytes(count * height * width);

        fixed (byte* dataPtr = dataBytes)
        {
            for (var i = 0; i < count; i++)
            {
                var image = new IplImage(new Size(width, height), IplDepth.U8, 1, new IntPtr(dataPtr + i * height * width));
                image = image.Clone();
                images.Add(image);
            }
        }
    }

    private void ReadLabelsAndAdd(BinaryReader reader, List<int> labels)
    {
        ReadBigEndianBytes(reader, 4);
        var count = BitConverter.ToInt32(ReadBigEndianBytes(reader, 4), 0);
        var dataBytes = reader.ReadBytes(count);

        for (var i = 0; i < count; i++)
        {
            labels.Add(Convert.ToInt32(dataBytes[i]));
        }
    }

    public IObservable<FashionMnistData> Process()
    {
        return Observable.Defer(() => {
            var trainingImagesGZPath = System.IO.Path.Combine(Path, "train-images-idx3-ubyte.gz");
            var trainingLabelsGZPath = System.IO.Path.Combine(Path, "train-labels-idx1-ubyte.gz");
            var testImagesGZPath = System.IO.Path.Combine(Path, "t10k-images-idx3-ubyte.gz");
            var testLabelsGZPath = System.IO.Path.Combine(Path, "t10k-labels-idx1-ubyte.gz");

            var fashionMnistData = new FashionMnistData();

            DecompressDataAndRead(trainingImagesGZPath, ReadImagesAndAdd, fashionMnistData.TrainImages);
            DecompressDataAndRead(trainingLabelsGZPath, ReadLabelsAndAdd, fashionMnistData.TrainLabels);
            DecompressDataAndRead(testImagesGZPath, ReadImagesAndAdd, fashionMnistData.TestImages);
            DecompressDataAndRead(testLabelsGZPath, ReadLabelsAndAdd, fashionMnistData.TestLabels);

            return Observable.Return(fashionMnistData);
        });
    }
}

public class FashionMnistData
{
    private List<IplImage> _trainImages = new List<IplImage>();
    public List<IplImage> TrainImages 
    { 
        get
        {
            return _trainImages;
        }
    }

    private List<int> _trainLabels = new List<int>();
    public List<int> TrainLabels
    {
        get
        {
            return _trainLabels;
        }
    }

    private List<IplImage> _testImages = new List<IplImage>();
    public List<IplImage> TestImages
    {
        get
        {
            return _testImages;
        }
    }

    private List<int> _testLabels = new List<int>();
    public List<int> TestLabels
    {
        get
        {
            return _testLabels;
        }
    }
}
