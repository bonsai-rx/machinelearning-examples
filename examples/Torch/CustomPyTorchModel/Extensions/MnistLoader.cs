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
[Description("Custom operator to load the MNIST dataset.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class MnistLoader
{
    [Description("Path to the MNIST dataset folder. The folder should contain the gzipped files for images and labels.")]
    [Editor("Bonsai.Design.FolderNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    public string Path { get; set; }

    [Description("Dataset split to load (Train or Test).")]
    private DatasetSplit _split = DatasetSplit.Test;
    public DatasetSplit Split
    {
        get
        {
            return _split;
        }
        set
        {
            _split = value;
        }
    }

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

            string imagesGZ;
            string labelsGZ;

            if (_split == DatasetSplit.Train)
            {
                imagesGZ = "train-images-idx3-ubyte.gz";
                labelsGZ = "train-labels-idx1-ubyte.gz";
            }
            else
            {
                imagesGZ = "t10k-images-idx3-ubyte.gz";
                labelsGZ = "t10k-labels-idx1-ubyte.gz";
            }

            var imagesGZPath = System.IO.Path.Combine(Path, imagesGZ);
            var labelsGZPath = System.IO.Path.Combine(Path, labelsGZ);

            var fashionMnistData = new FashionMnistData();

            DecompressDataAndRead(imagesGZPath, ReadImagesAndAdd, fashionMnistData.Images);
            DecompressDataAndRead(labelsGZPath, ReadLabelsAndAdd, fashionMnistData.Labels);

            return Observable.Return(fashionMnistData);
        });
    }
}

public class FashionMnistData
{
    private List<IplImage> _images = new List<IplImage>();
    public List<IplImage> Images
    {
        get
        {
            return _images;
        }
    }

    private List<int> _labels = new List<int>();
    public List<int> Labels
    {
        get
        {
            return _labels;
        }
    }
}

public enum DatasetSplit
{
    Train,
    Test
}